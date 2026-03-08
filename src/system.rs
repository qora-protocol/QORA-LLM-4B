//! System awareness: detect RAM, CPU threads, and smart parameter adjustment.

/// System information detected at startup.
pub struct SystemInfo {
    pub total_ram_mb: u64,
    pub available_ram_mb: u64,
    pub cpu_threads: usize,
}

/// Smart parameter limits based on system capabilities.
pub struct SmartLimits {
    pub max_think_budget: usize,
    pub max_tokens: usize,
    pub default_think_budget: usize,
    pub default_max_tokens: usize,
    pub warning: Option<&'static str>,
}

impl SystemInfo {
    /// Detect system info. Uses wmic on Windows, /proc/meminfo on Linux.
    /// Falls back to conservative defaults if detection fails.
    pub fn detect() -> Self {
        let cpu_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        let (total, available) = detect_ram();

        SystemInfo {
            total_ram_mb: total,
            available_ram_mb: available,
            cpu_threads,
        }
    }

    /// Compute smart limits based on available RAM.
    /// Returns both defaults (for when user didn't specify) and hard caps
    /// (to clamp even explicit user values on weak systems).
    pub fn smart_limits(&self) -> SmartLimits {
        let avail = self.available_ram_mb;
        if avail < 4096 {
            SmartLimits {
                max_think_budget: 256,
                max_tokens: 512,
                default_think_budget: 128,
                default_max_tokens: 256,
                warning: Some("Very low RAM — capping generation to 512 tokens"),
            }
        } else if avail < 8192 {
            SmartLimits {
                max_think_budget: 1024,
                max_tokens: 1024,
                default_think_budget: 256,
                default_max_tokens: 512,
                warning: Some("Low RAM — capping generation to 1024 tokens"),
            }
        } else if avail < 12288 {
            SmartLimits {
                max_think_budget: 2048,
                max_tokens: 2048,
                default_think_budget: 1024,
                default_max_tokens: 1024,
                warning: None,
            }
        } else {
            SmartLimits {
                max_think_budget: 8192,
                max_tokens: 8192,
                default_think_budget: 2048,
                default_max_tokens: 2048,
                warning: None,
            }
        }
    }
}

/// Detect total and available RAM in MB.
fn detect_ram() -> (u64, u64) {
    #[cfg(target_os = "windows")]
    {
        if let Some((total, avail)) = detect_ram_windows() {
            return (total, avail);
        }
    }

    #[cfg(target_os = "linux")]
    {
        if let Some((total, avail)) = detect_ram_linux() {
            return (total, avail);
        }
    }

    #[cfg(target_os = "macos")]
    {
        if let Some((total, avail)) = detect_ram_macos() {
            return (total, avail);
        }
    }

    // Fallback: assume 8 GB total, 4 GB available
    (8192, 4096)
}

#[cfg(target_os = "windows")]
fn detect_ram_windows() -> Option<(u64, u64)> {
    let output = std::process::Command::new("wmic")
        .args(["OS", "get", "TotalVisibleMemorySize,FreePhysicalMemory", "/format:csv"])
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&output.stdout);
    for line in text.lines() {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 3 {
            let free_kb: u64 = parts[1].trim().parse().ok().unwrap_or(0);
            let total_kb: u64 = parts[2].trim().parse().ok().unwrap_or(0);
            if total_kb > 0 {
                return Some((total_kb / 1024, free_kb / 1024));
            }
        }
    }
    None
}

#[cfg(target_os = "linux")]
fn detect_ram_linux() -> Option<(u64, u64)> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    let mut total_kb = 0u64;
    let mut avail_kb = 0u64;
    for line in text.lines() {
        if line.starts_with("MemTotal:") {
            total_kb = parse_meminfo_kb(line);
        } else if line.starts_with("MemAvailable:") {
            avail_kb = parse_meminfo_kb(line);
        }
    }
    if total_kb > 0 {
        Some((total_kb / 1024, avail_kb / 1024))
    } else {
        None
    }
}

#[cfg(target_os = "linux")]
fn parse_meminfo_kb(line: &str) -> u64 {
    line.split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}

#[cfg(target_os = "macos")]
fn detect_ram_macos() -> Option<(u64, u64)> {
    // Total RAM via sysctl hw.memsize (bytes)
    let total_output = std::process::Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()?;
    let total_bytes: u64 = String::from_utf8_lossy(&total_output.stdout)
        .trim()
        .parse()
        .ok()?;
    let total_mb = total_bytes / (1024 * 1024);

    // Available RAM: vm_stat gives pages, multiply by page size
    let vm_output = std::process::Command::new("vm_stat")
        .output()
        .ok()?;
    let vm_text = String::from_utf8_lossy(&vm_output.stdout);

    let page_size: u64 = {
        let ps_output = std::process::Command::new("sysctl")
            .args(["-n", "hw.pagesize"])
            .output()
            .ok()?;
        String::from_utf8_lossy(&ps_output.stdout)
            .trim()
            .parse()
            .unwrap_or(4096)
    };

    let mut free_pages: u64 = 0;
    let mut inactive_pages: u64 = 0;
    let mut speculative_pages: u64 = 0;
    for line in vm_text.lines() {
        if line.starts_with("Pages free:") {
            free_pages = parse_vm_stat_pages(line);
        } else if line.starts_with("Pages inactive:") {
            inactive_pages = parse_vm_stat_pages(line);
        } else if line.starts_with("Pages speculative:") {
            speculative_pages = parse_vm_stat_pages(line);
        }
    }
    let avail_mb = (free_pages + inactive_pages + speculative_pages) * page_size / (1024 * 1024);

    Some((total_mb, avail_mb))
}

#[cfg(target_os = "macos")]
fn parse_vm_stat_pages(line: &str) -> u64 {
    // Format: "Pages free:    12345."
    line.split(':')
        .nth(1)
        .map(|s| s.trim().trim_end_matches('.'))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}
