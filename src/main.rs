use std::path::PathBuf;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Parse flags
    let no_think = args.iter().any(|a| a == "--no-think");
    let greedy = args.iter().any(|a| a == "--greedy");
    let show_think = args.iter().any(|a| a == "--show-think");
    #[cfg(any(feature = "gpu", feature = "gpu-metal"))]
    let force_cpu = args.iter().any(|a| a == "--cpu");

    // Track whether user explicitly set these
    let mut max_tokens_explicit = false;
    let mut think_budget_explicit = false;

    // Parse key-value arguments
    let mut prompt = String::from("Hello, how are you?");
    let mut max_tokens: usize = 1024;
    let mut think_budget: usize = 1024;
    let mut image_path: Option<PathBuf> = None;
    let mut video_path: Option<PathBuf> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--prompt" => {
                if i + 1 < args.len() { prompt = args[i + 1].clone(); i += 1; }
            }
            "--max-tokens" => {
                if i + 1 < args.len() {
                    max_tokens = args[i + 1].parse().unwrap_or(1024);
                    max_tokens_explicit = true;
                    i += 1;
                }
            }
            "--think-budget" => {
                if i + 1 < args.len() {
                    think_budget = args[i + 1].parse().unwrap_or(1024);
                    think_budget_explicit = true;
                    i += 1;
                }
            }
            "--image" => {
                if i + 1 < args.len() { image_path = Some(PathBuf::from(&args[i + 1])); i += 1; }
            }
            "--video" => {
                if i + 1 < args.len() { video_path = Some(PathBuf::from(&args[i + 1])); i += 1; }
            }
            _ => {}
        }
        i += 1;
    }

    // System awareness
    let sys = qor4b::system::SystemInfo::detect();
    let limits = sys.smart_limits();
    eprintln!("QORA-4B - Pure Rust Multimodal Inference Engine");
    eprintln!("System: {} MB RAM ({} MB free), {} threads",
        sys.total_ram_mb, sys.available_ram_mb, sys.cpu_threads);

    // Set defaults if user didn't specify
    if !think_budget_explicit { think_budget = limits.default_think_budget; }
    if !max_tokens_explicit { max_tokens = limits.default_max_tokens; }

    // Hard cap: even explicit values get clamped on weak systems
    if think_budget > limits.max_think_budget {
        eprintln!("System cap: think-budget {} → {}", think_budget, limits.max_think_budget);
        think_budget = limits.max_think_budget;
    }
    if max_tokens > limits.max_tokens {
        eprintln!("System cap: max-tokens {} → {}", max_tokens, limits.max_tokens);
        max_tokens = limits.max_tokens;
    }

    if let Some(msg) = limits.warning {
        eprintln!("WARNING: {msg}");
    }

    // Auto-adjust: ensure max_tokens > think_budget so model has room to answer
    if !no_think && max_tokens <= think_budget {
        max_tokens = think_budget + 1024;
        eprintln!("Auto-adjusted max-tokens to {} (think-budget + 1024)", max_tokens);
    }

    let mode_str = if no_think { "no-think" } else { "think" };
    eprintln!("Prompt: {prompt}");
    if let Some(ref img) = image_path {
        eprintln!("Image: {}", img.display());
    }
    if let Some(ref vid) = video_path {
        eprintln!("Video: {}", vid.display());
    }
    eprintln!("Mode: {mode_str}");

    // Load model from same directory as the executable
    let exe_dir = std::env::current_exe()
        .expect("Cannot determine executable path")
        .parent().unwrap().to_path_buf();
    let model_path = exe_dir.join("model.qor4b");
    eprintln!("Loading model from {}...", model_path.display());
    let t0 = Instant::now();
    let weights = qor4b::save::load_model(&model_path)
        .expect("Failed to load model");
    let mem_mb = weights.memory_bytes() / (1024 * 1024);
    eprintln!("Model loaded in {:.1?} ({} format, {mem_mb} MB)", t0.elapsed(), weights.format_name);

    // Load tokenizer from same directory as the executable
    let tokenizer_path = exe_dir.join("tokenizer.json");
    let tokenizer = qor4b::tokenizer::QoraTokenizer::from_file(&tokenizer_path)
        .expect("Failed to load tokenizer");

    let temperature = if greedy { 0.0 } else if no_think { 0.7 } else { 1.0 };

    let params = qor4b::generate::GenerateParams {
        max_new_tokens: max_tokens,
        eos_token_id: weights.config.eos_token_id,
        temperature,
        top_p: 0.95,
        top_k: 20,
        think: !no_think,
        show_think,
        repetition_penalty: 1.0,
        presence_penalty: 1.5,
        think_budget,
    };

    // Try GPU for text-only mode
    #[cfg(any(feature = "gpu", feature = "gpu-metal"))]
    if !force_cpu && image_path.is_none() && video_path.is_none() {
        eprintln!("Attempting GPU inference...");
        match qor4b::gpu_inference::generate_gpu(&weights, &tokenizer, &prompt, &params) {
            Ok(()) => return,
            Err(e) => eprintln!("GPU not available ({}), falling back to CPU", e),
        }
    }

    if let Some(ref vid_path) = video_path {
        // Video + text mode
        let vision = weights.vision.as_ref()
            .expect("Model has no vision weights");

        eprintln!("Loading video frames...");
        let (pixels, num_frames, height, width) = qor4b::vision::load_video_frames(vid_path)
            .expect("Failed to load video frames");

        eprintln!("Running vision encoder (video)...");
        let t_vis = Instant::now();
        let (vision_embeds, num_vision_tokens) = vision.encode_video(&pixels, num_frames, height, width);
        eprintln!("Vision encoder: {} tokens in {:.1?}", num_vision_tokens, t_vis.elapsed());

        qor4b::generate::generate_with_video(
            &weights, &tokenizer, &prompt,
            &vision_embeds, num_vision_tokens,
            &params,
        );
    } else if let Some(ref img_path) = image_path {
        // Image + text mode
        let vision = weights.vision.as_ref()
            .expect("Model has no vision weights");

        eprintln!("Loading image...");
        let (pixels, height, width) = qor4b::vision::load_image(img_path)
            .expect("Failed to load image");

        eprintln!("Running vision encoder...");
        let t_vis = Instant::now();
        let (vision_embeds, num_vision_tokens) = vision.encode_image(&pixels, height, width);
        eprintln!("Vision encoder: {} tokens in {:.1?}", num_vision_tokens, t_vis.elapsed());

        qor4b::generate::generate_with_image(
            &weights, &tokenizer, &prompt,
            &vision_embeds, num_vision_tokens,
            &params,
        );
    } else {
        // Text-only mode
        qor4b::generate::generate(&weights, &tokenizer, &prompt, &params);
    }
}
