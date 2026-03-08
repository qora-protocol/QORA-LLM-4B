//! Tokenizer and chat template for QORA-4B.

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

// Special token IDs for QORA-4B
pub const IM_START: u32 = 248045;
pub const IM_END: u32 = 248046;
pub const EOS: u32 = 248044;  // <|endoftext|> (config.json eos_token_id)
pub const ENDOFTEXT: u32 = 248044;
pub const THINK_START: u32 = 248068;
pub const THINK_END: u32 = 248069;
pub const VISION_START: u32 = 248053;
pub const VISION_END: u32 = 248054;
pub const IMAGE_PAD: u32 = 248056;
pub const VIDEO_PAD: u32 = 248057;

pub struct QoraTokenizer {
    inner: tokenizers::Tokenizer,
}

impl QoraTokenizer {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| format!("Failed to load tokenizer: {e}"))?;
        Ok(Self { inner })
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let encoding = self.inner.encode(text, false).expect("Failed to encode");
        encoding.get_ids().to_vec()
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        self.inner.decode(ids, true).unwrap_or_default()
    }

    /// Build a full chat prompt for QORA-4B (text only).
    pub fn format_chat(&self, user_message: &str, think: bool, max_tokens: usize) -> Vec<u32> {
        let today = current_date_string();
        let system_content = build_system_prompt(&today, think, max_tokens);

        let full_text = format!(
            "<|im_start|>system\n{system_content}<|im_end|>\n\
             <|im_start|>user\n{user_message}<|im_end|>\n\
             <|im_start|>assistant\n"
        );

        let mut tokens = self.encode(&full_text);

        if think {
            // Think mode: <think>\n
            tokens.push(THINK_START);
            tokens.extend(self.encode("\n"));
        } else {
            // No-think mode: <think>\n\n</think>\n\n (empty think block)
            tokens.push(THINK_START);
            tokens.extend(self.encode("\n\n"));
            tokens.push(THINK_END);
            tokens.extend(self.encode("\n\n"));
        }

        tokens
    }

    /// Build a chat prompt with image for QORA-4B.
    /// `num_vision_tokens` is the number of merged vision tokens from the encoder.
    /// Format: <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|>×N<|vision_end|>\n{text}<|im_end|>\n<|im_start|>assistant\n
    pub fn format_chat_with_image(&self, user_message: &str, num_vision_tokens: usize, think: bool, max_tokens: usize) -> Vec<u32> {
        let today = current_date_string();
        let system_content = build_system_prompt(&today, think, max_tokens);

        let prefix = format!("<|im_start|>system\n{system_content}<|im_end|>\n<|im_start|>user\n");
        let suffix = format!("\n{user_message}<|im_end|>\n<|im_start|>assistant\n");

        let mut tokens = self.encode(&prefix);

        // Insert vision tokens: <|vision_start|> + N × <|image_pad|> + <|vision_end|>
        tokens.push(VISION_START);
        for _ in 0..num_vision_tokens {
            tokens.push(IMAGE_PAD);
        }
        tokens.push(VISION_END);

        tokens.extend(self.encode(&suffix));

        if think {
            tokens.push(THINK_START);
            tokens.extend(self.encode("\n"));
        } else {
            tokens.push(THINK_START);
            tokens.extend(self.encode("\n\n"));
            tokens.push(THINK_END);
            tokens.extend(self.encode("\n\n"));
        }

        tokens
    }

    /// Build a chat prompt with video for QORA-4B.
    /// `num_vision_tokens` is the total number of merged vision tokens from the video encoder.
    /// Format: <|im_start|>system\n...<|im_end|>\n<|im_start|>user\nVideo 1: <|vision_start|><|video_pad|>×N<|vision_end|>\n{text}<|im_end|>\n<|im_start|>assistant\n
    pub fn format_chat_with_video(&self, user_message: &str, num_vision_tokens: usize, think: bool, max_tokens: usize) -> Vec<u32> {
        let today = current_date_string();
        let system_content = build_system_prompt(&today, think, max_tokens);

        let prefix = format!("<|im_start|>system\n{system_content}<|im_end|>\n<|im_start|>user\nVideo 1: ");
        let suffix = format!("\n{user_message}<|im_end|>\n<|im_start|>assistant\n");

        let mut tokens = self.encode(&prefix);

        // Insert vision tokens: <|vision_start|> + N × <|video_pad|> + <|vision_end|>
        tokens.push(VISION_START);
        for _ in 0..num_vision_tokens {
            tokens.push(VIDEO_PAD);
        }
        tokens.push(VISION_END);

        tokens.extend(self.encode(&suffix));

        if think {
            tokens.push(THINK_START);
            tokens.extend(self.encode("\n"));
        } else {
            tokens.push(THINK_START);
            tokens.extend(self.encode("\n\n"));
            tokens.push(THINK_END);
            tokens.extend(self.encode("\n\n"));
        }

        tokens
    }
}

/// Build the system prompt content with metadata and thinking instructions.
fn build_system_prompt(today: &str, think: bool, max_tokens: usize) -> String {
    let reasoning_mode = if think { "/think" } else { "/no_think" };

    let thinking_instructions = if think {
        "When solving problems, think step by step in the <think> block.\n\
         Then give ONLY the final answer clearly. Do NOT repeat your reasoning after </think>.\n"
    } else {
        ""
    };

    let length_hint = if max_tokens <= 100 {
        "IMPORTANT: Keep your response very brief — 1-2 sentences only.\n"
    } else if max_tokens <= 300 {
        "IMPORTANT: Keep your response concise — a few sentences. Do not use bullet points or lists.\n"
    } else if max_tokens <= 500 {
        "Keep your response brief — a short paragraph. Avoid lengthy lists or breakdowns.\n"
    } else {
        ""
    };

    format!(
        "## Metadata\n\n\
         Knowledge Cutoff Date: June 2025\n\
         Today Date: {today}\n\
         Reasoning Mode: {reasoning_mode}\n\n\
         ## Instructions\n\n\
         You are QORA, a helpful AI assistant. \
         You provide accurate, clear responses.\n\
         {length_hint}\
         {thinking_instructions}"
    )
}

/// Get current date as "DD Month YYYY" string.
fn current_date_string() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let days = (secs / 86400) as i64;

    let z = days + 719468;
    let era = (if z >= 0 { z } else { z - 146096 }) / 146097;
    let doe = (z - era * 146097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };

    let month_name = match m {
        1 => "January", 2 => "February", 3 => "March", 4 => "April",
        5 => "May", 6 => "June", 7 => "July", 8 => "August",
        9 => "September", 10 => "October", 11 => "November", 12 => "December",
        _ => "Unknown",
    };

    format!("{d} {month_name} {y}")
}
