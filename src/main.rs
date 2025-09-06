use anyhow::{anyhow, Context, Result};
use dotenvy::dotenv;
use base64::Engine;
use glob::glob;
use reqwest::header::{CONTENT_TYPE, RETRY_AFTER};
use reqwest::StatusCode;
use regex::Regex;
use chrono::Local;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use tokio;
use std::time::Instant;
use tokio::time::{sleep, Duration};

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();

    let api_key = env::var("GEMINI_API_KEY")
        .context("GEMINI_API_KEY not found in env; please set it in .env")?;

    let client = GeminiClient::new(api_key)?;

    // Ensure audio output directory exists
    let audio_dir = Path::new("audio");
    if !audio_dir.exists() {
        fs::create_dir_all(audio_dir).context("failed to create audio/ directory")?;
    }

    // Build list of markdown files to process
    let mut args = env::args().skip(1);
    let paths: Vec<PathBuf> = if let Some(one_path) = args.next() {
        vec![PathBuf::from(one_path)]
    } else {
        let mut v = Vec::new();
        for entry in glob("book/src/*.md").context("glob pattern failed")? {
            v.push(entry?);
        }
        v
    };

    println!("Found {} markdown file(s) to process.", paths.len());
    for (i, path) in paths.iter().enumerate() {
        println!("[{} / {}] Starting {}", i + 1, paths.len(), path.display());
        let t0 = Instant::now();
        process_markdown_file(&client, path, audio_dir).await?;
        println!(
            "[{} / {}] Finished {} in {:?}",
            i + 1,
            paths.len(),
            path.display(),
            t0.elapsed()
        );
    }

    Ok(())
}

async fn process_markdown_file(client: &GeminiClient, path: &Path, audio_dir: &Path) -> Result<()> {
    let original = fs::read_to_string(path)
        .with_context(|| format!("failed to read file {}", path.display()))?;
    println!(
        "Reading {}: {} characters",
        path.display(),
        original.chars().count()
    );
    let (transformed, summarized_blocks) = replace_code_blocks_with_summaries(client, &original).await?;
    println!(
        "Summarized {} code block(s) in {}",
        summarized_blocks,
        path.display()
    );

    // Split content into <= 3000-char chunks on paragraph boundaries
    let tts_text = sanitize_markdown_for_tts(&transformed);
    println!(
        "Sanitized text for TTS (links/headers/lists/html/code fences): {} -> {} chars",
        transformed.chars().count(),
        tts_text.chars().count()
    );
    let chunks = split_into_chunks_by_paragraph(&tts_text, 3000);
    println!(
        "Chunked content into {} piece(s) (<=3000 chars each)",
        chunks.len()
    );

    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| anyhow!("invalid file name: {}", path.display()))?;

    let mut parts: Vec<(Vec<u8>, String)> = Vec::new();
    for (i, chunk) in chunks.iter().enumerate() {
    println!(
        "{} | TTS part {:02}: {} chars...",
        now_ts(),
        i + 1,
        chunk.chars().count()
    );
        let t0 = Instant::now();
        let (audio_bytes, mime_type) = client
            .tts_generate(chunk)
            .await
            .with_context(|| format!("TTS generation failed for {} (part {})", path.display(), i + 1))?;
        println!(
            "{} | TTS part {:02}: mime={}, {} bytes, took {:?}",
            now_ts(),
            i + 1,
            mime_type,
            audio_bytes.len(),
            t0.elapsed()
        );
        parts.push((audio_bytes, mime_type));
    }

    // Determine a single mime type (all parts should match); fall back to first.
    let mime = parts
        .get(0)
        .map(|(_, m)| m.clone())
        .unwrap_or_else(|| "application/octet-stream".to_string());
    let ext = guess_audio_extension(&mime);

    let merged: Vec<u8> = if parts.len() == 1 {
        parts.remove(0).0
    } else if mime.contains("mpeg") || mime.contains("mp3") {
        println!("Merging {} MP3 parts via concatenation", parts.len());
        merge_mp3(&parts.iter().map(|(b, _)| b.as_slice()).collect::<Vec<_>>())
    } else if mime.contains("wav") || mime.contains("x-wav") || mime.contains("pcm") {
        println!("Merging {} WAV parts with header rewrite", parts.len());
        match try_merge_wav(&parts.iter().map(|(b, _)| b.as_slice()).collect::<Vec<_>>()) {
            Ok(bytes) => bytes,
            Err(e) => {
                eprintln!("warn: WAV merge failed ({}). Falling back to naive concat.", e);
                merge_concat(&parts.iter().map(|(b, _)| b.as_slice()).collect::<Vec<_>>())
            }
        }
    } else {
        eprintln!(
            "warn: unsupported mime '{}' for merging; concatenating raw bytes (may not play correctly)",
            mime
        );
        merge_concat(&parts.iter().map(|(b, _)| b.as_slice()).collect::<Vec<_>>())
    };

    let out_path = audio_dir.join(format!("{}{}", stem, ext));
    fs::write(&out_path, &merged)
        .with_context(|| format!("failed to write audio file {}", out_path.display()))?;

    println!(
        "Processed {} => {} ({} bytes from {} chunks)",
        path.display(),
        out_path.display(),
        merged.len(),
        parts.len()
    );

    Ok(())
}

fn guess_audio_extension(mime: &str) -> &'static str {
    match mime {
        m if m.contains("mpeg") || m.contains("mp3") => ".mp3",
        m if m.contains("wav") || m.contains("x-wav") || m.contains("pcm") || m.contains("linear16") => ".wav",
        m if m.contains("ogg") => ".ogg",
        m if m.contains("flac") => ".flac",
        _ => ".bin",
    }
}

async fn replace_code_blocks_with_summaries(client: &GeminiClient, input: &str) -> Result<(String, usize)> {
    let mut out = String::with_capacity(input.len());
    let mut lines = input.lines();
    let mut in_block = false;
    let mut code_acc: Vec<String> = Vec::new();
    let mut count_blocks = 0usize;

    while let Some(line) = lines.next() {
        if !in_block {
            if is_fence_open(line) {
                in_block = true;
                out.push_str(line);
                out.push('\n');
                code_acc.clear();
            } else {
                out.push_str(line);
                out.push('\n');
            }
        } else {
            if is_fence_close(line) {
                // Summarize accumulated code
                let code_text = code_acc.join("\n");
                count_blocks += 1;
                println!(
                    "Summarizing code block #{} ({} chars)",
                    count_blocks,
                    code_text.chars().count()
                );
                let t0 = Instant::now();
                let summary = client
                    .summarize_code_block(&code_text)
                    .await
                    .unwrap_or_else(|e| format!("[summary failed: {e}]"));
                println!(
                    "Summary #{} done ({} chars) in {:?}",
                    count_blocks,
                    summary.chars().count(),
                    t0.elapsed()
                );

                // Replace the code block content with the summary text
                out.push_str(&summary);
                out.push('\n');

                // Close fence as-is
                out.push_str(line);
                out.push('\n');

                // Reset state
                in_block = false;
                code_acc.clear();
            } else {
                code_acc.push(line.to_string());
            }
        }
    }

    // Handle EOF inside a fence (unlikely). If so, append what we have.
    if in_block {
        let code_text = code_acc.join("\n");
        count_blocks += 1;
        println!(
            "Summarizing code block #{} at EOF ({} chars)",
            count_blocks,
            code_text.chars().count()
        );
        let t0 = Instant::now();
        let summary = client
            .summarize_code_block(&code_text)
            .await
            .unwrap_or_else(|e| format!("[summary failed: {e}]"));
        println!(
            "Summary #{} done ({} chars) in {:?}",
            count_blocks,
            summary.chars().count(),
            t0.elapsed()
        );
        out.push_str(&summary);
        out.push('\n');
    }

    Ok((out, count_blocks))
}

fn is_fence_open(line: &str) -> bool {
    let trimmed = line.trim_start();
    trimmed.starts_with("```")
}

fn is_fence_close(line: &str) -> bool {
    let trimmed = line.trim_start();
    trimmed.starts_with("```")
}

fn split_into_chunks_by_paragraph(input: &str, max_chars: usize) -> Vec<String> {
    if input.chars().count() <= max_chars {
        return vec![input.to_string()];
    }

    let mut chunks = Vec::new();
    let mut start = 0usize;
    let chars: Vec<char> = input.chars().collect();
    let total = chars.len();

    while start < total {
        let remaining = total - start;
        let take = remaining.min(max_chars);
        let mut end = start + take;

        if end < total {
            // try to find the last paragraph boundary (two or more newlines)
            let mut last_para_break: Option<usize> = None;
            let mut i = start;
            while i < end {
                if chars[i] == '\n' {
                    let mut j = i + 1;
                    let mut nl_count = 1;
                    while j < end && chars[j] == '\n' {
                        nl_count += 1;
                        j += 1;
                    }
                    if nl_count >= 1 { // one blank line separates paragraphs in Markdown
                        last_para_break = Some(i);
                    }
                    i = j;
                    continue;
                }
                i += 1;
            }

            if let Some(bp) = last_para_break {
                end = bp + 1; // include the newline
            } else {
                // fallback to last sentence end before limit
                let mut last_sentence_end: Option<usize> = None;
                for k in start..end {
                    if matches!(chars[k], '.' | '!' | '?') {
                        last_sentence_end = Some(k);
                    }
                }
                if let Some(se) = last_sentence_end { end = se + 1; }
            }
        }

        let slice: String = chars[start..end].iter().collect();
        chunks.push(slice);
        start = end;
    }

    chunks
}

fn remove_links_for_tts(input: &str) -> String {
    // 1) Drop reference-style link definitions like: [id]: url "title"
    let mut filtered_lines = Vec::new();
    for line in input.lines() {
        let trimmed = line.trim_start();
        // simple heuristic: starts with '[' and contains ']: '
        if trimmed.starts_with('[') && trimmed.contains("]: ") {
            continue; // skip definition line
        }
        filtered_lines.push(line);
    }
    let without_defs = filtered_lines.join("\n");

    // 2) Replace inline links [text](url) -> text
    let re_inline = Regex::new(r"\[([^\]]+)\]\([^\)]+\)").unwrap();
    let tmp = re_inline.replace_all(&without_defs, "$1").into_owned();

    // 3) Replace reference links [text][id] -> text
    let re_ref = Regex::new(r"\[([^\]]+)\]\[[^\]]*\]").unwrap();
    let tmp = re_ref.replace_all(&tmp, "$1").into_owned();

    // 4) Remove autolinks <http://...>
    let re_auto = Regex::new(r"<https?://[^>]+>").unwrap();
    let tmp = re_auto.replace_all(&tmp, "").into_owned();

    // 5) Remove bare urls https://...
    let re_bare = Regex::new(r"https?://\S+").unwrap();
    let tmp = re_bare.replace_all(&tmp, "").into_owned();

    tmp
}

fn sanitize_markdown_for_tts(input: &str) -> String {
    // First, remove links
    let mut text = remove_links_for_tts(input);

    // Drop lines starting with <Listing or </Listing
    let mut lines = Vec::new();
    for line in text.lines() {
        let t = line.trim_start();
        if t.starts_with("<Listing") || t.starts_with("</Listing") {
            continue;
        }
        // Drop code fence lines (``` ...)
        if t.starts_with("```") { continue; }
        lines.push(line);
    }
    text = lines.join("\n");

    // Remove inline HTML tags and comments
    let re_comment = Regex::new(r"(?s)<!--.*?-->").unwrap();
    text = re_comment.replace_all(&text, "").into_owned();
    let re_tags = Regex::new(r"</?[^>]+>").unwrap();
    text = re_tags.replace_all(&text, "").into_owned();

    // Remove backticks (inline code markers)
    text = text.replace('`', "");

    // Strip heading #'s, blockquote '>'s, and list markers on each line
    let re_heading = Regex::new(r"^\s*#{1,6}\s*").unwrap();
    let re_blockquote = Regex::new(r"^\s*>+\s*").unwrap();
    let re_bullet = Regex::new(r"^\s*[-*+]\s+").unwrap();
    let re_numbered = Regex::new(r"^\s*\d+[\.)]\s+").unwrap();

    let mut out_lines = Vec::new();
    for line in text.lines() {
        let mut l = line.to_string();
        l = re_heading.replace(&l, "").into_owned();
        l = re_blockquote.replace(&l, "").into_owned();
        l = re_bullet.replace(&l, "").into_owned();
        l = re_numbered.replace(&l, "").into_owned();
        out_lines.push(l);
    }
    let mut joined = out_lines.join("\n");

    // Replace any 'scr/' with 'source/' as requested
    joined = joined.replace("scr/", "source/");

    // Collapse 3+ newlines into 2 to avoid long silent gaps
    let re_multi_blank = Regex::new(r"\n{3,}").unwrap();
    joined = re_multi_blank.replace_all(&joined, "\n\n").into_owned();

    // Trim leading/trailing whitespace
    joined.trim().to_string()
}

fn now_ts() -> String {
    let now = Local::now();
    now.format("%Y-%m-%d %H:%M:%S%.3f").to_string()
}

fn merge_mp3(parts: &[&[u8]]) -> Vec<u8> {
    // Simple byte concatenation; most players handle back-to-back MP3 frames.
    merge_concat(parts)
}

fn merge_concat(parts: &[&[u8]]) -> Vec<u8> {
    let total: usize = parts.iter().map(|p| p.len()).sum();
    let mut out = Vec::with_capacity(total);
    for p in parts {
        out.extend_from_slice(p);
    }
    out
}

fn try_merge_wav(parts: &[&[u8]]) -> Result<Vec<u8>> {
    // Parse each WAV, validate same format, and concatenate data chunks; emit new header
    if parts.is_empty() {
        return Ok(Vec::new());
    }

    let mut data_blobs: Vec<&[u8]> = Vec::with_capacity(parts.len());
    let mut total_data_len: usize = 0;

    let (fmt, fmt_size) = parse_wav_fmt(parts[0])?;
    let first_data = parse_wav_data(parts[0])?;
    data_blobs.push(first_data);
    total_data_len += first_data.len();

    for wav in &parts[1..] {
        let (fmt_n, _fmt_size_n) = parse_wav_fmt(wav)?;
        if fmt != fmt_n {
            return Err(anyhow!("WAV format mismatch across chunks"));
        }
        let d = parse_wav_data(wav)?;
        data_blobs.push(d);
        total_data_len += d.len();
    }

    let mut out = Vec::with_capacity(44 + total_data_len);
    write_wav_header(&mut out, &fmt, fmt_size, total_data_len)?;
    for blob in data_blobs {
        out.extend_from_slice(blob);
    }
    Ok(out)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct WavFmt {
    audio_format: u16,   // 1 = PCM, 3 = IEEE float
    num_channels: u16,
    sample_rate: u32,
    byte_rate: u32,
    block_align: u16,
    bits_per_sample: u16,
}

fn parse_wav_fmt(bytes: &[u8]) -> Result<(WavFmt, u32)> {
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(anyhow!("invalid WAV header"));
    }
    let mut off = 12usize;
    let mut fmt: Option<(WavFmt, u32)> = None;
    while off + 8 <= bytes.len() {
        let id = &bytes[off..off + 4];
        let sz = u32::from_le_bytes(bytes[off + 4..off + 8].try_into().unwrap());
        let chunk_data_start = off + 8;
        let chunk_data_end = chunk_data_start + sz as usize;
        if chunk_data_end > bytes.len() { break; }
        if id == b"fmt " {
            if sz < 16 { return Err(anyhow!("fmt chunk too small")); }
            let audio_format = u16::from_le_bytes(bytes[chunk_data_start..chunk_data_start + 2].try_into().unwrap());
            let num_channels = u16::from_le_bytes(bytes[chunk_data_start + 2..chunk_data_start + 4].try_into().unwrap());
            let sample_rate = u32::from_le_bytes(bytes[chunk_data_start + 4..chunk_data_start + 8].try_into().unwrap());
            let byte_rate = u32::from_le_bytes(bytes[chunk_data_start + 8..chunk_data_start + 12].try_into().unwrap());
            let block_align = u16::from_le_bytes(bytes[chunk_data_start + 12..chunk_data_start + 14].try_into().unwrap());
            let bits_per_sample = u16::from_le_bytes(bytes[chunk_data_start + 14..chunk_data_start + 16].try_into().unwrap());
            fmt = Some((
                WavFmt { audio_format, num_channels, sample_rate, byte_rate, block_align, bits_per_sample },
                sz,
            ));
            break;
        }
        off = chunk_data_end + (sz as usize % 2); // chunks are word-aligned
    }
    fmt.ok_or_else(|| anyhow!("fmt chunk not found"))
}

fn parse_wav_data(bytes: &[u8]) -> Result<&[u8]> {
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(anyhow!("invalid WAV header"));
    }
    let mut off = 12usize;
    while off + 8 <= bytes.len() {
        let id = &bytes[off..off + 4];
        let sz = u32::from_le_bytes(bytes[off + 4..off + 8].try_into().unwrap());
        let chunk_data_start = off + 8;
        let chunk_data_end = chunk_data_start + sz as usize;
        if chunk_data_end > bytes.len() { break; }
        if id == b"data" {
            return Ok(&bytes[chunk_data_start..chunk_data_end]);
        }
        off = chunk_data_end + (sz as usize % 2);
    }
    Err(anyhow!("data chunk not found"))
}

fn write_wav_header(out: &mut Vec<u8>, fmt: &WavFmt, fmt_size: u32, data_len: usize) -> Result<()> {
    let fmt_size = if fmt_size < 16 { 16 } else { fmt_size };
    let riff_chunk_size: u32 = 4 + (8 + fmt_size) + (8 + (data_len as u32));

    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&riff_chunk_size.to_le_bytes());
    out.extend_from_slice(b"WAVE");

    // fmt chunk
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&fmt_size.to_le_bytes());
    out.extend_from_slice(&fmt.audio_format.to_le_bytes());
    out.extend_from_slice(&fmt.num_channels.to_le_bytes());
    out.extend_from_slice(&fmt.sample_rate.to_le_bytes());
    out.extend_from_slice(&fmt.byte_rate.to_le_bytes());
    out.extend_from_slice(&fmt.block_align.to_le_bytes());
    out.extend_from_slice(&fmt.bits_per_sample.to_le_bytes());
    if fmt_size > 16 {
        // pad extra fmt bytes with zeros
        out.resize(out.len() + (fmt_size as usize - 16), 0);
    }

    // data chunk
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data_len as u32).to_le_bytes());
    Ok(())
}

struct GeminiClient {
    http: reqwest::Client,
    api_key: String,
    base_url: String,
}

impl GeminiClient {
    fn new(api_key: String) -> Result<Self> {
        let http = reqwest::Client::builder()
            .user_agent("rust-the-audio-book/0.1")
            .build()?;
        Ok(Self {
            http,
            api_key,
            base_url: "https://generativelanguage.googleapis.com/v1beta".to_string(),
        })
    }

    async fn summarize_code_block(&self, code: &str) -> Result<String> {
        let prompt = format!(
            "You are helping write an audio book. Summarize the following code block succinctly (2-4 sentences) for listeners.\n\
             Focus on what it does and why it matters.\n\
             Avoid code, jargon, and backticks.\n\
             Keep it clear and engaging.\n\
\nCode block:\n{code}"
        );

        // Using non-stream generateContent for simpler parsing
        let url = format!(
            "{}/models/{}:{}?key={}",
            self.base_url, "gemini-2.5-flash", "generateContent", self.api_key
        );

        let body = serde_json::json!({
            "contents": [
                { "role": "user", "parts": [ { "text": prompt } ] }
            ],
            "generationConfig": {
                "thinkingConfig": { "thinkingBudget": -1 }
            }
        });

        let parsed: serde_json::Value = self.post_json_with_retries(&url, &body).await?;
        if let Some(text) = extract_first_text(&parsed) {
            return Ok(text.to_string());
        }
        Err(anyhow!("no text returned from summary response: {}", parsed))
    }

    async fn tts_generate(&self, input_text: &str) -> Result<(Vec<u8>, String)> {
        // Prefer non-stream generateContent to simplify, using audio response modality
        let url = format!(
            "{}/models/{}:{}?key={}",
            self.base_url, "gemini-2.5-pro-preview-tts", "generateContent", self.api_key
        );

        // Note: Following the user's example for config field names where possible
        let body = serde_json::json!({
            "contents": [
                { "role": "user", "parts": [ { "text": input_text } ] }
            ],
            "generationConfig": {
                "responseModalities": ["audio"],
                "temperature": 1,
                "speech_config": {
                    "voice_config": {
                        "prebuilt_voice_config": { "voice_name": "Zephyr" }
                    }
                }
            }
        });

        let json_val: serde_json::Value = self.post_json_with_retries(&url, &body).await?;
        if let Some((data_b64, mime)) = extract_audio_inline_data(&json_val) {
            let raw = base64::engine::general_purpose::STANDARD
                .decode(data_b64)
                .context("failed to decode base64 audio")?;

            // If Gemini returns raw PCM/LINEAR16 without a container, wrap as WAV for playability
            if is_raw_linear_pcm(&mime) {
                let sr = parse_sample_rate(&mime).unwrap_or(24000);
                let wav = wrap_pcm_to_wav(&raw, sr, 1, 16)?;
                return Ok((wav, "audio/wav".to_string()));
            }

            return Ok((raw, mime.to_string()));
        }
        Err(anyhow!(
            "TTS response parsed but no audio inline data found: {}",
            json_val
        ))
    }

    async fn post_json_with_retries(
        &self,
        url: &str,
        body: &serde_json::Value,
    ) -> Result<serde_json::Value> {
        let max_retries = 6;
        let mut attempt = 0;
        loop {
            let resp = self
                .http
                .post(url)
                .header(CONTENT_TYPE, "application/json")
                .json(body)
                .send()
                .await;

            match resp {
                Ok(r) if r.status().is_success() => {
                    let json_val: serde_json::Value = r.json().await?;
                    return Ok(json_val);
                }
                Ok(r) => {
                    let status = r.status();
                    let headers = r.headers().clone();
                    let text = r.text().await.unwrap_or_default();
                    if should_retry(status) && attempt < max_retries {
                        let wait = compute_backoff(attempt, headers.get(RETRY_AFTER));
                        eprintln!(
                            "warn: request to {} failed with {}. retrying in {:?} (attempt {}/{})",
                            url, status, wait, attempt + 1, max_retries
                        );
                        sleep(wait).await;
                        attempt += 1;
                        continue;
                    } else {
                        return Err(anyhow!(
                            "request failed: {} {} — body: {}",
                            status.as_u16(),
                            status.canonical_reason().unwrap_or(""),
                            text
                        ));
                    }
                }
                Err(e) => {
                    if attempt < max_retries {
                        let wait = compute_backoff(attempt, None);
                        eprintln!(
                            "warn: network error: {}. retrying in {:?} (attempt {}/{})",
                            e, wait, attempt + 1, max_retries
                        );
                        sleep(wait).await;
                        attempt += 1;
                        continue;
                    } else {
                        return Err(anyhow!("request network error after retries: {}", e));
                    }
                }
            }
        }
    }
}

fn extract_first_text(v: &serde_json::Value) -> Option<&str> {
    let candidates = v.get("candidates")?.as_array()?;
    for c in candidates {
        let content = c.get("content")?;
        let parts = content.get("parts")?.as_array()?;
        for p in parts {
            if let Some(t) = p.get("text").and_then(|t| t.as_str()) {
                if !t.is_empty() {
                    return Some(t);
                }
            }
        }
    }
    None
}

fn extract_audio_inline_data(v: &serde_json::Value) -> Option<(&str, &str)> {
    let candidates = v.get("candidates")?.as_array()?;
    for c in candidates {
        let content = c.get("content")?;
        let parts = content.get("parts")?.as_array()?;
        for p in parts {
            if let Some(inline) = p.get("inlineData") {
                let data = inline.get("data")?.as_str()?;
                let mime = inline
                    .get("mimeType")
                    .and_then(|m| m.as_str())
                    .unwrap_or("application/octet-stream");
                return Some((data, mime));
            }
            if let Some(inline) = p.get("inline_data") {
                let data = inline.get("data")?.as_str()?;
                let mime = inline
                    .get("mime_type")
                    .and_then(|m| m.as_str())
                    .unwrap_or("application/octet-stream");
                return Some((data, mime));
            }
        }
    }
    None
}

fn is_raw_linear_pcm(mime: &str) -> bool {
    let m = mime.to_ascii_lowercase();
    (m.contains("linear16") || m.contains("pcm")) && !m.contains("wav")
}

fn parse_sample_rate(mime: &str) -> Option<u32> {
    // examples: "audio/pcm;rate=24000" or "audio/linear16; sample_rate=16000"
    let lower = mime.to_ascii_lowercase();
    for key in ["rate=", "samplerate=", "sample_rate="] {
        if let Some(pos) = lower.find(key) {
            let tail = &lower[pos + key.len()..];
            let mut num = String::new();
            for ch in tail.chars() {
                if ch.is_ascii_digit() { num.push(ch); } else { break; }
            }
            if let Ok(v) = num.parse::<u32>() { return Some(v); }
        }
    }
    None
}

fn wrap_pcm_to_wav(pcm: &[u8], sample_rate: u32, channels: u16, bits_per_sample: u16) -> Result<Vec<u8>> {
    let block_align = channels * (bits_per_sample / 8);
    let byte_rate = sample_rate * block_align as u32;
    let fmt = WavFmt {
        audio_format: 1, // PCM
        num_channels: channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
    };
    let mut out = Vec::with_capacity(44 + pcm.len());
    write_wav_header(&mut out, &fmt, 16, pcm.len())?;
    out.extend_from_slice(pcm);
    Ok(out)
}

fn should_retry(status: StatusCode) -> bool {
    matches!(
        status,
        StatusCode::TOO_MANY_REQUESTS
            | StatusCode::REQUEST_TIMEOUT
            | StatusCode::INTERNAL_SERVER_ERROR
            | StatusCode::BAD_GATEWAY
            | StatusCode::SERVICE_UNAVAILABLE
            | StatusCode::GATEWAY_TIMEOUT
    )
}

fn compute_backoff(attempt: usize, retry_after_hdr: Option<&reqwest::header::HeaderValue>) -> Duration {
    if let Some(hv) = retry_after_hdr {
        if let Ok(s) = hv.to_str() {
            if let Ok(secs) = s.trim().parse::<u64>() {
                return Duration::from_secs(secs.max(1));
            }
        }
    }
    // Exponential backoff with basic jitter
    let base_secs = 1u64.checked_shl(attempt as u32).unwrap_or(u64::MAX).min(60);
    let jitter_ms = ((attempt as u64 + 1) * 137) % 500; // 0..500ms pseudo-jitter
    Duration::from_secs(base_secs) + Duration::from_millis(jitter_ms)
}
