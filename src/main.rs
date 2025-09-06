use anyhow::{Context, Result, anyhow};
use dotenvy::dotenv;
use glob::glob;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use rust_the_audio_book::audio::{guess_audio_extension, merge_concat, merge_mp3, try_merge_wav};
use rust_the_audio_book::markdown::{
    expand_includes, replace_code_blocks_with_summaries, sanitize_markdown_for_tts,
    split_into_chunks_by_paragraph,
};
use rust_the_audio_book::tts::{AVAILABLE_VOICES, GeminiClient};
use rust_the_audio_book::util::now_ts;

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();

    let api_key = env::var("GEMINI_API_KEY")
        .context("GEMINI_API_KEY not found in env; please set it in .env")?;

    let client = GeminiClient::new(api_key)?;

    let mut args = env::args();
    let program = args
        .next()
        .unwrap_or_else(|| "rust-the-audio-book".to_string());

    let mut voice_name: String = "Zephyr".to_string();
    let mut file_args: Vec<PathBuf> = Vec::new();

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" | "help" => {
                print_help(&program);
                std::process::exit(0);
            }
            "--list-voices" => {
                print_voices();
                std::process::exit(0);
            }
            "-v" | "--voice" => {
                let v = args.next().unwrap_or_else(|| {
                    eprintln!("error: --voice requires a value");
                    print_help(&program);
                    std::process::exit(2);
                });
                voice_name = v;
            }
            other => {
                file_args.push(PathBuf::from(other));
            }
        }
    }

    // Ensure audio output directory exists
    let audio_dir = Path::new("audio");
    if !audio_dir.exists() {
        fs::create_dir_all(audio_dir).context("failed to create audio/ directory")?;
    }

    // Build list of markdown files to process
    let paths: Vec<PathBuf> = if !file_args.is_empty() {
        file_args
    } else {
        let mut v = Vec::new();
        for entry in glob("book/src/*.md").context("glob pattern failed")? {
            v.push(entry?);
        }
        v
    };

    if !AVAILABLE_VOICES.iter().any(|(n, _)| n == &voice_name) {
        eprintln!(
            "warn: voice '{}' not in known list; proceeding anyway",
            voice_name
        );
    }

    println!(
        "Found {} markdown file(s) to process. Using voice: {}",
        paths.len(),
        voice_name
    );
    for (i, path) in paths.iter().enumerate() {
        println!("[{} / {}] Starting {}", i + 1, paths.len(), path.display());
        let t0 = Instant::now();
        process_markdown_file(&client, path, audio_dir, &voice_name).await?;
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

async fn process_markdown_file(
    client: &GeminiClient,
    path: &Path,
    audio_dir: &Path,
    voice_name: &str,
) -> Result<()> {
    let original = fs::read_to_string(path)
        .with_context(|| format!("failed to read file {}", path.display()))?;
    println!(
        "Reading {}: {} characters",
        path.display(),
        original.chars().count()
    );
    // Expand any mdBook-style includes before code summarization
    let expanded = expand_includes(path, &original)?;

    let (transformed, summarized_blocks) =
        replace_code_blocks_with_summaries(client, &expanded).await?;
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
            "{} | TTS part {:02}/{:02}: {} chars...",
            now_ts(),
            i + 1,
            chunks.len() - 1,
            chunk.chars().count()
        );
        let t0 = Instant::now();
        let (audio_bytes, mime_type) =
            client
                .tts_generate(chunk, voice_name)
                .await
                .with_context(|| {
                    format!(
                        "TTS generation failed for {} (part {})",
                        path.display(),
                        i + 1
                    )
                })?;
        println!(
            "{} | TTS part {:02}/{:02}: mime={}, {} bytes, took {:?}",
            now_ts(),
            i + 1,
            chunks.len() - 1,
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
                eprintln!(
                    "warn: WAV merge failed ({}). Falling back to naive concat.",
                    e
                );
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

fn print_help(program: &str) {
    println!(
        "Usage: {program} [OPTIONS] [MARKDOWN_FILE]\n\n\
Options:\n  -v, --voice <NAME>   Choose a voice (default: Zephyr)\n      --list-voices     List available voices and exit\n  -h, --help           Show this help and exit\n\n\
Args:\n  MARKDOWN_FILE        Optional single markdown file. If omitted, processes all book/src/*.md\n\n\
Examples:\n  {program} --voice Zephyr\n  {program} --voice Leda book/src/ch08-02-strings.md\n  {program} --list-voices\n"
    );
    print_voices_brief();
}

fn print_voices() {
    println!("Available voices:");
    for (name, desc) in AVAILABLE_VOICES {
        println!("  {:<16} {}", name, desc);
    }
}

fn print_voices_brief() {
    println!("\nVoices (name — description):");
    for (name, desc) in AVAILABLE_VOICES {
        println!("  {} — {}", name, desc);
    }
}
