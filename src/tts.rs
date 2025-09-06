use anyhow::{anyhow, Context, Result};
use base64::Engine;
use reqwest::header::{CONTENT_TYPE, RETRY_AFTER};
use reqwest::StatusCode;
use tokio::time::{sleep, Duration};

use crate::audio::{is_raw_linear_pcm, parse_sample_rate, wrap_pcm_to_wav};

pub struct GeminiClient {
    http: reqwest::Client,
    api_key: String,
    base_url: String,
}

impl GeminiClient {
    pub fn new(api_key: String) -> Result<Self> {
        let http = reqwest::Client::builder()
            .user_agent("rust-the-audio-book/0.1")
            .build()?;
        Ok(Self {
            http,
            api_key,
            base_url: "https://generativelanguage.googleapis.com/v1beta".to_string(),
        })
    }

    pub async fn summarize_code_block(&self, code: &str) -> Result<String> {
        let prompt = format!(
            "You are helping write an audio book. Summarize the following code block succinctly (2-4 sentences) for listeners.\n\
             Focus on what it does and why it matters.\n\
             Avoid code, jargon, and backticks.\n\
             Keep it clear and engaging.\n\
\nCode block:\n{code}"
        );

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

    pub async fn tts_generate(&self, input_text: &str) -> Result<(Vec<u8>, String)> {
        let url = format!(
            "{}/models/{}:{}?key={}",
            self.base_url, "gemini-2.5-pro-preview-tts", "generateContent", self.api_key
        );

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
    let base_secs = 1u64.checked_shl(attempt as u32).unwrap_or(u64::MAX).min(60);
    let jitter_ms = ((attempt as u64 + 1) * 137) % 500;
    Duration::from_secs(base_secs) + Duration::from_millis(jitter_ms)
}

