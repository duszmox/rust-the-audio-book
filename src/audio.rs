use anyhow::{Result, anyhow};

pub fn guess_audio_extension(mime: &str) -> &'static str {
    match mime {
        m if m.contains("mpeg") || m.contains("mp3") => ".mp3",
        m if m.contains("wav")
            || m.contains("x-wav")
            || m.contains("pcm")
            || m.contains("linear16") =>
        {
            ".wav"
        }
        m if m.contains("ogg") => ".ogg",
        m if m.contains("flac") => ".flac",
        _ => ".bin",
    }
}

pub fn merge_mp3(parts: &[&[u8]]) -> Vec<u8> {
    // Simple byte concatenation; most players handle back-to-back MP3 frames.
    merge_concat(parts)
}

pub fn merge_concat(parts: &[&[u8]]) -> Vec<u8> {
    let total: usize = parts.iter().map(|p| p.len()).sum();
    let mut out = Vec::with_capacity(total);
    for p in parts {
        out.extend_from_slice(p);
    }
    out
}

pub fn try_merge_wav(parts: &[&[u8]]) -> Result<Vec<u8>> {
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
    audio_format: u16, // 1 = PCM, 3 = IEEE float
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
        if chunk_data_end > bytes.len() {
            break;
        }
        if id == b"fmt " {
            if sz < 16 {
                return Err(anyhow!("fmt chunk too small"));
            }
            let audio_format = u16::from_le_bytes(
                bytes[chunk_data_start..chunk_data_start + 2]
                    .try_into()
                    .unwrap(),
            );
            let num_channels = u16::from_le_bytes(
                bytes[chunk_data_start + 2..chunk_data_start + 4]
                    .try_into()
                    .unwrap(),
            );
            let sample_rate = u32::from_le_bytes(
                bytes[chunk_data_start + 4..chunk_data_start + 8]
                    .try_into()
                    .unwrap(),
            );
            let byte_rate = u32::from_le_bytes(
                bytes[chunk_data_start + 8..chunk_data_start + 12]
                    .try_into()
                    .unwrap(),
            );
            let block_align = u16::from_le_bytes(
                bytes[chunk_data_start + 12..chunk_data_start + 14]
                    .try_into()
                    .unwrap(),
            );
            let bits_per_sample = u16::from_le_bytes(
                bytes[chunk_data_start + 14..chunk_data_start + 16]
                    .try_into()
                    .unwrap(),
            );
            fmt = Some((
                WavFmt {
                    audio_format,
                    num_channels,
                    sample_rate,
                    byte_rate,
                    block_align,
                    bits_per_sample,
                },
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
        if chunk_data_end > bytes.len() {
            break;
        }
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

pub fn is_raw_linear_pcm(mime: &str) -> bool {
    let m = mime.to_ascii_lowercase();
    (m.contains("linear16") || m.contains("pcm")) && !m.contains("wav")
}

pub fn parse_sample_rate(mime: &str) -> Option<u32> {
    // examples: "audio/pcm;rate=24000" or "audio/linear16; sample_rate=16000"
    let lower = mime.to_ascii_lowercase();
    for key in ["rate=", "samplerate=", "sample_rate="] {
        if let Some(pos) = lower.find(key) {
            let tail = &lower[pos + key.len()..];
            let mut num = String::new();
            for ch in tail.chars() {
                if ch.is_ascii_digit() {
                    num.push(ch);
                } else {
                    break;
                }
            }
            if let Ok(v) = num.parse::<u32>() {
                return Some(v);
            }
        }
    }
    None
}

pub fn wrap_pcm_to_wav(
    pcm: &[u8],
    sample_rate: u32,
    channels: u16,
    bits_per_sample: u16,
) -> Result<Vec<u8>> {
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

/// Estimate the ratio of samples that are effectively silent for WAV data.
/// Returns a value in [0.0, 1.0] where 1.0 means all samples are near-zero.
/// Supports PCM 16-bit integer and 32-bit float WAV.
pub fn estimate_wav_silence_ratio(bytes: &[u8]) -> Result<f32> {
    let (fmt, _fmt_size) = parse_wav_fmt(bytes)?;
    let data = parse_wav_data(bytes)?;

    if data.is_empty() {
        return Ok(1.0);
    }

    let bps = fmt.bits_per_sample;

    match (fmt.audio_format, bps) {
        // PCM 16-bit
        (1, 16) => {
            let sample_count = data.len() / 2; // 2 bytes per sample (per channel)
            if sample_count == 0 { return Ok(1.0); }
            let mut silent = 0usize;
            let mut total = 0usize;
            let threshold: i32 = (32767f32 * 0.01) as i32; // ~ -40 dBFS
            let mut i = 0;
            while i + 1 < data.len() {
                let s = i16::from_le_bytes([data[i], data[i+1]]) as i32;
                if s.abs() as i32 <= threshold { silent += 1; }
                total += 1;
                i += 2;
            }
            Ok((silent as f32) / (total as f32))
        }
        // IEEE float 32-bit
        (3, 32) => {
            let sample_count = data.len() / 4;
            if sample_count == 0 { return Ok(1.0); }
            let mut silent = 0usize;
            let mut total = 0usize;
            let mut i = 0;
            let threshold = 0.005f32; // 0.5% FS
            while i + 3 < data.len() {
                let v = f32::from_le_bytes([
                    data[i], data[i+1], data[i+2], data[i+3]
                ]);
                if v.abs() <= threshold { silent += 1; }
                total += 1;
                i += 4;
            }
            Ok((silent as f32) / (total as f32))
        }
        // Unsupported formats — fall back to a basic zero-byte heuristic per frame
        _ => {
            // Heuristic: consider frames (block_align) and count near-zero frames
            let ba = fmt.block_align.max(1) as usize;
            let mut silent = 0usize;
            let mut total = 0usize;
            let mut i = 0usize;
            while i + ba <= data.len() {
                let frame = &data[i..i+ba];
                // If most bytes in frame are 0x00 or 0x80 (common silence centers), mark silent
                let zeros = frame.iter().filter(|b| **b == 0x00 || **b == 0x80).count();
                if zeros * 2 >= frame.len() { // >= 50% zero-ish bytes
                    silent += 1;
                }
                total += 1;
                i += ba;
            }
            if total == 0 { return Ok(1.0); }
            Ok((silent as f32) / (total as f32))
        }
    }
}

/// Convenience helper: if `mime` indicates WAV, estimate silence ratio.
/// Returns None when mime is not WAV or parsing fails.
pub fn try_silence_ratio_from_mime(bytes: &[u8], mime: &str) -> Option<f32> {
    let lower = mime.to_ascii_lowercase();
    if lower.contains("wav") || lower.contains("x-wav") {
        estimate_wav_silence_ratio(bytes).ok()
    } else {
        None
    }
}
