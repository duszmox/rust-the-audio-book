use anyhow::Result;
use regex::Regex;
use std::time::Instant;

use crate::tts::GeminiClient;

pub async fn replace_code_blocks_with_summaries(
    client: &GeminiClient,
    input: &str,
) -> Result<(String, usize)> {
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
        } else if is_fence_close(line) {
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

            out.push_str(&summary);
            out.push('\n');
            out.push_str(line);
            out.push('\n');

            in_block = false;
            code_acc.clear();
        } else {
            code_acc.push(line.to_string());
        }
    }

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

pub fn split_into_chunks_by_paragraph(input: &str, max_chars: usize) -> Vec<String> {
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
                    if nl_count >= 1 {
                        last_para_break = Some(i);
                    }
                    i = j;
                    continue;
                }
                i += 1;
            }

            if let Some(bp) = last_para_break {
                end = bp + 1;
            } else {
                let mut last_sentence_end: Option<usize> = None;
                for k in start..end {
                    if matches!(chars[k], '.' | '!' | '?') {
                        last_sentence_end = Some(k);
                    }
                }
                if let Some(se) = last_sentence_end {
                    end = se + 1;
                }
            }
        }

        let slice: String = chars[start..end].iter().collect();
        chunks.push(slice);
        start = end;
    }

    chunks
}

fn remove_links_for_tts(input: &str) -> String {
    // 0) Convert Markdown images to their alt text (drop the image itself)
    //    Examples: ![Alt text](url) -> Alt text,  ![Alt][id] -> Alt
    let re_img_inline = Regex::new(r"!\[([^\]]+)\]\([^\)]+\)").unwrap();
    let tmp = re_img_inline.replace_all(input, "$1").into_owned();
    let re_img_ref = Regex::new(r"!\[([^\]]+)\]\[[^\]]*\]").unwrap();
    let tmp = re_img_ref.replace_all(&tmp, "$1").into_owned();

    // 1) Drop reference-style link definitions like: [id]: url "title"
    let mut filtered_lines = Vec::new();
    for line in tmp.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with('[') && trimmed.contains("]: ") {
            continue;
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

    // 5) Remove bare URLs http(s)://...
    let re_bare = Regex::new(r"https?://\S+").unwrap();
    let tmp = re_bare.replace_all(&tmp, "").into_owned();

    tmp
}

pub fn sanitize_markdown_for_tts(input: &str) -> String {
    // First, remove links
    let mut text = remove_links_for_tts(input);

    // Drop lines starting with <Listing or </Listing and code fences
    let mut lines = Vec::new();
    for line in text.lines() {
        let t = line.trim_start();
        if t.starts_with("<Listing") || t.starts_with("</Listing") {
            continue;
        }
        if t.starts_with("```") {
            continue;
        }
        lines.push(line);
    }
    text = lines.join("\n");

    // Replace HTML <img ... alt="..."> with its alt text before stripping tags
    let re_img_tag =
        Regex::new(r#"(?is)<img\b[^>]*?alt\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))[^>]*>"#)
            .unwrap();
    text = re_img_tag
        .replace_all(&text, |caps: &regex::Captures| {
            caps.get(1)
                .or_else(|| caps.get(2))
                .or_else(|| caps.get(3))
                .map(|m| m.as_str().to_string())
                .unwrap_or_default()
        })
        .into_owned();

    // Remove inline HTML tags and comments
    let re_comment = Regex::new(r"(?s)<!--.*?-->").unwrap();
    text = re_comment.replace_all(&text, "").into_owned();
    let re_tags = Regex::new(r"</?[^>]+>").unwrap();
    text = re_tags.replace_all(&text, "").into_owned();

    // Remove backticks (inline code markers)
    text = text.replace('`', "");

    // Strip heading #'s, blockquote '>'s, and list markers
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

    joined.trim().to_string()
}
