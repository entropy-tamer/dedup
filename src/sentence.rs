//! Sentence splitting using compiled regex with memchr optimization.

use once_cell::sync::Lazy;
use regex::Regex;
use memchr::memchr_iter;

/// A sentence with its byte offsets in the original text.
#[derive(Debug, Clone)]
pub struct Sentence {
    pub start: usize,
    pub end: usize,
    pub text: String,
}

/// Split text into sentences for comparison.
///
/// Splits on sentence-ending punctuation followed by whitespace or end of string.
/// Uses compiled regex for efficiency.
pub fn split_sentences(text: &str) -> Vec<Sentence> {
    static SENTENCE_REGEX: Lazy<Regex> = Lazy::new(|| {
        Regex::new(r"[.!?]+(?:\s+|$)").expect("Valid regex")
    });

    let mut sentences = Vec::new();
    let mut current_start = 0;

    for mat in SENTENCE_REGEX.find_iter(text) {
        let sentence_end = mat.end();
        let sentence_text = text[current_start..sentence_end].trim();

        if !sentence_text.is_empty() {
            sentences.push(Sentence {
                start: current_start,
                end: sentence_end,
                text: sentence_text.to_string(),
            });
        }

        current_start = sentence_end;
    }

    // Add remaining text
    let remaining = text[current_start..].trim();
    if !remaining.is_empty() {
        sentences.push(Sentence {
            start: current_start,
            end: text.len(),
            text: remaining.to_string(),
        });
    }

    sentences
}

/// Fast sentence boundary detection using memchr.
/// Returns byte offsets of sentence endings (.!? followed by space or end).
pub fn find_sentence_endings(text: &str) -> Vec<usize> {
    let bytes = text.as_bytes();
    let mut endings = Vec::new();

    for i in memchr_iter(b'.', bytes) {
        if is_sentence_end(bytes, i) {
            endings.push(i + 1);
        }
    }
    for i in memchr_iter(b'!', bytes) {
        if is_sentence_end(bytes, i) {
            endings.push(i + 1);
        }
    }
    for i in memchr_iter(b'?', bytes) {
        if is_sentence_end(bytes, i) {
            endings.push(i + 1);
        }
    }

    endings.sort_unstable();
    endings.dedup();
    endings
}

fn is_sentence_end(bytes: &[u8], idx: usize) -> bool {
    // Check if followed by whitespace or end
    let next_idx = idx + 1;
    if next_idx >= bytes.len() {
        return true;
    }
    matches!(bytes[next_idx], b' ' | b'\t' | b'\n' | b'\r')
}

/// Split sentences using fast memchr-based approach.
/// Less accurate than regex but faster for streaming.
pub fn split_sentences_fast(text: &str) -> Vec<Sentence> {
    let endings = find_sentence_endings(text);
    let mut sentences = Vec::new();
    let mut current_start = 0;

    for &end in &endings {
        let sentence_text = text[current_start..end].trim();
        if !sentence_text.is_empty() {
            sentences.push(Sentence {
                start: current_start,
                end,
                text: sentence_text.to_string(),
            });
        }
        current_start = end;
    }

    let remaining = text[current_start..].trim();
    if !remaining.is_empty() {
        sentences.push(Sentence {
            start: current_start,
            end: text.len(),
            text: remaining.to_string(),
        });
    }

    sentences
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_sentences_basic() {
        let text = "Hello world. How are you? I'm fine!";
        let sentences = split_sentences(text);
        assert_eq!(sentences.len(), 3);
        assert_eq!(sentences[0].text, "Hello world.");
        assert_eq!(sentences[1].text, "How are you?");
        assert_eq!(sentences[2].text, "I'm fine!");
    }

    #[test]
    fn test_split_sentences_no_punctuation() {
        let text = "Hello world";
        let sentences = split_sentences(text);
        assert_eq!(sentences.len(), 1);
        assert_eq!(sentences[0].text, "Hello world");
    }

    #[test]
    fn test_split_sentences_multiple_spaces() {
        let text = "Hello.   World?  Yes!";
        let sentences = split_sentences(text);
        assert_eq!(sentences.len(), 3);
    }

    #[test]
    fn test_find_sentence_endings() {
        let text = "Hello. World? Yes!";
        let endings = find_sentence_endings(text);
        assert_eq!(endings.len(), 3);
    }

    #[test]
    fn test_split_sentences_fast() {
        let text = "Hello. World? Yes!";
        let sentences = split_sentences_fast(text);
        assert_eq!(sentences.len(), 3);
    }
}