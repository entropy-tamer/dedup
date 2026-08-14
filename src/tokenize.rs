//! Tokenization with stop-word filtering using memchr for fast splitting.

use ahash::AHashSet;
use memchr::memchr_iter;
use once_cell::sync::Lazy;

/// Token set using ahash for performance.
pub type TokenSet = AHashSet<String>;

/// Stop words that inflate Jaccard similarity without contributing meaning.
/// Removing them makes similarity more discriminative between genuine
/// repetition and incidental shared function words.
static STOP_WORDS: Lazy<AHashSet<&'static str>> = Lazy::new(|| {
    let words = [
        "the", "a", "an", "is", "are", "was", "were", "of", "in", "to",
        "for", "and", "or", "but", "with", "this", "that", "on", "at",
        "by", "from", "as", "it", "be", "been", "being", "have", "has",
        "had", "do", "does", "did", "will", "would", "could", "should",
        "may", "might", "must", "can", "shall", "i", "you", "he", "she",
        "we", "they", "them", "their", "there", "here",
        "if", "then", "else", "than", "so", "such", "very", "too", "also",
        "just", "only", "even", "still", "yet", "already", "always",
        "never", "often", "sometimes", "usually", "generally", "mostly",
        "some", "any", "each", "every", "all", "both", "either", "neither",
        "one", "two", "first", "last", "next", "previous", "other",
        "another", "same", "different", "new", "old", "good", "bad",
        "right", "left", "up", "down", "out", "off", "over", "under",
        "again", "further", "once", "more", "most", "less", "least",
    ];
    words.iter().copied().collect()
});

/// Tokenize text into words for Jaccard similarity.
///
/// Lowercases and splits on whitespace/punctuation. Stop words are removed so
/// shared function words do not inflate similarity between sentences that
/// differ in meaning.
///
/// Uses memchr for fast byte-level splitting.
pub fn tokenize(text: &str) -> TokenSet {
    let normalized = text.to_lowercase();
    let mut tokens = TokenSet::with_capacity(normalized.len() / 6); // rough estimate

    let bytes = normalized.as_bytes();
    let mut start = 0;

    for i in memchr_iter(b' ', bytes) {
        if i > start {
            let word = &normalized[start..i];
            if word.len() > 1 && !STOP_WORDS.contains(word) {
                tokens.insert(word.to_string());
            }
        }
        start = i + 1;
    }

    // Last token
    if start < bytes.len() {
        let word = &normalized[start..];
        if word.len() > 1 && !STOP_WORDS.contains(word) {
            tokens.insert(word.to_string());
        }
    }

    // Also split on punctuation
    let punct_bytes = b".,;:!?(){}[]<>\"'`";
    let mut final_tokens = TokenSet::with_capacity(tokens.len());
    for token in tokens {
        let mut sub_start = 0;
        let token_bytes = token.as_bytes();
        for i in memchr_iter(token_bytes[0], punct_bytes) {
            // This is a simplified approach - in practice we'd scan the token
        }
        // For now, just filter the token as-is if it has no punctuation
        // A more complete version would split on punctuation too
        let has_punct = token_bytes.iter().any(|&b| {
            memchr_iter(b, punct_bytes).next().is_some()
        });
        if !has_punct {
            final_tokens.insert(token);
        } else {
            // Split on punctuation
            let mut sub_start = 0;
            for (idx, &b) in token_bytes.iter().enumerate() {
                if memchr_iter(b, punct_bytes).next().is_some() {
                    if idx > sub_start {
                        let sub = &token[sub_start..idx];
                        if sub.len() > 1 && !STOP_WORDS.contains(sub) {
                            final_tokens.insert(sub.to_string());
                        }
                    }
                    sub_start = idx + 1;
                }
            }
            if sub_start < token_bytes.len() {
                let sub = &token[sub_start..];
                if sub.len() > 1 && !STOP_WORDS.contains(sub) {
                    final_tokens.insert(sub.to_string());
                }
            }
        }
    }

    final_tokens
}

/// Simplified tokenize for when we just need word splitting.
pub fn tokenize_simple(text: &str) -> TokenSet {
    let normalized = text.to_lowercase();
    let mut tokens = TokenSet::with_capacity(normalized.len() / 6);

    let bytes = normalized.as_bytes();
    let mut start = 0;

    for i in memchr_iter(b' ', bytes) {
        if i > start {
            let word = &normalized[start..i];
            if word.len() > 1 && !STOP_WORDS.contains(word) {
                tokens.insert(word.to_string());
            }
        }
        start = i + 1;
    }

    if start < bytes.len() {
        let word = &normalized[start..];
        if word.len() > 1 && !STOP_WORDS.contains(word) {
            tokens.insert(word.to_string());
        }
    }

    tokens
}

/// Tokenize and return vector (for WASM export).
pub fn tokenize_vec(text: &str) -> Vec<String> {
    tokenize(text).into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_filters_stop_words() {
        let tokens = tokenize("The cat sat and the dog watched");
        assert!(!tokens.contains("the"));
        assert!(!tokens.contains("and"));
        assert!(tokens.contains("cat"));
        assert!(tokens.contains("dog"));
        assert!(tokens.contains("sat"));
        assert!(tokens.contains("watched"));
    }

    #[test]
    fn test_tokenize_keeps_longer_words() {
        let tokens = tokenize("a big ox and i");
        assert!(!tokens.contains("a"));
        assert!(!tokens.contains("i"));
        assert!(tokens.contains("ox"));
        assert!(tokens.contains("big"));
    }

    #[test]
    fn test_tokenize_empty() {
        let tokens = tokenize("");
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_tokenize_punctuation() {
        let tokens = tokenize("Hello, world! How are you?");
        assert!(tokens.contains("hello"));
        assert!(tokens.contains("world"));
        assert!(tokens.contains("how"));
        assert!(!tokens.contains("are"));
        assert!(!tokens.contains("you"));
    }

    #[test]
    fn test_tokenize_vec() {
        let tokens = tokenize_vec("hello world");
        assert_eq!(tokens.len(), 2);
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
    }
}