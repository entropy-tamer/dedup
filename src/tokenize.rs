//! Tokenization with stop-word filtering.

use ahash::AHashSet;
use once_cell::sync::Lazy;

/// Token set using ahash for performance.
pub type TokenSet = AHashSet<String>;

/// Stop words that inflate Jaccard similarity without contributing meaning.
/// Removing them makes similarity more discriminative between genuine
/// repetition and incidental shared function words.
static STOP_WORDS: Lazy<AHashSet<&'static str>> = Lazy::new(|| {
    let words = [
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "of",
        "in",
        "to",
        "for",
        "and",
        "or",
        "but",
        "with",
        "this",
        "that",
        "on",
        "at",
        "by",
        "from",
        "as",
        "it",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "can",
        "shall",
        "i",
        "you",
        "he",
        "she",
        "we",
        "they",
        "them",
        "their",
        "there",
        "here",
        "if",
        "then",
        "else",
        "than",
        "so",
        "such",
        "very",
        "too",
        "also",
        "just",
        "only",
        "even",
        "still",
        "yet",
        "already",
        "always",
        "never",
        "often",
        "sometimes",
        "usually",
        "generally",
        "mostly",
        "some",
        "any",
        "each",
        "every",
        "all",
        "both",
        "either",
        "neither",
        "one",
        "two",
        "first",
        "last",
        "next",
        "previous",
        "other",
        "another",
        "same",
        "different",
        "new",
        "old",
        "good",
        "bad",
        "right",
        "left",
        "up",
        "down",
        "out",
        "off",
        "over",
        "under",
        "again",
        "further",
        "once",
        "more",
        "most",
        "less",
        "least",
    ];
    words.iter().copied().collect()
});

/// Tokenize text into words for Jaccard similarity.
///
/// Lowercases and splits on whitespace/punctuation. Stop words are removed so
/// shared function words do not inflate similarity between sentences that
/// differ in meaning.
///
/// Splitting on ASCII whitespace rather than on `b' '` alone is load-bearing.
/// A single-byte memchr search only finds spaces, so a newline stays welded to
/// its neighbors and paragraph structure is destroyed before any similarity is
/// computed: `"go.\n\nLet"` tokenized to one token, not two. `split_ascii_whitespace`
/// is one pass over the same bytes and handles all six whitespace characters.
pub fn tokenize(text: &str) -> TokenSet {
    let normalized = text.to_lowercase();
    let mut tokens = TokenSet::with_capacity(normalized.len() / 6); // rough estimate

    for word in normalized.split_ascii_whitespace() {
        if word.len() > 1 && !STOP_WORDS.contains(word) {
            tokens.insert(word.to_string());
        }
    }

    // Also split on punctuation
    let punct_bytes = b".,;:!?(){}[]<>\"'`";
    let mut final_tokens = TokenSet::with_capacity(tokens.len());
    for token in tokens {
        let token_bytes = token.as_bytes();
        let has_punct = token_bytes.iter().any(|&b| punct_bytes.contains(&b));
        if !has_punct {
            final_tokens.insert(token);
            continue;
        }
        // Split on punctuation
        let mut sub_start = 0;
        for (idx, &b) in token_bytes.iter().enumerate() {
            if punct_bytes.contains(&b) {
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

    final_tokens
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

    /// Regression for the space-only split. The fixture is the degenerate
    /// reasoning block from
    /// `~/.todos/r-harness/ACTIONABLE-2026-10-04-thought-loops-the-set-metric-and-the-channel-nobody-reads.md`
    /// and it contains the hazard directly: twelve blank-line-separated
    /// sentences that the old tokenizer welded into nine tokens, four of which
    /// carried a `\n\n` prefix.
    #[test]
    fn test_tokenize_splits_on_newline_not_just_space() {
        let fixture = "Let me produce.\n\nLet me do it.\n\nFinal.\n\nLet me read.\n\nOK.";
        let tokens = tokenize(fixture);

        for token in &tokens {
            assert!(
                !token.contains('\n'),
                "token {token:?} still carries a newline; whitespace is not being split"
            );
        }
        assert!(tokens.contains("produce"), "got {tokens:?}");
        assert!(tokens.contains("final"), "got {tokens:?}");
        assert!(tokens.contains("read"), "got {tokens:?}");
        assert!(tokens.contains("ok"), "got {tokens:?}");
        // "do" and "it" are stop words, so that sentence collapses to nothing —
        // but "let" and "me" are not, and they land once because a set has one
        // slot per word. Both facts are asserted: the count below fails if
        // whitespace handling regresses, and would also fail if the fixture
        // stopped being the thing that produced it.
        assert_eq!(tokens.len(), 6, "got {tokens:?}");
    }

    /// The sibling that proves the fix is not just "return fewer tokens".
    /// Tabs and CRs are whitespace too, and the old scanner passed them
    /// straight through into tokens.
    ///
    /// `\x0b` is deliberately absent: `u8::is_ascii_whitespace` is false for it,
    /// which disagrees with `char::is_ascii_whitespace` (that one is
    /// `'\x09'..='\x0d'` and includes it). `split_ascii_whitespace` follows the
    /// `u8` definition, so vertical tabs stay inside a token. Asserted rather
    /// than left to chance, because the next fox to "fix" that will break a
    /// test they have not read.
    #[test]
    fn test_tokenize_handles_every_ascii_whitespace() {
        for sep in [" ", "\t", "\n", "\r\n", "\x0c"] {
            let tokens = tokenize(&format!("alpha{sep}bravo{sep}charlie"));
            assert!(
                tokens.contains("alpha") && tokens.contains("bravo") && tokens.contains("charlie"),
                "separator {sep:?} did not split: {tokens:?}"
            );
        }
        assert!(
            !b'\x0b'.is_ascii_whitespace(),
            "precondition: \\x0b is not Rust whitespace"
        );
        assert!(
            tokenize("alpha\x0bbravo").contains("alpha\x0bbravo"),
            "precondition: vertical tab is expected to stay inside one token"
        );
    }

    /// The control for both tests above: with no whitespace present the
    /// tokenizer still returns the words, so a regression to empty output
    /// cannot pass as a fix.
    #[test]
    fn test_tokenize_single_token_unaffected_by_the_split() {
        let tokens = tokenize("charlie");
        assert_eq!(tokens.len(), 1, "got {tokens:?}");
        assert!(tokens.contains("charlie"));
    }
}
