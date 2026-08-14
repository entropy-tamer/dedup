//! Similarity computations: Jaccard and character n-gram.

use ahash::AHashSet;
use std::collections::HashSet;

/// Token set type using ahash for performance.
pub type TokenSet = AHashSet<String>;

/// Compute Jaccard similarity between two token sets.
///
/// Jaccard = |A �� B| / |A �� B|
/// Early exit optimization: if intersection/union ratio can't reach threshold,
/// we could stop early, but we compute exactly for now.
pub fn jaccard_similarity(a: &TokenSet, b: &TokenSet) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }

    // Iterate over smaller set for efficiency
    let (smaller, larger) = if a.len() <= b.len() { (a, b) } else { (b, a) };

    let mut intersection = 0;
    for token in smaller {
        if larger.contains(token) {
            intersection += 1;
        }
    }

    let union = a.len() + b.len() - intersection;
    intersection as f64 / union as f64
}

/// Compute Jaccard similarity with early exit threshold.
///
/// Returns `None` if similarity definitely below threshold, `Some(score)` otherwise.
pub fn jaccard_similarity_thresholded(a: &TokenSet, b: &TokenSet, threshold: f64) -> Option<f64> {
    if a.is_empty() && b.is_empty() {
        return Some(1.0);
    }
    if a.is_empty() || b.is_empty() {
        return if threshold <= 0.0 { Some(0.0) } else { None };
    }

    let (smaller, larger) = if a.len() <= b.len() { (a, b) } else { (b, a) };

    let mut intersection = 0;
    let min_intersection_needed = (threshold * (a.len() + b.len()) as f64 / (1.0 + threshold)).ceil() as usize;

    for token in smaller {
        if larger.contains(token) {
            intersection += 1;
            if intersection >= min_intersection_needed {
                // Early exit: we know similarity will exceed threshold
                let union = a.len() + b.len() - intersection;
                return Some(intersection as f64 / union as f64);
            }
        }
    }

    let union = a.len() + b.len() - intersection;
    let similarity = intersection as f64 / union as f64;
    if similarity >= threshold {
        Some(similarity)
    } else {
        None
    }
}

/// Compute character n-gram similarity between two strings.
///
/// Uses fixed-size n-grams (default n=4). Creates sets of character n-grams
/// and computes Jaccard similarity.
pub fn char_ngram_similarity(a: &str, b: &str, ngram_size: usize) -> f64 {
    if a.len() < ngram_size || b.len() < ngram_size {
        return if a == b { 1.0 } else { 0.0 };
    }

    let a_ngrams = char_ngram_set(a, ngram_size);
    let b_ngrams = char_ngram_set(b, ngram_size);

    jaccard_similarity(&a_ngrams, &b_ngrams)
}

/// Extract character n-grams from a string into a set.
fn char_ngram_set(text: &str, ngram_size: usize) -> TokenSet {
    let bytes = text.as_bytes();
    let mut set = TokenSet::with_capacity(bytes.len().saturating_sub(ngram_size).max(1));

    for i in 0..=bytes.len().saturating_sub(ngram_size) {
        // SAFETY: We're creating valid UTF-8 substrings from valid UTF-8
        let ngram = std::str::from_utf8(&bytes[i..i + ngram_size]).unwrap_or("");
        if !ngram.is_empty() {
            set.insert(ngram.to_string());
        }
    }
    set
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jaccard_identical() {
        let mut a = TokenSet::new();
        a.insert("hello".to_string());
        a.insert("world".to_string());
        let mut b = a.clone();
        assert_eq!(jaccard_similarity(&a, &b), 1.0);
    }

    #[test]
    fn test_jaccard_disjoint() {
        let mut a = TokenSet::new();
        a.insert("hello".to_string());
        let mut b = TokenSet::new();
        b.insert("world".to_string());
        assert_eq!(jaccard_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_jaccard_partial() {
        let mut a = TokenSet::new();
        a.insert("hello".to_string());
        a.insert("world".to_string());
        a.insert("foo".to_string());
        let mut b = TokenSet::new();
        b.insert("hello".to_string());
        b.insert("world".to_string());
        b.insert("bar".to_string());
        // 2/4 = 0.5
        assert_eq!(jaccard_similarity(&a, &b), 0.5);
    }

    #[test]
    fn test_jaccard_empty() {
        let a = TokenSet::new();
        let b = TokenSet::new();
        assert_eq!(jaccard_similarity(&a, &b), 1.0);
    }

    #[test]
    fn test_char_ngram_similarity() {
        let a = "hello world";
        let b = "hello world";
        assert_eq!(char_ngram_similarity(a, b, 4), 1.0);
    }

    #[test]
    fn test_char_ngram_different() {
        let a = "hello world";
        let b = "hello moon";  // Shares some 4-char n-grams
        let sim = char_ngram_similarity(a, b, 4);
        assert!(sim < 1.0);
        assert!(sim > 0.0);
    }

    #[test]
    fn test_char_ngram_short() {
        let a = "hi";
        let b = "hi";
        assert_eq!(char_ngram_similarity(a, b, 4), 1.0);
    }

    #[test]
    fn test_jaccard_thresholded() {
        let mut a = TokenSet::new();
        a.insert("a".to_string());
        a.insert("b".to_string());
        let mut b = TokenSet::new();
        b.insert("a".to_string());
        b.insert("c".to_string());
        // similarity = 1/3 = 0.33
        assert!(jaccard_similarity_thresholded(&a, &b, 0.5).is_none());
        assert!(jaccard_similarity_thresholded(&a, &b, 0.3).is_some());
    }
}