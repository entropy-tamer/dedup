//! N-gram computation with rolling hash (Rabin-Karp) for O(n) performance.

use ahash::AHashMap;

/// Compute n-gram hashes using rolling hash for efficiency.
///
/// Instead of allocating substring slices, we compute a rolling hash
/// for each window position. This gives O(n) time vs O(n × window_size)
/// for naive substring extraction.
///
/// For step_size > 1, we compute all rolling hashes (step=1) then subsample,
/// since the rolling formula only works for single-character steps.
pub fn compute_ngram_hashes(text: &str, window_size: usize, step_size: usize) -> Vec<u64> {
    let bytes = text.as_bytes();
    let len = bytes.len();

    if len < window_size {
        return vec![hash_bytes(bytes)];
    }

    let hasher = NgramHasher::new(window_size);

    // Compute all rolling hashes with step=1
    let mut all_hashes = Vec::with_capacity(len - window_size + 1);
    let mut hash = hasher.initial_hash(bytes);
    all_hashes.push(hash);

    for i in 1..=len.saturating_sub(window_size) {
        hash = hasher.roll_hash(hash, bytes[i - 1], bytes[i + window_size - 1]);
        all_hashes.push(hash);
    }

    // Subsample by step_size
    let mut hashes = Vec::with_capacity((len - window_size) / step_size + 1);
    for i in (0..all_hashes.len()).step_by(step_size) {
        hashes.push(all_hashes[i]);
    }

    hashes
}

/// Compute n-grams as string slices (for verification on hash collision).
/// Used sparingly - only when hash comparison indicates potential match.
pub fn compute_ngram_strings(text: &str, window_size: usize, step_size: usize) -> Vec<&str> {
    let len = text.len();
    if len < window_size {
        return vec![text];
    }

    let mut ngrams = Vec::with_capacity((len - window_size) / step_size + 1);
    for i in (0..=len - window_size).step_by(step_size) {
        ngrams.push(&text[i..i + window_size]);
    }
    ngrams
}

/// N-gram hasher using polynomial rolling hash (Rabin-Karp).
///
/// Base 131, modulus 2^64 (natural overflow).
/// Fast, good distribution for text.
#[derive(Clone, Copy)]
pub struct NgramHasher {
    window_size: usize,
    base: u64,
    base_pow: u64, // base^(window_size-1)
}

impl NgramHasher {
    pub fn new(window_size: usize) -> Self {
        const BASE: u64 = 131;
        let mut base_pow = 1u64;
        for _ in 1..window_size {
            base_pow = base_pow.wrapping_mul(BASE);
        }
        Self {
            window_size,
            base: BASE,
            base_pow,
        }
    }

    /// Compute hash for first window.
    pub fn initial_hash(&self, bytes: &[u8]) -> u64 {
        let mut hash = 0u64;
        for &b in &bytes[..self.window_size.min(bytes.len())] {
            hash = hash.wrapping_mul(self.base).wrapping_add(b as u64);
        }
        hash
    }

    /// Roll hash: remove leading byte, add trailing byte.
    pub fn roll_hash(&self, prev_hash: u64, leading: u8, trailing: u8) -> u64 {
        let leading_contribution = (leading as u64).wrapping_mul(self.base_pow);
        prev_hash
            .wrapping_sub(leading_contribution)
            .wrapping_mul(self.base)
            .wrapping_add(trailing as u64)
    }
}

/// Hash arbitrary byte slice.
pub fn hash_bytes(bytes: &[u8]) -> u64 {
    const BASE: u64 = 131;
    let mut hash = 0u64;
    for &b in bytes {
        hash = hash.wrapping_mul(BASE).wrapping_add(b as u64);
    }
    hash
}

/// Find candidate matching n-gram pairs using hash comparison.
///
/// Returns pairs of (i, j) indices where hashes match.
/// Caller should verify with actual string comparison.
///
/// Performance note: each position is paired ONLY with the first occurrence
/// of its hash, giving O(n) candidate pairs even on fully repetitive text.
/// The old all-pairs-per-bucket behavior was O(k^2) per bucket and made
/// check_repetition super-quadratic on repetitive input (6.8s @ 2KB).
pub fn find_hash_matches(
    hashes: &[u64],
    _threshold: f64, // Not used for hash, but kept for API consistency
) -> Vec<(usize, usize)> {
    let mut positions: AHashMap<u64, Vec<usize>> = AHashMap::default();
    for (idx, &h) in hashes.iter().enumerate() {
        positions.entry(h).or_default().push(idx);
    }

    let mut matches = Vec::new();
    for positions_list in positions.values() {
        if positions_list.len() > 1 {
            let first = positions_list[0];
            for &later in &positions_list[1..] {
                matches.push((first.min(later), first.max(later)));
            }
        }
    }
    matches
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_ngram_hashes() {
        let text = "abcdefghij";
        let hashes = compute_ngram_hashes(text, 4, 2);
        // Should have 4 hashes for windows at 0,2,4,6
        assert_eq!(hashes.len(), 4);
    }

    #[test]
    fn test_rolling_hash_consistency() {
        let text = "abcdefghij";
        let hashes = compute_ngram_hashes(text, 4, 1);

        // Verify rolling hash matches direct computation
        let hasher = NgramHasher::new(4);
        let bytes = text.as_bytes();
        let direct_hash = hasher.initial_hash(&bytes[2..]);
        assert_eq!(hashes[2], direct_hash);
    }

    #[test]
    fn test_short_text() {
        let text = "abc";
        let hashes = compute_ngram_hashes(text, 10, 5);
        assert_eq!(hashes.len(), 1);
    }

    #[test]
    fn test_find_hash_matches() {
        // Text with repeated pattern
        let text = "hello world hello world";
        let hashes = compute_ngram_hashes(text, 5, 1);
        let matches = find_hash_matches(&hashes, 0.0);
        assert!(!matches.is_empty());
    }
}
