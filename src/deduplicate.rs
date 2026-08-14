//! High-level deduplication logic ported from TypeScript deduplication.ts

use crate::ngram::{compute_ngram_hashes, find_hash_matches, compute_ngram_strings};
use crate::similarity::{jaccard_similarity, char_ngram_similarity, TokenSet};
use crate::tokenize::tokenize;
use crate::sentence::{split_sentences, Sentence};
use ahash::AHashSet;
use serde::{Deserialize, Serialize};

/// Configuration for deduplication.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DedupSettings {
    pub enabled: bool,
    pub ngram_similarity_threshold: f64,
    pub sentence_similarity_threshold: f64,
    pub sentence_min_repeat_length: usize,
    pub context_gap_max: usize,
    pub window_size: usize,
    pub step_size: usize,
    pub min_repeat_length: usize,
    pub max_repeats: usize,
}

impl Default for DedupSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            ngram_similarity_threshold: 0.95,
            sentence_similarity_threshold: 0.75,
            sentence_min_repeat_length: 80,
            context_gap_max: 300,
            window_size: 100,
            step_size: 25,
            min_repeat_length: 50,
            max_repeats: 3,
        }
    }
}

/// Source of detection for a repeated section.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SectionSource {
    Ngram,
    Sentence,
}

/// A repeated section found in the text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepeatedSection {
    pub first_start: usize,
    pub first_end: usize,
    pub repeat_start: usize,
    pub repeat_end: usize,
    pub similarity: f64,
    pub source: SectionSource,
}

/// Result from checking text for repetition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepetitionCheck {
    pub repeated: bool,
    pub sections: Vec<RepeatedSection>,
}

/// Streaming state for incremental deduplication.
pub struct StreamingState {
    settings: DedupSettings,
    content: String,
    ngram_hashes: Vec<u64>,
    ngram_strings: Vec<String>, // Only stored for verification
    sentence_boundaries: Vec<usize>,
    pub repetition_count: usize,
    _last_check_content_len: usize,
}

impl StreamingState {
    pub fn new(settings: DedupSettings) -> Self {
        Self {
            settings,
            content: String::new(),
            ngram_hashes: Vec::new(),
            ngram_strings: Vec::new(),
            sentence_boundaries: Vec::new(),
            repetition_count: 0,
            _last_check_content_len: 0,
        }
    }

    /// Feed a new chunk and return repetition check result for the new content.
    pub fn feed_chunk(&mut self, chunk: &str) -> RepetitionCheck {
        self.content.push_str(chunk);

        // Only check if we have enough new content
        if self.content.len() < self.settings.min_repeat_length * 2 {
            return RepetitionCheck { repeated: false, sections: Vec::new() };
        }

        // Update incremental n-gram hashes
        self.update_ngrams();

        // Run full check (could be optimized to only check new regions)
                check_repetition_incremental(&self.content, &self.settings, self._last_check_content_len)
    }

    fn update_ngrams(&mut self) {
        // Recompute n-gram hashes for full content (simpler, still fast with rolling hash)
        // For true incremental, we'd only compute new windows
        self.ngram_hashes = compute_ngram_hashes(
            &self.content,
            self.settings.window_size,
            self.settings.step_size,
        );
        self.ngram_strings = compute_ngram_strings(
            &self.content,
            self.settings.window_size,
            self.settings.step_size,
        ).into_iter().map(|s| s.to_string()).collect();

        // Update sentence boundaries
        self.sentence_boundaries = crate::sentence::find_sentence_endings(&self.content);
    }
}

/// Check text for repetition using n-gram similarity and sentence comparison.
///
/// Two-pass approach:
/// 1. N-gram similarity for near-identical repetition (fast, catches exact duplicates)
/// 2. Sentence-level comparison for semantic repetition (catches rewording)
pub fn check_repetition(text: &str, settings: &DedupSettings) -> RepetitionCheck {
    if !settings.enabled || text.len() < settings.min_repeat_length * 2 {
        return RepetitionCheck { repeated: false, sections: Vec::new() };
    }

    let mut sections = Vec::new();

    // First pass: n-gram similarity for near-identical repetition
    let ngram_sections = check_ngram_pass(text, settings);
    sections.extend(ngram_sections);

    // Second pass: sentence-level comparison for semantic repetition
    let sentence_sections = check_sentence_pass(text, settings, &sections);
    sections.extend(sentence_sections);

    RepetitionCheck {
        repeated: !sections.is_empty(),
        sections,
    }
}

/// Incremental check - only analyzes newly added content.
fn check_repetition_incremental(
    text: &str,
    settings: &DedupSettings,
    last_check_len: usize,
) -> RepetitionCheck {
    // For now, run full check. True incremental would only compare
    // new windows against existing windows.
    check_repetition(text, settings)
}

/// First pass: n-gram similarity for near-identical repetition.
fn check_ngram_pass(text: &str, settings: &DedupSettings) -> Vec<RepeatedSection> {
    let half_text = text.len() / 2;
    let effective_window_size = settings.window_size.min(half_text).max(20);
    let effective_step_size = settings.step_size.min(effective_window_size / 2).max(1);

    // Use smaller window for hash matching (30-50 chars) to detect exact repetitions
    // regardless of surrounding context. Use step=1 to catch all positions.
    let hash_window_size = effective_window_size.min(50).max(30);
    let hash_step_size = 1;

    // Compute n-gram hashes with smaller window
    let hashes = compute_ngram_hashes(text, hash_window_size, hash_step_size);
    if hashes.len() < 2 {
        return Vec::new();
    }

    // Find hash matches
    let matches = find_hash_matches(&hashes, settings.ngram_similarity_threshold);
    if matches.is_empty() {
        return Vec::new();
    }

    // Get n-gram strings for verification using the hash window size (same indices)
    let ngram_strings = compute_ngram_strings(text, hash_window_size, hash_step_size);

    let mut sections = Vec::new();
    let mut flagged = AHashSet::new();

    for (i, j) in matches {
        let key = format!("{}-{}", i, j);
        if flagged.contains(&key) {
            continue;
        }

        // Verify with actual string comparison
        let a_text = ngram_strings[i];
        let b_text = ngram_strings[j];

        let a_tokens = tokenize(a_text);
        let b_tokens = tokenize(b_text);

        let word_sim = jaccard_similarity(&a_tokens, &b_tokens);
        let char_sim = char_ngram_similarity(a_text, b_text, 4);
        let similarity = word_sim.max(char_sim);

        if similarity >= settings.ngram_similarity_threshold {
            // i and j are hash indices (step=1), so they correspond to byte positions
            let a_start = i * hash_step_size;
            let a_end = (a_start + effective_window_size).min(text.len());
            let b_start = j * hash_step_size;
            let b_end = (b_start + effective_window_size).min(text.len());

            let repeat_length = b_end - b_start;
            if repeat_length >= settings.min_repeat_length {
                sections.push(RepeatedSection {
                    first_start: a_start,
                    first_end: a_end,
                    repeat_start: b_start,
                    repeat_end: b_end,
                    similarity,
                    source: SectionSource::Ngram,
                });

                // Flag overlapping ranges
                for k in i..=j {
                    for l in k + 1..=j + 1 {
                        flagged.insert(format!("{}-{}", k, l));
                    }
                }
            }
        }
    }

    sections
}

/// Second pass: sentence-level comparison for semantic repetition.
fn check_sentence_pass(
    text: &str,
    settings: &DedupSettings,
    existing_sections: &[RepeatedSection],
) -> Vec<RepeatedSection> {
    let sentences = split_sentences(text);
    if sentences.len() < 2 {
        return Vec::new();
    }

    let mut sentence_matches = Vec::new();

    for i in 0..sentences.len() {
        for j in i + 1..sentences.len() {
            let a = &sentences[i];
            let b = &sentences[j];

            // Skip pairs already covered by n-gram detection
            let already_flagged = existing_sections.iter().any(|s| {
                s.repeat_start <= b.start && s.repeat_end >= b.end &&
                s.first_start <= a.start && s.first_end >= a.end
            });
            if already_flagged {
                continue;
            }

            // Require proximity
            let gap = b.start.saturating_sub(a.end);
            if gap > settings.context_gap_max {
                continue;
            }

            let a_tokens = tokenize(&a.text);
            let b_tokens = tokenize(&b.text);
            let word_sim = jaccard_similarity(&a_tokens, &b_tokens);

            if word_sim >= settings.sentence_similarity_threshold {
                let repeat_length = b.end - b.start;
                if repeat_length >= settings.sentence_min_repeat_length {
                    sentence_matches.push((
                        a.start, a.end, b.start, b.end, word_sim
                    ));
                }
            }
        }
    }

    // Determine whether sentence-level matches constitute real repetition
    let strong_pair = sentence_matches.iter().find(|m| m.4 > 0.9);
    let ngram_confirms = |m: &(usize, usize, usize, usize, f64)| {
        existing_sections.iter().any(|s| {
            s.repeat_start <= m.3 && s.repeat_end >= m.2 &&
            s.first_start <= m.0 && s.first_end >= m.1
        })
    };

    let sufficient_evidence = sentence_matches.len() >= 2 ||
        (strong_pair.is_some() && ngram_confirms(strong_pair.unwrap()));

    if !sufficient_evidence {
        return Vec::new();
    }

    sentence_matches.into_iter().map(|(a_start, a_end, b_start, b_end, similarity)| {
        RepeatedSection {
            first_start: a_start,
            first_end: a_end,
            repeat_start: b_start,
            repeat_end: b_end,
            similarity,
            source: SectionSource::Sentence,
        }
    }).collect()
}

/// Deduplicate text by removing repeated sections.
pub fn deduplicate_text(text: &str, settings: &DedupSettings) -> String {
    let result = check_repetition(text, settings);

    if !result.repeated || result.sections.is_empty() {
        return text.to_string();
    }

    // Sort sections by repeat start position
    let mut sorted = result.sections.clone();
    sorted.sort_by_key(|s| s.repeat_start);

    // Remove overlapping sections (keep the one with highest similarity)
    let mut non_overlapping = Vec::new();
    for section in sorted {
        let overlaps = non_overlapping.iter().any(|s: &RepeatedSection| {
            section.repeat_start < s.repeat_end && section.repeat_end > s.repeat_start
        });
        if !overlaps {
            non_overlapping.push(section);
        }
    }

    // Build result by removing repeated sections (process in reverse)
    let mut result_text = text.to_string();
    for section in non_overlapping.iter().rev() {
        let start = section.repeat_start;
        let end = section.repeat_end.min(result_text.len());
        if start < end && start < result_text.len() {
            result_text = format!("{}{}", &result_text[..start], &result_text[end..]);
        }
    }

    result_text
}

/// Check if text has excessive repetition ratio.
pub fn get_repetition_ratio(text: &str, settings: &DedupSettings) -> f64 {
    let result = check_repetition(text, settings);

    if !result.repeated || result.sections.is_empty() {
        return 0.0;
    }

        // Deduplicate overlapping sections before summing
    let mut sorted = result.sections.clone();
    sorted.sort_by_key(|s| s.repeat_start);
    
    let mut non_overlapping = Vec::new();
    for section in sorted {
        let overlaps = non_overlapping.iter().any(|s: &RepeatedSection| {
            section.repeat_start < s.repeat_end && section.repeat_end > s.repeat_start
        });
        if !overlaps {
            non_overlapping.push(section);
        }
    }
    
    let total_repeated: usize = non_overlapping.iter()
        .map(|s| s.repeat_end.saturating_sub(s.repeat_start))
        .sum();

    total_repeated as f64 / text.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_repetition_exact() {
        let text = "This is a repeated section of text that appears multiple times. \
                    This is a repeated section of text that appears multiple times. \
                    Some other unique content here to make the text longer.";
        let settings = DedupSettings::default();
        let result = check_repetition(text, &settings);
        assert!(result.repeated);
        assert!(!result.sections.is_empty());
        assert!(result.sections.iter().any(|s| s.source == SectionSource::Ngram));
    }

    #[test]
    fn test_check_repetition_disabled() {
        let text = "repeated repeated repeated";
        let settings = DedupSettings { enabled: false, ..Default::default() };
        let result = check_repetition(text, &settings);
        assert!(!result.repeated);
    }

    #[test]
    fn test_check_repetition_short() {
        let text = "short";
        let settings = DedupSettings::default();
        let result = check_repetition(text, &settings);
        assert!(!result.repeated);
    }

    #[test]
    fn test_deduplicate_text() {
        let text = "Unique start. Repeated section here. Repeated section here. Unique end.";
                let settings = DedupSettings { ngram_similarity_threshold: 0.7, ..Default::default() };
        let result = deduplicate_text(text, &settings);
        assert!(result.len() <= text.len());
        assert!(result.contains("Unique start"));
        assert!(result.contains("Unique end"));
    }

    #[test]
    fn test_get_repetition_ratio() {
        let text = "This repeated section appears multiple times in the text message. \
                    This repeated section appears multiple times in the text message. \
                    This repeated section appears multiple times in the text message. \
                    Some other unique content to vary the text a bit here.";
        let settings = DedupSettings {
            ngram_similarity_threshold: 0.8,
            ..Default::default()
        };
        let ratio = get_repetition_ratio(text, &settings);
        assert!(ratio > 0.0);
    }

    #[test]
    fn test_sentence_pass_detection() {
        // Reordered word-identical sentences (semantic repetition)
        let toks = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi";
        let text = format!("{}. {}. {}. {}.", 
            toks,
            "phi upsilon tau sigma rho pi omicron xi nu mu lambda kappa iota theta eta zeta epsilon delta gamma beta alpha",
            "gamma delta beta alpha epsilon zeta eta theta iota lambda kappa mu nu xi omicron pi rho sigma tau upsilon phi",
            "theta iota kappa lambda alpha beta gamma delta epsilon zeta eta mu nu xi omicron pi rho sigma tau upsilon phi"
        );
        let settings = DedupSettings {
            ngram_similarity_threshold: 0.95,
            sentence_similarity_threshold: 0.75,
            sentence_min_repeat_length: 30,
            ..Default::default()
        };
        let result = check_repetition(&text, &settings);
        assert!(result.repeated);
        assert!(result.sections.iter().any(|s| s.source == SectionSource::Sentence));
        // Should NOT have ngram sections for purely reordered text
        assert!(!result.sections.iter().any(|s| s.source == SectionSource::Ngram));
    }

    #[test]
    fn test_streaming_state() {
        let settings = DedupSettings::default();
        let mut state = StreamingState::new(settings);
        
        // Feed chunks
        state.feed_chunk("This is a test. ");
        state.feed_chunk("This is a test. ");
        state.feed_chunk("This is a test. ");
        
        assert!(state.content.contains("This is a test"));
        assert_eq!(state.repetition_count, 0); // Updated by feed_chunk
    }

    #[test]
    fn test_streaming_reset() {
        let settings = DedupSettings::default();
        let mut state = StreamingState::new(settings);
        
        state.feed_chunk("repeated repeated repeated");
        assert!(!state.content.is_empty());
        
        state = StreamingState::new(DedupSettings::default());
        assert!(state.content.is_empty());
    }
}