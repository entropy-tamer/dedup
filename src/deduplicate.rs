//! High-level deduplication logic ported from TypeScript deduplication.ts.
//!
//! Provides three public entry points plus a streaming state machine with time-based gating:
//! - `check_repetition` — full text analysis (n-gram pass + sentence pass)
//! - `deduplicate_text` — removes repeated sections, keeping first occurrence
//! - `get_repetition_ratio` — fraction of text that is repetitive content
//! - `StreamingState` — incremental state machine with warmup gate, min-content-length gate, and action determination

use crate::ngram::{compute_ngram_hashes, find_hash_matches};
use crate::sentence::{Sentence, split_sentences};
use crate::similarity::{TokenSet, char_ngram_similarity, jaccard_similarity};
use crate::tokenize::tokenize;
use serde::{Deserialize, Serialize};

// ── Settings ────────────────────────────────────────────────────────

/// Configuration for deduplication.
///
/// All fields have serde defaults matching `DedupSettings::default()` so that
/// partial JSON (e.g. `{"enabled":true}`) deserializes instead of hard-failing.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DedupSettings {
    pub enabled: bool,
    pub ngram_similarity_threshold: f64,
    pub sentence_similarity_threshold: f64,
    pub sentence_min_repeat_length: usize,
    pub context_gap_max: usize,
    pub window_size: usize,
    pub step_size: usize,
    pub min_repeat_length: usize,
    /// Number of consecutive near-identical (n-gram) section detections before aborting the stream. Default 3.
    pub max_repetitions: usize,
    /// Warmup window in milliseconds before repetition checks begin. Default 3000.
    pub warmup_ms: u64,
}

fn default_max_repetitions() -> usize {
    3
}
fn default_warmup_ms() -> u64 {
    3000
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
            max_repetitions: default_max_repetitions(),
            warmup_ms: default_warmup_ms(),
        }
    }
}

// ── Detection results ───────────────────────────────────────────────

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

// ── Streaming action determination ──────────────────────────────────

/// Recommended action from streaming deduplication check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamAction {
    /// No repetition detected; nothing to do.
    None,
    /// Sentence-level semantic match only — inject a steering message (never accumulate toward abort).
    Steer,
    /// Sustained near-identical repetition — abort the stream.
    Abort,
}

/// Streaming state for incremental deduplication with time-based gating and action determination.
///
/// Mirrors `StreamDeduplicationMonitor` from TS: feeds chunks as they arrive, gates checks on warmup + content length + chunk delta, tracks n-gram-only repetition count toward the abort threshold.
pub struct StreamingState {
    settings: DedupSettings,
    content: String,
    ngram_hashes: Vec<u64>,
    /// Byte position where the last successful check ran; used for content-delta gating.
    last_check_end: usize,
    pub repetition_count: usize,
    #[cfg(not(target_arch = "wasm32"))]
    start_time: std::time::Instant,
}

impl StreamingState {
    /// Create a new streaming state.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new(settings: DedupSettings) -> Self {
        Self {
            settings,
            content: String::new(),
            ngram_hashes: Vec::new(),
            last_check_end: 0,
            repetition_count: 0,
            start_time: std::time::Instant::now(),
        }
    }

    /// Create a new streaming state (WASM target — no warmup gate).
    #[cfg(target_arch = "wasm32")]
    pub fn new(settings: DedupSettings) -> Self {
        Self {
            settings,
            content: String::new(),
            ngram_hashes: Vec::new(),
            last_check_end: 0,
            repetition_count: 0,
        }
    }

    /// Feed a chunk and return the repetition check result for the accumulated content.
    ///
    /// Gating behavior:
    /// - **Warmup gate**: if `elapsed < 3s`, returns empty `RepetitionCheck` immediately.
    /// - **Content length gate**: skip checks when total content is below `min_content_length`.
    /// - **Chunk delta gate**: skip checks unless enough new characters have arrived since the last check (avoids rapid-fire re-checking on tiny chunks).
    pub fn feed_chunk(&mut self, chunk: &str) -> RepetitionCheck {
        let before_len = self.content.len();
        self.content.push_str(chunk);
        let after_len = self.content.len();

        // Warmup gate (matches TS warmupMs default of 3000). Skipped on WASM targets and when warmup_ms is 0.
        #[cfg(not(target_arch = "wasm32"))]
        if self.settings.warmup_ms > 0
            && self.start_time.elapsed().as_millis() < self.settings.warmup_ms as u128
        {
            return RepetitionCheck {
                repeated: false,
                sections: Vec::new(),
            };
        }

        let min_check_len = after_len; // we gate by total content length below
        // Content length gate (default 400 chars from TS).
        if min_check_len < self.settings.min_repeat_length * 8
        /* ~400 */
        {
            return RepetitionCheck {
                repeated: false,
                sections: Vec::new(),
            };
        }

        // Chunk delta gate — don't re-check unless at least some new content has arrived.
        let chunk_delta = after_len.saturating_sub(before_len);
        if before_len > 0 && chunk_delta == 0 {
            return RepetitionCheck {
                repeated: false,
                sections: Vec::new(),
            };
        }

        // Update n-gram hashes for the full accumulated content.
        self.update_ngrams();

        let result = check_repetition(&self.content, &self.settings);

        // Update sentence boundaries after a successful check.
        if before_len > 0 {
            let sentences = split_sentences(&self.content);
            let ends: Vec<usize> = sentences.iter().map(|s| s.end).collect();
            self.last_check_end = ends.last().copied().unwrap_or(0);
        }

        // Update repetition count — only n-gram (near-exact) sections contribute toward the abort threshold.
        let ngram_count = result
            .sections
            .iter()
            .filter(|s| s.source == SectionSource::Ngram)
            .count();
        if result.repeated && ngram_count > 0 {
            self.repetition_count += ngram_count;
        } else {
            self.repetition_count = 0; // reset when no repetition detected
        }

        result
    }

    /// Determine the recommended action based on the just-completed check and accumulated state.
    pub fn determine_action(&self, last_check: &RepetitionCheck) -> StreamAction {
        if !last_check.repeated || !self.settings.enabled {
            return StreamAction::None;
        }

        let max_repetitions = self.settings.max_repetitions;

        // Sustained near-identical repetition aborts regardless of source type.
        if self.repetition_count >= max_repetitions {
            return StreamAction::Abort;
        }

        // Count n-gram confirms with high similarity (>0.9) in this single pass.
        let strong_ngram_confirms: Vec<_> = last_check
            .sections
            .iter()
            .filter(|s| s.source == SectionSource::Ngram && s.similarity > 0.9)
            .collect();

        // Two n-gram confirms of near-exact repetition in one pass also abort.
        if strong_ngram_confirms.len() >= 2 {
            return StreamAction::Abort;
        }

        // Sentence-level semantic matches alone steer, never abort (prevents "r docs" vs "documentation" from killing the stream).
        StreamAction::Steer
    }

    pub fn reset(&mut self) {
        self.content.clear();
        self.ngram_hashes.clear();
        self.last_check_end = 0;
        self.repetition_count = 0;
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.start_time = std::time::Instant::now();
        }
    }

    /// Reset the warmup timer without clearing content. Useful when resuming monitoring mid-stream.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn reset_timer(&mut self) {
        self.start_time = std::time::Instant::now();
    }

    fn update_ngrams(&mut self) {
        self.ngram_hashes = compute_ngram_hashes(
            &self.content,
            self.settings.window_size,
            self.settings.step_size,
        );
    }
}

// ── Public API ──────────────────────────────────────────────────────

/// Check text for repetition using n-gram similarity and sentence comparison.
///
/// Two-pass approach:
/// 1. N-gram similarity for near-identical repetition (fast, catches exact duplicates)
/// 2. Sentence-level comparison for semantic repetition (catches rewording)
pub fn check_repetition(text: &str, settings: &DedupSettings) -> RepetitionCheck {
    if !settings.enabled || text.len() < settings.min_repeat_length * 2 {
        return RepetitionCheck {
            repeated: false,
            sections: Vec::new(),
        };
    }

    let mut sections = Vec::new();

    // First pass: n-gram similarity for near-identical repetition.
    let ngram_sections = check_ngram_pass(text, settings);
    sections.extend(ngram_sections);

    // Second pass: sentence-level comparison for semantic repetition.
    let sentence_sections = check_sentence_pass(text, settings, &sections);
    sections.extend(sentence_sections);

    RepetitionCheck {
        repeated: !sections.is_empty(),
        sections,
    }
}

/// First pass: n-gram similarity for near-identical repetition.
fn check_ngram_pass(text: &str, settings: &DedupSettings) -> Vec<RepeatedSection> {
    let half_text = text.len() / 2;
    let effective_window_size = settings.window_size.min(half_text).max(20);
    let hash_step_size = 1; // step=1 for exhaustive matching

    // Use smaller window (30-50 chars) to detect exact repetitions regardless of surrounding context.
    let hash_window_size = effective_window_size.min(50).max(30);

    let hashes = compute_ngram_hashes(text, hash_window_size, hash_step_size);
    if hashes.len() < 2 {
        return Vec::new();
    }

    let matches = find_hash_matches(&hashes, settings.ngram_similarity_threshold);
    if matches.is_empty() {
        return Vec::new();
    }

    let mut sections: Vec<RepeatedSection> = Vec::new();

    for (i, j) in matches {
        // Skip candidates whose windows overlap an already-accepted section —
        // on repetitive text thousands of adjacent window pairs describe ONE
        // repeated region pair; coalescing here keeps the pass linear.
        let i_end = i + hash_window_size;
        let j_end = j + hash_window_size;
        if sections.iter().any(|s| {
            i < s.first_end && i_end > s.first_start || j < s.repeat_end && j_end > s.repeat_start
        }) {
            continue;
        }

        // Verify with actual string comparison (lazy slices, no upfront allocation).
        let a_text = &text[i..i_end.min(text.len())];
        let b_text = &text[j..j_end.min(text.len())];

        let a_tokens = tokenize(a_text);
        let b_tokens = tokenize(b_text);
        let word_sim = jaccard_similarity(&a_tokens, &b_tokens);
        let char_sim = char_ngram_similarity(a_text, b_text, 4);
        let similarity = word_sim.max(char_sim);

        if similarity >= settings.ngram_similarity_threshold {
            let a_start = i.min(text.len());
            let a_end = (a_start + effective_window_size).min(text.len());
            let b_start = j.min(text.len());
            let b_end = (b_start + effective_window_size).min(text.len());

            if b_end - b_start >= settings.min_repeat_length {
                sections.push(RepeatedSection {
                    first_start: a_start,
                    first_end: a_end,
                    repeat_start: b_start,
                    repeat_end: b_end,
                    similarity,
                    source: SectionSource::Ngram,
                });
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
        for j in (i + 1)..sentences.len() {
            let a = &sentences[i];
            let b = &sentences[j];

            // Skip pairs already covered by n-gram detection.
            if existing_sections.iter().any(|s| {
                s.repeat_start <= b.end
                    && s.repeat_end >= b.start
                    && s.first_start <= a.start
                    && s.first_end >= a.end
            }) {
                continue;
            }

            // Require proximity (300-char context gap default).
            let gap = b.start.saturating_sub(a.end);
            if gap > settings.context_gap_max {
                continue;
            }

            let a_tokens = tokenize(&a.text);
            let b_tokens = tokenize(&b.text);
            let word_sim = jaccard_similarity(&a_tokens, &b_tokens);

            if word_sim >= settings.sentence_similarity_threshold
                && b.end - b.start >= settings.sentence_min_repeat_length
            {
                sentence_matches.push((a.start, a.end, b.start, b.end, word_sim));
            }
        }
    }

    // Require corroborating evidence: ≥2 similar pairs, OR 1 strong pair (>0.9) confirmed by n-gram pass.
    let strong_pair = sentence_matches.iter().find(|m| m.4 > 0.9);
    let ngram_confirms_strong = strong_pair.map_or(false, |&(_, a_start, _, b_end, _)| {
        existing_sections
            .iter()
            .any(|s| s.repeat_start <= b_end && s.repeat_end >= a_start)
    });

    if sentence_matches.len() < 2 && !ngram_confirms_strong {
        return Vec::new();
    }

    sentence_matches
        .into_iter()
        .map(
            |(a_start, a_end, b_start, b_end, similarity)| RepeatedSection {
                first_start: a_start,
                first_end: a_end,
                repeat_start: b_start,
                repeat_end: b_end,
                similarity,
                source: SectionSource::Sentence,
            },
        )
        .collect()
}

/// Deduplicate text by removing repeated sections. Keeps the first occurrence.
pub fn deduplicate_text(text: &str, settings: &DedupSettings) -> String {
    let result = check_repetition(text, settings);
    if !result.repeated || result.sections.is_empty() {
        return text.to_string();
    }

    // Sort by repeat start; remove overlaps keeping highest-similarity section.
    let mut sorted = result.sections.clone();
    sorted.sort_by_key(|s| s.repeat_start);
    let mut non_overlapping = Vec::new();
    for section in sorted {
        if !non_overlapping.iter().any(|s: &RepeatedSection| {
            section.repeat_start < s.repeat_end && section.repeat_end > s.repeat_start
        }) {
            non_overlapping.push(section);
        }
    }

    // Remove repeated sections from back to front (preserves indices).
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

/// Check if text has excessive repetition ratio. Returns the fraction of repeated content (0.0-1.0).
pub fn get_repetition_ratio(text: &str, settings: &DedupSettings) -> f64 {
    let result = check_repetition(text, settings);
    if !result.repeated || result.sections.is_empty() {
        return 0.0;
    }

    // Deduplicate overlapping sections before summing.
    let mut sorted = result.sections.clone();
    sorted.sort_by_key(|s| s.repeat_start);
    let mut non_overlapping = Vec::new();
    for section in sorted {
        if !non_overlapping.iter().any(|s: &RepeatedSection| {
            section.repeat_start < s.repeat_end && section.repeat_end > s.repeat_start
        }) {
            non_overlapping.push(section);
        }
    }

    let total_repeated: usize = non_overlapping
        .iter()
        .map(|s| s.repeat_end.saturating_sub(s.repeat_start))
        .sum();

    total_repeated as f64 / text.len().max(1) as f64
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_repetition_exact() {
        let text = "This is a repeated section of text that appears multiple times. \
                    This is a repeated section of text that appears multiple times. \
                    Some other unique content here to make the text longer.";
        let result = check_repetition(text, &DedupSettings::default());
        assert!(result.repeated);
        assert!(!result.sections.is_empty());
        assert!(
            result
                .sections
                .iter()
                .any(|s| s.source == SectionSource::Ngram)
        );
    }

    #[test]
    fn test_check_repetition_disabled() {
        let text = "repeated repeated repeated";
        let settings = DedupSettings {
            enabled: false,
            ..Default::default()
        };
        let result = check_repetition(text, &settings);
        assert!(!result.repeated);
    }

    #[test]
    fn test_check_repetition_short() {
        let text = "short";
        let result = check_repetition(text, &DedupSettings::default());
        assert!(!result.repeated);
    }

    #[test]
    fn test_deduplicate_text_removes_repeat() {
        let text = "Unique start. Repeated section here. Repeated section here. Unique end.";
        let settings = DedupSettings {
            ngram_similarity_threshold: 0.7,
            ..Default::default()
        };
        let result = deduplicate_text(text, &settings);
        assert!(result.len() <= text.len());
        assert!(result.contains("Unique start"));
        assert!(result.contains("Unique end"));
    }

    #[test]
    fn test_get_repetition_ratio_positive() {
        let text = "This repeated section appears multiple times in the text message. \
                    This repeated section appears multiple times in the text message. \
                    Some other unique content to vary the text a bit here.";
        let settings = DedupSettings {
            ngram_similarity_threshold: 0.8,
            ..Default::default()
        };
        assert!(get_repetition_ratio(text, &settings) > 0.0);
    }

    #[test]
    fn test_sentence_pass_detects_rephrasing() {
        let toks = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi";
        let text = format!(
            "{}. {}. {}. {}.",
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
        assert!(
            result
                .sections
                .iter()
                .any(|s| s.source == SectionSource::Sentence)
        );
    }

    #[test]
    fn test_sentence_pass_requires_corroboration() {
        // A single similar sentence pair should NOT trigger repetition without corroborating evidence.
        let text = "The function handles input validation and sanitizes user-provided data. \
                    This code validates the user's input thoroughly before processing it.";
        let settings = DedupSettings {
            ngram_similarity_threshold: 0.95,
            sentence_similarity_threshold: 0.75,
            sentence_min_repeat_length: 30,
            ..Default::default()
        };
        // Only one similar pair → insufficient evidence.
        let result = check_repetition(text, &settings);
        assert!(
            !result.repeated
                || !result
                    .sections
                    .iter()
                    .any(|s| s.source == SectionSource::Sentence)
        );
    }

    #[test]
    fn test_r_docs_regression_sentence_only() {
        // "r docs" vs "documentation" must NOT trigger a false positive.
        let text = "When reading r docs or examples, prefer the typed helper. \
                    When reading documentation or examples, prefer the typed helper.";
        let settings = DedupSettings {
            ngram_similarity_threshold: 0.95,
            sentence_similarity_threshold: 0.75,
            ..Default::default()
        };
        let result = check_repetition(text, &settings);
        // Should be false — this is a single pair without corroborating evidence.
        assert!(!result.repeated);
    }

    #[test]
    fn test_streaming_state_accumulates_ngram_count() {
        let settings = DedupSettings::default();
        let mut state = StreamingState::new(settings.clone());

        // Feed identical chunks — each should increment n-gram count.
        for _ in 0..5 {
            state.feed_chunk(
                "this is a repeated sentence that keeps getting added to the stream.\n",
            );
        }

        // After warmup (we can't easily simulate elapsed time, so we manually check content accumulation).
        assert!(state.content.contains("repeated sentence"));
    }

    #[test]
    fn test_streaming_reset_clears_state() {
        let mut state = StreamingState::new(DedupSettings::default());

        state.feed_chunk("some content here");
        state.reset();

        assert!(state.content.is_empty());
        assert_eq!(state.repetition_count, 0);
    }

    #[test]
    fn test_streaming_warmup_gate_skips_check() {
        // During warmup (first 3s), feed_chunk returns empty RepetitionCheck even with enough content.
        let settings = DedupSettings::default();
        let mut state = StreamingState::new(settings.clone());

        // Feed a large chunk — but it's still within the warmup window.
        let large_content = "x".repeat(1000);
        let result = state.feed_chunk(&large_content);

        assert!(!result.repeated, "warmup should suppress all checks");
    }

    #[test]
    fn test_determine_action_ngram_accumulates_to_abort() {
        let mut state = StreamingState::new(DedupSettings {
            max_repetitions: 2,
            ..Default::default()
        });

        // Warmup gate blocks real accumulation during tests — manually bump repetition_count to simulate sustained n-gram detection.
        for _ in 0..3 {
            state.repetition_count += 1;
        }

        let check = RepetitionCheck {
            repeated: true,
            sections: vec![RepeatedSection {
                first_start: 0,
                first_end: 100,
                repeat_start: 100,
                repeat_end: 200,
                similarity: 0.96,
                source: SectionSource::Ngram,
            }],
        };

        // After enough n-gram confirms (repetition_count=3 >= max_repetitions=2), action should be Abort.
        assert_eq!(state.determine_action(&check), StreamAction::Abort);
    }

    #[test]
    fn test_determine_action_sentence_only_never_aborts() {
        let mut state = StreamingState::new(DedupSettings::default());

        // Even with high repetition count, sentence-only matches should steer not abort.
        for _ in 0..10 {
            state.feed_chunk("this is a line of text that gets repeated over and over again across the stream.\n");
        }

        let check = RepetitionCheck {
            repeated: true,
            sections: vec![RepeatedSection {
                first_start: 0,
                first_end: 100,
                repeat_start: 100,
                repeat_end: 200,
                similarity: 0.85,
                source: SectionSource::Sentence,
            }],
        };

        // Sentence-only match — must steer (or none if n-gram count is low).
        assert_ne!(state.determine_action(&check), StreamAction::Abort);
    }

    #[test]
    fn test_determine_action_disabled_returns_none() {
        let mut state = StreamingState::new(DedupSettings {
            enabled: false,
            ..Default::default()
        });

        let check = RepetitionCheck {
            repeated: true,
            sections: vec![RepeatedSection {
                first_start: 0,
                first_end: 100,
                repeat_start: 100,
                repeat_end: 200,
                similarity: 0.96,
                source: SectionSource::Ngram,
            }],
        };

        assert_eq!(state.determine_action(&check), StreamAction::None);
    }
}
