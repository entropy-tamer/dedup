//! High-level deduplication logic ported from TypeScript deduplication.ts.
//!
//! Provides three public entry points plus a streaming state machine with time-based gating:
//! - `check_repetition` — full text analysis (n-gram pass + sentence pass)
//! - `deduplicate_text` — removes repeated sections, keeping first occurrence
//! - `get_repetition_ratio` — fraction of text that is repetitive content
//! - `StreamingState` — incremental state machine with warmup gate, min-content-length gate, and action determination

use crate::ngram::{compute_ngram_hashes, find_hash_matches};
use crate::sentence::split_sentences;
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
    /// Minimum number of new bytes that must accumulate before a streamed check
    /// re-runs the full scan. `0` disables the gate and re-checks on every chunk.
    ///
    /// `check_repetition` is O(buffer length), and `feed_chunk` runs it over the
    /// *whole accumulated* buffer. Re-checking on every streamed delta therefore
    /// costs O(n²) in message length: measured per-check latency is linear
    /// (~1.7 ms at 4 KB, ~24 ms at 48 KB), so 48 KB at a 12-byte delta is 4096
    /// checks and tens of seconds of blocking. Gating on new bytes divides that
    /// by roughly `min_check_interval_bytes / delta`.
    ///
    /// The trade is detection granularity: a repeat completing mid-interval is
    /// reported up to `min_check_interval_bytes` late. 512 is chosen to sit
    /// above `context_gap_max` (300) — the widest gap within which the detector
    /// can pair two spans at all — so the interval never straddles more than one
    /// pairing window. This bounds the cost; it does not make the scan
    /// incremental. That is a separate change, tracked in the todos note.
    pub min_check_interval_bytes: usize,
}

fn default_max_repetitions() -> usize {
    3
}
fn default_warmup_ms() -> u64 {
    3000
}

fn default_min_check_interval_bytes() -> usize {
    512
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
            min_check_interval_bytes: default_min_check_interval_bytes(),
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
    /// Byte position where the last successful check ran; used for content-delta gating.
    last_check_end: usize,
    /// Buffer length at the last full `check_repetition` run; used for the
    /// byte-interval gate so skipped chunks still count toward the next check.
    last_check_len: usize,
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
            last_check_end: 0,
            last_check_len: 0,
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
            last_check_end: 0,
            last_check_len: 0,
            repetition_count: 0,
        }
    }

    /// Feed a chunk and return the repetition check result for the accumulated content.
    ///
    /// Gating behavior:
    /// - **Warmup gate**: if `elapsed < 3s`, returns empty `RepetitionCheck` immediately.
    /// - **Content length gate**: skip checks when total content is below `min_content_length`.
    /// - **Chunk delta gate**: skip checks unless enough new characters have arrived since the last check (avoids rapid-fire re-checking on tiny chunks).
    /// - **Interval gate**: skip unless `min_check_interval_bytes` have accumulated since the last full check. This is the one that bounds cost — the three above only avoid degenerate cases.
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

        // Chunk delta gate — don't re-check unless enough new content has arrived.
        //
        // `check_repetition` below is O(accumulated length), so re-running it per
        // streamed delta is O(n^2) in message length. Gate on bytes accumulated
        // since the last actual check, not merely on "a delta arrived": the
        // latter is what made a 48 KB message take tens of seconds.
        let chunk_delta = after_len.saturating_sub(before_len);
        if before_len > 0 && chunk_delta == 0 {
            return RepetitionCheck {
                repeated: false,
                sections: Vec::new(),
            };
        }
        if self.settings.min_check_interval_bytes > 0 && before_len > 0 {
            let since_last_check = after_len.saturating_sub(self.last_check_len);
            if since_last_check < self.settings.min_check_interval_bytes {
                return RepetitionCheck {
                    repeated: false,
                    sections: Vec::new(),
                };
            }
        }

        // Update n-gram hashes for the full accumulated content.
        // No n-gram pre-pass here: `check_ngram_pass` recomputes its own hashes at
        // step=1 for exhaustive matching, so caching a subsampled table first was a
        // full O(n) scan of the accumulated buffer thrown away on every token.
        let result = check_repetition(&self.content, &self.settings);
        self.last_check_len = after_len;

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
        self.last_check_end = 0;
        self.last_check_len = 0;
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
    let hash_window_size = effective_window_size.clamp(30, 50);

    let hashes = compute_ngram_hashes(text, hash_window_size, hash_step_size);
    if hashes.len() < 2 {
        return Vec::new();
    }

    let matches = find_hash_matches(&hashes, settings.ngram_similarity_threshold);
    if matches.is_empty() {
        return Vec::new();
    }

    let mut sections: Vec<RepeatedSection> = Vec::new();

    // Verify the region we are about to *claim*, not just the seed.
    //
    // `hash_window_size` (30..50) is only a seed detector, while the section
    // emitted below is sized by `effective_window_size` (up to 100). Emitting
    // the wider extent on the strength of the narrower verification overstated
    // the repeat: `similarity` described the seed alone, while `repeat_end` —
    // and therefore the span `deduplicate_text` deletes — was taken on trust.
    // Try the widest extent first, fall back to the seed, so the emitted extent
    // is always one that actually passed.
    //
    // Loop-invariant, so hoisted out of the candidate loop.
    let widths: &[usize] = if effective_window_size == hash_window_size {
        &[effective_window_size]
    } else {
        &[effective_window_size, hash_window_size]
    };

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

        let mut verified: Option<(usize, usize, f64)> = None;
        for &len in widths {
            // Lazy slices, no upfront allocation.
            let a_end = (i + len).min(text.len());
            let b_end = (j + len).min(text.len());
            let a_text = &text[i..a_end];
            let b_text = &text[j..b_end];

            let word_sim = jaccard_similarity(&tokenize(a_text), &tokenize(b_text));
            let char_sim = char_ngram_similarity(a_text, b_text, 4);
            let similarity = word_sim.max(char_sim);

            if similarity >= settings.ngram_similarity_threshold {
                verified = Some((a_end, b_end, similarity));
                break;
            }
        }

        let Some((a_end, b_end, similarity)) = verified else {
            continue;
        };

        let a_start = i.min(text.len());
        let b_start = j.min(text.len());

        // Now meaningful: `b_end - b_start` is the span that *verified*, so this
        // knob finally measures what its name says.
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

    // Tokenise every sentence ONCE, up front.
    //
    // The pair loop runs each sentence through `tokenize` once per partner. On
    // degenerate input -- thousands of very short sentences, which is exactly
    // what a thought loop produces -- the gap window still admits hundreds of
    // thousands of pairs, so each sentence was tokenised ~60 times to no
    // effect. `tokenize` is pure and `sentences` is fixed for the duration of
    // this pass, so hoisting it is exactly equivalent, not an approximation.
    //
    // Tempting next step: tokenise lazily, only for sentences that reach a
    // comparison, on the theory that most sentences fall outside every gap.
    // Measured, that is 0.92x — SLOWER. The reasoning is wrong because `j`
    // walks every sentence in the buffer, so each one is tokenised as the
    // right-hand side regardless; only the `i` side is windowed, and that
    // saves nothing over caching. Eager stays. See W1e in the note.
    let tokens: Vec<TokenSet> = sentences.iter().map(|s| tokenize(&s.text)).collect();

    // Pair enumeration is bounded by `context_gap_max` in BYTES, not by the
    // sentence count. A fixed "last k sentences" cap would silently drop pairs
    // the gap test accepts whenever sentences are short (1-byte sentences span
    // far less than 300 bytes per k), so it would change verdicts.
    //
    // Instead: `gap = b.start - a.end` grows monotonically as `i` walks
    // backwards, because `split_sentences` yields ordered, non-overlapping
    // spans. So for a fixed `j` the pairs that survive the gap test form the
    // contiguous suffix `[lo, j)`. A single `lo` pointer advances monotonically
    // as `j` increases, turning the O(sentences^2) sweep into
    // O(sentences + pairs_within_gap) while skipping only pairs the gap test
    // would have rejected anyway. Output is byte-for-byte identical.
    let mut lo = 0usize;

    for j in 1..sentences.len() {
        let b = &sentences[j];
        let earliest = b.start.saturating_sub(settings.context_gap_max);

        while lo < j && sentences[lo].end < earliest {
            lo += 1;
        }

        for i in lo..j {
            let a = &sentences[i];

            // Skip pairs already covered by n-gram detection.
            if existing_sections.iter().any(|s| {
                s.repeat_start <= b.end
                    && s.repeat_end >= b.start
                    && s.first_start <= a.start
                    && s.first_end >= a.end
            }) {
                continue;
            }

            let word_sim = jaccard_similarity(&tokens[i], &tokens[j]);

            if word_sim >= settings.sentence_similarity_threshold
                && b.end - b.start >= settings.sentence_min_repeat_length
            {
                sentence_matches.push((a.start, a.end, b.start, b.end, word_sim));
            }
        }
    }

    // Require corroborating evidence: ≥2 similar pairs, OR 1 strong pair (>0.9) confirmed by n-gram pass.
    let strong_pair = sentence_matches.iter().find(|m| m.4 > 0.9);
    let ngram_confirms_strong = strong_pair.is_some_and(|&(_, a_start, _, b_end, _)| {
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
        let state = StreamingState::new(DedupSettings {
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

    /// Degenerate cycle: the reported "thought loop" shape.
    const LOOP: &str =
        "Let go.\n\nLet write.\n\nLet go.\n\nLet construct.\n\nLet go.\n\nLet issue\n\n";

    /// Feed `text` in `chunk`-sized pieces, returning how many times the full
    /// scan actually ran, the number of chunks fed, and the last scan's result.
    ///
    /// A scan is observable because it advances `last_check_len`; a gated-out
    /// chunk leaves it untouched. Capturing the result at that same moment
    /// matters: reading the return value of the *final* `feed_chunk` would
    /// report the empty result of a skipped call, not the last real scan.
    fn feed_counting_scans(
        text: &str,
        chunk: usize,
        settings: DedupSettings,
    ) -> (usize, usize, Option<RepetitionCheck>) {
        let mut state = StreamingState::new(DedupSettings {
            warmup_ms: 0,
            ..settings
        });
        let mut scans = 0usize;
        let mut last_scan: Option<RepetitionCheck> = None;
        let mut seen = state.last_check_len;
        let mut i = 0;
        while i + chunk <= text.len() {
            let r = state.feed_chunk(&text[i..i + chunk]);
            if state.last_check_len != seen {
                scans += 1;
                seen = state.last_check_len;
                last_scan = Some(r);
            }
            i += chunk;
        }
        let r = state.feed_chunk(&text[i..]);
        if state.last_check_len != seen {
            last_scan = Some(r);
        }
        (scans, text.len() / chunk, last_scan)
    }

    #[test]
    fn test_interval_gate_cuts_scan_count_on_a_long_stream() {
        let text = &LOOP.repeat(300); // 6.6 KB, the degenerate shape
        let (scans, chunks, _) = feed_counting_scans(
            text,
            12,
            DedupSettings {
                min_check_interval_bytes: 512,
                ..Default::default()
            },
        );

        assert!(scans > 0, "the gate must still permit checks, got {scans}");
        // ~6600 bytes / 512 = ~13 scans versus 550 chunks. Assert the order of
        // magnitude rather than an exact count: the point is that the scan count
        // tracks bytes, not chunks. Paired with the zero-interval control below.
        assert!(
            scans * 10 < chunks,
            "interval gate should scan far less often than every chunk: {scans} scans for {chunks} chunks"
        );
    }

    #[test]
    fn test_interval_zero_scans_every_chunk() {
        // The control for the test above. Without it, the gate could be "fixed"
        // by never scanning at all and that would still pass. Not exactly
        // `chunks`: the ~400-byte content-length gate suppresses the first ~34
        // chunks regardless of the interval, so this asserts the same 10x
        // contrast in the other direction.
        let text = &LOOP.repeat(300);
        let (scans, chunks, _) = feed_counting_scans(
            text,
            12,
            DedupSettings {
                min_check_interval_bytes: 0,
                ..Default::default()
            },
        );

        assert!(
            scans * 10 > chunks,
            "disabling the gate must scan on (essentially) every chunk: {scans} scans for {chunks} chunks"
        );
    }

    #[test]
    fn test_interval_gate_still_detects_the_loop() {
        // A gate that suppresses checks but also suppresses findings would pass
        // the scan-count test while defeating the detector. Assert detection survives.
        let text = &LOOP.repeat(300);
        let (scans, _, last) = feed_counting_scans(
            text,
            12,
            DedupSettings {
                min_check_interval_bytes: 512,
                ..Default::default()
            },
        );

        assert!(scans > 0, "expected at least one scan");
        let last = last.expect("a scan must have run");
        assert!(
            last.repeated,
            "gated streaming must still report the degenerate loop it scanned for"
        );
        assert!(
            !last.sections.is_empty(),
            "a positive result carrying no sections would steer nothing"
        );
    }
}
