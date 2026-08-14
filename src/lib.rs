//! `dedup-wasm` — WASM binding for text deduplication.
//!
//! Provides high-performance n-gram, Jaccard, and sentence-level similarity
//! detection for streaming and batch text deduplication.
//!
//! ## Architecture
//!
//! - **N-gram computation**: Rolling hash (Rabin-Karp) for O(n) window hashing
//! - **Jaccard similarity**: Token-set intersection/union with ahash acceleration
//! - **Char n-gram similarity**: Fixed-width n-gram comparison
//! - **Sentence splitting**: Compiled regex with memchr optimization
//! - **Streaming state**: Incremental processing via `StreamingDedup` struct
//!
//! ## Exports
//!
//! - `dedup_check_repetition` — Batch repetition check
//! - `dedup_deduplicate_text` — Remove repeated sections
//! - `dedup_get_repetition_ratio` — Ratio of repeated content
//! - `StreamingDedup` — Incremental streaming deduplication
//! - `dedup_jaccard_similarity` — Token set Jaccard similarity
//! - `dedup_tokenize` — Tokenize text into word set

mod ngram;
mod similarity;
mod tokenize;
mod sentence;
mod deduplicate;

pub use deduplicate::{
    check_repetition, deduplicate_text, get_repetition_ratio,
    DedupSettings, RepeatedSection, RepetitionCheck, SectionSource,
};
pub use ngram::{compute_ngram_hashes, compute_ngram_strings, find_hash_matches, NgramHasher};
pub use similarity::{jaccard_similarity, char_ngram_similarity};
pub use tokenize::{tokenize, TokenSet};
pub use sentence::split_sentences;

use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Types for WASM exports ────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[wasm_bindgen]
pub struct WasmDedupSettings {
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmRepeatedSection {
    pub first_start: usize,
    pub first_end: usize,
    pub repeat_start: usize,
    pub repeat_end: usize,
    pub similarity: f64,
    pub source: String, // "ngram" | "sentence"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmRepetitionCheck {
    pub repeated: bool,
    pub sections: Vec<WasmRepeatedSection>,
}

// ── Streaming state ───────────────────────────────────────────────────

#[wasm_bindgen]
pub struct StreamingDedup {
    settings: DedupSettings,
    state: deduplicate::StreamingState,
}

#[wasm_bindgen]
impl StreamingDedup {
    #[wasm_bindgen(constructor)]
    pub fn new(settings_json: &str) -> Result<StreamingDedup, JsValue> {
        let wasm_settings: WasmDedupSettings = serde_json::from_str(settings_json)
            .map_err(|e| JsValue::from_str(&format!("Invalid settings JSON: {}", e)))?;
                let settings: DedupSettings = wasm_settings.into();
        let state = deduplicate::StreamingState::new(settings.clone());
        Ok(StreamingDedup { settings, state })
    }

    /// Feed a chunk of text and get immediate repetition check result.
    pub fn feed_chunk(&mut self, chunk: &str) -> Result<String, JsValue> {
        let result = self.state.feed_chunk(chunk);
        let wasm_result: WasmRepetitionCheck = result.into();
        serde_json::to_string(&wasm_result)
            .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
    }

    /// Reset the streaming state.
    pub fn reset(&mut self) {
        self.state = deduplicate::StreamingState::new(self.settings.clone());
    }

    /// Get current repetition count.
    pub fn get_repetition_count(&self) -> u32 {
        self.state.repetition_count as u32
    }
}

// ── Batch API ─────────────────────────────────────────────────────────

#[wasm_bindgen]
pub fn dedup_check_repetition(text: &str, settings_json: &str) -> Result<String, JsValue> {
    let wasm_settings: WasmDedupSettings = serde_json::from_str(settings_json)
        .map_err(|e| JsValue::from_str(&format!("Invalid settings JSON: {}", e)))?;
    let settings: DedupSettings = wasm_settings.into();
    let result = check_repetition(text, &settings);
    let wasm_result: WasmRepetitionCheck = result.into();
    serde_json::to_string(&wasm_result)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

#[wasm_bindgen]
pub fn dedup_deduplicate_text(text: &str, settings_json: &str) -> Result<String, JsValue> {
    let wasm_settings: WasmDedupSettings = serde_json::from_str(settings_json)
        .map_err(|e| JsValue::from_str(&format!("Invalid settings JSON: {}", e)))?;
    let settings: DedupSettings = wasm_settings.into();
    let result = deduplicate_text(text, &settings);
    Ok(result)
}

#[wasm_bindgen]
pub fn dedup_get_repetition_ratio(text: &str, settings_json: &str) -> Result<f64, JsValue> {
    let wasm_settings: WasmDedupSettings = serde_json::from_str(settings_json)
        .map_err(|e| JsValue::from_str(&format!("Invalid settings JSON: {}", e)))?;
    let settings: DedupSettings = wasm_settings.into();
    Ok(get_repetition_ratio(text, &settings))
}

// ── Utility exports for loop-detection ────────────────────────────────

#[wasm_bindgen]
pub fn dedup_jaccard_similarity(tokens_a_json: &str, tokens_b_json: &str) -> Result<f64, JsValue> {
    let tokens_a: Vec<String> = serde_json::from_str(tokens_a_json)
        .map_err(|e| JsValue::from_str(&format!("Invalid tokens_a JSON: {}", e)))?;
    let tokens_b: Vec<String> = serde_json::from_str(tokens_b_json)
        .map_err(|e| JsValue::from_str(&format!("Invalid tokens_b JSON: {}", e)))?;

    let set_a: TokenSet = tokens_a.into_iter().collect();
    let set_b: TokenSet = tokens_b.into_iter().collect();
    Ok(jaccard_similarity(&set_a, &set_b))
}

#[wasm_bindgen]
pub fn dedup_tokenize(text: &str) -> Result<String, JsValue> {
    let tokens = tokenize(text);
    let vec: Vec<String> = tokens.into_iter().collect();
    serde_json::to_string(&vec)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

// ── Conversion implementations ────────────────────────────────────────

impl From<WasmDedupSettings> for DedupSettings {
    fn from(w: WasmDedupSettings) -> Self {
        DedupSettings {
            enabled: w.enabled,
            ngram_similarity_threshold: w.ngram_similarity_threshold,
            sentence_similarity_threshold: w.sentence_similarity_threshold,
            sentence_min_repeat_length: w.sentence_min_repeat_length,
            context_gap_max: w.context_gap_max,
            window_size: w.window_size,
            step_size: w.step_size,
            min_repeat_length: w.min_repeat_length,
            max_repeats: w.max_repeats,
        }
    }
}

impl From<RepeatedSection> for WasmRepeatedSection {
    fn from(s: RepeatedSection) -> Self {
        WasmRepeatedSection {
            first_start: s.first_start,
            first_end: s.first_end,
            repeat_start: s.repeat_start,
            repeat_end: s.repeat_end,
            similarity: s.similarity,
            source: match s.source {
                SectionSource::Ngram => "ngram".to_string(),
                SectionSource::Sentence => "sentence".to_string(),
            },
        }
    }
}

impl From<RepetitionCheck> for WasmRepetitionCheck {
    fn from(r: RepetitionCheck) -> Self {
        WasmRepetitionCheck {
            repeated: r.repeated,
            sections: r.sections.into_iter().map(Into::into).collect(),
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wasm_settings_conversion() {
        let wasm = WasmDedupSettings {
            enabled: true,
            ngram_similarity_threshold: 0.95,
            sentence_similarity_threshold: 0.75,
            sentence_min_repeat_length: 80,
            context_gap_max: 300,
            window_size: 100,
            step_size: 25,
            min_repeat_length: 50,
            max_repeats: 3,
        };
        let settings: DedupSettings = wasm.into();
        assert_eq!(settings.enabled, true);
        assert_eq!(settings.ngram_similarity_threshold, 0.95);
    }

    #[test]
    fn test_streaming_dedup_creation() {
        let settings = WasmDedupSettings {
            enabled: true,
            ngram_similarity_threshold: 0.95,
            sentence_similarity_threshold: 0.75,
            sentence_min_repeat_length: 80,
            context_gap_max: 300,
            window_size: 100,
            step_size: 25,
            min_repeat_length: 50,
            max_repeats: 3,
        };
        let json = serde_json::to_string(&settings).unwrap();
        let mut dedup = StreamingDedup::new(&json).unwrap();
        let result = dedup.feed_chunk("hello world hello world").unwrap();
        assert!(result.contains("repeated"));
    }
}