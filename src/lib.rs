//! `dedup-wasm` — WASM binding for text deduplication.
//!
//! Provides high-performance n-gram, Jaccard, and sentence-level similarity
//! detection for streaming and batch text deduplication. Also exposes presets,
//! cross-turn loop detection (behind a feature flag), and a streaming state machine with
//! time-based gating and action determination.

mod ngram;
mod similarity;
mod tokenize;
mod sentence;
mod deduplicate;
pub mod presets;

pub mod loop_detect;

// ── Re-exports from sub-modules (for use by TS wrapper) ─────────────

pub use deduplicate::{
    check_repetition, deduplicate_text, get_repetition_ratio,
    DedupSettings, RepeatedSection as RustRepeatedSection, RepetitionCheck as RustRepetitionCheck, SectionSource,
    StreamAction, StreamingState,
};
pub use ngram::{compute_ngram_hashes, compute_ngram_strings, find_hash_matches, NgramHasher};
pub use similarity::{jaccard_similarity, char_ngram_similarity};
pub use tokenize::{tokenize, TokenSet};
pub use sentence::split_sentences;

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

// ── WASM-exported types ─────────────────────────────────────────────

#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmDedupSettings {
    pub enabled: bool,
    pub ngram_similarity_threshold: f64,
    pub sentence_similarity_threshold: f64,
    pub sentence_min_repeat_length: usize,
    pub context_gap_max: usize,
    pub window_size: usize,
    pub step_size: usize,
    pub min_repeat_length: usize,
    pub max_repetitions: usize,
}

#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmRepeatedSection {
    first_start: usize,
    first_end: usize,
    repeat_start: usize,
    repeat_end: usize,
    similarity: f64,
    source: String,
}

#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmRepetitionCheck {
    repeated: bool,
    sections: Vec<WasmRepeatedSection>,
}

/// Serialized loop detection settings for WASM export.
#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmLoopSettings {
    pub enabled: bool,
    pub window_size: u8,
    pub max_repeats: u8,
    pub escalation_interval: u8,
    pub text_similarity_threshold: f64,
    pub thought_loop_threshold: f64,
}

/// Serialized turn signature for cross-turn matching.
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct WasmTurnSignature {
    tool_calls: String,
    text_hash: String, // hex-encoded SHA-256
    text_content: String,
}

// ── Streaming state (WASM wrapper) ───────────────────────────────────

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

    /// Determine the recommended action (none/steer/abort) based on accumulated state.
    pub fn determine_action(&self, check_json: &str) -> Result<String, JsValue> {
        let check: RustRepetitionCheck = serde_json::from_str(check_json)
            .map_err(|e| JsValue::from_str(&format!("Invalid check JSON: {}", e)))?;
        let action = self.state.determine_action(&check);
        Ok(match action {
            StreamAction::None => "none".to_string(),
            StreamAction::Steer => "steer".to_string(),
            StreamAction::Abort => "abort".to_string(),
        })
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

// ── Preset resolution (WASM export) ──────────────────────────────────

#[wasm_bindgen]
pub fn dedup_resolve_preset(name: &str) -> Result<String, JsValue> {
    let settings = presets::resolve_dedup_preset(name);
    serde_json::to_string(&settings)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

// ── Cross-turn loop detection (WASM exports) — gated by feature flag ──

#[cfg(feature = "loop_detect")]
mod wasm_loop_exports {
    use super::*;
    use crate::loop_detect::{self, LoopDetectionSettings, TurnSignature};

    #[wasm_bindgen]
    pub fn dedup_compute_turn_signature(message_json: &str) -> Result<String, JsValue> {
        let sig: TurnSignature = serde_json::from_str(message_json)
            .map_err(|e| JsValue::from_str(&format!("Invalid message JSON: {}", e)))?;
        // Compute the signature fields and return as WASM-exported struct.
        Ok(WasmTurnSignature { tool_calls: sig.tool_calls, text_hash: hex_encode(&sig.text_hash), text_content: sig.text_content }.into())
    }

    #[wasm_bindgen]
    pub fn dedup_get_variant_suggestion(tool_name: &str) -> String {
        lock_detect::get_variant_suggestion(tool_name).unwrap_or_else(|| "Try a different tool or approach.".to_string())
    }
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
            max_repetitions: w.max_repetitions,
        }
    }
}

impl From<RustRepetitionCheck> for WasmRepetitionCheck {
    fn from(r: RustRepetitionCheck) -> Self {
        let sections = r.sections.into_iter()
            .map(|s| WasmRepeatedSection {
                first_start: s.first_start, first_end: s.first_end,
                repeat_start: s.repeat_start, repeat_end: s.repeat_end,
                similarity: s.similarity,
                source: match s.source { SectionSource::Ngram => "ngram".into(), _ => "sentence".into() },
            }).collect();

        WasmRepetitionCheck { repeated: r.repeated, sections }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wasm_settings_conversion() {
        let wasm = WasmDedupSettings {
            enabled: true, ngram_similarity_threshold: 0.95, sentence_similarity_threshold: 0.75,
            sentence_min_repeat_length: 80, context_gap_max: 300, window_size: 100,
            step_size: 25, min_repeat_length: 50, max_repetitions: 3,
        };
        let settings: DedupSettings = wasm.into();
        assert_eq!(settings.enabled, true);
        assert_eq!(settings.max_repetitions, 3);
    }

    #[test]
    fn test_streaming_dedup_creation() {
        let settings = WasmDedupSettings {
            enabled: true, ngram_similarity_threshold: 0.95, sentence_similarity_threshold: 0.75,
            sentence_min_repeat_length: 80, context_gap_max: 300, window_size: 100,
            step_size: 25, min_repeat_length: 50, max_repetitions: 3,
        };
        let json = serde_json::to_string(&settings).unwrap();
        let mut dedup = StreamingDedup::new(&json).unwrap();
        // Warmup gate should suppress checks.
        let result_str = dedup.feed_chunk("hello world hello world").unwrap();
        assert!(result_str.contains("\"repeated\":false"));
    }

    #[test]
    fn test_wasm_preset_resolution() {
        let default_json = dedup_resolve_preset("default").unwrap();
        let relaxed_json = dedup_resolve_preset("relaxed").unwrap();
        assert!(default_json.contains("\"ngram_similarity_threshold\":0.95"));
        assert!(relaxed_json.contains("\"ngram_similarity_threshold\":0.92"));
    }

    #[test]
    fn test_wasm_jaccard_roundtrip() {
        let tokens_a = serde_json::json!(["foo", "bar", "baz"]);
        let tokens_b = serde_json::json!(["foo", "qux"]);
        let sim = dedup_jaccard_similarity(&tokens_a.to_string(), &tokens_b.to_string()).unwrap();
        assert!((sim - 0.25).abs() < 1e-9); // Jaccard: intersection/union = 1/(3+2-1) = 1/4
    }

    #[test]
    fn test_wasm_tokenize_identity() {
        // Tokenize returns a JSON array — order may differ due to HashSet iteration,
        // so compare sorted token lists instead.
        let tokens_a: Vec<String> = serde_json::from_str(&dedup_tokenize("the cat sat on the mat").unwrap()).unwrap();
        let tokens_b: Vec<String> = serde_json::from_str(&dedup_tokenize("the cat sat on the mat").unwrap()).unwrap();
        let mut sa = tokens_a.clone();
        sa.sort();
        let mut sb = tokens_b.clone();
        sb.sort();
        assert_eq!(sa, sb); /* same tokens regardless of HashSet iteration order */
    }

    // ── Scenario A + G regression tests (ported from loop-dedup-bug.ts) ─

    #[test]
    fn scenario_a_cross_turn_loop_detection_stops_at_max_repeats() {
        use crate::loop_detect::{self, LoopCheckOutcome, LoopDetectionState};

        let settings = loop_detect::DEFAULT_LOOP_DETECTION_SETTINGS;
        let empty_state = LoopDetectionState {
            window: vec![], repeat_count: 0, corrective_injected: false,
            turn_count: 0, escalation_tier: 0,
        };
        let repeated_text = "echo test message content that will be hashed".to_string();

        // Turn 1: empty window -> NoMatch.
        let (outcome, state) = loop_detect::check_loop(&repeated_text, empty_state.clone(), &settings);
        assert!(matches!(outcome, LoopCheckOutcome::NoMatch));

        // Turn 2: first repeat detected. corrective injected with tier=1.
        let (outcome, state) = loop_detect::check_loop(&repeated_text, state, &settings);
        if let LoopCheckOutcome::EscalationMessage { message, tier, .. } = outcome {
            assert_eq!(tier, 1);
            assert!(message.contains("time(s) with the same arguments"));
        } else {
            panic!("Expected EscalationMessage on turn 2, got {:?}", outcome);
        }

        // Turn 3: second repeat. Constant backoff (interval=2): (2-1)%2 != 0 -> no injection.
        let (outcome, state) = loop_detect::check_loop(&repeated_text, state, &settings);
        assert!(matches!(outcome, LoopCheckOutcome::EscalationNotInjected { .. }));

        // Turn 4: third repeat reaches max_repeats=3 -> StopSession.
        let (outcome, _state) = loop_detect::check_loop(&repeated_text, state, &settings);
        if let LoopCheckOutcome::StopSession { reason, final_tier, .. } = outcome {
            assert!(reason.contains("Loop detected"), "StopSession reason missing 'Loop detected': {}", reason);
                        // Tier is 2 because the Constant backoff (interval=2) re-injected on repeat_count=3
            // before max_repeats kicked in to stop.
            assert_eq!(final_tier as u8, 2);
        } else {
            panic!("Expected StopSession on turn 4, got {:?}", outcome);
        }
    }

    #[test]
    fn scenario_a_model_breaks_pattern_after_corrective() {
        use crate::loop_detect::{self, LoopCheckOutcome, LoopDetectionState};

        let settings = loop_detect::DEFAULT_LOOP_DETECTION_SETTINGS;
        let empty_state = LoopDetectionState {
            window: vec![], repeat_count: 0, corrective_injected: false,
            turn_count: 0, escalation_tier: 0,
        };

        // Turn 1 + 2 same tool call (triggers corrective).
        let sig_a = "echo test message content that will be hashed".to_string();
        let (_o1, s1) = loop_detect::check_loop(&sig_a, empty_state.clone(), &settings);
        let (o2, s2) = loop_detect::check_loop(&sig_a, s1, &settings); // repeat detected
        assert!(matches!(o2, LoopCheckOutcome::EscalationMessage { .. }));

        // Turn 3: model breaks pattern — different tool call text.
        let sig_b = "read /some/other/file/path/to/something/unrelated".to_string();
        let (outcome, _) = loop_detect::check_loop(&sig_b, s2, &settings);

        match outcome {
            LoopCheckOutcome::NoMatch | LoopCheckOutcome::EscalationNotInjected { .. } => {},
            other => panic!("Expected NoMatch/NotInjected after pattern break, got {:?}", other),
        }
    }

    #[test]
    fn scenario_g_verbatim_repetition_detected() {
        let repeated_block = "The configuration system loads settings from a layered set of sources, \
                              merging them in a strict priority order that determines the final resolved values.";
        let loop_text = format!("{}\n\n{}\n\n{}", repeated_block, repeated_block, repeated_block);

        let settings = DedupSettings::default();
        let result = check_repetition(&loop_text, &settings);

        assert!(result.repeated,
            "verbatim repetition must be detected but got no match");
        // Verify the detected sections have high similarity (near-exact copy).
        for s in &result.sections {
            assert!((s.similarity - 1.0).abs() < 1e-6, 
                "Expected near-perfect similarity for verbatim repetition, got {:.4}", s.similarity);
        }
    }

    #[test]
    fn scenario_g_reworded_content_no_false_positive() {
        let text = "Let me check the r docs for the configuration schema and the expected field names. \
                    The documentation explains the exact structure and the validation rules we need to follow. \
                    We should consult the r docs once more to confirm the nested keys, then review the documentation for the type constraints.";

        let settings = DedupSettings::default();
        let result = check_repetition(text, &settings);

        assert!(!result.repeated,
            "reworded content must NOT be flagged as repetition (the 'r docs' regression)");
    }
}

// ── Helpers (private) ────────────────────────────────────────────────

#[cfg(feature = "loop_detect")]
fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}
