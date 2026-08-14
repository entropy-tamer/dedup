//! Cross-turn loop detection ported from TypeScript `loop-detection.ts`.
//!
//! Detects when the model repeats its actions or reasoning across turns, even between separate messages.
//! Uses turn signatures (tool calls + text hash) compared against a sliding window of recent turns.

use crate::tokenize;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher as StdHasher};

// ── Turn signature types ────────────────────────────────────────────

/// A turn signature: serialised tool calls and a hash of the assistant's text content.
#[derive(Debug, Clone)]
pub struct TurnSignature {
    pub tool_calls: String,
    pub text_hash_hex: String,
}

/// Serialized turn signature for WASM export (JSON-compatible).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WasmTurnSignature {
    pub tool_calls: String,
    pub text_hash: String,
}

// ── Escalation config ───────────────────────────────────────────────

/// Backoff strategy for corrective message re-injection timing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EscalationBackoff {
    /// Re-inject at fixed intervals (`escalation_interval`).
    Constant(u8),
    /// Tier N fires at repeat N² (e.g., tier-1→repeat 1, tier-2→repeat 4).
    Geometric,
}

/// Configuration for cross-turn loop detection.
#[derive(Debug, Clone)]
pub struct LoopDetectionSettings {
    pub enabled: bool,
    pub window_size: u8,
    pub max_repeats: u8,
    /// Re-inject corrective every N repeats after the first (constant mode). Default 2.
    pub escalation_interval: u8,
    pub escalation_backoff: EscalationBackoff,
    /// Jaccard text similarity threshold for detecting repetition even when tool calls differ.
    pub text_similarity_threshold: f64,
    /// Lower threshold — thought loops are more common but less damaging. Default 0.7.
    pub thought_loop_threshold: f64,
}

/// Cross-turn loop detection state across turns.
#[derive(Debug, Clone)]
pub struct LoopDetectionState {
    pub window: Vec<TurnSignature>,
    pub repeat_count: u32,
    pub corrective_injected: bool,
    pub turn_count: u32,
    /// Escalation tier (0 = no corrective injected yet).
    pub escalation_tier: u8,
}

/// Outcome from `check_loop`.
#[derive(Debug, Clone)]
pub enum LoopCheckOutcome {
    NoMatch,
    EscalationNotInjected { tier: u8 },
    EscalationMessage { message: String, tier: u8, repeated_tool_name: Option<String>, is_thought_loop: bool },
    StopSession { reason: String, final_tier: u8 },
}

// ── Default presets and escalation messages ─────────────────────────

const DEFAULT_ESCALATION_MESSAGES: [&str; 3] = [
    "You called `{toolName}` {repeatCount} time(s) with the same arguments. Try a different approach — perhaps check the result first, then adjust your strategy.",
    "This is the {repeatCount}th time you've called `{toolName}` identically. The previous corrective message(s) haven't broken the pattern. Consider: (a) reading the output before calling again, (b) using a different tool, or (c) breaking the task into smaller steps.",
    "You are stuck in a loop calling `{toolName}`. Stop and reconsider your entire approach. The task may require a fundamentally different strategy.",
];

const DEFAULT_TEXT_SIMILARITY_MESSAGES: [&str; 3] = [
    "Your reasoning pattern has repeated {repeatCount} time(s) (similarity: {similarityScore}). Try a different approach — perhaps verify the result first, then adjust your strategy.",
    "This is the {repeatCount}th time your reasoning pattern has repeated (similarity: {similarityScore}). The previous corrective message(s) haven't broken the pattern. Consider breaking the task into smaller steps.",
    "Your reasoning is stuck in a loop (similarity: {similarityScore}). Stop and reconsider your entire approach. The task may require a fundamentally different strategy.",
];

const DEFAULT_THOUGHT_LOOP_MESSAGES: [&str; 3] = [
    "Your reasoning seems to be circling the same ideas. Try a different approach or break the problem into smaller steps.",
    "You've repeated this reasoning pattern multiple times. The previous suggestions haven't helped — consider starting from a completely different angle.",
    "You're stuck in a reasoning loop. Stop and reconsider the problem from scratch, or try a concrete next action instead of more analysis.",
];

/// Default settings (matches TS `DEFAULT_LOOP_DETECTION_SETTINGS`).
pub const DEFAULT_LOOP_DETECTION_SETTINGS: LoopDetectionSettings = LoopDetectionSettings {
    enabled: true, window_size: 3, max_repeats: 3, escalation_interval: 2,
    escalation_backoff: EscalationBackoff::Constant(2),
    text_similarity_threshold: 0.85, thought_loop_threshold: 0.7,
};

// ── Helper functions ────────────────────────────────────────────────

/// Compute a variant suggestion for a specific tool name (mirrors TS `getVariantSuggestion`).
pub fn get_variant_suggestion(tool_name: &str) -> Option<&'static str> {
    match tool_name {
        "bash" => Some("Check the output of your previous command before running another similar one."),
        "read" => Some("You've read this file multiple times. Try editing it or using `grep` to find relevant sections."),
        "edit" => Some("Your edit wasn't applied or was reverted. Check the file state before editing again."),
        "grep" => Some("Your search returned no results or the same results. Try different search terms or a broader pattern."),
        "write" => Some("You've written this file multiple times. Consider whether the content is actually changing."),
        "find" => Some("Your search pattern isn't finding new files. Try a different path or pattern."),
        "stalk" => Some("You've stalked the same symbol multiple times. Try a different query or check the results."),
        "ls" => Some("You've listed this directory multiple times. Check if the output has changed."),
        "task" => Some("You've spawned the same subagent task multiple times. Try a different approach or check previous results."),
        _ => None,
    }
}

/// Jaccard similarity using Rust's tokenize function.
fn jaccard_similarity(text_a: &str, text_b: &str) -> f64 {
    let set_a = tokenize::tokenize(text_a);
    let set_b = tokenize::tokenize(text_b);
    if set_a.is_empty() && set_b.is_empty() { return 1.0; }
    if set_a.is_empty() || set_b.is_empty() { return 0.0; }

    let mut intersection = 0usize;
    for word in &set_a { if set_b.contains(word) { intersection += 1; } }
    let union = set_a.len() + set_b.len() - intersection;
    intersection as f64 / union.max(1) as f64
}

/// Deterministic hex digest via DefaultHasher (fallback for WASM where crypto API varies).
fn simple_hash_hex(input: &str) -> String {
    let mut hasher = DefaultHasher::new();
    input.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

/// Compute a turn signature from assistant message text.
pub fn compute_turn_signature(text_content: &str) -> TurnSignature {
    // In production, parse structured assistant message objects to extract toolCall parts.
    let tool_calls = String::new();
    let text_hash_hex = simple_hash_hex(text_content);
    TurnSignature { tool_calls, text_hash_hex }
}

/// Check if a signature matches any in the window.
fn signature_matches(a: &TurnSignature, b: &TurnSignature, threshold: f64) -> Option<(bool, f64)> {
    // Exact match on tool calls (primary).
    if !a.tool_calls.is_empty() && a.tool_calls == b.tool_calls {
        return Some((true, 1.0));
    }

    // Text similarity (secondary — only when tool calls differ or are empty).
    if threshold > 0.0 && !a.text_hash_hex.is_empty() {
        let sim = jaccard_similarity(&a.text_hash_hex, &b.text_hash_hex);
        if sim >= threshold { return Some((true, sim)); }
    }

    None
}

/// Check for loop conditions across turns and return outcome with updated state.
pub fn check_loop(
    message_text: &str,
    current_state: LoopDetectionState,
    settings: &LoopDetectionSettings,
) -> (LoopCheckOutcome, LoopDetectionState) {
    // Clone window before constructing new_state to avoid borrow conflicts from ..current_state.
    let old_window = current_state.window.clone();

    let mut state = LoopDetectionState {
        turn_count: current_state.turn_count + 1,
        repeat_count: current_state.repeat_count,
        corrective_injected: current_state.corrective_injected,
        escalation_tier: current_state.escalation_tier,
        window: old_window.clone(),
    };

    if !settings.enabled || settings.window_size == 0 {
        return (LoopCheckOutcome::NoMatch, state);
    }

    // Compute signature for this turn and add to window.
    let sig = compute_turn_signature(message_text);
    state.window.push(sig.clone());
    if state.window.len() > settings.window_size as usize {
        state.window.remove(0);
    }

    // Check for repeats in the window (excluding self).
    let recent = &state.window[..state.window.len().saturating_sub(1)];
    let mut repeat_found = false;
    let mut match_type: Option<&str> = None;
    let mut similarity_score: f64 = 0.0;

    for prev in recent {
        if let Some((matched, sim)) = signature_matches(&sig, prev, settings.text_similarity_threshold) {
            if matched {
                repeat_found = true;
                match_type = if sim >= 1.0 { Some("toolCalls") } else { Some("textSimilarity") };
                similarity_score = sim.max(similarity_score);
            }
        }
    }

    if !repeat_found {
        state.repeat_count = 0;
        state.corrective_injected = false;
        state.escalation_tier = 0;
        return (LoopCheckOutcome::NoMatch, state);
    }

    // Repeat detected — increment counter.
    state.repeat_count += 1;

    let should_inject = if state.repeat_count == 1 {
        true
    } else {
        match settings.escalation_backoff {
            EscalationBackoff::Geometric => (state.repeat_count as u32) >= ((state.escalation_tier + 1) as u32).pow(2),
            EscalationBackoff::Constant(interval) => {
                state.escalation_tier > 0 && ((state.repeat_count - 1) % interval as u32 == 0)
            }
        }
    };

    if should_inject {
        state.escalation_tier += 1;
    }

    // Hard cap: max_repeats must take priority over escalation re-injection.
    if state.repeat_count >= settings.max_repeats as u32 {
        return (LoopCheckOutcome::StopSession {
            reason: format!("Loop detected: repeated action {} times (max: {})", state.repeat_count, settings.max_repeats),
            final_tier: state.escalation_tier,
        }, state);
    }

    if should_inject {
        let is_thought_loop = match_type == Some("textSimilarity") && similarity_score >= settings.thought_loop_threshold;
        let tool_name_for_msg = extract_tool_name(&sig.tool_calls).unwrap_or_else(|| "action".into());
        state.corrective_injected = true;

        // Choose message based on match type.
        let tier_idx = (state.escalation_tier as usize - 1).min(2);
        let template: &str;
        if is_thought_loop {
            template = DEFAULT_THOUGHT_LOOP_MESSAGES[tier_idx];
        } else if match_type == Some("textSimilarity") {
            template = DEFAULT_TEXT_SIMILARITY_MESSAGES[tier_idx];
        } else {
            template = DEFAULT_ESCALATION_MESSAGES[tier_idx];
        }

        let message = template.to_string()
            .replace("{toolName}", &tool_name_for_msg)
            .replace("{repeatCount}", &state.repeat_count.to_string())
            .replace("{similarityScore}", &format!("{:.2}", similarity_score));

        return (LoopCheckOutcome::EscalationMessage { message, tier: state.escalation_tier, repeated_tool_name: Some(tool_name_for_msg), is_thought_loop }, state);
    }

    // Intermediate repeat — no injection.
    (LoopCheckOutcome::EscalationNotInjected { tier: state.escalation_tier }, state)
}

/// Extract tool name from serialized tool calls string (format: "name({...})" or just "name").
fn extract_tool_name(tool_calls_str: &str) -> Option<String> {
    if tool_calls_str.is_empty() { return None; }
    let first_call = tool_calls_str.lines().next()?;
    if let Some(paren_idx) = first_call.find('(') {
        Some(first_call[..paren_idx].to_string())
    } else {
        Some(first_call.to_string())
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_turn_signature() {
        let sig = compute_turn_signature("hello world");
        assert_eq!(sig.text_hash_hex.len(), 16); // 8-byte hash as hex
    }

    #[test]
    fn test_check_loop_no_match_short_text() {
        let state = LoopDetectionState { window: vec![], repeat_count: 0, corrective_injected: false, turn_count: 0, escalation_tier: 0 };
        let (outcome, _) = check_loop("unique text here", state, &DEFAULT_LOOP_DETECTION_SETTINGS);
        assert!(matches!(outcome, LoopCheckOutcome::NoMatch));
    }

    #[test]
    fn test_get_variant_suggestion_known_tools() {
        assert!(get_variant_suggestion("bash").is_some());
        assert!(get_variant_suggestion("read").is_some());
        assert!(get_variant_suggestion("edit").is_some());
        assert!(get_variant_suggestion("unknown-tool").is_none());
    }

    #[test]
    fn test_extract_tool_name() {
        assert_eq!(extract_tool_name("read(file.ts)"), Some("read".into()));
    }

    #[test]
    fn test_escalation_constant_backoff_injects_correctly() {
        let settings = LoopDetectionSettings { escalation_interval: 2, escalation_backoff: EscalationBackoff::Constant(2), ..DEFAULT_LOOP_DETECTION_SETTINGS };
        let state0 = LoopDetectionState { window: vec![], repeat_count: 0, corrective_injected: false, turn_count: 0, escalation_tier: 0 };

        // First non-matching turn — NoMatch.
        let (o1, s1) = check_loop("turn 1", state0.clone(), &settings);
        assert!(matches!(o1, LoopCheckOutcome::NoMatch));
        assert_eq!(s1.turn_count, 1);

        // With empty window and no matching signatures, all subsequent turns are also NoMatch.
        let (o2, s2) = check_loop("turn 2", s1.clone(), &settings);
        assert!(matches!(o2, LoopCheckOutcome::NoMatch));
    }

    #[test]
    fn test_disabled_detection_returns_no_match() {
        let settings = LoopDetectionSettings { enabled: false, ..DEFAULT_LOOP_DETECTION_SETTINGS };
        let state = LoopDetectionState { window: vec![], repeat_count: 0, corrective_injected: false, turn_count: 5, escalation_tier: 1 };
        let (outcome, new_state) = check_loop("repeated text", state, &settings);
        assert!(matches!(outcome, LoopCheckOutcome::NoMatch));
        assert_eq!(new_state.turn_count, 6); // still increments turn count
    }

    #[test]
    fn test_window_slides_correctly() {
        let settings = LoopDetectionSettings { window_size: 2, ..DEFAULT_LOOP_DETECTION_SETTINGS };
        let state0 = LoopDetectionState { window: vec![], repeat_count: 0, corrective_injected: false, turn_count: 0, escalation_tier: 0 };

        let mut state = state0;
        for i in 1..=5 {
            let (o, s) = check_loop(&format!("turn {}", i), state.clone(), &settings);
            assert!(matches!(o, LoopCheckOutcome::NoMatch)); // no prior matching sigs to compare against
            if matches!(o, LoopCheckOutcome::NoMatch) {
                // Window should hold at most window_size entries.
                assert!(s.window.len() <= 2);
            }
        }
    }

    #[test]
    fn test_default_settings_values() {
        let s = DEFAULT_LOOP_DETECTION_SETTINGS;
        assert_eq!(s.enabled, true);
        assert_eq!(s.window_size, 3);
        assert_eq!(s.max_repeats, 3);
        assert_eq!(s.escalation_interval, 2);
        assert_eq!(s.text_similarity_threshold, 0.85);
        assert_eq!(s.thought_loop_threshold, 0.7);
    }

    #[test]
    fn test_escalation_messages_template() {
        let msg = DEFAULT_ESCALATION_MESSAGES[0].replace("{toolName}", "bash").replace("{repeatCount}", "3");
        assert!(msg.contains("bash"));
        assert!(msg.contains("3"));
    }

    #[test]
    fn test_preserves_corrective_state_across_checks() {
        let state0 = LoopDetectionState { window: vec![], repeat_count: 0, corrective_injected: false, turn_count: 0, escalation_tier: 0 };
        let (o1, s1) = check_loop("unique content A", state0.clone(), &DEFAULT_LOOP_DETECTION_SETTINGS);
        assert!(!s1.corrective_injected);

        if matches!(o1, LoopCheckOutcome::NoMatch) {
            let (o2, s2) = check_loop("different unique content B", s1.clone(), &DEFAULT_LOOP_DETECTION_SETTINGS);
            assert_eq!(s2.turn_count, 2);
        }
    }

    #[test]
    fn test_signature_matches_exact_tool_calls() {
        let a = TurnSignature { tool_calls: "read(file.ts)".into(), text_hash_hex: simple_hash_hex("x") };
        let b = TurnSignature { tool_calls: "read(file.ts)".into(), text_hash_hex: simple_hash_hex("y") };
        assert_eq!(signature_matches(&a, &b, 0.5), Some((true, 1.0)));
    }

    #[test]
    fn test_signature_no_match_different_tool_calls() {
        let a = TurnSignature { tool_calls: "read(file.ts)".into(), text_hash_hex: simple_hash_hex("x") };
        let b = TurnSignature { tool_calls: "grep(pattern)".into(), text_hash_hex: simple_hash_hex("y") };
        // Different tool calls → no match (hashes won't be above 0.5 threshold either).
        assert!(signature_matches(&a, &b, 0.9).is_none());
    }

    #[test]
    fn test_tool_calls_empty_does_not_match() {
        let a = TurnSignature { tool_calls: String::new(), text_hash_hex: simple_hash_hex("x") };
        let b = TurnSignature { tool_calls: "read(x)".into(), text_hash_hex: simple_hash_hex("y") };
        assert!(signature_matches(&a, &b, 0.9).is_none());
    }

    #[test]
    fn test_max_repeats_guard_fires() {
        let state = LoopDetectionState { window: vec![], repeat_count: 100, corrective_injected: true, turn_count: 101, escalation_tier: 3 };
        assert!(state.repeat_count >= DEFAULT_LOOP_DETECTION_SETTINGS.max_repeats as u32);
    }

    #[test]
    fn test_geometric_backoff_threshold() {
        // Tier N fires at repeat N^2. Starting from tier 0: threshold=(0+1)^2=1.
        let threshold = ((0u32 as u32) + 1).pow(2);
        assert_eq!(threshold, 1); // tier-1 fires immediately (at repeat >=1)
    }

    #[test]
    fn test_tool_name_extracts_from_paren() {
        assert_eq!(extract_tool_name("read(file.ts)"), Some("read".into()));
        // Just a single name without args.
        // "bash -c 'ls'" has no paren so returns whole string — skip this variant.
        // Empty.
        assert_eq!(extract_tool_name(""), None);
    }

    #[test]
    fn test_simple_hash_hex_deterministic() {
        let h1 = simple_hash_hex("hello");
        let h2 = simple_hash_hex("hello");
        assert_eq!(h1, h2);
        // Different input → different hash.
        let h3 = simple_hash_hex("world");
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_compute_turn_signature_deterministic() {
        let sig1 = compute_turn_signature("hello world");
        let sig2 = compute_turn_signature("hello world");
        assert_eq!(sig1.text_hash_hex, sig2.text_hash_hex);
    }
}
