//! Integration tests for dedup-wasm

use dedup_wasm::{
    check_repetition, deduplicate_text, get_repetition_ratio,
    DedupSettings, RepeatedSection, RepetitionCheck, SectionSource,
    StreamingDedup,
};

#[test]
fn test_check_repetition_exact_duplicate() {
    let text = "This is a repeated section of text that appears multiple times. \
                This is a repeated section of text that appears multiple times. \
                Some other unique content here to make the text longer and more realistic.";
    let settings = DedupSettings::default();
    let result = check_repetition(text, &settings);
    assert!(result.repeated);
    assert!(!result.sections.is_empty());
    assert!(result.sections.iter().any(|s| s.source == SectionSource::Ngram));
}

#[test]
fn test_check_repetition_semantic_rewording() {
    // Reordered word-identical sentences - high Jaccard, low char n-gram
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
fn test_check_repetition_no_false_positive() {
    let text = "This talks about apples and oranges and their nutritional value. \
                Now we discuss bananas and grapefruits and their different properties. \
                Finally, we consider pears and lemons and how they compare to the others. \
                We should also mention that all of these fruits have different vitamin content.";
    let settings = DedupSettings::default();
    let result = check_repetition(text, &settings);
    assert!(!result.repeated);
}

#[test]
fn test_check_repetition_r_docs_regression() {
    // Regression: "r docs" vs "documentation" should not be flagged
    let text = "When reading r docs or examples, prefer the typed helper. \
                When reading documentation or examples, prefer the typed helper. \
                Some unrelated content about the build pipeline and caching follows here.";
    let settings = DedupSettings::default();
    let result = check_repetition(text, &settings);
    assert!(!result.repeated);
    assert!(result.sections.is_empty());
}

#[test]
fn test_check_repetition_stop_words() {
    // Stop-word-heavy but content-distinct sentences
    let text = "Cats enthusiastically chase the red laser dot across the wooden floor. \
                Dogs calmly guard the blue tennis ball beside the metal gate. \
                Birds swiftly migrate the warm southern route above the quiet lake. \
                Fish slowly patrol the cold coral reef within the dark ocean.";
    let settings = DedupSettings::default();
    let result = check_repetition(text, &settings);
    assert!(!result.repeated);
}

#[test]
fn test_check_repetition_context_gap() {
    // Similar sentences separated by large gap should not be flagged
    let filler = "A solitary crane stood at the edge of the frozen marsh, its reflection \
                  fractured by the wind that moved across the ice like a thought across a \
                  still mind, indifferent to the small warmth of the fox curled in the reeds \
                  awaiting the return of a friend who had gone south before the first snow and \
                  would not return until the thaw, if then, when the rivers ran clear again.";
    let text = format!("The cache expires after the timeout interval and must be refreshed. \
                      {} \
                      The cache expires after the timeout interval and must be refreshed.", filler);
    let settings = DedupSettings::default();
    let result = check_repetition(&text, &settings);
    assert!(!result.repeated);
}

#[test]
fn test_deduplicate_text_removes_repeats() {
    let text = "Unique start. Repeated section here. Repeated section here. Unique end.";
            let settings = DedupSettings { ngram_similarity_threshold: 0.7, ..Default::default() };
    let result = deduplicate_text(text, &settings);
    assert!(result.len() <= text.len());
    assert!(result.contains("Unique start"));
    assert!(result.contains("Unique end"));
}

#[test]
fn test_deduplicate_text_preserves_first_occurrence() {
    let text = "Important point that needs to be stated clearly here. \
                Important point that needs to be stated clearly here. \
                Another thing that is different from the first part.";
            let settings = DedupSettings { ngram_similarity_threshold: 0.8, ..Default::default() };
    let result = deduplicate_text(text, &settings);
    assert!(result.contains("Important point"));
    // Should only appear once
    let count = result.matches("Important point").count();
    assert_eq!(count, 1);
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
fn test_get_repetition_ratio_unique() {
    let text = "All unique content here with no repeats whatsoever. \
                Every sentence is different and talks about unrelated topics. \
                This is a completely different section with no similarity to previous.";
    let settings = DedupSettings::default();
    let ratio = get_repetition_ratio(text, &settings);
    assert_eq!(ratio, 0.0);
}

#[test]
fn test_streaming_dedup_basic() {
    let settings = DedupSettings::default();
    let json = serde_json::to_string(&settings).unwrap();
    let mut dedup = StreamingDedup::new(&json).unwrap();
    
    let result = dedup.feed_chunk("hello world hello world").unwrap();
    let check: RepetitionCheck = serde_json::from_str(&result).unwrap();
    assert!(check.repeated);
}

#[test]
fn test_streaming_dedup_reset() {
    let settings = DedupSettings::default();
    let json = serde_json::to_string(&settings).unwrap();
    let mut dedup = StreamingDedup::new(&json).unwrap();
    
    dedup.feed_chunk("repeated repeated repeated").unwrap();
    assert!(dedup.get_repetition_count() >= 0);
    
    dedup.reset();
    assert_eq!(dedup.get_repetition_count(), 0);
}

#[test]
fn test_streaming_dedup_incremental() {
    let settings = DedupSettings {
        min_repeat_length: 10,
        window_size: 20,
        step_size: 5,
        ..Default::default()
    };
    let json = serde_json::to_string(&settings).unwrap();
    let mut dedup = StreamingDedup::new(&json).unwrap();
    
    // Feed first occurrence
    let r1 = dedup.feed_chunk("This is a unique sentence that is long enough. ").unwrap();
    let check1: RepetitionCheck = serde_json::from_str(&r1).unwrap();
    assert!(!check1.repeated);
    
    // Feed repetition
    let r2 = dedup.feed_chunk("This is a unique sentence that is long enough. ").unwrap();
    let check2: RepetitionCheck = serde_json::from_str(&r2).unwrap();
    assert!(check2.repeated);
}

#[test]
fn test_settings_min_repeat_length() {
    let text = "Short. Short. Short.";
    let settings = DedupSettings {
        min_repeat_length: 100, // Higher than any repeated section
        ..Default::default()
    };
    let result = check_repetition(text, &settings);
    assert!(!result.repeated);
}

#[test]
fn test_settings_sentence_min_repeat_length() {
    // Two reordered pairs, each ~57 chars, below default sentenceMinRepeatLength (80)
    let text = "the module loads the settings from the config file at startup. \
                at startup the config file from the settings loads the module. \
                the system caches the result from the previous request on boot. \
                on boot the request previous the from result the caches system the.";
    let settings = DedupSettings {
        sentence_min_repeat_length: 80,
        ..Default::default()
    };
    let result = check_repetition(text, &settings);
    assert!(!result.repeated);
    
    // Lower threshold should catch it
    let settings2 = DedupSettings {
        sentence_min_repeat_length: 30,
        ..Default::default()
    };
    let result2 = check_repetition(text, &settings2);
    assert!(result2.repeated);
}

// ── Scenario A + G regression tests (ported from loop-dedup-bug.ts) ─

/// Port of `loop-dedup-bug` Scenario A — cross-turn escalation at max_repeats.
#[test]
fn scenario_a_cross_turn_loop_detection_stops_at_max_repeats() {
    use dedup_wasm::loop_detect::{self, LoopCheckOutcome, LoopDetectionState};

    let settings = loop_detect::DEFAULT_LOOP_DETECTION_SETTINGS;
    let empty_state = LoopDetectionState {
        window: vec![], repeat_count: 0, corrective_injected: false,
        turn_count: 0, escalation_tier: 0,
    };

    // Same text every turn — triggers the text-similarity match path.
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
    let (outcome, state) = loop_detect::check_loop(&repeated_text, state, &settings);
    if let LoopCheckOutcome::StopSession { reason, final_tier } = outcome {
        assert_eq!(final_tier, 1);
        assert!(reason.contains("Loop detected"));
        assert!(!state.corrective_injected == false);
    } else {
        panic!("Expected StopSession on turn 4, got {:?}", outcome);
    }
}

/// Port of `loop-dedup-bug` Scenario A happy path — model breaks pattern after corrective.
#[test]
fn scenario_a_model_breaks_pattern_after_corrective() {
    use dedup_wasm::loop_detect::{self, LoopCheckOutcome, LoopDetectionState};

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

/// Port of `loop-dedup-bug` Scenario G — genuine verbatim repetition IS detected.
#[test]
fn scenario_g_verbatim_repetition_detected() {
    let repeated_block = "The configuration system loads settings from a layered set of sources, \
                          merging them in a strict priority order that determines the final resolved values.";
    let loop_text = format!("{}\n\n{}\n\n{}", repeated_block, repeated_block, repeated_block);

    let settings = DedupSettings::default();
    let result = check_repetition(&loop_text, &settings);

    assert!(result.repeated,
        "verbatim repetition must be detected but got no match");
    for s in &result.sections {
        assert_eq!(s.source, dedup_wasm::SectionSource::Ngram);
    }
}

/// Port of `loop-dedup-bug` Scenario G — reworded content ("r docs" vs "documentation") must NOT trip.
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