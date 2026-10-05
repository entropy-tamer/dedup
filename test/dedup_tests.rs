//! Integration tests for dedup-wasm

use dedup_wasm::{
    DedupSettings, RustRepetitionCheck, SectionSource, StreamingDedup, WasmRepetitionCheck,
    check_repetition, deduplicate_text, get_repetition_ratio,
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
    assert!(
        result
            .sections
            .iter()
            .any(|s| s.source == SectionSource::Ngram)
    );
}

#[test]
fn test_check_repetition_semantic_rewording() {
    // Reordered word-identical sentences - high Jaccard, low char n-gram
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
    // Should NOT have ngram sections for purely reordered text
    assert!(
        !result
            .sections
            .iter()
            .any(|s| s.source == SectionSource::Ngram)
    );
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
    // Exact sentences separated by a large gap: the n-gram pass still flags these.
    let filler = "A solitary bird stood at the edge of the frozen marsh. Its reflection \
                  broke apart whenever the wind passed over the ice, scattering the pale shape into fragments that slowly gathered themselves again. In the reeds nearby, a fox lay curled against the cold, preserving what little warmth the afternoon had left. It had been waiting there for days, though it no longer knew exactly what it was waiting for. A friend had gone south before the first snow. They had promised to return with the thaw, when the rivers ran clear and the marsh began to move again. The fox had believed them then. Now, beneath the grey winter sky, it watched the bird lift its wings and disappear beyond the frozen reeds.";
    let text = format!(
        "The cache expires after the timeout interval and must be refreshed. \
                      {} \
                      The cache expires after the timeout interval and must be refreshed.",
        filler
    );
    // The n-gram pass detects exact repeats regardless of surrounding context
    // (context_gap_max only gates the sentence-level pass), so this must flag.
    let settings = DedupSettings::default();
    let result = check_repetition(&text, &settings);
    assert!(result.repeated);
}

#[test]
fn test_deduplicate_text_removes_repeats() {
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
fn test_deduplicate_text_preserves_first_occurrence() {
    let text = "Important point that needs to be stated clearly here. \
                Important point that needs to be stated clearly here. \
                Another thing that is different from the first part.";
    let settings = DedupSettings {
        ngram_similarity_threshold: 0.8,
        ..Default::default()
    };
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
    let settings = DedupSettings {
        min_repeat_length: 2,
        warmup_ms: 0,
        ..Default::default()
    };
    let json = serde_json::to_string(&settings).unwrap();
    let mut dedup = StreamingDedup::new(&json).unwrap();

    let result = dedup.feed_chunk("The quick brown fox jumps over the lazy dog near the river bank every morning. The quick brown fox jumps over the lazy dog near the river bank every morning.").unwrap();
    let check: WasmRepetitionCheck = serde_json::from_str(&result).unwrap();
    assert!(check.repeated());
}

#[test]
fn test_streaming_dedup_reset() {
    let settings = DedupSettings {
        warmup_ms: 0,
        ..Default::default()
    };
    let json = serde_json::to_string(&settings).unwrap();
    let mut dedup = StreamingDedup::new(&json).unwrap();

    dedup.feed_chunk("repeated repeated repeated").unwrap();
    assert_eq!(dedup.get_repetition_count(), 0);

    dedup.reset();
    assert_eq!(dedup.get_repetition_count(), 0);
}

#[test]
fn test_streaming_dedup_incremental() {
    let settings = DedupSettings {
        min_repeat_length: 10,
        window_size: 20,
        step_size: 5,
        warmup_ms: 0,
        // This test is about whether a repeat spanning two `feed_chunk` calls
        // is detected at all, not about when scans are scheduled. With the
        // default 512-byte interval these two ~53-byte chunks never reach the
        // scan, so the gate is disabled here to keep the test focused. The gate
        // has its own coverage in `test_interval_gate_*`.
        min_check_interval_bytes: 0,
        ..Default::default()
    };
    let json = serde_json::to_string(&settings).unwrap();
    let mut dedup = StreamingDedup::new(&json).unwrap();

    // Feed first occurrence
    let r1 = dedup
        .feed_chunk("This is a unique sentence that is long enough. ")
        .unwrap();
    let check1: WasmRepetitionCheck = serde_json::from_str(&r1).unwrap();
    assert!(!check1.repeated());

    // Feed repetition
    let r2 = dedup
        .feed_chunk("This is a unique sentence that is long enough. ")
        .unwrap();
    let check2: WasmRepetitionCheck = serde_json::from_str(&r2).unwrap();
    assert!(check2.repeated());
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

// ── Scenario G regression tests (ported from loop-dedup-bug.ts) ─

/// Port of `loop-dedup-bug` Scenario G — genuine verbatim repetition IS detected.
#[test]
fn scenario_g_verbatim_repetition_detected() {
    let repeated_block = "The configuration system loads settings from a layered set of sources, \
                          merging them in a strict priority order that determines the final resolved values.";
    let loop_text = format!(
        "{}\n\n{}\n\n{}",
        repeated_block, repeated_block, repeated_block
    );

    let settings = DedupSettings::default();
    let result = check_repetition(&loop_text, &settings);

    assert!(
        result.repeated,
        "verbatim repetition must be detected but got no match"
    );
    // N-gram pass must corroborate; the sentence pass may add sections too.
    assert!(
        result
            .sections
            .iter()
            .any(|s| s.source == dedup_wasm::SectionSource::Ngram)
    );
}

/// Port of `loop-dedup-bug` Scenario G — reworded content ("r docs" vs "documentation") must NOT trip.
#[test]
fn scenario_g_reworded_content_no_false_positive() {
    let text = "Let me check the r docs for the configuration schema and the expected field names. \
                The documentation explains the exact structure and the validation rules we need to follow. \
                We should consult the r docs once more to confirm the nested keys, then review the documentation for the type constraints.";

    let settings = DedupSettings::default();
    let result = check_repetition(text, &settings);

    assert!(
        !result.repeated,
        "reworded content must NOT be flagged as repetition (the 'r docs' regression)"
    );
}

/// A repeated region must never be reported as larger than it actually is.
///
/// The n-gram pass hashes small 30..50 byte *seed* windows to find candidates,
/// then used to size the emitted section by `effective_window_size` (up to 100)
/// without ever verifying that wider span. A 50-byte repeat was therefore
/// reported as 100 bytes of repetition with `similarity` taken from the seed
/// alone, and `deduplicate_text` deleted 50 bytes of text unique to the input.
///
/// The fixture must contain that hazard or it proves nothing: the two halves
/// share exactly their first 50 bytes, and their trailing 50 differ. On the old
/// code this reported `len=100` and destroyed `pqrs09876` outright.
#[test]
fn test_ngram_section_is_not_longer_than_the_repeat_that_verified() {
    fn filler(seed: &str, n: usize) -> String {
        seed.chars().cycle().take(n).collect()
    }

    let shared = filler("ABCDEFGHIJ", 50);
    let unique_a = filler("vwxyz12345", 50);
    let unique_b = filler("pqrs09876", 50);
    let text = format!("{}{}{}{}", shared, unique_a, shared, unique_b);

    assert_eq!(
        text.len(),
        200,
        "half_text=100 -> effective_window_size=100"
    );
    assert_eq!(&text[0..50], &text[100..150], "the seed really does repeat");
    assert_ne!(
        &text[50..100],
        &text[150..200],
        "the tails really do differ — the hazard lives here"
    );

    let result = check_repetition(&text, &DedupSettings::default());

    let ngram: Vec<_> = result
        .sections
        .iter()
        .filter(|s| s.source == SectionSource::Ngram)
        .collect();
    assert!(
        !ngram.is_empty(),
        "fixture is broken: it no longer detects the 50-byte repeat"
    );

    for s in &ngram {
        let claimed = s.repeat_end - s.repeat_start;
        assert!(
            claimed <= 50,
            "claimed a {claimed}-byte repeat, but only 50 bytes are shared \
             (similarity reported {:.3})",
            s.similarity
        );
        assert_eq!(
            claimed,
            s.first_end - s.first_start,
            "the two spans of one section must be the same width"
        );
    }

    // The consequence that actually matters: unique content must survive.
    let deduped = deduplicate_text(&text, &DedupSettings::default());
    assert!(
        deduped.contains("pqrs09876"),
        "deduplication destroyed text unique to the input"
    );
    assert!(
        deduped.contains("vwxyz12345"),
        "deduplication destroyed the other unique tail"
    );
    assert_eq!(deduped.len(), 150, "only the shared 50 bytes should go");
}

/// `min_repeat_length` must measure a span that was actually verified.
///
/// With the default window (`window_size` 100, `hash_window_size` 50) a
/// threshold above the seed width can only be satisfied by a genuine
/// full-width match — so a 100-byte threshold must reject this fixture
/// outright rather than inheriting an unverified 100-byte extent.
#[test]
fn test_min_repeat_length_gates_verified_length() {
    fn filler(seed: &str, n: usize) -> String {
        seed.chars().cycle().take(n).collect()
    }

    let shared = filler("ABCDEFGHIJ", 50);
    let text = format!(
        "{}{}{}{}",
        shared,
        filler("vwxyz12345", 50),
        shared,
        filler("pqrs09876", 50)
    );

    let settings = DedupSettings {
        min_repeat_length: 100,
        ..DedupSettings::default()
    };
    let result = check_repetition(&text, &settings);

    let ngram: Vec<_> = result
        .sections
        .iter()
        .filter(|s| s.source == SectionSource::Ngram)
        .collect();
    assert!(
        ngram.is_empty(),
        "min_repeat_length=100 must reject a 50-byte verified repeat, \
         but {} ngram section(s) survived",
        ngram.len()
    );
}

/// Regression for D10: `find_hash_matches` iterated an `AHashMap`, and `ahash`
/// draws a fresh `RandomState` **per instance**, so two calls in the same process
/// visited buckets in different orders. `check_ngram_pass` breaks on the first
/// verified pair, so the set of sections it reported varied between calls.
///
/// The hazard only reproduces by calling repeatedly *within one process* — a
/// "run the binary twice" test would pass, because a fresh process also gets a
/// fresh seed. The assertion below compares spans, not just counts, so an
/// ordering bug cannot satisfy it by coincidence.
#[test]
fn test_check_repetition_is_deterministic_across_calls() {
    let unit = "the quick brown fox jumps over the lazy dog near the riverbank ";
    let text = format!("{}{}", unit.repeat(200), unit.to_uppercase());

    let settings = DedupSettings::default();
    let spans = |result: &RustRepetitionCheck| -> Vec<(usize, usize, usize, usize)> {
        let mut v: Vec<_> = result
            .sections
            .iter()
            .map(|s| (s.first_start, s.first_end, s.repeat_start, s.repeat_end))
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    };

    let first = check_repetition(&text, &settings);

    // Precondition: the fixture must actually produce sections. Without this the
    // equality assertions below pass vacuously on an empty result.
    assert!(
        !first.sections.is_empty(),
        "fixture must yield sections to be a determinism test, got 0"
    );
    assert!(
        first.repeated,
        "fixture must be flagged repeated, got repeated=false"
    );

    for call in 1..12 {
        let again = check_repetition(&text, &settings);
        assert_eq!(
            first.sections.len(),
            again.sections.len(),
            "section count drifted on call {call}: {} vs {}",
            first.sections.len(),
            again.sections.len()
        );
        assert_eq!(
            spans(&first),
            spans(&again),
            "section spans drifted on call {call}"
        );
        assert_eq!(
            first.repeated, again.repeated,
            "verdict drifted on call {call}"
        );
    }
}
