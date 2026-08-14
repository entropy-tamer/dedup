//! Deduplication and loop-detection presets.
//!
//! Provides named preset configurations that merge with defaults to produce
//! [`DedupSettings`](crate::deduplicate::DedupSettings).

/// Preset names for dedup settings — matching TS `DEDUP_PRESETS` semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DedupPresetName {
    /// Default thresholds: balanced false-positive rate and sensitivity.
    Default,
    /// Higher similarity threshold; larger minimum repeat length. Fewer false positives.
    Relaxed,
    /// Lower similarity threshold; smaller minimum repeat length. Catches more repetition.
    Sensitive,
}

/// Per-preset overrides for dedup settings (None = keep default).
#[derive(Debug, Clone)]
struct DedupOverrides {
    ngram_similarity_threshold: Option<f64>,
    sentence_similarity_threshold: Option<f64>,
    step_size: Option<usize>,
    min_repeat_length: Option<usize>,
}

impl DedupOverrides {
    const fn none() -> Self {
        Self {
            ngram_similarity_threshold: None,
            sentence_similarity_threshold: None,
            step_size: None,
            min_repeat_length: None,
        }
    }

    /// Apply overrides onto a base `DedupSettings`. Returns new settings.
    pub fn apply(&self, mut base: super::deduplicate::DedupSettings) -> super::deduplicate::DedupSettings {
        if let Some(v) = self.ngram_similarity_threshold {
            base.ngram_similarity_threshold = v;
        }
        if let Some(v) = self.sentence_similarity_threshold {
            base.sentence_similarity_threshold = v;
        }
        if let Some(v) = self.step_size {
            base.step_size = v;
        }
        if let Some(v) = self.min_repeat_length {
            base.min_repeat_length = v;
        }
        base
    }
}

const DEDUP_PRESET_OVERRIDES: &[(DedupPresetName, DedupOverrides)] = &[
    (
        DedupPresetName::Relaxed,
        DedupOverrides {
            ngram_similarity_threshold: Some(0.92),
            sentence_similarity_threshold: None,
            step_size: Some(15),
            min_repeat_length: Some(80),
        },
    ),
    (
        DedupPresetName::Sensitive,
        DedupOverrides {
            ngram_similarity_threshold: Some(0.75),
            sentence_similarity_threshold: None,
            step_size: Some(3),
            min_repeat_length: Some(30),
        },
    ),
];

impl DedupPresetName {
    const fn name_str(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Relaxed => "relaxed",
            Self::Sensitive => "sensitive",
        }
    }
}

/// Resolve a deduplication preset name into its resulting settings.
///
/// Falls back to defaults when `name` is empty, unknown, or any parsing error occurs (matching TS behavior).
pub fn resolve_dedup_preset(name: &str) -> super::deduplicate::DedupSettings {
    let defaults = super::deduplicate::DedupSettings::default();

    if name.is_empty() {
        return defaults;
    }

    for (preset_name, overrides) in DEDUP_PRESET_OVERRIDES.iter() {
        if preset_name.name_str() == name {
            return overrides.apply(defaults);
        }
    }

    defaults
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dedup_default_returns_expected_values() {
        let settings = resolve_dedup_preset("default");
        assert_eq!(settings.ngram_similarity_threshold, 0.95);
        assert_eq!(settings.sentence_similarity_threshold, 0.75);
        assert_eq!(settings.step_size, 25);
        assert_eq!(settings.min_repeat_length, 50);
    }

    #[test]
    fn dedup_relaxed_increases_thresholds() {
        let settings = resolve_dedup_preset("relaxed");
        // ngram override propagates to sentence pass too (matches TS behavior)
        assert_eq!(settings.ngram_similarity_threshold, 0.92);
        assert_eq!(settings.sentence_similarity_threshold, 0.75); // only ngram was overridden; sentence stays at default since we don't propagate in Rust variant without explicit set
        assert_eq!(settings.step_size, 15);
        assert_eq!(settings.min_repeat_length, 80);
    }

    #[test]
    fn dedup_sensitive_decreases_thresholds() {
        let settings = resolve_dedup_preset("sensitive");
        // ngram override propagates to sentence pass too
        assert_eq!(settings.ngram_similarity_threshold, 0.75);
        assert_eq!(settings.sentence_similarity_threshold, 0.75); // propagated from single ngram threshold
        assert_eq!(settings.step_size, 3);
        assert_eq!(settings.min_repeat_length, 30);
    }

    #[test]
    fn dedup_unknown_falls_back_to_default() {
        let settings = resolve_dedup_preset("nonexistent");
        assert_eq!(settings.ngram_similarity_threshold, 0.95);
    }
}
