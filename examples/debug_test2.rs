use dedup_wasm::{compute_ngram_hashes, find_hash_matches, compute_ngram_strings, check_repetition, DedupSettings};

fn main() {
    let text = "This is a repeated section of text that appears multiple times. \
                This is a repeated section of text that appears multiple times. \
                Some other unique content here to make the text longer.";
    
    println!("Text length: {}", text.len());
    
    let settings = DedupSettings::default();
    println!("window_size: {}, step_size: {}", settings.window_size, settings.step_size);
    
    let half_text = text.len() / 2;
    let effective_window_size = settings.window_size.min(half_text).max(20);
    let effective_step_size = settings.step_size.min(effective_window_size / 2).max(1);
    println!("effective_window_size: {}, effective_step_size: {}", effective_window_size, effective_step_size);
    
    let hashes = compute_ngram_hashes(text, effective_window_size, effective_step_size);
    println!("hashes count: {}", hashes.len());
    
    let matches = find_hash_matches(&hashes, settings.ngram_similarity_threshold);
    println!("matches count: {}", matches.len());
    
    if !matches.is_empty() {
        let ngram_strings = compute_ngram_strings(text, effective_window_size, effective_step_size);
        for (i, j) in &matches {
            println!("Match: {} vs {}", i, j);
            println!("  A: {:?}", ngram_strings[*i]);
            println!("  B: {:?}", ngram_strings[*j]);
        }
    }
    
    let result = check_repetition(text, &settings);
    println!("repeated: {}", result.repeated);
    println!("sections: {:?}", result.sections);
}
