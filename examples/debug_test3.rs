use dedup_wasm::{compute_ngram_hashes, find_hash_matches, compute_ngram_strings, check_repetition, DedupSettings};

fn main() {
    let text = "This is a repeated section of text that appears multiple times. \
                This is a repeated section of text that appears multiple times. \
                Some other unique content here to make the text longer.";
    
    println!("Text length: {}", text.len());
    
    // Use the original settings - window_size=100, step_size=25
    let settings = DedupSettings::default();
    println!("window_size: {}, step_size: {}", settings.window_size, settings.step_size);
    
    // Original code uses effective_window_size = min(window_size, half_text).max(20)
    // But with our 200 char text, half is 100, so effective_window_size = 100
    // But the issue is the text is only ~200 chars, so window=100 means only 2 windows!
    
    let hashes = compute_ngram_hashes(text, 50, 10);
    println!("hashes count (50/10): {}", hashes.len());
    
    let matches = find_hash_matches(&hashes, settings.ngram_similarity_threshold);
    println!("matches count (50/10): {}", matches.len());
    
    let ngram_strings = compute_ngram_strings(text, 50, 10);
    for (i, j) in &matches {
        println!("Match: {} vs {}", i, j);
        println!("  A: {:?}", ngram_strings[*i]);
        println!("  B: {:?}", ngram_strings[*j]);
    }
    
    // Test with smaller window
    let hashes2 = compute_ngram_hashes(text, 30, 5);
    println!("\nhashes count (30/5): {}", hashes2.len());
    let matches2 = find_hash_matches(&hashes2, settings.ngram_similarity_threshold);
    println!("matches count (30/5): {}", matches2.len());
    
    let ngram_strings2 = compute_ngram_strings(text, 30, 5);
    for (i, j) in &matches2 {
        println!("Match: {} vs {}", i, j);
        println!("  A: {:?}", ngram_strings2[*i]);
        println!("  B: {:?}", ngram_strings2[*j]);
    }
}
