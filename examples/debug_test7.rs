use dedup_wasm::{compute_ngram_hashes, NgramHasher, find_hash_matches, compute_ngram_strings};

fn main() {
    let text = "This is a repeated section of text that appears multiple times. \
                This is a repeated section of text that appears multiple times. \
                Some other unique content here to make the text longer.";
    
    // With step_size=25, window=100
    let hashes = compute_ngram_hashes(text, 100, 25);
    println!("hashes count (step=25, window=100): {}", hashes.len());
    for (i, h) in hashes.iter().enumerate() {
        println!("  hash {} (pos {}): {}", i, i*25, h);
    }
    
    let ngram_strings = compute_ngram_strings(text, 100, 25);
    for (i, s) in ngram_strings.iter().enumerate() {
        println!("  string {} (pos {}): {:?}", i, i*25, s);
    }
    
    let matches = find_hash_matches(&hashes, 0.0);
    println!("matches: {:?}", matches);
}
