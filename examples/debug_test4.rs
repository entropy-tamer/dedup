use dedup_wasm::{compute_ngram_hashes, find_hash_matches, compute_ngram_strings, NgramHasher};

fn main() {
    let text = "This is a repeated section of text that appears multiple times. \
                This is a repeated section of text that appears multiple times. \
                Some other unique content here to make the text longer.";
    
    // Check if exact same substring produces same hash
    let s1 = "This is a repeated section of text that appears multiple times. ";
    let s2 = "This is a repeated section of text that appears multiple times. ";
    
    let hasher = NgramHasher::new(s1.len());
    let h1 = hasher.initial_hash(s1.as_bytes());
    let h2 = hasher.initial_hash(s2.as_bytes());
    println!("Same string hash: {} == {} ? {}", h1, h2, h1 == h2);
    
    // Test with window size = exact repeated length
    let window = s1.len();
    println!("Window size: {}", window);
    
    let hashes = compute_ngram_hashes(text, window, window);
    println!("hashes count (exact): {}", hashes.len());
    for (i, h) in hashes.iter().enumerate() {
        println!("  hash {}: {}", i, h);
    }
    
    let matches = find_hash_matches(&hashes, 0.0);
    println!("matches: {:?}", matches);
}
