use dedup_wasm::{compute_ngram_hashes, NgramHasher, find_hash_matches};

fn main() {
    let text = "This is a repeated section of text that appears multiple times. \
                This is a repeated section of text that appears multiple times. \
                Some other unique content here to make the text longer.";
    
    let s1 = "This is a repeated section of text that appears multiple times. ";
    let window = s1.len();
    let second_pos = s1.len() + 1;
    
    // With step_size = 1, we should hit exact positions
    let hashes = compute_ngram_hashes(text, window, 1);
    println!("hashes count (step=1): {}", hashes.len());
    println!("  hash at 0: {}", hashes[0]);
    println!("  hash at 65: {}", hashes[second_pos]);
    
    let matches = find_hash_matches(&hashes, 0.0);
    println!("matches: {:?}", matches);
}
