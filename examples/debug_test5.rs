use dedup_wasm::{compute_ngram_hashes, NgramHasher};

fn main() {
    let text = "This is a repeated section of text that appears multiple times. \
                This is a repeated section of text that appears multiple times. \
                Some other unique content here to make the text longer.";
    
    let s1 = "This is a repeated section of text that appears multiple times. ";
    let window = s1.len();
    println!("Window size: {}, text len: {}", window, text.len());
    
    let hasher = NgramHasher::new(window);
    
    // Direct hash at position 0
    let h0 = hasher.initial_hash(&text.as_bytes()[0..window]);
    println!("Hash at pos 0: {}", h0);
    
    // Direct hash at position where second occurrence starts
    let second_pos = s1.len() + 1; // +1 for the space
    println!("Second pos: {}", second_pos);
    let h1 = hasher.initial_hash(&text.as_bytes()[second_pos..second_pos+window]);
    println!("Hash at pos {}: {}", second_pos, h1);
    
    // Rolling hash from pos 0 to second_pos
    let mut rolling = h0;
    for i in 1..=second_pos {
        let leading = text.as_bytes()[i-1];
        let trailing = text.as_bytes()[i+window-1];
        rolling = hasher.roll_hash(rolling, leading, trailing);
        if i == second_pos {
            println!("Rolling hash at pos {}: {}", i, rolling);
        }
    }
    
    println!("Direct: {}, Rolling: {}", h1, rolling);
}
