use dedup_wasm::{compute_ngram_hashes, NgramHasher};

fn main() {
    let text = "This is a repeated section of text that appears multiple times in the message. \
                This is a repeated section of text that appears multiple times in the message. \
                Some other unique content here to make the text longer and more realistic.";
    
    // The repeated part starts at 0 and at 81
    let s1 = "This is a repeated section of text that appears multiple times in the message. ";
    let window = s1.len();
    let second_pos = s1.len(); // 81
    
    println!("text len: {}", text.len());
    println!("window: {}", window);
    println!("second_pos: {}", second_pos);
    
    // Check if rolling hash with step=1 works
    let hashes = compute_ngram_hashes(text, window, 1);
    println!("hashes count: {}", hashes.len());
    println!("hash at 0: {}", hashes[0]);
    println!("hash at 81: {}", hashes[second_pos]);
    println!("match: {}", hashes[0] == hashes[second_pos]);
}
