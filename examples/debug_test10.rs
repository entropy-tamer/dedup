use dedup_wasm::{tokenize, jaccard_similarity, char_ngram_similarity, compute_ngram_strings, NgramHasher};

fn main() {
    let text = "This is a repeated section of text that appears multiple times in the message. \
                This is a repeated section of text that appears multiple times in the message. \
                Some other unique content here to make the text longer and more realistic.";
    
    // The two repeated sections
    let s1 = "This is a repeated section of text that appears multiple times in the message. ";
    let s2 = "This is a repeated section of text that appears multiple times in the message. ";
    
    println!("s1 len: {}", s1.len());
    println!("s2 len: {}", s2.len());
    println!("s1 == s2: {}", s1 == s2);
    
    // Tokenize
    let t1 = tokenize(s1);
    let t2 = tokenize(s2);
    println!("t1: {:?}", t1);
    println!("t2: {:?}", t2);
    
    // Jaccard
    let j = jaccard_similarity(&t1, &t2);
    println!("Jaccard: {}", j);
    
    // Char n-gram
    let c = char_ngram_similarity(s1, s2, 4);
    println!("Char n-gram: {}", c);
    
    // Max
    let max_sim = j.max(c);
    println!("Max similarity: {}", max_sim);
    println!("Threshold (0.95): {}", max_sim >= 0.95);
}
