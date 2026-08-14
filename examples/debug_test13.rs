use dedup_wasm::{compute_ngram_hashes, compute_ngram_strings, find_hash_matches, tokenize, jaccard_similarity, char_ngram_similarity, DedupSettings};

fn main() {
    let text = "This is a repeated section of text that appears multiple times in the message. \
                This is a repeated section of text that appears multiple times in the message. \
                Some other unique content here to make the text longer and more realistic.";
    
    let settings = DedupSettings::default();
    
    // With step=1, window=100 - find the exact positions where the windows match
    let hashes = compute_ngram_hashes(text, 100, 1);
    println!("hashes count (step=1, window=100): {}", hashes.len());
    
    let matches = find_hash_matches(&hashes, settings.ngram_similarity_threshold);
    println!("matches: {:?}", matches);
    
    if !matches.is_empty() {
        let ngram_strings = compute_ngram_strings(text, 100, 1);
        for (i, j) in &matches {
            let a_text = ngram_strings[*i];
            let b_text = ngram_strings[*j];
            println!("\nMatch {} vs {}:", i, j);
            println!("  A: {:?}", a_text);
            println!("  B: {:?}", b_text);
            
            let a_tokens = tokenize(a_text);
            let b_tokens = tokenize(b_text);
            let word_sim = jaccard_similarity(&a_tokens, &b_tokens);
            let char_sim = char_ngram_similarity(a_text, b_text, 4);
            let similarity = word_sim.max(char_sim);
            println!("  word_sim: {}, char_sim: {}, max: {}", word_sim, char_sim, similarity);
        }
    }
}
