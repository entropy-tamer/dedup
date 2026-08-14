use dedup_wasm::{compute_ngram_hashes, find_hash_matches, compute_ngram_strings, tokenize, jaccard_similarity, char_ngram_similarity, DedupSettings};

fn main() {
    let text = "This is a repeated section of text that appears multiple times in the message. \
                This is a repeated section of text that appears multiple times in the message. \
                Some other unique content here to make the text longer and more realistic.";
    
    let settings = DedupSettings::default();
    
    let half_text = text.len() / 2;
    let effective_window_size = settings.window_size.min(half_text).max(20);
    let effective_step_size = settings.step_size.min(effective_window_size / 2).max(1);
    
    let hash_window_size = effective_window_size.min(50).max(30);
    let hash_step_size = effective_step_size.min(hash_window_size / 2).max(1);
    
    println!("effective_window_size: {}, effective_step_size: {}", effective_window_size, effective_step_size);
    println!("hash_window_size: {}, hash_step_size: {}", hash_window_size, hash_step_size);
    
    // Compute n-gram hashes with smaller window
    let hashes = compute_ngram_hashes(text, hash_window_size, hash_step_size);
    println!("hashes count: {}", hashes.len());
    
    let matches = find_hash_matches(&hashes, settings.ngram_similarity_threshold);
    println!("matches: {:?}", matches);
    
    if !matches.is_empty() {
        let ngram_strings = compute_ngram_strings(text, hash_window_size, hash_step_size);
        println!("ngram_strings count: {}", ngram_strings.len());
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
            println!("  threshold: {}", settings.ngram_similarity_threshold);
            println!("  passes: {}", similarity >= settings.ngram_similarity_threshold);
        }
    }
}
