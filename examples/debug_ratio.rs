use dedup_wasm::{check_repetition, get_repetition_ratio, DedupSettings};

fn main() {
    let text = "This repeated section appears multiple times in the text message. \
                This repeated section appears multiple times in the text message. \
                This repeated section appears multiple times in the text message. \
                Some other unique content to vary the text a bit here.";
    
    println!("Text length: {}", text.len());
    
    let settings = DedupSettings {
        ngram_similarity_threshold: 0.8,
        ..Default::default()
    };
    
    let result = check_repetition(text, &settings);
    println!("repeated: {}", result.repeated);
    println!("sections: {:?}", result.sections);
    for s in &result.sections {
        println!("  source: {:?}, similarity: {}, range: {}-{} / {}-{}", s.source, s.similarity, s.first_start, s.first_end, s.repeat_start, s.repeat_end);
    }
    
    let ratio = get_repetition_ratio(text, &settings);
    println!("ratio: {}", ratio);
}
