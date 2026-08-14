use dedup_wasm::{check_repetition, DedupSettings};

fn main() {
    let text = "This is a repeated section of text that appears multiple times in the message. \
                This is a repeated section of text that appears multiple times in the message. \
                Some other unique content here to make the text longer and more realistic.";
    
    println!("Text length: {}", text.len());
    
    let settings = DedupSettings::default();
    let result = check_repetition(text, &settings);
    println!("repeated: {}", result.repeated);
    println!("sections: {:?}", result.sections);
    for s in &result.sections {
        println!("  source: {:?}, similarity: {}, range: {}-{} / {}-{}", s.source, s.similarity, s.first_start, s.first_end, s.repeat_start, s.repeat_end);
    }
}
