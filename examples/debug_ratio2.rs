use dedup_wasm::{check_repetition, get_repetition_ratio, DedupSettings};

fn main() {
    let text = "repeat repeat repeat repeat unique content here";
    
    println!("Text length: {}", text.len());
    
    let settings = DedupSettings::default();
    
    let result = check_repetition(text, &settings);
    println!("repeated: {}", result.repeated);
    println!("sections count: {}", result.sections.len());
    for s in &result.sections {
        println!("  source: {:?}, similarity: {}, range: {}-{} / {}-{}", s.source, s.similarity, s.first_start, s.first_end, s.repeat_start, s.repeat_end);
    }
    
    let ratio = get_repetition_ratio(text, &settings);
    println!("ratio: {}", ratio);
}
