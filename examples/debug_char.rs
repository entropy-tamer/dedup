use dedup_wasm::char_ngram_similarity;

fn main() {
    let a = "hello world";
    let b = "goodbye moon";
    
    let sim = char_ngram_similarity(a, b, 4);
    println!("sim: {}", sim);
    println!("sim > 0: {}", sim > 0.0);
}
