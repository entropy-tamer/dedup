use dedup_wasm::tokenize;

fn main() {
    // Test 1: "a big ox and i"
    let tokens = tokenize("a big ox and i");
    println!("Test 1: {:?}", tokens);
    println!("  contains big: {}", tokens.contains("big"));
    println!("  contains ox: {}", tokens.contains("ox"));
    
    // Test 2: punctuation
    let tokens2 = tokenize("Hello, world! How are you?");
    println!("\nTest 2: {:?}", tokens2);
    println!("  contains hello: {}", tokens2.contains("hello"));
    println!("  contains world: {}", tokens2.contains("world"));
    println!("  contains how: {}", tokens2.contains("how"));
}
