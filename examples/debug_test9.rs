use dedup_wasm::{check_repetition, DedupSettings};

fn main() {
    let text = "This is a repeated section of text that appears multiple times in the message. \
                This is a repeated section of text that appears multiple times in the message. \
                Some other unique content here to make the text longer and more realistic.";
    
    let settings = DedupSettings::default();
    println!("min_repeat_length: {}", settings.min_repeat_length);
    println!("min_repeat_length * 2: {}", settings.min_repeat_length * 2);
    println!("text.len(): {}", text.len());
    println!("enabled: {}", settings.enabled);
    
    // Check what the half_text calculation gives
    let half_text = text.len() / 2;
    let effective_window_size = settings.window_size.min(half_text).max(20);
    let effective_step_size = settings.step_size.min(effective_window_size / 2).max(1);
    println!("half_text: {}", half_text);
    println!("effective_window_size: {}", effective_window_size);
    println!("effective_step_size: {}", effective_step_size);
    
    // Try with relaxed settings
    let settings2 = DedupSettings {
        min_repeat_length: 30,
        window_size: 50,
        step_size: 10,
        ..Default::default()
    };
    println!("\nWith relaxed settings:");
    let result = check_repetition(text, &settings2);
    println!("repeated: {}", result.repeated);
    println!("sections: {:?}", result.sections);
}
