mod transformer;
mod math;
mod preprocessing;
mod classifier;

use preprocessing::tokenize_initial;

fn main() {
    let tests = vec![
        "Hello world",
        "Hello, world!",
        "This is a test.",
        "Rust: fast, safe; productive.",
        "The price is £5.99!",
        "Don't stop-believing!",
        "Hello... what???",
    ];

    for (i, test) in tests.iter().enumerate() {
        println!("==========================");
        println!("Test {}", i + 1);
        println!("Input: {}", test);

        let tokens = tokenize_initial(test.to_string());

        println!("Output:");
        for token in &tokens {
            println!("{:?}", token);
        }
        println!();
    }
}
