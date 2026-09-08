mod transformer;
mod math;
mod preprocessing;
mod classifier;

use preprocessing::tokenize_initial;
use preprocessing::Tokenizer;
use classifier::Classifier;
use nalgebra::DMatrix;

fn main() {
    test_emotion_classifier();
}



fn test_emotion_classifier() {
    let classes = vec!["happy", "sad", "angry"];

    let sentences = vec![
        "I am so happy today",
        "I feel really sad right now",
        "This makes me so angry",
    ];

    // Build vocab from the sentences we're about to encode
    let corpus_text = sentences.join(" ");
    let tokenizer = Tokenizer::new(corpus_text);

    let mut encoded: Vec<Vec<usize>> = sentences.iter()
        .map(|s| tokenizer.encode(s.to_string()))
        .collect();

    tokenizer.pad_tokens(&mut encoded); // pads all sequences to the same length

    let batch_size = encoded.len();
    let max_seq_length = encoded[0].len();

    let mut input_matrix = DMatrix::<usize>::zeros(batch_size, max_seq_length);
    for (row, seq) in encoded.iter().enumerate() {
        for (col, &id) in seq.iter().enumerate() {
            input_matrix[(row, col)] = id;
        }
    }

    let vocab_size = tokenizer.vocabulary.len();
    let num_classes = classes.len();
    let n_layers = 2;

    let classifier = Classifier::empty(
        num_classes, n_layers, vocab_size, batch_size, max_seq_length
    );

    let predictions = classifier.forward(input_matrix);

    for (i, sentence) in sentences.iter().enumerate() {
        println!("Sentence: {}", sentence);
        for (j, class) in classes.iter().enumerate() {
            println!("  {}: {:.4}", class, predictions[(i, j)]);
        }
        println!();
    }
}