#![allow(warnings)]
mod transformer;
mod math;
mod preprocessing;
mod classifier;

use classifier::Classifier;
use nalgebra::DMatrix;
use preprocessing::{tokenize_initial, Tokenizer};
use rand::{rng, RngExt};

// ---- hyperparameters ----
const K_EPOCHS: usize = 20;
const LR: f64 = 0.0025;
const N_LAYERS: usize = 3;
const TARGET_VOCAB: usize = 1000;
const SEQ_LEN: usize = 48;
const NEUTRAL_THRESHOLD: f64 = 0.30;
const TRAIN_FRAC: f64 = 0.8;

// 7 classes after pruning [disgust, anxiety, love]
const EMOTIONS: [&str; 7] = [
    "happy", "sad", "angry", "fear", "surprise", "excitement", "embarrassment",
];

// GoEmotions label id -> our class id (1-based). None = drop this sample.
const GO_EMOTIONS_TO_CLASS: [Option<usize>; 28] = [
    None,     // 0  admiration   -> (was love, pruned)
    Some(1),  // 1  amusement    -> happy
    Some(3),  // 2  anger        -> angry
    Some(3),  // 3  annoyance    -> angry
    None,     // 4  approval
    None,     // 5  caring       -> (was love, pruned)
    None,     // 6  confusion
    None,     // 7  curiosity
    None,     // 8  desire       -> (was love, pruned)
    Some(2),  // 9  disappointment -> sad
    None,     // 10 disapproval  -> (was disgust, pruned)
    None,     // 11 disgust      -> (pruned)
    Some(7),  // 12 embarrassment -> embarrassment
    Some(6),  // 13 excitement   -> excitement
    Some(4),  // 14 fear         -> fear
    None,     // 15 gratitude    -> (was love, pruned)
    Some(2),  // 16 grief        -> sad
    Some(1),  // 17 joy          -> happy
    None,     // 18 love         -> (pruned)
    None,     // 19 nervousness  -> (was anxiety, pruned)
    Some(1),  // 20 optimism     -> happy
    Some(1),  // 21 pride        -> happy
    None,     // 22 realization
    Some(1),  // 23 relief       -> happy
    Some(2),  // 24 remorse      -> sad
    Some(2),  // 25 sadness      -> sad
    Some(5),  // 26 surprise     -> surprise
    None,     // 27 neutral
];

fn load_go_emotions_tsv(path: &str, limit: usize) -> Vec<(String, usize)> {
    let text = std::fs::read_to_string(path).expect("could not read tsv");
    let mut out = Vec::new();
    for line in text.lines() {
        if line.trim().is_empty() { continue; }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 2 { continue; }
        let sentence = fields[0].trim();
        if sentence.is_empty() { continue; }
        let first = fields[1].split(',').next().unwrap_or("").trim();
        let label_id: usize = match first.parse() { Ok(v) => v, Err(_) => continue };
        if label_id >= GO_EMOTIONS_TO_CLASS.len() { continue; }
        let Some(class_1based) = GO_EMOTIONS_TO_CLASS[label_id] else { continue; };
        out.push((sentence.to_string(), class_1based - 1));
        if out.len() >= limit { break; }
    }
    out
}

fn encode_fixed(tok: &Tokenizer, s: &str) -> DMatrix<usize> {
    let mut ids = tok.encode(s.to_string());
    ids.truncate(SEQ_LEN);
    while ids.len() < SEQ_LEN { ids.push(tok.vocabulary["<PAD>"]); }
    DMatrix::from_row_slice(1, SEQ_LEN, &ids)
}

fn row_vec(m: &DMatrix<f64>) -> Vec<f64> { m.row(0).iter().cloned().collect() }

fn argmax(row: &[f64]) -> usize {
    let mut b = 0;
    for i in 0..row.len() { if row[i] > row[b] { b = i; } }
    b
}

fn print_confusion_matrix(title: &str, pred: &[usize], truth: &[usize]) {
    let n = EMOTIONS.len();
    let total = truth.len();
    if total == 0 { println!("\n[{}] no samples", title); return; }

    let mut counts = vec![vec![0usize; n]; n];
    for (t, p) in truth.iter().zip(pred.iter()) { counts[*t][*p] += 1; }

    let short: Vec<String> = EMOTIONS.iter().map(|e| e.chars().take(4).collect()).collect();

    println!("\n=== {} (n={}) ===", title, total);
    print!("{:<14}", "");
    for s in &short { print!("{:>7}", s); }
    println!("{:>9}", "recall");
    println!("{}", "-".repeat(14 + 7 * n + 9));

    for t in 0..n {
        let row_sum: usize = counts[t].iter().sum();
        print!("{:<14}", EMOTIONS[t]);
        for p in 0..n {
            if row_sum == 0 { print!("{:>7}", "-"); }
            else if counts[t][p] == 0 { print!("{:>7}", "."); }
            else { print!("{:>7.1}", 100.0 * counts[t][p] as f64 / row_sum as f64); }
        }
        let recall = if row_sum == 0 { 0.0 } else { 100.0 * counts[t][t] as f64 / row_sum as f64 };
        println!("{:>9.1}", recall);
    }
    println!("{}", "-".repeat(14 + 7 * n + 9));

    print!("{:<14}", "precision");
    for p in 0..n {
        let col_sum: usize = (0..n).map(|t| counts[t][p]).sum();
        let prec = if col_sum == 0 { 0.0 } else { 100.0 * counts[p][p] as f64 / col_sum as f64 };
        print!("{:>7.1}", prec);
    }
    println!("\n");

    println!("{:<14}{:>10}{:>10}{:>10}{:>10}", "class", "support", "prec", "recall", "f1");
    println!("{}", "-".repeat(54));
    let (mut mp, mut mr, mut mf, mut k) = (0.0, 0.0, 0.0, 0);
    for c in 0..n {
        let support: usize = counts[c].iter().sum();
        let col_sum: usize = (0..n).map(|t| counts[t][c]).sum();
        let prec = if col_sum == 0 { 0.0 } else { counts[c][c] as f64 / col_sum as f64 };
        let rec  = if support == 0 { 0.0 } else { counts[c][c] as f64 / support as f64 };
        let f1   = if prec + rec == 0.0 { 0.0 } else { 2.0 * prec * rec / (prec + rec) };
        println!("{:<14}{:>10}{:>10.1}{:>10.1}{:>10.1}",
                 EMOTIONS[c], support, 100.0 * prec, 100.0 * rec, 100.0 * f1);
        if support > 0 { mp += prec; mr += rec; mf += f1; k += 1; }
    }
    if k > 0 {
        println!("{}", "-".repeat(54));
        println!("{:<14}{:>10}{:>10.1}{:>10.1}{:>10.1}",
                 "macro", "", 100.0 * mp / k as f64, 100.0 * mr / k as f64, 100.0 * mf / k as f64);
    }
}

// Compute inverse-frequency class weights: w_c = N / (num_classes * count_c),
// then normalize so they average to 1.0. Clamp to avoid exploding on tiny classes.
fn class_weights(counts: &[usize]) -> Vec<f64> {
    let n = counts.len();
    let total: usize = counts.iter().sum();
    let mut w: Vec<f64> = counts.iter().map(|&c| {
        if c == 0 { 0.0 } else { total as f64 / (n as f64 * c as f64) }
    }).collect();
    let mean: f64 = w.iter().sum::<f64>() / n as f64;
    if mean > 0.0 { for v in w.iter_mut() { *v /= mean; } }
    // clamp: never weight a class more than 10x average
    for v in w.iter_mut() { if *v > 10.0 { *v = 10.0; } }
    w
}

fn main() {
    let mut data = load_go_emotions_tsv("dataset/sub_dataset.tsv", 5000);
    println!("loaded {} usable samples (from GoEmotions tsv)", data.len());

    let mut counts = vec![0usize; EMOTIONS.len()];
    for (_, y) in &data { counts[*y] += 1; }
    println!("class balance (full set):");
    for (i, name) in EMOTIONS.iter().enumerate() {
        println!("  {:<15} {}", name, counts[i]);
    }

    let mut r = rng();
    for i in (1..data.len()).rev() { let j = r.random_range(0..=i); data.swap(i, j); }
    let n_train = (data.len() as f64 * TRAIN_FRAC) as usize;
    let (train, test) = data.split_at(n_train);

    // recompute counts on the TRAIN split only — that's what weighting should use
    let mut train_counts = vec![0usize; EMOTIONS.len()];
    for (_, y) in train { train_counts[*y] += 1; }
    println!("train split class counts:");
    for (i, name) in EMOTIONS.iter().enumerate() {
        println!("  {:<15} {}", name, train_counts[i]);
    }
    let weights = class_weights(&train_counts);
    println!("train class weights:");
    for (i, name) in EMOTIONS.iter().enumerate() {
        println!("  {:<15} {:.3}", name, weights[i]);
    }

    let all_text: String = train.iter().map(|(s, _)| s.as_str()).collect::<Vec<_>>().join(" ");
    let mut tok = Tokenizer::new(all_text);
    let corpus: Vec<Vec<String>> = train.iter().flat_map(|(s, _)| tokenize_initial(s.clone())).collect();
    tok.bpe(corpus, TARGET_VOCAB);
    println!("vocab size = {}", tok.vocabulary.len());

    let mut model = Classifier::empty(EMOTIONS.len(), N_LAYERS, tok.vocabulary.len(), 1, SEQ_LEN);

    for epoch in 1..=K_EPOCHS {
        let mut total_loss = 0.0;
        let mut correct = 0;
        for (s, y) in train.iter() {
            let probs = model.forward(encode_fixed(&tok, s));
            // weight the loss by inverse class frequency
            let w = weights[*y];
            total_loss += -w * probs[(0, *y)].max(1e-12).ln();
            if argmax(&row_vec(&probs)) == *y { correct += 1; }
            model.backwards(*y, weights[*y], LR);   // scale the gradient too
        }
        // report weighted loss normalized by total weight so epochs are comparable
        let total_w: f64 = train.iter().map(|(_, y)| weights[*y]).sum();
        println!("epoch {:>2}/{} | wloss {:.4} | train acc {:.1}%", epoch, K_EPOCHS,
                 total_loss / total_w,
                 100.0 * correct as f64 / train.len() as f64);
    }

    // held-out evaluation
    let mut correct = 0;
    let mut truth = Vec::with_capacity(test.len());
    let mut pred  = Vec::with_capacity(test.len());
    let mut neutral_from = vec![0usize; EMOTIONS.len()];
    let mut neutral_total = 0;

    for (s, y) in test.iter() {
        let row = row_vec(&model.forward(encode_fixed(&tok, s)));
        let best = argmax(&row);
        let conf = row[best];
        if best == *y { correct += 1; }
        truth.push(*y);
        pred.push(best);
        if conf < NEUTRAL_THRESHOLD { neutral_total += 1; neutral_from[*y] += 1; }
    }

    println!("\nheld-out acc (raw argmax): {:.1}% ({}/{})",
             100.0 * correct as f64 / test.len() as f64, correct, test.len());

    print_confusion_matrix("Held-out confusion — raw argmax", &pred, &truth);

    println!("\n=== NEUTRAL THRESHOLD (conf < {:.2} => class 0) ===", NEUTRAL_THRESHOLD);
    println!("routed to neutral: {} / {} ({:.1}%)",
             neutral_total, test.len(), 100.0 * neutral_total as f64 / test.len() as f64);
    println!("{:<15}{:>10}{:>10}", "true class", "→neutral", "% of class");
    for c in 0..EMOTIONS.len() {
        let support = truth.iter().filter(|&&t| t == c).count();
        if support == 0 { continue; }
        println!("{:<15}{:>10}{:>10.1}",
                 EMOTIONS[c], neutral_from[c], 100.0 * neutral_from[c] as f64 / support as f64);
    }

    // probes
    let tests = [
        "you are absolutely amazing",
        "im losing my shit",
        "i miss her so much it hurts",
        "something is behind me and im terrified",
        "thank you so much for everything",
        "the meeting is at three pm",
        "wait what? i did not see that coming",
        "i cant believe i said that in front of everyone",
        "this is going to be so much fun, i cant wait",
    ];
    println!();
    for s in tests {
        let row = row_vec(&model.forward(encode_fixed(&tok, s)));
        let best = argmax(&row);
        let conf = row[best];
        let label = if conf < NEUTRAL_THRESHOLD { "neutral (0)".to_string() }
                    else { format!("{} ({})", EMOTIONS[best], best + 1) };
        println!("{:<50} -> {:<18} conf {:.2}", s, label, conf);
    }
}