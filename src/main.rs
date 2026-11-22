#![allow(warnings)]
mod transformer;
mod math;
mod preprocessing;
mod classifier;

use classifier::{Classifier, ClassifierWeights};
use nalgebra::DMatrix;
use preprocessing::{tokenize_initial, Tokenizer};
use rand::{RngExt, SeedableRng, rngs::StdRng}; 
use std::time::Instant;

use std::io::{self, Write, BufRead};
use serde_json;

const MODEL_PATH: &str = "model.json";


// ---- hyperparameters ----
const K_EPOCHS: usize = 15;
const LR: f64 = 0.004;
const N_LAYERS: usize = 3;
const TARGET_VOCAB: usize = 1200;
const SEQ_LEN: usize = 48;
const NEUTRAL_THRESHOLD: f64 = 0.65;
const TRAIN_FRAC: f64 = 0.75;
const WEIGHT_DECAY: f64 = 0.01;
const N_SAMPLES: usize = 20000;

const EMOTIONS: [&str; 5] = [
    "happy", "sad", "angry", "fear", "love"//, "neutral"
];

const GO_EMOTIONS_TO_CLASS: [Option<usize>; 28] = [
    None,     // 0  admiration
    Some(1),  // 1  amusement     -> happy
    Some(3),  // 2  anger         -> angry
    Some(3),  // 3  annoyance     -> angry
    None,     // 4  approval
    Some(5),     // 5  caring
    None,  // 6  confusion     -> fear        (see note below)
    None,     // 7  curiosity
    None,     // 8  desire
    None,  // 9  disappointment-> sad
    None,     // 10 disapproval
    None,  // 11 disgust       -> fear        (see note below)
    None,  // 12 embarrassment -> embarrassment  (was Some(7))
    Some(1),     // 13 excitement    (dropped)
    Some(4),  // 14 fear          -> fear
    None,     // 15 gratitude
    Some(2),  // 16 grief         -> sad
    Some(1),  // 17 joy           -> happy
    Some(5),     // 18 love
    None,     // 19 nervousness
    Some(1),  // 20 optimism      -> happy
    Some(1),  // 21 pride         -> happy
    None,     // 22 realization
    Some(1),  // 23 relief        -> happy
    None,  // 24 remorse       -> sad
    Some(2),  // 25 sadness       -> sad
    None,     // 26 surprise      (dropped)
    None//Some(6),     // 27 neutral
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
    let pad = tok.vocabulary["<PAD>"];
    let mut ids = tok.encode(s.to_string());
    ids.truncate(SEQ_LEN);
    while ids.len() < SEQ_LEN { ids.push(pad); }
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
    for v in w.iter_mut() { if *v > 1.5 { *v = 1.5; } }
    w
}

fn main() {
    let mut data = load_go_emotions_tsv("dataset/sub_dataset.tsv", N_SAMPLES);
    println!("loaded {} usable samples (from GoEmotions tsv)", data.len());

    let mut counts = vec![0usize; EMOTIONS.len()];
    for (_, y) in &data { counts[*y] += 1; }
    println!("class balance (full set):");
    for (i, name) in EMOTIONS.iter().enumerate() {
        println!("  {:<15} {}", name, counts[i]);
    }

    let mut r  = StdRng::seed_from_u64(42);
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
    let train_start = Instant::now();
    for epoch in 1..=K_EPOCHS {
        let epoch_start = Instant::now();
        let mut total_loss = 0.0;
        let mut correct = 0;
        for (s, y) in train.iter() {
            let probs = model.forward(encode_fixed(&tok, s));
            // weight the loss by inverse class frequency
            let w = weights[*y];
            total_loss += -w * probs[(0, *y)].max(1e-12).ln();
            if argmax(&row_vec(&probs)) == *y { correct += 1; }
            model.backwards(*y, weights[*y], LR, WEIGHT_DECAY);   // scale the gradient too
        }
        // report weighted loss normalized by total weight so epochs are comparable
        let total_w: f64 = train.iter().map(|(_, y)| weights[*y]).sum();
        println!(
            "epoch {:>2}/{} | wloss {:.4} | train acc {:.1}% | {:.1}s",
            epoch, K_EPOCHS,
            total_loss / total_w,
            100.0 * correct as f64 / train.len() as f64,
            epoch_start.elapsed().as_secs_f64()
        );
    }
    println!("training took {:.1}s", train_start.elapsed().as_secs_f64());
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

    let cw = model.to_weights(&tok);
    save_model(&cw, MODEL_PATH);

    // (optional) demonstrate round-trip load and run probes with the reloaded model
    let (cw2, tok2) = load_model(MODEL_PATH);
    let mut reloaded = Classifier::from_weights(&cw2, SEQ_LEN);
    // sanity check: reloaded should match
    {
        let a = row_vec(&model.forward(encode_fixed(&tok, "you are absolutely amazing")));
        let b = row_vec(&reloaded.forward(encode_fixed(&tok2, "you are absolutely amazing")));
        let diff: f64 = a.iter().zip(&b).map(|(x,y)| (x-y).abs()).sum();
        println!("reload sanity diff = {:.3e}", diff);
    }

    // ---- interactive console loop ----
    interactive_loop(reloaded, tok2, NEUTRAL_THRESHOLD);
}




fn save_model(cw: &ClassifierWeights, path: &str) {
    let s = serde_json::to_string(cw).expect("serialize");
    std::fs::write(path, s).expect("write model");
    println!("saved model -> {}", path);
}

fn load_model(path: &str) -> (ClassifierWeights, Tokenizer) {
    let s = std::fs::read_to_string(path).expect("read model");
    let cw: ClassifierWeights = serde_json::from_str(&s).expect("parse model");
    let tok = Tokenizer::from_parts(cw.vocab.clone(), cw.merge_history.clone());
    (cw, tok)
}

fn interactive_loop(mut model: Classifier, tok: Tokenizer, threshold: f64) {
    println!("\n=== interactive mode (type a sentence, Ctrl+D / Ctrl+C to quit) ===");
    let stdin = io::stdin();
    loop {
        print!("> ");
        io::stdout().flush().ok();
        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) => { println!(); break; }             // EOF
            Ok(_) => {}
            Err(e) => { eprintln!("input error: {}", e); break; }
        }
        let s = line.trim();
        if s.is_empty() { continue; }

        let probs = model.forward(encode_fixed(&tok, s));
        let row = row_vec(&probs);
        let best = argmax(&row);
        let conf = row[best];

        // print full distribution too, it's useful
        print!("  [");
        for (i, p) in row.iter().enumerate() {
            if i > 0 { print!(", "); }
            print!("{}:{:.2}", EMOTIONS[i], p);
        }
        println!("]");

        if conf < threshold {
            println!("  -> neutral (conf {:.2} below {:.2})", conf, threshold);
        } else {
            println!("  -> {} (conf {:.2})", EMOTIONS[best], conf);
        }
    }
}