#![allow(warnings)]
mod classifier;
mod math;
mod preprocessing;
mod transformer;

use classifier::{Classifier, ClassifierWeights};
use math::AdamCfg;
use nalgebra::DMatrix;
use preprocessing::{tokenize_initial, Tokenizer};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use serde_json;
use std::io::{self, BufRead, Write};
use std::time::Instant;

const MODEL_PATH: &str = "model.json";
const DATA_PATH: &str = "dataset/sub_dataset.tsv";

// ---- hyperparameters ----
const K_EPOCHS: usize = 20;          // upper bound; early stopping usually ends sooner
const PATIENCE: usize = 4;           // stop after this many epochs without val macro-F1 improvement
const BATCH: usize = 32;
const LR: f64 = 5e-4;                // peak learning rate (AdamW)
const WARMUP_FRAC: f64 = 0.06;       // fraction of total steps used for linear warm-up
const MIN_LR_FRAC: f64 = 0.05;       // cosine decays down to LR * this
const WEIGHT_DECAY: f64 = 0.01;      // decoupled AdamW decay (weight matrices only)
const GRAD_CLIP: f64 = 1.0;          // global gradient-norm clip
const DROPOUT: f64 = 0.1;
const N_LAYERS: usize = 2;
const TARGET_VOCAB: usize = 3000;
const SEQ_LEN: usize = 48;           // max tokens kept (batches are padded only to their own longest sentence)
const TRAIN_FRAC: f64 = 0.80;
const VAL_FRAC: f64 = 0.10;          // rest is test
const N_SAMPLES: usize = 1_000_000;
const MAX_CLASS_WEIGHT: f64 = 3.0;

// neutral is now a REAL class (index 0) — no confidence threshold any more
const EMOTIONS: [&str; 6] = ["neutral", "happy", "sad", "angry", "fear", "surprise"];
const NUM_CLASSES: usize = EMOTIONS.len();

// GoEmotions label id -> class index into EMOTIONS (0 neutral, 1 happy, 2 sad, 3 angry, 4 fear, 5 surprise)
const GO_EMOTIONS_TO_CLASS: [Option<usize>; 28] = [
    Some(0), // 0  admiration
    Some(1), // 1  amusement
    Some(3), // 2  anger
    Some(3), // 3  annoyance
    Some(0), // 4  approval
    Some(1), // 5  caring
    Some(5), // 6  confusion
    Some(5), // 7  curiosity
    Some(0), // 8  desire
    Some(0), // 9  disappointment
    Some(0), // 10 disapproval
    Some(0), // 11 disgust
    Some(0), // 12 embarrassment
    Some(1), // 13 excitement
    Some(4), // 14 fear
    Some(0), // 15 gratitude
    Some(2), // 16 grief
    Some(1), // 17 joy
    Some(1), // 18 love
    Some(4), // 19 nervousness
    Some(1), // 20 optimism
    Some(1), // 21 pride
    Some(5), // 22 realization
    Some(1), // 23 relief
    Some(0), // 24 remorse
    Some(2), // 25 sadness
    Some(5), // 26 surprise
    Some(0), // 27 neutral
];

// ───────────────────────────── data ─────────────────────────────

struct Sample {
    text: String,
    target: Vec<f64>, // soft label over classes (fraction of the row's labels that fall in each class)
    hard: usize,      // argmax of target (ties -> the higher class index, i.e. prefer non-neutral)
}

struct Enc {
    ids: Vec<usize>,
    target: Vec<f64>,
    hard: usize,
    w: f64, // per-sample loss weight
}

fn load_go_emotions_tsv(path: &str, limit: usize) -> Vec<Sample> {
    let text = std::fs::read_to_string(path).expect("could not read tsv");
    let mut out = Vec::new();
    let (mut n_blank, mut n_short, mut n_empty_text, mut n_bad_label, mut n_oob, mut n_all_none, mut n_total) =
        (0usize, 0, 0, 0, 0, 0, 0);

    for line in text.lines() {
        n_total += 1;
        if line.trim().is_empty() { n_blank += 1; continue; }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 2 { n_short += 1; continue; }
        let sentence = fields[0].trim();
        if sentence.is_empty() { n_empty_text += 1; continue; }

        let ids: Vec<usize> = fields[1].split(',').filter_map(|t| t.trim().parse::<usize>().ok()).collect();
        if ids.is_empty() { n_bad_label += 1; continue; }
        if ids.iter().any(|&i| i >= GO_EMOTIONS_TO_CLASS.len()) { n_oob += 1; continue; }

        let mut target = vec![0.0; NUM_CLASSES];
        let mut n_mapped = 0.0;
        for &i in &ids {
            if let Some(c) = GO_EMOTIONS_TO_CLASS[i] {
                target[c] += 1.0;
                n_mapped += 1.0;
            }
        }
        if n_mapped == 0.0 { n_all_none += 1; continue; }
        for t in target.iter_mut() { *t /= n_mapped; }

        let mut hard = 0;
        for c in 0..NUM_CLASSES { if target[c] >= target[hard] { hard = c; } }

        out.push(Sample { text: sentence.to_string(), target, hard });
        if out.len() >= limit { break; }
    }

    eprintln!(
        "lines={} blank={} short={} empty_text={} bad_label={} oob={} all_none={} kept={}",
        n_total, n_blank, n_short, n_empty_text, n_bad_label, n_oob, n_all_none, out.len()
    );
    out
}

fn shuffle<T>(v: &mut [T], r: &mut StdRng) {
    for i in (1..v.len()).rev() {
        let j = r.random_range(0..=i);
        v.swap(i, j);
    }
}

fn encode_ids(tok: &Tokenizer, s: &str) -> Vec<usize> {
    let mut ids = tok.encode(s.to_string());
    ids.truncate(SEQ_LEN);
    ids
}

fn encode_all(tok: &Tokenizer, data: &[Sample], class_w: &[f64]) -> Vec<Enc> {
    data.iter()
        .map(|s| Enc {
            ids: encode_ids(tok, &s.text),
            target: s.target.clone(),
            hard: s.hard,
            w: s.target.iter().zip(class_w).map(|(t, w)| t * w).sum(),
        })
        .collect()
}

/// Pads only to the longest sentence IN THIS BATCH (big speed-up: most GoEmotions sentences are short).
fn make_batch(data: &[Enc], idx: &[usize], pad: usize) -> (DMatrix<usize>, DMatrix<f64>, Vec<f64>) {
    let l = idx.iter().map(|&i| data[i].ids.len()).max().unwrap_or(1).max(1);
    let ids = DMatrix::from_fn(idx.len(), l, |r, c| data[idx[r]].ids.get(c).copied().unwrap_or(pad));
    let targets = DMatrix::from_fn(idx.len(), NUM_CLASSES, |r, c| data[idx[r]].target[c]);
    let w = idx.iter().map(|&i| data[i].w).collect();
    (ids, targets, w)
}

// inverse-frequency weights from SOFT counts, mean-normalised, capped
fn class_weights(soft_counts: &[f64]) -> Vec<f64> {
    let n = soft_counts.len() as f64;
    let total: f64 = soft_counts.iter().sum();
    let mut w: Vec<f64> = soft_counts.iter().map(|&c| if c <= 0.0 { 0.0 } else { total / (n * c) }).collect();
    let mean = w.iter().sum::<f64>() / n;
    if mean > 0.0 { for v in w.iter_mut() { *v /= mean; } }
    for v in w.iter_mut() { if *v > MAX_CLASS_WEIGHT { *v = MAX_CLASS_WEIGHT; } }
    w
}

// ───────────────────────────── metrics ─────────────────────────────

fn argmax(row: &[f64]) -> usize {
    let mut b = 0;
    for i in 0..row.len() { if row[i] > row[b] { b = i; } }
    b
}

fn macro_f1(pred: &[usize], truth: &[usize]) -> f64 {
    let n = NUM_CLASSES;
    let mut tp = vec![0usize; n];
    let mut fp = vec![0usize; n];
    let mut fnn = vec![0usize; n];
    for (&p, &t) in pred.iter().zip(truth) {
        if p == t { tp[t] += 1; } else { fp[p] += 1; fnn[t] += 1; }
    }
    let (mut sum, mut k) = (0.0, 0);
    for c in 0..n {
        if tp[c] + fnn[c] == 0 { continue; } // class absent from this split
        let prec = if tp[c] + fp[c] == 0 { 0.0 } else { tp[c] as f64 / (tp[c] + fp[c]) as f64 };
        let rec = tp[c] as f64 / (tp[c] + fnn[c]) as f64;
        sum += if prec + rec == 0.0 { 0.0 } else { 2.0 * prec * rec / (prec + rec) };
        k += 1;
    }
    if k == 0 { 0.0 } else { sum / k as f64 }
}

struct EvalOut { loss: f64, acc: f64, f1: f64, pred: Vec<usize>, truth: Vec<usize> }

fn evaluate(model: &mut Classifier, data: &[Enc], pad: usize) -> EvalOut {
    let mut order: Vec<usize> = (0..data.len()).collect();
    order.sort_by_key(|&i| data[i].ids.len()); // length-sorted => little padding; results are per-sample anyway
    let mut pred = vec![0usize; data.len()];
    let (mut loss_sum, mut w_sum) = (0.0, 0.0);

    for chunk in order.chunks(64) {
        let (ids, targets, w) = make_batch(data, chunk, pad);
        let probs = model.forward(ids, false);
        for (r, &i) in chunk.iter().enumerate() {
            let mut ce = 0.0;
            for c in 0..NUM_CLASSES { ce -= targets[(r, c)] * probs[(r, c)].max(1e-12).ln(); }
            loss_sum += w[r] * ce;
            w_sum += w[r];
            let row: Vec<f64> = (0..NUM_CLASSES).map(|c| probs[(r, c)]).collect();
            pred[i] = argmax(&row);
        }
    }
    let truth: Vec<usize> = data.iter().map(|d| d.hard).collect();
    let acc = pred.iter().zip(&truth).filter(|(p, t)| p == t).count() as f64 / data.len().max(1) as f64;
    EvalOut { loss: loss_sum / w_sum.max(1e-12), acc, f1: macro_f1(&pred, &truth), pred, truth }
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
        let rec = if support == 0 { 0.0 } else { counts[c][c] as f64 / support as f64 };
        let f1 = if prec + rec == 0.0 { 0.0 } else { 2.0 * prec * rec / (prec + rec) };
        println!("{:<14}{:>10}{:>10.1}{:>10.1}{:>10.1}", EMOTIONS[c], support, 100.0 * prec, 100.0 * rec, 100.0 * f1);
        if support > 0 { mp += prec; mr += rec; mf += f1; k += 1; }
    }
    if k > 0 {
        println!("{}", "-".repeat(54));
        println!("{:<14}{:>10}{:>10.1}{:>10.1}{:>10.1}", "macro", "", 100.0 * mp / k as f64, 100.0 * mr / k as f64, 100.0 * mf / k as f64);
    }
}

// ───────────────────────────── schedule ─────────────────────────────

fn lr_at(step: usize, total: usize) -> f64 {
    let warm = ((total as f64) * WARMUP_FRAC).max(1.0);
    let s = step as f64;
    if s < warm {
        LR * (s + 1.0) / warm
    } else {
        let p = ((s - warm) / ((total as f64) - warm).max(1.0)).min(1.0);
        LR * (MIN_LR_FRAC + (1.0 - MIN_LR_FRAC) * 0.5 * (1.0 + (std::f64::consts::PI * p).cos()))
    }
}

// ───────────────────────────── main ─────────────────────────────

fn main() {
    let mut data = load_go_emotions_tsv(DATA_PATH, N_SAMPLES);
    println!("loaded {} usable samples", data.len());

    let mut counts = vec![0usize; NUM_CLASSES];
    for s in &data { counts[s.hard] += 1; }
    println!("class balance (full set, by hard label):");
    for (i, name) in EMOTIONS.iter().enumerate() { println!("  {:<10} {}", name, counts[i]); }

    // ---- split: train / val / test ----
    let mut rng = StdRng::seed_from_u64(42);
    shuffle(&mut data, &mut rng);
    let n_train = (data.len() as f64 * TRAIN_FRAC) as usize;
    let n_val = (data.len() as f64 * VAL_FRAC) as usize;
    let (train, rest) = data.split_at(n_train);
    let (val, test) = rest.split_at(n_val);
    println!("split: train {} | val {} | test {}", train.len(), val.len(), test.len());

    // class weights from the TRAIN split only (soft counts)
    let mut soft = vec![0.0; NUM_CLASSES];
    for s in train { for c in 0..NUM_CLASSES { soft[c] += s.target[c]; } }
    let weights = class_weights(&soft);
    println!("train soft counts / class weights:");
    for (i, name) in EMOTIONS.iter().enumerate() {
        println!("  {:<10} {:>9.1}   w={:.3}", name, soft[i], weights[i]);
    }

    // ---- tokenizer (trained on TRAIN text only) ----
    let all_text: String = train.iter().map(|s| s.text.as_str()).collect::<Vec<_>>().join(" ");
    let mut tok = Tokenizer::new(all_text);
    let corpus: Vec<Vec<String>> = train.iter().flat_map(|s| tokenize_initial(s.text.clone())).collect();
    let t0 = Instant::now();
    tok.bpe(corpus, TARGET_VOCAB);
    println!("vocab size = {} (BPE took {:.1}s)", tok.vocabulary.len(), t0.elapsed().as_secs_f64());
    let pad = tok.vocabulary["<PAD>"];

    // encode everything ONCE
    let train_enc = encode_all(&tok, train, &weights);
    let val_enc = encode_all(&tok, val, &weights);
    let test_enc = encode_all(&tok, test, &weights);

    // how long are sentences / how much do we truncate?
    let mut lens: Vec<usize> = train.iter().map(|s| tok.encode(s.text.clone()).len()).collect();
    lens.sort();
    let pct = |p: f64| lens[((lens.len() as f64 - 1.0) * p) as usize];
    println!("token length  p50={} p90={} p95={} p99={} max={}  (SEQ_LEN={})",
             pct(0.5), pct(0.9), pct(0.95), pct(0.99), lens[lens.len() - 1], SEQ_LEN);

    // ---- model ----
    let mut model = Classifier::empty(NUM_CLASSES, N_LAYERS, tok.vocabulary.len(), DROPOUT, pad);

    let steps_per_epoch = (train_enc.len() + BATCH - 1) / BATCH;
    let total_steps = steps_per_epoch * K_EPOCHS;
    let mut step = 0usize;
    let mut best_f1 = -1.0;
    let mut best_cw: Option<ClassifierWeights> = None;
    let mut best_epoch = 0;
    let mut bad_epochs = 0;
    let train_start = Instant::now();

    for epoch in 1..=K_EPOCHS {
        let epoch_start = Instant::now();

        // length-bucketed shuffled batches: shuffle, sort inside big chunks, cut into batches, shuffle batches
        let mut order: Vec<usize> = (0..train_enc.len()).collect();
        shuffle(&mut order, &mut rng);
        let mut batches: Vec<Vec<usize>> = Vec::new();
        for chunk in order.chunks(BATCH * 50) {
            let mut c = chunk.to_vec();
            c.sort_by_key(|&i| train_enc[i].ids.len());
            for b in c.chunks(BATCH) { batches.push(b.to_vec()); }
        }
        shuffle(&mut batches, &mut rng);

        let (mut loss_sum, mut w_sum, mut correct, mut seen, mut gnorm_sum) = (0.0, 0.0, 0usize, 0usize, 0.0);
        for b in &batches {
            let (ids, targets, w) = make_batch(&train_enc, b, pad);
            let probs = model.forward(ids, true);

            for (r, &i) in b.iter().enumerate() {
                let mut ce = 0.0;
                for c in 0..NUM_CLASSES { ce -= targets[(r, c)] * probs[(r, c)].max(1e-12).ln(); }
                loss_sum += w[r] * ce;
                w_sum += w[r];
                let row: Vec<f64> = (0..NUM_CLASSES).map(|c| probs[(r, c)]).collect();
                if argmax(&row) == train_enc[i].hard { correct += 1; }
                seen += 1;
            }

            model.backward(&targets, &w);
            step += 1;
            let cfg = AdamCfg {
                lr: lr_at(step - 1, total_steps),
                beta1: 0.9,
                beta2: 0.98,
                eps: 1e-8,
                weight_decay: WEIGHT_DECAY,
                t: step,
                grad_scale: 1.0,
            };
            gnorm_sum += model.step(cfg, GRAD_CLIP);
        }

        let v = evaluate(&mut model, &val_enc, pad);
        println!(
            "epoch {:>2}/{} | train wloss {:.4} acc {:.1}% | val wloss {:.4} acc {:.1}% macroF1 {:.1}% | lr {:.2e} | gnorm {:.2} | {:.0}s",
            epoch, K_EPOCHS,
            loss_sum / w_sum.max(1e-12), 100.0 * correct as f64 / seen as f64,
            v.loss, 100.0 * v.acc, 100.0 * v.f1,
            lr_at(step.saturating_sub(1), total_steps),
            gnorm_sum / batches.len() as f64,
            epoch_start.elapsed().as_secs_f64()
        );
        io::stdout().flush().ok();

        if v.f1 > best_f1 {
            best_f1 = v.f1;
            best_epoch = epoch;
            bad_epochs = 0;
            best_cw = Some(model.to_weights(&tok));
        } else {
            bad_epochs += 1;
            if bad_epochs >= PATIENCE {
                println!("early stop: no val macro-F1 gain for {} epochs", PATIENCE);
                break;
            }
        }
    }
    println!("training took {:.1}s; best val macro-F1 {:.1}% at epoch {}", train_start.elapsed().as_secs_f64(), 100.0 * best_f1, best_epoch);

    // ---- restore best checkpoint, evaluate on the untouched test split ----
    let best_cw = best_cw.expect("no checkpoint");
    let mut model = Classifier::from_weights(&best_cw);

    let t = evaluate(&mut model, &test_enc, pad);
    println!("\nTEST  acc {:.1}%  macroF1 {:.1}%  wloss {:.4}  (n={})", 100.0 * t.acc, 100.0 * t.f1, t.loss, test_enc.len());
    print_confusion_matrix("Test confusion", &t.pred, &t.truth);

    // ---- probes ----
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
    for s in tests { print_prediction(&mut model, &tok, s, pad, false); }

    save_model(&best_cw, MODEL_PATH);

    // round-trip sanity check
    let (cw2, tok2) = load_model(MODEL_PATH);
    let mut reloaded = Classifier::from_weights(&cw2);
    {
        let probe = "you are absolutely amazing";
        let a = predict(&mut model, &tok, probe, pad);
        let b = predict(&mut reloaded, &tok2, probe, pad);
        let diff: f64 = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).sum();
        println!("reload sanity diff = {:.3e}", diff);
    }

    interactive_loop(reloaded, tok2, pad);
}

// ───────────────────────────── inference / io ─────────────────────────────

fn predict(model: &mut Classifier, tok: &Tokenizer, s: &str, pad: usize) -> Vec<f64> {
    let ids = encode_ids(tok, s);
    let m = DMatrix::from_row_slice(1, ids.len(), &ids);
    let p = model.forward(m, false);
    (0..NUM_CLASSES).map(|c| p[(0, c)]).collect()
}

fn print_prediction(model: &mut Classifier, tok: &Tokenizer, s: &str, pad: usize, full: bool) {
    let row = predict(model, tok, s, pad);
    let best = argmax(&row);
    if full {
        print!("  [");
        for (i, p) in row.iter().enumerate() {
            if i > 0 { print!(", "); }
            print!("{}:{:.2}", EMOTIONS[i], p);
        }
        println!("]");
        println!("  -> {} (p {:.2})", EMOTIONS[best], row[best]);
    } else {
        println!("{:<50} -> {:<10} p {:.2}", s, EMOTIONS[best], row[best]);
    }
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

fn interactive_loop(mut model: Classifier, tok: Tokenizer, pad: usize) {
    println!("\n=== interactive mode (type a sentence, Ctrl+D / Ctrl+C to quit) ===");
    let stdin = io::stdin();
    loop {
        print!("> ");
        io::stdout().flush().ok();
        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) => { println!(); break; }
            Ok(_) => {}
            Err(e) => { eprintln!("input error: {}", e); break; }
        }
        let s = line.trim();
        if s.is_empty() { continue; }
        print_prediction(&mut model, &tok, s, pad, true);
    }
}