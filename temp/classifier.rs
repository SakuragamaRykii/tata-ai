use nalgebra::{DMatrix, RowDVector};
use serde::{Deserialize, Serialize};
 
use crate::{math::*, preprocessing::Tokenizer, transformer::*};
 
// ---- serializable weight structs (same format as before) ----
 
#[derive(Serialize, Deserialize)]
pub struct MhaWeights {
    pub w_q: Vec<Vec<f64>>,
    pub w_k: Vec<Vec<f64>>,
    pub w_v: Vec<Vec<f64>>,
    pub w_o: Vec<Vec<f64>>,
}
 
#[derive(Serialize, Deserialize)]
pub struct FfnWeights {
    pub w1: Vec<Vec<f64>>,
    pub w2: Vec<Vec<f64>>,
    pub b1: Vec<f64>,
    pub b2: Vec<f64>,
    pub d_ff: usize,
}
 
#[derive(Serialize, Deserialize)]
pub struct NormWeights {
    pub gamma: Vec<f64>,
    pub beta: Vec<f64>,
}
 
#[derive(Serialize, Deserialize)]
pub struct LayerWeights {
    pub mha: MhaWeights,
    pub norm_mha: NormWeights,
    pub ffn: FfnWeights,
    pub norm_ffn: NormWeights,
}
 
#[derive(Serialize, Deserialize)]
pub struct ClassifierWeights {
    pub num_classes: usize,
    pub n: usize,
    pub w: Vec<Vec<f64>>,
    pub b: Vec<f64>,
    pub embed_table: Vec<Vec<f64>>,
    pub vocab: Vec<String>, // inverse_vocab
    pub merge_history: Vec<(String, String)>,
    pub layers: Vec<LayerWeights>,
}
 
/// d_ff = D_FF_MULT * D_MODEL
const D_FF_MULT: usize = 2;
 
pub struct Classifier {
    num_classes: usize,
    n: usize,
    pad_id: usize,
    w: Param, // D_MODEL x num_classes
    b: Param, // 1 x num_classes
    encoder: Encoder,
    head_drop: Dropout,
    // caches (set by forward, used by backward)
    c_cache: DMatrix<f64>,      // pooled sentence vectors after dropout   (batch x D_MODEL)
    output_cache: DMatrix<f64>, // softmax probabilities                   (batch x num_classes)
    pool_counts: Vec<f64>,      // number of real tokens per sentence
    seq_len: usize,
}
 
impl Classifier {
    pub fn empty(num_classes: usize, n: usize, vocab_size: usize, dropout: f64, pad_id: usize) -> Classifier {
        let w = Param::new(xavier_init(D_MODEL, num_classes), true);
        let b = Param::new(DMatrix::zeros(1, num_classes), false);
 
        let d_ff = D_MODEL * D_FF_MULT;
        let mut layers = Vec::<EncoderLayer>::with_capacity(n);
        for _ in 0..n {
            let mha = MHA::new(
                xavier_init(D_MODEL, D_MODEL),
                xavier_init(D_MODEL, D_MODEL),
                xavier_init(D_MODEL, D_MODEL),
                xavier_init(D_MODEL, D_MODEL),
            );
            let ffn = FeedForward::new(
                xavier_init(D_MODEL, d_ff),
                xavier_init(d_ff, D_MODEL),
                RowDVector::<f64>::zeros(d_ff),
                RowDVector::<f64>::zeros(D_MODEL),
                d_ff,
            );
            layers.push(EncoderLayer::new(mha, Normalizer::new(), ffn, Normalizer::new(), dropout));
        }
 
        let embedder = Embedder::new(EmbedLayer::new(vocab_size));
        Classifier {
            num_classes,
            n,
            pad_id,
            w,
            b,
            encoder: Encoder::new(embedder, layers),
            head_drop: Dropout::new(dropout),
            c_cache: DMatrix::zeros(1, 1),
            output_cache: DMatrix::zeros(1, 1),
            pool_counts: Vec::new(),
            seq_len: 0,
        }
    }
 
    pub fn params_mut(&mut self) -> Vec<&mut Param> {
        let mut v: Vec<&mut Param> = vec![&mut self.w, &mut self.b];
        v.extend(self.encoder.params_mut());
        v
    }
 
    /// inputs: (batch x seq) token ids padded with `pad_id`. Returns (batch x num_classes) probabilities.
    pub fn forward(&mut self, inputs: DMatrix<usize>, training: bool) -> DMatrix<f64> {
        let batch = inputs.nrows();
        let l = inputs.ncols();
        self.seq_len = l;
 
        let enc = self.encoder.forward(inputs, self.pad_id, training); // (batch*l) x D_MODEL
        let mask = self.encoder.mask();
 
        // masked MEAN pooling over real tokens (every sentence has <CLS>, so count >= 1)
        let mut pooled = DMatrix::<f64>::zeros(batch, D_MODEL);
        self.pool_counts = vec![0.0; batch];
        for i in 0..batch {
            let mut cnt: f64 = 0.0;
            for r in 0..l {
                if mask[i * l + r] {
                    cnt += 1.0;
                    for k in 0..D_MODEL {
                        pooled[(i, k)] += enc[(i * l + r, k)];
                    }
                }
            }
            let cnt = cnt.max(1.0);
            self.pool_counts[i] = cnt;
            for k in 0..D_MODEL {
                pooled[(i, k)] /= cnt;
            }
        }
 
        let pooled = self.head_drop.forward(pooled, training);
        self.c_cache = pooled.clone();
 
        let mut logits = pooled * &self.w.w;
        add_row_bias(&mut logits, &self.b.w);
        softmax_inplace(&mut logits, batch);
        self.output_cache = logits.clone();
        logits
    }
 
    /// Soft-label weighted cross-entropy backward pass for the batch of the last forward().
    ///   targets : (batch x num_classes), rows sum to 1
    ///   sample_w: per-sample loss weight
    /// Loss = sum_i w_i * CE_i / sum_i w_i.  Gradients are ACCUMULATED; call `step()` to apply them.
    pub fn backward(&mut self, targets: &DMatrix<f64>, sample_w: &[f64]) {
        let batch = self.output_cache.nrows();
        let l = self.seq_len;
        let wsum: f64 = sample_w.iter().sum::<f64>().max(1e-12);
 
        // dL/dlogits = w_i * (p - y) / sum(w)
        let mut dldz = DMatrix::<f64>::zeros(batch, self.num_classes);
        for i in 0..batch {
            for c in 0..self.num_classes {
                dldz[(i, c)] = sample_w[i] * (self.output_cache[(i, c)] - targets[(i, c)]) / wsum;
            }
        }
 
        self.w.g += self.c_cache.transpose() * &dldz;
        self.b.g += col_sums(&dldz);
 
        let dldc = &dldz * self.w.w.transpose(); // batch x D_MODEL (uses weights BEFORE the update)
        let dldc = self.head_drop.backward(dldc);
 
        // spread the pooled gradient evenly over the real tokens of each sentence
        let mask = self.encoder.mask().to_vec();
        let mut dldo = DMatrix::<f64>::zeros(batch * l, D_MODEL);
        for i in 0..batch {
            let inv = 1.0 / self.pool_counts[i];
            for r in 0..l {
                if mask[i * l + r] {
                    for k in 0..D_MODEL {
                        dldo[(i * l + r, k)] = dldc[(i, k)] * inv;
                    }
                }
            }
        }
        self.encoder.backwards(dldo);
    }
 
    /// Global-norm gradient clipping + one AdamW step on every parameter. Returns the (pre-clip) gradient norm.
    pub fn step(&mut self, mut cfg: AdamCfg, max_norm: f64) -> f64 {
        let mut params = self.params_mut();
        let norm = params.iter().map(|p| p.grad_sq()).sum::<f64>().sqrt();
        cfg.grad_scale = if norm > max_norm { max_norm / norm } else { 1.0 };
        for p in params.iter_mut() {
            p.step(&cfg);
        }
        norm
    }
 
    pub fn to_weights(&self, tok: &Tokenizer) -> ClassifierWeights {
        let layers: Vec<LayerWeights> = self
            .encoder
            .get_layers()
            .iter()
            .map(|l| {
                let (wq, wk, wv, wo) = l.get_mha().get_weights();
                let (w1, w2, b1, b2, d_ff) = l.get_ffn().get_weights();
                let (g1, b1n) = l.get_norm_mha().get_weights();
                let (g2, b2n) = l.get_norm_ffn().get_weights();
                LayerWeights {
                    mha: MhaWeights {
                        w_q: mat_to_vec(&wq),
                        w_k: mat_to_vec(&wk),
                        w_v: mat_to_vec(&wv),
                        w_o: mat_to_vec(&wo),
                    },
                    norm_mha: NormWeights { gamma: dvec_to_vec(&g1), beta: dvec_to_vec(&b1n) },
                    ffn: FfnWeights {
                        w1: mat_to_vec(&w1),
                        w2: mat_to_vec(&w2),
                        b1: row_to_vec(&b1),
                        b2: row_to_vec(&b2),
                        d_ff,
                    },
                    norm_ffn: NormWeights { gamma: dvec_to_vec(&g2), beta: dvec_to_vec(&b2n) },
                }
            })
            .collect();
 
        ClassifierWeights {
            num_classes: self.num_classes,
            n: self.n,
            w: mat_to_vec(&self.w.w),
            b: row_to_vec(&RowDVector::from_row_slice(self.b.w.as_slice())),
            embed_table: mat_to_vec(self.encoder.get_embedder().embed_table()),
            vocab: tok.inverse_vocab.clone(),
            merge_history: tok.merge_history.clone(),
            layers,
        }
    }
 
    /// Rebuild a Classifier from saved weights (inference, or to continue training with fresh Adam state).
    pub fn from_weights(cw: &ClassifierWeights) -> Classifier {
        let pad_id = cw.vocab.iter().position(|t| t == "<PAD>").unwrap_or(0);
        let embedder = Embedder::new(EmbedLayer::from_table(vec_to_mat(&cw.embed_table)));
 
        let mut layers = Vec::with_capacity(cw.n);
        for l in &cw.layers {
            let mha = MHA::new(
                vec_to_mat(&l.mha.w_q),
                vec_to_mat(&l.mha.w_k),
                vec_to_mat(&l.mha.w_v),
                vec_to_mat(&l.mha.w_o),
            );
            let norm_mha =
                Normalizer::from_weights(vec_to_dvec(&l.norm_mha.gamma), vec_to_dvec(&l.norm_mha.beta));
            let ffn = FeedForward::new(
                vec_to_mat(&l.ffn.w1),
                vec_to_mat(&l.ffn.w2),
                vec_to_row(&l.ffn.b1),
                vec_to_row(&l.ffn.b2),
                l.ffn.d_ff,
            );
            let norm_ffn =
                Normalizer::from_weights(vec_to_dvec(&l.norm_ffn.gamma), vec_to_dvec(&l.norm_ffn.beta));
            layers.push(EncoderLayer::new(mha, norm_mha, ffn, norm_ffn, 0.1));
        }
 
        Classifier {
            num_classes: cw.num_classes,
            n: cw.n,
            pad_id,
            w: Param::new(vec_to_mat(&cw.w), true),
            b: Param::new(DMatrix::from_row_slice(1, cw.b.len(), &cw.b), false),
            encoder: Encoder::new(embedder, layers),
            head_drop: Dropout::new(0.1),
            c_cache: DMatrix::zeros(1, 1),
            output_cache: DMatrix::zeros(1, 1),
            pool_counts: Vec::new(),
            seq_len: 0,
        }
    }
}
 
// ═══════════════════════════ gradient check ═══════════════════════════
// `cargo test --release grad_check -- --nocapture`
#[cfg(test)]
mod tests {
    use super::*;
 
    fn loss_of(m: &mut Classifier, ids: &DMatrix<usize>, targets: &DMatrix<f64>, w: &[f64]) -> f64 {
        let p = m.forward(ids.clone(), false);
        let mut s = 0.0;
        for i in 0..p.nrows() {
            let mut ce = 0.0;
            for c in 0..p.ncols() {
                ce -= targets[(i, c)] * p[(i, c)].max(1e-300).ln();
            }
            s += w[i] * ce;
        }
        s / w.iter().sum::<f64>()
    }
 
    #[test]
    fn grad_check() {
        let classes = 3;
        let mut m = Classifier::empty(classes, 2, 20, 0.0, 0);
        // 0 = <PAD>; sentences have different amounts of padding
        let ids = DMatrix::from_row_slice(
            3,
            7,
            &[2, 5, 6, 7, 8, 0, 0, //
              2, 9, 10, 0, 0, 0, 0, //
              2, 11, 12, 13, 14, 15, 16],
        );
        let targets = DMatrix::from_row_slice(3, 3, &[1.0, 0.0, 0.0, 0.3, 0.7, 0.0, 0.0, 0.2, 0.8]);
        let w = [1.0, 2.0, 0.5];
 
        // analytic
        let _ = loss_of(&mut m, &ids, &targets, &w);
        m.backward(&targets, &w);
        let grads: Vec<DMatrix<f64>> = m.params_mut().iter().map(|p| p.g.clone()).collect();
 
        let eps = 1e-5;
        let mut worst: f64 = 0.0;
        let n_params = grads.len();
        for k in 0..n_params {
            // the 3 entries with the largest |grad|, plus one fixed entry
            let g = &grads[k];
            let mut idx: Vec<(usize, usize)> = (0..g.nrows()).flat_map(|i| (0..g.ncols()).map(move |j| (i, j))).collect();
            idx.sort_by(|a, b| g[*b].abs().partial_cmp(&g[*a].abs()).unwrap());
            let mut picks: Vec<(usize, usize)> = idx.iter().take(3).cloned().collect();
            picks.push((g.nrows() / 2, g.ncols() / 3));
 
            for (i, j) in picks {
                let orig = m.params_mut()[k].w[(i, j)];
                m.params_mut()[k].w[(i, j)] = orig + eps;
                let lp = loss_of(&mut m, &ids, &targets, &w);
                m.params_mut()[k].w[(i, j)] = orig - eps;
                let lm = loss_of(&mut m, &ids, &targets, &w);
                m.params_mut()[k].w[(i, j)] = orig;
 
                let numeric = (lp - lm) / (2.0 * eps);
                let analytic = g[(i, j)];
                let denom = numeric.abs() + analytic.abs();
                let rel = if denom < 1e-9 { 0.0 } else { (numeric - analytic).abs() / denom };
                worst = worst.max(rel);
                if rel > 1e-4 {
                    println!("param {k} ({i},{j}): analytic {analytic:.6e} numeric {numeric:.6e} rel {rel:.2e}");
                }
            }
        }
        println!("checked {} tensors, worst relative error = {:.2e}", n_params, worst);
        assert!(worst < 1e-4, "gradient mismatch");
    }
}
 