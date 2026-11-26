use crate::math::*;
 
use nalgebra::*;
 
pub const D_MODEL: usize = 128; // 512 for bigger tests
// dimensionality of a token
 
pub const N_HEADS: usize = 4;
// number of attention heads
 
pub const D_K: usize = D_MODEL / N_HEADS;
 
const LN_EPS: f64 = 1e-5;
const MASK_NEG: f64 = -1e9;
 
// ═════════════════════════════ Multi-head attention ═════════════════════════════
 
pub struct MHA {
    w_q: Param,
    w_k: Param,
    w_v: Param,
    w_o: Param,
 
    x_cache: DMatrix<f64>,
    q_cache: DMatrix<f64>,
    k_cache: DMatrix<f64>,
    v_cache: DMatrix<f64>,
    heads_cache: DMatrix<f64>,
    attention_cache: Vec<DMatrix<f64>>, // one (token_count x max_seq_length) matrix per head
}
 
impl MHA {
    pub fn new(w_q: DMatrix<f64>, w_k: DMatrix<f64>, w_v: DMatrix<f64>, w_o: DMatrix<f64>) -> MHA {
        MHA {
            w_q: Param::new(w_q, true),
            w_k: Param::new(w_k, true),
            w_v: Param::new(w_v, true),
            w_o: Param::new(w_o, true),
            x_cache: DMatrix::zeros(1, 1),
            q_cache: DMatrix::zeros(1, 1),
            k_cache: DMatrix::zeros(1, 1),
            v_cache: DMatrix::zeros(1, 1),
            heads_cache: DMatrix::zeros(1, 1),
            attention_cache: Vec::new(),
        }
    }
 
    pub fn get_weights(&self) -> (DMatrix<f64>, DMatrix<f64>, DMatrix<f64>, DMatrix<f64>) {
        (self.w_q.w.clone(), self.w_k.w.clone(), self.w_v.w.clone(), self.w_o.w.clone())
    }
 
    pub fn params_mut(&mut self) -> Vec<&mut Param> {
        vec![&mut self.w_q, &mut self.w_k, &mut self.w_v, &mut self.w_o]
    }
 
    /// `mask[r]` is true when token row `r` is a real token (false = <PAD>).
    /// Padded positions are never attended TO.
    pub fn mha_forward(&mut self, x: &DMatrix<f64>, max_seq_length: usize, mask: &[bool]) -> DMatrix<f64> {
        let q = x * &self.w_q.w;
        let k = x * &self.w_k.w;
        let v = x * &self.w_v.w;
 
        self.x_cache = x.clone();
        self.q_cache = q.clone();
        self.k_cache = k.clone();
        self.v_cache = v.clone();
 
        let token_count = x.nrows();
        self.attention_cache.clear();
 
        let mut head_attentions = DMatrix::<f64>::zeros(token_count, D_MODEL);
 
        for head in 0..N_HEADS {
            let col = head * D_K;
            let q_head = q.view((0, col), (token_count, D_K)).into_owned();
            let k_head = k.view((0, col), (token_count, D_K)).into_owned();
            let v_head = v.view((0, col), (token_count, D_K)).into_owned();
 
            let att = self.evaluate_attention(q_head, k_head, v_head, token_count, max_seq_length, mask);
            head_attentions.view_mut((0, col), (token_count, D_K)).copy_from(&att);
        }
        self.heads_cache = head_attentions.clone();
        head_attentions * &self.w_o.w
    }
 
    pub fn evaluate_attention(
        &mut self,
        q: DMatrix<f64>,
        k: DMatrix<f64>,
        v: DMatrix<f64>,
        token_count: usize,
        max_seq_length: usize,
        mask: &[bool],
    ) -> DMatrix<f64> {
        let dk_sqrt = (D_K as f64).sqrt();
        let num_sentences = token_count / max_seq_length;
 
        let mut result = DMatrix::<f64>::zeros(token_count, D_K);
        let mut attentions = DMatrix::<f64>::zeros(token_count, max_seq_length);
 
        for sentence in 0..num_sentences {
            let s0 = sentence * max_seq_length;
 
            let q_block = q.view((s0, 0), (max_seq_length, D_K)).into_owned();
            let k_block = k.view((s0, 0), (max_seq_length, D_K)).into_owned();
            let v_block = v.view((s0, 0), (max_seq_length, D_K)).into_owned();
 
            let mut qk_block = (q_block * k_block.transpose()) / dk_sqrt;
 
            // PADDING MASK: nobody may attend to a <PAD> key.
            for j in 0..max_seq_length {
                if !mask[s0 + j] {
                    for i in 0..max_seq_length {
                        qk_block[(i, j)] = MASK_NEG;
                    }
                }
            }
 
            softmax_inplace(&mut qk_block, max_seq_length);
 
            result.view_mut((s0, 0), (max_seq_length, D_K)).copy_from(&(&qk_block * v_block));
            attentions.view_mut((s0, 0), (max_seq_length, max_seq_length)).copy_from(&qk_block);
        }
        self.attention_cache.push(attentions);
        result
    }
 
    /// Accumulates parameter gradients; returns dL/dX.
    /// (Masked attention weights are exactly 0, so the softmax backward gives 0 for them -> no mask needed here.)
    pub fn backwards(&mut self, dldo: DMatrix<f64>) -> DMatrix<f64> {
        let t = self.x_cache.nrows();
        let l = self.attention_cache[0].ncols(); // max_seq_length
        let num_sentences = t / l;
        let dk_sqrt = (D_K as f64).sqrt();
 
        let dldw_o = self.heads_cache.transpose() * &dldo;
        let dldh = dldo * self.w_o.w.transpose(); // t x D_MODEL
 
        let mut dldq = DMatrix::<f64>::zeros(t, D_MODEL);
        let mut dldk = DMatrix::<f64>::zeros(t, D_MODEL);
        let mut dldv = DMatrix::<f64>::zeros(t, D_MODEL);
 
        for head in 0..N_HEADS {
            let col = head * D_K;
            for s in 0..num_sentences {
                let r0 = s * l;
                let a = self.attention_cache[head].view((r0, 0), (l, l)).into_owned();
                let q = self.q_cache.view((r0, col), (l, D_K)).into_owned();
                let k = self.k_cache.view((r0, col), (l, D_K)).into_owned();
                let v = self.v_cache.view((r0, col), (l, D_K)).into_owned();
                let dh = dldh.view((r0, col), (l, D_K)).into_owned();
 
                let dv = a.transpose() * &dh; // l x D_K
                let da = &dh * v.transpose(); // l x l
 
                // softmax backward, row by row: ds = a * (da - sum(a * da))
                let mut ds = DMatrix::<f64>::zeros(l, l);
                for i in 0..l {
                    let dot: f64 = (0..l).map(|j| a[(i, j)] * da[(i, j)]).sum();
                    for j in 0..l {
                        ds[(i, j)] = a[(i, j)] * (da[(i, j)] - dot);
                    }
                }
 
                let dq = (&ds * &k) / dk_sqrt;
                let dk_ = (ds.transpose() * &q) / dk_sqrt;
 
                dldq.view_mut((r0, col), (l, D_K)).copy_from(&dq);
                dldk.view_mut((r0, col), (l, D_K)).copy_from(&dk_);
                dldv.view_mut((r0, col), (l, D_K)).copy_from(&dv);
            }
        }
 
        let x_t = self.x_cache.transpose();
        self.w_o.g += dldw_o;
        self.w_q.g += &x_t * &dldq;
        self.w_k.g += &x_t * &dldk;
        self.w_v.g += &x_t * &dldv;
 
        &dldv * self.w_v.w.transpose() + &dldk * self.w_k.w.transpose() + &dldq * self.w_q.w.transpose()
    }
}
 
// ═════════════════════════════ Feed-forward ═════════════════════════════
 
pub struct FeedForward {
    w1: Param, // D_MODEL x d_ff
    w2: Param, // d_ff x D_MODEL
    b1: Param, // 1 x d_ff
    b2: Param, // 1 x D_MODEL
    d_ff: usize,
    x_cache: DMatrix<f64>,
    r_cache: DMatrix<f64>,
    a_cache: DMatrix<f64>,
}
 
impl FeedForward {
    pub fn new(
        w1: DMatrix<f64>,
        w2: DMatrix<f64>,
        b1: RowDVector<f64>,
        b2: RowDVector<f64>,
        d_ff: usize,
    ) -> FeedForward {
        let b1m = DMatrix::from_row_slice(1, b1.len(), b1.as_slice());
        let b2m = DMatrix::from_row_slice(1, b2.len(), b2.as_slice());
        FeedForward {
            w1: Param::new(w1, true),
            w2: Param::new(w2, true),
            b1: Param::new(b1m, false),
            b2: Param::new(b2m, false),
            d_ff,
            x_cache: DMatrix::zeros(1, 1),
            r_cache: DMatrix::zeros(1, 1),
            a_cache: DMatrix::zeros(1, 1),
        }
    }
 
    pub fn get_weights(&self) -> (DMatrix<f64>, DMatrix<f64>, RowDVector<f64>, RowDVector<f64>, usize) {
        (
            self.w1.w.clone(),
            self.w2.w.clone(),
            RowDVector::from_row_slice(self.b1.w.as_slice()),
            RowDVector::from_row_slice(self.b2.w.as_slice()),
            self.d_ff,
        )
    }
 
    pub fn params_mut(&mut self) -> Vec<&mut Param> {
        vec![&mut self.w1, &mut self.w2, &mut self.b1, &mut self.b2]
    }
 
    pub fn ffn(&mut self, x: DMatrix<f64>) -> DMatrix<f64> {
        // FFN(x) = max(0, xW1 + b1)W2 + b2
        self.x_cache = x.clone();
        let mut a = &x * &self.w1.w;
        add_row_bias(&mut a, &self.b1.w);
        self.a_cache = a.clone();
        let r = relu(a);
        self.r_cache = r.clone();
        let mut out = &r * &self.w2.w;
        add_row_bias(&mut out, &self.b2.w);
        out
    }
 
    pub fn backwards(&mut self, dldo: DMatrix<f64>) -> DMatrix<f64> {
        self.b2.g += col_sums(&dldo);
        self.w2.g += self.r_cache.transpose() * &dldo;
 
        let drda = self.a_cache.map(|x| if x > 0.0 { 1.0 } else { 0.0 });
        let dlda = (&dldo * self.w2.w.transpose()).component_mul(&drda);
 
        self.w1.g += self.x_cache.transpose() * &dlda;
        self.b1.g += col_sums(&dlda);
 
        dlda * self.w1.w.transpose()
    }
}
 
// ═════════════════════════════ Embeddings ═════════════════════════════
 
pub struct EmbedLayer {
    table: Param, // vocab_size x D_MODEL
    vocab_size: usize,
}
 
impl EmbedLayer {
    pub fn new(vocab_size: usize) -> EmbedLayer {
        EmbedLayer {
            table: Param::new(rand_matrix(vocab_size, D_MODEL, -1.0, 1.0), false),
            vocab_size,
        }
    }
 
    pub fn table(&self) -> &DMatrix<f64> {
        &self.table.w
    }
    pub fn from_table(table: DMatrix<f64>) -> EmbedLayer {
        let vocab_size = table.nrows();
        EmbedLayer { table: Param::new(table, false), vocab_size }
    }
}
 
fn build_positional_table(rows: usize) -> DMatrix<f64> {
    let base: f64 = 10000.0;
    DMatrix::from_fn(rows, D_MODEL, |pos, col| {
        let a = pos as f64 / base.powf(col as f64 / D_MODEL as f64);
        if col % 2 == 0 { a.sin() } else { a.cos() }
    })
}
 
pub struct Embedder {
    embed_layer: EmbedLayer,
    ids_cache: Vec<usize>, // flattened token ids of the last forward pass (for the embedding gradient)
    pe: DMatrix<f64>,      // cached positional encodings
}
 
impl Embedder {
    pub fn new(embed_layer: EmbedLayer) -> Embedder {
        Embedder { embed_layer, ids_cache: Vec::new(), pe: build_positional_table(64) }
    }
 
    pub fn embed_table(&self) -> &DMatrix<f64> {
        self.embed_layer.table()
    }
 
    pub fn params_mut(&mut self) -> Vec<&mut Param> {
        vec![&mut self.embed_layer.table]
    }
 
    pub fn embed_forward(&mut self, input_ids: &DMatrix<usize>) -> DMatrix<f64> {
        let batch_size = input_ids.nrows();
        let l = input_ids.ncols();
        if self.pe.nrows() < l {
            self.pe = build_positional_table(l.max(64));
        }
 
        let mut output = DMatrix::<f64>::zeros(batch_size * l, D_MODEL);
        self.ids_cache.clear();
        for row in 0..batch_size {
            for col in 0..l {
                let id = input_ids[(row, col)];
                self.ids_cache.push(id);
                let r = row * l + col;
                for k in 0..D_MODEL {
                    output[(r, k)] = self.embed_layer.table.w[(id, k)] + self.pe[(col, k)];
                }
            }
        }
        output
    }
 
    /// Scatter-add the input gradient into the embedding table's gradient.
    pub fn backwards(&mut self, dldx: &DMatrix<f64>) {
        for (r, &id) in self.ids_cache.iter().enumerate() {
            for k in 0..D_MODEL {
                self.embed_layer.table.g[(id, k)] += dldx[(r, k)];
            }
        }
    }
 
    pub fn project_decoder_out(&self, decoder_output: DMatrix<f64>) -> DMatrix<f64> {
        decoder_output * self.embed_layer.table.w.transpose()
    }
}
 
// ═════════════════════════════ Layer norm ═════════════════════════════
 
pub struct Normalizer {
    gamma: Param, // 1 x D_MODEL
    beta: Param,  // 1 x D_MODEL
    sigma_inverse_cache: Vec<f64>,
    x_hat_cache: DMatrix<f64>,
}
 
impl Normalizer {
    pub fn new() -> Normalizer {
        Normalizer {
            gamma: Param::new(DMatrix::from_element(1, D_MODEL, 1.0), false),
            beta: Param::new(DMatrix::zeros(1, D_MODEL), false),
            sigma_inverse_cache: Vec::new(),
            x_hat_cache: DMatrix::zeros(0, D_MODEL),
        }
    }
 
    pub fn get_weights(&self) -> (DVector<f64>, DVector<f64>) {
        (
            DVector::from_iterator(D_MODEL, self.gamma.w.iter().cloned()),
            DVector::from_iterator(D_MODEL, self.beta.w.iter().cloned()),
        )
    }
 
    pub fn from_weights(gamma: DVector<f64>, beta: DVector<f64>) -> Normalizer {
        let d = gamma.len();
        Normalizer {
            gamma: Param::new(DMatrix::from_row_slice(1, d, gamma.as_slice()), false),
            beta: Param::new(DMatrix::from_row_slice(1, d, beta.as_slice()), false),
            sigma_inverse_cache: Vec::new(),
            x_hat_cache: DMatrix::zeros(0, d),
        }
    }
 
    pub fn params_mut(&mut self) -> Vec<&mut Param> {
        vec![&mut self.gamma, &mut self.beta]
    }
 
    pub fn normalize_forward(&mut self, x: DMatrix<f64>) -> DMatrix<f64> {
        let (n, d) = x.shape();
        let df = d as f64;
        let mut x_hat = DMatrix::<f64>::zeros(n, d);
        let mut out = DMatrix::<f64>::zeros(n, d);
        self.sigma_inverse_cache = vec![0.0; n];
 
        for i in 0..n {
            let mut mu = 0.0;
            for j in 0..d {
                mu += x[(i, j)];
            }
            mu /= df;
            let mut var = 0.0;
            for j in 0..d {
                let t = x[(i, j)] - mu;
                var += t * t;
            }
            var /= df;
            let si = 1.0 / (var + LN_EPS).sqrt();
            self.sigma_inverse_cache[i] = si;
            for j in 0..d {
                let h = (x[(i, j)] - mu) * si;
                x_hat[(i, j)] = h;
                out[(i, j)] = h * self.gamma.w[(0, j)] + self.beta.w[(0, j)];
            }
        }
        self.x_hat_cache = x_hat;
        out
    }
 
    /// Accumulates gamma/beta gradients, returns dL/dX.
    /// dx = (1/sigma) * ( dxh - mean(dxh) - xh * mean(dxh * xh) ),   dxh = dy * gamma
    pub fn backward_norm(&mut self, dldy: &DMatrix<f64>) -> DMatrix<f64> {
        let (n, d) = dldy.shape();
        let df = d as f64;
        let mut dldx = DMatrix::<f64>::zeros(n, d);
 
        for i in 0..n {
            let si = self.sigma_inverse_cache[i];
            let mut sum_dxh = 0.0;
            let mut sum_dxh_xh = 0.0;
            for j in 0..d {
                let xh = self.x_hat_cache[(i, j)];
                let dy = dldy[(i, j)];
                self.gamma.g[(0, j)] += dy * xh;
                self.beta.g[(0, j)] += dy;
                let dxh = dy * self.gamma.w[(0, j)];
                sum_dxh += dxh;
                sum_dxh_xh += dxh * xh;
            }
            for j in 0..d {
                let xh = self.x_hat_cache[(i, j)];
                let dxh = dldy[(i, j)] * self.gamma.w[(0, j)];
                dldx[(i, j)] = si * (dxh - sum_dxh / df - xh * sum_dxh_xh / df);
            }
        }
        dldx
    }
}
 
// ═════════════════════════════ Encoder ═════════════════════════════
 
pub struct EncoderLayer {
    mha: MHA,
    norm_mha: Normalizer,
    ffn: FeedForward,
    norm_ffn: Normalizer,
    drop_mha: Dropout,
    drop_ffn: Dropout,
}
 
impl EncoderLayer {
    pub fn new(
        mha: MHA,
        norm_mha: Normalizer,
        ffn: FeedForward,
        norm_ffn: Normalizer,
        dropout: f64,
    ) -> EncoderLayer {
        EncoderLayer {
            mha,
            norm_mha,
            ffn,
            norm_ffn,
            drop_mha: Dropout::new(dropout),
            drop_ffn: Dropout::new(dropout),
        }
    }
    pub fn get_mha(&self) -> &MHA { &self.mha }
    pub fn get_ffn(&self) -> &FeedForward { &self.ffn }
    pub fn get_norm_mha(&self) -> &Normalizer { &self.norm_mha }
    pub fn get_norm_ffn(&self) -> &Normalizer { &self.norm_ffn }
 
    pub fn params_mut(&mut self) -> Vec<&mut Param> {
        let mut v = self.mha.params_mut();
        v.extend(self.norm_mha.params_mut());
        v.extend(self.ffn.params_mut());
        v.extend(self.norm_ffn.params_mut());
        v
    }
 
    pub fn forward(&mut self, x: DMatrix<f64>, max_seq_length: usize, mask: &[bool], training: bool) -> DMatrix<f64> {
        // sublayer 1: x -> LN(x + Dropout(MHA(x)))
        let a = self.mha.mha_forward(&x, max_seq_length, mask);
        let a = self.drop_mha.forward(a, training);
        let h1 = self.norm_mha.normalize_forward(a + x);
 
        // sublayer 2: h1 -> LN(h1 + Dropout(FFN(h1)))
        let f = self.ffn.ffn(h1.clone());
        let f = self.drop_ffn.forward(f, training);
        self.norm_ffn.normalize_forward(f + h1)
    }
 
    pub fn backwards(&mut self, dldo: DMatrix<f64>) -> DMatrix<f64> {
        let d_sum2 = self.norm_ffn.backward_norm(&dldo); // grad wrt (f + h1)
        let d_f = self.drop_ffn.backward(d_sum2.clone());
        let d_h1 = self.ffn.backwards(d_f) + d_sum2; // ffn path + residual path
 
        let d_sum1 = self.norm_mha.backward_norm(&d_h1); // grad wrt (a + x)
        let d_a = self.drop_mha.backward(d_sum1.clone());
        self.mha.backwards(d_a) + d_sum1 // mha path + residual path
    }


}
 
pub struct Encoder {
    layers: Vec<EncoderLayer>,
    input_emedder: Embedder,
    mask: Vec<bool>, // (batch * seq) : true = real token
}
 
impl Encoder {
    pub fn new(input_emedder: Embedder, layers: Vec<EncoderLayer>) -> Encoder {
        Encoder { layers, input_emedder, mask: Vec::new() }
    }
 
    pub fn get_layers(&self) -> &[EncoderLayer] { &self.layers }
    pub fn get_layers_mut(&mut self) -> &mut [EncoderLayer] { &mut self.layers }
    pub fn get_embedder(&self) -> &Embedder { &self.input_emedder }
 
    /// Mask of the last forward pass (true = real token, false = <PAD>).
    pub fn mask(&self) -> &[bool] { &self.mask }
 
    pub fn params_mut(&mut self) -> Vec<&mut Param> {
        let mut v = self.input_emedder.params_mut();
        for l in self.layers.iter_mut() {
            v.extend(l.params_mut());
        }
        v
    }
 
    pub fn forward(&mut self, x: DMatrix<usize>, pad_id: usize, training: bool) -> DMatrix<f64> {
        let (b, l) = x.shape();
        self.mask = (0..b * l).map(|r| x[(r / l, r % l)] != pad_id).collect();
 
        let mut out = self.input_emedder.embed_forward(&x);
        for layer in self.layers.iter_mut() {
            out = layer.forward(out, l, &self.mask, training);
        }
        out
    }
 
    pub fn backwards(&mut self, dldo: DMatrix<f64>) {
        let mut d = dldo;
        for layer in self.layers.iter_mut().rev() {
            d = layer.backwards(d);
        }
        self.input_emedder.backwards(&d);
    }


}
 