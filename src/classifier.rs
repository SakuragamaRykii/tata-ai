use std::print;

use nalgebra::{DMatrix, RowDVector};
use rand::{RngExt, rng};
use serde::{Deserialize, Serialize};

use crate::{classifier, math::*, preprocessing::Tokenizer, transformer::*};

// ---- serializable weight structs ----

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
    pub vocab: Vec<String>,          // inverse_vocab
    pub merge_history: Vec<(String, String)>,
    pub layers: Vec<LayerWeights>,
}


    // mha : MHA,
    // norm_mha : Normalizer,
    // ffn : FeedForward,
    // norm_ffn : Normalizer,
    // PER LAYEr

        // input_emedder : Embedder, for the encoder


pub struct Classifier{
    num_classes: usize,
    n: usize,
    w: DMatrix<f64>, // D_MODEL x num_classes, for crunching encoder output into class
    b: RowDVector<f64>, // 1 x num_classes. 
    encoder: Encoder,
    c_cache : DMatrix<f64>,
    output_cache : DMatrix<f64>,
    seq_len : usize,
}

impl Classifier{

    pub fn empty(num_classes: usize, n: usize, vocab_size: usize, batch_size: usize, context_length: usize, ) -> Classifier{
        //Innit w and b
        let w = xavier_init(D_MODEL, num_classes);
        let b = RowDVector::<f64>::zeros(num_classes);

        // Innit Encoder
        let mut encoder_layers = Vec::<EncoderLayer>::with_capacity(n);
        
        for i in 0..n{
            // INIT MHA
            let mha = MHA::new(
                xavier_init(D_MODEL, D_MODEL),
                xavier_init(D_MODEL, D_MODEL),
                xavier_init(D_MODEL, D_MODEL),
                xavier_init(D_MODEL, D_MODEL),
            );
            let norm_mha = Normalizer::new();

        // w1: DMatrix<f64>, // D_MODEL x d_ff
        // w2: DMatrix<f64>, // d_ff x D_MODEL
        // b1 : RowDVector<f64>, // 1 x d_ff
        // b2 : RowDVector<f64>, // 1 x D_MODEL

            // INIT FFN
            let d_ff: usize = D_MODEL * 4;

            let ffn = FeedForward::new(
                xavier_init(D_MODEL, d_ff),
                xavier_init(d_ff, D_MODEL),
                RowDVector::<f64>::zeros(d_ff),
                RowDVector::<f64>::zeros(D_MODEL), 
                d_ff);
            let norm_ffn = Normalizer::new();

            let layer = EncoderLayer::new(
                mha,
                norm_mha,
                ffn,
                norm_ffn);
            encoder_layers.push(layer);

        }

        let embed_layer = EmbedLayer::new(vocab_size);
        let embedder = Embedder::new(batch_size, context_length, num_classes, embed_layer);

        let encoder = Encoder::new(embedder, encoder_layers);
        let output_cache = DMatrix::<f64>::zeros(3, 1)
;
        Classifier{num_classes, n, w, b, encoder, output_cache, c_cache : DMatrix::<f64>::zeros(1, 1), seq_len : context_length}
    }

    pub fn forward(&mut self, inputs: DMatrix<usize>) -> DMatrix<f64>{
        // inputs should be preprocessed input sentences
        // max seq length to be predetermined?
        let batch_size = inputs.nrows();
        let max_seq_length = inputs.ncols();
        self.seq_len = max_seq_length;

        let encode = self.encoder.forward(inputs, max_seq_length);
        //outputs (batch_size*max_seq_length) x D_MODEL
        //so we gonna crunch this to a bunch of num_classes x D_MODEL
        let mut cls_accum = DMatrix::<f64>::zeros(batch_size, D_MODEL);

        for i in 0..batch_size{
            let idx = i * max_seq_length;
            let cls = encode.row(idx);
            //
            cls_accum.set_row(i, &cls);
        }
        self.c_cache = cls_accum.clone();
        let mut cls_applied = cls_accum * &self.w;
        for mut row in cls_applied.row_iter_mut(){row += &self.b;}

        softmax_inplace(&mut cls_applied, batch_size);
        self.output_cache = cls_applied.clone();
        cls_applied

        //[0.5, 0.2, 0.3] actual idx = 2 L =  - log( 0.3)

    }


    pub fn backwards(&mut self, actual_class: usize, sample_weight: f64, lr: f64, lambda:f64){
        // single-sentence (batch = 1) training step

        // softmax + cross-entropy: dL/dz = p - y_onehot
        let mut dldz = self.output_cache.clone();
        dldz[(0, actual_class)] -= 1.0;
        dldz *= sample_weight;

        // gradient norm clipping
        let norm = dldz.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1.0 { dldz /= norm; }

        let dldw = self.c_cache.transpose() * &dldz;
        let mut dldb = RowDVector::<f64>::zeros(self.num_classes);
        for row in dldz.row_iter() { dldb += row; }
        let dldc = &dldz * self.w.transpose(); // computed BEFORE w is updated

        self.w -= lr * l2_grad(&dldw, &self.w, lambda);
        self.b -= lr * dldb;

        // only the <CLS> row of each sentence receives gradient; shape must match encoder output
        let mut dldo = DMatrix::<f64>::zeros(dldc.nrows() * self.seq_len, D_MODEL);
        for i in 0..dldc.nrows() {
            dldo.row_mut(i * self.seq_len).copy_from(&dldc.row(i));
        }
        self.encoder.backwards(dldo, lr, lambda);
    }


    pub fn to_weights(&self, tok:&Tokenizer) -> ClassifierWeights {
        let layers: Vec<LayerWeights> = self.encoder.get_layers().iter().map(|l| {
            let (wq, wk, wv, wo) = l.get_mha().get_weights();
            let (w1, w2, b1, b2, d_ff) = l.get_ffn().get_weights();
            let (g1, b1n) = l.get_norm_mha().get_weights();
            let (g2, b2n) = l.get_norm_ffn().get_weights();
            LayerWeights {
                mha: MhaWeights {
                    w_q: mat_to_vec(&wq), w_k: mat_to_vec(&wk),
                    w_v: mat_to_vec(&wv), w_o: mat_to_vec(&wo),
                },
                norm_mha: NormWeights { gamma: dvec_to_vec(&g1), beta: dvec_to_vec(&b1n) },
                ffn: FfnWeights {
                    w1: mat_to_vec(&w1), w2: mat_to_vec(&w2),
                    b1: row_to_vec(&b1), b2: row_to_vec(&b2), d_ff,
                },
                norm_ffn: NormWeights { gamma: dvec_to_vec(&g2), beta: dvec_to_vec(&b2n) },
            }
        }).collect();

        ClassifierWeights {
            num_classes: self.num_classes,
            n: self.n,
            w: mat_to_vec(&self.w),
            b: row_to_vec(&self.b),
            embed_table: mat_to_vec(self.encoder.get_embedder().embed_table()),
            vocab: tok.inverse_vocab.clone(),
            merge_history: tok.merge_history.clone(),
            layers,
        }
    }

    /// Rebuild a Classifier from saved weights. `seq_len` must match what was used at training.
    pub fn from_weights(cw: &ClassifierWeights, seq_len: usize) -> Classifier {
        let embed_layer = EmbedLayer::from_table(vec_to_mat(&cw.embed_table));
        // Embedder does NOT take a tokenizer — 4 args only
        let embedder = Embedder::new(1, seq_len, cw.num_classes, embed_layer);

        let mut encoder_layers = Vec::with_capacity(cw.n);
        for l in &cw.layers {
            let mha = MHA::new(
                vec_to_mat(&l.mha.w_q), vec_to_mat(&l.mha.w_k),
                vec_to_mat(&l.mha.w_v), vec_to_mat(&l.mha.w_o),
            );
            let norm_mha = Normalizer::from_weights(
                vec_to_dvec(&l.norm_mha.gamma), vec_to_dvec(&l.norm_mha.beta));

            let ffn = FeedForward::new(
                vec_to_mat(&l.ffn.w1), vec_to_mat(&l.ffn.w2),
                vec_to_row(&l.ffn.b1), vec_to_row(&l.ffn.b2), l.ffn.d_ff);
            let norm_ffn = Normalizer::from_weights(
                vec_to_dvec(&l.norm_ffn.gamma), vec_to_dvec(&l.norm_ffn.beta));

            encoder_layers.push(EncoderLayer::new(mha, norm_mha, ffn, norm_ffn));
        }

        let encoder = Encoder::new(embedder, encoder_layers);
        Classifier {
            num_classes: cw.num_classes,
            n: cw.n,
            w: vec_to_mat(&cw.w),
            b: vec_to_row(&cw.b),
            encoder,
            c_cache: DMatrix::<f64>::zeros(1, 1),
            output_cache: DMatrix::<f64>::zeros(3, 1),
            seq_len,
        }
    }

}


fn xavier_init(x: usize, y: usize) -> DMatrix<f64> {
    let mut rng = rng();
    let a = (6.0 / (x + y) as f64).sqrt();
    DMatrix::<f64>::zeros(x, y).map(|elem| rng.random_range(-a..a))
}