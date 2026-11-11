use std::print;

use nalgebra::{DMatrix, RowDVector};
use rand::{RngExt, rng};

use crate::{classifier, math::{softmax, softmax_inplace}, transformer::*};

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


    pub fn backwards(&mut self, actual_class: usize, sample_weight: f64, lr: f64){
        // single-sentence (batch = 1) training step

        // softmax + cross-entropy: dL/dz = p - y_onehot
        let mut dldz = self.output_cache.clone();
        dldz[(0, actual_class)] -= 1.0;
        dldz *= sample_weight;

        let dldw = self.c_cache.transpose() * &dldz;
        let mut dldb = RowDVector::<f64>::zeros(self.num_classes);
        for row in dldz.row_iter() { dldb += row; }
        let dldc = &dldz * self.w.transpose(); // computed BEFORE w is updated

        self.w -= lr * dldw;
        self.b -= lr * dldb;

        // only the <CLS> row of each sentence receives gradient; shape must match encoder output
        let mut dldo = DMatrix::<f64>::zeros(dldc.nrows() * self.seq_len, D_MODEL);
        for i in 0..dldc.nrows() {
            dldo.row_mut(i * self.seq_len).copy_from(&dldc.row(i));
        }
        self.encoder.backwards(dldo, lr);
    }



}


fn xavier_init(x: usize, y: usize) -> DMatrix<f64> {
    let mut rng = rng();
    let a = (6.0 / (x + y) as f64).sqrt();
    DMatrix::<f64>::zeros(x, y).map(|elem| rng.random_range(-a..a))
}