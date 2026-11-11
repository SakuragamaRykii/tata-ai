use std::num::FpCategory::Normal;

use crate::math::*;

use nalgebra::*;
use rand::prelude::*;


pub const D_MODEL: usize = 64; // 512 for bigger tests
// dimentionality of a token

pub const N_HEADS: usize = 8;
//number of perspectives the models gonna have on

pub const D_K : usize = D_MODEL / N_HEADS;


pub struct MHA{
    w_q : DMatrix<f64>,
    w_k : DMatrix<f64>,
    w_v : DMatrix<f64>,
    w_o : DMatrix<f64>,

    x_cache: DMatrix<f64>,
    q_cache: DMatrix<f64>,
    k_cache: DMatrix<f64>,
    v_cache: DMatrix<f64>,
    heads_cache: DMatrix<f64>,
    attention_cache: Vec<DMatrix<f64>>, // one (token_count x max_seq_length) matrix per head
}

impl MHA{
    pub fn new(
        w_q : DMatrix<f64>,
        w_k : DMatrix<f64>,
        w_v : DMatrix<f64>,
        w_o : DMatrix<f64>,
    ) -> MHA {
        MHA{
            w_q,
            w_k,
            w_v,
            w_o,
            x_cache: DMatrix::zeros(1, 1),
            q_cache: DMatrix::zeros(1, 1),
            k_cache: DMatrix::zeros(1, 1),
            v_cache: DMatrix::zeros(1, 1),
            heads_cache : DMatrix::zeros(1, 1),
            attention_cache : Vec::new()
        }

    }

    pub fn mha_forward(&mut self, X : &DMatrix<f64>, max_seq_length: usize ) -> DMatrix<f64>{
        // this func does the MHA sublayer block for any given X
        let Q = X * &self.w_q;
        let K = X * &self.w_k;
        let V = X * &self.w_v;

        self.x_cache = X.clone();
        self.q_cache = Q.clone();
        self.k_cache = K.clone();
        self.v_cache = V.clone();

        let token_count = X.nrows();
        self.attention_cache.clear();

        let mut head_attentions = DMatrix::<f64>::zeros(token_count, D_MODEL);

        for head in 0..N_HEADS{
            let col = head * D_K;

            let Q_head = Q.view((0, col), (token_count, D_K)).into_owned();
            let K_head= K.view((0, col), (token_count, D_K)).into_owned();
            let V_head= V.view((0, col), (token_count, D_K)).into_owned();

            let attention_head = self.evaluate_attention(Q_head, K_head, V_head, token_count, max_seq_length);

            for i in 0..token_count{
                for j in 0..D_K{
                    head_attentions[(i, j + col)] = attention_head[(i, j)];

                }
            }


        }
        self.heads_cache = head_attentions.clone();
        let O = head_attentions * &self.w_o;
        O

    }

    pub fn cross_attention_forward(
        &mut self, 
        decoder_x: DMatrix<f64>,
        encoder_out: DMatrix<f64>,
        max_seq_length: usize
    ) -> DMatrix<f64>{
        let Q = decoder_x * &self.w_q;
        let K  = &encoder_out * &self.w_k;
        let V = &encoder_out * &self.w_v;

        let token_count = Q.nrows();
        self.attention_cache.clear();

        let mut head_attentions = DMatrix::<f64>::zeros(token_count, D_MODEL);

        for head in 0..N_HEADS{
            let col = head * D_K;

            let Q_head = Q.view((0, col), (token_count, D_K)).into_owned();
            let K_head= K.view((0, col), (token_count, D_K)).into_owned();
            let V_head= V.view((0, col), (token_count, D_K)).into_owned();

            let attention_head = self.evaluate_attention(Q_head, K_head, V_head, token_count, max_seq_length);

            for i in 0..token_count{
                for j in 0..D_K{
                    head_attentions[(i, j + col)] = attention_head[(i, j)];

                }
            }


        }
        self.heads_cache = head_attentions.clone();
        let O = head_attentions * &self.w_o;
        O
    }

    pub fn evaluate_attention(&mut self, Q : DMatrix<f64>, K : DMatrix<f64>, V : DMatrix<f64>,
         token_count: usize, max_seq_length: usize) -> DMatrix<f64>{

        let dk_sqrt = (D_K as f64).sqrt();
        let num_sentences = token_count / max_seq_length;
        
        let mut result = DMatrix::<f64>::zeros(token_count, D_K );
        let mut attentions = DMatrix::<f64>::zeros(token_count, max_seq_length);

        for sentence in 0..num_sentences{
            // get sentence
            let sentence_start = sentence * max_seq_length;

             let q_block = Q.view((sentence_start, 0),
                                                                        (max_seq_length, D_K)).into_owned();
             let k_block = K.view((sentence_start, 0),
                                                                        (max_seq_length, D_K)).into_owned();
             let v_block = V.view((sentence_start, 0),
                                                                        (max_seq_length, D_K)).into_owned();
            
            
             let mut qk_block = (q_block * k_block.transpose()) / dk_sqrt;
             // works by rows of words where each word is a bunch of numerical tokens
            // and these are blocked via view_mut

            softmax_inplace(&mut qk_block, max_seq_length);
            // softmaxxing the entire block to the entire sentence


            result.view_mut((sentence_start, 0), (max_seq_length, D_K))
            .copy_from(&(&qk_block*v_block));
        
            attentions.view_mut((sentence_start, 0), (max_seq_length, max_seq_length))
            .copy_from(&qk_block);

        }
        self.attention_cache.push(attentions);
        result
    }

    pub fn backwards(&mut self, dldo: DMatrix<f64>, lr: f64) -> DMatrix<f64>{
        let t = self.x_cache.nrows();
        let l = self.attention_cache[0].ncols(); // max_seq_length
        let num_sentences = t / l;
        let dk_sqrt = (D_K as f64).sqrt();

        let dldw_o = self.heads_cache.transpose() * &dldo;
        let dldh = dldo * self.w_o.transpose(); // t x D_MODEL

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

                let dv = a.transpose() * &dh;     // l x D_K
                let da = &dh * v.transpose();     // l x l

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
        let dldw_q = &x_t * &dldq;
        let dldw_k = &x_t * &dldk;
        let dldw_v = &x_t * &dldv;

        let dldx = &dldv * self.w_v.transpose()
                 + &dldk * self.w_k.transpose()
                 + &dldq * self.w_q.transpose();

        self.w_o -= lr * dldw_o;
        self.w_k -= lr * dldw_k;
        self.w_q -= lr * dldw_q;
        self.w_v -= lr * dldw_v;
        dldx
        //dl/dh = dldo * do/dh = dldo * V
        //H = A * V <- this is the gradient from the layer in front
        // dH/dv = A 
        // dl/dO    dl/dv = A^T dl/dO
        // V = X.clone() * &self.w_v   dv/dw_v = X
        // dl/dw_v  = X^TA^T dl/dO
        

        // Qs and Ks
        // A = softmax(S) = a bunch of e^si / sum of j e^sjs
        // da/ds = simplifies to a_i (delta - a_j) which essentially is a negative value matrix for all non diagonal entries
        // Forming jacobian dA/dS = diag(A) - AA^T for each row
        // dO/dA = V  dl/da = dldo * V^T
        // dl/ds = dl/da * da/ds = dldo * V^T(diag(A) - AA^T)
        // S = QK/dk_sqrt
        // ds/dq = K/dk_sqrt
        // ds/dk = Q/dk_sqrt
        // dl/dq = dl/ds * ds/dq = dldo*V^T(diag(A)-AA^T)K / dk_sqrt
        // dl/dk = dl/ds * ds/dk = dldo*V^T(diag(A)-AA^T)^TQ / dk_sqrt


        // Q = X.clone() * &self.w_q;
        // K = X.clone() * &self.w_k;
        // dq/w_q = X     dk/dw_k = X
        // dl/dw_q = dldo*V^T(diag(A)-AA^T)KX / dk_sqrt
        // dl/dw_k = dldo*V^T(diag(A)-AA^T)QX / dk_sqrt

        // O =  H * w_o
        // H = AV 
        // dl/do    do/w_o = H^Tdl/do

    }

}


pub struct FeedForward{
    w1: DMatrix<f64>,
    w2: DMatrix<f64>,
    b1 : RowDVector<f64>,
    b2 : RowDVector<f64>,
    d_ff : usize,
    x_cache: DMatrix<f64>, //k x D_MODEL
    r_cache : DMatrix<f64>, 
    a_cache : DMatrix<f64>,// k x d_ff
    

}

impl FeedForward{
    pub fn new(
        w1: DMatrix<f64>, // D_MODEL x d_ff
        w2: DMatrix<f64>, // d_ff x D_MODEL
        b1 : RowDVector<f64>, // 1 x d_ff
        b2 : RowDVector<f64>, // 1 x D_MODEL
        d_ff : usize,
    ) -> FeedForward{
        FeedForward { 
            w1, 
            w2,
            b1,
            b2,
            d_ff, 
            x_cache: DMatrix::zeros(1, 1),
            r_cache: DMatrix::zeros(1, 1),
            a_cache: DMatrix::zeros(1, 1),
        }
    }


    pub fn ffn(&mut self, X : DMatrix<f64>) -> DMatrix<f64>{
         //FFN(x) = max(0, xW1 + b1)W2 + b2
        // X is of dimension k x D_MODEL and is the output of a MHA evaluation followed by add and norm
        // cache X for backprop
        self.x_cache = X.clone();
        // X*W1 + b1
        let mut ffn_a = X * &self.w1; // k x d_ff
        for mut row in ffn_a.row_iter_mut(){
            row += &self.b1;
        }
        // cache for backprop
        self.a_cache = ffn_a.clone();
        // relu on above
        let ffn_relu = relu(ffn_a);
        self.r_cache = ffn_relu.clone();

        // ffn_relu * W2  IS              k x D_MODEL
        let mut ffn_return = ffn_relu * &self.w2;

        // WE NEED TO ADD b2 ON EACH ROW
        for mut row in ffn_return.row_iter_mut(){
            row += &self.b2;
        }

        ffn_return

        
    }

    pub fn backwards(&mut self, dldo: DMatrix<f64>, lr: f64) -> DMatrix<f64>{
        // O = R * w2 + b2
        
        // dl/db2 = dldo 

        // df/dw2 = R
        // dldw2 = R*dldo

        // R = relu(A)
        // drda = 1 if A > 0 else 0
        // dodr = w2 hence
        // dldr = dldo * w2
        // a = X * w1 + b1
        // dadw1 = X
        // dadb1 = 1
        // dlda = dldr * drda = dldo*w2 if A > 0 else 0

        // dldw1 = X * dlda if A > 0 else 0
        // dldb1 = dldr if A > 0 else 0

        let mut dldb2 = RowDVector::<f64>::zeros(D_MODEL);
        for row in dldo.row_iter() {
            dldb2 += row;
        }
        let dldw2 = self.r_cache.transpose() * &dldo;

        let drda = self.a_cache.clone().map(|x| (x > 0.0) as i64 as f64); // k x dff
        let dlda = (dldo * self.w2.transpose()).component_mul(&drda); // k x d_ff
        let dldw1 = self.x_cache.transpose() * &dlda;

        let mut dldb1 = RowDVector::<f64>::zeros(self.d_ff);
        for row in dlda.row_iter() {
            dldb1 += row;
        }

        let dldx = dlda * self.w1.transpose();

        self.w1 -= lr * dldw1;
        self.w2 -= lr * dldw2;
        self.b1 -= lr * dldb1;
        self.b2 -= lr * dldb2;
        //  d_ff x D_MODEL
        dldx
        // a = X * w1 + b1
        // dadx = w1
        // dldx = dlda * dadx

    }

}

pub struct EmbedLayer{
    table: DMatrix<f64>, //also the weight W for Linear Output Projector. vocab_size * D_MODEL
    vocab_size: usize,

}

impl EmbedLayer{

    pub fn new(vocab_size :usize ) -> EmbedLayer{

        let mut token_embedding_table: DMatrix<f64>
            = DMatrix::zeros(vocab_size, D_MODEL);
        let mut rng = rand::rng();
        token_embedding_table = 
            token_embedding_table.map(|_| 
                rng.random_range(-1.0..1.0));


        EmbedLayer{
            table: token_embedding_table,
            vocab_size,
        }
    }
}



pub struct Embedder{
    batch_size: usize,
    context_length : usize,
    num_classes: usize,
    embed_layer : EmbedLayer,

}

impl Embedder{
    pub fn new(
        batch_size: usize, context_length: usize,
        num_classes: usize,
        embed_layer : EmbedLayer,

    )
    -> Embedder 
    {

        Embedder { batch_size,
            context_length,
            num_classes,
            embed_layer,
        }
        
    }

    pub fn embed_forward(&self, input_ids : &DMatrix<usize>) -> DMatrix<f64> { //, embed_table: DMatrix<f64>) -> DMatrix<f64>{
        //input_ids : batch_size * sentence_length matrix
        let batch_size = input_ids.nrows();
        let max_seq_length : usize = input_ids.ncols();

        let mut output 
        = DMatrix::<f64>::zeros(batch_size*max_seq_length, D_MODEL);

        for row in 0..batch_size{
            for col in 0..max_seq_length{
                let id = input_ids[(row, col)];
                let sequence = self.embed_layer.table.row(id);

                let assign_index = (max_seq_length * row) + col;
                output.set_row(assign_index, &sequence);
            }
        }

        self.add_positional_encodings(&mut output, max_seq_length);

        output
        
    }

    pub fn add_positional_encodings(&self, embeddings: &mut DMatrix<f64>, max_seq_length: usize){//, embed_table: &mut DMatrix<f64>){
        // WE USING THIS FUNCTION SO THAT 
        // WE DIRECTLY ADD ENCODINGS INTO THE OUTPUT OF embed_forward()
        // SO THAT IT RETURNS THE ENCODED OUTPUT
        // SINCE INPUT EMBEDDING AND POSITIONAL ENCODING HAPPEN BACK TO BACK ALWAYS
        let nrows = embeddings.nrows();  
        let ncols = embeddings.ncols();
        let denom_base : f64 = 10000.0;

        for row in 0..nrows{
            for col in 0..ncols{
                let pos = (row % max_seq_length) as f64;
                let exp = col as f64 /D_MODEL as f64;
                if(col % 2 == 0){
                    // sin thing
                    let encoding = f64::sin(pos/ denom_base.powf(exp));
                    embeddings[(row, col)] += encoding;
                }
                else{
                    //cos thing
                    let encoding = f64::cos(pos/ denom_base.powf(exp));
                    embeddings[(row, col)] += encoding;
                }
            }
        }
    }

    pub fn project_decoder_out(&self, decoder_output: DMatrix<f64>) -> DMatrix<f64>{
        decoder_output * self.embed_layer.table.transpose()
    }

}


pub struct Normalizer{
    gamma : DVector<f64>, // rescale (can be learned via backwards)
    beta : DVector<f64>, // shift (can be learned via backwards)
    sigma_inverse_cache: DVector<f64>,
    x_hat_cache: DMatrix<f64>,
    
}

impl Normalizer{

    pub fn new() -> Normalizer{
        Normalizer{
            gamma: DVector::<f64>::repeat(D_MODEL, 1.0),
            beta : DVector::<f64>::zeros(D_MODEL),
            sigma_inverse_cache: DVector::<f64>::zeros(0),
            x_hat_cache: DMatrix::<f64>::zeros(0, D_MODEL),
        }
    }


    pub fn normalize_forward(&mut self, X : DMatrix<f64>) -> DMatrix<f64>{
        // Input X is always gonna be the output of a sublayer added with their own input
        let mut output = DMatrix::<f64>::zeros(X.nrows(), X.ncols());

        let x_rows = X.nrows();
        // conditional rebase of cache
        if self.x_hat_cache.nrows() != x_rows{
            self.x_hat_cache = DMatrix::<f64>::zeros(x_rows, D_MODEL);
            self.sigma_inverse_cache = DVector::<f64>::zeros(x_rows);
        }


        for i in 0..x_rows{
            let x = X.row(i);

            // ---  evaluate x_hat ---
            let mu = x.mean();
            let sigma_inverse = 1.0 / (x.variance() + 1e-3).sqrt();  // just 1/sigma with a slack
            let x_hat = x.map(|k| (k - mu) * sigma_inverse);      
            //--- caching ---
            self.sigma_inverse_cache[i] = sigma_inverse;
            self.x_hat_cache.row_mut(i).copy_from(&x_hat); 

            let y = x_hat.component_mul(&self.gamma.transpose()) + self.beta.transpose();
            // We use component_mul with x_hat and gamma because we want to keep y as a 1 * N column vector
            // cuz if just multiply x_hat with gamma normally it becomes an N * N matrix which gets applied to 
            // each row of output (line below me) and that makes no sense
            output.row_mut(i).copy_from(&y);
        }
        output
    }


    pub fn backward_norm(&self, X : DMatrix<f64>, dldy : DMatrix<f64>) 
    -> (DMatrix<f64>, DVector<f64>, DVector<f64>) // for updating X, gamma and beta
    {
        let mut dldgamma = DVector::<f64>::zeros(D_MODEL);
        let mut dldbeta = DVector::<f64>::zeros(D_MODEL);

        let num_tokens = X.nrows();
        let token_size = X.ncols();
        let mut dldx = DMatrix::<f64>::zeros(num_tokens, token_size);

        let D_MODEL_float = D_MODEL as f64;

        for i in 0..num_tokens{
            let dldy_row = dldy.row(i);
            let x = X.row(i);
            let x_hat = self.x_hat_cache[i];

            // for dldx
            let denominator = 1. * self.sigma_inverse_cache[i] / D_MODEL_float;

            let mut a = vec![0.0; D_MODEL]; // actually more efficient than initializing DVector Zeros
            let mut b = 0.0;
            let mut c_sum = 0.0;

            for j in 0..D_MODEL{
                dldgamma[j] += dldy[(i, j)] * self.x_hat_cache[j];
                dldbeta[j] += dldy[(i, j)] ;

                let dldx_hat_j = dldy_row[j] * self.gamma[j];

                a[j] = dldx_hat_j * D_MODEL_float;
                b += dldx_hat_j;
                c_sum += dldx_hat_j * self.x_hat_cache[j];

            }       

            for j in 0..D_MODEL{
                let c = c_sum * self.x_hat_cache[j];
                dldx[(i, j)] = (a[j] - b - c) * denominator;
            }

        }

        (dldx, dldgamma, dldbeta)

    }


}

pub struct EncoderLayer{
    mha : MHA,
    norm_mha : Normalizer,
    ffn : FeedForward,
    norm_ffn : Normalizer,

    sub_layer_mha : DMatrix<f64>,
    normalized_mha : DMatrix<f64>,
    feed : DMatrix<f64>,
    x_cache : DMatrix<f64>
}

impl EncoderLayer{
    pub fn new(
        mha : MHA,
        norm_mha : Normalizer,
        ffn : FeedForward,
        norm_ffn : Normalizer,


    ) -> EncoderLayer{
        EncoderLayer {mha, norm_mha, ffn, norm_ffn,

        sub_layer_mha : DMatrix::zeros(1, 1),
        normalized_mha : DMatrix::zeros(1, 1),
        feed : DMatrix::zeros(1, 1),
        x_cache : DMatrix::zeros(1, 1),
        
        }
    }

    pub fn forward(&mut self, X : DMatrix<f64>, max_seq_length: usize) -> DMatrix<f64>{
    
        //sublayer1
        let sub_layer_mha = self.mha.mha_forward(&X, max_seq_length) + X;
        self.sub_layer_mha = sub_layer_mha.clone();
        let normalized_mha_sub_layer = self.norm_mha.normalize_forward(sub_layer_mha);
        self.normalized_mha = normalized_mha_sub_layer.clone();
        //sublayer2 or output
        let feed = self.ffn.ffn(normalized_mha_sub_layer.clone()) + normalized_mha_sub_layer;
        self.feed = feed.clone();
        let normalized_ffn_sub_layer = self.norm_ffn.normalize_forward(feed);

        normalized_ffn_sub_layer
    }

    pub fn backwards(&mut self, dldo : DMatrix<f64>, lr: f64) -> DMatrix<f64>{
        // dldn2 = 
        let (dldfeed, dldgamma, dldbeta) = self.norm_ffn.backward_norm(self.feed.clone(), dldo);
        //dldn1 = dldffn + dldn1res
        let dldffn = dldfeed.clone();
        let dldn1res = dldfeed;
        let dldx_ffn = self.ffn.backwards(dldffn.clone(), lr);
        let dldn1 = dldx_ffn + dldn1res;
        //dldx is the same thing and we gonna feed it to the previous layer
        let (dldmhafeed, dldgamma_2, dldbeta_2) = self.norm_mha.backward_norm(self.sub_layer_mha.clone(), dldn1);
        let dldx_mha = self.mha.backwards(dldmhafeed.clone(), lr);
        let dldx = dldx_mha + dldmhafeed;
        dldx
    }

}

pub struct Encoder{
    layers : Vec<EncoderLayer>,
    input_emedder : Embedder,
}
impl Encoder{

    pub fn new(
        input_emedder : Embedder,
        layers : Vec<EncoderLayer>,
    ) -> Encoder{
        Encoder { layers, input_emedder }
    }
    // pub fn new(
    //     input_emedder : Embedder,
    //     N : usize
    // ) -> Encoder{
    //  // construct N empty layers 
    // }


    pub fn forward(&mut self, X : DMatrix<usize>, max_seq_length: usize) -> DMatrix<f64>{
        let embed_x = self.input_emedder.embed_forward(&X);

        let mut layer_output = embed_x;
        for layer in self.layers.iter_mut(){
            layer_output = layer.forward(layer_output, max_seq_length);
        }
        layer_output
    }

    pub fn backwards(&mut self, dldo: DMatrix<f64>, lr: f64) -> DMatrix<f64>{
        let mut current_dldo = dldo;
        for layer in self.layers.iter_mut().rev(){
            let dldx_layer = layer.backwards(current_dldo, lr);
            current_dldo = dldx_layer;
        }
        current_dldo
    }
}


// pub struct DecoderLayer{
//     masked_mha: MHA,
//     norm1: Normalizer,
//     encoder_mha: MHA,
//     norm2: Normalizer,
//     ffn: FeedForward,
//     norm3: Normalizer,


// }

// impl DecoderLayer{
//     pub fn new(
//         masked_mha: MHA,
//         norm1: Normalizer,
//         encoder_mha: MHA,
//         norm2: Normalizer,
//         ffn: FeedForward,
//         norm3: Normalizer,    
//     ) -> DecoderLayer{
//         DecoderLayer { masked_mha, norm1, encoder_mha, norm2, ffn, norm3 }
//     }




//     pub fn forward(&mut self, X : DMatrix<f64>, encoder_out : DMatrix<f64>,max_seq_length: usize) -> DMatrix<f64>{
//         // Q and K come from the output of Encoder

//         //sublayer 1
//         let masked_mha_sub_layer = self.masked_mha.mha_forward(X.clone(), max_seq_length) + X;
//         let masked_mha_normalized = self.norm1.normalize_forward(masked_mha_sub_layer);

//         //sublayer2
//         let encoder_mha_sub_layer = self.encoder_mha.cross_attention_forward(
//             masked_mha_normalized.clone(),
//             encoder_out,
//             max_seq_length
//         ) + masked_mha_normalized;
//         let encoder_mha_normalized = self.norm2.normalize_forward(encoder_mha_sub_layer);

//         //sublayer3
//         let feed_sub_layer = self.ffn.ffn(encoder_mha_normalized.clone()) + encoder_mha_normalized;
//         let feed_normalized = self.norm3.normalize_forward(feed_sub_layer);

//         feed_normalized
//     }

// }

// pub struct Decoder{
//     layers: Vec<DecoderLayer>,
//     output_emedder : Embedder,
//     encoder_out: DMatrix<f64>,
// }
// impl Decoder{
//     pub fn new(
//         layers: Vec<DecoderLayer>,
//         output_emedder : Embedder,
//         encoder_out: DMatrix<f64>,
//     ) -> Decoder{
//         Decoder { layers, output_emedder, encoder_out }
//     }


//     pub fn forward(&mut self, X : DMatrix<usize>, max_seq_length: usize) -> DMatrix<f64>{
//         let embed_x = self.output_emedder.embed_forward(&X);

//         let mut layer_output = embed_x;
//         for layer in self.layers.iter(){
//             layer_output = layer.forward(layer_output, self.encoder_out.clone(), max_seq_length);
//         }
//         // THIS IS NOT THE END OF THIS 
//         // NEXT SESH:
//         // make the linear block which is the output projection
//         // which basically uses layer_output to get actual token predictions
//         // Once we get this we can softmax it to get the probabilies and then
//         // THAT becomes the output of this function
//         softmax(self.output_emedder.project_decoder_out(layer_output))
//     }


// }


// pub struct Transformer{
//     encoder: Encoder,
//     decoder: Decoder,

// }

// impl Transformer{
//     pub fn new(n : usize, encoder_empty: Encoder, decoder_empty : Decoder){

        

//         for i in 0..n{

//         }

// FOR ENCODER
// mha : MHA,
// norm_mha : Normalizer,
// ffn : FeedForward,
// norm_ffn : Normalizer,

// FOR DECODER
//    masked_mha: MHA,
//     norm1: Normalizer,
//     encoder_mha: MHA,
//     norm2: Normalizer,
//     ffn: FeedForward,
//     norm3: Normalizer,


// INITIALIZE THE ABOVE N TIMES