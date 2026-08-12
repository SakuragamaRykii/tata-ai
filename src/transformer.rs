
use std::num::FpCategory::Normal;

use crate::math::*;

use nalgebra::{DMatrix, DVector, RowDVector,  coordinates::X, iter::MatrixIter};
use rand::prelude::*;


pub struct MHA{
    n_heads: usize,
    d_k : usize,
    d_model: usize,

    w_q : DMatrix<f64>,
    w_k : DMatrix<f64>,
    w_v : DMatrix<f64>,
    w_o : DMatrix<f64>,
}

impl MHA{
    pub fn new(
        n_heads: usize,
        d_k : usize,
        d_model: usize,

        w_q : DMatrix<f64>,
        w_k : DMatrix<f64>,
        w_v : DMatrix<f64>,
        w_o : DMatrix<f64>,
    ) -> MHA {
        MHA{
            n_heads,
            d_k,
            d_model,
            w_q,
            w_k,
            w_v,
            w_o,
        }

    }

    pub fn mha_forward(&self, X : DMatrix<f64>, max_seq_length: usize ) -> DMatrix<f64>{
        // this func does the MHA sublayer block for any given X
        let Q = X.clone() * &self.w_q;
        let K = X.clone() * &self.w_k;
        let V = X.clone() * &self.w_v;

        let token_count = X.nrows();

        let mut head_attentions = DMatrix::<f64>::zeros(token_count, self.d_model);

        for head in 0..self.n_heads{
            let col = head * self.d_k;

            let Q_head = Q.view((0, col), (token_count, self.d_k)).into_owned();
            let K_head= K.view((0, col), (token_count, self.d_k)).into_owned();
            let V_head= V.view((0, col), (token_count, self.d_k)).into_owned();

            let attention_head = self.evaluate_attention(Q_head, K_head, V_head, token_count, max_seq_length);

            for i in 0..token_count{
                for j in 0..self.d_k{
                    head_attentions[(i, j + col)] = attention_head[(i, j)];

                }
            }


        }
        
        let O = head_attentions * &self.w_o;
        O

    }

    pub fn cross_attention_forward(
        &self, 
        decoder_x: DMatrix<f64>,
        encoder_out: DMatrix<f64>,
        max_seq_length: usize
    ) -> DMatrix<f64>{
        let Q = decoder_x * &self.w_q;
        let K = encoder_out.clone() * &self.w_k;
        let V = encoder_out.clone() * &self.w_v;

        let token_count = Q.nrows();

        let mut head_attentions = DMatrix::<f64>::zeros(token_count, self.d_model);

        for head in 0..self.n_heads{
            let col = head * self.d_k;

            let Q_head = Q.view((0, col), (token_count, self.d_k)).into_owned();
            let K_head= K.view((0, col), (token_count, self.d_k)).into_owned();
            let V_head= V.view((0, col), (token_count, self.d_k)).into_owned();

            let attention_head = self.evaluate_attention(Q_head, K_head, V_head, token_count, max_seq_length);

            for i in 0..token_count{
                for j in 0..self.d_k{
                    head_attentions[(i, j + col)] = attention_head[(i, j)];

                }
            }


        }
        
        let O = head_attentions * &self.w_o;
        O
    }

    pub fn evaluate_attention(&self, Q : DMatrix<f64>, K : DMatrix<f64>, V : DMatrix<f64>,
         token_count: usize, max_seq_length: usize) -> DMatrix<f64>{

        let dk_sqrt = (self.d_k as f64).sqrt();
        let num_sentences = token_count / max_seq_length;
        
        let mut result = DMatrix::<f64>::zeros(token_count, self.d_k );

        for sentence in 0..num_sentences{
            // get sentence
            let sentence_start = sentence * max_seq_length;

             let q_block = Q.view((sentence_start, 0),
                                                                        (max_seq_length, self.d_k)).into_owned();
             let k_block = K.view((sentence_start, 0),
                                                                        (max_seq_length, self.d_k)).into_owned();
             let v_block = V.view((sentence_start, 0),
                                                                        (max_seq_length, self.d_k)).into_owned();
            
            
             let mut qk_block = (q_block * k_block.transpose()) / dk_sqrt;
             // works by rows of words where each word is a bunch of numerical tokens
            // and these are blocked via view_mut

            softmax_inplace(&mut qk_block, max_seq_length);
            // softmaxxing the entire block so the entire sentence


            result.view_mut((sentence_start, 0), (max_seq_length, self.d_k))
            .copy_from(&(qk_block*v_block));

        }

        result
    }



}


pub struct FeedForward{
    w1: DMatrix<f64>,
    w2: DMatrix<f64>,
    b1 : RowDVector<f64>,
    b2 : RowDVector<f64>,
    d_ff : usize,
    d_model : usize,
}

impl FeedForward{
    pub fn new(
        w1: DMatrix<f64>, // d_model x d_ff
        w2: DMatrix<f64>, // d_ff x d_model
        b1 : RowDVector<f64>, // 1 x d_ff
        b2 : RowDVector<f64>, // 1 x d_model
        d_ff : usize,
        d_model : usize,
    ) -> FeedForward{
        FeedForward { w1, w2, b1, b2, d_ff, d_model }
    }


    pub fn ffn(&self, X : DMatrix<f64>) -> DMatrix<f64>{
         //FFN(x) = max(0, xW1 + b1)W2 + b2
        // X is of dimension k x d_model and is the output of a MHA evaluation followed by add and norm

        // X*W1 + b1
        let mut ffn_a = X * &self.w1;
        for mut row in ffn_a.row_iter_mut(){
            row += &self.b1;
        }
        // relu on above
        let ffn_relu = relu(ffn_a);

        // ffn_relu * W2  IS              k x d_model
        let mut ffn_return = ffn_relu * &self.w2;

        // WE NEED TO ADD b2 ON EACH ROW
        for mut row in ffn_return.row_iter_mut(){
            row += &self.b2;
        }

        ffn_return

        
    }

}

pub struct EmbedLayer{
    table: DMatrix<f64>,
    d_model: usize,
    vocab_size: usize,

}

impl EmbedLayer{

    pub fn new(d_model:usize, vocab_size :usize ) -> EmbedLayer{

        let mut token_embedding_table: DMatrix<f64>
            = DMatrix::zeros(vocab_size, d_model);
        let mut rng = rand::rng();
        token_embedding_table = 
            token_embedding_table.map(|_| 
                rng.random_range(-1.0..1.0));


        EmbedLayer{
            table: token_embedding_table,
            d_model,
            vocab_size,
        }
    }
}



pub struct Embedder{
    batch_size: usize,
    context_length : usize,
    d_model: usize,
    n_heads: usize,
    d_k: usize, // d_model / n_heads
    num_classes: usize,
    embed_layer : EmbedLayer,

}

impl Embedder{
    pub fn new(
        batch_size: usize, context_length: usize,
        d_model: usize, n_heads: usize, 
        num_classes: usize,
        embed_layer : EmbedLayer,

    )
    -> Embedder 
    {

        Embedder { batch_size,
            context_length,
            d_model, 
            n_heads,
            d_k: d_model/n_heads, 
            num_classes,
            embed_layer,
        }
        
    }

    pub fn embed_forward(&self, input_ids : &DMatrix<usize>) -> DMatrix<f64> { //, embed_table: DMatrix<f64>) -> DMatrix<f64>{
        //input_ids : batch_size * sentence_length matrix
        let batch_size = input_ids.nrows();
        let max_seq_length : usize = input_ids.ncols();

        let mut output 
        = DMatrix::<f64>::zeros(batch_size*max_seq_length, self.d_model);

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
                let exp = col as f64 /self.d_model as f64;
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


}


pub struct EncoderLayer{
    mha : MHA,
    norm_mha : Normalizer,
    ffn : FeedForward,
    norm_ffn : Normalizer,
}

impl EncoderLayer{
    pub fn new(
        mha : MHA,
        norm_mha : Normalizer,
        ffn : FeedForward,
        norm_ffn : Normalizer,
    ) -> EncoderLayer{
        EncoderLayer {mha, norm_mha, ffn, norm_ffn }
    }

    pub fn forward(&self, X : DMatrix<f64>, max_seq_length: usize) -> DMatrix<f64>{
    
        //sublayer1
        let sub_layer_mha = self.mha.mha_forward(X.clone(), max_seq_length) + X;
        let normalized_mha_sub_layer = self.norm_mha.normalize_forward(sub_layer_mha);

        //sublayer2 or output
        let feed = self.ffn.ffn(normalized_mha_sub_layer.clone()) + normalized_mha_sub_layer;
        let normalized_ffn_sub_layer = self.norm_ffn.normalize_forward(feed);

        normalized_ffn_sub_layer
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


    pub fn forward(&self, X : DMatrix<usize>, max_seq_length: usize) -> DMatrix<f64>{
        let embed_x = self.input_emedder.embed_forward(&X);

        let mut layer_output = embed_x;
        for layer in self.layers.iter(){
            layer_output = layer.forward(layer_output, max_seq_length);
        }
        layer_output
    }
}


    // w_q : DMatrix<f64>,
    // w_k : DMatrix<f64>,
    // w_v : DMatrix<f64>,
    // w_o : DMatrix<f64>,
pub struct DecoderLayer{
    masked_mha: MHA,
    norm1: Normalizer,
    encoder_mha: MHA,
    norm2: Normalizer,
    ffn: FeedForward,
    norm3: Normalizer,


}

impl DecoderLayer{
    pub fn new(
        masked_mha: MHA,
        norm1: Normalizer,
        encoder_mha: MHA,
        norm2: Normalizer,
        ffn: FeedForward,
        norm3: Normalizer,    
    ) -> DecoderLayer{
        DecoderLayer { masked_mha, norm1, encoder_mha, norm2, ffn, norm3 }
    }


    pub fn forward(&self, X : DMatrix<f64>, encoder_out : DMatrix<f64>,max_seq_length: usize) -> DMatrix<f64>{
        // Q and K come from the output of Encoder

        //sublayer 1
        let masked_mha_sub_layer = self.masked_mha.mha_forward(X.clone(), max_seq_length) + X;
        let masked_mha_normalized = self.norm1.normalize_forward(masked_mha_sub_layer);

        //sublayer2
        let encoder_mha_sub_layer = self.encoder_mha.cross_attention_forward(
            masked_mha_normalized.clone(),
            encoder_out,
            max_seq_length
        ) + masked_mha_normalized;
        let encoder_mha_normalized = self.norm2.normalize_forward(encoder_mha_sub_layer);

        //sublayer3
        let feed_sub_layer = self.ffn.ffn(encoder_mha_normalized.clone()) + encoder_mha_normalized;
        let feed_normalized = self.norm3.normalize_forward(feed_sub_layer);

        feed_normalized
    }

}

pub struct Decoder{
    layers: Vec<DecoderLayer>,
    output_emedder : Embedder,
    encoder_out: DMatrix<f64>,
}
impl Decoder{
    pub fn new(
        layers: Vec<DecoderLayer>,
        output_emedder : Embedder,
        encoder_out: DMatrix<f64>,
    ) -> Decoder{
        Decoder { layers, output_emedder, encoder_out }
    }

    pub fn forward(&self, X : DMatrix<usize>, max_seq_length: usize) -> DMatrix<f64>{
        let embed_x = self.output_emedder.embed_forward(&X);

        let mut layer_output = embed_x;
        for layer in self.layers.iter(){
            layer_output = layer.forward(layer_output, self.encoder_out.clone(), max_seq_length);
        }
        layer_output
        // THIS IS NOT THE END OF THIS 
        // NEXT SESH:
        // make the linear block which is the output projection
        // which basically uses layer_output to get actual token predictions
        // Once we get this we can softmax it to get the probabilies and then
        // THAT becomes the output of this function
    }

}


// pub fn xavier_init(n_rows : usize, n_cols: usize)
// -> DMatrix<f64>{
//     let mut rng = rand::rng();

//     let limit = ()


// }