
use crate::math::*;

use nalgebra::{DMatrix, DVector, coordinates::X, iter::MatrixIter};
use rand::prelude::*;


struct MHA{
    n_heads: usize,
    d_k : usize,
    hidden_size: usize,

    w_q : DMatrix<f64>,
    w_k : DMatrix<f64>,
    w_v : DMatrix<f64>,
    w_o : DMatrix<f64>,
}

impl MHA{
    fn mha_forward(&self, X : DMatrix<f64>, max_seq_length: usize ) -> DMatrix<f64>{
        let Q = X.clone() * &self.w_q;
        let K = X.clone() * &self.w_k;
        let V = X.clone() * &self.w_v;

        let token_count = X.nrows();

        let mut head_attentions = DMatrix::<f64>::zeros(token_count, self.hidden_size);

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

    fn evaluate_attention(&self, Q : DMatrix<f64>, K : DMatrix<f64>, V : DMatrix<f64>,
         token_count: usize, max_seq_length: usize) -> DMatrix<f64>{

        let dk_sqrt = (self.d_k as f64).sqrt();
        let num_sentences = token_count / max_seq_length;
        let mut QK = (Q * K.transpose())/ dk_sqrt;

        for sentence in 0..num_sentences{
            // get sentence
            let sentence_start = sentence * max_seq_length;
            let mut sentence_block  = QK.view_mut((sentence_start, sentence_start),
             (max_seq_length, max_seq_length));
            softmax_inplace(sentence_block, max_seq_length);
            // for row in 0..max_seq_length {
            //     let current_row = sentence_block.row_mut(row);
            //     softmax_row_inplace(current_row);
            // }

        }

        QK * V
    }



}

struct EmbedLayer{
    table: DMatrix<f64>,
    hidden_size: usize,
    vocab_size: usize,

}

impl EmbedLayer{

    fn new(&mut self, hidden_size:usize, vocab_size :usize ) -> EmbedLayer{

        let mut token_embedding_table: DMatrix<f64>
            = DMatrix::zeros(vocab_size, hidden_size);
        let mut rng = rand::rng();
        token_embedding_table = 
            token_embedding_table.map(|_| 
                rng.random_range(-1.0..1.0));


        EmbedLayer{
            table: token_embedding_table,
            hidden_size,
            vocab_size,
        }
    }
}



pub struct Transformer{
    batch_size: usize,
    context_length : usize,
    hidden_size: usize,
    n_heads: usize,
    d_k: usize, // hidden_size / n_heads
    num_classes: usize,

}

impl Transformer{
    pub fn new(&mut self, 
        batch_size: usize, context_length: usize,
        hidden_size: usize, n_heads: usize, 
        num_classes: usize)
    -> Transformer
    {

        Transformer { batch_size,
            context_length,
            hidden_size, 
            n_heads,
            d_k: hidden_size/n_heads, 
            num_classes,
        }
        
    }

    pub fn embed_forward(&self, input_ids : &DMatrix<usize>, embed_table: DMatrix<f64>) -> DMatrix<f64>{
        //input_ids : batch_size * sentence_length matrix
        let batch_size = input_ids.nrows();
        let max_seq_length : usize = input_ids.ncols();

        let mut output 
        = DMatrix::<f64>::zeros(batch_size*max_seq_length, self.hidden_size);

        for row in 0..batch_size{
            for col in 0..max_seq_length{
                let id = input_ids[(row, col)];
                let sequence = embed_table.row(id);

                let assign_index = (max_seq_length * row) + col;
                output.set_row(assign_index, &sequence);
            }
        }

        output

    }

    pub fn add_positional_encodings(&mut self, embed_table: &mut DMatrix<f64>){
        let nrows = embed_table.nrows();  
        let ncols = embed_table.ncols();
        let denom_base : f64 = 10000.0;

        for row in 0..nrows{
            for col in 0..ncols{
                let pos = (row % ncols) as f64;
                let exp = col as f64 /self.hidden_size as f64;
                if(col % 2 == 0){
                    // sin thing
                    let encoding = f64::sin(pos/ denom_base.powf(exp));
                    embed_table[(row, col)] += encoding;
                }
                else{
                    //cos thing
                    let encoding = f64::cos(pos/ denom_base.powf(exp));
                    embed_table[(row, col)] += encoding;
                }
            }
        }
    }


}



// pub fn xavier_init(n_rows : usize, n_cols: usize)
// -> DMatrix<f64>{
//     let mut rng = rand::rng();

//     let limit = ()


// }