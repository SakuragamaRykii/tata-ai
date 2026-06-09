//use crate::math::*;

use std::intrinsics::{powf64, sinf64};

use nalgebra::DMatrix;
use rand::prelude::*;


struct MHA{
    Q : DMatrix<f64>,
    K : DMatrix<f64>,
    V : DMatrix<f64>,
    output: DMatrix<f64>,
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
    d_k: usize,
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