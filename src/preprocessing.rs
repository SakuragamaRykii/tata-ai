use std::collections::{HashMap, HashSet};

use nalgebra::{DMatrix, stack, zero};

pub struct Tokenizer {
    pub vocabulary : HashMap<String, usize>,
    pub inverse_vocab : Vec<String>,
    pub merge_history: Vec<(String, String)>

}


pub fn append_token(vocab : &mut HashMap<String, usize>, inverse : &mut Vec<String>, token : String){
    vocab.insert(token.clone(), inverse.len());
    inverse.push(token);
}

impl Tokenizer{

    pub fn new(&self, input : String) -> Tokenizer{
        let mut vocabulary = HashMap::<String, usize>::new();
        let mut inverse_vocab = Vec::<String>::new();

        let pad_sequence_token = "<PAD>".to_string();
        let fallback_token = "<UNK>".to_string();
        
        append_token(&mut vocabulary, &mut inverse_vocab, pad_sequence_token);
        append_token(&mut vocabulary, &mut inverse_vocab, fallback_token);
        
        for c in input.chars(){
            let token_str = c.to_string();
            
            if !c.is_whitespace() && !vocabulary.contains_key(&token_str){
                append_token(&mut vocabulary, &mut inverse_vocab, token_str);
            }
        }
        append_token(&mut vocabulary, &mut inverse_vocab, 'Ġ'.to_string());


        let merge_history: Vec<(String, String)> = Vec::<(String, String)>::new();
        Tokenizer{vocabulary, inverse_vocab, merge_history}
    }


    pub fn bpe(&mut self, mut corpus: Vec<Vec<String>>, target_vocab_size : usize){
        
        let mut highest_freq: usize = 0;



        while self.vocabulary.len() < target_vocab_size{
            let mut pair_tracker = HashMap::<(String, String), usize>::new();

            for word in &corpus{
                for c_pair in word.windows(2){
                    let c_strings = (c_pair[0].clone(), c_pair[1].clone());
                    *pair_tracker.entry(c_strings).or_insert(0) += 1;
                }
            } 

            if let Some((max_pair, max_freq)) = pair_tracker.iter().max_by_key(|&(_, freq) | freq){

                if *max_freq <= 1{
                    break;
                }


                let form_pair = format!("{}{}", max_pair.0, max_pair.1);

                self.append_token(form_pair.clone());

                self.merge_history.push((max_pair.0.clone(), max_pair.1.clone()));           


                for word in &corpus{
                    let mut updated_word = Vec::<String>::new();
                    let mut i: usize = 0;

                    while i < word.len() - 1{
                        if word[i] == max_pair.0 &&
                        word[i+1] == max_pair.1{
                            updated_word.push(form_pair.clone());
                            i += 2;
                        }
                        else{
                            updated_word.push(word[i].clone());
                            i += 1
                        }
                        

                    }
                } 

            }
        }
    }

    pub fn append_token(&mut self, token : String){
        self.vocabulary.insert(token.clone(), self.inverse_vocab.len());
        self.inverse_vocab.push(token);
    }
}