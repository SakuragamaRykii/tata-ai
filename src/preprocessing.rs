use std::{collections::{HashMap, HashSet}, vec};

use nalgebra::{DMatrix, stack, zero};



pub fn tokenize_initial(mut input: String) -> Vec<Vec<String>>{
    let mut out: Vec<Vec<String>> = Vec::new();

    // [? . ! - ; : ]

    let mut words = input.split(' ');
    if let Some(first_word ) = words.next(){
        let mut word_buffer = Vec::<String>::new();
        let mut pushed_punc : bool = false;

        let mut first_word_chars = first_word.chars().peekable();
        while let Some(c) = first_word_chars.next() {
            let c_token = c.to_string();

            if !first_word_chars.peek().is_none(){
                word_buffer.push(c_token);
                continue;
            }

            if !c.is_alphanumeric(){
                // add the current buffer to out
                // add c to out
                // and then break
                
                out.push(word_buffer.clone());
                out.push(vec![c_token]);
                pushed_punc = true;
            }
            else{
                word_buffer.push(c_token)
            }
        }
        if !pushed_punc{out.push(word_buffer);}
    }

    for word in words{
        let mut word_buffer = vec!['Ġ'.to_string()];
        let mut pushed_punc : bool = false;

        let mut word_chars = word.chars().peekable();
        while let Some(c) = word_chars.next() {
            let c_token = c.to_string();

            if !word_chars.peek().is_none(){
                word_buffer.push(c_token);
                continue;
            }

            if !c.is_alphanumeric(){
                // add the current buffer to out
                // add c to out
                // and then break
                
                out.push(word_buffer.clone());
                out.push(vec![c_token]);
                pushed_punc = true;
            }
            else{
                word_buffer.push(c_token)
            }
        }

        if pushed_punc{continue;}

        out.push(word_buffer);

    }

    out
    
}

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

    pub fn new(input : String) -> Tokenizer{
        let mut vocabulary = HashMap::<String, usize>::new();
        let mut inverse_vocab = Vec::<String>::new();

        let pad_sequence_token = "<PAD>".to_string();
        let fallback_token = "<UNK>".to_string();
        let cls_token = "<CLS>".to_string();
        
        append_token(&mut vocabulary, &mut inverse_vocab, pad_sequence_token);
        append_token(&mut vocabulary, &mut inverse_vocab, fallback_token);
        append_token(&mut vocabulary, &mut inverse_vocab, cls_token);
        append_token(&mut vocabulary, &mut inverse_vocab, 'Ġ'.to_string());

        for c in input.chars(){
            let token_str = c.to_string();
            
            if !c.is_whitespace() && !vocabulary.contains_key(&token_str){
                append_token(&mut vocabulary, &mut inverse_vocab, token_str);
            }
        }


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

            if let Some((max_pair, max_freq)) = pair_tracker
            .iter()
            .max_by_key(|&(_, freq) | freq)
            .map(|(pair, freq)| (pair.clone(), *freq))
            {
                if max_freq <= 1{
                    break;
                }
                let form_pair = format!("{}{}", max_pair.0, max_pair.1);

                self.append_token(form_pair.clone());

                self.merge_history.push((max_pair.0.clone(), max_pair.1.clone()));           

                corpus = corpus.into_iter()
                .map(|word: Vec<String>| {
                    let mut updated_word = Vec::<String>::new();
                    let mut i: usize = 0;
                    while i < word.len(){
                        if
                        i + 1 < word.len() && 
                        word[i] == max_pair.0 &&
                        word[i+1] == max_pair.1{
                            updated_word.push(form_pair.clone());
                            i += 2;
                        }
                        else{
                            updated_word.push(word[i].clone());
                            i += 1
                        }
                        

                    }

                    updated_word
                }).collect();
            }
            else{
                break;
            }
        }
    }
    pub fn append_token(&mut self, token : String){
        self.vocabulary.insert(token.clone(), self.inverse_vocab.len());
        self.inverse_vocab.push(token);
    }

    pub fn encode(&self, mut input: String) -> Vec<usize>{
        let mut input_tokenized = tokenize_initial(input);

        for (a, b) in &self.merge_history{
            for word in input_tokenized.iter_mut(){
                let mut token_merged = Vec::<String>::new();
                let mut i: usize = 0;
                while i < word.len(){

                    if 
                    i + 1 < word.len() &&
                    &word[i] == a &&
                    &word[i+1] == b{
                        token_merged.push(format!("{}{}", a, b));
                        i += 2;
                    }
                    else{
                        token_merged.push(word[i].clone());
                        i += 1
                    }
                    

                }
                *word = token_merged;
            }
        }

        input_tokenized.into_iter()
        .flatten()
        .map(|token| {
            *self.vocabulary.get(&token).unwrap_or(&self.vocabulary["<UNK>"])
        })
        .collect()

    }

    pub fn pad_tokens(&self, tokens: &mut Vec<Vec<usize>>){
        let max_len = tokens.iter().map(|v| v.len()).max().unwrap_or(0);
        if max_len == 0{return;}

        let mut tokens_iter = tokens.iter_mut();
        while let Some(token) = tokens_iter.next(){
            let length_difference = max_len - token.len();

            for i in 0..length_difference{
                token.push(self.vocabulary["<PAD>"])
            }

        }
    }

    pub fn produce_masks(&self, tokens: Vec<Vec<usize>>) -> Vec<Vec<usize>>
    {
        let mut masks = Vec::new();

        for token in tokens.iter(){
            let mask: Vec<usize> = token.clone().into_iter()
                                        .map(|t| {(t != self.vocabulary["<PAD>"]) as usize})
                                        .collect();
            masks.push(mask);
        }

        masks
    }


}





#[cfg(test)]
mod tests {
    use super::*;

    fn word(chars: &[&str]) -> Vec<String> {
        chars.iter().map(|s| s.to_string()).collect()
    }

    fn make_tokenizer(initial_tokens: &[&str]) -> Tokenizer {
        let mut vocabulary = HashMap::new();
        let mut inverse_vocab = Vec::new();
        for t in initial_tokens {
            append_token(&mut vocabulary, &mut inverse_vocab, t.to_string());
        }
        Tokenizer {
            vocabulary,
            inverse_vocab,
            merge_history: Vec::new(),
        }
    }

#[test]
fn bpe_merges_the_most_frequent_pair_once() {
    // "aab aab ab" -> pair ("a","b") appears 3 times (once per word),
    // pair ("a","a") appears only 2 times (once per "aab" word).
    // No other pair exists, so ("a","b") is an unambiguous winner.
    let corpus = vec![
        word(&["a", "a", "b"]),
        word(&["a", "a", "b"]),
        word(&["a", "b"]),
    ];

    let mut tokenizer = make_tokenizer(&["<PAD>", "<UNK>", "a", "b"]);
    let starting_len = tokenizer.vocabulary.len();

    tokenizer.bpe(corpus, starting_len + 1);

    assert_eq!(tokenizer.vocabulary.len(), starting_len + 1);
    assert!(tokenizer.vocabulary.contains_key("ab"));
    assert_eq!(
        tokenizer.merge_history,
        vec![("a".to_string(), "b".to_string())]
    );
}

    #[test]
    fn bpe_does_nothing_if_target_already_met() {
        let corpus = vec![word(&["l", "o", "w"])];
        let mut tokenizer = make_tokenizer(&["<PAD>", "<UNK>", "l", "o", "w"]);
        let starting_len = tokenizer.vocabulary.len();

        tokenizer.bpe(corpus, starting_len);

        assert_eq!(tokenizer.vocabulary.len(), starting_len);
        assert!(tokenizer.merge_history.is_empty());
    }

    #[test]
    fn bpe_merges_multiple_pairs_over_several_iterations() {
        // "low" x2, "lower" x1
        // Iteration 1: ("l","o") freq=3 -> merge to "lo"
        //   -> ["lo","w"] x2, ["lo","w","e","r"] x1
        // Iteration 2: ("lo","w") freq=3 -> merge to "low"
        //   -> ["low"] x2, ["low","e","r"] x1
        // Iteration 3: pairs left only in the "lower" word: ("low","e") freq=1, ("e","r") freq=1
        //   -> max_freq <= 1, loop should break here regardless of target
        let corpus = vec![
            word(&["l", "o", "w"]),
            word(&["l", "o", "w"]),
            word(&["l", "o", "w", "e", "r"]),
        ];
        let mut tokenizer = make_tokenizer(&["<PAD>", "<UNK>", "l", "o", "w", "e", "r"]);
        let starting_len = tokenizer.vocabulary.len();

        // Ask for more merges than are actually achievable (freq > 1 runs out after 2).
        tokenizer.bpe(corpus, starting_len + 10);

        // Only 2 merges should actually happen before max_freq <= 1 triggers the break.
        assert_eq!(tokenizer.vocabulary.len(), starting_len + 2);
        assert_eq!(
            tokenizer.merge_history,
            vec![
                ("l".to_string(), "o".to_string()),
                ("lo".to_string(), "w".to_string()),
            ]
        );
        assert!(tokenizer.vocabulary.contains_key("low"));
    }

    #[test]
    fn bpe_terminates_and_stops_when_no_pairs_remain() {
        // Every word is a single token already -> no pairs exist at all.
        let corpus = vec![word(&["a"]), word(&["b"]), word(&["c"])];
        let mut tokenizer = make_tokenizer(&["<PAD>", "<UNK>", "a", "b", "c"]);
        let starting_len = tokenizer.vocabulary.len();

        tokenizer.bpe(corpus, starting_len + 5);

        assert_eq!(tokenizer.vocabulary.len(), starting_len);
        assert!(tokenizer.merge_history.is_empty());
    }

    #[test]
    fn bpe_handles_empty_words_without_panicking() {
        // Includes a zero-length word to make sure the i + 1 < word.len() bounds check holds.
        let corpus = vec![word(&[]), word(&["x", "y"]), word(&["x", "y"])];
        let mut tokenizer = make_tokenizer(&["<PAD>", "<UNK>", "x", "y"]);
        let starting_len = tokenizer.vocabulary.len();

        tokenizer.bpe(corpus, starting_len + 1);

        assert_eq!(tokenizer.vocabulary.len(), starting_len + 1);
        assert!(tokenizer.vocabulary.contains_key("xy"));
    }

    #[test]
    fn bpe_updates_inverse_vocab_consistently() {
        let corpus = vec![word(&["a", "b"]), word(&["a", "b"])];
        let mut tokenizer = make_tokenizer(&["<PAD>", "<UNK>", "a", "b"]);
        let starting_len = tokenizer.vocabulary.len();

        tokenizer.bpe(corpus, starting_len + 1);

        // vocabulary and inverse_vocab should stay in sync: index lookup round-trips.
        let idx = *tokenizer.vocabulary.get("ab").unwrap();
        assert_eq!(tokenizer.inverse_vocab[idx], "ab");
    }
}

#[cfg(test)]
mod encode_tests {
    use super::*;

    fn word(chars: &[&str]) -> Vec<String> {
        chars.iter().map(|s| s.to_string()).collect()
    }

    fn make_tokenizer(initial_tokens: &[&str]) -> Tokenizer {
        let mut vocabulary = HashMap::new();
        let mut inverse_vocab = Vec::new();
        for t in initial_tokens {
            append_token(&mut vocabulary, &mut inverse_vocab, t.to_string());
        }
        Tokenizer {
            vocabulary,
            inverse_vocab,
            merge_history: Vec::new(),
        }
    }

    #[test]
    fn encode_maps_known_chars_to_their_ids() {
        // No merges trained -- each char should map to its own token id.
        let tokenizer = make_tokenizer(&["<PAD>", "<UNK>", "a", "b", "c"]);

        let ids = tokenizer.encode("ab".to_string());

        let a_id = tokenizer.vocabulary["a"];
        let b_id = tokenizer.vocabulary["b"];
        assert_eq!(ids, vec![a_id, b_id]);
    }

    #[test]
    fn encode_applies_learned_merges_in_order() {
        // Train on a corpus where ("a","b") merges first, then ("ab","c") merges second.
        // "abc abc abc" plus "ab" alone, so ("a","b") is strictly more frequent than ("b","c").
        let corpus = vec![
            word(&["a", "b", "c"]),
            word(&["a", "b", "c"]),
            word(&["a", "b", "c"]),
            word(&["a", "b"]),
        ];
        let mut tokenizer = make_tokenizer(&["<PAD>", "<UNK>", "a", "b", "c"]);
        let starting_len = tokenizer.vocabulary.len();

        // Ask for 2 merges: ("a","b") -> "ab", then ("ab","c") -> "abc"
        tokenizer.bpe(corpus, starting_len + 2);
        assert_eq!(
            tokenizer.merge_history,
            vec![
                ("a".to_string(), "b".to_string()),
                ("ab".to_string(), "c".to_string()),
            ]
        );

        // Now encode a fresh instance of "abc" using the trained tokenizer.
        let ids = tokenizer.encode("abc".to_string());

        let abc_id = tokenizer.vocabulary["abc"];
        assert_eq!(ids, vec![abc_id]);
    }

    #[test]
    fn encode_falls_back_to_unk_for_unseen_chars() {
        let tokenizer = make_tokenizer(&["<PAD>", "<UNK>", "a", "b"]);

        // 'z' was never in the training vocabulary.
        let ids = tokenizer.encode("z".to_string());

        let unk_id = tokenizer.vocabulary["<UNK>"];
        assert_eq!(ids, vec![unk_id]);
    }

    #[test]
    fn encode_handles_multi_word_input() {
        let tokenizer = make_tokenizer(&["<PAD>", "<UNK>", "a", "b", "Ġ"]);

        // "a b" -> tokenize_initial should treat this as two words,
        // with 'Ġ' prefixed onto the second word.
        let ids = tokenizer.encode("a b".to_string());

        let a_id = tokenizer.vocabulary["a"];
        let g_id = tokenizer.vocabulary["Ġ"];
        let b_id = tokenizer.vocabulary["b"];
        assert_eq!(ids, vec![a_id, g_id, b_id]);
    }

    #[test]
    fn encode_empty_string_returns_empty_vec() {
        let tokenizer = make_tokenizer(&["<PAD>", "<UNK>"]);

        let ids = tokenizer.encode("".to_string());

        assert!(ids.is_empty());
    }
}