use std::cell::RefCell;
use std::collections::HashMap;
 
/// Lower-cases, splits on whitespace, and returns one "word group" per word:
///   "Ġ" + the word's characters      (leading/trailing punctuation is split off as its own 1-char groups)
/// BPE merges happen inside a group only.
pub fn tokenize_initial(input: String) -> Vec<Vec<String>> {
    let input = input.to_lowercase();
    let mut out: Vec<Vec<String>> = Vec::new();
 
    for word in input.split_whitespace() {
        let chars: Vec<char> = word.chars().collect();
        let mut start = 0;
        let mut end = chars.len();
 
        while start < end && !chars[start].is_alphanumeric() {
            out.push(vec![chars[start].to_string()]);
            start += 1;
        }
        let mut trailing: Vec<String> = Vec::new();
        while end > start && !chars[end - 1].is_alphanumeric() {
            trailing.push(chars[end - 1].to_string());
            end -= 1;
        }
 
        if start < end {
            let mut group = vec!["Ġ".to_string()];
            group.extend(chars[start..end].iter().map(|c| c.to_string()));
            out.push(group);
        }
        for t in trailing.into_iter().rev() {
            out.push(vec![t]);
        }
    }
    out
}
 
pub struct Tokenizer {
    pub vocabulary: HashMap<String, usize>,
    pub inverse_vocab: Vec<String>,
    pub merge_history: Vec<(String, String)>,
    ranks: HashMap<(String, String), usize>,
    cache: RefCell<HashMap<String, Vec<usize>>>, // word -> token ids
}
 
pub fn append_token(vocab: &mut HashMap<String, usize>, inverse: &mut Vec<String>, token: String) {
    vocab.insert(token.clone(), inverse.len());
    inverse.push(token);
}
 
fn merge_pair_in_word(word: &[String], a: &str, b: &str, merged: &str) -> Vec<String> {
    let mut out = Vec::with_capacity(word.len());
    let mut i = 0;
    while i < word.len() {
        if i + 1 < word.len() && word[i] == a && word[i + 1] == b {
            out.push(merged.to_string());
            i += 2;
        } else {
            out.push(word[i].clone());
            i += 1;
        }
    }
    out
}
 
impl Tokenizer {
    pub fn new(input: String) -> Tokenizer {
        let input = input.to_lowercase();
        let mut vocabulary = HashMap::<String, usize>::new();
        let mut inverse_vocab = Vec::<String>::new();
 
        append_token(&mut vocabulary, &mut inverse_vocab, "<PAD>".to_string()); // id 0
        append_token(&mut vocabulary, &mut inverse_vocab, "<UNK>".to_string());
        append_token(&mut vocabulary, &mut inverse_vocab, "<CLS>".to_string());
        append_token(&mut vocabulary, &mut inverse_vocab, "Ġ".to_string());
 
        for c in input.chars() {
            let token_str = c.to_string();
            if !c.is_whitespace() && !vocabulary.contains_key(&token_str) {
                append_token(&mut vocabulary, &mut inverse_vocab, token_str);
            }
        }
 
        Tokenizer {
            vocabulary,
            inverse_vocab,
            merge_history: Vec::new(),
            ranks: HashMap::new(),
            cache: RefCell::new(HashMap::new()),
        }
    }
 
    pub fn from_parts(inverse_vocab: Vec<String>, merge_history: Vec<(String, String)>) -> Tokenizer {
        let mut vocabulary = HashMap::new();
        for (i, t) in inverse_vocab.iter().enumerate() {
            vocabulary.insert(t.clone(), i);
        }
        let ranks = merge_history
            .iter()
            .enumerate()
            .map(|(i, (a, b))| ((a.clone(), b.clone()), i))
            .collect();
        Tokenizer { vocabulary, inverse_vocab, merge_history, ranks, cache: RefCell::new(HashMap::new()) }
    }
 
    pub fn append_token(&mut self, token: String) {
        self.vocabulary.insert(token.clone(), self.inverse_vocab.len());
        self.inverse_vocab.push(token);
    }
 
    /// BPE training. The corpus is collapsed to (unique word -> count) first, so every merge step
    /// costs O(#unique words) instead of O(#all words). Ties are broken deterministically.
    pub fn bpe(&mut self, corpus: Vec<Vec<String>>, target_vocab_size: usize) {
        let mut counts: HashMap<Vec<String>, usize> = HashMap::new();
        for w in corpus {
            if !w.is_empty() {
                *counts.entry(w).or_insert(0) += 1;
            }
        }
        let mut words: Vec<(Vec<String>, usize)> = counts.into_iter().collect();
        words.sort();
 
        while self.vocabulary.len() < target_vocab_size {
            let best: Option<((String, String), usize)> = {
                let mut pairs: HashMap<(&str, &str), usize> = HashMap::new();
                for (w, c) in &words {
                    for p in w.windows(2) {
                        *pairs.entry((p[0].as_str(), p[1].as_str())).or_insert(0) += *c;
                    }
                }
                pairs
                    .into_iter()
                    // highest count; ties -> lexicographically smallest pair
                    .max_by(|a, b| a.1.cmp(&b.1).then_with(|| b.0.cmp(&a.0)))
                    .map(|((a, b), f)| ((a.to_string(), b.to_string()), f))
            };
 
            let Some(((a, b), freq)) = best else { break };
            if freq <= 1 {
                break;
            }
 
            let merged = format!("{}{}", a, b);
            self.append_token(merged.clone());
            self.ranks.insert((a.clone(), b.clone()), self.merge_history.len());
            self.merge_history.push((a.clone(), b.clone()));
 
            for (w, _) in words.iter_mut() {
                if w.windows(2).any(|p| p[0] == a && p[1] == b) {
                    *w = merge_pair_in_word(w, &a, &b, &merged);
                }
            }
        }
        self.cache.borrow_mut().clear();
    }
 
    /// Standard rank-based BPE encode: repeatedly merge the adjacent pair with the lowest merge rank.
    fn merge_word(&self, mut word: Vec<String>) -> Vec<String> {
        loop {
            let mut best: Option<(usize, usize)> = None; // (rank, position)
            for i in 0..word.len().saturating_sub(1) {
                if let Some(&r) = self.ranks.get(&(word[i].clone(), word[i + 1].clone())) {
                    if best.map_or(true, |(br, _)| r < br) {
                        best = Some((r, i));
                    }
                }
            }
            match best {
                None => break,
                Some((_, i)) => {
                    let m = format!("{}{}", word[i], word[i + 1]);
                    word[i] = m;
                    word.remove(i + 1);
                }
            }
        }
        word
    }
 
    fn id_of(&self, token: &str) -> usize {
        match self.vocabulary.get(token) {
            Some(&i) => i,
            None => self.vocabulary["<UNK>"],
        }
    }
 
    pub fn encode(&self, input: String) -> Vec<usize> {
        let mut ids = vec![self.id_of("<CLS>")];
        for group in tokenize_initial(input) {
            let key = group.concat();
            if let Some(cached) = self.cache.borrow().get(&key) {
                ids.extend_from_slice(cached);
                continue;
            }
            let merged = self.merge_word(group);
            let word_ids: Vec<usize> = merged.iter().map(|t| self.id_of(t)).collect();
            ids.extend_from_slice(&word_ids);
            self.cache.borrow_mut().insert(key, word_ids);
        }
        ids
    }
 
    pub fn pad_tokens(&self, tokens: &mut Vec<Vec<usize>>) {
        let max_len = tokens.iter().map(|v| v.len()).max().unwrap_or(0);
        if max_len == 0 {
            return;
        }
        for token in tokens.iter_mut() {
            let diff = max_len - token.len();
            for _ in 0..diff {
                token.push(self.vocabulary["<PAD>"]);
            }
        }
    }
 
    pub fn produce_masks(&self, tokens: Vec<Vec<usize>>) -> Vec<Vec<usize>> {
        tokens
            .iter()
            .map(|t| t.iter().map(|&x| (x != self.vocabulary["<PAD>"]) as usize).collect())
            .collect()
    }
}
 
#[cfg(test)]
mod tests {
    use super::*;
 
    fn word(chars: &[&str]) -> Vec<String> {
        chars.iter().map(|s| s.to_string()).collect()
    }
 
    fn make_tokenizer(initial_tokens: &[&str]) -> Tokenizer {
        Tokenizer::from_parts(initial_tokens.iter().map(|s| s.to_string()).collect(), Vec::new())
    }
 
    #[test]
    fn bpe_merges_the_most_frequent_pair_once() {
        let corpus = vec![word(&["a", "a", "b"]), word(&["a", "a", "b"]), word(&["a", "b"])];
        let mut t = make_tokenizer(&["<PAD>", "<UNK>", "a", "b"]);
        let n = t.vocabulary.len();
        t.bpe(corpus, n + 1);
        assert_eq!(t.vocabulary.len(), n + 1);
        assert!(t.vocabulary.contains_key("ab"));
        assert_eq!(t.merge_history, vec![("a".to_string(), "b".to_string())]);
    }
 
    #[test]
    fn bpe_does_nothing_if_target_already_met() {
        let mut t = make_tokenizer(&["<PAD>", "<UNK>", "l", "o", "w"]);
        let n = t.vocabulary.len();
        t.bpe(vec![word(&["l", "o", "w"])], n);
        assert_eq!(t.vocabulary.len(), n);
        assert!(t.merge_history.is_empty());
    }
 
    #[test]
    fn bpe_merges_multiple_pairs_and_stops_at_freq_1() {
        let corpus = vec![word(&["l", "o", "w"]), word(&["l", "o", "w"]), word(&["l", "o", "w", "e", "r"])];
        let mut t = make_tokenizer(&["<PAD>", "<UNK>", "l", "o", "w", "e", "r"]);
        let n = t.vocabulary.len();
        t.bpe(corpus, n + 10);
        assert_eq!(t.vocabulary.len(), n + 2);
        assert_eq!(
            t.merge_history,
            vec![("l".to_string(), "o".to_string()), ("lo".to_string(), "w".to_string())]
        );
    }
 
    #[test]
    fn bpe_handles_empty_words_and_singletons() {
        let mut t = make_tokenizer(&["<PAD>", "<UNK>", "x", "y"]);
        let n = t.vocabulary.len();
        t.bpe(vec![word(&[]), word(&["x", "y"]), word(&["x", "y"]), word(&["x"])], n + 1);
        assert!(t.vocabulary.contains_key("xy"));
    }
 
    #[test]
    fn encode_prepends_cls_and_g_prefix() {
        let t = make_tokenizer(&["<PAD>", "<UNK>", "<CLS>", "Ġ", "a", "b"]);
        let ids = t.encode("ab".to_string());
        assert_eq!(ids, vec![2, 3, 4, 5]);
        let ids = t.encode("a b".to_string());
        assert_eq!(ids, vec![2, 3, 4, 3, 5]);
        assert_eq!(t.encode("".to_string()), vec![2]);
    }
 
    #[test]
    fn encode_lowercases_and_falls_back_to_unk() {
        let t = make_tokenizer(&["<PAD>", "<UNK>", "<CLS>", "Ġ", "a"]);
        assert_eq!(t.encode("A".to_string()), vec![2, 3, 4]);
        assert_eq!(t.encode("z".to_string()), vec![2, 3, 1]);
    }
 
    #[test]
    fn encode_applies_learned_merges() {
        let mut t = make_tokenizer(&["<PAD>", "<UNK>", "<CLS>", "Ġ", "a", "b", "c"]);
        let corpus = vec![
            word(&["Ġ", "a", "b", "c"]),
            word(&["Ġ", "a", "b", "c"]),
            word(&["Ġ", "a", "b", "c"]),
            word(&["Ġ", "a", "b"]),
        ];
        let n = t.vocabulary.len();
        t.bpe(corpus, n + 3);
        let ids = t.encode("abc".to_string());
        assert_eq!(ids.len(), 2); // <CLS>, "Ġabc"
        assert_eq!(t.inverse_vocab[ids[1]], "Ġabc");
    }
 
    #[test]
    fn punctuation_is_split_off() {
        let g = tokenize_initial("Hello, (World)!!".to_string());
        let flat: Vec<String> = g.iter().map(|w| w.concat()).collect();
        assert_eq!(flat, vec!["Ġhello", ",", "(", "Ġworld", ")", "!", "!"]);
    }
}
 