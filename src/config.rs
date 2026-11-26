// ========= ALL FOR MAIN ==================
pub const K_EPOCHS: usize = 20;
pub const LR: f64 = 0.001;
pub const N_LAYERS: usize = 2;
pub const TARGET_VOCAB: usize = 3000;
pub const SEQ_LEN: usize = 48;
pub const TRAIN_FRAC: f64 = 0.75;
pub const WEIGHT_DECAY: f64 = 0.0001;
pub const N_SAMPLES: usize = 10000;


pub const MODEL_PATH: &str = "model.json";

// neutral is now a real, learned class (GoEmotions label 27) instead of a confidence threshold
pub const EMOTIONS: [&str; 6] = [
    "happy", "sad", "angry", "fear", "love", "neutral"
];
// zero-based index of neutral in EMOTIONS
pub const NEUTRAL_CLASS: usize = 5;

// values are 1-based class ids (class index = value - 1)
pub const GO_EMOTIONS_TO_CLASS: [Option<usize>; 28] = [
    None,     // 0  admiration
    Some(1),  // 1  amusement     -> happy
    Some(3),  // 2  anger         -> angry
    Some(3),  // 3  annoyance     -> angry
    None,     // 4  approval
    Some(5),  // 5  caring        -> love
    None,     // 6  confusion
    None,     // 7  curiosity
    None,     // 8  desire
    None,     // 9  disappointment
    None,     // 10 disapproval
    None,     // 11 disgust
    None,     // 12 embarrassment
    Some(1),  // 13 excitement    -> happy
    Some(4),  // 14 fear          -> fear
    None,     // 15 gratitude
    Some(2),  // 16 grief         -> sad
    Some(1),  // 17 joy           -> happy
    Some(5),  // 18 love          -> love
    None,     // 19 nervousness
    Some(1),  // 20 optimism      -> happy
    Some(1),  // 21 pride         -> happy
    None,     // 22 realization
    Some(1),  // 23 relief        -> happy
    None,     // 24 remorse
    Some(2),  // 25 sadness       -> sad
    None,     // 26 surprise
    Some(6),  // 27 neutral       -> neutral
];


// ============== ALL FOR TRANSFORMER ===============
pub const D_MODEL: usize = 64; // 512 for bigger tests
// dimentionality of a token

pub const N_HEADS: usize = 8;
//number of perspectives the models gonna have on

pub const D_K : usize = D_MODEL / N_HEADS;





// =================== ALL FOR MATH ==================

// adam shit
pub const ALPHA: f64 = 0.90;
pub const BETA: f64 = 0.99;
pub const EPSILON : f64 = 1e-8;
