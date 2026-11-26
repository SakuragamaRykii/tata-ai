use nalgebra::*;
use rand::{rng, RngExt};
 
// ───────────────────────── activations ─────────────────────────
 
pub fn relu(d: DMatrix<f64>) -> DMatrix<f64> {
    d.map(|x| x.max(0.0))
}
 
pub fn relu_row(d: RowDVector<f64>) -> RowDVector<f64> {
    d.map(|x| x.max(0.0))
}
 
pub fn lrelu(d: DMatrix<f64>, alpha: f64) -> DMatrix<f64> {
    d.map(|x| x.max(x * alpha))
}
 
pub fn softmax(d: DMatrix<f64>) -> DMatrix<f64> {
    let exsum: f64 = d.map(|x: f64| x.exp()).sum();
    d.map(|x| x.exp() / exsum)
}
 
/// Row-wise, numerically stable softmax, in place.
/// (Rewritten on plain indexing so it doesn't depend on nalgebra's view-storage generics.)
pub fn softmax_inplace(d: &mut DMatrix<f64>, n_rows: usize) {
    let ncols = d.ncols();
    for r in 0..n_rows {
        let mut mx = f64::NEG_INFINITY;
        for c in 0..ncols {
            mx = mx.max(d[(r, c)]);
        }
        let mut sum = 0.0;
        for c in 0..ncols {
            let e = (d[(r, c)] - mx).exp();
            d[(r, c)] = e;
            sum += e;
        }
        for c in 0..ncols {
            d[(r, c)] /= sum;
        }
    }
}
 
// ───────────────────────── small matrix helpers ─────────────────────────
 
/// x[r, :] += b[0, :] for every row r
pub fn add_row_bias(x: &mut DMatrix<f64>, b: &DMatrix<f64>) {
    for r in 0..x.nrows() {
        for c in 0..x.ncols() {
            x[(r, c)] += b[(0, c)];
        }
    }
}
 
/// column sums as a 1 x ncols matrix (bias gradient)
pub fn col_sums(x: &DMatrix<f64>) -> DMatrix<f64> {
    let mut out = DMatrix::<f64>::zeros(1, x.ncols());
    for r in 0..x.nrows() {
        for c in 0..x.ncols() {
            out[(0, c)] += x[(r, c)];
        }
    }
    out
}
 
// ───────────────────────── (de)serialisation helpers ─────────────────────────
 
pub fn mat_to_vec(m: &DMatrix<f64>) -> Vec<Vec<f64>> {
    (0..m.nrows()).map(|i| m.row(i).iter().cloned().collect()).collect()
}
pub fn vec_to_mat(v: &[Vec<f64>]) -> DMatrix<f64> {
    let rows = v.len();
    let cols = if rows == 0 { 0 } else { v[0].len() };
    DMatrix::from_fn(rows, cols, |i, j| v[i][j])
}
pub fn row_to_vec(v: &RowDVector<f64>) -> Vec<f64> { v.iter().cloned().collect() }
pub fn vec_to_row(v: &[f64]) -> RowDVector<f64> { RowDVector::from_row_slice(v) }
pub fn dvec_to_vec(v: &nalgebra::DVector<f64>) -> Vec<f64> { v.iter().cloned().collect() }
pub fn vec_to_dvec(v: &[f64]) -> nalgebra::DVector<f64> { nalgebra::DVector::from_row_slice(v) }
 
// kept for compatibility; AdamW now does (decoupled) weight decay inside Param::step
pub fn l2_grad(gradient: &DMatrix<f64>, weight: &DMatrix<f64>, lambda: f64) -> DMatrix<f64> {
    gradient + (weight * lambda)
}
 
// ───────────────────────── random init ─────────────────────────
 
pub fn rand_matrix(rows: usize, cols: usize, lo: f64, hi: f64) -> DMatrix<f64> {
    let mut r = rng();
    DMatrix::from_fn(rows, cols, |_, _| r.random_range(lo..hi))
}
 
pub fn xavier_init(x: usize, y: usize) -> DMatrix<f64> {
    let a = (6.0 / (x + y) as f64).sqrt();
    rand_matrix(x, y, -a, a)
}
 
// ───────────────────────── Adam / parameters ─────────────────────────
 
#[derive(Clone, Copy)]
pub struct AdamCfg {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64, // decoupled (AdamW). only applied to params with decay = true
    pub t: usize,          // 1-based step counter (for bias correction)
    pub grad_scale: f64,   // multiplies every gradient (used for global-norm clipping)
}
 
/// A trainable tensor: value, accumulated gradient, Adam moments.
pub struct Param {
    pub w: DMatrix<f64>,
    pub g: DMatrix<f64>,
    m: DMatrix<f64>,
    v: DMatrix<f64>,
    pub decay: bool,
}
 
impl Param {
    pub fn new(w: DMatrix<f64>, decay: bool) -> Param {
        let (r, c) = w.shape();
        Param {
            g: DMatrix::zeros(r, c),
            m: DMatrix::zeros(r, c),
            v: DMatrix::zeros(r, c),
            w,
            decay,
        }
    }
 
    pub fn grad_sq(&self) -> f64 {
        self.g.iter().map(|x| x * x).sum()
    }
 
    /// One AdamW update, then zero the gradient.
    pub fn step(&mut self, o: &AdamCfg) {
        let b1c = 1.0 - o.beta1.powi(o.t as i32);
        let b2c = 1.0 - o.beta2.powi(o.t as i32);
        let w = self.w.as_mut_slice();
        let g = self.g.as_mut_slice();
        let m = self.m.as_mut_slice();
        let v = self.v.as_mut_slice();
        for i in 0..w.len() {
            let gi = g[i] * o.grad_scale;
            m[i] = o.beta1 * m[i] + (1.0 - o.beta1) * gi;
            v[i] = o.beta2 * v[i] + (1.0 - o.beta2) * gi * gi;
            let mh = m[i] / b1c;
            let vh = v[i] / b2c;
            if self.decay {
                w[i] -= o.lr * o.weight_decay * w[i];
            }
            w[i] -= o.lr * mh / (vh.sqrt() + o.eps);
            g[i] = 0.0;
        }
    }
}
 
// ───────────────────────── dropout ─────────────────────────
 
pub struct Dropout {
    p: f64,
    mask: DMatrix<f64>,
    active: bool,
}
 
impl Dropout {
    pub fn new(p: f64) -> Dropout {
        Dropout { p, mask: DMatrix::zeros(1, 1), active: false }
    }
 
    pub fn forward(&mut self, x: DMatrix<f64>, training: bool) -> DMatrix<f64> {
        if !training || self.p <= 0.0 {
            self.active = false;
            return x;
        }
        self.active = true;
        let keep = 1.0 - self.p;
        let scale = 1.0 / keep;
        let mut r = rng();
        self.mask = DMatrix::from_fn(x.nrows(), x.ncols(), |_, _| {
            if r.random_range(0.0..1.0) < keep { scale } else { 0.0 }
        });
        x.component_mul(&self.mask)
    }
 
    pub fn backward(&self, d: DMatrix<f64>) -> DMatrix<f64> {
        if self.active { d.component_mul(&self.mask) } else { d }
    }
}
 