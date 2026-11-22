use std::num;

use nalgebra::*;
//use cudarc::*;


pub fn relu(d : DMatrix<f64>) -> DMatrix<f64> {
    d.map(|x| x.max(0.0))

}

pub fn relu_row(d : RowDVector<f64>) -> RowDVector<f64> {
    d.map(|x| x.max(0.0))

}


pub fn lrelu(d : DMatrix<f64>, alpha: f64) -> DMatrix<f64> {
    d.map(|x| x.max(x * alpha))

}

pub fn softmax(d: DMatrix<f64>) -> DMatrix<f64> {
    let exsum : f64 = d.map(|x : f64| x.exp()).sum();
    d.map(|x | x.exp() / exsum )
}


pub fn softmax_row_inplace(
    mut row: Matrix<f64, Const<1>, Dyn, ViewStorageMut<'_, f64, Const<1>, Dyn, Const<1>, Dyn>>
) {
    // 1. For numerical stability, find the max value in the row
    let max_val = row.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    
    // 2. Sum up the exponentials of the shifted values
    let exp_sum: f64 = row.iter().map(|x| (x - max_val).exp()).sum();
    
    // 3. Mutate the row elements in-place
    row.apply(|x| *x = (*x - max_val).exp() / exp_sum);
}

//mut d: Matrix<f64, Dyn, Dyn, VecStorage<f64, Dyn, Dyn>>

pub fn softmax_inplace(d: &mut DMatrix<f64>, n_rows: usize){
    for row in 0..n_rows {
        let current_row = d.row_mut(row);
        softmax_row_inplace(current_row);
    }

}

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


// L2
pub fn l2_grad(gradient: &DMatrix<f64>, weight:&DMatrix<f64>, lambda: f64) -> DMatrix<f64>{
    gradient + (weight * lambda)
    
}