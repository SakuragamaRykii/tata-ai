use nalgebra::*;
//use cudarc::*;


pub fn relu(d : DMatrix<f64>) -> DMatrix<f64> {
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
    mut row: Matrix<f64, Const<1>, Dyn, ViewStorageMut<'_, f64, Const<1>, Dyn, Const<1>, Dyn>>) {
    // 1. For numerical stability, find the max value in the row
    let max_val = row.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    
    // 2. Sum up the exponentials of the shifted values
    let exp_sum: f64 = row.iter().map(|x| (x - max_val).exp()).sum();
    
    // 3. Mutate the row elements in-place
    row.apply(|x| *x = (*x - max_val).exp() / exp_sum);
}

pub fn softmax_inplace(mut d: Matrix<f64, Dyn, Dyn, ViewStorageMut<'_, f64, Dyn, Dyn, Const<1>, Dyn>>, n_rows: usize){
    for row in 0..n_rows {
        let current_row = d.row_mut(row);
        softmax_row_inplace(current_row);
    }

}