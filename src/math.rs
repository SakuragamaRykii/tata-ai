use std::num;

use nalgebra::{coordinates::X, *};
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


pub struct Normalizer{
    d_model : usize,
    gamma : DVector<f64>, // rescale (can be learned via backwards)
    beta : DVector<f64>, // shift (can be learned via backwards)

    
}

impl Normalizer{

    pub fn new(d_model : usize) -> Normalizer{
        Normalizer{
            d_model,
            gamma: DVector::<f64>::repeat(d_model, 1.0),
            beta : DVector::<f64>::zeros(d_model)
        }
    }


    pub fn x_hat(&self, x : Matrix<f64, Const<1>, Dyn, ViewStorage<'_, f64, Const<1>, Dyn, Const<1>, Dyn>>) 
    -> Matrix<f64, Const<1>, Dyn, VecStorage<f64, Const<1>, Dyn>>{
        let mu = x.mean();
        let variance = x.variance() + 1e-3;
        x.map(|k| (k - mu) / variance.sqrt())
    }

    pub fn normalize_forward(&self, X : DMatrix<f64>) -> DMatrix<f64>{
        // Input X is always gonna be the output of a sublayer added with their own input
        let mut output = DMatrix::<f64>::zeros(X.nrows(), X.ncols());

        for i in 0..X.nrows(){
            let x = X.row(i);
            let x_hat = self.x_hat(x);

            let y = x_hat.component_mul(&self.gamma.transpose()) + self.beta.transpose();
            // We use component_mul with x_hat and gamma because we want to keep y as a 1 * N column vector
            // cuz if just multiply x_hat with gamma normally it becomes an N * N matrix which gets applied to 
            // each row of output (line below me) and that makes no sense
            output.row_mut(i).copy_from(&y);
        }
        output
    }


    pub fn backward_norm(&self, X : DMatrix<f64>, dldy : DMatrix<f64>) 
    -> (DMatrix<f64>, DVector<f64>, DVector<f64>) // for updating X, gamma and beta
    {
        let mut dldgamma = DVector::<f64>::zeros(self.d_model);
        let mut dldbeta = DVector::<f64>::zeros(self.d_model);

        let num_tokens = X.nrows();
        let token_size = X.ncols();
        let mut dldx = DMatrix::<f64>::zeros(num_tokens, token_size);

        let d_model_float = self.d_model as f64;

        for i in 0..num_tokens{
            let dldy_row = dldy.row(i);
            let x = X.row(i);
            let x_hat = self.x_hat(x);

            // for dldx
            let denominator = 1. / ( d_model_float * (x.variance() + 1e-3).sqrt());

            let mut a = vec![0.0; self.d_model]; // actually more efficient than initializing DVector Zeros
            let mut b = 0.0;
            let mut c_sum = 0.0;

            for j in 0..self.d_model{
                dldgamma[j] += dldy[(i, j)] * x_hat[j];
                dldbeta[j] += dldy[(i, j)] ;

                let dldx_hat_j = dldy_row[j] * self.gamma[j];

                a[j] = dldx_hat_j * d_model_float;
                b += dldx_hat_j;
                c_sum += dldx_hat_j * x_hat[j];

            }       

            for j in 0..self.d_model{
                let c = c_sum * x_hat[j];
                dldx[(i, j)] = (a[j] - b - c) * denominator;
            }

        }

        (dldx, dldgamma, dldbeta)

    }


}