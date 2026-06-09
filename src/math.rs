use nalgebra::DMatrix;
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