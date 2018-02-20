#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>

namespace Utilities {

using Eigen::MatrixXd;
using Eigen::MatrixXi;
using Eigen::SparseMatrix;

namespace Laplacian {


// V: vertices
// F: indices
// L: output, needed,   Laplace-Beltrami operator, 
// K: output, optional, discrete Dirichelet energy Hessian, 
void Precompute(const MatrixXd& V, const MatrixXi& F, SparseMatrix<double>& L, SparseMatrix<double>* K=nullptr);

// U: vertices input & output, output the smoothed input vertices
// F: indices
// L: precomputed Laplace-Beltrami operator
void Smooth(MatrixXd& U, const MatrixXi& F, const SparseMatrix<double>& L);

} // namespace Laplacian

} // namespace Utilities