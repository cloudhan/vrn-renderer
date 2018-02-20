#include "Utilities.h"

#include <iostream>

#include <igl/barycenter.h>
#include <igl/cotmatrix.h>
#include <igl/doublearea.h>
#include <igl/grad.h>

#include <igl/jet.h>
#include <igl/massmatrix.h>
#include <igl/repdiag.h>

namespace Utilities {
using namespace Eigen;
namespace Laplacian {

void Precompute(const MatrixXd& V, const MatrixXi & F, SparseMatrix<double>& L, SparseMatrix<double>* K)
{
	// Compute Laplace-Beltrami operator: #V by #V
	igl::cotmatrix(V, F, L);

	// Alternative construction of same Laplacian
	SparseMatrix<double> G;
	// Gradient/Divergence
	igl::grad(V, F, G);
	// Diagonal per-triangle "mass matrix"
	VectorXd dblA;
	igl::doublearea(V, F, dblA);
	// Place areas along diagonal #dim times
	const auto & T = 1.*(dblA.replicate(3, 1)*0.5).asDiagonal();
	// Laplacian K built as discrete divergence of gradient or equivalently
	// discrete Dirichelet energy Hessian
	if (K)
	{
		*K = -G.transpose() * T * G;
		std::cout << "|K-L|: " << (*K - L).norm() << std::endl;
	}
}

void Smooth(Eigen::MatrixXd& U, const Eigen::MatrixXi& F, const Eigen::SparseMatrix<double>& L)
{
	const double coeff = 0.00002;

	// Compute centroid and subtract (also important for numerics)
	VectorXd dblA;
	igl::doublearea(U, F, dblA);
	double area = 0.5*dblA.sum();
	MatrixXd BC;
	igl::barycenter(U, F, BC);
	RowVector3d centroid(0, 0, 0);
	for (int i = 0; i<BC.rows(); i++)
	{
		centroid += 0.5*dblA(i) / area * BC.row(i);
	}
	U.rowwise() -= centroid;
	// Normalize to unit surface area (important for numerics)
	U.array() /= sqrt(area);


	// Recompute just mass matrix on each step
	SparseMatrix<double> M;
	igl::massmatrix(U, F, igl::MASSMATRIX_TYPE_BARYCENTRIC, M);
	// Solve (M-delta*L) U = M*U
	const auto & S = (M - coeff*L);
	Eigen::SimplicialLLT<Eigen::SparseMatrix<double > > solver(S);
	assert(solver.info() == Eigen::Success);
	U = solver.solve(M*U).eval();


	// restore the original dimension
	U.array() *= sqrt(area);
	U.rowwise() += centroid;

}

} // namespace Laplacian

} // namespace Utilities

