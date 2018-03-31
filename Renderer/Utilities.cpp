#include "Utilities.h"
#include "BVHTree.h"

#include <iostream>

#include <igl/barycenter.h>
#include <igl/cotmatrix.h>
#include <igl/doublearea.h>
#include <igl/grad.h>

#include <igl/jet.h>
#include <igl/massmatrix.h>
#include <igl/repdiag.h>
#include <igl/AABB.h>

using namespace Eigen;


namespace Utilities {

void Laplacian::Precompute(const MatrixXd& V, const MatrixXi & F, SparseMatrix<double>& L, SparseMatrix<double>* K)
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

void Laplacian::Smooth(Eigen::MatrixXd& U, const Eigen::MatrixXi& F, const Eigen::SparseMatrix<double>& L)
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


void Clean::RemoveDuplicates(const MatrixXd & V, const MatrixXi & F, MatrixXd & NV, MatrixXi & NF, Eigen::VectorXi & I, const double epsilon)
{
	using namespace std;
	//// build collapse map
	int n = V.rows();

	I = VectorXi(n);
	I[0] = 0;

	BVHTree tree;
	tree.Build(V, F, epsilon);

	CandidateIndexPairs canditatePairs;
	tree.BroadPhaseDetect(canditatePairs);
	//std::cout << canditatePairs.size() << std::endl;

	std::vector<int> indicesMapping;
	indicesMapping.reserve(n);
	for (int i = 0; i < n; i++) indicesMapping.push_back(i);

	for (const auto& p : canditatePairs)
	{
		int idx1 = p.first;
		int idx2 = p.second;

		if ((V.row(idx1) - V.row(idx2)).norm() < epsilon)
		{
			int newIdx = idx1;
			while (indicesMapping[newIdx] != newIdx)
				newIdx = indicesMapping[newIdx]; // trace back to root indices
			indicesMapping[idx2] = newIdx;
		}

	}

	bool *VISITED = new bool[n];
	for (int i = 0; i <n; ++i)
		VISITED[i] = false;

	NV.resize(n, V.cols());
	int count = 0;
	for (int i = 0; i <n; ++i)
	{
		if (indicesMapping[i] == i)
		{
			NV.row(count) = V.row(i);
			I[i] = count;
			count++;
		}
		else
		{
			I[i] = I[indicesMapping[i]];
		}
	}

	NV.conservativeResize(count, Eigen::NoChange);

	count = 0;
	std::vector<int> face;
	NF.resizeLike(F);
	for (int i = 0; i <F.rows(); ++i)
	{
		face.clear();
		for (int j = 0; j< F.cols(); ++j)
			if (std::find(face.begin(), face.end(), I[F(i, j)]) == face.end())
				face.push_back(I[F(i, j)]);
		if (face.size() == size_t(F.cols()))
		{
			for (unsigned j = 0; j< F.cols(); ++j)
				NF(count, j) = face[j];
			count++;
		}
	}
	NF.conservativeResize(count, Eigen::NoChange);

	//delete[] VISITED;

}

} // namespace Utilities


