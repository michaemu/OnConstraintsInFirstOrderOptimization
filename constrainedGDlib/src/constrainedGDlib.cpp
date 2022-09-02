#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>


#include "Utils.h"
#include "QuadraticProgram.h"
#include "SparseMatrixCRS.h"
#include "DenseMatrix.h"
#include "SOCP.h"

using namespace pybind11::literals;

static void PrintVecPy(const real_t* vec, int_t nv) {
	for (int_t k = 0; k < nv; ++k) {
		pybind11::print(vec[k]);
	}
}


static void PrintVecPy(const int_t* vec, int_t nv) {
	for (int_t k = 0; k < nv; ++k) {
		pybind11::print(vec[k]);
	}
}

static void PrintParamsPy(const QuadraticProgram::Settings& set) {
	pybind11::print("Parameters -------------------");
	pybind11::print("T:", set.T);
	pybind11::print("alpha:", set.alpha);
	pybind11::print("omega:", set.omega);
	pybind11::print("TOL_const:", set.TOL_const);
	pybind11::print("MAXITER:", (real_t)set.MAXITER);
	pybind11::print("ABSTOL:", set.ABSTOL);
	pybind11::print("MAXITER_PROX:", (real_t)set.MAXITER_PROX);
	pybind11::print("ABSTOL_PROX:", set.ABSTOL_PROX);
	pybind11::print("RELTOL_PROX:", set.RELTOL_PROX);
	pybind11::print(" ----------------------- ");
}


static void PrintParamsPy(const SOCP::Settings& set) {
	pybind11::print("Parameters -------------------");
	pybind11::print("T:", set.T);
	pybind11::print("alpha:", set.alpha);
	pybind11::print("omega:", set.omega);
	pybind11::print("TOL_const:", set.TOL_const);
	pybind11::print("MAXITER:", (real_t)set.MAXITER);
	pybind11::print("ABSTOL:", set.ABSTOL);
	pybind11::print("MAXITER_PROX:", (real_t)set.MAXITER_PROX);
	pybind11::print("ABSTOL_PROX:", set.ABSTOL_PROX);
	pybind11::print("RELTOL_PROX:", set.RELTOL_PROX);
	pybind11::print(" ----------------------- ");
}

static void PrintSparseMatrixCRS(const SparseMatrixCRS& mat) {
	int_t val_idx = 0;
	for (int_t k = 0; k < mat.getnRows(); ++k) {
		for (int_t j = 0; j < mat.getnCols(); ++j) {
			if (val_idx < mat.getRowPtr()[k+1] && val_idx >= mat.getRowPtr()[k] && mat.getColIdx()[val_idx] == j) {
				pybind11::print(mat.getVal()[val_idx++],"\t","end"_a="");
			}
			else {
				pybind11::print("0.0 \t","end"_a="");
			}
		}
		pybind11::print("");
	}
	pybind11::print("");
}

static void PrintDenseMatrix(const DenseMatrix& mat) {
	for (int_t k = 0; k < mat.getnRows(); ++k) {
		for (int_t j = 0; j < mat.getnCols(); ++j) {
			pybind11::print(mat.getVal()[k*mat.getnCols()+j], "\t", "end"_a = "");
		}
		pybind11::print("");
	}
}




int solveQP(
	// H sparse matrix
	const pybind11::array_t<double>& H_val, const pybind11::array_t<int>& H_col_idx, const pybind11::array_t<int>& H_row_ptr, const pybind11::array_t<int>& H_params,
	// c vector
	const pybind11::array_t<double>& c,
	// A sparse matrix
	const pybind11::array_t<double>& A_val, const pybind11::array_t<int>& A_col_idx, const  pybind11::array_t<int>& A_row_ptr, const  pybind11::array_t<int>& A_params,
	// b vector
	const pybind11::array_t<double>& b,
	// G dense matrix
	const pybind11::array_t<double>& G_val, const pybind11::array_t<int> G_params,
	// parameters
	const pybind11::array_t<double>& params,
	// number of equality constraints
	const int& neq,
	// x0
	pybind11::array_t<double>& x0,
	// lambda
	pybind11::array_t<double>& lambda) {
	// convert to c++
	// H
	real_t* H_val_ = (real_t*)H_val.request().ptr;
	int_t* H_col_idx_ = (int_t*)H_col_idx.request().ptr;
	int_t* H_row_ptr_ = (int_t*)H_row_ptr.request().ptr;
	int_t* H_params_ = (int_t*)H_params.request().ptr;

	// c
	// real_t* c_ = (real_t*)c.request().ptr;

	// A
	real_t* A_val_ = (real_t*)A_val.request().ptr;
	int_t* A_col_idx_ = (int_t*)A_col_idx.request().ptr;
	int_t* A_row_ptr_ = (int_t*)A_row_ptr.request().ptr;
	int_t* A_params_ = (int_t*)A_params.request().ptr;

	// b
	// real_t* b_ = (real_t*)b.request().ptr;

	// G
	real_t* G_val_ = (real_t*)G_val.request().ptr;
	int_t* G_params_ = (int_t*)G_params.request().ptr;

	// params
	real_t* params_ = (real_t*)params.request().ptr;

	// x0
	real_t* x0_ = (real_t*)x0.request().ptr;

	// lambda
	real_t* lambda_=(real_t*)lambda.request().ptr;

	SparseMatrixCRS H(H_params_[0], H_params_[1], H_params_[2], H_val_, H_col_idx_, H_row_ptr_);
	SparseMatrixCRS A(A_params_[0], A_params_[1], A_params_[2], A_val_, A_col_idx_, A_row_ptr_);
	DenseMatrix G(G_params_[0], G_params_[1], G_val_);

	struct QuadraticProgram::Settings settings = { params_[0], params_[1], params_[2], params_[3], (int_t)params_[4], params_[5], (int_t)params_[6], params_[7], params_[8] };

	const int_t nx = x0.request().size;

#ifdef LOAD_DEBUG
	pybind11::print("mat H: ----------- ");
	PrintSparseMatrixCRS(H);

	pybind11::print("vec c: ------------ ");
	PrintVecPy(c.data(), c.size());

	pybind11::print("mat A: ----------- ");
	PrintSparseMatrixCRS(A);
	
	pybind11::print("vec b: ------------ ");
	PrintVecPy(b.data(), b.size());

	pybind11::print("mat G: ------------ ");
	PrintDenseMatrix(G);

	PrintParamsPy(settings);
	
	pybind11::print("vec x0: ----------- ");
	PrintVecPy(x0_,nx);
#endif

	QuadraticProgram QP(&H, &A, &G, c.data(), b.data(), settings, neq);

	int_t returnflag = -1;
	if (QP.solve(x0.data()))
		returnflag = QP.getniter();

	Utils::VectorCopy(QP.getSolution(), x0_, nx);
	Utils::VectorCopy(QP.getLambda(), lambda_, G_params_[1]);

	lambda_[G_params_[1]] = QP.getAvgNConstraints();

#ifdef DEBUG
	pybind11::print("vec x0 after solving");
	PrintVecPy(x0_,nx);
	pybind11::print("niter: ", QP.getniter());
#endif
	return returnflag;
}


int solveSOCP(
	// H sparse matrix
	const pybind11::array_t<double>& H_val, const pybind11::array_t<int>& H_col_idx, const pybind11::array_t<int>& H_row_ptr, const pybind11::array_t<int>& H_params,
	// c vector
	const pybind11::array_t<double>& c,
	// A sparse matrix
	const pybind11::array_t<double>& A_val, const pybind11::array_t<int>& A_col_idx, const  pybind11::array_t<int>& A_row_ptr, const  pybind11::array_t<int>& A_params,
	// b vector
	const pybind11::array_t<double>& b,
	// G dense matrix
	const pybind11::array_t<double>& G_val, const pybind11::array_t<int> G_params,
	// parameters
	const pybind11::array_t<double>& params,
	// nC
	const pybind11::array_t<int>& nC,
	// x0
	pybind11::array_t<double>& x0,
	// dual variables -> last is n_avg_constraints
	pybind11::array_t<double>& lambda) {
	// convert to c++
	// H
	real_t* H_val_ = (real_t*)H_val.request().ptr;
	int_t* H_col_idx_ = (int_t*)H_col_idx.request().ptr;
	int_t* H_row_ptr_ = (int_t*)H_row_ptr.request().ptr;
	int_t* H_params_ = (int_t*)H_params.request().ptr;

	// c
	real_t* c_ = (real_t*)c.request().ptr;

	// A
	real_t* A_val_ = (real_t*)A_val.request().ptr;
	int_t* A_col_idx_ = (int_t*)A_col_idx.request().ptr;
	int_t* A_row_ptr_ = (int_t*)A_row_ptr.request().ptr;
	int_t* A_params_ = (int_t*)A_params.request().ptr;

	// b
	real_t* b_ = (real_t*)b.request().ptr;

	// G
	real_t* G_val_ = (real_t*)G_val.request().ptr;
	int_t* G_params_ = (int_t*)G_params.request().ptr;

	// params
	real_t* params_ = (real_t*)params.request().ptr;

	// int_t* nC_ = (int_t*)nC.request().ptr;
	const int_t nnC = nC.request().size;

	// x0
	real_t* x0_ = (real_t*)x0.request().ptr;

	// lambda
	real_t* lambda_ = (real_t*)lambda.request().ptr;


	const SparseMatrixCRS H(H_params_[0], H_params_[1], H_params_[2], H_val_, H_col_idx_, H_row_ptr_);
	const SparseMatrixCRS A(A_params_[0], A_params_[1], A_params_[2], A_val_, A_col_idx_, A_row_ptr_);
	DenseMatrix G(G_params_[0], G_params_[1], G_val_);

	struct SOCP::Settings settings = { params_[0], params_[1], params_[2], params_[3], (int_t)params_[4], params_[5], (int_t)params_[6], params_[7], params_[8] };

	const int_t nx = x0.request().size;

#ifdef LOAD_DEBUG
	pybind11::print("mat H: ----------- ");
	PrintSparseMatrixCRS(H);

	pybind11::print("vec c: ------------ ");
	PrintVecPy(c.data(), c.size());

	pybind11::print("mat A: ----------- ");
	PrintSparseMatrixCRS(A);

	pybind11::print("vec b: ------------ ");
	PrintVecPy(b.data(), b.size());

	pybind11::print("mat G: ------------ ");
	PrintDenseMatrix(G);

	PrintParamsPy(settings);

	pybind11::print("vec x0: ----------- ");
	PrintVecPy(x0_, nx);

	pybind11::print("vec nC: ----------- ");
	PrintVecPy(nC.data(), nnC);
#endif

	SOCP socp(&H, &A, &G, c.data(), b.data(), nC.data(), nnC, settings);

	int_t returnflag = -1;
	if (socp.solve(x0.data()))
		returnflag = socp.getniter();

	Utils::VectorCopy(socp.getSolution(), x0_, nx);
	Utils::VectorCopy(socp.getLambda(), lambda_, G_params_[1]);
	lambda_[G_params_[1]] = socp.getAvgNConstraints();
	lambda_[G_params_[1]+1] = socp.getAvgNIneqConstraints();

#ifdef DEBUG
	pybind11::print("vec x0 after solving");
	PrintVecPy(x0_, nx);
	pybind11::print("niter: ", socp.getniter());
	pybind11::print("avg n constraints: ", socp.getAvgNConstraints());
	pybind11::print("Gparams: ", G_params_[1]);
#endif
	return returnflag;
}




PYBIND11_MODULE(constrainedGDlib, m) {
	m.doc() = "constrainedGD plugin"; // optional module 
	m.def("solveQP", &solveQP, "a QP solver",pybind11::return_value_policy::reference);
	m.def("solveSOCP", &solveSOCP, "a SOCP solver", pybind11::return_value_policy::reference);
}
