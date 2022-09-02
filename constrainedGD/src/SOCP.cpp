#include "SOCP.h"

SOCP::SOCP(const SparseMatrixCRS* H_, const SparseMatrixCRS* A_, DenseMatrix* G_,
	const real_t* c_, const real_t* b_, const int_t* nC_, int_t nnC_, const Settings& settings_) :nx(H_->nr),
	H(H_), A(A_), G(G_), c(c_), b(b_), nC(nC_), nnC(nnC_),
	settings(settings_),nineq((nnC_-2)+nC_[1]), neq(nC_[0]), 
	W_trans_nnz(A_->row_ptr[nC_[0] + nC_[1]] + (nnC_ - 2) * H_->nr){   // save remaining items in WT in dense format
	x = new real_t[nx];
	neg_df = new real_t[nx];
	lambda = new real_t[nineq + neq];
	lambdastart = new real_t[nineq + neq];

	// reset lambda
	resetLambda();

	idx_active = new int_t[nineq];
	gtmp = new real_t[A->nr];
	g = gtmp; // we will overwrite gtmp.

	setUpWtrans();

	// Debug
	//SparseMatrixCRS tmp(neq + nineq, nx, W_trans_nnz, W_trans_val, W_trans_col_ptr, W_trans_row_ptr);
	//printf("\n\n tmp print WT\n");
	//tmp.print();

}


void SOCP::resetLambda() {
	// reset lambda
	for (int_t k = 0; k < nineq + neq; ++k) {
		lambda[k] = 0;
	}
}

bool SOCP::solve(const real_t* x0) {

#ifdef DEBUG
	FILE* fp;
	fp = fopen("dump_trajectory.txt", "w+");
#endif

	Utils::VectorCopy(x0, x, nx);

	avg_n_constraints = 0.0;
	avg_n_ineq_constraints = 0.0;

	for (int_t k = 0; k < settings.MAXITER; ++k) {

#ifdef DEBUG
		Utils::printVecToFileColumnWise(x, nx, fp);
#endif

		// compute -\nabla f(x)
		H->vecMultAdd(x, c, -1.0, -1.0, neg_df);

		// check constraint violations
		A->vecMultAdd(x, b, 1.0, -1.0, gtmp);

		// in order to save computation we overwrite gtmp
		// equality constraints are always active -> we don't need to check those!
		// check first for inequality constraints
		n_idx_active = 0;
		for (int_t j = nC[0]; j < nC[0]+nC[1]; ++j) { 
			if (gtmp[j] <= settings.TOL_const) {
				idx_active[n_idx_active++] = j;
			}
		}

		avg_n_ineq_constraints += real_t(n_idx_active);

		// check for SOC constraints
		int_t idx_tmp = nC[1];
		int_t idx_A = nC[0] + nC[1];
		for (int_t j = 2; j < nnC; ++j) {
			const real_t* v1 = gtmp + idx_A + 1;
			const real_t v0 = gtmp[idx_A];
			const real_t v1_norm = Utils::VectorNorm(v1, nC[j] - 1);

			if (- v1_norm + v0 <= settings.TOL_const) {
				idx_active[n_idx_active++] = nC[0] + nC[1] + (j - 2); // these correspond to indices in W_trans and G
				gtmp[nC[0] + nC[1] + (j - 2)] = -v1_norm + v0; // set the constraint violation

				ComputeWTRow(j, idx_A, v1,v1_norm);

				// debug
				//SparseMatrixCRS tmp(neq + nineq, nx, W_trans_nnz, W_trans_val, W_trans_col_ptr, W_trans_row_ptr);
				//printf("\n\n tmp print WT\n");
				//tmp.print();
			}
			idx_A += nC[j];
		}

		avg_n_constraints += real_t(n_idx_active);

		

		if (neq > 0 || n_idx_active > 0) {
			// compute -W\T df + alpha g --> stored in g
			updateg();

			// update G matrix
			updateG();

			// compute lambda (solve the complementary problem)
			doProx();

			// compute -\nabla f(x) + W \lambda
			updateWlambda();

		}

		// compute x+= -\nabla f(x) + W \lambda
		Utils::VectorAdd(neg_df, settings.T, x, nx);

		if (Utils::VectorMaxNorm(neg_df, nx) < settings.ABSTOL) { // convergence!

			niter = k+1;
			avg_n_constraints /= (real_t)niter;
#ifdef DEBUG
			fclose(fp);
#endif
			return true;
		}

	}

	niter = settings.MAXITER;
	avg_n_constraints /= niter;
	avg_n_ineq_constraints /= niter;

#ifdef DEBUG
	fclose(fp);
#endif
	return false;
}


void SOCP::updateg() {


	// equality constraints
	for (int_t idx = 0; idx < neq; ++idx) {
		real_t tmp = 0.0;
		for (int_t j = W_trans_row_ptr[idx]; j < W_trans_row_ptr[idx + 1]; ++j) {
			tmp+= W_trans_val[j] * neg_df[W_trans_col_ptr[j]];
		}
		g[idx] = tmp+ g[idx] * settings.alpha;
	}

	// active inequality and SOCP constraints
	for (int_t k = 0; k < n_idx_active; ++k) {
		const int_t idx = idx_active[k];

		real_t tmp = 0.0;
		for (int_t j = W_trans_row_ptr[idx]; j < W_trans_row_ptr[idx + 1]; ++j) {
			tmp += W_trans_val[j] * neg_df[W_trans_col_ptr[j]];
		}
		g[idx] = tmp + g[idx] * settings.alpha;
	}


}

void SOCP::updateWlambda() {

	// loop over equality constraints
	for (int_t k = 0; k < neq; ++k) {
		for (int_t j = W_trans_row_ptr[k]; j < W_trans_row_ptr[k + 1]; ++j) {
			neg_df[W_trans_col_ptr[j]] += W_trans_val[j] * lambda[k];
		}
	}

	// loop over active inequality constraints
	for (int_t k = 0; k < n_idx_active; ++k) {
		for (int_t j = W_trans_row_ptr[idx_active[k]]; j < W_trans_row_ptr[idx_active[k] + 1]; ++j) {
			neg_df[W_trans_col_ptr[j]] += W_trans_val[j] * lambda[idx_active[k]];
		}
	}


}



bool SOCP::doProx() {
	// int_t niterprox=settings.MAXITER_PROX; // not sure if we should keep track of the prox-iter

	for (int_t k = 0; k < settings.MAXITER_PROX; ++k) {
		copyLambdaStart(); // lambdastart <- lambda
		real_t max_G_lambda_cmax = 0.0;


		// equality constraints
		for (int_t idx = 0; idx < neq; ++idx) {
			// fact_tmp=omega/G[idx,idx]
			const real_t fact_tmp = settings.omega / G->val[idx * (nineq + neq) + idx]; // add interface to G
			real_t tmp_val = 0;

			// calculate G[idx,:]*lambda
			// start with equality constraints
			for (int_t idx2 = 0; idx2 < neq; ++idx2) {
				tmp_val += G->val[idx * (nineq + neq) + idx2] * lambda[idx2];
			}
			// add inequality constraints
			for (int_t j2 = 0; j2 < n_idx_active; ++j2) {
				const int_t idx2 = idx_active[j2];
				tmp_val += G->val[idx * (nineq + neq) + idx2] * lambda[idx2];
			}
			
			// implement lambda=lambda-omega/G[idx,idx]*(G[idx,:]*lambda+c[idx]);
			lambda[idx] -= fact_tmp * (tmp_val + g[idx]);
		}

		for (int_t j = 0; j < n_idx_active; ++j) {
			const int_t idx = idx_active[j];

			// fact_tmp=omega/G[idx,idx]
			const real_t fact_tmp = settings.omega / G->val[idx * (nineq + neq) + idx]; // add interface to G
			real_t tmp_val = 0;

			// calculate G[idx,:]*lambda
			// start with equality constraints
			for (int_t idx2 = 0; idx2 < neq; ++idx2) {
				tmp_val += G->val[idx * (nineq + neq) + idx2] * lambda[idx2];
			}
			// add inequality constraints
			for (int_t j2 = 0; j2 < n_idx_active; ++j2) {
				const int_t idx2 = idx_active[j2];
				tmp_val += G->val[idx * (nineq + neq) + idx2] * lambda[idx2];
			}

			const real_t tmp_val2 = tmp_val + g[idx] - settings.alpha*settings.TOL_const/2.0; //G[idx,:]*lambda+c_max[idx]

			// implement tmp_val = lambda[idx] - omega/G[idx,idx]*(G[idx,:]*lambda + c[idx]);
			tmp_val = lambda[idx] - fact_tmp * (tmp_val + g[idx]);

			// do the projection
			if (tmp_val < 0.0) {
				tmp_val = 0.0;
			}
			// keep track of max(G[idx,:]*lambda+c_max[idx])
			else if (max_G_lambda_cmax < tmp_val2) {
				max_G_lambda_cmax = tmp_val2;
			}
			
			lambda[idx] = tmp_val;
		}
		
		// stopping criterion -> ensure that |lambda_{k+1}-lambda_{k}| is small and max(G lambda_{k+1} + c_max)_{lambda_{k+1}>0} <=0.0
		// the second part avoids cycling even if we solve the prox-equation only approximately
		if (normLambdaDiff() < settings.ABSTOL_PROX + settings.RELTOL_PROX * normLambdaStart() && max_G_lambda_cmax <= 0.0) {
			// niterprox=k; not sure if we should keep track of the prox iter
			return true;
		}
	}

	return false;
}

inline void SOCP::copyLambdaStart() {
	for (int_t k = 0; k < n_idx_active; ++k) {
		lambdastart[idx_active[k]] = lambda[idx_active[k]];
	}
	for (int_t k = 0; k < neq; ++k) {
		lambdastart[k] = lambda[k];
	}
}


inline real_t SOCP::normLambdaDiff() const {
	real_t res = 0;
	for (int_t k = 0; k < neq; ++k) {
		res += (lambdastart[k] - lambda[k]) * (lambdastart[k] - lambda[k]);
	}
	for (int_t k = 0; k < n_idx_active; ++k) {
		res += (lambdastart[idx_active[k]] - lambda[idx_active[k]]) * (lambdastart[idx_active[k]] - lambda[idx_active[k]]);
	}
	
	return Utils::getSqrt(res);
}


inline real_t SOCP::normLambdaStart() const {
	real_t res = 0;
	for (int_t k = 0; k < neq; ++k) {
		res += lambdastart[k] * lambdastart[k];
	}
	for (int_t k = 0; k < n_idx_active; ++k) {
		res += lambdastart[idx_active[k]] * lambdastart[idx_active[k]];
	}
	
	return Utils::getSqrt(res);
}


void SOCP::setUpWtrans() {
	W_trans_row_ptr = new int_t[nC[0]+nC[1]+(nnC-2)+1];
	// number of values in Aeq, Aineq:
	const int_t nval_Aeq_Aineq = A->row_ptr[nC[0] + nC[1]];
	
	W_trans_val = new real_t[W_trans_nnz];
	W_trans_col_ptr = new int_t[W_trans_nnz];

	for (int_t k = 0; k < nval_Aeq_Aineq; ++k) {
		W_trans_val[k] = A->val[k];
		W_trans_col_ptr[k] = A->col_idx[k];
	}
	for (int_t k = nval_Aeq_Aineq; k < W_trans_nnz; ++k) {
		W_trans_val[k] = 0.0;
		W_trans_col_ptr[k] = (k - nval_Aeq_Aineq) % nx;
	}

	// fill up W_trans_row_ptr;
	for (int_t k = 0; k < nC[0] + nC[1]; ++k) {
		W_trans_row_ptr[k] = A->row_ptr[k];
	}
	for (int_t k = nC[0]+nC[1]; k < nC[0] + nC[1] + (nnC-2); ++k) {
		W_trans_row_ptr[k] = nval_Aeq_Aineq + nx * (k - nC[0] - nC[1]);
	}
	W_trans_row_ptr[nineq + neq] = W_trans_nnz;
}

void SOCP::ComputeWTRow(int_t idx_nC, int_t idx_A, const real_t* v1, const real_t& v1_norm) {
	const int_t row_WT = idx_nC-2 + nC[0] + nC[1];

	// clear row first
	for (int_t k = W_trans_row_ptr[row_WT]; k < W_trans_row_ptr[row_WT+ 1]; ++k) {
		W_trans_val[k] = 0.0;
	}

	// calculate first element of [1, -v1/|v1|]  A
	for (int_t k = A->row_ptr[idx_A]; k < A->row_ptr[idx_A + 1]; ++k) {
		W_trans_val[W_trans_row_ptr[row_WT] + A->col_idx[k]] += A->val[k];
	}

	// calculate remaining elements of [1, -v1/|v1|]  A
	if (v1_norm >= ZERO_TRESH) {
		for (int_t j = 0; j < nC[idx_nC] - 1; ++j) {
			for (int_t k = A->row_ptr[idx_A + 1 + j]; k < A->row_ptr[idx_A + 2 + j]; ++k) {
				W_trans_val[W_trans_row_ptr[row_WT] + A->col_idx[k]] -= A->val[k] * v1[j] / v1_norm;
			}
		}
	}
	
}

void SOCP::updateG() {
	// are there any active SOCP constraints? -> move backwards through idx_active

	for (int_t j = n_idx_active - 1; j >= 0; --j) {
		const int_t idx_1 = idx_active[j];
		if (idx_1 < nC[0] + nC[1])
			break;

		// calculate inner products
		const real_t* v1 = &W_trans_val[W_trans_row_ptr[idx_active[j]]];

		// equality constraints
		for (int_t idx_2 = 0; idx_2 < neq; ++idx_2) {
			real_t out = 0.0;
			for (int_t k2 = W_trans_row_ptr[idx_2]; k2 < W_trans_row_ptr[idx_2 + 1]; ++k2) {
				out += W_trans_val[k2] * v1[W_trans_col_ptr[k2]];
			}
			G->val[idx_2 * (nineq + neq) + idx_1] = out;
			G->val[idx_1 * (nineq + neq) + idx_2] = out;
		}

		// remaining constraints
		for (int_t k = 0; k < n_idx_active; ++k) {
			const int_t idx_2 = idx_active[k];
			real_t out = 0.0;
			for (int_t k2 = W_trans_row_ptr[idx_2]; k2 < W_trans_row_ptr[idx_2 + 1]; ++k2) {
				out += W_trans_val[k2] * v1[W_trans_col_ptr[k2]];
			}
			G->val[idx_2 * (nineq + neq) + idx_1] = out;
			G->val[idx_1 * (nineq + neq) + idx_2] = out;
		}

	}
}