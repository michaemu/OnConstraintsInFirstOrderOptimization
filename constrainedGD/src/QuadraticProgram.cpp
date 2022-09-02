#include "QuadraticProgram.h"

QuadraticProgram::QuadraticProgram(const SparseMatrixCRS* H_, const SparseMatrixCRS* A_, const DenseMatrix* G_,
		const real_t* c_, const real_t* b_, const Settings& settings_, int_t neq_):nx(H_->nc),nineq(A_->nr - neq_),neq(neq_),
																H(H_),A(A_),G(G_),c(c_),b(b_),
																settings(settings_){
		x=new real_t[nx];
		neg_df=new real_t[nx];
		lambda=new real_t[nineq+neq];
		lambdastart=new real_t[nineq+neq];
		
		// reset lambda
		resetLambda();

		idx_active=new int_t[nineq];
		g=new real_t[nineq+neq];
	}


void QuadraticProgram::resetLambda(){
	// reset lambda
	for(int_t k=0;k<nineq+neq;++k){
		lambda[k]=0;
	}
}

bool QuadraticProgram::solve(const real_t* x0){

#ifdef DEBUG
		FILE* fp;
		fp= fopen("dump_trajectory.txt","w+");
#endif

		Utils::VectorCopy(x0,x,nx);
		
		avg_n_constraints = 0.0;

		for(int_t k=0;k<settings.MAXITER;++k){

#ifdef DEBUG
			Utils::printVecToFileColumnWise(x,nx,fp);
#endif

			// compute -\nabla f(x)
			H->vecMultAdd(x,c,-1.0,-1.0,neg_df);

			// check constraint violations
			A->vecMultAdd(x,b,1.0,-1.0,g);
			
			n_idx_active=0;
			for(int_t j=0;j<nineq;++j){ // equality constraints are always active -> we don't need to check those!
				if(g[j]<=settings.TOL_const){
					idx_active[n_idx_active++]=j;
				}
			}

			avg_n_constraints += real_t(n_idx_active) / real_t(nineq);

			if(neq>0 || n_idx_active>0){
				// compute -W\T df + alpha g --> stored in g
				updateg();

				// compute lambda (solve the complementary problem)
				doProx();

				// compute -\nabla f(x) + W \lambda
				updateWlambda();

			}

			// compute x+= -\nabla f(x) + W \lambda
			Utils::VectorAdd(neg_df,settings.T,x,nx);

			if(Utils::VectorMaxNorm(neg_df,nx) < settings.ABSTOL){ // convergence!
				
				niter=k+1;
				avg_n_constraints /= niter;
#ifdef DEBUG
				fclose(fp);
#endif
				return true;
			}

		}
		
		niter=settings.MAXITER;
		avg_n_constraints /= niter;

#ifdef DEBUG
		fclose(fp);
#endif
		return false;
	}


void QuadraticProgram::updateg(){
	
	// active inequality constraints
	for(int_t k=0;k<n_idx_active;++k){
		const int_t idx=idx_active[k];
		real_t tmp = 0.0;
		for(int_t j=A->row_ptr[idx];j<A->row_ptr[idx+1];++j){
			tmp+=A->val[j]*neg_df[A->col_idx[j]];
		}
		g[idx] = g[idx] * settings.alpha + tmp;
	}

	// equality constraints
	for(int_t idx=nineq;idx<nineq+neq;++idx){
		real_t tmp = 0.0;
		for(int_t j=A->row_ptr[idx];j<A->row_ptr[idx+1];++j){
			tmp+=A->val[j]*neg_df[A->col_idx[j]];
		}
		g[idx] = g[idx]*settings.alpha + tmp;
	}	
}

void QuadraticProgram::updateWlambda(){
		
		// loop over active inequality constraints
		for(int_t k=0;k<n_idx_active;++k){
			for(int_t j=A->row_ptr[idx_active[k]];j<A->row_ptr[idx_active[k]+1];++j){
				neg_df[A->col_idx[j]]+=A->val[j]*lambda[idx_active[k]];
			}
		}

		// loop over equality constraints
		for(int_t k=nineq;k<nineq+neq;++k){
			for(int_t j=A->row_ptr[k];j<A->row_ptr[k+1];++j){
				neg_df[A->col_idx[j]]+=A->val[j]*lambda[k];
			}
		}

	}



bool QuadraticProgram::doProx(){
		// int_t niterprox=settings.MAXITER_PROX; // not sure if we should keep track of the prox-iter

		for(int_t k=0;k<settings.MAXITER_PROX;++k){
			copyLambdaStart(); // lambdastart <- lambda
			real_t max_G_lambda_cmax = 0.0;

			for(int_t j=0;j<n_idx_active;++j){
				const int_t idx=idx_active[j];

				// fact_tmp=omega/G[idx,idx]
				const real_t fact_tmp=settings.omega/G->val[idx*(nineq+neq)+idx]; // add interface to G
				real_t tmp_val=0;

				// calculate G[idx,:]*lambda
				for(int_t j2=0;j2<n_idx_active;++j2){
					const int_t idx2=idx_active[j2];
					tmp_val+=G->val[idx*(nineq+neq)+idx2]*lambda[idx2];
				}
				for(int_t idx2=nineq;idx2<nineq+neq;++idx2){
					tmp_val+=G->val[idx*(nineq+neq)+idx2]*lambda[idx2];
				}

				const real_t tmp_val2 = tmp_val + g[idx] - settings.alpha*settings.TOL_const/2.0; // G[idx,:]*lambda + c_max[idx];

				// implement tmp_val = lambda[idx] - omega/G[idx,idx]*(G[idx,:]*lambda + c[idx]);
				tmp_val=lambda[idx]-fact_tmp*(tmp_val+g[idx]);
				
				// do the projection
				if (tmp_val < 0.0) {
					tmp_val = 0.0;
				}
				// keep track of max(G[idx,:]*lambda+c_max[idx])
				else if(max_G_lambda_cmax < tmp_val2){
					max_G_lambda_cmax = tmp_val2;
				}

				lambda[idx]=tmp_val;
			}
			for(int_t idx=nineq;idx<nineq+neq;++idx){
				// fact_tmp=omega/G[idx,idx]
				const real_t fact_tmp=settings.omega/G->val[idx*(nineq+neq)+idx]; // add interface to G
				real_t tmp_val=0;

				// calculate G[idx,:]*lambda
				for(int_t j2=0;j2<n_idx_active;++j2){
					const int_t idx2=idx_active[j2];
					tmp_val+=G->val[idx*(nineq+neq)+idx2]*lambda[idx2];
				}
				for(int_t idx2=nineq;idx2<nineq+neq;++idx2){
					tmp_val+=G->val[idx*(nineq+neq)+idx2]*lambda[idx2];
				}

				// implement lambda=lambda-omega/G[idx,idx]*(G[idx,:]*lambda+c[idx]);
				lambda[idx]-=fact_tmp*(tmp_val+g[idx]);
			}

			// stopping criterion -> ensure that |lambda_{k+1}-lambda_{k}| is small and max(G lambda_{k+1} + c_max)_{lambda_{k+1}>0} <=0.0
			// the second part avoids cycling even if we solve the prox-equation only approximately
			if(normLambdaDiff() < settings.ABSTOL_PROX + settings.RELTOL_PROX*normLambdaStart() && max_G_lambda_cmax <=0.0){ 
				// niterprox=k; not sure if we should keep track of the prox iter
				return true;
			}
		}

		return false;
	}

inline void QuadraticProgram::copyLambdaStart(){
		for(int_t k=0;k<n_idx_active;++k){
			lambdastart[idx_active[k]]=lambda[idx_active[k]];
		}
		for(int_t k=nineq;k<nineq+neq;++k){
			lambdastart[k]=lambda[k];
		}
	}


inline real_t QuadraticProgram::normLambdaDiff() const{
		real_t res=0;
		for(int_t k=0;k<n_idx_active;++k){
			res+=(lambdastart[idx_active[k]]-lambda[idx_active[k]])*(lambdastart[idx_active[k]]-lambda[idx_active[k]]);
		}
		for(int_t k=nineq;k<nineq+neq;++k){
			res+=(lambdastart[k]-lambda[k])*(lambdastart[k]-lambda[k]);
		}
		return Utils::getSqrt(res);
	}

inline real_t QuadraticProgram::normLambdaStart() const{
		real_t res=0;
		for(int_t k=0;k<n_idx_active;++k){
			res+=lambdastart[idx_active[k]]*lambdastart[idx_active[k]];
		}
		for(int_t k=nineq;k<nineq+neq;++k){
			res+=lambdastart[k]*lambdastart[k];
		}
		return Utils::getSqrt(res);
	}