#pragma once

#include "DefineSettings.h"
#include "SparseMatrixCRS.h"
#include "DenseMatrix.h"
#include "Utils.h"


class QuadraticProgram{
public:
	
	struct Settings{
		// default initialization
		Settings(): T(.1),alpha(.1),omega(1), TOL_const(1e-8),
					MAXITER(1000), ABSTOL(1e-6), 
					MAXITER_PROX(1000), ABSTOL_PROX(1e-11), RELTOL_PROX(1e-11) {};
		Settings(real_t T_, real_t alpha_, real_t omega_, real_t TOL_const_, int_t MAXITER_, real_t ABSTOL_, int_t MAXITER_PROX_, real_t ABSTOL_PROX_, real_t RELTOL_PROX_) {
			T = T_;
			alpha = alpha_;
			omega = omega_;
			TOL_const = TOL_const_;
			MAXITER = MAXITER_;
			ABSTOL = ABSTOL_;
			MAXITER_PROX = MAXITER_PROX_;
			ABSTOL_PROX = ABSTOL_PROX_;
			RELTOL_PROX = RELTOL_PROX_;
		};

		real_t T;
		real_t alpha;
		real_t omega;

		real_t TOL_const;

		int_t MAXITER;
		real_t ABSTOL;
		
		int_t MAXITER_PROX;
		real_t ABSTOL_PROX;
		real_t RELTOL_PROX;

		void printParameters(){
			printf("\n\n Parameters -------------------\n");
			printf("T:\t%e\n", T);
			printf("alpha:\t%e\n",alpha);
			printf("omega:\t%e\n\n",omega);
			printf("TOL_const:\t%e\n\n",TOL_const);
			printf("MAXITER:\t%e\n",(real_t)MAXITER);
			printf("ABSTOL:\t%e\n\n",ABSTOL);
			printf("MAXITER_PROX:\t%e\n",(real_t)MAXITER_PROX);
			printf("ABSTOL_PROX:\t%e\n",ABSTOL_PROX);
			printf("RELTOL_PROX:\t%e\n",RELTOL_PROX);
			printf(" ----------------------- \n\n");

		};
	};
	
	
	QuadraticProgram(const SparseMatrixCRS* H_, const SparseMatrixCRS* A_, const DenseMatrix* G_,
		const real_t* c_, const real_t* b_, const Settings& settings_, int_t neq_=0);
	~QuadraticProgram(){ delete[] x, delete[] neg_df, delete[] lambda, delete[] lambdastart, delete[] idx_active, delete[] g;	};

	void resetLambda();

	bool solve(const real_t* x0);


	int_t getniter() const{ return niter; };
	const real_t& getAvgNConstraints() const { return avg_n_constraints; }

	const real_t* getSolution() const{ return x; };
	const real_t* getLambda() const { return lambda; };

private:
	// g=\alpha g - W\T lambda
	void updateg();

	// neg_df+= W lambda, considering only the active constraints.
	void updateWlambda();


	bool doProx();

	// copies only the elements defined through idx_active, and equality constraints
	void copyLambdaStart();

	// calculates |lambdaStart-lambda|_2 --> only the active parts
	real_t normLambdaDiff() const;

	// calculates |lambdaStart|_2 --> only the active parts
	real_t normLambdaStart() const;

	int_t niter;

	const int_t nx;
	const int_t nineq;
	const int_t neq;

	const SparseMatrixCRS* H;
	const SparseMatrixCRS* A;
	const DenseMatrix* G;
	const real_t* c;
	const real_t* b;

	const Settings settings;

	real_t* x;
	real_t* neg_df;
	real_t* lambda;
	real_t* lambdastart;
	int_t* idx_active;
	int_t n_idx_active;
	real_t* g;

	real_t avg_n_constraints;
};

