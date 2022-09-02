#include "DenseMatrix.h"
#include <stdio.h>


void DenseMatrix::print() const{
	printf("dense matrix -------\n");
	for(int_t k=0;k<nr;++k){
		for(int_t j=0;j<nc;++j){
			printf("%f\t",val[k*nc+j]);
		}
		printf("\n");
	}
	printf("\n");

}

inline void DenseMatrix::vecMult(const real_t* vec, real_t* out) const{
	for(int_t k=0;k<nr;++k){
		out[k]=0;
		for(int_t j=0;j<nc;++j){
			out[k]+=val[k*nc+j]*vec[j];
		}
	}
}

inline void DenseMatrix::vecMultAdd(const real_t* x, real_t* y, const real_t& a, const real_t& b) const{
	for(int_t k=0;k<nr;++k){
		y[k]*=b;
		for(int_t j=0;j<nc;++j){
			y[k]+=a*val[k*nc+j]*x[j];
		}
	}
}


inline const real_t* DenseMatrix::operator[](int_t rowidx) const{
	return val+rowidx*nc;
}

inline const real_t& DenseMatrix::operator()(int_t rowidx, int_t colidx) const{
	return val[rowidx*nc+colidx];
}