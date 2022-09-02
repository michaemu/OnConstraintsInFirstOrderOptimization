#include "SparseMatrixCRS.h"

#include <stdio.h>


SparseMatrixCRS::SparseMatrixCRS(int_t nrin, int_t ncin, int_t nnzin, 
								 real_t* val_, int_t* col_idx_, int_t* row_ptr_):
									nr(nrin),nc(ncin),nnz(nnzin),val(val_),col_idx(col_idx_),row_ptr(row_ptr_),freeMem(false){
};

void SparseMatrixCRS::vecMultAdd(const real_t* x, real_t* y, const real_t& a, const real_t& b) const{
	for(int_t k=0;k<nr;++k){
		y[k]*=b;
		for(int_t j=row_ptr[k];j<row_ptr[k+1];++j){
			y[k]+=a*val[j]*x[col_idx[j]];
		}
	}

}

void SparseMatrixCRS::vecMultAdd(const real_t* x, const real_t* y, const real_t& a, const real_t& b, real_t* out) const{
	for(int_t k=0;k<nr;++k){
		out[k]=y[k]*b;
		for(int_t j=row_ptr[k];j<row_ptr[k+1];++j){
			out[k]+=a*val[j]*x[col_idx[j]];
		}
	}
	
}


void SparseMatrixCRS::vecMult(const real_t* vec, real_t* out) const{
	for(int_t k=0;k<nr;++k){
		out[k]=0;
		for(int_t j=row_ptr[k];j<row_ptr[k+1];++j){
			out[k]+=val[j]*vec[col_idx[j]];
		}
	}
}

real_t SparseMatrixCRS::operator()(int_t row, int_t col) const {
	for (int_t k = row_ptr[row]; k < row_ptr[row + 1]; ++k) {
		if (col_idx[k] == col) {
			return val[k];
		}
		// assuming that col_idx is ordered we can terminate early.
		else if (col_idx[k] > col) {
			return 0.0;
		}
	}
	return 0.0;
}

void SparseMatrixCRS::print() const{
	printf("sparse matrix ----------\n");
	int_t val_idx=0;
	for(int_t k=0;k<nr;++k){
		for(int_t j=0;j<nc;++j){
			if(val_idx >= row_ptr[k] && val_idx<row_ptr[k+1] && col_idx[val_idx]==j){
				printf("%f\t",val[val_idx++]);
			}else{
				printf("%f\t",0.0);
			}
		}
		printf("\n");
	}
	printf("\n");
}