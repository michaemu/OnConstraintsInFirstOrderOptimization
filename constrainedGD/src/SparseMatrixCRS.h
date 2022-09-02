#pragma once

#include "DefineSettings.h"
#include "Utils.h"


class SparseMatrixCRS{
public:
	friend class QuadraticProgram;
	friend class QuadraticProgram2;
	friend class QuadraticProgramPGD;
	friend class SOCP;

	SparseMatrixCRS():nr(0),nc(0),nnz(0),val(0),col_idx(0),row_ptr(0),freeMem(false){ };
	SparseMatrixCRS(int_t nrin, int_t ncin, int_t nnzin, real_t* val_, int_t* col_idx_, int_t* row_ptr_);
	~SparseMatrixCRS(){if(freeMem){ delete[] val; delete[] col_idx; delete[] row_ptr; }};

	// implements y=a*Mat*x+b*y
	// x,y must be preallocated, x and y need to be of approriate size!!!
	void vecMultAdd(const real_t* x, real_t* y, const real_t& a, const real_t& b) const;
	
	// implements out=a*Mat*x+b*y
	// x,y,out must be preallocated, x and y need to be of approriate size!!!
	void vecMultAdd(const real_t* x, const real_t* y, const real_t& a, const real_t& b, real_t* out) const;
	
	// implements y=Mat*x
	// out must be preallocated and be of appropriate size!!!
	void vecMult(const real_t* vec, real_t* out) const;

	real_t operator()(int_t row, int_t col) const;

	void print() const;

	void freeMemory(bool freeMem_){ freeMem=freeMem_; }

	int_t getnRows() const { return nr; }
	int_t getnCols() const { return nc; }
	int_t getnnz() const { return nnz; }

	const real_t* getVal() const { return val; }
	const int_t* getColIdx() const { return col_idx; }
	const int_t* getRowPtr() const { return row_ptr; }

private:
	bool freeMem;

	real_t* val;
	int_t* col_idx;
	int_t* row_ptr;

	int_t nr,nc,nnz;
};