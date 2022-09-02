#pragma once

#include "DefineSettings.h"

class DenseMatrix{
public:
	friend class QuadraticProgram;
	friend class SOCP;

	DenseMatrix():nr(0),nc(0),val(0),freeMem(false){}
	DenseMatrix(int_t nrin, int_t ncin, real_t* val_):nr(nrin),nc(ncin),val(val_),freeMem(false){}
	
	~DenseMatrix(){ if(freeMem){ delete[] val; } }

	// implements y=a*Mat*x+b*y
	// y must be preallocated, x and y need to be of approriate size!!!
	void vecMultAdd(const real_t* x, real_t* y, const real_t& a, const real_t& b) const;


	// implements y=Mat*x
	// out must be preallocated and be of appropriate size!!!
	void vecMult(const real_t* vec, real_t* out) const;

	// print
	void print() const;

	const real_t* operator[](int_t rowidx) const;
	const real_t& operator()(int_t rowidx, int_t colidx) const;

	void freeMemory(bool freeMem_){ freeMem=freeMem_; }


	int_t getnRows() const { return nr; }
	int_t getnCols() const { return nc; }

	const real_t* getVal() const { return val; }

private:
	bool freeMem;

	real_t* val;
	int_t nr, nc;
};