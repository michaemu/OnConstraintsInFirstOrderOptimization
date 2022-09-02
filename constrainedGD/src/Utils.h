#pragma once

#include <math.h>
#include <string>
#include "DefineSettings.h"


class Utils{
	public:

	static void LoadSparseMatrixRowPlain(const char* str,int_t*& row_ptr, int_t*& col_idx, real_t*& val,
								int_t& nr, int_t& nc, int_t& nnz);
	static void LoadDenseMatrix(const char* str, real_t*& value, int_t& nr, int_t& nc);
	
	static void LoadVec(const char* str, real_t** vec, int_t& nv);
	static void LoadVec(const char* str, int_t** vec, int_t& nv);

	static void FillVec(real_t* v, int_t nv, real_t fillvalue);

	static void PrintVec(const real_t* v, int_t nv);

	static void PrintVec(const int_t* v, int_t nv);

	static void PrintMat(const real_t* v, int_t nc, int_t nr);

	static void PrintSparseMat(const int_t* row_ptr, const int_t* col_idx, const real_t* val, int_t nr, int_t nc, int_t nnz);
	
	// Important; The function assumes that col_idx is sorted (ascending)!
	// Moreover, col_ptr must be an array of lenght nCols+1 (already initialized).
	static void ConvertCOSToCCS(const int_t* col_idx, const int_t& nnZ, int_t* col_ptr);

	static void DotProduct(const real_t* a, const real_t* b, const int& n, real_t& res);

	// implements a+b=c, where a,b,c are n-dimensional
	static void VectorAdd(const real_t* a, const real_t* b, real_t* c, int_t n);

	// implements out+=alpha*a
	static void VectorAdd(const real_t* a, const real_t& alpha, real_t* out, int_t n);

	// implements a-b=c, where a,b,c, are n-dimensional
	static void VectorSubstract(const real_t* a, const real_t* b, real_t* c, int_t n);

	// implements a*alpha+b*beta=c, where a,b,c are n-dimensional, and alpha and beta are scalars.
	static void VectorAddMultiply(const real_t* a, real_t alpha, const real_t* b, real_t beta, real_t* c, int_t n);

	// implements b=a
	static void VectorCopy(const real_t* a, real_t* b, int_t n);

	// returns |a|_2
	static real_t VectorNorm(const real_t* a, int_t n);

	// returns |a|_\infty
	static real_t VectorMaxNorm(const real_t* a, int_t n);

	// returns |a-b|_2
	static real_t VectorNormDiff(const real_t* a, const real_t* b, int_t n);
	
	// component wise multiplication of the vectors a and b
	static void VectorMult(const real_t* a, const real_t* b, real_t* c, int_t n);

	static int_t factorial(int_t n);

	static int_t isZero(const real_t& a,const real_t& tresh){
		return abs(a)<=tresh? 1:0;
	}

	static real_t getSqrt(const real_t& a){
#ifdef __linux__ 
		return a>=0? sqrt(a): 0;
#else
		return a>=0? sqrt(a): 0;
#endif

	}

	static int_t readFromFile(real_t* data, int_t nrow, int_t ncol, const char* datafilename);
	static int_t readFromFile(int_t* data, int_t n, const char* datafilename);
	static int_t readFromFile(real_t* data, int_t n, const char* datafilename);

	static void printVecToFileColumnWise(real_t* data, int_t n, FILE* fp);

	static real_t getCPUtime();

};