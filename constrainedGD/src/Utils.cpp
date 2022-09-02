#include "Utils.h"
#include <stdio.h>

#ifdef __linux__ 
	#include <time.h>
#else
		#include <windows.h>
#endif




void Utils::LoadSparseMatrixRowPlain(const char* str,int_t*& row_ptr, int_t*& col_idx, real_t*& val,
								int_t& nr, int_t& nc, int_t& nnz){
		std::string filename_base=str;
		filename_base.append("_n.txt");

		int_t size[4];
		readFromFile(size,4,filename_base.c_str());
		
		nr=size[0];
		nc=size[1];
		int_t row_ptr_length=size[2];
		nnz=size[3];

		// read column pointer
		filename_base=str;
		filename_base.append("_ia.txt");
		row_ptr=new int_t[row_ptr_length];
		readFromFile(row_ptr,row_ptr_length,filename_base.c_str());


		// read row indices
		filename_base=str;
		filename_base.append("_ja.txt");
		col_idx=new int_t[nnz];
		readFromFile(col_idx,nnz,filename_base.c_str());


		filename_base=str;
		filename_base.append("_ra.txt");
		val=new real_t[nnz];
		readFromFile(val,nnz,filename_base.c_str());

	}


void Utils::LoadDenseMatrix(const char* str, real_t*& value, int_t& nr, int_t& nc){
	std::string filename_base=str;
	filename_base.append("_n.txt");

	int_t size[2];
	readFromFile(size,2,filename_base.c_str());

	int_t n=size[0];
	int_t m=size[1];

	nr=n;	// number of rows
	nc=m;	// number of columns

	filename_base=str;
	filename_base.append("_d.txt");
	value=new real_t[n*m];
	readFromFile(value,n*m,filename_base.c_str());
}

void Utils::LoadVec(const char* str, real_t** vec, int_t& nv){
		std::string filename_base=str;
		filename_base.append(".txt");

		int_t size[1];
		readFromFile(size,1,filename_base.c_str());

		nv=size[0];
		real_t* tmp=new real_t[nv+1];
		
		readFromFile(tmp,nv+1,filename_base.c_str());
		*vec=new real_t[nv];

		for(int_t k=1;k<nv+1;++k){
			(*vec)[k-1]=tmp[k];
		}

		delete [] tmp;
	}

void Utils::LoadVec(const char* str, int_t** vec, int_t& nv) {
	std::string filename_base = str;
	filename_base.append(".txt");

	int_t size[1];
	readFromFile(size, 1, filename_base.c_str());

	nv = size[0];
	real_t* tmp = new real_t[nv + 1];

	readFromFile(tmp, nv + 1, filename_base.c_str());
	*vec = new int_t[nv];

	for (int_t k = 1; k < nv + 1; ++k) {
		(*vec)[k - 1] = (int_t)(tmp[k]+.1);  // ensure to be robust against round-off errors
	}

	delete[] tmp;
}

void Utils::FillVec(real_t* v, int_t nv, real_t fillvalue){
	for(int_t k=0;k<nv;++k){
		v[k]=fillvalue;
	}
}

void Utils::PrintSparseMat(const int_t* row_ptr, const int_t* col_idx, const real_t* val, int_t nr, int_t nc, int_t nnz){
		int_t val_idx=0;
		for(int_t k=0;k<nr;++k){
			for(int_t j=0;j<nc;++j){
				if(val_idx<nnz && col_idx[val_idx]==j){
					printf("%f\t",val[val_idx++]);
				}else{
					printf("%f\t",0.0);
				}
			}
		}

	}	

void Utils::PrintVec(const real_t* v, int_t nv){
		for(int_t k=0;k<nv;++k){
			printf("%f\n",v[k]);
		}
	}

void Utils::PrintVec(const int_t* v, int_t nv){
		for(int_t k=0;k<nv;++k){
			printf("%i\n",v[k]);
		}
	}

void Utils::PrintMat(const real_t* val, int_t nr, int_t nc){
		for(int_t k=0;k<nr;++k){
			for(int_t j=0;j<nc;++j){
				printf("%f\t",val[k*nc+j]);
			}
			printf("\n");
		}
	}


	// Important; The function assumes that col_idx is sorted (ascending)!
	// Moreover, col_ptr must be an array of lenght nCols+1 (already initialized).
void Utils::ConvertCOSToCCS(const int_t* col_idx, const int_t& nnZ,int_t* col_ptr){
		int_t oldCol=0;
		col_ptr[0]=0;
		for(int_t k=0;k<nnZ;++k){
			while(oldCol!=col_idx[k]){
				col_ptr[++oldCol]=k;
			}
		}
		col_ptr[oldCol+1]=nnZ;
	}

void Utils::DotProduct(const real_t* a, const real_t* b, const int& n, real_t& res){
		res=0;
		for(int_t i=0;i<n;++i){
			res+=a[i]*b[i];
		}
	}


	
void Utils::VectorAdd(const real_t* a, const real_t* b, real_t* c, int_t n){
		for(int_t k=0;k<n;++k){
			c[k]=a[k]+b[k];
		}
	}

void Utils::VectorAdd(const real_t* a, const real_t& alpha, real_t* out, int_t n){
	for(int_t k=0;k<n;++k){
		out[k]+=alpha*a[k];
	}
}

void Utils::VectorSubstract(const real_t* a, const real_t* b, real_t* c, int_t n){
		for(int_t k=0;k<n;++k){
			c[k]=a[k]-b[k];
		}
	}

void Utils::VectorAddMultiply(const real_t* a, real_t alpha, const real_t* b, real_t beta, real_t* c, int_t n){
		for(int_t k=0;k<n;++k){
			c[k]=alpha*a[k]+beta*b[k];
		}
	}

void Utils::VectorCopy(const real_t* a, real_t* b, int_t n){
		for(int_t k=0;k<n;++k){
			b[k]=a[k];
		}
	}


real_t Utils::VectorNorm(const real_t* a, int_t n){
		real_t res=0;
		for(int_t k=0;k<n;++k){
			res+=a[k]*a[k];
		}
		return getSqrt(res);
	}

real_t Utils::VectorMaxNorm(const real_t* a, int_t n){
		real_t res=0;
		for(int_t k=0;k<n;++k){
			res=abs(a[k])>res? abs(a[k]) : res;
		}
		return res;
}

	// returns ||a-b||_2
real_t Utils::VectorNormDiff(const real_t* a, const real_t* b, int_t n){
		real_t res=0;
		for(int_t k=0;k<n;++k){
			res+=(a[k]-b[k])*(a[k]-b[k]);
		}
		return Utils::getSqrt(res);
	}

	// component wise multiplication of the vectors a and b
void Utils::VectorMult(const real_t* a, const real_t* b, real_t* c, int_t n){
		for(int_t k=0;k<n;++k){
			c[k]=a[k]*b[k];
		}
	}

int_t Utils::factorial(int_t n){
		return (n == 1 || n == 0) ? 1 : factorial(n - 1) * n;
	}



	/*
	 *	r e a d F r o m F i l e
	 */
int_t Utils::readFromFile(real_t* data, int_t n, const char* datafilename){
		return readFromFile( data, n, 1, datafilename );
	}


	
int_t Utils::readFromFile(int_t* data, int_t n, const char* datafilename){
		
		int_t i;
		FILE* datafile;

		if ( ( datafile = fopen( datafilename, "r" ) ) == 0 )
		{
			printf("\n\runable to read file %s\n",datafilename);
			return -1;
		}

		for(i=0; i<n; ++i)
		{
			if(fscanf( datafile, "%d\n", &(data[i]) ) == 0)
			{
				fclose( datafile );
				printf("\n\runable to read file %s\n",datafilename);
				return -1;
			}
		}

		fclose( datafile );

		return 1;
	}

	
	/*
	*	r e a d F r o m F i l e
	*/
int_t Utils::readFromFile(real_t* data, int_t nrow, int_t ncol, const char* datafilename){
		int_t i, j;
		real_t float_data;
		FILE* datafile;

		if ( ( datafile = fopen( datafilename, "r" ) ) == 0 )
		{
			printf("\n\runable to read file %s\n",datafilename);
			return -1;
		}

		for( i=0; i<nrow; ++i )
		{
			for( j=0; j<ncol; ++j )
			{
				
				if ( fscanf( datafile, "%lf ", &float_data ) == 0 )
				{
					fclose( datafile );
					printf("\n\runable to read file %s\n",datafilename);
					return -1;
				}
				data[i*ncol + j] = ( (real_t) float_data );
			}
		}

		fclose( datafile );

		return -1;
}

void Utils::printVecToFileColumnWise(real_t* data, int_t n, FILE* fp){
	for(int_t k=0;k<n;++k){
		fprintf(fp,"%.18f\t",data[k]);
	}
	fprintf(fp,"\n");
}


real_t Utils::getCPUtime(){
	real_t current_time = -1.0;

#ifdef __linux__ 
	timespec tmp;
	clock_gettime(CLOCK_REALTIME,&tmp);
	current_time=tmp.tv_sec;
#else
	_LARGE_INTEGER counter, frequency;
	QueryPerformanceFrequency(&frequency);
	QueryPerformanceCounter(&counter);
	current_time = ((real_t) counter.QuadPart) / ((real_t) frequency.QuadPart);
	
#endif
	return current_time;
}