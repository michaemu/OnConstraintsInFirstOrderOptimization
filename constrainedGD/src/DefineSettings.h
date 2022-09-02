#pragma once

// precision
typedef double real_t;
typedef int int_t;
typedef unsigned int uint_t;


// zero and infinity
#define ZERO_TRESH 1e-14
#define INFVAL 1e16


#ifndef __SINGLE_OBJECT__

#define OUTPUT_ITER
//#define DEBUG
#define TIME

#ifdef DEBUG
	#define DEBUG_PROX
	//#define LOAD_DEBUG
#endif


#endif
