#ifndef _DEBUG_H_
#define _DEBUG_H_

#if defined(DEBUG)
    #include <stdio.h>

    #define CHECKED_CUDA_CALL(C) {
        C;
        if(cudaGetLastError() != cudaSuccess) fprintf(stderr, "%s\n", cudaGetErrorString(err));
    }

#else
	/* Declare empty macros if debug is not enabled */
	#define CHECKED_CUDA_CALL(C) C
#endif/* DEBUG */

#endif /* _DEBUG_H_ */