#ifndef CUDA_HELPER_H
#define CUDA_HELPER_H

#include <cmath>

#define cudaCall(x) { if((x) != cudaSuccess) { \
    printf("Error at %s:%d -- %s\n",__FILE__,__LINE__, cudaGetErrorString(x)); \
    fflush(stdin); \
    exit(EXIT_FAILURE);}}

#define cudaKernelRun(x) { (x); \
    cudaError_t error = cudaGetLastError(); \
    if(error != cudaSuccess) { \
    printf("Error at %s:%d -- %s\n",__FILE__,__LINE__, cudaGetErrorString(error)); \
    fflush(stdin); \
    exit(EXIT_FAILURE);}}

inline cudaError_t cudaMemoryAllocate(void *** buffer, int elementSize, int elementsCount)
{
    /*const size_t Mb = 1<<20; // Assuming a 1Mb page size here

    size_t available, total;
    cudaMemGetInfo(&available, &total);

    size_t nwords = ceil(total / elementSize);
    size_t words_per_Mb = ceil(Mb / elementSize);

    while(cudaMalloc(buffer,  elementsCount * elementSize) == cudaErrorMemoryAllocation)
    {
        nwords -= words_per_Mb;
        if( nwords  < words_per_Mb)
        {
            return cudaErrorMemoryAllocation;
        }
    }

    return cudaSuccess;*/
    return cudaMalloc(buffer,  elementsCount * elementSize);
}

#endif // CUDA_HELPER_H
