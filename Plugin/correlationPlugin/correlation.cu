#include "correlation.cuh"
#include "correlation.h"
#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <cassert>

using half = __half;

// alignd with torch Macros.h
// CUDA_MAX_THREADS_PER_BLOCK is same for all architectures currently
constexpr unsigned int CUDA_MAX_THREADS_PER_BLOCK = 1024;
// CUDA_THREADS_PER_BLOCK_FALLBACK is the "canonical fallback" choice of block size.
// 256 is a good number for this fallback and should give good occupancy and
// versatility across all architectures.
constexpr unsigned int CUDA_THREADS_PER_BLOCK_FALLBACK = 256;

#define C10_MAX_THREADS_PER_BLOCK(val) (((val) <= CUDA_MAX_THREADS_PER_BLOCK) ? (val) : CUDA_THREADS_PER_BLOCK_FALLBACK)
#define C10_LAUNCH_BOUNDS_1(max_threads_per_block) __launch_bounds__((C10_MAX_THREADS_PER_BLOCK((max_threads_per_block))))

// aligned with KernelUtils.h
#define CUDA_KERNEL_LOOP(i, n) \
  int32_t _i_n_d_e_x = blockIdx.x * blockDim.x + threadIdx.x;                                \
  for (int32_t i=_i_n_d_e_x; _i_n_d_e_x < (n); _i_n_d_e_x+=blockDim.x * gridDim.x, i=_i_n_d_e_x)


constexpr int32_t CUDA_NUM_THREADS = 1024;
inline int32_t GET_BLOCKS(const int32_t N)
{
    assert(N > 0);
    return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

template <typename  scalar_t>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void correlation_kernel(int32_t nthreads,const scalar_t* queryPtr, const scalar_t* referencePtr, scalar_t* const outputPtr,
                                    int32_t C,int32_t inp_H, int32_t inp_W, int32_t out_H, int32_t out_W,
                                    int32_t inp_sN,int32_t inp_sC,int32_t inp_sH,int32_t inp_sW,
                                    int32_t out_sN,int32_t out_sC,int32_t out_sH,int32_t out_sW,
                                    int32_t kH,int32_t kW,int32_t patchH,int32_t patchW,int32_t padH,int32_t padW,int32_t dilation)
{
    CUDA_KERNEL_LOOP(index, nthreads){
        const int32_t w = index % out_W;
        const int32_t h = (index / out_W) % out_H;
        const int32_t n = index / (out_H * out_W);
        for(int32_t ph=0;ph<patchH;++ph){
            for(int32_t pw=0;pw<patchW;++pw){

                int32_t channelCounter = ph * patchH + pw;
                if(channelCounter%dilation!=0)
                    continue;
                int32_t channelIndex = channelCounter / dilation;

                auto out_ptr_NCHW = outputPtr + n * out_sN + channelIndex * out_sC + h * out_sH + w * out_sW;
                *out_ptr_NCHW = static_cast<scalar_t>(0);

                int32_t rh = h + ph - padH;
                int32_t rw = w + pw - padW;

                if(within_bounds_2d(rh,rw,inp_H,inp_W)){
                    for(int32_t pc=0;pc<C;++pc){
                        auto query_ptr_NCHW = queryPtr + n * inp_sN + pc * inp_sC + h * inp_sH + w * inp_sW;
                        auto reference_ptr_NCHW = referencePtr + n * inp_sN + pc * inp_sC + rh * inp_sH + rw * inp_sW;
                        *out_ptr_NCHW += static_cast<scalar_t>((*query_ptr_NCHW)*(*reference_ptr_NCHW));
                    }
                }
            }
        }
    }
}

int32_t correlation_cuda(int32_t batchSize, const void* queryPtr, const void* referencePtr, void* const outputPtr,
    int32_t inp_C,int32_t inp_H, int32_t inp_W, int32_t out_H,int32_t out_W,
    int32_t inp_sN, int32_t inp_sC, int32_t inp_sH, int32_t inp_sW,
    int32_t out_sN, int32_t out_sC, int32_t out_sH, int32_t out_sW,
    int32_t kH,int32_t kW,int32_t patchH,int32_t patchW,int32_t padH, int32_t padW, int32_t dilation,
    torch::detail::CorrelationDataType dataType, cudaStream_t stream)
{
    int32_t count = out_H * out_W * batchSize;
    if(count > 0){
        if(dataType == torch::detail::CorrelationDataType::GHALF){
            correlation_kernel<half>
                <<<GET_BLOCKS(count),CUDA_NUM_THREADS,0,stream>>>(
                    count,static_cast<const half*>(queryPtr),static_cast<const half*>(referencePtr),
                    static_cast<half*>(outputPtr),
                    inp_C,inp_H,inp_W,out_H,out_W,
                    inp_sN,inp_sC,inp_sH,inp_sW,
                    out_sN,out_sC,out_sH,out_sW,
                    kH,kW,patchH,patchW,padH,padW,dilation
                    );
        }
        else{
            correlation_kernel<float>
                <<<GET_BLOCKS(count),CUDA_NUM_THREADS,0,stream>>>(
                    count,static_cast<const float*>(queryPtr),static_cast<const float*>(referencePtr),
                    static_cast<float*>(outputPtr),
                    inp_C,inp_H,inp_W,out_H,out_W,
                    inp_sN,inp_sC,inp_sH,inp_sW,
                    out_sN,out_sC,out_sH,out_sW,
                    kH,kW,patchH,patchW,padH,padW,dilation
                );
        }
    }
    return cudaGetLastError()!=cudaSuccess;
}

