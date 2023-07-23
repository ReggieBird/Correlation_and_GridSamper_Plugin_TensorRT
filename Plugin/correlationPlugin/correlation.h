//
// Created by jerry on 2022/2/26.
// Modify by Reggie Bird on 2023/07/23.
//

#ifndef TENSORRT_CORRELATION_H
#define TENSORRT_CORRELATION_H

#include <cuda.h>
#include <cuda_runtime.h>

namespace torch{
namespace detail{

enum class CorrelationDataType{
    GFLOAT,
    GHALF
};
}
}
int32_t correlation_cuda(int32_t batchSize, const void* queryPtr, const void* referencePtr, void* const outputPtr,
    int32_t inp_C,int32_t inp_H, int32_t inp_W, int32_t out_H,int32_t out_W,
    int32_t inp_sN, int32_t inp_sC, int32_t inp_sH, int32_t inp_sW,
    int32_t out_sN, int32_t out_sC, int32_t out_sH, int32_t out_sW,
    int32_t kH,int32_t kW,int32_t patchH,int32_t patchW,int32_t padH, int32_t padW, int32_t dilation,
    torch::detail::CorrelationDataType dataType, cudaStream_t stream);

#endif // TENSORRT_CORRELATION_H
