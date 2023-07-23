#include "gridSampler.h"
#include "gridSampler.cuh"
#include <stdio.h>
#include <assert.h>

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

template <typename scalar_t>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void grid_sampler_2d_kernel(
    int32_t nthreads,
    const scalar_t* inputPtr,
    const scalar_t* gridPtr,      
    scalar_t* const outputPtr,      
    int32_t C,
    int32_t inp_H,
    int32_t inp_W,
    int32_t out_H, // same as grid_H
    int32_t out_W, // same as grid_W
    int32_t inp_sN,
    int32_t inp_sC,
    int32_t inp_sH,
    int32_t inp_sW,
    int32_t grid_sN,
    int32_t grid_sH,
    int32_t grid_sW,
    int32_t grid_sCoor,
    int32_t out_sN,
    int32_t out_sC,
    int32_t out_sH,
    int32_t out_sW,
    torch::detail::GridSamplerInterpolation interpolation_mode,
    torch::detail::GridSamplerPadding padding_mode,
    bool align_corners) {

  CUDA_KERNEL_LOOP(index, nthreads) {
    const int32_t w = index % out_W;
    const int32_t h = (index / out_W) % out_H;
    const int32_t n = index / (out_H * out_W);
    const int32_t grid_offset = n * grid_sN + h * grid_sH + w * grid_sW;

    // get the corresponding input x, y co-ordinates from grid
    scalar_t ix = gridPtr[grid_offset];
    scalar_t iy = gridPtr[grid_offset + grid_sCoor];

    ix = grid_sampler_compute_source_index(ix, inp_W, padding_mode, align_corners);
    iy = grid_sampler_compute_source_index(iy, inp_H, padding_mode, align_corners);

    if (interpolation_mode == torch::detail::GridSamplerInterpolation::Bilinear) {
      // get NE, NW, SE, SW pixel values from (x, y)
      int32_t ix_nw = static_cast<int32_t>(floorf(ix));
      int32_t iy_nw = static_cast<int32_t>(floorf(iy));
      int32_t ix_ne = ix_nw + 1;
      int32_t iy_ne = iy_nw;
      int32_t ix_sw = ix_nw;
      int32_t iy_sw = iy_nw + 1;
      int32_t ix_se = ix_nw + 1;
      int32_t iy_se = iy_nw + 1;

      // get surfaces to each neighbor:
      scalar_t nw = (static_cast<scalar_t>(ix_se) - ix)    * (static_cast<scalar_t>(iy_se) - iy);
      scalar_t ne = (ix    - ix_sw) * (static_cast<scalar_t>(iy_sw) - iy);
      scalar_t sw = (static_cast<scalar_t>(ix_ne) - ix)    * (iy    - iy_ne);
      scalar_t se = (ix    - ix_nw) * (iy    - iy_nw);

      // calculate bilinear weighted pixel value and set output pixel
      auto inp_ptr_NC = inputPtr + n * inp_sN;
      auto out_ptr_NCHW = outputPtr + n * out_sN + h * out_sH + w * out_sW;
      for (int32_t c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCHW += out_sC) {
        *out_ptr_NCHW = static_cast<scalar_t>(0);
        if (within_bounds_2d(iy_nw, ix_nw, inp_H, inp_W)) {
          *out_ptr_NCHW += inp_ptr_NC[iy_nw * inp_sH + ix_nw * inp_sW] * nw;
        }
        if (within_bounds_2d(iy_ne, ix_ne, inp_H, inp_W)) {
          *out_ptr_NCHW += inp_ptr_NC[iy_ne * inp_sH + ix_ne * inp_sW] * ne;
        }
        if (within_bounds_2d(iy_sw, ix_sw, inp_H, inp_W)) {
          *out_ptr_NCHW += inp_ptr_NC[iy_sw * inp_sH + ix_sw * inp_sW] * sw;
        }
        if (within_bounds_2d(iy_se, ix_se, inp_H, inp_W)) {
          *out_ptr_NCHW += inp_ptr_NC[iy_se * inp_sH + ix_se * inp_sW] * se;
        }
      }
    } else if (interpolation_mode == torch::detail::GridSamplerInterpolation::Nearest) {
      int32_t ix_nearest = static_cast<int32_t>(roundf(ix));
      int32_t iy_nearest = static_cast<int32_t>(roundf(iy));

      // assign nearest neighor pixel value to output pixel
      auto inp_ptr_NC = inputPtr + n * inp_sN;
      auto out_ptr_NCHW = outputPtr + n * out_sN + h * out_sH + w * out_sW;
      for (int32_t c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCHW += out_sC) {
        if (within_bounds_2d(iy_nearest, ix_nearest, inp_H, inp_W)) {
          *out_ptr_NCHW = inp_ptr_NC[iy_nearest * inp_sH + ix_nearest * inp_sW];
        } else {
          *out_ptr_NCHW = static_cast<scalar_t>(0);
        }
      }

    }
  }
}

// No shape checking needed here. See # NOTE [ grid_sampler Native Functions ].
int32_t grid_sampler_2d_cuda(int32_t batchSize, const void* inputPtr, const void* gridPtr,
  void* const outputPtr,
  int32_t C,
  int32_t inp_H,
  int32_t inp_W,
  int32_t out_H, // same as grid_H
  int32_t out_W, // same as grid_W
  int32_t inp_sN,
  int32_t inp_sC,
  int32_t inp_sH,
  int32_t inp_sW,
  int32_t grid_sN,
  int32_t grid_sH,
  int32_t grid_sW,
  int32_t grid_sCoor,
  int32_t out_sN,
  int32_t out_sC,
  int32_t out_sH,
  int32_t out_sW,
  torch::detail::GridSamplerInterpolation interpolation_mode,
  torch::detail::GridSamplerPadding padding_mode,
  bool align_corners, torch::detail::GridSamplerDataType dataType, cudaStream_t stream)
{

  int32_t count = out_H * out_W * batchSize; // will have inner loop over C
  if (count > 0) {
    if (dataType == torch::detail::GridSamplerDataType::GHALF)
    {
      grid_sampler_2d_kernel<half>
        <<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(
          count,
          static_cast<const half*>(inputPtr), static_cast<const half*>(gridPtr),
          static_cast<half*>(outputPtr),
          C, inp_H, inp_W, out_H, out_W, inp_sN, inp_sC, inp_sH, inp_sW,
          grid_sN, grid_sH, grid_sW, grid_sCoor, out_sN, out_sC, out_sH, out_sW,
          interpolation_mode,
          padding_mode,
          align_corners);
    }
    else
    {
      grid_sampler_2d_kernel<float>
        <<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(
          count,
          static_cast<const float*>(inputPtr), static_cast<const float*>(gridPtr),
          static_cast<float*>(outputPtr),
          C, inp_H, inp_W, out_H, out_W, inp_sN, inp_sC, inp_sH, inp_sW,
          grid_sN, grid_sH, grid_sW, grid_sCoor, out_sN, out_sC, out_sH, out_sW,
          interpolation_mode,
          padding_mode,
          align_corners);
    }
  }

  return cudaGetLastError() != cudaSuccess;
}

