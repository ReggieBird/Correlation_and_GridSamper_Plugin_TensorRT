/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

 // Modify by Reggie Bird on 2023/07/23

#ifndef GRID_SAMPLER_H
#define GRID_SAMPLER_H

#include <cuda.h>
#include <cuda_runtime.h>

namespace torch {

namespace detail {

  enum class GridSamplerInterpolation {Bilinear, Nearest};
  enum class GridSamplerPadding {Zeros, Border, Reflection};
  enum class GridSamplerDataType {GFLOAT, GHALF};

}  // namespace detail

} // namespace torch


// function naming is algined with Torch
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
                            bool align_corners, torch::detail::GridSamplerDataType dataType, cudaStream_t stream);

#endif