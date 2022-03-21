// Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
//
// This work is made available under the Nvidia Source Code License-NC.
// To view a copy of this license, visit
// https://nvlabs.github.io/stylegan2/license.html

#include <torch/types.h>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_runtime.h>

static __host__ __device__ __forceinline__ int64_t floor_div(int64_t a, int64_t b) {
  int64_t c = a / b;

  if (c * b > a) {
    c--;
  }

  return c;
}

struct UpFirDn2DKernelParams {
  int64_t up_x;
  int64_t up_y;
  int64_t down_x;
  int64_t down_y;
  int64_t pad_x0;
  int64_t pad_x1;
  int64_t pad_y0;
  int64_t pad_y1;

  int64_t major_dim;
  int64_t in_h;
  int64_t in_w;
  int64_t minor_dim;
  int64_t kernel_h;
  int64_t kernel_w;
  int64_t out_h;
  int64_t out_w;
  int64_t loop_major;
  int64_t loop_x;
};

template <typename scalar_t>
__global__ void upfirdn2d_kernel_large(scalar_t *out, const scalar_t *input,
                                       const scalar_t *kernel,
                                       const UpFirDn2DKernelParams p) {
  int64_t minor_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t out_y = minor_idx / p.minor_dim;
  minor_idx -= out_y * p.minor_dim;
  int64_t out_x_base = blockIdx.y * p.loop_x * blockDim.y + threadIdx.y;
  int64_t major_idx_base = blockIdx.z * p.loop_major;

  if (out_x_base >= p.out_w || out_y >= p.out_h ||
      major_idx_base >= p.major_dim) {
    return;
  }
  int v_zero = 0;
  int64_t mid_y = out_y * p.down_y + p.up_y - 1 - p.pad_y0;
  int64_t in_y = min(max(int(floor_div(mid_y, p.up_y)), v_zero), int(p.in_h));
  int64_t h = min(max(int(floor_div(mid_y + p.kernel_h, p.up_y)), v_zero), int(p.in_h)) - in_y;
  int64_t kernel_y = mid_y + p.kernel_h - (in_y + 1) * p.up_y;
  int64_t zzero = 0;
  for (int64_t loop_major = 0, major_idx = major_idx_base;
       loop_major < p.loop_major && major_idx < p.major_dim;
       loop_major++, major_idx++) {
    for (int64_t loop_x = 0, out_x = out_x_base;
         loop_x < p.loop_x && out_x < p.out_w; loop_x++, out_x += blockDim.y) {
      int64_t mid_x = out_x * p.down_x + p.up_x - 1 - p.pad_x0;
      int64_t in_x = min(max(int(floor_div(mid_x, p.up_x)), v_zero), int(p.in_w));
      int64_t w = min(max(floor_div(mid_x + p.kernel_w, p.up_x),zzero), p.in_w) - in_x;
      int64_t kernel_x = mid_x + p.kernel_w - (in_x + 1) * p.up_x;

      const scalar_t *x_p =
          &input[((major_idx * p.in_h + in_y) * p.in_w + in_x) * p.minor_dim +
                 minor_idx];
      const scalar_t *k_p = &kernel[kernel_y * p.kernel_w + kernel_x];
      int64_t x_px = p.minor_dim;
      int64_t k_px = -p.up_x;
      int64_t x_py = p.in_w * p.minor_dim;
      int64_t k_py = -p.up_y * p.kernel_w;

      scalar_t v = 0.0f;

      for (int64_t y = 0; y < h; y++) {
        for (int64_t x = 0; x < w; x++) {
          v += static_cast<scalar_t>(*x_p) * static_cast<scalar_t>(*k_p);
          x_p += x_px;
          k_p += k_px;
        }

        x_p += x_py - w * x_px;
        k_p += k_py - w * k_px;
      }

      out[((major_idx * p.out_h + out_y) * p.out_w + out_x) * p.minor_dim +
          minor_idx] = v;
    }
  }
}

template <typename scalar_t, int64_t up_x, int64_t up_y, int64_t down_x, int64_t down_y,
          int64_t kernel_h, int64_t kernel_w, int64_t tile_out_h, int64_t tile_out_w>
__global__ void upfirdn2d_kernel(scalar_t *out, const scalar_t *input,
                                 const scalar_t *kernel,
                                 const UpFirDn2DKernelParams p) {
  const int64_t tile_in_h = ((tile_out_h - 1) * down_y + kernel_h - 1) / up_y + 1;
  const int64_t tile_in_w = ((tile_out_w - 1) * down_x + kernel_w - 1) / up_x + 1;

  __shared__ volatile float sk[kernel_h][kernel_w];
  __shared__ volatile float sx[tile_in_h][tile_in_w];

  int64_t minor_idx = blockIdx.x;
  int64_t tile_out_y = minor_idx / p.minor_dim;
  minor_idx -= tile_out_y * p.minor_dim;
  tile_out_y *= tile_out_h;
  int64_t tile_out_x_base = blockIdx.y * p.loop_x * tile_out_w;
  int64_t major_idx_base = blockIdx.z * p.loop_major;

  if (tile_out_x_base >= p.out_w | tile_out_y >= p.out_h |
      major_idx_base >= p.major_dim) {
    return;
  }

  for (int64_t tap_idx = threadIdx.x; tap_idx < kernel_h * kernel_w;
       tap_idx += blockDim.x) {
    int64_t ky = tap_idx / kernel_w;
    int64_t kx = tap_idx - ky * kernel_w;
    scalar_t v = 0.0;

    if (kx < p.kernel_w & ky < p.kernel_h) {
      v = kernel[(p.kernel_h - 1 - ky) * p.kernel_w + (p.kernel_w - 1 - kx)];
    }

    sk[ky][kx] = v;
  }

  for (int64_t loop_major = 0, major_idx = major_idx_base;
       loop_major < p.loop_major & major_idx < p.major_dim;
       loop_major++, major_idx++) {
    for (int64_t loop_x = 0, tile_out_x = tile_out_x_base;
         loop_x < p.loop_x & tile_out_x < p.out_w;
         loop_x++, tile_out_x += tile_out_w) {
      int64_t tile_mid_x = tile_out_x * down_x + up_x - 1 - p.pad_x0;
      int64_t tile_mid_y = tile_out_y * down_y + up_y - 1 - p.pad_y0;
      int64_t tile_in_x = floor_div(tile_mid_x, up_x);
      int64_t tile_in_y = floor_div(tile_mid_y, up_y);

      __syncthreads();

      for (int64_t in_idx = threadIdx.x; in_idx < tile_in_h * tile_in_w;
           in_idx += blockDim.x) {
        int64_t rel_in_y = in_idx / tile_in_w;
        int64_t rel_in_x = in_idx - rel_in_y * tile_in_w;
        int64_t in_x = rel_in_x + tile_in_x;
        int64_t in_y = rel_in_y + tile_in_y;

        scalar_t v = 0.0;

        if (in_x >= 0 & in_y >= 0 & in_x < p.in_w & in_y < p.in_h) {
          v = input[((major_idx * p.in_h + in_y) * p.in_w + in_x) *
                        p.minor_dim +
                    minor_idx];
        }

        sx[rel_in_y][rel_in_x] = v;
      }

      __syncthreads();
      for (int64_t out_idx = threadIdx.x; out_idx < tile_out_h * tile_out_w;
           out_idx += blockDim.x) {
        int64_t rel_out_y = out_idx / tile_out_w;
        int64_t rel_out_x = out_idx - rel_out_y * tile_out_w;
        int64_t out_x = rel_out_x + tile_out_x;
        int64_t out_y = rel_out_y + tile_out_y;

        int64_t mid_x = tile_mid_x + rel_out_x * down_x;
        int64_t mid_y = tile_mid_y + rel_out_y * down_y;
        int64_t in_x = floor_div(mid_x, up_x);
        int64_t in_y = floor_div(mid_y, up_y);
        int64_t rel_in_x = in_x - tile_in_x;
        int64_t rel_in_y = in_y - tile_in_y;
        int64_t kernel_x = (in_x + 1) * up_x - mid_x - 1;
        int64_t kernel_y = (in_y + 1) * up_y - mid_y - 1;

        scalar_t v = 0.0;

#pragma unroll
        for (int64_t y = 0; y < kernel_h / up_y; y++)
#pragma unroll
          for (int64_t x = 0; x < kernel_w / up_x; x++)
            v += sx[rel_in_y + y][rel_in_x + x] *
                 sk[kernel_y + y * up_y][kernel_x + x * up_x];

        if (out_x < p.out_w & out_y < p.out_h) {
          out[((major_idx * p.out_h + out_y) * p.out_w + out_x) * p.minor_dim +
              minor_idx] = v;
        }
      }
    }
  }
}

torch::Tensor upfirdn2d_op(const torch::Tensor &input,
                           const torch::Tensor &kernel, int64_t up_x, int64_t up_y,
                           int64_t down_x, int64_t down_y, int64_t pad_x0, int64_t pad_x1,
                           int64_t pad_y0, int64_t pad_y1) {
  int curDevice = -1;
  cudaGetDevice(&curDevice);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);

  UpFirDn2DKernelParams p;

  auto x = input.contiguous();
  auto k = kernel.contiguous();

  p.major_dim = x.size(0);
  p.in_h = x.size(1);
  p.in_w = x.size(2);
  p.minor_dim = x.size(3);
  p.kernel_h = k.size(0);
  p.kernel_w = k.size(1);
  p.up_x = up_x;
  p.up_y = up_y;
  p.down_x = down_x;
  p.down_y = down_y;
  p.pad_x0 = pad_x0;
  p.pad_x1 = pad_x1;
  p.pad_y0 = pad_y0;
  p.pad_y1 = pad_y1;

  p.out_h = (p.in_h * p.up_y + p.pad_y0 + p.pad_y1 - p.kernel_h + p.down_y) /
            p.down_y;
  p.out_w = (p.in_w * p.up_x + p.pad_x0 + p.pad_x1 - p.kernel_w + p.down_x) /
            p.down_x;

  auto out =
      at::empty({p.major_dim, p.out_h, p.out_w, p.minor_dim}, x.options());

  int64_t mode = -1;

  int64_t tile_out_h = -1;
  int64_t tile_out_w = -1;

  if (p.up_x == 1 && p.up_y == 1 && p.down_x == 1 && p.down_y == 1 &&
      p.kernel_h <= 4 && p.kernel_w <= 4) {
    mode = 1;
    tile_out_h = 16;
    tile_out_w = 64;
  }

  if (p.up_x == 1 && p.up_y == 1 && p.down_x == 1 && p.down_y == 1 &&
      p.kernel_h <= 3 && p.kernel_w <= 3) {
    mode = 2;
    tile_out_h = 16;
    tile_out_w = 64;
  }

  if (p.up_x == 2 && p.up_y == 2 && p.down_x == 1 && p.down_y == 1 &&
      p.kernel_h <= 4 && p.kernel_w <= 4) {
    mode = 3;
    tile_out_h = 16;
    tile_out_w = 64;
  }

  if (p.up_x == 2 && p.up_y == 2 && p.down_x == 1 && p.down_y == 1 &&
      p.kernel_h <= 2 && p.kernel_w <= 2) {
    mode = 4;
    tile_out_h = 16;
    tile_out_w = 64;
  }

  if (p.up_x == 1 && p.up_y == 1 && p.down_x == 2 && p.down_y == 2 &&
      p.kernel_h <= 4 && p.kernel_w <= 4) {
    mode = 5;
    tile_out_h = 8;
    tile_out_w = 32;
  }

  if (p.up_x == 1 && p.up_y == 1 && p.down_x == 2 && p.down_y == 2 &&
      p.kernel_h <= 2 && p.kernel_w <= 2) {
    mode = 6;
    tile_out_h = 8;
    tile_out_w = 32;
  }

  dim3 block_size;
  dim3 grid_size;

  if (tile_out_h > 0 && tile_out_w > 0) {
    p.loop_major = (p.major_dim - 1) / 16384 + 1;
    p.loop_x = 1;
    block_size = dim3(32 * 8, 1, 1);
    grid_size = dim3(((p.out_h - 1) / tile_out_h + 1) * p.minor_dim,
                     (p.out_w - 1) / (p.loop_x * tile_out_w) + 1,
                     (p.major_dim - 1) / p.loop_major + 1);
  } else {
    p.loop_major = (p.major_dim - 1) / 16384 + 1;
    p.loop_x = 4;
    block_size = dim3(4, 32, 1);
    grid_size = dim3((p.out_h * p.minor_dim - 1) / block_size.x + 1,
                     (p.out_w - 1) / (p.loop_x * block_size.y) + 1,
                     (p.major_dim - 1) / p.loop_major + 1);
  }

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "upfirdn2d_cuda", [&] {
    switch (mode) {
    case 1:
      upfirdn2d_kernel<scalar_t, 1, 1, 1, 1, 4, 4, 16, 64>
          <<<grid_size, block_size, 0, stream>>>(out.data_ptr<scalar_t>(),
                                                 x.data_ptr<scalar_t>(),
                                                 k.data_ptr<scalar_t>(), p);

      break;

    case 2:
      upfirdn2d_kernel<scalar_t, 1, 1, 1, 1, 3, 3, 16, 64>
          <<<grid_size, block_size, 0, stream>>>(out.data_ptr<scalar_t>(),
                                                 x.data_ptr<scalar_t>(),
                                                 k.data_ptr<scalar_t>(), p);

      break;

    case 3:
      upfirdn2d_kernel<scalar_t, 2, 2, 1, 1, 4, 4, 16, 64>
          <<<grid_size, block_size, 0, stream>>>(out.data_ptr<scalar_t>(),
                                                 x.data_ptr<scalar_t>(),
                                                 k.data_ptr<scalar_t>(), p);

      break;

    case 4:
      upfirdn2d_kernel<scalar_t, 2, 2, 1, 1, 2, 2, 16, 64>
          <<<grid_size, block_size, 0, stream>>>(out.data_ptr<scalar_t>(),
                                                 x.data_ptr<scalar_t>(),
                                                 k.data_ptr<scalar_t>(), p);

      break;

    case 5:
      upfirdn2d_kernel<scalar_t, 1, 1, 2, 2, 4, 4, 8, 32>
          <<<grid_size, block_size, 0, stream>>>(out.data_ptr<scalar_t>(),
                                                 x.data_ptr<scalar_t>(),
                                                 k.data_ptr<scalar_t>(), p);

      break;

    case 6:
      upfirdn2d_kernel<scalar_t, 1, 1, 2, 2, 4, 4, 8, 32>
          <<<grid_size, block_size, 0, stream>>>(out.data_ptr<scalar_t>(),
                                                 x.data_ptr<scalar_t>(),
                                                 k.data_ptr<scalar_t>(), p);

      break;

    default:
      upfirdn2d_kernel_large<scalar_t><<<grid_size, block_size, 0, stream>>>(
          out.data_ptr<scalar_t>(), x.data_ptr<scalar_t>(),
          k.data_ptr<scalar_t>(), p);
    }
  });

  return out;
}
