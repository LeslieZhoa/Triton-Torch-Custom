#include <torch/extension.h>
#include<torch/library.h>

torch::Tensor upfirdn2d_op(const torch::Tensor& input, const torch::Tensor& kernel,
                            int64_t up_x, int64_t up_y, int64_t down_x, int64_t down_y,
                            int64_t pad_x0, int64_t pad_x1, int64_t pad_y0, int64_t pad_y1);

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor upfirdn2d(const torch::Tensor& input, const torch::Tensor& kernel,
                        int64_t up_x, int64_t up_y, int64_t down_x, int64_t down_y,
                        int64_t pad_x0, int64_t pad_x1, int64_t pad_y0, int64_t pad_y1) {
    CHECK_CUDA(input);
    CHECK_CUDA(kernel);

    return upfirdn2d_op(input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1);
}

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//     m.def("upfirdn2d", &upfirdn2d, "upfirdn2d (CUDA)");
// }

TORCH_LIBRARY(upfirdn, m) {
    m.def("upfirdn2d", upfirdn2d);
}
