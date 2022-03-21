#include <torch/extension.h>
#include<torch/library.h>

torch::Tensor fused_bias_act_op(const torch::Tensor& input, const torch::Tensor& bias, const torch::Tensor& refer,
    int64_t act, int64_t grad, double alpha, double scale);

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor fused_bias_act(const torch::Tensor& input, const torch::Tensor& bias, const torch::Tensor& refer,
    int64_t act, int64_t grad, double alpha, double scale) {
    CHECK_CUDA(input);
    CHECK_CUDA(bias);

    return fused_bias_act_op(input, bias, refer, act, grad, alpha, scale);
}

TORCH_LIBRARY(fusebias, m) {
    m.def("fused_bias_act", fused_bias_act);
}
