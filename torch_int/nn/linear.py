import torch
from .._CUDA import (
    linear_a8_w8_b32_o32,
    linear_relu_a8_w8_b8_o8,
    linear_a8_w8_b8_o8,
    linear_a8_w8_b32_o32_with_scaling,
    linear_a8_w8_bfp32_ofp32,
    int8Matmul,
)
from ..functional.quantization import (
    quantize_per_tensor_absmax,
    quantize_weight_per_channel_absmax,
    fake_quantize_activation_per_tensor_absmax,
    fake_quantize_activation_per_token_absmax,
    quantize_per_tensor_absmax_custom_bits,
)
import transformers
import torch.nn.functional as F
import time

LOG = False
QUAN = False
B = 1
SEQ = 8
DIM = 1024
zeros_f1 = torch.zeros((B * SEQ, DIM), device="cpu")
zeros_f13 = torch.zeros((B * SEQ, DIM * 3), device="cpu")
zeros_f2 = torch.zeros((B * SEQ, DIM * 4), device="cpu")

zeros_f1_q = zeros_f1.to(torch.int8)
zeros_f13_q = zeros_f13.to(torch.int8)
zeros_f2_q = zeros_f2.to(torch.int8)

# Hack sparsity
rand_tensor = torch.randint(0, 5, (SEQ, DIM * 4))
ones_tensor = torch.where(rand_tensor > 3, torch.tensor(1), torch.tensor(0))
sparse = ones_tensor.to_sparse_coo()
indices = sparse.indices()
sz = sparse.size()

mask = (1 - ones_tensor).bool()


class W8A8B8O8Linear(torch.nn.Module):
    # For qkv_proj
    def __init__(self, in_features, out_features, alpha=1.0, beta=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer(
            "weight",
            torch.randint(
                -127,
                127,
                (self.out_features, self.in_features),
                dtype=torch.int8,
                requires_grad=False,
            ),
        )
        self.register_buffer(
            "bias",
            torch.zeros((1, self.out_features), dtype=torch.int8, requires_grad=False),
        )
        self.register_buffer("a", torch.tensor(alpha))
        self.register_buffer("b", torch.tensor(beta))

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        x_shape = x.shape
        x = x.view(-1, x_shape[-1])
        y = linear_a8_w8_b8_o8(x, self.weight, self.bias, self.a.item(), self.b.item())
        y = y.view(*x_shape[:-1], -1)
        return y

    @staticmethod
    def from_float(module: torch.nn.Linear, input_scale, output_scale):
        int8_module = W8A8B8O8Linear(module.in_features, module.out_features)
        int8_weight, weight_scale = quantize_per_tensor_absmax(module.weight)
        int8_bias, bias_scale = quantize_per_tensor_absmax(module.bias)
        alpha = input_scale * weight_scale / output_scale
        beta = bias_scale / output_scale
        # in case size mismatch
        int8_module.weight = torch.reshape(int8_weight, int8_module.weight.shape)
        int8_module.bias = torch.reshape(int8_bias, int8_module.bias.shape)
        int8_module.a = alpha
        int8_module.b = beta
        return int8_module


class W8A8B8O8LinearReLU(torch.nn.Module):
    # For fc1
    def __init__(self, in_features, out_features, alpha=1.0, beta=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer(
            "weight",
            torch.randint(
                -127,
                127,
                (self.out_features, self.in_features),
                dtype=torch.int8,
                requires_grad=False,
            ),
        )
        self.register_buffer(
            "bias",
            torch.zeros((1, self.out_features), dtype=torch.int8, requires_grad=False),
        )
        self.register_buffer("a", torch.tensor(alpha))
        self.register_buffer("b", torch.tensor(beta))

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        x_shape = x.shape
        x = x.view(-1, x_shape[-1])
        y = linear_relu_a8_w8_b8_o8(
            x, self.weight, self.bias, self.a.item(), self.b.item()
        )
        y = y.view(*x_shape[:-1], -1)
        return y

    @staticmethod
    def from_float(module: torch.nn.Linear, input_scale, output_scale):
        # TODO: add zero-point to prevent the bit waste
        int8_module = W8A8B8O8LinearReLU(module.in_features, module.out_features)
        int8_weight, weight_scale = quantize_per_tensor_absmax(module.weight)
        int8_bias, bias_scale = quantize_per_tensor_absmax(module.bias)
        alpha = input_scale * weight_scale / output_scale
        beta = bias_scale / output_scale
        # in case size mismatch
        int8_module.weight = torch.reshape(int8_weight, int8_module.weight.shape)
        int8_module.bias = torch.reshape(int8_bias, int8_module.bias.shape)
        int8_module.a = alpha
        int8_module.b = beta
        return int8_module


class W8A8B32O32LinearWithoutScaling(torch.nn.Module):
    # For fc2 and out_proj
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer(
            "weight",
            torch.randint(
                -127,
                127,
                (self.out_features, self.in_features),
                dtype=torch.int8,
                requires_grad=False,
            ),
        )
        self.register_buffer(
            "bias",
            torch.zeros((1, self.out_features), dtype=torch.int32, requires_grad=False),
        )

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        x_shape = x.shape
        x = x.view(-1, x_shape[-1])
        y = linear_a8_w8_b32_o32(x, self.weight, self.bias)
        y = y.view(*x_shape[:-1], -1)
        return y


class W8A8B32O32Linear(torch.nn.Module):
    # For fc2 and out_proj
    def __init__(self, in_features, out_features, alpha=1.0, beta=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer(
            "weight",
            torch.randint(
                -127,
                127,
                (self.out_features, self.in_features),
                dtype=torch.int8,
                requires_grad=False,
            ),
        )
        self.register_buffer(
            "bias",
            torch.zeros((1, self.out_features), dtype=torch.int32, requires_grad=False),
        )
        self.register_buffer("a", torch.tensor(alpha))
        self.register_buffer("b", torch.tensor(beta))

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        x_shape = x.shape
        x = x.view(-1, x_shape[-1])
        y = linear_a8_w8_b32_o32_with_scaling(
            x, self.weight, self.bias, self.a.item(), self.b.item()
        )
        y = y.view(*x_shape[:-1], -1)
        return y

    @staticmethod
    def from_float(module: torch.nn.Linear, input_scale, output_scale):
        int8_module = W8A8B32O32Linear(module.in_features, module.out_features)
        int8_weight, weight_scale = quantize_per_tensor_absmax(module.weight)
        module.bias = module.bias.float()
        bias_scale = module.bias.abs().max() / (2**31 - 1)
        int32_bias = (module.bias / bias_scale).round().to(torch.int32)
        alpha = input_scale * weight_scale / output_scale
        beta = bias_scale / output_scale
        int8_module.weight = int8_weight
        int8_module.bias = int32_bias
        int8_module.a = alpha
        int8_module.b = beta
        int8_module.input_scale = input_scale
        int8_module.output_scale = output_scale
        int8_module.weight_scale = weight_scale
        int8_module.bias_scale = bias_scale
        return int8_module


class W8A8BFP32OFP32Linear(torch.nn.Module):
    # For fc2 and out_proj
    def __init__(self, in_features, out_features, alpha=1.0, beta=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer(
            "weight",
            torch.randint(
                -127,
                127,
                (self.out_features, self.in_features),
                dtype=torch.int8,
                requires_grad=False,
            ),
        )
        self.register_buffer(
            "bias",
            torch.zeros(
                (1, self.out_features), dtype=torch.float32, requires_grad=False
            ),
        )
        self.register_buffer("a", torch.tensor(alpha))

    def _apply(self, fn):
        # prevent the bias from being converted to half
        super()._apply(fn)
        self.bias = self.bias.to(torch.float32)
        return self

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        self.bias = self.bias.to(*args, **kwargs)
        self.bias = self.bias.to(torch.float32)
        return self

    @torch.no_grad()
    def forward(self, x):
        x_shape = x.shape
        x = x.view(-1, x_shape[-1])
        self.bias = self.bias.to(torch.float32)
        y = linear_a8_w8_bfp32_ofp32(x, self.weight, self.bias, self.a.item(), 1)
        y = y.view(*x_shape[:-1], -1)
        return y

    @staticmethod
    def from_float(module: torch.nn.Linear, input_scale):
        int8_module = W8A8BFP32OFP32Linear(module.in_features, module.out_features)
        int8_weight, weight_scale = quantize_per_tensor_absmax(module.weight)
        alpha = input_scale * weight_scale
        # in case size mismatch
        int8_module.weight = torch.reshape(int8_weight, int8_module.weight.shape)
        int8_module.bias = torch.reshape(
            module.bias.to(torch.float32), int8_module.bias.shape
        )
        int8_module.a = alpha
        int8_module.input_scale = input_scale
        int8_module.weight_scale = weight_scale
        return int8_module


class W8A16Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(W8A16Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer(
            "weight",
            torch.randint(
                -127,
                127,
                (self.out_features, self.in_features),
                dtype=torch.int8,
                requires_grad=False,
            ),
        )
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(
                    (1, self.out_features), dtype=torch.float16, requires_grad=False
                ),
            )
        else:
            self.register_buffer("bias", None)
        self.register_buffer(
            "weight_scales",
            torch.ones(self.out_features, dtype=torch.float16, requires_grad=False),
        )

    def to(self, *args, **kwargs):
        super(W8A16Linear, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        self.weight_scales = self.weight_scales.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        weight_fp16 = self.weight.to(torch.float16)
        weight_fp16.mul_(self.weight_scales)
        y = torch.functional.F.linear(x, weight_fp16, self.bias)
        del weight_fp16
        return y

    @staticmethod
    def from_float(module, weight_quant="per_channel"):
        assert isinstance(module, torch.nn.Linear)
        new_module = W8A16Linear(
            module.in_features, module.out_features, module.bias is not None
        )
        if weight_quant == "per_channel":
            (
                new_module.weight,
                new_module.weight_scales,
            ) = quantize_weight_per_channel_absmax(module.weight)
        elif weight_quant == "per_tensor":
            new_module.weight, new_module.weight_scales = quantize_per_tensor_absmax(
                module.weight
            )
        else:
            raise ValueError('weight_quant must be "per_channel" or "per_tensor"')
        if module.bias is not None:
            new_module.bias = module.bias.to(torch.float16)
        return new_module

    def __repr__(self):
        return (
            super().__repr__()
            + f"({self.in_features}, {self.out_features}, bias={self.bias is not None})"
        )


class W8FakeA8Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, act_quant="per_token"):
        super(W8FakeA8Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer(
            "weight",
            torch.randint(
                -127,
                127,
                (self.out_features, self.in_features),
                dtype=torch.int8,
                requires_grad=False,
            ),
        )
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(
                    (1, self.out_features), dtype=torch.float16, requires_grad=False
                ),
            )
        else:
            self.register_buffer("bias", None)
        self.register_buffer(
            "weight_scales",
            torch.ones(self.out_features, dtype=torch.float16, requires_grad=False),
        )

        if act_quant == "per_token":
            self.activation_fake_quantizer = fake_quantize_activation_per_token_absmax
        elif act_quant == "per_tensor":
            self.activation_fake_quantizer = fake_quantize_activation_per_tensor_absmax
        else:
            raise ValueError('act_quant must be "per_token" or "per_tensor"')

    def to(self, *args, **kwargs):
        super(W8FakeA8Linear, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        self.weight_scales = self.weight_scales.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        weight_fp16 = self.weight.to(torch.float16)
        weight_fp16.mul_(self.weight_scales.view(-1, 1))
        x = self.activation_fake_quantizer(x)
        y = torch.functional.F.linear(x, weight_fp16, self.bias)
        del weight_fp16
        return y

    @staticmethod
    def from_float(module, act_quant="per_token"):
        assert isinstance(module, torch.nn.Linear)
        new_module = W8FakeA8Linear(
            module.in_features, module.out_features, module.bias is not None, act_quant
        )
        (
            new_module.weight,
            new_module.weight_scales,
        ) = quantize_weight_per_channel_absmax(module.weight)
        if module.bias is not None:
            new_module.bias = module.bias.to(torch.float16)
        return new_module

    def __repr__(self):
        return (
            super().__repr__()
            + f"({self.in_features}, {self.out_features}, bias={self.bias is not None})"
        )


class W8A8BFP32O32LinearWithoutScaling(torch.nn.Module):
    # For fc2 and out_proj
    def __init__(self, in_features, out_features, alpha=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer(
            "weight",
            torch.randint(
                -127,
                127,
                (self.out_features, self.in_features),
                dtype=torch.int8,
                requires_grad=False,
            ),
        )
        self.register_buffer(
            "bias",
            torch.zeros(
                (1, self.out_features), dtype=torch.float32, requires_grad=False
            ),
        )
        self.register_buffer(
            "placeholder_bias",
            torch.zeros((1, self.out_features), dtype=torch.int32, requires_grad=False),
        )
        self.register_buffer("a", torch.tensor(alpha))

    def _apply(self, fn):
        # prevent the bias from being converted to half
        super()._apply(fn)
        self.bias = self.bias.to(torch.float32)
        return self

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        self.bias = self.bias.to(*args, **kwargs)
        self.bias = self.bias.to(torch.float32)
        return self

    @torch.no_grad()
    def forward(self, x):
        x_shape = x.shape
        x = x.view(-1, x_shape[-1])
        # (int32)y = x * self.weight
        y = linear_a8_w8_b32_o32(x, self.weight, self.placeholder_bias)
        # (int32)y = y + bias
        y = y + self.bias.div(self.a.item()).round()
        y = y.view(*x_shape[:-1], -1)
        return y

    @staticmethod
    def from_float(module: torch.nn.Linear, input_scale):
        int8_module = W8A8BFP32O32LinearWithoutScaling(
            module.in_features, module.out_features, alpha=1.0
        )
        int8_weight, weight_scale = quantize_per_tensor_absmax(module.weight)
        alpha = input_scale * weight_scale
        int8_module.weight = int8_weight
        # quantize bias. not invoked
        # _bias = module.bias
        # _bias.div_(alpha).round_()
        # _bias = _bias.to(torch.float32)
        # int8_module.bias = _bias
        int8_module.bias = module.bias.to(torch.float32)
        int8_module.a = alpha
        int8_module.input_scale = input_scale
        int8_module.weight_scale = weight_scale
        return int8_module


class W8A8B8O32LinearWithoutScaling(torch.nn.Module):
    # For fc1
    def __init__(self, in_features, out_features, alpha=1.0, beta=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer(
            "weight",
            torch.randint(
                -127,
                127,
                (self.out_features, self.in_features),
                dtype=torch.int8,
                requires_grad=False,
            ),
        )
        self.register_buffer(
            "bias",
            torch.zeros((1, self.out_features), dtype=torch.int8, requires_grad=False),
        )
        self.register_buffer(
            "placeholder_bias",
            torch.zeros((1, self.out_features), dtype=torch.int32, requires_grad=False),
        )
        self.register_buffer("a", torch.tensor(alpha))
        self.register_buffer("b", torch.tensor(beta))

    # def _apply(self, fn):
    #     # prevent the bias from being converted to half
    #     super()._apply(fn)
    #     self.bias = self.bias.to(torch.float32)
    #     return self

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        x_shape = x.shape
        x = x.view(-1, x_shape[-1])
        # (int32)y = x * self.weight
        y = linear_a8_w8_b32_o32(x, self.weight, self.placeholder_bias)
        y = y.view(*x_shape[:-1], -1)
        return y

    @staticmethod
    def from_float(module: torch.nn.Linear, input_scale, output_scale):
        # TODO: add zero-point to prevent the bit waste
        int8_module = W8A8B8O32LinearWithoutScaling(
            module.in_features, module.out_features
        )
        int8_weight, weight_scale = quantize_per_tensor_absmax(module.weight)
        int8_bias, bias_scale = quantize_per_tensor_absmax(module.bias)
        alpha = input_scale * weight_scale / output_scale
        beta = bias_scale / output_scale
        int8_module.weight = int8_weight
        int8_module.bias = int8_bias
        int8_module.a = alpha
        int8_module.b = beta
        return int8_module


"""
Linear related interfaces
"""


class WqAqBFP16FP16Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, alpha=1.0, beta=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer(
            "weight",
            torch.randint(
                -127,
                127,
                (self.out_features, self.in_features),
                dtype=torch.int8,
                requires_grad=False,
            ),
        )
        self.register_buffer(
            "bias",
            torch.zeros(
                (1, self.out_features), dtype=torch.float16, requires_grad=False
            ),
        )
        self.register_buffer("a", torch.tensor(alpha))

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x, is_hybrid=True):
        x_shape = x.shape
        x = x.view(-1, x_shape[-1])
        if is_hybrid:
            # mask
            x = x.cpu()
            t1 = time.time()
            # rand_elements = torch.randint(-128, 127, x.shape, device="cpu").to(x.dtype)
            rand_elements = torch.zeros(x.shape, device="cpu").to(x.dtype)
            t2 = time.time()
            # mask_mul = F.linear(rand_elements.float(), self.weight.cpu().float())
            mask_mul = torch.zeros(x.shape[0], self.out_features)

            if LOG:
                print(
                    f"Offline r time: {(t2 - t1) * 1000} ms, Offline f(r) time: {(t2 - t1) * 1000} ms"
                )

            # mask
            t3 = time.time()
            masked_x = torch.add(
                x,
                rand_elements,
            )
            if QUAN:
                # compute wrap
                wraps = ((masked_x ^ x) & (masked_x ^ rand_elements) & 0x80) >> 7
                wraps[(wraps < 0) & (masked_x < 0)] = 1
                wraps = wraps.to(torch.int8)
                wraps = wraps.cuda()  # check security
            if LOG:
                print(f"\033[1;31;40mMask x time: {(time.time() - t3) * 1000} ms\033[m")

            # CPU->GPU IO
            t4 = time.time()
            masked_x = masked_x.cuda()
            if LOG:
                print(
                    f"\033[1;31;40mCPU->GPU IO time: {(time.time() - t4) * 1000} ms\033[m"
                )

            # GPU-side linear computation
            t5 = time.time()
            hidden_states = F.linear(masked_x.float(), self.weight.float())
            if LOG:
                print(
                    f"\033[1;31;40mOnline W*x'+b time: {(time.time() - t5) * 1000} ms\033[m"
                )

            # unmask

            t6 = time.time()
            if QUAN:
                # 1. plus w*W*M on GPU
                overflow_part = (
                    F.linear(wraps.float(), self.weight.float()).to(torch.int32) << 8
                )
                hidden_states = hidden_states + overflow_part
            # 2. minus wr on CPU
            t7 = time.time()
            y = hidden_states.cpu() - mask_mul
            if self.bias is not None:
                y = y * self.a.item() + self.bias.cpu()
            else:
                y = y * self.a.item()
            if LOG:
                print(
                    f"\033[1;31;40mOnline unmask time: {(time.time() - t7) * 1000} ms\033[m"
                )
        else:
            # x = x.cuda()  # x should be on GPU device
            y = F.linear(x.float(), self.weight.float()) * self.a.item() + (
                self.bias if self.bias else 0
            )
        y = y.view(*x_shape[:-1], -1)
        return y

    @staticmethod
    def from_float(module: torch.nn.Linear, input_scale):
        int8_module = W8A8BFP16FP16Linear(module.in_features, module.out_features)
        int8_weight, weight_scale = quantize_per_tensor_absmax(module.weight)
        alpha = input_scale * weight_scale
        # in case size mismatch
        int8_module.weight = torch.reshape(int8_weight, int8_module.weight.shape)
        try:  # in case some linear layer does not have bias
            int8_module.bias = torch.reshape(module.bias, int8_module.bias.shape)
        except:
            int8_module.bias = None
        int8_module.a = alpha
        return int8_module


# For QKV, PROJ and FC2.
# no de-quantization is performed here
class W8A8BFP16FP16Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, alpha=1.0, beta=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer(
            "weight",
            torch.randint(
                -127,
                127,
                (self.out_features, self.in_features),
                dtype=torch.int8,
                requires_grad=False,
            ),
        )
        self.register_buffer(
            "bias",
            torch.zeros(
                (1, self.out_features), dtype=torch.float16, requires_grad=False
            ),
        )
        self.register_buffer("a", torch.tensor(alpha))

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x, is_hybrid=False):
        """
        is_hybrid: whether use OTP and GPU to accelerate some part of the linear computations
        """
        x_shape = x.shape
        x = x.view(-1, x_shape[-1])
        if is_hybrid:
            # mask
            x = x.cpu()
            print(f"x dtype: {x.dtype}")
            t1 = time.time()
            # rand_elements = torch.randint(-128, 127, x.shape, device="cpu").to(x.dtype)
            rand_elements = torch.zeros(x.shape, device="cpu").to(x.dtype)
            t2 = time.time()
            # mask_mul = F.linear(rand_elements.float(), self.weight.cpu().float())
            mask_mul = torch.zeros(x.shape[0], self.out_features)

            if LOG:
                print(
                    f"Offline r time: {(t2 - t1) * 1000} ms, Offline f(r) time: {(t2 - t1) * 1000} ms"
                )

            # mask
            t3 = time.time()
            masked_x = torch.add(
                x,
                rand_elements,
            )
            if QUAN:
                # compute wrap
                wraps = ((masked_x ^ x) & (masked_x ^ rand_elements) & 0x80) >> 7
                wraps[(wraps < 0) & (masked_x < 0)] = 1
                wraps = wraps.to(torch.int8)
                wraps = wraps.cuda()  # check security
            if LOG:
                print(f"\033[1;31;40mMask x time: {(time.time() - t3) * 1000} ms\033[m")

            # CPU->GPU IO
            t4 = time.time()
            masked_x = masked_x.cuda()
            if LOG:
                print(
                    f"\033[1;31;40mCPU->GPU IO time: {(time.time() - t4) * 1000} ms\033[m"
                )

            # GPU-side linear computation
            t5 = time.time()
            hidden_states = F.linear(masked_x.float(), self.weight.float())
            if LOG:
                print(
                    f"\033[1;31;40mOnline W*x'+b time: {(time.time() - t5) * 1000} ms\033[m"
                )

            # unmask

            t6 = time.time()
            if QUAN:
                # 1. plus w*W*M on GPU
                overflow_part = (
                    F.linear(wraps.float(), self.weight.float()).to(torch.int32) << 8
                )
                hidden_states = hidden_states + overflow_part
            # 2. minus wr on CPU
            t7 = time.time()
            y = hidden_states.cpu() - mask_mul
            if self.bias is not None:
                y = y * self.a.item() + self.bias.cpu()
            else:
                y = y * self.a.item()
            if LOG:
                print(
                    f"\033[1;31;40mOnline unmask time: {(time.time() - t7) * 1000} ms\033[m"
                )
        else:
            # x = x.cuda()  # x should be on GPU device
            y = F.linear(x.float(), self.weight.float()) * self.a.item() + (
                self.bias if self.bias else 0
            )
        y = y.view(*x_shape[:-1], -1)
        return y

    @staticmethod
    def from_float(module: torch.nn.Linear, input_scale):
        int8_module = W8A8BFP16FP16Linear(module.in_features, module.out_features)
        int8_weight, weight_scale = quantize_per_tensor_absmax(module.weight)
        alpha = input_scale * weight_scale
        # in case size mismatch
        int8_module.weight = torch.reshape(int8_weight, int8_module.weight.shape)
        try:  # in case some linear layer does not have bias
            int8_module.bias = torch.reshape(module.bias, int8_module.bias.shape)
        except:
            int8_module.bias = None
        int8_module.a = alpha
        return int8_module


# For FC1. The input to FC2 should be int8
# requires runtime quantization, but does not require compute the max
class W8A8B8O32Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, alpha=1.0, beta=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer(
            "weight",
            torch.randint(
                -127,
                127,
                (self.out_features, self.in_features),
                dtype=torch.int8,
                requires_grad=False,
            ),
        )
        self.register_buffer(
            "bias",
            torch.zeros(
                (1, self.out_features), dtype=torch.float16, requires_grad=False
            ),
        )
        self.register_buffer("a", torch.tensor(alpha))
        # self.register_buffer("b", torch.tensor(beta))

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x, is_hybrid=True):
        x_shape = x.shape
        x = x.view(-1, x_shape[-1])
        # (int32)y = x * self.weight
        if is_hybrid:
            # mask
            x = x.cpu()
            t1 = time.time()
            # rand_elements = torch.randint(-128, 127, x.shape, device="cpu").to(x.dtype)
            rand_elements = torch.zeros(x.shape, device="cpu").to(x.dtype)
            t2 = time.time()
            # mask_mul = F.linear(rand_elements.float(), self.weight.cpu().float())
            mask_mul = torch.zeros(x.shape[0], self.out_features)

            print(
                f"Offline r time: {(t2 - t1) * 1000} ms, Offline f(r) time: {(t2 - t1) * 1000} ms"
            )

            # mask
            t3 = time.time()
            masked_x = torch.add(
                x,
                rand_elements,
            )
            # compute wrap
            wraps = ((masked_x ^ x) & (masked_x ^ rand_elements) & 0x80) >> 7
            wraps[(wraps < 0) & (masked_x < 0)] = 1
            wraps = wraps.to(torch.int8)
            wraps = wraps.cuda()  # check security
            print(f"\033[1;31;40mMask x time: {(time.time() - t3) * 1000} ms\033[m")

            # CPU->GPU IO
            t4 = time.time()
            masked_x = masked_x.cuda()
            print(
                f"\033[1;31;40mCPU->GPU IO time: {(time.time() - t4) * 1000} ms\033[m"
            )

            # GPU-side linear computation
            t5 = time.time()
            hidden_states = F.linear(masked_x.float(), self.weight.float())
            print(
                f"\033[1;31;40mOnline W*x'+b time: {(time.time() - t5) * 1000} ms\033[m"
            )

            # unmask
            t6 = time.time()
            # 1. plus w*W*M on GPU
            overflow_part = (
                F.linear(wraps.float(), self.weight.float()).to(torch.int32) << 8
            )
            hidden_states = hidden_states + overflow_part
            # 2. minus wr on CPU
            y = hidden_states.cpu() - mask_mul
            fp_y = y * self.a.item() + self.bias.cpu()
            print(
                f"\033[1;31;40mOnline unmask time: {(time.time() - t6) * 1000} ms\033[m"
            )
        else:
            y = F.linear(x.float(), self.weight.float())
            fp_y = y * self.a.item() + self.bias
        y = fp_y.round().clamp(-128, 127).to(torch.int8)
        # y = linear_a8_w8_b8_o8(x, self.weight, self.bias, self.a.item(), self.b.item())
        y = y.view(*x_shape[:-1], -1)
        return y

    @staticmethod
    def from_float(module: torch.nn.Linear, input_scale, output_scale):
        int8_module = W8A8B8O32Linear(module.in_features, module.out_features)
        int8_weight, weight_scale = quantize_per_tensor_absmax(module.weight)
        # int8_bias, bias_scale = quantize_per_tensor_absmax(module.bias)
        fp_bias = module.bias / output_scale
        alpha = input_scale * weight_scale / output_scale
        # beta = bias_scale / output_scale
        # in case size mismatch
        int8_module.weight = torch.reshape(int8_weight, int8_module.weight.shape)
        int8_module.bias = torch.reshape(fp_bias, int8_module.bias.shape)
        int8_module.a = alpha
        # int8_module.b = beta
        return int8_module


"""
Conv1D related interfaces
This is for GPT2 only
"""


# For QKV, PROJ and FC2.
# no de-quantization is performed here
class W8A8BFP16FP16Conv1D(torch.nn.Module):
    def __init__(self, out_features, in_features, alpha=1.0, beta=1.0):
        super().__init__()
        self.out_features = out_features
        self.in_features = in_features

        self.register_buffer(
            "weight",
            torch.randint(
                -127,
                127,
                (self.in_features, self.out_features),
                dtype=torch.int8,
                requires_grad=False,
            ),
        )
        self.register_buffer(
            "bias",
            torch.zeros(
                (1, self.out_features), dtype=torch.float16, requires_grad=False
            ),
        )
        self.register_buffer("a", torch.tensor(alpha))
        # self.register_buffer("b", torch.tensor(beta))

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x, is_hybrid=False, has_offline=False, hack_sparsity=False):
        x_shape = x.shape
        x = x.view(-1, x_shape[-1])
        t_weight = self.weight.t()
        if is_hybrid:
            if LOG:
                print("=== QKV, PRROJ, FC2 ===")
            # mask
            x = x.cpu()
            if has_offline:
                t1 = time.time()
                # rand_elements = torch.randint(-128, 127, x.shape, device="cpu").to(x.dtype)
                rand_elements = torch.zeros(x.shape, device="cpu").to(x.dtype)

                t2 = time.time()
                # mask_mul = F.linear(rand_elements.float(), t_weight.cpu().float())
                mask_mul = torch.zeros(x.shape[0], self.out_features)
                if LOG:
                    print(
                        f"Offline r time: {(t2 - t1) * 1000} ms, Offline f(r) time: {(t2 - t1) * 1000} ms"
                    )
            else:
                if self.in_features in [768, 1024]:
                    rand_elements = zeros_f1_q
                else:
                    rand_elements = zeros_f2_q
                if self.out_features in [768, 1024]:
                    mask_mul = zeros_f1
                elif self.out_features in [768 * 3, 1024 * 3]:
                    mask_mul = zeros_f13
                else:
                    mask_mul = zeros_f2
                    if hack_sparsity:
                        mask_mul = mask_mul.view(*x_shape[:-1], -1)
                        mask_mul = mask_mul[:, indices[0], indices[1]]

            # mask
            t3 = time.time()
            masked_x = x + rand_elements
            # compute wrap
            if QUAN:
                t_q1 = time.time()
                wraps = ((masked_x ^ x) & (masked_x ^ rand_elements) & 0x80) >> 7
                wraps[(wraps < 0) & (masked_x < 0)] = 1
                wraps = wraps.to(torch.int8)
                print(f"Compute wrap time {(time.time() - t_q1) * 1000} ms")
            if LOG:
                print(f"\033[1;31;40mMask x time: {(time.time() - t3) * 1000} ms\033[m")

            # CPU->GPU IO
            t4 = time.time()
            if QUAN:
                t_q2 = time.time()
                wraps = wraps.cuda()  # check security
                print(f"CPU->GPU wrap time {(time.time() - t_q2) * 1000} ms")

            masked_x = masked_x.cuda()  # FIXME: remove float()
            if LOG:
                print(
                    f"\033[1;31;40mCPU->GPU IO time: {(time.time() - t4) * 1000} ms\033[m"
                )

            # GPU-side linear computation
            t5 = time.time()
            hidden_states = F.linear(
                masked_x.float(), t_weight.float(), self.bias.float()
            ).to(torch.int32)

            if LOG:
                print(
                    f"\033[1;31;40mOnline W*x'+b time: {(time.time() - t5) * 1000} ms\033[m"
                )
            if hack_sparsity and self.out_features == DIM * 4:
                hs = hidden_states.view(*x_shape[:-1], -1)
                # n_samples = hs.shape[0]

                # for i in range(n_samples):
                #     hs[i][mask] = 0
                # hs = hs.view(-1, hs.shape[-1])
                # sparse_hs = hs.to_sparse_csr()
                # values = sparse_hs.values()
                values = hs[:, indices[0], indices[1]]
            # unmask
            t6 = time.time()
            # 1. plus w*W*M on GPU
            if QUAN:
                t_q3 = time.time()
                overflow_part = (
                    F.linear(wraps.float(), t_weight.float()).to(torch.int32) << 8
                )
                hidden_states = hidden_states + overflow_part
                print(f"GPU side wrap offset time {(time.time() - t_q3) * 1000} ms")

            # 2. minus wr on CPU
            t7 = time.time()
            if hack_sparsity and self.out_features == DIM * 4:
                values = values.cpu()
                values = values - mask_mul
                # TODO: fuse activation here
                values = F.gelu(values, approximate="tanh")
                hidden_states = torch.zeros(
                    *x_shape[:-1], self.out_features, dtype=values.dtype
                )
                hidden_states[:, indices[0], indices[1]] = values
                # hidden_states = torch.sparse_coo_tensor(
                #     indices,
                #     values,
                #     sz,
                #     dtype=values.dtype,
                #     device=values.device,
                # ).to_dense()
                hidden_states = hidden_states.view(*x_shape[:-1], -1)
            else:
                hidden_states = hidden_states.cpu()

            if LOG:
                print(
                    f"\033[1;31;40mGPU->CPU IO time: {(time.time() - t7) * 1000} ms\033[m"
                )

            if hack_sparsity:
                y = hidden_states
            else:
                t8 = time.time()
                y = hidden_states - mask_mul
                if LOG:
                    print(
                        f"\033[1;31;40mCPU minus f(r): {(time.time() - t8) * 1000} ms\033[m"
                    )

            t9 = time.time()
            y = y * self.a.item()
            if LOG:
                print(
                    f"\033[1;31;40mCPU multiply scale: {(time.time() - t9) * 1000} ms\033[m"
                )
                print(
                    f"\033[1;31;40mOnline unmask time: {(time.time() - t7) * 1000} ms\033[m"
                )
                print(f"Hybrid linear time: {(time.time() - t3) * 1000} ms")
        else:
            # y = int8Matmul(x, t_weight) * self.a.item() + self.bias
            tt = time.time()
            y = F.linear(x.float(), t_weight.float(), self.bias.float()) * self.a.item()
            print(f"CPU linear {(time.time() - tt) * 1000} ms")
        y = y.view(*x_shape[:-1], -1)
        return y

    @staticmethod
    def from_float(
        module: transformers.pytorch_utils.Conv1D,
        out_features,
        in_features,
        input_scale,
    ):
        int8_module = W8A8BFP16FP16Conv1D(out_features, in_features)
        int8_weight, weight_scale = quantize_per_tensor_absmax(module.weight)
        alpha = input_scale * weight_scale
        int8_bias = (module.bias / alpha).to(torch.int32)
        # beta = bias_scale / output_scale
        # in case size mismatch
        int8_module.weight = torch.reshape(int8_weight, int8_module.weight.shape)
        int8_module.bias = torch.reshape(int8_bias, int8_module.bias.shape)
        int8_module.a = alpha
        # int8_module.b = beta
        return int8_module


# For FC1. The input to FC2 should be int8
# requires runtime quantization, but does not require compute the max
class W8A8B8O32Conv1D(torch.nn.Module):
    def __init__(self, out_features, in_features, alpha=1.0, beta=1.0):
        super().__init__()
        self.out_features = out_features
        self.in_features = in_features

        self.register_buffer(
            "weight",
            torch.randint(
                -127,
                127,
                (self.in_features, self.out_features),
                dtype=torch.int8,
                requires_grad=False,
            ),
        )
        self.register_buffer(
            "bias",
            torch.zeros(
                (1, self.out_features), dtype=torch.float16, requires_grad=False
            ),
        )
        self.register_buffer("a", torch.tensor(alpha))
        # self.register_buffer("b", torch.tensor(beta))

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x, is_hybrid=True):
        x_shape = x.shape
        x = x.view(-1, x_shape[-1])
        # (int32)y = x * self.weight
        t_weight = self.weight.t()
        if is_hybrid:
            print("=== FC1 ===")
            # mask
            x = x.cpu()
            t1 = time.time()
            # rand_elements = torch.randint(-128, 127, x.shape, device="cpu").to(x.dtype)
            rand_elements = torch.zeros(x.shape, device="cpu").to(x.dtype)
            t2 = time.time()
            # mask_mul = F.linear(rand_elements.float(), t_weight.cpu().float())
            mask_mul = torch.zeros(x.shape[0], self.out_features)

            print(
                f"Offline r time: {(t2 - t1) * 1000} ms, Offline f(r) time: {(t2 - t1) * 1000} ms"
            )

            # mask
            t3 = time.time()
            masked_x = torch.add(
                x,
                rand_elements,
            )
            # compute wrap
            wraps = ((masked_x ^ x) & (masked_x ^ rand_elements) & 0x80) >> 7
            wraps[(wraps < 0) & (masked_x < 0)] = 1
            print(wraps)
            wraps = wraps.to(torch.int8)
            wraps = wraps.cuda()  # check security
            print(f"\033[1;31;40mMask x time: {(time.time() - t3) * 1000} ms\033[m")

            # CPU->GPU IO
            t4 = time.time()
            masked_x = masked_x.cuda()
            print(
                f"\033[1;31;40mCPU->GPU IO time: {(time.time() - t4) * 1000} ms\033[m"
            )

            # GPU-side linear computation
            t5 = time.time()
            hidden_states = F.linear(masked_x.float(), t_weight.float())
            print(
                f"\033[1;31;40mOnline W*x'+b time: {(time.time() - t5) * 1000} ms\033[m"
            )

            # unmask
            t6 = time.time()
            # 1. plus w*W*M on GPU
            overflow_part = (
                F.linear(wraps.float(), t_weight.float()).to(torch.int32) << 8
            )
            hidden_states = hidden_states + overflow_part
            # 2. minus wr on CPU
            y = hidden_states.cpu() - mask_mul
            fp_y = y * self.a.item() + self.bias.cpu()
            print(
                f"\033[1;31;40mOnline unmask time: {(time.time() - t6) * 1000} ms\033[m"
            )
        else:
            y = int8Matmul(x, t_weight)
            fp_y = y * self.a.item() + self.bias
        y = fp_y.round().clamp(-128, 127).to(torch.int8)
        # y = linear_a8_w8_b8_o8(x, self.weight, self.bias, self.a.item(), self.b.item())
        y = y.view(*x_shape[:-1], -1)
        return y

    @staticmethod
    def from_float(
        module: transformers.pytorch_utils.Conv1D, input_scale, output_scale
    ):
        int8_module = W8A8B8O32Conv1D(module.in_features, module.out_features)
        int8_weight, weight_scale = quantize_per_tensor_absmax(module.weight.t())
        # int8_bias, bias_scale = quantize_per_tensor_absmax(module.bias)
        fp_bias = module.bias / output_scale
        alpha = input_scale * weight_scale / output_scale
        # beta = bias_scale / output_scale
        # in case size mismatch
        int8_module.weight = torch.reshape(int8_weight.t(), int8_module.weight.shape)
        int8_module.bias = torch.reshape(fp_bias, int8_module.bias.shape)
        int8_module.a = alpha
        # int8_module.b = beta
        return int8_module
