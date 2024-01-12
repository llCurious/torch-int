import torch
from ..functional.fused import dq_add_layernorm_q_cpp


class RMSNormQ(torch.nn.Module):
    def __init__(self, dim, eps=1e-6, bits=8):
        super().__init__()
        self.input_scale = 1.0
        self.eps = eps
        self.bits = bits
        if self.bits == 8:
            self.q_type = torch.int8
        elif self.bits == 16:
            self.q_type = torch.int16
        else:
            raise Exception(f"Quan bits {self.bits} is not supported!")

        self.register_buffer("weight", torch.ones(dim, dtype=torch.float32))
        # self.register_buffer("bias", torch.zeros(dim, dtype=torch.float32))

    def forward(self, x):
        x = x.to(self.weight.dtype)

        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)

        ln_output_fp = self.weight * x
        ln_output_int = (
            ln_output_fp.round()
            .clamp(-(2 ** (self.bits - 1)), 2 ** (self.bits - 1) - 1)
            .to(self.q_type)
        )
        return ln_output_int

    @staticmethod
    def from_float(module: torch.nn.LayerNorm, output_scale: float, q_bits: int = 8):
        assert module.normalized_shape[0] == module.weight.numel()
        q_module = RMSNormQ(module.normalized_shape[0], module.eps, q_bits)
        q_module.weight = module.weight / output_scale
        return q_module


class LayerNormQ(torch.nn.Module):
    def __init__(self, dim, eps=1e-5, bits=8):
        super().__init__()
        self.input_scale = 1.0
        self.eps = eps
        self.bits = bits
        if self.bits == 8:
            self.q_type = torch.int8
        elif self.bits == 16:
            self.q_type = torch.int16
        else:
            raise Exception(f"Quan bits {self.bits} is not supported!")

        self.register_buffer("weight", torch.ones(dim, dtype=torch.float32))
        self.register_buffer("bias", torch.zeros(dim, dtype=torch.float32))

    def forward(self, x):
        x = x.to(self.weight.dtype)
        ln_output_fp = torch.nn.functional.layer_norm(
            x, x.shape[-1:], self.weight, self.bias, self.eps
        )
        ln_output_int = (
            ln_output_fp.round()
            .clamp(-(2 ** (self.bits - 1)), 2 ** (self.bits - 1) - 1)
            .to(self.q_type)
        )
        return ln_output_int

    @staticmethod
    def from_float(module: torch.nn.LayerNorm, output_scale: float, q_bits: int = 8):
        assert module.normalized_shape[0] == module.weight.numel()
        try:  # some customized layernorm does not have bias parameter
            assert module.normalized_shape[0] == module.bias.numel()
        except:
            pass
        q_module = LayerNormQ(module.normalized_shape[0], module.eps, q_bits)
        q_module.weight = module.weight / output_scale
        try:  # some customized layernorm does not have bias parameter
            q_module.bias = module.bias / output_scale
        except:
            print(f"LayerNormQ use bias=0 by default.")
        return q_module


class DQ_Add_LayerNorm_Q(torch.nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.input_scale = 1.0
        self.eps = eps
        self.register_buffer("weight", torch.ones(dim, dtype=torch.float32))
        self.register_buffer("bias", torch.zeros(dim, dtype=torch.float32))

    def forward(self, residual_input_fp, input_int32):
        # input_int32: [B, L, H] int32
        # residual_input_fp: [B, L, H] fp
        # return residual_output_fp, ln_output_int8
        return dq_add_layernorm_q_cpp(
            input_int32,
            self.input_scale,
            residual_input_fp,
            self.weight,
            self.bias,
            self.eps,
        )
