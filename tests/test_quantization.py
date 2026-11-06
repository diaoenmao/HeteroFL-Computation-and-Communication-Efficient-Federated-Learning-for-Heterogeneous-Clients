import os
import sys
import unittest
from collections import OrderedDict

import torch

# Ensure src/ is in sys.path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from quantization import (
    calc_compression_stats,
    dequantize_sliced_parameters,
    dequantize_tensor,
    quantize_sliced_parameters,
    quantize_tensor,
    simulate_quantize_parameters,
)


class TestHeteroFLQuantization(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_quantize_dequantize_int4_roundtrip(self):
        for shape in [(64,), (32, 32), (16, 3, 3, 3), (7, 13, 11)]:
            tensor = torch.randn(shape, dtype=torch.float32)
            q_dict = quantize_tensor(tensor, nbits=4, block_size=32)
            recon = dequantize_tensor(q_dict)

            self.assertEqual(recon.shape, tensor.shape)
            self.assertEqual(recon.dtype, tensor.dtype)
            cos_sim = torch.nn.functional.cosine_similarity(
                tensor.flatten(), recon.flatten(), dim=0
            )
            self.assertGreater(cos_sim.item(), 0.98)

    def test_quantize_dequantize_int8_roundtrip(self):
        tensor = torch.randn(64, 64, dtype=torch.float32)
        q_dict = quantize_tensor(tensor, nbits=8, block_size=64)
        recon = dequantize_tensor(q_dict)

        self.assertEqual(recon.shape, tensor.shape)
        cos_sim = torch.nn.functional.cosine_similarity(
            tensor.flatten(), recon.flatten(), dim=0
        )
        self.assertGreater(cos_sim.item(), 0.999)

    def test_zero_tensor_handling(self):
        tensor = torch.zeros(128, dtype=torch.float32)
        q_dict = quantize_tensor(tensor, nbits=4, block_size=32)
        recon = dequantize_tensor(q_dict)

        self.assertTrue(torch.equal(tensor, recon))
        self.assertFalse(torch.isnan(recon).any())
        self.assertFalse(torch.isinf(recon).any())

    def test_empty_tensor(self):
        tensor = torch.empty((0, 10), dtype=torch.float32)
        q_dict = quantize_tensor(tensor, nbits=4, block_size=32)
        recon = dequantize_tensor(q_dict)

        self.assertEqual(recon.shape, (0, 10))
        self.assertEqual(recon.numel(), 0)

    def test_sub_byte_wire_footprint(self):
        numel = 256
        tensor = torch.randn(numel, dtype=torch.float32)
        q_dict = quantize_tensor(tensor, nbits=4, block_size=64)

        # 256 4-bit nibbles should pack into exactly 128 uint8 bytes
        self.assertEqual(q_dict["data"].dtype, torch.uint8)
        self.assertEqual(q_dict["data"].numel(), numel // 2)

    def test_quantize_sliced_parameters(self):
        params = OrderedDict(
            {
                "conv1.weight": torch.randn(16, 8, 3, 3),
                "conv1.bias": torch.randn(16),
                "running_mean": torch.zeros(16),
                "step": torch.tensor(10, dtype=torch.long),
            }
        )

        q_params = quantize_sliced_parameters(params, nbits=4, block_size=32)
        restored = dequantize_sliced_parameters(q_params)

        for name in params:
            self.assertEqual(restored[name].shape, params[name].shape)
            if torch.is_floating_point(params[name]):
                if params[name].abs().sum() > 0:
                    cos = torch.nn.functional.cosine_similarity(
                        params[name].flatten(), restored[name].flatten(), dim=0
                    )
                    self.assertGreater(cos.item(), 0.98)
            else:
                self.assertTrue(torch.equal(params[name], restored[name]))

    def test_error_feedback_residual_accumulation(self):
        residuals = {}
        true_accum = torch.zeros(64)
        quant_accum = torch.zeros(64)

        for _ in range(10):
            step_delta = torch.randn(64) * 0.05
            true_accum += step_delta

            param_dict = OrderedDict({"weight": step_delta})
            q_dict = quantize_sliced_parameters(
                param_dict, nbits=4, block_size=32, residuals=residuals
            )
            recon_dict = dequantize_sliced_parameters(q_dict)
            quant_accum += recon_dict["weight"]

        drift = torch.norm(true_accum - quant_accum).item()
        # Residual tracking keeps drift bounded
        self.assertLess(drift, 0.05)

    def test_simulate_quantize_parameters(self):
        params = OrderedDict(
            {
                "fc.weight": torch.randn(32, 64),
                "fc.bias": torch.randn(32),
            }
        )
        simulated = simulate_quantize_parameters(params, nbits=4, block_size=32)

        for name in params:
            self.assertIsInstance(simulated[name], torch.Tensor)
            self.assertEqual(simulated[name].shape, params[name].shape)
            self.assertEqual(simulated[name].dtype, params[name].dtype)

    def test_calc_compression_stats(self):
        params = OrderedDict(
            {
                "conv.weight": torch.randn(64, 64, 3, 3),
                "conv.bias": torch.randn(64),
            }
        )
        q_params = quantize_sliced_parameters(params, nbits=4, block_size=64)
        stats = calc_compression_stats(params, q_params)

        self.assertGreater(stats["compression_ratio"], 6.0)
        self.assertGreater(stats["bandwidth_savings_pct"], 80.0)
        self.assertGreater(stats["raw_bytes"], stats["compressed_bytes"])

    def test_invalid_inputs(self):
        tensor = torch.randn(32)
        with self.assertRaises(ValueError):
            quantize_tensor(tensor, nbits=3)

        with self.assertRaises(ValueError):
            quantize_tensor(tensor, nbits=4, block_size=3)

    def test_heterofl_federation_integration(self):
        curr_dir = os.getcwd()
        src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
        try:
            os.chdir(src_dir)
            from config import cfg
            from fed import Federation

            cfg["model_split_mode"] = "fix"
            cfg["model_name"] = "conv"
            cfg["global_model_rate"] = 1.0
            cfg["model_rate"] = [1.0, 0.5, 0.25]
            cfg["num_users"] = 3
            cfg["scale"] = True

            global_params = OrderedDict(
                {
                    "conv.0.weight": torch.randn(64, 1, 3, 3),
                    "conv.0.bias": torch.randn(64),
                    "linear.weight": torch.randn(10, 64),
                    "linear.bias": torch.randn(10),
                }
            )

            fed = Federation(global_params, cfg["model_rate"], [[0, 1], [0, 1], [0, 1]])
            local_params, param_idx = fed.distribute([0, 1, 2])
            self.assertEqual(
                local_params[1]["conv.0.weight"].shape, torch.Size([32, 1, 3, 3])
            )

            quantized_locals = [
                quantize_sliced_parameters(lp, nbits=4, block_size=16)
                for lp in local_params
            ]
            dequantized_locals = [
                dequantize_sliced_parameters(qp) for qp in quantized_locals
            ]

            fed.combine(dequantized_locals, param_idx, [0, 1, 2])
            self.assertEqual(
                fed.global_parameters["conv.0.weight"].shape,
                torch.Size([64, 1, 3, 3]),
            )
        finally:
            os.chdir(curr_dir)


if __name__ == "__main__":
    unittest.main()
