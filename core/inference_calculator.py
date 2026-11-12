"""
Drisyon Sadhana · AI System Design Starter Kit
Module 1: LLM Inference Hardware & Memory Calculator
"""

from typing import Dict, Any, Literal


class InferenceCalculator:
    """Calculates model weight memory, KV Cache requirements, and inference metrics."""

    @staticmethod
    def calculate_weight_memory_gb(param_count_billions: float, precision: Literal["fp16", "int8", "int4"]) -> float:
        """
        Calculates raw model weight memory in Gigabytes.
        FP16: 2 bytes per param
        INT8: 1 byte per param
        INT4: 0.5 bytes per param
        """
        bytes_per_param = {
            "fp16": 2.0,
            "int8": 1.0,
            "int4": 0.5
        }
        if precision not in bytes_per_param:
            raise ValueError(f"Unsupported precision: {precision}")

        total_bytes = param_count_billions * 1e9 * bytes_per_param[precision]
        return total_bytes / (1024 ** 3)

    @staticmethod
    def calculate_kv_cache_bytes(
        n_layers: int,
        seq_len: int,
        head_dim: int,
        n_heads: int,
        precision_bytes: int = 2,
        is_gqa: bool = False,
        n_kv_heads: int = 8
    ) -> int:
        """
        Calculates KV Cache size in Bytes for a single request.
        KV Cache = 2 (keys + values) * n_layers * effective_heads * head_dim * precision_bytes * seq_len
        """
        effective_heads = n_kv_heads if is_gqa else n_heads
        return 2 * n_layers * effective_heads * head_dim * precision_bytes * seq_len

    @staticmethod
    def calculate_kv_cache_mb(
        n_layers: int,
        seq_len: int,
        head_dim: int,
        n_heads: int,
        precision_bytes: int = 2,
        is_gqa: bool = False,
        n_kv_heads: int = 8
    ) -> float:
        """Calculates KV Cache in Megabytes."""
        bytes_val = InferenceCalculator.calculate_kv_cache_bytes(
            n_layers, seq_len, head_dim, n_heads, precision_bytes, is_gqa, n_kv_heads
        )
        return bytes_val / (1024 ** 2)

    @staticmethod
    def estimate_latency_metrics(ttft_ms: float, generated_tokens: int, decode_duration_ms: float) -> Dict[str, Any]:
        """Calculates Tokens Per Second (TPS) and Total Request Latency."""
        if decode_duration_ms <= 0:
            raise ValueError("Decode duration must be positive.")

        tps = (generated_tokens / decode_duration_ms) * 1000.0
        total_latency_ms = ttft_ms + decode_duration_ms
        return {
            "ttft_ms": ttft_ms,
            "tps": round(tps, 2),
            "generated_tokens": generated_tokens,
            "total_latency_ms": round(total_latency_ms, 2)
        }
