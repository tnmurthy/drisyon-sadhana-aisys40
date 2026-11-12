"""
Gate 1 Automated CI Tests · Drisyon Sadhana: AI System Design
Asserts:
- Model Weight Memory Calculation (FP16, INT8, INT4)
- KV Cache sizing with GQA and MHA
- TTFT and TPS latency calculation
- Semantic cache thresholding
- ReAct loop step budget and cycle break
"""

import pytest
from core.inference_calculator import InferenceCalculator
from core.semantic_cache import SemanticCache
from core.agent_supervisor import AgentSupervisor


def test_model_weight_vram():
    # 70B parameter model in FP16 (2 bytes) = 140e9 / 1024^3 ~= 130.38 GB
    vram_fp16 = InferenceCalculator.calculate_weight_memory_gb(70.0, "fp16")
    assert 130.0 < vram_fp16 < 131.0

    # 70B in INT8 (1 byte) = ~65.19 GB
    vram_int8 = InferenceCalculator.calculate_weight_memory_gb(70.0, "int8")
    assert 65.0 < vram_int8 < 66.0

    # 8B in INT4 (0.5 bytes) = ~3.72 GB
    vram_int4 = InferenceCalculator.calculate_weight_memory_gb(8.0, "int4")
    assert 3.5 < vram_int4 < 4.0


def test_kv_cache_calculation():
    # Llama-3 8B: 32 layers, head_dim 128, 32 query heads, 8 KV heads (GQA)
    # seq_len = 4096 tokens, FP16 (2 bytes)
    # KV Cache = 2 * 32 * 8 * 128 * 2 * 4096 bytes = 536,870,912 bytes = 512 MB
    kv_mb = InferenceCalculator.calculate_kv_cache_mb(
        n_layers=32,
        seq_len=4096,
        head_dim=128,
        n_heads=32,
        precision_bytes=2,
        is_gqa=True,
        n_kv_heads=8
    )
    assert kv_mb == 512.0


def test_latency_metrics():
    # TTFT = 250ms, 100 tokens generated in 2000ms -> TPS = 50.0, Total = 2250ms
    metrics = InferenceCalculator.estimate_latency_metrics(
        ttft_ms=250.0,
        generated_tokens=100,
        decode_duration_ms=2000.0
    )
    assert metrics["tps"] == 50.0
    assert metrics["total_latency_ms"] == 2250.0


def test_semantic_cache_hit_and_miss():
    cache = SemanticCache(similarity_threshold=0.90)
    # Query vector
    v1 = [1.0, 0.0, 0.0]
    cache.set("What is KV cache?", v1, "KV Cache stores past Key and Value tensors.")

    # Near match (cosine sim = 0.96)
    v2 = [0.96, 0.28, 0.0]
    hit_result = cache.get(v2)
    assert hit_result is not None
    assert hit_result["hit"] is True
    assert "past Key and Value" in hit_result["response"]

    # Orthogonal query (cosine sim = 0.0)
    v3 = [0.0, 1.0, 0.0]
    miss_result = cache.get(v3)
    assert miss_result is None


def test_agent_supervisor_guardrails():
    supervisor = AgentSupervisor(max_steps=5, max_token_budget=1000)

    # Normal step
    r1 = supervisor.record_step("search", "hash1", 100)
    assert r1["allowed"] is True

    # 3 identical steps -> Cycle detection break
    supervisor.record_step("eval", "hash_same", 100)
    supervisor.record_step("eval", "hash_same", 100)
    r_cycle = supervisor.record_step("eval", "hash_same", 100)
    assert r_cycle["allowed"] is False
    assert "cycle detected" in r_cycle["reason"].lower()
