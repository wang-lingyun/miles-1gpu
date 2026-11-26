"""Check Megatron's attention backends on this GPU the way Miles trains: packed sequences (qkv_format thd, causal,
GQA 16/8, head_dim 128, bf16), forward and backward, against a float32 reference computed per document.

Why: on the RTX 5090 (sm_120) a kernel can run without error and still return wrong gradients. FA4 did exactly that
for multi-sequence varlen batches (see trl_rl docs/research.md). Each backend runs in its own process because
Transformer Engine picks its backend from NVTE_* variables, the ones Megatron's --attention-backend sets.

    python probe_attention.py               # flash-fa2, flash, fused, unfused
    python probe_attention.py flash-fa2     # what the rtx5090 profile trains with (flash + sm120.prefer_fa2 hook)
"""

from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys

BACKEND_ENV = {
    # --attention-backend flash plus the trainer hook from the rtx5090 profile
    "flash-fa2": {"NVTE_FLASH_ATTN": "1", "NVTE_FUSED_ATTN": "0", "NVTE_UNFUSED_ATTN": "0"},
    "flash": {"NVTE_FLASH_ATTN": "1", "NVTE_FUSED_ATTN": "0", "NVTE_UNFUSED_ATTN": "0"},
    "fused": {"NVTE_FLASH_ATTN": "0", "NVTE_FUSED_ATTN": "1", "NVTE_UNFUSED_ATTN": "0"},
    "unfused": {"NVTE_FLASH_ATTN": "0", "NVTE_FUSED_ATTN": "0", "NVTE_UNFUSED_ATTN": "1"},
}
TOLERANCE = 0.02  # relative L2 error vs fp32; bf16 kernels land around 0.005-0.01
HEADS, GROUPS, HEAD_DIM = 16, 8, 128  # Qwen3-0.6B
# several documents packed into one micro-batch, odd sizes included; PROBE_LENGTHS=64,128 overrides
LENGTHS = [int(x) for x in os.environ.get("PROBE_LENGTHS", "37,50,123,300,1").split(",")]


def reference(q, k, v, cu, dout):
    import torch

    qf, kf, vf = (x.detach().float().requires_grad_() for x in (q, k, v))
    outs = []
    for s, e in itertools.pairwise(cu.tolist()):
        qs = qf[s:e].transpose(0, 1)
        ks = kf[s:e].repeat_interleave(HEADS // GROUPS, dim=1).transpose(0, 1)
        vs = vf[s:e].repeat_interleave(HEADS // GROUPS, dim=1).transpose(0, 1)
        scores = (qs @ ks.transpose(-1, -2)) / HEAD_DIM**0.5
        mask = torch.ones(e - s, e - s, dtype=torch.bool, device=q.device).triu(1)
        probs = scores.masked_fill(mask, float("-inf")).softmax(-1)
        outs.append((probs @ vs).transpose(0, 1).reshape(e - s, HEADS * HEAD_DIM))
    ref = torch.cat(outs)
    ref.backward(dout.float())
    return ref.detach(), qf.grad, kf.grad, vf.grad


def probe_here(backend: str) -> dict:
    import torch
    import transformer_engine
    import transformer_engine.pytorch as te

    if backend == "flash-fa2":
        import sm120

        sm120.prefer_fa2()
    torch.manual_seed(0)
    dev = "cuda"
    cu = torch.tensor([0, *itertools.accumulate(LENGTHS)], dtype=torch.int32, device=dev)
    total = int(cu[-1])
    q = torch.randn(total, HEADS, HEAD_DIM, device=dev, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(total, GROUPS, HEAD_DIM, device=dev, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(total, GROUPS, HEAD_DIM, device=dev, dtype=torch.bfloat16, requires_grad=True)

    attn = te.DotProductAttention(
        num_attention_heads=HEADS, kv_channels=HEAD_DIM, num_gqa_groups=GROUPS,
        attention_dropout=0.0, qkv_format="thd", attn_mask_type="padding_causal",
    )
    out = attn(q, k, v, cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=max(LENGTHS), max_seqlen_kv=max(LENGTHS))
    out = out.reshape(total, HEADS * HEAD_DIM)
    dout = torch.randn_like(out)
    out.backward(dout)
    torch.cuda.synchronize()

    ref, dq, dk, dv = reference(q, k, v, cu, dout)

    def rel(a, b):
        return round(((a.float() - b).norm() / b.norm()).item(), 5)

    errors = {"fwd": rel(out, ref), "dq": rel(q.grad, dq), "dk": rel(k.grad, dk), "dv": rel(v.grad, dv)}
    return {
        "backend": backend,
        "ok": all(e < TOLERANCE for e in errors.values()),
        **errors,
        "te": transformer_engine.__version__,
        "torch": torch.__version__,
        "cudnn": torch.backends.cudnn.version(),
        "device": f"{torch.cuda.get_device_name()} sm_{''.join(map(str, torch.cuda.get_device_capability()))}",
    }


def probe_in_subprocess(backend: str) -> dict:
    env = {**os.environ, **BACKEND_ENV[backend], "NVTE_DEBUG": "1", "NVTE_DEBUG_LEVEL": "1"}
    proc = subprocess.run([sys.executable, __file__, "--worker", backend], env=env, capture_output=True, text=True,
                          timeout=600)
    selected = sorted({line.split("Running with", 1)[1].strip() for line in (proc.stdout + proc.stderr).splitlines()
                       if "Running with" in line})
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("{"):
            return {**json.loads(line), "selected": selected}
    tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
    return {"backend": backend, "ok": False, "error": " | ".join(tail), "selected": selected}


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        print(json.dumps(probe_here(sys.argv[2])), flush=True)
        return
    backends = sys.argv[1:] or list(BACKEND_ENV)
    results = [probe_in_subprocess(b) for b in backends]
    for r in results:
        print(json.dumps(r), flush=True)
    sys.exit(0 if all(r["ok"] for r in results if r["backend"] == "flash-fa2") else 1)


if __name__ == "__main__":
    main()
