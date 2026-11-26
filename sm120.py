"""RTX 5090 (sm_120) fix that runs inside the Miles Megatron trainer, through Miles' official init hook:

    --custom-megatron-init-path sm120.prefer_fa2

TransformerEngine's FlashAttention backend (`--attention-backend flash`) prefers FlashAttention 4 on sm_120 whenever
it is installed, and the image ships FA4 4.0.0b37 (with SGLang). On this GPU FA4 returns wrong dK/dV for packed
documents whose lengths aren't multiples of 64 (relative error ~1.2 vs fp32), and Miles packs documents at their
natural lengths, so training would get wrong gradients without any error. The image's flash-attn 2.7.4 has sm_120
kernels and matches the fp32 reference on the same inputs (probe_attention.py: flash vs flash-fa2). TE's cuDNN
backend (`fused`) is wrong on sm_120 too, and `unfused` is correct but materializes every score matrix.

This hides FA4 from TE's backend selection in the trainer process only, so TE falls back to its FA2 path. SGLang
(rollouts) is untouched. On any other GPU it does nothing.
"""

from __future__ import annotations


def prefer_fa2(args=None) -> None:
    import torch

    if torch.cuda.get_device_capability() != (12, 0):
        return
    # backends.py runs TE's FA2/FA3/FA4 detection at import; import it first so nothing re-enables FA4 afterwards
    import transformer_engine.pytorch.attention.dot_product_attention.backends  # noqa: F401
    from transformer_engine.pytorch.attention.dot_product_attention.utils import FlashAttentionUtils as fa

    if not fa.is_installed:
        raise RuntimeError("flash-attn 2 is not installed, so FA4 can't be avoided on sm_120; use --attention-backend unfused")
    if fa.v4_is_installed:
        fa.v4_is_installed = False
        print(f"[sm120] TE FlashAttention: FA4 {fa.fa4_version} disabled, using flash-attn {fa.version}", flush=True)
