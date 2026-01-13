# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright.
#
# This module monkey-patches Transformers' LlamaAttention to use FlashAttention (flash-attn) when available.
# If flash-attn (or its optional deps) is not installed, the patch is skipped and training can still proceed
# with the default attention implementation.

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import transformers
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

logger = logging.getLogger(__name__)


def replace_llama_attn_with_flash_attn() -> bool:
    """
    Try to patch `transformers.models.llama.modeling_llama.LlamaAttention.forward` to use flash-attn.

    Returns:
        True if the patch is applied; False otherwise.
    """
    try:
        from einops import rearrange  # type: ignore
    except ModuleNotFoundError as e:
        logger.warning("flash-attn patch skipped (missing `einops`): %s", e)
        return False

    try:
        # flash-attn API name differs across versions
        try:
            from flash_attn.flash_attn_interface import (  # type: ignore
                flash_attn_unpadded_qkvpacked_func,
            )
        except Exception:
            from flash_attn.flash_attn_interface import (  # type: ignore
                flash_attn_varlen_qkvpacked_func as flash_attn_unpadded_qkvpacked_func,
            )
        from flash_attn.bert_padding import unpad_input, pad_input  # type: ignore
    except ModuleNotFoundError as e:
        logger.warning("flash-attn patch skipped (missing `flash_attn`): %s", e)
        return False
    except Exception as e:  # pragma: no cover - depends on external flash-attn build
        logger.warning("flash-attn patch skipped: %s", e)
        return False

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # FlashAttention expects key_padding_mask semantics ([bsz, seq_len]) and does its own causal masking.
        return attention_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # Input shape: [bsz, q_len, hidden]
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        offset = 0
        if past_key_value is not None:
            offset = past_key_value[0].shape[-2]
            kv_seq_len += offset

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, offset=offset)

        assert not output_attentions, "output_attentions=True is not supported with flash-attn patch"
        assert not use_cache, "use_cache=True is not supported with flash-attn patch"
        assert past_key_value is None, "past_key_value is not supported with flash-attn patch"

        # Pack QKV to flash-attn expected shape.
        qkv = torch.stack([query_states, key_states, value_states], dim=2)  # [bsz, nh, 3, q_len, hd]
        qkv = qkv.transpose(1, 3)  # [bsz, q_len, 3, nh, hd]

        key_padding_mask = attention_mask  # [bsz, q_len] or None
        if key_padding_mask is None:
            qkv = rearrange(qkv, "b s ... -> (b s) ...")
            max_s = q_len
            cu_q_lens = torch.arange(
                0, (bsz + 1) * q_len, step=q_len, dtype=torch.int32, device=qkv.device
            )
            output = flash_attn_unpadded_qkvpacked_func(
                qkv, cu_q_lens, max_s, 0.0, softmax_scale=None, causal=True
            )
            output = rearrange(output, "(b s) ... -> b s ...", b=bsz)
        else:
            nheads = qkv.shape[-2]
            x = rearrange(qkv, "b s three h d -> b s (three h d)")
            x_unpad, indices, cu_q_lens, max_s = unpad_input(x, key_padding_mask)
            x_unpad = rearrange(x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads)
            output_unpad = flash_attn_unpadded_qkvpacked_func(
                x_unpad, cu_q_lens, max_s, 0.0, softmax_scale=None, causal=True
            )
            output = rearrange(
                pad_input(rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, bsz, q_len),
                "b s (h d) -> b s h d",
                h=nheads,
            )

        return self.o_proj(rearrange(output, "b s h d -> b s (h d)")), None, None

    transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = (  # type: ignore[attr-defined]
        _prepare_decoder_attention_mask
    )
    transformers.models.llama.modeling_llama.LlamaAttention.forward = forward  # type: ignore[attr-defined]
    return True

