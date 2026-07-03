# coding=utf-8
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team.
# Adapted for SketchWalk sparse attention.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional, Union

import torch
from torch import nn
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask  # kept for reference
from transformers.modeling_layers import (
    GenericForQuestionAnswering,
    GenericForSequenceClassification,
    GenericForTokenClassification,
    GradientCheckpointingLayer,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

@use_kernel_forward_from_hub("RMSNorm")
class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


# ---------------------------------------------------------------------------
# Rotary Embedding (Qwen3-style with rope_parameters dict)
# ---------------------------------------------------------------------------

class Qwen3RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, config: Qwen3Config, device=None):
        super().__init__()
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type", "default"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config

        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ---------------------------------------------------------------------------
# MLP / repeat_kv
# ---------------------------------------------------------------------------

class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# ---------------------------------------------------------------------------
# Attention backends
# ---------------------------------------------------------------------------

def sdpa_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    rw_allow_blocks: Optional[torch.Tensor] = None,
    rw_block_size: Optional[int] = None,
    rw_tq: Optional[int] = None,
    rw_tk: Optional[int] = None,
    rw_window_tokens: Optional[int] = None,
    **kwargs,
):
    if hasattr(module, "num_key_value_groups"):
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)

    if attention_mask is not None:
        if attention_mask.dim() == 2:
            attention_mask = attention_mask[:, None, None, :]
        elif attention_mask.dim() == 3:
            attention_mask = attention_mask.unsqueeze(1)
        elif attention_mask.dim() == 4 and attention_mask.size(1) == 0:
            attention_mask = attention_mask.new_zeros(
                (attention_mask.size(0), 1, attention_mask.size(2), attention_mask.size(3))
            )

    is_causal = False

    if query.dim() == 4 and rw_allow_blocks is not None:
        B, H, T_q, D = query.shape
        token_window_mask = None
        if rw_window_tokens is not None and rw_window_tokens > 0:
            q_idx = torch.arange(T_q, device=query.device).view(T_q, 1)
            k_idx = torch.arange(T_q if rw_tk is None else rw_tk, device=query.device).view(1, -1)
            token_window_mask = (k_idx <= q_idx) & (k_idx >= (q_idx - rw_window_tokens + 1))
            sink_mask = k_idx < rw_window_tokens
            token_window_mask = token_window_mask | sink_mask

        block_size = rw_block_size or 1
        allow_blocks = rw_allow_blocks
        num_blocks_q, num_blocks_k = allow_blocks.shape[1], allow_blocks.shape[2]
        allow = allow_blocks[:, :, None, :, None].expand(
            B, num_blocks_q, block_size, num_blocks_k, block_size
        )
        allow = allow.reshape(B, num_blocks_q * block_size, num_blocks_k * block_size)
        allow = allow[:, : (rw_tq or T_q), : (rw_tk or allow.size(-1))]
        if token_window_mask is not None:
            allow = allow | token_window_mask.unsqueeze(0)

        mask = attention_mask
        neg_inf = torch.finfo(query.dtype).min
        if mask is None:
            mask = (~allow).unsqueeze(1).to(query.dtype) * neg_inf
        else:
            if mask.dim() == 2:
                mask = mask[:, None, None, :]
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            allow_mask = (~allow).unsqueeze(1).to(mask.dtype) * neg_inf
            mask = mask + allow_mask

        attn_output = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=mask,
            dropout_p=dropout,
            scale=scaling,
            is_causal=is_causal,
        )
    else:
        attn_output = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attention_mask,
            dropout_p=dropout,
            scale=scaling,
            is_causal=is_causal,
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


# ---------------------------------------------------------------------------
# Qwen3Attention with SketchWalk
# ---------------------------------------------------------------------------

class Qwen3Attention(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Qwen3-specific: per-layer type (full_attention vs sliding_attention)
        self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else "full_attention"
        self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None

        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

        # Qwen3-specific: QK norms
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # ---- SketchWalk controls ----
        self.sparsity_mode = "both"
        self.random_walk_window = getattr(config, "random_walk_window", 0.1)
        # Sink size (fraction of context).  None → tied to random_walk_window.
        self.random_walk_sink = getattr(config, "random_walk_sink", None)
        self.walk_damping = getattr(config, "walk_damping", 0.25)
        self.random_walk_block_size = getattr(config, "random_walk_block_size", 64)
        self.random_walk_hadamard_dim = getattr(config, "random_walk_hadamard_dim", 128)
        self.random_walk_degree = getattr(config, "random_walk_degree", 8)
        self.random_walk_exact = getattr(config, "random_walk_exact", False)
        self.random_walk_query_window = getattr(config, "random_walk_query_window", None)
        print(f"Hadamard sketching dim is: {self.random_walk_hadamard_dim}")
        self._hadamard_params = {}

    # ---- SketchWalk helpers (identical to LlamaAttention) ----

    def _get_random_walk_states(self, past_key_values, runtime_states):
        if past_key_values is not None:
            states = getattr(past_key_values, "random_walk_states", None)
            if states is None:
                states = {}
                setattr(past_key_values, "random_walk_states", states)
            return states
        return runtime_states

    def _get_random_walk_cache(self, past_key_values, runtime_cache):
        if past_key_values is not None:
            cache = getattr(past_key_values, "random_walk_cache", None)
            if cache is None:
                cache = {}
                setattr(past_key_values, "random_walk_cache", cache)
            return cache
        return runtime_cache

    def _prepare_random_walk_state(self, prev, target_len, batch_size, dtype, device):
        if prev is None or prev.size(0) != batch_size:
            base = torch.eye(target_len, device=device, dtype=dtype).unsqueeze(0)
            if batch_size > 1:
                base = base.repeat(batch_size, 1, 1)
            return base
        prev = prev.to(device=device, dtype=dtype)
        prev_len = prev.size(-1)
        if prev_len < target_len:
            base = torch.eye(target_len, device=device, dtype=dtype).unsqueeze(0)
            if batch_size > 1:
                base = base.repeat(batch_size, 1, 1)
            base[:, :prev_len, :prev_len] = prev
            return base
        elif prev_len > target_len:
            return prev[:, :target_len, :target_len]
        return prev

    def _get_hadamard_params(self, n_pow2: int, device: torch.device, dtype: torch.dtype):
        key = (str(device), n_pow2)
        params = self._hadamard_params.get(key)
        if params is None:
            sign = torch.randint(0, 2, (1, 1, 1, n_pow2), device=device, dtype=torch.int8)
            idx = torch.randperm(n_pow2, device=device)
            params = (sign, idx)
            self._hadamard_params[key] = params
        sign, idx = params
        return sign.to(dtype=dtype), idx

    def _fwht(self, x: torch.Tensor) -> torch.Tensor:
        n = x.size(-1)
        y = x.reshape(-1, n)
        h = 1
        while h < n:
            y = y.view(-1, n // (2 * h), 2 * h)
            a = y[..., :h].clone()
            b = y[..., h : 2 * h]
            y[..., :h] = a + b
            y[..., h : 2 * h] = a - b
            h *= 2
        return y.view(*x.shape)

    def _hadamard_project(self, x: torch.Tensor, proj_dim: int) -> torch.Tensor:
        n = x.size(-1)
        n_pow2 = 1 << (n - 1).bit_length()
        if n_pow2 != n:
            x = F.pad(x, (0, n_pow2 - n), value=0.0)
        sign, idx = self._get_hadamard_params(n_pow2, x.device, x.dtype)
        sign = sign * 2 - 1
        x = x * sign
        x = self._fwht(x)
        proj_dim = min(proj_dim, n_pow2)
        x = x / ((n_pow2 / proj_dim) ** 0.5)
        return x.index_select(-1, idx[:proj_dim])

    def _hadamard_attention_probs(self, query_states, key_states, attention_mask):
        if key_states.size(1) != query_states.size(1):
            key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        proj_dtype = torch.bfloat16
        q = self._hadamard_project(query_states.to(proj_dtype), self.random_walk_hadamard_dim)
        k = self._hadamard_project(key_states.to(proj_dtype), self.random_walk_hadamard_dim)
        scale = q.size(-1) ** -0.5
        logits = torch.einsum("bhqd,bhkd->bhqk", q, k) * scale
        if attention_mask is not None:
            logits = logits + attention_mask
        attn_probs = torch.softmax(logits.clamp(-10, 10), dim=-1)
        pt_tok = attn_probs.sum(dim=1)
        return pt_tok / (pt_tok.sum(dim=-1, keepdim=True) + 1e-12)

    def _block_pool(self, x: torch.Tensor, block_size: int) -> torch.Tensor:
        if block_size <= 1:
            return x
        B, H, T, D = x.shape
        num_blocks = (T + block_size - 1) // block_size
        pad = num_blocks * block_size - T
        if pad:
            x = F.pad(x, (0, 0, 0, pad), value=0.0)
        return x.view(B, H, num_blocks, block_size, D).mean(dim=3)

    def _hadamard_attention_probs_block(self, query_states, key_states, attention_mask, block_size):
        if key_states.size(1) != query_states.size(1):
            key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        q = self._block_pool(query_states, block_size)
        k = self._block_pool(key_states, block_size)
        proj_dtype = torch.bfloat16
        q = self._hadamard_project(q.to(proj_dtype), self.random_walk_hadamard_dim)
        k = self._hadamard_project(k.to(proj_dtype), self.random_walk_hadamard_dim)
        scale = q.size(-1) ** -0.5
        logits = torch.einsum("bhqd,bhkd->bhqk", q, k) * scale
        if attention_mask is not None:
            logits = logits + attention_mask
        attn_probs = torch.softmax(logits / 100, dim=-1)
        pt_blk = attn_probs.sum(dim=1)
        return pt_blk / (pt_blk.sum(dim=-1, keepdim=True) + 1e-12)

    def _exact_attention_probs_block(self, query_states, key_states, attention_mask, block_size):
        if key_states.size(1) != query_states.size(1):
            key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        q = self._block_pool(query_states, block_size)
        k = self._block_pool(key_states, block_size)
        scale = q.size(-1) ** -0.5
        logits = torch.einsum("bhqd,bhkd->bhqk", q, k) * scale
        if attention_mask is not None:
            _, _, T_q, T_k = attention_mask.shape
            n_q = (T_q + block_size - 1) // block_size
            n_k = (T_k + block_size - 1) // block_size
            mask_blk = attention_mask[:, :, ::block_size, ::block_size][..., :n_q, :n_k]
            logits = logits + mask_blk
        attn_probs = torch.softmax(logits.clamp(-10, 10), dim=-1)
        pt_blk = attn_probs.sum(dim=1)
        return pt_blk / (pt_blk.sum(dim=-1, keepdim=True) + 1e-12)

    def _exact_attention_probs_block_row(self, query_states, key_states):
        scale = query_states.size(-1) ** -0.5
        logits = torch.einsum("bhqd,bhkd->bhqk", query_states, key_states) * scale
        attn_probs = torch.softmax(logits.clamp(-10, 10), dim=-1)
        pt_blk = attn_probs.sum(dim=1)
        return pt_blk / (pt_blk.sum(dim=-1, keepdim=True) + 1e-12)

    def _hadamard_attention_probs_block_row(self, query_states, key_states):
        proj_dtype = torch.bfloat16
        q = self._hadamard_project(query_states.to(proj_dtype), self.random_walk_hadamard_dim)
        k = self._hadamard_project(key_states.to(proj_dtype), self.random_walk_hadamard_dim)
        scale = q.size(-1) ** -0.5
        logits = torch.einsum("bhqd,bhkd->bhqk", q, k) * scale
        attn_probs = torch.softmax(logits / 100, dim=-1)
        pt_blk = attn_probs.sum(dim=1)
        return pt_blk / (pt_blk.sum(dim=-1, keepdim=True) + 1e-12)

    def _update_random_walk(self, attention_probs, past_key_values, runtime_states):
        per_head = attention_probs.dim() == 4
        if per_head:
            B, H, T_q, T_k = attention_probs.shape
            attention_probs = attention_probs.reshape(B * H, T_q, T_k)
        else:
            B, T_q, T_k = attention_probs.shape
        states = self._get_random_walk_states(past_key_values, runtime_states)
        prev = states.get(self.layer_idx - 1, None) if states is not None else None
        if prev is None:
            state = attention_probs
        else:
            state = self._prepare_random_walk_state(
                prev=prev, target_len=T_k, batch_size=B,
                dtype=attention_probs.dtype, device=attention_probs.device,
            )
        beta = float(self.walk_damping)
        anchor = attention_probs
        for _ in range(self.random_walk_degree):
            state = (1.0 - beta) * anchor + beta * torch.matmul(state, attention_probs)
        walk = state
        if states is not None:
            states[self.layer_idx] = walk.detach()
        if not per_head:
            return walk
        return walk.view(B, H, T_q, T_k)

    def _rw_prefill(
        self, query_states, key_states, value_states, attention_mask,
        past_key_values, random_walk_states, random_walk_cache,
        block_size, attention_interface, **kwargs,
    ):
        B, H, T_q, D = query_states.shape
        T_k = key_states.shape[2]
        device = query_states.device

        if self.random_walk_exact:
            teacher_attention_probs = self._exact_attention_probs_block(
                query_states, key_states, attention_mask=None, block_size=block_size
            )
        else:
            teacher_attention_probs = self._hadamard_attention_probs_block(
                query_states, key_states, attention_mask=None, block_size=block_size
            )
        rw = self._update_random_walk(teacher_attention_probs.detach(), past_key_values, random_walk_states)

        rw_cache = self._get_random_walk_cache(past_key_values, random_walk_cache)
        if rw_cache is not None:
            layer_cache = rw_cache.get(self.layer_idx)
            if layer_cache is None:
                layer_cache = {}
                rw_cache[self.layer_idx] = layer_cache
            layer_cache["a_blk"] = teacher_attention_probs.detach()
            # Seed q_window with the last W prefill queries so decode never starts cold.
            W = int(self.random_walk_query_window or block_size)
            W_eff = min(W, query_states.size(2))
            layer_cache["q_window"] = query_states[:, :, -W_eff:, :].detach()

        if block_size > 1:
            num_blocks_q = rw.size(1)
            num_blocks_k = rw.size(2)
            q_idx = torch.arange(num_blocks_q, device=device).view(num_blocks_q, 1)
            k_idx = torch.arange(num_blocks_k, device=device).view(1, num_blocks_k)
            causal_blocks = (k_idx <= q_idx).view(1, num_blocks_q, num_blocks_k)
            masked_probs = rw.float().masked_fill(~causal_blocks, float("nan"))

            block_window = self.random_walk_window
            if block_window is not None and 0 < block_window < 1:
                block_window = max(1, int(block_window * T_k))
            window_blocks = None
            if block_window is not None and block_window > 0:
                window_blocks = max(1, (block_window + block_size - 1) // block_size)
                window_blocks = min(window_blocks, num_blocks_k)

            exclude = torch.zeros_like(masked_probs, dtype=torch.bool)
            rw_sink = getattr(self, "random_walk_sink", None)
            if rw_sink is not None:
                rw_sink_tokens = max(1, int(rw_sink * T_k)) if 0 < rw_sink < 1 else int(rw_sink)
                sink_blocks = min(num_blocks_k, max(1, (rw_sink_tokens + block_size - 1) // block_size))
            else:
                sink_blocks = window_blocks if window_blocks is not None else 1
            exclude[:, :, :sink_blocks] = True
            if window_blocks is not None and window_blocks > 0:
                win_lo = q_idx - (window_blocks - 1)
                window_mask = (k_idx <= q_idx) & (k_idx >= win_lo)
                exclude |= window_mask.view(1, num_blocks_q, num_blocks_k)

            masked_probs = masked_probs.masked_fill(exclude, float("nan"))
            k_frac = getattr(self.config, "random_walk_kblocks_frac", 0.1)
            th = torch.nanquantile(masked_probs, 1 - k_frac, dim=-1, keepdim=True)
            if isinstance(th, torch.Tensor):
                th = th.clone()
                th[:, -4:, :] = 0
            allow_blocks = rw > th
        else:
            allow_blocks = rw >= 0
            block_window = None

        attn_out, attn_weights = attention_interface(
            self, query_states, key_states, value_states,
            attention_mask=attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            rw_allow_blocks=allow_blocks,
            rw_block_size=block_size,
            rw_tq=T_q,
            rw_tk=T_k,
            rw_window_tokens=block_window,
            **kwargs,
        )
        return attn_out, attn_weights, teacher_attention_probs

    def _rw_decode(
        self, query_states, key_states, value_states, attention_mask,
        past_key_values, random_walk_cache, block_size, attention_interface, **kwargs,
    ):
        B, H, T_q, D = query_states.shape
        T_k = key_states.shape[2]
        device = query_states.device

        rw_cache = self._get_random_walk_cache(past_key_values, random_walk_cache)
        layer_cache = rw_cache.get(self.layer_idx) if rw_cache is not None else None
        if layer_cache is None:
            layer_cache = {}
            if rw_cache is not None:
                rw_cache[self.layer_idx] = layer_cache

        q_tok = query_states[:, :, -1:, :]
        attn_row = self._hadamard_attention_probs(q_tok, key_states, attention_mask).squeeze(1)

        block_size = int(getattr(self, "random_walk_block_size", 64) or 64)
        a_blk = layer_cache.get("a_blk", None)

        # Bootstrap a_blk for decoding-only mode (prefill didn't run _rw_prefill).
        # Without this, the fallback uses token-level indices as block indices,
        # which restricts attention to only the first (T_k-1)//block_size tokens.
        if a_blk is None and rw_cache is not None:
            k_blk = self._block_pool(key_states, block_size)
            if k_blk.size(1) != q_tok.size(1):
                k_blk = k_blk.repeat_interleave(self.num_key_value_groups, dim=1)
            if self.random_walk_exact:
                row = self._exact_attention_probs_block_row(q_tok, k_blk).squeeze(1)
            else:
                row = self._hadamard_attention_probs_block_row(q_tok, k_blk).squeeze(1)
            Tb = row.size(-1)
            a_blk = row.unsqueeze(1).expand(-1, Tb, -1).clone()
            layer_cache["a_blk"] = a_blk

        if a_blk is not None and rw_cache is not None:
            W = int(self.random_walk_query_window or block_size)

            prev_window = layer_cache.get("q_window", None)
            if prev_window is None:
                q_window = q_tok
            else:
                if prev_window.size(1) != q_tok.size(1):
                    prev_window = prev_window[:, : q_tok.size(1)]
                q_window = torch.cat([prev_window.to(q_tok.dtype), q_tok], dim=2)
                if q_window.size(2) > W:
                    q_window = q_window[:, :, -W:, :]
            layer_cache["q_window"] = q_window.detach()

            k_blk = self._block_pool(key_states, block_size)
            if k_blk.size(1) != q_window.size(1):
                k_blk = k_blk.repeat_interleave(self.num_key_value_groups, dim=1)

            if self.random_walk_exact:
                q_scoring, k_scoring = q_window, k_blk
                scale = q_scoring.size(-1) ** -0.5
                logits = torch.einsum("bhwd,bhkd->bhwk", q_scoring, k_scoring) * scale
                attn_probs = torch.softmax(logits.clamp(-10, 10), dim=-1)
            else:
                proj_dtype = torch.bfloat16
                q_scoring = self._hadamard_project(q_window.to(proj_dtype), self.random_walk_hadamard_dim)
                k_scoring = self._hadamard_project(k_blk.to(proj_dtype),    self.random_walk_hadamard_dim)
                scale = q_scoring.size(-1) ** -0.5
                logits = torch.einsum("bhwd,bhkd->bhwk", q_scoring, k_scoring) * scale
                attn_probs = torch.softmax(logits / 100, dim=-1)
            attn_row_blk = attn_probs.sum(dim=(1, 2))
            attn_row_blk = attn_row_blk / (attn_row_blk.sum(dim=-1, keepdim=True) + 1e-12)
            Tb_cur = attn_row_blk.size(-1)
            Tb_prev = a_blk.size(-1)

            if Tb_cur > Tb_prev:
                pad = Tb_cur - Tb_prev
                a_blk = F.pad(a_blk, (0, pad))
                new_rows = attn_row_blk.unsqueeze(1).expand(-1, pad, -1)
                a_blk = torch.cat([a_blk, new_rows], dim=1)
                layer_cache["a_blk"] = a_blk

            cur_blk_idx = min((T_k - 1) // block_size, a_blk.size(1) - 1)
            step_in_block = (T_k - 1) % block_size
            alpha = 1.0 if step_in_block == 0 else 1.0 / (step_in_block + 1)
            a_blk = a_blk.clone()
            a_blk[:, cur_blk_idx, :Tb_cur] = (1 - alpha) * a_blk[:, cur_blk_idx, :Tb_cur] + alpha * attn_row_blk
            row_sum = a_blk[:, cur_blk_idx].sum(dim=-1, keepdim=True).clamp(min=1e-12)
            a_blk[:, cur_blk_idx] = a_blk[:, cur_blk_idx] / row_sum
            a_blk[:, :Tb_cur, cur_blk_idx] = (1 - alpha) * a_blk[:, :Tb_cur, cur_blk_idx] + alpha * attn_row_blk
            col_sum = a_blk[:, :, cur_blk_idx].sum(dim=-1, keepdim=True).clamp(min=1e-12)
            a_blk[:, :, cur_blk_idx] = a_blk[:, :, cur_blk_idx] / col_sum.squeeze(-1).unsqueeze(-1)
            layer_cache["a_blk"] = a_blk

            beta = float(self.walk_damping)
            state = attn_row_blk
            for _ in range(self.random_walk_degree):
                state = (1.0 - beta) * attn_row_blk + beta * torch.bmm(
                    state.unsqueeze(1), a_blk
                ).squeeze(1)
            walk_blk = state

            walk_tok = walk_blk.repeat_interleave(block_size, dim=-1)[:, :T_k]
            attn_row = walk_tok

        if a_blk is not None and rw_cache is not None:
            rw = walk_blk.unsqueeze(1)
        else:
            rw = attn_row.view(B, 1, -1)

        Tb = rw.size(-1)
        q_blk_idx = (T_k - 1) // block_size
        k_blk_idx = torch.arange(Tb, device=device).view(1, Tb)
        causal_blocks = (k_blk_idx <= q_blk_idx).view(1, 1, Tb)
        masked_probs = rw.float().masked_fill(~causal_blocks, float("nan"))

        rw_win = getattr(self, "random_walk_window", None)
        window_blocks = None
        if rw_win is not None:
            rw_win_tokens = max(1, int(rw_win * T_k)) if 0 < rw_win < 1 else int(rw_win)
            window_blocks = max(1, (rw_win_tokens + block_size - 1) // block_size)
            window_blocks = min(window_blocks, Tb)

        exclude = torch.zeros_like(masked_probs, dtype=torch.bool)
        rw_sink = getattr(self, "random_walk_sink", None)
        if rw_sink is not None:
            rw_sink_tokens = max(1, int(rw_sink * T_k)) if 0 < rw_sink < 1 else int(rw_sink)
            sink_blocks = min(Tb, max(1, (rw_sink_tokens + block_size - 1) // block_size))
        else:
            sink_blocks = window_blocks if window_blocks is not None else 1
        exclude[:, :, :sink_blocks] = True
        if window_blocks is not None and window_blocks > 0:
            win_lo_blk = q_blk_idx - (window_blocks - 1)
            window_blk_mask = (k_blk_idx <= q_blk_idx) & (k_blk_idx >= win_lo_blk)
            exclude |= window_blk_mask.view(1, 1, Tb)

        masked_probs = masked_probs.masked_fill(exclude, float("nan"))
        k_frac = getattr(self.config, "random_walk_kblocks_frac", 0.1)
        th = torch.nanquantile(masked_probs, 1 - k_frac, dim=-1, keepdim=True)
        if isinstance(th, torch.Tensor):
            th = th.clone()
            th[:, :, :1] = 0
        allow_blocks = rw > th

        block_window = rw_win
        if block_window is not None and 0 < block_window < 1:
            block_window = max(1, int(block_window * T_k))

        attn_out, attn_weights = attention_interface(
            self, query_states, key_states, value_states,
            attention_mask=attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            rw_allow_blocks=allow_blocks,
            rw_block_size=block_size,
            rw_tq=T_q,
            rw_tk=T_k,
            rw_window_tokens=block_window,
            **kwargs,
        )
        return attn_out, attn_weights

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        random_walk_states: Optional[dict] = None,
        random_walk_cache: Optional[dict] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], dict]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Qwen3: apply q_norm and k_norm on head dimension after projection
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states   = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        B, H, T_q, D = query_states.shape
        T_k = key_states.shape[2]

        attention_interface = sdpa_attention_forward
        estimator_log_probs = None
        teacher_attention_probs = None
        attn_out_masked = None
        attn_weights_masked = None

        is_prefill = (T_q > 1)
        is_decode  = (T_q == 1)
        mode = self.sparsity_mode

        do_rw = (
            (self.sliding_window is None)
            and (
                (mode == "both" and (is_prefill or is_decode))
                or (mode == "prefilling" and is_prefill)
                or (mode == "decoding" and is_decode)
            )
            and self.layer_idx > 1
        )

        if do_rw:
            block_size = int(getattr(self, "random_walk_block_size", 64) or 64)
            if is_prefill:
                attn_out_masked, attn_weights_masked, teacher_attention_probs = self._rw_prefill(
                    query_states=query_states, key_states=key_states, value_states=value_states,
                    attention_mask=attention_mask, past_key_values=past_key_values,
                    random_walk_states=random_walk_states, random_walk_cache=random_walk_cache,
                    block_size=block_size, attention_interface=attention_interface, **kwargs,
                )
            elif is_decode and block_size > 1:
                attn_out_masked, attn_weights_masked = self._rw_decode(
                    query_states=query_states, key_states=key_states, value_states=value_states,
                    attention_mask=attention_mask, past_key_values=past_key_values,
                    random_walk_cache=random_walk_cache,
                    block_size=block_size, attention_interface=attention_interface, **kwargs,
                )
            else:
                attn_out_masked, attn_weights_masked = attention_interface(
                    self, query_states, key_states, value_states,
                    attention_mask=attention_mask,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    scaling=self.scaling, **kwargs,
                )
        else:
            attn_out_masked, attn_weights_masked = attention_interface(
                self, query_states, key_states, value_states,
                attention_mask=attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=self.sliding_window,
                **kwargs,
            )
            if (
                is_prefill
                and mode == "decoding"
                and self.layer_idx > 1
                and self.sliding_window is None
            ):
                block_size = int(getattr(self, "random_walk_block_size", 64) or 64)
                rw_cache = self._get_random_walk_cache(past_key_values, random_walk_cache)
                if rw_cache is not None:
                    with torch.no_grad():
                        if self.random_walk_exact:
                            a_blk = self._exact_attention_probs_block(
                                query_states, key_states, attention_mask=None, block_size=block_size
                            )
                        else:
                            a_blk = self._hadamard_attention_probs_block(
                                query_states, key_states, attention_mask=None, block_size=block_size
                            )
                    layer_cache = rw_cache.get(self.layer_idx)
                    if layer_cache is None:
                        layer_cache = {}
                        rw_cache[self.layer_idx] = layer_cache
                    layer_cache["a_blk"] = a_blk.detach()
                    # Seed q_window from prefill context so the first decode step pools over W queries.
                    W = int(self.random_walk_query_window or block_size)
                    W_eff = min(W, query_states.size(2))
                    layer_cache["q_window"] = query_states[:, :, -W_eff:, :].detach()

        attn_out = attn_out_masked.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_out)

        extra = {
            "estimator_log_probs": estimator_log_probs,
            "teacher_attention_probs": teacher_attention_probs,
        }
        return attn_output, attn_weights_masked, extra


# ---------------------------------------------------------------------------
# Decoder Layer
# ---------------------------------------------------------------------------

class Qwen3DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.sparsity_mode = "both"

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        random_walk_states: Optional[dict] = None,
        random_walk_cache: Optional[dict] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, dict]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _, extra = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            random_walk_states=random_walk_states,
            random_walk_cache=random_walk_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, extra


# ---------------------------------------------------------------------------
# PreTrainedModel base
# ---------------------------------------------------------------------------

@auto_docstring
class Qwen3PreTrainedModel(PreTrainedModel):
    config: Qwen3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": Qwen3DecoderLayer,
        "attentions": Qwen3Attention,
    }


# ---------------------------------------------------------------------------
# Qwen3Model with SketchWalk
# ---------------------------------------------------------------------------

@auto_docstring
class Qwen3Model(Qwen3PreTrainedModel):
    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.sparsity_mode = "both"
        self.has_sliding_layers = (
            hasattr(config, "layer_types") and "sliding_attention" in config.layer_types
        )

        self.post_init()

    def _causal_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        input_embeds: torch.Tensor,
        past_key_values=None,
        sliding_window: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Build an explicit 4D causal additive mask (0 = attend, -inf = block).
        Always materializes the mask so SketchWalk's sdpa_attention_forward
        (which uses is_causal=False) still gets proper causal structure.
        """
        device = input_embeds.device
        dtype  = input_embeds.dtype
        neg_inf = torch.finfo(dtype).min
        B, T_q, _ = input_embeds.shape

        if attention_mask is not None:
            T_k = attention_mask.shape[1]
            past_len = max(T_k - T_q, 0)
        else:
            if past_key_values is not None:
                past_len = past_key_values.get_seq_length()
            else:
                past_len = 0
            T_k = past_len + T_q

        base = torch.zeros((1, 1, T_q, T_k), dtype=dtype, device=device)

        curr_k_len = max(min(T_q, T_k - past_len), 0)
        if curr_k_len > 0:
            tri_bool = torch.triu(
                torch.ones((T_q, curr_k_len), dtype=torch.bool, device=device), diagonal=1
            )
            curr_view = base[:, :, :, past_len : past_len + curr_k_len]
            base[:, :, :, past_len : past_len + curr_k_len] = curr_view.masked_fill(tri_bool, neg_inf)

        base = base.expand(B, 1, T_q, T_k).clone()

        if attention_mask is not None:
            key_pad = (attention_mask == 0).view(B, 1, 1, T_k)
            base = base.masked_fill(key_pad, neg_inf)

        if sliding_window is not None and sliding_window > 0:
            q_idx = torch.arange(T_q, device=device).view(T_q, 1) + past_len
            k_idx = torch.arange(T_k, device=device).view(1, T_k)
            sw_mask = k_idx < (q_idx - sliding_window + 1)
            base = base.masked_fill(sw_mask.unsqueeze(0).unsqueeze(0), neg_inf)

        return base.contiguous()

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, BaseModelOutputWithPast]:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # Build causal mask(s) — always materialize so SketchWalk (is_causal=False) gets proper causal structure.
        # create_causal_mask returns None for SDPA (expects is_causal=True internally), which breaks SketchWalk.
        sliding_win = getattr(self.config, "sliding_window", None)
        causal_mask_mapping = {
            "full_attention": self._causal_mask(attention_mask, inputs_embeds, past_key_values, sliding_window=None),
        }
        if self.has_sliding_layers:
            causal_mask_mapping["sliding_attention"] = self._causal_mask(
                attention_mask, inputs_embeds, past_key_values, sliding_window=sliding_win
            )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        loss_dtype  = hidden_states.dtype
        loss_device = hidden_states.device
        total_selector_loss = torch.tensor(0.0, device=loss_device, dtype=loss_dtype)
        num_contrib = 0

        runtime_random_walk_states = {} if past_key_values is None else None
        runtime_random_walk_cache  = {} if past_key_values is None else None

        layer_types = getattr(self.config, "layer_types", ["full_attention"] * self.config.num_hidden_layers)

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            layer_mask = causal_mask_mapping[layer_types[i]]
            hidden_states, extra = decoder_layer(
                hidden_states,
                attention_mask=layer_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                random_walk_states=runtime_random_walk_states,
                random_walk_cache=runtime_random_walk_cache,
                **kwargs,
            )

            log_p   = extra.get("estimator_log_probs", None)
            pt_attn = extra.get("teacher_attention_probs", None)
            if log_p is not None and pt_attn is not None:
                log_pt = (pt_attn.clamp_min(1e-12)).log()
                layer_loss = (pt_attn * (log_pt - log_p)).sum(dim=-1).mean()
                if layer_loss.dtype != loss_dtype:
                    layer_loss = layer_loss.to(loss_dtype)
                if layer_loss.device != loss_device:
                    layer_loss = layer_loss.to(loss_device)
                total_selector_loss = total_selector_loss + layer_loss
                num_contrib += 1

        if num_contrib > 0:
            total_selector_loss = total_selector_loss / num_contrib
        else:
            total_selector_loss = torch.zeros(1, device=loss_device, dtype=loss_dtype).squeeze(0)

        hidden_states = self.norm(hidden_states)
        return total_selector_loss, BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


# ---------------------------------------------------------------------------
# Qwen3ForCausalLM with SketchWalk
# ---------------------------------------------------------------------------

@auto_docstring
class Qwen3ForCausalLM(Qwen3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def set_sparsity_mode(self, mode: str):
        assert mode in {"prefilling", "decoding", "both"}
        self.config.sparsity_mode = mode
        for mod in self.modules():
            if hasattr(mod, "sparsity_mode"):
                mod.sparsity_mode = mode

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        if use_cache is None:
            use_cache = self.config.use_cache

        loss, outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        if labels is not None:
            loss = loss + self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class Qwen3ForSequenceClassification(GenericForSequenceClassification, Qwen3PreTrainedModel):
    pass


class Qwen3ForTokenClassification(GenericForTokenClassification, Qwen3PreTrainedModel):
    pass


class Qwen3ForQuestionAnswering(GenericForQuestionAnswering, Qwen3PreTrainedModel):
    base_model_prefix = "transformer"


__all__ = [
    "Qwen3ForCausalLM",
    "Qwen3ForQuestionAnswering",
    "Qwen3PreTrainedModel",
    "Qwen3Model",
    "Qwen3ForSequenceClassification",
    "Qwen3ForTokenClassification",
]
