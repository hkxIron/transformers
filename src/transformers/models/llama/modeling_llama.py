# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# "from ..."是代表从transformers这个根目录下开始引用
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache, StaticCache
from ...modeling_attn_mask_utils import AttentionMaskConverter
from ...modeling_flash_attention_utils import _flash_attention_forward
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from .configuration_llama import LlamaConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"
print("hkx llama for debug")


def _prepare_4d_causal_attention_mask_with_cache_position(
    attention_mask: torch.Tensor,
    sequence_length: int,
    target_length: int, # target_length=past_seen_token + seq_len+1
    dtype: torch.dtype,
    device: torch.device,
    min_dtype: float,
    cache_position: torch.Tensor,
    batch_size: int,
):
    """
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

    Args:
        attention_mask (`torch.Tensor`):
            A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
        sequence_length (`int`):
            The sequence length being processed.
        target_length (`int`):
            The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
        dtype (`torch.dtype`):
            The dtype to use for the 4D attention mask.
        device (`torch.device`):
            The device to plcae the 4D attention mask on.
        min_dtype (`float`):
            The minimum value representable with the dtype `dtype`.
        cache_position (`torch.Tensor`):
            Indices depicting the position of the input sequence tokens in the sequence.
        batch_size (`torch.Tensor`):
            Batch size.
    """
    if attention_mask is not None and attention_mask.dim() == 4:
        # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
        causal_mask = attention_mask
    else:
        # attention_mask:[batch, sequence_length], 值为1的地方是有效token，为0的地方是padding_id
        # target_length=past_seen_token + seq_len+1
        # casual_mask:[sequence_length, target_length], casual_mask=0的地方是需要参与attention的地方
        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device) # 注意：填充的为最小值
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1) # 只保留上三角(upper)矩阵原始的最小值, 下三角全置0, diagonal=1使用主对角线上面第1条对角线

        # casual_mask: [sequence_length, target_length=seq_len+1], 将超过cache_position的位置全mask为0
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        # casual_mask: [sequence_length, target_length=seq_len+1]
        # -> [batch, 1, sequence_length, target_length=seq_len+1]
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1) # expand会在该维度复制
        if attention_mask is not None: # attention_mask:[batch, sequence_length]
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            # attention_mask: (batch_size, key_value_length)
            mask_length = attention_mask.shape[-1]

            """
            # causal_mask:值为0的地方为需要attention的地方,值为-inf为不需要计算的地方
            # attention_mask:值为0的地方为不需要计算的地方，值为1的地方为需要计算的地方
            # 两者相加有如下几种情况：
            # casual_mask(0)+attn_mask(0)=0, 不参与计算， 即为padding_mask=0
            # casual_mask(0)+attn_mask(1)=1, 需要参与计算, 即为padding_mask=0
            # casual_mask(-inf)+attn_mask(0)=-inf, 不参与计算，即为padding_mask=0
            # casual_mask(-inf)+attn_mask(1)=-inf, 不参与计算，即为padding_mask=0
            """
            # causal_mask: (batch_size, 1, sequence_length, target_length=seq_len+1)
            # attention_mask: (batch_size, key_value_length)
            # => attention_mask: (batch_size, head_num=1, query_length=1, key_value_length)
            # padding_mask: (batch_size, head_num=1, seq_len, target_length=seq_len+1)
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :] # attention_mask:[batch,1,1, sequence_length]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length]\
                .masked_fill(padding_mask, value=min_dtype) # 将padding位置mask为-inf
    # casual_mask: [batch, 1, sequence_length, target_length=seq_len+1]
    return causal_mask


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    """
        rms_norm(x) = alpha * x/sqrt(mean(x^2))
    """
    def forward(self, hidden_states):
        # hidden_states:[batch, sequence_len, hidden_size]
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        # var(x) = x^2
        # 在最后hidden_size维度上求平均, 注意，rms_norm并没有去中心化操作
        # var:[batch, sequence_len, 1]
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        # rsqrt(x)=1/sqrt(x^2), 是对每个元素取平方根后再取倒数
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
        config: Optional[LlamaConfig] = None,
    ):
        super().__init__()
        # TODO (joao): remove the `if` below, only used for BC
        self.rope_kwargs = {}
        if config is None:
            logger.warning_once(
                "`LlamaRotaryEmbedding` can now be fully parameterized by passing the model config through the "
                "`config` argument. All other arguments will be removed in v4.45"
            )
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.rope_type = rope_type # llama的rope_type是'default'
            self.max_seq_len_cached = max_position_embeddings
            self.original_max_seq_len = max_position_embeddings
        else:
            # BC: "rope_type" was originally "type"
            if config.rope_scaling is not None:
                self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
            else:
                self.rope_type = "default"
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        # inv_freq shape: [dim//2], 其值为: 1 / 10000 ^ (even_dim_index / dim)
        # inv_freq = 1.0 / [base ** (range(0, dim, 2)/dim)], 为一个向量
        # inv_freq就是ROPE中的theta角度
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, **self.rope_kwargs)
        self.register_buffer("inv_freq", inv_freq, persistent=False) # buffer就是常量，不会计算梯度
        self.original_inv_freq = self.inv_freq # 取的buffer值

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad() # 注意：这里是no_grad
    def forward(self, x, position_ids):
        # 其实此处只用了x的device_type

        # x: [batch_size, num_key_value_heads, seq_len, head_dim], x为query或key
        # position_ids: [batch_size, sequence_length]

        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)


        # Core RoPE block
        # position_ids: [batch, sequence_length]
        # inv_freq shape: [dim/2], 其值为: 1 / 10000 ^ (even_dim_index / dim) ,even偶数
        # eg: [1/10000^(0/dim), 1/10000^(2/dim), 1/10000^(4/dim), ... , 1/10000^((dim/2-2)/dim), 1/10000^((dim/2-1)/dim)]
        # => [1, dim/2, 1],其中dim=head_dim
        # inv_freq_expand:[batch, dim/2, 1], expand只适用于复制维度为1的
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)

        # position_ids: [batch_size, sequence_length]
        # position_ids_expanded: [batch, 1, sequence_length]
        position_ids_expanded = position_ids[:, None, :].float()

        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            """
            inv_freq @ position_ids的物理意义是对batch的每条样本都将inv_freq向量复制一份
            
            freqs: 在最后一维head_dim维
            [batch, seq_len, head_dim/2]

            freqs示例数据： 
            [batch, seq_len=m, [m*theta0,
                                m*theta1,
                                m*theta2,
                                ... 
                                m*theta(head_dim//2-2),
                                m*theta(head_dim//2-1)
                                ]]
            """
            
            # inv_freq_expand, 即为其中的 theta(dim_index):[batch, dim/2, 1], 其值为: 1 / 10000 ^ (even_dim_index / dim)
            # position_ids_expanded: [batch, 1, seq_len],
            # => 
            # freqs: [batch, dim/2, seq_len]
            # transpose转置 => [batch, seq_len, dim/2]

            # freqs = position / (10000^(2*dim_idx/dim)),
            # position_embed(m) = e^(j*m*theta) = e^(j*m/[10000^(2*dim_idx/dim)]),其中j为虚数单位, m为token位置
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2) # 这里为batch matrix multiplication

            # 前半部分与后半部分拼接
            # freqs: [batch, seq_len, dim/2]
            # =>
            # emb:  [batch, seq_len, dim]
            emb = torch.cat((freqs, freqs), dim=-1)

            """
            cos: 在最后一维dim维
            [batch, seq_len, dim]
            示例数据： 
            [batch, seq_len==m, [
                                # 前半部分
                                cos(m*theta0),
                                cos(m*theta1),
                                cos(m*theta2),
                                ...
                                cos(m*theta(head_dim//2-2)),
                                cos(m*theta(head_dim//2-1)),
                                ---------- 
                                # 后半部分
                                cos(m*theta0),
                                cos(m*theta1),
                                cos(m*theta2),
                                ...
                                cos(m*theta(head_dim//2-2)),
                                cos(m*theta(head_dim//2-1))
                                ]]
                                
            sin: 在最后一维dim维
            [batch, seq_len==m, dim]
            示例数据： 
            [batch, seq_len=m, [
                                # 前半部分
                                sin(m*theta0),
                                sin(m*theta1),
                                sin(m*theta2),
                                ...
                                sin(m*theta(head_dim//2-2)),
                                sin(m*theta(head_dim//2-1)),
                                ---------- 
                                # 后半部分
                                sin(m*theta0),
                                sin(m*theta1),
                                sin(m*theta2),
                                ...
                                sin(m*theta(head_dim//2-2)),
                                sin(m*theta(head_dim//2-1)),
                                ]]
            """
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling # attention_scaling默认为1
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, *args, **kwargs):
        logger.warning_once(
            "`LlamaLinearScalingRotaryEmbedding` is deprecated an will be removed in v4.45. Please use "
            "`LlamaRotaryEmbedding`, which now also does linear scaling (simply pass the model config to __init__)."
        )
        kwargs["rope_type"] = "linear"
        super().__init__(*args, **kwargs)


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, *args, **kwargs):
        logger.warning_once(
            "`LlamaDynamicNTKScalingRotaryEmbedding` is deprecated an will be removed in v4.45. Please use "
            "`LlamaRotaryEmbedding`, which now also does dynamic ntk scaling (simply pass the model config to "
            "__init__)."
        )
        kwargs["rope_type"] = "dynamic"
        super().__init__(*args, **kwargs)


def rotate_half(x:torch.Tensor):
    """
    Rotates half the hidden dims of the input.
    将x的前后半部分进行交换， 然后将前半部分取反 
    """
    # x: [batch, num_head, seq_len, head_dim]
    x1 = x[..., : x.shape[-1] // 2] # 最后一维取前半部分
    x2 = x[..., x.shape[-1] // 2 :] # 最后一维取后半部分
    """
    x.shape: [batch, num_head, seq_len, head_dim]
    [batch=0, num_head=0, seq_len=0, head_dim= [ -x(dim/2+1),
                                                 -x(dim/2+2),
                                                 -x(dim/2+3),
                                                  ...,
                                                  -x(dim),
                                                  ----
                                                  x0,
                                                  x1,
                                                  x2,
                                                  ...,
                                                  x(dim/2)
                                                ]
    ]    
    """
    return torch.cat((-x2, x1), dim=-1) # 注意：这里将x2取反了


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    对query,key应用rope,因为它们要进行内积
    q: [batch, num_head, seq_len, head_dim]
    k: [batch, num_key_value_heads, seq_len, head_dim]

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """

    """
    # 每个hidden_dim上的单位旋转角度为：theta(dim_index):[batch, dim/2, 1], 其值为: 1 / 10000 ^ (even_dim_index / dim)

    cos: 在最后一维dim维
    [batch, seq_len, dim]
    示例数据： 
    [batch, seq_len=m, [
                        # 前半部分
                        cos(m*theta0),
                        cos(m*theta1),
                        cos(m*theta2),
                        ...
                        cos(m*theta(head_dim//2-2)),
                        cos(m*theta(head_dim//2-1)),
                        ---------- 
                        # 后半部分
                        cos(m*theta0),
                        cos(m*theta1),
                        cos(m*theta2),
                        ...
                        cos(m*theta(head_dim//2-2)),
                        cos(m*theta(head_dim//2-1))
                        ]]
    """
    # cos, sin: [batch, seq_len, head_dim]
    # =>        [batch, num_head=1, seq_len, head_dim]
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    """
    q:
    [batch==0, num_head==0, seq_len==0, head_dim= [ 
                                                 q0,
                                                 q1,
                                                 q2,
                                                 ...,
                                                 q(dim/2)
                                                 ----------
                                                 q(dim/2+1),
                                                 q(dim/2+2),
                                                 q(dim/2+3),
                                                  ...,
                                                 q(dim-1)
                                                ]
     
    cos: [batch, num_head=1, seq_len, dim]
    示例数据： 
    [batch, num_head=1, seq_len==m,dim=[
                        cos(m*theta0),
                        cos(m*theta1),
                        cos(m*theta2),
                        ...
                        cos(m*theta(head_dim//2-2)),
                        cos(m*theta(head_dim//2-1)),
                        ---------- 注意：后半部分与前半部分相同
                        cos(m*theta0),
                        cos(m*theta1),
                        cos(m*theta2),
                        ...
                        cos(m*theta(head_dim//2-2)),
                        cos(m*theta(head_dim//2-1))
                        ]]
    ]    
    
    rotate_half(q): 将q的前后半部分进行交换， 然后将前半部分取反 
    [batch==0, num_head==0, seq_len==0, head_dim=[-q(dim/2+1),
                                                 -q(dim/2+2),
                                                 -q(dim/2+3),
                                                  ...,
                                                  -q(dim-1),
                                                  ----
                                                  q0,
                                                  q1,
                                                  q2,
                                                  ...,
                                                  q(dim/2)
                                                ]
    
    sin: [batch,num_head=1, seq_len, dim]
    示例数据： 
    [batch, num_head=1, seq_len==m,dim=[
                        sin(m*theta0),
                        sin(m*theta1),
                        sin(m*theta2),
                        ...
                        sin(m*theta(head_dim//2-2)),
                        sin(m*theta(head_dim//2-1)),
                        ---------- 注意：后半部分与前半部分相同
                        sin(m*theta0),
                        sin(m*theta1),
                        sin(m*theta2),
                        ...
                        sin(m*theta(head_dim//2-2)),
                        sin(m*theta(head_dim//2-1))
                        ]]


    ====>                    
    正交旋转矩阵R(x):
    [
      [cos(x), -sin(x)],
      [sin(x),  cos(x)]
    ]

    由q_embed = (q * cos) + (rotate_half(q) * sin)可得如下矩阵，
    q_embed:
    [batch==0, num_head==0, seq_len==0, head_dim= [ 
                                                     q0*cos(m*theta0)-q(dim/2+1)*sin(m*theta0), # [q0, q(dim/2+1)]作为一个复向量，然后对此复向量旋转m*theta0角度
                                                     q1*cos(m*theta1)-q(dim/2+2)*sin(m*theta1),
                                                     q2*cos(m*theta2)-q(dim/2+3)*sin(m*theta2),
                                                     ...,
                                                     q(dim/2)*cos(m*theta(dim/2))-q(dim-1)*sin(m*theta(dim/2)),
                                                     ----------
                                                     q(dim/2+1)*cos(m*theta0)+q0*sin(m*theta0), # [q0, q(dim/2+1)]作为一个复向量，旋转角度theta0
                                                     q(dim/2+2)*cos(m*theta1)+q1*sin(m*theta1),
                                                     q(dim/2+3)*cos(m*theta2)+q2*sin(m*theta2),
                                                     ...,
                                                     q(dim-1)*cos(m*theta(dim/2))+q(dim/2)*sin(m*theta(dim/2))
                                                    ]
     
    注意：这里q_embed与RoFormer中的RoPE构造公式不同, 即RoFormer中是将相邻位置(q0,q1)作为复数的实部与虚部，而huggingface中llama中是将(qi,q(i+d/2))分别作为复数的实部与虚部,
    meta实现的llama则与原RoFormer中的Rope保持一致
    
    hf transformers中的llama是构造cos,sin矩阵与q相乘，实现时是将[qi,q(i+dim/2)]作为复数的虚部与实部
    facebook中的llama是构造复数e(i*m*theta),直接与q相乘，实现时与RoFormer中的ROPE一致，是将相邻元素[qi,q(i+1)]作为复数的虚部与实部.
    RoFormer rope: https://github.com/huggingface/transformers/blob/8e164c5400b7b413c7b8fb32e35132001effc970/src/transformers/models/roformer/modeling_roformer.py#L328-L331
    
    1.因此两种格式间需要转换，对于meta官方的模型转换成hf格式时，需要将meta interleaved-style ( GPT-NeoX style RoPE) 转换为hf two_half_style (GPT-J style RoPE)
    
    transformers/src/transformers/models/llama/convert_llama_weights_to_hf.py, Lines 113 to 115 in e42587f
    转换代码见：
    # permute for sliced rotary 
    def permute(w, n_heads=n_heads, dim1=dim, dim2=dim): 
        return w.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2) 
    
    2. 为何hf不采用llama中的interleaved的格式呢？
     - 最重要的是因为许可证的原因licence
     - 其次, Eleuther's Rope 是等价的, 并且op算子更少(不需要复数算子), 因此效率更高
    
    详细讨论见： 
    https://github.com/huggingface/transformers/issues/25199
    
    
    theta值为: [batch, dim/2, 1] = 1 / 10000 ^ (even_dim_index / dim)
    
    原始论文RoFormer中的Rope
    = [q0, q1, q2, ..., q(d-2), q(d-1)] .* [cos(m*theta0), cos(m*theta0), cos(m*theta1),cos(m*theta1), ..., cos(m*theta(head_dim/2)),cos(m*theta(dim/2))]  .*代表逐元素相乘
    + [-q1,q0,-q3, ...,-q(d-1), q(d-2)] .* [sin(m*theta0), sin(m*theta0), sin(m*theta1),sin(m*theta1), ..., sin(m*theta(head_dim/2)),sin(m*theta(dim/2))]
    = [
       q0*cos(m*theta0)-q1*sin(m*theta0), # 即[q0,q1]作为复向量,然后对此复向量逆时针旋转m*theta0角度
       q1*cos(m*theta0)+q0*sin(m*theta0), # 即[q0,q1]作为复向量
       
       q2*cos(m*theta1)-q3*sin(m*theta1),
       q3*cost(m*theta1)+q2*sin(m*theta1),
       ...
       q(d-2)*cos(m*theta(dim/2))-q(d-1)*sin(m*theta(dim/2)), # [q(d-2), q(d-1)]作为复向量
       q(d-1)*cos(m*theta(dim/2))-q(d-2)*sin(m*theta(dim/2))
    ]
    """
    # q:  [batch, num_head, seq_len, head_dim]
    # k:  [batch, num_key_value_heads, seq_len, head_dim]
    # cos:[batch, num_head=1, seq_len, head_dim]
    # q_embed: [batch, num_head, seq_len, head_dim]
    # k_embed: [batch, num_head, seq_len, head_dim]

    #rotate_half: 将q的前后半部分进行交换， 然后将前半部分取反 
    q_embed = (q * cos) + (rotate_half(q) * sin) # GPT-NeoX style RoPE, 即half-style rope
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


"""
我已看过图，的确如此
SiLU 函数将输入值x乘以 sigmoid(x) 函数的输出，其效果是在正值上非饱和，
负值上平滑并接近于零。与 ReLU 函数类似，SiLU 函数也能够创建非线性决策边界，
但它允许一些信息（即使是负值）传递，而不是像 ReLU 那样将所有负值置为零，这种特性可以帮助减轻梯度消失问题。
"""
class LlamaMLP(nn.Module):
    def __init__(self, config:LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        # gate_proj:[hidden_size, intermediate_size]
        # 其weight为：[hidden_size, intermediate_size].T
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        # up_proj:[hidden_size, intermediate_size]
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        # down_proj:[intermediate_size, hidden_size]
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        # llama中用的是silu
        # silu(x) = x * sigmoid(x)
        self.act_fn = ACT2FN[config.hidden_act] # silu

    def forward(self, x:torch.Tensor):
        # x:[batch, seq_len, hidden_size]
        # 张量并行：tensor parallel
        if self.config.pretraining_tp > 1:
            # 将intermediate切成多份
            slice = self.intermediate_size // self.config.pretraining_tp
            # 权重按行拆分为pretraining_tp份，每份行数为slice
            # gate_proj.weight:[hidden_size, intermediate_size].T
            # gate_proj_slices:([slice, hidden_size],...), 个数为tp_num
            # up_proj.weight:[hidden_size, intermediate_size].T
            # up_proj_slices:([slice, hidden_size]...),个数为tp_num
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)

            # down_proj.weight:[intermediate_size, hidden_size].T
            # down_proj_slices:([hidden_size, slice], ...),共有tp_num个
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            # x:[batch, seq_len, hidden_size]
            # gate_proj_slices:([slice, hidden_size],...), 个数为tp_num, 相当于Meta-llama源码中的ColumnParallelLinear按列分块,对w按列分块
            # F.linear(x, w) = x*(w.T)
            # 矩阵分块相乘后再concat,每块内数据X*W:[batch, seq_len, slice], 共有tp_num个
            # concat后：[batch, seq_len, intermediate_size=slice*tp_num]
            gate_proj = torch.cat([F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)
            # up_proj_slices:([slice, hidden_size]...),个数为tp_num
            # 矩阵分块后concat后：[batch, seq_len, intermediate_size=slice*tp_num]
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            # gate_proj：[batch, seq_len, intermediate_size=slice*tp_num]
            # up_proj:[batch, seq_len, intermediate_size=slice*tp_num]
            # intermediate_states:([batch, seq_len, slice], ...),个数为:tp_num
            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            # tp_num*slice = intermediate_size
            # 矩阵分块相乘,每块内数据X*(W.T),共tp_num个:
            #   intermediate_state:[batch, seq_len, slice]
            #   down_proj_slices:[hidden_size, slice]
            # down_proj:[batch, seq_len, hidden_size], 共tp_num个
            down_proj = [F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)]
            # down_proj:[batch, seq_len, hidden_size], 将各张量结果相加, 得到最终矩阵相乘结果
            down_proj = sum(down_proj)
        else:
            # NOTE：llama中是gate后加激活函数，up里面没有
            # y = down( silu(gate(x)) * up(x) )
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        # down_proj:[batch, seq_len, hidden_size]
        return down_proj


def repeat_kv(hidden_states: torch.Tensor, repeats: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seq_len, head_dim) to (batch, num_attention_heads, seq_len, head_dim)
    """
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    if repeats == 1: # group大小为1,是MHA,无需复制
        return hidden_states
    # torch.repeat_interleave不能升维
    # None代表插入一维，作用等同于hidden_states.unsqueeze(dim=2)一样
    # hidden_states: [batch, num_key_value_heads,  seq_len, head_dim]
    # => [batch, num_key_value_heads, 1, seq_len, head_dim]
    # => [batch, num_key_value_heads, repeats, seq_len, head_dim]
    """
    expand()函数可以将张量广播到新的形状。 
    注意：只能对维度值为1的维度进行扩展，无需扩展的维度，维度值不变，对应位置可写上原始维度大小或直接写作-1；且扩展的Tensor不会分配新的内存，只是原来的基础上创建新的视图并返回，返回的张量内存是不连续的。
    类似于numpy中的broadcast_to函数的作用。如果希望张量内存连续，可以调用contiguous函数。
    """
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, repeats, seq_len, head_dim) # 复制多份
    return hidden_states.reshape(batch, num_key_value_heads * repeats, seq_len, head_dim)


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    # 标准的多头注意力

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx # 层的index
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size # hidden_size:4096
        self.num_heads = config.num_attention_heads # num_heads=32
        self.head_dim = self.hidden_size // self.num_heads # head_dim=128
        """
         若：hidden_size=4096, num_heads=32, head_dim=128

         1.MHA: 
            num_key_value_heads=num_attention_heads=32时，为原始的multi head attention(MHA)
            k_proj/v_proj: [hidden_size=4096, num_key_value_heads*head_dim=32*128=4096]
            q_proj: [hidden_size=4096, num_attention_heads*head_dim=32*128=4096]

         2.MQA: 
            num_key_value_heads=1时，为multi query attention(MQA), 即所有head的query共享一个key/value
            num_key_value_group_size = num_attention_heads // num_key_value_heads=32//1=32, 即全部32个head共享一个key/value
            k_proj/v_proj: [hidden_size=4096, num_key_value_heads*head_dim=1*128=128],注意：128<head_dim=4096
            q_proj: [hidden_size=4096, num_attention_heads*head_dim=32*128=4096]

         3. GQA:
            num_key_value_heads=16<num_attention_heads=32时，为group query attention(GQA), 每个group大小为 num_key_value_group_size

            若两个head的query共享一个key/value, 即num_key_value_group_size=2, 
            num_key_value_heads=num_attention_heads//num_key_value_heads =16 < num_attention_heads=32

            k_proj/v_proj: [hidden_size=4096, num_key_value_heads*head_dim=16*128=2048],注意：2048<head_dim=4096
            q_proj: [hidden_size=4096, num_attention_heads*head_dim=32*128=4096]

        
        """
        self.num_key_value_heads = config.num_key_value_heads # 有多少个key/value head,默认为num_attention_heads=32, 有多少个key/value head group

        # 假设num_head=32, num_key_value_heads=16, 则：num_key_value_group_size=2,即2个head为一组, 即一个组里有多少个key/value head
        # 总共只有16个不同的kv了，kv cache大小节省了一半内存
        self.num_key_value_group_size = self.num_heads // self.num_key_value_heads # 每个GQA中组的大小(group size),后面组大小为多少就将kv复制多少份
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        # 默认即为因果推断
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        # q_proj: [hidden_size, hidden_size=num_head*head_dim]
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        # 如果是GQA/MQA,那么num_key_value_heads< num_heads
        # 即key, value向量维度小于query
        # NOTE: k_proj, v_proj: [hidden_size, num_key_value_head*head_dim], GQA/MQA中 k_proj/v_proj: 其维度小于q_proj = hidden_size
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        # q_proj: [hidden_size, hidden_size]
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias) # 一般的没有bias

        # TODO (joao): remove in v4.45 (RoPE is computed in the model, not in the decoder layers)
        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None, # 困果注意力的下三角attention mask , 已叠加样本组成batch时指示非padding与padding部分, padding处mask=0, 非padding处mask=1
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[DynamicCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        batch_size, seq_len, hidden_size = hidden_states.size()
        if self.config.pretraining_tp > 1: # Tensor parallelism
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)
        else:
            """
            kv cache可能会影响最终的推理准确性，原因主要是影响了prefill阶段query的input shape,导致矩阵相乘时的精度误差
            
            1. 无kv cache时, 手动拼接input_ids, input_ids=hidden_states: [batch, seq_len, hidden_size]
            for _ in range(5):
                next_logits = model(input_ids)["logits"][:, -1:]
                next_token_id = torch.argmax(next_logits, dim=-1)
                # 没有kv cached的解码，需要手动将next_token_id与input_ids拼接起来
                input_ids = torch.cat([input_ids, next_token_id], dim=-1)
                print("shape of input_ids:", input_ids.shape) 
                
            2. 有kv cache时, 无需拼接input_ids, next_token_id = hidden_states: [batch, seq_len=1, hidden_size], 注意seq_len=1
                past_key_values = None  # past_key_values is the key-value cache, kv cache初始化为None,后面动态增加
                for _ in range(5):
                    # 每次只输入下一个token
                    next_logits, past_key_values = model(next_token_id, past_key_values=past_key_values, use_cache=True).to_tuple()
                    # next_logits:[batch, seq_len, vocab_size]
                    # -> [batch, last_token=1, vocab_size]
                    next_logits = next_logits[:, -1:]
                    next_token_id = torch.argmax(next_logits, dim=-1) 
            
            If you place a breakpoint inside the model, and see what happens with and without KV caches, you'll see:
            1. During prefill (parsing the input prompt), the KV caches and the hidden states are exactly the same,
                as the inputs contain the same values and shapes.
            2. When generating one token at a time, you will see a divergence happening in the hidden states and
                the QKV after operations like linear layers.  
                我感觉应该是因为padding mask + softmax 造成的精度误差,此时query_len=1,而不是所有已输入token的length
            see: https://github.com/huggingface/transformers/issues/25420
            """
            # NOTE:先要进行Wq,Wk,Wk的变换，再进行rope+ attention
            # hidden_state:[batch_size, seq_len, hidden_size], 注意：在推理的prefill阶段, hidden_state的shape:[batch_size, 1, hidden_size]
            # q_proj: [hidden_size, hidden_size=num_head*head_dim]
            # query_states: [batch_size, seq_len, hidden_size]
            query_states = self.q_proj(hidden_states) # 对于bfloat16或float16时，seq_len不同会因为矩阵计算精度导致最终的query_states不同,最终导致是否kv cache的结果不同
            # k_proj, v_proj: [hidden_size, num_key_value_head*head_dim], 对于MHA而言，hidden_size=num_key_value_head*head_dim
            # hidden_state:[batch_size, seq_len, hidden_size]
            # key_states, value_states: [batch_size, num_key_value_head*head_dim]
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        # query_states: [batch_size, seq_len, hidden_size=num_head*head_dim]
        # -> [batch_size, seq_len, num_head, head_dim]
        # -> [batch_size, num_head, seq_len, head_dim]
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # query_states, value_states: [batch_size, seq_len, num_key_value_heads*head_dim]
        # -> [batch_size, seq_len, num_key_value_heads, head_dim], MQA, GQA中num_key_value_heads< num_heads=32
        # -> [batch_size, num_key_value_heads, seq_len, head_dim]
        key_states = key_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            # 如果位置编码为空，使用rope计算位置编码
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            # position_ids: [batch_size, seq_len]
            # 每个hidden_dim上的单位旋转角度为：theta(dim_index):[batch, dim/2, 1], 其值为: 1 / 10000 ^ (even_dim_index / dim)
            # =>
            # cos, sin: [batch, seq_len, dim]
            cos, sin = self.rotary_emb.forward(value_states, position_ids)
        else:
            # cos:[batch, seq_len, dim]
            # sin:[batch, seq_len, dim]
            cos, sin = position_embeddings

        # NOTE:必须要在attention内积之前应用rope
        # query_states:[batch_size, num_head, seq_len, head_dim], 在使用kvcache的推理阶段的step_by_step阶段，seq_len=1
        # key_states: [batch_size, num_key_value_heads, seq_len, head_dim]
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin) # 对 query,key应用rope,因为它们要进行内积

        if past_key_value is not None:
            # 如果启用了kv cache, 则更新KV cache
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            # 大模型推理阶段使用kvcache后,每次只有一个token_id输入, 因此此处的update会进行key,value与缓存的拼接
            # key_states:[batch, num_key_value_heads, 1, head_dim]
            # =>
            # key_states after kv cache:[batch, num_key_value_heads, seq_len, head_dim]
            key_states, value_states = past_key_value.update(key_states, value_states, layer_idx=self.layer_idx, cache_kwargs=cache_kwargs) # 注意：update会改变 key_states

        # 如果是GQA,那么num_key_value_heads< num_heads,即key, value向量维度小于query
        # 因此，GQA中需要进行key/value复制,以适合query的大小
        # NOTE: 所以，GQA并没有减少attention时的计算量，只是减少了推理阶段kv cache的内存大小!!!
        # NOTE: MQA, GQA中num_key_value_heads< num_heads=32
        # 在MQA中，query的大小为num_heads, key,value的大小为num_key_value_heads=1, 需要复制 num_key_value_group_size=num_heads//_num_key_value_heads 份
        # key_states, value_states: [batch_size, num_key_value_heads, seq_len, head_dim]
        # -> [batch_size, num_heads, seq_len, head_dim], 注意：这里最后复制为query相同的shape了
        key_states = repeat_kv(key_states, self.num_key_value_group_size) # group有多大，就对key/value复制多少份
        value_states = repeat_kv(value_states, self.num_key_value_group_size)

        # 所以可以看出，无论是MQA,GQA,它们最后都会进行kv复制，恢复成MHA的shape后再进行attention!!!
        # query_states, key_states: [batch_size, num_heads, seq_len, head_dim]
        # key_states: [batch_size, num_heads, seq_len, head_dim] 
        # attn_weights: [batch_size, num_heads, query_seq_len, key_seq_len]
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        """
        eg:
        样本0：I sell a car
        样本1:I was happy
        样本2：I think that I am happy
        样本3：I think that I was born in china.

        attention_mask # 一个batch有4个样本,第0个样本最长为4,第1个样本长度为3，最后一个样本(第3个样本)最长，mask均为1
        Out[10]:  
        tensor([[0, 0, 0,  0,  1, 1, 1, 1], # smaple[0].len=4
                [0, 0, 0,  0,  0, 1, 1, 1], # sample[1].len=3
                [0, 0, 1,  1,  1 , 1, 1, 1], # sample[2].len=6
                [1, 1, 1,  1,  1, 1, 1, 1]]) # sample[3].len=8

        causal_mask[1][0]>=0 # batch中第0个样本的attention_mask, 可以看到左边padding部分的mask均为False
        Out[15]: 
        # x横轴为当前参与attention的key token, y轴为当前的query token
        tensor([[False, False, False,  False, False, False, False, False],
                [False, False, False,  False, False, False, False, False],
                [False, False, False,  False, False, False, False, False],
                [False, False, False,  False, False, False, False, False],
                [False, False, False,  False, False, False, False, False],
                [False, False, False,  False, False,  True, False, False],
                [False, False, False,  False, False,  True,  True, False],
                [False, False, False,  False, False,  True,  True,  True]]) # 注意只有最后3列有值

        causal_mask[3][0]>=0 # batch中第3个样本的attention_mask，由于第3个样本最长，所以没有left_padding, 可以看到其mask为标准的下三角因果矩阵
        Out[16]: 
        tensor([[ True,  False, False, False,  False, False, False, False],
                [ True,  True,  False, False,  False, False, False, False],
                [ True,  True,  True,  False,  False, False, False, False],
                [ True,  True,  True,  True,   False, False, False, False],
                [ True,  True,  True,  True,   True,  False, False, False],
                [ True,  True,  True,  True,   True,  True,  False, False],
                [ True,  True,  True,  True,   True,  True,  True,  False],
                [ True,  True,  True,  True,   True,  True,  True,  True]])
        """
        # 在推理阶段时：attention_mask:在推理的prefill阶段为：[batch, num_heads=1, query_seq_len, key_seq_len], 
        # 在后面step by step decoding时 [batch, num_heads=1, query_seq_len=1, key_seq_len]
        # attention_mask: [batch, num_heads=1, query_seq_len, key_seq_len],  因果注意力的下三角attention mask, 每次都会传入, 但每个head的mask都一样
        if attention_mask is not None:  # no matter the length, we just slice it
            key_seq_len = key_states.shape[-2]
            causal_mask = attention_mask[:, :, :, : key_seq_len] # 此处的mask是0与-inf,0的地方参与attention, key_states.shape[-2]为seq_len
            attn_weights = attn_weights + causal_mask
            # 有的实现中是直接给0所在的位置一个非常大的负数，以至于在经过softmax后为会0分
            #attn_weights = attn_weights.masked_fill(casual_mask == 0, value=-1e9)

        # upcast attention to fp32,以防softmax计算exp溢出
        # attn_weights: [batch_size, num_heads, query_seq_len, key_seq_len]
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # 现在强制dropout,但attention_dropout默认为0
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training) # 注意：现在attn_weight之后会有一个dropout
        # attn_weights: [batch_size, num_heads, query_seq_len, key_seq_len]
        # value_states: [batch_size, num_heads, value_seq_len=key_seq_len, head_dim]
        # attn_output: [batch_size, num_heads, query_seq_len, head_dim]
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        # attn_output: [batch_size, num_heads, query_seq_len, head_dim]
        # -> [batch_size, query_seq_len, num_heads, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        # -> [batch_size, query_seq_len, num_heads*head_dim = hidden_size] , 记住：num_heads*head_dim = hidden_size
        attn_output = attn_output.reshape(batch_size, seq_len, -1)

        if self.config.pretraining_tp > 1:
            # 张量并行, 矩阵相乘中的列式并行(分块相乘后相加)
            """
            X = [ # 按列分块，假设tp_num=2,hidden_size=4,分为2块
                [x00 x01 | x02 x03],
                [x10 x11 | x12 x13],
                [x20 x21 | x22 x23],
                [x30 x31 | x32 x33],
            ]
            = [ XA, XB ] # XA: [4,2]

            W = [ # 按行分块，假设tp_num=2,hidden_size=4,分为2块
                [w00 w01 w02 w03],
                [w10 w11 w12 w13],
                ------------------
                [w20 w21 w22 w23],
                [w30 w31 w32 w33],
            ]
            = [
                [WA] shape:[2,4]
                ----
                [WB]
               ]

            X*W = sum_{k}(x(ik)*w(kj)) 
            张量并行：分块相乘后相加
            X = [
                [x00 x01 | x02 x03],
                [x10 x11 | x12 x13],
                [x20 x21 | x22 x23],
                [x30 x31 | x32 x33],
            ]
            = XA*WA + XB*WB # shape:[4,4]
            """
            # attn_output: [batch_size, query_seq_len, hidden_size], 在hidden_size维按列分块
            # =>
            # attn_output: ([batch_size, query_seq_len, hidden_size//tp_num],...), 共有tp_num个, 相当于Meta-llama源码中的ColumnParallelLinear按列分块
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            # o_proj: [hidden_size, hidden_size], 在第1维按行分块
            # =>
            # o_proj_slices: ([hidden_size, hidden_size//tp_num], ,,,) ,共有tp_num个
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            # 张量并行后，直接将结果相加即可
            # attn_output: [batch_size, query_seq_len, hidden_size]
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            # 记住：最后还有一次proj
            # q_proj: [hidden_size, hidden_size]
            # attn_output: [batch_size, query_seq_len, hidden_size]
            # =>
            # attn_output: [batch_size, query_seq_len, hidden_size]
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        # attn_output: [batch_size, query_seq_len, hidden_size]
        # attn_weights: [batch_size, num_heads, query_seq_len, key_seq_len]
        return attn_output, attn_weights, past_key_value


# class LlamaFlashAttention2(LlamaAttention):
#     """
#     Llama flash attention module. This module inherits from `LlamaAttention` as the weights of the module stays
#     untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
#     flash attention and deal with padding tokens in case the input contains any of them.
#     """
#
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#
#         # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
#         # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
#         # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
#         self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()
#
#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         attention_mask: Optional[torch.LongTensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_value: Optional[Cache] = None,
#         output_attentions: bool = False,
#         use_cache: bool = False,
#         cache_position: Optional[torch.LongTensor] = None,
#         position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
#     ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
#         if isinstance(past_key_value, StaticCache):
#             raise ValueError(
#                 "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
#                 "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
#             )
#
#         output_attentions = False
#
#         bsz, q_len, _ = hidden_states.size()
#
#         query_states = self.q_proj(hidden_states)
#         key_states = self.k_proj(hidden_states)
#         value_states = self.v_proj(hidden_states)
#
#         # Flash attention requires the input to have the shape
#         # batch_size x seq_length x head_dim x hidden_dim
#         # therefore we just need to keep the original shape
#         query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
#         key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
#         value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
#
#         if position_embeddings is None:
#             logger.warning_once(
#                 "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
#                 "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
#                 "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
#                 "removed and `position_embeddings` will be mandatory."
#             )
#             cos, sin = self.rotary_emb(value_states, position_ids)
#         else:
#             cos, sin = position_embeddings
#         query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
#
#         if past_key_value is not None:
#             # sin and cos are specific to RoPE models; cache_position needed for the static cache
#             cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
#             key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
#
#         # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
#         # to be able to avoid many of these transpose/reshape/view.
#         query_states = query_states.transpose(1, 2)
#         key_states = key_states.transpose(1, 2)
#         value_states = value_states.transpose(1, 2)
#
#         dropout_rate = self.attention_dropout if self.training else 0.0
#
#         # In PEFT, usually we cast the layer norms in float32 for training stability reasons
#         # therefore the input hidden states gets silently casted in float32. Hence, we need
#         # cast them back in the correct dtype just to be sure everything works as expected.
#         # This might slowdown training & inference so it is recommended to not cast the LayerNorms
#         # in fp32. (LlamaRMSNorm handles it correctly)
#
#         input_dtype = query_states.dtype
#         if input_dtype == torch.float32:
#             if torch.is_autocast_enabled():
#                 target_dtype = torch.get_autocast_gpu_dtype()
#             # Handle the case where the model is quantized
#             elif hasattr(self.config, "_pre_quantization_dtype"):
#                 target_dtype = self.config._pre_quantization_dtype
#             else:
#                 target_dtype = self.q_proj.weight.dtype
#
#             logger.warning_once(
#                 f"The input hidden states seems to be silently casted in float32, this might be related to"
#                 f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
#                 f" {target_dtype}."
#             )
#
#             query_states = query_states.to(target_dtype)
#             key_states = key_states.to(target_dtype)
#             value_states = value_states.to(target_dtype)
#
#         attn_output = _flash_attention_forward(
#             query_states,
#             key_states,
#             value_states,
#             attention_mask,
#             q_len,
#             position_ids=position_ids,
#             dropout=dropout_rate,
#             sliding_window=getattr(self, "sliding_window", None),
#             use_top_left_mask=self._flash_attn_uses_top_left_mask,
#             is_causal=self.is_causal,
#         )
#
#         attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
#         attn_output = self.o_proj(attn_output)
#
#         if not output_attentions:
#             attn_weights = None
#
#         return attn_output, attn_weights, past_key_value


# class LlamaSdpaAttention(LlamaAttention):
#     """
#     Llama attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
#     `LlamaAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
#     SDPA API.
#     """
#
#     # Adapted from LlamaAttention.forward
#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_value: Optional[Cache] = None,
#         output_attentions: bool = False,
#         use_cache: bool = False,
#         cache_position: Optional[torch.LongTensor] = None,
#         position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
#         **kwargs,
#     ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
#         if output_attentions:
#             # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
#             logger.warning_once(
#                 "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
#                 'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
#             )
#             return super().forward(
#                 hidden_states=hidden_states,
#                 attention_mask=attention_mask,
#                 position_ids=position_ids,
#                 past_key_value=past_key_value,
#                 output_attentions=output_attentions,
#                 use_cache=use_cache,
#                 cache_position=cache_position,
#                 position_embeddings=position_embeddings,
#             )
#
#         bsz, q_len, _ = hidden_states.size()
#
#         query_states = self.q_proj(hidden_states)
#         key_states = self.k_proj(hidden_states)
#         value_states = self.v_proj(hidden_states)
#
#         query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
#         key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
#         value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
#
#         if position_embeddings is None:
#             logger.warning_once(
#                 "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
#                 "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
#                 "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
#                 "removed and `position_embeddings` will be mandatory."
#             )
#             cos, sin = self.rotary_emb.forward(value_states, position_ids)
#         else:
#             cos, sin = position_embeddings
#
#         query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
#
#         if past_key_value is not None:
#             # sin and cos are specific to RoPE models; cache_position needed for the static cache
#             cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
#             key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
#
#         key_states = repeat_kv(key_states, self.num_key_value_group_size)
#         value_states = repeat_kv(value_states, self.num_key_value_group_size)
#
#         causal_mask = attention_mask
#         if attention_mask is not None:
#             causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]
#
#         # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
#         # Reference: https://github.com/pytorch/pytorch/issues/112577.
#         if query_states.device.type == "cuda" and causal_mask is not None:
#             query_states = query_states.contiguous()
#             key_states = key_states.contiguous()
#             value_states = value_states.contiguous()
#
#         # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
#         # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
#         is_causal = True if causal_mask is None and q_len > 1 else False
#
#         attn_output = torch.nn.functional.scaled_dot_product_attention(
#             query_states,
#             key_states,
#             value_states,
#             attn_mask=causal_mask,
#             dropout_p=self.attention_dropout if self.training else 0.0,
#             is_causal=is_causal,
#         )
#
#         attn_output = attn_output.transpose(1, 2).contiguous()
#         attn_output = attn_output.view(bsz, q_len, -1)
#
#         attn_output = self.o_proj(attn_output)
#
#         return attn_output, None, past_key_value


LLAMA_ATTENTION_CLASSES = {
    "eager": LlamaAttention,
    #"flash_attention_2": LlamaFlashAttention2,
    #"sdpa": LlamaSdpaAttention, # scaled_dot_product_attention
}


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        #self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)
        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor, # [batch, seq_len, embed_dim]
        attention_mask: Optional[torch.Tensor] = None, # casual_mask:[batch, 1, query_seq_len, key_seq_len]
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                casual attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        # hiddens_states: [batch, seq_len, hidden_size]
        residual = hidden_states

        # T1. Self Attention: [[pre_norm + self_attention + add]]
        # T1.1 layer norm
        hidden_states = self.input_layernorm(hidden_states) # [batch, seq_len, hidden_size]

        # T1.2 Casual Masked Self Attention(因果attention)
        hidden_states, self_attn_weights, present_key_value = self.self_attn.forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings, # rope旋转位置编码
            **kwargs,
        )

        # T1.3 Add
        # residual:[batch, seq_len, hidden_size]
        # hidden_states:[batch, seq_len, hidden_size]
        hidden_states = residual + hidden_states

        # T2. Fully Connected: [[pre_norm + mlp + add]]
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        # hidden_states:[batch, seq_len, hidden_size]
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)
        # 注意：outputs是个tuple,第0个元素是hidden_states,为啥不用OrderedDict?
        return outputs


LLAMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LlamaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig # 写在这里就是self.config_class
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors(key+value) of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*): 是否返回OrderDict
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily(反之) to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]
    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        # 注意：padding_id的embedding不会被更新
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        # 多层decoder layer
        self.layers:List[LlamaDecoderLayer] = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None, # [batch_size, sequence_length]
        attention_mask: Optional[torch.Tensor] = None,# [batch_size, sequence_length],同一batch内不同序列需要padding,1代表非padding的地方,0代表padding的地方,padding不参与attention
        position_ids: Optional[torch.LongTensor] = None, # [batch_size, sequence_length]
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None, # [sequence_length]
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions # False
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        ) # False
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None): # 异或, 两个值相异时才为true
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        # T1. embedding
        if inputs_embeds is None:
            # [batch, sequence_length, hidden_size]
            inputs_embeds = self.embed_tokens(input_ids)

        return_legacy_cache = False
        if (
            use_cache and not isinstance(past_key_values, Cache) and not self.training
        ):  # kept for BC (non `Cache` `past_key_values` inputs)
            # 注意：training时都是类似prefill的形式，并不需要kv cache
            return_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
                "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
            )

        """
        cache_position表示当前的input_ids中有多少个token,不论padding有多少个
        
        One important concept you need to know when writing your own generation loop, is cache_position. 
        In case you want to reuse an already filled Cache object by calling forward(), you have to pass in a valid cache_position which will indicate 
        the positions of inputs in the sequence. Note that cache_position is not affected by padding, and always adds one more position for each token. 
        For example, if key/value cache contains 10 tokens (no matter how many of it is a pad token), the cache position for the next token should be torch.tensor([10]). 
        """
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            input_seq_len = inputs_embeds.shape[1]
            # cache_position:[sequence_length]
            cache_position = torch.arange(past_seen_tokens, past_seen_tokens + input_seq_len, device=inputs_embeds.device)
        if position_ids is None:
            # position_ids:[1, sequence_length]
            position_ids = cache_position.unsqueeze(0)

        # 因果注意力mask
        # attention_mask: [batch, sequence_length], 为同一batch中不同样本的长效长度，为0的位置是padding
        # input_embeds:[batch, sequence_length, hidden_size]
        # =>
        # cache_position:[sequence_length]
        # casual_mask: [batch, head_num=1, sequence_length, target_length=seq_len+1]
        """
        eg:
        attention_mask # 一个batch有4个样本，最后一个样本(第3个样本)最长，mask均为1
        Out[10]:  
        tensor([[0, 0, 0,  ..., 1, 1, 1],
                [0, 0, 0,  ..., 1, 1, 1],
                [0, 0, 0,  ..., 1, 1, 1],
                [1, 1, 1,  ..., 1, 1, 1]])
        
        causal_mask[1][0]>=0 # batch中第0个样本的attention_mask, 可以看到左边padding部分的mask均为False
        Out[15]: 
        tensor([[False, False, False,  ..., False, False, False],
                [False, False, False,  ..., False, False, False],
                [False, False, False,  ..., False, False, False],
                ...,
                [False, False, False,  ...,  True, False, False],
                [False, False, False,  ...,  True,  True, False],
                [False, False, False,  ...,  True,  True,  True]])
                
        causal_mask[3][0]>=0 # batch中第3个样本的attention_mask，由于第3个样本最长，所以没有left_padding, 可以看到其mask为标准的下三角因果矩阵
        Out[16]: 
        tensor([[ True, False, False,  ..., False, False, False],
                [ True,  True, False,  ..., False, False, False],
                [ True,  True,  True,  ..., False, False, False],
                ...,
                [ True,  True,  True,  ...,  True, False, False],
                [ True,  True,  True,  ...,  True,  True, False],
                [ True,  True,  True,  ...,  True,  True,  True]])
            
        """
        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
        # inputs_embeds, hidden_states: [batch, sequence_length, hidden_size]
        hidden_states = inputs_embeds

        # 注意：旋转位置编码在各decoder层之间共享，只需要计算一次
        # create position embeddings to be shared across the decoder layers
        # position_ids: [batch_size, sequence_length]
        # position_embeddings: ([1, sequence_length, hidden_size], ),元素个数为batch
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        # T2: 多层 decoder
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,) # 将新的hidden_state追加

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states, # 上一层decoder的输出
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer.forward(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )
            # 注意：outputs是个tuple,第0个元素是hidden_states,第1个元素为attn_score
            hidden_states = layer_outputs[0]
            # 用这样的判断，为啥不用OrderedDict?
            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # T3. rms norm
        # hidden_states: [batch, sequence_length, hidden_size]
        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states, # [batch, sequence_length, hidden_size]
            past_key_values=next_cache, # key, value的shape均为(batch_size, num_heads, sequence_length, embed_size_per_head)
            hidden_states=all_hidden_states, # 共layer层，每层shape:(batch_size, sequence_length, hidden_size
            attentions=all_self_attns, # 共layer层，每层shape (batch_size, num_heads, sequence_length, sequence_length)
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor, # 注意对于batch_size>1, attention_mask一般不为空
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
        # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
        # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
        # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114

        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa( # sdpa不需要attention_mask
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            # target_length=past_seen_token+ seq_len+1
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # NOTE:将2D的attention_mask转换为4D的attention_mask
        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        # attention_mask:[batch, sequence_length]
        # target_length=past_seen_token + seq_len+1
        # cache_position:[sequence_len]
        # input_tensor:[batch, sequence_len, hidden_size]
        # casual_mask: [batch, 1, sequence_length, target_length=seq_len+1]
        causal_mask = _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            min_dtype=min_dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # 不会进入
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask


# LlamaForCasulalLM:只是在LlamaModel中加了一个LanguageModel head
class LlamaForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        # NOTE: 可以看出，LlamaForCasulalLM的lm_head的weight和embed的weight并没有共享的
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask, # 为在同一个batch中，不同sample的有效的input_ids所在的位置,1为有效id,0为padding位置
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # logits:[batch_size, sequence_length, vocab_size]
            # labels:[batch_size, sequence_length]
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            # shift_logits:[batch_size*sequence_length, vocab_size]
            # shift_labels:[batch_size*sequence_length]
            # 注意：此时的loss变成了token level的loss
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)

            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct.forward(shift_logits, shift_labels) # 默认是求batch*seq_len的loss的平均

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

                # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s  `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride during the decoding. Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case, `position_ids` is already contiguous but with varying stride which retriggers a capture.
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if inputs_embeds is not None:
                batch_size, sequence_length = inputs_embeds.shape
                device = inputs_embeds.device
            else:
                batch_size, sequence_length = input_ids.shape
                device = input_ids.device

            dtype = self.lm_head.weight.dtype
            min_dtype = torch.finfo(dtype).min

            attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_length(),
                dtype=dtype,
                device=device,
                min_dtype=min_dtype,
                cache_position=cache_position,
                batch_size=batch_size,
            )

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs


@add_start_docstrings(
    """
    The LLaMa Model transformer with a sequence classification head on top (linear layer).

    [`LlamaForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    LLAMA_START_DOCSTRING,
)
# LLamaForSequenceClassification只是在LlamaModel中添加了mlp用于分类
class LlamaForSequenceClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # hidden_states:[batch, seq_len, hidden_size]
        hidden_states = transformer_outputs[0]
        # logits:[batch, seq_len, num_labels]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        # logits:[batch, seq_len, num_labels]
        # pooled_logits:[batch, 1, num_labels], 只取最后一个时间步的进行求loss
        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 对pooled后的logits求loss
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss() # 单标签分类，如手写字体识别
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss() # 多标签，为每个logits分类值后接一个sigmoid,而不是接softmax单标签
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


@add_start_docstrings(
    """
The Llama Model transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    LLAMA_START_DOCSTRING,
)
# 在llamaModel上加了一个mlp头，用来预测答案在QUERY中的开始位置(start)与结束位置(end),即预测答案所在的SPAN
class LlamaForQuestionAnswering(LlamaPreTrainedModel):
    base_model_prefix = "transformer"

    # Copied from transformers.models.bloom.modeling_bloom.BloomForQuestionAnswering.__init__ with Bloom->Llama
    def __init__(self, config):
        super().__init__(config)
        self.transformer = LlamaModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.transformer.embed_tokens

    def set_input_embeddings(self, value):
        self.transformer.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        # logits:[batch, seq_len, 2]
        logits = self.qa_outputs(sequence_output)
        # start_logits:[batch, seq_len, 1]
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        # end_logits:[batch, seq_len, 1]
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1).to(start_logits.device)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1).to(end_logits.device)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            # 分别计算开始与结束位置的cross_entropy_loss,然后平均
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:] # +代表tuple相连
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    The Llama Model transformer with a token classification head on top (a linear layer on top of the hidden-states
    output) e.g. for Named-Entity-Recognition (NER) tasks.
    """,
    LLAMA_START_DOCSTRING,
)
# LlamaForTokenClassification只是在LlamaModel中添加了mlp用于序列标注，比如常见的NER, slot_filling任务,
# 对于每个位置都需要输出一个分类, 与 LlamaForSequenceClassification的区别在于前者是对于每个token都输出一个num_label的logits
# 而后者只是输出一个pooled的logits, 但基本没人在LLM中使用序列标注而是都使用基于bert的模型做序列标注，因为LLM都是decoder-only结构，
# 而bert是双向MLM，可以感知token前后的token
class LlamaForTokenClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        if getattr(config, "classifier_dropout", None) is not None:
            classifier_dropout = config.classifier_dropout
        elif getattr(config, "hidden_dropout", None) is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.score = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # seq_output:[batch, seq_len, hidden_size]
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        # logits:[batch, seq_len, num_labels]
        logits = self.score(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

"""
注释：
以llama-7B为例，属于decoder-only models的范畴，只有decoder层。
核心是包含了32层的decoder，每个decoder包含一个llamaAttention和llamaMLP

LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(32000, 4096, padding_idx=31999)
    (layers): ModuleList(
      (0-31): 32 x LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
          (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
          (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
          (act_fn): SiLUActivation()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
    )
    (norm): LlamaRMSNorm()
  )
  (lm_head): Linear(in_features=4096, out_features=32000, bias=False)
)
"""