#from ...models.llama import LlamaModel
# 直接从根目录引入
"""
注意：debug本项目时需要将${PROJECT_ROOT}/src添加到sys.path中，即sys.path.insert(0, ${PROJECT_ROOT}/src)
或者将${PROJECT_ROOT}/src设置为source root
"""
#import sys;sys.path.insert(0, "/home/hkx/data/work/open/transformers/src")

import sys
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama import (LlamaModel, LlamaConfig)
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb
import torch
from torch.nn import *
from typing import *

def setup_seed(seed):
    import numpy as np
    import random

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.backends.cudnn.deterministic = True # 会显著降低速度，将这个 flag 置为True的话，每次返回的卷积算法将是确定的，即默认算法，个人理解，如果没有用到drop out 等模型内部的随机trick，这一项可以去除

def show_paths():
    import sys
    import os
    print("sys path", sys.path)
    print("cur work path:", os.getcwd())

def my_decorator(func:Callable):
    def my_wrapper(*args, **kwargs):
        #start_time = time.time()
        print("="*30+f" {func.__name__} begin "+"="*30)
        result = func(*args, **kwargs)
        #end_time = time.time()
        print(f"{func.__name__} end\n\n\n")
        return result
    return my_wrapper

@my_decorator
def test_llama_forward():
    scale = 2
    llama_config = LlamaConfig(
        vocab_size=32000,
        hidden_size=4096//scale,
        intermediate_size=11008//scale,
        num_hidden_layers=32//scale,
        num_attention_heads=32//scale,
        attn_implementation='eager' # 默认的attention方式, hkx加入
    )
    print(f"config:\n{llama_config}")
    llama_model = LlamaModel(config=llama_config)
    batch,seq_len=2, 3 
    input_ids = torch.randint(low=0, high= llama_config.vocab_size, size=(batch, seq_len))
    model_out: BaseModelOutputWithPast = llama_model(input_ids)
    print("model_out shape:", model_out.last_hidden_state.shape)# torch.Size([2, 3, 2048])
    """
    LlamaModel(
  (embed_tokens): Embedding(32000, 2048)
  (layers): ModuleList(
    (0-15): 16 x LlamaDecoderLayer(
      (self_attn): LlamaAttention(
        (q_proj): Linear(in_features=2048, out_features=2048, bias=False)
        (k_proj): Linear(in_features=2048, out_features=2048, bias=False)
        (v_proj): Linear(in_features=2048, out_features=2048, bias=False)
        (o_proj): Linear(in_features=2048, out_features=2048, bias=False)
        (rotary_emb): LlamaRotaryEmbedding()
      )
      (mlp): LlamaMLP(
        (gate_proj): Linear(in_features=2048, out_features=5504, bias=False)
        (up_proj): Linear(in_features=2048, out_features=5504, bias=False)
        (down_proj): Linear(in_features=5504, out_features=2048, bias=False)
        (act_fn): SiLUActivation()
      )
      (input_layernorm): LlamaRMSNorm()
      (post_attention_layernorm): LlamaRMSNorm()
    )
  )
  (norm): LlamaRMSNorm()
)
    """
    print(llama_model)

@my_decorator
def test_rope():
    batch=2
    head_num=3

    head_dim=6
    max_seq_len=4
    base=10000
    rotary_emb = LlamaRotaryEmbedding(dim=head_dim, max_position_embeddings=max_seq_len, base=base, rope_type="default")

    setup_seed(3407)
    # [batch, head_num, seq_len, head_dim]
    query = torch.randn([batch, max_seq_len, head_num, head_dim]).transpose(1,2)
    key = query.detach().clone()
    value = query.detach().clone()

    position_ids_short = (torch.arange(max_seq_len, dtype=torch.long)
                          .unsqueeze(dim=0)
                          .expand([batch, max_seq_len]))
    print(position_ids_short.shape)

    print("query[0][0]:", query.transpose(1,2)[0][0]) # 与原始meta llama中的view保持一致
    print("query[0][1]:", query.transpose(1,2)[0][1]) # 与原始meta llama中的view保持一致
    print("query[1][1]:", query.transpose(1,2)[1][1]) # 与原始meta llama中的view保持一致
    cos, sin = rotary_emb.forward(value, position_ids_short)
    # q: [batch, num_head, seq_len, head_dim]
    query_rope, key_rope = apply_rotary_pos_emb(query, key, cos, sin)
    #print("query_rope shape:", query_rope.shape)
    print("query_rope[0][0]:\n", query_rope.transpose(1,2)[0][0])
    print("query_rope[0][1]:\n", query_rope.transpose(1,2)[0][1])
    print("query_rope[1][1]:\n", query_rope.transpose(1,2)[1][1])


from itertools import *
import torch.nn.functional as F

def _init_weights(module:Module):
    # 这里面的init都没有用到xarviar或kaiming init?
    if isinstance(module, Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02) # weight不能全为0
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02) # embedding不能全为0

class MyModel(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layers = ModuleList([Linear(fin, fout) for (fin, fout) in pairwise([10, 20, 10])])
        self.apply(_init_weights)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

@my_decorator
def test_grad_norm():
    model = MyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
    print(model)

    N = 20
    batch=5
    dim=10

    x, y = torch.randn((N, dim)), torch.randint(low=0, high=10, size=(N,))
    iter = 0
    for epoch in range(100):
        for ind in range(N // batch - 1):
            optimizer.zero_grad()
            logits = model(x[ind*batch:(ind+1)*batch,:])
            loss = F.cross_entropy(logits, y[ind*batch:(ind+1)*batch])
            loss.backward()

            #print([(x[0], x[1].shape) for x in model.named_parameters()])
            #print([(x[0], x[1].grad) for x in model.named_parameters()])
            total_grad = utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2.0)
            print(f"epoch:{epoch} iter:{iter} loss:{loss.item()} total_grad:{total_grad}")

            optimizer.step()
            iter+=1

if __name__ == "__main__":
    print("args:", sys.argv)
    show_paths()
    test_grad_norm()

    if False:
        test_rope()
        test_llama_forward()