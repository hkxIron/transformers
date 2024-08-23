#from ...models.llama import LlamaModel
# 直接从根目录引入
"""
注意：debug本项目时需要将${PROJECT_ROOT}/src添加到sys.path中，即sys.path.insert(0, ${PROJECT_ROOT}/src)
或者将${PROJECT_ROOT}/src设置为source root
"""
#import sys;sys.path.insert(0, "/home/hkx/data/work/open/transformers/src")

import sys
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama import LlamaModel, LlamaConfig
import torch

def show_paths():
    import sys
    import os
    print("sys path", sys.path)
    print("cur work path:", os.getcwd())


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

if __name__ == "__main__":
    print("args:", sys.argv)
    show_paths()
    test_llama_forward()