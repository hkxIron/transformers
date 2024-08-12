# 直接从根目录引入
#from ...models.llama import LlamaModel
from transformers.models.llama import LlamaModel, LlamaConfig
import torch
import sys

def test_llama_forward():
    scale = 2
    llama_config = LlamaConfig(
        vocab_size=32000,
        hidden_size=4096//scale,
        intermediate_size=11008//scale,
        num_hidden_layers=32//scale,
        num_attention_heads=32//scale,
    )
    print(f"config:\n{llama_config}")
    llama_model = LlamaModel(config=llama_config)
    batch,seq_len=2, 3 
    input_ids = torch.randint(low=0, high= llama_config.vocab_size, size=(batch, seq_len))
    res = llama_model(input_ids)
    print("res shape:", type(res))
    print(res.last_hidden_state.shape)# torch.Size([2, 3, 2048])
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
    test_llama_forward()