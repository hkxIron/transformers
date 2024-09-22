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
from hkx_test_llama_units import *
from transformers import LlamaTokenizer, LlamaForCausalLM

def test_llama_tokenzier():
    model_path = '/home/hkx/data/work/hf_data_and_model/models/TinyStories-LLaMA2-20M-256h-4l-GQA'
    tokenizer:LlamaTokenizer = LlamaTokenizer.from_pretrained(model_path, use_fast=False)

    print(f"pad_id:{tokenizer.pad_token_id}")
    print(f"bos_id:{tokenizer.bos_token_id}")
    print(f"eos_id:{tokenizer.eos_token_id}")
    print(f"test token to ids no tensor:{tokenizer('test token to ids')}")
    print(f"test token to ids with tensor:{tokenizer('test token to ids', return_tensors='pt')}")

    token_ids = torch.Tensor([1, 1, 15043, 29892, 14332, 29892, 29871, 30662, 30675, 29871, 2, 29889])
    text = tokenizer.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    print(f"generated text with special:\n{text}")

    text = tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print(f"generated text no special:\n{text}")

    token_ids_1 = tokenizer.encode("<s> Hello, everyone, 北京 </s>.", add_special_tokens=True)
    #token_ids_1 = tokenizer.encode("<s>Hello,北京 </s>.", add_special_tokens=False)
    print(f"token ids1:{token_ids_1}") # [1, 15043, 29871, 1, 29889]
    print(f"id to tokens:{tokenizer.convert_ids_to_tokens(token_ids_1)}")
    print(f"id to token map:{[(i, tokenizer.convert_ids_to_tokens(i)) for i in token_ids_1]}")

    print(f"id:0 to token:{tokenizer.convert_ids_to_tokens(0)}")
    print(f"id:1 to token:{tokenizer.convert_ids_to_tokens(1)}")
    print(f"id:33 to token:{tokenizer.convert_ids_to_tokens(33)}")
    print(f"id:65 to token:{tokenizer.convert_ids_to_tokens(65)}")
    print(f"id:48 to token:{tokenizer.convert_ids_to_tokens(48)}")
    # ------------
    tokens = ['<s>', '<s>', 'Hello', ',', '北', '京', '▁', ' ', '</s>', '.']
    print(f"tokens to ids:{tokenizer.convert_tokens_to_ids(tokens)}")
    # tokens to id map:[('<s>', 1), ('<s>', 1), ('Hello', 10994), (',', 29892), ('北', 30662), ('京', 30675), ('▁', 29871), (' ', 0), ('</s>', 2), ('.', 29889)]
    print(f"tokens to id map:{[(t, tokenizer.convert_tokens_to_ids(t)) for t in tokens]}")

def test_tiny_llama_20M():
    # 测试从huggface上下载的真实的小模型
    model_path = '/home/hkx/data/work/hf_data_and_model/models/TinyStories-LLaMA2-20M-256h-4l-GQA'
    #model_path = '/home/hkx/data/work/models/chinese-baby-llama2'
    tokenizer:LlamaTokenizer = LlamaTokenizer.from_pretrained(model_path, use_fast=False)
    model = LlamaForCausalLM.from_pretrained(model_path)

    prompt = "Hi, can you talk something about beijing?"
    #prompt = "你好，能介绍一下北京吗"
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate
    generate_input = {
        "input_ids": inputs.input_ids,
        "max_new_tokens": 512,
        "do_sample": True,
        "top_k": 50,
        "top_p": 0.95,
        "temperature": 0.3,
        "repetition_penalty": 1.3,
        "eos_token_id": tokenizer.eos_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "pad_token_id": tokenizer.pad_token_id
    }


    generate_ids = model.generate(**generate_input)
    #generate_ids = model.generate(inputs.input_ids, max_length=500)
    text = tokenizer.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
    print(f"generate_ids:\n{generate_ids}")
    print(f"generated text with special:\n{text}")

    text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(f"\ngenerated text skip special token:\n{text}")

if __name__ == "__main__":
    print("args:", sys.argv)
    test_llama_tokenzier()
    if False:
        test_tiny_llama_20M()