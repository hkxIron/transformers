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
from transformers import LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM, BatchEncoding
import numpy as np


@my_decorator
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

def get_batch_log_probs(logits: torch.FloatTensor, labels: torch.LongTensor, average_log_prob: bool = False) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    此处我们计算negative-log_likehood,与modelling_llama中的计算cross_entropy_loss实现方式不同，但结果相同
    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape

    # logits: [batch_size, sequence_length, vocab_size]
    # labels: (batch_size, sequence_length)
    logits = logits[:, :-1, :] # 取时间0~T-1
    labels = labels[:, 1:].clone() # 取时间1~T, labels取logits向后移一位的结果,注意是clone,国灰labels本身不需要计算梯度
    loss_mask = (labels != -100) # 对于labels为-100的token不计算loss

    # 与cross_entropy_loss是一样的
    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0 # labels=-100在作为index时调用下面torch.gather会越界，所以统一先设为0,后面有loss_mask会将相应的loss去掉
    # log_softmax数学上等价于log(softmax(x)), 但做这两个单独操作速度较慢，数值上也不稳定。这个函数使用另一种公式来正确计算输出和梯度。
    log_probs = logits.log_softmax(dim=-1)
    per_token_log_probs = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(2)

    log_prob_sum = (per_token_log_probs * loss_mask).sum(-1)
    valid_token_num =  loss_mask.sum(-1)
    if average_log_prob:
        return log_prob_sum / valid_token_num, valid_token_num
    else:
        return log_prob_sum, valid_token_num
@my_decorator
def test_tiny_llama_forward_attention_mask():
    model_path = '/home/hkx/data/work/hf_data_and_model/models/TinyStories-LLaMA2-20M-256h-4l-GQA'
    tokenizer:LlamaTokenizer = LlamaTokenizer.from_pretrained(model_path, use_fast=False)
    model:LlamaForCausalLM = LlamaForCausalLM.from_pretrained(model_path)
    prompt='you are a assistant.'
    in_text = [prompt+" can you tell me something about beijing?", prompt+" why sky is always blue?"]
    answer_text = ['beijing is the capital of China',
                   'it is mainly due to the scattering phenomenon of light by air molecules'
                  ]
    all_texts =[x+y for (x,y) in zip(in_text, answer_text)]
    # batch_token_ids:BatchEncoding = tokenizer.__call__(text=input_texts, return_tensors='pt',
    #                             padding='max_length',
    #                             max_length=15,
    #                             add_special_tokens=True)
    # print(f"max_length padding:{batch_token_ids}")
    # ================

    batch_all_token_ids:BatchEncoding = tokenizer.__call__(text=all_texts, return_tensors='pt', padding='longest', add_special_tokens=True)
    # batch_all_token_ids['input_ids'].append(tokenizer.eos_token_id)
    # batch_all_token_ids['attention_mask'].append(1)

    print(f"longest padding:{batch_all_token_ids}")
    #seq_len = batch_all_token_ids['input_ids'].shape[1]
    batch_labels_ids = batch_all_token_ids['input_ids'].clone()
    batch_input_token_len = tokenizer(text=in_text, return_tensors='pt', padding='longest', add_special_tokens=False)['attention_mask'].sum(axis=-1, keepdim=True)
    print(f"batch input token len:{batch_input_token_len}")
    for label,in_seq_len in zip(batch_labels_ids, batch_input_token_len):
        label[0:in_seq_len]= -100
    print(f"batch labels ids:{batch_labels_ids}")
    # train
    result = model.forward(input_ids=batch_all_token_ids['input_ids'], attention_mask=batch_all_token_ids['attention_mask'], labels=batch_labels_ids)
    logits = result.logits
    print(f"logits shape:{logits.shape}")
    print(f"logits:{logits}")
    print(f"cross_entropy_loss:{result.loss}") # cross_entropy_loss:10.587445259094238

    log_probs, valid_token_num = get_batch_log_probs(logits, batch_labels_ids, average_log_prob=False)
    print(f"log_probs:{log_probs}")
    #batch_num = batch_all_token_ids['input_ids'].shape[0]
    #avg_num = np.prod(list(batch_all_token_ids['input_ids'].shape)) - batch_num # logits进行了shift,少一个元素
    #print(f"avg_num:{avg_num}")
    print(f"valid_num:{valid_token_num}")
    avg_nll = -log_probs.sum()/valid_token_num.sum()
    print(f"negative log-likelihood:{avg_nll}") # negative log-likelihood:10.587445259094238
    """
    可以看出，我们自己算的negative-log-likelihood与cross_entropy_loss的结果相同
    """

@my_decorator
def test_tiny_llama_20M_generate():
    # 测试从huggface上下载的真实的小模型
    model_path = '/home/hkx/data/work/hf_data_and_model/models/TinyStories-LLaMA2-20M-256h-4l-GQA'
    #model_path = '/home/hkx/data/work/models/chinese-baby-llama2'
    tokenizer:LlamaTokenizer = LlamaTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_path)

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
    test_tiny_llama_20M_generate()
    test_tiny_llama_forward_attention_mask()

    if False:
        ...