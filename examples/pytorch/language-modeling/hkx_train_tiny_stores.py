import os
import torch
import random
import numpy as np
from transformers import Trainer
from datasets import load_dataset
from typing import Dict, List, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments
from transformers.models.llama.tokenization_llama_fast import LlamaTokenizer
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 设置 device，能用 cuda 就用 cuda，苹果 M 系列可以用 mps
#tokenizer = AutoTokenizer.from_pretrained('/home/hkx/data/work/hf_data_and_model/models/NousResearch/Llama-2-7b-hf')

max_seq_len = 2048
hidden_size = 256
intermediate_size = (int(hidden_size * 8 / 3 / 128) + 1) * 128  # 一般为hidden_size的4倍,此处设为128的8/3倍
num_hidden_layers=4

# Kaiming 初始化
def kaiming_initialization(model):
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() > 1:
            torch.nn.init.kaiming_uniform_(param, mode='fan_in', nonlinearity='leaky_relu')
        elif 'bias' in name:
            # 一般偏置项可以初始化为0
            torch.nn.init.constant_(param, 0)

# 打印模型的每一层及其参数大小
def print_model_parameters(model):
    print("Layer Name & Parameters")
    print("----------------------------")
    total_params = 0
    for name, parameter in model.named_parameters():
        param_size = parameter.size()
        param_count = torch.prod(torch.tensor(param_size)).item()
        total_params += param_count
        print(f"{name:50} | Shape: {str(param_size):30} | Count: {str(param_count):20}")
    print("----------------------------")
    print(f"Total Parameters: {total_params} ({total_params / 1000000:.1f} M)")

def inference(
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        #tokenizer: LlamaTokenizer,
        input_text: str = "Once upon a time, ",
        max_new_tokens: int = 16
):
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_k=40,
        top_p=0.95,
        temperature=0.8
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True) # 只看生成的第0个序列
    # print(outputs)
    print(generated_text)

def convert_text_to_ids(examples: Dict[str, List[str]], tokenizer:LlamaTokenizer)-> Dict[str, list]:
    encoded_texts = tokenizer(examples['text'], add_special_tokens=False)
    input_ids_list = encoded_texts['input_ids']

    new_input_ids_list, new_attn_mask_list = [], []
    for input_ids in input_ids_list:
        temp = input_ids[-max_seq_len + 1:] + [tokenizer.eos_token_id] # 加了一个eos_token_id=2
        new_input_ids_list.append(temp)
        new_attn_mask_list.append([1] * len(temp)) # 所有的token所在的地方均为1
    return {
        "input_ids": new_input_ids_list,
        "attention_mask": new_attn_mask_list,
        "labels": new_input_ids_list[:],  # 对于lm，input_ids与label_ids相同，transformers内部会将label进行向左shift
    }

def batch_padding(examples: List[Dict[str, Any]], tokenizer, padding_side='left') -> Dict[str, torch.Tensor]:
    """
        将List[Dict[str, Any]] 进行padding后转成Dict[str, Tensor]
        1. 将batch list中的多条样本变为一个batch中的单条样本
        2. 对一个batch中的样本进行padding
    """
    max_len_in_batch = min(max_seq_len+1, max([len(x["input_ids"]) for x in examples])) # 末尾多了一个eos
    padded_output = {}

    for example in examples:
        for key, value in example.items():
            if key == "labels":
                pad_id = -100
            elif key.startswith("attention"):
                pad_id = 0
            else:  # input token ids
                pad_id = tokenizer.pad_token_id
            # 截断
            value = value[:max_len_in_batch]
            to_pad_ids = [pad_id]*(max_len_in_batch-len(value))
            if padding_side == "left":
                padded_value = to_pad_ids + value
            else:
                padded_value = value + to_pad_ids
            update_value = padded_output.setdefault(key, [])
            update_value.append(padded_value)
            padded_output[key] = update_value
    # 转为tensor_ids
    padded_tensor = {k:torch.LongTensor(v) for k,v in padded_output.items()} # 均为torch.int64
    return padded_tensor


def train_pred(args):
    #tokenizer = AutoTokenizer.from_pretrained('NousResearch/Llama-2-7b-hf')
    set_seed(0)

    enable_wandb = args.enable_wandb
    os.environ['WANDB_DISABLED'] = str(not enable_wandb).lower()  # 禁用 wandb，也可以不用这一条
    tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_path)
    data_path=args.dataset_path

    config = AutoConfig.for_model(
        model_type="llama",
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=16,
        num_hidden_layers=num_hidden_layers,
        num_key_value_heads=8
    )

    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float32).to(device)
    # 初始化
    kaiming_initialization(model)

    print_model_parameters(model) # 如果是模型的话，大约20M的参数个数，存储大小为：20*4MB=80MB
    # 未训练之前，打印预测结果看下
    inference(model, tokenizer)

    # 应用全部训练集，约 2.7 M
    # ds_train = load_dataset('noanabeshima/TinyStoriesV2', split='train')
    # 这里可以调整比例，我只用了 10%，约 270 K
    ds_val = load_dataset(data_path, split='validation')

    if args.is_debug:
        ds_train = load_dataset(data_path, split='train[:1%]')
    else:
        ds_train = load_dataset(data_path, split='train')
    #ds_train = load_dataset(data_path, split='train[:1%]')

    print(f"train data :{ds_train}")
    print(f"test data: {ds_val}")

    ds_train = ds_train.shuffle()
    cpu_core = os.cpu_count()
    ds_train = ds_train.map(function=lambda x:convert_text_to_ids(x, tokenizer),
                            batched=True, # 将样本组成batch, batch为1000,此处的batch与train中的batch不是同一个
                            num_proc=cpu_core,
                            remove_columns=ds_train.column_names,  # 移除原dataset的列名
                            desc='Running tokenizer on train_set: '
                            )
    ds_val = ds_val.map(function=lambda x:convert_text_to_ids(x, tokenizer),
                        batched=True,
                        num_proc=cpu_core,
                        remove_columns=ds_val.column_names,
                        desc='Running tokenizer on val_set: '
                        )

    # ## **4. 训练**
    is_support_bf16=True if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else False
    is_support_fp16 =True if torch.cuda.is_available() and not is_support_bf16 else False

    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_path, 'checkpoints/'),
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        eval_steps=1000,
        per_device_train_batch_size=4, # batch_size=4
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        lr_scheduler_type='cosine',
        bf16=is_support_bf16,
        fp16=is_support_fp16,
        logging_steps=100,
        report_to='wandb' if enable_wandb else 'none',
        num_train_epochs=3,
        save_steps=1000,
        save_total_limit=2,
        seed=3407
    )
    print(f"train args: {training_args}")

    # collator: 对batched的list进行left padding, 如果使用DataCollatorForLanguageModeling,则需要将process_func中的"labels"生成逻辑去掉，
    my_data_collator = lambda x: batch_padding(x, tokenizer)

    # 因为如果使用DataCollatorForLanguageModeling已包含该逻辑
    # mlm=False,不能是mask language model
    data_collator_for_lm = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        tokenizer=tokenizer,
        data_collator=my_data_collator,
        #data_collator=data_collator_for_lm,
    )

    trainer.train()

    # 利用模型进行预测
    model.eval()
    inference(
        model,
        tokenizer,
        "Once upon a time, in a beautiful garden, there lived a little rabbit named Peter Rabbit."
    )

    model.save_pretrained(os.path.join(args.output_path, 'my_model/'))

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True

def load_model_and_infer():
    # 确实可以预测
    tokenizer = AutoTokenizer.from_pretrained('/home/hkx/data/work/hf_data_and_model/models/NousResearch/Llama-2-7b-hf')
    model = AutoModelForCausalLM.from_pretrained('checkpoints/checkpoint-207').to(device)
    model.eval()
    inference(
        model,
        tokenizer,
        "Once upon a time, in a beautiful garden, there lived a little black dog named jack."
    )

def test_my_data_collator():
    tokenizer = AutoTokenizer.from_pretrained('/home/hkx/data/work/hf_data_and_model/models/NousResearch/Llama-2-7b-hf')
    # DataCollatorForLanguageModeling
    # 这⾥的 tokenizer 选⽤的是 Qwen1.5 的，并⾮ LLaMA 的，只是做⼀个⽰意
    my_data_collator = lambda x: batch_padding(x, tokenizer)
    data = ['南京', '南京市', '南京市⻓江']
    raw_tokens = [tokenizer(text) for text in data]
    print(f'tokenizer.pad_token_id: {tokenizer.pad_token_id}\n')
    print("raw tokens:")
    print(raw_tokens)
    """
    raw tokens:
[{'input_ids': [1, 29871, 30601, 30675], 'attention_mask': [1, 1, 1, 1]}, 
{'input_ids': [1, 29871, 30601, 30675, 30461], 'attention_mask': [1, 1, 1, 1, 1]}, 
{'input_ids': [1, 29871, 30601, 30675, 30461, 229, 190, 150, 30775], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]}]
after collator:
{'input_ids': tensor([[    0,     0,     0,     0,     0,     1, 29871, 30601, 30675],
        [    0,     0,     0,     0,     1, 29871, 30601, 30675, 30461],
        [    1, 29871, 30601, 30675, 30461,   229,   190,   150, 30775]]), 
 'attention_mask': tensor([[0, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1]])}
    """

    print("after collator:")
    print(my_data_collator(raw_tokens))


def test_data_collator():
    tokenizer = AutoTokenizer.from_pretrained('/home/hkx/data/work/hf_data_and_model/models/NousResearch/Llama-2-7b-hf')
    # DataCollatorForLanguageModeling
    # 这⾥的 tokenizer 选⽤的是 Qwen1.5 的，并⾮ LLaMA 的，只是做⼀个⽰意
    dc = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    data = ['南京', '南京市', '南京市⻓江']
    raw_tokens = [tokenizer(text) for text in data]
    print(f'tokenizer.pad_token_id: {tokenizer.pad_token_id}\n')
    print("raw tokens:")
    print(raw_tokens)
    """
    raw_tokens:
    [{'input_ids': [1, 29871, 30601, 30675], 'attention_mask': [1, 1, 1, 1]}, 
     {'input_ids': [1, 29871, 30601, 30675, 30461], 'attention_mask': [1, 1, 1, 1, 1]}, 
     {'input_ids': [1, 29871, 30601, 30675, 30461, 229, 190, 150, 30775], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]}
    ]
    
    
    after collator: 
    {'input_ids': 
        tensor([[    0,     0,     0,     0,     0,     1, 29871, 30601, 30675],
            [    0,     0,     0,     0,     1, 29871, 30601, 30675, 30461],
            [    1, 29871, 30601, 30675, 30461,   229,   190,   150, 30775]]), 
    'attention_mask': tensor([[0, 0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1]]),
    'labels': tensor([[ -100,  -100,  -100,  -100,  -100,     1, 29871, 30601, 30675],
            [ -100,  -100,  -100,  -100,     1, 29871, 30601, 30675, 30461],
            [    1, 29871, 30601, 30675, 30461,   229,   190,   150, 30775]])
    }
    """

    print("after collator:")
    print(dc(raw_tokens))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--enable_wandb', type=bool, default=False)
    parser.add_argument('--is_debug', type=bool, default=True)
    parser.add_argument('--tokenizer_path', type=str, default="/home/hkx/data/work/hf_data_and_model/models/NousResearch/Llama-2-7b-hf/")
    parser.add_argument('--dataset_path', type=str, default="/home/hkx/data/work/hf_data_and_model/datas/TinyStoriesV2/")
    parser.add_argument('--output_path', type=str, default="./")

    args = parser.parse_args()
    print("parsed args:", args)

    train_pred(args)
    #load_model_and_infer()
    # test_data_c0llator()
    #test_my_data_collator()