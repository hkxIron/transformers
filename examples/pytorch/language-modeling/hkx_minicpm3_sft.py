# -*- coding: utf-8 -*-
import json
from dataclasses import dataclass, field
from typing import Dict, Optional, List

import torch
import transformers
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments, BitsAndBytesConfig, BatchEncoding, PreTrainedTokenizer,
                          default_data_collator)
from transformers.utils import PaddingStrategy


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="openbmb/MiniCPM-2B-sft-bf16")


@dataclass
class DataArguments:
    train_data_path: str = field(
        default="data/AdvertiseGenChatML/train.json",
        metadata={"help": "Path to the training data."},
    )
    eval_data_path: str = field(
        default="data/AdvertiseGenChatML/dev.json",
        metadata={"help": "Path to the test data."},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = field(default=False)
    qlora: bool = field(default=False)


class SupervisedDataset(torch.utils.data.Dataset):
    """Dataset for supervised fine-tuning.
    注意，并不是hf的Dataset
    """
    
    def __init__(
        self,
        data_path,
        tokenizer,
        model_max_length=4096,
        user_tokens='<用户>',
        assistant_tokens='<AI>',
    ):
        super(SupervisedDataset, self).__init__()
        data_list = []
        with open(data_path, "r") as fr:
            for line in fr.readlines():
                data_list.append(json.loads(line))
        self.data = data_list
        print(f"data len:{len(self.data)}")
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        self.user_tokens = self.tokenizer.encode(user_tokens) #针对不同模型，都可以对应到<用户>的id
        self.assistant_tokens = self.tokenizer.encode(assistant_tokens) #针对不同模型，都可以对应到<AI>的id
        self.ignore_index = -100
        item_dict = self.convert_to_input_label_ids(self.data[0])
        print("input:", self.tokenizer.decode(item_dict["input_ids"]))
        # labels = []
        # for id_ in item_dict["label_ids"]:
        #     if id_ == -100:
        #         continue
        #     labels.append(id_)
        # print("label:", self.tokenizer.decode(labels))

    def __len__(self):
        return len(self.data)
    
    def _get_ids_from_messages_format(self, example:Dict[str, str]):
        input_ids = [self.tokenizer.bos_token_id]
        label_ids = [self.ignore_index]
        for message in example["messages"]:
            role = message["role"]
            content = message["content"]
            content_ids = self.tokenizer.encode(content, add_special_tokens=False)

            if role == "user": # user是用户输入，不需要计算loss
                input_ids += self.user_tokens + content_ids
                label_ids += [self.ignore_index] * len(self.user_tokens) + [ self.ignore_index ] * len(content_ids)
            else: # assistant是llm的输出，需要计算loss
                input_ids += self.assistant_tokens + content_ids
                label_ids += [self.ignore_index] * len(self.assistant_tokens) + content_ids
        return input_ids, label_ids
    
    def _get_ids_from_prompt_input_format(self, example:Dict[str, str]):
        input_ids = [self.tokenizer.bos_token_id]
        label_ids = [self.ignore_index]
        
        prompt_text = example['prompt']
        input_text = example['input']
        output_text = example['output']
         
        prompt_input_ids = self.tokenizer.encode(prompt_text+input_text, add_special_tokens=False)
        input_ids += prompt_input_ids
        label_ids += [self.ignore_index] * len(prompt_input_ids)
        
        output_ids = self.tokenizer.encode(output_text, add_special_tokens=False)
        input_ids+=output_ids
        label_ids+=output_ids
        return input_ids, label_ids
         

    def convert_to_input_label_ids(self, example:Dict[str, str], pad_to_max_len=True)->Dict[str, List[int]]:
        if "messages" in example:
            input_ids, label_ids = self._get_ids_from_messages_format(example)
        else:
            #print(example)
            input_ids, label_ids = self._get_ids_from_prompt_input_format(example) 

        input_ids.append(self.tokenizer.eos_token_id)
        label_ids.append(self.tokenizer.eos_token_id)
        
        # truncate to max len
        input_ids = input_ids[: self.model_max_length]
        label_ids = label_ids[: self.model_max_length]
        attention_mask = [1] * len(input_ids)

        if pad_to_max_len:
            # pad to max len
            # 如果len是负数，追加的list就是空列表
            input_ids += [self.tokenizer.eos_token_id] * (self.model_max_length - len(input_ids)) # padding的部分为eos_token
            label_ids += [self.ignore_index] * (self.model_max_length - len(label_ids)) # padding的部分为ignore_index=-100
            attention_mask += [0] * (self.model_max_length - len(attention_mask)) # padding的部分为0

        # convert to pt tensor
        input_ids = torch.LongTensor(input_ids)
        label_ids = torch.LongTensor(label_ids)
        attention_mask = torch.LongTensor(attention_mask)

        return {
            "input_ids": input_ids,
            #"label_ids": label_ids,
            "labels": label_ids,
            "attention_mask": attention_mask,
        }

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return self.convert_to_input_label_ids(self.data[idx], pad_to_max_len=True)

def load_model_and_tokenizer(
    model_path: str,
    max_length: int = 4096,
    use_lora: bool = True,
    qlora: bool = False,
    bf16: bool = False,
    fp16: bool = False,
):
    """load model and tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    assert not (bf16 and fp16), "bf16 or fp16, not both"
    if bf16:
        dtype = torch.bfloat16
    elif fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32
    if qlora:
        assert use_lora, "use_lora must be True when use_qlora is True"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,  # 是否进行4bit量化, 需要gpu
            load_in_8bit=False,  # 是否进行8bit量化, 需要gpu
            bnb_4bit_compute_dtype=torch.float16,  # 计算精度设置
            bnb_4bit_quant_storage=torch.uint8,  # 量化权重的储存格式
            bnb_4bit_quant_type="nf4",  # 量化格式，这里用的是正太分布的int4
            bnb_4bit_use_double_quant=True,  # 是否采用双量化，即对zeropoint和scaling参数进行量化
            llm_int8_enable_fp32_cpu_offload=False,  # 是否llm使用int8，cpu上保存的参数使用fp32
            llm_int8_has_fp16_weight=False,  # 是否启用混合精度
            #llm_int8_skip_modules=["out_proj", "kv_proj", "lm_head"],  # 不进行量化的模块
            llm_int8_threshold=6.0,  # llm.int8()算法中的离群值，根据这个值区分是否进行量化
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            #quantization_config=quantization_config, # 量化需要gpu支持才行
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
    if use_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        lora_config = LoraConfig(
            init_lora_weights="gaussian",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "v_proj"],
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            inference_mode=False,
        )
        model = get_peft_model(model, lora_config)
        # trainable params: 2,949,120 || all params: 3,010,652,928 || trainable%: 0.09795616002669305
        model.print_trainable_parameters()
        # model.enable_input_require_grads()  # need when using adapter

    return model, tokenizer

"""
将多个input dict组成的list转换成一个dict,其中key不变，但value是多个原始数据组成的数组
eg:
input_dict_list:
[
    {
    "input_ids":[1,2,3,7,8,9],
    "label_ids":[-100,-100,-100,8,9,200],
    "attention_mask":[1,1,1],
    },
   {
    "input_ids":[1,2,3,7,8,9, 10, 11, 12],
    "label_ids":[-100,-100,-100,8,9,10,11,12, 200],
    "attention_mask":[1,1,1, 1,1,1],
    },
]

collator会先按max_seq_len进行padding后，再进行组成batch
BatchEncoding(
    data={
        "input_ids":[[1,2,3,7,8,9, 0,0,0],[1,2,3,7,8,9, 10, 11, 12]],
        "label_ids":[[-100,-100,-100, 8,9,200], [-100, -100, -100, 8,9,10,11,12, 200]],
        "attention_mask":[[1,1,1, 1,1,1,0,0,0], [1,1,1, 1,1,1,1,1,1]],
    }
)
"""
def get_collator(input_dict_list:List[Dict[str, list[int]]], max_seq_length:int, tokenizer: PreTrainedTokenizer)->BatchEncoding:
    collator_data:BatchEncoding = tokenizer.pad(
        input_dict_list,
        padding= PaddingStrategy.MAX_LENGTH,
        max_length=max_seq_length,
        return_tensors="pt",
    )
    # collator:也可以完全自己计算，不需要使用tranformer的函数
    #collator_data = default_data_collator(input_dict_list)
    #default_data_collator()
    print("collator_data:", collator_data)
    print("collator_data input_ids:", collator_data['input_ids'])
    print("collator_data input_ids shape:", collator_data['input_ids'].shape)
    print("collator_data labels shape:", collator_data['labels'].shape)
    return collator_data

if __name__ == "__main__":
    #model_path = "/mnt/data/user/tc_agi/yh/models/MiniCPM"
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model, tokenizer = load_model_and_tokenizer(
        model_path=model_args.model_name_or_path,
        max_length=training_args.model_max_length,
        use_lora=training_args.use_lora,
        qlora=training_args.qlora,
        bf16=training_args.bf16,
        fp16=training_args.fp16
    )

    train_dataset = SupervisedDataset(
        data_path=data_args.train_data_path,
        tokenizer=tokenizer,
        model_max_length=training_args.model_max_length,
    )
    eval_dataset = SupervisedDataset(
        data_path=data_args.eval_data_path,
        tokenizer=tokenizer,
        model_max_length=training_args.model_max_length,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=lambda x: get_collator(x, training_args.model_max_length, tokenizer)
    )

    trainer.train()
    # save the incremental PEFT weights, more details can be found in https://huggingface.co/blog/peft
    trainer.save_model("output_dir")