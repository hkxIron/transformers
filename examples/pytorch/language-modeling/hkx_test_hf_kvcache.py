from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, LlamaForCausalLM, LlamaTokenizer
from datasets import config
import torch

def bytes_to_giga_bytes(bytes):
  return bytes / 1024 / 1024 / 1024

def flush():
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

def test1():
    base_path="/media/hkx/win/hkx/ubuntu/work/hf_data_and_model/"
    model_name_or_path = f'{base_path}/models/ahxt/LiteLlama-460M-1T'
    print(f'huggingface cache path:{config.HF_DATASETS_CACHE}')
    # device_map='auto', 流水线并行
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto", pad_token_id=0)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    prompt = "Question: Please write a function in Python that transforms bytes to Giga bytes.\n\nAnswer:"
    result = pipe(prompt, max_new_tokens=60)[0]["generated_text"][len(prompt):]
    print(result)
    print(bytes_to_giga_bytes(torch.cuda.max_memory_allocated()))

def test_8bit():
    base_path="/media/hkx/win/hkx/ubuntu/work/hf_data_and_model/"
    model_name_or_path = f'{base_path}/models/ahxt/LiteLlama-460M-1T'
    print(f'huggingface cache path:{config.HF_DATASETS_CACHE}')
    # device_map='auto', 流水线并行
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, load_in_8bit=True , device_map="auto", pad_token_id=0)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    prompt = "Question: Please write a function in Python that transforms bytes to Giga bytes.\n\nAnswer:"
    result = pipe(prompt, max_new_tokens=60)[0]["generated_text"][len(prompt):]
    print(result)
    print(bytes_to_giga_bytes(torch.cuda.max_memory_allocated()))

def test_4bit():
    base_path="/media/hkx/win/hkx/ubuntu/work/hf_data_and_model/"
    model_name_or_path = f'{base_path}/models/ahxt/LiteLlama-460M-1T'
    print(f'huggingface cache path:{config.HF_DATASETS_CACHE}')
    # device_map='auto', 流水线并行
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, load_in_4bit=True , device_map="auto", pad_token_id=0)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    prompt = "Question: Please write a function in Python that transforms bytes to Giga bytes.\n\nAnswer:"
    result = pipe(prompt, max_new_tokens=60)[0]["generated_text"][len(prompt):]
    print(result)
    print(bytes_to_giga_bytes(torch.cuda.max_memory_allocated()))

def test_no_flashatten():
    base_path="/media/hkx/win/hkx/ubuntu/work/hf_data_and_model/"
    model_name_or_path = f'{base_path}/models/ahxt/LiteLlama-460M-1T'
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto", pad_token_id=0)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    import time
    long_prompt = "Question: Please write a function in Python that transforms bytes to Giga bytes.\n\nAnswer:"*5
    start_time = time.time()
    result = pipe(long_prompt, max_new_tokens=60)[0]["generated_text"][len(long_prompt):]
    print(f"Generated in {time.time() - start_time} seconds.")

def test_with_flashatten():
    #print(transformers.__version__)
    base_path="/media/hkx/win/hkx/ubuntu/work/hf_data_and_model/"
    model_name_or_path = f'{base_path}/models/ahxt/LiteLlama-460M-1T'
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto", pad_token_id=0)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    import time
    long_prompt = "Question: Please write a function in Python that transforms bytes to Giga bytes.\n\nAnswer:"*5
    start_time = time.time()
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
        result = pipe(long_prompt, max_new_tokens=60)[0]["generated_text"][len(long_prompt):]
    print(f"Generated in {time.time() - start_time} seconds.")

def test_generate_no_kv_cache():
    base_path="/media/hkx/win/hkx/ubuntu/work/hf_data_and_model/"
    model_name_or_path = f'{base_path}/models/ahxt/LiteLlama-460M-1T'
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto", pad_token_id=0)
    prompt = "Question: Please write a function in Python that transforms bytes to Giga bytes.\n\nAnswer:"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 设置 device，能用 cuda 就用 cuda，苹果 M 系列可以用 mps
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device) # input_ids: [batch, seq_len]

    for _ in range(5):
        next_logits = model(input_ids)["logits"][:, -1:]
        next_token_id = torch.argmax(next_logits, dim=-1)
        # 没有kv cached的解码，需要手动将next_token_id与input_ids拼接起来
        input_ids = torch.cat([input_ids, next_token_id], dim=-1)
        print("shape of input_ids:", input_ids.shape)

    generated_text = tokenizer.batch_decode(input_ids[:, -5:])
    print("generated_text:", generated_text)

def test_generate_with_kv_cache():
    base_path="/media/hkx/win/hkx/ubuntu/work/hf_data_and_model/"
    model_name_or_path = f'{base_path}/models/ahxt/LiteLlama-460M-1T'
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.padding_side ='left'
    print(tokenizer)
    model:LlamaForCausalLM = LlamaForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto", pad_token_id=0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 设置 device，能用 cuda 就用 cuda，苹果 M 系列可以用 mps
    print("pad token id:", tokenizer.pad_token_id)

    prompt = [ "Question: Please write a function in Python that transforms bytes to Giga bytes.\n\nAnswer:",
               "Question: count number in word 'chinese'.\n\n Answer:"]
    # 需要传入kv cache,并返回kv cache
    generated_tokens = []
    next_token = tokenizer(prompt, return_tensors="pt", padding=True)
    next_token_id = next_token["input_ids"].to(device)
    attn_mask = next_token["attention_mask"]
    past_key_values = None  # past_key_values is the key-value cache, kv cache初始化为None,后面动态增加
    batch_size, seq_len = next_token_id.shape
    cache_position = torch.arange(seq_len, dtype=torch.int64)

    for _ in range(5):
        # 每次只输入下一个token
        next_logits, past_key_values = model.forward(next_token_id, attention_mask = attn_mask, cache_position=cache_position, past_key_values=past_key_values, use_cache=True).to_tuple()
        # next_logits:[batch, seq_len, vocab_size]
        # -> [batch, last_token=1, vocab_size]
        next_logits = next_logits[:, -1:]
        next_token_id = torch.argmax(next_logits, dim=-1)
        attn_mask = torch.concat([attn_mask, torch.ones(batch_size, 1, dtype=torch.int64)], dim=1)

        print("shape of input_ids", next_token_id.shape)
        # 取出第0层的key的cache来看看
        print("layer num of key-value cache", len(past_key_values))  # 第1个0为第0层，第2个0为第key, past_key_values are of shape [num_layers, 0 for k and 1 for v, batch_size, length, hidden_dim]
        print("length of key cache shape of layer_0", past_key_values[0][0].shape)  # 第1个0为第0层，第2个0为第key, past_key_values are of shape [num_layers, 0 for k and 1 for v, batch_size, length, hidden_dim]
        print("length of value cache shape of layer_0", past_key_values[0][1].shape)  # 第1个0为第0层，第2个0为第key, past_key_values are of shape [num_layers, 0 for k and 1 for v, batch_size, length, hidden_dim]
        generated_tokens.append(next_token_id)
        cache_position = cache_position[-1:] + 1 # add one more position for the next token

        """
        输出：
        shape of input_ids torch.Size([1, 1])
        layer num of key-value cache 24
        length of key cache shape of layer_0 torch.Size([1, 2, 20, 64])
        length of value cache shape of layer_0 torch.Size([1, 2, 20, 64])
        shape of input_ids torch.Size([1, 1])
        layer num of key-value cache 24
        length of key cache shape of layer_0 torch.Size([1, 2, 21, 64])
        length of value cache shape of layer_0 torch.Size([1, 2, 21, 64])
        shape of input_ids torch.Size([1, 1])
        layer num of key-value cache 24
        length of key cache shape of layer_0 torch.Size([1, 2, 22, 64])
        length of value cache shape of layer_0 torch.Size([1, 2, 22, 64])
        shape of input_ids torch.Size([1, 1])
        layer num of key-value cache 24
        length of key cache shape of layer_0 torch.Size([1, 2, 23, 64])
        length of value cache shape of layer_0 torch.Size([1, 2, 23, 64])
        shape of input_ids torch.Size([1, 1])
        layer num of key-value cache 24
        length of key cache shape of layer_0 torch.Size([1, 2, 24, 64])
        length of value cache shape of layer_0 torch.Size([1, 2, 24, 64])
        
        generated_text: ['\n', 'def', ' transform', '_', 'bytes']
        """

    gen_tokens = torch.concat(generated_tokens, dim=-1)
    generated_text = tokenizer.batch_decode(gen_tokens)
    print("generated_text:", generated_text)

def matmul(X, W):
    return X@W

def matmul_by_token(X, W):
    batch, seq_len, hidden_dim = X.shape
    seq_out = []
    for seq_index in range(seq_len):
        x = X[:, seq_index, :]
        seq_out.append(x @ W)
    return torch.stack(seq_out, dim=1)

def matmul_multiple(X, W, fn, N):
    for _ in range(N):
        y = torch.sigmoid(fn(X, W))
        X = y
    return y

def test_matrix_mul():
    torch.manual_seed(0)
    batch, seq_len, hidden_dim = 2, 5, 3
    proj_dim = 3
    X = torch.randn((batch, seq_len, hidden_dim), dtype=torch.bfloat16)
    W = torch.randn((hidden_dim, proj_dim), dtype=torch.bfloat16)
    y1 = matmul_multiple(X, W, matmul, 100)
    y2 = matmul_multiple(X, W, matmul_by_token, 100)
    print(torch.abs(y1-y2).max())

def test_generate_kv_cache2():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

    #model_id = "meta-llama/Llama-2-7b-chat-hf"
    base_path="/media/hkx/win/hkx/ubuntu/work/hf_data_and_model/"
    model_id = f'{base_path}/models/ahxt/LiteLlama-460M-1T'

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    past_key_values = DynamicCache()
    messages = ["Hello, what's your name."]
    #inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    inputs = tokenizer(messages, return_tensors="pt", padding=True)
    print(f'{inputs=}')

    """
    One important concept you need to know when writing your own generation loop, is cache_position. 
    In case you want to reuse an already filled Cache object by calling forward(), you have to pass in a valid cache_position which will indicate the positions of inputs in the sequence.
    Note that cache_position is not affected by padding, and always adds one more position for each token. 
    For example, if key/value cache contains 10 tokens (no matter how many of it is a pad token), the cache position for the next token should be torch.tensor([10]).
    """
    generated_ids = inputs.input_ids
    cache_position = torch.arange(inputs.input_ids.shape[1], dtype=torch.int64)
    max_new_tokens = 10

    for _ in range(max_new_tokens):
        outputs = model(**inputs, cache_position=cache_position, past_key_values=past_key_values, use_cache=True)
        # Greedily sample one next token
        next_token_ids = outputs.logits[:, -1:].argmax(-1)
        generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)
        # Prepare inputs for the next generation step by leaaving unprocessed tokens, in our case we have only one new token
        # and expanding attn mask for the new token, as explained above
        attention_mask = inputs["attention_mask"]
        attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
        inputs = {"input_ids": next_token_ids, "attention_mask": attention_mask}
        cache_position = cache_position[-1:] + 1  # add one more position for the next token

    print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])

if __name__ == '__main__':
    #test1()
    #test_8bit()
    #test_4bit()
    #test_no_flashatten()
    #test_with_flashatten()
    #test_matrix_mul()
    #test_generate_with_kv_cache()
    #test_generate_no_kv_cache()
    test_generate_kv_cache2()
