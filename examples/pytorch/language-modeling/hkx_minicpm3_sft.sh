formatted_time=$(date +"%Y%m%d%H%M%S")
echo $formatted_time

use_lora=0

#模型详见：https://hf-mirror.com/openbmb/MiniCPM3-4B/tree/main

model_path="/home/hkx/data/work/hf_data_and_model/models/MiniCPM3-4B"
if [ $use_lora -eq 0 ];then
  python hkx_minicpm3_sft.py \
      --model_name_or_path ${model_path} \
      --report_to none \
      --output_dir output/AdvertiseGenSFT/$formatted_time/ \
      --train_data_path data/AdvertiseGenChatML/train.jsonl \
      --eval_data_path data/AdvertiseGenChatML/dev.jsonl \
      --learning_rate 5e-5 --per_device_train_batch_size 2 \
      --per_device_eval_batch_size 1 --bf16 \
      --gradient_accumulation_steps 2 --warmup_steps 100 \
      --max_steps 3000 --weight_decay 0.01 \
      --evaluation_strategy steps --eval_steps 100 \
      --save_strategy steps --save_steps 500 --seed 42 \
      --log_level info --logging_strategy steps --logging_steps 10
else
  # lora
  python hkx_minicpm3_sft.py \
      --model_name_or_path ${model_path} \
      --report_to none \
      --output_dir output/AdvertiseGenSFT/$formatted_time/ \
      --train_data_path data/AdvertiseGenChatML/train.jsonl \
      --eval_data_path data/AdvertiseGenChatML/dev.jsonl \
      --learning_rate 5e-5 --per_device_train_batch_size 2 \
      --per_device_eval_batch_size 1 --bf16 \
      --gradient_accumulation_steps 2 --warmup_steps 100 \
      --max_steps 3000 --weight_decay 0.01 \
      --evaluation_strategy steps --eval_steps 100 \
      --save_strategy steps --save_steps 500 --seed 42 \
      --log_level info --logging_strategy steps --logging_steps 10 \
      --use_lora true --qlora true
fi

