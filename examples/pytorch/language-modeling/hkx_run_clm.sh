  if [ -d output_model_dir ];then
    rm -rf output_model_dir
  fi

  model_path="/home/hkx/data/work/hf_data_and_model/models/TinyStories-LLaMA2-20M-256h-4l-GQA"
  #model_path="/home/hkx/data/work/hf_data_and_model/models/MiniCPM-1B-sft-bf16"
  python hkx_run_clm.py \
      --model_name_or_path ${model_path} \
      --trust_remote_code True \
      --dataset_name data/AdvertiseGenChatHF \
      --do_train True \
      --do_eval True \
      --report_to tensorboard \
      --output_dir output_model_dir \
      --eval_steps 10 \
      --evaluation_strategy steps \
      --save_strategy steps \
      --save_steps 100000 \
      --logging_steps 10 \
      --learning_rate 5e-5 \
      --num_train_epochs 2 \
      --max_train_samples 100 \
      --overwrite_output_dir

