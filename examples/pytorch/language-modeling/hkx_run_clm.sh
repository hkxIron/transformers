  python hkx_run_clm.py \
      --model_name_or_path /home/hkx/data/work/hf_data_and_model/models/MiniCPM-1B-sft-bf16 \
      --trust_remote_code True \
      --dataset_name data/AdvertiseGenChatHF \
      --do_train True \
      --do_eval True \
      --report_to None \
      --output_dir output_model_dir \
      --fp16 True \
      --eval_steps 1000 \
      --evaluation_strategy steps \
      --save_strategy steps \
      --save_steps 100000 \
      --logging_steps 10 \
      --learning_rate 5e-5 \
      --num_train_epochs 2 \

