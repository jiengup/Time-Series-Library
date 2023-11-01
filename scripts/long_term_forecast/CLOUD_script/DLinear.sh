export CUDA_VISIBLE_DEVICES=0,1

model_name=DLinear

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/clouddisk/ \
  --data_path disk_1.csv \
  --model_id disk_1_96_12 \
  --model $model_name \
  --data custom \
  --features M \
  --freq 5min \
  --seq_len 288 \
  --label_len 144 \
  --pred_len 1 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --itr 1 \
  --use_multi_gpu \
  --devices 0,1 \
