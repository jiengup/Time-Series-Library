export CUDA_VISIBLE_DEVICES=0,1

model_name=LightTS

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/clouddisk/ \
  --data_path disk_0.csv \
  --model_id CLOUD_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --freq 5min \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --itr 1
