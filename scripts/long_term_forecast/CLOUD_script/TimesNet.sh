export CUDA_VISIBLE_DEVICES=0,1

model_name=TimesNet

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/clouddisk/ \
  --data_path select_disks.csv \
  --model_id CLOUD_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --target 198_w \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 12 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 230 \
  --dec_in 230 \
  --c_out 230 \
  --d_model 256 \
  --d_ff 512 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1