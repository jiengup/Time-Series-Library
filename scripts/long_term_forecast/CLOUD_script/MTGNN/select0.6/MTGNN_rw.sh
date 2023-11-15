export CUDA_VISIBLE_DEVICES=0,1

model_name=MTGNN

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/clouddisk/ \
  --data_path select_disks_2.csv \
  --model_id CLOUD_96_12 \
  --model $model_name \
  --data custom \
  --features M \
  --target 198 \
  --freq 5min \
  --seq_len 12 \
  --label_len 6 \
  --pred_len 3 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 409 \
  --dec_in 409 \
  --c_in 2 \
  --c_out 409 \
  --des 'Exp' \
  --train_epochs 30 \
  --itr 1 \
  --n_vertex 409 \
#  --buildA 
