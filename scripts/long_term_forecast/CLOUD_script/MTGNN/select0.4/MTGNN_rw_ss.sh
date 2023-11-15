export CUDA_VISIBLE_DEVICES=0,1

model_name=MTGNN

python -u run.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --root_path ./dataset/clouddisk/select_0.4 \
  --data_path select_disks_2.csv \
  --model_id CLOUD_96_12 \
  --model $model_name \
  --data custom \
  --features M \
  --target 198 \
  --batch_size 8 \
  --freq 5min \
  --seq_len 288 \
  --label_len 0 \
  --pred_len 1 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 409 \
  --dec_in 409 \
  --c_in 2 \
  --c_out 409 \
  --des 'Exp_single_step' \
  --train_epochs 30 \
  --itr 1 \
  --n_vertex 409 \
  --buildA 
