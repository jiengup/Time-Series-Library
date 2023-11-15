export CUDA_VISIBLE_DEVICES=0,1

model_name=MTGNN

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/clouddisk/select_0.6 \
  --data_path select_disks.csv \
  --model_id CLOUD_0.6_12_3 \
  --model $model_name \
  --data custom \
  --features M \
  --target nouse \
  --freq 5min \
  --seq_len 12 \
  --label_len 6 \
  --pred_len 3 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 631 \
  --dec_in 631 \
  --c_in 2 \
  --c_out 631 \
  --des 'Exp_learngraph' \
  --train_epochs 30 \
  --itr 1 \
  --n_vertex 631 \
  --buildA 

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/clouddisk/select_0.6 \
  --data_path select_disks.csv \
  --model_id CLOUD_0.6_12_3 \
  --model $model_name \
  --data custom \
  --features M \
  --target nouse \
  --freq 5min \
  --seq_len 12 \
  --label_len 6 \
  --pred_len 3 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 631 \
  --dec_in 631 \
  --c_in 2 \
  --c_out 631 \
  --des 'Exp_nogcn' \
  --train_epochs 30 \
  --itr 1 \
  --n_vertex 631 \
  --gcn 0 \

