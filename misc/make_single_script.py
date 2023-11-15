cmd_base = '''
python -u run.py \
--task_name long_term_forecast \
--is_training 1 \
--root_path ./dataset/clouddisk/ \
--data_path disk_{}.csv \
--model_id disk_{}_96_12 \
--model $model_name \
--data custom \
--features M \
--freq 5min \
--seq_len 96 \
--label_len 48 \
--pred_len 12 \
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
'''
with open("DLinear.sh", "w") as f:
	for i in range(964):
		f.write(cmd_base.format(i, i))

