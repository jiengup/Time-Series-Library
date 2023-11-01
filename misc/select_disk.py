import pandas as pd
import numpy as np
from datetime import datetime

metric_file_p = "/home/xgj/Time-Series-Library/result_long_term_forecast.txt"
raw_dataset_p = "../dataset/tsdata_v1.npy"
select_out_csv_p = "../dataset/clouddisk/select_disks_2.csv"
r_select_out_csv_p = "../dataset/clouddisk/r_select_disks_2.csv"
w_select_out_csv_p = "../dataset/clouddisk/w_select_disks_2.csv"
select_index_out_csv_p = "../dataset/clouddisk/select_index_2.txt"

metrics = []

with open(metric_file_p, "r") as f:
    lines = f.readlines()
    line_len = len(lines)
    for idx in range(line_len):
        line = lines[idx]
        if "mse" in line:
            continue
        length = len(line)
        s = ""
        i = 0
        while i < length:
            s += line[i]
            if s == "long_term_forecast_disk_":
                num = ""
                j = i+1
                while line[j].isdigit():
                    num += line[j]
                    j += 1
                assert line[j] == '_'
                num = int(num)
                i = j
                s = ""
                metric_line = lines[idx+1]
                mse_part, mae_part = metric_line.strip().split(",")
                mse_part.strip()
                mae_part.strip()
                mse = float(mse_part.split(":")[-1].strip())
                mae = float(mae_part.split(":")[-1].strip())
                metrics.append({"index": num, "mse": mse, "mae": mae})
            i += 1

treshold = 0.4
metrics_df = pd.DataFrame(metrics)
select_df = metrics_df[(metrics_df["mse"] <= treshold) & (metrics_df["mae"] <= treshold)]
print("mse mean: ", select_df["mse"].mean())
print("mse std: ", select_df["mse"].std())
print("mae mean: ", select_df["mae"].mean())
print("mae std: ", select_df["mae"].std())

exit(0)

raw_data = np.load(raw_dataset_p)
select_data = raw_data[:, select_df["index"].to_list(), :]
print("select_data shape: ", select_data.shape)

start_timestamp = 1601481600
end_timestamp = 1609430400

start_datetime = datetime.fromtimestamp(start_timestamp)
end_datetime = datetime.fromtimestamp(end_timestamp)

freq = "5min"
dates = pd.date_range(start=start_datetime, end=end_datetime, freq=freq)

df_dic = {"date": dates}

for idx in select_df["index"]:
    col_name_r = "{}_r".format(idx)
    col_name_w = "{}_w".format(idx)
    df_dic[col_name_r] = raw_data[:, idx, 0]
    df_dic[col_name_w] = raw_data[:, idx, 1]

a_select_df = pd.DataFrame(df_dic)
print("a_select_df.shape: ", a_select_df.shape)
a_select_df.to_csv(select_out_csv_p, index=None)

r_df_dic = {"date": dates}
for idx in select_df["index"]:
    r_df_dic[idx] = raw_data[:, idx, 0]
    
r_select_df = pd.DataFrame(r_df_dic)
print("r_select_df.shape: ", r_select_df.shape)
r_select_df.to_csv(r_select_out_csv_p, index=None)

w_df_dic = {"date": dates}
for idx in select_df["index"]:
    w_df_dic[idx] = raw_data[:, idx, 1]
    
w_select_df = pd.DataFrame(w_df_dic)
print("w_select_df.shape: ", w_select_df.shape)
w_select_df.to_csv(w_select_out_csv_p, index=None)

with open(select_index_out_csv_p, "w") as f:
    for idx in select_df["index"]:
        f.write("{}\n".format(idx))