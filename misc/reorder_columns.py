import pandas as pd

select_disk_f = "/home/xgj/Time-Series-Library/dataset/clouddisk/select_disks_2.csv"
select_disk_out = "/home/xgj/Time-Series-Library/dataset/clouddisk/select_disks_2_reordered.csv"

data_df = pd.read_csv(select_disk_f)
print(data_df.head(10))

r_columns = []
w_columns = []
for i in range(1, len(data_df.columns), 2):
    r_columns.append(data_df.columns[i])
    w_columns.append(data_df.columns[i+1])

reordered_columns = ['date']

reordered_columns.extend(r_columns)
reordered_columns.extend(w_columns)

data_df_reordered = data_df[reordered_columns]
print(data_df_reordered.head(10))

data_df_reordered.to_csv(select_disk_out, index=None)