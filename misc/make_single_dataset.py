import os
import numpy as np
import pandas as pd
from datetime import datetime

aggr_pth_p = "/home/xgj/Time-Series-Library/dataset/tsdata_v1.npy"
out_dataset_d = "/home/xgj/Time-Series-Library/dataset/clouddisk"

start_timestamp = 1601481600
end_timestamp = 1609430400

start_datetime = datetime.fromtimestamp(start_timestamp)
end_datetime = datetime.fromtimestamp(end_timestamp)

freq = "5min"
dates = pd.date_range(start=start_datetime, end=end_datetime, freq=freq)

aggr_data = np.load(aggr_pth_p)
n_samples, n_disks, feature_dim = aggr_data.shape

for i in range(n_disks):
	out_csv_p = os.path.join(out_dataset_d, "disk_{}.csv".format(i))
	out_df = pd.DataFrame({"date": dates, "0": aggr_data[:, i, 0].astype(np.float32), "OT": aggr_data[:, i, 1].astype(np.float32)})
	out_df.to_csv(out_csv_p, index=None)
