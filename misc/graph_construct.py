# %%
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# %%
df = pd.read_csv("../dataset/clouddisk/selected_disk_subscription_info.csv")
df

# %%
(df["is_local"] == 0).all()

# %%
selected_cols = ["cluster_id", "disk_type", "project_name", "disk_size"]
selected_df = df[selected_cols]
selected_df

# %%
new_df = pd.DataFrame()
for col in selected_cols:
    onehot_encode = pd.get_dummies(selected_df[col], prefix=col)
    print(onehot_encode.shape)
    new_df = pd.concat([new_df, onehot_encode], axis=1)

# %%
onehot_features = new_df.values
onehot_features.shape

# %%
from scipy.spatial.distance import cosine
similarity_matrix = np.zeros((409, 409))

# 计算余弦相似度
for i in range(409):
    for j in range(409):
        similarity_matrix[i, j] = 1 - cosine(onehot_features[i], onehot_features[j])

# %%
similarity_matrix

# %%
np.save("dataset/clouddisk/cossim_A.npy", similarity_matrix)

# %% [markdown]
# 

# %%
onehot_features[1]

# %%



