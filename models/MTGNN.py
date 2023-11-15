import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from layers.MTGNN_Related import graph_constructor, dilated_inception, mixprop, LayerNorm
from layers.Embed import DataEmbedding_only_time
from sklearn.metrics.pairwise import cosine_similarity


def get_sorted_top_k(array, top_k=1, axis=-1):
    """
    多维数组排序
    Args:
        array: 多维数组
        top_k: 取数
        axis: 轴维度

    Returns:
        top_sorted_indexes: 位置
    """
    axis_length = array.shape[axis]
    partition_index = np.take(np.argpartition(array, kth=-top_k, axis=axis),
                                  range(axis_length - top_k, axis_length), axis)
    top_scores = np.take_along_axis(array, partition_index, axis)
    # 分区后重新排序
    sorted_index = np.argsort(top_scores, axis=axis)
    sorted_index = np.flip(sorted_index, axis=axis)
    top_sorted_indexes = np.take_along_axis(partition_index, sorted_index, axis)
    return top_sorted_indexes

def generate_graph():
    df = pd.read_csv("~/Time-Series-Library/dataset/clouddisk/select_0.4/selected_disk_subscription_info.csv")
    for i in ["id",
    "app_id",
    "disk_uuid",
    "cluster_id",
    "inst_id",
    "vm_uuid",
    "life_stat",
    "is_local",
    "disk_attr",
    "disk_type",
    "is_vip",
    "pay_mode",
    "pay_type",
    "vm_alias",
    "vm_cpu",
    "vm_mem",
    "app_name",
    "project_name",
    "disk_name",
    "disk_usage",
    "disk_size"
    ]:
        df[i] = df[i].astype(str)
    cols = ["app_id", "cluster_id", "disk_attr", "disk_type", "vm_alias", "vm_cpu", "vm_mem", "project_name", "disk_name"]
    df = df[cols]
    new_df = pd.get_dummies(df['app_id'], prefix = 'app_id')
    for i in ["cluster_id", "disk_attr", "disk_type", "vm_alias", "vm_cpu", "vm_mem", "project_name", "disk_name"]:
        encoded_data = pd.get_dummies(df[i], prefix = i)
        new_df = new_df.join(encoded_data)
    arr = np.array(new_df)
    similarity = cosine_similarity(arr)
    """
    阈值建图, 0.5则连边
    """
    # for i in range(409):
    #     for j in range(409):
    #         if similarity[i, j] < 0.5:
    #             similarity[i, j] = 0

    """
    取topk进行连边
    """
    top_k = 40
    sorted_index = get_sorted_top_k(similarity, top_k=40, axis=1)
    new_s = np.zeros((409, 409))
    for i in range(409):
        for j in range(top_k):
            a = sorted_index[i, j]
            b = similarity[i, a]
            new_s[i, a] = b

    sim_torch = torch.from_numpy(similarity)
    sim_torch = sim_torch.float()
    return sim_torch

class Model(nn.Module):
    def __init__(self, configs):
        # GCN related configs
        gcn_depth = 2
        subgraph_size = 20 # topK
        node_dim = 40

        # CNN dilated configs
        dilation_exponential=1
        kernel_size = 7

        # model struct configs
        conv_channels=32
        residual_channels=32
        skip_channels=64
        end_channels=128

        # task configs
        seq_length = configs.seq_len
        self.in_dim = configs.c_in
        out_dim = configs.pred_len * configs.c_in
        layers = 3
        
        #misc configs
        propalpha=0.05
        tanhalpha=3
        layer_norm_affline=True

        if configs.use_gpu:
            device = torch.device('cuda:{}'.format(configs.gpu))
        else:
            device = torch.device('cpu')
            

        super(Model, self).__init__()
        self.task_name = configs.task_name
        

        self.gcn_true = configs.gcn
        self.buildA_true = configs.buildA
        self.n_vertex = configs.n_vertex
        self.dropout = configs.dropout
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.timeembedding = DataEmbedding_only_time(1, configs.embed, configs.freq, self.dropout)
        self.start_conv = nn.Conv2d(in_channels=self.in_dim+1,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        if not self.buildA_true:
            adj_mat = np.load(os.path.join(configs.root_path, configs.predefinedA)).astype(np.float32)
            assert adj_mat.shape[0] == self.n_vertex
            adj_mat = torch.tensor(adj_mat) - torch.eye(self.n_vertex)
            self.predefined_A = adj_mat.to(device)
        else:
            self.gc = graph_constructor(self.n_vertex, subgraph_size, node_dim, device, alpha=tanhalpha, static_feat=None)

        self.seq_length = seq_length
        if dilation_exponential>1:
            self.receptive_field = int(1+(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
        else:
            self.receptive_field = layers*(kernel_size-1) + 1 # 19

        for i in range(1):
            if dilation_exponential>1:
                rf_size_i = int(1+i*(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
            else:
                rf_size_i = i*layers*(kernel_size-1)+1
            new_dilation = 1
            for j in range(1,layers+1):
                if dilation_exponential > 1:
                    rf_size_j = int(rf_size_i + (kernel_size-1)*(dilation_exponential**j-1)/(dilation_exponential-1))
                else:
                    rf_size_j = rf_size_i+j*(kernel_size-1)

                self.filter_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.gate_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.residual_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=residual_channels,
                                                 kernel_size=(1, 1)))
                if self.seq_length>self.receptive_field:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=skip_channels,
                                                    kernel_size=(1, self.seq_length-rf_size_j+1)))
                else:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=skip_channels,
                                                    kernel_size=(1, self.receptive_field-rf_size_j+1)))

                if self.gcn_true:
                    self.gconv1.append(mixprop(conv_channels, residual_channels, gcn_depth, self.dropout, propalpha))
                    self.gconv2.append(mixprop(conv_channels, residual_channels, gcn_depth, self.dropout, propalpha))

                if self.seq_length>self.receptive_field:
                    self.norm.append(LayerNorm((residual_channels, self.n_vertex, self.seq_length - rf_size_j + 1),elementwise_affine=layer_norm_affline))
                else:
                    self.norm.append(LayerNorm((residual_channels, self.n_vertex, self.receptive_field - rf_size_j + 1),elementwise_affine=layer_norm_affline))

                new_dilation *= dilation_exponential

        self.layers = layers
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                             out_channels=end_channels,
                                             kernel_size=(1,1),
                                             bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                             out_channels=out_dim,
                                             kernel_size=(1,1),
                                             bias=True)
        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=self.in_dim+1, out_channels=skip_channels, kernel_size=(1, self.seq_length), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, self.seq_length-self.receptive_field+1), bias=True)

        else:
            self.skip0 = nn.Conv2d(in_channels=self.in_dim+1, out_channels=skip_channels, kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1), bias=True)


        self.idx = torch.arange(self.n_vertex).to(device)


    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc: [batch_size, seq_len, enc_in]
        batch_size, his_len, _ = x_enc.shape
        # print(x_enc[0, :3, :10])
        x_enc = x_enc.reshape(batch_size, his_len, self.n_vertex, -1)
        # x_enc: [batch_size, c_in, his_len, num_vertex]
        # print(x_enc[0, :3, :5, :])
        x_enc = x_enc.permute(0, 3, 1, 2)
        # print(x_enc[0, :, :3, :5])
        
        _, c_in, _, _ = x_enc.shape
        assert c_in == self.in_dim

        # x: [batch_size, c_in+1, his_len, num_vertex]
        input = self.timeembedding(x_enc, x_mark_enc)
        # print(input[0, :, :3, :5])
        # input [batch_size, ch, num_node, ts]
        input = input.permute(0, 1, 3, 2)
        # print(input[0, :, :5, :3])
        seq_len = input.size(3)
        assert seq_len==self.seq_length, 'input sequence length not equal to preset sequence length'

        if self.seq_length<self.receptive_field:
            input = nn.functional.pad(input,(self.receptive_field-self.seq_length,0,0,0))

        if self.gcn_true:
            if self.buildA_true:
                adp = self.gc(self.idx)
            else:
                adp = self.predefined_A

        x = self.start_conv(input)
        skip = self.skip0(F.dropout(input, self.dropout, training=self.training))
        for i in range(self.layers):
            # when i = 0:
            residual = x
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate
            x = F.dropout(x, self.dropout, training=self.training)
            s = x
            s = self.skip_convs[i](s)
            skip = s + skip
            if self.gcn_true:
                x = self.gconv1[i](x, adp)+self.gconv2[i](x, adp.transpose(1,0))
            else:
                x = self.residual_convs[i](x)
            
            # ?????
            x = x + residual[:, :, :, -x.size(3):]
            x = self.norm[i](x,self.idx)

        skip = self.skipE(x) + skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        # [batch_size, out_dim, n_vertex,]
        x = self.end_conv_2(x)
        x = x.squeeze(dim=-1).transpose(1, 2)
        # [batch_size, n_vertex, pred_len, c_in]
        x = x.reshape(x.size(0), x.size(1), -1, self.in_dim)
        x = x.transpose(1, 2)
        x = x.reshape(x.size(0), x.size(1), -1)
        return x

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast':
            return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        else:
            raise NotImplementedError