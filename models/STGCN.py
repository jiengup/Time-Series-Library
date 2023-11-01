import torch
import torch.nn as nn
from layers.STGCN_Family import STConvBlock, OutputBlock
from layers.Embed import DataEmbedding_only_time

class Model(nn.Module):
    # STGCNChebGraphConv contains 'TGTND TGTND TNFF' structure
    # ChebGraphConv is the graph convolution from ChebyNet.
    # Using the Chebyshev polynomials of the first kind as a graph filter.
        
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (ChebGraphConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # D: Dropout

    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (ChebGraphConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # D: Dropout

    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normalization
    # F: Fully-Connected Layer
    # F: Fully-Connected Layer
    def __init__(self, configs):
        Kt = 3
        Ks = 3
        act_func = "glu"
        blocks = [
            [configs.c_in+1],
            [64, 16, 64],
            [64, 16, 64],
            [128, 64],
            [configs.pred_len*configs.c_in]
        ]
        super(Model, self).__init__()
        # self.dataembedding = DataEmbedding_wo_pos(configs.n_vertex, configs.n_vertex, 
        #                                           configs.embed, configs.freq, configs.dropout)
        self.task_name = configs.task_name
        self.n_vertex = configs.n_vertex
        self.c_in = configs.c_in
        self.pred_len = configs.pred_len

        self.timeembedding = DataEmbedding_only_time(1, configs.embed, configs.freq, configs.dropout)

        modules = []
        for l in range(len(blocks) - 3):
            modules.append(STConvBlock(Kt,Ks, 
                                       configs.n_vertex, 
                                       blocks[l][-1], blocks[l+1], 
                                       act_func, 
                                       graph_conv_type=None,
                                       gso=None,
                                       bias=True,
                                       droprate=configs.dropout))
        self.st_blocks = nn.Sequential(*modules)
        Ko = configs.seq_len - (len(blocks) - 3) * 2 * (Kt - 1)
        self.Ko = Ko
        if self.Ko > 0:
            self.output = OutputBlock(self.Ko, blocks[-3][-1], blocks[-2], blocks[-1][0], configs.n_vertex, act_func, bias=True, droprate=configs.dropout)
        elif self.Ko == 0:
            self.fc1 = nn.Linear(in_features=blocks[-3][-1], out_features=blocks[-2][0])
            self.fc2 = nn.Linear(in_features=blocks[-2][0], out_features=blocks[-1][0])
            self.relu = nn.ReLU()
            self.leaky_relu = nn.LeakyReLU()
            self.silu = nn.SiLU()
            self.dropout = nn.Dropout(p=configs.dropout)
        
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc: [batch_size, seq_len, enc_in]
        batch_size, his_len, _ = x_enc.shape
        x_enc = x_enc.view(batch_size, -1, his_len, self.n_vertex)
        _, c_in, _, _ = x_enc.shape
        assert c_in == self.c_in

        x = self.timeembedding(x_enc, x_mark_enc)
        # x: [batch_size, 1, his_len, num_vertex]
        # x_enc: [batch_size, c_in, his_len, num_vertex]
        x = torch.cat((x_enc, x), dim=1)
        # x: [batch_size, c_in+1, his_len, num_vertex]
        x = self.st_blocks(x)
        if self.Ko > 0:
            x = self.output(x)
        elif self.Ko == 0:
            x = self.fc1(x.permute(0, 2, 3, 1))
            x = self.relu(x)
            x = self.fc2(x).permute(0, 3, 1, 2)
        x = x.reshape(x.shape[0], self.pred_len, -1)
        return x

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast':
            return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        else:
            raise NotImplementedError