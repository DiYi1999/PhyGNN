import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F


class OneAdjGCN(nn.Module):
    def __init__(self, node_num, residual_alpha=0, LeakyReLU_slope=0.01):
        """
        This is a moudle of G3CN used in the paper, from https://github.com/DiYi1999/G3CN, compared with GCN and GAT, G3CN focuses on multi-sensor time series, especially when the correlation between sensor variables is not "similarity" but "complex nonlinear correlation.
        """

        super(OneAdjGCN, self).__init__()

        self.LeakyReLU_slope=LeakyReLU_slope
        self.residual_alpha=residual_alpha

        self.W = Parameter(torch.Tensor(node_num, node_num))
        self.b = Parameter(torch.Tensor(node_num))
        self.v = Parameter(torch.Tensor(node_num))
        self.reset_parameters()

    def reset_parameters(self):

        torch.nn.init.kaiming_normal_(self.W, nonlinearity='leaky_relu')
        torch.nn.init.normal_(self.b)
        torch.nn.init.normal_(self.v)

    def forward(self, X, A):

        fixed = torch.zeros_like(self.W, requires_grad=False)
        W = A * self.W + (1 - A) * fixed
        H = torch.matmul(W, X) + self.b.unsqueeze(0).unsqueeze(2)
        if self.residual_alpha != 0:
            H = torch.matmul(W, X) + self.residual_alpha * X + self.b.unsqueeze(0).unsqueeze(2)
        H = F.leaky_relu(H, negative_slope=self.LeakyReLU_slope)
        v = self.v.unsqueeze(0).unsqueeze(2)
        H = v * H
        return H


class MAdjGCN(nn.Module):
    def __init__(self, K, node_num, residual_alpha=0, LeakyReLU_slope=0.01):
        """
        This is a moudle of G3CN used in the paper, from https://github.com/DiYi1999/G3CN, compared with GCN and GAT, G3CN focuses on multi-sensor time series, especially when the correlation between sensor variables is not "similarity" but "complex nonlinear correlation.
        """

        super(MAdjGCN, self).__init__()

        self.K = K

        self.MAdjGCNlist = nn.ModuleList([OneAdjGCN(node_num=node_num,
                                                    residual_alpha=residual_alpha,
                                                    LeakyReLU_slope=LeakyReLU_slope) for _ in range(self.K)])
        
        # Some times, when data preprocessing uses StandardScaler(), the data distribution has a mean of 0 and a variance of 1. In this case, using the ReLU activation function in the last layer is not appropriate, as ReLU will turn negative numbers into 0. Therefore, a linear layer needs to be added at the end.
        self.last_linear_w = Parameter(torch.Tensor(node_num))
        self.last_linear_b = Parameter(torch.Tensor(node_num))
        self.reset_parameters()

    def reset_parameters(self):

        # torch.nn.init.xavier_uniform_(self.last_linear_w)
        torch.nn.init.normal_(self.last_linear_w, mean=1.0, std=0.1)
        torch.nn.init.normal_(self.last_linear_b)

    def forward(self, X, A):

        H = [self.MAdjGCNlist[i](X, A) for i in range(self.K)]
        H = torch.stack(H, dim=0).sum(dim=0)
        H = self.last_linear_w.unsqueeze(0).unsqueeze(2) * H + self.last_linear_b.unsqueeze(0).unsqueeze(2)
        return H


class CMTS_GCN(nn.Module):
    def __init__(self, CMTS_GCN_K_nums, node_num, CMTS_GCN_residual=0, LeakyReLU_slope=0.01):
        """
        This is the G3CN used in the paper, from https://github.com/DiYi1999/G3CN, compared with GCN and GAT, G3CN focuses on multi-sensor time series, especially when the correlation between sensor variables is not "similarity" but "complex nonlinear correlation.
        """

        super(CMTS_GCN, self).__init__()

        CMTS_GCN_list = []
        layer_num = len(CMTS_GCN_K_nums)
        for i in range(layer_num):
            CMTS_GCN_list.append(MAdjGCN(CMTS_GCN_K_nums[i],
                                         node_num,
                                         CMTS_GCN_residual,
                                         LeakyReLU_slope))
        self.CMTS_GCN_list = nn.ModuleList(CMTS_GCN_list)

    def forward(self, X, A):

        H = X
        for CMTS_GCN in self.CMTS_GCN_list:
            H = CMTS_GCN(H, A)
        return H



class GCN_layer(nn.Module):
    def __init__(self, node_num, input_len, output_len, LeakyReLU_slope=0.01):

        super(GCN_layer, self).__init__()
        self.LeakyReLU_slope = LeakyReLU_slope

        self.W = Parameter(torch.Tensor(input_len, output_len))
        self.b = Parameter(torch.Tensor(node_num, 1))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.W)
        torch.nn.init.normal_(self.b)

    def forward(self, X, A):

        H = torch.matmul(A, X)
        H = torch.matmul(H, self.W) + self.b
        H = F.leaky_relu(H, negative_slope=self.LeakyReLU_slope)
        return H, A


class GCN_s(nn.Module):
    def __init__(self, GCN_layer_nums, node_num, lag, LeakyReLU_slope=0.01):

        super(GCN_s, self).__init__()
        self.GCN_layer_nums = GCN_layer_nums
        self.lag = lag
        self.LeakyReLU_slope = LeakyReLU_slope
        self.node_num = node_num

        GCN_list = []
        layer_num = len(GCN_layer_nums)
        for i in range(layer_num):
            if i == 0:
                GCN_list.append(GCN_layer(node_num, lag, GCN_layer_nums[i], LeakyReLU_slope))
            else:
                GCN_list.append(GCN_layer(node_num, GCN_layer_nums[i-1], GCN_layer_nums[i], LeakyReLU_slope))

        self.GCN_list = nn.ModuleList(GCN_list)

        if self.GCN_layer_nums[-1] != self.lag:
            self.fc = nn.Linear(GCN_layer_nums[-1], self.lag)
            self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.zeros_(self.fc.bias)

    def forward(self, X, A):
        H = X
        for GCN_layer in self.GCN_list:
            H, A = GCN_layer(H, A)
        if self.GCN_layer_nums[-1] != self.lag:
            H = self.fc(H)

        return H


class Nothing_to_do_S(nn.Module):
    def __init__(self):
        super(Nothing_to_do_S, self).__init__()

    def forward(self, X, A):
        return X


class FeatureAttentionLayer(nn.Module):

    def __init__(self, n_features, window_size, dropout, alpha, embed_dim=None, use_gatv2=True, use_bias=True):

        super(FeatureAttentionLayer, self).__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.dropout = dropout
        self.embed_dim = embed_dim if embed_dim is not None else window_size
        self.use_gatv2 = use_gatv2
        self.num_nodes = n_features
        self.use_bias = use_bias

        if self.use_gatv2:
            self.embed_dim *= 2
            lin_input_dim = 2 * window_size
            a_input_dim = self.embed_dim
        else:
            lin_input_dim = window_size
            a_input_dim = 2 * self.embed_dim

        self.lin = nn.Linear(lin_input_dim, self.embed_dim)
        self.a = nn.Parameter(torch.empty((a_input_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(n_features, n_features))

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        if self.use_gatv2:
            a_input = self._make_attention_input(x)                 # (b, k, k, 2*window_size)
            a_input = self.leakyrelu(self.lin(a_input))             # (b, k, k, embed_dim)
            e = torch.matmul(a_input, self.a).squeeze(3)            # (b, k, k, 1)

        # Original GAT attention
        else:
            Wx = self.lin(x)                                                  # (b, k, k, embed_dim)
            a_input = self._make_attention_input(Wx)                          # (b, k, k, 2*embed_dim)
            e = self.leakyrelu(torch.matmul(a_input, self.a)).squeeze(3)      # (b, k, k, 1)

        if self.use_bias:
            e += self.bias

        # Attention weights
        attention = torch.softmax(e, dim=2)                          # (b, k, k, 1)
        attention = torch.dropout(attention, self.dropout, train=self.training)

        # Computing new node features using the attention
        h = self.sigmoid(torch.matmul(attention, x))                 # (b, k, n)

        return h

    def _make_attention_input(self, v):
        """
        Preparing the feature attention mechanism.
        Creating matrix with all possible combinations of concatenations of node.
        Each node consists of all values of that node within the window
            v1 || v1,
            ...
            v1 || vK,
            v2 || v1,
            ...
            v2 || vK,
            ...
            ...
            vK || v1,
            ...
            vK || vK,

        Proposed by Brody et. al., 2021 (https://arxiv.org/pdf/2105.14491.pdf) Section 3.3
        """

        K = self.num_nodes
        blocks_repeating = v.repeat_interleave(K, dim=1)  # Left-side of the matrix
        blocks_alternating = v.repeat(1, K, 1)  # Right-side of the matrix
        combined = torch.cat((blocks_repeating, blocks_alternating), dim=2)  # (b, K*K, 2*window_size)

        if self.use_gatv2:
            return combined.view(v.size(0), K, K, 2 * self.window_size)
        else:
            return combined.view(v.size(0), K, K, 2 * self.embed_dim)


class Muti_S_GAT(nn.Module):
    def __init__(self, Muti_S_GAT_K, Muti_S_GAT_embed_dim, node_num, lag, use_gatv2=True, dropout=0.0, LeakyReLU_slope=0.01):
        """
        GATv2和GAT在空间维度的多头, 输入X: (batch_size, node_num, lag), 输出H: (batch_size, node_num, lag)

        Args:
            Muti_S_GAT_K: 多头数
            Muti_S_GAT_embed_dim: GAT编码空间 高维表示向量维度，变的是lag那个维度
            node_num: 节点数
            lag: 滑窗大小
            use_gatv2:
            dropout:
            LeakyReLU_slope:
        """
        super(Muti_S_GAT, self).__init__()
        self.Muti_S_GAT_K = Muti_S_GAT_K

        self.GAT_list = nn.ModuleList([FeatureAttentionLayer(n_features=node_num,
                                                             window_size=lag,
                                                             dropout=dropout,
                                                             alpha=LeakyReLU_slope,
                                                             embed_dim=Muti_S_GAT_embed_dim,
                                                             use_gatv2=use_gatv2,
                                                             use_bias=False)
                                       for _ in range(Muti_S_GAT_K)])

    def forward(self, X, A):
        H = [self.GAT_list[i](X) for i in range(self.Muti_S_GAT_K)]
        H = torch.stack(H, dim=0).mean(dim=0)
        return H


























