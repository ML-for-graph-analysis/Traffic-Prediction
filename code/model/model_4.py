import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.autograd import Variable

import sys
import numpy as np
import pdb
import math

import pdb


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum("ncvl,vw->ncwl", (x, A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(
            c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True
        )

    def forward(self, x):
        return self.mlp(x)


class AVWGCN(nn.Module):
    def __init__(
        self, dim_in, dim_out, dropout, cheb_k=3, embed_dim=10, support_len=1, order=2
    ):
        super(AVWGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(
            torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out)
        )
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))

        self.nconv = nconv()
        dim_in = (order * support_len + 1) * dim_in
        self.mlp = linear(dim_in, dim_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, node_embeddings, support=None):
        # x shaped[B, C, N, Window], node_embeddings shaped [N, D] -> supports shaped [N, N]
        # output shape [B, C, N, Window]
        # d: embed dim, n: node, k: cheb_k, c,i: dim_in, o: dim_out, b: batch, w: window
        node_num = node_embeddings.shape[0]
        support_emb = F.softmax(
            F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1
        )
        support_set = [torch.eye(node_num).to(support_emb.device), support_emb]
        # default cheb_k = 3
        for k in range(2, self.cheb_k):
            support_set.append(
                torch.matmul(2 * support_emb, support_set[-1]) - support_set[-2]
            )
        support_emb = torch.stack(support_set, dim=0)
        weights = torch.einsum(
            "nd,dkio->nkio", node_embeddings, self.weights_pool
        )  # N, cheb_k, dim_in, dim_out
        bias = torch.matmul(node_embeddings, self.bias_pool)  # N, dim_out
        x_g = torch.einsum("knm,bcmw->bcknw", support_emb, x)  # B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 4, 3, 2, 1)  # B, window, N, cheb_k, c_in
        x_avwgcn = torch.einsum("bwnki,nkio->bwno", x_g, weights) + bias
        x_avwgcn = x_avwgcn.permute(0, 3, 2, 1)

        x_gcn = None
        if support is not None:
            out = [x]
            for a in support:
                x1 = self.nconv(x, a)
                out.append(x1)
                for k in range(2, self.order + 1):
                    x2 = self.nconv(x1, a)
                    out.append(x2)
                    x1 = x2

            x_gcn = torch.cat(out, dim=1)
            x_gcn = self.mlp(x_gcn)
            x_gcn = F.dropout(x_gcn, self.dropout, training=self.training)

        if x_gcn is not None:
            return [x_avwgcn, x_gcn]
        else:
            return x_avwgcn


class conv2d_(nn.Module):
    def __init__(
        self,
        input_dims,
        output_dims,
        kernel_size,
        stride=(1, 1),
        padding="SAME",
        use_bias=True,
        activation=F.relu,
    ):
        super(conv2d_, self).__init__()
        self.activation = activation
        if padding == "SAME":
            self.padding_size = math.ceil(kernel_size)
        else:
            self.padding_size = [0, 0]
        self.conv = nn.Conv2d(
            input_dims,
            output_dims,
            kernel_size,
            stride=stride,
            padding=0,
            bias=use_bias,
        )
        self.batch_norm = nn.BatchNorm2d(output_dims)
        torch.nn.init.xavier_uniform_(self.conv.weight)

        if use_bias:
            torch.nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        x = F.pad(
            x,
            (
                [
                    self.padding_size[1],
                    self.padding_size[1],
                    self.padding_size[0],
                    self.padding_size[0],
                ]
            ),
        )
        x = self.conv(x)
        x = self.batch_norm(x)
        # x = self.pair_norm(x)
        if self.activation is not None:
            x = F.relu(x)
        return x


class FC(nn.Module):
    def __init__(self, input_dims, units, activations, use_bias=True):
        super(FC, self).__init__()
        if isinstance(units, int):
            units = [units]
            input_dims = [input_dims]
            activations = [activations]
        elif isinstance(units, tuple):
            units = list(units)
            input_dims = list(input_dims)
            activations = list(activations)
        assert type(units) == list
        self.convs = nn.ModuleList(
            [
                conv2d_(
                    input_dims=input_dim,
                    output_dims=num_unit,
                    kernel_size=[1, 1],
                    stride=[1, 1],
                    padding="VALID",
                    use_bias=use_bias,
                    activation=activation,
                )
                for input_dim, num_unit, activation in zip(
                    input_dims, units, activations
                )
            ]
        )

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)  ## 32->16
        return x


class ST_Attention(nn.Module):
    def __init__(self, K, d, c_in):
        super(ST_Attention, self).__init__()
        D = K * d
        self.d = d
        self.K = K
        self.FC_q = FC(input_dims=c_in, units=2 * D, activations=F.relu)
        self.FC_k = FC(
            input_dims=c_in, units=2 * D, activations=F.relu
        )  # the size of distanc adj is 1 and head is 2
        self.FC_v = FC(input_dims=c_in, units=2 * D, activations=F.relu)
        self.FC = FC(input_dims=2 * D, units=c_in, activations=F.relu)

    def forward(self, x):
        # (batch, feature, node, window)
        # x.shape == [64, 32, 207, 12~1]
        batch_size = x[0].shape[0]

        query = self.FC_q(x[0])  # x[0]: random adj
        key = self.FC_k(x[1])  # x[1]: distance adj
        value = self.FC_v(x[0])

        query = torch.cat(torch.split(query, self.K, dim=1), dim=0)
        key = torch.cat(torch.split(key, self.K, dim=1), dim=0)
        value = torch.cat(torch.split(value, self.K, dim=1), dim=0)
        # [K * batch_size, num_step, num_vertex, num_vertex]
        attention = torch.matmul(query, key.transpose(2, 3))
        attention /= self.d ** 0.5
        attention = F.softmax(attention, dim=1)
        # [batch_size, num_step, num_vertex, D]
        x = torch.matmul(attention, value)
        x = torch.cat(
            torch.split(x, batch_size, dim=0), dim=1
        )  # orginal K, change to batch_size
        x = self.FC(x)
        del query, key, value, attention

        return x


class gwnet(nn.Module):
    def __init__(
        self,
        device,
        num_nodes,
        dropout=0.3,
        supports=None,
        gcn_bool=True,
        att_bool=True,
        addaptadj=True,
        aptinit=None,
        in_dim=2,
        out_dim=12,
        residual_channels=32,
        dilation_channels=32,
        skip_channels=256,
        end_channels=512,
        kernel_size=2,
        blocks=4,
        layers=2,
    ):
        super(gwnet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = True
        self.att_bool = True
        self.addaptadj = True

        self.supports = supports
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()

        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

        self.bn = nn.ModuleList()
        self.avwgconv = nn.ModuleList()
        self.att_conv = nn.ModuleList()

        self.start_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=residual_channels, kernel_size=(1, 1)
        )

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        receptive_field = 1

        if self.gcn_bool and self.addaptadj:
            self.node_embedding = nn.Parameter(
                torch.randn(num_nodes, 10), requires_grad=True
            )

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(
                    nn.Conv2d(
                        in_channels=residual_channels,
                        out_channels=dilation_channels,
                        kernel_size=(1, kernel_size),
                        dilation=new_dilation,
                    )
                )

                self.gate_convs.append(
                    nn.Conv1d(
                        in_channels=residual_channels,
                        out_channels=dilation_channels,
                        kernel_size=(1, kernel_size),
                        dilation=new_dilation,
                    )
                )

                # 1x1 convolution for residual connection
                self.residual_convs.append(
                    nn.Conv1d(
                        in_channels=dilation_channels,
                        out_channels=residual_channels,
                        kernel_size=(1, 1),
                    )
                )

                # 1x1 convolution for skip connection
                self.skip_convs.append(
                    nn.Conv1d(
                        in_channels=dilation_channels,
                        out_channels=skip_channels,
                        kernel_size=(1, 1),
                    )
                )
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    if (i + 1) % 2 == 1:
                        self.avwgconv.append(
                            AVWGCN(
                                dilation_channels,
                                residual_channels,
                                dropout,
                                support_len=self.supports_len,
                            )
                        )
                        self.att_conv.append(
                            ST_Attention(2, residual_channels, residual_channels)
                        )
                    else:
                        self.avwgconv.append(
                            AVWGCN(dilation_channels, residual_channels, dropout)
                        )

        self.end_conv_1 = nn.Conv2d(
            in_channels=skip_channels,
            out_channels=end_channels,
            kernel_size=(1, 1),
            bias=True,
        )

        self.end_conv_2 = nn.Conv2d(
            in_channels=end_channels,
            out_channels=out_dim,
            kernel_size=(1, 1),
            bias=True,
        )

        self.receptive_field = receptive_field

    def forward(self, input):
        in_len = input.size(3)
        if in_len < self.receptive_field:
            x = nn.functional.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            # (dilation, init_dilation) = self.dilations[i]

            # residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :, -s.size(3) :]
            except:
                skip = 0
            skip = s + skip

            if self.gcn_bool:
                if self.addaptadj:
                    if (i + 1) % 2 == 1:
                        x = self.avwgconv[i](x, self.node_embedding, self.supports)
                        x = self.att_conv[int(i / 2)](x)
                    else:
                        x = self.avwgconv[i](x, self.node_embedding)

            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3) :]

            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x

