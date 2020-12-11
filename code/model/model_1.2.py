import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch_scatter import scatter
from torch.autograd import Variable

import sys
import numpy as np
import pdb
import math

import pdb

# from tensorflow import math
class PairNorm(torch.nn.Module):
    def __init__(
        self, scale: float = 1.0, scale_individually: bool = False, eps: float = 1e-5
    ):
        super(PairNorm, self).__init__()

        self.scale = scale
        self.scale_individually = scale_individually
        self.eps = eps

    def forward(self, x) -> Tensor:
        scale = self.scale

        x = x - x.mean(dim=0, keepdim=True)

        if not self.scale_individually:
            return scale * x / (self.eps + x.pow(2).sum(-1).mean()).sqrt()
        else:
            return scale * x / (self.eps + x.norm(2, -1, keepdim=True))

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class nconv_original(nn.Module):
    def __init__(self):
        super(nconv_original, self).__init__()

    # AX를 받아서 PX를 만들어줌
    # P = A/rowsum(A)
    def forward(self, x, A):
        # einsum(): sum by column
        x = torch.einsum("ncvl, vw -> ncwl", (x, A))

        # 어떤 함수 결과가 실제로 메모리에도 우리가 기대하는 순서로 유지하려면
        # .contiguous()를 사용하여 에러가 발생하는 것을 방지할 수 있습니다.

        # x.shape == [64, 32, 207, 12~1]
        # (batch, feature, node, window)
        return x.contiguous()


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):

        input_x = x

        x = x / torch.norm(x, dim=1)[:, None, :, :]
        x = torch.einsum("bfnw, bfmw -> bnmw", (x, x))  # S_ij
        x = torch.mean(x, dim=0)
        x = torch.mean(x, dim=2)
        rowsum = x.sum(1).reshape(-1, 1)
        new_A = torch.div(x, rowsum)

        result = torch.einsum("ncvl, vw -> ncwl", (input_x, new_A * A))

        return result.contiguous()


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
        # self.pair_norm = PairNorm()
        torch.nn.init.xavier_uniform_(self.conv.weight)

        # print(input_dims, output_dims)

        if use_bias:
            torch.nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)
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
            x = F.relu_(x)
        return x.permute(0, 3, 2, 1)


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
        # print(x.shape)
        for conv in self.convs:
            x = conv(x)  ## 32->16
        # print(x.shape)
        return x


class ST_Attention(nn.Module):
    def __init__(self, K, d):
        super(ST_Attention, self).__init__()
        D = K * d
        self.d = d
        self.K = K
        # self.FC_q = FC(input_dims=2 * D, units=D, activations=F.relu)
        # self.FC_k = FC(input_dims=2 * D, units=D, activations=F.relu)
        # self.FC_v = FC(input_dims=2 * D, units=D, activations=F.relu)
        # self.FC = FC(input_dims=D, units=D, activations=F.relu)

        self.FC_q = FC(input_dims=2 * D, units=2 * D, activations=F.relu)
        # pdb.set_trace()
        self.FC_k = FC(input_dims=2 * D, units=2 * D, activations=F.relu)
        self.FC_v = FC(input_dims=2 * D, units=2 * D, activations=F.relu)

        self.FC = FC(input_dims=2 * D, units=2 * D, activations=F.relu)

    def forward(self, X):
        # (batch, feature, node, window)
        # x.shape == [64, 32, 207, 12~1] -> [64, 12~1, 207, 32]
        X = torch.einsum("bfnw -> bwnf", (X))

        batch_size = X.shape[0]
        # X = torch.cat((X, STE), dim=-1)

        # [batch_size, num_step, num_vertex, K * d]
        query = self.FC_q(X)  #
        key = self.FC_k(X)
        value = self.FC_v(X)
        # [K * batch_size, num_step, num_vertex, d]
        query = torch.cat(torch.split(query, self.K, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.K, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.K, dim=-1), dim=0)
        # [K * batch_size, num_step, num_vertex, num_vertex]
        attention = torch.matmul(query, key.transpose(2, 3))
        attention /= self.d ** 0.5
        attention = F.softmax(attention, dim=-1)
        # [batch_size, num_step, num_vertex, D]
        X = torch.matmul(attention, value)
        X = torch.cat(
            torch.split(X, batch_size, dim=0), dim=-1
        )  # orginal K, change to batch_size
        X = self.FC(X)
        del query, key, value, attention

        X = torch.einsum("bwnf -> bfnw", (X))

        return X


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        # 1*1 conv
        self.mlp = torch.nn.Conv2d(
            c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True
        )

    def forward(self, x):
        return self.mlp(x)


class gcn(
    nn.Module
):  # gcn(dilation_channels,residual_channels,dropout,support_len=self.supports_len))
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2, n_nodes=207):
        super(gcn, self).__init__()
        device = torch.device("cuda:0")
        self.nconv = nconv().to(device)
        self.nconv_original = nconv_original()
        # K: number of attention heads default=8
        # d: dimension of each attention outputs default=8
        # (2 x 3 + 1) x c_in = 7 x c_in
        c_in = (order * support_len + 1) * c_in
        # c_in = (order * 2 + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        # order == K
        self.order = order

    def forward(self, x, support):
        out = [x]

        for a in support:
            x1 = self.nconv_original(x, a)
            out.append(x1)

            for k in range(2, self.order + 1):
                x2 = self.nconv_original(x1, a)
                out.append(x2)
                x1 = x2

        # for idx, a in enumerate(support):
        #     if (idx !=2 ): # fw, bw
        #         x1 = self.nconv(x,a)
        #         out.append(x1)
        #         for k in range(2, self.order + 1):
        #             x2 = self.nconv(x1,a)
        #             out.append(x2)

        #     else: #self-adaptive
        #         x1 = self.nconv_original(x,a)
        #         out.append(x1)
        #         for k in range(2, self.order + 1):
        #             x2 = self.nconv_original(x1,a)
        #             out.append(x2)

        h = torch.cat(out, dim=1)

        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


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
        self.gcn_bool = gcn_bool
        self.att_bool = att_bool
        self.addaptadj = addaptadj

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()

        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

        self.bn = nn.ModuleList()
        self.pn = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.att_conv = nn.ModuleList()

        self.start_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=residual_channels, kernel_size=(1, 1)
        )
        self.supports = supports

        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if gcn_bool and addaptadj:
            if aptinit is None:
                if supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(
                    torch.randn(num_nodes, 10).to(device), requires_grad=True
                ).to(device)
                self.nodevec2 = nn.Parameter(
                    torch.randn(10, num_nodes).to(device), requires_grad=True
                ).to(device)
                self.supports_len += 1
            else:
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
                self.supports_len += 1

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
                self.pn.append(PairNorm())
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(
                        gcn(
                            dilation_channels,
                            residual_channels,
                            dropout,
                            support_len=self.supports_len,
                        )
                    )

                if self.att_bool:
                    self.att_conv.append(ST_Attention(8, 8))

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

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

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

            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                    x = self.pn[i](x)
                    if self.att_bool:
                        x = self.att_conv[i](x)
                else:
                    x = self.gconv[i](x, self.supports)
                    x = self.pn[i](x)
                    if self.att_bool:
                        x = self.att_conv[i](x)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3) :]

            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x

