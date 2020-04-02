# -*- coding: utf-8 -*-
# Copyright (c) 2016 - zihao.chen <zihao.chen@moji.com> 

"""
Author: zihao.chen
Create Date: 2020/4/2
Modify Date: 2020/4/2
descirption:
"""
import time
import torch
from torch import nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

work_time = 0


class ConvGRUCell(nn.Module):

    def __init__(self, input_size, hidden_size, kernel_size):
        super(ConvGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.dropout = nn.Dropout(p=0.5)
        self.ConvGates = nn.Conv2d(self.input_size + self.hidden_size, 2 * self.hidden_size, self.kernel_size,
                                   padding=self.kernel_size // 2)
        self.Conv_ct = nn.Conv2d(self.input_size + self.hidden_size, self.hidden_size, self.kernel_size,
                                 padding=self.kernel_size // 2)
        dtype = torch.FloatTensor

    def forward(self, input, hidden):
        if hidden is None:
            # print (input.data.size()[0])
            # print (self.hidden_size)
            # print (list(input.data.size()[2:]))
            size_h = [input.data.size()[0], self.hidden_size] + list(input.data.size()[2:])
            # print size_h
            hidden = torch.zeros(size_h, device=device)

        print (input.size(), input.dtype)
        print (hidden.size(), hidden.dtype)
        print ('=' * 20)
        c1 = self.ConvGates(torch.cat((input, hidden), 1))

        ru = self.dropout(torch.sigmoid(c1))

        (reset_gate, update_gate) = ru.chunk(2, 1)

        # reset_gate = self.dropout(f.sigmoid(rt))
        # update_gate = self.dropout(f.sigmoid(ut))
        gated_hidden = reset_gate * hidden
        ct = torch.tanh(self.Conv_ct(torch.cat((input, gated_hidden), 1)))
        # ct = f.tanh()

        next_h = update_gate * ct + (1 - update_gate) * hidden   # 展开算式
        #
        # next_h = update_gate * ct + hidden - update_gate * hidden

        # next_h = update_gate * (ct - hidden) + hidden  # 节省一次乘法

        return next_h


class ConvGRU(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, layers_num):
        super(ConvGRU, self).__init__()

        self.input_size = input_size

        if type(hidden_size) != list:
            self.hidden_sizes = [hidden_size] * layers_num
        else:
            assert len(hidden_size) == layers_num, '`hidden_size` must have the same length as n_layers'
            self.hidden_sizes = hidden_size

        if type(kernel_size) != list:
            self.kernel_sizes = [kernel_size] * layers_num
        else:
            assert len(kernel_size) == layers_num, '`kernel_size` must have the same length as n_layers'
            self.kernel_sizes = kernel_size

        self.layers_num = layers_num

        cells = []
        for i in range(self.layers_num):
            if i == 0:
                input_dim = self.input_size
            else:
                input_dim = self.hidden_sizes[i - 1]

            cell = ConvGRUCell(input_dim, self.hidden_sizes[i], self.kernel_sizes[i])
            name = 'ConvGRUCell_' + str(i).zfill(2)

            setattr(self, name, cell)
            cells.append(getattr(self, name))

        self.cells = cells

    def forward(self, input, hidden=None):
        """
        :param input: 4D (batch, channels, height, width)
        :param hidden: list of 4D (batch, channels, height, width)
        :return: 5D (layer, batch, channels, height, width)
        """

        if not hidden:
            hidden = [None] * self.layers_num

        update_hiddens = []

        for layers_index in range(self.layers_num):
            cell = self.cells[layers_index]

            cell_hidden = hidden[layers_index]
            update_hidden = cell(input, cell_hidden)
            update_hiddens.append(update_hidden)

            input = update_hidden

        return update_hiddens
