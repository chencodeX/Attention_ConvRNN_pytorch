# -*- coding: utf-8 -*-
# Copyright (c) 2016 - zihao.chen <zihao.chen@moji.com> 

"""
Author: zihao.chen
Create Date: 2020/4/17
Modify Date: 2020/4/17
descirption:
"""

import torch
from torch import nn
from convgru import ConvGRU, ConvGRUCell
import torch.nn.functional as F


def deconv2_act(inplanes, out_channels=8, kernel_size=7, stride=5, padding=1, bias=True):
    layers = []
    layers += [
        nn.ConvTranspose2d(inplanes, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                           bias=bias)]
    layers += [nn.BatchNorm2d(out_channels)]
    layers += [nn.LeakyReLU(negative_slope=0.2)]
    return nn.Sequential(*layers)


def conv2_act(inplanes, out_channels=8, kernel_size=7, stride=5, padding=1, bias=True):
    layers = []
    layers += [nn.Conv2d(inplanes, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                         bias=bias)]
    layers += [nn.BatchNorm2d(out_channels)]
    layers += [nn.LeakyReLU(negative_slope=0.2, inplace=True)]
    return nn.Sequential(*layers)


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.input_size = input_size
        self.bn1 = nn.BatchNorm2d(8)
        # self.relu1 = nn.Sigmoid()
        self.conv_pre_0 = conv2_act(input_size, out_channels=16, kernel_size=3, stride=1, padding=1)

        self.gruc_0 = ConvGRUCell(16, hidden_size[0], kernel_size[0])

        self.conv_pre_1 = conv2_act(hidden_size[0], out_channels=hidden_size[0], kernel_size=3, stride=2, padding=1)
        self.gruc_1 = ConvGRUCell(hidden_size[0], hidden_size[1], kernel_size[1])

        self.conv_pre_2 = conv2_act(hidden_size[1], out_channels=hidden_size[1], kernel_size=3, stride=2, padding=1)

        self.gruc_2 = ConvGRUCell(hidden_size[1], hidden_size[2], kernel_size[2])

        # self.init()

    def forward(self, input, hidden=None):
        print ('=======encode forward =========')
        input = self.conv_pre_0(input)
        print (input.size())
        hidden[0] = self.gruc_0(input, hidden[0])
        print (hidden[0].size())
        input = self.conv_pre_1(hidden[0])
        print (input.size())
        hidden[1] = self.gruc_1(input, hidden[1])
        print (hidden[1].size())
        input = self.conv_pre_2(hidden[1])
        print (input.size())
        hidden[2] = self.gruc_2(input, hidden[2])
        print (hidden[2].size())
        return hidden


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, input_size, kernel_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.input_size = input_size
        self.bn1 = nn.BatchNorm2d(8)
        # self.relu1 = nn.Sigmoid()
        self.gruc_0 = ConvGRUCell(input_size, hidden_size[0], kernel_size[0])
        self.conv_pre_0 = deconv2_act(hidden_size[0], out_channels=hidden_size[0], kernel_size=4, stride=2, padding=1)

        self.gruc_1 = ConvGRUCell(hidden_size[0], hidden_size[1], kernel_size[1])
        self.conv_pre_1 = deconv2_act(hidden_size[1], out_channels=hidden_size[1], kernel_size=4, stride=2, padding=1)

        self.gruc_2 = ConvGRUCell(hidden_size[1], hidden_size[2], kernel_size[2])
        self.conv_pre_2_0 = conv2_act(hidden_size[2], out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv_pre_2_1 = conv2_act(16, out_channels=1, kernel_size=3, stride=1)

        # self.init()

    def forward(self, input, hidden):
        print ('=======decoder forward =========')
        hidden[2] = self.gruc_0(input, hidden[2])
        print (hidden[2].size())
        input = self.conv_pre_0(hidden[2])
        print (input.size())
        hidden[1] = self.gruc_1(input, hidden[1])
        print (hidden[1].size())
        input = self.conv_pre_1(hidden[1])
        print (input.size())
        hidden[0] = self.gruc_2(input, hidden[0])
        print (hidden[0].size())
        input = self.conv_pre_2_0(hidden[0])
        input = self.conv_pre_2_1(input)

        return input,hidden
