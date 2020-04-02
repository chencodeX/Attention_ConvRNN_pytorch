# -*- coding: utf-8 -*-
# Copyright (c) 2016 - zihao.chen <zihao.chen@moji.com> 

"""
Author: zihao.chen
Create Date: 2020/4/2
Modify Date: 2020/4/2
descirption: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
"""
import torch
from torch import nn
from convgru import ConvGRU


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
    layers += [nn.LeakyReLU(negative_slope=0.2)]
    return nn.Sequential(*layers)


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, layers_num):
        super(EncoderRNN, self).__init__()
        # self.hidden_size = hidden_size

        self.gru = ConvGRU(input_size, hidden_size, kernel_size, layers_num)
        self.conv_pre = nn.Conv2d(in_channels=hidden_size[-1], out_channels=1, kernel_size=3, stride=1, padding=1,
                                  bias=True)

    def forward(self, input, hidden=None):
        hiddens = self.gru(input, hidden)
        # output = hiddens[-1]
        output = self.conv_pre(hiddens[-1])
        return output, hiddens

    # init in ConvGRUCell
    # def initHidden(self):
    #     return torch.zeros(1, 1, self.hidden_size, device=device)
