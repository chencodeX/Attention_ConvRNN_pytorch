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
    layers += [nn.LeakyReLU(negative_slope=0.2)]
    return nn.Sequential(*layers)


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, layers_num):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.input_size = input_size
        self.relu1 = [nn.BatchNorm2d(8)] + [nn.ReLU()]
        self.gru = ConvGRU(input_size, hidden_size, kernel_size, layers_num)
        self.conv_pre = nn.Conv2d(in_channels=hidden_size[-1], out_channels=8, kernel_size=3, stride=1, padding=1,
                                  bias=True)
        self.relu2 = [nn.BatchNorm2d(1)] + [nn.ReLU()]
        self.conv_pre1 = nn.Conv2d(in_channels=8, out_channels=input_size, kernel_size=1, stride=1,
                                  padding=0, bias=True)

    def forward(self, input, hidden=None):
        hiddens = self.gru(input, hidden)
        output = self.conv_pre(hiddens[-1])
        output = self.relu1(output)
        output = self.conv_pre1(output)
        output = self.relu2(output)
        return output, hiddens

    # init in ConvGRUCell
    # def initHidden(self):
    #     return torch.zeros(1, 1, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, kernel_size, layers_num):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.output_size = output_size
        # self.relu = nn.ReLU()
        self.relu1 = [nn.BatchNorm2d(8)] + [nn.ReLU()]
        self.gru = ConvGRU(output_size, hidden_size, kernel_size, layers_num)
        self.conv_pre = nn.Conv2d(in_channels=hidden_size[-1], out_channels=8, kernel_size=3, stride=1,
                                  padding=1, bias=True)
        self.relu2 = [nn.BatchNorm2d(1)] + [nn.ReLU()]
        self.conv_pre1 = nn.Conv2d(in_channels=8, out_channels=output_size, kernel_size=1, stride=1,
                                  padding=0, bias=True)

    def forward(self, input, hidden):
        hiddens = self.gru(input, hidden)
        output = self.relu(self.conv_pre(hiddens[-1]))

        output = self.relu(self.conv_pre1(output))
        return output, hiddens

    # def initHidden(self):
    #     return torch.zeros(1, 1, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, kernel_size, layers_num, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.output_size = output_size
        self.dropout_p = dropout_p

        self.attn = nn.Conv2d(self.output_size + self.hidden_size[0], 2 * self.hidden_size[0], self.kernel_size[0],
                              padding=self.kernel_size[0] // 2)

        self.attn_combine = nn.Conv2d(self.output_size + self.hidden_size[0], 2 * self.hidden_size[0],
                                      self.kernel_size[0],
                                      padding=self.kernel_size[0] // 2)

        self.dropout = nn.Dropout(p=self.dropout_p)

        self.relu = nn.ReLU()
        self.gru = ConvGRU(output_size, hidden_size, kernel_size, layers_num)
        self.conv_pre = nn.Conv2d(in_channels=hidden_size[-1], out_channels=8, kernel_size=3, stride=1,
                                  padding=1, bias=True)
        self.conv_pre1 = nn.Conv2d(in_channels=8, out_channels=output_size, kernel_size=1, stride=1,
                                  padding=0, bias=True)

    def forward(self, input, hiddens, encoder_outputs):
        batch_size = input.size()[0]

        # input = self.dropout(input.view(batch_size, 1, 1, -1))
        input = self.dropout(input)
        print (input.size())
        print (hiddens[0].size())
        attn_weights = F.softmax(
            self.attn(torch.cat((input, hiddens[0][:]), 1)), dim=1
        )

        # 维度可能还有问题
        attn_applied = torch.bmm(attn_weights,
                                 encoder_outputs)

        output = torch.cat((input, attn_applied), 1)
        output = self.attn_combine(output)
        output = self.relu(output)
        hiddens = self.gru(output, hiddens)
        output = self.relu(self.conv_pre(hiddens[-1]))
        output = self.conv_pre(output)

        return output, hiddens, attn_weights
