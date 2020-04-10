# -*- coding: utf-8 -*-
# Copyright (c) 2016 - zihao.chen <zihao.chen@moji.com> 

"""
Author: zihao.chen
Create Date: 2020/4/9
Modify Date: 2020/4/9
descirption:
"""
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import time
from Seq2Seq import EncoderRNN, DecoderRNN
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

teacher_forcing_ratio = 0.5

batch_size = 32


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        # m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(1)
    target_length = target_tensor.size(1) - 1

    # encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    encoder_hidden = None
    loss = 0
    print("====encoder=====")
    for ei in range(input_length - 1):
        encoder_output, encoder_hidden = encoder(
            input_tensor[:, ei], encoder_hidden)
        # print (encoder_hidden[0].mean())
        # print (encoder_output.mean())
        # print (input_tensor[:, ei + 1].mean())
        loss += criterion(encoder_output, input_tensor[:, ei + 1])
    encoder_output, encoder_hidden = encoder(input_tensor[:, 9], encoder_hidden)
    decoder_input = encoder_output

    decoder_hidden = encoder_hidden
    # print (encoder_hidden[0].mean())
    # print (encoder_output.mean())
    # print (target_tensor[:, 0].mean())
    loss += criterion(encoder_output, target_tensor[:, 0])

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    # print("====decoder=====")
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[:, di + 1])
            decoder_input = target_tensor[:, di]  # Teacher forcing
            # print (decoder_hidden[0].mean())
            # print (decoder_output.mean())
            # print (target_tensor[:, di + 1].mean())
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            # topv, topi = decoder_output.topk(1)
            decoder_input = decoder_output  # detach from history as input
            # print (decoder_hidden[0].mean())
            # print (decoder_output.mean())
            # print (target_tensor[:, di + 1].mean())
            loss += criterion(decoder_output, target_tensor[:, di + 1])
            # if decoder_input.item() == EOS_token:
            #     break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def trainIters(encoder, decoder, n_epoch, pairs, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    n_iters = n_epoch * len(pairs) // batch_size
    # encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08,
                                         weight_decay=0,
                                         amsgrad=False)

    # decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08,
                                         weight_decay=0,
                                         amsgrad=False)

    # training_pairs = [tensorsFromPair(random.choice(pairs))
    #                   for i in range(n_iters)]
    # criterion = nn.NLLLoss()
    criterion = nn.MSELoss()
    # criterion = nn.L1Loss()

    for iter in range(1, n_iters + 1):
        training_pair = [random.choice(pairs) for i in range(batch_size)]
        training_pair = torch.tensor(training_pair).to(device)
        input_tensor = training_pair[:, :10]
        target_tensor = training_pair[:, 10:]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            adjust_learning_rate(encoder_optimizer, learning_rate, iter)
            adjust_learning_rate(decoder_optimizer, learning_rate, iter)
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    # showPlot(plot_losses)


def evaluate(input_tensor, target_tensor, encoder, decoder ):
    with torch.no_grad():
        encoder_hidden = None
        print (input_tensor.size())
        input_length = input_tensor.size(1)
        target_length = target_tensor.size(1) - 1

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[:,ei],
                                                     encoder_hidden)
        result = []
        decoder_input = encoder_output
        print (decoder_input.size())
        result.append(encoder_output)
        decoder_hidden = encoder_hidden

        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            result.append(decoder_output)
            decoder_input = decoder_output

        return result


def adjust_learning_rate(optimizer, lr, iter):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr_ = lr * (0.5 ** (iter // 600))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_


if __name__ == '__main__':
    # hidden_size = 256

    # input_channels = 1
    mnist_moving_dataset = np.load('/home/meteo/zihao.chen/jupyter/mnist_test_seq.npy').astype(np.float32)

    mnist_moving_dataset /= 255.

    mnist_moving_dataset *= 2.
    #
    mnist_moving_dataset -= 1

    mnist_moving_dataset -= mnist_moving_dataset.mean()

    mnist_moving_dataset = mnist_moving_dataset.transpose([1, 0, 2, 3])

    mnist_moving_dataset = mnist_moving_dataset[:, :, np.newaxis, :, :]

    print (mnist_moving_dataset.shape)

    data_length = len(mnist_moving_dataset)
    nn_i = list(range(data_length))
    np.random.shuffle(nn_i)
    mnist_moving_dataset = mnist_moving_dataset[nn_i]

    train_pairs = mnist_moving_dataset[:int(data_length * 0.8)]

    test_pairs = mnist_moving_dataset[int(data_length * 0.8):]

    encoder1 = EncoderRNN(input_size=1, hidden_size=[32, 64, 32], kernel_size=[3, 5, 3], layers_num=3).to(device)

    decoder1 = DecoderRNN(output_size=1, hidden_size=[32, 64, 32], kernel_size=[3, 5, 3], layers_num=3).to(device)

    trainIters(encoder1, decoder1, n_epoch=15, pairs=train_pairs, print_every=10)
