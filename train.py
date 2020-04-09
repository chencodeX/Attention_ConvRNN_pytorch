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

batch_size = 4


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    # encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)

    decoder_input = encoder_output

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            # topv, topi = decoder_output.topk(1)
            # decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            # if decoder_input.item() == EOS_token:
            #     break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def trainIters(encoder, decoder, n_epoch, pairs, print_every=1000, plot_every=100, learning_rate=0.001):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    n_iters = n_epoch * len(pairs) / batch_size
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    # training_pairs = [tensorsFromPair(random.choice(pairs))
    #                   for i in range(n_iters)]
    # criterion = nn.NLLLoss()
    criterion = nn.MSELoss()

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
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    # showPlot(plot_losses)


# hidden_size = 256

# input_channels = 1
mnist_moving_dataset = np.load('/home/meteo/zihao.chen/jupyter/mnist_test_seq.npy')

mnist_moving_dataset.transpose([1, 0, 2, 3])

mnist_moving_dataset = mnist_moving_dataset[0, 1, np.newaxis, 2, 3]

print (mnist_moving_dataset.shape)

data_length = len(mnist_moving_dataset)
nn_i = range(data_length)
nn_i = np.random.shuffle(nn_i)
mnist_moving_dataset = mnist_moving_dataset[nn_i]

train_pairs = mnist_moving_dataset[:int(data_length * 0.8)]

test_pairs = mnist_moving_dataset[int(data_length * 0.8):]

encoder1 = EncoderRNN(input_size=1, hidden_size=[32, 64, 16], kernel_size=[3, 5, 3], layers_num=3).to(device)

decoder1 = DecoderRNN(output_size=1, hidden_size=[32, 64, 16], kernel_size=[3, 5, 3], layers_num=3).to(device)

trainIters(encoder1, decoder1, n_epoch=4, pairs=train_pairs, print_every=300)
