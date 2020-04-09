# -*- coding: utf-8 -*-
# Copyright (c) 2016 - zihao.chen <zihao.chen@moji.com> 

"""
Author: zihao.chen
Create Date: 2020/4/8
Modify Date: 2020/4/8
descirption:
"""
import math
import time


def pixelshuffle_invert(x, factor_hw):
    """

    :param x: B*C*H*W
    :param factor_hw:
    :return:
    """
    pH = factor_hw[0]
    pW = factor_hw[1]
    y = x
    B, iC, iH, iW = y.shape
    oC, oH, oW = iC * (pH * pW), iH // pH, iW // pW
    y = y.reshape(B, iC, oH, pH, oW, pW)
    y = y.permute(0, 1, 3, 5, 2, 4)  # B, iC, pH, pW, oH, oW
    y = y.reshape(B, oC, oH, oW)
    return y


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))
