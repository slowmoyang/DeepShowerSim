from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch.nn.modules.utils import _pair

import sympy
import math
from collections import Iterable


def get_conv_out_length(in_length,
                        kernel_size,
                        stride=1,
                        padding=0, 
                        dilation=1):
    out_length = ((in_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1 
    out_length = math.floor(out_length)
    return int(out_length)

def get_conv_transpose_length(in_length,
                              kernel_size,
                              stride=1,
                              padding=0,
                              out_padding=0,
                              dilation=1):
    out_length = (in_length - 1) * stride - 2 * padding + kernel_size + out_padding
    return out_length


def _get_conv_padding(in_length,
                     out_length,
                     kernel_size=1,
                     stride=1,
                     dilation=1):
    i = in_length
    o = out_length
    k = kernel_size
    s = stride
    d = dilation

    # padding
    p = sympy.symbols("p")

    eq_low = o - ((i + 2*p - d*(k-1) -1) / s + 1)
    eq_high = o + 1 - ((i + 2*p - d*(k-1) -1) / s + 1)

    p_low = sympy.solve(eq_low)[0]
    p_low = sympy.ceiling(p_low)
    p_low = int(p_low)

    # TODO if which == "lowest"

    return p_low


def get_conv_padding(in_spatial_size,
                      out_spatial_size,
                      kernel_size=1,
                      stride=1,
                      dilation=1):
    
    arguments = [in_spatial_size, out_spatial_size, kernel_size, stride, dilation]
    iterable_exists = any(isinstance(each, Iterable) for each in arguments)
    if iterable_exists:
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        dilation = _pair(dilation)
        zipped_args = zip(in_spatial_size, out_spatial_size, kernel_size, stride, dilation)
        padding = tuple(_get_conv_padding(*args) for args in zipped_args)
    else:
        padding = _get_conv_padding(in_spatial_size, out_spatial_size, kernel_size, dilation)
    return padding

