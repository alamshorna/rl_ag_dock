#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import sys
import datetime
import math
import torch
import torch.nn.functional as F


LEVELS = ['TRACE', 'DEBUG', 'INFO', 'WARN', 'ERROR']
LEVELS_MAP = None


def init_map():
    global LEVELS_MAP, LEVELS
    LEVELS_MAP = {}
    for idx, level in enumerate(LEVELS):
        LEVELS_MAP[level] = idx


def get_prio(level):
    global LEVELS_MAP
    if LEVELS_MAP is None:
        init_map()
    return LEVELS_MAP[level.upper()]


def print_log(s, level='INFO', end='\n', no_prefix=False):
    pth_prio = get_prio(os.getenv('LOG', 'INFO'))
    prio = get_prio(level)
    if prio >= pth_prio:
        if not no_prefix:
            now = datetime.datetime.now()
            prefix = now.strftime("%Y-%m-%d %H:%M:%S") + f'::{level.upper()}::'
            print(prefix, end='')
        print(s, end=end)
        sys.stdout.flush()
        

def segment_softmax(src: torch.Tensor, index: torch.Tensor, num_segments: int, eps: float = 1e-16):
    """
    A PyTorch-only replacement for torch_scatter.scatter_softmax.

    Args:
        src:   [E] values (e.g., attention scores per edge)
        index: [E] segment indices (e.g., row indices for edges)
        num_segments: number of segments (e.g., number of nodes)
        eps: small constant for numerical stability

    Returns:
        out: [E] softmax-normalized values per segment.
              For all edges e with index[e] == i, out[e] is softmax over that group.
    """
    # Ensure long dtype for index
    index = index.long()

    # 1) Compute per-segment max for numerical stability
    # Initialize with -inf so that scatter_reduce_ does amax correctly
    max_per_seg = torch.full(
        (num_segments,),
        -float("inf"),
        device=src.device,
        dtype=src.dtype,
    )
    # scatter_reduce_ is available in recent PyTorch versions (1.12+).
    # It does: max_per_seg[i] = max(max_per_seg[i], src[e] for all e with index[e] == i)
    max_per_seg.scatter_reduce_(
        dim=0,
        index=index,
        src=src,
        reduce="amax",
        include_self=True,
    )

    # 2) Shift by segment-wise max, exponentiate
    src_exp = torch.exp(src - max_per_seg[index])

    # 3) Compute per-segment denominator (sum of exp)
    denom = torch.zeros(
        (num_segments,),
        device=src.device,
        dtype=src.dtype,
    )
    denom.scatter_add_(0, index, src_exp)

    # 4) Normalize
    out = src_exp / (denom[index] + eps)
    return out
