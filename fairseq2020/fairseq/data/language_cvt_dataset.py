# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import numpy as np
import torch

from fairseq.data import data_utils, FairseqDataset


logger = logging.getLogger(__name__)

from fairseq.data.denoising_dataset import DenoisingDataset
from fairseq.data.denoising_dataset import collate as collate_deno
from fairseq.data.language_cloze_dataset import LanguageClozeDataset


def collate_cloze(
    samples,
    pad_idx,
    eos_idx,
    bos_idx,
    original_eos_idx,
    mask_idx,
    left_pad_source=True,
    left_pad_target=False,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge('source', left_pad=left_pad_source)
    # sort by descending source length
    src_lengths = torch.LongTensor([
        s['source'].ne(pad_idx).long().sum() for s in samples
    ])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    #AA sort of src moved from here to below as might need original orders first

    target = merge('target', left_pad=left_pad_target)
    target = target.index_select(0, sort_order)
    tgt_lengths = torch.LongTensor([
        s['target'].ne(pad_idx).long().sum() for s in samples
    ]).index_select(0, sort_order)
    ntokens = tgt_lengths.sum().item()

    #block_pos_left, block_pos_right, block_masks, maskToken_pos, maskToken_masks = data_utils.collate_mask_tokens(
    #        [s['block'] for s in samples], src_tokens, pad_idx, mask_idx,
    #        original_eos_idx, bos_idx )

    #AA sorting here now after use
    src_tokens = src_tokens.index_select(0, sort_order)

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
            #'block_pos_left': block_pos_left,
            #'block_pos_right': block_pos_right,
            #'block_mask': block_masks,
            #'maskToken_pos' : maskToken_pos,
            #'maskToken_masks': maskToken_masks,
        },
        'target': target,
    }

    return batch


def collate_dyn_targets(targets,
    pad_idx,
    eos_idx,
    left_pad_target=False,):

    def merge(samples, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    target = merge(targets, left_pad=left_pad_target) ## lets see the shape of targets coming here...

    # we create a shifted version of targets for feeding the
    # previous output token(s) into the next decoder step
    prev_output_tokens = merge(
        targets,
        left_pad=left_pad_target,
        move_eos_to_beginning=True,
    )

    tgt_lengths = target.ne(pad_idx).long().sum(1)

    return target, prev_output_tokens, tgt_lengths

def collate_dyn_src_tokens(src_tokens,
    pad_idx,):

    src_lengths = torch.LongTensor([
        s.ne(pad_idx).long().sum() for s in src_tokens
    ])

    x = data_utils.collate_tokens(
            src_tokens,
            pad_idx,  eos_idx=None, left_pad=True, move_eos_to_beginning=False,
        )


    return x, src_lengths


class CvtClozeDataset(LanguageClozeDataset):
    def __init__(
        self, src, src_sizes, src_dict,
        tgt, tgt_sizes, tgt_dict, block,
        left_pad_source=True, left_pad_target=False,
        shuffle=True,
        remove_eos_from_source=False, append_eos_to_target=False,
        append_bos=False, eos=None,
    ):
        super(CvtClozeDataset, self).__init__(src, src_sizes, src_dict,
        tgt, tgt_sizes, tgt_dict, block,
        left_pad_source=left_pad_source, left_pad_target=left_pad_target, shuffle=shuffle,
        remove_eos_from_source=remove_eos_from_source, append_eos_to_target=append_eos_to_target,
        append_bos=append_bos, eos=eos,)


    def __getitem__(self, index):
        src_item = self.src[index]
        tgt_item = src_item.clone()
        # recover the masked tokens
        masked_tokens = tgt_item.eq(self.src_dict.index('<mask>'))
        toks_in_src = torch.sum(masked_tokens).item()
        tgt_item[masked_tokens] = self.tgt[index][:toks_in_src]

        example = {
            'id': index,
            'source': src_item,
            'target': tgt_item,
            'block' : None,
        }
        return example


    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.
        """
        return collate_cloze(
            samples,
            pad_idx=self.src_dict.pad(),
            eos_idx=self.eos,
            bos_idx=self.src_dict.bos(),
            original_eos_idx = self.src_dict.eos(),
            mask_idx=self.src_dict.index('<mask>'),
            left_pad_source=True,
            left_pad_target=True,
        )
