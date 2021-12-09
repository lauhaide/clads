# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import numpy as np
import torch

from fairseq.data import data_utils, FairseqDataset


logger = logging.getLogger(__name__)


def collate(
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

    block_pos_left, block_pos_right, block_masks, maskToken_pos, maskToken_masks = data_utils.collate_mask_tokens(
            [s['block'] for s in samples], src_tokens, pad_idx, mask_idx,
            original_eos_idx, bos_idx )

    #AA sorting here now after use
    src_tokens = src_tokens.index_select(0, sort_order)

    # need to re-process this as some originally masked targets have been eliminated from the
    # input by source-truncation.
    # -1 to eliminate last </s> in targets by my_preprocess.py, need to fix there
    target = (target[:,:-1] * (~maskToken_masks).long())
    target = target.masked_fill(target == 0, pad_idx)

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
            'block_pos_left': block_pos_left,
            'block_pos_right': block_pos_right,
            'block_mask': block_masks,
            'maskToken_pos' : maskToken_pos,
            'maskToken_masks': maskToken_masks,
        },
        'target': target,
    }

    return batch


class LanguageClozeDataset(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets.
    """

    def __init__(
        self, src, src_sizes, src_dict,
        tgt, tgt_sizes, tgt_dict,
        block,
        left_pad_source=True, left_pad_target=False,
        shuffle=True,
        remove_eos_from_source=False, append_eos_to_target=False,
        append_bos=False, eos=None,
    ):
        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
        if tgt is not None:
            assert len(src) == len(tgt), "Source and target must contain the same number of examples {} == {}"\
                .format(len(src), len(tgt))
        self.src = src
        self.tgt = tgt
        self.block = block
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.shuffle = shuffle
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target
        self.append_bos = append_bos
        self.eos = (eos if eos is not None else src_dict.eos())
        self.buckets = None

    def get_batch_shapes(self):
        return self.buckets

    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item = self.src[index]
        block_item = self.block[index]
        #block_mask = self.block_mask[index]
        ## Append EOS to end of tgt sentence if it does not have an EOS and remove
        ## EOS from end of src sentence if it exists. This is useful when we use
        ## use existing datasets for opposite directions i.e., when we want to
        ## use tgt_dataset as src_dataset and vice versa
        #if self.append_eos_to_target:
        #    eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
        #    if self.tgt and self.tgt[index][-1] != eos:
        #        tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])

        if self.append_bos:
            #bos = self.tgt_dict.bos() if self.tgt_dict else self.src_dict.bos()
            #if self.tgt and self.tgt[index][0] != bos:
            #    tgt_item = torch.cat([torch.LongTensor([bos]), self.tgt[index]])

            bos = self.src_dict.bos()
            if self.src[index][-1] != bos:
                src_item = torch.cat([torch.LongTensor([bos]), self.src[index]])

        if self.remove_eos_from_source:
            eos = self.src_dict.eos()
            if self.src[index][-1] == eos:
                src_item = self.src[index][:-1]

        example = {
            'id': index,
            'source': src_item,
            'target': tgt_item,
            'block' : block_item,
        }  #            'block_mask' : block_mask,
        return example

    def __len__(self):
        return len(self.src)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.
        """
        return collate(
            samples,
            pad_idx=self.src_dict.pad(),
            eos_idx=self.eos,
            bos_idx=self.src_dict.bos(),
            original_eos_idx = self.src_dict.eos(),
            mask_idx=self.src_dict.index('<mask>'),
            left_pad_source=self.left_pad_source,
            left_pad_target=self.left_pad_target,
        )

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        if self.buckets is None:
            # sort by target length, then source length
            if self.tgt_sizes is not None:
                indices = indices[
                    np.argsort(self.tgt_sizes[indices], kind='mergesort')
                ]
            return indices[np.argsort(self.src_sizes[indices], kind='mergesort')]
        else:
            # sort by bucketed_num_tokens, which is:
            #   max(padded_src_len, padded_tgt_len)
            return indices[
                np.argsort(self.bucketed_num_tokens[indices], kind='mergesort')
            ]

    @property
    def supports_prefetch(self):
        return (
            getattr(self.src, 'supports_prefetch', False)
            and (getattr(self.tgt, 'supports_prefetch', False) or self.tgt is None)
            and (getattr(self.block, 'supports_prefetch', False) or self.block is None)
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        self.tgt.prefetch(indices)
        self.block.prefetch(indices)
