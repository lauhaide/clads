# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import itertools
import os
import logging



logger = logging.getLogger(__name__)

from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    data_utils,
    encoders,
    indexed_dataset,
    CatLanguagePairDataset,
    LanguagePairDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
    TokenBlockDataset,
)


from .translation import TranslationTask
from . import register_task

from .translation_from_pretrained_bart import TranslationFromPretrainedBARTTask

def load_langpair_dataset(
    data_path, split,
    src, src_dict,
    tgt, tgt_dict,
    combine, dataset_impl, upsample_primary,
    left_pad_source, left_pad_target, max_source_positions,
    max_target_positions, prepend_bos=False, load_alignments=False,
    truncate_source=False, append_source_id=False,
    num_buckets=0, mono=None,
    blocks=False,
    categories=False,
    cat_dict=None,
):

    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, '{}.{}-{}.{}'.format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else '')

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))

        end_token_src = None
        end_token_tgt = None
        if append_source_id:
            if mono is not None:
                end_token_tgt = tgt_dict.index('[{}]'.format(mono))
                end_token_src = src_dict.index('[{}]'.format(mono))
            else:
                end_token_tgt = tgt_dict.index("[{}]".format(tgt))
                end_token_src = src_dict.index('[{}]'.format(src))

        src_dataset = data_utils.load_indexed_dataset(prefix + src, src_dict, dataset_impl)

        if blocks:
            #print('****HERE SRC', len(src_dataset))
            # create continuous blocks of tokens
            src_dataset = TokenBlockDataset(
                src_dataset,
                src_dataset.sizes,
                50000,  # one less for <s>
                pad=src_dict.pad(),
                eos=end_token_src,
                break_mode="complete_doc", document_sep_len=1,
            )

        reserved_on_truncation = 1
        reserved_on_truncation += 1 if prepend_bos else 0
        reserved_on_truncation += 1 if append_source_id else 0
        if truncate_source:
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    max_source_positions - reserved_on_truncation,
                ),
                src_dict.eos(),
            )
        src_datasets.append(src_dataset)

        tgt_dataset = data_utils.load_indexed_dataset(prefix + tgt, tgt_dict, dataset_impl)

        if blocks:
            #print('****HERE TGT', len(tgt_dataset))
            tgt_dataset = TokenBlockDataset(
                tgt_dataset,
                tgt_dataset.sizes,
                10000,  # one less for <s>
                pad=tgt_dict.pad(),
                eos=end_token_tgt,
                break_mode="complete_doc", document_sep_len=1,
            )
            print(tgt_dataset)

        print('blocks', blocks, len(src_datasets) ,len(tgt_datasets))

        if tgt_dataset is not None:
            tgt_datasets.append(tgt_dataset)

        logger.info('{} {} {}-{} {} examples'.format(
            data_path, split_k, src, tgt, len(src_datasets[-1])
        ))

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
    else:
        # havent checked for this (I wont use it), so flag if comming here
        raise NotImplementedError

        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        if len(tgt_datasets) > 0:
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
        else:
            tgt_dataset = None

    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())

    eos = None
    if append_source_id:
        if mono is not None:
            src_dataset = AppendTokenDataset(src_dataset, src_dict.index('[{}]'.format(mono)))
            if tgt_dataset is not None:
                tgt_dataset = AppendTokenDataset(tgt_dataset, tgt_dict.index('[{}]'.format(mono)))
            eos = tgt_dict.index('[{}]'.format(mono))
        else:
            src_dataset = AppendTokenDataset(src_dataset, src_dict.index('[{}]'.format(src)))
            if tgt_dataset is not None:
                tgt_dataset = AppendTokenDataset(tgt_dataset, tgt_dict.index('[{}]'.format(tgt)))
            eos = tgt_dict.index('[{}]'.format(tgt))

    align_dataset = None
    if load_alignments:
        align_path = os.path.join(data_path, '{}.align.{}-{}'.format(split, src, tgt))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(align_path, None, dataset_impl)

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None

    return LanguagePairDataset(
        src_dataset, src_dataset.sizes, src_dict,
        tgt_dataset, tgt_dataset_sizes, tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        align_dataset=align_dataset, eos=eos,
        num_buckets=num_buckets,
    )




@register_task('sum_from_pretrained_bart')
class SumFromPretrainedBARTTask(TranslationFromPretrainedBARTTask):
    """
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        TranslationFromPretrainedBARTTask.add_args(parser)
        parser.add_argument('--mono-lang', metavar='MONO',
                            help='monolingual data language')

        parser.add_argument('--blocks', action='store_true',
                            help='break input in sentence blocks')
        parser.add_argument("--tokens-per-sample", default=512, type=int,
                            help="max number of total tokens over all segments per sample for dataset")
        parser.add_argument("--sample-break-mode", default="complete_doc", type=str,
                            help="mode for breaking sentence")

        parser.add_argument('--freeze-dec', action='store_true',
                            help='do cvt with decoder frozen, except encoder-attn and output_projection(embed_tokens).')
        parser.add_argument('--freeenc-exclude-layers', default=None, type=str,
                            help='exclude encoder layers when freezing.')
        parser.add_argument('--freeze-enc', action='store_true',
                            help='do cvt with encoder frozen, except given layers.')
        # fmt: on

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)
        self.langs = args.langs.split(',')
        for d in [src_dict, tgt_dict]:
            self.specialise_dictionary(d, args.langs.split(','))


    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = self.args.data.split(':')
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang

        self.datasets[split] = load_langpair_dataset(
            data_path, split, src, self.src_dict, tgt, self.tgt_dict,
            combine=combine, dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=getattr(self.args, 'max_source_positions', 1024),
            max_target_positions=getattr(self.args, 'max_target_positions', 1024),
            load_alignments=self.args.load_alignments,
            prepend_bos=getattr(self.args, 'prepend_bos', False),
            append_source_id=True,
            truncate_source=getattr(self.args, 'truncate_source', False),
            mono=getattr(self.args, 'mono_lang', None),
            blocks=getattr(self.args, 'blocks', False),
            )

    def build_generator(self, models, args):
        mono = getattr(self.args, 'mono_lang', None)
        if getattr(args, 'score_reference', False):
            from fairseq.sequence_scorer import SequenceScorer
            return SequenceScorer(
                self.target_dictionary,
                eos=self.tgt_dict.index('[{}]'.format(self.args.mono_lang if mono else self.args.target_lang))
            )
        else:
            from fairseq.sequence_generator import SequenceGenerator
            return SequenceGenerator(
                models,
                self.target_dictionary,
                beam_size=getattr(args, 'beam', 5),
                max_len_a=getattr(args, 'max_len_a', 0),
                max_len_b=getattr(args, 'max_len_b', 200),
                min_len=getattr(args, 'min_len', 1),
                normalize_scores=(not getattr(args, 'unnormalized', False)),
                len_penalty=getattr(args, 'lenpen', 1),
                unk_penalty=getattr(args, 'unkpen', 0),
                temperature=getattr(args, 'temperature', 1.),
                match_source_len=getattr(args, 'match_source_len', False),
                no_repeat_ngram_size=getattr(args, 'no_repeat_ngram_size', 0),
                eos=self.tgt_dict.index('[{}]'.format(self.args.mono_lang if mono else self.args.target_lang))
            )

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        src_lang_id = self.source_dictionary.index('[{}]'.format(self.args.mono_lang))
        source_tokens = []
        for s_t in src_tokens:
            s_t = torch.cat([s_t, s_t.new(1).fill_(src_lang_id)])
            source_tokens.append(s_t)
        dataset = LanguagePairDataset(source_tokens, src_lengths, self.source_dictionary)
        return dataset

    @classmethod
    def specialise_dictionary(cls, d, langs):
        """Add task specific symbols the dictionary from the filename """
        for l in langs:
            d.add_symbol('[{}]'.format(l))
        d.add_symbol('<mask>')
        return d
