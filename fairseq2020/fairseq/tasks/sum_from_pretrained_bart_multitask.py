# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import itertools
import os
import logging
from collections import OrderedDict
import string
import numpy as np

import contextlib
import torch

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
    RoundRobinZipDatasets,
    LanguageClozeDataset,
    CvtClozeDataset,
    FewSubsampleDataset,
)

from fairseq.data.language_cvt_dataset import collate_dyn_src_tokens, collate_dyn_targets

from fairseq.models import FairseqMultiModel

from .translation import TranslationTask
from . import FairseqTask, register_task

from .translation_from_pretrained_bart import TranslationFromPretrainedBARTTask

from fairseq import options
from fairseq import utils

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
    process_target=True,
    args=None,
    task_type=None,
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

        tgt_dataset = None
        if process_target:
            tgt_dataset = data_utils.load_indexed_dataset(prefix + tgt, tgt_dict, dataset_impl)

            if blocks:
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
            #if truncate_source:
            #    tgt_dataset = AppendTokenDataset(
            #        TruncateDataset(
            #            StripTokenDataset(tgt_dataset, tgt_dict.eos()),
            #            max_target_positions - reserved_on_truncation,
            #        ),
            #        tgt_dict.eos(),
            #    )
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
        elif 'cvt' in task_type :
            src_dataset = AppendTokenDataset(src_dataset, src_dict.index('[{}]'.format(src)))
            #eos = src_dict.index('[{}]'.format(src))
            if tgt_dataset is not None:
                tgt_dataset = AppendTokenDataset(tgt_dataset, tgt_dict.index('[{}]'.format(tgt)))
            eos = tgt_dict.index('[{}]'.format(tgt))
        else:
            src_dataset = AppendTokenDataset(src_dataset, src_dict.index('[{}]'.format(src)))
            if tgt_dataset is not None:
                tgt_dataset = AppendTokenDataset(tgt_dataset, tgt_dict.index('[{}]'.format(tgt)))
            eos = tgt_dict.index('[{}]'.format(tgt))
    print('*\t define eos: ', eos)

    align_dataset = None
    if load_alignments:
        align_path = os.path.join(data_path, '{}.align.{}-{}'.format(split, src, tgt))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(align_path, None, dataset_impl)


    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None

    ### do subsample and prepare few instances for few shot fine-tuning
    if args.cvt_few > 0 :
        assert args.cvt_few > 0, "For cvt few ratio should be given"
        assert args.cvt_few_ratio <= 1
        ratio = args.cvt_few_ratio if task_type else args.cvt_mono_ratio
        few = (np.ceil(args.cvt_few / 3).astype(int) if split=='valid' \
                        else (np.ceil(args.cvt_few * (2/3))).astype(int)) \
                                if task_type else None
        # assume that src/tgt must have same nb!:
        if task_type=='cvt':
            if split=='valid': #if valid total size and few inst should be same
                    actual_size = np.ceil(args.cvt_few / 3).astype(int)
            elif args.only_few:
                actual_size = np.ceil(args.cvt_few * (2/3)).astype(int)
            else:
                actual_size = np.ceil(len(tgt_dataset) * ratio).astype(int)
        else:
            # it will come here when loading the mono-lingual data part of the mono/cross lingual multitasks
            if split == 'valid': #and args.cvt_few>0: #TODO: check this condition in the general, now I use it in specific case, ????
                actual_size = np.ceil((args.cvt_few / 3)/2).astype(int)
            else:
                actual_size = np.ceil(len(tgt_dataset) * ratio).astype(int)

        print("*\t*Synthetic Few set (+ unlabelled for CVT) size (task:{}, split:{}) " \
              "==> {}".format(task_type, split, actual_size))
        

        # filter unwanted target length here.... :( roundrobinzipdataset has a bug otherwise
        idxs = np.arange(len(tgt_dataset))
        szs = np.asarray(tgt_dataset_sizes)
        valid_indices = list(idxs[np.where(szs <= (max_target_positions-reserved_on_truncation))])
        target_idxs = valid_indices
        print(
            "*    Indices excluded when subsampling dataset on target: {}, remain:{}".format(len(tgt_dataset) - len(target_idxs), len(target_idxs)))
        np.random.seed(args.seed)
        indices = np.random.choice(target_idxs, actual_size, replace=False)

        #if  task_type: #any cvt type
        src_dataset = FewSubsampleDataset(src_dataset, indices, ratio, few)
        tgt_dataset = FewSubsampleDataset(tgt_dataset, indices, ratio, few)

    if categories:
        print('*\t Loading categories...')
        cat_path = os.path.join(data_path, '{}.{}-{}.categories'.format(split, src, tgt))
        if indexed_dataset.dataset_exists(cat_path, impl=dataset_impl):
            categories_dataset = data_utils.load_indexed_dataset(cat_path, None, dataset_impl)
        return CatLanguagePairDataset(
            src_dataset, src_dataset.sizes, src_dict,
            tgt_dataset, tgt_dataset_sizes, tgt_dict,
            categories_dataset, cat_dict,
            left_pad_source=left_pad_source,
            left_pad_target=left_pad_target,
            align_dataset=align_dataset, eos=eos,
            num_buckets=num_buckets,
        )
    elif task_type=='cvt':
        return LanguagePairDataset(
            src_dataset, src_dataset.sizes, src_dict,
            tgt_dataset, tgt_dataset_sizes, tgt_dict,
            left_pad_source=left_pad_source,
            left_pad_target=left_pad_target,
            align_dataset=align_dataset, eos=eos,
            num_buckets=num_buckets,
        )
    else:
        return LanguagePairDataset(
            src_dataset, src_dataset.sizes, src_dict,
            tgt_dataset, tgt_dataset_sizes, tgt_dict,
            left_pad_source=left_pad_source,
            left_pad_target=left_pad_target,
            align_dataset=align_dataset, eos=eos,
            num_buckets=num_buckets,
        )


def load_langcloze_dataset(
    data_path, split,
    src, src_dict,
    tgt, tgt_dict,
    combine, dataset_impl, upsample_primary,
    left_pad_source, left_pad_target, max_source_positions,
    max_target_positions, prepend_bos=False,
    truncate_source=False, append_source_id=False,
    blocks=False, task_type=None,
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
            end_token_src = src_dict.index('[{}]'.format(src))
            #end_token_tgt = tgt_dict.index("[{}]".format(tgt)) ## NEED this?

        src_dataset = data_utils.load_indexed_dataset(prefix + src, src_dict, dataset_impl)

        if blocks:
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

        if task_type == 'cvt':
            newprex = (prefix + tgt).replace(tgt, 'disc') # in cloze the target files contain disc --> TODO better
            tgt_dataset = data_utils.load_indexed_dataset(newprex, tgt_dict, dataset_impl)
        else:
            tgt_dataset = data_utils.load_indexed_dataset(prefix + tgt, tgt_dict, dataset_impl)

        masks_dataset = data_utils.load_indexed_dataset(prefix + 'mask', None, dataset_impl)

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
        # havent checked for this (I wont use it), so flag if coming here
        raise NotImplementedError

    if prepend_bos:
        assert hasattr(src_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        #if tgt_dataset is not None:
        #    tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos()) ## NEED this?


    if append_source_id:
        src_dataset = AppendTokenDataset(src_dataset, src_dict.index('[{}]'.format(src)))
        ## NEED this?
        #if tgt_dataset is not None:
        #    tgt_dataset = AppendTokenDataset(tgt_dataset, tgt_dict.index('[{}]'.format(tgt)))
        #eos = tgt_dict.index('[{}]'.format(tgt))

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None

    if task_type == 'cvt':
        return CvtClozeDataset(
            src_dataset, src_dataset.sizes, src_dict,
            tgt_dataset, tgt_dataset_sizes, tgt_dict,
            masks_dataset,
            left_pad_source=left_pad_source,
            left_pad_target=left_pad_target,
            eos=end_token_src,  # this is used for target sequence prediction
        )
    else:
        return LanguageClozeDataset(
            src_dataset, src_dataset.sizes, src_dict,
            tgt_dataset, tgt_dataset_sizes, tgt_dict,
            masks_dataset,
            left_pad_source=left_pad_source,
            left_pad_target=left_pad_target,
            eos=None, # this is used for target sequence prediction
        )

@register_task('sum_from_pretrained_bart_mtl')
class SumFromPretrainedBARTMultiTask(FairseqTask):
    """
    Mono-lingual and cross-lingual summarisation with a model initialized with a multilingual pretrain, mBART.
    Multi-task with additional document understanding (Cloze Discourse, Document classification).
    Ideas from :
    https://github.com/pytorch/fairseq/blob/master/fairseq/tasks/translation_multi_simple_epoch.py
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: on

        TranslationFromPretrainedBARTTask.add_args(parser)

        parser.add_argument('--mono-lang', metavar='MONO',
                            help='monolingual data language')

        parser.add_argument('--lang-task-pairs', default=None, metavar='PAIRS',
                            help='comma-separated list of language pairs and their tasks (in training order): '
                            'en_XX-mono (summarisation monolingual supervised), '
                            'fr_XX-cvt-en_XX (only input documents in source lang (Fr), use CVT training to generate in target lang (En) ), '     
                            'fr_XX-cross-en_XX (summarisation cross-lingual supervised),'
                            'fr_XX-disc (discourse cloze task on fr_XX input documents),'
                            'fr_XX-cat (document classification on fr_XX input documents)'
                            'Note: cat amd cross have not been fully implemented for the moment.'
                            )

        # data loading
        parser.add_argument('--blocks', action='store_true',
                            help='break input in sentence blocks')
        parser.add_argument("--tokens-per-sample", default=512, type=int,
                            help="max number of total tokens over all segments per sample for dataset")
        parser.add_argument("--sample-break-mode", default="complete_doc", type=str,
                            help="mode for breaking sentence")

        ## add inference configs for cvt training
        parser.add_argument('--min-len-cvt', default=1, type=float, metavar='N',
                           help=('minimum generation length'))
        parser.add_argument('--max-len-b-cvt', default=200, type=int, metavar='N',
                           help=('generate sequences of maximum length ax + b, '
                                 'where x is the source length'))
        parser.add_argument('--beam-cvt', default=5, type=int, metavar='N',
                           help='beam size')
        parser.add_argument('--no-repeat-ngram-size-cvt', default=0, type=int, metavar='N',
                           help='ngram blocking such that this size ngram cannot be repeated in the generation')
        parser.add_argument('--lenpen-cvt', default=1, type=float,
                           help='length penalty: <1.0 favors shorter, >1.0 favors longer sentences')

        #cvt working
        parser.add_argument('--cvt-freeze-dec', action='store_true',
                            help='do cvt with decoder frozen, except encoder-attn and output_projection(embed_tokens).')
        parser.add_argument('--cvt-freeze-enc', action='store_true',
                            help='do cvt with encoder frozen except token- and position-embeddings.')
        parser.add_argument('--freeenc-exclude-layers', default=None, type=str,
                            help='exclude encoder layers when freezing.')
        parser.add_argument('--cvt-target-wei', action='store_true',
                            help='weight cvt loss according to 1/tok_ppl of the generated target')
        parser.add_argument('--cvt-proportion', default=0.00, type=float, help='')
        parser.add_argument('--mono-proportion', default=0.00, type=float, help='')
        parser.add_argument('--few-proportion', default=0.00, type=float, help='')
        parser.add_argument('--cvt-p', default=0.00, type=float,
            help='drop elements from encoder output as view generation',)
        parser.add_argument('--cvt-layers', default='', type=float,
            help='comma separated list of layer numbers to be used as views (all should be < than the nb of layers in the model)',)
        parser.add_argument('--cvt-few', default=0, type=int,
                            help='number of few instances combined with cvt',)
        parser.add_argument('--cvt-few-ratio', default=0.0, type=float,
                            help='sample dataset ratio for cvt',)
        parser.add_argument('--cvt-mono-ratio', default=0.0, type=float,
                            help='sample dataset ratio on mono for cvt',)
        parser.add_argument('--only-few', action='store_true',
                    help='use only the few instances, i.e. no cvt happening just few labelled instances.')
        parser.add_argument('--cvt-future-pred', action='store_true',
                    help='predict future token when doing cvt. NOT UNSED')

        # optimisation
        parser.add_argument('--updates-per-task', action='store_true',
                            help='group updates per task')

        # fmt: off

    @classmethod
    def setup_task(cls, args, **kwargs):
        dicts, training, aux_src_clasif_tasks = cls.prepare(args, **kwargs)
        return cls(args, dicts, training, aux_src_clasif_tasks)

    @classmethod
    def prepare(cls, args, **kargs):
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)

        if args.lang_task_pairs is None and (args.source_lang is None or args.target_lang is None):
            raise ValueError('--lang-task-pairs is required. List all the language/task pairs in the training objective.')
        if isinstance(args.lang_task_pairs, str):
            args.lang_task_pairs = args.lang_task_pairs.split(',')

        if args.source_lang is not None or args.target_lang is not None:
            training = False
        else:
            training = True

        data_paths = args.data.split(':')
        if training: #check data paths & tasks
            assert len(data_paths) == len(args.lang_task_pairs), \
                "Nb of tasks in --lang_task_pairs and data paths in [DATA] should coincide" # each task might have a different dir
                #well not necessarily different but by easy desing we assume a data definition for each task

        # load dictionaries
        aux_src_tasks = {}
        dicts = OrderedDict()
        if training:
          for task_lang, dp in zip(args.lang_task_pairs, data_paths):
            paths = utils.split_paths(dp)
            assert len(paths) > 0
            #infer language/task dictionaries
            if 'mono' in task_lang:
                langs = ['src', 'tgt']
            elif 'cvt' in task_lang:
                langs = task_lang.split('-cvt-')
            else:
                langs = task_lang.split('-cross-')
            assert len(langs) > 0
            for lang in langs:
                print("*\t Loading dictionary for lang/task: " + lang)
                dicts[lang] = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(lang)))
                if not lang in ['cat']: # target dictionaries for classif are not lang vocabularies
                    dicts[lang] = cls.specialise_dictionary(dicts[lang], args.langs.split(',')) # adds langs and <mask> tokens
                if len(dicts) > 0:
                    # compare all against the first dict, all should have same symbols
                    assert dicts[lang].pad() == next(iter(dicts.items()))[1].pad()
                    assert dicts[lang].eos() == next(iter(dicts.items()))[1].eos()
                    assert dicts[lang].unk() == next(iter(dicts.items()))[1].unk()

                logger.info('[{}] dictionary: {} types'.format(lang, len(dicts[lang])))
                if lang in ['disc', 'cat']:
                    aux_src_tasks[lang]=len(dicts[lang])
        else:
            paths = utils.split_paths(data_paths[0])
            for lang in [args.source_lang, args.target_lang]:
                dicts[lang] = \
                    cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(lang)))
                dicts[lang] = cls.specialise_dictionary(dicts[lang], args.langs.split(','))  # adds langs and <mask> tokens
        return dicts, training, aux_src_tasks

    @classmethod
    def specialise_dictionary(cls, d, langs):
        """Add task specific symbols the dictionary from the filename """
        for l in langs:
            d.add_symbol('[{}]'.format(l))
        d.add_symbol('<mask>')

        return d

    def __init__(self, args, dicts, training, aux_src_classif_tasks):
        super().__init__(args)
        self.dicts = dicts
        self.training = training
        if training:
            self.lang_task_pairs = args.lang_task_pairs
        else:
            # Finally not used in our models adaptations.
            self.lang_task_pairs = ['{}-eval-{}'.format(args.source_lang, args.target_lang)]
        # eval_lang_pairs for multilingual translation is usually all of the
        # lang_pairs. However for other multitask settings or when we want to
        # optimize for certain languages we want to use a different subset. Thus
        # the eval_lang_pairs class variable is provided for classes that extend
        # this class.
        self.eval_lang_pairs = self.lang_task_pairs
        # model_lang_pairs will be used to build encoder-decoder model pairs in
        # models.build_model(). This allows multitask type of sub-class can
        # build models other than the input lang_pairs
        self.model_lang_pairs = self.lang_task_pairs
        #self.langs = list(dicts.keys())
        self.langs = args.langs #all language tokens IDs
        self.langs_pairs = list(dicts.keys()) # different dictionaries for different tasks

        self._aux_src_classif_tasks = aux_src_classif_tasks
        self.task_types = list(self._aux_src_classif_tasks.keys()) + ['mainsum']

        self.aux_generator = None

    def getTaskType(self, task_lang_pair):
        """map command-line specification to internal task id recognised for this multi-task, model and criterion"""
        if 'disc' in task_lang_pair: ret = 'disc'
        elif 'cat' in task_lang_pair: ret = 'cat'
        elif 'cvt' in task_lang_pair: ret = 'cvt'
        else: ret = 'mainsum'
        return ret

    def getTaskLangPair(self, task_lang_pair):
        """map command-line specification to internal task id recognised for this multi-task, model and criterion"""
        ret = None
        if 'disc' in task_lang_pair: ret = (task_lang_pair.split('-disc')[0], 'disc')
        elif 'cat' in task_lang_pair: ret = (task_lang_pair.split('-cat')[0], 'cat')
        elif 'cvt' in task_lang_pair: ret = (task_lang_pair.split('-cvt-')[0], task_lang_pair.split('-cvt-')[1])
        elif 'cross' in task_lang_pair: ret = (task_lang_pair.split('-cross-')[0], task_lang_pair.split('-cross-')[1])
        elif 'mono' in task_lang_pair: ret = (task_lang_pair.split('-mono')[0], task_lang_pair.split('-mono')[0])

        return ret

    def setEvalLangPair(self, task_lang_pair):
        self.eval_lang_pairs = self.getTaskLangPair(task_lang_pair)

    def getDict(self, lang):
        return self.dicts[lang]

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        def pair_dataset(lang_task_pair, paths):
            # paths = utils.split_paths(self.args.data)
            assert len(paths) > 0
            data_path = paths[(epoch - 1) % len(paths)]
            # infer langcode and task,

            if 'mono' in lang_task_pair :
                src, tgt = 'src', 'tgt'
                mono_lang = lang_task_pair.split('-')[0]
                return load_langpair_dataset(
                    data_path, split, src, self.dicts[src], tgt, self.dicts[tgt],
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
                    mono=mono_lang,
                    blocks=getattr(self.args, 'blocks', False),
                    args=self.args
                    )
            elif 'eval' in lang_task_pair:
                src, tgt = lang_task_pair.split('-eval-')
                mono_lang = None ##TODO: revise this for mono eval
                return load_langpair_dataset(
                    data_path, split, src, self.dicts[src], tgt, self.dicts[tgt],
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
                    mono=mono_lang,
                    blocks=getattr(self.args, 'blocks', False),
                    args=self.args
                    )
            elif 'disc' in lang_task_pair:
                src, tgt = lang_task_pair.split('-')
                return load_langcloze_dataset(
                    data_path, split, src, self.dicts[src], tgt, self.dicts[tgt],
                    combine=combine, dataset_impl=self.args.dataset_impl,
                    upsample_primary=self.args.upsample_primary,
                    left_pad_source=self.args.left_pad_source,
                    left_pad_target=self.args.left_pad_target,
                    max_source_positions=getattr(self.args, 'max_source_positions', 1024),
                    max_target_positions=getattr(self.args, 'max_target_positions', 1024),
                    prepend_bos=getattr(self.args, 'prepend_bos', False),
                    append_source_id=True,
                    truncate_source=getattr(self.args, 'truncate_source', False),
                    blocks=getattr(self.args, 'blocks', False),
                    )
            elif 'cvt' in lang_task_pair:
                src, tgt = lang_task_pair.split('-cvt-')[0], lang_task_pair.split('-cvt-')[1]
                return load_langpair_dataset(
                    data_path, split, src, self.dicts[src], tgt, self.dicts[tgt],
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
                    blocks=getattr(self.args, 'blocks', False),
                    process_target=True,
                    args=self.args,
                    task_type='cvt',
                    )
            elif 'cross' in lang_task_pair or 'cat' in lang_task_pair:
                raise NotImplementedError

        self.datasets[split] = RoundRobinZipDatasets(
            OrderedDict([
                (lang_task_pair, pair_dataset(lang_task_pair, [paths]))
                for lang_task_pair, paths in zip(self.lang_task_pairs, self.args.data.split(':'))
            ]),
            eval_key=None if self.training else "%s-%s" % (self.args.source_lang, self.args.target_lang),
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        src_lang_id = self.source_dictionary.index('[{}]'.format(self.args.mono_lang))
        source_tokens = []
        for s_t in src_tokens:
            s_t = torch.cat([s_t, s_t.new(1).fill_(src_lang_id)])
            source_tokens.append(s_t)
        dataset = LanguagePairDataset(source_tokens, src_lengths, self.source_dictionary)
        return dataset

    def build_model(self, args):
        return super().build_model(args)

    def build_generator(self, models, args):
        src, tgt = self.eval_lang_pairs
        if getattr(args, 'score_reference', False):
            from fairseq.sequence_scorer import SequenceScorer
            return SequenceScorer(
                self.target_dictionary,
                eos=self.dicts[tgt].index('[{}]'.format(tgt)),
            )
        else:
            from fairseq.sequence_generator import SequenceGenerator
            return SequenceGenerator(
                models,
                self.dicts[tgt],
                beam_size=getattr(args, 'beam', getattr(args, 'beam_cvt', 5)),
                max_len_a=getattr(args, 'max_len_a', 0),
                max_len_b=getattr(args, 'max_len_b', getattr(args, 'max_len_b_cvt', 150)),
                min_len=getattr(args, 'min_len', getattr(args, 'min_len_cvt', 50)),
                normalize_scores=(not getattr(args, 'unnormalized', False)),
                len_penalty=getattr(args, 'lenpen', getattr(args, 'lenpen_cvt', 2)),
                unk_penalty=getattr(args, 'unkpen', 0),
                temperature=getattr(args, 'temperature', 1.),
                match_source_len=getattr(args, 'match_source_len', False),
                no_repeat_ngram_size=getattr(args, 'no_repeat_ngram_size', getattr(args, 'no_repeat_ngram_size_cvt', 0)),
                eos=self.dicts[tgt].index('[{}]'.format(tgt)),
            )

    def train_step(self, sample, model, criterion, optimizer, update_num, ignore_grad=False):
        model.train()
        model.set_num_updates(update_num)
        from collections import defaultdict
        agg_loss, agg_sample_size, agg_logging_output = 0., 0., defaultdict(float)
        curr_lang_task_pairs = [
            lang_task_pair
            for lang_task_pair in self.lang_task_pairs
            if lang_task_pair in sample.keys() and \
                    sample[lang_task_pair] is not None and len(sample[lang_task_pair]) != 0
        ]

        for idx, lang_task_pair in enumerate(curr_lang_task_pairs):

            def maybe_no_sync():
                if (
                    self.args.distributed_world_size > 1
                    and hasattr(model, 'no_sync')
                    and idx < len(curr_lang_task_pairs) - 1
                ):
                    return model.no_sync()
                else:
                    return contextlib.ExitStack()  # dummy contextmanager

            with maybe_no_sync():
                loss, sample_size, logging_output = criterion(model, (lang_task_pair, sample[lang_task_pair]))
                if ignore_grad:
                    loss *= 0
                optimizer.backward(loss)
                #print(lang_task_pair, loss)

            agg_loss += loss.detach().item()
            # TODO make summing of the sample sizes configurable
            agg_sample_size += sample_size
            for k in logging_output:
                agg_logging_output[k] += logging_output[k]
                agg_logging_output[f"{lang_task_pair}:{k}"] += logging_output[k]

        return agg_loss, agg_sample_size, agg_logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            from collections import defaultdict
            agg_loss, agg_sample_size, agg_logging_output = 0., 0., defaultdict(float)
            for lang_task_pair in self.lang_task_pairs:
                if lang_task_pair not in sample \
                        or sample[lang_task_pair] is None \
                        or len(sample[lang_task_pair]) == 0 :
                    continue
                loss, sample_size, logging_output = criterion(model, (lang_task_pair, sample[lang_task_pair]))
                agg_loss += loss.data.item()
                # TODO make summing of the sample sizes configurable
                agg_sample_size += sample_size
                for k in logging_output:
                    agg_logging_output[k] += logging_output[k]
                    agg_logging_output[f"{lang_task_pair}:{k}"] += logging_output[k]
        return agg_loss, agg_sample_size, agg_logging_output

    def target_step(self, train_task, samples, model):
        """This step is when doing CVT, it generate candidates with current model parametrization.
        And  sets these candiates as targets, these training pairs will be used in conjunction with
        the views automatically created out of the input document.

        Note: to do inference here we will create artificial batches to go faster. The overall adaptation
        has smaller (in fact size=1) batches, however, in this inference step we can regroup in batches,
        we then re-split back to the real batches of the task."""

        def keep(t):
            tstr = tgt_dict.symbols[t]
            return t!=tgt_dict.eos() and t!=tgt_dict.bos() \
                    and not any(x.isdigit() for x in tstr) \
                    and not any(x.isupper() for x in tstr) \
                    and not any(x in string.punctuation for x in tstr)
        model.eval()
        with torch.no_grad():

            if train_task == 'combined':
                if not samples[0]:
                    return samples # there are no samples to process ! TODO: debug
                for k in samples[0]: #we assume always a sample at least
                    if 'cvt' in k:
                        src, tgt = self.getTaskLangPair(k)
                        self.setEvalLangPair(k)
                        break
            else:
                src, tgt = self.getTaskLangPair(train_task)
                self.setEvalLangPair(train_task)

            # initialise generator if not created
            if not self.aux_generator:
                self.aux_generator = self.build_generator([model], self.args)
            self.aux_generator.update_models([model])

            src_tokens = []
            ori_batch = None
            if self.args.cvt_p > 0 or len(self.args.cvt_layers.split()) > 0: #view is generated dynamically in forward
                for s in samples:
                    if not s:
                        continue  # by some reason crashed s=None at end of epoch, cannot reproduce
                    for k in s:
                      if 'cvt' in k:
                        ori_batch = s[k]['net_input']['src_tokens'].size(0)
                        for n in torch.arange(0, s[k]['net_input']['src_tokens'].size(0)):
                            src_tokens.append(s[k]['net_input']['src_tokens'][n])
            else:
                # as is a cloze dataset in the target comes the original sequence
                for s in samples:
                    if not s:
                        continue  # by some reason crashed s=None at end of epoch, cannot reproduce
                    for k in s:
                      if 'cvt' in k:
                        ori_batch = s[k]['target'].size(0)
                        for n in torch.arange(0, s[k]['target'].size(0)):
                            src_tokens.append(s[k]['target'][n])


            tgt_dict = self.getDict(tgt) #tgt/src are same
            src_tokens, src_lengths = collate_dyn_src_tokens(src_tokens, tgt_dict.pad())

            if len(src_tokens) % 5 == 0:
                # try to group to go faster on predicting... I know my batches size can exploit this...
                part_src_tokens = torch.split(src_tokens, 5)
                part_src_lengths = torch.split(src_lengths, 5)
                ONRANGE = len(part_src_tokens)
            else:
                part_src_tokens = torch.split(src_tokens, 1)
                part_src_lengths = torch.split(src_lengths, 1)
                ONRANGE = src_tokens.size(0)

            all_pos_scores = []
            all_hypos = []
            for n in range(ONRANGE):
                if n >= len(part_src_tokens): # if a sample comes with less....
                    break
                fake_batch = {
                            'nsentences': part_src_tokens[n].size(0),
                            'ntokens': part_src_lengths[n].sum().item(),
                            'net_input': {
                                'src_tokens': part_src_tokens[n].cuda(),
                                'src_lengths': part_src_lengths[n].cuda(), #to(device) ????
                            }}

                hypos = self.inference_step(self.aux_generator, model, fake_batch)

                if self.args.cvt_target_wei:
                    bhypos = [hyp[0]['tokens'].tolist() for hyp in hypos]
                    bsrc = part_src_tokens[n].tolist()

                    #copied = [max(0, len(list(set(lst1) & set(lst2))) - 5 ) / len(lst1)  \
                    #            for lst1, lst2 in zip(bhypos, bsrc)]
                    #avg = [2 if x> 0 else 1 for x in copied] this is not correct! lower score here are good and the the isntance should receive more atte
                    #tokppl = [1 / (torch.exp(-1 * (sum(hyp[0]['positional_scores']) / len(hyp[0]['positional_scores'])))) \
                    #          for hyp in hypos]
                    #mix two criteria,
                    #hypo_wei = [(c+t)/g for c,t,g in zip(copied, tokppl, avg)]

                    overlaps = [list(set(lst1) & set(lst2)) for lst1, lst2 in zip(bhypos, bsrc)]
                    overlaps = list([tgt_dict.symbols[t] \
                                              for t in s if keep(t)] for s in overlaps)

                    if self.args.cvt_target_wei:
                        copied = [(max(0, len(list(o)) - 5) / len(lst1)) + 1 \
                                  for lst1, o in zip(bhypos, overlaps)]

                    hypo_wei = copied # will increase the loss according to nb of copied tokens from the input
                    all_pos_scores.extend(hypo_wei)

                all_hypos.extend([hyp[0]['tokens'] for hyp in hypos])

            target, prev_output_tokens, tgt_lengths = collate_dyn_targets(all_hypos, tgt_dict.pad(), tgt_dict.index('[{}]'.format(tgt)))

            #print(list(" ".join([tgt_dict.symbols[t] for t in s]) for s in prev_output_tokens.tolist()[:2]))
            #print(list(" ".join([tgt_dict.symbols[t] for t in s]) for s in src_tokens.tolist()[:2]))

            target = torch.split(target, ori_batch)
            prev_output_tokens = torch.split(prev_output_tokens, ori_batch)
            tgt_lengths = torch.split(tgt_lengths, ori_batch)
            assert ori_batch == 1
            i = 0
            for s in samples:
                if not s:
                    continue #by some reason crashed s=None at end of epoch, cannot reproduce
                for k in s:
                  if 'cvt' in k:
                    if 'few' in s[k].keys() and s[k]['few'][0]:
                        #print("\nORI", list(" ".join([tgt_dict.symbols[t] for t in s]) for s in s[k]['target'].tolist()))
                        #print(list(" ".join([tgt_dict.symbols[t] for t in s]) for s in s[k]['net_input']['prev_output_tokens'].tolist()))
                        #print("Do not rewrite target!")
                        i += 1
                        continue
                    s[k]['nsentences'] = target[i].size(0) # as used/set in collaters
                    s[k]['ntokens'] = tgt_lengths[i].sum().item() # as used/set in collaters
                    s[k]['net_input']['prev_output_tokens'] = prev_output_tokens[i]
                    s[k]['target'] = target[i]
                    if self.args.cvt_target_wei:
                        s[k]['target_score'] = all_pos_scores[i]

                    i += 1

        model.train()
        return samples


    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        if 'src' in self.dicts.keys():
            return self.dicts['src']
        else:
            return next(iter(self.dicts.items()))[1]
    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        if 'tgt' in self.dicts.keys():
            return self.dicts['tgt']
        else:
            return next(iter(self.dicts.items()))[1]

    @property
    def aux_src_classif_tasks(self):
        return self._aux_src_classif_tasks



from fairseq.file_io import PathManager
class CatDictionary(object):
    """A mapping from symbols to consecutive integers"""

    def __init__(
        self
    ):
        self.symbols = []
        self.count = []
        self.indices = {}

    def __eq__(self, other):
        return self.indices == other.indices

    def __getitem__(self, idx):
        if idx < len(self.symbols):
            return self.symbols[idx]
        return self.unk_word

    def __len__(self):
        """Returns the number of symbols in the dictionary"""
        return len(self.symbols)

    def __contains__(self, sym):
        return sym in self.indices

    def index(self, sym):
        """Returns the index of the specified symbol"""
        assert isinstance(sym, str)
        if sym in self.indices:
            return self.indices[sym]
        return self.unk_index

    def string(
        self,
        tensor,
    ):
        """Helper for converting a tensor of token indices to a string.
        """
        def token_string(i):
            if i == self.unk():
                if unk_string is not None:
                    return unk_string
                else:
                    return self.unk_string(escape_unk)
            else:
                return self[i]

        sent = " ".join(
            token_string(i)
            for i in tensor
            if utils.item(i) not in extra_symbols_to_ignore
        )
        return sent


    def add_symbol(self, word, n=1, overwrite=False):
        """Adds a word to the dictionary"""
        if word in self.indices and not overwrite:
            idx = self.indices[word]
            self.count[idx] = self.count[idx] + n
            return idx
        else:
            idx = len(self.symbols)
            self.indices[word] = idx
            self.symbols.append(word)
            self.count.append(n)
            return idx


    def update(self, new_dict):
        """Updates counts from new dictionary."""
        for word in new_dict.symbols:
            idx2 = new_dict.indices[word]
            if word in self.indices:
                idx = self.indices[word]
                self.count[idx] = self.count[idx] + new_dict.count[idx2]
            else:
                idx = len(self.symbols)
                self.indices[word] = idx
                self.symbols.append(word)
                self.count.append(new_dict.count[idx2])


    def finalize(self, threshold=-1, nwords=-1, padding_factor=8):
        """Sort symbols by frequency in descending order, ignoring special ones.

        Args:
            - threshold defines the minimum word count
            - nwords defines the total number of words in the final dictionary,
                including special symbols
            - padding_factor can be used to pad the dictionary size to be a
                multiple of 8, which is important on some hardware (e.g., Nvidia
                Tensor Cores).
        """
        if nwords <= 0:
            nwords = len(self)

        new_indices = dict(zip(self.symbols[: self.nspecial], range(self.nspecial)))
        new_symbols = self.symbols[: self.nspecial]
        new_count = self.count[: self.nspecial]

        c = Counter(
            dict(
                sorted(zip(self.symbols[self.nspecial:], self.count[self.nspecial:]))
            )
        )
        for symbol, count in c.most_common(nwords - self.nspecial):
            if count >= threshold:
                new_indices[symbol] = len(new_symbols)
                new_symbols.append(symbol)
                new_count.append(count)
            else:
                break

        assert len(new_symbols) == len(new_indices)

        self.count = list(new_count)
        self.symbols = list(new_symbols)
        self.indices = new_indices

        self.pad_to_multiple_(padding_factor)

    def _get_meta(self):
        return [], []

    def _load_meta(self, lines):
        return 0


    @classmethod
    def load(cls, f):
        """Loads the dictionary from a text file with the format:

        ```
        <symbol0> <count0>
        <symbol1> <count1>
        ...
        ```
        """
        d = cls()
        d.add_from_file(f)
        return d

    def add_from_file(self, f):
        """
        Loads a pre-existing dictionary from a text file and adds its symbols
        to this instance.
        """
        if isinstance(f, str):
            try:
                with PathManager.open(f, "r", encoding="utf-8") as fd:
                    self.add_from_file(fd)
            except FileNotFoundError as fnfe:
                raise fnfe
            except UnicodeError:
                raise Exception(
                    "Incorrect encoding detected in {}, please "
                    "rebuild the dataset".format(f)
                )
            return

        lines = f.readlines()
        indices_start_line = self._load_meta(lines)

        for line in lines[indices_start_line:]:
            try:
                line, field = line.rstrip().rsplit(" ", 1)
                if field == "#fairseq:overwrite":
                    overwrite = True
                    line, field = line.rsplit(" ", 1)
                else:
                    overwrite = False
                count = int(field)
                word = line
                if word in self and not overwrite:
                    raise RuntimeError(
                        "Duplicate word found when loading Dictionary: '{}'. "
                        "Duplicate words can overwrite earlier ones by adding the "
                        "#fairseq:overwrite flag at the end of the corresponding row "
                        "in the dictionary file. If using the Camembert model, please "
                        "download an updated copy of the model file."
                        .format(word)
                    )
                self.add_symbol(word, n=count, overwrite=overwrite)
            except ValueError:
                raise ValueError(
                    "Incorrect dictionary format, expected '<token> <cnt> [flags]'"
                )
