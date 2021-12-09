# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn

from fairseq import utils
from fairseq.criterions import FairseqCriterion, register_criterion

from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion

@register_criterion('mbart_mtl_loss')
class MBARTMtlLoss(FairseqCriterion):
    """This is a composite loss that, given a list of model outputs and a list of targets,
    computes an average of losses for each output-target pair"""

    def __init__(self, task, sentence_avg=None):
        super().__init__(task)


    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        LabelSmoothedCrossEntropyCriterion.add_args(parser)
        # TODO for this wrapper to combined losses ??
        # fmt: on

    @staticmethod
    def build_underlying_criterion(args, task):
        """ TODO: if all works... then improve this criterion building! read from config files, for the moment all tasks we have use the same criterion"""
        underlying_criterions = {}
        saved_criterion = args.criterion
        args.criterion = 'label_smoothed_cross_entropy'
        assert saved_criterion != 'label_smoothed_cross_entropy'
        underlying_criterions['mainsum'] = task.build_criterion(args)
        args.criterion = saved_criterion

        return underlying_criterions

    @classmethod
    def build_criterion(cls, args, task):
        underlying_criterions = MBARTMtlLoss.build_underlying_criterion(args, task)

        class FakeModel(nn.Module):

            def __init__(self, model, net_out, target):
                super().__init__()
                self.model = model
                self.net_out = net_out
                self.target = target

            def forward(self, **unused):
                return self.net_out

            def get_normalized_probs(self, net_output, log_probs, sample=None):
                return self.model.get_normalized_probs(net_output, log_probs, sample=sample)

            def get_targets(self, *unused):
                return self.target

            @property
            def decoder(self):
                return self.model.decoder

        class _CompositeLoss(FairseqCriterion):

            def __init__(self, task, underlying_criterions, args):
                super().__init__(task)
                self.underlying_criterions = underlying_criterions
                self.task = task
                self.generator = None
                self.args = args

            def forward(self, model, sample, reduce=True):
                lang_task_pair, sample = sample
                task_type = self.task.getTaskType(lang_task_pair)

                output_lens = 0
                if task_type == 'mainsum':
                    net_outputs = model(**sample['net_input'])

                elif task_type == 'disc':
                    net_outputs = model.forward_document_classif(**sample['net_input'])

                elif task_type == 'cvt':
                    if 'few' in sample.keys() and sample['few'][0]:
                        # if we are in the few instances, just standard forward
                        net_outputs = model(**sample['net_input'])
                    else:
                        # use different perturbations of the input and get the network outputs
                        if self.args.cvt_future_pred:
                            #prevouttokens
                            sample['net_input']['prev_output_tokens'][:, -1] = 1 ##TODO if works fix this in the code
                            net_outputs = model(**sample['net_input'])
                            # target
                            sample['target'] = torch.roll(sample['target'], shifts=-1, dims=1)
                            sample['target'][:, -1] = 1

                        else:
                            net_outputs = model.forward_cvt(**sample['net_input'])
                            output_lens = len(net_outputs[0])

                targets = [sample['target']]
                if output_lens > 1:
                    targets = [sample['target']]*output_lens
                outputs = net_outputs[0]
                extras = net_outputs[1]
                if output_lens == 0:
                    outputs = [outputs]
                    extras = [extras]

                bsz = targets[0].size(0)
                loss = net_outputs[0][0].new(1 if reduce else bsz).float().zero_()
                nll_loss = net_outputs[0][0].new(1 if reduce else bsz).float().zero_()

                sample_size = 0
                logging_output = {
                    'nll_loss': nll_loss.data,
                    'ntokens': 0,
                    'nsentences': 0,
                }
                for o, extr, t in zip(outputs, extras, targets):

                    m = FakeModel(model, (o, extr), t) # extr is extra data returned in a dictionary, may be empty dict
                    l, ss, lo = self.underlying_criterions['mainsum'](m, sample, reduce)

                    sample_size += ss
                    logging_output['ntokens'] += lo['ntokens']
                    logging_output['nsentences'] += lo['nsentences']
                    logging_output['nll_loss'] += lo['nll_loss']

                    # weight current cvt loss according to the quality of the generated candidate
                    if task_type == 'cvt' and self.args.cvt_target_wei and not ('few' in sample.keys() and sample['few'][0]):
                        l = l * sample['target_score']

                    if task_type == 'cvt' and not self.args.updates_per_task \
                                            and self.args.cvt_proportion>0 \
                                            and not ('few' in sample.keys() and sample['few'][0]):
                        l = l *  self.args.cvt_proportion # weight when evaluated on unlabelled data

                    if task_type == 'cvt' and not self.args.updates_per_task \
                                            and self.args.few_proportion>0 \
                                            and ('few' in sample.keys() and sample['few'][0]):
                        l = l *  self.args.few_proportion #weight when evaluated on few data

                    if task_type != 'cvt' and self.args.mono_proportion > 0 :
                        l = l * self.args.mono_proportion

                    loss += l

                    logging_output['loss'] = utils.item(loss.data) if reduce else loss.data
                    logging_output['sample_size'] = sample_size

                return loss, sample_size, logging_output

            @staticmethod
            def aggregate_logging_outputs(logging_outputs):
                return underlying_criterions['mainsum'].__class__.aggregate_logging_outputs(logging_outputs)

            @staticmethod
            def reduce_metrics(logging_outputs) -> None:
                underlying_criterions['mainsum'].__class__.reduce_metrics(logging_outputs)

            @staticmethod
            def logging_outputs_can_be_summed() -> bool:
                """
                Whether the logging outputs returned by `forward` can be summed
                across workers prior to calling `reduce_metrics`. Setting this
                to True will improves distributed training speed.
                """
                return True

        return _CompositeLoss(task, underlying_criterions, args)


