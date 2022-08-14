# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
BART: Denoising Sequence-to-Sequence Pre-training for
Natural Language Generation, Translation, and Comprehension

Code for Cross-lingual adaptation/fine-tuning derived from BART.
"""

import logging

import torch
import torch.nn as nn

from fairseq import utils
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import TransformerModel, TransformerDecoderCVT, TransformerEncoderCVT
from fairseq.modules.transformer_sentence_encoder import init_bert_params

from fairseq.models.fairseq_encoder import EncoderOut

from fairseq.models.fairseq_model import BaseFairseqModel

from fairseq.models.bart.model import BARTClassificationHead

from fairseq.checkpoint_utils import prune_state_dict

logger = logging.getLogger(__name__)


@register_model('bart_mtl')
class BARTMtlModel(TransformerModel):

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

        # We follow BERT's random weight initialization
        self.apply(init_bert_params)

        self.classification_heads = nn.ModuleDict()

        self.args = args

        self.multi_tasks = nn.ModuleDict()

    @staticmethod
    def add_args(parser):
        super(BARTMtlModel, BARTMtlModel).add_args(parser)
        parser.add_argument(
            '--pooler-dropout', type=float, metavar='D',
            help='dropout probability in the masked_lm pooler layers'
        )
        parser.add_argument(
            '--pooler-activation-fn',
            choices=utils.get_available_activation_fns(),
            help='activation function to use for pooler layer'
        )
        parser.add_argument(
            '--freeze-embeds', action='store_true',
        help='freeze embeddings on fine-tuning'
        )

        parser.add_argument('--load-checkpoint-mtasks', action='store_true',
                            help='(re-)register and load task-specific sub-network when loading checkpoints')

    @property
    def supported_targets(self):
        return {'self'}

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        transformer = super().build_model(args, task)

        if hasattr(task, 'aux_src_classif_tasks'):
            transformer.task_dict = task.aux_src_classif_tasks
            for task in task.aux_src_classif_tasks.keys():
                print("*\t Register classification for task: {} (#classes:{})".format(task, transformer.task_dict[task]))
                if task == 'disc':
                    transformer.register_discourse_cloze(task, transformer.task_dict[task])
                else:
                    transformer.register_classification_task(task, transformer.task_dict[task])

        #transformer.decoder.register_output_project_cvt()

        return transformer

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return TransformerEncoderCVT(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TransformerDecoderCVT(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )

    def forward(
        self, src_tokens, src_lengths, prev_output_tokens,
        features_only=False, classification_head_name=None, **kwargs
    ):
        if classification_head_name is not None:
            features_only = True

        encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            **kwargs,
        )

        x, extra = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            **kwargs,
        )

        if classification_head_name is not None:
            sentence_representation = x[
                src_tokens.eq(self.encoder.dictionary.eos()), :
            ].view(x.size(0), -1, x.size(-1))[:, -1, :]
            x = self.classification_heads[classification_head_name](
                sentence_representation
            )
        return x, extra


    def forward_document_classif(self, src_tokens, src_lengths,
                                       block_pos_left, block_pos_right, block_mask,
                                       maskToken_pos, maskToken_masks, **kwargs):
        """
        :param src_tokens:
        :param src_lengths:
        :param block_pos_left, block_pos_right:  long tensor [b x sent-boundaries]
        :param block_mask: bool tensor [b x sent-boundaries] False means DO NOT mask, True means needs masking
        :param maskToken_pos: long tensor [b x nb-target-tokens]
        :param maskToken_masks: bool tensor [b x nb-target-tokens] False means DO NOT mask, True means needs masking
        :param kwargs:
        :return:
        """
        encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            **kwargs,
        )
        # gather indices corresponding to sentence separation tokens, apply mask,
        # get tokens position embeddings, combine all and feed-forward to for prediction
        # span boundaries objective

        t = encoder_out.encoder_out.transpose(0, 1)

        left_boundary_tokens = t[torch.arange(block_pos_left.size(0)).long(), block_pos_left]  #* block_mask.transpose(0,1)
        right_boundary_tokens = t[torch.arange(block_pos_right.size(0)).long(), block_pos_right]  #* block_mask.transpose(0,1)
        span_positions = self.encoder.embed_positions(maskToken_pos)
        # TODO positions here should be relative inside the span (as SPANBert)???

        output = self.multi_tasks['disc'](torch.cat((left_boundary_tokens, right_boundary_tokens, span_positions), 2))

        return output, {}

    def forward_cvt(
        self, src_tokens, src_lengths, prev_output_tokens,
        features_only=False, **kwargs
    ):

        if len(self.args.cvt_layers.split()) > 0:
            return_specific_hiddens = [int(x) for x in self.args.cvt_layers.split()] #[6,10]
            encoder_out = self.encoder(
                src_tokens,
                src_lengths=src_lengths,
                return_specific_hiddens=return_specific_hiddens,
                **kwargs,
            )

            ret_x = []
            ret_extra = []
            views = encoder_out.encoder_states

            if return_specific_hiddens:
                for i in range(len(return_specific_hiddens)):
                    encoder_out_view = EncoderOut(
                        encoder_out=views[i],  # T x B x C
                        encoder_padding_mask=encoder_out.encoder_padding_mask,
                        # convert zeroes into paddings,  # B x T
                        encoder_embedding=encoder_out.encoder_embedding,  # B x T x C
                        encoder_states=None, ##encoder_out.encoder_states,  # List[T x B x C]
                        src_tokens=None,
                        src_lengths=None,
                    )
                    x, extra = self.decoder(
                        prev_output_tokens,
                        encoder_out=encoder_out_view,
                        features_only=features_only,
                        cvt_instance=True,
                        **kwargs,
                    )
                    ret_x.append(x)
                    ret_extra.append(extra)

        elif self.args.cvt_p > 0:
            ### mask here encoder output, encoder_padding_mask
            ### binary ByteTensor of shape `(batch, src_len)` where padding elements are indicated by ``1``.
            t = encoder_out.encoder_out.transpose(0,1)
            sz = encoder_out.encoder_padding_mask.size()
            dynmask = torch.cuda.FloatTensor(sz[0], sz[1]).uniform_() <= self.args.cvt_p
            ##t = t * ~ dynmask.view(sz[0], sz[1], -1).expand(sz[0], sz[1], t.size()[-1])
            encoder_out_view = EncoderOut(
                encoder_out=t,  # T x B x C
                encoder_padding_mask=encoder_out.encoder_padding_mask + dynmask,  # convert zeroes into paddings,  # B x T
                encoder_embedding=encoder_out.encoder_embedding,  # B x T x C
                encoder_states=None,  # List[T x B x C]
                src_tokens=None,
                src_lengths=None,
            )
            x, extra = self.decoder(
                prev_output_tokens,
                encoder_out=encoder_out_view,
                features_only=features_only,
                cvt_instance=True,
                **kwargs,
            )
            ret_x = [x]
            ret_extra = [extra]
        else:
            raise ValueError('If using CVT a method for creating input views should be specified.')

        return ret_x, ret_extra



    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        checkpoint_file='model.pt',
        data_name_or_path='.',
        bpe='gpt2',
        **kwargs,
    ):
        from fairseq import hub_utils
        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            bpe=bpe,
            load_checkpoint_heads=True,
            **kwargs,
        )
        return BARTHubInterface(x['args'], x['task'], x['models'][0])


    def register_classification_head(self, name, num_classes=None, inner_dim=None, **kwargs):
        """Register a classification head."""
        logger.info("Registering classification head: {0}".format(name))
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    'and inner_dim {} (prev: {})'.format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        self.classification_heads[name] = BARTClassificationHead(
            self.args.encoder_embed_dim,
            inner_dim or self.args.encoder_embed_dim,
            num_classes,
            self.args.pooler_activation_fn,
            self.args.pooler_dropout,
        )

    def register_classification_task(self, name, num_classes=None, inner_dim=None, **kwargs):
        """Register a classification head."""
        logger.info("Registering classification task: {0}".format(name))
        if name in self.multi_tasks:
            prev_num_classes = self.multi_tasks[name].out_proj.out_features
            prev_inner_dim = self.multi_tasks[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering task "{}" with num_classes {} (prev: {}) '
                    'and inner_dim {} (prev: {})'.format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        self.multi_tasks[name] = BARTClassificationHead(
            self.args.encoder_embed_dim,
            inner_dim or self.args.encoder_embed_dim,
            num_classes,
            self.args.pooler_activation_fn,
            self.args.pooler_dropout,
        )

    def register_discourse_cloze(self, name, num_classes=None, inner_dim=None, **kwargs):
        """Register a classification head."""
        logger.info("Registering classification task: {0}".format(name))

        self.multi_tasks[name] = ClozeClassificationHead(
            self.args.encoder_embed_dim*3, # concatenation of span boundaries & position
            inner_dim or self.args.encoder_embed_dim,
            num_classes,
            self.args.pooler_activation_fn,
            self.args.pooler_dropout,
        )

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)

        prefix = name + '.' if name != '' else ''
        current_head_names = [] if not hasattr(self, 'classification_heads') else \
            self.classification_heads.keys()

        # Handle new classification heads present in the state dict.
        keys_to_delete = []
        for k in state_dict.keys():
            if not k.startswith(prefix + 'classification_heads.'):
                continue

            head_name = k[len(prefix + 'classification_heads.'):].split('.')[0]
            num_classes = state_dict[prefix + 'classification_heads.' + head_name + '.out_proj.weight'].size(0)
            inner_dim = state_dict[prefix + 'classification_heads.' + head_name + '.dense.weight'].size(0)

            if getattr(self.args, 'load_checkpoint_heads', False):
                if head_name not in current_head_names:
                    self.register_classification_head(head_name, num_classes, inner_dim)
            else:
                if head_name not in current_head_names:
                    logger.warning(
                        'deleting classification head ({}) from checkpoint '
                        'not present in current model: {}'.format(head_name, k)
                    )
                    keys_to_delete.append(k)
                elif (
                    num_classes != self.classification_heads[head_name].out_proj.out_features
                    or inner_dim != self.classification_heads[head_name].dense.out_features
                ):
                    logger.warning(
                        'deleting classification head ({}) from checkpoint '
                        'with different dimensions than current model: {}'.format(head_name, k)
                    )
                    keys_to_delete.append(k)
        for k in keys_to_delete:
            del state_dict[k]

        def truncate_emb(key):
            if key in state_dict:
                state_dict[key] = state_dict[key][:-1, :]

        # When fine-tuning on translation task, remove last row of
        # embedding matrix that corresponds to mask_idx token.
        loaded_dict_size = state_dict['encoder.embed_tokens.weight'].size(0)
        if loaded_dict_size == len(self.encoder.dictionary) + 1 and '<mask>' not in self.encoder.dictionary:
            truncate_emb('encoder.embed_tokens.weight')
            truncate_emb('decoder.embed_tokens.weight')
            truncate_emb('encoder.output_projection.weight')
            truncate_emb('decoder.output_projection.weight')

        # When continued pre-training on new set of languages for mbart,
        # add extra lang embeddings at the end of embed_tokens.
        # Note: newly added languages are assumed to have been added at the end.
        if (self.args.task == 'multilingual_denoising' ) \
                and loaded_dict_size < len(self.encoder.dictionary):
            logger.info(
                "Adding extra language embeddings not found in pretrained model for "\
                "continued pretraining of MBART on new set of languages."
            )
            loaded_mask_token_embedding = state_dict['encoder.embed_tokens.weight'][-1, :]

            num_langids_to_add = len(self.encoder.dictionary) - loaded_dict_size
            embed_dim = state_dict['encoder.embed_tokens.weight'].size(1)

            new_lang_embed_to_add = torch.zeros(num_langids_to_add, embed_dim)
            nn.init.normal_(
                new_lang_embed_to_add,
                mean=0,
                std=embed_dim ** -0.5
            )
            new_lang_embed_to_add = new_lang_embed_to_add.to(
                dtype=state_dict['encoder.embed_tokens.weight'].dtype,
            )

            state_dict['encoder.embed_tokens.weight'] = torch.cat([
                state_dict['encoder.embed_tokens.weight'][:loaded_dict_size-1, :],
                new_lang_embed_to_add,
                loaded_mask_token_embedding.unsqueeze(0)]
            )
            state_dict['decoder.embed_tokens.weight'] = torch.cat([
                state_dict['decoder.embed_tokens.weight'][:loaded_dict_size-1, :],
                new_lang_embed_to_add,
                loaded_mask_token_embedding.unsqueeze(0)]
            )

        # Copy any newly-added classification heads into the state dict
        # with their current weights.
        if hasattr(self, 'classification_heads'):
            cur_state = self.classification_heads.state_dict()
            for k, v in cur_state.items():
                if prefix + 'classification_heads.' + k not in state_dict:
                    logger.info('Overwriting ' + prefix + 'classification_heads.' + k)
                    state_dict[prefix + 'classification_heads.' + k] = v

        prefix = name + '.' if name != '' else ''
        current_task_names = [] if not hasattr(self, 'multi_tasks') else \
            self.multi_tasks.keys()

        # Handle new classification heads present in the state dict.
        keys_to_delete = []
        for k in state_dict.keys():
            if not k.startswith(prefix + 'multi_tasks.'):
                continue

            task_name = k[len(prefix + 'multi_tasks.'):].split('.')[0]
            num_classes = state_dict[prefix + 'multi_tasks.' + task_name + '.out_proj.weight'].size(0)
            inner_dim = state_dict[prefix + 'multi_tasks.' + task_name + '.dense1.weight'].size(0)

            if getattr(self.args, 'load_checkpoint_mtasks', False) and self.args.load_checkpoint_mtasks:
                if task_name not in current_task_names:
                    if task_name == 'disc':
                        self.register_discourse_cloze(task, self.task_dict[task])
                    else:
                        self.register_classification_task(task, self.task_dict[task])
            else:
                if task_name not in current_head_names:
                    logger.warning(
                        'deleting classification head ({}) from checkpoint '
                        'not present in current model: {}'.format(task_name, k)
                    )
                    keys_to_delete.append(k)
                elif (
                    num_classes != self.multi_tasks[task_name].out_proj.out_features
                    or inner_dim != self.multi_tasks[task_name].dense.out_features
                ):
                    logger.warning(
                        'deleting classification task ({}) from checkpoint '
                        'with different dimensions than current model: {}'.format(task_name, k)
                    )
                    keys_to_delete.append(k)
        for k in keys_to_delete:
            del state_dict[k]

        # Copy any newly-added classification heads into the state dict
        # with their current weights.
        if hasattr(self, 'multi_tasks'):
            cur_state = self.multi_tasks.state_dict()
            for k, v in cur_state.items():
                if prefix + 'multi_tasks.' + k not in state_dict:
                    logger.info('Overwriting ' + prefix + 'multi_tasks.' + k)
                    state_dict[prefix + 'multi_tasks.' + k] = v


    def freeze_decoder(self):
        """ LP. Added as from Huggingface"""
        # https://github.com/huggingface/transformers/blob/ec0267475c16a1913e64cb4f81fd54d153e3d815/examples/seq2seq/finetune.py#L116

        def freeze_params(model):
            print("*\tFreezing decoder...")
            for name, param in model.named_parameters():
                if not ('encoder_attn' in name or 'layer_norm' in name or 'layernorm' in name):
                    print(name)
                    param.requires_grad = False

        freeze_params(self.decoder)

    def freeze_encoder(self):
        """ LP. Added as from Huggingface"""
        # https://github.com/huggingface/transformers/blob/ec0267475c16a1913e64cb4f81fd54d153e3d815/examples/seq2seq/finetune.py#L116

        exclude_layers = getattr(self.args, 'freeenc_exclude_layers', None)
        exclude_layers = exclude_layers.split(',') if exclude_layers else []
        def freeze_params(model):
            for name, param in model.named_parameters():
                if not ('layer_norm' in name or 'layernorm' in name or
                        (name.startswith('layers.') and name.split('.')[1] in exclude_layers)
                ):
                    print(name)
                    param.requires_grad = False

        for d in [self.encoder, self.decoder]:
             freeze_params(d.embed_positions)
             freeze_params(d.embed_tokens)


        freeze_params(self.encoder)
        logger.info("All encoder parameters frozen (excl. {}).".format(exclude_layers))


        freeze_params(self.encoder)
        logger.info("All encoder parameters frozen (excl. {}).".format(exclude_layers))

    def freeze_embeds(self):
        """ LP. Added as from Huggingface"""
        # https://github.com/huggingface/transformers/blob/ec0267475c16a1913e64cb4f81fd54d153e3d815/examples/seq2seq/finetune.py#L116

        def freeze_params(model):
            for name, param in model.named_parameters():
                if ('embed_tokens' in name or 'embed_positions' in name \
                        or 'layernorm_embedding' in name):
                    print(name)
                    param.requires_grad = False

        freeze_params(self.encoder)
        freeze_params(self.decoder)


class ClozeClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
            self,
            input_dim,
            inner_dim,
            num_classes,
            activation_fn,
            pooler_dropout,
    ):
        super().__init__()
        self.dense1 = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense1(x)
        x = self.activation_fn(x)
        x = self.out_proj(x)
        return x


@register_model_architecture('bart_mtl', 'bart_mtl_large')
def bart_large_architecture(args):
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4*1024)
    args.encoder_layers = getattr(args, 'encoder_layers', 12)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 16)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', True)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 12)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 16)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', False)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', True)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.max_target_positions = getattr(args, 'max_target_positions', 1024)
    args.max_source_positions = getattr(args, 'max_source_positions', 1024)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', True)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', True)

    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, 'no_scale_embedding', True)
    args.layernorm_embedding = getattr(args, 'layernorm_embedding', True)

    args.activation_fn = getattr(args, 'activation_fn', 'gelu')
    args.pooler_activation_fn = getattr(args, 'pooler_activation_fn', 'tanh')
    args.pooler_dropout = getattr(args, 'pooler_dropout', 0.0)


@register_model_architecture('bart_mtl', 'bart_mtl_base')
def bart_base_architecture(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 768)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4*768)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 12)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 12)
    bart_large_architecture(args)


@register_model_architecture('bart_mtl', 'mbart_mtl_large')
def mbart_large_architecture(args):
    args.no_scale_embedding = getattr(args, 'no_scale_embedding', False)
    bart_large_architecture(args)

@register_model_architecture('bart_mtl', 'mbart_mtl_large_ftfew')
def mbart_large_architecture(args):
    args.no_scale_embedding = getattr(args, 'no_scale_embedding', False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    bart_large_architecture(args)

@register_model_architecture('bart_mtl', 'mbart_mtl_base')
def mbart_base_architecture(args):
    args.no_scale_embedding = getattr(args, 'no_scale_embedding', False)
    bart_base_architecture(args)


@register_model_architecture('bart_mtl', 'mbart_mtl_mtl_base_wmt20')
def mbart_base_wmt20_architecture(args):
    args.layernorm_embedding = getattr(args, 'layernorm_embedding', False)
    mbart_base_architecture(args)
