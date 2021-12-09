# Models and Datasets for Cross-Lingual Summarisation
[https://aclanthology.org/2021.emnlp-main.742.pdf]

## Introduction

Fine tunning mBART and mBART50 for cross-lingual summarisation.

## Pre-trained models

We build on the following pre-trained models.

Model | Description | # params | Download
---|---|---|---
`mbart.CC25` | mBART model with 12 encoder and decoder layers trained on 25 languages' monolingual corpus | 610M | [mbart.CC25.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/mbart/mbart.CC25.tar.gz)
`mbart50` | finetune mBART50 on many-to-many |  | [mbart50.ft.nn.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/mbart50/mbart50.ft.nn.tar.gz)


To reduce the size of the model to fit
our GPU availability we carried out the following
modifications. We trimmed the vocabulary to
135K. We first applied the sentencepiece encoder
to the language sets in our XWikis corpus
and the English monolingual data to generate a reduced
dictionary. Then, we trimmed the dictionary and
the models’ embeddings (taking care to map
indices from the original dictionary to the reduced
one). We further slimmed-down the position
embeddings layer from 1,024 to 600.

```
LL=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN

python trim_mbart_emb.py \
   --pre-train-dir ${HOME}/pretrained/mbart50.ft.nn \
   --ft-dict ${HOME}/pretrained/mbart50.trimXWikis/dict.txt \
   --output ${HOME}/pretrained/mbart50.trimXWikis/model.pt \
   --langs $LL
```

Note: ```--ft-dict``` is the dictionary with the reduced vocabulary.

See script [../fairseq/data/trim_mbart_pos.py](../fairseq/data/trim_mbart_pos.py)

## BPE data (same as given for mBART and mBART50)
# download model
wget https://dl.fbaipublicfiles.com/fairseq/models/mbart/mbart.CC25.tar.gz  
tar -xzvf mbart.CC25.tar.gz
# bpe data
install SPM [here](https://github.com/google/sentencepiece)
```bash
SPM=/path/to/sentencepiece/build/src/spm_encode
MODEL=sentence.bpe.model
${SPM} --model=${MODEL} < ${DATA}/${pair}/sentence/${split}.${spair}_documents.txt > ${DATA}/${pair}/sentence/${split}.${spair}.spm.${SRCLANG}
${SPM} --model=${MODEL} < ${DATA}/${pair}/sentence/${split}.${spair}_summaries.txt > ${DATA}/${pair}/sentence/${split}.${spair}.spm.${TGTLANG}
```


## Preprocess data

Similar to mBART and mBART50, we just slightly adapted the *preprocess.py* script.

```bash
DICT=dict.txt
python fairseq_cli/my_preprocess.py \
  --source-lang ${SRC} \
  --target-lang ${TGT} \
  --trainpref ${DATA}/${TRAIN}.spm \
  --validpref ${DATA}/${VALID}.spm \
  --testpref ${DATA}/${TEST}.spm \
  --destdir ${DEST} \
  --thresholdtgt 0 \
  --thresholdsrc 0 \
  --srcdict ${DICT} \
  --tgtdict ${DICT} \
  --workers 200 \
  --task sum_from_pretrained_bart \
  --langs ${langs}
```

## Finetune mBART (mBART50) on Monolingual Data and Supervised Cross-Lingual

```bash
PRETRAIN=/path/to/mbart.trimXWikis2/model_600.pt
langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN
MAX=600

python fairseq-train $EXEC_DATA \
  --encoder-normalize-before --decoder-normalize-before \
  --arch mbart_large --layernorm-embedding \
  --task sum_from_pretrained_bart \
  --source-lang $SRC --target-lang $TGT \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.2 \
  --dataset-impl mmap \
  --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
  --lr-scheduler polynomial_decay --lr 3e-05 --min-lr -1 --warmup-updates 2500 --total-num-update 20000 \
  --dropout 0.3 --attention-dropout 0.1 --weight-decay 0.0 \
  --max-tokens $MAX --update-freq 20 \
  --save-interval 1 --save-interval-updates 5000 --keep-interval-updates 10 --no-epoch-checkpoints \
  --seed 222 --log-format simple --log-interval 10 \
  --reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler \
  --restore-file $PRETRAIN \
  --langs $langs \
  --ddp-backend no_c10d \
  --save-dir $MODEL_PATH \
  --skip-invalid-size-inputs-valid-test \
  --num-workers 10 \
  --truncate-source --max-source-positions $MAX --max-target-positions $MAX \
  --find-unused-parameters \
  --memory-efficient-fp16 --weight-decay 0.01 --clip-norm 0.1 \
  --prepend-bos --blocks
```

Note: for the monolingual data add the following flag to tell which is the monolingual language,
e.g. ```--mono-lang en_XX```, as SRC='src' and TGT='tgt'.

## Adapt Monolingual Summariser to Cross-lingual Task

The following is an example command to run the different model adaptation configurations.
It illustrates how to adapt by fine-tunning only the decoder-encoder attention (**FT** variant in the paper),
for the **cs-en** task, with only 300 few instances.

```bash
export EXEC_DATA="${CLUSTER_HOME}/storage/datasets/XWikis-debug2/en/bin-block2:${CLUSTER_HOME}/storage/datasets/XWikis-debug2/cs-en/bin-block2-v1"
lang_task_pairs='en_XX-mono,cs_CZ-cvt-en_XX'

PRETRAIN=/path/to/mbart.trimXWikis2/model_600.pt # or xwikis-en-mbart-ft-fp16/checkpoint_6_20000.pt
langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN
MAX=600


fairseq-train $EXEC_DATA \
  --encoder-normalize-before --decoder-normalize-before --layernorm-embedding \
  --arch mbart_mtl_large_ftfew \
  --task sum_from_pretrained_bart_mtl \
  --criterion mbart_mtl_loss --label-smoothing 0.2 \
  --dataset-impl mmap \
  --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
  --lr-scheduler polynomial_decay --lr 3e-05 --min-lr -1 --warmup-updates 100 --total-num-update 1000 \
  --dropout 0.3 --attention-dropout 0.1 \
  --max-tokens $MAX --update-freq 8 \
  --save-interval 1 --save-interval-updates 1000 --keep-interval-updates 10 --no-epoch-checkpoints \
  --seed 222 --log-format simple --log-interval 10 \
  --reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler \
  --restore-file $PRETRAIN \
  --langs $langs \
  --ddp-backend no_c10d \
  --save-dir $MODEL_PATH \
  --skip-invalid-size-inputs-valid-test \
  --num-workers 4 \
  --truncate-source --max-source-positions $MAX --max-target-positions $MAX \
  --find-unused-parameters \
  --memory-efficient-fp16 --weight-decay 0.01 --clip-norm 0.1 \
  --prepend-bos --blocks \
  --lang-task-pairs $lang_task_pairs \
  --max-len-b-cvt $MAXLENB --beam-cvt 1 --min-len-cvt $MINLEN --lenpen-cvt $LENPEN --no-repeat-ngram-size-cvt 3 \
  --no-save-optimizer-state \
  --cvt-layers '6,11'  --mono-proportion 0.25 --few-proportion 1 --cvt-proportion 0 \
  --cvt-few 300 --cvt-few-ratio 0.001 --cvt-mono-ratio 0.002 --only-few  \
  --freeze-embeds  --cvt-freeze-dec --cvt-freeze-enc
```

The command-line parameters for the different configurations are the following.

Model Variant | parameters
---|---
300 LF-MAML | ```--cvt-layers '6,11'  --mono-proportion 0.25 --few-proportion 1 --cvt-proportion 0 --cvt-few 300 --cvt-few-ratio 0.001 --cvt-mono-ratio 0.002 --only-few --freeze-embeds```
300 FT | given in the example above
300 CVT | ```--mono-proportion 0.5 --few-proportion 1 --cvt-proportion 0.1 --cvt-few 300 --cvt-few-ratio 1 --cvt-mono-ratio 0.007 --freeze-embeds```
1K LF-MAML | ```  --cvt-layers '6,11'  --mono-proportion 0.5 --few-proportion 1 --cvt-proportion 0 --cvt-few 1000 --cvt-few-ratio 0.01 --cvt-mono-ratio 0.01 --only-few --freeze-embeds```
---|---

## Generate

```
langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN
EVAL_SET=test

fairseq-generate $EXEC_DATA \
  --path $EXP_ROOT/checkpoints/$MODEL_NAME/checkpoint_best.pt \
  --task sum_from_pretrained_bart \
  --gen-subset $EVAL_SET \
  --source-lang $SRC --target-lang $TGT \
  --bpe 'sentencepiece' --sentencepiece-vocab $BPE_DATA/sentence.bpe.model \
  --skip-invalid-size-inputs-valid-test \
  --max-len-b $MAXLENB --beam 5 --min-len $MINLEN --lenpen $LENPEN --no-repeat-ngram-size 3 \
  --sacrebleu --compute-rouge \
  --max-sentences 20 --langs $langs \
  --results-path $RESULTS_PATH \
  --truncate-source --max-source-positions $MAXTOKENS --max-target-positions $MAXTOKENS \
  --memory-efficient-fp16 \
  --prepend-bos \
  --blocks \
  --model-overrides '{"load_checkpoint_mtasks":False}'
```


Note: for the monolingual data add the following flag to tell which is the monolingual language,
e.g. ```--mono-lang en_XX```, as SRC='src' and TGT='tgt'.

Note: for monolingual or cross-lingual supervised the following flag should be used ```--task sum_from_pretrained_bart```

Note: This generation step will create three files: the set of candidate outputs (NAME.candidate),
their corresponding gold summaries (NAME.gold), and their input _documents (NAME.raw_src).


Following BART summarisation evaluation, we evaluate the generated outputs with [files2rouge](https://github.com/pltrdy/files2rouge) package
```
export CLASSPATH=/path/to/stanford-corenlp-full-2016-10-31/stanford-corenlp-3.7.0.jar

# Tokenize hypothesis and target files.
cat $HYPO | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > test.hypo.tokenized
cat $TARGET | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > test.hypo.target
files2rouge test.hypo.tokenized test.hypo.target
```

## Citation

```bibtex
@InProceedings{clads-emnlp,
  author =      "Laura Perez-Beltrachini and Mirella Lapata",
  title =       "Models and Datasets for Cross-Lingual Summarisation",
  booktitle =   "Proceedings of The 2021 Conference on Empirical Methods in Natural Language Processing ",
  year =        "2021",
  address =     "Punta Cana, Dominican Republic",
}
```
