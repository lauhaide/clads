import argparse
import os
import glob

from fairseq.data import Dictionary
from fairseq.tokenizer import tokenize_line

HOME='/home/lperez/storage/datasets/'
corpus_data = ['cnndm/cnn_dm/test.spm.source', 'cnndm/cnn_dm/test.spm.target',
               'cnndm/cnn_dm/train.spm.source', 'cnndm/cnn_dm/train.spm.target',
               'cnndm/cnn_dm/val.spm.source', 'cnndm/cnn_dm/val.spm.target' ]
MLSUM_HOME = os.path.join(HOME, 'MLSUM')

LANGS = ['cs', 'de', 'fr', 'es', 'ru', 'tu']
SPLITS = ['train', 'val', 'test']
SRCTGT = ['src', 'tgt']

WIKIS_DIR = '/home/lperez/storage/datasets/XWikis-debug2/*/spm/*'

def pad_dict(d: Dictionary, num_extra_symbols: int, padding_factor: int = 8) -> None:
    i = 0
    while (len(d) + num_extra_symbols) % padding_factor != 0:
        symbol = f"madeupword{i:04d}"
        d.add_symbol(symbol, n=0)
        i += 1

def main() -> None:
    parser = argparse.ArgumentParser(description="Build vocabulary from corpus data.")
    parser.add_argument("--langs", type=str, required=True, help="The pre-trained model languages.")
    parser.add_argument("--output", type=str, required=True, help="The vocabulary file.")
    parser.add_argument("--dataset", type=str, required=True, help="XWikis or XNews.")
    args = parser.parse_args()



    langs = args.langs.split(",")
    ft_dict = Dictionary()

    if args.dataset == 'XNews':
        # Voc from CNNDM
        for data_path in corpus_data:
            Dictionary.add_file_to_dictionary(HOME+data_path, ft_dict, tokenize_line, 4)

        # Voc from MLSUM
        for l in LANGS:
            for s in SPLITS:
                for st in SRCTGT:
                    Dictionary.add_file_to_dictionary(os.path.join(MLSUM_HOME,
                                                                   "{}/{}.txt-raw.spm.{}".format(l,s,st)),
                                                      ft_dict, tokenize_line, 4)
    if args.dataset == 'XWikis':
        # Voc from XWikis
        nbFiles = 0
        for filename in glob.glob(WIKIS_DIR):
            print(filename)
            if ".spm." in filename:
                nbFiles +=1
                Dictionary.add_file_to_dictionary(filename, ft_dict, tokenize_line, 4)
        print("\t* {} spm files processed")
    ft_dict.finalize(padding_factor=0)
    pad_dict(ft_dict, len(langs) + 1)
    ft_dict.save(args.output)

if __name__ == "__main__":
    main()

# LL=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN
# python fairseq/data/build_vocab.py --langs $LL --dataset XNews --output /home/lperez/storage/pretrained/mbart.trimMLSUM/dict.txt
# python fairseq/data/build_vocab.py --langs $LL --dataset XWikis --output /home/lperez/storage/pretrained/mbart.trimXWikis/dict.txt