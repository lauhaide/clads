This repository contains data and code for our EMNLP 2021 paper [Models and Datasets for Cross-Lingual Summarisation](https://aclanthology.org/2021.emnlp-main.742.pdf). Please contact me at lperez@ed.ac.uk for any question.

Please cite this paper if you use our code or data.

```
@InProceedings{clads-emnlp,
  author =      "Laura Perez-Beltrachini and Mirella Lapata",
  title =       "Models and Datasets for Cross-Lingual Summarisation",
  booktitle =   "Proceedings of The 2021 Conference on Empirical Methods in Natural Language Processing ",
  year =        "2021",
  address =     "Punta Cana, Dominican Republic",
}
```


## The XWikis Corpus

**You can create the corpus using the instructions below. The original XWikis Corpus is available in [this data repository](https://datashare.ed.ac.uk/handle/10283/4188).**

Instructions to re-create our corpus and extract other languages are available [here](./XWikis-Corpus).



## Cross-lingual Summarisation Code


Our code is based on [Fairseq](https://github.com/pytorch/fairseq) and [mBART](https://github.com/pytorch/fairseq/tree/main/examples/mbart)/[mBART50](https://github.com/pytorch/fairseq/blob/main/examples/multilingual/README.md). You'll find our clone of Fairseq and the code extension to implement our models [here](./fairseq2020) and instructions to pre-process the data, and train and evaluate our models [here](./fairseq2020/examples/clads/README.md).


## Models' Outputs


