import gzip
import json
import os

import importlib

#from path import Path
#def get_folder(name):
#    mod = importlib.import_module(name)
#    return Path(mod.__path__[0]) #os.path.abspath()

#ASSETS_ROOT = get_folder('lexrank.assets')
ASSETS_ROOT = '/home/lperez/wikigen/data/crosslingualDS/baselines/assets'

file = os.path.join(ASSETS_ROOT, 'stopwords.json.gz')

with gzip.open(file, mode='rt', encoding='utf-8') as fp:
    _STOPWORDS = json.load(fp)

STOPWORDS = {}

for lang, stopwords in _STOPWORDS.items():
    STOPWORDS[lang] = set(stopwords)
