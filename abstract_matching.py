#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# Copyright (C) 2020 Alexander Brefeld <alexander.brefeld@protonmail.com>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import gensim
from shlex import quote
import string
import os
import numpy as np
import re
import hashlib
from scipy.optimize import linear_sum_assignment
from scipy.stats import percentileofscore
import pandas as pd
import urllib.parse
from collections import defaultdict
from shutil import move
import datetime as dt

# project packages
import abstrprep


def main(topics, data_dir_path):
    name_path = 'assignment_groups'

    # open name lists
    editor_names = pd.read_csv(os.path.join(name_path, 'editor_names.csv'))
    editor_names['position'] = 'editor'
    assistant_editor_names = pd.read_csv(os.path.join(name_path, 'assistant_editor_names.csv'))
    assistant_editor_names['position'] = 'assistant_editor'
    referee_names = pd.read_csv(os.path.join(name_path, 'referee_names.csv'))
    referee_names['position'] = 'referee'
    # full list of names
    names = editor_names.append([assistant_editor_names, referee_names], ignore_index=True)

    # find hash of name lists
    names_hash = file_hash(os.path.join(name_path, 'editor_names.csv'), os.path.join(name_path, 'assistant_editor_names.csv'), os.path.join(name_path, 'referee_names.csv'))

    # load / generate dictionary and corpus
    try:
        # load if exists
        dictionary = gensim.corpora.Dictionary.load('{}.dict'.format(names_hash))
        corpus = gensim.corpora.MmCorpus('{}.mm'.format(names_hash))
    except FileNotFoundError:
        # create dictionary & corpus
        dictionary, corpus = abstrprep.gen_dict_corpus(names)
        # save to disk
        dictionary.save(os.path.join('data/dictionaries', '{}.dict'.format(names_hash)))
        gensim.corpora.MMCorpus.serialize(os.path.join('data/corpora', '{}.mm'.format(names_hash)), corpus)
        
    # term frequency - inverse document frequency (tfidf) model
    tfidf = gensim.models.TfidfModel(corpus)
    corpus = tfidf[corpus]
    
    # latent semantic analysis model with tfidf transformed corups
    model = gensim.models.LsiModel(corpus, id2word=dictionary, num_topics=topics)

def file_hash(*args, block_size=262144):
    """Outputs hash of all file_path in *args as str"""
    hash_func = hashlib.sha256()
    for file_path in args:
        with open(file_path, 'rb') as f:
            data_block = f.read(block_size)
            while len(data_block) > 0:
                hash_func.update(data_block)
                data_block = f.read(block_size)
    return hash_func.hexdigest()


if __name__ == '__main__':
    main()
