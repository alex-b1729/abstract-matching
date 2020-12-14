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
from six import iteritems
import pprint


def gen_dict_corpus(names, assignment_group_path='assignment_groups'):
    """Returns (dictionary, corpus)"""
    # abstract file relative paths
    abstr_paths = []
    for row in names.itertuples():
        for num in range(1, row[2]+1):
            abstr_paths.append('{}/{}_{}_{}.text'.format(row[0][0].upper(), row[0], row[1], num))
    
    # iterator loads all abstracts
    dictionary = gensim.corpora.Dictionary(CleanAbstracts(abstr_paths))
    
    # remove words that appear only once accross all documents
    once_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs)
                if docfreq == 1]
    dictionary.filter_tokens(once_ids)
    # remove gaps in ids
    dictionary.compactify()
    
    # memory friendly corpus
    corpus = IterCorpus(abstr_paths)
    
    return dictionary, corpus
    

def extract_abstract(os_type, paper_file_path='papers_to_assign'):
    """Converts pdfs to txt and saves abstracts"""
    # txt abstract path
    abstr_txt_path = os.path.join(paper_file_path, 'abstract_txts')
    
    # abstract already converted to txt
    txt_names = []
    with os.scandir(os.path.join(os.getcwd(), abstr_txt_path)) as it:
        for entry in it:
            if entry.name.endswith('.txt') and entry.is_file() and not entry.name.startswith('.'):
                txt_names.append(entry.name)
                
    # list of paths to each .pdf not already converted to txt
    submissions = []
    sub_names = []
    with os.scandir(os.path.join(os.getcwd(), paper_file_path)) as it:
        for entry in it:
            if entry.name.endswith('.pdf') and entry.is_file() and not entry.name.startswith('.') and entry.name not in txt_names:
                submissions.append(entry.path)
                sub_names.append(entry.name[:-4])

# =============================================================================
#     # below needs work
# =============================================================================
    for paper_index in range(len(sub_names)):
        txt_path = os.path.join(abstr_txt_path, '{}.txt'.format(sub_names[paper_index]))
        paper_path = papers[paper_index]
        if txt_path not in query_paths:
            comd = '{} {} {}'.format(quote(xpdf_path), quote(paper_path), quote(txt_path))
            # get text from .pdf
            os.system(comd)    

    
class CleanAbstracts():
    def __init__(self, txt_paths):
        self.txt_paths = txt_paths
    
    def __iter__(self):
        stop_words = set(stopwords.words('english'))
        ps = PorterStemmer()
        for path in self.txt_paths:
            with open(os.path.join('data/learning_abstracts', path), encoding='utf-8', errors='ignore') as file:
                yield [ps.stem(wrd) for wrd in 
                       gensim.utils.simple_preprocess(file.read(), deacc=True)
                       if wrd not in stop_words]

class IterCorpus():
    def __init__(self, txt_paths):
        self.txt_paths = txt_paths
        
    def __iter__(self):
        stop_words = set(stopwords.words('english'))
        ps = PorterStemmer()        
        for path in self.txt_paths:
            with open(os.path.join('data/learning_abstracts', path), encoding='utf-8', errors='ignore') as file:
                yield dictionary.doc2bow([ps.stem(wrd) for wrd in 
                                          gensim.utils.simple_preprocess(file.read(), deacc=True)
                                          if wrd not in stop_words])





# =============================================================================
# 
# 
# path_list = ['/Users/abrefeld/Dropbox/UK/RA_assignments/JCF_summer_2020/Abstract_collection/learning_abstracts/F/Faccio_Mara_1.txt',
#              '/Users/abrefeld/Dropbox/UK/RA_assignments/JCF_summer_2020/Abstract_collection/learning_abstracts/F/Faccio_Mara_2.txt',
#              '/Users/abrefeld/Dropbox/UK/RA_assignments/JCF_summer_2020/Abstract_collection/learning_abstracts/F/Faccio_Mara_3.txt']
# dictionary = gensim.corpora.Dictionary(CleanAbstracts(path_list))
# # remove words that appear only once accross all documents
# once_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs)
#             if docfreq == 1]
# dictionary.filter_tokens(once_ids)
# # remove gaps in ids
# dictionary.compactify()
# print(dictionary.token2id)
# corpus = IterCorpus(path_list)
# pprint.pprint(list(corpus))
#     
# =============================================================================



