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
    
    
    
# =============================================================================
#     
#     # common word list
#     stop_words = set(stopwords.words('english'))
#     ps = PorterStemmer()
#     
#     # creates list of lists containing words of abstracts
#     # doesn't include stop words or punctuation
#     # stems words so that 'investing' and 'investment' become 'invest'
#     stemmed_abstrs = []
#     for abstr_path in abstr_paths:
#         with open(abstr_path, encoding='utf-8', mode='r', errors='ignore') as file:
#             wrds = word_tokenize(clean_txt(file.read()))
#             # remove periods, commas and common english words
#             abstr = [ps.stem(wrd) for wrd in wrds if wrd not in stop_words and wrd not in string.punctuation]
#             stemmed_abstrs.append(abstr)
#     
#     # remove words that appear more than once accross all abstracts.  
#     frequency = defaultdict(int)
#     for abstr in stemmed_abstrs:
#         for wrd in abstr:
#             frequency[wrd] += 1
#     cleaned_abstrs = [
#                       [wrd for wrd in abstr if frequency[wrd] > 1]
#                       for abstr in stemmed_abstrs
#                       ]
#     
#     # dictionary mapping words to id#s
#     dictionary = gensim.corpora.Dictionary(cleaned_abstrs)
#     dictionary.save('dictionaries/committee_abstr_kwrds_ten_track_ext.dict')
# #    print(dictionary.token2id)
# #    print(len(dictionary.token2id))
#     
#     # creates a 'bag of words', saves to disk
#     corpus = [dictionary.doc2bow(text) for text in cleaned_abstrs]
#     gensim.corpora.MmCorpus.serialize('corpus/committee_abstr_kwrds_ten_track_ext.mm', corpus)
# =============================================================================
    
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




# =============================================================================
# def clean_txt(txt, stem=False):
#     '''
#     Cleans string by removing special characters, numbers, and removing
#     the possesive ' or 's from the end of words.  Also breaks hyphenated words 
#     into two words. If the hyphenated word starts with "non-" it is kept 
#     as a prefix.
#     
#     If stem=True, the nltk.stem.porter PorterStemmer function is used
#     to reduce each word to its stem for more effective analysis
#     
#     Requires string.punctuation and nltk.stem.porter librarys.  
#     '''
#     ps = PorterStemmer()
#     
#     txt_lst = txt.split()
#     txt_clean = []
#     
#     for word in txt_lst:
#         if word not in string.punctuation:
#             word_clean = ''
#             aps = False  # if char is apostrophy
#             for char in word:
#                 # if not a special character and no apostrophy has been found
#                 if char.isalpha() and not aps:
#                     word_clean += char
#                 elif char == '-' and word_clean == 'non':
#                     word_clean += char
#                 # keep if number followed by '...' to keep jel codes
#                 elif char.isnumeric() and word_clean!='' and any(word_clean[0] == x for x in 'GQJHDRELKNM'):
#                     word_clean += char
#                 else:
#                     if char == "'":
#                         aps = True
#                     elif char == '-':
#                         word_clean += ' '
#                     else:
#                         word_clean += ' '
#             if stem:
#                 txt_clean.append(ps.stem(word_clean))
#             else:
#                 txt_clean.append(word_clean)
#             
#     txt = ' '.join(txt_clean).lower()
#     
#     return txt
# 
# def file_hash(file_path):
#     """Outputs hash of file as str"""
#     hash_func = hashlib.sha256()
#     with open(file_path, 'rb') as f:
#         hash_func.update(f.read())
#     return hash_func.hexdigest()
# =============================================================================
