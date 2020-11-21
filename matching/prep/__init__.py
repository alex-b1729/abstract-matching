#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# Copyright (C) 2020 Alexander Brefeld

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
from scipy.optimize import linear_sum_assignment
from scipy.stats import percentileofscore
import pandas as pd
import urllib.parse
from collections import defaultdict
from shutil import move
import datetime as dt


def get_paths(os_type='mac'):
    # file paths within /paper_author_similarity
    abstr_learning_path = '../../learning_abstracts/'

    
    abstr_paths = []
    fac_index = []
    with os.scandir(abstr_learning_path) as it:
        i = 0
        for entry in it:
            if not entry.name.startswith('.') and entry.is_file() and entry.name.endswith('.txt'):
                abstr_paths.append(entry.path)
                parts = entry.name.split(' ')
                last = parts[1]
                name = '{} {}'.format(parts[0], last).title()
                fac_index.append([i, name])
                i += 1
    fac_dict = dict(fac_index)

def gen_dict_corpus():
    """Reads all learning abstracts and creates 1 dictionary and 3 separate corpora."""
    
    abstr_paths
    
    # common word list
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    
    # creates list of lists containing words of abstracts
    # doesn't include stop words or punctuation
    # stems words so that 'investing' and 'investment' become 'invest'
    abstrs_many = []
    for abstr_path in abstr_paths:
        with open(abstr_path, encoding='utf-8', mode='r', errors='ignore') as file:
            wrds = word_tokenize(clean_txt(file.read()))
            # remove periods, commas and common english words
            abstr = [ps.stem(wrd) for wrd in wrds if wrd not in stop_words and wrd not in string.punctuation]
            abstrs_many.append(abstr)
    
    # remove words that appear more than once accross all abstracts.  
    frequency = defaultdict(int)
    for text in abstrs_many:
        for token in text:
            frequency[token] += 1
    abstrs = [
            [token for token in text if frequency[token] > 1]
            for text in abstrs_many
            ]
    
    # dictionary mapping words to id#s, saves to disk
    dictionary = gensim.corpora.Dictionary(abstrs)
    dictionary.save('dictionaries/committee_abstr_kwrds_ten_track_ext.dict')
#    print(dictionary.token2id)
#    print(len(dictionary.token2id))
    
    # creates a 'bag of words', saves to disk
    corpus = [dictionary.doc2bow(text) for text in abstrs]
    gensim.corpora.MmCorpus.serialize('corpus/committee_abstr_kwrds_ten_track_ext.mm', corpus)

def clean_txt(txt, stem=False):
    '''
    Cleans string by removing special characters, numbers, and removing
    the possesive ' or 's from the end of words.  Also breaks hyphenated words 
    into two words. If the hyphenated word starts with "non-" it is kept 
    as a prefix.
    
    If stem=True, the nltk.stem.porter PorterStemmer function is used
    to reduce each word to its stem for more effective analysis
    
    Requires string.punctuation and nltk.stem.porter librarys.  
    '''
    ps = PorterStemmer()
    
    txt_lst = txt.split()
    txt_clean = []
    
    for word in txt_lst:
        if word not in string.punctuation:
            word_clean = ''
            aps = False  # if char is apostrophy
            for char in word:
                # if not a special character and no apostrophy has been found
                if char.isalpha() and not aps:
                    word_clean += char
                elif char == '-' and word_clean == 'non':
                    word_clean += char
                # keep if number followed by '...' to keep jel codes
                elif char.isnumeric() and word_clean!='' and any(word_clean[0] == x for x in 'GQJHDRELKNM'):
                    word_clean += char
                else:
                    if char == "'":
                        aps = True
                    elif char == '-':
                        word_clean += ' '
                    else:
                        word_clean += ' '
            if stem:
                txt_clean.append(ps.stem(word_clean))
            else:
                txt_clean.append(word_clean)
            
    txt = ' '.join(txt_clean).lower()
    
    return txt