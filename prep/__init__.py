#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 13:54:43 2020

@author: abrefeld
"""

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

def func():
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