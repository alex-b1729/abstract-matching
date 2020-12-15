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
import shlex
import subprocess
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
    # xpdf path depends on os
    if os_type=='darwin':
        # mac
        xpdf_path = os.path.join(os.getcwd(), 'xpdf-tools-mac-4.02/bin64/pdftotext')
        
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
                
    # convert each pdf to txt using xpdf tool
    txt_issues = []
    for paper_index in range(len(sub_names)):
        txt_path = os.path.join(abstr_txt_path, '{}.txt'.format(sub_names[paper_index]))
        paper_path = submissions[paper_index].path
        comd = '{} -l 10 {} {}'.format(shlex.quote(xpdf_path), shlex.quote(paper_path), shlex.quote(txt_path))
        # get text from .pdf
        res = subprocess.run(comd, shell=True)
        # if issue with conversion
        if res.returncode!=0:
            txt_issues.append(submissions[paper_index])
        else:
            # get text of abstract
            for paper in sub_names:
                if paper not in txt_issues:
                    with open(paper.path, mode='r', errors='ignore') as file:
                        paper_txt = file.read()
                        abstr_txt = get_abstract(paper_txt)
    
    # # print txt file conversion issues
    # print('Issues converting the following files to .txt:')
    # for title in txt_issues:
    #     print(title.name)
        
    # extract abstracts


    
class CleanAbstracts():
    def __init__(self, txt, txt_paths):
        self.txt_paths = txt_paths
        self.txt = txt
    
    def __iter__(self):
        for path in self.txt_paths:
            with open(os.path.join('data/learning_abstracts', path), encoding='utf-8', errors='ignore') as file:
                yield self.meaningful_wrds(file.read())
    
    def meaningful_wrds(text):
        ps = PorterStemmer()
        stop_words = set(stopwords.words('english'))
        [ps.stem(wrd) for wrd in 
         gensim.utils.simple_preprocess(text, deacc=True)
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

class FindAbstract():
    def __init__(self, txt_paths):
        self.txt_paths = txt_paths
    
    def __iter__(self):


def get_abstract(txt):
    '''
    Attempts to extract abstract from text of paper by searching for the
    word 'abstract' then taking everything untill the next newline char.  
    
    Sometimes keywords are on the same line as the abstract so if 'keywords:'
    appears before the first newline char the abstract is assumed to end there.  
    
    In some cases the abstract covers multiple lines.  If the function finds
    that the first new line char appears within the first 120 characters
    then it searches for the first appearance of 'keywords:', 'we thank',
    'the authors thank', or 'jel classification' and assumes abstract texts 
    ends there.  
    
    Also attempts to find and include keywords and jel classifications.  
    
    Parameters: txt (str): paper text
    
    Returns: str or None if no abstract found
    
    Honastly this function is a mess but so far words as needed...
    
    Still needs work: isn't accurite as needed when abstract doesn't appear 
    but finds jel classifications
    '''
    
    abstr = None
    
    # reg expression to find jel classifications
    jels = re.search('JEL.*:\s([A-Z]{1}[0-9]{1,2})[,;A-Z 0-9]+', txt[:3000])
    if jels:
        all_jel = []
        if ',' in jels[0]:
            jels = jels[0].split(',')
            for jel in jels:
                if ':' in jel:
                    tem = jel.split(':')
                    jel = tem[1].strip()
                else:
                    jel = jel.strip()
                if not jel[-1].isnumeric():
                    jel = jel[:-1]
                all_jel.append(jel.strip())
            jels = ' '.join(all_jel)
        else:
            jels = jels[0].split(';')
            for jel in jels:
                if ':' in jel:
                    tem = jel.split(':')
                    jel = tem[1].strip()
                else:
                    jel = jel.strip()
                if not jel[-1].isnumeric():
                    jel = jel[:-1]
                all_jel.append(jel.strip())
            jels = ' '.join(all_jel)
        
    txt = txt.lower()
    
    loc = txt.find('abstract')
    # if found in txt
    if loc != -1:
        # text after appearance of 'abstract'
        txt = txt[loc+8:].strip()
        if any(x in txt for x in ['keywords:', 'key words:']):
            kwrd_1 = txt.find('keywords:')
            kwrd_2 = txt.find('key words:')
            kwrd = max(kwrd_1, kwrd_2)
            end = txt.find('\n', kwrd+5)
            txt = txt[:end]
            if 'jel' in txt:
                end = txt.find('jel')
                if end < kwrd and (kwrd-end) < 50:
                    first = txt[:end]
                    last = txt[kwrd:]
                    txt = ' '.join([first, last])
                else:
                    txt = txt[:end]
            kwrd_1 = txt.find('keywords:')
            kwrd_2 = txt.find('key words:')
            kwrd = max(kwrd_1, kwrd_2)
            txt = ' '.join([txt[:kwrd], txt[kwrd+9:]])
        
        else:
            end = txt.find('\n', 10)
            if end < 120:
                if any(x in txt.lower() for x in ['keywords:', 'jel classification', 'we thank', 'the authors thank', 'the authors are']):
                    kwrd = txt.find('keywords:')
                    if kwrd == -1:
                        kwrd = 10000
                    jelly = txt.find('jel classification')
                    if jelly == -1:
                        jelly = 10000
                    thx = txt.find('we thank')
                    if thx == -1:
                        thx = 10000
                    auth_thx = txt.find('the authors thank')
                    if auth_thx == -1:
                        auth_thx = 10000
                    auth_are = txt.find('the authors are')
                    if auth_are == -1:
                        auth_are = 10000
                    end = min(kwrd, jelly, thx, auth_thx, auth_are)
                    txt = txt[:end].strip()
                else:
                    txt = None
            else:
                txt = txt[:end].strip()
            if txt:
                if 'keywords:' in txt:
                    end = txt.find('keywords:')
                    txt = txt[:end]
        abstr = txt
        
    if jels:
        if abstr:
            abstr = '{} {}'.format(abstr,jels)
        else:
            abstr = jels
            
    return abstr

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



