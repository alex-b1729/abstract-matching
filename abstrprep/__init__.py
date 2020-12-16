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
    

def extract_abstract(os_type, pdf_dir='papers_to_assign'):
    """Converts pdfs to txt and saves abstracts"""
    # xpdf path depends on os
    if os_type=='darwin':
        # mac
        xpdf_path = os.path.join(os.getcwd(), 'xpdf-tools-mac-4.02/bin64/pdftotext')
        
    # pdf directory full path
    pdf_dir_path = os.path.join(os.getcwd(), pdf_dir)
    # txt abstract full path
    abstr_txt_dir_path = os.path.join(pdf_dir_path, 'abstract_txts')
    
    # check if pdf path exists
    if os.path.isdir(pdf_dir_path):
        if not os.path.isdir(abstr_txt_dir_path):
            # make abstract txt directory in pdf directory
            os.mkdir(abstr_txt_dir_path)
    else:
        # somethings wrong with the paths
        print('Directory not found: {}'.format(pdf_dir_path))

    # pdfs names
    pdf_file_names = []
    with os.scandir(pdf_dir_path) as it:
        for entry in it:
            if entry.name.endswith('.pdf') and entry.is_file() and not entry.name.startswith('.'):
                # save file name without extension
                pdf_file_names.append(entry.name[:-4])
                
    # find abstracts already converted to txt
    txt_file_names = []
    with os.scandir(abstr_txt_dir_path) as it:
        for entry in it:
            if entry.name.endswith('.txt') and entry.is_file() and not entry.name.startswith('.'):
                if entry.name[:-4] not in pdf_file_names:
                    txt_file_names.append(entry.name)
                
    # convert each pdf to txt using xpdf tool
    txt_issues = []
    abstr_issues = []
    for paper_name in pdf_file_names:
        if paper_name not in txt_file_names:
            pdf_convert_path = os.path.join(pdf_dir_path, '{}.pdf'.format(paper_name))
            txt_result_path = os.path.join(abstr_txt_dir_path, '{}.txt'.format(paper_name))
            # convert first 10 pages
            comd = '{} -l 10 {} {}'.format(shlex.quote(xpdf_path), shlex.quote(pdf_convert_path), shlex.quote(txt_result_path))
            # get text from .pdf
            res = subprocess.run(comd, shell=True)
            # if issue with conversion
            if res.returncode!=0:
                txt_issues.append(paper_name)
            else:
                # get text of abstract
                with open(txt_result_path, mode='r', errors='ignore') as file:
                    abstr_txt = get_abstract(file.read())
                    # overwrite txt file with abstract txt if abstract found
                    if abstr_txt:
                        file.seek(0)
                        file.write(abstr_txt)
                        file.truncate()
                    else:
                        abstr_issues.append(paper_name)
    
    # if issues with conversion
    if txt_issues != [] or abstr_issues != []:
        # txt conversion issues
        print('Issue converting following .pdf file to .txt:')
        for paper in txt_issues:
            print('{}.pdf'.format(paper))
        # issues finding abstracts
        print('Issues extracting abstract from following .txt files:')
        for paper in abstr_issues:
            print('{}.txt'.format(paper))
        
        # wait to allow user to manually save txt abstracts
        input('Please add missing abstracts to abstract_txts directory as a .txt file then press Enter...')
            
        # double check that all abstracts are present
        all_converted = False
        while not all_converted:
            any_not_converted = False
            with os.scandir(abstr_txt_dir_path) as it:
                for entry in it:
                    if entry.name.endswith('.txt') and entry.is_file() and not entry.name.startswith('.'):
                        if entry.name[:-4] not in pdf_file_names:
                            print('Abstract .txt file for {} not found.'.format(entry.name[:-4]))
                            any_not_converted = True
            if any_not_converted:
                input('Not all .txt files found. \nPlease add missing abstracts then press Enter...')
            else:
                all_converted = True
        


    
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
    
    # issue if abstract is too long or short
    abstr_wrds = len(abstr.split())
    if abstr_wrds < 50 or abstr_wrds > 200:
        abstr = None

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



