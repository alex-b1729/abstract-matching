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

def main(data_dir_path):
    name_path = os.path.join(data_dir_path, 'assignment_groups')

    # open name lists
    editor_names = pd.read_csv(os.path.join(name_path, 'editor_names.csv'))
    assistant_editor_names = pd.read_csv(os.path.join(name_path, 'assistant_editor_names.csv'))
    referee_names = pd.read_csv(os.path.join(name_path, 'referee_names.csv'))
    
    # find hash of name lists
    e_names = file_hash(os.path.join(name_path, 'editor_names.csv'))
    ae_hash = file_hash(os.path.join(name_path, 'assistant_editor_names.csv'))
    r_hash = file_hash(os.path.join(name_path, 'referee_names.csv'))
    
    try:
        dictionary = gensim.corpora.Dictionary.load('{}.dict'.format(e_hash))
        corpus = gensim.corpora.mmcorpus.MmCorpus('{}.mm'.format(e_hash))
    except FileNotFoundError:
        dictionary = DEFINE_FUNCTION
        corups = DEFINE_FUNCTION

def file_hash(file_path):
    """Outputs hash of file as str"""
    block_size = 32768
    hash_func = hashlib.sha256()
    with open(file_path, 'rb') as f:
        hash_func.update(f.read())
    return hash_func.hexdigest()