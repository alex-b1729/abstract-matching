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
import subprocess
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
import pprint

# project packages
import abstrprep


def main(num_topics, num_refs_to_assign=2, num_aes_to_assign=2, model='lsi', use_tfidf=True):
    pd.set_option('display.max_rows', 1000)
    
    os.chdir('/Users/abrefeld/Dropbox/UK/JCF_assignment')
    print(os.getcwd())
    
    name_path = 'assignment_groups'

    # open name lists
    editor_names = pd.read_csv(os.path.join(name_path, 'editor_names_testing.csv'))
    editor_names['position'] = 'editor'
    assistant_editor_names = pd.read_csv(os.path.join(name_path, 'assistant_editor_names_testing.csv'))
    assistant_editor_names['position'] = 'assistant_editor'
    referee_names = pd.read_csv(os.path.join(name_path, 'referee_names_testing.csv'))
    referee_names['position'] = 'referee'
    # full list of names
    names = editor_names.append([assistant_editor_names, referee_names], ignore_index=True)
    # find hash of name lists
    names_hash = file_hash(os.path.join(name_path, 'editor_names_testing.csv'), os.path.join(name_path, 'assistant_editor_names_testing.csv'), os.path.join(name_path, 'referee_names_testing.csv'))

    # generate dictionary matching corpus index to abstract sample
    ind = 0
    fac_index = 0
    fac_dict = {}
    all_last = []
    all_first = []
    all_pos = []
    all_index = []
    all_samps = []
    for row in names.itertuples():
        for num in range(1, int(row[3])+1):
            abstr_samp = '{}_{}_{}'.format(row[1], row[2], num)
            fac_dict[ind] = abstr_samp
            ind += 1
            all_last.append(row[1])
            all_first.append(row[2])
            all_pos.append(row[4])
            all_index.append(fac_index)
            all_samps.append(abstr_samp)
        fac_index += 1
    sample_df = pd.DataFrame({'lastname':all_last, 'firstname':all_first, 
                                    'position':all_pos, 'fac_index':all_index, 
                                    'samp_name':all_samps})
    # print('sample df', sample_df, sep='\n')

    # load / generate dictionary and corpus
    try:
        # load if exists
        dictionary = gensim.corpora.Dictionary.load(os.path.join('data', 'dictionaries', '{}.dict'.format(names_hash)))
        corpus = gensim.corpora.MmCorpus(os.path.join('data', 'corpora', '{}.mm'.format(names_hash)))
    except FileNotFoundError:
        print('gen dict corpus')
        # create dictionary & corpus
        dictionary, corpus = abstrprep.gen_dict_corpus(names)
        # save to disk
        dictionary.save(os.path.join('data', 'dictionaries', '{}.dict'.format(names_hash)))
        gensim.corpora.MmCorpus.serialize(os.path.join('data', 'corpora', '{}.mm'.format(names_hash)), corpus)
    
    # # print dictionary to file for reference
    # with open('/Users/abrefeld/Desktop/token2id.txt', mode='w') as file:
    #     print(dictionary.token2id, file=file)
    # with open('/Users/abrefeld/Desktop/corp.txt', mode='w') as file:
    #     for vec in corpus:
    #         print(vec, file=file)
        
    # load / train model
    try:
        model = gensim.models.LsiModel.load(os.path.join('data', 'models', '{}.lsi'.format(names_hash)))
    except FileNotFoundError:
        print('gen model')
        if use_tfidf:
            tfidf = gensim.models.TfidfModel(corpus)
            corpus = tfidf[corpus]
        model = gensim.models.LsiModel(corpus, id2word=dictionary, num_topics=num_topics)
        model.save(os.path.join('data', 'models', '{}.lsi'.format(names_hash)))
        
    # transform corups to LSI space and index
    # index = gensim.similarities.Similarity(output_prefix=None, corpus=corpus, num_features=num_topics)
    # memory friendly corpus
    # non-memory friendly corpus solution
    index = gensim.similarities.MatrixSimilarity(model[corpus])
    
    # convert pdf submissions to txt abstracts
    submission_names = abstrprep.extract_abstracts()
    
    # find 1 - cosine similarity for each submission
    similarity_vectors = []
    for query in submission_names:
        with open(os.path.join('papers_to_assign', 'abstract_txts', '{}.txt'.format(query))) as file:
            vec_bow = dictionary.doc2bow(abstrprep.meaningful_wrds(file.read()))
        vec_model = model[vec_bow]
        sims = index[vec_model]
        # append to data from for records
        sample_df[query] = sims
        similarity_vectors.append(sims)
        
        # # un-commenting the following lines will print the list of faculty and 
        # # cosine simlarity pairs in decending order for each query
        # sims_sort = sorted(enumerate(sims), key=lambda item: -item[1])
        # for i, s in enumerate(sims_sort):
        #     print(s, fac_dict[s[0]])
        # print('\n\n')
        
    # all matricies have researchers on rows and papers on columns
    # print('sample_df', sample_df, sep='\n')
    # print('similarity_vectors', similarity_vectors, sep='\n')
    cost_matrix = np.transpose(1 - np.array(similarity_vectors))
    # print('cost_matrix', cost_matrix, sep='\n')
    # print(cost_matrix.shape)
    
    # referee assignments
    # choose ref with highest similarity (lowest cost)
    referee_cost_df = pd.DataFrame(cost_matrix[sample_df['position']=='referee'], 
                                   index=sample_df[sample_df['position']=='referee']['fac_index'])
    # print('ref cost df', referee_cost_df, sep='\n')
    referee_cost_matrix = np.array(referee_cost_df.groupby('fac_index').min())
    
    referee_assignments = np.zeros(referee_cost_matrix.shape)
    # refs can only be assigned to 1 paper
    for i in range(num_refs_to_assign): 
        # print('ref cost matrix', referee_cost_matrix, sep='\n')
        # Assignes each ref to 1 paper using the Hungarian algorithm
        referee_assign_array, paper_assign_array = linear_sum_assignment(referee_cost_matrix)
        # print('ref', referee_assign_array)
        # print('paper', paper_assign_array)
        for assign_index, paper_col in enumerate(paper_assign_array):
            referee_row = referee_assign_array[assign_index]
            # mark faculty, paper as assigned
            referee_assignments[referee_row, paper_col] = 1
            # mark as costly so no 2nd assignment to same ref
            referee_cost_matrix[referee_row,:] = np.ones((1, referee_cost_matrix.shape[1]))
    # print('ref assignment', referee_assignments, sep='\n')
    # assistant editor assignments
    ae_cost_df = pd.DataFrame(cost_matrix[sample_df['position']=='assistant_editor'], 
                                   index=sample_df[sample_df['position']=='assistant_editor']['fac_index'])
    # print('ae cost df', ae_cost_df, sep='\n')
    ae_cost_matrix = np.array(ae_cost_df.groupby('fac_index').min())
    # print('ae cost matrix', ae_cost_matrix, sep='\n')
    ae_assignments = np.zeros(ae_cost_matrix.shape)
    # aes can only be assigned 1 paper
    for i in range(num_aes_to_assign):
        ae_assign_array, paper_assign_array = linear_sum_assignment(ae_cost_matrix)
        for assign_index, paper_col in enumerate(paper_assign_array):
            if ae_assignments[:,paper_col].sum() == 0:
                ae_row = ae_assign_array[assign_index]
                # mark faculty, paper as assigned
                ae_assignments[ae_row, paper_col] = 1
                # mark as costly so no 2nd assignment to same ref
                ae_cost_matrix[ae_row,:] = np.ones((1, ae_cost_matrix.shape[1]))
    # print('ae assignment', ae_assignments, sep='\n')
    # editor assignments
    editor_abstr_cost_matrix = cost_matrix[sample_df['position']=='editor']
    # print('e abstr cost matrix', editor_abstr_cost_matrix, sep='\n')
    editor_assignment_costs = np.ones(editor_abstr_cost_matrix.shape)
    # assign each paper to 5 (arbitrary number) editors 
    # assign based on sum of costs for each time an editor is assigned a paper
    for i in range(5):
        editor_assign_array, paper_assign_array = linear_sum_assignment(editor_abstr_cost_matrix)
        # print('editor', editor_assign_array)
        # print('paper', paper_assign_array)
        for assign_index, paper_col in enumerate(paper_assign_array):
            editor_row = editor_assign_array[assign_index]
            # mark faculty, paper as assigned using cost
            editor_assignment_costs[editor_row, paper_col] = editor_abstr_cost_matrix[editor_row, paper_col]
            # mark as costly so no 2nd assignment of same ref to same paper
            editor_abstr_cost_matrix[editor_row, paper_col] = 1
    # find sum of costs for each editor for each paper
    editor_assignment_cost_df = pd.DataFrame(editor_assignment_costs, 
                                             index=sample_df[sample_df['position']=='editor']['fac_index'])
    # print('e assignment cost df', editor_assignment_cost_df, sep='\n')
    editor_cost_matrix = np.array(editor_assignment_cost_df.groupby('fac_index').sum())
    # print('e cost matrix', editor_cost_matrix, sep='\n')
    editor_assignments = np.zeros(editor_cost_matrix.shape)
    # assign each paper to 1 editor
    while not np.all(editor_cost_matrix == 1):
        editor_assign_array, paper_assign_array = linear_sum_assignment(editor_cost_matrix)
        # print('editor', editor_assign_array)
        # print('paper', paper_assign_array)
        for assign_index, paper_col in enumerate(paper_assign_array):
            # if paper not already assigned
            if editor_assignments[:,paper_col].sum() == 0:
                editor_row = editor_assign_array[assign_index]
                # mark assignment
                editor_assignments[editor_row, paper_col] = 1
                # mark column of assigned paper with 1's
                editor_cost_matrix[:,paper_col] = np.ones(editor_cost_matrix.shape[0])
    # print('e assignment', editor_assignments, sep='\n')
    
    
    
    
    
    
    # subprocess.run(['open', '/Users/abrefeld/Dropbox/UK/JCF_assignment/assignment_results/12-17-2020.txt'])


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
    main(num_topics=10)
