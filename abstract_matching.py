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


import gensim
import os
import subprocess
import numpy as np
import hashlib
from scipy.optimize import linear_sum_assignment
import pandas as pd
import datetime as dt

# project packages
import abstrprep


def main(num_topics, convert, num_refs_to_assign=8, num_aes_to_assign=2, model='lsi', use_tfidf=True):
    # pd.set_option('display.max_rows', 1000)
    # pd.set_option('display.max_columns', 10)
    
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

    # generate dictionary matching faculty index to name
    fac_index = 0
    fac_dict = {}
    all_last = []
    all_first = []
    all_pos = []
    all_index = []
    all_samps = []
    for row in names.itertuples():
        fac_dict[fac_index] = '{} {}'.format(row[2], row[1])
        for num in range(1, int(row[3])+1):
            abstr_samp = '{}_{}_{}'.format(row[1], row[2], num)
            all_last.append(row[1])
            all_first.append(row[2])
            all_pos.append(row[4])
            all_index.append(fac_index)
            all_samps.append(abstr_samp)
        fac_index += 1
    sample_df = pd.DataFrame({'lastname':all_last, 'firstname':all_first, 
                                    'position':all_pos, 'fac_index':all_index, 
                                    'samp_name':all_samps})

    # load / generate dictionary and corpus
    try:
        # load if exists
        dictionary = gensim.corpora.Dictionary.load(os.path.join('data', 'dictionaries', '{}.dict'.format(names_hash)))
        corpus = gensim.corpora.MmCorpus(os.path.join('data', 'corpora', '{}.mm'.format(names_hash)))
    except FileNotFoundError:
        print('Building topic model...')
        # create dictionary & corpus
        dictionary, corpus = abstrprep.gen_dict_corpus(names)
        # save to disk
        dictionary.save(os.path.join('data', 'dictionaries', '{}.dict'.format(names_hash)))
        gensim.corpora.MmCorpus.serialize(os.path.join('data', 'corpora', '{}.mm'.format(names_hash)), corpus)
    
    # load / train model
    try:
        model = gensim.models.LsiModel.load(os.path.join('data', 'models', '{}.lsi'.format(names_hash)))
    except FileNotFoundError:
        if use_tfidf:
            tfidf = gensim.models.TfidfModel(corpus)
            corpus = tfidf[corpus]
        model = gensim.models.LsiModel(corpus, id2word=dictionary, num_topics=num_topics)
        model.save(os.path.join('data', 'models', '{}.lsi'.format(names_hash)))
        
    # transform corups to LSI space and index
    # non-memory friendly corpus solution
    index = gensim.similarities.MatrixSimilarity(model[corpus])
    
    # convert pdf submissions to txt abstracts
    submission_names = abstrprep.extract_abstracts(convert)

    # find 1 - cosine similarity for each submission
    similarity_vectors = []
    print('Generating abstract similarity:')
    for query in submission_names:
        print('\t{}'.format(query))
        with open(os.path.join('papers_to_assign', 'abstract_txts', '{}.txt'.format(query))) as file:
            vec_bow = dictionary.doc2bow(abstrprep.meaningful_wrds(file.read()))
        vec_model = model[vec_bow]
        sims = index[vec_model]
        # append to data from for records
        sample_df[query] = sims
        similarity_vectors.append(sims)
        
    # all matricies have researchers on rows and papers on columns
    cost_matrix = np.transpose(1 - np.array(similarity_vectors))
    
    # referee assignments
    print('Finding potential referees... ', end='')
    # choose ref with highest similarity (lowest cost)
    referee_cost_df = pd.DataFrame(cost_matrix[sample_df['position']=='referee'], 
                                   index=sample_df[sample_df['position']=='referee']['fac_index'])
    referee_cost_matrix = np.array(referee_cost_df.groupby('fac_index').min())
    
    referee_assignments = np.zeros(referee_cost_matrix.shape)
    # refs can only be assigned to 1 paper
    for i in range(num_refs_to_assign): 
        print('{}, '.format(i), end='')
        # Assignes each ref to 1 paper using the Hungarian algorithm
        referee_assign_array, paper_assign_array = linear_sum_assignment(referee_cost_matrix)
        for assign_index, paper_col in enumerate(paper_assign_array):
            referee_row = referee_assign_array[assign_index]
            # mark faculty, paper as assigned
            referee_assignments[referee_row, paper_col] = 1
            # mark as costly so no 2nd assignment to same ref
            referee_cost_matrix[referee_row,:] = np.ones((1, referee_cost_matrix.shape[1]))

    # assistant editor assignments
    print('\nFinding potential assistant editors... ', end='')
    ae_cost_df = pd.DataFrame(cost_matrix[sample_df['position']=='assistant_editor'], 
                                   index=sample_df[sample_df['position']=='assistant_editor']['fac_index'])
    ae_cost_matrix = np.array(ae_cost_df.groupby('fac_index').min())
    ae_assignments = np.zeros(ae_cost_matrix.shape)
    # aes can only be assigned 1 paper
    for i in range(num_aes_to_assign):
        print('{}, '.format(i), end='')
        ae_assign_array, paper_assign_array = linear_sum_assignment(ae_cost_matrix)
        for assign_index, paper_col in enumerate(paper_assign_array):
            if ae_assignments[:,paper_col].sum() < num_aes_to_assign:
                ae_row = ae_assign_array[assign_index]
                # mark faculty, paper as assigned
                ae_assignments[ae_row, paper_col] = 1
                # mark as costly so no 2nd assignment to same ref
                ae_cost_matrix[ae_row,:] = np.ones((1, ae_cost_matrix.shape[1]))
    
    # editor assignments
    print('\nAssigning editors... ')
    editor_abstr_cost_matrix = cost_matrix[sample_df['position']=='editor']
    editor_assignment_costs = np.ones(editor_abstr_cost_matrix.shape)
    # assign each paper to 5 (arbitrary number) editors 
    # assign based on sum of costs for each time an editor is assigned a paper
    for i in range(5):
        editor_assign_array, paper_assign_array = linear_sum_assignment(editor_abstr_cost_matrix)
        for assign_index, paper_col in enumerate(paper_assign_array):
            editor_row = editor_assign_array[assign_index]
            # mark faculty, paper as assigned using cost
            editor_assignment_costs[editor_row, paper_col] = editor_abstr_cost_matrix[editor_row, paper_col]
            # mark as costly so no 2nd assignment of same ref to same paper
            editor_abstr_cost_matrix[editor_row, paper_col] = 1
    # find sum of costs for each editor for each paper
    editor_assignment_cost_df = pd.DataFrame(editor_assignment_costs, 
                                             index=sample_df[sample_df['position']=='editor']['fac_index'])
    editor_cost_matrix = np.array(editor_assignment_cost_df.groupby('fac_index').sum()) / 5
    editor_assignments = np.zeros(editor_cost_matrix.shape)
    # assign each paper to 1 editor
    while not np.all(editor_cost_matrix == 1):
        editor_assign_array, paper_assign_array = linear_sum_assignment(editor_cost_matrix)
        for assign_index, paper_col in enumerate(paper_assign_array):
            # if paper not already assigned
            if editor_assignments[:,paper_col].sum() == 0:
                editor_row = editor_assign_array[assign_index]
                # mark assignment
                editor_assignments[editor_row, paper_col] = 1
                # mark column of assigned paper with 1's
                editor_cost_matrix[:,paper_col] = np.ones(editor_cost_matrix.shape[0])
    
    assignment_matrix = np.append(np.append(editor_assignments, ae_assignments, axis=0), referee_assignments, axis=0)
    names[submission_names] = assignment_matrix
    
    # output assignments
    print('Generating assignment message ')
    message = ''
    # find papers assigned to each editor
    for editor_index in range(names[names['position']=='editor'].shape[0]):
        editor_firstname = names['firstname'].iloc[editor_index]
        num_assignments = 0
        # check for paper assignments
        for submission in submission_names:
            # if paper assigned to editor
            if names[submission].iloc[editor_index] == 1:
                if num_assignments == 0:
                    message += f'Dear {editor_firstname}, \nThis week, you were matched to the following papers: \n\n'
                num_assignments += 1
                message += f'Manuscript: {submission}\n'
                # suggested assistant editors
                ae_suggestions = names[(names['position']=='assistant_editor') & (names[submission]==1)]
                ae_name_list = []
                for row in ae_suggestions.itertuples():
                    ae_name_list.append('{} {}'.format(row[2].capitalize(), row[1].capitalize()))
                message += 'Assistant Editor Ideas:\t{}\n'.format('\n\t\t\t'.join(ae_name_list))
                # suggested refs
                ref_suggestions = names[(names['position']=='referee') & (names[submission]==1)]
                ref_name_list = []
                for row in ref_suggestions.itertuples():
                    ref_name_list.append('{} {}'.format(row[2].capitalize(), row[1].capitalize()))
                message += 'Referee Ideas:\t{}\n\n'.format('\n\t\t'.join(ref_name_list))
        # space between editor messages
        if num_assignments != 0:
            message += '\n{}\n\n'.format('/'*80)
    
    # move txt abstracts to previous directory
    td = str(dt.date.today())
    dest_dir = os.path.join('previous_abstracts', td)
    os.mkdir(dest_dir)
    for sub in submission_names:
        current_dir = os.path.join('papers_to_assign', 'abstract_txts')
        os.rename(os.path.join(current_dir, '{}.txt'.format(sub)), os.path.join(dest_dir, '{}.txt'.format(sub)))
        
    # ask to delete pdf file
    # answer = input('Delete pdf submission documents from drive? [(y)/n] ')
    # if answer.lower().strip()=='y' or answer=='':
    #     for sub in submission_names:
    #         os.remove(os.path.join('papers_to_assign', '{}.pdf'.format(sub)))
    
    # save message to .txt
    dest_dir = os.path.join('assignment_results', td)
    os.mkdir(dest_dir)
    message_path = os.path.join(dest_dir, '{}_assignment_msg.txt'.format(td))
    with open(message_path, 'w') as file:
        file.write(message)
    # save similarity to .csv
    
    # open message
    subprocess.run(['open', message_path])


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

