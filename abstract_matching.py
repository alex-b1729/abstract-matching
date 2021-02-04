#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# Copyright (C) 2021 Alexander Brefeld <alexander.brefeld@protonmail.com>

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


def main(num_topics, convert, main_dir, sig_path, name_dir='assignment_groups', pdf_dir='papers_to_assign', data_dir='data', xpdf_dir='xpdf-tools-mac-4.02', num_refs_to_assign=8, num_aes_to_assign=2, model='lsi', use_tfidf=True):
    # pd.set_option('display.max_rows', 1000)
    # pd.set_option('display.max_columns', 10)

    os.chdir(main_dir)

    # open name lists
    editor_names = pd.read_csv(os.path.join(name_dir, 'editor_names.csv'))
    editor_names['position'] = 'editor'
    assistant_editor_names = pd.read_csv(os.path.join(name_dir, 'assistant_editor_names.csv'))
    assistant_editor_names['position'] = 'assistant_editor'
    referee_names = pd.read_csv(os.path.join(name_dir, 'referee_names.csv'))
    referee_names['position'] = 'referee'
    # full list of names
    names = editor_names.append([assistant_editor_names, referee_names], ignore_index=True)
    # find hash of name lists
    names_hash = file_hash(os.path.join(name_dir, 'editor_names.csv'), os.path.join(name_dir, 'assistant_editor_names.csv'), os.path.join(name_dir, 'referee_names.csv'))

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
        dictionary = gensim.corpora.Dictionary.load(os.path.join(data_dir, 'dictionaries', '{}.dict'.format(names_hash)))
        corpus = gensim.corpora.MmCorpus(os.path.join(data_dir, 'corpora', '{}.mm'.format(names_hash)))
    except FileNotFoundError:
        print('Building topic model...')
        # create dictionary & corpus
        dictionary, corpus = abstrprep.gen_dict_corpus(names, data_dir)
        # save to disk
        if not os.path.isdir(os.path.join(data_dir, 'dictionaries')):
            os.mkdir(os.path.join(data_dir, 'dictionaries'))
        dictionary.save(os.path.join(data_dir, 'dictionaries', '{}.dict'.format(names_hash)))
        if not os.path.isdir(os.path.join(data_dir, 'corpora')):
            os.mkdir(os.path.join(data_dir, 'corpora'))
        gensim.corpora.MmCorpus.serialize(os.path.join(data_dir, 'corpora', '{}.mm'.format(names_hash)), corpus)
    
    # load / train model
    try:
        model = gensim.models.LsiModel.load(os.path.join(data_dir, 'models', '{}.lsi'.format(names_hash)))
    except FileNotFoundError:
        if use_tfidf:
            tfidf = gensim.models.TfidfModel(corpus)
            corpus = tfidf[corpus]
        model = gensim.models.LsiModel(corpus, id2word=dictionary, num_topics=num_topics)
        if not os.path.isdir(os.path.join(data_dir, 'models')):
            os.mkdir(os.path.join(data_dir, 'models'))
        model.save(os.path.join(data_dir, 'models', '{}.lsi'.format(names_hash)))
        
    # transform corups to LSI space and index
    # non-memory friendly corpus solution
    index = gensim.similarities.MatrixSimilarity(model[corpus])
    
    # convert pdf submissions to txt abstracts
    submission_names_dict = abstrprep.extract_abstracts(convert, pdf_dir, xpdf_dir)
    submission_names = list(submission_names_dict.keys())

    # find 1 - cosine similarity for each submission
    similarity_vectors = []
    print('Generating abstract similarity:')
    for query in submission_names:
        print('\t{}'.format(query))
        with open(os.path.join(pdf_dir, 'abstract_txts', '{}.txt'.format(query))) as file:
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
    
    # ref costs to save
    ref_cost_output = referee_names.copy()
    ref_cost_output[submission_names] = referee_cost_matrix
    
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
    
    # ae costs to save
    ae_cost_output = assistant_editor_names.copy()
    ae_cost_output[submission_names] = ae_cost_matrix
    
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
    editor_cost_matrix = np.array(editor_assignment_cost_df.groupby('fac_index').sum())
    editor_cost_matrix = editor_cost_matrix / editor_cost_matrix.max()
    editor_assignments = np.zeros(editor_cost_matrix.shape)
    
    # editor cost df to save
    editor_cost_output = editor_names.copy()
    editor_cost_output[submission_names] = editor_cost_matrix
    
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
    assignment_df = names.copy()
    assignment_df[submission_names] = assignment_matrix
    # number of assignments to each name
    assignment_df['num_assigned'] = assignment_df.sum(axis=1)
    
    # output assignments
    print('Generating assignment message ')
    
    # open email signature if any
    sig = ''
    if sig_path is not None:
        if os.path.isfile(sig_path):
            with open(sig_path, 'r') as f:
                sig = f.read()
        else:
            print(f'{sig_path} is not a valid system path.  \nProceeding without email signature.  ')
    
    message = ''
    # find papers assigned to each editor
    for editor_index in range(assignment_df[assignment_df['position']=='editor'].shape[0]):
        editor_firstname = assignment_df['firstname'].iloc[editor_index]
        num_assignments = assignment_df['num_assigned'].iloc[editor_index]
        # set singular / plural for message
        sing_plur = ''
        if num_assignments>1:
            sing_plur = 's'
        # check for paper assignments
        for submission in submission_names:
            # if paper assigned to editor
            if assignment_df[submission].iloc[editor_index] == 1:
                if num_assignments == 0:
                    message += f'Dear {editor_firstname}, \nThis week you were matched to the following paper{sing_plur}: \n'
                num_assignments += 1
                message += '\nManuscript: {}\n'.format(submission_names_dict[submission])
                # suggested assistant editors
                ae_suggestions = assignment_df[(assignment_df['position']=='assistant_editor') & (assignment_df[submission]==1)]
                ae_name_list = []
                for row in ae_suggestions.itertuples():
                    ae_name_list.append('{} {}'.format(row[2].capitalize(), row[1].capitalize()))
                message += 'Assistant Editor Ideas:\n\t\t{}\n'.format('\n\t\t'.join(ae_name_list))
                # suggested refs
                ref_suggestions = assignment_df[(assignment_df['position']=='referee') & (assignment_df[submission]==1)]
                ref_name_list = []
                for row in ref_suggestions.itertuples():
                    ref_name_list.append('{} {}'.format(row[2].capitalize(), row[1].capitalize()))
                message += 'Referee Ideas:\n\t\t{}\n'.format('\n\t\t'.join(ref_name_list))
        # space between editor messages
        if num_assignments != 0:
            message += '{}\n{}\n\n'.format(sig, '/'*80)
    
    # move txt abstracts to previous directory
    td = str(dt.date.today())
    dest_dir = os.path.join('previous_abstracts', td)
    if not os.path.isdir(dest_dir):
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
    if not os.path.isdir(dest_dir):
        os.mkdir(dest_dir)
    message_path = os.path.join(dest_dir, 'assignment_msg_{}.txt'.format(td))
    # don't overwrite previous file if run twice in same day
    written = False
    tries = 1
    while not written and tries<6:
        if not os.path.isfile(message_path):
            with open(message_path, 'w') as file:
                file.write(message)
                written = True
        else:
            message_path = os.path.join(dest_dir, 'assignment_msg_{}_{}.txt'.format(td, tries))
            tries += 1
    
    # save similarity to .csv
    all_samp_costs = sample_df.copy()
    all_samp_costs[submission_names] = cost_matrix
    written = False
    tries = 1
    assignment_df_path = os.path.join(dest_dir, 'assignments_{}.csv'.format(td))
    all_samp_path = os.path.join(dest_dir, 'all_abstr_costs_{}.csv'.format(td))
    ref_cost_path = os.path.join(dest_dir, 'ref_costs_{}.csv'.format(td))
    ae_cost_path = os.path.join(dest_dir, 'ae_costs_{}.csv'.format(td))
    e_cost_path = os.path.join(dest_dir, 'editor_costs_{}.csv'.format(td))
    sub_names_path = os.path.join(dest_dir, 'submission_dictionary_{}.txt'.format(td))
    while not written and tries<6:
        if not os.path.isfile(all_samp_path):
            assignment_df.to_csv(assignment_df_path, mode='x')
            all_samp_costs.to_csv(all_samp_path, mode='x')
            ref_cost_output.to_csv(ref_cost_path, mode='x')
            ae_cost_output.to_csv(ae_cost_path, mode='x')
            editor_cost_output.to_csv(e_cost_path, mode='x')
            with open(sub_names_path, 'w') as f:
                print(submission_names_dict, file=f)
            written = True
        else:
            assignment_df_path = os.path.join(dest_dir, 'assignments_{}_{}.csv'.format(td, tries))
            all_samp_path = os.path.join(dest_dir, 'all_abstr_costs_{}_{}.csv'.format(td, tries))
            ref_cost_path = os.path.join(dest_dir, 'ref_costs_{}_{}.csv'.format(td, tries))
            ae_cost_path = os.path.join(dest_dir, 'ae_costs_{}_{}.csv'.format(td, tries))
            e_cost_path = os.path.join(dest_dir, 'editor_costs_{}_{}.csv'.format(td, tries))
            sub_names_path = os.path.join(dest_dir, 'submission_dictionary_{}_{}.txt'.format(td, tries))
            tries += 1
        
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












