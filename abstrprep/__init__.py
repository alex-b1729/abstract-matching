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



from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import gensim
import shlex
import subprocess
import os
from sys import platform
import re
from six import iteritems
import pprint


def gen_dict_corpus(names, data_dir):
    """Returns (dictionary, corpus)"""
    # abstract file relative paths
    abstr_paths = []
    for row in names.itertuples():
        num_samps = int(row[3])
        for num in range(1, num_samps+1):
            ln = '_'.join(row[1].split())
            fn = '_'.join(row[2].split())
            abstr_paths.append('{}/{}_{}_{}.txt'.format(ln[0].upper(), ln, fn, num))

    # finds missing / mislabeled abstract samples
    # probs = []
    # for p in abstr_paths:
    #     file_path = os.path.join('data/learning_abstracts', p)
    #     if not os.path.exists(file_path):
    #         probs.append(p)
    # pprint.pprint(probs)

    # iterator loads all abstracts
    dictionary = gensim.corpora.Dictionary(CleanAbstracts(abstr_paths, data_dir))

    # remove words that appear only once accross all documents
    once_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs)
                if docfreq == 1]
    dictionary.filter_tokens(once_ids)
    # remove gaps in ids
    dictionary.compactify()

    # memory friendly corpus
    # corpus = IterCorpus(abstr_paths, dictionary)
    # not memory friendly corpus
    corpus = [abstr_samp for abstr_samp in IterCorpus(abstr_paths, dictionary, data_dir)]

    return dictionary, corpus


def extract_abstracts(convert, pdf_dir, xpdf_dir):
    """Converts pdfs to txt and saves abstracts then returns file names"""
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
    
    # if converting from pdf files
    if convert:
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
                    txt_file_names.append(entry.name[:-4])
    
        # convert each pdf to txt using xpdf tool
        txt_issues = []
        abstr_issues = []
        for paper_name in pdf_file_names:
            if paper_name not in txt_file_names:
                pdf_convert_path = os.path.join(pdf_dir_path, '{}.pdf'.format(paper_name))
                txt_result_path = os.path.join(abstr_txt_dir_path, '{}.txt'.format(paper_name))
                # convert first 10 pages
                comd = '{} -l 10 {} {}'.format(shlex.quote(os.path.join(xpdf_dir, 'bin64', 'pdftotext')), shlex.quote(pdf_convert_path), shlex.quote(txt_result_path))
                # get text from .pdf
                res = subprocess.run(comd, shell=True)
                # if issue with conversion
                if res.returncode!=0:
                    txt_issues.append(paper_name)
                else:
                    # get text of abstract
                    with open(txt_result_path, mode='r+', errors='ignore') as file:
                        abstr_txt = get_abstract(file.read())
                        # overwrite txt file with abstract txt if abstract found
                        if abstr_txt:
                            file.seek(0)
                            file.write(abstr_txt)
                            file.truncate()
                    if not abstr_txt:
                        abstr_issues.append(paper_name)
    
        # if issues with conversion
        if txt_issues != [] or abstr_issues != []:
            # txt conversion issues
            if txt_issues != []:
                print('Issue converting following .pdf file(s) to .txt:')
                for paper in txt_issues:
                    print('{}.pdf'.format(paper))
            # issues finding abstracts
            if abstr_issues != []:
                print('Issue extracting abstract from following .txt file(s):')
                for paper in abstr_issues:
                    print('{}.txt'.format(paper))
    
            # wait to allow user to manually save txt abstracts
            input('Please add missing abstracts to abstract_txts directory as a .txt file then press Enter, or Ctrl+C to quit. ')
    
            # double check that all abstracts are present
            all_converted = False
            while not all_converted:
                any_not_converted = False
                for paper in pdf_file_names:
                    if not os.path.isfile(os.path.join(abstr_txt_dir_path, '{}.txt'.format(paper))):
                        print('Abstract file {}.txt not found.'.format(paper))
                        any_not_converted = True
                if any_not_converted:
                    input('Please add missing abstracts then press Enter, or Ctrl+C to quit. ')
                else:
                    all_converted = True
                    
    else:
        # find abstracts already converted to txt
        pdf_file_names = []
        with os.scandir(abstr_txt_dir_path) as it:
            for entry in it:
                if entry.name.endswith('.txt') and entry.is_file() and not entry.name.startswith('.'):
                    pdf_file_names.append(entry.name[:-4])

    return pdf_file_names


class CleanAbstracts():
    def __init__(self, txt_paths, data_dir):
        self.txt_paths = txt_paths
        self.data_dir = data_dir

    def __iter__(self):
        for path in self.txt_paths:
            # ADD 'data'
            with open(os.path.join(self.data_dir, 'learning_abstracts', path), encoding='utf-8', errors='ignore') as file:
                abstr = file.read()
                yield meaningful_wrds(abstr)


class IterCorpus():
    """Memory friendly way to generate corpus"""
    def __init__(self, txt_paths, dictionary, data_dir):
        self.txt_paths = txt_paths
        self.dictionary = dictionary
        self.data_dir = data_dir

    def __iter__(self):
        for path in self.txt_paths:
            # ADD 'data'
            with open(os.path.join(self.data_dir, 'learning_abstracts', path), mode='r', encoding='utf-8', errors='ignore') as file:
                yield self.dictionary.doc2bow(meaningful_wrds(file.read()))


def meaningful_wrds(text):
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    return [ps.stem(wrd) for wrd in
            gensim.utils.simple_preprocess(text, deacc=True)
            if wrd not in stop_words]


def get_abstract(txt):
    '''
    Extracts abstract and paper title from .txt file and returns (title, abstract). 
    '''
    abstr = None
    txt = txt[:3000]
    paragraphs = [t.strip() for t in txt.split('\n')]
    
    if '--Manuscript Draft--' in paragraphs:
        md_loc = paragraphs.index('--Manuscript Draft--')
        title = paragraphs[md_loc-1]
        
        # abstract should be first "long" paragraph after 'manuscript draft'
        abstr_found = False
        num_to_check = md_loc + 1
        while not abstr_found and num_to_check<=len(paragraphs):
            # paragraph assumed to be abstract if longer than 300 characters
            if len(paragraphs[num_to_check]) > 300:
                abstr = paragraphs[num_to_check]
                abstr_found = True
            num_to_check += 1
    
    return title, abstr            

