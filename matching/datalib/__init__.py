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

import hashlib
import datetime
import padas as pd
import os

def file_hash(file_path):
    """Outputs hash of file as str"""
    block_size = 32768
    hash_func = hashlib.sha256()
    with open(file_path, 'rb') as f:
        hash_func.update(f.read())
    return hash_func.hexdigest()

def gen_paths(name_file_path, data_file_path):
    """Returns absolute file path of abstract samples for names in file"""
    # NEEDS TO BE TESTED
    file_paths = []
    name_tbl = pd.read_csv(name_file_path, header=0)
    for row in name_tbl.iterrows():
        for num in range(1, row.num_abstracts+1):
            abs_dir_path = os.path.join(data_file_path, 'learning_abstracts/{1}'.format(name_tbl.lastname[0].upper()))
            file_paths.append(os.path.join(abs_dir_path, '{1}_{2}_{3}.txt'.format(name_tbl.lastname, name_tbl.firstname, name_tbl.num_abstracts)))
    return file_paths

