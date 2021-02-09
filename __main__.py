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


import argparse
import os
import sys

from abstract_matching import main

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--no-convert', action='store_false', help='Don\'t convert .pdf files to .txt. ')
parser.add_argument('-D', '--cwd', help='Set main working directory. ')
parser.add_argument('-x', '--xpdf', help='Set path to xpdf. ')
parser.add_argument('-S', '--signature_path', help='Path to .txt email signature')
parser.add_argument('-s', '--signature', action='store_true', help='Include default email signature.')
args = parser.parse_args()

if args.cwd is not None:
    main_dir = args.cwd
else:
    main_dir = os.getcwd()
    
if args.xpdf is not None:
    xpdf_path = args.xpdf
else:
    if sys.platform.startswith('darwin'):
        xpdf_path = 'xpdf-tools-mac-4.02'
        
if args.signature_path is not None:
    sig_path = args.signature_path

if args.signature:
    sig_path = 'data/email_signature.txt'
else:
    sig_path = None

main(num_topics=200, convert=args.no_convert, main_dir=main_dir, sig_path=sig_path, xpdf_dir=xpdf_path)
