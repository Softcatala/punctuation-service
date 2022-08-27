#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
#
# Copyright (c) 2022 Jordi Mas i Hernandez <jmas@softcatala.org>
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this program; if not, write to the
# Free Software Foundation, Inc., 59 Temple Place - Suite 330,
# Boston, MA 02111-1307, USA.

import datetime
import sys
import os

# Source corpus contains parts expected with comma and no comma
# We split them calculate precision and recall
def split_source_corpus(plain, plain_comma, plain_nocomma):

    comma = True
    with open(plain, 'r') as fp_plain, open(plain_comma, 'w') as fp_comma, open(plain_nocomma, 'w') as fp_nocomma:
        while True:

            line = fp_plain.readline()
            if not line:
                break
                
            line = line.strip()
            if 'Tipus: Sense comas' in line:
                comma = False

            if len(line) >0 and line[0] == "#":
                 continue

            if comma:
                fp = fp_comma
            else:
                fp = fp_nocomma

            fp.write(line + "\n")

def create_corpus_without_commas(plain, plain_nocomma):
    comma = True
    with open(plain, 'r') as fp_plain, open(plain_nocomma, 'w') as fp_nocomma:
        while True:

            line = fp_plain.readline()
            if not line:
                break

            line = line.strip()
            if len(line) == 0:
                continue
                
            line = line.replace(",", "")
            fp_nocomma.write(line + "\n")

            
def diff(reference, hypotesis):
    with open(reference, 'r') as fp_ref, open(hypotesis, 'r') as fp_hypo, open(hypotesis + ".diff", 'w') as fp_diff:
        ref_lines = fp_ref.readlines()
        hypo_lines = fp_hypo.readlines()
        
        if len(ref_lines) != len(hypo_lines):
            print(f"Different length - {reference} - {hypotesis}")
            exit(1)

        equal = 0
        diff = 0       
        for idx in range(0, len(ref_lines)):
            ref = ref_lines[idx].strip()
            hyp = hypo_lines[idx].strip()
            if ref == hyp:
                equal += 1
            else:
                diff += 1
                fp_diff.write("----" + "\n")
                fp_diff.write(f"{ref}" + "\n")
                fp_diff.write(f"{hyp}" + "\n")

        pequal = equal * 100 / (diff+equal)
        pdiff = diff * 100 / (diff+equal) 
        print(f"Equal {equal} ({pequal:.2f}%), diff {diff} ({pdiff:.2f}%)")
    
if __name__ == "__main__":
    print("Evaluate using the evaluation corpus")
    
    plain = "plain.ref"
    plain1 = "plain1.ref"
    plain2 = "plain2.ref"
    plain1_source = "plain1.source"
    plain2_source = "plain2.source"    
    split_source_corpus(plain, plain1, plain2)
    
    create_corpus_without_commas(plain1, plain1_source)
    create_corpus_without_commas(plain2, plain2_source)
    
    os.system(f"python3 inference.py {plain1_source}")
    os.system(f"python3 inference.py {plain2_source}")

    diff(plain1, f"{plain1_source}-comas.txt")
    diff(plain2, f"{plain2_source}-comas.txt")
