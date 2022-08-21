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
from punctuationmodel import PunctuationModel

# Remove warning 'Some weights of RobertaModel were not initialized from the model checkpoint'
import transformers
transformers.logging.set_verbosity_error()

model = PunctuationModel()

def do_inference(ref_lines, lines, output, file_type, punctuation):

    model.punctuation = punctuation
    with open(output, 'w') as ot:

        cnt = 0
        equal = 0
        diff = 0
        for idx in range(0, len(lines)):
            line = lines[idx].strip()
            ref_line = ref_lines[idx].strip()
        
            line = line.strip()
            if len(line) == 0:
                continue

#            print(f"source: '{line}'")
            line = model.restore_punctuation(line)

            ot.write(line + "\n")
            cnt += 1

#            if cnt % 100 == 0:
#                print(cnt)

#            print(f"{line == ref_line} - '{ref_line}' - '{line}'")
            if line == ref_line:
                equal += 1
            else:
                diff += 1

    pequal = equal * 100 / (diff+equal)
    print(f"Sentences: for {file_type} equal {equal}  ({pequal:.2f}%), diff {diff}")


if __name__ == "__main__":
    print("Inference using the same API than the punctuation web service"
   
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
       input_file =  'flores101_cat.txt'

    start_time = datetime.datetime.now()

    base_filename = input_file.replace(".txt", "")
    with open(input_file, 'r') as fp:
        lines = fp.readlines()
    
    output_file = f'{base_filename}-comas.txt'
    do_inference(lines, lines, output_file, "no commas", ",")
    s = 'Time used: {0}'.format(datetime.datetime.now() - start_time)
    print(f"Output file: {output_file}")
#    print(f"Model used: {model.model}, input file {input_file}")
    print(s)
