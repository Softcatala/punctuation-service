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

from __future__ import print_function
from flask import Flask, request, Response
from flask_cors import CORS
from punctuationmodel import PunctuationModel
import json
import os
import time
import logging
import logging.handlers
import datetime
import torch

app = Flask(__name__)
CORS(app)

inference_calls = 0
total_seconds = 0

def init_logging():
    LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
    logging.basicConfig(level=LOGLEVEL, format='%(asctime)s - %(levelname)s - %(message)s')

def json_answer(data, status = 200):
    json_data = json.dumps(data, indent=4, separators=(',', ': '))
    resp = Response(json_data, mimetype='application/json', status = status)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp
   
@app.route('/check', methods=['POST'])
def punctuation_api_post():
    return _punctuation_api(request.form)

@app.route('/check', methods=['GET'])
def punctuation_api_get():
    return _punctuation_api(request.args)

@app.route('/health', methods=['GET'])
def health_api_get():
    health = {}
    health['inference_calls'] = inference_calls
    health['torch_num_threads'] = torch.get_num_threads()
    health['average_time_per_request'] = total_seconds / inference_calls if inference_calls else 0

    total = hits + misses
    phits = (hits * 100 / total) if total else 0

    health['cache_misses'] = misses
    health['cache_hits'] = f"{hits} ({phits:.2f}%)"
    health['cache_size'] = len(g_cache)
    return health

model = PunctuationModel(punctuation = ",")

NEW_LINE = "\n"

g_cache = {}
g_cache_ttl = time.time()
misses = 0
hits = 0
EXPIRE_CACHE = 30 * 60
CACHE_ITEMS = 'CACHE_ITEMS'
g_cached_enabled = (CACHE_ITEMS in os.environ and os.environ[CACHE_ITEMS].lower() == 'false') == False
print(f"Sentence cache enabled {g_cached_enabled}")

def _punctuation_api(values):
    try:
      global inference_calls, total_seconds
      global misses, hits
        
      start_time = datetime.datetime.now()

      inference_calls += 1

      text = values['text']
      result = {}
      has_line = NEW_LINE in text
      sentences = text.split(NEW_LINE)
      corrected = ""

      for sentence in sentences:
        if len(sentence.strip()) > 0:

            if g_cached_enabled and sentence in g_cache:
                hits += 1
                corrected_local = g_cache[sentence]
            else:
                corrected_local = model.restore_punctuation(sentence)

                if g_cached_enabled:
                    global g_cache_ttl
                    if time.time() > EXPIRE_CACHE + g_cache_ttl:
                        g_cache.clear()
                        g_cache_ttl = time.time()

                    misses += 1
                    g_cache[sentence] = corrected_local

            corrected += corrected_local
        else:
            corrected += sentence

        if has_line:
            corrected += NEW_LINE

      time_used = datetime.datetime.now() - start_time
      total_seconds += (time_used).total_seconds()
      result['text'] = corrected
      result['time'] = str(time_used)
      return json_answer(result)

    except Exception as exception:
        logging.error(f"_punctuation_api. Error: {exception}")
        return json_answer({}, 200)

def init():
    init_logging()
    if 'THREADS' in os.environ:
        num_threads = int(os.environ['THREADS'])
        torch.set_num_threads(num_threads)


if __name__ == '__main__':
    app.debug = True
    init()
    app.run()
else:
    init()
