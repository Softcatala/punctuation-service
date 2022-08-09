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
import srx_segmenter

srx_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'segment.srx')
rules = srx_segmenter.parse(srx_filepath)



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

g_cache = {}
g_cache_ttl = time.time()
misses = 0
hits = 0
EXPIRE_CACHE = 30 * 60
CACHE_ITEMS = 'CACHE_ITEMS'
g_cached_enabled = (CACHE_ITEMS in os.environ and os.environ[CACHE_ITEMS].lower() == 'false') == False
print(f"Sentence cache enabled {g_cached_enabled}")

def _has_dot_or_equivalent(text):
    t = text

    if t[-1:]== '.' or t[-1:] == '…' or t[-1:] == '?' or t[-1:] == '!' or t[-1:] == ':':
        return True

    if t[-2:] == '.)' or t[-2:] == '."' or t[-2:] == '.\'':
        return True

    return False

def _ensure_dot_end_of_sentence(text):
    text = text.rstrip()

    if _has_dot_or_equivalent(text) is False:
        text += "."

    return text


def _punctuation_api(values):
    try:
      global inference_calls, total_seconds
      global misses, hits
        
      start_time = datetime.datetime.now()

      inference_calls += 1

      text = values['text']
      logging.debug(f"input text: '{text}'")
      result = {}

      segmenter = srx_segmenter.SrxSegmenter(rules["Catalan"], text)
      sentences, whitespaces = segmenter.extract()

      corrected_arr = []
      for sentence in sentences:
          if g_cached_enabled and sentence in g_cache:
              hits += 1
              corrected = g_cache[sentence]
          else:
              sentence_with_dot = _ensure_dot_end_of_sentence(sentence)

              added_dot = sentence_with_dot !=sentence
              if added_dot:
                logging.debug(f" sentence: '{sentence_with_dot} (dot added)'")

              corrected = model.restore_punctuation(sentence_with_dot)
        #      logging.debug(f" corrected: '{corrected}'")
        #      logging.debug(f" sentence: '{sentence}'")

              if added_dot and corrected[-1] == '.':
                corrected = corrected[:-1]


              if g_cached_enabled:
                  global g_cache_ttl
                  if time.time() > EXPIRE_CACHE + g_cache_ttl:
                      g_cache.clear()
                      g_cache_ttl = time.time()

                  misses += 1
                  g_cache[sentence] = corrected

          corrected_arr.append(corrected)

      corrected = whitespaces[0]
      for idx in range(0, len(sentences)):
        corrected += corrected_arr[idx]
        corrected += whitespaces[idx + 1]

      time_used = datetime.datetime.now() - start_time
      total_seconds += (time_used).total_seconds()
      result['text'] = corrected
      result['time'] = str(time_used)

      logging.debug(f"final corrected: '{corrected}'")
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
