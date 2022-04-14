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

app = Flask(__name__)
CORS(app)

inference_calls = 0
total_miliseconds = 0
total_words = 0
start_time = time.time()

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
    return health

model = PunctuationModel(punctuation = ",")

def _punctuation_api(values):
    try:
        global inference_calls
        start = datetime.datetime.now()

        inference_calls += 1

        text = values['text']
        result = {}
        result['text'] = model.restore_punctuation(text)
#       result['text'] = text
        end = datetime.datetime.now()
        result['time'] = (end-start).microseconds / 1000
        return json_answer(result)

    except Exception as exception:
        logging.error(f"_punctuation_api. Error: {exception}")
        return json_answer({}, 200)

if __name__ == '__main__':
    app.debug = True
    init_logging()
    app.run()
else:
    init_logging()
