#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from transformers import MT5Tokenizer
import ctranslate2
from collections import OrderedDict
import datetime
import os
import psutil
import logging
import logging.handlers
from fastapi.responses import FileResponse
from tempfile import NamedTemporaryFile
import tarfile
from logsetup import LogSetup


logsetup = LogSetup()
logsetup.init_logging()


calls = 0
total_seconds = 0
total_words = 0
total_cached_sentences = 0
total_uncached_sentences = 0
app = FastAPI()

model_name = "model"
tokenizer = MT5Tokenizer.from_pretrained(model_name)



def get_model():
    inter_threads = int(os.environ.get('CTRANSLATE_INTER_THREADS', 2))
    intra_threads = int(os.environ.get('CTRANSLATE_INTRA_THREADS', 6))
    device = os.environ.get('DEVICE', "cpu")
    logging.info(f"device: {device}, inter_threads: {inter_threads}, intra_threads: {intra_threads}")
    return ctranslate2.Translator(model_name, device="cpu", inter_threads=inter_threads, intra_threads=intra_threads)

model = get_model()

class LRUCache:
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key: str):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: str, value: str):
        if key in self.cache:
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)
        self.cache[key] = value
    
    def get_uncached(self, sentences: list):
        return [s for s in sentences if self.get(s) is None]

    def __repr__(self):
        return f"LRUCache({self.cache})"
    
cache = LRUCache(10000)

class TextInput(BaseModel):
    sentences: list[str]

def process_sentences(sentences: list[str]) -> list[str]:
    global calls, total_seconds, total_words, total_cached_sentences, total_uncached_sentences

    calls +=1
    start_time = datetime.datetime.now()
    uncached_sentences = cache.get_uncached(sentences)
    uncached_sentences_corrected = []
    if uncached_sentences:
        inputs = [tokenizer.convert_ids_to_tokens(tokenizer.encode(sentence, add_special_tokens=True)) for sentence in uncached_sentences]
        results = model.translate_batch(inputs, max_decoding_length=300, batch_type="examples", beam_size=2, use_vmap=True)
        # , max_batch_size=64, beam_size=4 (més precisió). beam_size=1 (més ràpid)
        for result in results:
            decoded_text = tokenizer.decode(tokenizer.convert_tokens_to_ids(result.hypotheses[0]), skip_special_tokens=True)
            uncached_sentences_corrected.append(decoded_text)
        for i in range(len(uncached_sentences)):
            cache.put(uncached_sentences[i], uncached_sentences_corrected[i])
    
    num_uncached_sentences = len(uncached_sentences)
    num_cached_sentences = len(sentences) - num_uncached_sentences

    total_cached_sentences += num_cached_sentences
    total_uncached_sentences += num_uncached_sentences

    output_sentences = [cache.get(s) for s in sentences]
    time_used = datetime.datetime.now() - start_time

    total_seconds += (time_used).total_seconds()
    words = sum(len(s.split()) for s in sentences)
    words_in_uncached_sentences = sum(len(s.split()) for s in uncached_sentences) if uncached_sentences else 0
    total_words += words
    logging.debug(f"words_total {words}, words_in_uncached_sentences {words_in_uncached_sentences}, time {time_used}, cached_sentences {num_cached_sentences}, uncached_sentences {num_uncached_sentences}, pid {os.getpid()}")
    return {"output_sentences": output_sentences, "time": str(time_used)}

# L'entrada del mètode POST és una llista de frases. 
@app.post("/check")
def process_text(input_text: TextInput):
    if not input_text.sentences:
        raise HTTPException(status_code=400, detail="Input text list cannot be empty")

    return process_sentences(input_text.sentences)

# El mètode GET només s'usa per a fer tests, amb una única frase.
@app.get("/check")
def process_text(text: str = Query(..., description="Text to check")):
    if not text:
        raise HTTPException(status_code=400, detail="Input text cannot be empty")
    input_text = TextInput(sentences=[])
    input_text.sentences.append(text)
    return process_sentences(input_text.sentences)
    
@app.get('/health')
def health_get():
    health = {}
    rss = psutil.Process(os.getpid()).memory_info().rss // 1024 ** 2
    health['id'] = os.getpid()
    health['rss'] = f"{rss} MB"
    health['average_time_per_request'] = total_seconds / calls if calls else 0
    health['calls'] = calls
    health['cached_sentences'] = total_cached_sentences
    health['uncached_sentences'] = total_uncached_sentences
    health['words_per_second'] = total_words / total_seconds if total_seconds else 0
    return health

@app.get('/download-log')
def download_log(code: str = None):

    CODE = os.environ.get("DOWNLOAD_CODE", "")
    if len(CODE) > 0 and code != CODE:
        raise HTTPException(status_code=401)
        
    log_files = logsetup.get_logfiles()

    if not log_files:
        raise HTTPException(status_code=404, detail="No log files found")

    with NamedTemporaryFile(delete=False, mode='wb') as temp_gzip:
        with tarfile.open(temp_gzip.name, 'w:gz') as tar:
            for log_file in log_files:
                tar.add(log_file, arcname=os.path.basename(log_file))

        return FileResponse(temp_gzip.name, media_type='application/gzip', filename='punctuation-service-logs.tar.gz')
