#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
from fastapi import FastAPI, HTTPException, Query, Request, Response
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
from starlette.middleware.base import BaseHTTPMiddleware
import asyncio
from asyncio import Semaphore
import time

logsetup = LogSetup()
logsetup.init_logging()


calls = {}

def inc_counter(key):
    global calls
    value = calls.get(key, 0)
    value +=1
    calls[key] = value


total_seconds = 0
total_words = 0
total_cached_sentences = 0
total_uncached_sentences = 0
total_tokens = 0
total_tokens_processed_ms = 0
app = FastAPI()
start_time = datetime.datetime.now()

model_name = "model"
tokenizer = MT5Tokenizer.from_pretrained(model_name)

too_busy_requests = 0  # Counter for requests dropped due to timeout
requests_had_to_wait = 0
requests_had_to_wait_time = 0
MAX_WAIT_SECONDS = 30   # Max waiting time before returning 503


# Limit to N concurrent requests
MAX_CONCURRENT_REQUESTS = 1
semaphore = Semaphore(MAX_CONCURRENT_REQUESTS)

class DropLongWaitingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        global too_busy_requests, requests_had_to_wait, requests_had_to_wait_time
        
        # Check if semaphore is immediately available
        had_to_wait = semaphore.locked()
        start_time = time.monotonic()
        
        # Try to acquire semaphore with timeout
        try:
            await asyncio.wait_for(semaphore.acquire(), timeout=MAX_WAIT_SECONDS)
            wait_time = time.monotonic() - start_time
            if had_to_wait:
                requests_had_to_wait += 1
                logging.debug(f"Request had to wait for {wait_time:.3f} seconds")
                time_used = time.monotonic() - start_time
                requests_had_to_wait_time += time_used
  
                  
        except asyncio.TimeoutError:
            too_busy_requests += 1
            logging.warning("Too busy, try later - timeout exceeded")
            return Response("Too busy, try later", status_code=503)
        
        try:
            response = await call_next(request)
            return response
        finally:
            semaphore.release()
            
app.add_middleware(DropLongWaitingMiddleware)


def get_model():
    inter_threads = int(os.environ.get('CTRANSLATE_INTER_THREADS', 2))
    intra_threads = int(os.environ.get('CTRANSLATE_INTRA_THREADS', 6))
    device = os.environ.get('DEVICE', "cpu")
    compute_type = os.environ.get("COMPUTE_TYPE", "int8")
    logging.info(f"device: {device}, inter_threads: {inter_threads}, intra_threads: {intra_threads}, compute_type: {compute_type}")
    return ctranslate2.Translator(model_name, device=device, inter_threads=inter_threads, intra_threads=intra_threads, compute_type=compute_type)

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
    global total_seconds, total_words, total_cached_sentences, total_uncached_sentences

    start_time = datetime.datetime.now()
    uncached_sentences = cache.get_uncached(sentences)
    uncached_sentences_corrected = []
    if uncached_sentences:
        inputs = [tokenizer.convert_ids_to_tokens(tokenizer.encode(sentence, add_special_tokens=True)) for sentence in uncached_sentences]
        total_input_tokens = sum(len(tokens) for tokens in inputs)
        total_tokens+= total_input_tokens
        
        start = time.perf_counter() 
        results = model.translate_batch(inputs, max_decoding_length=300, batch_type="examples", beam_size=2, use_vmap=True)
        end = time.perf_counter()
        elapsed_ms = (end - start) * 1000
        total_tokens_processed_ms += elapsed_ms
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
async def process_text_post(input_text: TextInput):
    inc_counter("post_check")
    if not input_text.sentences:
        raise HTTPException(status_code=400, detail="Input text list cannot be empty")

    return process_sentences(input_text.sentences)

# El mètode GET només s'usa per a fer tests, amb una única frase.
@app.get("/check")
async def process_text_get(text: str = Query(..., description="Text to check")):
    inc_counter("get_check")
    if not text:
        raise HTTPException(status_code=400, detail="Input text cannot be empty")
    input_text = TextInput(sentences=[])
    input_text.sentences.append(text)
    return process_sentences(input_text.sentences)
    
@app.get('/health')
async def health_get():
    inc_counter("health")

    health = {}
    rss = psutil.Process(os.getpid()).memory_info().rss // 1024 ** 2
    health['id'] = os.getpid()
    health['rss'] = f"{rss} MB"
    calls_check = calls.get("post_check", 0) + calls.get("get_check", 0)
    health['average_time_per_request'] = total_seconds / calls_check if calls_check else 0
    health['cached_sentences'] = total_cached_sentences
    health['uncached_sentences'] = total_uncached_sentences
    health['words_per_second'] = total_words / total_seconds if total_seconds else 0
    health['uptime'] = str(datetime.datetime.now() - start_time).split('.')[0]
    health['tokens_per_second'] = total_tokens / (total_tokens_processed_ms / 1000)  if total_tokens_processed_ms > 0 else 0
    health['too_busy_requests'] = too_busy_requests
    health['requests_had_to_wait'] = requests_had_to_wait
    health['requests_had_to_wait_avg_time'] = requests_had_to_wait_time / requests_had_to_wait if requests_had_to_wait else 0
    for key, value in calls.items():
        health[f'method_{key}'] = value
    return health

@app.get('/download-log')
async def download_log(code: str = None):
    inc_counter("download")

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
