#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from transformers import MT5Tokenizer
import ctranslate2
from collections import OrderedDict
import datetime

#TODO: health, log


app = FastAPI()

model_name = "model"
tokenizer = MT5Tokenizer.from_pretrained(model_name)
model = ctranslate2.Translator(model_name, device="cpu")

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
    start_time = datetime.datetime.now()
    uncached_sentences = cache.get_uncached(sentences)
    uncached_sentences_corrected = []
    if uncached_sentences:
        inputs = [tokenizer.convert_ids_to_tokens(tokenizer.encode(sentence, add_special_tokens=True)) for sentence in uncached_sentences]
        results = model.translate_batch(inputs, max_decoding_length=300)
        for result in results:
            decoded_text = tokenizer.decode(tokenizer.convert_tokens_to_ids(result.hypotheses[0]), skip_special_tokens=True)
            uncached_sentences_corrected.append(decoded_text)
        for i in range(len(uncached_sentences)):
            cache.put(uncached_sentences[i], uncached_sentences_corrected[i])
    output_sentences = [cache.get(s) for s in sentences]
    time_used = datetime.datetime.now() - start_time
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
    