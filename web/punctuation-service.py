#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import torch
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
from collections import OrderedDict
import datetime

#TODO: health, log, cache

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI()

model_name = "model/"
tokenizer = MT5Tokenizer.from_pretrained(model_name)
model = MT5ForConditionalGeneration.from_pretrained(model_name).to(device)

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
    if uncached_sentences:
        input_ids = tokenizer(uncached_sentences, return_tensors="pt", padding=True, truncation=True, max_length=256).input_ids.to(device)
        outputs = model.generate(input_ids, max_length=300)
        uncached_sentences_corrected = [tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True) for output in outputs]
        for i in range(len(uncached_sentences)):
            cache.put(uncached_sentences[i], uncached_sentences_corrected[i])
    output_sentences = [cache.get(s) for s in sentences]
    time_used = datetime.datetime.now() - start_time
    total_seconds += (time_used).total_seconds()
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
    