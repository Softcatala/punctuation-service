from transformers import pipeline, RobertaForTokenClassification, AutoTokenizer
import torch
from torch.quantization import quantize_dynamic
import torch.nn as nn
import re
import logging

class PunctuationModel():
    def __init__(self, model_name = "model/", punctuation = ".,;:?") -> None:

        self.pipe = self._regular_pipeline(model_name)
        self.punctuation = punctuation
        self.fixes_stats = {}

    def _regular_pipeline(self, model_name):
        if torch.cuda.is_available():
            pipe = pipeline("ner", model_name, grouped_entities=False, device=0)
        else:
            pipe = pipeline("ner", model_name, grouped_entities=False)

        return pipe

    # While this is twice faster, there is a ~15% drop in accuracy
    def _quantized_pipeline(self, model_name):
        model = (RobertaForTokenClassification.from_pretrained(model_name).to("cpu"))
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model_quantized = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

        if torch.cuda.is_available():
            pipe = pipeline("ner", model_quantized, grouped_entities=False, device=0, tokenizer=tokenizer)
        else:
            pipe = pipeline("ner", model_quantized, grouped_entities=False, tokenizer=tokenizer)

        return pipe

    def preprocess(self,text):
        #remove markers except for markers in numbers
        reg = f"(?<!\d)[{self.punctuation}](?!\d)"
        text = re.sub(reg,"",text)
#        text = re.sub(r"(?<!\d)[.,;:!?](?!\d)","",text) 
        #todo: match acronyms https://stackoverflow.com/questions/35076016/regex-to-match-acronyms
        text = text.split()
        return text

    def restore_punctuation(self,text):        
        result = self.predict(self.preprocess(text))
        prediction = self.prediction_to_text(result)
        if self._contains_invalid_prediction(prediction):
            return text

        return prediction
        
    def overlap_chunks(self,lst, n, stride=0):
        """Yield successive n-sized chunks from lst with stride length of overlap."""
        for i in range(0, len(lst), n-stride):
                yield lst[i:i + n]

    def predict(self,words):
        overlap = 5
        chunk_size = 230
        if len(words) <= chunk_size:
            overlap = 0

        batches = list(self.overlap_chunks(words,chunk_size,overlap))

        # if the last batch is smaller than the overlap, 
        # we can just remove it
        if len(batches[-1]) <= overlap:
            batches.pop()

        tagged_words = []     
        for batch in batches:
            # use last batch completely
            if batch == batches[-1]: 
                overlap = 0
            text = " ".join(batch)
            result = self.pipe(text)      
            assert len(text) == result[-1]["end"], "chunk size too large, text got clipped"
                
            char_index = 0
            result_index = 0
            for word in batch[:len(batch)-overlap]:                
                char_index += len(word) + 1
                # if any subtoken of an word is labled as sentence end
                # we label the whole word as sentence end        
                label = 0
                while result_index < len(result) and char_index > result[result_index]["end"] :
                    label = result[result_index]['entity']
                    score = result[result_index]['score']
                    result_index += 1                        
                tagged_words.append([word,label, score])
        
        assert len(tagged_words) == len(words)
        return tagged_words

    def prediction_to_text(self,prediction):
        result = ""
        for word, label, _ in prediction:
            result += word

            if label == "." and '.' in self.punctuation:
                result += "."
            elif label == "," and ',' in self.punctuation:
                result += ","
            elif label == "?" and '?' in self.punctuation:
                result += "?"
            elif label == "-" and '-' in self.punctuation:
                result += "-"
            elif label == ":" and ':' in self.punctuation:
                result += ":"

            result += " "

        return result.strip()

    def _save_fixes_stats(self, fix):
        if fix not in self.fixes_stats:
            self.fixes_stats[fix] = 1
        else:
            counter = self.fixes_stats[fix]
            counter += 1
            self.fixes_stats[fix] = counter

    HTML_REGEX = re.compile(r"\<(.*?)\>", re.VERBOSE)
    # Check for common prediction mistakes from the model.
    # If a mistake is found, invalidates the full prediction
    def _contains_invalid_prediction(self, prediction):
        invalid = False

        # Some times we get text with HTML tags (paste may be) and we do not predict well
        matches = self.HTML_REGEX.findall(prediction)
        if len(matches) > 0:
            self._save_fixes_stats("html tag")
            return True

        fixes = [
                ",,",
                ";,",
                ":,",
                ", —",
                ", perquè ",
                "a més, a més",
                ", de" # Com sempre em constesteu de seguida m'ha estranyat.
                # These are not incorrect but have very low ratio of acceptance by the users
               "bon dia,",
               "bona tarda,",
               "bona nit,",
               "bona vesprada,",
               "bon vespre,",
               "hola,",
               "bones,"
        ]

        text = prediction.lower()
        for fix in fixes:
            position = text.find(fix)
            if position != -1:
                self._save_fixes_stats(fix)
                invalid = True

        return invalid
