from transformers import AutoModelForCausalLM , AutoTokenizer
import sys
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from langchain import HuggingFaceHub
import re 
import argparse

import nltk
nltk.download("punkt_tab")
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger_eng")

class LMHeadModel:

    def __init__(self, model_name):
        # Initialize the model and the tokenizer.
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def get_tokenizer(self):
        return self.tokenizer
    
    def get_predictions(self, sentence):
        # Encode the sentence using the tokenizer and return the model predictions.
        inputs = self.tokenizer.encode(sentence, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(inputs)
            predictions = outputs[0]
        return predictions
    
    def get_next_word_probabilities(self, sentence, top_k_words):
        predictions = self.get_predictions(sentence)
        next_token_candidates_tensor =  predictions[0, -1, :]
        topk_candidates_indexes = torch.topk(
            next_token_candidates_tensor, top_k_words).indices.tolist()
        topk_candidates_tokens = \
            [self.tokenizer.decode([idx]).strip() for idx in topk_candidates_indexes]
        topk_candidates_indexes=[idx for idx in topk_candidates_indexes]
        all_candidates_probabilities = torch.nn.functional.softmax(
            next_token_candidates_tensor, dim=-1)
        topk_candidates_probabilities = \
            all_candidates_probabilities[topk_candidates_indexes].tolist()
        
        return zip(topk_candidates_tokens, topk_candidates_probabilities)
        
    

def generate_probs(lines, lmm, top_k, output_file):
    print("Generating probabilities...")
    Probs=[]
    Sentences=[]
 
    for i in tqdm(range(len(lines))):
        line=lines[i]
        sentence=""
        for time in range(0, len(line)):
            word=line[time] 
            sentence+=" "+word
            if time == len(line)-1:
                probs=lmm.get_next_word_probabilities(sentence, top_k)
                Probs.append(probs)  
                Sentences.append(sentence)

    outp=open(output_file+".txt","w",encoding="utf-8")
    outp2=open(output_file+".tsv","w",encoding="utf-8")

    for (sentence_idx,sentence, probs) in zip(range(len(Sentences)),Sentences, Probs):
        outp.write("Sentence:%s\n"%(sentence))
        count = 1
        for (word, prob) in probs:
            word = re.sub(r'[^a-zA-Z0-9]', '', word)
            if len(word) != 1:
                text = nltk.word_tokenize(word)
                pos_tagged = nltk.pos_tag(text)
                nouns = [word for word, tag in pos_tagged if tag in ('NN', 'NNS')]
                if nouns != []:
                    outp.write("{}\t{}\t{}\n".format(count, word, prob))
                    outp2.write("{}\t{}\t{}\t{}\n".format(sentence_idx,count, word, prob))
            count += 1
        outp.write("\n\n")

    outp.close()
    outp2.close()
    print("See %s for output."%(output_file))
    return
     

def main(input_file, top_k, output_file,llm):
    with open(input_file,"r") as f:
        lines = [z for z in [x.rstrip().split(" ") for x in f.readlines()]]
    generate_probs(lines, llm, top_k, output_file)
    
# Register at HuggingFace, and install the huggingface-cli (command line interface).
# Then: prior to running this script: huggingface-cli login 
# Enter your secret READ token (see Huggingface, under your profile: Access tokens)
# Then run this script. NB: it will download about 20GB model stuff. 

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='./input.txt')
    parser.add_argument('--output', default='./output.txt')
    parser.add_argument('--top_k', default=10,type=int)
    parser.add_argument('--gender',default=False,action='store_true')
    args = parser.parse_args()
    llm = LMHeadModel("meta-llama/Meta-Llama-3-8B-Instruct")
    if args.gender:
        gender = "man"
        main(f"{args.input}_{gender}.txt",args.top_k,f"{args.output}_{gender}",llm) # file with one sentence per line, top k word probabilities (number), output file name
        gender = "woman"
        main(f"{args.input}_{gender}.txt",args.top_k,f"{args.output}_{gender}",llm) # file with one sentence per line, top k word probabilities (number), output file name
    else:
        main(f"{args.input}.txt",args.top_k,f"{args.output}",llm)