import pandas as pd
import numpy as np
import argparse
from collections import defaultdict
from scipy.special import kl_div
from scipy.stats import pearsonr

def read_text():
    sen = []
    with open("sentences_man.txt","r") as f:
        for line in f:
            if line[-1]=="\n":
                sen.append(line[:-1])
            else:
                sen.append(line)
    return sen

def write_dict(dic,name,name2,double=False):
    with open(f"./rankings/ranking_{name}_{name2}.txt","w") as f:
        for (key,value) in dic:
            if not double:
                f.write(f"{key}\t{value}\n")
            else:
                f.write(f"{key}\t{value[0]}\t{value[1]}\n")


def all_sentences():
    sentences = read_text()
    df = pd.read_csv("output_man.tsv",delimiter="\t",names=["sentence","rank","word","prob"])
    df2 = pd.read_csv("output_woman.tsv",delimiter="\t",names=["sentence","rank","word","prob"])
    vocab = []
    vocab_man = defaultdict(np.float32)
    vocab_woman = defaultdict(np.float32)
    vocab_all = defaultdict(lambda: [0.0, 0.0])
    for i,row in df.iterrows():
        # print(row["prob"])
        if row["word"] not in vocab:
            vocab.append(row["word"])
        vocab_man[row["word"]] += np.float32(row["prob"])
        vocab_all[row["word"]][0] += row["prob"]

    for i,row in df2.iterrows():
        # print(row["prob"])
        if row["word"] not in vocab:
            vocab.append(row["word"])
        vocab_woman[row["word"]] += np.float32(row["prob"])
        vocab_all[row["word"]][1] += row["prob"]
    print(len(vocab))
    sort_man = sorted(list(vocab_man.items()),key=lambda x: x[1],reverse=True)
    sort_woman = sorted(list(vocab_woman.items()),key=lambda x: x[1],reverse=True)
    sort_all = sorted(list(vocab_all.items()),key=lambda x: x[1][0],reverse=True)

    name = "all"
    write_dict(sort_man,"man",name)
    write_dict(sort_woman,"woman",name)
    write_dict(sort_all,"all",name,True)

    sort_all_man,sort_all_woman = np.array([x[1][0] for x in sort_all]),np.array([x[1][1] for x in sort_all])
    sen_man = sort_all_man/len(sentences)
    sen_woman = sort_all_woman/len(sentences)
    sort_all_man /= np.sum(sort_all_man)
    sort_all_woman /= np.sum(sort_all_woman)
    kl = np.sum([sort_all_man[i] * np.log(sort_all_man[i]/sort_all_woman[i]) for i in range(len(sort_all_man)) if (sort_all_woman[i] != 0.0 and sort_all_man[i] != 0.0)])
    kl2 = np.sum([sort_all_woman[i] * np.log(sort_all_woman[i]/sort_all_man[i]) for i in range(len(sort_all_man)) if (sort_all_woman[i] != 0.0 and sort_all_man[i] != 0.0)])
    r2 = pearsonr(sen_man,sen_woman)
    print(f"KL divergence\nD(man,woman): {kl}\nD(woman,man): {kl2}\nPearson\nstat: {r2.statistic}\np-value:{r2.pvalue}")

def do_stuff(word_ending,writer,num):
    sentences = read_text()
    df = pd.read_csv("output_man.tsv",delimiter="\t",names=["sentence","rank","word","prob"])
    df2 = pd.read_csv("output_woman.tsv",delimiter="\t",names=["sentence","rank","word","prob"])
    vocab = []
    vocab_man = defaultdict(np.float32)
    vocab_woman = defaultdict(np.float32)
    vocab_all = defaultdict(lambda: [0.0, 0.0])
    for idx in range(len(sentences)):
        if sentences[idx].split(" ")[num] == word_ending:
            for i,row in df[df["sentence"]==idx].iterrows():
                # print(row["prob"])
                if row["word"] not in vocab:
                    vocab.append(row["word"])
                vocab_man[row["word"]] += np.float32(row["prob"])
                vocab_all[row["word"]][0] += row["prob"]

            for i,row in df2[df2["sentence"]==idx].iterrows():
                # print(row["prob"])
                if row["word"] not in vocab:
                    vocab.append(row["word"])
                vocab_woman[row["word"]] += np.float32(row["prob"])
                vocab_all[row["word"]][1] += row["prob"]
    sort_man = sorted(list(vocab_man.items()),key=lambda x: x[1],reverse=True)
    sort_woman = sorted(list(vocab_woman.items()),key=lambda x: x[1],reverse=True)
    sort_all = sorted(list(vocab_all.items()),key=lambda x: x[1][0],reverse=True)

    write_dict(sort_man,"man",word_ending)
    write_dict(sort_woman,"woman",word_ending)
    write_dict(sort_all,"all",word_ending,True)

    sort_all_man,sort_all_woman = np.array([x[1][0] for x in sort_all]),np.array([x[1][1] for x in sort_all])
    sen_man = sort_all_man/len(sentences)
    sen_woman = sort_all_woman/len(sentences)
    sort_all_man /= np.sum(sort_all_man)
    sort_all_woman /= np.sum(sort_all_woman)
    kl = np.sum([sort_all_man[i] * np.log(sort_all_man[i]/sort_all_woman[i]) for i in range(len(sort_all_man)) if (sort_all_woman[i] != 0.0 and sort_all_man[i] != 0.0)])
    kl2 = np.sum([sort_all_woman[i] * np.log(sort_all_woman[i]/sort_all_man[i]) for i in range(len(sort_all_man)) if (sort_all_woman[i] != 0.0 and sort_all_man[i] != 0.0)])
    r2 = pearsonr(sen_man,sen_woman)
    writer += f"\nword_class: {word_ending}\n{len(vocab)}\nKL divergence\nD(man,woman): {kl}\nD(woman,man): {kl2}\nPearson\nstat: {r2.statistic}\np-value:{r2.pvalue}\n"
    return writer

def do_stuff2(writer):
    sentences = read_text()
    df = pd.read_csv("output_man.tsv",delimiter="\t",names=["sentence","rank","word","prob"])
    df2 = pd.read_csv("output_woman.tsv",delimiter="\t",names=["sentence","rank","word","prob"])
    vocab = []
    vocab_man = defaultdict(np.float32)
    vocab_woman = defaultdict(np.float32)
    vocab_all = defaultdict(lambda: [0.0, 0.0])
    for idx in range(len(sentences)):
        if sentences[idx].split(" ")[1] != "man" and sentences[idx].split(" ")[1] != "boy":
            for i,row in df[df["sentence"]==idx].iterrows():
                # print(row["prob"])
                if row["word"] not in vocab:
                    vocab.append(row["word"])
                vocab_man[row["word"]] += np.float32(row["prob"])
                vocab_all[row["word"]][0] += row["prob"]

            for i,row in df2[df2["sentence"]==idx].iterrows():
                # print(row["prob"])
                if row["word"] not in vocab:
                    vocab.append(row["word"])
                vocab_woman[row["word"]] += np.float32(row["prob"])
                vocab_all[row["word"]][1] += row["prob"]
    sort_man = sorted(list(vocab_man.items()),key=lambda x: x[1],reverse=True)
    sort_woman = sorted(list(vocab_woman.items()),key=lambda x: x[1],reverse=True)
    sort_all = sorted(list(vocab_all.items()),key=lambda x: x[1][0],reverse=True)

    name = "extra"
    write_dict(sort_man,"man",name)
    write_dict(sort_woman,"woman",name)
    write_dict(sort_all,"all",name,True)

    sort_all_man,sort_all_woman = np.array([x[1][0] for x in sort_all]),np.array([x[1][1] for x in sort_all])
    sen_man = sort_all_man/len(sentences)
    sen_woman = sort_all_woman/len(sentences)
    sort_all_man /= np.sum(sort_all_man)
    sort_all_woman /= np.sum(sort_all_woman)
    kl = np.sum([sort_all_man[i] * np.log(sort_all_man[i]/sort_all_woman[i]) for i in range(len(sort_all_man)) if (sort_all_woman[i] != 0.0 and sort_all_man[i] != 0.0)])
    kl2 = np.sum([sort_all_woman[i] * np.log(sort_all_woman[i]/sort_all_man[i]) for i in range(len(sort_all_man)) if (sort_all_woman[i] != 0.0 and sort_all_man[i] != 0.0)])
    r2 = pearsonr(sen_man,sen_woman)
    writer += f"\nword_class: rest\n{len(vocab)}\nKL divergence\nD(man,woman): {kl}\nD(woman,man): {kl2}\nPearson\nstat: {r2.statistic}\np-value:{r2.pvalue}\n"
    return writer

def filter_sentences_ending():
    writer = ""
    for word_ending in ["as","an","a"]:
        writer = do_stuff(word_ending,writer,-1)
    print(writer)

def filter_sentences_gender():
    writer = ""
    for word_ending in ["boy","man"]:
        writer = do_stuff(word_ending,writer,1)
    writer = do_stuff2(writer)
    print(writer)

if __name__=="__main__":
    all_sentences()
    filter_sentences_ending()
    filter_sentences_gender() # file with one sentence per line, top k word probabilities (number), output file name

    # read_text()
