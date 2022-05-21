# -*- coding: utf-8 -*-
"""
Created on Tue May  3 18:54:37 2022

@author: eugen
"""
import re
import os
import numpy as np
import pickle
from zipfile import ZipFile
from sklearn.feature_extraction.text import CountVectorizer

corpus = []


archive = ZipFile('corpus2mw.zip', 'r')

fileList = archive.namelist()
for file in fileList:
    corpus.append((archive.read(file)).decode('UTF-8'))


regexp = re.compile('[a-zA-Z0-9 éáíúóãõçôâ-]') 

def dice(freq, pref_freqs, suff_freqs):
    pref_freqs = list(pref_freqs)
    suff_freqs = list(suff_freqs)
    return 2 * freq / (sum(pref_freqs) / len(pref_freqs) + sum(suff_freqs) / len(suff_freqs))


def scp(freq, pref_freqs, suff_freqs):
    multiplied_freqs = [pref_freq * suff_freq for pref_freq, suff_freq in zip(pref_freqs, suff_freqs)]
    return freq ** 2 * len(multiplied_freqs) / sum(multiplied_freqs)

def processText(corpus):
    corp = []
    for text in corpus:
        listT = list(text)
        i = 0
        for  c in listT:
            print(c)
            if not regexp.search(c) and not listT[i-1]==' ':
                listT.insert(i, ' ')
            i +=1
        corp.append(''.join(listT))
    return corp

readPickle = False
if readPickle:
    with open('corpusList', 'rb') as fp:
        n_list = pickle.load(fp)
else: 
    corpus =  processText(corpus)

with open('corpusList', 'wb') as fp:
        pickle.dump(corpus, fp)


vectorizer = CountVectorizer(analyzer='word', ngram_range=(2, 7))
vec_fit = vectorizer.fit_transform(corpus)
word_list = vectorizer.get_feature_names_out()
count_list = np.asarray(vec_fit.sum(axis=0))[0]
freq_dict = dict(zip(word_list,count_list))

freq_dict = {key:val for key, val in freq_dict.items() if val != 1}
    