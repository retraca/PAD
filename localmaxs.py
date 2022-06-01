# -*- coding: utf-8 -*-
"""
Created on Tue May  3 18:54:37 2022

@author: eugen
"""
import re
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from zipfile import ZipFile
from sklearn.feature_extraction.text import CountVectorizer

corpus = []


archive = ZipFile('corpus2mw.zip', 'r')

fileList = archive.namelist()
for file in fileList:
    corpus.append((archive.read(file)).decode('UTF-8'))


regexp = re.compile('[\w \-]') 

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
            if (not regexp.search(c) and not listT[i-1]==' ') or not regexp.search(listT[i-1]) and regexp.search(c):
                listT.insert(i, ' ')
            i +=1
        corp.append(''.join(listT))
    return corp

readPickle = False
if readPickle:
    with open('corpusList', 'rb') as fp:
        corpus = pickle.load(fp)
else: 
    corpus =  processText(corpus)

with open('corpusList', 'wb') as fp:
        pickle.dump(corpus, fp)



vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 8),token_pattern=r'(?u)\b\w\w+\b|(?u)\b^\w^\w+\b')
vec_fit = vectorizer.fit_transform(corpus)
word_list = vectorizer.get_feature_names_out()
count_list = np.asarray(vec_fit.sum(axis=0))[0]
freq_dict = dict(zip(word_list,count_list))

freq_dict = {key:val for key, val in freq_dict.items() if val > 1}




vectorizer = CountVectorizer(analyzer='word',token_pattern=r'\w+|[^\w]')
vec_fit = vectorizer.fit_transform(corpus)
single_word_list = vectorizer.get_feature_names_out()
single_count_list = np.asarray(vec_fit.sum(axis=0))[0]
single_freq_dict = dict(zip(single_word_list,single_count_list))
'''
vectorizer = CountVectorizer(analyzer='word',token_pattern=r'[\w^\w+]')
vec_fit = vectorizer.fit_transform(corpus)
single_s_char_list = vectorizer.get_feature_names_out()
single_s_char_count_list = np.asarray(vec_fit.sum(axis=0))[0]
single_freq_dict.update(dict(zip(single_s_char_list,single_s_char_count_list)))
'''
single_freq_dict = {key:val for key, val in single_freq_dict.items() if val > 1}
single_freq_dict = {k: v for k, v in sorted(single_freq_dict.items(), key=lambda item: item[1])}
values = np.fromiter(single_freq_dict.values(), dtype=float)
stop_words_list = np.stack((np.arange(0, len(single_freq_dict)), values), axis = -1)
list_of_counts = list(single_freq_dict.items())


from kneebow.rotor import Rotor
 
rotor = Rotor()
rotor.fit_rotate(stop_words_list)
elbow_idx = rotor.get_elbow_index()
print(elbow_idx)  
#rotor.plot_elbow()


'''
from kneed import KneeLocator
kn = KneeLocator(stop_words_list[:,0] ,stop_words_list[:,1], curve='convex', direction='increasing')
print(int(kn.knee))

stop = 0
deltaX = 100
for idx, i, j in zip(range(0, len(values)), values, values[deltaX:]):
    if((j-i)>stop): stop = idx
print(stop)
'''

fig = plt.figure()
ax = plt.gca()
ax.scatter(stop_words_list[:,0] ,stop_words_list[:,1] , s=1,c='blue', marker='.')
ax.set_yscale('log')
#ax.set_xscale('log')
fig.set_size_inches(10, 7)
plt.axvline(x=elbow_idx, color='r', linestyle='--')












    