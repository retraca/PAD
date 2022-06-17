# -*- coding: utf-8 -*-
"""
Created on Tue May  3 18:54:37 2022

@author: eugen
"""
import re
import os
import numpy as np
import math
import pickle
import matplotlib.pyplot as plt
from nltk import FreqDist
from nltk.util import ngrams 
from zipfile import ZipFile

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


def compute_freq_doc(text, minG, maxG):

   freq_dist = FreqDist()

   if len(text) > 1:
       tokens = text.strip().split()
        
       for i in range(minG, maxG+1):
           grams = ngrams(tokens, i)
           freq_dist.update(grams)

   return dict(freq_dist)


def compute_freq_corpus(minG, maxG):

   freq_dist = FreqDist()

   for text in corpus:
        if len(text) > 1:
            tokens = text.strip().split()
            
            for i in range(minG, maxG+1):
                grams = ngrams(tokens, i)
                freq_dist.update(grams)

   return dict(freq_dist)


freq_dict = compute_freq_corpus(1, 8)


#transform tuple keys to string and filter all ngrams value > 1
freq_dict = {' '.join(key):val for key, val in freq_dict.items() if val > 1}



#unigrams
single_freq_dict = {key:val for key, val in freq_dict.items() if len(key.split()) == 1}
#sort unigrams by value
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



from kneed import KneeLocator
kn = KneeLocator(stop_words_list[:,0] ,stop_words_list[:,1], curve='convex', direction='increasing')
print(int(kn.knee))

stop = 0
deltaX = 200
for idx, i, j in zip(range(0, len(values)), values, values[deltaX:]):
    if((j-i)>stop): stop = idx
print(stop)


fig = plt.figure()
ax = plt.gca()
ax.scatter(stop_words_list[:,0] ,stop_words_list[:,1] , s=1,c='blue', marker='.')
ax.set_yscale('log')
#ax.set_xscale('log')
fig.set_size_inches(10, 7)
plt.axvline(x=stop, color='r', linestyle='--')


def count_RE_in_doc(RE):
    count = 0
    for text in corpus:
        if RE in text:
            count += 1
    
    return count

def freq(RE,doc):
    freq_dict = compute_freq_doc(doc, len(RE.split()), len(RE.split()))
    
    return freq_dict[RE]


def tf_idf(RE, doc_idx):
    doc = corpus(idx)
    
    freq_RE = freq(RE,doc)
    
    return (freq_RE/len(doc))*math.log(len(corpus)/count_RE_in_doc(RE))

def calc_prob(word):
    sum_p = 0
    for doc in corpus:
        sum_p += freq(word, doc)/len(doc.split())
    return (1/len(corpus))*sum_p

def calc_cov(A,B):
    probA = calc_prob(A)
    probB = calc_prob(B)
    sum_p = 0
    for doc in corpus:
        sum_p += (freq(A, doc)/len(doc.split())-probA)*(freq(B, doc)/len(doc.split())-probB)
    return (1/len(corpus)-1)*sum_p

def correlation(A,B):
    return calc_cov(A, B)/(math.sqrt(calc_cov(A, A))*(math.sqrt(calc_cov(B, B))))


def get_distances(A,B,doc):
    listA = A.split()
    listB = B.split()
    listDoc = doc.strip().split()
    
    idx_pos_A_1 = [ i for i in range(len(listDoc)) if listDoc[i] == listA[0] ]
    idx_pos_A_2 = [ i for i in range(len(listDoc)) if listDoc[i] == listA[-1] ]
    idx_pos_B_1 = [ i for i in range(len(listDoc)) if listDoc[i] == listB[0] ]
    idx_pos_B_2 = [ i for i in range(len(listDoc)) if listDoc[i] == listB[-1] ]
    
    idx_pos_A_1_copy = idx_pos_A_1
    idx_pos_A_2_copy = idx_pos_A_2
    idx_pos_B_1_copy = idx_pos_B_1
    idx_pos_B_2_copy = idx_pos_B_2
    
    for pos, idx in enumerate(idx_pos_A_1_copy):
        for i, elem in enumerate(listA):
            if listDoc[idx+i] != elem:
                try: 
                    idx_pos_A_1.pop(pos)
                except:
                    pass
                
    for pos, idx in enumerate(idx_pos_A_2_copy):
        for i, elem in enumerate(reversed(listA)):
            if listDoc[idx-i] != elem:
                try: 
                    idx_pos_A_2.pop(pos)  
                except:
                    pass
                
    for pos, idx in enumerate(idx_pos_B_1_copy):
        for i, elem in enumerate(listB):
            if listDoc[idx+i] != elem:
                try: 
                    idx_pos_B_1.pop(pos)  
                except:
                    pass            
                
    for pos, idx in enumerate(idx_pos_B_2_copy):
        for i, elem in enumerate(reversed(listB)):
            if listDoc[idx-i] != elem:
                try: 
                    idx_pos_B_2.pop(pos)  
                except:
                    pass            
    
    listF = list(np.ma.concatenate([np.subtract(idx_pos_A_1,idx_pos_B_2),np.subtract(idx_pos_B_1,idx_pos_A_2)]))
    listF = [ i for i in listF if i > -1 ]
    
    return min(listF)/max(listF)
    
    
  
def IP(A,B):
    count = 0
    sum_dist = 0 
    for doc in corpus:
        if A in doc and B in doc:
            count += 1
            sum_dist += get_distances(A, B, doc)
        
    return 1-(1/count)*sum_dist
        
       
def sem_prox(A,B):
    return correlation(A, B)*math.sqrt(IP(A,B))


def score_implicit(RE,doc_idx):
    score = 0
    freq_dict = compute_freq_doc(corpus(doc_idx), 2, 8)
    #freq_dict = localmax(freq_dict)
    #freq_dict = filterStopWords(freq_dict)
    for k in freq_dict.keys():
        freq_dict[k] = tf_idf(k,doc_idx)
    result = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)[:10]
    for i, v in enumerate(result):
        score += sem_prox(RE, v)/(i+1)
    return score
            
            
            

    
    









    