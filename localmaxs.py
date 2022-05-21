# -*- coding: utf-8 -*-
"""
Created on Tue May  3 18:54:37 2022

@author: eugen
"""
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

corpus = ['Knight Lore é um jogo eletrônico de ação e aventura desenvolvido e lançado pela Ultimate Play the Game para o ZX Spectrum em 1984. O jogo é conhecido pelo uso de gráficos isométricos, que popularizou ainda mais em jogos eletrônicos. Em Knight Lore, Sabreman tem quarenta dias para coletar objetos em um castelo e preparar uma cura para sua maldição do lobisomem. Cada sala do castelo é representada em monocromático em sua própria tela e consiste em blocos para escalar, obstáculos para evitar e quebra-cabeças para resolver.',
          'Hello there General Kenobi.']


regexp = re.compile('[a-zA-Z0-9 éáíúóãõçôâ-]') 

'''
import os
for filename in os.listdir():
   with open(os.path.join(os.getcwd(), filename), 'r') as f:
       corpus.append(f.read())
'''

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
            if not regexp.search(c) and not listT[i-1]==' ':
                listT.insert(i, ' ')
            i +=1
        corp.append(''.join(listT))
    return corp

corpus =  processText(corpus)


vectorizer = CountVectorizer(analyzer='word', ngram_range=(2, 7))
vec_fit = vectorizer.fit_transform(corpus)
word_list = vec_fit.get_feature_names_out()
count_list = np.asarray(vec_fit.sum(axis=0))
freq_dict = dict(zip(word_list,count_list))