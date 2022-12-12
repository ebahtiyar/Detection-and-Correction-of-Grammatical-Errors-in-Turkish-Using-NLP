# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 14:56:24 2022
load model and spell checker 
@author: emreb
"""

from keras.models import load_model
import re
import os
from gensim.models import Word2Vec
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
#Model De Da
def model_de_da():
    model_deda = load_model(os.path.join(r"C:\Users\emreb\Documents\web_proje\models\model_da_de",'model_dade_w2v_3.h5'))
    Word2Vec_model_de =Word2Vec.load(r"C:\Users\emreb\Documents\web_proje\models\model_da_de\Word2Vec_model")
    return model_deda,Word2Vec_model_de

def model_ki():
    model_ki  = load_model(os.path.join(r"C:\Users\emreb\Documents\web_proje\models\model_ki",'model_ki_w2v_1.h5'))
    Word2Vec_model=Word2Vec.load(r"C:\Users\emreb\Documents\web_proje\models\model_ki\Word2Vec_model_ki")
    
    return model_ki,Word2Vec_model


def model_punc():
    model_comm = load_model(os.path.join("C:/Users/emreb/Desktop/web_proje/models/model_punc",'model_comm2.h5'))
    
    with open('C:/Users/emreb/Desktop/web_proje/models/model_punc/in_train_tokenizer_comm2', 'rb') as handle:
        inn_c = pickle.load(handle)


    with open('C:/Users/emreb/Desktop/web_proje/models/model_punc/out_train_tokenizer_comm2', 'rb') as handle:
        out_c = pickle.load(handle)
        
    model_pq = load_model(os.path.join("C:/Users/emreb/Desktop/web_proje/models/model_punc",'model_perque.h5'))

    with open('C:/Users/emreb/Desktop/web_proje/models/model_punc/in_train_tokenizer_perque', 'rb') as handle:
        inn_pq = pickle.load(handle)


    with open('C:/Users/emreb/Desktop/web_proje/models/model_punc/out_train_tokenizer_perque', 'rb') as handle:
        out_pq = pickle.load(handle)
        
    return model_comm,inn_c,out_c,model_pq,inn_pq ,out_pq

def model_punc1():
    model_main  = load_model(os.path.join("C:/Users/emreb/Desktop/web_proje/models/model_punc",'model_punc_1.h5'))

    with open('C:/Users/emreb/Desktop/web_proje/models/model_punc/in_train_tokenizer_punc', 'rb') as handle:
        inn_p = pickle.load(handle)


    with open('C:/Users/emreb/Desktop/web_proje/models/model_punc/out_train_tokenizer_punc', 'rb') as handle:
        out_p = pickle.load(handle)

    return model_main,inn_p ,out_p


def model_proper():
    model_main  = load_model(os.path.join("C:/Users/emreb/Desktop/web_proje/models/model_proper",'model_proper_d_1.h5'))

    with open('C:/Users/emreb/Desktop/web_proje/models/model_proper/in_train_tokenizer_proper_d', 'rb') as handle:
        inn_p = pickle.load(handle)


    with open('C:/Users/emreb/Desktop/web_proje/models/model_proper/out_train_tokenizer_proper_d', 'rb') as handle:
        out_p = pickle.load(handle)

    with open('C:/Users/emreb/Desktop/main_proje/nonproper',"rb") as fp:   # Unpickling
          controlWord = pickle.load(fp)
          
    return model_main,inn_p ,out_p ,controlWord

def model_hal():
    model = load_model(os.path.join("C:/Users/emreb/Desktop/web_proje/models/model_hal",'model_hal_ek_ss5.h5'))

    with open('C:/Users/emreb/Desktop/web_proje/models/model_hal/in_train_tokenizer_hal_ss', 'rb') as handle:
        inn_p = pickle.load(handle)


    with open('C:/Users/emreb/Desktop/web_proje/models/model_hal/out_train_tokenizer_hal_ss', 'rb') as handle:
        out_p = pickle.load(handle)
        
    return model,inn_p,out_p

from typing import List
from jpype import JClass, JString, getDefaultJVMPath, shutdownJVM, startJVM, java
import jpype

ZEMBEREK_PATH = r'C:\Users\emreb\Documents\web_proje\zemberek-full.jar'


if not jpype.isJVMStarted():
    startJVM(getDefaultJVMPath(), '-ea', '-Djava.class.path=%s' % (ZEMBEREK_PATH))
    
TurkishMorphology = JClass('zemberek.morphology.TurkishMorphology')
morphology = TurkishMorphology.createWithDefaults()

def find_analysis(word):
    root = list()  
    analysis: java.util.ArrayList = (
        morphology.analyzeAndDisambiguate(word).bestAnalysis())
    
    for i, analysis in enumerate(analysis, start=0):
     
        root.append( f'{str(analysis)}')
        
        
    return root


def find_root(word):
    root = list()  
    analysis: java.util.ArrayList = (
        morphology.analyzeAndDisambiguate(word).bestAnalysis())
    
    for i, analysis in enumerate(analysis, start=0):
     
        root.append( f'{str(analysis.getLemmas()[0])}')
        
        
    return root[0]