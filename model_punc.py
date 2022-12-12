# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 15:43:06 2022

@author: emreb
"""

import pickle
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch
import numpy as np
from bert_model import BertModel,return_model
#from load_model import  model_punc
from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained("dbmdz/bert-base-turkish-128k-cased")
#model_comm,inn_c,out_c,model_pq,inn_pq ,out_pq =  model_punc()
#model_main , in_train_tokenizer , out_train_tokenizer = model_punc1()
def label_prediction(sample,model,inn,out):
  sample = sample.lower()
  sample_pad = []
  for word in sample.split():
      if word in inn.word_index:
         sample_pad.append(inn.word_index[word])
      else:
         sample_pad.append(1)

  sample_pad = pad_sequences([sample_pad], maxlen=100, padding='post')
  pred = model.predict(sample_pad)
  y_id_to_word = {value: key for key, value in out.word_index.items()}
  y_id_to_word[0] = '<PAD>'
  puncs = [y_id_to_word[np.argmax(x)] for x in pred[0]]
  return puncs

"""
def model_punc_pred(sample):
    punc_comm = label_prediction(sample,model_comm,inn_c,out_c)
    punc_pq = label_prediction(sample,model_pq, inn_pq, out_pq)
    words = sample.split()
    new_sentence = ""
    if len(words) > 1:
        for n,c in enumerate(punc_comm):
            if c == "emp":
                new_sentence = new_sentence + words[n] + " "
            elif c == "com":
                new_sentence = new_sentence + words[n] + ", "
            elif c == "<PAD>":
                break
        new_sentence = new_sentence.strip()
        if  "dot" in punc_pq and not("que" in punc_pq):
            new_sentence = new_sentence + "."
        elif "que" in punc_pq and not("dot" in punc_pq):
            new_sentence = new_sentence + "?"
            
    else:
        for w in words:
            if  "dot" in punc_pq and not("que" in punc_pq):
                new_sentence = w + "."
            elif "que" in punc_pq and not("dot" in punc_pq):
                new_sentence = w + "?"
        
    return new_sentence.strip()

"""
unique_labels = set()
unique_labels = {'QUE', 'COM', 'EMP', 'PER'}
labels_to_ids = {k: v for v, k in enumerate(sorted(unique_labels))}
ids_to_labels = {v: k for v, k in enumerate(sorted(unique_labels))}
label_all_tokens = True

def align_word_ids(texts):
  
    tokenized_inputs = tokenizer(texts, padding='max_length', max_length=200, truncation=True)

    word_ids = tokenized_inputs.word_ids()

    previous_word_idx = None
    label_ids = []

    for word_idx in word_ids:

        if word_idx is None:
            label_ids.append(-100)

        elif word_idx != previous_word_idx:
            try:
                label_ids.append(1)
            except:
                label_ids.append(-100)
        else:
            try:
                label_ids.append(1 if label_all_tokens else -100)
            except:
                label_ids.append(-100)
        previous_word_idx = word_idx

    return label_ids

def decode_text(s,prd_l,s_t):
    pred_label = []
    k = 0
    if len(s.split(" ")) == len(prd_l):
       return prd_l
    else:
        for p,i in enumerate(s_t):
            if i == "[CLS]":
              continue
            elif i == "[SEP]":
               break
            else:
                if i.find("##")>-1:
                   k = k + 1
                else:
                   pred_label.append(prd_l[k])
                   k = k + 1
    return pred_label     

def evaluate_one_text(model, sentence):


    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    text = tokenizer(sentence, padding='max_length', max_length = 200, truncation=True, return_tensors="pt")

    mask = text['attention_mask'].to(device)
    input_id = text['input_ids'].to(device)
    label_ids = torch.Tensor(align_word_ids(sentence)).unsqueeze(0).to(device)

    logits = model(input_id, mask, None)
    logits_clean = logits[0][label_ids != -100]

    predictions = logits_clean.argmax(dim=1).tolist()
    prediction_label = [ids_to_labels[i] for i in predictions]
    pred_label = decode_text(sentence,prediction_label,tokenizer.convert_ids_to_tokens(text["input_ids"][0]))
    #print(sentence)
    #print(prediction_label)
    #print(pred_label)
    #print(tokenizer.convert_ids_to_tokens(text["input_ids"][0]))
    return pred_label

model = return_model()
import re  
def model_punc_pred(sample):


  """
  sample = sample.lower()
  sample_pad = []
  for word in sample.split():
      if word in in_train_tokenizer.word_index:
         sample_pad.append(in_train_tokenizer.word_index[word])
      else:
         sample_pad.append(1)

  sample_pad = pad_sequences([sample_pad], maxlen=34, padding='post')
  pred = model_main.predict(sample_pad)
  y_id_to_word = {value: key for key, value in out_train_tokenizer.word_index.items()}
  y_id_to_word[0] = '<PAD>'
  """
  sample = re.sub(r'[^\w\s]', '', sample)
  puncs = evaluate_one_text(model, sample)
  new_sentence = ""
  words = sample.split()
  words = [i for i in words if i]
  for n,word in enumerate(words):
     if puncs[n] == "EMP":
        new_sentence = new_sentence + word + " "
     elif puncs[n] == "COM":
        new_sentence = new_sentence + word + ", "
     elif puncs[n] == "PER" and n == (len(words)-1):
        new_sentence = new_sentence + word + ".  "    
     elif puncs[n] == "QUE" and n == (len(words)-1):
        new_sentence = new_sentence + word + "?  "
      
     else:
        new_sentence = new_sentence + word + " "
  
  return new_sentence.strip()
