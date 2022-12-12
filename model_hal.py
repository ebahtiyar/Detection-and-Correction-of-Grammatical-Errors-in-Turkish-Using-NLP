# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 12:29:10 2022

@author: emreb
"""

import string

hal_ekleri = ["Dat","Acc","Abl","Loc"]
zamir = ["ben","sen"]
#from tensorflow.keras.preprocessing.sequence import pad_sequences
#import numpy as np
from load_model import find_analysis,find_root
pass_word = ["şehir","kahır","emir","fikir","keyif","şekil","kalp","hal","saat"]

#model,inn_p,out_p = model_hal()

from transformers import BertForTokenClassification,BertTokenizerFast
import torch
unique_labels = {'_E', '_I', '_DE', '_DEN', '_S'}
labels_to_ids = {k: v for v, k in enumerate(sorted(unique_labels))}
ids_to_labels = {v: k for v, k in enumerate(sorted(unique_labels))}
label_all_tokens = True
class BertModel(torch.nn.Module):

    def __init__(self):

        super(BertModel, self).__init__()

        self.bert = BertForTokenClassification.from_pretrained('dbmdz/bert-base-turkish-128k-cased', num_labels=len(unique_labels))

    def forward(self, input_id, mask, label):

        output = self.bert(input_ids=input_id, attention_mask=mask, labels=label, return_dict=False)

        return output
model = BertModel()
model = torch.load(r'C:\Users\emreb\Documents\web_proje\models\model_hal\bert_hal4',map_location=torch.device('cpu'))
tokenizer = BertTokenizerFast.from_pretrained("dbmdz/bert-base-turkish-128k-cased")

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



    
"""    
def prediction_label(sample):
  sample_pad = []
  for word in sample.split():
      if word in inn_p.word_index:
         sample_pad.append(inn_p.word_index[word])
      else:
         sample_pad.append(1)

  sample_pad = pad_sequences([sample_pad], maxlen=22, padding='post')
  pred = model.predict(sample_pad)
  y_id_to_word = {value: key for key, value in out_p.word_index.items()}
  y_id_to_word[0] = '<PAD>'
  puncs = [y_id_to_word[np.argmax(x)] for x in pred[0] if y_id_to_word[np.argmax(x)] != "<PAD>" ]
  return puncs
"""
def labelling(sentence):
    #Abl-den Loc-de Acc-i Dat-a
    sentence =  sentence.translate(str.maketrans('', '', string.punctuation))
    sentence = sentence.replace("…","")
    analysis = find_analysis(sentence)
    label = ""
    for i in analysis:
        typ = i.split("]")[0].split(":")[1]
        if typ.find("Verb") == -1:
            ek = i.split("]")[1].split("+")
            if ek[len(ek)-1].find(":") != -1:
                if ek[len(ek)-1].split(":")[1] in hal_ekleri:
                    hal = ek[len(ek)-1].split(":")[1]
                    if hal == "Abl":
                        label = label + "_den" + " "
                    elif hal == "Loc":
                        label = label + "_de" + " "
                    elif hal == "Dat":
                        label = label + "_e" + " "
                    elif hal == "Acc":
                         label = label + "_i" + " "
                else:
                    label = label + "_s" + " "
            else:
                label = label + "_s" + " "
        else:
            label = label + "_s" + " "
            
    return label.strip()

def train_sentences1(sentence):
    sentence =  sentence.translate(str.maketrans('', '', string.punctuation))
    train_sentence = ""
    sentence = sentence.replace("’","'")
    
    labels = labelling(sentence)
    words = sentence.split()
    sert_ünsüzler = ["ç","k","p","t"]
    zamir = ["sen","ben"]
    for pos,label in enumerate(labels.split()):
        cozumle1 = find_analysis(words[pos])
        m_root = find_root(words[pos])
        typ = cozumle1[0].split("]")[0].split(":")[1]
        ek = cozumle1[0].split("]")[1].count("+")
        if label != "_s" and m_root not in pass_word:
            if label == "_den":
                train_sentence = train_sentence +  words[pos][:len(words[pos])-3] + " "
            elif label == "_de":
                train_sentence = train_sentence +  words[pos][:len(words[pos])-2] + " "
            elif label == "_e":
                if m_root in zamir and ek == 2:
                    train_sentence = train_sentence +  m_root + " "
                elif words[pos][len(words[pos])-2] == "y":
                   train_sentence = train_sentence +  words[pos][:len(words[pos])-2] + " "
                elif (m_root[len(m_root)-1] in sert_ünsüzler) and words[pos].find(m_root) == -1 and ek==2:
                    index = sert_ünsüzler.index(m_root[len(m_root)-1])
                    if index == 0:
                       w =  words[pos][:len(words[pos])-1]
                       w = 'ç'.join(w.rsplit('c', 1))
                       train_sentence = train_sentence +  w + " "
                       
                    elif index == 2:
                        w =  words[pos][:len(words[pos])-1]
                        w ='p'.join(w.rsplit('b', 1))
                        train_sentence = train_sentence +  w + " "
                    elif index == 3:
                        w =  words[pos][:len(words[pos])-1]
                        w ='t'.join(w.rsplit('d', 1))
                        train_sentence = train_sentence +  w + " "
                    elif index == 1:
                         if words[pos].find("ğ") > 0:
                             w =  words[pos][:len(words[pos])-1]
                             w ='k'.join(w.rsplit('ğ', 1))
                             train_sentence = train_sentence +  w + " "
                         else:
                             w =  words[pos][:len(words[pos])-1]
                             w ='k'.join(w.rsplit('g', 1))
                             train_sentence = train_sentence +  w + " "            
                else:
                   train_sentence = train_sentence +  words[pos][:len(words[pos])-1] + " "
            elif label == "_i":
                 if words[pos][len(words[pos])-2] == "y":
                    train_sentence = train_sentence +  words[pos][:len(words[pos])-2] + " "
                 elif (m_root[len(m_root)-1] in sert_ünsüzler) and words[pos].find(m_root) == -1 and ek==2:
                     index = sert_ünsüzler.index(m_root[len(m_root)-1])
                     if index == 0:
                        w =  words[pos][:len(words[pos])-1]
                        w = 'ç'.join(w.rsplit('c', 1))
                        train_sentence = train_sentence +  w + " "
                        
                     elif index == 2:
                         w =  words[pos][:len(words[pos])-1]
                         w ='p'.join(w.rsplit('b', 1))
                         train_sentence = train_sentence +  w + " "
                     elif index == 3:
                         w =  words[pos][:len(words[pos])-1]
                         w ='t'.join(w.rsplit('d', 1))
                         train_sentence = train_sentence +  w + " "
                     elif index == 1:
                          if words[pos].find("ğ") > 0:
                              w =  words[pos][:len(words[pos])-1]
                              w ='k'.join(w.rsplit('ğ', 1))
                              train_sentence = train_sentence +  w + " "
                          else:
                              w =  words[pos][:len(words[pos])-1]
                              w ='k'.join(w.rsplit('g', 1))
                              train_sentence = train_sentence +  w + " "            
                 else:
                    train_sentence = train_sentence +  words[pos][:len(words[pos])-1] + " "
                
        else:
            
            if typ.find("Verb") > -1:
                train_sentence = train_sentence + m_root + " "
            else:            
                train_sentence = train_sentence + words[pos] + " "
    train_sentence = train_sentence.replace("'","")
    return train_sentence.strip().lower(),labels  



ünlü_harfler = ["a","e","i","ı","o","ö","u","ü"]        
def findÜnlü(word):
    for i in range(len(word)-1,-1,-1):
        if word[i] in ünlü_harfler:
            return word[i]


           
def correction(org_sentence):
    sentence,_ = train_sentences1(org_sentence)
    labels = evaluate_one_text(model, sentence)
    label = labels
    words = sentence.split()
    org_words = org_sentence.split()
    correct_sent = ""
    sert_ünsüzler = ["p","ç","t","f","s","ş","h","k"]
    kalın_ünlü = ["a","ı","o","u"]
    yumusama = ["ç","k","p","t"]
    if len(label) == len(words):
        w_l = zip(words,label)
        for pos,(w,l) in enumerate(w_l):
            root = find_root(w)
            cozumle1 = find_analysis(w)
            ek = cozumle1[0].split("]")[1].count("+")
            if root not in pass_word:
                if l == "_DEN":
                    if w[len(w)-1] in sert_ünsüzler:
                        if findÜnlü(w) in kalın_ünlü:
                            correct_sent = correct_sent + w  +  "tan" + " "
                        else:
                            correct_sent = correct_sent + w + "ten" + " "
                    else:
                        if findÜnlü(w) in kalın_ünlü:
                            correct_sent = correct_sent + w +"dan" + " "
                        else:
                            correct_sent = correct_sent + w + "den" + " "
                            
                elif l == "_DE":
                     if w[len(w)-1] in sert_ünsüzler:
                         if findÜnlü(w) in kalın_ünlü:
                             correct_sent = correct_sent + w  +  "ta" + " "
                         else:
                             correct_sent = correct_sent + w + "te" + " "
                     else:
                         if findÜnlü(w) in kalın_ünlü:
                             correct_sent = correct_sent + w +"da" + " "
                         else:
                             correct_sent = correct_sent + w + "de" + " "
                             
                elif l == "_I":
                     xx = findÜnlü(w)
                     if xx in kalın_ünlü:
                        if xx == "u" or xx == "o":
                            g_e = "u"
                        else:
                            g_e = "ı"
                        
                     else:
                        if xx == "ü" or xx == "ö":
                            g_e = "ü"
                        else:
                            g_e = "i"
                    
                     if w[len(w) - 1] not in yumusama:
                        if root not in ünlü_harfler:
                            correct_sent = correct_sent + w + g_e + " "
                        elif root in ünlü_harfler and len(ek) == 2:
                            correct_sent = correct_sent + w + "y" + g_e + " "
                        else:
                            correct_sent = correct_sent + w + g_e + " "

                                
                     else:
                         index = yumusama.index(w[len(w)-1])
                         if index == 0:
                            if findÜnlü(w) in kalın_ünlü:
                                kk =  'c'.join(w.rsplit('ç', 1)) + g_e
                                if find_root(kk) != "UNK":
                                    correct_sent = correct_sent + kk + " "
                                else:
                                    correct_sent = correct_sent + w + g_e + " "
                                    
                            else:
                                kk =  'c'.join(w.rsplit('ç', 1)) + g_e
                                if find_root(kk) != "UNK":
                                    correct_sent = correct_sent + kk + " "
                                else:
                                    correct_sent = correct_sent + w + g_e + " "
                         elif index == 3:
                            if findÜnlü(w) in kalın_ünlü:
                                kk =  'd'.join(w.rsplit('t', 1)) + g_e
                                if find_root(kk) != "UNK":
                                    correct_sent = correct_sent + kk + " "
                                else:
                                    correct_sent = correct_sent + w + g_e + " "
                                    
                            else:
                                kk =  'd'.join(w.rsplit('t', 1)) + g_e
                                if find_root(kk) != "UNK":
                                    correct_sent = correct_sent + kk + " "
                                else:
                                    correct_sent = correct_sent + w + g_e + " "
                         
                            
                         elif index == 2:
                            if findÜnlü(w) in kalın_ünlü:
                                kk =  'b'.join(w.rsplit('p', 1)) + g_e
                                if find_root(kk) != "UNK":
                                    correct_sent = correct_sent + kk + " "
                                else:
                                    correct_sent = correct_sent + w + g_e + " "
                                    
                            else:
                                kk =  'b'.join(w.rsplit('p', 1)) + g_e
                                if find_root(kk) != "UNK":
                                    correct_sent = correct_sent + kk + " "
                                else:
                                    correct_sent = correct_sent + w + g_e + " "
                         
                         
                         
                         
                         elif index == 1:
                            if findÜnlü(w) in kalın_ünlü:
                                    
                                kk =  'g'.join(w.rsplit('k', 1)) + g_e
                                kk1 = 'ğ'.join(w.rsplit('k', 1)) + g_e
                                if find_root(kk1) != "UNK":
                                    correct_sent = correct_sent + kk1 + " "
                                elif find_root(kk) != "UNK":
                                    correct_sent = correct_sent + kk + " "
                                else:
                                    correct_sent = correct_sent + w + g_e + " "
                                    
                            else:
                                kk =  'g'.join(w.rsplit('k', 1)) + g_e
                                kk1 = 'ğ'.join(w.rsplit('k', 1)) + g_e
                                if find_root(kk1) != "UNK":
                                    correct_sent = correct_sent + kk1 + " "
                                elif find_root(kk) != "UNK":
                                    correct_sent = correct_sent + kk + " "
                                else:
                                    correct_sent = correct_sent + w + g_e + " "
            
                elif l == "_E":
                     m_root = find_root(w)
                     if m_root not in zamir:

                            xx = findÜnlü(w)
                            if xx in kalın_ünlü:
                                g_e = "a"
                            else:
                                g_e = "e"
                                
                                
                            if w[len(w) - 1] not in yumusama:
                               if w[len(w) - 1] not in ünlü_harfler:
                                   correct_sent = correct_sent + w + g_e + " "
                               else:
                                   correct_sent = correct_sent + w + "y" + g_e + " "
                   
                            else:
                                index = yumusama.index(w[len(w)-1])
                                if index == 0:
                                   if findÜnlü(w) in kalın_ünlü:
                                       kk =  'c'.join(w.rsplit('ç', 1)) + g_e
                                       if find_root(kk) != "UNK":
                                           correct_sent = correct_sent + kk + " "
                                       else:
                                           correct_sent = correct_sent + w + g_e + " "
                                           
                                   else:
                                       kk =  'c'.join(w.rsplit('ç', 1)) + g_e
                                       if find_root(kk) != "UNK":
                                           correct_sent = correct_sent + kk + " "
                                       else:
                                           correct_sent = correct_sent + w + g_e + " "
                                elif index == 3:
                                  if findÜnlü(w) in kalın_ünlü:
                                      kk =  'd'.join(w.rsplit('t', 1)) + g_e
                                      if find_root(kk) != "UNK":
                                          correct_sent = correct_sent + kk + " "
                                      else:
                                          correct_sent = correct_sent + w + g_e + " "
                                          
                                  else:
                                      kk =  'd'.join(w.rsplit('t', 1)) + g_e
                                      if find_root(kk) != "UNK":
                                          correct_sent = correct_sent + kk + " "
                                      else:
                                          correct_sent = correct_sent + w + g_e + " "            
                                elif index == 2:
                                  if findÜnlü(w) in kalın_ünlü:
                                      kk =  'b'.join(w.rsplit('p', 1)) + g_e
                                      if find_root(kk) != "UNK":
                                          correct_sent = correct_sent + kk + " "
                                      else:
                                          correct_sent = correct_sent + w + g_e + " "
                                          
                                  else:
                                      kk =  'b'.join(w.rsplit('p', 1)) + g_e
                                      if find_root(kk) != "UNK":
                                          correct_sent = correct_sent + kk + " "
                                      else:
                                          correct_sent = correct_sent + w + g_e + " "   
                                          
                                elif index == 2:
                                  if findÜnlü(w) in kalın_ünlü:
                                      kk =  'b'.join(w.rsplit('p', 1)) + g_e
                                      if find_root(kk) != "UNK":
                                          correct_sent = correct_sent + kk + " "
                                      else:
                                          correct_sent = correct_sent + w + g_e + " "
                                          
                                  else:
                                      kk =  'b'.join(w.rsplit('p', 1)) + g_e
                                      if find_root(kk) != "UNK":
                                          correct_sent = correct_sent + kk + " "
                                      else:
                                          correct_sent = correct_sent + w + g_e + " "
                                          
                                elif index == 1:
                                  if findÜnlü(w) in kalın_ünlü:
                                          
                                      kk =  'g'.join(w.rsplit('k', 1)) + g_e
                                      kk1 = 'ğ'.join(w.rsplit('k', 1)) + g_e
                                      if find_root(kk1) != "UNK":
                                          correct_sent = correct_sent + kk1 + " "
                                      elif find_root(kk) != "UNK":
                                          correct_sent = correct_sent + kk + " "
                                      else:
                                          correct_sent = correct_sent + w + g_e + " "
                                          
                                  else:
                                      kk =  'g'.join(w.rsplit('k', 1)) + g_e
                                      kk1 = 'ğ'.join(w.rsplit('k', 1)) + g_e
                                      if find_root(kk1) != "UNK":
                                          correct_sent = correct_sent + kk1 + " "
                                      elif find_root(kk) != "UNK":
                                          correct_sent = correct_sent + kk + " "
                                      else:
                                          correct_sent = correct_sent + w + g_e + " "
                     else:
                       if m_root == "sen":
                          correct_sent = correct_sent + "sana" + " "
                       else:
                          correct_sent = correct_sent + "bana" + " "
                                   
                elif l == "_S":
                    correct_sent = correct_sent + org_words[pos] + " "
                                   
            
            
            else:
                correct_sent = correct_sent + org_words[pos] + " "
    else:
         correct_sent = correct_sent + org_words[pos] + " "
    
                
                
                
    return correct_sent.strip()