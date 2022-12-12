# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 14:35:26 2022

@author: emreb
"""
import re
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from load_model import model_de_da

model_main,Word2Vec_model= model_de_da()



idx2Label = {0 :'PAD', 1:'UNK' , 2: 'o' , 3: 'e'}


def replace_str_index(text,index=0,replacement=''):
    return '%s%s%s'%(text[:index],replacement,text[index+1:])
# word tokenizer for de da suffix. Such as, ben de okula gittim -- [ben de , okula , gittim]
def word_tokenizer_de_da(line):
        line = re.sub(r'[^\w\s]', '', line)
        words = line.split()
        r_words = []
        try:
            for pos,word in enumerate(words):
                n = ""
                if (word.find("dede") > 0) or (word.find("dada") > 0):
                    if word.find("dede") > 0:
                        r_words.append(word.split("dede")[0])
                        r_words.append("dede")
                    elif word.find("dada") > 0:
                        r_words.append(word.split("dada")[0])
                        r_words.append("dada")
                        
                elif pos > 0:
                   if word == "de":
                      before_word = words[pos-1]
                      n = before_word + " de"
                      r_words.remove(before_word)
                      r_words.append(n)
                   elif word == "da":
                        before_word = words[pos-1]
                        n = before_word + " da"
                        r_words.remove(before_word)
                        r_words.append(n)
                           
                   else:
                       r_words.append(word)
                else:
                    r_words.append(word)
        except:
            pass
                
        return r_words
#control for word tokenizer de da suffix
def lastControl(pred_sentence):
    word_list = []
    r_words = word_tokenizer_de_da(pred_sentence)
    sentence = ""
    for pos,word in enumerate(r_words):
        if word == "dede" or word == "dada":
          if pos > 0:
              if word == "dede":
                before_word = r_words[pos-1]
                new_word = before_word + "de" + " de"
                word_list.append(new_word)
                word_list.remove(before_word)
              elif word == "dada":
                  before_word = r_words[pos-1]
                  new_word = before_word + "da" + " da"
                  word_list.append(new_word)
                  word_list.remove(before_word)
              else:
                word_list.append(word)
          else:
              word_list.append(word)
        else:
          word_list.append(word)
    for word in word_list:
        sentence = sentence + word + " "
    return sentence.lower()

def word_tokenizer(line):
        line = re.sub(r'[^\w\s]', '', line)
        words = line.split()
        r_words = []
        try:
            for pos,word in enumerate(words):
                n = ""
                if pos > 0:
                   if word == "de":
                      before_word = words[pos-1]
                      n = before_word + " de"
                      r_words.remove(before_word)
                      r_words.append(n)
                   elif word == "da":
                        before_word = words[pos-1]
                        n = before_word + " da"
                        r_words.remove(before_word)
                        r_words.append(n)
                   elif word == "ki":
                      before_word = words[pos-1]
                      n = before_word + " ki"
                      r_words.remove(before_word)
                      r_words.append(n)
                   else:
                       r_words.append(word)
                else:
                    r_words.append(word)
        except:
            pass
                
        return r_words  
# prediction and correction
def model_de_da_prediction(sample):
  sample_pad = []
  words = word_tokenizer(sample.lower())
  word2index = {token: token_index for token_index, token in enumerate(Word2Vec_model.wv.index2word)}
  for word in words:
      if word in Word2Vec_model.wv.index2word:
          
         sample_pad.append(word2index[word])
      else:
         sample_pad.append(word2index["UNK"])

  sample_pad = pad_sequences([sample_pad], maxlen=34, padding='post')

  pred = model_main.predict(sample_pad)
  label = [idx2Label[np.argmax(x)] for x in pred[0]]
  output_label = ""
  for i in label:
      output_label = output_label + i + " "
  

  
  pred_sent = ""
  for i in zip(words,label):
    if i[1] == "o":
      pred_sent = pred_sent +  i[0] + " "
    elif i[1] == "e":
        if i[0].find(" de") > 0:
            new_word = replace_str_index(i[0],index=i[0].find(" de"),replacement='')
            pred_sent = pred_sent + new_word + " "
        elif i[0].find(" da") > 0:
            new_word = replace_str_index(i[0],index=i[0].find(" da"),replacement='')
            pred_sent = pred_sent + new_word + " "
        elif i[0].find(" ") == -1 and i[0].find("da") == len(i[0]) - 2:
            new_word = replace_str_index(i[0],i[0].find("da"),replacement=' d')
            pred_sent = pred_sent + new_word + " "
        elif i[0].find(" ") == -1 and i[0].find("de") == len(i[0]) - 2:
            new_word = replace_str_index(i[0],i[0].find("de"),replacement=' d')
            pred_sent = pred_sent + new_word + " "
        else:
            pred_sent = pred_sent + i[0] + " "
            
    elif i[0][1] == "PAD":
        continue
  pred_sent = pred_sent.strip()
  pred_sent = lastControl(pred_sent)


  return pred_sent

    
def predict_de(sample):
    sample_pad = []
    words = word_tokenizer(sample.lower())
    word2index = {token: token_index for token_index, token in enumerate(Word2Vec_model.wv.index2word)}
    for word in words:
      if word in Word2Vec_model.wv.index2word:
          
         sample_pad.append(word2index[word])
      else:
         sample_pad.append(word2index["UNK"])

    sample_pad = pad_sequences([sample_pad], maxlen=34, padding='post')

    pred = model_main.predict(sample_pad)
    label = [idx2Label[np.argmax(x)] for x in pred[0]]
    output_label = ""
    for i in label:
        output_label = output_label + i + " "

    pred_sent = ""
    
    for i in zip(words,label):
      if i[1] == "o":
        pred_sent = pred_sent + "SSS "
      elif i[1] == "e":
        if i[0].find(" de") > 0:
            new_word = replace_str_index(i[0],index=i[0].find(" de"),replacement='')
            pred_sent = pred_sent + new_word + " "
        elif i[0].find(" da") > 0:
            new_word = replace_str_index(i[0],index=i[0].find(" da"),replacement='')
            pred_sent = pred_sent + new_word + " "
        elif i[0].find(" ") == -1 and i[0].find("da") == len(i[0]) - 2:
            new_word = replace_str_index(i[0],i[0].find("da"),replacement=' d')
            pred_sent = pred_sent + new_word + " "
        elif i[0].find(" ") == -1 and i[0].find("de") == len(i[0]) - 2:
            new_word = replace_str_index(i[0],i[0].find("de"),replacement=' d')
            pred_sent = pred_sent + new_word + " "
        else:
            pred_sent = pred_sent + i[0] + " "   
      elif i[0][1] == "PAD":
          continue
    pred_sent = pred_sent.strip()
    
    return pred_sent    
    
    
    
    
    
    
    
    
    