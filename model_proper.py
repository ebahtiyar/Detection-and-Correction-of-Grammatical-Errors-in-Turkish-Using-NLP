
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import time
from load_model import model_proper,find_root
time.sleep(1)
model_main,inn_p ,out_p ,controlWord = model_proper()

def label_prediction(sample):
  sample = sample.lower()
  sample_pad = []
  for word in sample.split():
      if word in inn_p.word_index:
         sample_pad.append(inn_p.word_index[word])
      else:
         sample_pad.append(1)

  sample_pad = pad_sequences([sample_pad], maxlen=45, padding='post')
  pred = model_main.predict(sample_pad)
  y_id_to_word = {value: key for key, value in out_p.word_index.items()}
  y_id_to_word[0] = '<PAD>'
  puncs = [y_id_to_word[np.argmax(x)] for x in pred[0]]
  return puncs


def proper_prediction(sample):
    label = label_prediction(sample)
    words = sample.split()

    roots = []
    for word in words:
        roots.append(find_root(word))
    
    
    
    
    
    new = ""
    for pos,l in enumerate(label):
        
        if l == "<PAD>":
            break
        
        elif l == "o":
            new = new + words[pos] + " "
            
        elif l == "d":
            if roots[pos] in controlWord:
                new = new + words[pos] + " "
            else:
                new = new + words[pos].capitalize() + " "
            
    return (new[0].upper() + new[1:]).strip()
            
            
        
        
s1 = "bugün bir bey beni aradı"
s2 = "bugün emre bey beni aradı"

s3 = "bugün onur bize geldi"
s4 = "öğretmen onur ne demek diye sordu"

s5 = "bu sene türk dili ve edebiyatı dersini alacağım"
s6 = "proje ödevim türk dili hakkında"

s7 = "bu kelime türk dil kurumuna göre hatalı"

s8 = "hep bahtiyar ol"
     
        
        
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    