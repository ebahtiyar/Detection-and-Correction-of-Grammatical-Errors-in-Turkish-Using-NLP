# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 20:54:33 2021
There are some useful functions for data mining and data preprocess

filters='\'!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
filters_included='\"’,.:;?!…'
@author: emreb
"""
import re
import random
import numpy as np

def preprocess(text):

    
    text = re.sub(","," , ",text)
    
    text = text.replace("."," . ")
    


    
    text = re.sub('\"',' \" ',text)
    text = re.sub('\"',' \" ' ,text)
    
    text = text.replace("?", " ? ")
    text = text.replace("!", " ! ")
    text = text.replace("’", " ’ ")
    
    text = re.sub(':',' : ' ,text)
    
    text = re.sub(";" , " ; ",text)
    text = re.sub("…"," … ",text)
    
    text = re.sub("#$&()*\+'¨´é€£½-—–/<=>@[\\]]^_{|}~","",text)
    
    return text

def find_loc(ch,sentence):
    loc = list()
    for pos,char in enumerate(sentence):
        if(char == ch):
           loc.append(pos)
    return loc


def filter_sentence(sentence):
    adding = False
    sentences = list()

    if len(sentence) > 20:
      #Fixes  
      sentence = sentence.replace("SORU:","")
      
      #For bracket
      bracket1 = "("
      bracket2 = ")"
      bracket1_loc = list()
      bracket2_loc = list()
      
      if sentence.count(bracket1) > 0 and sentence.count(bracket2) >0:
        for pos,char in enumerate(sentence):
          if(char == bracket1):
             bracket1_loc.append(pos)
      
        for pos,char in enumerate(sentence):
          if(char == bracket2):
             bracket2_loc.append(pos)
             
        if len(bracket1_loc) == len(bracket2_loc):
          for i in range(0,len(bracket1_loc)):
              if sentence.find("),") == -1:
                 diff = bracket2_loc[i]-bracket1_loc[i] + 1
                 sentence = sentence[:bracket1_loc[i]] + " " * diff+ sentence[bracket2_loc[i]+1:]
       
        sentence = " ".join(sentence.split())
        
      #For single quotes
      n_single_quotes_loc = list()
      single_quotes = "\'"

      
      for pos,char in enumerate(sentence):
          if(char == single_quotes):
             n_single_quotes_loc.append(pos)


      if sentence.count("'") !=0 and not(n_single_quotes_loc.count(0) > 0):
         for i in n_single_quotes_loc:
            if i > 2:
               try:
                  suffix_quotes = not((sentence[i-1]).isspace()) and not((sentence[i+1]).isspace())
               except:
                   suffix_quotes = False
               if suffix_quotes:
                  sentence = sentence[:i] + "’" + sentence[i+1:]
               else:
                  sentence = sentence[:i] + "\"" + sentence[i+1:]
         
            else:
              continue
      
      #For apostrophe
      r_apostrophe = "‘"
      l_apostrophe = "’"
      l_apostrophe_loc = list()
      if sentence.count(r_apostrophe) > 0 and sentence.find(r_apostrophe) > 0:
         sentence = sentence.replace(r_apostrophe,"\"")
    
         for pos,char in enumerate(sentence):
             if(char == l_apostrophe ):
                l_apostrophe_loc.append(pos)
             
         for i in l_apostrophe_loc:
             try:
                suffix_apostrophe = not((sentence[i-1]).isspace()) and not((sentence[i+1]).isspace())
             except:
                suffix_apostrophe = False
       
             if suffix_apostrophe == False:
                sentence = sentence[:i] + "\"" + sentence[i+1:]
             else:
                sentence = sentence[:i] + "’" + sentence[i+1:]

          
      if (sentence.count("“") !=0 and sentence.count("”") !=0):
          sentence = sentence.replace("“","\"")
          sentence = sentence.replace("”","\"")    

    
      if len(sentence) > 0:
          if sentence[0] == " ":
             sentence = sentence.replace(" ","",1)
    
      sentence = sentence.replace("'","’")
      sentence = sentence.replace("...","…")  
      sentence = sentence.replace("`","’")
    #Conditions
      filter_c_c = True
      filters_c = 'ABCÇDEFGĞHIİJKLMNOÖPQRSŞTUÜVWXYZabcçdefgğhıi̇jklmnoöpqrsştuüvwxyz"’,.:;?!…0123456789âÂ '
      for i in sentence:
         if i not in filters_c:
            filter_c_c = False
            break
    
    
    
      website = True
      if sentence.find(".com") != -1 or sentence.find("www") != -1 or sentence.find("http") != -1:
          website = False
    
    
      length = len(sentence) > 20 and len(sentence) < 500
      if length:
         numeric = not(sentence[len(sentence)-2].isnumeric())
         title1 = sentence[sentence.find(" ")+1].islower()
         title = not(sentence[1].isupper()) and not(sentence[len(sentence)-2].isupper())
         condition = not(sentence[0].islower()) and (sentence[0].isalpha() or sentence[0].isnumeric())
      else:
          numeric = False
          title = False
          title1 = True
          condition = False
     
      filter_punctuation = "#$&()*+¨´é€£½-—–/<=>@[\\]]^_{|}~"
      filter_p = True
      for  i in sentence:
          if  i in filter_punctuation:
              filter_p = False
              break

      
      filters_included='\"’.,:;?!…'
      single_punc = list()
      c_single_punc = True
      for pos,char in enumerate(sentence):
         if char in filters_included:
           try:
             if (sentence[pos-1] in filters_included) or (sentence[pos+1] in filters_included):
                 single_punc.append(False)
           except:
               pass
           
      if len(single_punc) > 0:
         c_single_punc = False          
      
      wordList = re.sub("[^\w]", " ",  sentence).split()
      w_control_list = list()
      w_control = True
      for i in wordList:
          for j in i:
             if j.isupper():
                w_control_list.append("True")
            
          if len(w_control_list) > 5:
             w_control = False
             break  
      
        
      row = sentence.find("\n") == -1
       
      #Degree
      degree = True
      matching_degree = list()
      d = ["Doç.","Dr.","Prof.","Yrd.","Dk.","İst.","Av.","Alb.","Gen.","Mah.","Sok.","bkz.","Bkz.","vb.","Vb.","vs.",".tr","hz.","Hz.",".TR","MENÜ","GİRİŞ","ANA SAYFA "]
      for i in d:
          dd = ((sentence.find(i) == len(sentence) - len(i)))
          matching_degree.append((dd))
      if matching_degree.count(True) > 0:
          degree = False 
        
      
      quotes = (sentence.count('\"') % 2 == 0)
      qoutes1 = not((sentence.count('\'\'') == 1))
      qoutes2 = ((sentence.count('“') == 1) and (sentence.count('”') == 1)) or((sentence.count('“') == 0) and (sentence.count('”') == 0))
      
      if(length and numeric and row and degree and quotes and qoutes1 and qoutes2 and title and condition and website and title1 and filter_p  and c_single_punc and w_control and filter_c_c):
    
         if(sentence.find("…") == -1):
               if((sentence[len(sentence)-1] == ".") or (sentence[len(sentence)-1] == "!") 
                    or (sentence[len(sentence)-1] == "?") or (sentence[len(sentence)-1] == ":") or (sentence[len(sentence)-1] == "…")):
                    adding = True
                    sentences.append(sentence)
         elif((sentence.find("…") < len(sentence) - 1) and sentence.find("…") != -1):
                    adding = False
         elif(sentence.find("…") == len(sentence)-1):
                sentences.append(sentence)
                adding = True
         else:
                adding = False
                
      else:
            adding = False
     
    else:
        adding = False
        
    return adding , sentences 
        
        
    
#filters='\'!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'    
def wrong_punctuation(sentence):
    filters_p = ',.?!'
    matching = list()
    punctuation = list()
    for pos,char in enumerate(sentence):
        if char in filters_p:
            matching.append(pos)
            punctuation.append(char)
    for i in range(0,len(punctuation)):
        filters_p = ',?!.'
        temp_p = filters_p.replace(punctuation[i],"") + " "
        sentence = sentence[:matching[i]] + random.choice(temp_p) + sentence[matching[i] + 1:]
        
        
        
                    
    return sentence 
    
    
def unique(list1):
    x = np.array(list1)
    x = np.unique(x)
    x = x.tolist()
    
    return x
    
#http://www.hurriyet.com.tr/230-milyon-liraya-imza-atti-40034099
def filter_links(link):
    adding = False
    
    if link.find("http://www.hurriyet.com.tr") == 0 and link[len(link)-1].isnumeric():
        adding = True
        
    return adding
    
def tuples_to_list(data):
    sentences = list()
    for i in range(0,len(data)):
        sentences.append(str(data[i][0]))
    return sentences    
    
    
def punctuation_filter(sentence):
    
    filters_included='\"’,.:;?!…'

    dot_loc = find_loc(".", sentence)
    dot_cond = True
 
    for i in dot_loc:
        
        if i != len(sentence) - 1:
           try:
             dot_cond = not (sentence[i-1].isalpha() and sentence[i+1].isalpha())
           except:
               pass
        if dot_cond == False:
             break
    
    single_punc = list()
    c_single_punc = True
    for pos,char in enumerate(sentence):
       if char in filters_included:
           try:
             if (sentence[pos-1] in filters_included) or (sentence[pos+1] in filters_included):
                 single_punc.append(False)
           except:
               pass
           
    if len(single_punc) > 0:
        c_single_punc = False           
         
    sentence = sentence.replace("'","’")
    
    numeric = "0123456789"
    numeric_filters = ".:,"
    c_numeric = True
    numeric_loc = list()
    
    
    for pos,char in enumerate(sentence):
        if char in numeric:
            numeric_loc.append(pos)
        

    for i in numeric_loc:
       try:
          if i == 0:
             if sentence[i+1] in numeric_filters:
                c_numeric = False
                break
            
          if i > 0:
             if (sentence[i+1]) in numeric_filters or (sentence[i-1] in numeric_filters):
                 c_numeric = False
                 break
     
       except:
           pass
    
    n = True
    for i in sentence:
        if i in numeric:
            n = False
            
    
    
    
    
    
    double_quotes = True
    double_quotes_loc = list()
    for pos,char in enumerate(sentence):
         if char == "\"":
            double_quotes_loc.append(pos)
        
        
    for i in double_quotes_loc:
        try:
            quoutes_cond = (sentence[i-1].isalpha() and sentence[i+1].isalpha()) or (sentence[i-1].isalpha() and sentence[i+1].isnumeric()) or (sentence[i-1].isnumeric() and sentence[i+1].isalpha())
        except:
             pass 
        if  quoutes_cond:
            double_quotes = False
            
    adding = False
    if double_quotes and c_numeric and dot_cond and c_single_punc and n:
        adding = True
        
    
    return adding, sentence
    
def filter_words(word):
    

    numeric = "0123456789"
    numeric_c = True
    
    symbol_filters='\'!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n‹√ '
    symbol_c = True
    
    len_word_c  = True
    for i in word:
        if i in numeric:
            numeric_c = False
            break
        if i in symbol_filters:
            symbol_c = False
            
    if len(word) < 2:
        len_word_c = False
        
        
    filters_w = 'ABCÇDEFGĞHIİJKLMNOÖPQRSŞTUÜVWXYZabcçdefgğhiıjklmnoöpqrsştuüvwxyzâÂ'
    letter_c = True
    adding = False
    for i in word:
        if i not in filters_w:
            letter_c = False
            break
    
    
    
    
    
    if numeric_c and symbol_c and len_word_c and letter_c:
        adding = True
        
    return adding , word









    

