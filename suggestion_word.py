# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 19:29:05 2022

@author: emreb
"""

import sqlite_functions as sql
import distance_algorithm as dist
from load_model import find_root
words_base = sql.takeSentence1("kelimeler.db", "roots_pos_flag")
import re
import string

def correction_sey(sentence):
    
    words = sentence.split()
    new_word = []
    for word in words:
        root = find_root(word)
        control = root.find("şey") > 0
        
        if control:
            control1 = word.find("şey") > 1
            if control1:
                index = word.find("şey")
                word = word[:index] + " " + word[index:]
                new_word.append(word)
            else:
                new_word.append(word)
        else:
            new_word.append(word)
    return " ".join(new_word)  


def find_word(word):
    
    suggestion_word = "None"
    for i in range(0,len(words_base)):
        if word == words_base[i][0]:
            suggestion_word = words_base[i][2]
            break
        
    return suggestion_word


def word_tokenizer(line):
        r_words = []
        punc = []
        for l in line.split():
            punc.append(l[len(l)-1])
        line = re.sub(r'[^\w\s]', '', line)
        words = line.split()
        try:
            for pos,word in enumerate(words):
                n = ""  
                if pos > -1:
                   if word == "de":
                      before_word = words[pos-1]
                      n = before_word + " de"
                      if punc[pos] in string.punctuation:
                          n = n + punc[pos]
                      r_words.remove(before_word)
                      r_words.append(n)
                   elif word == "da":
                        before_word = words[pos-1]
                        n = before_word + " da"
                        if punc[pos] in string.punctuation:
                          n = n + punc[pos]
                        r_words.remove(before_word)
                        r_words.append(n)
                   elif word == "ki":
                      before_word = words[pos-1]
                      n = before_word + " ki"
                      if punc[pos] in string.punctuation:
                          n = n + punc[pos]
                      r_words.remove(before_word)
                      r_words.append(n)
                   else:
                    if punc[pos] in string.punctuation:
                         r_words.append(word + punc[pos])
                    else:
                        r_words.append(word)
                else:
                    r_words.append(word)
        except:
            pass
                
        return r_words
            
def suggestion_e(sentence,correction):
    sentence =  correction_sey(sentence)   
    words = sentence.split()
    new_sentence = ""
    if correction:
        for word in words:
            root = find_root(word)
            if root != "UNK":
                suggestion_word = find_word(root)
                if (suggestion_word  == "X") or (suggestion_word == "Turkish"):
                    new_sentence = new_sentence + word + " "
                elif suggestion_word == "None":
                    possible_word = dist.correction(word)
                    print("\""+word+"\""+" kelimesi yanlış yazılmıştır.Bunu mu demek istediniz? " + "\"" + possible_word[1][0] + "\"")
                    print("\n")
                    new_sentence = new_sentence + word  + " "
                else:
                   print("\""+ word + "\""+ " kelimesi yabancı kökenlidir.")
                   print("Bu kelime yerine " + "\""+ suggestion_word +"\"" + " kelimesi/kelimeleri kullanabilirsiniz.")
                   print("\n")
                   new_sentence = new_sentence + word + " "
            else:
                possible_word = dist.correction(word)
                new_sentence = new_sentence + possible_word[1][0] + " "
                    
        
        
    else:
        for word in words:
            root = find_root(word)
            if root!= "UNK":
                suggestion_word = find_word(root)
                if (suggestion_word == "X") or suggestion_word == "Turkish":
                    continue
                elif suggestion_word == "None":
                    possible_word = dist.correction(word)
                    print("\""+word+"\""+" kelimesi yanlış yazılmıştır.Bunu mu demek istediniz? " )
                    for i in possible_word[1]:
                        print(i)
                    print("\n")
                else:
                   print("\""+ word + "\""+ " kelimesi yabancı kökenlidir.")
                   print("Bu kelime yerine " + "\""+ suggestion_word +"\"" + " kelimesi/kelimeleri kullanabilirsiniz.")
                   print("\n")
                
            else:
                possible_word = dist.correction(word)
                print("\""+word+"\""+" kelimesi yanlış yazılmıştır.Bunu mu demek istediniz? " )
                for i in possible_word[1]:
                    print(i)
                print("\n")
    
    return new_sentence.strip()


import numpy as np
def suggestion_word_web(sentence):
    s_list = []
    turksih_root = []
    words = word_tokenizer(sentence)
    p_w = []
    pos = []
    for p,word in enumerate(words):
        root = find_root(word)
        possible_word = []
        if root != "UNK":
            suggestion_word = find_word(root)
            
            if (suggestion_word == "X") or suggestion_word == "Turkish":
                possible_word.append(word)
                possible_word.append(word)
                possible_word.append(word)
                p_w.append(possible_word)
                
            
            elif suggestion_word == "None":
                possible_word  = dist.correction(word)[1]
                if len(possible_word) < 3:
                    for i in range(0,3-len(possible_word)):
                        possible_word.append(word)
                p_w.append(possible_word)
                
                pos.append(p)
            else:
                possible_word.append(word)
                possible_word.append(word)
                possible_word.append(word)
                sugg = "\""+ word + "\""+ " kelimesi yabancı kökenlidir. Bu kelime yerine " + "\""+ suggestion_word +"\"" + " kelimesi/kelimeleri kullanabilirsiniz."
                turksih_root.append(sugg)
                p_w.append(possible_word)
        else:
            possible_word = dist.correction(word)[1]
            if len(possible_word) < 3:
                for i in range(0,3-len(possible_word)):
                    possible_word.append(word)
                
                
            p_w.append(possible_word)
            pos.append(p)
    print(p_w)
    s = len(p_w)
    p_w = np.array(p_w)
    p_w = p_w.T
    p_w = p_w.tolist()        
    for p in p_w:
        s = ""
        for i in p:
            s = s + i + " "
        s_list.append(s.strip())
        

    if len(s_list) < 3:
        for i in range(0,3-len(s_list)):
            s_list.append("")   
        
    return s_list,turksih_root

    