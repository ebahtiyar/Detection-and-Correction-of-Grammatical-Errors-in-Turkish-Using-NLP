  # -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 15:53:28 2022

@author: emreb
"""
import time
import suggestion_word as sugg
print("Active suggestion_word")
time.sleep(2)
import model_de_da as de_da
print("Active model_de_da")
time.sleep(2)
import model_ki as ki
print("Active model_ki")
time.sleep(2)
import model_punc as punc
print("Active model_punc")
time.sleep(2)
import model_hal as hal
print("Active model_hal")
time.sleep(2)
#import model_proper as proper
print("Active model_hal")
time.sleep(2)
    

def norm(sentence):
    de_da_sentence = de_da.model_de_da_prediction(sentence.lower())
    ki_sentence = ki.model_ki_prediction(de_da_sentence.lower())
    #correction_sentence = sugg.suggestion_e(ki_sentence,correction).strip()
    hal_sentence = hal.correction(ki_sentence.lower())
    #proper_sentence = proper.prediction(hal_sentence)
    if len(ki_sentence ) > 1:
        punc_sentence = punc.model_punc_pred(hal_sentence.lower())
    else:
        punc_sentence = punc.model_punc_pred(ki_sentence.lower())

    punc_sentence = punc_sentence.capitalize()
    return punc_sentence


#Orginal Bu agresif tavırların onların da canını sıkmaya başladı. 
sentence = "bu agresif tavırların onlarında canını sıkmaya başladı"

#Orginal Asistana verilen doküman benimki ile aynı mıydı?
sentence1 = "asistanada verilen doküman benim ki ilee aynı mıydı"

#Orginal çarşamba ve perşembe günlerinde ders yapılmayacak mıymış?.
sentence2 ="çarşamba ve perşembe günlerin de ders yapılmıyacak mıymış"

#Orginal Ben de yaptım  ama onunki daha güzel oldu yine de.
sentence3 = "bende yaptım ama onun ki daha güzeel oldu yinede"

#Orginal Evde de elma, portakal kalmamış.
sentence4 = "evdede elma portakal kalmamış"

sentence5 = 'herkezin yalnış biliyor kanpanyatan rasgele aldım'