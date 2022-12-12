from cmath import exp
from django.shortcuts import render,redirect
from .forms import NormForm
from django.http import HttpResponseRedirect
import re
import time
import string
from model_ki import model_ki_prediction

unique_labels = set()
unique_labels = {'QUE', 'COM', 'EMP', 'PER'}
labels_to_ids = {k: v for v, k in enumerate(sorted(unique_labels))}
ids_to_labels = {v: k for v, k in enumerate(sorted(unique_labels))}
label_all_tokens = True





print("Model Ki Active")
time.sleep(2)
from model_de_da import model_de_da_prediction,lastControl
print("Model De Da Active")
time.sleep(2)
from bert_model import BertModel
model = BertModel()
from model_punc import model_punc_pred
print("Model Punctiation Active")
time.sleep(2)
from suggestion_word import suggestion_word_web
print("Model Suggestion Word Active")
time.sleep(2)
from model_hal import correction
print("Model Case Suffixes Active")




def index(request):
    return render(request,"index.html")

def about(request):
    return render(request,"about.html")
def removePunc(line):
    return re.sub(r'[^\w\s]', '', line)
def normalization(request):
    form = NormForm(request.POST or None)
    
    if form.is_valid():
        text = form.cleaned_data.get("content")
        try:
          text_de = model_de_da_prediction(text)
        except:
            text_de = text
        #text_de = model_de_da_prediction(text)
        try:         
            text_ki = model_ki_prediction(text)
        except:
            text_ki = text
        
        try:
         text_2 = model_de_da_prediction(text_ki).capitalize()
        except:
            text_2 = text
        #text_2 = model_de_da_prediction(text_ki).capitalize()
        try:
            de_words = de_ki_signal(text_ki.capitalize(),model_de_da_prediction(text_ki).capitalize(),de = True)
        except:
            de_words = []
            for i in range(0,len(text.split())):
                de_words.append("SSS")
        try: 
            ki_words = de_ki_signal(text_de.capitalize(),model_ki_prediction(text_de).capitalize(),de = False)
        except:
            ki_words = []
            for i in range(0,len(text.split())):
                ki_words.append("SSS")

        try:
            hal = correction(text_2).capitalize()
        except:
             hal = text
        
        try:
            hal_words = de_ki_signal(removePunc(text),hal,de = False)
        except:
            hal_words = []
            for i in range(0,len(text.split())):
                hal_words.append("SSS")
        
        try:
            punc = model_punc_pred(hal).capitalize()
        except:
            punc = text

        try:
            punc_words = de_ki_signal(text,punc,de = False)
        except:
            punc_words = []
            for i in range(0,len(text.split())):
                punc_words.append("SSS")

        sugg,turkish = suggestion_word_web(text_2)
        
        try:
            sugg_0 = de_ki_signal(text,model_punc_pred(correction(sugg[0])).capitalize(),de = False)
        except:
            sugg_0 = []
            for i in range(0,len(text.split())):
                sugg_0.append("SSS")
        
        try:
            sugg_1 = de_ki_signal(text,model_punc_pred(correction(sugg[1])).capitalize(),de = False)
        except:
            sugg_1 = []
            for i in range(0,len(text.split())):
                sugg_1.append("SSS")
        
        try:
            sugg_2 = de_ki_signal(text,model_punc_pred(correction(sugg[2])).capitalize(),de = False)
        except:
            sugg_2 = []
            for i in range(0,len(text.split())):
                sugg_2.append("SSS")
        
        words = word_tokenizer(text)
        xx = zip(hal_words,sugg_0,sugg_1,sugg_2,punc_words,de_words,ki_words)
        xx1 = zip(hal_words,sugg_0,sugg_1,sugg_2,punc_words,de_words,ki_words)
        lbl = label_word(xx1)
        kk = []
        for p,(hal,punc,s0,s1,s2,de,ki) in enumerate(xx):
            z = []
            if ki != "SSS":
                z.append(ki)
            if de != "SSS":    
                z.append(de)
            if s0 != "SSS":
                z.append(s0)
            if s1 != "SSS":
                z.append(s1)
            if s2 != "SSS":
                z.append(s2)
            if punc != "SSS":
                z.append(punc)
            if hal != "SSS":
                z.append(hal)
            kk.append(z)    
        kkk = zip(words,lbl) 
        print(xx1)
        content = {"form":form,"text":text,"words":kkk,"sugg":kk,"turkish":turkish}

        """
        text = form.cleaned_data.get("content")
        text_2 = norm(str(text))
        print(text_2)
        sugg_0  = de_ki_signal(text,text_2,de = True)
        sugg,turkish = suggestion_word_web(text_2)
        sugg_1 = de_ki_signal(text_2.lower(),sugg[0].capitalize(),de = False)
        sugg_2 = de_ki_signal(text_2.lower(),sugg[1].capitalize(),de = False)
        sugg_3 = de_ki_signal(text_2.lower(),sugg[2].capitalize(),de = False)
        xx = zip(sugg_0,sugg_1,sugg_2,sugg_3)
        xx1 = zip(sugg_0,sugg_1,sugg_2,sugg_3)
        lbl = label_word(xx1)
        words = word_tokenizer(text)
        kk = []
        for p,(s0,s1,s2,s3) in enumerate(xx):
            z = []
            if s0 != "SSS":
                z.append(s0)
            if s1 != "SSS":    
                z.append(s1)
            if s2 != "SSS":
                z.append(s2)
            if s3 != "SSS":
                z.append(s3)
            kk.append(z)    
        kkk = zip(words,lbl) 
        print(xx1)
        content = {"form":form,"text":text,"words":kkk,"sugg":kk,"turkish":turkish}
        """
        return render(request,"normalization.html",content)  
        
    return render(request,"normalization.html",{"form":form})   



def label_word(xx):
    lbl = []
    for h,s0,s1,s2,i,j,k in xx:
        c = i != "SSS" or j != "SSS" or k != "SSS" or s0 != "SSS" or s1 != "SSS" or s2 != "SSS" or h != "SSS"
        if c:
            lbl.append("1")
        else:
            lbl.append("0")
    return lbl
"""
def label_word(xx):
    lbl = []
    for s0,s1,s2,s3 in xx:
        c = s0 != "SSS" or s1 != "SSS" or s2 != "SSS" or s3 != "SSS"
        if c:
            lbl.append("1")
        else:
            lbl.append("0")
    return lbl
"""
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



def de_ki_signal(sentence,pred,de):
    words = word_tokenizer(sentence)
    if de:
        pred = lastControl(pred).capitalize()
    pred_w = word_tokenizer(pred)
    diff_w = []
    for p,w in enumerate(words):
        if w == pred_w[p]:
            diff_w.append("SSS")

        else:
            diff_w.append(pred_w[p])
    print(diff_w)
    return diff_w



