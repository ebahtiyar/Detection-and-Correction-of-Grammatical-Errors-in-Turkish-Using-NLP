# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 21:37:58 2021

@author: emreb
"""

import sqlite3


database = "sentences.db"

con = sqlite3.connect(database)
cursor = con.cursor()

#For news
def createTable(source):
    
    cursor.execute("CREATE TABLE IF NOT EXISTS " + source +" (Sentence TEXT UNIQUE, Date TEXT, Category TEXT, Url TEXT)")
    con.commit()
    
def addsentence(source,sentence,date,category,url):
    cursor.execute("Insert into "+ source +" Values(?,?,?,?)",(sentence,date,category,url))
    con.commit()
    



#Two options table
def createTable1(source):
    cursor.execute("CREATE TABLE IF NOT EXISTS " +source + " (Sentence TEXT UNIQUE,Context TEXT)")
    con.commit()

def addsentence1(source,sentence,context):
    cursor.execute("Insert into "+ source +" Values(?,?)",(sentence,context))
    con.commit()
    cursor.close()
        
def takeSentence(database,source):
    con = sqlite3.connect(database)
    cursor = con.cursor()
    cursor.execute("Select Sentences From " + source)
    data = cursor.fetchall()
    return data


def takeSentence1(database,source):
    con = sqlite3.connect(database)
    cursor = con.cursor()
    cursor.execute("Select * From " + source)
    data = cursor.fetchall()
    return data