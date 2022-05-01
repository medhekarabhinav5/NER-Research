import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import punkt
from gensim import corpora
from gensim import models 
import gensim
import pyLDAvis.gensim
from gensim.summarization.summarizer import summarize
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import re, os
import streamlit as st
import pandas as pd
from fetch import *
from textblob import TextBlob
import sys
from matplotlib import pyplot as plt

import spacy
from spacy import displacy
import en_core_web_sm
from spacy import load

filename = sys.argv[1]


def entity_analyzer(filename):
    # print("outside")
    if os.path.isfile(filename):
        # print("inside")
        file = open(filename, "r")
        data = "".join(file.readlines())
        file.close()
        dataStrip = str(data).strip()
        entityKeyValues = []
        nlp = en_core_web_sm.load()
        docx = nlp(dataStrip)
        tokens = [token.text for token in docx]
        for entity in docx.ents:
            tempDict = { entity.text : entity.label_ }
            entityKeyValues.append(tempDict)
        testDict = {}
        allData = { "Token" : tokens ,"Entities": entityKeyValues }
        for item in allData['Entities']:
            #k,v = list(item.items())[0]
            #print(f" key {k} indetified as {v}")
            testDict.update(item)
        for word in testDict.keys():
            exceptCount = 0
            tagCount = 0
            exceptList = list(testDict.keys())
            exceptList.remove(word)
            for exceptItem in exceptList:
                if word in exceptItem:
                    exceptCount += 1
            if exceptCount < 1: 
                # if re.search(r'\b' + str(word) + r'\b', dataStrip):
                #     res = [key for key, val in testDict.items() if word in key][0]
                #     # print(f"{str(res)} : {testDict[word]}")
                if testDict[word] == "GPE":
                    dataStrip = dataStrip.replace(str(word), f'<span class="label label-text" style="background-color: red;">{str(word)}<span class="label label-category label-default"> {testDict[word]}</span></span>')
                    # dataStrip2 = dataStrip2.replace("'","").replace(".","").replace("\\n", "")
                elif testDict[word] == "DATE":
                    dataStrip = dataStrip.replace(str(word), f'<span class="label label-text" style="background-color: pink;">{str(word)}<span class="label label-category label-default"> {testDict[word]}</span></span>')
                    # dataStrip2 = dataStrip2.replace("'","").replace(".","").replace("\\n", "")
                elif testDict[word] == "CARDINAL":
                    dataStrip = dataStrip.replace(str(word), f'<span class="label label-text" style="background-color: lime;">{str(word)}<span class="label label-category label-default"> {testDict[word]}</span></span>')
                    # dataStrip2 = dataStrip2.replace("'","").replace(".","").replace("\\n", "")
                elif testDict[word] == "PERSON":
                    dataStrip = dataStrip.replace(str(word), f'<span class="label label-text" style="background-color: cyan;">{str(word)}<span class="label label-category label-default"> {testDict[word]}</span></span>')
                    # dataStrip2 = dataStrip2.replace("'","").replace(".","").replace("\\n", "")
                elif testDict[word] == "ORG":
                    dataStrip = dataStrip.replace(str(word), f'<span class="label label-text" style="background-color: purple;">{str(word)}<span class="label label-category label-default"> {testDict[word]}</span></span>')
                    # dataStrip2 = dataStrip2.replace("'","").replace(".","").replace("\\n", "")
                elif testDict[word] == "EVENT":
                    dataStrip = dataStrip.replace(str(word), f'<span class="label label-text" style="background-color: orange;">{str(word)}<span class="label label-category label-default"> {testDict[word]}</span></span>')
                    # dataStrip2 = dataStrip2.replace("'","").replace(".","").replace("\\n", "")
                elif testDict[word] == "TIME":
                    dataStrip = dataStrip.replace(str(word), f'<span class="label label-text" style="background-color: green;">{str(word)}<span class="label label-category label-default"> {testDict[word]}</span></span>')
                    # dataStrip2 = dataStrip2.replace("'","").replace(".","").replace("\\n", "")
                else:
                    dataStrip = dataStrip.replace(str(word), f'<span class="label label-text" style="background-color: silver;">{str(word)}<span class="label label-category label-info"> {testDict[word]}</span></span>')
                    # dataStrip2 = dataStrip2.replace("'","").replace(".","").replace("\\n", "")
    return dataStrip
text = entity_analyzer(filename)
print(text)
# for word in testDict.keys():
#     if (" " + word + " ") in dataStrip:
#         print(word)

f = open('test.html', 'w')
  
# the html code which will go in the file GFG.html
html_template_before = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <title>Bootstrap Example</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/css/bootstrap.min.css">
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/js/bootstrap.min.js"></script>
    </head>
    <style>
        /* Set height of the grid so .sidenav can be 100% (adjust if needed) */
        .row.content {height: 1500px}
        
        /* Set gray background color and 100% height */
        .sidenav {
        background-color: #f1f1f1;
        height: 100%;
        }
        
        /* Set black background color, white text and some padding */
        footer {
        background-color: #555;
        color: white;
        padding: 15px;
        }
        
        /* On small screens, set height to 'auto' for sidenav and grid */
        @media screen and (max-width: 767px) {
        .sidenav {
            height: auto;
            padding: 15px;
        }

        span.label {
            font-size: 20px;
        }

        span.label:hover {
            background-color: white;
            color: black;
        }

        .row.content {height: auto;} 
        }
    </style>
    <body>

    <div class="container-fluid">
    <div class="row content">
        <div class="col-sm-3 sidenav">
        <h4>John's Blog</h4>
        <ul class="nav nav-pills nav-stacked">
            <li class="active"><a href="#section1">Home</a></li>
            <li><a href="#section2">Friends</a></li>
            <li><a href="#section3">Family</a></li>
            <li><a href="#section3">Photos</a></li>
        </ul><br>
        <div class="input-group">
            <input type="text" class="form-control" placeholder="Search Blog..">
            <span class="input-group-btn">
            <button class="btn btn-default" type="button">
                <span class="glyphicon glyphicon-search"></span>
            </button>
            </span>
        </div>
        </div>

        <div class="col-sm-9">
        <h4><small>RECENT POSTS</small></h4>
        <hr>"""

inputText = f"""<h2>I Love Food</h2>
        <h5><span class="glyphicon glyphicon-time"></span> Post by Jane Dane, Sep 27, 2015.</h5>
        <h5><span class="label label-danger">Food</span> <span class="label label-primary">Ipsum</span></h5><br>
        <p id="textInput" name="textInput">{text}</p>
        <br><br>"""
        
html_template_after = """<h4>Leave a Comment:</h4>
        <form role="form">
            <div class="form-group">
            <textarea class="form-control" rows="3" required></textarea>
            </div>
            <button type="submit" class="btn btn-success">Submit</button>
        </form>
        <br><br>
        
        <p><span class="badge">2</span> Comments:</p><br>
        
        <div class="row">
            <div class="col-sm-2 text-center">
            <img src="bandmember.jpg" class="img-circle" height="65" width="65" alt="Avatar">
            </div>
            <div class="col-sm-10">
            <h4>Anja <small>Sep 29, 2015, 9:12 PM</small></h4>
            <p>Keep up the GREAT work! I am cheering for you!! Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.</p>
            <br>
            </div>
            <div class="col-sm-2 text-center">
            <img src="bird.jpg" class="img-circle" height="65" width="65" alt="Avatar">
            </div>
            <div class="col-sm-10">
            <h4>John Row <small>Sep 25, 2015, 8:25 PM</small></h4>
            <p>I am so happy for you man! Finally. I am looking forward to read about your trendy life. Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.</p>
            <br>
            <p><span class="badge">1</span> Comment:</p><br>
            <div class="row">
                <div class="col-sm-2 text-center">
                <img src="bird.jpg" class="img-circle" height="65" width="65" alt="Avatar">
                </div>
                <div class="col-xs-10">
                <h4>Nested Bro <small>Sep 25, 2015, 8:28 PM</small></h4>
                <p>Me too! WOW!</p>
                <br>
                </div>
            </div>
            </div>
        </div>
        </div>
    </div>
    </div>

    <footer class="container-fluid">
    <p>Footer Text</p>
    </footer>

    </body>
    <script type="text/javascript" >
        jQuery(document).ready(function($) {

           $('#textInput').mouseup(function() {
                var text=getSelectedText();
                if (text!=''){
                    alert(text);
                }
            });
            function getSelectedText() {
                var selectedText = '';
  
                // window.getSelection
                if (window.getSelection) {
                    selectedText = window.getSelection();
                }
                // document.getSelection
                else if (document.getSelection) {
                    selectedText = document.getSelection();
                }
                // document.selection
                else if (document.selection) {
                    selectedText = 
                    document.selection.createRange().text;
                } else return;
                // To write the selected text into the textarea
                return selectedText;
            }
            $('.label-text').onclick(function() {
                var text = jQuery(this).text();
                alert(text.trim());
            });
            

        });
    </script>
    </html>
"""
  
# writing the code into the file
f.write(html_template_before + inputText + html_template_after)
  
# close the file
f.close()