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
	if os.path.isfile(filename):
		data = open(filename, "r").close()
		tempList = []
		nlp = en_core_web_sm.load()
		docx = nlp(data)
		tokens = [token.text for token in docx]
		entities = [{entity.text : entity.label_}for entity in docx.ents]

		allData = ['"Token":{},\n"Entities":{}'.format(tokens, entities)]
	return allData

# #Function for named entity recognition
# def ner(my_text):
#     nlp = en_core_web_sm.load()

#     doc = nlp(my_text)
#     html = displacy.render([doc], style="ent", page=False)
#     #st.write(html, unsafe_allow_html=True)
#     st.markdown(html, unsafe_allow_html=True)
#     st.markdown(" <br> </br>", unsafe_allow_html= True)

#     #displacy.serve(doc, style="ent")


# def process_text(text):
	
   
#     text = re.sub('[^A-Za-z]', ' ', text.lower())
#     tokenized_text = word_tokenize(text)
#     clean_text = [
#         word for word in tokenized_text
#         if word not in stopwords.words('english')
#     ]
#     #gensim.parsing.stem_text(word)
    
#     #word list only
#     return clean_text




# def topic_mod(my_text , num_topics=10,num_words=5):
    
#     nlp = en_core_web_sm.load()

#     doc = nlp(my_text)
    
    
#     text_data = [sent.string.strip() for sent in doc.sents] 
#     #st.write(text_data)
    
#     texts = [process_text(text) for text in text_data]
#     #st.write(texts)
#     dictionary = corpora.Dictionary(texts)
    
#     corpus = [dictionary.doc2bow(text) for text in texts]
    
#     model = models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10,random_state =2)
#     topics = model.print_topics(num_words=num_words)
#     for topic in topics:
#         st.write(topic)

# def text_analyzer(my_text):
# 	nlp = en_core_web_sm.load()

# 	docx = nlp(my_text)
	
# 	allData = [('"Token":{},\n"Lemma":{}'.format(token.text,token.lemma_))for token in docx ]
# 	return allData

# # FUnction for pos tagging
# @st.cache
# def pos_tagging(my_text):
# 	data ={}
# 	nlp = en_core_web_sm.load()

# 	doc = nlp(my_text)
	
# 	c_tokens = [token.text for token in doc]
# 	c_pos = [token.pos_ for token in doc]

# 	new_df = pd.DataFrame(zip(c_tokens,c_pos),
# 	                      columns=['Tokens', 'POS'])

# 	return new_df

# #Function for sentiment analysis

# def sent_analysis(my_text):
# 	testimonial = TextBlob(my_text)
# 	return testimonial.sentiment.polarity, testimonial.sentiment.subjectivity

# #Function for world cloud

# def word_cloud(my_text):
# 	wordcloud = WordCloud(width=1200, height=600,background_color='white',random_state=42, stopwords=set(STOPWORDS)).generate(my_text)
# 	plt.imshow(wordcloud, interpolation='bilinear')
# 	plt.axis("off")
# 	st.pyplot()
   
# def sumy_summarizer(docx):
# 	parser = PlaintextParser.from_string(docx, Tokenizer("english"))
# 	lex_summarizer = LexRankSummarizer()
# 	summary = lex_summarizer(parser.document, 3)
# 	summary_list = [str(sentence) for sentence in summary]
# 	result = ' '.join(summary_list)
# 	return result


# def gensim_summarizer(my_text):
    
#     # nlp = spacy.load("en_core_web_sm")
#     # doc = nlp(my_text)
#     # #text_data = [sent.string.strip() for sent in doc.sents]
#     return summarize(my_text)

# # def bert_sum(message):
# #     model = sz()
# #     result = model(message, min_length=60)
# #     full = ''.join(result)
# #     st.success(full) 
     


# def summarizevis(message,summary_options):
#     if summary_options == 'sumy':
#         st.text("Using Sumy Summarizer ..")
#         summary_result = sumy_summarizer(message)
        
#     elif summary_options == 'gensim':
#         st.text("Using Gensim Summarizer ..")
#         summary_result = gensim_summarizer(message)
        
#     else:
#         st.warning("Using Default Summarizer")
#         st.text("Using Gensim Summarizer ..")
#         summary_result = sumy_summarizer(message)
        
#     st.success(summary_result)

# def main():
# 	print(entity_analyzer(my_text))

# main()

print(entity_analyzer())