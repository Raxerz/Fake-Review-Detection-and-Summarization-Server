import os
import pickle
import settings
import networkx as nx
from time import sleep
from collections import Counter
from nltk.tokenize.punkt import PunktSentenceTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

debug=False
class Summarization:
	def __init__(self,text):
		self.text=text
		self.text = ' '.join(self.text.strip().split('\n'))
		self.sentence_splitter = PunktSentenceTokenizer()
		self.sentences = self.sentence_splitter.tokenize(text)

	def tokenization(self):
		if(debug):
			return self.sentences
	def bag_of_words(self):
    		self.bag_of_words_matrix = CountVectorizer().fit_transform(self.sentences)
		if(debug):
			return self.bag_of_words_matrix
	
	def normalization(self):
		self.normalized_matrix = TfidfTransformer().fit_transform(self.bag_of_words_matrix)
		self.similarity_graph = self.normalized_matrix * self.normalized_matrix.T
		if(debug):
			return self.normalized_matrix

	def similarity(self):
		if(debug):
			return self.similarity_graph			

	def textrank(self):
		self.nx_graph = nx.from_scipy_sparse_matrix(self.similarity_graph)
		self.scores = nx.pagerank(self.nx_graph)
		self.sorted_text = sorted(((self.scores[i],s) for i,s in enumerate(self.sentences)),reverse=True)
		return self.sorted_text
		
	def summarized_text(self):
		self.summary=""
		for i in range(len(self.sorted_text)/4):
			self.summary+=self.sorted_text[i][1]
		self.summary = ' '.join(self.summary.strip().split('\n'))
		self.summary = ' '.join(self.summary.split())	
		return self.summary

#if __name__ == "__main__":
def summaryGen(fileName,domain,debugging=False):
	global debug
	debug=debugging
	content=""
	#f=open("../datasets/"+domain+"/"+fileName+".txt","r")
	brands_reviews = pickle.load( open( settings.DATASET_PATH+domain.lower()+".pickle", "rb" ) )
	review_data = brands_reviews[fileName]
	for i in review_data:
		content+=i["review"]
	beforesummary = content
	summary = Summarization(content)
	tokenization = summary.tokenization()
	bow = summary.bag_of_words()
	normalization = summary.normalization()
	similaritygraph = summary.similarity()
	rankedText=summary.textrank()
	summarized=summary.summarized_text()
	summary_dict = {"beforesummary":beforesummary, "tokenization":tokenization, "rankedtext":rankedText, "summarizedcontent":summarized}
	return summary_dict

