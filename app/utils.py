import re
import sys
import csv
import gzip
import json
import nltk
import pickle
from .settings import *
from . import TextRank
import pandas as pd
from . import TFIDFSummary
from . import CustomReview
sys.path.append(SRC_PATH)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from numpy import savez_compressed, load


object_file = None
def brandsParse(domain):
	global object_file
	object_file = load(DATASET_BRANDS_PATH + domain.lower() + ".npz", allow_pickle=True)
	object_file = object_file['arr_0'].tolist()
	#f = open(DATASET_BRANDS_PATH + domain.lower() + ".pickle",'rb')
	#object_file = pickle.load(f)
	c=0
	brandslist={}
	for brand in object_file.keys():
		if c<20:
			brandslist[c+1]=brand
		else:
			break
		c+=1
	return brandslist

def prodsParse(domain,choice):
	global object_file
	prodslist={}
	brandslist = brandsParse(domain)
	selectedBrand = brandslist[choice]
	c=0
	for prods in range(len(object_file[selectedBrand])):
		if c<20:
			proddict = {}
			for prod in object_file[selectedBrand][prods].keys():
				proddict[object_file[selectedBrand][prods][prod]] = prod
				prodslist[c+1]= proddict
				#print str(c+1)+". "+prod+"\n"
		else:
			break
		c+=1
	return prodslist

def summarymain(domain, prodID, choice, ch_token, token=4):

	summary=""

	if choice==1:
		rankedText = TextRank.summaryGen(prodID,domain,debugging=True)

	if choice==2:
		if ch_token=="y" or ch_token=="yes":
			rankedText=TFIDFSummary.summaryGen(prodID,domain,gram=token,debug=True)
		else:
			rankedText=TFIDFSummary.summaryGen(prodID,domain,debug=True)

	return rankedText

	'''keys=keywords.extract_keywords(domain,prodslist[ch])
	rankedSummary=""
	for i in range(len(rankedText)):
		rankedSummary+=rankedText[i]
	stopwords=load_stop_words("../stoplist.txt")
	tokenizer = RegexpTokenizer("[\w']+", flags=re.UNICODE)
	tokens = tokenizer.tokenize(rankedSummary)
	tokens = [token for token in tokens if token.lower() not in stopwords]
	precision = float(len(set(tokens).intersection(set(keys))))/float(len(tokens))
	recall = float(len(set(tokens).intersection(set(keys))))/float(len(keys))
	fmeasure = 2*(precision*recall)/(precision+recall)
	return
	print "\n\n"
	print "Precision =",precision
	print "Recall =",recall
	print "F-Measure =",fmeasure'''

def reviewerBased(domain):
	fwo = open(DATASET_PATH + "goldreviewers.txt","r")
	goldstdreviewers = fwo.read().strip().split("\n")
	df = pd.read_csv(ML_PATH+domain.lower()+"_label.csv")
	reviewerlist = {'reviewerID':[],'label':[]}

	g = load(GZIP_PATH + domain.lower() + '.npz', allow_pickle=True)
	g = g['arr_0'].tolist()
	reviewerIdList = [data['reviewerID'] for data in g]
	for j in df[df['Label'] == 'FAKE']["ReviewerID"].tolist()[:4000]:
		if j not in goldstdreviewers and j in reviewerIdList:
			reviewerlist['reviewerID'].append(j)
			reviewerlist['label'].append("FAKE")
	for j in df[df['Label'] == 'TRUTHFUL']["ReviewerID"].tolist()[:4000]:
		if j in goldstdreviewers and j in reviewerIdList:
			reviewerlist['reviewerID'].append(j)
			reviewerlist['label'].append("TRUTHFUL")
	for j in df[df['Label'] == 'VERY TRUTHFUL']["ReviewerID"].tolist()[:4000]:
		if j in goldstdreviewers and j in reviewerIdList:
			reviewerlist['reviewerID'].append(j)
			reviewerlist['label'].append("VERY TRUTHFUL")
	return reviewerlist

def getreviewerdetails(domain, reviewerID):
	g = load(GZIP_PATH + domain.lower() + '.npz', allow_pickle=True)
	g = g['arr_0'].tolist()
	#g = gzip.open(GZIP_PATH + domain.lower() + '.json.gz', 'r')
	reviewerInfo = {"review":[],"brand":[],"reviewerName":"","reviewerID":""}
	for data in g:
		#json_data = json.dumps(eval(l))
		#json_obj = re.match(r'(\{.*})',json_data)
		#data = json.loads(json_obj.group())
		if data["reviewerID"]==reviewerID:
			reviewerInfo["review"].append(data["reviewText"])
			reviewerInfo["brand"].append(data["asin"])
			reviewerInfo["reviewerName"]=data["reviewerName"]
			reviewerInfo["reviewerID"]=data["reviewerID"]
	return reviewerInfo

def cosinesimilarity(domain):
	docs={}
	vocabulary = []
	reviewList = []
	#g = gzip.open(GZIP_PATH + domain.lower() + '.json.gz', 'r')
	g = load(GZIP_PATH + domain.lower() + '.npz', allow_pickle=True)
	g = g['arr_0'].tolist()
	index = 0
	for data in g:
		#t0 = time.time()
		if index<=5000:
			#json_data = json.dumps(eval(l))
			#json_obj = re.match(r'(\{.*})',json_data)
			#data = json.loads(json_obj.group())
			docs[index] = {'freq': {}, 'tf': {}, 'idf': {},
					'tf-idf': {}, 'tokens': [], 'reviewerID':"",'text':"",'asin':""}
			docs[index]["reviewerID"] = data["reviewerID"]
			docs[index]["text"] = data["reviewText"]
			docs[index]["asin"] = data["asin"]
			#t1 = time.time()
			#print t1-t0
			index+=1
		else:
			break
	documents=[]
	for i in docs:
		documents.append(docs[i]['text'])
	tfidf_vectorizer = TfidfVectorizer()
	tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
	m,n = tfidf_matrix.shape
	doneflag=[0 for i in range(m)]
	reviewer = {"reviewerID":[],"data":[]}
	fwo = open(DATASET_PATH + "goldreviewers.txt","r")
	goldstdreviewers = fwo.read().strip().split("\n")
	for i in range(m-1):
		tmp = {}
		cos_sim = cosine_similarity(tfidf_matrix[i:i+1], tfidf_matrix)
		for j in range(len(cos_sim[0])):
			if cos_sim[0][j] > 0.75 and i!=j and docs[i]['reviewerID']==docs[j]['reviewerID'] and not doneflag[i] and not doneflag[j] and docs[i]['reviewerID'] not in goldstdreviewers:
				doneflag[i] = doneflag[j] = 1
				tmp["asin1"] = docs[i]['asin']
				tmp["text1"] = docs[i]["text"]
				tmp["asin2"] = docs[j]['asin']
				tmp["text2"] = docs[j]["text"]
				reviewer['reviewerID'].append(docs[i]['reviewerID'])
				reviewer['data'].append(tmp)
	return reviewer

def reviewBased(domain):
	f = open(ML_PATH+domain.lower()+'_review_label.csv', 'rt')
	reviewerlist = {'reviewerID':[], 'brandID':[], 'label':[]}
	try:
		reader = csv.reader(f)
		for row in reader:
			if row[8]=="FAKE":
				reviewerlist["reviewerID"].append(row[0])
				reviewerlist["brandID"].append(row[1])
				reviewerlist["label"].append(row[8])
	finally:
	    f.close()
	return reviewerlist

def brandRecommendation(domain, prodID):
	f = open(ML_PATH+domain.lower()+'_review_label.csv', 'rt')
	reviewerlist = {'fakecnt':0,'totcnt':0}
	try:
		reader = csv.reader(f)
		for row in reader:
			if row[1]==prodID:
				reviewerlist["totcnt"]+=1
			if row[8]=="FAKE" and row[1]==prodID:
				reviewerlist["fakecnt"]+=1

	finally:
	    f.close()
	return reviewerlist

def getreviewdetails(domain, reviewerID, brandID):
	g = gzip.open(GZIP_PATH + domain.lower() + '.json.gz', 'r')
	reviewerInfo = {"review":[],"brand":[],"reviewerName":"","reviewerID":""}
	for l in g:
		json_data = json.dumps(eval(l))
		json_obj = re.match(r'(\{.*})',json_data)
		data = json.loads(json_obj.group())
		if data["reviewerID"]==reviewerID and data["asin"]==brandID:
			reviewerInfo["review"].append(data["reviewText"])
			reviewerInfo["brand"].append(data["asin"])
			reviewerInfo["reviewerName"]=data["reviewerName"]
			reviewerInfo["reviewerID"]=data["reviewerID"]
	return reviewerInfo

def custom(prodID,review, domain):
	detectedText = CustomReview.pred(prodID, review, domain)
	return detectedText
