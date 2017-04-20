from django.http import HttpResponse
from utils import *
import settings

def getBrands(request):
	brandslist = brandsParse(request.GET["domain"])
	return HttpResponse(json.dumps(brandslist), content_type="application/json")

def getProds(request):
	prodslist = prodsParse(request.GET["brand"], int(request.GET["brandchoice"]))
	return HttpResponse(json.dumps(prodslist), content_type="application/json")

def summary(request):
	data = summarymain(request.GET["domain"], request.GET["prodid"], int(request.GET["summary_ch"]), request.GET["token"])#, "cellphones", "B0050I1MHC", 2, "n")
	return HttpResponse(json.dumps(data), content_type="application/json")

def reviewer(request):
	data = reviewerBased(request.GET["domain"])
	return HttpResponse(json.dumps(data), content_type="application/json")

def reviewerInfo(request):
	data = getreviewerdetails(request.GET["domain"], request.GET["id"])
	return HttpResponse(json.dumps(data), content_type="application/json")

def cosineSim(request):
	data = cosinesimilarity(request.GET["domain"])
	return HttpResponse(json.dumps(data), content_type="application/json")

def review(request):
	data = reviewBased(request.GET["domain"])
	return HttpResponse(json.dumps(data), content_type="application/json")

def reviewInfo(request):
	data = getreviewdetails(request.GET["domain"], request.GET["id"], request.GET["prodid"])
	return HttpResponse(json.dumps(data), content_type="application/json")

