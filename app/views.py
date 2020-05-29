from django.http import HttpResponse
from .utils import *
from .settings import *
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

def getBrands(request):
	brandslist = brandsParse(request.GET["domain"])
	return JsonResponse(brandslist)

def getProds(request):
	prodslist = prodsParse(request.GET["domain"], int(request.GET["brandchoice"]))
	return JsonResponse(prodslist)

def summary(request):
	data = summarymain(request.GET["domain"], request.GET["prodid"], int(request.GET["summary_ch"]), request.GET["token"])#, "cellphones", "B0050I1MHC", 2, "n")
	return JsonResponse(data)

def reviewer(request):
	data = reviewerBased(request.GET["domain"])
	return JsonResponse(data)

def reviewerInfo(request):
	data = getreviewerdetails(request.GET["domain"], request.GET["id"])
	return JsonResponse(data)

def cosineSim(request):
	data = cosinesimilarity(request.GET["domain"])
	return JsonResponse(data)

def review(request):
	data = reviewBased(request.GET["domain"])
	return JsonResponse(data)

def reviewInfo(request):
	data = getreviewdetails(request.GET["domain"], request.GET["id"], request.GET["prodid"])
	return JsonResponse(data)

def brandreco(request):
	data = brandRecommendation(request.GET["domain"], request.GET["prodid"])
	return JsonResponse(data)

@csrf_exempt
def customReview(request):
	data = custom(request.POST["prodid"], request.POST["review"], request.POST["domain"])
	return 	JsonResponse(data)
