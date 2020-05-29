"""Fake Review Detection & Summarization URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/dev/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Add an import:  from blog import urls as blog_urls
    2. Add a URL to urlpatterns:  url(r'^blog/', include(blog_urls))
"""
from django.conf.urls import url
from django.contrib import admin

from . import views

urlpatterns = [
    url(r'^getbrands$', views.getBrands, name='brands'),
    url(r'^getprods$', views.getProds, name='prods'),
    url(r'^summary$', views.summary, name='summary'),
    url(r'^reviewerbased$', views.reviewer, name='reviewer'),
    url(r'^reviewerinfo$', views.reviewerInfo, name='reviewerinfo'),
    url(r'^cossim$', views.cosineSim, name='cossim'),
    url(r'^reviewbased$', views.review, name='review'),
    url(r'^reviewinfo$', views.reviewInfo, name='reviewinfo'),
    url(r'^custom$', views.customReview, name='customReview'),
    url(r'^brandreco$', views.brandreco, name='brandreco'),
]
