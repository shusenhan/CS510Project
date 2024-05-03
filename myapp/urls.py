from django.urls import include, path
from rest_framework import routers
from .views import compare_texts

urlpatterns = [
    path('core/', compare_texts, name='compare-texts'),
]