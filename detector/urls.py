"""URL configuration for the detector app."""
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('predict/', views.predict_view, name='predict'),
    path('result/<int:pk>/', views.result, name='result'),
    path('history/', views.history, name='history'),
    path('about/', views.about, name='about'),
]
