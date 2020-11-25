
from django.urls import path
from . import views

urlpatterns = [
    path('index/', views.index,name='home'),
    path('index/search/', views.companyStock, name='companyStock'),
    path('index/predictStocks',views.predictStocks,name='predictStocks'),
    path('index/tradingtime',views.tradingtime,name='tradingtime'),
    path('index/contact',views.contact,name='contact'),
    path('index/about',views.about,name='about'),
    path('index/loader',views.loader,name='loader')
    ]

    