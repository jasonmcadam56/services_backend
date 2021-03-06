"""qub_model_back URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.conf.urls import url

from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    url(r'^workers/$', views.worker, name='worker'),
    path('workers/<slug:worker_id>/', views.worker),
    path('model/<slug:model_id>/download/', views.download_model),
    path('dataset/upload/', views.upload_dataset),
    path('model/<slug:model_id>/test/', views.test_model),
    path('model/<slug:model_id>/test/download/', views.download_test_results),
    path('model/<slug:model_id>/predict/', views.predict, name='predict'),
    url(r'^$', views.index, name='index'),
]
