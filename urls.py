from django.conf.urls import include, url
from django.contrib import admin

urlpatterns = [
    url(r'^', include('qub_model_back.urls')),
    url(r'^', include('backend_service.urls')),
    url(r'^admin/', admin.site.urls),
]