from django.conf.urls import include, url
from django.contrib import admin

urlpatterns = [
    url(r'^ws/', include('qub_model_back.urls')),
    url(r'^admin/', admin.site.urls),
]