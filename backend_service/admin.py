from django.contrib import admin
from backend_service.models import Model

class ModelAdmin(admin.ModelAdmin):
    pass

admin.site.register(Model, ModelAdmin)