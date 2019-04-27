from django.contrib import admin
from backend_service.models import DataSet, Model

class DataSetAdmin(admin.ModelAdmin):
    pass

class ModelAdmin(admin.ModelAdmin):
    pass


admin.site.register(DataSet, DataSetAdmin)
admin.site.register(Model, ModelAdmin)