from django.db import models
import uuid


class Model(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.TextField(max_length=20)

    file = models.FileField(upload_to='nnmodels/')
    uploaded = models.DateField(auto_now=True)
    modified = models.DateField(auto_now_add=True)


class DataSet(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.TextField(max_length=20)

    file = models.FileField(upload_to='datasets/')
    uploaded = models.DateField(auto_now=True)
    modified = models.DateField(auto_now_add=True)
