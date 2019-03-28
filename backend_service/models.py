from django.db import models
import uuid


class Model(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.TextField(max_length=20)

    file_path = models.TextField(max_length=100)
    uploaded = models.DateField(auto_now=True)
    modified = models.DateField(auto_now_add=True)


class DataSet(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.TextField(max_length=20)

    file_path = models.TextField(max_length=100)
    uploaded = models.DateField(auto_now=True)
    modified = models.DateField(auto_now_add=True)
