from django.db import models
import uuid


class DataSet(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.TextField(max_length=20)

    file_path = models.TextField(max_length=100)
    uploaded = models.DateField(auto_now=True, null=True)
    modified = models.DateField(auto_now_add=True, null=True)


class Model(models.Model):
    TYPE_CHOICES = [('CNN', 'cnn'), ('GRID', 'grid')]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.TextField(max_length=20)
    type = models.TextField(max_length=10, choices=TYPE_CHOICES, default='cnn')

    dataset = models.ForeignKey(DataSet, on_delete=models.PROTECT)
    uploaded = models.DateField(auto_now=True, null=True)
    modified = models.DateField(auto_now_add=True, null=True)
