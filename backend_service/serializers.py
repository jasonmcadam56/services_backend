from rest_framework import serializers
from backend_service import models


class Modelserializer(serializers.Serializer):

    id = serializers.UUIDField(read_only=True)
    name = serializers.CharField(max_length=20)

    file_path = serializers.CharField(max_length=100)
    uploaded = serializers.DateField()
    modified = serializers.DateField()

    def create(self, validated_data):
        return models.Model.objects.create(**validated_data)


class DataSetserializer(serializers.Serializer):
    id = serializers.UUIDField(read_only=True)
    name = serializers.CharField(max_length=20)

    file_path = serializers.CharField(max_length=100)
    uploaded = serializers.DateField()
    modified = serializers.DateField()

    def create(self, validated_data):
        return models.DataSet.objects.create(**validated_data)
