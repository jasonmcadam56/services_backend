from rest_framework import serializers
from backend_service import models


class Modelserializer(serializers.Serializer):

    id = serializers.UUIDField(read_only=True)
    name = serializers.CharField(max_length=20)

    dataset = serializers.PrimaryKeyRelatedField(many=False, read_only=True)
    uploaded = serializers.DateField(required=False)
    modified = serializers.DateField(required=False)

    def create(self, validated_data):
        return models.Model.objects.create(**validated_data)


class DataSetserializer(serializers.Serializer):
    id = serializers.UUIDField(read_only=True)
    name = serializers.CharField(max_length=20)

    file_path = serializers.CharField(max_length=100)
    uploaded = serializers.DateField(required=False)
    modified = serializers.DateField(required=False)

    def create(self, validated_data):
        return models.DataSet.objects.create(**validated_data)
