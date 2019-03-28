from backend_service import serializers, models
from rest_framework import viewsets
from rest_framework.response import Response
from django.shortcuts import get_object_or_404


class ModelViewSet(viewsets.ViewSet):
    queryset = models.Model.objects.all()

    def list(self, request):
        serializer = serializers.Modelserializer(self.queryset, many=True)
        return Response(serializer.data)

    def retrieve(self, request, pk=None):
        model = get_object_or_404(self.queryset, pk=pk)
        serializer = serializers.Modelserializer(model)
        return Response(serializer.data)


class DataSetViewSet(viewsets.ViewSet):
    queryset = models.DataSet.objects.all()

    def list(self, request):
        serializer = serializers.DataSetserializer(self.queryset, many=True)
        return Response(serializer.data)

    def retrieve(self, request, pk=None):
        ds = get_object_or_404(self.queryset, pk=pk)
        serializer = serializers.DataSetserializer(ds)
        return Response(serializer.data)
