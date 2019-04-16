from rest_framework import viewsets
from rest_framework.response import Response

from backend_service import serializers, models
from backend_service.tasks import run
from qub_model_back.views import parse_args


class ModelViewSet(viewsets.ModelViewSet):
    """
    Viewset to encapsulate the usual model (django) behaviour
    Can create with POST, list with GET and retrieve a full list with GET
    """
    queryset = models.Model.objects.all()
    serializer_class = serializers.Modelserializer

    def create(self, request, *args, **kwargs):
        name = request.data.get('name')
        type = request.data.get('type')
        task = request.data.get('task')
        ds_id = request.data.get('dataset_id')

        if len(models.Model.objects.filter(name=name)) > 0:
            raise ValueError('Name \'{}\' already used. Model names must be unique'.format(name))

        ds = models.DataSet.objects.get(id=ds_id)

        model = models.Model.objects.create(name=name, dataset=ds)
        model.save()

        _args = parse_args({
            'run_type': 'train',
            'nn_type': type,
            'dataset_location': ds.file_path,
            'name': name
            }
        )

        _kwargs = {
            "name": name,
            "task": task,
        }

        run.apply_async(args=_args, kwargs=_kwargs)

        return Response('Training: {}'.format(model))


class DataSetViewSet(viewsets.ModelViewSet):
    """
    Viewset to encapsulate the usual model (django) behaviour
    Can create with POST, list with GET and retrieve a full list with GET
    """
    queryset = models.DataSet.objects.all()
    serializer_class = serializers.DataSetserializer

