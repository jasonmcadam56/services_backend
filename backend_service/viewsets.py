from backend_service import serializers, models
from rest_framework import viewsets
from rest_framework.response import Response
from django.shortcuts import get_object_or_404

from backend_service.tasks import run, debug

import os

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
        ds_id = request.data.get('dataset_id')

        if len(models.Model.objects.filter(name=name)) > 0:
            raise ValueError('Name \'{}\' already used. Model names must be unique'.format(name))

        ds = models.DataSet.objects.get(id=ds_id)

        model = models.Model.objects.create(name=name, dataset=ds)
        model.save()

        _args = parse_args({
            'run_type': 'train',
            'nn_type': type,
            'dataset_location': ds.file_path
            }
        )

        run.apply_async(args=_args)

        return Response('Training: {}'.format(model))


class DataSetViewSet(viewsets.ModelViewSet):
    queryset = models.DataSet.objects.all()
    serializer_class = serializers.DataSetserializer


def parse_args(data):

    args = []

    if data.get('run_type') == 'test':
        args.append('--test')
        args.append('--type={}'.format(data.get('nn_type')))
        args.append('--modelLoc=={}'.format(os.path.abspath('qub_model_back/nnmodels/{}'.format(data.get('nn_model')))))
        args.append('--data={}'.format(os.path.abspath('qub_model_back/datasets/{}'.format(data.get('dataset')))))
    elif data.get('run_type') == 'train':
        args.append('--train')
        args.append('--type={}'.format(data.get('nn_type')))
        args.append('--data={}'.format(data.get('dataset_location')))

    args.append('-v')

    return args
