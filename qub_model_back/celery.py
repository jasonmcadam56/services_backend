import os
from celery import Celery


os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'qub_model_back.settings')

# set the default Django settings module for the 'celery' program.
app = Celery('qub_model_back')

# Using a string here means the worker doesn't have to serialize
# the configuration object to child processes.
app.config_from_object('django.conf:settings', namespace='CELERY')

# Load task modules from all registered Django app configs. Note: this works as long as the tasks file is called tasks.py
app.autodiscover_tasks()
