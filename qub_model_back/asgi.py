import os
import django
import channels.asgi

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "qub_model_back.settings")
django.setup()

channel_layer = channels.asgi.get_channel_layer()
