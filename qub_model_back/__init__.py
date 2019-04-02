# This will make sure the app is always imported when
# Django starts so that shared_task will use this app.
from qub_model_back.tasks import app

__all__ = app
