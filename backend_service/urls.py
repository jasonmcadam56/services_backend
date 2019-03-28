from rest_framework import routers
from . import viewsets

router = routers.SimpleRouter()
router.register(r'model', viewsets.ModelViewSet)
router.register(r'dataset', viewsets.DataSetViewSet)

urlpatterns = router.urls
