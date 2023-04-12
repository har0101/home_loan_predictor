from django.urls import path
from . import views

urlpatterns = [
    path("",views.predictor,name = 'predictor'),
    path("result",views.take_info, name="result"),
    
]
