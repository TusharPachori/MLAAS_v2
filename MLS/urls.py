from django.contrib import admin
from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static



urlpatterns = [
    path('', views.index, name='index'),
    path('label_selection', views.load_dataset, name='load_dataset'),
    path('preprocess', views.preprocessing, name='preprocess'),
    path('modelSelection', views.modelSelection, name='modelSelection'),
    path('hyperParam', views.hyperParam, name='hyperParam'),
    path('Linear_Classifiers_Logistic_Regression', views.Linear_Classifiers_Logistic_Regression,
         name='Linear_Classifiers_Logistic_Regression'),
    path('Linear_Classifiers_Naive_Bayes_Classifier', views.Linear_Classifiers_Naive_Bayes_Classifier,
         name='Linear_Classifiers_Naive_Bayes_Classifier'),
    path('Nearest_Neighbor', views.Nearest_Neighbor, name='Nearest_Neighbor'),
    path('Support_Vector_Machines', views.Support_Vector_Machines, name='Support_Vector_Machines'),
    path('Decision_Trees', views.Decision_Trees, name='Decision_Trees'),
    path('Random_Forest', views.Random_Forest, name='Random_Forest'),
    path('Simple_Linear_Regression', views.Simple_Linear_Regression, name='Simple_Linear_Regression'),
    path('Polynomial_Regression', views.Polynomial_Regression, name='Polynomial_Regression'),
    path('Support_Vector_Regression', views.Support_Vector_Regression, name='Support_Vector_Regression'),
    path('Decision_Tree_Regression', views.Decision_Tree_Regression, name='Decision_Tree_Regression'),
    path('Random_Forest_Regression', views.Random_Forest_Regression, name='Random_Forest_Regression'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)