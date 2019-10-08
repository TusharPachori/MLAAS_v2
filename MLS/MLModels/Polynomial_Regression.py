from django.shortcuts import render, redirect
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from .Test_Train import TestTrainSplit
import os


def Polynomial_Regression(request):
    if request.method == 'POST':
        try:
            file_name = request.POST['filename']
            my_file = "media/user_{0}/processed_csv/{1}".format(request.user, file_name)
            features = request.POST.getlist('features')
            features_list = []
            for feature in features:
                feature = feature[1:-1]
                feature = feature.strip().split(", ")
                for i in feature:
                    features_list.append(i[1:-1])
            label = request.POST['label']
            ratio = request.POST['ratio']
            cv = int(request.POST['cv'])


            X, y, X_train, X_test, y_train, y_test = TestTrainSplit(my_file, features_list, label, int(ratio))

            degree = int(request.POST['degree'])
            interaction_only = True if request.POST['interaction_only'] == "True" else False
            include_bias = True if request.POST['include_bias'] == "True" else False
            order = request.POST['order']

            # print(degree, interaction_only, include_bias, order)

            regressor = PolynomialFeatures(degree=degree,
                                          interaction_only=interaction_only,
                                          include_bias=include_bias,
                                          order=order)

            return render(request, 'MLS/result.html', {"model": "Polynomial_Regression"})

        except Exception as e:
            return render(request, 'MLS/error.html', {"Error": e})