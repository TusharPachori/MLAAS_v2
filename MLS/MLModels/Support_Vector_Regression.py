import os

from django.shortcuts import render, redirect
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from .Test_Train import TestTrainSplit


def Support_Vector_Regression(request):
    if request.method == 'POST':

        file_name = request.POST['filename']
        my_file = "media_processed/user_{0}/{1}".format(request.user, file_name)
        features = request.POST.getlist('features')
        features_list = []
        for feature in features:
            feature = feature[1:-1]
            feature = feature.strip().split(", ")
            for i in feature:
                features_list.append(i[1:-1])
        label = request.POST['label']
        ratio = request.POST['ratio']

        X_train, X_test, y_train, y_test = TestTrainSplit(my_file, features_list, label, int(ratio))

        kernel = request.POST['kernel']
        degree = int(request.POST['degree'])
        gamma = request.POST['gamma']
        coef0 = float(request.POST['coef0'])
        tol = float(request.POST['tol'])
        C = float(request.POST['C'])
        epsilon = float(request.POST['epsilon'])
        shrinking = True if request.POST['shrinking'] == "True" else False
        cache_size = int(request.POST['cache_size'])
        verbose = True if request.POST['verbose'] == "True" else False
        max_iter = int(request.POST['max_iter'])

        # print(C, kernel, degree, gamma, coef0, shrinking, tol, epsilon, cache_size, verbose, max_iter)

        regressor = SVR(C=C,
                        kernel=kernel,
                        degree=degree,
                        gamma=gamma,
                        coef0=coef0,
                        shrinking=shrinking,
                        tol=tol,
                        epsilon=epsilon,
                        cache_size=cache_size,
                        verbose=verbose,
                        max_iter=max_iter,)

        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)
        result = mean_squared_error(y_test, y_pred)
        print(result)

        return render(request, 'MLS/result.html', {"model": "Support_Vector_Regression",
                                                   "metrics": "MEAN SQUARE ROOT",
                                                   "result": result})