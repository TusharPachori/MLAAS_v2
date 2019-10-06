import os

from django.shortcuts import render, redirect
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from .Test_Train import TestTrainSplit


def Simple_Linear_Regression(request):
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

            X_train, X_test, y_train, y_test = TestTrainSplit(my_file, features_list, label, int(ratio))

            fit_intercept = True if request.POST['fit_intercept'] == "True" else False
            normalize = True if request.POST['normalize'] == "True" else False
            copy_X = True if request.POST['copy_X'] == "True" else False
            n_jobs = None if request.POST['n_jobs'] == "True" else request.POST['n_jobs']

            regressor = LinearRegression(fit_intercept=fit_intercept, normalize=normalize, copy_X=copy_X, n_jobs=n_jobs)
            regressor.fit(X_train, y_train)
            y_pred = regressor.predict(X_test)
            result = mean_squared_error(y_test, y_pred)
            print(result)

            return render(request, 'MLS/result.html', {"model": "Simple_Linear_Regression",
                                                       "metrics": "MEAN SQUARE ROOT",
                                                       "result": result})
        except Exception as e:
            return render(request, 'MLS/result.html', {"model": "Simple_Linear_Regression",
                                                       "metrics": "MEAN SQUARE ROOT",
                                                       "Error": e})
