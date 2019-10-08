import os
import math
from django.shortcuts import render, redirect
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import numpy as np
from .Test_Train import TestTrainSplit
from sklearn.externals import joblib


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
            cv = int(request.POST['cv'])


            X, y, X_train, X_test, y_train, y_test = TestTrainSplit(my_file, features_list, label, int(ratio))

            fit_intercept = True if request.POST['fit_intercept'] == "True" else False
            normalize = True if request.POST['normalize'] == "True" else False
            copy_X = True if request.POST['copy_X'] == "True" else False
            n_jobs = None if request.POST['n_jobs'] == "True" else request.POST['n_jobs']

            regressor = LinearRegression(fit_intercept=fit_intercept, normalize=normalize, copy_X=copy_X, n_jobs=n_jobs)
            if request.POST['submit'] == "TRAIN":
                regressor.fit(X_train, y_train)
                download_link = "media/user_{0}/trained_model/{1}".format(request.user, 'regressor.pkl')
                joblib.dump(regressor, download_link)
                y_pred = regressor.predict(X_test)
                result = mean_squared_error(y_test, y_pred)
                result = math.sqrt(result)
                result = round(result, 2)

                print(result)

                return render(request, 'MLS/result.html', {"model": "Simple_Linear_Regression",
                                                           "metrics": "ROOT MEAN SQUARE ROOT",
                                                           "result": result,
                                                           "link": download_link})
            else:
                scores = cross_val_score(regressor, X, y, cv=cv, scoring='neg_mean_squared_error')
                rmse_score = np.sqrt(-scores)
                mean = scores.mean()
                std = scores.std()

                return render(request, 'MLS/validate.html', {"model": "Simple_Linear_Regression",
                                                             "scoring": "neg_mean_squared_error",
                                                             "scores": scores,
                                                             'mean': mean,
                                                             'std': std,
                                                             'rmse': rmse_score})
        except Exception as e:
            return render(request, 'MLS/error.html', {"Error": e})
