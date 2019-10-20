import os
import math
from django.shortcuts import render
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from .Test_Train import TestTrainSplit
import joblib


def Simple_Linear_Regression(request):
    if request.method == 'POST':
        try:
            if request.POST['submit'] != "RandomSearch":
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

                if request.POST['submit'] == "TRAIN" or request.POST['submit'] == "VALIDATE":
                    fit_intercept = True if request.POST['fit_intercept'] == "True" else False
                    normalize = True if request.POST['normalize'] == "True" else False
                    copy_X = True if request.POST['copy_X'] == "True" else False
                    n_jobs = None if request.POST['n_jobs'] == "None" else request.POST['n_jobs']
                else:
                    fit_intercept = True if request.POST['fit_intercept'] == "True" else False
                    normalize = True if request.POST['normalize'] == "True" else False
                    copy_X = True if request.POST['copy_X'] == "True" else False
                    n_jobs = None if request.POST['n_jobs'] == "None" else request.POST['n_jobs']
                    if n_jobs is not None:
                        n_jobs = int(n_jobs)

                if request.POST['submit'] == "TRAIN" or request.POST['submit'] == "TRAIN_Rand":
                    regressor = LinearRegression(fit_intercept=fit_intercept, normalize=normalize, copy_X=copy_X, n_jobs=n_jobs)
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
                elif request.POST['submit'] == "VALIDATE" or request.POST['submit'] == "VALIDATE_Rand":
                    regressor = LinearRegression(fit_intercept=fit_intercept, normalize=normalize, copy_X=copy_X, n_jobs=n_jobs)
                    scores = cross_val_score(regressor, X, y, cv=cv, scoring='neg_mean_squared_error')
                    rmse_score = np.sqrt(-scores)
                    rmse_score = np.round(rmse_score, 3)
                    mean = np.round(scores.mean(), 3)
                    std = np.round(scores.std(), 3)
                    scores = np.round(scores, 3)

                    return render(request, 'MLS/validate.html', {"model": "Simple_Linear_Regression",
                                                                 "scoring": "neg_mean_squared_error",
                                                                 "scores": scores,
                                                                 'mean': mean,
                                                                 'std': std,
                                                                 'rmse': rmse_score,
                                                                 'cv': range(cv),
                                                                 'cv_list': range(1, cv+1)})
            elif request.POST['submit'] == "RandomSearch":
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
                X, y, X_train, X_test, y_train, y_test = TestTrainSplit(my_file, features_list, label, int(ratio))
                rand_fit_intercept = [True, False, 'boolean', 'operation']
                rand_normalize = [True, False]
                rand_copy_X = [True, False]
                regressor = LinearRegression()
                hyperparameters = dict(fit_intercept=rand_fit_intercept, normalize=rand_normalize,
                                       copy_X=rand_copy_X)
                clf = RandomizedSearchCV(regressor, hyperparameters, random_state=1, n_iter=100,
                                         cv=5, verbose=0, n_jobs=1)
                best_model = clf.fit(X, y)
                parameters = best_model.best_estimator_.get_params()
                print('Best Parameters:', parameters)
                return render(request, 'MLS/RandomSearch.html', {"model": "Simple_Linear_Regression",
                                                                            "parameters": parameters,
                                                                            "features": features_list,
                                                                            "label": label,
                                                                            "filename": file_name})

        except Exception as e:
            return render(request, 'MLS/error.html', {"Error": e})
