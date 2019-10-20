import os
import math
from django.shortcuts import render
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.svm import SVR
from .Test_Train import TestTrainSplit
import numpy as np
import joblib


def Support_Vector_Regression(request):
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
                else:
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

                if request.POST['submit'] == "TRAIN" or request.POST['submit'] == "TRAIN_Rand":
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
                                    max_iter=max_iter)
                    regressor.fit(X_train, y_train)
                    if not os.path.exists("media/user_{}/trained_model".format(request.user)):
                        os.makedirs("media/user_{}/trained_model".format(request.user))
                    download_link = "media/user_{0}/trained_model/{1}".format(request.user, 'regressor.pkl')
                    joblib.dump(regressor, download_link)
                    y_pred = regressor.predict(X_test)
                    result = mean_squared_error(y_test, y_pred)
                    result = math.sqrt(result)
                    result = round(result, 2)

                    print(result)

                    return render(request, 'MLS/result.html', {"model": "Support_Vector_Regression",
                                                               "metrics": "ROOT MEAN SQUARE ROOT",
                                                               "result": result,
                                                               "link": download_link})
                elif request.POST['submit'] == "VALIDATE" or request.POST['submit'] == "VALIDATE_Rand":
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
                                    max_iter=max_iter)
                    scores = cross_val_score(regressor, X, y, cv=cv, scoring='neg_mean_squared_error')
                    rmse_score = np.sqrt(-scores)
                    rmse_score = np.round(rmse_score, 3)
                    mean = np.round(scores.mean(), 3)
                    std = np.round(scores.std(), 3)
                    scores = np.round(scores, 3)

                    return render(request, 'MLS/validate.html', {"model": "Support_Vector_Regression",
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
                cv = int(request.POST['cv'])
                X, y, X_train, X_test, y_train, y_test = TestTrainSplit(my_file, features_list, label, int(ratio))
                rand_C = [float(x) for x in np.linspace(1.0, 10.0, num = 10)]
                rand_kernel = ['linear', 'poly', 'rbf', 'sigmoid']
                rand_degree = [int(x) for x in np.linspace(2.0, 5.0, num = 3)]
                rand_shrinking = [True, False]
                regressor = SVR()
                hyperparameters = dict(C=rand_C,
                                       kernel=rand_kernel,
                                       degree=rand_degree,
                                       shrinking=rand_shrinking)

                clf = RandomizedSearchCV(regressor, hyperparameters, random_state=1, n_iter=100,
                                         cv=5, verbose=0, n_jobs=1)
                best_model = clf.fit(X, y)
                parameters = best_model.best_estimator_.get_params()
                print('Best Parameters:', parameters)
                return render(request, 'MLS/RandomSearch.html', {"model": "Support_Vector_Regression",
                                                                 "parameters": parameters,
                                                                 "features": features_list,
                                                                 "label": label,
                                                                 "filename": file_name
                                                                 })

        except Exception as e:
            return render(request, 'MLS/error.html', {"Error": e})