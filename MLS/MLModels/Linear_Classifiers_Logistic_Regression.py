from django.shortcuts import render
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
import numpy as np
from .Test_Train import TestTrainSplit
import joblib
import os


def Linear_Classifiers_Logistic_Regression(request):
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
                    penalty = request.POST['penalty']
                    dual = True if request.POST['dual'] == "True" else False
                    tol = float(request.POST['tol'])
                    C = float(request.POST['C'])
                    fit_intercept = True if request.POST['fit_intercept'] == "True" else False
                    intercept_scaling = float(request.POST['intercept_scaling'])
                    class_weight = None if request.POST['class_weight'] == "None" else request.POST['class_weight']
                    random_state = None if request.POST['random_state'] == "None" else request.POST['random_state']
                    solver = request.POST['solver']
                    max_iter = int(request.POST['max_iter'])
                    multi_class = request.POST['multi_class']
                    verbose = int(request.POST['verbose'])
                    warm_start = True if request.POST['warm_start'] == "True" else False
                    n_jobs = None if request.POST['n_jobs'] == "None" else request.POST['n_jobs']
                    if n_jobs is not None:
                        n_jobs = int(request.POST['n_jobs_value'])
                    l1_ratio = None if request.POST['l1_ratio'] == "None" else request.POST['l1_ratio']
                    if l1_ratio is not None:
                        l1_ratio = float(request.POST['l1_ratio_value'])
                else:
                    penalty = request.POST['penalty']
                    dual = True if request.POST['dual'] == "True" else False
                    tol = float(request.POST['tol'])
                    C = float(request.POST['C'])
                    fit_intercept = True if request.POST['fit_intercept'] == "True" else False
                    intercept_scaling = float(request.POST['intercept_scaling'])
                    class_weight = None if request.POST['class_weight'] == "None" else request.POST['class_weight']
                    random_state = None if request.POST['random_state'] == "None" else request.POST['random_state']
                    solver = request.POST['solver']
                    max_iter = int(request.POST['max_iter'])
                    multi_class = request.POST['multi_class']
                    verbose = int(request.POST['verbose'])
                    warm_start = True if request.POST['warm_start'] == "True" else False
                    n_jobs = None if request.POST['n_jobs'] == "None" else request.POST['n_jobs']
                    if n_jobs is not None:
                        n_jobs = int(n_jobs)
                    l1_ratio = None if request.POST['l1_ratio'] == "None" else request.POST['l1_ratio']
                    if l1_ratio is not None:
                        l1_ratio = float(l1_ratio)

                if request.POST['submit'] == "TRAIN" or request.POST['submit'] == "TRAIN_Rand":
                    classifier = LogisticRegression(penalty=penalty,
                                                    dual=dual,
                                                    tol=tol,
                                                    C=C,
                                                    fit_intercept=fit_intercept,
                                                    intercept_scaling=intercept_scaling,
                                                    class_weight=class_weight,
                                                    random_state=random_state,
                                                    solver=solver,
                                                    max_iter=max_iter,
                                                    multi_class=multi_class,
                                                    verbose=verbose,
                                                    warm_start=warm_start,
                                                    n_jobs=n_jobs,
                                                    l1_ratio=l1_ratio)
                    classifier.fit(X_train, y_train)
                    if not os.path.exists("media/user_{}/trained_model".format(request.user)):
                        os.makedirs("media/user_{}/trained_model".format(request.user))
                    download_link = "media/user_{0}/trained_model/{1}".format(request.user, 'classifier.pkl')
                    joblib.dump(classifier, download_link)
                    y_pred = classifier.predict(X_test)
                    result = accuracy_score(y_test, y_pred)
                    print(result)
                    return render(request, 'MLS/result.html', {"model": "Linear_Classifiers_Logistic_Regression",
                                                               "metrics": "Accuracy Score",
                                                               "result": result*100,
                                                               "link": download_link})
                elif request.POST['submit'] == "VALIDATE" or request.POST['submit'] == "VALIDATE_Rand":
                    classifier = LogisticRegression(penalty=penalty,
                                                    dual=dual,
                                                    tol=tol,
                                                    C=C,
                                                    fit_intercept=fit_intercept,
                                                    intercept_scaling=intercept_scaling,
                                                    class_weight=class_weight,
                                                    random_state=random_state,
                                                    solver=solver,
                                                    max_iter=max_iter,
                                                    multi_class=multi_class,
                                                    verbose=verbose,
                                                    warm_start=warm_start,
                                                    n_jobs=n_jobs,
                                                    l1_ratio=l1_ratio)
                    scores = cross_val_score(classifier, X, y, cv=cv, scoring='accuracy')
                    rmse_score = np.sqrt(scores)
                    rmse_score = np.round(rmse_score, 3)
                    mean = np.round(scores.mean(), 3)
                    std = np.round(scores.std(), 3)
                    scores = np.round(scores, 3)

                    return render(request, 'MLS/validate.html', {"model": "Linear_Classifiers_Logistic_Regression",
                                                                 "scoring": "accuracy",
                                                                 "scores": scores,
                                                                 'mean': mean,
                                                                 'std': std,
                                                                 'rmse': rmse_score,
                                                                 'cv': range(cv),
                                                                 'cv_list': range(1, cv + 1)})
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
                rand_penalty = ['l2']
                rand_C = [float(x) for x in np.linspace(1.0, 10.0, num = 10)]
                rand_fit_intercept = [True, False]
                rand_intercept_scaling = [float(x) for x in np.linspace(1.0, 10.0, num = 10)]
                rand_intercept_scaling.append(None)
                rand_solver = ['newton-cg', 'lbfgs', 'liblinear', 'saga']
                rand_max_iter = [int(x) for x in np.linspace(100, 350, num = 6)]
                rand_multi_class = ['ovr', 'auto']
                rand_warm_start = [True, False]
                clf = LogisticRegression()
                hyperparameters = dict(penalty=rand_penalty,
                                       C=rand_C,
                                       fit_intercept=rand_fit_intercept,
                                       intercept_scaling=rand_intercept_scaling,
                                       solver=rand_solver,
                                       max_iter=rand_max_iter,
                                       multi_class=rand_multi_class,
                                       warm_start=rand_warm_start,)

                clf = RandomizedSearchCV(clf, hyperparameters, random_state=1, n_iter=100, cv=5, verbose=0,
                                         n_jobs=1)
                best_model = clf.fit(X, y)
                parameters = best_model.best_estimator_.get_params()
                print('Best Parameters:', parameters)
                return render(request, 'MLS/RandomSearch.html', {"model": "Linear_Classifiers_Logistic_Regression",
                                                                 "parameters": parameters,
                                                                 "features": features_list,
                                                                 "label": label,
                                                                 "filename": file_name})

        except Exception as e:
            return render(request, 'MLS/error.html', {"Error": e})