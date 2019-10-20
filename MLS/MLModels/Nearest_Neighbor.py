from django.shortcuts import render
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from .Test_Train import TestTrainSplit
import joblib

import numpy as np
import os


def Nearest_Neighbor(request):
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
                    n_neighbors = int(request.POST['n_neighbors'])
                    weights = request.POST['weights']
                    algorithm = request.POST['algorithm']
                    leaf_size = int(request.POST['leaf_size'])
                    p = int(request.POST['p'])
                    metric = request.POST['metric']
                    metric_params = None if request.POST['metric_params'] == "None" else request.POST['metric_params']
                    n_jobs = None if request.POST['n_jobs'] == "None" else request.POST['n_jobs']
                    if n_jobs is not None:
                        n_jobs = int(request.POST['n_jobs_value'])
                else:
                    n_neighbors = int(request.POST['n_neighbors'])
                    weights = request.POST['weights']
                    algorithm = request.POST['algorithm']
                    leaf_size = int(request.POST['leaf_size'])
                    p = int(request.POST['p'])
                    metric = request.POST['metric']
                    metric_params = None if request.POST['metric_params'] == "None" else request.POST['metric_params']
                    n_jobs = None if request.POST['n_jobs'] == "None" else request.POST['n_jobs']
                    if n_jobs is not None:
                        n_jobs = int(n_jobs)

                if request.POST['submit'] == "TRAIN" or request.POST['submit'] == "TRAIN_Rand":
                    classifier = KNeighborsClassifier(n_neighbors=n_neighbors,
                                                      weights=weights,
                                                      algorithm=algorithm,
                                                      leaf_size=leaf_size,
                                                      p=p,
                                                      metric=metric,
                                                      metric_params=metric_params,
                                                      n_jobs=n_jobs)
                    classifier.fit(X_train, y_train)
                    if not os.path.exists("media/user_{}/trained_model".format(request.user)):
                        os.makedirs("media/user_{}/trained_model".format(request.user))
                    download_link = "media/user_{0}/trained_model/{1}".format(request.user, 'classifier.pkl')
                    joblib.dump(classifier, download_link)
                    y_pred = classifier.predict(X_test)
                    result = accuracy_score(y_test, y_pred)

                    print(result)

                    return render(request, 'MLS/result.html', {"model": "Nearest_Neighbor",
                                                               "metrics": "Accuracy Score",
                                                               "result": result*100,
                                                               "link": download_link})
                elif request.POST['submit'] == "VALIDATE" or request.POST['submit'] == "VALIDATE_Rand":
                    classifier = KNeighborsClassifier(n_neighbors=n_neighbors,
                                                      weights=weights,
                                                      algorithm=algorithm,
                                                      leaf_size=leaf_size,
                                                      p=p,
                                                      metric=metric,
                                                      metric_params=metric_params,
                                                      n_jobs=n_jobs)
                    scores = cross_val_score(classifier, X, y, cv=cv, scoring='accuracy')
                    rmse_score = np.sqrt(scores)
                    rmse_score = np.round(rmse_score, 3)
                    mean = np.round(scores.mean(), 3)
                    std = np.round(scores.std(), 3)
                    scores = np.round(scores, 3)

                    return render(request, 'MLS/validate.html', {"model": "Nearest_Neighbor",
                                                                 "scoring": "accuracy",
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
                rand_n_neighbors = [int(x) for x in np.linspace(1, 10, num = 10)]
                rand_weights = ["uniform", "distance"]
                rand_algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
                rand_leaf_size = [int(x) for x in np.linspace(20, 50, num = 4)]
                rand_p = [int(x) for x in np.linspace(1, 5, num = 5)]

                clf = KNeighborsClassifier()
                hyperparameters = dict(n_neighbors=rand_n_neighbors,
                                       weights=rand_weights,
                                       algorithm=rand_algorithm,
                                       leaf_size=rand_leaf_size,
                                       p=rand_p)

                clf = RandomizedSearchCV(clf, hyperparameters, random_state=1, n_iter=100, cv=5, verbose=0,
                                         n_jobs=1)
                best_model = clf.fit(X, y)
                parameters = best_model.best_estimator_.get_params()
                print('Best Parameters:', parameters)
                return render(request, 'MLS/RandomSearch.html', {"model": "Nearest_Neighbor",
                                                                 "parameters": parameters,
                                                                 "features": features_list,
                                                                 "label": label,
                                                                 "filename": file_name})

        except Exception as e:
            return render(request, 'MLS/error.html', {"Error": e})