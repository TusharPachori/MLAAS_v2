from django.shortcuts import render, redirect
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from .Test_Train import TestTrainSplit
from sklearn.externals import joblib

import numpy as np
import os


def Nearest_Neighbor(request):
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

            # print(n_neighbors, weights, algorithm, leaf_size, p, metric, metric_params, n_jobs)

            classifier = KNeighborsClassifier(n_neighbors=n_neighbors,
                                              weights=weights,
                                              algorithm=algorithm,
                                              leaf_size=leaf_size,
                                              p=p,
                                              metric=metric,
                                              metric_params=metric_params,
                                              n_jobs=n_jobs)

            if request.POST['submit'] == "TRAIN":
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
            else:
                scores = cross_val_score(classifier, X, y, cv=cv, scoring='accuracy')
                rmse_score = np.sqrt(scores)
                mean = scores.mean()
                std = scores.std()

                return render(request, 'MLS/validate.html', {"model": "Nearest_Neighbor",
                                                             "scoring": "accuracy",
                                                             "scores": scores,
                                                             'mean': mean,
                                                             'std': std,
                                                             'rmse': rmse_score})
        except Exception as e:
            return render(request, 'MLS/error.html', {"Error": e})