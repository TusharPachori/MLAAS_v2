from django.shortcuts import render, redirect
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from .Test_Train import TestTrainSplit
import os


def Nearest_Neighbor(request):
    if request.method == 'POST':
        try:
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
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            result = accuracy_score(y_test, y_pred)
            print(result)

            return render(request, 'MLS/result.html', {"model": "Nearest_Neighbor",
                                                       "metrics": "Accuracy Score",
                                                       "result":result*100})
        except Exception as e:
            return render(request, 'MLS/result.html', {"model": "Nearest_Neighbor",
                                                       "metrics": "Accuracy Score",
                                                       "Error": e})