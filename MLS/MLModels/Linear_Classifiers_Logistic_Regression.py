from django.shortcuts import render, redirect
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from .Test_Train import TestTrainSplit
import os


def Linear_Classifiers_Logistic_Regression(request):
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

            # print(penalty, dual, tol, C, fit_intercept, intercept_scaling, class_weight, random_state,
            #       solver, max_iter, multi_class, verbose, warm_start, n_jobs, l1_ratio)

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
            y_pred = classifier.predict(X_test)
            result = accuracy_score(y_test, y_pred)
            print(result)
            return render(request, 'MLS/result.html', {"model": "Linear_Classifiers_Logistic_Regression",
                                                       "metrics": "Accuracy Score",
                                                       "result": result*100})
        except Exception as e:
            return render(request, 'MLS/result.html', {"model": "Linear_Classifiers_Logistic_Regression",
                                                       "metrics": "Accuracy Score",
                                                       "Error": e})