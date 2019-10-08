import os
from sklearn.metrics import confusion_matrix, accuracy_score
from django.shortcuts import render, redirect
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from .Test_Train import TestTrainSplit
import numpy as np


def Support_Vector_Machines(request):
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

            kernel = request.POST['kernel']
            degree = int(request.POST['degree'])
            gamma = request.POST['gamma']
            coef0 = float(request.POST['coef0'])
            tol = float(request.POST['tol'])
            C = float(request.POST['C'])
            shrinking = True if request.POST['shrinking'] == "True" else False
            probability = True if request.POST['probability'] == "True" else False
            cache_size = int(request.POST['cache_size'])
            class_weight = None if request.POST['class_weight'] == 'None' else request.POST['class_weight']
            verbose = True if request.POST['verbose'] == "True" else False
            max_iter = int(request.POST['max_iter'])
            decision_function_shape = request.POST['decision_function_shape']
            random_state = None if request.POST['random_state'] == 'None' else request.POST['random_state']

            classifier = SVC(C=C,
                             kernel=kernel,
                             degree=degree,
                             gamma=gamma,
                             coef0=coef0,
                             shrinking=shrinking,
                             probability=probability,
                             tol=tol,
                             cache_size=cache_size,
                             class_weight=class_weight,
                             verbose=verbose,
                             max_iter=max_iter,
                             decision_function_shape=decision_function_shape,
                             random_state=random_state)
            if request.POST['submit'] == "TRAIN":
                classifier.fit(X_train, y_train)
                y_pred = classifier.predict(X_test)
                result = accuracy_score(y_test, y_pred)
                print(result)

                return render(request, 'MLS/result.html', {"model": "Support_Vector_Machines",
                                                           "metrics": "Accuracy Score",
                                                           "result":result*100})
            else:
                scores = cross_val_score(classifier, X, y, cv=cv, scoring='accuracy')
                rmse_score = np.sqrt(scores)
                mean = scores.mean()
                std = scores.std()

                return render(request, 'MLS/validate.html', {"model": "Support_Vector_Machines",
                                                             "scoring": "accuracy",
                                                             "scores": scores,
                                                             'mean': mean,
                                                             'std': std,
                                                             'rmse': rmse_score})
        except Exception as e:
            return render(request, 'MLS/error.html', {"Error": e})