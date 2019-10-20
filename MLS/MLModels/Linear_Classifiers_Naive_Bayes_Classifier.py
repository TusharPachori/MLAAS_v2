from django.shortcuts import render
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from .Test_Train import TestTrainSplit
from sklearn.metrics import accuracy_score
import joblib

import numpy as np
import os


def Linear_Classifiers_Naive_Bayes_Classifier(request):
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

            priors = None if request.POST['priors']=="None" else request.POST['priors']
            var_smoothing= float(request.POST['var_smoothing'])

            classifier = GaussianNB(priors=priors, var_smoothing=var_smoothing)

            if request.POST['submit'] == "TRAIN":
                classifier.fit(X_train, y_train)
                if not os.path.exists("media/user_{}/trained_model".format(request.user)):
                    os.makedirs("media/user_{}/trained_model".format(request.user))
                download_link = "media/user_{0}/trained_model/{1}".format(request.user, 'classifier.pkl')
                joblib.dump(classifier, download_link)
                y_pred = classifier.predict(X_test)
                result = accuracy_score(y_test, y_pred)

                return render(request, 'MLS/result.html', {"model": "Linear_Classifiers_Naive_Bayes_Classifier",
                                                           "metrics": "Accuracy Score",
                                                           "result": result*100,
                                                           "link": download_link})

            elif request.POST['submit'] == "VALIDATE":
                scores = cross_val_score(classifier, X, y, cv=cv, scoring='accuracy')
                rmse_score = np.sqrt(scores)
                rmse_score = np.round(rmse_score, 3)
                mean = np.round(scores.mean(), 3)
                std = np.round(scores.std(), 3)
                scores = np.round(scores, 3)

                return render(request, 'MLS/validate.html', {"model": "Linear_Classifiers_Naive_Bayes_Classifier",
                                                             "scoring": "accuracy",
                                                             "scores": scores,
                                                             'mean': mean,
                                                             'std': std,
                                                             'rmse': rmse_score,
                                                             'cv': range(cv),
                                                             'cv_list': range(1, cv+1)})

        except Exception as e:
            return render(request, 'MLS/error.html', {"Error": e})