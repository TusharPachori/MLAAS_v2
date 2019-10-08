from django.shortcuts import render, redirect
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from .Test_Train import TestTrainSplit
from sklearn.metrics import accuracy_score
import  numpy as np
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

            # priors= request.POST.getlist('priors')
            var_smoothing= float(request.POST['var_smoothing'])

            # priors = [float(i) for i in priors]

            classifier = GaussianNB(priors=priors, var_smoothing=var_smoothing)

            if request.POST['submit'] == "TRAIN":
                classifier.fit(X_train, y_train)
                y_pred = classifier.predict(X_test)
                result = accuracy_score(y_test, y_pred)

                return render(request, 'MLS/result.html', {"model": "Linear_Classifiers_Naive_Bayes_Classifier",
                                                           "metrics": "Accuracy Score",
                                                           "result": result*100})
            else:
                scores = cross_val_score(classifier, X, y, cv=cv, scoring='accuracy')
                rmse_score = np.sqrt(scores)
                mean = scores.mean()
                std = scores.std()

                return render(request, 'MLS/validate.html', {"model": "Linear_Classifiers_Naive_Bayes_Classifier",
                                                             "scoring": "accuracy",
                                                             "scores": scores,
                                                             'mean': mean,
                                                             'std': std,
                                                             'rmse': rmse_score})

        except Exception as e:
            return render(request, 'MLS/error.html', {"Error": e})