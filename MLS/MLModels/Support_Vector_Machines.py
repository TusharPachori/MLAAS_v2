import os
from sklearn.metrics import accuracy_score
from django.shortcuts import render
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.svm import SVC
from .Test_Train import TestTrainSplit
import joblib
import numpy as np


def Support_Vector_Machines(request):
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
                    shrinking = True if request.POST['shrinking'] == "True" else False
                    probability = True if request.POST['probability'] == "True" else False
                    cache_size = int(request.POST['cache_size'])
                    class_weight = None if request.POST['class_weight'] == 'None' else request.POST['class_weight']
                    verbose = True if request.POST['verbose'] == "True" else False
                    max_iter = int(request.POST['max_iter'])
                    decision_function_shape = request.POST['decision_function_shape']
                    random_state = None if request.POST['random_state'] == 'None' else request.POST['random_state']
                else:
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

                if request.POST['submit'] == "TRAIN" or request.POST['submit'] == "TRAIN_Rand":
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
                    classifier.fit(X_train, y_train)
                    if not os.path.exists("media/user_{}/trained_model".format(request.user)):
                        os.makedirs("media/user_{}/trained_model".format(request.user))
                    download_link = "media/user_{0}/trained_model/{1}".format(request.user, 'classifier.pkl')
                    joblib.dump(classifier, download_link)
                    y_pred = classifier.predict(X_test)
                    result = accuracy_score(y_test, y_pred)
                    print(result)

                    return render(request, 'MLS/result.html', {"model": "Support_Vector_Machines",
                                                               "metrics": "Accuracy Score",
                                                               "result":result*100,
                                                               "link": download_link})
                elif request.POST['submit'] == "VALIDATE" or request.POST['submit'] == "VALIDATE_Rand":
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
                    scores = cross_val_score(classifier, X, y, cv=cv, scoring='accuracy')
                    rmse_score = np.round(np.sqrt(scores), 3)
                    mean = np.round(scores.mean(), 3)
                    std = np.round(scores.std(), 3)
                    scores = np.round(scores, 3)

                    return render(request, 'MLS/validate.html', {"model": "Support_Vector_Machines",
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
                cv = int(request.POST['cv'])

                X, y, X_train, X_test, y_train, y_test = TestTrainSplit(my_file, features_list, label, int(ratio))
                rand_C = [float(x) for x in np.linspace(1.0, 10.0, num = 10)]
                rand_kernel = ['linear', 'poly', 'rbf', 'sigmoid']
                rand_degree = [int(x) for x in np.linspace(2.0, 5.0, num = 3)]
                rand_shrinking = [True, False]
                rand_probability = [True, False]
                rand_decision_function_shape = ['ovo', 'ovr']
                regressor = SVC()
                hyperparameters = dict(C=rand_C,
                                       kernel=rand_kernel,
                                       degree=rand_degree,
                                       shrinking=rand_shrinking,
                                       probability=rand_probability,
                                       decision_function_shape=rand_decision_function_shape,)
                clf = RandomizedSearchCV(regressor, hyperparameters, random_state=1, n_iter=100,
                                         cv=5, verbose=0, n_jobs=1)
                best_model = clf.fit(X, y)
                parameters = best_model.best_estimator_.get_params()
                print('Best Parameters:', parameters)
                return render(request, 'MLS/RandomSearch.html', {"model": "Support_Vector_Machines",
                                                                 "parameters": parameters,
                                                                 "features": features_list,
                                                                 "label": label,
                                                                 "filename": file_name
                                                                 })

        except Exception as e:
            return render(request, 'MLS/error.html', {"Error": e})