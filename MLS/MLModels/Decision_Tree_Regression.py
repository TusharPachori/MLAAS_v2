from django.shortcuts import render, redirect
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from .Test_Train import TestTrainSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
import os
import numpy as np
import math
import joblib


def Decision_Tree_Regression(request):
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
                    criterion = request.POST['criterion']
                    splitter = request.POST['splitter']
                    max_depth = None if request.POST['max_depth'] == "None" else request.POST['max_depth']
                    if max_depth is not None:
                        max_depth = float(request.POST['max_depth_value'])
                    min_samples_split = int(request.POST['min_samples_split'])
                    min_samples_leaf = int(request.POST['min_samples_leaf'])
                    min_weight_fraction_leaf = float(request.POST['min_weight_fraction_leaf'])
                    max_features = request.POST['max_features']
                    if max_features == "Int":
                        max_features = int(request.POST['max_features_integer'])
                    random_state = None if request.POST['random_state'] == 'None' else request.POST['random_state']
                    if random_state is not None:
                        random_state = int(request.POST['random_state_value'])
                    max_leaf_nodes = None if request.POST['max_leaf_nodes'] == 'None' else request.POST['max_leaf_nodes']
                    if max_leaf_nodes is not None:
                        max_leaf_nodes = int(request.POST['max_leaf_nodes_value'])
                    min_impurity_decrease = float(request.POST['min_impurity_decrease'])
                    min_impurity_split = float(request.POST['min_impurity_split'])
                    presort = True if request.POST['presort'] == "True" else False

                else:
                    criterion = request.POST['criterion']
                    splitter = request.POST['splitter']
                    max_depth = None if request.POST['max_depth'] == "None" else request.POST['max_depth']
                    if max_depth is not None:
                        max_depth = int(max_depth)
                    min_samples_split = int(request.POST['min_samples_split'])
                    min_samples_leaf = int(request.POST['min_samples_leaf'])
                    min_weight_fraction_leaf = float(request.POST['min_weight_fraction_leaf'])
                    max_features = None if request.POST['max_features'] == "None" else request.POST['max_features']
                    if max_features != "auto" and max_features != "sqrt" and max_features != "log2" and max_features is not None:
                        max_features = int(max_features)
                    random_state = None if request.POST['random_state'] == 'None' else request.POST['random_state']
                    if random_state is not None:
                        random_state = int(random_state)
                    max_leaf_nodes = None if request.POST['max_leaf_nodes'] == 'None' else request.POST[
                        'max_leaf_nodes']
                    if max_leaf_nodes is not None:
                        max_leaf_nodes = int(max_leaf_nodes)
                    min_impurity_decrease = float(request.POST['min_impurity_decrease'])
                    min_impurity_split = None if request.POST['min_impurity_split'] == 'None' else request.POST['min_impurity_split']
                    if min_impurity_split is not None:
                        min_impurity_split = int(min_impurity_split)
                    presort = True if request.POST['presort'] == "True" else False

                if request.POST['submit'] == "TRAIN" or request.POST['submit'] == "TRAIN_Rand":
                    regressor = DecisionTreeRegressor(criterion=criterion,
                                                      splitter=splitter,
                                                      max_depth=max_depth,
                                                      min_samples_split=min_samples_split,
                                                      min_samples_leaf=min_samples_leaf,
                                                      min_weight_fraction_leaf=min_weight_fraction_leaf,
                                                      max_features=max_features,
                                                      random_state=random_state,
                                                      max_leaf_nodes=max_leaf_nodes,
                                                      min_impurity_decrease=min_impurity_decrease,
                                                      min_impurity_split=min_impurity_split,
                                                      presort=presort)

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

                    return render(request, 'MLS/result.html', {"model": "Decision_Tree_Regression",
                                                               "metrics": "ROOT MEAN SQUARE ROOT",
                                                               "result": result,
                                                               "link": download_link})
                elif request.POST['submit'] == "VALIDATE" or request.POST['submit'] == "VALIDATE_Rand":
                    regressor = DecisionTreeRegressor(criterion=criterion,
                                                      splitter=splitter,
                                                      max_depth=max_depth,
                                                      min_samples_split=min_samples_split,
                                                      min_samples_leaf=min_samples_leaf,
                                                      min_weight_fraction_leaf=min_weight_fraction_leaf,
                                                      max_features=max_features,
                                                      random_state=random_state,
                                                      max_leaf_nodes=max_leaf_nodes,
                                                      min_impurity_decrease=min_impurity_decrease,
                                                      min_impurity_split=min_impurity_split,
                                                      presort=presort)
                    scores = cross_val_score(regressor, X, y, cv=cv, scoring='neg_mean_squared_error')
                    rmse_score = np.sqrt(-scores)
                    rmse_score = np.round(rmse_score, 3)
                    mean = np.round(scores.mean(), 3)
                    std = np.round(scores.std(), 3)
                    scores = np.round(scores, 3)

                    return render(request, 'MLS/validate.html', {"model": "Decision_Tree_Regression",
                                                                 "scoring": "neg_mean_squared_error",
                                                                 "scores": scores,
                                                                 'mean': mean,
                                                                 'std': std,
                                                                 'rmse': rmse_score,'cv': range(cv),
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
                rand_criterion = ["mse", 'mae']
                rand_splitter = ["best", "random"]
                rand_max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
                rand_max_depth.append(None)
                rand_min_samples_split = [int(x) for x in np.linspace(2, 20, num = 10)]
                rand_min_samples_leaf = [int(x) for x in np.linspace(2, 20, num = 10)]
                rand_max_features = ["auto", "sqrt", "log2", None]
                rand_max_leaf_nodes = [int(x) for x in np.linspace(2, 20, num = 10)]
                rand_max_leaf_nodes.append(None)
                rand_presort = [True, False]
                regressor = DecisionTreeRegressor()
                hyperparameters = dict(criterion=rand_criterion,
                                       splitter=rand_splitter,
                                       max_depth=rand_max_depth,
                                       min_samples_split=rand_min_samples_split,
                                       min_samples_leaf=rand_min_samples_leaf,
                                       max_features=rand_max_features,
                                       max_leaf_nodes=rand_max_leaf_nodes,
                                       presort=rand_presort)

                clf = RandomizedSearchCV(regressor, hyperparameters, random_state=1, n_iter=100, cv=5, verbose=0,
                                         n_jobs=1)
                best_model = clf.fit(X, y)
                parameters = best_model.best_estimator_.get_params()
                # print('Best Parameters:', parameters)
                return render(request, 'MLS/RandomSearch.html', {"model": "Decision_Tree_Regression",
                                                                        "parameters": parameters,
                                                                        "features": features_list,
                                                                        "label": label,
                                                                        "filename": file_name})

        except Exception as e:
            return render(request, 'MLS/error.html', {"Error": e})