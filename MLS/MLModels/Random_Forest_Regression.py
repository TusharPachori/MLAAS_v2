import os
from django.shortcuts import render, redirect
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import math
import numpy as np
from sklearn.model_selection import cross_val_score

from .Test_Train import TestTrainSplit


def Random_Forest_Regression(request):
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

            n_estimators = int(request.POST['n_estimators'])
            criterion = request.POST['criterion']
            max_depth = None if request.POST['max_depth'] == 'None' else request.POST['max_depth']
            if max_depth is not None:
                max_depth = int(request.POST['max_depth_values'])
            min_samples_split = int(request.POST['min_samples_split'])
            min_samples_leaf = int(request.POST['min_samples_leaf'])
            min_weight_fraction_leaf = float(request.POST['min_weight_fraction_leaf'])
            max_features = request.POST['max_features']
            if max_features == "Int":
                max_features = int(request.POST['max_features_integer'])
            max_leaf_nodes = None if request.POST['max_leaf_nodes'] == 'None' else request.POST['max_leaf_nodes']
            if max_leaf_nodes is not None:
                max_leaf_nodes = int(request.POST['max_leaf_nodes_value'])
            min_impurity_decrease = float(request.POST['min_impurity_decrease'])
            min_impurity_split = float(request.POST['min_impurity_split'])
            bootstrap = True if request.POST['bootstrap'] == "True" else False
            oob_score = True if request.POST['oob_score'] == "True" else False
            n_jobs = None if request.POST['n_jobs'] == 'None' else request.POST['n_jobs']
            if n_jobs is not None:
                n_jobs = int(request.POST['n_jobs_value'])
            random_state = None if request.POST['n_jobs'] == 'None' else request.POST['random_state']
            if random_state is not None:
                random_state = int(request.POST['random_state_value'])
            verbose = int(request.POST['verbose'])
            warm_start = True if request.POST['warm_start'] == "True" else False

            # print(n_estimators, criterion, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf,
            #       max_features, max_leaf_nodes, min_impurity_decrease, min_impurity_split, bootstrap,
            #       oob_score, n_jobs, random_state, verbose, warm_start)

            regressor = RandomForestRegressor(n_estimators=n_estimators,
                                              criterion=criterion,
                                              max_depth=max_depth,
                                              min_samples_split=min_samples_split,
                                              min_samples_leaf=min_samples_leaf,
                                              min_weight_fraction_leaf=min_weight_fraction_leaf,
                                              max_features=max_features,
                                              max_leaf_nodes=max_leaf_nodes,
                                              min_impurity_decrease=min_impurity_decrease,
                                              min_impurity_split=min_impurity_split,
                                              bootstrap=bootstrap,
                                              oob_score=oob_score,
                                              n_jobs=n_jobs,
                                              random_state=random_state,
                                              verbose=verbose,
                                              warm_start=warm_start)

            if request.POST['submit'] == "TRAIN":

                regressor.fit(X_train, y_train)
                y_pred = regressor.predict(X_test)
                result = mean_squared_error(y_test, y_pred)
                result = math.sqrt(result)
                result = round(result, 2)

                print(result)

                return render(request, 'MLS/result.html', {"model": "Random_Forest_Regression",
                                                           "metrics": "ROOT MEAN SQUARE ROOT",
                                                           "result": result})
            else:
                scores = cross_val_score(regressor, X, y, cv=cv, scoring='neg_mean_squared_error')
                rmse_score = np.sqrt(-scores)
                mean = scores.mean()
                std = scores.std()

                return render(request, 'MLS/validate.html', {"model": "Random_Forest_Regression",
                                                             "scoring": "neg_mean_squared_error",
                                                             "scores": scores,
                                                             'mean': mean,
                                                             'std': std,
                                                             'rmse': rmse_score})
        except Exception as e:
            return render(request, 'MLS/error.html', {"Error": e})