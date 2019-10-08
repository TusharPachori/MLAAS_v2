from django.shortcuts import render, redirect
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from .Test_Train import TestTrainSplit
from sklearn.model_selection import cross_val_score
import os
import numpy as np
import math



def Decision_Tree_Regression(request):
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

            if request.POST['submit'] == "TRAIN":
                regressor.fit(X_train, y_train)
                y_pred = regressor.predict(X_test)
                result = mean_squared_error(y_test, y_pred)
                result = math.sqrt(result)
                result = round(result, 2)
                print(result)

                return render(request, 'MLS/result.html', {"model": "Decision_Tree_Regression",
                                                           "metrics": "ROOT MEAN SQUARE ROOT",
                                                           "result": result})
            else:
                scores = cross_val_score(regressor, X, y, cv=cv, scoring='neg_mean_squared_error')
                rmse_score = np.sqrt(-scores)
                mean = scores.mean()
                std = scores.std()


                return render(request, 'MLS/validate.html', {"model": "Decision_Tree_Regression",
                                                           "scoring": "neg_mean_squared_error",
                                                           "scores": scores,
                                                           'mean': mean,
                                                           'std': std,
                                                           'rmse': rmse_score})

        except Exception as e:
            return render(request, 'MLS/error.html', {"Error": e})