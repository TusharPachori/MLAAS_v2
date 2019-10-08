from django.shortcuts import render, redirect
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib
from .Test_Train import TestTrainSplit
import os
import numpy as np


def Decision_Trees(request):
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
            class_weight = None if request.POST['class_weight'] == "None" else request.POST['class_weight']
            presort = True if request.POST['presort'] == "True" else False

            # print(criterion, splitter, max_depth, min_samples_split, min_samples_leaf,
            #       min_weight_fraction_leaf, max_features, random_state, max_leaf_nodes,
            #       min_impurity_decrease, min_impurity_split, class_weight, presort)

            classifier = DecisionTreeClassifier(criterion=criterion,
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
                                                class_weight=class_weight,
                                                presort=presort)

            if request.POST['submit'] == "TRAIN":
                classifier.fit(X_train, y_train)
                if not os.path.exists("media/user_{}/trained_model".format(request.user)):
                    os.makedirs("media/user_{}/trained_model".format(request.user))
                download_link = "media/user_{0}/trained_model/{1}".format(request.user, 'classifier.pkl')
                joblib.dump(classifier, download_link)
                y_pred = classifier.predict(X_test)
                result = accuracy_score(y_test, y_pred)
                print(result)

                return render(request, 'MLS/result.html', {"model": "Decision_Trees",
                                                           "metrics": "Accuracy Score",
                                                           "result": result*100,
                                                           "link": download_link})
            else:
                scores = cross_val_score(classifier, X, y, cv=cv, scoring='accuracy')
                rmse_score = np.sqrt(scores)
                mean = scores.mean()
                std = scores.std()

                return render(request, 'MLS/validate.html', {"model": "Decision_Trees",
                                                             "scoring": "accuracy",
                                                             "scores": scores,
                                                             'mean': mean,
                                                             'std': std,
                                                             'rmse': rmse_score})

        except Exception as e:
                return render(request, 'MLS/error.html', {"Error": e})