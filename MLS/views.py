from django.shortcuts import render, redirect
from sklearn.preprocessing import LabelEncoder
import numpy as np
from .models import DataSet
from .forms import DocumentForm
from .MLmodels import *
import pandas as pd
import io
import os
from sklearn.base import TransformerMixin


class DataFrameImputer(TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)


def index(request):
    if request.method == 'POST':
        form = DocumentForm()
        return render(request, 'MLS/index.html', {'form': form})
    else:
        return redirect("home")


def load_dataset(request):
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                newdoc = DataSet(Dataset=request.FILES['Dataset'], user=request.user)
                newdoc.save()
                file_name = request.FILES['Dataset'].name
                df = pd.read_csv("media/user_{0}/raw_csv/{1}".format(request.user, file_name))
                buf = io.StringIO()
                df.info(buf=buf)
                s = buf.getvalue()
                lines = s.strip().split('\n')
                entries = lines[1].strip().split()[1]
                size = (' '.join(lines[-1].strip().split()[2:]))
                columns = []
                for i in lines[3:-2]:
                    i = i.strip().split()
                    if df[i[0]].hasnans:
                        i.append(1)
                    else:
                        i.append(0)
                    if i[-2] == 'float64':
                        x = df[i[0]].describe()
                        x = x.values
                        for j in range(1, len(x)):
                            i.append(x[j].round(2))
                        i.extend([0]*3)
                    elif i[-2] == 'int64':
                        x = df[i[0]].describe()
                        x = x.values
                        for j in range(1, len(x)):
                            i.append(x[j])
                        i.extend([0]*3)
                    else:
                        i.extend([0] * 7)
                        x = df[i[0]].describe()
                        x = x.values
                        for j in range(1, len(x)):
                            i.append(x[j])
                    columns.append(i)

                return render(request, 'MLS/dataset.html', {'entries': entries,
                                                            'column': columns,
                                                            'size': size,
                                                            'filename':file_name})
            except Exception as e:
                return render(request, 'MLS/error.html', {"Error": e})
        else:
            return redirect('index')
    else:
        return redirect("home")


def preprocessing(request):
    if request.method == 'POST':
        try:
            file_name = request.POST['filename']
            label = request.POST['label']
            my_file = "media/user_{0}/raw_csv/{1}".format(request.user, file_name)
            df = pd.read_csv(my_file)
            buf = io.StringIO()
            df.info(buf=buf)
            s = buf.getvalue()
            lines = s.strip().split('\n')
            columns = []
            for i in lines[3:-2]:
                i = i.strip().split()
                columns.append(i)
            df = df.replace(0, np.NaN)

            dfi = DataFrameImputer()
            df = dfi.fit_transform(df)

            le = LabelEncoder()
            x_label = df[label]
            df.drop(label, axis=1, inplace=True)
            for i in columns:
                if i[3] == 'object':
                    if i[0] != label:
                        df = pd.concat([df, pd.get_dummies(df[i[0]], prefix=i[0], drop_first=True)], axis=1)
                        df.drop([i[0]], axis=1, inplace=True)
                    else:
                        x_label = le.fit_transform(x_label)
            df = df.assign(label= x_label)
            df = df.rename({'label': label}, axis=1)

            if not os.path.exists("media/user_{}/processed_csv".format(request.user)):
                os.makedirs("media/user_{}/processed_csv".format(request.user))
            df.to_csv('media/user_{}/processed_csv/{}'.format(request.user, file_name), index=False)
            corr = df[df.columns[:]].corr()[label][:]
            corr = corr.round(2).values
            columns = []
            for col in df.columns:
                columns.append(col)
            comment = []
            for i in corr:
                if i >= 0.6:
                    comment.append(5)
                elif i >= 0.2:
                    comment.append(4)
                elif i > 0.0:
                    comment.append(3)
                else:
                    comment.append(0)
            return render(request, 'MLS/feature_selection.html', {'comment': comment,
                                                                  'columns': columns,
                                                                  'corr': corr,
                                                                  'label': label,
                                                                  'filename':file_name})
        except Exception as e:
            return render(request, 'MLS/error.html', {"Error": e})
    else:
        return redirect("home")


def modelSelection(request):
    if request.method == 'POST':
        try:
            features = request.POST.getlist('features')
            label = request.POST['label']
            file_name = request.POST['filename']
            my_file = "media/user_{0}/processed_csv/{1}".format(request.user, file_name)
            df = pd.read_csv(my_file)
            u_values = df[label].unique()
            values = df.shape[0]
            if len(u_values)/values >=0.05:
                regression=1
            else:
                regression=0

            return render(request, 'MLS/modelSelection.html', {'regression': regression,
                                                               'features': features,
                                                               'label':label,
                                                               'filename': file_name})
        except Exception as e:
            return render(request, 'MLS/error.html', {"Error": e})
    else:
        return redirect("home")

def hyperParam(request):
    if request.method == 'POST':
        try:
            features = request.POST.getlist('features')
            file_name = request.POST['filename']
            features_list = []
            for feature in features:
                feature = feature[1:-1]
                feature = feature.strip().split(", ")
                for i in feature:
                    features_list.append(i[1:-1])
            label = request.POST['label']
            model = request.POST['model']
            return render(request, 'MLS/' + model + '.html', {"features": features_list,
                                                              "label": label,
                                                              "model": model,
                                                              "filename": file_name})
        except Exception as e:
            return render(request, 'MLS/error.html', {"Error": e})
    else:
        return redirect("home")

