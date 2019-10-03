from django.shortcuts import render, redirect
from sklearn.preprocessing import LabelEncoder
import numpy as np
from .models import DataSet
from .forms import DocumentForm
from .MLmodels import *
import pandas as pd
import io
import os


def read_dataset(filename):
    df = pd.read_csv(filename)
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
        columns.append(i)
    return columns, entries, size


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
            newdoc = DataSet(Dataset=request.FILES['Dataset'], user=request.user)
            newdoc.save()
            file_name = request.FILES['Dataset'].name
            print(file_name)
            columns, entries, size = read_dataset("media/user_{0}/{1}".format(request.user, file_name))
            return render(request, 'MLS/dataset.html', {'entries': entries,
                                                        'size': size,
                                                        'column': columns,
                                                        'filename':file_name})
        else:
            return redirect('index')

def preprocessing(request):
    if request.method == 'POST':
        file_name = request.POST['filename']
        my_file = "media/user_{0}/{1}".format(request.user, file_name)
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
        for i in columns:
            if i[3] == 'object':
                df[i[0]].fillna('0', inplace=True)
            else:
                df[i[0]].fillna(df[i[0]].mean(), inplace=True)

        label_encoder = LabelEncoder()
        for i in columns:
            if i[3] == 'object':
                df[i[0]] = label_encoder.fit_transform(df[i[0]])

        if not os.path.exists("media_processed/user_{}".format(request.user)):
            os.makedirs("media_processed/user_{}".format(request.user))
        df.to_csv('media_processed/user_{}/{}'.format(request.user, file_name), index=False)
        matrix = df.corr()
        matrix = matrix.round(2).values
        return render(request, 'MLS/preprocess.html', {'columns':columns,
                                                       'matrix': matrix,
                                                       'filename':file_name})

def correlation(request):
    if request.method == 'POST':
        label = request.POST['label']
        file_name = request.POST['filename']
        my_file = "media_processed/user_{0}/{1}".format(request.user, file_name)
        df = pd.read_csv(my_file)

        corr = df[df.columns[:]].corr()[label][:]
        corr = corr.round(2).values
        columns = []
        for col in df.columns:
            columns.append(col)
        comment = []
        for i in corr:
            if i>=0.6:
                comment.append(5)
            elif i>=0.2:
                comment.append(4)
            elif i>0.0:
                comment.append(3)
            else:
                comment.append(0)

        return render(request, 'MLS/feature_selection.html', {'comment': comment,
                                                              'columns': columns,
                                                              'corr': corr,
                                                              'label': label,
                                                              'filename':file_name})


def modelSelection(request):
    if request.method == 'POST':
        features = request.POST.getlist('features')
        label = request.POST['label']
        file_name = request.POST['filename']
        my_file = "media_processed/user_{0}/{1}".format(request.user, file_name)
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

def hyperParam(request):
    if request.method == 'POST':
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


