from django import forms
from .models import DataSet


class DocumentForm(forms.ModelForm):
    class Meta:
        model = DataSet
        fields = ('Dataset',)
