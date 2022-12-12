from dataclasses import fields
from tkinter import Widget
from django import forms

from .models import correction
class NormForm(forms.ModelForm):
    class Meta:
        model = correction
        fields = ["content"]
        widgets = {
            "content":forms.Textarea(attrs={
            "rows":5,
            "cols":100,
            "placeholder" :" Cümle giriniz...",
            }),
        }
     
