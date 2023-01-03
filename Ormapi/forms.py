from django import forms
from django.forms import fields
from .models import UploadImage, Result


class ImageForm(forms.ModelForm):

    class Meta:
        model = UploadImage
        fields = '__all__'



