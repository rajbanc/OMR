from django.contrib import admin
from .models import Exam, CandidateDetails, Result, MultipleImageUpload

# Register your models here.
admin.site.register(Exam)
admin.site.register(CandidateDetails)
admin.site.register(Result)
admin.site.register(MultipleImageUpload)