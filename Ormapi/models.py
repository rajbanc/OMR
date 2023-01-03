from django.db import models


# Create your models here.


class Exam(models.Model):
    # exam_id = models.BigIntegerField(primary_key=True, null=False, blank=False, auto_created=True)
    exam_name = models.CharField(max_length=300)
    exam_year = models.DateField(auto_now_add=True)
    reg_date_time = models.DateTimeField(auto_now_add=True)
    answersheet = models.ImageField(upload_to="answersheet")
    answers = models.JSONField()

    def __self__(self):
        return self.exam_name

class CandidateDetails(models.Model):
    name = models.CharField(max_length=300)
    symbol_no = models.CharField(max_length=100) 
    date_of_birth = models.DateField(max_length=20)
    exam_name = models.ForeignKey(Exam, on_delete=models.CASCADE)

    def __self__(self):
        return self.name


class Result(models.Model):
    candidate = models.ForeignKey(CandidateDetails, related_name="verify_symbol", on_delete=models.DO_NOTHING)
    score = models.IntegerField()
    total = models.IntegerField()
    percent = models.FloatField()
    solved_que = models.JSONField()
    exam = models.ForeignKey(Exam, related_name="exam", on_delete=models.DO_NOTHING)


class UploadImage(models.Model):
    image = models.ImageField(upload_to="images")
    exam_name = models.ForeignKey(Exam, related_name="exam_foreign", on_delete=models.DO_NOTHING)
    is_checked = models.BooleanField(default=False)
    def __self__(self):
        return self.exam_name

class MultipleImageUpload(models.Model):
    images = models.ImageField(upload_to="images")
    exam_name = models.ForeignKey(Exam, related_name="exam_FK", on_delete=models.DO_NOTHING)
    is_checked = models.BooleanField(default=False)








    
    

