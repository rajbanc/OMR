from django.urls import path
from . import views

urlpatterns = [
    path("",views.base, name ="base"),
    path("index/",views.index, name="ORM"),
    path("result/",views.result, name="result"),
    path("cand/",views.add_candidates, name= "candi"),
    path("can/", views.cand,name = "cand"),
    path("candd/",views.cand_details, name = "candd"),
    path("exam/",views.add_exam, name = "exam"),
    path("exams/",views.exam_detail, name = "exams"),
    path("view/",views.exam_view, name = "ex"),
    path("img/",views.upload, name = "image"),
    path("generate/result/",views.generate_result, name = "gen_result"),
    path("view_result/",views.result_view, name = "view_result"),
    path("result_detail/",views.result_detail, name = "result_detail")
]