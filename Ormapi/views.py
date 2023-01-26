import os
from django.shortcuts import render,redirect
from django.http import HttpResponse
from .import models
from .forms import ImageForm
from PIL import Image
import numpy as np
import io
import cv2
import copy
import imutils
from imutils.perspective import four_point_transform
from imutils import contours
import random
from .models import Result, CandidateDetails, Exam, MultipleImageUpload
from ORM.settings import MEDIA_ROOT

import json

docCnt = None


def get_image(img_data):
        """Converting InMemoryUploadedFile to numpy array"""
        img = Image.open(io.BytesIO(img_data))
        # img.save(f'./media/face{random.randint(100000,1000000)}.jpg')
        return np.array(img)

def base(request):
    return render(request,'base.html')

def index(request):
    if request.method == 'POST':
       form = ImageForm(request.POST, request.FILES)
       image_ = copy.deepcopy(request.FILES.getlist('images')[0])
       exam_name_req =  request.POST['exam_name']

       if image_ and exam_name_req:
        # obtain result
        exam = Exam.objects.filter(exam_name=exam_name_req)[0]
        ANSWER_KEY = json.loads(exam.answers)

        image = get_image(image_.read())
        candidate, score, total, percent, exam_QR, height, width, ANSWERS = calculate_result(image=image, exam_name_req=exam_name_req, ANSWER_KEY=ANSWER_KEY)
        solved_que = json.dumps({"answered_correct":ANSWERS})
    
        print(image_, exam)

        is_created = MultipleImageUpload.objects.create(images=image_, exam_name=exam)
        print(f"Record {is_created} is created succesfully.")
                        
        result_record = Result(candidate=candidate, score=score, total=total, percent=percent, solved_que=solved_que, exam=exam)
        result_record.save()
       
        ## Shifted to function
       """ image, height, width, details_image = crop_img(image)
       ## read name, symbol number, exam_name from QR code 
       # adding static details
       

             
       points = [(0,0)]
       points.append((height, width))
       vstack_array = vstacked_img(image,height,width)
       image, gray =  preprocess(vstack_array, need_rect=False)
       if docCnt is None:
            image, gray =  preprocess(vstack_array, need_rect=True)
    #    cv2.imshow('image', image)

       # Getting the bird's eye view, top-view of the document
       paper = four_point_transform(image, docCnt.reshape(4, 2))
       warped = four_point_transform(gray, docCnt.reshape(4, 2))
       # cv2.imshow("paper",paper)
       # cv2.imshow("warped",warped)
       paper = image
       warped = gray

       thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
       # cv2.imshow("thresh",thresh)

       cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       cnts = imutils.grab_contours(cnts)
       print("Total contours found after threshold {}".format(len(cnts)))

       questionCnts = []

       allContourImage = paper.copy()
       cv2.drawContours(allContourImage, cnts, -1, (0, 0, 255), 3)
    #    cv2.namedWindow("allContourImage", cv2.WINDOW_FREERATIO)
    #    cv2.imshow("allContourImage",allContourImage)
    #    cv2.waitKey(0)

       # Finding the questions contours
       cnt_area = []

       for c in cnts:
           (x, y, w, h) = cv2.boundingRect(c)
           ar = w / float(h)
           if w >= 20 and h >= 20 and ar >= 0.85 and ar <= 1.15:
               questionCnts.append(c)
               cnt_area.append(cv2.contourArea(c))

    #    print(f"cnt_area val  {cnt_area}")
       thresh_val = max(cnt_area)/1.4
       print(f"Thresh val  {thresh_val}")
       print("Total questions contours found: {}".format(len(questionCnts)))

       # Sorting the contours according to the question
       questionCnts = contours.sort_contours(questionCnts, method="top-to-bottom")[0]
       correct = 0
       questionsContourImage = paper.copy()
       cv2.drawContours(questionsContourImage, cnts, -1, (0, 0, 255), 3)
       cv2.imwrite('static/images/questionsContourImage.jpg',questionsContourImage)
   
       num_options = 4
       for (q, i) in enumerate(np.arange(0, len(questionCnts), num_options)):
           cnts = contours.sort_contours(questionCnts[i: i+num_options])[0]
           cv2.drawContours(questionsContourImage, cnts, -1, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 2)
           bubbled = None
           mask_value = []

           for (j, c) in enumerate(cnts):
               mask = np.zeros(thresh.shape, dtype="uint8")
               cv2.drawContours(mask, [c], -1, 255, -1)

               mask = cv2.bitwise_and(thresh, thresh, mask=mask)
            
               total = cv2.countNonZero(mask)
               mask_value.append(total)
            #    print(f"total {total}")
   
               if bubbled is None or total > bubbled[0]:
                   bubbled = (total, j)


           num_ticked = [val>thresh_val for val in mask_value]
        #    print(f"Num of ticked {num_ticked}")  
           num_ticked = num_ticked.count(True)


           k = ANSWER_KEY[q]
           color = (0, 0, 255)

           if num_ticked == 1:
               if k == bubbled[1]:
                   color = (0, 255, 0)
                   correct += 1
        #    print (f"k: {k} \nq: {q}")

           cv2.drawContours(paper, [cnts[k]], -1, color, 3)
        #    print('-'*50)
        

       questionsContourImage = paper.copy()
       cv2.drawContours(questionsContourImage, questionCnts, -1, (0, 0, 255), 3)
    #    cv2.imwrite('static/images/questionsContourImage.jpg', questionsContourImage)
    #    cv2.imshow("questionsContourImage", questionsContourImage)

       print(f"i ko val: {q+1}")
       marks_per_question = 2
       score = correct * marks_per_question
       total = (q+1) * marks_per_question
       percent = (score / total) * 100
       print("INFO Score: {:.0f}%".format(percent))
    #    cv2.putText(paper, "{:.2f}%".format(percent), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    #    cv2.namedWindow("paper", cv2.WINDOW_FREERATIO)
    #    cv2.imshow("paper", paper)

       ## write details into database
       cv2.imwrite('static/images/checked.jpg', paper)
    #    cv2.waitKey(0)  
       hstack_img(paper) """
       if form.is_valid():
            form.save()
            img_object = form.instance
            data = { 'img_obj':img_object, 'img_height':height, 'img_width':width, 'answer_key':ANSWER_KEY, 'percent':percent}#, 'form':form}
            return render(request,'index.html', data)
    else:
        form = ImageForm()
    return render(request, 'index.html', {'form': form})  

def crop_img(image_array):
    details_image = image_array[0:250, :]
    image = image_array[250:,:]
    height, width, _ = image.shape
    path = 'static/images/cropped.jpg'
    cv2.imwrite(path, image)
    return image, height, width, details_image

def vstacked_img(image, height,width):
    cropable_points_list =[]
    num_of_cols = 4
    x1, y1 = 0, 0
    x2, y2 = width-2, height

    width = x2 - x1
    if width % num_of_cols != 0: # not directly divisible
        x2 += 1
        column_width = int((width+1)/num_of_cols)
    else:
        column_width = int(width/num_of_cols)
    # print(f"Column Width: {column_width}")
    for i in range(num_of_cols):
        j = i+1
        cropable_points_list.append([(x1+column_width*i, y1),(x1+column_width*j, y2)])
        print(f"width: {(x1+column_width*j) - (x1+column_width*i)}")
    
    # print(f"cropable_points_list: {cropable_points_list}")
    for idx, point_pair in enumerate(cropable_points_list):
        x1, y1 = point_pair[0]
        x2, y2 = point_pair[1]

        val = image[y1:y2+1, x1:x2+1]

        if idx != 0:
            vstack_array = np.vstack((vstack_array,val))

        else:
            vstack_array =  val

    # cv2.imshow("vstack_array", vstack_array)
    # cv2.waitKey(0)
    cv2.imwrite('static/images/vstacked.jpg', vstack_array)
     
    return vstack_array

def hstack_img(vstack_array):
    """Converts vstacked image into hstacked image"""
    height, width, _ = vstack_array.shape
    cropable_points_list =[]
    num_of_rows = 4
    x1, y1 = 0, 0
    x2, y2 = width, height-2
    if height % num_of_rows != 0: # not directly divisible
        y2 += 1
        rows_height = int((height+1)/num_of_rows)
    else:
        rows_height = int(height/num_of_rows)
    print(f"rows_height: {rows_height}")
    for i in range(num_of_rows):
        j = i+1
        cropable_points_list.append([(x1, y1+rows_height*i),(x2, y1+rows_height*j)])
        print(f"height: {(y1+rows_height*j) - (y1+rows_height*i)}")
    
    # print(f"cropable_points_list: {cropable_points_list}")
    for idx, point_pair in enumerate(cropable_points_list):
        x1, y1 = point_pair[0]
        x2, y2 = point_pair[1]
        print(x1, y1, x2, y2)

        val = vstack_array[y1:y2, x1:x2+1]
        print('val shape: ', val.shape)
        if idx != 0:
            hstack_array = np.hstack((hstack_array,val))

        else:
            hstack_array =  val
            
    # cv2.imshow("hstack_array", hstack_array)
    cv2.waitKey(0)
    cv2.imwrite('static/images/hstacked.jpg', hstack_array)
    return hstack_array

def preprocess(image, need_rect=False):
    global docCnt
    # global gray

    # image = cv2.resize(image_org, None, fx=0.65, fy=0.65,  interpolation = cv2.INTER_AREA)
    h, w =  image.shape[:2] 
    print(h, w)
    if need_rect:
        image = cv2.rectangle(image, (0,0), (image.shape[1], image.shape[0]), (0,255,255), 2)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred,50,100)
    # cv2.namedWindow("edged", cv2.WINDOW_FREERATIO)
    # cv2.imshow("edged",edged)

    # find contours in the image
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    print(f"cnts num {len(cnts)}")

    if len(cnts) > 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=False)
        # loop over the contours
        for idx, cnt in enumerate(cnts):
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.01 * peri, True)
        
            # if the approximated contour has 4 vertices, then we are examining
            # a rectangle
            if len(approx) == 4:
                print(f"need_rect: {need_rect}")
                docCnt = approx
                break
    return image, gray
    
# generate result
def result_view(request):
    return render(request,'view_result.html')
def result(request):
   if request.method == 'POST':
        name = request.POST['name']
        symbol_no = request.POST['symbol_no']
        candidate = CandidateDetails.objects.filter(name=name, symbol_no=symbol_no)[0]
        exam_name = request.POST['exam_name']
        exam = Exam.objects.filter(exam_name=exam_name)[0]
        if candidate and exam:
            result_row = Result.objects.filter(candidate=candidate, exam=exam).values()
            print(result_row)

        return render(request,'result.html',{"result":result_row})
   return render(request,'result.html',{"result":'my_result'})

def generate_result(request):
    if request.method == "POST":
        exam_name_req =  request.POST['exam_name']
        exam = Exam.objects.filter(exam_name=exam_name_req)[0]
        # obtain result
        ANSWER_KEY = json.loads(exam.answers)
        unchecked_sheets = MultipleImageUpload.objects.filter(exam_name=exam, is_checked=False).values('images')

        for unchecked_sheet in unchecked_sheets:
            image_path = os.path.join(MEDIA_ROOT, unchecked_sheet['images'])
            image = cv2.imread(image_path)
            candidate, score, total, percent, exam_QR, height, width, ANSWERS = calculate_result(image=image, exam_name_req=exam_name_req, ANSWER_KEY=ANSWER_KEY)
            solved_que = json.dumps({"answerd_correct":ANSWERS})
            if exam_QR == exam:
                print('if valid')
                row = Result(candidate=candidate, score=score, total=total, percent=percent, solved_que=solved_que, exam=exam)
                ml=MultipleImageUpload.objects.filter(exam_name = exam, is_checked=False).update(is_checked = True)
                print(f"update is_checked {ml}")
                print('row made')
                row.save()               
                print('row saved')
        # images_path = unchecked_sheets[0]['images']
        # print(images_path)
    return render(request,'generate_results.html',{"result":'my_result'})

def result_detail(request): 
    if request.method == 'GET':
        result = Result.objects.values("candidate", "score", "percent")
        solved_que = Result.objects.values("solved_que")
        solved_que = [list(json.loads(json.loads(r['solved_que'])['answerd_correct']).keys()) for r in solved_que]
            # print(list(json.loads(json.loads(r['solved_que'])['answerd_correct']).keys()))
            # print('-'*10)
        
        context = {'result': [[data['candidate'], data['score'], data['percent'], solved] for data, solved in zip(result, solved_que)]}
       
    return render(request,'result_detail.html', context)

#add data in the candidate table
def cand(request):
    return render(request,'cand_view.html')

def add_candidates(request):
    if request.method == 'POST':
        # serial_no = request.POST['serial_no']
        name = request.POST['name']
        symbol_no = request.POST['symbol_no']
        date_of_birth = request.POST['date_of_birth']
        exam_name = request.POST['exam_name']
        exam_name = Exam.objects.filter(exam_name=exam_name)[0]
        candidate = CandidateDetails(name=name, symbol_no=symbol_no, date_of_birth=date_of_birth, exam_name=exam_name)
        candidate.save()
    return render(request, 'candidate.html',{})

def cand_details(request):
    if request.method == 'GET':
        names = request.GET.get("name", None)
        if(( names != '') and (names != None)):
            cand = CandidateDetails.objects.filter(name=names).values('name','symbol_no')
        else:
            cand = CandidateDetails.objects.values('name','symbol_no')
        context ={'cand':cand }
    return render(request,'cand_detail.html',context)

def exam_view(request):
    return render(request,'exam_view.html')    

#add data in the exam table
def add_exam(request):
    if request.method =='POST':
        # exam_id = request.POST['exam_id']
        exam_name = request.POST['exam_name']
        # exam_year = request.POST['exam_year']
        # reg_date_time = request.POST['reg_date_time']
        _image = request.FILES.getlist('images')[0]
        print(_image)
        image = get_image(_image.read())
        ANSWERS = calculate_result(image, source_answersheet=True)
        exam = Exam(exam_name=exam_name,answersheet=_image, answers=ANSWERS)
        exam.save()
    return render(request,'exam.html',{})

def exam_detail(request): 
    if request.method == 'GET':
        exam = Exam.objects.values("exam_name", "exam_year", "reg_date_time")
        context = {'exam':exam}
    return render(request,'exam_details.html', context)

def upload(request):
    if request.method == "POST":
        images = request.FILES.getlist('images')
        print(images)
        exam_name =  request.POST['exam_name']
        exam_name = Exam.objects.filter(exam_name=exam_name)[0]
        for image in images:
            is_created = MultipleImageUpload.objects.create(images=image, exam_name=exam_name)
            print(f"is_created {is_created}")     

    images = MultipleImageUpload.objects.all()
    context = {'images': images}
    return render(request, 'multipleimg.html',context )

def calculate_result(image, exam_name_req=None, ANSWER_KEY=None, source_answersheet=False):
    image, height, width, details_image = crop_img(image)
    ## read symbol number, exam_name from QR code 
    # adding static details
    if not source_answersheet:
        candidate = CandidateDetails.objects.filter(name='mausam',symbol_no='123')[0]
        exam = Exam.objects.filter(exam_name=exam_name_req)[0]
            
    points = [(0,0)]
    points.append((height, width))
    vstack_array = vstacked_img(image,height,width)
    image, gray =  preprocess(vstack_array, need_rect=False)
    if docCnt is None:
        image, gray =  preprocess(vstack_array, need_rect=True)

    # Getting the bird's eye view, top-view of the document
    paper = four_point_transform(image, docCnt.reshape(4, 2))
    warped = four_point_transform(gray, docCnt.reshape(4, 2))

    paper = image
    warped = gray

    thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # cv2.imshow("thresh",thresh)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    print("Total contours found after threshold {}".format(len(cnts)))

    questionCnts = []

    allContourImage = paper.copy()
    cv2.drawContours(allContourImage, cnts, -1, (0, 0, 255), 3)

    # Finding the questions contours
    cnt_area = []

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        if w >= 20 and h >= 20 and ar >= 0.85 and ar <= 1.15:
            questionCnts.append(c)
            cnt_area.append(cv2.contourArea(c))

#    print(f"cnt_area val  {cnt_area}")
    thresh_val = max(cnt_area)/1.4
    print(f"Thresh val  {thresh_val}")
    print("Total questions contours found: {}".format(len(questionCnts)))

    # Sorting the contours according to the question
    questionCnts = contours.sort_contours(questionCnts, method="top-to-bottom")[0]
    correct = 0
    questionsContourImage = paper.copy()
    cv2.drawContours(questionsContourImage, cnts, -1, (0, 0, 255), 3)
    # cv2.imwrite('static/images/questionsContourImage.jpg',questionsContourImage)

    
    ANSWERS = dict()

    num_options = 4

    for (q, i) in enumerate(np.arange(0, len(questionCnts), num_options)):
        cnts = contours.sort_contours(questionCnts[i: i+num_options])[0]
        cv2.drawContours(questionsContourImage, cnts, -1, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 2)
        bubbled = None
        mask_value = []

        for (j, c) in enumerate(cnts):
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)

            mask = cv2.bitwise_and(thresh, thresh, mask=mask)
        
            total = cv2.countNonZero(mask)
            mask_value.append(total)

            if bubbled is None or total > bubbled[0]:
                bubbled = (total, j)

        
        if source_answersheet:
            ANSWERS[q] = bubbled[1]
        else:
            num_ticked = [val>thresh_val for val in mask_value] 
            num_ticked = num_ticked.count(True)

            k = ANSWER_KEY[str(q)]
            color = (0, 0, 255)

            if num_ticked == 1:
                print(q, num_ticked)
                if k == bubbled[1]:
                    ANSWERS[q] = bubbled[1]
                    color = (0, 255, 0)
                    correct += 1

            cv2.drawContours(paper, [cnts[k]], -1, color, 3)
            # print('-'*50)
    

    questionsContourImage = paper.copy()
    cv2.drawContours(questionsContourImage, questionCnts, -1, (0, 0, 255), 3)

    print(f"i ko val: {q+1}")
    marks_per_question = 1
    score = correct * marks_per_question
    total = (q+1) * marks_per_question
    percent = (score / total) * 100
    percent = float("{:.2f}".format(percent))
    print("INFO Score: {:.2f}%".format(percent))

    ## write details into database
    cv2.imwrite('static/images/checked.jpg', paper)
#    cv2.waitKey(0)  
    hstack_img(paper)
    ANSWERS = json.dumps(ANSWERS)
    print(f"ANSWERS: {type(ANSWERS)}")
    if source_answersheet:
        return ANSWERS
    return candidate, score, total, percent, exam, height, width, ANSWERS










   
         
   
      
  


    

   

   



    



    
    