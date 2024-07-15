from flask import Flask,render_template,Response,request
import cv2
from ultralytics import YOLO
import supervision as sv
import os
import tensorrt

app=Flask(__name__)

file=0
model=YOLO("best.pt")
tensorrt_model = YOLO("best.engine")

def generate_frames():

    cap=cv2.VideoCapture(file)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    fourcc=cv2.VideoWriter_fourcc(*'mp4v')
    out=cv2.VideoWriter('OUTPUT.mp4',fourcc,30,(int(cap.get(3)),int(cap.get(4))))


    _, start_frame=cap.read() 
    start_frame=cv2.cvtColor(start_frame, cv2.COLOR_BGR2GRAY)
    #smoothing using gaussian filter
    start_frame=cv2.GaussianBlur(start_frame, (21,21),0)

    detection=False
    detection_mode=False
    detection_counter=0

    def warning():

        cv2.putText(frame,"UNUSUAL MOVEMENT",(30,30),cv2.FONT_HERSHEY_DUPLEX,0.75,(0,0,255),2)

    def detect(mode="OFF"):

        if mode=="OFF":

            cv2.putText(frame,"Detection mode:"+str(mode),(350,30),cv2.FONT_HERSHEY_DUPLEX,0.75,(255,0,0),2)
        
        else:

            cv2.putText(frame,"Detection mode:"+str(mode),(350,30),cv2.FONT_HERSHEY_DUPLEX,0.75,(0,255,0),2)


    box_annotator=sv.BoxAnnotator(
            thickness=2,
            text_thickness=1,
            text_scale=0.5
        )


    for result in tensorrt_model.predict(source=file,conf=0.3, stream=True):

        frame=result.orig_img
        detections= sv.Detections.from_yolov8(result)

        if result.boxes.id is not None:
           
           detections.tracker_id=result.boxes.id.cpu().numpy().astype(int)

        labels= [
            f"{model.model.names[class_id]}"
            for _, confidence, class_id, tracker_id
            in detections
        ]

        frame=box_annotator.annotate(scene=frame,detections=detections,labels=labels)

        boxes = result.boxes.xyxy.tolist()
        classes = result.boxes.cls.tolist()
        names = result.names
        confidences = result.boxes.conf.tolist()

        patient=False
        medical=False
        family=False

        for box, cls, conf in zip(boxes, classes, confidences):

            if(names[int(cls)]=="patient"):

                patient=True

            if(names[int(cls)]=="medical staff"):

                medical=True

            if(names[int(cls)]=="family"):

                family=True

            confidence = conf
            detected_class = cls
            name = names[int(cls)]
        
        detection_mode=bool(patient and (not medical) and (not family))

        if not(detection_mode):

            detection_counter=0
        
        if detection_mode:
        
            frame_bw = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            frame_bw = cv2.GaussianBlur(frame_bw,(5,5),0)
            difference=cv2.absdiff(frame_bw,start_frame)
            threshold= cv2.threshold(difference,25,255,cv2.THRESH_BINARY)[1]
            start_frame=frame_bw

            detect("ON")
            if threshold.sum() > 1200000:

                print(threshold.sum())
                detection_counter += 1

            elif detection_counter > 0:
                    
                detection_counter -= 1
        
        else:
            
            detect()

        if detection_counter > 10:
                
                if detection_mode==True:

                    detection=True

                    warning()

        out.write(frame)
        ret, jpeg = cv2.imencode('.jpg', frame) 
      
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')   
        


@app.route('/')

def index():

    return render_template('index.html')

@app.route('/video')

def video():

    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload_video')

def upload_video():

    return render_template('upl.html')

@app.route('/webcam') #route and function name has to be same

def webcam():

    global file
    file=0
    return render_template('Detection.html')

ALLOWED_EXTENSIONS=['mp4']

def allowed_file(filename):

    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload',methods=['POST'])

def upload():

    global file

    if 'video' not in request.files:

        return 'No Video File Found'
    
    video=request.files['video']

    if video.filename=="":

        return "no video file selected"
    
    if video and allowed_file(video.filename):

        video.save('static/videos/'+video.filename)
        file=(os.getcwd()+'\\static\\videos\\'+video.filename)
        return render_template('Detection.html')
    
    return "invalid file type"

if __name__=="__main__":

    app.run(debug=True)

