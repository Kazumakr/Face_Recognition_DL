import cv2

cap=cv2.VideoCapture(0)
i=0
while(True):

    ret,frame=cap.read()
    frame_gray=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)

    cascade_file='./haarcascade_frontalface_alt.xml'
    cascade=cv2.CascadeClassifier(cascade_file)

    faces=cascade.detectMultiScale(frame_gray)

    for x,y,w,h in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        face=frame[y:y+h,x:x+w]
        cv2.imshow('face',face)
    cv2.imshow('frame',frame)
    if ret==True:
        frame=cv2.flip(frame,0)

    key=cv2.waitKey(1) & 0xff
    cv2.imwrite("./Data/test/Sawai/img{}_1.jpg".format(i),face)
    i+=1
    print("{}".format(i))
    if key==ord('q'):
        break
    if i==500:
        break
cap.release()

cv2.destroyAllWindows()
