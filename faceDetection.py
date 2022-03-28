import cv2  


# face_cascade = cv2.CascadeClassifier("haar cascade files/haarcascade_frontalface_default.xml") 
  

# eye_cascade = cv2.CascadeClassifier("haar cascade files/haarcascade_eye_tree_eyeglasses.xml")  
face = cv2.CascadeClassifier('haarCascadeFiles/haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haarCascadeFiles/haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haarCascadeFiles/haarcascade_righteye_2splits.xml')
  
# capture frames from a camera 
cap = cv2.VideoCapture(0) 
  
# loop runs if capturing has been initialized. 
while 1:  
  
    # reads frames from a camera 
    ret, img = cap.read()  
  
    # convert to gray scale of each frames 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
  
    # Detects faces of different sizes in the input image 
    faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25)) 
  
    for (x,y,w,h) in faces: 
        # To draw a rectangle in a face  
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)  
        roi_gray = gray[y:y+h, x:x+w] 
        roi_color = img[y:y+h, x:x+w] 
  
        # Detects eyes of different sizes in the input image 
        eyes = leye.detectMultiScale(roi_gray) 
        eyes = reye.detectMultiScale(roi_gray)  
  
        #To draw a rectangle in eyes 
        for (ex,ey,ew,eh) in eyes: 
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,127,255),2) 
  
    # Display an image in a window 
    cv2.imshow('img',img) 
  
    # Wait for Esc key to stop 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# Close the window 
cap.release() 
  
# De-allocate any associated memory usage 
cv2.destroyAllWindows()  