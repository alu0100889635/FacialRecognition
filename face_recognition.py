import cv2
import face_recognition
import os
import numpy as np

path = 'ImagesAttendance'
images = []
classNames = []
myList = os.listdir(path)

for cl in myList:
    curImg =cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

print(classNames)

def findEncodings(images):
    encodingList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodingList.append(encode)
    return encodingList

encodeListKnown = findEncodings(images)
print('Encoding complete!')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    resImg = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    resImg = cv2.cvtColor(resImg, cv2.COLOR_BGR2RGB)

    facesCurrFrame = face_recognition.face_locations(resImg)
    encodesCurrFrame = face_recognition.face_encodings(resImg, facesCurrFrame)

    for encodeFace, faceLoc in zip(encodesCurrFrame, facesCurrFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDistance = face_recognition.face_distance(encodeListKnown, encodeFace)
        print(faceDistance)
        matchIndex = np.argmin(faceDistance)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x1, y2, x2 = y1 * 4, x1 * 4, y2 * 4, x2 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2),(0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# Close the window 
cap.release() 
  
# De-allocate any associated memory usage 
cv2.destroyAllWindows()  

# faceLoc = face_recognition.face_locations(imgElon)[0]
# encodeElon = face_recognition.face_encodings(imgElon)[0]
# cv2.rectangle(imgElon, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

# faceLocTest = face_recognition.face_locations(imgElonTest)[0]
# encodeElonTest = face_recognition.face_encodings(imgElonTest)[0]
# cv2.rectangle(imgElonTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

# #Comparamos los encodings del entrenamiento y el test para ver si se trata de la misma persona
# results = face_recognition.compare_faces([encodeElon], encodeElonTest)
# print(results)
# #Realmente lo que se quiere calcular es cuánto de similar es un encoding con otro, por tanto, se calcula la distancia entre
# #los encodings. Cuanto más pequeña sea la distancia, mejor será la aproximación. 
# faceDistance = face_recognition.face_distance([encodeElon], encodeElonTest)
# print(faceDistance)
# cv2.putText(imgElonTest, f'{results} {round(faceDistance[0],2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)