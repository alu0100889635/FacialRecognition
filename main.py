import requests
import cv2
import face_recognition
import numpy as np

serial_number = "nldsjkfnlkjab-fskajflfs"

response = requests.get("http://127.0.0.1:8000/vehicles/digitalKey/nldsjkfnlkjab-fskajflfs")
digitalKey = response.json()
facial_embedding = np.array(digitalKey["facial_embedding"])
print(type(facial_embedding))

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    resImg = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    resImg = cv2.cvtColor(resImg, cv2.COLOR_BGR2RGB)

    facesCurrFrame = face_recognition.face_locations(resImg)
    encodesCurrFrame = face_recognition.face_encodings(resImg, facesCurrFrame)

    for encodeFace, faceLoc in zip(encodesCurrFrame, facesCurrFrame):
        matches = face_recognition.compare_faces([facial_embedding], encodeFace)
        faceDistance = face_recognition.face_distance([facial_embedding], encodeFace)
        ##Mientras sea menor o igual que 0.6 se tratará de la misma persona.
        print(faceDistance)
        matchIndex = np.argmin(faceDistance)

        if matches[matchIndex]:
            name = digitalKey["full_name"].upper()
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