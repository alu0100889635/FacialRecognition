import cv2
import face_recognition

imgElon = face_recognition.load_image_file('ImagesBasic/elonMusk.jpg')
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)
imgElonTest = face_recognition.load_image_file('ImagesBasic/elon-musk_test.png')
imgElonTest = cv2.cvtColor(imgElonTest, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgElon)[0]
encodeElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

faceLocTest = face_recognition.face_locations(imgElonTest)[0]
encodeElonTest = face_recognition.face_encodings(imgElonTest)[0]
cv2.rectangle(imgElonTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

#Comparamos los encodings del entrenamiento y el test para ver si se trata de la misma persona
results = face_recognition.compare_faces([encodeElon], encodeElonTest)
print(results)
#Realmente lo que se quiere calcular es cuánto de similar es un encoding con otro, por tanto, se calcula la distancia entre
#los encodings. Cuanto más pequeña sea la distancia, mejor será la aproximación. 
faceDistance = face_recognition.face_distance([encodeElon], encodeElonTest)
print(faceDistance)
cv2.putText(imgElonTest, f'{results} {round(faceDistance[0],2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

cv2.imshow('Elon Musk', imgElon)
cv2.imshow('ELon Musk Test', imgElonTest)
cv2.waitKey(0)