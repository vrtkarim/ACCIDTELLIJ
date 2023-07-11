import cv2
import requests
import numpy as np
from tensorflow import keras
from keras.models import load_model
from keras.utils import img_to_array
from object_detection import ObjectDetection
from matplotlib.image import imsave
from datetime import datetime
 
now = datetime.now()
temps = now.strftime("%d/%m/%Y %H:%M:%S") #le temps

#initialisation du model
model = load_model('C:/Users/elyaa/Desktop/NEW_TIPE/source_code/model/tipemls') #notre modele pour la classification
#initialisation de YOLO algorithme
veh_detection = ObjectDetection()
#video d'experience
video = cv2.VideoCapture("C:/Users/elyaa/Desktop/NEW_TIPE/videos/surveillance_camera.mp4")
while True:
    k=0
    ret, frame = video.read()
    if not ret:
        break
    carres = veh_detection.detect(frame)[2]
    for carre in carres:
        (x, y, w, h) = carre
        if frame[y:y+h,x:x+w] is not None:
            veh = cv2.resize(frame[y:y+h,x:x+w], (200,200))
            veh = veh.astype('float')/255.0 #0-255 => 0-1
            veh = img_to_array(veh) #liste de listes => array 
            veh = np.expand_dims(veh, axis=0) #(200,200,3) => (1,200,200,3)
            resultat = model.predict(veh) #predire l'image par le model 
            if int(resultat)==1:
                #rectangle vert, et l'etat du vehicules
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, 'normal',(x,y-10), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0))

            else: #cas d'accident
                acc = frame[y-50:y+h+50, x-50:x+w+50]
                #rectangle rouge, et l'etat du vehicules
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, 'accident',(x,y-10), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 255))
                cv2.putText(frame,temps,(x,y+h+20), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 255))
                imsave('C:/Users/elyaa/Desktop/NEW_TIPE/source_code/accidents/'+str(k)+'.png', acc)#enregistrer l'image d'accident 
        else:
            pass

    #ouvrir l'image d'accident stockée et l'envoyer au site web
    acc = open('C:/Users/elyaa/Desktop/NEW_TIPE/source_code/accidents/'+str(k)+'.png', 'rb')
    requests.post('http://localhost:1880/IMAGE', files = acc)
    k+=1
    cv2.imshow("Surveillance", frame)#afficher la fenetre du video
    key = cv2.waitKey(1)
    if key == 27: 
        break
video.release()
cv2.destroyAllWindows()
