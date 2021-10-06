import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time
from twilio.rest  import Client

#Vinculamos nuestra cuenta de twilio para poder realizar el envio de mensajes
account_sid="AC61525cf352f75986df869821f8dc653f"
auth_token= "797a2909f086007cee083b12547043ee"
client = Client(account_sid, auth_token)

#Cargamos la alarma a nuestra variable sound
mixer.init()
sound = mixer.Sound('alarm.wav')


#Cargamos los clasificadores de la libreria cv2
face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')



print("Seleccione si desea realizar la captura por movil o por la camara web del computador")
print("1. Camara web 2. Celular")
print("(Tenga en cuenta que si selecciona el Celular debe tener abierto el servidor de IP Webcam)")
opcion=input()
opcion= int(opcion)
lbl=['Cerrado','Abierto']

model = load_model('models/Modelo.h5') #Cargamos nuestro modelo ya entrenado
path = os.getcwd() #Devuelve el directorio actual del proceso 

if(opcion==1):
    #Captura de video por camara web 
    cap = cv2.VideoCapture(0) 
if(opcion==2):
    #Captura de video desde la cámara del celular
    cap = cv2.VideoCapture("http://192.168.1.107:8080/video") 


font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count=0
score=0
thicc=2
rpred=[99]
lpred=[99]

while(True):
    ret, frame = cap.read()
    height,width = frame.shape[:2] 

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Clasificador de rostro
    faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25)) 

    #Clasificador de ojo izquierdo
    left_eye = leye.detectMultiScale(gray)

    #Clasificador de ojo derecho
    right_eye =  reye.detectMultiScale(gray)

    cv2.rectangle(frame, (0,height-50) , (300,height) , (0,0,0) , thickness=cv2.FILLED )

    #Dibujamos cuadros limites para cada cara
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (25,251,58) , 1 )

    #Dibujamos y ajustamos cuadros limites para cada ojo derecho
    for (x,y,w,h) in right_eye:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (25,251,58)  , 1 )
        r_eye=frame[y:y+h,x:x+w]
        count=count+1
        r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye,(24,24))
        r_eye= r_eye/255
        r_eye=  r_eye.reshape(24,24,-1)
        r_eye = np.expand_dims(r_eye,axis=0)
        rpred = model.predict_classes(r_eye)
        if(rpred[0]==1):
            lbl='Abierto' 
        if(rpred[0]==0):
            lbl='Cerrado'
        break

    #Dibujamos y ajustamos cuadros limites para cada ojo izquierdo
    for (x,y,w,h) in left_eye:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (25,251,58)  , 1 )
        l_eye=frame[y:y+h,x:x+w]
        count=count+1
        l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
        l_eye = cv2.resize(l_eye,(24,24))
        l_eye= l_eye/255
        l_eye=l_eye.reshape(24,24,-1)
        l_eye = np.expand_dims(l_eye,axis=0)
        lpred = model.predict_classes(l_eye)
        if(lpred[0]==1):
            lbl='Abierto'   
        if(lpred[0]==0):
            lbl='Cerrado'
        break

    #Dibujar texto y calcular puntuacion
    if(rpred[0]==0 and lpred[0]==0):
        score=score+1
        cv2.putText(frame,"Cerrado",(10,height-20), font, 1,(0,0,255),1,cv2.LINE_AA)
    else:
        score=score-1
        cv2.putText(frame,"Abierto",(10,height-20), font, 1,(113,243,9),1,cv2.LINE_AA)
    
        
    if(score<0):
        score=0   
    cv2.putText(frame,'Puntuacion:'+str(score),(120,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)


    if(score>15):
        #Tomaremos un pantallazo en el momeento que el conductor se quede dormido
        cv2.imwrite(os.path.join(path,'image.jpg'),frame)


        try:
            #Reproduciremos la alarma cuando se este quedando dormido el conductor
            sound.play()
            
        except:  
            pass

        #Cuando el conductor se este quedando dormido enviaremos una notificacion por SMS
        #al celular del conductor
           # message = client.messages.create(
           # to="+573134734200", 
           # from_="+14049984421",
           # body="PELIGRO SE ESTA QUEDANDO DORMIDO")

        #En caso de que el conductor siga dormido dibujaremos un marco rojo
        #y el tamaño dependera de la puntuacion 
        if(thicc<16):
            thicc= thicc+2
        else:
            thicc=thicc-2
            if(thicc<2):
                thicc=2
        cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc) 


    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
