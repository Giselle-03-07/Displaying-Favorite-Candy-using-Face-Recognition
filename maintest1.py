# importing libraries
import tkinter as tk
from tkinter import *
import cv2
import numpy as np
import os
from setuptools import setup
import pickle
from PIL import Image

face_cascade = cv2.CascadeClassifier('C:/Users/Giselle/Desktop/Why/cascades/data/haarcascade_frontalface_alt2.xml')

def faceRecord():

    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('C:/Users/Giselle/Desktop/Why/Faces/Output.avi',fourcc, 20.0, (640,480))

    while True:
        ret, frame = cap.read()
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out.write(frame)
        cv2.imshow("image", frame)

        k = cv2.waitKey(30) & 0xff
        if k == 27: # press 'ESC' to quit
            break


    cap.release()
    out.release()
    cv2.destroyAllWindows()

def Convert():
    face_cascade = cv2.CascadeClassifier('C:/Users/Giselle/Desktop/Why/cascades/data/haarcascade_frontalface_alt2.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    # Playing video from file:
    cap = cv2.VideoCapture('C:/Users/Giselle/Desktop/Why/Faces/Output.avi')

    CandyName = input("Enter your favorite Candy: ")

    try:
        if not os.path.exists('C:/Users/Giselle/Desktop/Why/Faces/Images/' + str(CandyName)):
            add = os.makedirs('C:/Users/Giselle/Desktop/Why/Faces/Images/' + str(CandyName))
    except OSError:
        print ('Error: Creating directory of data')

    currentFrame = 0
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Saves image of the current frame in jpg file
        name = 'C:/Users/Giselle/Desktop/Why/Faces/Images/' + str(CandyName) +'/'+ str(currentFrame) + '.jpg'
        print ('Creating...' + name)
        cv2.imwrite(name, frame)

        # To stop duplicate images
        currentFrame += 1


    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def train_test():
    face_cascade = cv2.CascadeClassifier('C:/Users/Giselle/Desktop/Why/cascades/data/haarcascade_frontalface_alt2.xml')
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = BASE_DIR+'\Faces'
    image_dir = os.path.join(BASE_DIR, "Images")

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    ##Loading images and cropping the image
    i = 0
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith("png") or file.endswith("jpg"):
                path = os.path.join(root, file)
                img = cv2.imread(path)
                img = cv2.resize(img, (280, 280))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                        img_name = str(root)+"/"+str(i)+".jpg"
                        print(img_name)
                        roi_color = img[y:y+h+20, x:x+w+20]
                        cv2.imwrite(img_name, roi_color)
                        #cv2.imshow("Frame",roi_color)
                        i += 1
                cv2.waitKey(0)

    current_id = 0
    label_ids = {}
    y_labels = []
    x_train = []

    for root, dirs, files in os.walk(image_dir):
    	for file in files:
    		if file.endswith("png") or file.endswith("jpg"):
    			path = os.path.join(root, file)
    			label = os.path.basename(root).replace(" ","-").lower()
    			#print(label,path)

    			if not label in label_ids:
    				label_ids[label] = current_id
    				current_id += 1
    			id_ = label_ids[label]
    			print(label_ids)

    			pil_image = Image.open(path).convert("L") # grayscale
    			#size = (550, 550)
    			#final_image = pil_image.resize(size, Image.ANTIALIAS)
    			image_array = np.array(pil_image, "uint8")
    			#print(image_array)
    			faces = face_cascade.detectMultiScale(image_array, 1.5, 5)

    			for (x,y,w,h) in faces:
    				roi = image_array[y:y+h, x:x+w]
    				x_train.append(roi)
    				y_labels.append(id_)

    #print(y_labels)
    #print(x_train)

    with open("labels.pickle", 'wb') as f:
    	pickle.dump(label_ids, f)

    recognizer.train(x_train, np.array(y_labels))
    recognizer.save("trainner.yml")


def Camera():
    face_cascade = cv2.CascadeClassifier('C:/Users/Giselle/Desktop/Why/cascades/data/haarcascade_frontalface_alt2.xml')

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trainner.yml")

    labels = {"person_name": 1}
    with open("labels.pickle", 'rb') as f:
    	og_labels = pickle.load(f)
    	labels = {v:k for k,v in og_labels.items()}

    cap = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
        for (x,y,w,h) in faces:
        	#print(x,y,w,h)
        	roi_gray = gray[y:y+h, x:x+w]      #(ycord_start , ycord_end)
        	roi_color = frame[y:y+h, x:x+w]

        	#RECOGNIZE?
        	id_, conf = recognizer.predict(roi_gray)
        	#if conf>=70:
        	#	print(id_)
        	#print(labels[id_])
        	font = cv2.FONT_HERSHEY_SIMPLEX
        	name = labels[id_]
        	color = (255,255,255)
        	stroke = 2
        	cv2.putText(frame, name, (x,y-10), font, 1, color, stroke, cv2.LINE_AA)

        	img_item = "my-image.png"
        	cv2.imwrite(img_item, roi_gray)

        	color =(255,0,0)  #BGR
        	stroke = 2
        	end_cord_x = x + w
        	end_cord_y = y + h
        	cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

        # Display the resulting frame
        cv2.imshow('frame',frame)
        #cv2.imshow('gray',gray)
        k = cv2.waitKey(30) & 0xff
        if k == 27: # press 'ESC' to quit
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

root = Tk()

root.title("Favourite Candy Predictor", )
root.geometry('625x625')
tf = Frame(root, width = 600, height = 100, bd = 2, relief = "ridge")
tf.place(x = 10, y = 10)
title = Label(tf,font = ("Helvetica",30),fg = "blue",text = "      Favourite Candy Dispenser     ")
title.pack()

button1 = Button(root, command = faceRecord, padx = 5, pady = 5, relief = RIDGE, width = 33, background = 'orange',font  = ("georgia",15), text = "Open webcam for personal face detection", bd = 5)
button1.place(x = 100, y = 80)


button2 = Button(root, command = Convert, padx = 5, pady = 5, relief = RIDGE, width = 33, background = 'orange',font  = ("georgia",15), text = "Convertion", bd = 5)
button2.place(x = 100, y = 200)


button3 = Button(root, command = train_test, padx = 5, pady = 5, relief = RIDGE, width = 33, background = 'orange',font  = ("georgia",15), text = "Training and Testing", bd = 5)
button3.place(x = 100, y = 320)

button4 = Button(root, command = Camera, padx = 5, pady = 5, relief = RIDGE, width = 33, background = 'orange',font  = ("georgia",15), text = "Detection and Recognition", bd = 5)
button4.place(x = 100, y = 440)

root.mainloop()
