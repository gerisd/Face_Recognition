#This script performs facial recognition for images rather than camera
import pickle
import signal
import sys
import os
import numpy as np
import time

import cv2
import face_recognition
import imutils
from imutils import paths

#Get image path
image_path = os.getcwd() + "/test dataset/Test"	

#Grab path to input images of dataset
imagePaths = list(paths.list_images(image_path))

#load model and label encoder
model = pickle.loads(open("preprocess/model.pickle", "rb").read())
le = pickle.loads(open("preprocess/le.pickle", "rb").read())

#load Haar cascade classifier for facial bounding box detection  
cascade = cv2.CascadeClassifier("cascade/haarcascade_frontalface_default.xml")

#Initalize MOG background subtractor
mog = cv2.bgsegm.createBackgroundSubtractorMOG()

#Current person in frame
curPerson = None

#how many consec frames do you want to check if its the same person
consecFrames = 1

#Option to display images
display_image = True

#Loop through images
for imagePath in imagePaths:

    #load image and resize
	image = cv2.imread(imagePath)
	image = imutils.resize(image, width=500)
	
	#convert the image to grayscale (for face detection)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #convert to RGB for face recognition
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #facial detection returning bounding box parameters (x,y,w,h)
	parameters = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))

	#Reconstruct parameters
	face_box = [(y, x+w, y+h, x) for (x,y,w,h) in parameters]

    #Check if there is a face
	if len(face_box) > 0:
		#get the face encodings for the model to predict on it
		encodings = face_recognition.face_encodings(rgb, face_box)

		prediction = model.predict_proba(encodings)[0]
		index = np.argmax(prediction)
		curPerson = le.classes_[index]

		#Draw bounding box around the face with the person's name
		(top, right, bottom, left) = face_box[0]
		cv2.rectangle(image, (left, top), (right,bottom), (0, 255, 0), 2)
        
		y = top - 15 if top - 15 > 15 else top + 15
		cv2.putText(image, curPerson, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

		# check if the person is an intruder
		if curPerson == "unknown":
			print(f"Unknown Person")

	#Else no face was detected
	else:
		print(f"[SYSTEM] NO FACE DETECTED...")

	if display_image == True:
		cv2.imshow("Image", image)
		#cv2.waitKey(0)
		key = cv2.waitKey(1) & 0xFF

		#if the 'q' key is pressed, break from the loop
		if key == ord("q"):
			break

	time.sleep(0.5)		
#cv2.destroyAllWindows()
