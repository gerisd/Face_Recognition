import pickle
import signal
import sys
import numpy as np
import time

import cv2
import face_recognition
import imutils

def signal_handler(sig, frame):
	print("[EXITING] You pressed CTRL+C...")
	sys.exit(0)

#load model and label encoder
model = pickle.loads(open("model.pickle", "rb").read())
le = pickle.loads(open("le.pickle", "rb").read())

#load Haar cascade classifier for facial bounding box detection  
cascade = cv2.CascadeClassifier("../cascade/haarcascade_frontalface_default.xml")

#Initalize MOG background subtractor
mog = cv2.bgsegm.createBackgroundSubtractorMOG()

#keep track of current and prev person 
curPerson = None
prevPerson = None

#how many consec frames do you want to check if its the same person
consecFrames = 1

frameArea = None

#signal.signal traps signals with 2 arguments (# of signals you want to trap, name of signal handler)
#Capturing SIGINT with CTRL+C
signal.signal(signal.SIGINT, signal_handler)
print("Press CTRL+C to exit")
#signal.pause() #wait for a signal

#Load Video stream for camera
vs = cv2.VideoStream(usePiCamera=True).start()
time.sleep(2.0)

#Loop through camera frames
while True:
	frame = vs.read()
	#Resize frame
	frame = imutils.resize(frame, width=500)

	#convert to grayscale
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	#convert to RGB for face_recognition
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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

cv2.destroyAllWindows()
vs.stop()
