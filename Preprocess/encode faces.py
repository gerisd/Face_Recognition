import face_recognition
import cv2
import pickle
import os
from tqdm import tqdm

#Get input images path
image_path = r"../dataset" #Path to train dataset
#image_path = r"../validation dataset" #Path to validation dataset

directory = [dr for dr in os.listdir(image_path)]
fileNames = [os.path.sep.join([image_path, dr]) for dr in directory]

#Keep track of name and encodings
names = []
encodings = []


#Go through the all the images for each person
for fileName in fileNames:
	files = list(os.listdir(fileName))

	#Get person name for labeling
	name = os.path.split(fileName)[-1]

	for image in tqdm(files):
		img = os.path.sep.join([fileName, image])

		#face_recognition uses RGB images
		image = cv2.imread(img)
		rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		#face_location is an array containing the coordinates of the face
		face_location = face_recognition.face_locations(rgb, model='cnn')
		encoding = face_recognition.face_encodings(rgb, face_location)

		#encoding contains a list within a set within a list [([])]
		#Extract the encoding and store it into encoding list with person's corresponding name
		for enc in encoding:
			encodings.append(enc)
			names.append(name)

#Store Encodings and names in it pickle for model training
features = {"names": names, "encodings": encodings}

with open("encodings.pickle", "wb") as handle:
	pickle.dump(features, handle, protocol = pickle.HIGHEST_PROTOCOL)
