#File designed to take the 128-D vector embeddings of the faces from the training image dataset 
#and train a model using the embeddings as the input.
import pickle
import os

from sklearn import preprocessing, svm
from sklearn.metrics import r2_score

#Classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


#load data
data = pickle.loads(open("encodings.pickle", "rb").read())
val_data = pickle.loads(open("validation_encodings.pickle", "rb").read())

#encode train names
le = preprocessing.LabelEncoder()
names = le.fit_transform(data["names"])

#encode validation names
val_names = le.fit_transform(val_data["names"])

#make the model - experiement using multiple different ones and experiment which one is better
class_name = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    svm.SVC(kernel="linear", C=1.0, probability=True),
    svm.SVC(gamma=2, C=1),
    GaussianProcessClassifier(),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]


#Classification dict that stores the scores of each classifier
#Select classifier with highest score
#Using a validation test set 
class_score = {}
for cl_name, classifier in zip(class_name, classifiers):
	classifier.fit(data["encodings"], names)
	pred = classifier.predict(val_data["encodings"])
	score = classifier.score(val_data["encodings"], val_names)
	print(f"Name: {cl_name}, Score: {score}")
	class_score[cl_name] = score


#Find models with the highest score
high_score = 0
best_models = {}
for key, value in sorted(class_score.items(), key = lambda kv: kv[1], reverse=True):
	if value >= high_score:
		high_score = value
		best_models[key] = value


#Selecting one of the best performed models and storing the model to disk 
top_model = next(iter(best_models))

#Get model 
model_index = class_name.index(top_model)
clf = classifiers[model_index] 

clf.fit(data["encodings"], names)

#Store the trained model to a pickle file
with open("model.pickle", "wb") as handle:
	pickle.dump(clf, handle, protocol = pickle.HIGHEST_PROTOCOL)

with open("le.pickle", "wb") as handle:
    pickle.dump(le, handle, protocol = pickle.HIGHEST_PROTOCOL)
