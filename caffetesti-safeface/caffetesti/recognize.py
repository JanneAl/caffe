# importtaus
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

# Ajoon kaytetyt argumentit
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-d", "--detector", required=True,
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", required=True,
	help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-r", "--recognizer", required=True,
	help="path to model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
	help="path to label encoder")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# Ladataan kasvojentunnistus
print("[INFO] loading safeface...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# Ladataan vertailtava kohde levylta
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

# lataa vertailtava kohde ja label
recognizer = pickle.loads(open(args["recognizer"], "rb").read())
le = pickle.loads(open(args["le"], "rb").read())

#Ladataan kuva ja varmistetaan paras koko 550
image = cv2.imread(args["image"])
image = imutils.resize(image, width=550)
(h, w) = image.shape[:2]

# Tehdaan kuvasta vertailtava
imageBlob = cv2.dnn.blobFromImage(
	cv2.resize(image, (300, 300)), 1.0, (300, 300),
	(104.0, 177.0, 123.0), swapRB=False, crop=False)

# Kaytetaan opencv deeplearning kirjastoa
detector.setInput(imageBlob)
detections = detector.forward()

# loopataan paremman tarkkuuden takia
for i in range(0, detections.shape[2]):
	#Tuodaan ulos arvioitu kasvojen tarkkuus
	confidence = detections[0, 0, i, 2]

	# Otetaan pois heikot tulokset
	if confidence > args["confidence"]:
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		# Otetaan ulos region of interest eli kasvot
		face = image[startY:endY, startX:endX]
		(fH, fW) = face.shape[:2]

		# Varmistutaan etta kasvot ovat tarpeeksi suuret
		if fW < 20 or fH < 20:
			continue

		# Verrataan aikaisemmin haettuun naamaan
		faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
			(0, 0, 0), swapRB=True, crop=False)
		embedder.setInput(faceBlob)
		vec = embedder.forward()

		# Tunnistetaan kasvojen luokat
		preds = recognizer.predict_proba(vec)[0]
		j = np.argmax(preds)
		proba = preds[j]
		name = le.classes_[j]

		# Piirettaan nelio
		text = "{}: {:.2f}%".format(name, proba * 100)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(image, (startX, startY), (endX, endY),
			(0, 0, 255), 2)
		cv2.putText(image, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

# Nayta loppu kuva
cv2.imshow("Image", image)
cv2.waitKey(0)