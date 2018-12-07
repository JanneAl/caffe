# importtaus
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os

# Ajoon kaytetyt argumentit
ap = argparse.ArgumentParser()
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

# Kaynnistetaan video ja kameran sensori
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Aloita framejen arviointi
fps = FPS().start()

# Looppaa videon frameja
while True:
	#Otetaan yksi frame videolta
	frame = vs.read()

	# Muutetaan frame vastmaamaan parhaaksi todettua 550
	frame = imutils.resize(frame, width=550)
	(h, w) = frame.shape[:2]

	# Tehdaan kuvasta vertailtava
	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(frame, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)

	# Kaytetaan deeplearning
	detector.setInput(imageBlob)
	detections = detector.forward()

	# loopataan paremman tarkkuuden takia
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the prediction
		confidence = detections[0, 0, i, 2]

		# pudotetaan huonot tulokset
		if confidence > args["confidence"]:
			# Laketaan naaman alue neliota varten
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# Otetaan ulos region of interest eli roi
			face = frame[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]

			# Varmistutaan siita, etta kuva on tarpeeksi iso
			if fW < 20 or fH < 20:
				continue

			# Verrataan aikaisemmin haettuun naamaan
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
				(96, 96), (0, 0, 0), swapRB=True, crop=False)
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
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				(0, 0, 255), 2)
			cv2.putText(frame, text, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

	# Siirrytaan seuraavaan frameen
	fps.update()

	# Nayta missa framessa mennaan
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# Jos painetaan q painiketta ohjelma sujetaan
	if key == ord("q"):
		break

# Lopetetaan ajaaston ja suljetaan
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# Suljetaan kaikki
cv2.destroyAllWindows()
vs.stop()