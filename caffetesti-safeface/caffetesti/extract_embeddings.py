# Importataan käytetyt kirjastot
from imutils import paths
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

# Argumentit joita käytetään ohjelman ajoon
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
	help="path to input directory of faces + images")
ap.add_argument("-e", "--embeddings", required=True,
	help="path to output serialized db of facial embeddings")
ap.add_argument("-d", "--detector", required=True,
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", required=True,
	help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())


# Ladataan kasvot
print("[INFO] loading safeface...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

#Ladataan tunnistus ohjelma
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

#Kerrotaan kuvien sijainti
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))

# Luodaan lista nimista ja kuvista
knownEmbeddings = []
knownNames = []

# Kerrotaan prosessoidut kasvot
total = 0

# Loopataan kuvien sijainnit
for (i, imagePath) in enumerate(imagePaths):
	# extract the person name from the image path
	print("[INFO] processing image {}/{}".format(i + 1,
		len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]

	# Ladataan kuva ja uudelleen kootaan se 550 kokoiseksi
	image = cv2.imread(imagePath)
	image = imutils.resize(image, width=550)
	(h, w) = image.shape[:2]

	# Luodaan malli kuvasta
	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(image, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)

	#Kaytetaan deeplearning kansiota
	detector.setInput(imageBlob)
	detections = detector.forward()

	# Varmistu etta edes yhdet kasvot loytyivat
	if len(detections) > 0:
		#Oletetaan etta vain yhdet kasvot parhaimman tunnistuksen takaamiseksi
		i = np.argmax(detections[0, 0, :, 2])
		confidence = detections[0, 0, i, 2]

		# Otetaan pois heikot tulokset
		if confidence > args["confidence"]:
			# Lasketaan nelion sijainti
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# Otetaan kasvoista kaappaus
			face = image[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]

			# Varmistutaan etta kvua on tarpeeksi suuri
			if fW < 20 or fH < 20:
				continue

			# Luodaan kaapatusta kuvasta vertailun kohde
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
				(96, 96), (0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()

			# Lisataan naama ja arvio ja kasiteltyihin naamoihin yksi lisaa
			knownNames.append(name)
			knownEmbeddings.append(vec.flatten())
			total += 1

# Tallennetaan nimi ja kuva
print("[INFO] serializing {} encodings...".format(total))
data = {"embeddings": knownEmbeddings, "names": knownNames}
f = open(args["embeddings"], "wb")
f.write(pickle.dumps(data))
f.close()