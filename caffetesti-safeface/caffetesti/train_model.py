# Importataan käytetyt kirjastot
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse
import pickle

# Argumentit joita käytetään ohjelman ajoon
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--embeddings", required=True,
	help="path to serialized db of facial embeddings")
ap.add_argument("-r", "--recognizer", required=True,
	help="path to output model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
	help="path to output label encoder")
args = vars(ap.parse_args())

# Ladataan kasvot
print("[INFO] loading face embeddings...")
data = pickle.loads(open(args["embeddings"], "rb").read())

# Annetaan labelit kasvoille
print("[INFO] encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

# treenataan ohjelmaa
print("[INFO] training model...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)

# Kirjoitetaan kasvojentunnistus levylle kayttaen picklea
f = open(args["recognizer"], "wb")
f.write(pickle.dumps(recognizer))
f.close()

# Kirjoitetaan labeli levylle
f = open(args["le"], "wb")
f.write(pickle.dumps(le))
f.close()