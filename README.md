# caffe

Testing Opencv and Deep learning with Caffe.

credit to Trung Tran https://chunml.github.io/ChunML.github.io/project/Installing-Caffe-Ubuntu/
And Adrian Rosenbrock https://www.pyimagesearch.com/2017/08/21/deep-learning-with-opencv/
Whose guides I followed to test out Caffe and learn about deeplearning

    Usage:

     	#Save dataset

python extract_embeddings.py --dataset dataset \
	--embeddings output/embeddings.pickle \
	--detector face_detection_model \
	--embedding-model openface_nn4.small2.v1.t7
	
    	 #Training model
		  
python train_model.py --embeddings output/embeddings.pickle \
	--recognizer output/recognizer.pickle \
	--le output/le.pickle
	
	#for picture insert image path

python recognize.py --detector face_detection_model \
	--embedding-model openface_nn4.small2.v1.t7 \
	--recognizer output/recognizer.pickle \
	--le output/le.pickle \
	--image images/kuva.jpg

      #Face reg of Video/ webcam

python recognize_video.py --detector face_detection_model \
	--embedding-model openface_nn4.small2.v1.t7 \
	--recognizer output/recognizer.pickle \
	--le output/le.pickle
