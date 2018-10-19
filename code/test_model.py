# USAGE
# python test_network.py --model santa_not_santa.model --image images/examples/santa_01.png

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
import os

# load the trained convolutional neural network
print("[INFO] loading network...")
model = load_model(os.getcwd()+"/output.model")
wardrobe_path = os.getcwd()+"/wardrobe"
cloth_paths = []
for cloth in os.listdir(wardrobe_path):
	cloth_path = wardrobe_path+"/"+cloth
	cloth_paths.append(cloth_path)
output_imgs = []
for cp in cloth_paths:
	# load the image
	image = cv2.imread(cp)
	orig = image.copy()

	# pre-process the image for classification
	image = cv2.resize(image, (28, 28))
	image = image.astype("float") / 255.0
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)

	# classify the input image
	(tshirt, bra, dress, hoodie, jeans, shorts) = model.predict(image)[0]
	all_prob = [tshirt, bra, dress, hoodie, jeans, shorts]
	cat = max(all_prob)
	cat_name_i = all_prob.index(cat)
	cat_name_dic = {0:"t-shirt", 1:"sports bra",2:"dress",3:"hoodie",4:"jeans",5:"running shorts"}
	
	label = cat_name_dic[cat_name_i]
	prob = cat
	summary = "{}: {:.2f}%".format(label, prob * 100)

	# draw the label on the image
	output = imutils.resize(orig, width=400)
	cv2.putText(output, summary, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
		0.7, (0, 255, 0), 2)
	output_imgs.append(output)


cv2.imshow("Output", output_imgs[5])
cv2.waitKey(0)