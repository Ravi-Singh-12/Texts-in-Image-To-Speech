from cv2 import displayOverlay
from imutils.object_detection import non_max_suppression
import numpy as np
import pytesseract
import argparse
import cv2
from sympy import true
from textblob import TextBlob
from gtts import gTTS
import os
import playsound
import re
import random
import matplotlib.pyplot as plt
import os

class Laplace(Exception):
	"""Image has a high degree of blur, aborting further processing"""

img_fpath = None
padding = 0.03
args = {
		'image': img_fpath, 
		'min_confidence': 0.04, 
		'height': 320, 
		'width': 320, 
		'east': r"C:\Users\yngwi\OneDrive\Desktop\textDetection\frozen_east_text_detection.pb", 
		'padding': padding,
		'min_confidence': 0.5
	}

def decode_predictions(scores, geometry):
	# grab the number of rows and columns from the scores volume, then
	# initialize our set of bounding box rectangles and corresponding
	# confidence scores
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []
	# loop over the number of rows
	for y in range(0, numRows):
		# extract the scores (probabilities), followed by the
		# geometrical data used to derive potential bounding box
		# coordinates that surround text
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]
		# loop over the number of columns
		for x in range(0, numCols):
			# if our score does not have sufficient probability,
			# ignore it
			if scoresData[x] < args["min_confidence"]:
				continue
			# compute the offset factor as our resulting feature
			# maps will be 4x smaller than the input image
			(offsetX, offsetY) = (x * 4.0, y * 4.0)
			# extract the rotation angle for the prediction and
			# then compute the sin and cosine
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)
			# use the geometry volume to derive the width and height
			# of the bounding box
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]
			# compute both the starting and ending (x, y)-coordinates
			# for the text prediction bounding box
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)
			# add the bounding box coordinates and probability score
			# to our respective lists
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])
	# return a tuple of the bounding boxes and associated confidences
	return (rects, confidences)

 

def main(img_fpath):

	image = cv2.imread(args["image"])
	orig = image.copy()
	(origH, origW) = image.shape[:2]
	# set the new width and height and then determine the ratio in change
	# for both the width and height
	(newW, newH) = (args["width"], args["height"])
	rW = origW / float(newW)
	rH = origH / float(newH)
	# resize the image and grab the new image dimensions
	image = cv2.resize(image, (newW, newH))
	(H, W) = image.shape[:2]

	
		
		
	layerNames = [
		"feature_fusion/Conv_7/Sigmoid",
		"feature_fusion/concat_3"]
	# load the pre-trained EAST text detector

	print("[INFO] loading EAST text detector...")
	net = cv2.dnn.readNet(args["east"])

	blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
		(123.68, 116.78, 103.94), swapRB=True, crop=False)
	net.setInput(blob)
	(scores, geometry) = net.forward(layerNames)


	(rects, confidences) = decode_predictions(scores, geometry)
	boxes = non_max_suppression(np.array(rects), probs=confidences)
	results = []
	# loop over the bounding boxes

	for (startX, startY, endX, endY) in boxes:
		# scale the bounding box coordinates based on the respective
		# ratios
		startX = int(startX * rW)
		startY = int(startY * rH)
		endX = int(endX * rW)
		endY = int(endY * rH)
		# in order to obtain a better OCR of the text we can potentially
		# apply a bit of padding surrounding the bounding box -- here we
		# are computing the deltas in both the x and y directions
		dX = int((endX - startX) * args["padding"])
		dY = int((endY - startY) * args["padding"])
		# apply padding to each side of the bounding box, respectively
		startX = max(0, startX - dX)
		startY = max(0, startY - dY)
		endX = min(origW, endX + (dX * 2))
		endY = min(origH, endY + (dY * 2))
		# extract the actual padded ROI
		roi = orig[startY:endY, startX:endX]

		roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
		roi = cv2.medianBlur(roi, 3)
		roi = cv2.bilateralFilter(roi,9,75,75)
		_, roi = cv2.threshold(roi, 127, 255, cv2.THRESH_OTSU)
		m, n = roi.shape
		black = cv2.countNonZero(roi)

		if (black < m*n-black):
			roi = cv2.bitwise_not(roi)

		config = ("-c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRTUVWXYZ -l eng --oem 1 --psm 7")
		text = pytesseract.image_to_string(roi, config=config)

		results.append(((startX, startY, endX, endY), text))

	results = sorted(results, key=lambda r:r[0][1])
	# loop over the results
	res_text = []

	output = orig.copy()
	for ((startX, startY, endX, endY), text) in results:


		res_text.append(text[:-2])

		
		text = "".join([c if ord(c) < 128 else "" for c in text]).strip()

		cv2.rectangle(output, (startX, startY), (endX, endY),
			(0, 0, 255), 2)
		cv2.putText(output, text, (startX, startY - 20),
			cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)


	res_string = ''

	for word in res_text:
		
		
		word = re.sub("\W", "", word)
		res_string+=(word+' ')

	res_string = res_string.lower()


	tb = TextBlob(res_string)

	final_text = str(tb.correct())

	fname = ""

	for i in range(10):
		c = chr(random.randint(65, 90))
		fname += c
	fname += '.mp3'
	myobj = gTTS(text=final_text, lang='en', slow=True)
	myobj.save(fname)


	audio_file = [fname, myobj]
	
	return final_text, audio_file, cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

RUN = True
prev = ''

while RUN:
	os.system('cls')
	if prev != '':
		print(f'Previous file entered: {prev}')
	
	img_fpath = input('Enter image path(jpg, jpeg, jfif, png): ')
	args['image'] = img_fpath

	
	try:
		test_img = cv2.imread(img_fpath)
		if cv2.Laplacian(cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()<50:
			raise Laplace(Exception)
	

		text, audio_file, image = main(img_fpath)

		run_loop = True
		while run_loop:
			print('1. Show image \n2. Play audio \n3. Take a new image\n------------')
			
			k = input('Enter option:')

			if k == '1':
				plt.figure(figsize = (20, 18))
				plt.imshow(image)
				plt.xticks([])
				plt.yticks([])
				plt.show()
				enter = input()
				

			elif k == '2':
				print(text)
				playsound.playsound(audio_file[0])
				enter = input('Press enter to continue')


			else:
				prev = img_fpath
				break
	except Laplace as ex:
		print('High degree of blur detected, aborting further processing...')
		enter = input('Press Enter to continue')
	except Exception:
		print('An exception occurred, either wrog image or image is not viable for recognition')
		enter = input('Press enter to continue')


					

