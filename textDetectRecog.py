from imutils.object_detection import non_max_suppression
import numpy as np
import time
import cv2
import pytesseract

net = cv2.dnn.readNet("frozen_east_text_detection.pb")


def text_detector(image):
	orig = image
	(H, W) = image.shape[:2]
    
	(newW, newH) = (320, 320)
	rW = W / float(newW)
	rH = H / float(newH)
	#print(rW,rH,H,W)

	image = cv2.resize(image, (newW, newH))
	(H, W) = image.shape[:2]
	#print(H,W)

	layerNames = [
		"feature_fusion/Conv_7/Sigmoid",
		"feature_fusion/concat_3"]


	blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
		(123.68, 116.78, 103.94), swapRB=True, crop=False)

	net.setInput(blob)
	(scores, geometry) = net.forward(layerNames)

	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []

	for y in range(0, numRows):

		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]

		# loop over the number of columns
		for x in range(0, numCols):
			# if our score does not have sufficient probability, ignore it
			if scoresData[x] < 0.5:
				continue

			# compute the offset factor as our resulting feature maps will
			# be 4x smaller than the input image
			(offsetX, offsetY) = (x * 4.0, y * 4.0)

			# extract the rotation angle for the prediction and then
			# compute the sin and cosine
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)

			# use the geometry volume to derive the width and height of
			# the bounding box
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]

			# compute both the starting and ending (x, y)-coordinates for
			# the text prediction bounding box
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)

			# add the bounding box coordinates and probability score to
			# our respective lists
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])

	boxes = non_max_suppression(np.array(rects), probs=confidences)

	for (startX, startY, endX, endY) in boxes:

		startX = int(startX * rW)
		startY = int(startY * rH)
		endX = int(endX * rW)
		endY = int(endY * rH)
		boundary = 2

		text = orig[startY-boundary:endY+boundary, startX - boundary:endX + boundary]
		text = cv2.cvtColor(text.astype(np.uint8), cv2.COLOR_BGR2GRAY)
		textRecongized = pytesseract.image_to_string(text)
		cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 3)
		#print(textRecongized)
		#cleaned_text = " ".join(textRecongized.split("\n"))
		#print(cleaned_text)
		#orig = cv2.putText(orig, textRecongized, (endX,endY+5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA) 
	return orig

image0 = cv2.imread('image0.jpg')
image1 = cv2.imread('image1.jpg')
image2 = cv2.imread('image2.jpg')
image3 = cv2.imread('image3.jpg')
image4 = cv2.imread('image4.jpg')
image5 = cv2.imread('image5.jpg')
image6 = cv2.imread('image6.jpg')
image7 = cv2.imread('image7.jpg')
image8 = cv2.imread('image8.jpg')
image9 = cv2.imread('image9.jpg')
image10 = cv2.imread('image10.jpg')
image11 = cv2.imread('image11.jpg')
image12 = cv2.imread('image12.jpg')
image13 = cv2.imread('image13.jpg')
image14 = cv2.imread('image14.jpg')
image15 = cv2.imread('image15.jpg')
image16 = cv2.imread('image16.jpg')
#image17 = cv2.imread('image1.png')
#image18 = cv2.imread('image2.png')
image16 = cv2.imread('image16.jpg')
#image17 = cv2.imread('image17.jpg')
image18 = cv2.imread('image18.jpg')
image19 = cv2.imread('image19.jpg')
image20 = cv2.imread('image20.jpg')
image21 = cv2.imread('image21.png')


#array = [image0,image11,image12,image13,image14,image15,image16,image0]
#array = [image0,image1,image2,image12,image13,image3,image4,image5,image6,image7,image8,image9,image10]
array = [image18,image19,image20,image21,image0]
for i in range(0,1):
	for img in array:
		imageO = cv2.resize(img, (640,320), interpolation = cv2.INTER_AREA)
		orig = cv2.resize(img, (640,320), interpolation = cv2.INTER_AREA)
		textDetected = text_detector(imageO)
		cv2.imshow("Orig Image", orig)
		cv2.imshow("Text Detection", textDetected)
		time.sleep(2)
		k = cv2.waitKey(0)
		if k == 27:
			break
cv2.destroyAllWindows()