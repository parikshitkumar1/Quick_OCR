from keras.models import load_model
import cv2 as cv
import imutils
from imutils.contours import sort_contours
import numpy as np


model=load_model('OCRmodel.h5')

image = cv.imread('1.jpg')


gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

blurred = cv.GaussianBlur(gray, (5, 5), 0)
# perform edge detection, find contours in the edge map, and sort the
# resulting contours from left-to-right
edged = cv.Canny(blurred, 30, 150)

cnts = cv.findContours(edged.copy(), cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sort_contours(cnts, method="left-to-right")[0]
# initialize the list of contour bounding boxes and associated
# characters that we'll be OCR'ing
chars = []



# loop over the contours
for c in cnts:
	# compute the bounding box of the contour
	(x, y, w, h) = cv.boundingRect(c)
	# filter out bounding boxes, ensuring they are neither too small
	# nor too large
	if (w >= 5 and w <= 150) and (h >= 15 and h <= 120):
		# extract the character and threshold it to make the character
		# appear as *white* (foreground) on a *black* background, then
		# grab the width and height of the thresholded image
		roi = gray[y:y + h, x:x + w]
		thresh = cv.threshold(roi, 0, 255,
			cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
		(tH, tW) = thresh.shape
		# if the width is greater than the height, resize along the
		# width dimension
		if tW > tH:
			thresh = imutils.resize(thresh, width=28)
		# otherwise, resize along the height
		else:
			thresh = imutils.resize(thresh, height=28)
		# re-grab the image dimensions (now that its been resized)
		# and then determine how much we need to pad the width and
		# height such that our image will be 28x28
		
		dX = int(max(0, 28 - tW) / 2.0)
		dY = int(max(0, 28 - tH) / 2.0)
		# pad the image and force 28x28 dimensions
		padded = cv.copyMakeBorder(thresh, top=dY, bottom=dY,
			left=dX, right=dX, borderType=cv.BORDER_CONSTANT,
			value=(0, 0, 0))
		padded = cv.resize(padded, (28, 28))
		# prepare the padded image for classification via our
		# handwriting OCR model
		padded = padded.astype("float32") / 255.0
		padded = np.expand_dims(padded, axis=-1)
		# update our list of characters that will be OCR'd
		chars.append((padded, (x, y, w, h)))
   




boxes = [b[1] for b in chars]
charsd = np.array([c[0] for c in chars], dtype="float32")



preds=model.predict(charsd)

labelNames = "0123456789"
labelNames += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
labelNames = [l for l in labelNames]

for (pred, (x, y, w, h)) in zip(preds, boxes):
    # find the index of the label with the largest corresponding
    # probability, then extract the probability and label
    i = np.argmax(pred)
    prob = pred[i]
    label = labelNames[i]
    # draw the prediction on the image
    print("[INFO] {} - {:.2f}%".format(label, prob * 100))
    cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv.putText(image,str(label), (x - 10, y - 10),cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
    
    # show the image
    cv.imshow("Image", image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    


