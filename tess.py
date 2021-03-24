import cv2
import pytesseract
import numpy as np
from PIL import ImageGrab
import time

yes = input(">>Input image path or if it's in the same directory simply input the name,(try 2.png) ")

x = input(">>Are you using windows or linux? input 0 for windows and 1 for linux  ")

y = int(x)

if y == 0:
	pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

else:
	pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'


img = cv2.imread(yes)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   #CONVERTING THE IMAGE TO RGB BECAUSE OPENCV TAKES IMAGE IN BGR

pytesseract

scale = 1 

imageWidth = img.shape[1]/1.5
imageHeight = img.shape[0]/1.5

fontScale = (imageWidth * imageHeight) / (1000 * 1000)    ### THIS IS TO ADJUST THE TEXT SIZE ACCORDING TO THE IMAGE SIZE


boxes = pytesseract.image_to_data(img)
for a,b in enumerate(boxes.splitlines()):
         print(b)
         if a!=0:
             b = b.split()
             if len(b)==12:
                 x,y,w,h = int(b[6]),int(b[7]),int(b[8]),int(b[9])
                 cv2.putText(img,b[11],(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,fontScale,(0,0,255),1)
                 


cv2.imshow('press 0 to exit', img)     
                                                                                                                                                                                        
cv2.waitKey(0)