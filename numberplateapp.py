import streamlit as st
import cv2
import numpy as np
from PIL import Image
import  imutils

st.title('Application')
st.write("Number Plate Recognition App")
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])


def number_Plate_Recognition(image):

    # Convert image into numpy array
    img_array = np.array(image)
    
    # Resize the image(width = 800)
    img = imutils.resize(img_array, width=800)
    
    # RGB to Gray scale conversion
    gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Noise removal with iterative bilateral filter
    filtered = cv2.bilateralFilter(gray_scale, 12, 20, 20)
    
    
    # Find Edges of the gray scale image
    edged = cv2.Canny(filtered, 60, 180)

    # Find contours based on Edges
    (contours, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    #sort contours based on their area keeping minimum required area as '30' (anything smaller than this will not be considered)
    contours=sorted(contours, key = cv2.contourArea, reverse = True)[:30] 
    NumberPlateCnt = None #Empty Number plate contour

    # loop over our contours to find the best possible approximate contour of number plate
    count = 0
    for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.018 * peri, True)
            if len(approx) == 4:  # Select the contour with 4 corners
                NumberPlateCnt = approx
                x,y,w,h = cv2.boundingRect(c) #finds co-ordinates of the plate
                new_img=img[y:y+h,x:x+w]
                
                break

    # Drawing the selected contour on the original image
    cv2.drawContours(img, [NumberPlateCnt], -1, (0,255,0), 2)
    
    #Display number plate detected Image
    st.image(img, caption = 'Final Image With Number Plate Detected' )
    
    #Display number plate cropped Image
    st.image(new_img,caption='cropped')

if file is None:
  st.text("Please upload an image file")
else:
  #read the image file from file_uploader
  image = Image.open(file)
  
  #Display uploaded Image
  st.image(image,caption='Original Image')
  number_Plate_Recognition(image)
 