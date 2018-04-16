import numpy as np		      # importing Numpy for use w/ OpenCV
import cv2                            # importing Python OpenCV
from datetime import datetime         # importing datetime for naming files w/ timestamp

def diffImg(t0, t1, t2):              # Function to calculate difference between images.
  d1 = cv2.absdiff(t2, t1)
  d2 = cv2.absdiff(t1, t0)
  return cv2.bitwise_and(d1, d2)

threshold = 70000                     # Threshold for triggering "motion detection"
#cam = cv2.VideoCapture('http://10.100.16.61:8080/video')             # Lets initialize capture on webcam
cam = cv2.VideoCapture(0)
winName = "Movement Indicator"	      # comment to hide window
cv2.namedWindow(winName)              # comment to hide window

# Read three images first:
t_minus = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
t = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
t_plus = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
# Lets use a time check so we only take 1 pic per sec
timeCheck = datetime.now().strftime('%Ss')

while True:
  ret, frame = cam.read()	      # read from camera
  totalDiff = cv2.countNonZero(diffImg(t_minus, t, t_plus))	# this is total difference number
  text = "threshold: " + str(totalDiff)				# make a text showing total diff.
  cv2.putText(frame, text, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)   # display it on screen
  if totalDiff > threshold and timeCheck != datetime.now().strftime('%Ss'):
    dimg= cam.read()[1]
    cv2.imwrite(datetime.now().strftime('%Y%m%d_%Hh%Mm%Ss%f') + '.jpg', dimg)
  timeCheck = datetime.now().strftime('%Ss')
  # Read next image
  t_minus = t
  t = t_plus
  t_plus = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
  cv2.imshow(winName, frame)
  
  key = cv2.waitKey(10)
  if key == 27:			 # comment this 'if' to hide window
    cv2.destroyWindow(winName)
    break
