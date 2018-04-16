import urllib
import cv2
import numpy as np

url='http://10.100.17.172:8080/shot.jpg'

while True:
    imgResp=urllib.request.urlopen(url)
    imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
    img=cv2.imdecode(imgNp,-1)

    # all the opencv processing is done here
    cv2.imshow('test',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break