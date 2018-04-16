
# coding: utf-8
print('ok')
# # Object Detection Demo
# Welcome to the object detection inference walkthrough!  This notebook will walk you step by step through the process of using a pre-trained model to detect objects in an image. Make sure to follow the [installation instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) before you start.

# # Imports

# In[ ]:


import numpy as np
import os
#import six.moves.urllib as urllib
import sys
#import tarfile
import tensorflow as tf
#import zipfile

#from collections import defaultdict
#from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

#if tf.__version__ != '1.4.0':
#  raise ImportError('Please upgrade your tensorflow installation to v1.4.0!')
#tesseract imports
import cv2
import pytesseract
# ## Env setup

# In[ ]:


# This is needed to display the images.
get_ipython().magic('matplotlib inline')

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")


# ## Object detection imports
# Here are the imports from the object detection module.

# In[ ]:


from utils import label_map_util

from utils import visualization_utils as vis_util


# # Model preparation 

# ## Variables
# 
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.  
# 
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# In[ ]:


# What model to download.
MODEL_NAME = 'numplate'
#MODEL_FILE = MODEL_NAME + '.tar.gz'
#DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/graph-200000/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('training', 'object-detection.pbtxt')

NUM_CLASSES = 1


# ## Download Model

# In[ ]:


# opener = urllib.request.URLopener()
# opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
# tar_file = tarfile.open(MODEL_FILE)
# for file in tar_file.getmembers():
#  file_name = os.path.basename(file.name)
#  if 'frozen_inference_graph.pb' in file_name:
#    tar_file.extract(file, os.getcwd())


# ## Load a (frozen) Tensorflow model into memory.

# In[ ]:


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

# In[ ]:


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# ## Helper code

# In[ ]:


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# # Detection

# In[ ]:


# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'png_tesseract/test_tesseract'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 14) ]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)
TEST_DHARUN=os.path.join('numplate')
count = 0
# In[ ]:


with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    for image_path in TEST_IMAGE_PATHS:
      image = Image.open(image_path)
      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      image_np = load_image_into_numpy_array(image)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      # Actual detection.
      (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      #numpy.asarray(Image.open('1-enhanced.png').convert('L'))
      # Visualization of the results of a detection
#      tf.image.draw_bounding_boxes(image_np, boxes, name=None)
#      print(np.squeeze(boxes))
#      print(category_index)
#      print(boxes.shape[0])
#      print(100*scores[0])
#      ase=tf.constant(detection_scores,tf.float32)
#      print(sess.run(ase))
#      vis_util.encode_image_array_as_png_str(image_np)
#      vis_util.save_image_array_as_png(image_np,TEST_DHARUN)
#      width, height = image.size
#      print(height)
#      ymin = boxes[0][i][0]*height
#      xmin = boxes[0][i][1]*width
#      ymax = boxes[0][i][2]*height
#      xmax = boxes[0][i][3]*width
#      print(ymin)
#      image=tf.image.convert_image_dtype(image,dtype=float,saturate=False,name=None)

      ymin = boxes[0,0,0]
      xmin = boxes[0,0,1]
      ymax = boxes[0,0,2]
      xmax = boxes[0,0,3]
      (im_width, im_height) = image.size
      (xminn, xmaxx, yminn, ymaxx) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
#      print(ymin,xmin,ymax,xmax)
      xxminn=int(xminn)
      xxmaxx=int(xmaxx)
      yyminn=int(yminn)
      yymaxx=int(ymaxx)
#      print(xxminn,xxmaxx,yyminn,yymaxx)
      cropped_image = tf.image.crop_to_bounding_box(image_np, int(yminn), int(xminn),int(ymaxx - yminn), int(xmaxx - xminn))
#      cropped_image = tf.image.crop_to_bounding_box(image, yyminn, xxminn, yymaxx - yyminn, xxmaxx - xxminn)
#      output_image = tf.image.encode_png(cropped_image)
#      sess = tf.Session()
      # tesseract part
      img_data = sess.run(cropped_image)
      
      gray = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
      gray = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
      gray = cv2.medianBlur(gray, 3)
      path_png='png_tesseract'
      count += 1
      filename = os.path.join(path_png,'image{}.png'.format(count))
      cv2.imwrite(filename, gray)
      pytesseract.tesseract_cmd = '/home/tensorflow-cuda/tesseract-master/tessdata/'
      text = pytesseract.image_to_string(Image.open(filename),lang=None) 
##      os.remove(filename)
      print('NUM PLATE : ',text)
      with open("Output1.txt","a") as text_file:
        text_file.write("%s\n\n" % (text))
      # End of tesseract 
#      sess.close()
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=5)
      
      
      plt.figure(figsize=IMAGE_SIZE)
      plt.imshow(image_np)
#      plt.imshow(img_data)
#      plt.imshow(output_image)
#      plt.show()
