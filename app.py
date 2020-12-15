
#write script in app.py

import streamlit as st
import os
import collections
import pandas as pd
import cv2 
import matplotlib.pyplot as plt 
import numpy as np
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import pathlib
import tensorflow as tf
tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import imutils
from object_detection.utils import ops as utils_ops
import warnings
warnings.filterwarnings('ignore')

def resize_image(img_path):
  img = Image.open(img_path) # image extension *.png,*.jpg
  new_width  = 640
  new_height = 640
  img_resize = img.resize((new_width, new_height), Image.ANTIALIAS)
  return np.asarray(img_resize)
  #img_resize.save('test_image.jpg')
  #plt.figure(figsize=(5,5))
  #plt.imshow(img_resize)



def final_pipeline1(test_img_path,detect_fn):

  utils_ops.tf = tf.compat.v1
  tf.gfile = tf.io.gfile
  PATH_TO_LABELS = '/content/drive/MyDrive/label_map.pbtxt'

  category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                      use_display_name=True)


  print('Running inference... ')

  image_np = resize_image(test_img_path)
  #print('\nimage shape : ', image_np.shape)
  # Things to try:
  # Flip horizontally
  # image_np = np.fliplr(image_np).copy()

  # Convert image to grayscale
  # image_np = np.tile(
  #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image_np)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  #print(input_tensor.shape)
  input_tensor = input_tensor[tf.newaxis, ...]
  #print(input_tensor.shape)
  # input_tensor = np.expand_dims(image_np, 0)
  detections = detect_fn(input_tensor)
  #print(detections)
  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(detections.pop('num_detections'))
  detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
  #print('detections : ',detections)
  detections['num_detections'] = num_detections
  detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
  #print(detections)

  image_np_with_detections = image_np.copy()

  viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.29,
        agnostic_mode=False)

  #plt.figure(figsize=(10,10))
  #plt.title('Detected Image : ')
  #plt.imshow(image_np_with_detections)
  st.text('Damage Detected Image : ')
  st.image(image_np_with_detections,caption = 'Detected Image')
  #print('Done')
  #plt.show()
@st.cache(allow_output_mutation = True)
def load_model():
  PATH_TO_MODEL_DIR = '/content/drive/MyDrive/object_detection/mobile_ner_v2_fpnlite_640x640/exporter2'

  PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"
  # Load saved model and build the detection function
  detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
  return detect_fn

st.set_option('deprecation.showfileUploaderEncoding',False)
st.title('Road Damage Detection App')
st.text('Build With Streamlit')

with st.spinner('Loading model ......'):
  detect_fn = load_model()

#upload image:
image_file = st.file_uploader('Upload Image',type=['jpg','png','jpeg'])
#inference:
if image_file is not None:
    our_image = Image.open(image_file)
    st.text('Original Image :')
    st.image(our_image)
    test_img_path =image_file
    with st.spinner('Detecting Road Damage ......'):
      final_pipeline1(test_img_path,detect_fn)