import skimage
import numpy as np
from skimage import io, transform
import os
import shutil
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import tensorflow as tf
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import urllib.request
import urllib.error
from sklearn.utils import shuffle
from utils import label_map_util
import visualization_utils as vis_util
from scipy.misc import imsave
import cv2

PATH_TO_CKPT='fine_tuned_model/frozen_inference_graph.pb'
PATH_TO_LABELS = 'data/label_map.pbtxt'
NUM_CLASSES = 6
# PATH_TO_TEST_IMAGES_DIR = 'test'

# TEST_IMAGE_PATHS = glob.glob('test/IMG_1109.jpg')
IMAGE_SIZE = (12, 12)

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def detection_img(filepath):
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
        # new_path = 'test_1/' + os.path.basename(filepath) + '_test.jpg'
        image = Image.open(filepath)
        w, h = image.size

        w_percent = 500/w;
        new_h = int(h*w_percent);
        resized_image = image.resize((500,new_h), Image.ANTIALIAS)
          # the array based representation of the image will be used later in order to prepare the
          # result image with boxes and labels on it.
        image_np = load_image_into_numpy_array(resized_image)
          # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
          # Actual detection.
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
          # Visualization of the results of a detection.
        img, output = vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=2)

        # imsave(new_path, img)
        # cv2.imshow('Imagee', img)
        # cv2.waitKey(0)

        return output, np.array(img)
