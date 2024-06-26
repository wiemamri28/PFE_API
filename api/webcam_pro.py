
import os
import cv2
import numpy as np
#import tensorflow as tf
from tensorflow import compat as ttf
tf = ttf.v1
import sys
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'
tf.disable_v2_behavior()
# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,"api\modeloffline",'saved_model.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'api\modeloffline','labelmap.pbtxt')
print(PATH_TO_LABELS)
# Number of classes the object detector can identify
NUM_CLASSES = 38

## Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `guava`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

print("categories" , categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

    # Define input and output tensors (i.e. data) for the object detection classifier

    # Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Output tensors are the detection boxes, scores, and classes
    # Each box represents a part of the image where a particular object was detected
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')


def visualize_test(frame, boxes, classes, scores):
    try:
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.85
        )
    except Exception as e:
        print(f"Visualization Error: {e}")


# Initialize webcam feed
#video = cv2.VideoCapture("rtsp://admin:admin@192.168.10.30:554/play1.sdp")
video = cv2.VideoCapture(0)
ret = video.set(3,640)
ret = video.set(4,480)

while True:
    try:
        # Acquire frame and expand frame dimensions
        ret, frame = video.read()
        frame_expanded = np.expand_dims(frame, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded}
        )

        # Debug prints
        print(f"Boxes: {boxes}")
        print(f"Scores: {scores}")
        print(f"Classes: {classes}")
        print(f"Num: {num}")

        # Draw the results of the detection
        visualize_test(frame, boxes, classes, scores)

        # Display the resulting frame
        cv2.imshow('Object detector', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break
    
    except Exception as e:
        print(f"Error in the while loop: {e}")
        break


# Clean up
video.release()
cv2.destroyAllWindows()
