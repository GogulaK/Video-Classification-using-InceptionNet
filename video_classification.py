# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np
import tensorflow as tf
import sys
import os
import cv2
import math


def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph


def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(
        file_reader, channels=3, name="png_reader")
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(
        tf.image.decode_gif(file_reader, name="gif_reader"))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
  else:
    image_reader = tf.image.decode_jpeg(
        file_reader, channels=3, name="jpeg_reader")
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.compat.v1.Session()
  result = sess.run(normalized)

  return result


def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label


if __name__ == "__main__":
  #file_name = "/content/drive/My Drive/grace_hopper.jpg"
  model_file = \
    "/content/drive/My Drive/inception_v3_2016_08_28_frozen.pb"
  label_file = "/content/drive/My Drive/imagenet_slim_labels.txt"
  input_height = 299
  input_width = 299
  input_mean = 0
  input_std = 255
  input_layer = "input"
  output_layer = "InceptionV3/Predictions/Reshape_1"

  graph = load_graph(model_file)
  
  input_name = "import/" + input_layer
  output_name = "import/" + output_layer
  input_operation = graph.get_operation_by_name(input_name)
  output_operation = graph.get_operation_by_name(output_name)
  
  writer = None
  
  video_path = "/content/drive/My Drive/football.mp4" ## path_to_video
  
  with tf.compat.v1.Session(graph=graph) as sess:
    
    ## start video capture.... read video file
    video_capture = cv2.VideoCapture(video_path)
    i = 0
    while True: 
        frame = video_capture.read()[1] ## get current frame
        frameId = video_capture.get(1) ## current frame number
        i = i + 1
        cv2.imwrite(filename="/content/drive/My Drive/screens/"+str(i)+"alpha.png", img=frame); ## write frame image to file
        
        file_name = "/content/drive/My Drive/screens/"+str(i)+"alpha.png" 
        
        ## convert frame to tensor
        t = read_tensor_from_image_file(
        file_name,
        input_height=input_height,
        input_width=input_width,
        input_mean=input_mean,
        input_std=input_std)

        predictions = sess.run(output_operation.outputs[0], {
        input_operation.outputs[0]: t })
        predictions = np.squeeze(predictions) ## predictions
        
        top_k = predictions.argsort()[-1:][::-1] ## top_k predictions.. set to top1 prediction for every frame
        labels = load_labels(label_file)
        pos = 1
        for i in top_k:
            human_string = labels[i]
            score = predictions[i]
            cv2.putText(frame, '%s (score = %.5f)' % (human_string, score), (40, 40 * pos), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255))
            print('%s (score = %.5f)' % (human_string, score))
            pos = pos + 1
        print ("\n\n")
        if writer is None:
            ## initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter("recognized.avi", fourcc, 10,
                (frame.shape[1], frame.shape[0]), True)
  
        ## write the output frame to video
        writer.write(frame)
        cv2.waitKey(1)
    writer.release()
    video_capture.release()
    cv2.destroyAllWindows()