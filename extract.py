import numpy as np
import argparse
import pickle
import cv2
import os

# 
ap = argparse.ArgumentParser()

ap.add_argument("-i", "--dataset", required = True,
    help = "Path to the input directory of faces to train against")
ap.add_argument("-e", "--embeddings", required = True,
    help = "path to output serialized db of facial embeddings")
ap.add_argument("-d", "--detector", required = True,
    help = "Path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", required = True,
    help = "Path to OpenCV's deep learning face embedding model")
ap.add_argument("-c", "--confidence", type = float, default = 0.5,
    help = "minimum probability to filter weak detections, default 0.5")

args = vars(ap.parse_args())