import os
import pickle

import cv2 as cv
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.svm import SVC

from skinbot.dataset import get_dataloaders, minmax
import torch
from skinbot.models import get_model
from skinbot.config import read_config, Config
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.optim as optim
import torch.nn as nn
import time
import copy

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random
random.seed(0)

config = read_config()
C = Config().set_config(config)
fold = 0
target = 'classification'
# model.load_state_dict(torch.load(path_to_model))
all_dataloader = get_dataloaders(config, batch=16, mode='all', target=target)
d_tr = get_dataloaders(config, batch=16, mode='train', target=target, fold_iteration=fold)
d_ts = get_dataloaders(config, batch=16, mode='test', target=target, fold_iteration=fold)
def kmean_bow(all_descriptors, num_cluster):
    bow_dict = []

    kmeans = KMeans(n_clusters = num_cluster)
    d = np.concatenate(all_descriptors, axis = 0)
    kmeans.fit(d)

    bow_dict = kmeans.cluster_centers_

    if not os.path.isfile('bow_dictionary.pkl'):
        pickle.dump(bow_dict, open('bow_dictionary.pkl', 'wb'))

    return bow_dict

def create_feature_bow(image_descriptors, BoW, num_cluster):

    X_features = []

    for i in range(len(image_descriptors)):
        features = np.array([0] * num_cluster)

        if image_descriptors[i] is not None:
            distance = cdist(image_descriptors[i], BoW)

            argmin = np.argmin(distance, axis = 1)

            for j in argmin:
                features[j] += 1
        X_features.append(features)

    return X_features
def extract_sift_features(gray, algorithm):

    # Create a SIFT detector

    # Detect keypoints and compute descriptors
    kp, des = algorithm.detectAndCompute(gray, None)

    return kp, des
# extract sift features from dataloader
def extract_sift_features_from_dataloader(dataloader):
    all_descriptors = []
    all_labels = []
    sift = cv.SIFT_create()
    for inputs, labels in dataloader:
        inputs = (255*minmax(inputs)).numpy().astype(np.uint8)
        labels = labels.numpy()
        for i in range(len(inputs)):
            img = inputs[i]
            img = np.transpose(img, (2,1,0))
            gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
            kp, des = extract_sift_features(gray, sift)
            print(des.shape)
            all_descriptors.append(des)
            all_labels.append(labels[i])
    return all_descriptors, all_labels

image_descriptors, labels = extract_sift_features_from_dataloader(all_dataloader)

# remove descriptors that are None
image_descriptors = [x for x in image_descriptors if x is not None]
labels = [labels[i] for i in range(len(image_descriptors)) if image_descriptors[i] is not None]

num_clusters = 50
bow = kmean_bow(image_descriptors, num_clusters)

X_features = create_feature_bow(image_descriptors, bow, num_clusters)

# split the data train, test
X_train, X_test, y_train, y_test = train_test_split(X_features, labels, test_size=0.2, random_state=0)
model_svm = SVC(kernel='linear', C=30, gamma='auto')
model_svm.fit(X_train, y_train)
filename = 'svm_model.sav'
pickle.dump(model_svm, open(filename, 'wb'))
print("score on training set params: ", model_svm.score(X_train, y_train))
print("score on testing set params: ", model_svm.score(X_test, y_test))
