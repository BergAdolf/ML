# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 15:11:48 2020

@author: adolf
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.neighbors  import KNeighborsClassifier 
from sklearn.svm import SVC 
from sklearn.datasets import fetch_lfw_people
from fetch_ORL_people import fetch_ORL_people
warnings.filterwarnings('ignore')

#获取数据集
data_name = "ORL"  # 可选 "LFW" 或者 "ORL"
method = 'FLD'     # 可选 "PCA" 或者 "FLD"
if data_name == "LFW":
    faces = fetch_lfw_people( min_faces_per_person=60) #可通过sklearn直接下载
    n_components = 30
    n_neighbors = 3
elif data_name == "ORL":
    faces = fetch_ORL_people('./Data/ORL') #需自行下载至目录，再运行。
    n_components = 30
    n_neighbors = 3
else:
    raise ValueError(f'Invalid argument for data_name: {data_name}')
    
n_examples, img_row, img_col = faces.images.shape
n_classes = faces.target_names.size
print(f'Num examples={n_examples}, num classes={n_classes}, image shape={img_row}*{img_col}, feature shape={img_row*img_col}')


#划分数据集中训练数据与测试数据
idx_train, idx_test = train_test_split(np.arange(faces.data.shape[0]), test_size=0.2, stratify=faces.target)
X_train, Y_train = faces.data[idx_train], faces.target[idx_train]
X_test, Y_test = faces.data[idx_test], faces.target[idx_test]

#采用PCA降维
pca = PCA(svd_solver='full', n_components=n_components) 
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

#采用LFD降维
fld = LinearDiscriminantAnalysis(solver ='svd', n_components=min(n_classes-1, n_components))
X_train_fld = fld.fit_transform(X_train, Y_train)
X_test_fld = fld.transform(X_test)

#预测数据
if method == 'PCA':
    train_embedding = X_train_pca
    test_embedding = X_test_pca
elif method == "FLD":
    train_embedding = X_train_fld
    test_embedding = X_test_fld
else:
    raise ValueError(f'Invalid argument for method: {method}')


#KNN
clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance', n_jobs=10)
clf.fit(train_embedding, Y_train)
acc = clf.score(test_embedding, Y_test)
print(f'KNN algorithm Accuracy {acc*100:.2f}%')

#svm
svc = SVC(kernel='rbf', class_weight='balanced', gamma='scale')
svc.fit(train_embedding, Y_train)
acc = svc.score(test_embedding, Y_test)
print(f'svM algorithm Accuracy {acc*100:.2f}%')