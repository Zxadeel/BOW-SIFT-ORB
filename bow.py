import scipy.cluster.vq as sciv
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import numpy as np


def kmeans(descriptors, k=200):
    codebook, variance = sciv.kmeans(descriptors, k, 1)
    return codebook, variance

def vector_quantization(num_imgs, img_p_d, codebook, k=200):
    img_features = np.zeros((num_imgs, k))

    for i in range(num_imgs):
        words, dist = sciv.vq(img_p_d[i][1],codebook)
        for w in words:
            img_features[i][w] += 1.

    #fitting features
    scaler = StandardScaler().fit(img_features)
    img_features = scaler.transform(img_features)

    return img_features

def create_SVM(img_features, truth_train, max_iter=80000):
    classifier = LinearSVC(max_iter=max_iter)
    classifier.fit(img_features,np.array(truth_train))  
    return classifier

def predict(classifer, test_features, truth_test, class_names):
    true_classes=[]
    for i in truth_test:
        if i==1:
            true_classes.append(class_names[1])
        else:
            true_classes.append(class_names[0])

    predict_classes=[]
    for i in classifer.predict(test_features):
        if i==1:
            predict_classes.append(class_names[1])
        else:
            predict_classes.append(class_names[0]) 

    accuracy = sklearn.metrics.accuracy_score(true_classes,predict_classes)

    return accuracy, true_classes, predict_classes

def incorrect_predictions(image_paths_test, true_classes, predict_classes):
    inc_img_paths = []
    for i in range(len(true_classes)):
        if true_classes[i] != predict_classes[i]:
            # print(true_classes[i], predict_classes[i])
            inc_img_paths.append((image_paths_test[i], predict_classes[i]))
    
    return inc_img_paths
