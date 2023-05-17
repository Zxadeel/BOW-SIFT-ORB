import time
import dataset
import bow
import feat_exe
import picture_viewer as p
from tkinter import *
import tkinter as tk
from PIL import ImageTk, Image


def main():
    print("Gathering Training and Testing images...")
    classes = dataset.get_class_names('data')
    img_paths_train, truth_train = dataset.img_paths_list(classes, 'data')
    img_paths_test, truth_test = dataset.img_paths_list(classes,'testset')
    # img_paths_train, truth_train, img_paths_test, truth_test = dataset.train_test_split(img_paths, truth)

    ###### using SIFT features
    print("Extracting SIFT features on training set...")
    a = time.time()
    sift_p_d, sift_descrip, _, _ = feat_exe.sift(img_paths_train)
    print("Creating vocabulary...")
    codebook, _ = bow.kmeans(sift_descrip)
    train_img_feat = bow.vector_quantization(len(img_paths_train), sift_p_d, codebook)
    print("Training classifier...")
    classifier = bow.create_SVM(train_img_feat, truth_train)
    print("Extracting SIFT features on test set...")
    sift_p_d_test, _, _, _ = feat_exe.sift(img_paths_test)
    test_img_feat = bow.vector_quantization(len(img_paths_test), sift_p_d_test, codebook)
    print("Making predictions...")
    sift_accuracy, true_class, pred_class = bow.predict(classifier, test_img_feat, truth_test, classes)
    sift_incorrect = bow.incorrect_predictions(img_paths_test, true_class, pred_class)
    b = time.time() - a
    print(f"Finished SIFT, it took {b} seconds!\n\n")

    ###### using ORB features
    print("Extracting ORB features on training set...")
    a = time.time()
    orb_p_d, orb_descrip, _, _ = feat_exe.orb(img_paths_train)
    print("Creating vocabulary...")
    codebook, _ = bow.kmeans(orb_descrip)
    train_img_feat = bow.vector_quantization(len(img_paths_train), orb_p_d, codebook)
    print("Training classifier...")
    classifier = bow.create_SVM(train_img_feat, truth_train)
    print("Extracting ORB features on test set...")
    orb_p_d_test, _, _, _ = feat_exe.orb(img_paths_test)
    test_img_feat = bow.vector_quantization(len(img_paths_test), orb_p_d_test, codebook)
    print("Making predictions...")
    orb_accuracy, otrue_class, opred_class = bow.predict(classifier, test_img_feat, truth_test, classes)
    orb_incorrect = bow.incorrect_predictions(img_paths_test, otrue_class, opred_class)
    b = time.time() - a
    print(f"Finished ORB, it took {b} seconds!\n\n")
    
    print(f"The accuracy when using SIFT Features was {sift_accuracy*100}%")
    print(f"The accuracy when using ORB Features was {orb_accuracy*100}%")
    time.sleep(2)

    return orb_incorrect, sift_incorrect


   
o,s = main()

feature_ex = "SIFT and ORB Mistakes"

#code below is modified from https://techvidvan.com/tutorials/python-image-viewer/

window = Tk()#creating window
window.geometry("900x700")#geometry of window
window.title(feature_ex)#title to window
Label(window,text=feature_ex,font=('bold',20)).pack()#label

#creating frame
frame=Frame(window, width=230, height=200, bg='white')
frame.pack()

lst = []
inc_labels = []
for img,lbl in o:
    disp_img = ImageTk.PhotoImage(Image.open(img).resize((500,500), Image.ANTIALIAS))
    lst.append(disp_img)
    inc_labels.append("This image was mistaken as the "+lbl+" class by ORB.")

for img,lbl in s:
    disp_img = ImageTk.PhotoImage(Image.open(img).resize((500,500), Image.ANTIALIAS))
    lst.append(disp_img)
    inc_labels.append("This image was mistaken as the "+lbl+" class by SIFT.")

i = 0
image_label = Label(frame, image=lst[i])#packing image into the window
text_label = Label(frame, text=inc_labels[i])
image_label.pack()
text_label.pack()

def Next():
    global i
    # global 
    i = i + 1
    try:

        image_label.config(image=lst[i])
        text_label.config(text=inc_labels[i])
        # print(i)
    except:
        i = -1
        Next()

def Back():
    global i
    i = i - 1
    try:
        image_label.config(image=lst[i])
        text_label.config(text=inc_labels[i])
        # print(i)
    except:
        i = 0
        Back()

Button(window,text='Back',command=Back,bg='light blue').place(x=230,y=40)
Button(window,text='Next',command=Next,bg='light blue').place(x=700,y=40)

window.mainloop()

