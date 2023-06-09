# BOW-SIFT-ORB
Languages: Python

In this project, I used the BOVW method to detect whether my water bottle was in an image or not. My methods could be easily altered for other objects.

My motivation behind exploring the Bag of Visual Words (BOVW) could be broken down into two reasons. I am familiar with the Bag of Words methodology from a data mining course I took last semester, so I was excited to revisit it. My second reasoning is that I wanted to explore how using different methods of feature descriptors could alter the BOVW method and how it would impact the accuracy of the image classifications. I chose the SIFT and ORB feature descriptors.
I also attempted to implement SIFT (located in feat_exe.py as adl_sift()) using Lowe’s groundbreaking paper and any other resource I could find to compare it with the other descriptors. Unfortunately, due to difficulties and time constraints, I had to abandon it. 

I used my iPhone to gather images for my dataset. I took images of the bottle from different angles and distances.After gathering my images, I converted them to the jpg format and decreased their sizes to 500 by 500 to help speed up the classification time. I split the images into training set and test set, then split each of those into “bottle” and “no_bottle” classes.

Results from the detection system varied heavily and were extremely inconsistent. Initially, I noticed that BOVW using SIFT would outperform BOVW with ORB in accuracy, but was much slower than BOVW with ORB. Later on, I noticed that they would sometimes have equal accuracies, which I noticed in my video, and other times SIFT would perform worse. 

What could be causing this issue? I suspect it could be an issue with the training and testing datasets. Much of my training dataset images were taken inside, while much of my testing datasets were taken outside. The fact that the classifiers were trained on lower light, indoor images, and then were tested on brighter, outside images may be a cause. It could also be that my images for the “bottle” training set were not good enough.
Although it performed poorly, with more time, I could improve results by choosing better images to improve the dataset, tweaking the number of clusters, or adjusting the frequency vectors. It would have been more interesting if I were able to put bounding boxes around the images that were identified as “bottle” I would also would have liked to explore alternative feature extraction methods and deep learning approaches.
