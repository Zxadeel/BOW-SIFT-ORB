# BOW-SIFT-ORB: Object Detection using Bag of Visual Words, SIFT, and ORB

**Languages:** Python

## Overview

The BOW-SIFT-ORB project is an exploration of the Bag of Visual Words (BOVW) method to detect the presence of a specific object, in this case, a water bottle, within an image. The methods employed in this project can be easily adapted for detecting other objects as well.

### Motivation

The project was motivated by two primary factors:

1. **Familiarity with BOVW**: The use of Bag of Words methodology from a previous data mining course served as a starting point for this project. Revisiting this concept was both exciting and intellectually stimulating.

2. **Exploring Feature Descriptors**: The project aimed to investigate how different feature descriptors could impact the BOVW method's accuracy. Specifically, the project utilized SIFT and ORB feature descriptors. There was also an attempt to implement SIFT using Lowe's seminal paper and other resources, although this effort had to be abandoned due to difficulties and time constraints.

## Data Collection

Images for the dataset were collected using an iPhone. Multiple images of the water bottle were taken from various angles and distances. These images were converted to the jpg format and resized to 500 by 500 pixels to enhance classification speed. The dataset was then split into training and test sets, further divided into "bottle" and "no_bottle" classes.

## Results and Challenges

The results obtained from the object detection system exhibited significant variability and inconsistency. Key observations included:

- Initially, BOVW with SIFT outperformed BOVW with ORB in terms of accuracy but was notably slower.
- Later, equal accuracies were observed between the two methods in some cases, as evident in video demonstrations, while in other cases, SIFT performed worse.

**Challenges and Potential Causes**:

The inconsistency in results could be attributed to several factors:

- **Differences in Training and Testing Datasets**: The training dataset predominantly contained indoor images with lower light conditions, whereas the testing dataset comprised brighter outdoor images. This discrepancy may have contributed to the variation in performance.

- **Dataset Quality**: The quality of images within the "bottle" training set may have influenced poor performance. With more time, improvements could be made by selecting higher-quality images, adjusting the number of clusters, or fine-tuning frequency vectors.

## Future Directions

Despite the challenges encountered, the project presents opportunities for further exploration and improvement:

- **Bounding Boxes**: Implementing bounding boxes around identified "bottle" images would enhance the project's visual output.

- **Enhanced Dataset**: Improving the dataset by selecting better images and refining preprocessing techniques.

- **Algorithm Optimization**: Fine-tuning parameters such as the number of clusters and frequency vectors for better accuracy.

- **Alternative Approaches**: Exploring alternative feature extraction methods and deep learning approaches for object detection.
