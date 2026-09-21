AI Internship
Week 1 - Data Analysis
This repository contains my Week 1 internship work.
The project demonstrates basic Python, NumPy, and Pandas operations.
The notebook loads a CSV dataset, explores the data, selects rows and columns, filters data, and calculates basic summary statistics.
# Week 2 — Exploratory Data Analysis
This project contains my Week 2 internship work on exploratory data analysis.
## Dataset
The dataset contains football player statistics from Kaggle, with 3,120 players and 65 columns.
## What I Learned
- Exploring datasets using Pandas
- Cleaning and converting data
- Filtering and grouping data
- Calculating summary statistics
- Creating visualizations using Matplotlib
- Finding patterns and insights in a real-world dataset

## Week 3 — First Machine Learning Model
This week I built my first machine learning classification model using the Iris dataset and scikit-learn.
### What I learned
- Split data into training and testing sets
- Created and trained a K-Nearest Neighbours (KNN) classifier
- Made predictions on unseen test data
- Measured model accuracy
- Created and interpreted a confusion matrix
- Used a classification report to evaluate the model
- Learned why testing on unseen data is important
### Result
The KNN model achieved 100% accuracy on the test set. The confusion matrix showed that all 30 test samples were classified correctly with no misclassifications.

# Week 4: Better Models and Honest Evaluation
## Overview
This week focused on improving and evaluating machine learning models using the Iris dataset. I worked with KNN, Decision Tree, and Random Forest classifiers and compared their performance using training accuracy, test accuracy, and cross-validation.

## What I Learned

* How Decision Tree and Random Forest classifiers work
* How to compare different machine learning models
* How to identify possible overfitting
* Why test accuracy can sometimes be misleading
* How changing the `random_state` can affect model performance
* How to use 5-fold cross-validation for a more reliable evaluation
* How feature scaling affects KNN performance
* How to interpret confusion matrices and classification reports

## Models Compared

| Model         | Training Accuracy | Test Accuracy |
| ------------- | ----------------: | ------------: |
| KNN           |            95.83% |          100% |
| Decision Tree |              100% |        93.33% |
| Random Forest |              100% |           90% |

## Cross-Validation

KNN was evaluated using 5-fold cross-validation.

**Average Cross-Validation Score: 95.83%**

This provides a more reliable estimate of model performance than relying on a single train-test split.

## Random State Investigation

The KNN model was tested using random states **42, 0, 1, and 7**. The test accuracy was 100% for random states 42, 0, and 7, while it decreased to 96.67% for random state 1. This showed that model accuracy can change depending on how the data is split.

## Conclusion

KNN achieved the highest test accuracy on this particular split, but the 5-fold cross-validation score of 95.83% gives a more honest estimate of its generalization performance. The experiment showed why a single accuracy score, especially a perfect 100% score, should be investigated rather than automatically treated as evidence of a perfect model.
# Week 5 – Deep Learning with PyTorch
## Overview
This week focused on learning the basics of deep learning using **PyTorch** and building a neural network to classify handwritten digits.
## What I Did
* Set up PyTorch and enabled the **Google Colab GPU (Tesla T4)**
* Learned about **PyTorch tensors and Autograd**
* Loaded and prepared the **MNIST dataset**
* Built a neural network from scratch using PyTorch
* Implemented the complete **training loop**
* Plotted the training loss curve
* Evaluated the model's test accuracy
* Examined misclassified images
* Added **Dropout** as an improvement and compared the results

## Technologies

* Python
* PyTorch
* Torchvision
* Matplotlib
* Google Colab
* MNIST Dataset

## Result
The neural network successfully learned to classify handwritten digits. The loss decreased during training, and the model achieved high test accuracy.
## Conclusion
This week helped me understand how PyTorch neural networks are trained using **forward pass, loss calculation, backpropagation, and optimizer updates**.

# Week 6 – Object Detection and OCR (ANPR)

## Overview

This project implements a basic **Automatic Number Plate Recognition (ANPR)** system using **YOLO** for license plate detection and **EasyOCR** for text recognition.

**Pipeline:**
Image → YOLO → License Plate Crop → EasyOCR → Plate Text

## Dataset

* Training Images: **346**
* Testing Images: **87**
* Classes: **1 (`license_plate`)**

## What I Did

* Prepared and verified the license plate dataset.
* Fine-tuned a pretrained YOLO model using transfer learning.
* Detected license plates in an unseen image.
* Cropped the detected license plate.
* Used EasyOCR to recognize the plate text.
* Evaluated the limitations of OCR.

## Results

* YOLO detection: **Successful**
* Best detection confidence: **0.933**
* License plate cropping: **Successful**
* EasyOCR: **Successfully executed**
* OCR: **Some characters were recognized with low confidence**

## Findings

YOLO successfully detected the license plate, while OCR performance was affected by image quality and plate characteristics. This showed me that accurate object detection does not always guarantee accurate OCR results.

## Conclusion

The complete **YOLO + EasyOCR ANPR pipeline** was successfully implemented and tested. The project provided practical experience with **object detection, transfer learning, image processing, and OCR**.

# Week 7: Capstone Build – ANPR Pipeline
## Overview
This project is an end-to-end **Automatic Number Plate Recognition (ANPR)** pipeline using YOLO and EasyOCR. It takes an input image, detects the license plate, reads its text, and returns the final result.
## Pipeline
**Input Image → YOLO Detection → Plate Crop → Preprocessing → EasyOCR → Text Cleaning → Final Result**

## Tools

* Python
* PyTorch
* Ultralytics YOLO
* EasyOCR
* OpenCV
* NumPy
* Google Colab

## Dataset & Evaluation

The dataset contains **346 training images** and **87 validation/test images**, with one class: `license_plate`.

The YOLO model was trained for **20 epochs**.

| Metric    | Result |
| --------- | -----: |
| Precision |  0.895 |
| Recall    |  0.883 |
| mAP@50    |  0.930 |
| mAP@50-95 |  0.565 |

## Example Input & Output

The complete pipeline was tested on a fresh image:

**Input:** `Cars164.png`

YOLO detected the license plate with **0.857 confidence**.

EasyOCR returned:

```text
Detected Text: GT
OCR Confidence: 0.998
```

The result was also saved to `anpr_result.txt`.

## Findings

* YOLO successfully detected the license plate.
* The detected region was cropped and passed to EasyOCR.
* EasyOCR returned readable text.
* The complete pipeline worked from a fresh image to a final result.

## Conclusion

This project combines **object detection and OCR into one working ANPR pipeline**. It can take a new vehicle image, detect the license plate, read its text, and return the result end to end.

## How to Run

Open `Week_7_Capstone_ANPR_Pipeline.ipynb` in Google Colab, provide a vehicle image, and run the cells in order to obtain the final plate detection and OCR result.

