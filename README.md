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
* How to tell when the difference between models is too small to mean anything

## Models Compared

| Model         | Training Accuracy | Test Accuracy | 5-Fold CV Mean |
| ------------- | ----------------: | ------------: | -------------: |
| KNN           |            95.83% |          100% |         95.83% |
| Decision Tree |              100% |        93.33% |         94.17% |
| Random Forest |              100% |           90% |         95.00% |

The Decision Tree and Random Forest scored 100% on the training data but lower on the test data, which is a sign of some overfitting.

## Cross-Validation

All three models were evaluated using 5-fold cross-validation on the training data. The averages were **95.83%** for KNN, **95.00%** for Random Forest, and **94.17%** for the Decision Tree.

This provides a more reliable estimate of model performance than relying on a single train-test split.

## Random State Investigation

The KNN model was tested using random states **42, 0, 1, and 7**. The test accuracy was 100% for random states 42, 0, and 7, while it decreased to 96.67% for random state 1. This showed that model accuracy can change depending on how the data is split.

## Feature Scaling

Scaling the features (fitted on the training data only, to avoid data leakage) did not improve KNN. Its test accuracy went down from 100% to 93.33%.

## Conclusion

KNN achieved the highest test accuracy on this particular split, but that 100% was partly luck, as the random state investigation showed. The cross-validation scores of the three models are very close (95.83%, 95.00% and 94.17%). That is a difference of only a couple of flowers out of 120, so on the Iris dataset the three models are **too close to call**. The experiment showed why a single accuracy score, especially a perfect 100% score, should be investigated rather than automatically treated as evidence of a perfect model.

# Week 5 – Deep Learning with PyTorch

## Overview
This week focused on learning the basics of deep learning using **PyTorch** and building a neural network to classify handwritten digits from the **MNIST dataset**.

## What I Did
* Set up PyTorch and enabled the **Google Colab GPU (Tesla T4)**
* Learned about **PyTorch tensors and Autograd**
* Loaded and prepared the **MNIST dataset** (60,000 training images and 10,000 test images)
* Built a neural network from scratch using PyTorch (784 → 128 → 10)
* Implemented the complete **training loop**
* Plotted the training loss curve
* Evaluated the model's test accuracy
* Examined misclassified images
* Added **Dropout (0.2)** as an improvement and compared it with the baseline using **three random seeds**

## Technologies

* Python
* PyTorch
* Torchvision
* Matplotlib
* Google Colab
* MNIST Dataset

## Results

| Model | Seed 42 | Seed 0 | Seed 7 | Average Test Accuracy |
|---|---|---|---|---|
| Baseline | 97.43% | 97.28% | 97.24% | **97.32%** |
| With Dropout (0.2) | 97.49% | 97.39% | 97.34% | **97.41%** |

The training loss decreased from 0.3567 to 0.0701 over 5 epochs, showing that the model learned from the training data.
In the single run, the baseline model achieved 97.38% test accuracy and misclassified 262 out of 10,000 test images. The Dropout model achieved 97.39% test accuracy.
Across three random seeds (42, 0, and 7), the baseline model achieved an average accuracy of 97.32%, while the Dropout model achieved 97.41%. Dropout performed slightly better in all three runs, with improvements of 0.06, 0.11, and 0.10 percentage points, respectively. This represents a small but consistent improvement.

## Conclusion
This week helped me understand how PyTorch neural networks are trained using **forward pass, loss calculation, backpropagation, and optimizer updates**. I also learned that a single run is not enough to say a change is an improvement. Dropout was slightly better in all three seeds, but the gain was tiny, so both models performed well on MNIST.
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
* Cropped the detected license plate and enlarged it 4× for OCR.
* Used EasyOCR to recognize the plate text.
* Applied a format check to correct likely OCR character errors.
* Evaluated the limitations of OCR.

## Results

* YOLO detection (87-image validation set): precision **0.933**, recall **0.851**, mAP@50 **0.923**, mAP@50-95 **0.541**
* Detection confidence on the test image: **0.933**
* License plate cropping: **Successful**
* EasyOCR on the enlarged crop returned `IND` (the country code, confidence 0.989) and `MHO 1AE8Q1Z|` (the plate, confidence 0.095)
* After the format check, `MHO 1AE8Q1Z|` became **`MH01AE8017`**, which matches the Indian plate format

## Findings

YOLO successfully detected the license plate, while OCR performance was affected by image quality and plate characteristics. The OCR result with the lowest confidence turned out to be the plate itself, showing that confidence does not always indicate whether the recognized text is correct.
Checking the recognized text against the expected plate format helped identify possible letter and digit mix-ups. However, a format match does not prove that the recognized plate is correct. Character corrections must be checked against the actual plate image and tested on more examples before being trusted.

## Conclusion

The complete **YOLO + EasyOCR ANPR pipeline** was successfully implemented and tested. The project provided practical experience with **object detection, transfer learning, image processing, OCR, and format checking**.

# Week 7: Capstone Build – ANPR Pipeline
## Overview
This project is an end-to-end **Automatic Number Plate Recognition (ANPR)** pipeline using YOLO and EasyOCR. It takes an input image, detects the license plate, reads its text, and returns the final result.
## Pipeline
**Input Image → YOLO Detection → Plate Crop → Preprocessing (4× enlargement) → EasyOCR → Final Result**

## Tools

* Python
* PyTorch
* Ultralytics YOLO
* EasyOCR
* OpenCV
* NumPy
* Google Colab

## Dataset & Evaluation

The dataset has one class: `license_plate`. In the first training run, the 87 test images were also used as the validation set, so the test images helped choose the model and the metrics were slightly optimistic (mAP@50 0.930). I fixed this by splitting a separate validation set from the training images and retraining for **20 epochs**:

* Training: **277 images**
* Validation: **69 images**
* Test: **87 images** (kept untouched for final evaluation)

| Metric (69-image validation set) | Result |
| --------- | -----: |
| Precision |  0.925 |
| Recall    |  0.826 |
| mAP@50    |  0.904 |
| mAP@50-95 |  0.554 |

## Example Input & Output

The complete pipeline was tested on a fresh image:

**Input:** `Cars164.png`

YOLO detected the license plate with **0.857 confidence**.

EasyOCR returned:

```text
Detected Text: GT
OCR Confidence: 0.998
```

This is correct for this plate. Weak text pieces below the 0.50 confidence threshold were dropped. The result was also saved to `anpr_result.txt`. One image is an example, not an evaluation, so I also tested 20 images.

### 20-Image Evaluation

I selected 20 test images and manually verified their license plate text to create a ground-truth answer key. The same `answer_key.json` file is used in both Week 7 and Week 8 to ensure consistent evaluation.

Two images were excluded from exact-match accuracy scoring:

- Cars162.png: The license plate was partly hidden.
- Cars87.png: Two license plates were visible, so there was no single clear answer.

The remaining 18 images were used for exact-match evaluation.

### Findings

- YOLO detected plates in 18 out of 20 images (90.0%).
- OCR correctly recognized 2 out of 18 scorable plates (11.1%).
- Plate detection and OCR accuracy measure different parts of the system.
- OCR still struggles with similar-looking letters and numbers.

### Conclusion

The Week 7 project successfully demonstrated an end-to-end license plate detection and OCR pipeline. However, the evaluation showed that recognizing plate text is more challenging than detecting the plate. Further OCR improvements are needed.

## How to Run

Open `Week_7_Capstone_ANPR_Pipeline.ipynb` in Google Colab, provide a vehicle image, and run the cells in order to obtain the final plate detection and OCR result.

## Week 8 – Capstone App: License Plate Detection and OCR

### Overview

This project packages my license plate detection and OCR model into a Gradio application. Users can upload a vehicle image, and the application detects the license plate and attempts to recognize its text.

### How It Works

Image → YOLO → License Plate Detection → Crop → EasyOCR → Result

### Model Results

- Training images: 277
- Validation images: 69
- Test images: 87
- Validation mAP@50: 0.904
- Selected test images evaluated: 20
- Plates detected: 18 out of 20 (90.0%)
- OCR exact-match accuracy: 2 out of 18 (11.1%)

### Evaluation Method

The old and updated pipelines were evaluated using the same 20 selected test images and the same verified `answer_key.json` file.

Two images were excluded from exact-match scoring:

- Cars162.png: The plate was partly hidden.
- Cars87.png: Two plates were visible, so there was no single clear answer.

The remaining 18 images were used for exact-match evaluation.

### Findings

- The old pipeline correctly recognized 2 out of 18 plates (11.1%).
- The updated pipeline also correctly recognized 2 out of 18 plates (11.1%).
- Both pipelines produced identical OCR predictions in this comparison.
- The updated pipeline did not demonstrate an improvement in exact-match accuracy.
- YOLO detected plates in 18 out of 20 images (90.0%).
- OCR still struggles with similar-looking characters.

### Conclusion

Both pipelines achieved the same exact-match accuracy of 2 out of 18 images (11.1%). A difference of only one correctly recognized plate would be too small to declare a reliable winner. A larger evaluation dataset is needed for a more reliable comparison.

### Files

- `app.py` – Gradio application.
- `best.pt` – trained YOLO model.
- `requirements.txt` – required Python packages.
- `Week_8_App_Deployment.ipynb` – Week 8 notebook.
- `answer_key.json` – shared verified answer key.
