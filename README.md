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
