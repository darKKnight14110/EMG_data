# EMG Gesture Recognition with Attention Model

## Overview

This repository contains an implementation of a simple model for recognizing gestures using surface electromyography (EMG) signals. The model incorporates an attention mechanism to focus on relevant parts of the input data, enhancing its ability to recognize gestures accurately.

## Dataset Collection

The dataset collection process for the EMG Gesture Recognition model involves the following steps:

1. **NinaPro DB5**: This dataset, also known as the "Double MYO" dataset, is the primary dataset used in the project. It consists of EMG data collected from subjects using two MYO armbands, capturing 16 channels of EMG signals along with triaxial accelerometer data.

2. **Gesture Variety**: NinaPro DB5 contains a diverse set of 53 unique gestures performed by subjects. These gestures include fine finger movements, wrist actions, functional grasping motions, and periods of rest.

3. **Data Organization**: Each gesture is repeated six times by subjects, with three seconds of rest between repetitions. The dataset is structured into approximately 600,000 rows per subject, with a sampling frequency of 200 Hz.

4. **Data Processing**: Raw EMG data undergoes preprocessing steps, including rectification, filtering, and smoothing. This process extracts relevant features for gesture recognition while removing noise and artifacts from the signals.

5. **Evaluation Strategy**: To assess model performance, the third repetition of gestures is used for parameter tuning, while the fifth repetition serves as an unseen holdout set for testing. This evaluation strategy ensures the robustness and generalizability of the model's performance.

In summary, the dataset collection process for the EMG Gesture Recognition model involves gathering diverse gesture data from NinaPro DB5, organizing and preprocessing the data, and employing a systematic evaluation strategy to validate model performance.

## Model Architecture

The attention-based model architecture consists of three main parts:

1. **Expansion Layer**: This layer expands the input data to a matrix format, making it easier for the model to extract features. The expansion layer transforms the input from a window of channels and timesteps to a matrix with standardized dimensions.

2. **Attention Mechanism**: The attention mechanism processes the expanded data, highlighting important features and suppressing irrelevant ones. It helps the model focus on relevant aspects of the input, improving its recognition accuracy.

3. **Classifier Network**: After the attention mechanism, the data is fed into a classifier network, which consists of fully connected layers. These layers analyze the extracted features and classify the input into different gesture categories.

## Implementation Details

The attention mechanism works as follows:
- The time series data processed by the previous layer, h ∈ ℝ^(T × C), where T represents timesteps and C represents features or channels, is inputted.
- h is transposed into a C × T matrix. Each row represents the time series produced by a channel.
- This matrix is then fed row by row to a standard, fully connected layer using the softmax activation function, resulting in a C × T matrix, α.
- The entry α_ij is the attention score for timestep j at channel i.

Each model was trained for 20 epochs with a batch size of 128, and a learning rate annealed from 1 × 10^(-3) to 1 × 10^(-5). The Mish activation function was used instead of ReLU, and the Ranger optimizer, combining Adam, was employed for optimization.

## Training Process

- The learning rate follows a delayed cosine annealing schedule.
- Dropout with a rate of 0.36 is applied to all layers in the classification network.
- Focal loss function and data augmentation with noise addition were used to address class imbalance in the dataset.

## Group Members

The development of the sEMG Gesture Recognition model involved the collaborative efforts of the following group members:

- Pradeep Kumar
- Sai Pranay deep
- Vedant Dinkar
- Sai Sanjana Reddy Algubelly
- Vaidehi Bhat
- Varad Pendse

**Mentored By:** Krish Agrawal

## Conclusion

The attention-based model presented in this repository offers a simple yet effective approach to EMG gesture recognition. By incorporating an attention mechanism and employing appropriate training techniques, the model achieves improved accuracy, even in the presence of class imbalance.
