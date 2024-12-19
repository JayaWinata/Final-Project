# Sentiment Analysis Using LSTM

## Overview

This project focuses on sentiment analysis utilizing Long Short-Term Memory (LSTM) networks. Sentiment analysis involves determining the sentiment expressed in textual data, categorizing it as positive, negative, or neutral. LSTM, a type of Recurrent Neural Network (RNN), is particularly effective for this task due to its ability to capture long-term dependencies in sequential data.

The data for this project was retrieved from user reviews of the **DANA App** on the Google Play Store. These reviews were used to train and evaluate the model.

---

## Key Features

- **Data Collection from Play Store**: Extracted reviews from the DANA App on Google Play Store for analysis.
- **Data Preprocessing**: Cleaned and tokenized text data to prepare it for model training.
- **LSTM Model Implementation**: Utilized LSTM networks for effective sentiment classification.
- **Performance Metrics**: Included accuracy evaluation to measure model effectiveness.
- **Visualization**: Provided visual insights into data distribution and model performance.

---

## Key Steps

1. **Data Collection**: 
   - Gathered user reviews from the DANA App on the Google Play Store.
   - Labeled reviews with corresponding sentiment categories (positive, negative, neutral).

2. **Data Preprocessing**:
   - Cleaned raw text to remove noise.
   - Tokenized and transformed the text into numerical format using embedding techniques.

3. **Model Building**:
   - Built an LSTM-based neural network using TensorFlow/Keras.
   - Designed layers with appropriate activation functions and optimizers.

4. **Model Training**:
   - Trained the model using the labeled dataset.
   - Monitored accuracy and loss during training.

5. **Evaluation and Testing**:
   - Tested the model on unseen data to assess its generalization capability.
   - Achieved an accuracy of **85%** on the test set.

---

## Libraries Used

The project leverages the following Python libraries:

- **TensorFlow**: For building and training the LSTM model.
- **Keras**: A high-level API within TensorFlow for constructing neural networks.
- **NumPy**: For numerical operations and handling arrays.
- **Pandas**: For data manipulation and analysis.
- **Scikit-learn**: For data preprocessing, splitting, and evaluation metrics.
- **Matplotlib**: For creating visualizations of data and model performance.
- **Seaborn**: For advanced visualization and insights.

---

## Results

The LSTM model successfully achieved an **accuracy of 85%**, demonstrating its capability to classify sentiments effectively. The model performed particularly well on longer reviews, where context played a significant role in sentiment determination.

---

## Project Limitations

- **Data Limitations**: 
  - The dataset is limited to reviews from the DANA App, which may not generalize well to other applications or domains.
  - Imbalanced classes in the dataset could affect the model's ability to predict less-represented sentiments.

- **Computational Resources**: 
  - Training LSTM networks is resource-intensive, which may limit scalability and experimentation with larger datasets.

- **Domain Generalization**: 
  - The model may require retraining or fine-tuning to work effectively with reviews from other apps or domains.

---

