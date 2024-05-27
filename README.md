# Music Genre Classification
- In this project, we are trying to classify a music file into genres. 
- Genres :
  - blues
  - classical
  - country
  - disco
  - hiphop
  - jazz
  - metal
  - pop
  - reggae
  - rock
### Data Preparation
- Our raw data contains 10 folders, each representing a genre, containing .au files of 30-second duration.

### Feature Extraction
- We write our own function for amplitude envelope. For other features, we use librosa library.
- We extract the following features for Machine Learning models and for DNN :
  - Amplitude envelope
  - Root Mean Square Energy
  - Zero Crossing Rate
  - Spectral Centroid
  - Spectral Rolloff
  - Spectral Bandwidth
  - Spectral Contrast
  - Chroma
  - Tonnetz
  - MFCC values
  - Delta MFCC values
  - Delta Delta MFCC values
  - Tempo
- For each of these features, we consider 7 statistics:
  - min
  - max
  - mean
  - median
  - std
  - skewness
  - kurtosis
- We save all these features into a csv file.
### Machine Learning Models
- We train following models using this csv file :
  - SVM
  - Random Forest
  - KNN
  - XG Boost
  - DNN
### Deep learning Models
- For CNN and LSTM, we use raw MFCC time series values without computing the statistical features.
- Deep Learning models requires comparatively larger amount of data, so we divide each music file of 30 seconds into 10 files of 3 seconds each.
- We perform standardization and then create train and validation data loaders.
- Then we do training using LSTM and CNN Models
### Model Performance
- Accuracy for different models :
  - LSTM - 86.0
  - XG Boost - 81.0
  - CNN - 80.8
  - SVM - 80.5
  - DNN - 80.0
  - Random Forest - 79.0
  - KNN - 77.0
- We get good accuracy using LSTM model because LSTM does really well on time series data. So, we save LSTM state dictionary into a .pt file and then use that for prediction. 
- For prediction, we divide the music file into 10 files, and then pass those 10 files through LSTM model and check what genre is predicted for most files and output that as the predicted genre.

Below is a detailed description of each file and its function:

## Files and Notebooks

- **feature_extraction.ipynb**: 
  This notebook is used for feature extraction. It processes the raw audio files to extract various features and saves them into a CSV file for further use in machine learning models.

- **features.ipynb**: 
  This notebook is for visualizing all the extracted features. It provides important information about each feature and explains why each feature has been considered for the task of music genre classification.

- **ml_training.ipynb**: 
  This notebook is used for training traditional machine learning models such as SVM, Random Forest, KNN, and XGBoost using the extracted features saved in the CSV file.

- **dnn_training.ipynb**: 
  This notebook is used for training a Deep Neural Network (DNN) model using the extracted features.

- **dl_training.ipynb**: 
  This notebook is used for training deep learning models, specifically LSTM and CNN, using MFCC time series values from the audio files.

- **kaggle_dl.ipynb**: 
  We use Kaggle's GPU resources to train the LSTM and CNN models. This notebook shows the training process using Kaggle's platform.

- **LSTM_best_model.pt**: 
  This file contains the saved parameters of the best-performing LSTM model during training. It is used for making predictions.

- **LSTM_last_epoch.pt**: 
  This file contains the saved parameters of the LSTM model from the last epoch of training. It can be used for further training or evaluation.

- **mean_std.npz**: 
  This file contains the mean and standard deviation values calculated from the training data. These values are used for standardizing the data during prediction.

- **predict.ipynb**: 
  This notebook is used for predicting the genre of a given music file using the trained LSTM model. It takes an audio file as input, processes it, and outputs the predicted genre.

## Usage

1. **Feature Extraction**: Use `feature_extraction.ipynb` to extract features from your audio files and save them to a CSV file.
2. **Feature Visualization**: Use `features.ipynb` to visualize and understand the importance of each feature.
3. **Model Training**:
   - For traditional ML models, use `ml_training.ipynb`.
   - For DNN, use `dnn_training.ipynb`.
   - For LSTM and CNN models, use `dl_training.ipynb`. You can also utilize `kaggle_dl.ipynb` to train these models on Kaggle's GPU.
4. **Prediction**: Use `predict.ipynb` to predict the genre of new audio files using the trained LSTM model.

## Notes

- Ensure that you have all the necessary dependencies listed in `requirements.txt` installed before running the notebooks.
- For training deep learning models on Kaggle, ensure that you have access to Kaggle's GPU resources.

Feel free to explore each notebook and modify the code as needed for your specific use case.
