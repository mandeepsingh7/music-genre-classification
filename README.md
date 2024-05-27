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
