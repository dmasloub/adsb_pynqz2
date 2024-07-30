import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import MinimalFCParameters

from src.config import MODEL_STANDARD_DIR, FEATURES, WINDOW_SIZE_STANDARD_AUTOENCODER, STANDARD_Q_THRESHOLD, STANDARD_AUTOENCODER_ENCODING_DIMENSION
from src.data_loader import DataLoader
from src.models.autoencoder import QuantizedAutoencoder
from src.utils.utils import get_windows_data, q_verdict
from src.utils.evaluation import classification_report

def test_autoencoder(custom_paths=None):
    # Load test data
    data_loader = DataLoader(paths=custom_paths)
    data_dict = data_loader.load_data()
    
    # Load preprocessing pipeline
    with open(MODEL_STANDARD_DIR + '/pipeline.pkl', 'rb') as f:
        pipeline = pickle.load(f)
    
    # Load metrics 
    with open(MODEL_STANDARD_DIR + '/metrics.pkl', 'rb') as f:
        metrics = pickle.load(f)
        
    (mu, std) = metrics

    # Define test datasets
    test_datasets = ['test_noise', 'test_landing', 'test_departing', 'test_manoeuver']
    all_reconstruction_errors = []
    actual_labels = {}
    predicted_labels = {}

    # Process each test dataset
    for dataset in test_datasets:
        windowed_test_data_list = []
        windowed_test_labels_list = []

        for df in data_dict[dataset]:
            X, y = get_windows_data(df[FEATURES], df["anomaly"], window_size=WINDOW_SIZE_STANDARD_AUTOENCODER, tsfresh=True)
            windowed_test_data_list.append(X)
            windowed_test_labels_list.append(y)

        extracted_test_features_list = []
        concatenated_test_labels = []

        for X_window, y_window in tqdm(zip(windowed_test_data_list, windowed_test_labels_list), desc="Extracting features from test datasets"):
            features = extract_features(
                X_window, 
                column_id="id", 
                column_sort="time", 
                default_fc_parameters=MinimalFCParameters()
            )
            imputed_features = impute(features)
            extracted_test_features_list.append(imputed_features)
            concatenated_test_labels.extend(y_window)

        X_test = pd.concat(extracted_test_features_list, ignore_index=True)
        y_test = np.array(concatenated_test_labels)

        # Preprocess data
        X_test_n = pipeline.transform(X_test)

        # Determine input_dim from preprocessed data
        input_dim = X_test_n.shape[1]

        # Load model with the correct input_dim
        autoencoder = QuantizedAutoencoder(input_dim=input_dim, encoding_dim=STANDARD_AUTOENCODER_ENCODING_DIMENSION)
        autoencoder.load(MODEL_STANDARD_DIR)

        # Predict
        preds_test = autoencoder.predict(X_test_n)

        # Calculate reconstruction errors
        reconstruction_errors = np.linalg.norm(X_test_n - preds_test, axis=1) ** 2
        all_reconstruction_errors.append(reconstruction_errors)

        # Store actual and predicted labels
        actual_labels[dataset] = y_test
        predicted_labels[dataset] = q_verdict(reconstruction_errors, mu, std, STANDARD_Q_THRESHOLD)

    # Calculate and print accuracy scores for each test dataset
    for dataset in test_datasets:
        acc_score = accuracy_score(actual_labels[dataset], predicted_labels[dataset])
        print(f"Accuracy score for {dataset}: {acc_score}")

    # Generate classification report
    classification_report_df = classification_report(
        [actual_labels[dataset] for dataset in test_datasets], 
        **{dataset: predicted_labels[dataset] for dataset in test_datasets}
    )
    print(classification_report_df)

if __name__ == "__main__":
    # Example usage with default paths
    test_autoencoder()

    # Example usage with custom paths
    # custom_paths = {
    #     "train": "custom_train_path",
    #     "validation": "custom_validation_path",
    #     "test_noise": "custom_test_noise_path",
    #     "test_landing": "custom_test_landing_path",
    #     "test_departing": "custom_test_departing_path",
    #     "test_manoeuver": "custom_test_manoeuver_path"
    # }
    # test_autoencoder(custom_paths=custom_paths)
