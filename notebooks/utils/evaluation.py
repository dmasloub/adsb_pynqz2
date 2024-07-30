from collections import OrderedDict
import numpy as np
import pandas as pd 
import scipy
from sklearn.metrics import precision_score, recall_score
from scipy import stats

def get_average_detection_delay(y_true, y_pred):
    """
    Calculate the average detection delay for true labels and predictions.
    
    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.
    
    Returns:
    float: Average detection delay.
    """
    assert len(y_true) == len(y_pred), "The lengths of true and predicted labels must be equal."

    in_window = False
    detected_in_window = False
    detection_delay_sum = 0
    windows_count = 0

    for i in range(len(y_true) - 1):
        curr_true = y_true[i]
        next_true = y_true[i + 1]
        curr_pred = y_pred[i]

        # If in a detection window and event not detected, increment delay
        if in_window and not detected_in_window:
            if curr_pred == 1:
                detected_in_window = True
            else:
                detection_delay_sum += 1

        # Check for start of a new event window
        if (curr_true == 0 and next_true == 1) or (curr_true == 1 and i == 0):
            in_window = True
            windows_count += 1

        # Check for end of an event window
        if curr_true == 1 and next_true == 0:
            in_window = False
            detected_in_window = False

    # Adjust for windows not padded at the end or beginning
    if y_true[-1] == 1:
        detection_delay_sum += 1
    if y_true[0] == 1:
        detection_delay_sum += 1

    return detection_delay_sum / windows_count if windows_count > 0 else 0

def classification_report(y_true_l, **kwargs):
    """
    Generate a classification report with benchmark results.
    
    Parameters:
    y_true_l (list): List of true labels.
    kwargs (dict): Dictionary of classifier names and their predictions.
    
    Returns:
    pd.DataFrame: DataFrame containing the classification report with metrics for each detector.
    """
    detector_dict = OrderedDict()
    
    # Adding Perfect Detector for benchmark comparison
    detector_dict["Perfect Detector"] = y_true_l[0], y_true_l[0]

    # Adding each classifier's predictions and true labels to the dictionary
    for i, (key, value) in enumerate(kwargs.items()):
        assert len(y_true_l[i]) == len(value), "Length of true labels and predictions must be equal"
        detector_dict[key] = value, y_true_l[i]

    # Adding Null Detectors (always predicting 0 or 1)
    detector_dict["Null Detector 1"] = [0] * len(y_true_l[0]), y_true_l[0]
    detector_dict["Null Detector 2"] = [1] * len(y_true_l[0]), y_true_l[0]

    # Adding Random Detector
    np.random.seed(0)
    detector_dict["Random Detector"] = np.where(np.random.rand(len(y_true_l[0])) >= 0.5, 1, 0), y_true_l[0]

    data = []

    # Calculating precision, recall, and average detection delay for each detector
    for key, (pred, true) in detector_dict.items():
        precision = round(precision_score(true, pred), 3)
        recall = round(recall_score(true, pred), 3)
        avg_delay = round(get_average_detection_delay(true, pred), 3)
        data.append([key, precision, recall, avg_delay])
  
    return pd.DataFrame(columns=["Detector", "Precision", "Recall", "Average Detection Delay"], data=data)


def anomaly_score(score, mu, sig):
    """
    Calculate anomaly scores based on the CDF of a normal distribution.
    
    Parameters:
    score (array-like): List of scores given by the model.
    mu (float): Mean of the normal distribution.
    sig (float): Standard deviation of the normal distribution.
    
    Returns:
    array-like: Anomaly scores.
    """
    # Input validation
    if not isinstance(score, (list, np.ndarray)):
        raise ValueError("score must be a list or array-like")
    if not isinstance(mu, (int, float)):
        raise ValueError("mu must be a numeric value")
    if not isinstance(sig, (int, float)):
        raise ValueError("sig must be a numeric value")

    return 1 - stats.norm.sf(score, mu, sig)

def q_verdict(x, mu, sig, n=0.1):
    """
    Provide a verdict on anomaly based on the CDF of a normal distribution.
    
    Parameters:
    x (array-like): List of scores given by the model.
    mu (float): Mean of the normal distribution.
    sig (float): Standard deviation of the normal distribution.
    n (float): Threshold for anomaly detection (default is 0.1).
    
    Returns:
    np.ndarray: Array of verdicts (1 for anomaly, 0 for normal).
    """
    # Input validation
    if not isinstance(x, (list, np.ndarray, pd.Series)):
        raise ValueError("x must be a list or array-like")
    if not isinstance(mu, (int, float)):
        raise ValueError("mu must be a numeric value")
    if not isinstance(sig, (int, float)):
        raise ValueError("sig must be a numeric value")
    if not isinstance(n, (int, float)):
        raise ValueError("n must be a numeric value")

    # Ensure x is a NumPy array
    x = np.asarray(x)

    anomaly_scores = anomaly_score(x, mu, sig)
    return np.where(anomaly_scores >= 1 - n, 1, 0)

def test_normal_dist(x, alpha=0.05):
    """
    Perform the Shapiro-Wilk test for normality.
    
    Parameters:
    x (array-like): The array containing the sample to be tested.
    alpha (float): Significance level for rejection of the null hypothesis (default is 0.05).
    
    Returns:
    None
    """
    # Input validation
    if not isinstance(x, (list, np.ndarray)):
        raise ValueError("x must be a list or array-like")
    if not isinstance(alpha, (int, float)):
        raise ValueError("alpha must be a numeric value")

    # For N > 5000 the W test statistic is accurate but the p-value may not be.
    # The chance of rejecting the null hypothesis when it is true is close to 5% regardless of sample size.
    length = min(len(x), 2500)
    stats, p = scipy.stats.shapiro(x[:length])
    print(f"p-value: {p}")
    if p < alpha:  # null hypothesis: the data was drawn from a normal distribution
        print("The null hypothesis can be rejected (data is not normal).")
    else:
        print("The null hypothesis cannot be rejected (data is normal).")