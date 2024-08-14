import numpy as np

# Initialize Kalman Filter parameters
def kalman_filter(prev_state, prev_covariance, measurement, measurement_noise, process_noise):
    # Prediction step
    predicted_state = prev_state
    predicted_covariance = prev_covariance + process_noise

    # Update step
    innovation = measurement - predicted_state
    innovation_covariance = predicted_covariance + measurement_noise
    kalman_gain = predicted_covariance / innovation_covariance
    updated_state = predicted_state + kalman_gain * innovation
    updated_covariance = (1 - kalman_gain) * predicted_covariance

    return updated_state, updated_covariance