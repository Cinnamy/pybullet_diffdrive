import numpy as np

def mahalanobis(z, z_hat, S):

    d = z - z_hat

    d[1] = np.arctan2(
        np.sin(d[1]),
        np.cos(d[1])
    )

    return d.T @ np.linalg.inv(S) @ d

def compute_ate(estimated, ground_truth):

    estimated = np.array(estimated)
    ground_truth = np.array(ground_truth)

    errors = np.linalg.norm(
        estimated - ground_truth,
        axis=1
    )

    rmse = np.sqrt(np.mean(errors**2))

    return rmse, errors

def compute_landmark_error(observed, true_landmarks):

    if len(observed) == 0:
        return None, None

    errors = []

    for obs in observed:

        distances = np.linalg.norm(
            true_landmarks - obs,
            axis=1
        )

        errors.append(np.min(distances))

    errors = np.array(errors)

    rmse = np.sqrt(np.mean(errors**2))

    return rmse