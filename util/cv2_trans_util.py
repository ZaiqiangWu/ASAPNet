import numpy as np

def get_inverse_trans(trans):
    full_matrix = np.vstack([trans, [0, 0, 1]])

    # Compute the inverse of the 3x3 matrix
    inverse_matrix = np.linalg.inv(full_matrix)

    # Extract the top 2 rows for use with cv2.warpAffine
    inverse_transform_matrix = inverse_matrix[:2, :]
    return inverse_transform_matrix