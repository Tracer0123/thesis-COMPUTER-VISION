import cv2
import numpy as np

# Load an image from file
#image = cv2.imread('lib/optical_flow/000000.png')

# Convert the image to grayscale
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply optical flow estimation (not shown in this code)
# You should use an optical flow algorithm to compute the flow vectors

# Apply RANSAC to estimate the FOE
def estimate_foe(flow_vectors, num_iterations=100, inlier_threshold=5):
    best_foe = (0, 0)  # Initialize FOE as (0, 0)
    best_inliers = 0
    best_k = [0,0]
    for _ in range(num_iterations):
        # Randomly select two indices
        s = np.shape(np.array(flow_vectors))
        if s[0] < 2:
            continue
        indices = np.random.choice(len(flow_vectors), 2, replace=False)
        vector1, vector2 = flow_vectors[indices[0]], flow_vectors[indices[1]]

        # Calculate the FOE using the two selected vectors
        x1, y1 = vector1
        x2, y2 = vector2
        a = y1 - y2
        b = x2 - x1
        c = x1 * y2 - x2 * y1

        # Calculate the number of inliers
        inliers = sum(
            abs(a * x + b * y + c) < inlier_threshold for x, y in flow_vectors
        )
        i = 0
        k = []
        for x, y in flow_vectors:
            if abs(a * x + b * y + c) < inlier_threshold:
                k.append(i)
            i += 1


        # Update FOE and inliers if this model is better
        if inliers > best_inliers:
            best_foe = (-c / (0.0000000001+a), -c / (0.0000000001+b))
            best_inliers = inliers
            best_k = k
    #print(best_k)
    return best_foe, best_k

# Sample: Replace this with actual optical flow vectors
# flow_vectors should be a list of (x, y) pairs representing the optical flow at each point
#flow_vectors = [(10, 20), (15, 25), (25, 40), (5, 10), (12, 23), (18, 30)]
#flow_vectors = q2
# Estimate the FOE using RANSAC
#foe_x, foe_y = estimate_foe(flow_vectors)
#print(foe_x,foe_y)
# Draw the FOE on the original image
#cv2.circle(image, (int(foe_x), int(foe_y)), 5, (0, 0, 255), -1)

# Display the image with FOE
#cv2.imshow('FOE using RANSAC', image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
