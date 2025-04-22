import cv2
import numpy as np

img = cv2.imread("/home/imad/SurgicalToolsPose/Scripts/img.jpg", cv2.IMREAD_GRAYSCALE)

#blurred = cv2.GaussianBlur(img, (3, 3), 0)

# Edge detection
grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
magnitude = cv2.magnitude(grad_x, grad_y)

# Thresh
_, edges = cv2.threshold(magnitude, 30, 255, cv2.THRESH_BINARY)
edges = edges.astype(np.uint8)

#cleaning
kernel = np.ones((3, 3), np.uint8)
cleaned = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

skeleton = cv2.ximgproc.thinning(cleaned) if hasattr(cv2, 'ximgproc') else cleaned

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned)

output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

for i in range(1, num_labels):  
    x, y, w, h, area = stats[i]
    if area < 20:
        continue  
    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 1)
    cv2.putText(output, f"#{i}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

cv2.imshow("Original", img)
cv2.imshow("Detected Cracks", output)
cv2.imshow("Edges", edges)
cv2.imshow("Cleaned", cleaned)
cv2.waitKey(0)
cv2.destroyAllWindows()
