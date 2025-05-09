# Ellipse Pose Estimator – Project Documentation

## Overview

The **EllipsePoseEstimator** is a computer vision and geometry-based system that estimates the 3D pose (position and orientation) of an elliptical object from a set of 3D points lying on its perimeter. It is designed to work with ellipses with a known aspect ratio and is particularly useful in robotics or surgical navigation systems where estimating the pose of a cylindrical or needle-like object is critical.

This project integrates:
- Principal Component Analysis (PCA) for plane fitting,
- Non-linear optimization for ellipse fitting in 2D,
- Rigid transformation estimation,
- Pose evaluation through synthetic data testing and visualization.

---

## 1. Core Concepts

### What Is Pose Estimation?

Pose estimation involves determining the **position** (x, y, z) and **orientation** (typically expressed as a rotation matrix or Euler angles) of an object in 3D space.

For an ellipse:
- The **position** is the 3D center of the ellipse.
- The **orientation** defines the plane in which the ellipse lies and its in-plane rotation.

### Why Use an Ellipse?

Ellipses commonly appear when a circular object is viewed at an angle. Knowing the object is an ellipse (e.g., from a cross-section of a cylindrical object like a needle) allows us to exploit its geometric properties to recover 3D pose.

---

## 2. Key Modules and Dependencies

- `numpy`: Numerical computations and vectorized operations.
- `scipy.optimize.least_squares`: Robust non-linear fitting of the ellipse model.
- `sklearn.decomposition.PCA`: Plane estimation via PCA.
- `scipy.spatial.transform.Rotation`: Conversions between rotation representations.
- `matplotlib`: 3D visualization and performance plots.

---

## 3. Class: `EllipsePoseEstimator`

### Initialization

```python
def __init__(self, aspect_ratio)
```
- `aspect_ratio (float)`: Ratio \( a/b \) where `a` is the semi-major axis and `b` is the semi-minor axis of the ellipse.

---

### Method: `fit_plane_pca(points)`

**Purpose:** Fit a plane to the 3D points using PCA.

- **Output:**
  - `centroid`: The mean position of the points.
  - `normal`: The normal vector of the best-fit plane.
  - `plane_basis`: Two orthogonal vectors lying in the plane (used for projection).

**Why PCA?** PCA identifies the principal axes of variation in the data. The axis with the smallest variance corresponds to the direction orthogonal to the best-fit plane.

---

### Method: `_project_to_plane(points, centroid, plane_basis)`

**Purpose:** Projects 3D points onto the 2D plane defined by `plane_basis`.

- Resulting 2D points (`u`, `v`) are suitable for ellipse fitting.

---

### Method: `_ellipse_residuals(params, u, v)`

**Purpose:** Defines the residuals for a 2D parametric ellipse.

- Uses geometric constraint:
  \[
  \left(\frac{u'}{a}\right)^2 + \left(\frac{v'}{b}\right)^2 = 1
  \]
- Applies rotation transformation for in-plane orientation.

---

### Method: `_fit_ellipse(u, v)`

**Purpose:** Optimizes ellipse parameters to best fit the 2D projected data.

- Uses `scipy.optimize.least_squares` with the **Huber loss** function to robustly fit the ellipse even in the presence of outliers.
- Returns optimal parameters:
  - Center (`u_c`, `v_c`)
  - Minor axis `b`
  - In-plane angle `theta`

---

### Method: `_compute_yaw_pitch_roll(normal, v1, v2)`

**Purpose:** Constructs a valid 3D rotation matrix from the in-plane basis and normal vector.

- Ensures right-handed coordinate system (if `det < 0`, it corrects it).
- Uses SVD to ensure orthonormality and eliminate numerical drift.

---

### Method: `estimate_pose(points)`

**Main public interface** for users.

**Steps:**
1. Fit the best plane using PCA.
2. Project points into the plane.
3. Fit a 2D ellipse in the plane.
4. Map ellipse center back into 3D.
5. Construct 3D rotation matrix from plane geometry.

**Returns:**
- `position`: 3D center of the ellipse.
- `rot_mat`: 3×3 rotation matrix describing orientation.

---

### `main()`

Runs a simple test case:
- Creates synthetic data.
- Applies a transformation.
- Estimates pose and prints results.

---

## 4. Notes on Robustness and Accuracy

- **Huber loss** helps the ellipse fitting step be less sensitive to noise or outliers.
- **PCA-based plane fitting** assumes that the majority of points lie roughly on a plane.
- The pipeline assumes a known aspect ratio; for unknown ellipses, further estimation or detection steps would be required.

---
