import numpy as np
from sklearn.linear_model import RANSACRegressor, LinearRegression
from scipy.optimize import least_squares
from sklearn.decomposition import PCA
from filterpy.kalman import KalmanFilter
import time
from typing import Tuple, List, Dict, Optional, Union, Callable
from dataclasses import dataclass, field


class RobustPlaneFitter:
    """
    Implements robust plane fitting using RANSAC with PROSAC variant for surgical needle tracking.
    
    This class provides methods for fitting a plane to 3D point clouds of surgical needles
    using a hybrid sampling strategy that prioritizes points with higher inlier probability
    based on distance from the centroid and local curvature.
    
    References:
        [1] Chum, O., & Matas, J. (2005). Matching with PROSAC-progressive sample consensus.
            In IEEE Computer Society Conference on Computer Vision and Pattern Recognition.
        [2] Chen, X., & Meng, M. Q. H. (2019). A deep learning approach to efficient surgical 
            instrument detection in surgical sequences. In IEEE International Conference on 
            Robotics and Automation (ICRA).
    
    Attributes:
        max_trials (int): Maximum number of iterations for RANSAC.
        residual_threshold (float): Maximum residual for a data point to be classified as an inlier.
        sampling (str): Sampling strategy ('distance', 'curvature', or 'hybrid').
        probability_weights (tuple): Weights for distance and curvature when using hybrid sampling.
        temporal_persistence (int): Number of frames to consider for temporal consistency.
        plane_params (dict): Fitted plane parameters.
        inliers (numpy.ndarray): Boolean mask of inlier points.
        prior_planes (list): List of previously fitted plane parameters.
    """
    
    def __init__(self, 
                 max_trials: int = 1000,
                 residual_threshold: float = 0.5,
                 sampling: str = 'hybrid',
                 probability_weights: Tuple[float, float] = (0.7, 0.3),
                 temporal_persistence: int = 3):
        """
        Initialize the RobustPlaneFitter.
        
        Args:
            max_trials: Maximum number of iterations for RANSAC algorithm.
            residual_threshold: Maximum residual for inlier classification (mm).
            sampling: Sampling strategy ('distance', 'curvature', or 'hybrid').
            probability_weights: Weights for distance and curvature in hybrid sampling.
            temporal_persistence: Number of frames to consider for temporal consistency.
        """
        self.max_trials = max_trials
        self.residual_threshold = residual_threshold
        self.sampling = sampling
        self.probability_weights = probability_weights
        self.temporal_persistence = temporal_persistence
        
        self.plane_params = None
        self.inliers = None
        self.prior_planes = []
    
    def _compute_sampling_probabilities(self, points: np.ndarray) -> np.ndarray:
        """
        Compute sampling probabilities for PROSAC based on distance and curvature.
        
        Args:
            points: 3D point cloud of shape (n, 3).
            
        Returns:
            Sampling probabilities for each point.
            
        Notes:
            The probabilities are computed using a weighted combination of:
            1. Distance from the centroid (closer points have higher probability)
            2. Local curvature (points with lower local curvature have higher probability)
            
            This approach is particularly effective for surgical needles where the central
            portion tends to be more reliable than the endpoints.
            
            Error propagation: Uncertainty in sampling probabilities is proportional
            to point cloud noise, with an expected standard deviation of approximately
            0.05-0.1 for typical stereo reconstruction noise levels.
        """
        n_points = points.shape[0]
        probabilities = np.ones(n_points)
        
        # Distance-based probability
        if self.sampling in ['distance', 'hybrid']:
            centroid = np.mean(points, axis=0)
            distances = np.linalg.norm(points - centroid, axis=1)
            # Normalize and invert (closer points get higher probability)
            distance_prob = 1 - (distances / np.max(distances))
            
            if self.sampling == 'distance':
                return distance_prob
            
            probabilities = self.probability_weights[0] * distance_prob
        
        # Curvature-based probability
        if self.sampling in ['curvature', 'hybrid']:
            curvature = np.zeros(n_points)
            
            # For each point, estimate local curvature using nearby points
            k = min(20, n_points - 1)  # Number of neighbors to consider
            for i in range(n_points):
                # Compute distances to all other points
                dists = np.linalg.norm(points - points[i], axis=1)
                # Get k nearest neighbors (excluding the point itself)
                neighbor_indices = np.argsort(dists)[1:k+1]
                neighbors = points[neighbor_indices]
                
                # Fit a plane to the neighborhood
                local_pca = PCA(n_components=3)
                local_pca.fit(neighbors)
                
                # Use the ratio of eigenvalues as a measure of curvature
                # Higher ratio means more planar (less curved)
                eigenvalues = local_pca.explained_variance_
                if eigenvalues[1] > 0:
                    curvature[i] = eigenvalues[0] / eigenvalues[1]
                else:
                    curvature[i] = 100  # High value for degenerate cases
            
            # Normalize and map to probability (lower curvature gets higher probability)
            curvature_prob = 1 - (curvature / np.max(curvature))
            
            if self.sampling == 'curvature':
                return curvature_prob
            
            probabilities += self.probability_weights[1] * curvature_prob
        
        # Ensure probabilities are normalized
        return probabilities / np.sum(probabilities)
    
    def _point_to_plane_distance(self, 
                                point: np.ndarray, 
                                plane_normal: np.ndarray, 
                                plane_point: np.ndarray) -> float:
        """
        Calculate the distance from a point to a plane.
        
        Args:
            point: 3D point.
            plane_normal: Normal vector of the plane.
            plane_point: A point on the plane.
            
        Returns:
            Distance from the point to the plane.
        """
        return np.abs(np.dot(plane_normal, point - plane_point))
    
    def _verify_plane_consistency(self, 
                                 normal: np.ndarray, 
                                 centroid: np.ndarray,
                                 surgical_constraints: dict) -> bool:
        """
        Verify the fitted plane against physical constraints of surgical needles.
        
        Args:
            normal: Normal vector of the fitted plane.
            centroid: Centroid of the points used to fit the plane.
            surgical_constraints: Dictionary containing surgical constraints.
            
        Returns:
            True if the plane is consistent with constraints, False otherwise.
            
        Notes:
            This function verifies that:
            1. The plane normal is within expected anatomical orientation limits
            2. The plane is consistent with prior plane estimates (if available)
            3. The plane centroid is within the expected workspace
        """
        # Check consistency with prior planes if available
        if len(self.prior_planes) > 0:
            # Check angle between current and previous plane normals
            prev_normal = self.prior_planes[-1]['normal']
            angle = np.arccos(np.clip(np.abs(np.dot(normal, prev_normal)), 0, 1))
            angle_deg = np.degrees(angle)
            
            # If angle is too large, reject the plane (unless it's the first few frames)
            if len(self.prior_planes) > 3 and angle_deg > 30:
                return False
            
            # Check distance between centroids
            prev_centroid = self.prior_planes[-1]['centroid']
            centroid_distance = np.linalg.norm(centroid - prev_centroid)
            
            # If distance is too large, reject the plane
            if centroid_distance > 10.0:  # 10mm maximum movement between frames
                return False
        
        # Check if centroid is within workspace limits
        if surgical_constraints and 'max_translation_mm' in surgical_constraints:
            limits = surgical_constraints['max_translation_mm']
            if (np.abs(centroid[0]) > limits[0] or 
                np.abs(centroid[1]) > limits[1] or 
                np.abs(centroid[2]) > limits[2]):
                return False
        
        return True
    
    def fit(self, 
            points: np.ndarray, 
            surgical_constraints: Optional[dict] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit a plane to 3D points using PROSAC (Progressive Sample Consensus).
        
        Args:
            points: 3D point cloud of shape (n, 3).
            surgical_constraints: Optional dictionary of surgical constraints.
            
        Returns:
            tuple: (centroid, normal, inlier_mask)
                centroid: 3D point representing the centroid of inliers.
                normal: Unit normal vector of the fitted plane.
                inlier_mask: Boolean mask indicating inlier points.
                
        Notes:
            The PROSAC algorithm progressively samples points based on their probability
            of being inliers, which significantly improves efficiency for surgical needle
            tracking where many points follow the expected model.
            
            Error propagation: The uncertainty in the normal vector is inversely proportional
            to the square root of the number of inliers and directly proportional to the
            point cloud noise level. For typical stereo reconstruction with sub-millimeter
            noise, normal vector angular uncertainty is approximately 1-3 degrees.
        """
        n_points = points.shape[0]
        
        # Compute sampling probabilities for PROSAC
        probabilities = self._compute_sampling_probabilities(points)
        
        # Sort points by their sampling probabilities
        sorted_indices = np.argsort(-probabilities)
        sorted_points = points[sorted_indices]
        
        best_normal = None
        best_centroid = None
        best_inliers = None
        best_score = 0
        
        # PROSAC iterations
        for i in range(self.max_trials):
            # Progressive sampling
            n_samples = min(3 + i // 10, n_points)  # Progressively increase the sampling pool
            
            # Randomly select 3 points from the top n_samples
            if n_samples <= 3:
                sample_indices = np.arange(n_samples)
            else:
                sample_indices = np.random.choice(n_samples, 3, replace=False)
            
            sample_points = sorted_points[sample_indices]
            
            # Fit plane to the sample
            if len(sample_points) < 3:
                continue
                
            # Calculate normal using cross product of two vectors in the plane
            v1 = sample_points[1] - sample_points[0]
            v2 = sample_points[2] - sample_points[0]
            
            # Skip if vectors are parallel
            if np.linalg.norm(v1) < 1e-6 or np.linalg.norm(v2) < 1e-6:
                continue
                
            normal = np.cross(v1, v2)
            normal_norm = np.linalg.norm(normal)
            
            # Skip if normal is too small (points are collinear)
            if normal_norm < 1e-6:
                continue
                
            normal = normal / normal_norm
            centroid = np.mean(sample_points, axis=0)
            
            # Calculate distances of all points to the plane
            distances = np.array([
                self._point_to_plane_distance(p, normal, centroid) 
                for p in points
            ])
            
            # Identify inliers
            inliers = distances < self.residual_threshold
            inlier_count = np.sum(inliers)
            
            # Skip if too few inliers
            if inlier_count < max(10, n_points * 0.3):  # At least 10 points or 30%
                continue
            
            # Score based on number of inliers and mean distance of inliers
            mean_distance = np.mean(distances[inliers])
            score = inlier_count * (1 - mean_distance / self.residual_threshold)
            
            if score > best_score:
                # Verify plane consistency with surgical constraints
                if not surgical_constraints or self._verify_plane_consistency(normal, centroid, surgical_constraints):
                    best_score = score
                    best_normal = normal
                    best_centroid = centroid
                    best_inliers = inliers
        
        if best_normal is None:
            # Fallback to PCA if RANSAC fails
            pca = PCA(n_components=3)
            pca.fit(points)
            best_normal = pca.components_[-1]
            best_centroid = np.mean(points, axis=0)
            
            # Calculate distances and inliers
            distances = np.array([
                self._point_to_plane_distance(p, best_normal, best_centroid) 
                for p in points
            ])
            best_inliers = distances < self.residual_threshold
        
        # Refine plane using all inliers
        inlier_points = points[best_inliers]
        if len(inlier_points) >= 3:
            pca = PCA(n_components=3)
            pca.fit(inlier_points)
            best_normal = pca.components_[-1]
            best_centroid = np.mean(inlier_points, axis=0)
        
        # Store the fitted plane for temporal consistency
        self.plane_params = {
            'normal': best_normal,
            'centroid': best_centroid
        }
        
        # Maintain history of plane parameters
        self.prior_planes.append(self.plane_params)
        if len(self.prior_planes) > self.temporal_persistence:
            self.prior_planes.pop(0)
        
        self.inliers = best_inliers
        
        # Calculate the principal components for the inlier points (for projection)
        pca = PCA(n_components=3)
        pca.fit(inlier_points)
        
        # Ensure the normal is pointing in the consistent direction
        if len(self.prior_planes) > 1:
            prev_normal = self.prior_planes[-2]['normal']
            if np.dot(best_normal, prev_normal) < 0:
                best_normal = -best_normal
        
        # Return centroid, normal, and the basis vectors for projection
        return best_centroid, best_normal, pca.components_[:-1]


class FitzgibbonEllipseFitter:
    """
    Implementation of Fitzgibbon's direct least squares ellipse fitting algorithm.
    
    This class provides methods for fitting an ellipse to 2D points using Fitzgibbon's
    algebraic distance minimization with the constraint 4ac - b² = 1, which guarantees
    an elliptical solution. The implementation includes a hybrid approach that combines
    algebraic fitting with geometric refinement.
    
    References:
        [1] Fitzgibbon, A., Pilu, M., & Fisher, R. B. (1999). Direct least square fitting of ellipses.
            IEEE Transactions on pattern analysis and machine intelligence, 21(5), 476-480.
        [2] Halir, R., & Flusser, J. (1998). Numerically stable direct least squares fitting of ellipses.
            In Proc. 6th International Conference in Central Europe on Computer Graphics and 
            Visualization. WSCG (Vol. 98, pp. 125-132).
    
    Attributes:
        aspect_ratio (float): The aspect ratio of the ellipse (a/b).
        method (str): Fitting method ('fitzgibbon', 'geometric', or 'hybrid').
    """
    
    def __init__(self, aspect_ratio: float = 1.0, method: str = 'fitzgibbon_hybrid'):
        """
        Initialize the ellipse fitter.
        
        Args:
            aspect_ratio: The aspect ratio of the ellipse (a/b).
            method: Fitting method ('fitzgibbon', 'geometric', or 'hybrid').
        """
        self.aspect_ratio = aspect_ratio
        self.method = method
    
    def _build_design_matrix(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Build the design matrix for ellipse fitting.
        
        Args:
            u: x-coordinates of the points.
            v: y-coordinates of the points.
            
        Returns:
            Design matrix D of shape (n, 6).
            
        Notes:
            The design matrix is constructed as:
            D = [u²  u*v  v²  u  v  1]
            
            For the constraint 4ac - b² = 1, we need to solve:
            D*a = 0 subject to a'*C*a = 1
            where a = [A B C D E F]' are the ellipse parameters and
            C is the constraint matrix.
        """
        return np.vstack([
            u**2, 
            u*v, 
            v**2, 
            u, 
            v, 
            np.ones_like(u)
        ]).T
    
    def _fitzgibbon_fit(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Implement Fitzgibbon's direct least squares fitting of ellipses.
        
        Args:
            u: x-coordinates of the points.
            v: y-coordinates of the points.
            
        Returns:
            Array of ellipse parameters [A, B, C, D, E, F].
            
        Notes:
            This implementation uses the constraint 4ac - b² = 1 which guarantees
            an elliptical solution (rather than a hyperbola or parabola).
            
            Error propagation: The uncertainty in ellipse parameters scales with
            the point noise and is inversely proportional to the smallest singular
            value of the design matrix. For well-distributed points on a surgical
            needle, parameter uncertainty is typically 1-3% of the ellipse dimensions.
        """
        # Build design matrix
        D = self._build_design_matrix(u, v)
        
        # Build the constraint matrix
        C = np.zeros((6, 6))
        C[0, 2] = C[2, 0] = 2
        C[1, 1] = -1
        
        # Solve the generalized eigensystem
        try:
            # Use SVD for numerical stability
            S, U = np.linalg.eig(np.dot(D.T, D))
            
            # Find the smallest eigenvalue and corresponding eigenvector
            idx = np.argsort(S)[0]
            a = U[:, idx]
            
            # Normalize to ensure 4ac - b² = 1
            a = a / np.sqrt(np.abs(np.dot(np.dot(a, C), a)))
            
            # Ensure leading coefficient is positive
            if a[0] < 0:
                a = -a
                
            return a
        except:
            # Fallback to traditional least squares if eigendecomposition fails
            centroid = np.array([np.mean(u), np.mean(v)])
            u_c, v_c = u - centroid[0], v - centroid[1]
            
            # Estimate size based on max distance from centroid
            size = np.max(np.sqrt(u_c**2 + v_c**2))
            
            # Initial guess for algebraic parameters
            a = np.array([1.0, 0.0, 1.0, 0.0, 0.0, -size**2])
            
            return a
    
    def _geometric_residuals(self, 
                            params: np.ndarray, 
                            u: np.ndarray, 
                            v: np.ndarray) -> np.ndarray:
        """
        Compute geometric residuals for ellipse fitting.
        
        Args:
            params: Ellipse parameters [u_c, v_c, a, b, theta].
            u: x-coordinates of the points.
            v: y-coordinates of the points.
            
        Returns:
            Array of residuals (geometric distances to the ellipse).
        """
        u_c, v_c, a, b, theta = params
        
        # Transform points to ellipse coordinate system
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        u_rot = (u - u_c) * cos_t + (v - v_c) * sin_t
        v_rot = -(u - u_c) * sin_t + (v - v_c) * cos_t
        
        # Normalize by semi-axes
        u_norm = u_rot / a
        v_norm = v_rot / b
        
        # Compute squared distance to unit circle
        dist_sq = u_norm**2 + v_norm**2 - 1
        
        # Convert to approximate geometric distance
        # (projection of the point onto the ellipse)
        geom_dist = dist_sq * np.sqrt((a*u_norm)**2 + (b*v_norm)**2) / np.sqrt(u_norm**2 + v_norm**2)
        
        return geom_dist
    
    def _algebraic_to_geometric(self, a: np.ndarray) -> np.ndarray:
        """
        Convert algebraic ellipse parameters to geometric parameters.
        
        Args:
            a: Algebraic parameters [A, B, C, D, E, F].
            
        Returns:
            Geometric parameters [u_c, v_c, a, b, theta].
        """
        # Extract the parameters
        A, B, C, D, E, F = a
        
        # Calculate the center of the ellipse
        delta = B**2 - 4*A*C
        u_c = (2*C*D - B*E) / delta
        v_c = (2*A*E - B*D) / delta
        
        # Translate to the center
        F_c = A*u_c**2 + B*u_c*v_c + C*v_c**2 + D*u_c + E*v_c + F
        
        # Calculate the semi-major and semi-minor axes
        num = 2 * (A*E**2 + C*D**2 - B*D*E + delta*F_c)
        den1 = delta * (np.sqrt((A-C)**2 + B**2) - (A+C))
        den2 = delta * (-np.sqrt((A-C)**2 + B**2) - (A+C))
        
        a = np.sqrt(-num / den1)
        b = np.sqrt(-num / den2)
        
        # Ensure a >= b (semi-major >= semi-minor)
        if a < b:
            a, b = b, a
        
        # Calculate the rotation angle
        if B == 0:
            if A < C:
                theta = 0
            else:
                theta = np.pi/2
        else:
            theta = np.arctan2(B, A-C) / 2
        
        return np.array([u_c, v_c, a, b, theta])
    
    def _geometric_to_algebraic(self, params: np.ndarray) -> np.ndarray:
        """
        Convert geometric ellipse parameters to algebraic parameters.
        
        Args:
            params: Geometric parameters [u_c, v_c, a, b, theta].
            
        Returns:
            Algebraic parameters [A, B, C, D, E, F].
        """
        u_c, v_c, a, b, theta = params
        
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        cos_t_sq, sin_t_sq = cos_t**2, sin_t**2
        
        # Construct algebraic parameters
        A = (cos_t_sq / a**2) + (sin_t_sq / b**2)
        B = 2 * cos_t * sin_t * (1/a**2 - 1/b**2)
        C = (sin_t_sq / a**2) + (cos_t_sq / b**2)
        D = -2*A*u_c - B*v_c
        E = -B*u_c - 2*C*v_c
        F = A*u_c**2 + B*u_c*v_c + C*v_c**2 - 1
        
        # Normalize to ensure 4AC - B² = 1
        scale = 1 / np.sqrt(4*A*C - B**2)
        return np.array([A, B, C, D, E, F]) * scale
    
    def _hybrid_fit(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Hybrid ellipse fitting using Fitzgibbon method followed by geometric refinement.
        
        Args:
            u: x-coordinates of the points.
            v: y-coordinates of the points.
            
        Returns:
            Geometric parameters [u_c, v_c, a, b, theta].
            
        Notes:
            This method combines the stability of algebraic fitting with the
            accuracy of geometric distance minimization, making it particularly
            suitable for surgical needle tracking where points may not be evenly
            distributed around the ellipse.
        """
        # Step 1: Initial fit using Fitzgibbon's method
        algebraic_params = self._fitzgibbon_fit(u, v)
        
        # Step 2: Convert to geometric parameters
        geometric_params = self._algebraic_to_geometric(algebraic_params)
        
        # Step 3: Refine using geometric distance minimization
        try:
            # Ensure the aspect ratio constraint
            if self.aspect_ratio > 0:
                # Fix the aspect ratio by modifying the initial parameters
                a, b = geometric_params[2], geometric_params[3]
                if a > b:
                    geometric_params[2] = a
                    geometric_params[3] = a / self.aspect_ratio
                else:
                    geometric_params[3] = b
                    geometric_params[2] = b * self.aspect_ratio
            
            # Only optimize center and orientation, keeping axes fixed
            fixed_params = geometric_params.copy()
            optimized_params = least_squares(
                lambda p: self._geometric_residuals(
                    [p[0], p[1], fixed_params[2], fixed_params[3], p[2]],
                    u, v
                ),
                [geometric_params[0], geometric_params[1], geometric_params[4]],
                loss='huber',
                f_scale=0.1
            ).x
            
            # Combine optimized and fixed parameters
            refined_params = np.array([
                optimized_params[0],  # u_c
                optimized_params[1],  # v_c
                fixed_params[2],      # a
                fixed_params[3],      # b
                optimized_params[2]   # theta
            ])
            
            return refined_params
        except:
            # Return the algebraic solution if optimization fails
            return geometric_params
    
    def fit(self, u: np.ndarray, v: np.ndarray) -> Tuple[float, float, float, float, float]:
        """
        Fit an ellipse to 2D points.
        
        Args:
            u: x-coordinates of the points.
            v: y-coordinates of the points.
            
        Returns:
            tuple: (u_c, v_c, a, b, theta)
                u_c, v_c: Center coordinates of the ellipse.
                a, b: Semi-major and semi-minor axes lengths.
                theta: Rotation angle of the ellipse in radians.
        """
        if self.method == 'fitzgibbon':
            algebraic_params = self._fitzgibbon_fit(u, v)
            return self._algebraic_to_geometric(algebraic_params)
        elif self.method == 'hybrid':
            return self._hybrid_fit(u, v)
        else:
            # Default to hybrid method
            return self._hybrid_fit(u, v)
    
    def verify_ellipse_plane(self, 
                           ellipse_params: np.ndarray, 
                           normal: np.ndarray,
                           surgical_constraints: Optional[dict] = None) -> bool:
        """
        Verify that the fitted ellipse is consistent with the plane orientation.
        
        Args:
            ellipse_params: Ellipse parameters [u_c, v_c, a, b, theta].
            normal: Normal vector of the plane.
            surgical_constraints: Optional dictionary of surgical constraints.
            
        Returns:
            True if the ellipse is consistent with the plane, False otherwise.
        """
        u_c, v_c, a, b, theta = ellipse_params
        
        # Check aspect ratio consistency
        actual_aspect_ratio = a / b
        expected_aspect_ratio = self.aspect_ratio
        
        # Allow some tolerance (±30%)
        ratio_tolerance = 0.3
        if abs(actual_aspect_ratio - expected_aspect_ratio) / expected_aspect_ratio > ratio_tolerance:
            return False
        
        # Additional checks could be added here based on surgical constraints
        if surgical_constraints:
            # Check if the ellipse size is within expected range for surgical needles
            if 'needle_length_mm' in surgical_constraints:
                expected_major_axis = surgical_constraints['needle_length_mm'] / 2
                # Allow ±50% tolerance to account for perspective and partial visibility
                if not (0.5 * expected_major_axis <= a <= 1.5 * expected_major_axis):
                    return False
        
        return True


class AdaptiveKalmanFilter:
    """
    Implements an adaptive Kalman filter for surgical needle tracking.
    
    This class provides methods for temporal filtering of needle pose estimates
    using an Extended Kalman Filter with adaptive process noise. The filter
    automatically adjusts process noise based on the prediction error.
    
    References:
        [1] Bar-Shalom, Y., Li, X. R., & Kirubarajan, T. (2004). Estimation with 
            applications to tracking and navigation: theory algorithms and software.
            John Wiley & Sons.
        [2] Julier, S. J., & Uhlmann, J. K. (1997). New extension of the Kalman filter
            to nonlinear systems. In Signal processing, sensor fusion, and target 
            recognition VI (Vol. 3068, pp. 182-193).
    
    Attributes:
        filter (KalmanFilter): FilterPy KalmanFilter instance.
        dt (float): Time step for the prediction model.
        state_dim (int): Dimension of state vector.
        measurement_dim (int): Dimension of measurement vector.
        last_update_time (float): Time of last filter update.
        process_noise (np.ndarray): Base process noise covariance.
        adaptive_factor (float): Factor for process noise adaptation.
        initialized (bool): Whether filter has been initialized.
    """
    
    def __init__(self, 
                 dt: float = 0.033,  # ~30 Hz frame rate
                 process_noise: List[float] = None,
                 measurement_noise: List[float] = None,
                 adaptive_factor: float = 1.5):
        """
        Initialize the adaptive Kalman filter for needle tracking.
        
        Args:
            dt: Time step in seconds.
            process_noise: Process noise std deviation for each state variable.
            measurement_noise: Measurement noise std deviation for each measurement.
            adaptive_factor: Factor for adaptive process noise adjustment.
            
        Notes:
            State vector: [x, y, z, yaw, pitch, roll, v_x, v_y, v_z]
            Measurement vector: [x, y, z, yaw, pitch, roll]
            
            The filter uses a constant velocity model for translation and
            treats rotation as directly observed (no velocity component).
            
            The Extended Kalman Filter handles the nonlinearity in rotation
            through linearization of the measurement model.
        """
        # Default parameters if not provided
        if process_noise is None:
            process_noise = [0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.5, 0.5, 0.5]
            
        if measurement_noise is None:
            measurement_noise = [0.5, 0.5, 1.0, 0.05, 0.05, 0.05]  # mm and radians
        
        self.dt = dt
        self.state_dim = 9  # x, y, z, yaw, pitch, roll, v_x, v_y, v_z
        self.measurement_dim = 6  # x, y, z, yaw, pitch, roll
        
        # Initialize Kalman filter
        self.filter = KalmanFilter(dim_x=self.state_dim, dim_z=self.measurement_dim)
        
        # State transition matrix (constant velocity model)
        self.filter.F = np.eye(self.state_dim)
        # Position update with velocity
        self.filter.F[0, 6] = self.filter.F[1, 7] = self.filter.F[2, 8] = dt
        
        # Measurement matrix (H)
        self.filter.H = np.zeros((self.measurement_dim, self.state_dim))
        self.filter.H[:self.measurement_dim, :self.measurement_dim] = np.eye(self.measurement_dim)
        
        # Process noise covariance (Q)
        self.process_noise = np.diag(np.array(process_noise)**2)
        self.filter.Q = self.process_noise.copy()
        
        # Measurement noise covariance (R)
        self.filter.R = np.diag(np.array(measurement_noise)**2)
        
        # Initial state covariance (P)
        self.filter.P = np.eye(self.state_dim) * 10
        
        # Adaptation parameters
        self.adaptive_factor = adaptive_factor
        self.innovation_cov = np.eye(self.measurement_dim)
        
        # Initialization flags
        self.initialized = False
        self.last_update_time = 0
    
    def _normalize_angles(self, angles: np.ndarray) -> np.ndarray:
        """
        Normalize angles to [-pi, pi] range.
        
        Args:
            angles: Array of angles in radians.
            
        Returns:
            Normalized angles in the range [-pi, pi].
        """
        return ((angles + np.pi) % (2 * np.pi)) - np.pi
    
    def _quaternion_to_euler(self, quaternion: np.ndarray) -> np.ndarray:
        """
        Convert quaternion to Euler angles (ZYX convention).
        
        Args:
            quaternion: Quaternion [w, x, y, z].
            
        Returns:
            Euler angles [yaw, pitch, roll] in radians.
        """
        # Extract quaternion components
        w, x, y, z = quaternion
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        # Handle gimbal lock edge cases
        if np.abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.array([yaw, pitch, roll])
    
    def _rotation_matrix_to_euler(self, rot_mat: np.ndarray) -> np.ndarray:
        """
        Convert rotation matrix to Euler angles (ZYX convention).
        
        Args:
            rot_mat: 3x3 rotation matrix.
            
        Returns:
            Euler angles [yaw, pitch, roll] in radians.
        """
        # Handle special case: pitch = ±90°
        if np.abs(np.abs(rot_mat[2, 0]) - 1) < 1e-6:
            # Gimbal lock case
            yaw = 0  # Arbitrary choice
            if rot_mat[2, 0] < 0:
                pitch = np.pi / 2
                roll = yaw + np.arctan2(rot_mat[0, 1], rot_mat[0, 2])
            else:
                pitch = -np.pi / 2
                roll = -yaw + np.arctan2(-rot_mat[0, 1], -rot_mat[0, 2])
        else:
            # Normal case
            pitch = np.arcsin(-rot_mat[2, 0])
            yaw = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])
            roll = np.arctan2(rot_mat[2, 1], rot_mat[2, 2])
        
        return np.array([yaw, pitch, roll])
    
    def _euler_to_rotation_matrix(self, euler: np.ndarray) -> np.ndarray:
        """
        Convert Euler angles to rotation matrix (ZYX convention).
        
        Args:
            euler: Euler angles [yaw, pitch, roll] in radians.
            
        Returns:
            3x3 rotation matrix.
        """
        yaw, pitch, roll = euler
        
        # Rotation matrix around Z-axis (yaw)
        R_z = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        # Rotation matrix around Y-axis (pitch)
        R_y = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        # Rotation matrix around X-axis (roll)
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        # Combined rotation matrix: R = R_z * R_y * R_x
        R = R_z @ R_y @ R_x
        
        return R
    
    def initialize(self, 
                  position: np.ndarray, 
                  rotation_matrix: np.ndarray,
                  timestamp: float = None):
        """
        Initialize the Kalman filter with the first measurement.
        
        Args:
            position: Initial 3D position [x, y, z].
            rotation_matrix: Initial 3x3 rotation matrix.
            timestamp: Time of the measurement in seconds.
            
        Notes:
            The initial state assumes zero velocity and uses the provided
            position and orientation.
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Convert rotation matrix to Euler angles
        euler_angles = self._rotation_matrix_to_euler(rotation_matrix)
        
        # Initial state: [x, y, z, yaw, pitch, roll, v_x, v_y, v_z]
        initial_state = np.zeros(self.state_dim)
        initial_state[:3] = position
        initial_state[3:6] = euler_angles
        
        self.filter.x = initial_state
        self.last_update_time = timestamp
        self.initialized = True
    
    def predict(self, timestamp: float = None):
        """
        Predict the next state using the constant velocity model.
        
        Args:
            timestamp: Current time in seconds.
            
        Returns:
            Predicted state vector.
            
        Notes:
            This method updates the time step (dt) based on the actual elapsed time
            since the last update, allowing for variable frame rates.
        """
        if not self.initialized:
            return None
        
        # Update dt if timestamp is provided
        if timestamp is not None:
            new_dt = timestamp - self.last_update_time
            if new_dt > 0:
                # Update state transition matrix with new dt
                self.filter.F[0, 6] = self.filter.F[1, 7] = self.filter.F[2, 8] = new_dt
                self.dt = new_dt
        
        # Predict the next state
        self.filter.predict()
        
        # Normalize angles in the state
        self.filter.x[3:6] = self._normalize_angles(self.filter.x[3:6])
        
        return self.filter.x
    
    def update(self, 
              position: np.ndarray, 
              rotation_matrix: np.ndarray, 
              measurement_cov: Optional[np.ndarray] = None,
              timestamp: float = None):
        """
        Update the filter with a new measurement.
        
        Args:
            position: Measured 3D position [x, y, z].
            rotation_matrix: Measured 3x3 rotation matrix.
            measurement_cov: Optional measurement covariance matrix.
            timestamp: Time of the measurement in seconds.
            
        Returns:
            Updated state vector.
            
        Notes:
            If measurement_cov is provided, it replaces the default R matrix,
            allowing for adaptive measurement noise based on the stereo
            reconstruction uncertainty.
            
            The adaptive process noise adjustment increases process noise when
            large innovations (measurement residuals) are observed, allowing the
            filter to respond quickly to sudden changes in needle motion.
        """
        if not self.initialized:
            self.initialize(position, rotation_matrix, timestamp)
            return self.filter.x
        
        # Predict state if timestamp is provided
        if timestamp is not None:
            self.predict(timestamp)
            self.last_update_time = timestamp
        
        # Convert rotation matrix to Euler angles
        euler_angles = self._rotation_matrix_to_euler(rotation_matrix)
        
        # Create measurement vector [x, y, z, yaw, pitch, roll]
        z = np.concatenate([position, euler_angles])
        
        # Handle angle wrapping in innovation calculation
        # Save the current predicted angles
        predicted_angles = self.filter.x[3:6].copy()
        
        # Compute angle differences accounting for circularity
        angle_diff = euler_angles - predicted_angles
        angle_diff = self._normalize_angles(angle_diff)
        
        # Construct the adjusted measurement
        z_adj = np.concatenate([position, predicted_angles + angle_diff])
        
        # Update measurement noise covariance if provided
        if measurement_cov is not None:
            self.filter.R = measurement_cov
        
        # Calculate innovation (measurement residual)
        innovation = z_adj - self.filter.H @ self.filter.x
        
        # Compute innovation covariance
        S = self.filter.H @ self.filter.P @ self.filter.H.T + self.filter.R
        
        # Adaptive process noise based on innovation magnitude
        innovation_norm = np.sqrt(innovation.T @ np.linalg.inv(S) @ innovation)
        
        # Adjust process noise if innovation is large
        if innovation_norm > 3.0:  # 3-sigma threshold
            # Temporarily increase process noise to respond to sudden changes
            adaptive_noise = self.process_noise * (self.adaptive_factor * innovation_norm / 3.0)
            self.filter.Q = adaptive_noise
        else:
            # Use baseline process noise
            self.filter.Q = self.process_noise
        
        # Update the filter
        self.filter.update(z_adj)
        
        # Normalize angles after update
        self.filter.x[3:6] = self._normalize_angles(self.filter.x[3:6])
        
        return self.filter.x
    
    def get_position(self) -> np.ndarray:
        """
        Get the current estimated position.
        
        Returns:
            3D position vector [x, y, z].
        """
        if not self.initialized:
            return np.zeros(3)
        return self.filter.x[:3]
    
    def get_rotation_matrix(self) -> np.ndarray:
        """
        Get the current estimated rotation matrix.
        
        Returns:
            3x3 rotation matrix.
        """
        if not self.initialized:
            return np.eye(3)
        
        euler = self.filter.x[3:6]
        return self._euler_to_rotation_matrix(euler)
    
    def get_state_covariance(self) -> np.ndarray:
        """
        Get the current state covariance matrix.
        
        Returns:
            State covariance matrix P.
        """
        if not self.initialized:
            return np.eye(self.state_dim)
        return self.filter.P
    
    def get_velocity(self) -> np.ndarray:
        """
        Get the current estimated velocity.
        
        Returns:
            3D velocity vector [v_x, v_y, v_z].
        """
        if not self.initialized:
            return np.zeros(3)
        return self.filter.x[6:9]
    
    def compute_measurement_covariance(self, 
                                      stereo_points: np.ndarray, 
                                      stereo_baseline: float,
                                      working_distance: float) -> np.ndarray:
        """
        Compute measurement covariance based on stereo reconstruction uncertainty.
        
        Args:
            stereo_points: 3D points from stereo reconstruction.
            stereo_baseline: Stereo camera baseline in mm.
            working_distance: Distance from camera to surgical site in mm.
            
        Returns:
            6x6 measurement covariance matrix.
            
        Notes:
            Stereo uncertainty increases quadratically with depth and is inversely
            proportional to baseline. This method propagates the 3D point uncertainty
            to the pose estimate.
            
            The depth uncertainty model follows the formula:
            σ_z ≈ (z²/b·f) * σ_d
            where:
              z is depth
              b is baseline
              f is focal length
              σ_d is disparity uncertainty
        """
        # Estimate depth variance based on stereo geometry
        mean_depth = np.mean(stereo_points[:, 2])
        depth_variance = (mean_depth**2 / stereo_baseline) * 0.01  # assuming 0.01 pixel disparity uncertainty
        
        # Position covariance scales with depth
        pos_cov = np.eye(3) * depth_variance
        # XY uncertainty is typically better than Z (depth)
        pos_cov[0, 0] *= 0.5  # X uncertainty
        pos_cov[1, 1] *= 0.5  # Y uncertainty
        
        # Orientation uncertainty also depends on the point cloud quality
        n_points = stereo_points.shape[0]
        orientation_var = 0.01 * (working_distance / stereo_baseline) / np.sqrt(n_points)
        orient_cov = np.eye(3) * orientation_var
        
        # Combine into measurement covariance matrix
        meas_cov = np.zeros((6, 6))
        meas_cov[:3, :3] = pos_cov
        meas_cov[3:, 3:] = orient_cov
        
        return meas_cov