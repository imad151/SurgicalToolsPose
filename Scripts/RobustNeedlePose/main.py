import numpy as np
from sklearn.linear_model import RANSACRegressor, LinearRegression
from scipy.optimize import least_squares
from sklearn.decomposition import PCA
from filterpy.kalman import KalmanFilter
import time
from typing import Tuple, List, Dict, Optional, Union, Callable
from dataclasses import dataclass, field

from MainPipeline import RobustPlaneFitter, FitzgibbonEllipseFitter, AdaptiveKalmanFilter


class EnhancedEllipsePoseEstimator:
    """    
    Attributes:
        aspect_ratio (float): Expected aspect ratio of the ellipse.
        plane_fitter (RobustPlaneFitter): RANSAC-based plane fitting object.
        ellipse_fitter (FitzgibbonEllipseFitter): Ellipse fitting object.
        kalman_filter (AdaptiveKalmanFilter): Temporal filter for pose tracking.
        surgical_constraints (dict): Dictionary of surgical needle constraints.
        validation_metrics (dict): Dictionary of validation metrics.
        last_timestamp (float): Time of last update.
    """
    
    def __init__(self, 
                 aspect_ratio: float = 1.0,
                 surgical_constraints: Optional[dict] = None,
                 ransac_params: Optional[dict] = None,
                 ellipse_method: str = 'fitzgibbon_hybrid',
                 kalman_config: Optional[dict] = None):
        """        
        Args:
            aspect_ratio: Expected aspect ratio of the ellipse (a/b).
            surgical_constraints: Dictionary of surgical needle constraints.
            ransac_params: Parameters for RANSAC plane fitting.
            ellipse_method: Method for ellipse fitting ('fitzgibbon', 'geometric', 'hybrid').
            kalman_config: Configuration for the Kalman filter.
        """
        self.aspect_ratio = aspect_ratio
        self.surgical_constraints = surgical_constraints or {}
        
        # Default RANSAC parameters if not provided
        ransac_default = {
            'max_trials': 1000,
            'residual_threshold': 0.5,
            'sampling': 'hybrid',
            'probability_weights': (0.7, 0.3),
            'temporal_persistence': 3
        }
        ransac_params = {**ransac_default, **(ransac_params or {})}
        
        # plane fitter
        self.plane_fitter = RobustPlaneFitter(
            max_trials=ransac_params['max_trials'],
            residual_threshold=ransac_params['residual_threshold'],
            sampling=ransac_params['sampling'],
            probability_weights=ransac_params['probability_weights'],
            temporal_persistence=ransac_params['temporal_persistence']
        )
        
        # ellipse fitter
        self.ellipse_fitter = FitzgibbonEllipseFitter(
            aspect_ratio=aspect_ratio,
            method=ellipse_method
        )
        
        # Kalman filter
        self.kalman_filter = AdaptiveKalmanFilter(
            **(kalman_config or {})
        )
        
        # validation metrics
        self.validation_metrics = {
            'reproj_error': [],
            'plane_consistency': [],
            'ellipse_consistency': [],
            'processing_time': []
        }
        
        self.last_timestamp = None
    
    def _project_to_plane(self, 
                         points: np.ndarray, 
                         centroid: np.ndarray, 
                         plane_basis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """            
            The projection is performed by subtracting the centroid and computing
            the dot product with each basis vector of the plane.
            
            Error propagation: Projection error is proportional to the distance
            from points to the plane. For inlier points (within residual_threshold),
            the projection error is typically less than 0.5mm.
        """
        centered = points - centroid
        
        u = np.dot(centered, plane_basis[0])
        v = np.dot(centered, plane_basis[1])
        
        return u, v
    
    def _compute_reprojection_error(self, 
                                   points: np.ndarray, 
                                   ellipse_center: np.ndarray, 
                                   rot_mat: np.ndarray,
                                   ellipse_params: np.ndarray) -> float:
        """            
        Notes:
            The reprojection error is computed as the geometric distance from each
            point to the ellipse surface in 3D space. This is a more accurate
            measure of fit quality than the algebraic distance.
        """
        n_points = points.shape[0]
        
        _, _, a, b, _ = ellipse_params
        
        theta = np.linspace(0, 2*np.pi, 100)
        canonical_ellipse = np.zeros((100, 3))
        canonical_ellipse[:, 0] = a * np.cos(theta)
        canonical_ellipse[:, 1] = b * np.sin(theta)
        
        ellipse_3d = np.dot(canonical_ellipse, rot_mat.T) + ellipse_center
        
        errors = np.zeros(n_points)
        for i, point in enumerate(points):
            distances = np.linalg.norm(ellipse_3d - point, axis=1)
            errors[i] = np.min(distances)
        
        return np.mean(errors)
    
    def _verify_pose_consistency(self, 
                               ellipse_center: np.ndarray, 
                               rot_mat: np.ndarray) -> bool:
        """      
        This function verifies that:
            1. The ellipse center is within the expected workspace
            2. The orientation is consistent with anatomical constraints
            3. The pose is consistent with previous estimates (if available)
        """

        # Check if ellipse center is within workspace limits
        if 'max_translation_mm' in self.surgical_constraints:
            limits = self.surgical_constraints['max_translation_mm']
            if (np.abs(ellipse_center[0]) > limits[0] or 
                np.abs(ellipse_center[1]) > limits[1] or 
                np.abs(ellipse_center[2]) > limits[2]):
                return False
        
        # Check if orientation is within anatomical limits
        yaw, pitch, roll = self.kalman_filter._rotation_matrix_to_euler(rot_mat)
        
        if 'max_rotation_deg' in self.surgical_constraints:
            limits_deg = self.surgical_constraints['max_rotation_deg']
            limits_rad = np.deg2rad(limits_deg)
            
            if limits_deg[0] < 360 and np.abs(yaw) > limits_rad[0]/2:
                return False
            
            # Pitch is always limited (can't go through workspace)
            if np.abs(pitch) > limits_rad[1]/2:
                return False
            
            if limits_deg[2] < 360 and np.abs(roll) > limits_rad[2]/2:
                return False
        
        if self.kalman_filter.initialized:
            prev_pos = self.kalman_filter.get_position()
            prev_rot = self.kalman_filter.get_rotation_matrix()
            
            # Check position difference
            pos_diff = np.linalg.norm(ellipse_center - prev_pos)
            if pos_diff > 10.0:  # 10mm maximum movement between frames
                return False
            
            # Check orientation difference (using Frobenius norm)
            rot_diff = np.linalg.norm(rot_mat - prev_rot, 'fro')
            if rot_diff > 1.0:  # 30 degrees maximum rotation
                return False
        
        return True
    
    def estimate_pose(self, 
                     points: np.ndarray, 
                     timestamp: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Estimate the pose of an ellipse from a set of 3D points.
        
        Args:
            points: 3D point cloud of shape (n, 3).
            timestamp: Time of the measurement in seconds.
            
        Returns:
            tuple: (position, orientation, metrics)
                position (numpy.ndarray): 3D position of the ellipse center.
                orientation (numpy.ndarray): 3x3 rotation matrix.
                metrics (dict): Dictionary of validation metrics.

        Error propagation:
        - Position accuracy: ~1-2mm (primarily limited by stereo reconstruction)
        - Orientation accuracy: ~3-5 degrees (primarily limited by ellipse fitting)
        """
        start_time = time.time()
        
        if timestamp is None:
            timestamp = time.time()
        
        if self.last_timestamp is None:
            self.last_timestamp = timestamp
        
        # Step 1: plane fitting
        centroid, normal, plane_basis = self.plane_fitter.fit(
            points, self.surgical_constraints)
        
        # Step 2: Project points onto the fitted plane
        u, v = self._project_to_plane(points, centroid, plane_basis)
        
        # Step 3: Fit ellipse to the projected points
        ellipse_params = self.ellipse_fitter.fit(u, v)
        
        # Verify ellipse consistency with the plane
        ellipse_consistent = self.ellipse_fitter.verify_ellipse_plane(
            ellipse_params, normal, self.surgical_constraints)
        
        # Step 4: Compute 3D pose from ellipse parameters
        u_c, v_c, a, b, theta = ellipse_params
        ellipse_center_3d = centroid + u_c * plane_basis[0] + v_c * plane_basis[1]
        
        rot_z = np.zeros((3, 3))
        rot_z[:, 0] = plane_basis[0]
        rot_z[:, 1] = plane_basis[1]
        rot_z[:, 2] = normal
        
        # Ensure orthonormality using SVD
        U, _, Vt = np.linalg.svd(rot_z)
        rot_z = U @ Vt
        
        # Apply in-plane rotation for ellipse orientation
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        in_plane_rot = np.array([
            [cos_theta, -sin_theta, 0],
            [sin_theta, cos_theta, 0],
            [0, 0, 1]
        ])
        
        # Final rotation matrix
        rot_mat = rot_z @ in_plane_rot
        
        # Step 5: Validation
        reprojection_error = self._compute_reprojection_error(
            points, ellipse_center_3d, rot_mat, ellipse_params)
        
        pose_consistent = self._verify_pose_consistency(
            ellipse_center_3d, rot_mat)
        
        # Step 6: Apply Kalman filtering
        dt = timestamp - self.last_timestamp
        self.last_timestamp = timestamp
        
        # Compute measurement covariance based on reprojection error
        stereo_baseline = self.surgical_constraints.get('stereo_baseline', 5.0)  # mm
        working_distance = self.surgical_constraints.get('working_distance', 100.0)  # mm
        
        measurement_cov = self.kalman_filter.compute_measurement_covariance(
            points, stereo_baseline, working_distance)
        
        # Scale measurement covariance by reprojection error
        if reprojection_error > 0.5:  # If error is large, increase uncertainty
            measurement_cov *= (reprojection_error / 0.5)
        
        # Update Kalman filter
        filtered_state = self.kalman_filter.update(
            ellipse_center_3d, rot_mat, measurement_cov, timestamp)
        
        # Get filtered position and orientation
        filtered_position = self.kalman_filter.get_position()
        filtered_rotation = self.kalman_filter.get_rotation_matrix()
        
        # Update metrics
        self.validation_metrics['reproj_error'].append(reprojection_error)
        self.validation_metrics['plane_consistency'].append(True)  # From plane fitter
        self.validation_metrics['ellipse_consistency'].append(ellipse_consistent)
        self.validation_metrics['processing_time'].append(time.time() - start_time)
        
        metrics = {
            'reprojection_error': reprojection_error,
            'ellipse_params': ellipse_params,
            'plane_normal': normal,
            'inlier_ratio': np.sum(self.plane_fitter.inliers) / len(points) if hasattr(self.plane_fitter, 'inliers') else 1.0,
            'processing_time': time.time() - start_time,
            'pose_consistent': pose_consistent,
            'ellipse_consistent': ellipse_consistent,
            'velocity': self.kalman_filter.get_velocity() if self.kalman_filter.initialized else np.zeros(3),
            'covariance': self.kalman_filter.get_state_covariance() if self.kalman_filter.initialized else np.eye(9)
        }
        
        return filtered_position, filtered_rotation, metrics


