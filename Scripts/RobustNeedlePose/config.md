| **Class**                  | **Parameter**                | **Default Value**                                          | **Description**                                                                                   |
|---------------------------|------------------------------|-------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| **RobustPlaneFitter**     | `max_trials`                 | `1000`                                                      | Max number of RANSAC iterations. Higher = better robustness, slower speed.                        |
|                           | `residual_threshold`         | `0.5` mm                                                    | Distance threshold to count a point as an inlier. Should match stereo noise.                      |
|                           | `sampling`                   | `'hybrid'`                                                  | Sampling strategy: `'distance'`, `'curvature'`, or `'hybrid'`.                                   |
|                           | `probability_weights`        | `(0.7, 0.3)`                                                | Weights for distance vs curvature when using hybrid sampling.                                    |
|                           | `temporal_persistence`       | `3`                                                         | How many prior frames to consider when validating plane consistency.                              |
|                           | *(internal)*                 | `angle_deg < 30`                                            | Plane normal deviation allowed between frames.                                                    |
|                           | *(internal)*                 | `centroid_distance < 10mm`                                  | Max plane centroid shift allowed across frames.                                                   |

---

| **Class**                     | **Parameter**             | **Default Value**                                          | **Description**                                                                                   |
|------------------------------|---------------------------|-------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| **FitzgibbonEllipseFitter**  | `aspect_ratio`            | `1.0`                                                       | Expected a/b ratio of ellipse. Enforced softly in hybrid fit.                                    |
|                              | `method`                  | `'fitzgibbon_hybrid'`                                       | `'fitzgibbon'` (algebraic), `'hybrid'` (algebraic + geometric refinement).                        |
|                              | *(internal)*              | `reproj loss: Huber`                                        | Loss function used for geometric residuals. Huber gives robustness.                               |
|                              | *(internal)*              | `f_scale = 0.1`                                             | Affects Huber sensitivity. Lower = more outlier resistance.                                       |
|                              | *(verify)*                | `±30%` ratio tolerance                                      | Allowed deviation from `aspect_ratio` for validation.                                             |
|                              | *(verify)*                | `±50%` size tolerance                                       | Allowed deviation from expected major axis (based on needle length).                              |

---

| **Class**                   | **Parameter**              | **Default Value**                                          | **Description**                                                                                   |
|----------------------------|----------------------------|-------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| **AdaptiveKalmanFilter**   | `dt`                       | `0.033` sec (~30Hz)                                         | Time step between frames. Auto-adjusted if timestamp provided.                                   |
|                            | `process_noise`            | `[0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.5, 0.5, 0.5]`           | Std devs for [x, y, z, yaw, pitch, roll, vx, vy, vz].                                             |
|                            | `measurement_noise`        | `[0.5, 0.5, 1.0, 0.05, 0.05, 0.05]`                         | Std devs for measured [x, y, z, yaw, pitch, roll].                                                |
|                            | `adaptive_factor`          | `1.5`                                                       | Scales process noise dynamically when innovation is large.                                        |
|                            | *(internal)*               | `initial P = I * 10`                                        | Initial state uncertainty matrix.                                                                 |
|                            | *(compute_measurement_covariance)* | stereo-based                                   | Scales covariance based on stereo baseline, depth, and point count.                              |
