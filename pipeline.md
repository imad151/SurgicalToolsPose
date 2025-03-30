# **Pipeline Breakdown**
We are dealing with **two sources of information**:  
1. **Joint Encoder Data** (forward kinematics) – gives an estimated pose.  
2. **Stereo Camera Data** – provides visual observations.  

We are fusing both to achieve **robust, real-time, and markerless pose estimation**.  

Here’s how I see the **multi-model approach** in the pipeline:

---

### **Step 1: Joint Encoder-Based Initial Pose Estimate**  
- The **joint encoders** give us joint angles (**q**) and velocities (**dq**).  
- Using **forward kinematics**, we compute the **expected keypoint positions** (e.g., tool tip, wrist, joints) in **robot base coordinates**.  
- This is our **first estimate**, but we know it has errors (due to backlash, sensor drift, and calibration issues).  

**Why do we need this?**  
- It provides a **baseline** before using vision, ensuring we don’t have to detect keypoints from scratch.  
- Even if vision fails (e.g., occlusion, poor lighting), we still have a fallback estimate.

---

### **Step 2: CNN-Based 2D Keypoint Detection**
- A **CNN (e.g., HRNet, SuperPoint, or a lightweight version of OpenPose)** is trained to detect keypoints in **stereo images**.
- The CNN outputs **2D keypoint heatmaps** for each image.  
- Key challenges here:
  - **Occlusion handling** – some parts of the robot may not always be visible.  
  - **Reflective surfaces** – surgical tools are metallic, making it hard to get sharp edges.  
  - **Low-texture regions** – traditional feature matching struggles on smooth surfaces.  

**Why is CNN needed?**  
- It helps us **detect keypoints accurately** even when hand-engineered feature detectors fail.  
- It learns the **structure of the robot** so it can infer missing keypoints in case of occlusions.  

---

### **Step 3: Stereo Matching for 3D Keypoints**
- Once we have **2D keypoints from the left and right images**, we match them using **epipolar constraints**.  
- This allows us to **triangulate** and recover the **3D position** of keypoints in **camera coordinates**.  
- This gives us a **vision-only pose estimate** (i.e., pose from camera observations).  

**Why is this important?**  
- **Encoders tell us where the robot *should* be.**  
- **Stereo vision tells us where the robot *actually* is.**  
- The difference between these two helps us correct systematic errors.

---

### **Step 4: Sensor Fusion – Combining Vision & Encoders**
- Now, we have two estimates:  
  1. **Forward kinematics estimate (from encoders)**  
  2. **Vision-based estimate (from CNN + Stereo Matching)**  
- We fuse them using **one of these approaches**:
  - **Kalman Filter** – if errors are Gaussian and small.  
  - **Factor Graph Optimization** – better for long-term consistency.  
  - **Neural Network-Based Correction** – learns systematic bias and compensates for it.  

This step allows us to correct **persistent kinematic errors**, such as:  
**Backlash effects** (mechanical slack in joints).  
**Calibration errors** (misalignment between encoders and actual movement).  
**Drift compensation** (joint encoders accumulate small errors over time).  

---

### **Step 5: Pose Estimation Using PnP (Perspective-n-Point)**
- The **refined 3D keypoints** are now used for final pose estimation.  
- Given the **known robot kinematic model**, we solve for the pose using **PnP + RANSAC**.  
- This helps us get a robust estimate of the **camera-to-robot transformation**.  

---

### **Step 6: Learning-Based Refinement (Optional)**
- Over time, we can train a **small MLP or Graph Neural Network** to predict **systematic deviations** and correct them.  
- Instead of computing corrections manually, the model can **learn drift patterns** and improve the pose estimate.  

---

## **Final Thoughts**
This **multi-model approach** ensures:  
**We are not fully dependent on vision** – encoders give a fallback.  
**We correct systematic drift** – vision detects real-world deviations.  
**It works in real-time** – stereo matching + Kalman filtering keeps computation low.  
**It is markerless** – no need for fiducials or predefined markers on the robot.



## **Some Noteable Models / Methods / Papers:**
- Deep Bayesian-Assisted Keypoint Detection
- Deep Stereo-Based 3D Keypoint Estimation for Object Pose Refinement
- SuperPoint: Self-Supervised Interest Point Detection and Description
- LOFTR: Detector-Free Local Feature Matching with Transformers 
