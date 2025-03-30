# **Pipeline Breakdown**
This pipeline takes in stereo images of a **dVRK (da Vinci Research Kit) robot** and estimates its **pose** in a robust, real-time, and markerless way. Instead of relying on traditional depth estimation and filtering techniques, it uses a combination of **deep learning, neural radiance fields (NeRF), graph neural networks (GNN), and physics-informed neural networks (PINN)** to refine the pose at each stage.

We are essentially doing **5 major things** in sequence:

1. **Stereo matching to get rid of explicit depth estimation.**
2. **Transformer-based 2D keypoint detection.**
3. **NeRF-based refinement for occlusion handling and multi-view consistency.**
4. **GNN-based temporal smoothing to fix jitter.**
5. **PINN-based physics correction to enforce real-world constraints.**

This ensures we are not just relying on one source but progressively improving the keypoints at each stage.

---

## **Step 1: Stereo Matching for Depth Estimation**
Instead of estimating a **full depth map**, we directly match corresponding keypoints between the **left and right stereo images** using **LoFTR (Local Feature TRansformer)**. LoFTR is a transformer-based model that finds **pixel correspondences** between images without relying on traditional feature detection and matching.

### **Why?**
- Depth estimation via triangulation is noisy and slow.
- This method directly gives us **3D keypoints** without needing to compute an explicit depth map.

### **How it Works**
1. LoFTR processes both stereo images and finds **correspondences** between them.
2. These matched keypoints are then converted to **3D points** using camera intrinsics.
3. The output is an initial set of **3D keypoints in camera coordinates**.

This is the **raw 3D keypoint data** that will be refined in later steps.

---

## **Step 2: Transformer-Based 2D Keypoint Detection**
Now that we have some **3D keypoints**, we need to **predict the correct set of keypoints that define the dVRK's pose**. We use a **Vision Transformer (ViT)-based network** to detect these keypoints directly from the images.

### **Why?**
- Traditional methods (like OpenPose) don’t generalize well to robots.
- Transformers capture **global context**, making them better at detecting keypoints in **occluded or low-texture regions**.

### **How it Works**
1. A ViT-based model is trained to **detect 21 keypoints** on the dVRK from **synthetic and real images**.
2. The detected **2D keypoints** are mapped to their **corresponding 3D keypoints** from the stereo matching step.
3. The output is an improved set of **3D keypoints**, but they may still have noise, especially in occluded areas.

At this stage, we have **better 3D keypoints**, but they still need refinement.

---

## **Step 3: NeRF-Based Refinement**
Keypoints estimated so far might be **inaccurate in occluded regions**. To fix this, we use a **Neural Radiance Field (NeRF)** to learn a **multi-view 3D representation** of the dVRK and refine the keypoints.

### **Why?**
- NeRF can **reconstruct missing details** from past views.
- It improves keypoints by enforcing multi-view consistency.

### **How it Works**
1. We train a NeRF model to learn the **3D structure of the robot** using past frames.
2. For each predicted 3D keypoint, we compare it with the **NeRF-predicted position** of the same keypoint.
3. The **NeRF-corrected keypoints** are used to refine the previous step’s outputs.

After this, the keypoints are **more robust to occlusions and view inconsistencies**.

---

## **Step 4: Graph Neural Network for Temporal Smoothing**
Even after NeRF refinement, the keypoints might be **jittery** across frames. To fix this, we use a **Graph Neural Network (GNN)** to smooth them over time.

### **Why?**
- Ensures keypoints are **temporally smooth**.
- Preserves **kinematic relationships** between different robot parts.

### **How it Works**
1. We construct a **graph** where:
   - **Nodes** = Keypoints at each frame.
   - **Edges** = Kinematic constraints between keypoints.
2. A **Graph Attention Network (GATv2)** learns to propagate corrections across frames.
3. The output is a **smoothed sequence of 3D keypoints**.

This step makes sure that keypoints **do not jump around inconsistently**.

---

## **Step 5: Physics-Based Correction with PINNs**
Even if keypoints are smooth, they might still violate the **physical constraints** of the dVRK. We use a **Physics-Informed Neural Network (PINN)** to enforce constraints based on **robot kinematics**.

### **Why?**
- Keypoints should obey **joint limits, velocity, and torque constraints**.
- Ensures that the estimated pose is **physically possible**.

### **How it Works**
1. The PINN is trained on **real dVRK movement data** to learn its kinematics.
2. It takes in the predicted keypoints and **adjusts them to ensure they obey dVRK’s physics**.
3. The final output is a **corrected, physics-valid set of 3D keypoints**.

At this stage, we have a **fully refined, smooth, and physically valid pose estimate**.

---

# **Final Summary: How Everything Connects**
1. **Stereo Matching (LoFTR)** → Converts stereo images **directly** to 3D keypoints.
2. **Transformer-Based Keypoint Detection** → Extracts robot keypoints and matches them with stereo output.
3. **NeRF-Based Refinement** → Corrects keypoints using multi-view consistency.
4. **Graph Neural Network (GNN)** → Smooths keypoints across time.
5. **Physics-Based Correction (PINN)** → Ensures final keypoints obey dVRK’s constraints.

This pipeline ensures that:
- **No explicit depth estimation is needed**.
- **Keypoints are accurate even in occlusions**.
- **Temporal jitter is reduced**.
- **Final pose estimates are physically valid**.

This method is more **robust, efficient, and adaptable** compared to traditional techniques.

