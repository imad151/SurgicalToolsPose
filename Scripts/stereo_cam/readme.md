# RC Visard Python Integration

This project provides a simple way to run an RC Visard using Python. It takes inspiration from the [rc_visard_opencv_example](https://github.com/roboception/rc_visard_opencv_example) repository.

## Setup Instructions

1. **Update the Original Repository**
   - Replace the code in `rc_visard_opencv_example/tools/rc_visard_show_streams.cc` with your own code.
   
2. **Build the Original Repository**
   - Follow the build instructions of `rc_visard_opencv_example`.
   - After building, ensure that `build/tools/` contains the `rc_visard_show_streams` executable.

3. **Move Executable**
   - Copy `rc_visard_show_streams` to the folder containing `wrapper.py`.

4. **Run the Scripts**
   - **Running `test.py`**:
     ```sh
     python test.py
     ```
     In the code, change the device id to yours. No additional CLI arguments are required.
   
   - **Running `wrapper.py`**:
     ```sh
     python wrapper.py <device_id> [options]
     ```
     - `<device_id>`: Device ID of the RC Visard.
     - Options:
       - `--executable <path>`: Path to the RC Visard executable (default: `./`).
       - `--left`: Enable left camera stream.
       - `--right`: Enable right camera stream.
       - `--disparity`: Enable disparity stream.
       - `--confidence`: Enable confidence stream.
       - `--error`: Enable error stream.
       - `--frame-rate <int>`: Set frame rate (default: `25`).

5. **Configuration**
   - Replace `<device_id>` with your actual device ID.
   - Apply the same change to `test.py` if needed.

## System Requirements
- **Tested on**: Ubuntu 22.04
- **Python Version**: 3.10

---
This is a simple side project. No claim is made on the original code.

