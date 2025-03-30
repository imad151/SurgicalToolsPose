import subprocess
import numpy as np
import cv2
import time
import os
import signal
import argparse
import sys
from typing import Optional, Tuple, Dict
import mmap
import struct
import ctypes
from contextlib import contextmanager

parser = argparse.ArgumentParser(description='RC Visard Python Wrapper')
parser.add_argument('device_id', default= '00000', help='Device ID of the RC Visard')
parser.add_argument('--executable', default='./', help='Path to the RC Visard executable')
parser.add_argument('--left', action='store_true', help='Enable left camera stream')
parser.add_argument('--right', action='store_true', help='Enable right camera stream')
parser.add_argument('--disparity', action='store_true', help='Enable disparity stream')
parser.add_argument('--confidence', action='store_true', help='Enable confidence stream')
parser.add_argument('--error', action='store_true', help='Enable error stream')
parser.add_argument('--frame-rate', type=int, default=25, help='Set frame rate (25 for default)')

class SharedImageData(ctypes.Structure):
    _fields_ = [
        ("rows", ctypes.c_int),
        ("cols", ctypes.c_int),
        ("type", ctypes.c_int),
        ("new_data", ctypes.c_bool),
        ("stream_name", ctypes.c_char * 32)
    ]

class RCVisardPython:
    def __init__(self, 
                 device_id: str,
                 left: bool = False, 
                 right: bool = False, 
                 disparity: bool = False, 
                 confidence: bool = False, 
                 error: bool = False,
                 frame_rate: int = 0,
                 shared_mem_name: str = "rc_visard_shared_mem",
                 mutex_name: str = "rc_visard_mutex"):
        self.device_id = device_id
        self.left = left
        self.right = right
        self.disparity = disparity
        self.confidence = confidence
        self.error = error
        self.frame_rate = frame_rate
        self.shared_mem_name = shared_mem_name
        self.mutex_name = mutex_name
        
        self.process = None
        self.running = False
        self.current_images = {}
        
        # List of enabled streams for reading from shared memory
        self.enabled_streams = []
        if self.left:
            self.enabled_streams.append("left")
        if self.right:
            self.enabled_streams.append("right")
        if self.disparity:
            self.enabled_streams.append("disparity")
        if self.confidence:
            self.enabled_streams.append("confidence")
        if self.error:
            self.enabled_streams.append("error")
        
    
    def get_image(self, timeout: float = 1.0) -> Optional[Dict[str, np.ndarray]]:
        if not self.running:
            print("RC Visard process is not running")
            return None
        
        result = {}
        
        # read from each enabled stream's shared memory
        for stream_name in self.enabled_streams:
            try:
                stream_shm_name = f"{self.shared_mem_name}_{stream_name}"
                
                if not os.path.exists(f"/dev/shm/{stream_shm_name}"):
                    continue
                
                with open(f"/dev/shm/{stream_shm_name}", "r+b") as f:
                    mapped_file = mmap.mmap(f.fileno(), 0)
                    
                    shared_data = SharedImageData.from_buffer(mapped_file, 0)
                    
                    if not shared_data.new_data:
                        if stream_name in self.current_images:
                            result[stream_name] = self.current_images[stream_name]
                        continue
                    
                    rows = shared_data.rows
                    cols = shared_data.cols
                    img_type = shared_data.type
                    
                    np_dtype = self._cv_type_to_numpy_dtype(img_type)
                    
                    channels = ((img_type >> 3) & 63) + 1
                    if (img_type & 0x18) >> 3:  # CV_8UC3
                        channels = 3
                        
                    data_size = rows * cols * np.dtype(np_dtype).itemsize * channels
                    
                    data_offset = ctypes.sizeof(SharedImageData)
                    image_data = mapped_file[data_offset:data_offset + data_size]
                    
                    if channels > 1:
                        img = np.frombuffer(image_data, dtype=np_dtype).reshape(rows, cols, channels).copy()
                    else:
                        img = np.frombuffer(image_data, dtype=np_dtype).reshape(rows, cols).copy()
                    
                    self.current_images[stream_name] = img
                    result[stream_name] = img
                    
                    mapped_file.close()
                    
            except FileNotFoundError:  # Expected for disabled streams and my laziness
                pass
            except Exception as e:
                pass # too lazy
        
        return result if result else self.current_images
    
    def stop(self) -> None:
        if self.process and self.running:
            self.process.send_signal(signal.SIGINT)
            self.process.wait(timeout=5)
            self.running = False
            self.process = None
            
            # Clean up 
            for stream_name in self.enabled_streams:
                stream_shm_name = f"{self.shared_mem_name}_{stream_name}"
                try:
                    os.unlink(f"/dev/shm/{stream_shm_name}")
                except:
                    pass

        
    def start(self, executable_path: str = "./rc_visard_show_streams") -> bool:
        if self.running:
            print("RC Visard process is already running")
            return False
            
        cmd = [executable_path]
        
        if self.left:
            cmd.append("--left")
        if self.right:
            cmd.append("--right")
        if self.disparity:
            cmd.append("--disparity")
        if self.confidence:
            cmd.append("--confidence")
        if self.error:
            cmd.append("--error")
            
        cmd.append("--ipc")
        
        if self.frame_rate > 0:
            cmd.append(f"--frame-rate={self.frame_rate}")
            
        cmd.append(f"--shared-mem-name={self.shared_mem_name}")
        cmd.append(f"--mutex-name={self.mutex_name}")
        
        cmd.append(self.device_id)
        
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            time.sleep(2)
            
            if self.process.poll() is not None:
                _, stderr = self.process.communicate()
                print(f"Failed to start RC Visard process: {stderr}")
                return False
                
            self.running = True
            return True
            
        except Exception as e:
            print(f"Error starting RC Visard process: {e}")
            return False
    

    def _cv_type_to_numpy_dtype(self, cv_type):
        depth = cv_type & 7
        
        if depth == 0:    # CV_8U
            return np.uint8
        elif depth == 1:  # CV_8S
            return np.int8
        elif depth == 2:  # CV_16U
            return np.uint16
        elif depth == 3:  # CV_16S
            return np.int16
        elif depth == 4:  # CV_32S
            return np.int32
        elif depth == 5:  # CV_32F
            return np.float32
        elif depth == 6:  # CV_64F
            return np.float64
        else:
            return np.uint8  # Default
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

if __name__ == "__main__":
    args = parser.parse_args()
    
    # Enable left camera by default if no streams are specified
    if not (args.left or args.right or args.disparity or args.confidence or args.error):
        args.left = True
    
    with RCVisardPython(
        device_id=args.device_id,
        left=args.left,
        right=args.right,
        disparity=args.disparity,
        confidence=args.confidence,
        error=args.error,
        frame_rate=args.frame_rate
    ) as visard:
        if not visard.start(args.executable):
            sys.exit(1)
        
        print("RC Visard running. Press Ctrl+C to exit.")
        
        try:
            if args.left:
                cv2.namedWindow("left", cv2.WINDOW_NORMAL)
            if args.right:
                cv2.namedWindow("right", cv2.WINDOW_NORMAL)
            if args.disparity:
                cv2.namedWindow("disparity", cv2.WINDOW_NORMAL)
            if args.confidence:
                cv2.namedWindow("confidence", cv2.WINDOW_NORMAL)
            if args.error:
                cv2.namedWindow("error", cv2.WINDOW_NORMAL)
            
            while True:
                # Get all available images
                images = visard.get_image()
                
                if images:
                    for stream_name, img in images.items():
                        cv2.imshow(stream_name, img)
                
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            pass
        finally:
            cv2.destroyAllWindows()