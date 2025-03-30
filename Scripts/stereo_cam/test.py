import numpy as np
import cv2
from wrapper import RCVisardPython

printed = False
with RCVisardPython(
    device_id="000000",
    left=True,
    right=True,
    disparity=True,
    confidence=True,
    error=True,
    frame_rate=25
) as visard:
    if visard.start("/home/imad/SurgicalToolsPose/Scripts/stereo_cam/rc_visard_show_streams"):
        try:
            if visard.left:
                cv2.namedWindow("left", cv2.WINDOW_NORMAL)
            if visard.right:
                cv2.namedWindow("right", cv2.WINDOW_NORMAL)
            if visard.disparity:
                cv2.namedWindow("disparity", cv2.WINDOW_NORMAL)
            if visard.confidence:
                cv2.namedWindow("confidence", cv2.WINDOW_NORMAL)
            if visard.error:
                cv2.namedWindow("error", cv2.WINDOW_NORMAL)
            
            while True:
                images = visard.get_image()
                
                if images:
                    if not printed:
                        print(f"Data being received: {images.keys()}")
                        printed = True
                    if "left" in images and visard.left:
                        left_img = images["left"]
                        cv2.imshow("left", left_img)
                        
                    if "right" in images and visard.right:
                        right_img = images["right"]
                        cv2.imshow("right", right_img)
                    
                    if "disparity" in images and visard.disparity:
                        disparity_img = images["disparity"]
                        cv2.imshow("disparity", disparity_img)
                    
                    if "confidence" in images and visard.confidence:
                        confidence_img = images["confidence"]
                        cv2.imshow("confidence", confidence_img)
                    
                    if "error" in images and visard.error:
                        error_img = images["error"]
                        cv2.imshow("error", error_img)
                
                if cv2.waitKey(1) == ord('q'):
                    break
                    
        finally:
            cv2.destroyAllWindows()
