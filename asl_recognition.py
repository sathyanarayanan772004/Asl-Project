from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import hand_detection as hd
import os
import re

class ASLRecognizer:
    def __init__(self):
        # Load model
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'CNN.h5')
        self.model = load_model(model_path)
        
        self.class_names = ["A", "B", "C", "D", "E", "F",
                           "G", "H", "I", "J", "K", "L",
                           "M", "N", "O", "P", "Q", "R",
                           "S", "T", "U", "V", "W", "X",
                           "Y", "Z", "del", "nothing", "space"]
        
        self.detector = hd.handDetector()
        self.samples_to_predict = []
    
    def process_frame(self, img):
        """Process a single frame and return the processed image and detected letter"""
        if img is None or img.size == 0:
            return img, None
        
        image = self.detector.findHands(img)
        landmark_list = self.detector.findPosition(img)
        
        # Get corner points of hand detection rectangle (bigger canvas)
        (startX, startY) = 50, 50
        (endX, endY) = 400, 400
        cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)
        
        detected_letter = None
        
        if len(landmark_list) != 0:
            # Check if hand is within the rectangle
            if (startX <= landmark_list[0][1] <= endX) and (startY <= landmark_list[0][2] <= endY):
                cropped_video = img[startY:endY, startX:endX]
                image_resized = cv2.resize(cropped_video, (64, 64))
                image_normalized = image_resized.astype('float32') / 255.0
                x = img_to_array(image_normalized)
                x = np.expand_dims(image_normalized, axis=0)
                
                prediction = self.model.predict(x, verbose=0)
                prediction = np.argmax(prediction)
                label = self.class_names[prediction]
                
                # Add prediction to samples
                for i in label:
                    self.samples_to_predict.append(i)
                
                # Get stable prediction (reduced threshold for faster detection)
                string_labels = self._list_to_string(self.samples_to_predict)
                if len(string_labels) >= 7:  # Reduced from 10 to 5 for faster detection
                    consecutive = [match[1] for match in re.findall(r'((\w)\2{4,})', string_labels)]  # Reduced from 9 to 4
                    if consecutive:
                        detected_letter = consecutive[0]
                        # Display stable prediction
                        cv2.putText(img, f"Detected: {detected_letter}", (90, 40), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Clear samples after stable detection
                        self.samples_to_predict = []
                
                # Display current prediction
                cv2.putText(img, f"Current: {label}", (90, 90), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        return img, detected_letter
    
    def _list_to_string(self, s):
        """Convert list to string"""
        str1 = ""
        for ele in s:
            str1 += ele
        return str1
