# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Main script to run the object detection routine."""
import argparse
import sys
import time
import cv2
import os
from datetime import datetime
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils
import math

################################################################################################################################
#Kode lagt til
# Funksjon for å beregne avstanden i piksler mellom to punkter
def pixel_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
# Funksjon for å konvertere pikseldistansen til geografiske koordinater
def pixel_to_coords(camera_coords, pixel_position, scale, image_center, direction_angle):
    distance_pixels = pixel_distance(pixel_position, image_center)
    distance_real = distance_pixels * scale
    delta_lat = distance_real / 111320 * math.cos(math.radians(direction_angle))
    delta_lon = distance_real / (111320 * math.cos(math.radians(camera_coords[0])))
    new_lat = camera_coords[0] + delta_lat
    new_lon = camera_coords[1] + delta_lon
    return new_lat, new_lon
#################################################################################################################################

def run(model: str, camera_id: int, width: int, height: int, num_threads: int, enable_edgetpu: bool) -> None:
    """Continuously run inference on images acquired from the camera.

    Args:
      model: Name of the TFLite object detection model.
      camera_id: The camera id to be passed to OpenCV.
      width: The width of the frame captured from the camera.
      height: The height of the frame captured from the camera.
      num_threads: The number of CPU threads to run the model.
      enable_edgetpu: True/False whether the model is a EdgeTPU model.
    """
  
    # Variables to calculate FPS
    counter, fps = 0, 0
    start_time = time.time()

    # Start capturing video input from the camera
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Visualization parameters
    row_size = 20  # pixels
    left_margin = 24  # pixels
    text_color = (0, 0, 255)  # red
    font_size = 1
    font_thickness = 1
    fps_avg_frame_count = 10

    # Initialize the object detection model
    base_options = core.BaseOptions(
        file_name=model, use_coral=enable_edgetpu, num_threads=num_threads)
    detection_options = processor.DetectionOptions(
        max_results=3, score_threshold=0.3)
    options = vision.ObjectDetectorOptions(
        base_options=base_options, detection_options=detection_options)
    detector = vision.ObjectDetector.create_from_options(options)

############################################################################################################
#Kode lagt til

    last_saved_timestamp = None  # variabel for å ta tiden av når man skal lagre bilder
    last_directory_timestamp = None # variabel for å ta tiden av når man skal oprette mappe

############################################################################################################
   
    # Continuously capture images from the camera and run inference
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            sys.exit(
                'ERROR: Unable to read from webcam. Please verify your webcam settings.'
            )

        counter += 1
        # Convert the image from BGR to RGB as required by the TFLite model.
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create a TensorImage object from the RGB image.
        input_tensor = vision.TensorImage.create_from_array(rgb_image)

        # Run object detection estimation using the model.
        detection_result = detector.detect(input_tensor)

        frame_copy = image.copy() #Tar en kopi av bildet utenramme

        # Draw keypoints and edges on input image
        image = utils.visualize(image, detection_result)

############################################################################################################
#Kode lagt til 
        # Resize the image to desired width and height that you want on your screen
        display_width = 1280  # For example, change this value to your desired width
        display_height = 720  # For example, change this value to your desired height
        image = cv2.resize(image, (display_width, display_height))

############################################################################################################

        # Calculate the FPS
        if counter % fps_avg_frame_count == 0:
            end_time = time.time()
            fps = fps_avg_frame_count / (end_time - start_time)
            start_time = time.time()

        # Show the FPS
        fps_text = 'FPS = {:.1f}'.format(fps)
        text_location = (left_margin, row_size)
        cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    font_size, text_color, font_thickness)

#########################################################################################################################################################################################
#Kode lagt til
        for detection in detection_result.detections:
            for category in detection.categories:
                if "boat" in category.category_name and category.score >= 0.5:
                    current_timestamp = datetime.now()
                    
                    # Sjekk om det har gått 2 minutter siden den siste mappen ble laget 
                    if last_directory_timestamp is None or (current_timestamp - last_directory_timestamp).seconds >= 120:
                        # Hvis det har gått 2 minutter, opprett en ny mappe
                        timestamp_directory = datetime.now().strftime('%Y-%m-%d_%H-%M')
                        directory_detection = f"Directory_for_detection_{timestamp_directory}"
                        parent_dir_detection = "/home/berpol/deteksjon/Fartøys_deteksjon/lite/examples/object_detection/raspberry_pi/deteksjoner"
                        path_dir_run = os.path.join(parent_dir_detection, directory_detection)
                        if not os.path.exists(path_dir_run):
                            os.mkdir(path_dir_run)
                        # Oppdater tidspunktet for når den siste mappen ble laget
                        last_directory_timestamp = current_timestamp
                    
                    if last_saved_timestamp is None or (current_timestamp - last_saved_timestamp).seconds >= 5:
                        image_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                        filename = f"detected_ship_{image_timestamp}.jpg"
                        filename_without_box = f"detected_ship_{image_timestamp}_WithoutBOX.jpg"
                        filepath = os.path.join(path_dir_run, filename)
                        filepath_without_box = os.path.join(path_dir_run, filename_without_box)
                        cv2.imwrite(filepath, image)
                        cv2.imwrite(filepath_without_box, frame_copy)
                        name_of_file = "Documented_detection_data.txt"
                        complete_name = os.path.join(path_dir_run, name_of_file)
                        
                        # Beregn objektets geografiske koordinater
                        object_center_low = (
                            detection.bounding_box.origin_x + detection.bounding_box.width / 2,
                            detection.bounding_box.origin_y + detection.bounding_box.height
                        )
                        # Du må definere følgende variabler basert på din kalibrering og kameraets retning
                        camera_coords = (60.2339, 5.1560) # referanse objektets geografiske posisjon
                        reference_object_distance = 3.53 * 1000 # avstanden til referanseobjekt i bildet i meter
                        reference_pixel_position = (775, 468)
                        image_center = (width // 2, height // 2)  # Sentrum av bildet
                        scale = reference_object_distance / pixel_distance(reference_pixel_position, image_center)  # Skalaen fra kalibreringen
                        direction_angle = 350  # Kameraets retning i grader; 350 for nord
                
                        # Få de beregnede koordinatene
                        geo_coords = pixel_to_coords(camera_coords, object_center_low, scale, image_center, direction_angle)
                        
                        with open(complete_name, "a") as f:
                            f.write(image_timestamp + "\t" + category.category_name + "\t" + "{:.2f}".format(category.score) + "\t" + filename_without_box + "/" + filename + "\t" + "Box coordinates: " + "[UPPER LEFT(" + str(detection.bounding_box.origin_x) + ", " + str(detection.bounding_box.origin_y) + "), " + "LOWER RIGHT(" + str(detection.bounding_box.origin_x + detection.bounding_box.width) + ", " + str(detection.bounding_box.origin_y + detection.bounding_box.height) + ")]" + "\t" + str(geo_coords) + "\n")
                        # Oppdater tidspunktet for det siste bildet som ble lagret
                        last_saved_timestamp = current_timestamp

#########################################################################################################################################################################################

        # Stop the program if the ESC key is pressed.
        if cv2.waitKey(1) == 27:
            break
        cv2.imshow('object_detector', image)

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model',
        help='Path of the object detection model.',
        required=False,
        default='efficientdet_lite0.tflite')
    parser.add_argument(
        '--cameraId', help='Id of camera.', required=False, type=int, default=0)
    parser.add_argument(
        '--frameWidth',
        help='Width of frame to capture from camera.',
        required=False,
        type=int,
        default=1280)
    parser.add_argument(
        '--frameHeight',
        help='Height of frame to capture from camera.',
        required=False,
        type=int,
        default=720)
    parser.add_argument(
        '--numThreads',
        help='Number of CPU threads to run the model.',
        required=False,
        type=int,
        default=4)
    parser.add_argument(
        '--enableEdgeTPU',
        help='Run model on EdgeTPU.',
        required=False,
        type=bool,
        default=False)
    args = parser.parse_args()
    run(args.model, args.cameraId, args.frameWidth, args.frameHeight, args.numThreads, args.enableEdgeTPU)


if __name__ == '__main__':
    main()
