# FaceMaskDetection
This project is an implementation of a Face Mask Detection system using computer vision and deep learning techniques. The goal is to automatically detect whether a person is wearing a face mask or not in real-time.

**Features**
Real-time Detection: The system can process video streams or images in real-time, making it suitable for monitoring scenarios where face mask compliance needs to be enforced.

Accurate Classification: Leveraging state-of-the-art deep learning models, the system achieves high accuracy in classifying individuals into "with mask" or "without mask" categories.

Easy Integration: The codebase is designed for easy integration into existing security systems, CCTV cameras, or as a standalone application.

**Technologies Used**
OpenCV: Used for real-time computer vision tasks, including face detection and image processing.

TensorFlow: Employed to train and deploy deep learning models for face mask classification.

MobileNetV2: The chosen neural network architecture for its balance between speed and accuracy in real-time applications.

**Usage**
Clone the repository: git clone https://github.com/ashutosh4972/FaceMaskDetection.git
Download the trained model from the drive lin or train the model using the notebook file and dataset from the drive link. 
Link - https://drive.google.com/drive/folders/1qlMBq1-AM0-ziR77XBgo_Xr8IFkII3OW?usp=sharing
Paste the model in the same directory.
Run the detection script: python detect_masks.py
