# ASL-Alphabet

ASL-Alphabet is a real-time American Sign Language (ASL) alphabet detection and interpretation project utilizing computer vision. It employs advanced image processing and machine learning techniques to recognize and interpret ASL alphabets through webcam input.

## Installation

To use ASL-Alphabet, install the required Python packages using the following commands:

```bash
pip install cvzone
pip install mediapipe
pip install opencv-python opencv-python-headless
pip install opencv-contrib-python
```

Make sure you are running in the correct Python environment. If you encounter the error "opencv2 could not be resolved," execute the command above to resolve it.

There is currently no training data for the letters "J" or "Z" as their signs require motion. 

## Dependencies

ASL-Alphabet relies on the following dependencies:

### cvzone-1.6.1
### numpy-1.26.3
### opencv-python-4.9.0.80

### mediapipe-0.10.9
### opencv-contrib-python-4.9.0.80

## Troubleshooting

If you encounter the error "AttributeError: module 'cv2' has no attribute 'imgshow'," run the following command:

```bash
pip install opencv-contrib-python
```

### Mac Users

Ensure that Airplay & Handoff -> Continuity Camera is disabled to prevent any potential conflicts.

Feel free to contribute to this project and help improve the ASL-Alphabet experience for everyone!
