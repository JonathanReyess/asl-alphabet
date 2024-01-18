# asl-alphabet
Detect and interprets American Sign Language (ASL) alphabets in real-time with computer vision by employing advanced image processing and machine learning.

install cvzone and mediapipe python packages

pip install cvzone

this includes the following: cvzone-1.6.1 numpy-1.26.3 opencv-python-4.9.0.80

pip install mediapipe

this includes the following: absl-py-2.1.0 attrs-23.2.0 contourpy-1.2.0 cycler-0.12.1 flatbuffers-23.5.26 fonttools-4.47.2 kiwisolver-1.4.5 matplotlib-3.8.2 mediapipe-0.10.9 opencv-contrib-python-4.9.0.80 pillow-10.2.0 protobuf-3.20.3 pyparsing-3.1.1 python-dateutil-2.8.2 six-1.16.0 sounddevice-0.4.6

you may get the error opencv2 could not be resolved, try this 

pip install opencv-python opencv-python-headless

ensure you are running /Users/jonathanreyes/miniconda3/lib/python3.11/site-packages

also run pip install opencv-contrib-python if you get a "AttributeError: module 'cv2' has no attribute 'imgshow'"

If you're on a Mac, ensure that Airplay & Handoff -> Continuity Camera is disabled