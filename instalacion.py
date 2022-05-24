import os
import sys
import subprocess

os.system("sudo apt-get install build-essential cmake pkg-config libx11-dev libatlas-base-dev libgtk-3-dev libboost-python-dev")
os.system("sudo apt-get install build-essential cmake pkg-config libx11-dev libatlas-base-dev libgtk-3-dev libboost-python-dev")
os.system("sudo apt-get install python3-dev python3-pip")
os.system("sudo -H pip3 install -U pip numpy")

# implement pip as a subprocess:
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
'opencv-python'])

subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
'dlib'])

subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
'face_recognition'])