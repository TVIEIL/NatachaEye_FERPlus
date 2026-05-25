import cv2
import onnxruntime
import torch
print("OpenCV version:", cv2.__version__)
print("ONNX Runtime version:", onnxruntime.__version__)
print("CUDA disponible:", torch.cuda.is_available())
