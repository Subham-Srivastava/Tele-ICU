from ultralytics import YOLO
import tensorrt
model=YOLO("best.pt")
model.export(format="engine")