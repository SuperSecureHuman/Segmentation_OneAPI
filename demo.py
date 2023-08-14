import cv2
import numpy as np
import time
from openvino.runtime import Core
import torch

core = Core()

devices = core.available_devices

for device in devices:
    device_name = core.get_property(device, "FULL_DEVICE_NAME")
    print(f"{device}: {device_name}")

ir_model = core.read_model(model='./model/model_int8.xml', weights='./model/model_int8.bin')
compiled_model = core.compile_model(model=ir_model, device_name="CPU")

input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

print(f"Input layer shape: {input_layer.shape}")
print(f"Output layer shape: {output_layer.shape}")

def preprocess_frame(frame):
    pre_time = time.time()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (256, 256))
    frame_org = frame
    frame = torch.from_numpy(frame).float().permute(2, 0, 1).unsqueeze(0)
    pre_time_end = time.time() - pre_time
    return frame, pre_time_end

def perform_inference(model, frame):
    start_time = time.time()
    mask = model(frame)[output_layer]
    inference_time = time.time() - start_time
    mask = torch.from_numpy(mask)
    mask = torch.sigmoid(mask.squeeze()).round().long()
    mask = mask.float().squeeze().detach().numpy()
    return mask, inference_time

def post_process(preds, original_frame):
    start_time = time.time()
    preds = cv2.resize(preds, (original_frame.shape[1], original_frame.shape[0]))
    preds = preds * 255
    preds = preds.astype(np.uint8)
    
    # Create a blue overlay of the prediction mask on the original frame
    blue_overlay = np.zeros_like(original_frame)
    blue_overlay[:, :, 0] = preds  # Set blue channel to the prediction mask
    blue_overlay = cv2.addWeighted(original_frame, 0.5, blue_overlay, 0.5, 0)
    
    post_processing_time = time.time() - start_time
    return blue_overlay, post_processing_time

cap = cv2.VideoCapture('./cat.mp4')

frame_count = 0
start_time = time.time()

while True:
    ret, frame_orig = cap.read()
    if ret:
        frame, pre_time = preprocess_frame(frame_orig)
        mask, inference_time = perform_inference(compiled_model, frame)
        output_frame, post_processing_time = post_process(mask, frame_orig)
        
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        
        update_str = f"FPS: {fps:.2f}, Frame Time: {1000 / fps:.2f} ms, Preprocessing Time: {pre_time:.2f} s, Inference Time: {inference_time:.2f} s, Post-processing Time: {post_processing_time:.2f} s"
        print(update_str, end='\r')

        
        cv2.imshow('frame', output_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    else:
        break
