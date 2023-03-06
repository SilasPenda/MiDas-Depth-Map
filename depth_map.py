# Import necessary libraries
import cv2
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Create arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='DPT_Hybrid', required=True, help='model type')
parser.add_argument('--source', default= 0, required=True, help='path to input source file') #.mp4, .m4v, .jpg, .jpeg, .png
parser.add_argument('--save', required=True, help='output path to save results to') # .mp4, .m4v
args = parser.parse_args()


# Load a model
#model_type = "DPT_Large"   #MiDas v3 - Large (Highest accuracy, slowest inference speed)
#model_type = "DPT_Hybrid"  #MiDas v3 - Hybrid (Medium accuracy, medium inference speed)
#model_type = "MiDaS_small"

model_type = args.model

midas = torch.hub.load('intel-isl/MiDaS', model_type)

# Select device to run model on i.e. Move to GPU is available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
midas.to(device)
midas.eval()

# Load transforms to resize and normalize the image for large or small model
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform
    
    
if args.source.endswith('.jpg' or '.jpeg' or '.png'):
    img = cv2.imread(args.source)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()
    
    depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)            
    depth_map = (depth_map*255).astype(np.uint8)
    depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)
    
    cv2.imwrite(args.save, depth_map)    
else:    
    # Run this for inference on videos or webcam
    cap = cv2.VideoCapture(args.source)

    # Get the Default resolutions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc  = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    # Define the codec and filename.
    writer = cv2.VideoWriter(args.save, fourcc, cap_fps, (frame_width, frame_height))

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print('No frame read!')
            break
        
        startTime = time.time()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        input_batch = transform(img).to(device)

        with torch.no_grad():
            prediction = midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

            depth_map = prediction.cpu().numpy()
            
            depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)

            endTime = time.time()
            fps = round(1/(endTime - startTime))
            
            depth_map = (depth_map*255).astype(np.uint8)
            depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)
            
            #cv2.putText(depth_map, f'FPS: {fps}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
            #cv2.putText(depth_map, f'Depth Map Test 5.2', (20,120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

            cv2.imshow('Image', depth_map)
            
            writer.write(depth_map)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
            time.sleep(0.01)
                
    cap.release()
    cv2.destroyAllWindows()
