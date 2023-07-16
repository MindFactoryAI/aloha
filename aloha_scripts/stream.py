import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda import gpuarray

# Set up video capture
cap = cv2.VideoCapture(0)  # Change the index to the appropriate camera source if needed

# Set up video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (640, 480))  # Adjust resolution and frame rate as needed

# Set up CUDA
cuda.init()
device = cuda.Device(0)  # Use the appropriate GPU device index if you have multiple GPUs
context = device.make_context()

# Create CUDA streams
stream = cuda.Stream()

# Loop to capture and encode video frames
while True:
    ret, frame = cap.read()

    # Convert frame to BGR
    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Allocate GPU memory for input frame
    gpu_frame = gpuarray.to_gpu(bgr_frame)

    # Convert frame to YUV format (required by NVENC)
    yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)

    # Allocate GPU memory for output frame
    gpu_output = cuda.mem_alloc(yuv_frame.nbytes)

    # Copy input frame to GPU memory
    cuda.memcpy_htod_async(gpu_output, yuv_frame, stream)

    # Encode frame using NVENC
    encoder = cv2.cuda.createVideoWriter('output.mp4', 0, 30, (640, 480), True)
    encoder.write(gpu_output)

    # Retrieve encoded frame from GPU memory
    encoded_frame = np.empty_like(yuv_frame)
    cuda.memcpy_dtoh_async(encoded_frame, gpu_output, stream)

    # Write encoded frame to video file
    out.write(encoded_frame)

    # Display the frame
    cv2.imshow('frame', frame)

    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()