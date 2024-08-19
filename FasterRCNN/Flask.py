from flask import Flask, request, jsonify, send_file
import torch
import torchvision
import cv2
import numpy as np
import os

# Initialize the Flask application
app = Flask(__name__)

# Determine the device to use (GPU if available, otherwise CPU)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load the fine-tuned Faster R-CNN model and move it to the appropriate device
try:
    model = torch.load('fine_tuned_faster_rcnn.pth', map_location=device)
    model = model.to(device)
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Function to perform inference on a frame
def detect_pedestrians(frame):
    """
    Perform pedestrian detection on a single frame using the loaded model.
    
    Args:
        frame: The input frame (image) as a numpy array.
        
    Returns:
        numpy array: Array of bounding boxes for detected pedestrians.
    """
    # Transform the frame to a tensor and move it to the appropriate device
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    frame_tensor = transform(frame).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(frame_tensor)

    # Extract the bounding boxes and scores from the predictions
    boxes = predictions[0]['boxes'].cpu()  # Move tensor back to CPU
    scores = predictions[0]['scores'].cpu()  # Move tensor back to CPU

    # Filter out low confidence detections
    threshold = 0.8
    boxes = boxes[scores > threshold].numpy()

    return boxes

@app.route('/process_video', methods=['POST'])
def process_video():
    """
    Handle video file upload, perform pedestrian detection,
    and return the annotated video file.
    
    Returns:
        Response: The annotated video file as an attachment.
    """
    # Check if a video file was provided in the request
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    # Save the uploaded video file
    video_file = request.files['video']
    input_path = os.path.join('uploads', video_file.filename)
    video_file.save(input_path)

    # Open the video file for reading
    video_capture = cv2.VideoCapture(input_path)

    if not video_capture.isOpened():
        return jsonify({'error': 'Error opening video file'}), 400

    # Get video properties
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))

    # Create a VideoWriter object to save the output video
    output_path = os.path.join('outputs', 'output_video.avi')
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width, frame_height))

    # Get the total number of frames in the video
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    try:
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break

            # Perform pedestrian detection on the current frame
            boxes = detect_pedestrians(frame)

            # Draw bounding boxes on the frame
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Write the processed frame to the output video
            out.write(frame)
            current_frame += 1
            print(f"Processing frame {current_frame}/{frame_count}")

        print(f"Processing complete. Video saved as '{output_path}'")
    except Exception as e:
        return jsonify({'error': f"Error during processing: {e}"}), 500
    finally:
        video_capture.release()
        out.release()
        print("Resources released.")

    return send_file(output_path, as_attachment=True)

# Error handler for 404 Not Found
@app.errorhandler(404)
def not_found_error(error):
    """
    Handle 404 Not Found error.
    
    Args:
        error: The error object.
        
    Returns:
        JSON response with error message and 404 status code.
    """
    return jsonify({'error': 'Not Found'}), 404

# Error handler for 400 Bad Request
@app.errorhandler(400)
def bad_request_error(error):
    """
    Handle 400 Bad Request error.
    
    Args:
        error: The error object.
        
    Returns:
        JSON response with error message and 400 status code.
    """
    return jsonify({'error': 'Bad Request'}), 400

# Error handler for 500 Internal Server Error
@app.errorhandler(500)
def internal_server_error(error):
    """
    Handle 500 Internal Server Error.
    
    Args:
        error: The error object.
        
    Returns:
        JSON response with error message and 500 status code.
    """
    return jsonify({'error': 'Internal Server Error'}), 500

if __name__ == '__main__':
    # Ensure the upload and output directories exist
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    # Run the Flask application
    app.run(debug=True, host='0.0.0.0', port=5000)
