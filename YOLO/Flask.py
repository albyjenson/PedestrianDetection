from flask import Flask, request, send_file, jsonify
from ultralytics import YOLO
import cv2
import os

# Initialize the Flask application
app = Flask(__name__)

# Set the upload folder for video files
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Load your trained YOLO model
model = YOLO('best (1).pt')

@app.route('/upload', methods=['POST'])
def upload_video():
    """
    Handle video file upload, perform object detection,
    and return the annotated video file.
    
    Returns:
        Response: The annotated video file as an attachment.
    """
    # Get the uploaded video file from the request
    file = request.files['video']
    filename = file.filename
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)
    
    # Open the uploaded video file
    cap = cv2.VideoCapture(file_path)
    
    if not cap.isOpened():
        return 'Error: Could not open video.'
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create a VideoWriter object to save the output video
    output_filename = os.path.join(OUTPUT_FOLDER, 'output_video.mp4')
    out = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run YOLO model inference on the frame
        results = model.predict(frame)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0]  # Get box coordinates in (left, top, right, bottom) format
                x1, y1, x2, y2 = map(int, b)
                # Draw a rectangle around the detected object
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Write the frame to the output video
        out.write(frame)
    
    # Release video objects
    cap.release()
    out.release()
    
    # Send the output video file as a response
    return send_file(output_filename, as_attachment=True)

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
    # Run the Flask application
    app.run(debug=True)
