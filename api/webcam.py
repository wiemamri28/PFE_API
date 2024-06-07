import cv2

# Initialize the video capture object
video = cv2.VideoCapture("rtsp://admin:admin@192.168.10.30:554/play1.sdp")

# Check if the webcam is opened correctly
if not video.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = video.read()
    
    # If frame is read correctly, ret is True
    if not ret:
        print("Error: Could not read frame.")
        break

    # Display the frame
    cv2.imshow("Webcam Feed", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture object and close all OpenCV windows
video.release()
cv2.destroyAllWindows()
