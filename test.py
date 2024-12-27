import cv2

# Open the default camera (0 is the default camera, 1 is the second, etc.)
camera = cv2.VideoCapture(0)

# Check if the camera was opened successfully
if not camera.isOpened():
    print("Error: Could not access the camera.")
else:
    print("Camera is successfully detected!")

    # Capture video frames from the camera
    while True:
        ret, frame = camera.read()

        # If frame is read correctly, ret will be True
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Display the resulting frame
        cv2.imshow("Camera Test", frame)

        # Wait for the user to press the 'q' key to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close any OpenCV windows
    camera.release()
    cv2.destroyAllWindows()
