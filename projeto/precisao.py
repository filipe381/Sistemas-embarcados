import face_recognition
import cv2

video_capture = cv2.VideoCapture(0)

total_frames = 0
successful_detections = 0

while True:
    # Capture frame
    ret, frame = video_capture.read()
    total_frames += 1

    # Detect faces in frame
    face_locations = face_recognition.face_locations(frame)

    # If a face is detected, count it
    if face_locations:
        successful_detections += 1

    # Show frame for visualization
    cv2.imshow('Video', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Calculate detection rate
detection_rate = successful_detections / total_frames * 100
print(f"Detection Rate: {detection_rate:.2f}%")

video_capture.release()
cv2.destroyAllWindows()
