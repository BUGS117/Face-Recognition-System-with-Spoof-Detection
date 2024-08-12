import cv2 as cv
import face_recognition
import matplotlib.pyplot as plt

# Load the known image
known_image = face_recognition.load_image_file("mayur.jpeg")
known_image = face_recognition.load_image_file("yash.jpg")
known_face_encoding = face_recognition.face_encodings(known_image, num_jitters=50, model='large')[0]

# Launch the live camera
cam = cv.VideoCapture(0)

# Check if camera is working
if not cam.isOpened():
    print("Camera not working")
    exit()

# When the camera is opened
while True:
    # Capture the image frame-by-frame
    ret, frame = cam.read()
    
    # Check if the frame is being read
    if not ret:
        print("Can't receive the frame")
        break

    # Face detection in the frame
    face_locations = face_recognition.face_locations(frame)

    for face_location in face_locations:
        top, right, bottom, left = face_location
        # Draw a rectangle with blue line borders of thickness of 2 px
        frame = cv.rectangle(frame, (left, top), (right, bottom), color=(0, 0, 255), thickness=2)
        
        # Encode the faces detected in the frame
        live_face_encodings = face_recognition.face_encodings(frame, known_face_locations=[face_location])
        
        for live_face_encoding in live_face_encodings:
            # Match with the known face
            matches = face_recognition.compare_faces([known_face_encoding], live_face_encoding)
            name = "Unknown"

            if matches[0]:
                name = "Yash"

            # Display the name of the person
            cv.putText(frame, name, (left, top - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv.imshow('Face Recognition', frame)

    # End the streaming
    if cv.waitKey(1) == ord('q'):
        break

# Release the capture
cam.release()
cv.destroyAllWindows()
