import cv2

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

# Create a named window before the loop and set it to fullscreen
cv2.namedWindow("Live Face Pixelation", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Live Face Pixelation", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame.")
        break\

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Extract the face region
        face_roi = frame[y:y+h, x:x+w]

        # Pixelate the face
        small = cv2.resize(face_roi, (16, 16), interpolation=cv2.INTER_LINEAR)
        pixelated_face = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

        # Replace original with pixelated
        frame[y:y+h, x:x+w] = pixelated_face

    # Show the frame in fullscreen
    cv2.imshow("Live Face Pixelation", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release and close
cap.release()
cv2.destroyAllWindows()
