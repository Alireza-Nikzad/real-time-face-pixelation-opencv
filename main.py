import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()


cv2.namedWindow("Live Face Pixelation", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Live Face Pixelation", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

mode = 'pixelate'

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame.")
        break\

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:

        face_roi = frame[y:y+h, x:x+w]

        if mode == 'pixelate':
            small = cv2.resize(face_roi, (16, 16), interpolation=cv2.INTER_LINEAR)
            pixelated_face = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
            frame[y:y+h, x:x+w] = pixelated_face
            
        
        elif mode == 'blur':
            blurred_face = cv2.GaussianBlur(face_roi, (43, 43), 0)
            frame[y:y+h, x:x+w] = blurred_face

    cv2.imshow("Live Face Pixelation", frame)


    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('p'):
        mode = 'pixelate'
        print("Switched to pixelation mode")
    elif key == ord('b'):
        mode = 'blur'
        print("Switched to blur mode")

cap.release()
cv2.destroyAllWindows()
