import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()


cv2.namedWindow("Live Face Pixelation", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Live Face Pixelation", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

mode = 'pixelate'

screenshot_count = 1

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame.")
        break\
            
    frame_height, frame_width = frame.shape[:2]
    
    zone_x1 = int(frame_width * 0.25)
    zone_x2 = int(frame_width * 0.75)
    zone_y1 = int(frame_height * 0.25)
    zone_y2 = int(frame_height * 0.75)
    
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    face_count = len(faces)
    
    for i, (x, y, w, h) in enumerate(faces, start= 1):

        cx = x + w // 2
        cy = y + h // 2
        
        face_roi = frame[y:y+h, x:x+w]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"Face {i}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)
        
        if zone_x1 <= cx <= zone_x2 and zone_y1 <= cy <= zone_y2:
            
            if mode == 'pixelate':
                small = cv2.resize(face_roi, (16, 16), interpolation=cv2.INTER_LINEAR)
                pixelated_face = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
                frame[y:y+h, x:x+w] = pixelated_face
                
            
            elif mode == 'blur':
                blurred_face = cv2.GaussianBlur(face_roi, (43, 43), 0)
                frame[y:y+h, x:x+w] = blurred_face
        else:
            cv2.putText(frame, f"Outside Zone", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (0, 0, 255), 2)


    cv2.putText(frame, f"Faces: {face_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    
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
    elif key == ord('s'):
        filename = f"screenshot_{screenshot_count}.png"
        cv2.imwrite(filename, frame)
        print(f"Saved screenshot as {filename}")
        screenshot_count +=1
cap.release()
cv2.destroyAllWindows()
