import cv2

def drawBox(frame, bbox):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2, 1)

video = cv2.VideoCapture("footvolleyball.mp4")
tracker = cv2.TrackerCSRT_create()

success, frame = video.read()
bbox = cv2.selectROI("Frame", frame, False)
tracker.init(frame, bbox)

while True:
    success, frame = video.read()
    if not success:
        break

    success, bbox = tracker.update(frame)
    if success:
        drawBox(frame, bbox)
    else:
        cv2.putText(frame, "Object lost", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()


