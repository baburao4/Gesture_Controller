from hand_tracker import HandTracker
from gesture_controller import GestureController
import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 60)

tracker    = HandTracker()
controller = GestureController()

cv2.namedWindow("Gesture Control", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Gesture Control", 1280, 720)

while True:
    success, img = cap.read()
    if not success:
        continue

    img = cv2.flip(img, 1)
    img = tracker.find_hands(img)
    lm_list = tracker.get_landmarks(img)

    if lm_list:
        controller.handle_gestures(lm_list, img)
    else:
        # Reset scroll state when hand leaves frame so it doesn't jump
        controller.scroll_prev_y = None

    cv2.imshow("Gesture Control", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
