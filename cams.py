import cv2

for i in range(5):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"index {i}: not available")
        continue
    ok, frame = cap.read()
    if ok:
        w, h = int(cap.get(3)), int(cap.get(4))
        print(f"index {i}: OPEN  {w}x{h}  - showing 3s, look at the window")
        cv2.imshow(f"camera index {i}", frame)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()
    cap.release()