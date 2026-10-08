import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np


# ==========================================
# 1. LOAD THE ASL MODEL
# ==========================================

model = tf.keras.models.load_model("asl_model_finetuned.keras")

class_names = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I",
    "J", "K", "L", "M", "N", "O", "P", "Q", "R",
    "S", "T", "U", "V", "W", "X", "Y", "Z",
    "del", "nothing", "space"
]


# ==========================================
# 2. SET UP MEDIAPIPE
# ==========================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)


# ==========================================
# 3. OPEN WEBCAM
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam.")
    exit()

print("ASL recognizer started.")
print("Press Q to quit.")


# ==========================================
# 4. RUN EVERYTHING
# ==========================================

with HandLandmarker.create_from_options(options) as landmarker:

    while True:
        success, raw = cap.read()
        if not success:
            print("Could not read frame.")
            break

        height, width, _ = raw.shape

        # Run MediaPipe on the UNFLIPPED frame
        rgb_frame = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = landmarker.detect(mp_image)

        hand_crop = None
        box = None
        label = None

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]

            xs = [int(lm.x * width) for lm in hand]
            ys = [int(lm.y * height) for lm in hand]

            # Make the box SQUARE, centered on the hand, with extra padding
            cx = (min(xs) + max(xs)) // 2
            cy = (min(ys) + max(ys)) // 2
            half = int(max(max(xs) - min(xs), max(ys) - min(ys)) * 0.7)

            x1, x2 = max(cx - half, 0), min(cx + half, width)
            y1, y2 = max(cy - half, 0), min(cy + half, height)

            hand_crop = raw[y1:y2, x1:x2]

            if hand_crop.size > 0:
                image = cv2.resize(hand_crop, (224, 224))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = np.expand_dims(image.astype(np.float32), axis=0)

                predictions = model.predict(image, verbose=0)[0]

                # Show top 3 so you can see what the model is "thinking"
                top3 = np.argsort(predictions)[::-1][:3]
                label = [f"{class_names[i]} {predictions[i]*100:.0f}%" for i in top3]
                box = (x1, y1, x2, y2)

        # Flip ONLY the copy used for display
        display = cv2.flip(raw, 1)

        if box is not None:
            x1, y1, x2, y2 = box
            # mirror the box x-coordinates to match the flipped display
            cv2.rectangle(display, (width - x2, y1), (width - x1, y2), (0, 255, 0), 2)
            for i, line in enumerate(label):
                cv2.putText(display, line, (30, 60 + i * 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        cv2.imshow("ASL Recognizer", display)

        if hand_crop is not None and hand_crop.size > 0:
            cv2.imshow("What CNN Sees", hand_crop)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# ==========================================
# 9. CLEAN UP
# ==========================================

cap.release()
cv2.destroyAllWindows()