# from flask import Flask, request, jsonify
# import cv2
# import mediapipe as mp
# import numpy as np
# import requests
# import os
# import math

# app = Flask(__name__)

# # Initialize MediaPipe Pose
# mp_pose = mp.solutions.pose
# mp_drawing = mp.solutions.drawing_utils
# pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# def calculate_angle(a, b, c):
#     a = np.array(a)
#     b = np.array(b)
#     c = np.array(c)
#     radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
#     angle = np.abs(radians*180.0/np.pi)
#     if angle > 180:
#         angle = 360 - angle
#     return angle

# def extract_landmarks(results):
#     if not results.pose_landmarks:
#         return None
#     landmarks = results.pose_landmarks.landmark
#     return [(lm.x, lm.y) for lm in landmarks]

# def get_point(landmarks, idx, shape):
#     h, w = shape
#     return (int(landmarks[idx][0] * w), int(landmarks[idx][1] * h))

# class Exercise:
#     def __init__(self, name):
#         self.name = name
#         self.stage = None
#         self.count = 0
#         self.feedback = ""
#         self.angle = 0

#     def process(self, landmarks, shape):
#         raise NotImplementedError

#     def to_dict(self):
#         return {
#             'name': self.name,
#             'reps': self.count,
#             'feedback': self.feedback,
#             'angle': round(self.angle, 1)
#         }

# class BicepCurl(Exercise):
#     def __init__(self):
#         super().__init__("Bicep Curl")

#     def process(self, landmarks, shape):
#         shoulder = get_point(landmarks, 11, shape)
#         elbow = get_point(landmarks, 13, shape)
#         wrist = get_point(landmarks, 15, shape)

#         self.angle = calculate_angle(shoulder, elbow, wrist)

#         if self.angle > 160:
#             self.stage = "down"
#             self.feedback = "Curl up"
#         elif self.angle < 30 and self.stage == "down":
#             self.stage = "up"
#             self.count += 1
#             self.feedback = "Good curl!"
#         elif self.angle < 160:
#             self.feedback = "Keep elbows steady"

# class Squat(Exercise):
#     def __init__(self):
#         super().__init__("Squat")

#     def process(self, landmarks, shape):
#         hip = get_point(landmarks, 23, shape)
#         knee = get_point(landmarks, 25, shape)
#         ankle = get_point(landmarks, 27, shape)

#         self.angle = calculate_angle(hip, knee, ankle)

#         if self.angle > 160:
#             self.stage = "up"
#             self.feedback = "Lower your hips"
#         elif self.angle < 90 and self.stage == "up":
#             self.stage = "down"
#             self.count += 1
#             self.feedback = "Nice squat!"
#         elif self.angle < 160:
#             self.feedback = "Go deeper"

# class LateralRaise(Exercise):
#     def __init__(self):
#         super().__init__("Lateral Raise")
    
#     def process(self, landmarks, shape):
#         shoulder = get_point(landmarks, 11, shape)
#         elbow = get_point(landmarks, 13, shape)
#         wrist = get_point(landmarks, 15, shape)

#         self.angle = calculate_angle(shoulder, elbow, wrist)

#         if self.angle > 160:
#             self.stage = "down"
#             self.feedback = "Raise arms to the side"
#         elif 70 < self.angle < 110 and self.stage == "down":
#             self.stage = "up"
#             self.count += 1
#             self.feedback = "Good! Control the movement"
#         elif self.angle < 60:
#             self.feedback = "Raise higher"

# class ShoulderPress(Exercise):
#     def __init__(self):
#         super().__init__("Shoulder Press")

#     def process(self, landmarks, shape):
#         shoulder = get_point(landmarks, 11, shape)
#         elbow = get_point(landmarks, 13, shape)
#         wrist = get_point(landmarks, 15, shape)

#         self.angle = calculate_angle(shoulder, elbow, wrist)

#         if self.angle > 150:
#             self.stage = "down"
#             self.feedback = "Press up fully"
#         elif self.angle < 70 and self.stage == "down":
#             self.stage = "up"
#             self.count += 1
#             self.feedback = "Nice rep!"
#         elif self.angle < 160:
#             self.feedback = "Lock elbows on top"

# class Pushup(Exercise):
#     def __init__(self):
#         super().__init__("Pushup")

#     def process(self, landmarks, shape):
#         shoulder = get_point(landmarks, 12, shape)
#         elbow = get_point(landmarks, 14, shape)
#         wrist = get_point(landmarks, 16, shape)

#         self.angle = calculate_angle(shoulder, elbow, wrist)

#         if self.angle > 150:
#             self.stage = "up"
#             self.feedback = "Lower your body"
#         elif self.angle < 90 and self.stage == "up":
#             self.stage = "down"
#             self.count += 1
#             self.feedback = "Great! Push back up"
#         elif self.angle < 150:
#             self.feedback = "Go lower"

# def draw_info(img, exercise):
#     cv2.putText(img, f"Exercise: {exercise.name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
#     cv2.putText(img, f"Reps: {exercise.count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)
#     cv2.putText(img, f"Feedback: {exercise.feedback}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
#     cv2.putText(img, f"Angle: {round(exercise.angle, 1)} deg", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

# EXERCISE_CLASSES = {
#     'bicep_curl': BicepCurl,
#     'squat': Squat,
#     'lateral_raise': LateralRaise,
#     'shoulder_press': ShoulderPress,
#     'pushup': Pushup
# }

# @app.route('/process-video', methods=['POST'])
# def process_video():
#     data = request.get_json()
#     video_url = data.get('video_url')
#     exercise_type = data.get('exercise_type')

#     if not video_url:
#         return jsonify({'error': 'No video URL provided'}), 400
#     if not exercise_type or exercise_type not in EXERCISE_CLASSES:
#         return jsonify({'error': 'Invalid or missing exercise type'}), 400

#     try:
#         # Download video from Cloudinary
#         response = requests.get(video_url, stream=True)
#         input_path = 'temp_input.mp4'
#         with open(input_path, 'wb') as f:
#             for chunk in response.iter_content(chunk_size=1024):
#                 if chunk:
#                     f.write(chunk)

#         # Process video
#         cap = cv2.VideoCapture(input_path)
#         if not cap.isOpened():
#             return jsonify({'error': 'Could not open video'}), 500

#         # Get video properties
#         width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         fps = int(cap.get(cv2.CAP_PROP_FPS))
        
#         # Output video setup
#         output_path = 'temp_output.mp4'
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

#         # Initialize exercise
#         exercise = EXERCISE_CLASSES[exercise_type]()
#         feedback_data = []

#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             # Convert frame to RGB for MediaPipe
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = pose.process(frame_rgb)

#             # Draw landmarks and process exercise
#             if results.pose_landmarks:
#                 mp_drawing.draw_landmarks(
#                     frame,
#                     results.pose_landmarks,
#                     mp_pose.POSE_CONNECTIONS,
#                     mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
#                     mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
#                 )
#                 landmarks = extract_landmarks(results)
#                 if landmarks:
#                     exercise.process(landmarks, frame.shape[:2])
#                     draw_info(frame, exercise)
#                     feedback_data.append(exercise.to_dict())

#             out.write(frame)

#         cap.release()
#         out.release()

#         # Upload processed video to Cloudinary
#         with open(output_path, 'rb') as video_file:
#             files = {'file': (os.path.basename(output_path), video_file, 'video/mp4')}
#             data = {
#                 'upload_preset': 'processed_workout_upload',
#                 'folder': 'processed_videos'
#             }
#             response = requests.post(
#                 'https://api.cloudinary.com/v1_1/dmwaesnu7/video/upload',
#                 files=files,
#                 data=data
#             )

#         if response.status_code != 200:
#             return jsonify({'error': 'Failed to upload processed video to Cloudinary'}), 500

#         upload_result = response.json()
#         feedback_video_url = upload_result['secure_url']

#         # Clean up
#         os.remove(input_path)
#         os.remove(output_path)

#         return jsonify({
#             'feedback_video_url': feedback_video_url,
#             'feedback_data': feedback_data
#         })

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))









from collections import deque
import numpy as np

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180:
        angle = 360 - angle
    return angle

def smooth_angle(current_angle, angle_buffer, window_size=5):
    angle_buffer.append(current_angle)
    if len(angle_buffer) > window_size:
        angle_buffer.popleft()
    return np.mean(angle_buffer)

def is_torso_aligned(landmarks, shape, max_tilt=20):
    """Check if torso is roughly vertical to prevent counting invalid reps."""
    hip = get_point(landmarks, 23, shape)
    shoulder = get_point(landmarks, 11, shape)
    torso_angle = np.abs(np.arctan2(shoulder[1] - hip[1], shoulder[0] - hip[0]) * 180.0 / np.pi)
    return abs(torso_angle - 90) < max_tilt

def get_point(landmarks, idx, shape):
    h, w = shape
    return (int(landmarks[idx][0] * w), int(landmarks[idx][1] * h))

class Exercise:
    def __init__(self, name):
        self.name = name
        self.stage = None
        self.count = 0
        self.feedback = ""
        self.angle = 0
        self.angle_buffer = deque(maxlen=5)  # For smoothing angles
        self.state_frames = 0  # Track frames in current state
        self.min_state_frames = 5  # Minimum frames to confirm state change
        self.last_rep_time = 0  # Frame count for last rep
        self.max_rep_duration = 120  # Max frames (e.g., 4s at 30fps) for a rep

    def process(self, landmarks, shape, frame_count):
        raise NotImplementedError

    def validate_form(self, landmarks, shape):
        """Override in subclasses for exercise-specific form checks."""
        return True

    def to_dict(self):
        return {
            'name': self.name,
            'reps': self.count,
            'feedback': self.feedback,
            'angle': round(self.angle, 1)
        }

class BicepCurl(Exercise):
    def __init__(self):
        super().__init__("Bicep Curl")

    def validate_form(self, landmarks, shape):
        # Ensure elbow stays relatively stationary
        elbow = get_point(landmarks, 13, shape)
        shoulder = get_point(landmarks, 11, shape)
        elbow_movement = np.sqrt((elbow[0] - shoulder[0])**2 + (elbow[1] - shoulder[1])**2)
        return elbow_movement < shape[1] * 0.3 and is_torso_aligned(landmarks, shape)

    def process(self, landmarks, shape, frame_count):
        if not self.validate_form(landmarks, shape):
            self.feedback = "Fix form: Keep elbow steady and torso upright"
            return

        shoulder = get_point(landmarks, 11, shape)
        elbow = get_point(landmarks, 13, shape)
        wrist = get_point(landmarks, 15, shape)

        current_angle = calculate_angle(shoulder, elbow, wrist)
        self.angle = smooth_angle(current_angle, self.angle_buffer)

        if self.angle > 160:
            if self.stage != "down":
                self.state_frames += 1
                if self.state_frames >= self.min_state_frames:
                    self.stage = "down"
                    self.feedback = "Curl up"
                    self.state_frames = 0
        elif self.angle < 30 and self.stage == "down":
            if self.state_frames >= self.min_state_frames:
                if frame_count - self.last_rep_time < self.max_rep_duration:
                    self.stage = "up"
                    self.count += 1
                    self.feedback = "Good curl!"
                    self.last_rep_time = frame_count
                    self.state_frames = 0
                else:
                    self.feedback = "Too slow, reset position"
        elif 30 <= self.angle <= 160:
            self.feedback = "Complete the full curl"
            self.state_frames = 0

class Squat(Exercise):
    def __init__(self):
        super().__init__("Squat")

    def validate_form(self, landmarks, shape):
        # Ensure knees don't go too far forward
        knee = get_point(landmarks, 25, shape)
        ankle = get_point(landmarks, 27, shape)
        knee_forward = abs(knee[0] - ankle[0])
        return knee_forward < shape[1] * 0.2 and is_torso_aligned(landmarks, shape)

    def process(self, landmarks, shape, frame_count):
        if not self.validate_form(landmarks, shape):
            self.feedback = "Fix form: Keep knees over ankles and torso upright"
            return

        hip = get_point(landmarks, 23, shape)
        knee = get_point(landmarks, 25, shape)
        ankle = get_point(landmarks, 27, shape)

        current_angle = calculate_angle(hip, knee, ankle)
        self.angle = smooth_angle(current_angle, self.angle_buffer)

        if self.angle > 160:
            if self.stage != "up":
                self.state_frames += 1
                if self.state_frames >= self.min_state_frames:
                    self.stage = "up"
                    self.feedback = "Lower your hips"
                    self.state_frames = 0
        elif self.angle < 90 and self.stage == "up":
            if self.state_frames >= self.min_state_frames:
                if frame_count - self.last_rep_time < self.max_rep_duration:
                    self.stage = "down"
                    self.count += 1
                    self.feedback = "Nice squat!"
                    self.last_rep_time = frame_count
                    self.state_frames = 0
                else:
                    self.feedback = "Too slow, reset position"
        elif 90 <= self.angle <= 160:
            self.feedback = "Go deeper"
            self.state_frames = 0

class LateralRaise(Exercise):
    def __init__(self):
        super().__init__("Lateral Raise")

    def validate_form(self, landmarks, shape):
        # Ensure arms are roughly symmetrical
        left_elbow = get_point(landmarks, 13, shape)
        right_elbow = get_point(landmarks, 14, shape)
        height_diff = abs(left_elbow[1] - right_elbow[1])
        return height_diff < shape[1] * 0.1 and is_torso_aligned(landmarks, shape)

    def process(self, landmarks, shape, frame_count):
        if not self.validate_form(landmarks, shape):
            self.feedback = "Fix form: Raise arms symmetrically and keep torso upright"
            return

        shoulder = get_point(landmarks, 11, shape)
        elbow = get_point(landmarks, 13, shape)
        wrist = get_point(landmarks, 15, shape)

        current_angle = calculate_angle(shoulder, elbow, wrist)
        self.angle = smooth_angle(current_angle, self.angle_buffer)

        if self.angle > 160:
            if self.stage != "down":
                self.state_frames += 1
                if self.state_frames >= self.min_state_frames:
                    self.stage = "down"
                    self.feedback = "Raise arms to the side"
                    self.state_frames = 0
        elif 70 < self.angle < 110 and self.stage == "down":
            if self.state_frames >= self.min_state_frames:
                if frame_count - self.last_rep_time < self.max_rep_duration:
                    self.stage = "up"
                    self.count += 1
                    self.feedback = "Good! Control the movement"
                    self.last_rep_time = frame_count
                    self.state_frames = 0
                else:
                    self.feedback = "Too slow, reset position"
        elif self.angle <= 70 or self.angle >= 110:
            self.feedback = "Adjust to shoulder height"
            self.state_frames = 0

class ShoulderPress(Exercise):
    def __init__(self):
        super().__init__("Shoulder Press")

    def validate_form(self, landmarks, shape):
        # Ensure elbows don't flare too wide
        elbow = get_point(landmarks, 13, shape)
        shoulder = get_point(landmarks, 11, shape)
        elbow_distance = abs(elbow[0] - shoulder[0])
        return elbow_distance < shape[1] * 0.4 and is_torso_aligned(landmarks, shape)

    def process(self, landmarks, shape, frame_count):
        if not self.validate_form(landmarks, shape):
            self.feedback = "Fix form: Keep elbows under wrists and torso upright"
            return

        shoulder = get_point(landmarks, 11, shape)
        elbow = get_point(landmarks, 13, shape)
        wrist = get_point(landmarks, 15, shape)

        current_angle = calculate_angle(shoulder, elbow, wrist)
        self.angle = smooth_angle(current_angle, self.angle_buffer)

        if self.angle > 150:
            if self.stage != "down":
                self.state_frames += 1
                if self.state_frames >= self.min_state_frames:
                    self.stage = "down"
                    self.feedback = "Press up fully"
                    self.state_frames = 0
        elif self.angle < 70 and self.stage == "down":
            if self.state_frames >= self.min_state_frames:
                if frame_count - self.last_rep_time < self.max_rep_duration:
                    self.stage = "up"
                    self.count += 1
                    self.feedback = "Nice rep!"
                    self.last_rep_time = frame_count
                    self.state_frames = 0
                else:
                    self.feedback = "Too slow, reset position"
        elif 70 <= self.angle <= 150:
            self.feedback = "Lock elbows on top"
            self.state_frames = 0

class Pushup(Exercise):
    def __init__(self):
        super().__init__("Pushup")

    def validate_form(self, landmarks, shape):
        # Ensure body forms a straight line
        hip = get_point(landmarks, 23, shape)
        shoulder = get_point(landmarks, 12, shape)
        ankle = get_point(landmarks, 28, shape)
        body_angle = calculate_angle(shoulder, hip, ankle)
        return 160 < body_angle < 200  # Roughly straight body

    def process(self, landmarks, shape, frame_count):
        if not self.validate_form(landmarks, shape):
            self.feedback = "Fix form: Keep body straight"
            return

        shoulder = get_point(landmarks, 12, shape)
        elbow = get_point(landmarks, 14, shape)
        wrist = get_point(landmarks, 16, shape)

        current_angle = calculate_angle(shoulder, elbow, wrist)
        self.angle = smooth_angle(current_angle, self.angle_buffer)

        if self.angle > 150:
            if self.stage != "up":
                self.state_frames += 1
                if self.state_frames >= self.min_state_frames:
                    self.stage = "up"
                    self.feedback = "Lower your body"
                    self.state_frames = 0
        elif self.angle < 90 and self.stage == "up":
            if self.state_frames >= self.min_state_frames:
                if frame_count - self.last_rep_time < self.max_rep_duration:
                    self.stage = "down"
                    self.count += 1
                    self.feedback = "Great! Push back up"
                    self.last_rep_time = frame_count
                    self.state_frames = 0
                else:
                    self.feedback = "Too slow, reset position"
        elif 90 <= self.angle <= 150:
            self.feedback = "Go lower"
            self.state_frames = 0