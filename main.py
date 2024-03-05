import dlib
import cv2
from scipy.spatial import distance as dist
from imutils import face_utils
import math
import playsound


def area_tri(a, b, c):
    a = float(a)
    b = float(b)
    c = float(c)
    if a <= 0 or b <= 0 or c <= 0:
        return
    else:
        zb = (a + b + c) / 2
        area = zb * math.sqrt((zb - a) + (zb - b) + (zb - c))
        return area


def area_rec(a, b):
    if a <= 0 or b <= 0:
        return
    area = a * b
    return area


def area_of_eye(eye):
    a = dist.euclidean(eye[0], eye[1])
    b = dist.euclidean(eye[1], eye[2])
    c = dist.euclidean(eye[2], eye[3])
    d = dist.euclidean(eye[3], eye[4])
    f = dist.euclidean(eye[5], eye[0])
    g = dist.euclidean(eye[1], eye[5])
    h = dist.euclidean(eye[2], eye[4])
    tri1 = area_tri(a, f, g)
    tri2 = area_tri(c, d, h)
    rec = area_rec(b, h)
    area = sum([tri1, tri2, rec])
    return area


def average(n, z):
    kl = (n + z) / 2.0
    return kl


def get_the_path_of_the_alarm_music():
    # n = input('Enter the path of the alarm : ')
    n = "D:\Alarm-Fast-A1-www.fesliyanstudios.com (1).mp3"
    return n


def the_video_path():
    n = input("Do you want video stream from the main camera of the laptop or you want to stream from another camera "
              "or you want to import a video from the laptop? device / another camera")
    if n == "device":
        kl = int(0)
        print('START VIDEO STREAM')
        return kl
    else:
        kl = int(input("Enter the number of the camera : "))
        print('START VIDEO STREAM')
        return kl


def count(avg_area, referenced, cont):
    while avg_area <= referenced:
        cont = cont + 1
        # print(cont)
        return cont
    while avg_area > referenced:
        cont = 0
        # print(cont)
        return cont


def alarm(cont, number_frames, path):
    if cont >= number_frames:
        b = playsound.playsound(path)
        return b
    if cont < number_frames:
        cont = 0
        return cont


def get_the_number_of_frames():
    n = int(input("Enter the number of seconds which the alarm will ring after getting drowsy : "))
    kl = n * 10
    return kl


def info():
    vk = "...VIDEO STREAMING"
    cv2.putText(frame, vk, (20, 430), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    kv = "...detecting the motion eye"
    cv2.putText(frame, kv, (20, 460), cv2.FONT_HERSHEY_PLAIN, 1.8, (200, 200, 200), 2)


def referenced_area():
    detector_ref = dlib.get_frontal_face_detector()
    predictor_ref = dlib.shape_predictor("C:/Users/aly farag/.spyder-py3/shape_predictor_68_face_landmarks (2).dat")

    (left_eye_start_ref, left_eye_end_ref) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (right_eye_start_ref, right_eye_end_ref) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    video_streaming_ref = cv2.VideoCapture(the_video_path())
    print("PLEASE CLOSE YOUR EYES")
    frames_ref = 0
    ref = []
    while True:
        ret_ref, frame_ref = video_streaming_ref.read()
        if not ret_ref:
            print("no cam")
            break

        if cv2.waitKey(1) == ord('a'):
            break

        rectangles_ref = detector_ref(frame_ref)
        for rect_ref in rectangles_ref:
            shape_ref = predictor_ref(frame_ref, rect_ref)
            shape_ref = face_utils.shape_to_np(shape_ref)
            for (f, p) in shape_ref:
                cv2.circle(frame_ref, (f, p), 1, (0, 0, 255), -1)
            left_eye_ref = shape_ref[left_eye_start_ref:left_eye_end_ref]
            right_eye_ref = shape_ref[right_eye_start_ref:right_eye_end_ref]
            left_eye_area_ref = area_of_eye(left_eye_ref)
            right_eye_area_ref = area_of_eye(right_eye_ref)
            referenced_area_ref = average(left_eye_area_ref, right_eye_area_ref)
            ref.append(referenced_area_ref)
            frames_ref = frames_ref + 1
            if frames_ref >= 30:
                avg = sum(ref) / len(ref)
                print(avg)
                return avg


counter = 0
eye_area = referenced_area()
eye_frames = get_the_number_of_frames()
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:/Users/aly farag/.spyder-py3/shape_predictor_68_face_landmarks (2).dat")
(left_eye_start, left_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(right_eye_start, right_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
video_streaming = cv2.VideoCapture(the_video_path())
while True:
    ret, frame = video_streaming.read()
    if not ret:
        print("no cam")
        break

    if cv2.waitKey(1) == ord('a'):
        break

    rectangles = detector(frame)
    for rect in rectangles:
        shape = predictor(frame, rect)
        shape = face_utils.shape_to_np(shape)

        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

        left_eye = shape[left_eye_start:left_eye_end]
        right_eye = shape[right_eye_start:right_eye_end]
        left_eye_area = area_of_eye(left_eye)
        right_eye_area = area_of_eye(right_eye)
        average_area = average(left_eye_area, right_eye_area)
        print(average_area)
        m = count(average_area, eye_area, counter)
        k = alarm(m, eye_frames, get_the_path_of_the_alarm_music())
        counter = m

    info()
    cv2.imshow("frame", frame)
video_streaming.release()
cv2.destroyAllWindows()
