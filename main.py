import cv2
import math
import numpy as np

def main():
    # training data untuk menentukan mata dan muka
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    # baca image
    img1 = cv2.imread('2.jpg')
    img2 = cv2.imread('4.jpg')
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # deteksi muka
    faces1 = face_cascade.detectMultiScale(gray1, 1.3, 5)
    faces2 = face_cascade.detectMultiScale(gray2, 1.3, 5)

    # deteksi objek mata dalam muka 1
    for (x, y, w, h) in faces1:
        cv2.rectangle(img1, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray1[y:y + h, x:x + w]
        roi_color = img1[y:y + h, x:x + w]

        eyes1 = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes1:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    # tampungan nilai muka 1 untuk penghitungan jarak
    arr_face1 = np.array(gray1[faces1[0][1]:faces1[0][1] + faces1[0][3], faces1[0][0]:faces1[0][0] + faces1[0][2]])
    print(arr_face1)
    cv2.imshow('img1', img1)

    # deteksi objek mata dalam muka 2
    for (x, y, w, h) in faces2:
        cv2.rectangle(img2, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray2[y:y + h, x:x + w]
        roi_color = img2[y:y + h, x:x + w]

        eyes2 = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes2:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    # tampungan nilai muka 2 untuk penghitungan jarak
    arr_face2 = np.array(gray2[faces2[0][1]:faces2[0][1] + faces1[0][3], faces2[0][0]:faces2[0][0] + faces1[0][2]])
    print(arr_face2)
    cv2.imshow('img2', img2)

    # tampilan perhitungan jarak
    print('Distance between image : ', str(calculateDistance(arr_face1, arr_face2)))

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# fungsi eucledian distance
def calculateDistance(image1, image2):
    distance = 0
    for i in range(len(image1)):
        for j in range(len(image1)):
            distance += math.pow((image1[i][j] - image2[i][j]), 2)
    distance = np.sqrt(distance)
    return distance

if __name__ == '__main__':
    main()