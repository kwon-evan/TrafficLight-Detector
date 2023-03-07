#!/usr/bin/env python
# coding: utf-8
# created by hevlhayt@foxmail.com
# Date: 2016/1/15
# Time: 19:20
#
import os
import cv2
import numpy as np


def get_traffic_light(img: np.ndarray) -> np.ndarray:
    """
    get traffic light from image
    Args:
        img: cv2 image
    Returns:
        cimg: cv2 image with traffic light
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    cimg = img.copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # color range
    maskr = cv2.add(
        cv2.inRange(
            hsv,
            np.array([0, 100, 100]),  # lower bound
            np.array([10, 255, 255]),  # upper bound
        ),
        cv2.inRange(
            hsv,
            np.array([160, 100, 100]),  # lower bound
            np.array([180, 255, 255]),  # upper bound
        ),
    )
    maskg = cv2.inRange(
        hsv,
        np.array([40, 50, 50]),  # lower bound
        np.array([90, 255, 255]),  # upper bound
    )
    masky = cv2.inRange(
        hsv,
        np.array([15, 150, 150]),  # lower bound
        np.array([35, 255, 255]),  # upper bound
    )

    size = img.shape
    # print size

    # hough circle detect
    r_circles = cv2.HoughCircles(
        maskr,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=80,
        param1=50,
        param2=10,
        minRadius=0,
        maxRadius=30,
    )

    g_circles = cv2.HoughCircles(
        maskg,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=60,
        param1=50,
        param2=10,
        minRadius=0,
        maxRadius=30,
    )

    y_circles = cv2.HoughCircles(
        masky,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=30,
        param1=50,
        param2=5,
        minRadius=0,
        maxRadius=30,
    )

    # traffic light detect
    r = 10
    bound = 1
    if r_circles is not None:
        r_circles = np.uint16(np.around(r_circles))

        for i in r_circles[0, :]:
            if i[0] > size[1] or i[1] > size[0] or i[1] > size[0] * bound:
                continue

            h, s = 0.0, 0.0
            for m in range(-r, r):
                for n in range(-r, r):
                    if (i[1] + m) >= size[0] or (i[0] + n) >= size[1]:
                        continue
                    h += maskr[i[1] + m, i[0] + n]
                    s += 1
            if h / s > 50:
                cv2.circle(cimg, (i[0], i[1]), i[2] + 10, (255, 255, 255), 2)
                cv2.putText(
                    cimg, "RED", (i[0], i[1]), font, 1, (0, 0, 255), 2, cv2.LINE_AA
                )

    if g_circles is not None:
        g_circles = np.uint16(np.around(g_circles))

        for i in g_circles[0, :]:
            if i[0] > size[1] or i[1] > size[0] or i[1] > size[0] * bound:
                continue

            h, s = 0.0, 0.0
            for m in range(-r, r):
                for n in range(-r, r):
                    if (i[1] + m) >= size[0] or (i[0] + n) >= size[1]:
                        continue
                    h += maskg[i[1] + m, i[0] + n]
                    s += 1
            if h / s > 100:
                cv2.circle(cimg, (i[0], i[1]), i[2] + 10, (255, 255, 255), 2)
                cv2.putText(
                    cimg, "GREEN", (i[0], i[1]), font, 1, (0, 255, 0), 2, cv2.LINE_AA
                )

    if y_circles is not None:
        y_circles = np.uint16(np.around(y_circles))

        for i in y_circles[0, :]:
            if i[0] > size[1] or i[1] > size[0] or i[1] > size[0] * bound:
                continue

            h, s = 0.0, 0.0
            for m in range(-r, r):
                for n in range(-r, r):
                    if (i[1] + m) >= size[0] or (i[0] + n) >= size[1]:
                        continue
                    h += masky[i[1] + m, i[0] + n]
                    s += 1
            if h / s > 50:
                cv2.circle(cimg, (i[0], i[1]), i[2] + 10, (255, 255, 255), 2)
                cv2.putText(
                    cimg, "YELLOW", (i[0], i[1]), font, 1, (0, 255, 255), 2, cv2.LINE_AA
                )

    # cv2.imshow("maskr", maskr)
    # cv2.imshow("maskg", maskg)
    # cv2.imshow("masky", masky)

    return cimg


if __name__ == "__main__":
    from time import time

    # path = os.path.abspath("..") + "/light/"
    #
    # for f in os.listdir(path):
    #     print(f)
    #     if (
    #         f.endswith(".jpg")
    #         or f.endswith(".JPG")
    #         or f.endswith(".png")
    #         or f.endswith(".PNG")
    #     ):
    #         img = cv2.imread(path + f)
    #         start = time()
    #         cimg = get_traffic_light(img)
    #         end = time()
    #         img_h, img_w, _ = img.shape
    #         print(f"Image Size: {(img_w, img_h)}, time: {(end - start) * 1000:.2f}ms")
    #         cv2.imshow("img", img)
    #         cv2.imshow("cimg", cimg)
    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows()

    paths = os.listdir("../vids/")
    for i, p in enumerate(paths):
        print(f"[{i}] {p}")
    path = paths[int(input("Choose a video: "))]

    cap = cv2.VideoCapture("../vids/" + path)
    r = cv2.selectROI("ROI", cap.read()[1])
    cv2.destroyWindow("ROI")

    times = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        start = time()
        cimg = get_traffic_light(frame[r[1] : r[1] + r[3], r[0] : r[0] + r[2]])
        end = time()
        times.append(end - start)
        cv2.imshow("img", frame)
        cv2.imshow("cimg", cimg)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    print(f"Average time: {sum(times) / len(times) * 1000:.2f}ms")
    print(f"Max time: {max(times) * 1000:.2f}ms")
    print(f"Min time: {min(times) * 1000:.2f}ms")
    print(f"ROI size: {r[3]}x{r[2]}")
