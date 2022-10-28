<p align="center">
  <img src="https://i.postimg.cc/gJKLSpRj/banner.png" />
</p>

Python module/script allowing a user to control any media player using hand gestures and face tracking.

This project is being made and maintained by the **Embedded Systems and Robotics Club, NIT Kurukshetra** (EmR) for its **Image processing workshop**. However, members outside the club are welcome to contribute (refer to [contribution guidelines](#contributions))

## Features

- [x] Hand gesture recognition and processing to allow user to play/pause, seek, change volume.
- [x] Face tracking for automatic play/pause based on whether user is viewing media/is present.
- [x] Functions to allow tuning of the image processing algorithms.
- [x] Use of auto keystroke and clicker modules to allow application to any media player.

## Installation and Usage
**Dependecies**

 - [opencv-python](https://pypi.org/project/opencv-python/) 
 - [mediapipe](https://pypi.org/project/mediapipe/)
 - [PyAutoGUI](https://pypi.org/project/PyAutoGUI/)
 - [Numpy](https://numpy.org/)

**Command to run the python script**

     python Gesture_control.py

## Project Explanation
The project working can be summarized into two parts:

 - **Hand Motion Tracking:** 
   - Mask of hand is created to detect the hand. 
     ```python
        lower = np.array([hue_min, sat_min, val_min])
        upper = np.array([hue_max, sat_max, val_max])
        mask = cv2.inRange(imgHSV, lower, upper)
     ```
   - Contours of mask is found out using findContours(). The contour with maximum area is selected as the contour of hand as other contours are due to noise.
     ```python
        contours, heirarchy = cv2.findContours(
           thresh, cv2.RETR_EXTERNAL,
           cv2.CHAIN_APPROX_SIMPLE) 
        max_cntr = max(contours, key=lambda x: cv2.contourArea(x))
        epsilon = 0.005 * cv2.arcLength(
             max_cntr, True
        )  # maximum distance from contour to approximated contour.
        max_cntr = cv2.approxPolyDP(max_cntr, epsilon, True)
     ```
   - Centroid of contour of hand is found out which will be used as a center of mass for tracking hand motion.
     ```python
        def centroid(contour):
           if len(contour) == 0:  # if the array is empty return (-1,-1)
              return (-1, -1)
           M = cv2.moments(
              contour)  # gives a dictionary of all moment values calculated
           try:
               x = int(
                M['m10'] / M['m00']
               )  # Centroid is given by the relations, 𝐶𝑥 =𝑀10/𝑀00 and 𝐶𝑦 =𝑀01/𝑀00
               y = int(M['m01'] / M['m00'])
           except ZeroDivisionError:
               return (-1, -1)
           return (x, y)
     ```
   - Hand motion is tracked by using the coordinates of centroid in different frames.
   
     Let say there are two frames: F1 and F2. The position of centroid in F1 is (x1,y1) and in F2 is (x2,y2). The time difference b/w F1 and F2 is t. Then          hand motion can be tracked as shown below:
     ```python
        def velocity(x1, x2, t):
            return (x2 - x1) / t

        def detect_motion(x1, y1, x2, y2, t):
            vel_x = velocity(x1, x2, t)
            vel_y = velocity(y1, y2, t)

            if vel_x > VEL_THRESHOLD:
                return MOTION_RIGHT
            elif vel_x < -VEL_THRESHOLD:
                return MOTION_LEFT
            elif vel_y > VEL_THRESHOLD:
                return MOTION_DOWN
            elif vel_y < -VEL_THRESHOLD:
                return MOTION_UP
            else:
                return NO_MOTION
     ```
   - Hand motion is mapped to keyboard's keys using [PyAutoGUI](https://pypi.org/project/PyAutoGUI/) for controlling the media player.
     ```python
        def control_media_player(hand_motion):
            if hand_motion == MOTION_RIGHT:
                pyautogui.press(KEY_FORWARD)
            elif hand_motion == MOTION_LEFT:
                pyautogui.press(KEY_BACKWARD)
            elif hand_motion == MOTION_UP:
                pyautogui.press(KEY_VOLUME_UP)
            elif hand_motion == MOTION_DOWN:
                pyautogui.press(KEY_VOLUME_DOWN)
     ```
     
 - **Eye Detection** 
 
     Eye detection is done using [mediapipe](https://pypi.org/project/mediapipe/) for tracking the user's eyes. If the user is not looking at the screen, then the media player pauses and when user looked back at the screen, it resumes.

     ```python
        eyes = EyeDetection.detectEyes(frame) > 0
        if eyes and not IS_VIDEO_PLAYING:
           play_video()
        elif not eyes and IS_VIDEO_PLAYING:
           pause_video()
     ```

## Demonstrations

Coming soon...

## Contributions

Members of EmR are requested to refer to the [M_CONTRIBUTING.md](./M_CONTRIBUTING.md) file before raising PRs or issues. Individuals outside of EmR must refer to the [CONTRIBUTING.md](./CONTRIBUTING.md) file before raising PRs or issues.
