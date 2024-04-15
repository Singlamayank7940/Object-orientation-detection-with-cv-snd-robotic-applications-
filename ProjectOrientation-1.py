import cv2 as cv
from math import atan2, cos, sin, sqrt, pi
import numpy as np
from pymata4 import pymata4
from time import sleep

def drawAxis(img, p1, q1, color, scale):
  p = list(p1)
  q = list(q1)
 
  ## Visualization start
  angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
  hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
 
  # Here we lengthen the arrow by a factor of scale
  q[0] = p[0] - scale * hypotenuse * cos(angle)
  q[1] = p[1] - scale * hypotenuse * sin(angle)
  cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv.LINE_AA)
 
  # create the arrow hooks
  p[0] = q[0] + 9 * cos(angle + pi / 4)
  p[1] = q[1] + 9 * sin(angle + pi / 4)
  cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv.LINE_AA)
 
  p[0] = q[0] + 9 * cos(angle - pi / 4)
  p[1] = q[1] + 9 * sin(angle - pi / 4)
  cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv.LINE_AA)
  ## Visualization1 end
 
def getOrientation(pts, img):
  ## PCA start
  # Construct a buffer used by the pca analysis
  sz = len(pts)
  data_pts = np.empty((sz, 2), dtype=np.float64)
  for i in range(data_pts.shape[0]):
    data_pts[i,0] = pts[i,0,0]
    data_pts[i,1] = pts[i,0,1]
 
  # Perform PCA analysis
  mean = np.empty((0))
  mean, eigenvectors, eigenvalues = cv.PCACompute2(data_pts, mean)
 
  # Store the center of the object
  cntr = (int(mean[0,0]), int(mean[0,1]))
  ## PCA end
 
  ## [visualization]
  # Draw Arrow
  cv.circle(img, cntr, 3, (0, 0, 255), 2)
  pt1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
  drawAxis(img, cntr, pt1, (255, 255, 0), 2.5) # 0.5 is the scale factor which may be increased or decreased
 
  angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
  ## [visualization]
 
  # Label with the rotation angle
  label = "  Rotation Angle: " + str(-int(np.rad2deg(angle)) - 0) + " degrees" # - 90
  textbox = cv.rectangle(img, (cntr[0], cntr[1]-25), (cntr[0] + 250, cntr[1] + 10), (255,255,255), -1)
  cv.putText(img, label, (cntr[0], cntr[1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv.LINE_AA)
 
  return angle
 

### Main program start

# Connecting to the board
stepsPerRev=2038
pins=[2,3,4,5]
board=pymata4.Pymata4()
board.set_pin_mode_stepper(stepsPerRev,pins)

ang=0

cap = cv.VideoCapture(1)
if not cap.isOpened():
  print("Cannot open camera")
  exit()

while True:
  ret, img = cap.read()
  # if frame is read correctly, ret is True
  if not ret:
    print("Can't receive frame (stream end?). Exiting ...")
    exit()
  cv.imshow("Component",img)
  if cv.waitKey(1) == ord('y'):
    cv.imwrite("component.jpg",img)
    break
    
# Convert image to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# Convert image to binary
ret, bw = cv.threshold(gray, 50, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
# Find all the contours in the thresholded image
contours, ret = cv.findContours(bw, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

for i, c in enumerate(contours):
  # Calculate the area of each contour
  area = cv.contourArea(c)
  # Ignore contours that are too small or too large
  if area < 3700 or 100000 < area:
    continue
  # Draw each contour
  cv.drawContours(img, contours, i, (0, 0, 255), 2)
  # Find the orientation of each shape
  ang=getOrientation(c, img)
  ang=-int(np.rad2deg(ang))
  print(ang)

cv.imshow('Component Orientation', img)
# Save the output image to the current directory
cv.imwrite("component_orientation.jpg", img)

if abs(ang) <= 3:
  print("Orientation OK")
elif abs(ang) <= 10:
  print("Orientation within limits, but large")
else:
  print("Orientation out of range, rotating gripper, please wait...")
  stepsToRun=int(stepsPerRev*ang/360)
  board.stepper_write(20,stepsToRun)
  
cv.waitKey(0)
cap.release()
cv.destroyAllWindows()
