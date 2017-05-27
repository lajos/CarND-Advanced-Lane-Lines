import numpy as np
import cv2

image_width = 1280
image_height = 720

stripe_height = 10

img = np.zeros((image_height, image_width), dtype=np.uint8)

for y in range(0,image_height,stripe_height*2):
    img[y:y+stripe_height, 0:image_width]=255

cv2.imshow('stripes',img)
cv2.waitKey(5000)
cv2.imwrite('stripes.png', img)

