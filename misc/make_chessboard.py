
import numpy as np
import cv2

block_size = 200
blocks_x = 10
blocks_y = 7

img = np.zeros((blocks_y*block_size, blocks_x*block_size), dtype=np.uint8)

for x in range(1,blocks_x,2):
    for y in range(0,blocks_y):
        xx=x
        if y%2==0:
            xx -= 1
        img[y*block_size:(y+1)*block_size, xx*block_size:(xx+1)*block_size]=255

cv2.imshow('chessboard',img)
cv2.waitKey(5000)
cv2.imwrite('chessboard.png', img)