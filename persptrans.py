from tkinter import *
from PIL import Image, ImageTk
import constants, utils
import numpy as np
import cv2

def warp_image(img, src_pts, dst_pts):
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

class AdjustTransform:
    def __init__(self, images):
        self.master = Tk()
        self.images = images
        self.img_index = 0
        self.src_top_width = 62
        self.src_top_pos = 445
        self.src_bottom_width = 545
        self.src_bottom_pos = 651
        self.show()

    def redraw(self):
        img1 = images[self.img_index].copy()
        w=img1.shape[1]
        h=img1.shape[0]

        src_pts = self.get_src_points()
        dst_pts = self.get_dst_points()

        utils.img_draw_poly(img1, src_pts[None,:], color=(0,0,255), thickness=3)

        img1 = Image.fromarray(img1[:,:,[2,1,0]]).resize((int(w/2), int(h/2)))
        self.imgtk1 = ImageTk.PhotoImage(image=img1)
        self.canvas1.create_image(0,0, image=self.imgtk1, anchor=NW)

        img2 = warp_image(images[self.img_index], np.float32(src_pts), np.float32(dst_pts))
        utils.img_draw_grid(img2, color=(128,255,128))
        img2 = Image.fromarray(img2[:,:,[2,1,0]]).resize((int(w/2), int(h/2)))
        self.imgtk2 = ImageTk.PhotoImage(image=img2)
        self.canvas2.create_image(0,0, image=self.imgtk2, anchor=NW)

    def update(self,v=None):
        self.img_index = self.s_image_index.get()
        self.src_top_width = self.s_src_top_width.get()
        self.src_top_pos = self.s_src_top_pos.get()
        self.src_bottom_width = self.s_src_bottom_width.get()
        self.src_bottom_pos = self.s_src_bottom_pos.get()
        self.redraw()

    def get_src_points(self):
        w=constants.image_width
        h=constants.image_height
        w2=int(w/2)
        return np.array([[w2-self.src_top_width, self.src_top_pos],
            [w2+self.src_top_width, self.src_top_pos],
            [w2+self.src_bottom_width, self.src_bottom_pos],
            [w2-self.src_bottom_width, self.src_bottom_pos]], dtype=np.int32)

    def get_dst_points(self):
        w=constants.image_width
        h=constants.image_height
        w2=int(w/2)
        return np.array([[w2-self.src_bottom_width, 0],
            [w2+self.src_bottom_width, 0],
            [w2+self.src_bottom_width, h],
            [w2-self.src_bottom_width, h],], dtype=np.int32)

    def show(self):
        w=constants.image_width
        h=constants.image_height
        self.canvas1 = Canvas(self.master, width=w/2, height=h/2)
        self.canvas1.grid(row=0,column=0,sticky=W)
        self.canvas2 = Canvas(self.master, width=w/2, height=h/2)
        self.canvas2.grid(row=0,column=1,sticky=W)
        self.s_image_index = Scale(self.master, label='image', from_=0, to=len(self.images)-1, length=200, command=self.update, orient=HORIZONTAL)
        self.s_image_index.grid(row=1, column=0,sticky=W)
        self.s_src_top_width = Scale(self.master, label='top width', from_=0, to=500, length=500, command=self.update, orient=HORIZONTAL)
        self.s_src_top_width.grid(row=2, column=0,sticky=W)
        self.s_src_top_pos = Scale(self.master, label='top y position', from_=0, to=500, length=500, command=self.update, orient=HORIZONTAL)
        self.s_src_top_pos.grid(row=3, column=0,sticky=W)
        self.s_src_bottom_width = Scale(self.master, label='bottom width', from_=300, to=800, length=500, command=self.update, orient=HORIZONTAL)
        self.s_src_bottom_width.grid(row=4, column=0,sticky=W)
        self.s_src_bottom_pos = Scale(self.master, label='pottom y position', from_=300, to=800, length=500, command=self.update, orient=HORIZONTAL)
        self.s_src_bottom_pos.grid(row=5, column=0,sticky=W)

        self.s_image_index.set(self.img_index)
        self.s_src_top_width.set(self.src_top_width)
        self.s_src_top_pos.set(self.src_top_pos)
        self.s_src_bottom_width.set(self.src_bottom_width)
        self.s_src_bottom_pos.set(self.src_bottom_pos)

        Button(self.master, text="close", command=self.master.destroy).grid(row=6,sticky=W)
        self.redraw()
        mainloop()

if __name__=='__main__':
    images = utils.read_folder_images(constants.test_images_folder)
    AdjustTransform(images)
