from tkinter import *
from PIL import Image, ImageTk
import constants, utils
import numpy as np
import cv2
import glob
import time

def thresh_channels(output, channels, use_channels, thresholds):
    thresholds = np.array(thresholds).reshape(-1,2).tolist()
    for c, uc, t in zip(channels, use_channels, thresholds):
        if uc:
            timg = np.zeros_like(output)
            timg[(c>=t[0]) & (c<=t[1])]=1
            output = output | timg
    return output

def sobel_thresh_channels(output, channels, use_channels, thresholds, sobel_kernel=3):
    thresholds = np.array(thresholds).reshape(-1,2).tolist()
    for c, uc, t in zip(channels, use_channels, thresholds):
        if uc:
            timg = np.zeros_like(output)
            sobelx = cv2.Sobel(c, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
            sobely = cv2.Sobel(c, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
            mag_sobel = np.sqrt(np.square(sobelx)+np.square(sobely))
            scaled_sobel = np.uint8(255*mag_sobel/np.max(mag_sobel))
            timg[(scaled_sobel >= t[0]) & (scaled_sobel <= t[1])] = 1
            output = output | timg
    return output

def thresh_image(img, use_rgb=None, rgb_thresh=None, use_hls=None, hls_thresh=None, use_luv=None, luv_thresh=None, use_sobel_hls=None, sobel_hls_thresh=None, sobel_hls_kernel=None):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    output = np.zeros_like(gray)

    if (use_rgb is not None) and (True in use_rgb):
        b,g,r = cv2.split(img)
        output = thresh_channels(output, (r,g,b), use_rgb, rgb_thresh)

    if (use_hls is not None) and (True in use_hls):
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        h,l,s = cv2.split(hls)
        output = thresh_channels(output, (h,l,s), use_hls, hls_thresh)

    if (use_luv is not None) and (True in use_luv):
        luv = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
        l,u,v = cv2.split(luv)
        output = thresh_channels(output, (l,u,v), use_luv, luv_thresh)

    if (use_sobel_hls is not None) and (True in use_sobel_hls):
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        h,l,s = cv2.split(hls)
        output = sobel_thresh_channels(output, (h,l,s), use_sobel_hls, sobel_hls_thresh, sobel_hls_kernel)

    return output

def thresh_folder(folder_name, use_rgb=None, rgb_thresh=None, use_hls=None, hls_thresh=None, use_luv=None, luv_thresh=None, use_sobel_hls=None, sobel_hls_thresh=None, sobel_hls_kernel=None, output_folder=None):
    if output_folder:
        utils.make_dir(output_folder)
    image_files = glob.glob('{}/*.jpg'.format(folder_name)) + glob.glob('{}/*.png'.format(folder_name))
    for i in range(len(image_files)):
        utils.print_progress_bar(i, len(image_files), prefix = 'thresh {}:'.format(folder_name))
        img = cv2.imread(image_files[i])
        img = thresh_image(img, use_rgb=use_rgb, rgb_thresh=rgb_thresh,
                           use_hls=use_hls, hls_thresh=hls_thresh,
                           use_luv=use_luv, luv_thresh=luv_thresh,
                           use_sobel_hls=use_sobel_hls, sobel_hls_thresh=sobel_hls_thresh,
                           sobel_hls_kernel=sobel_hls_kernel)
        img *= 255
        if output_folder:
            out_name = '{}/{}'.format(output_folder, utils.basename(image_files[i]))
            cv2.imwrite(out_name, img)
        else:
            cv2.imshow('img', img)
            cv2.waitKey(500)
    utils.print_progress_bar(len(image_files), len(image_files), prefix = 'thresh {}:'.format(folder_name))

class AdjustThresh:
    def __init__(self, images, config=None):
        self.master = Tk()
        self.images = images
        self.config = config
        self.init_values()
        self.img_index = 0
        self.accepted = False       # dismissed with cancel or ok?
        self.show()

    def init_values(self):
        self.use_rgb = [False, False, False]
        self.use_hls = [False, False, False]
        self.use_luv = [False, False, False]
        self.use_sobel_hls = [False, False, False]
        self.rgb_thresh = [0,255,0,255,0,255]
        self.hls_thresh = [0,255,0,255,0,255]
        self.luv_thresh = [0,255,0,255,0,255]
        self.sobel_hls_thresh = [0,255,0,255,0,255]
        self.sobel_hls_kernel = 3
        if self.config is not None:
            if 'thresh_use_rgb' in self.config:
                self.use_rgb = self.config['thresh_use_rgb']
            if 'thresh_use_hls' in self.config:
                self.use_hls = self.config['thresh_use_hls']
            if 'thresh_use_luv' in self.config:
                self.use_luv = self.config['thresh_use_luv']
            if 'thresh_use_sobel_hls' in self.config:
                self.use_sobel_hls = self.config['thresh_use_sobel_hls']
            if 'thresh_rgb_thresh' in self.config:
                self.rgb_thresh = self.config['thresh_rgb_thresh']
            if 'thresh_hls_thresh' in self.config:
                self.hls_thresh = self.config['thresh_hls_thresh']
            if 'thresh_luv_thresh' in self.config:
                self.luv_thresh = self.config['thresh_luv_thresh']
            if 'thresh_sobel_hls_thresh' in self.config:
                self.sobel_hls_thresh = self.config['thresh_sobel_hls_thresh']
            if 'thresh_sobel_hls_kernel' in self.config:
                self.sobel_hls_kernel = self.config['thresh_sobel_hls_kernel']

    def redraw(self):
        img1 = self.images[self.img_index].copy()
        w=img1.shape[1]
        h=img1.shape[0]

        img1 = Image.fromarray(img1[:,:,[2,1,0]]).resize((int(w/2), int(h/2)))
        self.imgtk1 = ImageTk.PhotoImage(image=img1)
        self.canvas1.create_image(0,0, image=self.imgtk1, anchor=NW)

        img2 = self.images[self.img_index].copy()
        img2 = thresh_image(img2, use_rgb=self.use_rgb, rgb_thresh=self.rgb_thresh,
                            use_hls=self.use_hls, hls_thresh=self.hls_thresh,
                            use_luv=self.use_luv, luv_thresh=self.luv_thresh,
                            use_sobel_hls=self.use_sobel_hls, sobel_hls_thresh=self.sobel_hls_thresh,
                            sobel_hls_kernel=self.sobel_hls_kernel)
        img2 *= 255
        img2 = Image.fromarray(img2).resize((int(w/2), int(h/2)))
        self.imgtk2 = ImageTk.PhotoImage(image=img2)
        self.canvas2.create_image(0,0, image=self.imgtk2, anchor=NW)

    def update(self,v=None):
        self.img_index = self.s_image_index.get()
        self.use_rgb = self.get_checkbuttons(self.v_use_rgb)
        self.use_hls = self.get_checkbuttons(self.v_use_hls)
        self.use_luv = self.get_checkbuttons(self.v_use_luv)
        self.use_sobel_hls = self.get_checkbuttons(self.v_use_sobel_hls)
        self.rgb_thresh = self.get_thresh_sliders(self.s_rgb)
        self.hls_thresh = self.get_thresh_sliders(self.s_hls)
        self.luv_thresh = self.get_thresh_sliders(self.s_luv)
        self.sobel_hls_thresh = self.get_thresh_sliders(self.s_sobel_hls)
        self.sobel_hls_kernel = self.s_sobel_hls_kernel.get()*2+1
        status = 'RGB: '+str(self.rgb_thresh).replace('[','').replace(']','')
        status += ' - '
        status += 'HLS: '+str(self.hls_thresh).replace('[','').replace(']','')
        status += ' - '
        status += 'Luv: '+str(self.luv_thresh).replace('[','').replace(']','')
        status += ' - '
        status += 'sob: '+str(self.sobel_hls_thresh).replace('[','').replace(']','')
        self.v_status.set(status)
        self.redraw()
        pass

    def accept(self):
        self.accepted = True
        self.config['thresh_use_rgb'] = self.use_rgb
        self.config['thresh_use_hls'] = self.use_hls
        self.config['thresh_use_luv'] =self.use_luv
        self.config['thresh_use_sobel_hls'] = self.use_sobel_hls
        self.config['thresh_rgb_thresh'] = self.rgb_thresh
        self.config['thresh_hls_thresh'] = self.hls_thresh
        self.config['thresh_luv_thresh'] = self.luv_thresh
        self.config['thresh_sobel_hls_thresh'] = self.sobel_hls_thresh
        self.config['thresh_sobel_hls_kernel'] = self.sobel_hls_kernel
        self.master.destroy()

    def set_checkbuttons(self, button_vars, values):
        for b, v in zip(button_vars,values):
            if v:
                b.set(1)
            else:
                b.set(0)

    def get_checkbuttons(self, button_vars):
        values = []
        for b in button_vars:
            if b.get():
                values.append(True)
            else:
                values.append(False)
        return values

    def set_thresh_sliders(self, sliders, values):
        for s,v in zip(sliders,values):
            s.set(v)

    def get_thresh_sliders(self, sliders):
        values = []
        for s in sliders:
            values.append(s.get())
        return values

    def show(self):
        w=constants.image_width
        h=constants.image_height
        self.canvas1 = Canvas(self.master, width=w/2, height=h/2)
        self.canvas1.grid(row=0,column=0,sticky=W)
        self.canvas2 = Canvas(self.master, width=w/2, height=h/2)
        self.canvas2.grid(row=0,column=1,sticky=W)
        self.s_image_index = Scale(self.master, label='image', from_=0, to=len(self.images)-1, length=200, command=self.update, orient=HORIZONTAL)
        self.s_image_index.grid(row=1, column=0,sticky=W)

        self.v_use_rgb_r = IntVar()
        self.c_use_rgb_r = Checkbutton(self.master, text="R channel (RGB)", variable=self.v_use_rgb_r, command=self.update)
        self.c_use_rgb_r.grid(row=2,column=0,sticky=W)
        self.s_rgb_r_min = Scale(self.master, from_=0, to=255, length=255, command=self.update, orient=HORIZONTAL, showvalue=False)
        self.s_rgb_r_min.grid(row=3,column=0,sticky=W)
        self.s_rgb_r_max = Scale(self.master, from_=0, to=255, length=255, command=self.update, orient=HORIZONTAL, showvalue=False)
        self.s_rgb_r_max.grid(row=3,column=1,sticky=W)

        self.v_use_rgb_g = IntVar()
        self.c_use_rgb_g = Checkbutton(self.master, text="G channel (RGB)", variable=self.v_use_rgb_g, command=self.update)
        self.c_use_rgb_g.grid(row=4,column=0,sticky=W)
        self.s_rgb_g_min = Scale(self.master, from_=0, to=255, length=255, command=self.update, orient=HORIZONTAL, showvalue=False)
        self.s_rgb_g_min.grid(row=5,column=0,sticky=W)
        self.s_rgb_g_max = Scale(self.master, from_=0, to=255, length=255, command=self.update, orient=HORIZONTAL, showvalue=False)
        self.s_rgb_g_max.grid(row=5,column=1,sticky=W)

        self.v_use_rgb_b = IntVar()
        self.c_use_rgb_b = Checkbutton(self.master, text="B channel (RGB)", variable=self.v_use_rgb_b, command=self.update)
        self.c_use_rgb_b.grid(row=6,column=0,sticky=W)
        self.s_rgb_b_min = Scale(self.master, from_=0, to=255, length=255, command=self.update, orient=HORIZONTAL, showvalue=False)
        self.s_rgb_b_min.grid(row=7,column=0,sticky=W)
        self.s_rgb_b_max = Scale(self.master, from_=0, to=255, length=255, command=self.update, orient=HORIZONTAL, showvalue=False)
        self.s_rgb_b_max.grid(row=7,column=1,sticky=W)

        self.v_use_hls_h = IntVar()
        self.c_use_hls_h = Checkbutton(self.master, text="H channel (HLS)", variable=self.v_use_hls_h, command=self.update)
        self.c_use_hls_h.grid(row=8,column=0,sticky=W)
        self.s_hls_h_min = Scale(self.master, from_=0, to=255, length=255, command=self.update, orient=HORIZONTAL, showvalue=False)
        self.s_hls_h_min.grid(row=9,column=0,sticky=W)
        self.s_hls_h_max = Scale(self.master, from_=0, to=255, length=255, command=self.update, orient=HORIZONTAL, showvalue=False)
        self.s_hls_h_max.grid(row=9,column=1,sticky=W)

        self.v_use_hls_l = IntVar()
        self.c_use_hls_l = Checkbutton(self.master, text="L channel (HLS)", variable=self.v_use_hls_l, command=self.update)
        self.c_use_hls_l.grid(row=10,column=0,sticky=W)
        self.s_hls_l_min = Scale(self.master, from_=0, to=255, length=255, command=self.update, orient=HORIZONTAL, showvalue=False)
        self.s_hls_l_min.grid(row=11,column=0,sticky=W)
        self.s_hls_l_max = Scale(self.master, from_=0, to=255, length=255, command=self.update, orient=HORIZONTAL, showvalue=False)
        self.s_hls_l_max.grid(row=11,column=1,sticky=W)

        self.v_use_hls_s = IntVar()
        self.c_use_hls_s = Checkbutton(self.master, text="S channel (HLS)", variable=self.v_use_hls_s, command=self.update)
        self.c_use_hls_s.grid(row=12,column=0,sticky=W)
        self.s_hls_s_min = Scale(self.master, from_=0, to=255, length=255, command=self.update, orient=HORIZONTAL, showvalue=False)
        self.s_hls_s_min.grid(row=13,column=0,sticky=W)
        self.s_hls_s_max = Scale(self.master, from_=0, to=255, length=255, command=self.update, orient=HORIZONTAL, showvalue=False)
        self.s_hls_s_max.grid(row=13,column=1,sticky=W)

        self.v_use_luv_l = IntVar()
        self.c_use_luv_l = Checkbutton(self.master, text="L channel (Luv)", variable=self.v_use_luv_l, command=self.update)
        self.c_use_luv_l.grid(row=14,column=0,sticky=W)
        self.s_luv_l_min = Scale(self.master, from_=0, to=255, length=255, command=self.update, orient=HORIZONTAL, showvalue=False)
        self.s_luv_l_min.grid(row=15,column=0,sticky=W)
        self.s_luv_l_max = Scale(self.master, from_=0, to=255, length=255, command=self.update, orient=HORIZONTAL, showvalue=False)
        self.s_luv_l_max.grid(row=15,column=1,sticky=W)

        self.v_use_luv_u = IntVar()
        self.c_use_luv_u = Checkbutton(self.master, text="u channel (Luv)", variable=self.v_use_luv_u, command=self.update)
        self.c_use_luv_u.grid(row=16,column=0,sticky=W)
        self.s_luv_u_min = Scale(self.master, from_=0, to=255, length=255, command=self.update, orient=HORIZONTAL, showvalue=False)
        self.s_luv_u_min.grid(row=17,column=0,sticky=W)
        self.s_luv_u_max = Scale(self.master, from_=0, to=255, length=255, command=self.update, orient=HORIZONTAL, showvalue=False)
        self.s_luv_u_max.grid(row=17,column=1,sticky=W)

        self.v_use_luv_v = IntVar()
        self.c_use_luv_v = Checkbutton(self.master, text="v channel (Luv)", variable=self.v_use_luv_v, command=self.update)
        self.c_use_luv_v.grid(row=18,column=0,sticky=W)
        self.s_luv_v_min = Scale(self.master, from_=0, to=255, length=255, command=self.update, orient=HORIZONTAL, showvalue=False)
        self.s_luv_v_min.grid(row=19,column=0,sticky=W)
        self.s_luv_v_max = Scale(self.master, from_=0, to=255, length=255, command=self.update, orient=HORIZONTAL, showvalue=False)
        self.s_luv_v_max.grid(row=19,column=1,sticky=W)

        self.v_use_sobel_hls_h = IntVar()
        self.c_use_sobel_hls_h = Checkbutton(self.master, text="sobel H channel (HLS)", variable=self.v_use_sobel_hls_h, command=self.update)
        self.c_use_sobel_hls_h.grid(row=20,column=0,sticky=W)
        self.s_sobel_hls_h_min = Scale(self.master, from_=0, to=255, length=255, command=self.update, orient=HORIZONTAL, showvalue=False)
        self.s_sobel_hls_h_min.grid(row=21,column=0,sticky=W)
        self.s_sobel_hls_h_max = Scale(self.master, from_=0, to=255, length=255, command=self.update, orient=HORIZONTAL, showvalue=False)
        self.s_sobel_hls_h_max.grid(row=21,column=1,sticky=W)

        self.v_use_sobel_hls_l = IntVar()
        self.c_use_sobel_hls_l = Checkbutton(self.master, text="sobel L channel (HLS)", variable=self.v_use_sobel_hls_l, command=self.update)
        self.c_use_sobel_hls_l.grid(row=22,column=0,sticky=W)
        self.s_sobel_hls_l_min = Scale(self.master, from_=0, to=255, length=255, command=self.update, orient=HORIZONTAL, showvalue=False)
        self.s_sobel_hls_l_min.grid(row=23,column=0,sticky=W)
        self.s_sobel_hls_l_max = Scale(self.master, from_=0, to=255, length=255, command=self.update, orient=HORIZONTAL, showvalue=False)
        self.s_sobel_hls_l_max.grid(row=23,column=1,sticky=W)

        self.v_use_sobel_hls_s = IntVar()
        self.c_use_sobel_hls_s = Checkbutton(self.master, text="sobel S channel (HLS)", variable=self.v_use_sobel_hls_s, command=self.update)
        self.c_use_sobel_hls_s.grid(row=24,column=0,sticky=W)
        self.s_sobel_hls_s_min = Scale(self.master, from_=0, to=255, length=255, command=self.update, orient=HORIZONTAL, showvalue=False)
        self.s_sobel_hls_s_min.grid(row=25,column=0,sticky=W)
        self.s_sobel_hls_s_max = Scale(self.master, from_=0, to=255, length=255, command=self.update, orient=HORIZONTAL, showvalue=False)
        self.s_sobel_hls_s_max.grid(row=25,column=1,sticky=W)

        self.s_sobel_hls_kernel = Scale(self.master, label="sobel kernel (v*2+1)", from_=0, to=15, length=255, command=self.update, orient=HORIZONTAL)
        self.s_sobel_hls_kernel.grid(row=26,column=0,sticky=W)

        self.v_use_rgb = [self.v_use_rgb_r, self.v_use_rgb_g, self.v_use_rgb_b]
        self.v_use_hls = [self.v_use_hls_h, self.v_use_hls_l, self.v_use_hls_s]
        self.v_use_luv = [self.v_use_luv_l, self.v_use_luv_u, self.v_use_luv_v]
        self.v_use_sobel_hls = [self.v_use_sobel_hls_h, self.v_use_sobel_hls_l, self.v_use_sobel_hls_s]

        self.set_checkbuttons(self.v_use_rgb, self.use_rgb)
        self.set_checkbuttons(self.v_use_hls, self.use_hls)
        self.set_checkbuttons(self.v_use_luv, self.use_luv)
        self.set_checkbuttons(self.v_use_sobel_hls, self.use_sobel_hls)


        self.s_rgb = [self.s_rgb_r_min, self.s_rgb_r_max,
                      self.s_rgb_g_min, self.s_rgb_g_max,
                      self.s_rgb_b_min, self.s_rgb_b_max]

        self.s_hls = [self.s_hls_h_min, self.s_hls_h_max,
                      self.s_hls_l_min, self.s_hls_l_max,
                      self.s_hls_s_min, self.s_hls_s_max]

        self.s_luv = [self.s_luv_l_min, self.s_luv_l_max,
                      self.s_luv_u_min, self.s_luv_u_max,
                      self.s_luv_v_min, self.s_luv_v_max]

        self.s_sobel_hls = [self.s_sobel_hls_h_min, self.s_sobel_hls_h_max,
                            self.s_sobel_hls_l_min, self.s_sobel_hls_l_max,
                            self.s_sobel_hls_s_min, self.s_sobel_hls_s_max]

        self.set_thresh_sliders(self.s_rgb, self.rgb_thresh)
        self.set_thresh_sliders(self.s_hls, self.hls_thresh)
        self.set_thresh_sliders(self.s_luv, self.luv_thresh)
        self.set_thresh_sliders(self.s_sobel_hls, self.sobel_hls_thresh)
        self.s_sobel_hls_kernel.set(int((self.sobel_hls_kernel-1)/2))

        self.v_status = StringVar()
        self.l_status = Label(self.master, textvariable=self.v_status)
        self.l_status.grid(row=99,sticky=W)

        Button(self.master, text="cancel", command=self.master.destroy).grid(row=100, column=0,sticky=W)
        Button(self.master, text="ok", command=self.accept).grid(row=100, column=1, sticky=W)
        self.redraw()
        mainloop()

def adjust_thresh(image_folder, config=None):
    images = utils.read_folder_images(image_folder)
    adjust_thresh = AdjustThresh(images, config)
    return (adjust_thresh.accepted, adjust_thresh.config)

if __name__=='__main__':
    images = utils.read_folder_images(constants.persp_trans_test_folder)
    AdjustThresh(images)
