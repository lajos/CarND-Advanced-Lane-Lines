
import cv2
import constants, utils, camcalib
import numpy as np



_ = {} # global data


def calibrate_camera(force=False):
    global _
    if force or not ('mtx' in _ and 'dist' in _):
        mtx, dist = camcalib.calibrate_folder(constants.camera_cal_folder, nx=9, ny=6)
        camcalib.test_folder(constants.camera_cal_folder, mtx, dist, constants.camera_cal_test_folder)
        _['mtx']=mtx
        _['dist']=dist


def main():
    global _
    _ = utils.load_globals()
    #print(_)

    calibrate_camera()

    #utils.display_image_file('test_images/straight_lines1.jpg')
    #utils.plt_image_file('test_images/straight_lines1.jpg', bgr=True)

    utils.save_globals(_)

if __name__=='__main__':
    main()
