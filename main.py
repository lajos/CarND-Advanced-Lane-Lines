
import cv2
import constants, utils, camcalib, persptrans
import numpy as np



_ = {} # global data


def calibrate_camera(force=False):
    global _
    if force or not ('cal_mtx' in _ and 'cal_dist' in _):
        mtx, dist = camcalib.calibrate_folder(constants.camera_cal_folder, nx=9, ny=6)
        _['cal_mtx']=mtx
        _['cal_dist']=dist
        camcalib.undistort_folder(constants.camera_cal_folder, mtx, dist, constants.camera_cal_test_folder)
        camcalib.undistort_folder(constants.test_images_folder, mtx, dist, constants.undistort_test_folder)

def adjust_perspective(force=False):
    global _
    if force or not ('persp_src_pts' in _ and 'persp_dst_pts' in _):
        src_pts = None
        dst_pts = None
        if 'persp_src_pts' in _:
            src_pts = _['persp_src_pts']
        if 'persp_dst_pts' in _:
            dst_pts = _['persp_dst_pts']
        ret, src_pts, dst_pts = persptrans.adjust_perspective(constants.test_images_folder, src_pts, dst_pts)
        if ret or not ('persp_src_pts' in _ and 'persp_dst_pts' in _):
            _['persp_src_pts'] = src_pts
            _['persp_dst_pts'] = dst_pts
        persptrans.warp_folder(constants.undistort_test_folder, _['persp_src_pts'], _['persp_dst_pts'], constants.persp_trans_test_folder)


def main():
    global _
    _ = utils.load_globals()
    #print(_)

    calibrate_camera(force=False)
    adjust_perspective(force=False)

    #utils.display_image_file('test_images/straight_lines1.jpg')
    #utils.plt_image_file('test_images/straight_lines1.jpg', bgr=True)

    utils.save_globals(_)

if __name__=='__main__':
    main()
