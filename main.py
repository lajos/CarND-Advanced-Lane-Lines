
import cv2
import constants, utils, camcalib, persptrans, thresh, find_lanes
import numpy as np
import sys, pprint, time
import matplotlib.pyplot as pyplot

_ = {} # global data

def calibrate_camera(cal_folder, test_images_folder, force=False):
    """ calibrate camera, update global values and process calibration test images"""
    global _
    if force or not ('cal_mtx' in _ and 'cal_dist' in _):
        mtx, dist = camcalib.calibrate_folder(cal_folder, nx=9, ny=6)
        _['cal_mtx']=mtx
        _['cal_dist']=dist
        camcalib.undistort_folder(cal_folder, mtx, dist, constants.camera_cal_test_folder)
        camcalib.undistort_folder(test_images_folder, mtx, dist, constants.undistort_test_folder)

def adjust_perspective(force=False):
    """ adjust perspective unwrapping, update global values and process perspective warp test images"""
    global _
    if force or not ('persp_src_pts' in _ and 'persp_dst_pts' in _):
        src_pts = None
        dst_pts = None
        if 'persp_src_pts' in _:
            src_pts = _['persp_src_pts']
        if 'persp_dst_pts' in _:
            dst_pts = _['persp_dst_pts']
        ret, src_pts, dst_pts = persptrans.adjust_perspective(constants.thresh_test_folder, src_pts, dst_pts)
        if ret or not ('persp_src_pts' in _ and 'persp_dst_pts' in _):
            _['persp_src_pts'] = src_pts
            _['persp_dst_pts'] = dst_pts
        persptrans.warp_folder(constants.thresh_test_folder, _['persp_src_pts'], _['persp_dst_pts'], constants.persp_trans_test_folder)

def adjust_thresh(force=False):
    """ adjust thresholding and channel mixing, update global values and process thresh test images"""
    global _
    if force or not ('thresh_use_rgb' in _):
        ret, config = thresh.adjust_thresh(constants.undistort_test_folder, _)
        if ret:
            _ = config
        thresh.thresh_folder(constants.undistort_test_folder,
                             use_rgb=_['thresh_use_rgb'], rgb_thresh=_['thresh_rgb_thresh'],
                             use_hls=_['thresh_use_hls'], hls_thresh=_['thresh_hls_thresh'],
                             use_luv=_['thresh_use_luv'], luv_thresh=_['thresh_luv_thresh'],
                             use_sobel_hls=_['thresh_use_sobel_hls'], sobel_hls_thresh=_['thresh_sobel_hls_thresh'],
                             sobel_hls_kernel=_['thresh_sobel_hls_kernel'],
                             output_folder=constants.thresh_test_folder)

current_video_frame = 0
poly_log = []

# accumulator for curve radius and center offset
acc_curve_rad = None
acc_center_offset = None

def process_image(img):
    global current_video_frame, poly_log, acc_center_offset, acc_curve_rad

    # process image (undistort -> thresh -> warp -> find_lanes)
    bgr = img[:,:,[2,1,0]]    # RGB -> BGR
    undistorted = camcalib.undistort_image(bgr, _['cal_mtx'], _['cal_dist'])
    threshed = thresh.thresh_image(undistorted,
                                   use_rgb=_['thresh_use_rgb'], rgb_thresh=_['thresh_rgb_thresh'],
                                   use_hls=_['thresh_use_hls'], hls_thresh=_['thresh_hls_thresh'],
                                   use_luv=_['thresh_use_luv'], luv_thresh=_['thresh_luv_thresh'],
                                   use_sobel_hls=_['thresh_use_sobel_hls'], sobel_hls_thresh=_['thresh_sobel_hls_thresh'],
                                   sobel_hls_kernel=_['thresh_sobel_hls_kernel'])
    warped = persptrans.warp_image(threshed, _['persp_src_pts'], _['persp_dst_pts'])
    img_centroids, img_overlay, l_poly, r_poly, curve_rad, center_offset = find_lanes.find_lanes(warped,
                                                                                                 s_min=_['find_lanes_s_min'],
                                                                                                 s_max=_['find_lanes_s_max'],
                                                                                                 min_points=_['find_lanes_min_points'],
                                                                                                 min_range=_['find_lanes_min_range'],
                                                                                                 max_gap=_['find_lanes_max_gap'],
                                                                                                 ym_per_pix=_['find_lanes_ym_per_pix'],
                                                                                                 xm_per_pix=_['find_lanes_xm_per_pix'])
    img_overlay = img_overlay[:,:,[2,1,0]]    # BGR -> RGB
    unwarped_overlay = persptrans.warp_image(img_overlay, _['persp_dst_pts'], _['persp_src_pts'])

    lerp_ratio = 0.05
    if acc_curve_rad is None:
        acc_curve_rad = curve_rad
    else:
        acc_curve_rad = utils.lerp(acc_curve_rad, curve_rad, lerp_ratio)

    if acc_center_offset is None:
        acc_center_offset = center_offset
    else:
        acc_center_offset = utils.lerp(acc_center_offset, center_offset, lerp_ratio)


    overlay_mask = unwarped_overlay[:,:,0]+unwarped_overlay[:,:,1]+unwarped_overlay[:,:,2]
    overlay_mask = np.minimum(np.dstack((overlay_mask, overlay_mask, overlay_mask)), np.zeros_like(img)+100)
    img_reduced = np.uint8(np.maximum(np.int16(img)-overlay_mask, np.zeros_like(img)))

    # save poly data for debugging
    poly_log.append(l_poly.tolist() + r_poly.tolist())

    # composite lane drawing over image
    output = cv2.add(img_reduced, unwarped_overlay)

    # render curve/offset values on image
    cv2.putText(output,'Curve radius: {:.0f}m'.format(acc_curve_rad), (50,50), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255,255,255))
    offset_label = ''
    if (acc_center_offset<0):
        offset_label = 'Offset from center: {:.2f}m left'.format(abs(acc_center_offset))
    else:
        offset_label = 'Offset from center: {:.2f}m right'.format(abs(acc_center_offset))
    cv2.putText(output, offset_label, (50,80), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255,255,255))

    cv2.imwrite('{}/{:04d}.jpg'.format(constants.video_test_folder, current_video_frame), img_centroids)
    current_video_frame += 1
    return output

def process_video(input_video, output_video):
    from moviepy.editor import VideoFileClip
    global current_video_frame, poly_log

    current_video_frame = 0
    utils.make_dir(constants.video_test_folder)

    find_lanes.reset_state()

    clip = VideoFileClip(input_video)
    processed_clip = clip.fl_image(process_image)
    processed_clip.write_videofile(output_video, audio=False)

    # write polygon data for debugging
    utils.write_csv('poly_log.csv', poly_log)


def main():
    global _
    _ = utils.load_globals()

    force = False
    # force = True
    # calibrate_camera(constants.camera_cal_folder, constants.test_images_folder, force=force)
    calibrate_camera(constants.camera_cal_nx6_folder, constants.test_images_nx6_folder, force=force)
    adjust_thresh(force=force)
    adjust_perspective(force=force)

    utils.save_globals(_)

    # sys.exit(0)

    # project video

    # challenge video
    # _['find_lanes_s_min'] = 0.08
    # _['find_lanes_s_max'] = 0.5
    # _['find_lanes_min_points'] = 17
    # _['find_lanes_min_range'] = 0.6
    # _['find_lanes_max_gap'] = 0.6
    # _['find_lanes_ym_per_pix'] = 30/720
    # _['find_lanes_xm_per_pix'] = 3.7/675

    # rain video
    _['find_lanes_s_min'] = 0.035
    _['find_lanes_s_max'] = 0.31
    _['find_lanes_min_points'] = 9
    _['find_lanes_min_range'] = 0.6
    _['find_lanes_max_gap'] = 0.7
    _['find_lanes_ym_per_pix'] = 25/720
    _['find_lanes_xm_per_pix'] = 3.7/610


    # process_video(constants.video_project, '{}/{}'.format(constants.output_folder, constants.video_project))
    # process_video(constants.video_challenge, '{}/{}'.format(constants.output_folder, constants.video_challenge))
    process_video(constants.video_rain, '{}/{}'.format(constants.output_folder, constants.video_rain))

    utils.save_globals(_)

if __name__=='__main__':
    main()
