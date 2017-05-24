
import cv2
import constants, utils, camcalib, persptrans, thresh, find_lanes
import numpy as np
import sys, pprint, time
import matplotlib.pyplot as pyplot

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
        ret, src_pts, dst_pts = persptrans.adjust_perspective(constants.thresh_test_folder, src_pts, dst_pts)
        if ret or not ('persp_src_pts' in _ and 'persp_dst_pts' in _):
            _['persp_src_pts'] = src_pts
            _['persp_dst_pts'] = dst_pts
        persptrans.warp_folder(constants.thresh_test_folder, _['persp_src_pts'], _['persp_dst_pts'], constants.persp_trans_test_folder)

def adjust_thresh(force=False):
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

def process_image(img):
    global current_video_frame
    global poly_log
    # if current_video_frame > 136:
    #     utils.write_csv('poly_log.csv', poly_log)
    #     sys.exit(0)
    bgr = img[:,:,[2,1,0]]
    undistorted = camcalib.undistort_image(bgr, _['cal_mtx'], _['cal_dist'])
    threshed = thresh.thresh_image(undistorted,
                                   use_rgb=_['thresh_use_rgb'], rgb_thresh=_['thresh_rgb_thresh'],
                                   use_hls=_['thresh_use_hls'], hls_thresh=_['thresh_hls_thresh'],
                                   use_luv=_['thresh_use_luv'], luv_thresh=_['thresh_luv_thresh'],
                                   use_sobel_hls=_['thresh_use_sobel_hls'], sobel_hls_thresh=_['thresh_sobel_hls_thresh'],
                                   sobel_hls_kernel=_['thresh_sobel_hls_kernel'])
    warped = persptrans.warp_image(threshed, _['persp_src_pts'], _['persp_dst_pts'])
    img_centroids, img_overlay, l_poly, r_poly, curve_rad, center_offset_m = find_lanes.find_lanes(warped)

    poly_log.append(l_poly.tolist() + r_poly.tolist())

    img_overlay = img_overlay[:,:,[2,1,0]]
    unwarped_overlay = persptrans.warp_image(img_overlay, _['persp_dst_pts'], _['persp_src_pts'])

    overlay_mask = unwarped_overlay[:,:,0]+unwarped_overlay[:,:,1]+unwarped_overlay[:,:,2]
    overlay_mask = np.minimum(np.dstack((overlay_mask, overlay_mask, overlay_mask)), np.zeros_like(img)+100)
    img_reduced = np.uint8(np.maximum(np.int16(img)-overlay_mask, np.zeros_like(img)))

    output = cv2.add(img_reduced, unwarped_overlay)

    cv2.putText(output,'Curve radius: {:.0f}m'.format(curve_rad), (50,50), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255,255,255))
    offset_label = ''
    if (center_offset_m<0):
        offset_label = 'Offset from center: {:.2f}m left'.format(abs(center_offset_m))
    else:
        offset_label = 'Offset from center: {:.2f}m right'.format(abs(center_offset_m))
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
    #print(poly_log)
    utils.write_csv('poly_log.csv', poly_log)


def main():
    global _
    _ = utils.load_globals()
    #print(_)

    force = False
    force = True
    calibrate_camera(force=force)
    adjust_thresh(force=force)
    adjust_perspective(force=force)

    #process_video(constants.video_project_part, '{}/{}'.format(constants.output_folder, constants.video_project_part))
    # process_video(constants.video_project, '{}/{}'.format(constants.output_folder, constants.video_project))

    #process_video(constants.video_challenge, '{}/{}'.format(constants.output_folder, constants.video_challenge))
    #process_video(constants.video_challenge_harder, '{}/{}'.format(constants.output_folder, constants.video_challenge_harder))

    #img = cv2.imread('{}/straight_lines2.jpg'.format(constants.thresh_test_folder))
    #img = cv2.imread('{}/test5.jpg'.format(constants.thresh_test_folder))
    #find_lanes.find_lanes(img)

    #utils.display_image_file('test_images/straight_lines1.jpg')
    #utils.plt_image_file('test_images/straight_lines1.jpg', bgr=True)

    utils.save_globals(_)

if __name__=='__main__':
    main()
