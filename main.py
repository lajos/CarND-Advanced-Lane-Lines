
import cv2
import constants, utils, camcalib, persptrans, thresh
import numpy as np
import sys, pprint
import matplotlib.pyplot as pyplot
from scipy import ndimage

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

def adjust_thresh(force=False):
    global _
    if force or not ('thresh_use_rgb' in _):
        ret, config = thresh.adjust_thresh(constants.persp_trans_test_folder, _)
        if ret:
            _ = config
        thresh.thresh_folder(constants.persp_trans_test_folder,
                             use_rgb=_['thresh_use_rgb'], rgb_thresh=_['thresh_rgb_thresh'],
                             use_hls=_['thresh_use_hls'], hls_thresh=_['thresh_hls_thresh'],
                             use_luv=_['thresh_use_luv'], luv_thresh=_['thresh_luv_thresh'],
                             use_sobel_hls=_['thresh_use_sobel_hls'], sobel_hls_thresh=_['thresh_sobel_hls_thresh'],
                             sobel_hls_kernel=_['thresh_sobel_hls_kernel'],
                             output_folder=constants.thresh_test_folder)


def test():
    img = cv2.imread('{}/straight_lines1.jpg'.format(constants.thresh_test_folder))

    warped_ , warped_, warped_ = cv2.split(img)
    #warped = warped/255

    # window settings
    window_width_ = 80
    window_height_ = 20 # Break image into 9 vertical layers since image height is 720
    margin = 100 # How much to slide left and right for searching

    def window_mask(width, height, img_ref, center,level):
        output = np.zeros_like(img_ref)
        output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
        return output

    def draw_centroids():
        pass

    def find_window_centroids(img, window_w, window_h, margin, strength_min=0.5):
        """[strength_min] ignore windows found with little data"""
        img_w=img.shape[1]
        img_h=img.shape[0]
        img_w2 = int(img_w/2)
        img_h2 = int(img_h/2)
        window_w2 = int(window_w/2)

        img_max = np.max(img)
        print(img_max)

        window_centroids = []        # store the (left, right, l_strength, r_strength) window centroid positions per level
        window = np.ones(window_w)   # create our window template that we will use for convolutions

        # first find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
        # and then np.convolve the vertical image slice with the window template

        # sum quarter bottom of image to get slice, could use a different ratio, use as hint for where to start searching
        l_sum = np.sum(img[int(img_h*0.75):, :img_w2], axis=0)
        l_center_hint = np.argmax(np.convolve(window, l_sum)) - window_w2
        r_sum = np.sum(img[int(img_h*.75):, img_w2:], axis=0)
        r_center_hint = np.argmax(np.convolve(window,r_sum)) - window_w2 + img_w2

        # go through each layer looking for max pixel locations
        for level in range(0, int(img_h/window_h)):
            # convolve the window into the vertical slice of the image
            image_layer = np.sum(img[int(img_h-(level+1)*window_h):int(img_h-level*window_h),:], axis=0)
            print(np.argmax(image_layer))
            conv_signal = np.convolve(window, image_layer)

            conv_signal = ndimage.filters.gaussian_filter1d(conv_signal, 17, mode='constant')

            # pyplot.plot(conv_signal)
            # pyplot.plot(filtered)
            # pyplot.show()
            # cv2.waitKey()
            # sys.exit(0)

            # find the best left centroid by using past left center as a reference
            # use window_w/2 as offset because convolution signal reference is at right side of window, not center of window
            offset = window_w2
            l_min_index = int(max(l_center_hint + window_w2 - margin,0))
            l_max_index = int(min(l_center_hint + window_w2 + margin, img_w))
            l_strength = np.sum(conv_signal[l_min_index:l_max_index]) / (l_max_index - l_min_index) / img_max / window_h
            l_center = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - window_w2

            # find the best right centroid by using past right center as a reference
            r_min_index = int(max(r_center_hint + window_w2 - margin, 0))
            r_max_index = int(min(r_center_hint + window_w2 + margin, img_w))
            r_strength = np.sum(conv_signal[r_min_index:r_max_index]) / (r_max_index - r_min_index) / img_max / window_h
            r_center = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - window_w2

            # add what we found for that layer
            window_centroids.append((l_center, r_center, l_strength, r_strength))

            # update hint only if we found enough data
            if l_strength > strength_min:
                l_center_hint = l_center
            if r_strength > strength_min:
                r_center_hint = r_center


        return window_centroids

    window_centroids = find_window_centroids(warped_, window_width_, window_height_, margin)
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(window_centroids)

    warped = warped_
    window_width = window_width_
    window_height = window_height_

    # If we found any window centers
    if len(window_centroids) > 0:

        # Points used to draw all the left and right windows
        l_points = np.zeros_like(warped)
        r_points = np.zeros_like(warped)

        # Go through each level and draw the windows
        for level in range(0,len(window_centroids)):
            # Window_mask is a function to draw window areas
            l_mask = window_mask(window_width,window_height,warped,window_centroids[level][0],level)
            r_mask = window_mask(window_width,window_height,warped,window_centroids[level][1],level)
            # Add graphic points from window mask here to total pixels found
            l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
            r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

        # Draw the results
        template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
        zero_channel = np.zeros_like(template) # create a zero color channel
        template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
        warpage = np.array(cv2.merge((warped,warped,warped)),np.uint8) # making the original road pixels 3 color channels
        output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results

    # If no window centers found, just display orginal road image
    else:
        output = np.array(cv2.merge((warped,warped,warped)),np.uint8)


    cv2.imshow('test', output)
    cv2.waitKey()

def main():
    global _
    _ = utils.load_globals()
    #print(_)

    calibrate_camera(force=False)
    adjust_perspective(force=False)
    adjust_thresh(force=False)

    test()

    #utils.display_image_file('test_images/straight_lines1.jpg')
    #utils.plt_image_file('test_images/straight_lines1.jpg', bgr=True)

    utils.save_globals(_)

if __name__=='__main__':
    main()
