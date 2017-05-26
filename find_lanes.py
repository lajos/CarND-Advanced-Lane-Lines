import numpy as np
import cv2, sys
import constants, utils
from scipy import ndimage

avg_centroids = None
last_l_poly = None
last_r_poly = None
last_l_weights = None
last_r_weights = None
last_l_centroids = None
last_r_centroids = None

def reset_state():
    global avg_centroids, last_l_poly, last_r_poly
    avg_centroids = None
    last_l_poly = None
    last_r_poly = None
    last_l_weights = None
    last_r_weights = None
    last_l_centroids = None
    last_r_centroids = None

def draw_centroids(img, centroids, window_size, l_poly, r_poly, strength_min, s_min, s_max):
    img = img.copy()
    # print(img.shape)
    w2 = int(window_size[0]/2)
    h = window_size[1]
    h2 = int(h/2)
    img_h = img.shape[0]
    nc = np.array(centroids)
    for c in centroids:
        if c[3]<strength_min:
            l_strength = 0
        else:
            l_strength = c[3]
        if c[4]<strength_min:
            r_strength = 0
        else:
            r_strength = c[4]
        c[0]=int(c[0])
        c[1]=int(c[1])
        utils.img_draw_rectangle(img, (c[0]-w2, c[2]), (c[0]+w2, c[2]-h), color=(int((l_strength/2+0.5)*255),0,0), thickness=1)
        utils.img_draw_rectangle(img, (c[1]-w2, c[2]), (c[1]+w2, c[2]-h), color=(int((r_strength/2+0.5)*255),0,0), thickness=1)

        if not l_strength==0:
            cv2.putText(img, '{:.2f}'.format(c[3]), (c[0]+w2*2, c[2]), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,0,255))
        if not r_strength==0:
            cv2.putText(img, '{:.2f}'.format(c[4]), (c[1]+w2*2, c[2]), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,0,255))

        if l_strength==0:
            utils.img_draw_dot(img, (c[0], c[2]), color=(255,0,0), radius=1)
        elif l_strength>s_min and l_strength<s_max:
            utils.img_draw_dot(img, (c[0], c[2]), color=(0,int((l_strength/2+0.5)*255),0))
        else:
            utils.img_draw_dot(img, (c[0], c[2]), color=(0,0,int((l_strength/2+0.5)*255)))

        if r_strength==0:
            utils.img_draw_dot(img, (c[1], c[2]), color=(255,0,0), radius=1)
        elif r_strength>s_min and r_strength<s_max:
            utils.img_draw_dot(img, (c[1], c[2]), color=(0,int((r_strength/2+0.5)*255),0))
        else:
            utils.img_draw_dot(img, (c[1], c[2]), color=(0,0,int((r_strength/2+0.5)*255)))

    for y in range(img_h):
        l_x = int(l_poly[0]*y**2+l_poly[1]*y+l_poly[2])
        r_x = int(r_poly[0]*y**2+r_poly[1]*y+r_poly[2])
        utils.img_draw_dot(img, (l_x, y+h), color=(0,255,255), radius=2)
        utils.img_draw_dot(img, (r_x, y+h), color=(0,255,255), radius=2)
    return img


def draw_overlay(img, centroids, window_size, l_poly, r_poly):
    pts = []
    img_h = img.shape[0]
    img_overlay = np.zeros_like(img)
    for y in range(0,img_h,10):
        r_x = int(r_poly[0]*y**2+r_poly[1]*y+r_poly[2])
        pts.append([r_x, y])
    for y in reversed(range(0,img_h,10)):
       l_x = int(l_poly[0]*y**2+l_poly[1]*y+l_poly[2])
       pts.append([l_x, y])
    cv2.fillPoly(img_overlay, np.array(pts, dtype=np.int32)[None,:], color=(0,200,0))
    for y in range(img_h):
        l_x = int(l_poly[0]*y**2+l_poly[1]*y+l_poly[2])
        r_x = int(r_poly[0]*y**2+r_poly[1]*y+r_poly[2])
        utils.img_draw_dot(img_overlay, (l_x, y), color=(255,0,0), radius=5)
        utils.img_draw_dot(img_overlay, (r_x, y), color=(0,0,255), radius=5)
    return img_overlay


def find_window_centroids(img, window_size , margin, strength_min=0, hint_centroids=None):
    """[strength_min] ignore windows found with little data
    centroids are array of (left_center_x, right_center_x, pos_y, l_strength, r_strength)
    """
    window_w, window_h = window_size
    img_w=img.shape[1]
    img_h=img.shape[0]
    img_w2 = int(img_w/2)
    img_h2 = int(img_h/2)
    window_w2 = int(window_w/2)

    img_max = np.max(img)

    window_centroids = []
    window = np.ones(window_w)   # convolution window

    # first find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template

    if (hint_centroids):
        l_center_hint = hint_centroids[0][0]
        r_center_hint = hint_centroids[0][1]
    else:
        # sum quarter bottom of image to get slice, could use a different ratio, use as hint for where to start searching
        l_sum = np.sum(img[int(img_h*0.75):, :img_w2], axis=0)
        l_center_hint = np.argmax(np.convolve(window, l_sum)) - window_w2
        r_sum = np.sum(img[int(img_h*.75):, img_w2:], axis=0)
        r_center_hint = np.argmax(np.convolve(window,r_sum)) - window_w2 + img_w2

    # go through each layer looking for max pixel locations
    for level in range(0, int(img_h/window_h)):
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(img[int(img_h-(level+1)*window_h):int(img_h-level*window_h),:], axis=0)
        # print(np.argmax(image_layer))
        conv_signal = np.convolve(window, image_layer)

        conv_signal = ndimage.filters.gaussian_filter1d(conv_signal, 17, mode='constant')

        # find the best left centroid by using past left center as a reference
        # use window_w/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = window_w2
        l_min_index = int(max(l_center_hint + window_w2 - margin,0))
        l_max_index = int(min(l_center_hint + window_w2 + margin, img_w))
        l_strength = np.sum(conv_signal[l_min_index:l_max_index]) / (l_max_index - l_min_index) / img_max / window_h / window_w
        l_center = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - window_w2

        # find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center_hint + window_w2 - margin, 0))
        r_max_index = int(min(r_center_hint + window_w2 + margin, img_w))
        r_strength = np.sum(conv_signal[r_min_index:r_max_index]) / (r_max_index - r_min_index) / img_max / window_h / window_w
        r_center = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - window_w2

        # add what we found for that layer
        window_centroids.append([l_center, r_center, img_h-level*window_h, l_strength, r_strength])

        # update hint only if we found enough data
        if l_strength > strength_min:
            l_center_hint = l_center
        elif hint_centroids is not None and hint_centroids[level][3]>strength_min:
            l_center_hint = hint_centroids[level][0]

        if r_strength > strength_min:
            r_center_hint = r_center
        elif hint_centroids is not None and hint_centroids[level][4]>strength_min:
            r_center_hint = hint_centroids[level][1]

    return window_centroids


def lerp_centroids(centroids, strength_min, lerp_ratio = 0.2):
    global avg_centroids
    if avg_centroids is None:
        avg_centroids = centroids.copy()
        return avg_centroids

    for i in range(len(centroids)):
        c = centroids[i]
        a_c = avg_centroids[i]

        if a_c[3]<=strength_min:
            avg_centroids[i][0] = c[0]
            avg_centroids[i][3] = c[3]
        elif c[3]>strength_min:
            avg_centroids[i][0] = utils.lerp(a_c[0], c[0], lerp_ratio)
            avg_centroids[i][3] = utils.lerp(a_c[3], c[3], lerp_ratio)
        else:
            # avg_centroids[i][3] = min(utils.lerp(a_c[3], 0, lerp_ratio/2),10)
            pass

        if a_c[4]<=strength_min:
            avg_centroids[i][1] = c[1]
            avg_centroids[i][4] = c[4]
        elif c[4]>strength_min:
            avg_centroids[i][1] = utils.lerp(a_c[1], c[1], lerp_ratio)
            avg_centroids[i][4] = utils.lerp(a_c[4], c[4], lerp_ratio)
        else:
            # avg_centroids[i][4] = min(utils.lerp(a_c[4], 0, lerp_ratio/2),10)
            pass

    return avg_centroids


def poly_fit_random_weight(c, last_poly, min_points):
    """ try to get better fit by dropping and randomly weighting centroids """
    best_poly = np.polyfit(c[:,1], c[:,0], 2)
    best_weights = np.ones_like(c[:,0])

    top_y = 0
    bottom_y = constants.image_height

    best_diff_top = abs(np.polyval(last_poly,top_y) - np.polyval(best_poly,top_y))
    best_diff_bottom = abs(np.polyval(last_poly,bottom_y) - np.polyval(best_poly,bottom_y))

    tail_idx = int(0.6 * len(c))
    tail_idx = max(tail_idx, min_points)

    # drop last centroids [from top of image]
    for i in reversed(range(tail_idx, len(c))):
        w = np.zeros_like(c[:,0])
        w[0:i] = 1
        l_poly = np.polyfit(c[:,1], c[:,0], 2, w=w)

        diff_top = abs(np.polyval(last_poly,top_y) - np.polyval(l_poly,top_y))
        diff_bottom = abs(np.polyval(last_poly,bottom_y) - np.polyval(l_poly,bottom_y))

        if diff_top<best_diff_top and diff_bottom<best_diff_bottom:
            # print('better without top')
            best_diff_top = diff_top
            best_diff_bottom = diff_bottom
            best_poly = l_poly
            best_weights = w

    # drop [n] consecutive centroids
    for o in range(1,4):
        for i in range(0, len(c)-o+1):
            w = np.ones_like(c[:,0])
            for j in range(0,o):
                w[i+j]=0
            l_poly = np.polyfit(c[:,1], c[:,0], 2, w=w)

            diff_top = abs(np.polyval(last_poly,top_y) - np.polyval(l_poly,top_y))
            diff_bottom = abs(np.polyval(last_poly,bottom_y) - np.polyval(l_poly,bottom_y))

            if diff_top<best_diff_top and diff_bottom<best_diff_bottom:
                best_diff_top = diff_top
                best_diff_bottom = diff_bottom
                best_poly = l_poly
                best_weights = w

    # randomly weight centroids
    for i in range(100):
        w = np.random.random(c[:,0].size) * 5
        l_poly = np.polyfit(c[:,1], c[:,0], 2, w=w)

        diff_top = abs(np.polyval(last_poly,top_y) - np.polyval(l_poly,top_y))
        diff_bottom = abs(np.polyval(last_poly,bottom_y) - np.polyval(l_poly,bottom_y))

        if diff_top<best_diff_top and diff_bottom<best_diff_bottom:
            best_diff_top = diff_top
            best_diff_bottom = diff_bottom
            best_poly = l_poly
            best_weights = w

    return best_poly, best_weights

def find_lanes(img, s_min=0.08, s_max=0.9, min_points=17, min_range=0.6, max_gap=0.6, ym_per_pix = 30/720, xm_per_pix = 3.7/675):
    """
    find lanes in [img]
    [s_min]: minimum strength of centroid to be evaluated
    [s_max]: maximum strength of centroid to be evaluated
    [min_range]: minimum y range of centroids to be valid (0:none - 1:img_height)
    [max_gap]: maximum gap between centroids (0:none - 1:img_height)
    [ym_per_pix]: meters per pixel in y dimension
    [xm_per_pix]: meters per pixel in x dimension
    """
    global avg_centroids, last_l_poly, last_r_poly, last_l_weights, last_r_weights
    global last_l_centroids, last_r_centroids

    # minimum strength of centroid to find
    strength_min = 0.005

    # concolution window settings
    window_width = 50
    window_height = 10
    margin = 50           # How much to slide left and right for searching

    input_image = img
    if len(img.shape)==3:
        input_image = img[:,:,0]
    if np.max(input_image)==1:
        input_image *= 255

    # cv2.imshow('find_lanes', input_image)
    # cv2.waitKey()

    window_size = (window_width, window_height)
    centroids = find_window_centroids(input_image, window_size, margin, strength_min=strength_min, hint_centroids=avg_centroids)

    if avg_centroids is None:
        avg_centroids = centroids
    else:
        avg_centroids = lerp_centroids(centroids, strength_min, lerp_ratio=0.7)

    c = np.array(centroids)

    lc = c[np.ix_(c[:,3]>s_min, (0,2,3))]   # select l_x, l_y, l_strength, where l_strength > strength_min
    rc = c[np.ix_(c[:,4]>s_min, (1,2,4))]   # select r_x, r_y, r_strength, where r_strength > strength_min
    lc = lc[np.ix_(lc[:,2]<s_max, (0,1,2))]
    rc = rc[np.ix_(rc[:,2]<s_max, (0,1,2))]

    # line length (range)
    if len(lc):
        lc_y_range = np.max(lc[:,1]) - np.min(lc[:,1])
    if len(rc):
        rc_y_range = np.max(rc[:,1]) - np.min(rc[:,1])

    # max gap
    if len(lc)>2:
        lc_max_y_gap = np.max(np.abs((lc[:,1] - np.roll(lc[:,1],1))[1:]))
    if len(rc)>2:
        rc_max_y_gap = np.max(np.abs((rc[:,1] - np.roll(rc[:,1],1))[1:]))

    # minimum number of points required to fit poly
    min_points = 9                                         # was 17
    min_y_range = constants.image_height * 0.6              # was 0.6
    max_y_gap = constants.image_height * 0.7                # was 0.6

    # fit left poly if we have enough left points
    if len(lc)>min_points and lc_y_range>min_y_range and lc_max_y_gap<max_y_gap:
        if last_l_poly is None:
            l_poly = np.polyfit(lc[:,1], lc[:,0], 2)
            l_weights = np.ones_like(lc[:,0])
        else:
            l_poly, l_weights = poly_fit_random_weight(lc, last_l_poly, min_points)
    else:
        l_poly = last_l_poly
        l_weights = last_l_weights
        lc = last_l_centroids

    # fit right poly if we have enough right points
    if len(rc)>min_points and rc_y_range>min_y_range and rc_max_y_gap<max_y_gap:
        if last_r_poly is None:
            r_poly = np.polyfit(rc[:,1], rc[:,0], 2)
            r_weights = np.ones_like(rc[:,0])
        else:
            r_poly, r_weights = poly_fit_random_weight(rc, last_r_poly, min_points)
    else:
        r_poly = last_r_poly
        r_weights = last_r_weights
        rc = last_r_centroids

    if last_l_poly is None:
        last_l_poly = l_poly
        last_l_weights = l_weights
        last_l_centroids = lc
    if last_r_poly is None:
        last_r_poly = r_poly
        last_r_weights = r_weights
        last_r_centroids = rc


    # check maximum abs(a) value of ax2*bx+c second degree poly
    poly_max_a_diff = 0.00025
    if abs(l_poly[0]-last_l_poly[0])>poly_max_a_diff:
        # print('l_poly: poly_max_a_diff{:.8f} {:.8f} {:.8f}'.format(l_poly[0], last_l_poly[0], abs(l_poly[0]-last_l_poly[0])))
        l_poly = last_l_poly
        l_weights = last_l_weights
        lc = last_l_centroids

    if abs(r_poly[0]-last_r_poly[0])>poly_max_a_diff:
        # print('r_poly: poly_max_a_diff {:.8f} {:.8f} {:.8f}'.format(r_poly[0], last_r_poly[0], abs(r_poly[0]-last_r_poly[0])))
        r_poly = last_r_poly
        r_weights = last_r_weights
        rc = last_r_centroids

    s_l_a = np.sign(l_poly[0])
    s_r_a = np.sign(r_poly[0])
    s_l_l_a = np.sign(last_l_poly[0])
    s_l_r_a = np.sign(last_r_poly[0])
    if not s_l_a==s_r_a and s_r_a==s_l_l_a:
        # print('left sign fix')
        l_poly = last_l_poly
        l_weights = last_l_weights
        lc = last_l_centroids
    if not s_r_a==s_l_a and s_l_a==s_l_r_a:
        # print('right sign fix')
        r_poly = last_r_poly
        r_weights = last_r_weights
        rc = last_r_centroids

    # save polys to last
    last_l_poly = l_poly
    last_l_weights = l_weights
    last_l_centroids = lc
    last_r_poly = r_poly
    last_r_weights = r_weights
    last_r_centroids = rc

    # calculate distance from center
    x_center = constants.image_width/2
    y_pos = constants.image_height
    lane_width_m = constants.lane_width_m
    l_x_pos = l_poly[0]**2*y_pos + l_poly[1]*y_pos + l_poly[2]
    r_x_pos = r_poly[0]**2*y_pos + r_poly[1]*y_pos + r_poly[2]
    lane_width = r_x_pos-l_x_pos
    center_offset = ((x_center-l_x_pos)-lane_width/2)/2
    center_offset_m = lane_width_m / lane_width * center_offset

    # define conversions in x and y from pixels space to meters
    y_eval = constants.image_height     # evaluate at bottom of image

    if len(lc)>2 and len(rc)>2:
        l_w_poly = np.polyfit(lc[:,1]*ym_per_pix, lc[:,0]*xm_per_pix, 2, w=l_weights)
        r_w_poly = np.polyfit(rc[:,1]*ym_per_pix, rc[:,0]*xm_per_pix, 2, w=r_weights)
        l_rad = ((1 + (2*l_w_poly[0]*y_eval*ym_per_pix + l_w_poly[1])**2)**1.5) / np.absolute(2*l_w_poly[0])
        r_rad = ((1 + (2*r_w_poly[0]*y_eval*ym_per_pix + r_w_poly[1])**2)**1.5) / np.absolute(2*r_w_poly[0])

        curve_rad = min(l_rad, r_rad)

    img3 = np.dstack((input_image, input_image, input_image))

    # draw centroids for debugging
    img_centroids = draw_centroids(img3, centroids, window_size, l_poly, r_poly, strength_min, s_min, s_max )

    img_overlay = draw_overlay(img3, centroids, window_size, l_poly, r_poly)

    return (img_centroids, img_overlay, l_poly, r_poly, curve_rad, center_offset_m)

if __name__=='__main__':
    img = cv2.imread('{}/r1_0001.png'.format(constants.persp_trans_test_folder))
    img_centroids, img_overlay, l_poly, r_poly, curve_rad, center_offset = find_lanes(img)
    cv2.imshow('centroids', img_centroids)
    # cv2.waitKey(3000)
    cv2.waitKey()
