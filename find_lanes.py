import numpy as np
import cv2
import constants, utils
from scipy import ndimage

avg_centroids = None
avg_curve_rad = None
avg_center_offset_m = None
last_l_poly = None
last_r_poly = None

def reset_state():
    global avg_centroids, avg_curve_rad, avg_center_offset_m, last_l_poly, last_r_poly
    avg_centroids = None
    avg_curve_rad = None
    avg_center_offset_m = None
    last_l_poly = None
    last_r_poly = None

def draw_centroids(img, centroids, window_size, l_poly, r_poly, strength_min):
    img = img.copy()
    # print(img.shape)
    w2 = int(window_size[0]/2)
    h = window_size[1]
    img_h = img.shape[0]
    nc = np.array(centroids)
    l_strength_max = np.max(nc[:,3])
    r_strength_max = np.max(nc[:,4])
    for c in centroids:
        if c[3]<strength_min:
            l_strength = 0
        else:
            l_strength = c[3]/l_strength_max
        if c[4]<strength_min:
            r_strength = 0
        else:
            r_strength = c[4]/r_strength_max
        c[0]=int(c[0])
        c[1]=int(c[1])
        utils.img_draw_rectangle(img, (c[0]-w2, c[2]), (c[0]+w2, c[2]-h), color=(int((l_strength/2+0.5)*255),0,0), thickness=1)
        utils.img_draw_rectangle(img, (c[1]-w2, c[2]), (c[1]+w2, c[2]-h), color=(int((r_strength/2+0.5)*255),0,0), thickness=1)
        cv2.putText(img, '{:.1f}'.format(c[3]), (c[0]+w2*2, c[2]-h), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,0,255))
        cv2.putText(img, '{:.1f}'.format(c[4]), (c[1]+w2*2, c[2]-h), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,0,255))
        if l_strength==0:
            utils.img_draw_dot(img, (c[0], c[2]), color=(255,0,0), radius=1)
        else:
            utils.img_draw_dot(img, (c[0], c[2]), color=(0,0,int((l_strength/2+0.5)*255)))
        if r_strength==0:
            utils.img_draw_dot(img, (c[1], c[2]), color=(255,0,0), radius=1)
        else:
            utils.img_draw_dot(img, (c[1], c[2]), color=(0,0,int((r_strength/2+0.5)*255)))
    for y in range(img_h):
        l_x = int(l_poly[0]*y**2+l_poly[1]*y+l_poly[2])
        r_x = int(r_poly[0]*y**2+r_poly[1]*y+r_poly[2])
        utils.img_draw_dot(img, (l_x, y), color=(0,255,255), radius=2)
        utils.img_draw_dot(img, (r_x, y), color=(0,255,255), radius=2)
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


def find_window_centroids(img, window_size , margin, strength_min=0.5, hint_centroids=None):
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


def lerp(a, b, ratio):
    return a*(1.0-ratio) + b * ratio

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
            avg_centroids[i][0] = lerp(a_c[0], c[0], lerp_ratio)
            avg_centroids[i][3] = lerp(a_c[3], c[3], lerp_ratio)
        else:
            # avg_centroids[i][3] = min(lerp(a_c[3], 0, lerp_ratio/2),10)
            pass

        if a_c[4]<=strength_min:
            avg_centroids[i][1] = c[1]
            avg_centroids[i][4] = c[4]
        elif c[4]>strength_min:
            avg_centroids[i][1] = lerp(a_c[1], c[1], lerp_ratio)
            avg_centroids[i][4] = lerp(a_c[4], c[4], lerp_ratio)
        else:
            # avg_centroids[i][4] = min(lerp(a_c[4], 0, lerp_ratio/2),10)
            pass

    return avg_centroids


def find_lanes(img):
    global avg_centroids, avg_center_offset_m, avg_curve_rad, last_l_poly, last_r_poly
    strength_min = 1

    # window settings
    window_width = 50
    window_height = 10
    margin = 50           # How much to slide left and right for searching

    #cv2.imshow('find lanes',img*255)
    #cv2.waitKey()

    input_image = img
    if len(img.shape)==3:
        input_image = img[:,:,0]
    if np.max(input_image)==1:
        input_image *= 255

    window_size = (window_width, window_height)
    centroids = find_window_centroids(input_image, window_size, margin, strength_min=1, hint_centroids=avg_centroids)
    #pp = pprint.PrettyPrinter(indent=4)
    #pp.pprint(centroids)

    avg_centroids = lerp_centroids(centroids, strength_min, lerp_ratio=0.7)

    # c = np.array(avg_centroids)
    c = np.array(centroids)

    s_min = 4
    s_max = 29
    lc = c[np.ix_(c[:,3]>s_min, (0,2,3))]   # select l_x, l_y, l_strength, where l_strength > strength_min
    rc = c[np.ix_(c[:,4]>s_min, (1,2,4))]   # select r_x, r_y, r_strength, where r_strength > strength_min
    lc = lc[np.ix_(lc[:,2]<s_max, (0,1,2))]
    rc = rc[np.ix_(rc[:,2]<s_max, (0,1,2))]

    # print('-----')
    # print(lc)
    # print('-----')
    # print(rc)

    # only fit poly if we have enough points
    if len(lc)>17:
        l_poly = np.polyfit(lc[:,1], lc[:,0], 2)
    else:
        l_poly = last_l_poly
    if len(rc)>17:
        r_poly = np.polyfit(rc[:,1], rc[:,0], 2)
    else:
        r_poly = last_r_poly

    if last_l_poly is None:
        last_l_poly = l_poly
    if last_r_poly is None:
        last_r_poly = r_poly

    # check maximum abs(a) value of ax2*bx+c second degree poly
    # poly_max_a = 0.001
    poly_max_a_diff = 0.0002
    # poly_max_b_diff = 0.05
    # if abs(l_poly[0])>poly_max_a:
    #     print('l_poly: poly_max_a', l_poly[0])
    #     l_poly = last_l_poly
    if abs(l_poly[0]-last_l_poly[0])>poly_max_a_diff:
        print('l_poly: poly_max_a_diff{:.8f} {:.8f} {:.8f}'.format(l_poly[0], last_l_poly[0], abs(l_poly[0]-last_l_poly[0])))
        l_poly = last_l_poly
    else:
        last_l_poly = l_poly

    # if abs(r_poly[0])>poly_max_a:  # or abs(r_poly[1]-last_r_poly[1])>poly_max_b_diff:
    #     print('r_poly: poly_max_a', r_poly[0])
    #     r_poly = last_r_poly
    if abs(r_poly[0]-last_r_poly[0])>poly_max_a_diff:
        print('r_poly: poly_max_a_diff {:.8f} {:.8f} {:.8f}'.format(r_poly[0], last_r_poly[0], abs(r_poly[0]-last_r_poly[0])))
        r_poly = last_r_poly
    else:
        last_r_poly = r_poly


    # calculate distance from center
    x_center = 1280/2
    y_pos = 720
    lane_width_m = 3.7
    l_x_pos = l_poly[0]**2*y_pos + l_poly[1]*y_pos + l_poly[2]
    r_x_pos = r_poly[0]**2*y_pos + r_poly[1]*y_pos + r_poly[2]
    lane_width = r_x_pos-l_x_pos
    center_offset = ((x_center-l_x_pos)-lane_width/2)/2
    center_offset_m = lane_width_m / lane_width * center_offset
    #print (l_x_pos, r_x_pos, lane_width, center_offset_m)

    if avg_center_offset_m is None:
        avg_center_offset_m = center_offset_m
    else:
        avg_center_offset_m = lerp(avg_center_offset_m, center_offset_m, 0.2)

    # define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720    # meters per pixel in y dimension
    xm_per_pix = 3.7/675   # meters per pixel in x dimension
    y_eval = 720

    if len(lc)>2 and len(rc)>2:
        l_w_poly = np.polyfit(lc[:,1]*ym_per_pix, lc[:,0]*xm_per_pix, 2)
        r_w_poly = np.polyfit(rc[:,1]*ym_per_pix, rc[:,0]*xm_per_pix, 2)
        l_rad = ((1 + (2*l_w_poly[0]*y_eval*ym_per_pix + l_w_poly[1])**2)**1.5) / np.absolute(2*l_w_poly[0])
        r_rad = ((1 + (2*r_w_poly[0]*y_eval*ym_per_pix + r_w_poly[1])**2)**1.5) / np.absolute(2*r_w_poly[0])

        curve_rad = (l_rad+r_rad) / 2

        if avg_curve_rad is None:
            avg_curve_rad = curve_rad
        else:
            avg_curve_rad = lerp(avg_curve_rad, curve_rad, 0.2)

    #print(l_rad, r_rad)

    # print('left poly:',l_poly)
    # print('right poly:',r_poly)

    img3 = np.dstack((input_image, input_image, input_image))

    img_centroids = draw_centroids(img3, centroids, window_size, l_poly, r_poly, strength_min)

    img_overlay = draw_overlay(img3, centroids, window_size, l_poly, r_poly)

    return (img_centroids, img_overlay, l_poly, r_poly, avg_curve_rad, avg_center_offset_m)

if __name__=='__main__':
    img = cv2.imread('{}/c0001.png'.format(constants.persp_trans_test_folder))
    img_centroids, img_overlay, l_poly, r_poly, curve_rad, center_offset = find_lanes(img)
    cv2.imshow('centroids', img_centroids)
    cv2.waitKey(3000)
