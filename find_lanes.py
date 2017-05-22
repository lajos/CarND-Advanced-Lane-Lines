import numpy as np
import cv2
import constants, utils
from scipy import ndimage


def draw_centroids(img, centroids, window_size, l_poly, r_poly):
    img = img.copy()
    # print(img.shape)
    w2 = int(window_size[0]/2)
    h = window_size[1]
    img_h = img.shape[0]
    nc = np.array(centroids)
    l_strenth_max = np.max(nc[:,3])
    r_strenth_max = np.max(nc[:,4])
    for c in centroids:
        l_strength = c[3]/l_strenth_max
        r_strength = c[4]/r_strenth_max
        c[0]=int(c[0])
        c[1]=int(c[1])
        utils.img_draw_rectangle(img, (c[0]-w2, c[2]), (c[0]+w2, c[2]-h), color=(int((l_strength/2+0.5)*255),0,0), thickness=1)
        utils.img_draw_rectangle(img, (c[1]-w2, c[2]), (c[1]+w2, c[2]-h), color=(int((r_strength/2+0.5)*255),0,0), thickness=1)
        utils.img_draw_dot(img, (c[0], c[2]), color=(0,0,int((l_strength/2+0.5)*255)))
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
#        l_x = int(l_poly[0]*y**2+l_poly[1]*y+l_poly[2])
        r_x = int(r_poly[0]*y**2+r_poly[1]*y+r_poly[2])
 #       utils.img_draw_dot(img, (l_x, y), color=(0,255,255), radius=2)
  #      utils.img_draw_dot(img, (r_x, y), color=(0,255,255), radius=2)
        pts.append([r_x, y])
    for y in reversed(range(0,img_h,10)):
       l_x = int(l_poly[0]*y**2+l_poly[1]*y+l_poly[2])
       pts.append([l_x, y])



    cv2.fillPoly(img_overlay, np.array(pts, dtype=np.int32)[None,:], color=(0,100,0))

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
        if r_strength > strength_min:
            r_center_hint = r_center

    return window_centroids

avg_centroids = None

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
            avg_centroids[i][3] = min(lerp(a_c[3], 0, lerp_ratio/2),10)

        if a_c[4]<=strength_min:
            avg_centroids[i][1] = c[1]
            avg_centroids[i][4] = c[4]
        elif c[4]>strength_min:
            avg_centroids[i][1] = lerp(a_c[1], c[1], lerp_ratio)
            avg_centroids[i][4] = lerp(a_c[4], c[4], lerp_ratio)
        else:
            avg_centroids[i][4] = min(lerp(a_c[4], 0, lerp_ratio/2),10)

    return avg_centroids

def reset_state():
    global avg_centroids
    avg_centroids = None

def find_lanes(img):
    global avg_centroids
    strength_min = 1

    # window settings
    window_width = 80
    window_height = 10
    margin = 100           # How much to slide left and right for searching

    #cv2.imshow('find lanes',img*255)
    #cv2.waitKey()

    input_image = img
    if len(img.shape)==3:
        input_image = img[:,:,0]
    if np.max(input_image)==1:
        input_image *= 255

    window_size = (window_width, window_height)
    centroids = find_window_centroids(input_image, window_size, margin, strength_min=1)
    #pp = pprint.PrettyPrinter(indent=4)
    #pp.pprint(centroids)

    avg_centroids = lerp_centroids(centroids, strength_min, lerp_ratio=0.4)

    c = np.array(avg_centroids)

    lc = c[np.ix_(c[:,3]>strength_min, (0,2,3))]   # select l_x, l_y, l_strength, where l_strength > strength_min
    rc = c[np.ix_(c[:,4]>strength_min, (1,2,4))]   # select r_x, r_y, r_strength, where r_strength > strength_min

    # print('-----')
    # print(lc)
    # print('-----')
    # print(rc)

    l_poly = np.polyfit(lc[:,1], lc[:,0], 2)
    r_poly = np.polyfit(rc[:,1], rc[:,0], 2)
    #r_poly = np.polyfit(pts_y, r_pts_x, 2)

    # print('left poly:',l_poly)
    # print('right poly:',r_poly)

    img3 = np.dstack((input_image, input_image, input_image))

    img_centroids = draw_centroids(img3, avg_centroids, window_size, l_poly, r_poly)

    img_overlay =draw_overlay(img3, avg_centroids, window_size, l_poly, r_poly)

    return (img_centroids, img_overlay, l_poly, r_poly)

if __name__=='__main__':
    img = cv2.imread('{}/test5.jpg'.format(constants.persp_trans_test_folder))
    img_centroids, img_overlay, l_poly, r_poly = find_lanes(img)
    cv2.imshow('centroids', img_overlay)
    cv2.waitKey(3000)
