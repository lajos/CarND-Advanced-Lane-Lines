import os, pickle, glob, csv
import constants
import cv2
import matplotlib.pyplot as plt

def make_dir(dir_name):
    """ make [dir_name] if doesn't exist, raise error if couldn't create"""
    try:
        os.makedirs(dir_name)
    except OSError:
        if not os.path.isdir(dir_name):
            raise

def basename(path):
    """return only file_name from [path]"""
    return os.path.basename(path)

def warning(message):
    clear_line = ' '*80
    print('\r%s' % (clear_line), end = '\r')
    print('WARNING: {}'.format(message))

def save_globals(globals):
    with open(constants.globals_file, 'wb') as f:
        pickle.dump(globals, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_globals():
    globals = {}
    try:
        with open(constants.globals_file, mode='rb') as f:
            globals = pickle.load(f)
    except:
        pass
    return globals

def display_image(img, timeout=None):
    cv2.imshow('image', img)
    if timeout:
        cv2.waitKey(timeout)
    else:
        cv2.waitKey()


def display_image_file(file_name, timeout=None):
    img = cv2.imread(file_name)
    display_image(img, timeout)

def plt_image(img, bgr=False):
    if bgr:
        plt.imshow(img[:,:,[2,1,0]])
    else:
        plt.imshow(img)
    plt.show()

def plt_image_file(file_name, bgr=False):
    img = cv2.imread(file_name)
    plt_image(img, bgr)

def read_folder_images(folder_name):
    images = []
    image_names = glob.glob('{}/*.jpg'.format(folder_name)) + glob.glob('{}/*.png'.format(folder_name))
    for i in image_names:
        images.append(cv2.imread(i))
    return images

def img_draw_poly(img, pts, color=(255,255,255), thickness=2):
    cv2.polylines(img, pts, True, color, thickness)

def img_draw_line(img, pt1, pt2, color=(255,255,255), thickness=2):
    cv2.line(img, pt1, pt2, color, thickness)

def img_draw_rectangle(img, pt1, pt2, color=(255,255,255), thickness=2):
    cv2.rectangle(img, pt1, pt2, color, thickness)

def img_draw_dot(img, center, radius=4, color=(255,255,255)):
    cv2.circle(img, center, radius, color, thickness=-1)

def img_draw_grid(img, n_width=16, n_height=9, color=(255,255,255)):
    w=img.shape[1]
    h=img.shape[0]
    x_step = w/n_width
    y_step = h/n_height
    for i in range(1,n_width):
        x = int(i*y_step)
        img_draw_line(img,(x,0),(x,h),color,thickness=2)
    for i in range(1,n_height):
        y = int(i*y_step)
        img_draw_line(img,(0,y),(w,y),color,thickness=2)

def print_progress_bar (iteration, total, prefix = 'progress:', suffix = ' ', decimals = 1, length = 30, fill = '='):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '.' * (length - filledLength)
    print('\r%s [%s] %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    if iteration == total:
        print()

def read_csv(csv_name):
    data = []
    with open(csv_name, newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            data.append(row)
    return data


def write_csv(csv_name, data):
    with open(csv_name, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        for d in data:
            csv_writer.writerow(d)
