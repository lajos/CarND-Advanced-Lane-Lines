import glob
import numpy as np
import cv2
import utils

def calibrate_images(image_names, nx=9, ny=6):
    """ find checkerboard points in [images_names]
    [nx] and [ny] are the number of points in x,y directions
    returns (mtx,dist)
    """

    # create object points (0,0,0), (1,0,0), (2,0,0), ... (nx, ny, 0)
    objp = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

    obj_points = [] # 3d points in world space
    img_points = [] # 2d points in image plane

    img_shape = ()

    for i in image_names:
        img = cv2.imread(i)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[:2]
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

        if ret == True:
            obj_points.append(objp)
            img_points.append(corners)
            cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
            # cv2.imshow('img', img)
            # cv2.waitKey(500)
        else:
            utils.warning('checkerboard corners not found: {}'.format(i))

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1],None,None)

    return (mtx, dist)

def calibrate_folder(folder_name, nx=9, ny=6):
    """ find checkerboard points in jpeg images from a [folder]
    [nx] and [ny] are the number of checkerboard points in x,y directions
    returns (mtx,dist)
    """
    images=[]
    for i in glob.glob('{}/*.jpg'.format(folder_name)):
        images.append(i)
    return calibrate_images(images)

def undistort_image(img, mtx, dist):
    """ undistort [img] based on [mtx] and [dist]
    returns undistorted image
    """
    return cv2.undistort(img, mtx, dist, None, mtx)

def test_folder(folder_name, mtx, dist, output_folder=None):
    """ show undistorted images from [folder], optionally save to [output_folder]"""
    if output_folder:
        utils.make_dir(output_folder)
    for i in glob.glob('{}/*.jpg'.format(folder_name)):
        img = cv2.imread(i)
        img = undistort_image(img, mtx, dist)
        if output_folder:
            out_name = '{}/{}'.format(output_folder, utils.basename(i))
            cv2.imwrite(out_name, img)
        else:
            cv2.imshow('img', img)
            cv2.waitKey(500)
