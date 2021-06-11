from ctypes import *
import math
import random
import cv2
import time
import pyrealsense2 as rs
import struct
import redis
import numpy as np
import json
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

    

#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
lib = CDLL("/home/ur5e/darknet/libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    im = load_image(image, 0, 0)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res

def toRedis(r, img, key):
    """Store given Numpy array 'a' in Redis under key 'n'"""
    h, w = img.shape[:2]
    shape = struct.pack('>II',h,w)
    encoded = shape + img.tobytes()

    # Store encoded data in Redis
    r.set(key, encoded)
    return

def send_object_list(redi, result):
    dict={}
    # for re in result:
    #     dict[re[0]] = {'bounding_box':[0, 0, 0, 0], 'position': [-0.3, 0.15, -0.95]}
    #     dict[re[0]+"2"] = {'bounding_box':[0, 0, 0, 0], 'position': [-0.4, 0.15, -0.95]}
    #     dict[re[0]+"3"] = {'bounding_box':[0, 0, 0, 0], 'position': [-0.5, 0.15, -0.95]}
    for idx, item in enumerate(result):
        dict[item[0]+str(idx)] = {'bounding_box':[0, 0, 0, 0], 'position': [-0.3, 0.15, -0.95]}

    if dict: # not empty
        redi.hset('object_list', 'one', json.dumps(dict))

def send_object_list_rs(redi, result, camera_coordinate):
    dict={}
    for idx, item in enumerate(result):
        # dict[item[0]+str(idx)] = {'bounding_box':[0, 0, 0, 0], 'position': camera_coordinate[idx]}
        dict[item[0]] = {'bounding_box':[0, 0, 0, 0], 'position': camera_coordinate[idx]}

    if dict: # not empty
        redi.hset('object_list', 'one', json.dumps(dict))


def convert_depth_to_phys_coord_using_realsense(x, y, depth, cameraInfo):
  _intrinsics = rs.intrinsics()
  _intrinsics.width = cameraInfo.width
  _intrinsics.height = cameraInfo.height
  _intrinsics.ppx = cameraInfo.K[2]
  _intrinsics.ppy = cameraInfo.K[5]
  _intrinsics.fx = cameraInfo.K[0]
  _intrinsics.fy = cameraInfo.K[4]
  #_intrinsics.model = cameraInfo.distortion_model
  _intrinsics.model  = rs.distortion.none
  _intrinsics.coeffs = [i for i in cameraInfo.D]
  result = rs.rs2_deproject_pixel_to_point(_intrinsics, [x, y], depth)
  #result[0]: right, result[1]: down, result[2]: forward
#   return result[2], -result[0], -result[1]
  return result


bridge = CvBridge()
def ros_img_callback(data):
    global ros_image_message
    try:
      ros_image_message = bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

def ros_depth_callback(data):
    global ros_depth_message
    try:
      ros_depth_message = bridge.imgmsg_to_cv2(data)
    except CvBridgeError as e:
      print(e)

def ros_depth_camera_info_callback(data):
    global ros_aligned_depth_info
    ros_aligned_depth_info = data


if __name__ == "__main__":

    rospy.init_node('object_dectection', anonymous=True)
    image_sub = rospy.Subscriber("/camera/color/image_raw", Image, ros_img_callback)
    depth_sub = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, ros_depth_callback)
    depth_camera_info_sub = rospy.Subscriber("/camera/aligned_depth_to_color/camera_info", CameraInfo, ros_depth_camera_info_callback)
    # Redis connection
    pool = redis.ConnectionPool(host='127.0.0.1',port=6379)
    redi = redis.Redis(connection_pool=pool)
    # redi = redis.Redis(host='localhost', port=6379, db=0)
    # cap = cv2.VideoCapture("nba2.mp4")
    # net = load_net("data_collection/picking.cfg", "data_collection/picking_10000.weights", 0)
    # meta = load_meta("data_collection/picking.data")
    net = load_net("config/yolov3-mydata.cfg", "config/yolov3-mydata_60000.weights", 0)
    meta = load_meta("config/mydata.data")

    while not rospy.is_shutdown():
        time_last = time.time()
        frame = ros_image_message
        cv2.imwrite('check.jpg',frame)
        result = detect(net, meta, "check.jpg")
        camera_coordinate = []

        for idx, r in enumerate(result):
            # print(result)
            # print (r[0])
            x = int(r[2][0])
            y = int(r[2][1])
            w = int(r[2][2])
            h = int(r[2][3])
            top_left_x = int(x-w/2)
            top_left_y = int(y-h/2)
            cv2.rectangle(frame,(top_left_x,top_left_y),(top_left_x+w,top_left_y+h),(0,255,0),2)
            cv2.putText(frame,r[0]+str(idx),(top_left_x+10,top_left_y-20),0,0.9,(0,255,0))

            aligned_depth_frame = ros_depth_message
            dis = aligned_depth_frame[y, x]/1000.0
            #print("x:{} y:{} z:{}".format(x,y,dis))
            # point_camera = rs.rs2_deproject_pixel_to_point(intrin=depth_intrin, pixel=[x, y], depth=dis)
            # point is described in camera_color_optical_frame
            point_camera = convert_depth_to_phys_coord_using_realsense(x, y, dis, ros_aligned_depth_info)
            camera_coordinate.append(point_camera)
        #cv2.imshow('yolo', frame)
        #print(time.time() - time_last)
        # print(camera_coordinate)
        send_object_list_rs(redi, result, camera_coordinate)
        toRedis(redi, frame, 'image')

        if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    rospy.spin()
    # cap.release()

