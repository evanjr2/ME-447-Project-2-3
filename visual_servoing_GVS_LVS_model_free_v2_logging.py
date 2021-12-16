#!/usr/bin/env python3import time
import serial
import json
import numpy as np
import cv2
import pyrealsense2 as rs
from pressure_to_arduino_v2 import Controller
from time import sleep,time
import csv
from cam2world import ML3,world2cam,Mtip,pix_to_omni,Mtip2
from visual_servoing_model_free_projection_v2 import VisualServoingFree
from estimate_global_desired_position import EstimateDesiredGlobal
from visual_servoing_desired_trajectory import CircleTrajectory
from loging_visual_servoing_experiments_v2 import LogData
from varname import nameof

from threading import Lock
frame_mutex = Lock()
frame_data = {"left"  : None,
              "right" : None,
              "timestamp_ms" : None
              }



def callback(frame):
    global frame_data
    if frame.is_frameset():
        frameset = frame.as_frameset()
        f1 = frameset.get_fisheye_frame(1).as_video_frame()
        f2 = frameset.get_fisheye_frame(2).as_video_frame()
        left_data = np.asanyarray(f1.get_data())
        right_data = np.asanyarray(f2.get_data())
        ts = frameset.get_timestamp()
        frame_mutex.acquire()
        frame_data["left"] = left_data
        frame_data["right"] = right_data
        frame_data["timestamp_ms"] = ts
        frame_mutex.release()

def get_image_frame():
    # Check if the camera has acquired any frames
    # frame_mutex.acquire()
    # valid = frame_data["timestamp_ms"] is not None
    # frame_mutex.release()
    # print(valid)
    vld = True
    frames = pipe.wait_for_frames()
    left = frames.get_fisheye_frame(1)
    left_data = np.asanyarray(left.get_data())
    right = frames.get_fisheye_frame(2)
    right_data = np.asanyarray(right.get_data())

    frame_data["left"] = left_data
    frame_data["right"] = right_data

    timestamp = frames.get_timestamp()

    return vld, frame_data, timestamp

# Declare RealSense pipeline, encapsulating the actual device and sensors
pipe = rs.pipeline()
rs.frame_queue(1)
cfg = rs.config()
profile = cfg.resolve(pipe)
dev = profile.get_device()
pipe.start(cfg)

num_readings = 50
num_markers = 5
# mag = MagSens("COM4",num_readings)    #MAAAG

Ard = Controller("COM3",250000)         # Initialize Controller                ### ARD

Pb = 650
Pr = 0
Ard.send_pressures(Pb, Pr)                                        ### ARD
u_DAC = np.array([Pb,250,250])
Ard.send_pressures_raw(u_DAC[0],u_DAC[1],u_DAC[2])                ### ARD

sleep(5)
pix = [0,0]

initial_pres = np.array([[800, 250, 250],
                         [1000, 250, 250],
                         [800, 250, 250],
                         [600, 250, 250],
                         [800, 250, 250],
                         [800, 500, 250],
                         [800, 250, 250],
                         [800, 250, 500],
                         [800, 250, 250],
                         [1000, 500, 250],
                         [800, 250, 250]])
BR2 = False
J0_G = np.array(
[[-3.07248041e-05,  5.48222564e-04],
 [-1.22536917e-03,  2.76868940e-04],
 [-3.98573874e-05, -1.10834055e-05],
 [-1.07521658e-04,  3.37058271e-05],
 [ 5.05163916e-06, -2.45250359e-05],
 [-2.07802292e-05,  4.40798479e-05]
])

J0_G = np.array([[0, 0],
                 [0, 0],
                 [0, 0],
 [-5.05163916e-06,  2.45250359e-05],
 [-1.07521658e-04, 3.37058271e-05],
 [-2.07802292e-05,  4.40798479e-05]
])



# J0_G = -np.array([[ 1.38210577e-03, -6.57916786e-04],
#        [ 1.08984024e-03, -6.43212213e-04],
#        [ 2.02518377e-04, -2.60015971e-04],
#        [ 3.49755644e-04, -2.82493698e-04],
#        [-3.92410796e-04,  2.14928382e-04],
#        [-8.37807017e-06,  9.37136961e-05]])

J0_L = np.array([  [ 1.27806112e-04, -1.28392688e-03],
                 [ 5.98041243e-05,  2.56023679e-03],
                 [ 3.81442871e-06, -4.68570715e-04],
                 [-2.36950534e-05, -7.37512716e-04],
                 [ 7.78951862e-05, -3.63458237e-04],
                 [ 1.23482781e-04, -1.29163436e-04]])

prev_ts = 0

marker = DetectMarker(2)

vis_serv_GVS = VisualServoingFree(J0 = -J0_G)
vis_serv_LVS = VisualServoingFree(J0 = "Jacobian_B3_4.json")#J0_L)#


u_bias = 250
softbot = "B3"
actuator = Actuator(softbot,u_bias)
cur_alpha = actuator.actuation_3D_t0_2D(u_DAC)


cam1 = cv2.VideoCapture(1)
frame_width = int(cam1.get(3))
frame_height = int(cam1.get(4))
desired_tip_trajectory = CircleTrajectory(np.array([frame_height,frame_width]),0,100)
des_target_pix_L = desired_tip_trajectory.get_new_desired_pixel()
p_dL = pix_to_omni(des_target_pix_L[::-1].reshape(2, 1), Mtip)
vis_serv_LVS.set_desired(p_dL)


length = 0.24
Xc_s = np.identity(4)
Xc_s[:3,3] = [-0.08,0,0.03]
Xc_s[:3,3] = [0.0,-0.035,0.005]
# Xc_s[:3,:3] = np.array([[0,1, 0],
#                         [-1,0,0],
#                         [0,0,1]])
est_desired = EstimateDesiredGlobal(length,Xc_s,softbot="B3")

marker.initialize_click()

dir = None
J = np.zeros((3,3))
step_time = .05

prev_time = time()
Rg_t = np.identity(3)

u = np.array([u_bias+1, u_bias, u_bias])
alpha1 = actuator.actuation_3D_t0_2D(u)

u = np.array([u_bias, u_bias+1, u_bias])
alpha2 = actuator.actuation_3D_t0_2D(u)

u = np.array([u_bias, u_bias, u_bias+1])
alpha3 = actuator.actuation_3D_t0_2D(u)


cur_target_prj_L = None
des_target_prj_L = None

cur_target_prj_G = None
cur_target_pos_G = None

cur_tip_prj_G = None
des_tip_pos_G = None
des_tip_prj_G = None


data_names = nameof(cur_target_prj_L, des_target_prj_L, cur_target_prj_G, cur_target_pos_G, cur_tip_prj_G,
                    des_tip_pos_G, des_tip_prj_G)
log = LogData("log_data/logged_data_",image_size=[(frame_width,frame_height),(848,800)],num_actuators=3,data_names=data_names,fps=1//step_time)

first = True
alpha = 1
d_alpha = 1e-2
alpha_limits = [0.2,1]
sign = -1
while True:

    # pose = mag.pose_measurement()     #MAAAG

    try:
        valid, frame_data,curr_ts = get_image_frame()
        if valid:

            # Hold the mutex only long enough to copy the stereo frames
            frame_copy = {"left": frame_data["left"].copy(),
                          "right": frame_data["right"].copy()}

            ret1, img1 = cam1.read()
            img2 = cv2.cvtColor(frame_copy["left"],cv2.COLOR_GRAY2BGR)
            if not ret1:  # or not ret2:
                print("failed to grab tip frame")
                break

            marker.detect([img1, img2],detect16=True)
            key = marker.draw_marker(direction=dir)

            if key == ord('q'):
                log.save_log()
                pipe.stop()
                Ard.send_pressures(0, 0)  ### ARD
                cv2.destroyAllWindows()
                print('\nGoodbye!')
                break


            curr_time = time()

            des_target_pix_L = desired_tip_trajectory.get_new_desired_pixel()
            des_target_prj_L = pix_to_omni(des_target_pix_L[::-1].reshape(2, 1), Mtip)
            vis_serv_LVS.set_desired(des_target_prj_L)
            marker.desired_pixel[0] = des_target_pix_L.reshape(2,1)

            if curr_time - prev_time > step_time:
                cur_target_prj_L = np.nan * np.ones((3,))

                cur_target_prj_G = np.nan * np.ones((3,))
                cur_target_pos_G = np.nan * np.ones((3,))

                cur_tip_prj_G = np.nan * np.ones((3,))
                des_tip_pos_G = np.nan * np.ones((3,))
                des_tip_prj_G = np.nan * np.ones((3,))

                target_in_L = False
                target_in_G = False
                tip_in_G = False
                in_control = 'N'

                tip_position, tip_rotation = marker.get_tip_Pose(1, ML3)
                cur_target_centers = marker.centers

                if len(cur_target_centers[0])>0:
                    cur_target_prj_L = pix_to_omni(np.array(cur_target_centers[0])[::-1].reshape(2, 1), Mtip)
                    depth = 0.1 # marker.depth
                    # marker.desired_pixel = None
                    target_in_L = True

                if len(cur_target_centers[1]) > 0:
                    cur_target_prj_G = pix_to_omni(np.array(cur_target_centers[1])[::-1].reshape(2, 1), ML3)
                    # cur_target_pos_G, target_rot = marker.get_tag_Pose(1, marker.chosen_tag, ML3)
                    # des_tip_pos_G = est_desired.get_estimated_position_2(cur_target_pos_G,)
                    if first:
                        des_tip_pos_G = est_desired.get_estimated_position_2(cur_target_prj_G,alpha = alpha)
                        # first = False
                        alpha += sign*d_alpha
                        print(alpha)
                        if sign*alpha > sign*alpha_limits[(sign+1)//2]:
                            sign *= -1
                    else:
                        est_desired.desired_dot = est_desired.get_error_2(cur_target_prj_G) + est_desired.desired_dot
                        des_tip_pos_G = est_desired.get_estimated_pos_inc(cur_target_prj_G,1e-1*np.identity(2),np.diag([1,1e-1]))

                    des_tip_prj_G = des_tip_pos_G/np.linalg.norm(des_tip_pos_G)
                    des_tip_pix_G = world2cam(des_tip_prj_G,ML3)
                    marker.desired_pixel[1] = des_tip_pix_G
                    target_in_G = True

                if tip_position is not None:
                    tip_depth = np.linalg.norm(tip_position)
                    cur_tip_prj_G = tip_position/tip_depth
                    tip_in_G = True



                if target_in_L:
                    marker.desired_pixel[1] =None
                    cur_alpha, LJ = vis_serv_LVS.control_law(1e2, cur_alpha, cur_target_prj_L.reshape(1, 3), cur_feature_depth=depth,
                                                         max_du=10, error_in_desired=0.05)
                    in_control = 'L'
                # elif tip_in_G and not np.all(np.isnan(vis_serv_GVS.feature_pos_desired)):
                elif tip_in_G and not np.all(np.isnan(des_tip_prj_G)):
                    vis_serv_GVS.set_desired(des_tip_prj_G)
                    cur_alpha, LJ = vis_serv_GVS.control_law(1e2, cur_alpha, cur_tip_prj_G.reshape(1, 3),
                                                             cur_feature_depth=tip_depth, max_du=10,
                                                             error_in_desired=0.004)
                    in_control = 'G'


                u_DAC = actuator.actuation_2D_t0_3D(cur_alpha)  # _DAC + 0.2*(u-u_DAC)
                print(u_DAC)
                u_DAC[u_DAC > 3000] = 3000
                Ard.send_pressures_raw(u_DAC[0], u_DAC[1], u_DAC[2])                ### ARD

                prev_time = curr_time
                data_save = [cur_target_prj_L.flatten(), des_target_prj_L.flatten(), cur_target_prj_G.flatten(), cur_target_pos_G.flatten(), cur_tip_prj_G.flatten(),
                             des_tip_pos_G.flatten(), des_tip_prj_G.flatten()]
                t = time()
                log.log_data([img1,img2],actuation=u_DAC,data_save=data_save,tag_id=marker.chosen_tag, in_control = in_control)
            # cv2.imshow("realsense", frame_copy["left"])

            if curr_ts == prev_ts:
                print("stoping pipeline ...")
                pipe.stop()
                sleep(10)
                pipe = rs.pipeline()
                # rs.frame_queue(1)
                # cfg = rs.config()
                # profile = cfg.resolve(pipe)
                # dev = profile.get_device()
                print("starting pipeline ...")
                pipe.start(cfg)#, callback)
                print("bismillah ...")

            prev_ts = curr_ts



    except KeyboardInterrupt:
        pipe.stop()
        log.save_log()
        Ard.send_pressures(0, 0)                ### ARD
        cv2.destroyAllWindows()
        print('\nGoodbye!')
        break




Ard.send_pressures(0, 0)                ### ARD

Ard.arduino.close()                ### ARD
# mag.close()   #MAAAG
