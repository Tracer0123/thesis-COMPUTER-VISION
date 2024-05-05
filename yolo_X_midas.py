"""
------ EMMANUEL ASAH ------
------ BEng Thesis THWS in Schweinfurt -----
------ WS 2023/24 -----

"""

from ultralytics import YOLO
import cv2
import torch
import math
import numpy as np
import os , sys
import matplotlib.pyplot as plt
from display import display
from pyr_lucas_kanade import lucas_pyramidal
import time
import shutil
from scipy.interpolate import RectBivariateSpline
import csv
import pandas as pd

# Download the MiDaS
midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
midas.to('cpu')
midas.eval()
# Input transformation pipeline
transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = transforms.small_transform

# Initialize prev_output holder
prev_out = None
prev_out_norm = None
prev_depth1 = 0.0
alpha = 0.2
frame_rate = 11.11*1.5
depth_scale = 1



# ============================================= Importing Yolo =======================================================

yolo = YOLO('yolov8m.pt')
drone = None
def yolo_box(img):
    box_positions = []
    ids_poistions = []
    clas_positions = []
    result = yolo.track(
        img, #image to be evaluated
        persist= True,
        # show= True,
        # classes= 0, #only people class detected
        conf= 0.55, #Confidence score allowed
        tracker = "bytetrack.yaml" # the tracker used ### alternative tracker "botsort.yaml"
    )
    for r in result:

        boxes = r.boxes.cpu().numpy()

        if not r:
            return -1
        else:
            for box in boxes:
                # get box coordinates in (top, left, bottom, right) format
                b = box.xyxy
                id = box.id
                clas = box.cls
                clas_positions.append(clas)
                box_positions.append([b])
                ids_poistions.append(id)

    return box_positions, ids_poistions, clas_positions


# ================================== Display fxn for boxes and ttc =============================

def display(frame, TTC, boxes, cl_id):
    frame1 = np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    for i in range(0, len(boxes)):
        x1, y1, x2, y2 = boxes[i, 0, 0]
        cl, id = cl_id[i, 0]
        ttc = round(TTC[i], 3)
        cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(255, 0, 0), 2)
        cv2.putText(frame, str(i+1), (int(x1+2), int(y1+3)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"{i+1} : {str(ttc)} S, ID: {id}, CL: {cl}", (int(20), int(40 + i*30)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # cv2.putText(frame, str(cl), (int(x1), int(y1-3)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # cv2.putText(frame, str(id), (int(x1 + 7), int(y1-3)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Annotated frame", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# =================================== Section determining function =============================

def getIndex(list, num):
    for idx, val in zip(range(1,10),list):
        if(int(num[0]) in range(val[0],val[1])) and (int(num[1]) in range(val[2],val[3])):
            return idx
    return -1

# ================================== Drone control function ====================================

def move(sect):
    if sect < 3:
        drone.move_right(10)
    if sect == 4:
        drone.move_down(10)
    if (sect == 3) or (sect == 5) or (sect == 6) or (sect == 9):
        drone.move_up(10)
    if (sect == 8) or (sect == 7):
        drone.move_left(10)

# =================================== depth exponential moving alverage filter =================

def apply_ema_filter(current_depth):
    global prev_depth1
    filtered_depth = (alpha*current_depth) + ((1-alpha)*prev_depth1)
    prev_depth1 = filtered_depth
    return filtered_depth


# ================================ depth to distance =======================================

def depth_to_dis(depth_val, depth_scale):
    return 1/(depth_scale*depth_val)



# ================================ using files from a directory =================================

# # path = "C:/docs/IMC8/thesis/codes/TTC101/test2/frames2"
# path = "C:/docs/IMC8/thesis/codes/TTC101/test2/frames1"
# dir = sorted([f for f in os.listdir(path) if f.endswith(('.jpg','.png','.jpeg'))])
# frame_count = 0
# for _frame in dir:  # using the saved frames ( pictures)
#     frame = cv2.imread( path + "/"+_frame ) # Colored image 
    
# ================================ main algorithm ===============================================
    # loop dependencies 

def TTC_midas(frame, prev_cl_id, prev_boxes, prev_out_norm):

    timer = time.time()
    prev_dist = []
    new_dist = []
    angles = []
    dist_diff = []
    new_posi = []
    results = yolo_box(frame)

    # Transform input for midas
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgbatch = transform(img).to('cpu')

    # Make a prediction
    with torch.no_grad():
        prediction = midas(imgbatch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size = img.shape[:2],
            mode='bicubic',
            align_corners=False
        ).squeeze()

        output = prediction.cpu().numpy()
        output_norm = cv2.normalize(output, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        if results == -1:
            print("No Trackable")
        else:
            boxes, ids, clases = results
            

            if len(prev_boxes) == 0:

                # =================================== frame to Gray ==================================================
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Gray image
                gray = np.array(gray)
                prev_gray = gray

                prev_boxes = np.array(boxes)
                prev_clases = clases
                prev_ids = ids

                for i in range(0, len(boxes)):
                    print(id)
                    id = ids[i][0]
                    clas = clases[i][0]
                    prev_cl_id.append([[clas, id]])
                prev_cl_id = np.array(prev_cl_id)
                prev_out = output
                prev_out_norm = output_norm
            
            else:
                prev_posi = []
                new_boxes = np.array(boxes)
                new_cl_id = []
                
                # =================================== frame to Gray ==================================================
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Gray image
                gray = np.array(gray)


                # preparing sections for analysis
                height, width, _ = frame.shape
                # frame1 = frame
                roi_width = width // 3
                roi_height = height // 3
                # Create a list of ROIs in the format [x0, x1, y0, y1]
                rois = [[x,x + roi_width, y, y + roi_height] for x in range(0, width, roi_width) for y in range(0, height, roi_height)][:9]

                h, w = output_norm.shape
                x_grid = np.arange(w)
                y_grid = np.arange(h)
                spline = RectBivariateSpline(y_grid, x_grid, output_norm)
                prev_spline = RectBivariateSpline(y_grid, x_grid, prev_out_norm)
                for i in range(0, len(boxes)):
                    if ids[i] == None:
                        id = 0
                    else:
                        id = ids[i][0]
                    clas = clases[i][0]
                    new_cl_id.append([[clas, id]])
                    a = np.where((prev_cl_id[:, 0, 0] == clas) & (prev_cl_id[:, 0, 1] == id))
                    if a[0].size > 0:
                        prev_posi.append(a[0][0])
                        new_posi.append(i)
                if len(new_posi) > 0:
                    new_posi = np.array(new_posi)
                    prev_posi = np.array(prev_posi)
                    new_cl_id = np.array(new_cl_id)
                    used_cl_id = new_cl_id[new_posi]

                    sections = []
                    centers = []
                    prev_box_depth = []
                    new_box_depth = []
                    if len(new_posi) == len(prev_posi):
                        new_boxes_ = new_boxes[new_posi]
                        prev_boxes_ = prev_boxes[prev_posi]
                        positiv_posi = []
                        current_TTC = []
                        sections = []

                        for i in range(0, len(new_posi)):
                            x01, y01, x02, y02 = prev_boxes_[i, 0, 0]
                            x11, y11, x12, y12 = new_boxes_[i, 0, 0]
                            p01 = [x01, y01]
                            p02 = [x02, y02]
                            p11 = [x11, y11]
                            p12 = [x12, y12]
                            prev_dist.append(math.sqrt((x02 - x01)**2 + (y02 - y01)**2))
                            new_dist.append(math.sqrt((x12 - x11)**2 + (y12 - y11)**2))
                            diff = new_dist[i] - prev_dist[i]
                            mid_x = (x12 + x11)/2
                            mid_y = (y12 + y11)/2
                            prev_mid_x = (x02 + x01)/2
                            prev_mid_y = (y02 + y01)/2


                            # ============= depth management2 =======================

                            depth_mid_filt = spline(mid_y, mid_x)
                            depth = depth_to_dis(depth_mid_filt, depth_scale)
                            depth_mid_filt = (apply_ema_filter(depth)/10)[0][0]
                            depth_mid_filt = round(depth_mid_filt, 3)

                            prev_depth_mid_filt = prev_spline(prev_mid_y, prev_mid_x)
                            prev_depth = depth_to_dis(prev_depth_mid_filt, depth_scale)
                            prev_depth_mid_filt = (apply_ema_filter(prev_depth)/10)[0][0]
                            prev_depth_mid_filt = round(prev_depth_mid_filt, 3)

                            # ============= depth management ========================
                            # now_out = output[int(x11):int(x12), int(y11):int(y12)].flatten()
                            # prev_out = prev_depth[int(x01):int(x02), int(y01):int(y02)].flatten()
                            # # print(now_out.shape)
                            # box_depth = np.median(now_out)
                            # max_depth = np.max(output)
                            # box_depth_old = np.median(prev_out)
                            # print(box_depth)
                            # prev_box_depth.append(box_depth_old)
                            # new_box_depth.append(box_depth)

                            relative_vel = abs(depth_mid_filt-prev_depth_mid_filt)/(4/frame_rate)
                            ttc = depth_mid_filt/relative_vel
                            ttc = round(ttc, 3)
                            if ttc > 50:
                                ttc = 50

                            print(ttc)
                            print("===================")
                            sec = []
                            center = [(x12-x11)/2, (y12-y11)/2]
                            idx = 1
                            for [x11, x12, y11, y12] in rois:
                                if (x11 <= center[0]) and (center[0] <= x12):
                                    if (y11 <= center[1]) and ( center[1] <= y12):
                                        sec.append(idx)
                                idx += 1

                            # sec = np.where((center[0] in np.arange(rois[:][0],rois[:][2])) and (center[1] in np.arange(rois[:][1],rois[:][3])))
                            sections.append(sec[0])


                            current_TTC.append(ttc)                                             


                    prev_ttc = current_TTC
                    # print(used_cl_id)
                    prev_cl_id = used_cl_id
                    if len(used_cl_id) < 3:
                        prev_cl_id = new_cl_id

            return current_TTC, prev_boxes, prev_out_norm, sections, used_cl_id

commands = {
            0: "move_forward",
            1: "move_back",
            2: "move_up",
            3: "move_down",
            4: "move_left",
            5: "move_right",
            6: "rotate_5",
            7: "rotate_-5",
            8: "rotate_180"
            }

def decision_sys(result1, result2, result3, section):
    global commands
    # Assuming commands is a list of commands to control the drone or device
    result1, result2, result3 = np.array(result1), np.array(result2), np.array(result3)
    
    diff_ab = np.abs(result1 - result2)
    diff_bc = np.abs(result2 - result3)
    positions = []
    for i in range(len(result1)):
        if (result1[i]>0 and result1[i]<5) and (result3[i]>0 and result3[i]<= 3) and (diff_bc[i]<= 1): 
            positions.append(i)

    # positions = np.where((result1[:] > 0) & (result1[:] < 10) &
    #                  (result3[:] > 0) & (result3[:] <= 3) & 
    #                  (diff_bc[:] <= 1.5), 
    #                  diff_bc, 1000)
    
    positions = np.array(positions)
    
    positn = None
    if len(positions) < 1:
        return "No change"
    if len(positions) == 1:
        print(positions)
        positn = section[positions[0]]
    if len(positions) > 1:
        value = diff_bc[positions]
        section = section[positions]
        minimum = np.min(value)
        positn0 = np.where(diff_bc[:] == minimum)  # Simplified position finding
        positn = section[positn0]
    print(positn)
    print("=============")
    
    # Decision-making based on position
    if positn < 3:
        command = commands[6]  # Example: rotate_clockwise(-5)
    elif (positn == 3) or (positn == 6) or (positn == 9):
        command = commands[3]  # Example: move_down(10)
    elif positn == 4:
        command = commands[2]  # Example: move_up(10)
    elif positn == 5:
        command = commands[1]  # Example: rotate_clockwise(5)
    elif (positn == 7) or (positn == 8):
        command = commands[7]
    else:
        command = None  # No command or default command if positn doesn't match any condition
    
    return command





    # Keep the plot open until closed by the user
    # Set up the plot
    # plt.figure()


# # path = "C:/docs/IMC8/thesis/codes/TTC101/test2/frames2"
# path = "C:/docs/IMC8/thesis/codes/TTC101/test2/frames1"
# dir = sorted([f for f in os.listdir(path) if f.endswith(('.jpg','.png','.jpeg'))])
# frame_count = 0
# y_vals = {}
# prev_cl_id = []
# prev_boxes = []
# prev_out_norm = None
# prev_sections = None
# for _frame in dir:  # using the saved frames ( pictures)
#     frame = cv2.imread( path + "/"+_frame ) # Colored image 
#     frame_count += 1
#     print(frame_count)
    
#     # =================================== frame to Gray ==================================================
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Gray image
#     gray = np.array(gray)
#     if prev_out_norm is None:
#         results = yolo_box(frame)
#         if results == -1:
#             print("No Trackable")
#         else:
#             boxes, ids, clases = results
#             prev_boxes = np.array(boxes)
#             prev_clases = clases
#             prev_ids = ids
#             # Transform input for midas
#             img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             imgbatch = transform(img).to('cpu')

#             # Make a prediction
#             with torch.no_grad():
#                 prediction = midas(imgbatch)
#                 prediction = torch.nn.functional.interpolate(
#                     prediction.unsqueeze(1),
#                     size = img.shape[:2],
#                     mode='bicubic',
#                     align_corners=False
#                 ).squeeze()

#                 output = prediction.cpu().numpy()
#                 output_norm = cv2.normalize(output, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#             prev_out_norm = output_norm
#             for i in range(0, len(boxes)):
#                 id = ids[i][0]
#                 clas = clases[i][0]
#                 prev_cl_id.append([[clas, id]])
#             prev_cl_id = np.array(prev_cl_id)

#     else: 
#         # print(yolo_ttc(frame, prev_cl_id, prev_boxes, prev_gray, frame_rate))
#         ttc, boxes, out_norm, sections, cl_id = TTC_midas(frame, prev_cl_id, prev_boxes, prev_out_norm)
#         print(ttc)
#         prev_cl_id, prev_sections, prev_boxes, prev_out_norm = cl_id, gray, boxes, out_norm

#         _, id_ = prev_cl_id[i][0]
#         if id_ in x_vals:
#             y_vals[id_].append(ttc)
#         else:
#             y_vals[id_] = []
#             y_vals[id_].append(ttc)
# print(y_vals)




# # Eliminating negative numbers
# def negativ_average(data):
#     for i in range(len(data)):
#         if i == 0:
#             if data[i]<0:
#                 data[i] = data[i+1]/2
#         elif i == (len(data)-1):
#             if data[i]<0:
#                 data[i] = data[i-1]/2
#         else:
#             if data[i]<0:
#                 data[i] = (data[i+1] + data[i-1])/2
#     return data
# for key in y_vals:
#     # Convert the dictionary to a DataFrame
#     df = pd.DataFrame(y_vals[key])
#     # Write the DataFrame to a CSV file
#     df.to_csv(f"exvelfiles/lab_data/04_16/midas/midas_frame_1_y{key}.csv", index=False)

# for key in x_vals:
#     df = pd.DataFrame(x_vals[key])
#     # Write the DataFrame to a CSV file
#     df.to_csv(f"exvelfiles/lab_data/04_16/midas/midas_frame_1_x{key}.csv", index=False)

# for key in x_vals:
#     y_vals[key] = negativ_average(y_vals[key])
# # Plot each set in the dictionaries
# for key in x_vals:
#     plt.plot(x_vals[key], y_vals[key], label=key)

# # Add labels and legend
# # plt.set_ylim(-5, 50)
# plt.title('Plotting of Yolo and Midas')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.legend()

# # Display the plot
# plt.show()
# # print(TTC_cl_id)
            
            


