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
import csv
from display import display
from new_LK_customized import lucas_pyramidal
import pandas as pd
# from TTC_yolo import estimate_ttc
import time
# from My_TTC_optimized import calculate_intersection, calculate_foe_ransac, calculate_ttc


# ============================================= Importing Yolo =======================================================

yolo = YOLO('yolov8m.pt')
drone = None
frame_rate = 22.22
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


def calculate_foe_ransac(inter_pts, roi):
    # Check for empty or insufficient points early
    if len(inter_pts) < 1:
        return 0, 0 # Returning consistent format for foe and roi
    
    x1, y1, x2, y2 = roi
    foe_roi_width = abs(x2 - x1) // 3
    foe_roi_height = abs(y2 - y1) // 3
    sections = [[x1, y1, int(x1+ (2*foe_roi_width)), int(y1+ (2*foe_roi_height))],
                [int(x1 + foe_roi_width), int(y1), int(x1+ (3*foe_roi_width)), int(y1+ (2*foe_roi_height))],
                [int(x1), int(y1 + foe_roi_height), int(x1+ (2*foe_roi_width)), int(y1+ (3*foe_roi_height))],
                [int(x1 + foe_roi_width), int(y1 + foe_roi_height), int(x1+ (3*foe_roi_width)), int(y1+ (3*foe_roi_height))],
                ]

    # Pre-calculate sections to avoid redundancy in loops
    # sections = [
    #     (x, y, min(x + 2 * foe_roi_width, x2), min(y + 2 * foe_roi_height, y2))
    #     for x in range(int(x1), int(x2), int(foe_roi_width)) for y in range(int(y1), int(y2), int(foe_roi_height))
    # ][:16]  # Limit to first 9 sections if there are more

    inter_pts = np.array(inter_pts)  # Ensure points are in a numpy array for efficient operations

    # Vectorized computation of point counts in sections
    section_pt_len = np.array([
        np.sum((inter_pts[:, 0] >= sec_x1) & (inter_pts[:, 0] < sec_x2) &
               (inter_pts[:, 1] >= sec_y1) & (inter_pts[:, 1] < sec_y2))
        for sec_x1, sec_y1, sec_x2, sec_y2 in sections
    ])

    # Find the section(s) with the maximum number of points
    max_pts_indices = np.where(section_pt_len == section_pt_len.max())[0]
    max_sections = [sections[i] for i in max_pts_indices]

    min_error = float('inf')
    best_foe = [0, 0]
    best_section = (0, 0, 0, 0)

    # Evaluate each section with the maximum points for determining the best foe
    for section in max_sections:
        x1, y1, x2, y2 = section
        foe = [(x1 + x2) / 2, (y1 + y2) / 2]  # Center of the section

        section_points = inter_pts[
            (inter_pts[:, 0] >= x1) & (inter_pts[:, 0] < x2) &
            (inter_pts[:, 1] >= y1) & (inter_pts[:, 1] < y2)
        ]

        err = calculate_error(section_points, foe)
        if err < min_error:
            min_error = err
            best_section = section
            best_foe = foe

    return best_foe, best_section


def calculate_error(inter_points, center):
    # Convert inter_points to a numpy array if not already
    inter_points = np.array(inter_points)
    y0, x0 = center  # Assuming center is in the format [y, x]

    # Calculate distances using numpy broadcasting and vectorization
    distances = np.sqrt((inter_points[:, 0] - y0)**2 + (inter_points[:, 1] - x0)**2)

    # Calculate median distance
    median_distance = np.median(distances)

    return median_distance


def create_line(p1, p2):
    # print(p1.shape)
    # print(p1)
    x1, y1 = p1
    x2, y2 = p2
    if x2 - x1 == 0:  # Vertical line
        return [np.inf, x1]
    else:
        m = (y2 - y1) / (x2 - x1)  # Slope
        b = y1 - m * x1  # Intercept
        return [m, b]

def calculate_intersection(p1, p2):
    if len(p1) < 2 or len(p2) < 2:
        return []
    

    # Generate line parameters for each pair
    # print(p1[0])
    # print(thg)
    line_data = np.array([create_line(p1[i], p2[i]) for i in range(len(p1))])
    
    intersections = []
    for i in range(len(line_data)):
        for j in range(i + 1, len(line_data)):
            m1, b1 = line_data[i]
            m2, b2 = line_data[j]

            if m1 == m2:  # Parallel or identical lines
                continue

            if m1 == np.inf:  # First line is vertical
                x_inter = b1
                y_inter = m2 * x_inter + b2
            elif m2 == np.inf:  # Second line is vertical
                x_inter = b2
                y_inter = m1 * x_inter + b1
            else:
                x_inter = (b2 - b1) / (m1 - m2)
                y_inter = m1 * x_inter + b1

            intersections.append([x_inter, y_inter])

    return intersections

def calculate_ttc(foe, rel_vel, p1):
    foe = np.array(foe)
    p1 = np.array(p1[:,0])
    # print(np.shape(p1))
    dis = p1 - foe
    distances = np.sqrt((dis[:, 0]**2)+ (dis[:, 1]**2))
    rel_vel = np.array(rel_vel)
    c = rel_vel/ distances
    ttc_all = 1/c/frame_rate
    ttc = np.mean(ttc_all)

    return ttc

def calculate_rel_vel(p0, p1): # we work only with flow of points in the sqr foe 
    rel_vel = None    
    if len(p1) == 0:
        return -1
    flow_diff = p1 - p0
    flow_mag = np.linalg.norm(flow_diff, axis= 1)
    rel_vel = np.median(flow_mag)

    return rel_vel


# def estimate_ttc(p0, p1, box):
#     intersections = calculate_intersection(p0, p1)
#     if len(intersections) == 0:
#         return -1
#     foe, section = calculate_foe_ransac(intersections, box)
#     if foe == 0:
#         return -1
#     rel_vel = calculate_rel_vel(p0, p1)
#     ttc = calculate_ttc(foe, rel_vel, box)

#     return ttc





# ================================== Display fxn for boxes and ttc =============================

def display(frame, TTC, boxes, cl_id):
    frame1 = np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    for i in range(0, len(boxes)):
        x1, y1, x2, y2 = boxes[i, 0, 0]
        cl, id = cl_id[i, 0]
        # print(TTC[i])
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



# ================================ main algorithm ===============================================

def yolo_ttc(frame, prev_cl_id, prev_boxes, prev_gray, frame_rate):

    height, width, _ = np.shape(frame)

    # loop dependencies 
    flow = []
    prev_dist = []
    new_dist = []
    angles = []
    dist_diff = []
    new_posi = []
    results = yolo_box(frame)
    current_TTC = []
    sections = []
    if results == -1:
        print("No element to track")
    else:
        boxes, ids, clases = results


        if (len(prev_boxes) == 0) and (len(prev_cl_id) == 0):
            

            # =================================== frame to Gray ==================================================
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Gray image
            gray = np.array(gray)
            prev_gray = gray

            prev_boxes = np.array(boxes)
            prev_clases = clases
            prev_ids = ids
            # for i, j in zip(prev_clases, prev_ids):
            #     prev_cl_id.append([i[0][0], j[0][0]])
            for i in range(0, len(boxes)):
                id = ids[i][0]
                clas = clases[i][0]
                prev_cl_id.append([[clas, id]])
            prev_cl_id = np.array(prev_cl_id)

            return current_TTC, prev_cl_id, gray, prev_boxes, sections
        
        else:
            prev_posi = []
            new_boxes = np.array(boxes)
            new_cl_id = []
            
            # =================================== frame to Gray ==================================================
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Gray image
            gray = np.array(gray)



            for i in range(0, len(boxes)):
                if ids[i] == None:
                    id = 0
                else:
                    id = ids[i][0]
                clas = clases[i][0]
                new_cl_id.append([[clas, id]])
                # print(prev_cl_id)
                a = np.where((prev_cl_id[:, 0, 0] == clas) & (prev_cl_id[:, 0, 1] == id))
                if a[0].size > 0:
                    prev_posi.append(a[0][0])
                    new_posi.append(i)
            if len(new_posi) > 0:
                new_posi = np.array(new_posi)
                prev_posi = np.array(prev_posi)
                new_cl_id = np.array(new_cl_id)
                used_cl_id = new_cl_id[new_posi]

                roi_width = width // 3
                roi_height = height // 3
                rois = [(x, y, x + roi_width, y + roi_height) for x in range(0, width, roi_width) for y in range(0, height, roi_height)][:9]
                if len(new_posi) == len(prev_posi):
                    new_boxes_ = new_boxes[new_posi]
                    prev_boxes_ = prev_boxes[prev_posi]

                    # p0, p1, box_lens, box_used = lucas_pyramidal(prev_gray,gray,prev_boxes_, 3, 3, 7)
                        
                    # start = 0
                    # stop = 0
                    box_used = new_boxes_
                    print("===================")
                    for i, box in zip(range(0, len(box_used)),box_used):
                        # print(prev_boxes[i])
                        x11, y11, x12, y12 = box[0,0]
                        x01, y01, x02, y02 = prev_boxes_[i][0,0]
                        # new_area = abs((x12 - x11)*(y12 - y11))
                        # old_area = abs((x02 - x01)*(y02 - y01))
                        new_area = abs(y12 - y11)
                        old_area = abs(y02 - y01)
                        area_ratio = (new_area - old_area)/new_area

                        ttc = (1/area_ratio/frame_rate) 

                        # for visible plot 
                        if (ttc > 50):
                            ttc = 50
                        if ttc < -10:
                            ttc = -10  
                        # print(ttc)

                        current_TTC.append(ttc)

                        # Getting the position of the 
                        sec = []
                        center = [(x12-x11)/2, (y12-y11)/2]
                        for idx, (x11, y11, x12, y12) in enumerate(rois):
                            if x11 <= center[0] <= x12 and y11 <= center[1] <= y12:
                                sec.append(idx)

                        # sec = np.where((center[0] in np.arange(rois[:][0],rois[:][2])) and (center[1] in np.arange(rois[:][1],rois[:][3])))
                        sections.append(sec[0])

                # display(frame, current_TTC, used_box, used_cl_id)

                prev_cl_id = used_cl_id
                if len(used_cl_id) < 3:
                    prev_cl_id = new_cl_id

            prev_gray = gray
            prev_boxes = new_boxes

            return current_TTC, prev_boxes, gray, sections, prev_cl_id

# using video feed or camera
# cap.release()


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
    print(positions)
    if len(positions) < 1:
        return "No change"
    if len(positions) == 1:
        positn = section[positions]
    else:
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





# ================================ Using video file =============================================

# # Open the video file
# cap = cv2.VideoCapture('output1.mp4')

# # Check if the video file opened successfully
# if not cap.isOpened():
#     print("Error: Could not open video file")
#     exit()

# # Read and display frames until the video ends
# frame_count = 0
# while cap.isOpened():
#     # Read a frame from the video
#     ret, frame = cap.read()

#     # Check if the frame was read successfully
#     if ret:
#         # Display the frame
#         # cv2.imshow('Frame', frame)

# ================================ using files from a directory =================================

# # path = "C:/docs/IMC8/thesis/codes/TTC101/test2/frames2"
# path = "C:/docs/IMC8/thesis/codes/TTC101/test2/frames1"
# dir = sorted([f for f in os.listdir(path) if f.endswith(('.jpg','.png','.jpeg'))])
# frame_count = 0
# prev_cl_id = []
# prev_boxes = []
# prev_gray = None
# for _frame in dir:  # using the saved frames ( pictures)
#     frame = cv2.imread( path + "/"+_frame ) # Colored image 
#     frame_count += 1
#     print(frame_count)
    
#     # =================================== frame to Gray ==================================================
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Gray image
#     gray = np.array(gray)
#     if prev_gray is None:
#         prev_gray = gray
#         results = yolo_box(frame)
#         boxes, ids, clases = results
#         prev_boxes = np.array(boxes)
#         prev_clases = clases
#         prev_ids = ids
#         # for i, j in zip(prev_clases, prev_ids):
#         #     prev_cl_id.append([i[0][0], j[0][0]])
#         for i in range(0, len(boxes)):
#             id = ids[i][0]
#             clas = clases[i][0]
#             prev_cl_id.append([[clas, id]])
#         prev_cl_id = np.array(prev_cl_id)

#     else: 
#         # print(yolo_ttc(frame, prev_cl_id, prev_boxes, prev_gray, frame_rate))
#         ttc, cl_id, gray, boxes, sections = yolo_ttc(frame, prev_cl_id, prev_boxes, prev_gray, frame_rate)
#         print(ttc)
#         print(sections)
#         prev_cl_id, prev_gray, prev_boxes = cl_id, gray, boxes


# # function to smoothen the curves

# def moving_average(data, window_size):
#     # Create an array to hold the smoothed values
#     smoothed_data = np.zeros(len(data))
    
#     # Calculate the half window size to handle edges
#     half_window = window_size // 2
    
#     # Compute the moving average for each element
#     for i in range(len(data)):
#         # Handle the start of the array
#         if i < half_window:
#             smoothed_data[i] = np.mean(data[:i+half_window+1])
#         # Handle the end of the array
#         elif i >= len(data) - half_window:
#             smoothed_data[i] = np.mean(data[i-half_window:])
#         # Handle the middle of the array
#         else:
#             smoothed_data[i] = np.mean(data[i-half_window:i+half_window+1])
    
#     return smoothed_data



# def moving_av(data):
#     smoothed_data = data.copy()  # Create a copy to prevent modifying the original list
#     for i in range(1, len(data)-1):
#         if ((data[i-1] + 7) > data[i]):
#             smoothed_data[i] = (data[i-1] + data[i+1]) / 2
#         else:
#             # To handle the case where i is near the end of the list
#             upper_bound = min(i + 3, len(data)) 
#             smoothed_data[i] = sum(data[i:upper_bound]) / (upper_bound - i)
#     return smoothed_data


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

# window_size = 5



