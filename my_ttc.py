"""
------ EMMANUEL ASAH ------
------ BEng Thesis THWS in Schweinfurt -----
------ WS 2023/24 -----

"""


# import dependencies 
import cv2
import numpy as np
# from display import display
import os , sys
from pyr_lucas_kanade import lucas_pyramidal
import matplotlib.pyplot as plt
import pandas as pd


    # Initialize featur parameters and Luca-Kanade parameters
feature_params = dict(maxCorners=300, qualityLevel=0.3, minDistance=7, blockSize=7)
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
quater_sections = []


def calculate_foe_ransac(inter_pts, roi, whole_im):
    # Check for empty or insufficient points early
    x1, y1, x2, y2 = roi
    global quater_sections
    if whole_im:
        sections = quater_sections
    else:
        foe_roi_width = abs(x2 - x1) // 4
        foe_roi_height = abs(y2 - y1) // 4

        # Pre-calculate sections to avoid redundancy in loops
        sections = [
            (x, y, min(x + 2 * foe_roi_width, x2), min(y + 2 * foe_roi_height, y2))
            for x in range(x1, x2, foe_roi_width) for y in range(y1, y2, foe_roi_height)
        ][:16]  # Limit to first 9 sections if there are more

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

    # Calculate mean or median distance
    # median_distance = np.median(distances)
    median_distance = np.mean(distances)

    return median_distance



def create_line(p1, p2):
    # print(p1[0])
    y1, x1 = p1[0]
    y2, x2 = p2[0]
    if x2 - x1 == 0:  # Vertical line
        return [np.inf, x1]
    else:
        m = (y2 - y1) / (x2 - x1)  # Slope
        b = y1 - m * x1  # Intercept
        return [m, b]

def calculate_intersection(p1, p2):
    if len(p1) == 0 or len(p2) == 0:
        return []

    # Generate line parameters for each pair
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

            intersections.append([y_inter, x_inter])

    return intersections

def calculate_rel_vel(flow, indices): # we work only with flow of points in the sqr foe 
    rel_vel = None    
    if indices == None:
        return -1
    mag = flow[indices]
    rel_vel = abs(np.mean(mag))

    return rel_vel



def calculate_ttc(foe, rel_vel, p1, frame_rate):
    global frame_count
    global frame_rate_stamp
    foe = np.array(foe)
    p1 = np.array(p1[:,0])
    # print(np.shape(p1))
    dis = p1 - foe
    distances = np.sqrt((dis[:, 0]**2)+ (dis[:, 1]**2))
    rel_vel = np.array(rel_vel)
    c = rel_vel/ distances
    # ttc_all = 1/c/frame_rate_stamp[frame_count-1] # when using time stamps
    ttc_all = 1/c/frame_rate 
    ttc = np.mean(ttc_all)

    return ttc


def find_TTC(frame1,frame, prev_gray, frame_rate):

    global frame_count

    # preparing sections for analysis
    height, width = frame.shape
    # frame1 = frame
    roi_width = width // 3
    roi_height = height // 3
    global quater_sections
    # Create a list of ROIs
    rois = [(x, y, x + roi_width, y + roi_height) for x in range(0, width, roi_width) for y in range(0, height, roi_height)][:9]

    x1, y1, x2, y2 = rois[0]
    foe_roi_width0 = abs(x2 - x1) // 3
    foe_roi_height0 = abs(y2 - y1) // 3
    # Pre-calculate sections to avoid redundancy in loops
    quater_sections = [
        (x, y, min(x + 2 * foe_roi_width0, x2), min(y + 2 * foe_roi_height0, y2))
        for x in range(x1, width, foe_roi_width0) for y in range(y1, height, foe_roi_height0)
    ][:64]
    # print(np.shape(quater_sections))
    # print(rois)

    gray = frame

    # flow_ , p1_ = optical_flow_roi(gray, prev_gray, prev_point)

    p0_1, p1_1 = lucas_pyramidal(prev_gray,gray, 3, 3, 7)

    # Cleaning the flows
    # flow_mag = np.sqrt((flow_[:, 0, 0]**2) + (flow_[:, 0, 1]**2))
    flow_diff = p1_1 - p0_1
    flow_mag = np.linalg.norm(flow_diff, axis= 1)
    flow = np.array([])
    p1 = []
    p0 = []
    for i,j,k in zip(flow_mag, p1_1, p0_1):
        if i < roi_width/2:
            flow = np.append(flow, i)
            p1.append([[j[0], j[1]]]) # using open cv lkflow p1.append([[j[0,0], j[0,1]]])
            p0.append([[k[0], k[1]]]) # using open cv lkflow p0.append([[k[0,0], k[0,1]]])
    prev_point = np.array(p0)
    p1 = np.array(p1)

    # Alternative cleaning for speed because flow is already cleaned

    # flow_mag = np.sqrt((p1_1[:,0]-p0_1[:,0])**2 + (p1_1[:,1]-p0_1[:,1])**2)
    # prev_point = np.array(p0_1)
    # p1 = np.array(p1_1)

    # Process each ROI sequentially
    sectionNum = 0
    TTC_all = []
    foe_all = []
    rel_vel_all = []
    place_hoder = 0
    points_of_inter = np.array([[0,0]])
    section_out_all = []


# ======================= entire image configuration ===========================================

    # inter = np.array(calculate_intersection(prev_point, p1))
    # roi = [0,0,0,0]
    # foe, section_out = calculate_foe_ransac(inter, roi, whole_im = True)
    # rel_vel = flow
    # if len(rel_vel) < 1:
    #     rel_vel_all.append(-1)
    # else:
    #     rel_vel_all.append(rel_vel)

    # if foe == 0:
    #     place_hoder
    # else:
    #     if len(rel_vel) < 1:
    #         ttc = 11111 # sets ttc to -1 when flow is 0

    #     else:
    #         ttc = round(calculate_ttc(foe, rel_vel, p1), 4)
    #         # print(ttc)
    #         section_out_all.append(section_out)

    #     TTC_all.append(ttc)
    #     foe_all.append(foe)


# =============================== entire image config up ======================================
    sec_count = []
    frame_used_ = []
    for roi in rois:


        x1, y1, x2, y2 = roi
        indices_x = np.where((p1[:, 0, 0] >= x1) & (p1[:, 0, 0] < x2) & (p1[:, 0, 1] >= y1) & (p1[:, 0, 1] < y2))
        p1_roi = np.array(p1[indices_x])
        p0_roi = np.array(prev_point[indices_x])
        inter = np.array(calculate_intersection(p0_roi, p1_roi))
        # print("===========")
        # print(len(p0_roi))
        # print(len(inter))
        # print("===========")

        if len(inter) > 1:

            inter_indices = np.where((inter[:, 0] >= x1) & (inter[:, 0] < x2) & (inter[:, 1] >= y1) & (inter[:, 1] < y2))
            # print(inter_indices)
            inter_in_roi = inter[inter_indices]
            # points_of_inter = np.concatenate((points_of_inter, inter_in_roi), axis= 0)
            if len(inter_in_roi) < 2 and len(inter) > 5:
                foe, section_out = calculate_foe_ransac(inter, roi, whole_im = True)
            else:
                foe, section_out = calculate_foe_ransac(inter_in_roi, roi, whole_im = False)

            rel_vel = flow[indices_x]
            # rel_vel = calculate_rel_vel(flow, indices_x)
            if len(rel_vel) < 1:
                rel_vel_all.append(-1)
            else:
                rel_vel_all.append(rel_vel)

            if foe == 0:
                place_hoder
                
            else:
                if len(rel_vel) < 1:
                    ttc = -1 # sets ttc to -1 when flow is 0
                    sec_count.append(.5)
                    frame_used_.append(frame_count)

                else:
                    ttc = round(calculate_ttc(foe, rel_vel, p1_roi, frame_rate), 4)
                    # print(ttc)
                    section_out_all.append(section_out)
                    sec_count.append(-1)
                    # frame_used_.append(frame_count)

                TTC_all.append(ttc)
                foe_all.append(foe)
            
        else:
            TTC_all.append(-2)
            sectionNum += 1
            sec_count.append(0)
            # frame_used_.append(frame_count)

    prev_point = p1
    # prev_point = cv2.goodFeaturesToTrack(frame, mask=None, **feature_params)
    points_of_inter1 = points_of_inter.tolist()

    # display(frame1, rois, foe_all, TTC_all, p0, p1, points_of_inter1)

    # return TTC_all, foe_all, rel_vel_all
    return TTC_all, foe_all

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


def decision_sys(result1, result2, result3, commands):
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
    print(positions)
    if len(positions) < 1:
        return "No change"
    value = diff_bc[positions]
    print("============")
    print(value)
    print("============")

    minimum = np.min(positions)
    positn0 = np.argmin(value)  # Simplified position finding
    positn = np.where(diff_bc[:] == minimum)
    # positn = positions[positn0]
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



# ================================ using files from a directory =================================
TTC = {"sec_1": [],
        "sec_2": [],
        "sec_3": [],
        "sec_4": [],
        "sec_5": [],
        "sec_6": [],
        "sec_7": [],
        "sec_8": [],
        "sec_9": []
        }

frame_used = {"sec_1": [],
        "sec_2": [],
        "sec_3": [],
        "sec_4": [],
        "sec_5": [],
        "sec_6": [],
        "sec_7": [],
        "sec_8": [],
        "sec_9": []
        }

sec_used = {"sec_1": [],
        "sec_2": [],
        "sec_3": [],
        "sec_4": [],
        "sec_5": [],
        "sec_6": [],
        "sec_7": [],
        "sec_8": [],
        "sec_9": []
        }


# using this file only


# # frame rate
# frame_rate = 10
# # Load the data from CSV file
# df = pd.read_csv('fps_gt/T_gt_UE_3.csv', header=None)

# # Convert the single column to a list
# t_fps = df[0].tolist()


# # frame_rate = 5
# ttc_new = []
# prev_gray = None
# frame_rate_stamp = 10
# path = "C:/docs/IMC8/thesis/codes/TTC101/test2/frames1"
# # path = "C:/docs/IMC8/thesis/UE_images/UE_2"
# dir = sorted([f for f in os.listdir(path) if f.endswith(('.jpg','.png','.jpeg'))])
# frame_count = 0
# for _frame in dir:  # using the saved frames ( pictures)
#     frame = cv2.imread( path + "/"+_frame ) # Colored image 
#     frame_count += 1
#     print(frame_count)
    
#     # =================================== frame to Gray ==================================================
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Gray image
#     # gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
#     gray = np.array(gray)
#     if prev_gray is None:
#         prev_gray = gray

#     else: 
#         ttc, foe_all, secs, frams = find_TTC(frame ,gray, prev_gray)
#         # print(vel)
#         prev_gray = gray
#         # when using entire image config ==========
#         # ttc_new.append(ttc)


#         # ===== normal config
#         i = 0
#         while i < len(ttc):
#             index = "sec_" + str(i+1)
#             TTC[index].append(round(ttc[i], 3))
#             i+=1
#         k = 0
#         while k < len(secs):
#             index = "sec_" + str(k+1)
#             sec_used[index].append(secs[k])
#             k+=1
#         l = 0
#         while l < len(frams):
#             index = "sec_" + str(l+1)
#             frame_used[index].append(frams[l])
#             l+=1


# function to smoothen the curves
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Eliminating negative numbers
def negativ_average(data):
    for i in range(len(data)):
        if i == 0:
            if data[i]<0:
                data[i] = data[i+1]/2
        elif i == (len(data)-1):
            if data[i]<0:
                data[i] = data[i-1]/2
        else:
            if data[i]<0:
                data[i] = (data[i+1] + data[i-1])/2
    return data

window_size = 10

