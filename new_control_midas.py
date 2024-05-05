"""
------ EMMANUEL ASAH ------
------ BEng Thesis THWS in Schweinfurt -----
------ WS 2023/24 -----

"""


from yolo_X_midas import yolo_box, TTC_midas, decision_sys
#from arUco_tracking import arUco_tracking
from sklearn import preprocessing
import matplotlib.pyplot as plt
import os , sys
import cv2
import numpy as np
import time as time
from threading import Thread, Event
import torch


# Download the MiDaS
midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
midas.to('cpu')
midas.eval()
# Input transformation pipeline
transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = transforms.small_transform




if __name__ == "__main__":
    frame_counter = 1
    my_TTC_counter = 0
    TTC_midas_counter = -1
    result1 = {}
    result2 = {}
    result3 = {}
    

    feature_params = dict(maxCorners=300, qualityLevel=0.3, minDistance=7, blockSize=7)
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
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
    mapTTC = {"sec_1": [],
           "sec_2": [],
           "sec_3": [],
           "sec_4": [],
           "sec_5": [],
           "sec_6": [],
           "sec_7": [],
           "sec_8": [],
           "sec_9": []
           }
    Rel_Vel = {"sec_1": [],
           "sec_2": [],
           "sec_3": [],
           "sec_4": [],
           "sec_5": [],
           "sec_6": [],
           "sec_7": [],
           "sec_8": [],
           "sec_9": []
           }
    ttc = None
    foe_all = None
    prev_rel_vel_depth = []

    # ======================================= Choise of run ============================================
    '''
    decision for which algorithm to run

    '''
    human = False
    obstacle = False
    # ======================================= Preparing Tello ==============================================

    # drone = tello.Tello()
    # drone.connect()
    # drone.set_speed(20)
    # print(f"Battery : {drone.get_battery()}% ")
    # drone.streamon()
    # drone.takeoff()
    # time.sleep(1)
    # # drone.move_up(75)  
    # drone.send_rc_control(0,0,20,0)
    # time.sleep(75/17)
    # time.sleep(1)
    # record = True
    # i = 0
    # def save_image(i,frame):
    #     cv2.imwrite('C:/docs/IMC8/thesis/codes/TTC101/drone_recording5' + '/rec_' +str(i).zfill(4) + '.png', frame)
    #     cv2.imshow('results', frame)
    #     # time.sleep(1/20)


    # ============================================= Dependencies and leading function =======================
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
    record = True

            

    def simple_moving_average(data_stream, window_size=3):
        """
        Calculates Simple Moving Average (SMA) for a data stream.

        Args:
        - data_stream: an iterable that provides a new data point with each call.
        - window_size: the number of data points to consider for the moving average.

        Yields:
        - The SMA value with each new data point.
        """
        data_queue = []  # Initialize a list to store recent data points

        for new_data_point in data_stream:
            data_queue.append(new_data_point)  # Add the new data point to the queue

            # When we have enough points, yield the SMA
            if len(data_queue) == window_size:
                # yield sum(data_queue) / window_size
                data_queue.pop(0)  # Remove the oldest data point to maintain the window size

            return sum(data_queue)/ window_size


    def exponential_moving_average(new_data_point, previous_smoothed_data_point, alpha = 0.2):
        """
        Apply Exponential Moving Average to smooth the data in real-time.

        Args:
        - new_data_point: The new data point to include in the smoothed data.
        - previous_smoothed_data_point: The previous data point in the smoothed curve.
        - alpha: The smoothing factor (0 < alpha < 1), which determines how much weight to give to newer data points.

        Returns:
        - The new smoothed data point.
        """
        if previous_smoothed_data_point is None:
            # If there's no previous data, the smoothed data is just the current point
            return new_data_point
        else:
            # Calculate the EMA based on the previous smoothed point and the new data
            return alpha * new_data_point + (1 - alpha) * previous_smoothed_data_point


    def execute_command1(command):
        print(command)

    def execute_command(drone, command):
        # Logic to execute commands on the Tello drone
        if command == "takeoff":
            drone.takeoff()
            time.sleep(.2)
        elif command == "land":
            drone.land()
        elif command == "move_forward":
            # drone.move_forward(5)
            drone.send_rc_control(0,5,0,0)
            time.sleep(1.1)
            # time.sleep(.1)
        elif command == "move_back":
            # drone.move_back(10)
            drone.send_rc_control(0,-10,0,0)
            time.sleep(1.1)
        elif command == "move_up":
            drone.move_up(10)
            time.sleep(.1)
        elif command == "move_down":
            drone.move_down(10)
            time.sleep(.1)
        elif command == "move_left":
            # drone.move_left(10)
            drone.send_rc_control(10,0,0,0)
            time.sleep(1.1)
        elif command == "move_right":
            # drone.move_right(10)
            drone.send_rc_control(-10,0,0,0)
            time.sleep(1.1)
        elif command == "rotate_10":
            drone.rotate_clockwise(10)
            time.sleep(.1)
        elif command == "rotate_180":
            drone.rotate_clockwise(180)
            time.sleep(.1)
        elif command[0:10] == "rotate_by_":
            drone.rotate_clockwise(int(command[10::]))
            time.sleep(.1)
        elif command[0:16] == "move_forward_by_":
            drone.rotate_clockwise(int(command[16::]))
            time.sleep(.1)
        # Add more commands as needed
    gray_img = None
    cmd_pr = ""
    count = 0
    def img_processing(event):
        global count
        global gray_img
        global cmd_pr
        global pics
        global result1
        global result2
        global result3
        global my_TTC_counter
        global commands
        prev_gray = None

        # ======================================== Yolo_X_midas dependencids ============================

        y_vals = {}
        prev_cl_id = []
        prev_boxes = []
        prev_out_norm = None
        prev_sections = None
 
        # ======================================== Using Tello =============================================

        # while record:
        #     cmd = ""
        #     frame = drone.get_frame_read().frame
        #     save_image(i, frame)
        #     i += 1
        #     key = cv2.waitKey(1)
        #     if key == 27:
        #         break
        
        # ======================================= using WebCam ============================================= 

        # cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # webcam by defualt is 640x480
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 936) # best posible width resolutions 1236 , 612 these are in x(12) for this special case
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) # best matching height resolutns 720 , 936

        # cam_on = True
        # while cam_on == True:
        #     ret, frame = cap.read()
        #     if not ret:
        #         break

        # ================================== Using files in a directory =====================================


        path = "C:/docs/IMC8/thesis/codes/TTC101/test2/frames1"
        dir = sorted([f for f in os.listdir(path) if f.endswith(('.jpg','.png','.jpeg'))])
        prev_gray = None
        t1 = time.time()
        t2 = time.time()
        frame_rate = 10
        lines = [[],[],[],[],[],[],[],[],[]]
        smooth_data = [None,None,None,None,None,None,None,None,None]
        
        for _frame in dir:  # using the saved frames ( pictures)

            cmd = ""
            frame = cv2.imread( path + "/"+_frame ) # Colored image 

            # =================================== frame to Gray ==================================================

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Gray image
            gray = np.array(gray)
            gray_img = gray
            data = {}


            # ================================= Image processing with yolo_X_midas =====================

            if count > 1:

                if prev_out_norm is None:
                    results = yolo_box(frame)
                    if results == -1:
                        print("No Trackable")
                    else:
                        boxes, ids, clases = results
                        prev_boxes = np.array(boxes)
                        prev_clases = clases
                        prev_ids = ids
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
                        prev_out_norm = output_norm
                        for i in range(0, len(boxes)):
                            id = ids[i][0]
                            clas = clases[i][0]
                            prev_cl_id.append([[clas, id]])
                        prev_cl_id = np.array(prev_cl_id)

                else: 
                    # print(yolo_ttc(frame, prev_cl_id, prev_boxes, prev_gray, frame_rate))
                    ttc, boxes, out_norm, sections, cl_id = TTC_midas(frame, prev_cl_id, prev_boxes, prev_out_norm)
                    print(ttc)
                    prev_cl_id, prev_sections, prev_boxes, prev_out_norm = cl_id, gray, boxes, out_norm

                    for _id in prev_cl_id:
                        _, id_ = _id[0]
                        if id_ in y_vals:
                            y_vals[id_].append(ttc[i])
                        else:
                            y_vals[id_] = []
                            y_vals[id_].append(ttc[i])

                        if id_ in data:
                            data[id_].append(ttc[i])
                        else:
                            data[id] = []
                            data[id_].append(ttc[i])

                    my_TTC_counter += 1
                    if my_TTC_counter == 1:
                        result1 = data
                    if my_TTC_counter == 2:
                        result2 = data
                    if my_TTC_counter == 3:
                        result3 = data
                    if my_TTC_counter >= 4:
                        result1 = result2
                        result2 = result3
                        result3 = data
                    resu1 = []
                    resu2 = []
                    resu3 = ttc
                    for id in data:
                        if id in result1:
                            resu1.append(result1[id])
                        else:
                            resu1.append(-1)
                        if id in result2:
                            resu2.append(result2[id])
                        else: 
                            resu2.append(-1)
                        
                        

                    if my_TTC_counter > 4:
                        drone = ""
                        cmd = decision_sys(resu1, resu2, resu3, sections)
                        execute_command1(cmd)
            prev_gray = gray
            cmd_pr = cmd
            count += 1
        pics = False
        event.set()
        
    record = True
    human = False ############ helps to decide which algorithm runs first
    obstacle = False

    positn = [] # OX, OY, OZ format


    # =================================== using Threads ==================================================
    event = Event()
    t = Thread(target= img_processing, args=(event, ))
    t.start()

    # =================================== arUco Processing  ==============================================
    run = True
    pics = True
    num = 0
    last_count = 0
    distance = 0
    while run and pics:


        # print(count)
        # if cv2.waitKey(10) & 0xFF == ord('q'): 
        #     cam_on = False
        #     break

        if cmd_pr != "" and count != last_count:
            execute_command(cmd_pr)
            cmd_pr = ""

        else:
            if distance < 101:
                # drone moves forward place holder is continue
                execute_command("move_forward")
                distance += 5
            
        last_count = count
        # print(last_count)
        # num += 1
    # cap.release()

    t.join()
    # print(mapTTC)

    fig, AXS = plt.subplots(3, 3)
    # plt.ylim(-5, 70)
    row = 0
    col = 0
    counter = 0
    for key in TTC:
        x_vals = np.arange(0,len(TTC[key]))
        y_vals = np.array(TTC[key])
        # normalized values
        # y_vals =  np.where(np.isnan(y_vals), -3, y_vals)
        # revert 22222 and 11111 exceptions to native -2 and -1
        y_vals = [-2 if value == 22222 else value for value in y_vals]
        y_vals = [-1 if value == 11111 else value for value in y_vals]
        # y_norm = (preprocessing.normalize([y_vals])*20).T
        if row>= 3:
            col += 1
            row = 0
                
        AXS[row,col].plot(x_vals,y_vals)
        # AXS[row,col].plot(x_vals,y_norm)
        AXS[row,col].set_ylim(-5, 20)
        AXS[row,col].set_title("TTC of Section "+ str(counter+1) )
        row += 1
        counter += 1

    

    plt.show()
    # print(TTC)
    cv2.destroyAllWindows()