"""
------ EMMANUEL ASAH ------
------ BEng Thesis THWS in Schweinfurt -----
------ WS 2023/24 -----

"""

# import dependencies 
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import time

def display(frame, rois, foe_all, TTC_all, p1, p2, point_of_intersection):
    frame1 = np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    height, width = frame1.shape
    # height, width = frame.shape
    # x1, y1, x2, y2 = rois[0]
    roi_width = width//3
    roi_height= height//3
    # Draw section lines on the merged frame
    
    for x in range(roi_width, width, roi_width):
        cv2.line(frame, (x, 0), (x, height), (0, 0, 255), 2)
    for y in range(roi_height, height, roi_height):
        cv2.line(frame, (0, y), (width, y), (0, 0, 255), 2)
    # cv2.imwrite('C:/docs/IMC8/thesis/codes/TTC101/records' + '/rec_' + '.png', frame)
    # Drawing intersection points
    for p in point_of_intersection:
        # print(p)
        cv2.circle(frame, (int(p[1]), int(p[0])), 2, (0, 125, 155), 2)
    # cv2.imwrite('C:/docs/IMC8/thesis/codes/TTC101/records' + '/rec_intersec' + '.png', frame)

    # # draw flow lines
    for a, b in zip(p1, p2):
        # print(b)
        ax, ay = a[0]
        bx,by = b[0]
        cv2.arrowedLine(frame,(int(ax), int(ay)),(int(bx), int(by)),[0,0,255],2,tipLength=0.2)
    # cv2.imwrite('C:/docs/IMC8/thesis/codes/TTC101/records' + '/rec_flow' + '.png', frame)
    # Writing the section numbers to each section
    i = 0
    while i < len(rois):
        [x1, y1 ,x2, y2] = rois[i]
        text_position = (x1 + 10, y1 + 30)
        if(TTC_all[i] >= 0):
            cv2.putText(frame, str(TTC_all[i]), text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        

        if(TTC_all[i] == 22222 or TTC_all[i] == 11111):
            cv2.putText(frame, "----", text_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)

        i= i+1

    # Drawing the Focus of Expansion
    for x,y in foe_all:
        # cv2.circle(frame, (int(foe_all[i][1]), int(foe_all[i][0])), 5, (255, 0, 0), 4)
        cv2.circle(frame, (int(y), int(x)), 5, (255, 0, 0), 4)
    # frame = cv2.resize(frame,(640, 480))
    # cv2.imwrite('C:/docs/IMC8/thesis/codes/TTC101/records' + '/rec_Foe' + '.png', frame)
    # # Drawing the Section of the FOE
    # for x1,y1,x2,y2 in section_out:
    #     cv2.rectangle(frame,(x1,y1),(x2,y2),(255, 0, 0), 2)
    # cv2.imwrite('C:/docs/IMC8/thesis/codes/TTC101/records' + '/rec_foe_sec' + '.png', frame)
    cv2.imshow("Annotated frame", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# def draw_flow(p1, p2, image):

#     for i in range(len(p1)):
#         cv2.arrowedLine(image,(p1[i,0], p1[i,1]),(p2[i,0], p2[i,1]),[255,255,255],1,tipLength=0.2)


    