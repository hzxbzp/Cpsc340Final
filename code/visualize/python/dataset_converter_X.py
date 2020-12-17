#!/usr/bin/env python

import csv
import numpy as np

from utils.dataset_types import MotionState, Track

if __name__ == "__main__":

    filename = "../recorded_trackfiles/.DR_USA_Intersection_MA/X_2.csv"
    YFile = False
    frameOffset = 0

    if (YFile):
        frameOffset = 10



    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        id_pos_dict_x = {}
        id_pos_dict_y = {}

        with open('../recorded_trackfiles/.DR_USA_Intersection_MA/vehicle_tracks_002.csv', mode='w', newline='', encoding='utf-8') as file:
            fieldnames = ['frame_id','timestamp_ms','track_id','agent_role','agent_type','x','y','vx','vy','psi_rad','length','width']
            writer = csv.writer(file)
            writer.writerow(fieldnames)
            #writer.writeheader()

            for i, row in enumerate(list(csv_reader)):

                timestamp_ms, id0, role0, type0, x0, y0, present0, id1, role1, type1, x1, y1, present1, id2, role2, type2, x2, y2, present2, id3, role3, type3, x3, y3, present3, id4, role4, type4, x4, y4, present4, id5, role5, type5, x5, y5, present5, id6, role6, type6, x6, y6, present6, id7, role7, type7, x7, y7, present7, id8, role8, type8, x8, y8, present8, id9, role9, type9, x9, y9, present9 = row

                if (i > 0):
                    for ii in range(10):
                        rowToWrite = []
                        rowToWrite.append(frameOffset+i)
                        rowToWrite.append(timestamp_ms)
                        present = row[6*ii + 6]
                        if present:
                            data = row[1+(6*ii):1+(6*ii)+5]
                            data[3] = float(data[3]) + 1010 - 33
                            data[4] = float(data[4]) + 1005
                            id = data[0]
                            if (id not in id_pos_dict_x):
                                id_pos_dict_x[id] = data[3]
                                id_pos_dict_y[id] = data[4]
                                yaw = np.pi / 2
                                speedX = 0
                                speedY = 0
                            else:
                                speedX = (float(data[3]) - float(id_pos_dict_x[id]))/0.1
                                speedY = (float(data[4]) - float(id_pos_dict_y[id]))/0.1
                                if (speedX == 0):
                                    yaw = np.pi / 2
                                else:
                                    yaw = np.arctan(speedY/speedX)
                                #yaw = np.arctan(speedY/speedX)
                                id_pos_dict_x[id] = data[3]
                                id_pos_dict_y[id] = data[4]
                            data.append(speedX)
                            data.append(speedY)
                            #yaw = np.arctan(speedY/speedX)
                            if (data[2] == ' car'):
                                data.append(yaw)
                                data.append(4.5)
                                data.append(2)
                            else:
                                data.append(yaw)
                                data.append(1)
                                data.append(1)
                            rowToWrite = rowToWrite + data
                            #print(len(rowToWrite))
                            writer.writerow(rowToWrite)
                            #print(data)
