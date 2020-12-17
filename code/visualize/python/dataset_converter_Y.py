#!/usr/bin/env python

import csv
import numpy as np

from utils.dataset_types import MotionState, Track

if __name__ == "__main__":

    filename = "../recorded_trackfiles/.DR_USA_Intersection_MA/Y_0.csv"
    YFile = True
    frameOffset = 0

    if (YFile):
        frameOffset = 10



    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        lastX = 0
        lastY = 0

        with open('../recorded_trackfiles/.DR_USA_Intersection_MA/vehicle_tracks_001.csv', mode='w', newline='', encoding='utf-8') as file:
            fieldnames = ['frame_id','timestamp_ms','track_id','agent_role','agent_type','x','y','vx','vy','psi_rad','length','width']
            writer = csv.writer(file)
            writer.writerow(fieldnames)

            #fieldCounter = 0

            data = list(csv_reader)
            for i in range(30):
                vals = data[i+1]
                print(vals)
                #print(data)
                time = vals[0]
                xval = float(vals[1]) + 1000.0
                yval = float(vals[2]) + 1000.0
                rowToWrite = []
                rowToWrite.append(frameOffset+i)
                rowToWrite.append(time)
                rowToWrite.append(0)
                rowToWrite.append("agent")
                rowToWrite.append("car")
                rowToWrite.append(xval)
                rowToWrite.append(yval)

                speedX = float((xval - lastX)/0.1)
                speedY = float((yval - lastY)/0.1)
                lastX = xval
                lastY = yval
                if (i==0):
                    yaw = 0
                else:
                    yaw = np.arctan(speedY/speedX)

                rowToWrite.append(speedX)
                rowToWrite.append(speedY)
                rowToWrite.append(yaw)

                rowToWrite.append(4.5)
                rowToWrite.append(1.5)

                writer.writerow(rowToWrite)
