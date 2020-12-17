#!/usr/bin/env python

import csv

from utils.dataset_types import MotionState, Track


# class Key:
#     track_id = "track_id"
#     frame_id = "frame_id"
#     time_stamp_ms = "timestamp_ms"
#     agent_type = "agent_type"
#     x = "x"
#     y = "y"
#     vx = "vx"
#     vy = "vy"
#     psi_rad = "psi_rad"
#     length = "length"
#     width = "width"

class Key:
    frame_id = "frame_id"
    time_stamp_ms = "timestamp_ms"
    track_id = "track_id"
    agent_role = "agent_role"
    agent_type = "agent_type"
    x = "x"
    y = "y"
    vx = "vx"
    vy = "vy"
    psi_rad = "psi_rad"
    length = "length"
    width = "width"

#
# class KeyEnum:
#     track_id = 0
#     frame_id = 1
#     time_stamp_ms = 2
#     agent_type = 3
#     x = 4
#     y = 5
#     vx = 6
#     vy = 7
#     psi_rad = 8
#     length = 9
#     width = 10


class KeyEnum:
    frame_id = 0
    time_stamp_ms = 1
    track_id = 2
    agent_role = 3
    agent_type = 4
    x = 5
    y = 6
    vx = 7
    vy = 8
    psi_rad = 9
    length = 10
    width = 11


def read_tracks(filename):

    print(filename)

    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        import operator
        csv_reader = sorted(csv_reader, key=operator.itemgetter(2), reverse=True)

        track_dict = dict()
        track_id = None
        pedDict = dict()

        nonPed = 0
        for i, row in enumerate(list(csv_reader)):
            badTrackID=False

            if i == 0:
                # check first line with key names
                assert(row[KeyEnum.track_id] == Key.track_id)
                assert(row[KeyEnum.frame_id] == Key.frame_id)
                assert(row[KeyEnum.time_stamp_ms] == Key.time_stamp_ms)
                assert(row[KeyEnum.agent_type] == Key.agent_type)
                assert(row[KeyEnum.x] == Key.x)
                assert(row[KeyEnum.y] == Key.y)
                assert(row[KeyEnum.vx] == Key.vx)
                assert(row[KeyEnum.vy] == Key.vy)
                assert(row[KeyEnum.psi_rad] == Key.psi_rad)
                assert(row[KeyEnum.length] == Key.length)
                assert(row[KeyEnum.width] == Key.width)
                continue
            #if (isinstance(row[KeyEnum.track_id], str)):
            if (not row[KeyEnum.track_id].isdigit()):
                if(row[KeyEnum.track_id] in pedDict):
                    tid = pedDict[row[KeyEnum.track_id]]
                else:
                    tid = nonPed + 1000
                    nonPed += 1
                    badTrackID = True
                    pedDict[row[KeyEnum.track_id]] = tid
            else:
                tid = int(row[KeyEnum.track_id])
            #if (True):
            if tid != track_id:
                # new track
                track_id = tid
                assert(track_id not in track_dict.keys()), \
                    "Line %i: Track id %i already in dict, track file not sorted properly" % (i+1, track_id)
                track = Track(track_id)
                track.agent_type = row[KeyEnum.agent_type]
                if (len(row) == 9):
                    track.length == float(1)
                    track.width == float(1)
                else:
                    track.length = float(row[KeyEnum.length])
                    track.width = float(row[KeyEnum.width])
                track.time_stamp_ms_first = int(row[KeyEnum.time_stamp_ms])
                track.time_stamp_ms_last = int(row[KeyEnum.time_stamp_ms])
                track_dict[track_id] = track

            track = track_dict[track_id]
            track.time_stamp_ms_last = int(row[KeyEnum.time_stamp_ms])
            ms = MotionState(int(row[KeyEnum.time_stamp_ms]))
            ms.x = float(row[KeyEnum.x])
            ms.y = float(row[KeyEnum.y])
            ms.vx = float(row[KeyEnum.vx])
            if (row[KeyEnum.agent_role] == ' agent'):
                ms.vx = float(10000)
                #print("Test")
            ms.vy = float(row[KeyEnum.vy])
            if (len(row) == 9):
                ms.psi_rad = float(1)
            else:
                ms.psi_rad = float(row[KeyEnum.psi_rad])
            track.motion_states[ms.time_stamp_ms] = ms

        return track_dict


def read_pedestrian(filename):

    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        track_dict = dict()
        track_id = None

        for i, row in enumerate(list(csv_reader)):

            if i == 0:
                # check first line with key names
                assert (row[KeyEnum.track_id] == Key.track_id)
                assert (row[KeyEnum.frame_id] == Key.frame_id)
                assert (row[KeyEnum.time_stamp_ms] == Key.time_stamp_ms)
                assert (row[KeyEnum.agent_type] == Key.agent_type)
                assert (row[KeyEnum.x] == Key.x)
                assert (row[KeyEnum.y] == Key.y)
                assert (row[KeyEnum.vx] == Key.vx)
                assert (row[KeyEnum.vy] == Key.vy)
                continue

            if row[KeyEnum.track_id] != track_id:
                # new track
                track_id = row[KeyEnum.track_id]
                assert (track_id not in track_dict.keys()), \
                    "Line %i: Track id %s already in dict, track file not sorted properly" % (i + 1, track_id)
                track = Track(track_id)
                track.agent_type = row[KeyEnum.agent_type]
                track.time_stamp_ms_first = int(row[KeyEnum.time_stamp_ms])
                track.time_stamp_ms_last = int(row[KeyEnum.time_stamp_ms])
                track_dict[track_id] = track

            track = track_dict[track_id]
            track.time_stamp_ms_last = int(row[KeyEnum.time_stamp_ms])
            ms = MotionState(int(row[KeyEnum.time_stamp_ms]))
            ms.x = float(row[KeyEnum.x])
            ms.y = float(row[KeyEnum.y])
            ms.vx = float(row[KeyEnum.vx])
            ms.vy = float(row[KeyEnum.vy])
            track.motion_states[ms.time_stamp_ms] = ms

        return track_dict
