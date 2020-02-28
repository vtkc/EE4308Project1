#!/usr/bin/env python

import roslib, rospy, rospkg
from numpy import *
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64MultiArray
import cv2
import numpy
import lab3A_aux
import sys

# ================================= CONSTANTS ==========================================        
# let's cache the SIN and POS so we don't keep recalculating it, which is slow
PI = pi # numpy.pi
DEG2RAD = [i/180.0*PI for i in xrange(360)] # DEG2RAD[3] means 3 degrees in radians
SIN = [sin(DEG2RAD[i]) for i in xrange(360)] # SIN[32] means sin(32degrees)
COS = [cos(DEG2RAD[i]) for i in xrange(360)]
SQRT2 = sqrt(2)
MAX_RNG = lab3A_aux.MAX_RNG
CELL_SIZE = 0.1
INF_RADIUS = 0.25
TWOPI = 2*PI
NEAR_RADIUS = 0.1
    
# =============================== SUBSCRIBERS =========================================  
def subscribe_move(msg):
    global msg_move
    t = msg.data
    msg_move[0] = t[0] # rx
    msg_move[1] = t[1] # ry
    msg_move[2] = t[2] # ro
    msg_move[3] = t[3] # positional error
    
def subscribe_scan(msg):
    # stores a 360 long tuple of LIDAR Range data into global variable rbt_scan. 
    # 0 deg facing forward. anticlockwise from top.
    global rbt_scan, write_scan, read_scan
    write_scan = True # acquire lock
    if read_scan: 
        write_scan = False # release lock
        return
    rbt_scan = msg.ranges
    write_scan = False # release lock
    
def get_scan():
    # returns scan data after acquiring a lock on the scan data to make sure it is not overwritten by the subscribe_scan handler while using it.
    global write_scan, read_scan
    read_scan = True # lock
    while write_scan:
        pass
    scan = rbt_scan # create a copy of the tuple
    read_scan = False
    return scan


# ================================ BEGIN ===========================================
def main(goals):
    # ---------------------------------- INITS ----------------------------------------------
    # init node
    rospy.init_node('main')
    
    # Set the labels below to refer to the global namespace (i.e., global variables)
    # global is required for writing to global variables. For reading, it is not necessary
    global rbt_scan, read_scan, write_scan, msg_move
    
    # Initialise global vars with NaN values 
    # nan and inf are imported from numpy. If you use "import numpy as np", then nan is np.nan, and inf is np.inf.
    rbt_scan = None
    msg_move = [-1. for i in xrange(4)]
    read_scan = False
    write_scan = False

    # Subscribers
    rospy.Subscriber('scan', LaserScan, subscribe_scan, queue_size=1)
    # ~ subscribe to topic we created from lab3_move.py
    rospy.Subscriber('move', Float64MultiArray, subscribe_move, queue_size=1)
    
    # Publishers
    # ~ publish to topic 'lab3', a float64multiarray message
    publisher_main = rospy.Publisher('main', Float64MultiArray, latch=True, queue_size=1)
    msg_main = [0. for i in xrange(3)] 
    # ~ [0] operating mode: 0. is run, 1. is stop running and exit.
    # ~ [1] px: the x position of the target for the robot to pursue
    # ~ [2] py: the y position of the target for the robot to pursue
    msg_m = Float64MultiArray()
    # cache the part where we want to modify for slightly faster access
    msg_m.data = msg_main
    # publish first data for main node to register
    publisher_main.publish(msg_m)
    
    # Wait for Subscribers to receive data.
    # ~ note imu will not publish if you press Ctrl+R on Gazebo. Use Ctrl+Shift+R instead
    while (rbt_scan is None or msg_move[0] == -1.) and not rospy.is_shutdown():
        pass
        
    print('[INFO] Done waiting for topics...')
    if rospy.is_shutdown():
        return
        
    # Data structures
    occ_grid = lab3A_aux.OccupancyGrid((-4,-4), (4,4), CELL_SIZE, 0, INF_RADIUS)
    los = lab3A_aux.GeneralLOS()
    post_process_los = lab3A_aux.GeneralIntLOS()
    # ~ moved motion model to lab3_move.py    
    planner = lab3A_aux.Astar(occ_grid)
    
    # ~ Cache methods for slightly faster access
    update_at_idx = occ_grid.update_at_idx
    # ~ Instead of idx2pos, which returns a tuple, individually return the i, j, x or y, for slightly faster access
    # ~ avoid i, j = (...) as much as possible. you should use: sth = (...); i = sth[0]; j = sth[1]
    # ~ cached arrays like the inflation mask should use lists [] instead of tuples ()
    # ~ lists[] are faster to access than tuples(). i.e. list_obj[num] faster than tuple_obj[num]
    # ~ but creating lists[] is slower than creating tuples()
    # ~ so use tuples() on dynamically (not cached) generated arrays like the A* path 
    x2iE = occ_grid.x2iE # ~ returns the exact, float index i from pos x
    y2jE = occ_grid.y2jE # ~ returns the exact, float index j from pos y
    x2i = occ_grid.x2i # ~ returns the rounded, int index i from pos x
    y2j = occ_grid.y2j # ~ returns the rounded, int index j from pos y
    i2x = occ_grid.i2x # ~ returns the float pos x from i
    j2y = occ_grid.j2y # ~ returns the float pos y from j
    # post process
    post_process = lab3A_aux.post_process
    
    # get the first goal pos
    gx = goals[0][0]; gy = goals[0][1]
    # number of goals (i.e. areas)
    g_len = len(goals)
    # get the first goal idx
    gi = occ_grid.x2i(gx); gj = occ_grid.y2j(gy)
    # set the goal number as zero
    g = 0
    
    # ---------------------------------- BEGIN ----------------------------------------------
    t = rospy.get_time()
    while (not rospy.is_shutdown()): # required to Keyboard interrupt nicely
        if (rospy.get_time() > t): # every 50 ms
            # get position
            rx = msg_move[1]
            ry = msg_move[2]
            ro = msg_move[3]
            
            # get scan
            scan = get_scan()
            
            # ~ exact rbt index
            riE = x2iE(rx); rjE = y2jE(ry)
            # ~ robot index using int and round. 
            # ~ int64 and float64 are slower than int and float respectively
            ri = int(round(riE)); rj = int(round(rjE))
            
            # ~ eiE, ejE  # exact, float index i of scan boundary; exact, float index j of scan boundary
            # ~ ei , ej   # rounded, int index i of scan boundary; rounded, int index j of scan boundary
            # for each degree in the scan
            for o in xrange(360):
                if scan[o] != inf: # range reading is < max range ==> occupied
                    # inv sensor model
                    eiE = x2iE(rx + scan[o] * cos(ro + DEG2RAD[o])); ejE = y2jE(ry + scan[o] * sin(ro + DEG2RAD[o]))
                    ei = int(round(eiE)); ej = int(round(ejE))
                    # set the obstacle cell as occupied
                    update_at_idx(ei, ej, True)
                else: # range reading is inf ==> no obstacle found
                    # inv sensor model
                    eiE = x2iE(rx + MAX_RNG * cos(ro + DEG2RAD[o])); ejE = y2jE(ry + MAX_RNG * sin(ro + DEG2RAD[o]))
                    ei = int(round(eiE)); ej = int(round(ejE))
                    update_at_idx(ei, ej, False)
                # set all cells between current cell and last cell as free
                los.init(riE, rjE, eiE, ejE)
                while los.i != ei or los.j != ej:
                    update_at_idx(los.i, los.j, False)
                    los.next()
                    # avoids last cell 
            
            # plan
            # ~ update_at_idx will signal a path replanning via occ_grid.need_path if the existing path (see next) lies on an inflated or obstacle cell
            # ~ update_at_idx checks against an internally stored path updated using occ_grid.update_path(p), 
            # ~ where p is the new path to update that is returned from the post-process or A*
            # ~ occ_grid.show_map also uses the internally stored path to draw the path in image
            # ~ methods below are not cached for clarity
            if occ_grid.need_path:
                if planner.get_path(ri, rj, gi, gj):
                    # ~ A* path is stored in the planner instance, in planner.path
                    # ~ Take note that it starts from [(goal idx i, goal idx j), ... , (rbt idx i, rbt idx j)]
                    print('[INFO] Recalculated Path')
                    # ~ path should be the post processed path containing all indices
                    # ~ pts will return turning points, including rbt idx and goal idx
                    # ~ u can use this tuple return once in a while, will not affect performance by too much
                    path, pts = post_process(planner.path, occ_grid.c_inf, post_process_los) 
                    # robot is at goal if there are only one turn pt. (the rbt idx == goal idx)
                    p = len(pts)
                    if p <= 1:
                        # send shutdown
                        msg_main[0] = 1.
                        publisher_main.publish(msg_m)
                        # wait for sometime for move node to pick up message
                        t += 0.3
                        while rospy.get_time() < t:
                            pass
                        break
                    # ~ update the internally stored path in occ_grid
                    occ_grid.update_path(path)
                    # ~ reset the occ_grid.need_path
                    occ_grid.need_path = False
                    # ~ get the pt. number, starting from the pt. after the rbt idx, for robot to pursue
                    p = len(pts) - 2
                    px = i2x(pts[p][0]); py = j2y(pts[p][1])
                else:
                    raise Exception('The error in pose was so bad that the goal idx({:3d},{:3d}) or rbt idx({:3d},{:3d}) is probably out of the map. Or some other error'.format(gi, gj, ri, rj))
            
            # check if rbt in range of pt
            dx = px-rx; dy = py-ry # final-initial
            # get position error
            err_pos = sqrt(dx*dx + dy*dy)
            if err_pos < NEAR_RADIUS: # 
                p -= 1
                if p < 0: # move to next goal, no more pts
                    # signal that a new path is needed
                    occ_grid.need_path = True
                    g += 1
                    if g >= g_len: # no more goals
                        # send shutdown
                        msg_main[0] = 1.
                        publisher_main.publish(msg_m)
                        # wait for sometime for move node to pick up message
                        t += 0.3
                        while rospy.get_time() < t:
                            pass
                        break
                    # get the next goal pos for replanning
                    gx = goals[g][0]; gy = goals[g][1]
                    gi = x2i(gx); gj = y2j(gy)
                else: #
                    px = i2x(pts[p][0]); py = j2y(pts[p][1])
                
            # prepare message for sending
            msg_main[1] = px
            msg_main[2] = py
            publisher_main.publish(msg_m)
            
            # show the map as a picture
            occ_grid.show_map(ri, rj, gi, gj)
            
            # increment the time counter
            et = rospy.get_time() - t
            print('[INFO] MAIN ({}, {:.3f})'.format(et <= 0.2, et))
            t += 0.2
    print('[INFO] MAIN stopped')
        
if __name__ == '__main__':      
    try: 
        goals = sys.argv[1]
        goals = goals.split('|')
        for i in xrange(len(goals)):
            tmp = goals[i].split(',')
            tmp[0] = float(tmp[0])
            tmp[1] = float(tmp[1])
            goals[i] = tmp
        main(goals)
    except rospy.ROSInterruptException:
        pass


