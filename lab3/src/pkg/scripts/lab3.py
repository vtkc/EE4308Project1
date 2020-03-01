#!/usr/bin/env python

import roslib, rospy, rospkg
from numpy import *
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64MultiArray
import cv2
import numpy
import lab3_aux
import sys

# ================================= CONSTANTS ==========================================        
# let's cache the SIN and POS so we don't keep recalculating it, which is slow
PI = pi # numpy.pi
DEG2RAD = [i/180.0*PI for i in xrange(360)] # DEG2RAD[3] means 3 degrees in radians
SIN = [sin(DEG2RAD[i]) for i in xrange(360)] # SIN[32] means sin(32degrees)
COS = [cos(DEG2RAD[i]) for i in xrange(360)]
SQRT2 = sqrt(2)
MAX_RNG = lab3_aux.MAX_RNG
CELL_SIZE = 0.1 # Changing here has no effect if the input argument is given in lab3.launch
INF_RADIUS = 0.2
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

# =============================== MOTION PLANNER =========================================  
class JPS:
    def __init__(self, occ_grid):
        self.occ_grid = occ_grid
        self.open_list = []
    def get_path(self, start_idx, goal_idx):
        path = []
        ni, nj = self.occ_grid.num_idx
        # resets h-cost, g-cost, update and occ for all cells
        for i in xrange(ni):
            for j in xrange(nj):
                dj = fabs(self.occ_grid.cells[i][j][7][1] - goal_idx[1])
                di = fabs(self.occ_grid.cells[i][j][7][0] - goal_idx[0])
                if dj > di:
                    a0,a1 = di, dj-di
                else:
                    a0,a1 = dj, di-dj
                self.occ_grid.cells[i][j][1] = [inf, inf, inf]
                self.occ_grid.cells[i][j][2] = [inf, inf, inf]
                self.occ_grid.cells[i][j][3] = [int64(a0), int64(a1), SQRT2 * int64(a0) + int64(a1) ]
                self.occ_grid.cells[i][j][4] = False
                self.occ_grid.cells[i][j][5] = None
                self.occ_grid.cells[i][j][6] = 0                

        start_cell = self.occ_grid.idx2cell(start_idx)
        start_cell[2] = [0, 0, 0]
        start_cell[1] = [start_cell[2][0] + start_cell[3][0], start_cell[2][1] + start_cell[3][1], start_cell[2][2] + start_cell[3][2]]
        #Move to cheapest neighbour around start cell (only needs to be done once)
        for nb_cell in self.get_free_neighbors(start_cell):
            ############COMPARE G COST############
            a = nb_cell[7]
            b = start_cell[7]
            dj = fabs(a[1] - b[1])
            di = fabs(a[0] - b[0])
            if dj > di:
                a0 = di
                a1 = dj-di
            else:
                a0 = dj
                a1 = di-dj
            dist_sep_g_cost = [float64(a0), float64(a1), SQRT2 * float64(a0) + float64(a1)]
            A = dist_sep_g_cost
            B = start_cell[2]         
            C = [A[0] + B[0] , A[1] + B[1], A[2] + B[2] ]   
            tentative_g_cost = C
            if nb_cell[2][2] > tentative_g_cost[2]:
            ############COMPARE G COST############

                ############SET NB_CELL############
                nb_cell[2] = tentative_g_cost
                nb_cell[1] = [nb_cell[2][0] + nb_cell[3][0], nb_cell[2][1] + nb_cell[3][1], nb_cell[2][2] + nb_cell[3][2]] 
                nb_cell[5] = start_cell
                nb_cell[10] = [start_cell[7][0], start_cell[7][1]]
                ############SET NB_CELL############

                ############ADD AND SORT############
                if not self.open_list:
                    self.open_list.append(nb_cell)
                else: 
                    i = 0; nl = len(self.open_list)
                    while i < nl:
                        list_cell = self.open_list[i] 
                        if nb_cell[1][2] < list_cell[1][2]:
                            break
                        elif nb_cell[1][2] == list_cell[1][2] and nb_cell[3][2] < list_cell[3][2]: 
                            break
                        i += 1  
                    self.open_list.insert(i, nb_cell)                  
                ############ADD AND SORT############

        ###########################PROPAGATE###########################
        while len(self.open_list) >0:

            #Get current cell and check if it is visited
            current_cell = self.open_list.pop(0)
            if current_cell[4]: continue
            else: current_cell[4] = True

            #Found goal
            if current_cell[7] == goal_idx:
                while True:
                    path.append(current_cell[7])
                    current_cell =  self.occ_grid.idx2cell(current_cell[10])
                    if current_cell[5] is None:
                        break
                break

            #PROBLEM!!!!! current_cell.parent == None:
            fwd_direction = (current_cell[7][0] - current_cell[10][0], current_cell[7][1] - current_cell[10][1])
            
            if (abs(fwd_direction[0]),abs(fwd_direction[1])) == (1,1): 
                self.ordinal_search(current_cell, fwd_direction)     
            else: 
                self.cardinal_search(current_cell, fwd_direction)    
                
        ###########################PROPAGATE###########################
        return path

    def ordinal_search(self,current_cell, fwd_direction):

        #get directions for cardinal search
        i = compass.index(fwd_direction)
        if i+1 > 7: j = 0
        else: j = i+1
        
        #perform cardinal search for both sides
        self.cardinal_search(current_cell, compass[i-1])
        self.cardinal_search(current_cell, compass[j])

        while True:
            #Get forward cell
            fwd_cell = self.occ_grid.idx2cell((current_cell[7][0] + fwd_direction[0],current_cell[7][1] + fwd_direction[1]))
            #check for forced neighbours
            forced_nbs = self.get_forced_neighbours(current_cell, fwd_direction)

            #check forced_nb exists and add to open list if cheaper
            if forced_nbs is not None:
                for forced_nb in forced_nbs:
                    ############COMPARE G COST############
                    a = forced_nb[7]
                    b = current_cell[7]
                    dj = fabs(a[1] - b[1])
                    di = fabs(a[0] - b[0])
                    if dj > di:
                        a0 = di
                        a1 = dj-di
                    else:
                        a0 = dj
                        a1 = di-dj
                    dist_sep_g_cost = [float64(a0), float64(a1), SQRT2 * float64(a0) + float64(a1)]
                    A = dist_sep_g_cost
                    B = current_cell[2]
                    C = [A[0] + B[0] , A[1] + B[1], A[2] + B[2] ] 
                    tentative_g_cost = C   
                    ############COMPARE G COST############
                    if forced_nb[2][2] > tentative_g_cost[2]:
                        forced_nb[2] = tentative_g_cost
                        forced_nb[1] = [forced_nb[2][0] + forced_nb[3][0], forced_nb[2][1] + forced_nb[3][1], forced_nb[2][2] + forced_nb[3][2]]                         
                        forced_nb[5] = current_cell
                        forced_nb[10] = [current_cell[7][0], current_cell[7][1]]
                        #open_list_add(forced_nb)
                        if not self.open_list:
                            self.open_list.append(forced_nb)
                        else: 
                            i = 0; nl = len(self.open_list)
                            while i < nl:
                                list_cell = self.open_list[i] 
                                if forced_nb[1][2] < list_cell[1][2]:
                                    break
                                elif forced_nb[1][2] == list_cell[1][2] and forced_nb[3][2] < list_cell[3][2]: 
                                    break
                                i += 1 
                            self.open_list.insert(i, forced_nb)                    
                        #end of open_list_add(forced_nb)

                #Check if front cell is accessible and add to open list if cheaper
                if fwd_cell is not None and (fwd_cell[9] <= L_THRESH and not fwd_cell[8]):
                    
                    ############COMPARE G COST############
                    a = fwd_cell[7]
                    b = current_cell[7]
                    dj = fabs(a[1] - b[1])
                    di = fabs(a[0] - b[0])
                    if dj > di:
                        a0 = di
                        a1 = dj-di
                    else:
                        a0 = dj
                        a1 = di-dj
                    dist_sep_g_cost = [float64(a0), float64(a1), SQRT2 * float64(a0) + float64(a1)]
                    A = dist_sep_g_cost
                    B = current_cell[2]
                    C = [A[0] + B[0] , A[1] + B[1], A[2] + B[2] ]   
                    tentative_g_cost = C
                    ############COMPARE G COST############
                    if fwd_cell[2][2] > tentative_g_cost[2]:
                        fwd_cell[2] = tentative_g_cost
                        fwd_cell[1] = [fwd_cell[2][0] + fwd_cell[3][0], fwd_cell[2][1] + fwd_cell[3][1], fwd_cell[2][2] + fwd_cell[3][2]] 
                        fwd_cell[5] = current_cell
                        fwd_cell[10] = [current_cell[7][0], current_cell[7][1]]
                        ############ADD to openlist############
                        if not self.open_list:
                            self.open_list.append(fwd_cell)
                        else: 
                            i = 0; nl = len(self.open_list)
                            while i < nl:
                                list_cell = self.open_list[i] 
                                if fwd_cell[1][2] < list_cell[1][2]:
                                    break
                                elif fwd_cell[1][2] == list_cell[1][2] and fwd_cell[3][2] < list_cell[3][2]: 
                                    break
                                i += 1
                            self.open_list.insert(i, fwd_cell)                     
                        ############ADD to openlist############
                break
            #move forward if forced_nb doesn't exist
            elif fwd_cell is not None and (fwd_cell[9] <= L_THRESH and not fwd_cell[8]):
                current_cell =  fwd_cell
            #break search if cannot move forward
            else:
                break

    def cardinal_search(self, current_cell, fwd_direction):

        while True:
            #Get forward cell
            fwd_cell = self.occ_grid.idx2cell((current_cell[7][0] + fwd_direction[0],current_cell[7][1] + fwd_direction[1]))
            #check for forced neighbours
            forced_nbs = self.get_forced_neighbours(current_cell, fwd_direction)

            #check forced_nb exists and add to open list if cheaper
            if forced_nbs is not None:
                #Loop through and add forced neighbour to open list if it is cheaper
                for forced_nb in forced_nbs:

                    ############COMPARE G COST############
                    a = forced_nb[7]
                    b = current_cell[7]
                    dj = fabs(a[1] - b[1])
                    di = fabs(a[0] - b[0])
                    if dj > di:
                        a0 = di
                        a1 = dj-di
                    else:
                        a0 = dj
                        a1 = di-dj
                    dist_sep_g_cost = [float64(a0), float64(a1), SQRT2 * float64(a0) + float64(a1)]
                    A = dist_sep_g_cost
                    B = current_cell[2]
                    C = [A[0] + B[0] , A[1] + B[1], A[2] + B[2] ] 
                    tentative_g_cost = C   
                    ############COMPARE G COST############

                    if forced_nb[2][2] > tentative_g_cost[2]:
                        forced_nb[2] = tentative_g_cost
                        forced_nb[1] = [forced_nb[2][0] + forced_nb[3][0], forced_nb[2][1] + forced_nb[3][1], forced_nb[2][2] + forced_nb[3][2]]                         
                        forced_nb[5] = current_cell
                        forced_nb[10] = [current_cell[7][0], current_cell[7][1]]
                        #open_list_add(forced_nb)
                        if not self.open_list:
                            self.open_list.append(forced_nb)
                        else: 
                            i = 0; nl = len(self.open_list)
                            while i < nl:
                                list_cell = self.open_list[i] 
                                if forced_nb[1][2] < list_cell[1][2]:
                                    break
                                elif forced_nb[1][2] == list_cell[1][2] and forced_nb[3][2] < list_cell[3][2]: 
                                    break
                                i += 1 
                            self.open_list.insert(i, forced_nb)                    
                        #end of open_list_add(forced_nb)

                #Check if front cell is accessible and add to open list if cheaper
                if fwd_cell is not None and (fwd_cell[9] <= L_THRESH and not fwd_cell[8]):
                    
                    ############COMPARE G COST############
                    a = fwd_cell[7]
                    b = current_cell[7]
                    dj = fabs(a[1] - b[1])
                    di = fabs(a[0] - b[0])
                    if dj > di:
                        a0 = di
                        a1 = dj-di
                    else:
                        a0 = dj
                        a1 = di-dj
                    dist_sep_g_cost = [float64(a0), float64(a1), SQRT2 * float64(a0) + float64(a1)]
                    A = dist_sep_g_cost
                    B = current_cell[2]
                    C = [A[0] + B[0] , A[1] + B[1], A[2] + B[2] ]   
                    tentative_g_cost = C
                    ############COMPARE G COST############
                    if fwd_cell[2][2] > tentative_g_cost[2]:
                        fwd_cell[2] = tentative_g_cost
                        fwd_cell[1] = [fwd_cell[2][0] + fwd_cell[3][0], fwd_cell[2][1] + fwd_cell[3][1], fwd_cell[2][2] + fwd_cell[3][2]] 
                        fwd_cell[5] = current_cell
                        fwd_cell[10] = [current_cell[7][0], current_cell[7][1]]
                        ############ADD to openlist############
                        if not self.open_list:
                            self.open_list.append(fwd_cell)
                        else: 
                            i = 0; nl = len(self.open_list)
                            while i < nl:
                                list_cell = self.open_list[i] 
                                if fwd_cell[1][2] < list_cell[1][2]:
                                    break
                                elif fwd_cell[1][2] == list_cell[1][2] and fwd_cell[3][2] < list_cell[3][2]: 
                                    break
                                i += 1
                            self.open_list.insert(i, fwd_cell)                     
                        ############ADD to openlist############
                break
            #move forward if forced_nb doesn't exist
            elif fwd_cell is not None and (fwd_cell[9] <= L_THRESH and not fwd_cell[8]):
                current_cell =  fwd_cell
            #break search if cannot move forward
            else:
                break
    
    def get_free_neighbors(self, cell):
        # start from +x (N), counter clockwise
        neighbors = []
        for rel_idx in REL_IDX:
            nb_idx = (rel_idx[0] + cell[7][0], rel_idx[1] + cell[7][1]) #non-numpy
            nb_cell =  self.occ_grid.idx2cell(nb_idx)
            #if nb_cell exists and nb_cell is free
            if nb_cell is not None and (nb_cell[9] <= L_THRESH and not nb_cell[8]):
                neighbors.append(nb_cell)
        return neighbors

    def get_forced_neighbours(self, current_cell, direction):
        i = compass.index(direction)
        idx_0 = current_cell[7][0]
        idx_1 = current_cell[7][1]
        i_plus_1 = i+1
        i_plus_2 = i+2
        if i == 6:
            i_plus_1 = 7
            i_plus_2 = 0
        if i == 7:
            i_plus_1 = 0
            i_plus_2 = 1

        nbs = []
        fwd_cell = self.occ_grid.idx2cell((idx_0 + direction[0],idx_1 + direction[1]))
    
        #Only checks for forced neighbours if fwd cell is not occupied
        if fwd_cell is not None and (fwd_cell[9] <= L_THRESH and not fwd_cell[8]):
            L_cell = self.occ_grid.idx2cell( (idx_0 + (compass[i-2])[0], idx_1 + (compass[i-2])[1] )) #potential obstacle
            R_cell = self.occ_grid.idx2cell(( idx_0 + (compass[i_plus_2])[0], idx_1 + (compass[i_plus_2])[1] )) #potential obstacle           
            
            #Check left and right cells for forced nb
            if (L_cell is not None) and not (L_cell[9] <= L_THRESH and not L_cell[8]):
                L_diag_cell = self.occ_grid.idx2cell(( idx_0 + (compass[i-1])[0], idx_1 + (compass[i-1])[1] )) #potential forced neighbour
                if L_diag_cell is not None and (L_diag_cell[9] <= L_THRESH and not L_diag_cell[8]): 
                    nbs.append(L_diag_cell)
                    
            if (R_cell is not None) and not (L_cell[9] <= L_THRESH and not L_cell[8]):
                R_diag_cell = self.occ_grid.idx2cell(( idx_0 + (compass[i_plus_1])[0], idx_1 + (compass[i_plus_1])[1] )) #potential forced neighbour
                if R_diag_cell is not None and (R_diag_cell[9] <= L_THRESH and not R_diag_cell[8]): 
                    nbs.append(R_diag_cell)
        return nbs



# ================================ BEGIN ===========================================
def main(goals, cell_size, min_pos, max_pos):
    # ---------------------------------- INITS ----------------------------------------------
    CELL_SIZE = cell_size
    
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
        
    if rospy.is_shutdown():
        return
        
    # Data structures
    occ_grid = lab3_aux.OccupancyGrid(min_pos, max_pos, CELL_SIZE, 0, INF_RADIUS)
    los = lab3_aux.GeneralLOS()
    los_int = lab3_aux.GeneralIntLOS()
    # ~ moved motion model to lab3_move.py    
    planner = JPS(occ_grid)
    
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
    post_process = lab3_aux.post_process
    
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
                    path, pts = post_process(planner.path, occ_grid.c_inf, los_int, los) 
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
        # parse goals
        goals = sys.argv[1]
        goals = goals.split('|')
        for i in xrange(len(goals)):
            tmp = goals[i].split(',')
            tmp[0] = float(tmp[0])
            tmp[1] = float(tmp[1])
            goals[i] = tmp
        
        # parse cell_size
        cell_size = float(sys.argv[2])
        
        # parse min_pos
        min_pos = sys.argv[3]
        min_pos = min_pos.split(',')
        min_pos = (float(min_pos[0]), float(min_pos[1]))
        
        # parse max_pos
        max_pos = sys.argv[4]
        max_pos = max_pos.split(',')
        max_pos = (float(max_pos[0]), float(max_pos[1]))
        
        main(goals, cell_size, min_pos, max_pos)
    except rospy.ROSInterruptException:
        pass


