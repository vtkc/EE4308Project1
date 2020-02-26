#!/usr/bin/env python

import roslib, rospy, rospkg
from numpy import *
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, JointState #Imu
from nav_msgs.msg import Odometry
from std_msgs import *
from tf.transformations import quaternion_from_euler, euler_from_quaternion, quaternion_multiply
from tf2_msgs.msg import TFMessage
import cv2
import numpy
import lab2_aux

# ================================= CONSTANTS ==========================================        
# let's cache the SIN and POS so we don't keep recalculating it, which is slow
DEG2RAD = [i/180.0*pi for i in xrange(360)] # DEG2RAD[3] means 3 degrees in radians
SIN = [sin(DEG2RAD[i]) for i in xrange(360)] # SIN[32] means sin(32degrees)
COS = [cos(DEG2RAD[i]) for i in xrange(360)]
MAX_RNG = 3.5
PATH_PKG = rospkg.RosPack().get_path('pkg') + '/'
PATH_WORLDS = PATH_PKG + 'worlds/'
SQRT2 = sqrt(2)
REL_IDX = ((1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1), (0,-1), (1,-1))
L_THRESH = lab2_aux.L_THRESH

# ================================== DATA STRUCTS ===========================================

class Cell:
    def __init__(self, idx, initial_value=0):
        """ Constructor for occupancy grid's cells
        Parameters:
            initial_value (float64):    The initial occupancy value. Default is 0
            idx (tuple of int64):       Index of cell
        """
        self.initial_value = initial_value
        self.g_cost = Distance(inf, inf)
        self.h_cost = Distance(0, 0)
        self.f_cost = Distance(inf, inf)
        self.visited = False
        self.parent = None
        self.update = 0
        self.idx = idx
        self.inf = set()
        self.occ = self.initial_value
    def reset_for_planner(self, goal_idx):
        """ Resets the cells for every run of non-dynamic path planners
        Parameters:
            goal_idx (tuple of int64):  Index of goal cell
        """
        self.g_cost = Distance(inf, inf)
        self.h_cost = Distance.from_separation(self.idx, goal_idx)
        self.f_cost = Distance(inf, inf)
        self.visited = False
        self.parent = None
        self.update = 0
    def set_occupancy(self, occupied=True):
        """ Updates the cell's observed occupancy state using the log-odds 
            Binary Bayes model
        Parameters:
            occupied (bool):    If True, the cell was observed to be occupied. 
                                False otherwise. Default is True
        """
        if occupied == True:
            self.occ += 1
        else:
            self.occ -= 1
        # lab2_aux.set_occupancy(self, occupied)
    def set_inflation(self, origin_idx, add=True):
        """ Updates the cell's inflation state
        Parameters:
            origin_idx (tuple of int64):    Index of cell that is 
                                            causing the current cell to 
                                            be inflated / deflated
            add (bool): If True, the cell at origin_idx is newly marked 
                        as occupied. If False, the cell at origin_idx 
                        is newly marked as free
        """
        if add:
            self.inf.add(origin_idx)
        else:
            self.inf.discard(origin_idx)
    def is_occupied(self):
        """ Returns True if the cell is certainly occupied
        Returns:
            bool : True if cell is occupied, False if unknown or free
        """
        return self.occ > L_THRESH
    def is_inflation(self):
        """ Returns True if the cell is inflated
        Returns:
            bool : True if cell is inflated
        """
        return not not self.inf # zero length
    def is_free(self):
        """ Returns True if the cell is certainly free.
        Returns:
            bool : True if cell is free, False if unknown or occupied
        """
        return self.occ < -L_THRESH 
    def is_unknown(self):
        """ Returns True if the cell's occupancy is unknown
        Returns:
            bool : True if cell's occupancy is unknown, False otherwise
        """
        return self.occ >= -L_THRESH and self.occ <= L_THRESH
    def is_planner_free(self):
        """ Returns True if the cell is traversable by path planners
        Returns:
            bool : True if cell is unknown, free and not inflated, 
        """
        return self.occ <= L_THRESH and not self.inf
    def set_g_cost(self, g_cost):
        """ Sets the g-cost of the cell and recalculates the f-cost
        Parameters:
            g_cost (Distance): the Distance instance specifying the g-cost
        """
        self.g_cost = g_cost
        self.f_cost = self.g_cost + self.h_cost
    def set_h_cost(self, h_cost):
        """ Sets the h-cost of the cell and recalculates the f-cost
        Parameters:
            h_cost (Distance): the Distance instance specifying the h-cost
        """
        self.h_cost = h_cost
        self.f_cost = self.g_cost + self.h_cost
    def __str__(self):
        """ Returns the string representation of the cell, 
            useful for debugging in print()
        Returns:
            str : the string containing useful information of the cell
        """
        return 'Cell{} occ:{}, f:{:6.2f}, g:{:6.2f}, h:{:6.2f}, visited:{}, parent:{}'\
        .format(self.idx, self.occ, self.f_cost.total, self.g_cost.total, self.h_cost.total, self.visited, \
        self.parent.idx if self.parent else 'None')
        
class OccupancyGrid: # Occupancy Grid
    def __init__(self, min_pos, max_pos, cell_size, initial_value, inflation_radius): 
        """ Constructor for Occupancy Grid
        Parameters:
            min_pos (tuple of float64): The smallest world coordinate (x, y). This 
                                        determines the lower corner of the rectangular 
                                        grid
            max_pos (tuple of float64): The largest world coordinate (x, y). This 
                                        determines the upper corner of the rectangular 
                                        grid
            cell_size (float64): The size of the cell in the real world, in meters.
            initial_value (float64): The initial value that is assigned to all cells
            inflation_radius (float64): Inflation radius. Cells lying within the inflation 
                                        radius from the center of an occupied cell will 
                                        be marked as inflated
        """
        di = int64(round((max_pos[0] - min_pos[0])/cell_size))
        dj = int64(round((max_pos[1] - min_pos[1])/cell_size))
        self.cell_size = cell_size
        self.min_pos = min_pos
        self.max_pos = max_pos
        di += 1; dj += 1
        self.num_idx = (di, dj) # number of (rows, cols)
        self.cells = [[Cell((i,j), initial_value) for j in xrange(dj)] for i in xrange(di)]
        self.mask_inflation = gen_mask(cell_size, inflation_radius)
        
        # CV2 inits
        self.img_mat = full((di, dj, 3), uint8(127))
        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('img', di*5, dj*5) # so each cell is 5px*5px
    def idx2pos(self, idx):
        """ Converts indices (map indices) to position (world coordinates)
        Parameters:
            idx (tuple of float64 or tuple of int64): Index tuple (i, j)
        Returns:
            tuple of float64: (i, j)
        """
        w = self.cell_size
        mp = self.min_pos
        return (idx[0] * w + mp[0], idx[1] * w + mp[1])
    def pos2idx(self, pos, rounded=True): 
        """ Converts position (world coordinates) to indices (map indices)
        Parameters:
            pos (tuple of float64): Position tuple (x, y)
            rounded (bool): By default True. Set True to return a integer indices that can 
                            be used to access the array. Set False to return exact indices.
        Returns:
            tuple of int64 or tuple of float64: (i, j), exact indices or integer indices
        """
        w = self.cell_size
        mp = self.min_pos
        idx = ((pos[0] - mp[0])/w, (pos[1] - mp[1])/w)
        if rounded:
            return (int64(round(idx[0])), int64(round(idx[1])))
        return idx
    def idx_in_map(self, idx): # idx must be integer
        """ Checks if the given index is within map boundaries
        Parameters:
            idx (tuple of int64): Index tuple (i, j) to be checked
        Returns:
            bool: True if in map, False if outside map
        """
        i, j = idx
        return i >= 0 and i < self.num_idx[0] and j >= 0 and j < self.num_idx[1]
    def idx2cell(self, idx, is_int=False):
        """ Retrieves the cell at a given index
        Parameters:
            idx (tuple of int64 or tuple of float64): Index tuple (i, j) of cell.
            is_int: By default False. If False, the tuple is converted to tuple of int64 by 
                    rounding. Set to True to skip this only if idx is tuple of int64
        Returns:
            None or float64:    None if the idx is outside the map. float64 value of the  
                                cell if idx is in the map.
        """
        if not is_int:
            idx = (int64(round(idx[0])), int64(round(idx[1])))
        if self.idx_in_map(idx):
            return self.cells[idx[0]][idx[1]]
        return None
    def update_at_idx(self, idx, occupied):
        """ Updates the cell at the index with the observed occupancy
        Parameters:
            idx (tuple of int64): The Index of the cell to update
            occupied (bool):    If True, the cell is currently observed to be occupied. 
                                False if free.
        """
        ok = self.idx_in_map
        # return if not in map
        if not ok(idx):
            return
        c = self.cells
        cell = c[idx[0]][idx[1]]
        
        # update occupancy
        was_occupied = cell.is_occupied()
        cell.set_occupancy(occupied)
        
        # check if the cell occupancy state is different, and update the masks accordingly 
        # (much faster than just blindly updating regardless of previous state)
        is_occupied = cell.is_occupied()
        if was_occupied != is_occupied:
            for rel_idx in self.mask_inflation:
                i = rel_idx[0] + idx[0]
                j = rel_idx[1] + idx[1]
                mask_idx = (i,j)
                if ok(mask_idx): # cell in map
                    c[i][j].set_inflation(idx, is_occupied)
    def update_at_pos(self, pos, occupied):
        """ Updates the cell at the position with the observed occupancy
        Parameters:
            pos (tuple of float64): The position of the cell to update
            occupied (bool):    If True, the cell is currently observed to be occupied. 
                                False if free.
        """
        self.update_at_idx(self.pos2idx(pos), occupied)
    def show_map(self, rbt_idx, path=None, goal_idx=None):
        """ Prints the occupancy grid and robot position on it as a picture in a resizable 
            window
        Parameters:
            rbt_pos (tuple of float64): position tuple (x, y) of robot.
        """
        c = self.cells
        img_mat = self.img_mat.copy()
        ni, nj = self.num_idx
        for i in xrange(ni):
            cc = c[i]
            for j in xrange(nj):
                cell = cc[j]
                if cell.is_occupied():
                    img_mat[i, j, :] = (255, 255, 255) # white
                elif cell.is_inflation():
                    img_mat[i, j, :] = (180, 180, 180) # light gray
                elif cell.is_free():
                    img_mat[i, j, :] = (0, 0, 0) # black
                    
        if path is not None:
            for k in xrange(len(path)):
#                idx = path[k]; next_idx = path[k+1]
                i, j = path[k]
                img_mat[i, j, :] = (0, 0, 255) # red
#                cv2.line(img_mat, idx, next_idx, (0,0,255), 1)
            
            
        # color the robot position as a crosshair
        img_mat[rbt_idx[0], rbt_idx[1], :] = (0, 255, 0) # green
        
        if goal_idx is not None:
            img_mat[goal_idx[0], goal_idx[1], :] = (255, 0, 0) # blue

        # print to a window 'img'
        cv2.imshow('img', img_mat)
        cv2.waitKey(10)

# =============================== PLANNER CLASSES =========================================   
class Distance:
    def __init__(self, ordinals=0, cardinals=0):
        """ The constructor for more robust octile distance calculation. 
            Stores the exact number of ordinals and cardinals and calculates
            the approximate float64 value of the resultant cost in .total
        Parameters:
            ordinals (int64): The integer number of ordinals in the distance
            cardinals (int64): The integer number of cardinals in the distance
        """
        self.axes = [float64(ordinals), float64(cardinals)]
        self.total = SQRT2 * self.axes[0] + self.axes[1]
    def __add__(self, distance):
        """ Returns a new Distance object that represents the addition of the current
            Distance object with another Distance object. For use with '+' operator
        Parameters:
            distance (Distance): the other Distance object to add
        Returns:
            Distance : the new Distance object
        """
        return Distance(self.axes[0] + distance.axes[0], self.axes[1] + distance.axes[1])
    def __eq__(self, distance):
        """ Returns True if the current has the same number of ordinals and cardinals 
            as distance. Used with '==' operator
        Parameters:
            distance (Distance): the other Distance object to equate
        Returns:
            bool : True if same number of ordinals and cardinals, False otherwise
        """
        return distance.axes[0] == self.axes[0] and distance.axes[1] == self.axes[1]
    def __ne__(self, distance):
        """ Returns True if the current has different number of ordinals and cardinals 
            as distance. Used with '!=' operator
        Parameters:
            distance (Distance): the other Distance object to equate
        Returns:
            bool : True if different number of ordinals and cardinals, False otherwise
        """
        return distance.axes[0] != self.axes[0] or distance.axes[1] != self.axes[1]
    def __lt__(self, distance):
        """ Returns True if the current Distance is less than distance.
            False otherwise. Used with '<' operator
        Parameters:
            distance (Distance): the other Distance object to check
        Returns:
            bool : True if the current.total is less than distance.total
        """
        return self.total < distance.total
    def __gt__(self, distance):
        """ Returns True if the current Distance is greater than distance.
            False otherwise. Used with '>' operator
        Parameters:
            distance (Distance): the other Distance object to check
        Returns:
            bool : True if the current.total is more than distance.total
        """
        return self.total > distance.total
    def __le__(self, distance):
        """ Returns True if the current Distance is less than or equals distance.
            False otherwise. Used with '<=' operator
        Parameters:
            distance (Distance): the other Distance object to check
        Returns:
            bool :  True if current has the same number of cardinals and ordinals as 
                    distance or if the current.total is less than distance.total. 
                    Used with '<=' operator
        """
        return distance.axes[0] == self.axes[0] and distance.axes[1] == self.axes[1]\
            or self.total < distance.total
    def __ge__(self, distance):
        """ Returns True if the current Distance is greater than or equals distance.
            False otherwise. Used with '>=' operator
        Parameters:
            distance (Distance): the other Distance object to check
        Returns:
            bool :  True if current has the same number of cardinals and ordinals as 
                    distance or if the current.total is greater than distance.total. 
                    Used with '>=' operator
        """
        return distance.axes[0] == self.axes[0] and distance.axes[1] == self.axes[1]\
            or self.total > distance.total
    def __str__(self):
        """ Returns the string representation of the current Distance, 
            useful for debugging in print()
        Returns:
            str : the string containing useful information of the Distance
        """
        return '${:6.2f}, O:{:6.2f}, C:{:6.2f}'.format(self.total, self.axes[0], self.axes[1])
    @staticmethod
    def from_separation(idx0, idx1):
        """ static method that returns a Distance object based on the octile distance
            between two indices idx0 and idx1
        Parameters:
            idx0 (tuple of int64): An index tuple
            idx1 (tuple of int64): Another index tuple
        """
        dj = fabs(idx0[1] - idx1[1])
        di = fabs(idx0[0] - idx1[0])
        if dj > di:
            a0 = di
            a1 = dj-di
        else:
            a0 = dj
            a1 = di-dj
        return Distance(a0, a1)
      
class Astar:
    def __init__(self, occ_grid):
        """ A* Path planner
        Parameters:
            occ_grid (OccupancyGrid) : The occupancy grid
        """
        self.occ_grid = occ_grid
        self.open_list = OpenList()
    def get_path(self, start_idx, goal_idx):
        """ Returns a list of indices that represents the octile-optimal
            path between the starting index and the goal index
        Parameters:
            start_idx (tuple of int64): the starting index
            goal_idx (tuple of int64): the goal index
        Returns:
            list of tuple of int64: contains the indices in the optimal path
        """
        occ_grid = self.occ_grid
        open_list = self.open_list
        # get number of rows ni (x) and number of columns nj (y)
        ni, nj = occ_grid.num_idx
        path = []
        
        # resets h-cost, g-cost, update and occ for all cells
        for i in xrange(ni):
            for j in xrange(nj):
                # ! use occ_grid.idx2cell() and the cell's reset_for_planner()
                # pass
                if occ_grid.idx2cell((i,j)) != None:
                    occ_grid.idx2cell((i,j)).reset_for_planner(goal_idx)
                
        # put start cell into open list
        # ! get the start cell from start_idx
        # ! set the start cell distance using set_g_cost and Distance(0, 0)
        # ! add the cell to open_list
        
        # ! get the start cell from start_idx
        start_cell = occ_grid.idx2cell(start_idx)
        # ! set the start cell distance using set_g_cost and Distance(0, 0)
        start_cell.set_g_cost(Distance(0, 0))
        # ! add the cell to open_list
        open_list.add(start_cell)
        
        # now we non-recursively search the map
        while open_list.not_empty():
            cell = open_list.remove()
            # skip if already visited, bcos a cheaper path was already found
            if cell.visited:
                continue
                
            # ! set the cell as visited
            cell.visited = True
            
            # goal
            if cell.idx == goal_idx:
                # while cell.parent is not None:
                #     # ! append the cell.idx onto path
                #     # ! let cell = cell's parent
                #     # ! if cell is None, break out of the while loop
                #     # pass
                #     path.append(cell.idx)
                #     cell = cell.parent
                # path.append(cell.idx)
                while True:
                    path.append(cell.idx)
                    cell = cell.parent
                    if cell == None:
                        break
                    
                break # breaks out of the loop: while open_list.not_empty()
                
            # if not goal or not visited, we try to add free neighbour cells into the open list
            for nb_cell in self.get_free_neighbors(cell):
                # ! calculate the tentative g cost of getting from current cell (cell) to neighbouring cell (nb_cell)...
                # !     use cell.g_cost and Distance.from_separation()
                # ! if the tentative g cost is less than the nb_cell.g_cost, ...
                # !     1. assign the tentative g cost to nb_cell's g cost using set_g_cost
                # !     2. set the nb_cell parent as cell
                # !     3. add the nb_cell to the open list using open_list.add()
                # pass
                tent_g_cost = Distance.from_separation(nb_cell.idx, cell.idx) + cell.g_cost
                if tent_g_cost < nb_cell.g_cost:
                    nb_cell.set_g_cost(tent_g_cost)
                    nb_cell.parent = cell
                    open_list.add(nb_cell)
                    
        return path
            
    def get_free_neighbors(self, cell):
        """ Checks which of the 8 neighboring cells around a cell are in the map, 
            free, unknown and not inflated and returns them as a list
        Parameters:
            cell (Cell): the cell in the occupancy grid
        Returns:
            list of Cells: the list of neighboring cells which are in the map, 
            free, unknown and not inflated
        """
        # start from +x (N), counter clockwise
        occ_grid = self.occ_grid
        neighbors = []
        idx = cell.idx
        for rel_idx in REL_IDX:
            nb_idx = (rel_idx[0] + idx[0], rel_idx[1] + idx[1]) #non-numpy
            nb_cell = occ_grid.idx2cell(nb_idx)
            if nb_cell is not None and nb_cell.is_planner_free():
                neighbors.append(nb_cell)
        return neighbors
        
class OpenList:
    def __init__(self):
        """ The constructor for the open list
        """
        # initialise with an list (array)
        self.l = []
    def add(self, cell):
        """ Adds the cell and sorts it based on its f-cost followed by the h-cost
        Parameters:
            cell (Cell): the Cell to be sorted, updated with f-cost and h-cost information
        """
        # set l as the open list
        l = self.l
        
        # if l is empty, just append and return
        if not l:
            l.append(cell)
            return
        
        # now we sort and add
        i = 0; nl = len(l)
        # we start searching from index (i) 0, where the cells should be cheapest
        while i < nl:
            # ! get the cell (list_cell) in the index (i) of the open list (l)
            # ! now if the cell's f_cost is less than the list_cell's f_cost, ...
            # !     or if the cell's f_cost = list_cell's f_cost but ...
            # !     cell's h_cost is less than the list_cell's h_cost...
            # !     we break the loop (while i < nl)
            if cell.f_cost < l[i].f_cost :
                break
            elif cell.f_cost== l[i].f_cost and cell.h_cost < l[i].h_cost :
                break
            # increment the index
            i += 1
            
        # insert the cell into position i of l
        l.insert(i, cell)
    def remove(self):
        """ Removes and return the cheapest cost cell in the open list
        Returns:
            Cell: the cell with the cheapest f-cost followed by h-cost
        """
        # return the first element in self.l
        # pass
        return self.l.pop(0)
    def not_empty(self):
        return not not self.l # self.l is False if len(self.l) is zero ==> faster
    def __str__(self):
        l = self.l
        s = ''
        for cell in l:
            s += '({:3d},{:3d}), F:{:6.2f}, G:{:6.2f}, H:{:6.2f}\n'.format(\
                cell.idx[0], cell.idx[1], cell.f_cost.total, cell.g_cost.total, cell.h_cost.total)
        return s
    
# =============================== SUBSCRIBERS =========================================  
def subscribe_true(msg):
    # subscribes to the robot's true position in the simulator. This should not be used, for checking only.
    global rbt_true
    msg_tf = msg.transforms[0].transform
    rbt_true = (\
        msg_tf.translation.x, \
        msg_tf.translation.y, \
        euler_from_quaternion([\
            msg_tf.rotation.x, \
            msg_tf.rotation.y, \
            msg_tf.rotation.z, \
            msg_tf.rotation.w, \
        ])[2]\
    )
    
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
    
def subscribe_wheels(msg):
    global rbt_wheels
    # rbt_wheels = lab2_aux.get_wheels(msg)
    right_wheel_angle = msg.position[0] # examine topic /joint_states
    left_wheel_angle = msg.position[1] # examine topic /joint_states
    rbt_wheels = (left_wheel_angle, right_wheel_angle)
    
def get_scan():
    # returns scan data after acquiring a lock on the scan data to make sure it is not overwritten by the subscribe_scan handler while using it.
    global write_scan, read_scan
    read_scan = True # lock
    while write_scan:
        pass
    scan = rbt_scan # create a copy of the tuple
    read_scan = False
    return scan
    
    
# ================================== PUBLISHERS ========================================
# none

# =================================== OTHER METHODS =====================================
def gen_mask(cell_size, radius):
    """ Generates the list of relative indices of neighboring cells which lie within 
        the specified radius from a center cell
    Parameters:
        radius (float64): the radius 
    """
    nCells_radius = int64(numpy.round(float64(radius) / float64(cell_size)))
    
    mask = []
    i = 0
    j = 0
    for i in xrange(0, nCells_radius):
        for j in xrange(0, nCells_radius):
            relative_distance_small = sqrt(pow((((cell_size/2)*nCells_radius) - ((cell_size/2)*i)),2) 
                                           + pow((((cell_size/2)*nCells_radius) - ((cell_size/2)*j)),2))
            if relative_distance_small < radius:
                mask.append((i ,j))
                mask.append((-i ,j))
                mask.append((i ,-j))
                mask.append((-i ,-j))
          
    return mask
    # return lab2_aux.gen_mask(cell_size, radius)

# ----------------------------------OWN IMPLEMENTATION-----------------------------------
class OdometryMM:
    def __init__(self, initial_pose, initial_wheels, axle_track, wheel_dia):
        self.x = initial_pose[0] # m, robot's x position in world
        self.y = initial_pose[1] # m, robot's y position in world
        self.o = initial_pose[2] # rad, robot's bearing in world
        self.wl = initial_wheels[0] # rad, robot's left angle
        self.wr = initial_wheels[1] # rad, robot's right angle
        self.L = axle_track # m, robot's axle track
        self.WR = wheel_dia/2.0 # m, robot's wheel RADIUS, not DIAMETER
        self.t = rospy.get_time() # s, time last calculated
    def calculate(self, wheels):
        # calculates the robot's new pose based on wheel encoder angles
        # INPUT: wheels: (left_wheel_angle, right_wheel_angle)
        # OUTPUT: a new pose (x, y, theta)
        
        # previous wheel angles stored in self.wl and self.wr, respectively. Remember to overwrite them
        # previous pose stored in self.x, self.y, self.o, respectively. Remember to overwrite them
        # previous time stored in self.t. Remember to overwrite it
        # axle track stored in self.L. Should not be overwritten.
        # wheel radius, NOT DIAMETER, stored in self.WR. Should not be overwritten.
        dt = rospy.get_time() - self.t # current time minus previous time
        dwl = wheels[0] - self.wl 
        dwr = wheels[1] - self.wr
        
        # Own Implementation of Odometry Motion Model
        
        # vt = (((self.WR * 2) / (4 * dt)) * (dwl + dwr))
        # dPhi = (((self.WR * 2) / (2 * self.L)) * (dwr - dwl))
        # dwt = (((self.WR * 2) / (2 * self.L * dt)) * (dwr - dwl))
        
        vt = ((self.WR*2)/4)*((dwr/dt)+(dwl/dt))
        dPhi = ((self.WR*2)/(2*self.L))*(dwr-dwl) 
        dwt = ((self.WR*2)/(2*self.L))*(((dwr/dt) - (dwl/dt)))
        
        if abs(dwt) < 1e-10:
        #   MM for move straight
            self.x = self.x + (vt * dt * math.cos(self.o)) #???
            self.y = self.y + (vt * dt * math.sin(self.o)) #???
            self.o = self.o #???
        
        else:
        #   MM for curve turns
            rt = ((self.L * (dwr + dwl)) / (2 * (dwr - dwl)))
            
            self.x = self.x + ((rt * math.sin(self.o + dPhi)) - (rt * math.sin(self.o))) #???
            self.y = self.y + ((rt * math.cos(self.o)) - (rt * math.cos(self.o + dPhi)))  #???
            self.o = self.o + dPhi #???
        
        # End of Own Implementation
        
        # self.x = 0 #???
        # self.y = 0 #???
        # self.o = 0 #???
        # if abs(w) < 1e-10:
        #   MM for move straight
        # else:
        #   MM for curve turns
        self.wl = wheels[0]
        self.wr = wheels[1]
        self.t = self.t + dt # update the current time. There's a reason why resampling the time is discouraged
        return (self.x, self.y, self.o)

class LOS:
    def __init__(self, map):
        self.pos2idx = map.pos2idx # based on the map (occ_grid) it return's the map index representation of the position pos
        # use self.pos2idx(pos, False) to return the exact index representation, including values that are less than 1.
        # use self.pos2idx(pos) to return the integer index representation, which is the rounded version of self.pos2idx(pos, False)
    def calculate(self, start_pos, end_pos):
        # sets up the LOS object to prepare return a list of indices on the map starting from start_pos (world coordinates) to end_pos (world)
        # start_pos is the robot position.
        # end_pos is the maximum range of the LIDAR, or an obstacle.
        # every index returned in the indices will be the index of a FREE cell
        # you can return the indices, or update the cells in here
        start_idx = self.pos2idx(start_pos)
        end_idx = self.pos2idx(end_pos)
        indices = [] # init an empty list
            
        # Start of General Line Algo
        
        # Getting Difference between start and final pose
        xi = float64(start_idx[0])
        yi = float64(start_idx[1])
        
        xf = float64(end_idx[0])
        yf = float64(end_idx[1])
        
        deltaX = xf - xi
        deltaY = yf - yi
        
        if (abs(deltaX) > abs(deltaY)):
            deltaLong = deltaX
            deltaShort = deltaY
            get_idx = lambda (intLineLongIni, intLineShortIni) : (intLineLongIni, intLineShortIni)
        else:
            deltaLong = deltaY
            deltaShort = deltaX
            get_idx = lambda (intLineLongIni, intLineShortIni) : (intLineShortIni, intLineLongIni)
        
        (lineLongIni, lineShortIni) = get_idx((xi, yi))
        (lineLongEnd, lineShortEnd) = get_idx((xf, yf))
        
        # Get integer (index) representation for accessing map
        intLineLongIni = round(lineLongIni)
        intLineShortIni = round(lineShortIni)
        intLineLongEnd = round(lineLongEnd)
        intLineShortEnd = round(lineShortEnd)
        
        # Get signs and increments
        delS = sign(deltaShort)
        delL = sign(deltaLong)
        psiS = deltaShort / abs(deltaLong)
        
        # Get Error
        errorS = lineShortIni - intLineShortIni
        
        # Get Lamda
        lam = abs(deltaShort / deltaLong) * (0.5 + (lineLongIni - intLineLongIni) * delL) - 0.5
        
        # Get Error Checker
        if (deltaShort >= 0):
            has_big_error = lambda e_k_s : e_k_s >= 0.5
        else:
            has_big_error = lambda e_k_s : e_k_s < -0.5
        
        # Initialize Counter
        i = 0
        
        # Propagate
        while (intLineLongIni, intLineShortIni) != (intLineLongEnd, intLineShortEnd):
            # print((intLineLongIni, intLineShortIni), (intLineLongEnd, intLineShortEnd), i)
            i += 1
            if i == 100:
                raise Exception('e')
            intLineLongIni += delL
            errorS += psiS
            
            if has_big_error(errorS):
                errorS -= delS
                intLineShortIni += delS
                # Previous Cell(s)
                lam_bar = errorS * delS
                if (lam_bar < lam): # Short Direction
                    idx = get_idx((int64(intLineLongIni), int64(intLineShortIni) - int64(delS)))
                    indices.append(idx)
                elif (lam_bar > lam): # Long Direction
                    idx = get_idx((int64(intLineLongIni) - int64(delL), int64(intLineShortIni)))
                    indices.append(idx)
                else: # Both Directions
                    idx = get_idx((int64(intLineLongIni) - int64(delL), int64(intLineShortIni)))
                    indices.append(idx)
                    idx = get_idx((int64(intLineLongIni), int64(intLineShortIni) - int64(delS)))
                    indices.append(idx)
            # Current Cell
            idx = get_idx((int64(intLineLongIni), int64(intLineShortIni)))
            indices.append(idx)
            
        # End of General Line Algo
        
        
        # indices.append(start_idx) # append the starting index into the cell
        indices.pop()
        return indices

def inverse_sensor_model(rng, deg, pose):
    # degree is the bearing in degrees # convert to radians
    # range is the current range data at degree
    # pose is the robot 3DOF pose, in tuple form, (x, y, o)
    x, y, o = pose
    xk = x + rng * math.cos(o + radians(deg)) #???
    yk = y + rng * math.sin(o + radians(deg)) #???
    return (xk, yk)

class VelocityMM:
    def __init__(self, initial_pose, axle_track, wheel_dia):
        self.x = initial_pose[0] # m, robot's x position in world
        self.y = initial_pose[1] # m, robot's y position in world
        self.o = initial_pose[2] # rad, robot's bearing in world
        self.L = axle_track # m, robot's axle track
        self.WR = wheel_dia/2.0 # m, robot's wheel RADIUS, not DIAMETER
        self.t = rospy.get_time() # s, time last calculated
    def calculate(self, control):
        # calculates the robot's new pose based on wheel encoder angles
        # INPUT: control: (vt, wt)
        # OUTPUT: a new pose (x, y, theta)
        
        # previous pose stored in self.x, self.y, self.o, respectively. Remember to overwrite them
        # previous time stored in self.t. Remember to overwrite it
        # axle track stored in self.L. Should not be overwritten.
        # wheel radius, NOT DIAMETER, stored in self.WR. Should not be overwritten.
        dt = rospy.get_time() - self.t # current time minus previous time
        
        # Own implementation of Velocity Motion Model
                                     
        if abs(control[1]) < 1e-10:
        #   MM for move straight
            self.x = self.x + (control[0] * dt * math.cos(self.o)) #???
            self.y = self.y + (control[0] * dt * math.sin(self.o))#???
            self.o = self.o #???
        
        else:
        #   MM for curve turns
            self.x = self.x + (((control[0] / control[1]) * math.sin(self.o + control[1] * dt)) - ((control[0] / control[1]) * math.sin(self.o))) #???
            self.y = self.y + (((control[0] / control[1]) * math.cos(self.o)) - ((control[0] / control[1]) * math.cos(self.o + control[1] * dt))) #???
            self.o = self.o + (control[1] + dt) #???
            
        # End of Own Implementation 
            
        # if abs(w) < 1e-10:
        #   MM for move straight
        # else:
        #   MM for curve turns
        self.t = self.t + dt # update the current time. There's a reason why resampling the time is discouraged
        return (self.x, self.y, self.o)

# ================================ BEGIN ===========================================
MotionModel = OdometryMM
LOS = LOS
inverse_sensor_model = inverse_sensor_model
PathPlanner = Astar
def main():
    # ---------------------------------- INITS ----------------------------------------------
    # init node
    rospy.init_node('main')
    
    # Set the labels below to refer to the global namespace (i.e., global variables)
    # global is required for writing to global variables. For reading, it is not necessary
    global rbt_scan, rbt_true, read_scan, write_scan, rbt_wheels, rbt_control
    
    # Initialise global vars with NaN values 
    # nan and inf are imported from numpy. If you use "import numpy as np", then nan is np.nan, and inf is np.inf.
    rbt_scan = [nan]*360 # a list of 360 nans
    rbt_true = [nan]*3
    rbt_wheels = [nan]*2
    read_scan = False
    write_scan = False

    # Subscribers
    rospy.Subscriber('scan', LaserScan, subscribe_scan, queue_size=1)
    rospy.Subscriber('tf', TFMessage, subscribe_true, queue_size=1)
    rospy.Subscriber('joint_states', JointState, subscribe_wheels, queue_size=1)
    
    # Wait for Subscribers to receive data.
    while isnan(rbt_scan[0]) or isnan(rbt_true[0]) or isnan(rbt_wheels[0]):
        pass
    
    # Data structures
    occ_grid = OccupancyGrid((-2,-3), (5,4), 0.1, 0, 0.2)
    los = LOS(occ_grid)
    motion_model = MotionModel((0,0,0), (0,0), 0.16, 0.066)
    planner = PathPlanner(occ_grid)
    goal_pos = (3.3, 1)
    
    # ---------------------------------- BEGIN ----------------------------------------------
    t = rospy.get_time()
    update = 0
    while (not rospy.is_shutdown()): # required to Keyboard interrupt nicely
        
        if (rospy.get_time() > t): # every 50 ms
            
            # get scan
            scan = get_scan()
            
            # calculate the robot position using the motion model
            rbt_pos = motion_model.calculate(rbt_wheels)
            
            # increment update iteration number
            update += 1
            
            # for each degree in the scan
            for i in xrange(360):
                if scan[i] != inf: # range reading is < max range ==> occupied
                    end_pos = inverse_sensor_model(scan[i], i, rbt_pos)
                    # set the obstacle cell as occupied
                    occ_grid.update_at_pos(end_pos, True)
                else: # range reading is inf ==> no obstacle found
                    end_pos = inverse_sensor_model(MAX_RNG, i, rbt_pos)
                    # set the last cell as free
                    occ_grid.update_at_pos(end_pos, False)
                # set all cells between current cell and last cell as free
                for idx in los.calculate(rbt_pos, end_pos):
                    occ_grid.update_at_idx(idx, False)
            
            # plan
            rbt_idx = occ_grid.pos2idx(rbt_pos)
            goal_idx = occ_grid.pos2idx(goal_pos)
            path = planner.get_path(rbt_idx, goal_idx)
            
            # show the map as a picture
            occ_grid.show_map(rbt_idx, path, goal_idx)
            
            # increment the time counter
            et = rospy.get_time() - t
            print(et <= 0.4, et)
            t += 0.4
        
        
if __name__ == '__main__':      
    try: 
        main()
    except rospy.ROSInterruptException:
        pass


