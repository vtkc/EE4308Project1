#!/usr/bin/env python

import roslib, rospy, rospkg
from numpy import *
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState, Imu
from std_msgs.msg import Float64MultiArray
from tf.transformations import quaternion_from_euler, euler_from_quaternion, quaternion_multiply
from tf2_msgs.msg import TFMessage
import cv2
import numpy
import lab3A_aux

# ================================= CONSTANTS ==========================================        
# let's cache the SIN and POS so we don't keep recalculating it, which is slow
DEG2RAD = [i/180.0*pi for i in xrange(360)] # DEG2RAD[3] means 3 degrees in radians
SIN = [sin(DEG2RAD[i]) for i in xrange(360)] # SIN[32] means sin(32degrees)
COS = [cos(DEG2RAD[i]) for i in xrange(360)]
SQRT2 = sqrt(2)
TWOPI = 2*pi
PI = pi # numpy pi

# ================================== DATA STRUCTS ===========================================
buf = None
# ================================== SUBSCRIBERS ======================================
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
        
def subscribe_wheels(msg):
    global rbt_wheels
    rbt_wheels = (msg.position[1], msg.position[0])

def subscribe_imu(msg):
    global rbt_imu_o, rbt_imu_w, rbt_imu_a
    t = msg.orientation
    rbt_imu_o = euler_from_quaternion([\
        t.x,\
        t.y,\
        t.z,\
        t.w\
        ])[2]
    rbt_imu_w = msg.angular_velocity.z
    rbt_imu_a = msg.linear_acceleration.x
    
def subscribe_main(msg):
    global msg_main
    msg_main[0] = msg.data[0] # operation state
    msg_main[1] = msg.data[1] # px
    msg_main[2] = msg.data[2] # py

# ================================ BEGIN ===========================================
def move():
    # ---------------------------------- INITS ----------------------------------------------
    # init node
    rospy.init_node('move')
    
    # Set the labels below to refer to the global namespace (i.e., global variables)
    # global is required for writing to global variables. For reading, it is not necessary
    global msg_main, rbt_imu_w, rbt_true, rbt_wheels
    
    # Initialise global vars with NaN values 
    # nan and inf are imported from numpy. If you use "import numpy as np", then nan is np.nan, and inf is np.inf.
    msg_main = [-1. for i in xrange(3)]
    rbt_true = None
    rbt_wheels = None
    rbt_imu_w = None

    # Subscribers
    rospy.Subscriber('main', Float64MultiArray, subscribe_main, queue_size=1)
    rospy.Subscriber('tf', TFMessage, subscribe_true, queue_size=1)
    rospy.Subscriber('joint_states', JointState, subscribe_wheels, queue_size=1)
    rospy.Subscriber('imu', Imu, subscribe_imu, queue_size=1)
    
    # Publishers
    publisher_u = rospy.Publisher('cmd_vel', Twist, latch=True, queue_size=1)
    publisher_move = rospy.Publisher('move', Float64MultiArray, latch=True, queue_size=1)
    # set up cmd_vel message
    u = Twist()
    # cache for faster access
    uv = u.linear #.x
    uw = u.angular #.z
    prev_v = 0
    prev_w = 0
    # set up move message
    msg_move = [0. for i in xrange(4)]
    msg_m = Float64MultiArray()
    msg_m.data = msg_move
    # publish first data for main node to register
    publisher_move.publish(msg_m)
    
    # Wait for Subscribers to receive data.
    print('[INFO] Waiting for topics... imu topic may not be broadcasting if program keeps waiting')
    while (msg_main[0] == -1. or rbt_imu_w is None or rbt_true is None \
        or rbt_wheels is None) and not rospy.is_shutdown():
        pass
        
    if rospy.is_shutdown():
        return
        
    # Data structures
    # ~ motion model this fuses imu and wheel sensors using a simple weighted average
    # ~ motion model also always returns an orientation that is >-PI and <PI
    # ~ notice it is intialised with true starting position and rbt_wheels
    motion_model = lab3A_aux.OdometryMM(rbt_true, rbt_wheels, 0.16, 0.066)
    
    
    # ---------------------------------- BEGIN ----------------------------------------------
    t = rospy.get_time()
    while (not rospy.is_shutdown()): # required to Keyboard interrupt nicely
        if (rospy.get_time() > t): # every 50 ms
            # ~ get main message
            # ~ break if main signals to stop
            if msg_main[0] == 1.:
                break
            
            # ~ retrieve pose (rx, ry, ro) from motion_model
            # ~ methods no longer returns tuples, but stored in object for slightly faster access
            motion_model.calculate(rbt_wheels, rbt_imu_w, rbt_imu_o, rbt_imu_a);
            rx = motion_model.x; ry = motion_model.y; ro = motion_model.o
            # ~ publish pose to move topic
            msg_move[1] = rx; msg_move[2] = ry; msg_move[3] = ro;
            publisher_move.publish(msg_m)
            
            # calculate target coordinate to pursue
            px = msg_main[1]; py = msg_main[2]
            dx = px - rx; dy = py - ry
            # calculate positional error
            err_pos = sqrt(dx*dx + dy*dy)
            # calculate angular error
            err_ang = arctan2(dy, dx) - ro
            if err_ang >= PI:
                err_ang -= TWOPI
            elif err_ang < -PI:
                err_ang += TWOPI
            
            v, w = lab3A_aux.get_v_w(err_pos, err_ang)
            
            uv.x = v
            uw.z = w
            publisher_u.publish(u)
            
            # increment the time counter
            et = rospy.get_time() - t
            print('[INFO] MOVE ({}, {:.3f}) v({:.3f}), w({:.3f})'.format(et <= 0.05, et, v, w))
            t += 0.05
    
    
    t += 0.3
    uv.x = 0; uw.y = 0;
    publisher_u.publish(u)
    while not rospy.is_shutdown() and rospy.get_time() < t:
        pass
        
    print('[INFO] MOVE stopped')
    
    
if __name__ == '__main__':      
    try: 
        move()
    except rospy.ROSInterruptException:
        pass


