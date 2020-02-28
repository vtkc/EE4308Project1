source ~/ws/devel/setup.bash
# export TURTLEBOT3_MODEL="burger" # already set
export GOALS="0.5,0.5|1.5,0.5|-1.5,-1.5|-2.0,-0.5"
export X_POS="-2.0"
export Y_POS="-0.5"
export WORLD="turtlebot3_world"
chmod +x ~/ws/src/pkg/scripts/*.py
roslaunch pkg lab3A.launch
