source ~/ws/devel/setup.bash
export X_POS="-2.0"
export Y_POS="-0.5"
export WORLD="turtlebot3_world"
chmod +x ~/ws/src/pkg/scripts/*.py
roslaunch pkg world.launch
