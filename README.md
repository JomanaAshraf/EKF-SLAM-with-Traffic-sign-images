# Hands On Localization Project in Simulation



## Members of the group

This project has been carried out by:

* Alaaeddine El Masri El Chaarani
* Jomana Ashraf
* Rahaf Abu-Hara

___
## How to use it
First open the gazebo environment with the robot and rviz by running this command:
* roslaunch localization_project localization.launch

In order to run the perception node to detect the traffic signs:
* rosrun localization_project vision.py

If you need to test with ArUco markers use command:
* rosrun localization_project perecption.py

Now, to run the EKF-SLAM with data association:
* rosrun localization_project localization_node_data_association.py

If you need to run the EKF-SLAM without data association "recommended if using the ArUco markers":
* rosrun localization_project localization_node.py


