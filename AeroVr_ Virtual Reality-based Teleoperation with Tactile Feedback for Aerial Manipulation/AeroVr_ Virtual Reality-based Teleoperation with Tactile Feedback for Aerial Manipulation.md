# AeroVR: Virtual Reality-based Teleoperation with Tactile Feedback for Aerial Manipulation

Grigoriy A. Yashin, Daria Trinitatova, Ruslan T. Agishev, Roman Ibrahimov, and Dzmitry Tsetserukou

Abstract— Drone application for aerial manipulation is tested in such areas as industrial maintenance, supporting the rescuers in emergencies, and e-commerce. Most of such applications require teleoperation. The operator receives visual feedback from the camera installed on a robot arm or drone. As aerial manipulation requires delicate and precise motion of robot arm, the camera data delay, narrow field of view, and blurred image caused by drone dynamics can lead the UAV to crash. The paper focuses on the development of a novel teleoperation system for aerial manipulation using Virtual Reality (VR). The controlled system consists of UAV with a 4-DoF robotic arm and embedded sensors. VR application presents the digital twin of drone and remote environment to the user through a headmounted display (HMD). The operator controls the position of the robotic arm and gripper with VR trackers worn on the arm and tracking glove with vibrotactile feedback. Control data is translated directly from VR to the real robot in realtime. The experimental results showed a stable and robust teleoperation mediated by the VR scene. The proposed system can considerably improve the quality of aerial manipulations.

## I. INTRODUCTION

The technologies aimed at aerial manipulation by unmanned aerial vehicle (UAV) draw deep attention from companies of e-commerce (e.g., Amazon, Walmart, etc.). In many applications, such as industrial maintenance, structure inspection, it is required to carry out aerial manipulation with the objects. For example, picking and delivering the parcel, inspection of the pipelines and bridges with the sensors, holding a heavy object with the swarm of drones, etc. In such situations, it is required to equip a drone with a dexterous robotic manipulator. Nowadays, several research teams are working on the development of UAV with a robotic arm. The control system, dynamics, and stabilization of multirotor with manipulators are studied in [1], [2], [3]. In the paper of T. W. Danko and P. Y. Oh [4], the manipulator is visually servoed using an eye-in-hand camera. Flying robot which is capable of grasping objects in flight is presented in [5]. In [6], the cable-suspended aerial manipulator is developed for safe aerial manipulation in a complex environment. A. Suarez et al. [7] developed the flying robot with the variable parameter integral backstepping controller for industrial maintenance. Japanese company PRODRONE demonstrated multirotor with two manipulators, which can lift objects from the ground and land on the railing by using two manipulators [8].

![](images/86cb21323d728478a9ccc4ec197166e045b183f523a948e26e4b5891dbaf971d.jpg)  
Fig. 1. The flight test of the aerial manipulation. a) Robot during teleoperation. b) VR visualization. c) Operator with HMD and motion capturing system.

It is worth to mention that the delivery of objects is of interest not only for commercial purposes but also for lifesaving, security, posting, etc. Industrial maintenance and assistance in rescue operations in dilapidated buildings are the two most important tasks for which flying robots capable of performing aerial manipulation are of the particular interest. For this, the robot should carry out specific tasks with high accuracy that could not be performed in autonomous mode. This can be achieved with teleoperation. Additionally, in this case, the robot and the operator can be located in the remote places, which is important for hazardous conditions. According to J. J. Roldan et al. [9], the integration of the robots with virtual reality interface can be the most convenient way to remotely control the robots. A solution for inspecting industrial facilities via teleoperation is proposed in [10]. A. Suarez et al. designed a hexarotor platform equipped with a 2-DOF compliant joint arm which is controlled by a wearable exoskeleton interface via visual feedback. Despite the proposed concepts and systems, to our knowledge, the experiments of teleoperation of aerial manipulation have never been demonstrated in real conditions.

The system proposed in this paper (Fig. 1) is aimed at accurate object manipulation by the UAV equipped with a robotic arm in a remote environment when there is no direct visual contact between a human operator and drone. This goal is achieved by specially developed human-robot interaction strategy. In our case, the robotic arm attached to the quadrotor is capable of reproducing human hand movements. Such intuitive control allows an operator to manipulate remote objects successfully without any preliminary training. In addition to this, the system provides an operator with the tactile feedback, informing whether the target object is grasped. To expand human capabilities to interact with objects of interests in remote or occluded environment, the robot with manipulator as well as the objects are also simulated in virtual reality. Their positions and displacements are transmitted to the VR environment in real-time. This enhances an operator perceptional capabilities about the hardly reachable or remote environment.

## II. SYSTEM ARCHITECTURE

Our solution for the teleoperation of the 4-DoF forcesensing manipulator attached to the flying robot (Fig. 1 (a)) is a virtual reality system that consists of VR application (Fig. 1 (b)), HMD, HTC VIVE trackers (trackers) and a glove (Fig. 1 (c)). Trackers and glove transfer the position and the orientation of the operator’s hand to the VR application designed in the Unity platform, which sends target angles to the robot manipulator. The glove controls the closing of the grip and delivers feedback to the operator via vibration motors when the gripper bars interact with an object. For verification of the whole developed system, we tested the VR teleoperation of aerial manipulation using a VICON motion capture system (mocap), which provides localization data of the robot and object.

## A. Robot Design

A payload of more than 1 kg, long continuous flight time, and size are the main UAV requirements. Taking into account these limitations, we assembled a quadrotor based on the frame DJI Flame Wheel ARF KIT F450 with propulsion system DJI E600, and Cube Flight Controller (based on Pixhawk 2.1). The robot is connected to a stationary power supply via cable to eliminate the need for a changing battery during long flight tests.

![](images/3a723212e5ad67c50e91272c27cca0ff4deff6767f26bf1dce71a29fb7eb7337.jpg)  
Fig. 2. Layout of the manipulator servo motors.

4-DoF manipulator (Fig. 2) consists of three motors in articulated joints (Dynamixel servomotors MX-106T, MX-64T and AX-12 in the shoulder, elbow and wrist joints respectively), two links, one servomotor for the grip rotation (Dynamixel AX-12) and the grip with 1-DoF (Futuba S9156). The grip design is based on the four-bar linkage mechanism. The technical characteristics of the developed robot are presented in Table I.

Manipulator links have a truss structure to achieve a light and rigid construction. The links are 3D-printed from PLA material. The stress-strain analysis of links was carried out in FE Software Abaqus 6.14 to select the most optimal design in terms of minimum weight and maximum rigidity. For the finite element method, the robot link was considered as cantilever beam loaded by uniform gravity force and torque of 5.3 N·m at the opposite end (while holding an object weighing 400 grams in the extended position of the manipulator). The distributions of displacement along Z-axis and stresses in the manipulator link are provided in Fig. 3(a) and Fig. 3(b), respectively. The displacement along Z-axis is less than 3.5 mm at maximum payload, the deflection angle is less than 1 degree. Thus, the manipulator has a robust structure while manipulating an object.

TABLE I  
TECHNICAL CHARACTERISTICS OF FLYING ROBOT

<table><tr><td>Recommended load of motors DJI E600</td><td>600 g/axis</td></tr><tr><td>Weight of flying robot</td><td>2080 g</td></tr><tr><td>Manipulator weight (with electronics)</td><td>918 g</td></tr><tr><td>Maximum robot payload</td><td>400 g</td></tr><tr><td>Manipulator length</td><td>740 mm</td></tr><tr><td>Distance between the edges of the propeller and end of the grip</td><td>400 mm</td></tr><tr><td>Number of the manipulator DoF</td><td>4</td></tr></table>

![](images/2e6136dc45ed5277568006ca10871a141e455b9c7deae88a8f87696b0aaaf9f2.jpg)  
Fig. 3. The distributions of displacement along Z-axis (a) and stresses (b) in manipulator link.

## B. VR System

The VR setup includes HTC Vive Pro base stations, HMD, and Vive trackers attached to the arm for tracking the user’s hand motion in VR. The first tracker is mounted on the shoulder to control the rotation of the shoulder joint of the manipulator. The second tracker is fastened around the elbow joint and aimed to control the elbow joint rotation of the manipulator. Experimentally we defined the optimal position of trackers, which provides the largest work area of the elbow and shoulder joint angles of the operator’s arm.

## C. System of Interaction with the Target Object

The robotic arm is equipped with two force sensors FSR 400 mounted on the gripper bars. These sensors detect the contact between the gripper and the object and determine the force with which the bars of the gripper hold the cargo.

The design of the tracking glove was inspired by the glove of T.K. Chan and et al. [11]. They tested the glove equipped with inertial measurement unit (IMU) and flex sensors. Our glove for grip control consists of the fabric glove, 5 embedded flex sensors, coin vibration motors, battery, and control electronics (Fig. 4). Spectra symbol flex sensors 4.5 measure flexion angles of fingers that directly controls the opening/closing of the gripper. IMU sensor AltIMU 10 v4 determines the rotation of the hand in space (angles of the wrist joint). We sewed 5 vibration motors into the glove at the fingertip areas that allows transferring the feedback about the interaction of the manipulator with the grasped object. The use of the vibration motors on the fingertips to get feedback from the micro UAVs was proposed in the paper [12]. Control of sensors and vibration motors is performed via Arduino Nano. The data exchange between the microcontroller and computer is carried out by Bluetooth module HC-05. The assembled electrical circuit is powered by a 7.5V Li-Po battery via DC-DC converter MP1584EN.

![](images/614f549870664d7efe21997d5497eefc3445d5b3e897b47e9212b14636f2c3e3.jpg)  
Fig. 4. The scheme of the smart glove.

## III. CONTROL SYSTEM OF THE ROBOT

## A. Manipulator Kinematics

We used inverse kinematics to calculate the desired position of the gripper and the possible geometric limitations of the manipulator. For VR teleoperation mode, we applied forward kinematics. At the calculation of the kinematics, we do not take into account the rotational degree of freedom of the grip since we are transmitting the desired angle directly to the servomotor controlling this DoF. Thus, we consider the kinematics of 3-DoF planar manipulator. Fig. 5 shows the scheme of the robotic arm. The abbreviations are as follows: $\theta , \beta ,$ and α are the joint angles; $l _ { 1 }$ and $l _ { 2 }$ are the link lengths;

![](images/09902a762712858abca8432e7f28b3a67304f2a4e44db01a0f9970f5e15bf81c.jpg)  
Fig. 5. Scheme of the manipulator.

$l _ { 3 }$ is the position of the grip end in relation to the wrist joint; $( x _ { i } , z _ { i } )$ coordinate system (CoS) is attached to i-th CoS transformation; $l _ { d i s }$ is the transition from CoS $( x _ { 2 } , z _ { 2 } )$ to CoS $( x _ { 3 } , z _ { 3 } )$ due to the shape of link $1 ; T _ { 1 } , T _ { 2 }$ , and $T _ { 3 }$ are torques in shoulder, elbow and wrist joints, respectively.

1) Inverse Kinematics: For the 3-DoF planar manipulator, we used the following transition matrix, which corresponds to the transformation from the CoS $( x _ { 5 } , z _ { 5 } )$ to drone coordinates $( x _ { 0 } , z _ { 0 } )$

$$
{ } _ { p } ^ { B } T = T _ { 5 } ^ { 0 } = \left[ \begin{array} { c c c c } \cos ( \phi ) & - \sin ( \phi ) & 0 & x \\ \sin ( \phi ) & \cos ( \phi ) & 0 & z \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{array} \right]\tag{1}
$$

$$
\text { where } \left\{ \begin{array}{c} \phi = \alpha - \beta + \theta \\ x = l _ {1} \cdot \cos (\theta) + l _ {2} \cdot \cos (\beta - \theta) - l _ {d i s} \cdot \sin (\theta) \\ z = l _ {1} \cdot \sin (\theta) - l _ {2} \cdot \sin (\beta - \theta) + l _ {d i s} \cdot \cos (\theta) \end{array} \right.
$$

Since the system is underconstrained, we decided to impose an additional limitation for α to keep the object parallel to the ground. This condition is supposed to provide the stability of the grabbed object. Firstly, we calculate $\beta$ from (1) using MATLAB function solve for symbolic computation. Secondly, using systems of equations (1) and (2) [13] we calculate θ and α:

$$
\left\{ \begin{array}{c} x = k _ {1} \cdot \cos (\theta) - k _ {2} \cdot \sin (\theta) \\ z = k _ {1} \cdot \sin (\theta) + k _ {2} \cdot \cos (\theta) \\ k _ {1} = l _ {2} \cdot \cos (\beta) + l _ {1} \\ k _ {2} = - l _ {2} \cdot \sin (\beta) + l _ {d} \end{array} \right.\tag{2}
$$

$$
\left\{ \begin{array}{c} \theta = - (\arctan (x, y) - \arctan (k _ {1}, k _ {2})) \\ \alpha = \beta - \theta \end{array} \right.\tag{3}
$$

2) Forward Kinematics: We used forward kinematics to display the position of manipulator components in Unity. To do this, it is necessary to write the transformation matrix for each joint of the manipulator via rotation and translation matrices from the drone $\mathrm { C o S } \left( x _ { 0 } , z _ { 0 } \right)$ to each joint CoS. Thus, we got the following equations for position of elbow, wrist joints (4, 5) and the grip end (6):

$$
\begin{array}{l} {x _ {e} = x _ {s} + l _ {1} \cdot \cos (\theta) - l _ {d i s} \cdot \sin (\theta)} \\ {z _ {e} = z _ {s} + l _ {1} \cdot \sin (\theta) + l _ {d i s} \cdot \cos (\theta)} \end{array}\tag{4}
$$

$$
\begin{array}{r} x _ {w} = x _ {e} + l _ {2} \cdot \cos (\beta - \theta) \\ z _ {w} = z _ {e} - l _ {2} \cdot \sin (\beta - \theta) \end{array}\tag{5}
$$

$$
\begin{array}{c} x _ {g} = x _ {w} + l _ {2} \cdot \cos (\alpha - \beta + \theta) \\ z _ {g} = z _ {w} + l _ {2} \cdot \sin (\alpha - \beta + \theta) \end{array}\tag{6}
$$

where $x _ { s } , \ z _ { s }$ are the coordinates of the shoulder joint that corresponds to the current position of the UAV.

![](images/d11048d744c9894ce13528502f0a4cc809ed029e265f6510d149cf47c5c93590.jpg)  
Fig. 6. GUI to monitor the manipulator state.

## B. GUI to Monitor the Manipulator State

The connection of motors and sensors to MATLAB is performed via board OpenCM 9.04. Wireless control of the robotic arm is carried out by Bluetooth module HC-05. Debugging of the control algorithm and monitoring the state of the robot is realized by the developed GUI interface in MATLAB (see Fig. 6). Using the GUI, we can conveniently configure the robot and observe the real-time manipulator behavior using a computer. In addition, we can send a command to turn on (off) the manipulator power supply using the transistor switch. It is necessary if the behavior of UAV will become unstable.

Depending on the manipulator components, GUI provides information about the work area of the manipulator and the required torque of servomotors. It also displays information about the current technical parameters of servomotors (voltage, speed, PID coefficients) and sensor data. The user can change the following manipulator settings:

• Control modes of motors depending on the controllers: Arduino UNO, USB2Dynamixel or OpenCM9.04.

• Work area via changing the acceptable value of joint angles.

• Lengths of manipulator links.

• Types of the manipulator motors (ordinary servomotors or Dynamixel smart servomotors).

• Teleoperation and manual manipulator control modes.

In manual mode, the operator can change the position of the gripper by writing desired X and Y coordinates or pushing on the arrow keys (forward, back, up or down). The step size is set separately for the usage of the arrow keys. Teleoperation mode switches the GUI to the mode of displaying and recording the experimental data, while control of the manipulator is transferred to Unity and the operator.

## C. The UAV Localization

The external position estimation system is used to localize the UAV during the flight. In order to get the high-quality tracking of the quadrotor and a target object during the experiments, we use mocap with 12 cameras (Vantage V5) covering 5 m x 5 m x 5 m space. Robot Operating System (ROS) Kinetic framework is used to run the development software and mavros ROS stack for drone control and ground station communication. The position and attitude update rate for the quadrotor and the target object is 100 Hz. The estimated poses of the tracked robot and object are sent to the computer running Unity 3D software for VR simulation. While the main goal of the experiment is to manipulate an object in virtual and real environments at the same time, the quadrotor is commanded to give in place holding its position close to the object (in this position, the object is located within the robotic arm’s manipulation area). During the aerial manipulation, the robot position is corrected by a UAV operator in order to prevent high oscillations around the desired location of the robot.

## IV. VR CONTROL SYSTEM

Fig. 7 shows the scheme of communication among mocap, ROS, Unity, the flying robot, and the operator. Unity application is the core part of this system, which receives the data from the operator and the robot through a Bluetooth connection and from ROS through a cable. Also, Unity sends the necessary commands to the devices in accordance with predefined scripts. To provide the user with a visual control, the operator wears HMD to experience VR application, which includes a simplified model of the room along with a flying robot. Mocap cameras transfer positioning and rotation tracking data of each object to the Unity 3D engine. The connection between Mocap and VR application is based on the method proposed in [14]. In addition, data from the manipulator are transmitted to the GUI.

![](images/4b0832316da1876791d20ffe3a1b9dabb23cf64a67f84a79501ff05c5d1e3d27.jpg)  
Fig. 7. The scheme of communication of the VR-based teleoperation.

The work space for testing the teleoperation system consists of two rooms (Fig. 8). The first room, where the aerial manipulation is carried out, is equipped with mocap cameras. In addition, the UAV operator with a radio remote control is present in this room to eliminate unforeseen high oscillations around the target position. In the second room, we installed HTC VIVE equipment and base station (PC with Unity). The operator can freely move his or her hand on which the trackers and glove are worn. Transfer data about the relative position of the robot and the object in VR from mocap allows the operator to accurately navigate robotic arm in the current situation which corresponds to the real context in the room where the flight experiment is carried out.

![](images/4402d7372a4018d2c46ddddce348bb9558ed508da5f05200540f85de4b615608.jpg)  
Fig. 8. Scheme of the rooms for VR-based teleoperation experiment.

## V. EXPERIMENTS

Experiments have been conducted in two stages: the flight test with GUI manual teleoperation and VR-based teleoperation in stationary and flight modes.

## A. Teleoperation Tests using GUI

In the first trials, we tested the robot operation in flight using a GUI control of the manipulator. We moved the manipulator along the specified path (Fig. 9, Fig. 10(a))

![](images/5f344c7dd2c6f9dcd32bffb2f3e4220c6f6de9aea98ecb512bd996a3a1b79a75.jpg)  
Fig. 9. The trajectory of the wrist joint position during experiment.

which includes a command to drop the object in point No. 9. A wooden cube weighing 77 grams was used as an object for this experiment.

Fig. 10(c) shows that the maximum deviation of the roll angle is of 5.22 degrees (standard deviation is of 0.99 degrees). The fluctuations in the roll direction in the range of points 3 to 9 arise due to the fast manipulator movement with a considerable distance. The standard deviation of pitch angle is of 3.51 degrees, the maximum pitch error of 9.38 degrees (see Fig. 10(d)) occurs when the manipulator is moved along one of the coordinates about 0.2 m, which is accompanied by a significant change of the servomotor torque in the elbow joint.

![](images/f3adf6dcea3d5824089f877e8c23eaed4fc4570b9b4da724265c796797a1d86b.jpg)  
Fig. 10. Dependences of coordinates (a) and motor torques (b) of the manipulator, roll (c) and pitch (d) angles from the experiment time for the flight test with the GUI teleoperation.

## B. VR Teleoperation Tests

For the VR teleoperation experiment we chose a plastic cylinder weighing 105 grams as the object to be grasped. During stationary testing of the teleoperation system, we defined that the position of the trackers on the shoulder and forearm influences the possibility of trackers’ circular offset on the hand, which can decrease the work area of θ and β. After that, we calibrated delays of control codes for the manipulator, glove, and Unity. Long delay or non-synchronous repetition of the movement of the operator’s hand by the manipulator can cause unsuccessful aerial manipulation with the object.

Fig. 11 shows the change of the position of the wrist joint, the torques of the servomotors of the manipulator, roll, and pitch angles of the UAV in time. The recorded time interval is shown between the moment when the robot hangs over the target position and the time when it is stabilized after releasing the object. Furthermore, the time-stamps of the key operations (contact with an object, object grasping, raising and releasing) are marked on this figure. The largest pitch deviation is 7.15 degrees (at 11.2 sec, Fig. 11(d)) that corresponds to the reliable contact of the manipulator with the object. Grasping the object and picking it after 2 seconds caused a significant fluctuation of the UAV attitude that affected on the roll angle (maximum standard roll deviation is of 2.61 degrees). After the object grasping, the drone was confined in movement in the direction along Y-axis. That is why its roll angle during this period (8 - 20 sec) deviated from the desired value, i.e. its set-point. The movement of the object in new position also increased the torque value in the elbow joint. The object release caused a sharp decrease of the robot mass (in comparison to the robot with the object) and the robot CoM moved quickly. The drone controller handled this situation, and the robot stabilized itself after 4 seconds. During aerial manipulation, standard deviations in pitch and roll directions were 2.03 and 0.83 degrees, respectively.

![](images/f35b8c754f245d36ba05a170712c43ed6d3b6ea198d9a7ad8d64120424d9e1eb.jpg)

![](images/a007cdfe31a271545139b6b96ef5c15e4ec19f6fcc1224a6048c55ccde05f95b.jpg)

![](images/0869f2c8c2f24bf9bf8a78646e3e63e90ab2322569ccd6cf245029fec9a1de34.jpg)

![](images/3349bacdb6e7de58eebd3bc68042fe50896a9da9f06c9a80dec62a4199e3089b.jpg)  
Fig. 11. Coordinates (a) and motor torques (b) of the manipulator, roll (c) and pitch (d) angles vs. time for the flight test with the VR-based teleoperation.

To conclude, the teleoperation of aerial manipulation was successful. Tactile feedback from vibration motors of the glove effectively complements the visual information of the virtual environment, being a tangible confirmation of contact with the object. The main difficulties of the teleoperation were related to the data transfer from Unity to the manipulator controller. There were the cases when it manifested itself in delays in the execution of real movements of the manipulator. Throughout the experiments, a live telephone connection was used between the operator and the technical operator of UAV. It was necessary to perform the adjustment of the position of the robot in the perpendicular direction to the manipulation. In the future, it will not be necessary since we plan to add the function of the UAV control to the VR application.

## VI. CONCLUSION

We have proposed a VR-based teleoperation system for the UAV robotic manipulator. The flight tests of the developed robot showed a stable UAV behavior while grasping of the target object. The combined use of HTC VIVE trackers and designed glove demonstrated a robust data transmission to the Unity application. The average deviations in pitch and roll directions were 2.03 and 0.83 degrees, respectively. The future work will be devoted to the development of the controller for decreasing deviations of the flying robot during the interaction with different objects and accurate robot position control in VR.

## REFERENCES

[1] C. Korpela, M. Orsag, M. Pekala, and P. Oh, “Dynamic stability of a mobile manipulating unmanned aerial vehicle,” in 2013 IEEE International Conference on Robotics and Automation. IEEE, 2013, pp. 4922–4927.

[2] A. Jimenez-Cano, J. Martin, G. Heredia, A. Ollero, and R. Cano, “Control of an aerial robot with multi-link arm for assembly tasks,” in 2013 IEEE International Conference on Robotics and Automation. IEEE, 2013, pp. 4916–4921.

[3] S. Kim, S. Choi, and H. J. Kim, “Aerial manipulation using a quadrotor with a two dof robotic arm,” in 2013 IEEE/RSJ International Conference on Intelligent Robots and Systems. IEEE, 2013, pp. 4990– 4995.

[4] T. W. Danko and P. Y. Oh, “Toward coordinated manipulator-host visual servoing for mobile manipulating uavs,” in 2014 IEEE International Conference on Technologies for Practical Robot Applications (TePRA). IEEE, 2014, pp. 1–6.

[5] J. Thomas, G. Loianno, J. Polin, K. Sreenath, and V. Kumar, “Toward autonomous avian-inspired grasping for micro aerial vehicles,” Bioinspiration & biomimetics, vol. 9, no. 2, p. 025010, 2014.

[6] Y. S. Sarkisov, M. J. Kim, D. Bicego, D. Tsetserukou, C. Ott, A. Franchi, and K. Kondak, “Development of sam: cable-suspended aerial manipulator,” arXiv preprint arXiv:1903.02426, 2019.

[7] A. Suarez, A. Jimenez-Cano, V. Vega, G. Heredia, A. Rodr´ıguez-Castano, and A. Ollero, “Lightweight and human-size dual arm aerial˜ manipulator,” in 2017 International Conference on Unmanned Aircraft Systems (ICUAS). IEEE, 2017, pp. 1778–1784.

[8] “Prodrone unveils the world’s first dual robot arm large-format drone,” in https://www.prodrone.jp/en/archives/1420/, Retrieved July 18, 2019.

[9] J. J. Roldan, E. Pe ´ na-Tapia, D. Garz ˜ on-Ramos, J. de Le ´ on, M. Garz ´ on,´ J. del Cerro, and A. Barrientos, “Multi-robot systems, virtual reality and ros: Developing a new generation of operator interfaces,” in Robot Operating System (ROS). Springer, 2019, pp. 29–64.

[10] A. Suarez, P. Sanchez-Cuevas, M. Fernandez, M. Perez, G. Heredia,´ and A. Ollero, “Lightweight and compliant long reach aerial manipulator for inspection operations,” in 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2018, pp. 6746–6752.

[11] T. K. Chan, Y. K. Yu, H. C. Kam, and K. H. Wong, “Robust hand gesture input using computer vision, inertial measurement unit (imu) and flex sensors,” in 2018 IEEE International Conference on Mechatronics, Robotics and Automation (ICMRA). IEEE, 2018, pp. 95–99.

[12] L. Labazanova, A. Tleugazy, E. Tsykunov, and D. Tsetserukou, “Swarmglove: A wearable tactile device for navigation of swarm of drones in vr environment,” in International AsiaHaptics conference. Springer, 2018, pp. 304–309.

[13] J. J. Craig, Introduction to robotics: mechanics and control, 3/E. Pearson Education India, 2009.

[14] R. Ibrahimov, E. Tsykunov, V. Shirokun, A. Somov, and D. Tsetserukou, “Dronepick: Object picking and delivery teleoperation with the drone controlled by a wearable tactile display,” in RO-MAN 2019-The 28th IEEE International Symposium on Robot and Human Interactive Communication. IEEE, 2019.