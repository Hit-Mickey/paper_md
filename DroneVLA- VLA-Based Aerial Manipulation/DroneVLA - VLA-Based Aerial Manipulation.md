# DroneVLA: VLA based Aerial Manipulation

Fawad Mehboob Fawad.Mehboob@skoltech.ru Skoltech Moscow, Russia

Jefrin Sam Jefrin.Sam@skoltech.ru Skoltech Moscow, Russia

Monijesu James<sup>∗</sup> Monijesu.James@skoltech.ru Skoltech Moscow, Russia

Miguel Altamirano Cabrera m.altamirano@skoltech.ru Skoltech Moscow, Russia

Amir Habel<sup>∗</sup> Amir.Habel@skoltech.ru Skoltech Moscow, Russia

Dzmitry Tsetserukou d.tsetserukou@skoltech.ru Skoltech Moscow, Russia

## Abstract

As aerial platforms evolve from passive observers to active manip ulators, the challenge shifts toward designing intuitive interfaces that allow non-expert users to command these systems naturally. This work introduces a novel concept of autonomous aerial manip ulation system capable of interpreting high-level natural language commands to retrieve objects and deliver them to a human user. The system is intended to integrate a MediaPipe based on Grounding DINO and a Vision-Language-Action (VLA) model with a custom built drone equipped with a 1-DOF gripper and an Intel RealSense RGB-D camera. VLA performs semantic reasoning to interpret the intent of a user prompt and generates a prioritized task queue for grasping of relevant objects in the scene. Grounding DINO and dynamic A\* planning algorithm are used to navigate and safely relocate the object. To ensure safe and natural interaction during the handover phase, the system employs a human-centric controller driven by MediaPipe. This module provides real-time human pose and orientation estimation, allowing the drone to employ visual servoing to maintain a stable, distinct position directly in front of the user, facilitating a comfortable handover. We demonstrate the system’s eficacy through real-world experiments for localiza tion and navigation, highlighting the feasibility of VLA for aerial manipulation operations.

## CCS Concepts

• Human-centered computing → Human–robot interaction; • Computing methodologies → Computer vision; Artificial in telligence; • Computer systems organization → Embedded and cyber-physical systems.

## Keywords

Aerial Manipulation, Vision-Language-Action Models, Human-Robot Interaction, Visual Surveying, Robotic Fetch-and-Carry

## 1 Introduction

Unmanned Aerial Vehicles (UAVs) have evolved to be used in an increasing number of applications, ranging from surveillance and inspection to agriculture, search and rescue missions, and much more. More recently, significant interest has developed in indoor UAV operations, such as inspection in industrial environments and warehouses. This has paved a way for a a growing interest in inte grating robotic grippers with the UAVs enabling them to manipulate objects. As a result, an emerging field of aerial manipulation has captured the interest of UAV researchers, transforming UAVs into Unmanned Aerial Manipulators (UAMs)

![](<images/a503c2fff9530661c5ee427d3e5a7f913a30cc8467638d1be44dcf3daea9b2cb.jpg>)  
Figure 1: Human Localization and Navigation by drone

Combining the high motion agility of UAVs with the manipulability of robotic arms is an interesting yet challenging field of research. A very useful scenario of indoor UAV use is the ability to open doors; this problem has been studied in [12] by proposing a novel force exertion scheme. Similar force exertion scenarios are studied for valve-turning operations using thrust vectoring in [14]. Moreover, an interesting operation of UAMs is demonstrated in aerial calligraphy, as proposed in [1]. In addition to this, the most commonly studied use case is the transportation of objects using UAMs [2], which also involves dealing with external forces, changes in vehicle inertia, and stability issues. A comprehensive review of UAMs discussing their utility and realization challenges is provided in [2] and [8]. More recently, the focus of aerial manipulation has shifted to interaction and collaboration with humans. To this end, a master-slave tele-operation framework has been demonstrated on an aerial manipulator in [4].

Additionally, the idea of incorporating Large Language Models (LLMs) to take prompts from humans and perform tasks in aerial manipulation seems promising. However, the reliability of such a framework is a critical challenge, especially in environments where drones interact with humans and the safety of both the human and the environment is of high concern. To this end a Vision-Language

![](<images/0fd86ba2af6e68ab62b28c10968ba9bf9bedc8c6aeb749246545ac4839848539.jpg>)  
Figure 2: Overview of the VLA based aerial manipulation and Path Planning Architecture

Model (VLM)-based mission was proposed in [5], with considerable eforts to make prompt-based aerial manipulation reliable by mitigating hallucinations using structured prompting and a predefined skill library. However, this approach lacks authority over drone control, which is critical in such scenarios given the highly dynamic nature of drone flights. Quadrotor flights are prone to high disturbances, especially close to the ground and near physical objects due to downwash produced by the propellers while hover ing [6]. To address this issue, a more robust framework is required that accounts for high-level drone control using VLA similar to as demonstrated for UAV flights in [9, 10]. Therefore, in this study, we propose a conceptual Vision-Language-Action (VLA) model for aerial manipulation of objects in a human-robot interaction envi ronment. Our VLA is implemented on a quadrotor UAV equipped with a simple gripping mechanism, designed to pick up and place objects from locations such as tables and shelves. To the best of our knowledge, this is the first proposal of VLA for aerial manipulation tasks.

## 2 System Architecture

The system comprises four major subsystems: (1) hardware platform, (2) VLA-based semantic reasoning module, (3) perception pipeline with open-vocabulary detection, and (4) human-centric handover controller. Fig. 2 illustrates the overall system architecture.

## 2.1 Hardware Platform

The hardware consists of a simple rigid-body open-close gripping mechanism powered by a Dynamixel AX-12A motor mounted hor izontally at the front of a quadrotor UAV. Considerable care has been taken in positioning the gripper away from the drone body to avoid interference while picking up objects, this helps mitigate the downwash which efects flight stability and also cause a problem of blowing the objects away from the drone while grasping. However, a trade-of must be made between the stability of the drone and the distance of the gripper from the drone’s center of mass (COM).

All modules communicate via ROS2 topics using standard message types. The distributed architecture enables computationally intensive perception and reasoning tasks to run ofboard while maintaining low-latency control loops onboard the OrangePi Fig. 2.

## 2.2 Vision-Language-Action (VLA) Module

To enable intuitive control on resource-constrained hardware, we employ a lightweight Vision-Language-Action (VLA) model inspired by architectures such as TinyVLA [13], which aims to bring multi-modal capabilities to embedded hardware. Here we propose a conceptual VLA system validated in Unity Simulation for detecting objects and grasping using a gripper.

The proposed VLA module accepts two primary inputs:

• Visual Observation (�<sub>�</sub>): A time-indexed RGB image stream captured from the onboard RealSense camera. Prior to fusion, the visual stream is down-sampled to 224 × 224 resolution and encoded using a compact convolutional visual backbone.

• Language Prompt (�): A natural language instruction from the operator specifying an object and intended action (e.g., ’pick up the red screwdriver’). In the current prototype, text is supplied via keyboard input.

The VLA architecture is deployed on the ground station rather than onboard the UAV, enabling increased model complexity without incurring latency or power constraints on the aerial platform. The TinyVLA is composed of a 6-layer ViT-Tiny visual encoder (patch size 16×16, 192-dim latent) and a 4-layer text transformer (256-dim hidden, 8 attention heads) for natural language prompt processing. Visual frames streamed from the RealSense camera over Wi-Fi are downsampled to 224×224 resolution and processed asynchronously at 15–20 Hz. The fused latent representation is passed through a 2-layer MLP action policy head that predicts discrete gripper actions Open,Close. The resulting model executes with sub-5 ms inference on a desktop-class RTX GPU. The ground station deployment also allows future expansion of the action vocabulary, multi-object grounding, and afordance reasoning without modifying the UAV flight stack.

Given the single-degree-of-freedom parallel gripper mounted beneath the quadrotor, the output action space is defined as a binary actuator command:

$$
a _ {t} = \pi (I _ {t}, L) \in \text { Open }, \text { Close }\tag{1}
$$

As evident from the action-space, the VLA implements a simple opening and closing action upon detection ofbeing in the vicinity of the object of interest. Meanwhile, spatial positioning and approach trajectory are handled by a human-centric controller driven by MediaPipe as illustrated in Fig. 2 and explained in the subsequent sections. This architecture decouples semantic manipulation reasoning from UAV stabilization and navigation, and serves as a first step that can be extended to a more general end-to-end architecture to apply a full 5-DOF control, that includes velocities in the x,y,z coordinates,yaw angular rate, and gripper state.

This direct mapping from perception to actuation allows the VLA to focus exclusively on grasping logic. The VLA continuously infers the correct gripper state based on the visual proximity to the object described in the text prompt, while the drone’s position controller handles the approach trajectory independently.

## 2.3 Perception Pipeline: Open-Vocabulary Detection and 3D Localization

Traditional object detection systems typically require retraining for new object categories. To enable true open-vocabulary manip ulation, we integrate Grounding DINO [3], a transformer-based open-set object detector that performs cross-modal fusion of vision and language features. Grounding DINO takes as input: RGB im ages from RealSense cameras, and text prompts specifying target objects(e.g., "red cup", "person in red shirt", "screwdriver").

The detector outputs bounding boxes with confidence scores for all instances that match the text query. This enables zero-shot detection of arbitrary objects without task-specific training.

3D Localization: For each bounding boxes detected by Ground ing DINO, we compute the centroid of bounding box in image coordinates (�, �), query the aligned depth map at (�, �) to obtain depth �, deproject to 3D camera coordinates and transform from camera frame to world frame using ROS2 TF2 transforms.

This pipeline provides real-time 3D positions of the target objects and humans in the global reference frame, enabling subsequent visual servoing control.

## 2.4 Human-Centric Handover Controller

Safe and natural human-robot handover requires understanding the human’s pose, orientation, and preferred interaction geometry. We employ MediaPipe Pose Landmarker to estimate human pose in real-time.

MediaPipe outputs: 33 skeletal keypoints in both normalized 2D image coordinates and 3D world coordinates, body orientation computed from shoulder and hip joint vectors, and Hand position estimate to determine handover side preference as shown in Fig. 3 .

![](<images/dca15b494da1a7f4b13ef5ffba48ba03370632b6266a2d072b599bd2aeb6a942.jpg>)  
Figure 3: Grounding DINO bounding box object detection and humans in RGB images from drone cameras

Human Orientation Estimation: Let $\mathrm { \bf p } _ { \mathrm { l e f t } }$ <sub>\_shoulder</sub> and p<sub>right\_shoulder</sub> denote the 3D positions of the left and right shoulder landmarks, the arc-tangent of the diference of these landmarks is used to compute the torso forward direction (yaw angle):

$$
\mathbf {v} _ {\text { torso }} = \mathbf {p} _ {\text { right\_shoulder }} - \mathbf {p} _ {\text { left\_shoulder }},\tag{2}
$$

$$
\theta_ {\mathrm{yaw}} = \arctan (\mathbf {v} _ {\mathrm{torso}, y}, \mathbf {v} _ {\mathrm{torso}, x})\tag{3}
$$

The perpendicular direction to the torso indicates the person’s facing direction.

Handover Pose Computation: Based on HRI handover litera ture [7, 11], we compute a target handover pose in form of Distance (approximately 0.6-0.8, in from of the human), Height (Chest level at approximately 1.0-1.3m from ground), and orientation (Drone faces human directly, aligned with human’s facing direction)

The target handover pose in the world frame is:

$$
{ } ^ { W } \mathbf { p } _ { \text {   h   a   n   d   o   v   e   r   } } = { } ^ { W } \mathbf { p } _ { \text {   h   u   m   a   n   } } + d _ { \text {   h   a   n   d   o   v   e   r   } } \cdot \left[ \begin{array} { c } \cos ( \theta _ { \text {   y   a   w   } } + \pi ) \\ \sin ( \theta _ { \text {   y   a   w   } } + \pi ) \\ 0 \end{array} \right] + \left[ \begin{array} { c } 0 \\ 0 \\ h _ { \text {   c   h   e   s   t   } } \end{array} \right] ,\tag{4}
$$

where $d _ { \mathrm { h a n d o v e r } }$ is the desired standof distance and $h _ { \mathrm { c h e s t } }$ is the vertical ofset to chest height.

Visual Servoing Control: We implement a hybrid servoing approach combining position-based visual servoing (PBVS) and image-based visual servoing (IBVS).

The PBVS is used for coarse navigation to the object and handover locations using 3D pose estimates from depth cameras.

While the IBVS is used for fine alignment during grasping and handover, directly using 2D image features to minimize drift and maintain object/human visibility.

The image-based error $\mathbf { e } _ { \mathrm { i m g } } = \mathbf { s } - \mathbf { s } ^ { \ast }$ (current vs. desired feature positions) drives the control.

Safety checks and failure recovery behaviors $( \mathrm { e . g . }$ , lost object detection, communication timeout) are integrated at each state transition.

## 2.5 Motion planning with human awareness

The global motion planner operates in the horizontal plane using a grid–based $\mathsf { A } ^ { \star }$ search. The room $[ x _ { \mathrm { m i n } } , x _ { \mathrm { m a x } } ] \times [ y _ { \mathrm { m i n } } , y _ { \mathrm { m a x } } ]$ is discretized into a uniform occupancy grid with resolution $\Delta ,$ where each cell represents a candidate drone position and the occupancy value occ(�, $j ) \in \{ 0 , 1 \}$ encodes free space or collision. Circular obstacles are rasterized by marking all cells whose centers lie within the corresponding radius; in particular, the human is modeled as a cylindrical obstacle with radius equal to the physical body radius plus an additional safety margin, enforcing a minimum stand-of distance throughout the mission. For each leg of the task (home–object, object–human, human–home), continuous start and goal positions are projected to the nearest free cells and $\mathsf { A } ^ { \star }$ is run on the 8-connected grid with unit cost for axial moves and $\sqrt { 2 }$ for diagonal moves, and a Euclidean heuristic in grid index space. The resulting sequence of cells is converted back to world coordinates and passed through a line-of-sight smoothing step, which removes redundant waypoints while preserving collision-free straight segments. The final polyline is then sent to the low-level controller as a sequence of position setpoints with altitude specified by the mission logic.

## 3 System Evaluation

## 3.1 Experimental Procedure

Setup. Experiments were conducted in an indoor laboratory space ofapproximately 6 m×6 m equipped with an overhead Vicon motion capture system, which provided ground–truth pose for the drone, the table, and the human participant. A rectangular table was placed in the center of the workspace and populated with 15–20 everyday objects (cups, tools, plants etc) arranged in diferent configurations. A human stood at a designated handover location approximately 2 m from the table. The onboard perception stack (Grounding DINO, depth sensing, and MediaPipe) and the human–aware $\mathsf { A } ^ { \star }$ motion planner were evaluated jointly as an integrated pipeline for scene surveying, object and human localization, and safe navigation.

Trials. We conducted � = 10 trials with randomized object arrangements. Each trial consisted of:

(1) Autonomous take-of and survey of the workspace;

(2) Object and human identification, classification, and localization;

(3) Motion planning and execution from home to object, object grasp, and human-aware navigation from object to human;

(4) Logging of trial outcome (success/failure), execution time, and failure mode (if any).

All onboard sensor data, perception outputs, and control commands were recorded as ROS 2 bag files for ofline analysis. Vi con trajectories were time-synchronized with the drone’s onboard logs to enable quantitative evaluation of perception accuracy and motion-planning performance.

## 3.2 Discussion

In the experiments performed, we demonstrated a fully functioning, end-to-end MediaPipe based on grounding DINO for object and human identification and localization.

Recent work such as RaceVLA [10] demonstrate the power of end-to-end learning, where a single transformer model maps raw visual inputs directly to 4D control commands. While RaceVLA achieves impressive agility and human-like behaviors in high speed navigation tasks, it operates as a "black box." In the context to aerial manipulation, this poses a safety risk: hallucination in the VLA could result in a direct collision command. Our approach mitigates this by decoupling reasoning from actuation. By using VLA solely for semantic intent and relying on deterministic visual servoing for the how, DroneVLA ensures drone’s motion is governed by verifiable control laws.

On the other end of the spectrum, systems like AERMANI-VLM used structured prompting to generate high-level reasoning traces that select discrete skill primitives. While robust for planning, these systems can sufer from rigidity in dynamic execution. DroneVLA occupies a middle ground; it employs the semantic flexibility of VLMs for object grounding but integrates tight closed-loop visual servoing to handle the continuous adjustments required for grasping.

A total of 10 experiments were performed and all experiments were successfully implemented with a max error of 0.164m, a mean euclidean error of 0.070m, and a root mean squared error of 0.084m. This can be visualized in the home-to-object and object-to-human trajectories of the 2-3-D plot of Fig. 4. Also it is noticeable that the drone stays a safe distance of 1 m away from the human achieving the desired safe handover trajectory.

Although the VLA architecture was not implemented in the real world flight control loop, a Unity-based simulation was performed to strictly validate the grasping policy. 10 simulation experiments were conducted where visual frames were fed to the model to verify the correctness of the predicted gripper action. In all tri als, the model correctly identified the appropriate binary action (open/close) based on the image state. It is important to note that the validated VLA operates as a binary actuation model: it analyzes the received image state—whether from simulation or real-world feeds—and triggers a discrete ‘open’ or ‘close’ gripper command. This confirms the feasibility of the vision-to-action mapping and validates the model’s decision-making logic prior to physical deployment.

![](<images/4ecb88eef9dbcec0ea577913f4d9a9fe29384a0cd8268cab034aaf8da0425379.jpg>)  
Figure 4: 2D and 3D views of the human–aware A\* motion plan, showing the reference trajectory, planned path, safety margins around the human and table with achieved trajectory in real flight.

## 4 Conclusion and Future Work

Having established an end-to-end MediaPipe and Grounding DINO framework for localization in versatile environments, we proposed and validated a novel concept for VLA-based aerial manipulation. The results demonstrate the robustness of the perception stack in real-world flights and the accuracy of the VLA’s grasp validation logic, which was successfully verified on both simulated and realworld visual data. This work serves as a foundational step towards embodied aerial intelligence.

Future work will focus on transitioning from open-loop vali dation to closed-loop real-time control. Crucially, we intend to leverage this complete framework as a data collection pipeline; by recording the visual inputs and corresponding actions during these flights, we will generate a comprehensive dataset to train a custom, end-to-end VLA model for full 5-DOF control. Finally, we aim to extend this architecture to industrial domains by developing a specialist VLA model tailored for complex indoor navigation and manipulation tasks, enabling fully autonomous workflows in dynamic operational environments.

## Acknowledgements

Research reported in this publication was financially supported by the RSF grant No. 24-41-02039.

## References

[1] X. Guo, G. He, J. Xu, M. Mousaei, J. Geng, S. Scherer, and G. Shi. 2024. Flying calligrapher: Contact-aware motion and force planning and control for aerial manipulation. IEEE Robotics and Automation Letters (2024). doi:10.1109/LRA. 2024.3486236

[2] H.B. Khamseh, F. Janabi-Sharifi, and A. Abdessameud. 2018. Aerial manipulation—A literature survey. Robotics and Autonomous Systems 107 (2018), 221–235. doi:10.1016/j.robot.2018.03.00

[3] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, and Lei Zhang. 2024. Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection. arXiv:2303.05499. Retrieved from https://arxiv.org/abs/2303.05499.

[4] L. Meng, Y. Rong, and W. Chou. 2025. Bilateral Teleoperation of Aerial Manipu lator with Hybrid Mapping Framework for Physical Interaction. Sensors 25, 18 (2025), 5844. doi:10.3390/s25185844

[5] Sarthak Mishra, Rishabh Dev Yadav, Avirup Das, Saksham Gupta, Wei Pan, and Spandan Roy. 2025. AERMANI-VLM: Structured Prompting and Reasoning for Aerial Manipulation with Vision Language Models. arXiv:2511.01472. Retrieved from https://arxiv.org/abs/2511.01472

[6] Ryosuke Morishita, Shin Kawai, and Hajime Nobuhara. 2024. Downwash reduc tion drone with adaptive rotors and its 3D aerodynamic analysis and stabilization control. IEEE Access 12 (2024), 22832–22840. doi:10.1109/ACCESS.2024.3362639

[7] Heramb Nemlekar, Dharini Dutia, and Zhi Li. 2019. Object transfer point es timation for fluent human-robot handovers. In Proc. Int. Conf. on Robotics and Automation (ICRA). IEEE, 2627–2633. doi:10.1109/ICRA.2019.8794008

[8] F. Ruggiero, V. Lippiello, and A. Ollero. 2018. Aerial manipulation: A literature review. IEEE Robotics and Automation Letters 3, 3 (2018), 1957–1964. doi:10.1109 LRA.2018.2822650

[9] Oleg Sautenkov, Yasheerah Yaqoot, Artem Lykov, Muhammad Ahsan Mustafa, Grik Tadevosyan, Aibek Akhmetkazy, Miguel Altamirano Cabrera, Mikhail Martynov, Sausar Karaf, and Dzmitry Tsetserukou. 2025. UAV-VLA: Vision language-action system for large scale aerial mission generation. In Proc. 20th ACM/IEEE Int. Conf. on Human-Robot Interaction (HRI). IEEE, 1588–1592. doi:10.1109/HRI61500.2025.10974117

[10] Valerii Serpiva, Artem Lykov, Artyom Myshlyaev, Muhammad Haris Khan, Ali Alridha Abdulkarim, Oleg Sautenkov, and Dzmitry Tsetserukou. 2025. RaceVLA: VLA-based Racing Drone Navigation with Human-like Behaviour. arXiv:2503.02572. Retrieved from https://arxiv.org/abs/2503.02572.

[11] Kyle Strabala, Min Kyung Lee, Anca Dragan, Jodi Forlizzi, Siddhartha S Srinivasa, Maya Cakmak, and Vincenzo Micelli. 2013. Toward seamless human-robot handovers. Journal ofHuman-Robot Interaction 2, 1 (2013), 112–132. doi:10.5898/ JHRI.2.1.Strabala

[12] H. Tsukagoshi, M. Watanabe, T. Hamada, D. Ashlih, and R. Iizuka. 2015. Aerial Manipulator with Perching and Door-Opening Capability. In Proc. IEEE Int. Conf. on Robotics and Automation (ICRA). IEEE, 4663–4668. doi:10.1109/ICRA.2015. 7139845

[13] Junjie Wen, Yichen Zhu, Jinming Li, Minjie Zhu, Kun Wu, Zhiyuan Xu, Ning Liu, Ran Cheng, Chaomin Shen, Yaxin Peng, Feifei Feng, and Jian Tang. 2025. TinyVLA: Toward Fast, Data-Eficient Vision-Language-Action Models for Robotic Manipulation. IEEE Robotics and Automation Letters 10, 4 (2025), 3988– 3995. doi:10.1109/LRA.2025.3544909

[14] M. Zhao, K. Nagato, K. Okada, M. Inaba, and M. Nakao. 2022. Forceful Valve Manipulation with Arbitrary Direction by Articulated Aerial Robot Equipped with Thrust Vectoring Apparatus. IEEE Robotics and Automation Letters 7, 2 (2022), 4893–4900. doi:10.1109/LRA.2022.3154018