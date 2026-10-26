# Flying Hand: End-Effector-Centric Framework for Versatile Aerial Manipulation Teleoperation and Policy Learning

Guanqi He\*<sup>†</sup>, Xiaofeng Guo\*<sup>†</sup>, Luyi Tang<sup>†</sup>, Yuanhang Zhang<sup>†</sup>, Mohammadreza Mousaei<sup>†</sup>, Jiahe Xu<sup>†</sup>, Junyi Geng<sup>‡</sup>, Sebastian Scherer<sup>†</sup>, Guanya Shi<sup>†</sup>

Abstract—Aerial manipulation has recently attracted increasing interest from both industry and academia. Previous approaches have demonstrated success in various specific tasks. However, their hardware design and control frameworks are often tightly coupled with task specifications, limiting the development of cross-task and cross-platform algorithms. Inspired by the success of robot learning in tabletop manipulation, we propose a unified aerial manipulation framework with an end-effector-centric interface that decouples high-level platformagnostic decision-making from task-agnostic low-level control. Our framework consists of a fully-actuated hexarotor with a 4-DoF robotic arm, an end-effector-centric whole-body model predictive controller, and a high-level policy. The high-precision end-effector controller enables efficient and intuitive aerial teleoperation for versatile tasks and facilitates the development of imitation learning policies. Real-world experiments show that the proposed framework significantly improves end-effector tracking accuracy, and can handle multiple aerial teleoperation and imitation learning tasks, including writing, peg-in-hole, pick and place, changing light bulbs, etc. We believe the proposed framework provides one way to standardize and unify aerial manipulation into the general manipulation community and to advance the field. Project website: https://lecar-lab.github.io/flying hand/.

## I. INTRODUCTION

Uncrewed Aerial Manipulators (UAMs), which target complex tasks at high altitudes [1], hold significant potential to reduce human labor in many elevated operations, such as changing light bulbs on tall towers, inspecting aircraft wings or turbine blades, and maintaining or painting bridges, which are not only costly but also pose substantial risks to human safety. Previous works have demonstrated the ability to achieve different specific complex aerial manipulation tasks, including drawing calligraphy [2], grasping [3], perching [4], drilling [5], etc. However, most previous works have been tailored to specific tasks, developing unique platforms and algorithms accordingly, lacking the ability to handle different types of tasks. In real-world scenarios, manipulation tasks can be complex and typically consist of multiple sub-tasks. For example, changing a light bulb can involve several motion primitives, including interaction, grasping, insertion, and rotation. This raises a requirement for a more general-purpose versatile aerial manipulation system, which essentially requires a versatile aerial manipulation framework to handle multiple tasks.

In robotic manipulation [6], the end-effector-centric (eecentric) approach is widely used. It defines tasks and policies [7, 8] in Cartesian space instead of specific robotic arm configuration space. By effectively decoupling highlevel policies from low-level control, it enables embodimentagnostic policies [9], [10], [11] and policy-agnostic low-level controllers development [12], enhancing framework versatility, cross-embodiment adaptability, and algorithm reuse capability [13]. Although shown the advantage of versatility in the manipulation field, applying the ee-centric paradigm to aerial manipulation systems presents significant challenges due to the UAM’s floating-base dynamics and the coupling effects between the UAV and the manipulator.

![](<images/ed6e611d781f227f2f21691ba93f35682822b8a6826f88b112c56817264cc8f2.jpg>)

![](<images/83ca344e21a36336a7cabb4d58b3ff8f3a23b869891e323a997348e25cc466f1.jpg>)

![](<images/39572be736a6be4b0a07723f161d8a644e73ffa1764618af3266fae8119ef5e6.jpg>)

![](<images/290522889052a54c875000f6f5046e287c3827d1208da36de0a71e5e863ce480.jpg>)  
Fig. 1. The proposed framework and system can accomplish multiple typical aerial manipulation tasks precisely and robustly, such as (a) writing ”2025”, (b) peg-in-hole, (c) pick-and-place, and (d) changing light bulbs.

In this work, inspired by the success of ee-centric interfaces in classic manipulation, we reformulate aerial manipulation from the manipulation perspective by proposing a versatile aerial manipulation framework with the ee-centric interface to address various aerial manipulation tasks. The framework consists of a versatile aerial manipulation platform capable of executing multiple tasks, a policy-agnostic controller that precisely tracks the target end effector state, and an ee-centric policy module responsible for generating target ee states. Specifically, we develop a fully-actuated hexarotor with a 4 DoF robotic arm, providing a sufficiently large workspace and wrench space for diverse tasks. We then develop an ee-centric whole-body Model Predictive Controller (ee-centric MPC) that precisely tracks the target end-effector state, even in the presence of model uncertainties. Moreover, to bring human cognitive skills into policy development and benefit from the ee-centric interface, we develop an ee-centric teleoperation interface and an imitation-learning-based framework to acquire autonomous policy learned from human demonstration. To the best of our knowledge, this is the first imitation learning-based framework for aerial manipulation. Real-world experiments show that the proposed framework achieves high-precision end-effector tracking and enables a wide range of aerial manipulation tasks, including aerial writing, peg-in-hole, pick and place, light bulb replacement, etc., as shown in Fig. 1. It also demonstrates how this modular and standardized ee-centric framework effectively decouples the high-level policy from the low-level controller, which enables seamless integration of existing standard high-level policy modules from the broader manipulation community, such as teleoperation and imitation learning, into the field of aerial manipulation. We believe the proposed framework provides a step toward standardizing and unifying aerial manipulation into the broader manipulation community, advancing the field toward greater versatility and generalization.

![](<images/b8f7d33d20756790785df595b272f65f0c033564a5143d0cf49ab9d33c96c318.jpg>)  
Fig. 2. The proposed end-effector-centric aerial manipulation framework includes the UAM platform, the ee-centric whole-body MPC, and the high-level policy including an ee-centric teleoperation interface, and an imitation learning-based framework using Action Chunk with Transformer (ACT) [14]. The high-level policy, either the human teleoperation or learned autonomous policy sends the target end-effector state to ee-centric MPC which then generates motor commands for the UAM platform to execute.

## In summary, our contributions are:

1). We reformulated the aerial manipulation problem within the unified manipulation framework, consisting of a UAM system, a controller encapsulated by the ee-centric interface, and a high-level policy.

2). We developed an end-effector-centric whole-body MPC for aerial manipulation that precisely tracks the targeted endeffector state while maintaining robustness against disturbances through L1 adaptation.

3). We developed an ee-centric teleoperation system and an imitation-learning-based autonomous system that learns from human teleoperation demonstration.

4). Rich real-world experiments demonstrated the versatility of the proposed framework, the effectiveness of the userfriendly teleoperation interface, and the potential to incorporate learning-based policies and other manipulation policies.

## II. RELATED WORKS

## A. Aerial Manipulation

There have been many research efforts exploring aerial manipulation for various kinds of tasks [15]. Based on the motion primitives they require, common aerial manipulation tasks can be categorized into: 1). Aerial Interaction, which requires maintaining contact with external objects, for tasks such as inspection [16] [17] [18], aerial writing [19] [20], [21] or pushing a target [22]. Researchers mostly developed a point-contact arm such as a rigid rod, and proposed the hybrid motion-force control framework, although achieving highprecision tracking performance, struggles to handle scenarios requiring grasping; 2) Aerial grasping [23], where previous works mainly focused on designing different custom endeffectors, such as claw [24] or soft gripper [25]. Some work also showed amazing results achieving high-speed grasping, or grasping moving objects [26], but sacrificed payload capacity or precision due to specialized hardware designs. 3). Aerial insertion. Typical work includes [27] where they propose a specific hole searching policy for bolt screwing tasks, and [28] where they achieved mm-level peg-in-hole task; 4).

Manipulate articulated objects such as doors [29], or valves [30]. In general, although different works have shown success on different specific tasks, the specific system design and algorithm development make the same hardware and algorithm hard to deploy to different tasks, reducing its potential for practical long-horizon versatile aerial manipulation tasks. In our work, we target all these four types of aerial manipulation tasks, developing a versatile framework to handle all of them.

## B. Manipulation with EE-Centric Interface

Defining tasks or commands using ee-centric approaches is widely adopted in general manipulation fields, as it is more intuitive and can be cross-embodiment. For example, [31] developed a teleoperation interface to enable full control of the end-effector pose. The Universal Manipulation Interface (UMI) [10] [32] demonstrates a data-collection and policylearning framework that allows direct skill transfer from inthe-wild human demonstrations to multiple robot embodiments. Their system employs a hand-held gripper and carefully designed hardware-agnostic policies, showcasing the potential for ee-centric solutions in multi-platform scenarios. Similarly, other mobile manipulation strategies, such as N2M2 [33] [34] and HarmonicMM [35], reduce the operator’s burden by extracting feasible base motions from end-effector trajectories. But they generally remain limited to ground robots. Although several aerial manipulation studies have adopted end-effectorcentric methods [2] [36], they have primarily focused on developing controllers to track specified end-effector trajectories without a systematic framework that tackles various tasks comprehensively. In our work, we propose a unified framework with the ee-centric interface for versatile aerial manipulation tasks.

## C. Teleportation and Imitation Learning

Developing a robust and practical autonomous aerial manipulation policy is extremely challenging due to complex realworld environments and high precision and safety requirements. Moreover, policies are typically designed to handle specific tasks and lack the generality to handle unexpected conditions. Therefore, teleoperation, which takes human effort into the loop for policy development, attracts researchers interest as a practical solution. For example, [37] developed the UAM with a fully actuated UAV with 0 DoF arm and controlled the end effector directly by teleoperation, but their method is highly coupled with the specific UAM design and the system suffers for versatile tasks due to the workspace limitation. In most structured UAM teleoperation works, users control each DoF separately [38] or explicitly switch to different modes during different phases [39]. Both of these increase the human teleoperator’s burden and require the teleoperator to have a rich understanding of the specific hardware system. Moreover, even if incorporated with human input, the previous works still have not shown vast versatility in different tasks and scenarios.

Recently, imitation learning (IL) has demonstrated significant potential for autonomous policy learning due to its high data efficiency, straightforward framework, and outstanding performance. Recent progress in both systems, such as ALOHA [14] and mobile ALOHA [40], and algorithms, such as ACT [14], and diffusion policy [41], facilitate the success of long-horizon, contact-rich, complex manipulation tasks. However, there is no precedent to incorporate such IL-based policy into aerial manipulation fields due to the lack of a mature demonstration collection system such as a well-development teleoperation system, and the lack of a proven framework. In this work, we develop an intuitive teleoperation system using the ee-interface framework. It also helps to collect human demonstration data, enabling us to develop an imitation learning-based policy for autonomous aerial manipulation.

## III. SYSTEM OVERVIEW

Our aerial manipulation system is designed to enable precise and versatile operations. The system incorporates an endeffector-centric (ee-centric) interface to decouple high-level decision-making from low-level control, increasing the framework’s versatility. As shown in Fig. 2, our system consists of an aerial manipulator platform, an ee-centric whole-body MPC, and an ee-centric policy module. The hardware platform consists of a fully-actuated hexarotor and a 4 DoF robotic arm. The platform has a large enough workspace and wrench space for different tasks. A motion capture system and onboard IMUs are used for drone state estimation. The joint encoders are used to get arm joint angles and the end-effector states are then calculated based on forward kinematics. The eecentric whole-body MPC reads the ee target from high-level policy and generates the reference trajectory and reference control for both the UAV and the robotic arm. An L1 online adaptation control term is designed to further improve the tracking performance. The UAV control commands are then sent to the control allocation to generate the motor commands for the UAV to execute. At the highest level, the ee-centric policy module gets current observations and generates the target ee states online without the need to consider the specific platform jointly. We developed two high-level policy modules. The first one is the ee-centric teleoperation interface that allows human users to directly control the end-effector pose. Based on the human demonstration, we adopted an imitationlearning-based method: Action Chunk with Transformer(ACT) [42], to learn an autonomous policy. The following sections will introduce the developed modules accordingly.

## IV. HARDWARE DESIGN

## A. Fully-Actuated UAV

The foundation of the system is a fully-actuated hexarotor UAV capable of independently generating six-dimensional forces and torques. This capability allows precise control of position and orientation, which is essential for executing complex aerial manipulation tasks. The robust design ensures stability in dynamic environments, while ensuring highprecision end-effector tracking. We used Tarot680 as the drone base, 6 KDE 4215XF motor with a 12-inch 2-blade propeller as our driving force, LiPo batteries for on-board power supply, an on-board computer Intel Nuc for on-board computation, and a customized PX4 autopilot for low-level flight control and information processing.

![](<images/ff460e4a5107db24d52991febde533ca85dae130178e8dafefa7a46d63ce03e0.jpg>)  
Fig. 3. UAM hardware system design, illustrating the key components: (1) fully-actuated hexarotor as the base structure, (2) 4 Dof manipulator, (3) Intel RealSense cameras for vision-based perception and feedback, and (4) end-effector gripper for object interaction. The frame notations in the right diagram represent the coordinate axes associated with the system.

## B. Manipulator

The UAV integrates a 4-DOF robotic manipulator optimized for versatile and precise task execution. The arm features three pitch joints and one roll joint, driven by Dynamixel XM540 and XM430 servo. Its configuration allows high-precision operations. The system achieves whole-body manipulation capabilities by combining the UAV’s actuation with the manipulator, enhancing task execution across diverse scenarios. The manipulator includes a modular end-effector, allowing interchangeable tools for specific tasks. For instance, a twofinger gripper with replaceable tips enables precision handling, while a circular gripper is ideal for changing light bulbs.

## C. Perception

In this work, we mainly focus on the in-door environment. We use both the motion capture system and the PX4 onboard IMU for drone state estimation. The motion capture system provides the UAV’s position and can be replaced with other localization methods, such as SLAM. The manipulator arm joint angles are estimated by the joint encoders and the end-effector states are online calculated based on forward kinematics (FK).

To further improve drone perception for teleoperation and autonomous policy development, we equip the aerial manipulator with two RealSense RGBD cameras. One camera is mounted on the UAV base to capture a broad view of the entire workspace, while the other is positioned near the endeffector to deliver detailed close-up views of the target area. This dual-camera setup ensures teleoperators and vision-based policies can maintain precise control and situational awareness during complex manipulation tasks.

## V. SYSTEM MODELING

## A. Frames and Notation

The frames depicted in the Fig 3 are defined as follows: $\mathcal { F } _ { W }$ is an inertial world frame with its z-axis opposite to the gravity vector, ensuring $\hat { z } _ { W }$ points upward. $\mathcal { F } _ { B }$ is rigidly attached to the $\mathrm { U A V } \mathbf { \dot { s } }$ body at its center of gravity, with axes $\left( x _ { B } , y _ { B } , z _ { B } \right)$ aligned with the UAV body frame. $\mathcal { F } _ { D }$ is the manipulator base frame, centered at the attachment point of the manipulator on the UAV. The transformation from $\mathcal { F } _ { B }$ to $\mathcal { F } _ { D }$ is defined by a constant translation $\mathbf { p } _ { D } \in \mathbb { R } ^ { 3 }$ and a fixed orientation $\pmb { R } _ { B } ^ { D } \in$ $S O ( 3 ) . \mathcal { F } _ { E }$ is the end-effector frame with axes $\left( x _ { E } , y _ { E } , z _ { E } \right)$ where $x _ { E }$ is aligned with the roll axis of the 4th joint of the manipulator, while $y _ { E }$ remains horizontal. Other symbols in the paper are listed in Table I for the convenience of the following discussion.

TABLE I  
NOTATION OVERVIEW

<table><tr><td>Symbol</td><td>Dimension</td><td>Description</td></tr><tr><td> $m$ </td><td> $\mathbb{R}$ </td><td>Mass</td></tr><tr><td> $J$ </td><td> $\mathbb{R}^{3 \times 3}$ </td><td>Inertia tensor</td></tr><tr><td> $p, R_{B}^{W}$ </td><td> $\mathbb{R}^{3}, SO(3)$ </td><td>UAV position expressed in  $\mathcal{F}_{W}$  and orientation between  $\mathcal{F}_{B}$  and  $\mathcal{F}_{W}$ </td></tr><tr><td> $v$ </td><td> $\mathbb{R}^{3}$ </td><td>Generalized velocity, expressed in  $\mathcal{F}_{W}$ </td></tr><tr><td> $\tau$ </td><td> $\mathbb{R}^{6}$ </td><td>Control wrench</td></tr><tr><td> $\tau_{\text{ext}}$ </td><td> $\mathbb{R}^{6}$ </td><td>External disturbance wrench</td></tr><tr><td> $g_{W}$ </td><td> $\mathbb{R}^{6}$ </td><td>Gravity vector in  $\mathcal{F}_{W}$ </td></tr><tr><td> $p_{E}, R_{E}^{W}$ </td><td> $\mathbb{R}^{3}, SO(3)$ </td><td>End-effector position expressed in  $\mathcal{F}_{W}$  and orientation between  $\mathcal{F}_{E}$  and  $\mathcal{F}_{W}$ </td></tr><tr><td> $\theta$ </td><td> $\mathbb{R}^{4}$ </td><td>Current joint angles</td></tr><tr><td> $\theta_{\text{cmd}}$ </td><td> $\mathbb{R}^{4}$ </td><td>Commanded joint angles</td></tr><tr><td> $d$ </td><td> $\mathbb{R}^{4}$ </td><td>Joint servo disturbance</td></tr><tr><td> $\zeta$ </td><td> $\mathbb{R}^{3 \times 4}$ </td><td>DH parameter with the  $i^{\text{th}}$  joint component  $[\theta_{i}, l_{i}, a_{i}, \alpha_{i}]^{T}$ </td></tr><tr><td> $\beta$ </td><td> $\mathbb{R}$ </td><td>Joint motor delay constant</td></tr></table>

## B. Fully-Actuated UAV Dynamics

A fully-actuated UAV is adopted as the base of the aerial manipulator, which can generate six-dimensional force and torque independently. Let the generalized position of the UAV be represented as $\dot { \textbf { \textit { q } } } = [ p , R _ { B } ^ { W } ]$ , where $\pmb { p } \in \mathbb { R } ^ { 3 }$ represents the position of the UAV in the global coordinate frame, and ${ \pmb R } _ { B } ^ { W } \ \in \ S O ( 3 )$ represents its orientation. The generalized velocity is denoted as $\pmb { v } = [ \dot { p } , \omega ]$ , where $\omega \in \mathbb { R } ^ { 3 }$ is the angular velocity.

The UAV dynamics can be formulated using Newton-Euler equations for rigid body motion as follows:

$$
M \dot {\boldsymbol {v}} + C \boldsymbol {v} + \boldsymbol {g} = \boldsymbol {\tau} + \boldsymbol {\tau} _ {\mathrm{ext}}\tag{1}
$$

with inertia matrix $M \in \mathbb { R } ^ { 6 \times 6 }$ , centrifugal and Coriolis term $C \in \mathbb { R } ^ { 6 \times 6 }$ , gravity wrench $\pmb { g } \in \mathbb { R } ^ { 6 }$ , control wrench $\tau \in \mathbb { R } ^ { 6 }$ from the UAM actuators, and unknown external wrench $\tau _ { e x t } \in$ $\mathbb { R } ^ { 6 }$ from model mismatch and manipulator interaction.

Specifically, we have

$$
\boldsymbol {M} = \operatorname{diag} \left(\left[ \begin{array}{c c} m \boldsymbol {I} _ {3 \times 3} & \boldsymbol {J} \end{array} \right]\right),\tag{2}
$$

$$
\boldsymbol {C} = \mathrm{diag} \left(\left[ \begin{array}{c c} m [ \boldsymbol {\omega} ] _ {\times} & - \boldsymbol {J} [ \boldsymbol {\omega} ] _ {\times} \end{array} \right]\right),\tag{3}
$$

$$
\pmb {g} = m \mathrm{diag} \left(\left[ \begin{array}{c c} \pmb {R} _ {B} ^ {W} & \mathbf {0} _ {3 \times 3} \end{array} \right]\right) \pmb {g} _ {W}.\tag{4}
$$

where m is the vehicle mass, J is the moment of inertia at the vehicle center of mass in the body frame, $\begin{array} { r l } { { \pmb g } _ { W } } & { { } = } \end{array}$ $[ 0 , 0 , g , 0 , 0 , 0 ] ^ { \top }$ is the gravitational acceleration in $\mathcal { F } _ { W }$ , and $[ * ] _ { \times }$ is the skew-symmetric matrix associated with vector ∗.

## C. Manipulator Kinematics

In this work, the manipulator employs servos as joint actuators, which cannot accurately control joint torques directly. Therefore, only the kinematics of the manipulator is considered in the system modeling. The interaction between the manipulator and the fully-actuated UAV is treated as a disturbance and is compensated in real-time using L1 adaptive control.

We use the standard Denavit–Hartenberg (DH) convention [43] to model the forward kinematics of our 4-DoF robotic arm. Under the DH formulation, the adjacent frame transformation $\pmb { T } _ { i } ^ { i - 1 }$ is characterized by four parameters $\theta _ { i } , \ d _ { i } ,$ $a _ { i } , ~ \alpha _ { i }$ , where the first one is the joint angle and the last three are pre-identified and fixed during robot movement. Define DH parameter $\zeta _ { i } = [ \theta _ { i } , l _ { i } , a _ { i } , \alpha _ { i } ] ^ { \top } \in \mathbb { R } ^ { 4 }$ . The frame transformation from end-effector frame to arm base body frame can be written as

$$
\boldsymbol {T} _ {E} ^ {D} (\boldsymbol {\theta}; \boldsymbol {\zeta}) = \prod_ {i = 1} ^ {4} \boldsymbol {T} _ {i} ^ {i - 1} \left(\theta_ {i}; \zeta_ {i}\right)\tag{5}
$$

Then the transformation from world frame to end-effector frame can be computed as follows:

$$
\boldsymbol {T} _ {E} ^ {W} = \left[ \begin{array}{c c} \boldsymbol {R} _ {E} ^ {W} & \boldsymbol {p} _ {E} ^ {W} \\ \boldsymbol {0} _ {1 \times 3} & 1 \end{array} \right] = \boldsymbol {T} _ {B} ^ {W} \cdot \boldsymbol {T} _ {D} ^ {B} \cdot \boldsymbol {T} _ {E} ^ {D}\tag{6}
$$

where $\pmb { T } _ { B } ^ { W }$ can be obtained from UAV odometery and $\pmb { T } _ { D } ^ { B }$ is a fixed transformation between the manipulator base and the UAV.

The accurate DH parameters are obtained through system identification. We collected motion data of the manipulator using a motion capture system and compute the DH parameters via least squares regression. The detailed parameter values are presented in Table III.

## D. Manipulator Motor Delay

The servo motor dynamics are approximated as first-order systems to account for command-to-state delay. For the 4-DoF manipulator, the relationship between commanded joint angles $\theta _ { \mathrm { c m d } } \in \mathbb { R } ^ { 4 }$ and actual joint angles $\pmb \theta \in \mathbb { R } ^ { 4 }$ is governed by:

$$
\mathrm{diag} (\boldsymbol {\beta}) \dot {\boldsymbol {\theta}} + \boldsymbol {\theta} = \boldsymbol {\theta} _ {\mathrm{cmd}} + \boldsymbol {d}\tag{7}
$$

where $\beta \in \mathbb { R }$ are the joint-specific time delay constants, and $\ b { d } \in \mathbb { R } ^ { 4 }$ is the unknown disturbance in servo control. This formulation captures the transient response characteristics of each actuator. The motor delay coefficients $\beta$ are identified alongside the DH parameters using least squares regression, with the results presented in Table III.

## VI. END-EFFECTOR-CENTRIC WHOLE-BODY CONTROL WITH ONLINE ADAPTATION

Given the over-actuated nature of our system and the users’ primary focus on the end-effector motion, we employ model predictive control to regulate the end-effector trajectory. This approach enables whole-body coordination between the manipulator and the fully-actuated UAV, ensuring precise and efficient end-effector motion control. As discussed in previous sections, the complex interaction between the manipulator and the UAV is treated as the disturbance in the modeling stage, which introduces uncertainty in the nominal model used in the whole-body MPC. To mitigate the disturbances and model uncertainties, we integrate the L1 adaptive controller, ensuring robust disturbance compensation and accurate tracking performance. The diagram of the control algorithm is illustrated in Fig. 2.

## A. End-Effector-Centric Model Predictive Controller

In the following, we describe the whole-body MPC framework used to optimize the end-effector reference trajectory.

For the MPC formulation, we define the following state and control variables:

$$
\boldsymbol {x} := \left[ \begin{array}{c c c c} \boldsymbol {p} _ {E} & \boldsymbol {R} _ {E} ^ {W} & \boldsymbol {v} & \boldsymbol {\theta} \end{array} \right] \quad \boldsymbol {u} := \left[ \begin{array}{c c} \boldsymbol {\tau} & \boldsymbol {\theta} _ {\mathrm{cmd}} \end{array} \right]\tag{8}
$$

We use the following error functions for the position of the end-effector, the UAM velocity, the wrench control input, and the manipulator joint angle, respectively:

$$
\boldsymbol {e} _ {p} = \boldsymbol {p} _ {E} - \boldsymbol {p} _ {E} ^ {r}\tag{9a}
$$

$$
\pmb {e} _ {R} = \frac {1}{2} \big (\pmb {R} _ {W} ^ {E r \top} \pmb {R} _ {W} ^ {E} - \pmb {R} _ {W} ^ {E \top} \pmb {R} _ {W} ^ {E r} \big) ^ {\vee}
$$

$$
\pmb {e} _ {v} = \pmb {v} - \pmb {v} ^ {r}\tag{9b}
$$

$$
\pmb {e} _ {\theta} = \pmb {\theta} - \pmb {\theta} ^ {r}\tag{9c}
$$

(9d)

$$
\boldsymbol {e} _ {u} = \boldsymbol {u} - \boldsymbol {u} ^ {r}\tag{9e}
$$

where $( * ) ^ { \vee }$ is the vee-operator that extracts a vector from a skew-symmetric matrix ∗, and $( \cdot ) ^ { r }$ represents the reference state values. The reference signals $\pmb { p } _ { E } ^ { r }$ and $R _ { E } ^ { W }$ are provided by a high-level teleoperation command or derived from an imitation learning policy. Manipulator default joint angle $\pmb { \theta } ^ { r }$ is pre-selected, and reference control ${ \pmb u } ^ { r } = [ { \bf 0 } _ { 3 } , \hat { \pmb \theta } ]$ , where $\hat { \pmb { \theta } }$ is the current joint states.

The MPC formulation minimizes a cost function over a finite time horizon H while subject to system dynamics and constraints:

$$
\boldsymbol {u} _ {\text { opt }} = \arg \min _ {\boldsymbol {u}} \left\{L _ {e} (\boldsymbol {x} _ {H}, \boldsymbol {x} _ {H} ^ {r}) + \sum_ {i = 1} ^ {H} L _ {r} (\boldsymbol {x} _ {n}, \boldsymbol {x} _ {n} ^ {r}, \boldsymbol {u} _ {n}) \right\}\tag{10a}
$$

$$
\mathrm{s.t.} \quad \boldsymbol {x} _ {n + 1} = \boldsymbol {f} _ {d y n} (\boldsymbol {x} _ {n}, \boldsymbol {\tau} _ {n})\tag{10b}
$$

$$
\boldsymbol {x} _ {0} = \hat {\boldsymbol {x}}, \quad \boldsymbol {x} _ {n} \in \mathcal {X}\tag{10c}
$$

$$
\boldsymbol {u} _ {\mathrm{lb}} \leq \boldsymbol {u} \leq \boldsymbol {u} _ {\mathrm{ub}}\tag{10d}
$$

Eq. (10a) defines the optimization objective, where H represents the discrete prediction horizon. The stage and terminal costs, $L _ { r }$ and $L _ { e }$ , are quadratic functions of the tracking errors, given by $e _ { i } ^ { \top } Q _ { i } e _ { i }$ , where $\pmb { x } _ { i } \in \{ e _ { p } , e _ { R } , e _ { v } , e _ { \theta } , e _ { u } \}$ . The gain matrices $Q _ { i }$ are positive definite and tuned experimentally to balance precision and robustness.

Eq. (10b) enforces the discrete-time system dynamics, incorporating the fully actuated UAV dynamics Eq. (1), manipulator kinematics Eq. (6), and joint servo dynamics Eq. (7). The continuous system dynamics are discretized using a fourthorder Runge-Kutta (RK4) integration scheme to maintain numerical stability and accuracy. Disturbances $\tau _ { \mathrm { e x t } }$ and d are ignored in the MPC formulation and solving process, but will be handled in the following Section VI-B via online L1 adaptation.

Eq. (10c) introduces state constraints, where xˆ represents the latest state estimate. The feasible state space X is defined by: 1) Self-collision avoidance constraints: ensuring that the manipulator does not collide with the UAV structure. 2) Environment collision constraints: preventing the UAV contact with external obstacles. 3) Safety constraints: including velocity limits and joint angle restrictions to ensure safe operation.

Eq. (10d) imposes actuation limits on the aerial manipulator, where $\mathbf { \Delta } \mathbf { u } _ { \mathrm { l b } }$ and $\mathbf { \Delta } \mathbf { u } _ { \mathrm { u b } }$ define the lower and upper bounds of the control inputs.

The end-effector centric whole-body MPC formulation is a general framework that can adapt to various vehicle types and can extend to systems with multiple end-effectors. The control inputs, UAV dynamics, and arm kinematics can be tailored to specific vehicle and manipulator configurations, ensuring flexibility across different aerial manipulation systems.

## B. L1 Online Adaptation

As discussed in previous sections, the complex interaction between the manipulator and the UAV is treated as the disturbance, which introduces uncertainty in the nominal model used in the whole-body MPC. Such model uncertainties are typically bounded, and prior knowledge about their characteristics is usually available. To mitigate the disturbances and model uncertainties, we integrate the L1 adaptive controller, ensuring robust disturbance compensation and accurate tracking performance.

We adopt the L1 adaptive controller from [44] in both the fully-actuated UAV motion control and the manipulator joint angle tracking control, to compensate the disturbance $\tau _ { \mathrm { e x t } }$ in Eq. (1) and d in Eq. (7).

The adaptation law is designed by

$$
M \dot {\hat {v}} + C \hat {v} + g = \tau + \hat {\tau} _ {\mathrm{ext}} + A _ {v} (\hat {v} - v)\tag{11a}
$$

$$
\hat {\pmb {\tau}} _ {\mathrm{ext}} ^ {\prime} = - (e ^ {\pmb {A} _ {v} d t} - \pmb {I} _ {6 \times 6}) ^ {- 1} \pmb {A} _ {v} e ^ {\pmb {A} _ {v} d t} (\hat {\pmb {v}} - \pmb {v})\tag{11b}
$$

$$
\hat {\boldsymbol {\tau}} _ {\text { ext }} \leftarrow \text { low   pass   filter } (\hat {\boldsymbol {\tau}} _ {\text { ext }}, \hat {\boldsymbol {\tau}} _ {\text { ext }} ^ {\prime})\tag{11c}
$$

where $\hat { \textbf { \textit { v } } } \in \textbf { R } ^ { 6 }$ denotes the estimated UAV velocity, $A _ { v }$ is a Hurwitz matrix, dt is the discretization step length, and $\hat { \tau } _ { \mathrm { e x t } } ~ \in ~ \mathbb { R } ^ { 6 }$ encapsulates the unknown wrench disturbances. Here Eq. (11a) is a velocity estimator and Eq. (11b) and Eq. (11c) update and filter the disturbance $\hat { \tau } _ { \mathrm { e x t } }$

Thus, the total UAV wrench control command $\tau ^ { * }$ is computed as

$$
\pmb {\tau} ^ {*} = \pmb {\tau} _ {\mathrm{mpc}} + \hat {\pmb {\tau}} _ {\mathrm{ext}}\tag{12}
$$

Similarly, the L1 adaptive controller for the manipulator joint angles is formulated to compensate for dynamic disturbances and model uncertainties:

$$
\mathrm{diag} (\pmb {\beta}) \dot {\pmb {\theta}} + \hat {\pmb {\theta}} = \pmb {\theta} _ {\mathrm{cmd}} + \hat {\pmb {d}} + \pmb {A} _ {d} (\hat {\pmb {\theta}} - \pmb {\theta}),\tag{13a}
$$

$$
\hat {\boldsymbol {d}} ^ {\prime} = - (e ^ {\boldsymbol {A} _ {d} d t} - \boldsymbol {I} _ {4 \times 4}) ^ {- 1} \boldsymbol {A} _ {d} e ^ {\boldsymbol {A} _ {d} d t} (\hat {\boldsymbol {\theta}} - \boldsymbol {\theta}),\tag{13b}
$$

$$
\hat {\boldsymbol {d}} \leftarrow \text { low   pass   filter } (\hat {\boldsymbol {d}}, \hat {\boldsymbol {d}} ^ {\prime}).\tag{13c}
$$

where $\hat { \pmb { \theta } }$ is the estimated joint state, and the disturbance term $\hat { \textbf { \textit { d } } } \in \mathbb { R } ^ { 4 }$ . Thus, the final joint control command $\pmb { \theta } ^ { * }$ incorporating the adaptive disturbance compensation, is given by:

$$
\pmb {\theta} ^ {*} = \pmb {\theta} _ {\mathrm{cmd}} + \hat {\pmb {d}}.\tag{14}
$$

## VII. EE-CENTRIC TELEOPERATION AND POLICYLEARNING

As we mentioned, our framework enables the decoupling between the high-level policy and low-level controller, while the interface between them is only the ee-centric interface. This allows the policy to be embodiment-agnostic, eliminating the need to consider low-level tracking control. In this section, we introduce two aerial manipulation system we developed based on this framework: ee-centric aerial teleoperation system and imitation-learning-based autonomous aerial manipulation system.

## A. EE-Centric Aerial Teleoperation

We developed an aerial manipulation teleoperation system with the ee-centric interface, allowing the operator to focus solely on controlling the target end-effector pose, as if they had complete control of a freely moving hand in 3D space.

Robotic teleoperation requires bidirectional communication between the user and the robot. For the user-to-robot command, we developed a gamepad program so that the user can control the end-effector position ${ \pmb p } _ { E } { } ^ { r }$ and orientation $R _ { E } ^ { W ^ { r } }$ with buttons and joysticks.

For robot-to-user communication, different from tabletop and mobile manipulation settings, in aerial manipulation, the user often lacks direct visual access to the workspace, necessitating reliance on onboard perception systems. In our work, we address this limitation by providing real-time visualization of RGB images captured from cameras mounted on both the end-effector and the base of the UAM. These images are displayed on monitors for continuous user observation. Further enhancing teleoperation efficacy, we have found it crucial to also visualize the user’s inputs directly. To this end, we render the commanded target end-effector pose trajectories in realtime within 3D world frame plots, as illustrated in Fig. 2. This dual approach of visual feedback not only improves spatial awareness but also significantly enhances user performance in teleoperated tasks.

## B. EE-Centric Policy Learning

To establish an autonomous aerial manipulating policy for versatile tasks, we develop an ee-centric policy learning framework based on imitation learning. Specifically, we adopt

Action Chunk with Transformer (ACT) as the network structure [14]. ACT utilizes a Conditional Variational Autoencoder (CVAE) where the encoder compresses action sequences and joint observations into a latent style variable. The transformerbased decoder generates action sequences from the latent variable (only during training and set to be the mean of the prior during testing), current joint observations, and encoded image features. The action chunking mitigates compounding errors and enhances the model’s ability by predicting multiple future actions at once.

In this work, the ACT policy as well as policy observation and action are defined as follow:

$$
\pmb {a} _ {t: t + K} = \pi_ {\varphi} (\pmb {o} _ {t})\tag{15}
$$

$$
\boldsymbol {o} _ {t} = \left\{I _ {E}, I _ {B}, \boldsymbol {p} _ {E}, \boldsymbol {R} _ {E t} ^ {W} \right\}\tag{16}
$$

$$
\boldsymbol {a} _ {t} = \left\{\boldsymbol {p} _ {E} ^ {r}, \boldsymbol {R} _ {E} ^ {W ^ {r}} \right\} _ {t}\tag{17}
$$

where π denotes the ACT policy, $\varphi$ is the network parameter, K is the chunking size, $I _ { B } , \ I _ { E }$ are RGB images from the base camera and the end-effector camera, each with $6 4 0 \times 4 8 0$ resolution. $\pmb { p } _ { E } , \pmb { R } _ { E } ^ { W } , \pmb { p } _ { E } ^ { r } , \pmb { R } _ { E } ^ { W ^ { r } }$ denotes the current and target UAM position and orientation, respectively. We use ResNet-18 as the backbone to encode the RGB images before inputting them into the transformer encoder. The flowchart of the algorithm implementation is illustrated in Fig. 2.

## VIII. EXPERIMENTS

To validate the effectiveness of our proposed framework, we conduct a series of experiments focusing on end-effector trajectory tracking, aerial teleoperation, and policy learning for autonomous aerial manipulation<sup>1</sup>. Our goal is to assess whether the proposed whole-body MPC with L1 adaptation approach enhances trajectory tracking accuracy. Additionally, we evaluate how the ee-centric interface facilitates intuitive and precise teleoperation, reducing operator burden and improving task execution in a series of teleoperation tasks. Finally, we investigate whether the high-quality teleoperation demonstrations can be leveraged to train imitation learningbased policies for autonomous aerial manipulation in both simulation and real-world environments.

## A. Experimental Setup

1) Trajectory Tracking Task Setup: To show the effectiveness of our proposed method in end-effector trajectory tracking tasks, we perform a comparison between our control methods against two baseline approaches:

• w.o. MPC: This baseline disregards both the fullyactuated UAV and arm dynamics, relying solely on inverse kinematics (IK) for motion planning. The UAV follows the desired trajectory generated by the IK planner using a cascade PID controller.

• w.o. L1: This baseline excludes the L1 adaptive component, leaving disturbances from UAV and manipulator interactions and modeling uncertainties uncompensated during control execution.

We conduct experiments with three types of reference trajectories for the end-effector, each lasting 60 seconds. The setpoint trajectory requires the aerial manipulator to keep the end-effector hover at a fixed position $p _ { E } ~ = ~ [ 0 . 0 , 0 . 0 , 1 . 3 ]$ The ellipse trajectory requires to track a sinusoidal trajectory defined as $\pmb { p } _ { E } = [ 0 . 5 \sin ( 0 . 3 t ) , 0 . 0 , 1 . 4 + 0 . 2 \sin ( 0 . 3 t + 0 . 7 5 ) ]$ ]. The figure-8 trajectory requires to track a trajectory $\begin{array} { r l } { { \pmb p } _ { E } } & { { } = } \end{array}$ $[ 0 . 1 + 0 . 6 \sin ( 0 . 3 t ) , 0 . 0 , 1 . 3 5 + 0 . 2 5 \sin ( 0 . 6 t ) ]$ . The maximum velocity in the reference trajectory is about 0.2 m/s. The pitch, row and yaw attitude of the end-effector is fixed at zero during the tracking. Root Mean Square Error (RMSE) is used as the tracking performance evaluation criterion. Each trajectory is repeated three times to compute the mean and standard deviation.

2) Aerial Manipulation Task Setup: We conducted a series of experiments to evaluate the capabilities and applications of our aerial manipulation system. We select different typical tasks from each category we discussed in Sec. II-A, including:

• Aerial Writing: Drawing a target shape (the digit “2025”) on a vertical wall, with an overall size of approximately 3m×0.8m. This task required precise specification and tracking of the end-effector (EE) pose trajectory while maintaining stable contact with the surface.

• Aerial Peg-in-Hole: Inserting a 20mm diameter pole into a 50mm diameter hole positioned around 150cm above the ground.

• Rotate Valve: Manipulating the articulated valve by grasping its handle and rotating it along a 20cm diameter circle, emulating industrial valve manipulation.

• Aerial Pick and Place: Grasping and placing various objects with different shapes and sizes including the screwdriver, pen, tape, and glue bottle.

• Unmount Light Bulb: Grasping a mounted light bulb and unscrewing it out of the socket.

• Mount Light Bulb: A long horizon task that requires a sequence of motion, including inserting a light bulb into a socket, screwing it in, and subsequently turning it on by pressing the button.

For different tasks, different end-effectors are adopted, including the parallel jaw gripper for the pick and place and peg-in-hole task, a passive elastic claw for grasping the light bulb, and a bucket-shaped gripper for rotating the valve.

## B. Implementation Details

The optimal control problem in the ee-centric MPC is implemented using ACADOS [45] with a 25ms discretisation step and a 2.5s constant prediction horizon, running in 100 Hz, and other controller parameters are listed in Table II. The control output is executed in a receding horizon style, where at each iteration, only the first control input $\mathbf { \delta } \mathbf { \em u } _ { 0 }$ is applied to the system.

Since both the L1 adaptive controller and the MPC controller require accurate system modeling $\pmb { f } _ { d y n }$ to achieve effective control performance, we perform system identification to estimate uncertain parameters shown in Table III. We excite the system with two types of motions. First, arm motion-only trajectories are executed while keeping the UAV stationary to calibrate the DH parameters $\zeta$ and joint servo delay $\beta .$ These trajectories ensure that the manipulator kinematic parameters and joint motor dynamics accurately reflect the actual manipulator motion response. Second, UAV free-flight trajectories are conducted to identify the drone dynamics described in Eq. (1).

TABLE II CONTROLLER PARAMETERS

<table><tr><td>Horizon Length T</td><td>2.5 s</td></tr><tr><td>Horizon Steps N</td><td>100</td></tr><tr><td>State Cost  $\mathbf{Q}_{p}$ </td><td>diag(12, 12, 12)</td></tr><tr><td>Rotation Cost  $\mathbf{Q}_{R}$ </td><td>diag(10, 10, 10)</td></tr><tr><td>Velocity Cost  $\mathbf{Q}_{v}$ </td><td>diag(0.1, 0.1, 0.1)</td></tr><tr><td>Joint Angle Cost  $\mathbf{Q}_{\theta}$ </td><td>diag(0.1, 0.1, 0.1)</td></tr><tr><td>Control Cost  $\mathbf{Q}_{u}$ </td><td>diag(0.03, 0.03, 0.03, 0.1, 0.1, 0.1)</td></tr></table>

TABLE III SYSTEM IDENTIFICATION RESULTS

<table><tr><td>Mass Matrix  $M$ </td><td>diag(0.105, 0.121, 0.101, 0.025, 0.011, 0.013)</td></tr><tr><td>Motor Delay  $\beta$ </td><td>(0.66, 0.68, 0.81, 0.85)</td></tr><tr><td>Joint 1 DH Param  $\zeta_1$ </td><td> $d_1 = 0.0$ ,  $a_1 = 0.363$ ,  $\alpha_1 = 0.10$ </td></tr><tr><td>Joint 2 DH Param  $\zeta_2$ </td><td> $d_2 = 0.050$ ,  $a_2 = 0.441$ ,  $\alpha_2 = -0.10$ </td></tr><tr><td>Joint 3 DH Param  $\zeta_3$ </td><td> $d_3 = 0.0$ ,  $a_3 = 0.007$ ,  $\alpha_3 = -1.578$ </td></tr><tr><td>Joint 4 DH Param  $\zeta_4$ </td><td> $d_4 = 0.076$ ,  $a_4 = 0.200$ ,  $\alpha_4 = 0.0$ </td></tr></table>

## C. Experiment I: Trajectory Tracking

Table IV shows the comparison of our approach against w.o. MPC and w.o. L1 baselines in three types of reference end-effector trajectories. The results show that our proposed method achieves the lowest tracking error, with approximately 1 cm in hover and 4 cm during motion. In contrast, the baseline w.o. L1 exhibits 1.3 cm and 6.5 cm, respectively, while the baseline w.o. MPC performs the worst, with 2 cm in hover and 10 cm in motion. The increased tracking error is primarily due to the complex interaction between the UAV and the manipulator, where simultaneous movement introduces dynamic coupling effects that make precise tracking more challenging.

Fig. 4 shows that our method (blue) demonstrates the best tracking performance. The baseline w.o. L1 (green) exhibits overshoot in the X and Z axes and bias in Y, indicating that the L1 controller effectively mitigates both transient and steady-state errors caused by model uncertainties. The baseline w.o. MPC (orange) suffers from significant motion lag, as its inverse kinematics fail to account for UAV dynamics. Fig. 5 depicts the error distribution for all trajectories using the proposed methods and two baselines, showing that our method achieves the smallest and most centered error, whereas the L1 baseline exhibits steady-state errors and the MPC baseline displays a broader error spread due to dynamic lag. These results confirm the effectiveness of our proposed control

TABLE IV  
END-EFFECTOR TRAJECTORY TRACKING PERFORMANCE

<table><tr><td>RMSE (cm)</td><td>Setpoint</td><td>Ellipse</td><td>Figure-8</td></tr><tr><td>Our Method</td><td> $1.00 \pm 0.11$ </td><td> $3.98 \pm 0.41$ </td><td> $4.62 \pm 0.58$ </td></tr><tr><td>w.o. L1 Adaptation</td><td> $1.33 \pm 0.16$ </td><td> $6.67 \pm 0.42$ </td><td> $6.28 \pm 0.50$ </td></tr><tr><td>w.o. MPC</td><td> $2.07 \pm 0.19$ </td><td> $11.25 \pm 0.83$ </td><td> $9.64 \pm 1.07$ </td></tr></table>

![](<images/f11b5a604d658ac340e83cffbc83cc563150512b326101140856234808a37a7e.jpg>)  
Fig. 4. End-effector tracking performance of aerial manipulator in Ellipse trajectory. Tracking results indicate that the w.o. MPC baseline exhibits significant tracking lag, while the w.o. L1 baseline suffers from static tracking errors due to model mismatches.

scheme.

We also investigate the contribution of arm flexibility to EE tracking performance by increasing the arm control cost 5 times larger. Fig. 6a Slow-Arm results show a 35% larger EE tracking error when arm flexibility is restricted, with a more oscillating EE trajectory (Fig. 6c) compared to the MPC with flexible arm (Fig. 6b). This aligns with the intuition that higher arm flexibility improves EE tracking performance, as the arm typically responds faster than the drone base.

Despite the high tracking performance of the proposed methods, Fig. 7 reveals that tracking error increases at lower altitudes (around 1m), likely due to unmodeled ground and wall effect disturbances. Additionally, oscillations in the endeffector trajectory are observed across all methods. We notice servo backlash (around $0 . 5 ^ { \circ }$ dead zone), which results in a 2 cm control dead zone in the end-effector task space, limiting tracking precision during fast UAV maneuvers. Further improvements can be achieved through more accurate system modeling and higher-precision hardware to enhance tracking accuracy.

![](<images/ccd87c20ff0877af5584c930fcf5a99f436d29ae0517b148da0aeefa6c5a380b.jpg>)

Fig. 5. End-effector tracking error distribution for three types of trajectories using our methods and two baselines.  
![](<images/73f4ee3a02ecce6f4a118aaf49c94311b5e351432a0ce30f9593e3cc0a872130.jpg>)  
Fig. 6. Arm flexibility ablation study for MPC controlle

## D. Experiment II: Aerial Teleoperation

First, we demonstrated the effectiveness and versatility of the proposed teleoperation system, by targeting aerial writing, rotating the valve, aerial pick and place, unmount, and mount light bulb tasks. As shown in Fig. 9 and Fig. 10, human teleoperators can easily achieve all aerial manipulation tasks with little learning and operation cost. One key success factor we attribute is the ee-centric interface, which reduces human effort and improves the quality of teleoperation data for future policy learning.

In a subsequent experiment, we evaluated the benefits of directly controlling the end-effector’s pose using our framework against controlling each degree of freedom (DoF) for UAVs and robotic arms [38] in a simulated peg-in-hole task. The teleoperation command trajectories are illustrated in Fig. 8a. Direct control of the end-effector allowed operators to issue more fluid command trajectories, significantly enhancing the precision of end-effector movements and decreasing the time required to complete the task.

## E. Experiment III: Learning from Demonstration

1) Simulation Experiments: We first demonstrate our learning from demonstration framework in Mujoco [46] simulator with four tasks: (i) peg-in-hole, (ii) rotate valve, (iii) pick and place and (iv) open and retrieve, as shown in Fig. 11 (a). To collect demonstrations, we use a scripted policy for each task. Every episode of the scripted policy lasts about 12 seconds for each task. We collect 50 episodes for each task and the control frequency is 50 Hz. Note that with our ee-centric interface, we do not consider any joint configuration when collecting demonstrations, which allows us to efficiently collect smooth demonstrations without tediously adjusting each joint position to complete the task. Our ACT policy for each task in simulation is trained with 100 action chunk size and limited 5k epochs. After training, we choose the policy with the least validation loss to perform 50 evaluation trials. The evaluation result is shown in Table V.

![](<images/c5913f0dfcdba66edfe3dbefb7e02c99e98f2c232433d19621df25c0fbd3f1d0.jpg>)  
Fig. 7. Comparison of Figure-8 and Ellipse trajectory tracking performance across three methods. Our approach achieves the lowest tracking error in dynamic trajectory tracking tasks.

![](<images/4def40ea43fa8f84a591998cc54d785056e35baeafd71a13b16674f26f7f35cc.jpg>)

(b)  
![](<images/137ef7392929e6d4d68dda80ff1446c0dca8764732567dc0a7dcb5be98d5389d.jpg>)  
Fig. 8. (a). End-effector command trajectory using the ee-centric teleoperation interface and full-DoF teleoperation interface, in a simulated peg-in-hole task. (b). End-effector command trajectory of the learned autonomous policy during 50 test trials using ee-centric interface and using full-DoF interface, in a simulated pick and place task.

To showcase the advantage of learning from an ee-centric demonstration compared to a joint space demonstration, we use the same demonstration trajectory but change the observation and action to be in UAM configuration space, i.e., the UAV position and orientation, and each joint angle of the robotic arm. After that, we train a joint space ACT policy with the same training setting as the ee-centric ACT policy, except that the EE pose in the observation and action space is replaced by the drone base pose and full manipulator joint angles. We compare their success rate in evaluation summarized in Table V. It shows that with limited 5k training epochs, eecentric policy outperforms the joint space policy in challenging tasks including pick and place, peg in hole, and open and retrieve. As illustrated in Fig. 8b, the ee-centric policy targets the pickup region and the place point more precisely, while the joint space policy is more prone to generate wrong targets.

![](<images/ccf774455b4168f8d6e9d2d3f24d71ad648c6ef84ab7e0537b83472ef226f3f6.jpg>)  
Fig. 9. Aerial Teleoperation Manipulation Tasks. We target 1) Aerial Writing: UAM with a marker pen writes ’2025’ on a whiteboard. 2) Rotate Valve: UAM grasps the handle and rotates the valve with one loop. 3) Pick-and-Place: UAM grasps an object and relocates it to a designated area.

TABLE V  
IMITATION LEARNING SIMULATION SUCCESS RATE

<table><tr><td></td><td>Rotate Valve</td><td>Pick &amp; Place</td><td>Peg in Hole</td><td>Open &amp; Retrieve</td></tr><tr><td>Joint Space</td><td>50/50</td><td>38/50</td><td>9/50</td><td>8/50</td></tr><tr><td>EE</td><td>50/50</td><td>48/50</td><td>23/50</td><td>17/50</td></tr></table>

Overall, the experimental results reveal several complementary insights:

• Geometric Precision Advantage: Our ee-centric policy achieves 2.5× higher success rate in geometrically sensitive peg in hole task, directly benefiting from taskspace supervision that eliminates the accumulated EE error from the joint space.

• Multi-Skill Composition: In the open and retrieve task, our ee-centric policy achieves 2× higher success rate than joint space policy, which demonstrates its inherent advantages in multi-skill decomposition and execution.

2) Real-world Experiments: We adopt the aerial peg-inhole task to demonstrate our capability to derive an autonomous policy from human demonstrations for aerial manipulation in the real world. The task configurations are illustrated in Fig. 12 and Fig. 13.

We collected 25 episodes of demonstration data via human teleoperation with varying hole horizontal positions, with each episode taking approximately 2 minutes, culminating in a total of around 50 minutes of operational data and about 2 hours of wall-clock time. The data was downsampled to 10 Hz, and the action chunk size was empirically set to 100 during the training process. After training through 100,000 epochs, the policy with the least validation loss is selected. We tested with random unseen horizontal hole positions and the learned policy successfully completed 4 out of 5 real-world peg-inhole tests, i.e., 80% successful rate. The UAM pushed the peg forward to the edge of the hole and didn’t insert it inside successfully. These results underline the potential of learningbased approaches in aerial manipulation under our developed framework, though also highlight the need to develop robust recovery policies, especially for aerial manipulation.

## F. Discussion

The experiments demonstrate our framework in end-effector trajectory tracking, aerial teleoperation, and policy learning for autonomous aerial manipulation. The precise end-effector control framework demonstrated superior end-effector tracking accuracy with minimal error. The high-precision control enables efficient user-friendly aerial teleoperation, allowing human operators to perform multiple complex tasks, which also helps high-quality demonstration data collection. Leveraging the ee-centric framework, advanced high-level policies such as imitation learning can be easily incorporated into the aerial manipulation field, which further expands the field development.

## IX. LIMITATIONS

Although we have demonstrated the proposed framework through various real-world experiments, there are still several limitations due to time constraints and methodological limitations. Firstly, all of our experiments were conducted indoors within a motion capture system, where we achieved millimeter-level state estimation of the UAV and end-effector states using forward kinematics. This setup limits its practical application in real-life scenarios. Secondly, the current safety constraints are model-based, which further restricts practical applications. Incorporating onboard perception to detect obstacles and generate safety constraints in real-time will be our next step, as various studies have demonstrated the feasibility of UAV collision-free flight. Thirdly, the current performance is limited by the robotic arm actuators, which have relatively large backlash and would generate unavoidable vibrations. Finally, although the proposed framework, which decouples different modules, demonstrates the potential for cross-platform compatibility and integration with general manipulation fields, more real-world experiments are planed to take to further validate its effectiveness.

## X. CONCLUSION

This work presents a unified aerial manipulation framework with the ee-centric interface for versatile aerial manipulation tasks. Our system includes a versatile hardware platform that consists of a fully actuated UAV and a 4-DOF manipulator, an ee-centric whole-body MPC to ensure precise end-effector tracking, and an ee-centric high-level policy, where we developed both an intuitive teleoperation and an imitation learningbased autonomous policy. Through extensive real-world experiments, we demonstrated the system’s versatility across various aerial manipulation tasks, including writing, peg-inhole, pick-and-place, valve rotating, and light bulb replacement. More importantly, we demonstrate how this modular and standardized EE-centric framework effectively decouples the high-level policy from the low-level controller, which enables seamless integration of existing standard high-level policy modules from the broader manipulation community, such as teleoperation and imitation learning, into the field of aerial manipulation. The proposed framework achieved high precision, adaptability, and robust performance, making it a significant step toward standardizing aerial manipulation within the broader manipulation field. Future work will extend the framework’s applicability to outdoor environments, incorporate onboard perception for obstacle avoidance, and further improve the end-effector tracking performance.

![](<images/0b6c0e55b034cf8cc48ef395c162fb3632284b1282064cc370043ab5ac351051.jpg>)  
Fig. 10. Long horizon aerial teleoperation light bulb changing task. UAM grasps the light bulb and unscrews it down during the first flight. And it inserts, screws a new light bulb, and presses the button to turn on the light during the second flight.

![](<images/9fcc4185a91543d5ca9e63eb7082a7edb26dc75ca9d8fa02379ff185233300ed.jpg>)

![](<images/e701fe86070aed91d85eaf4740a415347973ab36f1508fbec9da24c72e21819f.jpg>)

![](<images/b73b866d4f04e0ab5909343c1a41267b2cadc8d76a5b4304bf6c23505cdfb1f4.jpg>)

![](<images/127c958e31fade3e5aacd80b8eab300e38649058b408e05a7b6d5ea5b013f4eb.jpg>)  
Fig. 11. Task setup in Mujoco simulation, including (a) Peg-in-Hole; (b) Rotate the Valve; (c) Pick and Place; and (d), a long horizon Open and Retrieve task.

## REFERENCES

[1] A. Suarez, V. M. Vega, M. Fernandez, G. Heredia, and A. Ollero, “Benchmarks for aerial manipulation,” IEEE

Robotics and Automation Letters, vol. 5, no. 2, pp. 2650– 2657, 2020.

[2] X. Guo, G. He, J. Xu, M. Mousaei, J. Geng, S. Scherer, and G. Shi, “Flying calligrapher: Contact-aware motion and force planning and control for aerial manipulation,” IEEE Robotics and Automation Letters, 2024.

[3] D. Mellinger, Q. Lindsey, M. Shomin, and V. Kumar, “Design, modeling, estimation and control for aerial grasping and manipulation,” in 2011 IEEE/RSJ International Conference on Intelligent Robots and Systems. IEEE, 2011, pp. 2668–2673.

[4] J. Ji, T. Yang, C. Xu, and F. Gao, “Real-time trajectory planning for aerial perching,” in 2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2022, pp. 10 516–10 522.

[5] C. Ding, L. Lu, C. Wang, and C. Ding, “Design, sensing, and control of a novel uav platform for aerial drilling and screwing,” IEEE Robotics and Automation Letters, vol. 6, no. 2, pp. 3176–3183, 2021.

[6] M. T. Mason, “Toward robotic manipulation,” Annual Review of Control, Robotics, and Autonomous Systems, vol. 1, no. 1, pp. 1–28, 2018.

[7] O. Khatib, “A unified approach for motion and force control of robot manipulators: The operational space formulation,” IEEE Journal on Robotics and Automation, vol. 3, no. 1, pp. 43–53, 1987.

[8] O. Kroemer, S. Niekum, and G. Konidaris, “A review of robot learning for manipulation: Challenges, representations, and algorithms,” Journal of machine learning research, vol. 22, no. 30, pp. 1–82, 2021.

[9] O. M. Team, D. Ghosh, H. Walke, K. Pertsch, K. Black, O. Mees, S. Dasari, J. Hejna, T. Kreiman, C. Xu et al., “Octo: An open-source generalist robot policy,” arXiv

![](<images/8e9d2d574fb46c7b74a45fb0b9a8b1d37429917afcd00c6e4557c25605a08a00.jpg>)  
Fig. 12. Autonomous aerial manipulation peg-in-hole policy experiment. The UAM inserts the peg precisely, highlighting both the accuracy of the learned policy and the low-level controller.

![](<images/eb55e7c602847c01fd69b55300d6ed64362c4a8ba6cac2a0b113d161ba7a98e6.jpg>)  
Fig. 13. Task scenario randomization for peg-in-hole data collection and test.

preprint arXiv:2405.12213, 2024.

[10] C. Chi, Z. Xu, C. Pan, E. Cousineau, B. Burchfiel, S. Feng, R. Tedrake, and S. Song, “Universal manipulation interface: In-the-wild robot teaching without in-the-wild robots,” 2024. [Online]. Available: https://arxiv.org/abs/2402.10329

[11] S. Song, A. Zeng, J. Lee, and T. Funkhouser, “Grasping in the wild: Learning 6dof closed-loop grasping from low-cost demonstrations,” IEEE Robotics and Automation Letters, vol. 5, no. 3, pp. 4978–4985, 2020.

[12] F. L. Lewis, D. M. Dawson, and C. T. Abdallah, Robot manipulator control: theory and practice. CRC Press, 2003.

[13] J. Yang, C. Glossop, A. Bhorkar, D. Shah, Q. Vuong, C. Finn, D. Sadigh, and S. Levine, “Pushing the limits of cross-embodiment learning for manipulation and navigation,” arXiv preprint arXiv:2402.19432, 2024.

[14] T. Z. Zhao, V. Kumar, S. Levine, and C. Finn, “Learning fine-grained bimanual manipulation with low-cost hard-

ware,” arXiv preprint arXiv:2304.13705, 2023.

[15] A. Ollero, M. Tognon, A. Suarez, D. Lee, and A. Franchi, “Past, present, and future of aerial robotic manipulators,” IEEE Transactions on Robotics, vol. 38, no. 1, pp. 626– 645, 2021.

[16] K. Bodie, M. Brunner, M. Pantic, S. Walser, P. Pfandler,¨ U. Angst, R. Siegwart, and J. Nieto, “An omnidirectional aerial manipulation platform for contact-based inspection,” arXiv preprint arXiv:1905.03502, 2019.

[17] ——, “Active interaction force control for contact-based inspection with a fully actuated aerial vehicle,” IEEE Transactions on Robotics, vol. 37, no. 3, pp. 709–722, 2020.

[18] X. Guo, G. He, M. Mousaei, J. Geng, G. Shi, and S. Scherer, “Aerial interaction with tactile sensing,” in 2024 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2024, pp. 1576–1582.

[19] C. Lanegger, M. Ruggia, M. Tognon, L. Ott, and R. Siegwart, “Aerial layouting: Design and control of a compliant and actuated end-effector for precise in-flight marking on ceilings,” Proceedings of Robotics: Science and System XVIII, p. p073, 2022.

[20] D. Tzoumanikas, F. Graule, Q. Yan, D. Shah, M. Popovic, and S. Leutenegger, “Aerial manipulation using hybrid force and position nmpc applied to aerial writing,” arXiv preprint arXiv:2006.02116, 2020.

[21] G. He, Y. Jangir, J. Geng, M. Mousaei, D. Bai, and S. Scherer, “Image-based visual servo control for

aerial manipulation using a fully-actuated uav,” in 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2023, pp. 5042–5049.

[22] M. Brunner, L. Giacomini, R. Siegwart, and M. Tognon, “Energy tank-based policies for robust aerial physical interaction with moving objects,” in 2022 International Conference on Robotics and Automation (ICRA). IEEE, 2022, pp. 2054–2060.

[23] J. Meng, J. Buzzatto, Y. Liu, and M. Liarokapis, “On aerial robots with grasping and perching capabilities: A comprehensive review,” Frontiers in Robotics and AI, vol. 8, p. 739173, 2022.

[24] W. R. Roderick, M. R. Cutkosky, and D. Lentink, “Birdinspired dynamic grasping and perching in arboreal environments,” Science Robotics, vol. 6, no. 61, p. eabj7562, 2021.

[25] J. Fishman, S. Ubellacker, N. Hughes, and L. Carlone, “Dynamic grasping with a” soft” drone: From theory to practice,” in 2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2021, pp. 4214–4221.

[26] S. Ubellacker, A. Ray, J. M. Bern, J. Strader, and L. Carlone, “High-speed aerial grasping using a soft drone with onboard perception,” npj Robotics, vol. 2, no. 1, p. 5, 2024. [Online]. Available: https://doi.org/10.1038/s44182-024-00012-1

[27] M. Schuster, D. Bernstein, P. Reck, S. Hamaza, and M. Beitelschmidt, “Automated aerial screwing with a fully actuated aerial manipulator,” in 2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2022, pp. 3340–3347.

[28] M. Wang, Z. Chen, K. Guo, X. Yu, Y. Zhang, L. Guo, and W. Wang, “Millimeter-level pick and peg-in-hole task achieved by aerial manipulator,” IEEE Transactions on Robotics, 2023.

[29] Y. Su, J. Li, Z. Jiao, M. Wang, C. Chu, H. Li, Y. Zhu, and H. Liu, “Sequential manipulation planning for over-actuated unmanned aerial manipulators,” in 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2023, pp. 6905–6911.

[30] M. Brunner, G. Rizzi, M. Studiger, R. Siegwart, and M. Tognon, “A planning-and-control framework for aerial manipulation of articulated objects,” IEEE Robotics and Automation Letters, vol. 7, no. 4, pp. 10 689–10 696, 2022.

[31] F. Baberg, Y. Wang, S. Caccamo, and P. ˚ Ogren, “Adap-<sup>¨</sup> tive object centered teleoperation control of a mobile manipulator,” in 2016 IEEE International Conference on Robotics and Automation (ICRA), 2016, pp. 455–461.

[32] H. Ha, Y. Gao, Z. Fu, J. Tan, and S. Song, “Umi on legs: Making manipulation policies mobile with manipulation-centric whole-body controllers,” 2024. [Online]. Available: https://arxiv.org/abs/2407.10353

[33] D. Honerkamp, T. Welschehold, and A. Valada, “N<sup>2</sup>m<sup>2</sup>: Learning navigation for arbitrary mobile manipulation motions in unseen and dynamic environments,” IEEE

Transactions on Robotics, 2023.

[34] ——, “Learning kinematic feasibility for mobile manipulation through deep reinforcement learning,” IEEE Robotics and Automation Letters (RA-L), 2021.

[35] R. Yang, Y. Kim, R. Hendrix, A. Kembhavi, X. Wang, and K. Ehsani, “Harmonic mobile manipulation,” 2024. [Online]. Available: https://arxiv.org/abs/2312.06639

[36] K. Bodie, M. Tognon, and R. Siegwart, “Dynamic end effector tracking with an omnidirectional parallel aerial manipulator,” IEEE Robotics and Automation Letters, vol. 6, no. 4, pp. 8165–8172, 2021.

[37] M. Allenspach, N. Lawrance, M. Tognon, and R. Siegwart, “Towards 6dof bilateral teleoperation of an omnidirectional aerial vehicle for aerial physical interaction,” in 2022 International Conference on Robotics and Automation (ICRA). IEEE, 2022, pp. 9302–9308.

[38] G. A. Yashin, D. Trinitatova, R. T. Agishev, R. Ibrahimov, and D. Tsetserukou, “Aerovr: Virtual reality-based teleoperation with tactile feedback for aerial manipulation,” in 2019 19th International Conference on Advanced Robotics (ICAR). IEEE, 2019, pp. 767–772.

[39] A. Coelho, Y. Sarkisov, X. Wu, H. Mishra, H. Singh, A. Dietrich, A. Franchi, K. Kondak, and C. Ott, “Wholebody teleoperation and shared control of redundant robots with applications to aerial manipulation,” Journal of Intelligent & Robotic Systems, vol. 102, pp. 1–22, 2021.

[40] Z. Fu, T. Z. Zhao, and C. Finn, “Mobile aloha: Learning bimanual mobile manipulation with low-cost whole-body teleoperation,” 2024. [Online]. Available: https://arxiv.org/abs/2401.02117

[41] C. Chi, Z. Xu, S. Feng, E. Cousineau, Y. Du, B. Burchfiel, R. Tedrake, and S. Song, “Diffusion policy: Visuomotor policy learning via action diffusion,” The International Journal of Robotics Research, p. 02783649241273668, 2023.

[42] T. Z. Zhao, V. Kumar, S. Levine, and C. Finn, “Learning fine-grained bimanual manipulation with low-cost hardware,” 2023. [Online]. Available: https: //arxiv.org/abs/2304.13705

[43] J. Denavit and R. S. Hartenberg, “A kinematic notation for lower-pair mechanisms based on matrices,” 1955.

[44] Z. Wu, S. Cheng, K. A. Ackerman, A. Gahlawat, A. Lakshmanan, P. Zhao, and N. Hovakimyan, “L1adaptive augmentation for geometric tracking control of quadrotors,” in 2022 International Conference on Robotics and Automation (ICRA), 2022, pp. 1329–1336.

[45] R. Verschueren, G. Frison, D. Kouzoupis, J. Frey, N. van Duijkeren, A. Zanelli, B. Novoselnik, T. Albin, R. Quirynen, and M. Diehl, “acados – a modular open-source framework for fast embedded optimal control,” Mathematical Programming Computation, 2021.

[46] E. Todorov, T. Erez, and Y. Tassa, “Mujoco: A physics engine for model-based control,” in 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems, 2012, pp. 5026–5033.