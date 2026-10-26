# Aerial Manipulation in the Wild with Onboard Perception, Policy Learning, and Whole-Body Control

Yuanzhu Zhan<sup>∗1</sup>, Yufei Jiang<sup>∗1</sup>, Zemu Zhang<sup>1</sup>, Junyi Geng<sup>1</sup>

Abstract— Aerial manipulation in outdoor environments remains challenging due to the simultaneous requirements of reliable state estimation, stable aerial motion, and precise manipulation under external disturbances. In this work, we present a real-world outdoor aerial manipulation framework that integrates imitation learning, onboard LiDAR-inertial state estimation, and whole-body model predictive control. A Diffusion Policy is trained from manipulation demonstrations to generate desired end-effector motions from onboard observations. These learned commands are executed by a whole-body MPC that jointly coordinates the aerial platform and manipulator to realize the desired end-effector trajectory. To eliminate reliance on external motion-capture infrastructure, the platform employs onboard LiDAR-inertial odometry for state estimation during outdoor operation. We validate the complete framework on a physical aerial manipulator and demonstrate successful execution of outdoor manipulation tasks. The experimental results show that demonstration-driven manipulation policies can be effectively integrated with onboard state estimation and model-based whole-body control to enable aerial manipulation beyond controlled indoor environments.

## I. INTRODUCTION

Traditional unmanned aerial vehicles (UAVs) have demonstrated strong capabilities in autonomous navigation, sensing, inspection, and data collection in environments that are difficult or unsafe for humans to access [1]. Aerial manipulation extends these capabilities from passive observation to physical interaction by equipping UAVs with robotic manipulators or specialized end effectors [2], [3], [4]. However, manipulation introduces additional challenges because arm motion affects the floating aerial base, while the vehicle must simultaneously maintain stable flight and accurately control the end effector [5], [6]. These challenges are further amplified outdoors, where high-precision state estimation cannot rely on laboratory motion-capture infrastructure and environmental conditions are less controlled.

Imitation learning provides an alternative to manually designing task-specific manipulation behaviors. Visuomotor policies can learn directly from demonstrations [7], [8], while Diffusion Policy has shown strong performance in generating temporally coherent manipulation motions [9]. Recent works such as UMI-on-Legs, UMI-on-Air, and Flying Hand have demonstrated the potential of demonstration-driven policies for mobile and aerial manipulation [10], [11], [12]. However, these learning-based aerial manipulation systems have largely been evaluated in controlled indoor environments.

Compared with controlled indoor settings, outdoor deployment introduces additional challenges including less structured visual conditions, stronger environmental disturbances, and the absence of laboratory-grade localization infrastructure. Existing systems typically address these challenges through carefully designed task-specific perception and control modules [13], [14], [15], but such pipelines do not directly address whether manipulation behaviors learned from demonstrations can transfer across environments.

Motivated by this gap, we investigate whether a visuomotor policy trained solely from indoor demonstrations can be deployed for outdoor aerial manipulation without retraining. We develop and experimentally evaluate a framework that integrates Diffusion Policy, LiDAR-inertial state estimation, and whole-body MPC for real-world outdoor aerial manipulation. We validate the complete system on a physical aerial manipulator and demonstrate successful outdoor manipulation using a policy trained entirely from indoor demonstrations.

The main contributions of this work are:

• An integrated framework combining diffusion-policybased imitation learning, onboard LiDAR-inertial state estimation, and whole-body MPC for outdoor aerial manipulation.

• A whole-body MPC formulation for executing learned end-effector motions through coordinated aerial-base and manipulator motion.

• Real-world outdoor validation without external motion capture, outdoor demonstrations, or policy retraining.

## II. METHODOLOGY

## A. System Overview

Our aerial manipulation system consists of a fully actuated hexarotor, a 4-DoF serial manipulator, and an end-effector equipped with an OAK-1W camera, as shown in Fig. 1. During deployment, the visuomotor policy generates endeffector commands, while onboard LiDAR–inertial odometry provides floating-base state feedback and a whole-body controller coordinates the aerial base and manipulator.

For policy training, demonstrations are collected offline using a handheld UMI device [11]. This shared end-effectorcentric interface facilitates transferring the learned policy from handheld demonstrations to the aerial manipulator.

## B. Diffusion Policy for End-Effector Action Generation

Diffusion Policy formulates visuomotor control as conditional generative modeling of future action trajectories. During training, noise is progressively added to demonstrated action sequences, and a conditional denoising network learns to recover the original actions given the observation history. During inference, the policy starts from a noisy action sequence and iteratively denoises it to generate a future trajectory that is consistent with the current observations. Following DDIM [16], the denoising process can be written as

![](<images/2043853b4f792e46778b5ff4d4a4eefc0ddbf400a1a2a57c7d1bbf6c151b72b5.jpg>)

(a) AM platform  
![](<images/7ae5ef984bd63ef90324e9d3aa00c1fdea14810b53ef2b881b28d72c02009032.jpg>)  
(b) UMI handheld gripper  
Fig. 1: System Overview

$$
\mathbf {a} ^ {k - 1} = \mathbf {a} ^ {k} + \psi_ {k} (\pi_ {\theta} (\mathbf {a} ^ {k}, t | \mathcal {O})),\tag{1}
$$

where $\mathbf { a } ^ { k }$ denotes the noisy action trajectory at diffusion step $k , ~ \pi _ { \theta }$ is the trained denoiser, O is the conditioning observation, and $\psi _ { k }$ is the DDIM update function for step k.

For aerial manipulation, we adopt the end-effector-centric representation used in [17], [11]. At each inference step, the policy receives a temporal observation window containing egocentric RGB images I, relative end-effector poses $\mathbf { T } ^ { \mathrm { e e } } ,$ and gripper widths w, so the observation O is defined as:

$$
\mathcal {O} = \{\mathbf {I} _ {t}, \mathbf {T} _ {t} ^ {\mathrm{ee}}, w _ {t} \} _ {t = 1} ^ {L},\tag{2}
$$

where L denotes the observation horizon. The policy predicts a horizon of future end-effector actions represented by relative end-effector positions $\mathbf { p } ^ { \mathrm { e e } }$ , orientations $\mathbf { R } ^ { \mathrm { e e } }$ , and gripperwidth commands w,

$$
\mathbf {a} = \{\mathbf {p} _ {t} ^ {\mathrm{ee}}, \mathbf {R} _ {t} ^ {\mathrm{ee}}, w _ {t} \} _ {t = 1} ^ {N},\tag{3}
$$

where N denotes the planning horizon.

Because the policy operates in end-effector space rather than directly commanding the aerial base, manipulator joints, or actuators, the end-effector-centric architecture decouples the learned policy from the platform-specific kinematics and dynamics, which are handled by the downstream whole-body controller. The relative pose representation further reduces the policy’s dependence on precise global localization.

Human demonstrations are collected using the handheld UMI device described in Sec. II-A, whose camera–gripper configuration closely matches that of the aerial end-effector. The collected demonstrations are converted into the observation–action sequences defined in (2-3) and used to train the conditional UNet-based [18] Diffusion Policy.

## C. Whole-Body Model Predictive Control

To execute the end-effector trajectories generated by the learned policy, we employ a whole-body model predictive controller that jointly coordinates the aerial platform and manipulator. The state and control vectors are defined as

$$
\begin{array}{l} \mathbf {x} = \left[ \begin{array}{c c c c c} \mathbf {p} ^ {\top} & \mathbf {v} ^ {\top} & \boldsymbol {\eta} ^ {\top} & \boldsymbol {\omega} ^ {\top} & \mathbf {q} ^ {\top} \end{array} \right] ^ {\top}, \\ \mathbf {u} = \left[ \begin{array}{c c c} \mathbf {f} ^ {\top} & \boldsymbol {\tau} ^ {\top} & \mathbf {q} _ {\mathrm{ref}} ^ {\top} \end{array} \right] ^ {\top}, \end{array}\tag{4}
$$

where $\mathbf { p } , \mathbf { v } \ \in \ \mathbb { R } ^ { 3 }$ are the aerial-base position and linear velocity, η, $\boldsymbol { \omega } \in \mathbb { R } ^ { 3 }$ are the base Euler angles and angular velocity, and $ { \mathbf { q } } \in \mathbb { R } ^ { 4 }$ denotes the manipulator joint configuration. The control inputs $\mathbf { f } , \tau \in \mathbb { R } ^ { 3 }$ are the commanded force and torque, and $\mathbf { q } _ { \mathrm { r e f } } \in \mathbb { R } ^ { 4 }$ is the desired joint configuration.

At each control cycle, the MPC solves the finite-horizon nonlinear optimal control problem

$$
\begin{array}{r l} \underset {\mathbf {x} _ {0: N}, \mathbf {u} _ {0: N - 1}} {\min} & \sum_ {k = 0} ^ {N - 1} \ell_ {k} (\mathbf {x} _ {k}, \mathbf {u} _ {k}) + \ell_ {N} (\mathbf {x} _ {N}) \\ \text {s.t.} & \mathbf {x} _ {k + 1} = f (\mathbf {x} _ {k}, \mathbf {u} _ {k}), \\ & \mathbf {x} _ {\min} \leq \mathbf {x} _ {k} \leq \mathbf {x} _ {\max}, \\ & \mathbf {u} _ {\min} \leq \mathbf {u} _ {k} \leq \mathbf {u} _ {\max}, \\ & \mathbf {x} _ {0} = \mathbf {x} _ {\text {meas}}. \end{array}\tag{5}
$$

The state-transition function $f ( \mathbf { x } _ { k } , \mathbf { u } _ { k } )$ is obtained by discretizing the following lightweight whole-body dynamics model:

$$
\begin{array}{r l} & {\dot {\mathbf {p}} = \mathbf {v},} \\ & {\dot {\mathbf {v}} = \frac {1}{m} \mathbf {f},} \\ & {\dot {\boldsymbol {\eta}} \approx \boldsymbol {\omega},} \\ & {\dot {\boldsymbol {\omega}} = \mathbf {I} ^ {- 1} \boldsymbol {\tau},} \\ & {\dot {\mathbf {q}} = k _ {q} (\mathbf {q} _ {\mathrm{ref}} - \mathbf {q}),} \end{array}\tag{6}
$$

where m is the total mass of the aerial manipulator, ${ \textbf { I } } =$ diag $\left( I _ { x } , I _ { y } , I _ { z } \right)$ is the rotational inertia, and $k _ { q }$ approximates the closed-loop response of the manipulator joint controller. A small-angle approximation, $\dot { \eta } \approx \omega ,$ , is used for the base attitude dynamics.

The stage cost in (5) is defined as

$$
\begin{array}{l} \ell_ {k} = \| \mathbf {p} _ {E, k} - \mathbf {p} _ {E, k} ^ {d} \| _ {\mathbf {Q} _ {p}} ^ {2} + \| \boldsymbol {\eta} _ {E, k} - \boldsymbol {\eta} _ {E, k} ^ {d} \| _ {\mathbf {Q} _ {\eta}} ^ {2} \\ \qquad + \| \mathbf {v} _ {k} \| _ {\mathbf {Q} _ {v}} ^ {2} + \| \boldsymbol {\eta} _ {k} - \boldsymbol {\eta} _ {B, k} ^ {d} \| _ {\mathbf {Q} _ {B}} ^ {2} + \| \boldsymbol {\omega} _ {k} \| _ {\mathbf {Q} _ {\omega}} ^ {2} \\ \qquad + \| \mathbf {q} _ {k} - \mathbf {q} _ {0} \| _ {\mathbf {Q} _ {q}} ^ {2} + \| \mathbf {u} _ {k} - \mathbf {u} _ {0} \| _ {\mathbf {R}} ^ {2} + \| \mathbf {u} _ {k} - \mathbf {u} _ {k - 1} \| _ {\mathbf {R} _ {\Delta}} ^ {2}. \end{array}\tag{7}
$$

Here, $\mathbf { p } _ { E , k }$ and $\scriptstyle \eta _ { E , k }$ denote the end-effector position and orientation, while $\mathbf { p } _ { E , k } ^ { d }$ and $\eta _ { E , k } ^ { d }$ are the corresponding references generated by the learned policy. The first two terms penalize end-effector tracking errors. The remaining state terms regularize base velocity, attitude, angular velocity, and manipulator configuration. The nominal arm configuration q encourages the manipulator to remain near a preferred posture, while the control and input-smoothness terms penalize excessive control effort and abrupt command changes. The desired base orientation $\eta _ { B , k } ^ { d }$ maintains a near-level vehicle attitude while allowing the yaw direction to follow the end-effector reference. The terminal cost $\ell _ { N }$ consists of the end-effector tracking and state-regularization terms evaluated at the final prediction step, excluding the controleffort and input-smoothness penalties.

The end-effector states appearing in (7) are obtained from the aerial-base state and manipulator configuration through the whole-body forward kinematics,

$$
{ } ^ { W } \mathbf { T } _ { E , k } = \mathcal { F } ( \mathbf { p } _ { k } , \boldsymbol { \eta } _ { k } , \mathbf { q } _ { k } )   ,\tag{8}
$$

where $W _ { \mathbf { T } _ { E , k } }$ denotes the end-effector pose in the world frame, from which $\mathbf { p } _ { E , k }$ and η are extracted.

The nonlinear optimal control problem is implemented in acados using a nonlinear least-squares objective and solved with sequential quadratic programming real-time iteration (SQP-RTI). At each control step, only the first optimized control input is applied in a receding-horizon manner. This formulation enables coordinated motion of the aerial base and manipulator for tracking the policy-generated endeffector trajectory.

## III. EXPERIMENTS

We evaluate the proposed system to answer two questions: (1) whether onboard LIO provides sufficiently reliable state feedback for outdoor aerial manipulation, and (2) whether a policy trained exclusively from indoor demonstrations can transfer to an unseen outdoor environment without retraining.

## A. Experimental Setup

We evaluate the system on an aerial peg-in-hole task in which the vehicle must insert a 20 mm diameter stick into a 50 mm diameter hole positioned approximately 130 cm above the ground. The visuomotor policy is trained entirely from 80 indoor demonstrations collected with the handheld UMI device. During demonstration collection, we apply domain randomization by varying the initial gripper position and the target hole location on the wall, exposing the policy to diverse approach and insertion configurations. The resulting policy is used unchanged for both indoor and outdoor evaluation. During outdoor deployment, the aerial platform relies on FAST-LIO2 [19] for onboard LiDARinertial state estimation of the floating base without external motion-capture infrastructure. No outdoor demonstrations, policy fine-tuning, or environment-specific adaptation are used.

A trial is considered successful when the stick is inserted into the target hole without human intervention after policy execution begins. We conduct 5 trails indoors and 3 trails in an unseen outdoor grass field.

TABLE I: End-effector tracking and base attitude performance in indoor and outdoor experiments.

<table><tr><td>Environment</td><td>Norm. EE Error</td><td>Max. Base Tilt [deg]</td></tr><tr><td>Indoor</td><td>0.102</td><td>3.32</td></tr><tr><td>Outdoor</td><td>0.104</td><td>1.84</td></tr></table>

Several metrics are defined to quantify the controller’s performance:

• Normalized end-effector position tracking error (Norm. EE Error)

$$
\rho_ {\mathrm{ee}} = \frac {1}{N _ {t}} \sum_ {t = 1} ^ {N _ {t}} \frac {\left\| \mathbf {p} _ {t} ^ {e e} - \mathbf {p} _ {t} ^ {e e , \mathrm{des}} \right\| _ {2}}{L _ {\mathrm{arm}}}.
$$

• Maximum base tilt: This metric quantifies the largest roll–pitch attitude excursion during task execution. Let $\eta _ { \mathrm { r o l l } } ( t )$ and $\eta _ { \mathrm { p i t c h } } ( t )$ denote the base roll and pitch angles at timestep t. We define the maximum roll-pitch tilt magnitude over the rollout:

$$
U _ {\mathrm{tilt}} = \max _ {t \in \{1, \dots , N _ {t} \}} \sqrt {\eta_ {\mathrm{roll}} (t) ^ {2} + \eta_ {\mathrm{pitch}} (t) ^ {2}}.
$$

## B. Indoor and Outdoor Results

1) Indoor Experiments: We first evaluate the learned policy in the indoor environment. The system successfully completes all five peg-in-hole trials despite variations in the initial gripper configuration and target hole location. A representative trial is shown in Fig. 2. The whole-body MPC achieves a normalized end-effector tracking error of 0.102 while maintaining small vehicle tilt.

Fig. 4 shows representative end-effector tracking performance under whole-body MPC. The controller coordinates the aerial base and manipulator to track the policy-generated trajectory while maintaining stable vehicle motion. These indoor experiments establish the baseline performance of the learned policy before outdoor evaluation.

2) Outdoor Experiments: We next evaluate the same learned policy in an unseen outdoor environment to assess its ability to generalize beyond the indoor demonstration setting. During outdoor deployment, the aerial platform relies entirely on onboard LIO for floating-base state estimation, while the learned Diffusion Policy and low-level controller remain unchanged.

We conduct three outdoor peg-in-hole trials, and the system successfully completes all three trials without human intervention after policy execution begins. A representative trial is shown in Fig. 3. Starting from an initial configuration different from those observed during data collection, the policy guides the aerial manipulator toward the target and generates the final insertion motion required to place the stick into the hole.

Table I shows comparable normalized end-effector tracking errors for indoor and outdoor experiments. Despite replacing motion-capture system with onboard LIO during outdoor deployment, the whole-body controller maintains similar trajectory-tracking performance. The maximum base tilt also remains below 4<sup>◦</sup> in both environments, indicating that the manipulation trajectory is executed without excessive vehicle attitude excursion.

![](<images/2b2c4ba9bbb188a0b13bea727075836ab30579db3d1cc70619796ebd016e9188.jpg>)  
(a) Policy inference initialization

![](<images/eb8554fa702509fb032592f3ae80d1ecdfee07d4dceb766961010bdb7875baef.jpg>)  
(b) Approach to the target

![](<images/e82400157343a76f54561d05b86efbc07e0a40d385d4e676661ee43a9f06b890.jpg>)  
(c) Peg insertion

Fig. 2: Indoor peg-in-hole execution in the training environment, showing the initial configuration, target approach, and successful insertion. Insets show the onboard first person view used by the visuomotor policy.  
![](<images/714bc94f97f933cefbd38a6b84fd15a03dfe38d1c8a0075e2b751693286b1c42.jpg>)  
(a) Policy inference initialization

![](<images/e9044ec21181deb6466bcf34e96d5b3816852e3bad3060fd8cdf832eef068e8e.jpg>)  
(b) Approach to the target

![](<images/791d938849d27cf7346e025492898016543cedfbf22d2152a3bda92dacda5366.jpg>)  
(c) Peg insertion

Fig. 3: Outdoor peg-in-hole execution in an unseen environment using the same learned diffusion policy, with onboard LiDAR-inertial odometry for state estimation. Insets show the onboard first person view used by the visuomotor policy.  
![](<images/766a87509c97262de7533cd3f475d8496d95a43852d1b6255b473fcd811e45ea.jpg>)  
(a) Indoor

End Effector Position  
![](<images/6b600fee754be87251a35ead85e3836f27116c7cf34e7501a2d06b0263ca5918.jpg>)  
(b) Outdoor  
Fig. 4: End-effector position tracking performance in both inddor and outdoor environments. Solid lines denote the measured end-effector position, while dashed lines denote the policy-generated reference trajectory.

The outdoor experiments show that a visuomotor policy trained solely from indoor demonstrations can generalize to a visually and geometrically different environment without retraining, while onboard LiDAR–inertial odometry provides sufficiently stable state feedback for execution outside a motion-capture environment. Together, these results demonstrate the feasibility of transferring demonstration-driven aerial manipulation from controlled indoor settings to outdoor deployment.

## IV. CONCLUSION

This work presented an outdoor aerial manipulation framework integrating Diffusion Policy, onboard LiDAR–inertial state estimation, and whole-body MPC. Experiments on a physical aerial manipulator demonstrated that a policy trained entirely from indoor demonstrations can be deployed in an unseen outdoor environment without external motion capture, outdoor demonstrations, or policy retraining. The results show that combining transferable visuomotor policies with onboard localization and coordinated whole-body control provides a practical pathway toward aerial manipulation beyond controlled laboratory environments. Future work will evaluate the framework on more diverse manipulation tasks and challenging outdoor conditions.

## ACKNOWLEDGMENT

This work was supported in part by the National Science Foundation under Awards No. 2541976. Any opinions, findings, and conclusions or recommendations expressed in this paper are those of the author and do not necessarily reflect the views of the National Science Foundation.

## REFERENCES

[1] Y. Jiang, Y. Zhan, H. V. Gupta, C. Borde, and J. Geng, “A selfsupervised learning approach with differentiable optimization for uav trajectory planning,” in 2026 IEEE International Conference on Robotics and Automation (ICRA), 2026.

[2] F. Ruggiero, V. Lippiello, and A. Ollero, “Aerial manipulation: A literature review,” IEEE Robotics and Automation Letters, vol. 3, no. 3, pp. 1957–1964, 2018.

[3] A. Ollero, M. Tognon, A. Suarez, D. Lee, and A. Franchi, “Past, present, and future of aerial robotic manipulators,” IEEE Transactions on Robotics, vol. 38, no. 1, pp. 626–645, 2022.

[4] Y. Zhan, Y. Jiang, M. Cao, and J. Geng, “Aerial manipulation with contact-aware onboard perception and hybrid control,” in 2026 IEEE International Conference on Robotics and Automation (ICRA), 2026.

[5] H. Yang and D. Lee, “Dynamics and control of quadrotor with robotic manipulator,” in 2014 IEEE International Conference on Robotics and Automation, 2014, pp. 5544–5549.

[6] S. Kim, H. Seo, J. Shin, and H. J. Kim, “Cooperative aerial manipulation using multirotors with multi-DOF robotic arms,” IEEE/ASME Transactions on Mechatronics, vol. 23, no. 2, pp. 702–713, 2018.

[7] S. Levine, C. Finn, T. Darrell, and P. Abbeel, “End-to-end training of deep visuomotor policies,” Journal of Machine Learning Research, vol. 17, no. 39, pp. 1–40, 2016.

[8] T. Z. Zhao, V. Kumar, S. Levine, and C. Finn, “Learning fine-grained bimanual manipulation with low-cost hardware,” in Proceedings of Robotics: Science and Systems, 2023.

[9] C. Chi, S. Feng, Y. Du, Z. Xu, E. Cousineau, B. C. M. Burchfiel, and S. Song, “Diffusion policy: Visuomotor policy learning via action diffusion,” in Proceedings of Robotics: Science and Systems, 2023.

[10] H. Ha, Y. Gao, Z. Fu, J. Tan, and S. Song, “UMI-on-legs: Making manipulation policies mobile with manipulation-centric whole-body controllers,” in Proceedings of the 8th Conference on Robot Learning, ser. Proceedings of Machine Learning Research, vol. 270. PMLR, 2025, pp. 5254–5270.

[11] H. Gupta, X. Guo, H. Ha, C. Pan, M. Cao, D. Lee, S. Scherer, S. Song, and G. Shi, “UMI-on-air: Embodiment-aware guidance for embodiment-agnostic visuomotor policies,” arXiv preprint arXiv:2510.02614, 2025.

[12] G. He, X. Guo, L. Tang, Y. Zhang, M. Mousaei, J. Xu, J. Geng, S. Scherer, and G. Shi, “Flying hand: End-effector-centric framework for versatile aerial manipulation teleoperation and policy learning,” in Proceedings of Robotics: Science and Systems, 2025.

[13] P. Ramon-Soria, B. C. Arrue, and A. Ollero, “Grasp planning and ´ visual servoing for an outdoors aerial dual manipulator,” Engineering, vol. 6, no. 1, pp. 77–88, 2020.

[14] S. Ubellacker, A. Ray, J. M. Bern, J. Strader, and L. Carlone, “Highspeed aerial grasping using a soft drone with onboard perception,” npj Robotics, vol. 2, p. 5, 2024.

[15] E. Bauer, M. Blochlinger, P. Strauch, A. Raayatsanati, C. Curdin,¨ and R. K. Katzschmann, “An open-source soft robotic platform for autonomous aerial manipulation in the wild,” in Proceedings of the 8th Conference on Robot Learning, ser. Proceedings of Machine Learning Research, vol. 270, 2025, pp. 3094–3106.

[16] J. Song, C. Meng, and S. Ermon, “Denoising diffusion implicit models,” in International Conference on Learning Representations, 2021. [Online]. Available: https://openreview.net/forum?id=St1giarCHLP

[17] C. Chi, Z. Xu, C. Pan, E. Cousineau, B. Burchfiel, S. Feng, R. Tedrake, and S. Song, “Universal Manipulation Interface: In-The-Wild Robot Teaching Without In-The-Wild Robots,” in Proceedings of Robotics: Science and Systems, Delft, Netherlands, July 2024.

[18] O. Ronneberger, P. Fischer, and T. Brox, “U-net: Convolutional networks for biomedical image segmentation,” in Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015, N. Navab, J. Hornegger, W. M. Wells, and A. F. Frangi, Eds. Cham: Springer International Publishing, 2015, pp. 234–241.

[19] W. Xu, Y. Cai, D. He, J. Lin, and F. Zhang, “Fast-lio2: Fast direct lidarinertial odometry,” IEEE Transactions on Robotics, vol. 38, no. 4, pp. 2053–2073, 2022.