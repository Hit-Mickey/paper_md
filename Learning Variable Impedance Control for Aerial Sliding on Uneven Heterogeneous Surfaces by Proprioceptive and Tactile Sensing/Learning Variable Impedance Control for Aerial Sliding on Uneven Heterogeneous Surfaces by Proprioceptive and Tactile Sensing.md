# Learning Variable Impedance Control for Aerial Sliding on Uneven Heterogeneous Surfaces by Proprioceptive and Tactile Sensing

Weixuan Zhang, Lionel Ott, Marco Tognon, and Roland Siegwart

Abstract— The recent development of novel aerial vehicles capable of physically interacting with the environment leads to new applications such as contact-based inspection. These tasks require the robotic system to exchange forces with partially-known environments, which may contain uncertainties including unknown spatially-varying friction properties and discontinuous variations of the surface geometry. Finding a control strategy that is robust against these environmental uncertainties remains an open challenge. This paper presents a learning-based adaptive control strategy for aerial sliding tasks. In particular, the gains of a standard impedance controller are adjusted in real-time by a policy based on the current control signals, proprioceptive measurements, and tactile sensing. This policy is trained in simulation with simplified actuator dynamics in a student-teacher learning setup. The real-world performance of the proposed approach is verified using a tiltarm omnidirectional flying vehicle. The proposed controller structure combines data-driven and model-based control methods, enabling our approach to successfully transfer directly and without adaptation from simulation to the real platform. Compared to fine-tuned state of the art interaction control methods we achieve reduced tracking error and improved disturbance rejection.

## I. INTRODUCTION

Aerial interaction tasks such as contact-based inspections [1]–[3] require the flying vehicle to slide a sensor along the surface to be inspected and maintain contact. The state of the art proposes several solutions for the related general push-and-slide problem [4]. Most existing approaches show interactions with simple continuous surfaces (e.g., planes, cylinders, etc), where the geometry is assumed known and friction properties are assumed to be identical everywhere (i.e., homogeneous) [5]–[7].

However, in real applications, interaction surfaces are often discontinuous and inhomogeneous, and their model knowledge is only partial due to perception inaccuracy. Firstly, the surface friction property cannot be directly measured and may change discontinuously if the surface consists of different materials (i.e., heterogeneous surface). The surface geometry estimated with visual sensors contains inaccuracies on the order of 1 cm and a few degrees in orientation. Secondly, from the control perspective, the presence of these unexpected environment features introduces discontinuities in contact forces. The induced discontinuous interaction wrench causes large abrupt changes in the robot dynamics which easily destabilizes the system. Thus, a control strategy capable of adapting to the unknown environment and ensure a stable flight is needed. This work investigates methods to overcome these challenges.

![](images/2a04e30170dbda388e1c92ff5b1546e8042b111b560113b329f6efcb651c86db.jpg)  
Fig. 1: A tilt-arm omnidirectional flying vehicle, the Omav, is sliding along surfaces with different friction properties and geometries. The images shows the end effector is in contact with a double-sided tape while sliding on a rock like papier-machˆ e.´

Literature on adaptive sliding on partially-known and uneven surfaces exists mainly in the manipulation community, using passivity [8] or adaptive force control [9], [10], but they do not directly transfer to aerial robots. Adaptive force control is often used to modify the interaction force along the di rection normal to the surface according to geometry changes, while keeping the force constant along the sliding direction. As a result, an adaptation to heterogeneous surface is not possible. One could also employ disturbance observer-based robust control [11]. Such an approach has the disadvantage of being slow to react, especially in the presence of noisy measurements and inaccurate process models. Consequently, the flying vehicle will struggle to handle abrupt changes in the environment. Finally, another option is to use mechanical compliance at the flying machine’s end effector [12]. However, this increases the mechanical complexity and cost while further decreasing the system’s payload. Furthermore, it remains limited in terms of adaptability to environmental conditions.

An alternative approach to address the aforementioned sensing challenge is to use proprioceptive measurements and tactile sensing to react to changes in the geometry and friction properties of the environment. This type of approach can be found in quadrupedal robots like the MIT Cheetah [13] and Anymal [14]. These robots use either control signals to infer a leg touchdown event or use IMU signals, robot states, and control commands to implicitly infer the surface properties. Compared to 3D visual sensors, these two sensing modalities are more direct in measuring the contact and are more suitable in sensing the discontinuous features of the environment. However, they were never exploited in the field to aerial manipulation to face unknown environments.

From a control perspective, impedance control [15] is a suitable control strategy for physical interaction tasks, as it introduces algorithmic compliance, making the controlled system less susceptible to disturbances. This has been successfully demonstrated for sliding tasks on homogeneous surfaces with aerial vehicles [5], [7] or human-robot interaction [16]. Selection of the impedance gains is a tradeoff between controller tracking performance and system compliance. Depending on the task, different impedance parameters may perform optimally [17]. For this reason, a variable impedance controller [18] is an attractive solution. The works in [19] and [7] employed variable impedance control for aerial manipulation, recently formulated in terms of passivity-based control [20], [21].

A straightforward application of variable impedance control in sliding applications, is to change the stiffness gain according to the estimate of the friction coefficient obtained from a force-torque sensor. However, a low-pass filter for the noisy force-torque sensor measurements may introduce delay and the time-varying sensor bias may lead to incorrect estimates during flight. Another way would be to infer an adapting gain policy directly from the sensors reading, without passing though an explicit estimation of the environment parameters.

With recent advances in machine learning, deep reinforcement learning (RL) techniques [22] have become a popular tool to generate highly nonlinear and effective control policies using neural networks. Learning from simulation instead of from real-world is the preferred approach, especially concerning flying vehicles where a failure typically leads to a crash. In particular, student teacher setups [23] are exploited for improved learning efficiency, since a teacher policy can use privileged information from simulation to guide the student policy. On one hand, there exists a large quantity of works on end-to-end (i.e, from perception to actuator commands directly) reinforcement learning from simulation (see [24] and references therein). While being powerful, they are generally not efficient and require a high-fidelity simulation of the environment. On the other hand, methods combining reinforcement learning with variable impedance control have been proposed. Examples include manipulator control for light switch turning [25], peg-in-hole tasks [26], [27], human-robot collaboration [28], and hopper jumping [29]. They keep the impedance controller structure in the learning framework improving data-efficiency and requiring less fidelity in the simulation.

Considering the problem of aerial physical interaction with unknown surfaces, taking inspiration from different robotic domains, this paper presents a novel control method that combines the benefits of proprioceptive and tactile sensing, variable impedance control and reinforcement learning. Contrary to the previously mentioned works, instead of estimating the friction coefficients or surface geometry, we directly learn a mapping from the proprioceptive and tactile signals to the impedance control gains. When the environmental property changes, this neural network policy rejects these disturbances, preserving contact and keeping the orientation of the flying vehicle’s end effector constant. In particular, the policy changes the impedance control parameters according to tracking errors, IMU sensor measurements, and wrench sensing. A key hypothesis here is that there exists an underlying mapping between these signals and the control gains. The training of this policy is conducted entirely in simulation exploiting the simplified closed-loop dynamics, in a studentteacher setup. This solution allows a direct transfer of the learned policy from simulation to an omnidirectional aerial vehicle (Omav), significantly improving its robustness during interaction (Omav as shown in Fig. 1 and described in [30]).

Our contributions can be summarized as follows:

• An aerial sliding control strategy that adapts its gain to reject disturbances present in the interacting environment, including discontinuous changes in surface friction and geometry;

• Insights on how to address sim-to-real transfer by including a closed-loop controller to suppress model uncertainty and learning from simplified actuator dynamics for aerial interaction tasks.

The above contributions have been validated in the experiments using an Omav with a rigid single-body end effector for the task of sliding across challenging surfaces including a step and discontinuous surface friction changes.

## II. PRELIMINARIES

In this section we provide a brief overview of the models used to represent robot dynamics and environment interactions before describing the basics of impedance control.

## A. Robot dynamics

It is assumed that the flying vehicle has a single-body end effector rigidly attached to its body (see Fig. 2). The robot is modeled as a single rigid body and its dynamics is expressed using Newton-Euler method in free flight and interaction are given by the following equation

$$
M \dot {\tilde {v}} + C \tilde {v} + g = w _ {\mathrm{act}} + w _ {\mathrm{dist}},\tag{1}
$$

where $M \in \mathbb { R } ^ { 6 \times 6 }$ is the symmetric positive definite inertia matrix and $C \in \mathbb { R } ^ { 6 \times 6 }$ contains the centrifugal and Coriolis terms, and $\pmb { g } \in \mathbb { R } ^ { 6 }$ is the gravity. The generalized velocity $\tilde { \textbf { \textit { v } } } \in \ \mathbb { R } ^ { 6 }$ represents the center of mass velocity and body rates of the system. The generalized acceleration vector <sub>are</sub> d<sub>enote</sub>d <sub>as</sub> v˜˙ <sub>. T</sub>h<sub>e terms</sub> ${ \pmb w } _ { \mathrm { a c t } }$ and $\pmb { w } _ { \mathrm { d i s t } } \ \in \ \mathbb { R } ^ { 6 }$ are both stacked force and torque vectors acting on the system generated by rotor actuation and disturbance sources (e.g., contact or wind disturbances), respectively.

## B. Interaction with the environment

When the robot is sliding along the surface with its end effector, the force disturbance $\pmb { f } _ { \mathrm { d i s t } }$ has three sources: 1) environmental aerodynamic effects, e.g. ground effects, wall effects, and wind gusts, 2) actuation modeling errors [31], and 3) the contact force $f _ { \mathrm { c o n } }$ . Wall effects and wind gusts are assumed to be negligible for indoor flights as they account for less than one percent of prediction error according to [32]. The disturbance caused by modeling errors is an order of magnitude smaller than that of the contact force and thus also assumed to be negligible.

During interaction flights, the robot is assumed to have a single contact point with the uneven surface at C (Fig. 2). A local contact frame $\mathcal { F } _ { T }$ is attached to the contact point such that its x-axis is normal to the tangent plane at C. The contact force acting on the end effector expressed in the body frame $\mathcal { F } _ { B }$ is modeled as follows:

![](images/15be12a275baf1095ce48c9eed0490e47d47795db2bbac5d8d502c7247b31394.jpg)  
Fig. 2: Omav in interaction, showing some of the symbols and quantities required to model the system

$$
\boldsymbol {f} _ {\mathrm{dist}} ^ {B} = \boldsymbol {f} _ {\mathrm{con}} ^ {B} = \boldsymbol {R} ^ {B T} (F _ {\perp} \boldsymbol {n} _ {\perp} ^ {T} + F _ {\parallel} \boldsymbol {n} _ {\parallel} ^ {T}),\tag{2}
$$

where $F _ { \bot }$ is the scalar normal force and $F _ { \parallel }$ is the scalar friction force. The coordinate transformation matrix $R ^ { B T }$ transforms a vector from $\mathcal { F } _ { T }$ to $\mathcal { F } _ { B }$ . The unit normal vector $\smash { n _ { \perp } ^ { T } }$ is perpendicular to the tangent plane at C and ${  { n _ { | | } } } ^ { T }$ is the sliding force direction parallel to the tangent plane at C. The relative orientation $R ^ { \dot { B } T }$ is assumed to be partially known due to imperfect map. In addition, the end effector and the contact force creates a torque around the vehicle’s center of mass with the lever arm length denoted as l.

When the end effector is sliding with nonzero velocity on the surface, a Coulomb friction model is assumed, i.e.:

$$
F _ {\parallel} = \mu (\boldsymbol {p} _ {C}) F _ {\perp},\tag{3}
$$

where $\mu ( \pmb { p } _ { C } )$ is the friction coefficient that can vary spatially across the surface, depending the position of contact $p _ { C }$

## C. Impedance controller with constant gains

An impedance controller with constant control gains is a common approach used for aerial sliding tasks [30] and used in this paper as the baseline approach.

Given a desired sliding path on the surface, a reference pose trajectory is designed based on the given surface map . This reference trajectory consists of a desired center of mass position which results in a end effector position that is always behind the sliding surface by a constant distance $\delta \ \in \ \mathbb { R }$ (Fig. 2), also denoted as the penetration level. For the attitude part of the trajectory, the vector along the tool arm should align with the contact frame x-axis.

Given this desired reference trajectory, an impedance controller with constant control gains has the following form:

$$
\begin{array}{r l} & {\pmb {w} _ {\mathrm{act}} = \pmb {C} \tilde {\pmb {v}} + \pmb {g} + (M M _ {\mathrm{des}} ^ {- 1} - \mathbb {I} _ {6}) \pmb {w} _ {\mathrm{dist}}} \\ & {\qquad - M M _ {\mathrm{des}} ^ {- 1} (- M \dot {\tilde {\pmb {v}}} _ {\mathrm{ref}} + D _ {\mathrm{des}} \tilde {\pmb {e}} _ {v} + K _ {\mathrm{des}} \tilde {\pmb {e}} _ {s}),} \end{array}\tag{4}
$$

with $\tilde { e } _ { s } \in \mathbb { R } ^ { 6 }$ , containing the position and attitude tracking error as shown in [30]. $\mathbf { \bar { \Lambda } } M _ { \mathrm { d e s } } , \mathbf { \bar { \Lambda } } D _ { \mathrm { d e s } } , K _ { \mathrm { d e s } } \in \mathbb { R } ^ { 6 \times 6 }$ are the desired inertia, damping, and stiffness matrices, respectively.

Plugging (4) into (1) results in

$$
M _ {\mathrm{des}} \dot {\tilde {\boldsymbol {e}}} _ {v} + D _ {\mathrm{des}} \tilde {\boldsymbol {e}} _ {v} + K _ {\mathrm{des}} \tilde {\boldsymbol {e}} _ {s} = \boldsymbol {w} _ {\mathrm{dist}}.\tag{5}
$$

This implies that an impedance controller shapes the closedloop system as a second-order system. Note that there will be inevitable pose error due to the contact wrench acting on the flying vehicle. The wrench command ${ \pmb w } _ { \mathrm { a c t } }$ is then allocated through a chosen mapping and a saturation function to individual actuator commands (for more details see [30]).

## III. METHODOLOGY

## A. Problem statement

The goal is to enable a flying vehicle to accurately follow a trajectory planned based on an imperfect map while remaining stable and staying in contact with a surface which has unknown discontinuities in geometry and unknown friction properties $\mu ( \pmb { p } _ { C } )$ ). Given a task-space reference trajectory, we assume the robot has access to contact wrench measurements ${ \pmb w } _ { \mathrm { m e a s } }$ via a force torque sensor, and it is controlled by an impedance controller with a gain-adjusting policy $\pi _ { \theta } ,$ which is parametrized by θ. To achieve the above goal, we propose a strategy to find a deterministic policy $\pi _ { \theta }$ that adjusts the controller’s gains to fulfill the following criteria: 1) minimize the tracking error ${ \left\| \tilde { e } _ { v } \right\| } ^ { 2 } + { \left\| \tilde { e } _ { s } \right\| } ^ { 2 }$ where k·k denotes the Euclidean norm,; 2) ensure $f _ { \mathrm { c o n } } ^ { B } > 0 ; 3 )$ ensure the platform stability.

## B. Simulation using simplified dynamics

To allow for efficient evaluation and training of the policy $\pi _ { \theta } ,$ a simplified dynamics simulation is used. The flying vehicle is simulated as a single rigid body and the simulation of the individual actuator dynamics are approximated collectively as a single process. A saturation function on the wrench command is implemented, the output of which is delayed and set as external force and torque directly acting on the robot. Both the saturation threshold and the system delay are a conservative estimate of the empirically obtained actuation limits. This ensures that the actuator limits are well respected and the closed-loop system behaves like a delayed second-order system as designed. The inertia and mass are obtained from CAD. Although the actuator dynamics are simplified, special attention is paid to identify the correct center of mass position and the relative position of the end effector in the body frame $( r _ { \mathrm { { c o m } } }$ and $r _ { \mathrm { e n d } }$ in Fig. 2). They together determine the induced torque disturbance from a given contact force, which is essential for the simulation to learn the correct disturbance rejection strategy.

To simulate the interaction environment, surfaces that have different friction coefficients<sup>1</sup> are generated and concatenated together. Therefore, when the robot’s end effector slides across the border between two surfaces with different friction properties, it experiences discontinuous changes in interaction forces. Furthermore, each surface can have a different height that leads to an uneven surface as a whole.

## C. Variable impedance learning controller

The proposed approach adds a control gain adaptation policy to the standard impedance controller (4). The policy adapts the impedance controller gains based on the proprioceptive measurements and the tactile feedback via the adaptive unit described in Fig. 3. In particular, the desired stiffness $K _ { \mathrm { d e s } } ( z ) \in \mathbb { R } ^ { 6 \times 6 }$ is a function of the m-dimensional measurements $z \in \mathbb { R } ^ { m }$

![](images/ed081fdc38862e05e79fd4ea02be727796cd1c2e7bd860fab996e09fa7b22774.jpg)  
Fig. 3: Variable impedance learning controller. It augments the control strategy depicted in [30] by adding a control gain adaptation policy, which takes as input the state, control error and the wrench measurements from the wrench sensor and map them to impedance gain.

$$
\boldsymbol {K} _ {\mathrm{des}} (\boldsymbol {z}) = \boldsymbol {K} _ {\min} + \left(\boldsymbol {K} _ {\max} - \boldsymbol {K} _ {\min}\right) \operatorname{diag} \{\pi (\boldsymbol {z}) \}\tag{6}
$$

where m is the dimension of the measurements, $K _ { \mathrm { m i n } }$ and $K _ { \mathrm { m a x } }$ are diagonal positive semidefinite matrices, and $\pi :$ $\mathbb { R } ^ { m } \to \mathbb { R } ^ { 6 }$ where $\mathbf { 0 } < \pi ( \cdot ) < \mathbf { 1 }$ with vector 0 and 1 of dimension 6. These constraints on the mapped value make sure the adaptive gains have lower and upper bounds. The lower bound $K _ { \mathrm { m i n } }$ ensures a minimum tracking performance while the upper bound $K _ { \mathrm { m a x } }$ prevents the system from instabilities caused by actuator saturation and system delay. These limits are both derived empirically through experiments. For brevity, in the following we use π instead of $\pi _ { \theta } .$

To obtain a damped second-order system, we impose a fixed relationship between $D _ { \mathrm { d e s } }$ and $\kappa _ { \mathrm { d e s } } ( z )$ . The desired damping $D _ { \mathrm { d e s } }$ is varied with the square root of the diagonal components of $\kappa _ { \mathrm { d e s } } ( z )$

$$
\pmb {D} _ {\mathrm{des}} = 2 \zeta \sqrt {\pmb {K} _ {\mathrm{des}} (\pmb {z})},\tag{7}
$$

where ζ is a damping ratio. While we assume that the desired stiffness and damping can be well tracked, the desired inertia is in practice challenging to track as it requires an accurate actuation control [5]. We therefore set $M _ { \mathrm { d e s } }$ equal to M and the adaptation of $M _ { \mathrm { d e s } }$ is deferred to future works.

With $M = M _ { \mathrm { d e s } }$ and (7) inserted into (4), the following adaptive controller command can be obtained,

$$
\pmb {w} _ {\mathrm{act}} = M \dot {\tilde {\pmb {v}}} _ {\mathrm{ref}} - 2 \zeta \sqrt {\pmb {K} _ {\mathrm{des}} (\pmb {z})} \tilde {\pmb {e}} _ {v} - \pmb {K} _ {\mathrm{des}} (\pmb {z}) \tilde {\pmb {e}} _ {s} + C \tilde {\pmb {v}} + \pmb {g}.\tag{8}
$$

With (8) plugged into (1), the closed-loop error dynamics are shaped as a second-order system,

$$
M _ {\mathrm{des}} \dot {\tilde {e}} _ {v} + 2 \zeta \sqrt {K _ {\mathrm{des}} (z)} \tilde {e} _ {v} + K _ {\mathrm{des}} (z) \tilde {e} _ {s} = w _ {\mathrm{dist}}.\tag{9}
$$

Note that changing the stiffness $\kappa _ { \mathrm { d e s } } ( z )$ affects the interaction wrench ${ \pmb w } _ { \mathrm { d i s t } }$ . To see this, consider (9) at steady state $\dot { \tilde { \mathbf { e } } } _ { v } = \tilde { e } _ { v } = \mathbf { 0 }$ and projected along the surface normal direction. We obtain

$$
k _ {\perp} (\boldsymbol {z}) \delta = F _ {\perp},\tag{10}
$$

where $k _ { \perp } ( z )$ is the position stiffness gain in the surface normal direction. Assuming the end effector is in contact with the surface, which results in a constant position tracking error $\delta , F _ { \perp }$ is thus proportional to $k _ { \perp } ( z )$ . This is the intuition of why we adapt the stiffness gain to counteract surface disturbances.

## D. Reinforcement learning of gain adaptation policy

We assume that the mapping π can be modeled as a discrete-time continuous Markov decision process (MDP). An MDP is defined by a state space S, an action space A, a scalar reward function $\mathcal { R }$ , and the transition probability $P$ that dictates the stochastic system dynamics. A learning agent selects an action a from its policy π and receives a reward r. The objective of the RL framework is to find an optimal policy $\pi ^ { * }$ that maximizes the discounted sum of rewards over an infinite time horizon:

$$
\pi^ {*} = \arg \max _ {\pi} \mathbb {E} _ {\tau (\pi)} \sum_ {t = 0} ^ {\infty} \gamma^ {t} r [ t ]\tag{11}
$$

where $\gamma \in \mathsf { \Gamma } ( 0 , 1 )$ is the discount factor, and $\tau ( \pi )$ is the trajectory distribution under policy π, with t denoting the discretized time indices. The reward $r [ t ]$ at k is

$$
\begin{array}{r} r [ t ] = - l _ {\pmb {e} _ {R}} \left\| \pmb {e} _ {R} [ t ] \right\| ^ {2} - l _ {p} \left\| \pmb {e} _ {p} [ t ] \right\| ^ {2} - l _ {d} \left\| d [ t ] \right\| ^ {2} \\ - l _ {\pmb {\omega}} \left\| \pmb {\omega} [ t ] \right\| ^ {2} - l _ {\pmb {a}} \left\| \frac {\pmb {a} [ t ]}{\left\| \pmb {a} [ t ] \right\|} - \frac {\pmb {a} [ t - 1 ]}{\left\| \pmb {a} [ t - 1 ] \right\|} \right\| ^ {2} \end{array}\tag{12}
$$

$l _ { \star }$ with subscripts ? denotes the corresponding weight chosen such that all individual reward terms are at the same order of magnitude. $e _ { R } [ t ]$ and $e _ { p } [ t ]$ denotes the attitude and translational tracking error. $d [ t ]$ denotes the scalar distance along the x-axis of the local contact frame between the end effector and the surface. The purpose of this term is to make sure that the policy keeps the end-effector in contact with the surface. $\omega [ t ]$ denotes the angular velocity. We penalize large angular velocities to avoid instabilities. The action $\mathbf { \Delta } \mathbf { a } [ t ]$ is the output of the policy $\pi ( z )$ at time t. The associated term is to make sure that the control inputs are smooth. The loss components are designed to reduce the tracking error while keeping contact with the surface without causing discontinuities in the actions and thus the actuator commands. These terms are in line with the problem statement in Sec. III-A. Note that since the simulation of each individual actuator dynamics is omitted. The energy consumption of the flying vehicle can only be indirectly inferred and is therefore not included in the reward function.

The policy π is a fully connected neural network with three hidden layers of 32 units, its activation functions being leaky ReLu, and its last layer being a Sigmoid layer which guarantees to map to a bounded interval. For training we use the off-the-shelf RL algorithm proximal policy optimization (PPO) [33], a policy gradient algorithm that has been demonstrated to work for variable impedance control in contact tasks with a manipulator [34].

## E. Learning from simulaiton

To efficiently learn an optimal policy that determines the adaptive stiffness $\kappa _ { \mathrm { d e s } } ( z )$ (see (6)), a student teacher learning approach [23] is deployed.

Fig. 4 provides an overview of this approach: Firstly, we design a teacher with access to privileged (ground-truth)

![](images/c2debcb6e545286df832e73f49b995ddc6cee3e3884048db332b750fb8568665.jpg)  
Fig. 4: This block diagram shows the main steps in the student-teacher learning of the control gain adaptation policy.

information to dynamically select the desired stiffness in the variable impedance controller. Then a policy is learned to emulate the teacher and may be further improved using RL. The policy can be directly transferred to real-world without any additional sim-to-real adaptation.. The intuition behind the student teacher learning is that the teacher has access to the privileged information is much easier to design or train in an RL setting. We can also embed empirical tuning experience or other adaptive variable impedance strategies into the teacher. This is helpful for challenging tasks, as we find out empirically a direct reinforcement learning always lead to instabilities of the flying vehicle and prevents successful learning.

1) Teacher design: The teacher $\pi _ { \mathrm { t } }$ serves as a guidance policy for the student policy. It makes use of the ground truth information $z _ { \mathrm { g t } }$ (friction coefficients and local height map) from the simulation, to which the student policy does not have access in real deployment.

In this work, we employ either a simple handcrafted policy $\pi _ { \mathrm { t } }$ or a neural network learned from simulation using reinforcement learning as the teacher. For handcrafted policy, the design intuition of the handcrafted policy is mainly based on the fact that with a larger friction coefficient, the angular stiffness gain should be increased to counteract the increased torque disturbance. At the same time, the translational stiffness gain should be decreased to reduce normal force $F _ { \bot }$ (see (10)) and the torque disturbance. For reinforcement learning of the teacher, friction coefficient at contact point and the surface normal vector in the neighbourhood are used in addition to z. They directly provide information about the interacting environment: the surface unevenness and friction property.

2) Student learning: The control gain adaptation policy $\pi ^ { * } ( z )$ is bootstrapped via supervised learning with the following loss function

$$
\pi^ {*} = \arg \min _ {\pi} \left\| \pi_ {\mathrm{t}} (\boldsymbol {z} _ {\mathrm{gt}}) - \pi (\boldsymbol {z})\right) \| ^ {2}.\tag{13}
$$

where the feature z contains control signal, state estimate, interaction wrench measurements and the IMU signals. Here, the control signal, the IMU measurements and the state estimate are the propprioceptive sensing and interaction wrench measurements are tactile sensing. They indirectly provide the information about the interaction between the environment and the robot.

Training data is collected by rolling out the simulation using the teacher. For each rollout, the robot first approaches the surface, gets into contact and starts sliding following the desired trajectory. The policy $\pi ^ { * } ( z )$ can be further refined using RL.

3) Data processing: Noisy observations (especially the interaction wrench measurements) are first-order low-pass filtered before they are input to the student policy. To have a smooth change in the stiffness gain $\kappa _ { \mathrm { d e s } } ( z )$ , the output of the student policy is also low-pass filtered before they are used to compute $\kappa _ { \mathrm { d e s } } ( z )$

## F. Remark

a) Stability: We remark that the stability using RL may be ensured using the concept of passivity [35], or Lyapunov methods [36]. This is left as future work. Instead we discuss here practical measures to face possible causes of the instability. Those are mainly twofold: actuator saturation due to the rapid changes in the control gains or high gains and low gains which is incapable of stabilizing this openloop unstable system. Thus we implemented the following strategies: 1) Lower and upper bounds on the stiffness gain as shown in (6); 2) Slew rate limit on the gains are empirically determined; 3) The output of the policy $\pi _ { \mathrm { s } } ^ { * } ( z )$ is filtered for a smooth control signal.

b) Sim-to-real transfer: The sim-to-real gap, i.e., the mismatch between simulation and reality, is a challenging problem when learning from simulation, and can limit the transferability of the learned policy. Frequent causes of simto-real gaps are inaccurate modeling of the actuator dynamics and delays in the system [37]. This is especially a problem for end-to-end learning approaches. Without feedback controller in the loop the learning procedure relies on the accuracy of the open-loop dynamics simulation and the simto-real gap can diverge exponentially. However, given a welldesigned feedback controller (e.g., the one one presented in Fig. ??), which shapes the system to a desired secondorder system (9) and suppresses model uncertainty, the gap between the reality and the simulation is kept small. This is particularly advantageous for a complicated system like the Omav, where a large amount of training data is required for an accurate model learning of the whole body dynamics (18 actuators and 6 degrees of freedom). As a comparison, a million samples are required for the modeling of a single one degree of freedom actuator of quadrapedal walking robot [37]. What is worse, if the Omav crashes or if its configuration changes, training data needs to be recollected again for an accurate model learning. Our approach does not require such a meticulous effort.

## IV. EXPERIMENTS

## A. Experimental setup

The experimental set-up and the platform are shown in Fig. 1. The experiments are carried out at the indoor aerial robotic testbed of the Autonomous System Lab, ETH Zurich. The Omav weighs 4.5 kg and is equipped with an NUC i7 computer and a PixHawk flight controller. For a more complete description of this platform see [30]. A pole end effector is rigidly attached to the vehicle and points outwards for a sufficient margin between the propellers and the surface. A ping pong ball is attached at the pole tip to ensure a single contact during sliding movements. A force torque sensor from Rokubi, mounted to the end effector, measures the interaction wrench. There are three interaction environments for real experiments: sand paper on a flat white board to investigate interaction with heterogeneous surface, a step of 2 cm on a white board (Fig. 6) to investigate interaction with discontinuous surface geometry, and finally, a rocklike structure (Fig. 1) which combines both traits to form a challenging environment typically seen in real applications. A motion capture system provides pose estimates for both the robot and the whiteboard/rock at 100 Hz. The robot is unaware of their surface properties and the local unevenness on the surface. The task trajectory is to follow a straight line trajectory parallel to the gravity vector with zero pitch and roll while sliding on the vertical surface. The penetration level δ is set to 0.07 m, which leaves sufficient margin to ensure contact under the state estimate uncertainty.

For the simulation we use RaiSim [38], a cross-platform multi-body physics engine for robotics. During training in the simulation, for each rollout, the robot approaches the surface and starts sliding along the surface for 15 seconds, which emulates a task trajectory from the real-world experiments. During each rollout, the Omav reaches a speed of $\mathrm { 0 . 2 m s ^ { - 1 } }$ and slides across surfaces with different friction coefficients. For each interaction environment, we set up a different interaction environment in the simulation and learn a different policy to treat each problem separately. Find a single policy that tackles different environments can be studied in the context of continual learning and is planned for future work.

To train the teacher or student policy using RL, a hundred simulation instances are spawned with slightly perturbed vehicle and environment properties. In each epoch, these instances are simultaneously simulated to obtain a batch of rewards for stochastic gradient policy optimization. The policy was trained on a single NVIDIA 3060Ti GPU. which takes from 2 to 8 hours depending on the interacting environment.

Since there is only a straight line to follow, the dimension of the measurement vector z is reduced to six: filtered pitch velocity, pitch error, position control error in the sliding direction, linear velocity, friction force filtered, normal force filtered. The output to the action space are the linear gain in the surface normal direction and the angular gain in the pitch direction, i.e. the axis of torque disturbance. The rest of the controller gains is kept constant.

## B. Sim-to-real transfer

Fig. 5 demonstrates the policy transfer from simulation to reality. It compares four experiments completing the same sliding task with different controllers (baseline or our approach) and different set-ups (simulation or real experiment). The plots are aligned using the measured force impact when the Omav enters from free flight into contact with the whiteboard.

For evaluation, the Omav must slide across three concatenated surfaces: the first is the whiteboard with low friction, then a sand paper with high friction, and finally the whiteboard again. The friction coefficients are empirically estimated through force torque sensor measurements, with which three surfaces are generated in the simulation that replicate the experimental evaluation set-up. The approach in evaluation is a policy $\pi ^ { * }$ defined in the (13) (without further refinement using RL) and a baseline impedance controller with constant control gains. The policy is trained in a simulation environment with six different surface friction coefficients (0.05, 0.15, 0.25, 0.45, 0.55, 0.62) that covers a range of friction coefficients. During collection of data using the teacher, each surface is randomly assigned one of the six friction coefficients to robustify the learned policy.

![](images/90d96e5edb2e31011892aaa34dd6369cf1261f2c04e94fde00b9bf780487877d.jpg)  
Fig. 5: Comparison between baseline and our approach while the robot approaches and slides across two surfaces with different friction coefficients, both in simulation (dashed line) and real-world (solid line). $k _ { a }$ and k denote the angular stiffness gain and translation stiffness gain, respectively.

Fig. 5 demonstrates that the learned adaptive control strategy can be successfully transferred from simulation to the real-world. Given the same controller (either baseline or our approach), the simulation and the real-world experiments result in a close similarity of the respective robot state (the pitch angle) and controller gains (the angular stiffness gain $k _ { a }$ and translational stiffness gain $k _ { l } )$ . The pitch angle is shown since it has the most obvious correlation with the surface friction coefficient given the baseline controller.

## C. Sliding across a heterogeneous flat surface

Fig. 5 also showcases the performance of the regressed student policy on heterogenous surfaces. In this subsection, we only evaluate the real experimental data. While sliding on the surface, the contact force unavoidably leads to a tilted angle of the Omav. With the baseline controller, the vehicle tilts on average 4.9° after encountering a high friction surface (from 199 s to 202 s), whereas with our approach, the tilt of the vehicle only increases to about $1 . 3 ^ { \circ }$ on average. This shows that the Omav is able to keep a almost constant tilt angle when sliding across different surfaces, thus improve the attitude tracking performance and reducing the chance of a crash at the transition of difference surfaces.

## D. Sliding over a surface with discontinuous geometry

The experiment (Fig. 6) presented in this section aims to investigate the ability to reject the disturbance caused by surface discontinuity and remain stable (criteria 3 in Sec. III-A ). For the real experiment, a piece of foam is taped to the whiteboard (Fig. 6) and creates a step of about 2 cm along the sliding trajectory. During sliding, the step blocks the end-effector, which leads to an increase of the pitch angle. The end-effector then detachs from the surface and the sudden disappearance of the contact force presents as a large disturbance to the robot (see video attachment).

![](images/e30c03d8c852591b42bc6d8978e2d8f251918ab569972cc9ff00448977642968.jpg)  
Fig. 6: The end effector is blocked by a step of 2 cm made out of foam while sliding on a slippery whiteboard.

Our approach is trained as follows: Initially an even surface with randomly generated friction coefficients as described in Sec. IV-C is used to bootstrap the policy. Then a piece of uneven surface with 1 cm steps is created by setting each neighbouring surface to have a height difference of 1 cm. The policy is refined and trained for 900 epochs. The height difference is then changed to 2 cm with another 700 epochs of training. Training was terminated early when the reward stopped increasing for 300 epochs.

The behaviour exhibited by our approach when a step is encountered is to adapt the control gain to be more compliant. This leads to less oscillations in pitch velocity when the Omav’s end effector slips and is out of contact (the yellow region in Fig. 6) compared to the baseline approach.

## E. Sliding across a challenging surface

A challenging rock-like papier-mach ˆ e surface (Fig. 1) is´ set up to compare our approach with baseline controllers. The surface of this rock is uneven and heterogeneous. Doublesided tapes and plastic surface with lubricant are added to the surface to emulate heterogeneous surface. Two trajectories were tested for more variety. For the training environment, we use procedurally generated terrain maps that have similar surface variations to the rock. During rollout, the fricition coefficients are randomly selected from six friction coefficients (0.1, 0.15, 0.45, 0.6, 0.75, 0.9) every four seconds. Note that the evaluation and training environments are distinct and demonstrated the generalizability of our approach.

As shown in Fig. 7, our approach consistently outperforms baseline controller with nine combinations of low, middle and high angular and translation gain. These different combinations are tested to see whether a tuning process that is often practiced in real-world can outperform our approach. Each data point represents the average over the sliding of the same trajectory for the three times with the same controller. Several interesting observations can be made: firstly, among the baseline approaches, a good tracking performance is generally achieved by high stiffness controller but the converse is not true. It is observed in the experiment that a high gain controller (both high in $k _ { l }$ and $k _ { a } )$ can lead to instability. However, we never experienced instability issues with our policy throughout the experiments. This indicates that our approach adapts the controller to be more compliant when necessary. Secondly, for each trajectory, the optimal set of constant gains are different. For example, to achieve best tracking performance in pitch, trajectory red and trajectory blue have different optimal gains (upper left in the plot). This means that in reality, the engineer have to tune the parameters for each specific trajectory and surface to gain optimal performance. However, our approach always performs well for both trajectories. Our approach are both located in the lower left of the plot, which means good tracking performance in both position and orientation. This implies that our policy adapts to the surface unevenness and varying friction coefficients.

![](images/f1a1bbc4b1895606d6957b51b4c02d4ff25c627e6165dc61dcfb7e42b398a126.jpg)  
Fig. 7: Comparison of tracking performance of the learned policy with different combinations of constant gains. The learned policy outperforms all the constant gain performance. Note that it is plotted in logarithm scale for a better visualization of data. Two different colors are used to denote different trajectories. $k _ { a }$ and $k _ { l }$ denote the angular stiffness gain and translation stiffness gain, respectively.

![](images/71169b09909ca659d6e5af6f24a3dc1524ac8e5834bd5b65282952a1ec6a3810.jpg)  
Fig. 8: An time history plot of partial inputs and outputs of the policy when sliding on the rock-like surface

An time history plot of partial inputs and outputs of the policy in plot Fig. 8 provides an intuition on how our policy behaves. Note that when the pitch error increases from 194 s to 195.5 s, the angular gain $k _ { a }$ is also increased to maximum to reduce the tracking error. However, around 195.5, the endeffector detaches from the surface and causes oscillations on pitch rate, the angular gain is decreased to be more compliant.

## V. CONCLUSION

This paper presented an approach to reject disturbances caused by discontinuous surface variations in geometry and friction during aerial sliding tasks for fully actuated flying vehicles. When the environmental property changes, an adaptation policy adjusts the control gains of a standard impedance controller to reject these disturbances. Experimental results demonstrated that the policy learned in simulation can be directly transferred to the aerial vehicle without adaptation. The learned policy is able to slide on a challenging rock-like surface and outperform state-of-art interaction controllers .

## REFERENCES

[1] M. Tognon, H. A. T. Chavez, E. Gasparin, Q. Sabl ´ e, D. Bicego,´ A. Mallet, M. Lany, G. Santi, B. Revaz, J. Cortes´ et al., “A trulyredundant aerial manipulator system with application to push-and-slide inspection in industrial plants,” IEEE Robotics and Automation Letters, vol. 4, no. 2, pp. 1846–1851, 2019.

[2] M. A. Trujillo, J. R. Mart<sup>´</sup> ´ınez-de Dios, C. Mart´ın, A. Viguria, and A. Ollero, “Novel aerial manipulator for accurate and robust industrial ndt contact inspection: A new tool for the oil and gas inspection industry,” Sensors, vol. 19, no. 6, p. 1305, 2019.

[3] R. Watson, M. Kamel, D. Zhang, G. Dobie, C. MacLeod, S. G. Pierce, and J. Nieto, “Dry coupled ultrasonic non-destructive evaluation using an over-actuated unmanned aerial vehicle,” IEEE Transactions on Automation Science and Engineering, 2021.

[4] A. Ollero, M. Tognon, A. Suarez, D. Lee, and A. Franchi, “Past, present, and future of aerial robotic manipulators,” IEEE Transactions on Robotics, 2021.

[5] M. Ryll, G. Muscio, F. Pierri, E. Cataldi, G. Antonelli, F. Caccavale, D. Bicego, and A. Franchi, “6d interaction control with aerial robots: The flying end-effector paradigm,” The International Journal of Robotics Research, vol. 38, no. 9, pp. 1045–1062, 2019.

[6] D. Tzoumanikas, F. Graule, Q. Yan, D. Shah, M. Popovic, and S. Leutenegger, “Aerial manipulation using hybrid force and position nmpc applied to aerial writing,” Robotics: Science and Systems, 2020.

[7] K. Bodie, M. Brunner, M. Pantic, S. Walser, P. Pfandler, U. Angst,¨ R. Siegwart, and J. Nieto, “Active interaction force control for contactbased inspection with a fully actuated aerial vehicle,” IEEE Transactions on Robotics, vol. 37, no. 3, pp. 709–722, 2020.

[8] E. Shahriari, L. Johannsmeier, E. Jensen, and S. Haddadin, “Power flow regulation, adaptation, and learning for intrinsically robust virtual energy tanks,” IEEE Robotics and Automation Letters, vol. 5, no. 1, pp. 211–218, 2019.

[9] S. Jung, T. C. Hsia, and R. G. Bonitz, “Force tracking impedance control of robot manipulators under unknown environment,” IEEE Transactions on Control Systems Technology, vol. 12, no. 3, pp. 474– 483, 2004.

[10] W. Amanhoud, M. Khoramshahi, M. Bonnesoeur, and A. Billard, “Force adaptation in contact tasks with dynamical systems,” in 2020 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2020, pp. 6841–6847.

[11] D. Lee, H. Seo, I. Jang, S. J. Lee, and H. J. Kim, “Aerial manipulator pushing a movable structure using a dob-based robust controller,” IEEE Robotics and Automation Letters, vol. 6, no. 2, pp. 723–730, 2020.

[12] A. Suarez, G. Heredia, and A. Ollero, “Physical-virtual impedance control in ultralightweight and compliant dual-arm aerial manipulators,” IEEE Robotics and Automation Letters, vol. 3, no. 3, pp. 2553– 2560, 2018.

[13] D. J. Hyun, S. Seok, J. Lee, and S. Kim, “High speed trotrunning: Implementation of a hierarchical controller using proprioceptive impedance control on the mit cheetah,” The International Journal of Robotics Research, vol. 33, no. 11, pp. 1417–1445, 2014.

[14] J. Lee, J. Hwangbo, L. Wellhausen, V. Koltun, and M. Hutter, “Learning quadrupedal locomotion over challenging terrain,” Science robotics, vol. 5, no. 47, 2020.

[15] N. Hogan, “Adaptive tank-based control for aerial physical interaction with uncertain dynamic environments with energy-task estimation,” in 1984 American control conference. IEEE, 1984, pp. 304–313.

[16] M. Tognon, R. Alami, and B. Siciliano, “Physical human-robot interaction with a tethered aerial vehicle: Application to a force-based human guiding problem,” IEEE Transactions on Robotics, vol. 37, no. 3, pp. 723–734, 2021.

[17] D. S. Walker, J. K. Salisbury, and G. Niemeyer, “Demonstrating the benefits of variable impedance to telerobotic task execution,” in 2011 IEEE International Conference on Robotics and Automation. IEEE, 2011, pp. 1348–1353.

[18] R. Ikeura and H. Inooka, “Variable impedance control of a robot for cooperation with a human,” in Proceedings of 1995 IEEE International Conference on Robotics and Automation, vol. 3. IEEE, 1995, pp. 3097–3102.

[19] A. Y. Mersha, S. Stramigioli, and R. Carloni, “Variable impedance control for aerial interaction,” in 2014 IEEE/RSJ International Conference on Intelligent Robots and Systems. IEEE, 2014, pp. 3435–3440.

[20] B. Gerlagh, “Energy-aware adaptive interaction control using offline task-based optimization in an impedance control framework,” Master’s thesis, University of Twente, 2020.

[21] N. Hogan, “Impedance control: An approach to manipulation,” in Submitted to IEEE RA-L Variable Impedance Control And Learning In Complex Interaction Scenarios: Challenges And Opportunities. IEEE, 2022.

[22] R. S. Sutton and A. G. Barto, Reinforcement learning: An introduction. MIT press, 2018.

[23] D. Chen, B. Zhou, V. Koltun, and P. Krahenb ¨ uhl, “Learning by ¨ cheating,” in Conference on Robot Learning. PMLR, 2020, pp. 66– 75.

[24] A. S. Polydoros and L. Nalpantidis, “Survey of model-based reinforcement learning: Applications on robotics,” Journal of Intelligent & Robotic Systems, vol. 86, no. 2, pp. 153–173, 2017.

[25] J. Buchli, F. Stulp, E. Theodorou, and S. Schaal, “Learning variable impedance control,” The International Journal of Robotics Research, vol. 30, no. 7, pp. 820–833, 2011.

[26] C. C. Beltran-Hernandez, D. Petit, I. G. Ramirez-Alpizar, and K. Harada, “Variable compliance control for robotic peg-in-hole assembly: A deep-reinforcement-learning approach,” Applied Sciences, vol. 10, no. 19, p. 6923, 2020.

[27] C. C. Beltran-Hernandez, D. Petit, I. G. Ramirez-Alpizar, T. Nishi, S. Kikuchi, T. Matsubara, and K. Harada, “Learning force control for contact-rich manipulation tasks with rigid position-controlled robots,” IEEE Robotics and Automation Letters, vol. 5, no. 4, pp. 5709–5716, 2020.

[28] L. Roveda, J. Maskani, P. Franceschi, A. Abdi, F. Braghin, L. M. Tosatti, and N. Pedrocchi, “Model-based reinforcement learning variable impedance control for human-robot collaboration,” Journal of Intelligent & Robotic Systems, vol. 100, no. 2, pp. 417–433, 2020.

[29] M. Bogdanovic, M. Khadiv, and L. Righetti, “Learning variable impedance control for contact sensitive tasks,” IEEE Robotics and Automation Letters, vol. 5, no. 4, pp. 6129–6136, 2020.

[30] K. Bodie, M. Brunner, M. Pantic, S. Walser, P. Pfndler, U. Angst, R. Siegwart, and J. Nieto, “An Omnidirectional Aerial Manipulation Platform for Contact-Based Inspection,” Robotics: Science and Systems XV, 2019.

[31] W. Zhang, M. Brunner, L. Ott, M. Kamel, R. Siegwart, and J. Nieto, “Learning dynamics for improving control of overactuated flying systems,” IEEE Robotics and Automation Letters, vol. 5, no. 4, pp. 5283–5290, 2020.

[32] A. Garofano-Soldado, G. Heredia, and A. Ollero, “Aerodynamic interference in confined environments with tilted propellers: Wall effect and corner effect,” in 2021 Aerial Robotic Systems Physically Interacting with the Environment (AIRPHARO). IEEE, 2021, pp. 1–8.

[33] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal policy optimization algorithms,” arXiv preprint arXiv:1707.06347, 2017.

[34] R. Mart´ın-Mart´ın, M. A. Lee, R. Gardner, S. Savarese, J. Bohg, and A. Garg, “Variable impedance control in end-effector space: An action space for reinforcement learning in contact-rich tasks,” in 2019 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2019, pp. 1010–1017.

[35] E. Spyrakos-Papastavridis, P. R. Childs, and J. S. Dai, “Passivity preservation for variable impedance control of compliant robots,” IEEE/ASME Transactions on Mechatronics, vol. 25, no. 5, pp. 2342– 2353, 2019.

[36] S. A. Khader, H. Yin, P. Falco, and D. Kragic, “Stability-guaranteed reinforcement learning for contact-rich manipulation,” IEEE Robotics and Automation Letters, vol. 6, no. 1, pp. 1–8, 2020.

[37] J. Hwangbo, J. Lee, A. Dosovitskiy, D. Bellicoso, V. Tsounis, V. Koltun, and M. Hutter, “Learning agile and dynamic motor skills for legged robots,” Science Robotics, 2019.

[38] J. Hwangbo, J. Lee, and M. Hutter, “Per-contact iteration method for solving contact dynamics,” IEEE Robotics and Automation Letters, vol. 3, no. 2, pp. 895–902, 2018. [Online]. Available: www.raisim.com