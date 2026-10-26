# Whole-Body Integrated Motion Planning for Aerial Manipulators

Weiliang Deng , Hongming Chen , Biyu Ye , Haoran Chen , Ziliang Li and Ximin Lyu

Abstract—Expressive motion planning for Aerial Manipulators (AMs) is essential for tackling complex manipulation tasks, yet achieving coupled trajectory planning adaptive to various tasks remains challenging, especially for those requiring aggressive maneuvers. In this work, we propose a novel wholebody integrated motion planning framework for quadrotor-based AMs that leverages flexible waypoint constraints to achieve versatile manipulation capabilities. These waypoint constraints enable the specification of individual position requirements for either the quadrotor or end-effector, while also accommodating higher-order velocity and orientation constraints for complex manipulation tasks. To implement our framework, we exploit spatio-temporal trajectory characteristics and formulate an optimization problem to generate feasible trajectories for both the quadrotor and manipulator while ensuring collision avoidance considering varying robot configurations, dynamic feasibility, and kinematic feasibility. Furthermore, to enhance the maneuverability for specific tasks, we employ Imitation Learning (IL) to facilitate the optimization process to avoid poor local optima. The effectiveness of our framework is validated through comprehensive simulations and real-world experiments, where we successfully demonstrate nine fundamental manipulation skills across various environments.

Index Terms—Aerial manipulator, delta arm, motion planning, collision avoidance, waypoint constraint.

## I. INTRODUCTION

ERIAL manipulators, which consist of Unmanned Aerial Vehicles (UAVs) and manipulators, have garnered significant interest from researchers and industries due to their wide operational range and agile manipulation capabilities [1]. These systems have proven to operate effectively in challenging or hazardous environments, facilitating applications across various fields such as 3D printing, inspection, transportation, grasping, repair and perching [2]–[7]. Various aspects of aerial manipulators have been extensively studied, including platform design [8]–[11], motion planning [12], [13], and precise control [14]–[16]. However, most existing approaches have been developed for specific tasks with corresponding algorithms and hardware [17]. Developing integrated planning frameworks that can autonomously accomplish multiple manipulation skills, particularly those requiring aggressive maneuvers, remains challenging.

![](images/a4195ca4111bbe27b0d7b01f7fa9ad4a97f4cca0012ee54b9b3514d5f88126dd.jpg)

![](images/4d8e7f54fcc34f4a39cbf63f861ec86664e827dbfa6d5513649c255dfe72160e.jpg)  
Fig. 1. Demonstration of aerial striking using whole-body integrated motion planning. (a) The aerial manipulator executes a planned trajectory to reach a constrained waypoint, maintaining an attitude parallel to the inclined plane while configuring its delta arm for balloon striking. (b) Time-lapse visualization capturing the complete execution of the planned motion.

Unlike decoupled approaches that separately plan UAV and manipulator trajectories, coupled planning enables coordinated motion generation for the entire system, preserving its full maneuverability for complex tasks [18]. However, coupled planning for aerial manipulators presents significant challenges, with the most fundamental stemming from the high-dimensional state space. The inclusion of additional joints from the manipulators greatly increases the system’s dimensionality, resulting in a vastly expanded search space. To address the challenges of high-dimensional motion planning, various approaches have been developed. Early work proposes an optimization-based framework for aerial manipulators, focusing on aerial grasping of moving targets [19]. However, this approach primarily prioritizes time minimization while overlooking other critical performance metrics, such as energy consumption. The method in [20] performs trajectory search in the task space and employs a controller as a local planner to generate whole-body motions. However, this approach lacks global optimization considerations and therefore cannot guarantee overall trajectory smoothness or energy efficiency. The approach in [21] integrates end-effector and UAV dynamics within an NMPC framework for aerial writing tasks. More recently, a discrete-mechanics-based time-optimal trajectory planning framework with complementarity constraints was developed for aerial manipulators [4]. However, both approaches are designed for specific single tasks and their formulations make them difficult to systematically determine more constrained intermediate states, for example striking a point or being inclined at a certain angle, and lack flexibility to address broader manipulation tasks.

![](images/628e3a67ac9e18a7c6f6c9285b921c16841dd26413dc91da9f3bed9b4d606b56.jpg)  
Fig. 2. Coordinate frames of the aerial manipulator system: World frame $( \bar { \mathcal { F } } _ { \mathcal { W } } )$ , Body frame $( \mathcal { F } _ { B } )$ , and Delta frame $( \bar { \mathcal { F } _ { \mathcal { D } } } )$ .

To enable versatile aerial manipulation, a planning framework should be capable of accommodating various simple skills, including grasping, striking, pushing, pulling, writing, and so on [22], [23]. Intuitively, these complex skills can be naturally broken down into fundamental subtasks: point traversing for the quadrotor, point reaching for the endeffector, and combining with extra velocity and orientation requirements as needed at these points. Therefore, we introduce flexible partial waypoint constraints, which allow selective specification of position, velocity, or orientation requirements for individual quadrotors or end-effectors at different waypoints, rather than imposing complete state constraints at every point. However, the diverse types of waypoints pose significant challenges due to the need to satisfy precise intermediatestate requirements while maintaining feasibility across the entire trajectory [24]. Existing methods for handling waypoint constraints fall into two main categories: multi-stage sampling and optimization integration methods. Multi-stage sampling divides the trajectory into segments by setting multiple start and goal points [25]. While this approach can effectively handle partial waypoint constraint problems, it primarily focuses on position without considering orientation information. Introducing orientation constraints may lead to a significantly enlarged search space or even result in the inability to find feasible solutions. For optimization integration methods, several approaches have been proposed with varying strategies. The method in [26] treats both polynomial trajectories and waypoint arrival times as optimization variables, then penalizes the distance between the end-effector and target points at those specific times. This method provides valuable insights, but for multiple waypoint constraints, it should introduce more time optimization variables to determine the waypoint constraints scope and more variable constraints which may make it difficult to solve. Gaussian penalty functions have been used as soft constraints to incorporate waypoint constraints into MPC [27]. However, since arrival times are provided by higher-level planners, this approach still lacks adaptability and an inappropriate time may lead to an unfeasible solution. Recently, a general spatio-temporal optimization framework with geometric constraints, named MINCO [28], has been proposed to address whole-body trajectory generation for quadrotors. It operates by optimizing the positions of intermediate waypoints and the intervals between these points and then solves the complete trajectory. This optimization structure makes it particularly well-suited for incorporating waypoint constraints on these intermediate points, which forms the foundation for our implementation approach.

For most skills, given a target waypoint along with some orientation or velocity constraints, it is generally possible to optimize a feasible trajectory that accomplishes the task. However, for highly dynamic maneuvers, such as parallel grasping or striking on an inclined plane, as shown in Figure 1 (a), directly using a single waypoint constraint may converge to poor suboptimal solutions. For such highly attitude-constrained trajectories, there exist inherent underlying motion patterns that should be leveraged. For example, quadrotors tend to follow arc-shaped trajectory patterns to satisfy their dynamics, as shown in Figure 1 (b), with this tendency becoming more pronounced as the attitude angle increases. Therefore, we propose a novel guided optimization approach for aerial manipulations that initially imposes additional waypoint constraints to steer the trajectory optimization toward desired local optima that better satisfy both task requirements and dynamic feasibility constraints, and then progressively relaxes these auxiliary constraints to allow for further refinement. The question then becomes: how should we properly select these waypoints? Recently, IL has demonstrated significant capabilities in the robot manipulation domain. For instance, Diffusion Policy [29] learns data distributions through imitation learning to achieve policy learning, while OpenVLA [30] and $\pi _ { 0 }$ [31] achieve task generalization through imitation learning on large-scale robot training data. A recent work, Flying Hand [17], is the first to incorporate teleoperation into aerial manipulators, training an ACT [32] policy and achieving some basic tasks. Current mainstream IL approaches typically learn to output singlestep actions or complete trajectories based on environmental information. However, we aim to leverage this capability of imitation learning through a lightweight network that imitates our aerial manipulator’s local behavior near target points when performing large-attitude manipulation, using this information as a prior for guided optimization.

In this paper, we propose a whole-body integrated planning framework for aerial manipulators that incorporates flexible waypoint constraints which can be used to freely combine into many fundamental skills. First, we develop an improved safe flight corridor (SFC) generation strategy that not only expands the feasible solution space but also guarantees the feasible waypoint constraint implementation. Second, we modify the dimension and the dynamics from [28] and then formulate a spatio-temporal optimization problem for aerial manipulators to generate a 6-dimensional trajectory, which includes the quadrotor’s trajectory in the world frame and the manipulator’s trajectory in the local frame. In addition, we improve the constraint formulation from [28], specifically by introducing a varying ellipsoid to model the changing collision volume of the aerial manipulator caused by the motion of the manipulator. Third, we introduce a comprehensive approach for handling waypoint constraints and incorporate various task-specific constraints, including partial or complete position constraints (with individual x, $y , \ z$ axis constraint), velocity, and orientation requirements. Building upon this constraint handling approach, we further introduce the concept of IL-guided optimization that employs IL to generate guide points, which serve as waypoint constraints to improve trajectory convergence. Finally, we validate our framework through comprehensive comparative and ablation studies in simulation, as well as real-world aerial manipulation experiments to demonstrate its effectiveness. The main contributions of this work are:

1) Whole-body Planning Framework: We propose a novel whole-body integrated motion planning framework for aerial manipulators. This framework simultaneously optimizes the trajectory of both the quadrotor and the manipulator to achieve coordinating whole-body motion and complete a series of tasks, strike, grasp, lift, write, pull, push, wind, cross and press. Additionally, we develop a novel varying ellipsoid method to model the changing collision volume to guarantee the whole-body collision-free manipulations under any requirements.

2) Flexible Partial Waypoint Constraint: We introduce a comprehensive waypoint constraint formulation that enables selective specification of position, velocity, or orientation requirements for individual quadrotors or end-effectors at different waypoints. This approach allows flexible task-specific partial waypoint constraints that only constrain the necessary degrees of freedom while leaving others free for optimization, providing significantly greater flexibility and more constraint options compared to existing methods.

3) IL-Guided Optimization for Complex Tasks: We introduce an IL-guided optimization approach to overcome poor local optima in challenging manipulation scenarios, such as grasping or striking on inclined surfaces. The method learns from high-quality demonstration data to generate a sequence of waypoints to guide the optimization process in the first stage, and then progressively relaxes these guided constraints in the second stage to find better solutions.

4) Comprehensive Validation and Open-source Implementation: We demonstrate the versatility and effectiveness of our framework through extensive validation, including diverse benchmark tasks, ablation studies, simulations on different types of manipulators, and realworld aerial manipulation experiments. The complete framework is made publicly available to support the research community and promote reproducible research in aerial manipulation planning.<sup>1</sup>

Our framework overview is depicted in Figure 3. The paper is organized as follows. Section II introduces the notation, system model, and trajectory representation. Section III presents the improved SFC generation strategy, while Section IV elaborates on the basic optimization problem formulation. Section V introduces the implementation of waypoint constraint and ILguided optimization. Section VI validates the effectiveness of the proposed method through simulations and experiments. Finally, conclusions are presented in Section VII.

## II. PRELIMINARIES

## A. Notation

The aerial manipulator system discussed in this paper consists of a quadrotor platform integrated with a 3-DOF parallel delta robot with 3-RSS configuration similar to [33], as illustrated in Figure 2. We experimentally determine the Center of Mass (CoM) of the entire aerial manipulator system by placing the delta arm in its initial state and measuring the resulting CoM, which we denote as $\mathbf { \boldsymbol { c } } _ { m 0 }$ . For convenience, we define three distinct coordinate frames:

1) World Frame $( { \mathcal { F } } _ { \mathcal { W } } ) { : }$ : This frame is denoted by $\{ \pmb { x } _ { \mathcal { W } } , \pmb { y } _ { \mathcal { W } } , \pmb { z } _ { \mathcal { W } } \}$ , where $z _ { \mathcal { W } }$ points upwards, opposing the direction of gravity.

2) Body Frame $( \mathcal { F } _ { B } ) \mathrm { : }$ : This frame is represented by $\{ \boldsymbol { x } _ { B } , \boldsymbol { y } _ { B } , z _ { B } \}$ . The origin of this frame is located at $c _ { m 0 }$ In this frame, $_ { \pmb { x } _ { B } }$ points forward and $_ { z _ { B } }$ aligns with the total thrust generated by the rotors.

3) Delta Frame $( \mathcal { F } _ { \mathcal { D } } ) \colon$ This frame is specified as $\{ \pmb { x } _ { \mathcal { D } } , \pmb { y } _ { \mathcal { D } } , \pmb { z } _ { \mathcal { D } } \}$ and is related to the body frame $\mathcal { F } _ { B }$ through a translational transformation only.

Unless explicitly stated otherwise, the pre-superscript of a vector indicates the frame in which the vector is expressed. For conciseness, when a vector is expressed in the world frame, the pre-superscript will be omitted. For instance, ${ \mathbf { } } p _ { b }$ represents the position of the CoM of the aerial manipulator, in $\mathcal { F } _ { \mathcal { W } }$ while ${ { \mathbf { \mathit { \phi } } } ^ { D } } { \mathbf { \mathit { p } } } _ { e }$ denotes the position of the end-effector in $\mathcal { F } _ { \mathcal { D } }$ Analogously, the matrix $\mathbf { R } _ { B }$ represents the rotation from $\mathcal { F } _ { \mathcal { W } }$ to $\mathcal { F } _ { B }$

![](images/a217cd609ed5978bc2af2d2f615e77604e67b84d21df81c7968af13bccd84ec0.jpg)  
Fig. 3. Overview of the proposed integrated motion planning framework for aerial manipulation. The framework consists of seven key components: (a) Prior Information module that employs a Diffusion Transformer to generate trajectory tendencies and processes user-defined waypoint constraints; (b) Quadrotor module integrating motion capture (Mocap), Extended Kalman Filter (EKF), and IMU for robust state estimation; (c) Delta Arm module handling servo control and forward kinematics for Cartesian space state computation; (d) Safe Flight Corridor (SFC) Construction module for path searching, active polyhedron generation, and conventional SFC construction; (e) Optimization Framework addressing waypoint constraints, collision avoidance, and dynamic/kinematic feasibility; (f) IL-guided two-stage optimization where guide points serve as hard constraints in the initial stage and are subsequently released for free optimization; (g) Trajectory Generation module producing smooth polynomial trajectories; and (h) Skills module performing various basic manipulation skills

T $\pmb { p } _ { f } = \left[ \pmb { p } _ { b , f } ^ { \top } , \mathcal { D } \pmb { p } _ { e , f } ^ { \top } \right] ^ { \top } \in$ Let p<sub>o</sub> = hp<sup>⊤</sup><sub>b,o</sub>, <sup>D</sup>p<sup>⊤</sup><sub>e,o</sub> ∈ R<sup>6</sup> and $\mathbb { R } ^ { 6 }$ denote the origin and final positions of the aerial manipulator in $\mathcal { F } _ { \mathcal { W } }$ and end-effector in $\mathcal { F } _ { \mathcal { D } }$ , respectively. We define a waypoint constraint set in frame $\mathcal { F } _ { \mathcal { W } }$ that consists of arbitrary combinations of two types of waypoint constraints:

$$
\mathbf {W} _ {\text { cons }} := \{\boldsymbol {w} _ {\lambda} \mid \lambda \in \{1, \dots , n _ {\Lambda} \} \}\tag{1}
$$

The two types of waypoint constraints are defined as:

$$
\boldsymbol {w} _ {\lambda} := \left\{ \begin{array}{l l} \boldsymbol {p} _ {b, \lambda} ^ {\text { cons }}, & \text { if   } \lambda \in \Lambda_ {b} \\ \boldsymbol {p} _ {e, \lambda} ^ {\text { cons }}, & \text { if   } \lambda \in \Lambda_ {e} \end{array} \right.\tag{2}
$$

where $\Lambda _ { b }$ and $\Lambda _ { e }$ are disjoint index sets with $\Lambda _ { b }$ ∪ $\begin{array} { l l } { \Lambda _ { e } } & { = } \end{array}$ $\{ 1 , \dots , n _ { \Lambda } \}$ , representing quadrotor and end-effector waypoint constraints, respectively.

## B. Modeling

We use Newton-Euler equations to model the whole-body dynamics. Following [34], we treat the aerial manipulator as a rigid body system and focus on quasi-static forces from manipulator motion and environmental interactions. This approach is reasonable as the delta arm’s mass represents only a small fraction of the total system mass [35], and we will strictly limit the manipulator’s velocity during planning. For instantaneous actions such as grasping and striking, we cannot model these transient disturbances, but we do model the environmental external forces, that occur during sustained operations like pulling and holding grasped objects.

1) Aerial Manipulator Dynamics: The coupled dynamics of the aerial manipulator system are described by:

$$
\dot {\pmb {p}} _ {b} = \pmb {v} _ {b}\tag{3}
$$

$$
\dot {\pmb {v}} _ {b} = \frac {1}{m _ {c}} \mathbf {R} _ {\mathcal {B}} (\mathbf {\beta} \pmb {f} + \mathbf {\beta} \pmb {f} _ {e}) - g \pmb {z} _ {\mathcal {W}}\tag{4}
$$

$$
{ } ^ { \mathcal { B } } \dot { \boldsymbol { \omega } } _ { b } = \mathbf { I } _ { c } ^ { - 1 } ( { } ^ { \mathcal { B } } \boldsymbol { \tau } + { } ^ { \mathcal { B } } \boldsymbol { \tau } _ { e } - { } ^ { \mathcal { B } } \boldsymbol { \omega } _ { b } \times \mathbf { I } _ { c } \cdot { } ^ { \mathcal { B } } \boldsymbol { \omega } _ { b } )\tag{5}
$$

$$
\mathbf {I} _ {c} = \mathbf {I} + m _ {e} \cdot \mathrm{diag} (^ {\mathcal {B}} \pmb {p} _ {e} - ^ {\mathcal {B}} \pmb {p} _ {e 0}) ^ {2}\tag{6}
$$

where ${ \mathbf { } } p _ { b }$ and ${ \pmb v } _ { b }$ represent the position and linear velocity of the aerial manipulator, respectively. The combined system mass and the end-effector mass are denoted by $m _ { c }$ and $m _ { e }$ respectively. $^ B f$ and $s _ { \tau }$ are the thrust and torque generated by the quadrotor, while $^ B f _ { e }$ and $s _ { \tau _ { e } }$ are the force and moment generated by the end-effector. $\mathbf { I } _ { c }$ is the combined inertia, where $\mathbf { I } = \mathrm { d i a g } ( I _ { x } , I _ { y } , I _ { z } )$ is the inertia tensor of the aerial manipulator with the arm at its initial state. Here, $^ B p _ { e 0 }$ and $^ B p _ { e }$ denote the initial and current positions of the end-effector in the body frame $\mathcal { F } _ { B }$ , respectively.

Now we consider the force ${ \boldsymbol { \mathsf { s } } } _ { f _ { e } }$ and moment $s _ { \tau _ { e } }$ caused by the end-effector:

$$
{ } ^ { \mathcal { B } } \boldsymbol { f } _ { e } = { } ^ { \mathcal { B } } \mathbf { R } _ { \mathcal { W } } \boldsymbol { f } _ { \mathrm{ext} }\tag{7}
$$

$$
{ } ^ { \mathcal { B } } \pmb { \tau } _ { e } = { } ^ { \mathcal { B } } \pmb { p } _ { e } \times { } ^ { \mathcal { B } } \pmb { f } _ { e } + ( { } ^ { \mathcal { B } } \pmb { p } _ { e } - { } ^ { \mathcal { B } } \pmb { p } _ { e 0 } ) \times ( { } ^ { \mathcal { B } } \mathbf { R } _ { \mathcal { W } } \cdot m _ { e } g \pmb { z } _ { \mathcal { W } } )\tag{8}
$$

where $\pmb { f } _ { \mathrm { e x t } }$ is the external forces applied to the end-effector, for example, the gravitational force when grasping an object, or the frictional force when pulling or pushing.

2) Delta arm Kinematics: Our quadrotor incorporates a 3- DOF delta arm mounted directly beneath the vehicle. This configuration offers significant advantages in terms of lightweight construction, cost-effectiveness, and operational speed, making it widely adopted in recent research [2], [35]. The kinematic model for the delta arm is formulated as follows:

$$
\pmb {p} _ {e} = \mathbf {R} _ {\mathcal {B}} (\mathbf {\Sigma} ^ {\mathcal {B}} \pmb {p} _ {\mathcal {D}} + \mathbf {\Sigma} ^ {\mathcal {D}} \pmb {p} _ {e}) + \pmb {p} _ {b}\tag{9}
$$

$$
\pmb {v} _ {e} = [ \pmb {\omega} _ {\mathcal {B}} ] ^ {\times} \mathbf {R} _ {\mathcal {B}} (\mathbf {\Phi} ^ {\mathcal {B}} \pmb {p} _ {\mathcal {D}} + \mathbf {\Phi} ^ {\mathcal {D}} \pmb {p} _ {e}) + \mathbf {R} _ {\mathcal {D}} \mathbf {\Phi} ^ {\mathcal {D}} \pmb {v} _ {e} + \pmb {v} _ {b}\tag{10}
$$

where ${ \pmb p } _ { e }$ and ${ \pmb v } _ { e }$ represent the position and velocity of the endeffector in frame $\mathcal { F } _ { \mathcal { W } }$ , and $[ \cdot ] ^ { \times }$ denotes the skew-symmetric matrix operator.

Since the above representation operates in Cartesian space, inverse kinematics (IK) must be employed to establish the mapping between the end-effector position ${ { \mathit { \Sigma } } ^ { { \mathcal { D } } } } _ { { \pmb { p } } _ { e } }$ and the joint angles $\textbf {  { q } } = \ \left[ q _ { 1 } , q _ { 2 } , q _ { 3 } \right] ^ { \top } \ \in \ \mathbb { R } ^ { 3 }$ . In this formulation, $L _ { u }$ and $L _ { l }$ denote the lengths of the upper arm and lower arm, respectively, while $r _ { s }$ represents the circumradius of the static platform and $r _ { d }$ represents the circumradius of the end-effector. Following the methodology outlined in [36], the relationship between the end-effector position in the delta frame $^ D p _ { e }$ and the joint angles $\pmb q$ is obtained by solving the following equation:

$$
\left\| \mathbf {\Phi} ^ {\mathcal {D}} \pmb {p} _ {e} - \mathbf {\Phi} ^ {\mathcal {D}} \mathbf {R} _ {i} \cdot \left[ \begin{array}{c} r _ {s} + L _ {u} \sin q _ {i} \\ 0 \\ - L _ {u} \cos q _ {i} \end{array} \right] \right\| _ {2} ^ {2} = L _ {l} ^ {2},\tag{11}
$$

where $i \in \{ 1 , 2 , 3 \}$ , and ${ \mathcal { D } } _ { \mathbf { R } _ { i } }$ denotes the z-axis rotation matrix by $\{ 0 , { \frac { 2 } { 3 } } \pi , { \frac { 4 } { 3 } } \pi \}$ radians, respectively. By differentiating both sides of the above equation with respect to time and rearranging the terms, we obtain:

$$
\dot {\pmb {q}} = \textbf {J} ^ {\mathcal {D}} \dot {\pmb {p}} _ {e},\tag{12}
$$

where J denotes the Jacobian matrix. The explicit expression for this matrix is detailed in [36].

## C. Trajectory Representation

In this study, we adopt a hybrid coordinate representation for trajectory planning: the quadrotor’s trajectory is planned in the frame $\mathcal { F } _ { \mathcal { W } }$ while the end-effector trajectory is planned in the frame $\mathcal { F } _ { \mathcal { D } }$ . This definition offers two key advantages: (1) it enables better capture of the end-effector’s motion relative to the quadrotor, facilitating accurate collision volume modeling and workspace constraint enforcement; (2) when no manipulation tasks are required, the system can directly operate as a standard quadrotor without representation switching.

Based on this design, we utilize 6-dimensional spatiotemporal polynomial trajectories that decompose multisegment trajectories into waypoints and their corresponding time intervals. We denote the trajectory as:

$$
\mathfrak {T} = \{\boldsymbol {p} (t): [ 0, T _ {i} ] \mapsto \mathbb {R} ^ {m} \mid \mathbf {C} = C (\mathbf {q}, \mathbf {T}), i = 1, \dots , M \}\tag{13}
$$

where $m \ = \ 6$ represents the dimension of the trajectory. $\mathbf { C } = \left\lceil \mathbf { c } _ { 1 } ^ { \top } , \cdot \cdot \cdot , \mathbf { c } _ { M } ^ { \top } \right\rceil ^ { \top } \in \mathbb { R } ^ { 2 M s \times 6 }$ denotes the polynomial coefficient matrix, where M represents the number of segments in the polynomial trajectory and s represents the order of the integrator chain for control effort. $\mathbf { q } = \left[ \pmb { q } _ { 1 } , \cdots , \pmb { q } _ { M - 1 } \right] ^ { \top } \in$ $\mathbb { R } ^ { ( M - 1 ) \times 6 }$ represents the intermediate points, where the first 3 dimensions correspond to the quadrotor and the last 3 dimensions correspond to the end-effector. $\mathbf { T } = \left[ T _ { 1 } , \cdots , T _ { M } \right] ^ { \top } \in$ $\mathbb { R } _ { > 0 } ^ { M }$ represents the non-negative time vector. $C ( \mathbf { q } , \mathbf { T } )$ denotes the parameter mapping constructed from Theorem 2 in [28].

For the i-th segment of the trajectory, the position at time t is defined as:

$$
\boldsymbol {p} _ {i} (t) = \mathbf {c} _ {i} ^ {\top} \boldsymbol {\beta} (t), \quad t \in [ 0, T _ {i} ],\tag{14}
$$

where $i \in \{ 1 , 2 , \ldots , M \}$ , and $T _ { i }$ represents the duration of the i-th segment. The polynomial coefficient matrix for the i-th segment is denoted by $\mathbf { c } _ { i } \in \mathbb { R } ^ { 2 s \times 6 }$ , and the basis vector function $\beta ( x )$ is defined as $\beta ( x ) ~ = ~ \left[ 1 , x , \cdot \cdot \cdot , x ^ { 2 s - 1 } \right] ^ { \top } \in$ $\mathbb { R } ^ { 2 s }$

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1 Active SFC Generation
1: Input: path, $\{p_{x,\lambda}^{\text{cons}}\}_{\lambda=1}^{n_{\Lambda}}$ where $x \in \{b, e\}$
2: Output: $S$ // $S$: Corridor
3: $S \leftarrow \emptyset$, $Q \leftarrow \emptyset$ // $Q$: Polyhedron queue
4: for $\lambda = 1$ to $n_{\Lambda}$ do
5: $(P_1, P_2) \leftarrow \text{ACTIVEGENERATE}(p_{x,\lambda}^{\text{cons}})$
6: $Q.\text{push}(P_1)$, $Q.\text{push}(P_2)$
7: end for
8: $i \leftarrow 1$
9: while $i &lt; |\text{path}|$ do
10:    for $j = i + 1$ to $|\text{path}|$ do
11:    if ISFARTHESTPOINT(path[i], path[j]) then
12:    $S.\text{push(GENERATE(path[i], path[j]))}$
13:    $i \leftarrow \text{LASTPOINTINPOLY(path, } S[-1])$, break
14:    end if
15:    if $Q \neq \emptyset$ and INSIDEPOLY($Q[0]$, path[j]) then
16:    $S.\text{push(Q.pop())}$, $S.\text{push(Q.pop())}$
17:    $i \leftarrow \text{LASTPOINTINPOLY(path, } S[-1])$, break
18:    end if
19:    end for
20: end while
21: return $S$
</div>

## III. ACTIVE SFC GENERATION

Given the constrained waypoint positions $p _ { b , \lambda } ^ { \mathrm { c o n s } }$ and $p _ { e , \lambda } ^ { \mathrm { c o n s } }$ $( \lambda ~ \in ~ \{ 1 , 2 , \cdot \cdot \cdot ~ , n _ { \Lambda } \} )$ , our objective is to generate a safe and feasible trajectory for the aerial manipulator while satisfying all waypoint constraints. However, according to [37], waypoints should intersect with non-empty regions formed by the intersection of two consecutive polyhedra. To achieve this objective, we propose a modified SFC generation strategy that combines multi-stage path searching with an active polyhedron generation method at constrained points, which not only enables waypoint constraint implementation but also enlarges the solution space. The complete algorithm is presented in Algorithm 1.

![](images/33deed3bf00e9deaae05b384d3b010301805d3480a4ba238856bb94283cc305e.jpg)  
Fig. 4. Active polyhedrons generation algorithm: (a) hollow spaces from SFC generation on inclined surfaces; (b) computation of averaged points ${ \pmb p } _ { 1 } ^ { \mathrm { a v g } }$ and $\pmb { p } _ { 2 } ^ { \mathrm { a v g } }$ from path points; (c) projection onto auxiliary planes yielding $p _ { 1 } ^ { \mathrm { p r o j } } .$ $p _ { 2 } ^ { \mathrm { p r o j } }$ and extensions $p _ { 1 } ^ { \mathrm { e x t } } , p _ { 2 } ^ { \mathrm { e x t } }$ ; (d) polyhedron generation using constrained waypoints with points $\bar { p } _ { 1 } ^ { \mathrm { e x t } }$ and $\pmb { p } _ { 2 } ^ { \mathrm { e x t } }$ for compact adherence to the surface.

## A. Active Polyhedron Generation

Prior to the SFC construction, we leverage existing path searching algorithms JPS [38] to obtain a reference path that sequentially traverses through all waypoints $p _ { b , \lambda } ^ { \mathrm { c o n s } }$ and $\pmb { p } _ { e , \lambda } ^ { \mathrm { c o n s } }$ from the start to the end. Traditional construction methods [37], [39] directly generate a sequence of continuously intersecting polyhedrons by identifying and linking the farthest unblocked points along the reference path. However, obstacle occlusion and environmental factors introduce significant uncertainties during the polyhedron generation process. These uncertainties make it difficult to accurately locate intersection regions and can lead to reduced feasible solution spaces, such as hollow areas that may form between surfaces and polyhedrons, as illustrated in Figure 4 (a). To address the challenge, we develop an active polyhedron generation strategy that extends the polyhedron around constrained waypoints of interest, inspired by the method proposed in [40].

As illustrated in Figure 4 (a), we define an arbitrary and winding purple path that orderly connects the start point ${ \displaystyle p _ { b , o } , }$ several constrained waypoints $( p _ { b , \lambda } ^ { \mathrm { c o n s } }$ or $p _ { e , \lambda } ^ { \mathrm { c o n s } } )$ ), and the end point $\textstyle p _ { b , f } .$ For clarity, we focus on the situation at one of the constrained points. According to [37], the convex polyhedron generation is strongly related to the input line segment, obstacle distribution, and the bounding box configuration. This relationship naturally leads us to consider a key geometric constraint: constructing two line segments that connect the constrained waypoint to its previous and subsequent points can guarantee that the two corresponding polyhedrons form a non-empty intersection region containing the constrained waypoint. More crucially, if the expanding ellipsoid’s x-axis is aligned parallel to the inclined plane, the bounding box can be optimally oriented to maximize coverage of the available space on the slope, thereby enlarging the feasible solution space for trajectory optimization. To determine the directions towards and away from the constrained points, we calculate the average of a given number of points before and after the constrained point along the path, denoted as $\pmb { p } _ { 1 } ^ { \mathrm { a v g } }$ and $ { p _ { \mathrm { 2 } } } ^ { \mathrm { a v g } }$ respectively, as shown in Figure 4 (b). These points are then projected onto an auxiliary plane parallel to the previous plane but passing through the constrained waypoint, resulting in two projected vectors $p _ { 1 } ^ { \mathrm { p r o j } }$ and $p _ { 2 } ^ { \mathrm { p r o j } }$ as illustrated in Figure 4 (c). After normalizing and scaling these vectors by $r _ { p } ,$ we obtain two new generation points, $\pmb { p } _ { 1 } ^ { \mathrm { e x t } }$ and $\pmb { p } _ { 2 } ^ { \mathrm { e x t } }$ . By connecting the constrained point to these generation points, we can directly generate two polyhedrons that are tightly fitted to the inclined plane, as illustrated in Figure 4 (d). These constrained polyhedrons are then stored in a temporary queue Q. Additionally, we define two jump points, $p _ { 1 } ^ { \mathrm { j u \bar { m } p } }$ and $\bar { p } _ { 2 } ^ { \mathrm { { j u m p } } }$ which are the first and last points along the path within these two polyhedrons respectively. Their specific applications will be discussed in the following subsection.

## B. SFC Generation

After generating two polyhedrons for each constrained point as described above, the complete SFC is constructed through an iterative process along the path. During this iteration, when the current point reaches a predefined threshold distance from the last point, the algorithm pauses to generate a new polyhedron and then resumes iteration from the last point within this newly generated polyhedron. When the iterative point reaches $p _ { 1 } ^ { \mathrm { j u \bar { m } p } }$ , the iteration immediately pauses, and the two constrained polyhedrons are popped from the temporary queue Q and incorporated into the SFC. The iteration then resumes from the corresponding $p _ { 2 } ^ { \mathrm { i u m p } }$ and continues the standard generation process. Ultimately, we obtain the SFC in H-representation as a series of consecutively intersected polyhedrons [41]:

$$
\begin{array}{l} \mathcal {S} := \bigcup_ {i = 1} ^ {M} \mathcal {H} _ {i} \subset \mathbb {R} ^ {3} \\ \mathcal {H} _ {i} := \left\{\boldsymbol {x} \in \mathbb {R} ^ {3} \mid \mathbf {A} _ {\boldsymbol {i}} \boldsymbol {x} \preceq \boldsymbol {b} _ {\boldsymbol {i}} \right\} \end{array}\tag{15a}
$$

(15b)

where M is the total number of polyhedrons in the SFC.

## IV. BASIC PROBLEM FORMULATION

This section presents our basic optimization problem formulation for aerial manipulators, which guarantees trajectory smoothness, safety, and dynamic and kinematic feasibility without considering task-specific waypoint constraints. We begin by formulating a comprehensive objective function that balances control effort and time consumption metrics. The objective function is subject to multiple constraint categories: collision-free constraints implemented through a varying ellipsoid, and maximum velocity, acceleration, thrust, body rate, and workspace constraints.

## A. Objective Function

We focus on generating a 6-dimensional trajectory $\mathbf { \nabla } _ { \mathbf { } ^ { p } } ( t ) =$ $[ { \pmb p } _ { b } ^ { \top } ( t ) , { \pmb \mathcal { D } } { \pmb p } _ { e } ^ { \top } ( t ) ] ^ { \top } \ \in \ \mathbb { R } ^ { 6 }$ for an aerial manipulator. As discussed in Section II-C, we decompose the complete trajectory into intermediate points and intervals. For convenience, we separate these intermediate points for the quadrotor and the manipulator into:

$$
\mathbf {P} = \left[ \pmb {p} _ {b, 0}, \pmb {p} _ {b, 1}, \dots , \pmb {p} _ {b, M} \right] ^ {\top} \in \mathbb {R} ^ {(M + 1) \times 3},\tag{16}
$$

$$
\mathbf {Q} = \left[ ^ {\mathcal {D}} \boldsymbol {p} _ {e, 0}, ^ {\mathcal {D}} \boldsymbol {p} _ {e, 1}, \dots , ^ {\mathcal {D}} \boldsymbol {p} _ {e, M} \right] ^ {\top} \in \mathbb {R} ^ {(M + 1) \times 3}.\tag{17}
$$

Accordingly, the coefficient matrix is also decomposed as $\mathbf { C } =$ $[ \mathbf { c } _ { b } , \mathbf { c } _ { e } ]$ . Combined with the previously mentioned time vector T, we can formulate the optimization problem as:

$$
\min _ {\mathbf {P}, \mathbf {Q}, \mathbf {T}} \mathcal {J} = \int_ {0} ^ {T _ {\sigma}} \| \pmb {p} ^ {(s)} (t) \| ^ {2} \mathrm{d} t + \rho T _ {\sigma}\tag{18a}
$$

$$
\mathrm{s.t.} \quad \mathcal {G} (\boldsymbol {p} (t), \dots , \boldsymbol {p} ^ {(s)} (t)) \preceq \mathbf {0}, \forall t \in [ 0, T _ {\sigma} ]\tag{18b}
$$

$$
\pmb {p} _ {1} ^ {[ s - 1 ]} (0) = \mathbf {p} _ {o}, \quad \pmb {p} _ {M} ^ {[ s - 1 ]} (T _ {M}) = \mathbf {p} _ {f}\tag{18c}
$$

$$
T _ {i} > 0, \quad i \in \{1, 2, \dots , M \}\tag{18d}
$$

where $\begin{array} { r } { T _ { \sigma } ~ = ~ \sum _ { i = 1 } ^ { M } T _ { i } } \end{array}$ is the total time consumption with regularization factor $\rho > 0$ , and $\mathcal { G }$ denotes the inequality constraints over $[ 0 , T _ { \sigma } ]$ . The notation $p ^ { [ s - 1 ] }$ represents the multi-order differentials matrix $[ \pmb { p } , \pmb { \dot { p } } , \dots , \pmb { p } ^ { ( s - 1 ) } ] \ \in \ \mathbb { R } ^ { 6 \times s }$ Let $\mathbf { p } _ { o } , \mathbf { p } _ { f } \in \mathbb { R } ^ { 6 \times s }$ denote the initial and terminal conditions for the aerial manipulator in $\mathcal { F } _ { \mathcal { W } }$ and the end-effector in $\mathcal { F } _ { \mathcal { D } }$ $T _ { i }$ denotes the duration of the i-th segment, and M is the total number of segments.

The inequality constraints in Equation (18b) encompass multiple aspects, including safety, dynamic, kinematic and attitude constraints. We adopt a penalty-based approach to handle these constraints efficiently by incorporating them into the objective function with appropriate weights, thereby transforming the original constrained optimization problem into an unconstrained one. To reduce the computational complexity, we approximate the continuous integral using discrete numerical integration:

$$
\int_ {0} ^ {T _ {\sigma}} J _ {\mathcal {G}} \mathrm{d} t \approx \sum_ {i = 1} ^ {M} \sum_ {n = 0} ^ {N - 1} \frac {T _ {i}}{N} J _ {\mathcal {G}} ^ {i} (\frac {n T _ {i}}{N})\tag{19}
$$

where $J _ { \mathcal { G } } ^ { i }$ represents the soft penalty terms for the i-th segment, and $N$ denotes the number of discretization points per segment.

## B. Control Effort and Total Time Penalty

We define the control effort penalty $J _ { c }$ as the integral of the squared third-order derivatives (i.e., minimum jerk) for both the aerial manipulator and the end-effector. The total control effort is given by:

$$
J _ {c} = \int_ {0} ^ {T _ {\sigma}} \left\| \boldsymbol {p} ^ {(3)} (t) \right\| ^ {2} \mathrm{d} t = \sum_ {i = 1} ^ {M} \int_ {0} ^ {T _ {i}} \left\| \boldsymbol {p} _ {i} ^ {(3)} (t) \right\| ^ {2} \mathrm{d} t.\tag{20}
$$

In addition, to ensure trajectory efficiency, we incorporate a time penalty term $J _ { t } ,$ , defined as the total trajectory duration:

$$
J _ {t} = T _ {\sigma} = \sum_ {i = 1} ^ {M} T _ {i},\tag{21}
$$

where $T _ { i }$ represents the duration of the i-th trajectory segment.

![](images/ce1bd7a7a48ac4983f3127dd3b3b78f5558eae78f4e42aceec8e9f8987c2866d.jpg)  
Fig. 5. Visualization of the safety penalty mechanism: The aerial manipulator achieves whole-body collision avoidance by optimizing its configuration through orientation adjustment and delta arm extension, while maintaining task constraints at specified waypoints within the SFC limits.

## C. Safety Penalty

We extend the approaches proposed by [41] and [37] by modeling the entire aerial manipulator system as a varying ellipsoid during task execution. The ellipsoid, designed with fixed semi-major axes of length $r _ { e }$ in the xy-plane, incorporates a variational function $h ( t )$ to dynamically adjust its length along the z-axis, accounting for the manipulator’s motion range and ensuring complete coverage of the entire system, as shown in Figure 5. The formulation of this ellipsoidal model is given by:

$$
\mathcal {E} (\pmb {\xi}, t) = \mathbf {G} (t) \cdot (\mathbf {R} _ {\mathcal {B}} (t) \pmb {\xi}) + \pmb {p} _ {b} (t), \quad \pmb {\xi} ^ {\top} \pmb {\xi} \leq 1,
$$

$$
\mathbf {G} (t) = \mathrm{diag} (r _ {e}, r _ {e}, h (t)) \in \mathbb {R} _ {> 0} ^ {3 \times 3},\tag{22a}
$$

(22b)

$$
h (t) = \left(^ {\mathcal {D}} \pmb {p} _ {\mathcal {B}} - ^ {\mathcal {D}} \pmb {p} _ {e} (t)\right) ^ {\top} \pmb {e} _ {3},\tag{22c}
$$

where $\pmb { \xi } \in \mathbb { R } ^ { 3 }$ represents any vector within a unit sphere, and E represents the approximated ellipsoid in the world frame $\mathcal { F } _ { \mathcal { W } }$ The matrix $\mathbf G ( t )$ encodes the semi-lengths of the ellipsoid’s three axes along its diagonal. $\boldsymbol { e } _ { 3 } = [ 0 , 0 , 1 ] ^ { \intercal }$ is the unit vector. ${ { ^ D } _ { { p } _ { B } } }$ denotes the translation from $\mathcal { F } _ { \mathcal { D } }$ to $\mathcal { F } _ { B }$

To guarantee safety, we constrain the dynamic ellipsoid to remain within the SFC using the aforementioned methodologies, as shown in Figure 5. This containment constraint can be mathematically expressed as:

$$
\mathcal {E} \subset \mathcal {S}\tag{23}
$$

where $s$ is the space defined in Equation (15a). To achieve this, we define the following penalty functions:

$$
J _ {s} = \sum_ {i = 1} ^ {M} \sum_ {n = 0} ^ {N - 1} \sum_ {k = 1} ^ {K _ {i}} \frac {T _ {i}}{N} \mathcal {K} \left(\mathcal {P} _ {s, i, n, k}\right),\tag{24a}
$$

$$
\mathcal {P} _ {s, i, n, k} = \left(\pmb {p} _ {i, n} ^ {k} (\tau) - \hat {\pmb {p}} _ {i} ^ {k}\right) ^ {\top} \hat {\pmb {n}} _ {i} ^ {k} + d _ {s}\tag{24b}
$$

where $\begin{array} { r } { \tau \ = \ \frac { n T _ { i } } { N } } \end{array}$ , a definition that holds for all subsequent derivations. Here, $\kappa ( \cdot )$ denotes the cubic smooth function defined as $K ( x ) \ = \ \operatorname* { m a x } ( x ^ { 3 } , 0 )$ For the i-th polyhedron containing $K _ { i }$ half-planes, $\hat { p } _ { i } ^ { k }$ represents an arbitrary point on the k-th half-plane with $\hat { \pmb { n } } _ { i } ^ { k }$ as its outward-pointing normal vector, and $d _ { s }$ defines a small safety margin. The term $\pmb { p } _ { i , n } ^ { k }$ represents the point on the ellipsoid’s surface where its normal vector is perpendicular to the k-th plane. This point can be solved by inverting the ellipsoidal tangent point equation and be computed as follows:

$$
\left\{ \begin{array}{l} \boldsymbol {p} _ {i, n} ^ {k} (\tau) = \mathbf {R} _ {\mathcal {B}, i} (\tau) \cdot^ {\mathcal {B}} \boldsymbol {p} _ {i, n} ^ {k} (\tau) + \boldsymbol {p} _ {b, i} (\tau), \\ ^ {\mathcal {B}} \boldsymbol {p} _ {i, n} ^ {k} (\tau) = \pm \frac {\mathbf {G} ^ {2} (\tau) \cdot (\mathbf {R} _ {\mathcal {B} , i} ^ {\top} (\tau) \hat {\boldsymbol {n}} _ {i} ^ {k})}{\left\| \mathbf {G} (\tau) \cdot (\mathbf {R} _ {\mathcal {B} , i} ^ {\top} (\tau) \hat {\boldsymbol {n}} _ {i} ^ {k}) \right\| _ {2}}, \\ ^ {\mathcal {B}} \boldsymbol {p} _ {i, n} ^ {k} (\tau) \cdot \left(\mathbf {R} _ {\mathcal {B}, i} ^ {\top} (\tau) \cdot \hat {\boldsymbol {n}} _ {i} ^ {k}\right) \geq 0. \end{array} \right.\tag{25}
$$

Then, we can easily derive the gradient of the safety cost with respect to the optimization variables P, Q, and T.

## D. Kinematics and Dynamics Penalty

To ensure the physical feasibility of the generated trajectories, we need to consider both kinematic and dynamic constraints. The kinematic constraints govern the system’s workspace, velocities, and body rates, while the dynamic constraints regulate the thrust forces.

We first consider the kinematic constraints. For the delta arm to operate safely and effectively, its end-effector must remain within a prescribed workspace. Following [12], we model this workspace as a cuboid in the Cartesian coordinate system, where $\kappa \in \{ x , y , z \}$ denotes any of the three axes. The workspace is bounded by:

$$
\kappa_ {e, \ell} \leq \pmb {e} _ {\kappa} ^ {\top \mathcal {D}} \pmb {p} _ {e} (\tau) \leq \kappa_ {e, u},\tag{26a}
$$

where $\kappa _ { e , \ell }$ and $\kappa _ { e , u }$ denote the lower and upper bounds along axis κ, respectively, and $\mathbf { e } _ { \kappa }$ represents the unit vector along the κ-axis. To enforce these workspace constraints, we introduce the following penalty terms:

$$
J _ {k} = \sum_ {i = 1} ^ {M} \sum_ {n = 0} ^ {N - 1} \sum_ {\kappa} \frac {T _ {i}}{N} \left(\mathcal {K} \left(\mathcal {P} _ {k _ {\ell}, i, n, \kappa}\right) + \mathcal {K} \left(\mathcal {P} _ {k _ {u}, i, n, \kappa}\right)\right),\tag{27a}
$$

$$
\mathcal {P} _ {k _ {\ell}, i, n, \kappa} = \left(\pmb {e} _ {\kappa} ^ {\top \mathcal {D}} \pmb {p} _ {e, i} (\tau)\right) ^ {2} - \kappa_ {e, u} ^ {2},\tag{27b}
$$

$$
\mathcal {P} _ {k _ {u}, i, n, \kappa} = \kappa_ {e, \ell} ^ {2} - \left(\pmb {e} _ {\kappa} ^ {\top \mathcal {D}} \pmb {p} _ {e, i} (\tau)\right) ^ {2}.\tag{27c}
$$

To prevent excessive motion, we also constrain the velocities of both the aerial manipulator and the end-effector. The aerial manipulator’s velocity is limited to an upper bound $v _ { b , u }$ while the end-effector velocity is constrained by $v _ { e , u }$ . These constraints are incorporated through the following penalty terms:

$$
J _ {k, v e l} = \sum_ {i = 1} ^ {M} \sum_ {n = 0} ^ {N - 1} \frac {T _ {i}}{N} \Big (\mathcal {K} \left(\mathcal {P} _ {b, v, i, n}\right) + \mathcal {K} \left(\mathcal {P} _ {e, v, i, n}\right) \Big),\tag{28a}
$$

$$
\mathcal {P} _ {b, v, i, n} = \left\| \dot {\pmb {p}} _ {b, i} (\tau) \right\| ^ {2} - v _ {b, u} ^ {2},\tag{28b}
$$

$$
\mathcal {P} _ {e, v, i, n} = \left\| ^ {\mathcal {D}} \dot {\pmb {p}} _ {e, i} (\tau) \right\| ^ {2} - v _ {e, u} ^ {2},\tag{28c}
$$

where $\dot { p } _ { b , i } ( \tau )$ represents the velocity of the aerial manipulator in $\mathcal { F } _ { \mathcal { W } }$ , and ${ } ^ { \mathcal { D } } \dot { p } _ { e , i } ( \tau )$ represents the velocity of the end-effector in $\mathcal { F } _ { \mathcal { D } }$

As another kinematic constraint, we consider the aerial manipulator’s body rates. Given that yaw angle variations are neglected in our framework, we focus only on the body rates in the xy-plane. The corresponding penalty terms are defined as:

$$
J _ {k, b d r} = \sum_ {i = 1} ^ {M} \sum_ {n = 0} ^ {N - 1} \frac {T _ {i}}{N} \mathcal {K} \left(\mathcal {P} _ {\omega , i, n}\right),\tag{29a}
$$

$$
\mathcal {P} _ {\omega , i, n} = \left\| ^ {\mathcal {B}} \pmb {\omega} _ {x y, i} \right\| ^ {2} - \omega_ {x y, u} ^ {2},\tag{29b}
$$

where ${ } ^ { B } \omega _ { x y , i } = \left[ \omega _ { x } , \omega _ { y } \right] { } ^ { \top }$ represents the body angular rates in the xy-plane, and $\omega _ { x y , u }$ denotes the upper limit of the body rates.

For dynamic constraints, we consider the aerial manipulator’s thrust capabilities. The thrust f must remain within physically realizable bounds:

$$
f _ {\ell} \leq f \leq f _ {u},\tag{30}
$$

where $f _ { \ell }$ and $f _ { u }$ denote the lower and upper thrust limits. We formulate the corresponding penalty terms as:

$$
J _ {d, t h r} = \sum_ {i = 1} ^ {M} \sum_ {n = 0} ^ {N - 1} \frac {T _ {i}}{N} \Big (\mathcal {K} \left(\mathcal {P} _ {f _ {u}, i, n}\right) + \mathcal {K} \left(\mathcal {P} _ {f _ {\ell}, i, n}\right) \Big),\tag{31a}
$$

$$
\mathcal {P} _ {f _ {u}, i, n} = \left\| m _ {c} (\ddot {\pmb {p}} _ {b, i} (\tau) + g \pmb {z} _ {\mathcal {W}}) - \pmb {f} _ {\mathrm{ext}} \right\| ^ {2} - f _ {u} ^ {2},\tag{31b}
$$

$$
\mathcal {P} _ {f _ {\ell}, i, n} = f _ {\ell} ^ {2} - \left\| m _ {c} (\ddot {\pmb {p}} _ {b, i} (\tau) + g \pmb {z} _ {\mathcal {W}}) - \pmb {f} _ {\mathrm{ext}} \right\| ^ {2},\tag{31c}
$$

where $\ddot { p } _ { b , i } ( \tau )$ denotes the aerial manipulator’s acceleration.

The gradients of all kinematics and dynamics penalty terms with respect to the optimization variables P, Q and T can be efficiently computed using the method described in [39], [42].

With the complete formulation of constraints and their corresponding gradients for the aerial manipulator’s basic flight planning, we can solve the optimization problem using L-BFGS to obtain collision-free flight trajectories for the aerial manipulator.

## V. WAYPOINT CONSTRAINTS AND IL-GUIDED OPTIMIZATION

Having established the basic optimization formulation in the previous section, we now turn to the incorporation of taskspecific waypoint constraints, which can be applied to either the quadrotor or the end-effector in the frame $\mathcal { F } _ { \mathcal { W } }$ independently, as shown in Figure 6. These constraints include higherorder velocity and orientation requirements at designated waypoints. Building upon this framework, we further propose an IL-guided optimization approach to improve performance for specific manipulation tasks.

## A. Waypoint Constraints Implementation

Based on our modified SFC generation strategy, each constrained waypoint corresponds to an intersection of two polyhedrons and an intermediate point. We define the mapping

$$
\phi : \{1, \dots , n _ {\Lambda} \} \to \{1, \dots , M - 1 \}\tag{32}
$$

to map constrained waypoint indices to their corresponding intermediate point indices.

1) Quadrotor Waypoint Constraints: Since the quadrotor trajectory is planned in frame $\mathcal { F } _ { \mathcal { W } }$ , we exploit the spatiotemporal trajectory decomposition property by fixing certain waypoints as constants while maintaining others as optimization variables. The quadrotor waypoint constraints are denoted as:

![](images/ce129ee058aeb54a013d60f37e550f4a4e2d626c18bafc6efea38ac8f925ee7a.jpg)  
Fig. 6. Illustration of representative manipulation primitives achieved through waypoint constraints. The figure demonstrates various fundamental skills including Grasp, Strike, Pull, Cross, Push, Lift, and Write. All these manipulation tasks can be effectively accomplished by implementing appropriate waypoint constraints combined with motion restrictions during execution.

$$
\mathbf {P} _ {b} ^ {\text { cons }} := \{\boldsymbol {p} _ {b, \phi (\lambda)} = \boldsymbol {p} _ {b, \lambda} ^ {\text { cons }} \mid \lambda \in \Lambda_ {b} \}.\tag{33}
$$

The modified optimization variables become:

$$
\bar {\mathbf {P}} = \mathbf {P} \setminus \mathbf {P} _ {b} ^ {\mathrm{cons}}\tag{34}
$$

where $\backslash$ denotes set difference. Due to the spatio-temporal trajectory properties, the resulting trajectory is guaranteed to traverse all constrained waypoints in $\mathbf { P } _ { b } ^ { \mathrm { c o n s } }$

2) End-Effector Waypoint Constraints: The end-effector trajectory is planned in $\mathcal { F } _ { \mathcal { D } }$ and cannot be directly fixed like the quadrotor trajectory. Instead, we enforce end-effector waypoint constraints through a soft penalty approach. We have generated polyhedron intersections near each $\pmb { p } _ { e , \lambda } ^ { \mathrm { c o n s } }$ to ensure the corresponding trajectory intermediate points pass through their vicinity. Therefore, using the end-effector’s world frame position from Equation (9), we formulate the penalty term:

$$
J _ {e} = \sum_ {\lambda \in \Lambda_ {e}} \left\| \boldsymbol {p} _ {e, \phi (\lambda)} - \boldsymbol {p} _ {e, \lambda} ^ {\mathrm{cons}} \right\| ^ {2}.\tag{35}
$$

## B. Task-Specific Constraints

1) Axis-Wise End-Effector Motion Constraint: When executing specific manipulation skills such as pulling, pushing, and writing tasks, the aerial manipulator’s end-effector must maintain its position parallel to a defined plane [43]. More restrictively, operations involving sliding mechanisms require the end-effector to remain constrained to a line throughout the motion. These geometric constraints pose significant challenges for trajectory planning, as conventional point-to-point planning methods cannot guarantee such continuous spatial restrictions.

To address these requirements, we introduce axis-wise endeffector position motion constraints that enforce planar or linear motion during designated trajectory segments. The constraints are formulated as follows:

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 2 IL-guided Optimization
1: Input: $p_o, p_f, p_{\text{inc}}, o_{\text{inc}}, \epsilon_{\text{loose}}, \epsilon_{\text{strict}}$
2: Output: $p(t), t \in [0, T_\sigma]$
3: $a \leftarrow$ IMITATIONLEARNING($p_{\text{inc}}, o_{\text{inc}}, p_o$)
4: $w_{\text{guide}} \leftarrow$ SAMPLEPOINTS($a$)
5: $w_{\text{task}} \leftarrow$ TASKSPECIFIC($p_{\text{inc}}, o_{\text{inc}}$)
6: $x_1 \leftarrow$ INIT($p_o, [w_{\text{guide}}, w_{\text{task}}], p_f$)
7: $x_2 \leftarrow$ OPTIMIZATION($x_1, \epsilon_{\text{loose}}, [w_{\text{guide}}, w_{\text{task}}]$)
8: // Release the IL guided constraints
9: $x_3 \leftarrow$ OPTIMIZATION($x_2, \epsilon_{\text{strict}}, w_{\text{task}}$)
10: return POLYNOMIAL($x_3$)
</div>

$$
J _ {p e} = \sum_ {i = 1} ^ {M} \mathcal {I} _ {p} (i) \sum_ {n = 0} ^ {N - 1} \frac {T _ {i}}{N} \| \boldsymbol {c} _ {p} ^ {\top} (\boldsymbol {p} _ {e, i} (\tau) - \boldsymbol {p} _ {\mathrm{des}, \lambda}) \| ^ {2}\tag{36}
$$

where ${ \cal T } _ { p } ( i ) \in \{ 0 , 1 \}$ that activates constraints for specific trajectory segments, and $c _ { p } \in \mathbb { R } ^ { 3 }$ is a user-defined binary vector where setting any dimension to 1 indicates that the corresponding coordinate should be constrained. These axiswise motion constraints are typically employed in conjunction with two end-effector waypoint constraints to ensure that the constrained dimensions maintain identical values between consecutive waypoints.

2) Velocity Constraint: In certain manipulation tasks, the end-effector’s velocity must be constrained to ensure precise operation. For instance, during object grasping maneuvers, the end-effector may need to maintain zero velocity or a small downward velocity along the $_ { z _ { B } }$ to achieve stable contact.

The velocity constraint is formulated as:

$$
J _ {v e} = \sum_ {\lambda \in \Lambda_ {e}} \mathcal {I} _ {v} (\lambda) \| \boldsymbol {c} _ {v} ^ {\top} (\dot {\boldsymbol {p}} _ {e, \phi (\lambda)} - \dot {\boldsymbol {p}} _ {\mathrm{des}, \lambda}) \| ^ {2}\tag{37}
$$

where ${ \mathcal { Z } } _ { v } ( \lambda ) \ \in \ \{ 0 , 1 \}$ is a binary selector that activates velocity constraints for waypoint λ, $\boldsymbol { c } _ { v } \in \mathbb { R } ^ { 3 }$ is a userdefined binary vector specifying which velocity components to constrain, and $\dot { p } _ { \mathrm { d e s } , \lambda }$ is the desired end-effector velocity at the constrained waypoint.

3) Orientation Constraint: Since the end-effector in the delta arm configuration shares the same orientation as the quadrotor, orientation constraints can be naturally imposed on the quadrotor’s intermediate waypoints. This inherent coupling eliminates the need to distinguish between quadrotor and endeffector orientations. The orientation constraint is formulated as:

$$
J _ {o e} = \sum_ {\lambda \in \Lambda_ {e}} \mathcal {I} _ {o} (\lambda) \| \boldsymbol {f} _ {\lambda} - \| \boldsymbol {f} _ {\lambda} \| \boldsymbol {o} _ {\mathrm{des}, \lambda} \| ^ {2}\tag{38}
$$

where ${ \mathcal { T } } _ { o } ( \lambda ) ~ \in ~ \{ 0 , 1 \}$ is a binary selector for activating orientation constraints, ${ \pmb f } _ { \lambda } = m _ { c } ( \ddot { \pmb p } _ { b , \phi ( \lambda ) } + g z _ { \mathcal { W } } ) - { \pmb f } _ { \mathrm { e x t } }$ is the actual thrust vector, and $_ { o \mathrm { d e s } , \lambda }$ is the desired unit orientation vector at the λ-th waypoint.

## C. IL Prior

Several complex manipulation tasks requiring extreme orientations, such as inclined surface grasping and striking, present significant challenges for trajectory optimization. These tasks involve inherently conflicting constraints - for example, orientation requirements conflict with maximum body rate limitations, and zero end-effector velocity during grasping conflicts with time penalty objectives. The complexity of these constraint interactions significantly amplifies the nonconvex nature of the optimization problem. Since optimization fundamentally involves balancing competing objectives, poor initial solutions can lead the optimizer to converge to local optima that completely sacrifice task constraints [44].

![](images/a307924b44027b527423c3ab6b9ba219bfabe15d79605c432b2265489bcb4cc2.jpg)  
Fig. 7. Visualization of the IL-guided optimization framework. The process begins with collecting various successful spatio-temporal trajectories from demonstration data. These trajectories are fitted using polynomial approximation to extract motion tendencies. A DiT model learns to reconstruct these motion tendencies through noise embedding, condition embedding, and time embedding. The optimization process consists of two stages: Stage 1 employs guide waypoints as hard constraints; Stage 2 releases these hard constraints, allowing free optimization.

We focus on two tasks requiring aggressive maneuvers: inclined grasping and inclined striking. For these complex tasks, designing simple waypoint constraints ${ \pmb w } _ { \mathrm { t a s k } }$ that contain position $p _ { \mathrm { i n c } }$ and corresponding orientations $o _ { \mathrm { i n c } }$ often results in poor local optima that fail to satisfy overall task requirements. To address this challenge, we analyze successful trajectories from both tasks and discover that they exhibit distinct flight ”tendencies.” For example, striking tasks consistently demonstrate arc-like trajectory patterns. These observed patterns provide valuable priors that can guide more effective flight planning.

Given the recent influence of imitation learning in trajectory planning [17], we employ IL to incorporate the observed trajectory priors into our planning framework. To promote training efficiency and reduce unnecessary input complexity, we adopt a compact observation space $\mathbf { \phi } _ { o } : = \mathbf { \phi } [ p _ { \mathrm { r e l } , x y } , \theta _ { \mathrm { i n c } } ]$ where $p _ { \mathrm { r e l } , x y }$ encodes the relative position of the quadrotor in the horizontal plane with respect to the inclined surface, and $\theta _ { \mathrm { i n c } }$ denotes the surface’s inclination angle, derived from the surface orientation vector $\mathbf { \delta } _ { o \mathrm { i n c } }$ . The action space comprises local polynomial trajectory parameters near the surface, defined as $\pmb { a } : = [ \pmb { c } _ { x } , \pmb { c } _ { y } , \pmb { c } _ { z } ]$ . This polynomial parameterization enables a compact network architecture to learn trajectory ”tendencies” more effectively and stably than approaches based on discrete trajectory points.

Moreover, for analogical observations, multiple reasonable trajectories may exist. A deterministic network modeling $( o \to a )$ risks learning averaged action patterns that may not represent valid individual trajectories [45]. Diffusion models can naturally handle this multi-modality by modeling the conditional probability $p ( { \pmb a } | { \pmb o } )$ . Therefore, we design a Diffusion model to address the multi-modal nature of valid trajectories [46]. The model structure can be seen in Figure 7.

We collect many high-quality, effective single-task trajectories as training data. For each clean action $\mathbf { \Delta } _ { \mathbf { a } ^ { \left( \bar { \mathbf { \Lambda } } \right) } }$ , we perform K Gaussian noise addition steps to obtain noisy action $\mathbf { \pmb { a } } ^ { K }$ :

$$
\boldsymbol {a} ^ {k} = \sqrt {\bar {\alpha} _ {k}} \boldsymbol {a} ^ {0} + \sqrt {1 - \bar {\alpha} _ {k}} \boldsymbol {\epsilon}, \boldsymbol {\epsilon} \sim \mathcal {N} (\mathbf {0}, \mathbf {I}), k = 1, \dots , K\tag{39}
$$

where $\begin{array} { r } { \bar { \alpha } _ { k } = \prod _ { i = 1 } ^ { k } \alpha _ { i } } \end{array}$ and $\alpha _ { k } = 1 - \beta _ { k }$ with $\beta _ { k }$ being the noise schedule. The denoising network $f _ { \theta }$ is trained to predict the noise:

$$
\mathcal {L} (\theta) = \mathbb {E} _ {k, \boldsymbol {a} ^ {0}, \epsilon} \left[ \| \boldsymbol {\epsilon} - f _ {\theta} (\boldsymbol {a} ^ {k}, k, \boldsymbol {o}) \| ^ {2} \right]\tag{40}
$$

During inference, we sample from the learned distribution through the reverse process:

$$
\boldsymbol {a} ^ {k - 1} = \frac {1}{\sqrt {\alpha_ {k}}} \left(\boldsymbol {a} ^ {k} - \frac {\beta_ {k}}{\sqrt {1 - \bar {\alpha} _ {k}}} f _ {\theta} (\boldsymbol {a} ^ {k}, k, \boldsymbol {o})\right) + \sigma_ {k} \boldsymbol {z}\tag{41}
$$

where $\sigma _ { k } = \sqrt { \beta _ { k } }$ and $z \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ . This reverse diffusion process generates diverse trajectory parameters that capture the multi-modal distribution of successful manipulation strategies.

## D. IL-guided Optimization

After learning polynomial trajectories through IL, we leverage the IL prior trajectories as guidance for optimization by incorporating them as quadrotor waypoint constraints. We sample k guide points from the IL prior polynomial trajectory and insert them into Equation (1):

$$
\boldsymbol {w} _ {\text { guide }} := \left\{\boldsymbol {p} _ {\text { guide }, 1}, \dots , \boldsymbol {p} _ {\text { guide }, k} \right\} \subset \mathbf {W} _ {\text { cons }}\tag{42}
$$

where $p _ { \mathrm { g u i d e } , i }$ represents the i-th guide point. These sampled guide points ensure that the aerial manipulator follows the desired trajectory direction, thereby improving success rates to some extent.

However, trajectories constrained to fixed guide points may still be suboptimal due to their rigidity. To address this limitation, we implement a novel two-stage optimization strategy. In the first stage, we enforce guide point constraints with a loose gradient tolerance condition $\epsilon _ { \mathrm { l o o s e } }$ , which ensures that the optimization is guided toward the desired direction during the initial search phase. In the second stage, we remove the guide point constraints and apply strict convergence criteria $\epsilon _ { \mathrm { s t r i c t } } ,$ allowing the optimizer to discover locally superior solutions while maintaining the beneficial trajectory structure established in the first stage. This two-stage approach combines the global guidance from successful demonstrations with local optimization refinement, significantly improving both success rates and trajectory quality for complex manipulation tasks. The complete procedure is detailed in Algorithm 2.

## VI. EXPERIMENTS

This section presents a comprehensive experimental evaluation of the proposed framework across multiple dimensions. We begin by detailing the imitation learning data collection and training procedures for the DiT model. Subsequently, we conduct ablation studies to validate key components: the SFC enlarge strategy and the IL-guided optimization approach. We then benchmark different collision volume approximation methods, comparing static ellipsoid and multi-sphere fitting approaches. Simulation experiments demonstrate the framework’s versatility across various manipulator configurations, including telescopic manipulators, 1-DOF, and 2-DOF robotic arms. Finally, real-world experiments validate the system’s performance through the execution of 9 fundamental manipulation skills: strike, grasp, lift, press, wind, write, cross, push, and pull, confirming the practical applicability and versatility of our integrated approach.

## A. Data Collection and IL Training

For the IL-guided optimization, we focus on grasp and strike skills. Expert trajectories are generated on inclined plane scenarios with angles uniformly sampled from 20<sup>◦</sup> to 60<sup>◦</sup>. Initial robot positions are randomly sampled within annular regions of 1.5m to 3.0m radius centered on the inclined plane, with a task-specific waypoint constraint for the endeffector positioned at 0.05m directly above the plane. Due to low success rates in direct planning, we apply the twostage optimization algorithm similar to Section V-D. However, the guide points are strategically selected through expert knowledge before and after critical strike and grasp phases. Generated trajectories with total optimization cost exceeding $1 \times 1 0 ^ { 5 }$ are discarded to ensure solution quality.

To obtain stable trajectory tendencies within local regions, we extract trajectory segments of ±1s duration around the grasping and striking phases. From each segment, we sample 15 discrete points that are subsequently fitted using a threedimensional fifth-order polynomial $[ c _ { x } , c _ { y } , c _ { z } ]$ . The resulting 15 polynomial coefficients constitute the trajectory tendency representation that serves as training targets for the diffusion model. Through this process, we collected approximately 3,000 high-quality trajectory samples for each skill type (grasping and striking) along with their corresponding polynomial representations. The data collection process requires an average generation time of about 33s per sample, which allows for practical data collection at scale.

The imitation learning component employs a Diffusion Transformer (DiT) architecture specifically designed for trajectory tendency learning. The model utilizes a 128-dimensional embedding space with 8 attention heads across 6 transformer layers, incorporating adaptive layer normalization and crossattention mechanisms for effective condition-context fusion. With approximately 3M parameters, the network maps 3- dimensional input conditions (relative position $p _ { \mathrm { r e l } , x y }$ and inclination angle $\theta _ { \mathrm { i n c } } )$ to 15-dimensional polynomial coefficients representing trajectory tendencies. Training is conducted on an NVIDIA RTX 4090 with 24GB memory for approximately one hour using mixed precision and a learning rate of $3 \times 1 0 ^ { - 4 }$ with cosine annealing.

![](images/b975308ac4e61f8d696bbf6d083e0fbdc7becc60506b92fdcc878bf90e010906.jpg)  
Fig. 8. Comparison of different polyhedron generation methods for inclined surface grasping: (a) original method from [13] without modifications, (b) ablated method with manually constrained intersection above the inclined surface, and (c) our proposed Active Polyhedron Generation method. The pink boxes indicate the locations of adjacent polyhedron intersections.

TABLE I  
QUANTITATIVE COMPARISON OF ACTIVE POLYHEDRON GENERATION METHODS

<table><tr><td>Method</td><td>Utilization Rate (%)</td><td>Final Cost</td><td>Distance (m)</td></tr><tr><td>Ablated</td><td>27.1</td><td> $9.51 \times 10^{5}$ </td><td>0.267</td></tr><tr><td>Proposed</td><td>100.0</td><td> $3.54 \times 10^{3}$ </td><td>0.027</td></tr></table>

## B. Ablation Study

We designed two ablation experiments to validate the effectiveness of our Active Polyhedron Generation and demonstrate the improvement provided by IL-guided Optimization.

1) Active Polyhedron Generation: To verify that our Active Polyhedron Generation not only maintains compatibility with our waypoint constraint framework but also expands the solution space, we designed an inclined surface grasping scenario with three different configurations: (1) the original method from [13] without any modifications; (2) a modified version that ensures the intersection of two polyhedrons exists directly above the inclined surface to enable waypoint constraint application; and (3) our proposed Active Polyhedron Generation method.

The results are shown in Figure 8. In Figure 8 (a), without predetermined intervention points in SFC generation, the resulting SFC exhibits significant randomness. The polyhedron intersections are located on both sides of the inclined surface, making it difficult to effectively apply waypoint constraints. In Figure 8 (b), although we constrained the intersection to exist directly above the inclined surface, substantial unused space remains, particularly near the center of the inclined surface, resulting in the ”hollow” region we mentioned. In contrast, Figure 8 (c) demonstrates our proposed Active Polyhedron Generation, which actively generates polyhedrons that nearly achieve full utilization of the feasible space above the inclined surface.

![](images/28291825bcbddeb1a9cc5bf9c6ae9070726b123e363a4c6577522c4f8a8b132a.jpg)  
Fig. 9. Convergence results for IL-guided optimization variants. Yellow trajectories represent the motion of the quadrotor, while orange trajectories show the end-effectors movement in $\bar { \mathcal { F } _ { \mathcal { W } } } .$ Red points indicate IL-derived guide points, and aerial manipulator orientations at strike instant are shown in the lower-left corners.

Since configuration (1) cannot accommodate waypoint constraints, we conducted quantitative analysis only on configurations (2) and (3). We applied our optimization framework to optimize the grasp skill trajectory, with results shown in the lower panels of Figure 8 (a) and (c), where orange trajectories represent the end-effector path and yellow trajectories represent the quadrotor path. Our proposed method successfully reaches the target, while the ablated method suffers from conflicts between end-effector waypoint constraints and safety constraints due to limited solution space. This leads to failure in reaching the target point, and excessive safety penalties in the vicinity result in significantly degraded trajectory quality.

Table I presents a quantitative comparison of three key metrics. The Utilization Rate measures the percentage of free space coverage within a 0.5m radius sphere around the target point. The Final Cost represents the objective function value after convergence. The Distance quantifies the end-effector positioning error at the waypoint constrained location.

2) IL-guided Optimization: To validate the effectiveness of our IL-guided optimization framework, we conduct a comprehensive ablation study examining strike success rates on inclined surfaces. This study addresses three key questions: whether our IL-guided optimization can improve strike success rates, the rationale for sampling points from polynomial trajectories rather than learning discrete points directly, and the optimal number of sampling points for trajectory guidance.

We design the experimental scenario with varying inclined surfaces and designated target points for strike operations.

![](images/649abc88e9d00a0c95796249e69b6f36f39b6143dc873a18d4db1fae1c35fe83.jpg)  
Fig. 10. Convergence curves comparison across optimization methods. Normalized iteration lengths account for varying convergence requirements. Dashed lines: first-stage optimization; solid lines: second-stage optimization.

A successful strike requires two simultaneous conditions at contact: the aerial manipulator maintains parallel alignment with the inclined surface, and the end-effector remains within 0.03m of the target center. Initial positions are randomly sampled within a 1.5 − 3m annular region around the surface for training, while out-of-distribution (OOD) testing uses a 3 − 4m annular region to evaluate generalization performance. We assume prior knowledge of target location and surface inclination angle, establishing fixed end-effector waypoint constraints and corresponding orientations. Our experimental design comprises six configurations tested across four scenarios: 25<sup>◦</sup>, 40<sup>◦</sup>, and 55<sup>◦</sup> surface angles with in-distribution initial positions, 40 with OOD initial positions.

The Baseline completely removes IL-guided optimization, relying solely on direct trajectory optimization. The second configuration (Learned Points Guidance + Relaxation) trains the IL model to predict four discrete points, maintaining consistency with our training methodology but reducing the projection dimension from 15 to 12. The third approach (Two-Point Guidance, No Relaxation) implements our polynomial trajectory imitation learning, sampling one point before and after strike as guide points with strict convergence criteria. Our primary method (Two-Point Guidance + Relaxation) extends the third configuration by incorporating relaxed convergence criteria and two-stage optimization, removing guide points in the second stage for stricter optimization. Finally, we evaluate Four-Point and Six-Point Guidance + Relaxation variants, differing only in the number of polynomial trajectory sampling points.

Figure 9 illustrates convergence results for challenging cases where initial positions lie behind the inclined surface. The baseline approach in Figure 9 (a) demonstrates complete failure, with the end-effector trajectory (orange) missing the target entirely due to convergence to poor local optima. In contrast, learned point guidance in Figure 9 (b) successfully captures required trajectory tendencies, achieving target strikes. Our two-point sampling approach Figure 9 (c) produces quadrotor trajectories (yellow) that accurately traverse the red guide points, resulting in desired trajectory generation. Both twopoint and four-point guidance Figure 9 (d) and (e) consistently accomplish the task, while six-point guidance Figure 9 (f) suffers from over-constraint, leading to suboptimal solutions. Figure 10 quantitatively demonstrates that all proposed ILguided methods achieve superior local optima compared to baseline optimization, as evidenced by their lower convergence costs. Single-stage optimization outperforms baseline but remains inferior to two-stage approaches, confirming that the relaxation strategy further enhances local optimum extraction by allowing more effective exploration in the first stage followed by refined convergence in the second stage.

TABLE II  
COMPARISON OF ROBOTIC MANIPULATION OPTIMIZATION METHODS ACROSS DIFFERENT ROLL ANGLES

<table><tr><td rowspan="2">Method</td><td rowspan="2">Roll (°)</td><td colspan="2">Performance</td><td colspan="3">Trajectory Quality</td><td colspan="2">Efficiency</td></tr><tr><td>Success Rate (%)</td><td>Cost</td><td>Time (s)</td><td>Length</td><td>Jerk</td><td>Iteration</td><td>Runtime (s)</td></tr><tr><td rowspan="4">Baseline (Complete Ablation)</td><td>25</td><td>100</td><td>2137.4</td><td>3.7</td><td>6.2</td><td>277.7</td><td>1536</td><td>5.1</td></tr><tr><td>40</td><td> $50^†$ </td><td>4541.7</td><td>6.8</td><td>9.6</td><td>281.1</td><td>1726</td><td>6.1</td></tr><tr><td>55</td><td> $0^†$ </td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr><tr><td>40(OOD)</td><td>70</td><td>3703.1</td><td>4.6</td><td>8.5</td><td>488.2</td><td>2619</td><td>7.9</td></tr><tr><td rowspan="4">Learned Points Guidance + Relaxation</td><td>25</td><td>90</td><td>3903.5</td><td>6.6</td><td>8.1</td><td>338.5</td><td> $34306^†$ </td><td>132.4</td></tr><tr><td>40</td><td>70</td><td> $5527.5^†$ </td><td> $10.1^†$ </td><td> $9.7^†$ </td><td>400.8</td><td>37209</td><td> $139.4^†$ </td></tr><tr><td>55</td><td>80</td><td>5127.2</td><td>8.2</td><td>11.1</td><td>886.0</td><td>19653</td><td>86.5</td></tr><tr><td>40(OOD)</td><td>60</td><td>4081.4</td><td>6.2</td><td>8.8</td><td>440.6</td><td> $47118^†$ </td><td> $151.2^†$ </td></tr><tr><td rowspan="4">Two-Point Guidance (No Relaxation)</td><td>25</td><td>100</td><td> $3994.9^†$ </td><td>5.8</td><td>8.0</td><td> $401.5^†$ </td><td>8272</td><td>31.7</td></tr><tr><td>40</td><td>70</td><td>3243.0</td><td>4.7</td><td>8.0</td><td> $579.1^†$ </td><td>10653</td><td>35.4</td></tr><tr><td>55</td><td>100</td><td>4137.0</td><td>6.1</td><td>12.6</td><td> $974.4^†$ </td><td>5590</td><td>21.9</td></tr><tr><td>40(OOD)</td><td>90</td><td>3035.4</td><td>4.9</td><td>9.7</td><td>580.1</td><td>6904</td><td>27.2</td></tr><tr><td rowspan="4">Two-Point Guidance + Relaxation (Ours)</td><td>25</td><td>100</td><td>2568.9</td><td>4.4</td><td>6.8</td><td>317.2</td><td>9364</td><td>34.6</td></tr><tr><td>40</td><td>100</td><td>3584.1</td><td>5.4</td><td>9.0</td><td>563.1</td><td>16357</td><td>50.9</td></tr><tr><td>55</td><td>100</td><td>4612.0</td><td>7.3</td><td> $13.2^†$ </td><td>857.1</td><td>6741</td><td>27.4</td></tr><tr><td>40(OOD)</td><td>100</td><td>2719.2</td><td>4.2</td><td>8.8</td><td> $584.9^†$ </td><td>9566</td><td>35.6</td></tr><tr><td rowspan="4">Four-Point Guidance + Relaxation (Ours)</td><td>25</td><td>80</td><td>2842.3</td><td>4.4</td><td>7.3</td><td>377.8</td><td>22529</td><td>88.2</td></tr><tr><td>40</td><td>70</td><td>3112.3</td><td>5.0</td><td>8.5</td><td>575.8</td><td>37550</td><td>126.2</td></tr><tr><td>55</td><td>70</td><td>4376.9</td><td>6.9</td><td>11.9</td><td>852.5</td><td>17751</td><td>88.4</td></tr><tr><td>40(OOD)</td><td>90</td><td> $4053.7^†$ </td><td> $7.1^†$ </td><td> $11.4^†$ </td><td>446.8</td><td>7460</td><td>34.0</td></tr><tr><td rowspan="4">Six-Point Guidance + Relaxation (Ours)</td><td>25</td><td> $70^†$ </td><td>3838.8</td><td> $6.9^†$ </td><td> $8.6^†$ </td><td>325.7</td><td>33666</td><td> $172.9^†$ </td></tr><tr><td>40</td><td>50</td><td>4250.7</td><td>6.8</td><td>9.1</td><td>519.4</td><td> $56956^†$ </td><td>239.6</td></tr><tr><td>55</td><td>60</td><td> $6097.0^†$ </td><td> $10.5^†$ </td><td>12.6</td><td>737.4</td><td> $23621^†$ </td><td> $146.0^†$ </td></tr><tr><td>40(OOD)</td><td> $30^†$ </td><td>3389.2</td><td>5.9</td><td>9.3</td><td>406.6</td><td>37985</td><td>149.3</td></tr></table>

Bold: Best performance within each roll <sup>†</sup> Worst performance within each roll OOD: Given out-of-distribution initial states

Table II presents comprehensive quantitative analysis across performance metrics, trajectory quality, and computational efficiency. Our Two-Point Guidance + Relaxation achieves 100% success rates across all scenarios, significantly outperforming alternatives, particularly in the challenging 55<sup>◦</sup> case where baseline success rate drops to zero. This demonstrates the critical importance of IL guidance for steep surface manipulation. Two-Point Guidance without relaxation ranks second, while Four-Point and Six-Point Guidance exhibit reduced success rates due to over-constraint in the first optimization stage, where two points already provide sufficient information for trajectory guidance, and additional points excessively constrain the trajectory during the first stage, thereby compromising the second stage’s search capability.

Regarding generalization performance, while Learned Points Guidance + Relaxation achieves comparable success rates in standard scenarios, it significantly underperforms in OOD conditions. This performance gap stems from polynomial fitting’s superior ability to integrate multi-point information, producing more stable results compared to discrete point prediction, which suffers from data sparsity as identical trajectories generate numerous possible point combinations.

Additionally, Learned Points Guidance + Relaxation exhibits significantly longer convergence times and more iteration times compared to our proposed methods, indicating that the learned points deviate substantially from the desired optimal trajectory, further highlighting the deficiencies of this approach.

Computational analysis reveals that increasing guide points extends convergence time compared to baseline performance, as guide points direct optimization toward potentially distant desired directions, which increases the iteration requirements. This indicates that setting more guide points leads to increased solution time and complexity for the optimization problem.

Interestingly, Two-Point Guidance + Relaxation occasionally produces higher final costs than the no-relaxation variant, reflecting optimization randomness where relaxed first-stage criteria may lead to different local optima. However, cost differences remain minimal, which is still acceptable. In terms of trajectory quality, we observe that IL successfully learns high-quality trajectory tendencies from demonstration data. Furthermore, increasing the number of guide points yields lower final jerk values, indicating that more guide points drive the optimization closer to these learned high-quality trajectories.

These findings collectively demonstrate that our IL-guided optimization framework successfully enhances performance, with our polynomial approach exhibiting superior stability and generalization compared to learned point methods. Moreover, selecting merely two points can provide sufficient trajectory tendency information while maintaining computational efficiency.

![](images/7f4d1837facab1d15f94b1653ea1c3bc36c89a84d78d35f93255ea6a2c14eca2.jpg)  
Fig. 11. Collision avoidance performance comparison across three benchmark scenarios: narrow gate (left), tilted hole (middle), and random cubes (right). The fixed ellipsoid method (top row) results in collisions due to rigid volume representation. The spheres method (middle row) achieves collisionfree navigation. Our varying ellipsoid method (bottom row) demonstrates superior performance through adaptive manipulator retraction.

## C. Benchmark

To validate the superiority of our varying ellipsoid method for collision volume approximation, we designed three obstacle-rich scenarios and evaluated the aerial manipulator’s collision avoidance performance through each environment.

The first scenario, termed ”Narrow Gate”, consists of two rectangular obstacles positioned to create a narrow passage between them, as illustrated in the first column of Figure 11. We systematically varied the gap width from 0.60m to 0.25m in 0.05m decrements, resulting in eight experimental trials. Throughout these experiments, the aerial manipulator was manually configured to maintain an extended state at both the initial and final positions, with the end-effector position set to ${ { \mathcal P } _ { p _ { e , z } } } = - 0$ .2m to ensure consistent testing conditions. The second scenario, ”Tilted Hole,” features a wall with an inclined hole, as shown in the second column of Figure 11. We examined three different inclination angles: 20<sup>◦</sup>, 40<sup>◦</sup>, and 60<sup>◦</sup>. For each angle, the manipulator’s initial end-effector positions were set to $^ { \mathcal { D } } p _ { e , z } = - 0 . 0 7 \mathrm { m } , - 0 . 1 3 \mathrm { m }$ , and −0.2m respectively, yielding a total of nine experimental configurations. The third scenario, ”Random Cubes”, involves randomly distributing 12 cubic obstacles within a three-dimensional workspace. Ten different random configurations were generated to provide diverse navigation challenges for the aerial manipulator. For all experiments, a trial was considered successful if the aerial manipulator reached the target position without any collision occurrence during the entire trajectory.

We compared our approach against two established collision volume approximation methods: Fast-Racing [39] and REMANI [47]. The Fast-Racing method employs a single fixed ellipsoid to approximate the entire aerial manipulator system. To ensure collision safety, this fixed ellipsoid must be sized to accommodate the manipulator in its fully extended configuration. The REMANI method utilizes multiple spheres to represent the aerial manipulator’s geometry: four spheres with 0.08m radius for the quadrotor body, one sphere with 0.025m radius for the end-effector, and fifteen spheres with 0.025m radius distributed along the upper and lower arm segments. All experimental comparisons maintained identical constraint weights and hyperparameters across the 27 total trials.

The results presented in Figure 11 demonstrate clear performance distinctions between methods. The fixed ellipsoid approach failed to navigate collision-free through gaps smaller than or equal to 0.45m, as the rigid geometric representation prevented adaptive arm retraction during narrow passage traversal. Similarly, this method failed in the tilted hole scenario when the initial end-effector positions were set to −0.13m and −0.2m. The random cubes scenario also resulted in collisions when narrow passages were encountered.

As summarized in Table III, the fixed ellipsoid method achieved only 37.5% and 33.3% success rates in the first two scenarios, respectively, while maintaining 70% success in the random cubes scenario due to the relatively sparse obstacle distribution.

In contrast, both the sphere-based method and our proposed varying ellipsoid approach achieved 100% success rates in the narrow gate and tilted hole scenarios. However, their collision avoidance strategies differed significantly, as observable in the bottom-right panel of the second column in Figure 11. The sphere-based method tends to induce lateral (xy-axis) manipulator movements, while our approach primarily achieves collision avoidance through strategic end-effector retraction along the z-axis.

For the random cubes scenario, our proposed method achieved an 80% success rate compared to 70% for the spherebased approach. The failures in both methods were primarily attributed to path planning limitations that forced the aerial manipulator into excessively constrained regions where sufficient SFC volume could not be established. Additionally, the sphere-based method’s failures resulted from the cumulative high costs generated by multiple spheres in proximity to obstacles, preventing the optimizer from adequately balancing other essential constraints and leading to localized collisions.

The computational efficiency comparison in Table III reveals that our method’s average iteration time closely matches that of the fixed ellipsoid approach while significantly outperforming the sphere-based method. This efficiency stems from our approach’s O(n) computational complexity, as only a single ellipsoid-to-polyhedron cost evaluation is required per iteration, compared to the O(mn) complexity of the spherebased method that necessitates individual cost calculations for each sphere.

In summary, our varying ellipsoid fitting method successfully combines the computational efficiency of ellipsoidal representations with comprehensive whole-body collision avoidance capabilities for aerial manipulators.

TABLE III  
PERFORMANCE COMPARISON OF OBSTACLE AVOIDANCE METHODS

<table><tr><td>Method</td><td>Narrow Gate</td><td>Tilted Hole</td><td>Random Cubes</td><td>Avg. Iteration Time</td></tr><tr><td>Fast-Racing [39]</td><td>37.5%</td><td>33.3%</td><td>70.0%</td><td>33.85 ms</td></tr><tr><td>REMANI [47]</td><td>100.0%</td><td>100.0%</td><td>70.0%</td><td>123.73 ms</td></tr><tr><td>Proposed</td><td>100.0%</td><td>100.0%</td><td>80.0%</td><td>45.85 ms</td></tr></table>

![](images/689ca3bacfdea73f8649ab114487c1519fe129a99f493ae7cd06cad776daafe9.jpg)

(a)  
![](images/7223c261c3fe7f7f28c1cf5c0c9bd57f6d03eb4455523d50d2aa1830bda8d072.jpg)

![](images/883c2ceafbeb905a49ae072768386caf8b131a86f846f5200ccaaad9f36b0334.jpg)

![](images/a22563be3e254055f82b1c0eb3d868f5318da435fc63d1d22b494e7bbae836aa.jpg)

![](images/6c513a85acb861832fdffbeb4e7822b686dbb82b034321426f6faddde15c67df.jpg)

![](images/c5705772741e255b8fa83b037d0e068ba49afa2a3d2a2060fe0ec3e3987f3b96.jpg)

![](images/a1025e12c41900ed35de495f64105011da08d565741a2042869c27d79e935349.jpg)  
Fig. 12. Validation of the planning framework across different manipulator configurations. (a) Three manipulator types: vertical motion (top), 1-DOF circular arc motion (middle), and 2-DOF planar motion (bottom). Left column shows 3D end-effector trajectories within their respective workspaces (gray regions). Right column displays the corresponding position profiles for each axis over time. All trajectories remain strictly within workspace boundaries, demonstrating the framework’s adaptability to various kinematic constraints.

## D. Simulation

We formulate the end-effector planning in Cartesian space specifically to accommodate different aerial manipulator configurations. This section validates that our approach can work with various manipulator types through comprehensive simulation studies.

Due to hardware limitations, we evaluate the feasibility of our approach through simulation analysis. As illustrated in Figure 12 (a), we consider three distinct manipulator configurations with different kinematic constraints. The first type is a telescopic manipulator that permits only vertical (z-axis) motion of the end-effector. The second type is a 1- DOF manipulator that allows the end-effector to move along a circular arc in the xz-plane. The third type is a 2-DOF manipulator that enables planar motion within the entire xzplane. The respective workspaces for each configuration are visualized as gray regions in the left column of Figure 12.

We designed a planar grasping scenario with a waypoint constraint for the end-effector in the world frame $\mathcal { F } _ { \mathcal { W } } .$ , setting the terminal velocity to zero at the target position. For the telescopic manipulator, the workspace bounds were set to $^ { \mathcal { D } } p _ { e , z } ~ \in ~ [ - 0 . 2 5 , - 0 . 0 6 ] \mathrm { m }$ . For the 1-DOF manipulator, the workspace is constrained to a circular arc with radius 0.18m in the xz-plane. For the 2-DOF planar manipulator, the workspace was defined as $^ { \mathcal { D } } p _ { e , x } ~ \in ~ [ - 0 . 2 , 0 . 2 ] \mathrm { \bar { m } }$ and $^ { \mathcal { D } } p _ { e , z } \in [ \bar { - } 0 . 2 5 , - 0 . 0 6 ] \mathrm { m }$

![](images/1b7d3dd707c2ea3d9248490646106f4b65c9d86a7935e2636453547aa4a82b57.jpg)  
Fig. 13. Real-world aerial manipulator platform used for validation.

The trajectory planning results demonstrate high accuracy, with minimum distances to the target point of $6 . 8 5 1 \times 1 0 ^ { - 3 } \mathrm { m }$ $4 . 8 5 7 \times 1 0 ^ { - 3 } \mathrm { m } .$ , and $2 . 0 5 5 \times 1 0 ^ { - 3 } \mathrm { m }$ for the three manipulator types, respectively. These errors are sufficiently small to confirm successful target reaching. The 3D end-effector trajectories in the frame $\mathcal { F } _ { \mathcal { D } } .$ , shown in the left column of Figure 12, strictly adhere to their respective workspace constraints. The right column presents the individual axis variations over time, further confirming that all planned trajectories remain within the prescribed workspace boundaries.

These results demonstrate that our whole-body planning framework successfully adapts to different manipulator types while maintaining workspace compliance and trajectory accuracy.

## E. Real-world Implementation

Our system is implemented in C++11 on Ubuntu 20.04 with ROS Noetic. As illustrated in Figure 13, the experimental platform integrates an NVIDIA Jetson Orin NX 16GB as the onboard computer and an NxtPX4v2 flight controller on a quadrotor equipped with a delta arm. The delta arm utilizes three DYNAMIXEL XL430-W250-T servo motors and features a needle end-effector for precise manipulation tasks. For state estimation, the system employs an Extended Kalman Filter (EKF) that fuses data from the NOKOV Motion Capture System and an onboard IMU. We implement a separated control framework inspired by the composite control scheme in [35], which employs an NDOB to compensate for internal coupling and external disturbances, while using a high-pass filter to enhance manipulator agility. Throughout the experimental validation, we assume that interaction forces including gravity and friction are constant and known during the planning phase.

To validate our approach in real-world scenarios, we focus on a fundamental question: whether our flexible waypoint constraints can effectively operate in practical environments and successfully execute diverse manipulation skills. To address this question systematically, we design 9 fundamental manipulation skills: striking, grasping, lifting, pressing, winding, pulling, pushing, crossing, and writing. We conduct a total of 17 experiments across these skills and analyze their performance comprehensively.

X (m)  
(g)  
![](images/4b6c36c933d867558252aa62ca07e42e707d8560cb9ea32ac707b539227a1f7a.jpg)

![](images/1717c29d4695d03e187cbd3434f4b198f941aef7bbaba11e3c79c6bba371b83e.jpg)

![](images/2ed462cec95575a7783aa9f0bfeca8e1293a96f90de6a829ba7c421082776257.jpg)

![](images/0fb9e7c34023150d388a6ca65ea60f00d521816f03533055245260c4848660c4.jpg)

![](images/16fade0f15a1ec2a7088d8c2c534cbe2a415de11013d205490d913004792a05c.jpg)

![](images/4a0c977b6aa559c95a380101aa3cd755fc38ff511e4fc198ddaaa9d2d3a1e03e.jpg)

![](images/46e342c23d4c1aa3907d303f154d29e8731c36e0f85283f3a7e295bb567e8296.jpg)

![](images/a6d78929d98df2ae74b558fab3e59b9f0a3ca1c908c84505d3e552918aacc552.jpg)  
(d)  
X (m)

![](images/2f8a69dc5624545d07040c73c3da4385b1f0a46fe87def6e1db51c95ab5917b8.jpg)  
X (m)

![](images/ca1b909f8008e95a2d0cbb00eb2b14053b83d17ca6d175238684c6bce6957603.jpg)

![](images/01149c7678cf15f17c6e76ec38a95b29bf04000e881670b74972e2796b14eed6.jpg)

![](images/903915e0da901880bfba6975276344ac87f7a5bff7931c56e1e90d3223e3acb1.jpg)

![](images/7f7ebcdaca52c22a76319dcd4e60080442457cad27a2f748ad8c5172f2776296.jpg)  
X (m)

![](images/f7b81b8b51ec5f408bd442e92a6881715278c08fececd996b8e39ed4740c9815.jpg)  
X (m)

![](images/78a7957559ffd8ccb9e041bb8f84880366b1fb737e5b4833697b675cd069ea20.jpg)

![](images/e328fcfafbefa16413cf50cfff92c61cd772d9d5111c0b38abc6c2c7e8f72e2b.jpg)  
Fig. 14. Experimental results for strike and grasp tasks. (a)-(c) show aerial striking experiments while (e) and (f) present aerial grasping results. (d) The first two plots demonstrate IL-generated guide points and resulting quadrotor and end-effector motion in the xy-plane. The third plot shows roll angle variations across four strike maneuvers. (g) The first two plots show IL guide point generation for grasp skills. The third and fourth plots present quadrotor velocity and comparison between planned and actual thrust curves for the aerial manipulator.

1) Strike: We design a striking scenario with an inclined surface positioned at approximately 37.6 degrees at the center of the test area. To thoroughly validate our IL-guided optimization framework, we position the aerial manipulator behind the inclined surface and conduct experiments from two distinct initial positions: one within the training data distribution (1.5- 3m radius) and another outside this distribution (beyond 3m radius). The strike skill requires setting a single end-effector waypoint constraint in $\mathcal { F } _ { \mathcal { W } }$ along with its corresponding orientation.

As shown in Figure 14 (a) and (b), given the initial relative position and the surface inclination angle, our ILguided framework generates guide points as demonstrated in Figure 14 (d) (first two plots). To further validate the orientation capabilities of our waypoint constraints, we test the aerial manipulator’s striking performance on four different inclined surfaces with angles of 15<sup>◦</sup>, 30<sup>◦</sup>, 45<sup>◦</sup>, and 60<sup>◦</sup>. To eliminate the influence of other constraints and focus solely on orientation variations, we specify that the delta arm should extend to ${ } ^ { \mathcal { D } } p _ { e } = [ 0 . 0 , 0 . 0 , - 0 . { \overset { . } { 2 } } ] ^ { \top }$ m, and subsequently calculate the corresponding quadrotor position to set as the waypoint constraint. This approach enables isolated observation of orientation changes. As illustrated in Figure 14 (c) and the third plot in Figure 14 (d), the aerial manipulator successfully achieves the desired corresponding attitude at the target position.

2) Grasp: We design complementary grasping scenarios with two experimental configurations. In each test, an object of known mass is placed on an inclined surface, with its position tracked using the motion capture system. The aerial manipulator is tasked with planning and executing a grasping maneuver. This grasp skill requires setting an endeffector waypoint constraint, and to improve success rates, we additionally impose a velocity constraint of 0.2 m/s downward for end-effector along z<sub>B</sub> at the waypoint, while constraining velocities in x and ${ \pmb y } _ { B }$ directions to zero.

![](images/bb6c5c87563734063e31e124254203ba5e014945ce8d8d64304799bae0bcc6b6.jpg)  
Fig. 15. Real flight results for seven tasks: lift, press, wind, pull, push, cross, and write. (a)-(e) show xyz trajectory variations in the world frame for the first five tasks. (f) presents end-effector z-axis motion trajectory in F<sub>D</sub> during the cross task. (g) and (h) show actual end-effector movement during the write task and corresponding error distributions, respectively.

To validate our framework’s advantages, we again position the aerial manipulator behind the inclined surface as the initial position. The first experimental group has an initial position at (-3, 1) m with a surface inclination of approximately 42.3<sup>◦</sup>, while the second group starts at (-2.5, 2.5) m with an inclination of 26.7<sup>◦</sup>. The actual flight performance is shown in Figure 14 (e) and (f), demonstrating successful object grasping in both cases. Figure 14 (g) (first two plots) illustrates the corresponding IL-generated guide points and the resulting quadrotor and end-effector trajectories in the xy-plane. The third plot in Figure 14 (g) shows the quadrotor’s grasping velocity, demonstrating rapid grasping speeds of nearly 3 m/s while confirming that the quadrotor reduces speed appropriately to satisfy the constraints. The fourth plot compares the planned versus actual thrust forces, showing that despite using a simplified model that considers only quasi-static forces and assumes fixed external forces, these assumptions remain reasonable for simple manipulation tasks. However, the observed discrepancies during contact phases indicate that more sophisticated contact-rich planning methods would be necessary for more complex or dynamic manipulation scenarios.

3) Lift, Press, Wind: These three tasks can be completed by applying one or several waypoint constraints. The lift task requires the aerial manipulator to use a hook to lift a basket, which is considered successful when the basket is lifted. We apply two waypoint constraints: one below the basket handle and another directly above it. The press task requires the end-effector to press a button, with success indicated by the button emitting red light, achieved by setting a single waypoint constraint at the button’s position. The wind task involves threading a line through three hooks at known positions, with success defined as the line successfully passing through all hooks. This requires setting three end-effector waypoint constraints. The experimental results shown in Figure 15 (Lift, Press, Wind) demonstrate successful completion of all three tasks. The corresponding motion trajectory curves are presented in Figure 15 (a)-(c), where gray areas indicate the task execution regions.

4) Pull, Push: We design specific scenarios to validate the two skills independently. For the pull task, we attach a sponge to the end-effector and program the aerial manipulator to perform a table-wiping task. We set three coplanar waypoint constraints beyond the table surface and constrain the zcomponent variation of the end-effector position to zero within these points, enabling movement within a single plane. For the push task, we place a wooden block on a table surface and program the aerial manipulator to push it off the table while restricting end-effector movement to the x-axis direction only, with y and z-axis position components constrained. The results shown in Figure 15 (Pull, Push) demonstrate successful task execution. Figure 15 (d) and (e) show that the gray constraint regions effectively limit motion: the pull task maintains minimal z-axis variation, while the push task shows consistent y and z-axis values within the gray constraint region.

5) Cross: We design a narrow racing gate that requires the aerial manipulator to retract its robotic arm to successfully pass through. To increase experimental complexity and comprehensively validate our framework’s capability, we extend the endeffector to the -0.2m position before and after crossing, thereby demonstrating both obstacle avoidance and manipulation capabilities. Figure 15 (Cross) shows the aerial manipulator’s state during gate traversal, while Figure 15 (f) presents the end-effector’s z-axis trajectory in $\mathcal { F } _ { \mathcal { D } }$ during flight, clearly demonstrating the robotic arm’s retraction-extension sequence from initial extension through retraction for gate passage to re-extension afterward.

6) Write: The final task we design is writing, which is comparatively complex, requiring numerous waypoint constraints while ensuring end-effector movement within the same plane. The positional constraints vary dynamically; for example, writing ”T” requires initial constraint to x-axis movement, lifting the brush pen, then constraining to y-axis movement. Writing ”R” requires multiple constrained movement transitions. The experimental results shown in Figure 15 (Write) demonstrate successful completion of writing the letters ”TRO” using a brush pen. The actual end-effector movement can be observed in Figure 15 (g). We note slight oscillations at corners when writing $\because \mathrm { O } _ { \mathrm { , } } ^ { \mathrm { , , } \mathrm { , } }$ attributed to unstable connections between the end-effector and brush, as well as uneven paper surfaces. The position errors for these three experiments, shown in Figure 15 (h), maintain median end-effector errors within 3cm, which falls within acceptable ranges. This further validates that while our planning employs simplified models, most disturbances and unconsidered dynamic coupling effects can be resolved at the controller level, with our planner providing highly effective reference trajectories.

## VII. CONCLUSION

This paper introduces a novel whole-body integrated motion planning framework for aerial manipulators that jointly optimizes the motions of the quadrotor and the manipulator. We propose an efficient varying ellipsoid collision avoidance method that dynamically adapts to the aerial manipulator’s configuration changes, enabling safe navigation in complex environments. The framework introduces flexible partial waypoint constraints for precise task specification and an ILguided optimization approach to address the limitations of standard optimization methods in handling large-attitude manipulation tasks. This IL-guided method leverages imitation learning to provide appropriate guide points, significantly improving planning success rates for challenging manipulation scenarios. Our comprehensive experimental validation through 17 real-world experiments spanning 9 fundamental manipulation skills demonstrates the framework’s excellent scalability and versatility across diverse manipulation tasks.

While our algorithm demonstrates outstanding performance across various manipulation skills, the current computational requirements prevent real-time operation. This limitation restricts the framework’s ability to handle highly dynamic and complex environments where real-time replanning and adaptive adjustments are essential for safe operation. Additionally, as evidenced by the discrepancies observed in thrust force comparisons during grasp experiments, the simplified quasistatic force models limit our ability to consider tasks involving strong interaction forces during the planning stage. Therefore, future work will focus on developing advanced dynamic models and improving computational efficiency, with the goal of enabling real-time performance in complex, contact-rich scenarios.

## REFERENCES

[1] A. Ollero, M. Tognon, A. Suarez, D. Lee, and A. Franchi, “Past, present, and future of aerial robotic manipulators,” IEEE Transactions on Robotics, vol. 38, no. 1, pp. 626–645, 2022.

[2] K. Zhang, P. Chermprayong, F. Xiao et al., “Aerial additive manufacturing with multiple autonomous robots,” Nature, vol. 609, no. 7928, pp. 709–717, 2022. [Online]. Available: https://doi.org/10.1038/ s41586-022-04988-4

[3] M. . Trujillo, J. R. Martnez-de Dios, C. Martn, A. Viguria, and A. Ollero, “Novel aerial manipulator for accurate and robust industrial ndt contact inspection: A new tool for the oil and gas inspection industry,” Sensors, vol. 19, no. 6, 2019. [Online]. Available: https://www.mdpi.com/1424-8220/19/6/1305

[4] P. Foehn, A. Romero, and D. Scaramuzza, “Time-optimal planning for quadrotor waypoint flight,” Science Robotics, vol. 6, no. 56, p. eabh1221, 2021. [Online]. Available: https://www.science.org/doi/abs/ 10.1126/scirobotics.abh1221

[5] P. Chermprayong, K. Zhang, F. Xiao, and M. Kovac, “An integrated delta manipulator for aerial repair: A new aerial robotic system,” IEEE Robotics & Automation Magazine, vol. 26, no. 1, pp. 54–66, 2019.

[6] K. Hang, X. Lyu, H. Song, J. A. Stork, A. M. Dollar, D. Kragic, and F. Zhang, “Perching and restinga paradigm for uav maneuvering with modularized landing gears,” Science Robotics, vol. 4, no. 28, p. eaau6637, 2019. [Online]. Available: https://www.science.org/doi/abs/ 10.1126/scirobotics.aau6637

[7] H.-N. Nguyen, S. Park, J. Park, and D. Lee, “A novel robotic platform for aerial manipulation using quadrotors as rotating thrust generators,” IEEE Transactions on Robotics, vol. 34, no. 2, pp. 353–369, 2018.

[8] M. Xu, Q. De, D. Yu, A. Hu, Z. Liu, and H. Wang, “Biomimetic morphing quadrotor inspired by eagle claw for dynamic grasping,” IEEE Transactions on Robotics, vol. 40, pp. 2513–2528, 2024.

[9] J. Thomas, J. Polin, K. Sreenath, and V. Kumar, “Avian-inspired grasping for quadrotor micro UAVs,” in Volume 6A: 37th Mechanisms and Robotics Conference. American Society of Mechanical Engineers, p. V06AT07A014.

[10] M. Ryll, G. Muscio, F. Pierri, E. Cataldi, G. Antonelli, F. Caccavale, D. Bicego, and A. Franchi, “6d interaction control with aerial robots: The flying end-effector paradigm,” vol. 38, no. 9, pp. 1045–1062. [Online]. Available: http://journals.sagepub.com/doi/10.1177/0278364919856694

[11] K. Bodie, M. Brunner, M. Pantic, S. Walser, P. Pfndler, U. Angst, R. Siegwart, and J. Nieto, “Active interaction force control for contactbased inspection with a fully actuated aerial vehicle,” IEEE Transactions on Robotics, vol. 37, no. 3, pp. 709–722, 2021.

[12] H. Cao, J. Shen, C. Liu, B. Zhu, and S. Zhao, “Motion planning for aerial pick-and-place with geometric feasibility constraints,” IEEE Transactions on Automation Science and Engineering, vol. 22, pp. 2577– 2594, 2025.

[13] H. Lee, H. Kim, and H. J. Kim, “Planning and control for collisionfree cooperative aerial transportation,” IEEE Transactions on Automation Science and Engineering, vol. 15, no. 1, pp. 189–201, 2018.

[14] M. Wang, Z. Chen, K. Guo, X. Yu, Y. Zhang, L. Guo, and W. Wang, “Millimeter-level pick and peg-in-hole task achieved by aerial manipulator,” IEEE Transactions on Robotics, vol. 40, pp. 1242–1260, 2024.

[15] Y. Chen, J. Liang, Y. Wu, Z. Miao, H. Zhang, and Y. Wang, “Adaptive sliding-mode disturbance observer-based finite-time control for unmanned aerial manipulator with prescribed performance,” IEEE Transactions on Cybernetics, vol. 53, no. 5, pp. 3263–3276, 2023.

[16] H. Cao, Y. Li, C. Liu, and S. Zhao, “Eso-based robust and highprecision tracking control for aerial manipulation,” IEEE Transactions on Automation Science and Engineering, vol. 21, no. 2, pp. 2139–2155, 2024.

[17] G. He, X. Guo, L. Tang, Y. Zhang, M. Mousaei, J. Xu, J. Geng, S. Scherer, and G. Shi, “Flying hand: End-effector-centric framework for versatile aerial manipulation teleoperation and policy learning.” [Online]. Available: http://arxiv.org/abs/2504.10334

[18] M. Spahn, B. Brito, and J. Alonso-Mora, “Coupled mobile manipulation via trajectory optimization with free space decomposition,” in 2021 IEEE International Conference on Robotics and Automation (ICRA). IEEE, pp. 12 759–12 765. [Online]. Available: https://ieeexplore.ieee. org/document/9561821/

[19] R. Spica, A. Franchi, G. Oriolo, H. H. Bulthoff, and P. R. Giordano, “Aerial grasping of a moving target with a quadrotor UAV,” in 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems. IEEE, pp. 4985–4992. [Online]. Available: http://ieeexplore.ieee.org/document/6385771/

[20] M. Tognon, E. Cataldi, H. A. T. Chavez, G. Antonelli, J. Corts, and A. Franchi, “Control-aware motion planning for task-constrained aerial manipulation,” IEEE Robotics and Automation Letters, vol. 3, no. 3, pp. 2478–2484, 2018.

[21] Suseong Kim, Seungwon Choi, and H. J. Kim, “Aerial manipulation using a quadrotor with a two DOF robotic arm,” in 2013 IEEE/RSJ International Conference on Intelligent Robots and Systems. IEEE, pp. 4990–4995. [Online]. Available: http://ieeexplore.ieee.org/document/ 6697077/

[22] M. Brunner, G. Rizzi, M. Studiger, R. Siegwart, and M. Tognon, “A planning-and-control framework for aerial manipulation of articulated objects,” IEEE Robotics and Automation Letters, vol. 7, no. 4, pp. 10 689–10 696, 2022.

[23] X. Guo, G. He, J. Xu, M. Mousaei, J. Geng, S. Scherer, and G. Shi, “Flying calligrapher: Contact-aware motion and force planning and control for aerial manipulation.” [Online]. Available: http://arxiv.org/abs/2407.05587

[24] P. Foehn, A. Romero, and D. Scaramuzza, “Time-optimal planning for quadrotor waypoint flight,” vol. 6, no. 56, p. eabh1221. [Online]. Available: https://www.science.org/doi/10.1126/scirobotics.abh1221

[25] H. Kim, H. Seo, J. Kim, and H. J. Kim, “Sampling-based motion planning for aerial pick-and-place,” in 2019 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2019, pp. 7402– 7408.

[26] H. Seo, S. Kim, and H. J. Kim, “Locally optimal trajectory planning for aerial manipulation in constrained environments,” in 2017 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, pp. 1719–1724. [Online]. Available: http://ieeexplore.ieee.org/document/8205984/

[27] M. Neunert, C. de Crousaz, F. Furrer, M. Kamel, F. Farshidian, R. Siegwart, and J. Buchli, “Fast nonlinear model predictive control for unified trajectory optimization and tracking,” in 2016 IEEE International Conference on Robotics and Automation (ICRA), 2016, pp. 1398–1404.

[28] Z. Wang, X. Zhou, C. Xu, and F. Gao, “Geometrically constrained trajectory optimization for multicopters,” IEEE Transactions on Robotics, vol. 38, no. 5, pp. 3259–3278, 2022.

[29] C. Chi, Z. Xu, S. Feng, E. Cousineau, Y. Du, B. Burchfiel, R. Tedrake, and S. Song, “Diffusion policy: Visuomotor policy learning via action diffusion.” [Online]. Available: http://arxiv.org/abs/2303.04137

[30] M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. Foster, G. Lam, P. Sanketi, Q. Vuong, T. Kollar, B. Burchfiel, R. Tedrake, D. Sadigh, S. Levine, P. Liang, and C. Finn, “OpenVLA: An open-source vision-language-action model.” [Online]. Available: http://arxiv.org/abs/2406.09246

[31] K. Black, N. Brown, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, L. Groom, K. Hausman, B. Ichter, S. Jakubczak, T. Jones, L. Ke, S. Levine, A. Li-Bell, M. Mothukuri, S. Nair, K. Pertsch, L. X. Shi, J. Tanner, Q. Vuong, A. Walling, H. Wang, and U. Zhilinsky, “\$ 0\$: A vision-language-action flow model for general robot control.” [Online]. Available: http://arxiv.org/abs/2410.24164

[32] T. Z. Zhao, V. Kumar, S. Levine, and C. Finn, “Learning fine-grained bimanual manipulation with low-cost hardware.” [Online]. Available: http://arxiv.org/abs/2304.13705

[33] M. Ghazi, Q. Nazir, S. Butt, and A. Baqai, “Accuracy analysis of 3-rss delta parallel manipulator,” Procedia Manufacturing, vol. 17, pp. 174– 182, 01 2018.

[34] D. Tzoumanikas, F. Graule, Q. Yan, D. Shah, M. Popovic, and S. Leutenegger, “Aerial manipulation using hybrid force and position NMPC applied to aerial writing.” [Online]. Available: http://arxiv.org/abs/2006.02116

[35] H. Chen, B. Ye, X. Liang, W. Deng, and X. Lyu, “Ndob-based control of a uav with delta-arm considering manipulator dynamics,” in 2025 IEEE International Conference on Robotics and Automation (ICRA), 2025, pp. 7505–7511.

[36] A. Codourey, “Dynamic modelling and mass matrix evaluation of the delta parallel robot for axes decoupling control,” in Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS ’96), vol. 3, 1996, pp. 1211–1218.

[37] S. Liu, M. Watterson, K. Mohta, K. Sun, S. Bhattacharya, C. J. Taylor, and V. Kumar, “Planning dynamically feasible trajectories for quadrotors using safe flight corridors in 3-d complex environments,” IEEE Robotics and Automation Letters, vol. 2, no. 3, pp. 1688–1695, 2017.

[38] S. Liu, K. Mohta, N. Atanasov, and V. Kumar, “Search-based motion planning for aggressive flight in se(3),” IEEE Robotics and Automation Letters, vol. 3, no. 3, pp. 2439–2446, 2018.

[39] Z. Han, Z. Wang, N. Pan, Y. Lin, C. Xu, and F. Gao, “Fast-racing: An open-source strong baseline for SE(3) planning in autonomous drone racing,” IEEE Robotics and Automation Letters, vol. 6, no. 4, pp. 8631– 8638, 2021.

[40] Y. Ren, S. Liang, F. Zhu, G. Lu, and F. Zhang, “Online whole-body motion planning for quadrotor using multi-resolution search,” in 2023 IEEE International Conference on Robotics and Automation (ICRA). IEEE, pp. 1594–1600.

[41] S. Yang, B. He, Z. Wang, C. Xu, and F. Gao, “Whole-body realtime motion planning for multicopters,” in 2021 IEEE International Conference on Robotics and Automation (ICRA). IEEE, pp. 9197–9203. [Online]. Available: https://ieeexplore.ieee.org/document/9561526/

[42] Y. Gao, J. Ji, Q. Wang, R. Jin, Y. Lin, Z. Shang, Y. Cao, S. Shen, C. Xu, and F. Gao, “Adaptive tracking and perching for quadrotor in dynamic scenarios,” IEEE Transactions on Robotics, vol. 40, pp. 499–519, 2024.

[43] C. A. Dimmig and M. Kobilarov, “Non-prehensile aerial manipulation using model-based deep reinforcement learning,” in 2024 IEEE 20th International Conference on Automation Science and Engineering (CASE), 2024, pp. 2194–2200.

[44] R. Ni, T. Schneider, D. Panozzo, Z. Pan, and X. Gao, “Robust & asymptotically locally optimal UAV-trajectory generation based on spline subdivision,” in 2021 IEEE International Conference on Robotics and Automation (ICRA). IEEE, pp. 7715–7721. [Online]. Available: https://ieeexplore.ieee.org/document/9561272/

[45] S. Liu, L. Wu, B. Li, H. Tan, H. Chen, Z. Wang, K. Xu, H. Su, and J. Zhu, “RDT-1b: a diffusion foundation model for bimanual manipulation.” [Online]. Available: http://arxiv.org/abs/2410.07864

[46] W. Peebles and S. Xie, “Scalable diffusion models with transformers.” [Online]. Available: http://arxiv.org/abs/2212.09748

[47] C. Wu, R. Wang, M. Song, F. Gao, J. Mei, and B. Zhou, “Real-time whole-body motion planning for mobile manipulators using environment-adaptive search and spatial-temporal optimization,” in 2024 IEEE International Conference on Robotics and Automation (ICRA). IEEE, pp. 1369–1375. [Online]. Available: https://ieeexplore. ieee.org/document/10610192/

![](images/bf2ba508685fc105d2af2dcc04d5b58f745659b359e75b3b421ea7c742a954c2.jpg)

Weiliang Deng received his B.Eng. degree in Intelligence Science and Technology from Sun Yatsen University, Shenzhen, China, in 2025. He is currently working toward the M.Phil. degree with the Department of Intelligent Systems Engineering, Sun Yat-sen University, Shenzhen, China.

His research interests include motion planning, aerial robots and robot manipulation.

![](images/d34f08529739ccada6f50e933be7b8ecb8ce55ff5fac7b500ceae22d574ce0f4.jpg)

Haoran Chen received his B.Eng. degree in Intelligence Science and Technology from Sun Yatsen University, Shenzhen, China, in 2025. He is currently working toward the M.S. degree in the department of mechanical engineering at the National University of Singapore, Singapore.

His research interests include aerial manipulation, and hardware design, locomotion control, and planning of legged robots.

![](images/357c15a59323297cb29001eb9b01c37bae144098ca92178f04151e293abe193d.jpg)

Hongming Chen received the B.Eng. degree in Software Engineering from the University of Electronic Science and Technology of China, Chengdu, China, in 2024. He is currently working toward the M.Phil. degree with the Department of Intelligent Systems Engineering, Sun Yat-sen University, Shenzhen, China.

His research interests include unmanned aerial vehicles, semantic navigation, and aerial manipulation.

![](images/6120aba57c3871a74b6b933de48a5f83eebd8ad750b9e34853ce621ac184080a.jpg)

Biyu Ye received the B.Eng. degree in Traffic Engineering from Sun Yat-sen University, Guangzhou, China, in 2024. He is currently working toward the M.S. degree in aerial vehicle design with The School of Intelligent Systems Engineering, Sun Yatsen University.

Ziliang Li received his B.Eng. degree in Intelligence Science and Technology from Sun Yat-sen University, Shenzhen, China, in 2024. He is currently working toward the M.Phil. degree with the Department of Intelligent Systems Engineering, Sun Yat-sen University, Shenzhen, China.

![](images/0969b12958952c91a21795198588f1c4778c38b7cd099a8c589c4744e2e24780.jpg)

His research interests include the perception and control of aerial robots.

His research interests include optimal control, motion planning, and aerial manipulation.

![](images/439ea125439c331ba5dfe6e7d3f7983a8fdae4937dad7cef2553111a16be6d79.jpg)

Ximin Lyu received his B.Eng. and M.Phil. degrees in Aircraft Manufacturing from Harbin Institute of Technology, Harbin, China, in 2012 and 2014, respectively. He received his Ph.D. in Electronic and Computer Engineering from the Hong Kong University of Science and Technology, Hong Kong, China, in 2019.

In 2018, he served as a Senior Flight Control Researcher at Da Jiang Innovation (DJI) Technology Company in China. Since 2021, he has been an Associate Professor with the Department of Intelligent

Systems Engineering, Sun Yat-sen University, Shenzhen, China. His research interests encompass robotics and controls, with a specific focus on UAV/UGV design, control, and planning.