# Human-Aware Physical Human–Robot Collaborative Transportation and Manipulation With Multiple Aerial Robots

Guanrui Li , Graduate Student Member, IEEE, Xinyang Liu , and Giuseppe Loianno , Member, IEEE

Abstract—Human–robot interaction will play an essential role in various industries and daily tasks, enabling robots to effectively collaborate with humans and reduce physical workload. Most existing approaches for physical human–robot interaction focus on collaboration between a human and a single ground or aerial robot. In recent years, very little progress has been made in this research area when considering multiple aerial robots, which offer increased versatility and mobility. This article presents a novel approach for physical human–robot collaborative transportation and manipulation of a cable-suspended payload with multiple aerial robots. The proposed method enables smooth and intuitive interaction between the transported objects and a human worker. We address the inter-robots and inter-robot–human separation during the operations by exploiting the internal redundancy of the multirobot transportation system. The key elements of our approach are, first, a collaborative payload external wrench estimator that does not rely on any force sensor; second, a 6-D admittance controller for human–aerial–robot collaborative transportation and manipulation; third, a human-aware force distribution that exploits the internal system redundancy to guarantee the execution of additional tasks such as inter-human–robot separation without compromising the payload trajectory tracking or interaction quality. We validate our approach through extensive simulation and real-world experiments. These include scenarios where the robot team assists the human in transporting and manipulating a load, or where the human helps the robot team navigate the environment. We experimentally demonstrate for the first time, to the best of authors’ knowledge that our approach enables a quadrotor team to physically collaborate with a human in manipulating a payload in all 6 degrees of freedom in collaborative human–robot transportation and manipulation tasks.

Index Terms—Aerial robotics, physical human–robot interaction (pHRI).

## NOMENCLATURE

, , <sub>k</sub> World frame, payload frame, kth robot frame. $m _ { L } , m _ { k } \in \mathbb { R }$ Mass of payload, kth robot.

Received 5 April 2024; revised 26 September 2024; accepted 15 October 2024. Date of publication 19 November 2024; date of current version 27 December 2024. This work was supported by in part by the NSF CPS under Grant CNS-2121391 and in part by Qualcomm Research, Nokia, and NYU Wireless. This article was recommended for publication by Associate Editor H. G. d. Marina and Editor P. R. Giordano upon evaluation of the reviewers’ comments. (Guanrui Li and Xinyang Liu contributed equally to this work.) (Corresponding author: Guanrui Li.)

The authors are with the New York University, New York, NY 10012 USA (email: lguanrui@nyu.edu, gli7@wpi.edu; liuxy@nyu.edu; loiannog@nyu.edu).

${ \bf x } _ { L } , { \bf x } _ { k } \in \mathbb { R } ^ { 3 }$ $\dot { \mathbf { x } } _ { L } , \ddot { \mathbf { x } } _ { L } \in \mathbb { R } ^ { 3 }$

$$
\dot {\mathbf {x}} _ {k}, \ddot {\mathbf {x}} _ {k} \in \mathbb {R} ^ {3}
$$

$$
\boldsymbol {R} _ {L} \in \mathrm{SO} (3)
$$

$$
\boldsymbol {R} _ {k} \in \mathrm{SO} (3)
$$

$$
\Theta_ {k} \in \mathbb {R} ^ {3}
$$

$$
\Omega_ {L}, \dot {\Omega} _ {L} \in \mathbb {R} ^ {3}
$$

$$
\Omega_ {k}, \dot {\Omega} _ {k} \in \mathbb {R} ^ {3}
$$

$$
\mathbf {J} _ {L}, \mathbf {J} _ {k} \in \mathbb {R} ^ {3 \times 3}
$$

$$
\boldsymbol {\xi} _ {k} \in S ^ {2}
$$

$$
\boldsymbol {\omega} _ {k} \in \mathbb {R} ^ {3}, l _ {k} \in \mathbb {R}
$$

$$
\mu_ {k} \in \mathbb {R}
$$

$$
\mathbf {F} _ {H}, \mathbf {F} _ {L} \in \mathbb {R} ^ {3}
$$

${ \mathbf { M } } _ { H } \in \mathbb { R } ^ { 3 }$ $\mathbf { M } _ { L } \in \mathbb { R } ^ { 3 }$ $f _ { k } \in \mathbb { R }$ $\mathbf { F } _ { k } \in \mathbb { R } ^ { 3 }$ ${ \bf M } _ { k } \in \mathbb { R } ^ { 3 }$ ${ \boldsymbol \rho } _ { k } \in \mathbb { R } ^ { 3 }$ $\mathbf { p } _ { H } , \mathbf { p } _ { \mathrm { a t t , k } } \in \mathbb { R } ^ { 3 }$

Position of payload, kth robot in . Linear velocity, acceleration of payload in .

Linear velocity, acceleration of kth robot in .

Orientation of payload with respect to $\mathcal { T } .$ Orientation of kth robot with respect to $\mathcal { T } .$ Vector of kth robot’s yaw, pitch, roll in . Payload’s angular velocity, acceleration in .

kth robot’s angular velocity, acceleration in $\boldsymbol { B } _ { k }$

Moment of inertia of payload, kth robot. Unit vector from kth robot to attach point in .

Angular velocity, length of kth cable. Tension magnitude within the kth cable. External human force, net force on payload in .

External human moment on payload in $\mathcal { L } .$ Net moment on payload in $\mathcal { L } .$ Total thrust ofkth quadrotor. Control force on kth robot in $\mathcal { T } .$ Control moment on kth robot in $B _ { k } .$ kth attach point position in $\mathcal { L } .$ Human position, kth attach point position in .

## I. INTRODUCTION

S ENVISIONED in the Industry 4.0 revolution, human– robot interaction will play an increasingly significant role in future industries and daily life [1]. While most research in human–robot interaction has concentrated on collaborations between humans and ground robots, only a limited number of approaches have been developed for aerial robots, with the majority being confined to teleoperation. Unlike ground robots, collaborative microaerial vehicles (MAVs) show additional flexibility and maneuverability due to their 3-D mobility and compact size. Moreover, a team of collaborative MAVs can provide increased adaptivity, resilience, and robustness during a task or multiple simultaneous tasks compared to a single aerial robot. For example, MAV teams can assist humans in executing complex or dangerous tasks, including but not limited to inspection [2], [3], mapping [4], [5], environment interaction [6], [7], surveillance [8], and autonomous transportation and manipulation [9], [10]. Specifically, in autonomous aerial transportation and manipulation, there are many possible usage scenarios. For instance, in a postdisaster response task, a team of aerial robots can cooperatively deliver emergency supplies to designated rescue locations based on the first respondent’s guidance. Alternatively, on construction sites, an aerial robot team can cooperatively manipulate oversized construction materials with human workers to expedite the installation process and reduce physical workload.

![](images/97684adcff9ff6c584eaf5541fa05898199a30afb4d4b7db8b9a540576a8c3b7.jpg)  
Fig. 1. Human operator collaborates with three quadrotors, transporting and manipulating a payload. On the left: The quadrotors are moving to keep a distance from the human operator without affecting payload tracking. On the right: The aerial robot team is transporting the payload using the human operator’s interactive force and moment as commands.

This article proposes a novel approach that enables a team of aerial robots to transport and manipulate a cable-suspended payload in physical collaboration with a human operator, as depicted in Fig. 1. As discussed in [11], cable mechanisms stand out compared to other existing solutions, such as simple spherical joints, or robot arms [12], [13], because of their lighter weight, lower costs, simpler design requirements, and zeroactuation-energy consumption. Therefore, they are particularly suited for size, weight, and power (SWaP) aerial platforms.

Cables also present a good balance among maneuverability, manipulability, and safety for physical human–aerial–robots collaboration compared to other solutions. For instance, several solutions attach the robots directly to the payloads via passive mechanisms like spherical joints [14], magnets [15] or active mechanisms like grippers [16]. However, these mechanisms offer reduced maneuverability and manipulability during a manipulation or physical interaction task compared to cables. Conversely, other complex actuated solutions based on robot arms [17], [18], [19] can enhance maneuverability and flexibility. However, this generally comes at the price of increased system inertia and power, potentially compromising the operator’s safety. Therefore, compared to other existing solutions, we believe that lightweight cable mechanisms can provide a good tradeoff in terms of maneuverability, manipulability, and safety while concurrently offering good flexibility to execute multiple tasks.

We present an innovative control, planning, and estimation framework that enables a human operator to physically collaborate with a team of quadrotors for the transportation and manipulation of a rigid-body payload in all 6 degrees of freedom (DoF). A key contribution of this work is the exploitation of system redundancy, allowing for secondary tasks, such as human-aware human–robot interaction. Specifically, our approach ensures distancing between agents and the human operator during physical collaboration, enabling effective human-aware interaction, as depicted in Fig. 1 (left).

Existing approaches to human and aerial robot collaboration have largely focused on single aerial robot interactions [18], [20], [21]. When considering multiple aerial robots, teleoperation becomes a common solution [6], [22]. However, few solutions exist for human physical interaction and collaboration with several MAVs [23], [24]. However, the human operator’s physical collaboration is limited to a 2-D horizontal plane. Moreover, these approaches overlook the potential of exploiting the additional DoF to enhance both the system’s awareness of the human’s presence.

In summary, the contributions of this article are the following.

1) We propose a novel control method that enables a team of quadrotors to manipulate a payload while exploiting the system’s redundancy to achieve secondary tasks (e.g., maintaining distances from the human operator or ensuring inter-robot separation). This solution facilitates the physical interaction between the quadrotor team and a human operator.

2) We introduce a collaborative external wrench estimator that allows the robot team to collaboratively measure an external human force input without relying on any external force sensors. In addition, we demonstrate that this approach outperforms existing state-of-the-art solutions.

3) We complement our control solution with a 6-DoF admittance controller, which utilizes the estimated human wrench. It enables physical interaction between a human operator and a team of aerial robots for collaborative manipulation and transportation tasks.

4) We experimentally demonstrate for the first time, to the best of authors’ knowledge that our approach enables a quadrotor team to physically collaborate with a human in manipulating a payload in all 6 DoF in collaborative human–robot transportation and manipulation tasks.

The rest of this article is organized as follows. In Section II, we review relevant literature on cooperative aerial manipulation and physical human–robot interaction. In Section IV, we review the nonlinear system dynamics, considering the external wrench from a human operator. In Section V, we discuss the proposed human-aware control framework that considers the nonlinear system dynamics. Section VI details the state estimation strategy and admittance control framework for intended human–aerial– robot collaborative manipulation. Section VII presents realworld experiment results validating the proposed framework. Finally, Section IX concludes this article.

## II. RELATED WORKS

## A. Cooperative Aerial Manipulation

In the subsequent discussion, we focus on the existing related works on control, planning, and estimation techniques for aerial transportation and manipulation using suspended cables. This focus arises from the distinct advantages that cable mechanisms offer over other methods, as previously mentioned.

Past literature includes several control and estimation methods [11], [22], [25], [26] for autonomous aerial transportation and manipulation using multiple aerial robots equipped with cables. For example, several works [25], [26], [27], [28], [29] proposed formation controllers for a team of MAVs to fly in a desired formation when carrying the suspended-payload. The carried payload is not modeled as an integrated part ofthe system but as an external disturbance that each MAV controller tries to compensate for. Therefore, it is expected that these solutions can struggle to transport the payload to a given position. In [30], by assuming the payload is a point-mass, the authors analyze the full nonlinear dynamics of the system. Based on the dynamic model, the authors design a geometric controller to transport the payload to the desired position moving the quadrotor team to accommodate the desired load motions. Moreover, the previously mentioned methods [26], [27], [28], [30], treat the payload as a point-mass, hence restricting the manipulation capabilities to payload’s positional movements only, with no control over its orientation.

Other approaches for autonomous aerial transportation and manipulation rely on a leader–follower paradigm [31], [32], [33], [34], [35]. The leader robot follows the desired trajectory, whereas the followers maintain either a constant distance from the leader [33], or adapt to forces exerted on them when tracking their trajectory [14], [31]. However, these methods are subject to error compounding failure since they rely on the leader as a fundamental control unit for navigation. Furthermore, they cannot accurately guarantee the payload’s transportation to the desired location or manipulate its orientation.

Several works analyze the system’s complex nonlinear dynamics and mechanics and propose corresponding controllers to control the payload’s pose in 6 DoF [36], [37], [38], [39]. For example, the authors in [38] and [39] assume the system is in a quasi-static state and analyze the corresponding static system mechanics. A payload pose controller assigns the quadrotors desired position to manipulate the payload to the desired pose. In [36] and [37], the complex nonlinear dynamics in the system are thoroughly analyzed using Lagrangian mechanics. Leveraging this model, nonlinear geometric controllers enable the payload to follow the desired pose trajectory. More recently, several solutions propose optimal control strategies [40], [41]. Although all these works consider the payload a rigid body, the redundant control DoF available in the system [42] are not exploited to accommodate additional tasks like obstacle avoidance or to ensure safety distance among agents.

Some recent literature starts to investigate this aspect [7], [22], [43]. However, they are specifically designed for a team of three or four quadrotors. Masone et al. [44] attempt to leverage system redundancy for a team with an arbitrary number of aerial robots, implementing an optimization formulation for the parallel robot that exclusively optimizes the tension magnitude in the cables, without considering cable directions. However, the methodology presented in [44] does not provide clear instructions for determining cable direction. Consequently, its applicability for secondary tasks such as obstacle avoidance or spatial separation from the human operator remains ambiguous. In this work, we formulate a human-aware controller for any $n \geq 3$ quadrotors that exploit the additional system redundancy at the control level, allowing the system to achieve some secondary tasks, such as avoiding obstacles, inter-robot separation or keeping a safe distance among robots and human operators.

However, since the abovementioned methods are designed to control the payload’s pose explicitly, it is essential to have a reasonable estimation of the payload’s states (i.e., pose and twists) to be fed back into the controller to have a good tracking performance of the payload’s pose. Some estimation approaches can recover the payload pose in [22] and [45], but they rely on GPS and, therefore, cannot be employed in indoor environments or areas where the GPS signal is shadowed. Conversely, in our previous work [11], we tackle the payload pose inference problem using onboard vision sensors and IMU to obtain closed-loop control of the payload pose.

However, the proposed vision-based estimation method might be subject to onboard visual-inertial odometry drift, leading to some constant offset errors. Although such errors will not affect the stability of the system, they will influence the task performance. For instance, consider a scenario where the system is required to transport a payload to a specific location, but due to state estimation drift, there is a 1-m offset from the desired destination. In such cases, as shown in this article, our proposed method enables a local human operator to guide the system and correct the payload to the desired pose, ensuring that the payload is transported to the correct final destination.

## B. Physical Human–Robot Interaction (pHRI)

pHRI is a rapidly growing field in robotics, facilitating the collaboration between humans and robots in various scenarios such as manufacturing, healthcare, and service industry. Most of the research in this field focuses on the collaboration between a single robot and a human. For instance, this includes cooperative manipulation with a robot arm attached to a ground wheeled base [46], [47], a fixed-base robot arm [48], a humanoid robot [49], or an aerial robot with a manipulator [18], [20]. Past researchers’ works also present approaches like admittance control [20], [50], compliance control, impedance control for single-human–single-robot physical collaboration.

To increase payload capacity, the common strategies are either deploying a more powerful robot or utilizing a team of robots. The latter approach not only increases redundancy and provides the potential for fault tolerance, but also brings in specialized capabilities that enhance the team’s resilience and performance compared to a single robot. However, employing a team of robots demands effective coordination and collaboration across estimation, planning, and control levels.

While multiple robot–human interactions with teleoperation via haptic devices [51] or mixed reality glasses [52], [53] is a widely explored research topic, physical human–multi–robot cooperative manipulation remains mostly underexplored [54]. Recently, a handful of works started to research on direct physical interaction between a human and multiple ground robots to cooperatively manipulate or transport an object [54], [55], [56]. Carey and Werfel [56] demonstrated that 4 omnidirectional ground robots with robot arms can physically collaborate with a human using their proposed force-mediated controller but only simulation results were presented. Elwin et al. [54] designed an omni-robot system that can lift payloads. Then, the multiple proposed omni-robots can collaborate with humans to manipulate objects toward desired locations. Sirintuna et al. [55] used two robot arms with omnidirectional wheeled base to physically collaborate with a human to manipulate oversize objects. They introduce an admittance control module on both ground robots to adapt the human motion as the human leads the manipulation. However, the system dynamics for aerial robots are inherently different from ground robots as aerial robots move in 3-D. Hence, the established dynamics models and corresponding control methods for ground robots cannot be directly translated to aerial robots. Moreover, for the SWAP-constrained aerial robots, designing a lightweight, computationally efficient sensing strategy is also essential.

Rastgoftar and Atkins [23] proposed a framework for physical human–robot collaborative transportation of cable-suspended payload with a team of quadrotors. The proposed approach models the payload as a point mass and assumes external forces applied on the payload to be constrained in 2-D. Leveraging their previous work [33], the designed controller assigns three quadrotors as leaders and the remaining robots as followers in the team. When an external force is applied to the payload, a fixed step is given to the leaders’ positions along the estimated force direction. In [24], five quadrotors collaborate with a human operator to transport a point mass payload. The force applied by a human on the payload is estimated by summing the cable tension forces and subtracting the gravity. The human-applied force is fed into an admittance controller, which updates the desired quadrotor position and velocity in the formation. However, compared to our work, the cable tension magnitude is measured by a custom tension measurement module, and its direction by a motion capture system. Moreover, the aforementioned works present substantial limitations as they presume the payload to be a point mass and constrain the human operator to manipulate the payload in only 2-D. On the contrary, in our proposed work, we widen the scope of physical interactions between the human operator and the payload to all 6 DoF. This enhancement is achieved by modeling the payload as a rigid body with 6 DoF and developing an estimator to estimate the full 6 DoF wrench acting on the payload. Furthermore, our system removes the use of any force-measuring devices on the robots or payload (except to obtain the ground truth during testing for validating our estimation approach). These are unique characteristics that increase the flexibility and applicability of our solution compared to existing ones.

![](images/9da4b54244298e6b978d46ca86056d1a0ecfea297d66531856949f32c26cc485.jpg)  
Fig. 2. Frame convention: , , <sub>k</sub> denote the world frame, the payload body frame, and the kth robot body frames, respectively, for a generic quadrotor team that’s cooperatively transporting and manipulating a cable-suspended payload.

## III. OVERVIEW

In this section, we present an overview of our proposed system designed to enable a team of quadrotors to collaborate with a human operator to manipulate a rigid-body payload. Depicted in Fig. 2, the system comprises n quadrotors, and each quadrotor is tethered by a cable to its center of mass. The software architecture of the system encompasses couple of primary components: planning and control, and pHRI, as shown in Fig. 3. We provide a more detailed description of each module as follows.

## A. Planning and Control

The planning module generates a desired trajectory for the payload, encompassing both position and orientation. The trajectory is represented by a polynomial trajectory of time. By differentiating the polynomial with respect to time, we can obtain payload state’s respective first and second derivatives. These derivatives represent the twist and acceleration of the payload pose. We use the similar planning method as the one we used in our previous work to generate the trajectory for payload [57]. The desired values will be fed into the admittance controller. If human interaction with the payload occurs, the admittance controller will update the trajectory in response to the human inputs.

![](images/330f11b783e6f3d4a80e343a139c0d727e80ee1175b6f9d7e47728fcd894c3e8.jpg)  
Fig. 3. Block diagram of the system illustrating the overview of the system. It begins with a trajectory generator, which outputs the desired trajectory of the payload in all 6 DoF. The payload admittance controller updates this trajectory based on the estimated human input, adapting to the interaction wrench exerted by the human on the payload. The modified trajectory is then passed to the payload controller as the desired payload state to track. The payload controller calculates the desired wrench to control the payload’s motion. Subsequently, the dynamic force distribution module allocates the desired cable forces based on the human’s position and the system dynamics, and distributes the desired cable forces for each robot to track. Each robot controller will then track its corresponding desired cable force and output the corresponding thrust and moment commands to the quadrotor platform.

The control proposed in this article, described in Section V, adopts a hierarchical design that comprises the payload tracking controller, dynamic force distribution, and robot controller. The hierarchical controller design offers several advantages. First, it simplifies the complex task of controlling the entire system by breaking it down into several manageable subtasks. At the base of our design is the robot controller, which computes the control actions based on local information, such as cable direction and robot orientation. This strategy allows us to first test the individual robot controller with a single robot, thereby ensuring we do not risk compromising the entire system. After successfully building and testing the robot controller, we can proceed to test the higher level modules, including the payload controller and dynamic force distribution.

Moreover, the modularity enhances the system’s maintainability and scalability, as changes or improvements to one part of the control system can be implemented without affecting other components. For example, in our case, we can introduce the adaptation module at the payload level without changing the robot controller.

In the following sections, we will provide a more detailed overview of the controllers within the control design.

1) Payload Controller: The payload controller’s function is to control the payload’s state, enabling it to track the adapted desired payload states from the admittance controller. It generates the desired manipulation force and moment on the payload to track the desired payload states.

2) Dynamic Force Distribution: The dynamic force distribution dynamically distributes the desired cable tension forces that each quadrotor needs to exert to manipulate the payload. It consists of two parts: nominal force distribution and human-aware force modification.

Nominal force distribution: The nominal module processes the desired forces and moments on the payload, as computed by the payload controller. It maps the desired wrench on the payload into nominal desired cable tension forces for each quadrotor.

Human-aware force distribution: This module leverages the redundancy inherent in a multiquadrotor system to adjust the nominal cable tension forces calculated by the nominal tension distribution module. It modifies the nominal cable tension forces without impacting the overall manipulation forces and moments exerted on the payload computed by the payload controller.

3) Robot Controller: Each quadrotor runs an individual robot controller to track the desired cable tension forces assigned from the dynamic force distribution module. It commands thrust and moment to the quadrotor such that the quadrotor can track the desired cable direction as well as exert the desired tension in the cable.

## B. Physical Human–Robot Interaction

The human interaction module proposed in this article in Section VI, comprises two main parts: the admittance controller and human input estimation.

## C. Robot State Estimation

Each robot runs an onboard unscented Kalman filter (UKF) to estimate the robot’s position, velocity, orientation, angular velocity, and the cable force applied to the robot. The cable force estimated by each quadrotor will be shared among the team to collaboratively estimate the human wrench.

1) Human Wrench Estimation: This component estimates the human’s input wrench on the payload by collecting all the cable forces estimated by each quadrotor. Given the challenges associated with adding additional force sensors to the payload for real-world applications, we aim to minimize modifications to maintain cost-effectiveness. Our innovative solution, which does not require any additional force sensor, involves each quadrotor estimating its cable force. And the estimated cable force is then shared among the team to deduce the total forces and moments from all cables, further derive the human input wrench.

2) Payload Admittance Controller: The admittance controller modifies the desired payload trajectory based on humanapplied forces and moments. This would allow the system to adapt to the human input, leading to collaborative transportation and manipulation of the payload between the human and the team of quadrotor robots.

## IV. SYSTEM DYNAMICS

This section presents the modeling of the overall system dynamics. We consider a scenario where a team of n quadrotors cooperatively manipulates a rigid body payload, as illustrated in Fig. 2. We establish the world frame on the ground. The payload frame  is located at the payload’s center of mass. The payload’s position and orientation relative to  are denoted by $\mathbf { x } _ { L }$ and $R _ { L }$ , respectively.

The relevant variables in this article are summarized in the Nomenclature. We denote the three elements of any 3-D vector using subscripts $^ { * } x , y , z$

The system dynamics models are developed based on the following assumptions.

1) Aerodynamic interactions with the ground and other effects caused by high robot velocity are ignored due to its insignificant effect at a low moving speed that’s achieved by this system.

2) Each cable is assumed to be attached at the center of mass of each robot, each robot’s center of gravity coincides with its geometrical center, and all cables are assumed to be massless with no dynamic effects on the system.

3) Wind disturbances are ignored, the human operator would only interact with the payload, and all external forces on the payload and each robot are considered to be exerted by a human operator.

Assumption 1 is justified by the operational velocity, and spatial separation among agents we propose in Section V-B-2, which minimizes aerodynamic effects. Assumption 2 is based on the symmetrical design of the MAV and the lightweight nature of the cables. Finally, Assumption 3 is relevant due to our focus on indoor operation.

## A. Basic Geometry

As shown in Fig. 2, the kth quadrotor attached one massless cable with length $l _ { k }$ from its center of mass to the kth attach point on the payload. The location of the attach point k with respect to $\mathcal { L }$ and is represented by constant vector ${ \boldsymbol \rho } _ { k } \in \mathbb { R } ^ { 3 }$ and vector $\mathbf { p } _ { \mathrm { a t t , k } } \in \mathbb { R } ^ { 3 }$ , respectively. Hence, from the geometry,

we can have

$$
\mathbf {p} _ {\mathrm{att,k}} = \mathbf {x} _ {L} + R _ {L} \boldsymbol {\rho} _ {k}.\tag{1}
$$

As we can also observe from Fig. 2, when the cable is taut, the robot’s position can be obtained by using the attach point position and the cable as follows:

$$
\mathbf {x} _ {k} = \mathbf {p} _ {\mathrm{att,k}} - l _ {k} \boldsymbol {\xi} _ {k}\tag{2}
$$

where $\mathbf { x } _ { k }$ is the kth quadrotor’s position in $\mathcal { T } ,$ and $\xi _ { k }$ is the unit vector that represents the cable direction from the robot to the corresponding attach point in .

## B. Payload Dynamics

The net force $\mathbf { F } _ { L }$ in and moment ${ \bf M } _ { L }$ in $\mathcal { L }$ on the payload is determined by all the cable tension forces $\mu _ { k } , k = 1 , \ldots , n .$ gravitational pull $\mathbf { g } ,$ and external wrench $\mathbf { F } _ { H } , \mathbf { M } _ { H }$ applied by the human operator

$$
\left[ \begin{array}{c} \mathbf {F} _ {L} \\ \mathbf {M} _ {L} \end{array} \right] = \left[ \begin{array}{c} \mathbf {F} _ {H} \\ \mathbf {M} _ {H} \end{array} \right] + \mathbf {P} \boldsymbol {\mu} - \left[ \begin{array}{c} m _ {L} \mathbf {g} \\ \mathbf {0} \end{array} \right], \quad \boldsymbol {\mu} = \left[ \begin{array}{c} \boldsymbol {\mu} _ {1} \\ \vdots \\ \boldsymbol {\mu} _ {n} \end{array} \right]\tag{3}
$$

where $m _ { L }$ is the payload mass, $\mathbf { g } = g \mathbf { e } _ { 3 } , g = 9 . 8 1 \mathrm { m } / \mathrm { s } ^ { 2 } , \mathbf { e } _ { 3 } =$ $\left[ 0 \quad \begin{array} { l l l } { 0 } & { } & { 1 } \end{array} \right] ^ { \top }$ . In (3), the matrix $\mathbf { P } \in \mathbb { R } ^ { 6 \times 3 n }$ maps tension vectors of all n MAVs in to the wrench on the payload with force in and moments in

$$
\mathbf {P} = \left[ \begin{array}{c c c c} \mathbf {I} _ {3 \times 3} & \mathbf {I} _ {3 \times 3} & \dots & \mathbf {I} _ {3 \times 3} \\ \hat {\rho} _ {1} \boldsymbol {R} _ {L} ^ {\top} & \hat {\rho} _ {2} \boldsymbol {R} _ {L} ^ {\top} & \dots & \hat {\rho} _ {n} \boldsymbol {R} _ {L} ^ {\top} \end{array} \right].\tag{4}
$$

where $\mathbf { I } _ { 3 \times 3 } \in \mathbb { R } ^ { 3 \times 3 }$ is an identity matrix and the hat map $\hat { \cdot } :$ $\mathbb { R } ^ { 3 } \to { \mathfrak { s o } } ( 3 )$ is defined such that $\pmb { \hat { a } } \mathbf { b } = \pmb { a } \times \mathbf { b } , \forall \pmb { a } , \mathbf { b } \in \mathbb { R } ^ { 3 }$

By inspecting the matrix P, we observe that P has 6 rows, independent of the number of robots in the system. For $n \geq 3$ robots, the number of columns of P surpasses the number of rows, causing the dimension of the domain of P to exceed that of its image. Hence, there is an additional nullity in matrix P, which can be represented by the null space of the matrix P, denoted as $\mathcal { N } ( \mathbf { P } ) \subset \mathbb { R } ^ { 3 n - 6 }$ . The system can utilize the nullity to accomplish secondary tasks, such as obstacle avoidance, inter-robot separation, or keeping a distance between human and robot [58], which we will introduce in Section V.

Through standard rigid body dynamics, we can obtain the translational and rotational dynamics of the payload as

$$
m _ {L} \ddot {\mathbf {x}} _ {L} = \mathbf {F} _ {L}, \quad \mathbf {J} _ {L} \dot {\boldsymbol {\Omega}} _ {L} = \mathbf {M} _ {L} - \boldsymbol {\Omega} _ {L} \times \mathbf {J} _ {L} \boldsymbol {\Omega} _ {L}\tag{5}
$$

where $\ddot { \mathbf { x } } _ { L } , \dot { \Omega } _ { L } \in \mathbb { R } ^ { 3 }$ is the payload linear and angular acceleration, respectively, $\Omega _ { L } \in \mathbb { R } ^ { 3 }$ is the payload angular velocity, and $\mathbf { J } _ { L } \in \mathbb { R } ^ { \mathbf { \hat { 3 } } \times 3 }$ is the payload’s inertia matrix.

## C. Quadrotor Dynamics

Based on Assumptions 2 and 3, we consider the translational and rotational dynamics of the kth quadrotor as follows:

$$
m _ {k} \ddot {\mathbf {x}} _ {k} = \mathbf {u} _ {k} - \pmb {\mu} _ {k} - m _ {k} \mathbf {g}\tag{6}
$$

$$
\mathbf {J} _ {k} \dot {\boldsymbol {\Omega}} _ {k} = \mathbf {M} _ {k} - \boldsymbol {\Omega} _ {k} \times \mathbf {J} _ {k} \boldsymbol {\Omega} _ {k}\tag{7}
$$

where $\mathbf { u } _ { k }$ and ${ { \bf { M } } _ { k } }$ are the control force and moment on the kth quadrotor and $\pmb { \mu _ { k } } = - \mu _ { k } \pmb { \xi } _ { k }$ is the tension force applied on the kth quadrotor.

When the cable is taut, the motion of kth quadrotor is constrained to the surface of a sphere centered at the kth attach point, with a radius equal to the cable length [42]. The dynamics of the cable direction can then be derived using the Lagrange d’Alembert principle as follows [9]:

$$
\ddot {\pmb {\xi}} _ {k} = \frac {1}{m _ {k} l _ {k}} \hat {\pmb {\xi}} _ {k} ^ {2} (\mathbf {u} _ {k} - m _ {k} \mathbf {a} _ {k}) - \left\| \dot {\pmb {\xi}} _ {k} \right\| _ {2} ^ {2} \pmb {\xi} _ {k}\tag{8}
$$

where $\mathbf { \delta } _ { a _ { k } }$ is the acceleration of the kth attachment point

$$
\pmb {a} _ {k} = \ddot {\mathbf {x}} _ {L} + \mathbf {g} - R _ {L} \hat {\rho} _ {k} \dot {\Omega} _ {L} + R _ {L} \hat {\Omega} _ {L} ^ {2} \rho_ {k}.\tag{9}
$$

## V. CONTROL

In this section, we introduce a hierarchical nonlinear controller that enables a team of n quadrotors to manipulate a rigidbody load suspended by cables. The formulation ofthe controller is based on the system dynamics presented in Section IV. Fig. 3 illustrates the hierarchical structure of the controller.

The hierarchy begins with a payload controller, detailed in Section $\mathrm { V } { \mathrm { - } } \mathrm { A } .$ which generates the desired wrenches $\mathbf { F } _ { L , \mathrm { d e s } } , \mathbf { M } _ { L , \mathrm { d e s } }$ to control the position and orientation of the payload. Subsequently, the dynamic force distribution module, described in Section V-B, assigns desired cable force vectors $\mu _ { k } , k = 1 , \ldots , n$ for each robot, based on the desired payload wrench $\mathbf { F } _ { L , \mathrm { d e s } } , \mathbf { M } _ { L , \mathrm { d e s } }$ . The control hierarchy finishes with the robot controller at its lowest level, where the individual robot controller on each robot tracks its corresponding desired cable force vector $\mu _ { k }$ . Each robot controller, associated with the kth robot, computes the appropriate thrust and moment commands for the robot, as further shown in Section V-C.

## A. Payload Controller

We present a payload controller that enables the load to follow the desired trajectory in a closed loop. The subscript $^ { * } \mathrm { d e s }$ denotes the desired value given by the trajectory planner. The desired forces and moments acting on the payload are designed as

(10)

$$
\begin{array}{l} \mathbf {F} _ {L, \mathrm{des}} = m _ {L} \boldsymbol {a} _ {L, c} \\ \boldsymbol {a} _ {L, c} = \mathbf {K} _ {p} \mathbf {e} _ {\mathbf {x} _ {L}} + \mathbf {K} _ {d} \mathbf {e} _ {\dot {\mathbf {x}} _ {L}} + \mathbf {K} _ {i} \int_ {0} ^ {t} \mathbf {e} _ {\mathbf {x} _ {L}} d \tau + \ddot {\mathbf {x}} _ {L, \mathrm{des}} + \mathbf {g} \\ \mathbf {M} _ {L, \mathrm{des}} = \mathbf {K} _ {\boldsymbol {R} _ {L}} \mathbf {e} _ {\boldsymbol {R} _ {L}} + \mathbf {K} _ {\boldsymbol {\Omega} _ {L}} \mathbf {e} _ {\boldsymbol {\Omega} _ {L}} + \mathbf {J} _ {L} \boldsymbol {R} _ {L} ^ {\top} \boldsymbol {R} _ {L, \mathrm{des}} \dot {\boldsymbol {\Omega}} _ {L, \mathrm{des}} \\ + (\boldsymbol {R} _ {L} ^ {\top} \boldsymbol {R} _ {L, \mathrm{des}} \boldsymbol {\Omega} _ {L, \mathrm{des}}) ^ {\wedge} \mathbf {J} _ {L} \boldsymbol {R} _ {L} ^ {\top} \boldsymbol {R} _ {L, \mathrm{des}} \boldsymbol {\Omega} _ {L, \mathrm{des}} ( \end{array}\tag{11}
$$

where ${ \bf K } _ { p } , { \bf K } _ { d } , { \bf K } _ { i } , { \bf K } _ { R _ { L } } , { \bf K } _ { \Omega _ { L } } \in \mathbb { R } ^ { 3 \times 3 }$ are constant diagonal positive definite matrices, and

$$
\begin{array}{r l} & {\mathbf {e _ {x}} _ {L} = \mathbf {x} _ {L, \mathrm{des}} - \mathbf {x} _ {L}, \mathbf {e } = \dot {\mathbf {x}} _ {L, \mathrm{des}} - \dot {\mathbf {x}} _ {L}} \\ & {\mathbf {e _ {R}} _ {L} = \frac {1}{2} \left(\boldsymbol {R} _ {L} ^ {\top} \boldsymbol {R} _ {L, \mathrm{des}} - \boldsymbol {R} _ {L, \mathrm{des}} ^ {\top} \boldsymbol {R} _ {L}\right) ^ {\vee}} \\ & {\mathbf {e _ {\Omega}} _ {L} = \boldsymbol {R} _ {L} ^ {\top} \boldsymbol {R} _ {L, \mathrm{des}} \boldsymbol {\Omega} _ {L, \mathrm{des}} - \boldsymbol {\Omega} _ {L}.} \end{array}\tag{12}
$$

In the abovementioned equation, the vee map $ { \vee } : \mathfrak { s o } ( 3 ) \to \mathbb { R } ^ { 3 }$ is the reverse of the hat map ˆ.

## B. Dynamic Force Distribution

In this section, we present our dynamic force distribution method that allocates the desired payload force $\mathbf { F } _ { L , \mathrm { d e s } }$ and moment ${ \bf M } _ { L , \mathrm { d e s } }$ to the desired cable tension forces $\mu _ { k }$ . The force distribution comprises two segments: the nominal force distribution and the human-aware force distribution.

The nominal force distribution, originated from nonlinear geometric control method [9], [11], distributes the desired payload wrench to the desired cable tension forces using minimum norm solution.

On the other hand, the human-aware force distribution leverages the system redundancy, given that the robot number $n \geq 3 .$ to obtain the cable forces that yield a zero effective wrench on the payload. This adjustment enables the system to modify the desired cable forces from the nominal distribution for secondary objectives, such as maintaining a distance from the robots to the human operator and between the robots themselves, without impacting the original manipulation tasks.

In the following, we will first present the nominal force distribution method in Section V-B-1, followed by the human-aware force distribution method in Section V-B-2.

1) Nominal Force Distribution: Once the desired payload wrench $\mathbf { F } _ { L , \mathrm { d e s } } , \mathbf { M } _ { L , \mathrm { d e s } }$ is obtained, it can be distributed to the desired tension force $\bar { \mu } _ { k , \mathrm { d e s } }$ along each cable as

$$
\bar {\boldsymbol {\mu}} _ {\mathrm{des}} = \left[ \begin{array}{c} \bar {\boldsymbol {\mu}} _ {1, \mathrm{des}} \\ \vdots \\ \bar {\boldsymbol {\mu}} _ {n, \mathrm{des}} \end{array} \right] = \mathbf {P} ^ {\dagger} \left[ \begin{array}{c} \mathbf {F} _ {L, \mathrm{des}} \\ \mathbf {M} _ {L, \mathrm{des}} \end{array} \right]\tag{13}
$$

where ${ \bf P } ^ { \dagger } = { \bf P } ^ { \top } ( { \bf P } { \bf P } ^ { \top } ) ^ { - 1 }$ is the Moore–Penrose inverse of P. The abovementioned solution can be directly used as the desired cable tension vector for the robot, like in our previous works [9], [11]. However, the abovementioned solution does not exploit the possibility of the quadrotor team’s needs to accomplish secondary tasks. For example, the second task can be avoiding obstacles or, as shown in this article, spatially separating the human and the robots during physical collaboration.

2) Human-Aware Force Distribution: As we have mentioned before, the human-aware force distribution exploits the system redundancy to modify the desired cable forces from the nominal distribution for secondary tasks. The secondary tasks can be maintaining a distance from the robots to the human operator and between the robots themselves, without impacting the original manipulation tasks.

To accomplish these, we propose and discuss two distinct approaches in this section. The system aims to allocate the wrench to the desired cable tension forces with the following two principal goals in mind.

1) Minimize the total cable tension forces to conserve the robot’s energy.

2) Utilize the null space of the cable distribution matrix P to facilitate secondary tasks, such as ensuring a safety distance between the robots and potential human operators within the team.

In the following, we introduce two methods enabling each MAV to maintain distance from an object. Concurrently, these methods allow the team to maintain the original desired load forces, $\mathbf { F } _ { L , \mathrm { d e s } }$ , and moments, ${ \mathbf { M } } _ { L , \mathrm { d e s } }$ , as outlined in Section V-A. This approach ensures that the original objectives in load manipulation are not compromised.

The methods leverage the redundancy in system configurations involving more than three MAVs, as explained in Section IV. We intend to find a desired tension force modifier, $\tilde { \mu } _ { \mathrm { d e s } } \in \mathcal { N } ( \mathbf { P } )$ that modifies $\bar { \mu } _ { \mathrm { d e s } }$ in (13), and satisfies

$$
\mathbf {P} \tilde {\boldsymbol {\mu}} _ {\mathrm{des}} = \mathbf {0}, \quad \tilde {\boldsymbol {\mu}} _ {\mathrm{des}} = \left[ \begin{array}{c} \tilde {\boldsymbol {\mu}} _ {1, \mathrm{des}} \\ \vdots \\ \tilde {\boldsymbol {\mu}} _ {n, \mathrm{des}} \end{array} \right].\tag{14}
$$

Equation (14) means the tension modifiers result in zero wrench on the payload, which does not affect the original desired manipulation wrench from (10) and (11). With the tension force modifier, we can update kth robot’s desired cable tension force as

$$
\boldsymbol {\mu} _ {k, \text { des }} = \bar {\boldsymbol {\mu}} _ {k, \text { des }} + \tilde {\boldsymbol {\mu}} _ {k, \text { des }}.\tag{15}
$$

Intuitively, $\mathcal { N } ( \mathbf { P } )$ provides all the possible combinations of n cable tension vectors that can generate internal motions of the structure (i.e., variations of the cables’ directions) that do not affect the load configuration controlled by the method presented in Section V-A. This is confirmed by (3) and (15), as $\bar { \pmb { \mu } } _ { k }$ ,des would create a nonzero net wrench on the payload while $\tilde { \mu } _ { k }$ ,des creates zero net wrench. Moreover, $\tilde { \mu } _ { k , \mathrm { d e s } }$ can be related to the position of each robot

$$
\mathbf {p} _ {\mathrm{att,k}} + l _ {k} \tilde {\boldsymbol {\xi}} _ {k, \mathrm{des}} = \mathbf {x} _ {k}\tag{16}
$$

and

$$
\tilde {\boldsymbol {\xi}} _ {k, \text { des }} = - \boldsymbol {\xi} _ {k, \text { des }} = \frac {\bar {\boldsymbol {\mu}} _ {k , \text { des }} + \tilde {\boldsymbol {\mu}} _ {k , \text { des }}}{\| \bar {\boldsymbol {\mu}} _ {k , \text { des }} + \tilde {\boldsymbol {\mu}} _ {k , \text { des }} \|}.\tag{17}
$$

The advantage is that we can exploit $\tilde { \mu } _ { k , \mathrm { d e s } }$ to enforce the kth robot to maintain a certain distance with respect to the other agents in the system and other objects in the environments, like a potential human operator.

Therefore, the human-aware force distribution needs to find the aforementioned tension force modifier, $\tilde { \mu } _ { \mathrm { d e s } } \in \mathcal { N } ( \mathbf { P } )$ , and use (15) to move MAVs based on (16) without affecting the payload. We propose the following two approaches for finding $\tilde { \mu } _ { \mathrm { d e s } }$

1) Gradient-Based Method: Find a $\tilde { \mu } _ { \mathrm { d e s } }$ such that each MAV maximizes the distance between itself and the object using a gradient ascent method.

2) Optimization-Based Method: Find a $\tilde { \mu } _ { \mathrm { d e s } }$ such that each MAV guarantees a predetermined minimal safe distance between all its neighboring drones and the object by using nonlinear optimization.

In the following, we can describe the obstacle or human operator as a particular object of interest in the environment. The corresponding point position with respect to is denoted as p<sub>O</sub> in the following controller formulation.

Gradient-based method: Inspired by strategies used for redundant rigid link robot arms in [59], we introduce a gradient-based method to compute $\tilde { \mu } _ { \mathrm { d e s } }$ . Specifically, a pseudotension force modifier, $\mu _ { \mathrm { d e s } } ^ { 0 } ,$ is found by maximizing the distance between objects in the environment and each drone. $\mu _ { \mathrm { d e s } } ^ { 0 }$ is then projected into $\mathcal { N } ( \mathbf { P } )$ to become the tension force modifier $\tilde { \mu } _ { \mathrm { d e s } }$ . To update $\mu _ { \mathrm { d e s } } ^ { 0 } .$ , we propose

$$
\boldsymbol {\mu} _ {\mathrm{des}} ^ {0} = \mathbf {Q} \frac {\partial \mathbf {w} (\tilde {\boldsymbol {\mu}} _ {\mathrm{des}})}{\partial \tilde {\boldsymbol {\mu}} _ {\mathrm{des}}}\tag{18}
$$

where $\mathbf { Q } \in \mathbb { R } ^ { 3 n \times 3 n }$ is a diagonal positive-definite matrix with variable and tunable coefficients on its diagonal, and $\mathbf { w } ( \tilde { \pmb { \mu } } _ { \mathrm { d e s } } )$ is the cost function. w $\left( \tilde { \mu } _ { \mathrm { d e s } } \right)$ is defined as the squared distance between each drone and the object from which the system is required to keep a safe distance as follows:

$$
\mathbf {w} \left(\tilde {\boldsymbol {\mu}} _ {\mathrm{des}}\right) = \sum_ {i = 1} ^ {k} \left\| \mathbf {p} _ {O} - \left(\mathbf {p} _ {\text {att}, \mathrm{k}} + \mathbf {l} _ {k} \tilde {\boldsymbol {\xi}} _ {k, \mathrm{des}}\right) \right\| ^ {2}\tag{19}
$$

where $\mathbf { l } _ { k } = l _ { k } \mathbf { I } _ { 3 \times 3 }$ . Note that $\tilde { \xi } _ { k , \mathrm { d e s } }$ points from the kth attach point to the kth robot. We can now compute the partial derivative of the cost function corresponding to the kth robot by using (19) and obtain the following:

$$
\begin{array}{l} \frac {\partial \mathbf {w} (\tilde {\boldsymbol {\mu}} _ {k , \mathrm{des}})}{\partial \tilde {\boldsymbol {\mu}} _ {k , \mathrm{des}}} = - 2 \mathbf {l} _ {k} \left[ \mathbf {p} _ {O} - \mathbf {p} _ {\mathrm{att}, \mathrm{k}} \right] ^ {\top} \frac {\partial \tilde {\boldsymbol {\xi}} _ {\mathrm{des}}}{\partial \tilde {\boldsymbol {\mu}} _ {k , \mathrm{des}}} \\ = - 2 \mathbf {l} _ {k} \left[ \mathbf {p} _ {O} - \mathbf {p} _ {\mathrm{att}, \mathrm{k}} \right] ^ {\top} \frac {(\mathbf {I} _ {3 \times 3} - \tilde {\boldsymbol {\xi}} _ {k , \mathrm{des}} \tilde {\boldsymbol {\xi}} _ {k , \mathrm{des}} ^ {\top})}{\| \bar {\boldsymbol {\mu}} _ {k , \mathrm{des}} + \tilde {\boldsymbol {\mu}} _ {k , \mathrm{des}} \|}. \end{array}\tag{20}
$$

For each control step, we update $\mu _ { \mathrm { d e s } } ^ { 0 }$ based on (18), performing gradient ascent to maximize the distance between each robot and the objects in the environment. To regulate the effect of gradient on each robot when the object is far away, we propose each element of Q as an exponential decay function of the robot-toobject distance

$$
\mathbf {Q} = \mathrm{diag} \left(\mathbf {Q} _ {1}, \ldots , \mathbf {Q} _ {n}\right), \quad \mathbf {Q} _ {k} = a e ^ {- b \| \mathbf {p} _ {O} - \mathbf {x} _ {k} \|} \mathbf {I} _ {3 \times 3}\tag{21}
$$

where $a , b \in \mathbb { R }$ are tunable coefficients. With these varying coefficients, we can implicitly impose distance limits as the gradients will only have impact on the tension modification when kth robot is close enough to the object. However, till now, $\mu _ { \mathrm { d e s } } ^ { 0 }$ from (18) is not yet in $\mathcal { N } ( \mathbf { P } )$ . Hence, we consider the following optimization to project $\mu _ { \mathrm { d e s } } ^ { 0 }$ into the null space of the matrix P

$$
\begin{array}{l l} \min _ {\boldsymbol {\mu} _ {\mathrm{des}}} & \| \boldsymbol {\mu} _ {\mathrm{des}} ^ {0} - \tilde {\boldsymbol {\mu}} _ {\mathrm{des}} \| ^ {2} \\ \text {s.t.} & \mathbf {P} \tilde {\boldsymbol {\mu}} _ {\mathrm{des}} = 0. \end{array}\tag{22}
$$

Since the abovementioned optimization problem is a quadratic programming problem with linear equality constraints, there is a closed-form solution shown as follows [60]:

$$
\tilde {\boldsymbol {\mu}} _ {\mathrm{des}} = \mathbf {B} \boldsymbol {\mu} _ {\mathrm{des}} ^ {0} = \left(\boldsymbol {I} - \mathbf {P} ^ {\dagger} \mathbf {P}\right) \boldsymbol {\mu} _ {\mathrm{des}} ^ {0}\tag{23}
$$

where $\mathbf { P } ^ { \dagger }$ is the pseudoinverse as in (13), and B is an orthogonal projector that projects any $\mu _ { \mathrm { d e s } } ^ { 0 }$ orthogonally into the null space of P.

Using this result, we project $\mu ^ { 0 }$ into $\mathcal { N } ( \mathbf { P } )$ with (23), ensuring zero additional wrenches being applied on the payload when each robot is maximizing distance from the object. Finally, we update the desired tension vector as

$$
\boldsymbol {\mu} _ {\text { des }} = \bar {\boldsymbol {\mu}} _ {\text { des }} + \tilde {\boldsymbol {\mu}} _ {\text { des }} = \bar {\boldsymbol {\mu}} _ {\text { des }} + \mathbf {B} \boldsymbol {\mu} _ {\text { des }} ^ {0}.\tag{24}
$$

Optimization-based method: In this section, we directly formulate an optimization problem to solve for a tension force modifier $\tilde { \mu } _ { \mathrm { d e s } } \mathrm { i n } \mathcal { N } ( \mathbf { P } )$ that guarantees safety distance among the objects and the robots. The nonlinear optimization problem is to minimize the total square norm of the resulting cable tension vector. Furthermore, we formulate n robot-to-object distance constraints, as well as another $\textstyle { { \binom { n } { 2 } } = { \frac { n ( n - 1 ) } { 2 } } }$ constraints are added to prevent each pair of robots from collision. Consider the following nonlinear optimization problem:

$$
\begin{array}{r l} \underset {\mathbf {c}} {\min} & \| \bar {\boldsymbol {\mu}} _ {\mathrm{des}} + \mathbf {N} \boldsymbol {\Lambda} \| ^ {2} \\ \text {s.t.} & \| \mathbf {p} _ {O} - \mathbf {x} _ {k} \| ^ {2} \geq^ {\mathrm{h}} r ^ {2}, 0 <   k \leq n \\ & \| \mathbf {x} _ {i} - \mathbf {x} _ {j} \| ^ {2} \geq^ {\mathrm{r}} r ^ {2}, 0 <   i <   j \leq n \end{array}\tag{25}
$$

where the columns of N spans $\mathcal { N } ( \mathbf { P } )$ and $\mathbf { c } \in \mathbb { R } ^ { 3 n - 6 }$ is the vector to be optimized. <sup>r</sup>r and $\mathrm { h } _ { r }$ are two scalar values denoting the predetermined safe minimum distance allowed between robots and between the object and each robot, respectively. The kth robot’s position is expressed in terms of ${ \bf N } _ { k } \Lambda$ and $\bar { \mu } _ { k , \mathrm { d e s } }$

$$
\mathbf {x} _ {k} = \mathbf {p} _ {\text {att,k}} + l _ {k} \frac {\bar {\boldsymbol {\mu}} _ {k , \text {des}} + \mathbf {N} _ {k} \boldsymbol {\Lambda}}{\| \bar {\boldsymbol {\mu}} _ {k , \text {des}} + \mathbf {N} _ {k} \boldsymbol {\Lambda} \|}\tag{26}
$$

where ${ \bf N } _ { k }$ represents the three rows from the kth row to the k + 2th row of the null space basis matrix N, which corresponds to the kth MAV. Since (25) is a nonlinear optimization problem with quadratic cost function and quadratic constraints, we use sequential quadratic programming solver for nonlinearly constrained gradient-based optimization [61] in NLOPT [62] to solve (25) and obtain c. After obtaining c, the desired cable tension forces can be obtained as follows:

$$
\boldsymbol {\mu} _ {\mathrm{des}} = \bar {\boldsymbol {\mu}} _ {\mathrm{des}} + \tilde {\boldsymbol {\mu}} _ {\mathrm{des}} = \bar {\boldsymbol {\mu}} _ {\mathrm{des}} + \mathbf {N} \boldsymbol {\Lambda}.\tag{27}
$$

Discussion: The proposed methods are both effective for the quadrotor team to keep a safe distance away from a given object, as we also experimentally verify in Section VII. However, considering computational aspects, the gradient-based method requires fewer resources compared to the optimization-based method. This is primarily due to the closed-form solution offered by the gradient-based approach, as demonstrated by (20). On the other hand, the optimization-based method needs to solve a nonlinear optimization problem. In Section VIII, we provide a quantitative analysis of the computational complexity and resource usage for both methods based on our implementation.

## C. Robot Controller

In this section, we present the controller on each robot that enables the quadrotor to execute the desired cable tension force. The same robot controller has been used in our previous works [9], [11].

Once we obtain the desired tension forces $\mu _ { \mathrm { d e s } }$ from (24) or (27), we can obtain the desired direction $\xi _ { k , \mathrm { d e s } }$ and the desired angular velocity $\omega _ { k , \mathrm { d e s } }$ of the kth cable link as

$$
\boldsymbol {\xi} _ {k, \mathrm{des}} = - \frac {\boldsymbol {\mu} _ {k , \mathrm{des}}}{\| \boldsymbol {\mu} _ {k , \mathrm{des}} \|}, \omega_ {k, \mathrm{des}} = \boldsymbol {\xi} _ {k, \mathrm{des}} \times \dot {\boldsymbol {\xi}} _ {k, \mathrm{des}}
$$

where $\dot { \xi } _ { k , \mathrm { d e s } }$ is the derivative of the desired cable direction $\xi _ { k , \mathrm { d e s } }$ . After we obtain the desired cable direction $\xi _ { k , \mathrm { d e s } }$ and cable angular velocity $\omega _ { k , \mathrm { d e s } }$ , we can determine the desired force vector for the robot $\mathbf { u } _ { k }$ as

$$
\begin{array}{l} \mathbf {u} _ {k} = \mathbf {u} _ {k} ^ {\parallel} + \mathbf {u} _ {k} ^ {\perp} \\ \mathbf {u} _ {k} ^ {\perp} = m _ {k} l _ {k} \boldsymbol {\xi} _ {k} \times \left[ - \mathbf {K} _ {\boldsymbol {\xi} _ {k}} \mathbf {e} _ {\boldsymbol {\xi} _ {k}} - \mathbf {K} _ {\omega_ {k}} \mathbf {e} _ {\omega_ {k}} - \hat {\boldsymbol {\xi}} _ {k} ^ {2} \boldsymbol {\omega} _ {k, \mathrm{des}} \right. \\ \left. - (\boldsymbol {\xi} _ {k} \cdot \boldsymbol {\omega} _ {k, \mathrm{des}}) \dot {\boldsymbol {\xi}} _ {k, \mathrm{des}} \right] - m _ {k} \hat {\boldsymbol {\xi}} _ {k} ^ {2} \boldsymbol {a} _ {k, c} \\ \mathbf {u} _ {k} ^ {\parallel} = \boldsymbol {\xi} _ {k} \boldsymbol {\xi} _ {k} ^ {\top} \boldsymbol {\mu} _ {k, \mathrm{des}} + m _ {k} l _ {k} \| \boldsymbol {\omega} _ {k} \| _ {2} ^ {2} \boldsymbol {\xi} _ {k} + m _ {k} \boldsymbol {\xi} _ {k} \boldsymbol {\xi} _ {k} ^ {\top} \boldsymbol {a} _ {k, c} \\ \boldsymbol {a} _ {k, c} = \boldsymbol {a} _ {L, c} - \boldsymbol {R} _ {L} \hat {\rho} _ {k} \dot {\Omega} _ {L} + \boldsymbol {R} _ {L} \hat {\Omega} _ {L} ^ {2} \rho_ {k} \end{array}\tag{28}
$$

where $\mathbf { K } _ { \xi _ { k } }$ and $\mathbf { K } _ { \omega _ { k } } \in \mathbb { R } ^ { 3 \times 3 }$ are constant diagonal positive definite matrices, $\mathbf { e } _ { \xi _ { k } }$ and $\mathbf { e } _ { \omega _ { k } } \in \mathbb { R } ^ { 3 }$ are the cable direction and cable angular velocity errors, respectively

$$
\mathbf {e} _ {\boldsymbol {\xi} _ {k}} = \boldsymbol {\xi} _ {k, \mathrm{des}} \times \boldsymbol {\xi} _ {k}, \mathbf {e} _ {\omega_ {k}} = \boldsymbol {\omega} _ {k} + \boldsymbol {\xi} _ {k} \times \boldsymbol {\xi} _ {k} \times \boldsymbol {\omega} _ {k, \mathrm{des}}.
$$

As we obtain the desired force vector of the quadrotor from (28), we can follow [63] to derive the desired rotation $R _ { k , \mathrm { d e s } }$ and angular velocity $\Omega _ { k , \mathrm { d e s } }$ with desired yaw angle and desired yaw angular velocity from the robot’s own planner. The thrust command $f _ { k }$ and moment command ${ { \bf { M } } _ { k } }$ to the kth quadrotor are therefore selected as

$$
\begin{array}{l} f _ {k} = \mathbf {u} _ {k} \cdot \boldsymbol {R} _ {k} \mathbf {e} _ {3} \\ \mathbf {M} _ {k} = \mathbf {K} _ {R} \mathbf {e} _ {R _ {k}} + \mathbf {K} _ {\Omega} \mathbf {e} _ {\Omega_ {k}} + \boldsymbol {\Omega} _ {k} \times \mathbf {J} _ {k} \boldsymbol {\Omega} _ {k} \\ - \mathbf {J} _ {k} \left(\hat {\boldsymbol {\Omega}} _ {k} \boldsymbol {R} _ {k} ^ {\top} \boldsymbol {R} _ {k, \mathrm{des}} \boldsymbol {\Omega} _ {k, \mathrm{des}} - \boldsymbol {R} _ {k} ^ {\top} \boldsymbol {R} _ {k, \mathrm{des}} \dot {\boldsymbol {\Omega}} _ {k, \mathrm{des}}\right) \end{array}\tag{29}
$$

(30)

where ${ \bf K } _ { R } , { \bf K } _ { \Omega } \in \mathbb { R } ^ { 3 \times 3 }$ are constant diagonal positive definite matrices, ${ \bf e } _ { R _ { k } } \in \mathbb { R } ^ { 3 }$ and $\mathbf { e } _ { \Omega _ { k } } \in \mathbb { R } ^ { 3 }$ are the orientation and angular velocity errors similarly defined using (12). The readers can refer to [42] for stability analysis of the controller.

## VI. PHYSICAL HUMAN–ROBOT INTERACTION

This section introduces the pHRI module that enables a human operator to physically cooperate with a team of n quadrotors in manipulating a suspended rigid-body payload. The module comprises two main sub-blocks: the estimation module and the admittance controller.

The estimation module is designed to facilitate the quadrotor team in estimating the human operator’s input wrench exerted on the payload. The admittance controller takes the estimated human wrench and the desired payload state as input and generates a desired payload state to adapt the human’s action.

## A. Estimation

We present the estimator design that allows the quadrotor team to estimate the external wrench applied to the payload by the human operator. First, in Section VI-A-1, we introduce a quadrotor state estimator based on UKF that runs onboard each quadrotor in a distributed fashion. Each quadrotor can leverage the estimator to estimate the cable force applied to it, without the need for a force sensor. Subsequently, in Section VI-A-2, we show how we can estimate the external wrench applied on the payload by the human operator via sharing the cable force on each quadrotor among the team.

1) Robot State Estimation: We consider the kth MAV to have the following state $\mathbf { S } _ { k }$ :

$$
\mathbf {S} _ {k} = \left[ \begin{array}{c c c c c c c} \mathbf {x} _ {k} ^ {\top} & \dot {\mathbf {x}} _ {k} ^ {\top} & \boldsymbol {\Theta} _ {k} ^ {\top} & \boldsymbol {\Omega} _ {k} ^ {\top} & \boldsymbol {\xi} _ {k} & \dot {\boldsymbol {\xi}} _ {k} ^ {\top} & \mu_ {k} \end{array} \right] ^ {\top}\tag{31}
$$

where $\Theta _ { k } \in \mathbb { R } ^ { 3 }$ is a vector of the 3 Euler angles expressed according to the ZYX convention representing the robot’s orientation. And the input is defined as

$$
\mathbf {U} _ {k} = \left[ \begin{array}{c c} f _ {k} & \mathbf {M} _ {k} ^ {\top} \end{array} \right] ^ {\top}\tag{32}
$$

where $f _ { k }$ and ${ { \bf { M } } _ { k } }$ are obtained based on motor speed measured by the electronic speed controllers on the robot. The relationship between motor speed and the resultant thrust and moment is expressed as follows:

$$
\left[ \begin{array}{c} f _ {k} \\ \mathbf {M} _ {k} \end{array} \right] = \left[ \begin{array}{c c c c} k _ {f} & k _ {f} & k _ {f} & k _ {f} \\ d _ {x} k _ {f} & d _ {x} k _ {f} & - d _ {x} k _ {f} & - d _ {x} k _ {f} \\ - d _ {y} k _ {f} & d _ {y} k _ {f} & d _ {y} k _ {f} & - d _ {y} k _ {f} \\ k _ {m} & - k _ {m} & k _ {m} & - k _ {m} \end{array} \right] \left[ \begin{array}{c} \omega_ {m 1} ^ {2} \\ \omega_ {m 2} ^ {2} \\ \omega_ {m 3} ^ {2} \\ \omega_ {m 4} ^ {2} \end{array} \right]\tag{33}
$$

where $k _ { f }$ and $k _ { m }$ represent the motor constants corresponding to rotor force and moment, respectively. $d _ { x }$ and $d _ { y }$ denote the distances from the rotor to the body’s x and y axes. In addition, $\omega _ { m j }$ signifies the motor speed of the jth motor. We denote the current time step as $\ast ^ { t }$ and the previous time step as $\ast ^ { t - 1 }$ Subsequently, we present the nonlinear process model and the linear measurement model of the UKF.

Process Model: Based on MAV equations ofmotion presented in (7), discretizations of quadrotor states are performed by assuming each control step moves forward in time by δt. The discrete-time nonlinear process model is

$$
\mathbf {S} _ {k} ^ {t}   =   g (\mathbf {S} _ {k} ^ {t - 1}, \mathbf {U} _ {k} ^ {t}) = \left[ \begin{array}{c} \mathbf {x} _ {k} ^ {t} \\ \dot {\mathbf {x}} _ {k} ^ {t} \\ \boldsymbol {\Theta} _ {k} ^ {t} \\ \boldsymbol {\Omega} _ {k} ^ {t} \\ \boldsymbol {\xi} _ {k} ^ {t} \\ \dot {\boldsymbol {\xi}} _ {k} ^ {t} \\ \mu_ {k} ^ {t} \end{array} \right] = \left[ \begin{array}{c} \mathbf {x} _ {k} ^ {t - 1} + \dot {\mathbf {x}} _ {k} ^ {t - 1} \delta t + \ddot {\mathbf {x}} _ {k} ^ {t} \frac {\delta t ^ {2}}{2} \\ \dot {\mathbf {x}} _ {k} ^ {t - 1} + \ddot {\mathbf {x}} _ {k} ^ {t} \delta t \\ \lfloor \boldsymbol {R} _ {k} ^ {t - 1} \exp \left[ \boldsymbol {R} _ {k} ^ {t - 1} \boldsymbol {\Omega} _ {k} ^ {t - 1} \delta t \right] \rfloor \\ \boldsymbol {\Omega} _ {k} ^ {t - 1} + \dot {\boldsymbol {\Omega}} _ {k} ^ {t} \delta t \\ \boldsymbol {\xi} _ {k} ^ {t - 1} + \dot {\boldsymbol {\xi}} _ {k} ^ {t} \delta t + \ddot {\boldsymbol {\xi}} _ {k} ^ {t} \frac {\delta t ^ {2}}{2} \\ \dot {\boldsymbol {\xi}} _ {k} ^ {t - 1} + \ddot {\boldsymbol {\xi}} _ {k} ^ {t} \delta t \\ \mu_ {k} ^ {t - 1} \end{array} \right]\tag{.(34}
$$

For updating the Euler angles, a few nonlinear mappings are used as in [64].

1)  that maps $* \in \mathrm { S O ( 3 ) }$ to $\Theta \in \mathbb { R } ^ { 3 }$

2) exp[ ] that maps $\ast \in \mathbb { R } ^ { 3 }$ to $\mathrm { S O ( 3 ) }$ ; or maps axis-angle so(3), to rotation matrix SO(3).

In (34), $\Omega _ { k } ^ { t - 1 }$ is rotated into . After time step δt, the robot’s angular displacement in , expressed in so(3) is mapped into $\mathrm { S O ( 3 ) }$ . This new robot orientation in $\mathrm { S O ( 3 ) }$ is then added to the previous orientation and the resultant orientation is converted to

Euler angle using . The equations of motion for unit cable direction are provided in [11], with the results presented here

$$
\ddot {\boldsymbol {\xi}} _ {k} ^ {t} = \frac {\left(\hat {\boldsymbol {\xi}} _ {k} ^ {t - 1}\right) ^ {2} \left(\mathbf {u} _ {k} - m _ {k} \boldsymbol {a} _ {k}\right)}{m _ {k} l _ {k}} - \left\| \dot {\boldsymbol {\xi}} _ {k} ^ {t - 1} \right\| _ {2} ^ {2} \boldsymbol {\xi} _ {k} ^ {t - 1}.\tag{35}
$$

In our system, we employ a $1 6 \times 1 6$ time-invariant process noise diagonal covariance matrix, operating under the assumption that the system has zero-mean, additive Gaussian process noise. For the prediction phase, the UKF algorithm utilizes (34). It is important to note that in instances where measurements are not available at every control time step, the UKF implements a nonlinear prediction method that assumes zero process noise.

Measurement Model: Using an indoor MOCAP system, we can measure everything in the state except tension magnitude. The measured states for kth robot, therefore, are

$$
\mathbf {Z} _ {k} = \left[ \begin{array}{c c c c c c} \mathbf {x} _ {k} ^ {\top} & \dot {\mathbf {x}} _ {k} ^ {\top} & \boldsymbol {\Theta} _ {k} ^ {\top} & \boldsymbol {\Omega} _ {k} ^ {\top} & \boldsymbol {\xi} _ {k} ^ {\top} & \dot {\boldsymbol {\xi}} _ {k} ^ {\top} \end{array} \right] ^ {\top}.\tag{36}
$$

We also want to note that using onboard visual inertial odometry and vision-based methods from our previous work [11] can provide the same measurements as the motion capture does and can potentially make the entire measurement update process run fully onboard. However, this is out of the scope of this article and we refer to it as future works.

Denoting the time step when the UKF measurement update is triggered as $m ,$ the UKF finds the state prior to m, $\mathbf { S } _ { k } ^ { m - 1 }$ . A nonlinear propagation through (34) is performed by propagating sigma points around $\mathbf { S } _ { k } ^ { m - 1 }$ through the system model in (34) as shown in [65].

The linear measurement model is

$$
\mathbf {H} = \left[ \begin{array}{c c} \mathbf {I} _ {1 8 \times 1 8} & \mathbf {0} _ {1 8 \times 1} \\ \mathbf {0} _ {1 \times 1 8} & \mathbf {0} _ {1 \times 1} \end{array} \right].
$$

The system state prediction is compared to the actual measurement using the abovementioned model

$$
\mathbf {S} _ {k} ^ {m} = \mathbf {K} _ {k} (\mathbf {H} \bar {\mathbf {S}} _ {k} ^ {m} - \mathbf {Z} _ {k})\tag{37}
$$

where $\bar { \mathbf { S } } _ { k } ^ { m }$ is the averaged state after sigma point propagation and ${ \bf K } _ { k }$ is the Kalman gain. Kalman gain is computed based on the standard UKF update step as in [65].

2) Human Wrench Estimation: Rearranging (3), we obtain

$$
\left[ \begin{array}{c} m _ {L} \ddot {\mathbf {x}} _ {L} \\ \mathbf {J} _ {L} \dot {\boldsymbol {\Omega}} _ {L} \end{array} \right] = \left[ \begin{array}{c} \mathbf {F} _ {H} \\ \mathbf {M} _ {H} \end{array} \right] + \mathbf {P} \left[ \begin{array}{c} \boldsymbol {\mu} _ {1} \\ \vdots \\ \boldsymbol {\mu} _ {n} \end{array} \right] - \left[ \begin{array}{c} m \mathbf {g} \\ \mathbf {0} _ {3 \times 1} \end{array} \right].\tag{38}
$$

Considering quasi-static operating conditions, we can assume the payload linear and angular acceleration terms can be neglected. Therefore, leveraging this assumption and rearranging based on (38) we obtain

$$
\left[ \begin{array}{c} \mathbf {F} _ {H} \\ \mathbf {M} _ {H} \end{array} \right] = - \mathbf {P} \left[ \begin{array}{c} \boldsymbol {\mu} _ {1} \\ \vdots \\ \boldsymbol {\mu} _ {n} \end{array} \right] + \left[ \begin{array}{c} m \mathbf {g} \\ \mathbf {0} _ {3 \times 1} \end{array} \right].\tag{39}
$$

Extracting tension values from each robot’s state allows us to compute the external wrenches on the payload.

## B. Payload Admittance Controller

The admittance controller is a high-level controller that updates $\mathbf { x } _ { L , \mathrm { d e s } }$ and $R _ { L , \mathrm { d e s } }$ in (12) for the payload controller (see Section V-A) based on the external force applied on the payload. When interacting with the payload, it allows the human operator to experience a virtual mass-spring-damper system rather than the actual mass. By setting the admittance controller’s tunable parameters to the desired values, the payload can be either sensitive or insensitive to external forces regardless of the payload’s actual property.

The admittance controller conceptualizes the payload as a virtual mass-spring-damper system, responding to external forces and moments from the human operator. It takes the external wrench, represented as $\begin{array} { r l } { \left\lceil \mathbf { F } _ { H } ^ { \top } \quad \mathbf { M } _ { H } ^ { \top } \right\rceil ^ { \top } \in \mathbb { R } ^ { 6 \times 1 } } \end{array}$ , as the input. The output from this controller includes the desired payload twist, denoted as $\dot { \mathcal { X } } _ { \mathrm { d e s } } \in \mathbb { R } ^ { 6 \times 1 }$ , along with the desired linear and angular positions ofthe payload, $\chi _ { \mathrm { d e s } } ^ { \mathrm { ~ ~ } } \in \mathbb { R } ^ { 6 \times 1 }$ , where the angular position is expressed in Euler angles. In addition, the admittance controller calculates the desired linear and angular accelerations, $\ddot { \mathcal { X } } _ { \mathrm { d e s } } \in \mathbb { R } ^ { 6 \times 1 }$ . However, it is noteworthy that these acceleration outputs are not utilized by the lower level payload controller.

The admittance controller assumes the following dynamics for the payload:

$$
\begin{array}{l} \mathbf {M} \ddot {\mathbf {e}} _ {\mathrm{adm}} + \mathbf {D} \dot {\mathbf {e}} _ {\mathrm{adm}} + \mathbf {K} \mathbf {e} _ {\mathrm{adm}} = \left[ \begin{array}{c} \mathbf {F} _ {H} \\ \mathbf {M} _ {H} \end{array} \right] \\ \ddot {\mathbf {e}} _ {\mathrm{adm}} = \ddot {\mathcal {X}} _ {\mathrm{des}} - \ddot {\mathcal {X}} _ {\mathrm{traj}}, \dot {\mathbf {e}} _ {\mathrm{adm}} = \dot {\mathcal {X}} _ {\mathrm{des}} - \dot {\mathcal {X}} _ {\mathrm{traj}} \\ \mathbf {e} _ {\mathrm{adm}} = \mathcal {X} _ {\mathrm{des}} - \mathcal {X} _ {\mathrm{traj}} \end{array}\tag{40}
$$

where M, D, and $\mathbf { K } \in \mathbb { R } ^ { 6 \times 6 }$ are tunable diagonal positive semidefinite matrices denoting the desired mass, damping, and spring property of the payload. Based on the initial starting condition of the payload, $\ddot { \chi } _ { \mathrm { t r a j } } , \dot { \chi } _ { \mathrm { t r a j } } .$ , and $\mathcal { X } _ { \mathrm { t r a j } }$ can be set accordingly. We choose to set them to be the planned trajectory. Closed-form solutions exist for (40) with the assumption that the input wrench is a predetermined function (e.g., a linear function). However, such an assumption is not ideal for our use case. Therefore, we choose to solve $\chi _ { \mathrm { d e s } } , \dot { \chi } _ { \mathrm { d e s } } , \ddot { \chi } _ { \mathrm { d e s } }$ with Runge–Kutta fourth-order approximation.

## VII. EXPERIMENTAL RESULTS

The experiments are conducted in an indoor testbed with a flying space of $\mathrm { 1 0 \times 6 \times 4 m ^ { 3 } }$ of the ARPL lab at New York University. We use three quadrotors to carry a triangular payload via suspended cables. The quadrotor platform used in the experiments is equipped with a QualcommSnapdragon 801 board for on-board computing [66]. A laptop equipped with an Intel i9-9900K CPU obtains the Vicon<sup>1</sup> motion capture system data via ethernet cable.

The framework has been developed in robot operating system (ROS) <sup>2</sup> and the robots’ clocks are synchronized by Chrony.<sup>3</sup> The mass of the payload is 310 g, which exceeds the payload capacity of every single vehicle. The pose and twist of the payload and quadrotors, the position and velocity of attachment points, and the human operator’s position are estimated using the Vicon data at a frequency of 100 Hz. The unit vector of each cable direction $\xi _ { k }$ and the corresponding velocity $\dot { \xi } _ { k }$ are estimated by

![](images/8ab48a50ab45aa143f92bac7dc8b6c9150b6d62717afc64e568da081e0dd3f05.jpg)  
Fig. 4. Wrench estimation evaluation. The human operator uses a force measurement device to measure the applied wrench on the payload, which is used to validate our wrench estimation algorithm results. On the left, we show the human operator applies force via the force measurement device, and, on the right, we show the force measurement device in detail.

$$
\boldsymbol {\xi} _ {k} = \frac {\mathbf {p} _ {\text { att } , k} - \mathbf {x} _ {k}}{\| \mathbf {p} _ {\text { att } , k} - \mathbf {x} _ {k} \|}, \dot {\boldsymbol {\xi}} _ {k} = \frac {\dot {\mathbf {p}} _ {\text { att } , k} - \dot {\mathbf {x}} _ {k}}{l _ {k}}\tag{41}
$$

where $\mathbf { p } _ { \mathrm { a t t } , k } , \dot { \mathbf { p } } _ { \mathrm { a t t } , k }$ are position and velocity of the kth attach point in and $\mathbf { x } _ { k } , \dot { \mathbf { x } } _ { k }$ are position and velocity of the kth robot in $\mathcal { T } ,$ all of which are estimated by the motion capture system.

## A. Cable Force and External Wrench Estimation

In this section, we validate our cable force and external wrench estimation algorithm by comparing the estimation results obtained using the approach presented in Section VI-A-2 with the ground truth from the wrench measurement device, as shown in Fig. 4.

We can identify the ground truth force by measuring the force direction and force magnitude separately via the wrench measurement device. As the ground truth force direction is along the cable between the measurement device and the other end where the device is attached to the system, it is measured by computing the difference between the load cell’s position and the attach point position using the Vicon motion capture system. The ground truth force magnitude is measured via a Phidget micro load cell,<sup>4</sup> as shown in Fig. 4. The measured cable direction and tension magnitude are postprocessed to obtain the ground truth force.

In the cable force estimation experiment, we hover a quadrotor in midair and run the proposed UKF onboard. The measurement device is connected to the center of mass of the quadrotor and a human operator pulls the measurement device into various directions to evaluate the algorithms. The results are shown in Fig. 5. In the plots, we compare the measured forces to the estimated forces in all 3 DoF and the estimated cable forces track measured ground truth accurately.

During the wrench estimation experiment, we hover the system with the regular payload controller without activating the admittance controller. Subsequently, the human operator pulls the payload with the force measurement device, and we record both the ground truth wrench and the estimation results. In addition to the ground truth force, the ground truth external torque is obtained by crossing the attached point position vector in $\mathcal { L }$ and the measured ground truth force vector from the measurement device. The payload is pulled so that the external wrench is nonzero in all 6 DoF, as shown in Fig. 4.

![](images/c74aa3582fa7f682b837634fb6e3763856f11b8b3c3719316487d0e83e285cbf.jpg)  
time (s)

![](images/80adeba4e7a628eaee91419d6db2ba75264852c8386c369b4aff617dcf5113d6.jpg)  
time (s)

![](images/ce81ccd0521e4968311775c74ad2cac35fada87dbb18e9b793b8c373747267c0.jpg)  
Fig. 5. Cable force estimation experiment results. Comparison between the cable force estimation and the force measurements from the force measurement device in all 3 DoF.

TABLE I  
RMSE OF WRENCH ESTIMATION AND MEASUREMENT

<table><tr><td rowspan="2"></td><td colspan="3">Force (N)</td><td colspan="3">Moment (N·m)</td></tr><tr><td>x</td><td>y</td><td>z</td><td>roll</td><td>pitch</td><td>yaw</td></tr><tr><td>Ours</td><td>0.0185</td><td>0.0117</td><td>0.0564</td><td>0.0088</td><td>0.0066</td><td>0.0045</td></tr><tr><td>[7]</td><td>0.0282</td><td>0.0164</td><td>0.0419</td><td>0.0148</td><td>0.0040</td><td>0.0120</td></tr></table>

The results are shown in Fig. 6. In the plots, we compare the measured wrenches to the estimated wrenches using our proposed method in Section VI-A-2 and the momentum observer method presented in [7] in all 6 DoF. As we can observe in Fig. 6, the estimated wrenches from our method track the measured ground truth quite accurately. However, on the other hand, the momentum observer method tends to smooth the estimates excessively, leading to underestimating the external wrench. The root mean square errors in all six directions are also reported in Table I, confirming a good accuracy.

## B. Admittance Control With Wrench Estimation

After validating the wrench estimation, we jointly test it with the admittance controller.

1) Virtual Impedance Realization: In this section, we present results in simulation to quantitatively analyze the performance of our proposed methods, particularly regarding the rendering of the desired virtual impedance of our proposed system. We deploy robot teams that consist of 3 “Dragonfly” quadrotors or 3 “Hummingbird” quadrotors with 1 m cable in our open-source simulator [57] to validate the realization of desired impedance values. In the following experiments, we introduce a step wrench input and observe the system’s response.

In the first experiment, we apply a step force input on the payload in the positive x direction and set different impedance values (1, 2, 5, 10) in the admittance controller. The second experiment involves the application of a step moment input on the payload in the positive yaw direction, with different impedance values (0.05, 0.25, 1.25, 2.5) set in the admittance controller. The results are shown in Figs. 7–9. We evaluate the actual impedance in the system using the ground truth force and moment, divided by the actual velocity of the payload along the corresponding direction. Hence, the actual impedance we obtain in Figs. 7–9 are $\frac { \mathbf { F } _ { H x } } { \dot { \mathbf { x } } _ { L x } }$ and $\frac { \mathbf { M } _ { H z } } { \dot { \Omega } _ { L z } }$ , respectively. In the plots, we compare the actual impedance with the desired impedance set in the admittance controller. As illustrated by the plots, upon application of step force and moment inputs to the system, the payload promptly accelerates in the x and yaw directions, respectively. As the wrench estimation updates, the admittance controller starts to adjust the desired payload state to adapt the human input, shown by the dashed lines in the bottom plots in both Figs. 7–9. Through the comparison, we can see that with larger values of $D _ { x } , D _ { \psi }$ , the desired velocity derived from the admittance controller evolves slower and smaller in magnitude. As the experiment proceeds, the desired velocity from the admittance controller ultimately converges to final values, respectively, equivalent to $\frac { \mathbf { F } _ { H x } } { D _ { x } }$ and $\frac { \mathbf { M } _ { H z } } { D _ { \psi } }$ . Moreover, a larger desired impedance value results in noticeable spikes in the actual impedance before the convergence. This can be attributed to the fact that a higher impedance value leads to a smaller corresponding desired velocity, which in turn causes larger velocity errors at the initial stage. This subsequently results in an overshooting response of the actual payload velocity, causing it to cross the zero line and trigger spikes in the actual impedance realization.

Lastly, as the experiments progress and the transient effects resulting from the step input reduce, the payload’s linear and angular velocities converge toward the desired payload velocity determined by the admittance controller. Consequently, the actual virtual impedance also aligns with the value set by the admittance controller. We observe similar results for the other Cartesian and angular axes that are not reported for simplicity.

2) Real-World Experiments: We conduct six tests involving all 6 DoF of the admittance controller in real-world experiments. The human operator manipulates the payload by translating the payload in x, y, and z and rotating the payload in roll, pitch, and yaw, respectively, to show that the load can be fully manipulated. At the end of each experiment, the human operator releases the payload. The square gain matrices in the admittance controller have a blockdiagonal structure as $\mathbf { M } = \mathrm { d i a g } ( 0 . 2 5 \mathbf { I } _ { 3 \times 3 } , 0 . 1 \mathbf { I } _ { 3 \times 3 } )$ , D <sub>=</sub> diag $1 . 2 5 \mathbf { I } _ { 3 \times 3 } , 5 . 0 \mathbf { I } _ { 3 \times 3 } )$ $\mathbf { K } = \mathrm { d i a g } ( \mathbf { 0 } _ { 3 \times 3 } , \mathbf { 0 } _ { 3 \times 3 } )$

The experimental results are presented in Fig. 10. As shown in the plots, the human operator translates the payload approximately 1m in x and y direction, 0.4 m in the z direction. In the rotation part of the experiment, the human operator rotates the payload approximately $3 0 ^ { \circ }$ in the roll and pitch direction and 60◦ in the yaw direction. The tests show that the admittance controller, coupled with the wrench estimator, can successfully update the payload’s desired position or orientation according to the human operator’s interactive force as input. As the human operator releases the payload, the wrench estimation outputs $\left[ 0 _ { 6 \times 1 } \right]$ as wrench estimation. Since K in the admittance controller is $\left\lceil \mathbf { 0 } _ { 6 \times 6 } \right\rceil$ in this set of experiments, the payload remains at the position or orientation released by the human operator without returning to its original reference position or orientation. It further confirms the effectiveness ofthe wrench estimation and admittance controller pipeline in assisting object transportation and manipulation.

![](images/4d2e9143b090a3ab4b4195a08826a9f52e718263caed89f4c125a353cfb0f5df.jpg)  
Fig. 6. Results of the wrench estimation experiment. This figure compares the wrench estimation results from our proposed wrench estimation algorithm (blue) with those obtained using the momentum observer method (red) [7], as well as with the actual measurements recorded by the wrench measurement device (green) across all 6 DoF.

![](images/8ec2c5fb6c4e9e90f7c5bdc0eb96c8b8404439a0b557e0f31fb7d7f3e04327c3.jpg)  
Fig. 7. Virtual impedance realization in the x direction of the translational motion. A step force input of $\mathbf { F } _ { H } = [ 0 . 5 0 \ \mathrm { ~ 0 } ] ^ { \top } \mathrm { N }$ is given into the system at the start of the plots. We choose the parameters of the admittance controller in the x direction as M = 0.25, K = 0.0, and $\overline { { D _ { x } } } = 1 , 2 , 5$ , 10 for comparison.

## C. Human-Aware Human–Robot Collaborative Transportation

In this section, we show that our system enables a human operator to physically collaborate with the robot team to accomplish the following two tasks.

1) The robot team and the human collaboratively manipulate the payload to a goal location, as shown in Fig. 11.

2) Human operator corrects the payload trajectory to avoid an obstacle in an existing trajectory, as shown in Fig. 12

The gradient-based and optimization-based methods for human-aware force distribution are tested for each of the two tasks. For the optimization-based method, the drone-to-drone distance limit is set to be $\ge 0 . 7 5 \mathrm { m }$ , and the human-to-drone distance limit is set to be $\ge 0 . 7 5 \mathrm { m }$ . The gradient-based method does not require a predetermined distance.

In addition, as we show in Fig. 3 and discuss in Section V-B, we feed the human operator’s position $\mathbf { p } _ { H }$ from the Vicon in as p<sub>O</sub> to (20) and (25) for application to pHRI. We would also like to note that by using deep-learning-based human pose estimation techniques [67], the robots can also use onboard camera to estimate $\mathbf { p } _ { H }$ , but this is out of the scope of this article and we refer to it as future work.

![](images/d48604da12adcd863b7eea4dcc671d71a2a8bfa07ff7c1673aac2fc1c6f8d475.jpg)  
Fig. 8. Virtual impedance realization in the x direction of the translational motion. A step force input of ${ \bf F } _ { H } = [ 1 $ .0 0 0]N is given into the system at the start of the plots. We choose the parameters of the admittance controller in the x direction as M = 0.25, $K _ { x } = 0 . { \overset { \cdot } { 0 } } ,$ and $D _ { x } = 1 ,$ , 2, 5, 10 for comparison.

![](images/265d7f59559d3ae78bbd730da8d020b8cfaf4870037d110dccc4678d542f526c.jpg)

![](images/d47eeb21b1692c02342c8772bf95a676c3c44cdb07f49d7fb94c914799999154.jpg)  
Fig. 9. Virtual impedance realization in the yaw direction of the rotational motion. A step moment input of M $\dot { \cdot } { \cal H } = [ 0$ 0 $0 . 0 5 7 ^ { \top } N$ m is given into the system at the start of the plots. We choose the parameters of the admittance controller in the yaw direction as $M _ { \psi } = 0 . 1$ $\dot { K _ { \psi } } = 0 . 0$ , and $D _ { \psi } = 0 . 0 5$ , 0.25, 1.25, 2.5 for comparison.

1) Human–Robot Collaborative Transportation: In this experiment, the robot team and the human operator collaborate together to move a payload from the starting location to the final location via direct force interaction. Payload translates in all three axes, as shown in Fig. 11. The square gain matrices are selected block-diagonal as $\mathbf { M } = \mathrm { d i a g } ( 0 . 2 5 \mathbf { I } _ { 3 \times 3 } , 0 . 1 \mathbf { I } _ { 3 \times 3 } )$ , D <sub>=</sub> diag $\left( 5 . 0 \mathbf { I } _ { 3 \times 3 } , 5 . 0 \mathbf { I } _ { 3 \times 3 } \right)$ 7 $\mathbf { K } = \mathrm { d i a g } ( \mathbf { 0 } _ { 3 \times 3 } , \mathbf { 0 } _ { 3 \times 3 } )$ . Note that the spring constant coefficient for the admittance controller is set to zero so that the payload stays at the position/orientation once the human operator releases the payload.

The experimental results are shown in Fig. 13, where we compare the actual payload position with the desired payload position from the admittance controller. The results show that the proposed methods can confidently update the desired payload position to satisfy the human operator’s intention of moving the payload under both gradient-based and optimization-based methods. Furthermore, the movement introduced by the human-aware force distribution does not affect the performance of the payload wrench estimation or admittance controller.

![](images/9a29a76f42f9884a2d38cad165903a34f4b0d5d76f24fe7bdcec6b998d1e17e3.jpg)  
Fig. 10. We test the admittance controller together with wrench estimation in all 6 DoF. The $\mathcal { X } _ { \mathrm { t r a j } }$ are where the plots start and the derivative $\dot { \mathcal { X } } _ { \mathrm { t r a j } } \ddot { \mathcal { X } } _ { \mathrm { t r a j } }$ are zero. The actual payload pose, desired payload pose from the admittance controller, and estimated wrench are plotted. On the top: Translation tests in x, y, z in . On the bottom: Rotation tests in roll, pitch, yaw in .  
TABLE II

![](images/4c4a48a49a297a80df4378af6d89fed296f5a8f2e6cd48873ce5ff0fa6e2c0ae.jpg)

Fig. 11. Human–robot collaborative transportation task. The human operator collaborates with a team of quadrotors to transport a payload from the start position ( ) to the final position (-). The human operator and the quadrotor team translate the payload along the x, y, and z axes.  
![](images/0595930b36fc242b40aef47aaee0cb524a98930aa72b8178af2d1b9150d30fb1.jpg)  
Fig. 12. Human-assisted obstacle avoidance task. The human operator moves the payload from the desired trajectory (yellow path) to guide the payload away from an unknown obstacle and then releases the payload, allowing it to rejoin the desired trajectory and reach the final position (-).

DISTANCE BETWEEN ROBOT–ROBOT AND ROBOT-HEAD PAIRS

<table><tr><td>Method</td><td>Pair Type</td><td>Mean (m)</td><td>Std (m)</td></tr><tr><td rowspan="6">Gradient-based</td><td>1-2</td><td>0.580</td><td>0.0964</td></tr><tr><td>1-3</td><td>0.796</td><td>0.0845</td></tr><tr><td>2-3</td><td>0.547</td><td>0.0764</td></tr><tr><td>Head-1</td><td>0.875</td><td>0.0977</td></tr><tr><td>Head-2</td><td>1.043</td><td>0.0411</td></tr><tr><td>Head-3</td><td>0.791</td><td>0.0718</td></tr><tr><td rowspan="6">Optimization-based</td><td>1-2</td><td>0.814</td><td>0.0532</td></tr><tr><td>1-3</td><td>1.251</td><td>0.108</td></tr><tr><td>2-3</td><td>0.748</td><td>0.0501</td></tr><tr><td>Head-1</td><td>0.892</td><td>0.0577</td></tr><tr><td>Head-2</td><td>0.961</td><td>0.0804</td></tr><tr><td>Head-3</td><td>0.805</td><td>0.0387</td></tr></table>

In Fig. 14, we show the effects of the two methods for human-aware force distribution with a top view of the entire collaboration task. From the plots, we can observe that, as the human operator, denoted by the purple star, approaches the 3 robots with a suspended payload, the human-aware force distribution starts to be effective. The controller expands the 2 robots (blue and green circles) that are close to the human operator to keep the distance.

To quantitatively support our analysis, we conducted multiple iterations of the same experiments and recorded the distances between the robots and the human operator. The distribution of these distance measurements is illustrated in Figs. 15 and 16, with corresponding mean and standard deviation values summarized in Table II .

As shown in Figs. 15 and 16, both the optimization-based and gradient-based methods effectively maintain a consistent distance between the robots and the human operator throughout multiple repeated experiments. This observation is further validated by the statistical data presented in Table II. Notably, the optimization-based method distinguishes itself from the gradient-based method as it enforces inter-robot distance constraints. This is evident in the left plot of Fig. 16, where all three drones maintain a minimum separation of 0.75 m, as specified by the constraint.

![](images/974107dd1802d771c1572059b8c768f039f0c8e736acd814d651407ee6728092.jpg)

![](images/0c22e9369cbde3a2dadcef4d52bde691637ad91381a9c5fa529957273b951389.jpg)

![](images/bf4a424aac0cadf5e61e96881b94dc879b8832155959c1c4deeb3d45c2328add.jpg)

![](images/56a56741fd0b0249d37db935e10c5bed973b0e1e354123365c4aa5133d3dad40.jpg)

![](images/57b898c1048d28055462b8b7271772c3544da5e40a7cf89ed4080a482dbf7f1c.jpg)

![](images/d3175ae5d17a42f4e9dc2cec88189c2bad64af8fb6a830f585910f9d9e3296c3.jpg)

![](images/d3f2308594641a0e4233c175f629799805b2b569deaf9dfa2ea3208057b6eeff.jpg)

![](images/8311c3c379759372fa583ab9baad2acbeae7a48906a5837217e902aeec232e24.jpg)  
Fig. 13. Human–robot collaborative transportation experiment results. Payload position versus desired position from admittance controller. Transnational results when optimization-based human-aware method is used (top row). Transnational results when gradient-based human-aware method is used (bottom row).

![](images/d5c976c7d4e9fce1bc139f137cd79bd73f26840ea550373541408902282f3d2d.jpg)

![](images/65f0eb602f931195148409ed5bfc68e43541676ec58d814aba25c680c0be9071.jpg)

![](images/065432001f01625af36ab0067455807241eaae45ffb286f1a41d4759b4ac26fc.jpg)

![](images/fdabe477dc0f931df7006b8433ece009994f298320f30915c66df8fcf6726fe3.jpg)

![](images/74abd551dd9b6b304f5638507cb03964f1f0381075aa27e7e7d4907437c3a320.jpg)

![](images/bc91cd901c5606fce6dfcf20722fa6a21496dcbfc1973785bcb07c77a28021e3.jpg)  
Fig. 14. Human–robot collaborative transportation experiment results. We show the top down view of the human operator (star) and the team of drones (circles). On the top: Trajectory when optimization-based human-aware method is used. On the bottom: Trajectory when gradient-based human-aware method is used.

![](images/b82f51ff576a0e83bb44c3b8a3e5673e42c674b682545efccffc0e594b84c286.jpg)

![](images/090fc3387b55ab204c56288fca50aa1327418c21a22fc14f70278fca1f72f2e2.jpg)  
Fig. 15. Box plot with mean lines of the robot to robot distances and robot to head distances during the human-aware collaborative transportation experiments using the gradient-based method. The data are collected from multiple repeated experiments.

![](images/45269a2e3d265adf9e2d711acf652d89388ba07764923722fa509d1e26e8459c.jpg)

![](images/32fd48d26632640b9d32683b47a9ec117bd7998c9ebc30a9cc8a288615c8feec.jpg)  
Fig. 16. Box plot with mean lines of the robot to robot distances and robot to head distances during the human-aware collaborative transportation experiments using the optimization-based method. The data are collected from multiple repeated experiments.

![](images/5d99033d3f6d6be851182f223fbb0571802a9ee8139bef7906d38d4629466ab2.jpg)

![](images/357a11656d122bb80ea0e6ebc1f191b81c6bdf77376c7534f340087fb6634b6f.jpg)  
Fig. 17. Human–robot collaborative transportation experiment results. Drone to drone distance and human to drone distance when gradient-based method is used; The gradient-based method does not require a predetermined distance.

![](images/5d4b050e5ed17e1b575d2827fde3f7cc26ba4eba9ae2752c783817a8440e6d5a.jpg)  
Fig. 18. Human–robot collaborative transportation experiment results. Droneto-drone distance and human-to-drone distance when optimization-based method is used; Minimum drone-to-drone distance is set to be 0.75 m, and the minimum human-to-drone distance is set to be 0.75 m.

To provide additional insights, Figs. 17 and 18 depict the distances between each robot and the human operator, as well as the inter-robot distances, throughout the duration of a single sample experiment. Initially, the human operator starts approximately 2 and 3 m away from the robot team. As the operator approaches, the distances between the human and robots 1 and 3 (represented by blue and green) decrease. At this point, the human-aware force distribution becomes active, maintaining stable human– robot distances.

2) Human-Assisted Obstacle Avoidance: In this experiment, the payload follows a straight trajectory from the starting location to the final location, as the robot team is unaware of the obstacle. The human operator corrects the payload trajectory to avoid the obstacle, as shown in Fig. 12. Both gradient-based and optimization-based methods are also applied here. The square gain matrices for the admittance controller have a blockdiagonal structure as $\mathbf { M } = \mathrm { d i a g } ( 0 . 2 5 \mathbf { I } _ { 3 \times 3 } , 0 . 1 \mathbf { I } _ { 3 \times 3 } )$ , D <sub>=</sub> diag $\left( 5 . 0 \mathbf { I } _ { 3 \times 3 } , 5 . 0 \mathbf { I } _ { 3 \times 3 } \right)$ , K <sub>=</sub> diag $( 1 . 2 \mathbf { I } _ { 3 \times 3 } , \mathbf { 0 } _ { 3 \times 3 } )$

![](images/3bd36498128bfef198716fa0b6b68ff73388b24d587022fae0583e0fe228fa56.jpg)

![](images/205bbfe6452c48e26fbeccb38c5724dd512ded313f3e1df0871578863bdab8f9.jpg)  
Fig. 19. Correcting payload trajectory experiment result. Comparison between actual trajectory, desired trajectory, admittance controller output, and estimated external wrench on the payload. Optimization-based safety controller is used (left). Gradient-based safety controller is used (right).

TABLE III  
COMPUTATIONAL COMPLEXITY SUMMARY

<table><tr><td></td><td>Scalability</td><td>Time complexity</td></tr><tr><td colspan="3">pHRI</td></tr><tr><td>Robot state estimation</td><td>High</td><td> $\mathcal{O}(1)$ </td></tr><tr><td>Human wrench estimation</td><td>Medium</td><td> $\mathcal{O}(n)$ </td></tr><tr><td>Payload admittance controller</td><td>High</td><td> $O(1)$ </td></tr><tr><td colspan="3">Planning and control</td></tr><tr><td>Payload trajectory planner</td><td>High</td><td> $\mathcal{O}(1)$ </td></tr><tr><td>Payload trajectory tracking controller</td><td>High</td><td> $\mathcal{O}(1)$ </td></tr><tr><td>Nominal force distribution</td><td>Medium</td><td> $\mathcal{O}(n)$ </td></tr><tr><td>Force distribution: Gradient- based</td><td>High</td><td> $\mathcal{O}(n)$ </td></tr><tr><td>Force distribution: Optimization- based</td><td>Low</td><td> $\mathcal{O}(n^3)$ </td></tr><tr><td>Robot controller</td><td>High</td><td> $\mathcal{O}(1)$ </td></tr></table>

Note that the constant spring coefficient for the admittance controller is no longer zero. The payload will now return to the position/orientation commanded by the trajectory when the human operator releases the payload.

As we can see from Fig. 19, the correction takes effects according to the admittance controlled trajectory. Once the human operator stops the correction, the nonzero K constant starts to allow the corrected trajectory to converge with the original trajectory. As expected, such behavior is present in both the gradient-based and optimization-based methods.

## VIII. COMPUTATIONAL COMPLEXITY DISCUSSION

In this section, we discuss the theoretical computational time complexity of the methods proposed in this article, focusing on how each algorithm’s computational complexity scales with the number of robots. Through this discussion, we aim to offer insights into the proposed methods and guide the corresponding system design choices. We summarize the results regarding the computational complexity of our algorithms in Table III.

## A. Physical Human–Robot Interaction

We begin our discussion with the pHRI module, comprising the robot state estimator, the human wrench estimator, and the payload admittance controller.

Each quadrotor runs a UKF state estimator with a fixed state vector size of 19 and a fixed input vector size of 4. Similarly, the payload admittance controller’s computation, detailed in (40), is independent of the number of quadrotors, as it controls only the 6 DoF of the payload. Consequently, both the UKF and the payload admittance controller exhibit a computational complexity of (1), independent of the number of quadrotors n in the system.

Next, the human wrench estimation requires quadrotors in the team to share individual estimated cable tension forces, which are then aggregated to derive the total external wrench on the payload using (39). The computation, shown in (39), scales linearly with the number of quadrotors due to the linear tension mapping matrix P. Thus, the computation complexity in (39) is bounded by ${ \mathcal { O } } ( n )$ , with the dimension of matrix P, as depicted in (4), scaling accordingly with n. This linear time complexity is manageable with the available computational resources, as n would need to reach the order of thousands to make this linear-complexity matrix multiplication the system’s bottleneck.

## B. Planning and Control

In planning and control, the system includes payload trajectory planner, payload trajectory tracking controller, dynamic force distribution, and robot controller. The payload trajectory planner and tracking controller function similarly to the payload admittance controller, meaning their computation also remains independent of the number of quadrotors n. In addition, each quadrotor independently runs its robot controller as specified in (29) and (30), thus, these components also maintain a computational complexity of $\mathcal { O } ( 1 )$ .

The dynamic force distribution involves two parts: nominal force distribution and human-aware force distribution. It requires linear mapping via matrix multiplication, as shown in (13), with a complexity bound of ${ \mathcal { O } } ( n )$

Regarding human-aware force distribution, we propose two methods: an optimization-based method and a gradient-based method. We discuss them separately as follows.

i) The optimization-based method employs the sequential least-squares quadratic programming solver, requiring $\mathcal { O } ( a ^ { 2 } )$ storage and $\mathcal { O } ( a ^ { 3 } )$ time, where $a = 3 \times ( n - 6 )$ represents the problem dimension.

ii) The gradient-based method computes the cable tension force modifier through null space projection of a scaled gradient vector, optimizing (19). The sum of $\mathcal { L } _ { 2 }$ -norm distances between each quadrotor and the human operator, combined with the closed-form solution for null space projection, leads to a computational complexity of ${ \mathcal { O } } ( n )$ This complexity is less than that of the optimizationbased method, attributed to the use of closed-form solutions for both gradient computation and null space projection.

## IX. CONCLUSION

In this article, we presented a human-aware human–robot collaborative transportation and manipulation approach considering a team ofaerial robots with a cable-suspended payload. Our approach combines a novel control method that leverages system redundancy with a collaborative wrench estimator, enabling a human operator to interact in 6 DoF with a rigid structure being transported by a team of aerial robots via cable. In addition, the system can achieve secondary tasks like keeping a certain distance between the human operator and robots, or inter-robot separation by exploiting the additional system redundancy without affecting the quality or accuracy of the interactive experience. We demonstrated, through real-world experiments, our system’s capabilities. The system can assist the human operator in manipulation tasks, as well as enable the human operator to effectively assist the load navigation, as demonstrated in the experiments.

In future research, we aim to expand our study into humancentric considerations, prioritizing metrics related to comfortness and acceptance of human operators. These elements are crucial in the domain of human–robot interaction. In addition, we plan to develop safety methods to counteract unexpected human actions, such as sudden or forceful human physical inputs to the load that could lead to cable slack or actuator overload. This could ensure further robust operation under varied conditions.

In addition, we want to extend our framework to explicitly address collision avoidance between the cables and the human operator as well. Our current approach relies on the human’s ability to navigate around the cables. However, by modeling the cables as convex polygons and incorporating them into the optimization process, we can develop a more comprehensive collision avoidance strategy that ensures human safety. Further developments also include the design of an onboard sensing mechanism. We plan to employ tension-measurement tools, onboard cameras, inertia measurement units (IMUs), and electronic speed control (ESC)s on each vehicle. Our goal is to achieve comprehensive onboard state estimation, therefore eliminating dependence on external motion capture systems. We also intend to investigate the impacts of state estimation and control delays, lags, and noise on system performance. Understanding these factors will enable us to improve our the robustness of our framework, enhancing interaction experience.

Finally, we would like to integrate a more advanced onboard perception module. It can empower the robot team to identify and navigate around complex hazardous spaces. This feature will enable autonomous obstacle avoidance maneuvers while leveraging the system’s redundancy to maintain the intended payload trajectory without compromise. We also envision employing deep learning techniques with robots’ onboard cameras to analyze the human operator’s posture, enhancing our humanaware force distribution strategy.

## ACKNOWLEDGMENT

The authors would like to thank Jimmy Lee, Manling Li, and Luca Morando for their help and support during the experiments.

## REFERENCES

[1] K. Schwab, The Fourth Industrial Revolution. London, U.K.: Portfolio Penguin, 2017.

[2] S. S. Mansouri, C. Kanellakis, E. Fresk, D. Kominiak, and G. Nikolakopoulos, “Cooperative coverage path planning for visual inspection,” Control Eng. Pract., vol. 74, pp. 118–131, 2018.

[3] K. Shah, G. Ballard, A. Schmidt, and M. Schwager, “Multidrone aerial surveys of penguin colonies in antarctica,” Sci. Robot., vol. 5, no. 47, 2020, Art. no. eabc3000.

[4] G. Loianno et al., “Autonomous flight and cooperative control for reconstruction using aerial robots powered by smartphones,” Int. J. Robot. Res., vol. 37, no. 11, pp. 1341–1358, 2018.

[5] N. Michael et al., “Collaborative mapping of an earthquake-damaged building via ground and aerial robots,” J. Field Robot., vol. 29, no. 5, pp. 832–841, 2012.

[6] A. E. Jiménez-Cano, D. Sanalitro, M. Tognon, A. Franchi, and J. Cortés, “Precise cable-suspended pick-and-place with an aerial multi-robot system,” J. Intell. Robotic Syst., vol. 105, no. 3, 2022, Art. no. 68.

[7] D. Sanalitro, M. Tognon, A. J. Cano, J. Cortés, and A. Franchi, “Indirect force control of a cable-suspended aerial multi-robot manipulator,” IEEE Robot. Autom. Lett., vol. 7, no. 3, pp. 6726–6733, Jul. 2022.

[8] M. Saska, V. Vonásek, J. Chudoba, J. Thomas, G. Loianno, and V. Kumar, “Swarm distribution and deployment for cooperative surveillance by micro-aerial vehicles,” J. Intell. Robot. Syst., vol. 84, no. 1, pp. 469–492, 2016.

[9] G. Li and G. Loianno, “Design and experimental evaluation of distributed cooperative transportation of cable suspended payloads with micro aeria vehicles,” in Experimental Robotics, B. Siciliano, C. Laschi, and O. Khatib, Eds. Cham, Switzerland: Springer, 2021, pp. 28–36.

[10] B. E. Jackson, T. A. Howell, K. Shah, M. Schwager, and Z. Manchester, “Scalable cooperative transport of cable-suspended loads with UAVs using distributed trajectory optimization,” IEEE Robot.Autom. Lett., vol. 5, no. 2, pp. 3368–3374, Apr. 2020.

[11] G. Li, R. Ge, and G. Loianno, “Cooperative transportation of cable suspended payloads with MAVs using monocular vision and inertial sensing,” IEEE Robot. Autom. Lett., vol. 6, no. 3, pp. 5316–5323, Jul. 2021.

[12] A. Ollero, M. Tognon, A. Suarez, D. Lee, and A. Franchi, “Past, present, and future of aerial robotic manipulators,” IEEE Trans. Robot., vol. 38, no. 1, pp. 626–645, Feb. 2022.

[13] A. Afifi, G. Corsini, Q. Sable, Y. Aboudorra, D. Sidobre, and A. Franchi, “Physical human-aerial robot interaction and collaboration: Exploratory results and lessons learned,” in Proc. Int. Conf. Unmanned Aircr. Syst., 2023, pp. 956–962.

[14] A. Tagliabue, M. Kamel, R. Siegwart, and J. Nieto, “Robust collaborative object transportation using multiple MAVs,” Int. J. Robot. Res., vol. 38, no. 9, pp. 1020–1044, 2019.

[15] G. Loianno and V. Kumar, “Cooperative transportation using small quadrotors using monocular vision and inertial sensing,” IEEE Robot. Autom. Lett., vol. 3, no. 2, pp. 680–687, Apr. 2018.

[16] D. Mellinger, M. Shomin, N. Michael, and V. Kumar, Cooperative Grasping and Transport Using Multiple Quadrotors. Berlin, Germany: Springer, 2013, pp. 545–558.

[17] J. Thomas, G. Loianno, J. Polin, K. Sreenath, and V. Kumar, “Toward autonomous avian-inspired grasping for micro aerial vehicles,” Bioinspiration Biomimetics, vol. 9, no. 2, 2014, Art. no. 025010.

[18] A. Afifi, M. V. Holland, and A. Franchi, “Toward physical human-robot interaction control with aerial manipulators: Compliance, redundancy resolution, and input limits,” in Proc. IEEE Int. Conf. Robot. Autom., 2022, pp. 4855–4861.

[19] G. Corsini, M. Jacquet, H. Das, A. Afifi, D. Sidobre, and A. Franchi, “Nonlinear model predictive control for human-robot handover with application to the aerial case,” in Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst., 2022, pp. 7597–7604.

[20] M. Tognon, R. Alami, and B. Siciliano, “Physical human-robot interaction with a tethered aerial vehicle: Application to a force-based human guiding problem,” IEEE Trans. Robot., vol. 37, no. 3, pp. 723–734, Jun. 2021.

[21] J. Lee et al., “Visual-inertial telepresence for aerial manipulation,” in Proc. IEEE Int. Conf. Robot. Autom., 2020, pp. 1222–1229.

[22] J. Geng and J. Langelaan, “Cooperative transport of a slung load using load-leading control,” J. Guid., Control, Dyn., vol. 43, no. 7, pp. 1313–1331, 2020.

[23] H. Rastgoftar and E. M. Atkins, “Cooperative aerial payload transport guided by an in situ human supervisor,” IEEE Trans. Control Syst. Technol., vol. 27, no. 4, pp. 1452–1467, Jul. 2019.

[24] M. Romano, A. Ye, J. Pye, and E. Atkins, “Cooperative multilift slung load transportation using haptic admittance control guidance,” J. Guid., Control, Dyn., vol. 45, no. 10, pp. 1899–1912, 2022.

[25] K. Klausen, C. Meissen, T. I. Fossen, M. Arcak, and T. A. Johansen, “Cooperative control for multirotors transporting an unknown suspended load under environmental disturbances,” IEEE Trans. Control Syst. Technol., vol. 28, no. 2, pp. 653–660, Mar. 2020.

[26] M. Bernard, K. Kondak, I. Maza, and A. Ollero, “Autonomous transportation and deployment with aerial robots for search and rescue missions,” J. Field Robot., vol. 28, no. 6, pp. 914–931, 2011.

[27] V. P. Tran, F. Santoso, M. A. Garratt, and S. G. Anavatti, “Distributed artificial neural networks-based adaptive strictly negative imaginary formation controllers for unmanned aerial vehicles in time-varying environments,” IEEE Trans. Ind. Informat., vol. 17, no. 6, pp. 3910–3919, Jun. 2021.

[28] X. Zhang et al., “Self-triggered based coordinate control with low communication for tethered multi-UAV collaborative transportation,” IEEE Robot. Autom. Lett., vol. 6, no. 2, pp. 1559–1566, Apr. 2021.

[29] Y. Liu, F. Zhang, P. Huang, and X. Zhang, “Analysis, planning and control for cooperative transportation of tethered multi-rotor UAVs,” Aerosp. Sci. Technol., vol. 113, 2021, Art. no. 106673.

[30] T. Lee, K. Sreenath, and V. Kumar, “Geometric control of cooperating multiple quadrotor UAVs with a suspended payload,” in Proc. 52nd IEEE Conf. Decis. Control, 2013, pp. 5510–5515.

[31] A. Tagliabue, M. Kamel, S. Verling, R. Siegwart, and J. Nieto, “Collaborative transportation using MAVs via passive force control,” in Proc. IEEE Int. Conf. Robot. Autom., 2017, pp. 5766–5773.

[32] M. Gassner, T. Cieslewski, and D. Scaramuzza, “Dynamic collaboration without communication: Vision-based cable-suspended load transport with two quadrotors,” in Proc. IEEE Int. Conf. Robot. Autom., 2017, pp. 5196–5202.

[33] H. Rastgoftar and E. M. Atkins, “Continuum deformation of a multiple quadcopter payload delivery team without inter-agent communication,” in Proc. Int. Conf. Unmanned Aircr. Syst., 2018, pp. 539–548.

[34] H. Xie, K. Dong, and P. Chirarattananon, “Cooperative transport of a suspended payload via two aerial robots with inertial sensing,” IEEE Access, vol. 10, pp. 81764–81776, 2022.

[35] M. Tognon, C. Gabellieri, L. Pallottino, and A. Franchi, “Aerial comanipulation with cables: The role of internal force for equilibria, stability, and passivity,” IEEE Robot. Autom. Lett., vol. 3, no. 3, pp. 2577–2583, Jul. 2018.

[36] T. Lee, “Geometric control of quadrotor UAVs transporting a cablesuspended rigid body,” IEEE Trans. Control Syst. Technol., vol. 26, no. 1, pp. 255–264, Jan. 2018.

[37] G. Wu and K. Sreenath, “Geometric control of multiple quadrotors transporting a rigid-body load,” in Proc. 53rd IEEE Conf. Decis. Control, 2014, pp. 6141–6148.

[38] J. Fink, N. Michael, S. Kim, and V. Kumar, “Planning and control for cooperative manipulation and transportation with aerial robots,” Int. J. Robot. Res., vol. 30, no. 3, pp. 324–334, 2011.

[39] N. Michael, J. Fink, and V. Kumar, “Cooperative manipulation and transportation with aerial robots,” Auton. Robots, vol. 30, no. 1, pp. 73–86, 2011.

[40] R. C. Sundin, P. Roque, and D. V. Dimarogonas, “Decentralized model predictive control for equilibrium-based collaborative UAV bar transportation,” in Proc. IEEE Int. Conf. Robot. Autom., 2022, pp. 4915–4921.

[41] G. Tartaglione, E. D’Amato, M. Ariola, P. S. Rossi, and T. A. Johansen, “Model predictive control for a multi-body slung-load system,” Robot. Auton. Syst., vol. 92, pp. 1–11, 2017.

[42] T. Lee, “Geometric control of multiple quadrotor UAVs transporting a cable-suspended rigid body,” in Proc. 53rd IEEE Conf. Decis. Control, 2014, pp. 6155–6160.

[43] J. Geng, P. Singla, and J. W. Langelaan, “Load-distribution-based trajectory planning and control for a multilift system,” J. Aerosp. Inf. Syst., vol. 19, no. 5, pp. 366–381, 2022.

[44] C. Masone, H. H. Bülthoff, and P. Stegagno, “Cooperative transportation of a payload using quadrotors: A reconfigurable cable-driven parallel robot,” in Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst., 2016, pp. 1623–1630.

[45] E. Bulka, C. He, J. Wehbeh, and I. Sharf, “Experiments on collaborative transport of cable-suspended payload with quadrotor UAVs,” in Proc. Int. Conf. Unmanned Aircr. Syst., 2022, pp. 1465–1473.

[46] L. v. d. Spaa, M. Gienger, T. Bates, and J. Kober, “Predicting and optimizing ergonomics in physical human-robot cooperation tasks,” in Proc. IEEE Int. Conf. Robot. Autom., 2020, pp. 1799–1805.

[47] J. Stuckler and S. Behnke, “Following human guidance to cooperatively carry a large object,” in Proc. 11th IEEE-RAS Int. Conf. Humanoid Robots, Bled, Slovenia, 2011, pp. 218–223.

[48] M. Gienger et al., “Human-robot cooperative object manipulation with contact changes,” in Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst., 2018, pp. 1354–1360.

[49] W. Sheng, A. Thobbi, and Y. Gu, “An integrated framework for human– robot collaborative manipulation,” IEEE Trans. Cybern., vol. 45, no. 10, pp. 2030–2041, Oct. 2015.

[50] F. Augugliaro and R. D’Andrea, “Admittance control for physical humanquadrocopter interaction,” in Proc. Eur. Control Conf., 2013, pp. 1805– 1810.

[51] D. Sieber, S. Musi´c, and S. Hirche, “Multi-robot manipulation controlled by a human with haptic feedback,” in Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst., 2015, pp. 2440–2446.

[52] G. A. Yashin, D. Trinitatova, R. T. Agishev, R. Ibrahimov, and D. Tsetserukou, “AeroVr: Virtual reality-based teleoperation with tactile feedback for aerial manipulation,” in Proc. Int. Conf. Adv. Robot., 2019, pp. 767–772.

[53] S. O. Sachidanandam, S. Honarvar, and Y. Diaz-Mercado, “Effectiveness of augmented reality for human swarm interactions,” in Proc. IEEE Int. Conf. Robot. Autom., 2022, pp. 11258–11264.

[54] M. L. Elwin, B. Strong, R. A. Freeman, and K. M. Lynch, “Humanmultirobot collaborative mobile manipulation: The omnid mocobots,” IEEE Robot. Autom. Lett., vol. 8, no. 1, pp. 376–383, Jan. 2023.

[55] D. Sirintuna, I. Ozdamar, and A. Ajoudani, “Carrying the uncarriable: A deformation-agnostic and human-cooperative framework for unwieldy objects using multiple robots,” in Proc. IEEE Int. Conf. Robot. Autom., 2023, pp. 7497–7503.

[56] N. E. Carey and J. Werfel, “A force-mediated controller for cooperative object manipulation with independent autonomous robots,” in Proc. Int. Symp. Distrib. Auton. Robotic Syst., 2022, pp. 140–155.

[57] G. Li, X. Liu, and G. Loianno, “RotorTM: A flexible simulator for aerial transportation and manipulation,” IEEE Trans. Robot., vol. 40, pp. 831–850, 2024.

[58] G. Li and G. Loianno, “Nonlinear model predictive control for cooperative transportation and manipulation of cable suspended payloads with multiple quadrotors,” in Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst., 2023, pp. 5034–5041.

[59] B. Siciliano, L. Sciavicco, L. Villani, and G. Oriolo, Robotics Modelling, Planning and Control. London, U.K.: Springer, 2009.

[60] D. Bertsekas, Nonlinear Programming. Nashua, NH, USA: Athena Sci., 1999.

[61] D. Kraft, “Algorithm 733: Tomp–fortran modules for optimal control calculations,” ACM Trans. Math. Softw., vol. 20, no. 3, pp. 262–281, 1994.

[62] S. G. Johnson, “The NLopt nonlinear-optimization package,” 2011. [Online]. Available: http://ab-initio.mit.edu/nlopt

[63] D. Mellinger and V. Kumar, “Minimum snap trajectory generation and control for quadrotors,” in Proc. IEEE Int. Conf. Robot. Autom., 2011, pp. 2520–2525.

[64] J. Sola, “Quaternion kinematics for the error-state Kalman filter,” 2017, arXiv:1711.02508.

[65] S. Thrun, W. Burgard, and D. Fox, Probabilistic Robotics. Cambridge, MA, USA: MIT Press, 2005.

[66] G. Loianno, C. Brunner, G. McGrath, and V. Kumar, “Estimation, control, and planning for aggressive flight with a small quadrotor with a single camera and imu,” IEEE Robot. Autom. Lett., vol. 2, no. 2, pp. 404–411, Apr. 2017.

[67] S. Ren, K. He, R. Girshick, and J. Sun, “Faster R-CNN: Towards real-time object detection with region proposal networks,” in Proc. Annu. Conf. Neural Inf. Process. Syst., vol. 28, 2015. [Online]. Available: https://papers.nips.cc/paper\_files/paper/2015/hash/ 14bfa6bb14875e45bba028a21ed38046-Abstract.html

![](images/829e6f43ad52df15148fcd8e6021b149bcf8ebf162624dc46cad63cedb4dbf2d.jpg)

Guanrui Li (Graduate Student Member, IEEE) received the bachelor’s degree in theoretical and applied mechanics from Sun Yat-Sen University, Guangzhou, China, in 2016, the master’s degree in robotics from the GRASP Lab, University of Pennsylvania, Philadelphia, PA, USA, in 2018, and the Ph.D. degree in electrical and computer engineering, with a focus on robotics and aerial systems, from New York University, New York, NY, USA, in 2024.

He is currently an Assistant Professor with the Worceter Polytechnic Institute (WPI), Worcester,

MA, USA, and the Director of the Aerial-robot Control and Perception Lab (ACP Lab), WPI. His research is centered on the dynamics, planning, and control of robotics systems, with applications in aerial transportation and manipulation, as well as human-robot collaboration

Dr. Li was the recipient of several notable recognitions, including the NSF CPS Rising Stars in 2023, the Outstanding Deployed System Paper Award finalist at 2022 IEEE ICRA, and the 2022 Dante Youla Award for Graduate Research Excellence at NYU. He has an extensive publication record in top-tier robotics conferences and journals like International Conference on Robotics and Automation, IEEE Robotics and Automation Letters, and IEEE TRANSACTIONS ON ROBOTICS, and his work has garnered attention in various media, including IEEE Spectrum and the Discovery Channel. He was recognized as an Honors Undergraduate for his B.Sc. degree.

![](images/4cc421f304b424bc2b90c3a4a5664d5595235981538d1c849f933feb362d45c0.jpg)

Xinyang Liu was born in Beijing, China, in 1999. He received the B.S. degree in mechanical engineering with double minors in robotics and computer science in 2022 from New York University, New York, NY, USA. He is currently working toward the M.S. degree in aeronautics and astronautics with Stanford University, Stanford, CA, USA.

His research interests include control theory, robotics, human–robot interactions, and autonomous systems.

Mr. Liu was a University Honor Scholar at New York University and was a member of Tau Beta Pi.

![](images/30d0173624a51bf16cc035d69fa254c3313ed5395e8074a28651454194cbf209.jpg)

Giuseppe Loianno (Member, IEEE) received the Ph.D. degree in robotics from the University of Naples “Federico II,” Naples, Italy, in 2014.

He is currently an Assistant Professor with the New York University (NYU), New York, NY, USA, and the Director of the Agile Robotics and Perception Lab working on autonomous robots. Prior joining NYU, he was a Postdoctoral Researcher, Research Scientist, and team leader with the GRASP Lab, University of Pennsylvania, Philadelphia, PA, USA. He has authored or coauthored more than 70 conference papers,

journal papers, and book chapters. His research interests include perception, learning, and control for autonomous robots.

Dr. Loianno was the recipient of the NSF CAREER Award in 2022 and DARPA Young Faculty Award in 2022. He was also the recipient of the IROS Toshio Fukuda Young Professional Award in 2022, Conference Editorial Board Best Associate Editor Award at ICRA 2022, Best Reviewer Award at ICRA 2016, and he was selected as Rising Star in AI from KAUST in 2023. He is also currently the Co-Chair of the IEEE RAS Technical Committee on Aerial Robotics and Autonomous Aerial Vehicles. He was the General Chair of the IEEE International Symposium on Safety, Security and Rescue Robotics (SSRR) in 2021 as well as Program Chair in 2019, 2020, and 2022. His work has been featured in a large number of renowned international news and magazines.