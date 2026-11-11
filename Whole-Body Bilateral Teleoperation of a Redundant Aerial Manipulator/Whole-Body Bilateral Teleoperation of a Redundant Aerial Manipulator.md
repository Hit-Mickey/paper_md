# Whole-Body Bilateral Teleoperation of a Redundant Aerial Manipulator

Andre Coelho, Harsimran Singh, Konstantin Kondak, Christian Ott

Abstract— Attaching a robotic manipulator to a flying base allows for significant improvements in the reachability and versatility of manipulation tasks. In order to explore such systems while taking advantage of human capabilities in terms of perception and cognition, bilateral teleoperation arises as a reasonable solution. However, since most telemanipulation tasks require visual feedback in addition to the haptic one, real-time (task-dependent) positioning of a video camera, which is usually attached to the flying base, becomes an additional objective to be fulfilled. Since the flying base is part of the kinematic structure of the robot, if proper care is not taken, moving the video camera could undesirably disturb the endeffector motion. For that reason, the necessity of controlling the base position in the null space of the manipulation task arises. In order to provide the operator with meaningful information about the limits of the allowed motions in the null space, this paper presents a novel haptic concept called Null-Space Wall. In addition, a framework to allow stable bilateral teleoperation of both tasks is presented. Numerical simulation data confirm that the proposed framework is able to keep the system passive while allowing the operator to perform time-delayed telemanipulation and command the base to a task-dependent optimal pose.

## I. INTRODUCTION

The use of unmanned aerial vehicles (UAV) as a flying base for robotic manipulators has been object of intensive research in the recent years [1]–[4]. One of the main goals of such systems is to replace and assist humans in tasks as inspection and repairing of bridges, high-voltage electric lines, and wind-turbine blades [1], [2]. The use of such systems help overcome a series of limitations of fixed-based robots, e.g., limited workspace. Some recent works focused on performing aerial manipulation tasks autonomously [5]– [7]. However, in order to more deeply explore the manipulation capacity of such systems, telemanipulation arises as an alternative [8]–[11]. Adding a human operator to the loop allows for the simplification of many of the system features, especially in what concerns perception and reasoning.

Nevertheless, during field experiments performed in the scope of the AEROARMS project [1], [11], an additional condition for aerial telemanipulation has been found. Since tasks are usually performed without direct eye contact, the use of video cameras to stream images to the distant operator is a necessary add-on to haptic feedback. Adding to that, in order to facilitate the operator’s reasoning, a first-person or eye-to-hand view, i.e., a view of the entire arm, has been found beneficial. In the case of aerial manipulators, like the DLR Suspended Aerial Manipulator (SAM) [11]–[13], a camera is usually attached to the flying base in order to provide an eye-to-hand view (see Fig. 1). If correctly positioned, the camera image allows the operator to have a view of the arm and the area of interest for manipulation.

![](images/326fd53f71fc7efc9eaeb29106d4bc114939d708eaf03cbf88e8c741e472fbf9.jpg)  
Fig. 1: (1) DLR Suspended Aerial Manipulator (SAM). (2) eye-to-hand camera (CAM 1). (3) hand-eye camera (CAM 2).

In case the aerial manipulator is redundant, there might be some freedom to move the base and, consequently, the video camera without affecting the end-effector. For such systems, defining the camera pose autonomously could allow for a better view to the operator. However, since its optimal location is rather subjective and task-dependent, it would be reasonable to provide the operator with some authority to control that task without disturbing the end-effector pose. Adding to that, since the set of possible base poses for a given end-effector pose is usually a smaller subset of the Cartesian-space, the master device can be used in order to inform the operator if the pose to which the base is being commanded is approaching the limits of that subset.

In light of that, this paper proposes a framework to allow a human operator to teleoperate the flying base of a redundant aerial manipulator in order to achieve a desired view of the task being performed while receiving meaningful haptic feedback about the reachability of the commanded pose without disturbing the end-effector. For that purpose, a previously presented whole-body control framework is applied in order to define a task hierarchy, where the positioning of the base is performed in the null space [14] of the main telemanipulation task. Adding to that, a novel haptic concept called Null Space Wall is introduced in order to inform the operator in an intuitive way whether the pose to where the base is being commanded is reachable. This concept is defined in such a way that no additional fixtures have to be added in order to create haptic feedback. Elseways, the already computed forces from the whole-body controller are explored. The use of haptic feedback in order to constrain the commanded motion has been extensively explored in the literature (e.g., [15]–[17]). However, to the best of the authors’ knowledge this is the first time that such a technique is applied in order to allow hierarchical whole-body telemanipulation of redundant robots.

In order to evaluate the proposed framework, bilateral teleoperation of the SAM (see Fig. 1) is simulated with roundtrip time-delays up to 300 ms. The stability and efficacy of the approach in providing the user with meaningful haptic information about the limits of the null space can be verified in the data.

## II. TIME DOMAIN PASSIVITY APPROACH

Time Domain Passivity Approach (TDPA) [18]–[20] was developed in order to enforce stability of both haptic and bilateral teleoperation setups where velocity and force signals are exchanged. In teleoperation systems, time-delay and package losses introduced by the channel might compromise the overall stability of the system, whereas in haptics instability can be a result of sampling.

In bilateral teleoperation setups, the communication channel is usually represented by one or more Time Delay Power Networks (TDPNs, [19]), which are two-port networks that exchange velocities and forces. In addition to constant or variable time-delays, TDPNs can also model package losses in the signals being transmitted. Fig. 2 shows the signal flow of the TDPN. $E ^ { M }$ and $E ^ { S }$ are the energies computed on the master and slave sides, respectively. The in and out subscripts are used to represent the direction of flow, namely into or out of the channel.

![](images/2514ea640bff72b7145746db2738c64e58cf78db41d46908839924b20f75b82e.jpg)  
Fig. 2: Signal flow of the TDPN.

The pairs $\nu _ { 1 } / f _ { 1 }$ and $\nu _ { 2 } / f _ { 2 }$ from Fig. 2 are the flow-effort pairs on each side of the TDPN, such that in coordinates

$$
E ^ {M} (k) = \Delta T \sum_ {j = 1} ^ {k} f _ {1} (j) ^ {T} v _ {1} (j),\tag{1}
$$

$$
E ^ {S} (k) = \Delta T \sum_ {j = 1} ^ {k} - f _ {2} (j) ^ {T} v _ {2} (j),\tag{2}
$$

where ∆T is the sampling time and k is the discrete-time index.

A sufficient condition for passivity of a TDPN is that

$$
E _ {o b s} ^ {L 2 R} (k) = E _ {i n} ^ {M} \left(k - T _ {f} (k)\right) - E _ {o u t} ^ {S} (k) \geq 0, \quad \forall k \geq 0,\tag{3}
$$

$$
E _ {o b s} ^ {R 2 L} (k) = E _ {i n} ^ {S} (k - T _ {b} (k)) - E _ {o u t} ^ {M} (k) \geq 0, \quad \forall k \geq 0,\tag{4}
$$

where $E _ { o b s } ^ { L 2 R } ( k )$ and $E _ { o b s } ^ { R 2 L } ( k )$ are the observed left-to-right and right-to-left energy flows observed on the right and left-hand sides of the TDPN. $T _ { f } ( k )$ and $T _ { b } ( k )$ are the forward and backward delays, respectively.

One of the most common teleoperation schemes is the P-F architecture [21], where the master velocity is sent through the channel and serves as desired velocity to the slave. In turn, the force produced by the slave-side controller is sent back to the master. Following the framework presented by Artigas et al. [22], using a hybrid of circuit and network representation, the slave side of the P-F architecture can be represented as shown in Fig. 3. There, the communication channel is represented by a TDPN; $V _ { m }$ and $V _ { s }$ are the velocities of the master and slave devices; ${ \pmb F } _ { s }$ is the force exerted by the slave-side controller; $V _ { s d }$ is the delayed master velocity; and $\hat { F } _ { m }$ is the delayed slave control force applied to the master device. Adding to that, $V _ { a d }$ and $\beta$ are the the drift compensation velocity source [20], [23] and the admittance-type passivity controller, which will be addressed subsequently.

![](images/671c9a9a5e3bf78425c2453324ef88bf1d8e31ea53304243554481a6e26ba772.jpg)  
Fig. 3: Slave side of a P-F architecture. The passivity observer-passivity controller (PO-PC) pair is applied in admittance configuration $( \beta )$ . $V _ { a d }$ is the drift-compensation velocity.

1) Passivity Observer: In order to take into account the energy removed by the passivity controllers up to the previous time steps $( E _ { P C } ^ { \bar { M } } ( k - 1 )$ and $E _ { P C } ^ { S } ( k - 1 ) )$ ), the energy flow on each side of the TDPN is computed as

$$
W _ {M} (k) = E _ {i n} ^ {S} (k - T _ {b} (k)) - E _ {o u t} ^ {M} (k) + E _ {P C} ^ {M} (k - 1),\tag{5}
$$

$$
W _ {S} (k) = E _ {i n} ^ {M} (k - T _ {f} (k)) - E _ {o u t} ^ {S} (k) + E _ {P C} ^ {S} (k - 1).\tag{6}
$$

2) Passivity Controller: The passivity controller (PC) acts as an adaptive damping in order to guarantee the passivity of the channel. It can be applied in impedance or admittance configuration, according to the architecture requirements. In Fig. 3 the $\mathrm { P C } \left( \beta \right)$ is being applied in admittance configuration in order to modify the velocity coming out of the channel.

In order to fulfill the passivity conditions from (3) and (4) in a P-F architecture, the passivity controller is applied in admittance configuration on the slave side and in impedance configuration on the master side.

The admittance-type PC can be applied as [23]

$$
\boldsymbol {\beta} (k) = d _ {f} (k) \boldsymbol {\Gamma} (k),\tag{7}
$$

$$
d _ {f} (k) = \left\{ \begin{array}{l l} 0 & \text { if } W _ {S} (k) > 0 \\ - \frac {W _ {S} (k)}{\Delta T | | \boldsymbol {F} _ {s} (k) | | _ {\Gamma} ^ {2}} & \text { else, if } | | \boldsymbol {F} _ {s} (k) | | _ {\Gamma} ^ {2} > 0, \end{array} \right.\tag{8}
$$

where

$$
| | \boldsymbol {F} _ {s} (k) | | _ {\Gamma} ^ {2} = \boldsymbol {F} _ {s} (k) ^ {T} \boldsymbol {\Gamma} (k) \boldsymbol {F} _ {s} (k),\tag{9}
$$

and $T ( k )$ is a symmetric positive-definite weighting matrix. The velocity removed by the PC from the delayed master velocity in order to keep the system passive will be

$$
\boldsymbol {V} _ {p c} (k) = \beta (k) \boldsymbol {F} _ {s} (k),\tag{10}
$$

and the resulting velocity used as a reference by the slave will be

$$
\boldsymbol {V} _ {s d} (k) = \hat {\boldsymbol {V}} _ {s d} (k) - \boldsymbol {V} _ {p c} (k),\tag{11}
$$

assuming all velocities are represented in the same frame.

Likewise, the impedance-type passivity controller can be applied as [24]

$$
\alpha (k) = d _ {v} (k) \boldsymbol {\Psi} (k),\tag{12}
$$

$$
d _ {v} (k) = \left\{ \begin{array}{l l} 0 & \text { if } W _ {M} (k) > 0 \\ - \frac {W _ {M} (k)}{\Delta T | | \boldsymbol {V} _ {m} (k) | | _ {\Psi} ^ {2}} & \text { else, if } | | \boldsymbol {V} _ {m} (k) | | _ {\Psi} ^ {2} > 0, \end{array} \right.\tag{13}
$$

where

$$
| | \pmb {V} _ {m} (k) | | _ {\Psi} ^ {2} = \pmb {V} _ {m} (k) ^ {T} \pmb {\varPsi} (k) \pmb {V} _ {m} (k),\tag{14}
$$

and $\varPsi ( k )$ is a symmetric positive-definite weighting matrix. The force removed by the passivity controller from the delayed slave force in order to keep the system passive will be

$$
\boldsymbol {F} _ {p c} (k) = \alpha (k) \boldsymbol {V} _ {m} (k),\tag{15}
$$

and the resulting velocity used as a reference by the slave will be

$$
\pmb {F} _ {m} (k) = \hat {\pmb {F}} _ {m} (k) - \pmb {F} _ {p c} (k).\tag{16}
$$

## III. WHOLE-BODY BILATERAL TELEOPERATION OFREDUNDANT AERIAL MANIPULATORS

## A. Notations and Definitions

1) The Special Euclidean group and its Lie algebra: The pose of a rigid body in space can be represented by elements of the special Euclidean group $S E ( 3 )$ , whose matrix form is

$$
\boldsymbol {g} = \left[ \begin{array}{c c} \boldsymbol {R} & \boldsymbol {p} \\ \boldsymbol {0} & 1 \end{array} \right] \in S E (3),\tag{17}
$$

where $\pmb { p } \in \mathbb { R } ^ { 3 }$ describes the position vector and $\pmb { R }$ is an element of the Special Orthogonal group $S O ( 3 )$ relative to the rotation of the body. Elements of $S E ( 3 )$ can also be identified as $\pmb { g } = ( \pmb { R } , \pmb { p } )$ . Furthermore, the velocity of a rigid body can be expressed by elements of the Lie algebra of $S E ( 3 )$ , namely $\mathfrak { s e } ( 3 )$ , as

$$
[ \boldsymbol {V} ] ^ {\wedge} = \left[ \begin{array}{c c} \widehat {\boldsymbol {\omega}} & \boldsymbol {v} \\ \boldsymbol {0} & 0 \end{array} \right] \in \mathfrak {s e} (3),\tag{18}
$$

where b· indicates the skew-symmetric operator applied to a vector in ${ \mathbb { R } } ^ { 3 }$ and $\boldsymbol { \omega } , \boldsymbol { v } \in \mathbb { R } ^ { 3 }$ are angular and linear velocities, respectively. Adding to that, due to the isomorphism between $\mathfrak { s e } ( 3 )$ and $\mathbb { R } ^ { 6 }$ , it is useful to define the operator $[ \cdot ] ^ { \wedge } :$ $\mathbb { R } ^ { 6 } \to \mathfrak { s e } ( 3 )$ , such that the velocities of rigid bodies can be expressed as elements of $\mathbb { R } ^ { 6 }$ or $\mathfrak { s e } ( 3 )$ . Such velocities can be represented in body $( ^ { B } V )$ , or in spatial frame $( ^ { S } V )$ [14].

2) Dynamical systems in $S E ( 3 )$ : A dynamical system with state $\pmb { \mathscr { g } } \in S E ( 3 )$ and body velocity $[ ^ { B } V ] ^ { \wedge } \in \mathfrak { s e } ( 3 )$ evolves according to the following differential equation in continuous time [25]

$$
\dot {\boldsymbol {g}} (t) = \boldsymbol {g} (t) \left[ ^ {B} \boldsymbol {V} (t) \right] ^ {\wedge},\tag{19}
$$

whose recursive solution in discrete time, given a set of initial conditions, can be approximated to

$$
\boldsymbol {g} (k) = \boldsymbol {g} (k - 1) \exp_ {\mathrm{SE} (3)} \left([ ^ {B} \boldsymbol {V} (k) ] ^ {\wedge} \Delta T\right),\tag{20}
$$

where $\exp _ { \mathrm { S E } ( 3 ) }$ is the $S E ( 3 )$ exponential map [23], [25].

## B. Decoupling Control of Redundant Aerial Manipulators

The dynamics of a fully actuated aerial manipulator with n joints can be written as

$$
M (\boldsymbol {q}) \ddot {\boldsymbol {q}} + C (\boldsymbol {q}, \dot {\boldsymbol {q}}) \dot {\boldsymbol {q}} + G (\boldsymbol {q}) = \tau + \tau_ {e x t},\tag{21}
$$

where $\pmb q \in \mathbb { R } ^ { n }$ is a set of generalized coordinates, $M ( q ) \in$ $\mathbb { R } ^ { n \times n }$ is the inertia matrix, $C ( \pmb q , \dot { \pmb q } ) \dot { \pmb q } \in \mathbb { R } ^ { n }$ is a vector of Coriolis and centrifugal forces, and $G ( q ) \in \mathbb { R } ^ { n }$ is the gravitational generalized torque vector. $\tau \in \mathbb { R } ^ { n }$ and $\tau _ { e x t } \in \mathbb { R } ^ { n }$ are the generalized control and external torque vectors, respectively.

Following the framework presented by Ott et al. [26] for hierarchical whole-body control of kinematically redundant robots, where the number of minimal end-effector pose coordinates, $\pmb { \xi } ( \pmb { q } ) \in \mathbb { R } ^ { m }$ , is less than the number of joint generalized coordinates, i.e., $m < n$ , the set of task coordinates can be extended to

$$
\left[ \begin{array}{c} V _ {x} \\ V _ {n} \end{array} \right] = \bar {\boldsymbol {J}} (\boldsymbol {q}) \dot {\boldsymbol {q}} = \left[ \begin{array}{c} \boldsymbol {J} (\boldsymbol {q}) \\ \boldsymbol {N} (\boldsymbol {q}) \end{array} \right] \dot {\boldsymbol {q}},\tag{22}
$$

where $V _ { \pmb { x } } \in \mathbb { R } ^ { m }$ and $V _ { n } \in \mathbb { R } ^ { n - m }$ are the set of Cartesian and null space velocities, respectively. $\pmb { J } ( \pmb q )$ is the Jacobian matrix that maps generalized to end-effector velocities. In this paper, it is assumed that $V _ { x } \in \mathbb { R } ^ { 6 }$ is the body velocity of the end-effector with respect to some fixed inertial frame. Therefore $\pmb { J } ( \pmb q )$ is the body Jacobian [14] of the Cartesian task. Under the assumption of full row rank of $\pmb { J } ( \pmb q )$ , the matrix $N ( q )$ can be constructed as [26]

$$
\boldsymbol {N} (\boldsymbol {q}) = \left(\boldsymbol {Z} (\boldsymbol {q}) \boldsymbol {M} (\boldsymbol {q}) \boldsymbol {Z} (\boldsymbol {q}) ^ {T}\right) ^ {- 1} \boldsymbol {Z} (\boldsymbol {q}) \boldsymbol {M} (\boldsymbol {q}),\tag{23}
$$

where $\pmb { Z } ( \pmb { q } )$ is a full row rank null space base matrix, such that $J ( \pmb q ) \dot { Z } ( \pmb q ) ^ { T } = \pmb 0$

Such formulation allows the manipulator dynamics to be written as

$$
\begin{array}{l} \left[ \begin{array}{c c} \boldsymbol {\Lambda} _ {\boldsymbol {x}} (\boldsymbol {q}) & 0 \\ 0 & \boldsymbol {\Lambda} _ {\boldsymbol {n}} (\boldsymbol {q}) \end{array} \right] \left[ \begin{array}{c} \dot {\boldsymbol {V}} _ {\boldsymbol {x}} \\ \dot {\boldsymbol {V}} _ {\boldsymbol {n}} \end{array} \right] + \left[ \begin{array}{c c} \mu_ {\boldsymbol {x}} (\boldsymbol {q}, \dot {\boldsymbol {q}}) & \mu_ {\boldsymbol {x n}} (\boldsymbol {q}, \dot {\boldsymbol {q}}) \\ \mu_ {\boldsymbol {n x}} (\boldsymbol {q}, \dot {\boldsymbol {q}}) & \mu_ {\boldsymbol {n}} (\boldsymbol {q}, \dot {\boldsymbol {q}}) \end{array} \right] \left[ \begin{array}{c} \boldsymbol {V} _ {\boldsymbol {x}} \\ \boldsymbol {V} _ {\boldsymbol {n}} \end{array} \right] \\ = \bar {\boldsymbol {J}} (\boldsymbol {q}) ^ {- T} (\tau + \tau_ {e x t} - \boldsymbol {G} (\boldsymbol {q})). \end{array} \tag {24}
$$

where $\varLambda _ { x } ( q )$ and $\varLambda _ { n } ( q )$ are the elements of a blockdiagonal Cartesian inertia matrix $\pmb { \varDelta } ( \pmb q )$ . In addition, $\mu _ { x } ( q , \dot { q } )$ , $\mu _ { x n } ( q , \dot { q } ) , \mu _ { n x } ( q , \dot { q } )$ , and $\mu _ { n } ( q , \dot { q } )$ are the elements of the Cartesian Coriolis-centrifugal matrix $\mu ( q , \dot { q } )$ .

In order to control the pose of the flying base as the secondary task and the end-effector as the primary one, the following hierarchically-decoupling control law can be applied [26]

$$
\boldsymbol {\tau} = \tau_ {x} + \tau_ {n} + \tau_ {\mu} + \boldsymbol {G} (\boldsymbol {q}),\tag{25}
$$

$$
\boldsymbol {\tau} _ {x} = \boldsymbol {J} (\boldsymbol {q}) ^ {T} \boldsymbol {F} _ {\boldsymbol {x}},\tag{26}
$$

$$
\boldsymbol {\tau} _ {n} = \boldsymbol {N} (\boldsymbol {q}) ^ {T} \boldsymbol {Z} (\boldsymbol {q}) \boldsymbol {J} _ {\mathrm{b}} (\boldsymbol {q}) ^ {T} \boldsymbol {F} _ {b},\tag{27}
$$

$$
\tau_ {\mu} = \bar {J} (\boldsymbol {q}) ^ {T} \left[ \begin{array}{c c} \boldsymbol {0} & \mu_ {x n} (\boldsymbol {q}, \dot {\boldsymbol {q}}) \\ \mu_ {n x} (\boldsymbol {q}, \dot {\boldsymbol {q}}) & \boldsymbol {0} \end{array} \right] \left[ \begin{array}{c} V _ {x} \\ V _ {n} \end{array} \right],\tag{28}
$$

where $J _ { \mathrm { b } } ( \pmb q )$ is the Jacobian that maps the joint velocities $\dot { \pmb q }$ to the body velocities of the flying base $V _ { b } \in \mathbb { R } ^ { 6 }$ . Moreover, ${ \pmb F } _ { \pmb x }$ is a Cartesian-space wrench applied to the end-effector and $\pmb { F } _ { b }$ is the control wrench of the secondary task, before being projected into the null space of the primary one. Both wrenches are assumed to be body wrenches [14].

In order to control the pose of the end-effector and of the flying base (as a secondary task), the following error elements can be defined

$$
\boldsymbol {g} _ {\mathrm{Ex}} = \boldsymbol {g} _ {x, d e s} ^ {- 1} \boldsymbol {g} _ {x} = \left(\boldsymbol {R} _ {\mathrm{Ex}}, \boldsymbol {p} _ {\mathrm{Ex}}\right), \boldsymbol {g} _ {\mathrm{Eb}} = \boldsymbol {g} _ {b, d e s} ^ {- 1} \boldsymbol {g} _ {b} = \left(\boldsymbol {R} _ {\mathrm{Eb}}, \boldsymbol {p} _ {\mathrm{Eb}}\right),\tag{29}
$$

where ${ \bf g } _ { x , d e s } , ~ { \bf g } _ { x } , ~ { \bf g } _ { b , d e s } ,$ , and $\mathbf { \nabla } _ { \mathbf { \boldsymbol { { g } } } _ { b } }$ are homogeneous transformation matrices in $S E ( 3 )$ , which describe the desired and current poses of the end-effector (subscript x) and of the flying base (subscript b) with respect to a fixed inertial frame. The body velocities relative to these elements are $V _ { x , d e s } ,$ $V _ { x } , \ V _ { b , d e s }$ , and $V _ { b } ,$ , respectively, which relate to their time derivative according to (19).

To accomplish the desired tasks, the body Cartesian control wrenches ${ \pmb F } _ { \pmb x }$ and $\pmb { F } _ { b }$ can be defined as [27], [28]

$$
\boldsymbol {F} _ {\boldsymbol {x}} = \left[ \begin{array}{c} - \boldsymbol {R} _ {\mathrm{Ex}} ^ {T} \boldsymbol {K} _ {P x} \boldsymbol {p} _ {\mathrm{Ex}} \\ - 2 \boldsymbol {R} _ {\mathrm{Ex}} ^ {T} \boldsymbol {E} (\eta_ {E x}, \varepsilon_ {E x}) ^ {T} \boldsymbol {K} _ {O x} \varepsilon_ {E x} \end{array} \right] + \boldsymbol {K} _ {D x} \boldsymbol {V} _ {E x},\tag{30}
$$

$$
\boldsymbol {F} _ {b} = \left[ \begin{array}{c} - \boldsymbol {R} _ {\mathrm{Eb}} ^ {T} \boldsymbol {K} _ {P b} \boldsymbol {p} _ {\mathrm{Eb}} \\ - 2 \boldsymbol {R} _ {\mathrm{Eb}} ^ {T} \boldsymbol {E} (\eta_ {E b}, \varepsilon_ {E b}) ^ {T} \boldsymbol {K} _ {O b} \varepsilon_ {E b} \end{array} \right] + \boldsymbol {K} _ {D b} \boldsymbol {V} _ {E b},\tag{31}
$$

where the matrices $K _ { ( . ) }$ are positive definite gain matrices. Furthermore, $\eta _ { E x }$ and $\varepsilon _ { E x }$ are the scalar and vector parts of a quaternion representation of $R _ { \mathrm { E x } }$ , respectively, and $E ( \eta _ { E x } , \varepsilon _ { E x } ) = \eta _ { E x } I _ { 3 } - \widehat \varepsilon _ { E x }$ . Moreover, $V _ { E x }$ and $V _ { E b }$ are the body velocities of $\mathbf { \pmb { g } } _ { E x }$ and $\mathbf { \Im } _ { \mathbf { \mathcal { I } } \mathit { E } b }$ , respectively.

It is important to note in (27) that the control force $\pmb { F } _ { b }$ goes through a projection before being commanded to the actuators. In this projection, the components of $\pmb { F } _ { b }$ that conflict with the primary task are projected into zero and will not be commanded to the actuators via $\tau _ { n }$

## C. Null-Space Wall Concept

Although $\pmb { F } _ { b }$ is not always entirely applied to the base, it carries out important information about the limits of the null space of the primary task. In case the base is teleoperated, the operator directly commands $g _ { b , d e s }$ and $V _ { b , d e s }$ . However, in case a position which is not reachable without affecting the end-effector position is commanded, the resulting force $\pmb { F } _ { b }$ will be projected to zero. If the operator keeps commanding $V _ { b , d e s }$ towards that direction, the errors between desired and current pose in (31) will increase. As a consequence, an increase in $\pmb { F } _ { b }$ will be observed. This behavior resembles what happens when the robot being teleoperated hits a wall. Therefore, by sending $\pmb { F } _ { b }$ as haptic feedback, the operator will have the feeling as if the robot were trying to penetrate a wall when an unfeasible pose is commanded to the flying base. The stiffness of that wall will be initially defined by $K _ { P b }$ and $K _ { O b }$ , but could be scaled according to the task requirements.

![](images/6b3747c1831fa62d8cdebd20d973fcbf4e58d48da2b11fb745514bfc647d07e5.jpg)  
Fig. 4: The Null-Space Wall concept.

On the other hand, if the base pose commanded by the operator is reachable without disturbing the end-effector, the operator will perceive lower forces, which are due to the imperfect tracking capabilities of the controller, as is usually the case for a P-F architecture in teleoperation. Therefore, for each end-effector pose there would be a region where the operator will perceive the robot in free-space motion and another where a wall penetration feeling will be perceived, which is the case when the commanded position is not feasible. An illustration of this behavior can be seen in Fig. 4.

It is important to remark that, without the aforementioned haptic feedback the operator would have to rely on visual feedback in order to know if the flying base is following the commands. This could be a complex task since (1) the camera is usually attached to the base, which, therefore, does not appear in the frame, (2) it is hard to distinguish motion of the base from that of the end-effector in the camera images. On the other hand, the wall penetration feeling perceived by the operator would indicate that the desired direction of motion is not feasible.

## D. Bilateral Null-Space Teleoperation

In order to allow bilateral teleoperation, the slave side of the P-F architecture (see Fig. 3) can be modified to implement a logic where the operator is able to switch between the primary task and the secondary one. For that purpose, the whole-body architecture depicted in Fig. 5 is proposed. In that architecture, the operator has the possibility of switching between the primary and the secondary task.

![](images/884efe4ec391739ac404bd5075bba26f67996b51ccfb501004f57c2da458a856.jpg)  
Fig. 5: Block diagram of the proposed approach.

Initially, the initial values of $\mathbf { \sigma } _ { \mathbf { \boldsymbol { g } } _ { X , d e s } }$ and $g _ { b , d e s }$ have to be set in the integrators represented by the summation blocks, which implement (20). When the switch is set for the primary task, the velocity commanded by the operator will be integrated into $\mathbf { \ u } _ { g _ { X , d e s } }$ while $g _ { b , d e s }$ will hold its value $( V _ { x , d e s } ( k ) = V _ { s d } ( k ) , ~ V _ { b , d e s } ( k ) = \mathbf { 0 } )$ . In addition, the force ${ \pmb F } _ { \pmb x }$ will be fed back through the communication channel while $\pmb { F } _ { b }$ will act locally in order to keep the base at its last commanded pose. On the other hand, when the operator switches to the secondary task, the commanded velocity will be integrated into $g _ { b , d e s } \ ( V _ { b , d e s } ( k ) = V _ { s d } ( k ) , V _ { x , d e s } ( k ) = 0 )$ and the force $\pmb { F } _ { b }$ will be sent back through the channel before going through the change of coordinates defined by $J _ { \mathrm { b } } ( \mathbf { \bar { q } } ) ^ { T }$ and the projection defined by ${ \cal N } ( q ) ^ { T } { \cal Z } ( q )$ . In addition, $\tau _ { \mu } + G ( \pmb q )$ will be applied at all moments in order to compensate for gravity and diagonalize the matrix $\mu ( q , \dot { q } )$ (see Section III-B).

It is also important to note that, if the communication channel introduces time-delay and/or package loss, the teleoperation system is likely to become unstable. Therefore, in order to keep the communication channel passive, four PO-PC pairs can be implemented, two on the master and two on the slave side $( { \bf P } { \bf C } _ { n }$ and $\mathbf { P C } _ { x }$ in Fig. 5). These controllers will be responsible for monitoring and correcting the value of four potential functions, namely $W _ { M , x } ( k ) , W _ { S , x } ( k ) , W _ { M , b } ( k )$ and $W _ { S , b } ( k )$ , which can be defined as

$$
W _ {M, x} (k) = E _ {i n, x} ^ {S} (k - T _ {b} (k)) - E _ {o u t, x} ^ {M} (k) + E _ {P C, x} ^ {M} (k - 1)\tag{32}
$$

$$
W _ {S, x} (k) = E _ {i n, x} ^ {M} \left(k - T _ {f} (k)\right) - E _ {o u t, x} ^ {S} (k) + E _ {P C, x} ^ {S} (k - 1),\tag{33}
$$

$$
W _ {M, b} (k) = E _ {i n, b} ^ {S} (k - T _ {b} (k)) - E _ {o u t, b} ^ {M} (k) + E _ {P C, b} ^ {M} (k - 1)\tag{34}
$$

$$
W _ {S, b} (k) = E _ {i n, b} ^ {M} (k - T _ {f} (k)) - E _ {o u t, b} ^ {S} (k) + E _ {P C, b} ^ {S} (k - 1)\tag{35}
$$

The directions in and out of the energies are defined by the sign of the power variables $P ^ { M } ( k )$ and $P ^ { S } ( k )$ [19], which can be calculated as

$$
P ^ {M} (k) = \boldsymbol {V} _ {m} (k) ^ {T} \hat {\boldsymbol {F}} _ {m} (k),\tag{36}
$$

$$
\boldsymbol {P} ^ {S} (k) = \hat {\boldsymbol {V}} _ {s d} (k) ^ {T} \boldsymbol {F} _ {s} (k),\tag{37}
$$

where $V _ { m } ( k )$ and $\hat { F } _ { m } ( k )$ are the velocity of the master device and the force applied to it, as defined in Section II, while $\hat { V } _ { s d } ( k )$ and ${ \cal F } _ { s } ( k )$ will be the desired velocity and the computed force of the primary or the secondary task, depending on which one is being teleoperated (see Fig. 5). Furthermore, the energy values can then be defined as

$$
E _ {x} ^ {M} (k) = \left\{ \begin{array}{l l} E _ {x} ^ {M} (k - 1) + \Delta T P ^ {M} (k) & \text { if } N S = 0 \\ E _ {x} ^ {M} (k - 1) & \text { else }, \end{array} \right.\tag{38}
$$

$$
E _ {x} ^ {S} (k) = \left\{ \begin{array}{l l} E _ {x} ^ {S} (k - 1) + \Delta T P ^ {S} (k) & \text { if } N S = 0 \\ E _ {x} ^ {S} (k - 1) & \text { else }, \end{array} \right.\tag{39}
$$

$$
E _ {b} ^ {M} (k) = \left\{ \begin{array}{l l} E _ {b} ^ {M} (k - 1) + \Delta T P ^ {M} (k) & \text { if } N S = 1 \\ E _ {b} ^ {M} (k - 1) & \text { else }, \end{array} \right.\tag{40}
$$

$$
E _ {b} ^ {S} (k) = \left\{ \begin{array}{l l} E _ {b} ^ {S} (k - 1) + \Delta T P ^ {S} (k) & \text { if } N S = 1 \\ E _ {b} ^ {S} (k - 1) & \text { else }, \end{array} \right.\tag{41}
$$

![](images/f709892a210d6fc54fcc939518d2f49790ecef811e9d893bf04deb1f1373a5cb.jpg)  
Fig. 6: Null-space-wall penetration – no time-delay.

where NS is the flag for null-space teleoperation.

The use of separate energy functions and their dissipation only in the respective hierarchical level was chosen in order to prevent the loss of the energetic decoupling between them. However, the cross-coupled dissipation law presented in [24] can also be applied without compromising the passivity of the system.

## IV. NUMERICAL VALIDATION

In order to validate the proposed whole-body teleoperation approach, three sets of numerical simulation were performed. In all of them, the master device was modeled as a nonredundant robot with dynamics as in (21) and the slave was the simulated SAM. Initially, the linear axes of the base were teleoperated in the null space of the end-effector, which was locally commanded to keep its initial pose. In that task whose purpose was to clearly show the behavior of the null-space wall, no time-delay was simulated. Fig. 6 shows the positions and forces of the flying base, which shows that, between $t \approx 7$ and $t \approx 1 7$ seconds, the position commanded by the master $x _ { b } ^ { M } , y _ { b } ^ { M }$ , and $z _ { b } ^ { M }$ could not be followed by the slave because it reached the limits of the null-space motion. From that moment on, the interaction forces were increased as the master kept penetrating the null-space wall. The force in the opposite direction, which is fed to the master device, in this case, would inform the operator that that the commanded position is not reachable.

The second task involved teleoperating the end-effector under 300 ms round-trip delays and, at $t = 5$ seconds (dashed vertical line), switching to the base teleoperation. For the sake of simplicity, the weighting matrices $T ( k )$ and $\varPsi ( k )$ described in Section II were set to be identity matrices. Additionally, the drift compensator from [23] was applied in order to avoid drift caused by TDPA. Fig. 7 displays the master and slave positions of both the end-effector and the flying base, and the force transmitted through the communication channel, which before 5 seconds is relative to the free motion of the end-effector and, after that, is due to the motion of the flying base, which hits the null-space wall twice. It can be noted that the end-effector motion also required the flying base to slightly move, which is due to its higher level of priority in the control structure. Adding to that, since the end-effector was teleoperated in free-space motion, the force feedback provided to the user in that period is much smaller that the one generated by the reaching of the null-space limits, which resembles wall-contact forces. It can also be seen that, since there is a dependency between the wall position in all three axes, the wall is not completely flat. However, since the end-effector is steady, the curvature of the wall is reasonably smooth. Fig. 8 presents the two energy flow of the channel. It can be noted that the TDPA was able to keep the channel passive at all time steps.

![](images/c10c658ad4b554b7c97ceea461a9aee7a06dd0fc7484623818c841b590f89b45.jpg)

![](images/f86ea8f69c3420aee9a97b01ec898c90c21b5a37ea13a727c227192eaa01afff.jpg)  
Fig. 7: Whole-body teleoperation, positions for primary (left) and secondary task (middle), and haptic forces $( \mathrm { r i g h t } ) - T _ { r t } =$ 300 ms.  
Fig. 8: Whole-body teleoperation, input and output energies for primary (left) and secondary task $( \mathrm { r i g h t } ) - T _ { r t } = 3 0 0$ ms.

The last task aimed at analyzing the behavior of the wall when the end-effector is moving. For that purpose, after switching to null-space teleoperation, the end-effector was commanded to move autonomously in open loop. It can be seen from Fig. 9 that the limits of the null-space wall changed according to the end-effector motion, making the slave oscillate when in contact with the wall. It is also important to remark that the haptic forces displayed a high frequency sinusoidal wave, which is due to the end-effector motion. This feature could also be beneficial to inform the operator in case the end-effector is externally disturbed during teleoperation of the flying base. In can be noted that, despite not being included in the theoretical analysis, a moving null-space wall seems not to destabilize the system.

![](images/4e2aa332ad0a9c389420d258d2e6628dc088df12558dbdb60320e19013c0fa2b.jpg)

Fig. 9: Whole-body teleoperation with autonomous endeffector motion during null-space teleoperation, positions for primary (left) and secondary task (middle), and haptic forces (right) – $T _ { r t } = 3 0 0$ ms.  
![](images/383a653bc04cdd86e20fbe88086ec68b7599be3e58402a51ab8afa6a6a7ab65f.jpg)

![](images/81ca842cf03ae35d3dc9cb08c25f14164bdff5634780952b3d5238f03535afca.jpg)

![](images/b0c619b4d89e929f3eb7f22c686f9f4a7dbef3f2ed911cbd1fec4c01884e2926.jpg)

![](images/07dd2a0fe1359885c243e0a95beb6995e915f6d496f3d0c67ea3d1f6f886b435.jpg)  
Fig. 10: Whole-body teleoperation with autonomous endeffector motion during null-space teleoperation, input and output energies for primary (left) and secondary task (right) $- \ T _ { r t } = 3 0 0 \ \mathrm { m s } .$

## V. CONCLUSION AND FUTURE WORK

This paper presented a framework for performing stable bilateral teleoperation of both the end-effector and the base of redundant aerial manipulators. This feature is especially beneficial when a video camera is attached to the flying base whose position has to be changed without disturbing the endeffector. It has been shown that applying a hierarchicallydecoupling controller and feeding back the secondary control force before being multiplied by the projection term provides the operator with a wall-contact feeling whenever the base is commanded to move to an unreachable position. This characteristic was confirmed by numerical simulation data, which also showed that the proposed framework can render the system passive in the presence of communication delays. Future work will involve applying the proposed framework to teleoperation of arbitrary hierarchy levels [26]. In addition, an extension to a four-channel architecture [29] to include measured-force feedback will be studied.

[1] A. Ollero, G. Heredia, A. Franchi, G. Antonelli, K. Kondak, A. Sanfeliu, A. Viguria, J. R. Martinez-de Dios, F. Pierri, J. Cortes, A. Santamaria-Navarro, M. A. Trujillo Soto, R. Balachandran, J. Andrade-Cetto, and A. Rodriguez, “The aeroarms project: Aerial robots with advanced manipulation capabilities for inspection and maintenance,” IEEE Robotics Automation Magazine, vol. 25, no. 4, pp. 12–23, Dec 2018.

[2] F. Ruggiero, V. Lippiello, and A. Ollero, “Aerial manipulation: A literature review,” IEEE Robotics and Automation Letters, vol. 3, no. 3, pp. 1957–1964, 2018.

[3] H. B. Khamseh, F. Janabi-Sharifi, and A. Abdessameud, “Aerial manipulationa literature survey,” Robotics and Autonomous Systems, vol. 107, pp. 221–235, 2018.

[4] M. Orsag, C. Korpela, P. Oh, S. Bogdan, and A. Ollero, Aerial Manipulation. Springer, 2018.

[5] J. Thomas, J. Polin, K. Sreenath, and V. Kumar, “Avian-inspired grasping for quadrotor micro uavs,” in ASME 2013 International Design Engineering Technical Conferences and Computers and Information in Engineering Conference. American Society of Mechanical Engineers, 2013.

[6] K. Kondak, F. Huber, M. Schwarzbach, M. Laiacker, D. Sommer, M. Bejar, and A. Ollero, “Aerial manipulation robot composed of an autonomous helicopter and a 7 degrees of freedom industrial manipulator,” in 2014 IEEE international conference on robotics and automation (ICRA). IEEE, 2014, pp. 2107–2112.

[7] A. E. Jimenez-Cano, P. J. Sanchez-Cuevas, P. Grau, A. Ollero, and G. Heredia, “Contact-based bridge inspection multirotors: Design, modeling, and control considering the ceiling effect,” IEEE Robotics and Automation Letters, vol. 4, no. 4, pp. 3561–3568, 2019.

[8] G. Gioioso, A. Franchi, G. Salvietti, S. Scheggi, and D. Prattichizzo, “The flying hand: A formation of uavs for cooperative aerial telemanipulation,” in 2014 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2014, pp. 4335–4341.

[9] M. Mohammadi, A. Franchi, D. Barcelli, and D. Prattichizzo, “Cooperative aerial tele-manipulation with haptic feedback,” in 2016 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2016, pp. 5092–5098.

[10] S. Islam, P. X. Liu, A. El Saddik, R. Ashour, J. Dias, and L. D. Seneviratne, “Artificial and virtual impedance interaction force reflectionbased bilateral shared control for miniature unmanned aerial vehicle,” IEEE Transactions on Industrial Electronics, vol. 66, no. 1, pp. 329– 337, 2019.

[11] J. Lee, R. Balachandran, Y. Sarkisov, M. De Stefano, A. Coelho, K. Shinde, M. J. Kim, R. Triebel, and K. Kondak, “Visual-inertial telepresence for aerial manipulation,” in 2020 IEEE International Conference on Robotics and Automation (ICRA), 2020.

[12] Y. S. Sarkisov, M. J. Kim, D. Bicego, D. Tsetserukou, C. Ott, A. Franchi, and K. Kondak, “Development of sam: cable-suspended aerial manipulator,” in IEEE International Conference on Robotics and Automation (ICRA), 2019.

[13] Y. Sarkisov, M. J. Kim, A. Coelho, D. Tsetserukou, C. Ott, and K. Kondak, “Optimal oscillation damping control of cable-suspended aerial manipulator with a single imu sensor,” in 2020 IEEE International Conference on Robotics and Automation (ICRA), 2020.

[14] R. M. Murray, A mathematical introduction to robotic manipulation. CRC press, 1994.

[15] H. Boessenkool, D. A. Abbink, C. J. Heemskerk, F. C. van der Helm, and J. G. Wildenbeest, “A task-specific analysis of the benefit of haptic shared control during telemanipulation,” IEEE Transactions on Haptics, vol. 6, no. 1, pp. 2–12, 2012.

[16] C. Masone, M. Mohammadi, P. Robuffo Giordano, and A. Franchi, “Shared planning and control for mobile robots with integral haptic feedback,” The International Journal of Robotics Research, vol. 37, no. 11, pp. 1395–1420, 2018.

[17] M. M. Marinho, B. V. Adorno, K. Harada, and M. Mitsuishi, “Dynamic active constraints for surgical robots using vector-field inequalities,” IEEE Transactions on Robotics, vol. 35, no. 5, pp. 1166–1185, Oct 2019.

[18] B. Hannaford and J.-H. Ryu, “Time-domain passivity control of haptic interfaces,” IEEE Transactions on Robotics and Automation, vol. 18, no. 1, pp. 1–10, 2002.

[19] J.-H. Ryu, J. Artigas, and C. Preusche, “A passive bilateral control scheme for a teleoperator with time-varying communication delay,” Mechatronics, vol. 20, no. 7, pp. 812–823, 2010.

[20] A. Coelho, H. Singh, T. Muskardin, R. Balachandran, and K. Kondak, “Smoother position-drift compensation for time domain passivity approach based teleoperation,” in 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2018, pp. 5525–5532.

[21] D. A. Lawrence, “Stability and transparency in bilateral teleoperation,” IEEE transactions on robotics and automation, vol. 9, no. 5, pp. 624– 637, 1993.

[22] J. Artigas, J.-H. Ryu, C. Preusche, and G. Hirzinger, “Network representation and passivity of delayed teleoperation systems,” in 2011 IEEE/RSJ International Conference on Intelligent Robots and Systems. IEEE, 2011, pp. 177–183.

[23] A. Coelho, C. Ott, H. Singh, F. Lizarralde, and K. Kondak, “Multidof time domain passivity approach based drift compensation for telemanipulation,” in 2019 19th International Conference on Advanced Robotics (ICAR), Dec 2019, pp. 695–701.

[24] C. Ott, J. Artigas, and C. Preusche, “Subspace-oriented energy distribution for the time domain passivity approach,” in 2011 IEEE/RSJ International Conference on Intelligent Robots and Systems. IEEE, 2011, pp. 665–671.

[25] F. Bullo and R. M. Murray, “Proportional derivative (pd) control on the euclidean group,” in European Control Conference, vol. 2, 1995, pp. 1091–1097.

[26] C. Ott, A. Dietrich, and A. Albu-Schaffer, “Prioritized multi-task¨ compliance control of redundant manipulators,” Automatica, vol. 53, pp. 416–423, 2015.

[27] S. Zhang and E. D. Fasse, “Spatial compliance modeling using a quaternion-based potential function method,” Multibody System Dynamics, vol. 4, no. 1, pp. 75–101, 2000.

[28] C. Ott, M. A. Roa, and G. Hirzinger, “Posture and balance control for biped robots based on contact force optimization,” in 2011 11th IEEE-RAS International Conference on Humanoid Robots, Oct 2011, pp. 26–33.

[29] J. Artigas, R. Balachandran, C. Riecke, M. Stelzer, B. Weber, J.-H. Ryu, and A. Albu-Schaeffer, “Kontur-2: force-feedback teleoperation from the international space station,” in 2016 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2016, pp. 1166–1173.