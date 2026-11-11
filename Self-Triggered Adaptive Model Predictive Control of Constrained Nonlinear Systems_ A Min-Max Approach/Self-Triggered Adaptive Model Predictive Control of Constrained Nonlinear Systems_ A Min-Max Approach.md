# Self-Triggered Adaptive Model Predictive Control of Constrained Nonlinear Systems: A Min-Max Approach

Kunwu Zhang, Student Member, IEEE, Changxin Liu, Student Member, IEEE, and Yang Shi, Fellow, IEEE

Abstract—In this paper, a self-triggered adaptive model predictive control (MPC) algorithm is proposed for constrained discrete-time nonlinear systems subject to parametric uncertainties and additive disturbances. To bound the parametric uncertainties with reduced overestimation, a zonotope-based setmembership parameter estimator is developed, which is also compatible with the aperiodic sampling resulted from the selftriggering mechanism. The estimation of uncertainties is employed to reformulate the optimization problem in a min-max MPC scheme to reduce the conservatism. By designing a timevarying penalty in the cost function, the estimation of uncertainties is implicitly considered in the self-triggering scheduler, therefore making the triggering interval further optimized. The resulting self-triggered adaptive MPC algorithm guarantees the recursive feasibility, while providing less conservative performance compared with the self-triggered robust MPC method. Furthermore, we theoretically show that the closed-loop system is input-to-state practical stable (ISpS) at triggering time instants. A numerical example and comparison study are performed to demonstrate the efficacy of the proposed method.

Index Terms—Adaptive model predictive control, uncertain nonlinear systems, self-triggered control, robust control

## I. INTRODUCTION

The problem of addressing the computation and communication constraints explicitly in networked dynamic systems has attracted increasing attention in recent years [1]. Compared with periodic implementations, the event-based aperiodic control is a more promising solution to achieve the trade-off between the closed-loop performance and the overall communication load, since the control input is not calculated and transmitted until a certain well-defined event related to closed-loop behaviors occurs. Such an event is generally triggered at time instants when the system output or state leaves a certain set [1]–[3]. Hence, even-triggered control requires continuously monitoring system states to determine the computation and communication of control inputs, which may be infeasible for some networked systems with limited communication resources. To further reduce the communication load, the self-triggered approaches have been proposed, where the next sampling time instant is determined by the triggering scheduler at the current time instant so that the system states are only measured at triggering time instants [4]. A comprehensive introduction to event- and self-triggered control can be referred to [1].

In past decades, model predictive control (MPC) has achieved the phenomenal success in process industries due to its capability of efficiently handling hard constraints on inputs and states for complicated systems [5]. In MPC, the control input is obtained by solving a finite-horizon optimal control problem at each time instant [6], [7]. This strategy ensures the optimal performance with respect to a certain performance index, however, it inherently introduces the increased computational complexity and hence may restrict its application to many practical control problems. Unlike conventional periodic MPC, in event-triggered MPC the new control inputs are only computed and transmitted if a certain triggering threshold is reached, consequently saving communication and computational power, e.g. [8]–[10].

Compared with the event-triggered method, using the selftriggered approach can not only reduce the average frequency of computing the control input, but also reduce the overall communication load since the system state or output is only measured and transmitted at triggering time instants [1]. Some results addressing self-triggered MPC have been reported in the literature, e.g., [11]–[17]. A constrained nonlinear system is considered in [14], where the authors proposed a co-design strategy such that the maximum triggering interval and the optimal control inputs can be simultaneously obtained by solving an optimization problem. For disturbed linear systems, a tube-based self-triggered MPC algorithm is presented in [16], where the static state tube is constructed to guarantee the robust constraint satisfaction. To enlarge the region of attraction, the integration of self-triggered MPC with homothetic tubes is proposed in [17], where the state tubes are optimized online to reduce the conservatism. The probabilistic constraints and stochastic disturbances are considered in [12], [18]. For nonlinear systems, in [11] a robust self-triggered MPC scheme is developed for unconstrained nonlinear affine systems, where the triggering interval is maximized by comparing the current optimal cost and the predicted optimal cost. To relieve the computational burden, the authors in [13] have proposed an adaptive mechanism for the prediction horizon in the dual mode MPC framework. By combining the self-triggering mechanism with the min-max MPC strategy, a recent work in [15] provided a novel robust self-triggered MPC algorithm for general nonlinear systems considering both parametric uncertainties and additive disturbances.

In this work, we investigate self-triggered adaptive MPC for discrete-time nonlinear systems subject to both parametric uncertainties and additive disturbances. Our primary motivation is as follows. In most of aforementioned works on self-triggered robust MPC approaches, the uncertainty is handled by considering its worst-case realization or tightening state constraints, which essentially and heavily relies on the initial guess for bounds on uncertainties. Inherently, those methods are conservative for handling fixed or slowly changing uncertainties. It has been found in studies that adaptive MPC is a promising solution to mitigate the conservatism of robust MPC [19], [20], where its main insight is to accommodate the online estimation of uncertainty within a robust MPC framework. To the best of our knowledge, selftriggered adaptive MPC for nonlinear systems has not been studied since synthesizing uncertainty estimation with selftriggered robust MPC introduces some new theoretical and practical problems. For example, estimating the uncertainty based on the input and state history may result in recursive updates of the system model, deteriorating the performance or even destroy the closed-loop stability, especially for nonlinear systems. In addition, the self-triggering scheduler makes the system sampled aperiodically, which renders the uncertainty estimation scheme infeasible. Another remarkable difficulty lies in guaranteeing the robust constraint satisfaction for the aperiodically sampled system with online changing models.

To solve these problems, we develop a self-triggered adaptive MPC algorithm with the following features: 1) A zonotope-based set-membership parameter estimator is developed to approximate the feasible solution set (FSS) of unknown parameters with reduced overestimation. By estimating the reachable set of system states, the proposed estimator becomes compatible with the aperiodic sampling. 2) The codesign of MPC optimization and triggering time instants is considered in the proposed self-triggering mechanism. According to the estimated FSS (EFSS), we firstly construct the cost function consisting of the penalized stage costs with the open loop prediction, the stage costs with the closed-loop prediction and the terminal cost. The open loop stage costs are penalized with time-varying weights so that the new estimation of uncertainty is implicitly considered in the proposed selftriggering mechanism. By comparing the optimal cost with different open loop scenarios, the optimal triggering intervals are determined, therefore leading to the reduced average sampling frequency in the closed-loop system. 3) A selftriggered adaptive MPC algorithm is proposed based on the min-max MPC technique. We effectively facilitate the online parameter adaption in the proposed MPC scheme by reformulating the optimization problem based on the new EFSS. The resulting self-triggered adaptive MPC algorithm guarantees the recursive feasibility, while providing comparable closedloop performance and reduced average sampling frequency compared with the self-triggered robust MPC method. It is also theoretically shown that the closed-loop system is inputto-state practical stable (ISpS) at triggering time instants.

The remainder of this paper is organized as follows: In Section II, the problem setup is demonstrated. Section III describes the design of the set-membership parameter estimator under the self-triggering mechanism. An self-triggered adaptive MPC algorithm is presented in Section IV, followed by the analysis of the theoretical properties. In Section V, a numerical example and comparison study are given. Finally, some concluding remarks are presented in Section VI.

Notation: In this paper, we use the notations $\mathbb { R } , \mathbb { R } ^ { n }$ and $\mathbb { R } ^ { m \times n }$ to denote the sets of real numbers, column real vectors with n elements and real matrices consisting of n columns and m rows, respectively. The set $\mathbb { B } ^ { m } = \left\{ b \in \mathbb { R } ^ { m } : \| b \| _ { \infty } \leq 1 \right\}$ is called a unit hypercube of order m. Let N denote the set of non-negative integers, then sets $ { \mathbb { N } } _ { \geq a }$ and $\mathbb { N } _ { [ a , b ] }$ represent $\{ x \in$ $\mathbb { N } : x \geq a \}$ and $\{ x \in \mathbb { N } : a \leq { \bar { x } } \leq b , b \geq a \}$ , respectively. Similarly, the notation $\mathbb { R } _ { \geq a }$ stands for the set $\{ x \in \mathbb { R } : x \geq a \}$ Given a vector $x \in \mathbb { R } ^ { n }$ , we use $\| x \|$ and $\| { \boldsymbol x } \| _ { \infty }$ to represent the Euclidean norm and infinity norm of x, respectively. Given two sets $X \subseteq \mathbb { R } ^ { n }$ and $Y \subseteq \mathbb { R } ^ { n }$ , their Pontryagin difference is denoted by $X \ominus Y = \{ z \in \mathbb { R } ^ { n } : z + y \in X ; \forall y \in Y \}$ , and their Minkowski sum is $X \oplus Y = \{ x + y | x \in X , y \in Y \} . { \mathrm { ~ A } }$ continuous and strictly increasing function $\alpha : \mathbb { R } _ { \geq 0 }  \mathbb { R } _ { \geq 0 }$ is called a K-function if $\alpha ( 0 ) = 0$ and $\alpha ( x ) > 0$ for all $x > 0 .$ A function $\alpha : \mathbb { R } _ { \geq 0 }  \mathbb { R } _ { \geq 0 }$ is called a $\kappa _ { \infty }$ -function if it is a K-function and $\alpha ( x )  \infty { \mathrm { ~ a s ~ } } x  \infty$

## II. PRELIMINARIES AND PROBLEM FORMULATION

Consider a discrete-time nonlinear system subject to parametric uncertainties and additive disturbances

$$
x _ {t + 1} = \mathcal {F} (x _ {t}, u _ {t}, v _ {t}, d _ {t}) \triangleq f (x _ {t}, u _ {t}) + g (x _ {t}, u _ {t}) v _ {t} + d _ {t},\tag{1}
$$

where $x _ { t } ~ \in ~ \mathbb { R } ^ { n _ { x } } , u _ { t } ~ \in ~ \mathbb { R } ^ { n _ { u } } , v _ { t } ~ \in ~ \mathbb { R } ^ { n _ { v } }$ and $d _ { t } ~ \in ~ \mathbb { R } ^ { n _ { x } }$ are the system state, the control input, the time-varying parametric uncertainty and the additive disturbance, respectively. $f : \mathbb { R } ^ { n _ { x } } \times \mathbb { R } ^ { n _ { u } } \to \mathbb { R } ^ { n _ { x } }$ and $g ~ : ~ \mathbb { R } ^ { n _ { x } } \times \mathbb { R } ^ { n _ { u } } \to \mathbb { R } ^ { n _ { x } \times n _ { v } }$ are known nonlinear functions satisfying $f ( 0 , 0 ) ~ = ~ 0$ and $g ( 0 , 0 ) = 0 .$ . It is assumed that $x _ { t } \ \in \ \mathcal { X } , u _ { t } \ \in \ \mathcal { U } , v _ { t } \ \in \ \mathcal { V }$ and $d _ { t } ~ \in ~ \mathcal { D }$ , where X, U are compact sets and V and D are compact and convex polytopes. We also assume that $\mathcal { X } , \mathcal { V } , \mathcal { D } , \mathcal { U }$ contain the origin.

Before presenting the main results, we firstly recall some well-established definitions used in this paper.

Definition 1 (RPI set $I 2 I J .$ Consider a discrete-time uncertain system $x _ { t + 1 } = \mathcal { G } ( x _ { t } , w _ { t } )$ , where $w _ { t } \in \mathcal W$ denotes the model uncertainty and $x _ { t }$ is the system state. A set Ω is a robust positively invariant (RPI) set for the system $x _ { t + 1 } ~ = ~ \mathcal { G } ( x _ { t } , w _ { t } )$ if ${ \mathcal G } ( x _ { t } , w _ { t } ) ~ \in ~ \Omega$ for all $x _ { t } ~ \in ~ \Omega$ and $w _ { t } \in \mathcal W$

Definition 2 (Zonotope of order $n \times m ~ [ 2 2 ] ) \colon$ Given $p \in \mathbb { R } ^ { n }$ and $H \in \mathbb { R } ^ { n \times m }$ , a zonotope of order $n \times m$ is a set of ndimensional vectors defined by ${ \mathcal { Z } } = p \oplus H \mathbb { B } ^ { m } = \{ p + H s :$ $s \in \mathbb { B } ^ { m } \}$

Definition 3 (ISpS-Lyapunov function $I 2 I J .$ Consider a nonlinear system in (1). A function $V ( \cdot ) \ : \ \mathbb { R } ^ { n _ { x } } \ \to \ \mathbb { R } _ { + }$ is called ISpS-Lyapunov function if there exist K-functions $\alpha _ { 1 } ( \cdot ) , \alpha _ { 2 } ( \cdot ) , \alpha _ { 3 } ( \cdot )$ , constants $\gamma _ { 1 } , \gamma _ { 2 }$ and a K-function $\rho ( \cdot )$ such that

$$
\alpha_ {1} (\| x \|) \leq V (x) \leq \alpha_ {2} (\| x \|) + \gamma_ {1},\tag{2a}
$$

$$
V (\mathcal {G} (x, v, d)) - V (x) \leq - \alpha_ {3} (\| x \|) + \rho (\| d \|) + \gamma_ {2}.\tag{2b}
$$

In the standard periodic MPC framework, a sequence of optimal control actions is obtained by solving the optimization problem at each time instant. Then the first element in this sequence will be sent to the actuator through the communication network and be implemented to the plant. In order to reduce communication load, a self-triggered adaptive control is proposed: The unknown parameters are identified at sampling time instants. Then based on the new estimation of unknown parameters, the optimization problem is reformulated and solved to determine the control input and the triggering interval. Let $t _ { k }$ denote the triggering time instant when the optimization problem needs to be solved. We consider the control policy and the set-based parameter estimator in the form of

$$
\begin{array}{l} u _ {t} = \tau (x _ {t _ {k}}, t - t _ {k}), t \in \mathbb {N} _ {[ t _ {k}, t _ {k + 1} - 1 ]} \\ \hat {\mathcal {V}} _ {t _ {k + 1}} = \Psi (\hat {\mathcal {V}} _ {t _ {k}}, x _ {t _ {k}}, x _ {t _ {k + 1}}, \mathbf {u} _ {[ t _ {k}, t _ {k + 1} - 1 ]}, \mathcal {D}) \end{array}\tag{3}
$$

where $\hat { \mathcal { V } } _ { t _ { k } }$ is the EFSS of unknown parameters at time instant $t _ { k }$ with $\ddot { \mathcal { V } } _ { 0 } = \mathcal { V } ; \mathbf { u } _ { [ t _ { k } , t _ { k + 1 } - 1 ] } = \{ u _ { t _ { k } } , u _ { t _ { k } + 1 } , \cdot \cdot \cdot , u _ { t _ { k + 1 } - 1 } \} . t _ { k }$ is obtained by using the following self-triggering scheduler

$$
t _ {k + 1} = t _ {k} + H ^ {*} (x _ {t _ {k}}, \hat {\mathcal {V}} _ {t _ {k}}), k \in \mathbb {N} _ {> 0},\tag{4}
$$

with $t _ { 0 } ~ = ~ 0$ . Our objective is to design the control policy $\tau ~ : ~ \mathbb { R } ^ { n _ { x } } \times \mathbb { N } \to \mathbb { R } ^ { n _ { u } }$ , the set-valued mapping $\Psi :$ $\mathbb { R } ^ { n _ { v } } \times \mathbb { R } ^ { n _ { x } } \times \mathbb { R } ^ { n _ { x } } \times \mathbb { R } ^ { n _ { x } } \times \mathbb { R } ^ { ( t _ { k + 1 } - 1 - t _ { k } ) n _ { u } } \times \mathbb { R } ^ { n _ { d } } \Longrightarrow \mathbb { R } ^ { n _ { v } }$ and the scheduling function $H ^ { * } : \mathbb { R } ^ { n _ { x } } \times \mathbb { R } ^ { n _ { v } } \to \mathbb { N } _ { > 0 }$ such that: 1) The proposed set-membership parameter estimator provides a tight overestimation of the FSS; 2) the online parameter adaption is facilitated in the co-design of control inputs and triggering intervals to further reduce the average sampling frequency; 3) the system in (1) is robustly stabilized with guaranteed robust constraint satisfaction and less conservative performance under recursive updates of the system model.

## III. PARAMETER ESTIMATION

Define $y _ { t } = x _ { t } - f ( x _ { t - 1 } , u _ { t - 1 } )$ and $\phi _ { t - 1 } = g ( x _ { t - 1 } , u _ { t - 1 } )$ According to (1), we have the following regression model

$$
y _ {t} = \phi_ {t - 1} v _ {t - 1} + d _ {t - 1}.\tag{5}
$$

Before presenting the parameter estimation algorithm, we firstly introduce the definition of the information set.

Definition 4 (Information set): A set $\mathcal { L } _ { t }$ is called the information set at time t if it is consistent with the system in (5), the measurements $y _ { t } ,$ , the regressor $\phi _ { t - 1 }$ and the uncertainty set ${ \mathcal { D } } ,$ , namely: $\mathcal { L } _ { t } = \left\{ v _ { t } \in \mathbb { R } ^ { n _ { v } } : y _ { t } - \phi _ { t - 1 } v _ { t } \in \mathcal { D } \right\}$

Consider the dynamic evolution of model uncertainty $v _ { t }$ with a general form

$$
v _ {t + 1} = \eta (v _ {t}, \delta_ {t}),\tag{6}
$$

where $\delta _ { t } \in \mathcal { M } \subseteq \mathbb { R } ^ { n _ { \delta } }$ is an auxiliary variable and $\eta : \mathbb { R } ^ { n _ { v } } \times$ $\mathbb { R } ^ { n _ { \delta } } \to \mathbb { R } ^ { n _ { v } }$ is a nonlinear function. Let $\nu _ { t }$ denote the FSS of the uncertain parameter $v _ { t }$ at time t. Given the information set $\mathcal { L } _ { t + 1 }$ , based on the system in (6), we have

$$
\mathcal {V} _ {t + 1} = \eta (\mathcal {V} _ {t}, \mathcal {M}) \cap \mathcal {L} _ {t + 1}.\tag{7}
$$

It can be seen from (7) that the parameter estimation problem consists of two parts: 1) The set computation associated with the dynamic evolution, and 2) the intersection of the information set and the set obtained from the dynamic evolution.

Remark 1: The proposed method is derived based on the model of parametric uncertainty in (6). Its main benefit is to reduce the overestimation by effectively using the model information. But in many practical control problems, it may be difficult to find an exact model to describe $v _ { t }$ . Alternatively, we consider the auxiliary variable $\delta _ { t }$ in (6) to establish the uncertain model for $v _ { t }$ . For example, if $v _ { t }$ in (1) is subject to a bounded change rate $\bar { \delta } > 0 .$ i.e., $\lVert v _ { t + 1 } - v _ { t } \rVert \leq \bar { \delta } .$ , we can find the following uncertain linear system $v _ { t + 1 } = \eta ( v _ { t } , \delta _ { t } ) \triangleq$ $\boldsymbol { v } _ { t } + \boldsymbol { \delta } _ { t }$ with $\delta _ { t } \in \mathcal { M } = \{ \delta \in \mathbb { R } ^ { n _ { v } } : \| \delta \| \leq \bar { \delta } \}$ . In addition, the set M can also be time-varying, which can be handled directly via replacing M by $\mathcal { M } _ { t }$ at sampling time instants.

As shown in (7), the key issue of the parameter estimation problem is how to calculate the set-based dynamic evolution with less overestimation. In this section, we start by recalling the preliminary results on computing $\eta ( \mathcal { V } _ { t } , \mathcal { M } )$ based on the indirect polytopic set computation [22]. Then a zonotopebased set-membership parameter estimator is presented for the nonlinear system in (1) under the aperiodic sampling induced by the self-triggering scheme.

## A. Indirect polytopic set computation

Let $\hat { \mathcal { V } } _ { t }$ denote the EFSS for $v _ { t }$ at time t. If $\hat { \mathcal { V } } _ { t }$ is a zonotope with $\hat { \mathcal { V } } _ { t } = p _ { t } \oplus H _ { t } \mathbb { B } ^ { n _ { v _ { t } } }$ , the set-based dynamic evolution of the system in (5) can be computed by using the centered inclusion function [23, Theorem 2],

$$
\eta_ {c} (\hat {\mathcal {V}} _ {t}, \mathcal {M}) = \eta (p _ {t}, \mathcal {M}) + \nabla_ {v} \eta (\bar {\mathcal {V}} _ {t}, \mathcal {M}) (\hat {\mathcal {V}} _ {t} \ominus p _ {t}),\tag{8}
$$

where $\bar { \mathcal { V } } _ { t }$ is a box bounding the set $\hat { \mathcal { V } } _ { t }$ . Assume that $\eta ( p _ { t } , \mathcal { M } )$ is bounded by a zonotope $\check { p } _ { t } \oplus \check { H } _ { t } \mathbb { B } ^ { n _ { v _ { t } } }$ , then we have

$$
\eta_ {c} (\hat {\mathcal {V}} _ {t}, \mathcal {M}) \subseteq \check {p} _ {t} \oplus \check {H} _ {t} \mathbb {B} ^ {n _ {\check {v} _ {t}}} \oplus \mathbb {M} _ {t} \mathbb {B} ^ {n _ {v _ {t}}}
$$

where $\hat { \mathbb M } _ { t } = \nabla _ { v } \eta ( \bar { \mathcal { V } } _ { t } , \mathcal { M } ) H _ { t }$ . By using the zonotope inclusion operator shown in [23, Theorem 3], we can further find a zonotope that bounds $\eta _ { c } ( \nu _ { t } , \mathcal { M } )$

Remark 2: As proposed in [23], the zonotope bounding the function $\eta ( p _ { t } , \mathcal { M } )$ can be found by using a natural interval extension or a mean value extension. Furthermore, if M is a zonotope with $\mathcal { M } ~ = ~ p _ { \delta } \oplus H _ { \delta } \mathbb { B } ^ { n _ { \delta } }$ , and $\eta ( v _ { t } , \delta _ { t } )$ is an affine function of the disturbance $\delta _ { t } , \ \mathrm { i . e . , } \ \eta ( v _ { t } , \delta _ { t } ) \ =$ $\eta _ { 1 } ( v _ { t } ) + \eta _ { 2 } ( v _ { t } ) \delta _ { t }$ , we can find a zonotope $p _ { a s } \oplus H _ { a s } \mathbb { B } ^ { n _ { \delta } }$ such that $\eta ( v _ { t } , \mathcal { M } ) = p _ { a s } \oplus H _ { a s } \mathbb { B } ^ { n _ { \delta } }$ where $p _ { a s } = \eta _ { 1 } ( p _ { t } ) + \eta _ { 2 } ( p _ { t } ) p _ { \delta }$ and $H _ { a s } = \eta _ { 2 } ( p _ { t } ) H _ { \delta }$

Although the set obtained from the center inclusion function (8) is a zonotope, performing the intersection operation in (7) may result in a polytope, rendering the center inclusion function (8) infeasible at next time instant. Hence, in the conventional zonotope-based set-membership methods, $\mathrm { e . g . }$ [23], [24], a zonotopic bounding process is imposed on the set resulted from the intersection. But this bounding process may lead to unnecessary overestimation. To avoid this, we recall the novel indirect polytopic set computation technique [22] in the following.

Suppose that $\hat { \mathcal { V } }$ is a polytope and $\hat { \mathcal { V } } = \mathcal { Z } _ { 1 } \cap \mathcal { Z } _ { 2 }$ , where $\mathcal { Z } _ { 1 }$ and $\mathcal { Z } _ { 2 }$ are two polytopes. According to set theory, it can be derived that $\eta ( { \hat { \mathcal { V } } } , { \mathcal { M } } ) ~ = ~ \eta ( { \mathcal { Z } } _ { 1 } \cap { \mathcal { Z } } _ { 2 } , { \mathcal { M } } ) ~ \subseteq$ $\eta ( \mathcal { Z } _ { 1 } , \mathcal { M } ) \cap \eta ( \mathcal { Z } _ { 2 } , \mathcal { M } )$ . If $\mathcal { Z } _ { 1 }$ and $\mathcal { Z } _ { 2 }$ are zonotopic, $\eta ( \mathcal { Z } _ { 1 } , \mathcal { M } )$

and $\eta ( \mathcal { Z } _ { 2 } , \mathcal { M } )$ can be computed by using the center inclusion function (8) directly without the bounding process. Therefore, the essence of indirect polytopic set computation is to find a group of zonotopes whose intersection is the polytope. It is worth mentioning that the results in [22] consider a 2- dimensional case; the following lemma extends this useful technique to accommodate general n-dimensional cases.

Lemma 1: Let $\mathcal { X } ~ \in ~ \mathbb { R } ^ { n }$ denote a convex and compact polytope. Suppose that X consists of m half-spaces. Then the polytope X can be exactly represented by the intersection of at most m zonotopic sets.

Proof: Let $h _ { i } , i \in \mathbb { N } _ { [ 1 , m ] }$ denote the ith half-space in X. By using $h _ { i }$ as an edge, a parallelotope or a zonotope $\mathcal { H } _ { i }$ can be constructed such that $\mathcal { X } \subseteq \mathcal { H } _ { i }$ . Therefore, we can find a sequence of zonotopes $\{ \mathcal { H } _ { i } \} _ { i \in \mathbb { N } _ { [ 1 , m ] } }$ whose intersection is $\mathcal { X } .$ . In addition, if the polytope is symmetric, there exist $i , j \in \mathbb { N } _ { [ 1 , m ] }$ such that $\mathcal { X } \subseteq \mathcal { H } _ { i } \subseteq \mathcal { H } _ { j }$ . Consequently, at most m zonotopes are needed to construct the polytope. ■

Remark 3: For each $h _ { i } ,$ we can find a point $\bar { x } _ { i } ~ \in ~ \mathcal X$ by solving a linear programming problem such that distance between $x _ { i }$ and $h _ { i }$ is maximized. According to $x _ { i }$ , we can find a half-space $\bar { h } _ { i }$ that is parallel to $h _ { i } .$ Suppose that $\mathcal { X } \triangleq \mathsf { c o n v } \big ( \{ \hat { x } _ { l } \} _ { l \in \mathbb { N } _ { [ 0 , \bar { n } ] } } \big )$ , where $\mathsf { c o n v } ( \cdot )$ is the convex hull of a set of points; xˆ denotes the lth vertex of $x ;$ n¯ is the amount of vertexes in X. Let $\mathcal { N } _ { i } = \{ l | i \in \mathbb { N } _ { [ 1 , m ] } , l \in \mathbb { N } _ { [ 0 , \bar { n } ] } \}$ denote the set of indices of vertex located at $h _ { i } ,$ and $\tilde { x } _ { l }$ is the projection of $\hat { x } _ { l }$ onto $\bar { h } _ { i }$ with $l \in \mathcal N _ { i }$ . Then the zonotope $\mathcal { H } _ { i }$ can be constructed as $\mathcal { H } _ { i } = \mathrm { c o n v } ( \{ \hat { x } _ { l } , \tilde { x } _ { l } \} _ { l \in \mathcal { N } _ { i } } )$

B. Set-membership parameter estimation under the selftriggering mechanism

Without loss of generality, it is assumed that the triggering interval computed at time $t _ { k }$ is $H _ { t _ { k } } ^ { * }$ . If $H _ { t _ { k } } ^ { * } = 1$ , according to Definition $^ { 4 , }$ , the information set $\mathcal { L } _ { t _ { k } }$ can be calculated as follows

$$
\begin{array}{c} \mathcal {L} _ {t _ {k}} = \{v _ {t _ {k}} \in \mathbb {R} ^ {n _ {v}}: x _ {t _ {k}} - f (x _ {t _ {k} - 1}, u _ {t _ {k} - 1}) \\ \qquad \qquad \qquad \qquad \qquad \qquad - g (x _ {t _ {k} - 1}, u _ {t _ {k} - 1}) v _ {t _ {k}} \in \mathcal {D} \}. \end{array}\tag{9}
$$

However, under the self-triggering scheduler (4), the system states at two consecutive time instants become inaccessible if $H _ { t _ { k } } ^ { * } > 1$ , making the calculation of $\mathcal { L } _ { t _ { k } }$ in (9) infeasible. To solve this problem, we propose to 1) estimate the reachable sets of system states between two sampling instants, and 2) approximate the information sets at triggering time instants by using the estimated reachable set instead of the measured state.

Suppose that $H _ { t _ { k } } ^ { * } ~ > ~ 1$ . Since all of the system states are accessible at triggering time instants, we define $X _ { t _ { k } } \ { \stackrel { \Delta } { = } }$ $\{ x _ { t _ { k } } \} ~ = ~ p _ { t _ { k } } ^ { x } \oplus O _ { t _ { k } } ^ { x } \mathbb { B } ^ { n _ { t _ { k } } ^ { x } }$ with $p _ { t _ { k } } ^ { x } ~ = ~ x _ { t _ { k } } , n _ { t _ { k } } ^ { x } ~ = ~ 0$ and $O _ { t _ { k } } ^ { x } \ = \ \mathbf { 0 }$ . Let $\hat { X } _ { l + t _ { k } } \triangleq p _ { l + t _ { k } } ^ { x } \oplus O _ { l + t _ { k } } ^ { x } \mathbb { B } ^ { n _ { l + t _ { k } } ^ { x } }$ denote the estimated reachable set for the system state $x _ { l + t _ { k } }$ from $X _ { t _ { k } } ,$ where $l \in \mathbb { N } _ { [ 0 , H _ { t _ { k } } ^ { * } - 1 ] }$ and $\hat { X } _ { t _ { k } } ~ = ~ X _ { t _ { k } }$ . Then by using the zonotopic set computation method in Section III-A, $\hat { X } _ { l + 1 + t _ { k } }$ can be estimated as follows:

$$
\hat {X} _ {l + 1 + t _ {k}} = \mathcal {F} _ {c} (\hat {X} _ {l + t _ {k}}, u _ {l + t _ {k}}, \hat {\mathcal {V}} _ {l + t _ {k}}, \mathcal {D})\tag{10}
$$

$$
\begin{array}{l} \text {where the function} \mathcal {F} _ {c} (\hat {X} _ {l + t _ {k}}, u _ {l + t _ {k}}, \hat {\mathcal {V}} _ {l + t _ {k}}, \mathcal {D}) \\ \nabla_ {x} \mathcal {F} (\bar {X} _ {l + t _ {k}}, u _ {l + t _ {k}}, \hat {\mathcal {V}} _ {l + t _ {k}}, \mathcal {D}) (\hat {X} _ {l + t _ {k}} - p _ {l + t _ {k}} ^ {x}) \end{array}
$$

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1 Zonotope-based set-membership parameter estimation algorithm

Require: Measured system states  $x_{t_{k-1}}$  and  $x_{t_k}$ ; sampling time instants  $t_{k-1}$  and  $t_k$ ; control input sequence  $\{u_i\}, i \in N_{[t_{k-1}, t_{k-1}]}$ ; EFSS  $\hat{V}_{t_{k-1}}$ ; uncertainty set D.

1: Find a sequence of zonotopes  $\{\check{V}_{i, t_{k-1}}\}$  such that  $\hat{V}_{t_{k-1}} = \cap_{i=1}^{n_{t_{k-1}}} \{\check{V}_{i, t_{k-1}}\}$ ; Set  $\hat{X}_{t_{k-1}} = \{x_{t_{k-1}}\}$ .

2: if  $t_k - t_{k-1} &gt; 1$  then

3: for  $j = t_{k-1}, t_{k-1} + 1, \cdots, t_k - 2$  do

4: Estimate  $\hat{X}_{j+1}$  and  $\hat{V}_{j+1}$  by using (10) and (11), respectively.

5: end for

6: Compute  $L_{t_k}$  by using (12), then  $\hat{V}_{t_k}$  can be obtained by following (13).

7: else

8: Calculate the information set  $L_{t_k}$  by using (9), and then compute the new EFSS  $\hat{V}_{t_k} = \eta_c(\check{V}_{1, t_{k-1}}, M) \cap \eta_c(\check{V}_{2, t_{k-1}}, M) \cap \cdots \cap \eta_c(\check{V}_{n_{t_{k-1}}, t_{k-1}}, M) \cap L_{t_k}$ .

9: end if
</div>

$\mathcal { F } ( p _ { l + t _ { k } } ^ { x } , u _ { l + t _ { k } } ^ { * } , \hat { \mathcal { V } } _ { l + t _ { k } } , \mathcal { D } )$ is the centered inclusion function derived in [23, Theorem $3 ] ; { \bar { X } } _ { l + t _ { k } }$ is a box bounding $\hat { X } _ { l + t _ { k } }$ Since $x _ { t _ { k } + 1 }$ is inaccessible, we consider the relatively conservative estimation of $\gamma _ { t _ { k } + l }$ in (10) by ignoring the intersecting operation

$$
\hat {\mathcal {V}} _ {t _ {k} + l + 1} = (\cap_ {i = 0} ^ {n _ {l}} \eta_ {c} (\check {\mathcal {V}} _ {i, l}, \mathcal {M})) \cap \mathcal {V}\tag{11}
$$

where $i \in \mathbb { N } _ { [ 1 , n _ { t _ { k } } ] } , \ l \ \in \ \mathbb { N } _ { [ 0 , H _ { t _ { k } } ^ { * } - 2 ] } ,$ and $\{ \check { \mathcal { V } } _ { i , l + t _ { k } } \}$ is the sequence of zonotopes satisfying $\hat { \mathcal { V } } _ { l + t _ { k } } = \cap \{ \check { \mathcal { V } } _ { i , l + t _ { k } } \}$ . Consequently, based on the system in (1), the information set at time $t _ { k + 1 }$ can be approximated by

$$
\begin{array}{c} \hat {\mathcal {L}} _ {t _ {k + 1}} = \{v _ {t _ {k + 1}} \in \mathbb {R} ^ {n _ {v}}: x _ {t _ {k + 1}} - f (x, u _ {t _ {k + 1} - 1}) \\ \qquad - g (x, u _ {t _ {k + 1} - 1}) v _ {t _ {k + 1}} \in \mathcal {D}, x \in \hat {X} _ {t _ {k + 1} - 1} \}. \end{array}\tag{12}
$$

Therefore, we have

$$
\hat {\mathcal {V}} _ {t _ {k + 1}} = (\cap_ {i = 0} ^ {n _ {l}} \eta_ {c} (\check {\mathcal {V}} _ {i, t _ {k + 1} - 1}, \mathcal {M})) \cap \hat {\mathcal {L}} _ {t _ {k + 1}} \cap \mathcal {V}.\tag{13}
$$

In summary, the procedure for updating the EFSS $\hat { \mathcal { V } } _ { t + 1 }$ <sub>1</sub> is presented in Algorithm 1.

## IV. SELF-TRIGGERED ADAPTIVE MIN-MAX MPC

In this section, based on the set-membership parameter estimation scheme described in Section III, we firstly present the min-max MPC optimization problem. Thereafter, a selftriggering scheduler accommodating the estimation of uncertainty is proposed, followed by a summary of the proposed self-triggered adaptive min-max MPC algorithm. Finally, this section concludes with an analysis of closed-loop stability and recursive feasibility.

## A. Min-max optimization

Given the prediction horizon $N \in  { \mathbb { N } } _ { \geq 0 }$ and an integer $H \in$ $\mathbb { N } _ { [ 1 , N ] }$ , we formulate the following cost function for the MPC problem at time $t _ { k }$

$$
J _ {N} ^ {H} (x _ {t _ {k}}, \mathbf {u} _ {t _ {k}, N}, \mathbf {v} _ {t _ {k}, N}, \mathbf {d} _ {t _ {k}, N}, \beta_ {t _ {k}}) \triangleq \sum_ {l = 0} ^ {H - 1} \frac {1}{\beta_ {t _ {k}}} \ell (x _ {l | t _ {k}}, u _ {l | t _ {k}})
$$

$$
+ \sum_ {l = H} ^ {N - 1} \ell (x _ {l | t _ {k}}, u _ {l | t _ {k}}) + \ell_ {f} (x _ {N | t _ {k}}),
$$

where $\beta _ { t _ { k } } ~ \in ~ \mathbb { R } _ { \geq 1 }$ is a scalar to be designed, $\boldsymbol { x } _ { l \left. t _ { k } \right. }$ is the predicted system state l steps ahead from the time $t _ { k }$ with $l \in \mathbb { N } _ { [ 0 , N ] } . \ \mathbf { u } _ { t _ { k } , N }$ denotes the control sequence and $\mathbf { u } _ { t _ { k } , N } =$ $\{ u _ { 0 | t _ { k } } , u _ { 1 | t _ { k } } , \cdot \cdot \cdot , u _ { N - 1 | t _ { k } } \} . \mathbf { v } _ { t _ { k } , N } = \{ v _ { 1 | t _ { k } } , v _ { 2 | t _ { k } } , \cdot \cdot \cdot , v _ { N | t _ { k } } \}$ and ${ \bf d } _ { t _ { k } , N } = \{ d _ { 1 | t _ { k } } , d _ { 2 | t _ { k } } , \cdot \cdot \cdot , d _ { N | t _ { k } } \}$ are sequences of multiplicative and additive disturbances, respectively. We suppose that the stage cost function $\ell : \mathbb { R } ^ { n _ { x } } \times \mathbb { R } ^ { n _ { u } } \to \mathbb { R } _ { > 0 }$ and the terminal cost function $\ell _ { f } : \mathbb { R } ^ { n _ { x } } \to \mathbb { R } _ { \geq 0 }$ are continuous functions satisfying $\ell ( 0 , 0 ) = 0$ and $\ell _ { f } ( 0 ) = 0$

In order to guarantee robust constraint satisfaction, we propose a self-triggered adaptive min-max MPC approach. Let ${ \mathbf { u } } _ { t _ { k } , N }$ denote the decision variable for the MPC optimization problem, then the proposed adaptive MPC algorithm is based on the following min-max optimization problem

$$
V _ {N} ^ {H} (x _ {t _ {k}}, \beta_ {t _ {k}}) = \min _ {u _ {l | t _ {k}} \in \mathcal {U}, l \in \mathbb {N} _ {[ 0, H - 1 ]}} \big \{\max _ {v _ {l | t _ {k}} \in \hat {\mathcal {V}} _ {l | t _ {k}}, d _ {l | t _ {k}} \in \mathcal {D}}
$$

$$
\{\sum_ {l = 0} ^ {H - 1} \frac {1}{\beta_ {t _ {k}}} \ell (x _ {l | t _ {k}}, u _ {l | t _ {k}}) + V _ {N - H} (x _ {H | t _ {k}}) \} \text {   such   that }
$$

$$
x _ {H \mid t _ {k}} \in \mathcal {X} _ {N - H}, \forall v _ {l \mid t _ {k}} \in \hat {\mathcal {V}} _ {l \mid t _ {k}}, d _ {l \mid t _ {k}} \in \mathcal {D}, l \in \mathbb {N} _ {[ 0, H - 1 ]} \},\tag{14a}
$$

$$
\text { s.t. } x _ {l + 1 | t _ {k}} = \mathcal {F} (x _ {l | t _ {k}}, u _ {l | t _ {k}}, v _ {l | t _ {k}}, d _ {l | t _ {k}}), l \in \mathbb {N} _ {[ 0, H - 1 ]},
$$

$$
x _ {H | t _ {k}} \in \mathcal {X} _ {N - H},\tag{14b}
$$

(14c)

$$
x _ {0 | t _ {k}} = x _ {t _ {k}}, x _ {l | t _ {k}} \in \mathcal {X}, l \in \mathbb {N} _ {[ 0, H - 1 ]},\tag{14d}
$$

where

$$
\begin{array}{l} V _ {N - i} (x _ {i | t _ {k}}) = \min _ {u _ {i | t _ {k}} \in \mathcal {U}} \{\max _ {v _ {i | t _ {k}} \in \hat {\mathcal {V}} _ {i | t _ {k}}, d _ {i | t _ {k}} \in \mathcal {D}} \{\ell (x _ {i | t _ {k}}, u _ {i | t _ {k}}) \\ + V _ {N - i - 1} (\mathcal {F} (x _ {i | t _ {k}}, u _ {i | t _ {k}}, v _ {i | t _ {k}}, d _ {i | t _ {k}})) \} \text {   such   that } \\ \mathcal {F} (x _ {i | t _ {k}}, u _ {i | t _ {k}}, v _ {i | t _ {k}}, d _ {i | t _ {k}}) \in \mathcal {X} _ {N - i - 1}, \forall v _ {i | t _ {k}} \in \hat {\mathcal {V}} _ {i | t _ {k}}, \\ \forall d _ {i | t _ {k}} \in \mathcal {D} \}, \end{array} \tag {(1)}\tag{15}
$$

$i \in \mathbb { N } _ { [ H , N - 1 ] }$ and $\mathcal { X } _ { N - i }$ is the set of admissible states which can be robustly steered into the terminal set $\chi _ { f }$ within $N - i$ steps. $\hat { \mathcal { V } } _ { i \mid t _ { l } }$ is the predicted EFSS i step ahead from the time $t _ { k }$ with $\hat { \mathcal { V } } _ { 0 \mid t _ { k } } = \hat { \mathcal { V } } _ { t _ { k } }$ . In addition, since $\boldsymbol { v } _ { t } \in \mathcal { V }$ for all $t \geq 0$ $\hat { \mathcal { V } } _ { i \vert t _ { k } }$ can be calculated recursively by following

$$
\hat {\mathcal {V}} _ {i + 1 | t _ {k}} = (\cap_ {s = 0} ^ {n _ {l}} \eta_ {c} (\check {\mathcal {V}} _ {s, l | t _ {k}}, \mathcal {M})) \cap \mathcal {V}, \quad i \in \mathbb {N} _ {[ 0, N - 1 ]}\tag{16}
$$

where $\hat { \mathcal { V } } _ { i \mid t _ { k } } = \cap _ { s = 0 } ^ { n _ { l } } \{ \check { \mathcal { V } } _ { s , i \mid t _ { k } } \}$ . Furthermore, the initial conditions are specified as $V _ { 0 } ( x _ { N | t _ { k } } ) \triangleq \ell _ { f } ( x _ { N | t _ { k } } )$ and $\chi _ { 0 } \triangleq \chi _ { f }$ <sub>f</sub> , respectively.

As shown in the min-max optimization problem (14), $\beta _ { t _ { k } }$ determines the penalty on the cost for open loop scenarios. Hence, it is possible to achieve a less conservative closedloop performance by designing suitable $\beta _ { t _ { k } }$ . In order to further reduce the average sampling frequency, we propose the evolution of $\beta _ { t _ { k } }$ by implicitly considering the EFSS $\hat { \mathcal { V } } _ { t _ { I } }$ k in the following.

## B. Self-triggering scheduler

Under the self-triggering mechanism, the MPC problem is solved at the triggering time instant only. Therefore, between two consecutive triggering time instants, the system in (1) is operated with the open loop control actions determined by (14), i.e., $u _ { t } \ = \ \tau ( x _ { t _ { k } } , t - t _ { k } ) \ = \ u _ { t - t _ { k } | t _ { k } } ^ { * } , t \ \in \ \mathbb { N } _ { [ t _ { k } , t _ { k + 1 } - 1 ] }$ and $\{ u _ { t - t _ { k } | t _ { k } } ^ { * } \}$ denotes the optimal solution of the MPC optimization problem (14) obtained at time $t _ { k }$ . Inspired by [15], we design the following self-triggering scheduler

$$
\begin{array}{c} t _ {k + 1} = t _ {k} + H ^ {*} (x _ {t _ {k}}, \hat {\mathcal {V}} _ {t _ {k}}) \\ H ^ {*} (x _ {t _ {k}}, \hat {\mathcal {V}} _ {t _ {k}}) \triangleq \max \{H \in \mathbb {N} _ {[ 1, H _ {\max} ]} | \\ V _ {N} ^ {H} (x _ {t _ {k}}, \beta_ {t _ {k}}) \leq V _ {N} ^ {1} (x _ {t _ {k}}, \beta_ {t _ {k}}) \} \end{array}\tag{17}
$$

where $H _ { \operatorname* { m a x } } \in \mathbb { N } _ { [ 1 , N ] }$ denotes the maximum number of time instants allowed for the open loop scenario. Consequently, the system in (1) becomes

$$
x _ {t + 1} = \mathcal {F} (x _ {t}, \tau (x _ {t _ {k}}, t - t _ {k}), v _ {t}, d _ {t}).\tag{18}
$$

As shown in (17), the performance of the proposed selftriggered adaptive MPC algorithm depends on the parameter $\beta _ { t _ { k } }$ . Using a larger $\beta _ { t _ { k } }$ will result in a larger triggering interval $H ^ { * } ( x _ { t _ { k } } , \hat { \mathcal { V } } _ { t _ { k } } )$ , but will deteriorate the regulation performance [15]. Furthermore, the triggering interval is also determined by the bound of uncertainties: A larger bound of uncertainties renders a shorter triggering interval by following (17). To further reduce the average sampling frequency, we propose the following adaptive mechanism associated with $\hat { \mathcal { V } } _ { t _ { j } }$ k

$$
\beta_ {t _ {k}} = \min (\frac {\xi_ {0}}{\xi_ {t _ {k}}} \beta_ {0}, \beta_ {\max}),\tag{19}
$$

where $\begin{array} { r } { \xi _ { t _ { k } } \ = \ \operatorname* { s u p } _ { v _ { 1 } , v _ { 2 } \in \hat { \mathcal { V } } _ { t _ { k } } } \| v _ { 1 } - v _ { 2 } \| ; \ \beta _ { 0 } \ \in \ \mathbb { R } _ { \ge 1 } } \end{array}$ is a fixed constant, and $\beta _ { \mathrm { m a x } }$ is a positive number. Since $H ^ { * } ( x _ { t _ { k } } , \hat { \mathcal { V } } _ { t _ { k } } )$ is bounded by the prediction horizon N, increasing $\beta _ { t _ { k } }$ cannot enlarge the triggering intervals if $\beta _ { t _ { k } }$ is sufficiently large. Hence we manually bound $\beta _ { k }$ by $\beta _ { \mathrm { m a x } }$ for all $t _ { k } \geq 0$

Remark 4: To design a self-triggered adaptive MPC algorithm, the offline constructed RPI set $\chi _ { f }$ is commonly used to guarantee robust stability in the presence of additive and multiplicative uncertainties. At each triggering time instant, it is possible to improve the closed-loop behaviors by recomputing the terminal set $\chi _ { f }$ in (14) based on the new estimated uncertainty set. However, as remarked in [20], if the local control policy $\kappa _ { f }$ were not simultaneously updated based on the new EFSS, the closed-loop performance would not be significantly improved. On the other hand, updating both $\chi _ { f }$ and the feedback gain would render the MPC problem much more complicated. Consequently, we consider the fixed terminal set $\mathcal { X } _ { f }$ in our method, which is designed offline based on the initial uncertainty sets V and D.

According to the developed set-membership parameter estimator, the proposed self-triggered adaptive MPC scheme is summarized in Algorithm 2.

## C. Recursive feasibility and closed-loop stability

To ensure the closed-loop stability and recursive feasibility, we make the following standard assumptions.

Assumption 1: There exist a local controller $\kappa _ { f } : \mathbb R ^ { n _ { x } } $ R<sup>n</sup>u , a K-function α and some constants $\sigma _ { 1 } , \sigma _ { 2 }$ and $\sigma _ { 3 }$ such that

1) $\mathcal X _ { f } \subseteq \mathcal X$ is an RPI set containing the origin for the closedloop system $x _ { t + 1 } = \mathcal { F } ( x _ { t } , \kappa _ { f } ( x _ { t } ) , v _ { t } , d _ { t } )$

2) $\kappa _ { f } ( x _ { t } ) \subseteq \mathcal { U }$ for all $x \in \mathcal { X } _ { f }$

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 2 Self-triggered adaptive MPC with state and parameter estimation
Require: Initial system state  $x_{0}$ ; initial EFSS  $\hat{V}_{0}$ ; uncertainty set D; tuning parameters  $\beta_{0}$  and  $H_{max}$ .
1: Set t = 0,  $t_{k} = 0$  and k = 0.
2: while The control action is not stopped do
3: Measure the state  $x_{t_{k}}$  of the system in (1).
4: if  $t_{k} &gt; 0$  then
5: Compute the EFSS  $\hat{V}_{t_{k}}$  by following Algorithm 1.
6: end if
7: Reformulate and solve the optimization problems in (14) based on  $\hat{V}_{t_{k}}$  to obtain the optimal control sequence  $u_{t_{k},N}^{*}$  and determine the next sampling time instant  $t_{k+1} = t_{k} + H^{*}(t_{k})$  by using (14) and (17).
8: Implement the control input  $u_{t_{k}+i} = u_{i|t_{k}}^{*}$  to the system in (1) where  $i \in N_{[0,H^{*}(x_{t_{k}},\hat{V}_{t_{k}})-1]}$ .
9: Set  $t = t + H^{*}(x_{t_{k}}, \hat{V}_{t_{k}}), k = k + 1$  and  $t_{k} = t$ .
10: end while
</div>

3) $\ell ( x _ { t } , u _ { t } ) \geq \sigma _ { 1 } \| x _ { t } \|$ for all $x _ { t } \in \mathcal { X }$ and $u _ { t } \in \mathcal { U } .$

$$
\sigma_ {2} \| x _ {t} \| \leq \ell_ {f} (x _ {t}) \leq \sigma_ {3} \| x _ {t} \|
$$

$$
x _ {t} \in \mathcal {X} _ {f}
$$

$$
\ell_ {f} \left(\mathcal {F} \left(x _ {t}, \kappa_ {f} \left(x _ {t}\right), v _ {t}, d _ {t}\right)\right) - \ell_ {f} \left(x _ {t}\right) \leq - \ell \left(x _ {t}, \kappa_ {f} \left(x _ {t}\right)\right) +
$$

$$
\alpha (\| d _ {t} \|)
$$

$$
x _ {t} \in \mathcal {X} _ {f}, v _ {t} \in \mathcal {V}
$$

$$
d _ {t} \in \mathcal {D}.
$$

Theorem 1: For the nonlinear system in (1), suppose that Assumption 1 holds, then the proposed self-triggered adaptive MPC scheme, as presented in Algorithm 2 is recursively feasible for all $x _ { 0 } \in \mathcal { X } _ { N }$

Proof: Let $\hat { \mathcal { V } } _ { t _ { k } }$ denote the EFSSs obtained by Algorithm 2. At time $t _ { k } ,$ suppose that $\begin{array} { r l r } { x _ { t _ { k } } } & { { } \in } & { \mathcal { X } _ { N } } \end{array}$ and the next sampling time instant is $\begin{array} { r c l } { t _ { k + 1 } } & { = } & { t _ { k } \ + } \end{array}$ $H ^ { * } ( x _ { t _ { k } } , \hat { \mathcal { V } } _ { t _ { k } } )$ . For simplicity, we use the notation $H _ { t _ { k } } ^ { * }$ to denote $H ^ { * } ( x _ { t _ { k } } , \hat { \mathcal { V } } _ { t _ { k } } )$ in the following. Let $\begin{array} { r l } { \mathbf { u } _ { t _ { k } , N } ^ { * } } & { { } = } \end{array}$ $\{ u _ { 0 | t _ { k } } ^ { * } , u _ { 1 | t _ { k } } ^ { * } , \cdot \cdot \cdot , u _ { N - 1 | t _ { k } } ^ { * } \}$ denote the optimal solution of the optimization problem (14) at time $t _ { k }$ , then we construct the candidate input sequence at time $\begin{array} { r c l c r } { t _ { k + 1 } } & { : } & { \bar { \mathbf { u } } _ { t _ { k + 1 } , N } } & { = } \end{array}$ $\{ u _ { H ^ { * } ( t _ { k } ) | t _ { k } } ^ { * } , u _ { H ^ { * } ( t _ { k } ) + 1 | t _ { k } } ^ { * } , \cdot \cdot \cdot , u _ { N - 1 | t _ { k } } ^ { * } , \kappa _ { f } ( x _ { N | t _ { k } } ) , \cdot \cdot \cdot \cdot , u _ { N - 1 | t _ { k } } ^ { * } , \kappa _ { f } ( t _ { k } ) , \}$ $\kappa _ { f } \left( x _ { N + H ^ { * } \left( t _ { k } \right) - 1 \mid t _ { k } } \right) \big \}$ . It follows from (15) that each element in $\bar { \mathbf { u } } _ { t _ { k + 1 } , N }$ is a feedback control policy. Since $\chi _ { f }$ is an RPI set for the system in (1) and $\hat { \mathcal { V } } _ { i | t _ { k } } \subseteq \mathcal { V }$ for all $t _ { k } \in \mathbb { N } _ { \geq 0 }$ and $i \in \mathbb { N } _ { [ 0 , N ] }$ , X<sub>N</sub> is an invariant set for the system in (1) under the proposed self-triggered adaptive MPC algorithm for all $v _ { t } \in \mathcal V _ { t }$ and $d _ { t } ~ \in ~ \mathcal { D }$ . Hence the recursive feasibility is proved. ■

Remark 5: As shown in Theorem 1, we investigate the recursive feasibility of the proposed method at triggering time instants. Although the control inputs are executed in an open loop configuration between two triggering time instants, it still can be guaranteed that $x _ { t } ~ \in ~ \mathcal { X }$ and $u _ { t } \in \mathcal { U }$ for all $t \geq 0$ and $x _ { 0 } ~ \in ~ \mathcal { X } _ { N }$ by considering the worst-case realization of uncertainty in the MPC optimization problem.

Theorem 2: For the nonlinear system in (1), suppose that Assumption 1 holds and $x _ { 0 } ~ \in ~ \mathcal { X } _ { N }$ . If the triggering time instants $t _ { k }$ are determined by (17), then the closed-loop system in (18) is ISpS at the triggering time instants.

Proof: To prove ISpS of the closed-loop system, we need to show that the optimal cost function is an ISpS Lyapunov function. Next we will demonstrate the satisfaction of (2a) and (2b) for the optimal cost function $V _ { N } ^ { H _ { t _ { k } } ^ { * } } ( x _ { t _ { k } } , \beta _ { t _ { k } } )$

By applying Algorithm 2, it follows from (14) that

$$
\begin{array}{r l} & V _ {N} ^ {H _ {t _ {k}} ^ {*}} (x _ {t _ {k}}, \beta_ {t _ {k}}) = J _ {N} ^ {H} (x _ {t _ {k}}, \mathbf {u} _ {t _ {k}, N} ^ {*}, \mathbf {v} _ {t _ {k}, N} ^ {*}, \mathbf {d} _ {t _ {k}, N} ^ {*}, \beta_ {t _ {k}}) \\ & \qquad \geq J _ {N} ^ {H} (x _ {t _ {k}}, \mathbf {u} _ {t _ {k}, N} ^ {*}, \mathbf {0}, \mathbf {0}, \beta_ {t _ {k}}) \\ & \qquad \geq \min _ {\mathbf {u} _ {t _ {k}} ^ {*}} J _ {N} ^ {H} (x _ {t _ {k}}, \mathbf {u} _ {t _ {k}, N}, \mathbf {0}, \mathbf {0}, \beta_ {t _ {k}}) \\ & \qquad \geq \frac {\sigma_ {1}}{\beta_ {\max}} \| x _ {t _ {k}} \|. \end{array}
$$

Hence the lower bound of $V _ { N } ^ { H _ { t _ { k } } ^ { * } } ( x _ { t _ { k } } , \beta _ { t _ { k } } )$ is derived. Furthermore, let $\check { \mathbf { u } } _ { t _ { k } , N }$ denote the optimal solution associated with $V _ { N } ^ { 1 } ( x _ { t _ { k } } , \beta _ { t _ { k } } )$ , and define a control sequence $\tilde { \mathbf { u } } _ { t _ { k } , N + 1 }$ as $\tilde { \mathbf { u } } _ { t _ { k } , N + 1 } = \{ \check { \mathbf { u } } _ { t _ { k } , N } , \kappa _ { f } ( x _ { N | t _ { k } } ) \}$ . As such, we have

$$
\begin{array}{r l} & J _ {N + 1} ^ {1} (x _ {t _ {k}}, \tilde {\mathbf {u}} _ {t _ {k}, N + 1}, \mathbf {v} _ {t _ {k}, N + 1}, \mathbf {d} _ {t _ {k}, N + 1}, \beta_ {t _ {k}}) = \\ & \qquad J _ {N} ^ {1} (x _ {t _ {k}}, \breve {\mathbf {u}} _ {t _ {k}, N}, \mathbf {v} _ {t _ {k}, N}, \mathbf {d} _ {t _ {k}, N}, \beta_ {t _ {k}}) - \ell_ {f} (x _ {N | t _ {k}}) \\ & \qquad + \ell_ {f} (x _ {N + 1 | t _ {k}}) + \ell (x _ {N | t _ {k}}, \kappa_ {f} (x _ {N | t _ {k}}). \end{array}
$$

for all $\boldsymbol { x } _ { t _ { k } } \in \mathcal { X } _ { N }$ . Then it follows from Assumption 1 that

$$
\begin{array}{r l} & J _ {N + 1} ^ {1} (x _ {t _ {k}}, \tilde {\mathbf {u}} _ {t _ {k}, N + 1}, \mathbf {v} _ {t _ {k}, N + 1}, \mathbf {d} _ {t _ {k}, N + 1}, \beta_ {t _ {k}}) \\ & \quad \leq J _ {N} ^ {1} (x _ {t _ {k}}, \check {\mathbf {u}} _ {t _ {k}, N}, \mathbf {v} _ {t _ {k}, N}, \mathbf {d} _ {t _ {k}, N}, \beta_ {t _ {k}}) + \alpha (\| d _ {t _ {k}} \|). \end{array}
$$

According to the suboptimality of the sequence $\tilde { \mathbf { u } } _ { t _ { k } , N + 1 }$ , it is inferred that

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
$V_{N + 1}^{1}(x_{t_k},\beta_{t_k})$ $\leq \max_{\mathbf{d}_{t_k,N + 1},\mathbf{v}_{t_k,N + 1}}J_{N + 1}^1 (x_{t_k},\tilde{\mathbf{u}}_{t_k,N + 1},\mathbf{v}_{t_k,N + 1},\mathbf{d}_{t_k,N + 1},\beta_{t_k})$ $\leq \max_{\mathbf{d}_{t_k,N},\mathbf{v}_{t_k,N}}J_N^1 (x_{t_k},\check{\mathbf{u}}_{t_k,N},\mathbf{v}_{t_k,N},\mathbf{d}_{t_k,N},\beta_{t_k}) + \bar{d}$ $\leq V_N^1 (x_{t_k},\beta_{t_k}) + \bar{d},$ (20)
</div>

where $\bar { d } = \operatorname* { m a x } _ { d \in \mathcal { D } } \alpha ( \| d \| )$ . Consequently, for all $x _ { t _ { k } } \in \mathcal X _ { f }$ we have

$$
\begin{array}{r l} & V _ {N + 1} ^ {1} (x _ {t _ {k}}, \beta_ {t _ {k}}) \leq V _ {1} ^ {1} (x _ {t _ {k}}, \beta_ {t _ {k}}) + (N - 1) \max _ {d} \alpha (\| d \|) \\ & \quad \leq V _ {1} (x _ {t _ {k}}) + \frac {1 - \beta_ {t _ {k}}}{\beta_ {t _ {k}}} \ell (x _ {t _ {k}}, \kappa_ {f} (x _ {t _ {k}})) + \\ & \quad \quad (N - 1) \bar {d} \\ & \quad \leq \ell_ {f} (x _ {t _ {k}}) + \frac {1 - \beta_ {t _ {k}}}{\beta_ {t _ {k}}} \ell (x _ {t _ {k}}, \kappa_ {f} (x _ {t _ {k}})) + N \bar {d} \\ & \quad \leq \sigma_ {3} \| x _ {t _ {k}} \| + N \bar {d}. \end{array}
$$

Hence, by induction, it is derived from the triggering condition (17) that $V _ { N } ^ { H _ { t _ { k } } ^ { * } } ( x _ { t _ { k } } , \beta _ { t _ { k } } ) \leq \sigma _ { 3 } \| x _ { t _ { k } } \| + N \bar { d } , \forall x _ { t _ { k } } \in \mathcal { X } _ { f }$ . For $x _ { t _ { k } } ~ \in ~ \mathcal { X } _ { N }$ but $x _ { t _ { k } } \notin \mathcal { X } _ { f }$ , the upper bound can be found by following Lemma 1 in [21]. Let $B _ { r }$ denote a set and $B _ { r } = \{ x \in \mathbb { R } ^ { n _ { x } } | \| x \| \leq r \}$ . Since $x , u , \nu$ and D are compact sets, there definitely exists a finite constant $\bar { V } _ { N } \in \mathbb { R }$ such that $V _ { N } ^ { H _ { t _ { k } } ^ { * } } ( x _ { t _ { k } } , \beta _ { t _ { k } } ) \leq \bar { V } _ { N }$ for all $x _ { t _ { k } } \in \mathcal { X } _ { N }$ . For $x _ { t _ { k } } \in \mathcal { X } _ { N }$ but $x _ { t _ { k } } ~ \notin ~ \mathcal { B } _ { r }$ , we have $\| \boldsymbol { x } _ { t _ { k } } \| \geq r$ and $V _ { N } ^ { H _ { t _ { k } } ^ { * } } ( x _ { t _ { k } } , \beta _ { t _ { k } } ) \ \leq \ \bar { V } _ { N }$ which in turn leads to $\begin{array} { r } { V _ { N } ^ { H _ { t _ { k } } ^ { * } } ( x _ { t _ { k } } , \beta _ { t _ { k } } ) \ \leq \ \frac { \bar { V } _ { N } } { r } \| x _ { t _ { k } } \| } \end{array}$ . Consequently, we have $V _ { N } ^ { H _ { t _ { k } } ^ { * } } ( x _ { t _ { k } } , \beta _ { t _ { k } } ) _ { - } \leq \sigma _ { \Delta } \| x _ { t _ { k } } \| + N \bar { d }$ for all $x _ { t _ { k } } \in \mathcal X _ { n }$ , where $\begin{array} { r } { \sigma _ { \Delta } = \operatorname* { m a x } ( \sigma _ { 3 } , \frac { V _ { N } } { r } \| x _ { t _ { k } } \| ) } \end{array}$

According to the triggering condition (17) and $\nu _ { l + t _ { k } } ~ \subseteq$ $\hat { \mathcal { V } } _ { l \vert t _ { k } }$ , for all $\boldsymbol { x } _ { t _ { k } } \in \mathcal { X } _ { N }$ , we have

$$
V _ {N} ^ {H ^ {*} (x _ {t _ {k + 1}})} (x _ {k + 1}, \beta_ {t _ {k + 1}}) - V _ {N} ^ {H _ {t _ {k}} ^ {*}} (x _ {t _ {k}}, \beta_ {t _ {k}})
$$

$$
\leq V _ {N} ^ {1} (x _ {k + 1}, \beta_ {t _ {k + 1}}) - V _ {N} ^ {H _ {t _ {k}} ^ {*}} (x _ {t _ {k}}, \beta_ {t _ {k}})\tag{21a}
$$

$$
\leq V _ {N} ^ {1} (x _ {k + 1}, \beta_ {t _ {k + 1}}) - \max _ {v _ {l | t _ {k}} \in \hat {\mathcal {V}} _ {l | t _ {k}}, d _ {l | t _ {k}} \in \mathcal {D}, l \in \mathbb {N} _ {[ 0, H - 1 ]}} \{\sum_ {l = 0} ^ {H _ {t _ {k}} ^ {*} - 1}
$$

$$
\frac {1}{\beta_ {t _ {k}}} \ell (x _ {l | t _ {k}}, u _ {l | t _ {k}} ^ {*}) + V _ {N - H _ {t _ {k}} ^ {*}} (x _ {H | t _ {k}}) \}\tag{21b}
$$

$$
\leq V _ {N} ^ {1} (x _ {k + 1}, \beta_ {t _ {k + 1}}) - V _ {N - H _ {t _ {k}} ^ {*}} (x _ {t _ {k + 1}})
$$

$$
- \sum_ {l = 0} ^ {H _ {t _ {k}} ^ {*} - 1} \frac {1}{\beta_ {t _ {k}}} \ell (x _ {t _ {k} + l}, u _ {l | t _ {k}} ^ {*}).\tag{21c}
$$

It is similar to (20) that $V _ { N + 1 } ( x _ { t _ { k + 1 } } ) \ : - \ : V _ { N } ( x _ { t _ { k + 1 } } ) \ : \ : \leq \ :$ ${ \bar { d } } ,$ which yields $V _ { N } ( x _ { t _ { k + 1 } } ) - V _ { N - H _ { t _ { k } } ^ { * } } ( x _ { t _ { k + 1 } } ) \ \leq \ H _ { t _ { k } } ^ { * } \bar { d }$ for k $x _ { t _ { k + 1 } } \in \mathcal { X } _ { N - H _ { t _ { k } } ^ { * } }$ . Hence, we have $\hat { V } _ { N } ^ { H _ { t _ { k + 1 } } ^ { * } } ( x _ { k + 1 } , \beta _ { t _ { k + 1 } } ) \ -$ $\begin{array} { r l r } { V _ { N } ^ { H _ { t _ { k } } ^ { * } } ( x _ { t _ { k } } , \beta _ { t _ { k } } ) } & { \leq } & { H _ { t _ { k } } ^ { * } \bar { d } - \sum _ { l = 0 } ^ { H _ { t _ { k } } ^ { * } - 1 } \frac { 1 } { \beta _ { t _ { k } } } \ell ( x _ { t _ { k } + l } , u _ { l | t _ { k } } ^ { * } ) \leq } \end{array}$ $\begin{array} { r } { H _ { t _ { k } } ^ { * } \bar { d } - \sum _ { l = 0 } ^ { H _ { t _ { k } } ^ { * } - 1 } \frac { 1 } { \beta _ { t _ { k } } } \ell ( x _ { t _ { k } + l } , u _ { l | t _ { k } } ^ { * } ) } \end{array}$ . In summary, it is inferred that, by applying the proposed self-triggered adaptive MPC scheme presented in Algorithm 2 to the system in (1), the corresponding optimal value function $V _ { N } ^ { H _ { t _ { k } } ^ { * } } ( x _ { t _ { k } } , \beta _ { t _ { k } } )$ is an ISpS Lyapunov function. According to [21, Theorem 1], it is proved that the closed-loop system in (1) is ISpS under the proposed self-triggered adaptive MPC algorithms for all $x _ { 0 } \in \mathcal { X } _ { N }$ at triggering time instants.

Remark 6: Compared with the self-triggered robust method, the time-varying $\beta _ { k }$ is considered and updated with respect to $\hat { \mathcal { V } } _ { t _ { k } }$ in the proposed method. This strategy improves the effective use of the EFSS in MPC framework to reduce the conservatism. The design of $\beta _ { t _ { k } }$ in (19) is to further enlarge the triggering interval when a less conservative EFSS is obtained. For a better control performance, we can choose a smaller $\beta _ { k }$ if the size of $\hat { \mathcal { V } } _ { t _ { k } }$ becomes smaller. Therefore, a suitable evolution of $\beta _ { k }$ helps achieve a trade-off between the communicational load and performance.

## V. ILLUSTRATIVE EXAMPLE

In this section, a numerical example is presented to validate our theoretical results. Consider the following discrete-time cart and spring-damper system from [15], [25]

$$
\begin{array}{r l} & x _ {t + 1} (1) = x _ {t} (1) + T x _ {t} (2), \\ & x _ {t + 1} (2) = - \frac {l T}{m} e ^ {- x _ {t} (1)} + \frac {m - h T}{m} x _ {t} (2) + \frac {T}{m} u _ {t} \\ & \qquad - \frac {T}{m} v _ {t} x _ {t} (2) + \frac {T}{m} d _ {t}. \end{array}
$$

where $x _ { t } = [ x _ { t } ( 1 ) \ x _ { t } ( 2 ) ] ^ { \mathrm { T } }$ and $u _ { t }$ are the system state and input satisfying the constraints $| u _ { t } | \leq 4 . 5 ~ N$ and $| x _ { t } ( 1 ) | \leq$ $2 \ m$ . The additive and parametric uncertainties are limited by $| d _ { t } | ~ \le ~ 0 . 1 , | v _ { t } | ~ \le ~ 0 . 1 5$ and $\| v _ { t + 1 } - v _ { t } \| ~ \leq ~ 0 . 0 0 8$ More specifically, we consider the following sequence of uncertainties in this example: $v _ { t } ~ = ~ 0 . 1 \sin ( \frac { 4 t } { \pi } )$ and $d _ { t } \ =$

![](images/1969722bff872f52c572c0d98303f01121f8747491da25f0f64396d30ef6fc5d.jpg)  
Fig. 1. The time evolution of system states.

TABLE I  
CLOSED-LOOP PERFORMANCE COMPARISON.

<table><tr><td></td><td> $J_p$ </td><td>Average sampling time</td></tr><tr><td>Algorithm 2</td><td>13.4122</td><td>2.6333</td></tr><tr><td>ST-RMPC [15]</td><td>13.1229</td><td>1.6250</td></tr></table>

$0 . 1 5 \sin ( \frac { t } { \pi } )$ . The system parameters are given as follows: $m = 1 \ k g ; \ l = 0 . 3 3 \ N / m ; \ h = 1 . 1 \ N s / m ; \ T = 0 . 4 \ s .$ For the proposed self-triggered adaptive MPC scheme, we set the prediction horizon $N = 6$ . The stage cost function is chosen as $\ell ( x , u ) = x ^ { \mathrm { { T } } } Q x + u ^ { \mathrm { { T } } } R u$ with $Q = \mathrm { d i a g } ( 0 . 6 4 , 0 . 6 4 )$ and $R = 1$ . According to [15], [25], we set $\mathcal { X } _ { f } = \{ x : x ^ { \mathrm { T } } P x \leq$ 3.8} with $P = \Big | 4 . 5 6 7 8 \quad 3 . 2 0 1 8 \Big | . \ \ell _ { f } ( x ) = x ^ { \mathrm { T } } P x ; \kappa _ { f } ( x ) =$ $[ - 0 . 7 7 9 7 ~ - 1 . 1 \dot { 0 } 2 9 ] x$ . The feedback policy in (15) is set as $u ( x ) = a \kappa _ { f } ( x ) + b x ^ { \mathrm { T } } x + c$ where $a , b , c \in \mathbb { R }$ are the decision variables for the optimization problem in (15). To demonstrate the effectiveness of the proposed methods, the self-triggered robust MPC [15] (ST-RMPC) is also implemented with the same parameters for the purpose of comparison. For the selftriggering scheduler, we set $H _ { \mathrm { m a x } } = 5$ and $\beta _ { 0 } = 1 . 1$

Starting from the initial condition $x _ { 0 } ~ = ~ [ 1 ~ 1 ] ^ { \mathrm { T } }$ , the trajectories of system states are shown in Fig. 1, and the time evolution of control inputs is plotted in Fig. 2. The triggering time instants are reported in Fig. 3. To further illustrate the performance of the proposed methods, we introduce the regulation performance index $\begin{array} { r } { J _ { p } = \sum _ { t = 0 } x _ { t } ^ { \mathrm { T } } Q x _ { t } + u _ { t } ^ { \mathrm { T } } R u _ { t } } \end{array}$ . Table I shows the comparison of performance index and average sampling time, where we assume that the average sampling time for the periodic MPC is 1. It can be seen from Figs. 1-2 and Table I that both state and input constraints are satisfied for all $t \geq 0$ , but the proposed self-triggered adaptive MPC method can further reduce the average sampling frequency, while the comparable closed-loop performance is guaranteed. Fig. 4 demonstrates the evolution of EFSS for unknown parameters, where the red mark indicates real value of $v _ { t }$ and the length of the blue vertical line indicates the size of EFSS. It is worthwhile to observe that the falsified parameter can be removed by using the proposed set-membership parameter estimator under the self-triggering mechanism.

## VI. CONCLUSION

In this work, we developed a self-triggered adaptive MPC approach for constrained discrete-time nonlinear systems subject to parametric uncertainties and additive disturbances. A set-membership parameter estimator was designed to estimate the FSS of unknown parameters by using the indirect polytopic set computation. The zonotopic description of the FSS was considered to reduce overestimation. By integrating the proposed estimator with the min-max MPC technique, the robust constraint satisfaction was guaranteed with reduced conservatism. The new estimation of uncertainties was considered in both the triggering scheduler and the formulation of MPC optimization problem, therefore giving rise to the decreased sampling frequency compared with the robust selftriggered method. It has been proved that the proposed selftriggered adaptive MPC algorithm is recursively feasible and the closed-loop system is ISpS. We illustrated the efficacy and advantages of the proposed method over [15] by using a numerical example.

![](images/60cbfb23d047b145a67fa17479444b6f947eb7580d80db3c9eb05020a9211674.jpg)  
Fig. 2. The time evolution of control input $u _ { t } .$

![](images/44db1785384e01c61e409dd6ffcf98d89c26a2638f39de35bd0521f1bf63bdab.jpg)  
Fig. 3. Triggering time instants for the proposed method and the ST-RMPC.

## REFERENCES

[1] W. Heemels, K. H. Johansson, and P. Tabuada, “An introduction to event-triggered and self-triggered control,” in Proceedings of 51st IEEE

![](images/4f1a823a696bbf6c45afca843ec5c6f29eb3166b9a587d822efcb87fa660ee99.jpg)  
Fig. 4. Feasible solution set for the time-varying unknown parameters estimated at triggering time instants.

Conference on Decision and Control (CDC 2012). Maui, Hawaii, USA: IEEE, December 10-13, 2012, pp. 3270–3285.

[2] P. Tabuada, “Event-triggered real-time scheduling of stabilizing control tasks,” IEEE Transactions on Automatic Control, vol. 52, no. 9, pp. 1680–1685, 2007.

[3] X. Wang and M. D. Lemmon, “Event-triggering in distributed networked control systems,” IEEE Transactions on Automatic Control, vol. 56, no. 3, pp. 586–601, 2010.

[4] M. Velasco, J. Fuertes, and P. Marti, “The self triggered task model for real-time control systems,” in Work-in-Progress Session of the 24th IEEE Real-Time Systems Symposium (RTSS 2003), vol. 384, 2003.

[5] S. J. Qin and T. A. Badgwell, “A survey of industrial model predictive control technology,” Control Engineering Practice, vol. 11, no. 7, pp. 733–764, 2003.

[6] H. Li and Y. Shi, Robust Receding Horizon Control for Networked and Distributed Nonlinear Systems. Springer, 2016, vol. 83.

[7] D. Q. Mayne, “Model predictive control: Recent developments and future promise,” Automatica, vol. 50, no. 12, pp. 2967–2986, 2014.

[8] H. Li and Y. Shi, “Event-triggered robust model predictive control of continuous-time nonlinear systems,” Automatica, vol. 50, no. 5, pp. 1507–1513, 2014.

[9] H. Li, W. Yan, Y. Shi, and Y. Wang, “Periodic event-triggering in distributed receding horizon control of nonlinear systems,” Systems & Control Letters, vol. 86, pp. 16–23, 2015.

[10] C. Liu, H. Li, Y. Shi, and D. Xu, “Co-design of event trigger and feedback policy in robust model predictive control,” IEEE Transactions on Automatic Control, 2019.

[11] K. Hashimoto, S. Adachi, and D. V. Dimarogonas, “Self-triggered model predictive control for nonlinear input-affine dynamical systems via adaptive control samples selection,” IEEE Transactions on Automatic Control, vol. 62, pp. 177–189, 2016.

[12] J. Chen, Q. Sun, and Y. Shi, “Stochastic self-triggered MPC for linear constrained systems under additive uncertainty and chance constraints,” Information Sciences, vol. 459, pp. 198–210, 2018.

[13] Z. Sun, L. Dai, K. Liu, D. V. Dimarogonas, and Y. Xia, “Robust selftriggered MPC with adaptive prediction horizon for perturbed nonlinear systems,” IEEE Transactions on Automatic Control, 2019.

[14] H. Li, W. Yan, and Y. Shi, “Triggering and control codesign in selftriggered model predictive control of constrained systems: With guaranteed performance,” IEEE Transactions on Automatic Control, vol. 63, no. 11, pp. 4008–4015, 2018.

[15] C. Liu, H. Li, J. Gao, and D. Xu, “Robust self-triggered minmax model predictive control for discrete-time nonlinear systems,” Automatica, vol. 89, pp. 333–339, 2018

[16] F. D. Brunner, W. Heemels, and F. Allg¨ower, “Robust self-triggered MPC for constrained linear systems,” in Proceedings of 2014 European Control Conference (ECC 2014). IEEE, 2014, pp. 472–477.

[17] F. D. Brunner, M. Heemels, and F. Allg¨ower, “Robust self-triggered MPC for constrained linear systems: A tube-based approach,” Automatica, vol. 72, pp. 73–83, 2016.

[18] L. Dai, Y. Gao, L. Xie, K. H. Johansson, and Y. Xia, “Stochastic selftriggered model predictive control for linear systems with probabilistic constraints,” Automatica, vol. 92, pp. 9–17, 2018.

[19] V. Adetola, D. DeHaan, and M. Guay, “Adaptive model predictive control for constrained nonlinear systems,” Systems & Control Letters, vol. 58, no. 5, pp. 320–326, 2009.

[20] M. Lorenzen, M. Cannon, and F. Allg¨ower, “Robust MPC with recursive model update,” Automatica, vol. 103, pp. 461–471, 2019.

[21] D. Lim´on, T. Alamo, F. Salas, and E. F. Camacho, “Input to state stability of min–max MPC controllers for nonlinear systems with bounded uncertainties,” Automatica, vol. 42, no. 5, pp. 797–803, 2006.

[22] J. Wan, S. Sharma, and R. Sutton, “Guaranteed state estimation for nonlinear discrete-time systems via indirectly implemented polytopic set computation,” IEEE Transactions on Automatic Control, vol. 63, no. 12, pp. 4317–4322, 2018.

[23] T. Alamo, J. M. Bravo, and E. F. Camacho, “Guaranteed state estimation by zonotopes,” Automatica, vol. 41, no. 6, pp. 1035–1043, 2005.

[24] H. Wang, I. V. Kolmanovsky, and J. Sun, “Zonotope-based recursive estimation of the feasible solution set for linear static systems with additive and multiplicative uncertainties,” Automatica, vol. 95, pp. 236– 245, 2018.

[25] D. M. Raimondo, D. Limon, M. Lazar, L. Magni, and E. F. Camacho, “Min-max model predictive control of nonlinear systems: A unifying overview on stability,” European Journal of Control, vol. 15, no. 1, pp. 5–21, 2009.