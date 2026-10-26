# Tube Diffusion Policy: Reactive Visual-Tactile Policy Learning for Contact-rich Manipulation

Teng Xue<sup>1,2,3</sup>, Alberto Rigo<sup>1</sup>, Bingjian Huang<sup>1,4</sup>, Jiayi Shen<sup>1</sup>, Zhengtong Xu<sup>1,5</sup>, Nick Colonnese<sup>1</sup>, Amirhossein H. Memar<sup>1</sup>

![](images/1790cc56ece6234a1a70ac46e291323f5e278cd6495d9b27f96bff566ebb1cc8.jpg)  
Fig. 1: Overview of the full pipeline. We first collect visual-tactile human demonstrations via teleoperation in both simulation and on a physical robot, using a robotic arm equipped with a dexterous hand. Based on these demonstrations, Tube Diffusion Policy (TDP) learns a hybrid action velocity field that combines diffusion-based action generation with streaming conditiona feedback flows, forming an action tube that constrains the generated trajectory around the demonstration manifold. This design enables reactive control at every timestep, allowing rapid adaptation to contact uncertainty and external disturbances. Moreover, stepwise correction reduces the required denoising steps and significantly lowers inference latency. We evaluate TDP on three virtual and two physical visual–tactile dexterous manipulation tasks, demonstrating strong performance and consistent gains over state-of-the-art baselines.

Abstract—Contact-rich manipulation is central to many everyday human activities, requiring continuous adaptation to contact uncertainty and external disturbances through multimodal perception, particularly vision and tactile feedback. While imitation learning has shown strong potential for learning complex manipulation behaviors, most existing approaches rely on action chunking, which fundamentally limits their ability to react to unforeseen observations during execution. This limitation becomes especially critical in contact-rich scenarios, where physical uncertainty and high-frequency tactile feedback demand rapid, reactive control. To address this challenge, we propose Tube Diffusion Policy (TDP), a novel reactive visual–tactile policy learning framework that bridges diffusion-based imitation learning with tube-based feedback control. By leveraging the expressive power of generative models, TDP learns an observation-conditioned feedback flow around nominal action chunks, forming an action tube that enables fast and adaptive reactions during execution. We evaluate TDP on the widely used Push-T benchmark and three additional challenging visual–tactile dexterous manipulation tasks. Across all benchmarks, TDP consistently outperforms state-of-the-art imitation learning baselines. Two real-world experiments further validate its robust reactivity under contact uncertainty and external disturbances. Moreover, the step-wise correction mechanism enabled by action tube significantly reduces the required denoising steps, making TDP well suited for real-time, high-frequency feedback control in contact-rich manipulation.

Index Terms—Dexterous Manipulation, Visual-Tactile Policy Learning, Learning from Demonstration, Reactive Control

## I. INTRODUCTION

Contact-rich manipulation plays a fundamental role in human daily activities and inherently involves rich physical interactions among robots, objects, and the environment. Humans are able to perform such tasks with highly reactive behaviors that adapt instantaneously to contact uncertainty and multimodal observations. However, to achieve similar capabilities on robots, there are still two open challenges. First, contact uncertainty arises from unknown or varying object geometry, inertial properties, and frictional characteristics, leading to highly nonlinear dynamics [45, 47]. Second, multi-modal perception—particularly vision–tactile sensing—introduces heterogeneous observation modalities, where tactile signals provide contact-related information at high frequency and demand rapid control responses [44]. Together, these challenges highlight the necessity of reactive control policies that can precisely model complicated behaviors while quickly adapt to new observations in real time.

Recently, imitation learning based on generative models has emerged as a dominant paradigm for learning robot control policies, such as Diffusion Policy [7] and Flow Matching Policy [51]. These approaches excel at capturing the multimodal distribution of expert demonstrations and have achieved strong performance across a wide range of manipulation tasks. However, a fundamental limitation of these methods lies in their slow inference, which makes it impractical to generate actions at every timestep. As a result, many approaches adopt action chunking [52], where a sequence of future actions is generated conditioned on the current observation and then executed without intermediate feedback. While this design enables temporally coherent behavior generation and reduces the frequency of expensive inference, the policy operates in an open-loop manner over the chunk horizon. From a control perspective, such generative imitation learning policies can be interpreted as a form of receding-horizon control, analogous to model predictive control (MPC) [21], in which a nominal action trajectory is repeatedly replanned based on the latest observation. However, unlike classical MPC—which executes only the first control input and replans at high frequency—these methods execute multiple actions between updates due to inference latency, limiting their ability to incorporate high-frequency feedback. This limitation is particularly problematic in contact-rich manipulation tasks, where unmodeled interactions and rapid tactile feedback are critical for success. The lack of reactivity can lead to degraded performance under disturbances and model mismatch. Moreover, executing multi-step actions implicitly assumes accurate prediction of system evolution over the entire horizon, an assumption that often breaks down in contact-rich settings, resulting in brittle behaviors.

In the control literature, robustness to uncertainty has been widely studied [53, 10, 8, 36]. One representative approach is tube model predictive control (tube MPC) [23, 30], which decomposes control into a nominal feedforward trajectory and a stabilizing feedback controller that keeps the actual system state within a bounded tube around the nominal trajectory. This feedback mechanism enables fast local corrections in response to disturbances and modeling errors. However, classical tube MPC typically relies on linear or locally linear dynamics, requires explicit system models, and incurs substantial computational overhead, making it difficult to scale to complex, nonlinear, contact-rich manipulation tasks.

Inspired by the tube MPC paradigm, this work aims to endow diffusion-based imitation learning policies with similar robustness and reactivity by learning a local feedback flow around nominal trajectories. Instead of explicitly modeling system dynamics or deriving analytical feedback gains, we leverage the expressive capacity of generative models to directly learn an observation-conditioned feedback policy from demonstrations. To this end, we propose Tube Diffusion Policy (TDP), a reactive visual–tactile policy learning framework that augments diffusion-based imitation learning with learned feedback control during action execution. The central idea is that, rather than committing to a fixed action sequence as action chunk, we construct an action tube that implicitly constrains execution within a bounded neighborhood around a nominal trajectory while allowing actions to be locally adjusted through a learned feedback flow. In particular, at the beginning of each action chunk horizon, a diffusion model generates an initial action that accounts for the full system dynamics and serves as a nominal initialization. The learned feedback flow then streams subsequent actions conditioned on fresh observations, enabling local regulation within the action tube during execution. The entire structure is learned directly from demonstrations without requiring explicit system models, making it well suited for nonlinear and contact-rich manipulation.

To reconcile expressive generative modeling with efficient feedback control within a unified policy architecture, we introduce a dual-time formulation that explicitly separates diffusion and streaming flow in policy learning. A spatial timescale governed by diffusion dynamics captures the multimodal action distribution while accounting for full contact dynamics, whereas a temporal timescale governs action evolution by streaming a learned flow defined by a velocity field. This dualtime architecture enables accurate action generation through diffusion while supporting fast observation-conditioned corrections during execution. As a result, TDP mitigates the openloop execution inherent in action chunking approaches, while retaining the expressiveness of diffusion-based policy learning.

Overall, this framework can be viewed as a learning-based realization of tube MPC under unknown dynamics. Rather than solving a model-based optimization problem, TDP leverages a learned action velocity field to induce locally stabilizing behavior. Extensive simulation and real-world experiments on challenging visual–tactile manipulation tasks demonstrate substantial improvements over state-of-the-art imitation learning baselines in both task success and reaction under uncertainty.

The main contributions of this work are summarized as follows:

• We introduce action tube, which extends the widely used action chunk with a learned local feedback flow, enabling reactive control during execution. This formulation also relaxes the need for highly accurate action generation through denoising and thereby significantly reduces con-

trol latency.

• We propose Tube Diffusion Policy (TDP), a novel architecture that integrates diffusion-based action generation with learned feedback flow, enabling reactive visual–tactile policy learning for contact-rich manipulation.

• We provide a theoretical stability analysis showing that reactive control can be formulated as learning an observation-conditioned action velocity field under locally linearized dynamics, while diffusion performs chunk-level corrections that account for full system dynamics.

• We validate the proposed approach through extensive simulation and real-world experiments on challenging visual–tactile manipulation tasks, demonstrating consistent improvements over state-of-the-art imitation learning baselines in task success and reactivity under perturbations.

## II. RELATED WORK

The proposed Tube Diffusion Policy sits at the intersection of three research threads: (1) chunk-based policy learning, which provides the temporal abstraction and generative modeling that TDP uses for action generation; (2) visual–tactile manipulation, which motivates the need for high-frequency reactive control under contact uncertainty; and (3) learning dynamical systems, which supplies the velocity-field formulation that TDP adopts for streaming execution. As summarized in Table I, each thread individually addresses only a subset of the requirements for reactive contact-rich manipulation. We review each thread below, focusing on the specific limitation that TDP is designed to overcome.

## A. Chunk-based Policy Learning

Action chunking is a concept originating in psychology that groups a sequence of actions into a single chunk and executes it as a unit [22]. It has since been widely adopted to address long-horizon decision making in continuous control [25, 52]. In imitation learning, chunk-based formulations enable policies to model expert behavior at the level of short trajectory segments rather than individual actions. Early works focus on learning reusable latent skills or motion primitives [12, 29, 46]. More recent approaches directly predict action chunks using sequence models. Notably, Action Chunking Transformer (ACT) [52] demonstrates that predicting fixed-length action sequences can significantly reduce error accumulation and enable long-horizon bimanual manipulation, even on low-cost hardware platforms. However, ACT executes the predicted chunks largely in an open-loop manner, and its performance can degrade under contact uncertainty or external disturbances within each chunk.

Generative models further enhance chunk-based policies by enabling expressive multi-step action generation. They generate entire action chunks conditioned on current observations, allowing the policy to represent multimodal behaviors and capture long-horizon dependencies [7, 51, 17]. While effective in visuomotor manipulation, these methods typically replan only at the chunk boundary and provide limited mechanisms for correcting high-frequency deviations caused by modeling errors or contact variations during execution.

As a result, existing chunk-based and generative policy learning methods provide powerful abstractions for long-horizon planning but remain fundamentally limited by their weak intra-chunk feedback mechanisms, particularly in contact-rich settings with tactile feedback [28, 44]. Our work aims to address this issue by integrating chunk-based generative policies with structured local feedback flows, enabling systematic disturbance rejection while preserving the benefits of temporal abstraction.

## B. Visual-tactile Manipulation

Visual–tactile manipulation has attracted increasing attention, particularly in contact-rich tasks that require force regulation and fine-grained interaction. Early work leveraged tactile signals for contact detection, slip estimation, and state inference, demonstrating that tactile feedback is an essential component in tasks such as grasp stabilization [4, 1], in-hand manipulation [41], and deformable object manipulation [48].

Recent advances in visuomotor policy learning [7, 52, 49, 51] have further enabled the learning of complex manipulation behaviors, motivating the integration of tactile sensing into policy learning frameworks. Most existing approaches incorporate tactile signals as additional sensory inputs by concatenating them with visual observations into a unified observation space, and train policies end-to-end using standard visuomotor learning pipelines [24, 11, 50]. While effective in many settings, such formulations often rely on computationally intensive generative models with relatively slow inference, and therefore overlook the fact that tactile sensing is inherently high-frequency and contact-driven, requiring rapid reactive responses during execution.

To better exploit tactile feedback, visual–tactile policy learning should adopt a reactive formulation. One representative approach is Reactive Diffusion Policy (RDP) [44], which employs a slow–fast architecture that combines diffusion-based planning with fast tactile-driven refinement, enabling improved reactivity to external disturbances in contact-rich manipulation tasks. Nevertheless, it has been primarily validated on parallel grippers rather than dexterous hands, leaving its effectiveness in high-dimensional, multi-contact manipulation settings unclear. Moreover, it still relies on slow denoising at the beginning of each action chunk, introducing latency and limiting responsiveness, which can induce periodic jerky motions similar to those observed in standard diffusion policies.

Overall, most existing visual–tactile policies either execute planned action chunks in an open-loop manner or rely on auxiliary controllers that operate in latent space, limiting their ability to support fully reactive visual–tactile control for high-DoF dexterous manipulation. In contrast, we propose TDP, which embeds reactivity directly into the diffusion policy through a streaming integration phase, enabling online action updates under an observation-conditioned vector field.

## C. Learning Dynamical Systems

Learning policies and motions as dynamical systems has a long history in robotics and imitation learning. Early work represents robot behaviors as stable nonlinear dynamical systems or attractor models, enabling globally convergent motions learned from demonstrations [18, 19, 15, 35]. These approaches impose strong structural constraints to guarantee stability, which simplifies the underlying dynamics and facilitates analysis, but limits expressiveness when dealing with highly nonlinear or contact-rich interactions.

TABLE I: Comparison of policy learning paradigms for reactive visual-tactile manipulation.

<table><tr><td>Method</td><td>Visual-Tactile Observation</td><td>Closed-loop Execution</td><td>Contact-rich Dynamics</td><td>Fast Inference</td></tr><tr><td>Action Chunking with Transformers (ACT) [52]</td><td>✗</td><td>✗</td><td>√</td><td>√</td></tr><tr><td>Diffusion Policy (DP) [7]</td><td>✗</td><td>✗</td><td>√</td><td>✗</td></tr><tr><td>Dynamical Systems [18, 19, 15, 35]</td><td>✗</td><td>√</td><td>✗</td><td>√</td></tr><tr><td>Streaming Flow Policy (SFP) [17]</td><td>✗</td><td>✗</td><td>✗</td><td>√</td></tr><tr><td>Reactive Diffusion Policy (RDP) [44]</td><td>√</td><td>√</td><td>√</td><td>✗</td></tr><tr><td>Tube Diffusion Policy (TDP)</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

Recent learning-based methods extend this perspective by parameterizing continuous-time dynamics with neural networks. Neural ODEs and their variants provide a flexible framework for learning nonlinear dynamical systems from data [6, 34, 20], while recent work explicitly enforces stability properties in deep dynamical systems [37]. Although more expressive, these methods are typically used as motion generators or nominal dynamics models, rather than as fully reactive controllers under complex disturbances.

In parallel, generative modeling techniques such as diffusion models, continuous normalizing flows, and flow matching reinterpret trajectory or action generation as the integration of learned vector fields [39, 14, 26, 27]. Building on this perspective, Streaming Flow Policy (SFP) [17] formulates policy generation as a continuous-time dynamical system evolving in action space, enabling smooth action evolution and improved coverage of multimodal behaviors. However, due to the computational cost of generative inference, these methods primarily produce open-loop or chunk-based action trajectories, with limited ability to react to execution-time disturbances.

Overall, most prior learning-based dynamical system approaches either rely on simplified or abstracted dynamics, or structure execution around chunk-based or open-loop evolution, limiting their ability to represent complex behaviors or regulate deviations arising from disturbances or contact interactions. Our approach combines the strengths of diffusionbased policies and streaming flows, where diffusion is used to capture complex nonlinear behaviors, while observationconditioned streaming flow enables real-time, reactive control during execution.

## III. PROBLEM FORMULATION

We consider contact-rich manipulation systems with observation ${ \pmb { o } } _ { t } = \left( { \pmb { o } } _ { t } ^ { o } , { \pmb { o } } _ { t } ^ { r } \right) \in \mathbb { O } \subseteq \mathbb { R } ^ { n _ { o } }$ , where ${ \mathbf { } } _ { o _ { t } ^ { r } }$ denotes the robot’s proprioceptive state and $\mathbf { \delta } _ { \mathbf { \alpha } ^ { o } \mathbf { \beta } _ { t } ^ { o } }$ represents the object state, which may consist of either low-dimensional object configurations or high-dimensional, multimodal sensory inputs such as visual and tactile observations. The control input $\pmb { { u } } _ { t } \in \mathbb { U } \subseteq \mathbb { R } ^ { n _ { u } }$ specifies the commanded robot action. The system evolves according to the continuous-time dynamics

$$
\dot {\boldsymbol {o}} (t) = f (\boldsymbol {o} (t), \boldsymbol {u} (t)),\tag{1}
$$

where $f ( \cdot )$ captures the underlying contact-rich interaction dynamics between the robot and its environment.

To enable reactive feedback control, we adopt a tube-based control formulation. Specifically, by locally linearizing the system over a short time horizon, we can obtain the following control law with a proportional feedback correction:

$$
\boldsymbol {u} (t) = \bar {\boldsymbol {u}} (t) + K (\bar {\boldsymbol {o}} (t) - \boldsymbol {o} (t)),\tag{2}
$$

where $\bar { \pmb u } ( t )$ and $\bar { \bf { o } } ( t )$ denote the nominal action and observation trajectories, respectively, and the feedback gain $K \succ 0$ is held constant within the horizon. Differentiating (2) with respect to time yields

$$
\dot {\boldsymbol {u}} (t) = \dot {\bar {\boldsymbol {u}}} (t) + K (\dot {\bar {\boldsymbol {o}}} (t) - \dot {\boldsymbol {o}} (t)).\tag{3}
$$

We consider a neighborhood of the nominal trajectory in which the observation dynamics are differentiable and admit a first-order approximation. In particular, we apply a Taylor expansion around $( \bar { \pmb { o } } ( t ) , \bar { \pmb { u } } ( t ) )$ of system dynamics (1), leading to

$$
\dot {\boldsymbol {o}} (t) \approx \dot {\bar {\boldsymbol {o}}} (t) + A (t) (\boldsymbol {o} (t) - \bar {\boldsymbol {o}} (t)) + B (t) (\boldsymbol {u} (t) - \bar {\boldsymbol {u}} (t)),\tag{4}
$$

where $\begin{array} { r } { \boldsymbol { A } ( t ) \triangleq \left. \frac { \partial f } { \partial \boldsymbol { o } } \right| _ { ( \bar { \boldsymbol { o } } ( t ) , \bar { \boldsymbol { u } } ( t ) ) } } \end{array}$ and $\begin{array} { r } { B ( t ) \triangleq \left. \frac { \partial f } { \partial \pmb { u } } \right| _ { ( \bar { \pmb { o } } ( t ) , \bar { \pmb { u } } ( t ) ) } . } \end{array}$

Within each time interval, we adopt the widely used quasidynamic assumption for contact-rich systems: over the short duration of each interval, the observation error $e ( t ) = \bar { o } ( t ) -$ ${ \mathbf { } } _ { o ( t ) }$ remains bounded and varies slowly, such that its contribution can be neglected in the local approximation. This yields

$$
\dot {\boldsymbol {o}} (t) - \dot {\boldsymbol {o}} (t) \approx - \Gamma (t) (\boldsymbol {u} (t) - \bar {\boldsymbol {u}} (t)), \quad \Gamma (t) \triangleq B (t),\tag{5}
$$

where $\Gamma ( t )$ summarizes the local input-to-observation sensitivity induced by contact interactions and the environment, and is understood to be valid locally.

Substituting (5) into (3) yields the action-space velocity field

$$
\dot {\boldsymbol {u}} (t) \approx \dot {\bar {\boldsymbol {u}}} (t) - K \Gamma (t) (\boldsymbol {u} (t) - \bar {\boldsymbol {u}} (t)).\tag{6}
$$

Equation (6) decomposes the controller into a trajectoryfollowing term $\dot { \bar { \mathbf { u } } } ( t )$ and a stabilizing contraction term that drives system deviations toward the nominal trajectory, thereby locally maintaining the execution error within a bounded tube.

For simplicity, we approximate the product $K \Gamma ( t )$ by an isotropic contraction term λI. This approximation is motivated by the observation that action–observation coupling in contact-rich manipulation is often dominated by a few lowdimensional interaction modes within a local contact regime. For instance, in planar pushing with persistent contact, the interaction is typically governed by contact modes such as sticking or sliding at the contact interface [45, 47]. Within each mode, the resulting action–observation coupling is effectively low-dimensional, making an isotropic contraction a reasonable local approximation. Here, $\lambda > 0$ denotes the effective local contraction rate in action space. Substituting this approximation into (3) yields the simplified action-space dynamics

$$
\dot {\boldsymbol {u}} (t) \approx \dot {\bar {\boldsymbol {u}}} (t) - \lambda (\boldsymbol {u} (t) - \bar {\boldsymbol {u}} (t)), \quad \lambda \triangleq \gamma k > 0,\tag{7}
$$

which describes a contracting flow around the nominal action trajectory.

This formulation is structurally consistent with [17], which frames policy learning as the imitation of action trajectories. While effective in some settings, [17] abstracts away robot–environment coupling and is generally insufficient, as control policies are fundamentally designed to regulate system states or observations rather than actions alone. We instead adopt a full-system perspective and identify physically grounded assumptions under which action-trajectory imitation provides a valid local approximation of closed-loop control. We show that this approximation holds only in a neighborhood of the nominal trajectory under a quasi-dynamic execution regime with bounded tracking errors. This analysis provides the theoretical motivation for the proposed approach, in which diffusion provides chunk-level corrections that account for full system dynamics, while the streaming flow enables local feedback control during execution.

## IV. METHODOLOGY

From (7), we observe that after local linearization, the resulting feedback flow depends only on the nominal action trajectory. This insight suggests that the conventional decisionmaking process can be conceptually decomposed into two stages: (i) periodically generating actions that account for the full nonlinear dynamics to correct accumulated drift, and (ii) applying a local feedback controller based on linearized dynamics.

Motivated by this decomposition, we propose Tube Diffusion Policy (TDP), a framework for reactive policy learning under real-time observations. At the beginning of each action chunk, a diffusion model performs multi-step denoising to generate an initial action that accounts for the full nonlinear contact dynamics. Subsequent actions are then generated through a streaming feedback flow learned from demonstrations, which locally regulates deviations based on fresh observations. Although the diffusion stage does not explicitly generate a full nominal action trajectory, it implicitly defines one by providing the initial condition for the subsequent action evolution. The resulting action sequence can therefore be interpreted as a nominal trajectory around which closedloop regulation is performed. This structure aligns with the action chunk paradigm in chunk-based control, but crucially augments it with step-wise closed-loop feedback during execution. We refer to this formulation as an action tube, where diffusion provides chunk-level action initialization and the learned feedback flow enforces real-time corrections within the tube. This section presents the overall architecture for learning such reactive control policies from demonstrations.

![](images/b2a3ec6038f8c5327a5ad1822972fa97417773ea9d9d1b6a996d9c073873d85a.jpg)  
Fig. 2: Overview of Tube Diffusion Policy (TDP). TDP adopts a dual-time formulation with a multi-step denoising stage along $t _ { 1 }$ and an observation-conditioned streaming stage along $t _ { 2 } .$ . The method operates in two phases: a denoising phase and a streaming phase. At the beginning of each action chunk horizon, an initial action is generated through multi-step denoising, accounting for the full nonlinear system dynamics; the resulting trajectory is shown in red. Subsequent actions are produced by streaming action flow conditioned on the observation at each timestep, yielding the orange trajectory. The green curve denotes the nominal trajectory. This process can be interpreted as executing actions within an action tube (dashed boundary), where local deviations are continuously regulated by switching among a sequence of observationconditioned flow models (illustrated as funnels). Here, $h _ { t }$ denotes the observation history, and $\Delta t$ controls the streaming temporal granularity. By combining diffusion-based chunklevel initialization with tube-based reactive feedback, TDP enables robust, high-frequency closed-loop control for contactrich manipulation.

Given a dataset of demonstration episodes, similar to chunkbased approaches [7, 52], we decompose each episode into multiple observation and action sequences of length $H _ { p } ,$ yielding a dataset $\mathcal { D } = \{ ( \zeta _ { i } ^ { o } , \zeta _ { i } ^ { u } ) \} _ { i = 1 } ^ { N }$ . Our goal is to learn the conditional velocity field $v _ { \theta } ( \boldsymbol { \mathbf { \mathit { u } } } , t _ { 1 } , t _ { 2 } \mid h _ { \tau } )$ by minimizing (8), enabling both diffusion-based action generation and flowbased reactive control. Here, $\begin{array} { r c l } { h _ { \tau } } & { = } & { \overline { { { \{ \pmb { o } _ { t } \} _ { t = \tau } ^ { \tau + H _ { o } } } } } } \end{array}$ denotes a finite observation history of length $H _ { o } . \mathrm { A t }$ execution time, the learned policies are deployed in a receding-horizon fashion, with an action horizon $H _ { a }$ that may differ from the prediction horizon $H _ { p }$ used during training. The complete pseudocode for training and inference is provided in Alg. 1 and Alg. 2.

## A. Overall Architecture

a) Dual-Time Formulation: TDP introduces two time variables with distinct semantics:

$t _ { 1 } ~ \in ~ [ 0 , T _ { \mathrm { d i f f } } ]$ : diffusion time, governing the diffusion denoising process while the physical system remains stationary;

$t _ { 2 } ~ \in ~ [ 0 , 1 ] \colon$ trajectory time, governing the streaming control process that evolves in real time with the physical system.

The policy is conditioned on both variables, allowing a single network to operate in two modes: diffusion-based action generation and flow-based streaming.

b) Training Objective: The overall training objective combines losses from both modes:

$$
\mathcal {L} = \mathcal {L} _ {\text { diff }} + \mathcal {L} _ {\text { stream }},\tag{8}
$$

where ${ \mathcal { L } } _ { \mathrm { d i f f } }$ is the diffusion noise-prediction loss and $\mathcal { L } _ { \mathrm { s t r e a m } }$ supervises the streaming velocity field. We use equal weighting for the two objectives, as they correspond to complementary roles—global action generation and local feedback correction—and operate on comparable scales in practice.

## B. Diffusion Phase: Diffusion-based Action Generation

The diffusion phase follows the standard Denoising Diffusion Probabilistic Models (DDPM) formulation with ϵ- prediction [14]. Given a clean action $\begin{array} { r } { \pmb { u } _ { 0 } = \zeta ^ { u } [ \pmb { H } _ { o } - 1 : \pmb { H } _ { o } ] } \end{array}$ that is aligned with the observation history $h _ { t _ { 1 } } = \zeta ^ { o } [ : H _ { o } ]$ the forward noising process is defined as

$$
\pmb {u} _ {t _ {1}} = \sqrt {\bar {\alpha} _ {t _ {1}}} \pmb {u} _ {0} + \sqrt {1 - \bar {\alpha} _ {t _ {1}}} \pmb {\epsilon}, \qquad \pmb {\epsilon} \sim \mathcal {N} (\mathbf {0}, \mathbf {I}),\tag{9}
$$

where $\bar { \alpha } _ { t _ { 1 } }$ is specified by the noise schedule in [14]. The network $v _ { \theta } ( \mathbf { u } , t _ { 1 } , t _ { 2 } \ \mid h _ { t _ { 1 } } )$ is trained to predict the injected noise conditioned on the aligned observation history $h _ { t _ { 1 } }$ by minimizing

$$
\mathcal {L} _ {\text { diff }} = \mathbb {E} _ {\boldsymbol {u} _ {0}, h _ {t _ {1}}, \epsilon , t _ {1}} \left[ \left\| \boldsymbol {\epsilon} - v _ {\theta} (\boldsymbol {u} _ {t _ {1}}, t _ {1}, t _ {2} = 0 \mid h _ {t _ {1}}) \right\| ^ {2} \right],\tag{10}
$$

where the expectation is taken over aligned observation–action pairs $( h _ { t _ { 1 } } , { \boldsymbol { \mathbf { \mathit { u } } } } _ { 0 } )$ sampled from the demonstration dataset ${ \mathcal { D } } ,$ and $t _ { 1 }$ is sampled uniformly from $[ 0 , T _ { \mathrm { d i f f } } ]$ , with $[ 0 , T _ { \mathrm { d i f f } } ]$ specifying the length of the diffusion process.

## C. Streaming Phase: Flow-based Action Streaming

By locally linearizing the system dynamics, the tubebased feedback controller can be projected from the observation–action space into the action space. Under this formulation, the streaming phase learns a reactive controller by directly modeling action evolution as a velocity field over the trajectory time, enabling high-frequency closed-loop execution. This design is inspired by [17].

Following (7), we define the target velocity field along the action trajectory as

$$
v _ {\zeta} (\boldsymbol {u}, t _ {2}) = \frac {d \zeta^ {u}}{d t _ {2}} - \lambda \bigl (\boldsymbol {u} - \zeta^ {u} (t _ {2}) \bigr),\tag{11}
$$

where $\zeta ^ { u } ( t _ { 2 } )$ denotes the continuous nominal action trajectory parameterized by the trajectory time $t _ { 2 } \in [ 0 , 1 ]$ . The first term provides feedforward progression along it, while the second induces feedback toward the nominal trajectory.

To enable reactive control within the action tube, we extract step-wise observation history aligned with the trajectory time via

$$
h _ {t _ {2}} = \zeta^ {o} [ l: l + H _ {o} ],\tag{12}
$$

where $l = \lfloor t _ { 2 } H _ { p } \rfloor$ is the index corresponding to $t _ { 2 }$ on the discretized observation trajectory, with ⌊·⌋ denoting the floor (round-down) operation. $H _ { p }$ denotes the prediction horizon, and $H _ { o }$ is the length of the observation history window.

The network $v _ { \theta } ( \mathbf { u } , t _ { 1 } , t _ { 2 } ~ | ~ h _ { t _ { 2 } } )$ is trained to predict the streaming velocity field conditioned on the aligned observation history $h _ { t _ { 2 } }$ by minimizing

$$
\mathcal {L} _ {\text {stream}} = \mathbb {E} _ {\boldsymbol {u}, h, t _ {2}} \left[ \left\| v _ {\theta} (\boldsymbol {u}, t _ {1} = 0, t _ {2} \mid h _ {t _ {2}}) - v _ {\zeta} (\boldsymbol {u}, t _ {2}) \right\| ^ {2} \right],\tag{13}
$$

where $t _ { 2 } \sim \mathcal { U } ( [ 0 , 1 ] )$ , and u is sampled from the stabilizing conditional distribution

$$
\boldsymbol {u} \sim p _ {\zeta} (\boldsymbol {u} \mid t _ {2}) = \mathcal {N} \bigl (\zeta^ {u} (t _ {2}), \sigma_ {0} ^ {2} e ^ {- 2 k t _ {2}} \mathbf {I} \bigr),\tag{14}
$$

which is a Gaussian centered at the action trajectory $\zeta ^ { u } ( t _ { 2 } )$ with a small variance $\sigma _ { 0 } ^ { 2 } ,$ as introduced in [17].

By extracting step-wise observation histories, we learn a family of observation-conditioned flow models of the form $v _ { \theta } ( \mathbf { u } , t _ { 1 } = 0 , t _ { 2 } \mid h _ { t _ { 2 } } )$ , each conditioned on a distinct observation history $h _ { t _ { 2 } }$ . During execution, the action sequence within the chunk horizon is not governed by a single conditional flow model; instead, actions are generated by sequentially switching among multiple observation-conditioned flows as new observation becomes available and provide updated conditioning.

At each step, the action is updated by integrating the currently active streaming velocity field,

$$
\pmb {u} \leftarrow \pmb {u} + v _ {\theta} (\pmb {u}, t _ {1} = 0, t _ {2} \mid h _ {t _ {2}}) \Delta t _ {2},\tag{15}
$$

where $\Delta t _ { 2 }$ denotes the discretization step size of the trajectory time t<sub>2</sub>. This step-wise integration incorporates fresh observations at every step, enabling closed-loop and reactive control.

## D. Network Architecture and Encoding

We employ a conditional 1D U-Net as the shared backbone for both diffusion-based action generation and streamingbased reactive flow. The network predicts action evolution conditioned on observations, outputting either diffusion noise or action velocity depending on the time conditioning, which enables parameter sharing across the diffusion and streaming phases.

To support the dual-time formulation, the network uses separate sinusoidal embeddings for the diffusion time $t _ { 1 }$ and the streaming time $t _ { 2 }$ . The two embeddings are scaled differently to reflect their distinct semantics: diffusion time follows the standard DDPM time scale, while trajectory time is amplified to provide finer resolution over the normalized execution horizon [0, 1]. This separation prevents interference between diffusion and streaming behaviors while allowing a unified architecture.

Observation features and time embeddings are injected into the network via Feature-wise Linear Modulation (FiLM) [31]. This conditioning mechanism enables the policy to adapt its action predictions to both the current observation context and the operating mode (diffusion or streaming) in a smooth and modular manner.

Visual observations are encoded using ResNet-18 backbones [13]. In simulation, tactile inputs are processed using a lightweight multilayer perceptron, as the tactile sensor provides array-based measurements. In real-world experiments, tactile observations from Digit360 sensors are image-based and are therefore encoded using multiple ResNet-18 backbones. The resulting features are concatenated with proprioceptive signals to form a global observation representation, which conditions the action policy through FiLM modulation. Detailed network architecture and hyperparameters are provided in Appendix B.

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1 Training
Require: Training set $\mathcal{D} = \{(\zeta_i^o, \zeta_i^u)\}_{i=1}^N$, $H_o$, $H_p$, $T_{\text{diff}}$
1: $\zeta^o$ and $\zeta^u$ have time horizon $T_{\text{pred}}$ rescaled to [0,1]
2: while not converged do
3: $(\zeta^o, \zeta^u) \sim \mathcal{D}$
4: // ====== Diffusion phase ($t_1$) ====
5: $t_1 \sim \text{Uniform}\{0, \ldots, T_{\text{diff}}\}$ // diffusion time
6: $h_{t_1} \leftarrow \zeta^o[: H_o]$
7: $\boldsymbol{u}_0 \leftarrow \zeta^u[H_o - 1 : H_o]$
8: $\epsilon \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$
9: $\boldsymbol{u}_{t_1} = \sqrt{\bar{\alpha}_{t_1}} \boldsymbol{u}_0 + \sqrt{1 - \bar{\alpha}_{t_1}} \epsilon$ // Eq. (9)
10: $\mathcal{L}_{\text{diff}} \leftarrow \| \epsilon - v_\theta(\boldsymbol{u}_{t_1}, t_1, t_2 = 0 | h_{t_1})\|^2$ // Eq. (10)
11: // ====== Streaming phase ($t_2$) ====
12: $t_2 \sim \text{Uniform}(0, 1)$ // trajectory time
13: $\boldsymbol{u} \sim p_{\zeta^u}(\boldsymbol{u} | t_2)$
14: $l \leftarrow \lfloor t_2 H_p \rfloor$
15: $h_{t_2} \leftarrow \zeta^o[l : l + H_o]$ // Eq. (12)
16: $v_\zeta(\boldsymbol{u}, t_2) \leftarrow E q.$ (11)
17: $\mathcal{L}_{\text{stream}} \leftarrow \| v_\zeta(\boldsymbol{u}, t_2) - v_\theta(\boldsymbol{u}, t_1 = 0, t_2 | h_{t_2})\|^2$
18: $\theta \leftarrow \theta - \gamma \nabla_\theta(\mathcal{L}_{\text{diff}} + \mathcal{L}_{\text{stream}})$
19: end while
20: return $v_\theta$
</div>

Actions consist of end-effector position, orientation, and hand joint commands. For orientation, we adopt the continuous 6D rotation representation [54], which avoids discontinuities associated with quaternion parameterizations and is better suited for regression-based learning.

## E. Stability Analysis

TDP decomposes control into two complementary mechanisms: diffusion correction, which periodically contracts the accumulated drift, and streaming imitation, which provides high-frequency reactive control but may introduce bounded tracking error. Together they form a hybrid control system that admits a practical stability guarantee.

Consider the discrete-time system

$$
\boldsymbol {o} _ {t + 1} = g (\boldsymbol {o} _ {t}, \boldsymbol {u} _ {t}) + \boldsymbol {w} _ {t},\tag{16}
$$

where $g ( \cdot )$ denotes the discrete-time dynamics corresponding to (1), and ${ \pmb w } _ { t }$ is a bounded disturbance.

Let $( o _ { t } ^ { * } , u _ { t } ^ { * } )$ be a demonstrated reference trajectory satisfying

$$
\pmb {o} _ {t + 1} ^ {*} = g (\pmb {o} _ {t} ^ {*}, \pmb {u} _ {t} ^ {*}),\tag{17}
$$

and define the tracking error ${ \boldsymbol { e } } _ { t } ~ = ~ { \boldsymbol { o } } _ { t } - { \boldsymbol { o } } _ { t } ^ { * }$ and the value function $V ( e _ { t } ) = \| e _ { t } \|$

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 2 Inference
Require: $v_{\theta}(\boldsymbol{u}, t_1, t_2 \mid h)$, $H_o$, $H_p$, $H_a$, $T_{\text{ddim}}$, $\Delta t$
1: $h \leftarrow \{\boldsymbol{o}_i\}_{i=1}^{H_o}$ // observation history
2: while True do
3: // ===== Phase 1: Denoising ($t_1$)=====
4: $\boldsymbol{u} \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$
5: for $t_1 = T_{\text{ddim}}, \ldots, 1$ do
6: $\hat{\epsilon} \leftarrow v_{\theta}(\boldsymbol{u}, t_1, t_2 = 0 \mid h)$
7: $\boldsymbol{u} \leftarrow \text{DDIMStep}(\boldsymbol{u}, \hat{\epsilon}, t_1)$
8: end for
9: // ===== Phase 2: Streaming ($t_2$)=====
10: $t_2 \leftarrow \Delta t$
11: while $t_2 \leq H_a / H_p$ do
12: $o \leftarrow \text{Execute}(\boldsymbol{u})$ // stream action
13: $h \leftarrow h \cup \{o\}$ // fresh observation
14: $\boldsymbol{u} \leftarrow \boldsymbol{u} + v_{\theta}(\boldsymbol{u}, t_1 = 0, t_2 \mid h) \cdot \Delta t$ // integration
15: $t_2 \leftarrow t_2 + \Delta t$
16: end while
17: end while
</div>

To facilitate the stability analysis, we introduce the following conditions.

Assumption 1 (Lipschitz dynamics). The discrete-time dynamics satisfy

$$
\left\| g \left(\boldsymbol {o} _ {1}, \boldsymbol {u} _ {1}\right) - g \left(\boldsymbol {o} _ {2}, \boldsymbol {u} _ {2}\right) \right\| \leq L _ {x} \left\| \boldsymbol {o} _ {1} - \boldsymbol {o} _ {2} \right\| + L _ {u} \left\| \boldsymbol {u} _ {1} - \boldsymbol {u} _ {2} \right\|.\tag{18}
$$

The Lipschitz condition bounds the sensitivity of the system dynamics to state and action deviations, enabling tractable propagation of tracking errors.

Assumption 2 (Streaming imitation accuracy). The streaming flow policy is trained to imitate the reference action trajectory. As a result, the action error remains bounded:

$$
\| \pmb {u} _ {t} - \pmb {u} _ {t} ^ {*} \| \leq \varepsilon_ {a}.\tag{19}
$$

This condition reflects the training objective of the streaming phase in TDP and is expected to hold when the policy accurately reproduces the demonstrated actions.

Assumption 3 (Bounded disturbance). The external disturbance is bounded:

$$
\left\| \boldsymbol {w} _ {t} \right\| \leq \bar {w}.\tag{20}
$$

Assumption 4 (Diffusion correction property). At correction steps $t _ { k }$ , the diffusion module produces a corrective action that reduces the accumulated tracking error:

$$
V (\boldsymbol {e} _ {t _ {k} + 1}) \leq \lambda V (\boldsymbol {e} _ {t _ {k}}) + \varepsilon_ {d}, \quad 0 <   \lambda <   1.\tag{21}
$$

This condition corresponds to the training objective of the diffusion phase in TDP, which aims to contract the tracking error at correction steps by accounting for the full system dynamics.

Lemma 1 (Streaming-phase bound). During streaming execution, the value function satisfies

$$
V (\boldsymbol {e} _ {t + 1}) \leq L _ {x} V (\boldsymbol {e} _ {t}) + L _ {u} \varepsilon_ {a} + \bar {w}.\tag{22}
$$

Proof. From the system dynamics,

$$
\boldsymbol {e} _ {t + 1} = g (\boldsymbol {o} _ {t}, \boldsymbol {u} _ {t}) + \boldsymbol {w} _ {t} - g (\boldsymbol {o} _ {t} ^ {*}, \boldsymbol {u} _ {t} ^ {*}).
$$

Applying Lipschitz continuity and Assumption 2 yields

$$
\| \pmb {e} _ {t + 1} \| \leq L _ {x} \| \pmb {e} _ {t} \| + L _ {u} \varepsilon_ {a} + \bar {w}.
$$

Substituting the definition of V proves the result.

The lemma shows that the streaming policy may introduce bounded tracking drift due to imitation error and disturbances, but the error growth remains bounded at each control step.

Theorem 1 (Practical stability of TDP). Suppose diffusion correction occurs every $H _ { a }$ steps and $L _ { x } < 1$ . Then the closedloop system under TDP satisfies the following properties:

(1) Uniform ultimate boundedness. The tracking error sequence is uniformly ultimately bounded.

(2) Bounded error within each diffusion cycle. During each correction cycle $t \in \{ t _ { k } , \ldots , t _ { k } + H _ { a } - 1 \}$ , the tracking error remains bounded as a function of the post-correction error.

Proof sketch. Lemma 1 shows that during streaming execution the value function grows at most linearly due to imitation error and disturbances. At correction instants, diffusion reduces the accumulated tracking error according to Assumption 4. Combining bounded growth with periodic contraction yields a stable affine recursion on the post-correction error, which implies uniform ultimate boundedness of the tracking error. Moreover, the streaming bound ensures that the error remains bounded throughout each diffusion cycle. □

In the ideal case where the imitation error, disturbance, and diffusion residual vanish $( \varepsilon _ { a } = 0 , \bar { w } = 0 ,$ and $\varepsilon _ { d } = 0 )$ , the equilibrium $e = 0$ is asymptotically stable and the tracking error satisfies lim $\mathbf { \Lambda } _ { t \to \infty } e _ { t } ~ = ~ 0$ . The full derivation of the stability bound is provided in Appendix.

## V. EXPERIMENTS

We first use a toy 1-D point-mass system to illustrate the effectiveness of action tube under external disturbances. We then evaluate TDP on the widely used Push-T task and three additional visual–tactile dexterous manipulation tasks in simulation, followed by two challenging real-world experiments that require reactive control under contact uncertainty and external disturbances.

## A. 1-D Point-mass System

We consider a 1-D point-mass system with single-integrator dynamics $x _ { t + 1 } = x _ { t } + u _ { t } \Delta t .$ . The goal is to use imitation learning to reproduce a demonstrated trajectory, as illustrated in Fig. 3. The horizontal axis denotes time, while the vertical axis shows the evolution of the state $x _ { t }$ . To highlight the difference between action chunk and action tube, we assume that the entire trajectory is executed within a single actionchunk horizon. The white curves depict the movement flow of the point mass under different initializations in both cases. Without loss of generality, we consider the example in which the system is initialized at the same state as the demonstration, and an external disturbance is injected at the midpoint of the trajectory, as indicated by the orange dashed line. This disturbance can be interpreted as either human perturbation or physical uncertainty.

Chunk-based approaches generate a fixed sequence of actions at the beginning of execution. As a result, after the disturbance occurs, the controller continues to apply the precomputed action sequence, ultimately driving the system to an incorrect final state. In contrast, the action-tube formulation employs a local feedback controller around the nominal trajectory, which actively steers the system back toward the desired path after the disturbance.

This toy example illustrates the importance of action tube for achieving reactive and robust control in the presence of external disturbances or contact uncertainty.

## B. Push-T

We further benchmark TDP on Push-T task against state-ofthe-art imitation learning methods, including Diffusion Policy (DP) [7] with 100 DDPM steps, DP with 10 Denoising Diffusion Implicit Models (DDIM) [38] steps, Flow Matching Policy [51], and Streaming Flow Policy (SFP) [17]. We also have one ablation study that removes the diffusion phase of TDP, namely TDP (streaming only). Following [17], we use the current robot configuration as a warm-start initialization for SFP, and also apply the same strategy to the TDP (streaming only) ablation, as it does not include an initial denoising stage. Push-T is a widely-used task to benchmark imitation learning approaches, and it is particularly suitable to be used in this work since it has nonlinear dynamics and physical uncertainty. For example, the friction coefficients are hard to be identified and varied with different objects. Following [7], we report the average score of the 5 best checkpoints and also the maximum score across all checkpoints. Moreover, there is another metric called steps to success, which describes the task completion time for the successful trials. This further evaluates the effectiveness of each action across all cases that ultimately lead to success. We report the average steps of the 5 best checkpoints and the minimum steps of the best checkpoint. Table II demonstrates the results for both state-based and image-based Push-T tasks, where state-based Push-T task uses low-dim state as the input, including 2-D position of the cylindrical pusher and 3-D pose of the T-shape block on planar space. Image-based push-T task use RGB image as the input. The actions are 2-D setpoints tracked by a PD controller. This task contains 200 demonstrations and 50 evaluation episodes.

As reported in Table II, TDP achieves the highest performance on both tasks while requiring the fewest steps, demonstrating the effectiveness of real-time reactions at each step given the current observation for task accomplishment. Moreover, the ablation results show that removing the diffusion phase degrades performance, highlighting the limitations of relying solely on local linearization without a diffusion-based correction that accounts for the full system dynamics. This observation is consistent with the results of SFP, which can also be interpreted as relying primarily on locally linearized dynamics without global correction. Compared to diffusionbased baselines such as DDPM and DDIM, which achieve competitive success rates but require more steps, TDP attains higher success rates with fewer steps, highlighting the effectiveness of step-wise reactions under contact uncertainty. In contrast, flow matching shows inferior performance, reflecting its relatively limited capacity to model complex contact-rich behaviors.

![](images/8b1c639040bc34aff11730a9c97447d85f65f093d8894c3f2a7210d4a0331bd9.jpg)  
Fig. 3: 1-D Point-Mass Example: Action-Tube Robustness to External Disturbances. We consider a 1-D point-mass system with a single demonstration to contrast action chunk and action tube. The x-axis denotes execution time and the y-axis denotes system state. Left: Nominal trajectory without disturbance, with color indicating progression along the trajectory from dark green (start) to light green (end). Middle: Action chunk execution. A disturbance is applied at the midpoint (orange dashed line), after which the system continues executing the precomputed action sequence (i.e., the 1-D velocity) without feedback, resulting in deviation from the target. Right: Action tube execution under the same disturbance. A local feedback controller drives the system back toward the nominal trajectory and ultimately reaches the target. Blue curves denote trajectories before disturbance, and red curves denote trajectories after disturbance.

We further conduct an ablation study on the number of DDIM inference steps used in the denoising phase for state-based Push-T task, as shown in Fig. 5. We conduct five trials, each consisting of 20 episodes, and report both the average and maximum scores. The results indicate that competitive performance can already be achieved with as few as two DDIM steps. This observation suggests that our method does not require highly accurate action generation during the denoising stage. Instead, the diffusion model only needs to produce an initial action that initializes execution while accounting for the full system dynamics, after which the streaming policy regulates deviations through closed-loop feedback. This property highlights a key advantage of the proposed tube-based formulation over conventional chunkbased methods that rely on precise action chunks. Moreover, reducing the number of DDIM steps substantially lowers inference latency, demonstrating the potential of TDP for realtime, high-frequency control.

## C. Visual–Tactile Manipulation in a Virtual Environment

We evaluate our method on three additional visual–tactile dexterous manipulation tasks in a simulation environment built with Unreal Engine. The simulator integrates a real-time finite element solver through a native plugin, enabling physically realistic interactions between rigid and compliant objects. By leveraging reduced-order models, the solver efficiently simulates contact dynamics at interactive rates while providing detailed per-contact state information for dexterous manipulation [42, 55, 5, 56]. The simulated system consists of a UR5 robotic arm and a 20-DoF Tesollo DG5F dexterous hand equipped with a dense tactile array containing 768 sensing elements. Each element measures three-axis forces, resulting in a tactile observation of 2304 dimensions. Visual observations are obtained from two RGB cameras: one mounted above the workspace and another attached to the robot wrist. To collect demonstrations, we use a VR-based teleoperation interface with a Meta Quest 3 headset, as illustrated in Fig. 4. The tracked human hand pose determines the pose of the robot end-effector, while the dexterous hand is controlled through fingertip retargeting. Specifically, the human fingertip positions are scaled to match the kinematic workspace of the DG5F hand. The corresponding finger joint targets are then computed using RelaxedIK [33, 43]. All simulation experiments are performed on a Lenovo P8 desktop equipped with an NVIDIA RTX 4080 GPU, responsible for robot control, visual–tactile sensing, and policy inference.

![](images/db4220b9a9173292968da4e9d52139cf21aae0d5d3833860ded23fdf76047106.jpg)  
Fig. 4: Teleoperation system. Human demonstrations are collected through teleoperation in both simulation and the real world. In simulation, a Meta Quest 3 headset is used for VRbased hand tracking, while in the physical setup an OptiTrack motion-capture system provides mocap-based hand tracking. The captured hand pose and dorsum pose are then processed by the same retargeting module and controllers to actuate the dexterous hand and robot arm.

<table><tr><td rowspan="2"><img src="images/e1a7493afe3e26cd62ebaeb0e6a9e753f8c13624d8384cdb9d4e9e8bc5844146.jpg"/></td><td colspan="2">Push-T with state input</td><td colspan="2">Push-T with image input</td></tr><tr><td>Avg/Max scores↑</td><td>Avg/Min steps to success↓</td><td>Avg/Max scores↑</td><td>Avg/Min steps to success↓</td></tr><tr><td>DP (DDPM) [7]</td><td>93.2%/93.8%</td><td>125.9/128.5</td><td>76.0%/77.5%</td><td>114.7/107.9</td></tr><tr><td>DP (DDIM) [7]</td><td>91.8%/93.9%</td><td>124.6/117.1</td><td>68.5%/69.0%</td><td>107.3/104.8</td></tr><tr><td>Flow Matching [51]</td><td>91.8%/93.3%</td><td>122.8/115.9</td><td>65.3%/68.3%</td><td>108.9/102.5</td></tr><tr><td>SFP [17]</td><td>88.6%/89.7%</td><td>124.4/116.1</td><td>73.3%/74.8%</td><td>112/108.6</td></tr><tr><td>TDP (streaming only)</td><td>84.3%/85.0%</td><td>109.6/108.1</td><td>77.1%/77.7%</td><td>117.5/96.3</td></tr><tr><td>TDP</td><td>96.1%/96.9%</td><td>106.2/105.1</td><td>82.0%/85.6%</td><td>103.5/91.3</td></tr></table>

TABLE II: Virtual push-T results with state and image inputs. Our method (green) is compared against baselines (orange) and ablation variants (blue).

<table><tr><td rowspan="2"><img src="images/d8a48a189d20f1c394e7e80c13ed4a768b49141af9d7f5095cd4300185c1a422.jpg"/></td><td colspan="3">Stable Grasping</td><td colspan="2">On-table Reorientation</td></tr><tr><td>Success rate (Grasp)↑</td><td>Success rate (Stable grasp)↑</td><td>Steps to success↓</td><td>Success rate↑</td><td>Steps to success↓</td></tr><tr><td>DP (DDIM) [7]</td><td>88%</td><td>60%</td><td>28.0 ± 3.70</td><td>82%</td><td>41.7 ± 8.51</td></tr><tr><td>SFP [17]</td><td>78%</td><td>72%</td><td>29.6 ± 4.75</td><td>80%</td><td>43.1 ± 12.9</td></tr><tr><td>TDP (streaming only)</td><td>72%</td><td>44%</td><td>25.5 ± 5.1</td><td>76%</td><td>60.6 ± 25.3</td></tr><tr><td>TDP</td><td>96%</td><td>88%</td><td>27.8 ± 3.06</td><td>90%</td><td>40.0 ± 15.3</td></tr></table>

TABLE III: Comparison on virtual stable grasping and on-table reorientation. Our method (green) is compared against baselines (orange) and ablation variants (blue).

![](images/b844fe90ce5a75001c5ec4ed6c5e33df62176c49d76e477cbbc2ea1035e8a644.jpg)  
Fig. 5: Ablation study on the number of DDIM denoising steps within TDP for the virtual Push-T task. We vary the number of DDIM inference steps used in the denoising phase of TDP. The results show that TDP maintains competitive performance with as few as two denoising steps, highlighting its suitability for high-frequency control.

The first task, stable grasping, requires the robot to grasp a mustard bottle and maintain a secure hold. Performance is evaluated using grasp success rate, stable grasp success rate, and steps to success, where a stable grasp requires holding the object for at least three seconds. The second task, on-table reorientation, involves rotating a sugar box by 90 degrees via in-hand manipulation. The third task, dish cleaning, requires cleaning a dirty dish using a sponge. Task performance is measured by the percentage of the dirty region cleaned. We collect 150 demonstrations for each task and evaluate each method over 50 episodes.

<table><tr><td rowspan="2"><img src="images/da3df21490768452963a2689aad6b675cb4588b2a71f51967f0532a8287942b4.jpg"/></td><td rowspan="2">Score↑</td><td colspan="2">Latency</td></tr><tr><td>Denoising (↓)</td><td>Streaming (↓)</td></tr><tr><td>DP (DDIM-10)</td><td>98.4%</td><td colspan="2">72.9 ms</td></tr><tr><td>DP (DDIM-2)</td><td>5.3%</td><td colspan="2">17.1 ms</td></tr><tr><td>TDP</td><td>98.0%</td><td>13.3 ms</td><td>6.50 ms</td></tr></table>

TABLE IV: Comparison of DP and TDP on the dishcleaning task. TDP achieves comparable success rate while significantly reducing inference latency.

We first compare our approach against state-of-the-art baselines and conduct ablation studies on the stable grasping and on-table reorientation tasks, with results summarized in Table III. Across both tasks, our method consistently outperforms all baselines in terms of success rate while requiring comparable or fewer steps to reach successful outcomes. In the stable grasping task, our approach achieves a grasp success rate of 96% and a stable grasp success rate of 88%, substantially exceeding both DP and SFP, while maintaining efficient execution. Similar improvements are observed in the on-table reorientation task, where our method attains the highest success rate (90%) and reduces the number of steps compared to prior approaches. These results demonstrate the effectiveness of the proposed tube-based formulation in leveraging real-time visual–tactile feedback for robust and efficient manipulation. We further conduct an ablation study by removing the diffusion phase and relying solely on the streaming controller. In this setting, success rates drop significantly for both tasks. For on-table reorientation, small errors tend to accumulate and often require multiple retries, leading to a higher number of steps to success despite eventual task completion. Interestingly, the streaming-only variant achieves the lowest number of steps in the stable grasping task, which makes sense since this metric only accounts for completed episodes. In contrast to reorientation, failed grasp attempts typically cause the object to fall, making retries infeasible within the same episode. As a result, the steps-to-success values for grasping remain on a similar scale (and are even slightly better) than those of the full TDP method, as this metric mainly reflects the simplest successful scenarios. This further highlights the efficiency of step-wise reactive control for contact-rich manipulation, as this variant preserves a fully reactive formulation with fresh observations at every time step.

![](images/c994f4a58f8c6da3dffdda4ae5da7b5d9f785769f147964e7c1061ea5a7219fb.jpg)  
(a) Grasping

![](images/fc977287b36d784b5b8844b2cdde57d0f4fedc66786c1c9090fc3471ff8cdd7a.jpg)  
(b) Stable grasping  
Fig. 6: Ablation study on DDIM inference steps. We compare TDP and DP under varying numbers of DDIM inference steps on both grasping and stable grasping tasks. TDP consistently performs well with fewer inference steps than DP, demonstrating its ability to reduce control latency in practice. Furthermore, with the same number of inference steps, TDP achieves higher success rates owing to its step-wise reactive control mechanism.

Furthermore, we perform an ablation study on the number of DDIM inference steps to evaluate execution efficiency on the stable grasping task (Fig. 6). With a single DDIM step, both TDP and DP fail to complete the task. As the number of inference steps increases, TDP exhibits a rapid performance gain with only two to three steps, whereas DP remains at a 0% success rate and requires substantially more denoising steps (e.g., 5–10) to achieve similar performance. This indicates that the step-wise feedback regulation of the action tube can effectively compensate for coarse diffusion outputs, allowing TDP to attain high success rates with significantly fewer denoising steps and, consequently, reduced diffusion-stage inference latency for high-frequency, real-time control.

Based on this observation, we further evaluate both methods on the dish cleaning task, with results shown in Table IV.

While DP with 10 DDIM steps achieves the highest cleaning score, it leads to significant control latency due to the long inference time. In contrast, TDP requires only 2 DDIM steps, making it possible to run at approximately 75 Hz during the denoising phase and up to 150 Hz during the streaming phase. When DP is constrained to a comparable inference budget (i.e., 2 DDIM steps), its performance drops significantly. This result highlights the capability of TDP for real-time, high-frequency control.

## D. Real-world experiments

We further validate TDP on two real-world visual–tactile dexterous manipulation tasks: jar opening and on-table reorientation, as illustrated in Fig. 1. These tasks involve contactrich interactions, pose uncertainty, and frequent variations in hand–object contact, posing significant challenges for policies relying on open-loop execution.

All physical experiments are conducted using a Franka Research 3 robotic arm equipped with an Allegro V5 dexterous hand. Visual observations are provided by an Intel RealSense D405 camera for the wrist-mounted view and a D435F camera for the global view. In addition, Digit360 tactile sensors are attached to the fingertips to capture tactile feedback during manipulation. The system runs on two workstations. A Lenovo P620 desktop (NVIDIA RTX 3080 GPU) is used for low-level robot control and real-time visual–tactile sensing processing, while a Lenovo P8 desktop (NVIDIA RTX 4080 GPU) is dedicated to policy inference. To collect demonstrations, we employ a teleoperation system based on an OptiTrack motioncapture setup together with an instrumented glove for hand tracking, as illustrated in Fig. 4. The pose of the robot endeffector is determined by tracking the pose of the human hand. For hand teleoperation, we adopt a fingertip retargeting strategy: fingertip positions estimated from the glove are mapped to the workspace of the Allegro hand while ignoring the human pinky finger. We then solve the inverse kinematics using RelaxedIK [33, 43] to obtain the desired finger joint targets.

Success Case (TDP): reactive action  
![](images/240fd9fe7d61eecc26a7ea1470f162ccff0811c444308cd822ec7c22bba6cfef.jpg)  
Fig. 7: On-table manipulation task under external disturbance. The disturbance is introduced by bending the thumb fingers of the Allegro hand. Tube Diffusion Policy (TDP) reacts quickly to the perturbation, while Diffusion Policy (DP) exhibits a delayed response and the fingers become stuck, resulting in task failure.

Success Case (TDP): reactive action  
![](images/267005ff26229effd0adb295dde0b8252f03bae6929fdaaec386a958aae3e951.jpg)  
Fig. 8: Jar-opening task under external disturbance. We introduce disturbance by shifting the jar’s position. Tube Diffusion Policy (TDP) quickly adapts to the new position and continues opening the jar, whereas Diffusion Policy (DP) blindly executes the pre-generated action sequence, resulting in slow response to the disturbance.

<table><tr><td rowspan="2"><img src="images/404cadfb6a29eb5a83677534c464637fc3f878a6d076bf65235b659d2069dbf5.jpg"/></td><td rowspan="2">On-table ReorientationSuccess rate (↑)</td><td rowspan="2">Jar OpeningSuccess rate (↑)</td><td colspan="2">Latency</td></tr><tr><td>Denoising (↓)</td><td>Streaming (↓)</td></tr><tr><td>DP</td><td>60%</td><td>84%</td><td colspan="2">0.037s</td></tr><tr><td>TDP</td><td>96%</td><td>96%</td><td>0.008s</td><td>0.003s</td></tr></table>

TABLE V: Comparison of DP and TDP on Two Physical Experiments. TDP achieves higher success rate with significantly lower inference latency.

Success Case (TDP with 3 DDIM steps)  
![](images/1114c221551dc15e558b86c7c15593ecefc462be63614938862033de15517123.jpg)  
(a) On-table reorientation task with 3 DDIM steps.

Success Case (TDP with 3 DDIM steps)  
![](images/584b58a246c8efa5187c813ebf8dc5e57616960ec306bb72459e01210848fe95.jpg)  
(b) Jar-opening task with 3 DDIM steps.  
Fig. 9: Real-world manipulation with few DDIM inference steps. Tube Diffusion Policy (TDP) maintains strong performance with only three DDIM inference steps, enabling high-frequency control in the real world. In contrast, Diffusion Policy (DP) produces inaccurate behaviors under the same inference budget, indicating its reliance on more DDIM steps for stable execution and consequently higher control latency.

We first conduct 25 evaluation episodes for each task using both TDP and DP. The quantitative results are summarized in Table V. Benefiting from its closed-loop reactive control, TDP achieves significantly higher success rates on both tasks. Moreover, TDP allows the use of fewer DDIM inference steps thanks to the tube-based correction mechanism applied at each timestep. In our experiments, we use only 3 DDIM steps during denoising, which results in substantially lower inference latency and enables control frequencies above 100 Hz, approximately four times faster than DP (∼ 25 Hz).

To further examine this property, we also evaluate DP using the same number of DDIM inference steps for high-frequency control. As shown in Fig. 9, DP with only 3 DDIM steps produces inaccurate behaviors and fails to complete the tasks, indicating that DP requires more inference steps for stable action generation.

We additionally evaluate the step-wise reactive capability of TDP by introducing external disturbances during task execution. Fig. 7 and Fig. 8 show representative keyframes before and after disturbances. In the on-table reorientation task, we introduce perturbations by bending the thumb finger of the Allegro hand. As shown in Fig. 7, TDP quickly reacts to the perturbation and recovers to a stable configuration. In contrast, DP continues executing the pre-generated action sequence, causing the fingers to become stuck on the object and resulting in task failure. In the jar-opening task, we introduce disturbance by shifting the position of the jar during execution. As shown in Fig. 8, TDP rapidly tracks the updated jar pose and continues the opening motion. In contrast, DP blindly executes the pre-generated action chunk, failing to adapt to the changed object position.

Overall, the physical experiments demonstrate that TDP consistently achieves higher success rates and more stable manipulation behaviors than DP. When unexpected contact changes or disturbances occur, TDP can react immediately and adjust the hand configuration online. In contrast, DP tends to continue executing pre-generated action sequences in an openloop manner, which often leads to failure.

Importantly, TDP supports inference with very few DDIM steps, significantly reducing latency and enabling fast, highfrequency control. Combined with its per-step reactive control within the learned action tube, this makes TDP particularly well suited for real-world contact-rich dexterous manipulation.

## VI. LIMITATIONS AND FUTURE WORK

While we have demonstrated the effectiveness of our approach in both simulation and real-world settings, several limitations remain and point to promising directions for future work. First, we simply concatenate visual, tactile and proprioceptive perception as multi-modal observation, which neglects the inherent sparsity and asynchronous nature of tactile sensing. Future work could explore more structured or learned tactile representations to better exploit contact information. Second, although tube-based formulation allows the use of a small number of DDIM inference steps to reduce latency, further acceleration could be achieved by leveraging recent advances in diffusion model acceleration techniques, such as one-step or distillation-based denoising methods [40, 32, 9]. Third, while this work focuses on improving standard diffusion policies, the proposed tube-based formulation is broadly applicable to many existing chunkbased or open-loop policy learning methods. Extending this framework to other generative or sequence-based control approaches remains an important direction for future research. For example, most Vision–Language–Action (VLA) models [2, 16, 3] rely on flow-based action chunking, which could benefit from integrating the action tube formulation to enable fast and reactive execution. Overall, we believe this approach provides a general pathway for mitigating the limitations of open-loop chunk execution and improving robustness in contact-rich manipulation.

## VII. CONCLUSION

In this work, we present Tube Diffusion Policy (TDP), a reactive visual-tactile policy learning framework that augments diffusion-based imitation learning with feedback regulation during action execution. In contrast to conventional action chunking approaches that execute pre-generated action sequences in an open-loop manner, TDP constructs an action tube that enables step-wise feedback control conditioned on fresh observations. This structure allows the policy to react to unexpected contacts and external disturbances during execution, which is particularly important for contact-rich dexterous manipulation.

We evaluate TDP on three visual–tactile manipulation tasks in simulation and two tasks in the real world. Experimental results demonstrate that the proposed approach consistently improves task success rates and robustness compared with state-of-the-art imitation learning methods. In real-world experiments, TDP further demonstrates rapid recovery from external disturbances and unexpected contact variations during execution. Moreover, TDP does not require highly accurate generation of full action sequences. Instead, the diffusion phase only needs to produce a coarse corrective action that accounts for the full system dynamics, after which actions are generated by streaming conditional feedback flows. This design substantially reduces inference latency and enables high-frequency control in practice.

Overall, this work provides a principled bridge between generative policy learning and feedback control, addressing a key limitation of existing chunk-based methods—open-loop execution under uncertainty. We believe the proposed tubebased formulation offers a promising direction for enabling robust, reactive, and high-frequency control in visual-tactile contact-rich manipulation.

## REFERENCES

[1] Maria Bauza, Antonia Bronars, Yifan Hou, Ian Taylor, Nikhil Chavan-Dafle, and Alberto Rodriguez. Simple, a visuotactile method learned in simulation to precisely pick, localize, regrasp, and place objects. Science Robotics, 9(91):eadi8808, 2024.

[2] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. π<sub>0</sub>: A visionlanguage-action flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024.

[3] Kevin Black, Manuel Y Galliker, and Sergey Levine. Real-time execution of action chunking flow policies. arXiv preprint arXiv:2506.07339, 2025.

[4] Roberto Calandra, Andrew Owens, Dinesh Jayaraman, Justin Lin, Wenzhen Yuan, Jitendra Malik, Edward H Adelson, and Sergey Levine. More than a feeling: Learning to grasp and regrasp using vision and touch. IEEE Robotics and Automation Letters (RA-L), 3(4): 3300–3307, 2018.

[5] Yue Chang, Peter Yichen Chen, Zhecheng Wang, Maurizio M. Chiaramonte, Kevin Carlberg, and Eitan Grinspun. Licrom: Linear-subspace continuous reduced order modeling with neural fields. In SIGGRAPH Asia 2023 Conference Papers, pages 1–12, 2023.

[6] Ricky TQ Chen, Yulia Rubanova, Jesse Bettencourt, and David K Duvenaud. Neural ordinary differential equations. Advances in Neural Information Processing Systems (NIPS), 31, 2018.

[7] Cheng Chi, Siyuan Feng, Yilun Du, Zhenjia Xu, Eric Cousineau, Benjamin Burchfiel, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. In Proceedings of Robotics: Science and Systems (RSS), 2023.

[8] Ana Lucia D Franco, Henri Bourles, Edson R De Pieri,\` and Herve Guillard. Robust nonlinear control associating robust feedback linearization and $H _ { \infty }$ control. IEEE Transactions on Automatic Control, 51(7):1200–1207, 2006.

[9] Kevin Frans, Danijar Hafner, Sergey Levine, and Pieter Abbeel. One step diffusion via shortcut models. In Proc. Intl Conf. on Learning Representations (ICLR), 2025.

[10] Michael Green and David JN Limebeer. Linear robust control. Courier Corporation, 2012.

[11] Johanna Hansen, Francois Hogan, Dmitriy Rivkin, David Meger, Michael Jenkin, and Gregory Dudek. Visuotactile-rl: Learning multimodal manipulation policies with deep reinforcement learning. In Proc. IEEE Intl Conf. on Robotics and Automation (ICRA), pages 8298–8304, 2022.

[12] Karol Hausman, Jost Tobias Springenberg, Ziyu Wang, Nicolas Heess, and Martin Riedmiller. Learning an embedding space for transferable robot skills. In Proc. Intl Conf. on Learning Representations (ICLR), 2018.

[13] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 770–778, 2016.

[14] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems (NIPS), 33:6840–6851, 2020.

[15] Auke Jan Ijspeert, Jun Nakanishi, Heiko Hoffmann, Peter Pastor, and Stefan Schaal. Dynamical movement

primitives: learning attractor models for motor behaviors. Neural computation, 25(2):328–373, 2013.

[16] Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, et al. π : a vision-language-action model with open-world generalization. arXiv preprint arXiv:2504.16054, 2025.

[17] Sunshine Jiang, Xiaolin Fang, Nicholas Roy, Tomas´ Lozano-Perez, Leslie Pack Kaelbling, and Siddharth An-´ cha. Streaming flow policy: Simplifying diffusion/flowmatching policies by treating action trajectories as flow trajectories. In Conference on Robot Learning (CoRL), 2025.

[18] S Mohammad Khansari-Zadeh and Aude Billard. Imitation learning of globally stable non-linear point-to-point robot motions using nonlinear programming. In Proc. IEEE/RSJ Intl Conf. on Intelligent Robots and Systems (IROS), pages 2676–2683, 2010.

[19] S Mohammad Khansari-Zadeh and Aude Billard. Learning stable nonlinear dynamical systems with gaussian mixture models. IEEE Transactions on Robotics, 27(5): 943–957, 2011.

[20] Patrick Kidger, James Morrill, James Foster, and Terry Lyons. Neural controlled differential equations for irregular time series. Advances in Neural Information Processing Systems (NIPS), 33:6696–6707, 2020.

[21] Basil Kouvaritakis and Mark Cannon. Model predictive control. Switzerland: Springer International Publishing, 38(13-56):7, 2016.

[22] Lucy Lai, Ann Zixiang Huang, and Samuel J Gershman. Action chunking as policy compression. PsyArXiv, 2022.

[23] Wilbur Langson, Ioannis Chryssochoos, SV Rakovic, and´ David Q Mayne. Robust model predictive control using tubes. Automatica, 40(1):125–133, 2004.

[24] Michelle A Lee, Yuke Zhu, Peter Zachares, Matthew Tan, Krishnan Srinivasan, Silvio Savarese, Li Fei-Fei, Animesh Garg, and Jeannette Bohg. Making sense of vision and touch: Learning multimodal representations for contact-rich tasks. IEEE Transactions on Robotics, 36(3):582–596, 2020.

[25] Qiyang Li, Zhiyuan Zhou, and Sergey Levine. Reinforcement learning with action chunking. In Advances in Neural Information Processing Systems (NIPS), 2025.

[26] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. Proc. Intl Conf. on Learning Representations (ICLR), 2023.

[27] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. Proc. Intl Conf. on Learning Representations (ICLR), 2023.

[28] Yuejiang Liu, Jubayer Ibn Hamid, Annie Xie, Yoonho Lee, Max Du, and Chelsea Finn. Bidirectional decoding: Improving action chunking via guided test-time sampling. Proc. Intl Conf. on Learning Representations (ICLR), 2025.

[29] Corey Lynch, Mohi Khansari, Ted Xiao, Vikash Kumar, Jonathan Tompson, Sergey Levine, and Pierre Sermanet.

Learning latent plans from play. In Conference on Robot Learning (CoRL), 2019.

[30] David Q Mayne, Erric C Kerrigan, EJ Van Wyk, and Paola Falugi. Tube-based robust nonlinear model predictive control. International journal of robust and nonlinear control, 21(11):1341–1353, 2011.

[31] Ethan Perez, Florian Strub, Harm De Vries, Vincent Dumoulin, and Aaron Courville. Film: Visual reasoning with a general conditioning layer. In Proc. AAAI Conference on Artificial Intelligence, volume 32, 2018.

[32] Aaditya Prasad, Kevin Lin, Jimmy Wu, Linqi Zhou, and Jeannette Bohg. Consistency policy: Accelerated visuomotor policies via consistency distillation. In Proc. Robotics: Science and Systems (R:SS), 2024.

[33] Daniel Rakita, Bilge Mutlu, and Michael Gleicher. Relaxedik: Real-time synthesis of accurate and feasible robot arm motion. In Proc. Robotics: Science and Systems (R:SS), volume 14, pages 26–30, 2018.

[34] Yulia Rubanova, Ricky TQ Chen, and David K Duvenaud. Latent ordinary differential equations for irregularly-sampled time series. Advances in Neural Information Processing Systems (NIPS), 32, 2019.

[35] Stefan Schaal, Auke Ijspeert, and Aude Billard. Computational approaches to motor learning by imitation. Philosophical Transactions of the Royal Society of London. Series B: Biological Sciences, 358(1431):537–547, 2003.

[36] Yuri Shtessel, Christopher Edwards, Leonid Fridman, and Arie Levant. Sliding mode control and observation, volume 10. Springer, 2014.

[37] Andreas Sochopoulos, Michael Gienger, and Sethu Vijayakumar. Learning deep dynamical systems using stable neural odes. In Proc. IEEE/RSJ Intl Conf. on Intelligent Robots and Systems (IROS), pages 11163– 11170, 2024.

[38] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In Proc. Intl Conf. on Learning Representations (ICLR), 2021.

[39] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Scorebased generative modeling through stochastic differential equations. Proc. Intl Conf. on Learning Representations (ICLR), 2021.

[40] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. In Proc. Intl Conf. on Machine Learning (ICML), 2023.

[41] Sudharshan Suresh, Haozhi Qi, Tingfan Wu, Taosha Fan, Luis Pineda, Mike Lambeta, Jitendra Malik, Mrinal Kalakrishnan, Roberto Calandra, Michael Kaess, et al. Neuralfeels with neural fields: Visuotactile perception for in-hand manipulation. Science Robotics, 9(96):eadl0628, 2024.

[42] Yutian Tao, Maurizio Chiaramonte, and Pablo Fernandez. Interpolated adaptive linear reduced order modeling for deformation dynamics. arXiv preprint arXiv:2509.25392, 2025.

[43] Yeping Wang, Pragathi Praveena, Daniel Rakita, and Michael Gleicher. Rangedik: An optimization-based robot motion generation method for ranged-goal tasks.

In Proc. IEEE Intl Conf. on Robotics and Automation (ICRA), pages 9700–9706, 2023.

[44] Han Xue, Jieji Ren, Wendi Chen, Gu Zhang, Yuan Fang, Guoying Gu, Huazhe Xu, and Cewu Lu. Reactive diffusion policy: Slow-fast visual-tactile policy learning for contact-rich manipulation. In Proc. Robotics: Science and Systems (R:SS), 2025.

[45] T. Xue, H. Girgin, T. Lembono, and S. Calinon. Demonstration-guided optimal control for long-term nonprehensile planar manipulation. In Proc. IEEE Intl Conf. on Robotics and Automation (ICRA), pages 4999–5005, 2023.

[46] T. Xue, A. Razmjoo, S. Shetty, and S. Calinon. Logicskill programming: An optimization-based approach to sequential skill planning. 2024.

[47] T. Xue, A. Razmjoo, S. Shetty, and S. Calinon. Robust contact-rich manipulation through implicit motor adaptation. International Journal of Robotics Research (IJRR), 2025.

[48] Wenzhen Yuan, Chenzhuo Zhu, Andrew Owens, Mandayam A Srinivasan, and Edward H Adelson. Shapeindependent hardness estimation using deep learning and a gelsight tactile sensor. In Proc. IEEE Intl Conf. on Robotics and Automation (ICRA), pages 951–958, 2017.

[49] Yanjie Ze, Gu Zhang, Kangning Zhang, Chenyuan Hu, Muhan Wang, and Huazhe Xu. 3d diffusion policy: Generalizable visuomotor policy learning via simple 3d representations. In Proc. Robotics: Science and Systems (R:SS), 2024.

[50] Fan Zhang and Yiannis Demiris. Visual-tactile learning of garment unfolding for robot-assisted dressing. IEEE Robotics and Automation Letters (RA-L), 8(9):5512– 5519, 2023.

[51] Fan Zhang and Michael Gienger. Affordance-based robot manipulation with flow matching. arXiv preprint arXiv:2409.01083, 2024.

[52] Tony Zhao, Vikash Kumar, Sergey Levine, and Chelsea Finn. Learning fine-grained bimanual manipulation with low-cost hardware. In Proc. Robotics: Science and Systems (R:SS), 2023.

[53] Kemin Zhou and John Comstock Doyle. Essentials of robust control, volume 104. Prentice hall Upper Saddle River, NJ, 1998.

[54] Yi Zhou, Connelly Barnes, Jingwan Lu, Jimei Yang, and Hao Li. On the continuity of rotation representations in neural networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5745–5753, 2019.

[55] Yifeng Zhu, Abhishek Joshi, Peter Stone, and Yuke Zhu. Viola: Imitation learning for vision-based manipulation with object proposal priors. arXiv preprint arXiv:2210.11339, 2022. doi: 10.48550/arXiv.2210. 11339.

[56] Zeshun Zong, Xuan Li, Minchen Li, Maurizio M. Chiaramonte, Wojciech Matusik, Eitan Grinspun, Kevin Carlberg, Chenfanfu Jiang, and Peter Yichen Chen. Neural stress fields for reduced-order elastoplasticity and fracture. In SIGGRAPH Asia 2023 Conference Papers, pages

## APPENDIX

## A. Stability Analysis of Tube Diffusion Policy

This section provides a more detailed analysis and formal proof of the stability properties of Tube Diffusion Policy (TDP). TDP decomposes control into two complementary mechanisms: diffusion correction, which periodically contracts the accumulated drift, and streaming imitation, which provides high-frequency reactive control but introduces bounded drift. Together, these mechanisms form a hybrid control system that admits a practical stability guarantee.

1) Problem Setup: Consider the discrete-time nonlinear system

$$
\boldsymbol {o} _ {t + 1} = g (\boldsymbol {o} _ {t}, \boldsymbol {u} _ {t}) + \boldsymbol {w} _ {t},\tag{23}
$$

where ${ \pmb O } _ { t } \in \mathbb { R } ^ { n _ { o } }$ is the system state, $\mathbf { \boldsymbol { u } } _ { t } \in \mathbb { R } ^ { n _ { u } }$ is the control input, and ${ \pmb w } _ { t }$ is a bounded disturbance.

We assume access to a reference demonstration trajectory

$$
(\pmb {o} _ {t} ^ {*}, \pmb {u} _ {t} ^ {*})\tag{24}
$$

that satisfies the nominal dynamics

$$
\pmb {o} _ {t + 1} ^ {*} = g (\pmb {o} _ {t} ^ {*}, \pmb {u} _ {t} ^ {*}).\tag{25}
$$

Tube Diffusion Policy consists of two components:

a) Diffusion correction: At correction instants $t _ { k }$ , a diffusion model produces a corrective action that compensates for the drift accumulated during streaming execution.

b) Streaming flow policy: A learned policy generates actions sequentially,

$$
\pmb {u} _ {t + 1} = \pmb {u} _ {t} + v _ {\theta} (\pmb {u}, t _ {1} = 0, t _ {2} | h) \cdot \Delta t,\tag{26}
$$

and is trained to imitate the demonstration action trajectory.

We define the tracking error

$$
\boldsymbol {e} _ {t} := \boldsymbol {o} _ {t} - \boldsymbol {o} _ {t} ^ {*}.\tag{27}
$$

2) Assumptions: To analyze the stability of Tube Diffusion Policy, we introduce the following assumptions. Assumptions 1 and 3 characterize properties of the system dynamics and disturbances, while Assumptions 2 and 4 describe the desired behavior of the learned streaming and diffusion policies induced by training.

Assumption 1 (Lipschitz dynamics). There exist constants $L _ { x } \ > \ 0$ and $L _ { u } \ > \ 0$ such that for all admissible states and actions,

$$
\left\| g \left(\boldsymbol {o} _ {1}, \boldsymbol {u} _ {1}\right) - g \left(\boldsymbol {o} _ {2}, \boldsymbol {u} _ {2}\right) \right\| \leq L _ {x} \left\| \boldsymbol {o} _ {1} - \boldsymbol {o} _ {2} \right\| + L _ {u} \left\| \boldsymbol {u} _ {1} - \boldsymbol {u} _ {2} \right\|.\tag{28}
$$

The Lipschitz condition bounds the sensitivity of the system dynamics to state and action deviations, enabling tractable propagation of tracking errors.

Assumption 2 (Streaming imitation accuracy). During streaming execution, the learned policy approximately imitates the reference actions, resulting in a bounded action error:

$$
\| \pmb {u} _ {t} - \pmb {u} _ {t} ^ {*} \| \leq \varepsilon_ {a}.\tag{29}
$$

This condition reflects the training objective of the streaming phase and is expected to hold when the policy accurately reproduces the demonstrated actions.

Assumption 3 (Bounded disturbance). The external disturbance is bounded:

$$
\left\| \boldsymbol {w} _ {t} \right\| \leq \bar {w}.\tag{30}
$$

Assumption 4 (Diffusion correction property). At each correction instant $t _ { k } ,$ , the diffusion module produces a corrective action that contracts the tracking error:

$$
\left\| \boldsymbol {e} _ {t _ {k} + 1} \right\| \leq \lambda \left\| \boldsymbol {e} _ {t _ {k}} \right\| + \varepsilon_ {d}, \quad 0 \leq \lambda <   1.\tag{31}
$$

This condition corresponds to the training objective of the diffusion phase, which seeks to reduce the accumulated tracking error at correction steps by accounting for the full system dynamics.

3) Streaming Phase Error Dynamics: From (23), (25), and (27), the error dynamics during the streaming phase satisfy

$$
\boldsymbol {e} _ {t + 1} = g (\boldsymbol {o} _ {t}, \boldsymbol {u} _ {t}) + \boldsymbol {w} _ {t} - g (\boldsymbol {o} _ {t} ^ {*}, \boldsymbol {u} _ {t} ^ {*})\tag{32}
$$

$$
= g (\pmb {o} _ {t}, \pmb {u} _ {t}) - g (\pmb {o} _ {t} ^ {*}, \pmb {u} _ {t} ^ {*}) + \pmb {w} _ {t}.\tag{33}
$$

Applying Assumption 1 gives

$$
\left\| \boldsymbol {e} _ {t + 1} \right\| \leq L _ {x} \left\| \boldsymbol {e} _ {t} \right\| + L _ {u} \left\| \boldsymbol {u} _ {t} - \boldsymbol {u} _ {t} ^ {*} \right\| + \left\| \boldsymbol {w} _ {t} \right\|.\tag{34}
$$

Using Assumptions 2 and 3,

$$
\left\| \boldsymbol {e} _ {t + 1} \right\| \leq L _ {x} \left\| \boldsymbol {e} _ {t} \right\| + c, \quad c := L _ {u} \varepsilon_ {a} + \bar {w}.\tag{35}
$$

4) Lyapunov Function for the Error Dynamics: Consider the Lyapunov candidate

$$
V (\boldsymbol {e} _ {t}) := \| \boldsymbol {e} _ {t} \|.\tag{36}
$$

This function is positive definite and radially unbounded with respect to the tracking error $e _ { t }$ . Then (35) implies

$$
V (\boldsymbol {e} _ {t + 1}) \leq L _ {x} V (\boldsymbol {e} _ {t}) + c.\tag{37}
$$

Thus, the streaming phase is input-to-state stable with respect to the combined input $c ,$ although it does not necessarily contract to zero in the absence of diffusion correction.

Lemma 1 (Streaming-phase ISS bound): For any integer $j \geq 0$ , the tracking error during the streaming phase satisfies

$$
V (\boldsymbol {e} _ {t + j}) \leq L _ {x} ^ {j} V (\boldsymbol {e} _ {t}) + \sum_ {i = 0} ^ {j - 1} L _ {x} ^ {i} c.\tag{38}
$$

If $L _ { x } < 1$ , then

$$
V (\pmb {e} _ {t + j}) \leq L _ {x} ^ {j} V (\pmb {e} _ {t}) + \frac {1 - L _ {x} ^ {j}}{1 - L _ {x}} c.\tag{39}
$$

Proof: The result follows by recursively unrolling (37). When $L _ { x } < 1$ , the geometric series identity

$$
\sum_ {i = 0} ^ {j - 1} L _ {x} ^ {i} = \frac {1 - L _ {x} ^ {j}}{1 - L _ {x}}
$$

yields (39).

5) Error Evolution Across Correction Cycles: The diffusion correction is applied every $H _ { a }$ steps. Let $z _ { k } : = V ( e _ { t _ { k } } )$ denote the tracking error immediately after the diffusion correction at time $t _ { k }$

After the correction at $t _ { k }$ , the system evolves under the streaming controller for the next $H _ { a } \mathrm { ~ - ~ } 1$ steps, from $t _ { k } + 1$ to $t _ { k } + H _ { a } - 1$ . By Lemma 1, the error at time $t _ { k } + H _ { a } - 1$ satisfies

$$
V (\pmb {e} _ {t _ {k} + H _ {a} - 1}) \leq L _ {x} ^ {H _ {a} - 1} z _ {k} + \frac {1 - L _ {x} ^ {H _ {a} - 1}}{1 - L _ {x}} c.\tag{40}
$$

At time $t _ { k + 1 } = t _ { k } + H _ { a }$ , a new diffusion correction is applied. Using Assumption 4 gives

$$
z _ {k + 1} \leq \lambda \left(L _ {x} ^ {H _ {a} - 1} z _ {k} + \frac {1 - L _ {x} ^ {H _ {a} - 1}}{1 - L _ {x}} c\right) + \varepsilon_ {d}.\tag{41}
$$

Equivalently,

$$
z _ {k + 1} \leq \alpha z _ {k} + \beta ,\tag{42}
$$

where

$$
\alpha := \lambda L _ {x} ^ {H _ {a} - 1}, \quad \beta := \lambda \frac {1 - L _ {x} ^ {H _ {a} - 1}}{1 - L _ {x}} c + \varepsilon_ {d}.\tag{43}
$$

Lemma 2 (Contraction across correction cycles): Suppose $L _ { x } ~ < ~ 1$ . Then $0 ~ \leq ~ \alpha ~ < ~ 1$ , and the post-correction error sequence satisfies

$$
z _ {k} \leq \alpha^ {k} z _ {0} + \frac {1 - \alpha^ {k}}{1 - \alpha} \beta .\tag{44}
$$

Hence,

$$
\limsup _ {k \to \infty} z _ {k} \leq \frac {\beta}{1 - \alpha}.\tag{45}
$$

Proof: Since $0 \leq \lambda < 1$ and $L _ { x } ~ < ~ 1$ , we have $0 ~ \leq$ $\alpha = \lambda L _ { x } ^ { \dot { H } _ { a } - 1 } < 1$ . The result follows by unrolling the affine recursion (42).

6) Main Result:

Theorem 1 (Practical stability of Tube Diffusion Policy): Suppose Assumptions 1–4 hold and $L _ { x } \quad < \quad 1$ . Then the closed-loop system under TDP satisfies the following properties:

(1) Uniform ultimate boundedness. The tracking error sequence is uniformly ultimately bounded, i.e.,

$$
\limsup _ {t \to \infty} \| e _ {t} \| <   \infty .
$$

(2) Bounded error within each correction cycle. For any correction cycle $t \in \{ t _ { k } , \ldots , t _ { k } + H _ { a } - 1 \}$ , the tracking error remains bounded as a function of the post-correction error $z _ { k }$

Proof: First, by Lemma 2, the post-correction error satisfies the affine recursion

$$
z _ {k + 1} \leq \alpha z _ {k} + \beta ,
$$

with $0 ~ \leq ~ \alpha ~ < ~ 1$ . Standard results for linear difference inequalities imply

$$
\limsup _ {k \to \infty} z _ {k} \leq \frac {\beta}{1 - \alpha}.
$$

Hence the sequence of post-correction errors is uniformly ultimately bounded.

TABLE VI: Network Architecture and Hyperparameters

<table><tr><td>Component</td><td>Specification</td></tr><tr><td>U-Net backbone</td><td>1D U-Net with 3 resolution levels (channels 256, 512, 1024)</td></tr><tr><td>Residual blocks</td><td>12 FiLM-conditioned residual blocks</td></tr><tr><td>Backbone encoders</td><td>6 ResNet-18 encoders (2 vision, 4 tactile; no weight sharing)</td></tr><tr><td>FiLM conditioning</td><td>Linear layer with conditioning dimension 3118 generating scale and shift</td></tr><tr><td>U-Net backbone</td><td>44.7M</td></tr><tr><td>Policy network</td><td>105.2M</td></tr><tr><td>Full model</td><td>172.8M</td></tr></table>

Next, consider any time $t \in \{ t _ { k } , \ldots , t _ { k } + H _ { a } - 1 \}$ . By Lemma 1,

$$
\| \boldsymbol {e} _ {t} \| \leq L _ {x} ^ {H _ {a} - 1} z _ {k} + \frac {1 - L _ {x} ^ {H _ {a} - 1}}{1 - L _ {x}} c.
$$

Thus the tracking error remains bounded during each streaming phase as a function of $z _ { k }$ . Combining this result with the boundedness of $z _ { k }$ establishes uniform ultimate boundedness of the full error trajectory.

In the ideal case where $\varepsilon _ { a } = 0 , \bar { w } = 0$ , and $\varepsilon _ { d } = 0$ , we have $c = 0$ and $\beta = 0$ , so the recursion reduces to

$$
z _ {k + 1} \leq \alpha z _ {k}, \quad 0 \leq \alpha <   1.
$$

Hence $z _ { k }  0$ . Using Lemma 1 with $c = 0$ further implies $e _ { t } \to 0 .$ , and the closed-loop system is asymptotically stable.

In summary, Theorem 1 formalizes the role of the two components in TDP. At each correction cycle, the diffusion module first performs a correction step, followed by $H _ { a } \mathrm { ~ - ~ } 1$ streaming control steps. The streaming flow provides highfrequency reactive control. However, due to bounded imitation error and external disturbances, it introduces a bounded tracking drift during the streaming phase. The diffusion module periodically contracts this accumulated drift at the beginning of each correction cycle. Their combination yields a hybrid closed-loop system in which the tracking error remains within a bounded tube around the demonstrated reference trajectory. This matches the intuition behind TDP: streaming flow ensures responsive step-wise control, while diffusion correction prevents long-term drift accumulation.

## B. Network Architecture and Hyperparameters

We provide additional details of the policy network architecture. Our method employs a conditional 1D U-Net backbone for diffusion-based action generation and streaming flow, augmented with visual and tactile encoders.

a) U-Net backbone.: The policy network is a 1D U-Net with three resolution levels (channel dimensions 256, 512, and 1024), consisting of symmetric downsampling and upsampling stages with 12 FiLM-conditioned residual blocks.

b) Conditioning.: Observation conditioning is incorporated via feature-wise linear modulation (FiLM). The conditioning dimension is 3118, and FiLM layers produce perchannel scale and shift parameters.

c) Observation encoders.: Visual and tactile inputs are encoded using six ResNet-18 backbones (two for vision and four for tactile inputs, without parameter sharing).

d) Model size.: The policy U-Net contains 44.7M parameters. The full policy network, including the U-Net backbone and action prediction heads but excluding observation encoders, contains 105.2M parameters. The complete model, including all visual and tactile encoders, contains 172.8M parameters.

Table VI summarizes the key components and hyperparameters.