# π, But Make It Fly: Physics-Guided Transfer of VLA Models to Aerial Manipulation

Johnathan Tucker<sup>1</sup> Denis Liu<sup>1</sup> Aiden Swann<sup>1</sup> Allen Ren<sup>2</sup> Javier Yu<sup>1</sup> Jiankai Sun<sup>1</sup> Brandon Kim<sup>1</sup> Lachlain McGranahan<sup>1</sup> Quan Vuong<sup>2</sup> Mac Schwager<sup>1</sup> <sup>1</sup>Stanford University <sup>2</sup>Physical Intelligence <https://airvla.github.io>

![](<images/7e8c48430991ac620784ac836e847fa6fa9071c2ed5df530383e06a58792321c.jpg>)
Fig. 1: Overview of AirVLA: Our method fine-tunes the $\pi _ { 0 }$ vision-language-action model on a combination of teleoperated and 3D Gaussian Splatting synthetic data. (Left) The policy processes multimodal inputs including natural language commands, proprioception, and multi-view camera observations. (Right) To ensure robust flight during manipulation, we introduce a payload-aware guidance signal, $F _ { g u i d e } ,$ , combined with real-time chunking. (Center) This enables AirVLA to execute novel, zero-shot compositional tasks, such as navigating through gates and manipulating objects.

Abstract—Vision-Language-Action (VLA) models such as π have demonstrated remarkable generalization across diverse fixed-base manipulators. However, transferring these foundation models to aerial platforms remains an open challenge due to the fundamental mismatch between the quasi-static dynamics of fixed-base arms and the underactuated, highly dynamic nature of flight. In this work, we introduce AirVLA, a system that investigates the transferability of manipulation-pretrained VLAs to aerial pick-and-place tasks. We find that while visual representations transfer effectively, the specific control dynamics required for flight do not. To bridge this ”dynamics gap” without retraining the foundation model, we introduce a Payload-Aware Guidance mechanism that injects payload constraints directly into the policy’s flow-matching sampling process. To overcome data scarcity, we further utilize a Gaussian Splatting pipeline to synthesize navigation training data. We evaluate our method through a cumulative 460 real-world experiments which demonstrate that this synthetic data is a key enabler of performance, unlocking 100% success in navigation tasks where directly finetuning on teleoperation data alone attains 81% success. Our inference-time intervention, Payload-Aware Guidance, increases real-world pick-and-place task success from 23% to 50%. Finally, we evaluate the model on a long-horizon compositional task, achieving a 62% overall success rate. These results suggest that pre-trained manipulation VLAs, with appropriate data augmentation and physics-informed guidance, can transfer to aerial manipulation and navigation, as well as the composition of these tasks.

Index Terms—Vision-Language-Action (VLA) Model, Vision-Language Model (VLM), Cross-Embodiment Transfer, Aerial Robotics, Unmanned Aerial Manipulators (UAMs)

## I. INTRODUCTION

Vision-Language-Action (VLA) foundation models have demonstrated remarkable cross-embodiment generalization, enabling robots with diverse morphologies to complete manipulation tasks from natural language instructions. Models such as RT-X [12], Octo [34], and π [3] leverage large-scale multimodal pretraining across dozens of robot embodiments, learning representations that transfer across manipulators with different kinematics, workspaces, and end-effector designs. However, all such demonstrations share a fundamental constraint: they operate from stable, fixed or mobile bases in quasi-static regimes. This raises a central question: can VLA policies transfer to aerial manipulators, robots that couple perception, language understanding, and contact-rich interaction with underactuated 6-DoF flight control? Such transfer would represent an extreme test of cross-embodiment learning, as aerial platforms differ fundamentally from ground robots in dynamics, control authority, and sensing characteristics.

Answering this question has both scientific and practical significance. Scientifically, aerial manipulation provides a stress test for the representations learned by VLA foundation models: if policies pretrained entirely on fixed-base manipulators can partially transfer to flying robots through fine-tuning, it suggests these models capture manipulation primitives that transcend specific embodiment constraints. Practically, languageconditioned aerial manipulation would enable applications previously beyond reach: delivering medical supplies in inaccessible terrain, clearing debris in collapsed structures, or manipulating infrastructure at height. If VLA policies could be adapted for such platforms, non-expert operators could issue high-level language commands while the policy handles lowlevel perception and control.

Several fundamental challenges distinguish aerial manipulation from the ground-robot scenarios that dominate VLA training data. First, quadrotors are underactuated: thrust and attitude are tightly coupled, and small errors in predicted actions can induce large deviations in pose. This failure mode rarely occurs in fixed-base manipulation. Second, onboard cameras experience large ego-motion, rapid viewpoint changes, motion blur, and scale variation that differ significantly from tabletop scenarios in most VLA datasets [25, 12]. Finally, contact with objects induces payload changes [5, 21] and aerodynamic disturbances [37, 35] that violate quasi-static assumptions implicit in most manipulation policies.

While recent work has begun exploring languageconditioned aerial systems and cross-embodiment transfer to drones, no prior work has systematically investigated whether manipulation-pretrained VLA foundation models can transfer to aerial platforms. Concurrent efforts take fundamentally different approaches: UMI-on-Air [19] trains embodimentagnostic diffusion policies from scratch on handheld demonstrations and introduces inference-time guidance to compensate for aerial dynamics, but does not leverage or evaluate VLA foundation models. SINGER [1] and GRaD-Nav++ [9] demonstrate language-conditioned drone navigation using learned visuomotor policies, but focus exclusively on navigation without physical manipulation or contact. More broadly, while cross-embodiment datasets such as Open X-Embodiment [12] include dozens of manipulators, they contain no aerial platforms or underactuated flight dynamics, leaving the transferability of VLA representations to aerial manipula-

tion unexplored.

To address these gaps, we introduce AirVLA, the first systematic investigation of whether manipulation-pretrained VLA foundation models can transfer to aerial manipulators through fine-tuning. Our system consists of a ModalAI Starling 2 Max quadrotor equipped with a lightweight compliant gripper derived from the Universal Manipulation Interface (UMI) [11, 20, 19], forward- and downward-facing onboard cameras, and a ROS-based control stack that treats the drone as a “flying end-effector.” We construct a diagnostic task suite spanning (i) pick-and-place manipulation collected via human teleoperation, (ii) navigation tasks with human teleoperation data and synthetic training trajectories generated using model predictive control (MPC) in Gaussian Splatting reconstructions, and (iii) compositional tasks that chain both behaviors. This design enables us to isolate which aspects of manipulation skill transfer to aerial platforms and which fail.

We directly fine-tune the foundation-scale VLA policy π<sub>0</sub> (pretrained on large manipulation datasets containing no aerial platforms) on our aerial manipulation data. Our experiments reveal partial but significant skill transfer: the finetuned policy learns stable hovering and coarse manipulation behaviors, demonstrating that foundation models capture some manipulation primitives that generalize to flight. However, standard fine-tuning fails at task completion due to sensitivity to payload dynamics and explicit obstacle navigation limitations. To address these specific failure modes, we introduce domain-adapted components: (i) a physics-aware low-level controller that wraps VLA outputs to enforce dynamically feasible commands, and (ii) synthetic navigation augmentation that enriches the training distribution. While these adaptations improve performance on navigation and stabilization during manipulation, fundamental limitations remain for long-horizon compositional manipulation and out-of-distribution adaptation, highlighting important open challenges in extending VLA models to underactuated, dynamic embodiments.

In summary, this paper makes the following contributions:

• We present AirVLA, the first demonstration of a pretrained VLA fine-tuned for an aerial manipulation platform.

• We manually collect a dataset of 270 aerial manipulation and navigation teleop-demos, to be open-sourced.

• We synthesize 50 3DGS-based corrective navigation examples to help supervise VLA fine-tuning, showing a 20% increase in navigation task success compared to finetuning with teleop only.

• We introduce a physics-informed guidance within realtime-chunking to adapt at runtime to the mass of a grasped object, improving task success by 10% − 40% in pick and place tasks.

## II. RELATED WORK

## A. Vision-Language-Action Policies for Manipulation

Vision-language-action (VLA) policies leverage large-scale multimodal pretraining to enable general-purpose robot control from natural language instructions. Early approaches such as RT-1 [6] and RT-2 [7] demonstrated that vision-language models pretrained on internet-scale data can be fine-tuned for robotic manipulation, mapping visual observations and language to action sequences. Subsequent work has scaled this paradigm through cross-embodiment training: the Open X-Embodiment dataset [12] aggregates demonstrations from dozens of robot morphologies, enabling models like RT-X [12] to generalize across diverse manipulators. More recent robot foundation models [2, 26] such as Octo [34] and $\pi _ { 0 }$ [3] incorporate flow-matching and diffusion architectures, achieving state-of-the-art performance on manipulation benchmarks spanning tabletop tasks, mobile manipulation, and dexterous control.

These successes demonstrate that VLA policies can learn manipulation primitives that transfer across embodiments with different kinematics, workspaces, and end-effector designs. However, essentially all demonstrations in existing crossembodiment datasets operate from stable bases (fixed or mobile) in quasi-static regimes. No prior work has investigated whether these learned representations transfer to aerial platforms, where underactuated flight dynamics, large ego-motion, and contact-induced disturbances fundamentally differ from ground-robot scenarios. Our work provides the first systematic evaluation of this transfer gap.

## B. Vision-Language and Vision-Language-Action Models for UAV Navigation

Vision-language models have recently been applied to drone navigation [46], typically for high-level semantic reasoning rather than low-level visuomotor control. VLMaps [23] constructs semantic maps for language-conditioned robot navigation but act as a perception and planning front-end rather than an end-to-end policy. SEEK [17] and VISTA [33] demonstrate uncertainty-aware exploration from semantic goals on quadrupeds and drones, but are limited to object localization and do not extend to interaction or contact. More recently, SINGER [1] introduces an end-to-end visuomotor policy for language-conditioned drone navigation trained on synthetic trajectories in Gaussian Splatting environments, achieving onboard inference through semantic image preprocessing with CLIPSeg [32]. GRaD-Nav++ [9] extends this approach with differentiable dynamics in Gaussian Splatting scenes, enabling multi-task generalization across navigation behaviors. Both methods demonstrate impressive sim-to-real transfer for navigation tasks.

In contrast to SINGER and GRaD-Nav++, which focus on language-conditioned navigation without contact and use Gaussian Splatting primarily as a training environment, our work targets aerial manipulation with an onboard gripper, uses Gaussian Splatting as a source of synthetic demonstration data for a specific navigation subtask, and explicitly studies how a foundation VLA policy can be leveraged when navigation and manipulation must be composed in a single language-specified behavior.

## C. Aerial Manipulation with Learned Policies

Unmanned aerial manipulation (UAM) combines multirotor flight with contact-rich interaction, introducing challenges such as underactuated dynamics, aerodynamic disturbances near surfaces, and strict payload limits. Classical systems demonstrate contact-based tasks including surface inspection [41, 5], painting [42, 18, 29], drilling [14], and object grasping [16, 45], typically using carefully engineered controllers tailored to specific scenarios. Recent work has explored learning-based approaches: model-free reinforcement learning for aerial manipulation [13], one-shot learning for autonomous grasping [49], and image-based visual servoing [40, 28]. However, these methods either lack language grounding or are limited to single tasks without generalization across manipulation primitives.

Most relevant to our work, UMI-on-Air [19] addresses the challenge of transferring manipulation policies to aerial platforms by training a single-task diffusion policy on handheld UMI demonstrations and introducing embodiment-aware guidance at inference time: gradient feedback from a lowlevel MPC controller steers trajectory generation toward dynamically feasible actions during deployment. This approach achieves impressive results on high-precision aerial manipulation tasks.

Compared to UMI-on-Air, which studies a single-task diffusion policy trained from handheld UMI demonstrations and introduces embodiment-aware guidance at inference time via MPC gradients, AirVLA studies cross-embodiment transfer in a generalist VLA foundation model (π<sub>0</sub>) fine-tuned on aerial manipulation data. While both methods use guidance at inference time, AirVLA evaluates a broader set of tasks spanning manipulation, navigation, and the zero-shot composition of both. This design lets us isolate and study the representational generalization limits of existing VLA models for aerial manipulation and navigation.

More broadly, while prior aerial manipulation work either (i) relies on task-specific controllers without language grounding or (ii) uses cross-embodiment guidance to make existing policies feasible on drones, none of these efforts evaluate a generalist VLA manipulation model on a unified suite of pickand-place, navigation, and compositional aerial tasks the way AirVLA does.

## III. METHOD

## A. System Overview

AirVLA is a vision-language-action system for aerial manipulation that transfers manipulation capabilities from foundation models pretrained on fixed-base robot arms to an underactuated quadrotor platform. The system takes as input RGB images from multiple viewpoints and a natural language task description, and outputs relative end-effector pose commands executed by a low-level flight controller.

The key challenge in this transfer is the mismatch between the quasi-static regimes of tabletop manipulation and aerial manipulation, where the platform must continuously stabilize against gravity while simultaneously executing precise gripper motions. Grasping an object introduces a step change in effective mass that, if uncompensated, causes the drone to sag and potentially fail the task.

![](<images/ffd173265b32181d6c26f46de7c5409eaaa3564390e9f2c1f20b887e8e731930.jpg>)  
Fig. 2: Custom gripper installed on the Starling 2 Max drone. The design prioritizes low weight for extended flight time while maintaining sufficient grip strength for the target objects.

Our approach addresses this challenge through two main contributions: (1) a physics-aware guidance mechanism that augments the pretrained policy’s action generation process with payload-aware vertical compensation at inference time, and (2) a Gaussian-splatting data pipeline that enables efficient collection and synthesis of diverse training trajectories from a small number of seed flights. Together, these allow a VLA model pretrained on large-scale robot manipulation data to perform aerial pick-and-place and navigation with minimal drone-specific fine-tuning.

## B. Hardware

The system integrates the ModalAI Starling 2 Max drone with a customized Universal Manipulation Interface (UMI) gripper [11] and multiple cameras to enable autonomous aerial manipulation. The Starling 2 Max is powered by the VOXL 2 companion computer (Qualcomm QRB5165). The customized UMI-style gripper is attached to the underside of the drone, enabling dynamic grasping. The system integrates one external camera and two onboard cameras (downward and forwardfacing), providing RGB images at 5Hz.

1) Gripper Design: We showcase the design of our gripper in Fig. 2. The gripper is designed to be built cheaply without specialized tools. The frame is entirely 3D printed, with two hobby-grade servos slotting in without screws. The fingers are adapted from the UMI gripper [11], facilitating direct comparison with arm-based policies trained with similar endeffector geometry.

2) Observation and Action Spaces: The observation space consists of RGB images from three cameras at $[ 2 5 6 \times 2 5 6 ]$ resolution, the drone’s estimated pose from a motion capture system, and gripper aperture. The action space, consisting of the drone position and yaw, is represented as an action chunk $\boldsymbol { A } \in \mathbb { R } ^ { H \times \bar { \boldsymbol { D } } }$ (i.e., end-effector 4 DoF delta poses and gripper commands D over horizon H). Actions are generated at 10

Hz and executed by the PX4 flight controller via position setpoints.

## C. Policy Architecture

We build on $\pi _ { 0 } ~ [ 3 ] ,$ , a vision-language-action model that represents the conditional action distribution using a flowmatching (continuous-time) generative model [30]. Given an observation o (images, proprioception, language), π<sub>0</sub> defines a velocity field $v _ { \theta } ( x _ { \tau } , o , \tau )$ over latent action chunks $x _ { \tau } \in { }$ $\mathbb { R } ^ { H \times D }$ and diffusion/flow time $\tau \in \ [ 0 , 1 ]$ . Sampling draws $x _ { 0 } \sim \mathcal { N } ( 0 , I )$ and integrates the ODE

$$
\frac {d x _ {\tau}}{d \tau} = v _ {\theta} (x _ {\tau}, o, \tau)\tag{1}
$$

to obtain the generated action chunk $A = x _ { 1 }$

For real-time execution, we employ Real-Time Chunking (RTC) [4], which enables asynchronous inference by freezing the prefix of the next chunk that will execute before inference completes, and inpainting the remaining suffix conditioned on the frozen prefix. Concretely, RTC defines a soft temporal mask over the horizon that blends previously committed actions with newly generated actions to avoid discontinuities at chunk boundaries.

## D. Physics-Aware Guidance for Action Generation

RTC shows that inference-time steering can be implemented by modifying the velocity field during sampling [4]. We generalize this idea by introducing a loss $\Phi ( A ; o )$ over the denoised action chunk $\textit { A } \ = \ x _ { 1 }$ , and adding a gradient correction term derived from $\nabla _ { A } \Phi$ to the base velocity field, analogous in spirit to gradient-based guidance methods in generative modeling [22, 38]. Intuitively, we seek samples that both (i) have high probability under the base policy and (ii) minimize the guidance loss:

$$
p _ {\text { guid }} (A \mid o) \propto p _ {\theta} (A \mid o) \exp (- \Phi (A; o)).\tag{2}
$$

This formulation seeks a ’sweet spot’ between the policy’s priors and physical constraints. It biases the sampling process toward actions that are high-probability under the $\mathrm { V L A }$ (preserving learned manipulation skills) while simultaneously minimizing the cost Φ (enforcing flight feasibility), effectively steering the drone at runtime without retraining.

Operationally, at intermediate time $\tau ,$ we compute the model’s current prediction of the terminal action chunk $\hat { A } _ { \theta } ( x _ { \tau } , o , \tau ) \approx x _ { 1 } ~ ( \mathrm { e . g . } ~ $ , via the model’s internal denoised estimate), evaluate the gradient $\nabla _ { A } \Phi ( \hat { A } _ { \theta } ; o )$ , and map it back to a correction in latent space through a vector–Jacobian product $\boldsymbol { \xi } : = \left( \nabla _ { x _ { \tau } } \hat { A } _ { \theta } ( x _ { \tau } , o , \tau ) \right) ^ { \top } \nabla _ { A } \Phi \left( \hat { A } _ { \theta } ( x _ { \tau } , o , \tau ) ; o \right)$ . This yields the guided velocity field

$$
v _ {\mathrm{guid}} (x _ {\tau}, o, \tau) = v _ {\theta} (x _ {\tau}, o, \tau) - s (\tau) \xi .\tag{3}
$$

where $s ( \tau )$ is a scalar guidance schedule. Multiple loss terms compose additively because $- \log p _ { \mathrm { g u i d } }$ adds losses, so RTC-style continuity objectives and task-specific physics objectives can be applied simultaneously without changing the pretrained model weights.

1) General Tracking-Error Guidance: Suppose we have a reference trajectory $A ^ { \mathrm { d e s } } ( o ) \in \mathbb { R } ^ { H \times D }$ derived from the current observation. We define a tracking loss

$$
\Phi_ {\text { track }} (A; o) = \frac {1}{2} \sum_ {t = 0} ^ {H - 1} \sum_ {d = 1} ^ {D} \lambda_ {d} w _ {t} \left(A _ {t, d} - A _ {t, d} ^ {\text { des }} (o)\right) ^ {2},\tag{4}
$$

with per-dimension strengths $\lambda _ { d } \geq 0$ and temporal weights $w _ { t } ~ \geq ~ 0 .$ . The temporal weights follow the soft masking schedule from RTC [4]: $w _ { t } = 1$ for frozen prefix actions, the weights then exponentially decay for the intermediate region and are zero for freshly generated actions. This ensures the guidance strongly enforces continuity with committed actions while allowing freedom for new generation. The corresponding gradient correction is derived from

$$
\frac {\partial \Phi_ {\mathrm{track}}}{\partial A _ {t , d}} = \lambda_ {d} w _ {t} \left(A _ {t, d} - A _ {t, d} ^ {\mathrm{des}} (o)\right),\tag{5}
$$

which pulls the denoised chunk toward the reference trajectory in each dimension proportionally to $\lambda _ { d } .$

1) Payload-Aware Vertical Guidance: In aerial manipulation, the dominant disturbance during grasping is an effective mass increase that manifests as vertical sag under load. Rather than modeling full 6-DOF dynamics, we instantiate (4) only on the altitude-related action dimension $d _ { z } \mathbf { ; }$

$$
\Phi_ {\text { payload }} (A; o, A _ {t - 1}) = \frac {\lambda_ {z}}{2} \alpha (o, A _ {t - 1}) \sum_ {t = 0} ^ {H - 1} w _ {t} \left(z _ {t} (A) - z _ {\mathrm{des}} (o)\right) ^ {2},\tag{6}
$$

where $z _ { t } ( A )$ denotes the z-component of the action at timestep $t ,$ and

$$
z _ {\mathrm{des}} (o) = z _ {\mathrm{curr}} (o) + \Delta z\tag{7}
$$

biases the drone toward slightly higher altitude under load $( \Delta z > 0 )$ . Here $z _ { \mathrm { c u r r } } ( o )$ comes from a motion capture system, and $\Delta z$ is a tuned offset capturing the expected sag for typical payloads.

The payload confidence $\alpha ( o , A _ { t - 1 } ) \ \in \ [ 0 , 1 ]$ is computed from (i) the previously executed action chunk $A _ { t - 1 }$ and (ii) the current measured gripper aperture in o. Concretely, we compute a smooth, bounded payload confidence by combining recent gripper command intent with the current measured aperture. Let $u _ { t } ~ \in ~ [ - 1 , 1 ]$ denote the gripper command at timestep t (with +1 corresponding to $\mathbf { \tilde { \Sigma } } ^ { 6 6 } \mathbf { c } ] 0 \mathbf { s } \mathbf { e } ^ { 7 } )$ , and let $g ( o ) \in [ 0 , 1 ]$ denote the measured aperture (normalized so that larger values are more open). From the previously executed chunk $A _ { t - 1 }$ , we form continuous close/open intents using the last K commands,

$$
\begin{array}{l} c _ {\text { intent }} = \frac {1}{K} \sum_ {i = H - K} ^ {H - 1} \operatorname{clip} \left(\frac {u _ {i} + 1}{2},   0,   1\right), \\ o _ {\text { intent }} = \frac {1}{K} \sum_ {i = H - K} ^ {H - 1} \operatorname{clip} \left(1 - \frac {u _ {i} + 1}{2},   0,   1\right), \end{array}\tag{8}
$$

and compute soft aperture-based scores $c _ { \mathrm { m e a s } } , o _ { \mathrm { m e a s } } \in [ 0 , 1 ]$ from $g ( o )$ (higher when the gripper is closed/open, respectively). We then define an “open” gate and payload confidence

as

$$
\begin{array}{c} o _ {\text {flag}} = \operatorname{clip} \bigl (\frac {1}{2} o _ {\text {intent}} + \frac {1}{2} o _ {\text {meas}},   0,   1 \bigr), \\ \alpha (o, A _ {t - 1}) = \operatorname{clip} ((1 - o _ {\text {flag}})   c _ {\text {intent}}   c _ {\text {meas}},   0,   1). \end{array}\tag{9}
$$

When $\alpha \approx 0 ,$ the loss vanishes and sampling reduces to vanilla RTC; when $\alpha \approx 1$ , the sampler prefers chunks whose vertical component is biased toward $z _ { \mathrm { d e s } } ,$ , compensating for payload. This can be viewed as mode-dependent feedforward (gain scheduling / gravity compensation), but injected inside the generative policy’s sampling dynamics by adding the gradient correction term derived from $\nabla _ { A } \Phi _ { \mathrm { p a y l o a d } }$ via (3).

## E. Gaussian Splat Data Pipeline

Collecting aerial manipulation demonstrations is timeconsuming and requires skilled pilots. To efficiently generate diverse training data, we adopt a “Flying in Gaussian Splats”- style approach that couples photorealistic Gaussian-splatting reconstructions with a lightweight drone dynamics model to synthesize large volumes of training data from a small set of seed demonstrations [31]. Concretely, our pipeline proceeds by first reconstructing the static environment as a Gaussian Splat and isolating the gripper visuals to prevent observation bias. We then couple a drone dynamics model with these assets to synthesize diverse, physics-feasible training trajectories that cover both nominal navigation and recovery behaviors.

1) Scene Reconstruction: We reconstruct each scene from short walk-throughs captured with the drone’s forward-facing camera. This results in a set of metrically scaled poses for each image, which are used to train a 3D Gaussian splatting model [24] using Nerfstudio [39, 44]. The resulting model ${ \mathcal { G S } } _ { \phi }$ renders photorealistic images from arbitrary camera poses in the captured region. Given a camera pose $( p , q )$ , the rendered image is $I = { \mathcal G S } _ { \phi } ( p , q )$

2) Gripper Segmentation and Compositing: The downward-facing camera provides critical visual information for manipulation, but the gripper is persistently visible in its field of view. Including raw downward facing images in the synthetic training data from the Gaussian splatting model would introduce an observation bias into the policy that would result in unwanted behavior. Thus, we explicitly treat the gripper as a separate foreground layer and composite it onto renders from a gripper-free scene model. Concretely, for each downward-facing training frame $I _ { \mathrm { d o w n } } ,$ we compute a gripper mask $M _ { \mathrm { g r i p } }$ using Segment Anything (SAM) [27, 36] with a fixed bounding box prompt corresponding to the gripper’s known image-space location. We then extract a masked gripper patch

$$
G = I _ {\text { down }} \odot M _ {\text { grip }},\tag{10}
$$

to form a small library $\{ G _ { a } \}$ of representative gripper appearances.

We train the Gaussian splatting model ${ \mathcal { G S } } _ { \phi }$ on images from cameras where the gripper is not visible (namely, the forwardfacing views). At synthesis time, we render the clean scene from ${ \mathcal G S } _ { \phi }$ for the downward-facing viewpoint and composite the gripper foreground:

$$
I _ {\text { down }} ^ {\text { synth }} = (1 - M _ {\text { grip }}) \odot \mathcal {G S} _ {\phi} (p, q) + M _ {\text { grip }} \odot G _ {a (t)}.\tag{11}
$$

This yields a downward-view image whose background is fully determined by the scene model, while the gripper appearance is consistent with the commanded aperture $a ( t )$

1) Drone Dynamics Model: Following [31], we simulate drone motion with a semi-kinematic state $\overset { \cdot } { x } = \overset { \cdot } { ( } p ^ { W } , v ^ { W } , q _ { B } ^ { W } )$ comprising position, velocity, and orientation quaternion. Controls are normalized thrust $f _ { \mathrm { t h } }$ and body angular velocity $\omega ^ { B }$ The dynamics are

$$
\dot {p} ^ {W} = v ^ {W},\tag{12}
$$

$$
\dot {v} ^ {W} = g e _ {3} ^ {W} + \frac {k _ {\mathrm{th}}}{m} f _ {\mathrm{th}} R (q _ {B} ^ {W}) e _ {3} ^ {B},\tag{13}
$$

$$
\dot {q} _ {B} ^ {W} = \frac {1}{2} \Omega (\omega^ {B}) q _ {B} ^ {W},\tag{14}
$$

where $g$ is gravitational acceleration, $e _ { 3 } ^ { W } , e _ { 3 } ^ { B }$ are the z-axis unit vectors in world/body frames, $R ( \cdot )$ converts quaternion to rotation, and $( k _ { \mathrm { t h } } , m )$ are thrust coefficient and mass. We forward-integrate using ACADOS [43] and render images using the body-to-camera transform $T _ { C } ^ { B }$

1) Domain-Randomized Data Synthesis: To enable the policy to recover from dangerous states near the obstacle, we additionally synthesize recovery trajectories by randomizing task geometry, similar to FiGS [31]. For each nominal trajectory $( X ^ { d } , U ^ { d } )$ , we generate $N _ { s }$ randomized rollouts by sampling an initial state perturbation $x _ { 0 } ^ { s } \sim \mathcal { U } ( x _ { 0 } ^ { d } - \Delta x , x _ { 0 } ^ { d } + \Delta x )$ , then simulating the resulting trajectory rendering multi-view images from the Gaussian splat scene model ${ \mathcal { G S } } _ { \phi }$

We additionally randomize intermediate waypoints to increase diversity and induce recovery behaviors. For the navigation task, we randomize the terminal hover location to lie above the goal by sampling a height offset uniformly in [1.0, 1.5] meters. Let $p _ { \mathrm { o b j } }$ denote the object goal position in the scene and let $h \sim \mathcal { U } ( 1 . 0 , 1 . 5 ) \mathrm { m }$ . We set the goal position as

$$
p _ {\text { goal }} = p _ {\text { obj }} + [ 0, 0, h ] ^ {\top}.\tag{15}
$$

To diversify gate exits, we randomize the post-gate waypoint within a ball of diameter 0.25 m centered at a nominal aftergate position waypoint $p _ { \mathrm { a f t e r } } ^ { d }$ . With radius $r = 0 . 1 2 5$ m and $\delta \sim \mathcal { U } ( \mathbb { B } ( 0 , r ) )$ ) where $\mathbb { B } ( 0 , r ) = \{ \delta \in \mathbb { R } ^ { 3 } : \| \delta \| _ { 2 } \leq r \}$ , we define

$$
p _ {\mathrm{after}} = p _ {\mathrm{after}} ^ {d} + \delta .\tag{16}
$$

Finally, we perturb trajectories to elicit recovery behavior by inserting an additional waypoint between the start and the gate that forces the drone to pass near one of four gate extremities, namely top, bottom, left, or right. We sample a side $s \sim \mathrm { U n i f } ( \{ \log $ , bottom, left, right}) and define a sidespecific waypoint

$$
p _ {\mathrm{wp}} = \bar {p} _ {s} + b n _ {s},\tag{17}
$$

where $\bar { p } _ { s }$ is a reference point near the corresponding gate boundary, $n _ { s }$ is the offset direction away from the gate frame, and $b \ > \ 0$ is a fixed geometry buffer chosen to ensure collision-free clearance.

After sampling these randomized waypoint targets, we generate controls by tracking the waypoint-conditioned plan under the drone dynamics, producing the rollout $( X ^ { s } , U ^ { s } )$ . Figure

![](<images/fb634dfd6e20505b85c60a55765f88c75d35d10b77969e154fccf51e3ff65ce3.jpg>)  
Left • Nominal

![](<images/fab6d3033da2104c3e92eecc705bb31d2474cf93498e03afcd3cf949436c8e98.jpg>)  
Left • Corrective

![](<images/1615c21f771718dffba87512531b2ab9ade023d71c5b96b2c84cf93ec3de2a06.jpg>)

![](<images/d64132309c4ea9f19e0fe52286863c745fc8b1a62dc043628bb7b6aa582fc002.jpg>)  
Right • Nominal  
Right · Corrective  
Fig. 3: Examples of synthetic trajectories used to generate training data. Nominal rollouts and corrective rollouts overlay sampled trajectories in the gate-manipulation scene.

3 depicts nominal and corrective synthetic trajectories for both the left gate and right gate scenarios. We render images along $X ^ { s }$ from ${ \mathcal { G S } } _ { \phi }$ using the body-to-camera transform $T _ { C } ^ { B }$ and composite the gripper for downward-facing views. This procedure yields a large set of physically and geometrically plausible trajectories that cover both nominal executions and off-nominal approaches requiring recovery near the gate.

## IV. EXPERIMENTS

To evaluate our methods, we define a task-specific score rubric, report quantitative results under that rubric, and organize experiments to isolate the contribution of pretraining transfer, inference-time stabilization, and data augmentation.

Through our experiments, we seek to study the following questions:

1) Transfer: How much manipulation-pretrained VLA capability transfers to an underactuated aerial manipulator after fine-tuning?

2) Inference-time control: Can real-time chunking (RTC) and payload-aware guidance reduce the sensitivity of aerial execution to chunk discontinuities and payload disturbances?

3) Data synthesis: Does Gaussian-splat-based synthetic navigation augmentation improve real-world navigation performance?

4) Compositionality: Can a single policy reliably compose navigation and grasping into a multi-stage task (navigate-then-grasp)?

## A. Task suite

We evaluate using the following task prompts:

• Penguin Grasp (Pick-and-Place): pick up the stuffed animal and put it in the blue bin.

TABLE I: Single Task Performance Benchmarks (%). Each method was evaluated across twenty trials per task.

<table><tr><td rowspan="2">Method</td><td colspan="2">Penguin Grasp</td><td colspan="2">Navigation (Non-Synthetic)</td><td colspan="2">Navigation (Synthetic)</td></tr><tr><td>Pick</td><td>Place</td><td>Gate</td><td>Hover</td><td>Gate</td><td>Hover</td></tr><tr><td> $\pi_0$  naive</td><td>50.0</td><td>0.0</td><td>50.0</td><td>60.0</td><td>45.0</td><td>100.0</td></tr><tr><td> $\pi_0 + RTC$ </td><td>85.0</td><td>23.5</td><td>80.0</td><td>81.2</td><td>95.0</td><td>100.0</td></tr><tr><td> $\pi_0 + RTC$  w/ payload-aware guidance (ours)</td><td>100.0</td><td>50.0</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>ACT</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>Diffusion Policy</td><td>10.0</td><td>0.0</td><td>15.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr></table>

• Gate Navigation: fly through the gate and hover over the stuffed animal.

• Compositional (Navigate-then-Grasp): fly through the gate and hover over the stuffed animal and then pick up the stuffed animal and put it in the blue bin.

The penguin grasp task evaluates the model’s ability to transfer manipulation skills from pretraining to a drone embodiment. To ensure robustness, we vary the object’s position within a box during both data collection and evaluation. The gate navigation task tests drone-specific navigation requiring explicit obstacle avoidance. By using two different gate positions (left and right), we force the policy to localize the gate prior to navigation. Finally, to test the model’s ability to handle novel, composite tasks in a zero-shot manner, we fine-tune the policy on the combined datasets and evaluate it using a combined prompt unseen during training.

## B. Evaluation rubric and protocol

For each task, we define a rubric that measures progress toward completion. In our setting, the primary metric is binary task success, and we additionally record a diagnostic failure taxonomy to attribute failures for qualitative analysis:

• Penguin Grasp: a two stage task with success markers triggered after (1) picking up the stuffed animal and (2) placing it in the bin. Failure modes include grasping followed by immediate drops or crashes as well as failure to pick.

• Gate Navigation: a two stage task with success markers triggered after (1) navigating through the gate and (2) holding a hover over the stuffed animal. Failure modes include passing through the gate but not hovering over the penguin, and crashing or failing to pass through the gate.

• Compositional Navigate and Grasp: a four stage task combining the success and failure markers for the above with an additional failure marker of incorrect subtask order (attempted grasp before passing the gate).

## C. Methods compared

We compare the following methods:

$\pi _ { 0 }$ naive: roll out action chunks, pausing for inference at the end of each action chunk.

• π + RTC: real-time chunking that re-samples suffix actions to avoid discontinuities [4].

• π<sub>0</sub> + RTC + payload-aware guidance (ours): inferencetime steering implemented via a gradient correction term applied inside the sampler dynamics [4].

• ACT [47] and Diffusion Policy [10]

For the navigation and compositional tasks, we additionally evaluate synthetic and non-synthetic variants, corresponding to whether the policy was trained with synthetic data augmentation. Training details for these methods are provided in Appendix VII-A.

## D. Quantitative results

Each of the above methods are evaluated across twenty trials per task for a cumulative 460 flight trials. We omit evaluations of the payload-aware guidance method on pure navigation tasks, as the guidance method is identical to RTC when objects are not being manipulated. We present the results of these evaluations in Tables I and II. The reported success rates are conditioned on the number of trials that succeeded in the prior stage of the task. For example, the place success rate is conditioned on the number of trials that succeeded in picking the object.

To evaluate generalization to novel objects, we conduct 10 additional pick-and-place trials, replacing the objects used during training with previously unseen ones. Furthermore, to assess spatial robustness, we vary the gate location across three regions (front, left, and right). We conduct five trials per region, shifting and rotating the gate within each area. These out-of-distribution results are detailed in Table III. We discuss these results further in Appendix VII-B, and interpret them as evidence of partial semantic transfer together with substantial overfitting in the small-data fine-tuning regime for navigation.

## E. Analysis

Inference-time structure is critical for aerial pick-andplace. Naive fine-tuning is insufficient for full pick-and-place success, attaining 0%, with failures dominated by missed grasps and post-contact disturbances. RTC improves success (up to 23.5%) by stabilizing execution across chunk boundaries, and payload-aware guidance further improves success (up to 50%), consistent with the need to compensate for payload-induced altitude sag and other underactuated dynamics. Although the pick-and-place task is deliberately constrained relative to unconstrained 6-DoF aerial manipulation, it remains challenging for end-to-end visuomotor learning, as reflected in the poor performance of ACT and Diffusion Policy. This suggests that embodiment-specific adaptation, such as payload-aware guidance, is necessary to achieve reliable success even in controlled top-down grasping due to the underactuated flight dynamics.

TABLE II: Compositional Navigate-then-Grasp Success (%). Each method was evaluated across twenty trials per task.

<table><tr><td>Data Setting</td><td>Method</td><td>Gate</td><td>Hover</td><td>Pick</td><td>Place</td></tr><tr><td rowspan="5">No Synthetic</td><td> $\pi_0$  naive</td><td>35.0</td><td>85.7</td><td>42.9</td><td>0.0</td></tr><tr><td> $\pi_0 + RTC$ </td><td>80.0</td><td>100.0</td><td>81.2</td><td>15.4</td></tr><tr><td> $\pi_0 + RTC$  w/ payload-aware guidance (ours)</td><td>70.0</td><td>100.0</td><td>100.0</td><td>35.7</td></tr><tr><td>ACT</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>Diffusion Policy</td><td>5.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td rowspan="5">Synthetic</td><td> $\pi_0$  naive</td><td>70.0</td><td>85.7</td><td>25.0</td><td>0.0</td></tr><tr><td> $\pi_0 + RTC$ </td><td>95.0</td><td>94.7</td><td>83.3</td><td>20.0</td></tr><tr><td> $\pi_0 + RTC$  w/ payload-aware guidance (ours)</td><td>85.0</td><td>100.0</td><td>94.1</td><td>62.5</td></tr><tr><td>ACT</td><td>25.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>Diffusion Policy</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr></table>

TABLE III: Out-of-Distribution (OOD) Robustness (%).Success rates for variations in object class and gate positioning. Place and Hover are reported as conditional percentages.

<table><tr><td colspan="3">Grasp (Object Variation)</td><td colspan="3">Navigation (Gate Locations)</td></tr><tr><td>Object</td><td>Pick</td><td>Place</td><td>Location</td><td>Gate</td><td>Hover</td></tr><tr><td>Chips</td><td>10.0</td><td>0.0</td><td>Front</td><td>0.0</td><td>-</td></tr><tr><td>Sandwich</td><td>70.0</td><td>57.1</td><td>Left</td><td>0.0</td><td>-</td></tr><tr><td>Box</td><td>30.0</td><td>33.3</td><td>Right</td><td>40.0</td><td>100.0</td></tr></table>

RTC substantially improves gate navigation. In the nonsynthetic gate navigation trials, RTC increases gate traversal success from 50% to 80% and from 45% to 95% in the synthetic gate navigation trials. This indicates that RTC is not merely helpful but necessary for reliable navigation, since re-planning at runtime mitigates compounding errors during aggressive motion.

Synthetic augmentation helps most when paired with RTC. In the synthetic setting, naive performance is asymmetric across gates while π<sub>0</sub>+RTC achieves 95% success at the gate traversal component of the task. This suggests that synthetic augmentation is valuable but insufficient on its own: it expands coverage of approaches and recoveries, while reliable deployment still depends on inference-time stabilization from RTC. Qualitatively, we observed that the policy would periodically fall back on ”synthetic modes” that replay the GS-generated trajectories, ensuring successful gate traversal.

Compositional tasks reveal an ability to generalize from atomic tasks. Our method shows a strong ability to generalize to compositional tasks in a zero-shot matter attaining a 62% overall conditioned success rate. The compositional task evaluations further highlight the importance of closed-loop policy inference as well as the necessity of GS synthetic trajectories to ensure successful gate traversal.

Out-of-distribution trials demonstrate generalization capabilities. Our method generalizes effectively in pick-and-place tasks, achieving up to 57% task success on a novel sandwich object. However, performance varies by object geometry: the model achieved only a 10% pick success rate with a bag of chips, compared to 70% for the sandwich. The primary failure mode was the policy’s inability to adapt grasp poses to the chip bags geometry. In the navigation task, the policy adapts best to the ”right” region (40% success). Conversely, trials in the ”front” and ”left” regions failed completely. Specifically, the drone consistently bypassed the gate in front-region trials, while left-region trials resulted in collisions due to incomplete position adaptation.

## V. LIMITATIONS AND FUTURE WORK

We present AirVLA, the first systematic evaluation of transferring a manipulation-pretrained VLA foundation model to an aerial platform. Our results provide a nuanced answer to the challenge of cross-embodiment transfer: while the visual and semantic representations of models like π transfer robustly to drone viewpoints, the intricate dynamics of aerial manipulation require specific interventions. By integrating payload-aware guidance directly into the flow-matching sampling loop, we improve single-task placement success from 0% (naive baseline) to 50%, reconciling the generative diversity of a VLA with the strict physical constraints of flight. Furthermore, we show that the policy can execute compositional tasks in a zeroshot manner, achieving 85% gate traversal and up to 62.5% place success rates. Although our method currently relies on motion capture to isolate the cross-embodiment transfer problem from localization engineering issues; future work will replace this dependency with onboard VIO or SLAM. Doing so would expand the operating domain to settings such as multi-room navigation that are difficult to study under motion capture. Finally, our out-of-distribution evaluations highlight a disparity in generalization: while the model achieves 57% success on novel object classes, the lower 40% success rate in novel navigation scenarios indicates that significantly more data is needed to enable robust aerial navigation.

## VI. ACKNOWLEDGMENTS

The authors would like to thank Timothy Chen and Ola Shorinwa for their advice and reviews in the paper writing stage. This work was supported in part by a gift from Google, NSF grant FRR 2342246, and ONR grant N00014-23-1-2354.

[1] Maximilian Adang, JunEn Low, Ola Shorinwa, and Mac Schwager. Singer: An onboard generalist visionlanguage navigation policy for drones. arXiv preprint arXiv:2509.18610, 2025.

[2] Johan Bjorck, Fernando Castaneda, Nikita Cherniadev,˜ Xingye Da, Runyu Ding, Linxi Fan, Yu Fang, Dieter Fox, Fengyuan Hu, Spencer Huang, et al. Gr00t n1: An open foundation model for generalist humanoid robots. arXiv preprint arXiv:2503.14734, 2025.

[3] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, and et al. π<sub>0</sub>: A vision-language-action flow model for general robot control, Nov 2024. URL <https://doi.org/10.48550/arXiv>. 2410.24164.

[4] Kevin Black, Manuel Y Galliker, and Sergey Levine. Real-time execution of action chunking flow policies. arXiv preprint arXiv:2506.07339, 2025.

[5] Karen Bodie, Maximilian Brunner, Michael Pantic, Stefan Walser, Patrick Pfandler, Ueli Angst, Roland Sieg-¨ wart, and Juan Nieto. An omnidirectional aerial manipulation platform for contact-based inspection. arXiv preprint arXiv:1905.03502, 2019.

[6] Anthony Brohan et al. Rt-1: Robotics transformer for real-world control at scale. In arXiv preprint arXiv:2212.06817, 2022.

[7] Anthony Brohan et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. In arXiv preprint arXiv:2307.15818, 2023.

[8] Remi Cadene, Simon Alibert, Alexander Soare, Quentin Gallouedec, Adil Zouitine, Steven Palma, Pepijn Kooijmans, Michel Aractingi, Mustafa Shukor, Dana Aubakirova, Martino Russi, Francesco Capuano, Caroline Pascal, Jade Choghari, Jess Moss, and Thomas Wolf. Lerobot: State-of-the-art machine learning for real-world robotics in pytorch. <https://github.com/huggingface/> lerobot, 2024.

[9] Qianzhong Chen, Naixiang Gao, Suning Huang, JunEn Low, Timothy Chen, Jiankai Sun, and Mac Schwager. Grad-nav++: Vision-language model enabled visual drone navigation with gaussian radiance fields and differentiable dynamics. arXiv preprint arXiv:2506.14009, 2025.

[10] Cheng Chi, Siyuan Feng, Yilun Du, Zhenjia Xu, Eric Cousineau, Benjamin Burchfiel, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. In Proceedings of Robotics: Science and Systems (RSS), 2023.

[11] Cheng Chi, Zhenjia Xu, Chuer Pan, Eric Cousineau, Benjamin Burchfiel, Siyuan Feng, Russ Tedrake, and Shuran Song. Universal manipulation interface: Inthe-wild robot teaching without in-the-wild robots. In Proceedings of Robotics: Science and Systems (RSS), 2024.

[12] Open X-Embodiment Collaboration et al. Open X-Embodiment: Robotic learning datasets and RT-X models. <https://arxiv.org/abs/2310.08864>, 2023.

[13] Eugenio Cuniato, Ismail Geles, Weixuan Zhang, Olov Andersson, Marco Tognon, and Roland Siegwart. Learning to open doors with an aerial manipulator. In 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 6942–6948. IEEE, 2023.

[14] Caiwu Ding, Lu Lu, Cong Wang, and Caiwen Ding. Design, sensing, and control of a novel uav platform for aerial drilling and screwing. IEEE Robotics and Automation Letters, 6(2):3176–3183, 2021.

[15] Senyu Fei, Siyin Wang, Junhao Shi, Zihao Dai, Jikun Cai, Pengfang Qian, Li Ji, Xinzhe He, Shiduo Zhang, Zhaoye Fei, et al. Libero-plus: In-depth robustness analysis of vision-language-action models. arXiv preprint arXiv:2510.13626, 2025.

[16] Vaibhav Ghadiok, Jeremy Goldin, and Wei Ren. Autonomous indoor aerial gripping using a quadrotor. In 2011 IEEE/RSJ International Conference on Intelligent Robots and Systems, pages 4645–4651. IEEE, 2011.

[17] Muhammad Fadhil Ginting, Sung-Kyun Kim, David D Fan, Matteo Palieri, Mykel J Kochenderfer, and Ali-akbar Agha-Mohammadi. Seek: Semantic reasoning for object goal navigation in real world inspection tasks. arXiv preprint arXiv:2405.09822, 2024.

[18] Xiaofeng Guo, Guanqi He, Jiahe Xu, Mohammadreza Mousaei, Junyi Geng, Sebastian Scherer, and Guanya Shi. Flying calligrapher: Contact-aware motion and force planning and control for aerial manipulation. IEEE Robotics and Automation Letters, 2024.

[19] Harsh Gupta, Xiaofeng Guo, Huy Ha, Chuer Pan, Muqing Cao, Dongjae Lee, Sebastian Sherer, Shuran Song, and Guanya Shi. Umi-on-air: Embodiment-aware guidance for embodiment-agnostic visuomotor policies. arXiv preprint arXiv:2510.02614, 2025.

[20] Huy Ha, Yihuai Gao, Zipeng Fu, Jie Tan, and Shuran Song. Umi on legs: Making manipulation policies mobile with manipulation-centric whole-body controllers. arXiv preprint arXiv:2407.10353, 2024.

[21] Guanqi He, Yash Janjir, Junyi Geng, Mohammadreza Mousaei, Dongwei Bai, and Sebastian Scherer. Imagebased visual servo control for aerial manipulation using a fully-actuated uav. In 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2023. URL <https://arxiv.org/pdf/2306.16530.pdf>.

[22] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance, 2022. URL <https://arxiv.org/abs/2207.12598>.

[23] Chenguang Huang, Oier Mees, Andy Zeng, and Wolfram Burgard. Visual language maps for robot navigation. arXiv preprint arXiv:2210.05714, 2022.

[24] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler,¨ and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139– 1, 2023.

[25] Alexander Khazatsky, Karl Pertsch, Suraj Nair, Ash-

win Balakrishna, Sudeep Dasari, Siddharth Karamcheti, Soroush Nasiriany, Mohan Kumar Srirama, Lawrence Yunliang Chen, Kirsty Ellis, et al. Droid: A large-scale in-the-wild robot manipulation dataset. arXiv preprint arXiv:2403.12945, 2024.

[26] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024.

[27] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4015–4026, 2023.

[28] Maximilian Laiacker, Felix Huber, and Konstantin Kondak. High accuracy visual servoing for aerial manipulation using a 7 degrees of freedom industrial manipulator. In 2016 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 1631–1636, 2016. doi: 10.1109/IROS.2016.7759263.

[29] Christian Lanegger, Marco Ruggia, Marco Tognon, Lionel Ott, and Roland Siegwart. Aerial layouting: Design and control of a compliant and actuated end-effector for precise in-flight marking on ceilings. Proceedings of Robotics: Science and System XVIII, page p073, 2022.

[30] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling, 2023. URL <https://arxiv.org/abs/2210.02747>.

[31] JunEn Low, Maximilian Adang, Javier Yu, Keiko Nagami, and Mac Schwager. Sous vide: Cooking visual drone navigation policies in a gaussian splatting vacuum. IEEE Robotics and Automation Letters, 10(5): 5122–5129, 2025. doi: 10.1109/LRA.2025.3553785.

[32] Timo Luddecke and Alexander Ecker. Image segmenta-¨ tion using text and image prompts. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 7086–7096, 2022.

[33] Keiko Nagami, Timothy Chen, Javier Yu, Ola Shorinwa, Maximilian Adang, Carlyn Dougherty, Eric Cristofalo, and Mac Schwager. Vista: Open-vocabulary, taskrelevant robot exploration with online semantic gaussian splatting. arXiv preprint arXiv:2507.01125, 2025.

[34] Octo Model Team, Dibya Ghosh, Homer Walke, Karl Pertsch, Kevin Black, Oier Mees, Sudeep Dasari, Joey Hejna, Charles Xu, Jianlan Luo, Tobias Kreiman, You Liang Tan, Lawrence Yunliang Chen, Pannag Sanketi, Quan Vuong, Ted Xiao, Dorsa Sadigh, Chelsea Finn, and Sergey Levine. Octo: An open-source generalist robot policy. In Proceedings of Robotics: Science and Systems, Delft, Netherlands, 2024.

[35] Michael O’Connell, Guanya Shi, Xichen Shi, Kamyar Azizzadenesheli, Anima Anandkumar, Yisong Yue, and Soon-Jo Chung. Neural-fly enables rapid learning for agile flight in strong winds. Science Robotics, 7(66):

eabm6597, 2022.

[36] Tianhe Ren, Shilong Liu, Ailing Zeng, Jing Lin, Kunchang Li, He Cao, Jiayu Chen, Xinyu Huang, Yukang Chen, Feng Yan, et al. Grounded sam: Assembling openworld models for diverse visual tasks. arXiv preprint arXiv:2401.14159, 2024.

[37] Pedro Sanchez-Cuevas, Guillermo Heredia, and Anibal Ollero. Characterization of the aerodynamic ground effect and its influence in multirotor control. International Journal of Aerospace Engineering, 2017(1): 1823056, 2017.

[38] Jiaming Song, Arash Vahdat, Morteza Mardani, and Jan Kautz. Pseudoinverse-guided diffusion models for inverse problems. In International Conference on Learning Representations, 2023.

[39] Matthew Tancik, Ethan Weber, Evonne Ng, Ruilong Li, Brent Yi, Terrance Wang, Alexander Kristoffersen, Jake Austin, Kamyar Salahi, Abhik Ahuja, et al. Nerfstudio: A modular framework for neural radiance field development. In ACM SIGGRAPH 2023 conference proceedings, pages 1–12, 2023.

[40] Justin Thomas, Giuseppe Loianno, Koushil Sreenath, and Vijay Kumar. Toward image based visual servoing for aerial grasping and perching. In 2014 IEEE International Conference on Robotics and Automation (ICRA), pages 2113–2118, 2014. doi: 10.1109/ICRA.2014.6907149.

[41] Miguel Angel Trujillo, Jose Ramiro Mart´ ´ınez-de Dios, Carlos Mart´ın, Antidio Viguria, and An´ıbal Ollero. Novel aerial manipulator for accurate and robust industrial ndt contact inspection: A new tool for the oil and gas inspection industry. Sensors, 19(6):1305, 2019.

[42] Anurag Sai Vempati, Mina Kamel, Nikola Stilinovic, Qixuan Zhang, Dorothea Reusser, Inkyu Sa, Juan Nieto, Roland Siegwart, and Paul Beardsley. Paintcopter: An autonomous uav for spray painting on three-dimensional surfaces. IEEE Robotics and Automation Letters, 3(4): 2862–2869, 2018. doi: 10.1109/LRA.2018.2846278.

[43] Robin Verschueren, Gianluca Frison, Dimitris Kouzoupis, Jonathan Frey, Niels van Duijkeren, Andrea Zanelli, Branimir Novoselnik, Thivaharan Albin, Rien Quirynen, and Moritz Diehl. acados—a modular open-source framework for fast embedded optimal control. Mathematical Programming Computation, 14 (1):147–183, 2022.

[44] Vickie Ye, Ruilong Li, Justin Kerr, Matias Turkulainen, Brent Yi, Zhuoyang Pan, Otto Seiskari, Jianbo Ye, Jeffrey Hu, Matthew Tancik, et al. gsplat: An open-source library for gaussian splatting. Journal of Machine Learning Research, 26(34):1–17, 2025.

[45] Haijie Zhang, Jiefeng Sun, and Jianguo Zhao. Compliant bistable gripper for aerial perching and grasping. In 2019 International Conference on Robotics and Automation (ICRA), pages 1248–1253. IEEE, 2019.

[46] Yuhang Zhang, Haosheng Yu, Jiaping Xiao, and Mir Feroskhan. Grounded vision-language navigation for uavs with open-vocabulary goal understanding. arXiv

preprint arXiv:2506.10756, 2025.

[47] Tony Z. Zhao, Vikash Kumar, Sergey Levine, and Chelsea Finn. Learning fine-grained bimanual manipulation with low-cost hardware, 2023. URL <https://arxiv>. org/abs/2304.13705.

[48] Xueyang Zhou, Yangming Xu, Guiyao Tie, Yongchao Chen, Guowen Zhang, Duanfeng Chu, Pan Zhou, and Lichao Sun. Libero-pro: Towards robust and fair evaluation of vision-language-action models beyond memorization. arXiv preprint arXiv:2510.03827, 2025.

[49] Claudio Zito and Eliseo Ferrante. One-shot learning for autonomous aerial manipulation. Frontiers in Robotics and AI, 9:960571, 2022. doi: 10.3389/frobt.2022.960571. URL <https://doi.org/10.3389/frobt.2022.960571>.

## VII. APPENDIX

## A. Training Details

Here we provide training details for the fine-tuned $\pi _ { 0 }$ policies. The baseline methods were trained using the LeRobot [8] code base with default configurations.

1) Pretraining: We initialize from the public $\pi _ { 0 }$ base checkpoint [3], which is pretrained on large-scale multiembodiment robot manipulation data.

2) Fine-tuning: We fine-tune on our drone dataset consisting of 120 to 150 (or about 10 hours) of real demonstrations for the manipulation and navigation tasks, respectively. The navigation dataset is further augmented to 200 total trajectories via Gaussian-splat trajectories (Sec. III-E). Fine-tuning is performed for 30,000 gradient steps using AdamW optimization with a cosine learning-rate schedule with warmup: the learning rate linearly warms up over the first 1,000 steps to a peak value of $2 . 5 \times 1 0 ^ { - 5 }$ and then decays following a cosine schedule to $2 . 5 \times 1 0 ^ { - 6 }$ by the end of training. We use a global batch size of 32 and an action chunk horizon of $H = 5 0$ . Observations consist of three RGB views (one external, two bodymounted) and robot proprioception. Actions are represented in the dataset’s native 4-DoF plus gripper control space. During training, we pad both proprioception and actions to match the model’s expected dimensionality; at inference, we extract the first 7 action dimensions as the gripper is in the 7th dimension.

3) Inference Configuration: At inference time, for the vanilla implementation, actions are executed at 10 Hz and we run inference once every 50 executed actions (i.e., at 0.2 Hz). We use RTC [4] with a chunk horizon of $H = 2 5$ , using an exponential prefix-attention schedule with a prefix-attention horizon of 25 and 10 denoising steps per chunk.

For payload-aware guidance (Sec. III-D), we set $\lambda _ { z } = 0 . 5 ,$ $\Delta z ~ = ~ 0 . 1 5 \mathrm { m }$ , and $w _ { t } ~ = ~ ( t / ( H - 1 ) ) ^ { \gamma }$ with $\gamma ~ = ~ 1$ concentrating guidance on later timesteps where post-grasp transport dominates. The payload indicator used to gate the vertical objective (Eq. 9) is computed via a deterministic heuristic from the previous chunk’s gripper command history together with the current gripper state: we average the last $K { = } 4$ gripper commands to obtain a continuous close/open intent score, combine command-based open intent with an aperture-based open score, and form the final payload flag as a clipped product of close intent and a soft closedness score derived from the measured gripper aperture.

## B. AirVLA Generalization

A natural question after fine-tuning is how well the policy generalizes to out-of-distribution (OOD) settings. We probe this along three axes in our pick-and-place task: (i) novel object generalization, (ii) novel language prompt generalization, and (iii) novel environment generalization. We emphasize that these evaluations are not meant to demonstrate robust open-world aerial manipulation. Rather, they test whether a manipulation-pretrained VLA can retain useful behavior after transfer to an underactuated aerial platform.

We ran experiments varying the object and fixing the language prompt in the pick-and-place task: “pick up the stuffed animal and put it in the blue bucket”. When changing the stuffed animal from a penguin to a goomba (see Figure 4), we noticed that the policy would reliably attempt to pick the novel object as long as it was initially within the training region on the table. This suggests that the fine-tuned policy transfers a grasping behavior beyond the exact training sequence, but that task performance is tightly linked to a workspace prior such as familiar grasping regions on the table.

![](<images/6b7ba99568394557cc2dbe78d1e8fa62197158397e5d88f4fe5c7bbbd55a3361.jpg>)  
Fig. 4: Out-of-distribution objects used for generalization probes. From left to right: goomba, box, chips, and sandwich. The policy was fine-tuned on a penguin but evaluated on these novel objects to test object-level generalization beyond training instances.

![](<images/4619e0eb5c2a65d1a7c7916b12dbc2f46f430557f01eeccc8f8f16eba1a4a75e.jpg>)  
Fig. 5: Trajectory visualization for the left gate task.The top three rows display real-world RGB observations from external, forward, and downward cameras. The bottom two rows show corresponding synthetic views rendered from Gaussian Splats.

To evaluate a wider range of novel objects, we varied the task prompt to match the object: “pick up the <OBJECT >and put it in the blue bin.” We detail the results of these evaluations in the main paper (see Table III). The objects used for these experiments are shown in Figure 4. An interesting qualitative observation from these evaluations is that the policy would attempt to pick objects in the training region even when no prompt was supplied. This indicates that part of the behavior may be driven by a strong visuomotor prior (i.e. “if an object is present, attempt to grasp it”) rather than language grounding.

Naturally, such behavior begs the question of how well would the policy perform in the presence of distractor objects. To test this, we repeated the penguin grasp task with an additional out-of-distribution object within the training region. We noticed that regardless of the relative positions within the training region, the policy moved the drone to attempt grasps on the penguin rather than the distractor. This shows that fine-tuning can still induce useful object discrimination in manipulation, even when overall OOD robustness remains limited.

Overall, we view these results as evidence of partial transfer rather than broad generalization. The manipulation OOD results suggest that the pretrained VLA can still produce useful outputs under variation in object class, while the navigation OOD results in the main paper show substantial overfitting to the demonstration data and highlight the difficulty of cross-embodiment transfer when the pretrained model has not encountered a similar task. This interpretation is consistent with recent results showing that VLA models do not typically exhibit generalized zero-shot behavior after supervised finetuning, especially in the small-data regime [15, 48]. Instead, a more realistic promise of VLAs is that supervised fine-tuning can enable few-shot adaptation to related new tasks.

We hypothesize that the limitations observed in these probes are partly driven by data scarcity (about 270 teleoperation demonstrations across all tasks), which can encourage reliance on workspace priors and habitual action routines. As larger and more diverse robot datasets become available, spanning broader embodiments, objects, and environments, we expect models such as AirVLA to improve generalization across each of the settings tested here. In future work, we will test this hypothesis by scaling demonstrations and measuring OOD performance as a function of data diversity and coverage.
