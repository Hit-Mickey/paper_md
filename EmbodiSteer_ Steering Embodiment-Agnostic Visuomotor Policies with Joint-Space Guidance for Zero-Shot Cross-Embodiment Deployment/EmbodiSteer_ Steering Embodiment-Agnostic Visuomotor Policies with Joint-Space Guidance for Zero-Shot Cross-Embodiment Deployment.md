# EmbodiSteer: Steering Embodiment-Agnostic Visuomotor Policies with Joint-Space Guidance for Zero-Shot Cross-Embodiment Deployment

Shihefeng Wang<sup>∗,1,2,3</sup> Kangchen Lv<sup>∗,1,2,3</sup> Mingrui Yu<sup>†,1,2,3</sup> Xiang Li<sup>†,1,2,3</sup> <sup>1</sup>Tsinghua University, <sup>2</sup>Beijing Key Laboratory of Embodied Intelligence Systems, <sup>3</sup>BNRist <sup>∗</sup>Equal contribution. <sup>†</sup>Corresponding author.

![](images/603fd664aace73dcb0451b1303cbb84a0611671271db4a5c75735af1455973ab.jpg)  
Figure 1: We present EMBODISTEER, an inference-time steering framework for embodimentaware deployment of embodiment-agnostic visuomotor policies. Given a trained Cartesian policy (left), EmbodiSteer lifts the sampling process into the target robot’s joint space and incorporates robot embodiment and obstacle guidance during denoising (middle), enabling zero-shot whole-body collision-aware execution across diverse robots without finetuning (right).

Abstract: Scalable robot imitation learning relies on large-scale heterogeneous data from diverse robots or body-free data, making Cartesian end-effector actions a key interface for embodiment-agnostic policy learning. However, end-effectoronly abstraction leaves Cartesian policies unaware of the deployed robot body, making them brittle under robot-specific constraints such as whole-body collision avoidance. To overcome this limitation, we present EMBODISTEER, a trainingfree framework that steers embodiment-agnostic visuomotor policies toward zeroshot, embodiment-aware deployment. EMBODISTEER keeps policy learning in Cartesian space while efficiently lifting inference-time diffusion sampling into the target robot’s joint space via forward kinematics and Jacobian-based updates. With whole-body collision-aware guidance over joint trajectories after each denoising step, the arm can be steered away from collisions while preserving learned end-effector behavior. Across 9 simulated robots, EMBODISTEER reduces collision rate by 46.1 percentage points and increases task success by 28.5 percentage points over Cartesian-only execution; on two physical robots in highly constrained scenarios, the respective improvements are 90.0 and 36.7 percentage points. Our project page is at https://frankwang67.github.io/EmbodiSteer-Page.

Keywords: Cross-Embodiment Deployment, Whole-Body Collision Avoidance

## 1 Introduction

Recent progress in robot imitation learning has been driven by large-scale behavior cloning methods, from diffusion policies [1, 2] to vision-language-action models [3, 4, 5, 6]. One of the key factors behind this scalability is the use of Cartesian end-effector poses as a unified action representation [7], whose dimensionality and semantics are independent of a robot’s degrees of freedom, joint ordering, or link geometry. Consequently, demonstrations from heterogeneous robots, simulators, and bodyfree data collection systems such as UMI [8] can be represented in a shared action space, providing an embodiment-agnostic foundation for cross-embodiment policy learning.

However, the embodiment-agnostic property of Cartesian policies comes with a deploymenttime limitation of body unawareness, which becomes problematic when execution must satisfy embodiment-specific constraints. For example, the entire arm must avoid collisions in cluttered scenes, and predicted end-effector motions may become infeasible near workspace boundaries for a specific robot. Training directly in joint space could introduce embodiment awareness, but joint actions are morphology-specific and unavailable in UMI-style body-free demonstrations, making zero-shot deployment to diverse embodiments difficult without robot-specific finetuning data.

These observations motivate us to revisit the central challenge of cross-embodiment deployment of embodiment-agnostic policies. Existing research on cross-embodiment policy generalization largely treats embodiment differences as shifts in action and state distributions, addressing them through embodiment-specific action decoders [9, 10, 11], reusable low-level action primitives or skills [12, 13, 14], and embodiment-conditioned policy representations [5, 15, 16]. However, these works leave execution-time constraints of the target robot body largely unaddressed, so end-effector motion may transfer across robots while whole-body collisions still cause failures. In this work, we focus on the complementary problem of feasibly and safely executing an embodiment-agnostic policy on different robot arms at inference time. We argue that true cross-embodiment deployment is more than reproducing end-effector behavior across robots, since the policy output must be realized as embodiment-aware joint-space motion that respects the target robot’s whole-body constraints.

To address this problem, we propose EMBODISTEER, a training-free inference framework for embodiment-aware deployment of learned Cartesian policies. The policy is trained only with the end-effector actions, thus preserving its compatibility with embodiment-agnostic demonstrations. At inference, EMBODISTEER lifts Cartesian diffusion sampling process into the target robot’s joint space, where embodiment-specific constraints can steer the sampled motion toward collision-free, robot-feasible execution. Concretely, EMBODISTEER maintains a target-robot joint trajectory as the sampling state, but queries the frozen denoiser in its original Cartesian action space via forward kinematics. Each denoised Cartesian target is tracked in joint space with damped Jacobian updates, while CBF-inspired whole-body guidance applies joint corrections that move the robot body away from obstacles without largely disrupting the policy-predicted end-effector motion.

We validate EMBODISTEER by directly deploying trained Cartesian policies across robot embodiments in whole-body collision-aware manipulation scenarios, covering 9 simulated and 2 real robot embodiments. Compared with Cartesian-only execution, EMBODISTEER reduces collision rate and increases task success by 46.1 and 28.5 percentage points, respectively, in simulation, while consistently outperforming post-processing variants and cost-gradient-based steering baselines. On two physical robots, the respective improvements are 90.0 and 36.7 percentage points in highly constrained manipulation tasks, without robot-specific finetuning.

Overall, our contributions are summarized as follows:

1. We propose EMBODISTEER, a training-free inference framework that lifts inference-time sampling of embodiment-agnostic Cartesian policies into the target robot’s joint space, enabling zeroshot cross-embodiment deployment while retaining the scalability of Cartesian policy learning.

2. We introduce CBF-inspired whole-body collision-aware guidance for sampled joint trajectories, which uses small corrective joint updates to steer the robot body away from obstacles while leveraging a Jacobian-based task-space cost to preserve the learned end-effector behavior.

3. We design systematic simulation and real-world benchmarks for cross-embodiment deployment, demonstrating that EMBODISTEER achieves zero-shot whole-body collision-aware execution across diverse embodiments while outperforming Cartesian-only execution and other steering baselines.

## 2 Related Work

Cross-Embodiment Policy Learning. Large-scale cross-embodiment pretraining over diverse robot data is central to robot generalist policies, and a common strategy is to unify heterogeneous data in a shared action space. Cartesian end-effector poses are widely adopted for their embodimentagnostic semantics, and also support body-free data collection systems [8, 17, 18]. Beyond directly mixing data from different platforms or aligning coordinate frames for training, prior works have explored several forms of embodiment alignment, such as latent action spaces with embodimentspecific decoders [9, 10, 19], reusable action primitives or skill abstractions [12, 13, 14], and embodiment-conditioned policy representations [20, 21, 15, 5]. More recently, ego-centric approaches use human hand motion as a shared intermediate space [22, 23, 24]. These approaches address observation, state, and action distribution shifts across embodiments at the policy-learning level. Complementing this line of work, XMoP [25] learns a goal-conditioned whole-body motion planner from synthetic multi-embodiment planning demonstrations and transfers to unseen robots with the same policy weights. Our work instead introduces target-robot whole-body guidance at deployment time for an existing embodiment-agnostic Cartesian manipulation policy, without replacing or finetuning it.

Steering Diffusion Generation with Guidance. Diffusion-based generation in robotics can be steered at inference time without modifying pretrained models, through mechanisms such as noise biasing [26], post-hoc filtering [27, 28], and guided sampling. While classifier-free guidance [29] requires guidance signals during training [30], classifier guidance [31] can leverage inference-time gradients without retraining, and has been widely applied to robotic diffusion generation [32, 33, 34, 35, 36, 37]. For visuomotor policies, DynaGuide [38] uses a learned dynamics model to steer base policies toward complex objectives, and ITPS [39] incorporates real-time user interactions through test-time guidance. JM2D [40] combines pretrained diffusion planners with model-based constraint modules through joint importance sampling, without additional training. Most related to our setting, EADP [41] steers embodiment-agnostic policies for embodiment-aware deployment, but focuses on low-level controller tracking costs for aerial manipulators. We instead target collision-aware wholebody execution for robot arms by lifting Cartesian policies into joint-space sampling and applying collision-aware guidance.

Collision-Aware Visuomotor Policies. Safety-critical robot manipulation in cluttered environments requires explicit collision awareness. Learning-based visuomotor policies can incorporate safety through post-processing mechanisms, such as control barrier functions [42, 43, 44], model predictive filters [45, 46], and risk estimators [47, 28]. For diffusion-based policies, obstacle-related costs can further steer the sampling process toward safer trajectories at inference time [48, 49, 50]. Whole-body collision avoidance has also been studied in learned motion planning: MπNets [51] generates robot motion from depth point clouds, while MPD [52] combines learned diffusion trajectory priors with collision-cost guidance. Recent works address robot-body safety by encoding whole-body geometry with point clouds, keypoints, or signed distance fields and learning from collision-free demonstrations [53, 54, 55, 56]. These methods learn whole-body motion representations or policies from dedicated planning or robot-body demonstrations. In contrast, we use a pretrained embodiment-agnostic policy as the base and introduce whole-body collision constraints during zero-shot inference.

## 3 Preliminaries: Cartesian Diffusion Policy

Imitation learning aims to learn a visuomotor policy π from expert demonstrations, which maps observations such as images and proprioception to actions. Diffusion Policy [1] formulates action generation as a conditional sampling problem with diffusion models. During training, Gaussian noise is added to a clean action sample $a _ { 0 }$ from the dataset according to $A _ { t } = \sqrt { \bar { \alpha } _ { t } } A _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon _ { t }$ at diffusion timestep $t ,$ where $\bar { \alpha } _ { t }$ denotes noise schedule coefficient and $\epsilon _ { t } \sim \mathcal { N } ( 0 , I )$ . The denoising model is trained to predict the added noise by minimizing $\mathcal { L } = \mathrm { M S E L o s s } ( \epsilon _ { t } , \epsilon _ { \theta } ( A _ { t } , o , t ) )$ ). At inference time, starting from $A _ { T } \sim \mathcal { N } ( 0 , I )$ , the sample is iteratively denoised by the DDPM reverse

![](images/b1465995c689a4a5d58cfdafc710bc8322e5b0b02583c666c6248af02bf820b6.jpg)  
Figure 2: Overview of EMBODISTEER. (a) A trained Cartesian policy is lifted into the target-robot joint space for inference-time sampling. (b) CBF-inspired guidance steers sampled joint trajectories away from whole-body collisions while preserving end-effector behavior. (c) Visualization of realworld deployment across diverse robot embodiments on constrained manipulation tasks.

update as

$$
A _ {t - 1} = \frac {1}{\sqrt {\alpha_ {t}}} \left(A _ {t} - \frac {1 - \alpha_ {t}}{\sqrt {1 - \bar {\alpha} _ {t}}} \epsilon_ {\theta} (A _ {t}, o, t)\right) + \sigma_ {t} z, \quad z \sim \mathcal {N} (0, I).\tag{1}
$$

We use Cartesian end-effector poses as the action space for embodiment-agnostic policies, which predict an action chunk $A = [ a _ { 1 } , \dotsc , a _ { H } ]$ over horizon H. Here, each action $a _ { i } = \mathbf { \bar { \phi } } [ \delta p _ { i } , \ r _ { i } ] \in \mathbb { R } ^ { 9 }$ is represented relative to the chunk-start end-effector pose, where $\delta p _ { i } \in \mathbb { R } ^ { 3 }$ and $r _ { i } \in \mathbb { R } ^ { 6 }$ denote the relative translation and 6D rotation representation [57], respectively. Notably, the gripper command is also included in the action but will be omitted for writing brevity in the following sections.

## 4 Method

As shown in Fig. 2, the Cartesian end-effector policy is trained on embodiment-agnostic demonstrations and maintained frozen during cross-embodiment inference-time steering. In this section, we first detail our problem formulation of whole-body collision-aware deployment, followed by our proposed joint-space denoising and CBF-inspired whole-body collision-aware guidance mechanism.

## 4.1 Problem Formulation

We consider deploying a frozen embodiment-agnostic Cartesian diffusion policy π on a target robot embodiment E with obstacle information O. The policy generates an end-effector action chunk $A =$ $[ a _ { 1 } , \dotsc , a _ { H } ]$ in Cartesian space, but has no access to the robot’s joint configuration and whole-body collision state during sampling. Therefore, the target-robot joint trajectory $Q = [ q _ { 1 } , \dotsc , q _ { H } ]$ , where $q _ { i } \in \mathbb { R } ^ { D }$ for robot with D controllable arm joints, is required for reasoning about embodimentspecific constraints. The desired whole-body safety condition is

$$
h (q _ {i}) = \mathrm{SDF} (\mathcal {E} (q _ {i}), \mathcal {O}) \geq d _ {\mathrm{safe}}, \quad \forall i = 1, \dots , H,\tag{2}
$$

where positive signed distance indicates clearance, negative values indicate penetration, and $d _ { \mathrm { s a f e } } >$ 0 is the clearance margin.

## 4.2 Joint-Space Denoising with Pretrained Cartesian Diffusion Policy

A straightforward way to lift Cartesian denoising into joint space is to solve inverse kinematics under the target robot E after each denoising step, i.e., $q _ { t , i } = \mathrm { I K } _ { \mathcal { E } } ( a _ { t , i } )$ [37]. However, repeatedly solving IK inside the denoising loop is costly and hinders real-time deployment. Instead, we treat the Cartesian denoising update as a local task-space residual and map it to joint space using the target robot’s Jacobian $J _ { \mathcal { E } } ( \boldsymbol { q } ) \stackrel { - } { \in } \mathbb { R } ^ { 6 \times D }$ . Since adjacent denoising steps typically induce small Cartesian changes, the first-order kinematic relation provides a reliable joint-space update without IK solving at every step. Notably, the damped pseudoinverse Jacobian $J _ { \mathcal { E } } ^ { + } ( \boldsymbol { q } ) = J _ { \mathcal { E } } ( \boldsymbol { q } ) ^ { \top } \left( J _ { \mathcal { E } } ( \boldsymbol { q } ) J _ { \mathcal { E } } ( \boldsymbol { q } ) ^ { \top } + \lambda _ { \mathrm { p i n v } } I \right) ^ { - 1 }$ is used here to improve numerical stability and prevent large joint motions near singularities, where $\lambda _ { \mathrm { p i n v } }$ is the damping factor.

```txt
Algorithm 1: EMBODISTEER: Joint-Space Inference with Whole-Body Collision Guidance
Input: Observation o, obstacles O, embodiment E
Output: Joint action chunk Q0
1 Sample Cartesian noise aT,i and initialize QT = [qT,1, ..., qT,H] using (3);
2 for t = T, ..., 1 do
    // Joint Space Denoising
3    Map Qt to Cartesian action At using FKε;    // Kinematic mapping
4    At-1 ← DiffusionStep(εθ(At, t, o), t, At);    // Frozen Cartesian policy
5    [ξt,i] ← Tpose (At-1At-1);    // Cartesian denoising residual
6    Update joint space sample Qt-1 diff using (4);    // Jacobian residual update
    // Whole-Body Collision-Aware Guidance
7    Solve (6) in batch to get solution ΔQt-1;    // Solve QP
8    Apply the solution ΔQt-1 to Qt-1 diff using (7) to get Qt-1;    // Apply the guidance
9 end
10 return Q0
```

Denoising Initialization. We initialize joint-space sampling by drawing Cartesian Gaussian noise $a _ { T , i } \in \mathbb { R } ^ { 9 }$ and locally mapping it around the chunk-start configuration $q ^ { \mathrm { s t a r t } }$

$$
q _ {T, i} = q ^ {\mathrm{start}} + \alpha \cdot \operatorname{clip} \left(J _ {\mathcal {E}} ^ {+} (q ^ {\mathrm{start}}) \xi_ {T, i}, - \Delta q _ {\max}, \Delta q _ {\max}\right), \quad \xi_ {T, i} = \mathcal {T} _ {\mathrm{pose}} (a _ {T, i}).\tag{3}
$$

Here, $\mathcal { T } _ { \mathrm { p o s e } }$ converts the Cartesian noise into a 6D pose twist relative to the chunk-start end-effector pose, $\Delta q _ { \mathrm { m a x } }$ bounds the per-joint perturbation, and α is a global scaling factor. This local kinematic mapping keeps joint-space samples consistent with the Cartesian diffusion initialization.

Joint-Space Denoising Update. At denoising step t, EMBODISTEER maintains a noisy joint action chunk $Q _ { t } = [ q _ { t , 1 } , \dots , q _ { t , H } ]$ . We first project $Q _ { t }$ through forward kinematics and convert the resulting end-effector poses into the relative Cartesian action chunk $A _ { t } .$ . The frozen Cartesian denoising network is then queried in its original action space, yielding the Cartesian denoising target $A _ { t - 1 } =$ DifusionStep $\left( \epsilon _ { \theta } ( A _ { t } , t , o ) , t , A _ { t } \right)$ . Each denoising update is converted into a local 6D pose twist $\xi _ { t , i } = \mathcal { T } _ { \mathrm { p o s e } } ( a _ { t , i } ^ { - 1 } a _ { t - 1 , i } )$ , and mapped back to the joint space with the damped Jacobian:

$$
q _ {t - 1, i} ^ {\mathrm{diff}} = q _ {t, i} + \mathrm{clip} \big (J _ {\mathcal {E}} ^ {+} (q _ {t, i}) \xi_ {t, i}, - \Delta q _ {\mathrm{max}}, \Delta q _ {\mathrm{max}} \big).\tag{4}
$$

## 4.3 CBF-Inspired Whole-Body Collision-Aware Guidance

Maintaining the denoising state in joint space allows EMBODISTEER to consider whole-body constraints during sampling. A direct approach is energy-based guidance with gradients over the joint trajectory[48, 58], which penalizes unsafe configurations at each denoising step: $q _ { t - 1 , i } \ =$ $q _ { t - 1 , i } ^ { \mathrm { d i f f } } - \bar { \rho } _ { t } \nabla _ { q } \bar { \mathcal { L } } _ { \mathrm { c o l l } } \left( q _ { t - 1 , i } ^ { \mathrm { d i f f } } \right)$ , where $\mathcal { L } _ { \mathrm { c o l l } } ( q )$ is the cost function punishing whole-body collision, and $\rho _ { t }$ is the guidance strength. However, because whole-body collision gradients often originate from non-end-effector links such as the elbow or forearm, pushing the robot body away from obstacles can also perturb the end-effector trajectory needed for task completion.

CBF-Inspired Guidance Formulation. EMBODISTEER adopts a CBF-inspired collision-aware guidance to encourage whole-body safety while preserving end-effector motion. For clarity, we describe the guidance for a single horizon step $q _ { t - 1 , i } ^ { \mathrm { d i f f } } ,$ and apply the same update independently to all steps in the horizon. We compute a guided joint configuration $q _ { t - 1 , i }$ by solving

$$
\min _ {q _ {t - 1, i}} \frac {1}{2} \left\| \mathrm{FK} \left(q _ {t - 1, i} ^ {\text {diff}}\right) ^ {- 1} \mathrm{FK} \left(q _ {t - 1, i}\right) \right\| _ {W} ^ {2} + \frac {1}{2} \lambda_ {\text {cbf}} \left\| q _ {t - 1, i} - q _ {t - 1, i} ^ {\text {diff}} \right\| ^ {2}, \quad \text {s.t.} h (q _ {t - 1, i}) \geq d _ {\text {safe}},\tag{5}
$$

Whole-Body Safety-Critical Manipulation Tasks  
![](images/b2cf1121222c1c2ffe7880260891df0b38f23220841ab2855f567e70c2f18258.jpg)  
Figure 3: We evaluate EMBODISTEER on three manipulation tasks requiring both task completion and whole-body obstacle avoidance. Cartesian policies are trained from obstacle-free floatinggripper demonstrations and deployed zero-shot on 9 robot embodiments with test-time obstacles.

where $W$ is a diagonal weight matrix, and $\lambda _ { \mathrm { c b f } }$ balances task-space preservation and joint-space regularization. The objective tracks the diffusion update in both task and joint space, while allowing the safety constraint to move the robot body away from obstacles when necessary.

QP Linearization for Efficient Update. To obtain a lightweight per-step update, we linearize the forward kinematics and safety function at $q _ { t - 1 , i } ^ { \mathrm { d i f f } }$ . Let $\Delta q _ { t - 1 , i } = q _ { t - 1 , i } - q _ { t - 1 , i } ^ { \mathrm { d i f f } }$ . Substituting FK $\left( q _ { t - 1 , i } ^ { \mathrm { d i f f } } \right) ^ { - 1 } \mathrm { F K } \left( q _ { t - 1 , i } \right) \approx J _ { \mathcal { E } } \left( q _ { t - 1 , i } ^ { \mathrm { d i f f } } \right) \Delta q _ { i }$ and $h \left( q _ { t - 1 , i } \right) \approx h \left( q _ { t - 1 , i } ^ { \mathrm { d i f f } } \right) + \nabla _ { q } h \left( q _ { t - 1 , i } ^ { \mathrm { d i f f } } \right) ^ { \top } \Delta q _ { i }$ into (5) yields the single-constraint QP:

$$
\min _ {\Delta q _ {i}} \frac {1}{2} \Delta q _ {t - 1, i} ^ {\top} H _ {\mathcal {E}} \left(q _ {t - 1, i} ^ {\mathrm{diff}}\right) \Delta q _ {t - 1, i}, \quad \mathrm{s.t.} \nabla_ {q} h \left(q _ {t - 1, i} ^ {\mathrm{diff}}\right) ^ {\top} \Delta q _ {t - 1, i} \geq \gamma_ {t} \left[ d _ {\mathrm{safe}} - h \left(q _ {t - 1, i} ^ {\mathrm{diff}}\right) \right].\tag{6}
$$

where $H _ { \mathcal { E } } \left( q _ { t - 1 , i } ^ { \mathrm { d i f f } } \right) = J _ { \mathcal { E } } \left( q _ { t - 1 , i } ^ { \mathrm { d i f f } } \right) ^ { \top } W J _ { \mathcal { E } } \left( q _ { t - 1 , i } ^ { \mathrm { d i f f } } \right) + \lambda _ { \mathrm { c b f } } I ,$ and $\gamma _ { t }$ controls the strength of the constraint. Since the QP has only one linear inequality, it can be solved efficiently in closed form or by a batched small-scale QP solver. Denoting the solution as $\Delta q _ { t - 1 , i } ^ { \mathrm { c b f } }$ , the final guided sample is

$$
q _ {t - 1, i} = q _ {t - 1, i} ^ {\mathrm{diff}} + \mathrm{clip} (\Delta q _ {t - 1, i} ^ {\mathrm{cbf}}, - \Delta q _ {\max} ^ {\mathrm{cbf}}, \Delta q _ {\max} ^ {\mathrm{cbf}}).\tag{7}
$$

## 5 Results

## 5.1 Simulation Results

Experiment Setup. We design three challenging manipulation tasks in ManiSkill [59] to evaluate body-aware cross-embodiment deployment, as shown in Fig. 3. These tasks require both accurate end-effector motion and collision avoidance between surrounding obstacles and the full robot body. For each task, we collect 200 diverse demonstrations with a floating gripper, recording only wrist-view RGB observations, end-effector states, and end-effector actions to keep the policy embodiment-agnostic. The trained Cartesian policy is deployed zero-shot on 9 robot arms with different morphologies, while sharing the same gripper and wrist-camera mounting.

Baselines. We consider four baseline groups: 1) Cartesian denoising without guidance (EE) directly executes the Cartesian end-effector actions predicted by the learned policy; 2) Joint-space denoising without guidance (Joint) lifts Cartesian sampling into the target robot’s joint space without collision guidance, isolating the effect of joint-space denoising alone; 3) Cartesian denoising with post-hoc correction leaves Cartesian sampling unchanged and applies whole-body collision avoidance only after trajectory generation, where EE w/ Sampling samples multiple actions and selects the candidate with the largest realized clearance, and EE w/ CBF solves IK followed by a one-shot CBF-QP correction; 4) Joint-space denoising with guidance adds collision-aware guidance during every joint-space denoising step, where Joint w/ CG uses cost-gradient guidance and EMBODISTEER uses our proposed task-aware CBF-QP guidance.

![](images/3dc84256970c551fc1bebdd4519104e1738e95091e7c2672e962fc65ecb42634.jpg)

Table 1: Quantitative results on manipulation tasks with obstacles, averaged over 9 robot embodiments. TSR is task success rate (%), RWD is maximum reward per episode, and COR is collision occurrence rate (%). Best results among obstacle-present methods are shown in bold. Repeatedseed aggregate results are reported in Tab. 2.

<table><tr><td rowspan="2">Obstacles</td><td rowspan="2">Method</td><td colspan="3">PlaceToast</td><td colspan="3">TurnOnFaucet</td><td colspan="3">MakeCoffee</td><td colspan="3">Average</td></tr><tr><td>TSR↑</td><td>RWD↑</td><td>COR↓</td><td>TSR↑</td><td>RWD↑</td><td>COR↓</td><td>TSR↑</td><td>RWD↑</td><td>COR↓</td><td>TSR↑</td><td>RWD↑</td><td>COR↓</td></tr><tr><td rowspan="2">w/o Obs.</td><td>EE</td><td>96.8</td><td>.980</td><td>-</td><td>83.6</td><td>.937</td><td>-</td><td>89.7</td><td>.932</td><td>-</td><td>90.0</td><td>.950</td><td>-</td></tr><tr><td>Joint</td><td>94.4</td><td>.965</td><td>-</td><td>87.9</td><td>.957</td><td>-</td><td>89.7</td><td>.929</td><td>-</td><td>90.7</td><td>.950</td><td>-</td></tr><tr><td rowspan="5">w/ Obs.</td><td>EE</td><td>43.4</td><td>.634</td><td>52.6</td><td>41.6</td><td>.752</td><td>61.8</td><td>22.2</td><td>.456</td><td>58.3</td><td>35.7</td><td>.614</td><td>57.6</td></tr><tr><td>EE w/ Sampling</td><td>47.1</td><td>.661</td><td>51.3</td><td>43.9</td><td>.767</td><td>53.2</td><td>24.1</td><td>.478</td><td>60.0</td><td>38.4</td><td>.635</td><td>54.8</td></tr><tr><td>EE w/ CBF</td><td>65.8</td><td>.764</td><td>15.3</td><td>56.3</td><td>.797</td><td>29.9</td><td>48.9</td><td>.606</td><td>8.6</td><td>57.0</td><td>.722</td><td>17.9</td></tr><tr><td>Joint w/ CG</td><td>57.7</td><td>.687</td><td>0.2</td><td>10.3</td><td>.403</td><td>59.6</td><td>22.4</td><td>.424</td><td>40.6</td><td>30.1</td><td>.505</td><td>33.5</td></tr><tr><td>EMBODISTEER</td><td>74.8</td><td>.829</td><td>4.6</td><td>60.6</td><td>.833</td><td>27.1</td><td>57.2</td><td>.670</td><td>2.8</td><td>64.2</td><td>.777</td><td>11.5</td></tr></table>

Figure 4: Per-embodiment comparison on the PLACETOAST task, where the task success rate (left) and collision rate (right) across five representative robot embodiments are reported.

Results and Analysis. Tab. 1 reports quantitative results averaged over 9 robot embodiments. In obstacle-free settings, the Cartesian policy transfers directly across diverse arms with over 90% average task success. Joint-space denoising preserves task success and even yields a slight improvement, showing that sampling in joint space maintains the learned end-effector behavior. With obstacles, however, Cartesian execution drops to 35.7% success and 0.614 reward, with 57.6% collision rate, revealing the limitation of end-effector-only deployment under whole-arm collision constraints. Repeating the obstacle-present evaluations with two additional random seeds yields the three-seed aggregate mean ± standard deviation in Tab. 2.

Among obstacle-aware methods, EMBODISTEER achieves the best overall balance between task success and whole-body collision avoidance. Fig. 4 further compares success rate and collision rate across different embodiments, showing consistent gains over baselines. EE w/ Sampling only modestly improves success by 2.7 percentage points and reduces collision failures by 2.8 percentage points, while incurring much higher cost from repeated sampling and collision checking. EE w/CBF provides a stronger post-hoc correction, improving success by over 20 percentage points, but applies safety only after denoising and cannot shape the generated trajectory. In contrast, EMBODISTEER injects guidance at every joint-space denoising step, producing safer and more feasible trajectories.

Joint w/ CG exhibits large task-dependent variation. It performs competitively on PLACETOAST, where coarse end-effector motion is often sufficient, but struggles on TURNONFAUCET and MAKE-COFFEE, which require precise interactions. Although collision-cost gradients can push the arm away from obstacles, they can also significantly distort the end-effector trajectory. This highlights the benefit of EMBODISTEER’s CBF-QP guidance, which moves the robot body while minimizing deviation from the policy-predicted Cartesian motion. Finally, the constraint strength (the scale of $\gamma )$ ablation shows the trade-off between safety and task preservation. With $\gamma = 0$ , EMBODISTEER reduces to joint-space denoising without collision guidance. Increasing γ initially reduces collisions and improves success, but overly large guidance eventually lowers task success by driving samples away from the learned policy distribution.

Table 2: Simulation results with obstacles, averaged over three tasks and nine robot embodiments. Entries report aggregate mean ± standard deviation across three evaluation seeds.

<table><tr><td>Method</td><td>TSR (%) ↑</td><td>RWD ↑</td><td>COR (%) ↓</td></tr><tr><td>EE</td><td>36.0 ± 0.2</td><td>.614 ± .004</td><td>56.8 ± 0.9</td></tr><tr><td>EE w/ Sampling</td><td>37.7 ± 0.5</td><td>.621 ± .012</td><td>53.1 ± 1.3</td></tr><tr><td>EE w/ CBF</td><td>57.2 ± 0.6</td><td>.726 ± .005</td><td>18.3 ± 0.3</td></tr><tr><td>Joint w/ CG</td><td>30.3 ± 0.5</td><td>.503 ± .003</td><td>33.5 ± 0.6</td></tr><tr><td>EMBODISTEER</td><td>63.3 ± 1.4</td><td>.771 ± .011</td><td>11.7 ± 0.8</td></tr></table>

![](images/369b5bba8ee5cf13ccd4b7a7c7b1fce5276993ee07c71371ef276546dc153321.jpg)  
Figure 6: Demonstration coverage on MAKECOFFEE: varying the number of demonstrations (left) and motion-planning waypoint randomization relative to the original data-generation setting (right).

Alternative Constraint-Integration Baseline. To evaluate another training-free strategy for combining the frozen Cartesian policy with whole-body constraints, we implement a JM2D-style baseline [40] that jointly samples trajectories and collision corrections through repeated candidate denoising. Both methods use the same frozen Cartesian policy. On PLACETOAST with Panda, under identical RTX 4090, batch-1, CUDA-synchronized conditions, the adaptation requires 8.65 s per chunk with 16 proposals and 16 denoising steps, compared with 0.254 s for EMBODIS-TEER. Even reducing JM2D to four steps requires 3.27 s per chunk. This computational cost limits the practicality

![](images/aec6ea240323d7338b879f54a6431d011c1db5d82356acdb51b2f5384f3640bc.jpg)  
Figure 5: Ablation for the constraint strength on TURNONFAUCET.

of the adaptation for closed-loop replanning in our setting, so we report its computational feasibility rather than a full task-performance comparison.

Demonstration Coverage. Collision guidance perturbs the motions proposed by the Cartesian policy, so task performance under steering also depends on the behaviors covered by its demonstrations. We examine this dependence on MAKECOFFEE by retraining the Cartesian policy with 20, 50, 100, or 200 demonstrations. Separately, we vary trajectory diversity by scaling the waypoint randomization used during motion-planning data generation to 10%, 33%, 67%, or 100% of the original setting, keeping 200 demonstrations at each diversity level (Fig. 6). For each training setting, we compare unguided execution with the same guidance strategies used in the main benchmark. Increasing either quantity or diversity improves task success, with the broader coverage helping the policy complete tasks under guidance-induced motion perturbations. Additional obstacle-related evaluations are provided in Appendix C.5.

![](images/94c98c94ec65225021c945fd2615fb340a9219cbf9d06d7d0bbbf0481622bc24.jpg)  
Figure 7: Real-world deployment of an arm-agnostic Cartesian policy on UR5 and Panda. Without guidance, the base policy often preserves task intent but collides with obstacles (yellow region), whereas EMBODISTEER steers joint trajectories to avoid arm-body collisions and complete tasks.

## 5.2 Real-World Experiments

Experiment Setup. We further evaluate EMBODISTEER on two physical embodiments, UR5 and Franka Panda, across three tabletop manipulation tasks, as shown in Fig. 7. For each task, we collect 200 demonstrations without extra obstacles, with the UMI handheld-gripper [8] and train a Cartesian diffusion policy without robot-specific demonstrations or finetuning. Obstacles are introduced only at deployment time, thus directly evaluating whether EMBODISTEER can introduce embodimentaware collision avoidance to an embodiment-agnostic policy. EMBODISTEER also supports realtime physical deployment, achieving an inference rate of approximately 9.7 Hz on an RTX 4070 Ti SUPER.

Results. For each robot and task, we conduct 10 no-obstacle trials to evaluate base task competence and 10 obstacle trials to evaluate whole-body collision avoidance. The no-obstacle setting confirms that trained embodiment-agnostic Cartesian policies transfer to both physical embodiments, achieving 49/60 successes. With obstacles, however, the base policy drops to 25/60 successes and collides in 58/60 trials, showing that end-effector-only policies lack reliable embodiment-aware execution. In contrast, EMBODISTEER recovers performance to 47/60 successes while reducing collisions to 4/60. Qualitative comparisons in Fig. 7 further illustrate how EMBODISTEER improves collisionaware deployment while maintaining practical control rates across robots and tasks.

## 6 Limitations & Conclusion

Limitations. EMBODISTEER has several limitations. First, our guidance linearizes the wholebody constraint into a local QP and applies clipped joint corrections for stable execution; therefore, it provides an efficient feasible direction for collision avoidance but does not guarantee globally collision-free motion. Second, the guidance is activated only when the robot enters the SDF margin, making it a local correction mechanism rather than a global planner that can select the optimal collision-free motion modes far in advance. Finally, strong guidance may steer the robot into states outside the training distribution of the Cartesian policy, where task execution can fail, suggesting that broader and more diverse demonstrations remain important for robust deployment.

Conclusions. We propose EMBODISTEER for zero-shot embodiment-aware deployment of Cartesian diffusion policies by lifting inference-time sampling into the target robot’s joint space and applying task-preserving whole-body collision guidance. By keeping policy learning in an embodiment-agnostic Cartesian action space while enforcing robot-specific constraints only at deployment time, the method preserves the scalability of Cartesian imitation learning and adds the body awareness needed for safe execution. Across simulation and real-world experiments, EM-BODISTEER improves task success and substantially reduces collisions over Cartesian execution and post-hoc guidance baselines, demonstrating a practical path toward reusable manipulation policies across diverse robot embodiments.

## Acknowledgments

We thank Yongpeng Jiang for insightful discussions, Ting-Yu Tsui for assistance with the hardware setup, and Chendong Xin for help with the experiments. We also thank the anonymous reviewers and the Area Chair for their constructive feedback.

## References

[1] C. Chi, Z. Xu, S. Feng, E. Cousineau, Y. Du, B. Burchfiel, R. Tedrake, and S. Song. Diffusion policy: Visuomotor policy learning via action diffusion. The International Journal of Robotics Research, 44(10-11):1684–1704, 2025. doi:10.1177/02783649241273668.

[2] Y. Ze, G. Zhang, K. Zhang, C. Hu, M. Wang, and H. Xu. 3D diffusion policy: Generalizable visuomotor policy learning via simple 3D representations. In Proceedings ofRobotics: Science and Systems, 2024. doi:10.15607/RSS.2024.XX.067.

[3] B. Zitkovich, T. Yu, S. Xu, P. Xu, T. Xiao, F. Xia, J. Wu, P. Wohlhart, S. Welker, A. Wahid, et al. RT-2: Vision-language-action models transfer web knowledge to robotic control. In Conference on Robot Learning, pages 2165–2183. PMLR, 2023.

[4] M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. P. Foster, P. R. Sanketi, Q. Vuong, T. Kollar, B. Burchfiel, R. Tedrake, D. Sadigh, S. Levine, P. Liang, and C. Finn. OpenVLA: An open-source vision-language-action model. In Proceedings of The 8th Conference on Robot Learning, volume 270 of Proceedings of Machine Learning Research, pages 2679–2713. PMLR, 2025. URL https://proceedings.mlr.press/v270/kim25c. html.

[5] J. Zheng, J. Li, Z. Wang, D. Liu, X. Kang, Y. Feng, Y. Zheng, J. Zou, Y. Chen, J. Zeng, T. Wang, Y.-Q. Zhang, J. Liu, and X. Zhan. X-VLA: Softprompted transformer as scalable cross-embodiment vision-language-action model. In International Conference on Learning Representations, volume 2026, pages 60580– 60606, 2026. URL https://proceedings.iclr.cc/paper\_files/paper/2026/hash/ 630d293833e09e1ecd892a898a20b074-Abstract-Conference.html.

[6] S. Liu, L. Wu, B. Li, H. Tan, H. Chen, Z. Wang, K. Xu, H. Su, and J. Zhu. RDT-1B: a diffusion foundation model for bimanual manipulation. In International Conference on Learning Representations, volume 2025, pages 29982–30009, 2025.

[7] Y. Wang, S. Zheng, H. Luo, W. Zhang, H. Yuan, C. Xu, H. Xu, Y. Feng, M. Yu, Z. Kang, et al. Rethinking visual-language-action model scaling: Alignment, mixture, and regularization. arXiv preprint arXiv:2602.09722, 2026.

[8] C. Chi, Z. Xu, C. Pan, E. Cousineau, B. Burchfiel, S. Feng, R. Tedrake, and S. Song. Universal manipulation interface: In-the-wild robot teaching without in-the-wild robots. In Proceedings ofRobotics: Science and Systems, 2024. doi:10.15607/RSS.2024.XX.045.

[9] E. Bauer, E. Nava, and R. K. Katzschmann. Latent action diffusion for cross-embodiment manipulation. In 2026 IEEE International Conference on Robotics and Automation (ICRA), pages 511–518, 2026. doi:10.1109/icra57385.2026.11696023.

[10] Q. Bu, Y. Yang, J. Cai, S. Gao, G. Ren, M. Yao, P. Luo, and H. Li. Learning to act anywhere with task-centric latent actions. In Proceedings of Robotics: Science and Systems, 2025. doi: 10.15607/RSS.2025.XXI.014.

[11] L. Wang, X. Chen, J. Zhao, and K. He. Scaling proprioceptive-visual learning with heterogeneous pre-trained transformers. In Advances in Neural Information Processing Systems, volume 37, pages 124420–124450. Curran Associates, Inc., 2024. doi:10.52202/079017-3952.

[12] M. Xu, Z. Xu, C. Chi, M. Veloso, and S. Song. Xskill: Cross embodiment skill discovery. In Conference on robot learning, pages 3536–3555. PMLR, 2023.

[13] L. Zha, A. J. Hancock, M. Zhang, T. Yin, Y. Huang, D. Shah, A. Z. Ren, and A. Majumdar. LAP: Language-action pre-training enables zero-shot cross-embodiment transfer. In Proceedings ofRobotics: Science and Systems, 2026. doi:10.15607/RSS.2026.XXII.203.

[14] J. Zheng, J. Li, D. Liu, Y. Zheng, Z. Wang, Z. Ou, Y. Liu, J. Liu, Y.-Q. Zhang, and X. Zhan. Universal actions for enhanced embodied foundation models. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 22508–22519, 2025. doi:10.1109/ cvpr52734.2025.02096.

[15] A. Patel and S. Song. GET-Zero: Graph embodiment transformer for zero-shot embodiment generalization. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pages 14262–14269. IEEE, 2025.

[16] A. Gupta, L. Fan, S. Ganguli, and L. Fei-Fei. MetaMorph: Learning universal controllers with transformers. In International Conference on Learning Representations, 2022. URL https://metamorph-iclr.github.io/site/.

[17] M. Xu, H. Zhang, Y. Hou, Z. Xu, L. Fan, M. Veloso, and S. Song. DexUMI: Using human hand as the universal manipulation interface for dexterous manipulation. In Proceedings of The 9th Conference on Robot Learning, volume 305 of Proceedings of Machine Learning Research, pages 437–459. PMLR, 2025. URL https://proceedings.mlr.press/v305/ xu25b.html.

[18] H. Ha, Y. Gao, Z. Fu, J. Tan, and S. Song. UMI-on-Legs: Making manipulation policies mobile with manipulation-centric whole-body controllers. In Proceedings of The 8th Conference on Robot Learning, volume 270 of Proceedings ofMachine Learning Research, pages 5254–5270. PMLR, 2025. URL https://proceedings.mlr.press/v270/ha25a.html.

[19] J. Bjorck, F. Castaneda, N. Cherniadev, X. Da, R. Ding, L. Fan, Y. Fang, D. Fox, F. Hu, ˜ S. Huang, et al. GR00T N1: An open foundation model for generalist humanoid robots. arXiv preprint arXiv:2503.14734, 2025.

[20] C. Sferrazza, D.-M. Huang, F. Liu, J. Lee, and P. Abbeel. Body transformer: Leveraging robot embodiment for policy learning. In Proceedings of The 8th Conference on Robot Learning, volume 270 of Proceedings of Machine Learning Research, pages 3407–3424. PMLR, 2025. URL https://proceedings.mlr.press/v270/sferrazza25a.html.

[21] N. Bohlinger, G. Czechmanowski, M. P. Krupka, P. Kicki, K. Walas, J. Peters, and D. Tateo. One policy to run them all: an end-to-end learning approach to multi-embodiment locomotion. In Proceedings of The 8th Conference on Robot Learning, volume 270 of Proceedings of Machine Learning Research, pages 3356–3378. PMLR, 2025. URL https: //proceedings.mlr.press/v270/bohlinger25a.html.

[22] H. Luo, W. Zhang, Y. Feng, S. Zheng, H. Xu, C. Xu, Z. Xi, Y. Fu, and Z. Lu. Being-H0.7: A latent world-action model from egocentric videos. arXiv preprint arXiv:2605.00078, 2026.

[23] R. Yang, Q. Yu, Y. Wu, R. Yan, B. Li, A.-C. Cheng, X. Zou, Y. Fang, X. Cheng, R.-Z. Qiu, et al. EgoVLA: Learning vision-language-action models from egocentric human videos. arXiv preprint arXiv:2507.12440, 2025.

[24] R. Zheng, D. Niu, Y. Xie, J. Wang, M. Xu, Y. Jiang, F. Castaneda, F. Hu, Y. L. Tan, L. Fu, et al.˜ EgoScale: Scaling dexterous manipulation with diverse egocentric human data. arXiv preprint arXiv:2602.16710, 2026.

[25] P. K. Rath and N. Gopalan. XMoP: Whole-body control policy for zero-shot cross-embodiment neural motion planning. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pages 8299–8306, 2025. doi:10.1109/icra55743.2025.11127979.

[26] A. Wagenmaker, Y. Zhang, M. Nakamoto, S. Park, W. Yagoub, A. Nagabandi, A. Gupta, and S. Levine. Steering your diffusion policy with latent space reinforcement learning. In Proceedings of The 9th Conference on Robot Learning, volume 305 of Proceedings of Machine Learning Research, pages 258–282. PMLR, 2025. URL https://proceedings.mlr. press/v305/wagenmaker25a.html.

[27] Y. Liu, J. Hamid, A. Xie, Y. Lee, M. Du, and C. Finn. Bidirectional decoding: Improving action chunking via guided test-time sampling. In International Conference on Learning Representations, volume 2025, pages 4594–4627, 2025.

[28] R. Romer, J. Balletshofer, J. Thumm, M. Pavone, A. P. Schoellig, and M. Althoff. From¨ demonstrations to safe deployment: Path-consistent safety filtering for diffusion policies. In 2026 IEEE International Conference on Robotics and Automation (ICRA), pages 1513–1520, 2026. doi:10.1109/icra57385.2026.11695773.

[29] J. Ho and T. Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.

[30] M. Reuss, M. Li, X. Jia, and R. Lioutikov. Goal-conditioned imitation learning using scorebased diffusion policies. In Proceedings of Robotics: Science and Systems, 2023. doi:10. 15607/RSS.2023.XIX.028.

[31] P. Dhariwal and A. Nichol. Diffusion models beat GANs on image synthesis. In Advances in Neural Information Processing Systems, volume 34, pages 8780–8794. Curran Associates, Inc., 2021. URL https://proceedings.neurips.cc/paper/2021/hash/ 49ad23d1ec9fa4bd8d77d02681df5cfa-Abstract.html.

[32] K. M. Lee, S. Ye, Q. Xiao, Z. Wu, Z. Zaidi, D. B. D’Ambrosio, P. R. Sanketi, and M. C. Gombolay. Learning diverse robot striking motions with diffusion models and kinematically constrained gradient guidance. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pages 12017–12024, 2025. doi:10.1109/icra55743.2025.11127310.

[33] W. Xiao, J. T.-H. Wang, C. Gan, R. Hasani, M. Lechner, and D. Rus. SafeDiffuser: Safe planning with diffusion probabilistic models. In International Conference on Learning Representations, volume 2025, pages 100391–100412, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/hash/ f95606d8e870020085990d9650b4f2a1-Abstract-Conference.html.

[34] J. Zhang, L. Zhao, A. Papachristodoulou, and J. Umenberger. Constrained diffusers for safe planning and control. In Advances in Neural Information Processing Systems, volume 38, pages 34965–34998. Curran Associates, Inc., 2025. doi:10.52202/085713-1174.

[35] Y. Zhong, Q. Jiang, J. Yu, and Y. Ma. DexGrasp Anything: Towards universal robotic dexterous grasping with physics awareness. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 22584–22594, 2025.

[36] Z. Weng, H. Lu, D. Kragic, and J. Lundell. DexDiffuser: Generating dexterous grasps with diffusion models. IEEE Robotics and Automation Letters, 9(12):11834–11840, 2024. doi: 10.1109/LRA.2024.3498776.

[37] Y. Jia, Y. Jiang, K. Lv, Y. Ren, and X. Li. Arm-aware guided dexterous grasp generation with arm-agnostic grasp models. IEEE Robotics and Automation Letters, 11(5):5875–5882, 2026. doi:10.1109/lra.2026.3674025.

[38] M. Du and S. Song. DynaGuide: Steering diffusion polices with active dynamic guidance. In Advances in Neural Information Processing Systems, volume 38, pages 44192–44221. Curran Associates, Inc., 2025. doi:10.52202/085713-1473.

[39] Y. Wang, L. Wang, Y. Du, B. Sundaralingam, X. Yang, Y.-W. Chao, C. Perez-D’Arpino,´ D. Fox, and J. Shah. Inference-time policy steering through human interactions. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pages 15626–15633. IEEE, 2025. doi:10.1109/icra55743.2025.11127931.

[40] W. Jung, U. A. Mishra, N. R. Arachchige, Y. Chen, D. Xu, and S. Kousik. Joint model-based model-free diffusion for planning with constraints. In Proceedings of The 9th Conference on Robot Learning, volume 305 of Proceedings of Machine Learning Research, pages 4328–4350. PMLR, 2025. URL https://proceedings.mlr.press/v305/jung25d.html.

[41] H. Gupta, X. Guo, H. Ha, C. Pan, M. Cao, D. Lee, S. Scherer, S. Song, and G. Shi. UMIon-Air: Embodiment-aware guidance for embodiment-agnostic visuomotor policies. In 2026 IEEE International Conference on Robotics and Automation (ICRA), pages 9719–9726, 2026. doi:10.1109/icra57385.2026.11696455.

[42] A. D. Ames, S. Coogan, M. Egerstedt, G. Notomista, K. Sreenath, and P. Tabuada. Control barrier functions: Theory and applications. In 2019 18th European Control Conference (ECC), pages 3420–3431. IEEE, 2019. doi:10.23919/ECC.2019.8796030.

[43] S. Hu, Z. Liu, S. Liu, J. Cen, Z. Meng, S. Wang, X. Li, and X. He. VLSA: Vision-languageaction models with plug-and-play safety constraint layer. arXiv preprint arXiv:2512.11891, 2025. URL https://arxiv.org/abs/2512.11891v2.

[44] L. Brunke, Y. Zhang, R. Romer, J. Naimer, N. Staykov, S. Zhou, and A. P. Schoellig. Semanti-¨ cally safe robot manipulation: From semantic scene understanding to motion safeguards. IEEE Robotics and Automation Letters, 10(5):4810–4817, 2025. doi:10.1109/lra.2025.3553046.

[45] K. P. Wabersich and M. N. Zeilinger. A predictive safety filter for learning-based control of constrained nonlinear dynamical systems. Automatica, 129:109597, 2021. doi:10.1016/j. automatica.2021.109597.

[46] S. Gros, M. Zanon, and A. Bemporad. Safe reinforcement learning via projection on a safe set: How to achieve optimality? IFAC-PapersOnLine, 53(2):8076–8081, 2020. doi:10.1016/j. ifacol.2020.12.2276.

[47] Y. Liu, B. Ou, and H. Zhang. CoFreeVLA: Short-horizon collision-free dual-arm manipulation via vision-language-action model and risk estimation. arXiv preprint arXiv:2601.21712, 2026. URL https://arxiv.org/abs/2601.21712v3.

[48] H. Li, Q. Feng, Z. Zheng, J. Feng, Z. Chen, and A. Knoll. Language-guided object-centric diffusion policy for generalizable and collision-aware manipulation. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pages 12834–12841. IEEE, 2025.

[49] H. Deng, W. Guo, Q. Wang, Z. Wu, and Z. Wang. SafeBimanual: Diffusion-based trajectory optimization for safe bimanual manipulation. In Proceedings of The 9th Conference on Robot Learning, volume 305 of Proceedings of Machine Learning Research, pages 3218–3238. PMLR, 2025. URL https://proceedings.mlr.press/v305/deng25c.html.

[50] A. Dastider, H. Fang, and M. Lin. APEX: Ambidextrous dual-arm robotic manipulation using collision-free generative diffusion models. In 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 9526–9533. IEEE, 2024. doi:10.1109/iros58592. 2024.10802655.

[51] A. Fishman, A. Murali, C. Eppner, B. Peele, B. Boots, and D. Fox. Motion policy networks. In Proceedings of The 6th Conference on Robot Learning, volume 205 of Proceedings of Machine Learning Research, pages 967–977. PMLR, 2023. URL https://proceedings.mlr. press/v205/fishman23a.html.

[52] J. Carvalho, A. T. Le, M. Baierl, D. Koert, and J. Peters. Motion planning diffusion: Learning and planning of robot motions with diffusion models. In 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 1916–1923, 2023. doi: 10.1109/IROS55552.2023.10342382.

[53] K. Lv, M. Yu, Y. Jia, C. Zhang, and X. Li. Kinematics-aware diffusion policy with consistent 3D observation and action space for whole-arm robotic manipulation. IEEE Robotics and Automation Letters, 11(6):7340–7347, 2026. doi:10.1109/lra.2026.3685437.

[54] Q. Lv, H. Li, X. Deng, R. Shao, Y. Li, J. Hao, L. Gao, M. Y. Wang, and L. Nie. Spatialtemporal graph diffusion policy with kinematic modeling for bimanual robotic manipulation. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 17394–17404, 2025. doi:10.1109/cvpr52734.2025.01621.

[55] K. Chen, Z. Bi, G. Zhao, C. Zheng, Y. Li, H. Zhao, and J. Ma. SAMP: Spatial anchor-based motion policy for collision-aware robotic manipulators. arXiv preprint arXiv:2509.11185, 2025.

[56] A. Fishman, A. Walsman, M. Bhardwaj, W. Yuan, B. Sundaralingam, B. Boots, and D. Fox. Avoid everything: Model-free collision avoidance with expert-guided fine-tuning. In Proceedings of The 8th Conference on Robot Learning, volume 270 of Proceedings of Machine Learning Research, pages 1925–1948. PMLR, 2025. URL https://proceedings.mlr.press/ v270/fishman25a.html.

[57] Y. Zhou, C. Barnes, J. Lu, J. Yang, and H. Li. On the continuity of rotation representations in neural networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5745–5753, 2019.

[58] H. Ma, S. Bodmer, A. Carron, M. Zeilinger, and M. Muehlebach. Constraint-aware diffusion guidance for robotics: Real-time obstacle avoidance for autonomous racing. In Proceedings of The 9th Conference on Robot Learning, volume 305 of Proceedings of Machine Learning Research, pages 1756–1776. PMLR, 2025. URL https://proceedings.mlr.press/v305/ ma25a.html.

[59] S. Tao, F. Xiang, A. Shukla, Y. Qin, X. Hinrichsen, X. Yuan, C. Bao, X. Lin, Y. Liu, T.-K. Chan, Y. Gao, X. Li, T. Mu, N. Xiao, A. Gurha, V. N. Rajesh, Y. W. Choi, Y.-R. Chen, Z. Huang, R. Calandra, R. Chen, S. Luo, and H. Su. Demonstrating GPU parallelized robot simulation and rendering for generalizable embodied AI with ManiSkill3. In Proceedings of Robotics: Science and Systems, 2025. doi:10.15607/RSS.2025.XXI.021.

## A Method Details

## A.1 Cartesian Action Conversion and Pose-Twist Mapping

The main text writes the Cartesian denoising residual compactly as $\mathcal { T } _ { \mathrm { p o s e } } ( a _ { t , i } ^ { - 1 } a _ { t - 1 , i } )$ . Here we make this notation explicit for the 9D pose component of the Cartesian action. Let the chunk-start endeffector pose be $T _ { 0 } = ( p _ { 0 } , R _ { 0 } )$ ) and let $\boldsymbol { a } _ { i } = [ \delta p _ { i } , r _ { i } ]$ , where $r _ { i } \in \mathbb { R } ^ { 6 }$ is the continuous 6D rotation representation. We convert $a _ { i }$ into a world-frame pose through

$$
\Phi (a _ {i}; T _ {0}) = (p _ {i}, R _ {i}), \qquad p _ {i} = p _ {0} + R _ {0} \delta p _ {i}, \qquad R _ {i} = R _ {0} \phi_ {\mathrm{6D}} (r _ {i}),\tag{8}
$$

where $\phi _ { \mathrm { 6 D } } ( \cdot )$ maps the 6D rotation representation to $S O ( 3 )$ [57]. The shorthand $a _ { t , i } ^ { - 1 } a _ { t - 1 , i }$ in the main text therefore denotes the relative transform

$$
\Phi (a _ {t, i}; T _ {0}) ^ {- 1} \Phi (a _ {t - 1, i}; T _ {0}) = (\Delta p _ {t, i}, \Delta R _ {t, i}).\tag{9}
$$

The pose-twist operator then returns

$$
\xi_ {t, i} = \mathcal {T} _ {\mathrm{pose}} \left(\Phi (a _ {t, i}; T _ {0}) ^ {- 1} \Phi (a _ {t - 1, i}; T _ {0})\right) = \left[ \begin{array}{c} \Delta p _ {t, i} \\ \operatorname{Log} (\Delta R _ {t, i}) \end{array} \right],\tag{10}
$$

where Log $: S O ( 3 )  \mathbb { R } ^ { 3 }$ is the rotation-vector map.

The joint-space initialization follows (3). In experiments, $q ^ { \mathrm { s t a r t } }$ is the current robot joint configuration at the beginning of the action chunk. Cartesian Gaussian noise is converted relative to $T _ { 0 }$ represented as a pose twist, and locally mapped to joint perturbations with the damped Jacobian pseudo-inverse. During denoising, the local kinematic approximation $\xi _ { t , i } \approx J _ { \mathcal { E } } ( q _ { t , i } ) \Delta q _ { t , i }$ yields (4). The per-joint clipping in Eqs. (3) and (4) is used to avoid large local linearization errors and unstable motions near singularities.

## A.2 Whole-Body SDF Representation

The whole-body safety function $h ( q )$ is computed through cuRobo’s collision geometry and signeddistance interfaces. The robot body is approximated by a set of collision spheres attached to the robot links, and we query the SDF value and gradient of these spheres with respect to known obstacle geometry. Let $d _ { j } ( q )$ denote the signed distance returned for the $j \cdot$ -th sphere at configuration $q ,$ , where the convention is that positive values indicate penetration and negative values indicate clearance. Under this convention, the exact worst signed distance over all robot collision spheres is

$$
d _ {\max} (q) = \max _ {j} d _ {j} (q).\tag{11}
$$

Equivalently, under the paper’s convention that positive $h ( q )$ indicates clearance and negative values indicate penetration, the exact whole-body clearance is $h _ { \mathrm { e x a c t } } ( q ) = - d _ { \mathrm { m a x } } ( q )$

Using the exact max gives the correct worst-sphere value but can introduce abrupt gradient switches when different collision spheres become the maximum. For smoother guidance gradients, we use a normalized smooth maximum over only the top-k largest cuRobo signed distances. Let $\mathcal { K } ( q )$ be the indices of the k largest values in $\{ d _ { j } ( q ) \} _ { j }$ , and let $m ( q ) = \mathrm { m a x } _ { j \in { \mathcal { K } } ( q ) } d _ { j } ( q )$ . We compute

$$
\widetilde {d} _ {\max} (q) = m (q) + \frac {1}{\tau} \log \left(\frac {1}{k} \sum_ {j \in \mathcal {K} (q)} \exp \left(\tau \left[ d _ {j} (q) - m (q) \right]\right)\right),\tag{12}
$$

where $\tau$ is the smoothmax temperature. The CBF safety value used by the QP is

$$
h (q) = - \widetilde {d} _ {\max} (q).\tag{13}
$$

Because (12) is a smooth approximation to the exact max over spheres, it can introduce a small bias in the estimated worst distance. We use it because the smoother gradient is more stable for local guidance than the discontinuous gradient of the exact max. If the opposite SDF sign convention is used, the same idea becomes a top-k smooth minimum over clearances.

In the experiments, obstacles are modeled as known cuboids in both simulation and real-world evaluation. We therefore assume access to obstacle pose and shape during deployment; perception of unknown obstacles is outside the scope of this work. Fig. 8 illustrates the representation: the physical obstacle layout is converted into known cuboids, while the robot body is approximated by link-attached collision spheres for whole-body SDF queries.

![](images/86d94859df3209f97254c6c8cd33a27eb296cdea999f04020747156a290d0568.jpg)

![](images/cb23dd7b1f4f043134b8100cecbb59cc98ff8b5bcbeb1f849d5a81b9bc2ceb1f.jpg)  
Figure 8: Whole-body SDF representation. The left image shows the real scene. The right image shows the corresponding cuRobo scene used for SDF queries, where the robot is represented by collision spheres and the obstacle is represented as a cuboid.

## A.3 CBF-Inspired QP Details

The main text derives the CBF-inspired nonlinear guidance objective and its linearized QP form. Here we provide additional details omitted for space. For each horizon step, (6) can be written as

$$
\min _ {\Delta q} \frac {1}{2} \Delta q ^ {\top} H \Delta q, \qquad \mathrm{s.t.} \quad a ^ {\top} \Delta q \geq b,\tag{14}
$$

where

$$
a = \nabla_ {q} h (q _ {t - 1, i} ^ {\mathrm{diff}}), \qquad b = \gamma_ {t} \left[ d _ {\mathrm{safe}} - h (q _ {t - 1, i} ^ {\mathrm{diff}}) \right], \qquad H = J _ {\mathcal {E}} ^ {\top} W J _ {\mathcal {E}} + \lambda_ {\mathrm{cbf}} I.\tag{15}
$$

The scalar $\gamma _ { t }$ is the scheduled constraint strength at diffusion timestep t. Consistent with the reverse denoising update in the main text, t decreases from $T - 1$ to 0 during inference. Following [58], we use

$$
\gamma_ {t} = \gamma \cdot \sigma \bigg (\beta \cdot \left(c - \frac {t}{T - 1}\right) \bigg), \qquad \sigma (z) = \frac {1}{1 + \exp (- z)},\tag{16}
$$

where $\gamma$ is the base constraint strength. We set the slope $\beta$ to 50 and the transition point c to 0.7 in all experiments. This schedule gives smaller constraint strength in early denoising steps and approaches $\gamma$ in later steps, when samples are closer to the data manifold and less likely to be washed out by subsequent denoising updates. Fig. 9 visualizes this schedule for the default $T = 1 6$ denoising steps and $\gamma = 1 . 0$ . The $\mathrm { Q P }$ uses the nonnegative right-hand side max(b, 0), so configurations already satisfying the safety margin receive zero correction from the constraint.

The closed-form solution follows from the KKT conditions of the single-constraint QP. With the clipped right-hand side $\bar { b } = \operatorname* { m a x } ( b , 0 )$ , the Lagrangian is

$$
\mathcal {L} (\Delta q, \mu) = \frac {1}{2} \Delta q ^ {\top} H \Delta q + \mu (\bar {b} - a ^ {\top} \Delta q), \qquad \mu \geq 0.\tag{17}
$$

Stationarity gives $H \Delta q - \mu a = 0$ , hence $\Delta q = \mu H ^ { - 1 } a . \mathrm { ~ I f ~ } \bar { b } = 0$ , the unconstrained minimizer $\Delta q = 0$ is feasible. Otherwise, complementary slackness makes the constraint active, so $a ^ { \top } \Delta q =$ $\mu a ^ { \top } H ^ { - 1 } a = \bar { b }$ and $\mu = \bar { b } / ( a ^ { \top } H ^ { - 1 } a )$ . The resulting solution is

![](images/791c4f6433e16c51bd701ca5a044d62638512977b8648862fc2dcb56b3400876.jpg)  
Figure 9: Scheduled constraint strength $\gamma _ { t }$ over reverse denoising steps. The guidance is weak in early noisy steps and approaches the base scale $\gamma$ in later steps.

$$
\Delta q ^ {\star} = \left\{ \begin{array}{l l} 0, & b \leq 0, \\ \frac {b}{a ^ {\top} H ^ {- 1} a + \varepsilon}   H ^ {- 1} a, & b > 0, \end{array} \right.\tag{18}
$$

where ε is a small numerical stabilizer. The case $b \leq 0$ corresponds to a locally satisfied safety constraint, for which the unconstrained minimizer $\Delta q = 0$ is feasible. The update is then clipped as in (7). The $\mathrm { Q P }$ is solved over the arm joints only; the gripper command is not modified by the collision-avoidance correction. The task-preservation metric uses the end-effector geometric Jacobian with separate position and rotation weights. Because the method uses a local linearization and clipped updates, the guidance should be interpreted as an efficient feasible correction direction rather than a formal global safety certificate.

## B Implementation Details

## B.1 Diffusion Policy Architecture

We use the same Cartesian diffusion-policy architecture across all embodiments. In both simulation and real-world experiments, RGB observations are encoded with the pretrained CLIP ViT-B/16 backbone (vit base patch16 clip 224.openai) loaded through timm and finetuned during policy training. The resulting observation feature is used as a global condition for a conditional 1D U-Net denoiser over the action chunk. The denoiser takes a noisy Cartesian action sequence and a diffusion timestep as input, and predicts the noise $\epsilon _ { \theta }$ used by the reverse diffusion step. At deployment, we use a DDIM scheduler with 16 denoising steps. The action sequence has horizon H and action dimension 10, consisting of relative end-effector translation, a continuous 6D rotation representation, and one gripper command. During deployment, EMBODISTEER keeps this trained denoiser unchanged and only changes the sampling variable from Cartesian actions to target-robot joint trajectories.

## B.2 Hyperparameters

Table 3 lists the hyperparameters used for training and deployment.

## B.3 Runtime Breakdown

Fig. 10 reports the measured runtime breakdown of one guided joint-space inference call on an RTX 4070 Ti SUPER, including all 16 reverse-diffusion steps. The full EMBODISTEER inference time is 103 ms, corresponding to approximately 9.7 Hz deployment. The base Cartesian policy already requires observation encoding and U-Net denoising, which account for 46.8 ms. The remaining measured time is introduced by joint-space lifting, whole-body SDF queries, CBF-QP guidance, and associated runtime overhead.

Table 3: Training and deployment hyperparameters.

<table><tr><td>Quantity</td><td>Value</td><td>Notes</td></tr><tr><td>Observation horizon</td><td>2</td><td>Diffusion policy input setting.</td></tr><tr><td>Action horizon H</td><td>16</td><td>Predicted action chunk length.</td></tr><tr><td>Execution horizon</td><td>16 in simulation, 6 in real world.</td><td>Number of actions executed before replanning.</td></tr><tr><td>Visual backbone</td><td>CLIP ViT-B/16</td><td>vit_base_patch16_clip_224.openai loaded through timm, pretrained and finetuned.</td></tr><tr><td>U-Net channels</td><td>[256, 512, 1024]</td><td>Conditional 1D U-Net downsampling dimensions.</td></tr><tr><td>Diffusion step embedding</td><td>128</td><td>Timestep embedding dimension.</td></tr><tr><td>Diffusion training timesteps</td><td>50</td><td>Squared-cosine beta schedule and epsilon prediction.</td></tr><tr><td>Inference denoising steps</td><td>16</td><td>DDIM reverse diffusion steps at deployment.</td></tr><tr><td>Control frequency</td><td>10</td><td>Simulation and real-world control rate.</td></tr><tr><td>Training epochs</td><td>120</td><td>Used for both simulation and real-world policies.</td></tr><tr><td>Batch size</td><td>128 in simulation, 64 in real world.</td><td>Training batch size for simulation and real-world checkpoints.</td></tr><tr><td>Optimizer</td><td>AdamW</td><td>Learning rate  $3 \times 10^{-4}$ , betas (0.95, 0.999), weight decay  $10^{-6}$ .</td></tr><tr><td>Learning-rate schedule</td><td>Cosine</td><td>2000 warmup steps.</td></tr><tr><td>EMA</td><td>0.9999 max decay</td><td>EMA enabled during training.</td></tr><tr><td>Image augmentation</td><td>Crop + color jitter</td><td>Random crop ratio 0.95; brightness 0.3, contrast 0.4, saturation 0.5, hue 0.08.</td></tr><tr><td>Noise initialization scale α</td><td>0.1</td><td>Scale for Jacobian-projected noise.</td></tr><tr><td>Jacobian damping  $\lambda_{\text{pinv}}$ </td><td>0.001</td><td>Used in  $J_{\mathcal{E}}^{+}$ .</td></tr><tr><td>Joint update clip  $\Delta q_{\text{max}}$ </td><td>0.5</td><td>Used in joint denoising update.</td></tr><tr><td>Safety margin  $d_{\text{safe}}$ </td><td>0.05~0.10</td><td>Task-dependent; set to 0.05, 0.07, or 0.10 by obstacle layout.</td></tr><tr><td>SDF aggregation top-k</td><td>4</td><td>Number of critical collision spheres.</td></tr><tr><td>Smoothmax temperature τ</td><td>20.0</td><td>Temperature in (12).</td></tr><tr><td>Base constraint strength γ</td><td>1.0</td><td>Base multiplier in (16).</td></tr><tr><td>QP regularization  $\lambda_{\text{cbf}}$ </td><td>0.01</td><td>Joint-space regularization.</td></tr><tr><td>Task position weight</td><td>1.0</td><td>Position rows in  $J_{\mathcal{E}}^{\top} W J_{\mathcal{E}}$ .</td></tr><tr><td>Task rotation weight</td><td>0.1</td><td>Rotation rows in  $J_{\mathcal{E}}^{\top} W J_{\mathcal{E}}$ .</td></tr><tr><td>CBF update clip  $\Delta q_{\text{max}}^{\text{cbf}}$ </td><td>0.1</td><td>Clipping applied to the guidance update.</td></tr></table>

## B.4 Baseline Implementation Details

Table 4 summarizes how each baseline is implemented. All baselines use the same trained Cartesian policy checkpoint for a given task.

## C Simulation Results

## C.1 Task Definitions and Evaluation Protocol

The simulation benchmark contains three manipulation tasks: PLACETOAST, TURNONFAUCET, and MAKECOFFEE. Cartesian policies are trained from obstacle-free floating-gripper demonstrations generated by task-specific motion planning, and are evaluated zero-shot on robot embodiments with different kinematics and arm geometries. In obstacle-present settings, obstacles are introduced only at test time. Success rate (TSR), maximum reward per episode (RWD), and collision occurrence rate (COR) are reported as in the main text.

<table><tr><td>Component</td><td>Included operations</td><td>ms</td><td>%</td></tr><tr><td colspan="4">Base Cartesian policy components</td></tr><tr><td>Diffusion U-Net</td><td>U-Net denoising, 16 steps</td><td>41.2</td><td>39.9</td></tr><tr><td>Observation Encoder</td><td>Image/state observation encoding</td><td>5.6</td><td>5.4</td></tr><tr><td>Base policy subtotal</td><td>Measured base-policy components</td><td>46.8</td><td>45.4</td></tr><tr><td colspan="4">Additional EMBODISTEER components</td></tr><tr><td>Kinematic Mapping</td><td>FK, Jacobian, DLS, twist</td><td>13.5</td><td>13.1</td></tr><tr><td>CBF Guidance</td><td>CBF autograd, QP, linearization setup</td><td>11.5</td><td>11.2</td></tr><tr><td>Collision SDF Query</td><td>Sphere FK, SDF query, world setup</td><td>2.8</td><td>2.7</td></tr><tr><td>Pose / Norm. / Scheduler</td><td>Pose conversion, normalizer, scheduler step</td><td>11.1</td><td>10.8</td></tr><tr><td>I/O and Action Formatting</td><td>Obs. preprocessing, joint tensor, action conversion</td><td>1.0</td><td>1.0</td></tr><tr><td>Other Runtime Overhead</td><td>Uninstrumented loop/PyTorch overhead</td><td>16.3</td><td>15.8</td></tr><tr><td>Added subtotal</td><td>Joint-space lifting and guidance overhead</td><td>56.2</td><td>54.6</td></tr><tr><td>Full total</td><td>Guided joint-space inference</td><td>103.0</td><td>100.0</td></tr></table>

![](images/9b742a6c10a747ac1ce1ace093bc09e73cbbe6c8a7d8c30ef18c677bbfa56dbc.jpg)

Figure 10: Runtime breakdown for one guided joint-space inference call on an RTX 4070 Ti SUPER.  
Table 4: Baseline implementation details.

<table><tr><td>Method</td><td>Implementation</td></tr><tr><td>EE</td><td>Runs the Cartesian diffusion policy in its original action space and executes the predicted end-effector actions without collision-aware guidance.</td></tr><tr><td>Joint</td><td>Uses the same joint-space denoising procedure as EMBODISTEER, including FK queries to the frozen Cartesian denoiser and damped Jacobian updates, but does not apply collision guidance.</td></tr><tr><td>EE w/ Sampling</td><td>Keeps Cartesian denoising unchanged, samples multiple candidate action chunks, realizes them through the target robot, and selects the candidate with the largest clearance. Number of candidates: 16.</td></tr><tr><td>EE w/ CBF</td><td>Keeps Cartesian denoising unchanged, solves IK for the generated action chunk, and applies a one-shot CBF-QP correction after generation.</td></tr><tr><td>Joint w/ CG</td><td>Adds collision-cost gradient guidance during joint-space denoising. Unlike EM-BODISTEER, the update is not explicitly regularized by the task-space QP objective, so whole-body collision gradients can perturb the end-effector trajectory.</td></tr></table>

As illustrated in Fig. 3, the benchmark uses the same three tasks and obstacle layouts across 9 robot embodiments. We therefore avoid duplicating the task and robot lists here; the appendix focuses on the complete per-robot quantitative results and guidance ablations.

## C.2 Full Quantitative Results

Table 5 reports the per-robot simulation results behind Table 1. For EMBODISTEER, the listed guidance strengths match the main-paper setting for each task.

Table 5: Full per-robot simulation results. Obs. indicates whether obstacles are present during evaluation; TSR is success-once rate, RWD is maximum reward per episode, and COR is collision occurrence rate. COR is reported only for obstacle-present settings.

<table><tr><td>Obs.</td><td>Method</td><td>Robot</td><td>TSR↑</td><td>RWD↑</td><td>COR↓</td><td>Robot</td><td>TSR↑</td><td>RWD↑</td><td>COR↓</td><td>Robot</td><td>TSR↑</td><td>RWD↑</td><td>COR↓</td></tr><tr><td colspan="14">PLACETOAST</td></tr><tr><td rowspan="3"></td><td rowspan="3">EE</td><td>UR5</td><td>100.0%</td><td>1.000</td><td>-</td><td>Panda</td><td>100.0%</td><td>1.000</td><td>-</td><td>xArm6</td><td>100.0%</td><td>1.000</td><td>-</td></tr><tr><td>xArm7</td><td>98.0%</td><td>0.989</td><td>-</td><td>iiwa7</td><td>98.0%</td><td>0.989</td><td>-</td><td>Gen3-6D</td><td>81.0%</td><td>0.879</td><td>-</td></tr><tr><td>Gen3-7D</td><td>98.0%</td><td>0.989</td><td>-</td><td>Rizon4</td><td>99.0%</td><td>0.991</td><td>-</td><td>Sawyer</td><td>97.0%</td><td>0.984</td><td>-</td></tr><tr><td>w/o Obs.</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>Avg.</td><td>96.8%</td><td>0.980</td><td>-</td></tr></table>

Continued on next page

<table><tr><td>Obs.</td><td>Method</td><td>Robot</td><td>TSR↑</td><td>RWD↑</td><td>COR↓</td><td>Robot</td><td>TSR↑</td><td>RWD↑</td><td>COR↓</td><td>Robot</td><td>TSR↑</td><td>RWD↑</td><td>COR↓</td></tr><tr><td rowspan="3"></td><td rowspan="3">Joint</td><td>UR5</td><td>100.0%</td><td>1.000</td><td>-</td><td>Panda</td><td>100.0%</td><td>1.000</td><td>-</td><td>xArm6</td><td>98.0%</td><td>0.988</td><td>-</td></tr><tr><td>xArm7</td><td>99.0%</td><td>0.995</td><td>-</td><td>iiwa7</td><td>98.0%</td><td>0.989</td><td>-</td><td>Gen3-6D</td><td>73.0%</td><td>0.827</td><td>-</td></tr><tr><td>Gen3-7D</td><td>85.0%</td><td>0.907</td><td>-</td><td>Rizon4</td><td>98.0%</td><td>0.983</td><td>-</td><td>Sawyer</td><td>99.0%</td><td>0.995</td><td>-</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>Avg.</td><td>94.4%</td><td>0.965</td><td>-</td></tr><tr><td rowspan="3"></td><td rowspan="3">EE</td><td>UR5</td><td>39.0%</td><td>0.560</td><td>50.0%</td><td>Panda</td><td>46.0%</td><td>0.585</td><td>54.0%</td><td>xArm6</td><td>45.0%</td><td>0.676</td><td>52.0%</td></tr><tr><td>xArm7</td><td>48.0%</td><td>0.695</td><td>56.0%</td><td>iiwa7</td><td>47.0%</td><td>0.689</td><td>49.0%</td><td>Gen3-6D</td><td>36.0%</td><td>0.611</td><td>51.0%</td></tr><tr><td>Gen3-7D</td><td>43.0%</td><td>0.665</td><td>51.0%</td><td>Rizon4</td><td>42.0%</td><td>0.552</td><td>51.0%</td><td>Sawyer</td><td>45.0%</td><td>0.676</td><td>59.0%</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>Avg.</td><td>43.4%</td><td>0.634</td><td>52.6%</td></tr><tr><td rowspan="3"></td><td rowspan="3">EE w/ Samp.</td><td>UR5</td><td>48.0%</td><td>0.626</td><td>62.0%</td><td>Panda</td><td>54.0%</td><td>0.651</td><td>46.0%</td><td>xArm6</td><td>55.0%</td><td>0.736</td><td>49.0%</td></tr><tr><td>xArm7</td><td>52.0%</td><td>0.717</td><td>50.0%</td><td>iiwa7</td><td>44.0%</td><td>0.663</td><td>40.0%</td><td>Gen3-6D</td><td>36.0%</td><td>0.613</td><td>47.0%</td></tr><tr><td>Gen3-7D</td><td>45.0%</td><td>0.677</td><td>53.0%</td><td>Rizon4</td><td>48.0%</td><td>0.601</td><td>62.0%</td><td>Sawyer</td><td>42.0%</td><td>0.661</td><td>53.0%</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>Avg.</td><td>47.1%</td><td>0.661</td><td>51.3%</td></tr><tr><td rowspan="3"></td><td rowspan="3">EE w/ CBF</td><td>UR5</td><td>76.0%</td><td>0.797</td><td>11.0%</td><td>Panda</td><td>64.0%</td><td>0.692</td><td>17.0%</td><td>xArm6</td><td>75.0%</td><td>0.849</td><td>8.0%</td></tr><tr><td>xArm7</td><td>66.0%</td><td>0.796</td><td>14.0%</td><td>iiwa7</td><td>62.0%</td><td>0.768</td><td>14.0%</td><td>Gen3-6D</td><td>62.0%</td><td>0.766</td><td>5.0%</td></tr><tr><td>Gen3-7D</td><td>68.0%</td><td>0.809</td><td>13.0%</td><td>Rizon4</td><td>50.0%</td><td>0.580</td><td>37.0%</td><td>Sawyer</td><td>69.0%</td><td>0.816</td><td>19.0%</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>Avg.</td><td>65.8%</td><td>0.764</td><td>15.3%</td></tr><tr><td rowspan="3"></td><td rowspan="3">Joint w/ CG</td><td>UR5</td><td>35.0%</td><td>0.386</td><td>0.0%</td><td>Panda</td><td>76.0%</td><td>0.789</td><td>0.0%</td><td>xArm6</td><td>62.0%</td><td>0.752</td><td>0.0%</td></tr><tr><td>xArm7</td><td>67.0%</td><td>0.786</td><td>0.0%</td><td>iiwa7</td><td>44.0%</td><td>0.628</td><td>1.0%</td><td>Gen3-6D</td><td>54.0%</td><td>0.699</td><td>0.0%</td></tr><tr><td>Gen3-7D</td><td>57.0%</td><td>0.721</td><td>0.0%</td><td>Rizon4</td><td>62.0%</td><td>0.665</td><td>0.0%</td><td>Sawyer</td><td>62.0%</td><td>0.753</td><td>1.0%</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>Avg.</td><td>57.7%</td><td>0.687</td><td>0.2%</td></tr><tr><td rowspan="3"></td><td rowspan="3">Ours</td><td>UR5</td><td>78.0%</td><td>0.807</td><td>0.0%</td><td>Panda</td><td>82.0%</td><td>0.842</td><td>3.0%</td><td>xArm6</td><td>77.0%</td><td>0.859</td><td>0.0%</td></tr><tr><td>xArm7</td><td>68.0%</td><td>0.805</td><td>0.0%</td><td>iiwa7</td><td>84.0%</td><td>0.905</td><td>9.0%</td><td>Gen3-6D</td><td>54.0%</td><td>0.712</td><td>0.0%</td></tr><tr><td>Gen3-7D</td><td>76.0%</td><td>0.853</td><td>1.0%</td><td>Rizon4</td><td>82.0%</td><td>0.849</td><td>16.0%</td><td>Sawyer</td><td>72.0%</td><td>0.831</td><td>12.0%</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>Avg.</td><td>74.8%</td><td>0.829</td><td>4.6%</td></tr><tr><td colspan="14">TURNONFAUCET</td></tr><tr><td rowspan="3">w/o Obs.</td><td rowspan="3">EE</td><td>Panda</td><td>81.0%</td><td>0.943</td><td>-</td><td>xArm6</td><td>99.0%</td><td>0.998</td><td>-</td><td>xArm7</td><td>78.0%</td><td>0.937</td><td>-</td></tr><tr><td>UR5</td><td>90.0%</td><td>0.918</td><td>-</td><td>iiwa7</td><td>87.0%</td><td>0.964</td><td>-</td><td>Gen3-6D</td><td>96.0%</td><td>0.988</td><td>-</td></tr><tr><td>Gen3-7D</td><td>53.0%</td><td>0.880</td><td>-</td><td>Rizon4</td><td>73.0%</td><td>0.822</td><td>-</td><td>Sawyer</td><td>95.0%</td><td>0.987</td><td>-</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>Avg.</td><td>83.6%</td><td>0.937</td><td>-</td></tr><tr><td rowspan="3"></td><td rowspan="3">Joint</td><td>UR5</td><td>88.0%</td><td>0.912</td><td>-</td><td>Panda</td><td>90.0%</td><td>0.974</td><td>-</td><td>xArm6</td><td>100.0%</td><td>1.000</td><td>-</td></tr><tr><td>xArm7</td><td>90.0%</td><td>0.971</td><td>-</td><td>iiwa7</td><td>91.0%</td><td>0.975</td><td>-</td><td>Gen3-6D</td><td>95.0%</td><td>0.988</td><td>-</td></tr><tr><td>Gen3-7D</td><td>63.0%</td><td>0.909</td><td>-</td><td>Rizon4</td><td>87.0%</td><td>0.952</td><td>-</td><td>Sawyer</td><td>87.0%</td><td>0.936</td><td>-</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>Avg.</td><td>87.9%</td><td>0.957</td><td>-</td></tr><tr><td rowspan="3"></td><td rowspan="3">EE</td><td>UR5</td><td>11.0%</td><td>0.417</td><td>86.0%</td><td>Panda</td><td>23.0%</td><td>0.471</td><td>94.0%</td><td>xArm6</td><td>26.0%</td><td>0.791</td><td>89.0%</td></tr><tr><td>xArm7</td><td>41.0%</td><td>0.823</td><td>61.0%</td><td>iiwa7</td><td>66.0%</td><td>0.915</td><td>28.0%</td><td>Gen3-6D</td><td>73.0%</td><td>0.931</td><td>41.0%</td></tr><tr><td>Gen3-7D</td><td>42.0%</td><td>0.860</td><td>40.0%</td><td>Rizon4</td><td>12.0%</td><td>0.657</td><td>88.0%</td><td>Sawyer</td><td>80.0%</td><td>0.904</td><td>29.0%</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>Avg.</td><td>41.6%</td><td>0.752</td><td>61.8%</td></tr><tr><td rowspan="3"></td><td rowspan="3">EE w/ Samp.</td><td>UR5</td><td>6.0%</td><td>0.453</td><td>86.0%</td><td>Panda</td><td>35.0%</td><td>0.554</td><td>87.0%</td><td>xArm6</td><td>31.0%</td><td>0.811</td><td>84.0%</td></tr><tr><td>xArm7</td><td>32.0%</td><td>0.789</td><td>47.0%</td><td>iiwa7</td><td>69.0%</td><td>0.911</td><td>18.0%</td><td>Gen3-6D</td><td>74.0%</td><td>0.935</td><td>28.0%</td></tr><tr><td>Gen3-7D</td><td>50.0%</td><td>0.874</td><td>21.0%</td><td>Rizon4</td><td>13.0%</td><td>0.665</td><td>90.0%</td><td>Sawyer</td><td>85.0%</td><td>0.911</td><td>18.0%</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>Avg.</td><td>43.9%</td><td>0.767</td><td>53.2%</td></tr><tr><td rowspan="3"></td><td rowspan="3">EE w/ CBF</td><td>UR5</td><td>33.0%</td><td>0.506</td><td>29.0%</td><td>Panda</td><td>48.0%</td><td>0.661</td><td>46.0%</td><td>xArm6</td><td>60.0%</td><td>0.857</td><td>36.0%</td></tr><tr><td>xArm7</td><td>55.0%</td><td>0.833</td><td>26.0%</td><td>iiwa7</td><td>78.0%</td><td>0.945</td><td>9.0%</td><td>Gen3-6D</td><td>87.0%</td><td>0.942</td><td>12.0%</td></tr><tr><td>Gen3-7D</td><td>48.0%</td><td>0.851</td><td>1.0%</td><td>Rizon4</td><td>8.0%</td><td>0.632</td><td>97.0%</td><td>Sawyer</td><td>90.0%</td><td>0.949</td><td>13.0%</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>Avg.</td><td>56.3%</td><td>0.797</td><td>29.9%</td></tr><tr><td rowspan="3"></td><td rowspan="3">Joint w/ CG</td><td>UR5</td><td>8.0%</td><td>0.448</td><td>88.0%</td><td>Panda</td><td>7.0%</td><td>0.339</td><td>82.0%</td><td>xArm6</td><td>6.0%</td><td>0.432</td><td>75.0%</td></tr><tr><td>xArm7</td><td>0.0%</td><td>0.239</td><td>65.0%</td><td>iiwa7</td><td>12.0%</td><td>0.501</td><td>42.0%</td><td>Gen3-6D</td><td>33.0%</td><td>0.667</td><td>30.0%</td></tr><tr><td>Gen3-7D</td><td>11.0%</td><td>0.481</td><td>23.0%</td><td>Rizon4</td><td>2.0%</td><td>0.173</td><td>70.0%</td><td>Sawyer</td><td>14.0%</td><td>0.350</td><td>61.0%</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>Avg.</td><td>10.3%</td><td>0.403</td><td>59.6%</td></tr><tr><td rowspan="3"></td><td rowspan="3">Ours</td><td>UR5</td><td>32.0%</td><td>0.502</td><td>23.0%</td><td>Panda</td><td>57.0%</td><td>0.802</td><td>42.0%</td><td>xArm6</td><td>75.0%</td><td>0.900</td><td>30.0%</td></tr><tr><td>xArm7</td><td>52.0%</td><td>0.842</td><td>28.0%</td><td>iiwa7</td><td>76.0%</td><td>0.926</td><td>17.0%</td><td>Gen3-6D</td><td>90.0%</td><td>0.971</td><td>7.0%</td></tr><tr><td>Gen3-7D</td><td>61.0%</td><td>0.897</td><td>2.0%</td><td>Rizon4</td><td>15.0%</td><td>0.744</td><td>88.0%</td><td>Sawyer</td><td>87.0%</td><td>0.916</td><td>7.0%</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>Avg.</td><td>60.6%</td><td>0.833</td><td>27.1%</td></tr></table>

Continued on next page

<table><tr><td>Obs.</td><td>Method</td><td>Robot</td><td>TSR↑</td><td>RWD↑</td><td>COR↓</td><td>Robot</td><td>TSR↑</td><td>RWD↑</td><td>COR↓</td><td>Robot</td><td>TSR↑</td><td>RWD↑</td><td>COR↓</td></tr><tr><td rowspan="8">w/o Obs.</td><td rowspan="3">EE</td><td>UR5</td><td>94.0%</td><td>0.952</td><td>-</td><td>Panda</td><td>94.0%</td><td>0.951</td><td>-</td><td>xArm6</td><td>87.0%</td><td>0.917</td><td>-</td></tr><tr><td>xArm7</td><td>90.0%</td><td>0.943</td><td>-</td><td>iiwa7</td><td>85.0%</td><td>0.913</td><td>-</td><td>Gen3-6D</td><td>93.0%</td><td>0.960</td><td>-</td></tr><tr><td>Gen3-7D</td><td>94.0%</td><td>0.965</td><td>-</td><td>Rizon4</td><td>88.0%</td><td>0.904</td><td>-</td><td>Sawyer</td><td>82.0%</td><td>0.881</td><td>-</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>Avg.</td><td>89.7%</td><td>0.932</td><td>-</td></tr><tr><td rowspan="3">Joint</td><td>UR5</td><td>94.0%</td><td>0.951</td><td>-</td><td>Panda</td><td>93.0%</td><td>0.943</td><td>-</td><td>xArm6</td><td>91.0%</td><td>0.937</td><td>-</td></tr><tr><td>xArm7</td><td>90.0%</td><td>0.936</td><td>-</td><td>iiwa7</td><td>92.0%</td><td>0.959</td><td>-</td><td>Gen3-6D</td><td>95.0%</td><td>0.975</td><td>-</td></tr><tr><td>Gen3-7D</td><td>88.0%</td><td>0.927</td><td>-</td><td>Rizon4</td><td>80.0%</td><td>0.833</td><td>-</td><td>Sawyer</td><td>84.0%</td><td>0.900</td><td>-</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>Avg.</td><td>89.7%</td><td>0.929</td><td>-</td></tr><tr><td rowspan="20">w/Obs.</td><td rowspan="3">EE</td><td>UR5</td><td>8.0%</td><td>0.277</td><td>85.0%</td><td>Panda</td><td>24.0%</td><td>0.408</td><td>60.0%</td><td>xArm6</td><td>27.0%</td><td>0.536</td><td>70.0%</td></tr><tr><td>xArm7</td><td>28.0%</td><td>0.540</td><td>69.0%</td><td>iiwa7</td><td>26.0%</td><td>0.548</td><td>59.0%</td><td>Gen3-6D</td><td>25.0%</td><td>0.497</td><td>45.0%</td></tr><tr><td>Gen3-7D</td><td>24.0%</td><td>0.513</td><td>47.0%</td><td>Rizon4</td><td>24.0%</td><td>0.385</td><td>44.0%</td><td>Sawyer</td><td>14.0%</td><td>0.401</td><td>46.0%</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>Avg.</td><td>22.2%</td><td>0.456</td><td>58.3%</td></tr><tr><td rowspan="3">EE w/Samp.</td><td>UR5</td><td>8.0%</td><td>0.243</td><td>87.0%</td><td>Panda</td><td>24.0%</td><td>0.393</td><td>63.0%</td><td>xArm6</td><td>24.0%</td><td>0.539</td><td>71.0%</td></tr><tr><td>xArm7</td><td>30.0%</td><td>0.577</td><td>64.0%</td><td>iiwa7</td><td>39.0%</td><td>0.635</td><td>56.0%</td><td>Gen3-6D</td><td>20.0%</td><td>0.490</td><td>56.0%</td></tr><tr><td>Gen3-7D</td><td>24.0%</td><td>0.523</td><td>52.0%</td><td>Rizon4</td><td>23.0%</td><td>0.384</td><td>56.0%</td><td>Sawyer</td><td>25.0%</td><td>0.516</td><td>35.0%</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>Avg.</td><td>24.1%</td><td>0.478</td><td>60.0%</td></tr><tr><td rowspan="3">EE w/CBF</td><td>UR5</td><td>19.0%</td><td>0.314</td><td>8.0%</td><td>Panda</td><td>62.0%</td><td>0.683</td><td>11.0%</td><td>xArm6</td><td>58.0%</td><td>0.699</td><td>14.0%</td></tr><tr><td>xArm7</td><td>48.0%</td><td>0.615</td><td>2.0%</td><td>iiwa7</td><td>66.0%</td><td>0.762</td><td>4.0%</td><td>Gen3-6D</td><td>67.0%</td><td>0.769</td><td>6.0%</td></tr><tr><td>Gen3-7D</td><td>43.0%</td><td>0.574</td><td>4.0%</td><td>Rizon4</td><td>39.0%</td><td>0.494</td><td>21.0%</td><td>Sawyer</td><td>38.0%</td><td>0.543</td><td>7.0%</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>Avg.</td><td>48.9%</td><td>0.606</td><td>8.6%</td></tr><tr><td rowspan="3">Joint w/ CG</td><td>UR5</td><td>13.0%</td><td>0.287</td><td>79.0%</td><td>Panda</td><td>18.0%</td><td>0.333</td><td>51.0%</td><td>xArm6</td><td>29.0%</td><td>0.528</td><td>56.0%</td></tr><tr><td>xArm7</td><td>31.0%</td><td>0.519</td><td>36.0%</td><td>iiwa7</td><td>17.0%</td><td>0.385</td><td>28.0%</td><td>Gen3-6D</td><td>31.0%</td><td>0.541</td><td>39.0%</td></tr><tr><td>Gen3-7D</td><td>33.0%</td><td>0.544</td><td>36.0%</td><td>Rizon4</td><td>5.0%</td><td>0.183</td><td>13.0%</td><td>Sawyer</td><td>25.0%</td><td>0.493</td><td>27.0%</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>Avg.</td><td>22.4%</td><td>0.424</td><td>40.6%</td></tr><tr><td rowspan="3">Ours</td><td>UR5</td><td>28.0%</td><td>0.387</td><td>3.0%</td><td>Panda</td><td>75.0%</td><td>0.798</td><td>4.0%</td><td>xArm6</td><td>68.0%</td><td>0.757</td><td>3.0%</td></tr><tr><td>xArm7</td><td>64.0%</td><td>0.756</td><td>2.0%</td><td>iiwa7</td><td>73.0%</td><td>0.815</td><td>0.0%</td><td>Gen3-6D</td><td>69.0%</td><td>0.795</td><td>2.0%</td></tr><tr><td>Gen3-7D</td><td>56.0%</td><td>0.662</td><td>1.0%</td><td>Rizon4</td><td>38.0%</td><td>0.481</td><td>9.0%</td><td>Sawyer</td><td>44.0%</td><td>0.574</td><td>1.0%</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>Avg.</td><td>57.2%</td><td>0.670</td><td>2.8%</td></tr></table>

## C.3 Repeated Evaluations

Tab. 6 reports EMBODISTEER evaluations with three different random seeds for each task, using the same trained task policy. Each row averages over the nine robot embodiments, using the task success, maximum reward per episode, and collision occurrence metrics of the main benchmark.

Table 6: Repeated EMBODISTEER evaluations with three different random seeds for each task.

<table><tr><td>Task</td><td>Evaluation</td><td>TSR (%) ↑</td><td>RWD ↑</td><td>COR (%) ↓</td></tr><tr><td rowspan="3">MAKECOFFEE</td><td>Run 1</td><td>57.2</td><td>.670</td><td>2.8</td></tr><tr><td>Run 2</td><td>57.9</td><td>.677</td><td>3.4</td></tr><tr><td>Run 3</td><td>53.8</td><td>.649</td><td>4.2</td></tr><tr><td rowspan="3">TURNONFAUCET</td><td>Run 1</td><td>60.6</td><td>.833</td><td>27.1</td></tr><tr><td>Run 2</td><td>62.0</td><td>.843</td><td>23.9</td></tr><tr><td>Run 3</td><td>57.8</td><td>.809</td><td>29.3</td></tr><tr><td rowspan="3">PLACETOAST</td><td>Run 1</td><td>74.8</td><td>.829</td><td>4.6</td></tr><tr><td>Run 2</td><td>73.1</td><td>.818</td><td>5.4</td></tr><tr><td>Run 3</td><td>72.6</td><td>.810</td><td>4.7</td></tr></table>

The aggregate mean and standard deviation of these three-seed evaluations are reported for EM-BODISTEER in Tab. 2.

## C.4 Additional Ablations and Sensitivity Analyses

We first analyze the main design choices and hyperparameters of EMBODISTEER. The constraint strength γ controls the strength of the CBF-QP constraint relative to the diffusion update: too small a value may leave collisions unresolved, while too large a value can over-steer the sample and hurt task execution. Following the linearized QP derivation, we use $\gamma = 1 . 0$ as the default scale and sweep around this value. Fig. 11 therefore visualizes a guidance-strength sensitivity analysis across tasks.

![](images/b5301be1e8f82125f796c1f14b655ad9df4cc37f5f3c67b0ab206dac4072e879.jpg)  
Figure 11: CBF-QP guidance-strength sensitivity analysis. The default $\gamma = 1 . 0$ follows the linearized QP derivation and provides a robust task–collision tradeoff across the three tasks.

The regularization weight $\lambda _ { \mathrm { c b f } }$ controls how much the QP penalizes joint-space deviation from the denoised sample. This term is included to prevent collision avoidance from being absorbed by unnecessarily large joint motions. Table 7 shows that performance is stable over a broad range, with the default $\lambda _ { \mathrm { c b f } } = 0 . 0 1$ giving a balanced setting.

Table 7: CBF-QP regularization sensitivity analysis averaged over the 9 robot embodiments used in the paper. The $\lambda _ { \mathrm { c b f } } \overset { = } 0 . 0 1$ row corresponds to the default setting.

<table><tr><td>Task</td><td> $\lambda_{cbf}$ </td><td>TSR ↑</td><td>RWD ↑</td><td>COR ↓</td></tr><tr><td rowspan="8">PLACETOAST</td><td>0.000</td><td>70.8%</td><td>0.800</td><td>5.9%</td></tr><tr><td>0.001</td><td>70.0%</td><td>0.797</td><td>6.1%</td></tr><tr><td>0.005</td><td>72.4%</td><td>0.809</td><td>5.0%</td></tr><tr><td>0.010</td><td>71.8%</td><td>0.812</td><td>6.9%</td></tr><tr><td>0.050</td><td>70.9%</td><td>0.797</td><td>5.3%</td></tr><tr><td>0.100</td><td>70.1%</td><td>0.797</td><td>5.7%</td></tr><tr><td>0.500</td><td>69.9%</td><td>0.797</td><td>5.9%</td></tr><tr><td>1.000</td><td>70.9%</td><td>0.803</td><td>7.1%</td></tr><tr><td rowspan="8">TURNONFAUCET</td><td>0.000</td><td>57.2%</td><td>0.829</td><td>26.3%</td></tr><tr><td>0.001</td><td>58.9%</td><td>0.842</td><td>25.7%</td></tr><tr><td>0.005</td><td>58.6%</td><td>0.833</td><td>24.9%</td></tr><tr><td>0.010</td><td>60.6%</td><td>0.833</td><td>27.1%</td></tr><tr><td>0.050</td><td>57.0%</td><td>0.817</td><td>26.6%</td></tr><tr><td>0.100</td><td>59.6%</td><td>0.831</td><td>25.9%</td></tr><tr><td>0.500</td><td>59.7%</td><td>0.838</td><td>25.7%</td></tr><tr><td>1.000</td><td>60.2%</td><td>0.835</td><td>24.6%</td></tr><tr><td rowspan="8">MAKECOFFEE</td><td>0.000</td><td>46.3%</td><td>0.603</td><td>3.6%</td></tr><tr><td>0.001</td><td>42.4%</td><td>0.574</td><td>3.2%</td></tr><tr><td>0.005</td><td>45.3%</td><td>0.591</td><td>2.6%</td></tr><tr><td>0.010</td><td>57.2%</td><td>0.670</td><td>2.8%</td></tr><tr><td>0.050</td><td>47.2%</td><td>0.606</td><td>2.0%</td></tr><tr><td>0.100</td><td>42.7%</td><td>0.570</td><td>3.4%</td></tr><tr><td>0.500</td><td>44.7%</td><td>0.590</td><td>3.6%</td></tr><tr><td>1.000</td><td>46.4%</td><td>0.603</td><td>3.6%</td></tr></table>

We also ablate the guidance schedule by comparing the scheduled strength in (16) with a constant scale $\gamma _ { t } = \gamma = 1 . 0$ throughout denoising. The schedule is intended to avoid strong corrections in early noisy denoising steps, where samples are still far from the policy distribution. Table 8 shows that scheduling helps preserve task performance while maintaining similar collision rates.

Table 8: Guidance schedule ablation averaged over the 9 robot embodiments used in the paper. Scheduled guidance uses Eq. (16); constant guidance keeps $\gamma _ { t } = \gamma = 1 . 0$ for all denoising steps.

<table><tr><td>Task</td><td>Guidance scale</td><td>TSR ↑</td><td>RWD ↑</td><td>COR ↓</td></tr><tr><td rowspan="2">PLACETOAST</td><td>Scheduled  $\gamma_t$ </td><td>71.8%</td><td>0.812</td><td>6.9%</td></tr><tr><td>Constant  $\gamma_t$ </td><td>70.2%</td><td>0.796</td><td>5.6%</td></tr><tr><td rowspan="2">TURNONFAUCET</td><td>Scheduled  $\gamma_t$ </td><td>60.6%</td><td>0.833</td><td>27.1%</td></tr><tr><td>Constant  $\gamma_t$ </td><td>57.8%</td><td>0.827</td><td>26.1%</td></tr><tr><td rowspan="2">MAKECOFFEE</td><td>Scheduled  $\gamma_t$ </td><td>57.2%</td><td>0.670</td><td>2.8%</td></tr><tr><td>Constant  $\gamma_t$ </td><td>46.2%</td><td>0.593</td><td>2.8%</td></tr></table>

Finally, we sweep the cost-gradient guidance scale $\rho$ for the gradient-based baseline (Joint w/ CG). This verifies that the baseline is not disadvantaged by a single poorly chosen scale. As shown in Fig. 12, tuning $\rho$ can reduce collisions, but the unconstrained gradient update often degrades task performance, especially in tasks requiring precise end-effector motion.

![](images/8377b978111b003641bce0ad2bd5c9e8fa76ce7a491a19e4ba4b5a206c05a904.jpg)  
Figure 12: Cost-gradient guidance-strength sensitivity analysis. Solid curves show the CG baseline under different $\rho ,$ while dashed horizontal lines show EMBODISTEER with its default setting. Tuning $\rho$ can reduce collisions, but CG consistently gives a worse task–collision tradeoff.

## C.5 Obstacle-Present Training and Obstacle-Estimation Noise

Obstacle-Present Training. The main benchmark trains Cartesian policies on obstacle-free demonstrations and introduces surrounding obstacles at deployment. To test whether exposing the Cartesian policy to obstacles during training can improve its subsequent collision-aware task performance, we additionally train a policy on obstacle-present demonstrations and evaluate it on TURNON-FAUCET. These demonstrations remain body-free: they can encode obstacle avoidance for the end effector, but contain neither the target robot’s joint configurations nor its whole-body collision states. Thus, adding obstacles to the demonstrations alone does not provide the embodiment-specific information needed to learn whole-body avoidance for the deployed arm. The obstacle-present training baseline increases TSR by 2.4 percentage points over obstacle-free training, while remaining 16.6 percentage points below EMBODISTEER. The comparison indicates that obstacle-present training alone does not recover the task performance achieved by joint-space guidance in this setting.

Obstacle-Estimation Noise. Our guidance evaluates whole-body clearance against modeled cuboid obstacles. To examine how errors in that geometric input affect execution, we perturb the obstacle parameters supplied to guidance on MAKECOFFEE, while keeping the evaluated scene unchanged. Translation and cuboid extent use standard deviations of 5 or 10 mm; rotation uses a standard deviation of $3 ^ { \circ }$ in both noise settings. Tab. 9 shows that task success decreases and collision occurrence increases with larger estimation noise.

Table 9: Obstacle-estimation noise on MAKECOFFEE. Translation and cuboid extent share the listed standard deviation; rotation standard deviation is 3 for both perturbed settings.

<table><tr><td>Translation / extent std</td><td>TSR (%) ↑</td><td>COR (%) ↓</td></tr><tr><td>No perturbation</td><td>57.2</td><td>2.8</td></tr><tr><td>5 mm</td><td>50.4</td><td>4.4</td></tr><tr><td>10 mm</td><td>44.6</td><td>6.8</td></tr></table>

## C.6 Demonstration Coverage Settings

The policies used in the main experiments are trained on 200 diverse obstacle-free demonstrations. To separate the effects of data quantity and motion diversity, the MAKECOFFEE ablations in Fig. 6 retrain the Cartesian policy under two sweeps. The quantity sweep uses 20, 50, 100, and 200 demonstrations. The diversity sweep holds the demonstration count at 200 and scales the waypoint randomization used by the demonstration motion planner to 10%, 33%, 67%, and 100% of the original data-generation setting. Thus, the diversity percentages measure waypoint randomization, rather than the fraction of demonstrations retained. Each trained policy is evaluated with EE, EE w/ CBF, Joint w/ CG, and EMBODISTEER. Fig. 6 shows increasing success with both data quantity and waypoint randomization, supporting broad motion coverage for guided policy deployment.

## C.7 Candidate Sampling Budget

The EE w/ Sampling baseline generates multiple Cartesian action chunks from the frozen policy, realizes them on the target robot, and selects the candidate with the largest whole-body clearance. To examine whether a larger candidate pool can improve this strategy, we vary K from 4 to 32 on TURNONFAUCET; the main benchmark uses $K = 1 6$ . Tab. 10 reports success averaged over the nine robot embodiments. Increasing the budget yields modest gains: even K = 32 reaches 45.9% TSR, compared with 60.6% for EMBODISTEER in Tab. 1. Inference latency grows approximately linearly with K, reaching 5.55 s per chunk at K = 16 and 10.97 s at K = 32 (Tab. 10), making larger candidate pools costly for closed-loop replanning.

Table 10: Candidate-budget sensitivity of EE w/ Sampling on TURNONFAUCET, averaged over nine robot embodiments, and inference latency (mean ± standard deviation) measured separately on PLACETOAST with Panda and an RTX 4090.

<table><tr><td>Candidates K</td><td>4</td><td>8</td><td>16</td><td>32</td></tr><tr><td>TSR (%)</td><td>43.1</td><td>44.4</td><td>43.9</td><td>45.9</td></tr><tr><td>Latency (s)</td><td> $1.4803 \pm 0.0255$ </td><td> $2.8291 \pm 0.0536$ </td><td> $5.5529 \pm 0.0834$ </td><td> $10.9728 \pm 0.1641$ </td></tr></table>

## C.8 Action-Chunk Continuity

The QP corrects horizon states independently. We examine whether these corrections change jointtrajectory continuity using paired guided and unguided chunks on Panda and UR5 across the three tasks. Each setting uses 10 random seeds with one episode per seed, giving 60 episodes and 2,540 paired chunks in total. Each pair shares the observation, current joint state, and diffusion random state; only the guided chunk is executed. These are trajectory diagnostics, rather than separate guided/unguided rollout performance comparisons. The gripper is excluded from all metrics.

For adjacent configurations, we measure $s _ { i } = \| q _ { i } - q _ { i - 1 } \| _ { 2 } / \sqrt { D }$ , with $q _ { 0 }$ the current configuration and D the number of arm joints. We report the 95th percentile over the executed prefixes, including the current-state-to-first-action transition. The second-difference ratio compares the maximum second joint difference of guided and unguided trajectories over the executed prefix. QP activity is measured at the final denoising iteration over that prefix. Tab. 11 shows that the 95th-percentile step changes by at most 1.6% in five of six settings; TURNONFAUCET/Panda increases by 12.7%.

Table 11: Continuity over executed action prefixes. P95 step is measured in radians; “ratio” is guided/unguided. r<sub>2</sub> is the maximum second-difference ratio. QP activity is the percentage of active horizon-point constraints.

<table><tr><td>Task</td><td>Robot</td><td>P95 step unguided</td><td>P95 step guided</td><td>Ratio</td><td> $r_2$ </td><td>QP active (%)</td></tr><tr><td rowspan="2">MAKECOFFEE</td><td>Panda</td><td>0.0348</td><td>0.0351</td><td>1.011</td><td>1.000</td><td>10.5</td></tr><tr><td>UR5</td><td>0.0216</td><td>0.0216</td><td>0.999</td><td>4.582</td><td>28.7</td></tr><tr><td rowspan="2">TURNONFAUCET</td><td>Panda</td><td>0.0392</td><td>0.0442</td><td>1.127</td><td>1.000</td><td>28.8</td></tr><tr><td>UR5</td><td>0.0135</td><td>0.0135</td><td>1.003</td><td>1.292</td><td>54.5</td></tr><tr><td rowspan="2">PLACETOAST</td><td>Panda</td><td>0.0244</td><td>0.0246</td><td>1.007</td><td>1.000</td><td>7.7</td></tr><tr><td>UR5</td><td>0.0171</td><td>0.0174</td><td>1.016</td><td>2.693</td><td>7.2</td></tr></table>

Overall, these results show that collision guidance largely preserves joint-step continuity across tasks and embodiments. Although the maximum second-difference ratios reach 4.58 and 2.69 on UR5 for MAKECOFFEE and PLACETOAST, respectively, the P95 step sizes in these settings change by at most 1.6%, indicating that the bulk of the joint-step distribution remains stable under guidance.

## C.9 Additional Qualitative Results

Fig. 13 shows representative simulation rollouts across the three benchmark tasks. The base Cartesian policy often produces task-directed end-effector motion but cannot reason about out-ofdistribution obstacles, which leads to end-effector collisions near the task object. It also ignores the target robot’s arm geometry, leading to arm-body obstacle contact. In contrast, EMBODISTEER preserves the same task intent while steering both the end-effector and the full robot body around the obstacle.

![](images/276a46854c50508f5d6113eb28e94bc8ccbdda1cd020b3c9f2781370d9d709fb.jpg)  
Figure 13: Additional simulation qualitative results. The base Cartesian policy can collide at the end-effector or along the arm body when obstacles are added at test time, while EMBODISTEER generates collision-aware whole-body motion and completes the task.

## D Real-World Results

## D.1 Setup and Annotation Protocol

Each real-world task uses 200 UMI handheld-gripper demonstrations collected without extra obstacles. Because these demonstrations are collected with a handheld gripper rather than a particular robot arm, the training data are embodiment-agnostic. For each task, a single Cartesian diffusion policy checkpoint is trained and shared by UR5 and Franka Panda without robot-specific finetuning. The three tasks are illustrated in Fig. 14: MAKEICEDCOFFEE requires grasping blocks from a bowl, which represent ice cubes, and placing them into a cup; PUTFLOWERINVASE requires grasping the bouquet and inserting it into the vase; and ARRANGEBANANA requires grasping a banana from a pen holder and placing it into a cup. For each robot-task pair, we evaluate 10 no-obstacle trials and 10 obstacle-present trials distributed across the obstacle types in Fig. 15. A trial is marked successful if the task object reaches the intended final state without timeout, object drop, or human intervention.

![](images/5347d0fde5fed84b18c4d41b5091159a585bafbff2eb698e22c126234eb9c250.jpg)  
Figure 14: Real-world task protocols. Each row shows representative stages of one UMI-trained task: MAKEICEDCOFFEE, PUTFLOWERINVASE, and ARRANGEBANANA.

Task success and collision are annotated independently. For safety in the real-world setup, obstacles are movable: after contact, they can be knocked over or pushed aside, unlike the fixed obstacles in simulation that often stop the robot motion. As a result, a rollout can still finish the task after a collision. COR therefore measures whether any end-effector or arm-body obstacle contact occurs during the rollout, separately from task success.

## D.2 Full Real-World Results

Table 12 reports the real-world results by robot and task. Each task checkpoint is shared by both robots.

## D.3 Obstacle Layouts and Qualitative Results

The real-world obstacle layouts are shown in Fig. 15. The obstacle type and pose are varied across episodes. These layouts require the robot to avoid collisions with both the arm links and the end effector.

Table 12: Real-world results by robot and task. TSR and COR are reported as success/collision counts over 10 trials for each robot-task pair; the total row aggregates over 60 trials.

<table><tr><td rowspan="2">Robot</td><td rowspan="2">Task</td><td>No Obstacle</td><td colspan="2">Obstacle, Base Policy</td><td colspan="2">Obstacle, EMBODISTEER</td></tr><tr><td>TSR ↑</td><td>TSR ↑</td><td>COR ↓</td><td>TSR ↑</td><td>COR ↓</td></tr><tr><td rowspan="3">UR5</td><td>MAKEICEDCOFFEE</td><td>9/10</td><td>4/10</td><td>10/10</td><td>8/10</td><td>1/10</td></tr><tr><td>PUTFLOWERINVASE</td><td>7/10</td><td>4/10</td><td>9/10</td><td>6/10</td><td>1/10</td></tr><tr><td>ARRANGEBANANA</td><td>8/10</td><td>4/10</td><td>9/10</td><td>9/10</td><td>0/10</td></tr><tr><td rowspan="3">Panda</td><td>MAKEICEDCOFFEE</td><td>9/10</td><td>6/10</td><td>10/10</td><td>8/10</td><td>1/10</td></tr><tr><td>PUTFLOWERINVASE</td><td>6/10</td><td>1/10</td><td>10/10</td><td>6/10</td><td>0/10</td></tr><tr><td>ARRANGEBANANA</td><td>10/10</td><td>6/10</td><td>10/10</td><td>10/10</td><td>1/10</td></tr><tr><td>Total</td><td>All tasks</td><td>49/60</td><td>25/60</td><td>58/60</td><td>47/60</td><td>4/60</td></tr></table>

![](images/97e36055cb72292cea0129eeb76774766a2b1d9bd921f28c64c4ba248b7e0e47.jpg)  
Figure 15: Real-world obstacle layouts. The obstacles are highlighted with yellow masks. The obstacle type and pose are varied across episodes.

Fig. 17 shows representative real-world rollouts for all three tasks. Each row compares the base Cartesian policy and EMBODISTEER on both UR5 and Franka Panda. The qualitative results mirror the quantitative trend in Table 12: the base policy often reaches toward the task object while contacting the obstacle, whereas EMBODISTEER modifies the whole-body motion to avoid end-effector and arm-body obstacles.  
![](images/b3ca37ad2fa0dcb9ffde5f1aeb89883b9286c67256fac81e9645649bf35333e3.jpg)  
Figure 16: Representative failure modes. Guidance may move the robot into out-of-distribution states, fail to fully prevent collision under local linearization and clipping, or disrupt precisioncritical pick/place motions.

![](images/cb5a11eb8269e0e45054b06ea37b5513701d96362d04196bd1c47ffa549555b6.jpg)  
Figure 17: Real-world qualitative results. From top to bottom: MAKEICEDCOFFEE, PUTFLOWER-INVASE, and ARRANGEBANANA. Each row compares the base Cartesian policy and EMBODIS-TEER on UR5 and Franka Panda under obstacle layouts that test end-effector and arm-body avoidance.

## E Failure Mode Analysis

Fig. 16 summarizes three representative failure modes observed in our experiments. First, although guidance improves collision avoidance on average, a large correction can steer the robot into joint states that are rarely covered by the demonstrations. The frozen Cartesian policy may then receive observations outside its training distribution and fail to recover task progress. Second, the CBF-QP update is based on a local linearization and clipped joint correction, so it provides a practical avoidance direction rather than a formal collision-free certificate; collisions can still occur in highly constrained layouts or when the required correction is too large. Third, some task phases are more sensitive to small end-effector deviations than others. In pick-and-place tasks, for example, guidance applied near grasping or placement can perturb precision-critical motion and cause grasp misses, drops, or misplacement, even when the resulting motion avoids obstacles.