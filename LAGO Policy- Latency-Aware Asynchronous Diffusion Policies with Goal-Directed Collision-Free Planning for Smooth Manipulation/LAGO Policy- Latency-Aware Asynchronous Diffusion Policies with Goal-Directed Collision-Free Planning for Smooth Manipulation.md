# LAGO Policy: Latency-Aware Asynchronous Diffusion Policies with Goal-Directed Collision-Free Planning for Smooth Manipulation

Guowei Shi<sup>1,2,4,∗</sup>, Xupeng Xie<sup>1,2,4,∗</sup>, Yiming Luo<sup>3</sup>, Jian Guo<sup>2</sup>, Jun Ma<sup>1</sup>, Boyu Zhou<sup>4,†</sup>

Abstract— Diffusion-based visuomotor policies deployed with asynchronous inference often exhibit inter-chunk discontinuities and lack explicit mechanisms for obstacle-aware execution, leading to jerky motions and collisions that hinder reliable manipulation in real-world scenes. To address these issues, we propose LAGO Policy, a unified asynchronous action-generation framework that integrates trajectory optimization with diffusion policy for smooth and safe execution. LAGO Policy improves inter-chunk consistency via latency-aware classifier-free guidance conditioning on future actions. It further enables goaldirected collision-free trajectory planning by predicting a taskrelevant interaction goal from demonstrations. Finally, spatialtemporal trajectory optimization refines the actions to be executed for low-jerk and feasible motion. Extensive real-world experiments demonstrate that LAGO Policy achieves smooth collision-free execution with high task success across challenging manipulation tasks. https://lago-policy.github.io/

## I. INTRODUCTION

Recently, visuomotor imitation learning has emerged as a scalable paradigm for acquiring robotic manipulation skills directly from expert demonstrations [1]–[4]. In this context, generative control policies (GCPs) instantiate closed-loop visuomotor controllers via generative models, mapping highdimensional observations to multi-step action sequences. In particular, diffusion-based policies generate actions through iterative denoising and have demonstrated strong performance across a broad range of manipulation tasks. Despite these advances, deploying diffusion-based GCPs on physical manipulators still faces fundamental challenges.

Firstly, ensuring continuous and smooth motion under non-negligible inference latency remains unresolved [5]. To avoid stop-and-infer execution, policies are typically deployed with asynchronous inference, where the next action chunk is generated in parallel while the robot executes the current one. This creates a perception-execution misalignment because each chunk is conditioned on observations acquired before inference, but is executed after inference finishes, when the robot and scene states may have changed. Consequently, consecutive chunks can disagree over their overlap, producing jerky motions that degrade fine manipulation. Existing action-conditioning methods use predicted future actions to improve inter-chunk consistency, but tightly coupling them with observations makes the policy brittle to temporally shifted future-action conditions and prone to discontinuities at chunk boundaries [6].

![](images/b564dc9c207233088d4e913e4139d5ddc341206252dfe3942d89ae72e0fcd34e.jpg)  
Fig. 1. LAGO Policy. Latency-aware classifier-free guidance improves inter-chunk consistency under asynchronous inference, goal prediction enables goal-directed collision-free planning, and spatial-temporal trajectory optimization refines the executed motion for smooth execution.

Secondly, GCPs lack an efficient mechanism for safe action generation in the presence of unseen obstacles. Since most GCPs are trained to imitate demonstrations without explicitly modeling the collision-free feasible set, they can generate geometrically infeasible trajectories once out-ofdistribution obstacles obstruct the motion toward the task goal, especially during long-range goal-reaching motions. Existing inference-time safeguards, such as collision-cost guidance or safety filters, are inherently local and shorthorizon, often yielding suboptimal avoidance behaviors. Moreover, they can push execution away from the demonstration manifold, causing safety-induced distribution shift and compounding errors under closed-loop execution.

To tackle these challenges, we propose LAGO Policy, a unified asynchronous action-generation framework that integrates model-based trajectory optimization with diffusion-based GCPs to achieve smooth, temporally consistent, and safe execution in real-world manipulation. For smooth and temporally consistent execution, we introduce a latency-aware training scheme with classifier-free guidance (CFG) [7]. Specifically, we decouple the future-action condition from observation features and randomize the temporal delay of the future-action condition during training, which makes the policy robust to temporal shifts in conditioning signals. For safe action generation, we propose a goaldirected paradigm that jointly predicts an action chunk and a task-relevant goal learned from expert demonstrations. When the end-effector is far from the predicted goal or the direct motion toward it is obstructed by obstacles, we activate goal-conditioned trajectory optimization to generate a smooth collision-free motion toward the predicted goal. This planned motion steers execution back toward in-distribution states, thereby enabling task completion. We further apply spatial-temporal trajectory optimization to refine the executed motion, including both policy-generated action chunks and planner-generated trajectories, for smooth and timeefficient execution. We validate our method through realworld manipulation tasks, demonstrating smooth and safe action generation while improving task success rates. Our contributions are as follows:

1) A latency-aware training scheme leveraging CFG that decouples observation and future-action conditioning with randomized delay, enabling smooth asynchronous chunked execution.

2) A goal-directed safe generation paradigm that couples demonstration-derived goal prediction with trajectory optimization to produce safe and smooth motions.

3) Comprehensive real-world validation demonstrating improved smoothness, safety, and task success. The code will be made public.

## II. RELATED WORKS

## A. Diffusion Models for Robotics

Diffusion models were originally developed for highfidelity image generation [8]. More recently, diffusion models have been adopted for planning and policy learning, where their ability to represent multimodal action distributions, scale to high-dimensional action spaces, and train stably is particularly attractive. In motion planning, Diffuser formulates trajectory synthesis as an iterative denoising process that generates complete trajectories [9]. In imitation learning, Diffusion Policy models a closed-loop visuomotor controller via conditional action diffusion and executes in a receding-horizon manner to preserve reactivity [10]. However, this closed-loop, chunked execution also exposes a practical deployment bottleneck. Multi-step denoising incurs non-negligible inference latency, and synchronous execution often induces stop-and-infer behavior with pauses between consecutive action chunks. Building on Diffusion Policy, our work targets continuous real-robot execution under such latency via asynchronous inference.

## B. Asynchronous Inference

Asynchronous inference enables continuous execution for diffusion-based policies by generating the next action chunk while executing the current one. However, the resulting prediction-execution misalignment often breaks inter-chunk consistency and induces jerky motions. Several recent works mitigate this issue via additional conditioning or correction mechanisms. SAIL improves continuity with CFGstyle future-action conditioning [6], but its observationconcatenated implementation is brittle to future-action condition shifts. VLASH makes the policy future-state-aware by rolling the robot state forward with committed actions and conditioning on the estimated execution-time state, thereby reducing state staleness under asynchronous inference [11]. However, this mechanism does not explicitly promote consistency in the overlapping segment between consecutive action chunks. Training-time RTC improves continuity via inpainting-style action conditioning, but it is tailored to diffusion-transformer-like architectures [12] and less directly compatible with CNN-based denoisers that use a globally injected diffusion timestep [13]. Our work follows CFG-style conditioning and explicitly addresses future-action condition shifts under asynchronous inference.

## C. Collision-Free Execution for Learned Policies

Collision-free execution of learned manipulation policies is often achieved via inference-time safeguards that modify nominal policy outputs to avoid collisions. Costguided diffusion methods inject collision-related costs into the denoising process to bias sampling toward collision-free trajectories [14]. VLSA integrates control barrier functions as a safety layer to enforce explicit state constraints [15]. RAIL applies a reachability-based safety filter that validates candidate motions and switches to a safe fallback when violations are detected [16]. A common thread across these approaches is that collision avoidance is handled via local, short-horizon corrections at test time, which can yield suboptimal motions, introduce frequent interventions, and push execution toward states that are rare in the demonstrations, degrading closed-loop task performance. In contrast, our method performs goal-directed collision-free planning toward a demonstration-derived interaction goal, producing globally consistent avoidance motion and steering execution back toward in-distribution states to support task completion.

## III. METHOD

Fig. 2 overviews LAGO Policy. At each asynchronous cycle, a diffusion policy with latency-aware CFG generates an action chunk, while a goal-prediction head estimates a task-relevant goal. We trigger goal-conditioned trajectory generation when the direct motion to the predicted goal is obstructed by obstacles or the end-effector is far from the goal; otherwise we execute the policy output. All executed motions are further refined by spatial-temporal trajectory optimization for smooth and feasible execution.

## A. Preliminaries

Our method is built on a CNN-based Diffusion Policy backbone [10]. It models the conditional distribution $p _ { \theta } ( \mathbf { A } _ { t } \mid \mathbf { O } _ { t } )$ over a $T _ { p }$ -step future action sequence ${ \bf A } _ { t } \ =$ $\left\{ \mathbf { a } _ { t + k } \right\} _ { k = 0 } ^ { T _ { p } - 1 }$ , conditioned on the most recent $T _ { o }$ observations $\mathbf O _ { t } \ = \ \left[ \mathbf o _ { t - T _ { o } + 1 } , \dots , \mathbf o _ { t } \right]$ , where each observation decomposes as $\mathbf o _ { t } = \left[ \mathbf v _ { t } , \mathbf x _ { t } \right]$ with visual input $\mathbf { v } _ { t }$ and robot state $\mathbf { x } _ { t }$ . The visual input is encoded as ${ \mathbf z } _ { t } = f _ { \mathrm { e n c } } ( { \mathbf v } _ { t } )$ , and we form the per-timestep conditioning vector $\mathbf { c } _ { t } = [ \mathbf { z } _ { t } , \mathbf { x } _ { t } ]$ . The denoising network is conditioned on the history conditioning vector $\mathbf { C } _ { t } = [ \mathbf { c } _ { t - T _ { o } + 1 } , \ldots , \mathbf { c } _ { t } ] ,$ , and the observation features are fused to the policy network through FiLM [17].

Diffusion Policy adopts a DDPM [8] denoising formulation. The forward diffusion process progressively corrupts a clean action sequence by adding Gaussian noise:

$$
q (\mathbf {A} _ {t} ^ {n} \mid \mathbf {A} _ {t} ^ {n - 1}) = \mathcal {N} \bigl (\sqrt {\alpha_ {n}} \mathbf {A} _ {t} ^ {n - 1}, (1 - \alpha_ {n}) \mathbf {I} \bigr),\tag{1}
$$

![](images/3bdace2e93d68f0218b985d3acfaf47f35bc7d22cc5897a36c37dd52b99564dc.jpg)  
Fig. 2. Overview of LAGO Policy. LAGO Policy unifies temporally consistent action generation with goal-directed collision-free motion generation for smooth robotic manipulation. Latency-aware classifier-free guidance improves inter-chunk consistency, goal prediction enables task-directed planning, and spatial-temporal optimization refines the resulting motion for low-jerk, feasible execution.

![](images/3d0d8d59d5f4f15a51b2798d2da50fe34ceb6453acb4e5eec7b4e20f15031164.jpg)  
Fig. 3. Motivation for latency-aware CFG under asynchronous inference. This figure highlights a key issue of future-action-conditioned CFG: while it improves cross-chunk continuity under temporally aligned future-action conditions, inference latency can shift the effective condition at deployment and break chunk-to-chunk smoothness. (a) Aligned case. Under asynchronous prediction and execution, CFG conditions denoising on the future-action condition ${ \bf A } _ { t } ^ { c }$ extracted from the unexecuted part of the previous chunk (red box), improving consistency in the overlap and smoothing the executed trajectory (black dashed line). (b) Training with aligned future-action conditions. $\mathbf { A } _ { t }$ and $\mathbf { \bar { A } } _ { t } ^ { c }$ are sampled from the same demonstration chunk and aligned to start at t $( \mathrm { i } . \mathrm { e } . , \delta = 0 )$ . (c) Shifted future-action conditions at deployment. Inference latency shifts the effective future-action condition by δ steps, misaligning the intended overlap and inducing discontinuities and jerk motions at chunk boundaries.

where $\begin{array} { r } { \alpha _ { n } \in ( 0 , 1 ) , \bar { \alpha } _ { n } = \prod _ { i = 1 } ^ { n } \alpha _ { i } } \end{array}$ and $n \in \{ 1 , \ldots , N \}$ denotes the diffusion step. Equivalently,

$$
\mathbf {A} _ {t} ^ {n} = \sqrt {\bar {\alpha} _ {n}} \mathbf {A} _ {t} ^ {0} + \sqrt {1 - \bar {\alpha} _ {n}}   \boldsymbol {\epsilon}, \qquad \boldsymbol {\epsilon} \sim \mathcal {N} (\mathbf {0}, \mathbf {I}).\tag{2}
$$

The reverse process is parameterized by a noise-prediction network $\mathbf { \epsilon } _ { \theta } ( \mathbf { A } _ { t } ^ { n } , \mathbf { C } _ { t } , n )$ . Let $\tilde { \mathbf { A } } _ { t } ^ { n } = \sqrt { \bar { \alpha } _ { n } } \mathbf { A } _ { t } ^ { 0 } + \sqrt { 1 - \bar { \alpha } _ { n } } \pmb { \epsilon }$ . The denoiser is trained with the standard DDPM objective:

$$
\mathcal {L} _ {\mathrm{DP}} = \mathbb {E} _ {t, n, \epsilon} \Big [ \big \| \boldsymbol {\epsilon} - \boldsymbol {\epsilon} _ {\theta} (\tilde {\mathbf {A}} _ {t} ^ {n}, \mathbf {C} _ {t}, n) \big \| _ {2} ^ {2} \Big ].\tag{3}
$$

At inference, we use DDIM [18] sampling to efficiently generate an action sequence, and deploy it in a recedinghorizon manner by executing only the first $T _ { a }$ actions $( T _ { a } <$ $T _ { p } )$ before replanning.

## B. Latency-Aware Classifier-Free Guidance

CFG is a guidance technique for conditional diffusion models that strengthens adherence to a condition by combining conditional and unconditional denoising predictions during sampling with a tunable guidance scale. In diffusionbased policies, CFG provides a lightweight way to incorporate a future-action condition to improve inter-chunk temporal consistency. However, as illustrated in Fig. 3, existing future-action-conditioned CFG methods assume temporally aligned future-action conditions during training and can become brittle when inference latency shifts the effective condition at deployment [6].

As illustrated in Fig. 3(a), the start time of the previous inference cycle is denoted as $t ^ { - }$ . Under an asynchronous inference manner, the current inference cycle starts at

$$
t \triangleq t ^ {-} + (T _ {a} - T _ {d}),\tag{4}
$$

where $T _ { d }$ denotes the inference delay. There remains a segment of actions that were predicted in the previous cycle but not yet executed. As shown in the red box of Fig. 3(a), we extract a length-T<sub>f</sub> subsequence from this unexecuted segment starting at t and define it as the future-action condition for the current cycle: $\mathbf { A } _ { t } ^ { c } ~ = ~ [ \mathbf { a } _ { t } , \dots , \mathbf { a } _ { t + T _ { f } - 1 } ]$ When ${ \bf A } _ { t } ^ { c }$ is temporally aligned with the overlap between consecutive chunks, it provides an effective guidance signal that promotes cross-chunk continuity.

In SAIL [6], the future-action condition is concatenated with the observation, $\mathbf { o } _ { t } = \left[ \mathbf { v } _ { t } , \mathbf { x } _ { t } , \mathbf { A } _ { t } ^ { c } \right]$ , which yields the pertimestep conditioning vector $\mathbf { c } _ { t } = [ \mathbf { z } _ { t } , \mathbf { x } _ { t } , \mathbf { A } _ { t } ^ { c } ]$ . Consequently, the same future-action condition is replicated across the $T _ { o } .$ step history conditioning vector $\mathbf { C } _ { t } .$ , tightly coupling ${ \bf A } _ { t } ^ { c }$ with the history of visual and proprioceptive features. Since ${ \bf A } _ { t } ^ { c }$ is meant to constrain the overlap between consecutive action chunks, this coupling makes the noise-prediction network $\epsilon _ { \theta } ( \mathbf { A } _ { t } ^ { n } , \mathbf { C } _ { t } , n )$ sensitive to temporal shifts in ${ \bf A } _ { t } ^ { c } ,$ which can degrade chunk-to-chunk continuity.

To avoid this issue, we treat ${ \bf A } _ { t } ^ { c }$ as a separate conditioning variable and inject it through CFG during sampling. Given a noisy action sequence $\mathbf { A } _ { t } ^ { n }$ at diffusion step n and the observation conditioning $\mathbf { C } _ { t } .$ our CFG computes conditional and unconditional noise predictions as

$$
\left\{ \begin{array}{l} \boldsymbol {\epsilon} _ {\text {cond}} := \boldsymbol {\epsilon} _ {\theta} (\mathbf {A} _ {t} ^ {n},   \mathbf {C} _ {t},   \mathbf {A} _ {t} ^ {c},   n), \\ \boldsymbol {\epsilon} _ {\text {uncond}} := \boldsymbol {\epsilon} _ {\theta} (\mathbf {A} _ {t} ^ {n},   \mathbf {C} _ {t},   \varnothing ,   n), \end{array} \right. \quad \varnothing \equiv \mathbf {0},\tag{5}
$$

and combines them to obtain the guided noise estimate

$$
\hat {\epsilon} = \epsilon_ {\mathrm{uncond}} + w \big (\epsilon_ {\mathrm{cond}} - \epsilon_ {\mathrm{uncond}} \big),\tag{6}
$$

where w is the guidance scale. In DDIM sampling, ˆϵ is used in place of the original noise prediction at each denoising step, yielding a denoising trajectory that is more consistent with the future-action condition ${ \bf A } _ { t } ^ { c }$

In conventional training, the future-action condition is constructed to be temporally aligned with the target action chunk, as shown in Fig. 3(b). However, under asynchronous inference, the time at which ${ \bf A } _ { t } ^ { c }$ is generated is decoupled from the time at which it is actually used for action execution. As shown in Fig. 3(c), inference latency can shift the effective ${ \bf A } _ { t } ^ { c }$ by δ action steps relative to the aligned training case, so the denoiser is guided to match the overlap at incorrect temporal positions, degrading interchunk continuity. Training only with aligned future-action conditions therefore makes the policy brittle to such shifts.

To address this issue, we introduce delay randomization during training. Specifically, we sample an offset δ in action steps and shift ${ \bf A } _ { t } ^ { c }$ accordingly, as detailed in Algo. 1. To train a single noise-prediction network for both conditional and unconditional denoising, we adopt the CFG dropout scheme [7]. This exposes the denoiser to shifted future-action conditions during training, improving robustness to futureaction condition shifts and enabling smoother inter-chunk execution under asynchronous inference.

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1: Latency-Aware CFG Training

Input: Demo dataset D; horizons  $T_{o}, T_{p}, T_{f}$ ;
diffusion steps  $n \sim \{1, \ldots, N\}$ ; drop prob p;
shift distribution  $\delta \sim P_{\delta}$ 

for each training iteration do
    Sample timestep t and extract
    $O_{t}, A_{t}^{0} = [a_{t}, \ldots, a_{t+T_{p}-1}]$ ;
    Sample latency shift  $\delta \sim P_{\delta}$ ;
    Set  $A_{t}^{c} = [a_{t+\delta}, \ldots, a_{t+\delta+T_{f}-1}]$ ;
    With probability p:  $A_{t}^{c} \leftarrow 0$ ;
    Form noisy actions  $A_{t}^{n} = \sqrt{\bar{\alpha}_{n}} A_{t}^{0} + \sqrt{1 - \bar{\alpha}_{n}} \epsilon$ ;
    Update  $\theta$  by minimizing
    $\| \epsilon - \epsilon_{\theta}(A_{t}^{n}, C_{t}, A_{t}^{c}, n) \|_{2}^{2}$ ;
</div>

## C. Goal-Prediction Head for Diffusion Policy

Prior safe diffusion-policy deployments remain within a receding-horizon control manner, applying safety corrections only to short-horizon action chunks. However, efficient and globally coherent obstacle avoidance requires an explicit task objective beyond short-horizon correction. Therefore, we equip the diffusion policy with a goal-prediction head that enables task-relevant goal prediction, which guides collisionfree trajectory generation in cluttered scenes.

Specifically, we attach a goal-prediction head to the denoising U-Net [19] to predict a task-relevant goal from its bottleneck representation $\mathbf { h } _ { t } .$ , as shown in Fig. 2. The feature $\mathbf { h } _ { t }$ is taken from the lowest-resolution stage of the U-Net encoder and serves as a compact latent summary that aggregates global context from the observation conditioning while capturing the coarse motion intent encoded in the denoised action chunk. This combination makes $\mathbf { h } _ { t }$ well suited for inferring task-relevant interaction goals. Global average pooling (GAP) is applied to $\mathbf { h } _ { t }$ to average over the action-horizon axis, yielding a fixed-length feature for goal prediction. The goal $\hat { \bf g } _ { t }$ is then predicted by a multi-layer perceptron (MLP):

$$
\hat {\mathbf {g}} _ {t} = g _ {\psi} (\mathrm{GAP} (\mathbf {h} _ {t})).\tag{7}
$$

Goal supervision is imposed with the regression loss

$$
\mathcal {L} _ {\mathrm{goal}} (\theta , \psi) = \| \mathbf {g} _ {t} ^ {*} - \hat {\mathbf {g}} _ {t} \| _ {2} ^ {2},\tag{8}
$$

where θ denotes the parameters of Diffusion Policy including the observation encoders and denoising U-Net, ψ denotes the parameters of the goal-prediction head, and $\mathbf { g } _ { t } ^ { * }$ is a task-relevant goal extracted from expert demonstrations. The overall training objective is a weighted sum of the diffusion denoising loss and the goal loss

$$
\mathcal {L} (\theta , \psi) = \mathcal {L} _ {\mathrm{DP}} (\theta) + \lambda_ {\mathrm{goal}} \mathcal {L} _ {\mathrm{goal}} (\theta , \psi),\tag{9}
$$

where nonnegative weight $\lambda _ { \mathrm { g o a l } }$ balances noise prediction and goal prediction.

## D. Collision-Free Trajectory Generation

After obtaining the task-relevant goal $\hat { \bf g } _ { t }$ , we generate a task-space end-effector motion toward $\hat { \mathbf { g } } _ { t }$ . The end-effector is approximated as a collision sphere centered at the current position $\mathbf { x } _ { t } ^ { \mathrm { e e } }$ with radius $r _ { \mathrm { e e } }$ . We perform a line-of-sight collision check along the straight segment from $\mathbf { x } _ { t } ^ { \mathrm { e e } }$ to $\hat { \mathbf { g } } _ { t }$ testing whether the swept sphere intersects any obstacles, and compute the goal distance $d _ { t } = \lVert \mathbf { x } _ { t } ^ { \mathrm { e e } } - \hat { \mathbf { g } } _ { t } \rVert _ { 2 }$ . Trajectory optimization is triggered if a collision is detected or if $d _ { t }$ exceeds a threshold $d _ { \mathrm { t h } }$ . When the line-of-sight motion is collision-free, the optimizer generates a smooth and feasible goal-reaching trajectory; when obstacles obstruct the motion, the collision term enforces obstacle clearance.

The avoidance module draws inspiration from EGO-Planner [20], an efficient spline-based optimizer that does

![](images/092d0b0c8cfce3cb64eb0bde8654bd12432cfbfb405c2e0a6278c062df15ca75.jpg)

Fig. 4. The initial trajectory $\Phi _ { \mathrm { i n i t } }$ collides with an unseen obstacle. $\mathbf { A } ^ { * }$ provides a collision-free connector Γ, from which $\{ \mathbf { p } , \mathbf { v } \}$ pairs guide optimization to a smooth collision-free trajectory Φ<sup>∗</sup> not require a collision-free initialization. This property aligns with our setting, where a natural initialization is the direct motion toward $\hat { \bf g } _ { t }$ , which may intersect obstacles. Specifically, the end-effector position trajectory is parameterized by a uniform B-spline curve $\Phi ( t )$ whose decision variables are the control points $\mathbf { Q } = \{ \mathbf { Q } _ { i } \} _ { i = 1 } ^ { N _ { c } }$

$$
\Phi (t) = \sum_ {i = 1} ^ {N _ {c}} B _ {i, d} (t) \mathbf {Q} _ {i},\tag{10}
$$

where $N _ { c }$ is the number of control points and $B _ { i , d } ( t )$ denotes the B-spline basis of degree $d .$

As shown in Fig. 4, an initial trajectory $\Phi _ { \mathrm { i n i t } }$ is constructed to satisfy terminal constraints between the current endeffector position $\mathbf { x } _ { t } ^ { \mathrm { e e } }$ and the predicted goal $\hat { \bf g } _ { t }$ , without requiring collision-free initialization. Trajectory generation is formulated as minimizing a weighted sum of smoothness, collision, and feasibility costs:

$$
\arg \min _ {\mathbf {Q}} \mathcal {J} (\mathbf {Q}) = \lambda_ {s} \mathcal {J} _ {\mathrm{s}} (\mathbf {Q}) + \lambda_ {c} \mathcal {J} _ {\mathrm{c}} (\mathbf {Q}) + \lambda_ {f} \mathcal {J} _ {\mathrm{f}} (\mathbf {Q}),\tag{11}
$$

where $\mathcal { I } _ { \mathrm { s } }$ penalizes high-order derivatives for smooth motion, $\mathcal { T } _ { \mathrm { f } }$ penalizes violations of physical limits, and $\mathcal { T } _ { \mathrm { c } }$ enforces obstacle clearance. The nonnegative weights $\lambda _ { s } , \lambda _ { c } ,$ , and $\lambda _ { f }$ trade off smoothness, collision avoidance, and feasibility, respectively. The specific forms of $\mathcal { I } _ { \mathrm { s } }$ and $\mathcal { T } _ { \mathrm { f } }$ follow standard spline-optimization formulations and are referred to EGO-Planner [20]. The collision cost is detailed below.

For each colliding segment detected on $\Phi _ { \mathrm { i n i t } }$ , a collisionfree guiding path Γ is generated using $\mathbf { A } ^ { * } \left[ 2 1 \right]$ . Each control point $\mathbf { Q } _ { i }$ on the colliding segment is assigned $N _ { p } ( i )$ anchordirection pairs $\{ \mathbf { p } _ { i j } , \mathbf { v } _ { i j } \} _ { j = 1 } ^ { N _ { p } ( \check { i } ) }$ , where $i \in \mathbb { N } _ { + }$ indexes control points and $j \in \{ 1 , \dots , \check { N } _ { p } ( i ) \}$ indexes the pairs associated with $\mathbf { Q } _ { i }$ . For each pair, $\mathbf { v } _ { i j }$ is a unit repulsive direction that is orthogonal to the local spline tangent at $\mathbf { Q } _ { i }$ and oriented toward the guiding path Γ. The anchor point $\mathbf { p } _ { i j }$ is selected on the obstacle surface along the direction of $\mathbf { v } _ { i j } .$ , forming a signed distance $d _ { i j } = ( \mathbf { Q } _ { i } - \mathbf { p } _ { i j } ) ^ { \top } \mathbf { v } _ { i j }$ . A safety clearance is enforced by adopting a margin $s _ { m }$ beyond the end-effector sphere radius $r _ { \mathrm { e e } }$ . For each anchor-direction pair, we define the clearance violation relative to the signed distance $d _ { i j }$ and apply a twice continuously differentiable penalty:

$$
\begin{array}{r l} & {\phi (c _ {i j}) = \left\{ \begin{array}{l l} 0, & c _ {i j} \leq 0, \\ c _ {i j} ^ {3}, & 0 <   c _ {i j} \leq s _ {f}, \\ 3 s _ {f} c _ {i j} ^ {2} - 3 s _ {f} ^ {2} c _ {i j} + s _ {f} ^ {3}, & c _ {i j} > s _ {f}, \end{array} \right.} \\ & {\quad c _ {i j} = s _ {f} - d _ {i j}, \qquad s _ {f} = r _ {\mathrm{ee}} + s _ {m}.} \end{array}\tag{12}
$$

The collision cost for each control point $\mathbf { Q } _ { i }$ is accumulated over its associated pairs, and the overall collision cost is

$$
\mathcal {J} _ {\mathrm{c}} (\mathbf {Q}) = \sum_ {i = 1} ^ {N _ {c}} \sum_ {j = 1} ^ {N _ {p} (i)} \phi (c _ {i j}).\tag{13}
$$

The collision term induces gradients colinear with ${ \bf v } _ { i j } .$ deforming colliding control points along these directions to push the spline out of obstacles. The optimized trajectory $\Phi ^ { * } ( t )$ is then obtained by substituting the optimized control points $\mathbf { Q } ^ { * }$ from (11) into (10).

## E. Spatial-Temporal Trajectory Optimization

The discrete action sequence ${ \bf A } _ { t }$ can be obtained either from the diffusion policy or by sampling the collision-free trajectory $\Phi ^ { * } ( t )$ . It is treated as a set of action keypoints, and is refined by spatial-temporal trajectory optimization to reduce control effort while enforcing feasibility constraints.

Let $\Delta t$ denote the sampling period and define the trajectory time allocation as $T = T _ { a } \Delta t$ . A continuous-time action trajectory is parameterized as polynomial $\mathbf { A } ( t ) = \mathbf { c } ^ { \top } \boldsymbol { \beta } ( t )$ , where $\bar { \beta ( t ) } \overset { - } { = } [ 1 , t , \dotsc , t ^ { 2 s - 1 } ] ^ { \bar { \top } }$ is the time basis and c is a coefficient matrix. The refinement is formulated as

$$
\begin{array}{c} \min _ {\mathbf {c}} J (\mathbf {c}) = \lambda_ {E} J _ {E} + \lambda_ {T} J _ {T} + \lambda_ {P} J _ {P} \\ \text {s.t.} \mathbf {A} (k \Delta t) = \mathbf {a} _ {t + k}, \qquad k = 0, \ldots , T _ {a} - 1, \end{array}\tag{14}
$$

where $\lambda _ { E } , \lambda _ { T }$ , and $\lambda _ { P }$ are nonnegative weights that balance control effort, execution time, and feasibility, respectively. The control-effort term is

$$
J _ {E} = \int_ {0} ^ {T} \left\| \mathbf {A} ^ {(s)} (t) \right\| _ {2} ^ {2} d t,\tag{15}
$$

the time regularization term is $J _ { T } = T$ , and $J _ { P }$ penalizes violations of action limits over $t \in [ 0 , T ]$ , including elementwise bounds on action, velocity, and acceleration:

$$
\begin{array}{r l} & {\mathbf {A} _ {\mathrm{min}} \leq \mathbf {A} (t) \leq \mathbf {A} _ {\mathrm{max}},} \\ & {\dot {\mathbf {A}} _ {\mathrm{min}} \leq \dot {\mathbf {A}} (t) \leq \dot {\mathbf {A}} _ {\mathrm{max}},} \\ & {\ddot {\mathbf {A}} _ {\mathrm{min}} \leq \ddot {\mathbf {A}} (t) \leq \ddot {\mathbf {A}} _ {\mathrm{max}}.} \end{array}\tag{16}
$$

In practice, these inequality constraints are transcribed into a differentiable penalty and evaluated along the trajectory. The optimization is solved efficiently using MINCO [22], and the optimized trajectory is resampled for execution.

## IV. EXPERIMENTS

We design experiments to answer the following questions:

• Q1: Does latency-aware classifier-free guidance maintain inter-chunk consistency under temporally shifted future-action conditions?

• Q2: Does LAGO Policy produce smoother execution trajectories in real-world manipulation?

• Q3: Does goal-directed collision-free planning improve task completion in the presence of unseen obstacles?

## A. Experiment Setup

1) Hardware Platform: Experiments are conducted on two manipulation platforms: a 6-DoF ARX5 arm and a 7- DoF Franka arm. Each setup is equipped with two external

TABLE I  
REAL-WORLD EVALUATION

<table><tr><td></td><td colspan="4">Pick &amp; Place</td><td colspan="4">Pen Insertion</td><td colspan="4">Pouring</td><td colspan="4">Cup Transfer</td></tr><tr><td>Method</td><td>SR↑</td><td>ATR↓</td><td>CON↓</td><td>ISJ↓</td><td>SR↑</td><td>ATR↓</td><td>CON↓</td><td>ISJ↓</td><td>SR↑</td><td>ATR↓</td><td>CON↓</td><td>ISJ↓</td><td>SR↑</td><td>ATR↓</td><td>CON↓</td><td>ISJ↓</td></tr><tr><td>DP</td><td>0.85</td><td>16.91</td><td>0.034</td><td>27.29</td><td>0.70</td><td>8.76</td><td>0.0092</td><td>32.71</td><td>0.25</td><td>20.24</td><td>0.033</td><td>22.44</td><td>0.10</td><td>10.27</td><td>0.041</td><td>26.34</td></tr><tr><td>Ours</td><td>1.0</td><td>15.41</td><td>0.019</td><td>24.37</td><td>0.90</td><td>7.94</td><td>0.011</td><td>21.45</td><td>0.65</td><td>19.41</td><td>0.012</td><td>15.74</td><td>0.40</td><td>10.62</td><td>0.0083</td><td>10.11</td></tr><tr><td></td><td colspan="4">Towel Folding</td><td colspan="4">Box Organizing</td><td colspan="4">Tape Hanging</td><td colspan="4">Screw Sorting</td></tr><tr><td>Method</td><td>SR↑</td><td>ATR↓</td><td>CON↓</td><td>ISJ↓</td><td>SR↑</td><td>ATR↓</td><td>CON↓</td><td>ISJ↓</td><td>SR↑</td><td>ATR↓</td><td>CON↓</td><td>ISJ↓</td><td>SR↑</td><td>ATR↓</td><td>CON↓</td><td>ISJ↓</td></tr><tr><td>DP</td><td>0.35</td><td>18.24</td><td>0.058</td><td>33.11</td><td>0.60</td><td>28.13</td><td>0.072</td><td>96.10</td><td>0.80</td><td>32.82</td><td>0.0066</td><td>115.51</td><td>0.60</td><td>53.50</td><td>0.0062</td><td>116.67</td></tr><tr><td>Ours</td><td>0.70</td><td>14.27</td><td>0.017</td><td>15.25</td><td>0.85</td><td>27.31</td><td>0.026</td><td>29.07</td><td>0.90</td><td>36.80</td><td>0.0059</td><td>21.16</td><td>0.95</td><td>52.97</td><td>0.0037</td><td>18.77</td></tr></table>

![](images/31ae061a4559e084aa25fc3364fd6a7173ed2c01375cae2edca54ea7742adc9f.jpg)  
Fig. 5. Real-world task setups. Tape Hanging and Screw Sorting are evaluated on the Franka arm, while the remaining tasks are evaluated on ARX5.

Intel RealSense D435 / L515 RGB-D cameras that provide workspace-scale coverage of the manipulation area, and a wrist-mounted fisheye camera with a 180<sup>◦</sup> field of view that captures close-up observations for fine manipulation. All devices are connected to a workstation with an Intel Core i7- 14700 CPU and an NVIDIA RTX 4070 SUPER GPU, which is used for both data collection and policy evaluation.

2) Task: As illustrated in Fig. 5, we evaluate our method on eight real-world manipulation tasks spanning pick-andplace, insertion, transport of liquid-filled objects, deformable object manipulation, and articulated-object manipulation. These tasks require accurate grasping and end-effector pose alignment, stable contact-rich control, and smooth low-jerk execution. For example, Cup Transfer requires the robot to transport a tea-filled cup and place it onto a coaster without spilling, while Pouring requires the gripper to precisely thread through the teapot handle and maintain stable motion during pouring. Additional task details and full execution sequences are provided in the supplementary video.

3) Expert Demonstrations: Expert demonstrations for all tasks are collected via human teleoperation. The ARX5 arm is controlled using a SpaceMouse, whereas the Franka arm is controlled via keyboard teleoperation. For each task, we record 50 demonstrations, each consisting of multiview RGB images, end-effector poses, gripper states, and teleoperation commands.

4) Implementation Details: We follow the Diffusion Policy setup with an action prediction horizon $T _ { p } ~ = ~ 1 6 ,$ an observation horizon $T _ { o } = 2 .$ , and an action execution horizon $T _ { a } = 8$ . Following SAIL, we set the future-action horizon to $T _ { f } = 4$ . We use a DDIM noise scheduler with 100 diffusion steps during training and 8 denoising steps at inference.

Models are optimized with AdamW using an initial learning rate of $1 \times 1 0 ^ { - 4 }$ and a cosine decay schedule. We train the model for 350 epochs with a batch size of 64 for each task.

5) Evaluation Metrics: Our evaluation uses metrics that capture both task-level performance and motion quality. Task performance and efficiency are measured by the success rate (SR) and the average time of successful rollouts (ATR). Motion quality is evaluated by inter-chunk consistency (CON) and trajectory smoothness. $\mathbf { C O N } = | | \mathbf { A } _ { t } ^ { c } - \mathbf { A } _ { t } ^ { 0 : T _ { f } - 1 } | |$ measures the discrepancy between the overlapping parts of consecutive action chunks at the chunk boundary, where ${ \bf A } _ { t } ^ { 0 : T _ { f } - 1 }$ denotes the first $T _ { f }$ actions of the current chunk $\mathbf { A } _ { t } .$ Smoothness is quantified by the integrated squared jerk (ISJ) of the executed action trajectory: $\begin{array} { r } { \mathbf { I } \mathbf { \bar { S } J } = \int _ { 0 } ^ { T _ { \mathrm { e x e c } } } | | \mathbf { \bar { \dot { A } } } ( t ) | | _ { 2 } ^ { 2 } d t } \end{array}$ where $T _ { \mathrm { e x e c } }$ is the execution duration of the rollout.

## B. Robustness to Future-Action Condition Shifts

Latency-aware CFG is robust to temporally shifted future-action conditions (Q1). We evaluate robustness to future-action condition shifts, which can arise from asynchronous inference latency and tracking error, by comparing three variants: (i) SAIL, which concatenates ${ \bf A } _ { t } ^ { c }$ with the observation and replicates it across the $T _ { o } .$ -step conditioning sequence; (ii) OURS-CFG, which injects ${ \bf A } _ { t } ^ { c }$ as a separate conditioning variable via CFG during sampling but trains without delay randomization; and (iii) OURS-LA-CFG, which additionally randomizes the temporal offset of ${ \bf A } _ { t } ^ { c }$ during training. Experiments are conducted on Box Organizing task, which is sensitive to inter-chunk continuity. We set the guidance scale to $w = 1$ , run 20 rollouts per setting, and report SR and CON under simulated futureaction condition shifts of 0–4 steps.

![](images/33c661cdb29e92e27b4d4c6bc78ee8d5a22735341782fd6993a84ba0e2899e99.jpg)  
Fig. 6. SR and CON under simulated future-action shifts δ. As the shift increases, OURS-LA-CFG maintains a higher success rate and exhibits less inter-chunk discontinuity.

The results are shown in Fig. 6. We first compare SAIL and OURS-CFG. As the future-action condition becomes increasingly shifted, both methods exhibit degraded task performance and motion quality, with SR decreasing and CON increasing. However, OURS-CFG achieves higher SR and a smaller increase in CON than SAIL, indicating improved robustness and better inter-chunk continuity under misaligned future-action conditions. This benefit is attributed to the conditioning interface. In SAIL, the shifted futureaction condition is embedded into the observation history, leading to repeated exposure to a misaligned signal across $T _ { o }$ timesteps. In contrast, OURS-CFG injects A<sup>c</sup> only once through CFG during sampling, which mitigates the adverse effect of temporally shifted conditioning on denoising and action generation. We then compare OURS-CFG and OURS-LA-CFG. In contrast to the monotonic degradation observed above, OURS-LA-CFG remains markedly more stable as the shift increases, maintaining a higher SR and exhibiting a smaller change in CON. These results show that delay randomization during training further improves robustness to temporally shifted future-action conditions at deployment, yielding more consistent inter-chunk execution.

## C. Smooth Execution in Real-World Manipulation

LAGO Policy produces smoother real-world execution (Q2). We compare motion quality between Diffusion Policy and LAGO Policy with its goal-directed planning module disabled on eight real-world manipulation tasks, as shown in Fig. 5. We deploy the trained models for 20 rollouts per task and log robot states during execution. The results in Table I show that LAGO Policy achieves lower CON on seven tasks and lower ISJ on all tasks, indicating improved inter-chunk continuity and smoother execution trajectories.

Qualitatively, LAGO Policy is more robust to action ambiguity induced by similar observations across consecutive cycles. For example, in Screw Sorting, opening and closing the sliding drawer often yields near-identical visual states, under which Diffusion Policy may hesitate between modes and produce stop-and-go motions. Conditioning with futureaction guidance encourages temporally consistent chunk transitions and yields continuous motion. Moreover, for contact-rich tasks where stable gripper behavior is critical, such as Towel Folding and Pouring, we observe that Diffusion Policy can generate inconsistent gripper commands after grasping, occasionally opening the gripper during transport and causing object drops or spills. LAGO Policy improves gripper command continuity, leading to more reliable contact maintenance and higher rollout success.

![](images/2fa98cfb5e419769a73109dffb0b73b4d3c43682b74f55ee62aa6e95de90c578.jpg)  
Fig. 7. Obstacles absent from the training demonstrations are added to four tasks and highlighted in yellow. Our GOAL-DIRECTED method plans smooth collision-free motion from the start to the predicted goal.

To further assess the contribution of spatial-temporal trajectory optimization to motion quality, we conduct an ablation on Box Organizing by removing the optimization module from LAGO Policy. Without this module, CON increases to 0.041 and ISJ increases to 40.03, indicating degraded inter-chunk continuity and higher jerk in the executed trajectory. These results suggest that spatial-temporal trajectory optimization improves trajectory smoothness, which is particularly beneficial for tasks requiring low-jerk motions.

## D. Goal-Directed Obstacle Avoidance and Task Success

Goal-directed collision-free planning improves task completion under unseen obstacles (Q3). As shown in Fig. 7, we introduce unseen obstacles that are absent from the expert demonstrations in four tasks and evaluate task success rate by comparing three variants: (i) NO-AVOID, which disables the collision-free trajectory generation module and directly executes the diffusion policy outputs; (ii) LOCAL, which replaces collision-free trajectory generation with a local safety filter; and (iii) GOAL-DIRECTED, our full method with goal prediction and goal-conditioned collisionfree planning. The LOCAL baseline performs collision checking on the action chunk predicted by diffusion policy at each inference cycle. When a collision is detected, the colliding actions are projected to the nearest collision-free end-effector states in task space, and $\mathbf { A } ^ { * }$ is used to generate a short collision-free detour for execution. This setup follows a common local safety filtering paradigm that enforces collision avoidance by minimally modifying the policy output [15].

TABLE II  
SUCCESS RATE ON FOUR REAL-WORLD MANIPULATION TASKS.

<table><tr><td>Method</td><td>Box</td><td>Cup</td><td>Pouring</td><td>Pen</td></tr><tr><td>NO-AVOID</td><td>0.40</td><td>0.0</td><td>0.0</td><td>0.60</td></tr><tr><td>LOCAL</td><td>0.60</td><td>0.20</td><td>0.40</td><td>0.80</td></tr><tr><td>GOAL-DIRECTED</td><td>1.0</td><td>0.80</td><td>0.80</td><td>1.0</td></tr></table>

![](images/cbed15e2be110c4ae214ed1abaa546bc7c99b41f5bdd1a487c537e20a508c307.jpg)  
Task Completion Time: 53.04 s

![](images/eaaf7028169cb302499550c822961d33daa6fd2a386f40dd83d50e449f355034.jpg)  
Task Completion Time: 27.12 s  
Fig. 8. Effect of goal-directed motion planning. (a) Without goal-directed motion planning, the motion remains piecewise and inefficient over long distances. (b) With goal-directed motion planning, the robot reaches the task-relevant goal along a smooth direct trajectory.

Each method is evaluated with five independent rollouts per task, and the resulting success rates are reported in Table II. Across all tasks, NO-AVOID frequently collides with the introduced obstacles, leading to low success rates. This effect is particularly pronounced in Cup Transfer, where even slight contact perturbs the liquid and causes failure. While LOCAL reduces direct collisions, its short-horizon corrections often push the robot into out-of-distribution states relative to the demonstrations, inducing unstable behaviors such as abrupt gripper commands and jerk motions that hinder task completion. For example, in Pouring, an upward reactive avoidance maneuver induces jerky corrections and an unintended gripper opening, causing the teapot to tip over. In contrast, GOAL-DIRECTED generates a smooth collisionfree trajectory toward the predicted task goal and then hands control back to the policy for the final interaction, thereby avoiding the failure modes induced by reactive short-horizon corrections in LOCAL and achieving the highest success rates under unseen obstacles.

## E. Goal-Directed Planning for Long-Range Motion

We further study the effect of goal-directed planning on long-range motion efficiency. On Franka, demonstrations are collected via keyboard teleoperation, which often produces axis-aligned, piecewise motions. The learned policy consequently reproduces such trajectories, resulting in long travel distance and increased task completion time. Goal-directed planning instead generates a smooth direct trajectory toward the task-relevant goal, substantially reducing task completion time, as shown in Fig. 8 and in the supplementary video.

## V. CONCLUSION

We presented LAGO Policy, a unified framework for smooth and safe diffusion-based manipulation under asynchronous inference. By combining latency-aware classifierfree guidance, goal-directed trajectory generation, and spatial-temporal trajectory optimization, LAGO Policy improves inter-chunk consistency, reduces jerk, and enables collision-free execution toward task-relevant goals. Experiments on real-world manipulation tasks demonstrate smoother execution and improved task success in cluttered environments. For future work, we will extend the current task-space collision handling to whole-body collision-aware motion planning, and improve task-relevant goal prediction in more complex manipulation settings.

## REFERENCES

[1] T. Z. Zhao et al., “Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware,” arXiv preprint arXiv:2304.13705, 2023.

[2] C. Pan et al., “Much Ado About Noising: Dispelling the Myths of Generative Robotic Control,” arXiv preprint arXiv:2512.01809, 2026.

[3] Y. Ze et al., “3D Diffusion Policy: Generalizable Visuomotor Policy Learning via Simple 3D Representations,” arXiv preprint arXiv:2403.03954, 2024.

[4] D. Wang et al., “Hierarchical Diffusion Policy: Manipulation Trajectory Generation via Contact Guidance,” IEEE Transactions on Robotics, vol. 41, pp. 2086–2104, 2025.

[5] S. H. Høeg, Y. Du, and O. Egeland, “Fast Policy Synthesis with Variable Noise Diffusion Models,” in 2025 IEEE International Conference on Robotics and Automation (ICRA), 2025, pp. 4821–4828.

[6] N. R. Arachchige et al., “SAIL: Faster-than-Demonstration Execution of Imitation Learning Policies,” in Conference on Robot Learning. PMLR, 2025, pp. 721–749.

[7] J. Ho and T. Salimans, “Classifier-Free Diffusion Guidance,” arXiv preprint arXiv:2207.12598, 2022.

[8] J. Ho, A. Jain, and P. Abbeel, “Denoising Diffusion Probabilistic Models,” Advances in neural information processing systems, vol. 33, pp. 6840–6851, 2020.

[9] M. Janner et al., “Planning with Diffusion for Flexible Behavior Synthesis,” in Proceedings of the 39th International Conference on Machine Learning, vol. 162. PMLR, 2022, pp. 9902–9915.

[10] C. Chi et al., “Diffusion Policy: Visuomotor Policy Learning via Action Diffusion,” The International Journal of Robotics Research, vol. 44, no. 10-11, pp. 1684–1704, 2025.

[11] J. Tang et al., “VLASH: Real-Time VLAs via Future-State-Aware Asynchronous Inference,” arXiv preprint arXiv:2512.01031, 2025.

[12] W. Peebles and S. Xie, “Scalable Diffusion Models with Transformers,” in 2023 IEEE/CVF International Conference on Computer Vision (ICCV), 2023, pp. 4172–4182.

[13] K. Black et al., “Training-Time Action Conditioning for Efficient Real-Time Chunking,” arXiv preprint arXiv:2512.05964, 2025.

[14] H. Li et al., “Language-Guided Object-Centric Diffusion Policy for Generalizable and Collision-Aware Manipulation,” in 2025 IEEE International Conference on Robotics and Automation (ICRA), 2025, pp. 12834-12841.

[15] S. Hu et al., “VLSA: Vision-Language-Action Models with Plug-and-Play Safety Constraint Layer,” arXiv preprint arXiv:2512.11891, 2025.

[16] W. Jung et al., “RAIL: Reachability-Aided Imitation Learning for Safe Policy Execution,” in 2025 IEEE International Conference on Robotics and Automation (ICRA), 2025, pp. 3582–3589.

[17] E. Perez et al., “FiLM: Visual Reasoning with a General Conditioning Layer,” in Proceedings of the AAAI conference on artificial intelligence, vol. 32, no. 1, 2018.

[18] J. Song, C. Meng, and S. Ermon, “Denoising Diffusion Implicit Models,” arXiv preprint arXiv:2010.02502, 2020.

[19] O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional Networks for Biomedical Image Segmentation,” in International Conference on Medical image computing and computer-assisted intervention. Springer, 2015, pp. 234–241.

[20] X. Zhou et al., “EGO-Planner: An ESDF-Free Gradient-Based Local Planner for Quadrotors,” IEEE Robotics and Automation Letters, vol. 6, no. 2, pp. 478–485, 2020.

[21] P. E. Hart, N. J. Nilsson, and B. Raphael, “A Formal Basis for the Heuristic Determination of Minimum Cost Paths,” IEEE Transactions on Systems Science and Cybernetics, vol. 4, no. 2, pp. 100–107, 1968.

[22] Z. Wang et al., “Geometrically Constrained Trajectory Optimization for Multicopters,” IEEE Transactions on Robotics, vol. 38, no. 5, pp. 3259–3278, 2022.