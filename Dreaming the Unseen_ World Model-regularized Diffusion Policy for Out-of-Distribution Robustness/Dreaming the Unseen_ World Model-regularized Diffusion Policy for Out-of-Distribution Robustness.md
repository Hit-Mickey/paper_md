# Dreaming the Unseen: World Model-regularized Difusion Policy for Out-of-Distribution Robustness

Ziou Hu<sup>1⋆</sup>, Xiangtong Yao<sup>1⋆</sup>, Yuan Meng<sup>1</sup>, Zhenshan Bing<sup>2,1</sup>, and Alois Knoll<sup>1</sup>

<sup>1</sup> School of Computation, Information and Technology, Technical University of Munich, Garching, Germany

<sup>2</sup> State Key Laboratory for Novel Software Technology and the School of Science and Technology, Nanjing University (Suzhou Campus), China

Abstract. Difusion policies excel at visuomotor control but often fail catastrophically under severe out-of-distribution (OOD) disturbances, such as unexpected object displacements or visual corruptions. To address this vulnerability, we introduce the Dream Difusion Policy (DDP), a framework that deeply integrates a difusion world model into the policy’s training objective via a shared 3D visual encoder. This co-optimization endows the policy with robust state-prediction capabilities. When encountering sudden OOD anomalies during inference, DDP detects the real-imagination discrepancy and actively abandons the corrupted visual stream. Instead, it relies on its internal “imagination”—autoregressively forecasted latent dynamics—to safely bypass the disruption, generating imagined trajectories before smoothly realigning with physical reality. Extensive evaluations demonstrate DDP’s exceptional resilience. Notably, DDP achieves a 73.8% OOD success rate on MetaWorld (vs. 23.9% without predictive imagination) and an 83.3% success rate under severe real-world spatial shifts (vs. 3.3% without predictive imagination). Furthermore, as a stress test, DDP maintains a 76.7% real-world success rate even when relying entirely on open-loop imagination post-initialization.

Keywords: World Model · Difusion Policy · Out-of-Distribution Robustness

## 1 Introduction

Robotic manipulation in unstructured environments demands policies that can transform rich visual observations into precise, temporally extended motor commands. Recent visuomotor methods, like difusion-based policies, have emerged as a compelling paradigm for this setting: instead of regressing a single action, they model action generation as a conditional denoising process [7,20,57] or via flow matching [58]. These policies have demonstrated remarkable dexterity and robustness in intricate, high-dimensional robot manipulation tasks [7,19,37,47].

However, standard visuomotor imitation learning remains fundamentally brittle due to its strict reliance on continuous, uncorrupted visual streams. In realworld deployments, dynamic disturbances (e.g., severe object displacements or camera occlusions) instantly push observations into out-of-distribution (OOD) states, and without explicit physical reasoning, standard policies cannot recover from this massive covariate shift, inevitably leading to compounding errors and catastrophic failure [12,43,61]. Existing methods attempt to mitigate these shifts through various strategies, including domain randomization [38], test-time adaptation [39, 53, 54], explicit expansion of training datasets with synthetic spatial perturbations [59], or the application of large pretrained visual models [11]. Other approaches enhance representational and physical priors [1, 21, 40, 45], or learn explicit inverse recovery policies [12,36]. While somewhat efective, these mitigations typically introduce highly complex architectural pipelines and aggressively alter expert-derived actions during adaptation. This fundamentally compromises the core objective of imitation learning, as the high-fidelity skills carefully extracted from experts are ultimately overwritten rather than preserved.

To overcome these limitations, we rethink how agents handle corrupted visual states by drawing inspiration from human motor mastery. When environmental visual cues become chaotic or deceptive, the optimal human strategy is often to efectively close one’s eyes and rely on internal kinesthetic intuition, guided by a single, unwavering focal point. We translate this philosophy into robotic continuous control, arguing that a difusion policy can be made substantially more resilient by endowing it with a built-in predictive capability.

The natural technical counterpart to this internal intuition is a world model. Traditionally, these models predict future latent states to support planning [15], enable highly sample-eficient learning [35], and equip agents with crucial predictive reasoning [18]. Difusion-based world models have emerged as a powerful tool to flexibly forecast complex physical dynamics [3, 6], making them ideal candidates to provide predictive intuition. Yet, current methods typically treat them as weakly coupled auxiliary modules. Lacking a mechanism to dynamically substitute corrupted vision with imagined latents, these policies remain fatally tethered to deceptive observations and collapse under severe OOD shifts. To sustain control, the world model must transcend passive regularization and become a tightly integrated, active predictive intuition.

We propose the Dream Difusion Policy (DDP) to tackle severe out-ofdistribution (OOD) disturbances. Rather than passively relying on vulnerable visual streams, DDP tightly couples a difusion policy [7] with a predictive world model [10] via a shared 3D visual encoder. As the policy learns standard behavior cloning [7], the world model simultaneously predicts future observation latents. This shared representation captures robust geometric priors, empowering the world model to actively assist when physical observations become unreliable. During inference, DDP dynamically shifts between physical reality and internal “imagination.” By tracking targets and executing core actions entirely within its hallucinated latent space, the robot safely bypasses corrupted vision and physically recovers the environment to an In-Distribution (ID) state.

We extensively validate DDP across 10 MetaWorld [56] tasks, 3 complex Adroit [31] tasks, and 3 real-world Franka Panda tasks. While Adroit evaluations confirm our world model’s predictive stability in high-DoF continuous control, MetaWorld and real-robot experiments demonstrate DDP’s exceptional resilience to severe OOD disruptions—such as sudden displacements and complete visual occlusions. Notably, DDP achieves a 73.8% OOD success rate on MetaWorld (vs. 23.9% for a tracking-augmented baseline) and an 83.3% success rate under severe real-world spatial shifts (vs. 3.3%). Furthermore, as a stress test, DDP maintains a remarkable 76.7% real-world success rate even when relying entirely on open-loop imagination post-initialization. In summary, our primary contributions are:

– World-Model-Regularized Training Pipeline: We propose the Dream Difusion Policy, which co-optimizes behavior cloning and world-model objectives via a shared 3D encoder. This unified representation allows the world model’s future predictions to directly regularize and guide policy inference.

– Robust Inference via Predictive Imagination: We design a novel, subtask-aware execution mechanism that actively detects OOD states via a highly intuitive real-imagination discrepancy $\left( \mathcal { D } _ { R - I } \right)$ , bypasses corrupted vision using autoregressive imagination, and triggers a localized physical recovery to the ID state to resume normal execution.

– State-of-the-Art OOD Robustness: We demonstrate that DDP significantly outperforms state-of-the-art baselines under severe OOD conditions across both simulated and real-world tasks, maintaining exceptional resilience even when operating completely blind after initial observation.

## 2 Related Works

Difusion Policy for Robotic Manipulation. By framing action generation as conditional denoising, difusion models [20] efectively capture complex, multimodal distributions in high-dimensional continuous control [7,57]. These policies achieve state-of-the-art results across diverse applications, including kinematicsaware [27], contact-rich [41, 48], and cross-manipulator tasks [34, 54]. Additionally, mechanisms like receding-horizon control and stochastic Langevin dynamics provide inherent robustness against minor physical perturbations and transient visual distractions in real-world deployments [7, 48].

World Models for Decision Making and Robotics. World models learn internal representations to predict environmental dynamics and guide decisionmaking [9]. Pioneered by the Dreamer series [16–18], agents can master complex continuous control entirely within a learned latent imagination. This paradigm is now advancing embodied AI: models like NVIDIA Cosmos [2] enforce strict physical laws as “physical AI” brains, while architectures like DreamZero [55] jointly forecast future states and motor actions. Ultimately, integrating predictive world models directly into control policies provides the crucial “imagination” required for robust robotic manipulation.

OOD Challenges and Mitigation Strategies. Deployed policies inevitably face Out-of-Distribution (OOD) shifts. Standard mitigations expand training diversity via domain randomization [38], synthetic trajectories [21], or reinforcement learning (RL) [40]. However, these methods fail under severe conditions—like camera occlusions or target shifts—because they assume a continuous visual stream. Without visual evidence, policies sufer from shortcut learning [13], fatally over-relying on proprioception for spatial localization. Furthermore, RL fine-tuning [6, 8] cannot reconstruct missing visual data at inference, while heavy online pipelines [11] and video-difusion reward models [22] are computationally prohibitive.

While world models [15] and difusion variants [10] forecast states to handle partial observability, current integrations treat them as weakly coupled auxiliary modules that still collapse under severe shifts. To overcome this, we embed lightweight predictive capabilities directly into the policy. By co-optimizing a difusion policy and world model via a shared-encoder regularizer, we avoid complex decoupled architectures and expensive RL fine-tuning [6]. This tight coupling empowers the policy to seamlessly “dream” missing visual context, sustaining robust control even during severe sensor dropouts.

## 3 Methods

Building on DP3 [57], our Dream Difusion Policy (DDP) utilizes a shared 3D encoder to extract unified geometric latent embeddings from point clouds and robot proprioception. These embeddings jointly condition both the Difusion Policy and the Difusion World Model. During training, the World Model loss acts as a regularizer for the policy, embedding environmental kinematics and dynamics into the shared representation. During inference, this coupled architecture empowers the agent to use “Imagination”—forecasting future observations—to sustain robust execution under Out-of-Distribution scenarios, as outlined in Fig. 1.

## 3.1 Action and Latent Embedding Chunking

Action chunking acts as policy compression in human motor control [24]. In robotic imitation learning, it better captures temporal dependencies [26] and mitigates compounding errors common in strictly autoregressive models [5]. Motivated by this, we synchronize our policy and world model via explicit chunking:

Action Chunking: Our Difusion Policy denoises a full trajectory over horizon H. To balance long-term consistency and local reactivity, the agent executes only a sub-sequence “chunk” of length N.

Latent Embedding Observation Chunking: To enable forward-looking “imagination”, the World Model operates over the same chunk. Given M historical observations and the N-step planned action chunk, it predicts N future latent states in parallel. For continuous OOD rollouts, we apply a sliding window: the final M predicted latents become the historical condition for the next cycle, tightly aligning planned actions with expected outcomes.

![](images/eb83303e4b7450a320fc5a27f9055459465ffd72c536af80f37b164ddc29c557.jpg)  
Fig. 1: Overview of the Dream Difusion Policy (DDP) framework. (a) A shared 3D encoder extracts unified latent embeddings from point clouds and proprioception. These history latents jointly condition the Difusion Policy to denoise action chunks, and the Difusion World Model (via channel-wise concatenation) to forecast future latent states. (b) During inference stage, upon detecting an Out-of-Distribution anomaly, the system halts, tracks the object, and transitions into an imagination loop. The World Model autoregressively forecasts latent states to drive the policy’s action generation, safely bypassing corrupted observations until the recovery is triggered.

## 3.2 Difusion Policy

Building upon standard DDPM [20], our difusion policy $\pi _ { \theta }$ formulates action generation as a conditional denoising task, predicting an action sequence $\mathbf { a } _ { 0 \ldots H - 1 }$ over horizon H. Following DP3 [57], we utilize a shared 3D encoder $E _ { \psi }$ to process raw historical observations $\mathbf { X } _ { 0 \ldots M - 1 }$ . This encoder extracts spatial features from geometric point clouds via PointNet and encodes proprioceptive states (e.g., end-efector pose) via an MLP, concatenating both streams to yield the unified context $\mathbf { O } _ { 0 \ldots M - 1 }$ . During the forward difusion process, Gaussian noise $\epsilon _ { p } \sim$ $\mathcal { N } ( 0 , \bf { I } )$ is added to the ground-truth actions to generate a noisy trajectory $\mathbf { a } ^ { k _ { p } }$ at a randomly sampled timestep $k _ { p } \sim \mathcal { U } ( 1 , K )$

The core of the policy is a 1D Conditional U-Net. The encoded history context $\mathbf { O } _ { 0 \ldots M - 1 }$ serves as a global condition. It is injected into the intermediate residual blocks of the U-Net via Feature-wise Linear Modulation (FiLM) [30], allowing the visual state to guide the action denoising process efectively. The objective function for the policy is standard Behavior Cloning (BC) via noise matching:

$$
\mathcal {L} _ {B C} = \mathbb {E} _ {\mathbf {a}, \mathbf {X}, \epsilon_ {p}, k _ {p}} \left[ \left\| \epsilon_ {p} - \pi_ {\theta} \left(\mathbf {a} _ {0 \dots H - 1} ^ {k _ {p}}, k _ {p}, \mathrm{FiLM} (\mathbf {O} _ {0 \dots M - 1})\right) \right\| _ {2} ^ {2} \right]\tag{1}
$$

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1: Dream Diffusion Policy Training
Input: Dataset D, Max steps K, 3D Encoder  $E_{\psi}$ , Policy  $\pi_{\theta}$ , World Model  $W_{\phi}$ 
1 while not converged do
// 1. Sample Trajectories(M: observation steps, N: action steps, H: horizon)
2 Sample ( $X_{0...M+N-1}, a_{0...H-1}$ ) ~ D;
// 2. Encode Observations
3  $O_{0...M-1} \leftarrow E_{\psi}(X_{0...M-1})$ ;
4  $O_{M...M+N-1} \leftarrow E_{\psi}(X_{M...M+N-1})$ ;
// 3. Diffusion Policy Loss
5  $k_{p} \sim \mathcal{U}(1, K), \epsilon_{p} \sim \mathcal{N}(0, \mathbf{I})$ ;
6  $a_{0...H-1}^{k_{p}} \leftarrow \text{AddNoise}(a_{0...H-1}, \epsilon_{p}, k_{p})$ ;
7  $L_{BC} \leftarrow \text{MSE}\left(\epsilon_{p}, \pi_{\theta}(a_{0...H-1}^{k_{p}}, k_{p}, \text{FiLM}(O_{0...M-1}))\right)$ ;
// 4. Diffusion World Model Loss
8  $k_{w} \sim \mathcal{U}(1, K), \epsilon_{w} \sim \mathcal{N}(0, \mathbf{I})$ ;
9  $O_{M...M+N-1}^{k_{w}} \leftarrow \text{AddNoise}(O_{M...M+N-1}, \epsilon_{w}, k_{w})$ ;
10  $C_{wm} \leftarrow \text{cat}(O_{0...M-1}, O_{M...M+N-1}^{k_{w}})$ ;
11  $L_{WM} \leftarrow \text{MSE}\left(\epsilon_{w}, W_{\phi}(C_{wm}, k_{w}, \text{FiLM}(a_{M-1...M+N-2}))\right)$ ;
// 5. Optimization
12  $L_{total} \leftarrow L_{BC} + \lambda L_{WM}$ ;
13 Update  $\psi, \theta, \phi$  using  $\nabla_{\psi, \theta, \phi} L_{total}$ ;
</div>

## 3.3 Difusion World Model

Based on the framework in [10], our Difusion World Model $\mathcal { W } _ { \phi }$ predicts Nstep future latent embeddings $\mathbf { O } _ { M \dots M + N - 1 }$ . During training, the shared 3D encoder $E _ { \psi }$ extracts these future targets, which are then corrupted with noise $\epsilon _ { w }$ at timestep $k _ { w }$ to yield $\mathbf { O } _ { M \ldots M + N - 1 } ^ { k _ { w } }$ . Because actions (directional controls) and observations (geometric context) possess fundamentally distinct properties, we deviate from the sampling-based “prefix-fixing” conditioning used in [10]. We argue that fixing initial states during reverse difusion lacks the persistent grounding required for high-dimensional point cloud latents. Instead, we employ direct feature-level integration [32]. By repeating the historical latent $\mathbf { x } _ { o b s }$ across the temporal horizon and concatenating it channel-wise with the noisy samples, we formulate the input as:

$$
\mathbf {X} _ {i n} = \mathrm{Repeat} (\mathbf {x} _ {o b s}, N) \oplus \mathbf {X} _ {s a m p l e}\tag{2}
$$

This ensures every convolutional layer in the 1D U-Net has immediate, continuous access to the historical structural context.

Concurrently, the executed action sequence a $\cdot { \cal M } - 1 . . . { \cal M } + N - 2$ serves as a global condition. As in the policy architecture, these action representations are fused with the difusion timestep embeddings and injected via FiLM. The world model is optimized by minimizing the noise prediction error:

$$
\mathcal {L} _ {W M} = \mathbb {E} _ {\mathbf {O}, \mathbf {a}, \epsilon_ {w}, k _ {w}} \left[ \| \epsilon_ {w} - \mathcal {W} _ {\phi} \left(\mathbf {C} _ {w m}, k _ {w}, \mathrm{FiLM} (\mathbf {a} _ {M - 1 \dots M + N - 2})\right) \| _ {2} ^ {2} \right]\tag{3}
$$

## 3.4 Training Objective

We jointly optimize the Difusion Policy and World Model end-to-end. Using normalized expert demonstrations, the shared encoder $E _ { \psi }$ extracts latent embeddings for both historical context and future targets. The policy $\pi \theta$ learns to predict Gaussian noise added to the action horizon $\mathbf { a } _ { 0 \ldots H - 1 }$ conditioned on the history $\mathbf { O } _ { 0 \ldots M - 1 }$ . Simultaneously, the world model $\mathcal { W } _ { \phi }$ predicts the noise injected into future observation latents $\mathbf { O } _ { M \dots M + N - 1 }$ , conditioned on the history and executed actions. Following established regularization strategies [1,21], the world model acts as a geometric and dynamic regularizer. We update the network parameters $( \psi , \theta , \phi )$ via gradient descent using a combined loss:

$$
\mathcal {L} _ {t o t a l} = \mathcal {L} _ {B C} + \lambda \mathcal {L} _ {W M}\tag{4}
$$

where λ balances the two objectives. Finally, we apply an Exponential Moving Average (EMA) [29] to the model weights to stabilize convergence and enhance inference robustness (Algorithm 1).

## 3.5 Inference Stage

During inference, dynamic object displacements break the In-Distribution (ID) environment, triggering an Out-of-Distribution (OOD) state. To maintain spatial awareness during these disruptions, we integrate a 6D pose estimator (e.g., FoundationPose [44]). By combining this estimator with the World Model’s predictive “imagination,” we introduce a hybrid mechanism of detection, recursive imagination, and physical recovery to safely bridge OOD gaps, as illustrated in the closed-loop workflow of Fig. 2. The pseudocode for closed-Loop inference is provided in Algorithm 2 in Appendix.

Subtask Definition For long-horizon manipulation, we define a subtask sequence $\boldsymbol { S } = \{ S _ { 1 } , \ldots , S _ { J } \}$ , each targeting an object $C _ { j }$ . Lacking explicit dataset annotations, we bypass complex clustering and use simple expert heuristics for completion triggers (e.g., gripper closure for picking, allocated time for pouring). This flexibility allows DDP to solve complex, multi-stage challenges.

Real-Imagination Discrepancy as OOD Detector. Unlike methods relying on reconstruction errors [14, 23] or difusion loss [25], we take a highly intuitive approach: detecting OOD anomalies via the latent discrepancy between the real observation ${ \bf O } _ { r e a l } ^ { t } = E _ { \psi } ( X _ { r e a l } ^ { t } )$ and the World Model’s prediction $\mathbf { O } _ { p r e d } ^ { t } \mathrm { : \quad }$

$$
\mathcal {D} _ {R - I} (t) = \| \mathbf {O} _ {r e a l} ^ {t} - \mathbf {O} _ {p r e d} ^ {t} \| _ {2} ^ {2}\tag{5}
$$

Nominal actions maintain a low $\mathcal { D } _ { R - I }$ . When an external disturbance pushes $\mathcal { D } _ { R - I } ( t )$ above a threshold $\tau _ { d i f f }$ , an OOD event is flagged. To identify which objects moved, we query the pose estimator, comparing current poses to expected ID states to isolate the displaced objects $\mathcal { C } _ { d i s p } .$

Simplified Object Tracking under OOD. Rather than relying on continuous tracking during chaotic displacements, we assume the displaced object $C _ { j }$ eventually settles into a static pose. Once motion ceases, we query the 6D pose estimator for its new resting pose $\mathbf { p } _ { C _ { j } } \in S E ( 3 )$ . By comparing this to its expected ID pose ${ \bf p } _ { C _ { j } } ^ { e x p }$ , we compute a single static spatial displacement:

![](images/7ccfa7bba0371d22ccebdd9f040612a5d2efaff1408a4e68b7d6368c8c894c1b.jpg)  
Fig. 2: Overview of the DDP workflow. Illustrating the closed-loop transitions between physical reality (ID) and internal imagination (OOD) across subtasks.

$$
\Delta \mathbf {p} _ {C _ {j}} = \mathbf {p} _ {C _ {j}} - \mathbf {p} _ {C _ {j}} ^ {e x p}\tag{6}
$$

To re-align the robot with the displaced target, we directly apply this ofset as a compensatory tracking action for the end-efector, defined as $\mathbf { a } _ { j } ^ { t r a c k } = \varDelta \mathbf { p } _ { C _ { j } }$ Recursive Imagination for Action Generation To bypass corrupted physical observations during an OOD state, we implement a recursive “wheel” mechanism for imagined execution after tracking (Fig. 1(b)). Instead of using real visual inputs, the policy generates action chunks $\left( a _ { 0 } \ldots a _ { N } \right)$ conditioned entirely on the previously predicted latent embedding. Concurrently, the Difusion World Model uses this imagined latent and the newly predicted actions to forecast the next latent state $\left( O _ { 0 } \ldots O _ { M } \right)$ . Autoregressively feeding these forecasts back into the policy sustains a coherent trajectory purely within the agent’s internal imagination, enabling safe task execution until target recovery is triggered.

Target-Specific Recovery. Since simultaneous global recovery is impractical, we restore the environment iteratively. For an imagined subtask $S _ { j }$ targeting $C _ { j } \in \mathcal { C } _ { d i s p } ,$ we define a simple heuristic boolean trigger $\mathcal { T } _ { e n d } ( S _ { j } )$ . Once $\begin{array} { r } { \mathcal { T } _ { e n d } ( S _ { j } ) \stackrel { \cdot } { = } \mathrm { T r u e } , } \end{array}$ the agent applies an inverse spatial translation $\mathbf { a } _ { j } ^ { r e c } = - \varDelta \mathbf { p } _ { C _ { j } }$ to realign the imagined trajectory with the physical object, removing $C _ { j }$ from $\mathcal { C } _ { d i s p }$ . We then re-evaluate $\mathcal { D } _ { R - I } \colon$ if it remains above $\tau _ { d i f f }$ , the policy continues executing via imagination. Otherwise, the environment is fully recovered, and the policy reverts to the real visual latent $\mathbf { O } _ { r e a l } ^ { t }$ for standard ID execution.

## 4 Experiments

## 4.1 Simulation Experiments

Benchmarks and Datasets To evaluate the proposed method, we utilize two widely adopted continuous control simulation benchmarks: MetaWorld [56] and Adroit [31]. Specifically, we construct our evaluation pool using 10 MetaWorld tasks, ranging from simple to hard dificulties, to test general robotic manipulation capabilities. Additionally, to assess performance on highly complex dynamical systems, we select 3 tasks from the Adroit benchmark.

![](images/d2618a26e489c3bae459131487cde005376216363ff62f37cbcd82fe39fa21dd.jpg)  
Fig. 3: Out-of-Distribution (OOD) spatial disturbances in MetaWorld. Yellow dashed boxes highlight target objects, and solid arrows indicate the direction of dynamic displacements injected mid-task. The exact linear translation vectors (x, y, z) applied to shift each object out-of-distribution are detailed below.

Model Setup The shared 3D visual encoder processes M = 2 observation steps, concatenating 64-dimensional features from both a PointNet (point cloud) and an MLP (proprioception) into a 128-dimensional unified latent embedding. Conditioned on this M = 2 history, the Difusion Policy predicts an action horizon of H = 16 and executes an N = 8 step chunk. It employs a 1D U-Net with down-sampling channels of [256, 512, 1024] and a 128-dimensional difusion step embedding. Concurrently, the Difusion World Model uses a separate, identically parameterized 1D U-Net to predict N = 8 future latent embeddings conditioned on the M = 2 history and N = 8 planned actions. λ is set to 1.

Out-of-Distribution Setup To test robustness against spatial shifts, we evaluate OOD performance exclusively in MetaWorld. During training, targets are randomized within a fixed In-Distribution area. For OOD inference, we introduce sudden mid-task displacements, moving objects entirely outside these training boundaries (Fig. 3). Applied as direct linear coordinate translations in MuJoCo, these shifts provide measurable disturbances to reliably evaluate recovery.

Training and Evaluation For a fair and rigorous comparison, we benchmark our Dream Difusion Policy (DDP) against two state-of-the-art baselines: DP3 [57] and FlowPolicy [58]. All models are trained for 500 epochs with 3 seeds (seed0, 1, 2) using a fixed dataset of 50 demonstrations per task. To ensure an equitable evaluation, the foundational policy architectures and core hyperparameters across all methods are kept strictly identical. During inference, we evaluate each method for 100 episodes per task, per seed, and calculate the average success rate. Our evaluation is systematically divided into the following phases:

Table 1: Evaluation Results (Success Rate %). We report the In-Distribution (ID) and Out-of-Distribution (OOD) performance of the base policies, the trackingaugmented policies (DP3 ablation, Flow Policy, our DDP) under OOD, and the robustness of DDP under various imagination substitution rates.

<table><tr><td rowspan="2">Algorithm /Condition</td><td colspan="3">Adroit</td><td colspan="10">MetaWorld</td><td colspan="3">Adroit MetaWorld Total</td></tr><tr><td>Door</td><td>Hammer</td><td>Pen</td><td>Assembly</td><td>ButtonPress</td><td>CoffeePull</td><td>DoorUnlock</td><td>DrawerOpen</td><td>Hammer</td><td>PickPlace</td><td>Push</td><td>Soccer</td><td>Sweep</td><td>Avg</td><td>Avg</td><td>Avg</td></tr><tr><td colspan="17">Base Policy (ID)</td></tr><tr><td>DP3</td><td> $69 \pm 4$ </td><td> $100 \pm 0$ </td><td> $76 \pm 7$ </td><td> $100 \pm 0$ </td><td> $100 \pm 0$ </td><td> $99 \pm 1$ </td><td> $100 \pm 0$ </td><td> $100 \pm 0$ </td><td> $100 \pm 0$ </td><td> $95 \pm 1$ </td><td> $94 \pm 1$ </td><td> $35 \pm 3$ </td><td> $100 \pm 0$ </td><td>81.5</td><td>92.3</td><td>89.8</td></tr><tr><td>FlowPolicy</td><td> $64 \pm 4$ </td><td> $88 \pm 1$ </td><td> $67 \pm 4$ </td><td> $100 \pm 0$ </td><td> $100 \pm 0$ </td><td> $100 \pm 0$ </td><td> $100 \pm 0$ </td><td> $100 \pm 0$ </td><td> $98 \pm 1$ </td><td> $96 \pm 1$ </td><td> $95 \pm 1$ </td><td> $53 \pm 6$ </td><td> $100 \pm 0$ </td><td>72.9</td><td>94.2</td><td>89.3</td></tr><tr><td>DDP (Ours)</td><td> $63 \pm 3$ </td><td> $99 \pm 1$ </td><td> $71 \pm 2$ </td><td> $100 \pm 0$ </td><td> $100 \pm 0$ </td><td> $100 \pm 0$ </td><td> $100 \pm 0$ </td><td> $100 \pm 0$ </td><td> $100 \pm 0$ </td><td> $95 \pm 1$ </td><td> $93 \pm 1$ </td><td> $42 \pm 7$ </td><td> $100 \pm 0$ </td><td>77.8</td><td>93.0</td><td>89.5</td></tr><tr><td colspan="17">Base Policy (OOD)</td></tr><tr><td>DP3</td><td>-</td><td>-</td><td>-</td><td> $0 \pm 0$ </td><td> $0 \pm 0$ </td><td> $0 \pm 0$ </td><td> $2 \pm 2$ </td><td> $4 \pm 2$ </td><td> $0 \pm 0$ </td><td> $0 \pm 0$ </td><td> $0 \pm 0$ </td><td> $1 \pm 1$ </td><td> $0 \pm 0$ </td><td>-</td><td>0.8</td><td>0.8</td></tr><tr><td>FlowPolicy</td><td>-</td><td>-</td><td>-</td><td> $0 \pm 0$ </td><td> $11 \pm 5$ </td><td> $0 \pm 0$ </td><td> $53 \pm 14$ </td><td> $0 \pm 0$ </td><td> $1 \pm 1$ </td><td> $0 \pm 0$ </td><td> $0 \pm 0$ </td><td> $2 \pm 2$ </td><td> $0 \pm 0$ </td><td>-</td><td>6.7</td><td>6.7</td></tr><tr><td>DDP (Ours)</td><td>-</td><td>-</td><td>-</td><td> $0 \pm 0$ </td><td> $0 \pm 0$ </td><td> $0 \pm 0$ </td><td> $5 \pm 5$ </td><td> $3 \pm 0$ </td><td> $1 \pm 1$ </td><td> $0 \pm 0$ </td><td> $0 \pm 0$ </td><td> $4 \pm 2$ </td><td> $0 \pm 0$ </td><td>-</td><td>1.3</td><td>1.3</td></tr><tr><td colspan="17">Tracking-Augmented (OOD)</td></tr><tr><td>FlowPolicy</td><td>-</td><td>-</td><td>-</td><td> $0 \pm 0$ </td><td> $13 \pm 2$ </td><td> $0 \pm 0$ </td><td> $41 \pm 5$ </td><td> $0 \pm 0$ </td><td> $0 \pm 0$ </td><td> $0 \pm 0$ </td><td> $0 \pm 0$ </td><td> $0 \pm 0$ </td><td> $0 \pm 0$ </td><td>-</td><td>5.4</td><td>5.4</td></tr><tr><td>DP3 (Ablation)</td><td>-</td><td>-</td><td>-</td><td> $0 \pm 0$ </td><td> $93 \pm 3$ </td><td> $23 \pm 4$ </td><td> $59 \pm 16$ </td><td> $3 \pm 2$ </td><td> $1 \pm 0$ </td><td> $13 \pm 13$ </td><td> $9 \pm 3$ </td><td> $16 \pm 4$ </td><td> $22 \pm 14$ </td><td>-</td><td>23.9</td><td>23.9</td></tr><tr><td>DDP (Ours)</td><td>-</td><td>-</td><td>-</td><td> $96 \pm 2$ </td><td> $100 \pm 0$ </td><td> $94 \pm 3$ </td><td> $74 \pm 4$ </td><td> $100 \pm 0$ </td><td> $100 \pm 0$ </td><td> $39 \pm 5$ </td><td> $39 \pm 6$ </td><td> $17 \pm 4$ </td><td> $79 \pm 7$ </td><td>-</td><td>73.8</td><td>73.8</td></tr><tr><td colspan="17">DDP Imagination Interventions (ID)</td></tr><tr><td>DDP (10% Imagined)</td><td> $95 \pm 3$ </td><td> $99 \pm 0$ </td><td> $92 \pm 1$ </td><td> $100 \pm 0$ </td><td> $100 \pm 0$ </td><td> $100 \pm 0$ </td><td> $99 \pm 0$ </td><td> $98 \pm 1$ </td><td> $100 \pm 0$ </td><td> $93 \pm 3$ </td><td> $93 \pm 2$ </td><td> $40 \pm 3$ </td><td> $100 \pm 0$ </td><td>95.2</td><td>92.3</td><td>93.0</td></tr><tr><td>DDP (50% Imagined)</td><td> $96 \pm 1$ </td><td> $98 \pm 2$ </td><td> $87 \pm 3$ </td><td> $99 \pm 0$ </td><td> $100 \pm 0$ </td><td> $98 \pm 1$ </td><td> $99 \pm 1$ </td><td> $90 \pm 1$ </td><td> $100 \pm 0$ </td><td> $81 \pm 6$ </td><td> $91 \pm 2$ </td><td> $39 \pm 3$ </td><td> $98 \pm 1$ </td><td>94.0</td><td>89.4</td><td>90.5</td></tr><tr><td>DDP (Open-loop)</td><td> $96 \pm 3$ </td><td> $98 \pm 2$ </td><td> $86 \pm 3$ </td><td> $18 \pm 2$ </td><td> $100 \pm 0$ </td><td> $54 \pm 3$ </td><td> $97 \pm 1$ </td><td> $48 \pm 3$ </td><td> $99 \pm 0$ </td><td> $21 \pm 3$ </td><td> $38 \pm 3$ </td><td> $42 \pm 3$ </td><td> $97 \pm 1$ </td><td>93.2</td><td>61.4</td><td>68.7</td></tr></table>

1. Base Policy Accuracy (ID & OOD): Evaluating standard In-Distribution and Out-of-distribution performance of DP3, FlowPolicy, and DDP.

2. Dream Difusion Policy Accuracy (OOD): Evaluating DDP using our tracking, imagination and recovery mechanism under OOD conditions.

3. Ablation Tracking-Augmented Accuracy (OOD): To isolate the benefits of our World Model, we perform a direct ablation by applying the tracking intervention to the base DP3 policy. We additionally evaluate FlowPolicy equipped with the same tracking mechanism for a broader comparison.

4. World Model Imagination Analysis (ID): To test the World Model’s reliability, we randomly replace real observation with imagined latents after the initial chunk at 10%, 50% and 100% (open-loop imagined rollout) probabilities.

Results As shown in Tab. 1, standard base policies fail under severe OOD shifts. However, our tracking-augmented DDP efectively handles these disturbances, achieving a 73.8% average success rate on MetaWorld. Applying identical tracking to the baselines yields lower success rates (23.9% for DP3, 5.4% for Flow-Policy). This proves tracking alone is insuficient; the Difusion World Model’s coupled latents are crucial for robust task completion.

Figure 4 illustrates DDP’s execution on the MetaWorld Assembly and Push tasks. In Assembly (2 subtasks), the second object peg is displaced Out-of-Distribution after the robot grasps the ring. After tracking this shift, DDP executes a precise action to successfully insert the ring. In Push (2 subtasks), the first object puck is displaced before grasping. DDP tracks the shifted puck, execute actions via imagination, utilizes a localized recovery action to move the end-efector to ID position, and accurately pushes it to the target. For more comprehensive results, please refer to Sec. B in the Appendix.

Furthermore, Tab. 1 demonstrates DDP’s resilience when replacing real observations with imagined latents. On MetaWorld, DDP maintains near-baseline accuracy at 10% (92.3%) and 50% (89.4%) substitution rates, remaining robust even at 100% (61.4%). On the complex Adroit benchmark, imagination actually boosts success from a 77.8% baseline to 95.2% at 10% substitution. This confirms the built-in World Model not only bridges visual gaps but also acts as a temporal smoother, stabilizing high-DoF continuous control.

![](images/5e21bf863041a40f9c155c831e9a6122c2a6bbec0192abfd58f970b33d884604.jpg)  
Fig. 4: DDP execution on MetaWorld Assembly and Push. Sequences show the transition from the ID Base Policy (green) to OOD tracking (yellow), bypassing visual shifts via the imagined Dream Policy (pink), and finally recovering to the target.

## 4.2 Real World Experiments

Real Robot Benchmark To evaluate physical eficacy and performance under OOD, we design three real-world manipulation tasks of varying complexity. During training, initial target positions are uniformly randomized along a 10 cm linear range to ensure baseline state variance.

– Press Button (Easy | 1 Subtask, 1 OOD Target): The robot closes its gripper to precisely press a button. Success requires depressing it exactly to the activation threshold, avoiding both incomplete contact (under-pressing) and excessive force (over-pressing).

– Pour Tea (Medium | 2 Subtasks, 1 OOD Target): The robot grasps a teapot, navigates to a cup, executes a pouring motion, and returns the teapot to the table. Success requires the spout to remain precisely aligned with the cup’s center during the entire pour.

– Stack Blocks (Hard | 2 Subtasks, 2 OOD Targets): The robot picks up a pink block and places it atop a green base block. Success requires the pink block to remain perfectly stable and balanced after the gripper releases.

Hardware and State Setup We deploy our framework on a Franka Emika Panda with a Robotiq 2F-85 gripper. An Intel RealSense L515 captures visual observations, while a D455 camera continuously tracks objects. To ensure low latency, raw point clouds are downsampled via Farthest Point Sampling to 2048 points. Hardware setup is shown in Fig. 5. The proprioceptive state consists of the end-efector pose. The action space is formulated as 7-DoF relative commands $( \varDelta x , \varDelta y , \varDelta z$ , roll, pitch, yaw) resolved via the franky [33] inverse kinematics library, plus a binary gripper state (Close or Open).

Data Collection, Network Architecture and Compute We collect 25 expert demonstrations per task via a scripted kinematic oracle that injects spatial variance to eficiently mimic human data distributions. For high-speed realtime control, we employ “Simple DDP”, a lightweight architecture inspired by DP3. It utilizes reduced-capacity 1D U-Nets with down-sampling dimensions of [128, 256, 384] for the Difusion Policy and [128, 256, 512] for the Difusion World Model. Finally, we retain all sequence hyperparameters from simulation: M = 2, H = 16, N = 8, and λ = 1. The models are trained for 2000 epochs on a NVIDIA 4090 GPU. During deployment, the DDP architecture is hosted on an NVIDIA 4070 GPU, with inference times of 40 ms for the difusion policy and 70 ms for the world model. Concurrently, to guarantee robustness in real-time tracking, we utilize the FoundationPose++ [49] module, as shown in Fig. 5(b). For further comprehensive detail, please refer to Sec. C in the Appendix.

![](images/a65850ecd29f1333080bbbe3c0ba59ba6d058fd5350000491bae7baeb21b85e1.jpg)

![](images/ce8ffb856ef2dbda9d9fd2dc07f09ab9bb446af7daf7cca9395ac07f4b0cd630.jpg)  
Fig. 5: Robot Setup and Objects Tracking. (a) is all the hardware for the realworld experiment. (b) shows the bounding box for object tracking across all tasks.

Table 2: Real Robot Evaluation Results (Success Rate %). We evaluate the baseline and our Dream Difusion Policy (DDP) across three tasks (10 trials each) of varying dificulty under In-Distribution (ID) and Out-of-Distribution (OOD) settings.

<table><tr><td>Algorithm / Condition</td><td>Press Button</td><td>Pour Tea</td><td>Stack Blocks</td><td>Average</td></tr><tr><td colspan="5">Base Policy (ID)</td></tr><tr><td>DP3</td><td>100</td><td>90</td><td>90</td><td>93.3</td></tr><tr><td>DDP (Ours)</td><td>100</td><td>90</td><td>80</td><td>90.0</td></tr><tr><td colspan="5">Tracking-Augmented (OOD)</td></tr><tr><td>DP3 (Ablation)</td><td>10</td><td>0</td><td>0</td><td>3.3</td></tr><tr><td>DDP (Ours)</td><td>100</td><td>90</td><td>60</td><td>83.3</td></tr><tr><td colspan="5">Imagination Intervention (ID)</td></tr><tr><td>DDP (Open-loop Imagination)</td><td>100</td><td>80</td><td>50</td><td>76.7</td></tr></table>

Results Fig. 10 in the Appendix illustrates the successful execution sequences of DDP under In-Distribution (ID) settings for the three real-world tasks: Press Button, Pour Tea, and Stack Blocks. To evaluate OOD robustness, target objects are abruptly moved completely outside the In-Distribution (ID) zone mid-task. As shown in Fig. 6, our OOD detector identifies these displacements by monitoring latent discrepancy in real-time, which quantifies the latent distance between actual visual observations and the policy’s imagined projections. Sudden dis-

OOD Detection(Object Displacement) via Real-Imagination Discrepancy

![](images/66ebceb4f7fa505663aa66abcc932fae506ca56f5b61978f6df586b54068a1b4.jpg)

![](images/c8d1a934761697e7cd8a1e8f054562e3517bb0305421574dad7b6813571ff905.jpg)

![](images/bc1128b5c5239b96b3890c4abaa9d1cc8f110623ca28f32ec16dd9739bc60a65.jpg)

Fig. 6: OOD Detection via Real-Imagination Discrepancy. The detector successfully identifies anomalies when objects are accidentally shifted across three diferent tasks. An OOD state is triggered (shaded regions) when the latent discrepancy $\left( \mathcal { D } _ { R - I } \right)$ exceeds a task-specific threshold $\tau _ { d i f f }$ . Specifically, we set $\tau _ { d i f f } = 0 . 0 0 0 2$ for Stack Blocks, $\tau _ { d i f f } = 0 . 0 0 0 4$ for Pour Tea, and $\tau _ { d i f f } = 0 . 0 0 1 2$ for Press Button.  
![](images/2b46cd6ab397f571b2659d16e6ec53df0c5f2141fa26e7cc80b75b9dd0f1df75.jpg)  
Fig. 7: Qualitative real-robot execution on Stack Blocks, Pour Tea, and Press Button tasks. DDP successfully tracks OOD objects, finishes the main actions of subtasks via the imagined Dream Policy, and executes recover to achieve the remaining task success. In contrast, the tracking-augmented DP3 ablation fails across severe OOD scenarios.

(a) Occlusion scene  
![](images/22663c87617ce36214687a38b8c3a497811d7491c0d225a4fd1aed24a281181b.jpg)  
(b) Stack Blocks (Left: DP3, right: DDP)  
(c) Pour Tea (Left: DP3, right: DDP)  
Fig. 8: Qualitative results under visual occlusion. (a) A hand obstructs the camera mid-task. In (b) Stack Blocks and (c) Pour Tea, baseline DP3 fails due to visual corruption, whereas DDP seamlessly bypasses the occlusion using internal imagination.

placements cause the observed visual state to sharply deviate from the imagined trajectory, spiking the discrepancy above a threshold to reliably flag the OOD condition.

As shown in Tab. 2, our tracking-augmented DDP achieves an 83.3% average success rate, drastically outperforming the DP3 tracking ablation (3.3%). Figure 7 illustrates these execution diferences. In the two-subtask Stack Blocks sequence, DDP tracks OOD shifts for each object, finishes the remaining motions via imagination, and triggers a precise physical recovery between subtasks based on the gripper stage; conversely, the DP3 ablation entirely fails to pick the displaced block. In Pour Tea, DDP tracks the shifted cup and seamlessly executes pouring actions from imagination after picking the teapot, whereas DP3 accidentally drops the pot. For Press Button, DDP recovers to press the target precisely to its threshold, while the DP3 ablation fails by either under-pressing or over-pressing. Detailed failure mode illustration for each real-world task is provided Fig. 11.

Furthermore, we test DDP’s reliance on its internal world model by executing tasks using 100% imagined latents after the first observation chunk (Open-loop Imagination). As detailed in Tab. 2, DDP maintains a remarkable 76.7% average success rate. This performance confirms that the network’s imagined representations are stable and reliable for continuous real-world control.

Robustness to Visual Occlusion. To further evaluate DDP against severe visual occlusion, we block the camera mid-task. This creates a massive realimagination discrepancy $\left( \mathcal { D } _ { R - I } \right)$ , instantly triggering OOD mode. While the DP3 baseline fails when blinded, DDP safely bypasses the corruption to complete both Stack Blocks and Pour Tea purely via autoregressive imagination (Fig. 8).

## 5 Conclusion

We introduced the Dream Difusion Policy (DDP) to enhance visuomotor robustness against severe out-of-distribution (OOD) disturbances. Co-optimizing a difusion policy and world model via a shared 3D encoder embeds predictive dynamics directly into DDP’s representation. During inference, DDP detects OOD anomalies via a real-imagination discrepancy $\left( \mathcal { D } _ { R - I } \right)$ . Rather than collapsing under corrupted vision, it safely bypasses these shifts using an autoregressive imagination loop—executing actions from latent forecasts before triggering a localized physical recovery. Evaluations confirm DDP significantly outperforms baselines specifically in OOD scenarios, maintaining a high success rate even under almost complete visual blindness where standard methods fail.

Limitations and Future Work. DDP has three main limitations: (1) It highly relies on external tracking modules, allowing drift to accumulate across multi-stage subtasks, (2) tasks must begin strictly In-Distribution (ID) to properly anchor the initial observation, and (3) the inference mechanism struggles with compounding OOD events or low-level action disruptions (e.g., gripper slippage). Addressing these dependencies remains a key direction for future research. Potential solutions, such as employing meta-reinforcement learning [4, 42, 50] to adaptively fine-tune the world model online, integrating the lifelong learning mechanism [28] to preserve and combine prior knowledge, or integrating low-level tactile feedback [46], could further enhance DDP’s resilience in real-world applications. We also plan to extend DDP to more complex multi-object manipulation tasks and explore its integration with language-conditioned methods [51, 60] or large language models for higher-level reasoning and planning [52].

## References

1. Ada, S.E., Oztop, E., Ugur, E.: Difusion policies for out-of-distribution generalization in ofline reinforcement learning. IEEE Robotics and Automation Letters 9(4), 3116–3123 (2024)

2. Agarwal, N., Ali, A., Bala, M., Balaji, Y., Barker, E., Cai, T., Chattopadhyay, P., Chen, Y., Cui, Y., Ding, Y., et al.: Cosmos world foundation model platform for physical ai. arXiv preprint arXiv:2501.03575 (2025)

3. Alonso, E., Jelley, A., Micheli, V., Kanervisto, A., Storkey, A.J., Pearce, T., Fleuret, F.: Difusion for world modeling: Visual details matter in atari. Advances in Neural Information Processing Systems 37, 58757–58791 (2024)

4. Bing, Z., Koch, A., Yao, X., Huang, K., Knoll, A.: Meta-reinforcement learning via language instructions. In: 2023 IEEE International Conference on Robotics and Automation (ICRA). pp. 5985–5991. IEEE (2023)

5. Cen, J., Yu, C., Yuan, H., Jiang, Y., Huang, S., Guo, J., Li, X., Song, Y., Luo, H., Wang, F., et al.: Worldvla: Towards autoregressive action world model. arXiv preprint arXiv:2506.21539 (2025)

6. Chandra, A.L., Nematollahi, I., Huang, C., Welschehold, T., Burgard, W., Valada, A.: Diwa: Difusion policy adaptation with world models. In: Conference on Robot Learning. pp. 3378–3400. PMLR (2025)

7. Chi, C., Xu, Z., Feng, S., Cousineau, E., Du, Y., Burchfiel, B., Tedrake, R., Song, S.: Difusion policy: Visuomotor policy learning via action difusion. The International Journal of Robotics Research 44(10-11), 1684–1704 (2025)

8. Curtis, A., Li, E., Noseworthy, M., Gothoskar, N., Chitta, S., Li, H., Kaelbling, L.P., Carey, N.E.: Flow-based domain randomization for learning and sequencing robotic skills. In: Forty-second International Conference on Machine Learning (2025), https://openreview.net/forum?id=9JQXuyzdGL

9. Ding, J., Zhang, Y., Shang, Y., Zhang, Y., Zong, Z., Feng, J., Yuan, Y., Su, H., Li, N., Sukiennik, N., et al.: Understanding world or predicting future? a comprehensive survey of world models. ACM Computing Surveys 58(3), 1–38 (2025)

10. Ding, Z., Zhang, A., Tian, Y., Zheng, Q.: Difusion world model: Future modeling beyond step-by-step rollout for ofline reinforcement learning. arXiv preprint arXiv:2402.03570 (2024)

11. Du, M., Song, S.: Dynaguide: Steering difusion polices with active dynamic guidance. In: The Thirty-ninth Annual Conference on Neural Information Processing Systems (2025), https://openreview.net/forum?id=XOw7Yf8qN3

12. Gao, G.J., Li, T., Figueroa, N.: Out-of-distribution recovery with object-centric keypoint inverse policy for visuomotor imitation learning. In: 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). pp. 9816–9828. IEEE (2025)

13. Geirhos, R., Jacobsen, J.H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M., Wichmann, F.A.: Shortcut learning in deep neural networks. Nature Machine Intelligence 2(11), 665–673 (2020)

14. Graham, M.S., Pinaya, W.H., Tudosiu, P.D., Nachev, P., Ourselin, S., Cardoso, J.: Denoising difusion models for out-of-distribution detection. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 2948–2957 (2023)

15. Ha, D., Schmidhuber, J.: World models. arXiv preprint arXiv:1803.10122 (2018)

16. Hafner, D., Lillicrap, T., Ba, J., Norouzi, M.: Dream to control: Learning behaviors by latent imagination. arXiv preprint arXiv:1912.01603 (2019)

17. Hafner, D., Lillicrap, T., Norouzi, M., Ba, J.: Mastering atari with discrete world models. arXiv preprint arXiv:2010.02193 (2020)

18. Hafner, D., Pasukonis, J., Ba, J., Lillicrap, T.: Mastering diverse domains through world models. arXiv preprint arXiv:2301.04104 (2023)

19. He, C., Liu, X., Camps, G.S., Sartoretti, G., Schwager, M.: Demystifying difusion policies: Action memorization and simple lookup table alternatives. arXiv preprint arXiv:2505.05787 (2025)

20. Ho, J., Jain, A., Abbeel, P.: Denoising difusion probabilistic models. Advances in neural information processing systems 33, 6840–6851 (2020)

21. Huang, D., Navab, N., Jiang, Z.: Improving robustness to out-of-distribution states in imitation learning via deep koopman-boosted difusion policy. IEEE Transactions on Robotics 41, 6680–6692 (2025)

22. Huang, T., Jiang, G., Ze, Y., Xu, H.: Difusion reward: Learning rewards via conditional video difusion. In: European Conference on Computer Vision. pp. 478–495. Springer (2024)

23. Isaku, E., Sartaj, H., Ali, S., Sanguino, B., Wang, T., Li, G., Zhang, H., Peyrucain, T.: Out of distribution detection in self-adaptive robots with ai-powered digital twins. arXiv preprint arXiv:2509.12982 (2025)

24. Lai, L., Huang, A.Z., Gershman, S.J.: Action chunking as policy compression. PsyArXiv (2022)

25. Lee, S.W., Kang, X., Kuo, Y.L.: Dif-dagger: Uncertainty estimation with difusion policy for robotic manipulation. In: 2025 IEEE International Conference on Robotics and Automation (ICRA). pp. 4845–4852. IEEE (2025)

26. Liu, Y., Hamid, J.I., Xie, A., Lee, Y., Du, M., Finn, C.: Bidirectional decoding: Improving action chunking via guided test-time sampling. arXiv preprint arXiv:2408.17355 (2024)

27. Ma, X., Patidar, S., Haughton, I., James, S.: Hierarchical difusion policy for kinematics-aware multi-task robotic manipulation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 18081– 18090 (2024)

28. Meng, Y., Bing, Z., Yao, X., Chen, K., Huang, K., Gao, Y., Sun, F., Knoll, A.: Preserving and combining knowledge in robotic lifelong reinforcement learning. Nature Machine Intelligence 7(2), 256–269 (2025)

29. Morales-Brotons, D., Vogels, T., Hendrikx, H.: Exponential moving average of weights in deep learning: Dynamics and benefits. arXiv preprint arXiv:2411.18704 (2024)

30. Perez, E., Strub, F., De Vries, H., Dumoulin, V., Courville, A.: Film: Visual reasoning with a general conditioning layer. In: Proceedings of the AAAI conference on artificial intelligence. vol. 32 (2018)

31. Rajeswaran, A., Kumar, V., Gupta, A., Vezzani, G., Schulman, J., Todorov, E., Levine, S.: Learning complex dexterous manipulation with deep reinforcement learning and demonstrations. arXiv preprint arXiv:1709.10087 (2017)

32. Saharia, C., Chan, W., Chang, H., Lee, C., Ho, J., Salimans, T., Fleet, D., Norouzi, M.: Palette: Image-to-image difusion models. In: ACM SIGGRAPH 2022 conference proceedings. pp. 1–10 (2022)

33. Schneider, T.: franky: High-level control library for franka robots. https:// github.com/TimSchneider42/franky (2024), version 1.1.3

34. Seo, M., Park, H.A., Yuan, S., Zhu, Y., Sentis, L.: Legato: Cross-embodiment imitation using a grasping tool. IEEE Robotics and Automation Letters 10(3), 2854–2861 (2025). https://doi.org/10.1109/LRA.2025.3535182

35. Seo, Y., Hafner, D., Liu, H., Liu, F., James, S., Lee, K., Abbeel, P.: Masked world models for visual control. In: Conference on Robot Learning. pp. 1332–1344. PMLR (2023)

36. Sun, Z., Song, S.: Latent policy barrier: Learning robust visuomotor policies by staying in-distribution. arXiv preprint arXiv:2508.05941 (2025)

37. Tian, J., Wang, L., Zhou, S., Wang, S., Li, J., Sun, H., Tang, W.: Pdfactor: Learning tri-perspective view policy difusion field for multi-task robotic manipulation. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 15757–15767 (2025)

38. Tobin, J., Fong, R., Ray, A., Schneider, J., Zaremba, W., Abbeel, P.: Domain randomization for transferring deep neural networks from simulation to the real world. In: 2017 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). pp. 23–30. IEEE (2017)

39. Wang, D., Shelhamer, E., Liu, S., Olshausen, B., Darrell, T.: Tent: Fully testtime adaptation by entropy minimization. In: International Conference on Learning Representations (ICLR) (2021)

40. Wang, D., Liu, C., Chang, F., Xu, Y.: Hierarchical difusion policy: manipulation trajectory generation via contact guidance. IEEE Transactions on Robotics (2025)

41. Wang, D., Liu, C., Chang, F., Xu, Y.: Hierarchical difusion policy: Manipulation trajectory generation via contact guidance. IEEE Transactions on Robotics 41, 2086–2104 (2025). https://doi.org/10.1109/TRO.2025.3547272

42. Wang, M., Bing, Z., Yao, X., Wang, S., Kai, H., Su, H., Yang, C., Knoll, A.: Metareinforcement learning based on self-supervised task representation learning. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 37, pp. 10157– 10165 (2023)

43. Wang, R., Zhuang, Z., Kragic, D., Pokorny, F.T.: Palm: Enhanced generalizability for local visuomotor policies via perception alignment. arXiv preprint arXiv:2601.19514 (2026)

44. Wen, B., Yang, W., Kautz, J., Birchfield, S.: Foundationpose: Unified 6d pose estimation and tracking of novel objects. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 17868–17879 (2024)

45. Wu, S., Zhu, Y., Huang, Y., Zhu, K., Gu, J., Yu, J., Shi, Y., Wang, J.: Aforddp: Generalizable difusion policy with transferable afordance. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 6971–6980 (2025)

46. Wu, Y., Chen, Z., Wu, F., Chen, L., Zhang, L., Bing, Z., Swikir, A., Haddadin, S., Knoll, A.: Tacdifusion: Force-domain difusion policy for precise tactile manipulation. In: 2025 IEEE International Conference on Robotics and Automation (ICRA). pp. 11831–11837. IEEE (2025)

47. Wu, Y., Wang, H., Chen, Z., Pang, J., Xu, D.: On-device difusion transformer policy for eficient robot manipulation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 14073–14083 (2025)

48. Xue, H., Ren, J., Chen, W., Zhang, G., Fang, Y., Gu, G., Xu, H., Lu, C.: Reactive difusion policy: Slow-fast visual-tactile policy learning for contact-rich manipulation. In: Proceedings of Robotics: Science and Systems (RSS) (2025)

49. Yan, W., Chu, J.: Foundationpose++: Simple tricks boost foundationpose performance in high-dynamic scenes. https://github.com/teal024/FoundationPoseplus-plus (2025), gitHub Repository

50. Yao, X., Bing, Z., Zhuang, G., Chen, K., Zhou, H., Huang, K., Knoll, A.: Learning from symmetry: Meta-reinforcement learning with symmetrical behaviors and language instructions. In: IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). pp. 5574–5581. IEEE (2023)

51. Yao, X., Blei, T., Meng, Y., Zhang, Y., Zhou, H., Bing, Z., Huang, K., Sun, F., Knoll, A.: Long-horizon language-conditioned imitation learning for robotic manipulation. IEEE/ASME Transactions on Mechatronics pp. 1–12 (2025)

52. Yao, X., Zhou, H., Mees, O., Meng, Y., Xiao, T., Bisk, Y., Oh, J., Johns, E., Shridhar, M., Shah, D., et al.: Bridging language and action: A survey of languageconditioned robot manipulation. arXiv preprint arXiv:2312.10807 (2023)

53. Yao, X., Zhou, Y., Meng, Y., Dong, L., Hong, L., Zhang, Z., Bing, Z., Huang, K., Sun, F., Knoll, A.: Pick-and-place manipulation across grippers without retraining: A learning-optimization difusion policy approach. arXiv preprint arXiv:2502.15613 (2025)

54. Yao, X., Zhou, Y., Meng, Y., Liu, Y., Dong, L., Zhang, Z., Bing, Z., Huang, K., Sun, F., Knoll, A.: Inference-stage adaptation-projection strategy adapts difusion policy to cross-manipulators scenarios. arXiv preprint arXiv:2509.11621 (2025)

55. Ye, S., Ge, Y., Zheng, K., Gao, S., Yu, S., Kurian, G., Indupuru, S., Tan, Y.L., Zhu, C., Xiang, J., et al.: World action models are zero-shot policies. arXiv preprint arXiv:2602.15922 (2026)

56. Yu, T., Quillen, D., He, Z., Julian, R., Hausman, K., Finn, C., Levine, S.: Metaworld: A benchmark and evaluation for multi-task and meta reinforcement learning. In: Conference on robot learning. pp. 1094–1100. PMLR (2020)

57. Ze, Y., Zhang, G., Zhang, K., Hu, C., Wang, M., Xu, H.: 3d difusion policy: Generalizable visuomotor policy learning via simple 3d representations. In: Proceedings of Robotics: Science and Systems (RSS) (2024)

58. Zhang, Q., Liu, Z., Fan, H., Liu, G., Zeng, B., Liu, S.: Flowpolicy: Enabling fast and robust 3d flow-based policy via consistency flow matching for robot manipulation.

In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 39, pp. 14754–14762 (2025)

59. Zhang, X., Chang, M., Kumar, P., Gupta, S.: Difusion meets dagger: Supercharging eye-in-hand imitation learning. arXiv preprint arXiv:2402.17768 (2024)

60. Zhou, H., Bing, Z., Yao, X., Su, X., Yang, C., Huang, K., Knoll, A.: Languageconditioned imitation learning with base skill priors under unstructured data. IEEE Robotics and Automation Letters 9(11), 9805–9812 (2024)

61. Zhu, Y., Joshi, A., Stone, P., Zhu, Y.: Viola: Imitation learning for vision-based manipulation with object proposal priors. In: Conference on Robot Learning. pp. 1199–1210. PMLR (2023)

## Appendix

## A Pseudocode for Closed-Loop Inference

To further clarify the transition between physical reality (In-Distribution) and internal imagination (Out-of-Distribution), we provide the detailed pseudocode for the DDP inference stage in Algorithm 2. The algorithm highlights the real-time detection via latent discrepancy $\mathcal { D } _ { R - I } .$ , the recursive action generation during the OOD state, and the target-specific recovery mechanism triggered at the end of each subtask.

## B Details of Simulation OOD Experiments

In this section, we provide the exact details for the Out-of-Distribution (OOD) evaluation on the MetaWorld benchmark. We detail the mechanism for implementing object displacements, how the robot tracks these shifts in simulation, and the precise task-specific parameters (timing, displacement vectors, and target objects).

Oracle State Transitions and OOD Implementation To explicitly isolate and highlight the predictive power of the World Model’s “imagination” capability, we simplify the simulation evaluation pipeline. Specifically, we intentionally bypass the automated $\mathcal { D } _ { R - I } \mathrm { ~ O O D }$ detector and the heuristic recovery triggers described in the main text. Instead, we implement a controlled Oracle Tracking and State Transition Mechanism via direct MuJoCo backend manipulations. By hard-coding the exact timing of both the OOD anomaly injection and the physical recovery to specific action chunks, we ensure that the evaluation purely reflects the policy’s ability to execute manipulation actions via internal imagination, independent of external tracking accuracy or detection latency.

During the pre-defined execution step (i.e., the specific “chunk” detailed in Tab. 3), the environment execution is temporarily paused, and a sudden spatial shift is injected. As illustrated in Fig. 9 for the remaining MetaWorld tasks, the implementation follows two main logic branches depending on the nature of the task:

![](images/7dbcab69bec8d3e4335ddefdc8121cb2b09931ef3e5c99f8a8067c4bd4b6d38b.jpg)  
Fig. 9: DDP execution on the MetaWorld benchmark (Remaining 8 tasks). Sequences show the transition from the ID Base Policy (green) to OOD tracking (yellow), bypassing visual shifts via the imagined Dream Policy (pink), and finally triggering physical recovery (green dotted) to complete the target.

1. Single-Stage Tracking (Target-Centric Tasks): For tasks where the robot only needs to interact with the shifted object (e.g., Button Press, Drawer Open), we simultaneously apply a coordinate translation vector $\varDelta \mathbf { P } = [ d x , d y , d z ]$ to both the target object’s body position (sim.model.body\_pos) and the robot’s end-efector mock-up position (data.mocap\_pos). By directly hard-coding this ofset to the mocap, the robot instantly re-aligns its physical hand with the newly displaced object, maintaining the relative spatial geometry required for the internal “imagined” execution.

2. Two-Stage Tracking and Recovery (Pick-and-Move Tasks): For multisubtask challenges where an object is displaced but the final destination (goal) remains fixed (e.g., Pick Place, Push, Sweep, Cofee Pull), the script executes a hard-coded two-stage intervention:

– Stage 1 (Track): At the specified anomaly chunk, the object and the robot’s end-efector are shifted by ∆P so the robot can successfully grasp the displaced object. (Note: For Pick Place, a “Safe Lift” logic is explicitly coded—lifting the arm 5cm, shifting it, and lowering it back—to prevent clipping through the table during teleportation).

– Stage 2 (Recover and Policy Transition): At a strictly pre-defined subsequent chunk (representing the moment the object is securely grasped), a reverse ofset −∆P is applied exclusively to the robot’s end-efector. This forces the robot to instantly “snap back” to the original ID area. Crucially, the Oracle then perfectly mimics the $\mathcal { D } _ { R - I }$ re-evaluation logic defined in the main text to determine the subsequent policy mode:

• For Pick Place, Push, and Sweep, bringing the grasped object back to the ID zone efectively restores global visual consistency. Consequently, the Oracle successfully re-anchors the environment and transitions the system back to the real visual stream (ID Base Policy).

• Conversely, for Cofee Pull, the recovery step only snaps the arm and the grasped mug back to the ID target zone, while the large cofee machine is left permanently displaced. Because this leaves the global visual context strictly corrupted (which would naturally trigger $\mathcal { D } _ { R - I } > \tau _ { d i f f }$ in reality), the Oracle enforces that the system must continue executing purely via internal hallucination (OOD Dream Policy) to safely finish placing the mug.

Task-Specific OOD Parameters The exact parameters for all 10 MetaWorld tasks are summarized in Tab. 3. The “Intervention Timing” refers to the specific action chunk index where the sudden displacement is injected.

Detailed Breakdown of Multi-Subtask Interventions:

– Pick Place: At chunk 2, the Puck is shifted by [−0.2, −0.1, 0]. The robot performs a collision-free tracking maneuver. At chunk 6, the robot arm snaps back by $[ + 0 . 2 , + 0 . 1 , 0 ]$ to place the puck into the fixed goal, successfully recovering visual consistency and resuming the ID Base Policy.

– Push: At chunk 2, the Puck is shifted by [−0.25, 0, 0]. The robot tracks it. At chunk 6, the robot arm snaps back by [+0.25, 0, 0] to push the object toward the original fixed target, resuming the ID Base Policy.

– Sweep: At chunk 2, the Puck is shifted by [+0.2, +0.1, 0]. The robot tracks it. At chunk 6, the robot arm snaps back $\mathrm { b y \ [ - 0 . 2 , - 0 . 1 , 0 ] }$ to sweep the object into the fixed hole, resuming the ID Base Policy.

– Cofee Pull: At chunk 2, both the Mug and the Cofee Machine are shifted by [−0.2, 0, 0], but the target placement zone remains fixed. At chunk 8, the robot arm snaps back by [+0.2, 0, 0] to pull the mug into the correct target zone. Since the machine remains displaced (OOD), the agent continues executing via the OOD Dream Policy to complete the task.

## C Further Details of Real-World Experiments

In this section, we provide additional configuration details for our real-world deployment, including the control hyper-parameters, visual processing pipeline, and the exact camera setup.

![](images/9b02b65f0b276213217904399d195171441df41439b596d8ad1880064e556123.jpg)  
Fig. 10: In-Distribution Scenarios. Real-world execution sequences demonstrating successful nominal manipulation for the Stack Blocks, Pour Tea, and Press Button tasks.

Network Inference and Control Frequency To achieve responsive real-time control, we accelerate the reverse difusion process during inference by employing a fast-sampling scheduler (DDIM), reducing the denoising steps to K = 10 for both the Difusion Policy and the Difusion World Model. Despite this acceleration, running the dual-difusion pipeline alongside high-fidelity 3D visual encoding, real-time 6D object tracking, and continuous OOD anomaly detection (D<sub>R−I</sub>) is computationally intensive. Consequently, our overall system control frequency naturally bounds to 2 Hz. Notably, this rate identically matches the frequency of our human teleoperation data collection, naturally aligning the model’s temporal execution with the expert demonstrations.

We intentionally accept this inference speed to prioritize the synchronous and stable execution of all heavy subsystems. This design choice directly aligns with our primary research objective: achieving reliable, highly accurate recovery from severe OOD disturbances, rather than optimizing for high-speed dynamic manipulation. Because our evaluated tasks (e.g., Stack Blocks, Pour Tea) demand precise spatial reasoning and alignment rather than split-second reactive motions, the 2 Hz frequency proves entirely suficient. Furthermore, the inherent temporal smoothing provided by our action chunking mechanism (N = 8 executed steps per cycle) perfectly bridges the gap between consecutive network queries, yielding smooth, robust, and physically stable closed-loop execution.

Visual Processing and Point Cloud Downsampling The primary visual observation is captured via a static Intel RealSense L515 camera. To achieve the required real-time processing speed, we implement a highly eficient two-stage point cloud downsampling pipeline:

![](images/aa081fe58eaeea4adda6a4d9a70fcf9c76004fbf5f0b4d43074bf3d705b79ef9.jpg)  
Fig. 11: Failure modes of the DP3 tracking ablation. Real-world execution sequences demonstrating how the baseline policy fails under severe Out-of-Distribution (OOD) spatial shifts, resulting in missing grasps, dropping objects, and imprecise pressing (pressing the box after pressing the button).

1. Random Cropping: The raw depth camera captures approximately 240, 000 points per frame. We first apply a random sub-sampling filter to drastically reduce this dense cloud to a manageable intermediate size of 10, 000 points.

2. Farthest Point Sampling (FPS): We then apply FPS (accelerated via PyTorch3D) to extract a structurally uniform and geographically representative final set of exactly 2, 048 points.

This optimized two-stage pipeline executes in just ∼ 15 ms on the GPU, successfully feeding high-quality geometric features into the 3D encoder without introducing bottleneck latency.

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 2: DDP Inference with Target-Specific Recovery

Input: Visual Encoder $E_{\psi}$, Policy $\pi_{\theta}$, World Model $\mathcal{W}_{\phi}$, Pose Estimator
Input: Subtasks $\mathcal{S} = \{S_1, \ldots, S_J\}$, Expected ID poses $\mathbf{p}^{exp}$, Threshold $\tau_{diff}$
1 OOD_Mode ← False, $\mathcal{C}_{disp} ← \emptyset$, $\mathbf{O}_{curr} ← E_{\psi}(X_{real}^0)$;
2 for subtask $S_j \in \mathcal{S}$ targeting object $C_j$ do
3 while completion trigger $\mathcal{T}_{end}(S_j)$ is False do
4 $\mathbf{O}_{real}^t ← E_{\psi}(X_{real}^t)$;
// 1. OOD Detection &amp; Tracking
5 if not OOD_Mode then
6 $\mathbf{O}_{pred}^t ← \mathcal{W}_{\phi}(\mathbf{O}_{real}^{t-1}, \mathbf{a}_{t-1})$;
7 $\mathcal{D}_{R-I}(t) ← ||\mathbf{O}_{real}^t - \mathbf{O}_{pred}^t||_2^2$;
8 if $\mathcal{D}_{R-I}(t) &gt; \tau_{diff}$ then
9 OOD_Mode ← True, $\mathcal{C}_{disp} ←$ Identify displaced objects;
10 if $C_j \in \mathcal{C}_{disp}$ then
11 Execute tracking action $\mathbf{a}_j^{track} = \mathbf{p}_{C_j} - \mathbf{p}_{C_j}^{exp}$;
12 $\mathbf{O}_{curr} ← \mathbf{O}_{pred}^t$; // Enter Imagination
13 else
14 $\mathbf{O}_{curr} ← \mathbf{O}_{real}^t$; // Nominal ID Reality
// 2. Action Execution &amp; Recursive Imagination
15 $\mathbf{a}_t \sim \pi_{\theta}(\cdot | \mathbf{O}_{curr})$, Execute action chunk $\mathbf{a}_t$;
16 if OOD_Mode then
17 $\mathbf{O}_{curr} ← \mathcal{W}_{\phi}(\mathbf{O}_{curr}, \mathbf{a}_t)$; // Autoregressive Forecast
18 $t ← t + 1$;
// 3. Target-Specific Recovery
19 if OOD_Mode and $C_j \in \mathcal{C}_{disp}$ then
20 Execute recovery action $\mathbf{a}_j^{rec} = \mathbf{p}_{C_j} - \mathbf{p}_{C_j}^{exp}$;
21 $\mathcal{C}_{disp} ← \mathcal{C}_{disp} \setminus \{C_j\}$;
// Re-evaluate ID State
22 $\mathcal{D}_{R-I}(t) ← ||E_{\psi}(X_{real}^t) - \mathcal{W}_{\phi}(\mathbf{O}_{curr}, \mathbf{a}_{t-1})||_2^2$;
23 if $\mathcal{D}_{R-I}(t) ≤ \tau_{diff}$ then
24 OOD_Mode ← False; // Return to Reality
</div>

Table 3: Detailed OOD Disturbance Parameters for MetaWorld Tasks. The displacement vector ∆P is represented as $[ d x , d y , d z ]$ in meters.

<table><tr><td>Task Name</td><td>Displaced Object(s)</td><td>Intervention</td><td>Timing Displacement Vector (ΔP)</td><td>Tracking Logic</td></tr><tr><td>Assembly</td><td>Peg (peg)</td><td>Chunk 6</td><td>[-0.25, 0.0, 0.0]</td><td>Single-Stage</td></tr><tr><td>Button Press</td><td>Box &amp; Button (box)</td><td>Chunk 3</td><td>[-0.25, 0.0, 0.0]</td><td>Single-Stage</td></tr><tr><td>Drawer Open</td><td>Drawer (drawer)</td><td>Chunk 2</td><td>[0.0, 0.0, +0.10]</td><td>Single-Stage</td></tr><tr><td>Hammer</td><td>Box &amp; Nail (box, nailHead)</td><td>Chunk 5</td><td>[0.0, +0.10, 0.0]</td><td>Single-Stage</td></tr><tr><td>Door Unlock</td><td>Door (door)</td><td>Chunk 2</td><td>[0.0, +0.10, 0.0]</td><td>Single-Stage</td></tr><tr><td>Soccer</td><td>Goal Net (goal_whole)</td><td>Chunk 2</td><td>[-0.20, +0.10, 0.0]</td><td>Single-Stage</td></tr><tr><td>Pick Place</td><td>Puck (obj)</td><td>Chunk 2</td><td>[-0.20, -0.10, 0.0]</td><td>Two-Stage (+ Safe Lift)</td></tr><tr><td>Push</td><td>Puck (obj)</td><td>Chunk 2</td><td>[-0.25, 0.0, 0.0]</td><td>Two-Stage</td></tr><tr><td>Sweep</td><td>Puck (obj)</td><td>Chunk 2</td><td>[+0.20, +0.10, 0.0]</td><td>Two-Stage</td></tr><tr><td>Coffee Pull</td><td>Mug &amp; Machine (obj, coffee_machine)</td><td>Chunk 2</td><td>[-0.20, 0.0, 0.0]</td><td>Two-Stage</td></tr></table>