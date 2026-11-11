# AdaWorldPolicy: World-Model-Driven Diffusion Policy with Online Adaptive Learning for Robotic Manipulation

Ge Yuan<sup>1</sup>, Qiyuan Qiao<sup>2</sup>, Jing Zhang<sup>2</sup>, Dong Xu<sup>1\*</sup>

gavinyuan97@gmail.com, qiaoqy@connect.hku.hk, zhang jing@buaa.edu.cn, dongxu@hku.hk <sup>1</sup>The University of Hong Kong <sup>2</sup>Beihang University

## Abstract

Effective robotic manipulation requires policies that can anticipate physical outcomes and adapt to real-world environments. In this work, we introduce a unified framework, World-Model-Driven Diffusion Policy with Online Adaptive Learning (AdaWorldPolicy) to enhance robotic manipulation under dynamic conditions with minimal human involvement. Our core insight is that world models provide strong supervision signals, enabling online adaptive learning in dynamic environments, which can be complemented by force-torque feedback to mitigate dynamic force shifts. Our AdaWorldPolicy integrates a world model, an action expert, and a force predictor—all implemented as interconnected Flow Matching Diffusion Transformers (DiT). They are interconnected via the multi-modal self-attention layers, enabling deepfeature exchangeforjoint learning while preserving their distinct modularity characteristics. Wefurther propose a novel Online Adaptive Learning (AdaOL) strategy that dynamically switches between an Action Generation mode and a Future Imagination mode to drive reactive updates across all three modules. This creates a powerful closed-loop mechanism that adapts to both visual andphysical domain shifts with minimal overhead. Across a suite ofsimulated and real-robot benchmarks, our AdaWorldPolicy achieves state-of-the-art performance, with dynamical adaptive capacity to out-of-distribution scenarios. Homepage: https://AdaWorldPolicy.github.io

## 1. Introduction

Robotic manipulation requires policies that can perceive, anticipate, and act reliably under contact-rich and dynamically changing conditions. Vision-Language-Action (VLA) models [6, 9, 23] attempt to meet this requirement by integrating robotic action into large-scale pretrained Multimodal Large Language Models [43, 44, 48]. However, they normally require large amounts of human demonstration data and struggle to generalize to unseen or dynamic environments in contact-rich tasks. Concurrently, researchers have explored how to integrate world models [1, 3–5] for robot manipulation by providing rich information about real-world dynamics and agent-environment interactions. However, many existing approaches only use the world model as a passive “digital twin” for offline evaluation [21, 50]. Another more integrated strategy attempts to create a unified architecture for both action generation and world prediction [10, 25, 51]. These methods typically adopt the offline training strategy that fails to dynamically adapt to real-world environmental changes. Therefore, how to fully exploit the world model’s potential for robotic manipulation in dynamic scenarios with minimal human efforts remains a challenging task.

![](images/e336338999c3a6625d77944d4d876dc111594edc40a862eedfa4eadafd70a347.jpg)  
Figure 1. An overview of our AdaWorldPolicy with Adaptive Online Learning (AdaOL). At timestep t, our AdaWorldPolicy Network operates in two modes. Mode I (Action Generation): AdaWorldPolicy network acts as an action policy generator $P ^ { \mathrm { p o l i c y } } ( a | o )$ , which takes the current multi-modal observation o (from static/gripper cameras and force sensor) to generate an action a. This action is then executed by the robot. During offline training, this step is supervised by the imitation loss $\mathcal { L } _ { 1 }$ (see Section 3.3). Mode II (Future Imagination): Subsequently, our AdaWorldPolicy network turns into an action-conditioned world model $P ^ { \mathrm { i m a g i n e } } ( o ^ { \prime } | o , a )$ which takes the same observation o and the executed action a to predict an Imagined Observation at timestep t + 1. The core of our AdaOL strategy lies in the online updating loop (red arrows). The discrepancy between the Imagined Observation and the real Observation at timestep t + 1 (e.g., under in-domain setup or under domain shifts like lighting or pose variations) is quantified by a prediction loss $\mathcal { L } _ { 2 }$ . This loss drives an online update to a small subset of shared network parameters, creating a closed-loop system that continuously adapts to real-world dynamics.

In this work, we enhance robot manipulation when facing environmental changes and dynamic force shifts in a reactive manner with minimal human involvement. Our key insight is that world models provide strong supervision signals for robotic manipulation and enable online adaptive learning in dynamic environments. Additionally, we find that force-torque feedback is valuable to reduce dynamic force shifts during real-world deployment on contact-rich tasks.

To this end, we introduce AdaWorldPolicy, a unified multi-modal framework consisting of three components: a world model, an action expert, and a force predictor. All three components are implemented as Flow Matching [28] Diffusion Transformers (DiT) [34]. The world model is built upon the state-of-the-art Cosmos Predict2 [3], while the action expert and force predictor are lightweight DiT models. These three modules operate in parallel and are interconnected via the multi-modal self-attention layers, enabling deep feature exchange while preserving their distinct computational pathways. The world model provides supervision signals for the action expert and also enables online adaptive learning in dynamic environments by identifying discrepancies between its action-conditioned predictions and actual real-world feedback. The force predictor addresses dynamic force shifts by minimizing the discrepancy between predicted and actual force readings. To jointly train the three modules, we propose a test-time online adaptive learning (AdaOL) strategy where AdaWorld-Policy switches between the following two modes. In Mode I (Action Generation), the action model generates actions based on the current observations. In Mode II (Future Imagination), two types of discrepancies including these between the world model’s action-conditioned predictions and actual real-world feedback, as well as these between predicted and actual force readings, drive self-supervised online updates for the network parameters in all three modules. These modules are updated in a reactive manner to achieve fast adaptation and closed-loop control in response to dynamically changing environments.

Our main contributions are summarized as follows:

• A unified multi-modal framework, AdaWorldPolicy. This framework fully exploits the potential of world models and force-torque feedback for robotic manipulation in contact-rich dynamic environments by synergistically modeling an action expert, the pre-trained world model, and the force predictor within a unified diffusion-based network architecture.

• A novel test-time online adaptive learning (AdaOL) strategy. The AdaOL strategy rapidly reduces both visual and physical domain shifts by updating the network parameters based on real-world feedback.

• State-of-the-art performance across diverse benchmarks. Our base AdaWorldPolicy framework achieves state-of-the-art results on a suite of benchmarks, including PushT [20], CALVIN [33], and LIBERO [29]. Furthermore, we demonstrate that our base AdaWorldPolicy with AdaOL improves out-of-distribution (OOD) performance by over 5%, meanwhile enhancing in-domain results by ∼1%. Real-world experiments validate the effectiveness of both AdaOL and the force-aware extension in completing challenging manipulation tasks.

## 2. Related Work

World Models for Robotic Control. Robotic policy learning methods are broadly divided into three paradigms. The first one is based on model-free, end-to-end visuomotor learning, where transformer-based models ingest raw visual (and sometimes language) input and directly output robot actions. For example, generalist systems are trained on vast teleoperation datasets [8, 9, 23, 45]. Although these agents perform well in multi-task scenarios through large-scale pre-training, they lack the capacity for explicit physical or dynamic reasoning, which degrades their performance in novel or out-of-distribution environments.

The second paradigm is based on model-based learning. Rather than mapping observations directly to actions, agents learn an internal world model that predicts environment dynamics, enabling planning, policy optimization, or latent imagination [15, 16, 40]. While these methods offer stronger physical grounding and better use of training data, traditionally they have been applied in narrower domains or simulated settings.

More recently, hybrid approaches are emerging: world models are either used as external validators over large pretrained modules (e.g., large vision-language models) [50], or they are tightly integrated into unified architectures that combine world modeling, perception and policy in one system [10, 25]. In contrast, our work belongs to this hybrid paradigm: we propose a novel fusion mechanism in which the world model actively supervises policy learning and also corrects its output, rather than passively validating it.

Diffusion Models for Decision Making. In parallel, diffusion model-based approaches have been used for decision-making and control. Early works such as Diffusion Policy and its extensions model robot trajectories via diffusion processes [2, 12, 20]. These models outperform traditional behavior cloning methods in many multi-modal settings. However, a major drawback is that they often ignore dynamics and outcome modeling. Specifically, because they are trained via behavior cloning, they may generate actions that seem plausible but are physically inconsistent. More recent works aim to address this issue by introducing hierarchical generation strategies, kinematics-aware constraints, and contact-aware trajectory rollout [7, 32]. Following this research trend, we introduce a world model within the diffusion process to explicitly impose physical consistency while using model-predicted rollouts to generate better actions.

Online Adaptation for Robotics. Deploying robots in real-world, dynamic, and previously unseen environments is challenging due to distribution mismatch between training and testing processes. Test-time adaptation (TTA) provides a mechanism for agents to adjust after deployment based on unlabeled inputs. In vision and perception domains, TTA methods based on self-supervised objectives or memory buffers have shown promising results [27, 42, 46]. For large models, parameter-efficient fine-tuning (e.g., LoRA) enables rapid updates during inference with minimal overhead [17]. More recently, robotics-specific TTA approaches have been proposed, including plug-and-play transformer modules for adaptation [11], low-rank adaptation based on confidence maximization [19], and embodied adaptation for real robotic grasping or navigation tasks [30]. Our Adaptive Online Learning (AdaOL) strategy extends these works by using prediction error from the improved world model powered module as a self-supervised signal, enabling fast, physically grounded adaptation to both visual and dynamic force shifts.

In summary, compared to existing works, our AdaWorld-Policy introduces a unified diffusion-based framework where the world model, action expert, and force/contact predictor are deeply integrated through a multi-modal selfattention mechanism. Rather than simply validating or predicting, the world model becomes part of the learning loop—its prediction error serves as a self-supervised adaptation signal for correction of both visual and dynamic force shifts.

## 3. Methodology

## 3.1. Problem Formulation

We address the problem of learning a goal-conditioned robotic manipulation policy that can rapidly adapt to novel dynamics at test time without requiring new human demonstrations. Formally, at each timestep t, the input is a multi-modal observation history $o \ = \ \{ x _ { \mathrm { s t a t i c } } , x _ { \mathrm { g r i p p e r } } , f \}$ which includes image sequences from a static camera x<sub>static,</sub> $\mathrm { , } t - T _ { \mathrm { c } } \mathrm { + } 1 { : } t$ and a gripper camera $x _ { \mathrm { g r i p p e r } , t - T _ { \mathrm { c } } + 1 : t } ,$ , and force-torque sensor readings $f _ { t - T _ { \mathrm { c } } + 1 : t }$ , all sharing the same context length $T _ { \mathrm { c } }$ . The goal is to generate a sequence of future actions $a = a _ { t : t + T _ { \mathrm { a } } - 1 }$ with horizon $T _ { \mathbf { a } }$ that successfully completes the task.

The core challenge is test-time adaptation to environmental changes and dynamic force shifts. The policy is initially trained offline on a static dataset collected in the source domain. When deployed in a new test environment with different dynamics, the initial parameters $\theta _ { 0 }$ may be suboptimal. The goal is to devise a policy that leverages its own interaction experience at test time, $\{ \left( o _ { t } , a _ { t } , o _ { t + 1 } \right) \} _ { t = 0 } ^ { T } ,$ to update its parameters from $\theta _ { t }$ to $\theta _ { t + 1 }$ . This update mechanism must be self-supervised, enabling the policy to improve over time in the new environment without external labels or human intervention.

## 3.2. Method Overview

To address this challenge, we introduce AdaWorldPolicy (AWP), a framework designed for reactive, self-supervised online adaptation in novel environments. Our key insight is to transform the world model from a passive predictor into an active supervisor that drives closed-loop adaptation.

As illustrated on the left of Figure 2, our framework is comprised of three parallel components, all implemented as Diffusion Transformers (DiTs) [34] trained through Flow Matching loss [28]:

• A foundational World Model, built upon the powerful, pretrained Cosmos-Predict2 [3], which is responsible for predicting future visual states $( x _ { \mathrm { s t a t i c } } ^ { \prime } , x _ { \mathrm { g r i p p e r } } ^ { \prime } )$

• A lightweight Force Predictor, which complements the visually-focused World Model by extending the system’s predictive capabilities into physical dynamics, anticipating future interaction forces $( f ^ { \prime } )$

• A lightweight Action Model, which serves as the core policy for generating robot actions (a).

These modules are interconnected via Multi-modal Self-Attention (MMSA) [14] to enable deep feature exchange. The entire framework is conditioned on shared inputs—including text embeddings $v _ { \mathrm { t e x t } }$ extracted by T5- XXL [37], robot state vector q extracted by an MLP, and the diffusion noise level $\sigma$ (also processed through an MLP)—which are added and injected into each of the three backbones through adaLN [34] layers. This unified architecture operates in two distinct modes, determined by the role of the Action Model. In Mode I: Action Generation, it generates an action $a _ { t }$ from Gaussian noise. In Mode II: Future Imagination, it takes an action $a _ { t }$ as input to predict the future state. This dual-mode capability is fundamental to our closed-loop online adaptive learning (AdaOL) strategy, allowing the agent to imagine the consequences of its actions and then correct itself based on real-world outcomes.

In the following sections, we will first detail the World-

![](images/e765f3e62abbd5fe3ad1ca652e276eef503ba141768b5445ddc6b6540b2f1bc3.jpg)  
Figure 2. Network architecture and workflow of AdaWorldPolicy. Our unified multi-modal framework builds upon a shared multimodal transformer backbone. It synergistically integrates three modules: a World Model for visual prediction, a Force Predictor for physical dynamics modeling, and an Action Model for policy generation. All modules are implemented as Flow Matching Diffusion Transformers (DiT) and interact through a shared Multi-modal Self-attention layer. Input modalities (vision, action, force, text) are first encoded, conditioned with global features (text, state, noise level) via the adaLN module, and then processed by the shared multi-modal self-attention layer. In our framework, the operational mode is determined by a switch on the action input: in Mode I (Action Generation), the action token is provided as noise for the model to generate an action; in Mode II (Future Imagination), a known action is provided as a condition for future prediction. A LoRA-based mechanism enables efficient online updates of a small set of parameters.

Model-Driven Diffusion Policy, then describe its extension to force feedback, and finally explain the closed-loop adaptation mechanism.

## 3.3. World-Model-Driven Diffusion Policy

World Model. Our World Model builds upon the pretrained Cosmos-Predict2 [3], which we extend to support multi-view video by concatenating the VAE [1] tokens from each camera view along the temporal dimension. To maintain spatial and temporal coherence across views, we assign each view’s token sequence an independent Rotary Position Embedding (RoPE) [41]. Input tokens are paired with a binary mask where ‘1’ indicates a known condition (e.g., $x _ { \mathrm { s t a t i c } } , x _ { \mathrm { s t a t i c } } , f )$ and $\cdot _ { 0 } \cdot \mathrm { \ }$ indicates a target for prediction (e.g., a noised version of $x _ { \mathrm { s t a t i c } } ^ { \prime } , x _ { \mathrm { g r i p p e r } } ^ { \prime } )$ . This mask not only helps the model distinguish between conditional inputs and prediction targets, but is also used after the final denoising step to replace the generated conditional parts with the original inputs, ensuring they are perfectly preserved.

Force Predictor. To enhance the agent’s capability in contact-rich scenarios, we introduce a dedicated Force Predictor. This module complements the visually-focused

World Model by extending the system’s predictive power to physical dynamics. It is implemented as a lightweight DiT, structurally similar to the World Model but with significantly fewer parameters (0.4B only). Its role is to predict future force-torque readings $( f ^ { \prime } )$ based on the current state and action, providing crucial information for tasks involving physical interaction.

Action Model and Dual-Mode Operation. The Action Model is a lightweight DiT responsible for action generation. Crucially, its input determines the operational mode of the entire AWP framework, a “switch” implemented via input masking:

• Mode I: Action Generation. To generate an action, the action token input is pure noise, and its corresponding mask is set to all zeros. The model denoises this token using the given observations, effectively computing $a _ { t }$ from $o _ { t } .$ , which is formulated as:

$$
\mathcal {L} _ {1} (\theta) = \mathbb {E} \left[ \| \mathbf {u} _ {\theta} (a _ {k}, k, o; \theta) - \mathbf {v} _ {k} (a _ {k}, a) \| ^ {2} \right],\tag{1}
$$

where $a _ { k }$ denotes the noised version of a and $v _ { k }$ is the target vector field defined in Flow Matching [28].

• Mode II: Future Imagination. To predict the future, a known, concrete action a is provided as a condition, and its mask is set to all ones. This conditions the World Model (and Force Predictor) to imagine the future state based on the action, computing $\hat { o } ^ { \prime }$ from $o , a ,$ , which can be formulated as:

$$
\mathcal {L} _ {2} (\theta) = \mathbb {E} \left[ \left\| \mathbf {u} _ {\theta} \left(o _ {k} ^ {\prime}, k, o, a; \theta\right) - \mathbf {v} _ {k} \left(o _ {k} ^ {\prime}, o ^ {\prime}\right) \right\| ^ {2} \right],\tag{2}
$$

where $o _ { k } ^ { \prime }$ is a noised version of $o ^ { \prime }$ .

This dual-mode capability is fundamental to our online adaptation strategy, as it allows the agent to first act, then imagine the consequences, and finally compare imagination with reality. Detailed input and output of our unified model in two different modes are shown in Figure 3.

Multi-modal Self-Attention. While the three backbones have different functions, they exchange information via MMSA layers. Unlike concatenation, which forces features into a shared space, MMSA allows each module to query information from others while maintaining its own specialized representations. This enables flexible allocation of model capacity (e.g., a large World Model, a small Action Model) and prevents feature corruption. The attention is computed as:

$$
\begin{array}{c} \text {MMSA} (Q, K, V) = A ([ Q _ {x}, Q _ {f}, Q _ {a} ], \\ [ K _ {x}, K _ {f}, K _ {a} ], \\ [ V _ {x}, V _ {f}, V _ {a} ]), \end{array}\tag{3}
$$

where A denotes self-attention operation, [·] indicates token-wise concatenation, $Q _ { x } , K _ { x } , V _ { x }$ are the query, key, and value from the World Model, and similarly for the Force Predictor (f) and Action Model (a).

Joint Training. The framework is trained end-to-end with a single Flow Matching loss [28]. During training, we randomly switch between the two operational modes. With probability $p _ { \mathrm { a } } ,$ we train in Action Generation mode; otherwise, we train in Future Imagination mode. The total loss is the weighted sum of the losses from both modes:

$$
\mathcal {L} _ {\mathrm{total}} (\theta) = p _ {\mathrm{a}} \cdot L _ {1} + (1 - p _ {\mathrm{a}}) \cdot L _ {2},\tag{4}
$$

where θ represents all trainable parameters, $\mathbf { u } _ { \theta }$ is the model’s predicted vector field, and $\mathbf { v } _ { k }$ is the target vector field at noise step k as defined by Flow Matching. The first term corresponds to the Action Generation loss, where the model learns to predict the vector field for a noised action $a _ { k }$ conditioned on the current observation $o .$ The second term is the Future Imagination loss, where the model predicts the vector field for a noised future observation $o _ { k } ^ { \prime }$ conditioned on both the current observation o and the action a. This joint objective enables the modules to develop a shared understanding of world dynamics, which is the key to enabling our Online Adaptive Learning at test time.

![](images/b39c6777533505f356e78dfff11f231810584a45a5a8331ec32e4b9a8c5524fc.jpg)  
Figure 3. Input and output details of our unified model in two different modes. Mode I (Action Generation): The model takes an observation history o (e.g., context length $T _ { c } ~ = ~ 5 )$ and predicts a future action sequence $\{ a _ { t } , a _ { t + 1 } , \cdot \cdot \cdot , a _ { t + T _ { \mathrm { a } } } \}$ (e.g., action horizon $T _ { a } = 8 )$ . At test time, the robot executes this predicted action sequence. Mode II (Future Imagination): The model is conditioned on both the observation history o and a ground-truth action sequence a, then predicts the corresponding future observation sequence $\hat { o } ^ { \prime }$ . The discrepancy between this prediction and real environmental feedback is used to update the network parameters during our Adaptive Online Learning (AdaOL) phase.

## 3.4. Closed-loop Online Adaptive Learning

While offline pre-training provides a strong prior, realworld environments inevitably present visual and physical domain shifts that can degrade performance. To bridge this gap, we introduce Online Adaptive Learning (AdaOL), a closed-loop strategy that enables the agent to reactively selfcorrect using real-world feedback.

The AdaOL strategy operates as a tight, closed-loop cycle following each action execution, as illustrated in Figure 1. The process unfolds in the following steps:

1. Action Generation: At timestep t, our AWP runs in action generation mode which takes the current observation $o _ { t }$ as input and generate action $a _ { t }$

2. Execution: The robot executes the action $a _ { t }$ in the test environment.

3. Real-world Feedback: The robot observes the true outcome from the environment, the real future state $o _ { t + 1 }$

4. Future Imagination: Concurrently, our AWP runs in future imagination mode which predicts the future state $\hat { o } _ { t + 1 }$ based on the same observation $o _ { t }$ and the executed action $a _ { t } .$

5. Loss and Update: The core of AdaOL lies in leveraging the discrepancy between reality and prediction. A loss is computed based on the prediction error, typically in a latent space provided by a VAE encoder [3]:

$$
\mathcal {L} _ {\mathrm{AdaOL}} = \left\| E (o _ {t + 1}) - E (\hat {o} _ {t + 1}) \right\| _ {2} ^ {2},\tag{5}
$$

where $E ( \cdot )$ is the encoder. This loss drives an online update by producing a corrective gradient $\Delta w$

This cycle allows the agent to constantly ground its internal world model (and force predictor) in reality, correcting for any drift or model inaccuracy before the next iteration begins.

![](images/be1c2360d4e4de4e96ee1663ba58467d53099859bf80f22cbda8de6c141ea8dd.jpg)  
Figure 4. Visualizations of the simulated benchmarks used in our experiments: Variant PushT for out-of-distribution robustness, LIBERO for long-horizon skills, and the CALVIN benchmark for language-conditioned tasks across different domains.

To make this online adaptation computationally feasible, we employ Low-Rank Adaptation (LoRA) [17] for the parameter updates. Instead of backpropagating through the entire multi-billion parameter model, the gradients from ${ \mathcal { L } } _ { \mathrm { A d a O I } }$ are only used to update only a small set (less than 0.1%) of trainable low-rank matrices. This approach significantly reduces the computational and memory overhead at each update step, making the online adaptation practical for deployment on resource-constrained real robots.

## 4. Experiments

## 4.1. Experimental Setup

Simulated Benchmarks. As illustrated in Figure 4, we evaluate our method on three diverse simulated benchmarks. (1) LIBERO-10 [29]: A benchmark for longhorizon, compositional skills across various household scenarios. It tests the policy’s ability to handle complex task sequences. (2) Variant PushT [20]: We modify the classic PushT task to specifically test out-of-distribution (OOD) robustness. After training on the original setting, we evaluate the policy’s ability to adapt to test-time variations in background texture, random time-varying lighting, and random time-varying color. Task success is measured by the Intersection over Union (IoU) between the goal region and the final object position. (3) CALVIN [33]: The CALVIN benchmark consists of four environments, A, B, C, and D, each belonging to different domains. Each domain provides 6 hours of human teleoperated data across 34 different tasks. To validate the cross-domain adaptive ability of methods, we only use the ABC→D evaluation protocol. In each sequence, the robot is required to continuously solve five tasks in a row.

Real-World Setup. Our real-world experiments are conducted using a 6-DoF robot arm equipped with a gripper camera and a wrist-mounted force-torque sensor, supplemented by a static third-person camera (Figure 5). For offline training, we collected a dataset of human demonstrations via teleoperation with a PS5 controller. The controller provided haptic feedback by mapping force sensor readings to vibrations, allowing the operator to feel the interaction forces.

Table 1. Comparison of success rates on the LIBERO-10 benchmark, under two settings: with only a static camera, and with full multi-modal inputs. Note: AWP is the short name of our method AdaWorldPolicy without online adaptive learning.

<table><tr><td>Methods</td><td>Pub.</td><td>Static Camera</td><td>Gripper Camera</td><td>Joint States</td><td>Success</td></tr><tr><td>UVA [25]</td><td>RSS&#x27;25</td><td>√</td><td></td><td></td><td>0.89</td></tr><tr><td>AWP</td><td>Ours</td><td>√</td><td></td><td></td><td>0.91</td></tr><tr><td>OpenVLA [23]</td><td>CoRL&#x27;24</td><td>√</td><td>√</td><td>√</td><td>0.54</td></tr><tr><td>SpatialVLA [36]</td><td>RSS&#x27;25</td><td>√</td><td>√</td><td>√</td><td>0.56</td></tr><tr><td>Pi0-fast [35]</td><td>RSS&#x27;25</td><td>√</td><td>√</td><td>√</td><td>0.60</td></tr><tr><td>FlowVLA [49]</td><td>arXiv&#x27;25</td><td>√</td><td>√</td><td>√</td><td>0.73</td></tr><tr><td>MODE [39]</td><td>NIPS&#x27;24</td><td>√</td><td>√</td><td>√</td><td>0.94</td></tr><tr><td>OpenVLA-OFT [22]</td><td>RSS&#x27;25</td><td>√</td><td>√</td><td>√</td><td>0.94</td></tr><tr><td>AWP</td><td>Ours</td><td>√</td><td>√</td><td>√</td><td>0.96</td></tr></table>

We evaluate our method on four challenging, longhorizon tasks requiring both precise motion and rich physical interaction: (1) sweeping coffee beans into a dustpan, (2) picking and placing two eggs into a box, (3) pouring water from a measuring cup, and (4) wiping a whiteboard. To specifically test our online adaptive learning strategy, we introduce various domain shifts during deployment. These include visual perturbations (e.g., changing lighting, tablecloth textures) and physical perturbations (e.g., substituting objects with different weights, altering the whiteboard’s incline). Further experimental details are available in the appendix.

## 4.2. Implementation Details

We implement our model in PyTorch, building upon the publicly available Cosmos-Predict2 architecture [3]. Our full AdaWorldPolicy model consists of a 2B parameter world model and two 0.4B parameter DiTs for action and force prediction. For simulator benchmarks without force data, we remove the force predictor.

Offline training. For the offline training stage, we largely follow the training recipe of Cosmos-Predict2. We train the model using the AdamW optimizer [31] on 8 A100 (80GB) GPUs. To maximize hardware utilization, we use the largest possible per-GPU batch size, resulting in a global batch size that varies between 64 and 256 across different datasets. The initial learning rate is set to $1 \times 1 0 ^ { - 4 }$ . Once the training loss plateaus, we apply a linear decay schedule for a maximum of 20k steps, reducing the learning rate to 1% of its initial value.

Online Learning. We perform test-time adaptation on a single NVIDIA RTX 5880 (48GB) GPU. To keep the update process lightweight, we employ LoRA with rank 16, applying adaptable matrices only to the first 4 layers of each backbone. For each incoming data sample (effective batch size of 1), we perform two gradient descent steps with a small, constant learning rate of $5 \times 1 0 ^ { - 7 }$ . This targeted, lightweight update strategy minimizes computational overhead. As a result, the average inference speed with TTA enabled is only approximately 5% slower than the baseline without adaptation.

![](images/cf0954eda4cf25401d35c3404c3e71c408edc62da92ca879502785e448b5f563.jpg)  
Figure 5. Overview of our real-robot evaluation. Our experimental setup (center) features an INOVO robotic arm with multi-modal sensing capabilities, including gripper/static cameras and a force sensor. We test our AdaWorldPolicy on four diverse manipulation tasks (left), such as sweeping beans and placing eggs. To specifically evaluate the effectiveness of our AdaOL strategy, we introduce a variety of challenging out-of-distribution (OOD) shifts during execution (right). These include visual perturbations like drastic lighting and texture changes, as well as physical perturbations like swapping objects and altering the workspace geometry (e.g., tilting the whiteboard).

Further details on the network architecture, data processing pipelines for each benchmark, and a complete list of hyperparameters can be found in the Appendix.

## 4.3. Main Results

LIBERO-10 Results. As shown in Table 1, our base AWP architecture achieves a new state-of-the-art results on the LIBERO-10 benchmark. It outperforms prior methods like UVA and MODE in both unimodal (static camera) and full multi-modal settings, demonstrating the strength of our MMSA-based design even without online adaptation.

Variant PushT Results. The Variant PushT benchmark (Table 2) highlights the critical role of our online adaptation. While the offline-trained AWP degrades significantly under visual domain shifts, activating online learning (AWP (ol)) consistently and substantially recovers performance. This makes AWP (ol) the top-performing method across all challenging out-of-distribution scenarios, demonstrating the effectiveness of our TTA mechanism.

CALVIN Results. On the long-horizon CALVIN benchmark (Table 3), our base AWP model again achieves stateof-the-art results, outperforming all prior methods in longsequence task completion. Enabling online learning provides a further consistent performance boost, demonstrating that our TTA strategy can effectively fine-tune an already well-generalized policy.

Performance in Real-World Experiments. We validate our approach on a physical robot across four challenging long-horizon tasks, testing it to various out-of-distribution (OOD) shifts at test time. The results are summarized in Figure 6. In the original in-domain environment (left), our offline-trained AWP model already establishes a strong performance, outperforming the DP-Force and UVA baselines on most tasks.

Table 2. Performance on the Variant PushT benchmark under different distribution shifts. We report the mean Intersection-over-Union (IoU) results. Note: AWP/AWP (ol) is the short name of our method AdaWorldPolicy without/with adaptive online learning.

<table><tr><td>Methods</td><td>Pub.</td><td>Original</td><td>Texture</td><td>Rand Light</td><td>Rand Color</td></tr><tr><td>DP [12]</td><td>IJRR&#x27;24</td><td>0.78</td><td>0.18</td><td>0.14</td><td>0.11</td></tr><tr><td>OpenVLA [23]</td><td>CoRL&#x27;24</td><td>0.35</td><td>0.22</td><td>0.20</td><td>0.14</td></tr><tr><td>UniPi [13]</td><td>NIPS&#x27;23</td><td>0.42</td><td>0.35</td><td>0.33</td><td>0.18</td></tr><tr><td>UVA [25]</td><td>RSS&#x27;25</td><td>0.94</td><td>0.11</td><td>0.54</td><td>0.13</td></tr><tr><td>AWP</td><td>Ours</td><td>0.97</td><td>0.47</td><td>0.71</td><td>0.61</td></tr><tr><td>AWP (ol)</td><td>Ours</td><td>0.98</td><td>0.51</td><td>0.77</td><td>0.66</td></tr></table>

Table 3. Performance on the CALVIN benchmark for the crossdomain setting (ABC for training, D for evaluation). We report the success rate (%) for different task lengths (the number of consecutive instructions) and the average length of successfully completed sub-sequences (Avg. Len.). We compare with the baseline methods without pretraining on any extra human demonstration data outside the benchmark. Note: AWP/AWP (ol) is the short name of our method AdaWorldPolicy without/with adaptive online learning.

<table><tr><td rowspan="2">Methods</td><td rowspan="2">Pub.</td><td colspan="5">Success Rate (%) for Task Length</td><td rowspan="2">Avg. Len.</td></tr><tr><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td></tr><tr><td>DP [12]</td><td>IJRR&#x27;24</td><td>62.2</td><td>30.9</td><td>13.2</td><td>5.0</td><td>1.6</td><td> $1.13 \pm 0.02$ </td></tr><tr><td>MDT [38]</td><td>RSS&#x27;24</td><td>61.7</td><td>40.6</td><td>23.8</td><td>14.7</td><td>8.7</td><td> $1.54 \pm 0.04$ </td></tr><tr><td>RoboFlamingo [26]</td><td>ICLR&#x27;24</td><td>82.4</td><td>61.9</td><td>46.6</td><td>33.1</td><td>23.5</td><td> $2.47 \pm 0.00$ </td></tr><tr><td>GR-1 [47]</td><td>ICLR&#x27;24</td><td>85.4</td><td>71.2</td><td>59.6</td><td>49.7</td><td>40.1</td><td> $3.06 \pm 0.00$ </td></tr><tr><td>OpenVLA [23]</td><td>CoRL&#x27;24</td><td>91.3</td><td>77.8</td><td>62.0</td><td>52.1</td><td>43.5</td><td> $3.27 \pm 0.00$ </td></tr><tr><td>MoDE [39]</td><td>ICLR&#x27;25</td><td>91.5</td><td>79.2</td><td>67.3</td><td>55.8</td><td>45.3</td><td> $3.39 \pm 0.03$ </td></tr><tr><td>GR-MG [24]</td><td>RAL&#x27;25</td><td>91.0</td><td>79.1</td><td>67.8</td><td>56.9</td><td>47.7</td><td> $3.42 \pm 0.28$ </td></tr><tr><td>AWP</td><td>Ours</td><td>91.8</td><td>79.2</td><td>68.5</td><td>62.8</td><td>48.0</td><td> $3.51 \pm 0.03$ </td></tr><tr><td>AWP (ol)</td><td>Ours</td><td>92.0</td><td>79.6</td><td>68.6</td><td>63.0</td><td>48.0</td><td> $3.54 \pm 0.04$ </td></tr></table>

The critical advantage of our method emerges under the four OOD scenarios (right). While the performance of all methods degrades under these shifts, our full model with online learning, AWP (ol), consistently and significantly outperforms its offline-only counterpart (AWP) and all other baselines. For instance, under the “Object Change” shift, our AWP (ol) improves the success rate on the “Pour” task from 80% to 90%. This consistent performance gain across all tasks and perturbation types validates the effectiveness of our Test-Time Adaptation strategy. The entire closedloop process, including action generation, online updating, and device latency, runs at an average of 4Hz.

![](images/8422c6e0b9bb7e9a420cb7e735c8a1b70ed634292ba95c65d1bd7829eec23e6e.jpg)  
Original Environment

![](images/6bb19474979add93b7baed5cc9299cb505ff7d0f6c592a2f4b6da86d02437ee1.jpg)  
Shift A: Change Tablecloth

![](images/1588dc28405bd226aa1ca674a5dba7768ef727f1d5b566631f83f891700b48cc.jpg)

![](images/ccbf750d1994c6c5d0badf729531274fec7b1c8ffd882f4df39078d78baec4ca.jpg)  
Shift C: Change Objects in the Task

![](images/af0d3cc0775062862d35395cdb29d8406e9504bce93cfa5173b87acb0fc86c1e.jpg)  
Shift D: Add Random Lighting  
Figure 6. Real-world evaluation results under domain shifts. Our base model, AdaWorldPolicy (or AWP in short), shows a strong performance in the original environment (left). When suffering from various visual and physical shifts at test time (right), our full mode with online adaptive learning, AWP (ol), consistently and significantly improves the success rate, showcasing robust online adaptation.

Table 4. Ablation study on the main components of AdaWorldPolicy. We report the average success rate (%) across four real-world in-domain tasks. Our full method with online adaptive learning achieves the best performance, while removing or replacing any key component leads to a significant performance drop, which validates our design choices.

<table><tr><td>Configuration</td><td>Success Rate (%)</td></tr><tr><td>AdaWorldPolicy w/ AdaOL (Full)</td><td>76.3</td></tr><tr><td>AdaWorldPolicy w/o AdaOL</td><td>72.5</td></tr><tr><td>w/o Force Predictor</td><td>53.8</td></tr><tr><td>w/o World Model Supervision</td><td>46.3</td></tr><tr><td>MMSA → Concatenation</td><td>36.3</td></tr><tr><td>MMSA → Cross-Attention</td><td>50.0</td></tr></table>

## 4.4. Ablation Studies

We conduct ablation studies to validate our design choices, with results reported as average success rates across four real-world in-domain tasks in Table 4. Our full model, AdaWorldPolicy with AdaOL, achieves the highest score of 76.3%. Disabling test-time online learning (AdaWorld-

Policy w/o AdaOL) reduces the success rate to 72.5%, confirming the value of continuous adaptation. Removing the Force Predictor module causes a significant drop to 53.8%, highlighting the necessity of modeling physical contact dynamics. The most critical component is the world model’s supervision without its training loss, the framework degenerates into a behavioral cloning policy, and performance plummets to 46.3%. This validates our core premise of using the world model as an active supervisor. Finally, replacing our Multi-modal Self-Attention (MMSA) layers with simpler fusion methods, such as concatenation (36.3%) or cross-attention (50.0%), severely degrades performance, confirming that MMSA is superior for integrating the different modules while preserving their specialized representations.

## 5. Conclusion

We have introduced AdaWorldPolicy, a unified framework for robotic manipulation that integrates a world-modeldriven diffusion policy with a novel online adaptive learning strategy (AdaOL). This strategy uses prediction errors from the improved world model as a self-supervised signal, driving efficient LoRA-based updates to continuously reduce visual and physical domain shifts. Extensive experiments show that our AdaWorldPolicy with online adaptive learning significantly outperforms strong non-adaptive baselines under out-of-distribution conditions in both simulation and real-world settings, achieving robust performance with minimal computational latency. Future work will extend this adaptation mechanism to further address long-horizon planning failures and scale it to larger network architectures.

## References

[1] Niket Agarwal, Arslan Ali, Maciej Bala, Yogesh Balaji, Erik Barker, Tiffany Cai, Prithvijit Chattopadhyay, Yongxin Chen, Yin Cui, Yifan Ding, et al. Cosmos world foundation model platform for physical ai. arXiv preprint arXiv:2501.03575, 2025. 1, 4

[2] Anurag Ajay, Yilun Du, Abhi Gupta, Joshua Tenenbaum, Tommi Jaakkola, and Pulkit Agrawal. Is conditional generative modeling all you need for decision-making? In ICLR, 2023. 2

[3] Arslan Ali, Junjie Bai, Maciej Bala, Yogesh Balaji, Aaron Blakeman, Tiffany Cai, Jiaxin Cao, Tianshi Cao, Elizabeth Cha, Yu-Wei Chao, et al. World simulation with video foundation models for physical ai. arXiv preprint arXiv:2511.00062, 2025. 1, 2, 3, 4, 5, 6, 11, 13, 14, 15

[4] Mido Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Mojtaba Komeili, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, Sergio Arnaud, Abha Gejji, Ada Martin, Francois Robert Hogan, Daniel Dugas, Piotr Bojanowski, Vasil Khalidov, Patrick Labatut, Francisco Massa, Marc Szafraniec, Kapil Krishnakumar, Yong Li, Xiaodong Ma, Sarath Chandar, Franziska Meier, Yann LeCun, Michael Rabbat, and Nicolas Ballas. V-jepa 2: Self-supervised video models enable understanding, prediction and planning. arXiv preprint arXiv:2506.09985, 2025.

[5] Amir Bar, Gaoyue Zhou, Danny Tran, Trevor Darrell, and Yann LeCun. Navigation world models. In CVPR, 2025. 1

[6] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. π0: A visionlanguage-action flow model for general robot control. corr, abs/2410.24164, 2024. doi: 10.48550. In Robotics: Science and Systems, 2025. 1

[7] Jean-Baptiste Bouvier, Kanghyun Ryu, Kartik Nagpal, Qiayuan Liao, Koushil Sreenath, and Negar Mehr. Ddat: Diffusion policies enforcing dynamically admissible robot trajectories. In arXiv preprint arXiv:2502.15043, 2025. 3

[8] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, et al. RT-1: Robotics transformer for real-world control at scale. arXiv preprint arXiv:2212.06817, 2022. 2

[9] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, et al. RT-2: Vision-language-action models transfer web knowledge to robotic control. arXiv preprint arXiv:2307.15818, 2023. 1, 2

[10] Jun Cen, Chaohui Yu, Hangjie Yuan, Yuming Jiang, Siteng Huang, Jiayan Guo, Xin Li, Yibing Song, Hao Luo, Fan Wang, et al. Worldvla: Towards autoregressive action world model. arXiv preprint arXiv:2506.21539, 2025. 2

[11] Xiangyu Chang, Sk Miraj Ahmed, Srikanth V. Krishnamurthy, Basak Guler, Ananthram Swami, and Amit K. Roy-Chowdhury. Plug-and-play transformer modules

for test-time adaptation (pluto). In arXiv preprint arXiv:2401.04130v3, 2024. 3

[12] Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song. Diffusion Policy: Visuomotor policy learning via action diffusion. The International Journal of Robotics Research, page 02783649241273668, 2024. 2, 7

[13] Yilun Du, Mengjiao Yang, Bo Dai, Hanjun Dai, Ofir Nachum, Joshua B Tenenbaum, Dale Schuurmans, and Pieter Abbeel. Learning universal policies via text-guided video generation. In NIPS, pages arXiv–2302, 2023. 7

[14] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Muller, Harry Saini, Yam Levi, Dominik¨ Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In ICML, 2024. 3, 14

[15] Danijar Hafner, Timothy Lillicrap, Jimmy Ba, and Mohammad Norouzi. Dream to control: Learning behaviors by latent imagination. In ICLR, 2020. 2

[16] Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, and Timothy Lillicrap. Mastering diverse control tasks through world models. Nature, pages 1–7, 2025. 2

[17] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. In ICLR, 2022. 3, 6

[18] Xun Huang and Serge Belongie. Arbitrary style transfer in real-time with adaptive instance normalization. In ICCV, pages 1501–1510, 2017. 14

[19] Muntasir Imam et al. Test-time low rank adaptation via confidence maximization for zero-shot generalization (ttl). In WACV, 2025. 3

[20] Michael Janner, Yilun Du, Joshua B Tenenbaum, and Sergey Levine. Planning with diffusion for flexible behavior synthesis. arXiv preprint arXiv:2205.09991, 2022. 2, 6, 11, 12

[21] Yuxin Jiang, Shengcong Chen, Siyuan Huang, Liliang Chen, Pengfei Zhou, Yue Liao, Xindong He, Chiming Liu, Hongsheng Li, Maoqing Yao, et al. Enerverse-ac: Envisioning embodied environments with action condition. In Robotics: Science and Systems, 2025. 2

[22] Moo Jin Kim, Chelsea Finn, and Percy Liang. Fine-tuning vision-language-action models: Optimizing speed and success. In Robotics: Science and Systems, 2025. 6

[23] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan P Foster, Pannag R Sanketi, Quan Vuong, Thomas Kollar, Benjamin Burchfiel, Russ Tedrake, Dorsa Sadigh, Sergey Levine, Percy Liang, and Chelsea Finn. Openvla: An opensource vision-language-action model. In Proceedings ofThe 8th Conference on Robot Learning, pages 2679–2713, 2025. 1, 2, 6, 7

[24] Peiyan Li, Hongtao Wu, Yan Huang, Chilam Cheang, Liang Wang, and Tao Kong. GR-MG: Leveraging partially annotated data via multi-modal goal conditioned policy. arXiv preprint arXiv:2408.14368, 2024. 7

[25] Shuang Li, Yihuai Gao, Dorsa Sadigh, and Shuran Song. Unified video action model. In Robotics: Science and Systems, 2025. 2, 6, 7

[26] Xinghang Li, Minghuan Liu, Hanbo Zhang, Cunjun Yu, Jie Xu, Hongtao Wu, Chilam Cheang, Ya Jing, Weinan Zhang, Huaping Liu, et al. Vision-Language foundation models as effective robot imitators. In ICLR, 2024. 7

[27] Zhixuan Liang, Yao Mu, Mingyu Ding, Fei Ni, Masayoshi Tomizuka, and Ping Luo. Adaptdiffuser: Diffusion models as adaptive self-evolving planners. arXiv preprint arXiv:2302.01877, 2023. 3

[28] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In ICLR, 2023. 2, 3, 4, 5

[29] Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. LIBERO: Benchmarking knowledge transfer for lifelong robot learning. NeurIPS, 36, 2024. 2, 6, 11, 12

[30] Jin Liu, Jialong Xie, Leibing Xiao, Chaoqun Wang, and Fengyu Zhou. Embodied perception for test-time grasping detection adaptation with knowledge infusion. In arXiv preprint arXiv:2504.04795, 2025. 3

[31] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 6

[32] Yueen Ma, Zixing Song, Yuzheng Zhuang, Jianye Hao, and King Irwin. Hierarchical diffusion policy for kinematics-aware multi-task robotic manipulation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 3

[33] Oier Mees, Lukas Hermann, Erick Rosete-Beas, and Wolfram Burgard. CALVIN: A benchmark for languageconditioned policy learning for long-horizon robot manipulation tasks. IEEE Robotics and Automation Letters, 7(3): 7327–7334, 2022. 2, 6, 11, 12

[34] William Peebles and Saining Xie. Scalable diffusion models with transformers. In CVPR, pages 4195–4205, 2023. 2, 3

[35] Karl Pertsch, Kyle Stachowicz, Brian Ichter, Danny Driess, Suraj Nair, Quan Vuong, Oier Mees, Chelsea Finn, and Sergey Levine. Fast: Efficient action tokenization for visionlanguage-action models. In Robotics: Science and Systems, 2025. 6

[36] Delin Qu, Haoming Song, Qizhi Chen, Yuanqi Yao, Xinyi Ye, Yan Ding, Zhigang Wang, JiaYuan Gu, Bin Zhao, Dong Wang, et al. Spatialvla: Exploring spatial representations for visual-language-action model. In Robotics: Science and Systems, 2025. 6

[37] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal ofmachine learning research, 21(140):1–67, 2020. 3

[38] Moritz Reuss, Omer Erdinc Yagmurlu, Fabian Wenzel, and Rudolf Lioutikov. Multimodal diffusion transformer: Learning versatile behavior from multimodal goals. In Robotics: Science and Systems, 2024. 7

[39] Moritz Reuss, Jyothish Pari, Pulkit Agrawal, and Rudolf Lioutikov. Efficient diffusion transformer policies with mixture of expert denoisers for multitask learning. In ICLR, 2025. 6, 7

[40] Julian Schrittwieser, Ioannis Antonoglou, Thomas Hubert, Karen Simonyan, Laurent Sifre, Simon Schmitt, Arthur

Guez, Edward Lockhart, Demis Hassabis, Thore Graepel, et al. Mastering atari, go, chess and shogi by planning with a learned model. Nature, 588(7839):604–609, 2020. 2

[41] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024. 4, 14

[42] Baochen Sun, Yang Wei, Chen Feng, Bo Wang, Sophia D’Souza, and Derek Hoiem. Test-time training with selfsupervision for generalization under distribution shifts. In Proceedings of the 37th International Conference on Machine Learning (ICML), pages 9336–9346, 2020. 3

[43] Andrew Szot, Max Schwarzer, Harsh Agrawal, Bogdan Mazoure, Rin Metcalf, Walter Talbott, Natalie Mackraz, R Devon Hjelm, and Alexander T Toshev. Large language models as generalizable policies for embodied tasks. In ICLR, 2023. 1

[44] Weihao Tan, Wentao Zhang, Shanqi Liu, Longtao Zheng, Xinrun Wang, and Bo An. True knowledge comes from practice: Aligning llms with embodied environments via reinforcement learning. arXiv preprint arXiv:2401.14151, 2024.

[45] Octo Model Team, Dibya Ghosh, Homer Walke, Karl Pertsch, Kevin Black, Oier Mees, Sudeep Dasari, Joey Hejna, Tobias Kreiman, Charles Xu, et al. Octo: An opensource generalist robot policy. In Robotics: Science and Systems, 2024. 2

[46] Qin Wang, Olga Fink, Luc Van Gool, and Dengxin Dai. Continual test-time domain adaptation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7201–7211, 2022. 3

[47] Hongtao Wu, Ya Jing, Chilam Cheang, Guangzeng Chen, Jiafeng Xu, Xinghang Li, Minghuan Liu, Hang Li, and Tao Kong. Unleashing large-scale video generative pre-training for visual robot manipulation. In ICLR, 2024. 7

[48] Sipeng Zheng, Jiazheng Liu, Yicheng Feng, and Zongqing Lu. Steve-Eye: Equipping llm-based embodied agents with visual perception in open worlds. arXiv preprint arXiv:2310.13255, 2023. 1

[49] Zhide Zhong, Haodong Yan, Junfeng Li, Xiangchen Liu, Xin Gong, Wenxuan Song, Jiayi Chen, and Haoang Li. Flowvla: Thinking in motion with a visual chain of thought. arXiv e-prints, pages arXiv–2508, 2025. 6

[50] Gaoyue Zhou, Hengkai Pan, Yann LeCun, and Lerrel Pinto. Dino-wm: World models on pre-trained visual features enable zero-shot planning. arXiv preprint arXiv:2411.04983, 2024. 2

[51] Chuning Zhu, Raymond Yu, Siyuan Feng, Benjamin Burchfiel, Paarth Shah, and Abhishek Gupta. Unified world models: Coupling video and action diffusion for pretraining on large robotic datasets. In Robotics: Science and Systems, 2025. 2

# AdaWorldPolicy: World-Model-Driven Diffusion Policy with Online Adaptive Learning for Robotic Manipulation

Supplementary Material

In the supplementary material, we report additional results and also provide more implementation details.

## 6. Additional Results

## 6.1. Real-world Video Results

To provide a more intuitive and comprehensive view of our real-world experiments, we have prepared an offline webpage containing video demonstrations of all tasks discussed in the main paper. This webpage allows for easy browsing the rollout results from our method without requiring an internet connection.

Accessing the Video Results via Web Browser. The video results are organized in a local HTML file included in the supplementary material zip package. To view them: 1. Unzip the supplementary archive to a local folder.

2. Open the index.html file located in the AdaWorldPolicy Homepage/ directory by using any modern web browser (e.g., Chrome, Firefox).

3. The page contains embedded players for all videos, which will play automatically.

Content Organization. The webpage visualizes the performance of our method, AdaWorldPolicy, across four distinct manipulation tasks: T1 - Sweep Coffee Beans, T2 - Long-horizon Pick-and-Place Eggs, T3 - Pour Water, and T4 - Wipe Whiteboard. The video demonstrations contain two main parts:

• In-Domain Settings: We first demonstrate the rollout results of our AdaWorldPolicy (without online adaptation) in the original training environments for all four tasks.

• Domain Shift Settings: We then showcase the effectiveness of our full method, AdaWorldPolicy (with AdaOL), under four challenging domain shift conditions for each task. These shifts include changes in tablecloths, the addition of distractors, changes in object instances, and random lighting variations.

All robotic arm operations shown in the videos are automatically generated through model inference and are accelerated by 10× for efficient viewing.

## 6.2. Visualization of Imagined Future Frames

To show that the world model can provide good supervision signals in our method, we provide a qualitative analysis of the future frames imagined by our AdaWorldPol-

Table 5. Ablation study on sampling steps, adaptation, and fusion mechanisms. We analyze the impact of inference sampling steps, the effectiveness of our AdaOL, and the choice of multimodal fusion strategy. While reducing sampling steps slightly lowers performance, our method remains robust. Our AdaOL provides a consistent improvement. In addition, our MMSA module significantly outperforms the standard fusion baselines (Concatenation and Cross-Attention), validating its design. Note: AWP denotes our AdaWorldPolicy without using our online adaptive learning strategy (AdaOL). The symbol “ ” indicates that the result was reported in the main paper.

<table><tr><td>Configuration</td><td>#Sampling Steps ↓</td><td>Success Rate (%) ↑</td></tr><tr><td rowspan="4">AWP</td><td>20</td><td>96.33</td></tr><tr><td>10</td><td> $\underline{95.53}$ </td></tr><tr><td>5</td><td>94.67</td></tr><tr><td>2</td><td>94.00</td></tr><tr><td>+ AdaOL</td><td>10</td><td>96.05</td></tr><tr><td>MMSA</td><td>10</td><td>95.53</td></tr><tr><td>Concatenation</td><td>10</td><td>89.67</td></tr><tr><td>Cross-Attention</td><td>10</td><td>91.21</td></tr></table>

icy across different tasks and domains. We compare the generated video predictions (Imagined Future) with the actual environmental observations (Real Observation) over time in both simulated benchmarks (see Fig. 7) and realworld setup (see Fig. 8). In simulation environments such as PushT [20], CALVIN [33], and Libero10 [29](Fig. 7 ac), our world model generates high-fidelity future frames that are highly consistent with the ground truth, effectively capturing the dynamics of the robot and objects. In realworld scenarios (Fig. 8), the model successfully predicts the gripper’s motion and key interactions. However, due to the inherent capacity limitations of the base video generation model (Cosmos-Predict2 [3]), some visual artifacts and blurring are observable in complex real-world scenes, particularly in Task 2 (Fig. 8 (b)) where the background is cluttered and multiple small objects (eggs) exist. Despite observing these imperfect visual artifacts, the structural and semantic consistency remains sufficient for effective policy manipulation.

## 6.3. Ablation Studies on Libero10

We further conduct comprehensive ablation studies on the Libero benchmark to validate key design choices of our AdaWorldPolicy, focusing on inference efficiency, adaptation strategies, and multi-modal fusion mechanisms. The results are summarized in Table 5.

![](images/3795d23511b4c777bd04c8f52725ce5ef98a5e8cc331de97f832ee4ec707c463.jpg)  
(c) Three Scenes in Libero10  
Figure 7. Visualization of imagined future frames in simulated benchmarks generated by our AdaWorldPolicy. In each subfigure (a-c), we compare the imagined future frames (bottom row) with the actual real observations (top row). Across the PushT[20], CALVIN[33], and Libero10 [29] environments, the imagined frames demonstrate high consistency with the ground truth. While minor artifacts and slight deviations in agent position are present, the model effectively captures the core dynamics of the tasks.

(d) Real-world Task 4: Wipe Whiteboard  
![](images/7874df9b0be123b1912e7907f86202a92f63c24bb59a4eaeb8f9a3b9ecfc3e64.jpg)  
Figure 8. Visualization of imagined future frames in real-world scenarios generated by our AdaWorldPolicy. In each subfigures (a-d), we compare the imagined future frames (bottom row) with the actual real observations (top row) across various simulation and real-world tasks. In real-world scenarios (a-d), while the model generally captures the dynamics, some artifacts are observable due to the limitations of the base Cosmos-Predict2 model [3], particularly in complex scenes with cluttered backgrounds or numerous objects, such as the egg manipulation task in (b).

Impact of Sampling Steps. We first investigate the tradeoff between inference speed and performance by varying the number of denoising sampling steps. As shown in the first section of Table 5, the success rate exhibits a positive correlation with the number of steps. However, the policy generated from our method demonstrates remarkable robustness; even when the sampling steps are aggressively reduced to 2, the method maintains a competitive success rate of 94.00%. This suggests that our diffusion-based policy learns a high-quality manifold that can be traversed efficiently.

Effectiveness of AdaOL. We further evaluate the contribution of our online adaptation mechanism. Enabling AdaOL (with 10 sampling steps) improves the success rate from 95.53% to 96.05%. This improvement indicates that AdaOL effectively mitigates subtle distribution shifts between the training data and the test environment, improving the policy’s precision during execution.

Importance of MMSA Fusion. Finally, we analyze the efficacy of our Multi-Modal Self-Attention (MMSA) module by replacing it with standard fusion baselines while keeping other components fixed (using 10 sampling steps).

• MMSA → Concatenation: Replacing MMSA with simple feature concatenation results in a significant performance drop to 89.67%. This suggests that naive combination of visual and proprioceptive features is insufficient for capturing complex inter-modal dependencies.

• MMSA → Cross-Attention: Using a standard Cross-Attention mechanism yields a success rate of 91.21%, which is still notably lower than our MMSA-based approach (95.53%).

These results strongly validate the design of our MMSA, highlighting its superior ability to integrate multi-modal information for precise manipulation tasks than the conventional fusion approaches. This finding aligns with the observation drawn from our real-world ablation experiments presented in the main paper.

## 7. Implementation Details

## 7.1. Network Details

Our framework builds upon Cosmos-Predict2 [3], a world foundation model designed to generate future video frames conditioned on single-view video history and textual descriptions. While the model is available with both 2B and 14B parameters, we adopt the 2B version as our backbone to balance generation quality and computational efficiency under limited GPU resources. Although the original architecture supports action conditioning via Adaptive Layer Normalization (AdaLN) [18], our empirical investigations revealed that this mechanism is sub-optimal for the specific task of action prediction. Consequently, we implement a separate action model branch integrated via our proposed Multi-Modal Self-Attention (MMSA) [14] mechanism. This decoupling design allows us to seamlessly utilize the pre-trained weights of the world model while maintaining full flexibility in the action model’s architecture. Additionally, when extending the original single-view Cosmos-Predict2 to support multi-view inputs, we found that concatenating tokens from different views along the temporal dimension and sharing Rotary Positional Embeddings (RoPE) [41] lead to effective fine-tuning with stable convergence.

We tailor the scale of our models to suit the specific requirements of different experimental domains. In our realworld setup, which incorporates haptic feedback, we configure both the action model and the force predictor to have approximately 0.4B parameters each. Conversely, for simulation benchmarks where force data is unavailable, we reallocate the computational budget to enhance the policy’s expressivity by increasing the size of the action model to 0.6B parameters. This adjustment allows the model to better capture complex manipulation behaviors in the absence of haptic cues, providing a performance boost with only a marginal increase in computational overhead.

Table 6. Hyperparameter configurations for different benchmarks. We adjust image resolution and temporal parameters to balance the trade-off between visual precision, memory consumption, and inference speed. Notably, the Real-world setting uses a sparse prediction strategy to minimize latency.

<table><tr><td>Benchmark</td><td>Image Size</td><td>History Length</td><td>Action Horizon</td><td>#Imagined Frames</td></tr><tr><td>Libero10</td><td> $128 \times 128$ </td><td>5</td><td>20</td><td>20</td></tr><tr><td>PushT</td><td> $256 \times 256$ </td><td>5</td><td>20</td><td>20</td></tr><tr><td>CALVIN</td><td> $192 \times 192$ </td><td>1</td><td>12</td><td>12</td></tr><tr><td>Real-world</td><td> $112 \times 160$ </td><td>1</td><td>32</td><td>4</td></tr></table>

## 7.2. Data Processing

In the learning-based robotic manipulation domain, it is well-established that larger training batch sizes generally correlate with improved policy performance and stability. However, within our AdaWorldPolicy framework, the video generation backbone (World Model) dominates GPU memory consumption. This creates a critical trade-off: while higher image resolutions are essential for the model to perceive fine-grained details necessary for precise manipulation (“seeing clearly”), they significantly increase the memory footprint, thereby limiting the maximum trainable batch size. To address this, we carefully tuned the hyperparameters for each benchmark, balancing training memory constraints, batch size, inference latency, and the required spatiotemporal resolution. The specific configurations for each environment are detailed in Table 6.

It is worth noting that for the Real-world setting, the number of imagined frames (4) is significantly lower than the action horizon (32), unlike in the simulation benchmarks. This design choice was made to minimize inference latency. We implemented a skipped frame prediction strategy, where the model predicts future frames at fixed intervals rather than generating a dense sequence. This approach allows the model to cover the full temporal span of the action horizon while drastically reducing the computational cost associated with video generation.

In our real-world experiments, we also observed that the preprocessing of force data requires distinct handling process compared to image or relative action modalities. Unlike pixel values or relative end-effector poses, force measurements typically lack strict theoretical upper or lower bounds and can exhibit significant variance or spikes. Consequently, standard normalization techniques (e.g., minmax scaling based on absolute extremes) can be unstable or lead to compressed data distributions. We empirically found that employing quantile-based normalization—specifically, scaling data based on statistical percentiles (i.e., the 1<sup>st</sup> and 99<sup>th</sup> percentiles)—provides a robust mapping. This strategy effectively mitigates the impact of outliers and resulted in the optimal balance of convergence speed and final policy performance.

## 7.3. Real-world Evaluation Protocol

Data Collection and Training. For each of the four realworld manipulation tasks, we collected 150 expert demonstrations in the in-domain environment via teleoperation. These datasets were utilized to train the policy using standard offline behavior cloning. Once trained, the models were evaluated directly in both the original in-domain setting and four distinct domain-shift scenarios to assess robustness and generalization capability.

Evaluation Metrics and Baselines. For every model configuration, we conduct 30 evaluation trials per task under each domain distribution. A maximum limit of 1500 execution steps is enforced for all tasks. To ensure a fair comparison, we enhanced the standard Diffusion Policy baseline to incorporate haptic feedback. Specifically, 6-dimensional force data is injected into the model by concatenating it with image observation features and processing it via crossattention within the Transformer architecture. We refer to this haptic-enabled baseline as DP-Force.

Success Criteria. The specific success conditions for the four tasks are defined as follows:

• T1 - Sweep Beans: The task is considered successful if no more than three coffee beans (or corn kernels) remain on the table surface.

• T2 - Pick and Place Eggs: The egg must be transported without touching the table surface during the whole trajectory and must be placed securely into an empty slot in the carton.

• T3 - Pour Water: The final volume of water in the target glass must exceed 90% of the initial water volume in the source cup.

• T4 - Wipe Whiteboard: The length of any remaining marker trace on the whiteboard must not exceed 3 cm.

AdaOL Testing Protocol. To rigorously evaluate the effectiveness of our online adaptation mechanism, we adopt a specific protocol for the AdaOL experiments. For each task and domain combination, the model is reset to its pretrained state before the first rollout. The evaluation proceeds in two sequential phases. First, during the adaptation phase (Trials 1-15), the model performs the task while continuously updating its parameters online via test-time adaptation, with the outcomes of these trials recorded. Subsequently, during the frozen phase (Trials 16-30), the model weights are frozen, and the remaining 15 trials are executed without further updates to evaluate the stability of the adapted policy. The final reported success rate is the overall performance from all 30 trials.

## 8. Limitation and Future Work

Despite our AdaWorldPolicy has achieved promising results, we point out several limitations that may lead to future research directions.

First, while our backbone world model, Cosmos-Predict2 [3], is pre-trained on millions of hours of video data and represents the state-of-the-art embodied world modeling capability, its generalization capabilities remain imperfect under significant domain shifts. In our experiments, particularly within real-world settings, we observed that the quality of predicted future frames degrades when the environment undergoes drastic changes, such as continuous and varying lighting conditions. This suggests that this world model still lacks sufficient robustness to handle extreme out-of-distribution scenarios. However, it is important to note that our primary objective is effective policy execution rather than photorealistic video generation. Our results demonstrate that our method can generate reasonable policy even when the predicted visual future contains minor artifacts. We anticipate that as stronger and more generalizable world models emerge, integrating them into our framework will naturally alleviate these visual distortions and further enhance execution performance.

Second, regarding our online adaptation mechanism, we employed a fixed set of hyperparameters for AdaOL across all domain shift experiments to demonstrate the method’s general applicability. We did not perform task-specific or environment-specific fine-tuning of the adaptation parameters. Future work could explore adaptive or meta-learning approaches to automatically tune AdaOL’s hyperparameters during deployment, which may potentially lead to further performance improvements.