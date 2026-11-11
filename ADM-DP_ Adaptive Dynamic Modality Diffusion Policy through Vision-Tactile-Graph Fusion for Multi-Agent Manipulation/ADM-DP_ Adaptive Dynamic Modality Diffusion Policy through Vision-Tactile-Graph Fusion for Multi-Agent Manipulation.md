# ADM-DP: Adaptive Dynamic Modality Diffusion Policy through Vision-Tactile-Graph Fusion for Multi-Agent Manipulation

Enyi Wang, Wen Fan and Dandan Zhang

Abstract— Multi-agent robotic manipulation remains challenging due to the combined demands of coordination, grasp stability, and collision avoidance in shared workspaces. To address these challenges, we propose the Adaptive Dynamic Modality Diffusion Policy (ADM-DP), a framework that integrates vision, tactile, and graph-based (multi-agent pose) modalities for coordinated control. ADM-DP introduces four key innovations. First, an enhanced visual encoder merges RGB and point-cloud features via Feature-wise Linear Modulation (FiLM) modulation to enrich perception. Second, a tactileguided grasping strategy uses Force-Sensitive Resistor (FSR) feedback to detect insufficient contact and trigger corrective grasp refinement, improving grasp stability. Third, a graphbased collision encoder leverages shared tool center point (TCP) positions of multiple agents as structured kinematic context to maintain spatial awareness and reduce inter-agent interference. Fourth, an Adaptive Modality Attention Mechanism (AMAM) dynamically re-weights modalities according to task context, enabling flexible fusion. For scalability and modularity, a decoupled training paradigm is employed in which agents learn independent policies while sharing spatial information. This maintains low interdependence between agents while retaining collective awareness. Across seven multi-agent tasks, ADM-DP achieves 12-25% performance gains over state-of-the-art baselines. Ablation studies show the greatest improvements in tasks requiring multiple sensory modalities, validating our adaptive fusion strategy and demonstrating its robustness for diverse manipulation scenarios. https://Enyi-Bean.github.io/ ADM-DP/

## I. INTRODUCTION

Imitation learning has become a powerful paradigm for robotic manipulation, enabling robots to acquire complex skills from demonstrations without explicit programming [1]. Recent advances in Diffusion-based policies have revolutionized this field: Diffusion Policy (DP) [2] models actions as conditional denoising processes, and 3D Diffusion Policy [3] leverages point clouds for improved spatial reasoning. Subsequent work has explored richer visual representations [4], [5], while flow-matching alternatives such as Flow Policy [6], [7] achieve faster inference via straight-through trajectory generation. However, these vision-centric methods are predominantly designed for single-agent settings; extending them to multi-agent manipulation exposes unresolved challenges in coordination, collision avoidance, and multi-sensory fusion.

Current multi-agent systems face a critical trade-off between scalability and coordination effectiveness [8]. Centralized approaches that jointly process all agents’ observations suffer from exponential state-space growth. Recent work has explored decoupled training paradigms, with Jiang et al. [9] proposing a decoupled interaction framework for bimanual tasks and RoboFactory [10] extending this to multi-agent scenarios. While these methods achieve better scalability, they lack explicit collision awareness mechanisms. Graph neural networks (GNNs) show promise for multi-robot coordination [11], yet require explicit scene modeling that limits task generalization. The recent KStar Diffuser [12] constructs comprehensive joint-level spatio-temporal graphs, but includes many task-irrelevant nodes (e.g., base joints) that increase computational overhead without improving end-effector coordination where collisions actually occur. This suggests a key insight: multi-agent systems require lightweight coordination which focuses on task-critical interactions.

![](images/4b188e887ea1f08913d437aa44dc2aefe205e8d09eca9854f669b97150f03b88.jpg)  
Fig. 1. The overview pipline of our method framework.

Furthermore, the demand upon coordination highlights only one side of the challenge. Equally important is how multi-agent systems handle multi-modal sensory input, which can undermine both efficiency and robustness. Inspired by bionic behaviour, humans naturally modulate sensory attention during manipulation, relying on vision for approach, touch for contact verification, and spatial awareness for coordination. Yet most of current multi-modal methods [13]–[15] employ static fusion strategies, only concatenating or uniformly weighting modalities regardless of task phase. This creates fundamental inefficiencies across all sensory channels: tactile signals are zero during approach phases, providing no information yet still being encoded as high-dimensional features that inject noise into the network; spatial awareness for collision avoidance is unnecessary when agents are distant but becomes critical when proximate; even visual features may be redundant during stable grasping when tactile feedback should dominate.

Advanced tactile-integrated architectures like Reactive Diffusion Policy’s dual-frequency design [14] or 3D-ViTac’s unified tactile point cloud representation [13] continue processing these zero-valued or irrelevant channels throughout execution. Methods like Bi-Touch [16] assume all modalities are ‘always useful’, degrading sample efficiency when certain readings are meaningless noise. The core limitation is clear: without dynamic modality weighting that adapts to task context, policies waste computational resources on redundant sensing information irrelevant to current task state while potentially allowing their noise to corrupt useful signals from task-critical modalities.

This limitation becomes especially critical in multi-agent manipulation, where mutual occlusions and workspace interference amplify the impact of poorly integrated sensory inputs, leading to unstable grasps. While tactile sensing provides rich contact information, existing methods [13], [14], [16] treat it as supplementary observation rather than leveraging it for active control. The challenge extends beyond sensor integration, it requires data collection strategies that teach policies to recognize insufficient grasps through tactile signatures and execute corrective actions, transforming tactile feedback from passive sensing to active refinement signal.

These interconnected challenges motivate our core insight: multi-agent manipulation requires adaptive sensory fusion that dynamically adjusts to task phases, not static architectures that process all modalities uniformly. To achieve these objectives, we define the modality used in this work as any structured information source that contributes complementary perspectives for cooperative manipulation across multiple agents, including sensory signals (e.g., vision, tactile) and relational encodings (e.g., graph-based spatial awareness). Accordingly, we propose ADM-DP (Adaptive Dynamic Modality Diffusion Policy), a framework illustrated in Fig. 1. In summary, we introduce three key contributions:

(1) Tactile-guided grasping strategy. We design a datacollection protocol that deliberately includes shallow-to-deep grasp refinements in 30% of demonstrations. In these trajectories, an initial shallow grasp, indicated by weak Force-Sensitive Resistor (FSR)-based tactile signals and incomplete contact, is followed by a corrective deepening motion. This exposure enables the policy to associate tactile signatures with insufficient contact and to execute refinement actions, thereby elevating tactile sensing from passive observation to active control. Combined with the decoupled training paradigm, in which agents learn independent policies while conditioning on shared TCP position information, this strategy supports scalable learning while preserving effective inter-agent coordination.

(2) Multi-modal Encoding for Complementary Sensing: ADM-DP processes each modality through specialized encoders designed for their unique characteristics. We enhance visual perception by combining RGB semantics with point cloud geometry via FiLM modulation, providing robust 3D understanding despite occlusions. Tactile signals from FSR arrays are encoded with spatial positions to preserve contact patterns crucial for grasp stability. Graph Attention Networks (GAT) [17] process shared TCP positions in graph structural format to enable lightweight collision avoidance between multiple agents without modeling entire kinematic chains. Each encoder extracts task-critical features while minimizing computational overhead.

(3) Dynamic Modality Fusion via AMAM: Unlike static fusion methods that process all modalities uniformly, our proposed AMAM dynamically allocates attention based on task context through learnable importance weights. With entropy regularization preventing both uniform averaging and modality collapse, AMAM learns to suppress tactile noise during approach, amplify it during contact, and activate spatial awareness when agents converge. This adaptive fusion enables policies to automatically adjust sensory priorities throughout task execution without manual phase detection or switching. By addressing the fundamental limitation of static fusion and introducing adaptive mechanisms for multi-agent coordination, ADM-DP advances toward robotic systems that dynamically modulate their sensory focus based on task demands, knowing when to prioritize vision, when to rely on touch, and when to maintain spatial awareness.

## II. METHODOLOGY

We address the problem of multi-agent robotic manipulation where n agents must coordinate to complete complex tasks. Let $A = \{ { \mathcal { A } } _ { 1 } , . . . , { \mathcal { A } } _ { n } \}$ represent the entire action space where $A _ { i }$ is the action space for specific agent i. The observation space for each agent i consists of both local and shared components: $\mathcal { O } _ { i } \ = \ \{ \mathcal { O } _ { i } ^ { \mathrm { l o c a l } } , \mathcal { O } ^ { \mathrm { s h a r e d } } \}$ . The local observations $\mathcal { O } _ { i } ^ { \mathrm { l o c a l } } = \{ I _ { i } , P _ { i } , T _ { i } , \dot { q } _ { i } , \dot { L } _ { i } \}$ include RGB images $I _ { i } ~ \in ~ \mathbb { R } ^ { H \times W \times 3 }$ , point clouds $P _ { i } ~ \in ~ \mathbb { R } ^ { N \times 6 }$ , tactile readings $T _ { i } \in \mathbb { R } ^ { 3 2 }$ from FSR sensors, joint states $q _ { i } \in \mathbb { R } ^ { d _ { q } }$ and a language instruction $L _ { i }$ specifying the agent’s sub-task. The shared observations $\mathcal { O } ^ { \mathrm { s h a r e d } } = \{ p _ { 1 } ^ { \mathrm { t c p } } , . . . , p _ { n } ^ { \mathrm { t c p } } \}$ contain the end-effector TCP positions $p _ { i } ^ { \mathsf { t c p } } \in \mathbb { R } ^ { 3 }$ of all agents, enabling collision-aware coordination. Our goal is to learn a set of decoupled policies $\{ \pi _ { 1 } , . . . , \pi _ { n } \}$ where each policy $\pi _ { i } : { \mathcal { O } } _ { i } \to$ $\mathbf { \mathcal { A } } _ { i }$ maps agent i’s observations to actions. We present our approach in the following sections: the decoupled training paradigm (Sec. II-A), the ADM-DP architecture (Sec. II-B), and tactile-guided grasping strategy (Sec. II-C).

## A. Decoupled Training and Evaluation Paradigm

Traditional approaches of bimanual or multi-agent robotic manipulation often employ a centralized policy $\pi : \mathcal { O } _ { 1 } \times$ $\ldots \times \mathcal { O } _ { n } \to \mathcal { A } _ { 1 } \times \ldots \times \mathcal { A } _ { n }$ that jointly processes all agents observations and outputs all actions simultaneously as well. However, this approach suffers from several limitations: exponential growth in observation and action spaces with increasing agents, difficulty in generalizing to different numbers of agents, and high sample complexity for training.

In contrast, recent works have explored decoupled approaches for multi-arm manipulation. Jiang et al. [9] proposed a decoupled interaction framework for bimanual tasks, while RoboFactory [10] demonstrated that training separate policies for each agent can effectively scale to multi-agent scenarios. Building upon these decoupled training strategies, we adopt a similar paradigm with an important extension: our agents share TCP positions with each other in addition to their local observations to enhance coordination. Therefore, we adopt a decoupled training paradigm. During training, we learn independent policies for each agent:

![](images/e306a748244a7738205e89b78c7af5de951251a07dd41a4dd8edb69050d6fa9c.jpg)  
Fig. 2. Overview of ADM-DP architecture. Multi-modal observations (vision, tactile, graph) are processed through specialized encoders and dynamicall fused via AMAM. The fused features are conditioned on language instructions through FiLM [18] modulation before guiding the diffusion process for action generation.

$$
\text { Training: } \pi_ {i}: \mathcal {O} _ {i} \to \mathcal {A} _ {i}, \quad i \in \{1,..., n \}\tag{1}
$$

where each policy $\pi _ { i }$ is optimized separately using demonstrations collected via motion planning with randomized initializations. Each agent i observes $\mathcal { O } _ { i } ^ { - } = \{ \mathcal { O } _ { i } ^ { \mathrm { l o c a l } } , \mathcal { O } ^ { \mathrm { s h a r e d } } \}$ where the shared component contains all agents’ TCP positions for spatial awareness. At evaluation time, the trained policies execute in parallel:

$$
\text { Evaluation: } a _ {i} ^ {t} = \pi_ {i} (\mathcal {O} _ {i} ^ {t}), \quad \forall i \in \{1,..., n \}\tag{2}
$$

where each agent $\mathbf { \mathcal { A } } _ { i }$ independently generates actions $a _ { i } ^ { t }$ based on its current observations $\mathcal { O } _ { i } ^ { t } .$ , with TCP positions updated in real-time from all agents.

This decoupled approach significantly reduces training complexity as each policy only needs to learn single-agent behaviors rather than the exponentially larger joint action space. Moreover, it enables modular deployment where agents can be added or removed without retraining the entire system.

## B. ADM-DP Architecture

As illustrated in Fig. 2, ADM-DP processes multi-modal observations through specialized encoders and dynamically fuses them via AMAM, which consists of three main components: (1) multi-modal encoders for vision, tactile, and graph modalities; (2) AMAM fusion module; and (3) diffusionbased action decoder:

1) Multi-modal Encoders: (1) Vision Encoding: We process visual inputs through complementary RGB and point cloud pathways. RGB images $I _ { i } ~ \in ~ \bar { \mathbb { R } ^ { H \times W \times 3 } }$ are encoded via ResNet [19] to extract semantic features $f _ { i m g } =$ $\mathbf { R e s N e t } ( I _ { i } ) \ \in \ \mathbb { R } ^ { 5 1 2 }$ . Point clouds undergo preprocessing including workspace cropping and downsampling to $N =$ 1024 points while preserving color information, resulting in $P _ { i } \in \mathbb { R } ^ { 1 0 2 4 \times 6 }$ . These are processed through a modified PointNet [20] architecture where we remove T-Net and BatchNorm layers following insights from DP3 [3], yielding geometric features $f _ { p c } = \bar { \mathrm { P o i n t N e t } } ( P _ { i } ) \in \mathbb { R } ^ { 1 0 2 4 }$ . To leverage the complementary strengths of both modalities, semantic understanding from RGB and precise 3D geometry from point clouds, we integrate them using FiLM [18] modulation:

$$
f _ {v} = \gamma (f _ {p c}) \odot f _ {i m g} + \beta (f _ {p c})\tag{3}
$$

where $\gamma ( \cdot )$ and $\beta ( \cdot )$ are learned affine transformations. This fusion strategy allows geometric features to adaptively modulate semantic features, producing enhanced visual representations $f _ { v } \in \mathbb { R } ^ { 5 1 2 }$ that are more robust to visual ambiguities and occlusions common in multi-agent scenarios.

(2) Tactile Encoding: Each gripper is equipped with 4×4 FSR sensors on both fingertips, providing tactile readings $T _ { i } ~ \in ~ \mathbb { R } ^ { 3 2 }$ . To preserve spatial structure crucial for grasp adjustment, we incorporate positional encoding for each taxel. The tactile encoder first reshapes the input into $T _ { i } ~ \in ~ \mathbb { R } ^ { 2 \times 4 \times 4 }$ (two fingers, each with 4×4 grid), applies log-normalization for stability, and concatenates 2D grid positions $( x , y ) \in [ - 1 , 1 ] ^ { 2 }$ to each taxel reading. Per-finger features are extracted through 1D convolutions with adaptive pooling. Additionally, we compute contact dynamics including (1) resultant force (sum of all taxel readings), and (2) differential force between fingers (indicating grasp balance) for each finger to track contact point locations. These physical features, combined with the convolutional features, are fused through a feedforward network to produce $f _ { t } \in \mathbb { R } ^ { 6 4 }$ , enabling the model to understand both fine-grained contact patterns and global force distributions critical for stable grasping.

(3) Graph-based Collision Encoding: To enable collision-aware coordination, we encode the shared TCP positions $\mathcal { O } ^ { \mathrm { s h a r e d } } = \{ p _ { 1 } ^ { \mathrm { t c p } } , . . . , p _ { n } ^ { \mathrm { t c p } } \}$ using a Graph Attention Network (GAT) [17]. Each TCP position forms a node in a fully connected graph with edge weights inversely

![](images/bea2410e0342acd52a746cca4b2ce9d1239c3b29f85c2a3b90b75ace37b859a5.jpg)  
Fig. 3. Adaptive Modality Attention Mechanism (AMAM) dynamically allocates importance weights to vision, tactile, and graph modalities based on task context.

proportional to distances:

$$
e _ {i j} = \frac {1}{\| p _ {i} ^ {\mathrm{tcp}} - p _ {j} ^ {\mathrm{tcp}} \| _ {2} + \epsilon}\tag{4}
$$

The GAT aggregates spatial relationships through attention mechanisms, producing graph features $f _ { g } \in \mathbb { R } ^ { 6 4 }$ that capture proximity-aware inter-agent relationships.

2) AMAM Fusion Module: (1) AMAM: Different task phases require adaptive modality priorities: vision dominates during approaching before contact, tactile becomes crucial during grasping, and graph awareness intensifies when agents are in close proximity. As shown in Fig. 3, our AMAM module dynamically allocates importance weights to each modality based on the current context. Given the encoded features $\{ f _ { v } , f _ { t } , f _ { g } \}$ in terms of vision, tactile and graph, AMAM computes importance weights through a gated attention mechanism:

$$
\alpha = \operatorname{softmax} (\mathrm{MLP} ([ f _ {v}; f _ {t}; f _ {g} ]) / \tau)\tag{5}
$$

where τ is a temperature parameter and MLP is a multi-layer perceptron. The fused feature is computed as:

$$
f _ {v t g} = \alpha_ {v} \cdot f _ {v} + \alpha_ {t} \cdot f _ {t} + \alpha_ {g} \cdot f _ {g}\tag{6}
$$

To prevent weight collapse where all modalities receive equal attention, we introduce an entropy regularization term:

$$
\mathcal {L} _ {r e g} = - \lambda \sum_ {m \in \{v, t, g \}} \alpha_ {m} \log (\alpha_ {m})\tag{7}
$$

This encourages the model to make decisive modality selections rather than uniform averaging.

(2) Conditional Feature Integration: The fused multimodal features are concatenated with proprioceptive joint states: $f _ { o b s } = [ f _ { v t g } ; q _ { i } ]$ . Language instructions $L _ { i }$ are encoded using a frozen CLIP [21] text encoder to obtain $f _ { l } = \mathbf { C } \mathbf { L I P } ( L _ { i } )$ . The final conditioning feature is obtained through FiLM modulation:

$$
f _ {c o n d} = \gamma (f _ {l}) \odot f _ {o b s} + \beta (f _ {l})\tag{8}
$$

This conditioning vector $f _ { c o n d }$ guides the diffusion process for action generation, enabling task-specific behaviors while maintaining multi-modal awareness.

3) Diffusion-based Action Generation: We employ a conditional diffusion model to generate action trajectories given the multi-modal conditioning features. Following the DDPM framework [22], we define a forward diffusion process that gradually adds Gaussian noise to action trajectories over $T$ timesteps:

$$
q (a ^ {k} | a ^ {k - 1}) = \mathcal {N} (a ^ {k}; \sqrt {1 - \beta_ {k}} a ^ {k - 1}, \beta_ {k} \mathbf {I})\tag{9}
$$

where $a ^ { 0 }$ represents the clean action trajectory, $a ^ { T }$ is pure Gaussian noise, and $\{ \beta _ { k } \} _ { k = 1 } ^ { T }$ is a variance schedule.

The reverse process learns to denoise actions conditioned on the observation features $f _ { c o n d } \mathrm { : }$

$$
p _ {\theta} (a ^ {k - 1} | a ^ {k}, f _ {c o n d}) = \mathcal {N} (a ^ {k - 1}; \mu_ {\theta} (a ^ {k}, k, f _ {c o n d}), \sigma_ {k} ^ {2} \mathbf {I})\tag{10}
$$

where $\mu _ { \theta }$ is parameterized by a U-Net architecture that takes the noisy action, timestep, and conditioning features as input.

During training, we optimize the network to predict the noise added at each timestep:

$$
\mathcal {L} _ {d i f f} = \mathbb {E} _ {k, \epsilon , a ^ {0}} \left[ \| \epsilon - \epsilon_ {\theta} (a ^ {k}, k, f _ {c o n d}) \| ^ {2} \right]\tag{11}
$$

where $\epsilon \sim \mathcal { N } ( 0 , \bf { I } )$ is the noise added to create $a ^ { k }$ from $a ^ { 0 }$ The total training loss combines the diffusion loss with the modality regularization term:

$$
\mathcal {L} _ {t o t a l} = \mathcal {L} _ {d i f f} + \mathcal {L} _ {r e g}\tag{12}
$$

where $\mathcal { L } _ { r e g }$ is the entropy regularization from AMAM that prevents uniform modality weighting.

For efficient inference, we adopt DDIM [23] sampling to accelerate inference, requiring only 20 denoising steps to generate actions. We utilize a history of 3 observation frames to predict action chunks of horizon H = 8 timesteps, then execute the first 6 actions before replanning. This chunked prediction reduces compounding errors while maintaining smooth trajectories necessary for stable multi-agent coordination [24].

## C. Tactile-guided Grasping Strategy

A critical challenge in multi-agent manipulation is achieving stable grasps despite visual uncertainties and occlusions. We observe that policies trained solely on standard grasping demonstrations often fail when deployed, with objects slipping during manipulation due to partial or misaligned contact. This occurs because visual perception alone cannot accurately determine optimal grasp configuration, especially when multiple agents create occlusions or when object surfaces have complex geometries.

To address this challenge, we propose a tactile-guided grasping strategy that leverages FSR feedback during data collection to teach the policy how to refine grasp contact dynamically. As illustrated in Fig. 4, our approach consists of two complementary data collection patterns:

(1) Standard full-contact grasps (70% of data): The gripper approaches the object and executes a stable grasp in which the FSR array exhibits broad activation and sufficient contact force across sensors. These demonstrations establish the nominal grasping behavior under accurate visual perception and serve as the baseline for successful manipulation.

![](images/efef6d6ba77d7f4fc053a194098eb9f905ac2df57c7bda8e2c269d2c8c5b9864.jpg)  
Fig. 4. Tactile-guided grasping strategy. (a) Standard full-contact grasp with complete tactile coverage across all FSR sensors. (b) Contact-refinement pattern where initial partial contact (limited FSR activation) triggers progressive grasping motion until a stable tactile signature is achieved. This teaches the policy to use tactile feedback for grasp refinement.

(2) Contact-refinement adjustments (30% of data): We deliberately begin with an incomplete grasp, indicated by weak or spatially partial FSR responses (e.g., activation concentrated on the lower sensors). The gripper then executes a corrective deepening motion along the approach direction while monitoring tactile feedback, continuing until broad sensor engagement and a more balanced force distribution are reached. This pattern teaches the policy to detect insufficient contact from tactile signatures and to perform refinement actions that recover a stable grasp.

Within the contact-refinement procedure, we employ two variants: (1) release-regrasp: an initial partial-contact grasp followed by gripper release, deeper repositioning, and regrasping, which teaches the policy to recover from failed grasps via explicit retry; and (2) in-grasp tightening: a partially closed grasp with weak tactile readings followed by progressive closure to full contact, enabling continuous in-grasp adjustment without releasing the object.

This protocol trains a tactile-conditioned refinement behavior. At test time, when the policy detects weak or spatially incomplete tactile contact, it triggers corrective grasp adjustments guided by the tactile encoder. The resulting closedloop refinement improves grasp success, particularly under visual ambiguity or when inter-agent occlusions degrade depth perception.

## III. EXPERIMENTS

## A. Experimental Setup and Evaluation Tasks

We evaluate our approach on seven multi-agent manipulation tasks using Franka Panda robots, each with 7 degrees of freedom plus gripper control. Each gripper is equipped with custom 4×4 FSR sensor arrays installed on the rubber tips of both fingers, as shown in Fig. 5, providing 32 tactile readings per end-effector for fine-grained contact sensing.

![](images/599c5b7be3cea157b0255862266c7134ca3f20b4396e1df7ce80c289b4c7c3d2.jpg)  
Fig. 5. FSR sensor integration on the Franka Panda gripper. Each FSR array is embedded within the compliant rubber fingertip, enabling spatially resolved tactile feedback during manipulation.

Our benchmark consists of four dual-arm and three triarm manipulation tasks adapted from ManiSkill [25] and RoboFactory [10], as illustrated in Fig. 6 and Fig. 7. The dual-arm tasks include Lift Barrier, Pass Peg, Lift Arm, and Two Robots Stack Cube, while the tri-arm tasks comprise Pass Shoe, Take Photo of Tissue, and Three Robots Stack Cube. Notably, the Pass Peg and Pass Shoe tasks are specifically designed with close-proximity handover sequences where agents operate within overlapping workspaces, creating challenging scenarios that test our graph-based collision avoidance mechanism. Besides, each agent receives taskspecific language instructions.

For each task, we collect expert demonstrations using motion planners with randomized object positions and orientations. We evaluate all methods using two data regimes: 50 and 150 demonstrations per agent, and measure the performance by success rate.

![](images/5822ca4d82ff0b628c06014cb97e6ca198aac331f60cb1b96275b0f0fac515b8.jpg)  
Fig. 6. Dual-arm manipulation tasks. From left to right: Lift Barrier, Pass Peg, Lift Arm, and Two Robots Stack Cube. Each task requires precise coordination between two agents with distinct roles specified through language instructions.

![](images/de88de78ece59daae666948efcab50d90e4c15067a0c711a13d83656feaeb987.jpg)  
Fig. 7. Tri-arm manipulation tasks. From left to right: Pass Shoe, Take Photo of Tissue, and Three Robots Stack Cube. These tasks demonstrate scalability to three agents with complex spatial coordination requirements.

## B. Baseline Methods and Comparative Results

We evaluate our approach against state-of-the-art imitation learning methods for robotic manipulation. As baselines, we selected two diffusion-based policies: (1) Diffusion Policy (DP) [2], which utilizes RGB images as observations to generate actions via diffusion models; and (2) 3D Diffusion Policy (DP3) [3], which replaces image inputs in Diffusion Policy with point clouds for improved 3D spatial reasoning. Additionally, we compare with (3) Flow Policy [6], which leverages flow matching [7] with point cloud observations, offering faster inference through straight trajectory generation while maintaining generation quality comparable to diffusion models.

Table I presents the comparative results across all tasks. Our method consistently outperforms all baselines in both data regimes, with particularly significant improvements in the low-data setting (50 demonstrations). In the 50- demonstration regime, ADM-DP achieves an average improvement of 13.3% over the best baseline, demonstrating superior data efficiency. This advantage is maintained with 150 demonstrations, where we achieve 57.1% overall success rate compared to 45.4% for the next best method.

![](images/f5b847ee59f2e3c20f734f5b3899f9e2e02e2f60af358b00e342b630822f9813.jpg)  
Grasp too shallow object slips

![](images/4ca5bf1107071094d1e921f0c1772a13733c41044842e523ff80b820bf9bb5a5.jpg)  
Deeper grasp, object stable

Fig. 8. Left: Shallow grasp causes slipping. Right: Tactile feedback enables more refine, stable grasping.  
![](images/1f5809bd6f59829615367eeb53827bac418ba4f36ff60a076296133be18f4e9f.jpg)  
Fig. 9. Self-collisions during multi-agent manipulation without graph-based coordination.

The performance gains are particularly pronounced in tasks requiring substantial tactile feedback. For Lift Barrier and Lift Arm, which involve precise force control for stable lifting, our tactile-guided approach achieves 92% and 78% success rates respectively with 150 demonstrations, significantly outperforming DP3’s 77% and 62%. The tactile modality enables our policy to detect and correct unstable grasps that vision-only methods fail to identify. For handover tasks (Pass Peg and Pass Shoe), which demand close-proximity coordination, our graph-based collision encoding provides critical spatial awareness. Baseline methods frequently collide when agents operate in overlapping workspaces; in contrast, ADM-DP maintains safe separation by leveraging TCP-based graph features. As a result, ADM-DP achieves 74% success on Pass Peg, compared to 53% for Flow Policy.

Notably, even on vision-dominant tasks such as Stack Cube, our enhanced visual encoder—fusing RGB and pointcloud features via FiLM yields more reliable 3D scene understanding and consistently outperforms single-modality baselines. These gains across heterogeneous task requirements support the effectiveness of adaptive fusion: multimodal tasks (e.g., Pass Peg, which requires vision, tactile feedback, and inter-agent spatial reasoning) benefit most when AMAM emphasizes complementary signals, whereas simpler tasks improve when AMAM down-weights irrelevant modalities to reduce noise. Finally, the results highlight the increased difficulty of scaling to three-agent settings, where all methods exhibit performance degradation. Nevertheless, ADM-DP preserves the largest relative advantage, achieving nearly twice the success rate of DP in the three-agent regime, underscoring the scalability benefits of our design.

TABLE I. Success rates (%) on multi-agent manipulation tasks with different numbers of demonstrations

<table><tr><td rowspan="2">Agents</td><td rowspan="2">Task</td><td colspan="4">50 Demonstrations</td><td colspan="4">150 Demonstrations</td></tr><tr><td>DP</td><td>DP3</td><td>Flow Policy</td><td>Ours</td><td>DP</td><td>DP3</td><td>Flow Policy</td><td>Ours</td></tr><tr><td rowspan="4">Two-Agent</td><td>Lift Barrier</td><td>25</td><td>33</td><td>36</td><td>55</td><td>68</td><td>77</td><td>75</td><td>92</td></tr><tr><td>Pass Peg</td><td>28</td><td>41</td><td>35</td><td>52</td><td>42</td><td>50</td><td>53</td><td>74</td></tr><tr><td>Lift Arm</td><td>17</td><td>26</td><td>24</td><td>41</td><td>36</td><td>62</td><td>58</td><td>78</td></tr><tr><td>Two Robots Stack Cube</td><td>14</td><td>19</td><td>22</td><td>24</td><td>21</td><td>37</td><td>40</td><td>44</td></tr><tr><td rowspan="3">Three-Agent</td><td>Pass Shoe</td><td>9</td><td>15</td><td>13</td><td>36</td><td>16</td><td>30</td><td>28</td><td>37</td></tr><tr><td>Take Photo of Tissue</td><td>7</td><td>17</td><td>19</td><td>31</td><td>19</td><td>33</td><td>31</td><td>42</td></tr><tr><td>Three Robots Stack Cube</td><td>8</td><td>19</td><td>21</td><td>25</td><td>21</td><td>28</td><td>32</td><td>35</td></tr><tr><td colspan="2">Average (Two-Agent)</td><td>21.0</td><td>29.8</td><td>29.3</td><td>43.0</td><td>41.8</td><td>56.5</td><td>56.5</td><td>72.0</td></tr><tr><td colspan="2">Average (Three-Agent)</td><td>8.0</td><td>17.0</td><td>17.7</td><td>30.7</td><td>18.7</td><td>30.3</td><td>30.3</td><td>38.0</td></tr><tr><td colspan="2">Overall Average</td><td>15.4</td><td>24.3</td><td>24.3</td><td>37.6</td><td>31.7</td><td>45.1</td><td>45.4</td><td>57.1</td></tr></table>

TABLE II. Ablation study results showing success rates (%) with 150 demonstrations

<table><tr><td>Agents</td><td>Task</td><td>ADM-DP</td><td>ADM-NoPC</td><td>ADM-NoTact</td><td>ADM-NoGraph</td><td>ADM-NoAM</td></tr><tr><td rowspan="4">Two-Agent</td><td>Lift Barrier</td><td>92</td><td>83</td><td>78</td><td>89</td><td>84</td></tr><tr><td>Pass Peg</td><td>74</td><td>64</td><td>70</td><td>67</td><td>68</td></tr><tr><td>Lift Arm</td><td>78</td><td>61</td><td>68</td><td>76</td><td>73</td></tr><tr><td>Two Robots Stack Cube</td><td>44</td><td>26</td><td>41</td><td>42</td><td>41</td></tr><tr><td rowspan="3">Three-Agent</td><td>Pass Shoe</td><td>37</td><td>32</td><td>30</td><td>28</td><td>28</td></tr><tr><td>Take Photo of Tissue</td><td>42</td><td>31</td><td>36</td><td>40</td><td>37</td></tr><tr><td>Three Robots Stack Cube</td><td>35</td><td>24</td><td>33</td><td>34</td><td>33</td></tr><tr><td>Average</td><td></td><td>57.4</td><td>45.9</td><td>50.9</td><td>53.7</td><td>52.0</td></tr></table>

## C. Ablation Studies

To analyze the contribution of each component in our architecture, we conduct ablation studies by systematically removing key modules. Table II presents the results with 150 demonstrations, where ADM-NoPC removes point cloud input (using only RGB), ADM-NoTact removes tactile sensing, ADM-NoGraph removes the graph-based collision encoding, and ADM-NoAM removes the AMAM module (using simple concatenation instead).

The ablation results reveal distinct patterns in component importance across different tasks. Removing point cloud input (ADM-NoPC) causes the most significant performance degradation overall (11.5% drop), particularly affecting visually-demanding tasks like Two Robots Stack Cube where success rate drops from 44% to 26%. This validates our enhanced visual encoding strategy that leverages both RGB semantics and point cloud geometry.

Tactile feedback is critical for tasks that demand grasp stability and force regulation. On Lift Barrier, removing tactile input leads to a 14% absolute drop in success rate (92% → 78%), as the policy can no longer reliably detect insufficient contact or execute corrective grasp refinements. Likewise, performance on Lift Arm decreases by 10%, further supporting the effectiveness of our tactile-guided grasping strategy for mitigating unstable grasps. As illustrated in Fig. 8, shallow grasps can induce slipping when tactile feedback is absent, due to the lack of signal to identify inadequate contact and adjust grasp depth accordingly.

The graph module’s importance is most evident in closeproximity coordination tasks. Pass Peg and Pass Shoe show 7% and 9% drops respectively without graph encoding, as agents lose spatial awareness and experience more frequent collisions during handovers. The graph features enable implicit coordination through shared TCP positions, critical for safe multi-agent interaction. Fig. 9 shows typical selfcollisions that occur without graph-based spatial awareness.

![](images/09f7e3136faae06775077a60dc66b2c29f2e10790268e34269b08d84171b5faf.jpg)  
Fig. 10. Dynamic modality weight adaptation by AMAM.

Removing AMAM (ADM-NoAM) results in a consistent performance decrease across all tasks (5.4% average drop), with the impact correlating strongly with task complexity. Multi-modal tasks show the largest degradation: Pass Shoe, which requires coordinated use of vision, tactile, and graph features, drops 9% (from 37% to 28%), while simpler vision-dominant tasks like Stack Cube only drop 3%. This pattern confirms that AMAM’s dynamic weighting becomes increasingly valuable as tasks demand integration of multiple sensory streams. Rather than using fixed fusion weights that may over-emphasize irrelevant modalities, AMAM adaptively allocates attention based on task context, preventing noise from unused sensors while ensuring critical modalities receive appropriate emphasis when needed. Fig. 10 visualizes this dynamic adaptation, showing AMAM automatically shifting from vision-dominant weights (0.70) before grasping to balanced multi-modal attention during grasp, where tactile increases to 0.35 for contact feedback while vision decreases to 0.50, demonstrating context-aware modality prioritization.

These ablations show that each component targets a distinct bottleneck in multi-agent manipulation: enhanced vision improves perception, tactile feedback stabilizes grasps, graph encoding reduces collisions, and AMAM enables contextdependent fusion.

## IV. CONCLUSIONS AND FUTURE WORK

We presented ADM-DP, a Multimoal Diffusion Policy for multi-agent robotic manipulation that addresses key challenges in coordination, grasping stability, and collision avoidance. Our approach combines four key innovations: (1) enhanced visual encoding through FiLM-based fusion of RGB and point cloud features, (2) tactile-guided grasping strategy with spatial encoding that enables dynamic grasp adjustment, (3) graph-based collision awareness through shared TCP positions, and (4) AMAM that dynamically allocates attention across modalities based on task context. Through extensive experiments on seven multi-agent tasks involving two and three robots, we demonstrated that ADM-DP sig nificantly outperforms state-of-the-art baselines, achieving 57.1% average success rate compared to 45.4% for the next best method. Our ablation studies confirmed that each component addresses specific challenges, with performance gains most pronounced in tasks requiring multiple sensory modalities. The decoupled training paradigm enables efficient scaling to multiple agents while maintaining modularity for system deployment.

Future work will focus on real robot deployment to validate ADM-DP’s effectiveness beyond simulation, addressing sensor noise and real-time constraints. We also plan to explore advanced tactile sensing technologies like visionbased tactile sensors for more complex manipulation tasks requiring delicate force control.

## REFERENCES

[1] A. Hussein, M. M. Gaber, E. Elyan, and C. Jayne, “Imitation learning: A survey of learning methods,” ACM Computing Surveys (CSUR), vol. 50, no. 2, pp. 1–35, 2017.

[2] C. Chi, Z. Xu, S. Feng, E. Cousineau, Y. Du, B. Burchfiel, R. Tedrake, and S. Song, “Diffusion policy: Visuomotor policy learning via action diffusion,” The International Journal of Robotics Research, p. 02783649241273668, 2023.

[3] Y. Ze, G. Zhang, K. Zhang, C. Hu, M. Wang, and H. Xu, “3d diffusion policy: Generalizable visuomotor policy learning via simple 3d representations,” arXiv preprint arXiv:2403.03954, 2024.

[4] J. Ma, Y. Qin, Y. Li, X. Liao, Y. Guo, and R. Zhang, “Cdp: Towards robust autoregressive visuomotor policy learning via causal diffusion,” arXiv preprint arXiv:2506.14769, 2025.

[5] S. Wu, Y. Zhu, Y. Huang, K. Zhu, J. Gu, J. Yu, Y. Shi, and J. Wang, “Afforddp: Generalizable diffusion policy with transferable affordance,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 6971–6980.

[6] Q. Zhang, Z. Liu, H. Fan, G. Liu, B. Zeng, and S. Liu, “Flowpolicy: Enabling fast and robust 3d flow-based policy via consistency flow matching for robot manipulation,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 14, 2025, pp. 14 754– 14 762.

[7] Y. Lipman, R. T. Chen, H. Ben-Hamu, M. Nickel, and M. Le, “Flow matching for generative modeling,” arXiv preprint arXiv:2210.02747, 2022.

[8] J. Chen, Z. Yang, H. G. Xu, D. Zhang, and G. Mylonas, “Multiagent systems for robotic autonomy with llms,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 4194–4204.

[9] J.-J. Jiang, X.-M. Wu, Y.-X. He, L.-A. Zeng, Y.-L. Wei, D. Zhang, and W.-S. Zheng, “Rethinking bimanual robotic manipulation: Learning with decoupled interaction framework,” arXiv preprint arXiv:2503.09186, 2025.

[10] Y. Qin, L. Kang, X. Song, Z. Yin, X. Liu, X. Liu, R. Zhang, and L. Bai, “Robofactory: Exploring embodied agent collaboration with compositional constraints,” arXiv preprint arXiv:2503.16408, 2025.

[11] M. Lai, K. Go, Z. Li, T. Kroger, S. Schaal, K. Allen, and J. Scholz,¨ “Roboballet: Planning for multirobot reaching with graph neural networks and reinforcement learning,” Science Robotics, vol. 10, no. 106, p. eads1204, 2025.

[12] Q. Lv, H. Li, X. Deng, R. Shao, Y. Li, J. Hao, L. Gao, M. Y. Wang, and L. Nie, “Spatial-temporal graph diffusion policy with kinematic modeling for bimanual robotic manipulation,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 17 394–17 404.

[13] B. Huang, Y. Wang, X. Yang, Y. Luo, and Y. Li, “3d-vitac: Learning fine-grained manipulation with visuo-tactile sensing,” arXiv preprint arXiv:2410.24091, 2024.

[14] H. Xue, J. Ren, W. Chen, G. Zhang, Y. Fang, G. Gu, H. Xu, and C. Lu, “Reactive diffusion policy: Slow-fast visual-tactile policy learning for contact-rich manipulation,” arXiv preprint arXiv:2503.02881, 2025.

[15] N. Gu, K. Kosuge, and M. Hayashibe, “Tactilealoha: Learning bimanual manipulation with tactile sensing,” IEEE Robotics and Automation Letters, 2025.

[16] Y. Lin, A. Church, M. Yang, H. Li, J. Lloyd, D. Zhang, and N. F. Lepora, “Bi-touch: Bimanual tactile manipulation with sim-to-real deep reinforcement learning,” IEEE Robotics and Automation Letters, vol. 8, no. 9, pp. 5472–5479, 2023.

[17] P. Velickoviˇ c, G. Cucurull, A. Casanova, A. Romero, P. Lio,´ and Y. Bengio, “Graph attention networks,” arXiv preprint arXiv:1710.10903, 2017.

[18] E. Perez, F. Strub, H. De Vries, V. Dumoulin, and A. Courville, “Film: Visual reasoning with a general conditioning layer,” in Proceedings of the AAAI conference on artificial intelligence, vol. 32, no. 1, 2018.

[19] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2016, pp. 770–778.

[20] C. R. Qi, H. Su, K. Mo, and L. J. Guibas, “Pointnet: Deep learning on point sets for 3d classification and segmentation,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 652–660.

[21] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable visual models from natural language supervision,” in International conference on machine learning. PmLR, 2021, pp. 8748–8763.

[22] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” Advances in neural information processing systems, vol. 33, pp. 6840–6851, 2020.

[23] J. Song, C. Meng, and S. Ermon, “Denoising diffusion implicit models,” arXiv preprint arXiv:2010.02502, 2020.

[24] T. Z. Zhao, V. Kumar, S. Levine, and C. Finn, “Learning fine-grained bimanual manipulation with low-cost hardware,” arXiv preprint arXiv:2304.13705, 2023.

[25] S. Tao, F. Xiang, A. Shukla, Y. Qin, X. Hinrichsen, X. Yuan, C. Bao, X. Lin, Y. Liu, T.-k. Chan et al., “Maniskill3: Gpu parallelized robotics simulation and rendering for generalizable embodied ai,” arXiv preprint arXiv:2410.00425, 2024.