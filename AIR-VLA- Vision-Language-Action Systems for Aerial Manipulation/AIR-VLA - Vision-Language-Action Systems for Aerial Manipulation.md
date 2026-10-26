# AIR-VLA: Vision-Language-Action Systems for Aerial Manipulation

Jianli Sun <sup>1</sup> Bin Tian <sup>1</sup> Qiyao Zhang <sup>2</sup> Chengxiang Li <sup>3</sup> Zihan Song <sup>4</sup> Zhiyong Cui <sup>5</sup> Yisheng Lv <sup>1</sup> Yonglin Tian <sup>1</sup> <sup>\*</sup>

![](<images/ea496fe57f9636f8fe018535162e137f7ed8d11ebcd77b58b8ea3cc81c1d33ed.jpg>)

Aerial Manipulation Task  
![](<images/9de897fce3f12fdf25898707473b6742a38432f04aad3a3d8906a6297a5d5ffb.jpg>)  
Figure 1. Overview of AIR-VLA model framework and dataset of aerial manipulation tasks.

## Abstract

While Vision-Language-Action (VLA) models have achieved remarkable success in groundbased embodied intelligence, their application to Aerial Manipulation Systems (AMS) remains a largely unexplored frontier. The inherent characteristics of AMS, including floating-base dynamics, strong coupling between the UAV and the manipulator, and the multi-step, long-horizon nature of operational tasks, pose severe challenges to existing VLA paradigms designed for static or 2D mobile bases. To bridge this gap, we propose AIR-VLA, the first VLA benchmark specifically tailored for aerial manipulation. We construct a physics-based simulation environment and release a high-quality multimodal dataset comprising 3000 manually teleoperated demonstrations, covering base manipulation, object & spatial understanding, semantic reasoning, and long-horizon planning. Leveraging this platform, we systematically evaluate mainstream VLA models and stateof-the-art VLM models. Our experiments not only validate the feasibility of transferring VLA paradigms to aerial systems but also, through multi-dimensional metrics tailored to aerial tasks, reveal the capabilities and boundaries of current models regarding UAV mobility, manipulator control, and high-level planning. AIR-VLA establishes a standardized testbed and data foundation for future research in general-purpose aerial robotics. The resource of AIR-VLA will be available at https://github.com/SpencerSon2001/AIR-VLA.

## 1. Introduction

With breakthrough advancements in Large Language Models (LLMs) and Vision-Language Models (VLMs), Embodied AI is undergoing a paradigm shift from specialized skill learning to general-purpose intelligence. Recently, Vision-Language-Action (VLA) models, represented by RT-1 (Brohan et al., 2023), OpenVLA (Kim et al., 2024), and π<sub>0</sub> (Black et al., 2026), have demonstrated exceptional capability in handling open-world tasks driven by natural language instructions. However, existing VLA research is predominantly confined to Ground Mobile Manipulators, where the operational space is restricted to 2D planar navigation and limited working heights. This significantly constrains the potential of embodied intelligence in exploring broader three-dimensional physical spaces.

As a pivotal yet challenging branch of embodied intelligence, Aerial Manipulation Systems (AMS) integrate the high mobility of multi-rotor UAVs with the dexterous manipulation capabilities of robotic arms. By transcending the physical limitations of terrain and altitude, AMS demonstrate irreplaceable value in complex scenarios such as logistics transportation, disaster rescue, infrastructure inspection, and high-altitude operations. However, extending VLA models to aerial platforms introduces unique physical and control challenges. AMS must contend not only with the instability of a floating base and the kinematic coupling between the UAV and the manipulator, but also with the demands of 3D spatial reasoning within an expansive workspace and the inherently long-horizon, multi-step nature of operational tasks. Consequently, this requires that models possess not only general semantic understanding, but also the mastery of complex coordination strategies, ranging from 3D omnidirectional navigation to fine-grained end-effector manipulation, along with high-level planning capabilities for complex missions.

Despite its promising prospects, research on VLA for aerial manipulation currently faces a severe “data scarcity” and an “absence of benchmarks”. Existing embodied intelligence benchmarks are predominantly designed for tabletop manipulation(Liu et al., 2023) or ground navigation(Yenamandra et al., 2023). Consequently, they fail to effectively evaluate omnidirectional mobility in 3D space, manipulation robustness under floating-base conditions, and long-horizon temporal reasoning capabilities conditioned on language. Furthermore, current research applying VLA to the UAV domain remains focused on navigation(Lykov et al., 2025), task generation(Sautenkov et al., 2025b), or single-DoF VLA manipulation tightly coupled with path planning(Mehboob et al., 2026), leaving the application of VLA to full-spectrum aerial manipulation tasks largely unexplored.

However, with the maturation of AMS hardware and control theory, combined with the rapid advancement of VLMs in robotics, applying the scene generalization capabilities of VLAs and the high-level planning abilities of VLMs to aerial manipulation tasks has demonstrated immense potential. Therefore, there is an urgent need in the academic community for a dataset and evaluation framework specifically focused on aerial manipulation.

To address these challenges, we propose AIR-VLA, the first VLA training and evaluation benchmark designed specifically for Aerial Manipulation Systems. We construct a simulation environment integrated with a physics engine based on NVIDIA Isaac Sim, simulating the collaborative operation of UAVs and manipulators under complex dynamic conditions. On this basis, we construct and release a multimodal dataset covering tasks ranging from basic primitive actions to complex 3D dynamic interactions requiring longhorizon planning. Spanning diverse simulated scenarios, covering indoor rooms, factories, and outdoor environments, the dataset comprises 3000 high-quality manually teleoperated aerial manipulation episodes. AIR-VLA aims to provide a comprehensive performance profile of aerial embodied intelligence across multiple dimensions, including basic motion control, fine-grained visual and spatial understanding, semantic instruction following, and long-horizon task planning.

The main contributions of this paper are summarized as follows:

• Pioneering Aerial Manipulation VLA Benchmark: We propose the first VLA benchmark testbed specifically designed for AMS, filling the evaluation gap in the domain of 3D aerial manipulation and providing a standardized research tool for the community. Tailored to the unique characteristics of aerial operations, we design a multi-suite dataset rich in sensory information (RGB, depth, proprioception) and diverse language instructions, providing high-quality data support for training aerial VLA manipulation policies.

• Tailored Evaluation Metrics: Addressing the unique characteristics of aerial manipulation, specifically the tight coordination between the UAV and manipulator and the long-horizon, multi-step nature of tasks, we move beyond relying solely on task success rates. We introduce multi-dimensional evaluation metrics designed to assess the aerial manipulation performance of VLAs and the long-horizon planning capabilities of VLMs. These metrics, which cover dimensions such as UAV localization, navigation, and spatial perception, allow a deep exploration of the performance boundaries of the models.

• Comprehensive Baseline Analysis: We conduct extensive benchmarking and in-depth analysis of mainstream VLA and VLM models on AIR-VLA. By quantifying the performance of current mainstream VLA models on aerial tasks and the high-level planning capabilities of VLMs, we reveal critical challenges in the transfer process from ground to aerial platforms and provide guidance for future research directions.

## 2. Related Works

## 2.1. Aerial Manipulation Systems

AMS extend UAV capabilities from passive reconnaissance to active physical interaction, though the coupling between the floating base and manipulator introduces severe nonlinear dynamics. In mechanical design, research has prioritized bio-inspired and compliant architectures to enhance adaptability and stability, such as owl-inspired limbs (Liu et al., 2024) and continuum elephant-trunk manipulators (Peng et al., 2025). Dual-arm systems have also been explored to execute complex multi-step tasks infeasible for single-arm platforms (Ghorbani et al., 2023). On the control frontier, strategies address external disturbances and uncertainties through refined anti-disturbance techniques (Liu et al., 2024), adaptive prescribed performance control (Liang et al., 2023), and Reinforcement Learning (RL)-driven hierarchical policies (Deshmukh et al., 2025). Additionally, visual impedance control (Xu et al., 2023) and whole-body planning (Wang et al., 2025a) are employed to improve interaction precision. Despite these advances, traditional AMS frameworks generally lack semantic grounding. Operating under closed-world assumptions prevents them from generalizing to unstructured environments or interpreting abstract instructions (Zhang et al., 2025; Carvajal et al., 2024; Deng et al., 2025), limiting their utility as intelligent assistants.

## 2.2. Vision-Language-Action Models

VLA models shift Embodied AI from modular pipelines to end-to-end architectures, unifying perception, language, and action into a single transformer backbone. Foundational models like RT-1 (Brohan et al., 2023) and OpenVLA (Kim et al., 2024) demonstrate that large-scale training enables open-vocabulary task execution. The field is rapidly evolving toward high-frequency control and long-horizon capabilities. The π series, including $\pi _ { 0 }$ (Black et al., 2026) and $\pi _ { 0 . 5 }$ (Black et al., 2025), introduces flow-matching for smooth action generation. Notably, $\pi _ { 0 . 6 }$ (Intelligence et al., 2025) exhibits robust autonomous execution of long-horizon tasks, indicating that VLAs are maturing into reliable systems for complex physical interaction.

## 2.3. VLA for Aerial Robotics

Recognizing VLA potential, recent works have integrated these models into aerial platforms, focusing primarily on navigation and trajectory generation. Benchmarks like IndoorUAV (Liu et al., 2025) leverage task decomposition and $\pi _ { 0 }$ for 3D language grounding, while RaceVLA (Serpiva et al., 2025) enables high-speed agility via videoconditioned control. To address onboard constraints, frameworks utilizing 3D Gaussian Splatting (Wu et al., 2025), fine-grained instruction mapping (Wang et al., 2025c), and cognitive reasoning fusion (Lykov et al., 2025) have been proposed. However, current applications remain predominantly confined to trajectory planning or high-level mission generation. Approaches like UAV-VLA (Sautenkov et al., 2025b) and DroneVLA (Mehboob et al., 2026) largely limit the VLA’s role to mission planning or simple 1-DoF gripping. These methods fail to leverage the UAV’s potential for complex physical interaction in unstructured environments. The lack of a mature framework for full-spectrum aerial manipulation represents a critical research gap.

## 2.4. Benchmarks for Robot Manipulation

High-quality benchmarks drive VLA research but currently focus on ground-based platforms. For static manipulation, LIBERO (Liu et al., 2023) and RoboTwin 2.0 (Chen et al., 2025) evaluate lifelong learning and dual-arm coordination, respectively. For mobile manipulation, OVMM (Yenamandra et al., 2023) and Mobile ALOHA (Fu et al., 2024) address open-vocabulary tasks and low-cost imitation learning. While comprehensive for ground robots, these benchmarks are ill-suited for AMS. Specifically, they lack the robotic platforms, corresponding kinematic designs, and aerial manipulation task data specific to Aerial Manipulation Systems. This absence restricts the development of aerial VLA models. To bridge this gap, we propose AIR-VLA, the first benchmark specifically designed to extend VLA intelligence to the challenging realm of aerial mobile manipulation.

## 3. AIR-VLA

To comprehensively evaluate the generalization capability and policy robustness of VLA models in the aerial manipulation domain, AIR-VLA is systematically designed across four dimensions: task taxonomy, evaluation framework, simulation environment, and dataset construction. This design aims to provide standardized metrics while simulating authentic physical characteristics.

## 3.1. Task Design Taxonomy

AIR-VLA comprises four core task suites targeting manipulation precision, semantic reasoning, visual perception, and planning. Unlike ground-based benchmarks, it emphasizes deep coordination between 3D UAV positioning and manipulator actions, covering a spectrum from atomic actions to complex interactions. Tailored to AMS characteristics, the dataset features an average task length of approximately 475 time steps, significantly exceeding traditional benchmarks to reflect temporal complexity.The specific objectives of the four suites are:

![](<images/3b6c339196117fb62a7eae29a38ee4a36caf402b1caf2e9ad6618d5d2a104a51.jpg>)  
Figure 2. Overview of the AIR-VLA benchmark. AIR-VLA serves as a full-stack Vision-Language-Action platform tailored for aerial manipulation systems. It integrates a simulation-based teleoperation data acquisition pipeline, an online simulation environment, and diverse multimodal datasets. Furthermore, it provides a comprehensive benchmark for evaluating mainstream VLA and VLM models across varied aerial manipulation tasks.

• Base Manipulation: Evaluates low-level motion control and UAV-arm coordination with minimal distractors. Tasks prioritize optimal 3D hovering and precise manipulation, requiring the model to handle complex contact dynamics.

• Object & Spatial: Assesses fine-grained understanding of physical attributes (color, shape, material) and spatial relationships. This suite imposes stringent requirements on cross-modal alignment and 3D geometric reasoning.

• Semantic Understanding: Tests robustness to unstructured language instructions. Tasks feature diverse styles and implicit intents, requiring the model to infer underlying goals from complex natural language.

• Long-Horizon: Evaluates multi-step reasoning with temporal dependencies. The AMS must alternate between wide-range maneuvers and local fine-grained operations. Any coordination failure in the chain results in total task failure, representing the most challenging scenario.

## 3.2. Evaluation Framework

Compared to traditional ground robot tasks, aerial mobile manipulation introduces unique challenges such as dynamic coupling of the floating base, volumetric workspaces, and temporal complexity of long-horizon tasks. To comprehensively evaluate the limit performance of embodied intelligence models in this high-dimensional domain, we constructed a two-layer evaluation framework covering perception, planning, and control. For VLA models, we conduct closed-loop evaluations based on an online simulation environment, focusing on UAV-Arm Coordination capabilities under real-time inference. Specifically, we quantify the feasibility of UAV spatial navigation, the manipulation precision of the end-effector, and dynamic safety during longhorizon task execution. For long-horizon tasks, considering the critical role of logical reasoning, we propose a multidimensional evaluation metric set for the high-level task planning capabilities of VLMs. This assesses the embodied reasoning ability of VLMs from dimensions including sub-task process construction, 3D spatial perception, finegrained object grounding, and atomic skill selection. For detailed definitions and mathematical formulations of these metrics, please refer to Appendix A.3.

## 3.3. Simulation Environment

Simulator. Built on NVIDIA Isaac Sim, AIR-VLA leverages the Omniverse platform to bridge the Sim-to-Real gap. We utilize the PhysX 5 engine and ray-tracing to ensure realistic simulation from fluid dynamic disturbances to lighting reflections. Furthermore, the platform’s USD pipeline and GPU parallel acceleration enable flexible scene configuration and efficient data collection.

Robot System. To fully test the adaptability of VLA models to high-dimensional state spaces, we constructed an AMS consisting of a quadrotor UAV equipped with a 7- DoF Franka Panda manipulator. The UAV control space is defined as position changes in Cartesian coordinates and Yaw rotation; the manipulator control space includes 7 joint poses and the gripper state. This configuration constitutes a high-dimensional control problem characterized by high redundancy and coupling.

## 3.4. Dataset Construction

Data collection utilizes the Isaac Sim-based environment, emphasizing diversity and physical realism.

Data Collection Strategy. To capture atypical aerial coordination patterns, we employ Human Teleoperation rather than scripted generation. Expert demonstrators utilize gamepad interfaces to generate task-oriented trajectories, covering multiple collaborative solutions for the UAV and manipulator within unified tasks.

Multimodal Perception System. Tailored to aerial perspectives, the sensor configuration comprises: (1) a UAV front-down RGB-D camera for global bird’s-eye views, (2) a manipulator wrist RGB-D camera for local fine-grained observation, and (3) a traditional third-person perspective camera.

Action and State Space. The dataset records full proprioceptive information, including the UAV’s 4D pose and derivatives (velocity/angular velocity), manipulator joint poses, and end-effector pose. Standardized data interfaces ensure compatibility with the input layers of diverse VLA models.

Scene and Instruction Diversity. To enhance generalization, we constructed diverse environments (residential, industrial, outdoor) with varying lighting conditions. Furthermore, LLMs are leveraged to generate natural language instructions featuring complex structures and implicit intents, ensuring dense coverage of the semantic space.

## 4. Experiments

## 4.1. VLA Experiments

We evaluated a series of advanced VLA models to assess their performance in aerial manipulation tasks. While VLA models have demonstrated superior efficacy in fixed-base and ground mobile manipulation tasks, we apply them to AMS for the first time. Our experiments are structured around the following research questions:

Q1: Can VLA models maintain superior task completion performance in high-DoF (12-DoF in experiments) robotic system manipulation tasks?

Q2: Can VLA models cope with external disturbances in AMS and complete tasks under random base jitter?

Q3: What is the dependency of VLA models on visual inputs in AMS tasks, and which viewing angles significantly enhance task completion?

Q4: How do VLA models perform in UAV movement and manipulator operation respectively in AMS tasks, and how is their coordination?

Q5: Safety constraints are critical in robotic tasks. Do VLA models introduce more severe safety issues in AMS tasks?

## 4.1.1. EXPERIMENTAL SETUP

Baseline Models. To establish a representative benchmark, we evaluate six diverse models: π (Black et al., 2026) and π (Black et al., 2025), Flow Matching-based foundation models pre-trained on cross-embodiment data, represent large-scale transfer capabilities to the aerial domain. π<sub>0</sub>-FAST(Pertsch et al., 2025) serves as an efficiencyoptimized variant, utilizing step distillation to enable the high-frequency inference crucial for agile aerial maneuvers. ACT (Zhao et al., 2023) combines CVAEs and Transformers to predict “Action Chunks,” effectively modeling long-term dependencies. Diffusion Policy (Chi et al., 2024) utilizes conditional diffusion for iterative denoising, robustly handling multi-modal distributions essential for floating-base stability.

Dataset & Training. The training dataset is derived from our human-teleoperated simulation data, fully reflecting the diversity of collaborative actions between UAV and manipulator. We adopted a differentiated fine-tuning strategy based on task difficulty: 30 trajectories were used for fine-tuning simple single-step tasks with less physical interaction, while 50 trajectories were used for multi-step long-horizon tasks or tasks with rich physical interactions.

Robustness Evaluation. To simulate real flight environments (Sim-to-Real Gap), we introduced dynamic disturbance test scenarios. During testing, Gaussian noise was injected into the UAV’s state space to simulate the effects of wind gusts or sensor drift on flight stability. Additionally, we designed a ”No Fixed Third-Person View” test, retaining only onboard and wrist cameras, to investigate the model’s dependency on global perspectives.

## 4.1.2. MAIN RESULTS AND ANALYSIS

In this section, we report the quantitative performance of baseline models on the four task suites of AIR-VLA. Table 1 presents the comprehensive evaluation results under standard disturbance-free conditions.

Overall Performance. Experimental results indicate that large-scale pre-trained models, represented by $\pi _ { 0 . 5 }$ and $\pi _ { 0 } ,$ demonstrate significant advantages in the AIR-VLA evaluation, outperforming traditional imitation learning baselines such as ACT and Diffusion Policy across all metrics. However, in the complex domain of aerial manipulation tasks, $\pi _ { 0 } .$ -FAST exhibits a marked performance decline compared to $\pi _ { 0 } .$ suggesting that the performance degradation induced by distillation and step reduction is significantly amplified in aerial scenarios.

Overall, the results validate the feasibility of transferring VLA paradigms to Aerial Manipulation Systems. Furthermore, they demonstrate that pre-training on massive crossembodiment data (including mobile robotics) enables models to acquire generalizable mechanical manipulation priors. Notably, even under few-shot fine-tuning settings with only 30–50 demonstrations, foundation models like $\pi _ { 0 . 5 }$ rapidly adapt to aerial manipulation paradigms unseen during pretraining, achieving the highest performance among all evaluated models. Nevertheless, there remains substantial room for improvement in overall task completion rates. Compared to low-DoF ground-based platforms, the performance of existing VLA models on high-DoF aerial platforms remains suboptimal.

$\pi _ { 0 }$ achieves its peak success rate in Base Manipulation tasks characterized by minimal environmental interference; however, its performance exhibits a significant decaying trend with the introduction of intense visual distractors, complex semantic instructions, and long-horizon planning requirements. Specifically, within the Object & Spatial suite, dense visual distractors lead to a marked increase in target misidentification rates. Notably, in spatial understanding tasks, the models exhibit Spatial Grounding Failure: although the correct object category is identified, the agent manipulates an identical object at an incorrect location due to an inability to accurately parse relative positional instructions. In comparison, $\pi _ { 0 . 5 }$ demonstrates superior robustness against object interference relative to $\pi _ { 0 }$ . In the Semantic Understanding suite, where visual scene complexity is controlled to be comparable to that of the Object & Spatial suite, we observe a slight performance decline in both $\pi _ { 0 . 5 }$ and $\pi _ { 0 } ,$ , suggesting that complex semantic instructions impose a distinct penalty on model performance. In the Long-Horizon suite, we observe that successful executions are predominantly confined to the initial sub-task, with performance on the subsequent sub-task exhibiting notable degradation compared to the first.

Regarding collaborative control, VLA models demonstrate significantly better performance in UAV motion control than in manipulator operation. This is partly attributable to the fact that UAV movement in aerial manipulation is relatively coarse-grained, whereas manipulator operation necessitates extremely high precision. Existing models have successfully acquired UAV navigation strategies to guide the system to feasible operating positions; however, executing fine-grained manipulation at these positions remains a critical technical bottleneck. Furthermore, safety constraints cannot be overlooked. Due to the inherent characteristics of the floating base, collisions and unreasonable physical interactions cause significantly more severe disturbances to the system than in ground-based robotics. Our evaluation revealed destructive interactions with the environment in certain episodes, indicating that addressing safety constraints in Aerial Manipulation Systems remains a pivotal direction for future research.

Robustness Analysis. We selected the best-performing model, $\pi _ { 0 . 5 } ,$ for UAV dynamic disturbance and third-person lacking tests (results in Table 2). Under UAV disturbance, the decline in performance metrics for $\pi _ { 0 . 5 }$ is relatively small, indicating that VLA models possess a degree of implicit compensation capability. However, in the absence of fixed third-person view input, model performance decays significantly. This reveals the current models’ dependency on stable global perspectives; the lack of a global view severely limits the system’s executable space and exploration capabilities. Introducing active perception strategies based on onboard UAV cameras may be an effective solution to this problem.

## 4.2. VLM Experiments

Leveraging the capability of VLMs to integrate visual and linguistic instructions for high-level planning, we conduct a comprehensive evaluation of several mainstream VLMs. Specifically, we assess their planning performance across the dimensions of Process Planning, Spatial Navigation, Object Grounding, and Skill Selection, aiming to verify their potential for aerial manipulation tasks.

Table 1. Detailed performance evaluation of models across four task suites and overall average. The table displays normalized scores for sub-metrics and weighted total scores for each model under different task types. Weights are set as $w _ { p o s } = 0 . 2 5 , w _ { a r m } =$ 0.25, $w _ { s a f e } = 0 . 1 0 , w _ { t a s k } = 0 .$ 40. Abbreviations: Pos: Base Positioning Accuracy $( S _ { p o s } ) ;$ Arm: Manipulator Efficacy $( S _ { a r m } ) ;$ Safe: Environmental Safety $( S _ { s a f e } ) ;$ Task: Task Progression $( S _ { t a s k } ) ;$ Tot: Weighted Total Score $( S _ { t o t a l } )$

<table><tr><td rowspan="2">Model</td><td colspan="5">Base Manipulation</td><td colspan="5">Object &amp; Spatial</td><td colspan="5">Semantic Understanding</td><td colspan="5">Long-Horizon</td><td colspan="5">Overall Average</td></tr><tr><td>Pos</td><td>Arm</td><td>Safe</td><td>Task</td><td>Tot</td><td>Pos</td><td>Arm</td><td>Safe</td><td>Task</td><td>Tot</td><td>Pos</td><td>Arm</td><td>Safe</td><td>Task</td><td>Tot</td><td>Pos</td><td>Arm</td><td>Safe</td><td>Task</td><td>Tot</td><td>Pos</td><td>Arm</td><td>Safe</td><td>Task</td><td>Tot</td></tr><tr><td>ACT</td><td>0.0</td><td>20.0</td><td>100.0</td><td>0.0</td><td>15.0</td><td>0.0</td><td>16.0</td><td>100.0</td><td>0.0</td><td>14.0</td><td>0.0</td><td>16.0</td><td>100.0</td><td>0.0</td><td>14.0</td><td>0.0</td><td>10.0</td><td>100.0</td><td>0.0</td><td>12.5</td><td>0.0</td><td>16.5</td><td>100.0</td><td>0.0</td><td>13.9</td></tr><tr><td>Diffusion Policy</td><td>0.0</td><td>20.0</td><td>100.0</td><td>0.0</td><td>15.0</td><td>0.0</td><td>16.0</td><td>100.0</td><td>0.0</td><td>14.0</td><td>0.0</td><td>16.0</td><td>100.0</td><td>0.0</td><td>14.0</td><td>0.0</td><td>10.0</td><td>100.0</td><td>0.0</td><td>12.5</td><td>0. 0</td><td>16.5</td><td>100.0</td><td>0.0</td><td>13.9</td></tr><tr><td> $\pi_0$ -FAST</td><td>27.5</td><td>23.9</td><td>98.1</td><td>0.0</td><td>22.7</td><td>5.1</td><td>18.9</td><td>98.3</td><td>0.0</td><td>15.8</td><td>6.5</td><td>16.9</td><td>99.3</td><td>0.0</td><td>15.8</td><td>41.0</td><td>32.6</td><td>99.4</td><td>0.0</td><td>28.3</td><td>15.0</td><td>20.8</td><td>98.5</td><td>0.0</td><td>18.8</td></tr><tr><td> $\pi_0$ </td><td>56.6</td><td>48.0</td><td>87.5</td><td>8.0</td><td>38.1</td><td>58.1</td><td>38.2</td><td>75.4</td><td>2.0</td><td>32.4</td><td>55.7</td><td>36.1</td><td>85.1</td><td>2.00</td><td>32.3</td><td>65.1</td><td>38.5</td><td>85.9</td><td>2.0</td><td>32.3</td><td>58.8</td><td>42.0</td><td>83.5</td><td>3.50</td><td>34.5</td></tr><tr><td> $\pi_{0.5}$ </td><td>78.8</td><td>49.9</td><td>87.2</td><td>11.0</td><td>45.3</td><td>70.9</td><td>44.1</td><td>81.2</td><td>13.5</td><td>42.3</td><td>65.4</td><td>44.7</td><td>82.7</td><td>11.0</td><td>40.2</td><td>62.4</td><td>40.8</td><td>84.8</td><td>7.0</td><td>37.1</td><td>71.3</td><td>45.9</td><td>84.7</td><td>10.6</td><td>42.0</td></tr></table>

Table 2. Robustness evaluation of $\pi _ { 0 . 5 }$ under disturbance and perception-deprived conditions. The table shows absolute scores for each metric, with values in parentheses indicating the performance drop (↓ ∆) compared to the Standard condition.

<table><tr><td>Condition</td><td> $S_{pos}$ </td><td> $S_{arm}$ </td><td> $S_{safe}$ </td><td> $S_{task}$ </td><td> $S_{total}$ </td></tr><tr><td>Standard (None)</td><td>71.3</td><td>45.9</td><td>84.7</td><td>10.6</td><td>42.0</td></tr><tr><td>UAV Disturb.</td><td>68.2 (↓3.1)</td><td>45.6 (↓0.3)</td><td>83.2 (↓1.5)</td><td>10.5 (↓0.1)</td><td>41.0 (↓1.0)</td></tr><tr><td>View Lacking</td><td>58.9 (↓12.4)</td><td>35.6 (↓10.3)</td><td>85.6 (↑0.9)</td><td>5.1 (↓5.5)</td><td>34.5 (↓7.5)</td></tr></table>

## 4.2.1. EXPERIMENTAL SETUP

Baseline Models. To establish a representative benchmark, we analyze six diverse open-source VLMs. Molmo-7B-D-0924 (Deitke et al., 2024) employs a data-centric strategy via the PixMo dataset for efficient alignment. The Qwen series, comprising Qwen3-VL-8B-Instruct (Bai et al., 2025a) and Qwen2.5-VL-7B-Instruct (Bai et al., 2025b), excels in adaptability; the former leverages dynamic resolution, while the latter utilizes M-RoPE to unify image and video understanding. GLM-4V-9B (GLM et al., 2024) focuses on deep vision-language alignment for high-resolution features. InternVL3 5-8B (Wang et al., 2025b) adopts a “strong vision encoder + strong language decoder” design, targeting geometric perception. Finally, llava-onevisionqwen2-7b-ov-hf (Li et al., 2024) balances static and temporal analysis through a unified mixed-data paradigm.

Assessment Protocol. In the VLM evaluation phase, we adopted a non-interactive evaluation process. We input the initial four-view images of the task scene and description instructions to the VLM, providing skill library descriptions, output format requirements, and few-shot examples as System Prompts. The VLM is required to generate highlevel task planning sequences conforming to specifications, enabling standardized capability assessment.

## 4.2.2. RESULTS AND ANALYSIS

Quantitative results demonstrate that Qwen3-VL-8B-Instruct and Qwen2.5-VL-7B-Instruct establish a substantial performance advantage in high-level planning for aerial manipulation tasks. Notably, Qwen3-VL achieves state-ofthe-art (SOTA) performance across all baseline models in four core dimensions: Process Planning, Spatial Navigation, Object Grounding, and Skill Selection, highlighting its exceptional potential as a multimodal “brain.”

Regarding metric distribution, the capability profiles across models exhibit high consistency. While demonstrating strong performance in semantic understanding and skill matching, all models reveal a significant deficiency in Spatial Navigation. In-depth analysis indicates that this lack of 3D spatial awareness is the primary bottleneck limiting the end-to-end planning Success Rate.

In terms of instruction robustness, the performance gap between explicit, structured instructions and implicit, complex instructions is negligible. This suggests that current VLMs possess powerful semantic generalization capabilities, effectively aligning unstructured linguistic inputs with specific operational intents. Regarding task scenarios, VLMs demonstrate excellent temporal stability; their performance in Long-Horizon tasks does not exhibit significant deterioration, standing in sharp contrast to the performance decay often observed in traditional VLA models over long sequences. However, in scenarios introducing visual Distractor Interference, most models show a slight decline in object grounding metrics, reflecting the challenge complex backgrounds pose to fine-grained visual perception.

In summary, VLMs hold immense potential for high-level planning in aerial manipulation, particularly in mitigating the long-horizon reasoning limitations of VLA models. However, overcoming the critical bottleneck of spatial orientation understanding remains a pivotal challenge for achieving highly reliable aerial agents in the future.

![](<images/70c5a85461698f25f1ac734a5c594e94f755a15b16c4ac898db22b8da967406a.jpg>)  
Figure 3. Evaluation pipeline for VLM high-level planning capabilities in aerial manipulation tasks.

Table 3. Detailed evaluation of VLM task planning capabilities. The table displays normalized sub-metric scores and planning success rates (Succ, %) for each model across different task scenarios and instruction types. Models are sorted in ascending order by their Overall Average Total score. Abbreviations: Exp.: Explicit; Imp.: Implicit. Weights: $\bar { w } _ { p l a n } = 0 . 2 5 , w _ { l o c } = 0 . 2 5 , w _ { o b j } = 0 . 2 5 , w _ { s k i l l } = 0 . 2 5$

<table><tr><td rowspan="2">Model</td><td rowspan="2">Instr.</td><td colspan="6">Base Manipulation</td><td colspan="6">Distractor Interference</td><td colspan="6">Long-Horizon</td><td colspan="5">Overall Average</td><td></td></tr><tr><td>Plan</td><td>Loc</td><td>Obj</td><td>Skill</td><td>Tot</td><td>Succ</td><td>Plan</td><td>Loc</td><td>Obj</td><td>Skill</td><td>Tot</td><td>Succ</td><td>Plan</td><td>Loc</td><td>Obj</td><td>Skill</td><td>Tot</td><td>Succ</td><td>Plan</td><td>Loc</td><td>Obj</td><td>Skill</td><td>Tot</td><td>Succ</td></tr><tr><td rowspan="2">LLaVA-OV</td><td>Exp.</td><td>10.0</td><td>38.0</td><td>0.0</td><td>80.0</td><td>32.0</td><td>0.0</td><td>12.0</td><td>38.0</td><td>0.0</td><td>75.0</td><td>31.3</td><td>0.0</td><td>72.0</td><td>29.3</td><td>22.5</td><td>62.5</td><td>46.6</td><td>0.0</td><td rowspan="2">31.8</td><td rowspan="2">35.4</td><td rowspan="2">9.6</td><td rowspan="2">73.2</td><td rowspan="2">37.5</td><td rowspan="2">0.0</td></tr><tr><td>Imp.</td><td>6.0</td><td>38.0</td><td>0.0</td><td>85.0</td><td>32.3</td><td>0.0</td><td>22.0</td><td>43.0</td><td>10.0</td><td>70.0</td><td>36.3</td><td>0.0</td><td>68.7</td><td>26.0</td><td>25.0</td><td>66.7</td><td>46.6</td><td>0.0</td></tr><tr><td rowspan="2">Molmo-7B</td><td>Exp.</td><td>33.0</td><td>42.0</td><td>0.0</td><td>70.0</td><td>36.3</td><td>0.0</td><td>38.0</td><td>47.0</td><td>0.0</td><td>50.0</td><td>33.8</td><td>0.0</td><td>72.0</td><td>37.8</td><td>13.3</td><td>77.5</td><td>50.2</td><td>0.0</td><td rowspan="2">47.6</td><td rowspan="2">40.4</td><td rowspan="2">4.9</td><td rowspan="2">61.2</td><td rowspan="2">38.5</td><td rowspan="2">0.0</td></tr><tr><td>Imp.</td><td>27.0</td><td>37.0</td><td>0.0</td><td>55.0</td><td>29.8</td><td>0.0</td><td>43.0</td><td>49.0</td><td>0.0</td><td>40.0</td><td>33.0</td><td>0.0</td><td>72.4</td><td>29.5</td><td>15.8</td><td>75.0</td><td>48.2</td><td>0.0</td></tr><tr><td rowspan="2">InternVL3.5</td><td>Exp.</td><td>23.0</td><td>0.0</td><td>30.0</td><td>100.0</td><td>38.3</td><td>0.0</td><td>32.5</td><td>8.0</td><td>20.0</td><td>80.0</td><td>35.1</td><td>0.0</td><td>37.7</td><td>3.5</td><td>100</td><td>100</td><td>60.3</td><td>0.0</td><td rowspan="2">31.8</td><td rowspan="2">3.2</td><td rowspan="2">56.1</td><td rowspan="2">89.4</td><td rowspan="2">45.2</td><td rowspan="2">0.0</td></tr><tr><td>Imp.</td><td>30.0</td><td>0.0</td><td>80.0</td><td>100</td><td>52.5</td><td>0.0</td><td>32.0</td><td>8.0</td><td>30.0</td><td>80.0</td><td>37.5</td><td>0.0</td><td>35.6</td><td>0.0</td><td>76.7</td><td>76.7</td><td>47.2</td><td>0.0</td></tr><tr><td rowspan="2">GLM-4V</td><td>Exp.</td><td>76.5</td><td>7.00</td><td>65.0</td><td>90.0</td><td>59.6</td><td>10.0</td><td>81.0</td><td>14.0</td><td>20.0</td><td>85.0</td><td>50.0</td><td>0.0</td><td>76.1</td><td>26.7</td><td>36.7</td><td>75.0</td><td>53.6</td><td>10.0</td><td rowspan="2">79.8</td><td rowspan="2">16.8</td><td rowspan="2">46.2</td><td rowspan="2">84.4</td><td rowspan="2">56.8</td><td rowspan="2">6.7</td></tr><tr><td>Imp.</td><td>80.0</td><td>11.0</td><td>60.0</td><td>100.0</td><td>62.8</td><td>10.0</td><td>79.0</td><td>13.0</td><td>40.0</td><td>80.0</td><td>53.0</td><td>0.0</td><td>86.0</td><td>29.3</td><td>55.8</td><td>76.7</td><td>62.0</td><td>10.0</td></tr><tr><td rowspan="2">Qwen2.5-VL</td><td>Exp.</td><td>94.0</td><td>37.0</td><td>75.0</td><td>100</td><td>76.5</td><td>10.0</td><td>92.0</td><td>46.0</td><td>50.0</td><td>90.0</td><td>69.5</td><td>0.0</td><td>76.2</td><td>27.2</td><td>60.9</td><td>75.0</td><td>59.8</td><td>20.0</td><td rowspan="2">92.2</td><td rowspan="2">38.7</td><td rowspan="2">75.1</td><td rowspan="2">88.9</td><td rowspan="2">73.7</td><td rowspan="2">13.3</td></tr><tr><td>Imp.</td><td>100.0</td><td>37.0</td><td>100.0</td><td>100.0</td><td>84.3</td><td>20.0</td><td>98.0</td><td>44.0</td><td>95.0</td><td>80.0</td><td>79.3</td><td>10.0</td><td>93.0</td><td>41.2</td><td>70.0</td><td>88.3</td><td>73.1</td><td>20.0</td></tr><tr><td rowspan="2">Qwen3-VL</td><td>Exp.</td><td>93.0</td><td>36.0</td><td>100.0</td><td>100.0</td><td>82.3</td><td>20.0</td><td>100.0</td><td>49.0</td><td>90.0</td><td>100</td><td>84.8</td><td>20.0</td><td>98.7</td><td>43.5</td><td>90.0</td><td>90.0</td><td>80.5</td><td>30.0</td><td rowspan="2">96.9</td><td rowspan="2">43.5</td><td rowspan="2">93.3</td><td rowspan="2">96.7</td><td rowspan="2">82.4</td><td rowspan="2">23.3</td></tr><tr><td>Imp.</td><td>93.0</td><td>38.0</td><td>100.0</td><td>100.0</td><td>82.8</td><td>30.0</td><td>100.0</td><td>44.0</td><td>90.0</td><td>100.0</td><td>83.5</td><td>10.0</td><td>96.7</td><td>46.0</td><td>90.0</td><td>90.0</td><td>80.7</td><td>30.0</td></tr></table>

## 5. Conclusion

This paper presents AIR-VLA, the first full-stack VLA benchmark for Aerial Manipulation Systems. Through the construction of a simulation environment, a physics-aware dataset, and a multi-dimensional evaluation framework, we systematically quantify the capability boundaries of current mainstream VLA models in 3D aerial manipulation tasks.

Our findings reveal that while transferring pre-trained VLA models to aerial platforms is feasible, existing models still face severe challenges in handling floating-base dynamic coupling, 3D long-horizon planning, and disturbance rejection. AIR-VLA not only provides a standardized evaluation tool for the community but also highlights a critical future direction: tightly integrating explicit multi-body dynamics priors with the generalized reasoning capabilities of large models. We hope this benchmark will inspire further exploration into general-purpose aerial embodied intelligence, propelling robotics from the 2D ground into the vast 3D space.

## Impact Statement

This paper presents work whose goal is to advance the field of aerial robotics and embodied AI. The primary societal implication of this research is the potential development of more autonomous and versatile unmanned aerial vehicles (UAVs), which could benefit applications such as search and rescue, logistics, and infrastructure inspection. However, integrating VLMs into physical control systems introduces challenges related to safety, reliability, and the potential for unintended physical actions due to model hallucinations or grounding failures. By systematically evaluating these failure modes, our work aims to promote the safe and robust deployment of such systems. We do not feel there are other specific negative societal consequences that must be highlighted here.

## References

Author, N. N. Suppressed for anonymity, 2021.

Bai, S., Cai, Y., Chen, R., Chen, K., Chen, X., Cheng, Z., Deng, L., Ding, W., Gao, C., Ge, C., Ge, W., Guo, Z., Huang, Q., Huang, J., Huang, F., Hui, B., Jiang, S., Li, Z., Li, M., Li, M., Li, K., Lin, Z., Lin, J., Liu, X., Liu, J., Liu, C., Liu, Y., Liu, D., Liu, S., Lu, D., Luo, R., Lv, C., Men, R., Meng, L., Ren, X., Ren, X., Song, S., Sun, Y., Tang, J., Tu, J., Wan, J., Wang, P., Wang, P., Wang, Q., Wang, Y., Xie, T., Xu, Y., Xu, H., Xu, J., Yang, Z., Yang, M., Yang, J., Yang, A., Yu, B., Zhang, F., Zhang, H., Zhang, X., Zheng, B., Zhong, H., Zhou, J., Zhou, F., Zhou, J., Zhu, Y., and Zhu, K. Qwen3-vl technical report, 2025a.

Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., Zhong, H., Zhu, Y., Yang, M., Li, Z., Wan, J., Wang, P., Ding, W., Fu, Z., Xu, Y., Ye, J., Zhang, X., Xie, T., Cheng, Z., Zhang, H., Yang, Z., Xu, H., and Lin, J. Qwen2.5-vl technical report, 2025b.

Black, K., Brown, N., Darpinian, J., Dhabalia, K., Driess, D., Esmail, A., Equi, M. R., Finn, C., Fusai, N., Galliker, M. Y., Ghosh, D., Groom, L., Hausman, K., brian ichter, Jakubczak, S., Jones, T., Ke, L., LeBlanc, D., Levine, S., Li-Bell, A., Mothukuri, M., Nair, S., Pertsch, K., Ren, A. Z., Shi, L. X., Smith, L., Springenberg, J. T., Stachowicz, K., Tanner, J., Vuong, Q., Walke, H., Walling, A., Wang, H., Yu, L., and Zhilinsky, U. \$\pi {0.5}\$: a visionlanguage-action model with open-world generalization. In 9th Annual Conference on Robot Learning, 2025.

Black, K., Brown, N., Driess, D., Esmail, A., Equi, M., Finn, C., Fusai, N., Groom, L., Hausman, K., Ichter, B., Jakubczak, S., Jones, T., Ke, L., Levine, S., Li-Bell, A., Mothukuri, M., Nair, S., Pertsch, K., Shi, L. X., Tanner, J.,

Vuong, Q., Walling, A., Wang, H., and Zhilinsky, U. π<sub>0</sub>: A vision-language-action flow model for general robot control, 2026.

Brohan, A., Brown, N., Carbajal, J., Chebotar, Y., Dabis, J., Finn, C., Gopalakrishnan, K., Hausman, K., Herzog, A., Hsu, J., Ibarz, J., Ichter, B., Irpan, A., Jackson, T., Jesmonth, S., Joshi, N. J., Julian, R., Kalashnikov, D., Kuang, Y., Leal, I., Lee, K.-H., Levine, S., Lu, Y., Malla, U., Manjunath, D., Mordatch, I., Nachum, O., Parada, C., Peralta, J., Perez, E., Pertsch, K., Quiambao, J., Rao, K., Ryoo, M., Salazar, G., Sanketi, P., Sayed, K., Singh, J., Sontakke, S., Stone, A., Tan, C., Tran, H., Vanhoucke, V., Vega, S., Vuong, Q., Xia, F., Xiao, T., Xu, P., Xu, S., Yu, T., and Zitkovich, B. Rt-1: Robotics transformer for real-world control at scale, 2023.

Cao, H., Shen, J., Liu, C., Zhu, B., and Zhao, S. Motion planning for aerial pick-and-place with geometric feasibility constraints. IEEE Transactions on Automation Science and Engineering, 22:2577–2594, 2025a. doi: 10.1109/TASE.2024.3382296.

Cao, H., Shen, J., Zhang, Y., Fu, Z., Liu, C., Sun, S., and Zhao, S. Proximal cooperative aerial manipulation with vertically stacked drones. Nature, pp. 1–8, 2025b.

Carvajal, C. P., Andaluz, G. M., Andaluz, V. H., Roberti, F., Palacios-Navarro, G., and Carelli, R. Multitask control of aerial manipulator robots with dynamic compensation based on numerical methods. Robotics and Autonomous Systems, 173:104614, 2024. ISSN 0921-8890. doi: https: //doi.org/10.1016/j.robot.2023.104614.

Chen, T., Chen, Z., Chen, B., Cai, Z., Liu, Y., Li, Z., Liang, Q., Lin, X., Ge, Y., Gu, Z., Deng, W., Guo, Y., Nian, T., Xie, X., Chen, Q., Su, K., Xu, T., Liu, G., Hu, M., ang Gao, H., Wang, K., Liang, Z., Qin, Y., Yang, X., Luo, P., and Mu, Y. Robotwin 2.0: A scalable data generator and benchmark with strong domain randomization for robust bimanual robotic manipulation, 2025.

Chi, C., Xu, Z., Feng, S., Cousineau, E., Du, Y., Burchfiel, B., Tedrake, R., and Song, S. Diffusion policy: Visuomotor policy learning via action diffusion, 2024.

Deitke, M., Clark, C., Lee, S., Tripathi, R., Yang, Y., Park, J. S., Salehi, M., Muennighoff, N., Lo, K., Soldaini, L., Lu, J., Anderson, T., Bransom, E., Ehsani, K., Ngo, H., Chen, Y., Patel, A., Yatskar, M., Callison-Burch, C., Head, A., Hendrix, R., Bastani, F., VanderBilt, E., Lambert, N., Chou, Y., Chheda, A., Sparks, J., Skjonsberg, S., Schmitz, M., Sarnat, A., Bischoff, B., Walsh, P., Newell, C., Wolters, P., Gupta, T., Zeng, K.-H., Borchardt, J., Groeneveld, D., Nam, C., Lebrecht, S., Wittlif, C., Schoenick, C., Michel, O., Krishna, R., Weihs, L., Smith, N. A., Hajishirzi, H., Girshick, R., Farhadi, A.,

and Kembhavi, A. Molmo and pixmo: Open weights and open data for state-of-the-art vision-language models, 2024.

Deng, W., Chen, H., Ye, B., Chen, H., Li, Z., and Lyu, X. Whole-body integrated motion planning for aerial manipulators. IEEE Transactions on Robotics, 41:6661– 6679, 2025. doi: 10.1109/TRO.2025.3626619.

Deshmukh, S., Alonso-Mora, J., and Sun, S. Global endeffector pose control of an underactuated aerial manipulator via reinforcement learning, 2025.

Duda, R. O., Hart, P. E., and Stork, D. G. Pattern Classification. John Wiley and Sons, 2nd edition, 2000.

Fu, Z., Zhao, T. Z., and Finn, C. Mobile ALOHA: Learning bimanual mobile manipulation using low-cost wholebody teleoperation. In 8th Annual Conference on Robot Learning, 2024.

Ghorbani, S., Samadikhoshkho, Z., and Janabi-Sharifi, F. Dual-arm aerial continuum manipulation systems: modeling, pre-grasp planning, and control. Nonlinear Dynamics, 111(8):7339–7355, Apr 2023. ISSN 1573-269X. doi: 10.1007/s11071-022-08212-w.

GLM, T., :, Zeng, A., Xu, B., Wang, B., Zhang, C., Yin, D., Zhang, D., Rojas, D., Feng, G., Zhao, H., Lai, H., Yu, H., Wang, H., Sun, J., Zhang, J., Cheng, J., Gui, J., Tang, J., Zhang, J., Sun, J., Li, J., Zhao, L., Wu, L., Zhong, L., Liu, M., Huang, M., Zhang, P., Zheng, Q., Lu, R., Duan, S., Zhang, S., Cao, S., Yang, S., Tam, W. L., Zhao, W., Liu, X., Xia, X., Zhang, X., Gu, X., Lv, X., Liu, X., Liu, X., Yang, X., Song, X., Zhang, X., An, Y., Xu, Y., Niu, Y., Yang, Y., Li, Y., Bai, Y., Dong, Y., Qi, Z., Wang, Z., Yang, Z., Du, Z., Hou, Z., and Wang, Z. Chatglm: A family of large language models from glm-130b to glm-4 all tools, 2024.

Guo, X., He, G., Xu, J., Mousaei, M., Geng, J., Scherer, S., and Shi, G. Flying calligrapher: Contact-aware motion and force planning and control for aerial manipulation. IEEE Robotics and Automation Letters, 9(12): 11194–11201, 2024. doi: 10.1109/LRA.2024.3486236.

He, G., Guo, X., Tang, L., Zhang, Y., Mousaei, M., Xu, J., Geng, J., Scherer, S., and Shi, G. Flying hand: Endeffector-centric framework for versatile aerial manipulation teleoperation and policy learning, 2025.

Intelligence, P., Amin, A., Aniceto, R., Balakrishna, A., Black, K., Conley, K., Connors, G., Darpinian, J., Dhabalia, K., DiCarlo, J., Driess, D., Equi, M., Esmail, A., Fang, Y., Finn, C., Glossop, C., Godden, T., Goryachev, I., Groom, L., Hancock, H., Hausman, K., Hussein, G., Ichter, B., Jakubczak, S., Jen, R., Jones, T., Katz, B., Ke,

L., Kuchi, C., Lamb, M., LeBlanc, D., Levine, S., Li-Bell, A., Lu, Y., Mano, V., Mothukuri, M., Nair, S., Pertsch, K., Ren, A. Z., Sharma, C., Shi, L. X., Smith, L., Springenberg, J. T., Stachowicz, K., Stoeckle, W., Swerdlow, A., Tanner, J., Torne, M., Vuong, Q., Walling, A., Wang, H., Williams, B., Yoo, S., Yu, L., Zhilinsky, U., and Zhou, Z. $\pi _ { 0 . 6 } ^ { * } \colon$ a vla that learns from experience, 2025.

Kearns, M. J. Computational Complexity of Machine Learning. PhD thesis, Department of Computer Science, Harvard University, 1989.

Kim, M. J., Pertsch, K., Karamcheti, S., Xiao, T., Balakrishna, A., Nair, S., Rafailov, R., Foster, E., Lam, G., Sanketi, P., Vuong, Q., Kollar, T., Burchfiel, B., Tedrake, R., Sadigh, D., Levine, S., Liang, P., and Finn, C. Openvla: An open-source vision-language-action model, 2024.

Langley, P. Crafting papers on machine learning. In Langley, P. (ed.), Proceedings ofthe 17th International Conference on Machine Learning (ICML 2000), pp. 1207–1216, Stanford, CA, 2000. Morgan Kaufmann.

Li, B., Zhang, Y., Guo, D., Zhang, R., Li, F., Zhang, H., Zhang, K., Zhang, P., Li, Y., Liu, Z., and Li, C. Llavaonevision: Easy visual task transfer, 2024.

Li, J., Chen, P., Li, M., and Ren, L. Ib-amg: Aircraft mission generation with inference-based vision-language-action model. In 2025 40th Youth Academic Annual Conference ofChinese Association ofAutomation (YAC), pp. 2323– 2328, 2025. doi: 10.1109/YAC66630.2025.11149880.

Liang, J., Chen, Y., Wu, Y., Miao, Z., Zhang, H., and Wang, Y. Adaptive prescribed performance control of unmanned aerial manipulator with disturbances. IEEE Transactions on Automation Science and Engineering, 20 (3):1804–1814, 2023. doi: 10.1109/TASE.2022.3186315.

Liu, B., Zhu, Y., Gao, C., Feng, Y., Liu, Q., Zhu, Y., and Stone, P. Libero: Benchmarking knowledge transfer for lifelong robot learning. In Oh, A., Naumann, T., Globerson, A., Saenko, K., Hardt, M., and Levine, S. (eds.), Advances in Neural Information Processing Systems, volume 36, pp. 44776–44791. Curran Associates, Inc., 2023.

Liu, Q., Liu, Y., Chen, Z., Guo, K., Yu, X., Zhang, Y., and Guo, L. A compact aerial manipulator: Design and control for dexterous operations. Journal ofIntelligent & Robotic Systems, 110(2):66, Apr 2024. ISSN 1573-0409. doi: 10.1007/s10846-024-02090-7.

Liu, X., Liu, Y., Qiu, H., Qirong, Y., and Lian, Z. Indooruav: Benchmarking vision-language uav navigation in continuous indoor environments, 2025.

Lykov, A., Serpiva, V., Khan, M. H., Sautenkov, O., Myshlyaev, A., Tadevosyan, G., Yaqoot, Y., and Tsetserukou, D. Cognitivedrone: A vla model and evaluation benchmark for real-time cognitive task solving and reasoning in uavs, 2025.

Mehboob, F., James, M., Habel, A., Sam, J., Cabrera, M. A., and Tsetserukou, D. Dronevla: Vla based aerial manipulation, 2026.

Michalski, R. S., Carbonell, J. G., and Mitchell, T. M. (eds.). Machine Learning: An Artificial Intelligence Approach, Vol. I. Tioga, Palo Alto, CA, 1983.

Mitchell, T. M. The need for biases in learning generalizations. Technical report, Computer Science Department, Rutgers University, New Brunswick, MA, 1980.

Newell, A. and Rosenbloom, P. S. Mechanisms of skill acquisition and the law of practice. In Anderson, J. R. (ed.), Cognitive Skills and Their Acquisition, chapter 1, pp. 1–51. Lawrence Erlbaum Associates, Inc., Hillsdale, NJ, 1981.

Ollero, A., Tognon, M., Suarez, A., Lee, D., and Franchi, A. Past, present, and future of aerial robotic manipulators. IEEE Transactions on Robotics, 38(1):626–645, 2022. doi: 10.1109/TRO.2021.3084395.

Peng, R., Wang, Y., Lu, M., and Lu, P. A dexterous and compliant aerial continuum manipulator for cluttered and constrained environments. Nature Communications, 16(1):889, Jan 2025. ISSN 2041-1723. doi: 10.1038/s41467-024-55157-2.

Pertsch, K., Stachowicz, K., Ichter, B., Driess, D., Nair, S., Vuong, Q., Mees, O., Finn, C., and Levine, S. Fast: Efficient action tokenization for vision-language-action models, 2025.

Samuel, A. L. Some studies in machine learning using the game of checkers. IBM Journal of Research and Development, 3(3):211–229, 1959.

Sautenkov, O., Yaqoot, Y., Lykov, A., Mustafa, M. A., Tadevosyan, G., Akhmetkazy, A., Cabrera, M. A., Martynov, M., Karaf, S., and Tsetserukou, D. Uav-vla: Visionlanguage-action system for large scale aerial mission generation. In 2025 20th ACM/IEEE International Conference on Human-Robot Interaction (HRI), pp. 1588–1592, 2025a. doi: 10.1109/HRI61500.2025.10974117.

Sautenkov, O., Yaqoot, Y., Lykov, A., Mustafa, M. A., Tadevosyan, G., Akhmetkazy, A., Cabrera, M. A., Martynov, M., Karaf, S., and Tsetserukou, D. Uav-vla: Visionlanguage-action system for large scale aerial mission generation, 2025b.

Serpiva, V., Lykov, A., Myshlyaev, A., Khan, M. H., Abdulkarim, A. A., Sautenkov, O., and Tsetserukou, D. Racevla: Vla-based racing drone navigation with humanlike behaviour, 2025.

Shukor, M., Aubakirova, D., Capuano, F., Kooijmans, P., Palma, S., Zouitine, A., Aractingi, M., Pascal, C., Russi, M., Marafioti, A., Alibert, S., Cord, M., Wolf, T., and Cadene, R. Smolvla: A vision-language-action model for affordable and efficient robotics, 2025.

Sumathy, V., Abdul, S. A. M., and Ghose, D. Projection operator-based robust adaptive control of an aerial robot with a manipulator. Journal of Field Robotics, 40(5): 1069–1096, 2023.

Wang, M., Lyu, S., Liu, Q., Yang, Z., Guo, K., and Yu, X. Precise end-effector control for an aerial manipulator under composite disturbances: Theory and experiments. IEEE Transactions on Automation Science and Engineering, 22:4006–4021, 2025a. doi: 10.1109/TASE.2024. 3406754.

Wang, W., Gao, Z., Gu, L., Pu, H., Cui, L., Wei, X., Liu, Z., Jing, L., Ye, S., Shao, J., Wang, Z., Chen, Z., Zhang, H., Yang, G., Wang, H., Wei, Q., Yin, J., Li, W., Cui, E., Chen, G., Ding, Z., Tian, C., Wu, Z., Xie, J., Li, Z., Yang, B., Duan, Y., Wang, X., Hou, Z., Hao, H., Zhang, T., Li, S., Zhao, X., Duan, H., Deng, N., Fu, B., He, Y., Wang, Y., He, C., Shi, B., He, J., Xiong, Y., Lv, H., Wu, L., Shao, W., Zhang, K., Deng, H., Qi, B., Ge, J., Guo, Q., Zhang, W., Zhang, S., Cao, M., Lin, J., Tang, K., Gao, J., Huang, H., Gu, Y., Lyu, C., Tang, H., Wang, R., Lv, H., Ouyang, W., Wang, L., Dou, M., Zhu, X., Lu, T., Lin, D., Dai, J., Su, W., Zhou, B., Chen, K., Qiao, Y., Wang, W., and Luo, G. Internvl3.5: Advancing open-source multimodal models in versatility, reasoning, and efficiency, 2025b.

Wang, X., Yang, D., Liao, Y., Zheng, W., wenjun wu, Dai, B., Li, H., and Liu, S. Uav-flow colosseo: A real-world benchmark for flying-on-a-word uav imitation learning, 2025c.

Wu, Y., Zhu, M., Li, X., Du, Y., Fan, Y., Li, W., Han, Z., Zhou, X., and Gao, F. Vla-an: An efficient and onboard vision-language-action framework for aerial navigation in complex environments, 2025.

Xu, M., Hu, A., and Wang, H. Image-based visual impedance force control for contact aerial manipulation. IEEE Transactions on Automation Science and Engineering, 20(1):518–527, 2023. doi: 10.1109/TASE.2022. 3162207.

Yenamandra, S., Ramachandran, A., Yadav, K., Wang, A., Khanna, M., Gervet, T., Yang, T.-Y., Jain, V., Clegg, A. W., Turner, J., et al. Homerobot: Open-vocabulary

mobile manipulation. arXiv preprint arXiv:2306.11565, 2023.

Zhang, Z., Yu, H., Chai, Y., Yang, Z., Liang, X., Fang, Y., and Han, J. An end-effector-oriented coupled motion planning method for aerial manipulators in constrained environments. IEEE/ASME Transactions on Mechatronics, 30(6):6027–6037, 2025. doi: 10.1109/TMECH.2025. 3550562.

Zhao, T. Z., Kumar, V., Levine, S., and Finn, C. Learning fine-grained bimanual manipulation with low-cost hardware, 2023.

## A. Benchmark Implementation Details

## A.1. Task Description and Taxonomy

Task Overview. AIR-VLA aims to establish a comprehensive evaluation platform covering the full skill spectrum of aerial mobile manipulation. We have constructed a diverse family of tasks comprising 60 atomic and composite tasks, logically categorized into four core suites: Base Manipulation, Object & Spatial, Semantic Understanding, and Long-Horizon. The complete task list is detailed in Table 5. These tasks range from omnidirectional pick-and-place in 3D space and precision insertion in confined spaces, to articulated object interaction involving drawers and cabinets, and twist-and-rotate operations requiring torque control. Furthermore, the benchmark integrates hybrid navigation-manipulation tasks, requiring agents to seamlessly switch between extensive scene movement and local fine-grained manipulation. Beyond basic motor skills, this benchmark emphasizes deep evaluation of advanced embodied intelligence across multiple dimensions, including voxel-level spatial perception, dynamic pose compensation of the floating base, cross-modal semantic generalization, and long-horizon temporal logical planning.

![](<images/b0e22f44fd832105a68c6deffe6cf7675e665d5e5dd9f403c4798837f98ddfea.jpg>)  
Figure 4. Task examples in AIR-VLA dataset.

Coupled Dynamics and Long-Horizon Reasoning. Unlike traditional benchmarks that often treat navigation and manipulation as decoupled phases or focus solely on 2D planar movement, the core feature of AIR-VLA is the emphasis on spatio-temporal coupling between the UAV base and the manipulator. In our context, ”reasoning” is endowed with a dual meaning: first, kinematic feasibility reasoning, where the agent must real-time assess whether the target is within the manipulator’s reachable workspace under the floating state of the UAV with drift and jitter; second, long-horizon temporal planning, where early interaction actions (e.g., opening a drawer and maintaining a specific opening) must create necessary preconditions for subsequent geometric constraints (e.g., placing an object inside). Our dataset statistics indicate that the average episode length is approximately 475 time steps, with the Long-Horizon suite exceeding 600 time steps (at 20Hz control frequency), significantly surpassing traditional tabletop manipulation datasets. This long-horizon characteristic requires policies not only to maintain logical consistency over extended periods but also to continuously output control signals to stabilize the base and resist environmental disturbances while executing fine-grained sub-tasks.

## A.2. Observation Space

To support perception and decision-making in complex environments, AIR-VLA constructs a multimodal observation space rich in semantic information. In terms of visual perception, the system synchronously captures RGB-D image streams from four perspectives: a front-down chassis camera provides a first-person flight view for global path planning and targe approach; an eye-in-hand camera mounted on the manipulator end-effector resolves occlusion issues and supports fine manipulation; while global overhead and side-view cameras provide third-person god views to assist state estimation and training supervision. Regarding proprioception, the system records full state vectors at high frequency, including the UAV’s position, orientation quaternion, linear velocity, and angular velocity in the world frame, the manipulator’s 7 joint poses and velocities, as well as the end-effector’s 6-DoF pose and gripper state. All sensory data is strictly timestamp-synchronized, providing a precise data alignment foundation for multimodal learning.

![](<images/af5043d58cbd7472ec9ef13a3214a5fcb471af301f9fdaa60d0de2d57cb6ca27.jpg>)  
Figure 5. Observation Space. This figure illustrates instances of observations from various camera perspectives.

## A.3. Evaluation Metrics

## A.3.1. VLA EVALUATION METRICS

Traditional binary metrics fail to capture continuous state changes in aerial manipulation. To evaluate UAV movement, coordination, and safety at a fine granularity, we propose a four-dimensional system. The total score $S _ { t o t a l }$ is a weighted sum of metrics $S _ { i } \in [ 0 , 1 0 0 ]$ ], where the maximum possible score is 100:

$$
S _ {t o t a l} = w _ {p o s} S _ {p o s} + w _ {a r m} S _ {a r m} + w _ {s a f e} S _ {s a f e} + w _ {t a s k} S _ {t a s k}\tag{1}
$$

Base Positioning Accuracy $( S _ { p o s } )$ This metric assesses if the UAV reaches a Feasible Workspace (collision-free Inverse Kinematics (IK) solution). We model positioning quality relative to optimal distance $\mu$ using a Gaussian decay function, scored hierarchically by interaction state on a 0-100 scale:

$$
s _ {p o s} ^ {(i)} = \left\{ \begin{array}{l l} 1 0 0, & \text { if   operation   succeeds } \\ 1 0 0 \cdot \mathcal {G} (d _ {i}; \lambda_ {a c t}), & \text { if   interaction   attempted } \\ 1 0 0 \cdot \mathcal {G} (d _ {i}; \lambda_ {h o v}), & \text { if   hovering   only } \end{array} \right.\tag{2}
$$

where $\begin{array} { r } { \mathcal { G } ( d ; \lambda ) = \lambda \cdot \exp { \left( - \frac { ( d - \mu ) ^ { 2 } } { 2 \sigma ^ { 2 } } \right) } \cdot \mathbb { I } ( d _ { m i n } \leq d \leq d _ { m a x } ) . \lambda _ { a c t } > \lambda _ { h o v } } \end{array}$ rewards active attempts.

Manipulator Efficacy $\left( S _ { a r m } \right)$ To quantify resilience against floating-base jitter, we combine orientation consistency $S _ { o r i } \in \{ 0 , 1 \}$ (Euler angle alignment) and interaction precision $s _ { i n t } ^ { ( i ) }$ to form a score out of 100:

$$
S _ {a r m} = 1 0 0 \cdot \left(\beta S _ {o r i} + \frac {1 - \beta}{N} \sum_ {i = 1} ^ {N} s _ {i n t} ^ {(i)}\right)\tag{3}
$$

where interaction precision uses a Gaussian kernel on end-effector distance $d _ { e e } .$ :

$$
s _ {i n t} ^ {(i)} = \exp \left(- \frac {d _ {e e} ^ {2}}{2 \sigma_ {a r m} ^ {2}}\right) \cdot \mathbb {I} (d _ {e e} \leq d _ {t h r e s h})\tag{4}
$$

Environmental Disturbance Safety $( S _ { s a f e } )$ Collision risk is measured via the total displacement $\Delta _ { t o t a l }$ of non-target objects. The score decays from 100 (static environment) to 0 (exceeding limit $\Delta _ { l i m i t } )$ :

$$
S _ {s a f e} = 1 0 0 \cdot \exp \left(- \frac {\Delta_ {t o t a l} ^ {2}}{2 \sigma_ {s a f e} ^ {2}}\right) \cdot \mathbb {I} (\Delta_ {t o t a l} <   \Delta_ {l i m i t})\tag{5}
$$

Task Progression $( S _ { t a s k } )$ For Long-Horizon Tasks with K sub-goals, we measure the percentage of completion to reveal planning decay:

$$
S _ {t a s k} = \frac {1 0 0}{K} \sum_ {k = 1} ^ {K} \mathbb {I} (\text { Sub - goal } _ {k} \text {   completed })\tag{6}
$$

## A.3.2. VLM TASK PLANNING EVALUATION METRICS

We evaluate high-level planning across process logic, spatial understanding, object grounding, and skill selection using scores scaled to [0, 100]. The total weighted score is:

$$
S _ {t o t a l} = w _ {p l a n} S _ {p l a n} + w _ {l o c} S _ {l o c} + w _ {o b j} S _ {o b j} + w _ {s k i l l} S _ {s k i l l}\tag{7}
$$

Process Planning Capability $( S _ { p l a n } )$ Evaluates action sequence correctness via Type Accuracy (agent match M) and Length Consistency (step count K vs. ground truth N), scaled to 100:

$$
S _ {p l a n} = 1 0 0 \cdot \left[ \alpha \cdot \frac {M}{N} + (1 - \alpha) \cdot \max \left(0, 1 - \frac {| K - N |}{N}\right) \right]\tag{8}
$$

Spatial Navigation Understanding $( S _ { l o c } )$ Measures UAV movement logic. For $N _ { m o v e }$ steps, we score the deviation $\Delta \theta _ { i }$ between predicted and optimal directions using a discrete point system:

$$
\omega (\Delta \theta_ {i}) = \left\{ \begin{array}{l l} 1 0 0, & \text { if } \Delta \theta_ {i} = 0 ^ {\circ} (\text { Correct }) \\ 4 0, & \text { if } \Delta \theta_ {i} = 4 5 ^ {\circ} (\text { Neighboring }) \\ 0, & \text { otherwise } \end{array} \right.\tag{9}
$$

$$
S _ {l o c} = \frac {1}{N _ {m o v e}} \sum_ {i = 1} ^ {N _ {m o v e}} \omega (\Delta \theta_ {i})
$$

(10)

Object Grounding Capability $( S _ { o b j } )$ Penalizes hallucinations during manipulation. With $M _ { o b j }$ correct and $K _ { o b j }$ incorrect predictions out of $N _ { o b j }$ total objects, the score is calculated out of 100:

$$
S _ {o b j} = 1 0 0 \cdot \max \left(0, \frac {M _ {o b j} - K _ {o b j}}{N _ {o b j}}\right)\tag{11}
$$

Skill Selection Capability $( S _ { s k i l l } )$ The coverage percentage of correctly selected atomic skills $M _ { s k i l l }$ out of $N _ { s k i l l }$ total skills:

$$
S _ {s k i l l} = 1 0 0 \cdot \frac {M _ {s k i l l}}{N _ {s k i l l}}\tag{12}
$$

## B. Dataset Construction

## B.1. Scene Diversity

To minimize the Sim-to-Real gap and provide diverse visual inputs, we built a simulation environment based on NVIDIA Isaac Sim. We designed six variant environments covering three core space types: household, industrial, and outdoor settings.

## B.2. Assets Library

Our asset library leverages the Universal Scene Description (USD) ecosystem of Isaac Sim and deeply integrates high-quality assets from ShapeNet and Google Scanned Objects. The library contains objects of various categories, covering a wide range of interactive objects from rigid bodies to articulated mechanisms. To ensure the realism of contact dynamics, all objects undergo rigorous physical property calibration, including convex decomposition optimization of collision meshes, inertia tensor calculation for mass distribution, and fine-tuning of surface friction coefficients.

Table 4. Statistics of interactive assets in the simulation environment.

<table><tr><td>ASSET CATEGORY</td><td>QUANTITY</td></tr><tr><td>STOVE</td><td>1</td></tr><tr><td>MUG</td><td>4</td></tr><tr><td>CABINET</td><td>1</td></tr><tr><td>TOY</td><td>2</td></tr><tr><td>SNACK</td><td>3</td></tr><tr><td>INGREDIENT</td><td>1</td></tr><tr><td>CONDIMENT</td><td>2</td></tr><tr><td>BOWL</td><td>3</td></tr><tr><td>BOX</td><td>5</td></tr><tr><td>BOX CONTAINER</td><td>4</td></tr><tr><td>TRAY</td><td>3</td></tr><tr><td>HOUSEHOLD</td><td>2</td></tr><tr><td>INDUSTRIAL PART</td><td>7</td></tr><tr><td>TABLE</td><td>3</td></tr></table>

## B.3. Data Collection Pipeline

To capture natural and diverse action distributions and avoid the behavioral rigidity of rule-based scripts, we employed a ”Human-in-the-Loop” simulation teleoperation data collection scheme. Operators synergistically control the UAV’s flight and the manipulator’s fine movements via gamepad or keyboard. Successful task segments are retained and processed into formats including raw Isaac Sim data, Robomimic-compatible HDF5 format, and LeRobot standardized format adapted for the Hugging Face open-source ecosystem, satisfying the requirements of different training frameworks.

## B.4. VLM Evaluation Setup

To quantitatively evaluate the task planning capabilities of VLMs, we defined a standardized task description format based on JSON. This format structurally articulates the ideal execution flow of long-horizon tasks, detailing sub-task decomposition logic, execution agents (UAV or ARM), target objects, and required skills. This structured data serves as Ground Truth for automatically evaluating the correctness of VLM output planning logic.

```json
{
    "subtask": 0,
    "tasktype": "UAV",
    "direction": "frontleft"
},
{
    "subtask": 1,
    "tasktype": "ARM",
    "object": "rubiks_cube",
    "skill": "pick"
}
```

## B.5. Instruction Generation and Augmentation

To comprehensively evaluate the generalization capabilities of models in natural language understanding, we developed an instruction augmentation engine leveraging LLMs. For each specific task, the LLM is prompted to generate novel instructions that retain complete action semantics while employing diverse linguistic variations, including object rephrasing, implicit action inference, and environmental descriptions. This approach ensures the inclusion of unseen yet semantically equivalent instructions within the training set, enabling effective assessment of the model’s semantic robustness against unstructured linguistic inputs.The specific prompt utilized for instruction generation is presented below:

```txt
Listing 1. Prompt for instruction generation.

The current action description is a bit one-dimensional. I need to make some modifications: 1. Replace the description of the object, 2. Replace the description of the action, 3. Add additional semantic scene information, 4. It can be expressed in a more conversational way.
```

## C. Experimental Implementation

To establish robust baselines and evaluate the transferability of VLA models to the aerial manipulation domain, we selected five representative architectures for full-parameter fine-tuning: π<sub>0</sub>, π<sub>0.5</sub>, π<sub>0</sub>-FAST, ACT (Action Chunking Transformer), and Diffusion Policy.

Training Configuration. All models accept RGB images from four perspectives (resized to 224 × 224) alongside 7-DoF manipulator joint positions and gripper states as inputs. The models predict a hybrid action vector comprising 4-DoF relative UAV displacement commands, 7-DoF manipulator joint position targets, and the gripper state. We utilized default hyperparameters for all models. Using training volume as the standardization metric, all models were trained for 10,000 epochs with identical batch sizes. All experiments were executed in parallel on two NVIDIA A100 GPUs (80GB VRAM). We utilized the official open-source implementations for each model while maintaining default architectural hyperparameters. This setup was chosen to verify the potential of general-purpose VLA architectures to adapt to high-dimensional aerial manipulation tasks without domain-specific modifications.

Evaluation Protocol. During the evaluation phase, we maintained consistency with the training data regarding the formatting of images, language instructions, observation states, and action states, adopting a control frequency of 20Hz. For the inference strategy, specifically for π<sub>0</sub>, π<sub>0.5</sub> and π<sub>0</sub>-FAST, we executed the first 5 actions of the predicted action chunk at each inference step. For Diffusion Policy and ACT, we directly executed the predicted actions. For each specific task, we conducted 10 independent evaluation trials and reported the average metrics across these trials as the final result.

## D. Full Task List

Table 5. Detailed Task List of AIR-VLA (Summary Version)

<table><tr><td>Task Name</td><td>Suite</td><td>Description</td></tr><tr><td>Standard Pick &amp; Place</td><td>Base Manip.</td><td>Basic task: Pick gray cube, place in yellow basket.</td></tr><tr><td>Place on Flat Surface</td><td>Base Manip.</td><td>Precision placement: Pick cube, place stably on plate.</td></tr><tr><td>Color Distractor (Green)</td><td>Base Manip.</td><td>Color discrimination: Pick green cube amidst gray distractors, place in basket.</td></tr><tr><td>Container Geometry (Bowl)</td><td>Base Manip.</td><td>Container adaptation: Place green cube into small-aperture bowl.</td></tr><tr><td>Deformable Object (Banana)</td><td>Base Manip.</td><td>Irregular object: Pick banana and place in basket, handling non-rigid grasping.</td></tr><tr><td>Irregular on Plate</td><td>Base Manip.</td><td>Placement stability: Place banana stably on plate.</td></tr><tr><td>Tool Manipulation (Mug)</td><td>Base Manip.</td><td>Handled object: Pick red mug and place on plate.</td></tr><tr><td>Precise Insertion (Mug)</td><td>Base Manip.</td><td>Fine manipulation: Insert mug accurately into bowl, avoiding collision.</td></tr><tr><td>Cylinder Manipulation</td><td>Base Manip.</td><td>Pose constraint: Pick meat can and place upright on plate.</td></tr><tr><td>Tall Object (Cleanser)</td><td>Base Manip.</td><td>Center of mass control: Pick tall cleanser bottle and place in deep basket.</td></tr><tr><td>Variant: Banana in Box</td><td>Base Manip.</td><td>Variant: Place cube in purple storage box.</td></tr><tr><td>Variant: Teddy Bear</td><td>Base Manip.</td><td>Complex model: Pick teddy bear and place in purple storage box.</td></tr><tr><td>Variant: Meat Can in Box</td><td>Base Manip.</td><td>Variant: Place meat can in purple storage box.</td></tr><tr><td>Variant: Chocolate Box</td><td>Base Manip.</td><td>Box-shaped object: Pick chocolate box and place in purple storage box.</td></tr><tr><td>Variant: Wood Block</td><td>Base Manip.</td><td>Textured object: Pick wood block and place in purple storage box.</td></tr><tr><td>Variant: Tomato Can</td><td>Base Manip.</td><td>Variant: Place chocolate box in red bowl.</td></tr><tr><td>Variant: Teddy in Bowl</td><td>Base Manip.</td><td>Size adaptation: Place large teddy bear in bowl.</td></tr><tr><td>Variant: Banana in Bowl</td><td>Base Manip.</td><td>Variant: Adapt banana curvature to bowl shape.</td></tr><tr><td>Stacking: Can on Wood</td><td>Base Manip.</td><td>Simple stacking: Place meat can on top of wood block.</td></tr><tr><td>Stacking: Box on Wood</td><td>Base Manip.</td><td>Simple stacking: Place chocolate box on top of wood block.</td></tr><tr><td>Target Color (Yellow)</td><td>Object &amp; Spatial</td><td>Multi-container grounding: Identify yellow basket among multiple and place cube.</td></tr><tr><td>Target Color (Gray)</td><td>Object &amp; Spatial</td><td>Low-contrast grounding: Identify gray basket similar to ground color and place.</td></tr><tr><td>Target Category (Banana)</td><td>Object &amp; Spatial</td><td>Category &amp; location: Identify banana and place in specified gray basket.</td></tr><tr><td>Pick Left Cup (Red)</td><td>Object &amp; Spatial</td><td>Spatial &amp; visual matching: Pick the left red cup and place on red plate.</td></tr><tr><td>Visual Combination (Blue-Red)</td><td>Object &amp; Spatial</td><td>Color combination: Place blue cup on red plate, testing instruction following.</td></tr><tr><td>Cross-Category Transfer</td><td>Object &amp; Spatial</td><td>Cross-category move: Place meat can into gray basket.</td></tr><tr><td>Slim Object in Bowl</td><td>Object &amp; Spatial</td><td>Slim object grounding: Place cleanser into yellow bowl.</td></tr><tr><td>Spatial Ref. (Middle Drawer)</td><td>Object &amp; Spatial</td><td>Vertical spatial grounding: Identify middle handle, pull drawer &gt; 0.3m.</td></tr><tr><td>Small Object (Bolt)</td><td>Object &amp; Spatial</td><td>Tiny object grounding: Identify tiny bolt and place in bowl.</td></tr><tr><td>Spatial Ref. (Top Drawer)</td><td>Object &amp; Spatial</td><td>Dynamic height adaptation: Ascend to identify top handle, pull drawer &gt; 0.3m.</td></tr><tr><td>Pick Left Box (White)</td><td>Object &amp; Spatial</td><td>Spatial constraint: Pick the white box on the left and place in the left basket.</td></tr><tr><td>Variant: Green Nut</td><td>Object &amp; Spatial</td><td>Small part: Pick green nut and place in yellow basket.</td></tr><tr><td>Assembly (Camera on Lidar)</td><td>Object &amp; Spatial</td><td>Fine assembly: Place camera module on top of LiDAR.</td></tr><tr><td>Non-Prehensile (Push Cube)</td><td>Object &amp; Spatial</td><td>Non-prehensile: Push cube next to gray basket.</td></tr><tr><td>Clear Desk (Push Box)</td><td>Object &amp; Spatial</td><td>Clearing action: Push brown box off the table.</td></tr><tr><td>Synonym: Container</td><td>Semantic Und.</td><td>Hypernym: "Container" → Basket. Action: Cube → Yellow Basket</td></tr><tr><td>Synonym: Block &amp; Bin</td><td>Semantic Und.</td><td>Double synonym: "Block" → Cube, "Bin" → Basket.</td></tr><tr><td>Verb: Drop</td><td>Semantic Und.</td><td>Verb nuance: "Drop" implies lower precision release. Action: Banana → Gray Basket</td></tr><tr><td>Synonym: Cup &amp; Dish</td><td>Semantic Und.</td><td>Noun substitution: "Cup" → Mug, "Dish" → Plate.</td></tr><tr><td>Preposition: Onto</td><td>Semantic Und.</td><td>Preposition: "Onto" emphasizes surface support. Action: Blue Mug → Red Plate</td></tr><tr><td>Material: Tin</td><td>Semantic Und.</td><td>Material reference: "Tin" → Metal Can. Action: Meat Can → Gray Basket</td></tr><tr><td>Function: Detergent</td><td>Semantic Und.</td><td>Function reference: "Detergent" → Cleanser. Action: Cleanser → Yellow Bowl</td></tr><tr><td>Synonym: Center</td><td>Semantic Und.</td><td>Position semantic: "Center" → Middle Drawer. Pull action.</td></tr><tr><td>Attribute: Metal</td><td>Semantic Und.</td><td>Attribute reference: "Metal" → Bolt. Place in bowl.</td></tr><tr><td>Superlative: Highest</td><td>Semantic Und.</td><td>Extreme semantic: "Highest" → Top Drawer. Pull action.</td></tr><tr><td>Negation Constraint</td><td>Semantic Und.</td><td>Negation: "Not the brown one...". Exclude brown, pick white box.</td></tr><tr><td>Contextual: Spare Part</td><td>Semantic Und.</td><td>Contextual: "Spare part" refers to green nut, place in parts box.</td></tr><tr><td>Jargon: Vision Sensor</td><td>Semantic Und.</td><td>Jargon: "Vision sensor" → Camera. Place on LiDAR.</td></tr><tr><td>Preposition: By / Don't</td><td>Semantic Und.</td><td>Position &amp; Negation: "By" + "Don't put in". Push action.</td></tr><tr><td>Abstract Intent: Clear</td><td>Semantic Und.</td><td>Abstract intent: "Clear the desk" → Push object off table.</td></tr><tr><td>Multi-Object Sorting</td><td>Long-Horizon</td><td>Multi-step sorting: 1. Rubiks → Yellow; 2. Cube → Gray.</td></tr><tr><td>Batch Collection</td><td>Long-Horizon</td><td>Batch collection: 1. Rubiks → Yellow; 2. Cube → Yellow.</td></tr><tr><td>Color-Matched Sorting</td><td>Long-Horizon</td><td>Color matching: 1. Black Mug → Black Plate; 2. Yellow Mug → Red Plate.</td></tr><tr><td>Cross-Color Sorting</td><td>Long-Horizon</td><td>Spatial crossing: 1. Black Mug → Left(Red); 2. Yellow Mug → Right(Black).</td></tr><tr><td>Category Sorting</td><td>Long-Horizon</td><td>Category sorting: 1. Food(Can) → Right; 2. Fruit(Banana) → Right.</td></tr><tr><td>Stacking &amp; Nesting</td><td>Long-Horizon</td><td>Nesting dependency: 1. Move Bowl → Plate; 2. Place Cube → Inside Bowl.</td></tr><tr><td>Tower Stacking</td><td>Long-Horizon</td><td>Stacking balance: 1. Place Rubiks → Basket; 2. Place Cube → On Rubiks.</td></tr><tr><td>Hybrid Interaction (Stove)</td><td>Long-Horizon</td><td>Hybrid skill: 1. Rotate Switch &gt; 75°; 2. Place Bowl → Stove.</td></tr><tr><td>Drawer Interaction</td><td>Long-Horizon</td><td>Mechanism interaction: 1. Pull Top Drawer; 2. Place Banana → Inside.</td></tr><tr><td>Store &amp; Close</td><td>Long-Horizon</td><td>Full storage: 1. Place Rubiks → Drawer; 2. Close Drawer.</td></tr></table>