# PACE: Phase-Aware Chunk Execution for Robot Policies with Action Chunking

Junnan Nie<sup>1,2</sup>, Jiayi Li<sup>2</sup>, Chenghao Liu<sup>1</sup>, Junyi Lao<sup>1</sup>, Jiachen Zhang<sup>1</sup>, Tianle Zhang<sup>2</sup>, Songfang Huang<sup>1</sup> <sup>1</sup>Peking University <sup>2</sup>JD Explore Academy

## Abstract

Action-chunking robot policies predict sequences of future actions, but deployment requires choosing how many actions to execute before replanning. The execution horizon trades off feedback frequency against motion continuity. However, success varies non-monotonically with the execution horizon, and the best-performing horizon differs across tasks. These effects make a single fixed horizon unreliable as a default execution strategy across tasks. We propose PACE (Phase-Aware Chunk Execution), a training-free method that selects execution horizons online from predicted action chunks. PACE uses prominent low-speed valleys in the predicted speed profile as candidate replanning boundaries, exploiting the phase-dependent kinematics of manipulation trajectories. By default, PACE calibrates the valley threshold from each task’s existing training demonstrations, without task-specific fixed-horizon tuning. On the 50-task RoboTwin2.0 benchmark, PACE improves average success from 57.8% to 64.2% over the strongest fixed-horizon baseline and outperforms the adaptive execution baselines AAC and AutoHorizon. With GR00T N1.7 on LIBERO, PACE achieves the highest overall success rate among the evaluated strategies, supporting its applicability across base policies, benchmarks, and action representations. On real robots, PACE improves the average task score from 60.7 to 77.7 and success rate from 50.7% to 70.4% over full-chunk execution. Code is available on our project page.

## 1 Introduction

Recent progress in vision-language-action models and diffusion-based robot policies has largely come from training-side gains. Larger backbones, broader demonstration data, and stronger multi-modal pretraining have all improved predictive performance [1–7]. At deployment, however, performance depends not only on how accurately a policy predicts, but also on how those predictions are executed. To improve local motion continuity, these policies often adopt action chunking [4, 8], where each query predicts a short sequence of future actions rather than a single-step command.

Yet an action chunk only specifies what the policy predicts, not how much of that prediction the robot should execute. Its length fixes the prediction horizon, but a separate choice must still be made about how many actions to execute before acquiring a new observation and querying the policy again. We refer to this executed prefix length as the execution horizon. Unlike the prediction horizon, the execution horizon is neither learned nor prescribed by the policy [4, 8]. It is an inference-time choice that trades off feedback frequency against within-chunk motion continuity. As such, the execution horizon is not a mere implementation detail, but a central variable for reliable deployment of action-chunking robot policies.

Dynamic execution within one rollout  
![](images/a4202513c8514652ea4c7cdb484f5ebf95969bbce8af60a592614f576e1190aa.jpg)

![](images/766eb28a0f40009cb56b3a6c003e5838a060ffff19d8541f43f5a14a190ac8dc.jpg)  
Figure 1: Overview of PACE. Fixed horizons can be unreliable because success varies with H across tasks. PACE selects the executed prefix online from each predicted chunk, improving performance in simulation and real-robot experiments while adapting the horizon in a rollout.

In practice, the execution horizon is typically fixed to a constant H throughout an episode [4, 8]. At each replanning step, the robot executes the first H actions of the predicted chunk before querying the policy again, tying all chunks to the same replanning interval regardless of the motion they predict. A smaller H incorporates new observations more often but can interrupt a coherent local motion, while a larger H follows the predicted chunk longer but delays reactive correction. Without a principled selection criterion, H is usually chosen empirically.

Sweeping H from 1 to 50 on three RoboTwin2.0 tasks reveals strongly non-monotonic success rates and task-dependent preferred horizons (Fig. 1, left). On Click Bell, success peaks near H ≈ 6, drops by nearly 40 points around H ≈ 25, and rebounds to a second peak near H ≈ 35. A fixed horizon can therefore cause substantial performance degradation on some tasks. The best horizon found offline on one task also does not transfer to new tasks or real-robot deployment. These findings indicate that the execution horizon must be selected online rather than fixed in advance.

We propose PACE (Phase-Aware Chunk Execution), a test-time method that selects the execution horizon online from the speed profile of each predicted action chunk. Manipulation trajectories consist of locally coherent motion segments, with transitions often accompanied by deceleration. Policies trained via imitation learning preserve this structure in their predicted action chunks, where phase transitions can appear as low-speed valleys in the speed profile. PACE treats such valleys as candidate replanning points and executes the chunk up to the earliest accepted valley, aiming to preserve coherent motion within a segment and re-query the policy before the next begins. PACE uses no model-internal signals or learned components and requires no policy changes or retraining.

On the 50-task RoboTwin2.0 benchmark [9], PACE improves the average success rate of π<sub>0.5</sub> from 57.8% to 64.2% over the best fixed-horizon baseline and outperforms the adaptive baselines AAC and AutoHorizon. On real robots, it improves average task score from 60.7 to 77.7 and success rate from 50.7% to 70.4% over full-chunk execution. With GR00T N1.7 on LIBERO, PACE achieves the highest overall success rate among the evaluated strategies, supporting its applicability across policies, benchmarks, and action representations. Ablations, sensitivity analyses, and fixed-horizon sweeps on representative tasks assess its design choices, while analyses of successful and failed rollouts reveal its working mechanism and failure modes. Our contributions are summarized as follows:

• We identify the execution horizon as a critical yet overlooked test-time variable, and show empirically that its effect on success is both task-dependent and non-monotonic, indicating that no single fixed horizon provides a reliable default across tasks.

![](images/186e7dc3964a9b0f3e14b78e6048ea40d6009d4cbf111a232c4969df9b768eb5.jpg)  
Figure 2: PACE framework. PACE selects an execution horizon from low-speed valleys in the predicted chunk’s smoothed speed profile. The robot executes the selected prefix, discards the suffix, and then queries the policy again.

• We propose PACE, a plug-and-play test-time method that selects the executed prefix online from the phase-dependent kinematic structure of predicted action chunks. It requires no model-internal signals, no learned module, and no modification or retraining of the policy.

• We validate PACE with $\pi _ { 0 . 5 }$ on all 50 RoboTwin2.0 tasks, with GR00T N1.7 on LIBERO, and on real robots. Comparisons with fixed-horizon and adaptive baselines demonstrate its effectiveness, while ablations and rollout analyses explain its performance gains.

## 2 Related work

Robot policies with action chunking. Action chunking has become a common interface in robot policy learning. It was popularized by ACT for fine-grained bimanual imitation learning [8] and Diffusion Policy for receding-horizon action-sequence generation [4]. Subsequent work applies this interface to mobile manipulation [10], 3D visuomotor control [11], and large-scale bimanual policy learning [12]. Generalist policies such as Octo, π , and $\pi _ { 0 . 5 }$ extend chunk prediction to diverse tasks and environments [3, 13, 14]. VLA research also explores efficient action tokenization [15] and faster inference for action prediction [16, 17]. While these works focus on generating, representing, or scaling action chunks, PACE addresses the complementary deployment-time question of how much of each predicted chunk to execute before incorporating new observations and replanning.

Test-time execution of action chunks. Many policies execute action chunks using fixed testtime rules: ACT smooths overlapping predictions through temporal ensembling, while Diffusion Policy executes a fixed action horizon in a receding-horizon loop [4, 8]. Recent inference-time methods reduce latency-induced discontinuities or stabilize chunk execution [18, 19]. Another line modifies training or model design to reason over multiple horizons [20]. More closely related are methods that adapt the execution horizon at inference time. Adaptive Action Chunking (AAC) [21] estimates prefix-wise action entropy from multiple sampled chunks and selects the horizon subject to a minimum-movement constraint. AutoHorizon [22] estimates the execution horizon from action-token self-attention in flow-based VLAs, treating attention as a proxy for the model’s predictive limit. PACE instead selects replanning boundaries from prominent low-speed valleys in the kinematic profile of a single decoded action chunk, without requiring model-internal signals or additional chunk samples.

## 3 Method

## 3.1 Problem formulation

We consider a manipulation policy with action chunking. At the i-th policy query, the policy receives the observation $o _ { \tau _ { i } }$ and the language instruction ℓ, and outputs

$$
\mathbf {A} _ {i} = \pi_ {\theta} (o _ {\tau_ {i}}, \ell) \in \mathbb {R} ^ {L \times d _ {a}}, \quad \mathbf {A} _ {i} = (a _ {i, 1}, \dots , a _ {i, L}),\tag{1}
$$

where L is the prediction horizon and $d _ { a }$ is the action dimension. Let $h _ { i } \in \{ 1 , \ldots , L \}$ denote the execution horizon, i.e., the number of predicted actions executed before the next policy query. After executing the prefix $( a _ { i , 1 } , \ldots , a _ { i , h _ { i } } )$ , the next query occurs at $\tau _ { i + 1 } = \tau _ { i } + h _ { i }$

A fixed-horizon rule sets $h _ { i } ^ { \mathrm { f i x e d } } = H$ for all $i ,$ whether the predicted chunk describes a smooth transit or a contact event. PACE instead conditions the execution horizon on the predicted chunk,

$$
h _ {i} = g (\mathbf {A} _ {i}), \qquad g: \mathbb {R} ^ {L \times d _ {a}} \to \{1, \dots , L \}.\tag{2}
$$

The deterministic, training-free function $g$ uses fixed hyperparameters (Secs. 3.2–3.3) without modifying the trained policy $\pi _ { \theta }$ . The prediction horizon L remains the policy’s training horizon (Sec. 4.3).

## 3.2 Kinematic profile extraction

Expert manipulation trajectories are rarely uniform in speed. They organize into alternating fast and slow segments, where low-speed regions typically arise under contact, stabilization, or sub-stage transitions. An imitation-trained policy tends to preserve this structure in its predicted action chunks. This makes the speed profile a kinematic cue for where one locally coherent motion segment ends and the next begins, and thus a natural basis for the selection rule g.

Concretely, for each controllable arm group $b = 1 , \dots , B$ , let $q _ { i , j } ^ { b } \in \mathbb { R } ^ { d _ { q } }$ denote the joint command at predicted step j. We define the speed profile $\mathbf v _ { i } ^ { b } = ( v _ { i , 1 } ^ { b } , \dots , v _ { i , L - 1 } ^ { b } )$ by

$$
v _ {i, j} ^ {b} = f _ {s} \left\| q _ {i, j + 1} ^ {b} - q _ {i, j} ^ {b} \right\| _ {2}, \qquad j = 1, \ldots , L - 1,\tag{3}
$$

where $f _ { s }$ is the control frequency. Raw speed profiles contain high-frequency fluctuations that create spurious local valleys and, if untreated, trigger overly frequent replanning. We therefore apply a Savitzky–Golay filter S to obtain the smoothed profile $\tilde { \mathbf { v } } _ { i } ^ { b } = \mathcal { S } ( \mathbf { v } _ { i } ^ { b } )$ , whose low-speed valleys serve as candidate replanning boundaries for the selection rule in Sec. 3.3. We use joint-space speed by default; a Cartesian variant yields comparable success after smoothing (Sec. 4.3), and filter parameters are given in Appendix A.

## 3.3 Phase-aware execution horizon selection

Given the kinematic profile from Sec. 3.2, we now ask where inside the chunk the policy should stop and replan. Our guiding observation, illustrated in Fig. 2, is that phase boundaries such as contacts, grasps, and releases can manifest as pronounced low-speed valleys in $\tilde { \mathbf { v } } _ { i } ^ { b } \colon$ the end-effector decelerates as it approaches the event and accelerates again once it is resolved. The function g therefore searches for the earliest such valley and treats it as a candidate point to hand control back to the policy.

Not every local minimum qualifies: a valley is informative only when preceded by a clearly faster segment, otherwise it merely reflects noise on an already slow trajectory. Let $\mathcal { V } _ { i } ^ { b } \subseteq \{ 1 , \dotsc , \mathbf { \bar { L } } - 1 \}$ denote the local minima of $\tilde { \mathbf { v } } _ { i } ^ { b }$ satisfying a minimum spacing constraint. For each candidate $r \in \mathcal { V } _ { i } ^ { b }$ we define the preceding maximum

$$
m _ {i} ^ {b} (r) = \max _ {1 \leq j \leq r} \tilde {v} _ {i, j} ^ {b},\tag{4}
$$

and the contrast score

$$
\Delta_ {i} ^ {b} (r) = \min \bigl \{m _ {i} ^ {b} (r) - \tilde {v} _ {i, 1} ^ {b}, m _ {i} ^ {b} (r) - \tilde {v} _ {i, r} ^ {b} \bigr \},\tag{5}
$$

which requires the speed to both rise substantially before the valley and drop substantially into it, so that r sits at the bottom of a genuine deceleration rather than on a flat region. A valley is accepted when $\Delta _ { i } ^ { b } ( r ) \ge \delta _ { T }$ , where $\delta \tau$ is calibrated once per task from offline training trajectories using a fixed percentile rule shared across tasks.

Table 1: RoboTwin2.0 results with $\pi _ { 0 . 5 }$ . Success rates (%) on six representative tasks, grouped by task duration, and all 50 tasks. Overall is the mean success rate over all 50 tasks. Bold marks the highest mean in each column. Red values after the slash report PACE improvements over the best of H = 5, 25, 50 in each column (percentage points, computed before rounding).

<table><tr><td rowspan="2">Method</td><td colspan="2">Short</td><td colspan="2">Medium</td><td colspan="2">Long</td><td rowspan="2">Overall(50 tasks)</td></tr><tr><td>Pick Bottles</td><td>Grab Roller</td><td>Place A2B</td><td>Dump Bin</td><td>Put Bottles</td><td>Stack Bowls</td></tr><tr><td>H = 5</td><td>13.3</td><td>89.9</td><td>20.0</td><td>58.7</td><td>0.3</td><td>88.7</td><td>48.8</td></tr><tr><td>H = 25</td><td>45.3</td><td>96.2</td><td>47.8</td><td>85.1</td><td>79.3</td><td>88.4</td><td>57.8</td></tr><tr><td>H = 50</td><td>38.7</td><td>96.2</td><td>36.8</td><td>84.3</td><td>70.8</td><td>85.1</td><td>53.4</td></tr><tr><td>AutoHorizon</td><td>41.6</td><td>96.9</td><td>43.1</td><td>88.1</td><td>74.3</td><td>91.0</td><td>59.0</td></tr><tr><td>AAC</td><td>48.9</td><td>97.7</td><td>46.7</td><td>89.1</td><td>76.6</td><td>88.2</td><td>59.1</td></tr><tr><td>PACE (Ours)</td><td>48.2 / +2.9</td><td>98.6 / +2.3</td><td>54.6 / +6.8</td><td>93.0 / +7.9</td><td>88.9 / +9.6</td><td>95.6 / +6.9</td><td>64.2 / +6.4</td></tr></table>

Accepted valleys from all arm groups are pooled into

$$
\mathcal {R} _ {i} = \bigcup_ {b = 1} ^ {B} \big \{r \in \mathcal {V} _ {i} ^ {b}: \Delta_ {i} ^ {b} (r) \geq \delta_ {\mathcal {T}} \big \}.\tag{6}
$$

We take the union because a candidate phase transition on any single arm is already a reason to replan, and we select the earliest candidate, aiming to query the policy before, rather than after, a boundary is crossed with stale observations. When no valley is accepted, PACE falls back to the maximum execution horizon, giving

$$
h _ {i} = g (\mathbf {A} _ {i}) = \left\{ \begin{array}{l l} \min \mathcal {R} _ {i}, & \mathcal {R} _ {i} \neq \emptyset , \\ H _ {\max}, & \mathcal {R} _ {i} = \emptyset , \end{array} \right.\tag{7}
$$

where $H _ { \mathrm { m a x } } \leq L$ is the execution horizon used when no valley is accepted. Calibration of $\delta \tau$ , the valley-spacing constant, and the value of $H _ { \mathrm { m a x } }$ are deferred to Appendix A.

## 4 Simulation experiments

Experimental setup. We evaluate PACE on 50 RoboTwin2.0 bimanual manipulation tasks, training one $\pi _ { 0 . 5 }$ checkpoint per task under the standard protocol. All methods share the same checkpoint and differ only in test-time execution. Unless otherwise noted, ourfull evaluation protocol uses 900 episodes per task (three seeds, each with three batches of 100 episodes), with success rates averaged over all 50 tasks. Task-level tables show six representative tasks balanced across short-, medium-, and long-duration groups. Full results and training details are in Appendix B.

Overview of experiments. We compare PACE as a default execution strategy with fixed-horizon and adaptive baselines (Sec. 4.1) and analyze its horizon adaptation within rollouts (Sec. 4.2). We ablate the phase profile, training prediction horizon, and threshold sharing (Sec. 4.3), and use task-specific fixed-horizon sweeps as a diagnostic upper bound (Sec. 4.4).

## 4.1 Simulation results

We compare PACE against fixed-horizon execution with $H \in \{ 5 , 2 5 , 5 0 \}$ , corresponding to short, intermediate, and full-chunk prefixes (recall $H _ { \mathrm { t r a i n } } { = } 5 0 )$ ). We also evaluate Adaptive Action Chunking (AAC) [21] and AutoHorizon [22] using the same task-specific $\pi _ { 0 . 5 }$ checkpoints. Our bimanual jointspace adaptation of AAC estimates action entropy from 20 sampled chunks per policy query. PACE calibrates one threshold per task from the checkpoint’s existing training demonstrations, without using evaluation rollouts, success labels, or a fixed-horizon search (Appendix A.3).

Table 1 shows that PACE achieves the highest 50-task average, exceeding the best global fixed horizon (H = 25) by 6.4 points and AAC and AutoHorizon by approximately 5.1 and 5.2 points, respectively. PACE also exceeds the strongest fixed horizon on every displayed task and achieves the highest mean on five of the six. AAC leads on Pick Bottles (48.9% versus 48.2%).

![](images/ea36f1665072722c83e28a432015d273932e42ba301117221e54beae4b3fac09.jpg)

![](images/9779cbe08b0210d7fe2ba53498af1754aefc0ffe56319f49be45f7ce601f9bfb.jpg)  
Figure 3: Rollout-level behavior of PACE. The rollout is from place\_shoe. The top rows show head and front camera observations at the six replanning timesteps (t = 0, 32, 46, 83, 108, 127), with the selected execution horizon annotated between consecutive queries. The bottom panel shows the six predicted action chunks along the rollout timeline, one color per plan. For each plan, the horizontal bar marks the executed prefix (solid) and discarded suffix (dashed), and the curve below shows the corresponding predicted velocity profile under the same solid/dashed convention. Circles mark the execution start point of each plan, and vertical dashed lines mark the replanning boundaries.

Cross-policy evaluation. We evaluate PACE with GR00T N1.7 [23] on all four LIBERO suites [24], covering 40 tasks. GR00T predicts 16-step Cartesian-delta chunks. Without retraining or modifying GR00T, PACE constructs its motion profile from the translational-delta norm and applies the same valley-based selection rule. PACE achieves the highest overall success rate, 95.33%, compared with 94.47% for the strongest global fixed horizon (H = 16) and 94.92% for AutoHorizon. These results support the applicability of PACE across different base policies, benchmarks, action representations, and chunk lengths. The horizon selector takes 0.0566 ms per query, approximately 0.044% of the time required for a single policy forward pass (Appendix B.2).

## 4.2 Visualizing adaptive execution horizons

We visualize a representative place\_shoe rollout in Fig. 3 to examine how PACE adapts its execution horizon. Selected horizons range from 14 to 37 steps, with longer horizons covering coherent motion (the approach in Plan 3, 37 steps) and shorter horizons near low-speed transitions (Plan 2, 14 steps). A fixed horizon cannot adapt to this within-episode structure.

The discarded suffixes show how PACE limits open-loop execution as the scene changes. PACE replans at boundaries selected from the predicted motion structure, and the new plan often differs from the discarded suffix over the overlapping interval, as seen in the mismatched solid and dashed curves of adjacent plans around t=83. This limits reliance on stale observations while preserving long horizons for coherent motion, helping explain the gains over fixed-horizon baselines in Sec. 4.1. Appendix D.3 quantifies the alignment of replanning boundaries with motion-phase transitions.

Table 2: Ablation studies. Left: phase-profile ablation on all 50 RoboTwin2.0 tasks; Avg. Horizon is the mean selected execution horizon. Right: training-horizon ablation on six representative tasks; each row fixes $H _ { \mathrm { e v a l } }$ and varies $H _ { \mathrm { t r a i n } }$ . Gain@50 is the gain of $H _ { \mathrm { t r a i n } } { = } 5 0$ over the shortest feasible setting. Success rates are in %. <sup>†</sup> marks the default.

<table><tr><td>Configuration</td><td>Succ.</td><td>Avg. Horizon</td></tr><tr><td>Cartesian, raw</td><td>62.2</td><td>11.3</td></tr><tr><td>Cartesian, smoothed</td><td>64.5</td><td>19.5</td></tr><tr><td>Joint, raw</td><td>60.4</td><td>16.7</td></tr><tr><td>Joint, smoothed $^{\dagger}$ </td><td>64.2</td><td>24.3</td></tr></table>

<table><tr><td> $H_{eval} \backslash H_{train}$ </td><td>10</td><td>20</td><td>30</td><td>40</td><td>50</td><td>Gain@50</td></tr><tr><td>10</td><td>57.2</td><td>68.7</td><td>69.9</td><td>74.6</td><td>77.9</td><td>+20.7</td></tr><tr><td>20</td><td>-</td><td>68.7</td><td>72.6</td><td>73.8</td><td>75.3</td><td>+6.6</td></tr><tr><td>30</td><td>-</td><td>-</td><td>70.2</td><td>71.4</td><td>73.8</td><td>+3.6</td></tr><tr><td>40</td><td>-</td><td>-</td><td>-</td><td>70.1</td><td>73.4</td><td>+3.3</td></tr></table>

## 4.3 Ablation analysis

We examine how motion-profile construction, the training prediction horizon, and threshold sharing affect PACE’s performance. We evaluate motion-profile variants and compare shared versus per-task thresholds on all 50 RoboTwin2.0 tasks, with unchanged policy checkpoints. To study the training prediction horizon $H _ { \mathrm { t r a i n } } .$ , we train policies to predict action chunks of different lengths. We compare their success rates at the same fixed execution horizon to assess whether supervision over longer action sequences improves the quality of the executed prefix. Since each setting requires retraining, we evaluate them on six representative tasks. Further details and results appear in Appendix C.

Phase-profile construction. We compare joint-space and Cartesian speed profiles, with and without smoothing, to examine whether PACE depends on a specific motion representation and how smoothing affects boundary selection. The Cartesian variant computes end-effector linear speed through forward kinematics. We keep the valley-selection rule and calibration percentile fixed across all four variants. Table 2 (left) shows that smoothing improves success in both spaces and increases the mean selected execution horizon. This pattern suggests that smoothing suppresses noisy local valleys that would otherwise trigger overly frequent replanning. After smoothing, Cartesian and joint-space profiles achieve comparable success, while the joint-space variant selects longer execution horizons on average. We adopt the smoothed joint-space profile by default. It is within 0.3 points of the best variant, matches the policy action space, and avoids forward kinematics at every replanning query.

Training prediction horizon. Each row of Table 2 (right) fixes $H _ { \mathrm { e v a l } }$ and varies $H _ { \mathrm { t r a i n } } ;$ pairs with $H _ { \mathrm { t r a i n } } < H _ { \mathrm { e v a l } }$ are infeasible because the policy predicts too few actions. Success increases monotonically with $H _ { \mathrm { t r a i n } }$ , and the gain of $H _ { \mathrm { t r a i n } } { = } 5 0$ over the shortest feasible target shrinks from +20.7 at $H _ { \mathrm { e v a l } } { = } 1 0$ to +3.3 at $H _ { \mathrm { e v a l } } { = } 4 0$ . This trend shows that longer training targets provide stronger sequence-level supervision: they constrain not just the action after the observation, but how motion evolves across a longer expert segment. Since the executed prefix is predicted jointly with the rest of the chunk, it inherits this structure. We adopt $H _ { \mathrm { t r a i n } } { = } 5 0$ as the default prediction horizon.

Table 3: Shared versus per-task threshold calibration. Success rates (%), reported as mean ± sample SD across three evaluation seeds.

<table><tr><td>Execution</td><td>Threshold calibration</td><td>Success (%)</td></tr><tr><td>Fixed  $H = 25$ </td><td>Not applicable</td><td> $57.80 \pm 1.49$ </td></tr><tr><td>PACE, shared</td><td>Pooled training demonstrations</td><td> $62.44 \pm 0.92$ </td></tr><tr><td>PACE, per-task</td><td>Each task’s training demonstrations</td><td> $\mathbf{64.20} \pm 0.88$ </td></tr></table>

Shared versus per-task thresholds. By default, PACE calibrates a separate valley-acceptance threshold from each task’s training demonstrations. To assess how much PACE depends on per-task calibration, we pool the training demonstrations from all 50 tasks to calibrate a single threshold and apply it unchanged across tasks. All other PACE settings and evaluation conditions remain unchanged. The shared threshold governs valley acceptance, while execution horizons still adapt to the speed profile of each predicted action chunk. Table 3 shows that the shared threshold retains most of PACE’s gain over fixed H = 25, improving success by 4.64 percentage points. Per-task calibration provides a further gain of approximately 1.8 points and remains our default (Appendix C.4).

![](images/32bc0b6b134f980526d30c8b9e8320549e86ff28f0872a4a662583b04cda150b.jpg)  
Figure 4: PACE compared with fixed-horizon sweeps. Each panel shows one of the six representative tasks. The green curve reports success as the fixed execution horizon H varies from 1 to 50. The red star marks PACE’s success rate at its mean selected execution horizon.

## 4.4 PACE versus fixed-horizon sweeps

We compare PACE with task-specific fixed-horizon sweeps on six representative tasks. For each task, we evaluate $H = 1 , \ldots , 5 0$ to obtain the success-rate curve and the best observed fixed-horizon performance. These results serve as diagnostic references for evaluating PACE, which selects execution horizons online without task-specific fixed-horizon search.

Figure 4 shows that the success curves and preferred horizons vary substantially across tasks. For example, the best fixed horizon is H = 8 on Stack Bowls Two but H = 37 on Dump Bin Bigbin. PACE reaches the high-success region of each task without task-specific fixed-horizon tuning, with mean execution horizons ranging from 19.7 to 28.9 steps. These results are achieved without uniformly short execution prefixes.

Table 4: Matched-horizon comparison on RoboTwin2.0. For each task, the fixed horizon is obtained by rounding PACE’s mean execution horizon to the nearest integer. Red values after the slash report PACE’s success-rate gains over the matched baseline.

<table><tr><td rowspan="2">Setting</td><td colspan="2">Short</td><td colspan="2">Medium</td><td colspan="2">Long</td></tr><tr><td>Pick Bottles</td><td>Grab Roller</td><td>Place A2B</td><td>Dump Bin</td><td>Put Bottles</td><td>Stack Bowls</td></tr><tr><td>Fixed H</td><td>26</td><td>29</td><td>24</td><td>23</td><td>22</td><td>20</td></tr><tr><td>PACE mean H</td><td>25.5</td><td>28.9</td><td>23.8</td><td>22.7</td><td>22.3</td><td>19.7</td></tr><tr><td>Fixed (%)</td><td>46.7</td><td>96.0</td><td>44.0</td><td>88.7</td><td>87.0</td><td>88.3</td></tr><tr><td>PACE (%)</td><td>48.2 / +1.6</td><td>98.6 / +2.6</td><td>54.6 / +10.6</td><td>93.0 / +4.3</td><td>88.9 / +1.9</td><td>95.6 / +7.2</td></tr></table>

Matched execution horizons. To test whether PACE’s performance can be reproduced by a constant execution length, we round its mean horizon on each task to the nearest integer and evaluate that fixed horizon (Appendix D). Despite similar mean horizons, the baseline replans at fixed intervals, whereas PACE selects a horizon from each predicted chunk. Table 4 shows higher observed success for PACE on all six tasks. On Place A2B Right, PACE achieves 54.6% success with a mean horizon of 23.8 steps, versus 44.0% for fixed H = 24. These results suggest that matching the mean horizon alone is insufficient to reproduce PACE’s performance.

Table 5: Real-robot results. Task scores and full-task success rates (%) on RoboChallenge and the Franka task. Red values after the slash indicate gains over fixed H = 50.

<table><tr><td rowspan="2">Method</td><td colspan="2">Stack Bowls</td><td colspan="2">Put Pen in Case</td><td colspan="2">Place Object on Plate</td></tr><tr><td>Score</td><td>Succ.</td><td>Score</td><td>Succ.</td><td>Score</td><td>Succ.</td></tr><tr><td>Baseline</td><td>73.0</td><td>70.0</td><td>37.2</td><td>10.0</td><td>72.0</td><td>72.0</td></tr><tr><td>PACE</td><td>93.5 / +20.5</td><td>90.0 / +20.0</td><td>51.7 / +14.5</td><td>33.3 / +23.3</td><td>88.0 / +16.0</td><td>88.0 / +16.0</td></tr></table>

![](images/093b5be11da8db0e2c2ddc067cb2746bcbbef3b79ab7f17a0c2a35fe8d63eaed.jpg)  
Figure 5: Successful rollout. The rollout is from stack\_bowls. Blue: policy-query timestep. Green: horizon length. The horizon is long during approach and transport (27, 30, 31), short at contact-sensitive alignment (7), and long again afterward (43).

## 5 Real-robot experiments

We evaluate PACE on real robots in two settings: RoboChallenge [25] and our in-lab place\_object\_on\_plate task family. RoboChallenge is a public real-robot benchmark; we evaluate stack\_bowls and put\_pen\_into\_pencil\_case with 30 trials per method per task. In place\_object\_on\_plate, a Franka robot places one of five objects onto a plate in our laboratory setup. Within each task, all methods use the same fine-tuned $\pi _ { 0 . 5 }$ checkpoint and differ only in the test-time execution scheme. We report task Score and full-completion rate Succ. for both settings; task definitions, scoring rules, training details, and per-object in-lab results appear in Appendix E.

## 5.1 Real-robot results

Table 5 compares PACE with the full-chunk execution baseline $( H = 5 0 )$ on the three real-robot tasks. PACE improves both task score and success rate on every task, raising the average success rate from 50.7% to 70.4% and the average task score from 60.7 to 77.7, with each task weighted equally.

To assess whether more frequent replanning with a fixed horizon yields similar performance, we further evaluate $H = 5$ and H = 25 on the Franka task. PACE achieves an 88% success rate, compared with 87% for $H = 5$ and 82% for $H = 2 5$ . It thus attains a success rate comparable to $H = 5$ while requiring 84.6% fewer policy queries per successful episode (Appendix E.3).

## 5.2 PACE horizon selection in a successful rollout

Figure 5 visualizes a successful stack\_bowls rollout and shows that PACE adjusts its execution horizon according to the local manipulation phase rather than a fixed replanning cadence. During approach and transport, the method selects long horizons (27, 30, 31 actions), allowing the robot to maintain continuous motion while open-loop execution remains reliable. Near contact-sensitive alignment, where small pose errors determine whether the bowl stacks, the horizon contracts to 7 actions so the next query can incorporate an updated observation. Once alignment is resolved, the horizon expands again to 43, showing that the method does not remain in a high-frequency replanning mode when a coherent motion segment becomes available.

This behavior mirrors the simulation analysis in Sec. 4.2 and helps explain the real-robot gains in Table 5: PACE introduces feedback where stale observations are most costly, while avoiding unnecessary queries during smooth motion. Additional rollout visualizations for representative RoboTwin2.0 and RoboChallenge tasks are provided in Appendix F.

![](images/d70ce60df06e75045408d8188a4b0b4bbd97831da858ff5727ed6e4b3e84b6e0.jpg)  
Figure 6: Failure case. The rollout is from put\_pen\_into\_pencil\_case (top: head view; bottom: left-wrist view). Blue: policy-query timestep. Green: horizon length. The pencil case remains only partially opened while the right arm moves the pen toward it.

## 5.3 Failure case and scope of test-time execution control

Figure 6 shows a failed put\_pen\_into\_pencil\_case rollout. The task is strictly sequential: the left gripper must fully open the pencil case before the right arm inserts the pen. In this rollout, PACE selects short horizons (7 and 6 actions) around the insertion phase, so the policy refreshes its action chunk multiple times from updated observations. Yet the pencil case remains only partially opened throughout the segment, and the task fails.

The failure is therefore not due to feedback timing, but to the learned action distribution of the base policy. PACE already shortens the execution horizon near insertion and refreshes the chunk from updated observations, but the refreshed chunks are still produced by the same policy and do not yield task-completing behavior in this rollout. This case highlights the boundary of test-time execution control: it can decide when to query the policy, but it cannot compensate when the policy itself fails to generate actions sufficient for task completion.

## 6 Limitations

PACE cannot resolve failures when the base policy continues to predict incomplete or incorrect actions (Sec. 5.3). Its replanning rule can miss phase transitions without clear low-speed valleys. If no valley is accepted, PACE falls back to the maximum execution horizon. Effectiveness with policy families and action representations beyond those evaluated here remains to be assessed. Threshold calibration requires training demonstrations, either per task or pooled across tasks for a shared threshold. Sharing avoids separate per-task calibration but retains this data requirement. Earlier replanning can increase policy queries relative to full-chunk execution, raising inference cost even though the selector is inexpensive.

## 7 Conclusion

We showed that the execution horizon is a critical deployment variable with strongly task-dependent and non-monotonic effects on success. PACE selects execution horizons online from the phasedependent kinematics of predicted action chunks, using prominent low-speed valleys as candidate replanning boundaries without additional training. On RoboTwin2.0, PACE improves the average success rate of $\pi _ { 0 . 5 }$ over both fixed-horizon and adaptive execution baselines. Applying PACE to GR00T N1.7 on LIBERO also yields the highest overall success rate among the evaluated strategies, supporting its applicability across different base policies, benchmarks, and action representations. Real-robot evaluations further demonstrate improvements in task score and success rate over fullchunk execution. Ablations support motion-profile smoothing and longer training prediction horizons, while a shared threshold retains most gains. Gains over matched fixed-horizon baselines further support adaptation within a rollout. PACE offers a practical way to improve task performance through adaptive execution without modifying or retraining the policy.

## Acknowledgments and Disclosure of Funding

This work was carried out at Peking University and JD Explore Academy. This work was supported by the Beijing Natural Science Foundation International Scientists Project (Grant No. IS26042). The authors declare no other competing interests.

## References

[1] Anthony Brohan et al. RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control, 2023. URL https://arxiv.org/abs/2307.15818.

[2] Moo Jin Kim et al. OpenVLA: An Open-Source Vision-Language-Action Model, 2024. URL https://arxiv.org/abs/2406.09246.

[3] Kevin Black et al. π<sub>0</sub>: A Vision-Language-Action Flow Model for General Robot Control, 2024. URL https://arxiv.org/abs/2410.24164.

[4] Cheng Chi et al. Diffusion Policy: Visuomotor Policy Learning via Action Diffusion, 2023. URL https://arxiv.org/abs/2303.04137.

[5] Tianle Zhang et al. JoyAI-RA 0.1: A Foundation Model for Robotic Autonomy, 2026. URL https://arxiv.org/abs/2604.20100.

[6] JoyAI-RA Team. JoyAI-RA 0.5: Scaling Robot Manipulation Learning via Dual Action Alignment, 2026. URL https://arxiv.org/abs/2608.05674.

[7] Yixue Zhang et al. RoboGene: Diversity-Driven Agentic Task Generation Framework for VLA Pre-training. In Proceedings ofthe 10th Conference on Robot Learning (CoRL), 2026. URL https://arxiv.org/abs/2602.16444.

[8] Tony Z. Zhao et al. Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware, 2023. URL https://arxiv.org/abs/2304.13705.

[9] Tianxing Chen et al. RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation, 2025. URL https: //arxiv.org/abs/2506.18088.

[10] Zipeng Fu et al. Mobile ALOHA: Learning Bimanual Mobile Manipulation with Low-Cost Whole-Body Teleoperation, 2024. URL https://arxiv.org/abs/2401.02117.

[11] Yanjie Ze et al. 3D Diffusion Policy: Generalizable Visuomotor Policy Learning via Simple 3D Representations, 2024. URL https://arxiv.org/abs/2403.03954.

[12] Songming Liu et al. RDT-1B: A Diffusion Foundation Model for Bimanual Manipulation, 2024. URL https://arxiv.org/abs/2410.07864.

[13] Octo Model Team et al. Octo: An Open-Source Generalist Robot Policy, 2024. URL https: //arxiv.org/abs/2405.12213.

[14] Physical Intelligence et al. π<sub>0.5</sub>: A Vision-Language-Action Model with Open-World Generalization, 2025. URL https://arxiv.org/abs/2504.16054.

[15] Karl Pertsch et al. FAST: Efficient Action Tokenization for Vision-Language-Action Models, 2025. URL https://arxiv.org/abs/2501.09747.

[16] Moo Jin Kim et al. Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success, 2025. URL https://arxiv.org/abs/2502.19645.

[17] Mustafa Shukor et al. SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics, 2025. URL https://arxiv.org/abs/2506.01844.

[18] Kevin Black et al. Real-Time Execution of Action Chunking Flow Policies, 2025. URL https://arxiv.org/abs/2506.07339.

[19] Haoxuan Wang et al. Real-Time Robot Execution with Masked Action Chunking, 2026. URL https://arxiv.org/abs/2601.20130.

[20] Dong Jing et al. Mixture of Horizons in Action Chunking, 2025. URL https://arxiv.org/ abs/2511.19433.

[21] Yuanchang Liang et al. Adaptive Action Chunking at Inference-time for Vision-Language-Action Models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 20802–20811, June 2026. URL https://arxiv.org/abs/2604. 04161.

[22] Haoxuan Wang et al. VLA Knows Its Limits: Adaptive Execution Horizons for Robot Policies, 2026. URL https://arxiv.org/abs/2602.21445. ECCV 2026.

[23] NVIDIA. NVIDIA Isaac GR00T N1.7. https://github.com/NVIDIA/Isaac-GR00T, 2026.

[24] Bo Liu et al. LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning, 2023. URL https://arxiv.org/abs/2306.03310.

[25] Adina Yakefu et al. RoboChallenge: Large-scale Real-robot Evaluation of Embodied Policies, 2025. URL https://arxiv.org/abs/2510.17950.

[26] Alexander Khazatsky et al. DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset, 2024. URL https://arxiv.org/abs/2403.12945.

[27] Qingwen Bu et al. AgiBot World Colosseo: A Large-Scale Manipulation Platform for Scalable and Intelligent Embodied Systems. In 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2025. URL https://doi.org/10.1109/IROS60139. 2025.11247088.

[28] Kun Wu et al. RoboMIND: Benchmark on Multi-Embodiment Intelligence Normative Data for Robot Manipulation. In Proceedings of Robotics: Science and Systems (RSS), 2025. URL https://www.roboticsproceedings.org/rss21/p152.html.

[29] Shihan Wu et al. RoboCOIN: An Open-Sourced Bimanual Robotic Data Collection for Integrated Manipulation, 2025. URL https://arxiv.org/abs/2511.17441.

## A PACE implementation details

This appendix specifies the default configuration, online selection algorithm, and demonstration-based threshold calibration for PACE.

## A.1 Default configuration

Table 6 summarizes the default PACE configuration for the $\pi _ { 0 . 5 }$ experiments on RoboTwin2.0. The GR00T N1.7 configuration on LIBERO is described in Appendix B.2.

Table 6: Default PACE configuration for $\pi _ { 0 . 5 }$ on RoboTwin2.0.

<table><tr><td>Component</td><td>Default value</td></tr><tr><td>Prediction horizon  $L$ </td><td>50</td></tr><tr><td>Maximum execution horizon  $H_{\text{max}}$ </td><td>50</td></tr><tr><td>Control frequency  $f_s$ </td><td>50 Hz</td></tr><tr><td>Speed signal</td><td>Joint-space speed</td></tr><tr><td>Smoothing</td><td>Savitzky–Golay filter</td></tr><tr><td>Filter window / polynomial order</td><td>11 / 3</td></tr><tr><td>Minimum valley spacing  $d_{\text{min}}$ </td><td>10 steps</td></tr><tr><td>Valley score</td><td>Absolute peak-height contrast</td></tr><tr><td>Threshold percentile</td><td>5th percentile from training demonstrations</td></tr><tr><td>Multi-arm fusion</td><td>Earliest accepted boundary across arms</td></tr></table>

The default speed signal is computed from arm joint commands and excludes gripper commands. This matches the policy action space and avoids forward kinematics during deployment.

## A.2 Online horizon selection

Algorithm 1 gives the test-time procedure at one policy query. Let B denote the set of controllable arm groups. In our bimanual RoboTwin2.0 experiments, $B ^ { ' } = \mathrm { \{ l e f t , r i g h t \} }$ . For each arm group, PACE scans low-speed valleys in temporal order and accepts the first valley whose peak-height contrast exceeds the task threshold. The final execution horizon is the earliest accepted boundary across arms. If no valid valley is found, the method falls back to full-chunk execution.

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1 PACE at a policy query
Require: Predicted action chunk  $A_{i}$ , arm groups B, maximum horizon  $H_{max}$ , threshold  $\delta_{T}$ 
Ensure: Execution horizon  $h_{i}$ 

1: for each arm group  $b \in B$  do
2: Compute speed profile  $v_{i}^{b}$  from  $A_{i}$ 
3: Smooth the profile:  $\tilde{v}_{i}^{b} \leftarrow S(\mathbf{v}_{i}^{b})$ 
4: Detect candidate valleys  $V_{i}^{b}$  with spacing  $d_{min}$ 
5: Set  $h_{i}^{b} \leftarrow H_{max}$ 
6: for valley  $r \in V_{i}^{b}$  in temporal order do
7: if  $r &gt; H_{max}$  then
8: break
9: end if
10: Compute peak-height contrast  $\Delta_{i}^{b}(r)$ 
11: if  $\Delta_{i}^{b}(r) \geq \delta_{T}$  then
12:  $h_{i}^{b} \leftarrow r$ 
13: break
14: end if
15: end for
16: end for
17: return  $h_{i} = \min\left(\{h_{i}^{b}: b \in B\} \cup \{H_{\max}, L\}\right)$
</div>

For a single-arm policy or a single-arm task, we simply set $\boldsymbol { B } = \left\{ \boldsymbol { b } \right\}$ , where b is the executed arm. No dummy second arm is introduced. The final horizon then reduces to

$$
h _ {i} = \min \{h _ {i} ^ {b}, H _ {\max}, L \}.\tag{8}
$$

Thus, the single-arm implementation is the same algorithm without the cross-arm minimum. If the single arm has no accepted valley, the fallback remains $h _ { i } = H _ { \operatorname* { m a x } }$

## A.3 Offline threshold calibration from demonstrations

The only task-specific quantity in PACE is the valley-acceptance threshold $\delta \tau$ . We calibrate this threshold once per task from training demonstrations and then keep it fixed for evaluation. This calibration does not use evaluation rollouts, success labels, or fixed-horizon sweeps.

For each task, we compute speed profiles from 50 training demonstrations using the same signal type, smoothing filter, and minimum-spacing parameter as the online controller. Candidate valleys divide each demonstration into motion segments. For a segment $s = [ u , v ]$ , we compute its peak-height contrast as

$$
\Delta (s) = \min \left\{\max _ {u \leq j \leq v} \tilde {v} _ {j} - \tilde {v} _ {u}, \max _ {u \leq j \leq v} \tilde {v} _ {j} - \tilde {v} _ {v} \right\}.\tag{9}
$$

We collect positive contrasts across training segments and set the task threshold to

$$
\delta_ {\mathcal {T}} = \text { Percentile } _ {5} \left(\{\Delta (s): s \in \text { training   segments   of   task } \mathcal {T} \}\right)\tag{10}
$$

The same calibration procedure is used for single-arm and multi-arm settings; the only difference is the arm set B used to construct speed profiles.

We also pool training demonstrations from all 50 RoboTwin2.0 tasks and apply the same calibration procedure to obtain a shared threshold (Appendix C.4).

## B Simulation protocols and results

## B.1 RoboTwin2.0

## B.1.1 Evaluation protocol

We evaluate all methods on the 50-task RoboTwin2.0 benchmark under the demo\_clean setting. Each task uses one trained $\pi _ { 0 . 5 }$ checkpoint with prediction horizon $L = 5 0$ . Within each task, all compared execution schemes share the same checkpoint, and only the test-time execution scheme is changed.

For each task, we fine-tune the pretrained $\pi _ { 0 . 5 }$ model on 50 demo\_clean demonstrations. All model parameters are updated using AdamW with a global batch size of 32 and a fixed training seed of 42. The learning rate is warmed up to $2 . 5 \times 1 0 ^ { - 5 }$ over 1,000 steps and then decayed to $2 . 5 \times 1 0 ^ { - 6 }$ following a cosine schedule. We evaluate the checkpoint saved at step 30,000, using exponential moving average weights with a decay of 0.99.

We compare PACE against fixed-horizon execution with $H \in \{ 5 , 2 5 , 5 0 \}$ , AAC, and AutoHorizon. The $H = 5 0$ baseline corresponds to full-chunk execution. For fixed-horizon execution, the robot executes the first H actions of each predicted chunk before re-querying the policy. For PACE, the executed prefix is selected by the rule in Appendix A. The AAC and AutoHorizon implementations are described in Appendix B.1.3. In all methods, the executed prefix is clamped by the returned chunk length.

For each task and method, we evaluate seeds 0, 1, 2. For each seed, we run three batches of 100 episodes. Thus, each task-method pair is evaluated over $3 \times 3 \times 1 0 0 = 9 0 0$ episodes. Success is measured using the RoboTwin2.0 task success predicate. The task-level success rate is averaged over the nine 100-episode runs, and the 50-task average is the equal-weight average over all task-level success rates.

## B.1.2 Full results and aggregate statistics

Table 7 reports success rates for PACE, fixed-horizon execution, AAC, and AutoHorizon on all 50 RoboTwin2.0 tasks. The best single global fixed-horizon baseline is $H = 2 5$ , with a 50-task average success rate of 57.8%. PACE improves the average success rate to 64.2%.

Table 7: Results on all 50 RoboTwin2.0 tasks. Success rates (%). Bold marks the highest mean in each row. The final rows report the mean and sample SD of the three seed-level overall success rates.

<table><tr><td>Task</td><td>H = 5</td><td>H = 25</td><td>H = 50</td><td>AutoHorizon</td><td>AAC</td><td>PACE</td></tr><tr><td>adjust_bottle</td><td>99.0</td><td>98.8</td><td>96.1</td><td>98.7</td><td>99.7</td><td>99.6</td></tr><tr><td>beat_block_hammer</td><td>81.1</td><td>67.7</td><td>66.6</td><td>71.3</td><td>64.8</td><td>80.7</td></tr><tr><td>blocks_ranking_rgb</td><td>7.3</td><td>50.0</td><td>36.1</td><td>66.1</td><td>52.4</td><td>61.4</td></tr><tr><td>blocks_ranking_size</td><td>23.9</td><td>18.6</td><td>9.2</td><td>25.9</td><td>20.2</td><td>32.9</td></tr><tr><td>click_alarmclock</td><td>92.4</td><td>65.3</td><td>65.9</td><td>71.4</td><td>86.7</td><td>87.6</td></tr><tr><td>click_bell</td><td>97.8</td><td>49.2</td><td>41.4</td><td>70.8</td><td>64.3</td><td>91.4</td></tr><tr><td>dump_bin_bigbin</td><td>58.7</td><td>85.1</td><td>84.3</td><td>88.1</td><td>89.1</td><td>93.0</td></tr><tr><td>grab_roller</td><td>89.9</td><td>96.2</td><td>96.2</td><td>96.9</td><td>97.7</td><td>98.6</td></tr><tr><td>handover_block</td><td>3.6</td><td>33.2</td><td>52.6</td><td>43.1</td><td>43.2</td><td>53.1</td></tr><tr><td>handover_mic</td><td>77.2</td><td>97.7</td><td>96.8</td><td>97.7</td><td>96.8</td><td>99.4</td></tr><tr><td>hanging_mug</td><td>16.2</td><td>15.8</td><td>16.7</td><td>20.1</td><td>23.6</td><td>21.8</td></tr><tr><td>lift_pot</td><td>50.9</td><td>61.2</td><td>62.0</td><td>62.2</td><td>63.1</td><td>65.4</td></tr><tr><td>move_can_pot</td><td>55.4</td><td>74.3</td><td>58.2</td><td>56.8</td><td>60.2</td><td>72.3</td></tr><tr><td>move_pillbottle_pad</td><td>34.8</td><td>49.9</td><td>29.1</td><td>44.3</td><td>46.6</td><td>45.0</td></tr><tr><td>move_playingcard_away</td><td>79.2</td><td>79.7</td><td>69.6</td><td>72.1</td><td>70.4</td><td>79.3</td></tr><tr><td>move_stapler_pad</td><td>12.6</td><td>11.4</td><td>7.1</td><td>13.6</td><td>10.6</td><td>18.6</td></tr><tr><td>open_laptop</td><td>84.9</td><td>84.8</td><td>88.8</td><td>89.6</td><td>89.3</td><td>88.0</td></tr><tr><td>open_microwave</td><td>33.3</td><td>26.0</td><td>60.3</td><td>55.4</td><td>52.4</td><td>45.0</td></tr><tr><td>pick_diverse_bottles</td><td>13.4</td><td>23.9</td><td>24.1</td><td>26.4</td><td>27.9</td><td>35.7</td></tr><tr><td>pick_dual_bottles</td><td>13.3</td><td>45.3</td><td>38.7</td><td>41.6</td><td>48.9</td><td>48.2</td></tr><tr><td>place_a2b_left</td><td>54.6</td><td>52.7</td><td>43.2</td><td>46.4</td><td>50.4</td><td>55.9</td></tr><tr><td>place_a2b_right</td><td>20.0</td><td>47.8</td><td>36.8</td><td>43.1</td><td>46.7</td><td>54.6</td></tr><tr><td>place_bread_basket</td><td>55.3</td><td>53.9</td><td>30.3</td><td>42.6</td><td>46.2</td><td>49.1</td></tr><tr><td>place_bread_skillet</td><td>38.4</td><td>39.8</td><td>29.0</td><td>33.2</td><td>42.0</td><td>37.0</td></tr><tr><td>place_burger_fries</td><td>88.9</td><td>89.3</td><td>85.8</td><td>87.1</td><td>91.9</td><td>91.3</td></tr><tr><td>place_can_basket</td><td>18.1</td><td>56.8</td><td>38.2</td><td>52.1</td><td>66.3</td><td>53.6</td></tr><tr><td>place_cans_plasticbox</td><td>85.1</td><td>83.0</td><td>81.8</td><td>85.2</td><td>85.8</td><td>90.6</td></tr><tr><td>place_container_plate</td><td>88.3</td><td>81.7</td><td>91.6</td><td>91.2</td><td>89.1</td><td>96.1</td></tr><tr><td>place_dual_shoes</td><td>19.9</td><td>40.6</td><td>33.7</td><td>31.0</td><td>40.3</td><td>37.3</td></tr><tr><td>place_empty_cup</td><td>59.4</td><td>48.1</td><td>45.8</td><td>51.8</td><td>45.9</td><td>58.8</td></tr><tr><td>place_fan</td><td>35.3</td><td>34.0</td><td>25.8</td><td>31.4</td><td>35.7</td><td>39.1</td></tr><tr><td>place_mouse_pad</td><td>4.3</td><td>14.1</td><td>12.8</td><td>16.3</td><td>15.1</td><td>17.8</td></tr><tr><td>place_object_basket</td><td>35.9</td><td>59.4</td><td>58.0</td><td>67.8</td><td>64.8</td><td>70.2</td></tr><tr><td>place_object_scale</td><td>23.8</td><td>29.4</td><td>18.9</td><td>23.8</td><td>24.8</td><td>40.7</td></tr><tr><td>place_object_stand</td><td>64.2</td><td>57.3</td><td>53.4</td><td>56.6</td><td>54.9</td><td>59.2</td></tr><tr><td>place_phone_stand</td><td>52.9</td><td>59.9</td><td>54.7</td><td>54.7</td><td>55.4</td><td>58.7</td></tr><tr><td>place_shoe</td><td>43.9</td><td>64.4</td><td>56.1</td><td>58.6</td><td>63.0</td><td>62.8</td></tr><tr><td>press_stapler</td><td>68.2</td><td>60.6</td><td>56.0</td><td>59.8</td><td>61.6</td><td>61.8</td></tr><tr><td>put_bottles_dustbin</td><td>0.3</td><td>79.3</td><td>70.8</td><td>74.3</td><td>76.6</td><td>88.9</td></tr><tr><td>put_object_cabinet</td><td>7.6</td><td>51.9</td><td>50.6</td><td>50.6</td><td>48.7</td><td>57.4</td></tr><tr><td>rotate_qrcode</td><td>48.9</td><td>69.9</td><td>66.1</td><td>70.2</td><td>68.4</td><td>71.7</td></tr><tr><td>scan_object</td><td>20.9</td><td>43.1</td><td>28.9</td><td>39.7</td><td>39.9</td><td>40.3</td></tr><tr><td>shake_bottle</td><td>95.2</td><td>95.7</td><td>96.7</td><td>96.7</td><td>96.2</td><td>98.4</td></tr><tr><td>shake_bottle_horizontally</td><td>97.3</td><td>96.8</td><td>96.9</td><td>97.1</td><td>98.6</td><td>98.8</td></tr><tr><td>stack_blocks_three</td><td>39.7</td><td>43.2</td><td>28.7</td><td>54.6</td><td>39.1</td><td>55.2</td></tr><tr><td>stack_blocks_two</td><td>12.4</td><td>80.1</td><td>63.9</td><td>82.9</td><td>77.3</td><td>82.3</td></tr><tr><td>stack_bowls_three</td><td>45.0</td><td>61.0</td><td>67.6</td><td>70.8</td><td>64.0</td><td>79.8</td></tr><tr><td>stack_bowls_two</td><td>88.7</td><td>88.4</td><td>85.1</td><td>91.0</td><td>88.2</td><td>95.6</td></tr><tr><td>stamp_seal</td><td>67.4</td><td>34.2</td><td>24.3</td><td>37.0</td><td>32.1</td><td>47.8</td></tr><tr><td>turn_switch</td><td>34.4</td><td>39.4</td><td>37.1</td><td>41.8</td><td>36.7</td><td>42.6</td></tr><tr><td>Mean</td><td>48.79</td><td>57.80</td><td>53.36</td><td>59.03</td><td>59.07</td><td>64.20</td></tr><tr><td>Seed SD</td><td>0.89</td><td>1.49</td><td>0.44</td><td>0.53</td><td>0.56</td><td>0.88</td></tr></table>

For each evaluation seed, we average the three batches within each task and then average equally over all 50 tasks. The final two rows of Table 7 summarize the resulting three overall success rates. Each task uses the same checkpoint across seeds, so the SD reflects evaluation variability.

PACE achieves a higher mean success rate than AutoHorizon on 46 of the 50 tasks and than AAC on 39 of the 50 tasks.

Using unrounded success rates, PACE improves overall success by 6.40 percentage points over fixed $H = 2 5$ and by 5.13 points over AAC. Paired bootstraps over the 50 task-level mean differences yield 95% confidence intervals of [+4.21, +8.86] and $[ + 3 . { \bar { 2 } } 2 , + 7 . 1 0 ]$ , respectively. Against AutoHorizon, PACE gains 5.18 percentage points, with a 95% confidence interval of $[ + 3 . { \dot { 6 } } 2 , + \bar { 6 } . 7 8 ]$ from a paired hierarchical bootstrap. These confidence intervals describe uncertainty in the gains over each baseline, while the SD in Table 7 describes variability across evaluation seeds.

## B.1.3 Adaptive baseline implementations

Shared experimental setting. AutoHorizon and AAC use the same task-specific $\pi _ { 0 . 5 }$ checkpoints as the fixed-horizon baselines and PACE, without additional policy training. Both use ten flowsampling steps to predict 50-step chunks of decoded 14-D actions, with six absolute joint-position commands and one gripper command per arm. They follow the evaluation protocol in Appendix B.1.1. For each baseline, the three batches within a task and evaluation seed reuse the same set of 100 evaluation scenes. Their selector settings are fixed across tasks and are not tuned using evaluation success labels.

AutoHorizon. We implement AutoHorizon [22] in JAX following the official implementation. The selector uses model-internal action-token self-attention from denoising step 3 and applies the bidirectional soft-pointer rule with $\tau = 0 . 3 .$ , entropy quantile 0.9, and run length 1. The robot executes the selected prefix of the decoded action chunk, discards the remaining actions, and queries the policy again using a new observation.

AAC adapted to bimanual joint-space actions. We adapt AAC [21] to RoboTwin’s bimanual joint-space actions and sample 20 candidate chunks per policy query. At each prediction step, we estimate Gaussian entropy from the six normalized joint commands of each arm across the candidates, with covariance regularization $1 0 ^ { - 6 } I .$ . For each gripper, we binarize its decoded commands at 0.5 and compute Bernoulli entropy.

Movement is measured from the decoded absolute joint targets relative to the current state, summing the arm-wise displacements and weighting gripper toggles by 0.2. We use the movement threshold of 3.0 reported in the AAC supplementary material. The final execution horizon is the larger of the horizon selected by AAC’s prefix-entropy turning-point rule and the shortest prefix satisfying the movement constraint.

To generate AAC’s 20 candidate chunks, our $\pi _ { 0 . 5 }$ implementation encodes the image–language–state prefix once per query and reuses its KV cache across candidates. The action head processes candidate 0 separately and the remaining 19 candidates as a batch. This grouping preserves candidate 0’s agreement with standard $\pi _ { 0 . 5 }$ sampling. The robot executes the selected prefix of candidate 0.

## B.2 GR00T N1.7 on LIBERO

Policy and evaluation protocol. We evaluate PACE using the official NVIDIA/LeRobot GR00T N1.7 checkpoints [23] on LIBERO-Spatial, LIBERO-Object, LIBERO-Goal, and LIBERO-10 [24]. The four suites contain 10 tasks each, for 40 tasks in total. Within each suite, all execution strategies use the same checkpoint without retraining or modifying the base policy. We evaluate each task on 50 initial states for each of three evaluation seeds, yielding 150 episodes per task and 6,000 episodes per strategy.

For each seed, Overall is the equal-weight mean success rate across all 40 tasks. We report the mean and sample standard deviation across the three seeds for each suite and for Overall.

Adapting PACE. GR00T N1.7 predicts 16-step Cartesian-delta chunks, whereas the $\pi _ { 0 . 5 }$ policies evaluated on RoboTwin2.0 predict 50-step joint-space chunks. PACE constructs the motion profile from the norm of the predicted translational action deltas and applies the same valley-based horizonselection rule. PACE calibrates its acceptance thresholds from LIBERO training demonstrations using valley contrasts normalized by peak speed.

We compare PACE with fixed horizons $H = 2 , 4 , 8 ,$ , 16 and AutoHorizon under the same evaluation protocol. The $H = 1 6$ baseline executes the full predicted chunk. AutoHorizon follows its published horizon-selection rule adapted to GR00T N1.7.

Table 8: GR00T N1.7 results on LIBERO. Success rates (%), reported as mean ± sample SD across three evaluation seeds. Bold marks the highest mean in each column.

<table><tr><td>Execution strategy</td><td>Spatial</td><td>Object</td><td>Goal</td><td>LIBERO-10</td><td>Overall</td></tr><tr><td>H = 2</td><td>97.20 ± 0.92</td><td>91.47 ± 0.50</td><td>85.87 ± 1.17</td><td>77.73 ± 1.33</td><td>88.07 ± 0.28</td></tr><tr><td>H = 4</td><td>97.67 ± 0.31</td><td>94.33 ± 0.90</td><td>93.13 ± 1.62</td><td>83.33 ± 1.92</td><td>92.12 ± 0.30</td></tr><tr><td>H = 8</td><td>96.87 ± 0.90</td><td>96.20 ± 0.72</td><td>96.00 ± 0.53</td><td>88.00 ± 0.87</td><td>94.27 ± 0.37</td></tr><tr><td>H = 16</td><td>96.27 ± 0.99</td><td>97.00 ± 0.92</td><td>96.53 ± 1.01</td><td>88.07 ± 0.31</td><td>94.47 ± 0.23</td></tr><tr><td>AutoHorizon</td><td>96.13 ± 0.42</td><td>96.67 ± 0.31</td><td>95.80 ± 0.80</td><td>91.07 ± 1.45</td><td>94.92 ± 0.44</td></tr><tr><td>PACE</td><td>97.00 ± 0.72</td><td>97.73 ± 0.23</td><td>96.93 ± 0.90</td><td>89.67 ± 1.17</td><td>95.33 ± 0.18</td></tr></table>

Results and interpretation. PACE achieves the highest overall success rate among the evaluated strategies (Table 8). Using unrounded rates, it exceeds $H = 1 6 ,$ the strongest global fixed-horizon baseline, by 0.87 percentage points. A paired hierarchical bootstrap stratified by suite gives a 95% confidence interval of $[ + 0 . \mathrm { \bar { 1 0 , + 1 } } . 7 0 ]$ percentage points for this gain. Its overall success rate is also 0.42 percentage points higher than AutoHorizon. Across individual suites, PACE leads on Object and Goal, fixed $H = 4$ on Spatial, and AutoHorizon on LIBERO-10.

Policy-query and inference overhead. We measure inference latency on a dedicated NVIDIA H20 using the LIBERO setup described above. Across the four suite-specific GR00T N1.7 checkpoints, model-forward latency ranges from 127.55 to 130.06 ms per query. PACE’s horizon selector takes $0 . 0 5 6 6 \pm 0 . 0 0 4 6$ ms per query, approximately 0.044% of the time required for a policy forward pass. The total selector time per successful episode is $0 . 5 9 5 \pm 0 . 0 0 1$ ms.

Table 9 reports policy queries and total inference time per successful episode for each execution strategy. These metrics are computed separately for each evaluation seed using that strategy’s successful episodes. We report the mean and sample standard deviation across the three seeds. The reported times cover inference computation only.

Table 9: Policy queries and inference time with GR00T N1.7 on LIBERO. Total inference time includes policy forward passes and, for PACE, horizon selection.

<table><tr><td>Execution strategy</td><td>Queries per successful episode</td><td>Total inference time per successful episode (s)</td></tr><tr><td>H = 2</td><td>78.21 ± 0.43</td><td>10.075 ± 0.056</td></tr><tr><td>H = 4</td><td>38.49 ± 0.10</td><td>4.959 ± 0.013</td></tr><tr><td>H = 8</td><td>19.05 ± 0.08</td><td>2.455 ± 0.010</td></tr><tr><td>H = 16</td><td>9.69 ± 0.01</td><td>1.248 ± 0.002</td></tr><tr><td>PACE</td><td>10.53 ± 0.02</td><td>1.357 ± 0.003</td></tr></table>

Compared with full-chunk execution $( H = 1 6 )$ , PACE improves success by 0.87 percentage points while increasing total inference time per successful episode by 8.66%. This increase comes almost entirely from the 0.83 additional policy queries per successful episode, with horizon selection contributing little. All differences are computed from unrounded measurements.

## C Ablation and robustness studies

The phase-profile and valley-selection studies follow the full RoboTwin2.0 protocol in Appendix B.1.1: 50 tasks and 900 episodes per task across three seeds. Average horizon denotes the mean planned execution length. The training-horizon study requires retraining and uses six representative tasks, seed 0, and three 100-episode batches per setting.

## C.1 Phase-profile construction

We compare Cartesian and joint-space speed, with and without Savitzky–Golay smoothing, while keeping the valley-selection rule and calibration percentile fixed. The Cartesian variant computes end-effector linear speed using forward kinematics; the raw variants detect valleys without smoothing. Table 2 (left) reports the results. Smoothing improves both representations, consistent with suppressing spurious short-range valleys. The smoothed Cartesian variant has a slight success-rate advantage, while the default smoothed joint-space variant produces longer execution horizons and avoids forward kinematics at test time.

## C.2 Training prediction horizon

We study how the policy prediction horizon used during training affects fixed-horizon execution at test time. Here $H _ { \mathrm { t r a i n } }$ denotes the supervised action-chunk length during policy training, while $H _ { \mathrm { e v a l } }$ denotes the fixed executed prefix length at evaluation. Entries with $H _ { \mathrm { t r a i n } } < H _ { \mathrm { e v a l } }$ are infeasible because the policy does not predict enough actions to execute the requested prefix. We sweep both axes on a finer grid, from 5 to 50 in steps of 5, using six representative tasks, seed $0 ,$ and three 100-episode batches per setting.

![](images/353af2ee6b6425ce2a2dfbf31f4af8c05306ce6bb7758d1d06e15c7d5e11e13a.jpg)  
Figure 7: Training prediction horizon ablation. Each cell shows the success gain of training horizon $H _ { \mathrm { t r a i n } }$ (columns) relative to the shortest feasible training horizon for a given evaluation horizon $H _ { \mathrm { e v a l } }$ (rows), i.e., $H _ { \mathrm { t r a i n } } { = } H _ { \mathrm { e v a l } }$ . Blue indicates a gain, orange a loss; the diagonal is zero by construction. Cells below the diagonal are infeasible and left blank.

Figure 7 shows that, for $H _ { \mathrm { e v a l } } \geq 1 0 .$ , increasing $H _ { \mathrm { t r a i n } }$ generally improves success, with gains reaching $+ 2 0 . 7 , + 1 3 . 0 .$ , and $+ 6 . 6$ points at $H _ { \mathrm { e v a l } } = 1 0 , 1 5 , 2 0$ , respectively. Very short execution is an exception: at $H _ { \mathrm { e v a l } } = 5$ , most gains lie between −4.7 and 0. For $\dot { H _ { \mathrm { e v a l } } } > 2 0$ , gains remain below $+ 7$ points and become more variable, including near-zero and slightly negative values. Longer training horizons thus benefit short but non-trivial execution prefixes most, motivating the default $H _ { \mathrm { t r a i n } } = 5 0$

## C.3 Valley-selection sensitivity

We vary the minimum spacing between candidate valleys and the calibration percentile $\rho$ used to set the contrast threshold $\delta ,$ keeping all other settings fixed. Larger ρ produces a stricter threshold, which accepts fewer valleys and therefore tends to increase the selected execution horizon.

Table 10: Sensitivity to valley-selection parameters. All results use the smoothed joint-space phase profile and are averaged over all 50 RoboTwin2.0 tasks.

<table><tr><td>Min. spacing</td><td>Percentile ρ</td><td>Succ.</td><td>Avg. Horizon</td></tr><tr><td>5</td><td>0</td><td>64.3</td><td>18.1</td></tr><tr><td>5</td><td>5</td><td>63.5</td><td>23.4</td></tr><tr><td>5</td><td>10</td><td>63.3</td><td>24.6</td></tr><tr><td>5</td><td>25</td><td>62.5</td><td>31.9</td></tr><tr><td>10</td><td>0</td><td>64.0</td><td>19.8</td></tr><tr><td>10</td><td> $5^†$ </td><td>64.2</td><td>24.3</td></tr><tr><td>10</td><td>10</td><td>63.0</td><td>25.5</td></tr><tr><td>10</td><td>25</td><td>62.0</td><td>32.9</td></tr></table>

PACE is not sensitive to a single narrowly tuned setting. Several nearby configurations achieve similar success rates. Very strict thresholds, such as $\rho = 2 5$ , select substantially longer execution windows and reduce success, suggesting that rejecting too many phase valleys delays feedback in contact-sensitive stages. We use minimum spacing 10 and $\rho = 5$ as the default because it provides high success while keeping the average execution horizon moderate.

## C.4 Shared versus per-task threshold calibration

To evaluate whether PACE can use one valley-acceptance threshold across tasks, we compare pertask and shared-threshold calibration. The default setting calibrates a separate threshold from each task’s training demonstrations. For the shared setting, we pool training demonstrations from all 50 RoboTwin2.0 tasks and apply the same calibration procedure (Appendix A.3), retaining the speed-profile construction and 5th-percentile rule. The resulting threshold is used unchanged across all 50 tasks. All compared execution strategies use the same task-specific $\pi _ { 0 . 5 }$ checkpoints, and all other PACE settings and evaluation conditions remain unchanged.

Following the evaluation protocol in Appendix B.1.1, we compute the equal-weight mean success rate over all 50 tasks for each of three evaluation seeds. Table 3 reports the mean and sample standard deviation of these three averages.

Shared-threshold PACE improves success over fixed H = 25 by 4.64 percentage points, with a 95% confidence interval of $[ + 2 . 6 1 , + 6 . 9 9 ]$ . Per-task calibration provides an additional gain of approximately 1.8 points over the shared threshold, with a 95% confidence interval of $[ + 1 . 1 8 , + 2 . 3 5 ]$ We compute each interval with a paired bootstrap over the 50 task-level differences using unrounded mean success rates. These results show that a single shared threshold preserves most of PACE’s improvement over fixed $H = 2 5$

## C.5 Calibration stability and trajectory sensitivity

Data and sampling. To assess the stability of PACE’s horizon decisions, we conduct an offline analysis of trajectories from DROID [26], AgiBot World Beta [27], RoboMIND [28], and Robo-COIN [29]. We use stratified sampling by task and robot configuration, covering 1,384 eligible strata across 31 dataset-defined robot configurations. For each stratum, we use 50 episodes for calibration and up to 50 held-out episodes for evaluation. The resulting sample contains 135,046 distinct episodes and 263,319 fixed 50-step evaluation windows.

Calibration resampling. For each stratum, we first compute a reference valley-acceptance threshold from its 50 calibration demonstrations using the fifth-percentile rule. We then resample these demonstrations with replacement 500 times and recompute the threshold for each sample. On the same held-out evaluation windows, we compare the horizons selected by each recalibrated threshold with those selected by the reference threshold.

Trajectory perturbation. To assess sensitivity to trajectory fluctuations, we double the highfrequency residual in the joint-position action sequences. This residual is the difference between each sequence and its trend obtained with a five-step, second-order Savitzky–Golay filter. We perturb both the calibration trajectories and the held-out evaluation windows, and recalibrate the threshold on the perturbed demonstrations. On the perturbed evaluation windows, we measure horizon agreement within five control steps of the original decisions.

Table 11: Offline stability of PACE’s horizon decisions. For each metric, we average equally over task/configuration strata within each dataset family and then equally across the four families.

<table><tr><td>Test</td><td>Metric</td><td>Result</td></tr><tr><td rowspan="4">Calibration resampling</td><td>Threshold coefficient of variation (dataset-family-balanced)</td><td>18.7%</td></tr><tr><td>Identical horizon</td><td>99.08%</td></tr><tr><td>Absolute horizon shift &gt; 5 steps</td><td>0.78%</td></tr><tr><td>Mean absolute horizon shift</td><td>0.14 steps</td></tr><tr><td>High-frequency residual doubled</td><td>Absolute horizon shift ≤ 5 steps</td><td>80.6%</td></tr></table>

Table 11 reports horizon consistency under calibration resampling and trajectory perturbation. On a fixed evaluation window, threshold changes leave the horizon unchanged whenever the same earliest valley remains accepted or all candidates remain rejected

## D Execution-horizon analysis

## D.1 Fixed-horizon sweeps

For each of the six representative tasks in Sec. 4.4, we evaluate every integer fixed horizon $H \in$ $\{ 1 , \ldots , 5 0 \}$ . Each fixed-horizon setting uses seed 0 and three 100-episode batches, giving 300 episodes per horizon. PACE uses three seeds with three 100-episode batches per seed, giving 900 episodes per task. Its mean execution horizon is averaged over all selected execution windows. The best swept horizon $H ^ { \star }$ is selected using test-task outcomes and serves as a diagnostic reference within fixed-horizon schedules, not a deployable selection rule.

Table 12: Diagnostic fixed-horizon sweep summary. $H ^ { \star }$ denotes the best fixed horizon in the sweep, with ties listed together. All success rates are percentages.

<table><tr><td>Task</td><td> $H^{\star}$ </td><td>Fixed at  $H^{\star}$ </td><td>PACE</td></tr><tr><td>Pick Dual Bottles</td><td>14</td><td>56.7</td><td>48.2</td></tr><tr><td>Grab Roller</td><td>7, 9, 32</td><td>97.7</td><td>98.6</td></tr><tr><td>Place A2B Right</td><td>11</td><td>53.3</td><td>54.6</td></tr><tr><td>Dump Bin Bigbin</td><td>37</td><td>90.3</td><td>93.0</td></tr><tr><td>Put Bottles Dustbin</td><td>18</td><td>88.7</td><td>88.9</td></tr><tr><td>Stack Bowls Two</td><td>8</td><td>95.3</td><td>95.6</td></tr></table>

Table 12 shows substantial task dependence in $H ^ { \star }$ , ranging from short horizons such as $7$ or 8 to 37. Grab Roller has three tied best horizons, illustrating the non-monotonic effect of execution length.

## D.2 Matched execution horizons

We obtain the matched-horizon results in Table 4 from the sweep by selecting the fixed horizon nearest to PACE’s mean execution horizon on each task. PACE achieves a higher observed success rate on all six tasks, with gains of 1.6–10.6 percentage points computed from unrounded values. These results suggest that matching the mean execution horizon alone is insufficient to reproduce PACE’s performance.

Pick Dual Bottles illustrates the difference between the two fixed-horizon comparisons. PACE achieves 48.2% success, exceeding the matched fixed horizon $H = 2 6 ( 4 6 . 7 \% )$ but remaining below the best swept horizon $H ^ { \star } = 1 4 ( \bar { 5 } 6 . 7 \% )$ .

In the seed-0 evaluation, PACE achieved 43.7% success, compared with 46.0% for fixed $H = 2 5$ with 300 episodes per method. Across all three seeds (900 episodes per method), the corresponding success rates are 48.2% and 45.3% (Appendix B), reversing the seed-0 ordering.

## D.3 Phase alignment

To examine whether PACE replans near motion-phase transitions, we compare its executed valleytriggered boundaries with independently annotated transitions. The analysis covers 89 successful rollouts from three RoboTwin2.0 tasks: Place A2B Right, Place Phone Stand, and Pick Dual Bottles.

Blind annotation and matching. Annotators identified motion-phase transitions from paired head camera and external-observer videos without access to PACE’s replanning boundaries or boundary overlays. After the annotations were finalized, we extracted the executed valley-triggered boundaries from the rollout logs. We matched boundaries one-to-one with annotated transitions within ±5 control steps (±0.1 s at 50 Hz). We chose the matching with the largest number of pairs, breaking ties by the smallest total temporal distance. The one-to-one constraint prevents multiple boundaries from being matched to the same transition.

Of the 391 executed replanning boundaries, six were $H _ { \mathrm { m a x } }$ fallbacks and were counted separately. The primary alignment metric uses the remaining 385 valley-triggered boundaries.

Metrics and temporal controls. Alignment precision is the fraction of executed valley-triggered boundaries matched to annotated transitions. Transition coverage is the fraction of annotated transitions matched to a boundary. For each task, we obtain 95% confidence intervals for PACE precision by bootstrapping the audited successful rollouts at the trajectory level.

To assess whether the observed alignment exceeds chance levels, we compare PACE with two offline controls that retain the same number of boundaries. The first samples boundary times uniformly at random with a minimum spacing of 10 control steps. The second circularly shifts PACE’s boundaries while preserving their relative spacing. Each control is evaluated over 10,000 randomizations.

Table 13: Phase alignment on successful rollouts. The last three columns report alignment precision (%). Overall is computed from pooled counts across tasks.

<table><tr><td>Task</td><td>Annotated transitions</td><td>Aligned / valleys</td><td>PACE precision (95% CI)</td><td>Uniform random</td><td>Circular shift</td></tr><tr><td>Place A2B Right</td><td>182</td><td>135 / 172</td><td>78.5 [71.1, 85.1]</td><td>45.9</td><td>46.1</td></tr><tr><td>Place Phone Stand</td><td>145</td><td>96 / 123</td><td>78.0 [72.1, 84.0]</td><td>43.7</td><td>43.8</td></tr><tr><td>Pick Dual Bottles</td><td>120</td><td>76 / 90</td><td>84.4 [78.5, 90.2]</td><td>41.9</td><td>42.3</td></tr><tr><td>Overall</td><td>447</td><td>307 / 385</td><td>79.7</td><td>-</td><td>-</td></tr></table>

Results. Table 13 shows that 307 of 385 valley-triggered boundaries align with annotated motionphase transitions, yielding 79.7% precision. These matches cover 68.7% of the 447 annotated transitions. Matched boundaries lie an average of 2.3 control steps from their corresponding transitions. PACE achieves higher alignment precision than both the uniform-random and circular-shift controls on all three tasks, with $p \leq 0 . 0 0 0 1$ in all six randomization tests.

Interpretation. The observed alignment on successful rollouts from these three tasks supports PACE’s use of prominent low-speed valleys as cues for candidate replanning boundaries. Assessing the effect of individual boundary choices on task success would require controlled interventions on boundary timing.

## E Real-robot experiment details

## E.1 Tasks, scenes, and scoring rules

Within each task, all execution strategies use the same fine-tuned $\pi _ { 0 . 5 }$ checkpoint and differ only in the test-time execution rule. Table 14 summarizes the robots, demonstrations, and trial counts.

Table 14: Real-robot task and data summary. Demos denotes the number of demonstration episodes used for fine-tuning

<table><tr><td>Task</td><td>Robot</td><td>Demos</td><td>Trials / method</td></tr><tr><td>stack_bowls</td><td>ALOHA</td><td>1,047</td><td>30</td></tr><tr><td>put_pen_into_pencil_case</td><td>ALOHA</td><td>1,220</td><td>30</td></tr><tr><td>place_object_on_plate</td><td>Franka</td><td>301</td><td>5 × 20</td></tr></table>

RoboChallenge. We evaluate stack\_bowls and put\_pen\_into\_pencil\_case on RoboChallenge Table30 V1, following its task prompts, reset procedures, and grading rules. Figure 8 shows the initial scenes. We report the official partial-credit Score and full-completion rate Succ., using the scoring steps below.

![](images/2968080713773ba094e03865ca74e92c882279a6c14c28fe9947ee0dcff00a8a.jpg)  
put\_pen\_into\_pencil\_case

![](images/9f717d27849729ac981a61c4e2e962620a6ac80f666c02a3e16dbf5c18dda914.jpg)  
stack\_bowls  
Figure 8: Initial frames of the RoboChallenge tasks. Left: put\_pen\_into\_pencil\_case. Right: stack\_bowls. Both tasks use an ALOHA robot.

stack\_bowls.

• The task is to nest the bowls from largest to smallest.

• 2.0 × 2 points: pick up a bowl with the gripper.

• 2.5 × 2 points: successfully stack the bowls.

• 1.0 point: retract the gripper to its original position.

put\_pen\_into\_pencil\_case. Note: The official RoboChallenge description labels the pencilcase opening step with the opposite arm; in the evaluation setup and rollout videos, the left gripper opens the pencil case while the right arm manipulates the pen. We therefore report the scoring steps using the observed arm assignment.

• The task is to place the pen on the table into the pencil case.

• 2.0 points: left gripper reaches and opens the pencil case.

• 2.0 points: the pencil case is sufficiently open for insertion.

• 2.0 points: right arm picks up the pen.

• 3.0 points: right arm places the pen into the pencil case.

• 1.0 point: reset both arms.

Franka place\_object\_on\_plate. The robot picks up one of five objects—corn, cabbage, green pepper, red pepper, and garlic—and places it onto a fixed plate (Fig. 9). A single fine-tuned checkpoint is shared across all five object variants. A trial succeeds only if the object has been released and remains fully inside the plate at the end of the trial; otherwise it fails. Successful trials receive 1 point

Object and plate layout

and failures 0, so Score equals the success rate in percentage points. Table 15 reports the per-object outcomes.

![](images/f7896e66a08a6fb66dba1881eade3354dc473b5ec4b3181ab9f86828458dba39.jpg)

![](images/97e62b548d56ff1999639b9c71f92d11fdbb231e0c04b2e9748eec49dd13a0ba.jpg)  
Franka tabletop setup  
Figure 9: In-lab place\_object\_on\_plate setup. The robot places one of five objects onto a fixed target plate.

Table 15: Per-object results for place\_object\_on\_plate. Each object variant is evaluated over 20 trials per method.

<table><tr><td>Object</td><td>Baseline</td><td>PACE</td><td> $\Delta$ </td></tr><tr><td>Corn</td><td>16 / 20 (80.0%)</td><td>18 / 20 (90.0%)</td><td>+10.0</td></tr><tr><td>Cabbage</td><td>15 / 20 (75.0%)</td><td>17 / 20 (85.0%)</td><td>+10.0</td></tr><tr><td>Green pepper</td><td>13 / 20 (65.0%)</td><td>18 / 20 (90.0%)</td><td>+25.0</td></tr><tr><td>Red pepper</td><td>14 / 20 (70.0%)</td><td>17 / 20 (85.0%)</td><td>+15.0</td></tr><tr><td>Garlic</td><td>14 / 20 (70.0%)</td><td>18 / 20 (90.0%)</td><td>+20.0</td></tr><tr><td>Total</td><td>72 / 100 (72.0%)</td><td>88 / 100 (88.0%)</td><td>+16.0</td></tr></table>

## E.2 Fine-tuning configuration

For each RoboChallenge task, we fine-tune a task-specific $\pi _ { 0 . 5 }$ policy initialized from the PI0.5 base checkpoint. Training is performed on four H200 GPUs with FSDP and bfloat16 precision. We use AdamW with a global batch size of 256, global gradient clipping at norm 1.0, EMA decay 0.99, and eight data-loader workers. The learning rate follows a cosine decay schedule with 1000 warmup steps, peak learning rate $2 . 5 \times 1 0 ^ { - 5 }$ , decay horizon 30,000 steps, and final learning rate $2 . 5 \times 1 0 ^ { - 6 }$ For all reported RoboChallenge trials, we deploy the 30,000-step checkpoint.

For place\_object\_on\_plate, we fine-tune a single $\pi _ { 0 . 5 }$ checkpoint over all five object variants using 301 demonstrations. The training recipe follows the RoboChallenge setting except that we use a batch size of 32 and train on one RTX A6000 Pro GPU. We also deploy the 30,000-step checkpoint for all in-lab trials.

## E.3 Evaluation and fixed-horizon comparisons

On RoboChallenge, we compare PACE with full-chunk execution $( H = 5 0 )$ . On Franka, we additionally evaluate fixed $H = 5$ and $H = 2 5$ . PACE selects the executed prefix using the default configuration in Appendix A, without task-specific horizon tuning during evaluation.

Table 16 reports success rates and policy queries per successful episode on Franka.

Table 16: Comparison with fixed-horizon execution on Franka. Query counts are averaged over each strategy’s successful episodes.

<table><tr><td>Execution</td><td>Successes / trials</td><td>Fixed H / PACE mean H</td><td>Queries / successful episode</td></tr><tr><td>Fixed H = 5</td><td>87 / 100</td><td>5</td><td>36.22</td></tr><tr><td>Fixed H = 25</td><td>82 / 100</td><td>25</td><td>6.36</td></tr><tr><td>Fixed H = 50</td><td>72 / 100</td><td>50</td><td>3.14</td></tr><tr><td>PACE</td><td>88 / 100</td><td>28.64</td><td>5.57</td></tr></table>

PACE achieves a success rate comparable to fixed H = 5 (88% versus 87%), while using 84.6% fewer policy queries per successful episode. Relative to fixed H = 25, PACE improves success by 6 percentage points and reduces policy queries per successful episode by 12.4%. Fixed $H = 5 0$ requires the fewest queries per successful episode, but its success rate is lower at 72%.

## F Additional PACE rollout visualizations

These additional RoboTwin2.0 and RoboChallenge rollouts show the policy-query timesteps and execution horizons selected by PACE. Each caption summarizes how the horizon changes across the task’s motion phases.

## F.1 RoboTwin2.0 rollouts

![](images/9d3db1284c34a04aa6da184027f22242038b10c2113c1c874b41b31eff5501f0.jpg)  
Figure 10: PACE rollout visualization on scan\_object. PACE selects execution horizons from the predicted kinematic profile, allowing the rollout to keep longer open-loop segments when motion is smooth and to replan earlier around phase transitions.

![](images/a38bd2ca43b51941965ea1be5423fd4b7d4117c9080accc59a465bf9030b4fe5.jpg)  
Figure 11: PACE rollout visualization on place\_can\_basket. The selected execution horizon adapts to the local phase structure of the rollout, with shorter prefixes near the placement transition and longer prefixes during smooth motion segments.

![](images/591df8a0445f201e837fab05af6539a4b24d1728ac888a340045727dcb6f92af.jpg)

![](images/273791a1ec720077f3597179f2cf0090dd3aaf4fc40fe702a9ca27d4f843f2e9.jpg)  
Figure 12: PACE rollout visualization on hanging\_mug. The execution horizon is shortened around the task transition that requires precise alignment and interaction, and lengthened during more stable motion segments.

## F.2 RoboChallenge rollout

![](images/b3846e8e2211d7d497c43f9b78a13c2a7147c16f10eb5955e8ae3319c3848b36.jpg)  
Figure 13: PACE rollout visualization on RoboChallenge stack\_bowls. The selected execution horizon is longer during smooth approach and transport, and shorter near the stacking phase, reflecting the same phase-aware replanning rule used throughout the paper.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

Answer: [Yes].

Justification: The abstract and Section 1 state the main claims: execution horizon is a critical test-time variable, fixed horizons are task-dependent and non-monotonic, and PACE improves simulation and real-robot execution without retraining. These claims are supported by the simulation results in Section 4.1, real-robot results in Section 5.1, and diagnostic analyses in Sections 4.2 and 4.4.

Guidelines:

• The answer [N/A] means that the abstract and introduction do not include the claims made in the paper.

• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A [No] or [N/A] answer to this question will not be perceived well by the reviewers.

• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.

• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes].

Justification: Sec. 6 discusses PACE’s dependence on the base policy, reliance on low-speed valley cues, and need for demonstration-based calibration. It also addresses the scope of the empirical evaluation and the inference cost of additional policy queries.

Guidelines:

• The answer [N/A] means that the paper has no limitation while the answer [No] means that the paper has limitations, but those are not discussed in the paper.

• The authors are encouraged to create a separate “Limitations” section in their paper.

• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.

• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.

• The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.

• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.

• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.

• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren’t acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [N/A].

Justification: The paper does not present theoretical results or formal proofs. The equations in Sections 3 and A define the action-chunking setup, speed profiles, and deterministic PACE selection rule.

Guidelines:

• The answer [N/A] means that the paper does not include theoretical results.

• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.

• All assumptions should be clearly stated or referenced in the statement of any theorems.

• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.

• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.

• Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes].

Justification: Sections 3 and A specify the PACE algorithm, speed signal, smoothing, threshold calibration, and multi-arm fusion rule. Sections 4, B, C, D, and E describe the benchmarks, seeds, trial counts, baselines, training settings, and evaluation protocols.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• If the paper includes experiments, a [No] answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.

• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.

• Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.

• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example

(a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.

(b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.

(c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).

(d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [No].

Justification: Code and usage instructions are available through the PACE project page. The release covers a subset of the reported experiments.

Guidelines:

• The answer [N/A] means that paper does not include experiments requiring code.

• Please see the NeurIPS code and data submission guidelines (https://neurips.cc/ public/guides/CodeSubmissionPolicy) for more details.

• While we encourage the release of code and data, we understand that this might not be possible, so [No] is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).

• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https: //neurips.cc/public/guides/CodeSubmissionPolicy) for more details.

• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.

• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.

• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).

• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer) necessary to understand the results?

Answer: [Yes].

Justification: The main text and appendix specify the prediction horizon, execution baselines, PACE configuration, threshold calibration from demonstrations, seeds, episode counts, ablation settings, and real-robot trial counts. Appendix E.2 reports optimizer, batch size, learning-rate schedule, precision, checkpoint step, and hardware for real-robot fine-tuning. Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.

• The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [No].

Justification: The simulation results and phase-alignment analysis include uncertainty estimates (Appendices B.1.2, B.2, C.4, and D.3). The main real-robot results are reported without uncertainty estimates.

## Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The authors should answer [Yes] if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.

• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).

• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)

• The assumptions made should be given (e.g., Normally distributed errors).

• It should be clear whether the error bar is the standard deviation or the standard error of the mean.

• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.

• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g., negative error rates).

• If error bars are reported in tables or plots, the authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [No].

Justification: Appendix E.2 reports the GPU setup used for real-robot fine-tuning, and the simulation and test-time execution experiments were run on internal GPU servers with two Intel Xeon Platinum 8468V sockets, approximately 1.9 TiB system memory, and eight NVIDIA H20 GPUs with 97,871 MiB memory each. However, the current manuscript does not provide precise wall-clock runtimes, GPU-hours, or total compute budget for each experiment.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.

• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes].

Justification: The work uses simulation benchmarks and controlled robot manipulation experiments, and does not involve private data, human-subject studies, or deceptive userfacing deployment. We have reviewed the NeurIPS Code of Ethics and are not aware of any deviation.

Guidelines:

• The answer [N/A] means that the authors have not reviewed the NeurIPS Code of Ethics.

• If the authors answer [No], they should explain the special circumstances that require a deviation from the Code of Ethics.

• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [No].

Justification: The current manuscript focuses on test-time execution for robot policies and does not include a dedicated broader-impact discussion. Potential positive impacts include more reliable robot execution without retraining, while negative impacts would mainly arise from deploying more capable robot policies without appropriate real-world safety controls.

Guidelines:

• The answer [N/A] means that there is no societal impact of the work performed.

• If the authors answer [N/A] or [No], they should explain why their work has no societal impact or why the paper does not address societal impact.

• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate Deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.

• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.

• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pre-trained language models, image generators, or scraped datasets)?

Answer: [N/A].

Justification: We release PACE code and task-specific RoboTwin policy checkpoints for manipulation research. The release does not include high-risk models or datasets.

Guidelines:

• The answer [N/A] means that the paper poses no such risks.

• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.

• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.

• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [No].

Justification: The paper cites the original sources. OpenPI, LeRobot, GR00T, and AutoHori zon code use Apache License 2.0. AAC’s LIBERO implementation, LIBERO code, and RoboTwin2.0 code/data use MIT. The PaliGemma backbone of π<sub>0.5</sub> is subject to the Gemma Terms of Use. GR00T N1.7 weights use the NVIDIA Open Model License Agreement, distinct from the code license. LIBERO demonstrations and DROID data use CC BY 4.0. We access DROID through the DreamZero conversion. RoboMIND 1.x, RoboMIND 2.0, and RoboCOIN data use Apache License 2.0. AgiBot World Beta data and RoboChallenge inference code use CC BY-NC-SA 4.0, with attribution and noncommercial-use conditions, and share-alike conditions when adaptations are shared. The consulted Table30 V1 demonstration releases do not specify a data license.

Guidelines:

• The answer [N/A] means that the paper does not use existing assets.

• The authors should cite the original paper that produced the code package or dataset.

• The authors should state which version of the asset is used and, if possible, include a URL.

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.

• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.

• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.

• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.

• If this information is not available online, the authors are encouraged to reach out to the asset’s creators.

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [Yes].

Justification: The released code includes usage and licensing documentation. The RoboTwin policy checkpoints are accompanied by a model card describing training and usage, together with the applicable model terms.

Guidelines:

• The answer [N/A] means that the paper does not release new assets.

• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.

• The paper should discuss whether and how consent was obtained from people whose asset is used.

• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [N/A].

Justification: The paper does not involve crowdsourcing, user studies, or research with human subjects.

Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.

• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [N/A].

Justification: The paper does not involve crowdsourcing, user studies, or research with human subjects, so IRB approval is not applicable.

Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.

• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigor, or originality of the research, declaration is not required.

Answer: [Yes].

Justification: The paper studies vision-language-action robot policies and describes the use of language-conditioned robot policies such as π throughout Sections 1, 2, 3, and 5. LLMs used only for writing, editing, or formatting are not material to the research claims.

Guidelines:

• The answer [N/A] means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.

• Please refer to our LLM policy in the NeurIPS handbook for what should or should not be described.