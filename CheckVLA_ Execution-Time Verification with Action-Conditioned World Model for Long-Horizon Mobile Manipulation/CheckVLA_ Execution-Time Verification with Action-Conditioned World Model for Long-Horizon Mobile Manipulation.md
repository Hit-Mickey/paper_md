# CheckVLA: Execution-Time Verification with Action-Conditioned World Model for Long-Horizon Mobile Manipulation

Yushan Liu<sup>1</sup>, Peibo Sun<sup>2</sup>, Xintao Chao<sup>1</sup>, Zhenyang Yang<sup>3</sup>, Yifan Xie<sup>1</sup>, Lingfeng Zhang<sup>1</sup>, Shoujie Li<sup>4</sup>, Chenyu Tang<sup>3</sup>, Fang Chen<sup>2</sup>, Xiao-Ping Zhang<sup>1</sup>, Wenbo Ding<sup>1,5∗</sup>

<sup>1</sup>Tsinghua University <sup>2</sup>Shanghai Jiao Tong University <sup>3</sup>Peking University <sup>4</sup>Nanyang Technological University <sup>5</sup>Xspark AI

## Abstract

Vision-language-action (VLA) policies commonly execute long-horizon mobile manipulation through open-loop action chunks, issuing multiple actions without receiving new highlevel visual input. A committed chunk therefore implies how observations should evolve, but accidental deviations can violate this expectation while the remaining actions continue to propagate the error: commit-time policy confidence cannot react to a deviation that occurs after dispatch, and observationonly anomaly scores lack an action-conditioned reference for separating expected efects from unexplained changes. We propose CheckVLA, which verifies execution with a separately trained, frozen action-conditioned world model. A conformally calibrated risk threshold bounds the episode-level probability ofan unnecessary first intervention and determines when to intervene, its exceedance controls how strongly the rewritten sufix retains the superseded chunk, latency-aware hard prefixing restricts replacement to actions that remain deployable, and an event-driven keyframe bank preserves evidence of prior progress across repairs. On RoboCasa365, under a common training recipe and a matched invocation budget, CheckVLA attains a 36.1% average success rate against 27.6% for periodic replanning (+8.5 points). At a matched 5% episode-level false-alarm target, action conditioning raises timely recall to 77.9%, against 48.6% for an observation-only control and 37.9% for an action-shufled control. These simulation results support action-conditioned verification as a way to restore feedback during chunked execution while keeping the repair consistent with inference latency.

## Introduction

Long-horizon mobile manipulation interleaves navigation with contact-rich interaction, making it a demanding testbed for embodied intelligence. Recent VLA systems support mobile whole-body control across increasingly diverse tasks (Physical Intelligence et al. 2025b; Torne et al. 2026; Physical Intelligence et al. 2026; Chen et al. 2026; Jiang et al. 2025; Fan et al. 2026). Many execute a fixed-length action chunk without re-querying the policy on updated observations (Zhao et al. 2023; Chi et al. 2023; Black et al. 2024). Chunking amortizes inference and improves temporal consistency, but each issued chunk also predicts that observations will evolve consistently with its committed actions.

A post-dispatch deviation can violate this prediction without prompting replanning: an object may slip while subsequent actions still assume a secure grasp, or a displaced base may continue smooth but incorrect motion. Because the remaining actions can stay locally plausible after their premise fails, commit-time policy scores may miss this confidently wrong regime. The resulting errors compound over long horizons (Ross and Bagnell 2010; Ross, Gordon, and Bagnell 2011; Spencer et al. 2021), motivating an executiontime test of whether committed actions produce their expected consequences.

Existing signals cover only parts of this problem. Committime uncertainty (Liu et al. 2024; Jing et al. 2025; Wang et al. 2026a) cannot use observations arriving after dispatch without another policy call. Post-dispatch monitors use observations, policy features, or world-model scores (Liu, Bahety, and Song 2023; Guo et al. 2023; Gu et al. 2025; Zhang et al. 2026; Ho et al. 2026), but observation-only variants lack an action-conditioned reference. Adaptive execution and asynchronous continuity methods (Physical Intelligence et al. 2025a, 2026; Wang et al. 2026c; Black, Galliker, and Levine 2025; Wang et al. 2026b) address when to refresh a chunk or how to preserve continuity, yet do not generally combine an episode-calibrated online test with latency-aware sufix repair. This leaves open how to turn action-conditioned consequence predictions into a calibrated intervention and a deployable correction.

We present CheckVLA, which tests committed actions against observed consequences during execution (Figure 1). A separately trained, frozen world model predicts shorthorizon features from the latest observation and remaining actions; a causal risk head aggregates prediction–observation discrepancies. On exchangeable nominal successes, a functional conformal threshold bounds the probability of an unnecessary first intervention. The first crossing sets repair timing, while a validation-selected exceedance map sets reference retention. The same VLA rewrites only the latencyfeasible sufix, and a keyframe bank preserves episodic context.

On RoboCasa365 (Nasiriany et al. 2026), CheckVLA achieves 36.1% average success, setting the state of the art among published Human300 policy-pretraining results (Table 2). Under a common internal training recipe and paired seeds, it exceeds invocation-matched periodic replanning by 8.5 percentage points. At a common target episodelevel family-wise error rate (FWER), removing action conditioning reduces timely recall from 77.9% to 48.6% for an observation-only control and 37.9% for an action-shufled control.

![](images/9f41546962d766f6bbce66416fd6bb454651666274501c35a2b7f3b71071b773.jpg)  
Figure 1: Overview of CheckVLA. Open-loop chunks can propagate in-chunk failures such as slippage. CheckVLA compares action-conditioned predictions with arriving observations, uses an episode-calibrated threshold to trigger intervention, and rewrites the repairable sufix while preserving episodic context.

Our contributions are as follows:

• We cast an open-loop action chunk as a testable prediction of near-future observations, linking failure detection to action-conditioned consequence prediction and repair to rewriting the latency-feasible sufix.

• We develop CheckVLA with an episode-calibrated trigger, validation-selected risk-adaptive retention, a latencyaware hard prefix, and event-driven episodic memory.

• CheckVLA reaches state-of-the-art average success among RoboCasa365 Human300 policy-pretraining methods and exceeds invocation-matched periodic replanning under a common recipe; controlled studies audit detection, repair, memory, latency, robustness, runtime, and failures.

## Related Work

## VLA for Mobile Manipulation

VLA models jointly predict base, arm, and gripper actions for mobile platforms (Fu, Zhao, and Finn 2024; Black et al. 2024; Physical Intelligence et al. 2025b; Jiang et al. 2025; Liu et al. 2025); cross-embodiment training, subtask decoding, video memory, and action-subspace decoupling broaden their coverage (Physical Intelligence et al. 2025b; Torne et al. 2026; Physical Intelligence et al. 2026; Chen et al. 2026; Wang et al. 2026d; Chao et al. 2025). Because these systems commonly commit fixed-length chunks (Zhao et al. 2023; Chi et al. 2023), long-horizon extensions have mainly changed training or boundary-time context (Physical Intelligence et al.

2025a; Chen et al. 2026; Physical Intelligence et al. 2026). CheckVLA instead verifies a chunk during execution and rewrites its remaining sufix.

## World Models for Robotics

Action-conditioned world models support policy conditioning, data generation, and planning (Zhang et al. 2025; Cen et al. 2025; Zhou et al. 2024b; Assran et al. 2025; Liu et al. 2026b); several predict frozen-space features without pixel decoding (Zhou et al. 2024b; Liu et al. 2026c), building on latent imagination (Hafner et al. 2018, 2019; Rao et al. 2026). Related systems condition a VLA on predicted features (Sun et al. 2026a), couple imagination to action generation (Chen et al. 2026; Zhong et al. 2026), verify world-model predictions (Liu et al. 2026a), or adapt execution length by comparing predicted and realized futures (Wang et al. 2026c). CheckVLA uses consequence predictions as a calibrated execution-time test, separates the verifier from the policy path, and converts an alarm into latency-aware sufix repair.

## Execution Monitoring and Intervention

Adaptive chunking changes the execution window using latency or policy-side uncertainty and preserves continuity under asynchronous inference (Black, Galliker, and Levine 2025; Liu et al. 2024; Jing et al. 2025; Wang et al. 2026a,b). Monitoring uses constraints, generated checks, policy features, or world-model scores (Guo et al. 2023; Zhou et al. 2024a; Gu et al. 2025; Zhang et al. 2026; Ho et al. 2026), while pre-dispatch screening, backup policies, guided regeneration, and keyframe memory address complementary stages (Sun et al. 2026b; Chen 2026; Pan et al. 2026; Zeng et al. 2026; Torne et al. 2026). CheckVLA combines an episode-calibrated action-conditioned trigger with a validation-selected exceedance map for latency-feasible

<table><tr><td>Method</td><td>Act.-cond. signal</td><td>Calib. trigger</td><td>In-chunk repair</td><td>Latency aware</td><td>Epis. memory</td></tr><tr><td>SAFE</td><td>-</td><td>√</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Foresight</td><td>√</td><td>√</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Pre-VLA</td><td>√</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Future-reality verif.</td><td>√</td><td>-</td><td>~</td><td>-</td><td>-</td></tr><tr><td>RTC</td><td>-</td><td>-</td><td>~</td><td>√</td><td>-</td></tr><tr><td>CheckVLA (ours)</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

Table 1: Execution-time capability comparison with SAFE (Gu et al. 2025), Foresight (Zhang et al. 2026), Pre-VLA (Sun et al. 2026b), future–reality verification (Wang et al. 2026c), and RTC (Black, Galliker, and Levine 2025). $\checkmark :$ core mechanism; ∼: partial (execution-length adaptation; switch-time continuity rather than deviation-triggered repair); –: absent. Marks reflect published mechanisms, not re-implementations.

repair and persistent episodic context; Table 1 positions it against the closest execution-time systems.

## Method

We introduce CheckVLA, a closed-loop execution framework that verifies the expected consequences of committed action chunks and repairs deviations while recovery remains feasible. As shown in Figure 2, four coupled components— action-conditioned rolling prediction, calibrated risk triggering, latency-aware sufix rewriting, and episodic context— implement the loop.

## Problem Formulation

At a chunk boundary t, the policy $\pi _ { \theta }$ encodes the instruction $L ,$ multi-view observation $o _ { t } ,$ and episodic context $\rho _ { t }$ as $c _ { t } = \mathrm { V L M } ( L , o _ { t } ; \rho _ { t } )$ and commits

$$
A _ {t} = (a _ {t, 0}, \dots , a _ {t, H - 1}) \sim \pi_ {\theta} (\cdot | c _ {t}, s _ {t}),\tag{1}
$$

where $s _ { t }$ is proprioception and $A _ { t } \in \mathbb { R } ^ { H \times M }$ spans mobility and manipulation. We use h for a candidate position, $h _ { \tau }$ for the active in-chunk index at control time τ, and ${ \bar { a } } _ { \tau }$ for the action then committed for execution. The latest observation time $t ^ { \prime }$ is an anchor, $\ell \leq k$ is a prediction span, and $u \in$ [0, 1] is flow-integration time. During a chunk, the high-level policy receives no new visual input (Zhao et al. 2023; Chi et al. 2023; Black et al. 2024), whereas the monitor does; repair is possible only while a useful sufix remains.

Suppose the first trigger occurs at $t ^ { * }$ with next old-chunk index $j ~ = ~ h _ { t ^ { * } }$ . The fixed schedule allocates $d _ { \mathrm { l a t } }$ control steps to inference, while $d _ { \mathrm { e f f } }$ denotes the elapsed steps when the replacement actually becomes deployable. Normally $d _ { \mathrm { e f f } } = d _ { \mathrm { l a t } } ;$ an early call waits, whereas an overrun makes $d _ { \mathrm { e f f } } > d _ { \mathrm { l a t } }$ and its elapsed replacement positions are discarded. Candidate positions $h < d _ { \mathrm { e f f } }$ are therefore irreversible, and only $h \geq d _ { \mathrm { e f f } }$ can be deployed (supplementary Secs. A–B).

## Action-Conditioned Rolling Prediction

While the chunk executes, each arriving observation restarts a short verification cycle: if the plan remains valid, future observations should agree with the action-conditioned predictions of the committed actions.

Let $z _ { t ^ { \prime } } = \phi ( o _ { t ^ { \prime } } )$ be the latest observation feature under a separately pretrained, frozen monitoring encoder $\phi .$ Conditioned on the still-valid committed actions and nominal proprioceptive rollout $\hat { s } _ { t ^ { \prime } + i } = f ( s _ { t ^ { \prime } } , \bar { a } _ { t ^ { \prime } : t ^ { \prime } + i - 1 } )$ , the world model Ψ predicts only the next k steps $( k \ll H )$

$$
\begin{array}{r l} \hat {z} _ {t ^ {\prime} + i + 1} = \Psi \big (z _ {t ^ {\prime}}, \hat {z} _ {t ^ {\prime} + 1: t ^ {\prime} + i}, \\ & \bar {a} _ {t ^ {\prime}: t ^ {\prime} + i}, \hat {s} _ {t ^ {\prime}: t ^ {\prime} + i} \big), \quad i = 0, \ldots , k - 1. \end{array}\tag{2}
$$

Each observation re-anchors the rollout. Predictions crossing a chunk boundary are generated only after the next chunk is committed; after a rewrite, predictions conditioned on superseded actions are invalidated, and retrieval uses the latest valid anchor. Short rollouts limit autoregressive drift, while span-specific statistics below handle residual scale diferences.

Ψ is trained first by teacher forcing and then on its own rollouts, matching the online autoregressive context (Hafner et al. 2018, 2019; Rao et al. 2026); training and prediction remain in feature space (supplementary Sec. D). The frozen encoder isolates monitoring from policy optimization. Although $\pi _ { \theta }$ is stochastic, Ψ provides a deterministic reference; empirical residual variation is absorbed by the fixed standardization and calibration pipeline.

## Calibrated Risk Triggering

As $z _ { t }$ arrives, the monitor retrieves the prediction from the latest valid anchor with span $\ell ( t ) \ \leq \ k ,$ computes $d _ { t } = \mathrm { d i s t } \big ( \hat { z } _ { t | t - \ell ( t ) } , z _ { t } \big )$ on pooled, normalized features, and standardizes it with held-out-success discrepancy statistics:

$$
\tilde {d} _ {t} = \frac {d _ {t} - \mu_ {d} ^ {(\ell)}}{\max (\sigma_ {d} ^ {(\ell)} , \varepsilon)}.\tag{3}
$$

The distance metric is selected on validation data; each observation then becomes a new anchor, and monitoring resumes at span one after a chunk switch.

A single-step discrepancy is weak evidence on its own, because contact, occlusion, and sensing noise all produce transient residuals. A temporally causal risk head R therefore aggregates persistent and increasing patterns (Ho et al. 2026) over the window $x _ { t - w + 1 : t }$ of per-step tuples:

$$
x _ {\tau} = (\tilde {d} _ {\tau}, \ell (\tau) / k, h _ {\tau} / H, \Delta_ {\mathrm{sw}, \tau}),\tag{4}
$$

where $h _ { \tau }$ is the action index within the current chunk and $\Delta _ { \mathrm { s w } , \tau }$ is the number of control steps since the most recent chunk switch. The risk head additionally receives the episodic summary $m ( \rho _ { t } )$ introduced below, yielding

$$
r _ {t} = R \big (x _ {t - w + 1: t}, m (\rho_ {t}) \big) \in [ 0, 1 ].\tag{5}
$$

The risk head is trained on nominally successful episodes, natural failures of the open-loop backbone, and onsetannotated physical perturbations; successes and benign deviations serve as negatives, and supervision is restricted to pre-failure evidence (objective and perturbation details in supplementary Sec. D).

![](images/aebed813adc3a2d3d72a96dbf69fba0b11d8786b01398449e670a7a9b66059a7.jpg)

Figure 2: The CheckVLA framework. A separately trained, frozen encoder supplies monitoring features; the world model predicts near-term consequences of the committed actions; the causal risk head compares aggregated risk $r _ { t }$ with the calibrated threshold $\delta _ { t } ;$ upon a declared failure, the same VLA rewrites the repairable sufix under the hard-prefix constraint with riskgraded guidance.  
![](images/bde19b2540b684bea8f53726f71a1b04254367b05d6127cbf1f9979963f96b09.jpg)  
Figure 3: Composite RoboCasa365 episodes (Arrange Bread Basket, top; Gather Tableware, bottom): subtask prompts aligned with third- and first-person keyframes; blue denotes manipulation and orange base motion.

Because one unnecessary intervention can spoil an otherwise successful episode, CheckVLA controls the probability of an unnecessary first intervention over the episode. After fixing the policy, monitor, retrieval, discrepancy standardization, and memory rules, we fit risk statistics $\left( \mu _ { r , t } , \sigma _ { r , t } \right)$ on nominal-success shadow-mode trajectories disjoint from conformal calibration and set $\tilde { \sigma } _ { r , t } ~ = ~ \operatorname* { m a x } ( \sigma _ { r , t } , \varepsilon )$ . Split functional conformal calibration uses the per-trajectory scores su $\mathfrak { p } _ { t } ( r _ { t } ^ { ( i ) } - \mu _ { r , t } ) / \tilde { \sigma } _ { r , t }$ and their finite-sample quantile $\hat { q } _ { \alpha }$ at miscoverage α (Zhang et al. 2026). The online threshold is

$$
\delta_ {t} = \mu_ {r, t} + \hat {q} _ {\alpha} \tilde {\sigma} _ {r, t},\tag{6}
$$

and, for a new nominal-success trajectory exchangeable with calibration under this same fixed pipeline, satisfies the trajectory-marginal guarantee

$$
\operatorname * {P r} \left(\exists t: r _ {t} > \delta_ {t} \mid \text { nominal   success }\right) \leq \alpha .\tag{7}
$$

This is only a first-intervention guarantee: it covers neither failure recall, post-repair safety, repeated interventions, nor distribution shift. Those properties are evaluated empirically (supplementary Table S17). The first crossing $t ^ { * } = \operatorname* { m i n } \{ t$ $r _ { t } > \delta _ { t } \}$ triggers repair.

## Sufix Rewriting

At t<sup>∗</sup>, execution continues while the same VLA generates a candidate indexed relative to the trigger. Its old-chunk reference is $A _ { \mathrm { r e f } , h } = \bar { a } _ { t ^ { * } + h } ^ { \mathrm { o l d } }$ for $0 \leq h \overset { \vartriangle } { < } H - j$ . At every flowintegration step, candidate positions $h < d _ { \mathrm { l a t } }$ are clamped to this executing prefix. Under nominal timing deployment starts at $h = d _ { \mathrm { l a t } } ;$ after an overrun it starts at $h \ : = \ : d _ { \mathrm { e f f } }$ discarding the extra elapsed positions. The hard clamp is an empirical projection, not exact conditional sampling (supplementary Sec. J).

The retention strength of the superseded chunk is determined by the exceedance ofthe same calibrated threshold that produced the trigger. The remaining actions of that chunk provide a channel-wise reference, weighted by the standardized exceedance

<table><tr><td>Method</td><td>Atomic-Seen</td><td>Composite-Seen</td><td>Composite-Unseen</td><td>Average</td></tr><tr><td>Diffusion Policy (Chi et al. 2023)</td><td>15.7</td><td>0.2</td><td>1.3</td><td>6.1</td></tr><tr><td> $\pi_0$  (Black et al. 2024)</td><td>34.6</td><td>6.1</td><td>1.1</td><td>14.8</td></tr><tr><td> $\pi_{0.5}$  (Physical Intelligence et al. 2025b)</td><td>39.6</td><td>7.1</td><td>1.2</td><td>16.9</td></tr><tr><td>GigaWorld-Policy 0.1 (Ye et al. 2026)</td><td>44.4</td><td>11.8</td><td>2.9</td><td>20.7</td></tr><tr><td>GR00T-N1.5 (Bjorck et al. 2025)</td><td>50.7</td><td>14.8</td><td>2.7</td><td>23.9</td></tr><tr><td>GR00T-N1.6 (GEAR Team et al. 2025)</td><td>51.1</td><td>9.4</td><td>1.7</td><td>21.9</td></tr><tr><td>RLDX-1 (Kim et al. 2026)</td><td>63.0</td><td>27.5</td><td>5.4</td><td>33.2</td></tr><tr><td>Qwen-RobotManip (Yuan et al. 2026)</td><td>68.6</td><td>20.1</td><td>14.9</td><td>35.9</td></tr><tr><td>WorldDreamer (Wang et al. 2024)</td><td>66.3</td><td>26.7</td><td>9.0</td><td>35.3</td></tr><tr><td>CheckVLA (ours)</td><td>63.7</td><td>30.9</td><td>10.2</td><td>36.1</td></tr></table>

Table 2: RoboCasa365 mean task success (%) with policy pretraining limited to Human300, over 18 A-S, 16 C-S, and 16 C-U tasks. Average is task-count weighted; bold/underline denote the best/second-best. Published results provide descriptive positioning rather than paired comparison; CheckVLA additionally trains its verifier on training-side rollouts, and Table 3 give the controlled comparison.

$$
e _ {t ^ {*}} = \left(\left(r _ {t ^ {*}} - \mu_ {r, t ^ {*}}\right) / \tilde {\sigma} _ {r, t ^ {*}} - \hat {q} _ {\alpha}\right) _ {+},\tag{8}
$$

the amount by which the standardized risk exceeds the threshold; the mapping from exceedance to retention below is selected on validation data and carries no conformal guarantee. A base weight $w _ { 0 } ( e ) = w _ { \mathrm { m i n } } + ( 1 - w _ { \mathrm { m i n } } ) \exp ( - \beta e )$ , with $w _ { \mathrm { m i n } } \in [ 0 , 1 )$ and $\beta > 0$ , decreases monotonically in e: a small exceedance retains strong reference guidance, while a large one yields weaker retention and permits a larger correction. The weight decays with position at a channel-specific rate,

$$
W _ {h, m} = w _ {0} (e _ {t ^ {*}}) \exp \big (- \lambda_ {m} (h - d _ {\mathrm{lat}}) \big).\tag{9}
$$

This applies for $d _ { \mathrm { l a t } } \leq h < H - j$ and $W = 0$ otherwise, with rates $\lambda _ { m } > 0$ for the mobility and manipulation channels. Guidance mixes the reference into the flow velocity,

$$
v = (1 - W) \odot v _ {\theta} + W \odot v _ {\mathrm{ref}},\tag{10}
$$

where y is the current flow state, $\Delta u$ the integration step, and $v _ { \mathrm { r e f } } = ( A _ { \mathrm { r e f } } - y ) / \operatorname* { m a x } ( 1 - u , \Delta u )$ . After switching, the loop resumes from the next observation (regular-transition guidance and suppression in supplementary Sec. D).

## Episodic Context

A repair conditioned only on the current view may forget completed subgoals. CheckVLA therefore sets $\rho _ { t } \equiv B _ { t } =$ $\{ ( \boldsymbol { F _ { j } } , t _ { j } ) \} _ { j = 1 } ^ { m _ { t } }$ , an event-driven bank of pooled policy features from real observations. Pause and diversity criteria govern writes across chunks and repairs. The policy reads the full bank through gated cross-attention, whereas the risk head receives only the compact summary $m ( \rho _ { t } )$ ; these policy features $F _ { j }$ are distinct from the monitor features $z _ { t }$ (supplementary Sec. E).

## Implementation

CheckVLA uses the $\pi _ { 0 . 5 }$ flow-matching backbone (Physica Intelligence et al. 2025b). Mobility and manipulation experts have separate parameters but exchange information through joint self-attention over the shared VLM prefix (Chen et al. 2026); stop-gradient blocks their training signal from the VLM backbone (Driess et al. 2025). Monitoring uses a frozen V-JEPA 2-AC encoder (Assran et al. 2025). Supplementary Secs. C–E describe the decoupled action experts, verifier training and calibration, and memory fusion.

## Experiments

## Protocol and Public Benchmark

We evaluate on RoboCasa365 (Nasiriany et al. 2026), comprising 365 household mobile-manipulation tasks in 2,500 kitchens; Figure 3 shows two composite episodes. Following the oficial protocol, the action policy uses only Human300 (300 tasks, 100 human demonstrations each) and is tested on 18 Atomic-Seen (A-S), 16 Composite-Seen (C-S), and 16 Composite-Unseen (C-U) tasks. The verifier uses auxiliary nominal, natural-failure, and physics-perturbed rollouts from training-side tasks and scenes. Neither target-suite scenes nor C-U tasks train any component (splits and leakage guards in supplementary Sec. F).

Table 2 lists published Human300 flagship results. Check-VLA achieves 36.1% average success, establishing a new state of the art among the methods considered: 19.2 percentage points above the published $\pi _ { 0 . 5 }$ and 0.2 above Qwen-RobotManip. It ranks first on C-S and second on C-U while remaining competitive on A-S, consistent with execution verification benefiting tasks that chain subgoals. Because public systems difer beyond policy-pretraining data, this state-ofthe-art positioning is descriptive; the controlled attribution follows.

## Fair Internal Controls

Table 3 builds the system under common splits, optimization, and paired evaluation seeds, first controlling action-model capacity and then adding transition guidance, periodic replanning, verified timing, adaptive guidance, and memory. Changed policy inputs are retrained and afected monitors recalibrated. Validation selects the periodic interval, yielding 10.1 calls per episode versus 10.2 for CheckVLA. Replacing

a Episode-level calibration

<table><tr><td>Variant</td><td>A-S</td><td>C-S</td><td>C-U</td><td>Average</td><td>Calls/ep.</td><td>Δ Avg.</td></tr><tr><td>Reproduced π0.5</td><td>39.8</td><td>7.4</td><td>1.5</td><td>17.2</td><td>7.8</td><td>-</td></tr><tr><td>Capacity-matched monolithic expert</td><td>45.9</td><td>10.8</td><td>2.8</td><td>20.9</td><td>7.8</td><td>+3.7</td></tr><tr><td>Decoupled action backbone, open loop</td><td>46.8</td><td>11.9</td><td>3.1</td><td>21.6</td><td>7.8</td><td>+0.7</td></tr><tr><td>+ regular-transition guidance</td><td>48.7</td><td>13.6</td><td>3.8</td><td>23.1</td><td>7.8</td><td>+1.5</td></tr><tr><td>+ invocation-matched periodic replan</td><td>53.9</td><td>19.8</td><td>5.8</td><td>27.6</td><td>10.1</td><td>+4.5</td></tr><tr><td>+ verified trigger, fixed suffix guidance</td><td>58.8</td><td>24.7</td><td>7.5</td><td>31.5</td><td>10.0</td><td>+3.9</td></tr><tr><td>+ risk-adaptive guidance, no memory</td><td>61.5</td><td>28.3</td><td>9.0</td><td>34.1</td><td>10.0</td><td>+2.6</td></tr><tr><td>Full CheckVLA</td><td>63.7</td><td>30.9</td><td>10.2</td><td>36.1</td><td>10.2</td><td>+2.0</td></tr></table>

Table 3: Sequential build-up on RoboCasa365 under a common training recipe and paired seeds. Average is task-count weighted; Calls/ep. reports the invocation budget. Supplementary Secs. H–P provide detector, binding, repair, robustness, memory, runtime, and failure audits.

![](images/95e4962a16ef7e34ea985a95821d80e86853c1456a9ef5637090fa9727c8a748.jpg)

b Latency-sensitive recovery  
![](images/c1d72376227d53a486fdac76236ff480be9d4f1866ace9055e80e7bf85d09026.jpg)

c Benefit grows with horizon  
![](images/8f755e2c8cf0a2e6e999dcfc23abebece2461a94edb9684cb636a18cd1641f16.jpg)

Figure 4: Closed-loop diagnostics. (a) Episode-level unnecessary-first-intervention frequency versus target conformal level. (b) Perturbation recovery across inference latency. (c) Paired gain over open loop across horizon quartiles, with and without memory. Bars and bands are 95% intervals: exact binomial in (a), hierarchical paired bootstrap in (b) and (c).

<table><tr><td>Detector</td><td>EpisodeFWER (%)</td><td>Timelyrecall (%) ↑</td><td>Perturbedsuccess (%) ↑</td></tr><tr><td>MC action entropy</td><td>5.0</td><td>58.7</td><td>38.9</td></tr><tr><td>Flow-path variance</td><td>5.2</td><td>61.4</td><td>40.2</td></tr><tr><td>Sampling disagreement</td><td>4.9</td><td>64.8</td><td>41.5</td></tr><tr><td>Policy-feature probe</td><td>5.1</td><td>68.6</td><td>43.7</td></tr><tr><td>Obs.-only world predictor</td><td>5.3</td><td>48.6</td><td>35.2</td></tr><tr><td>Action-shuffled predictor</td><td>5.1</td><td>37.9</td><td>31.0</td></tr><tr><td>No rolling re-anchor</td><td>5.4</td><td>62.8</td><td>38.7</td></tr><tr><td>Instantaneous score</td><td>5.0</td><td>66.1</td><td>41.2</td></tr><tr><td>Constant threshold</td><td>7.1</td><td>71.5</td><td>45.0</td></tr><tr><td>Full action-cond. verifier</td><td>4.8</td><td>77.9</td><td>47.6</td></tr></table>

Table 4: Trigger comparison at target episode FWER $\alpha =$ 0.05 with a shared rewriter and latency. Policy-side scores are evaluated at commit time; the constant threshold is validation-tuned, not conformally calibrated; the third block ablates verifier mechanisms.

periodic timing with the verified trigger under fixed guidance adds +3.9 points; risk-adaptive guidance and memory add +2.6 and +2.0.

## Ablation Studies and Deployment Audits

A controlled study injects physical perturbations across task phases in 12 composite tasks. Conformal detector variants are independently recalibrated at the common target episode

FWER, detector comparisons replay identical shadow trajectories, and rewrite variants branch from the same firsttrigger snapshot with shared randomness. Episode FWER is the fraction of nominal successes with an unnecessary first intervention. A detection is timely when it crosses after annotated onset and deploys before the end of the empirically estimated recovery window under the tested repair set. Rescue and harm are paired outcome changes from the trigger snapshot (supplementary Sec. G); the constant-threshold control is deliberately nonconformal.

Action conditioning. To separate action–consequence evidence from observation novelty, we retrain an action-shufled world model that preserves action-token marginals and an observation-only predictor. Timely recall falls from 77.9% to 37.9% and $4 \dot { 8 } . \dot { 6 } \% ,$ , and perturbed success from 47.6% to 31.0% and 35.2% (Table 4). These controls support the joint action-conditioned pathway; a frozen-model counterfactual audit confirms action–future binding without separating action tokens from their deterministic proprioceptive rollout (supplementary Sec. I).

Verifier mechanisms. Removing rolling re-anchoring or temporal aggregation costs 15.1 and 11.8 points of timely recall, respectively; a validation-tuned constant threshold raises FWER from 4.8% to 7.1% (Table 4; Figure 4a). Together these ablations support the composite design: short reanchored spans and persistence aggregation improve timeliness, while functional calibration sets the deployed operating

<table><tr><td>Policy unc.</td><td>WM risk</td><td>Ep. share (%)</td><td>Fail. rate (%)</td><td>Fail. share (%)</td><td>Timely recov. (%)</td><td>Rescue (%) ↑</td></tr><tr><td>Low</td><td>Low</td><td>52.0</td><td>12.0</td><td>25.2</td><td>18.0</td><td>6.0</td></tr><tr><td>High</td><td>Low</td><td>14.0</td><td>18.0</td><td>10.2</td><td>28.0</td><td>11.0</td></tr><tr><td>Low</td><td>High</td><td>25.0</td><td>48.0</td><td>48.4</td><td>76.0</td><td>31.0</td></tr><tr><td>High</td><td>High</td><td>9.0</td><td>45.0</td><td>16.2</td><td>73.0</td><td>29.0</td></tr></table>

Table 5: Commit-time policy uncertainty versus world-model risk on held-out natural executions assignable at the preregistered decision window (supplementary Sec. L), each signal split at its independently calibrated threshold; the lowuncertainty, high-risk cell is the confidently-wrong quadrant.

point.

Policy-side alternatives. We calibrate four commit-time detectors under the same protocol: Monte Carlo action entropy, flow-path variance, sampling disagreement, and a frozen policy-feature probe. The full verifier exceeds the strongest by 9.3 points in timely recall and 3.9 in perturbed success; refreshing policy scores at the observation rate would add 382.2 VLA evaluations per episode (supplementary Sec. H). Thus post-dispatch action-conditioned evidence adds information at monitor rather than policy cost.

Confidently-wrong quadrant. Table 5 splits held-out natural executions by independently calibrated commit-time uncertainty and world-model risk. The low-uncertainty, highrisk quadrant contains 25.0% of episodes but 48.4% of failures; 76.0% remain timely recoverable and 31.0% are rescued by the fixed rule. This operational pattern supports complementarity between the signals. Natural-failure and benign-shift tests probe distribution sensitivity, while leaveone-family-out, severity, and data-scale studies probe monitor generalization (supplementary Secs. L and N).

Repair rules. Six rewrite rules and a wait-for-boundary baseline branch from identical first-trigger snapshots: unconstrained generation with post-hoc prefix disposal, hard-prefix full rewrite, RTC inpainting (Black, Galliker, and Levine 2025), fixed guidance, adaptive guidance with shufled exceedance, and correctly paired adaptive guidance (Table 6). The paired rule reaches 16.9% rescue at 2.8% harm, versus 12.8%/3.7% for fixed guidance and 14.4%/3.6% when exceedance is shufled (Figure 4b; supplementary Secs. J– K). The pairing, not merely variable weights, improves both outcomes.

Episodic memory. A fully retrained and, where needed, recalibrated policy-bank × risk-summary factorial avoids the confounding of test-time zeroing. The full design cuts completed-subgoal regressions from 1.34 to 0.18 per episode, with success gains increasing over the horizon (Figure 4c). The policy reader drives most progress retention; the smaller risk-summary efect brings empirical FWER closer to target.

Deployment cost. The full system adds 88.4M monitor parameters and 16.9 GFLOPs per step, runs at 1.18× the open-loop wall-clock cost, and keeps monitor p95 latency below one control period. It intervenes in 4.8% of clean episodes and harms 2.1% of otherwise successful clean continuations; on held-out natural executions, success rises from 61.4% to 68.9% at 2.7% harm (supplementary Tables S8, S17, and S18). A blinded audit assigns residual failures to detection, physics, policy, or repair causes (supplementary Sec. P).

<table><tr><td>Rewrite rule</td><td>Rescue(%) ↑</td><td>Harm(%) ↓</td><td>Switchjump ↓</td><td>Success(%) ↑</td></tr><tr><td>Wait for boundary</td><td>7.4</td><td>4.8</td><td>-</td><td>34.0</td></tr><tr><td>Unconstrained, discard prefix</td><td>10.1</td><td>8.9</td><td>0.284</td><td>39.8</td></tr><tr><td>Hard prefix, full rewrite</td><td>13.2</td><td>6.8</td><td>0.171</td><td>42.4</td></tr><tr><td>RTC inpainting</td><td>13.7</td><td>4.1</td><td>0.091</td><td>45.6</td></tr><tr><td>Fixed guidance W</td><td>12.8</td><td>3.7</td><td>0.104</td><td>46.0</td></tr><tr><td>Shuffled exceedance</td><td>14.4</td><td>3.6</td><td>0.103</td><td>44.3</td></tr><tr><td>Adaptive W(e)</td><td>16.9</td><td>2.8</td><td>0.072</td><td>47.6</td></tr></table>

Table 6: Fixed-trigger repair at nominal latency $( d _ { \mathrm { l a t } } = 3 ) _ { \mathrm { : } }$ branched from identical first-trigger snapshots. Rescue, harm, and switch jump are one-shot outcomes with subsequent triggers disabled; success continues the closed loop (full latency sweep in supplementary Sec. J).

## Discussion

Our findings suggest that an action chunk is both a control command and a testable prediction of future observations. A separately trained action-conditioned verifier can restore feedback during open-loop execution by deciding when another policy call is warranted. Threshold exceedance then determines how strongly to revise the committed sufix. This is not a general safety guarantee, because functional conformal calibration controls only the probability of an unnecessary first intervention on exchangeable nominal-success episodes. It does not guarantee recall, post-repair safety, or coverage under distribution shift. Repair is also constrained by latency and policy capability: if too little deployable sufix remains or no viable replacement exists, even a correct warning cannot recover the episode. Evidence is currently limited to RoboCasa365 simulation, a finite perturbation family, and a frozen representation. Changes in sensing, hardware timing, or task distribution require recalibration and retesting. Dynamic latency-aware scheduling and hardware validation are therefore the natural next steps.

## Conclusion

We propose CheckVLA, which reframes a committed action chunk as a testable prediction and verifies it during execution with a separately trained action-conditioned world model, restoring feedback during chunked execution. At a matched policy-call budget on RoboCasa365, it improves average success by 8.5 percentage points over periodic replanning; a fixed-guidance ablation attributes 3.9 points to verified timing. These results suggest that execution-time verification complements policy scaling by restoring feedback that a stronger open-loop policy alone does not provide.

## References

Assran, M.; Bardes, A.; Fan, D.; Garrido, Q.; Howes, R.; Komeili, M.; Muckley, M.; Rizvi, A.; Roberts, C.; Sinha, K.; et al. 2025. V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning. arXiv:2506.09985.

Black, K.; Brown, N.; Driess, D.; Esmail, A.; Equi, M.; Finn, C.; Fusai, N.; Groom, L.; Hausman, K.; Ichter, B.; et al. 2024. π : A Vision-Language-Action Flow Model for General Robot Control. arXiv:2410.24164.

Black, K.; Galliker, M. Y.; and Levine, S. 2025. Real-Time Execution of Action Chunking Flow Policies. arXiv:2506.07339.

Cen, J.; Yu, C.; Yuan, H.; Jiang, Y.; Huang, S.; Guo, J.; Li, X.; Song, Y.; Luo, H.; Wang, F.; Zhao, D.; and Chen, H. 2025. WorldVLA: Towards Autoregressive Action World Model. arXiv:2506.21539.

Chao, X.; Mu, S.; Liu, Y.; Li, S.; Lyu, C.; Zhang, X.-P.; and Ding, W. 2025. Exo-ViHa: A Cross-Platform Exoskeleton System with Visual and Haptic Feedback for Eficient Dexterous Skill Learning. arXiv:2503.01543.

Chen, J. 2026. AEGIS: A Backup Reflex for Physical AI. arXiv:2606.06660.

Chen, R.; Yang, Y.; Tang, Z.; Huo, D.; Lin, T.; Wu, H.; Liu, H.; Chen, Y.; et al. 2026. ABot-M0.5: Unified Mobility-and-Manipulation World Action Model. arXiv:2607.00678.

Chi, C.; Xu, Z.; Feng, S.; Cousineau, E.; Du, Y.; Burchfiel, B.; Tedrake, R.; and Song, S. 2023. Difusion Policy: Visuomotor Policy Learning via Action Difusion. arXiv:2303.04137.

Driess, D.; Springenberg, J. T.; Ichter, B.; Yu, L.; Li-Bell, A.; Pertsch, K.; Ren, A. Z.; Walke, H.; Vuong, Q.; Shi, L. X.; and Levine, S. 2025. Knowledge Insulating Vision-Language-Action Models: Train Fast, Run Fast, Generalize Better. arXiv:2505.23705.

Fan, J.; Liu, Y.; Li, S.; Ren, B.; Li, S.; Zhang, X.-P.; Ding, W.; and Deng, Z. 2026. FUTURE-VLA: Forecasting Unified Trajectories Under Real-time Execution. arXiv:2602.15882.

Fu, Z.; Zhao, T. Z.; and Finn, C. 2024. Mobile ALOHA: Learning Bimanual Mobile Manipulation with Low-Cost Whole-Body Teleoperation. arXiv:2401.02117.

Gu, Q.; Ju, Y.; Sun, S.; Gilitschenski, I.; Nishimura, H.; Itkina, M.; and Shkurti, F. 2025. SAFE: Multitask Failure Detection for Vision-Language-Action Models. arXiv:2506.09937.

Guo, Y.; Wang, Y.-J.; Zha, L.; and Chen, J. 2023. DoReMi: Grounding Language Model by Detecting and Recovering from Plan-Execution Misalignment. arXiv:2307.00329.

Hafner, D.; Lillicrap, T.; Ba, J.; and Norouzi, M. 2019. Dream to Control: Learning Behaviors by Latent Imagination. arXiv:1912.01603.

Hafner, D.; Lillicrap, T.; Fischer, I.; Villegas, R.; Ha, D.; Lee, H.; and Davidson, J. 2018. Learning Latent Dynamics for Planning from Pixels. arXiv:1811.04551.

Ho, M.; Ginting, M. F.; Ward, I. R.; Reinke, A.; Kochenderfer, M. J.; Agha-Mohammadi, A.-a.; and Omidshafiei, S. 2026. World Model Failure Classification and Anomaly Detection for Autonomous Inspection. arXiv:2602.16182.

Jiang, H.; Chen, J.; Bu, Q.; Chen, L.; Shi, M.; Zhang, Y.; Li, D.; Suo, C.; Wang, C.; Peng, Z.; and Li, H. 2025. Whole-BodyVLA: Towards Unified Latent VLA for Whole-Body Loco-Manipulation Control. arXiv:2512.11047.

Jing, D.; Wang, G.; Liu, J.; Tang, W.; Sun, Z.; Yao, Y.; Wei, Z.; Liu, Y.; Lu, Z.; and Ding, M. 2025. Mixture of Horizons in Action Chunking. arXiv:2511.19433.

Kim, D.; Jang, H.; Koo, M.; Jang, S.; Kim, T.; Kim, B.; Yoon, B.; Jang, C.; Choi, D.; Han, D.; et al. 2026. RLDX-1 Technical Report. arXiv:2605.03269.

Liu, Y.; Feng, F.; Kong, L.; Lu, W.; Tang, J.; Zhang, K.; Murphy, K.; Finn, C.; and Du, Y. 2026a. World Action Verifier: Self-Improving World Models via Forward-Inverse Asymmetry. arXiv:2604.01985.

Liu, Y.; Hamid, J. I.; Xie, A.; Lee, Y.; Du, M.; and Finn, C. 2024. Bidirectional Decoding: Improving Action Chunking via Guided Test-Time Sampling. arXiv:2408.17355.

Liu, Y.; Lv, T.; Wang, B.; Fan, H.; Zhao, C.; Zheng, H.; Zhong, X.; Xie, Y.; Zhao, C.; Liao, Z.; Luo, L.; Cai, Y.; Zhang, X.-P.; and Ding, W. 2026b. PerceptDrive: Perception Prior World-Action Modeling with Adaptive Expert Routing for End-to-End Autonomous Driving. arXiv:2607.20175.

Liu, Y.; Mu, S.; Chao, X.; Li, Z.; Mu, Y.; Chen, T.; Li, S.; Lyu, C.; Zhang, X.-P.; and Ding, W. 2025. AVR: Active Vision-Driven Precise Robot Manipulation with Viewpoint and Focal Length Optimization. arXiv:2503.01439.

Liu, Y.; Sun, P.; Li, S.; Xie, Y.; Zhang, L.; Chao, X.; Dong, S.; Chen, F.; Zhang, X.-P.; and Ding, W. 2026c. OA-WAM: Object-Addressable World Action Model for Robust Robot Manipulation. arXiv:2605.06481.

Liu, Z.; Bahety, A.; and Song, S. 2023. REFLECT: Summarizing Robot Experiences for Failure Explanation and Correction. arXiv:2306.15724.

Nasiriany, S.; Nasiriany, S.; Maddukuri, A.; and Zhu, Y. 2026. RoboCasa365: A Large-Scale Simulation Framework for Training and Benchmarking Generalist Robots. arXiv:2603.04356.

Bjorck, J.; Blukis, V.; Castañeda, F.; Cherniadev, N.; Da, X.; Ding, R.; Fan, L.; Fang, Y.; Fox, D.; Hu, F.; et al. 2025. GR00T N1.5: An Improved Open Foundation Model for Generalist Humanoid Robots. NVIDIA GEAR, 11 June 2025. https://research.nvidia.com/labs/gear/gr00t-n1\_5/.

GEAR Team; Azzolini, A.; Bjorck, J.; Blukis, V.; Castañeda, F.; Chand, R.; Chang, Y.; Chen, D.; Cherniadev, N.; Da, X.; et al. 2025. GR00T N1.6: An Improved Open Foundation Model for Generalist Humanoid Robots. NVIDIA GEAR, 15 December 2025. https://research.nvidia.com/labs/gear/ gr00t-n1\_6/.

Pan, Y.; Pan, M.; Lu, Q.; Huang, J.; Zhang, M.; Huang, S.; Li, X.; Zhang, J.; Shen, Y.; Zhang, X.; and Zhang, W. 2026. VLA-Corrector: Lightweight Detect-and-Correct Inference for Adaptive Action Horizon. arXiv:2607.01804.

Physical Intelligence; Ai, B.; Amin, A.; Aniceto, R.; Balakrishna, A.; Balke, G.; Black, K.; Bokinsky, G.; Cao, S.; et al. 2026. π<sub>0.7</sub>: a Steerable Generalist Robotic Foundation Model with Emergent Capabilities. arXiv:2604.15483.

Physical Intelligence; Amin, A.; Aniceto, R.; Balakrishna, A.; Black, K.; Conley, K.; Connors, G.; Darpinian, J.; Dhabalia, K.; et al. 2025a. π<sup>∗</sup><sub>0.6</sub>: a VLA That Learns From Experience. arXiv:2511.14759.

Physical Intelligence; Black, K.; Brown, N.; Darpinian, J.; Dhabalia, K.; Driess, D.; Esmail, A.; Equi, M.; Finn, C.; et al. 2025b. $\pi _ { 0 . 5 } \colon \mathbf { A }$ Vision-Language-Action Model with Open-World Generalization. arXiv:2504.16054.

Rao, P.; Zhang, W.; Balestriero, R.; LeCun, Y.; and Loianno, G. 2026. SkyJEPA: Learning Long-Horizon World Models for Zero-Shot Sim-to-Real Control of Quadrotors. arXiv:2606.23444.

Ross, S.; and Bagnell, D. 2010. Eficient Reductions for Imitation Learning. In Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics (AISTATS), 661–668. PMLR 9.

Ross, S.; Gordon, G. J.; and Bagnell, J. A. 2011. A Reduction ofImitation Learning and Structured Prediction to No-Regret Online Learning. In Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics (AISTATS), 627–635. PMLR 15. arXiv:1011.0686.

Spencer, J.; Choudhury, S.; Venkatraman, A.; Ziebart, B.; and Bagnell, J. A. 2021. Feedback in Imitation Learning: The Three Regimes of Covariate Shift. arXiv:2102.02872.

Sun, J.; Zhang, W.; Qi, Z.; Ren, S.; Liu, Z.; Zhu, H.; Sun, G.; Jin, X.; and Chen, Z. 2026a. VLA-JEPA: Enhancing Vision-Language-Action Model with Latent World Model. arXiv:2602.10098.

Sun, Z.; Guo, Y.; Sun, H.; Wang, L.; Lu, W.; Ji, J.; Ji, S.; Xiong, J.; and Meng, Z. 2026b. Pre-VLA: Preemptive Runtime Verification for Reliable Vision-Language-Action and World-Model Rollouts. arXiv:2605.22446.

Torne, M.; Pertsch, K.; Walke, H.; Vedder, K.; Nair, S.; Ichter, B.; Ren, A. Z.; Wang, H.; Tang, J.; Stachowicz, K.; et al. 2026. MEM: Multi-Scale Embodied Memory for Vision Language Action Models. arXiv:2603.03596.

Wang, H.; Zhang, G.; Yan, Y.; Kompella, R. R.; and Liu, G. 2026a. VLA Knows Its Limits: Adaptive Execution Horizons for Robot Policies. arXiv:2602.21445.

Wang, H.; Zhang, G.; Yan, Y.; Shang, Y.; Kompella, R. R.; and Liu, G. 2026b. Real-Time Robot Execution with Masked Action Chunking. arXiv:2601.20130.

Wang, R.; Zhang, Y.; Lin, J.; Luo, K.; Wang, J.; Wang, Z.; and Qi, X. 2026c. When to Trust Imagination: Adaptive Action Execution for World Action Models. arXiv:2605.06222.

Wang, T.; Hou, H.; Hu, Y.; Liu, Y.; Li, Q.; Jiang, Y.; Wang, Y.; Ma, C.; Wang, R.; and Gao, Y. 2026d. When Does Legacy Data Start to Help? Emergent Transfer in Cross-Configuration Robot Learning. arXiv:2607.25593.

Wang, X.; Zhu, Z.; Huang, G.; Wang, B.; Chen, X.; and Lu, J. 2024. WorldDreamer: Towards General World Models for Video Generation via Predicting Masked Tokens. arXiv:2401.09985.

Ye, A.; Wang, B.; Ni, C.; Huang, G.; Zhao, G.; Li, H.; Li, H.; Li, J.; Lv, J.; Liu, J.; Cao, M.; Li, P.; Deng, Q.; Mei, W.; Wang, X.; Chen, X.; Zhou, X.; Wang, Y.; Chang, Y.; Li, Y.;

Zhou, Y.; Ye, Y.; Liu, Z.; and Zhu, Z. 2026. GigaWorld-Policy: An Eficient Action-Centered World–Action Model. arXiv:2603.17240.

Yuan, H.; Liang, Z.; Chen, A.; Wang, Y.; Li, H.; Lin, P.; Huang, Y.; Lei, Z.; Zhang, T.; Zhang, J.; Zhang, J.; Fan, J.; Zhou, G.; Peng, Q.; Lv, C.; Chen, X.; Yang, A.; Huang, F.; Lin, J.; Liu, D.; Zhou, J.; Wu, C.; and Chen, X.-H. 2026. Qwen-RobotManip Technical Report: Alignment Unlocks Scale for Robotic Manipulation Foundation Models. arXiv:2606.17846.

Zeng, Y.; Ye, M.; Chen, Y.; Shentu, Y.; Wu, P.; Yan, Z.; and Li, Z. 2026. KEMO: Event-Driven Keyframe Memory for Long-Horizon Robot Manipulation with VLA Policies. arXiv:2606.23589.

Zhang, H.; Lu, Y.; Wang, B.; Kang, X.; Kuo, Y.-L.; Cheng, Z.; Wang, M.; and Jenkins, O. C. 2026. Foresight: Failure Detection for Long-Horizon Robotic Manipulation with Action-Conditioned World Model Latents. arXiv:2606.23085.

Zhang, W.; Liu, H.; Qi, Z.; Wang, Y.; Yu, X.; Zhang, J.; Dong, R.; He, J.; Lu, F.; Wang, H.; et al. 2025. DreamVLA: A Vision-Language-Action Model Dreamed with Comprehensive World Knowledge. arXiv:2507.04447.

Zhao, T. Z.; Kumar, V.; Levine, S.; and Finn, C. 2023. Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware. arXiv:2304.13705.

Zhong, X.; Zheng, H.; Zhao, C.; Lv, T.; Fan, H.; Wang, B.; Liu, Y.; Gao, L.; Liao, Z.; Luo, L.; Zhao, C.; and Cai, Y. 2026. ForgeDrive: Bidirectional Cross-Conditioning for Unified Visual-Action Generation in Autonomous Driving. arXiv:2606.31226.

Zhou, E.; Su, Q.; Chi, C.; Zhang, Z.; Wang, Z.; Huang, T.; Sheng, L.; and Wang, H. 2024a. Code-as-Monitor: Constraint-aware Visual Programming for Reactive and Proactive Robotic Failure Detection. arXiv:2412.04455.

Zhou, G.; Pan, H.; LeCun, Y.; and Pinto, L. 2024b. DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning. arXiv:2411.04983.

## Appendix

## A Execution-Semantics Illustration

Figure S1 gives a schematic episode that illustrates the execution semantics formalized around Eq. 1 and the sufixrewriting mechanism in Eqs. 9–10. Both panels share the same horizontal axis (control steps 0–50, one chunk lengt H) and receive the same perturbation: at step 15, the carried object slips inside the gripper. Panel (a) shows open-loop chunked execution: the policy receives no new high-level visual input within the chunk, the deviation accumulates from step 15 onward, and the failure can surface only at the next policy call at step 50. Panel (b) shows the same episode under CheckVLA. The world model predicts near-term features conditioned on the committed actions; the prediction– observation discrepancies are aggregated by the temporally causal risk head into the risk score $r _ { t }$ (red curve), which rises after the perturbation and crosses the conformally calibrated, time-varying threshold $\delta _ { t }$ (gray dashed line) at $\bar { t } ^ { * }$ (step 20), triggering a replan. A trigger does not mean immediate correction: one inference of the same policy takes $d _ { \mathrm { l a t } } ~ = ~ 3$ control steps (amber band), during which old-chunk steps 20–22 still execute and form the irreversible prefix. Here $d _ { \mathrm { l a t } }$ is a fixed deployment schedule chosen to cover the median measured latency (286 ms at the 100 ms control period; Table S18): the switch is scheduled $d _ { \mathrm { l a t } }$ ticks after the trigger, an earlier-finishing call waits for the scheduled tick, and a cal that overruns deploys at the first tick after completion with the elapsed replacement positions discarded (the efective delay $d _ { \mathrm { e f f } }$ defined immediately after Eq. 1), so latency jitter only shortens the deployed sufix. The reported experiments impose the nominal schedule uniformly, and the sweep in Table S6 brackets slower calls up to $d _ { \mathrm { l a t } } { = } 1 0$ . The corresponding prefix positions of the new chunk are hard-constrained to these executed actions (hatched cells), reducing the command discontinuity at the switch; the new chunk deploys from $h = d _ { \mathrm { l a t } }$ (step 23, green), and the remainder of the old chunk (gray hatching) is replaced and never executes. The bottom decay strip shows the per-position guidance weight W on the flagged old chunk: its entry scale $w _ { 0 } ( e _ { t ^ { * } } ) < 1$ is set by the standardized exceedance at the trigger and decays as $\mathrm { e x p } ( - \lambda _ { m } ( h - d _ { \mathrm { l a t } } ) )$ ; the larger the exceedance, the smaller $w _ { 0 }$ and the stronger the permitted correction. The first de ployed position therefore departs from the flagged reference instead of copying it. Step indices are illustrative values from the experimental configuration $( H { = } 5 0 , d _ { \mathrm { l a t } } { = } 3 )$ ; the real distributions of perturbation types, trigger times, and lead times are reported in supplementary Sec. F.

## B Execution-Loop Pseudocode

Algorithm S1 consolidates the monitoring, triggering, suppression, and re-anchoring rules formalized across Eqs. 1–10 into a single loop; every equation reference below resolves to a labeled equation in the main text.

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm S1 CheckVLA execution loop (one episode)

Require: policy  $\pi_{\theta}$ , frozen encoder  $\phi$ , world model  $\Psi$ , risk head R, calibrated  $(\mu_{r,t}, \tilde{\sigma}_{r,t}, \hat{q}_{\alpha})$ , span k, scheduled latency  $d_{lat}$ , suppression length  $s_{min}$ 

1: commit  $A_{t} \sim \pi_{\theta}(\cdot \mid c_{t}, s_{t})$  (main Eq. 1); anchor at the current observation

2: while episode not finished do

3: on each new observation  $o_{t}: z_{t} \leftarrow \phi(o_{t})$ ; write the corresponding pooled policy feature  $F_{t}$  to the keyframe bank  $\rho_{t}$  if the pause and diversity criteria hold

4: retrieve the prediction  $\hat{z}_{t|t-\ell(t)}$  issued at the latest valid anchor; standardize the discrepancy by span (main Eq. 3) and form the tuple  $x_{t}$  (main Eq. 4)

5:  $r_{t} \leftarrow R(x_{t-w+1:t}, m(\rho_{t}))$ ;  $\delta_{t} \leftarrow \mu_{r,t} + \hat{q}_{\alpha} \tilde{\sigma}_{r,t}$  (main Eqs. 5–6)

6: if  $r_{t} &gt; \delta_{t}$ , no rewrite in progress, and no post-deployment suppression then

7: compute the standardized exceedance  $e_{t^{*}}$  (main Eq. 8); rewrite with  $\pi_{\theta}$ : clamp positions  $h &lt; d_{lat}$  to the executing prefix at every integration step and guide the suffix with  $W_{h,m}$  (main Eqs. 9–10)

8: at the scheduled tick deploy from  $h = d_{lat}$ ; if inference overruns, set  $d_{eff}$  to elapsed ticks, deploy from  $h = d_{eff}$ , and discard earlier positions; invalidate superseded predictions, suppress triggers for  $s_{min}$  steps, and re-anchor at the next observation

9: else

10: anchor at  $z_{t}$ ; roll out  $\hat{z}$  for the next k steps conditioned on the remaining committed actions (main Eq. 2)

11: end if

12: at a regular chunk boundary: commit the next chunk with bank  $\rho_{t}$  (e=0 transition guidance)

13: end while
</div>

## C Decoupled Action Experts and Training Details

Action generation uses decoupled mobility and manipulation experts (Figure S2). The mobility expert receives basestate and base-action tokens, while the manipulation expert receives arm–gripper state and action tokens; each expert has separate query/key/value projections, a feed-forward network, and a denoising head. Their projected tokens interact in joint self-attention over the shared prefix $c _ { t }$ , allowing coordination without additional cross-expert connections; the design follows evidence that action-subspace decoupling can reduce gradient interference (Chen et al. 2026).

For $e \in \{$ {mobility, manip}, we sample a flow-integration time $u \sim \tilde { \mathcal { U } } ( 0 , 1 )$ and form noisy actions by linear interpolation:

$$
\tilde {A} _ {t} ^ {e, u} = u A _ {t} ^ {e} + (1 - u) \epsilon^ {e}, \qquad \epsilon^ {e} \sim \mathcal {N} (0, I),\tag{S1}
$$

and training minimizes the conditional flow-matching loss

$$
\mathcal {L} _ {\mathrm{AE}} = \sum_ {e} \mathbb {E} _ {u, \epsilon^ {e}} \left\| v _ {\theta} ^ {e} \big (\tilde {A} _ {t} ^ {e, u}, s _ {t} ^ {e}; c _ {t}, u \big) - \big (A _ {t} ^ {e} - \epsilon^ {e} \big) \right\| ^ {2},\tag{S2}
$$

where $\boldsymbol { s } _ { t } ^ { e }$ is the corresponding proprioceptive state and u is injected through adaptive layer normalization. A stop-gradient isolates the VLM backbone from the flow-matching objective. At inference, the experts synchronously integrate their vector fields under the shared prefix; a replan trigger invokes the same network with the guided flow integration in Eqs. 9– 10.

(a) Open-loop chunked execution (baseline)  
![](images/ab36eb61f6b0b4104e4dbb59afe80d2f872353e7d8e87f1f96677b25be2b8347.jpg)

(b) CheckVLA: in-chunk verification and latency-consistent suffix rewriting  
![](images/85962fa005afbcd5116828c17e2ea6aa4336e24bd740dfde578226481ce0455e.jpg)  
Figure S1: Sufix rewriting under inference latency. (a) Open-loop chunked execution: a perturbation at step 15 is not observed at the policy level, and the deviation persists until the next policy call at the chunk boundary; the task fails. (b) CheckVLA: the action-conditioned risk score $r _ { t }$ crosses the calibrated threshold $\delta _ { t }$ at $t ^ { * }$ (step 20) and triggers replanning by the same policy. While inference runs $( d _ { \mathrm { l a t } } { = } 3$ control steps), the flagged chunk keeps executing (irreversible prefix); the new chunk is hard-constrained to match these executed actions and deploys from $h { = } d _ { \mathrm { l a t } }$ (step 23), replacing the flagged remainder. The guidance weight starts at $w _ { 0 } ( e _ { t ^ { * } } ) < 1$ and decays along the sufix: a larger standardized exceedance permits a stronger correction.

## D Verifier Training and Calibration Details

The world model Ψ applies block-causal attention over the interleaved feature and action tokens, with block i predicting frame i+1. Training first minimizes a teacher-forcing objective with a Huber loss against frozen encoder targets,

$$
\mathcal {L} _ {\Psi} ^ {\mathrm{tf}} = \sum_ {i = 1} ^ {k} \ell_ {\mathrm{Huber}} \big (\hat {z} _ {t + i}, \operatorname{sg} (z _ {t + i}) \big),\tag{S3}
$$

and then replaces the prediction context with $\Psi ^ { \ast } \mathbf { s }$ own rollouts so that training matches the autoregressive context used online. Predictions are stored by anchor and target time, and targets that cross a chunk boundary are generated after the next chunk is committed; no pixel decoding is used at any stage. At inference, the monitor retrieves the prediction from the most recent anchor whose rollout is available and whose conditioning actions have not been superseded. Under nominal timing (monitor p95 of 16.4 ms against the 100 ms control period; Table S18) this is the previous observation, so the retrieved span is ℓ = 1; spans up to k arise when monitor computation overruns a control period, an observation drops, or a chunk switch invalidates predictions, and the span-specific statistics of main Eq. 3 keep these regimes on a common scale.

The risk head is trained on three trajectory classes: nominally successful episodes, natural failures of the openloop backbone, and physics-level perturbations—impulses to objects or the base, transient joint ofsets, and object displacements—annotated by their onset $\tau _ { \mathrm { o n } } .$ . The objective applies binary cross-entropy to the softmax-pooled risk within $[ \tau _ { \mathrm { o n } } , \tau _ { \mathrm { o n } } { + } \Delta ]$ ], penalizes risk before onset, and masks gradients after task failure; nominally successful episodes and benign deviations (after which the task still succeeds) serve as negatives throughout. Supervision is therefore restricted to pre-failure evidence, and perturbations act at the physics level; they are never injected directly into the discrepancy sequence.

The functional conformal threshold in Eqs. 6–7 is constructed as follows. Per-step risk statistics $\left( \mu _ { r , t } , \sigma _ { r , t } \right)$ are fitted on nominally successful shadow-mode trajectories of the same training seed, disjoint from the n calibration trajectories, with $\tilde { \sigma } _ { r , t } = \operatorname* { m a x } ( \sigma _ { r , t } , \varepsilon )$ . Each calibration trajectory i receives the score $\kappa ^ { ( i ) } = \operatorname* { s u p } _ { t } \big ( r _ { t } ^ { ( i ) } - \mu _ { r , t } \big ) / \tilde { \sigma } _ { r , t }$ , and $\hat { q } _ { \alpha }$ is the $\lceil ( n { + } 1 ) ( 1 { - } \alpha ) \rceil -$ th order statistic of these scores, set to +∞ when the index exceeds n. Every score-shaping statistic, including discrepancy-span statistics and temporal normalization, and every online retrieval and memory rule is fixed before the conformal quantile is computed; exchangeability between calibration and deployment episodes under the same fixed pipeline then yields the first-intervention episode-level bound in Eq. 7.

![](images/a4a6bc167b46f60859a8e17790e583d7a4b17737a603fb71c3767afd92e9d23a.jpg)  
Figure S2: Decoupled action experts. The mobility and manipulation experts keep separate QKV projections, feedforward networks, and denoising heads, and interact only through joint self-attention over the shared VLM prefix tokens, repeated over $N$ blocks. Each expert denoises its own noisy action tokens $\tilde { A } _ { t } ^ { e , u } ~ ( \mathrm { E q . } ~ \mathrm { S 1 } )$ into the committed subspace actions $A _ { t } ^ { e }$

During regular chunk transitions, reference guidance uses $e \ = \ 0 .$ , hence $w _ { 0 } = 1$ , with a slower positional decay to prioritize continuity. After a repair switch, predictions conditioned on superseded actions are invalidated and the next observation forms a new anchor; further triggers are suppressed while a rewrite is in progress and for $s _ { \mathrm { m i n } }$ steps after deployment, and the reference weight $W$ is zero beyond the end of the remaining reference actions. Table S20 lists the complete hyperparameter configuration.

## E Episodic Context Implementation

A post-replan invocation sees only the current observation, while evidence of completed stages may have left the field of view. CheckVLA therefore maintains an event-driven keyframe bank $\boldsymbol { B _ { t } } = \{ ( F _ { j } , t _ { j } ) \} _ { i = 1 } ^ { m _ { t } }$ of real-observation features that persists across chunks and repairs and is cleared only at episode end. A non-learned two-stage filter proposes writes: delay-confirmed local minima ofthe meanjoint-space displacement $\begin{array} { r } { \overline { { \Delta q } } _ { t } = \frac { 1 } { w _ { s } } \sum _ { j = t - w _ { s } + 1 } ^ { t } \lVert q _ { j } - q _ { j - 1 } \rVert _ { 2 } } \end{array}$ <sub>2</sub> mark candidate pauses, and candidates too similar or too close in time to the latest stored frame are rejected. Accepted entries store pooled policy-encoded tokens $F _ { j }$ with timestamps; positional indices are rebuilt from bank order at read time.

The policy needs the evidence itself to choose an action, whereas the risk head needs only enough context to judge whether a discrepancy is innocuous. The policy therefore fuses the full bank once per invocation through gated crossattention,

$$
\begin{array}{r l} & X _ {t} ^ {\prime} = \mathrm{CrossAttn} \big (X _ {t}, \tilde {\mathcal {B}} _ {t} \big), \\ & \hat {X} _ {t} = X _ {t} + \sigma \big (g ([ X _ {t}; X _ {t} ^ {\prime} ]) \big) \odot X _ {t} ^ {\prime}, \end{array}\tag{S4}
$$

where $g$ is negatively initialized and $\hat { X } _ { t }$ preserves the token shape expected by the VLM. The risk head reads only $m ( \rho _ { t } ) = \left( t - t _ { \mathrm { l a s t } } , \ m _ { t } , \ \bar { F } _ { \mathrm { l a s t } } \right)$ . Both paths are trained with real banks reconstructed ofline by the same filter. Consequently, the factorial memory study retrains each policyreader variant and independently recalibrates each changed risk reader; test-time zeroing is not treated as a valid substitute.

## F Evaluation Protocol and Data Separation

The supplementary evaluation is designed to test one claim: under the same action backbone and comparable intervention budgets, action-conditioned verification should identify useful intervention times, and latency-consistent sufix rewriting should turn those warnings into recoveries without increasing harm on episodes that would otherwise succeed. This claim is bounded to the RoboCasa365 simulator and to the task, scene, and perturbation distributions described below; it does not establish hardware safety.

Table S1 records the separation between optimization, model selection, calibration, and final evaluation. The policy and its action experts are trained only on the oficial Human300 demonstrations. Auxiliary trajectories for the world model and risk head are generated from training-side tasks and scenes and are accounted for separately from Human300 policy training. Splits are assigned by task–scene– initialization group before rollout generation, preventing near-duplicate trajectories from crossing partitions. Neither Composite-Unseen tasks nor target-suite scenes are used to train the policy, world model, or risk head.

The perturbation study uses the 12 composite tasks visualized in Figures S8–S10. It balances object or base impulses, transient joint ofsets, and moved-object interventions across navigation, pre-contact, post-grasp transport, and applianceinteraction phases. Paired variants share the task, scene, initial state, perturbation type and onset, environment seed, and policy-sampling seed. Detector-only comparisons replay the same shadow-mode trajectories. Rewrite comparisons restore the same simulator snapshot at the first trigger and reuse common random numbers, so they change the repair rule but neither the evidence available at the trigger nor the pre-trigger history.

The allocation uses 3 independently trained seeds, 10 clean episodes per task and seed, and 10 paired perturbation episodes per task, perturbation family, and seed, with severity and onset phase balanced within each block. Three nominal pools serve distinct roles and never mix: 400 shadow-mode nominal successes per training seed, generated by that seed’s own policy and monitor, fit the conformal calibration—each checkpoint is calibrated independently, so exchangeability is with respect to deployment of the same checkpoint— a disjoint pool of 50 nominal shadow-mode episodes per perturbation-study task and seed (1,800 episodes) estimates empirical episode FWER, and the 1,500-episode locked clean suite reports benchmark success only. Table S2 breaks the full system’s headline metrics down by training seed. Task-success uncertainty is computed by a 95% hierarchical paired bootstrap over tasks and then episodes. Episode FWER uses an exact binomial interval, and paired success changes use an exact McNemar test. Hyperparameters are selected once on validation data; every detector variant is then independently recalibrated at target episode-level $\alpha = 0 . 0 5$ before the locked test is read.

Table S1: Roles of the data partitions. The oficial clean benchmark aggregate is never mixed with the controlled perturbation study.

<table><tr><td>Partition</td><td>Allowed data</td><td>Permitted use</td><td>Leakage guard</td></tr><tr><td>Policy training</td><td>Official Human300 demonstrations</td><td>Optimize the VLA and decoupled action experts</td><td>No target-suite trajectory or injected perturbation</td></tr><tr><td>Monitor training</td><td>Training-side nominal and physics-perturbed rollouts</td><td>Optimize the rolling predictor and causal risk head</td><td>No validation, calibration, or target group</td></tr><tr><td>Validation</td><td>Held-out training-side task-scene groups</td><td>Select discrepancy, span, window, suppression, guidance, and periodic-replan settings</td><td>No gradient update and no final conformal quantile</td></tr><tr><td>Conformal calibration</td><td>Nominal successful shadow-mode episodes</td><td>Fit span statistics, temporal normalization, and the functional conformal quantile</td><td>Disjoint from validation and both locked tests</td></tr><tr><td>Locked clean test</td><td>Official 50-task target suite</td><td>Report A-S, C-S, C-U, and task-weighted success</td><td>Evaluation only; no post-test tuning</td></tr><tr><td>Locked perturbation test</td><td>Preregistered composite-task sub-set with physical perturbations</td><td>Detector, fixed-trigger repair, latency, and failure analyses</td><td>Perturbation schedule fixed before unblinding variants</td></tr></table>

Table S2: Per-seed results for full CheckVLA. Mean matches the aggregates reported in Tables 3 and 4.

<table><tr><td>Seed</td><td>Clean Average (%)</td><td>Perturbed success (%)</td><td>Timely recall (%)</td><td>Episode FWER (%)</td></tr><tr><td>Seed 1</td><td>35.6</td><td>48.2</td><td>77.1</td><td>5.2</td></tr><tr><td>Seed 2</td><td>36.9</td><td>46.9</td><td>79.0</td><td>4.5</td></tr><tr><td>Seed 3</td><td>35.8</td><td>47.7</td><td>77.6</td><td>4.7</td></tr><tr><td>Mean</td><td>36.1</td><td>47.6</td><td>77.9</td><td>4.8</td></tr></table>

## G Metric Definitions

Let $\mathcal { N }$ denote shadow-mode episodes that succeed under nominal execution, $T _ { i }$ their horizons, and $t _ { i } ^ { * } = \operatorname* { i n f } \{ t : r _ { i , t } >$ $\delta _ { i , t } \}$ the first alarm $( t _ { i } ^ { * } = \infty$ when no alarm occurs). The empirical episode-level family-wise error rate (episode FWER) is

$$
\widehat {\mathrm{FWER}} _ {\mathrm{ep}} = \frac {1}{| \mathcal {N} |} \sum_ {i \in \mathcal {N}} \mathbb {1} [ \exists t \leq T _ {i}: r _ {i, t} > \delta_ {i, t} ].\tag{S5}
$$

This is the probability of at least one unnecessary first intervention on an episode that would otherwise succeed; a step-wise false-positive rate is not a substitute because it ignores repeated testing over variable horizons.

For each disrupted reference episode i that fails without intervention, $\tau _ { \mathrm { o n } , i }$ is the annotated onset of the observable deviation and $\tau _ { \mathrm { i r r e v } , i }$ is the earliest time after which no tested sufix repair can restore task success. Only the onset is a human annotation; $\tau _ { \mathrm { i r r e v } , i }$ is operational, fixed by repair outcomes under the tested rewrite rules, so timeliness is not judged against an annotator’s opinion of recoverability. (In the natural-execution audit, where deviations are not injected, blinded annotators mark both times as described there.) We call the interval before $\tau _ { \mathrm { i r r e v } , i }$ the episode’s empirical recovery window; irreversibility always refers to the close of this window under the tested repair set, not to physical impossibility. A trigger is timely only if it occurs no earlier than onset and its repaired chunk can deploy before irreversibility. Timely recall and efective lead are therefore

$$
\widehat {\mathrm{Recall}} _ {\mathrm{timely}} = \frac {1}{| \mathcal {F} |} \sum_ {i \in \mathcal {F}} \mathbb {1} \left[ \begin{array}{c} \tau_ {\mathrm{on}, i} \leq t _ {i} ^ {*}, \\ t _ {i} ^ {*} + d _ {\mathrm{lat}, i} <   \tau_ {\mathrm{irrev}, i} \end{array} \right],\tag{S6}
$$

$$
\ell_ {i} = \tau_ {\mathrm{irrev}, i} - (t _ {i} ^ {*} + d _ {\mathrm{lat}, i}),\tag{S7}
$$

where $\mathcal { F }$ contains the disrupted reference episodes. We report the median and interquartile range of $\ell _ { i }$ among timely alarms and retain premature, missed, and late alarms as failures in Eq. S6. Premature-alarm frequency, $\begin{array} { r } { \vert \mathcal { F } \vert ^ { - 1 } \sum _ { i } \mathcal { H } [ t _ { i } ^ { * } < \tau _ { \mathrm { o n } , i } ] , } \end{array}$ is reported separately.

Repair quality is measured with paired counterfactual outcomes. Let $y _ { i } ^ { 0 }$ be success from the stored trigger snapshot under the reference continuation and $y _ { i } ^ { m }$ success under repair method m. Conditional rescue and harm rates are

$$
\operatorname{Rescue} (m) = \frac {\sum_ {i} \mathbb {1} [ y _ {i} ^ {0} = 0 \wedge y _ {i} ^ {m} = 1 ]}{\sum_ {i} \mathbb {1} [ y _ {i} ^ {0} = 0 ]},\tag{S8}
$$

$$
\operatorname{Harm} (m) = \frac {\sum_ {i} \mathbb {1} [ y _ {i} ^ {0} = 1 \wedge y _ {i} ^ {m} = 0 ]}{\sum_ {i} \mathbb {1} [ y _ {i} ^ {0} = 1 ]}.\tag{S9}
$$

Rescue and harm isolate a single repair: the branch executes rule m once with subsequent triggers disabled. Perturbed success instead continues the full closed loop from the same snapshot with re-intervention enabled (audited in Table S17), so a rule can gain more in perturbed success than its oneshot rescue–harm gap alone implies. We additionally report task success, policy invocations per episode, and the switch discontinuity $\| a _ { \mathrm { n e w } , d _ { \mathrm { l a t } } } - a _ { \mathrm { o l d } , j + d _ { \mathrm { l a t } } } \| _ { 2 }$ . Together, rescue and harm distinguish useful recovery from indiscriminate replanning that merely trades one set of failures for another.

## H Action-Conditioned Detector Ablations

We isolate the verifier with the action backbone, downstream rewriter, test trajectories, and target episode FWER held fixed. Each detector variant is retrained when applicable instead of being disabled at test time, and its discrepancy statistics and threshold are refit on its own calibration split. The action-shufled row is a training-time negative control: it preserves action-token marginals and architecture while breaking the correspondence between a committed action and its predicted consequence. The constant-threshold row replaces the time-varying threshold of the main method with a single validation-tuned scalar; unlike every other row, it is not conformally calibrated, so it measures the combined efect of removing the functional shape and the conformal calibration.

The cross-family comparison additionally evaluates four policy-side signals. For the Monte Carlo entropy proxy and sampling disagreement, the policy draws the same Q normalized action chunks per decision; entropy is estimated with a diagonal-Gaussian approximation, whereas disagreement is their mean pairwise squared distance. Denoising/flowpath variance is computed from clean-action estimates at validation-selected integration times. The frozen-policy feature probe is a two-layer causal classifier trained on policy features without updating the VLA. The primary comparison evaluates all policy signals at commit time. Observationrefreshed variants rerun the required policy computation after each new observation and report the resulting compute separately (Table S4). An average episode lasts 390 control steps at 10 Hz, so observation-rate refreshing adds 382.2 policy evaluations per episode, and its per-score p95 latency exceeds the 100 ms control period; these variants are ofline upper bounds, not deployable monitors.

The verifier-control rows in Table S3 separate alarm calibration from alarm usefulness. Shufling action–future training pairs sharply degrades event ranking and timeliness. Rolling re-anchoring recovers lead time relative to a fullchunk rollout, and temporal aggregation improves over an instantaneous residual. The time-varying conformal rule improves timely recall while bringing empirical episode FWER back to the prescribed operating point, unlike the validationtuned constant threshold. Among the policy-side rows, the frozen feature probe is strongest, yet the full verifier improves timely recall by 9.3 points and perturbed success by 3.9 points at a comparable target FWER.

## I Inference-Time Action–Consequence Binding

The training-time shufled control does not by itself show that the trained predictor uses its action input at inference. We therefore freeze the full world model and restore heldout simulator snapshots. From each snapshot, we execute the factual action and four physically plausible counterfactuals under a common physics seed: a circular temporal shift, a gripper-state flip, a phase-matched mobility replacement, and a phase-matched manipulation replacement. Each replacement receives a consistent nominal proprioceptive rollout. For prediction u and realized future $v ,$ , we compute the span-averaged feature distance $D ( u , v )$ . The assignment margin compares diagonal matching, $D ( 0 , 0 ) + D ( v , v )$ , with the crossed assignment, $D ( 0 , v ) + \dot { D } ( v , 0 )$ . Positive margins and pairwise accuracy above 50% in Table S5 indicate action–future binding, not observation-only sensitivity. Both the action tokens and the nominal proprioceptive rollout sˆ (itself a deterministic function of the actions) carry action information, so the audit certifies binding of the combined action conditioning without separating these two pathways.

## J Fixed-Trigger Rewriting and Latency

To prevent detector quality from confounding repair quality, every row in Table S6 starts from the same first-trigger simulator snapshots. Every one of the 1,080 injected episodes produces at least one threshold crossing, so the branch set covers the full cohort; the online deployed-rewrite rate in Table S17 (86.2%) is lower because online deployment additionally respects the trigger-suppression window and the empty-sufix rule. The flagged chunk, standardized exceedance, observation, episodic bank, and policy-sampling noise are shared across variants. The boundary baseline ignores the trigger until the next regular call. The unconstrained baseline generates a complete chunk and discards its already elapsed prefix only after sampling; in contrast, hard-prefix variants overwrite the irreversible prefix at every flow-integration step. Because the overwritten values do not match the training-time noise level at intermediate integration times, the clamp is an empirical projection rather than exact sampling from the conditional sufix distribution; the RTC (Black, Galliker, and Levine 2025) row provides the principled prefix-preserving inpainting alternative under the same trigger, and the switch-jump column reports the residual discontinuity of each rule. The shufled-exceedance control preserves the marginal distribution of repair strengths but breaks its pairing with the current deviation. The validation-tuned fixed weight is $W { = } 0 . 5$

The paired results attribute diferent roles to the two repair constraints. Hard prefixing reduces the deployment discontinuity and harm relative to post-hoc prefix disposal, whereas a full rewrite with $W { = } 0$ discards all reference guidance and over-corrects otherwise recoverable trajectories. A fixed guidance weight improves this trade-of, but shufling exceedance produces little additional gain; only the correctly paired $W ( e )$ improves rescue while lowering harm. Under the shared trigger, adaptive guidance improves nominallatency success by 2.0 points over RTC and lowers harm by 1.3 points. Performance decreases with inference latency for every in-chunk method because a longer irreversible prefix leaves fewer executable actions to change. The full method retains the largest advantage at the longest tested latency, but the remaining decline makes clear that verification cannot recover an episode after the useful intervention window has closed. Figure S3 plots the sweep: every in-chunk method declines as the irreversible prefix grows, and adaptive guidance

Table S3: Detector comparison at a common target episode FWER of $\alpha = 0 . 0 5$ . Each detector is calibrated independently on nominal successful episodes and evaluated on the same shadow trajectories. Event AUPRC treats each injected deviation as one event; efective lead includes deployment latency, and perturbed success uses the same downstream rewriter for every detector.

<table><tr><td>Detector variant</td><td>Episode FWER(%)</td><td>EventAUPRC ↑</td><td>Timely recall(%) ↑</td><td>Effective lead(steps) ↑</td><td>Perturbed success(%) ↑</td></tr><tr><td colspan="6">Panel A: Cross-family detector comparison</td></tr><tr><td>Policy MC action-entropy proxy, commit-time</td><td>5.0</td><td>0.512</td><td>58.7</td><td>7.8</td><td>38.9</td></tr><tr><td>Policy denoising/flow-path variance,commit-time</td><td>5.2</td><td>0.548</td><td>61.4</td><td>8.4</td><td>40.2</td></tr><tr><td>Policy sampling disagreement, commit-time</td><td>4.9</td><td>0.586</td><td>64.8</td><td>9.1</td><td>41.5</td></tr><tr><td>Frozen-policy feature probe, commit-time</td><td>5.1</td><td>0.621</td><td>68.6</td><td>10.0</td><td>43.7</td></tr><tr><td>Observation-only world predictor</td><td>5.3</td><td>0.442</td><td>48.6</td><td>6.1</td><td>35.2</td></tr><tr><td>Action-shuffled world model, retrained</td><td>5.1</td><td>0.318</td><td>37.9</td><td>4.2</td><td>31.0</td></tr><tr><td>Full action-conditioned verifier</td><td>4.8</td><td>0.764</td><td>77.9</td><td>13.2</td><td>47.6</td></tr><tr><td colspan="6">Panel B: Internal verifier mechanisms</td></tr><tr><td>Observation-only world predictor</td><td>5.3</td><td>0.442</td><td>48.6</td><td>6.1</td><td>35.2</td></tr><tr><td>Action-shuffled world model, retrained</td><td>5.1</td><td>0.318</td><td>37.9</td><td>4.2</td><td>31.0</td></tr><tr><td>Action conditioned, no rolling re-anchor</td><td>5.4</td><td>0.561</td><td>62.8</td><td>9.2</td><td>38.7</td></tr><tr><td>Rolling predictor, instantaneous score</td><td>5.0</td><td>0.634</td><td>66.1</td><td>10.1</td><td>41.2</td></tr><tr><td>Causal risk head, constant threshold</td><td>7.1</td><td>0.701</td><td>71.5</td><td>11.4</td><td>45.0</td></tr><tr><td>Full: time-varying conformal threshold</td><td>4.8</td><td>0.764</td><td>77.9</td><td>13.2</td><td>47.6</td></tr></table>

Table S4: Sensitivity to refreshing policy-side signals after new observations. Commit-time variants reuse the score produced with the dispatched chunk. Observation-refreshed variants rerun the required policy computation at the monitoring rate, with the added VLA evaluations and p95 score latency reported explicitly.

<table><tr><td>Signal</td><td>Update mode</td><td>Extra VLA evals / ep.</td><td>Score p95 (ms)</td><td>Episode FWER (%)</td><td>Timely recall (%)↑</td><td>Perturbed success (%)↑</td></tr><tr><td>MC action-entropy proxy</td><td>Commit-time</td><td>0.0</td><td>338</td><td>5.0</td><td>58.7</td><td>38.9</td></tr><tr><td>MC action-entropy proxy</td><td>Observation-refreshed</td><td>382.2</td><td>338</td><td>5.2</td><td>69.5</td><td>44.0</td></tr><tr><td>Sampling disagreement</td><td>Commit-time</td><td>0.0</td><td>338</td><td>4.9</td><td>64.8</td><td>41.5</td></tr><tr><td>Sampling disagreement</td><td>Observation-refreshed</td><td>382.2</td><td>338</td><td>5.1</td><td>72.3</td><td>45.0</td></tr><tr><td>Frozen policy-feature probe</td><td>Commit-time</td><td>0.0</td><td>6.8</td><td>5.1</td><td>68.6</td><td>43.7</td></tr><tr><td>Frozen policy-feature probe</td><td>Observation-refreshed</td><td>382.2</td><td>344</td><td>5.0</td><td>73.1</td><td>45.4</td></tr><tr><td>Full action-conditioned veri-fer</td><td>Observation-rate moni-tor</td><td>0.0</td><td>16.4</td><td>4.8</td><td>77.9</td><td>47.6</td></tr></table>

Table S5: Inference-time action–consequence binding of the frozen world model. Each candidate is generated from the same simulator snapshot and rolled out under a common physics seed. Accuracy is the fraction of pairs whose diagonal action–future assignment has lower cost than the crossed assignment.

<table><tr><td>Counterfactual action</td><td>Anchor pairs</td><td>Predicted-future separation ↑</td><td>Realized-future separation ↑</td><td>Assignment margin ↑</td><td>2AFC binding accuracy (%) ↑</td></tr><tr><td>Within-chunk circular temporal shift</td><td>720</td><td>0.214</td><td>0.237</td><td>0.118</td><td>78.6</td></tr><tr><td>Gripper open/close flip</td><td>412</td><td>0.181</td><td>0.205</td><td>0.094</td><td>74.8</td></tr><tr><td>Phase-matched mobility-action replacement</td><td>720</td><td>0.268</td><td>0.301</td><td>0.143</td><td>82.1</td></tr><tr><td>Phase-matched manipulation-action replacement</td><td>720</td><td>0.246</td><td>0.279</td><td>0.131</td><td>80.4</td></tr><tr><td>Pooled counterfactuals</td><td>2572</td><td>0.233</td><td>0.262</td><td>0.125</td><td>79.5</td></tr></table>

Table S6: Fixed-trigger repair comparison. Rescue, harm, and switch jump are one-shot paired outcomes at the nominal latency of three control steps, with subsequent triggers disabled; perturbed success continues the full closed loop from the same snapshot with re-intervention enabled. The latency sweep varies the executable sufix while all methods start from the same first-trigger snapshots.

<table><tr><td rowspan="2">Rewrite rule</td><td rowspan="2">Rescue(%) ↑</td><td rowspan="2">Harm(%) ↓</td><td rowspan="2">Switchjump ↓</td><td colspan="5">Perturbed success (%) ↑</td></tr><tr><td> $d_{\text{lat}}=0$ </td><td> $d_{\text{lat}}=1$ </td><td> $d_{\text{lat}}=3$ </td><td> $d_{\text{lat}}=5$ </td><td> $d_{\text{lat}}=10$ </td></tr><tr><td>Wait for next chunk boundary</td><td>7.4</td><td>4.8</td><td>-</td><td>34.0</td><td>34.0</td><td>34.0</td><td>34.0</td><td>34.0</td></tr><tr><td>Unconstrained generation, discard prefix</td><td>10.1</td><td>8.9</td><td>0.284</td><td>48.9</td><td>46.0</td><td>39.8</td><td>33.2</td><td>23.4</td></tr><tr><td>Hard prefix, full rewrite (W=0)</td><td>13.2</td><td>6.8</td><td>0.171</td><td>49.5</td><td>47.0</td><td>42.4</td><td>36.0</td><td>25.8</td></tr><tr><td>RTC prefix-preserving inpainting</td><td>13.7</td><td>4.1</td><td>0.091</td><td>53.0</td><td>50.6</td><td>45.6</td><td>39.7</td><td>29.1</td></tr><tr><td>Hard prefix, validation-tuned fixed W</td><td>12.8</td><td>3.7</td><td>0.104</td><td>53.5</td><td>51.0</td><td>46.0</td><td>39.9</td><td>29.6</td></tr><tr><td>Hard prefix, adaptive W, shuffled exceedance</td><td>14.4</td><td>3.6</td><td>0.103</td><td>51.9</td><td>49.2</td><td>44.3</td><td>38.5</td><td>27.6</td></tr><tr><td>Hard prefix, adaptive W(e)</td><td>16.9</td><td>2.8</td><td>0.072</td><td>55.8</td><td>53.2</td><td>47.6</td><td>42.0</td><td>31.4</td></tr></table>

keeps the largest margin at every tested latency.  
![](images/61ab7a7e9e487cf905ea3bbd0cdf988756cd43eefa28530636b39494aa9842aa.jpg)  
Figure S3: Perturbed success across inference latency for the fixed-trigger repair rules of Table S6. All rules branch from the same first-trigger snapshots; the boundary baseline ignores the trigger and is latency-independent.

## K Exceedance-to-Retention Mechanism

To test the mapping from standardized exceedance to reference retention, we partition positive exceedances into four equal-frequency bins using validation data and freeze the bin edges before evaluating the locked snapshots. Every snapshot (all 1,080 first-crossing snapshots, 270 per bin) is branched with the same policy noise under $W \in \{ \bar { 0 } , 0 . 2 5 , 0 . 5 , 0 . 7 5 , 1 \}$ and under the frozen adaptive mapping. The validationselected fixed weight decreases across exceedance bins, while the test set is used only to evaluate the frozen choices. Averaged over bins, the best single fixed weight attains

46.0% perturbed success, the frozen continuous mapping $4 7 . 6 \%$ , and bin-wise validation-selected weights 48.8%: exceedance-conditioned retention carries the gain, the continuous mapping captures most of it without bin edges or per-bin tuning, and the shift of the per-bin optimum toward smaller weights at larger exceedance is the monotonicity that $w _ { 0 } ( e )$ encodes. Figure S4 plots the same grid: the best fixed weight shifts from 0.75 toward 0 as the exceedance bin rises, and the frozen adaptive mapping stays within 1.5 points of the per-bin envelope without bin-specific tuning.

![](images/38e74729dfb8da19e84dc561c5f77f603a281b1fc6d7be84fa905185b3a7f1ca.jpg)  
Figure S4: Perturbed success under fixed retention weights, evaluated separately in each exceedance bin of Table S7. Stars mark the frozen adaptive mapping $W ( e )$ at each bin’s median exceedance.

## L Natural Failures and Distribution Shifts

The controlled perturbation study provides known onsets and paired snapshots, but it does not establish performance on naturally occurring failures or benign observation shifts. We therefore evaluate a held-out natural-execution cohort drawn from training-side task–scene groups excluded from monitor training and validation. Two annotators, blinded to detector scores, mark the first observable deviation, last recoverable time, failure type, and whether a nonempty sufix could plausibly repair the episode. We report both all failures and the actionable subset, without excluding failures that the policy cannot repair. The audit set is case–control: it contains every annotated failure and a matched sample of natural successes, so no success rate is defined on it. The execution-mode panel of Table S8 instead re-executes the same 606 task–scene– initialization configurations under each mode with fresh randomness, so its success rates estimate the population rate rather than the case–control composition.

Table S7: Exceedance-conditioned reference retention. Panel A reports perturbed success from common first-trigger snapshots under a fixed weight grid and the frozen adaptive mapping. Panel B summarizes the validation-selected weight and adaptive repair outcomes.

<table><tr><td rowspan="2">Exceedance bin</td><td rowspan="2">Triggers</td><td colspan="5">Perturbed success under fixed W (%) ↑</td><td colspan="2">Frozen mappings</td></tr><tr><td>W=0</td><td>W=0.25</td><td>W=0.50</td><td>W=0.75</td><td>W=1</td><td>Adaptive W(e)</td><td> $W_{val}$ </td></tr><tr><td> $B_1$  (lowest)</td><td>270</td><td>39.2</td><td>42.8</td><td>45.6</td><td>48.3</td><td>47.5</td><td>47.2</td><td>0.75</td></tr><tr><td> $B_2$ </td><td>270</td><td>41.0</td><td>44.6</td><td>48.7</td><td>47.9</td><td>45.0</td><td>47.6</td><td>0.50</td></tr><tr><td> $B_3$ </td><td>270</td><td>44.2</td><td>49.1</td><td>47.6</td><td>44.0</td><td>39.8</td><td>48.0</td><td>0.25</td></tr><tr><td> $B_4$  (highest)</td><td>270</td><td>49.1</td><td>47.2</td><td>42.0</td><td>36.7</td><td>31.8</td><td>47.6</td><td>0.00</td></tr><tr><td>Exceedance bin</td><td>Median e</td><td> $W_{val}$ </td><td>Success at  $W_{val}$ (%)</td><td colspan="2">Median adaptive W(e)</td><td>Adaptive rescue (%) ↑</td><td colspan="2">Adaptive harm (%) ↓</td></tr><tr><td> $B_1$ </td><td>0.18</td><td>0.75</td><td>48.3</td><td colspan="2">0.78</td><td>12.4</td><td colspan="2">2.3</td></tr><tr><td> $B_2$ </td><td>0.46</td><td>0.50</td><td>48.7</td><td colspan="2">0.53</td><td>15.8</td><td colspan="2">2.5</td></tr><tr><td> $B_3$ </td><td>0.91</td><td>0.25</td><td>49.1</td><td colspan="2">0.29</td><td>18.7</td><td colspan="2">2.8</td></tr><tr><td> $B_4$ </td><td>1.74</td><td>0.00</td><td>49.1</td><td colspan="2">0.08</td><td>23.1</td><td colspan="2">3.2</td></tr></table>

The confidently-wrong analysis uses the strongest policyside detector selected on validation data. Both policy uncertainty and world-model risk are centered at their independently calibrated thresholds; no median split or triggerconditioned selection is used. Each episode is assigned at a preregistered decision window before irreversibility, and rescue uses the same fixed repair rule in all quadrants. The resulting quadrant decomposition is reported in Table 5: the low-uncertainty, high-risk cell concentrates 48.4% of all failures and remains the most recoverable.

The benign-shift test keeps the nominal conformal threshold fixed. Shift families and severities are selected on validation data, and at least two held-out families are absent from the benign negatives used for risk-head training. We report all predefined injections and separately require the paired no-intervention continuation to remain successful when estimating unnecessary intervention and harm.

Leave-one-family-out (LOFO) training distinguishes two scopes. Risk-head-only LOFO tests whether the head memorizes perturbation labels while retaining a world model exposed to that family. Full-verifier LOFO excludes the family from both world-model and risk-head optimization. Every changed detector is independently recalibrated on the same nominal population. Severity extrapolation is evaluated separately because high-severity events can be easier to detect but harder to repair. Figure S5 visualizes both panels: family exclusion lowers timely recall by roughly five points for the risk head and eight points for the full verifier, every held-out family stays far above the action-shufled control, and heavier held-out severities are detected more reliably while being physically harder to repair.

## M Episodic-Memory Interaction Ablation

The policy reads the full keyframe bank, whereas the risk head reads only its compact summary. We therefore run a fully retrained $2 \times 2$ factorial ablation (Table S11) and do not rely on test-time zeroing. When the risk-summary reader changes, its threshold is independently recalibrated; when the policy reader changes, the action model is retrained with the corresponding input path. Completed-subgoal regression counts an already satisfied task predicate that becomes unsatisfied after a replan and remains unsatisfied at episode end.

The interaction distinguishes the two consumers. The policy reader accounts for most of the reduction in completedsubgoal regression and contributes most strongly on composite tasks, where evidence of earlier stations has often left the current field of view. The risk summary has a smaller direct efect on task success but brings empirical episode FWER closer to its calibrated target. Their combination gives the strongest clean-suite average and the fewest regressions, consistent with preserving both action-level progress and the context used to judge whether a discrepancy is anomalous.

To separate persistent context from storage capacity, we additionally compare memory forms under the same maximum frame budget K. Each policy-reader variant is retrained with its corresponding memory path, and every changed risk reader is independently recalibrated; no row is created by masking memory only at test time.

## N Calibration, Data Scale, and Repeated Interventions

Calibration-size sensitivity changes only the nominal calibration sample, not detector weights, validation choices, or locked test episodes. We repeatedly subsample the same calibration pool and retain only sizes for which the finite-sample conformal quantile is defined at $\alpha \ : = \ : 0 . 0 5$ . Monitor-data sensitivity instead retrains the world model and risk head on nested task–scene–initialization groups, followed by an independent nominal calibration for each row. Figure S6 summarizes both sensitivities: detection metrics saturate as either resource grows, while the empirical episode FWER stays near the 5% target at every size.

Table S8: Held-out natural-failure audit; natural executions contain no injected perturbation, and onset and last recoverable time are annotated blind to detector output. The upper panel’s audit set is case–control (all annotated failures plus matched success controls), so no success rate is defined on it; the lower panel re-executes the same configurations under each execution mode with fresh randomness.

<table><tr><td>Evaluation subset</td><td>Failure episodes</td><td>Success controls</td><td>Event AUPRC ↑</td><td>Timely recall (%) ↑</td><td>Effective lead (steps) ↑</td></tr><tr><td>All annotated natural failures</td><td>186</td><td>420</td><td>0.683</td><td>70.4</td><td>10.6</td></tr><tr><td>Failures with a nonempty actionable suffix</td><td>139</td><td>420</td><td>0.721</td><td>78.4</td><td>12.1</td></tr></table>

<table><tr><td>Execution mode</td><td>Overall task success (%) ↑</td><td>Natural-failure rescue (%) ↑</td><td>Natural-success harm (%) ↓</td><td>Deployed rewrites per episode ↓</td><td>Subgoal regressions per episode ↓</td></tr><tr><td>Open-loop action backbone</td><td>61.4</td><td>-</td><td>-</td><td>0.0</td><td>0.74</td></tr><tr><td>Invocation-matched periodic replanning</td><td>64.8</td><td>12.3</td><td>4.6</td><td>2.3</td><td>0.62</td></tr><tr><td>Full CheckVLA</td><td>68.9</td><td>21.6</td><td>2.7</td><td>0.44</td><td>0.31</td></tr></table>

Table S9: Benign-deviation stress test with the nominal threshold kept fixed. Excess intervention is measured in percentage points relative to matched nominal episodes; harm is conditioned on paired no-intervention continuations that succeed.

<table><tr><td>Benign shift</td><td>Episodes</td><td>Reference success (%) ↑</td><td>Unnecessary first intervention (%) ↓</td><td>Excess vs. nominal (points) ↓</td><td>Harm (%) ↓</td><td>CheckVLA task success (%) ↑</td></tr><tr><td>Small camera jitter</td><td>360</td><td>98.1</td><td>6.1</td><td>1.3</td><td>1.1</td><td>97.2</td></tr><tr><td>Brief partial occlusion</td><td>360</td><td>96.9</td><td>7.4</td><td>2.6</td><td>1.5</td><td>95.8</td></tr><tr><td>Task-irrelevant background change</td><td>360</td><td>98.6</td><td>5.6</td><td>0.8</td><td>0.7</td><td>98.1</td></tr><tr><td>Small self-correcting object/base displacement</td><td>360</td><td>95.8</td><td>6.7</td><td>1.9</td><td>1.3</td><td>94.9</td></tr><tr><td>Pooled benign shifts</td><td>1440</td><td>97.4</td><td>6.5</td><td>1.7</td><td>1.2</td><td>96.5</td></tr></table>

![](images/e14c20aa02cf4549632a1b421bb21ecad21e629770e9e96a9abe924581ca97c0.jpg)  
(a) Leave-one-family-out training

![](images/aeff7232060280b02872ff07ba29795929d57f38369223430080c07a4962753f.jpg)  
Figure S5: Generalization beyond monitor-training perturbations, from Table S10. (a) Timely recall under leave-one-family-out training; the dashed line is the action-shufled control. (b) Severity extrapolation with the heavy range excluded from monitor training.  
(b) Severity extrapolation

Table S10: Generalization beyond monitor-training perturbations. Panel A compares all-family training, risk-head-only LOFO, and full-verifier LOFO. Panel B evaluates severity extrapolation with the heavy range excluded from monitor training and model selection.

<table><tr><td>Test family</td><td>Training exclusion</td><td>EpisodeFWER (%)</td><td>EventAUPRC ↑</td><td>Timely recall(%) ↑</td><td>Effective lead(steps) ↑</td><td>Perturbed success(%) ↑</td></tr><tr><td>Object/base impulse</td><td>None; all-family reference</td><td>4.8</td><td>0.781</td><td>80.4</td><td>13.8</td><td>49.2</td></tr><tr><td>Object/base impulse</td><td>Risk head only</td><td>5.1</td><td>0.728</td><td>74.1</td><td>12.2</td><td>46.1</td></tr><tr><td>Object/base impulse</td><td>World model and risk head</td><td>5.0</td><td>0.694</td><td>70.8</td><td>11.4</td><td>44.8</td></tr><tr><td>Transient joint offset</td><td>None; all-family reference</td><td>4.7</td><td>0.752</td><td>77.1</td><td>12.9</td><td>47.4</td></tr><tr><td>Transient joint offset</td><td>Risk head only</td><td>4.9</td><td>0.711</td><td>72.6</td><td>11.6</td><td>45.3</td></tr><tr><td>Transient joint offset</td><td>World model and risk head</td><td>5.2</td><td>0.681</td><td>69.4</td><td>10.8</td><td>43.9</td></tr><tr><td>Moved-object intervention</td><td>None; all-family reference</td><td>4.9</td><td>0.759</td><td>76.2</td><td>12.8</td><td>46.2</td></tr><tr><td>Moved-object intervention</td><td>Risk head only</td><td>5.0</td><td>0.702</td><td>71.0</td><td>11.3</td><td>44.0</td></tr><tr><td>Moved-object intervention</td><td>World model and risk head</td><td>4.8</td><td>0.673</td><td>68.7</td><td>10.5</td><td>42.6</td></tr><tr><td>Test severity</td><td>Seen in monitor training?</td><td>EpisodeFWER (%)</td><td>EventAUPRC ↑</td><td>Timely recall(%) ↑</td><td>Recoverablefraction (%)</td><td>Perturbed success(%) ↑</td></tr><tr><td>Light, in-range</td><td>Yes</td><td>4.7</td><td>0.721</td><td>71.2</td><td>90.4</td><td>50.1</td></tr><tr><td>Medium, in-range</td><td>Yes</td><td>4.9</td><td>0.772</td><td>78.3</td><td>74.2</td><td>47.8</td></tr><tr><td>Heavy, held out</td><td>No</td><td>5.1</td><td>0.801</td><td>82.0</td><td>42.1</td><td>34.2</td></tr></table>

Table S11: 2×2 memory ablation. A-S, C-S, C-U, and Average are clean-suite task success rates (%); Average uses the oficial 18/16/16 task weighting. Every variant is retrained, and every risk-reader variant is recalibrated.

<table><tr><td>Policy bank</td><td>Risk summary</td><td>A-S</td><td>C-S</td><td>C-U</td><td>Average</td><td>Subgoal regressions / ep. ↓</td><td>Episode FWER (%) ↓</td></tr><tr><td>No</td><td>No</td><td>61.5</td><td>28.3</td><td>9.0</td><td>34.1</td><td>1.34</td><td>5.6</td></tr><tr><td>Yes</td><td>No</td><td>63.0</td><td>29.7</td><td>9.8</td><td>35.3</td><td>0.55</td><td>5.5</td></tr><tr><td>No</td><td>Yes</td><td>62.4</td><td>29.3</td><td>9.4</td><td>34.8</td><td>1.08</td><td>4.9</td></tr><tr><td>Yes</td><td>Yes</td><td>63.7</td><td>30.9</td><td>10.2</td><td>36.1</td><td>0.18</td><td>4.8</td></tr></table>

Table S12: Memory-form comparison at a matched maximum frame budget K. Recent-K and uniformly spaced memories control for storage capacity, whereas the event-driven bank writes frames only at validated pause-and-diversity events.

<table><tr><td>Memory form</td><td>Maximum stored frames</td><td>A-S</td><td>C-S</td><td>C-U</td><td>Average</td><td>Subgoal regressions per episode ↓</td><td>Episode FWER (%)</td></tr><tr><td>No persistent memory</td><td>-</td><td>61.5</td><td>28.3</td><td>9.0</td><td>34.1</td><td>1.34</td><td>5.6</td></tr><tr><td>Recent-K FIFO frames</td><td>K</td><td>62.3</td><td>29.0</td><td>9.5</td><td>34.7</td><td>0.92</td><td>5.2</td></tr><tr><td>Uniformly spaced K frames</td><td>K</td><td>62.7</td><td>29.4</td><td>9.7</td><td>35.1</td><td>0.71</td><td>5.0</td></tr><tr><td>Event-driven keyframe bank</td><td>K</td><td>63.7</td><td>30.9</td><td>10.2</td><td>36.1</td><td>0.18</td><td>4.8</td></tr></table>

Benign negatives can lower false interventions under harmless shifts, but they could also suppress sensitivity to real deviations. We therefore retrain the risk head with and without benign negatives while holding the world model, policy, data budget, and nominal calibration protocol fixed (Table S15). Benign negatives leave sensitivity to real deviations essentially unchanged while cutting benign first interventions from 14.8% to 6.5% and benign harm from 4.7% to 1.2%; the 1.1-point higher timely recall of the benign-free head is obtained at a 6.2% empirical FWER, not at the target operating point. Although both variants are recalibrated on the same nominal pool, the head trained without benign negatives reacts to harmless appearance variation that the calibration pool covers only sparsely, which shifts its deployed error rate above the target; benign-aware training keeps the deployed operating point at the prescribed 4.8%.

Detection performance is also decomposed by task phase in Table S16, so that aggregate recall cannot hide late or weak performance around contact. Phase labels and negative nominal windows are fixed before detector outputs are read. Step-wise false-positive rate is reported only as a diagnostic alongside episode FWER, not as a replacement for repeatedtesting control.

The conformal statement covers the first intervention only. We therefore audit repeated deployment after the first repair, counting a rewrite only when its sufix is actually deployed.

![](images/8fa05a5ba030c94b1031b702bd352f5bba9bf60c6b92ef31adeb2e58b48e25db.jpg)

![](images/077e7b15034886bfd2ffcbd06dc92de3f9c6756a013a41616a108915d493c09f.jpg)

![](images/442b06e0c44a16edaef4e7b569be8e053f58026a64f3fefd19a3c3eef36e12b1.jpg)  
(a) Nominal calibration episodes (%)

![](images/3d4578a1ab290c005ab83ea1102503391bf69dd69a2838c8710a1250fc2d95d8.jpg)  
(b) Auxiliary monitor-training data (%)  
Figure S6: Sensitivity to (a) the number of nominal calibration episodes and (b) the auxiliary monitor-training budget, from Tables S13 and S14. Solid lines use the left axis; the dashed line is the empirical episode FWER on the right axis, with the dotted line at the 5% target.

Table S13: Sensitivity to the number of nominal calibration episodes. Values for subsampled rows are averages across 20 resamples; event AUPRC is threshold-independent and is therefore unchanged because detector weights and event scores are fixed.

<table><tr><td>Calibration fraction</td><td> $n_{cal}$ </td><td>Resamples</td><td>Finite  $\hat{q}_{\alpha}$ (%)</td><td>EpisodeFWER (%)</td><td>EventAUPRC</td><td>Timely recall(%)</td><td>Effective lead(steps)</td><td>Perturbed success(%)</td></tr><tr><td>25%</td><td>100</td><td>20</td><td>100</td><td>5.4</td><td>0.764</td><td>74.1</td><td>12.1</td><td>45.8</td></tr><tr><td>50%</td><td>200</td><td>20</td><td>100</td><td>5.0</td><td>0.764</td><td>76.3</td><td>12.7</td><td>46.7</td></tr><tr><td>75%</td><td>300</td><td>20</td><td>100</td><td>4.9</td><td>0.764</td><td>77.1</td><td>13.0</td><td>47.1</td></tr><tr><td>100%</td><td>400</td><td>-</td><td>100</td><td>4.8</td><td>0.764</td><td>77.9</td><td>13.2</td><td>47.6</td></tr></table>

Table S14: Sensitivity to auxiliary monitor-training data. Training subsets are nested by task–scene–initialization group; the action backbone is fixed, each monitor is retrained from scratch, and each row receives an independent nominal conformal calibration.

<table><tr><td>Monitor-training fraction</td><td>Training rollouts</td><td>Episode FWER (%)</td><td>Event AUPRC ↑</td><td>Timely recall (%) ↑</td><td>Effective lead (steps) ↑</td><td>Perturbed success (%) ↑</td></tr><tr><td>25%</td><td>900</td><td>5.3</td><td>0.638</td><td>65.2</td><td>9.8</td><td>41.3</td></tr><tr><td>50%</td><td>1800</td><td>5.1</td><td>0.701</td><td>71.4</td><td>11.4</td><td>44.2</td></tr><tr><td>75%</td><td>2700</td><td>4.9</td><td>0.742</td><td>75.6</td><td>12.6</td><td>46.1</td></tr><tr><td>100%</td><td>3600</td><td>4.8</td><td>0.764</td><td>77.9</td><td>13.2</td><td>47.6</td></tr></table>

Table S15: Ablation of benign negatives in risk-head training. The benign intervention and harm columns use the held-out shift families in Table S9; detector metrics use the unchanged controlled perturbation test.

<table><tr><td>Risk-head training</td><td>EpisodeFWER (%)</td><td>Benign firstintervention (%) ↓</td><td>Benign harm(%) ↓</td><td>EventAUPRC ↑</td><td>Timely recall(%) ↑</td><td>Perturbed success(%) ↑</td></tr><tr><td>Without benign negatives</td><td>6.2</td><td>14.8</td><td>4.7</td><td>0.772</td><td>79.0</td><td>47.4</td></tr><tr><td>With benign negatives</td><td>4.8</td><td>6.5</td><td>1.2</td><td>0.764</td><td>77.9</td><td>47.6</td></tr></table>

Table S16: Task-phase and false-alarm decomposition. Panel A reports event-level detector behavior by perturbation onset phase. Panel B reports nominal false alarms at both episode and step levels under the same target episode FWER.

<table><tr><td>Perturbation onset phase</td><td>Events</td><td>Event AUPRC ↑</td><td>Timely recall (%) ↑</td><td>Effective lead (steps) ↑</td><td>Premature alarm (%) ↓</td><td>Perturbed success (%) ↑</td></tr><tr><td>Navigation</td><td>270</td><td>0.731</td><td>73.7</td><td>14.8</td><td>2.2</td><td>46.1</td></tr><tr><td>Pre-contact alignment</td><td>270</td><td>0.748</td><td>75.9</td><td>13.9</td><td>2.6</td><td>46.9</td></tr><tr><td>Post-grasp transport</td><td>270</td><td>0.792</td><td>82.2</td><td>12.7</td><td>3.0</td><td>49.8</td></tr><tr><td>Appliance interaction</td><td>270</td><td>0.761</td><td>79.6</td><td>11.4</td><td>3.3</td><td>47.6</td></tr><tr><td>Detector</td><td></td><td>Nominal episodes</td><td>Episode FWER (%)</td><td>False alarms per 1000 steps</td><td>Median first-alarm time (steps)</td><td>Policy calls per episode</td></tr><tr><td>Observation-only world predictor</td><td></td><td>1800</td><td>5.3</td><td>1.31</td><td>84</td><td>10.0</td></tr><tr><td>Action-shuffled world model</td><td></td><td>1800</td><td>5.1</td><td>1.27</td><td>86</td><td>9.8</td></tr><tr><td>Full action-conditioned verifier</td><td></td><td>1800</td><td>4.8</td><td>1.12</td><td>91</td><td>10.2</td></tr></table>

Re-alarms during inference or cooldown are logged separately and do not count as additional interventions.

## O Runtime and Complexity

Runtime is measured end to end with low-level control active, after warm-up, on the same workstation and batch size for every variant. The periodic baseline runs the verifier in shadow mode and is tuned on validation data to match the full method’s mean number of VLA invocations; it therefore controls for both monitoring compute and extra policy calls. Added trainable parameters and monitor FLOPs are measured relative to the same decoupled action backbone and exclude the frozen world encoder, whose memory footprint is nevertheless included in peak VRAM.

The VLA latency is 286 / 338 ms at p50/p95 under a 10 Hz controller, corresponding to the nominal $d _ { \mathrm { l a t } } = 3$ steps illustrated in Figure S1. The verifier remains below one control period at p95 and overlaps with execution. Invocation matching makes the principal comparison interpretable: a diference between periodic replanning and CheckVLA reflects how the extra calls are timed and used rather than how many are spent; the verified-trigger row of Table 3 isolates the timing component alone (+3.9 points).

The fixed-trigger comparison isolates the repair sampler, whereas RTC (Black, Galliker, and Levine 2025) and REMAC (Wang et al. 2026b) also define native asynchronous execution procedures. We therefore report a separate end-toend comparison. RTC uses its published training-free schedule; full REMAC is retrained with masked action chunks and is not represented by an inference-only sampler. Validation selects each asynchronous horizon before the locked test, and policy calls and wall-clock cost remain explicit.

## P Failure Analysis and Evaluation Boundaries

A blinded annotation audit assigns one primary proximate cause to each of a sample of 140 residual failures from the full method on the perturbation test, reaching Cohen’s $\kappa = 0 . 7 9$ before adjudication. The taxonomy attributes 31.4% of residual failures to a missed or late alarm that left no useful sufix, 23.6% to a physically unrecoverable state before deployment,

18.6% to world-model ambiguity under occlusion or contact, 14.3% to a semantic or action-generation error that a correct trigger could not repair, 7.1% to repair-induced state regression or repeated triggering, and 5.0% to an unnecessary intervention on an otherwise successful continuation. These mutually exclusive shares sum to 100.0% after adjudication.

The distribution separates two distinct limiting factors. Late alarms and long inference latency reduce the sufix that remains physically modifiable; better rewriting cannot recover this group. Conversely, policy-semantic errors and ambiguous world-model features may be detected on time but still lack a competent replacement action. This distinction motivates reporting timely recall, rescue, and harm separately; final success alone is not a detector metric. It also bounds the current claim: functional conformal calibration controls only the unnecessaryfirst intervention on exchangeable nominal-success episodes. It does not guarantee recall under perturbations, safety after a repair, coverage after distribution shift, or transfer from simulation to hardware. Those properties remain empirical and must be re-evaluated whenever the task distribution, observation stack, controller frequency, or risk-reader memory changes.

## Q Additional Visualizations

Figure S7 complements the tabular results with three outcome views. The left panel decomposes the perturbedsuccess diference between the open-loop backbone (28.6%) and full CheckVLA (47.6%) into the contributions of the timely verified trigger, the latency-aware sufix rewrite, and risk-adaptive guidance, with each level anchored to a row of Table S6 or Table S19. The middle panel shows paired rescue and harm rates for the fixed-trigger repair rules of Table S6; thejoint improvement of higher rescue at lower harm appears only for the correctly paired adaptive guidance. The right panel reports completed-subgoal regression for the matched memory forms of Table S12, isolating the contribution of the event-driven bank to preserving prior progress.

Table S17: Audit of repeated interventions after the first deployment. Post-repair re-alarm uses a validation-fixed window of $K _ { \mathrm { a u d i t } }$ control steps.

<table><tr><td>Evaluation cohort</td><td>Episodes</td><td>≥ 1 deployed rewrite (%)</td><td>≥ 2 deployed rewrites (%)</td><td>Mean rewrites per episode</td><td>Re-alarm within  $K_{audit}$  (%)</td><td>Post-repair subgoal regression (%)</td><td>Harm (%) ↓</td></tr><tr><td>Clean target-suite episodes</td><td>1500</td><td>4.8</td><td>0.7</td><td>0.06</td><td>8.3</td><td>1.1</td><td>2.1</td></tr><tr><td>Benign-shift episodes</td><td>1440</td><td>6.5</td><td>1.0</td><td>0.08</td><td>9.4</td><td>1.3</td><td>1.2</td></tr><tr><td>Injected-perturbation episodes</td><td>1080</td><td>86.2</td><td>16.8</td><td>1.04</td><td>14.5</td><td>3.2</td><td>2.8</td></tr><tr><td>Held-out natural executions</td><td>606</td><td>34.8</td><td>7.9</td><td>0.44</td><td>12.1</td><td>2.9</td><td>2.7</td></tr></table>

Table S18: Runtime and complexity comparison. The periodic baseline matches the full method’s average VLA-call budget and carries the same shadow verifier.

<table><tr><td>Execution mode</td><td>Added trainable params (M)</td><td>Added GFLOPs / step</td><td>Monitor p50 / p95 (ms)</td><td>VLA calls / ep.</td><td>Wall-clock factor</td><td>Peak VRAM (GB)</td></tr><tr><td>Open-loop action backbone</td><td>0.0</td><td>0.0</td><td>-</td><td>7.8</td><td>1.00×</td><td>21.4</td></tr><tr><td>Verifier in shadow mode</td><td>88.4</td><td>16.9</td><td>12.1 / 16.4</td><td>7.8</td><td>1.06×</td><td>23.1</td></tr><tr><td>Invocation-matched periodic re-planning</td><td>88.4</td><td>16.9</td><td>12.1 / 16.4</td><td>10.1</td><td>1.18×</td><td>23.3</td></tr><tr><td>Full CheckVLA</td><td>88.4</td><td>16.9</td><td>12.1 / 16.4</td><td>10.2</td><td>1.18×</td><td>23.3</td></tr></table>

Table S19: End-to-end asynchronous execution comparison. Clean Average uses the oficial task weighting; perturbed success, harm, policy calls, and wall-clock cost use the controlled perturbation protocol. Training adaptation distinguishes training-free scheduling from methods that modify the action model or add a verifier.

<table><tr><td>Execution mode</td><td>Clean Average (%)</td><td>Perturbed success (%)</td><td>VLA calls per episode</td><td>Wall-clock factor</td><td>Harm (%)</td><td>Training adaptation</td></tr><tr><td>Open-loop action backbone</td><td>21.6</td><td>28.6</td><td>7.8</td><td>1.00×</td><td>-</td><td>None</td></tr><tr><td>RTC, native asynchronous schedule</td><td>30.2</td><td>40.8</td><td>10.1</td><td>1.17×</td><td>4.0</td><td>None</td></tr><tr><td>Full REMAC</td><td>31.4</td><td>42.6</td><td>10.3</td><td>1.21×</td><td>3.5</td><td>Masked-chunk training</td></tr><tr><td>Full CheckVLA</td><td>36.1</td><td>47.6</td><td>10.2</td><td>1.18×</td><td>2.8</td><td>Verifier training</td></tr></table>

![](images/381065903b20738d3e453cb78639f2b4a1fee2781e6d2035df5c612a62c0e35f.jpg)  
b

![](images/64502cb6b64af43e5c03eb731199d2aaccf333b21ee874c8cd9511a2c60b63fc.jpg)

![](images/bbd4a510692a2db6c01e047cf505f3b1a9687e89ced05aa857c74412714b6874.jpg)  
Figure S7: Outcome visualization. The panels show (left) an additive waterfall of perturbed success from the open-loop backbone to full CheckVLA, (middle) paired rescue and harm rates for the fixed-trigger repair rules, and (right) completedsubgoal regression for matched memory forms.

## R Task Visualizations on RoboCasa365

Figures S8–S10 visualize twelve representative composite tasks from the oficial RoboCasa365 target benchmark. Each composite task issues a natural-language instruction that chains several sub-goals, often across diferent stations of the kitchen, so the mobile manipulator must interleave base locomotion with arm control: fetching an object at one station, transporting it while grasped, and operating articulated fixtures (doors, drawers, knobs, buttons) at another.

Each storyboard follows the same format. The top ribbon decomposes the episode’s instruction into sub-task prompts, aligned with the keyframe columns they span; blue boxes denote manipulation segments, and orange boxes denote segments that involve base locomotion. The two rows below show seven time-aligned keyframes from the third-person scene camera and the egocentric wrist camera, the two views consumed by the policy.

The three figures group the tasks by workspace structure: Figure S8 shows cross-station deliveries, whose instructions explicitly require driving between stations (e.g., taking a straw from a drawer and delivering it to a cup on the dining counter); Figure S9 shows tasks that interleave locomotion with appliance operation (microwave, freezer, stove, cutting station); and Figure S10 shows stationary appliance and counter-top manipulation. Episodes of this kind chain partially irreversible sub-goals over a long horizon— switching appliances on, releasing objects into containers, closing doors on loaded racks—which is precisely the setting in which an in-chunk deviation is costly if it goes unnoticed, and which motivates the action-conditioned verification and in-chunk replanning mechanism formalized in Eqs. 2–10.

![](images/f4aeb46a51dddcab906458c361481a413f5fbb0486b30854f1b9a92859d17db3.jpg)  
Figure S8: Task visualizations on RoboCasa365 (I): cross-station delivery tasks. Blue prompt boxes mark manipulation segments and orange boxes mark base locomotion; the two rows show time-aligned third-person and egocentric wrist views. Tasks: Deliver Straw, Get Toasted Bread, Store Leftovers in Bowl, Make Ice Lemonade.

![](images/c0e328dc2fdd16504177a38ca293f7102753af7efc32af3fdcb511cba1ed51aa.jpg)  
Figure S9: Task visualizations on RoboCasa365 (II): tasks interleaving base locomotion with appliance operation. Format as in Figure S8. Tasks: Steam in Microwave, Separate Freezer Rack, Searing Meat, Set Up Cutting Station.

![](images/1a8592d805e92caed950e3cd387a8a2b4355e107b2c0aec91549a365321b7408.jpg)  
Figure S10: Task visualizations on RoboCasa365 (III): stationary appliance and counter-top manipulation tasks. Format as in Figure S8. Tasks: Prepare Cofee, Kettle Boiling, Pan Transfer, Pre-Soak Pan.

## S Complete Configuration

Table S20 consolidates every hyperparameter of the executed system, its training and calibration, and the baseline configurations. Validation-selected entries were fixed on the validation split of Table S1 before either locked test was read; no value was tuned on the target suites.

Table S20: Complete configuration. Validation-selected values are chosen once on the validation split before the locked tests; architecture and infrastructure entries are fixed design choices.

<table><tr><td>Component</td><td>Hyperparameter</td><td>Value</td><td>Selected on</td></tr><tr><td colspan="4">Execution</td></tr><tr><td>Chunked execution</td><td>Chunk length H</td><td>50</td><td>Fixed a priori</td></tr><tr><td>Chunked execution</td><td>Flow-integration steps</td><td>10 (Δu=0.1)</td><td>Fixed a priori</td></tr><tr><td>Chunked execution</td><td>Deployment schedule  $d_{\text{lat}}$ </td><td>3 control steps</td><td>Fixed a priori</td></tr><tr><td colspan="4">Verification</td></tr><tr><td>Rolling prediction</td><td>Span bound k</td><td>8</td><td>Validation</td></tr><tr><td>Discrepancy</td><td>Distance dist</td><td>cosine on pooled, normalized features</td><td>Validation</td></tr><tr><td>Discrepancy</td><td>Standardization floor ε</td><td> $10^{-3}$ </td><td>Fixed a priori</td></tr><tr><td>Risk head</td><td>Window w</td><td>32</td><td>Validation</td></tr><tr><td>Trigger</td><td>Target level α</td><td>0.05</td><td>Fixed a priori</td></tr><tr><td>Trigger</td><td>Suppression length  $s_{\text{min}}$ </td><td>8 (= k)</td><td>Validation</td></tr><tr><td colspan="4">Suffix rewriting</td></tr><tr><td>Retention</td><td>Entry-scale floor  $w_{\text{min}}$ </td><td>0.0</td><td>Validation</td></tr><tr><td>Retention</td><td>Exceedance sensitivity β</td><td>1.38</td><td>Validation</td></tr><tr><td>Positional decay</td><td> $λ_{\text{mobility}}/\lambda_{\text{manip}}$ , replan</td><td>0.60 / 0.90</td><td>Validation</td></tr><tr><td>Positional decay</td><td> $λ_{\text{mobility}}/\lambda_{\text{manip}}$ , regular</td><td>0.10 / 0.15</td><td>Validation</td></tr><tr><td colspan="4">Episodic memory</td></tr><tr><td>Keyframe bank</td><td>Budget K / tokens per keyframe</td><td>8 / 16</td><td>Validation</td></tr><tr><td>Write filter</td><td>Pause window  $w_s$  / confirm delay</td><td>5 / 3</td><td>Validation</td></tr><tr><td>Write filter</td><td>Min. write gap / similarity threshold</td><td>10 / 0.95</td><td>Validation</td></tr><tr><td>Fusion</td><td>Gate bias initialization</td><td>-4</td><td>Fixed a priori</td></tr><tr><td colspan="4">Monitor architecture</td></tr><tr><td>Latent predictor Ψ</td><td>Width / depth / heads</td><td>960 / 7 / 12</td><td>Fixed a priori</td></tr><tr><td>Risk head R</td><td>Width / depth / heads</td><td>384 / 4 / 6</td><td>Fixed a priori</td></tr><tr><td>Monitor total</td><td>Added trainable parameters</td><td>88.4M</td><td>-</td></tr><tr><td colspan="4">Training and calibration</td></tr><tr><td>Ψ optimization</td><td>AdamW: lr / batch / steps</td><td> $3×10^{-4}/32/60k TF+20k self-rollout$ </td><td>-</td></tr><tr><td>R optimization</td><td>AdamW: lr / batch / steps</td><td> $1×10^{-4}/64/20k$ </td><td>-</td></tr><tr><td>R supervision</td><td>Onset window Δ</td><td>20 control steps</td><td>Fixed a priori</td></tr><tr><td>R supervision</td><td>Mix nominal : natural : perturbed</td><td>2:1:1 (1,800 / 900 / 900 rollouts)</td><td>Fixed a priori</td></tr><tr><td>Conformal</td><td>Fit / calibration episodes per seed</td><td>200 / 400</td><td>Fixed a priori</td></tr><tr><td>Audits</td><td>Re-alarm window  $K_{\text{audit}}$ </td><td>20 control steps</td><td>Validation</td></tr><tr><td colspan="4">Baselines</td></tr><tr><td>Periodic replanning</td><td>Replan interval</td><td>39 control steps</td><td>Validation</td></tr><tr><td>Policy-side detectors</td><td>Sample count Q</td><td>8</td><td>Fixed a priori</td></tr><tr><td>RTC / REMAC</td><td>Asynchronous horizon</td><td>25 control steps</td><td>Validation</td></tr><tr><td colspan="4">Infrastructure</td></tr><tr><td>Hardware</td><td>GPU / precision</td><td>A100 80 GB / bf16</td><td>-</td></tr><tr><td>Software</td><td>Framework</td><td>PyTorch 2.3.1, CUDA 12.1</td><td>-</td></tr><tr><td>Software</td><td>Simulator</td><td>RoboCasa365 (robosuite 1.4.1, MuJoCo 3.1.4)</td><td>-</td></tr><tr><td>Runs</td><td>Random seeds</td><td>1 / 2 / 3</td><td>-</td></tr></table>