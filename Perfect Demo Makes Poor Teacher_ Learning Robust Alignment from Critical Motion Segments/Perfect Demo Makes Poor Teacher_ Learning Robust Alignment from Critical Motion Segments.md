# Perfect Demo Makes Poor Teacher: Learning Robust Alignment from Critical Motion Segments

Mingyu Liu<sup>1,2∗</sup> Zeju Li<sup>1∗</sup> Jiuhe Shu<sup>1</sup>

Hanqing Wang<sup>3</sup> Yuhao Chao<sup>2,4</sup> Hao Chen<sup>1</sup> Chunhua Shen<sup>1†</sup>

<sup>1</sup>Zhejiang University <sup>2</sup>Shanghai Innovation Institute <sup>3</sup>Hong Kong University of Science and Technology (GZ) <sup>4</sup>Nanjing University github.com/aim-uofa/STAIR

Abstract: Expert demonstrations are widely assumed to be the gold standard for robot imitation learning. Yet for fine-grained manipulation such as insertion, stacking, and alignment, we uncover a counterintuitive failure mode: fluent demonstrations can be poor teachers. A skilled teleoperator compresses the decisive moments of alignment and recovery into a brief temporal window, leaving the policy flooded with redundant free-space motion and starved of supervision exactly where precision determines success. We address this bottleneck at two levels. At the data level, slowing down near alignment and resampling critical segments both help, yet the gain comes mainly from broadening the coverage of recovery states the policy must learn, not from reweighting frames it already has. Such data-side fixes, however, leave the policy’s per-frame view untouched: a single image still maps directly to an action, and the local motion that governs correction stays implicit. We therefore turn to the representation level and introduce STAIR (Spatio-Temporal feature As an Interface for Robot learning), a compact dynamic feature that bridges the vision-language model and the action expert, distilling the short-horizon motion already recorded in each trajectory into dense, motion-aware supervision. Trained on fluent data alone, STAIR recovers most of the deliberate-demonstration gain (50.0 to 62.2% overall, approaching the 64.4% of deliberate demonstrations). These results call for a more pedagogical view of robot data, optimized for machine learnability rather than human efficiency alone.

Keywords: Robot Manipulation, Representation Learning, Data Curation

## 1 Introduction

“In the beginner’s mind there are many possibilities, but in the expert’s there arefew.”

— Shunryu Suzuki, Beginner’s Mind

Robots, in this sense, are still beginners. A skilled teleoperator can make fine manipulation look effortless. In tasks such as block stacking or tight insertion, the motion looks smooth from start to finish: a fast approach, a brief final adjustment, a clean completion. Such demonstrations seem ideal for imitation learning [1]. Yet in practice we observe the opposite. Policies trained on these fluent demonstrations master the approach but fail in the last few centimeters, where a small alignment error turns into collision or unrecoverable drift. This exposes a simple tension: a perfect demonstration can make a poor teacher.

The reason is intuitive yet easy to overlook: a demonstration optimized for human execution is not optimized for robot learning [2]. In most fine-manipulation trajectories, the vast majority of frames describe an easy free-space transport, while success is decided by a short alignment segment near contact. A fluent expert compresses this decisive segment into a handful of frames (Fig. 1, top: alignment occupies only ≈ 10% of frames, and the policy succeeds 35% of the time). Because the imitation loss weights every frame equally, the policy receives abundant supervision for the easy transport and only faint supervision for the phase that actually determines success. Worse, an expert almost never enters a misaligned state, so the data shows the robot what success looks like but rarely how to recover toward it.

![](images/29207ea38a005948089d3bc2794d01846cfb55a5dbcac3cd0d33b7f471f9ffec.jpg)  
Figure 1: Where policy gains actually live. On a pen-cap insertion task, a fluent expert (top) spends ≈ 90% of frames on free-space transport and compresses the decisive alignment phase into ≈ 10% of frames, yielding 35% success. A deliberate demonstration (bottom) redistributes time toward alignment (≈ 50% of frames), exposing the corrective micro-adjustments near contact and raising success to 75%. Success is decided in the alignment phase, yet fluent demonstrations supervise it the least.

A first response is to fix the data, and we explore two remedies. Most directly, we change the demonstrations themselves: operators transport naturally but act as teachers near contact, slowing down, making small corrective motions, and occasionally approaching off-axis before recovering. We call these deliberate demonstrations (Fig. 1, bottom). Alternatively, we keep the original data and resample it, upweighting frames around task-critical moments such as gripper changes, contact, and insertion [3, 4, 5, 6]. Both improve alignment success, confirming that the learning signal is imbalanced across the trajectory. Crucially, however, they help for different reasons: resampling merely replays the frames the dataset already contains, whereas deliberate collection expands the coverage of correction and recovery states the policy would otherwise never see. This is why deliberate demonstrations win, and why data-only fixes hit a ceiling: slow collection is costly, and resampling can neither create recovery behavior nor add temporal detail that was never recorded.

These data-level fixes share a blind spot: they change how many alignment frames the policy sees, not how it sees each one. If the policy still maps a single static image to an action, the local motion that governs correction, whether the gripper is drifting, whether contact has begun, whether the error is shrinking, stays implicit. We therefore attack the bottleneck at the representation level and introduce STAIR, a compact dynamic feature for fine alignment: instead of reading a single frame, STAIR trains the vision-language model to predict how the local scene evolves over a short video neighborhood and uses this prediction to condition the action expert. The supervision is essentially free, distilled from short-horizon motion already recorded in every trajectory, so even fluent data yields dense signal about contact and correction. At deployment the neighborhood is no longer needed: the policy predicts the dynamic feature from the current frame alone, preserving the standard VLA interface. In short, STAIR does not ask for more frames, it extracts more signal from the frames already collected. Our contributions are threefold:

1. We identify a counterintuitive failure mode of imitation learning: fluent expert demonstrations can hurt fine manipulation by compressing the critical alignment and recovery behaviors that decide success.

2. We study two data-level remedies, deliberate slow-at-alignment collection and critical-segment resampling, and show that deliberate demonstrations win primarily by expanding recovery-state coverage rather than by reweighting existing frames.

3. We introduce STAIR, a compact dynamic feature bridging the VLM and action expert that recovers most of the deliberate-demonstration gain from fluent data alone (50.0 to 62.2% overall across six real-world tasks).

## 2 Related Work

## 2.1 Vision-Language-Action Models

Vision-language-action (VLA) models map language and visual observations to robot actions and are now a mainstream paradigm for general-purpose manipulation [7, 8, 9, 10, 11, 12], with recent work strengthening action parameterizations [13, 14, 15] and latent representations [16, 17, 18, 19, 20, 21]. These models condition each action on a per-step visual state, yet fine alignment is decided by short critical segments that fluent demonstrations compress into few frames. STAIR keeps the VLA interface intact but replaces this static per-step state with a compact dynamic feature, conditioning the action expert on local motion rather than a single appearance snapshot.

## 2.2 Robotic Representation Learning

Robotic representation learning exposes task-relevant structure that raw images hide, via semantic or spatial priors [22, 23, 24, 25, 26, 27] or motion-oriented features [28, 29, 30]. StaMo [31] differences two static state tokens, which breaks down when the decisive signal is short-horizon, while a complementary line predicts a compact future condition for general future modeling [27]. STAIR shares the compact-condition view but targets the fine-alignment bottleneck: rather than differencing endpoints or summarizing the full horizon, it compresses a short temporal neighborhood around the action chunk into an action-shaped dynamic feature carrying the correction and contact cues fluent demonstrations leave implicit.

## 2.3 Data Curation for Robot Learning

Beyond scaling dataset coverage [32, 33, 34, 35], data-centric work studies how composition affects transfer [36, 5] and curates demonstrations by filtering for quality [37, 5], rebalancing mixtures [6], retrieving relevant transitions [38, 39], or querying novel and risky states [40, 41]. Our slow-atalignment protocol is trajectory-centric, densifying supervision where visual change determines corrective action. Crucially, our controlled experiments separate two effects prior reweighting conflates: denser alignment frames help, but the larger gain comes from broader coverage of correction and recovery states that resampling existing frames cannot provide. STAIR is the representation-side counterpart, recovering this signal from ordinary fluent demonstrations without curated collection.

## 3 Method

## 3.1 Formalizing the Alignment Bottleneck

We begin by formalizing the bottleneck revealed by fluent demonstrations. A trajectory $\tau =$ $\{ ( o _ { t } , a _ { t } ) \} _ { t = 1 } ^ { T }$ splits into a transport phase $\mathcal { T } _ { \mathrm { t r } }$ , where the robot moves toward the target, and an alignment phase $\mathcal { T } _ { \mathrm { a l } }$ , where the final contact, insertion, or placement is decided. Let $\alpha = | { \mathcal { T } } _ { \mathrm { a l } } | / T$ be the fraction of frames spent in alignment. Since most imitation losses weight every step equally, the objective decomposes as

$$
\mathcal {L} _ {\mathrm{BC}} = (1 - \alpha) \mathcal {L} _ {\mathrm{tr}} + \alpha \mathcal {L} _ {\mathrm{al}},\tag{1}
$$

so a fluent demonstration with small α spends almost all of its training weight on transport.

Yet importance does not scale with duration. With phase action-sensitivity $\sigma _ { \phi } \propto$ $\left| \partial P ( { \mathrm { s u c c e s s } } ) / \partial a _ { t } \right|$ , fine manipulation obeys $\sigma _ { \mathrm { a l } } \gg \sigma _ { \mathrm { t r } } :$ a transport error is recoverable, while the same error near contact causes collision, jamming, or irreversible drift. Fluent demonstrations thus induce a duration–sensitivity mismatch, allocating weight by how long a phase lasts rather than how much it decides success, and the signal is also temporal: a single frame near alignment rarely reveals whether to continue, stop, slide, or recover. We attack this bottleneck with three remedies across two levels: at the data level, increasing the number of alignment frames (deliberate demonstrations) and their weight (critical-segment resampling); and at the representation level, how each frame is encoded (dynamic visual features).

## 3.2 Deliberate Demonstrations

The most direct remedy is to change the demonstration protocol itself. Rather than executing the task as efficiently as possible, operators perform transport naturally but act like teachers near contact: they slow down, make small corrective motions, and occasionally approach with a mild offset before recovering. We call this deliberate demonstration. By construction it raises the alignment-frame ratio, $\alpha _ { \mathrm { d e l i b } } > \alpha _ { \mathrm { f l u e n t } }$ , so under the uniform loss of Eq. 1 the effective training weight on the highsensitivity phase increases without touching the model or objective. More importantly, it exposes the local correction process that fluent experts compress away, how pose errors are reduced, how contact is handled, and how to recover from slight misalignment, precisely the corrective behaviors a policy needs at rollout. Deliberate demonstrations are thus not merely slower; they spend time where the policy is most sensitive. Sec. 4.3 compares fluent and deliberate data under a matched policy, task set, training budget, and trajectory count.

## 3.3 Critical-Segment Resampling

Deliberate collection changes the data distribution but requires new demonstrations. As a trainingtime alternative, we keep the original trajectories fixed and instead sample task-critical segments more often. Given a set of key moments $\mathcal { K } = \{ k _ { 1 } , \dots , k _ { M } \}$ (gripper opening or closing, contact, insertion, placement), we form a critical window $\begin{array} { r } { \mathcal { W } = \{ t : \operatorname* { m i n } _ { k \in \mathcal { K } } | t - k | \le r \} } \end{array}$ of radius $^ { r , }$ upweight frames inside it by $\lambda ( w _ { t } = \lambda \operatorname { i f } t \in \mathcal { W }$ , else 1), and train with

$$
\mathcal {L} _ {\mathrm{resample}} = \frac {1}{\sum_ {t} w _ {t}} \sum_ {t = 1} ^ {T} w _ {t} \ell \big (\pi_ {\theta} (o _ {t}, l), a _ {t} \big).\tag{2}
$$

Resampling is the training-time analogue of slowing down: it raises the effective contribution of alignment frames without altering the recorded trajectory, isolating how much of the gain is simply more supervision on high-sensitivity segments. Its limitation is equally clear: replaying the same frames cannot create missing recovery behavior or add temporal detail never captured, and overweighting narrow windows reduces batch diversity. Sec. 4.3 compares resampling against deliberate demonstrations.

## 3.4 Dynamic Alignment Features

Deliberate collection and critical-segment resampling both expose more supervision around alignment, but the policy still has to infer local motion from isolated observations. STAIR addresses the same bottleneck at the representation level: it supplies the action expert with a compact dynamic feature for local motion during alignment, so correction and contact progress do not have to be recovered from a single image.

For each training sample, we associate the supervised action chunk with a short dynamic observation clip,

$$
\mathcal {N} _ {t} = \{o _ {t + \delta} \} _ {\delta \in \Delta},\tag{3}
$$

where $\Delta$ denotes a small set of temporal offsets around the action chunk. We first encode this future observation neighbourhood with a frozen WanVAE encoder $E _ { \mathrm { w a n } }$ . A trainable projector $P _ { \psi }$ implemented as a 3D convolution followed by a Q-Former, then compresses the video latent into compact dynamic tokens:

$$
\mathbf {f} _ {t} ^ {\mathrm{dyn}} = P _ {\psi} (E _ {\mathrm{wan}} (\mathcal {N} _ {t})).
$$

![](images/e7d167a1862e30f17e8d0a1d64f4e9746caa5e86f9747d9380d13ac63b08563b.jpg)  
Figure 2: Dynamic alignment feature.

(4)

The feature $\mathbf f _ { t } ^ { \mathrm { d y n } }$ is not the full Wan VAE latent used by the flow-matching model. It is the compact token sequence produced after the frozen projector compresses the local video segment. Because the tokens are computed from local dynamics, they can carry cues such as moving into contact, slipping away, or correcting toward the target, while giving the policy a concentrated summary rather than raw frames or high-dimensional video latents.

The design keeps the compact-state view, but the token now summarizes local dynamics rather than static appearance. The policy does not need raw visual history if a small action-relevant token set preserves the changes needed for control. This is where fluent demonstrations are thinnest: the correction around contact is brief, but small differences there often decide success. The detailed training protocol is provided in the supplementary material.

## 3.5 Training and Inference

Our policy is a vision-language-action model that couples a VLM backbone [7] with a flow-matching DiT action expert. Given the current observation $o _ { t }$ and language instruction l, the VLM produces a latent representation $z _ { t } = f _ { \mathrm { V L M } } ( o _ { t } , l )$ . The action expert then predicts an action chunk $A _ { t : t + H }$ . The dynamic alignment feature is first mapped by an MLP into the action-conditioning space and then enters the action expert through cross-attention, so the action model conditions on both the current VLM state and local motion

Training uses a two-stage protocol. In Stage I, only the WanVAE encoder $E _ { \mathrm { w a n } }$ is frozen. We jointly train the projector $P _ { \psi }$ , the MLP adapter, the action expert, and the selected $\mathrm { V L A }$ modules. Writing $\mathbf { c } _ { t } ^ { \mathrm { { d y n } } } = \mathrm { { M L P } } _ { \eta } ( \mathbf { f } _ { t } ^ { \mathrm { { d y n } } } )$ for the projected dynamic condition, the action flow-matching objective is

$$
\mathcal {L} _ {\mathrm{I}} = \mathbb {E} _ {s, A} \left[ \left\| v _ {\theta} (A _ {s}, s, z _ {t}, \mathbf {c} _ {t} ^ {\mathrm{dyn}}) - v ^ {*} \right\| _ {2} ^ {2} \right],\tag{5}
$$

where s is the flow timestep and $v ^ { * }$ is the target velocity. The action gradient flows back through $\mathrm { M L P } _ { \eta }$ and the projector $P _ { \psi }$ (but not the frozen WanVAE encoder), shaping the dynamic tokens to be action-relevant.

In Stage II, the entire dynamic-feature pathway $( E _ { \mathrm { w a n } }$ , the projector $P _ { \psi }$ , and the MLP adapter) is fixed to provide a stable target. STAIR adds a QFormer query module on top of the VLM hidden states and trains it to predict the same projected dynamic condition from the current observation and instruction:

$$
\widehat {\mathbf {c}} _ {t} ^ {\mathrm{dyn}} = q _ {\phi} (\text { hidden } (f _ {\mathrm{VLM}} (o _ {t}, l))).\tag{6}
$$

We supervise this prediction with a cosine distance to the frozen projected-condition target,

$$
\mathcal {L} _ {\mathrm{dyn}} = 1 - \mathcal {S} (\widehat {\mathbf {c}} _ {t} ^ {\mathrm{dyn}}, \mathbf {c} _ {t} ^ {\mathrm{dyn}}),\tag{7}
$$

and continue training the action expert with the frozen projected condition:

$$
\mathcal {L} _ {\mathrm{II}} = \mathcal {L} _ {\mathrm{action}} (z _ {t}, \mathbf {c} _ {t} ^ {\mathrm{dyn}}) + \lambda_ {\mathrm{dyn}} \mathcal {L} _ {\mathrm{dyn}}.\tag{8}
$$

During Stage II, the action expert still receives the frozen projected condition. The query module learns the target while the action condition stays fixed.

At inference time, the video neighbourhood is no longer available. The policy receives only the current RGB observation and instruction. The VLM produces $z _ { t } ,$ , the query module predicts $\dot { \bf c } _ { t } ^ { \mathrm { d y n } }$ and the action expert uses $( z _ { t } , \widehat { \mathbf { c } } _ { t } ^ { \mathrm { { d y n } } } )$ to generate the action chunk. Dynamic feature extraction is only used during training, while deployment keeps the same observation interface as a standard VLA policy.

## 4 Experiments

Our experiments are organized around four questions: (Q1) Do data-level remedies (deliberate collection and critical-segment resampling) improve fine alignment, and is the effect model-agnostic?

We test this with $\pi _ { 0 . 5 }$ as an independent base (Sec. 4.2). (Q2) Can our model-level dynamic feature recover the deliberate-demonstration gain from fluent data alone on a controlled architecture (Sec. 4.3)? (Q3) Are the gains explained by recovery-state coverage, as the bottleneck analysis predicts (Sec. 4.4)? (Q4) Does STAIR remain competitive as a general policy (Sec. 4.5)? Q1–Q3 are studied on a real robot across six tasks of increasing alignment difficulty; Q4 is verified on SimplerEnv.

## 4.1 Setup

Hardware All real-world experiments are conducted on a dual-arm Piper robot platform from AgileX Robotics, as shown in Fig. 3. For visual observation, we use one Intel RealSense D435 as the primary camera and two wrist-mounted Intel RealSense D405 cameras, one on each arm. All cameras operate at 30 FPS and capture images at a resolution of $6 4 0 \times 3 6 0$

![](images/9733f0159de25908365f20edb55a0b41b0e250e0f837431b273d3b041ac05263.jpg)

Tasks We design six fine-grained manipulation tasks grouped into three difficulty levels by the alignment tolerance and the contact complexity of the final phase, as illustrated in Fig. 4. From easy to hard, the tasks are Stack Bowls, Stack Lego Blocks, Insert Flowers, Insert Pen Cap, Put Coin into Safe, and

Figure 3: Setup.

Open the Lock, with alignment requirements ranging from coarse position tolerance to multi-stage, contact-aware correction.

![](images/38d4a8ab2502ee1cc2098b1be1a0b1e91f789d7407a9c5bb2f90ff770b96c779.jpg)  
Figure 4: Six real-world manipulation tasks grouped by alignment difficulty.

Data conditions and protocol. For every task we collect 100 demonstrations and train by SFT on the mixture, evaluating each policy with 15 trials per task. We compare three data conditions: Fluent (efficient expert demonstrations), Deliberate (slow-at-alignment demonstrations, matched at 100 trajectories per task), and Resample (critical-segment reweighting of the fluent set; Sec. 3.3). Fluent and deliberate sets are matched in trajectory count; deliberate trajectories are longer and thus contain more frames, an effect we isolate in Sec. 4.4.

Base policies. To show the data-level effect is not tied to one architecture, we study it on two bases. Part A uses $\pi _ { 0 . 5 }$ [13], a strong and independent policy, to establish that the effect is modelagnostic. Part B then studies our model-level remedy on a controlled VLA of our own (a VLM with a flow-matching DiT action expert; Sec. 3.5). Here the Baseline is STAIR with the dynamic-feature pathway ablated: it keeps the same VLM, action expert, and query module, but is trained with the action loss only and without the video encoder, so that Baseline and STAIR differ by exactly one factor, whether the conditioning tokens are supervised by the dynamic feature. Training schedules for both bases are given in the supplementary material.

## 4.2 Data-Level Remedies are Model-Agnostic

We first establish, on the $\pi _ { 0 . 5 }$ base, that the data-level remedies improve fine alignment independently of our model contribution. Table 1 reports success rates across the six tasks. Deliberate collection improves over fluent demonstrations, and the gap widens with alignment difficulty; criticalsegment resampling recovers part of this gain without new data. Because $\pi _ { 0 . 5 }$ shares none of our model-level components, this isolates the data effect itself.

Table 1: Part A: $\pi _ { 0 . 5 }$ base. Success rate (%), averaged by difficulty level. Data effect on an independent policy.

<table><tr><td>Data</td><td>Easy</td><td>Mid.</td><td>Hard</td><td>Overall</td></tr><tr><td>Fluent</td><td>90.0%</td><td>56.7%</td><td>10.0%</td><td>52.2%</td></tr><tr><td>Deliberate</td><td>93.3%</td><td>73.3%</td><td>36.7%</td><td>67.8%</td></tr><tr><td>+ Resample</td><td>90.0%</td><td>66.7%</td><td>20.0%</td><td>58.9%</td></tr></table>

Table 2: Part B: Controlled study on our VLA base.

<table><tr><td>Method</td><td>Data</td><td>Easy</td><td>Mid.</td><td>Hard</td><td>Overall</td></tr><tr><td>Baseline</td><td>Fluent</td><td>90.0%</td><td>40.0%</td><td>3.3%</td><td>44.4%</td></tr><tr><td>Baseline</td><td>Deliberate</td><td>93.3%</td><td>56.7%</td><td>20.0%</td><td>56.7%</td></tr><tr><td>+ Resample</td><td>Fluent</td><td>90.0%</td><td>46.7%</td><td>10.0%</td><td>48.9%</td></tr><tr><td>+ STAIR (ours)</td><td>Fluent</td><td>93.3%</td><td>56.7%</td><td>13.3%</td><td>54.4%</td></tr><tr><td>+ STAIR (ours)</td><td>Deliberate</td><td>96.7%</td><td>63.3%</td><td>26.7%</td><td>62.2%</td></tr></table>

Table 3: Per-task success rates (%) on the six real-world tasks. Same setup as Table 2, broken out per task.

<table><tr><td rowspan="2">Method</td><td rowspan="2">Data</td><td colspan="2">Easy</td><td colspan="2">Middle</td><td colspan="2">Hard</td><td rowspan="2">Avg</td></tr><tr><td>Stack Bowls</td><td>Stack Lego</td><td>Insert Flowers</td><td>Insert Pen Cap</td><td>Coin into Safe</td><td>Open Lock</td></tr><tr><td>Baseline</td><td>Fluent</td><td>93.3%</td><td>86.7%</td><td>53.3%</td><td>26.7%</td><td>6.7%</td><td>0.0%</td><td>44.4%</td></tr><tr><td>Baseline</td><td>Deliberate</td><td>93.3%</td><td>93.3%</td><td>66.7%</td><td>46.7%</td><td>26.7%</td><td>13.3%</td><td>56.7%</td></tr><tr><td>+ Resample</td><td>Fluent</td><td>93.3%</td><td>86.7%</td><td>60.0%</td><td>33.3%</td><td>13.3%</td><td>6.7%</td><td>48.9%</td></tr><tr><td>+ STAIR (ours)</td><td>Fluent</td><td>93.3%</td><td>93.3%</td><td>66.7%</td><td>46.7%</td><td>20.0%</td><td>6.7%</td><td>54.4%</td></tr><tr><td>+ STAIR (ours)</td><td>Deliberate</td><td>100.0%</td><td>93.3%</td><td>73.3%</td><td>53.3%</td><td>33.3%</td><td>20.0%</td><td>62.2%</td></tr></table>

## 4.3 STAIR on a Controlled Architecture

Table 2 reports success rates on our VLA base, averaged within each difficulty level and overall. Three patterns emerge. First, the data trend of Part A reappears on this architecture: the deliberate baseline beats the fluent one by a margin that widens with alignment difficulty, and resampling the fluent set recovers part of the gap, confirming the data effect is not specific to $\pi _ { 0 . 5 } .$ . Second, STAIR trained on fluent data alone recovers most of the deliberate-demonstration gain, approaching the deliberate baseline without changing the data, which is the central result of the paper. Third, STAIR trained on deliberate data is strongest overall, showing that the data-level and model-level remedies are complementary. Table 3 reports per-task success rates, confirming the averaged trend is not driven by a single dominant task. The largest single-task gaps appear on Insert Pen Cap, Put Coin into Safe, and Open the Lock, which require multi-millimeter alignment under partial occlusion.

## 4.4 Why Deliberate Demonstrations Outperform Resampling

Deliberate demonstrations consistently outperform resampling (Table 2) even though both raise the effective alignment weight. Fig. 5 explains why: we tune resampling so its alignment budget (88 frames per trajectory) matches deliberate’s (95), holding the weight fixed, yet resampling inherits the narrow state coverage of the fluent set while deliberate substantially expands it with off-axis approaches and recovery. The remaining gap is therefore a coverage effect, not a weight effect: once alignment is sufficiently weighted, success is limited by whether the data exposes the local correction process, which replaying the same frames cannot create. This is precisely the gap that the model-level dynamic feature (Sec. 3.4) closes by extracting the correction signal from video.

## 4.5 Generalization on SimplerEnv

To test whether the dynamic feature is generally useful rather than specific to fine alignment, we evaluate STAIR on SimplerEnv [42], a standard general-manipulation benchmark, against a broad set of state-of-the-art VLA models. Here STAIR is trained purely on large-scale robot data (no fluent/deliberate distinction): pretrained on the OXE mixture, then trained on Bridge and Fractal, and at inference it receives only the current RGB observation, exactly like a standard VLA. We report 30 closed-loop rollouts on each of nine tasks (five Google Robot, four WidowX Bridge); training and protocol details are in the supplementary material.

As shown in Tables 4 and 5, STAIR is competitive with state-of-the-art generalist policies on both robots, with gains spanning grasping and final placement. Conditioning on compact dynamic feature thus benefits general manipulation broadly, not only the fine-alignment tasks that motivate it.

![](images/450fc1a2aef98428acbb502002a8b325632e26e5cd3862131ebf8c70af2f364b.jpg)  
Figure 5: Alignment-phase coverage vs. effective training weight. Each panel is one demonstration set. The scatter plots the end-effector position relative to the target (∆x, ∆z) over all alignmentphase frames, with the success region in blue; a wider spread means broader state coverage. The inset bar chart reports the average number of frames per trajectory in the transport and alignment phases (for resampling, the alignment bar is the effective count after up-weighting). Resampling raises the alignment count from 22 to 88 yet keeps fluent’s narrow coverage, whereas deliberate demonstrations reach a comparable count (95) while expanding coverage with off-axis approaches and recovery.

Table 4: SimplerEnv evaluation across different models on Google Robot tasks.

<table><tr><td rowspan="2">Model</td><td colspan="4">Visual Matching</td><td colspan="4">Variant Aggregation</td><td rowspan="2">Overall Avg.</td></tr><tr><td>Pick Coke</td><td>Move Near</td><td>Drawer</td><td>Avg.</td><td>Pick Coke</td><td>Move Near</td><td>Drawer</td><td>Avg.</td></tr><tr><td>RT-1-X [43]</td><td>56.7%</td><td>31.7%</td><td>59.7%</td><td>53.4%</td><td>49.0%</td><td>32.3%</td><td>29.4%</td><td>39.7%</td><td>46.6%</td></tr><tr><td>Octo-Base [44]</td><td>17.0%</td><td>4.2%</td><td>22.7%</td><td>16.8%</td><td>0.6%</td><td>3.1%</td><td>1.1%</td><td>1.2%</td><td>9.0%</td></tr><tr><td> $\pi_0$  [45]</td><td>72.7%</td><td>65.3%</td><td>38.3%</td><td>58.8%</td><td>75.2%</td><td>63.7%</td><td>25.6%</td><td>54.8%</td><td>56.8%</td></tr><tr><td> $\pi_0$ -FAST [46]</td><td>75.3%</td><td>67.5%</td><td>42.9%</td><td>61.9%</td><td>77.6%</td><td>68.2%</td><td>31.3%</td><td>59.0%</td><td>60.5%</td></tr><tr><td>OpenVLA [7]</td><td>16.3%</td><td>46.2%</td><td>35.6%</td><td>32.7%</td><td>54.5%</td><td>47.7%</td><td>17.7%</td><td>39.8%</td><td>33.8%</td></tr><tr><td>GR00T-N1 [15]</td><td>47.0%</td><td>70.0%</td><td>18.1%</td><td>45.0%</td><td>78.8%</td><td>62.5%</td><td>13.2%</td><td>51.5%</td><td>48.4%</td></tr><tr><td>Moto [29]</td><td>74.0%</td><td>60.4%</td><td>43.1%</td><td>59.2%</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Ours</td><td>84.0%</td><td>72.5%</td><td>62.3%</td><td>72.9%</td><td>81.9%</td><td>75.0%</td><td>27.3%</td><td>61.4%</td><td>67.2%</td></tr></table>

Table 5: SimplerEnv evaluation across different models on WidowX Robot tasks.

<table><tr><td rowspan="2">Model</td><td colspan="2">Put Spoon on Towel</td><td colspan="2">Stack Green on Yellow</td><td colspan="2">Put Carrot on Plate</td><td colspan="2">Put Eggplant in Basket</td><td colspan="2">Overall Average</td></tr><tr><td>Grasp Spoon</td><td>Success</td><td>Grasp G Block</td><td>Success</td><td>Grasp Carrot</td><td>Success</td><td>Grasp Eggplant</td><td>Success</td><td>Grasp Avg.</td><td>Success Avg.</td></tr><tr><td>RT-1-X [43]</td><td>16.7%</td><td>0.0%</td><td>8.3%</td><td>0.0%</td><td>20.8%</td><td>4.2%</td><td>0.0%</td><td>0.0%</td><td>11.5%</td><td>1.1%</td></tr><tr><td>Octo-Base [44]</td><td>34.7%</td><td>12.5%</td><td>31.9%</td><td>0.0%</td><td>52.8%</td><td>8.3%</td><td>66.7%</td><td>43.1%</td><td>46.5%</td><td>16.0%</td></tr><tr><td>SpatialVLA [47]</td><td>25.0%</td><td>20.8%</td><td>58.3%</td><td>25.0%</td><td>41.7%</td><td>20.8%</td><td>79.2%</td><td>70.8%</td><td>51.1%</td><td>34.4%</td></tr><tr><td> $\pi_0$  [45]</td><td>45.8%</td><td>29.1%</td><td>25.0%</td><td>0.0%</td><td>50.0%</td><td>16.7%</td><td>91.6%</td><td>62.5%</td><td>40.1%</td><td>27.1%</td></tr><tr><td> $\pi_0$ -FAST [46]</td><td>62.5%</td><td>29.1%</td><td>58.5%</td><td>21.9%</td><td>54.0%</td><td>10.8%</td><td>83.3%</td><td>66.6%</td><td>48.3%</td><td>32.1%</td></tr><tr><td>OpenVLA [7]</td><td>4.1%</td><td>0.0%</td><td>33.0%</td><td>0.0%</td><td>12.5%</td><td>0.0%</td><td>8.3%</td><td>4.1%</td><td>7.8%</td><td>1.1%</td></tr><tr><td>GR00T-N1 [15]</td><td>83.3%</td><td>62.5%</td><td>54.2%</td><td>45.8%</td><td>70.8%</td><td>16.7%</td><td>41.7%</td><td>20.8%</td><td>49.5%</td><td>36.5%</td></tr><tr><td>UniVLA [16]</td><td>76.4%</td><td>52.8%</td><td>66.7%</td><td>2.8%</td><td>79.2%</td><td>55.6%</td><td>87.5%</td><td>66.7%</td><td>77.5%</td><td>45.6%</td></tr><tr><td>Ours</td><td>81.8%</td><td>69.2%</td><td>73.1%</td><td>62.3%</td><td>76.8%</td><td>57.4%</td><td>88.6%</td><td>73.4%</td><td>80.1%</td><td>65.6%</td></tr></table>

## 5 Conclusion

We studied why fluent expert demonstrations can be poor teachers for fine-grained manipulation: skilled operators compress the alignment and recovery motions that decide success into a brief window, leaving the policy under-supervised exactly where precision matters. We addressed this at two levels: deliberate collection and critical-segment resampling add supervision on these highsensitivity moments, while STAIR distills the same signal into a compact dynamic feature shared by the VLM and the action expert. More broadly, our results argue that robust alignment depends less on perfectly efficient trajectories than on demonstrations and representations that make correction visible to the learner.

## 6 Limitations

Our approach has several limitations. At inference time, the dynamic feature is predicted deterministically from the current observation, which cannot capture multi-modal or partially observed contact futures. It also relies primarily on visual dynamics, and therefore does not replace explicit geometric, tactile, or force feedback in highly constrained contact tasks. Finally, key-frame and neighbourhood selection are currently treated as hyperparameters; learning task-conditioned temporal selection remains an important direction for future work.

## References

[1] H. Ravichandar, A. S. Polydoros, S. Chernova, and A. Billard. Recent advances in robot learning from demonstration. Annual review of control, robotics, and autonomous systems, 3 (1):297–330, 2020.

[2] M. Laskey, J. Lee, R. Fox, A. Dragan, and K. Goldberg. Dart: Noise injection for robust imitation learning. In Conference on robot learning, pages 143–156. PMLR, 2017.

[3] B. Akgun, M. Cakmak, K. Jiang, and A. L. Thomaz. Keyframe-based learning from demonstration: Method and evaluation. International Journal ofSocial Robotics, 4(4):343–355, 2012.

[4] L. Kou, F. Ni, Y. ZHENG, J. Liu, Y. Yuan, Z. Dong, and J. HAO. KISA: A unified keyframe identifier and skill annotator for long-horizon robotics demonstrations. In Forty-first International Conference on Machine Learning, 2024. URL https://openreview.net/forum? id=oCI9gHocws.

[5] J. Hejna, S. Mirchandani, A. Balakrishna, A. Xie, A. Wahid, J. Tompson, P. Sanketi, D. Shah, C. Devin, and D. Sadigh. Robot data curation with mutual information estimators. arXiv preprint arXiv:2502.08623, 2025.

[6] J. Hejna, C. Bhateja, Y. Jiang, K. Pertsch, and D. Sadigh. Re-mix: Optimizing data mixtures for large scale imitation learning. arXiv preprint arXiv:2408.14037, 2024.

[7] M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. Foster, G. Lam, P. Sanketi, et al. Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024.

[8] B. Zitkovich, T. Yu, S. Xu, P. Xu, T. Xiao, F. Xia, J. Wu, P. Wohlhart, S. Welker, A. Wahid, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. In Conference on Robot Learning, pages 2165–2183. PMLR, 2023.

[9] Y. Wang, H. Zhu, M. Liu, J. Yang, H.-S. Fang, and T. He. Vq-vla: Improving vision-languageaction models via scaling vector-quantized action tokenizers. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 11089–11099, 2025.

[10] M. Liu, Z. Huang, X. Lin, M. Zhu, C. Zhao, Z. Du, Y. Wang, H. Zhu, H. Chen, and C. Shen. Bridge thinking and acting: Unleashing physical potential of vlm with generalizable action expert. arXiv preprint arXiv:2510.03896, 2025.

[11] Z. Huang, M. Liu, X. Lin, M. Zhu, C. Zhao, Z. Du, Y. Lin, X. Li, Y. Jia, H. Zhong, et al. Notvla: Semantics-preserving robot adaptation via narrative action interfaces. arXiv preprint arXiv:2510.03895, 2025.

[12] K. Wang, L. Lu, M. Liu, J. Jiang, Z. Li, B. Zhang, W. Zheng, X. Yu, H. Chen, and C. Shen. Odyssey: Open-world quadrupeds exploration and manipulation for long-horizon tasks. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 18602–18610, 2026.

[13] P. Intelligence, K. Black, N. Brown, J. Darpinian, K. Dhabalia, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, et al. π<sub>0.5</sub>: A vision-language-action model with open-world generalization. arXiv preprint arXiv:2504.16054, 2025.

[14] K. Black, N. Brown, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, L. Groom, K. Hausman, B. Ichter, et al. \π : A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024.

[15] J. Bjorck, F. Castaneda, N. Cherniadev, X. Da, R. Ding, L. Fan, Y. Fang, D. Fox, F. Hu, ˜ S. Huang, et al. Gr00t n1: An open foundation model for generalist humanoid robots. arXiv preprint arXiv:2503.14734, 2025.

[16] Q. Bu, Y. Yang, J. Cai, S. Gao, G. Ren, M. Yao, P. Luo, and H. Li. Univla: Learning to act anywhere with task-centric latent actions. arXiv preprint arXiv:2505.06111, 2025.

[17] J. Cen, C. Yu, H. Yuan, Y. Jiang, S. Huang, J. Guo, X. Li, Y. Song, H. Luo, F. Wang, et al. Worldvla: Towards autoregressive action world model. arXiv preprint arXiv:2506.21539, 2025.

[18] W. Zhang, H. Liu, Z. Qi, Y. Wang, X. Yu, J. Zhang, R. Dong, J. He, H. Wang, Z. Zhang, et al. Dreamvla: a vision-language-action model dreamed with comprehensive world knowledge. Advances in Neural Information Processing Systems, 38:24195–24228, 2026.

[19] H. Liang, X. Chen, B. Wang, M. Chen, Y. Liu, Y. Zhang, Z. Chen, T. Yang, Y. Chen, J. Pang, et al. Mm-act: Learn from multimodal parallel generation to act. arXiv preprint arXiv:2512.00975, 2025.

[20] S. Liu, L. Wu, B. Li, H. Tan, H. Chen, Z. Wang, K. Xu, H. Su, and J. Zhu. Rdt-1b: a diffusion foundation model for bimanual manipulation. In International Conference on Learning Representations, volume 2025, pages 29982–30009, 2025.

[21] J. Yang, Y. Shi, H. Zhu, M. Liu, K. Ma, Y. Wang, G. Wu, T. He, and L. Wang. Como: Learning continuous latent motion from internet videos for scalable robot learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 42352–42363, 2026.

[22] S. Nair, A. Rajeswaran, V. Kumar, C. Finn, and A. Gupta. R3m: A universal visual representation for robot manipulation. arXiv preprint arXiv:2203.12601, 2022.

[23] Y. J. Ma, S. Sodhani, D. Jayaraman, O. Bastani, V. Kumar, and A. Zhang. Vip: Towards universal visual reward and representation via value-implicit pre-training. arXiv preprint arXiv:2210.00030, 2022.

[24] M. Caron, H. Touvron, I. Misra, H. Jegou, J. Mairal, P. Bojanowski, and A. Joulin. Emerging´ properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9650–9660, 2021.

[25] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.

[26] S. Karamcheti, S. Nair, A. S. Chen, T. Kollar, C. Finn, D. Sadigh, and P. Liang. Languagedriven representation learning for robotics. arXiv preprint arXiv:2302.12766, 2023.

[27] Y. Su, S. Chen, H. Shi, M. Liu, Z. Zhang, N. Huang, W. Zhong, Z. Zhu, Y. Liu, and X. Liu. World guidance: World modeling in condition space for action generation. arXiv preprint arXiv:2602.22010, 2026.

[28] S. Ye, J. Jang, B. Jeon, S. J. Joo, J. Yang, B. Peng, A. Mandlekar, R. Tan, Y.-W. Chao, B. Y. Lin, et al. Latent action pretraining from videos. In International Conference on Learning Representations, volume 2025, pages 28213–28239, 2025.

[29] Y. Chen, Y. Ge, W. Tang, Y. Li, Y. Ge, M. Ding, Y. Shan, and X. Liu. Moto: Latent motion token as the bridging language for learning robot manipulation from videos. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 19752–19763, 2025.

[30] Y. Wang, X. Li, W. Wang, J. Zhang, Y. Li, Y. Chen, X. Wang, and Z. Zhang. Unified visionlanguage-action model. arXiv preprint arXiv:2506.19850, 2025.

[31] M. Liu, J. Shu, H. Chen, Z. Li, C. Zhao, J. Yang, S. Gao, H. Chen, and C. Shen. StaMo: Unsupervised learning of generalizable robot motion from compact state representation. arXiv preprint arXiv:2510.05057, 2025.

[32] A. O’Neill, A. Rehman, A. Maddukuri, A. Gupta, A. Padalkar, A. Lee, A. Pooley, A. Gupta, A. Mandlekar, A. Jain, et al. Open x-embodiment: Robotic learning datasets and rt-x models: Open x-embodiment collaboration 0. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 6892–6903. IEEE, 2024.

[33] A. Khazatsky, K. Pertsch, S. Nair, A. Balakrishna, S. Dasari, S. Karamcheti, S. Nasiriany, M. K. Srirama, L. Y. Chen, K. Ellis, et al. Droid: A large-scale in-the-wild robot manipulation dataset. arXiv preprint arXiv:2403.12945, 2024.

[34] F. Ebert, Y. Yang, K. Schmeckpeper, B. Bucher, G. Georgakis, K. Daniilidis, C. Finn, and S. Levine. Bridge data: Boosting generalization of robotic skills with cross-domain datasets. arXiv preprint arXiv:2109.13396, 2021.

[35] C. Chi, Z. Xu, C. Pan, E. Cousineau, B. Burchfiel, S. Feng, R. Tedrake, and S. Song. Universal manipulation interface: In-the-wild robot teaching without in-the-wild robots. arXiv preprint arXiv:2402.10329, 2024.

[36] J. Yang, C. Finn, and D. Sadigh. Data analogies enable efficient cross-embodiment transfer. arXiv preprint arXiv:2603.06450, 2026.

[37] S. Kuhar, S. Cheng, S. Chopra, M. Bronars, and D. Xu. Learning to discern: Imitating heterogeneous human demonstrations with preference and representation learning. In Conference on Robot Learning, pages 1437–1449. PMLR, 2023.

[38] M. Du, S. Nair, D. Sadigh, and C. Finn. Behavior retrieval: Few-shot imitation learning by querying unlabeled datasets. arXiv preprint arXiv:2304.08742, 2023.

[39] L.-H. Lin, Y. Cui, A. Xie, T. Hua, and D. Sadigh. Flowretrieval: Flow-guided data retrieval for few-shot imitation learning. arXiv preprint arXiv:2408.16944, 2024.

[40] Y. Cui, D. Isele, S. Niekum, and K. Fujimura. Uncertainty-aware data aggregation for deep imitation learning. In 2019 International Conference on Robotics and Automation (ICRA), pages 761–767. IEEE, 2019.

[41] R. Hoque, A. Balakrishna, E. Novoseller, A. Wilcox, D. S. Brown, and K. Goldberg. Thriftydagger: Budget-aware novelty and risk gating for interactive imitation learning. arXiv preprint arXiv:2109.08273, 2021.

[42] X. Li, K. Hsu, J. Gu, K. Pertsch, O. Mees, H. R. Walke, C. Fu, I. Lunawat, I. Sieh, S. Kirmani, et al. Evaluating real-world robot manipulation policies in simulation. arXiv preprint arXiv:2405.05941, 2024.

[43] A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, J. Dabis, C. Finn, K. Gopalakrishnan, K. Hausman, A. Herzog, J. Hsu, et al. Rt-1: Robotics transformer for real-world control at scale. arXiv preprint arXiv:2212.06817, 2022.

[44] O. M. Team, D. Ghosh, H. Walke, K. Pertsch, K. Black, O. Mees, S. Dasari, J. Hejna, T. Kreiman, C. Xu, et al. Octo: An open-source generalist robot policy. arXiv preprint arXiv:2405.12213, 2024.

[45] K. Black, N. Brown, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, L. Groom, K. Hausman, B. Ichter, et al. π0: A vision-language-action flow model for general robot control. corr, abs/2410.24164, 2024. doi: 10.48550. arXiv preprint ARXIV.2410.24164.

[46] K. Pertsch, K. Stachowicz, B. Ichter, D. Driess, S. Nair, Q. Vuong, O. Mees, C. Finn, and S. Levine. Fast: Efficient action tokenization for vision-language-action models. arXiv preprint arXiv:2501.09747, 2025.

[47] D. Wu, F. Liu, Y.-H. Hung, and Y. Duan. Spatial-mllm: Boosting mllm capabilities in visualbased spatial intelligence. arXiv preprint arXiv:2505.23747, 2025.