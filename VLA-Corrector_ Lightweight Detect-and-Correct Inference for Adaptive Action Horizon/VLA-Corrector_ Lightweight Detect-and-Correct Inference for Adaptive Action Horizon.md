OmniAI Group of ZJU ACES Lab Zhejiang University

O OmniAl

# VLA-Corrector: Lightweight Detect-and-Correct Inference for Adaptive Action Horizon

Yi Pan<sup>1</sup>, Miao Pan<sup>1</sup>, Qi Lu<sup>1</sup>, Jiaming Huang<sup>1</sup>, Man Zhang<sup>1</sup>, Siteng Huang<sup>2</sup>, Xin Li<sup>2</sup>, Jie Zhang<sup>1</sup>, Yongliang Shen<sup>1</sup>, Xuhong Zhang<sup>1</sup>, Wenqi Zhang

<sup>1</sup>Zhejiang University <sup>2</sup>Alibaba DAMO Academy

Vision-Language-Action (VLA) foundation models have recently achieved strong progress in embodied intelligence. To reduce policy-call frequency while preserving temporal coherence, most generative policies adopt an action chunk mechanism, executing multiple future actions in an open-loop manner under a fixed action horizon. However, this “predict-then-blindly-execute” paradigm sacrifices closedloop reactivity: in contact-rich physical interactions, even small local perturbations can rapidly amplify within the open-loop blind spot, leading to compounding errors and ultimately task failure. To address this limitation, we propose VLA-Corrector, a lightweight corrective inference framework for action-chunked VLA policies. Without modifying the backbone policy weights, VLA-Corrector introduces a lightweight Latent-space Vision Monitor (LVM) that continuously compares predicted and actual visual feature evolution, enabling online detection of visual dynamics deviations. Once persistent deviation is detected, the system triggers a truncation event, discards the remaining stale actions, and invokes corrective replanning via Online Gradient Guidance (OGG). The detect-andcorrect mechanism of VLA-Corrector naturally induces an event-triggered adaptive action horizon: it preserves long-horizon execution when the current chunk remains reliable, and invokes short-horizon corrective replanning when execution begins to drift. In doing so, VLA-Corrector mitigates the trade-of imposed by static horizons between execution robustness and policy-call frequency. It can be integrated into diferent VLA models without further retraining the VLA backbone, interrupting compounding errors while preserving much of the eficiency benefit of action chunking and substantially improving robustness in long-horizon, contact-rich robotic manipulation tasks. Our code is available at https://github.com/ZJU-OmniAI/vla-corrector.

Yi Pan: panyi0304@gmail.com Wenqi Zhang: zhangwenqi@zju.edu.cn Date: October 2026

## Contents

1 Introduction 3   
2 Preliminaries 5   
3 VLA-Corrector: A Lightweight Corrective Inference Framework 5   
3.1 Training the External Latent Dynamics Corrector . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5   
3.2 Latent Visual Dynamics for Online Anomaly Detection . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6   
3.3 Event-Triggered Truncation under Robust Online Monitoring 6   
3.4 Online Gradient Guidance for Corrective Inference 7   
4 Experiments 7   
4.1 Main Results 7   
4.2 Mechanism Analysis 9   
4.3 Real-World Evaluation 10   
4.4 Ablation Studies 11   
5 Conclusion 11   
A Related Work 15   
A.1 Generative VLA Models 15   
A.2 Action Chunk and Horizon Trade-off 15   
A.3 Failure Recovery in Visuomotor Policies 15   
B Method Details 16   
B.1 Details of Event-Triggered Truncation 16   
B.2 LVM and OGG Runtime Parameters 16   
C Training and Implementation Details 16   
C.1 Benchmarks and Evaluation Protocol 16   
C.2 External Corrector Architecture 17   
C.3 Corrector Training 17   
C.4 Compute Resources 17   
D Additional Experimental Results 18   
D.1 Ablation Sensitivity 18   
D.2 Cross-Domain Corrector Generalization 18   
D.3 Full Performance-Efficiency Results 19   
D.4 Data Efficiency of Corrector Training 19   
D.5 Inference-Time Cost and Control Delay 20   
D.6 Asynchronous and RTC Comparisons 21   
E Real-World Details 22   
E.1 Robot Platform and Task Suite 22   
E.2 Per-Task Real-World Results 23   
E.3 Real-World Protocol and Task Details 23   
E.4 Observed Real-World Failure Modes 24   
E.5 Real-World Demonstration Examples 24

## 1 Introduction

Vision-Language-Action (VLA) foundation models have emerged as a promising path toward general-purpose robot control, unifying perception, language, and action generation within a single framework Zitkovich et al. (2023); O’Neill et al. (2024); Kim et al. (2024); Black et al. (2024); Li et al. (2023); Bjorck et al. (2025); Intelligence et al. (2025); Ma et al. (2024); Wang et al. (2026a); Cen et al. (2025); Jiang et al. (2025b). Modern VLA policies often rely on generative action models, such as difusion models and flow matching, to capture the high-dimensional and multi-modal nature of continuous robot actions Black et al. (2024); Chi et al. (2025); Jiang et al. (2025a); Wang et al. (2024); Lipman et al. (2022). However, the per-step latency of generative models precludes closed-loop replanning, creating a fundamental tension between action expressiveness and high-frequency feedback control.

A common engineering compromise is to use an action chunk Zhao et al. (2023a): in a single forward pass, the policy predicts a sequence of future actions, while the controller only executes the first H steps as the action horizon Jing et al. (2025); Liu et al. (2024); Sendai et al. (2025); Chi et al. (2025); Khan and Tanimura (2025); So et al. (2025). This design amortizes policy-call frequency and improves temporal smoothness, but it also creates an open-loop blind spot Wang et al. (2026b); Xu et al. (2026) that introduces two compounding risks. First, the policy lacks real-time reactivity. Fresh observations arrive at every control step, yet the system ignores them until the horizon ends, leaving it unable to respond to unexpected slippage, collision, or pose drift as they occur. Second, errors can accumulate beyond the point that replanning can recover from during the blind spot. If deviations go uncorrected long enough, the robot may drift into an out-of-distribution state that the policy has rarely encountered during training, at which point even the next replanning call cannot steer execution back to the intended trajectory Wang et al. (2026b); Liu et al. (2024); Sendai et al. (2025); Wu et al. (2026a); Xu et al. (n.d.), and the task ultimately fails. Crucially, both risks worsen as the horizon grows longer. As illustrated in Fig. 1, under a long horizon (H = 10), the robot keeps executing stale actions after a deviation and eventually gets stuck; in contrast, with H = 1, the policy replans at every step and avoids the same error accumulation.

Yet H = 1 is not a practical solution: it requires a full VLA inference at every control step, negating the eficiency gains that action chunking was designed to provide. The contrast between these two extremes thus exposes a fundamental trade-of: smaller horizons improve closed-loop responsiveness but multiply policy-call frequency, while larger horizons amortize computation but widen the blind spot where errors silently accumulate.

To quantify this trade-of, we conduct systematic experiments comparing task success rate and policy frequency across a range of action horizons on three VLA backbones. As shown in Fig. 2, larger horizons substantially reduce policy calls but also lead to lower success rates, and this trend is consistently observed across all three backbones. For $\pi _ { 0 . 5 }$ Intelligence et al. (2025), for example, increasing the horizon reduces policy calls by about $4 \times ,$ but lowers success from about 64% to below 49%; similar trends hold for SmolVLA Shukor et al. (2025)

![](images/8f7032d603de1b2bf7d19995020dba561f9de1bd3a06427dd16a1a50c0150d55.jpg)  
Figure 2 Performance–efficiency trade-off across fixed action horizons. Smaller horizon achieves higher success rates, while larger horizon preserves the chunking eficiency.

and X-VLA Zheng et al. (2025). Furthermore, since the best horizon depends on task dificulty, environmental dynamics, and sim-to-real mismatch, it is dificult to predefine a single static horizon that remains optimal across scenarios. This suggests that the key is not to choose a better fixed horizon, but to decide when the current chunk should stop being trusted.

The above analysis raises two key questions: (1) How to detect execution deviations in time and terminate stale actions before errors compound beyond recovery; (2) How to correct the trajectory after truncation— since naive replanning alone is often insuficient: the VLA may re-generate actions that still fail to escape the

![](images/7339e02ef3719a42c404d9d84aaa470188bb8cefdd3d6f6770044a54c2964820.jpg)  
Figure 1 Open-loop vs. Closed-loop execution. The top row (H = 10) illustrates how blindly executing long action chunks leads to compounding errors, causing the robot to get stuck during a drawer-opening task. In contrast, the bottom row (H = 1) demonstrates strict closed-loop execution, which maintains environmental reactivity and yields a smooth, successful manipulation.

## deviated state, leaving the robot trapped again.

Motivated by these insights, we propose VLA-Corrector, a lightweight corrective inference framework that addresses both questions. For the first, VLA-Corrector introduces a Latent-space Vision Monitor (LVM) that continuously compares predicted and actual visual evolution during open-loop execution; once persistent deviation is detected, it triggers an interrupt event that immediately truncates the remaining stale actions and invokes a new action generation. For the second, rather than relying on naive replanning, VLA-Corrector activates Online Gradient Guidance (OGG), which exploits the discrepancy between predicted and observed visual dynamics to inject a corrective gradient during new action generation—actively steering the robot back toward the intended trajectory, instead of simply hoping that re-inference will naturally recover. Together, LVM-triggered truncation and OGG-guided re-inference transform a fixed action horizon into an adaptive action horizon with self-correcting re-inference, preserving long-horizon eficiency and short-horizon responsiveness.

These two mechanisms jointly improve success-per-call eficiency: the robot avoids wasting long chunks on stale actions, so each policy query contributes more efectively to task completion. For π<sub>0.5</sub> at horizon 50, success rises from 48.7% to 58.7% while average policy calls drop from 5.15 to 4.98—a +24.6% success-per-call eficiency gain. Gains are consistent across all three backbones and grow larger at longer horizons, where the open-loop blind spot is widest. Beyond robustness, VLA-Corrector also improves sample eficiency: on LIBERO, a few-shot fine-tuned model augmented with VLA-Corrector reaches 97.8% average success, surpassing the fully fine-tuned baseline (96.9%).

## Our contributions are summarized as follows:

• We systematically quantify the performance–eficiency trade-of induced by fixed action horizon in actionchunked VLA policies, showing that the open-loop blind spot consistently afects robustness across diferent VLA backbones.

• We propose VLA-Corrector, a lightweight inference-time framework that augments a frozen VLA backbone with latent-space monitoring, event-triggered truncation, and guidance toward recovery-oriented inference.

• We validate VLA-Corrector across simulation and real-world tasks, showing that it achieves stronger robustness, higher success-per-call eficiency, and adaptive correction behavior.

## 2 Preliminaries

VLA Policies with Action Chunks. In embodied continuous control, the current visual observation $o _ { t }$ is first encoded into a latent representation $Z _ { t } ^ { \mathrm { r e a l } } = \mathcal { E } ( o _ { t } )$ by a visual encoder $\mathcal { E } .$ To balance policy-call frequency and action smoothness, modern generative VLA policies parameterized by θ commonly predict an action chunk in a single inference call,

$$
A _ {t} = [ a _ {t}, a _ {t + 1}, \ldots , a _ {t + C - 1} ] \sim \pi_ {\theta} (\cdot | Z _ {t} ^ {\mathrm{real}}, l),\tag{1}
$$

where $a _ { t }$ denotes a single control action and l is the language instruction. During deployment, only the first H actions, with $H \leq C .$ are executed sequentially, and we refer to this execution window as the action horizon. The corresponding execution queue is $Q _ { t } = [ a _ { t } , a _ { t + 1 } , \dotsc , a _ { t + H - 1 } ]$ , during which the controller does not query the VLA policy again.

## 3 VLA-Corrector: A Lightweight Corrective Inference Framework

VLA-Corrector addresses the open-loop blind spot of action-chunked VLA policies. During the action horizon, fresh observations are available, but the policy does not use them until the horizon ends. We therefore decouple action generation from execution monitoring: the VLA backbone generates action chunks, while VLA-Corrector monitors whether execution remains on track and intervenes only when drift emerges. We organize the method as follows: (1) Corrector Training describes the external latent dynamics corrector; (2) LVM Detection introduces online visual deviation detection; (3) Event-Triggered Truncation explains how persistent deviations induce adaptive horizons; and (4) OGG Correction presents corrective replanning after truncation.

## 3.1 Training the External Latent Dynamics Corrector

VLA-Corrector is trained after a VLA policy has already been obtained. We first fine-tune the VLA backbone on the benchmark training set, then freeze the VLA and use its visual encoder $\mathcal { E }$ to extract visual latents from demonstration trajectories. Given a transition $\left( o _ { t } , a _ { t } , o _ { t + k } \right)$ , we compute

$$
Z _ {t} ^ {\mathrm{real}} = \mathcal {E} (o _ {t}), \qquad Z _ {t + k} ^ {\mathrm{real}} = \mathcal {E} (o _ {t + k}), \qquad \Delta Z _ {t + k} ^ {*} = Z _ {t + k} ^ {\mathrm{real}} - Z _ {t} ^ {\mathrm{real}},\tag{2}
$$

where $\Delta Z _ { t + k } ^ { * }$ is the target short-horizon visual latent evolution induced by the executed action. We then train a lightweight external dynamics corrector $M _ { \phi }$ to predict this residual evolution from the current latent state and action:

$$
\Delta \hat {Z} _ {t + k} = M _ {\phi} (Z _ {t} ^ {\mathrm{real}}, a _ {t}).\tag{3}
$$

Rather than predicting the absolute future latent state, $M _ { \phi }$ instead predicts a residual visual latent evolution, which actively suppresses static scene content and encourages the model to focus on task-relevant dynamics. The training objective combines magnitude matching and directional consistency:

$$
\mathcal {L} _ {\mathrm{corr}} = \left\| \Delta \hat {Z} _ {t + k} - \Delta Z _ {t + k} ^ {*} \right\| _ {2} ^ {2} + \beta \left[ 1 - \mathrm{CosSim} \left(\Delta \hat {Z} _ {t + k}, \Delta Z _ {t + k} ^ {*}\right) \right],\tag{4}
$$

where $\beta$ balances residual accuracy and directional alignment. The trainable component is therefore not the VLA policy itself, but a lightweight latent dynamics module trained separately on top of frozen VLA features. This modular design allows the corrector to be retrained or replaced for each benchmark without re-optimizing the expensive VLA backbone. The corrector is trained on benchmark demonstration trajectories because they are easy to obtain and broadly reflect successful task progress. Its goal is not to model all possible futures as a full world model would $\mathrm { d o } ,$ but to learn whether local latent dynamics remain consistent with on-track execution. Demonstrations are suitable for this purpose: although they may contain small teleoperation jitter or local imperfections, they still capture behavior that keeps the task progressing correctly. Since this is a local and low-dimensional prediction task, a lightweight MLP is suficient in practice; in our experiments, a 40M corrector already provides an efective monitoring and corrective signal, which can be trained at a very low cost.

![](images/b720c383b0ad5a54c1bcb5f6df0832b8048c3342732d6cea1a0700367941612f.jpg)  
Figure 3 Overview of VLA-Corrector. Starting from a standard chunked VLA pipeline (A), we add a Latent-space Vision Monitor (LVM) that detects persistent execution drift and triggers an interrupt event (B). The event truncates stale actions and switches the next replan from normal flow matching to OGG-guided flow matching (C). OGG uses the expected and observed latent evolution to guide the replan back toward a recoverable trajectory (D).

## 3.2 Latent Visual Dynamics for Online Anomaly Detection

After training $M _ { \phi } ,$ , we use it online to monitor whether the current action chunk remains reliable. During execution, Latent-space Vision Monitor (LVM) compares the expected latent visual evolution predicted by $M _ { \phi }$ with the actual evolution observed from fresh camera input, detecting deviations before they accumulate into failure.

Expected residual dynamics. At control step t, LVM uses the executed action $a _ { t }$ and current latent state $Z _ { t } ^ { \mathrm { r e a l } }$ to predict the expected short-horizon latent residual, $\Delta { Z } _ { t + k } ^ { \mathrm { e x p } } = M _ { \phi } ( { Z } _ { t } ^ { \mathrm { r e a l } } , a _ { t } )$ , where k is the prediction interval.

Online inconsistency score. During rollout, the latest observation remains available even though the policy is not re-queried. We encode the future observation and compute the actual residual $\Delta Z _ { t + k } ^ { \mathrm { r e a l } } = Z _ { t + k } ^ { \mathrm { r e a l } } - \bar { Z _ { t } ^ { \mathrm { r e a l } } }$ LVM then measures the mismatch between expected and actual latent evolution:

$$
E _ {t} = 1 - \operatorname{CosSim} \left(\Delta Z _ {t + k} ^ {\exp}, \Delta Z _ {t + k} ^ {\text {real}}\right), \qquad \operatorname{CosSim} (\mathbf {u}, \mathbf {v}) = \mathbf {u} ^ {\top} \mathbf {v} / (\| \mathbf {u} \| \| \mathbf {v} \|).\tag{5}
$$

A larger $E _ { t }$ indicates stronger visual dynamics mismatch, providing a continuous signal for later event-triggered truncation.

## 3.3 Event-Triggered Truncation under Robust Online Monitoring

Given the continuous inconsistency score $E _ { t }$ , we need to decide when a deviation is persistent enough to justify intervention. Directly thresholding $E _ { t }$ can be unstable, since transient visual outliers may cause false triggers. We therefore use a robust event-triggered rule based on dynamic thresholds and persistence checking. Specifically, we maintain a sliding window of recent scores, $\mathbf { E } _ { W } = \left\{ E _ { t - w + 1 } , \ldots , E _ { t } \right\}$ , and compute its median $M _ { e }$ and median absolute deviation,

$$
\mathrm{MAD} = \operatorname{median} \left(\left| E _ {i} - M _ {e} \right|\right), \quad E _ {i} \in \mathbf {E} _ {W}.\tag{6}
$$

We then define two adaptive thresholds,

$$
T _ {\mathrm{on}} = M _ {e} + \lambda_ {\mathrm{on}} \mathrm{MAD}, \qquad T _ {\mathrm{off}} = M _ {e} + \lambda_ {\mathrm{off}} \mathrm{MAD}, \qquad \lambda_ {\mathrm{on}} > \lambda_ {\mathrm{off}},\tag{7}
$$

where $T _ { \mathrm { o n } }$ confirms persistent deviation and $T _ { \mathrm { o f f } }$ provides hysteresis for recovery. An interrupt event is triggered only when $E _ { t } > T _ { \mathrm { o n } }$ holds for $p$ consecutive steps; isolated spikes are ignored.

Once an interrupt is triggered, VLA-Corrector discards the remaining actions in the current queue and queries the policy again under corrective mode. If the controller has executed h actions from the original queue, the realized horizon becomes $H _ { \mathrm { a d a p t i v e } } = h < H$ . Thus, the fixed action horizon is shortened only when sustained visual drift indicates that the current chunk has become stale. The full thresholding state machine is provided in Appendix B.1.

## 3.4 Online Gradient Guidance for Corrective Inference

Truncation stops stale actions, but recovery still depends on the next replan. After an interrupt event, VLA-Corrector applies Online Gradient Guidance (OGG) Park et al. (2025) only to the single policy call immediately following the interrupt event, guiding this recovery replan toward a corrective latent direction.

Predicted action effect. At denoising step τ of flow matching, let $A ^ { \tau }$ be the noisy action chunk. The VLA predicts a velocity field $v _ { \tau } = \pi _ { \theta } ( A ^ { \tau } , Z _ { t } ^ { \mathrm { r e a l } } , \tau )$ , from which we estimate the clean chunk $\hat { A } _ { 0 } = A ^ { \tau } - \tau v _ { \tau }$ and take its first action $\hat { a } _ { t } = \hat { A } _ { 0 } [ 0 ]$ . The corrector then predicts the latent efect of this candidate action:

$$
\Delta \hat {Z} _ {\mathrm{act}} = M _ {\phi} (Z _ {t} ^ {\mathrm{real}}, \hat {a} _ {t}).\tag{8}
$$

Corrective target. Let $t - k$ be the last stable step before an interrupt event. We define the expected residual $\Delta { \cal Z } _ { \mathrm { e x p } } = \Delta { \cal Z } _ { t } ^ { \mathrm { e x p } } = M _ { \phi } ( { \cal Z } _ { t - k } ^ { \mathrm { r e a l } } , a _ { t - k } )$ , which is predicted by $M _ { \phi }$ from step $t - k$ . The accumulated deviation is $\Delta Z _ { \mathrm { d e v } } = \Delta Z _ { t } ^ { \ast } = Z _ { t } ^ { \mathrm { r e a l } } - Z _ { t - k } ^ { \mathrm { r e a l } }$ . The corrective latent direction is

$$
\Delta Z _ {\mathrm{corr}} = \Delta Z _ {\mathrm{exp}} - \Delta Z _ {\mathrm{dev}}.\tag{9}
$$

This direction preserves the intended local dynamics while compensating for the drift accumulated during open-loop execution.

Guided velocity update. OGG aligns the predicted action efect with $\Delta Z _ { \mathrm { c o r r } }$ through

$$
\mathcal {L} _ {\mathrm{OGG}} = 1 - \mathrm{CosSim} (\Delta \hat {Z} _ {\mathrm{act}}, \Delta Z _ {\mathrm{corr}}),\tag{10}
$$

and injects the resulting gradient into the flow-matching velocity:

$$
v _ {\tau} ^ {\mathrm{guide}} = v _ {\tau} - \eta \nabla_ {v _ {\tau}} \mathcal {L} _ {\mathrm{OGG}}, A ^ {\tau - \Delta \tau} = A ^ {\tau} - \Delta \tau v _ {\tau} ^ {\mathrm{guide}}.\tag{11}
$$

Here, η controls the guidance strength. Since OGG modifies the velocity field rather than directly perturbing action coordinates, it remains compatible with the original flow-matching process and yields smoother corrective replanning.

## 4 Experiments

We evaluate VLA-Corrector on MetaWorld Yu et al. (2020), LIBERO Liu et al. (2023), RoboTwin v2.0 Chen et al. (2025b), and a real AgileX PiPER robot, using $\pi _ { 0 . 5 }$ Intelligence et al. (2025) as the main backbone and SmolVLA Shukor et al. (2025)/X-VLA Zheng et al. (2025) for cross-architecture evaluation. The experiments cover four aspects: (1) Performance and policy-call efficiency, evaluating success rate and success-per-call across benchmarks and horizons; (2) Mechanism analysis, examining LVM detection, truncation timing, and OGG recovery; (3) Real-world transfer, testing the method on physical robot manipulation; and (4) Ablations, analyzing component necessity and sensitivity. Detailed protocols, metrics, and compute resources are provided in Appendix C.1.

## 4.1 Main Results

Cross-Architecture Evaluation on MetaWorld. VLA-Corrector consistently improves all three VLA backbones on MetaWorld, with larger gains on harder tasks. As shown in Table 1, the average success rate improves by 15.65 points for $\pi _ { 0 . 5 } .$ , 4.75 points for SmolVLA, and 4.05 points for X-VLA. The strongest gain appears on the Very Hard split of $\pi _ { 0 . 5 }$ , increasing success from 41.0% to 65.0%.

Table 1 Cross-architecture generalization on MetaWorld. Success rate (%, ↑) across dificulty splits. Green arrows show absolute improvements over the corresponding baseline.

<table><tr><td>Backbone</td><td>Method</td><td>Easy↑</td><td>Medium↑</td><td>Hard↑</td><td>Very Hard↑</td><td>Avg.↑</td></tr><tr><td rowspan="2"> $\pi_{0.5}$ </td><td>Baseline</td><td>70.5</td><td>45.0</td><td>38.3</td><td>41.0</td><td>48.70</td></tr><tr><td>+ VLA-Corrector</td><td>83.2 ↑ 12.7</td><td>61.7 ↑ 16.7</td><td>47.5 ↑ 9.2</td><td>65.0 ↑ 24.0</td><td>64.35 ↑ 15.65</td></tr><tr><td rowspan="2">SmolVLA</td><td>Baseline</td><td>81.3</td><td>53.6</td><td>51.7</td><td>61.0</td><td>61.90</td></tr><tr><td>+ VLA-Corrector</td><td>83.4 ↑ 2.1</td><td>56.0 ↑ 2.4</td><td>64.2 ↑ 12.5</td><td>63.0 ↑ 2.0</td><td>66.65 ↑ 4.75</td></tr><tr><td rowspan="2">X-VLA</td><td>Baseline</td><td>72.5</td><td>46.4</td><td>48.3</td><td>55.0</td><td>55.55</td></tr><tr><td>+ VLA-Corrector</td><td>74.4 ↑ 1.9</td><td>50.0 ↑ 3.6</td><td>50.0 ↑ 1.7</td><td>64.0 ↑ 9.0</td><td>59.60 ↑ 4.05</td></tr></table>

Table 2 RoboTwin v2.0 evaluation. Success rate (%, ↑), average policy calls per episode (↓), and per-step wallclock latency (↓). The benchmark covers all 50 tasks in clean and randomized configurations, with 50 episodes per task/configuration (5,000 episodes per method); the same frozen $\pi _ { 0 . 5 }$ backbone is used throughout.

<table><tr><td>Method</td><td>Clean↑</td><td>Random↑</td><td>Overall (95% CI)↑</td><td>Calls/ep.↓</td><td>ms/step↓</td></tr><tr><td>Baseline H = 50</td><td>68.2</td><td>62.6</td><td>65.4 [64.1,66.7]</td><td>5.74</td><td>8.10</td></tr><tr><td>Baseline H = 30 (latency-matched)</td><td>72.0</td><td>66.8</td><td>69.4 [68.1,70.7]</td><td>9.41</td><td>9.98</td></tr><tr><td>VLA-Corrector H = 50</td><td>76.0</td><td>70.2</td><td>73.1 [71.9,74.3]</td><td>6.88</td><td>10.02</td></tr></table>

Table 3 Sample efficiency on LIBERO. Success rate (%, ↑). Green arrows compare VLA-Corrector against the few-shot fine-tuned baseline.

<table><tr><td>Model</td><td>Object↑</td><td>Spatial↑</td><td>Goal↑</td><td>Long↑</td><td>Avg.↑</td></tr><tr><td> $\pi_{0.5}$  (Full Fine-tuned)</td><td>99.4</td><td>98.2</td><td>97.8</td><td>92.4</td><td>96.95</td></tr><tr><td> $\pi_{0.5}$  (Few-shot Fine-tuned)</td><td>97.8</td><td>95.4</td><td>96.2</td><td>86.6</td><td>94.00</td></tr><tr><td> $\pi_{0.5}$  (Few-shot) + VLA-Corrector</td><td>99.8 ↑ 2.0</td><td>100.0 ↑ 4.6</td><td>98.0 ↑ 1.8</td><td>93.4 ↑ 6.8</td><td>97.80 ↑ 3.80</td></tr></table>

RoboTwin Evaluation. On the broader RoboTwin v2.0 benchmark, VLA-Corrector improves overall success from 65.4% to 73.1% over the original H = 50 baseline and from 69.4% to 73.1% over the latency-matched H = 30 baseline. The gain over the matched baseline is 3.7 points while per-step latency remains nearly identical (10.02 vs. 9.98 ms); VLA-Corrector also uses 27% fewer policy calls than the matched baseline (6.88 vs. 9.41 calls/episode). The improvement persists in both clean and randomized configurations. A more comprehensive and detailed analysis can be found in Appendix D.6.

Sample Efficiency on LIBERO. We further evaluate whether VLA-Corrector can compensate for limited task-specific data. Table 3 compares a fully fine-tuned $\pi _ { 0 . 5 }$ model with a few-shot fine-tuned counterpart on LIBERO. Starting from the publicly released LeRobot Cadene et al. (2026) few-shot checkpoint pi05\_- libero\_base, VLA-Corrector improves the average success rate from 94.00% to 97.80%, even surpassing the fully fine-tuned baseline (96.95%).

These results suggest that few-shot fine-tuning can already learn much of the normal task trajectories, but may lack coverage of drifted states and their recovery behaviors. Instead of improving recovery by exposing the backbone to more rare failure cases during training, VLA-Corrector makes recovery easier at inference time by interrupting error accumulation early and guiding corrective replanning, thereby alleviating the dependence on additional post-training data for robust recovery.

Performance–Efficiency Trade-off. VLA-Corrector improves success-per-call eficiency across action horizons rather than simply querying the policy more often. As shown in Figure 4, the largest success-per-call gains reach 29.9% for $\pi _ { 0 . 5 } ,$ 45.3% for SmolVLA, and 39.1% for X-VLA. For example, on SmolVLA with horizon 10, success improves from 61.90% to 73.00% while policy calls decrease from 19.27 to 15.64. This suggests that VLA-Corrector makes each policy query more useful by interrupting stale chunks and guiding recovery. Full results are provided in Appendix D.3.

![](images/d9ddec1224f63bb621d1c628d881365d1db200631568851213acbfdd513ed12d.jpg)

![](images/8515b3f79d2e8f85bb1134526031609deab81b3e282ca65fda4f5b3154959cae.jpg)  
Figure 4 Performance–efficiency analysis on π . Left: performance–eficiency trade-of across action horizons. Right: success-per-call eficiency. VLA-Corrector improves success rate across action horizons and yields consistent eficiency gains.

## 4.2 Mechanism Analysis

In this section, we will analyze the mechanisms behind VLA-Corrector from three perspectives: 1) LVM detection, whether LVM provides a useful and well-timed drift signal; and 2) OGG correction, whether OGG improves recovery after an interrupt event; 3) controlled recovery, whether the full framework improves recovery in a controlled failure case.

![](images/1654c9807eebba0771538ff9212487379377eb8662602d638f167882f4eb4230.jpg)

![](images/959992a76a9e2dca6d0383910c6c450a41c3d0c77fa11b8b78dcf264a87ef32c.jpg)

![](images/26f2a3e5d0fd67d3cbe0b289df9f988e7a2b50679714e4cfd1c6a66549e9494c.jpg)  
Figure 5 LVM detection analysis. Left: distribution of the inconsistency score $E _ { t }$ , where successful episodes concentrate at low values and failed episodes show a heavier high-score tail. Right: interrupt frequency, where failed episodes trigger more interrupt events than successful ones.

LVM Detection. LVM provides an informative drift signal. As shown in Figure 5, successful episodes concentrate at low $E _ { t }$ , while failed episodes show a heavier high-score tail and trigger more interrupt events. This indicates that $E _ { t }$ captures failure-prone visual dynamics mismatch.

![](images/0c2d3943aaf6b3e770999cfaa078873e03fb2aa75d84fd757072f638ac92b8e7.jpg)

<table><tr><td colspan="2">Truncation Distribution by Task Phase</td></tr><tr><td>Phase</td><td>Rate</td></tr><tr><td>Critical</td><td>83.7%</td></tr><tr><td>Non-critical</td><td>16.3%</td></tr></table>

5.1× more truncations occur in critical phases than in non-critical phases.  
Figure 6 Task-phase analysis of LVM-triggered truncation. We manually divide MetaWorld trajectories into critical and non-critical phases. Left: representative trajectories show that truncation is triggered much more frequently during critical phases. Right: 83.7% of truncations occur in critical phases, while only 16.3% occur in non-critical phases.

LVM also intervenes at meaningful moments. We manually divide MetaWorld trajectories into critical phases, such as precise grasping and alignment, and non-critical phases, such as tolerant transport after a stable grasp. Figure 6 shows that 83.7% of truncations occur in critical phases. This result verifies the core adaptive-horizon intuition of VLA-Corrector: it does not simply shorten the horizon everywhere, but instead preserves long-horizon eficiency in tolerant phases and restores short-horizon precision when the task enters error-sensitive phases.

![](images/d38618cc68a151821e0f7d07c88c95beceb94ac243210db9fde31ba1ef5c8ff9.jpg)  
Figure 7 Controlled recovery case. Given the same initial state and detected grasping error, the monitored baseline continues the original chunk and fails (top), while VLA-Corrector truncates stale actions, replans with OGG, and completes the task (bottom).

OGG Correction. We isolate OGG by comparing standard re-inference and OGG-guided re-inference

after the same interrupt-triggered truncation. Recovery is counted when $E _ { t } < T _ { \mathrm { o f f } }$ within the next 10 steps. As shown in Fig. 8, OGG improves recovery across all dificulty levels, with an average gain of 0.23. This indicates that, after truncation stops stale actions, OGG-guided re-inference further improves the quality of the recovery replan compared to standard re-inference.

Controlled Recovery Case. Figure 7 shows a representative LIBERO episode where both methods start from the same initial state and encounter the same grasping error when approaching the cup handle. In the top row, the baseline rollout is only monitored by LVM for error alignment without truncating the remaining actions; after the deviation is detected, it continues executing the original action chunk, leading to an unstable grasp, cup drop, and task failure. In

![](images/9f9912481259925c687a3b99983832e89455c52c9f924680c15b7fbd6f2060d9.jpg)  
Figure 8 Post-interrupt recovery. OGG-guided inference consistently outperforms standard inference.

the bottom row, VLA-Corrector detects the same deviation but actively truncates the stale actions and triggers corrective replanning, allowing the robot to recover a stable grasp and place the cup at the target location. This controlled comparison highlights the overall efectiveness of VLA-Corrector under the same initial condition and the same detected execution error.

## 4.3 Real-World Evaluation

Setup. We evaluate VLA-Corrector on an AgileX PiPER 6-DoF arm using $\pi _ { 0 . 5 }$ as the backbone. The benchmark includes three task groups, each with three tasks and 20 trials: 1) Pick-and-place tests standard manipulation, 2) Alignment tests precision-sensitive placement or insertion, and 3) Disturbance recovery tests recovery when objects or targets are manually shifted during execution. Full task definitions, protocols, failure analysis, and representative demonstrations are provided in Appendix E.

Results. Table 4 shows that VLA-Corrector improves all three task groups, increasing average success from 55.6% to 73.3%. The gain is modest on pick-and-place (+8.3), larger on alignment (+16.6), and largest on disturbance recovery (+28.3). This trend matches the intended role of VLA-Corrector: it preserves standard execution performance, improves precision-sensitive manipulation, and is most beneficial when online disturbances make the remaining action chunk outdated.

Table 4 Real-world evaluation on AgileX PiPER. Success rate $( \% , \uparrow )$ over three task groups. Each group contains three tasks with 20 trials per task. We report 95% binomial confidence intervals.

<table><tr><td>Method</td><td>Pick-place↑</td><td>Alignment↑</td><td>Disturbance↑</td><td>Avg.↑</td></tr><tr><td> $\pi_{0.5}$  Baseline</td><td>70.0 ± 11.6</td><td>56.7 ± 12.5</td><td>40.0 ± 12.4</td><td>55.6 ± 7.3</td></tr><tr><td>+ VLA-Corrector</td><td>78.3 ± 10.4 ↑ 8.3</td><td>73.3 ± 11.2 ↑ 16.6</td><td>68.3 ± 11.8 ↑ 28.3</td><td>73.3 ± 6.5 ↑ 17.7</td></tr></table>

## 4.4 Ablation Studies

Table 5 Component ablation on MetaWorld. Success rate $( \% , \uparrow )$ . Green arrows indicate average improvement over the baseline.

<table><tr><td>Variant</td><td>Easy↑</td><td>Medium↑</td><td>Hard↑</td><td>Very Hard↑</td><td>Avg.↑</td></tr><tr><td>Baseline (Open-loop)</td><td>70.5</td><td>45.0</td><td>38.3</td><td>41.0</td><td>48.70</td></tr><tr><td>+ Truncation Only</td><td>81.8</td><td>53.6</td><td>50.0</td><td>56.0</td><td>60.35 ↑ 11.65</td></tr><tr><td>+ Truncation + OGG</td><td>83.2</td><td>61.7</td><td>47.5</td><td>65.0</td><td>64.35 ↑ 15.65</td></tr></table>

We ablate VLA-Corrector on $\pi _ { 0 . 5 }$ in MetaWorld, focusing on (1) components: necessity of truncation and OGG; (2) detector design: decoupling monitoring from backbone fine-tuning; and (3) sensitivity, with detailed results in Appendix D.1.

Component effects. Table 5 shows that truncation alone raises average success from 48.70% to 60.35%, confirming the importance of stopping stale actions. Adding OGG further improves the average to 64.35%, showing that post-truncation recovery also benefits from guided re-inference.

Detector design and sensitivity. We compare our decoupled LVM with an internal detection head trained during $\pi _ { 0 . 5 }$ fine-tuning using the same auxiliary latent-residual prediction target. The internal head takes the final-layer hidden state of the last token, combines it with the executed action, and predicts the same task as the external LVM. As shown in Appendix D.1, decoupled LVM improves average success from 49.55% to 64.35%, suggesting that separating monitoring from action generation preserves the backbone representation and yields a cleaner drift signal. The same appendix also shows that moderate guidance strength and a 40M LVM are suficient; overly strong guidance or larger monitors provide little additional benefit.

## 5 Conclusion

This paper studies the open-loop blind spot in action-chunked VLA policies, where fixed horizons improve policy-call eficiency but allow stale actions to accumulate errors. VLA-Corrector addresses this issue with a lightweight detect-and-correct layer that monitors latent visual dynamics, truncates stale actions when drift persists, and guides the next inference toward recovery. Our results show that small inference-time modules can provide targeted robustness gains without retraining the VLA backbone. Rather than replacing action chunking, VLA-Corrector makes it adaptive: long-horizon execution is preserved when reliable, while corrective replanning is invoked when the current chunk should no longer be trusted.

## References

Johan Bjorck, Fernando Castañeda, Nikita Cherniadev, Xingye Da, Runyu Ding, Linxi Fan, Yu Fang, Dieter Fox, Fengyuan Hu, Spencer Huang, and 1 others. 2025. Gr00t n1: An open foundation model for generalist humanoid robots. arXiv preprint arXiv:2503.14734.

Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, and 1 others. 2024. π<sub>0</sub>: A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164.

Remi Cadene, Simon Aliberts, Francesco Capuano, Michel Aractingi, Adil Zouitine, Pepijn Kooijmans, Jade Choghari, Martino Russi, Caroline Pascal, Steven Palma, and 1 others. 2026. Lerobot: An open-source library for end-to-end robot learning. arXiv preprint arXiv:2602.22818.

Jun Cen, Siteng Huang, Yuqian Yuan, Kehan Li, Hangjie Yuan, Chaohui Yu, Yuming Jiang, Jiayan Guo, Xin Li, Hao Luo, and 1 others. 2025. Rynnvla-002: A unified vision-language-action and world model. arXiv preprint arXiv:2511.17502.

Hao Chen, Jiaming Liu, Chenyang Gu, Zhuoyang Liu, Renrui Zhang, Xiaoqi Li, Xiao He, Yandong Guo, Chi-Wing Fu, Shanghang Zhang, and 1 others. 2025a. Fast-in-slow: A dual-system foundation model unifying fast manipulation within slow reasoning. arXiv preprint arXiv:2506.01953.

Tianxing Chen, Zanxin Chen, Baijun Chen, Zijian Cai, Yibin Liu, Zixuan Li, Qiwei Liang, Xianliang Lin, Yiheng Ge, Zhenyu Gu, and 1 others. 2025b. Robotwin 2.0: A scalable data generator and benchmark with strong domain randomization for robust bimanual robotic manipulation. arXiv preprint arXiv:2506.18088.

Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song. 2025. Difusion policy: Visuomotor policy learning via action difusion. The International Journal of Robotics Research, 44(10-11):1684–1704.

Danny Driess, Jost Tobias Springenberg, Brian Ichter, Lili Yu, Adrian Li-Bell, Karl Pertsch, Allen Z Ren, Home Walke, Quan Vuong, Lucy Xiaoyang Shi, and 1 others. 2025. Knowledge insulating vision-language-action models: Train fast, run fast, generalize better. arXiv preprint arXiv:2505.23705.

Yichao Fu, Xuewei Wang, Yuandong Tian, and Jiawei Zhao. 2025. Deep think with confidence. arXiv preprint arXiv:2508.15260.

George Jiayuan Gao, Tianyu Li, and Nadia Figueroa. 2025. Out-of-distribution recovery with object-centric keypoint inverse policy for visuomotor imitation learning. In 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 9816–9828. IEEE.

Dibya Ghosh, Homer Walke, Karl Pertsch, Kevin Black, Oier Mees, Sudeep Dasari, Joey Hejna, Tobias Kreiman, Charles Xu, Jianlan Luo, and 1 others. 2024. Octo: An open-source generalist robot policy. In Robotics: Science and Systems (RSS).

Ruiyu Gou. 2024. Learning temporal action chunking for motor control. Ph.D. thesis, University of British Columbia.

Zhi Hou, Tianyi Zhang, Yuwen Xiong, Haonan Duan, Hengjun Pu, Ronglei Tong, Chengyang Zhao, Xizhou Zhu, Yu Qiao, Jifeng Dai, and 1 others. 2025. Dita: Scaling difusion transformer for generalist vision-language-action policy. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7686–7697.

Zheyuan Hu, Robyn Wu, Naveen Enock, Jasmine Li, Riya Kadakia, Zackory Erickson, and Aviral Kumar. 2025. Rac: Robot learning for long-horizon tasks by scaling recovery and correction. arXiv preprint arXiv:2509.07953.

Nils Ingelhag, Jesper Munkeby, Michael C Welle, Marco Moletta, and Danica Kragic. 2025. Real-time operator takeover for visuomotor difusion policy training. arXiv preprint arXiv:2502.02308.

Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, and 1 others. 2025. \pi\_ {0.5} : a vision-language-action model with open-world generalization. arXiv preprint arXiv:2504.16054.

Arnav Kumar Jain, Vibhakar Mohta, Subin Kim, Atiksh Bhardwaj, Juntao Ren, Yunhai Feng, Sanjiban Choudhury, and Gokul Swamy. 2025. A smooth sea never made a skilled sailor: Robust imitation via learning to search. arXiv preprint arXiv:2506.05294.

Sunshine Jiang, Xiaolin Fang, Nicholas Roy, Tomás Lozano-Pérez, Leslie Pack Kaelbling, and Siddharth Ancha. 2025a. Streaming flow policy: Simplifying difusion/flow-matching policies by treating action trajectories as flow trajectories. arXiv preprint arXiv:2505.21851.

Yuming Jiang, Siteng Huang, Shengke Xue, Yaxi Zhao, Jun Cen, Sicong Leng, Kehan Li, Jiayan Guo, Kexiang Wang, Mingxiu Chen, and 1 others. 2025b. Rynnvla-001: Using human demonstrations to improve robot manipulation. arXiv preprint arXiv:2509.15212.

Dong Jing, Gang Wang, Jiaqi Liu, Weiliang Tang, Zelong Sun, Yunchao Yao, Zhenyu Wei, Yunhui Liu, Zhiwu Lu, and Mingyu Ding. 2025. Mixture of horizons in action chunking. arXiv preprint arXiv:2511.19433.

Sarosh Khan and Ellie Tanimura. 2025. Test-time stochasticity estimation for adaptive action chunk selection.

Moo Jin Kim, Chelsea Finn, and Percy Liang. 2025. Fine-tuning vision-language-action models: Optimizing speed and success. arXiv preprint arXiv:2502.19645.

Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, and 1 others. 2024. Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246.

Wei Li, Renshan Zhang, Rui Shao, Jie He, and Liqiang Nie. 2025. Cogvla: Cognition-aligned vision-language-action model via instruction-driven routing & sparsification. arXiv preprint arXiv:2508.21046.

Xinghang Li, Minghuan Liu, Hanbo Zhang, Cunjun Yu, Jie Xu, Hongtao Wu, Chilam Cheang, Ya Jing, Weinan Zhang, Huaping Liu, and 1 others. 2023. Vision-language foundation models as efective robot imitators. arXiv preprint arXiv:2311.01378.

Yuanchang Liang, Xiaobo Wang, Kai Wang, Shuo Wang, Xiaojiang Peng, Haoyu Chen, David Kim Huat Chua, and Prahlad Vadakkepat. 2026. Adaptive action chunking at inference-time for vision-language-action models. arXiv preprint arXiv:2604.04161.

Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. 2022. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747.

Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. 2023. Libero: Benchmarking knowledge transfer for lifelong robot learning. Advances in Neural Information Processing Systems, 36:44776–44791.

Isabella Liu, An-Chieh Cheng, Rui Yan, Geng Chen, Ri-Zhao Qiu, Xueyan Zou, Sha Yi, Hongxu Yin, Xiaolong Wang, and Sifei Liu. 2026. Long-horizon manipulation via trace-conditioned vla planning. arXiv preprint arXiv:2604.21924.

Yuejiang Liu, Jubayer Ibn Hamid, Annie Xie, Yoonho Lee, Maximilian Du, and Chelsea Finn. 2024. Bidirectional decoding: Improving action chunking via guided test-time sampling. arXiv preprint arXiv:2408.17355.

Yueen Ma, Zixing Song, Yuzheng Zhuang, Jianye Hao, and Irwin King. 2024. A survey on vision-language-action models for embodied ai. arXiv preprint arXiv:2405.14093.

Cyrus Neary, Omar G Younis, Artur Kuramshin, Ozgur Aslan, and Glen Berseth. 2025. Improving pre-trained vision-language-action policies with model-based search. arXiv preprint arXiv:2508.12211.

Abby O’Neill, Abdul Rehman, Abhiram Maddukuri, Abhishek Gupta, Abhishek Padalkar, Abraham Lee, Acorn Pooley, Agrim Gupta, Ajay Mandlekar, Ajinkya Jain, and 1 others. 2024. Open x-embodiment: Robotic learning datasets and rt-x models: Open x-embodiment collaboration 0. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 6892–6903. IEEE.

Minho Park, Kinam Kim, Junha Hyung, Hyojin Jang, Hoiyeong Jin, Jooyeol Yun, Hojoon Lee, and Jaegul Choo. 2025. Acg: Action coherence guidance for flow-based vla models. arXiv preprint arXiv:2510.22201.

Karl Pertsch, Kyle Stachowicz, Brian Ichter, Danny Driess, Suraj Nair, Quan Vuong, Oier Mees, Chelsea Finn, and Sergey Levine. 2025. Fast: Eficient action tokenization for vision-language-action models. arXiv preprint arXiv:2501.09747.

Kohei Sendai, Maxime Alvarez, Tatsuya Matsushima, Yutaka Matsuo, and Yusuke Iwasawa. 2025. Leave no observation behind: Real-time correction for vla action chunks. arXiv preprint arXiv:2509.23224.

Archit Sharma, Ahmed M Ahmed, Rehaan Ahmad, and Chelsea Finn. 2023. Self-improving robots: End-to-end autonomous visuomotor reinforcement learning. arXiv preprint arXiv:2303.01488.

Mustafa Shukor, Dana Aubakirova, Francesco Capuano, Pepijn Kooijmans, Steven Palma, Adil Zouitine, Michel Aractingi, Caroline Pascal, Martino Russi, Andres Marafioti, and 1 others. 2025. Smolvla: A vision-language-action model for afordable and eficient robotics. arXiv preprint arXiv:2506.01844.

Junhyuk So, Chiwoong Lee, Shinyoung Lee, Jungseul Ok, and Eunhyeok Park. 2025. Improving generative behavio cloning via self-guidance and adaptive chunking. arXiv preprint arXiv:2510.12392.

Haoming Song, Delin Qu, Yuanqi Yao, Qizhi Chen, Qi Lv, Yiwen Tang, Modi Shi, Guanghui Ren, Maoqing Yao, Bin Zhao, and 1 others. 2025a. Hume: Introducing system-2 thinking in visual-language-action model. arXiv preprint arXiv:2505.21432.

Wenxuan Song, Jiayi Chen, Pengxiang Ding, Han Zhao, Wei Zhao, Zhide Zhong, Zongyuan Ge, Jun Ma, and Haoang Li. 2025b. Accelerating vision-language-action model integrated with action chunking via parallel decoding. arXiv preprint arXiv:2503.02310.

Zhanyi Sun and Shuran Song. 2025. Latent policy barrier: Learning robust visuomotor policies by staying in-distribution. arXiv preprint arXiv:2508.05941.

Libo Wang. 2025. Sigma: The key for vision-language-action models toward telepathic alignment. arXiv preprint arXiv:2512.00783.

Yihao Wang, Pengxiang Ding, Lingxiao Li, Can Cui, Zirui Ge, Xinyang Tong, Wenxuan Song, Han Zhao, Wei Zhao, Pengxu Hou, and 1 others. 2026a. Vla-adapter: An efective paradigm for tiny-scale vision-language-action model. Proceedings of the AAAI conference on artificial intelligence, 40(22):18638–18646.

Zhendong Wang, Zhaoshuo Li, Ajay Mandlekar, Zhenjia Xu, Jiaojiao Fan, Yashraj Narang, Linxi Fan, Yuke Zhu, Yogesh Balaji, Mingyuan Zhou, and 1 others. 2024. One-step difusion policy: Fast visuomotor policies via difusion distillation. arXiv preprint arXiv:2410.21257.

Zihua Wang, Zhitao Lin, Ruibo Li, Yu Zhang, Xu Yang, Siya Mi, and Xiu-Shen Wei. 2026b. Open-loop planning, closed-loop verification: Speculative verification for vla. arXiv preprint arXiv:2604.02965.

Junjie Wen, Yichen Zhu, Minjie Zhu, Zhibin Tang, Jinming Li, Zhongyi Zhou, Xiaoyu Liu, Chaomin Shen, Yaxin Peng, and Feifei Feng. 2025. Difusionvla: Scaling robot foundation models via unified difusion and autoregression. In Forty-second International Conference on Machine Learning.

Pengyuan Wu, Pingrui Zhang, Zhigang Wang, Dong Wang, Bin Zhao, and Xuelong Li. 2026a. Closed-loop action chunks with dynamic corrections for training-free difusion policy. arXiv preprint arXiv:2603.01953.

Zhichao Wu, Junyin Ye, Zhilong Zhang, Yihao Sun, Haoxin Lin, Jiaheng Luo, Haoxiang Ren, Lei Yuan, and Yang Yu. 2026b. Speedup patch: Learning a plug-and-play policy to accelerate embodied manipulation. arXiv preprint arXiv:2603.20658.

Changhua Xu, Jie Lu, Junyu Xuan, and En Yu. 2026. Vgas: Value-guided action-chunk selection for few-shot vision-language-action adaptation. arXiv preprint arXiv:2602.07399.

Shaoqing Xu, Fang Li, Zhixiang Duan, Yifan Yang, Tianshi Xie, and Zhi-Xin Yang. n.d. Vla-in-the-loop: Online policy correction with world models for robust robotic grasping. OpenReview manuscript. ICLR 2026 submission.

Tianhe Yu, Deirdre Quillen, Zhanpeng He, Ryan Julian, Karol Hausman, Chelsea Finn, and Sergey Levine. 2020. Meta-world: A benchmark and evaluation for multi-task and meta reinforcement learning. In Conference on robot learning, pages 1094–1100. PMLR.

Thomas TCK Zhang, Daniel Pfrommer, Chaoyi Pan, Nikolai Matni, and Max Simchowitz. 2026. Action chunking and data augmentation yield exponential improvements in behavior cloning for continuous spaces. In The Fourteenth International Conference on Learning Representations.

Chen Zhao, Zhuoran Wang, Haoyang Li, Shifeng Bao, Guanlin Li, Youhe Feng, Yang Li, Jie Tang, and Jing Zhang. 2026. Action draft and verify: A self-verifying framework for vision-language-action model. arXiv preprint arXiv:2603.18091.

Qingqing Zhao, Yao Lu, Moo Jin Kim, Zipeng Fu, Zhuoyang Zhang, Yecheng Wu, Zhaoshuo Li, Qianli Ma, Song Han, Chelsea Finn, and 1 others. 2025. Cot-vla: Visual chain-of-thought reasoning for vision-language-action models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 1702–1713.

Tony Z Zhao, Vikash Kumar, Sergey Levine, and Chelsea Finn. 2023a. Learning fine-grained bimanual manipulation with low-cost hardware. arXiv preprint arXiv:2304.13705.

Tony Z. Zhao, Vikash Kumar, Sergey Levine, and Chelsea Finn. 2023b. Learning fine-grained bimanual manipulation with low-cost hardware. In Robotics: Science and Systems (RSS).

Jinliang Zheng, Jianxiong Li, Zhihao Wang, Dongxiu Liu, Xirui Kang, Yuchun Feng, Yinan Zheng, Jiayin Zou, Yilun Chen, Jia Zeng, and 1 others. 2025. X-vla: Soft-prompted transformer as scalable cross-embodiment vision-language-action model. arXiv preprint arXiv:2510.10274.

Brianna Zitkovich, Tianhe Yu, Sichun Xu, Peng Xu, Ted Xiao, Fei Xia, Jialin Wu, Paul Wohlhart, Stefan Welker, Ayzaan Wahid, and 1 others. 2023. Rt-2: Vision-language-action models transfer web knowledge to robotic control. In Conference on Robot Learning, pages 2165–2183. PMLR.

## Appendix

## A Related Work

## A.1 Generative VLA Models

Embodied intelligence is shifting from task-specific policies toward unified Vision-Language-Action (VLA) foundation models. Representative works have shown that combining large-scale vision-language pretraining with robot control can substantially improve cross-task generalization, semantic understanding, and crossplatform transfer O’Neill et al. (2024); Black et al. (2024); Zitkovich et al. (2023); Kim et al. (2024); Ghosh et al. (2024); Li et al. (2023); Bjorck et al. (2025); Pertsch et al. (2025); Song et al. (2025b). On the action modeling side, modern VLA and general robot policies often adopt generative frameworks to model high-dimensional, continuous, and multi-modal action distributions: difusion models generate actions through iterative denoising Chi et al. (2025); Wen et al. (2025); Black et al. (2024); Chen et al. (2025a); Li et al. (2025); Zhao et al. (2025); Hou et al. (2025); Kim et al. (2025); Song et al. (2025a), while flow matching directly learns velocity fields along continuous trajectories. These approaches typically ofer stronger action expressivity, but they also incur higher policy-call costWang et al. (2024). Consequently, reducing control latency without sacrificing generation quality has become a central challenge in deploying VLA models.

## A.2 Action Chunk and Horizon Trade-off

To alleviate the tension between high-frequency control and expensive policy inference, action chunk has been widely adopted in continuous control. The core idea is to predict multiple future actions in a single inference call and then execute them sequentially with a downstream controller Chi et al. (2025); Zhao et al. (2023b); Black et al. (2024); Shukor et al. (2025); Bjorck et al. (2025); Wang (2025); Driess et al. (2025). This design efectively amortizes policy-call frequency and improves temporal smoothness, but it also reduces the use of fresh observations within a fixed action horizon. A longer horizon further lowers inference frequency and maintains action consistency, but weakens responsiveness to environmental changes; a shorter horizon more closely resembles strict closed-loop control and often provides better local recovery, yet requires much more frequent policy calls Jing et al. (2025); Zhang et al. (2026); Liu et al. (2024); Khan and Tanimura (2025); So et al. (2025); Fu et al. (2025); Zhao et al. (2026); Liu et al. (2026). Most existing approaches therefore seek a compromise under a static horizon, whereas our work focuses on an event-triggered dynamic horizon: the system retains the eficiency of long chunks during stable phases and truncates them on demand to recover closed-loop responsiveness during local failures.

## A.3 Failure Recovery in Visuomotor Policies

Long-horizon visuomotor control commonly sufers from compounding errors and covariate shift: once execution drifts away from the training distribution, the policy often cannot recover on its own. Existing approaches can be broadly divided into two categories. One line of work expands the training distribution through additional interaction, human intervention, or recovery data collection, and then updates the policy accordingly Sharma et al. (2023); Ingelhag et al. (2025); Hu et al. (2025); Neary et al. (2025). Another line explicitly introduces recovery policies, search mechanisms, or distributional constraints to pull the system back to a safe region once it departs from the expert manifold Gao et al. (2025); Sun and Song (2025); Jain et al. (2025); Liang et al. (2026); Gou (2024); Wu et al. (2026b); Xu et al. (n.d.). These works collectively show that explicit recovery mechanisms can substantially improve the robustness of visuomotor policies; however, they typically rely on additional recovery data, separately trained recovery modules, or joint optimization tailored to a specific policy backbone.

## B Method Details

## B.1 Details of Event-Triggered Truncation

This section provides the full event-triggered truncation rule used by LVM. Given the online inconsistency score $E _ { t } .$ , the monitor maintains a sliding window of recent scores,

$$
\mathbf {E} _ {W} = \{E _ {t - w + 1}, \dots , E _ {t} \}.
$$

Let

$$
M _ {e} = \mathrm{median} (\mathbf {E} _ {W}), \qquad \mathrm{MAD} = \mathrm{median} (| E _ {i} - M _ {e} |), \quad E _ {i} \in \mathbf {E} _ {W}.
$$

We use the median absolute deviation because it is less sensitive to transient score spikes than the mean and variance. Based on this robust statistic, we define asymmetric on/of thresholds,

$$
T _ {\mathrm{on}} = M _ {e} + \lambda_ {\mathrm{on}} \mathrm{MAD}, \qquad T _ {\mathrm{off}} = M _ {e} + \lambda_ {\mathrm{off}} \mathrm{MAD}, \qquad \lambda_ {\mathrm{on}} > \lambda_ {\mathrm{off}}.
$$

The higher threshold $T _ { \mathrm { o n } }$ is used to confirm abnormal deviation, while the lower threshold $T _ { \mathrm { o f f } }$ provides hysteresis and prevents rapid oscillation between normal and abnormal states.

To avoid triggering on isolated spikes, we maintain a persistence counter:

$$
c _ {t} = \left\{ \begin{array}{l l} c _ {t - 1} + 1, & \text {if E_{t} > T_{on}}, \\ 0, & \text {if E_{t} <   T_{off}}, \\ c _ {t - 1}, & \text {otherwise}. \end{array} \right.\tag{12}
$$

An interrupt event is triggered when

$$
c _ {t} \geq p,\tag{13}
$$

where p is the patience parameter. After an interrupt event, the remaining actions in the current execution queue are discarded, the counter is reset, and the next policy query is performed under corrective mode. If h actions have already been executed from the current queue, the realized action horizon is

$$
H _ {\mathrm{adaptive}} = h <   H.
$$

In this way, the system preserves long-horizon execution during stable phases, while shortening the horizon when persistent visual drift indicates that the current action chunk is no longer reliable.

## B.2 LVM and OGG Runtime Parameters

LVM monitoring parameters. The Latent-space Vision Monitor uses a robust online thresholding state machine. We maintain a sliding window of recent inconsistency scores with window size 15. The dynamic thresholds are computed using $\lambda _ { \mathrm { o n } } = 3 . 0$ and $\lambda _ { \mathrm { o f f } } = 2 . 0$ . An interrupt event is triggered only when the inconsistency score exceeds the activation threshold for $p = 5$ consecutive steps. Similarly, the monitor requires 5 consecutive safe steps for reset. To avoid repeated triggers immediately after intervention, we use an interrupt cooldown of 10 steps.

OGG parameters. Online Gradient Guidance is invoked only for the single policy query immediately following an interrupt-triggered truncation. When an interrupt event occurs, the remaining action queue is cleared, the next replan is marked as a guidance replan, and OGG is applied during that next action-chunk generation. Subsequent policy calls return to standard inference unless another interrupt event is detected. The guidance strength is controlled by η. In our experiments, we use $\eta = 1$ as the default setting for $\pi _ { 0 . 5 } .$ , SmolVLA, and X-VLA. The sensitivity of performance to η is analyzed in Table 7.

## C Training and Implementation Details

## C.1 Benchmarks and Evaluation Protocol

We evaluate VLA-Corrector on three simulation benchmarks: MetaWorld Yu et al. (2020), LIBERO Liu et al. (2023), and RoboTwin v2.0. MetaWorld tests contact-rich manipulation robustness across dificulty splits, LIBERO evaluates language-conditioned long-horizon task execution, and RoboTwin broadens the evaluation to 50 tasks under clean and randomized scene configurations. We use $\pi _ { 0 . 5 }$ Intelligence et al. (2025) as the main backbone, and further test SmolVLA Shukor et al. (2025) and X-VLA Zheng et al. (2025) for cross-architecture evaluation. We report task success rate as the primary metric, together with policy calls, success-per-call eficiency, and post-interrupt recovery rate when applicable.

For MetaWorld, tasks are grouped into four dificulty splits: Easy, Medium, Hard, and Very Hard. All online evaluations are conducted task by task within each split, and we report the average success rate over tasks in the corresponding split. Unless otherwise specified, each task is evaluated with 20 episodes. For the corrector data-scaling study, we use a fixed episode-level split before subsampling the training data: 10% of episodes are held out for validation, 10% are held out for testing, and the remaining episodes form the training pool. We then subsample this training pool according to the training ratio r. In the multi-seed data-scaling analysis, we use split seeds {42, 123, 999} and require at least 20 episodes in the train, validation, and test partitions. All simulation experiments are conducted on 8 NVIDIA A100-SXM4-40GB GPUs.

For RoboTwin v2.0, we evaluate all 50 tasks in both the clean and randomized configurations, using 50 episodes per task and configuration. Initial conditions and evaluation seeds are paired across methods, yielding 5,000 episodes per method. We compare the original fixed-horizon baseline $( H = 5 0 )$ , a latency-matched baseline $\left( H = 3 0 \right)$ , and VLA-Corrector $\left( H = 5 0 \right)$ , all with the same frozen $\pi _ { 0 . 5 }$ backbone. We report clean, randomized, and pooled success rates with 95% Wilson confidence intervals, together with average policy calls per episode and measured wall-clock latency per environment step.

## C.2 External Corrector Architecture

The external corrector $M _ { \phi }$ is implemented as a residual MLP for short-horizon latent dynamics prediction. We first map the executed action through a linear embedding layer, concatenate the resulting action feature with the current visual latent $Z _ { t } ^ { \mathrm { r e a l } }$ , and feed the combined representation into a stack of residual MLP blocks. The network predicts the short-horizon latent residual rather than the absolute future latent state. In all experiments, we use a four-layer hidden configuration with width [2048, 2048, 2048, 2048]. The exact number of parameters varies slightly across benchmarks and VLA backbones because both the action dimensionality and the visual latent dimensionality difer across settings. Overall, the correctors used in our experiments contain approximately 38–42M parameters, and we refer to this external module as a lightweight ∼40M MLP corrector.

## C.3 Corrector Training

The corrector is trained after the VLA backbone has been fine-tuned on the benchmark training set. We freeze the VLA backbone and use its visual encoder to extract latent representations from demonstration trajectories. The corrector is then trained on these frozen latents to predict the short-horizon visual latent residual induced by the executed action. We use AdamW as the optimizer with learning rate $3 \times 1 0 ^ { - 4 }$ and weight decay $1 0 ^ { - 4 }$ . The learning rate is scheduled by cosine annealing with $\eta _ { \mathrm { m i n } } = 0 . 0 1 \times \mathrm { l r }$ . We train for 30 epochs with early stopping patience $5 .$ For the ratio-sweep training runs, we use batch size 128 and evaluation batch size 256. For the deployed h1-k10 correctors used in evaluation, we use batch size 512, train for 30 epochs, and optimize a cosine-based training loss. The number of optimization steps depends on the number of training samples under each data ratio:

$$
\text { steps   per   epoch } = \left\lceil \frac {N _ {\text { train }}}{B} \right\rceil , \quad \text { total   steps } = 3 0 \times \text { steps   per   epoch },
$$

where $N _ { \mathrm { t r a i n } }$ is the number of training samples and B is the batch size.

## C.4 Compute Resources

All simulation experiments are conducted on NVIDIA A100-SXM4-40GB GPUs. The main training and evaluation jobs are run on a server with 8 such GPUs. Corrector training is lightweight compared with VLA backbone fine-tuning because it only optimizes the external MLP on frozen VLA features. During online evaluation, LVM monitoring adds only a forward pass through the external corrector, while OGG introduces additional gradient computation only for the single recovery query following an interrupt event.

Table 6 Coupled vs. decoupled detection. Success rate $( \% , \uparrow )$ . Both variants use OGG; only the detector design difers.

<table><tr><td>Detector</td><td>Easy↑</td><td>Medium↑</td><td>Hard↑</td><td>Very Hard↑</td><td>Avg.↑</td></tr><tr><td>Internal Head + OGG</td><td>65.2</td><td>46.0</td><td>36.5</td><td>50.5</td><td>49.55</td></tr><tr><td>Decoupled LVM + OGG</td><td>83.2 ↑ 18.0</td><td>61.7 ↑ 15.7</td><td>47.5 ↑ 11.0</td><td>65.0 ↑ 14.5</td><td>64.35 ↑ 14.80</td></tr></table>

Table 7 Ablation on OGG guidance strength η. Success rate $( \% , \uparrow )$ on MetaWorld. The highlighted row is the default setting.

<table><tr><td>Guidance Strength</td><td>Easy↑</td><td>Medium↑</td><td>Hard↑</td><td>Very Hard↑</td><td>Avg.↑</td></tr><tr><td> $\eta = 0.1$ </td><td>82.7</td><td>56.8</td><td>45.2</td><td>66.0</td><td>62.68</td></tr><tr><td> $\eta = 1$  (default)</td><td>83.2 ↑ 0.5</td><td>61.7 ↑ 4.9</td><td>47.5 ↑ 2.3</td><td>65.0 ↓ 1.0</td><td>64.35 ↑ 1.67</td></tr><tr><td> $\eta = 10$ </td><td>82.5</td><td>56.8</td><td>38.3</td><td>65.0</td><td>60.65 ↓ 3.70</td></tr><tr><td> $\eta = 100$ </td><td>80.9</td><td>55.0</td><td>36.7</td><td>63.0</td><td>58.90 ↓ 5.45</td></tr></table>

Table 8 Ablation on LVM capacity. Success rate $( \% , \uparrow )$ on MetaWorld. The highlighted row is the default setting.

<table><tr><td>Monitor Capacity</td><td>Easy↑</td><td>Medium↑</td><td>Hard↑</td><td>Very Hard↑</td><td>Avg.↑</td></tr><tr><td>LVM-10M</td><td>78.5</td><td>52.4</td><td>39.8</td><td>55.6</td><td>56.58</td></tr><tr><td>LVM-40M (default)</td><td>83.2 ↑ 4.7</td><td>61.7 ↑ 9.3</td><td>47.5 ↑ 7.7</td><td>65.0 ↑ 9.4</td><td>64.35 ↑ 7.77</td></tr><tr><td>LVM-160M</td><td>82.8 ↓ 0.4</td><td>62.1 ↑ 0.4</td><td>48.0 ↑ 0.5</td><td>64.2 ↓ 0.8</td><td>64.28 ↓ 0.07</td></tr></table>

## D Additional Experimental Results

## D.1 Ablation Sensitivity

We provide additional ablations on detector design, OGG guidance strength, and LVM capacity. All experiments are conducted with $\pi _ { 0 . 5 }$ on MetaWorld.

For the internal detector, we fine-tune $\pi _ { 0 . 5 }$ with an additional auxiliary loss that matches the prediction task of LVM: the final-layer hidden state of the last token is combined with the executed action and used to predict the same short-horizon latent residual. Table 6 shows that this coupled design performs much worse than the decoupled external LVM. This suggests that joint auxiliary detection training can interfere with the action-generation representation, whereas the external LVM preserves the backbone policy and provides a cleaner monitoring signal.

Tables 7 and 8 show that VLA-Corrector does not require overly strong guidance or excessively large monitors. η = 1 achieves the best average success, while larger guidance weakens performance on harder tasks. Increasing LVM capacity from 10M to 40M substantially improves success, but 160M provides almost no additional average gain.

## D.2 Cross-Domain Corrector Generalization

To examine whether the external corrector learns a transferable latent-dynamics signal rather than only memorizing benchmark-specific trajectories, we evaluate cross-domain corrector transfer on MetaWorld. We use the same $\pi _ { 0 . 5 }$ MetaWorld baseline policy and compare two VLA-Corrector variants: one using a corrector trained on LIBERO demonstrations, and the other using a corrector trained on MetaWorld demonstrations. Both are evaluated on MetaWorld under the same protocol.

Table 9 shows that the LIBERO-trained corrector still improves the MetaWorld baseline from 48.7% to 51.8%, suggesting that the learned latent-dynamics consistency signal has limited but non-trivial cross-domain transfer. However, the MetaWorld-trained corrector achieves a much larger gain, improving the average success rate to 58.7%. This indicates that while the corrector can capture partially transferable notions of on-track visual dynamics, domain-matched demonstrations remain important for accurate deviation detection and efective corrective guidance.

Table 9 Cross-domain corrector generalization on MetaWorld. Success rate $( \% , \uparrow )$ of the same $\pi _ { 0 . 5 }$ MetaWorld baseline equipped with correctors trained from diferent demonstration domains. Green arrows indicate absolute improvement over the baseline.

<table><tr><td>Method</td><td>Easy↑</td><td>Medium↑</td><td>Hard↑</td><td>Very Hard↑</td><td>Avg.↑</td></tr><tr><td> $\pi_{0.5}$  Baseline</td><td>70.5</td><td>45.0</td><td>38.3</td><td>41.0</td><td>48.7</td></tr><tr><td>+ LIBERO-trained Corrector</td><td>70.9 ↑ 0.4</td><td>50.5 ↑ 5.5</td><td>40.7 ↑ 2.4</td><td>45.0 ↑ 4.0</td><td>51.8 ↑ 3.1</td></tr><tr><td>+ MetaWorld-trained Corrector</td><td>74.1 ↑ 3.6</td><td>53.8 ↑ 8.8</td><td>52.9 ↑ 14.6</td><td>54.0 ↑ 13.0</td><td>58.7 ↑ 10.0</td></tr></table>

Table 10 Performance–efficiency trade-off across VLA backbones and action horizons. Success rate, average policy calls per episode, and relative eficiency gains are reported. The arrows (↑/↓) in the header explicitly indicate whether a higher or lower metric is optimal.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Horizon</td><td colspan="2">Baseline</td><td colspan="2">Ours (LVM+OGG)</td><td colspan="3">Improvements</td></tr><tr><td>Succ (%)↑</td><td>Calls ↓</td><td>Succ (%)↑</td><td>Calls ↓</td><td>ΔSucc (%)↑</td><td>ΔCalls ↓</td><td>Eff. Gain (%)↑</td></tr><tr><td rowspan="5"> $\pi_{0.5}$ </td><td>10</td><td>64.50</td><td>20.41</td><td>72.40</td><td>17.64</td><td>↑ 7.90</td><td>↓ 2.77</td><td>+29.9%</td></tr><tr><td>20</td><td>61.40</td><td>10.77</td><td>69.10</td><td>9.95</td><td>↑ 7.70</td><td>↓ 0.82</td><td>+21.8%</td></tr><tr><td>30</td><td>56.10</td><td>7.97</td><td>63.50</td><td>7.92</td><td>↑ 7.40</td><td>↓ 0.05</td><td>+13.9%</td></tr><tr><td>40</td><td>54.45</td><td>5.96</td><td>60.80</td><td>6.04</td><td>↑ 6.35</td><td>↑ 0.08</td><td>+10.2%</td></tr><tr><td>50</td><td>48.72</td><td>5.15</td><td>58.70</td><td>4.98</td><td>↑ 9.98</td><td>↓ 0.17</td><td>+24.6%</td></tr><tr><td rowspan="5">SmolVLA</td><td>10</td><td>61.90</td><td>19.27</td><td>73.00</td><td>15.64</td><td>↑ 11.10</td><td>↓ 3.63</td><td>+45.3%</td></tr><tr><td>20</td><td>58.90</td><td>11.32</td><td>68.90</td><td>9.50</td><td>↑ 10.00</td><td>↓ 1.82</td><td>+39.4%</td></tr><tr><td>30</td><td>58.10</td><td>7.90</td><td>65.20</td><td>7.60</td><td>↑ 7.10</td><td>↓ 0.30</td><td>+16.6%</td></tr><tr><td>40</td><td>56.80</td><td>5.64</td><td>67.20</td><td>5.13</td><td>↑ 10.40</td><td>↓ 0.51</td><td>+30.0%</td></tr><tr><td>50</td><td>54.20</td><td>4.86</td><td>62.90</td><td>4.68</td><td>↑ 8.70</td><td>↓ 0.18</td><td>+20.6%</td></tr><tr><td rowspan="5">X-VLA</td><td>4</td><td>68.50</td><td>46.58</td><td>72.00</td><td>35.20</td><td>↑ 3.50</td><td>↓ 11.38</td><td>+39.1%</td></tr><tr><td>8</td><td>67.80</td><td>23.18</td><td>72.10</td><td>19.48</td><td>↑ 4.30</td><td>↓ 3.70</td><td>+26.5%</td></tr><tr><td>16</td><td>55.50</td><td>13.71</td><td>60.90</td><td>13.92</td><td>↑ 5.40</td><td>↑ 0.21</td><td>+8.1%</td></tr><tr><td>24</td><td>52.90</td><td>10.13</td><td>59.20</td><td>9.92</td><td>↑ 6.30</td><td>↓ 0.21</td><td>+14.3%</td></tr><tr><td>32</td><td>44.00</td><td>8.61</td><td>54.40</td><td>8.34</td><td>↑ 10.40</td><td>↓ 0.27</td><td>+27.6%</td></tr></table>

∆Succ: absolute success-rate improvement. ∆Calls: absolute change in policy calls per episode. Eff. Gain: relative improvement in success-per-call over the baseline.

## D.3 Full Performance–Efficiency Results

Table 10 provides the full performance–eficiency results across all evaluated VLA backbones and action horizons. These results complement the main-text analysis by reporting the exact success rates, average policy calls, and success-per-call eficiency gains for each setting. Overall, VLA-Corrector consistently improves task success while maintaining or reducing the number of policy calls in most cases, leading to positive success-per-call eficiency gains across all backbones. The gains are especially pronounced at longer horizons, where the open-loop blind spot is larger and stale actions are more likely to accumulate errors before the next policy query.

## D.4 Data Efficiency of Corrector Training

Analysis. To quantify how much data is needed to train the external corrector, we subsample the MetaWorld demonstration set with training ratios $r \in \{ 0 . 2 , 0 . 4 , 0 . 6 , 0 . 8 , 1 . 0 \}$ and evaluate the resulting VLA-Corrector on the same task suite. As shown in Table 11, the corrector is already efective with limited data: using only 20% of the demonstrations yields a 48.32% macro-average success rate. Performance improves as more data is added and reaches 52.20% at $r = 0 . 6 .$ , after which the gain becomes marginal. This suggests that the corrector does not require exhaustive coverage of all task variations; once it has seen enough examples of on-track local dynamics, it can provide a useful signal for detecting deviations and guiding recovery.

The trend also shows that the corrector benefits from additional data, but with diminishing returns. Increasing the training ratio from 0.2 to 0.6 brings a clear improvement, while increasing it further from 0.6 to 0.8 changes the macro average only slightly. Full-data training still achieves the best overall result, especially on the Hard split, but the relatively strong performance at moderate data ratios supports our design motivation: VLA-Corrector learns a local consistency signal rather than a full policy or heavy world model, making it data-eficient to train. On the Very Hard split, the small fluctuation across higher ratios indicates that extremely challenging contact-rich tasks are influenced by factors beyond corrector data scale alone, such as disturbance severity and backbone recovery capability.

Table 11 Data efficiency of corrector training on MetaWorld. We compare the original open-loop baseline at NAS=50 with VLA-Corrector trained using diferent fractions of demonstration trajectories. The results show that moderate corrector data already surpasses the baseline and that performance largely saturates around 60% of the training set. Bold indicates the best result in each column.

<table><tr><td>Method / Training Ratio</td><td>Easy</td><td>Medium</td><td>Hard</td><td>Very Hard</td><td>Macro Avg.</td></tr><tr><td>Baseline (NAS=50)</td><td>70.54</td><td>45.00</td><td>38.33</td><td>41.00</td><td>48.72</td></tr><tr><td>VLA-Corrector (r = 0.2)</td><td>71.07</td><td>49.55</td><td>36.67</td><td>36.00</td><td>48.32</td></tr><tr><td>VLA-Corrector (r = 0.4)</td><td>70.36</td><td>45.91</td><td>38.33</td><td>42.00</td><td>49.15</td></tr><tr><td>VLA-Corrector (r = 0.6)</td><td>70.89</td><td>52.73</td><td>39.17</td><td>46.00</td><td>52.20</td></tr><tr><td>VLA-Corrector (r = 0.8)</td><td>71.07</td><td>53.64</td><td>38.33</td><td>46.00</td><td>52.26</td></tr><tr><td>VLA-Corrector (r = 1.0)</td><td>73.21</td><td>55.91</td><td>44.17</td><td>44.00</td><td>54.32</td></tr></table>

Table 12 LVM timing during deployment on MetaWorld. We measure 500 held-out observations after removing 20 warm-up observations. “Full LVM computation” is all work done by the monitor; “wait added to control loop” is the part that the controller actually waits for.

<table><tr><td>Component</td><td>Mean (ms)</td><td>p95 (ms)</td></tr><tr><td>Preprocessing</td><td>1.79</td><td>2.06</td></tr><tr><td>Frozen vision encoder</td><td>21.31</td><td>21.46</td></tr><tr><td>40M residual MLP</td><td>2.79</td><td>2.93</td></tr><tr><td>Mismatch + threshold</td><td>0.91</td><td>1.08</td></tr><tr><td>Full LVM computation</td><td>26.96</td><td>27.57</td></tr><tr><td>Wait added to control loop</td><td>0.008</td><td>0.009</td></tr></table>

Table 13 OGG timing during deployment on an NVIDIA A100-SXM4-40GB. We compare 200 matched ordinary/OGG pairs after 20 warm-up runs. The random state is restored for each pair, the order is balanced, and CUDA timing is synchronized; every OGG call uses two guidance steps.

<table><tr><td>Call</td><td>Mean latency</td></tr><tr><td>Ordinary</td><td>232.490 ms</td></tr><tr><td>OGG</td><td>243.223 ms</td></tr><tr><td>Increment per trigger</td><td>10.733 ms</td></tr><tr><td>95% CI</td><td>[10.430, 11.053] ms</td></tr></table>

## D.5 Inference-Time Cost and Control Delay

The monitor runs in parallel with the robot’s current action. We therefore report two diferent times: (1) the time needed to complete the monitor’s computation, and (2) the part of that time that actually makes the controller wait. We also measure the extra time used by the OGG recovery call. All measurements use synchronized CUDA timing. The LVM measurements are listed in Table 12, the OGG measurements in Table 13, and the physical-control check in Table 14.

As shown in Table 12, the monitor performs 26.96 ms of computation on average and 27.57 ms at p95. Because this work runs while the robot is executing, it adds only 0.008 ms of average waiting time (0.009 ms at p95) to the control loop. Thus, the computation is real, but almost all of it is hidden by parallel execution; the 0.008 ms number is only the extra wait, not the total monitor cost.

Table 14 Timing check on AgileX PiPER. PiPER executes 10 actions per second, so the controller has 100 ms for each action.

<table><tr><td>Quantity</td><td>Result</td></tr><tr><td>Control period</td><td>100 ms/action (10 Hz)</td></tr><tr><td>Full LVM</td><td>26.96 ms mean / 27.57 ms p95</td></tr><tr><td>p95 remaining margin</td><td>72.43 ms</td></tr><tr><td>Deadline misses</td><td>0 / 500</td></tr><tr><td>Wait added to control loop</td><td>0.008 ms mean</td></tr></table>

Table 15 Asynchronous deployment-path accounting. Intrinsic computation is measured with CUDA synchronization; exposed wait is the delay visible to the control loop after asynchronous pipelining. OGG is applied only to the single recovery query following an interrupt.

<table><tr><td>Quantity</td><td>Measured result</td></tr><tr><td>Full LVM intrinsic computation</td><td>26.96 ms mean / 27.57 ms p95</td></tr><tr><td>Exposed asynchronous control-path wait</td><td>0.008 ms mean / 0.009 ms p95</td></tr><tr><td>Ordinary policy call</td><td>232.490 ms</td></tr><tr><td>OGG-guided policy call</td><td>243.223 ms</td></tr><tr><td>Increment on the recovery call</td><td>10.733 ms/trigger, 95% CI [10.430, 11.053] ms</td></tr><tr><td>PiPER control period</td><td>100 ms/action (10 Hz)</td></tr><tr><td>Observed deadline misses</td><td>0/500</td></tr></table>

Table 16 RTC context under the 10 Hz control-period model. The response intervals are obtained from the ordinary/guided call latencies and the persistence rule (p = 5); they clarify the diference between periodic scheduling and failure-aware truncation.

<table><tr><td>Configuration</td><td>Requery rule</td><td>Calls/step</td><td>Relative calls</td><td>Response under the timing model</td></tr><tr><td>LVM-gated truncation</td><td>H=50; extra replan after persistent mismatch</td><td>0.0379</td><td>1.0×</td><td>Stale suffix stopped at ≈400 ms</td></tr><tr><td>Reviewer-style RTC</td><td>Periodic, s=3</td><td>0.333</td><td>8.8×</td><td>Replacement at 297–597 ms</td></tr><tr><td>Call-matched RTC</td><td>Periodic, s≈26.4</td><td>0.0379</td><td>1.0×</td><td>Replacement at ≈297–2,937 ms</td></tr></table>

Table 13 shows that an ordinary policy call takes 232.490 ms, while an OGG-guided call takes 243.223 ms. OGG therefore adds 10.733 ms only when it is used for recovery after an interrupt; it does not slow down ordinary policy calls.

Discussion. Table 14 summarizes the physical check. MetaWorld is evaluated without a real-time clock, but PiPER must complete one action every 100 ms. The monitor’s p95 computation time is 27.57 ms, leaving 72.43 ms in each control period, and none of the 500 measured steps missed its deadline. The monitor therefore remains responsive in the physical setup: its full computation is accounted for, while only 0.008 ms of average waiting is added to the controller. OGG adds 10.733 ms only on the occasional recovery call.

## D.6 Asynchronous and RTC Comparisons

Because Real-Time Chunking (RTC) periodically refreshes action chunks while the previous chunk is executing, it overlaps with the asynchronous execution path of VLA-Corrector and is a natural comparison point. The two mechanisms nevertheless address diferent parts of the control problem: RTC schedules the next policy query, whereas VLA-Corrector uses the observed-vs. predicted visual-dynamics mismatch to produce an explicit error signal, stop stale actions, and guide the recovery replan. We therefore report both the deployment-path timing and a direct overlapped-execution success comparison. Table 15 collects the asynchronous timing measurements, Table 16 gives the RTC timing context, and Table 17 reports the direct overlapped-execution experiment.

Table 15 makes the deployment claim precise: asynchronous execution does not remove the LVM’s 26.96 ms of intrinsic computation, but only 0.008 ms of mean waiting time is exposed on the control path. The OGG increment is similarly localized to the recovery call, rather than being charged to every action step.

Table 17 Direct overlapped-execution comparison on MetaWorld-50. All methods use the same frozen $\pi _ { 0 . 5 } ,$ horizon $H = 5 0$ , asynchronous execution path, and 50 paired episodes per task (2,500 episodes per method). The RTC baseline periodically alternates $s = 2 6$ . The three E -gated variants share the same detector and truncation settings and difer only in the triggered replan; the random direction is scaled to the mean OGG guidance norm.

<table><tr><td>Method</td><td>Triggered guidance</td><td>Success (95% CI)</td><td>Calls/100 steps</td></tr><tr><td>RTC, s = 26</td><td>None</td><td>61.3 [59.4, 63.2]</td><td>3.85</td></tr><tr><td>Et-gated replan</td><td>None</td><td>62.4 [60.5, 64.3]</td><td>4.08</td></tr><tr><td>Et-gated random</td><td>Random</td><td>61.9 [60.0, 63.8]</td><td>4.12</td></tr><tr><td>Et-gated OGG</td><td>OGG</td><td>67.1 [65.3, 68.9]</td><td>3.88</td></tr></table>

Table 16 gives an analytical timing context rather than a hardware benchmark. It shows that periodic RTC can deliver a newer ordinary chunk, but does not itself provide an explicit failure signal or a recovery-directed target. At the same call budget, RTC must distribute policy calls uniformly, whereas LVM-gated execution allocates the additional replan after an observed persistent mismatch. The direct experiment below tests whether this distinction remains visible in task success when all methods use overlapped execution.

For clarity, all main-text experiments and the previously reported VLA-Corrector results use rtc\_config=None: RTC itself is disabled in those experiments. This setting does not make LVM monitoring synchronous; the deployment-path measurements above still run the monitor asynchronously while the current action executes. To compare against RTC directly, we additionally run the experiment in Table 17 with asynchronous inference enabled for every method and with an RTC baseline included only for this comparison. For context, the RTC analysis reports 97 ms for inference-time RTC versus 76 ms for vanilla π<sub>0.5</sub> on an RTX 4090, because prefix inpainting requires gradient computation; its training-time variant removes that inference operation only through a separately trained prefix-conditioned $\pi _ { 0 . 6 }$ checkpoint. These context figures are kept separate from the direct overlapped-execution experiment and the A100 deployment-path measurements above.

As shown in Table 17, the additional experiment directly evaluates VLA-Corrector under overlapped execution rather than inferring its behavior from sequential timing. At comparable policy-call rates, E -gated OGG reaches 67.1% success versus 61.3% for periodic RTC, an improvement of 5.8 points (paired 95% CI [+4.4, +7.2], McNemar $p < 1 0 ^ { - 1 3 } )$ . Thus, the advantage persists when policy inference is overlapped with action execution and is not explained by a higher call frequency. The matched-norm random control reaches 61.9%, which is 5.2 points below E<sub>t</sub>-gated OGG (95% $\mathrm { C I } \ [ + 3 . 8 , + 6 . 6 ] , p < 1 0 ^ { - 1 1 } )$ , and does not improve over the ordinary $E _ { t } { \mathrm { - g a t e d } }$ replan (61.9% vs. 62.4%). These results isolate the contribution of the recovery direction: the gain comes from using the LVM-derived error signal to guide OGG toward a corrective target, rather than from arbitrary perturbation or simply requesting another policy chunk.

## E Real-World Details

## E.1 Robot Platform and Task Suite

Robot platform. All real-world experiments are conducted on an AgileX PiPER 6-DoF robotic arm. We use π<sub>0.5</sub> as the action-chunked VLA baseline and evaluate VLA-Corrector on top of the same fine-tuned backbone. Both methods share the same RGB camera observation, language instruction, action horizon, and initial task conditions. Each task is evaluated with 20 trials per method. A trial is counted as successful if the robot completes the specified task without human reset after execution starts.

Task design. We design three groups of real-world tasks with increasing dificulty. The first group evaluates standard pick-and-place behavior, where the original action chunk often remains valid. The second group evaluates precise alignment, where small accumulated errors can cause missed placement, collision, or failed insertion. The third group introduces online human disturbances during precision-sensitive phases, directly testing whether VLA-Corrector can recover when the remaining action chunk becomes stale. Table 18 summarizes the task definitions.

Table 18 Real-world task suite on AgileX PiPER. The benchmark contains three task groups and nine tasks in total. The disturbance group reuses manipulation skills from the first two groups but introduces online pose changes during precision-sensitive phases.

<table><tr><td>Group</td><td>Task</td><td>Description</td></tr><tr><td rowspan="3">Pick-and-place</td><td>Cube-to-region</td><td>Pick up a cube and place it inside a marked target region.</td></tr><tr><td>Cube-to-drawer-top</td><td>Pick up a cube and place it on top of a small drawer platform.</td></tr><tr><td>Object-to-container</td><td>Pick up a cube and place it into a specific bowl.</td></tr><tr><td rowspan="3">Alignment</td><td>Corner placement</td><td>Place a cube precisely at the upper-left corner of the drawer top.</td></tr><tr><td>Square-hole insertion</td><td>Align and insert a small cube into a square hole or slot.</td></tr><tr><td>Edge alignment</td><td>Align an object with a narrow marked edge or target strip.</td></tr><tr><td rowspan="3">Disturbance recovery</td><td>Moving-object grasp</td><td>Move the target object when the gripper is about to grasp it.</td></tr><tr><td>Moving-placement target</td><td>Move the placement region after the object is grasped.</td></tr><tr><td>Moving-insertion target</td><td>Move the square hole or alignment target before insertion.</td></tr></table>

## E.2 Per-Task Real-World Results

Table 19 reports per-task success rates. The gains are modest on pick-and-place tasks because the baseline can often complete the task when the object and target remain static. The gains become larger on alignment tasks, where small pose errors are less tolerable. The largest improvement appears in disturbance recovery tasks, where the current action chunk often becomes stale after the object or target is manually moved.

Table 19 Per-task real-world success rates. Success rate (%, ↑) over 20 trials per task on the AgileX PiPER 6-DoF arm. Group and overall rows report mean ± standard deviation across tasks. Green arrows indicate absolute improvement over the π baseline.

<table><tr><td>Group</td><td>Task</td><td> $\pi_{0.5}$  Baseline↑</td><td>+ VLA-Corrector↑</td></tr><tr><td rowspan="4">Pick-and-place</td><td>Cube-to-region</td><td>75.0</td><td>85.0 ↑ 10.0</td></tr><tr><td>Cube-to-drawer-top</td><td>70.0</td><td>80.0 ↑ 10.0</td></tr><tr><td>Object-to-container</td><td>65.0</td><td>70.0 ↑ 5.0</td></tr><tr><td>Group Avg.</td><td>70.0 ± 5.0</td><td>78.3 ± 7.6 ↑ 8.3</td></tr><tr><td rowspan="4">Alignment</td><td>Corner placement</td><td>60.0</td><td>75.0 ↑ 15.0</td></tr><tr><td>Square-hole insertion</td><td>50.0</td><td>70.0 ↑ 20.0</td></tr><tr><td>Edge alignment</td><td>60.0</td><td>75.0 ↑ 15.0</td></tr><tr><td>Group Avg.</td><td>56.7 ± 5.8</td><td>73.3 ± 2.9 ↑ 16.6</td></tr><tr><td rowspan="4">Disturbance recovery</td><td>Moving-object grasp</td><td>45.0</td><td>75.0 ↑ 30.0</td></tr><tr><td>Moving-placement target</td><td>40.0</td><td>70.0 ↑ 30.0</td></tr><tr><td>Moving-insertion target</td><td>35.0</td><td>60.0 ↑ 25.0</td></tr><tr><td>Group Avg.</td><td>40.0 ± 5.0</td><td>68.3 ± 7.6 ↑ 28.3</td></tr><tr><td>Overall</td><td>Avg. across 9 tasks</td><td>55.6 ± 13.8</td><td>73.3 ± 7.1 ↑ 17.7</td></tr></table>

## E.3 Real-World Protocol and Task Details

Evaluation protocol. For each real-world task, we collect 10–30 human teleoperated demonstration trajectories and use them to fine-tune the $\pi _ { 0 . 5 }$ backbone for that task. The same fine-tuned backbone is then used by both the baseline and VLA-Corrector, so the comparison isolates the efect of the inference-time correction module. For each task and each method, we run 20 evaluation trials. Initial object poses are randomized within a small predefined region while keeping the task feasible for the 6-DoF PiPER arm.

Pick-and-place tasks. The pick-and-place group includes placing a cube into a marked region, placing a cube on top of a drawer platform, and placing a small object into a bowl. These tasks are relatively tolerant to small pose errors, so the original action chunk often remains usable. VLA-Corrector mainly helps when occasional drift occurs during approach or release.

Alignment tasks. The alignment group increases the precision requirement. In corner placement, the robot must place a cube at a specific corner of the drawer top. In square-hole insertion, the cube must be aligned with a square opening before insertion. In edge alignment, the robot must align an object with a narrow visual marker or edge. These tasks are more sensitive to accumulated error, making stale action execution more harmful near the final contact.

Disturbance recovery tasks. The disturbance group directly evaluates the open-loop blind spot. We introduce manual perturbations at predefined task phases rather than at fixed time steps. For moving-object grasp, the target object is shifted when the gripper is close to grasping it. For moving-placement target, the robot first grasps the object normally, and the target platform or placement region is shifted before release. For moving-insertion target, the square hole or alignment target is shifted when the robot is about to align or insert. The disturbance is applied once per trial, kept within the camera view, and constrained to a physically recoverable range. The same timing rule and disturbance range are used for both the baseline and VLA-Corrector.

## E.4 Observed Real-World Failure Modes

Despite the improvements, VLA-Corrector still fails in several real-world cases. If the target object or fixture is moved beyond the reachable region, or if the disturbance happens after the gripper has entered an unfavorable pose, the robot may no longer have enough workspace or time to recover through a single corrective replan. In tight alignment tasks, a 6-DoF arm without force feedback may fail due to contact geometry, friction, or small height errors even when the visual target is corrected. We also observe failures caused by visual ambiguity, such as partial occlusion by the gripper or poor contrast between the object and the target region. Finally, OGG still depends on the frozen $\pi _ { 0 . 5 }$ action prior: it can bias the next generation toward a better corrective direction, but it cannot create a recovery behavior that the backbone policy cannot represent.

## E.5 Real-World Demonstration Examples

We provide representative real-world demonstration examples mainly for the disturbance recovery tasks. In these cases, the object or target fixture is manually shifted during execution, allowing us to visualize the practical robustness of VLA-Corrector under human-induced disturbances.

![](images/5666a3197d8a1ad8f1a7e5c98e92a3db048c51b8fc08b01f97fc383b27780d02.jpg)  
Figure 9 Demo: Moving-object grasp. The robot picks up the cube and places it into the white bowl, while a human manually changes the cube’s position during the process.

![](images/718e31feb0eeb4be03a8f8750f3b62bfee4d32923191e146f258612b7520dde3.jpg)  
Figure 10 Demo: Moving-placement target. The robot picks up the cube and places it into the blue bowl, while a human manually changes the bowl’s position during the process.

![](images/bc4d7bb911324da0b8b9ad094916ab537cd05dd4b589e546b1d148a37396882c.jpg)  
Figure 11 Demo: Moving-insertion target. The robot picks up the cube and places it at the upper-left corner of the drawer top, while a human manually changes the drawer’s position during the process.