# When would Vision-Proprioception Policies Fail in Robotic Manipulation?

Jingxian Lu<sup>1,2,∗</sup>, Wenke Xia<sup>1,2,∗</sup>, Yuxuan Wu<sup>3</sup>, Zhiwu Lu<sup>1,2</sup>, Di Hu<sup>1,2,B</sup>

<sup>1</sup>Gaoling School of Artificial Intelligence, Renmin University of China, <sup>2</sup>Beijing Key Laboratory of Research on Large Models and Intelligent Governance, <sup>3</sup>School of Artificial Intelligence, Beihang University

<sup>∗</sup>Equal contribution, Corresponding author

## Abstract

Proprioceptive information is critical for precise servo control by providing real-time robotic states. Its collaboration with vision is highly expected to enhance performances of the manipulation policy in complex tasks. However, recent studies have reported inconsistent observations on the generalization of visionproprioception policies. In this work, we investigate this by conducting temporally controlled experiments. We found that during task sub-phases that robot’s motion transitions, which require target localization, the vision modality of the vision-proprioception policy plays a limited role. Further analysis reveals that the policy naturally gravitates toward concise proprioceptive signals that offer faster loss reduction when training, thereby dominating the optimization and suppressing the learning of the visual modality during motion-transition phases. To alleviate this, we propose the Gradient Adjustment with Phase-guidance (GAP) algorithm that adaptively modulates the optimization of proprioception, enabling dynamic collaboration within the vision-proprioception policy. Specifically, we leverage proprioception to capture robotic states and estimate the probability of each timestep in the trajectory belonging to motion-transition phases. During policy learning, we apply fine-grained adjustment that reduces the magnitude of proprioception’s gradient based on estimated probabilities, leading to robust and generalizable vision-proprioception policies. The comprehensive experiments demonstrate GAP is applicable in both simulated and real-world environments, across one-arm and dual-arm setups, and compatible with both conventional and Vision-Language-Action models. We believe this work can offer valuable insights into the development of vision-proprioception policies in robotic manipulation.

Email: Jingxian Lu at lujingxian1122@ruc.edu.com, Di Hu at dihu@ruc.edu.cn Project Page: https://gewu-lab.github.io/GAP/

## 1 Introduction

Proprioceptive information has long been recognized as a cornerstone of low-level robotic control, enabling smooth motor behavior through immediate access to the robot’s internal state. This capability is especially critical in tasks requiring high accuracy and fast correction, such as posture control [1, 2] and locomotion [3, 4]. In recent years, there has been growing interest in introducing proprioception to learning-based manipulation [5–7]. Despite the expectations that its inclusion will empower manipulation policies to maintain precision and robustness across various scenarios, existing works have reported inconsistent observations: HPT [8] demonstrated clear improvements under the joint utilization of vision and proprioception, while Octo [9] observed policies trained with additional propioception seemed generally worse than vision-only policies. This discrepancy exposes a critical obstacle to understanding: when vision-proprioception policies would fail in robotic manipulation?

![](images/a35d9b10049eb669ff5c614c505efbb6c988401962e01f46a05826ca7dcca465.jpg)  
Figure 1 The generalization of vision-proprioception policies. (left) Vision-Proprioception policies perform 15.8% worse than Vision-only policies. (right) We explore this through intervening the task execution of vision-only policy during different periods, by switching to vision-proprio policy. Such intervention has minimal impact during motion-consistent phases. However, during motion-transition phases, switching leads to noticeable degradation, indicating the vision modality fails to take effect.

Extensive prior studies have revealed that the importance of visual and proprioceptive information could change over time within manipulation [10–12], which referred to as Modality Temporality. For example, during motion-consistent phases where the robot performs ongoing movements, the policy can benefit more from proprioceptive signals. In contrast, during the transition intervals where the robot’s motion shifts, it is required to rely more on visual cues fo accurate target localization. To verify whether the vision-proprioception policy exhibits such collaboration, we conduct an intervention experiment in the controlled simulation environment. Concretely, we execute the “assembly” task using the vision-only policy, but for a specific 10-timestep period, we replace executed actions with those predicted by the vision-proprioception policy under the same observations. As shown in Figure 1 (right), the intervention brings minimal impact during motion-consistent phases like “move forward”, during motion-transition phases like “locate base” and “assemble them”, the switching leads to noticeable degradation. It suggests that the vision modality of the vision-proprioception policy fails to take effect during motion-transition phases.

We further investigate the underlying cause from an optimization perspective. During motion-transition phases, visual cues tend to be subtle and may only differ at the pixel level [13]. As a result, the vision-proprioception policy naturally gravitates toward the more concise proprioceptive signals to minimize the training loss, thereby dominating the optimization [14, 15]. Such dominance suppresses the learning of vision modality and ultimately leads to under-utilized visual information during motion-transition phases.

To alleviate this, we propose the Gradient Adjustment with Phase-guidance (GAP) algorithm that adaptively modulates the optimization of proprioception. Specifically, we first define the motion of the robot using concise proprioception signals and segment the trajectory into motion-consistent phases. Robot’s motion transits within the intervals between these phases, we thus employ a temporal network like LSTM to model transition processes. It helps eliminate potential errors introduced by the segmentation and estimates the probability that each timestep belongs to motion-transition phases. During policy learning, we guide the vision-proprioception policy to focus on essential visual cues of motiontransition phases, by applying fine-grained gradient adjustment that reduces the magnitude of proprioception’s gradient.

Our GAP algorithm facilitates the vision-proprioception policy to effectively utilize proprioception without suppressing the learning of visual modality. GAP is compatible with both conventional and Vision-Language-Action models, and its versatility and effectiveness have been validated by extensive experiments in both simulated and real-world environments. Our evaluations span multiple policy architectures, including MLP-based, diffusion-based, and transformer-based policies. Further, they cover articulated object manipulation, contact-rich interactions, rotation-sensitive tasks, and soft objec manipulation, across both one-arm and dual-arm robotic setups. Across all experimental settings and configurations, vision-proprioception policies equipped with our proposed GAP consistently lead to superior performances. We believe this work can offer valuable insights into the development of vision-proprioception policies in robotic manipulation.

## 2 Related Works

## 2.1 Vision-Proprioception Policy in Manipulation.

Vision has been the most commonly used modality in robotic manipulation policies [16–18]. While it provides sufficient information to complete many manipulation tasks, visual data often includes a large amount of noise, such as irrelevant background distractions [13]. Therefore, more concise proprioceptive information has been introduced by many works to assist robotic manipulation policy, with the expectation that it can provide complementary and physically grounded information for precise and robust task execution [6, 8, 19–23]. However, existing studies have reported confused observations: some works demonstrate clear improvements when integrating proprioceptive information with vision [6, 8], others observe limited gains or even detrimental effects [9, 19]. [21] attributes this to overfitting while [9] suggests it arises from causal confusion between the proprioceptive information and the target actions. In this study, we further explore the utilizations of each modality and introduce a modality-temporality perspective to offer valuable insights into the development of vision-proprioception policies for robotic manipulation.

## 2.2 Modality Temporality.

In manipulation tasks, each modality’s contribution to decision-making can vary significantly over time. For example, in “pick-place” task, policy must first rely on vision to locate the target object. When moving toward the object, proprioception becomes more critical for executing consistent and precise actions. It is proven by strong correlations between variations in modality data and task stages [12, 24, 25]. [11] summarizes such property of manipulation tasks as modality temporality. Given this nature of robotic manipulation tasks, recent works have proposed approaches based on dynamic fusion [12, 26] and modality selection [25] to improve the performance of multimodal manipulation policies. In this study, we introduce the modality-temporality perspective to understand the roles of vision and propriocetion and propose the gradient adjustment algorithm to enhance dynamic collaboration within the vision-proprioception policy.

## 3 Optimization Analysis of Vision-Proprioception Policies

In this section, we first formalize the problem and further analyze the generalization of vision-proprioception policies from an optimization perspective. The vision-proprioception policy is learned under the Behavior Cloning (BC) paradigm, which can be formulated as the Markov Decision Process (MDP) framework [27]. Formally, the policy π takes the environment observation $o _ { t } ~ \in ~ O$ as input at each timestep t. In this work, $o _ { t }$ includes RGB-sensor readings $v _ { t } ,$ and for vision-proprioception policy $\pi _ { v + s } ,$ it includes robot proprioceptive information $s _ { t }$ additionally. This proprioceptive information consists of the 6D pose of robot’s gripper $( p _ { t } ^ { x } , p _ { t } ^ { y } , p _ { t } ^ { z } , \theta _ { t } ^ { x } , \theta _ { t } ^ { y } , \theta _ { t } ^ { z } ) \in \mathbb { R } ^ { 6 }$ in Cartesian space and orientation, and a continuous value $g _ { t } \in [ 0 , 1 ]$ representing the degree of gripper opening, with $g _ { t } = 1$ denoting fully open and $g _ { t } = 0$ denoting fully closed.

The policy π maps the observation history to a sequence of actions: $\hat { a } _ { t + L } = \pi ( o _ { t - H : t } )$ , where L and H indicate the length of predicted action sequence and observation history respectively. For simplicity, we set omit them in the following discussion. The training objective can be formulated as:

$$
\pi^ {*} = \operatorname{argmin} _ {\pi} \mathbb {E} _ {(o _ {t}, a _ {t}) \sim \tau_ {e}} [ \mathcal {L} _ {B C} (\pi (o _ {t})), a _ {t} ],\tag{1}
$$

where $\tau _ { e }$ is expert demonstration dataset and $a _ { t }$ is action labels. In vanilla BC, L typically represents the Mean Squared Error (MSE) loss for continuous action spaces, or Cross-Entropy (CE) loss for discrete action spaces. We focus solely on the vanilla MSE loss here.

In this work, we adopt standard joint-learning architecture to design the vision-proprioception policy, which extracts features from both vision and proprioception modalities using two separate chunks $\phi _ { v } , \phi _ { s }$ . These features from two modalities are then concatenated and fed into the policy head ψ. Although some recent works have tried exploring alternative modality fusion approaches [8, 11], concatenation remains the most widely used approach [5, 6, 19]. To support our analysis under this fusion approach, we split the first layer of MLP-based policy head ψ into $\psi _ { s } , \psi _ { v }$ and rewrite the action prediction as:

$$
\hat {a} = (\psi_ {s} (f _ {s}) + \psi_ {v} (f _ {v})) \cdot W _ {s h a r e} + b,\tag{2}
$$

where $f _ { s } , f _ { v }$ is the feature extracted by $\phi _ { s } ( o ) , \phi _ { v } ( o )$ respectively. Under Gradient Descent (GD)-based policy learning, the optimization of the vision chunk’s parameters $\omega _ { v }$ is influenced by:

$$
\frac {\partial \mathcal {L} _ {B C}}{\partial \omega_ {v}} = \frac {\partial | | \hat {a} - a | | _ {2} ^ {2}}{\partial \hat {a}} \cdot \frac {\partial (\psi_ {s} (f _ {s}) + \psi_ {v} (f _ {v})) \cdot W _ {s h a r e} + b)}{\partial f _ {v}} \cdot \frac {\partial f _ {v}}{\partial \omega_ {v}}.\tag{3}
$$

Within the execution trajectory of the task, changes in visual cues are usually subtle compared to proprioceptive signals. For example, when the gripper is closing, visual cues differ only at pixel-level [13], while concise and lowdimension proprioceptive signals directly represent this process via changes in opening degree g. As a result, the vision-proprioception policy naturally gravitates toward proprioceptive signals to minimize the training loss. It leads to optimization dominated by proprioception and suppresses the learning of $\omega _ { v }$ due to vision modality’s low contribution to action prediction [14, 15].

As shown in Figure 1 (right), such overreliance to proprioception brings negligible impact during motion-consistent phases, since the execution of ongoing movements benefits significantly from proprioceptive signals. However, the initial positions of the target objects vary during testing and the proprioceptive signal does not contain object-related information. During motion-transition phases, the policy is required to accurately locate the target objects. The suppressed learning of vision modality thus regretfully impairs generalization of the vision-proprioception policy<sup>1</sup>.

## 4 Method

To alleviate the suppression of the learning of vision modality during motion-transition phases, we propose the Gradient Adjustment with Phase-guidance (GAP) algorithm. As shown in Figure 2, we initially define the representation of robot’s motion and identify motion-consistent phases. Motion-transition phase indicators are then predicted to estimate the probability that each timestep belongs to motion-transition phases. Based on these indicators, we apply fine-grained gradient adjustment during policy learning, facilitating dynamic collaboration within the vision-proprioception policy.

## 4.1 Motion Representation of Robot

Proprioceptive signals of the trajectory $[ s _ { 1 } , s _ { 2 } , . . . , s _ { N } ]$ directly provide the state of the gripper’s position $p ,$ orientation θ, and opening degree g. The variations in them effectively capture the motion of the robot arm over time. We first define the representation of motion for further motion-transition phase estimation. Specifically, the motion between timestep i and timestep j is defined as: $m _ { i : j } = \{ p _ { i : j } , \theta _ { i : j } , g _ { i : j } \}$ , where $p _ { i : j } = p _ { j } - p _ { i }$ denotes the change in the gripper’s 3D position, $\theta _ { i : j } = \theta _ { j } - \theta _ { i }$ denotes the change in orientation, and $g _ { i : j } = g _ { j } - g _ { i }$ denotes the change in gripper opening. Together, these three dimensions provide a complete representation of the robot’s motion.

## 4.2 Motion-Transition Phase Estimation

The represented motion captures the movement of robot arm, allowing expert demonstrations to be segmented into sequences of continuous states that correspond to semantically similar motions. To leverage this property for identifying motion-consistent phases, we employ the simple yet effective Change Point Detection (CPD) algorithm [28, 29]. The overall motion of a trajectory phase $\tau _ { t _ { 1 } : t _ { 2 } }$ can be characterized by $m _ { t _ { 1 } : t _ { 2 } }$ . Based on whether the directions of these changes are consistent, we define the following distance between phase motion $m _ { t _ { 1 } : t _ { 2 } }$ and adjacent motion $m _ { i : i + 1 } { : }$

$$
d (m _ {t _ {1}: t _ {2}}, m _ {i: i + 1}) = - \cos (p _ {t _ {1}: t _ {2}}, p _ {i: i + 1}) - \alpha \cos (\theta_ {t _ {1}: t _ {2}}, \theta_ {i: i + 1}) - \beta (\operatorname{sgn} (g _ {t _ {1}: t _ {2}}) = = \operatorname{sgn} (g _ {i: i + 1})),\tag{4}
$$

![](images/e0b2752ee5a9373b9ed806c4b6fc80ffd9831f2fb012585def9a6a36cf8ef2d6.jpg)  
Figure 2 The pipeline of our Gradient Adjustment with Phase-guidance (GAP) algorithm. We define the motion representation and identify the motion-consistent phases by minimizing the total cost between phase motion and each adjacent motion. Motion transition phase indicators are then estimated to reduce the magnitude of proprioception’s backward gradient. GAP facilitates vision-proprioception policies to effectively utilize proprioception without suppressing vision modality.

where $\operatorname { s g n } ( { \mathord { \cdot } } )$ denotes the sign function, and $\alpha , \beta$ are weighting factors for the orientation and opening degree component respectively. The statistic that measure the motion inconsistency of phase $\tau _ { t _ { 1 } : t _ { 2 } }$ is defined as $c _ { t _ { 1 } : t _ { 2 } } =$ $\begin{array} { r } { \sum _ { i = t _ { 1 } } ^ { t _ { 2 } - 1 } d ( m _ { t _ { 1 } : t _ { 2 } } , m _ { i : i + 1 } ) } \end{array}$ . The Change Point Detection algorithm leverages dynamic programming to identify a set of indices I that minimize the total cost $\sum _ { I } c _ { \tau _ { I } }$ , segmenting the trajectory into motion-consistent phases.

Motion of the robot transits within the intervals between these phases, requiring the policy to locate target object. Vision is therefore expected to play a more significant role. However, transition is a continuous process and the discrete labels from CPD can’t capture this continuous nature effectively, which is demonstrated by our ablation experiments in Section B.2 of the Appendix. Therefore, we further utilize the temporal differences of proprioceptive information $\Delta s _ { i } = s _ { i + 1 } - s _ { i }$ and leverage their sequential context with an temporal network such as LSTM. It predict motion-transition phase indicators $\rho _ { i }$ to estimate the probability that timestep i belongs to motion-transition phases. The predicted indicators $\rho$ is under the supervision of indices set I. Additionally, for timesteps within a range near the transition, we reduce the penalty applied to them in order to better capture the inherently continuous and gradual transition process.

## 4.3 Gradient Adjustment for Modality Collaboration

The vision-proprioception policy extracts features from both vision and proprioception modalities using two separate chunks $\phi _ { v } , \phi _ { s } ,$ , which consist of an encoder and a temporal transformer, these features are then fused and fed into policy head to predict the action. However, since visual cues during motion-transition phases may be subtle, the policy tends to rely heavily on features of proprioception. As a result, the gradient optimization for corresponding samples becomes dominated by proprioceptive inputs, which in turn constrains the learning of the vision modality chunk $\phi _ { v }$

To mitigate this, we employ gradient adjustment to control the optimization of proprioceptive chunk $\phi _ { s }$ during motiontransition phases, thereby guiding the vision-proprioception policy to focus more on visual cues and preventing the degradation of its generalization. Concretely, in the j-th epoch of Gradient Descent (GD)-based optimization, the parameters of the proprioceptive feature chunk $\omega _ { s } ^ { j }$ are updated according to the following formula:

$$
\omega_ {s} ^ {j + 1} = \omega_ {s} ^ {j} - \lambda \cdot (1 - \rho) \cdot \eta \nabla \omega_ {s} ^ {j} \mathcal {L} _ {B C} (\omega_ {s} ^ {j}),\tag{5}
$$

where η is the learning rate, λ is a hyper-parameter that controls the degree of adjustment. For each timestep, we modulate the magnitude of the proprioception backward gradient based on its indicator $\rho$ of belonging to motion-transition phases. The higher value of $\rho$ leads to greater degree of modulation.

By applying gradient adjustment with phase-guidance as illustrated in Algorithm 1, the vision-proprioception policy is enabled to effectively leverage proprioceptive information without compromising its generalization ability.

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Notations: Expert demonstrations  $o_{e}$ , proprioceptive signals  $s_{e}$ , epoch number T, vision-proprioception policy  $\pi_{v+s}$ , proprioception chunk parameters  $\omega_{s}$ , vision chunk parameters  $\omega_{v}$ .

Motion-Transition Phase Estimation
Identify motion-consistent phases by Change Point Detection  $I \leftarrow \text{CPD}(s_{e})$ ;
Predict motion-transition phase indicators  $\rho \leftarrow \text{LSTM}(\Delta s_{e})$ ;

Gradient Adjustment during Policy Learning
for  $j = 0, 1, \cdots, T - 1$  do
    Sample a fresh mini-batch  $B_{j}$  from expert demonstrations  $o_{e}$ ;
    Feed-forward the batched data  $B_{j}$  to  $\pi_{v+s}$ ;
    Calculate average indicator  $\rho_{j}$  of  $B_{j}$ ;
    Update proprioception chunk  $\omega_{s}^{j+1}$  using Equation 5;
    Update vision chunk  $\omega_{v}^{j+1}$ .

end for
</div>

## 5 Experiments

In this section, we validate the versatility and effectiveness of our proposed Gradient Adjustment with Phase-guidance (GAP) algorithm through a series of question-driven experiments. The evaluations comprehensively cover a wide range of manipulation tasks, including articulated object manipulation, rotation-sensitive tasks, as well as long-horizon and contact-rich tasks. Ablations on each modules and hyperparameters are presented in Section B due to space limitations.

## 5.1 Experimental Setup

We select two simulated environments as benchmarks: Meta-World [30] and RoboSuite [31]. Tasks in Meta-World are relatively simple, featuring a 4-dim action space that includes the gripper’s position and its opening degree, while tasks in RoboSuite involve complex scenarios, longer task sequence horizons and richer physical interactions, with the action space further including the orientation of the gripper. For real-world experiments shown in Figure 3, we use a 6-DoF xArm 6 robotic arm equipped with a Robotiq gripper. Moreover, we utilize the open-source Cobot Magic platform to support tasks that require dual-arm collaboration. In all tasks, the initial position of target object varies randomly in each validation, while the initial position of gripper remains fixed. Tasks in simulation and real-world are evaluated with 100 and 20 rollouts, respectively. The detailed task descriptions are provided in Section D of the Appendix.

## 5.2 Can GAP lead to more robust vision-proprioception policies?

Vision-Proprioception policies perform generally worse than vision-only policies. Can GAP lead to more robust vision-proprioception policies? To answer this, we conducted comparative analyses between our algorithm and the following baselines:

• MS-Bot [11]: this method uses state tokens with stage information to guide the dynamic collaboration of modalities within multi-modality policy.

• Auxiliary Loss (Aux): following HumanPlus [21], we use visual feature to predict the next frames as an auxiliary loss, which tries to enhance the vision modality.

• Mask: to prevents the overfitting to specific modality, RDT-1B [22] randomly and independently masks each uni modal input with a certain probability during encoding. We adapt the algorithm by masking only proprioception modality instead.

![](images/bdfa66c32e270e24cd3ca794eaa5cd60543c5f38bb78c74075ff694156937c8e.jpg)  
Figure 3 Visualization of real-world tasks. Our experiments cover a wide range of manipulation tasks, including both One-Arm and Dual-Arm Setups.

Results in Table 1 demonstrate that vision-proprioception policies with our GAP applied outperform vision-only policies and other methods. Although MS-Bot achieves overall improvements over the vision-only policy by incorporating stage information, it focuses on the semantic stage instead of motion-transition phases. As a result, its benefits are marginal in tasks like “push-wall” and “lift lid and pour”, where motion frequently transits. This highlights the necessity of fine-grained gradient adjustment during motion-transition phases. Auxiliary loss forces the vision-proprioception polic to concentrate on visual input during the whole task, which falls short in tasks requiring proprioception to enhance the precision and robustness of manipulation, such as “threading”. Meanwhile, masking the proprioceptive input with a fixed probability overlooks the modality temporality of manipulation tasks, resulting in minimal improvement. By adaptively applying fine-grained gradient adjustment during motion-transition phases, GAP enables the vision-proprioception policy to effectively leverage these two modalities and outperform both the vision-only policy and other methods.

Table 1 Comparisons with other methods in both simulated and real-world environments. Average success rate and standard deviation of simulation results are calculated over 5 seeds. The vision-proprioception policies after our gradient adjustment significantly outperform other methods.

<table><tr><td colspan="2">Suite</td><td colspan="5">Meta-World</td><td colspan="3">RoboSuite</td></tr><tr><td>Method\Task</td><td>pick-place</td><td>assembly</td><td>disassemble</td><td>push-wall</td><td>bin-picking</td><td>put hammer into drawer</td><td>stack</td><td>threading</td><td></td></tr><tr><td>Vision-only</td><td>91.8±1.5</td><td>82.6±3.0</td><td>84.4±2.2</td><td>63.2±1.9</td><td>63.2±1.5</td><td>85.8±1.3</td><td>65.8±2.4</td><td>43.6±1.7</td><td></td></tr><tr><td>Concatenation</td><td>78.4±1.5</td><td>74.6±2.1</td><td>79.6±1.1</td><td>54.4±2.9</td><td>48.8±2.2</td><td>78.6±1.1</td><td>54.6±2.3</td><td>33.2±1.9</td><td></td></tr><tr><td>MS-Bot [11]</td><td>90.8±1.6</td><td>91.0±2.1</td><td>87.8±1.9</td><td>66.2±1.6</td><td>69.4±1.9</td><td>87.6±1.1</td><td>69.4±1.9</td><td>51.2±1.3</td><td></td></tr><tr><td>Aux [21]</td><td>88.2±2.6</td><td>90.4±1.8</td><td>77.8±1.9</td><td>51.0±3.2</td><td>54.4±1.5</td><td>71.2±1.9</td><td>54.0±2.2</td><td>45.4±2.1</td><td></td></tr><tr><td>Mask [22]</td><td>86.4±1.8</td><td>89.2±1.6</td><td>83.8±2.7</td><td>79.2±3.7</td><td>60.4±1.1</td><td>78.6±1.8</td><td>61.8±1.9</td><td>47.2±1.6</td><td></td></tr><tr><td>GAP (Ours)</td><td>94.2±1.5</td><td>94.2±0.8</td><td>91.4±1.1</td><td>73.6±1.3</td><td>70.4±1.1</td><td>91.0±0.7</td><td>77.6±0.9</td><td>53.0±1.2</td><td></td></tr></table>

<table><tr><td colspan="2">Setup</td><td colspan="3">Real One-Arm</td><td colspan="3">Real Dual-Arm</td></tr><tr><td>Method\Task</td><td>press button</td><td>cube</td><td colspan="2">use rag to sweep table</td><td>handover</td><td>put thermos into bag</td><td>lift lid and pour</td></tr><tr><td>Vision-only</td><td>18/20</td><td>14/20</td><td colspan="2">9/20</td><td>15/20</td><td>11/20</td><td>9/20</td></tr><tr><td>Concatenation</td><td>12/20</td><td>11/20</td><td colspan="2">5/20</td><td>12/20</td><td>7/20</td><td>5/20</td></tr><tr><td>MS-Bot [11]</td><td>20/20</td><td>16/20</td><td colspan="2">11/20</td><td>16/20</td><td>13/20</td><td>10/20</td></tr><tr><td>Aux [21]</td><td>19/20</td><td>16/20</td><td colspan="2">11/20</td><td>15/20</td><td>13/20</td><td>8/20</td></tr><tr><td>Mask [22]</td><td>18/20</td><td>14/20</td><td colspan="2">7/20</td><td>15/20</td><td>9/20</td><td>7/20</td></tr><tr><td>GAP (Ours)</td><td>20/20</td><td>17/20</td><td colspan="2">13/20</td><td>18/20</td><td>16/20</td><td>15/20</td></tr></table>

![](images/b792f2f7ba59df1216fdb182df9e12119fa24921b8698744c2c7c657909f3b95.jpg)  
Figure 4 The intervention experiment of the GAP-equipped vision-proprioception policy. The sight changes in success rate indicate that GAP does enhance the utilization of vision modality.

## 5.3 Does GAP enhance the utilization of vision modality?

Although vision-proprioception policies outperform vision-only policies after apply GAP, it remains unclear whether GAP truly enhances the utilization of the vision modality within vision-proprioception policies. To answer this, we first conducted intervention experiment under the same settings as described in Section 1. As shown in the Figure 4, the degrees of suppression of vision modality during motion-transition phases are significantly reduced after applying GAP, indicating GAP does enhance the utilization of vision modality. We further evaluated the generalization of the vision-proprioception policies in out-of-distribution (OOD) scenarios. In each scenario, the initial distribution of object positions differs from that in the training dataset of expert demonstrations. The vision-only policies are less affected by such changes due to well-utilized vision modality as demonstrated in Tabel 2. Vision-proprioception policies exhibit poor generalization with suppressed vision. Meanwhile, Our algorithm alleviates this by regulating the optimization of the proprioceptive, preventing the suppression. The maintained superior performance over vision-only policy also indicates the effectiveness of introducing proprioception modality for precise and robust manipulation. The linear-probing experiments that directly evaluate the vision modality are provided in Section C.1.

Table 2 Experiments under out-of-distribution settings. For each task, our proposed GAP algorithm enhances the generalization of the vision-proprioception policies.

<table><tr><td>Setup</td><td colspan="2">Meta-World</td><td colspan="2">RoboSuite</td><td>Real One-Arm</td><td>Real Dual-Arm</td></tr><tr><td>Method\Task</td><td>assembly</td><td>bin-picking</td><td>stack</td><td>threading</td><td>cube</td><td>handover</td></tr><tr><td>Vision-only</td><td>78%</td><td>59%</td><td>63%</td><td>32%</td><td>12/20</td><td>12/20</td></tr><tr><td>Concatenation</td><td>62%</td><td>32%</td><td>49%</td><td>28%</td><td>7/20</td><td>9/20</td></tr><tr><td>GAP (Ours)</td><td>88%</td><td>67%</td><td>72%</td><td>49%</td><td>15/20</td><td>15/20</td></tr></table>

## 5.4 Is GAP compatible with Vision-Language-Action models?

Above experiments have demonstrated that GAP facilitates dynamic collaboration within conventional vision-proprioception models. We further investigate is GAP compatible with Vision-Language-Action (VLA) models. Specifically, we compare fine-tuned Octo model [9] using only visual information (Octo-V) versus using both vision and proprioception (Octo-VP), and tries to apply our gradient adjustment algorithm during fine-tuning. As reported in the original paper, policies trained with additional propioception seemed generally worse than vision-only policies. We observe the same trend across various tasks in Table 3. However, after applying our gradient adjustment algorithm, Octo-VP† achieves an

average improvement of 17% and exhibits stronger generalization ability than Octo-V. These results suggest that our algorithm effectively enhances dynamic collaboration between vision and proprioception within VLA models.

Table 3 Performances of fine-tuned Octo. † indicates GAP is applied.

<table><tr><td>Suite</td><td colspan="2">Meta-World</td><td colspan="2">RoboSuite</td></tr><tr><td>Model\Task</td><td>disassemble</td><td>push-wall</td><td>put hammer into drawer</td><td>threading</td></tr><tr><td>Octo-V</td><td>95%</td><td>77%</td><td>92%</td><td>69%</td></tr><tr><td>Octo-VP</td><td>82%</td><td>65%</td><td>88%</td><td>57%</td></tr><tr><td>Octo-VP†</td><td>100%</td><td>85%</td><td>97%</td><td>78%</td></tr></table>

## 5.5 Can GAP be applied to various modality fusion approaches?

The preliminary results in Section 3 reveal that the vision-proprioception policy using straightforward concatenation tends to perform worse than the vision-only policy. We further explore a broader set of fusion approaches and validate the versatility of our algorithm. Specifically, we apply GAP to three typical and widely used fusion approaches: Concatenation, Summation and FiLM [32].

As reported in Table 4, vision-only policies outperforms all three fusion approaches in tasks such as “pick-place” and “put hammer into drawer”, indicating that vision modality suffice for certain tasks. However, they fail drastically in task “threading” due to demands for precise manipulation and exhibits suboptimal performance in task “push-wall”, which involves visual occlusions at the target location, highlighting the necessity of the inclusion of proprioceptive information for precise and robust manipulation.

Table 4 Performance of typical fusion approaches combined with GAP. † indicates GAP is applied.

<table><tr><td colspan="2">Suite</td><td colspan="5">Meta-World</td><td colspan="3">RoboSuite</td></tr><tr><td>Method\Task</td><td>pick-place</td><td>assembly</td><td>disassemble</td><td>push-wall</td><td>bin-picking</td><td>put hammer into drawer</td><td>stack</td><td>threading</td><td></td></tr><tr><td>Vision-only</td><td>92%</td><td>82%</td><td>85%</td><td>64%</td><td>63%</td><td>86%</td><td>67%</td><td>44%</td><td></td></tr><tr><td>Concatenation</td><td>79%</td><td>76%</td><td>80%</td><td>56%</td><td>49%</td><td>79%</td><td>56%</td><td>34%</td><td></td></tr><tr><td>Summation</td><td>78%</td><td>95%</td><td>80%</td><td>54%</td><td>61%</td><td>75%</td><td>49%</td><td>30%</td><td></td></tr><tr><td>FiLM</td><td>75%</td><td>91%</td><td>47%</td><td>67%</td><td>59%</td><td>76%</td><td>53%</td><td>41%</td><td></td></tr><tr><td>Concatenation†</td><td>94%</td><td>96%</td><td>91%</td><td>73%</td><td>70%</td><td>91%</td><td>77%</td><td>52%</td><td></td></tr><tr><td>Summation†</td><td>92%</td><td>97%</td><td>93%</td><td>66%</td><td>70%</td><td>88%</td><td>82%</td><td>48%</td><td></td></tr><tr><td>FiLM†</td><td>90%</td><td>94%</td><td>85%</td><td>74%</td><td>68%</td><td>95%</td><td>72%</td><td>46%</td><td></td></tr></table>

Concatenation preserves raw features from both modalities, but the high-dimensional redundancy hinders the policy to dynamically utilize each modality. As a result, it underperforms in tasks like “push-wall”, where effective coordination is required. Simple summation may obscure critical details, whose limitation is evident in precise manipulation tasks such as “threading” and “push-wall”. Meanwhile, FiLM applies affine transformations to conditionally adjust features, making it more suitable for tasks requiring modality collaboration. For instance, it achieves a notably higher score in “push-wall” task. However, its performance tends to degrade in simpler tasks where such complex conditioning may be unnecessary. Conversely, GAP successfully unlocked the full potential of the vision-proprioception policy, outperforming vision-only policies in all three fusion approaches.

## 6 Visualization

## 6.1 Visualization of Motion-Transition Phase Estimation

To better understand the Motion-Transition Phase Estimation, we present visualizations that display the estimated transition probabilities ρ alongside corresponding RGB images of “threading” task in this section. We also show the relationship between $\rho$ and visual uncertainty. Specifically, we leverage R3M [33] to extract universal visual features of this task, and calculate the local entropy of features based on a sliding window covariance, denoted as u. The visual uncertainty u is then normalized to the range [0, 1] using $( \operatorname { t a n h } ( u ) + 1 ) / 2$ . As shown in Figure 5, the value of $\cdot _ { \rho }$ increases sharply at key moments, such as when the robotic arm grasps an object or transitions from a vertical to a rotational movement. This demonstrates that our method accurately captures motion transitions during the robot’s operation. Moreover, we observe that increases in $\rho$ are consistently accompanied by decreases in scaled visual uncertainty, indicating a clear inverse relationship between them. It implies that changes in the visual input tend to be relatively subtle during motion transitions, highlighting the importance of enhancing the learning of the visual modality.

![](images/bf2f4e34cbf20ed28cbe6e8248a3c05043a6456191c9ae4f6a52191b19d5bedc.jpg)  
Figure 5 Visualization of Motion-Transition Phase Estimation.

## 6.2 Visualization of Loss Curves

Furthermore, we demonstrate the effect of GAP on the loss curves of vision-proprioception policy in Figure 6. Due to the adjustment of GAP to the gradient updates for proprioceptive parameters, the loss decreases more slowly during the early and middle stages of training. However, the policy ultimately converges to a lower loss, indicating that GAP effectively enhances the performance of vision-proprioception policies.

![](images/c4f2946d3996f813929a2133c7f9d971a15e7125381327fee1da90642c1f40d2.jpg)  
Figure 6 Visualization of loss curves.

## 7 Conclusion and Limitation

In this work, we illustrate the vision modality of the vision-proprioception policy plays a limited role during motiontransition phases due to suppression. To alleviate this, we propose the Gradient Adjustment with Phase-guidance (GAP) algorithm, enabling dynamic collaboration between vision and proprioception. We believe this work can offer valuable insights into the development of vision-proprioception policies for robotic manipulation.

Limitations. All vision-proprioception policies are trained on single embodiment in this work. Exploring the role of proprioception in cross-embodiment datasets would be promising for future research.

## ACKNOWLEDGMENTS

This work was supported by Beijing Natural Science Foundation (4262050), in part by National Natural Science Foundation of China (62376274) and the fund for building world-class universities (disciplines) of Renmin University of China.

## References

[1] J. Allum, B. Bloem, M. Carpenter, M. Hulliger, and M. Hadders-Algra, “Proprioceptive control of posture: a review of new concepts,” Gait & posture, vol. 8, no. 3, pp. 214–242, 1998.

[2] B. Henze, C. Ott, and M. A. Roa, “Posture and balance control for humanoid robots in multi-contact scenarios based on model predictive control,” in Proceedings ofInternational Conference on Intelligent Robots and Systems (IROS), 2014, pp. 3253–3258.

[3] J. Lee, J. Hwangbo, L. Wellhausen, V. Koltun, and M. Hutter, “Learning quadrupedal locomotion over challenging terrain,” Science robotics, vol. 5, no. 47, p. eabc5986, 2020.

[4] Y. Yang, J. Norby, J. K. Yim, and A. M. Johnson, “Proprioception and tail control enable extreme terrain traversal by quadruped robots,” in Proceedings ofInternational Conference on Intelligent Robots and Systems (IROS), 2023, pp. 735–742.

[5] S. Levine, C. Finn, T. Darrell, and P. Abbeel, “End-to-end training of deep visuomotor policies,” Journal of Machine Learning Research, vol. 17, no. 39, pp. 1–40, 2016.

[6] L. Cong, H. Liang, P. Ruppel, Y. Shi, M. Görner, N. Hendrich, and J. Zhang, “Reinforcement learning with vision-proprioception model for robot planar pushing,” Frontiers in Neurorobotics, vol. 16, p. 829437, 2022.

[7] G. Jiang, Y. Sun, T. Huang, H. Li, Y. Liang, and H. Xu, “Robots pre-train robots: Manipulation-centric robotic representation from large-scale robot datasets,” in Proceedings ofInternational Conference on Learning Representations (ICLR), 2025.

[8] L. Wang, X. Chen, J. Zhao, and K. He, “Scaling proprioceptive-visual learning with heterogeneous pre-trained transformers,” Advances in Neural Information Processing Systems (NeurIPS), vol. 37, pp. 124 420–124 450, 2024.

[9] Octo Model Team, D. Ghosh, H. Walke, K. Pertsch, K. Black, O. Mees, S. Dasari, J. Hejna, C. Xu, J. Luo, T. Kreiman, Y. Tan, L. Y. Chen, P. Sanketi, Q. Vuong, T. Xiao, D. Sadigh, C. Finn, and S. Levine, “Octo: An open-source generalist robot policy,” in Proceedings ofRobotics: Science and Systems (RSS), Delft, Netherlands, 2024.

[10] F. R. Sarlegna and R. L. Sainburg, “The roles of vision and proprioception in the planning of reaching movements,” Progress in Motor Control: A Multidisciplinary Perspective, pp. 317–335, 2009.

[11] R. Feng, D. Hu, W. Ma, and X. Li, “Play to the score: Stage-guided dynamic multi-sensory fusion for robotic manipulation,” in Proceedings of Conference on Robot Learning (CoRL), 2024.

[12] Z. He, H. Fang, J. Chen, H.-S. Fang, and C. Lu, “Foar: Force-aware reactive policy for contact-rich robotic manipulation,” IEEE Robotics and Automation Letters, 2025.

[13] N. Tsagkas, A. Sochopoulos, D. Danier, S. Vijayakumar, C. X. Lu, and O. Mac Aodha, “When pre-trained visual representations fall short: Limitations in visuo-motor robot learning,” arXiv preprint arXiv:2502.03270, 2025.

[14] Y. Huang, J. Lin, C. Zhou, H. Yang, and L. Huang, “Modality competition: What makes joint training of multi-modal network fail in deep learning?(provably),” in Proceedings ofInternational Conference on Machine Learning (ICML). PMLR, 2022, pp. 9226–9259.

[15] Y. Fan, W. Xu, H. Wang, J. Wang, and S. Guo, “Pmr: Prototypical modal rebalance for multimodal learning,” in Proceedings of Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 20 029–20 038.

[16] B. Zitkovich, T. Yu, S. Xu, P. Xu, T. Xiao, F. Xia, J. Wu, P. Wohlhart, S. Welker, A. Wahid et al., “Rt-2: Vision-language-action models transfer web knowledge to robotic control,” in Proceedings ofConference on Robot Learning (CoRL). PMLR, 2023, pp. 2165–2183.

[17] M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. Foster, G. Lam, P. Sanketi et al., “Openvla: An open-source vision-language-action model,” in Proceedings ofConference on Robot Learning (CoRL), 2024.

[18] J. Zeng, Q. Bu, B. Wang, W. Xia, L. Chen, H. Dong, H. Song, D. Wang, D. Hu, P. Luo et al., “Learning manipulation by predicting interaction,” in Proceedings of Robotics: Science and Systems (RSS), 2024.

[19] A. Mandlekar, D. Xu, J. Wong, S. Nasiriany, C. Wang, R. Kulkarni, L. Fei-Fei, S. Savarese, Y. Zhu, and R. Martín-Martín, “What matters in learning from offline human demonstrations for robot manipulation,” in Proceedings ofConference on Robot Learning (CoRL), 2022, pp. 1678–1690.

[20] C. Chi, S. Feng, Y. Du, Z. Xu, E. Cousineau, B. Burchfiel, and S. Song, “Diffusion policy: Visuomotor policy learning via action diffusion,” in Proceedings ofRobotics: Science and Systems (RSS), 2023.

[21] Z. Fu, Q. Zhao, Q. Wu, G. Wetzstein, and C. Finn, “Humanplus: Humanoid shadowing and imitation from humans,” in Proceedings ofConference on Robot Learning (CoRL), 2024.

[22] S. Liu, L. Wu, B. Li, H. Tan, H. Chen, Z. Wang, K. Xu, H. Su, and J. Zhu, “Rdt-1b: a diffusion foundation model for bimanual manipulation,” arXiv preprint arXiv:2410.07864, 2024.

[23] W. Xia, R. Feng, D. Wang, and D. Hu, “Phoenix: A motion-based self-reflection framework for fine-grained robotic action correction,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2025, pp. 6981–6990.

[24] M. A. Lee, Y. Zhu, K. Srinivasan, P. Shah, S. Savarese, L. Fei-Fei, A. Garg, and J. Bohg, “Making sense of vision and touch: Self-supervised learning of multimodal representations for contact-rich tasks,” in Proceedings ofInternational Conference on Robotics and Automation (ICRA), 2019, pp. 8943–8950.

[25] J. Jiang, K. Ota, D. K. Jha, and A. Kanezaki, “Modality selection and skill segmentation via cross-modality attention,” https://arxiv.org/abs/2504.14573, 2025.

[26] H. Li, Y. Zhang, J. Zhu, S. Wang, M. A. Lee, H. Xu, E. Adelson, L. Fei-Fei, R. Gao, and J. Wu, “See, hear, and feel: Smart sensory fusion for robotic manipulation,” in Proceedings ofConference on Robot Learning (CoRL), 2023, pp. 1368–1378.

[27] F. Torabi, G. Warnell, and P. Stone, “Behavioral cloning from observation,” in Proceedings ofInternational Joint Conference on Artificial Intelligence (IJCAI), 2018, pp. 4950–4957.

[28] S. Liu, M. Yamada, N. Collier, and M. Sugiyama, “Change-point detection in time-series data by relative density-ratio estimation,” Neural Networks, vol. 43, pp. 72–83, 2013. [Online]. Available: https://www.sciencedirect.com/science/article/pii/ S0893608013000270

[29] S. Aminikhanghahi and D. J. Cook, “A survey of methods for time series change point detection,” Knowledge and Information Systems, vol. 51, no. 2, pp. 339–367, 2017.

[30] T. Yu, D. Quillen, Z. He, R. Julian, K. Hausman, C. Finn, and S. Levine, “Meta-world: A benchmark and evaluation for multi-task and meta reinforcement learning,” in Proceedings ofConference on Robot Learning (CoRL), 2020, pp. 1094–1100.

[31] Y. Zhu, J. Wong, A. Mandlekar, R. Martín-Martín, A. Joshi, S. Nasiriany, Y. Zhu, and K. Lin, “robosuite: A modular simulation framework and benchmark for robot learning,” in arXiv preprint arXiv:2009.12293, 2020.

[32] E. Perez, F. Strub, H. De Vries, V. Dumoulin, and A. Courville, “Film: Visual reasoning with a general conditioning layer,” in Proceedings ofthe AAAI conference on artificial intelligence, vol. 32, no. 1, 2018.

[33] S. Nair, A. Rajeswaran, V. Kumar, C. Finn, and A. Gupta, “R3m: A universal visual representation for robot manipulation,” in Proceedings ofConference on Robot Learning (CoRL). PMLR, 2023, pp. 892–909.

[34] A. Mandlekar, S. Nasiriany, B. Wen, I. Akinola, Y. Narang, L. Fan, Y. Zhu, and D. Fox, “Mimicgen: A data generation system for scalable robot learning using human demonstrations,” in Conference on Robot Learning (CoRL). PMLR, 2023, pp. 1820–1864.

[35] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proceedings ofConference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770–778.

[36] T. Zhao, V. Kumar, S. Levine, and C. Finn, “Learning fine-grained bimanual manipulation with low-cost hardware,” in Proceedings ofRobotics: Science and Systems (RSS), 2023.

[37] Q. Sun, P. Hong, T. D. Pala, V. Toh, U. Tan, D. Ghosal, S. Poria et al., “Emma-x: An embodied multimodal action model with grounded chain of thought and look-ahead spatial reasoning,” arXiv preprint arXiv:2412.11974, 2024.

[38] Z. Jia, V. Thumuluri, F. Liu, L. Chen, Z. Huang, and H. Su, “Chain-of-thought predictive control,” in International Conference on Machine Learning (ICML). PMLR, 2024, pp. 21 768–21 790.

[39] J. Lu, W. Xia, D. Wang, Z. Wang, B. Zhao, D. Hu, and X. Li, “Koi: Accelerating online imitation learning via hybrid key-state guidance,” in Proceedings ofConference on Robot Learning (CoRL). PMLR, 2025, pp. 3847–3865.

## Appendix

## A Implementation Details

In this section, we provide implementation details, including experimental settings and algorithmic specifics, to facilitate the reproduction of our work.

## A.1 Experiment Settings

Simulation experiments are conducted on two representative environments: Meta-World [30] and RoboSuite [31]. Tasks in Meta-World are relatively simple, featuring a 4-dimensional action space that includes the gripper’s position and its opening degree, while RoboSuite tasks involve complex scenarios, longer task sequence horizons and richer physica interactions, with the action space further including the orientation of the gripper. During policy learning, task in Meta-World uses 100 hard-coded expert demonstrations from its official open-source implementation, while task in RoboSuite leverages 500 synthesized trajectories that generated from human demonstrations [34].

We use an observation history window H of 5 with the learning rate $3 e ^ { - 4 }$ to train the vision-proprioception policy network. The parameters of the policy are optimized by Adam. The vision-proprioception policy consists of two feature extraction chunks and a policy head. For the visual input, spatial features are first extracted using a ResNet-18 [35] backbone, then projected into a 512-dimensional hidden representation. A 4-layer temporal transformer with a hidden dimension of 256 is applied to capture temporal features. For proprioceptive input, spatial features are extracted using a 3-layer MLP. These features are fused via concatenation and then passed through a 3-layer MLP to output a sequence of actions of length $L = 9$ . We train with batch size 128 on a single NVIDIA RTX 3090 GPU for 100 epochs (around 2 hours for 100 Meta-World demonstrations and 8 hours for 500 RoboSuite demonstrations).

For real-world experiments, we use a 6-DoF xArm 6 robotic arm equipped with a Robotiq gripper for all one-arm tasks. The visual information is provided by RGB images captured by an Intel RealSense D435i camera mounted on the wrist. Moreover, we utilize the open-source Cobot Magic platform to support tasks requiring dual-arm collaboration. It features four robot arms and three Intel RealSense D435 RGB-D cameras. Due to extended reach of the dual-arm setup, the two wrist-mounted cameras can’t fully capture the task scene. Therefore, we also incorporate the front camera on the platform to obtain a more comprehensive RGB observation of the task. Policies of real-world experiments are trained with 50 demonstrations collected by teleoperation. For one-arm tasks, the vision-proprioception policy is the same as in simulation. For dual-arm tasks, we modified the ACT [36] architecture. we train the left-arm policy using observations from the left-wrist camera, the front camera, and the left-arm proprioceptive information, and similar settings are applied to the right-arm policy.

In all tasks, the initial position of target object varies randomly in each validation, while the initial position of gripper remains fixed. In out-of-distribution (OOD) scenarios, the initial distribution of object positions differs from that in the training dataset of expert demonstrations.

## A.2 Details of GAP

In this study, we illustrate that vision-proprioception policy would fail during motion-transition phases due to its suppressed vision modality. To alleviate this, we propose the Gradient Adjustment with Phase-guidance (GAP) algorithm, enabling dynamic collaboration between vision and proprioception within vision-proprioception policy.

The complete list of hyperparameters used in Equations 4 and 5 is provided in Table 5. Since the motion inconsistency of the gripper’s opening state is measured using binary (0/1) values, we set β to a lower value to balance the numerical scale, while keeping $\alpha = 1$ to account for the overall gripper motion. To enable fine-grained gradient adjustment, we set the parameter λ controlling the degree of adjustment to a moderate level, avoiding excessive modulation leading to unstable gradient updates and potential policy collapse. Additionally, we apply gradient adjustment only during the early stage of policy learning, such as the first 50 epochs. Ablation studies on these hyperparameters are provided in Section B.3. For bimanual tasks, the proprioceptive information from each arm differs in physical meaning. The trajectories of the two arms thus have distinct motion transition phases. Therefore, we train policies for each arm independently. In transformer-based models, modalities become highly integrated within the transformer, we thus only applied gradient adjustment to the proprioception feature extractor (i.e., the parameters before the transformer) for Octo-VP.

Table 5 List of hyperparameters uesd in GAP.

<table><tr><td>Hyperparameters</td><td>Value</td></tr><tr><td> $\alpha$ </td><td>1</td></tr><tr><td> $\beta$ </td><td> $2e^{-3}$ </td></tr><tr><td> $\lambda$ </td><td>0.3</td></tr></table>

## B Ablation Experiments

## B.1 Ablation on Motion-Transition Phase Estimation

In our proposed GAP framework, the Motion-Transition Phase Estimation module is essential for enhancing vision proprioception policies. To thoroughly analyze its effectiveness and robustness, we conducted comprehensive ablation studies.

First, we kept other components unchanged and tested the impact of different trajectory decomposition methods on policy learning. The detailed descriptions of the comparison methods are as follows:

• Human: Since obtaining ground truth phase transition labels is infeasible due to the complexity and diversity of manipulation tasks, we instead employ human experts to manually label phase transitions.

• HDBSCAN: Following Emma-X [37] , we utilize the HDBSCAN algorithm to cluster our distance metric defined in Equation 4, decomposing trajectories into phases.

• CoTPC [38]: To decompose task trajectories into temporally close and functionally similar subskills, CoTPC applies the Change Point Detection (CPD) algorithm on the sequence of robot actions using Cosine distance.

As shown in Table $^ { 6 , }$ using human-labeled phase transitions for gradient adjustment does provide a slight improvement in performance. However, since human annotators tend to decompose trajectories from a semantic perspective rather than based on actual motion changes, this approach may miss certain transition points, resulting in only limited gains. The cluster-based HDBSCAN method disrupts the temporal structure inherent to manipulation tasks, leading to even worse policy performance compared to human labeling. Notably, both CoTPC and our GAP framework employ the CPD algorithm for phase estimation and generally outperform human annotation across most tasks, suggesting that automated phase estimation can capture motion transitions more effectively than manual labeling. However, CoTPC relies on a simple cosine distance, which lacks the ability to represent the nuanced characteristics of robot motion, thus yielding less significant performance improvements compared to GAP. In our approach, the motion-consistent distance defined in Equation 4 captures spatial position changes, rotational changes, and gripper opening state changes, enabling more accurate and efficient prediction of motion transitions.

Table 6 Performance Comparison between GAP and other decomposition methods.

<table><tr><td colspan="2">Suite</td><td colspan="5">Meta-World</td><td colspan="3">RoboSuite</td></tr><tr><td>Method\Task</td><td>pick-place</td><td>assembly</td><td>disassemble</td><td>push-wall</td><td>bin-picking</td><td>put hammer into drawer</td><td>stack</td><td>threading</td><td></td></tr><tr><td>Vision-only</td><td>92%</td><td>82%</td><td>85%</td><td>64%</td><td>63%</td><td>86%</td><td>67%</td><td>44%</td><td></td></tr><tr><td>Human</td><td>95%</td><td>83%</td><td>88%</td><td>68%</td><td>65%</td><td>88%</td><td>70%</td><td>47%</td><td></td></tr><tr><td>HDBSCAN [37]</td><td>88%</td><td>81%</td><td>87%</td><td>60%</td><td>57%</td><td>82%</td><td>61%</td><td>37%</td><td></td></tr><tr><td>CoTPC [38]</td><td>91%</td><td>85%</td><td>90%</td><td>66%</td><td>62%</td><td>85%</td><td>69%</td><td>50%</td><td></td></tr><tr><td>GAP (Ours)</td><td>94%</td><td>96%</td><td>91%</td><td>73%</td><td>70%</td><td>91%</td><td>77%</td><td>52%</td><td></td></tr></table>

To further understand the performance gap between GAP and human labeling, we calculated the precision and recall between GAP outputs and human labels. Our results show that GAP achieves a high recall (97% on average) but a relatively lower precision (73% on average), suggesting that the CPD algorithm not only accurately detects semantic changes during manipulation but also identifies motion transitions that may be overlooked by humans. This experiment highlights that our method offers an effective and efficient approach, given that obtaining ground truth phase transition labels is infeasible due to the complexity and diversity of manipulation tasks.

## B.2 Ablation on Temporal Network

We further validate the effectiveness of the LSTM component by conducting comprehensive ablation studies. Specifically, we compared the LSTM-based phase transition modeling with the following alternatives:

• Fixed $\rho \colon$ For the phase transition indices output by CPD, we applied a fixed gradient adjustment magnitude $\rho$ to the corresponding samples during policy learning.

• Smooth: Following KOI [39], we modeled the transition process by treating the CPD output index as the mean of a Gaussian distribution.

The results in Table 7 show that while alternative methods such as smoothing may serve as substitutes for modeling transitions, they regretfully fail to provide significant performance improvements.

Table 7 Ablation Studies on the LSTM component.

<table><tr><td>Method\Task</td><td>put hammer into drawer</td><td>threading</td><td>cube</td></tr><tr><td>Vision-only</td><td>86%</td><td>44%</td><td>70%</td></tr><tr><td>Fixed ρ=0.3</td><td>77%</td><td>44%</td><td>65%</td></tr><tr><td>Fixed ρ=0.5</td><td>89%</td><td>48%</td><td>75%</td></tr><tr><td>Fixed ρ=0.7</td><td>82%</td><td>38%</td><td>65%</td></tr><tr><td>Smooth</td><td>88%</td><td>46%</td><td>80%</td></tr><tr><td>GAP (Ours)</td><td>91%</td><td>52%</td><td>85%</td></tr></table>

Additionally, to demonstrate the LSTM‘s effectiveness in eliminating potential estimation errors, we conducted noise injection experiments by randomly selecting a direction and shifting each CPD output index with a probability of 0.5, repeating this shift process recursively. As shown in Table 8, even when considerable noise is injected into the CPD outputs, the policy performance remains robust. We attribute this robustness to the LSTM’s capability to model continuous transitions, which effectively alleviates the impact of errors introduced by the CPD algorithm.

Table 8 Noise Injection Experiments.

<table><tr><td>Method\Task</td><td>put hammer into drawer</td><td>threading</td><td>cube</td></tr><tr><td>Vision-only</td><td>86%</td><td>44%</td><td>70%</td></tr><tr><td>Concatenation</td><td>79%</td><td>34%</td><td>55%</td></tr><tr><td>Noise-Injected</td><td>88%</td><td>49%</td><td>85%</td></tr><tr><td>GAP (Ours)</td><td>91%</td><td>52%</td><td>85%</td></tr></table>

<table><tr><td>Method\Task</td><td>assembly</td><td>threading</td><td>put the banana in the box</td></tr><tr><td>Vision-only</td><td>83%</td><td>44%</td><td>55%</td></tr><tr><td>Concatenation</td><td>75%</td><td>33%</td><td>50%</td></tr><tr><td> $\alpha = 0.5, \beta = 2e^{-3}$ </td><td>93%</td><td>49%</td><td>65%</td></tr><tr><td> $\alpha = 2, \beta = 2e^{-3}$ </td><td>91%</td><td>57%</td><td>60%</td></tr><tr><td> $\alpha = 1, \beta = 10e^{-3}$ </td><td>88%</td><td>51%</td><td>60%</td></tr><tr><td>GAP ( $\alpha = 1, \beta = 2 \times 10^{-3}$ )</td><td>94%</td><td>53%</td><td>65%</td></tr></table>

## B.3 Ablations on Hyperparameters

To validate the effectiveness and robustness of our proposed GAP framework, we conduct comprehensive ablation studies on the key hyperparameters of the gradient adjustment.

First, we examine the selection of hyperparameters α and $\beta ,$ which serve to balance the relative contribution of each term in Equation 4, thus providing a more accurate measurement of whether the directions of these changes are consistent. As described in Section A.1, we set α = 1 so that displacement consistency and rotation consistency are equally weighted. Since gripper opening consistency is measured via sign function (which outputs 0 or 1), we set $\beta = 2 e ^ { - \mathrm { { \bar { 3 } } } }$ to ensure this term would not overshadow the first two terms. The results in Table 9 show that when α is small, displacement consistency contributes more to overall motion consistency, which leads to higher success rate of tasks where displacement dominates, such as "assembly" and "put the banana in the box". In contrast, when $\alpha = 2 .$ , the final policy performs better on the "threading" task, where rotation is more important. Increasing $\beta$ to $1 0 \times 1 0 ^ { - 3 }$ to emphasize the importance of gripper opening does not yield substantial improvement. Notably, all three sets of hyperparameter choices result in vision-proprioception policies that outperform policies without GAP, and are also superior to vision-only policies, indicating that our proposed GAP method is robust to hyperparameter selection.

We also perform ablation studies on the λ in Equation 5. In our GAP framework, we set $\lambda = 0 . 3$ by default to control the degree of gradient adjustment. As shown in Table 10, the policy performance is not highly sensitive to changes in λ between 0.2 and 0.4. However, setting λ too small results in insufficient attenuation of the proprioception gradients, causing the policy performance to become similar to or even worse than the simple concatenation baseline. Conversely, if λ is set too large, it leads to instability and even collapse during policy training.

Table 10 Ablation Studies on λ.

<table><tr><td>Method\Task</td><td>assembly</td><td>threading</td><td>put the banana in the box</td></tr><tr><td>Vision-only</td><td>83%</td><td>44%</td><td>55%</td></tr><tr><td>Concatenation</td><td>75%</td><td>33%</td><td>50%</td></tr><tr><td>λ = 0.1</td><td>77%</td><td>41%</td><td>50%</td></tr><tr><td>λ = 0.2</td><td>91%</td><td>57%</td><td>60%</td></tr><tr><td>λ = 0.4</td><td>96%</td><td>49%</td><td>65%</td></tr><tr><td>λ = 0.8</td><td>70%</td><td>37%</td><td>50%</td></tr><tr><td>GAP (λ = 0.3)</td><td>94%</td><td>53%</td><td>65%</td></tr></table>

The impact of stages of applying GAP has been evaluated as well. In all of our experiments, policies are trained for 100

epochs. We apply GAP during the first 50 epochs to avoid excessive modulation, which may cause unstable gradient updates and potentially lead to policy collapse. Table 11 presents the impact of the number of training stages x during which GAP is applied on the final policy performance. Applying GAP in the early stages of training improves the performance of vision-proprioception policies. However, if the gradient updates for proprioception are suppressed for too many epochs, the performance of policies deteriorates.

Table 11 Ablation Studies on the number of stages of applying GAP x.

<table><tr><td>Method\Task</td><td>assembly</td><td>threading</td><td>put the banana in the box</td></tr><tr><td>Vision-only</td><td>83%</td><td>44%</td><td>55%</td></tr><tr><td>Concatenation</td><td>75%</td><td>33%</td><td>50%</td></tr><tr><td>x = 10</td><td>77%</td><td>38%</td><td>50%</td></tr><tr><td>x = 30</td><td>88%</td><td>47%</td><td>60%</td></tr><tr><td>x = 70</td><td>83%</td><td>45%</td><td>55%</td></tr><tr><td>x = 90</td><td>72%</td><td>35%</td><td>50%</td></tr><tr><td>GAP (x = 50)</td><td>94%</td><td>53%</td><td>65%</td></tr></table>

## C Additional Experiments

## C.1 Linear-Probing Experiments on the Visual Features

To directly evaluate the visual features and verify whether our proposed GAP framework enhances the learning of visual representations, we conduct linear-probing experiments on the vision features extracted before and after applying GAP. Specifically, we extract and freeze the visual branch of the vision-proprioception policy and train a separate policy head to predict actions using only these frozen visual features. This linear-probing setup ensures that any performance differences directly reflect the quality of the learned visual representations.

These policies are evaluated in both in-distribution and out-of-distribution settings, where object positions at test time differ from those in the training data. As shown in Table 12, the visual branch trained with GAP significantly outperforms that without GAP in both in-distribution and out-of-distribution scenarios across different tasks.

Table 12 Linear-Probing Experiments on the Visual Features.

<table><tr><td>Method\Task</td><td>assembly</td><td>assembly (OOD)</td><td>threading</td><td>threading (OOD)</td></tr><tr><td>Visual branch without GAP</td><td>61%</td><td>45%</td><td>26%</td><td>14%</td></tr><tr><td>Visual branch with GAP</td><td>74%</td><td>69%</td><td>41%</td><td>27%</td></tr></table>

## C.2 Experiments with Various Policy Heads

The previous experiments have demonstrated the effectiveness of our GAP on policies with MLP-based policy heads. Additionally, we conducted experiments using the popular UNet-based diffusion policy head [20] and reported the results in Table 13. The results show that policies using a diffusion-based policy head are generally better those with MLP-based heads, while applying the GAP method still outperforms vision-only policies and other methods.

Table 13 Comparisons with other methods in simulated environments. Vision-Proprioception policies are equipped with UNet-based diffusion policy head.

<table><tr><td colspan="2">Suite</td><td colspan="3">Meta-World</td><td colspan="2">RoboSuite</td></tr><tr><td>Method\Task</td><td>disassemble</td><td>push-wall</td><td>bin-picking</td><td>stack</td><td>threading</td><td></td></tr><tr><td>Vision-only</td><td>92%</td><td>70%</td><td>66%</td><td>73%</td><td>51%</td><td></td></tr><tr><td>Concatenation</td><td>82%</td><td>62%</td><td>51%</td><td>62%</td><td>40%</td><td></td></tr><tr><td>MS-Bot [11]</td><td>93%</td><td>75%</td><td>72%</td><td>77%</td><td>58%</td><td></td></tr><tr><td>Aux [21]</td><td>93%</td><td>71%</td><td>75%</td><td>72%</td><td>55%</td><td></td></tr><tr><td>Mask [22]</td><td>87%</td><td>83%</td><td>67%</td><td>66%</td><td>53%</td><td></td></tr><tr><td>GAP (Ours)</td><td>97%</td><td>88%</td><td>81%</td><td>87%</td><td>63%</td><td></td></tr></table>

## C.3 More Out-Of-Distribution (OOD) Evaluations

To further assess the robustness of GAP under diverse visual distribution shifts, we conducted additional out-ofdistribution experiments beyond the object placement variations reported in Table 2. Specifically, we evaluated three additional types of visual perturbations:

• Table Color: The color of the table was modified during testing, altering the overall scene appearance.

• Object Color: Objects with different colors from those used during training were employed, introducing colorbased perturbations.

• Lighting: Colored spot lighting was introduced to create additional visual disturbances during testing.

The results in Table 14 and Table 15 show that GAP consistently outperforms baseline methods under all tested OOD conditions. Among different perturbations, placement changes exert the most significant impact on performance, underscoring the importance of accurate target object localization for successful manipulation. While concatenation benefits from proprioceptive information that remains unaffected by visual disturbances, the lack of precise position perception results in overall suboptimal performance.

In contrast, GAP achieves superior performance compared to vision-only approaches, highlighting that leveraging proprioceptive information in collaboration with vision enhances policy robustness under challenging OOD scenarios.

Table 14 OOD evaluation of the task “assembly” in Meta-World.

<table><tr><td>Method\Task</td><td>Placement</td><td>Table Color</td><td>Object Color</td><td>Lighting</td></tr><tr><td>Vision-only</td><td>78%</td><td>80%</td><td>75%</td><td>73%</td></tr><tr><td>Concatenation</td><td>62%</td><td>72%</td><td>76%</td><td>74%</td></tr><tr><td>GAP (Ours)</td><td>88%</td><td>85%</td><td>83%</td><td>90%</td></tr></table>

Table 15 OOD evaluation of the task “cube” in the Real-World.

<table><tr><td>Method\Task</td><td>Placement</td><td>Table Color</td><td>Object Color</td><td>Lighting</td></tr><tr><td>Vision-only</td><td>60%</td><td>65%</td><td>60%</td><td>55%</td></tr><tr><td>Concatenation</td><td>35%</td><td>50%</td><td>55%</td><td>50%</td></tr><tr><td>GAP (Ours)</td><td>75%</td><td>80%</td><td>85%</td><td>80%</td></tr></table>

## C.4 Experiments of extending GAP to on-policy updates

To verify whether GAP can be effectively applied to on-policy updates, we conduct experiments using online learning settings in this section. In these experiments, we use vision-proprioception policies initialized with behavior cloning and train the LSTM to predict motion transition probabilities in real-time during exploration, thereby adjusting proprioception gradient updates dynamically in online learning. This setup allows us to assess whether GAP maintains its effectiveness when the policy is updated based on newly collected experience, rather than a fixed offline dataset.

Table 16 Experiments with On-Policy Updates.

<table><tr><td>Method\Task</td><td>assembly</td><td>bin-picking</td><td>threading</td></tr><tr><td>Vision-only</td><td>88%</td><td>71%</td><td>50%</td></tr><tr><td>Concatenation</td><td>78%</td><td>50%</td><td>37%</td></tr><tr><td>GAP</td><td>94%</td><td>82%</td><td>61%</td></tr></table>

The results in Table 16 demonstrate that policies trained with online learning are more robust, and GAP can be effectively applied to on-policy updates across different tasks and environments. Additionally, the online learning experiments demonstrate that GAP does not over-dampen necessary proprioception reliance, as the policies maintain strong performance while still benefiting from the enhanced visual representations.

## D Task Descriptions

In this subsection, we provide detailed descriptions of tasks we evaluated and visualize additionally simulation tasks in Figure 7. The evaluations comprehensively cover a wide range of manipulation tasks, including simple pick-and-place tasks, rotation-sensitive tasks, as well as long-horizon and contact-rich tasks and includes one-arm and dual-arm robot setups.

• pick-place: The robot arm first needs to locate the position of the cylinder, then move toward it. After grasping the object, it locates the target circle and moves the object to the target position.

• assembly: The robot arm first locates the position of the part, then moves toward it. After grasping the part, it locates the position of the base, moves the part above it, and finally assembles them.

• disassemble: The robot arm first locates the position of the part, then moves toward it. After grasping the part, it disassembles the part from the base.

• push-wall: The robot arm first locates the position of the cylinder, then moves toward it. After grasping it, the arm pushes it to a target point behind the wall. The target point is not visible through visual observation, but it location remains fixed across all evaluations.

• bin-picking: The robot arm first locates the position of the green cube, then moves toward it. After grasping it, the arm removes it from the red container and finally places it into the blue container.

• put hammer into drawer: The robot arm first locates the position of the drawer and then moves toward it. After opening the drawer, it locates the position of the hammer, grasps it, places it inside the drawer, and finally closes the drawer.

• stack: The robot arm first locates the position of the red cube and then moves toward it. After grasping it, it locates the position of the green cube and finally stacks the red cube on top of the green cube.

• threading: The robot arm first locates the position of the pin and then moves toward it. After grasping it, it locates the position of the pinhole, then moves and rotates to align the pin with the hole, and finally inserts the pin into the hole.

• press button: The robot arm first locates the position of the button, then moves toward it, closes the gripper, and presses the button.

• cube: The robot arm first locates the position of the cube, then moves toward it. After grasping the cube, it locates the drawer and finally places the cube in the drawer.

• use rag to sweep table: The robot arm first locates the position of the rag, then moves toward the rag. It rotates the arm to grasp the rag, takes it down from the shelf, and finally rotates the rag to sweep the table.

• handover: The robot first locates the position of the rectangular block, then its right arm moves toward it. After grasping the block, the right arm moves it to the center, where the left arm takes over the block in a handover. Finally, the left arm places it on the left side of the table.

• put thermos into bag: The robot first locates the position of the bag, then its right arm moves toward it, lifts it, and rotates the wrist to open the bag. Next, robot locates the thermos, uses the left arm to grasp it, and puts it into the bag.

• lift lid and pour: The robot first locates the position of the cup, then its left arm moves toward it to grasp and lift the lid. Next, robot locates the graduated cylinder, uses the right arm to grasp it, moves it above the cup, and finally rotates the cylinder to pour.

![](images/57062c3cdcbd7e4964fff29f6765a066803f54bfa6c78c628fb17d0b74ae8c5e.jpg)  
Figure 7 Visualization of tasks we evaluated in simulation environments.