# FreqPolicy: Efficient Flow-based Visuomotor Policy via Frequency Consistency

Yifei Su<sup>1,2∗</sup>, Ning Liu<sup>1∗†</sup> , Dong Chen<sup>1</sup>, Zhen Zhao<sup>1</sup>, Kun Wu<sup>1</sup>, Meng Li<sup>1</sup>, Zhiyuan Xu<sup>1</sup>, Zhengping Che<sup>1†‡</sup>, Jian Tang<sup>1‡</sup> <sup>1</sup>Beijing Innovation Center of Humanoid Robotics <sup>2</sup>NLPR, MAIS, Institute of Automation of Chinese Academy of Sciences

## Abstract

Generative modeling-based visuomotor policies have been widely adopted in robotic manipulation, attributed to their ability to model multimodal action distributions. However, the high inference cost of multi-step sampling limits its applicability in real-time robotic systems. Existing approaches accelerate sampling in generative modeling-based visuomotor policies by adapting techniques originally developed to speed up image generation. However, a major distinction exists: image generation typically produces independent samples without temporal dependencies, while robotic manipulation requires generating action trajectories with continuity and temporal coherence. To this end, we propose FreqPolicy, a novel approach that first imposes frequency consistency constraints on flow-based visuomotor policies. Our work enables the action model to capture temporal structure effectively while supporting efficient, high-quality one-step action generation. Concretely, we introduce a frequency consistency constraint objective that enforces alignment of frequency-domain action features across different timesteps along the flow, thereby promoting convergence of one-step action generation toward the target distribution. In addition, we design an adaptive consistency loss to capture structural temporal variations inherent in robotic manipulation tasks. We assess FreqPolicy on 53 tasks across 3 simulation benchmarks, proving its superiority over existing one-step action generators. We further integrate FreqPolicy into the vision-language-action (VLA) model and achieve acceleration without performance degradation on 40 tasks of Libero. Besides, we show efficiency and effectiveness in real-world robotic scenarios with an inference frequency of 93.5 Hz.

## 1 Introduction

Recently, the generative modeling-based visuomotor policy framework extends the powerful capabilities of generative models in text-to-image synthesis [58, 17, 56, 60, 9, 21] to imitation learning for robotic manipulation [27, 11, 39, 19, 6], achieving significant breakthroughs. A prominent class of such policies is diffusion-based approaches [11, 76, 46, 75, 83, 23], which are widely adopted for the ability to model complex multimodal distributions in high-dimensional action spaces [11]. More recently, flow matching [40, 38, 13, 36] has emerged as a generalization of diffusion models, offering simpler optimization objectives and more stable training [68]. It has been applied to robotics [59, 19, 12, 77, 6, 25], proving the efficacy of flow-based policies. Despite these advancements, these generative modeling-based visuomotor policies rely on an iterative sampling process to transform Gaussian noise into actions, resulting in high-latency inference. This presents a significant bottleneck for real-time inference in robotic manipulation, where stable and smooth action execution is essential, particularly in long-horizon or dynamic tasks [53, 42].

To mitigate this issue, recent efforts leverage acceleration methods from the image generation domain to accelerate the action generation by reducing the sampling steps [53, 42, 28, 65, 16]. A pioneering work, Consistency Policy (CP) [53], adapts the consistency distillation from the text-to-image domain [62, 31, 43, 64, 41] to the robotics domain. CP first trains a powerful diffusion-based teacher using the EDM model [29], and subsequently distills its knowledge into a student model via the Consistency Trajectory Model objective [31], enabling the diffusion-based policy one-step action generation. ManiCM [42] extends the consistency distillation to 3D diffusion-based policy [76] with point cloud inputs. SDM [28] and OneDP [65] adopt distribution matching distillation originally introduced in image generation [52, 66, 71, 14], distilling pretrained diffusion-based policies into one-step action generators by aligning the action distributions. While the aforementioned methods primarily focus on diffusion-based policies, FlowPolicy [78] attempts to reduce the sampling step of flow-based policy. It uses consistent constraints from Consistency-FM [68] originally developed for image generation, to enforce velocity consistency across different points along the flow without distillation, facilitating straight flows and enabling one-step generation.

In fact, a key distinction between image generation and robotic manipulation lies in the presence of temporal dependencies. While image generation produces individual samples that are typically independent across time, robotic manipulation involves executing action trajectories that form a timeseries process. Therefore, it requires modeling outputs that are continuous and temporally coherent. We argue that leveraging temporal characteristics is essential, as it provides richer contextual information that can significantly enhance action generation. In this work, we proposed a novel one-step visuomotor policy, named FreqPolicy, that first imposes frequency consistency constraints on flow-based visuomotor policy, thus leveraging temporal knowledge and enabling efficient and high-quality one-step action generation. Specifically, we introduce a frequency consistency objective that enforces alignment of frequency-domain action features across timesteps, improving temporal coherence in generated actions. Furthermore, we design an adaptive frequency component loss to learn the structural temporal variations inherent in robotic manipulation tasks.

Frequency Consistency Objective. Inspired by the fields of time-series forecasting and speech processing [82, 81, 70, 73], frequency representations capture better non-stationary and oscillatory patterns, where time-domain features often fail. Thus, frequency-domain features show high effectiveness for modeling temporal dynamics. In robotic manipulation, for high-frequency sampled action chunks, frequency features provide finer discrimination of subtle variations in smooth trajectories. To exploit this advantage, we enforce consistency between velocities of action chunks across different timesteps in the frequency space, thereby promoting the convergence of action generation.

Adaptive Frequency Component Loss. In robotic manipulation, the distribution of frequency components within each action chunk varies over time throughout task execution. This variability arises because robotic manipulation sequences typically alternate between stationary and non-stationary motion phases. During low-dynamic movements (e.g., reaching or moving between positions), only a subset of action dimensions exhibit noticeable variation, while others remain relatively smooth.Conversely, during high-dynamic movements (e.g., transitions between skills or contact-rich interactions), high-frequency variations are more prominent and informative. To capture this structured temporal variation, we draw inspiration from the Focal Loss [35] and propose an adaptive weighting scheme that dynamically emphasizes frequency components with greater discrepancy.

Our contributions can be summarized as: 1) FreqPolicy is the first one-step visuomotor policy that imposes temporal knowledge for robotic manipulation. 2) Inspired by the time-series and speech processing field, we propose the frequency consistency constraint objective to enhance the regularization of two arbitrary action velocities. Moreover, an adaptive frequency component loss is proposed to capture the structured temporal variation of the action sequence. 3) We conduct extensive experiments in both simulation and the real world to evaluate FreqPolicy, demonstrating its superiority over existing one-step action generators, e.g., achieving 78.5% in MetaWorld. 4) We further integrate FreqPolicy into a vision-language-action (VLA) model [6, 32], achieving a significant improvement in inference speed (e.g., 5× faster) without compromising overall task performance.

## 2 Related Work

Generative Modeling-based Visuomotor Policy. Learning from human demonstrations (a.k.a. imitation learning [2, 51, 20, 1]) has shown remarkable performance in robotic manipulation [8, 7, 34, 33]. Compared to deterministic policies [50, 79, 26, 80, 4, 18, 54], generative modeling-based policies are widely used for their ability to model multimodal action distributions, enhancing training stability in high-dimensional action spaces [11]. The pioneer work, Diffusion Policy [11], formulated visuomotor policy learning as a conditional denoising diffusion process, achieving impressive manipulation performance through a multi-step sampling inference process. Subsequent works [76, 30] extended diffusion-based policies to 3D point cloud inputs, enabling action generation directly from point cloud observations. [46, 67] factorizes the policy into a high-level key waypoint predictor and low-level trajectory generator, achieving higher task success rates. Recently, another generative modeling paradigm, Flow Matching (FM) [40, 38] has shown strong performance in domains including image generation [13, 45, 17] and super-resolution [84]. Compared to diffusion-based generative models, flow matching (FM) directly defines probability paths via ordinary differential equations (ODEs) to transport the simple prior distribution to a target distribution, offering better numerical stability and fewer inference steps [36]. Hence, it has been applied to robotic manipulation [6, 25, 61, 5]. As a pioneer, Rouxel et al. [59] extend flow matching to multi-support tasks and demonstrated strong performance. Subsequent works [19, 12] incorporate point cloud inputs, achieving manipulation in 3D scenery. Zhang et al. [77] innovatively integrate spatial affordance prediction with flow matching for action prediction.

Accelerated Visuomotor Policy. Despite the impressive performance of generative-based policies, they are limited in real-world deployment by the high inference cost of multi-step sampling [53]. To address this issue, recent efforts have accelerated action sampling through four main mainstream as: (1) Trajectory distillation. Consistency Policy [53] pioneered consistency distillation [62, 43, 31] in robot manipulation, enabling a few-step action generation by constraining denoising trajectories from different steps toward the same step. ManiCM [42] extended this approach to 3D scenarios and achieved better one-step inference than 3D Diffusion Policy [76]. FlowPolicy [78], inspired by ConsistencyFM [68], applied constraints on the velocity field and transporting process in flow matching, enabling one-step action generation. (2) Partial action denoising. Instead of denoising from Gaussian noise per step, SDP [22] outputs a partially denoised action with variable levels of noise, where the noise-free part guides current action execution and the noisy portion serves as input for future rollouts. Falcon [10] proposed to select an action chunk from the historical denoised trajectory as the input. Both enhance the inference speed but still require multiple sampling steps. (3) Distribution matching. OneDP [65] and SDM [28] incorporated variational score distillation [66, 44, 72, 71, 48, 14] into imitation learning and achieved strong one-step generation performance in both 2D [47] and 3D [74] input settings. Although effective, such approaches typically require a pretrained teacher model. In addition to the above methods, some works have introduced advanced Riemannian flow matching [15] and variance-based adaptive sampling strategies [24] to achieve faster generation, respectively. This work aims to accelerate flow-based visuomotor policies using consistency-based paradigms, motivated by their potential to eliminate the need for teacher models. Unlike existing works, we propose a novel perspective based on frequency consistency to enable more effective one-step action generation by leveraging temporal knowledge in robotic manipulation.

## 3 Methodology

## 3.1 Task Setting

We focus on the robotic manipulation task via the imitation learning paradigm: given a dataset D containing n observations $\boldsymbol { \dot { \mathcal { O } } } = \left\{ o \right\} _ { i = 1 } ^ { n }$ and corresponding expert action $\bar { \mathcal { A } } = \{ \breve { a } \} _ { i = 1 } ^ { n }$ , the goal is to train a flow matching-based policy ${ \overline { { \pi } } } _ { \theta } ( a _ { 1 : H } | o ) : \mathcal { O } \to \mathcal { A }$ to map the observations $o \in { \mathcal { O } }$ to action $a _ { 1 : H }$ with chunking size H, enabling the robot to replicate expert behaviors and generalize across diverse scenarios. Observations typically include proprioceptive states, RGB images from various viewpoints $( \mathrm { e . g . }$ , eye-in-hand or third-person cameras), and point cloud data. The action space varies depending on the task and robot setup, and commonly includes either SE(3) motions of the end-effector or joint states. To thoroughly evaluate the effectiveness of the proposed method, we conduct extensive experiments under both 2D and 3D observation settings.

![](images/099c7013a9d684871cccbc439180214df6cbffb988955f9eae213cf38edc5d6c.jpg)  
Figure 1: Overview of FreqPolicy. a) For training, we use the frequency consistency constraint to align the velocity vectors across different time steps in the frequency space. Besides, we introduce an adaptive frequency component loss to accommodate the diverse frequency structures in manipulation tasks. b) FreqPolicy takes 2D or 3D input and predicts the velocity vector of the action as output.

## 3.2 Preliminaries

Flow Matching. Let $\mathcal { R } ^ { d }$ denote the data space with data points $\boldsymbol { x } \in \mathcal { R } ^ { d } .$ , and consider a simple prior distribution $x _ { 0 } \sim p _ { 0 } ( x )$ , FM aims to learn a time-dependent vector field $v ( t , x ) : [ 0 , 1 ] \times \mathcal { R } ^ { \hat { d } }  \mathcal { R } ^ { d }$ that generates a time-conditioned ordinary differential equation (ODE) as follows:

$$
\frac {d \psi (t , x)}{d t} = v (t, \psi (t, x)), \psi (0, x) \sim p _ {0}\tag{1}
$$

$\psi ( \cdot ) : [ 0 , 1 ] \times \mathcal { R } ^ { d } \to \mathcal { R } ^ { d }$ is the solution to the ODE, also known as the flow. Based on this, the distribution $p _ { 0 }$ can be transported to a more complex distribution $p _ { t }$ via the push-forward equation:

$$
p _ {t} = p _ {0} \left(\psi^ {- 1} (t, x)\right) \det \left[ \frac {\partial}{\partial x} \psi^ {- 1} (t, x) \right]\tag{2}
$$

To transform the prior distribution $p _ { 0 }$ into an unknown target distribution $p _ { 1 }$ via flow $\psi ,$ , with the marginal distribution satisfying $p _ { t = 0 } = p _ { 0 }$ and $p _ { t = 1 } = p _ { 1 }$ , flow matching seeks to optimize the vector field $v ( t , x )$ by minimizing the following objective:

$$
\mathcal {L} _ {\theta} = \mathbb {E} _ {t, p _ {t}} \left\| v _ {\theta} (t, x _ {t}) - u (t, x _ {t}) \right\| _ {2}\tag{3}
$$

where θ denotes the learnable parameters of the vector field, which in turn leads to a deep parametric model of the flow $\psi . \boldsymbol { u } ( t , \boldsymbol { x } _ { t } )$ ) is the corresponding target vector field. Despite the simplicity of the objective, directly optimizing it in practice is challenging due to the lack of prior knowledge of the desired $p _ { t }$ and $u ( t , x _ { t } )$ . Therefore, Conditional Flow Matching [36] proposes to regress $v _ { \theta } ( t , x _ { t } )$ via a conditional vector field $u ( t , x _ { t } | x _ { 1 } )$ and a conditional probability path $p _ { t } ( x _ { t } | x _ { 1 } )$ as:

$$
\mathcal {L} _ {\theta} = \mathbb {E} _ {t, p _ {1} (x _ {1}), p _ {t} (x _ {t} | x _ {1})} \left\| v _ {\theta} (t, x _ {t}) - u (t, x _ {t} | x _ {1}) \right\| _ {2}\tag{4}
$$

Here Eq. 3 and $\operatorname { E q . }$ 4 share the same gradient with respect to $\theta ,$ thus Eq. 3 can be estimated if $u ( t , x _ { t } | x _ { 1 } )$ and $p ( t , x _ { t } | x _ { 1 } )$ is tractable. The choice of $u ( t , x _ { t } | x _ { 1 } )$ is not unique, and a classic instance is obtained by linearly interpolating between $x _ { t }$ and $x _ { 1 }$ [40, 38], which owns the desirable property of generating straight probability paths:

$$
u (t, x _ {t} | x _ {1}) = \frac {x _ {1} - x _ {t}}{1 - t}\tag{5}
$$

$$
\mathcal {L} _ {\theta} = \mathbb {E} _ {t, p _ {1} (x _ {1}), p _ {t} (x _ {t})} \left\| v _ {\theta} (t, \psi (t, x _ {t})) - (x _ {1} - x _ {t}) \right\| _ {2}\tag{6}
$$

## 3.3 FreqPolicy

Flow-based policies suffer from slow inference due to their iterative sampling process, limiting their use in high-frequency control scenarios and deployment on edge devices. We propose FreqPolicy, a novel approach that enables efficient one-step action generation by introducing temporal constraints from the frequency perspective, as shown in Fig. 1. Inspired by the time-series and speech processing fields, we propose the frequency consistency constraint objective to enhance the regularization of two arbitrary action velocities. Besides, the adaptive frequency component loss is proposed to effectively capture the structured temporal variation of the action sequence. We provide details on the model architecture in Sec. A.1.

Basic Flow Matching Learning Objective. An intuitive way to apply flow matching in robotic manipulation is to regress a vector field mapping noise to action chunks, as adopted in prior works [19, 59, 12, 24, 6]. We first equip FreqPolicy with this objective to enable multimodal action generation.

Specifically, considering a simple noise $a _ { 0 } \sim \mathcal { N } ( 0 , I )$ and a corresponding expert action $a _ { 1 } \sim p _ { \mathrm { e x p e r t } } .$ we aim to learn the vector field $v _ { \theta } ( t , a _ { t } )$ that maps the prior noise to expert actions by approximating the real conditional velocity field $u ( t , a _ { t } )$ . Following flow matching based on optimal transport theory [40, 38], we instantiate the target velocity field $u ( t , a _ { t } )$ as a constant vector along the optimal linear transport from $a _ { 0 }$ to $a _ { 1 }$ , namely:

$$
u (t, a _ {t}) = \int_ {0} ^ {1} u (z, a _ {z}) d z = a _ {1} - a _ {0}\tag{7}
$$

Next, for any timestep $t \in [ 0 , 1 ]$ , we optimize the policy by minimizing the following objective:

$$
\mathcal {L} _ {\mathrm{fm}} = \mathbb {E} _ {t \sim \mathcal {U} (0, 1), (a _ {0}, a _ {1}) \sim \mathcal {D}} \| v _ {\theta} (t, a _ {t}) - (a _ {1} - a _ {0}) \| _ {2}\tag{8}
$$

where $a _ { t }$ is defined as the linear interpolation between $a _ { 0 }$ and $a _ { 1 }$ with respect to time $t , a _ { t } \ =$ $( 1 - t ) \cdot a _ { 0 } + t \cdot a _ { 1 }$ . For inference, we sample an initial noise vector $a _ { 0 }$ and deterministically integrate the learned vector field $v _ { \theta } ( t , a _ { t } )$ from $t = 0 \mathrm { t o } t = 1$ using an ODE solver (e.g., Euler or Runge-Kutta), yielding the final action $a _ { 1 }$ . While optimizing the objective in $\mathtt { E q }$ . 8 yields a functional policy model, it still requires multiple sampling steps to generate a desired action (e.g., 10 steps in $\pi _ { 0 } \left[ 6 \right]$ and 4 steps in GR00T N1 [5]).

Frequency Consistency Constraint Objective. In the fields of time-series analysis and speech processing [82, 81, 70, 73], frequency-domain features show superior ability to model temporal dynamics, e.g., non-stationary [69] and oscillatory patterns, where time-domain signals often fall short. In robotic manipulation, frequency features provide finer discrimination of subtle variations in high-frequency sampled action chunks [49]. Therefore, differing from prior approaches [78] that treat multi-dimensional action chunks as static vectors, the proposed FreqPolicy regards each action chunk as a temporal signal. We aim to enforce consistency in the frequency space between action velocity vectors across different timesteps of the flow, promoting straighter flow and one-step action generation. Concretely, given a pair of initial noise and target actions $( a _ { 0 } , a _ { 1 } )$ sampled from the prior and expert distributions, we select two arbitrary timesteps $s , r \in [ 0 , 1 ]$ ], and then construct the interpolated state between $a _ { 0 }$ and $a _ { 1 }$ as:

$$
a _ {r} = (1 - r) \cdot a _ {0} + r \cdot a _ {1}, \quad a _ {s} = (1 - s) \cdot a _ {0} + s \cdot a _ {1}\tag{9}
$$

We enforce the velocities $v _ { \theta } ( s , a _ { s } )$ and $v _ { \theta } ( r , a _ { r } )$ to be consistent across timesteps through the following training objectives:

$$
\begin{array}{c} \mathcal {L} _ {\mathrm{freq}} = \mathbb {E} _ {r, s \sim \mathcal {U} (0, 1), (a _ {r}, a _ {s}) \sim \mathcal {D}} \left[ \mathrm{Sim} (v _ {\theta} (s, a _ {s}), v _ {\theta} (r, a _ {r})) \right] + \\ \mathbb {E} _ {r, s, u \sim \mathcal {U} (0, 1), (a _ {r}, a _ {s}, a _ {u}) \sim \mathcal {D}} \left[ \mathrm{Sim} (a _ {s} + (u - s)   v _ {\theta}   (s, a _ {s})  , a _ {r} + (u - r)   v _ {\theta}   (r, a _ {r})) \right] \end{array}\tag{10}
$$

where $\mathrm { S i m } ( \cdot )$ is a function measuring the consistency of two velocities, and u satisfies $r < s < u$ The first part of Eq. 10 directly ensures the consistency of the two velocity vectors, while the second part enforces the consistency of the vector field from a trajectory perspective. Specifically, starting from arbitrary two points, $a _ { r }$ and $a _ { s }$ , are expected to converge to the same point at time u, thereby providing a way to directly define straight flows [68]. These constraints enforce consistent and straight flow across different timesteps, thus facilitating effective one-step action generation.

To leverage the temporal characteristic of the action chunks, we strengthen the temporal constraint with the frequency regularization. Namely, we propose to project the velocity vectors into the frequency domain using the type-II Discrete Cosine Transform (DCT):

$$
F (v _ {t}) _ {k} = \sum_ {n = 0} ^ {H - 1} v _ {t} (n) \cdot \cos \left[ \frac {\pi}{N} \left(n + \frac {1}{2}\right) k \right], \quad \text { for } k = 0, \dots , H - 1\tag{11}
$$

where $F ( v _ { t } ) _ { k }$ denotes the spectral coefficient of the k-th frequency component of $v _ { \theta } ( t , a _ { t } )$ . The function Sim(·) in Eq. 10 is then defined as the $\ell _ { 2 }$ norm between the two frequency coefficients as:

$$
\operatorname{Sim} \left(v _ {r}, v _ {s}\right) = \left\| F (v _ {r}) - F (v _ {s}) \right\| _ {2}\tag{12}
$$

This loss encourages the frequency spectral profiles of velocity signals at different timesteps to match, thereby aligning their temporal dynamics. As a result, the action chunks generated across different timesteps exhibit greater temporal consistency, which in turn guides the policy towards learning straighter and more stable flow trajectories suitable for one-step inference.

Adaptive Frequency Component Loss. In robotic manipulation, the distribution of frequency com ponents within each action chunk varies over time throughout task execution. This variability arises because robotic manipulation sequences typically alternate between stationary and non-stationary motion phases. During low-dynamic movements (e.g., reaching or moving between positions), only a subset of dimensions in the action chunks exhibit noticeable variation, while others remain relatively smooth. Conversely, during high-dynamic movements (e.g., transitions between skills or contact-rich interactions), high-frequency variations are more prominent and informative. We show some visualization examples in Sec. A.2. To effectively capture this structured temporal variation, we draw inspiration from the Focal Loss [35] and propose an adaptive weighting scheme that emphasizes frequency components with greater discrepancy. Concretely, for each frequency band k, we compute a frequency-domain weight $w _ { k }$ based on the difference between the corresponding frequency components of two action velocity vectors:

$$
w _ {k} = \frac {\exp \left(\| F (v _ {r}) _ {k} - F (v _ {s}) _ {k} \| _ {2}\right)}{\sum_ {j = 0} ^ {H - 1} \exp \left(\| F (v _ {r}) _ {j} - F (v _ {s}) _ {j} \| _ {2}\right)}\tag{13}
$$

where $F ( v _ { t } ) _ { k }$ is the DCT coefficient of the k-th frequency component. We then define the adaptive frequency component loss as:

$$
\operatorname{Sim} \left(v _ {r}, v _ {s}\right) = \sum_ {k = 0} ^ {H - 1} w _ {k} \cdot \| F (v _ {r}) _ {k} - F (v _ {s}) _ {k} \| _ {2}\tag{14}
$$

This adaptive loss encourages the model to focus more on frequency bands that encode informative dynamic variation across time. As a result, it strengthens the model’s ability to align the temporal structure of action chunks across different timesteps. We investigate the effectiveness of the adaptivity capability of the proposed loss in Sec. 4.3.

Overall Training Objective. Our training objective consists of two complementary components: a standard flow matching loss in Eq. 8 supervising the flow from prior noise to expert actions, and a frequency-domain loss in Eq. 14 that enforces temporal consistency across various timesteps. We jointly optimize these two objectives through the final loss:

$$
\mathcal {L} _ {\text { total }} = \mathcal {L} _ {\text { fm }} + \mathcal {L} _ {\text { freq }}\tag{15}
$$

This unified formulation ensures that FreqPolicy learns both accurate and temporally consistent flows, ultimately enabling reliable one-step action generation across diverse manipulation tasks.

## 4 Experimental Results

## 4.1 Simulation Experiments

To thoroughly validate the FreqPolicy, we conduct simulation experiments under both 2D [47] and 3D input settings [74, 3, 55] as shown in Fig. 2, and perform comprehensive comparisons. We further integrate FreqPolicy with existing vision-language-action (VLA) models to demonstrate its generalization ability.

Experiment with 2D Inputs. The FreqPolicy is evaluated on 5 tasks from the widely-used Robomimic [47] benchmark: Lift, Can, Square, Transport, and Tool Hang. For each task, we use proficient human demonstrations datasets with image-based observations, containing 200 demonstrations per task. We evaluate the average success rate across 3 random seeds for all tasks. For each seed, we evaluate each task over 50 different initializations and compute the mean success rate. The Number of Function Evaluations (NFE) metric is reported to measure the number of generation steps.

Table 1 compares our approach with existing policies on the Robomimic benchmark. FreqPolicy surpasses prior one-step methods, e.g., CP [53] and IMLE Policy [57]. Moreover, FreqPolicy even outperforms a few classical multi-step policies. We further reproduced Consistency-FM [68] that uses spatial-domain constraints for robotic manipulation with the same setting. FreqPolicy achieves superior performance, improving the Transport task by 6% and the Tool Hang task by 5%, demonstrating the effectiveness of the proposed frequency-domain consistency.

![](images/dd9b1e04af86fe0aecbeff9e0800f198f01f0be9a8c96df64e85e8432068b224.jpg)  
Figure 2: Experimental benchmarks. We evaluate FreqPolicy on 5 benchmarks, including a total of 93 simulated tasks (left) and 3 real robotics tasks (right).

Table 1: Performance on the Robomimic dataset [47]. We compare the FreqPolicy against a set of state-of-the-art multi-step generative visuomotor policies. We report the mean and standard deviation of success rates. Tasks marked with ∗ are the result of our implementation.

<table><tr><td>Method</td><td>NFE</td><td>Lift</td><td>Can</td><td>Square</td><td>Transport</td><td>Toolhang</td></tr><tr><td>DDPM [11]</td><td>15</td><td>1.00</td><td> $0.98 \pm .01$ </td><td> $0.91 \pm .01$ </td><td> $0.80 \pm .04$ </td><td> $0.52 \pm .05$ </td></tr><tr><td>DDiM [11]</td><td>15</td><td>1.00</td><td> $0.99 \pm .01$ </td><td> $0.92 \pm .03$ </td><td> $0.79 \pm .04$ </td><td> $0.55 \pm .05$ </td></tr><tr><td>RectifiedFlow* [40]</td><td>15</td><td>1.00</td><td> $0.96 \pm .02$ </td><td> $0.90 \pm .02$ </td><td> $0.84 \pm .04$ </td><td> $0.90 \pm .02$ </td></tr><tr><td>ActionFlow [19]</td><td>10</td><td>1.00</td><td> $0.99 \pm .01$ </td><td> $0.87 \pm .10$ </td><td>-</td><td> $0.81 \pm .09$ </td></tr><tr><td>AdaFlow [24]</td><td>-</td><td>1.00</td><td>1.00</td><td>0.98</td><td>0.92</td><td>0.88</td></tr><tr><td>ConsistencyPolicy* [53]</td><td>3</td><td>1.00</td><td> $0.95 \pm .02$ </td><td> $0.96 \pm .01$ </td><td> $0.88 \pm .02$ </td><td> $0.77 \pm .03$ </td></tr><tr><td>DDiM [11]</td><td>1</td><td>0.04</td><td> $0.00 \pm .00$ </td><td> $0.00 \pm .00$ </td><td> $0.00 \pm .00$ </td><td> $0.00 \pm .00$ </td></tr><tr><td>ConsistencyPolicy* [53]</td><td>1</td><td>1.00</td><td> $0.98 \pm .01$ </td><td> $0.92 \pm .02$ </td><td> $0.78 \pm .03$ </td><td> $0.70 \pm .03$ </td></tr><tr><td>Consistent-FM* [68]</td><td>1</td><td>1.00</td><td> $0.94 \pm .02$ </td><td> $0.90 \pm .01$ </td><td> $0.84 \pm .02$ </td><td> $0.80 \pm .02$ </td></tr><tr><td>IMLE Policy [57]</td><td>1</td><td>1.00</td><td>0.98</td><td>0.82</td><td>0.90</td><td>0.81</td></tr><tr><td>Ours</td><td>1</td><td>1.00</td><td> $0.98 \pm .02$ </td><td> $0.92 \pm .02$ </td><td> $0.90 \pm .02$ </td><td> $0.85 \pm .03$ </td></tr></table>

Experiment with 3D Inputs. To implement fair comparisons, we follow the existing methods [76, 28] and conduct experiments on 53 tasks across two benchmarks: Adroit [55] and MetaWorld [74]. Specifically, we use reinforcement learning with VRL3 [63] to collect expert demonstrations for Adroit, and use scripted policies to obtain demonstrations for MetaWorld. Training is conducted using 10 expert demonstrations per task. Following the evaluation protocol in previous works [76, 28, 78], we report the performance for each task across 3 random seeds. For each seed, we evaluate 20 segments every 200 training epochs and compute the average success rate of the top-5 trails, along with the average inference time per task.

Table 2 presents the comparisons with previous methods and demonstrates consistently improved performance. Compared to the SDM [28] strategy using distribution matching distillation, FreqPolicy achieves consistent gains in average success rate across 3 benchmarks, e.g., MetaWorld rises from 74.8% to 78.5%. Unlike SDM, FreqPolicy eliminates the dependence on pretrained teachers. Moreover, against FlowPolicy [78] applying a direct consistency constraint in the spatial domain, FreqPolicy maintains a 1.3% lead on MetaWorld. This advantage is particularly pronounced on the Medium, Hard, and Very Hard splits of MetaWorld, further validating the benefit of our frequency consistency supervision. Please refer to Sec. A.3 for details.

Experiment with VLA settings. We combine different policies with the classic OpenVLA model [33] and conduct systematic experiments on the LIBERO simulation benchmark [37], covering 4 task suites: LIBERO-Spatial, LIBERO-Object, LIBERO-Goal, and LIBERO-Long. Each suite provides 500 expert demonstrations across 10 tasks, designed to evaluate policy generalization across varying spatial layouts, object types, goal specifications, and long-horizon tasks. All models are evaluated under the same experimental protocol, with results averaged success rate (%) over 500 trials per suite (10 tasks × 50 episodes). Please refer to Sec. A.4 for more details.

We integrate FreqPolicy, Diffusion Policy (DP) [11], and Flow Matching (FM) [59, 19] with the OpenVLA. Table 3 shows the comparisons of VLA models with various policy heads. Our method (OpenVLA-FreqPolicy) achieves the highest average success rate of 94.8%, outperforming both

Table 2: Performance on the benchmarks with 3D inputs [74, 55]. We assess performance on 53 challenging tasks with 3 random seeds, reporting the mean success rate (%) and standard deviation. Tasks marked with ∗ are reproduced results.

<table><tr><td>Method</td><td>NFE</td><td>Adroit (3)</td><td>Meteworld Easy (28)</td><td>Metaworld Medium (11)</td><td>Metaworld Hard (6)</td><td>Metaworld Very Hard (5)</td><td>Average</td></tr><tr><td>DP [11]</td><td>10</td><td>31.7</td><td>83.6</td><td>31.1</td><td>9.0</td><td>26.6</td><td> $55.5 \pm 3.6$ </td></tr><tr><td>DP3* [76]</td><td>10</td><td>74.3</td><td>89.0</td><td>72.7</td><td>38.0</td><td>75.8</td><td> $76.1 \pm 2.3$ </td></tr><tr><td>ManiCM [42]</td><td>1</td><td>72.3</td><td>83.6</td><td>55.6</td><td>33.3</td><td>67.0</td><td> $69.0 \pm 4.6$ </td></tr><tr><td>SDM [28]</td><td>1</td><td>74.0</td><td>86.5</td><td>65.8</td><td>35.8</td><td>71.6</td><td> $74.8 \pm 4.5$ </td></tr><tr><td>FlowPolicy* [78]</td><td>1</td><td>69.3</td><td>89.6</td><td>66.5</td><td>43.8</td><td>75.8</td><td> $77.2 \pm 2.8$ </td></tr><tr><td>Ours</td><td>1</td><td>72.6</td><td>90.6</td><td>65.8</td><td>46.8</td><td>79.4</td><td> $78.5 \pm 2.6$ </td></tr></table>

Table 3: Success rates and inference speed on LIBERO benchmark [37]. Methods with ∗ indicate our implementations. FreqPolicy $( \tt N F E = 1 )$ outperforms Diffusion Policy (NFE = 50) and Flow Matching Policy $( \tt N F E = 1 0 \ \& \ 1 )$ , achieving higher success rates and faster speed, demonstrating superior effectiveness.

<table><tr><td>Method</td><td>NFE</td><td>Spatial (%)</td><td>Object (%)</td><td>Goal (%)</td><td>Long (%)</td><td>Average (%)</td><td>Speed (Hz)</td></tr><tr><td>OpenVLA-DP [32]</td><td>50</td><td>92.0</td><td>75.0</td><td>93.4</td><td>11.8</td><td>68.1</td><td>0.32</td></tr><tr><td>OpenVLA-FlowMatching*</td><td>1</td><td>95.0</td><td>97.6</td><td>96.0</td><td>85.2</td><td>93.5</td><td>5.92</td></tr><tr><td>OpenVLA-FlowMatching*</td><td>10</td><td>96.0</td><td>97.2</td><td>97.8</td><td>83.6</td><td>93.7</td><td>1.26</td></tr><tr><td>OpenVLA-FreqPolicy (Ours)</td><td>1</td><td>97.0</td><td>98.6</td><td>96.0</td><td>87.6</td><td>94.8</td><td>6.05</td></tr></table>

OpenVLA-DP and OpenVLA-FlowMatching in all categories except for the Goal suite, where it is tied for second-best. Compared to OpenVLA-DP, which has an average success rate of 68.1%, OpenVLA-FreqPolicy shows a significant improvement, especially in the Long task suite (87.6% vs. 11.8%). Notably, OpenVLA-FreqPolicy with just 1 NFE achieves a higher average success rate than OpenVLA-FlowMatching with 10 NFE (94.8% vs. 93.7%), while offering significantly faster inference speed (5 times faster).

## 4.2 Real World Experiments

Task Design. We design 3 long-horizon real-world tasks on 2 different robotic arms as illustrated in Fig. 3, to evaluate the FreqPolicy: 1) Fruit Sorting Task. The Franka robot is required to sequentially place a banana, an avocado, and a mango into a basket. This is a compound pick-and-place task involving ordered object manipulation. 2) Toy Organization Task. The Franka robot must first pull open a drawer, place two toy dolls inside, and then close the cabinet door. This task integrates multiple skills, including pull, push, and pick-and-place. 3) Trash Disposal Task. The UR robot needs to open a trash bin lid, place several pieces of food waste (e.g., half a bun and half a piece of bread) into the bin, and then close the lid. This task requires a combination of pressing and pick-and-place fine manipulation capabilities. More demonstrations can be found in Sec. A.5.

Main Results. We primarily compare against Diffusion Policy [11] and Flow Matching Policy [59, 12, 19] with 1-step and 10-step inference settings, respectively. For each evaluation, we perform 20 trials with various initializations on the physical robot and report the mean success rate. We also measure and report the real-time inference frequency of each policy using an NVIDIA RTX 4090. As shown in Fig. 3a and Fig. 3b, FreqPolicy achieves success rates comparable to, and in some cases exceeding, those of multi-step policy models, while using only single-step inference. For example, on Task 1, FreqPolicy achieves a top success rate of 70% at an inference speed of 93.5 Hz, outperforming multi-step Diffusion Policy (55% at 19.8 Hz) and Flow Matching Policy (60% at 20.2 Hz). Similar trends are observed across all tasks. These results demonstrate that FreqPolicy can match or surpass the performance of multi-step policy models with just one inference step, while significantly reducing computational overhead. Additional evaluation details are provided in Sec. A.5.

## 4.3 Ablation Study

Table 4 shows the effect of various consistency constraints on one-step action prediction. Model #1 serves as a baseline, using vanilla flow matching for one-step action generation. Model #2 equips

![](images/6a0c0455855e74fc7025a6394321e009f6210815ba3fd104eef620cf1760baa6.jpg)

![](images/6883a9e77c7af12e6942156fa68452a30d9722f00717888cc47993d3126f3b9c.jpg)  
(a) Success rate (%) on three real-world tasks.

![](images/514b2101c86050dae78c93ea080607c29442c03d51f89a6baef407c65dfda552.jpg)  
(b) Inference speed (Hz) on three real-world tasks.  
Figure 3: Demonstrations of three real-world tasks (top) and the test results of different policies (bottom). We evaluated the success rate (%) and inference speed (Hz) for all methods. DP stands for Diffusion Policy, FM for Flow Matching Policy, and the number represents inference steps of the corresponding policy model. FreqPolicy consistently outperformed baselines in both success rate and inference speed, demonstrating its effectiveness on real-world robotic platforms.

Table 4: Ablations on Robomimic. We analyze the impact of various designs in transitioning from a vanilla flow matching policy with one-step action generation to our FreqPolicy. Results on the Lift are omitted, as all methods achieve saturated performance. In addition, we evaluate FreqPolicy under different spectral supervision settings to assess the importance of frequency-aware consistency.

<table><tr><td>Model</td><td>Consistency Objectives</td><td>Can</td><td>Square</td><td>Transport</td><td>Toolhang</td></tr><tr><td># 1</td><td>Vanilla Flow Matching (NFE=1)</td><td>0.94</td><td>0.90</td><td>0.84</td><td>0.76</td></tr><tr><td># 2</td><td>w/ consistency constrain</td><td>0.98</td><td>0.92</td><td>0.90</td><td>0.78</td></tr><tr><td># 3</td><td>w/ frequency consistency constrain - high</td><td>0.96</td><td>0.92</td><td>0.88</td><td>0.80</td></tr><tr><td># 4</td><td>w/ frequency consistency constrain - low</td><td>0.95</td><td>0.90</td><td>0.88</td><td>0.82</td></tr><tr><td># 5</td><td>w/ frequency consistency constrain - full</td><td>0.96</td><td>0.92</td><td>0.92</td><td>0.82</td></tr><tr><td># 6</td><td>w/ frequency consistency constrain - adaptive</td><td>0.97</td><td>0.93</td><td>0.92</td><td>0.88</td></tr></table>

Model #1 with a consistency loss similar to Consistency-FM [68], yielding a modest improvement (e.g., success rate increases to 90% on the transport task).

To investigate the effect of supervising different frequency bands, Model #3 and Model #4 introduce low-frequency and high-frequency consistency constraints on top of Model #1, respectively. These methods manually select specific frequency components for constraints and result in minor improvements. We attribute this to the diverse and dynamic nature of frequency components in action chunks during robotic manipulation. Model #5 applies consistency constraints across the full frequency component, leading to a more substantial gain, raising transport from 90% to 92% and toolhang from 78% to 82%, thereby validating the efficacy of frequency consistency. Finally, Model #6 incorporates our adaptive frequency loss, further boosting performance to state-of-the-art levels and confirming the advantage of adaptive frequency constraints.

## 5 Limitation & Conclusion

In this work, we introduce FreqPolicy, a novel one-step action generation policy that performs robustly under both 2D and 3D inputs. For the first time, FreqPolicy imposes a frequency-domain consistency constraint on flow matching, encouraging consistent velocities across different timesteps, and incorporates an adaptive loss that emphasizes higher-variance frequency components. Through this training regimen, FreqPolicy learns a straight-line flow to produce single-step actions. We validate its effectiveness on 53 simulated tasks and demonstrate a significant speed-up on 40 Libero tasks for a VLA model. We hope that FreqPolicy will advance embodied vision–motion planning and VLA models in real-world applications. However, our work presents certain limitations. We primarily validated the effectiveness of our proposed FreqPolicy on flow matching. In principle, the same core idea can be extended to diffusion-based visual-motor policies, which we will systematically investigate in future work. Our method demands more computational resources, as it requires twice forward passes to process two random samples. In future research, we will explore the computationally efficient one-step action generators.

## References

[1] Pieter Abbeel and Andrew Y Ng. Apprenticeship learning via inverse reinforcement learning. In Proceedings ofthe twenty-first international conference on Machine learning, page 1, 2004.

[2] Michael Bain and Claude Sammut. A framework for behavioural cloning. In Machine intelligence 15, pages 103–129, 1995.

[3] Chen Bao, Helin Xu, Yuzhe Qin, and Xiaolong Wang. Dexart: Benchmarking generalizable dexterous manipulation with articulated objects. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21190–21200, 2023.

[4] Homanga Bharadhwaj, Jay Vakil, Mohit Sharma, Abhinav Gupta, Shubham Tulsiani, and Vikash Kumar. Roboagent: Generalization and efficiency in robot manipulation via semantic augmentations and action chunking. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 4788–4795. IEEE, 2024.

[5] Johan Bjorck, Fernando Castañeda, Nikita Cherniadev, Xingye Da, Runyu Ding, Linxi Fan, Yu Fang, Dieter Fox, Fengyuan Hu, Spencer Huang, et al. Gr00t n1: An open foundation model for generalist humanoid robots. arXiv preprint arXiv:2503.14734, 2025.

[6] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. pi0: A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024.

[7] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, et al. Rt-2: Vision-language action models transfer web knowledge to robotic control. arXiv preprint arXiv:2307.15818, 2023.

[8] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, et al. Rt-1: Robotics transformer for real-world control at scale. arXiv preprint arXiv:2212.06817, 2022.

[9] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, et al. Video generation models as world simulators. OpenAI Blog, 1:8, 2024.

[10] Haojun Chen, Minghao Liu, Xiaojian Ma, Zailin Ma, Huimin Wu, Chengdong Ma, Yuanpei Chen, Yifan Zhong, Mingzhi Wang, Qing Li, et al. Fast visuomotor policies via partial denoising. arXiv preprint arXiv:2503.00339, 2025.

[11] Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. The International Journal ofRobotics Research, page 02783649241273668, 2023.

[12] Eugenio Chisari, Nick Heppert, Max Argus, Tim Welschehold, Thomas Brox, and Abhinav Valada. Learning robotic manipulation policies from point clouds with conditional flow matching. arXiv preprint arXiv:2409.07343, 2024.

[13] Quan Dao, Hao Phung, Binh Nguyen, and Anh Tran. Flow matching in latent space. arXiv preprint arXiv:2307.08698, 2023.

[14] Trung Dao, Thuan Hoang Nguyen, Thanh Le, Duc Vu, Khoi Nguyen, Cuong Pham, and Anh Tran. Swiftbrush v2: Make your one-step diffusion model better than its teacher. In European Conference on Computer Vision, pages 176–192. Springer, 2024.

[15] Haoran Ding, Noémie Jaquier, Jan Peters, and Leonel Rozo. Fast and robust visuomotor riemannian flow matching policy. arXiv preprint arXiv:2412.10855, 2024.

[16] Yuhang Dong, Haizhou Ge, Yupei Zeng, Jiangning Zhang, Beiwen Tian, Guanzhong Tian, Hongrui Zhu, Yufei Jia, Ruixiang Wang, Ran Yi, et al. Imit diff: Semantics guided diffusion transformer with dual resolution fusion for imitation learning. arXiv preprint arXiv:2502.09649, 2025.

[17] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024.

[18] Peter Florence, Lucas Manuelli, and Russ Tedrake. Self-supervised correspondence in visuomotor policy learning. IEEE Robotics and Automation Letters, 5(2):492–499, 2019.

[19] Niklas Funk, Julen Urain, Joao Carvalho, Vignesh Prasad, Georgia Chalvatzaki, and Jan Peters. Actionflow: Equivariant, accurate, and efficient policies with spatially symmetric flow matching. arXiv preprint arXiv:2409.04576, 2024.

[20] Jonathan Ho and Stefano Ermon. Generative adversarial imitation learning. Advances in neural information processing systems, 29, 2016.

[21] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. Advances in Neural Information Processing Systems, 35:8633–8646, 2022.

[22] Sigmund H Høeg, Yilun Du, and Olav Egeland. Streaming diffusion policy: Fast policy synthesis with variable noise diffusion models. arXiv preprint arXiv:2406.04806, 2024.

[23] Zhi Hou, Tianyi Zhang, Yuwen Xiong, Haonan Duan, Hengjun Pu, Ronglei Tong, Chengyang Zhao, Xizhou Zhu, Yu Qiao, Jifeng Dai, et al. Dita: Scaling diffusion transformer for generalist vision-language-action policy. arXiv preprint arXiv:2503.19757, 2025.

[24] Xixi Hu, Qiang Liu, Xingchao Liu, and Bo Liu. Adaflow: Imitation learning with varianceadaptive flow-based policies. Advances in Neural Information Processing Systems, 37:138836– 138858, 2024.

[25] Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, et al. pi0.5: a visionlanguage-action model with open-world generalization. arXiv preprint arXiv:2504.16054, 2025.

[26] Eric Jang, Alex Irpan, Mohi Khansari, Daniel Kappler, Frederik Ebert, Corey Lynch, Sergey Levine, and Chelsea Finn. Bc-z: Zero-shot task generalization with robotic imitation learning. In Conference on Robot Learning, pages 991–1002. PMLR, 2022.

[27] Michael Janner, Yilun Du, Joshua B Tenenbaum, and Sergey Levine. Planning with diffusion for flexible behavior synthesis. arXiv preprint arXiv:2205.09991, 2022.

[28] Bofang Jia, Pengxiang Ding, Can Cui, Mingyang Sun, Pengfang Qian, Siteng Huang, Zhaoxin Fan, and Donglin Wang. Score and distribution matching policy: Advanced accelerated visuomotor policies via matched distillation. arXiv preprint arXiv:2412.09265, 2024.

[29] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. Advances in neural information processing systems, 35:26565–26577, 2022.

[30] Tsung-Wei Ke, Nikolaos Gkanatsios, and Katerina Fragkiadaki. 3d diffuser actor: Policy diffusion with 3d scene representations. arXiv preprint arXiv:2402.10885, 2024.

[31] Dongjun Kim, Chieh-Hsin Lai, Wei-Hsiang Liao, Naoki Murata, Yuhta Takida, Toshimitsu Uesaka, Yutong He, Yuki Mitsufuji, and Stefano Ermon. Consistency trajectory models: Learning probability flow ode trajectory of diffusion. arXiv preprint arXiv:2310.02279, 2023.

[32] Moo Jin Kim, Chelsea Finn, and Percy Liang. Fine-tuning vision-language-action models: Optimizing speed and success. arXiv preprint arXiv:2502.19645, 2025.

[33] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024.

[34] Xinghang Li, Minghuan Liu, Hanbo Zhang, Cunjun Yu, Jie Xu, Hongtao Wu, Chilam Cheang, Ya Jing, Weinan Zhang, Huaping Liu, et al. Vision-language foundation models as effective robot imitators. arXiv preprint arXiv:2311.01378, 2023.

[35] Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollár. Focal loss for dense object detection. In Proceedings of the IEEE international conference on computer vision, pages 2980–2988, 2017.

[36] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.

[37] Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. Libero: Benchmarking knowledge transfer for lifelong robot learning. Advances in Neural Information Processing Systems, 36:44776–44791, 2023.

[38] Qiang Liu. Rectified flow: A marginal preserving approach to optimal transport. URL https://arxiv. org/abs/2209.14577, 2022.

[39] Songming Liu, Lingxuan Wu, Bangguo Li, Hengkai Tan, Huayu Chen, Zhengyi Wang, Ke Xu, Hang Su, and Jun Zhu. Rdt-1b: a diffusion foundation model for bimanual manipulation. arXiv preprint arXiv:2410.07864, 2024.

[40] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022.

[41] Cheng Lu and Yang Song. Simplifying, stabilizing and scaling continuous-time consistency models. arXiv preprint arXiv:2410.11081, 2024.

[42] Guanxing Lu, Zifeng Gao, Tianxing Chen, Wenxun Dai, Ziwei Wang, Wenbo Ding, and Yansong Tang. Manicm: Real-time 3d diffusion policy via consistency model for robotic manipulation. arXiv preprint arXiv:2406.01586, 2024.

[43] Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. Latent consistency models: Synthesizing high-resolution images with few-step inference. arXiv preprint arXiv:2310.04378, 2023.

[44] Weijian Luo, Tianyang Hu, Shifeng Zhang, Jiacheng Sun, Zhenguo Li, and Zhihua Zhang. Diffinstruct: A universal approach for transferring knowledge from pre-trained diffusion models. Advances in Neural Information Processing Systems, 36:76525–76546, 2023.

[45] Nanye Ma, Mark Goldstein, Michael S Albergo, Nicholas M Boffi, Eric Vanden-Eijnden, and Saining Xie. Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers. In European Conference on Computer Vision, pages 23–40. Springer, 2024.

[46] Xiao Ma, Sumit Patidar, Iain Haughton, and Stephen James. Hierarchical diffusion policy for kinematics-aware multi-task robotic manipulation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18081–18090, 2024.

[47] Ajay Mandlekar, Danfei Xu, Josiah Wong, Soroush Nasiriany, Chen Wang, Rohun Kulkarni, Li Fei-Fei, Silvio Savarese, Yuke Zhu, and Roberto Martín-Martín. What matters in learning from offline human demonstrations for robot manipulation. arXiv preprint arXiv:2108.03298, 2021.

[48] Thuan Hoang Nguyen and Anh Tran. Swiftbrush: One-step text-to-image diffusion model with variational score distillation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7807–7816, 2024.

[49] Karl Pertsch, Kyle Stachowicz, Brian Ichter, Danny Driess, Suraj Nair, Quan Vuong, Oier Mees, Chelsea Finn, and Sergey Levine. Fast: Efficient action tokenization for vision-language-action models. arXiv preprint arXiv:2501.09747, 2025.

[50] Dean A Pomerleau. Alvinn: An autonomous land vehicle in a neural network. Advances in neural information processing systems, 1, 1988.

[51] Dean A Pomerleau. Efficient training of artificial neural networks for autonomous navigation. Neural computation, 3(1):88–97, 1991.

[52] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988, 2022.

[53] Aaditya Prasad, Kevin Lin, Jimmy Wu, Linqi Zhou, and Jeannette Bohg. Consistency policy: Accelerated visuomotor policies via consistency distillation. In Robotics: Science and Systems, 2024.

[54] Rouhollah Rahmatizadeh, Pooya Abolghasemi, Ladislau Bölöni, and Sergey Levine. Visionbased multi-task manipulation for inexpensive robots using end-to-end learning from demonstration. In 2018 IEEE international conference on robotics and automation (ICRA), pages 3758–3765. IEEE, 2018.

[55] Aravind Rajeswaran, Vikash Kumar, Abhishek Gupta, Giulia Vezzani, John Schulman, Emanuel Todorov, and Sergey Levine. Learning complex dexterous manipulation with deep reinforcement learning and demonstrations. arXiv preprint arXiv:1709.10087, 2017.

[56] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1(2):3, 2022.

[57] Krishan Rana, Robert Lee, David Pershouse, and Niko Suenderhauf. Imle policy: Fast and sample efficient visuomotor policy learning via implicit maximum likelihood estimation. arXiv preprint arXiv:2502.12371, 2025.

[58] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.

[59] Quentin Rouxel, Andrea Ferrari, Serena Ivaldi, and Jean-Baptiste Mouret. Flow matching imitation learning for multi-support manipulation. in 2024 ieee-ras 23rd international conference on humanoid robots (humanoids), 2024.

[60] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in neural information processing systems, 35:36479–36494, 2022.

[61] Lucy Xiaoyang Shi, Brian Ichter, Michael Equi, Liyiming Ke, Karl Pertsch, Quan Vuong, James Tanner, Anna Walling, Haohuan Wang, Niccolo Fusai, et al. Hi robot: Open-ended instruction following with hierarchical vision-language-action models. arXiv preprint arXiv:2502.19417, 2025.

[62] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. In Proceedings of the 40th International Conference on Machine Learning, 2023.

[63] Che Wang, Xufang Luo, Keith Ross, and Dongsheng Li. Vrl3: A data-driven framework for visual deep reinforcement learning. Advances in Neural Information Processing Systems, 35:32974–32988, 2022.

[64] Fu-Yun Wang, Zhaoyang Huang, Alexander Bergman, Dazhong Shen, Peng Gao, Michael Lingelbach, Keqiang Sun, Weikang Bian, Guanglu Song, Yu Liu, et al. Phased consistency models. Advances in neural information processing systems, 37:83951–84009, 2024.

[65] Zhendong Wang, Zhaoshuo Li, Ajay Mandlekar, Zhenjia Xu, Jiaojiao Fan, Yashraj Narang, Linxi Fan, Yuke Zhu, Yogesh Balaji, Mingyuan Zhou, et al. One-step diffusion policy: Fast visuomotor policies via diffusion distillation. arXiv preprint arXiv:2410.21257, 2024.

[66] Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. Advances in Neural Information Processing Systems, 36:8406–8441, 2023.

[67] Zhou Xian and Nikolaos Gkanatsios. Chaineddiffuser: Unifying trajectory diffusion and keypose prediction for robotic manipulation. In Conference on Robot Learning/Proceedings of Machine Learning Research. Proceedings of Machine Learning Research, 2023.

[68] Ling Yang, Zixiang Zhang, Zhilong Zhang, Xingchao Liu, Minkai Xu, Wentao Zhang, Chenlin Meng, Stefano Ermon, and Bin Cui. Consistency flow matching: Defining straight flows with velocity consistency. arXiv preprint arXiv:2407.02398, 2024.

[69] Weiwei Ye, Songgaojun Deng, Qiaosha Zou, and Ning Gui. Frequency adaptive normalization for non-stationary time series forecasting. arXiv preprint arXiv:2409.20371, 2024.

[70] Kun Yi, Jingru Fei, Qi Zhang, Hui He, Shufeng Hao, Defu Lian, and Wei Fan. Filternet: Harnessing frequency filters for time series forecasting. Advances in Neural Information Processing Systems, 37:55115–55140, 2024.

[71] Tianwei Yin, Michaël Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Fredo Durand, and Bill Freeman. Improved distribution matching distillation for fast image synthesis. Advances in neural information processing systems, 37:47455–47487, 2024.

[72] Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6613–6623, 2024.

[73] Guoqi Yu, Yaoming Li, Juncheng Wang, Xiaoyu Guo, Angelica I Aviles-Rivero, Tong Yang, and Shujun Wang. Refocus: Reinforcing mid-frequency and key-frequency modeling for multivariate time series forecasting. arXiv preprint arXiv:2502.16890, 2025.

[74] Tianhe Yu, Deirdre Quillen, Zhanpeng He, Ryan Julian, Karol Hausman, Chelsea Finn, and Sergey Levine. Meta-world: A benchmark and evaluation for multi-task and meta reinforcement learning. In Conference on robot learning, pages 1094–1100. PMLR, 2020.

[75] Yanjie Ze, Zixuan Chen, Wenhao Wang, Tianyi Chen, Xialin He, Ying Yuan, Xue Bin Peng, and Jiajun Wu. Generalizable humanoid manipulation with improved 3d diffusion policies. arXiv preprint arXiv:2410.10803, 2024.

[76] Yanjie Ze, Gu Zhang, Kangning Zhang, Chenyuan Hu, Muhan Wang, and Huazhe Xu. 3d diffusion policy: Generalizable visuomotor policy learning via simple 3d representations. arXiv preprint arXiv:2403.03954, 2024.

[77] Fan Zhang and Michael Gienger. Affordance-based robot manipulation with flow matching. arXiv preprint arXiv:2409.01083, 2024.

[78] Qinglun Zhang, Zhen Liu, Haoqiang Fan, Guanghui Liu, Bing Zeng, and Shuaicheng Liu. Flowpolicy: Enabling fast and robust 3d flow-based policy via consistency flow matching for robot manipulation. In Proceedings ofthe AAAI Conference on Artificial Intelligence, pages 14754–14762, 2025.

[79] Tianhao Zhang, Zoe McCarthy, Owen Jow, Dennis Lee, Xi Chen, Ken Goldberg, and Pieter Abbeel. Deep imitation learning for complex manipulation tasks from virtual reality teleoperation. In 2018 IEEE international conference on robotics and automation (ICRA), pages 5628–5635. Ieee, 2018.

[80] Tony Z Zhao, Vikash Kumar, Sergey Levine, and Chelsea Finn. Learning fine-grained bimanual manipulation with low-cost hardware. arXiv preprint arXiv:2304.13705, 2023.

[81] Tian Zhou, Ziqing Ma, Qingsong Wen, Liang Sun, Tao Yao, Wotao Yin, Rong Jin, et al. Film: Frequency improved legendre memory model for long-term time series forecasting. Advances in neural information processing systems, 35:12677–12690, 2022.

[82] Tian Zhou, Ziqing Ma, Qingsong Wen, Xue Wang, Liang Sun, and Rong Jin. Fedformer: Frequency enhanced decomposed transformer for long-term series forecasting. In International conference on machine learning, pages 27268–27286. PMLR, 2022.

[83] Minjie Zhu, Yichen Zhu, Jinming Li, Junjie Wen, Zhiyuan Xu, Ning Liu, Ran Cheng, Chaomin Shen, Yaxin Peng, Feifei Feng, et al. Scaling diffusion policy in transformer to 1 billion parameters for robotic manipulation. arXiv preprint arXiv:2409.14411, 2024.

[84] Yuanzhi Zhu, Ruiqing Wang, Shilin Lu, Junnan Li, Hanshu Yan, and Kai Zhang. Oftsr: Onestep flow for image super-resolution with tunable fidelity-realism trade-offs. arXiv preprint arXiv:2412.09465, 2024.

## A Technical Appendices and Supplementary Material

In this appendix, we first present further details on the model architecture of FreqPolicy in Sec. A.1. Next, Sec. A.2 provides additional visualizations and frequency-domain analyses of various action chunks. We then supplement the simulation experiments with additional information in Sec. A.3, along with details on integrating FreqPolicy into VLA models in Sec. A.4. Finally, Sec. A.5 describes the setup and results of the real-world experiments.

## A.1 More Model Configuration

As a Visuomotor Policy. Following previous works [11, 76, 28, 42, 78], we adopt a standard 1D CNN-based U-Net architecture as the backbone of FreqPolicy to ensure a fair comparison with existing models. FreqPolicy is designed to accept both 2D and 3D observations as input. For 2D images, we employ ResNet-18 as the visual encoder. For 3D input, we follow [76, 78] and use a lightweight MLP to encode the input point cloud.

As a Head of the VLA Model. FreqPolicy can also serve as a policy head for existing VLA models, as long as the underlying VLA is capable of predicting action vector fields. In this work, we integrate FreqPolicy into OpenVLA [33], following the setup in [32], where the predicted action vector field is obtained by applying a nonlinear mapping to the noise-conditioned latent features.

## A.2 Extended Frequency Analysis for Robotic Manipulation

In Fig. 4, we present visualizations of action chunks from different real-world and simulation scenarios, including the observed images, the corresponding multi-dimensional temporal action signals, and the transformed DCT frequency coefficients. In the simulation scenario shown in Fig. 4a, the Franka robot transitions from “approaching the blue can” to “grasping the can”, during which the expected action chunk contains high-frequency variations associated with gripper motion. Similarly, in Fig. 4b, as the robot moves from “approaching the stove” to “placing down the kettle”, the gripper opening action introduces high-frequency signals, while other control signals remain relatively smooth. In the real-world scenario shown in Fig. 4c, during the process of picking up the doll, the Franka robot exhibits relatively smooth motion transitions, with the action signal primarily dominated by low-frequency components. Hence, based on the above observations of different action chunks and their frequency-domain characteristics, we justify the motivation of the adaptive frequency coefficient loss introduced in Sec 3. This loss enables the model to focus more effectively on the frequency components that exhibit meaningful variation across diverse action chunks.

## A.3 More Details on Visuomotor Policy Simulation

## A.3.1 More Details on Simulation with 2D Inputs

Implementation Details. We implement the RectifiedFlow [40, 38], ConsistencyPolicy [53], and Consistency-FM [68] to comprehensively evaluate our proposed FreqPolicy. For RectifiedFlow, we adopt a 1D CNN-based U-Net to predict action vector fields from observations. Following [6], we sample time steps t from a Beta distribution during training to interpolate intermediate states. For ConsistencyPolicy, we follow the procedure in [53], where an EDM [29] teacher model is first trained in the initial stage, and then a student model is distilled for one-step action generation using the CTM objective [31]. For Consistency-FM, we also use a 1D CNN-based U-Net to predict action vector fields from observations. We then sample different time steps r and s from a uniform schedule and apply velocity field consistency constraints between them. To train all models, we use observations from the past 2 time steps as input and predict a 16-step action chunk, from which the first 8 steps are selected for execution. All models are trained using a batch size of 128 with the AdamW optimizer and a learning rate of 1.0e-4. Training is conducted for 1000 epochs on a single NVIDIA A100 GPU. For ConsistencyPolicy, the student model is trained for 450 epochs.

## A.3.2 More Details on Simulation with 3D Inputs

Implementation Details. Following the prior works [76, 78], we use a lightweight MLP to encode the point cloud and use a 1D CNN-based U-Net to predict the action vector field. To train the FreqPolicy model, we use a batch size of 128 and the AdamW optimizer with a learning rate of

![](images/c55b5ea79b480484af4606daada27f600a543f89435eb7172cfaa2cf04708a55.jpg)

![](images/083ee561f69361b83a0a9b54a13e4277be9148426d024ccc21d8b8eb6c9bd85d.jpg)

![](images/4d678b1f9445aae78955ff56aea654518426e1a78ac60fc78cbe03add874e419.jpg)

(a) Visualization of an action chunk from the Franka simulation dataset. When the robot transitions from “reach the blue can” to “grasping it”, the change in gripper introduces high-frequency signal variations. Other states exhibit no significant changes and tend to be smoother.  
![](images/80ed60ea9b38e7b6b3a8f517f6589bd4e9ddcd7611ee969a6fdad68c4331d2bb.jpg)

![](images/104f5a8274955c8bc6f289513da6aed2840740f70e440696ac10ad66f6e95b8e.jpg)

![](images/86cfb14eab98cae2c715a75847ff34a20d3467628e4ff7713e7bd8c430d15c12.jpg)

(b) Visualization of an action chunk from the Franka simulation dataset. During the transition from “approaching the stove” to “placing down the kettle”, the gripper exhibits high-frequency variations, while other states remain nearly unchanged.  
![](images/36e7b5d56a77b2e9396b2c8346997e838735603c635b753a66f557cd06b6853b.jpg)

![](images/f16ca477fdc5ad075bd979b7377140fc6355b4a2a9b5b5b59bd1508dc3c84c8c.jpg)

![](images/8fda09c1bfe8d09a891b6d3b5b308e9124909a82117eef58dd4f7eef9b7ae8bb.jpg)  
(c) Visualization of an action chunk from the Franka robot demonstrations. Once the robot successfully grasps the doll and begins placing it into the drawer, the overall motion exhibits relatively slow changes under a 30 Hz action sampling rate, dominated by low-frequency signals.  
Figure 4: Visualization of the spectral and temporal signals across different action chunks.

Table 5: We report the evaluation details of the 53 challenging tasks from Adroit [55] and Meta-World [74] under 3 random seeds, and report the mean success rate (%) and standard deviation for each task. Tasks marked with an asterisk ∗ indicate re-implemented versions. Compared to ManiCM [42], a one-step action generation model using consistency distillation based on diffusion-based policies, our method achieves superior performance. Similarly, we achieve an overall improvement of 9.4% over SDM [28] that adopts variational score distillation [72, 71] to achieve a one-step diffusion policy. Finally, in comparison to the flow-based policy FlowPolicy [78], which uses only spatial-domain consistency loss, FreqPolicy still leads by 2.8% gains, demonstrating the effectiveness of our frequency consistency constraint.

<table><tr><td rowspan="2">Alg / Task</td><td colspan="3">Adroit</td><td colspan="3">Meta-World (Easy)</td></tr><tr><td>Adroit Hammer</td><td>Adroit Door</td><td>Adroit Pen</td><td>Button Press</td><td>Coffee Button</td><td>Plate Slide Back Side</td></tr><tr><td>Diffusion Policy</td><td>45 ± 5</td><td>37 ± 2</td><td>13 ± 2</td><td>99 ± 1</td><td>99 ± 1</td><td>100 ± 0</td></tr><tr><td>3D Diffusion Policy</td><td>100 ± 0</td><td>75 ± 3</td><td>48 ± 3</td><td>100 ± 0</td><td>100 ± 0</td><td>100 ± 0</td></tr><tr><td>ManiCM</td><td>100 ± 0</td><td>68 ± 1</td><td>49 ± 4</td><td>100 ± 0</td><td>100 ± 0</td><td>100 ± 0</td></tr><tr><td>SDM Policy</td><td>100 ± 0</td><td>73 ± 2</td><td>49 ± 4</td><td>100 ± 0</td><td>100 ± 0</td><td>100 ± 0</td></tr><tr><td>FlowPolicy*</td><td>97 ± 3</td><td>62 ± 6</td><td>49 ± 5</td><td>100 ± 0</td><td>100 ± 0</td><td>100 ± 0</td></tr><tr><td>Ours</td><td>98 ± 2</td><td>68 ± 5</td><td>52 ± 4</td><td>100 ± 0</td><td>100 ± 0</td><td>100 ± 0</td></tr></table>

<table><tr><td rowspan="2">Alg / Task</td><td colspan="6">Meta-World (Easy)</td></tr><tr><td>Button Press Topdown</td><td>Button Press Topdown Wall</td><td>Button Press Wall</td><td>Peg Unplug Side</td><td>Door Close</td><td>Door Lock</td></tr><tr><td>Diffusion Policy</td><td>98 ± 1</td><td>96 ± 3</td><td>97 ± 3</td><td>74 ± 3</td><td>100 ± 0</td><td>86 ± 8</td></tr><tr><td>3D Diffusion Policy</td><td>99 ± 1</td><td>96 ± 3</td><td>100 ± 0</td><td>93 ± 3</td><td>100 ± 0</td><td>96 ± 3</td></tr><tr><td>ManiCM</td><td>100 ± 0</td><td>96 ± 2</td><td>98 ± 3</td><td>71 ± 15</td><td>100 ± 0</td><td>98 ± 2</td></tr><tr><td>SDM Policy</td><td>98 ± 2</td><td>99 ± 1</td><td>100 ± 0</td><td>74 ± 19</td><td>100 ± 0</td><td>96 ± 2</td></tr><tr><td>FlowPolicy*</td><td>100 ± 0</td><td>100 ± 0</td><td>100 ± 0</td><td>88 ± 5</td><td>100 ± 0</td><td>100 ± 0</td></tr><tr><td>Ours</td><td>100 ± 0</td><td>100 ± 0</td><td>100 ± 0</td><td>87 ± 4</td><td>100 ± 0</td><td>100 ± 0</td></tr></table>

$$
9 8 \pm 3
$$

$$
9 8 \pm 3
$$

$$
1 0 0 \pm 0
$$

$$
1 0 0 \pm 0
$$

$$
1 0 0 \pm 0
$$

$$
1 0 0 \pm 0
$$

$$
9 3 \pm 3
$$

$$
1 0 0 \pm 0
$$

$$
8 2 \pm 1 6
$$

$$
1 0 0 \pm 0
$$

$$
1 0 0 \pm 0
$$

$$
8 1 \pm 4
$$

$$
1 0 0 \pm 0
$$

$$
1 0 0 \pm 0
$$

$$
1 0 0 \pm 0
$$

$$
1 0 0 \pm 0
$$

$$
2 7 \pm 2 2
$$

$$
1 0 0 \pm 0
$$

$$
1 0 0 \pm 0
$$

$$
1 0 0 \pm 0
$$

$$
1 0 0 \pm 0
$$

$$
1 0 0 \pm 0
$$

$$
1 0 0 \pm 0
$$

$$
1 0 0 \pm 0
$$

$$
1 0 0 \pm 0
$$

$$
9 9 \pm 1
$$

$$
1 0 0 \pm 0
$$

$$
1 0 0 \pm 0
$$

$$
\mathbf {1 0 0} \pm 0
$$

$$
\mathbf {1 0 0} \pm 0
$$

$$
9 9 \pm 0
$$

$$
\mathbf {1 0 0} \pm 0
$$

$$
1 0 \pm 1 0
$$

$$
\mathbf {1 0 0} \pm 0
$$

$$
1 0 0 \pm 0
$$

$$
1 0 0 \pm 0
$$

$$
\mathbf {1 0 0} \pm 0
$$

$$
1 0 0 \pm 0
$$

$$
2 8 \pm 1 1
$$

$$
\mathbf {1 0 0} \pm 0
$$

$$
2 5 \pm 8
$$

<table><tr><td rowspan="2">Alg Task</td><td rowspan="2">Handle Press Side</td><td rowspan="2">Handle Pull Side</td><td colspan="6">Meta-World (Easy)</td></tr><tr><td>Lever Pull</td><td>Plate Slide</td><td>Plate Slide Back</td><td>Dial Turn</td><td>Reach</td><td>Reach Wall</td></tr><tr><td>Diffusion Policy</td><td> $100 \pm 0$ </td><td> $23 \pm 17$ </td><td> $49 \pm 5$ </td><td> $83 \pm 4$ </td><td> $99 \pm 0$ </td><td> $63 \pm 10$ </td><td> $18 \pm 2$ </td><td> $59 \pm 7$ </td></tr><tr><td>3D Diffusion Policy</td><td> $0 \pm 0$ </td><td> $\mathbf{82} \pm 5$ </td><td> $\mathbf{84} \pm 8$ </td><td> $\mathbf{100} \pm 0$ </td><td> $100 \pm 0$ </td><td> $91 \pm 0$ </td><td> $26 \pm 3$ </td><td> $74 \pm 3$ </td></tr><tr><td>ManiCM</td><td> $0 \pm 0$ </td><td> $48 \pm 11$ </td><td> $82 \pm 7$ </td><td> $100 \pm 0$ </td><td> $96 \pm 5$ </td><td> $84 \pm 2$ </td><td> $33 \pm 3$ </td><td> $62 \pm 5$ </td></tr><tr><td>SDM Policy</td><td> $0 \pm 0$ </td><td> $68 \pm 6$ </td><td> $84 \pm 9$ </td><td> $100 \pm 0$ </td><td> $100 \pm 0$ </td><td> $88 \pm 3$ </td><td> $\mathbf{34} \pm 3$ </td><td> $\mathbf{80} \pm 1$ </td></tr><tr><td>FlowPolicy*</td><td> $100 \pm 0$ </td><td> $50 \pm 6$ </td><td> $74 \pm 5$ </td><td> $95 \pm 3$ </td><td> $100 \pm 0$ </td><td> $81 \pm 4$ </td><td> $29 \pm 10$ </td><td> $69 \pm 6$ </td></tr><tr><td>Ours</td><td> $\mathbf{100} \pm 0$ </td><td> $59 \pm 6$ </td><td> $80 \pm 5$ </td><td> $96 \pm 2$ </td><td> $\mathbf{100} \pm 0$ </td><td> $\mathbf{91} \pm 5$ </td><td> $32 \pm 10$ </td><td> $72 \pm 5$ </td></tr></table>

<table><tr><td rowspan="2">Alg Task</td><td colspan="3">Meta-World (Easy)</td><td colspan="5">Meta-World (Medium)</td></tr><tr><td>Plate Slide Side</td><td>Window Close</td><td>Window Open</td><td>Basketball</td><td>Bin Picking</td><td>Box Close</td><td>Coffee Pull</td><td>Coffee Push</td></tr><tr><td>Diffusion Policy</td><td> $100 \pm 0$ </td><td> $100 \pm 0$ </td><td> $100 \pm 0$ </td><td> $85 \pm 6$ </td><td> $15 \pm 4$ </td><td> $30 \pm 5$ </td><td> $34 \pm 7$ </td><td> $67 \pm 4$ </td></tr><tr><td>3D Diffusion Policy</td><td> $100 \pm 0$ </td><td> $100 \pm 0$ </td><td> $99 \pm 1$ </td><td> $\mathbf{100} \pm 0$ </td><td> $\mathbf{56} \pm 14$ </td><td> $59 \pm 5$ </td><td> $79 \pm 2$ </td><td> $96 \pm 2$ </td></tr><tr><td>ManiCM</td><td> $100 \pm 0$ </td><td> $100 \pm 0$ </td><td> $80 \pm 26$ </td><td> $4 \pm 4$ </td><td> $49 \pm 17$ </td><td> $\mathbf{73} \pm 2$ </td><td> $68 \pm 18$ </td><td> $96 \pm 3$ </td></tr><tr><td>SDM Policy</td><td> $100 \pm 0$ </td><td> $100 \pm 0$ </td><td> $78 \pm 18$ </td><td> $28 \pm 26$ </td><td> $55 \pm 13$ </td><td> $61 \pm 3$ </td><td> $72 \pm 9$ </td><td> $\mathbf{97} \pm 2$ </td></tr><tr><td>FlowPolicy*</td><td> $100 \pm 0$ </td><td> $100 \pm 0$ </td><td> $100 \pm 0$ </td><td> $85 \pm 7$ </td><td> $45 \pm 7$ </td><td> $56 \pm 4$ </td><td> $\mathbf{89} \pm 3$ </td><td> $94 \pm 2$ </td></tr><tr><td>Ours</td><td> $\mathbf{100} \pm 0$ </td><td> $\mathbf{100} \pm 0$ </td><td> $\mathbf{100} \pm 0$ </td><td> $77 \pm 4$ </td><td> $40 \pm 7$ </td><td> $61 \pm 5$ </td><td> $84 \pm 5$ </td><td> $95 \pm 3$ </td></tr></table>

<table><tr><td rowspan="2">Alg Task</td><td colspan="6">Meta-World (Medium)</td><td colspan="3">Meta-World (Hard)</td></tr><tr><td>Hammer</td><td>Peg Insert Side</td><td>Push Wall</td><td>Soccer</td><td>Sweep</td><td>Sweep Into</td><td>Assembly</td><td>Hand Insert</td><td>Pick Out of Hole</td></tr><tr><td>Diffusion Policy</td><td> $15 \pm 6$ </td><td> $34 \pm 7$ </td><td> $20 \pm 3$ </td><td> $14 \pm 4$ </td><td> $18 \pm 8$ </td><td> $10 \pm 4$ </td><td> $15 \pm 1$ </td><td> $0 \pm 0$ </td><td> $0 \pm 0$ </td></tr><tr><td>3D Diffusion Policy</td><td> $\mathbf{100} \pm 0$ </td><td> $79 \pm 4$ </td><td> $78 \pm 5$ </td><td> $23 \pm 4$ </td><td> $\mathbf{92} \pm 4$ </td><td> $\mathbf{38} \pm 9$ </td><td> $100 \pm 0$ </td><td> $\mathbf{28} \pm 8$ </td><td> $\mathbf{44} \pm 3$ </td></tr><tr><td>ManiCM</td><td> $98 \pm 2$ </td><td> $75 \pm 8$ </td><td> $31 \pm 7$ </td><td> $27 \pm 3$ </td><td> $54 \pm 16$ </td><td> $37 \pm 13$ </td><td> $87 \pm 3$ </td><td> $28 \pm 15$ </td><td> $30 \pm 16$ </td></tr><tr><td>SDM Policy</td><td> $98 \pm 2$ </td><td> $\mathbf{83} \pm 5$ </td><td> $\mathbf{83} \pm 4$ </td><td> $25 \pm 2$ </td><td> $90 \pm 6$ </td><td> $32 \pm 15$ </td><td> $100 \pm 0$ </td><td> $24 \pm 14$ </td><td> $34 \pm 24$ </td></tr><tr><td>FlowPolicy*</td><td> $97 \pm 3$ </td><td> $69 \pm 5$ </td><td> $56 \pm 6$ </td><td> $24 \pm 6$ </td><td> $91 \pm 3$ </td><td> $26 \pm 5$ </td><td> $91 \pm 2$ </td><td> $23 \pm 5$ </td><td> $33 \pm 3$ </td></tr><tr><td>Ours</td><td> $95 \pm 2$ </td><td> $70 \pm 4$ </td><td> $64 \pm 11$ </td><td> $\mathbf{27} \pm 4$ </td><td> $88 \pm 4$ </td><td> $23 \pm 4$ </td><td> $\mathbf{100} \pm 0$ </td><td> $18 \pm 3$ </td><td> $40 \pm 4$ </td></tr></table>

$$
0 \pm 0
$$

$$
5 6 \pm 5
$$

$$
0 \pm 0
$$

$$
1 1 \pm 3
$$

$$
4 7 \pm 2
$$

$$
4 3 \pm 7
$$

$$
0 \pm 0
$$

$$
5 5 \pm 2
$$

$$
6 3 \pm 3
$$

$$
1 1 \pm 2
$$

$$
0 \pm 0
$$

$$
\mathbf {9 1} \pm 4
$$

$$
6 7 \pm 0
$$

$$
5 5. 5 \pm 3. 5 8
$$

$$
4 8 \pm 3
$$

$$
1 0 0 \pm 0
$$

$$
8 7 \pm 3
$$

$$
0 \pm 0
$$

$$
7 4 \pm 4
$$

$$
5 7 \pm 0
$$

$$
6 3 \pm 2
$$

$$
7 6. 1 \pm 2. 3 2
$$

$$
1 0 0 \pm 0
$$

$$
5 1 \pm 4
$$

$$
3 7 \pm 1 6
$$

$$
5 7 \pm 4
$$

$$
5 9 \pm 5
$$

$$
8 6 \pm 1 0
$$

$$
6 9. 0 \pm 4. 6 0
$$

$$
0 \pm 0
$$

$$
6 8 \pm 1 0
$$

$$
\mathbf {6 3} \pm 5
$$

$$
4 7 \pm 6
$$

$$
5 3 \pm 1 2
$$

$$
\mathbf {6 0} \pm 5
$$

$$
7 4 \pm 5
$$

$$
\mathbf {5 1} \pm 3
$$

$$
6 6 \pm 7
$$

$$
7 4. 8 \pm 4. 5 1
$$

$$
8 2 \pm 6
$$

$$
1 0 0 \pm 0
$$

$$
7 4 \pm 6
$$

$$
\mathbf {1 0 0} \pm 0
$$

$$
9 2 \pm 4
$$

$$
9 0 \pm 4
$$

$$
7 7. 2 \pm 2. 8 4
$$

$$
7 8. 5 \pm 2. 6 1
$$

1.0e-4, training for 3000 epochs. Evaluation is conducted every 200 epochs, and the best-performing checkpoint is saved. All experiments are conducted on a single NVIDIA A100 GPU.

Result Details. We report the detailed success rates of FreqPolicy on 53 tasks from the Adroit and MetaWorld benchmark in Table 5. The success rate of each task is averaged over experiments conducted with 3 random seeds.

## A.4 More Details about VLA Settings

Implementation. For Diffusion Policy, we leverage the implementation of ‘OpenVLA (fine-tuned) + PD&AC, Cont-Diffusion’ from OpenVLA-OFT (OpenVLA-DP in this paper). For the Flow Matching Policy, we integrate it into the OpenVLA pipeline (i.e., OpenVLA-FlowMatching) based on the original implementation [40]. We train for 150K gradient steps for all models using a batch size of 16 across 4 A100 GPUs, policies receive one third-person image, one wrist camera image, robot proprioceptive state, and language instruction as input. The detailed hyperparameters are shown in Table 6. Please note that our training settings are not consistent with the original paper. In the default settings of OpenVLA-OFT [32], they train for 150K steps, using a batch size of 64 across 8 GPUs. However, in our experiments, considering the computational overhead and resource capabilities, we did not follow the full training settings consistent with the original paper (for example, the results we show are all trained on 2.4M samples, while the original paper trained 9.6M samples or even more), but to show the compatibility and superiority with VLA in comparison with the multi-step policy baselines. In the test, we perform 50-steps inference for OpenVLA-DP, and 1-step and 10-steps inference for OpenVLA-FlowMatching. The inference speed is averaged over the first three episodes of LIBERO-spatial to estimate performance. We follow the default settings in $\mathrm { O p e n V L A } { \cdot } \mathrm { O P T } ^ { 4 }$ for other configurations.

Table 6: The hyperparameter settings for VLA experiments. These values are kept consistent across all methods.

<table><tr><td>Parameter</td><td>Values</td></tr><tr><td>use_film</td><td>True</td></tr><tr><td>use_proprio</td><td>True</td></tr><tr><td>num_images_in_input</td><td>2</td></tr><tr><td>lora_rank</td><td>32</td></tr><tr><td>image_aug</td><td>True</td></tr><tr><td>num_steps_before_decay</td><td>100,000</td></tr><tr><td>max_steps</td><td>150,000</td></tr><tr><td>num_gpus</td><td>4</td></tr><tr><td>batch_size</td><td>4</td></tr><tr><td>learning_rate</td><td>5e-4</td></tr><tr><td>num_trials_per_task</td><td>50</td></tr><tr><td>num_inference_steps</td><td>1 or 10 or 50</td></tr></table>

Compare with OpenVLA-OFT. OpenVLA-OFT is an improved version of OpenVLA [33] that integrates parallel decoding, action chunking, continuous action representation, and an L1 regressionbased learning objective. It achieves state-of-the-art performance on the LIBERO simulation bench mark, significantly improving the average success rate while increasing action generation throughput by 26 times. We reproduced OpenVLA-OFT [32] under the same training and testing settings, and the results are shown in Table 7. Specifically, our method leads or is on par with OpenVLA-OFT in all categories except LIBERO-10, and leads OpenVLA-OFT in average success (94.8% vs. 94.2%). Moreover, even though FreqPolicy leverages an additional policy head, the inference speed remains competitive. (6.05Hz vs. 6.15Hz).

Table 7: Comparison with OpenVLA-OFT on LIBERO simulation benchmark.

<table><tr><td>Method</td><td>NFE</td><td>Spatial (%)</td><td>Object (%)</td><td>Goal (%)</td><td>Long (%)</td><td>Average (%)</td><td>Speed (Hz)</td></tr><tr><td>OpenVLA-DP</td><td>50</td><td>92.0</td><td>75.0</td><td>93.4</td><td>11.8</td><td>68.1</td><td>0.32</td></tr><tr><td>OpenVLA-FlowMatching*</td><td>1</td><td>95.0</td><td>97.6</td><td>96.0</td><td>85.2</td><td>93.5</td><td>5.92</td></tr><tr><td>OpenVLA-FlowMatching*</td><td>10</td><td>96.0</td><td>97.2</td><td>97.8</td><td>83.6</td><td>93.7</td><td>1.26</td></tr><tr><td>OpenVLA-OFT</td><td>-</td><td>96.6</td><td>98.6</td><td>91.0</td><td>90.6</td><td>94.2</td><td>6.15</td></tr><tr><td>OpenVLA-FreqPolicy</td><td>1</td><td>97.0</td><td>98.6</td><td>96.0</td><td>87.6</td><td>94.8</td><td>6.05</td></tr></table>

## A.5 More Details about Real-World Tasks

Implementation. The real-world experiments are based on the open-source LeRobot framework <sup>5</sup>. LeRobot is designed to support real-world robotics research by providing models, datasets, and tools in PyTorch. It includes state-of-the-art methods, such as Diffusion Policy, which we use as a baseline, that have demonstrated strong transfer capabilities to real-world settings, with a focus on imitation learning. We convert our collected real-world data into the LeRobot-supported format and integrate both the Flow Matching Policy and our proposed FreqPolicy into the framework. This enables training directly on real data, facilitating practical deployment and inference. Training was conducted on a single NVIDIA A100 GPU. To ensure a fair comparison, all policy models were trained with aligned hyperparameters provided in Table 8. We also provide additional demonstration images for the three tasks in Fig. 5, as well as demonstration videos in the supplementary materials to facilitate better understanding.

Table 8: The hyperparameter settings for real-world experiments. These values are kept consistent across all methods.

<table><tr><td>Parameter</td><td>Values</td></tr><tr><td>input_images_Frankainput_images_UR</td><td>[camera_front, camera_wrist][camera_left, camera_wrist]</td></tr><tr><td>use_robot_state</td><td>True</td></tr><tr><td>image_size</td><td>480 × 480</td></tr><tr><td>vision_backbone</td><td>ResNet-18</td></tr><tr><td>n_obs_steps</td><td>2</td></tr><tr><td>horizon</td><td>48</td></tr><tr><td>n_action_steps</td><td>48</td></tr><tr><td>num_inference_steps</td><td>1 or 10</td></tr><tr><td>batch_size</td><td>64</td></tr><tr><td># of training steps</td><td>100,000</td></tr><tr><td>optimizer</td><td>Adam</td></tr><tr><td>learning_rate</td><td>1e-4</td></tr><tr><td>weight_decay</td><td>1e-6</td></tr><tr><td>grad_clip_norm</td><td>10.0</td></tr></table>

Evaluation Criteria. The initialization of the task environment plays a critical role, as it directly affects whether the robot can complete a task under the guidance of the policy model. Moreover, the conditions leading to task failure vary across different tasks. To facilitate a deeper understanding of the real-world experiments presented in this work, we provide target and object placement rules for the three tasks, along with detailed descriptions of the conditions under which task execution fails:

1) Fruit Sorting.

Target: Pick up bananas, avocados, and mangoes and put them into the basket in order.

Object placement rules: During execution, the fruit basket remains stationary in a fixed position. The spatial order of the three fruits—banana, avocado, and mango, from right to left—remains unchanged. However, in each trial, the exact placement of the fruits is randomly initialized within the front half of the basket.

![](images/a394620ca4fd9ed10bb532c410e73d057f66cf2eb53d454e485ce33a530ab420.jpg)

(a) Demonstrations of real-world Task\_1. This task is carried out on the Franka platform. The robot arm is tasked with sequentially picking up three different fruits—banana, avocado, and mango—and placing them into a basket. The fruits are positioned one at a time in random locations in front of the basket, and the task is deemed successful only if all three fruits are correctly picked up and placed in the basket.  
![](images/835ac22062550f85d2ade45d9e7f2e6e130ed0da13f451329bf749f04dd3f045.jpg)

(b) Demonstrations of real-world Task\_2. This task is carried out on the Franka platform. The robot arm must first rotate to the appropriate angle, open its gripper, and insert it into the drawer handle to pull the drawer open. It then sequentially picks up two plush toys placed randomly on the table and places them inside the drawer. Finally, it closes the gripper and pushes the handle to shut the drawer. This requires the robot to accurately perceive the drawer handle’s start and end positions and be robust to the random positions of the toys.  
![](images/535759f7c285b6d9eb1bd56eb74cf8dc56b434bde469acbbd0fd5e33b67898dd.jpg)  
(c) Demonstrations of real-world Task\_3. This task is carried out on the UR platform. The robot arm first closes its gripper and attempts to tap a specific spot on the trash bin lid with the tip of the gripper to trigger it to pop open. Then, with the gripper open, the arm sequentially picks up two pieces of soft food waste—a half-steamed bun and a piece of bread—randomly placed on the table. Finally, the gripper closes again, moves behind the lid, and pushes it while tapping once more to fully close the bin. This process requires precise tapping actions and robustness to the random placement of the food waste.  
Figure 5: Demonstrations of three long-horizon real-world tasks. Following a consistent protocol, 300 episodes are collected for each task for training, after which the trained policies are evaluated. The numbers in each image indicate the sequential steps in the task execution process.

Cases oftask executionfailed:

\- If any fruit falls while picking it up, it will be considered a failure case.

\- If the same fruit fails to be picked up more than 3 times in total, it will be considered a failure case.

\- If all fruits are not completely put into the basket, it will be considered a failure case.

## 2) Toy Organization.

Target: Open the drawer, then put the two dolls into the drawer one by one, and close the drawer. Object placement rules: During each execution, the cabinet remains in a fixed position, while two dolls are reinitialized with their relative spatial positions preserved.

Cases oftask executionfailed:

\- If the drawer is not successfully opened, it is considered a failure case.

\- If the drawer is not closed after all the dolls are placed, it is considered a failure case.

\- If all the dolls are not placed in the drawer, it is considered a failure case.

\- If the cumulative number of failures to pick up the same doll is 3 or more, it is considered a failure case.

## 3) Trash Disposal.

Target: Tap the trash can lid to make it pop open, put the food waste (steamed buns and bread) into the trash can one by one, and tap the lid again to close it.

Object placement rules: During the execution, the position of the trash can is fixed, the relative spatial positions of the two types of food waste remain unchanged, and the trash positions are reinitialized for each execution.

Cases oftask executionfailed:

\- If the lid is not closed after picking up the food waste, it is considered a failure case.

\- If the number of failed attempts to pick up the same food waste exceeds 3 times, the machine will be deemed as a failed case.

\- If the lid is closed before all the food wastes are put into the trash can, it will be considered a failure case.

\- If the lid is not opened and food waste is taken, it will be deemed as a failure case.