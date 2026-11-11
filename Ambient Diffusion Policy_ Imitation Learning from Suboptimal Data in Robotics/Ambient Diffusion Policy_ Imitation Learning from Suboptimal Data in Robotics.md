# Ambient Diffusion Policy: Imitation Learning from Suboptimal Data in Robotics

Adam Wei\* Nicholas Pfaff† Thomas Cohn† Arif Kerem Dayi
Constantinos Daskalakis Giannis Daras Russ Tedrake

https://ambient-diffusion-policy.github.io/

MIT

![](images/6b52c966e97d8e853cacf7bf4c06599f6d37e3f878fc9765d70a21b419c6fd1e.jpg)  
Figure 1: Training Algorithm for Ambient Diffusion Policy. High-quality data can be used during training at all diffusion times. Suboptimal data is restricted to training at low or high diffusion times $t \in [0, t_{\max}) \cup (t_{\min}, T]$ . Task visualizations. We evaluate Ambient Diffusion Policy on four types of action suboptimality, six different tasks, and scale to the Open X-Embodiment (OXE) [1].

Abstract: We propose Ambient Diffusion Policy, a simple and principled method for imitation learning from suboptimal data in robotics. High-quality, task-specific robot data is expensive and time-consuming to collect, while suboptimal datasets with lower-quality or out-of-distribution demonstrations are abundant. Existing methods that co-train on both data sources in robotics often fail to separate the meaningful and the harmful features in the suboptimal samples. In contrast, our method extracts only the useful features by introducing a new axis to co-training in robotics: noise-dependent data usage. Ambient Diffusion Policy restricts the contribution of suboptimal data during training to only the high and low diffusion times. To rigorously justify our approach, we first observe that robot action data exhibits a spectral power law. This induces two important properties on the optimal Diffusion Policy that we exploit: a global-to-local hierarchy and locality. We theoretically formalize this discussion using a simplified model. Our experiments validate Ambient Diffusion Policy on four types of suboptimal action data (noisy trajectories, sim-to-real gap, task mismatch, and large-scale data mixtures) across six tasks. The results show that it effectively learns from arbitrary sources of suboptimal data. Notably, it outperforms existing co-training baselines by up to 33% when scaled to Open X-Embodiment—a large dataset with heterogeneous data quality and unstructured distribution shifts. Overall, Ambient Diffusion Policy increases the utility of suboptimal demonstrations and expands the set of usable data sources in robotics.

## 1 Introduction

The training corpus of nearly every large-scale robot policy spans different tasks, real and simulated environments, embodiments, and even modalities $[2, 3, 4, 5, 1, 6, 7, 8, 9]$ . One reason for this data heterogeneity is that high-quality, task-specific robot data is expensive and time-consuming to collect; it requires skilled teleoperators and well-tuned low-level controllers. In contrast, suboptimal data is abundant. Any real-world data collection effort naturally produces failures and trajectories of differing quality $[10, 1, 8, 11]$ . Out-of-distribution (OOD) data sources are also plentiful and widely available $[1, 8]$ . These include simulation $[12, 13]$ , cross-embodied data $[1]$ , and ego-centric video $[14, 15]$ . Practitioners often draw from data sources of varying quality $[1, 8, 5, 16, 11]$ to create massive pretraining sets, yet methods for learning from arbitrary suboptimal or shifted distributions are underexplored in robotics $[5, 2, 12, 16]$ . We propose a simple and principled method for training robot policies that can leverage suboptimal and OOD datasets.

The simplest way to handle heterogeneous datasets with mixed quality is to discard the lowest-quality samples $[5, 2, 17, 18]$ , but this data filtering is wasteful: even suboptimal samples contain useful learning signal $[19]$ . The most common alternative in robotics is to “co-train” on everything while down-weighting suboptimal datasets $[2, 3, 4, 5, 6, 12, 13]$ . However, training a model to sample from a mixture with low-quality distributions fundamentally biases the policy $[20, 10]$ . Finetuning is a complementary method $[2, 11, 18, 4]$ , but can be insufficient alone (Section 7.5).

Our key intuition is that suboptimal and high-quality samples differ in some features, but align in others. For example, non-expert teleoperators may share the same high-level plan as experts, but exhibit less precise manipulation skills. Conversely, pick-and-place data may be useful for grasping primitives, but encode the wrong task.

We turn this intuition into an algorithm by observing that robotic data exhibits a spectral power law $[21]$ . We show that this spectral structure induces a global-to-local hierarchy in action diffusion and a locality property $[22]$ in the optimal denoisers. Concretely, Diffusion Policy $[23]$ learns high-level planning at high noise and motion primitives at low noise. Thus, a policy can selectively learn useful features from suboptimal data by using it only at noise levels where it aligns with the target distribution. This unlocks a new design axis for co-training in robotics: noise-dependent data usage.

To this end, we propose Ambient Diffusion Policy, a principled algorithm for training Diffusion Policies on suboptimal robot data. Each suboptimal sample can only contribute to training at the high or low diffusion times where it aligns with the target data. Our method extends the Ambient Diffusion Omni framework $[24, 19]$ (or “Ambient” for short), which has been applied to computer vision and protein design $[25]$ . Our contributions are as follows:

\- Properties of Robot Data. We empirically demonstrate that robot action data exhibits a spectral power law, which induces a global-to-local hierarchy and locality in action diffusion. These properties make Ambient Diffusion Omni well-suited for robotics (Section 4).

\- Ambient Diffusion Policy. We propose a simple and principled method for learning from suboptimal data that requires just a single change to Diffusion Policy's data sampler (Section 6).

\- Generality. Ambient outperforms baselines when training on three common types of action suboptimality: noisy demonstrations, sim-to-real gap, and task mismatch (Section 7).

\- Scale. When trained on Open X-Embodiment [1]—a large dataset with mixed data quality and unstructured distribution shifts—Ambient outperforms the co-training baseline by up to $33\%$ on two real-world tasks. Additionally, Ambient continues to improve as we scale the amount of suboptimal data in the training mixture, whereas co-training plateaus (Section 8).

\- Theory. We prove that for a simple theoretical model, the spectral power law implies fast contraction through noise (Theorem 1) and locality (Theorem 2) of the optimal policies. These contributions justify our algorithm and advance the theoretical foundations of the broader Ambient Diffusion framework (Section 5).

## 2 Related Work

Dataset Re-weighting. The prevailing method for training on heterogeneous datasets in robotics is to re-weight each dataset's contribution to the loss [2, 3, 4, 5, 6, 7, 12, 13, 26]. In our problem setting, this method would optimize the loss

$$
\mathcal {L} _ {\text { co - training }} (\theta) = \alpha   \mathcal {L} _ {\mathcal {D} _ {p}} (\theta) + (1 - \alpha)   \mathcal {L} _ {\mathcal {D} _ {q}} (\theta),\tag{1}
$$

where $L_{D_{p}}$ and $L_{D_{q}}$ denote the loss on the target and suboptimal datasets, and $\alpha \in [0,1]$ is the re-weighting parameter. In the remainder of the paper, we use “co-training” to refer to this standard approach. A core challenge is weight selection for each dataset [7, 5, 27, 28, 29, 16, 30]. But even with the optimal weights, training a policy to sample from a mixture with suboptimal distributions biases the policy [20, 10]. Our method avoids this issue by only training on suboptimal samples in specific intervals of the diffusion process. Importantly, these intervals have precise interpretations and can be automatically annotated.

Data Filtering. A related method is data filtering, which rejects suboptimal samples entirely $[5, 2, 17, 31, 32]$ . Our experiments and prior work $[19, 11, 10]$ show that data filtering is wasteful. For example, Dexora $[17]$ improves policy performance by discarding 80% of its training demonstrations using a kinematic smoothness heuristic. Section 7.1 shows the opposite for Ambient Diffusion Policy: training on additional non-smooth demonstrations improves performance while retaining the policy's smoothness. Similarly, OpenVLA's Magic Soup++ mixture $[5]$ includes only 27 of the datasets in OXE $[1]$ ; Section 8 shows that our method can still extract useful learning signal from the discarded datasets.

Learning from Suboptimal or OOD Data. Prior works studied learning from corrupted data $[33, 24, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44]$ ; however, many require multiple training runs $[45, 46, 47]$ or rely on knowing the form of the corruption. The latter is unrealistic in robotics. For example, the sim-to-real gap has no closed-form description. Recent works in robotics have also begun exploring training methods for suboptimal data. $\pi_{0.7}$ $[10]$ labels data quality and LDA-1B $[48]$ relegates low-quality trajectories to an auxiliary dynamics objective. Other approaches learn invariant representations between the target and suboptimal data $[49, 50]$ , but are challenging to scale $[12]$ . Tangkaratt et al. $[20]$ and Liu et al. $[51]$ propose methods for robust imitation learning; however, they focus on ignoring corrupted samples instead of learning from them. They also require over half the data to be high-quality. Concurrent work explored a similar idea to ours for hand-tracking data $[52]$ . In contrast, our method learns from arbitrary mixtures of suboptimal action data, scales to large real-world datasets $[1]$ , has theoretical grounding, and is simple to implement.

## 3 Background

## 3.1 Robot Imitation Learning and Diffusion Policy

Given a dataset $\mathcal{D}_{p}=\{(O^{(i)},A^{(i)})\}_{i=1}^{n_{p}}$ of observation-action pairs, imitation learning aims to train a policy $\pi(A\mid O)$ that approximates the conditional distribution $p(A\mid O)$ [23]. Here, A and O represent chunks of future actions and recent observations [53]. Diffusion Policy [23] samples from $\pi(A\mid O)$ with a conditional denoising process over the action space [54]. Concretely, it learns a family of denoisers $\{h_{\theta}(A_{t},O,t)\}_{t\in[0,T]}$ that predict the clean action $A_{0}$ from a noisy version $^{2}$

$$
A _ {t} = A _ {0} + \sigma (t)   Z, \quad Z \sim \mathcal {N} (0, I),\tag{2}
$$

where T is a sufficiently large constant, $\sigma(t)$ is a strictly increasing function known as the noise schedule, and $\sigma(0)=0$ . For brevity, we write $\sigma_{t}:=\sigma(t)$ .

These denoisers are trained by minimizing the denoising loss $[54]$ (or a reparametrization of it, as in flow matching $[56]$ ):

$$
\mathcal {L} (\theta) = \mathbb {E} _ {t, (O, A _ {0}), A _ {t} | A _ {0}} \left[ \| h _ {\theta} (A _ {t}, O, t) - A _ {0} \| ^ {2} \right],\tag{3}
$$

where $t \sim \mathcal{U}[0, T]$ , $(O, A_{0}) \sim \mathcal{D}_{p}$ , and $A_{t}$ is sampled from Eq. (2). For a sufficiently expressive $h_{\theta}$ , the minimizer of Eq. (3) is the conditional expectation $h_{\theta}^{*}(A_{t}, O, t) = \mathbb{E}[A_{0} \mid A_{t}, O]$ . Given access to the conditional expectations, one can sample from the target distribution $\pi(A \mid O)$ by running a reverse diffusion process [23, 54, 57]. Appendix E.2 provides implementation details for Diffusion Policy.

## 3.2 Problem Setting

Suppose we have access to a (small) dataset, $\mathcal{D}_{p}=\{(O^{(i)},A^{(i)})\}_{i=1}^{n_{p}}$ , from a target distribution, p, and a much larger dataset, $\mathcal{D}_{q}=\{(O^{(i)},A^{(i)})\}_{i=1}^{n_{q}}$ , from a shifted or suboptimal distribution, q. As discussed in the Introduction, this is a common problem setting in robotics.

Problem Statement: Our goal is to leverage the vast pool of suboptimal samples in $D_{q}$ to improve the estimation of the denoisers, $E_{p}[A_{0} \mid A_{t}, O]$ , on the user-defined target distribution.

The definitions of p and q can be arbitrary. In general, $D_{p}$ is a dataset that a practitioner considers “high-quality” for a downstream application. In this paper, we define $D_{p}$ as expert demonstrations on the target robot, environment, and task; however, our framework applies equally well to any other definition, including heuristic measures [58] or human labels [10]. In Section 7, we construct $D_{q}$ to contain controlled distribution shifts. In Section 8, we scale $D_{q}$ to Open X-Embodiment (OXE) [1].

## 3.3 Ambient Diffusion Omni

This Problem Statement is not unique to robotics. Ambient Diffusion Omni was proposed by Daras et al. [19] to address a similar problem in the vision domain. Their key idea is to allow suboptimal samples from $q$ to be used only at certain diffusion times during training. The algorithm assigns two parameters to samples from $\mathcal{D}_q$ , denoted $t_{\mathrm{min}}$ and $t_{\mathrm{max}}$ . Samples from $\mathcal{D}_q$ can only contribute to the learning at diffusion times $t \in [0, t_{\mathrm{max}}) \cup (t_{\mathrm{min}}, T]$ .

The algorithm leverages noise as a contraction mechanism: for any distributions $p$ and $q$ supported on a subset of $\mathbb{R}^d$ with diameter $D$ , the authors show that

$$
\mathrm{d} _ {\mathrm{TV}} (p _ {t}, q _ {t}) \leq \mathrm{d} _ {\mathrm{TV}} (p, q) \cdot \frac {D}{2 \sigma_ {t}},\tag{4}
$$

where $d_{TV}$ is the Total Variation distance, $p_{t}=p\circledast\mathcal{N}(0,\sigma_{t}I)$ , and $q_{t}=q\circledast\mathcal{N}(0,\sigma_{t}I)$ .

Simply put, noise erases distribution differences, i.e. p and q contract towards one another. Hence, there exists a sufficiently high diffusion time, $t_{min}$ , for which noisy actions from both the high-quality and suboptimal distributions become effectively indistinguishable. Consequently, $D_{q}$ can safely contribute to training at high noise when $t \in (t_{\min}, T]$ . The utility of q is highest when $t_{min}$ is small (i.e. when p and q contract quickly). We discuss sufficient conditions for fast contraction and justification for using $D_{q}$ for $t \in [0, t_{\max})$ in subsequent sections.

Section 4 examines structural properties of action data that make this method well-suited for robotics. Section 5 theoretically formalizes the discussion in both Section 4 and prior work $[19, 25]$ . Section 6 outlines our algorithm. Despite the in-depth motivation, the resulting method is remarkably simple: it requires a single change to the Diffusion Policy $[23]$ data sampler. We conclude with experiments that demonstrate the generality and scalability of our method.

## 4 Distributional Properties of Robot Data

We empirically show that robot data exhibits a spectral power law. This spectral structure is special for two reasons. First, it is not universal: it is absent in natural audio $[59, 60]$ , stellar spectra $[61]$ , and Poisson point processes $[62]$ . Second, we show that it makes Ambient Diffusion effective: it implies a global-to-local hierarchy (Figure 2), fast contraction through noise (Theorem 1), and a locality property (Theorem 2). Hence, Ambient Diffusion is well-suited for robotics.

![](images/e1aaeb4b8e6e6a98195242a06ba89eeb01f15c4a0b14ba0874fffa8ba193f7cb.jpg)  
Figure 2: The spectral power law induces a global-to-local hierarchy in action diffusion. This is analogous to the “coarse-to-fine” hierarchy in image diffusion [60]. (a) $t \geq t_{2}$ (global planning): the policy commits to a high-level path through the maze. $t < t_{2}$ (local refinement): it refines the plan to be collision-free at $t_{1}$ and smooth at $t_{0}$ . (b) $t_{3}$ : the policy plans to move a block to the right bin. $t_{2}$ : it plans which block to grasp. $t < t_{2}$ : it refines the grasping motion.

![](images/8c4b469f8b4bc80c3f8ddd30f2c6cedc675dd7b0b6897621ba4a80c8d0ca533f.jpg)  
(a) Spectral power law in robot action data. The power spectral density for OXE [1], a collection of 2.4M robot episodes across 70+ datasets, follows a power law. Figure 11 shows that this spectral power law is a more general property of robot data.

![](images/bbe27a1e93ace63dd93d41246aa46d778263fe0c7cc097f7c80db826009fd8b7.jpg)  
(b) Locality in robot action data. Visualizing the sensitivity of the 8th action estimate to its neighboring actions (Eq. (83)). At low noise, the 8th action estimate is most sensitive to its temporal neighbors.

## 4.1 Spectral Power Law in Robotics

The power spectral density (PSD) of a signal measures its energy at each frequency component, f. The signal exhibits a spectral power law if its PSD decays approximately as $|f|^{-\alpha}$ , where $\alpha > 0$ . In other words, its energy is concentrated in low-frequency components. A formal definition extended to random vectors is presented in Definition 1. Figures 3a and 11 show that robot action data exhibits a spectral power law.

Implications for Diffusion Policy $[23]$ : When a spectral power law exists, Dieleman $[60]$ showed that denoisers generate low-frequency features at high noise, and high-frequency features at low noise. Figure 4 illustrates that in robotics, low frequencies encode global features (e.g. the high-level plan), while high frequencies encode local features (e.g. grasping primitives and smoothness). Thus, Diffusion Policies exhibit a global-to-local hierarchy: they learn global planning at high noise and local motion primitives at low noise. Appendix D provides further experimental evidence.

This hierarchy unlocks a new design axis for co-training algorithms in robotics: noise-dependent data usage. Diffusion Policies learn different features of the actions at each noise level; thus, suboptimal samples should only contribute to training at noise levels where p and q align.

Implications for Ambient Diffusion Policy: Due to the spectral power law, adding Gaussian noise to robot data acts as a high-frequency mask (Figure 4). Thus, if differences between p and q are concentrated in a high-frequency tail (i.e. low-level actions), then $t_{min}$ will be small. We formalize this in Theorem 1. By only learning from suboptimal samples when $t > t_{min}$ , policies can learn the useful global features in $D_{q}$ and ignore the local suboptimality (which is masked by noise).

![](images/7c976d646db685e0c30a8f8db935f3034f5f8528276bca3f18320ea0bfd3edf2.jpg)  
Figure 4: The spectral power law makes Gaussian noise act as a high-frequency mask. (a) Noisy trajectory visualizations. (b) PSD analysis for the noisy trajectories in the x-dimension at each diffusion time: the gray regions indicate masked frequencies, which have low SNR. As t increases, adding Gaussian noise destroys the signal in the trajectory, starting from the local (high-frequency) features first. This effect is visible in both the noisy trajectories and the PSD plots.

Fortunately, the suboptimality in many robot datasets is in the local actions. For instance, non-expert and expert demonstrations may aim to achieve the same goal, but differ in the quality of the low-level motions. Similarly, in sim-to-real co-training or cross-embodied data, the high-level plan may be shared while the precise contact dynamics or feasible motions may differ.

## 4.2 Locality in Robotics

A denoiser exhibits “locality” if its output at each coordinate depends primarily on a small receptive field in the noisy input $[22, 19]$ .³ In robotics, this means that each action output from the denoiser is most sensitive to its temporal neighbors in the noisy input. In Theorem 2, we prove that the spectral power law can imply locality in the optimal denoisers. Indeed, Figure 3b empirically identifies this phenomenon in robotics. Intuitively, this is because these denoisers focus on resolving local motion primitives, so they do not need to attend to distant actions in the input.

Now suppose that p and q share the same local motion primitives but differ globally. Due to locality, there exists a sufficiently small $t_{max}$ such that for all $t < t_{max}$ , p and q agree within the receptive field of the optimal denoiser. Section 5 formalizes this statement. Section 7.4 leverages locality to learn local grasping primitives from data at low noise, even when its task-level structure is incorrect.

Remark: A spectral power law and locality exists in all the examined robot datasets (Table 4). This suggests that these are general properties of robot data. Section 6 provides methods for estimating $t_{min}$ and $t_{max}$ to leverage these properties in Ambient Diffusion Policy.

## 5 Theoretical Foundations

We prove two theorems that extend the theoretical foundations of Ambient Diffusion and rigorously justify our method for robotics. Prior works $[19]$ empirically identify the spectral power law and locality as important properties. We prove that for a simplified model, the former implies both fast contraction through noise (Theorem 1) and locality of the optimal denoiser (Theorem 2). This establishes the spectral power law as the single fundamental property underlying the framework.

Both theorems are proved for zero-mean stationary Gaussians. We adopt this setting for two reasons: 1) it simplifies the bounds, and 2) it is a natural model for latent-space diffusion, which has been proposed in robotics [63], and is common in generative modeling more broadly [64]. Extending the analysis to general distributions is left to future work. Proofs are presented in Appendix A.

We begin by formally defining the spectral power law for general random vectors.

Definition 1 (Spectral Power Law). Let X be a zero-mean, stationary random vector on $R^{N}$ , and let $S_{X}(f) := \mathbb{E}\left[|\mathcal{F}X)(f)|^{2}\right]$ denote its power spectral density in the orthonormal Fourier basis. We say that the distribution of X follows a spectral power law if there exist a constant C > 0, a rolloff exponent $\alpha > 1$ , and a cutoff frequency $f_{0} \geq 1$ such that the high-frequency spectrum satisfies:

$$
S _ {X} (f) = C | f | ^ {- \alpha} \quad \text { for   all } | f | \geq f _ {0}.
$$

The exact spectral behavior below the cutoff $f_0$ is permitted to be arbitrary but bounded.

## 5.1 Contraction Through Noise

Theorem 1 states that a spectral power law and low-frequency alignment between $p$ and $q$ imply fast contraction through noise. Moreover, this contraction is independent of the original distance.

Theorem 1 (Spectral Power Law implies fast contraction through noise). Let $p_0 = \mathcal{N}(0, \Sigma_p)$ and $q_0 = \mathcal{N}(0, \Sigma_q)$ be zero-mean stationary Gaussians on $\mathbb{R}^N$ with power spectra $S_p, S_q$ . Fix a cutoff frequency $f^\star \geq 1$ , an exponent $\alpha > \frac{1}{2}$ , and a constant $C > 0$ . Assume:

1. (Low-frequency agreement.) $S_{p}(f) = S_{q}(f)$ for every $f \leq f^{\star}$ .

2. (Power-law tail.) $\max(S_p(f), S_q(f)) \leq C f^{-\alpha}$ for every $f$ such that $f^{\star} < f \leq \lfloor N/2 \rfloor$ .

Let $p_t := p_0 * \mathcal{N}(0, \sigma_t^2 I)$ and $q_t := q_0 * \mathcal{N}(0, \sigma_t^2 I)$ . If

$$
\sigma_ {t} ^ {2} \geq C (f ^ {\star}) ^ {- \alpha},\tag{★}
$$

then

$$
\boxed {\mathrm{d} _ {\mathrm{KL}} (p _ {t} \| q _ {t}) \leq \frac {4 C ^ {2}}{(2 \alpha - 1) \sigma_ {t} ^ {4} (f ^ {\star}) ^ {2 \alpha - 1}}}
$$

$$
\boxed {\mathrm{d} _ {\mathrm{TV}} (p _ {t}, q _ {t}) \leq \frac {\sqrt {2} C}{\sigma_ {t} ^ {2} \sqrt {(2 \alpha - 1) (f ^ {\star}) ^ {2 \alpha - 1}}}}
$$

In contrast, Eq. (4) depends on the original TV distance (which can be arbitrarily close to 1) and decays only as $1/\sigma$ . For our simple theoretical model, the spectral power law enables fast contraction and, consequently, high utilization of the suboptimal data in Ambient Diffusion.

## 5.2 Spectral Power Law Implies Locality

The next theorem shows that the spectral power law implies locality for the optimal denoiser at low noise levels.

Theorem 2 (Spectral Power Law implies locality of the optimal denoiser). Let $X_0 \sim \mathcal{N}(0, \Sigma)$ be a zero-mean stationary Gaussian on $\mathbb{R}^N$ whose power spectrum $S(f)$ follows a spectral power law with constant $C > 0$ and exponent $\alpha > 1$ . Let $X_t = X_0 + \sigma_t Z$ be the variance-exploding noisy observation, and $h_t^\star(x) := \mathbb{E}[X_0 \mid X_t = x]$ denote the MMSE optimal denoiser.

Let $M_i$ be a circulant mask that sets to zero all coordinates of the input $x$ at a circular distance strictly greater than $L$ from index $i$ . Then, for any mask size $1 \leq L \leq \lfloor N/2 \rfloor$ , the absolute error in the $i$ -th coordinate between the full optimal denoiser and the masked denoiser is bounded by:

$$
\boxed {\left| (h _ {t} ^ {\star} (x)) _ {i} - (h _ {t} ^ {\star} (M _ {i} x)) _ {i} \right| \leq \frac {\alpha N \| x \| _ {\infty}}{8 L}.}
$$

In short, at low noise levels the optimal denoiser is functionally myopic: it can reconstruct the signal by observing only a local spatial neighborhood, safely ignoring the global structure of the input. This is precisely the property that Ambient Diffusion Policy exploits to learn local action primitives from suboptimal data, even when its global structure is incorrect (Section 7.4).

## 6 Ambient Diffusion Policy: Method

Since robot data exhibits a spectral power law, learning from suboptimal data in robotics with Ambient is empirically and theoretically justified. We now present our algorithm: Ambient Diffusion Policy. It has two phases: a) data annotation, and b) training.

## 6.1 Phase 1: Dataset annotation

This phase annotates $t_{max}$ and $t_{min}$ for the suboptimal samples from $D_{q}$ . The simplest method is to perform a hyperparameter sweep. A more principled solution is to use a classification network. In particular, $\sigma_{t_{min}}$ corresponds to the minimal amount of noise required to make the distributions $p_{t}$ and $q_{t}$ indistinguishable [19]. Hence, one way to find this noise level is to train a classifier, $c_{\phi}(A_{t}, t)$ that predicts the probability that the noisy action $A_{t}$ came from $p_{t}$ (as opposed to $q_{t}$ ). Then, $t_{min}$ is the minimum noise level required to fool the classifier into classifying samples from $q_{t}$ as samples from $p_{t}$ (i.e. the classifier cannot reliably distinguish $p_{t}$ and $q_{t}$ ). Formally, for some small $\tau > 0$ , we assign

$$
t _ {\min} = \inf \left\{t: \mathbb {E} _ {(O, A _ {0}) \sim q, A _ {t} | A _ {0}} \left[ c _ {\phi^ {*}} (A _ {t}, t) \right] > 0. 5 - \tau \right\}.\tag{5}
$$

Figure 13 visualizes this method for $\tau = 0.05$ ; details are in the Appendix. If the classifier is well-trained, we show that this annotation method guarantees distributional closeness between $p_t$ and $q_t$ for $t > t_{\mathrm{min}}$ (Theorem 4).

This classifier can also be used to annotate samples at different granularities (e.g. one $t_{min}$ per dataset vs one $t_{min}$ per sample); Appendix G.3 and Figure 19f provide additional details. A similar classifier-based approach can be used to annotate $t_{max}$ [19], although we do not use it in this paper.

## 6.2 Phase 2: Training

The training loop for Ambient Diffusion Policy is the same as Diffusion Policy [23] with one required change: a custom data sampler. The custom sampler first draws a batch of diffusion times $t^{(i)} \sim \mathcal{U}[0, T]$ . Afterward, it draws admissible samples from $D_{p} \cup D_{q}$ . Samples from $D_{p}$ are always admissible; suboptimal samples from $D_{q}$ are admissible only if $t^{(i)} \in [0, t_{\max}) \cup (t_{\min}, T]$ . Figure 1 visualizes the intervals where each dataset can contribute to training. Inference is identical to Diffusion Policy. We defer the full training pseudocode to Appendix E.1 to prevent implementation details from obscuring the essence of the method.

Variants of Ambient Diffusion Policy. Ambient Diffusion Policy provides an (optional) alternative loss function $[24]$ , which is discussed in Appendix C. Additionally, there is a variation of the algorithm that uses the classifier from Phase 1 for rejection sampling at training-time. We provide pseudocode and theoretical justification in Appendix E.3 and leave experimental verification of this variant to future work.

Table 1: Maze and Neural Motion Planning results (1000 trials each). In both settings, Ambient Diffusion Policy achieves the highest success rate and is substantially smoother than co-training.

<table><tr><td>Training Algo.</td><td>Success Rate ↑</td><td>Smoothness ↓ (Eq. (84))</td></tr><tr><td>Data Filtering</td><td>57.5+3.1-3.1%</td><td>31.9</td></tr><tr><td>Co-train (α* = 0.019)</td><td>99.4+0.4-0.7%</td><td>62.2</td></tr><tr><td>Ambient (σt*min = 0.074)</td><td>99.5+0.3-0.7%</td><td>31.0</td></tr></table>

<table><tr><td>Training Algo.</td><td>Success Rate ↑</td><td>Smoothness ↓ (Eq. (84))</td></tr><tr><td>Data Filtering</td><td> $46.0^{+3.1}_{-3.1} \%$ </td><td>3.9</td></tr><tr><td>Co-train( $\alpha^{*} = 0.091$ )</td><td> $59.9^{+3.1}_{-3.1} \%$ </td><td>42.7</td></tr><tr><td>Ambient( $\sigma_{t*}_{\min} = 0.025$ )</td><td> $65.9^{+2.9}_{-3.0} \%$ </td><td>31.4</td></tr></table>

(a) 2D maze.  
![](images/57ef4999da7ce62b0579cb2c79c98780cf79a3613c687d196d8d774697808a02.jpg)  
(a) 2D maze.

(b) 7-DoF neural motion planning.  
![](images/ddb56a94cb0361ac8c5045738b0b5814319860239a3dc53d8483963466e15af5.jpg)  
(b) 7-DoF neural motion planning.  
Figure 6: The “smoothness vs success rate” Pareto frontier for Ambient (blue) dominates co-training (red) in both the (a) maze and (b) neural motion planning experiments. This trade-off is controlled by $\sigma_{t_{min}}$ for Ambient and $\alpha$ for co-training.

## 7 Controlled Experiments

We evaluate Ambient Diffusion Policy on three distribution shifts in robot action data to illustrate its generality: noisy trajectories, sim-to-real gap, and task mismatch. This section compares our method against data filtering, co-training, and finetuning. Appendix F provides experimental details.

## 7.1 Noisy Trajectories

Consider a 2D point robot that must navigate between random start and goal positions in a maze environment (Figure 1a). A policy rollout is successful if it reaches an $\epsilon$ -ball around the goal within a time limit without collision. We measure a policy's global understanding of the maze using success rate, and its local smoothness using average squared acceleration (Eq. (84)).

$D_{p}$ contains 50 smooth trajectories from GCS [65]; $D_{q}$ contains 5,000 cheaper but jittery trajectories from RRT [66]. Figure 5 visualizes the datasets. This mimics real-world sources of

![](images/f90f0562d0ffd978f7fcd838f1f785336503f925bde8ab24422c97f1e601853b.jpg)

![](images/176689d9c6d51aa8066d5833312e34dc88f5448863d729d58413e72b63f32c09.jpg)  
Figure 5: GCS (left) vs RRT (right) data. The high-quality GCS [65] trajectories are smoother and shorter than their RRT [66] counterparts.

suboptimality, such as teleoperation with low-quality hardware or noisy control stacks [67].

The policy trained with data filtering is smooth, but performs poorly (Table 1a). Co-training and Ambient both perform well (99.0%+ success), but the Ambient Diffusion Policy is 2× smoother. This is because RRT's jittery behavior is a local property; thus, excluding it from training at low diffusion times prevents the policy from learning its non-smooth behavior. Figure 14 (Appendix G.1)

Table 2: Main planar pushing results (200 trials). All Ambient variations outperform co-training [12].

<table><tr><td>Training Algo.</td><td>Data Filtering</td><td>Co-training [12]</td><td>Ambient $(t_{\min} \text{ per dataset})$ </td><td>Ambient $(t_{\min} \text{ per datapoint})$ </td><td>Ambient + Locality $(\sigma_{t_{\max}} = 0.025)$ </td></tr><tr><td>Success Rate ↑</td><td>56.5+7.0-7.2%</td><td>84.5+4.7-5.8%</td><td>87.0+4.3-5.5%</td><td>93.5+3.0-4.4%</td><td>92.0+3.4-4.7%</td></tr></table>

visualizes the rollouts. Lastly, training on $\mathcal{D}_q$ trades smoothness for better data coverage and success rates. Ambient's Pareto frontier for this trade-off strictly dominates the co-training baseline (Figure 6a).

Ablation: We claim that $p_{t} \approx q_{t}$ for $t > t_{min}$ . If this were true, then we should be able to train an effective policy without using $D_{p}$ for $t > t_{min}$ . To test this, we train a policy that exclusively trains on $D_{q}$ for $t \in (t_{\min}, T]$ and $D_{p}$ for $t \in [0, t_{\min}]$ . Remarkably, it matches the performance of the Ambient Diffusion Policy, with a success rate of 99.0% and a squared acceleration of 29.9.

## 7.2 Neural Motion Planning

We extend the maze experiment to neural motion planning $[68, 69, 70]$ with a 7-DoF robot arm in a cluttered environment (Figure 1b). The setup and metrics are analogous to the 2D maze, except that start and goal configurations are sampled around objects and shelves, and the planner predicts the entire plan open-loop. Appendix Table 7 compares the two setups. The main results in Table 1b and Figure 6b mirror the 2D maze experiments: Ambient achieves the highest success rate while remaining substantially smoother than co-training. The full experimental details and ablations are in Appendix G.2.

Implications for Neural Motion Planning (NMP). Most neural motion planners $[71, 69]$ are trained on large datasets of sampling-based trajectories (akin to $D_{q}$ ) since they are inexpensive to generate. Unfortunately, sampling-based trajectories are non-smooth $[66]$ ; thus, these approaches choose to prioritize data scale over quality. Our results suggest that an NMP trained with Ambient can enjoy the best of both worlds: augmenting the existing sampling-based datasets with a small amount of high-quality data could yield planners that are both general and smooth.

## 7.3 Sim-to-Real Distribution Shift

We evaluate Ambient Diffusion Policy on sim-and-real co-training using the planar pushing setup from $[12]$ (Figure 1c), a canonical task in robot imitation learning. Our experiments use the sim-and-target problem from Wei et al. $[12]$ as a controlled proxy for the sim-and-real problem.

$D_{p}$ consists of 50 teleoperated demonstrations in the target environment; $D_{q}$ consists of 2000 trajectories generated by a motion planner [72] in the simulated environment. We use the same datasets from Wei et al. [12] to enable direct comparisons with their method. A policy rollout is successful if it pushes the T to the target pose within a time limit.

Table 2 reports the success rate for three variations of Ambient Diffusion Policy that annotate $t_{min}$ at two granularities: per dataset (with a parameter sweep) and per datapoint (with the classifier approach). We explore intermediate granularities, scaling laws, and more ablations in Appendix G.3. All Ambient variations outperform the best co-trained policy [12]. Annotating $t_{min}$ per datapoint performs the best because even within a single dataset, the utility of each datapoint can vary. As before, Ambient learns from suboptimal simulation data by leveraging the diffusion hierarchy. $D_{p}$ and $D_{q}$ share the same high-level strategy; thus, $D_{q}$ is safe to use at high noise. Locality ( $t_{max} > 0$ ) does not help since $D_{p}$ and $D_{q}$ exhibit different low-level contact dynamics.

Table 3: Main block sorting results ( $\sim$ 800 blocks). Only Ambient Diffusion Policy (Locality) performs well on all metrics.

<table><tr><td>Training Algo.</td><td>Data Filtering</td><td> $\mathcal{D}_{q} Only$ </td><td>Co-train $(\alpha^{*}=0.9)$ </td><td>Locality $(\sigma_{t_{max}}^{*}=0.46)$ </td></tr><tr><td>Logic ↑</td><td> $98.6^{+0.8}_{-1.5}\%$ </td><td> $3.0^{+1.6}_{-1.2}\%$ </td><td> $26.0^{+3.4}_{-3.2}\%$ </td><td> $98.2^{+0.8}_{-1.2}\%$ </td></tr><tr><td>Motion ↑</td><td> $61.9^{+3.4}_{-3.5}\%$ </td><td> $83.0^{+2.5}_{-2.8}\%$ </td><td> $87.2^{+2.2}_{-2.5}\%$ </td><td> $95.0^{+1.4}_{-1.7}\%$ </td></tr><tr><td>Success Rate ↑</td><td> $61.0^{+3.4}_{-3.5}\%$ </td><td> $2.5^{+1.3}_{-1.0}\%$ </td><td> $22.7^{+3.1}_{-2.9}\%$ </td><td> $93.3^{+1.6}_{-2.0}\%$ </td></tr></table>

![](images/dc44e43277e5795d66aa3f1db8570933e8263a7038c5a6f0ddf775f637671657.jpg)

![](images/3170ea59334327541b37266ca29648e73bc8174fd4fe8f6467f384a744539aa5.jpg)

![](images/90eb4afc13e7ecb2bd87e33b2b921ef881e6aba6b35f212ebf2ab338333aa7a1.jpg)  
Figure 7: Left: Performance metrics vs $\sigma_{t_{\mathrm{max}}}$ . The logic metric plateaus before $\sigma_{t_{\mathrm{max}}^*} = 0.46$ and deteriorates rapidly afterwards; the opposite is true for the motion metric. Thus, the policy learns local motion-level features for $t \in [0, t_{\mathrm{max}}^*)$ and global logic-level features for $t \in (t_{\mathrm{max}}^*, T]$ . Right: Unlike co-training, Ambient achieves a near-optimal trade-off between logic and motion metrics.

## 7.4 Task Mismatch

Ambient can leverage “locality” to learn local action primitives (e.g., grasping) from data collected on a different task. Consider the block sorting task in Figure 1d. The robot must sort red and blue blocks into the left and right bins, respectively. This requires two distinct skills: motion (grasping and placing blocks) and logic (selecting the correct bin). We design a metric to evaluate each skill:

\- Motion metric: $\frac{\# blocks placed}{\# total blocks}$ . This metric is independent of logical reasoning because the numerator counts any placed block, regardless of logical correctness.

\- Logic metric: $\frac{\# blocks placed correctly}{\# blocks placed}$ . This metric is independent of pick-and-place ability because the denominator only counts successfully placed blocks.

The overall success rate is $\frac{\# blocks placed correctly}{\# total blocks}$ , the product of the two metrics. $D_{p}$ contains 50 demonstrations with the correct sorting. $D_{q}$ contains 200 demonstrations with the opposite sorting. To isolate the effect of locality, we set $t_{min} = 0$ and sweep $t_{max}$ for $D_{q}$ .

Table 3 presents the results. Co-training performs poorly because training on $D_{q}$ at all diffusion times teaches the policy both the useful motion primitives and the incorrect logic. Ambient Diffusion Policy solves this problem by restricting $D_{q}$ to $t \in [0, t_{\max})$ . As illustrated in Figures 2 and 7, the policy learns motion primitives at low diffusion times without attending to the (incorrect) task-level structure of the demonstrations. This results in near-perfect scores on both metrics simultaneously.

Ablations: We present two ablations in Appendix G.4. 1) we add a one-hot encoding indicating the sorting direction to the policies. Ambient still outperforms the co-training. 2) we train a policy that exclusively uses $D_{q}$ for $t \in [0, t_{\max})$ and $D_{p}$ for $t \in [t_{\max}, T]$ . This is the locality-version of the ablation in the 2D maze experiment. The resulting policy achieves a logic and motion score of 97.9% and 93.8%, reinforcing our claim that action diffusion is hierarchical.

![](images/dbbe4679c2d9588798e3b5e4df6e34605ea6b279e269c57872aa527521eda286.jpg)  
Figure 8: Left: success rate of policies finetuned from different base policies. Right: smoothness of finetuned policies in the maze experiment. We compare smoothness instead of success rate since all policies achieved 99.0%+ success rate.

## 7.5 Finetuning Comparison

Finally, we compare against the finetuning baseline. For each experiment, we finetune the best co-trained policy and the best Ambient policy on $D_{p}$ ; implementation details are in Appendix I.1.

Figure 8 suggests three findings. First, finetuning an Ambient base policy consistently outperforms finetuning a co-trained base policy. Second, the Ambient base policy can often outperform the finetuned co-trained policy. Third, finetuning an Ambient base policy does not always help. We hypothesize that the Ambient policies already ignore undesirable features in the suboptimal data.

## 8 Scaling Experiments: Open X-Embodiment

Having shown the generality of Ambient Diffusion Policy, we now scale to Open X-Embodiment (OXE) [1]. OXE is a large collection of real-world data with heterogeneous quality and unstructured distribution shifts. Our goal is to extract useful learning signal from as much of this “suboptimal” data as possible. The experiments evaluate Ambient Diffusion Policy on two real-world tasks.

1) Table Cleaning (Figure 1e): the robot must open the drawer, place objects inside, and close the drawer. The diversity of the test-time objects (unseen during training) makes this challenging. Performance is measured with the task completion rubric in Table 11.

2) Tower Building (Figure 1f): the robot must stack blocks in alternating directions and colors as high as possible. This task requires precise manipulation since placement errors compound as the tower grows. Performance is measured by the number of blocks placed before toppling the tower.

The training procedure for both tasks is identical. $D_{p}$ consists of a small number of demonstrations for each task. $D_{q}$ is one of two subsets of the Open X-Embodiment (OXE) dataset: 1) Magic Soup++ (MS++), 27 datasets from OXE used by OpenVLA [5], or 2) Custom OXE (COXE), 48 datasets from OXE that contain MS++ as a subset. $t_{min}$ and $t_{max}$ are annotated according to Phase 1 of the method. We note that COXE was curated to be as inclusive as possible; details are in Appendix H.1.

Lastly, for both co-training and Ambient, we re-weight each dataset's sampling probability at every diffusion time. Let $w_{i}$ denote the weight for dataset $i$ , and $S_{t}$ denote the indices of datasets that are permissible at diffusion time $t$ . The sampling probability of dataset $i$ at diffusion time $t$ is

$$
\tilde {w} _ {t} ^ {(i)} = \left\{ \begin{array}{c l} \frac {w _ {i}}{\sum_ {j \in S _ {t}} w _ {j}} & \text { if } i \in S _ {t} \\ 0 & \text { otherwise } \end{array} \right..\tag{6}
$$

![](images/24cbca38019e4eedf5102c235846ea0002f93aa12d35fe34ac6b2f9d3eae5274.jpg)  
Figure 9: When scaled to OXE, Ambient Diffusion Policy outperforms data filtering and co-training by up to 15% on table cleaning and 84% on tower building (20 trials per policy).

In co-training, $S_{t}$ contains all datasets for all $t \in [0,T]$ ; in Ambient, $S_{t}$ depends on each dataset's $t_{\mathrm{min}}$ and $t_{\mathrm{max}}$ (see Figures 21 and 22). Re-weighting marginally improves the Ambient policies but is not required; in contrast, it is necessary for co-training (see Section 8.2).

## 8.1 Main Results

Table Cleaning ( $D_{p}$ with 50 demos): Data filtering achieves 68.2% task completion; co-training adds just 2-3% and plateaus when $D_{q}$ scales from MS++ to COXE. On the other hand, the best Ambient Diffusion Policy outperforms co-training by 12% and improves when $D_{q}$ scales from MS++ to COXE. Adding locality ( $t_{max} > 0$ ) appears to provide further gains.

Table Cleaning ( $\mathcal{D}_p$ with 150 demos): When $\mathcal{D}_p$ is large, data filtering improves to 80.1%, and training on suboptimal data provides less relative value. However, for many real-world tasks, collecting enough data in $\mathcal{D}_p$ could be prohibitively expensive [73]. Nonetheless, Ambient continues to outperform co-training by up to 10%. Adding locality did not improve performance in this setting. We hypothesize that when $\mathcal{D}_p$ is large, the policy can learn the relevant motion primitives from $\mathcal{D}_p$ , making data from $\mathcal{D}_q$ unnecessary or even harmful with the wrong choice of $t_{\text{max}}$ . This is consistent with the classifier annotation method, which would have assigned $t_{\text{max}} = 0$ to nearly all datasets.

Tower Building ( $D_{p}$ with 35 demos): On average, the Ambient Diffusion Policies build up to 84% and 33% taller than the data filtering and co-training baselines, respectively (Figure 9). Including $D_{q}$ at low diffusion times appears to improve performance.

## 8.2 Ablations

Dataset Re-weighting: Figure 10 ablates the effect of re-weighting in Eq. (6) on co-training and Ambient. Re-weighting improves policy performance on the table cleaning task by 3–9%. On the other hand, re-weighting is necessary for co-training: the unweighted co-trained policies were too unsafe to evaluate on hardware. This contrast highlights another practical advantage of Ambient: the training algorithm is inherently less sensitive to dataset weights because $D_{q}$ is only used when $p_{t} \approx q_{t}$ . Co-training uses $D_{q}$ at all diffusion times, making the weights critical and fragile.

Additional Ablations: We present two further ablations in Appendix H.2. 1) Finetuning the OXE policies on $D_{p}$ produced no statistically significant change in performance (Appendix I.2), consistent with the controlled experiments (Section 7.5). 2) On table cleaning, the Ambient Diffusion Policies require 25-40% fewer grasps and less time per object than their co-trained counterparts.

Re-weighting Ablation: Table Cleaning  
![](images/8bdf2a7d9a725dd2adaf98b3d0268e773e33921516c050e3f4c8cad321c6c0f5.jpg)  
Figure 10: Ambient is significantly less sensitive to dataset re-weighting than co-training. The Ambient Diffusion Policy performed at most 9% worse without re-weighting. The co-trained policies were too dangerous to evaluate without re-weighting.

## 9 Limitations and Future Work

Ambient Diffusion Policy does not yet handle distribution shifts in the observation space. We explore two initial attempts in Appendix J. The classifier-based annotation method for $t_{min}$ and $t_{max}$ is theoretically grounded, but does not always outperform a (costly) hyperparameter sweep. Better annotation methods could improve performance. In addition, the theory in Section 5 is limited to Gaussians; extending the results to a broader class of distributions could provide further insights. Lastly, our method is more computationally expensive than data filtering since it trains on more data, but in most cases, its benefits are worthwhile.

Ambient Diffusion Policy requires the user to partition their data into $D_{p}$ and $D_{q}$ . The algorithm itself poses no restrictions on this partitioning, but it is an important design decision for downstream performance. This paper defines $D_{p}$ as data from the target task and environment, but the right choice depends on the goal. For task-specific training or finetuning, our definition is appropriate; for pretraining a generalist policy, prior work suggests that $D_{p}$ should contain diverse and high-quality demonstrations [74, 58]. The former suggests that we could expand our finetuning dataset to include suboptimal data. The latter requires a more principled understanding of data quality in robotics [58, 75]. Both settings are exciting directions for future work.

## 10 Conclusion

High-quality robot data is scarce, making it essential to learn from suboptimal data sources. But even as data collection scales, suboptimal data will continue to grow alongside high-quality data. Thus, the problem of learning from suboptimal data sources is not a temporary artifact of today's data scarcity, but rather a fundamental and persistent challenge in robot imitation learning. The dominant approach for addressing this problem is to filter the worst data and co-train on the rest via re-weighting; but filtering is wasteful, and co-training teaches a policy the meaningful and the harmful parts of suboptimal distributions. In this paper, we propose Ambient Diffusion Policy, a principled method for training on suboptimal demonstrations.

Neighboring fields have recognized the power of noise-dependent data usage $[19, 25]$ . Our key experimental and theoretical insight is that this training paradigm is also well-suited for robotics. Our method makes no assumptions about the nature of the action suboptimality, outperforms existing baselines, and greatly expands the set of useful data sources for robot imitation learning. With Ambient Diffusion Policy, the question shifts from which data sources should be used, to when each should be used in the diffusion process.

## Acknowledgments

We thank Pablo Parrilo, Asuman Özdağlar, Abhinav Agarwal, Cole Becker, Ananth Kumar Srinivasan, Sarthak Kaingade, Max Wilcoxson, and the Robot Locomotion Group for fruitful discussions. We thank John Marangola for help with the hardware setup.

This work was supported by the National Science Foundation Graduate Research Fellowship Program under Grant No. 2141064, and the Natural Sciences and Engineering Research Council of Canada CGS D-587703. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation. We thank MIT Supercloud [76] for providing computational resources.

## References

[1] O. X.-E. Collaboration, A. O'Neill, A. Rehman, A. Gupta, A. Maddukuri, A. Gupta, A. Padalkar, A. Lee, A. Pooley, A. Gupta, A. Mandlekar, A. Jain, A. Tung, A. Bewley, A. Herzog, A. Irpan, A. Khazatsky, A. Rai, A. Gupta, A. Wang, A. Kolobov, A. Singh, A. Garg, A. Kembhavi, A. Xie, A. Brohan, A. Raffin, A. Sharma, A. Yavary, A. Jain, A. Balakrishna, A. Wahid, B. Burgess-Limerick, B. Kim, B. Schölkopf, B. Wulfe, B. Ichter, C. Lu, C. Xu, C. Le, C. Finn, C. Wang, C. Xu, C. Chi, C. Huang, C. Chan, C. Agia, C. Pan, C. Fu, C. Devin, D. Xu, D. Morton, D. Driess, D. Chen, D. Pathak, D. Shah, D. Büchler, D. Jayaraman, D. Kalashnikov, D. Sadigh, E. Johns, E. Foster, F. Liu, F. Ceola, F. Xia, F. Zhao, F. V. Frujeri, F. Stulp, G. Zhou, G. S. Sukhatme, G. Salhotra, G. Yan, G. Feng, G. Schiavi, G. Berseth, G. Kahn, G. Yang, G. Wang, H. Su, H.-S. Fang, H. Shi, H. Bao, H. B. Amor, H. I. Christensen, H. Furuta, H. Bharadhwaj, H. Walke, H. Fang, H. Ha, I. Mordatch, I. Radosavovic, I. Leal, J. Liang, J. Abou-Chakra, J. Kim, J. Drake, J. Peters, J. Schneider, J. Hsu, J. Vakil, J. Bohg, J. Bingham, J. Wu, J. Gao, J. Hu, J. Wu, J. Wu, J. Sun, J. Luo, J. Gu, J. Tan, J. Oh, J. Wu, J. Lu, J. Yang, J. Malik, J. Silvério, J. Hejna, J. Booher, J. Tompson, J. Yang, J. Salvador, J. J. Lim, J. Han, K. Wang, K. Rao, K. Pertsch, K. Hausman, K. Go, K. Gopalakrishnan, K. Goldberg, K. Byrne, K. Oslund, K. Kawaharazuka, K. Black, K. Lin, K. Zhang, K. Ehsani, K. Lekkala, K. Ellis, K. Rana, K. Srinivasan, K. Fang, K. P. Singh, K.-H. Zeng, K. Hatch, K. Hsu, L. Itti, L. Y. Chen, L. Pinto, L. Fei-Fei, L. Tan, L. J. Fan, L. Ott, L. Lee, L. Weihs, M. Chen, M. Lepert, M. Memmel, M. Tomizuka, M. Itkina, M. G. Castro, M. Spero, M. Du, M. Ahn, M.C.Yip,M.Zhang,M.Ding,M.Heo,M.K.Srirama,M.Sharma,M.J.Kim,M.Z.Irshad,N.Kanazawa,N.Hansen,N.Heess,N.J.Joshi,N.Suenderhauf,N.Liu,N.D.Palo,N.M.M.Shafiullah,O.Mees,O.Kroemer,O.Bastani,P.R.Sanketi,P.T.Miller,P.Yin,P.Wohlhart,P.Xu,P.D.Fagan,P.Mitrano,P.Sermanet,P.Abbeel,P.Sundaresan,Q.Chen,Q.Vuong,R.Rafailov,R.Tian,R.Doshi,R.Mart'in-Mart'in,R.Baijal,R Scalise,R.Hendrix,R.Lin,R.Qian,R.Zhang,R.Mendonca,R.Shah,R.Hoque,R.Julian,S.Bustamante,S.Kirmani,S.Levine,S.Lin,S.Moore,S.Bahl,S.Dass,S.Sonawani,S.Tulsiani,S.Song,S.Xu,S.Haldar,S.Karamcheti,S.Adebola,S.Guist,S.Nasiriany,S.Schaal,S.Welker,S.Tian,S.Ramamoorthy,S.Dasari,S.Belkhale,S.Park,S.Nair,S.Mirchandani,T.Osa,T.Gupta,T.Harada,T.Matsushima,T.Xiao,T.Kollar,T.Yu,T.Ding,T.Davchev,T.Z.Zhao,T.Armstrong,T.Darrell,T.Chung,V.Jain,V.Kumar,V.Vanhoucke,V.Guizilini,W.Zhan,W.Zhou,W.Burgard,X.Chen,X.Chen,X.Wang,X.Zhu,X.Geng,X.Liu,X.Liangwei,X.Li,Y.Pang,Y.Lu,Y.J.Ma,Y.Kim,Y.Chebotar,Y.Zhou,Y.Zhu,Y.Wu,Y.Xu,Y.Wang,Y.Bisk,Y.Dou,Y.Cho,Y.Lee,Y.Cui,Y.Cao,Y.-H.Wu,Y.Tang,Y.Zhu,Y.Zhang,Y.Jiang,Y.Li,Y.Li,Y.Iwasawa,Y.Matsuo,Z.Ma,Z.Xu,Z.J.Cui,Z.Zhang,Z.Fu,and Z.Lin.Open X-Embodiment: Robotic learning datasets and RT-X models. $https://arxiv.org/abs/2310_08864$ , 2023

[2] P. Intelligence, K. Black, N. Brown, J. Darpinian, K. Dhabalia, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, M. Y. Galliker, D. Ghosh, L. Groom, K. Hausman, B. Ichter, S. Jakubczak, T. Jones, L. Ke, D. LeBlanc, S. Levine, A. Li-Bell, M. Mothukuri, S. Nair, K. Pertsch, A. Z. Ren, L. X. Shi, L. Smith, J. T. Springenberg, K. Stachowicz, J. Tanner, Q. Vuong, H. Walke,

A. Walling, H. Wang, L. Yu, and U. Zhilinsky. $\pi_{0.5}$ : a vision-language-action model with open-world generalization, 2025. URL https://arxiv.org/abs/2504.16054.

[3] NVIDIA, :, J. Bjorck, F. Castañeda, N. Cherniadev, X. Da, R. Ding, L. J. Fan, Y. Fang, D. Fox, F. Hu, S. Huang, J. Jang, Z. Jiang, J. Kautz, K. Kundalia, L. Lao, Z. Li, Z. Lin, K. Lin, G. Liu, E. Llontop, L. Magne, A. Mandlekar, A. Narayan, S. Nasiriany, S. Reed, Y. L. Tan, G. Wang, Z. Wang, J. Wang, Q. Wang, J. Xiang, Y. Xie, Y. Xu, Z. Xu, S. Ye, Z. Yu, A. Zhang, H. Zhang, Y. Zhao, R. Zheng, and Y. Zhu. Gr00t n1: An open foundation model for generalist humanoid robots, 2025. URL https://arxiv.org/abs/2503.14734.

[4] T. L. Team, J. Barreiros, A. Beaulieu, A. Bhat, R. Cory, E. Cousineau, H. Dai, C.-H. Fang, K. Hashimoto, M. Z. Irshad, M. Itkina, N. Kuppuswamy, K.-H. Lee, K. Liu, D. McConachie, I. McMahon, H. Nishimura, C. Phillips-Grafflin, C. Richter, P. Shah, K. Srinivasan, B. Wulfe, C. Xu, M. Zhang, A. Alspach, M. Angeles, K. Arora, V. C. Guizilini, A. Castro, D. Chen, T.-S. Chu, S. Creasey, S. Curtis, R. Denitto, E. Dixon, E. Dusel, M. Ferreira, A. Goncalves, G. Gould, D. Guoy, S. Gupta, X. Han, K. Hatch, B. Hathaway, A. Henry, H. Hochsztein, P. Horgan, S. Iwase, D. Jackson, S. Karamcheti, S. Keh, J. Masterjohn, J. Mercat, P. Miller, P. Mitiguy, T. Nguyen, J. Nimmer, Y. Noguchi, R. Ong, A. Onol, O. Pfannenstiehl, R. Poyner, L. P. M. Rocha, G. Richardson, C. Rodriguez, D. Seale, M. Sherman, M. Smith-Jones, D. Tago, P. Tokmakov, M. Tran, B. V. Hoorick, I. Vasiljevic, S. Zakharov, M. Zolotas, R. Ambrus, K. Fetzer-Borelli, B. Burchfiel, H. Kress-Gazit, S. Feng, S. Ford, and R. Tedrake. A careful examination of large behavior models for multitask dexterous manipulation, 2025. URL https://arxiv.org/abs/2507.05331.

[5] K. Kim, Moo Jin andwo, Z. Pan, et al. Openvla: An open-source vision-language-action model. In arXiv preprint arXiv:2406.09246, 2024.

[6] F. Lin, K. Arora, J. Mercat, H. Nishimura, P. Shah, C. Xu, M. Zhang, M. Zolotas, M. Angeles, O. Pfannenstiehl, A. Beaulieu, and J. Barreiros. A systematic study of data modalities and strategies for co-training large behavior models for robot manipulation, 2026. URL https://arxiv.org/abs/2602.01067.

[7] G. R. Team, S. Abeyruwan, J. Ainslie, J.-B. Alayrac, M. G. Arenas, T. Armstrong, A. Balakrishna, R. Baruch, M. Bauza, M. Blokzijl, S. Bohez, K. Bousmalis, A. Brohan, T. Buschmann, A. Byravan, S. Cabi, K. Caluwaerts, F. Casarini, O. Chang, J. E. Chen, X. Chen, H.-T. L. Chiang, K. Choromanski, D. D'Ambrosio, S. Dasari, T. Davchev, C. Devin, N. D. Palo, T. Ding, A. Dostmohamed, D. Driess, Y. Du, D. Dwibedi, M. Elabd, C. Fantacci, C. Fong, E. Frey, C. Fu, M. Giustina, K. Gopalakrishnan, L. Graesser, L. Hasenclever, N. Heess, B. Hernaez, A. Herzog, R. A. Hofer, J. Humplik, A. Iscen, M. G. Jacob, D. Jain, R. Julian, D. Kalashnikov, M. E. Karagozler, S. Karp, C. Kew, J. Kirkland, S. Kirmani, Y. Kuang, T. Lampe, A. Laurens, I. Leal, A. X. Lee, T.-W. E. Lee, J. Liang, Y. Lin, S. Maddineni, A. Majumdar, A. H. Michaely, R. Moreno, M. Neunert, F. Nori, C. Parada, E. Parisotto, P. Pastor, A. Pooley, K. Rao, K. Reymann, D. Sadigh, S. Saliceti, P. Sanketi, P. Sermanet, D. Shah, M. Sharma, K. Shea, C. Shu, V. Sindhwani, S. Singh, R. Soricut, J. T. Springenberg, R. Sterneck, R. Surdulescu, J. Tan, J. Tompson, V. Vanhoucke, J. Varley, G. Vesom, G. Vezzani, O. Vinyals, A. Wahid, S. Welker, P. Wohlhart, F. Xia, T. Xiao, A. Xie, J. Xie, P. Xu, S. Xu, Y. Xu, Z. Xu, Y. Yang, R. Yao, S. Yaroshenko, W. Yu, W. Yuan, J. Zhang, T. Zhang, A. Zhou, and Y. Zhou. Gemini robotics: Bringing ai into the physical world, 2025. URL https://arxiv.org/abs/2503.20020.

[8] AgiBot-World-Contributors, Q. Bu, J. Cai, L. Chen, X. Cui, Y. Ding, S. Feng, S. Gao, X. He, X. Hu, X. Huang, S. Jiang, Y. Jiang, C. Jing, H. Li, J. Li, C. Liu, Y. Liu, Y. Lu, J. Luo, P. Luo, Y. Mu, Y. Niu, Y. Pan, J. Pang, Y. Qiao, G. Ren, C. Ruan, J. Shan, Y. Shen, C. Shi, M. Shi, M. Shi, C. Sima, J. Song, H. Wang, W. Wang, D. Wei, C. Xie, G. Xu, J. Yan, C. Yang, L. Yang, S. Yang, M. Yao, J. Zeng, C. Zhang, Q. Zhang, B. Zhao, C. Zhao, J. Zhao, and J. Zhu. Agibot world colosseo: A large-scale manipulation platform for scalable and intelligent embodied systems, 2025. URL https://arxiv.org/abs/2503.06669.

[9] C. Chi, Z. Xu, C. Pan, E. Cousineau, B. Burchfiel, S. Feng, R. Tedrake, and S. Song. Universal manipulation interface: In-the-wild robot teaching without in-the-wild robots, 2024. URL https://arxiv.org/abs/2402.10329.

[10] Physical Intelligence, B. Ai, A. Amin, R. Aniceto, A. Balakrishna, G. Balke, K. Black, G. Bokinsky, S. Cao, T. Charbonnier, V. Choudhary, F. Collins, K. Conley, G. Connors, J. Darpinian, K. Dhabalia, M. Dhaka, J. DiCarlo, D. Driess, M. Equi, A. Esmail, Y. Fang, C. Finn, C. Glossop, T. Godden, I. Goryachev, L. Groom, H. Habeeb, H. Hancock, K. Hausman, G. Hussein, V. Hwang, B. Ichter, C. Jacobsen, S. Jakubczak, R. Jen, T. Jones, G. Kammerer, B. Katz, L. Ke, M. Khadikov, C. Kuchi, M. Lamb, D. LeBlanc, B. LeCount, S. Levine, X. Li, A. Li-Bell, V. Lialin, Z. Liang, W. Lim, Y. Lu, E. Luo, V. Mano, N. Marwaha, A. Mongush, L. Murphy, S. Nair, T. Patterson, K. Pertsch, A. Z. Ren, G. Schelske, C. Sharma, B. Shi, L. X. Shi, L. Smith, J. T. Springenberg, K. Stachowicz, W. Stoeckle, J. Tang, J. Tanner, S. Tekeste, M. Torne, K. Vedder, Q. Vuong, A. Walling, H. Wang, J. Wang, X. Wang, C. Whalen, S. Whitmore, B. Williams, C. Xu, S. Yoo, L. Yu, W. Zhang, Z. Zhang, and U. Zhilinsky. $\pi_{0.7}$ : a steerable generalist robotic foundation model with emergent capabilities. Technical report, Physical Intelligence, 2026. URL https://pi.website/pi07.

[11] P. Intelligence, A. Amin, R. Aniceto, A. Balakrishna, K. Black, K. Conley, G. Connors, J. Darpinian, K. Dhabalia, J. DiCarlo, D. Driess, M. Equi, A. Esmail, Y. Fang, C. Finn, C. Glossop, T. Godden, I. Goryachev, L. Groom, H. Hancock, K. Hausman, G. Hussein, B. Ichter, S. Jakubczak, R. Jen, T. Jones, B. Katz, L. Ke, C. Kuchi, M. Lamb, D. LeBlanc, S. Levine, A. Li-Bell, Y. Lu, V. Mano, M. Mothukuri, S. Nair, K. Pertsch, A. Z. Ren, C. Sharma, L. X. Shi, L. Smith, J. T. Springenberg, K. Stachowicz, W. Stoeckle, A. Swerdlow, J. Tanner, M. Torne, Q. Vuong, A. Walling, H. Wang, B. Williams, S. Yoo, L. Yu, U. Zhilinsky, and Z. Zhou. $\pi_{0.6}^{*}$ : a vla that learns from experience, 2025. URL https://arxiv.org/abs/2511.14759.

[12] A. Wei, A. Agarwal, B. Chen, R. Bosworth, N. Pfaff, and R. Tedrake. Empirical analysis of sim-and-real cotraining of diffusion policies for planar pushing from pixels. arXiv preprint arXiv:2503.22634, 2025.

[13] A. Maddukuri, Z. Jiang, L. Y. Chen, S. Nasiriany, Y. Xie, Y. Fang, W. Huang, Z. Wang, Z. Xu, N. Chernyadev, et al. Sim-and-real co-training: A simple recipe for vision-based robotic manipulation. arXiv preprint arXiv:2503.24361, 2025.

[14] K. Grauman, A. Westbury, L. Torresani, K. Kitani, J. Malik, T. Afouras, K. Ashutosh, V. Baiyya, S. Bansal, B. Boote, E. Byrne, Z. Chavis, J. Chen, F. Cheng, F.-J. Chu, S. Crane, A. Dasgupta, J. Dong, M. Escobar, C. Forigua, A. Gebreselasie, S. Haresh, J. Huang, M. M. Islam, S. Jain, R. Khirodkar, D. Kukreja, K. J. Liang, J.-W. Liu, S. Majumder, Y. Mao, M. Martin, E. Mavroudi, T. Nagarajan, F. Ragusa, S. K. Ramakrishnan, L. Seminara, A. Somayazulu, Y. Song, S. Su, Z. Xue, E. Zhang, J. Zhang, A. Castillo, C. Chen, X. Fu, R. Furuta, C. Gonzalez, P. Gupta, J. Hu, Y. Huang, Y. Huang, W. Khoo, A. Kumar, R. Kuo, S. Lakhavani, M. Liu, M. Luo, Z. Luo, B. Meredith, A. Miller, O. Oguntola, X. Pan, P. Peng, S. Pramanick, M. Ramazanova, F. Ryan, W. Shan, K. Somasundaram, C. Song, A. Southerland, M. Tateno, H. Wang, Y. Wang, T. Yagi, M. Yan, X. Yang, Z. Yu, S. C. Zha, C. Zhao, Z. Zhao, Z. Zhu, J. Zhuo, P. Arbelaez, G. Bertasius, D. Crandall, D. Damen, J. Engel, G. M. Farinella, A. Furnari, B. Ghanem, J. Hoffman, C. V. Jawahar, R. Newcombe, H. S. Park, J. M. Rehg, Y. Sato, M. Savva, J. Shi, M. Z. Shou, and M. Wray. Ego-exo4d: Understanding skilled human activity from first- and third-person perspectives, 2024. URL https://arxiv.org/abs/2311.18259.

[15] S. Gao, W. Liang, K. Zheng, A. Malik, S. Ye, S. Yu, W.-C. Tseng, Y. Dong, K. Mo, C.-H. Lin, Q. Ma, S. Nah, L. Magne, J. Xiang, Y. Xie, R. Zheng, D. Niu, Y. L. Tan, K. R. Zentner, G. Kurian, S. Indupuru, P. Jannaty, J. Gu, J. Zhang, J. Malik, P. Abbeel, M.-Y. Liu, Y. Zhu, J. Jang, and L. J. Fan. Dreamdojo: A generalist robot world model from large-scale human videos, 2026. URL https://arxiv.org/abs/2602.06949.

[16] J. Hejna, C. Bhateja, Y. Jiang, K. Pertsch, and D. Sadigh. Re-mix: Optimizing data mixtures for large scale imitation learning, 2024. URL https://arxiv.org/abs/2408.14037.

[17] Z. Zhang, J. Pang, Z. Yang, K. Li, M. Liao, S. Zhang, G. Chi, J. Guo, H. ang Gao, M. Shi, D. Ge, Y. Mu, J. Gu, R. Chen, H. Dong, H. Xu, L. Yi, Y. Zhu, H. Zhao, P. Wang, S. Zhang, G. Yao, J. Chen, H. Li, and H. Zhao. Dexora: Open-source vla for high-dof bimanual dexterity, 2026. URL https://arxiv.org/abs/2605.18722.

[18] J. Wen, Y. Zhu, J. Li, Z. Tang, C. Shen, and F. Feng. Dexvla: Vision-language model with plug-in diffusion expert for general robot control, 2025. URL https://arxiv.org/abs/2502.05855.

[19] G. Daras, A. Rodriguez-Munoz, A. Klivans, A. Torralba, and C. C. Daskalakis. Ambient diffusion omni: Training good models with bad data. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview.net/forum?id=MVYz4GmcUH.

[20] V. Tangkaratt, N. Charoenphakdee, and M. Sugiyama. Robust imitation learning from noisy demonstrations, 2021. URL https://arxiv.org/abs/2010.10181.

[21] A. Torralba and A. Oliva. Statistics of natural image categories. Network: Computation in Neural Systems, 14(3):391–412, 2003.

[22] A. Lukoianov, C. Yuan, J. Solomon, and V. Sitzmann. Locality in image diffusion models emerges from data statistics, 2025. URL https://arxiv.org/abs/2509.09672.

[23] C. Chi, Z. Xu, S. Feng, E. Cousineau, Y. Du, B. Burchfiel, R. Tedrake, and S. Song. Diffusion policy: Visuomotor policy learning via action diffusion, 2024. URL https://arxiv.org/abs/2303.04137.

[24] G. Daras, A. Dimakis, and C. C. Daskalakis. Consistent diffusion meets tweedie: Training exact ambient diffusion models with noisy data. In Forty-first International Conference on Machine Learning, 2024. URL https://openreview.net/forum?id=PlVjIGaFdH.

[25] G. Daras, J. Ouyang-Zhang, K. Ravishankar, C. C. Daskalakis, A. Klivans, and D. J. Diaz. Ambient proteins - training diffusion models on noisy structures. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview.net/forum?id=X1101vw606.

[26] S. Nasiriany, A. Maddukuri, L. Zhang, A. Parikh, A. Lo, A. Joshi, A. Mandlekar, and Y. Zhu. Robocasa: Large-scale simulation of everyday tasks for generalist robots, 2024. URL https://arxiv.org/abs/2406.02523.

[27] S. Ben-David, J. Blitzer, K. Crammer, A. Kulesza, F. Pereira, and J. Vaughan. A theory of learning from different domains. Machine Learning, 79:151–175, 2010.

[28] J. Hejna, S. Mirchandani, A. Balakrishna, A. Xie, A. Wahid, J. Tompson, P. Sanketi, D. Shah, C. Devin, and D. Sadigh. Robot data curation with mutual information estimators, 2025. URL https://arxiv.org/abs/2502.08623.

[29] S. M. Xie, H. Pham, X. Dong, N. Du, H. Liu, Y. Lu, P. Liang, Q. V. Le, T. Ma, and A. W. Yu. Doremi: Optimizing data mixtures speeds up language model pretraining, 2023. URL https://arxiv.org/abs/2305.10429.

[30] C. Agia, R. Sinha, J. Yang, R. Antonova, M. Pavone, H. Nishimura, M. Itkina, and J. Bohg. Cupid: Curating data your robot loves with influence functions, 2025. URL https://arxiv.org/abs/2506.19121.

[31] S. Goyal, P. Maini, Z. C. Lipton, A. Raghunathan, and J. Z. Kolter. Scaling laws for data filtering–data curation cannot be compute agnostic. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 22702–22711, June 2024.

[32] J. Li, A. Fang, G. Smyrnis, M. Ivgi, M. Jordan, S. Gadre, H. Bansal, E. Guha, S. Keh, K. Arora, S. Garg, R. Xin, N. Muennighoff, R. Heckel, J. Mercat, M. Chen, S. Gururangan, M. Wortsman, A. Albalak, Y. Bitton, M. Nezhurina, A. Abbas, C.-Y. Hsieh, D. Ghosh, J. Gardner, M. Kilian, H. Zhang, R. Shao, S. Pratt, S. Sanyal, G. Ilharco, G. Daras, K. Marathe, A. Gokaslan, J. Zhang, K. Chandu, T. Nguyen, I. Vasiljevic, S. Kakade, S. Song, S. Sanghavi, F. Faghri, S. Oh, L. Zettlemoyer, K. Lo, A. El-Nouby, H. Pouransari, A. Toshev, S. Wang, D. Groeneveld, L. Soldaini, P. W. Koh, J. Jitsev, T. Kollar, A. G. Dimakis, Y. Carmon, A. Dave, L. Schmidt, and V. Shankar. Datacomp-lm: In search of the next generation of training sets for language models, 2025. URL https://arxiv.org/abs/2406.11794.

[33] A. Aali, M. Arvinte, S. Kumar, and J. I. Tamir. Solving inverse problems with score-based generative priors learned from noisy data. In 2023 57th Asilomar Conference on Signals, Systems, and Computers, page 837–843. IEEE, Oct. 2023. doi:10.1109/ieeeconf59524.2023.10477042. URL http://dx.doi.org/10.1109/IEEECONF59524.2023.10477042.

[34] T. Chen, Y. Zhang, Z. Wang, Y. N. Wu, O. Leong, and M. Zhou. Denoising score distillation: From noisy diffusion pretraining to one-step high-quality generation, 2025. URL https://arxiv.org/abs/2503.07578.

[35] G. Daras, H. Chung, C.-H. Lai, Y. Mitsufuji, J. C. Ye, P. Milanfar, A. G. Dimakis, and M. Delbracio. A survey on diffusion models for inverse problems, 2024. URL https://arxiv.org/abs/2410.00083.

[36] K. Shah, A. Kalavasis, A. Klivans, and G. Daras. Does generation require memorization? creative diffusion models using ambient diffusion. In Forty-second International Conference on Machine Learning, 2025. URL https://openreview.net/forum?id=GGPMOz3dhU.

[37] A. Rodríguez-Muñoz, W. Daspit, A. Klivans, A. Torralba, C. Daskalakis, and G. Daras. Ambient dataloops: Generative models for dataset refinement, 2026. URL https://arxiv.org/abs/2601.15417.

[38] H. Lu, Q. Wu, and Y. Yu. SFBD: A method for training diffusion models with noisy data. In Frontiers in Probabilistic Inference: Learning meets Sampling Workshop at ICLR, 2025. URL https://openreview.net/pdf/8a1f8db4471b46d787ac9e685c15009b4566b5ed.pdf.

[39] H. Lu, Y. Yu, and D. Lo. Sfbd-omni: Bridge models for lossy measurement restoration with limited clean samples, 2026. URL https://arxiv.org/abs/2512.17051.

[40] C. Y. Park, S. Shoushtari, H. An, and U. S. Kamilov. Measurement score-based diffusion model, 2025. URL https://arxiv.org/abs/2505.11853.

[41] C. Modi, J. Han, E. Vanden-Eijnden, and J. Bruna. Generative modeling from black-box corruptions via self-consistent stochastic interpolants, 2026. URL https://arxiv.org/abs/2512.10857.

[42] B. Kawar, N. Elata, T. Michaeli, and M. Elad. Gsure-based diffusion model training with corrupted data, 2024. URL https://arxiv.org/abs/2305.13128.

[43] Y. Matsuzaki, S. Uchida, and S. Takezaki. Score: Clean image generation from diffusion models trained on noisy images, 2026. URL https://arxiv.org/abs/2604.09436.

[44] A. Bora, E. Price, and A. G. Dimakis. AmbientGAN: Generative models from lossy measurements. In International Conference on Learning Representations, 2018. URL https://openreview.net/forum?id=Hy7fDogOb.

[45] W. Bai, Y. Wang, W. Chen, and H. Sun. An expectation-maximization algorithm for training clean diffusion models from corrupted observations, 2024. URL https://arxiv.org/abs/2407.01014.

[46] D. Hosseintabar, F. Chen, G. Daras, A. Torralba, and C. Daskalakis. Diffem: Learning from corrupted data with diffusion models via expectation maximization, 2025. URL https://arxiv.org/abs/2510.12691.

[47] F. Rozet, G. Andry, F. Lanusse, and G. Louppe. Learning diffusion priors from observations by expectation maximization, 2025. URL https://arxiv.org/abs/2405.13712.

[48] J. Lyu, K. Liu, X. Zhang, H. Liao, Y. Feng, W. Zhu, T. Shen, J. Chen, J. Zhang, Y. Dong, W. Cui, S. Qi, S. Wang, Y. Zheng, M. Yan, X. Shi, H. Li, D. Zhao, M.-Y. Liu, Z. Zhang, L. Yi, Y. Wang, and H. Wang. Lda-1b: Scaling latent dynamics action model via universal embodied data ingestion, 2026. URL https://arxiv.org/abs/2602.12215.

[49] S. Cheng, L. Ma, Z. Chen, A. Mandlekar, C. Garrett, and D. Xu. Generalizable domain adaptation for sim-and-real policy co-training, 2026. URL https://arxiv.org/abs/2509.18631.

[50] R. Punamiya, D. Patel, P. Aphiwetsa, P. Kuppili, L. Y. Zhu, S. Kareer, J. Hoffman, and D. Xu. Egobridge: Domain adaptation for generalizable imitation from egocentric human data, 2025. URL https://arxiv.org/abs/2509.19626.

[51] L. Liu, Z. Tang, L. Li, and D. Luo. Robust imitation learning from corrupted demonstrations, 2022. URL https://arxiv.org/abs/2201.12594.

[52] M. A. Pace, P. Dan, C. Ning, A. Bhardwaj, A. Du, E. W. Duan, W.-C. Ma, and K. Kedia. X-diffusion: Training diffusion policies on cross-embodiment human demonstrations, 2025. URL https://arxiv.org/abs/2511.04671.

[53] T. Z. Zhao, V. Kumar, S. Levine, and C. Finn. Learning fine-grained bimanual manipulation with low-cost hardware, 2023. URL https://arxiv.org/abs/2304.13705.

[54] J. Ho, A. Jain, and P. Abbeel. Denoising diffusion probabilistic models, 2020. URL https://arxiv.org/abs/2006.11239.

[55] Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole. Score-based generative modeling through stochastic differential equations, 2021. URL https://arxiv.org/abs/2011.13456.

[56] Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, and M. Le. Flow matching for generative modeling, 2023. URL https://arxiv.org/abs/2210.02747.

[57] J. Song, C. Meng, and S. Ermon. Denoising diffusion implicit models, 2022. URL https://arxiv.org/abs/2010.02502.

[58] S. Belkhale, Y. Cui, and D. Sadigh. Data quality in imitation learning, 2023. URL https://arxiv.org/abs/2306.02437.

[59] P. Pestana, Z. Ma, J. Reiss, Á. Barbosa, and D. Black. Spectral characteristics of popular commercial recordings 1950-2010. 01 2013.

[60] S. Dieleman. Diffusion is spectral autoregression, 2024. URL https://sander.ai/2024/09/02/spectral-autoregression.html.

[61] D.F. Gray. The Observation and Analysis of Stellar Photospheres. Cambridge Univ. Press, third edition edition, Nov. 2005.

[62] A. Papoulis and S. U. Pillai. Probability, Random Variables, and Stochastic Processes. McGraw-Hill, 4th edition, 2002.

[63] E. Bauer, E. Nava, and R. K. Katzschmann. Latent action diffusion for cross-embodiment manipulation, 2026. URL https://arxiv.org/abs/2506.14608.

[64] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer. High-resolution image synthesis with latent diffusion models, 2022. URL https://arxiv.org/abs/2112.10752.

[65] T. Marcucci, M. Petersen, D. von Wrangel, and R. Tedrake. Motion planning around obstacles with convex optimization. Science Robotics, 8(84):eadf7843, 2023. doi:10.1126/scirobotics.adf7843. URL https://www.science.org/doi/abs/10.1126/scirobotics.adf7843.

[66] S. M. LaValle. Rapidly-exploring random trees: a new tool for path planning. The annual research report, 1998. URL https://api.semanticscholar.org/CorpusID:14744621.

[67] L. Foland, T. Cohn, A. Wei, N. Pfaff, B. Chen, and R. Tedrake. How well do diffusion policies learn kinematic constraint manifolds?, 2025. URL https://arxiv.org/abs/2510.01404.

[68] M. Dalal, A. Mandlekar, C. Garrett, A. Handa, R. Salakhutdinov, and D. Fox. Imitating task and motion planning with visuomotor transformers, 2023. URL https://arxiv.org/abs/2305.16309.

[69] A. H. Qureshi, A. Simeonov, M. J. Bency, and M. C. Yip. Motion planning networks, 2019. URL https://arxiv.org/abs/1806.05767.

[70] J. Carvalho, A. T. Le, M. Baierl, D. Koert, and J. Peters. Motion planning diffusion: Learning and planning of robot motions with diffusion models, 2024. URL https://arxiv.org/abs/2308.01557.

[71] M. Dalal, J. Yang, R. Mendonca, Y. Khaky, R. Salakhutdinov, and D. Pathak. Neural mp: A generalist neural motion planner, 2024. URL https://arxiv.org/abs/2409.05864.

[72] B. P. Graesdal, S. Y. C. Chia, T. Marcucci, S. Morozov, A. Amice, P. A. Parrilo, and R. Tedrake. Towards tight convex relaxations for contact-rich manipulation, 2024. URL https://arxiv.org/abs/2402.10312.

[73] T. Z. Zhao, J. Tompson, D. Driess, P. Florence, K. Ghasemipour, C. Finn, and A. Wahid. Aloha unleashed: A simple recipe for robot dexterity, 2024. URL https://arxiv.org/abs/2410.13126.

[74] Y. Hu, F. Lin, P. Sheng, C. Wen, J. You, and Y. Gao. Data scaling laws in imitation learning for robotic manipulation, 2025. URL https://arxiv.org/abs/2410.18647.

[75] A. Mandlekar, D. Xu, J. Wong, S. Nasiriany, C. Wang, R. Kulkarni, L. Fei-Fei, S. Savarese, Y. Zhu, and R. Martín-Martín. What matters in learning from offline human demonstrations for robot manipulation, 2021. URL https://arxiv.org/abs/2108.03298.

[76] A. Reuther, J. Kepner, C. Byun, S. Samsi, W. Arcand, D. Bestor, B. Bergeron, V. Gadepally, M. Houle, M. Hubbell, M. Jones, A. Klein, L. Milechin, J. Mullen, A. Prout, A. Rosa, C. Yee, and P. Michaleas. Interactive supercomputing on 40,000 cores for machine learning and data analysis. In 2018 IEEE High Performance extreme Computing Conference (HPEC), page 1–6. IEEE, 2018.

[77] Y. Polyanskiy and Y. Wu. Information theory: From coding to learning. Cambridge university press, 2025.

[78] J. Zhang, Z. Zhou, H. Li, Y. Lai, W. Xia, H. Song, Y. Gong, and J. Mei. Hydra-dp3: Frequency-aware right-sizing of 3d diffusion policies for visuomotor control, 2026. URL https://arxiv.org/abs/2605.01581.

[79] C. R. Garrett, R. Chitnis, R. Holladay, B. Kim, T. Silver, L. P. Kaelbling, and T. Lozano-Pérez. Integrated task and motion planning, 2020. URL https://arxiv.org/abs/2010.01083.

[80] Y. Wang, Z. Wang, M. Nakura, P. Bhowal, C.-L. Kuo, Y.-T. Chen, Z. Erickson, and D. Held. Articubot: Learning universal articulated object manipulation policy via large scale simulation, 2025. URL https://arxiv.org/abs/2503.03045.

[81] H. Zhao, Y. Qi, B. Hu, Y. Zhu, Z. Chen, H. Tian, X. Zhu, O. Howell, H. Huang, R. Walters, D. Wang, and R. Platt. Generalizable hierarchical skill learning via object-centric representation, 2025. URL https://arxiv.org/abs/2510.21121.

[82] N. Gireesh, Y. Ju, C. Xu, W. Liu, Y. Wan, and H. Wang. Hdflow: Hierarchical diffusion-flow planning for long-horizon tasks, 2026. URL https://arxiv.org/abs/2605.04525.

[83] R. Tedrake. Robotic Manipulation. 2024. URL http://manipulation.mit.edu.

[84] J. J. Kuffner and S. M. LaValle. RRT-connect: An efficient approach to single-query path planning. In Proceedings 2000 ICRA. Millennium conference. IEEE international conference on robotics and automation. Symposia proceedings (Cat. No. 00CH37065), volume 2, pages 995–1001. IEEE, 2000.

[85] G. Izatt. Capturing Distributions over Worlds for Robotics with Spatial Scene Grammars. Ph.d. thesis, Massachusetts Institute of Technology, Cambridge, MA, May 2022.

[86] N. Pfaff, H. Dai, S. Zakharov, S. Iwase, and R. Tedrake. Steerable scene generation with post training and inference-time search, 2025. URL https://arxiv.org/abs/2505.04831.

[87] P. E. Gill, W. Murray, and M. A. Saunders. Snopt: An sqp algorithm for large-scale constrained optimization. SIAM review, 47(1):99-131, 2005.

[88] J. J. Kuffner and S. M. LaValle. Rrt-connect: An efficient approach to single-query path planning. In Proceedings 2000 ICRA. Millennium conference. IEEE international conference on robotics and automation. Symposia proceedings (Cat. No. 00CH37065), volume 2, pages 995–1001. IEEE, 2000.

[89] R. Geraerts and M. H. Overmars. Creating high-quality paths for motion planning. The international journal of robotics research, 26(8):845–863, 2007.

[90] R. Tedrake and the Drake Development Team. Drake: Model-based design and verification for robotics, 2019. URL https://drake.mit.edu.

[91] A. Wächter and L. T. Biegler. On the implementation of an interior-point filter line-search algorithm for large-scale nonlinear programming. Mathematical programming, 106(1):25–57, 2006.

[92] A. Khazatsky, K. Pertsch, S. Nair, A. Balakrishna, S. Dasari, S. Karamcheti, S. Nasiriany, M. K. Srirama, L. Y. Chen, K. Ellis, P. D. Fagan, J. Hejna, M. Itkina, M. Lepert, Y. J. Ma, P. T. Miller, J. Wu, S. Belkhale, S. Dass, H. Ha, A. Jain, A. Lee, Y. Lee, M. Memmel, S. Park, I. Radosavovic, K. Wang, A. Zhan, K. Black, C. Chi, K. B. Hatch, S. Lin, J. Lu, J. Mercat, A. Rehman, P. R. Sanketi, A. Sharma, C. Simpson, Q. Vuong, H. R. Walke, B. Wulfe, T. Xiao, J. H. Yang, A. Yavary, T. Z. Zhao, C. Agia, R. Baijal, M. G. Castro, D. Chen, Q. Chen, T. Chung, J. Drake, E. P. Foster, J. Gao, V. Guizilini, D. A. Herrera, M. Heo, K. Hsu, J. Hu, M. Z. Irshad, D. Jackson, C. Le, Y. Li, K. Lin, R. Lin, Z. Ma, A. Maddukuri, S. Mirchandani, D. Morton, T. Nguyen, A. O'Neill, R. Scalise, D. Seale, V. Son, S. Tian, E. Tran, A. E. Wang, Y. Wu, A. Xie, J. Yang, P. Yin, Y. Zhang, O. Bastani, G. Berseth, J. Bohg, K. Goldberg, A. Gupta, A. Gupta, D. Jayaraman, J. J. Lim, J. Malik, R. Martín-Martín, S. Ramamoorthy, D. Sadigh, S. Song, J. Wu, M. C. Yip, Y. Zhu, T. Kollar, S. Levine, and C. Finn. Droid: A large-scale in-the-wild robot manipulation dataset. 2024.

[93] M. M. Hong, J. Zhang, A. Nagabandi, and A. Gupta. Tmrl: Diffusion timestep-modulated pretraining enables exploration for efficient policy finetuning, 2026. URL https://arxiv.org/abs/2605.12236.

[94] J. Ho and T. Salimans. Classifier-free diffusion guidance, 2022. URL https://arxiv.org/abs/2207.12598.

A Ambient Diffusion: Theoretical Foundations 25
A.1 Spectral Power Law implies fast contraction through noise . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
A.2 Spectral Power Law implies locality . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
A.3 Theoretical Justification of the Classifier-Based Annotation ..... 30
B Variance-Preserving vs Variance Exploding Diffusion Processes 31
C VP Ambient Loss Derivation 32
D Characterization of Robot Action Data and Action Diffusion 34
E Pseudocode and Possible Extensions 36
E.1 Ambient Diffusion Policy Pseudocode ..... 36
E.2 Diffusion Policy Implementation Details ..... 36
E.3 Future Direction: Classifier-Based Rejection Sampling ..... 37
F Experiments Overview 38
G Controlled Experiments 39
G.1 Noisy Trajectories: 2D Maze ..... 39
G.2 Noisy Trajectories: Neural Motion Planning ..... 40
G.3 Sim-and-Real Co-training: Planar Pushing ..... 43
G.4 Task Mismatch: Block Sorting ..... 44
H Scaling Experiments: Open X-Embodiment 46
H.1 OXE Experiments ..... 46
H.2 OXE Ablations ..... 47
I Finetuning 49
I.1 Implementation ..... 49
I.2 Table Cleaning (OXE) ..... 50
J Distribution Shifts in the Observation Space 50
J.1 Noising the Observations ..... 51
J.2 Classifier-Free Guidance ..... 51

## A Ambient Diffusion: Theoretical Foundations

This appendix provides the proofs of the theoretical results stated in Section 5: the spectral power law implies both fast contraction through noise (Theorem 1) and locality (Theorem 2) of the optimal denoiser. We then formally justify the classifier-based $t_{\mathrm{min}}$ annotation method from Phase 1 of our algorithm (Theorem 4). The Spectral Power Law (Definition 1) and the general TV-contraction bound of Eq. (4) are stated in Section 5.

Preliminaries. Given distributions P, Q, we recall the following information-theoretic divergences, used throughout this appendix:

$$
\mathrm{d} _ {\mathrm{TV}} (P, Q) = \mathbb {E} _ {Q} \left| 1 - \frac {d P}{d Q} \right|,
$$

$$
\chi^ {2} (P \| Q) = \mathbb {E} _ {Q} \left(\frac {d P}{d Q}\right) ^ {2} - 1,
$$

$$
\operatorname{LC} (P, Q) = 2 \chi^ {2} \left(P \| \frac {1}{2} P + \frac {1}{2} Q\right).
$$

Pinsker's inequality gives $\mathrm{d}_{\mathrm{TV}}(P,Q)^{2} \leq \frac{1}{2}\mathrm{KL}(P,Q)$ , and the LeCam inequality gives $\mathrm{d}_{\mathrm{TV}}(P,Q)^{2} \leq \frac{1}{2}\mathrm{LC}(P,Q)$ . We refer the reader to [77] for further background.

## A.1 Spectral Power Law implies fast contraction through noise

Proof of Theorem 1. We start by making the observation that since the Discrete Fourier Transform (DFT) is an invertible operation, it holds that:

$$
\mathrm{d} _ {\mathrm{KL}} (p _ {t} \| q _ {t}) = \mathrm{d} _ {\mathrm{KL}} (\mathcal {F} \# p _ {t} \| \mathcal {F} \# q _ {t}).\tag{7}
$$

We denote with $F$ the DFT-matrix. Multiplying with $F$ vectors that are distributed according to a zero-mean Gaussian gives another zero-mean Gaussian distribution with covariance:

$$
\Sigma_ {\mathcal {F} \# p _ {t}} = \mathbb {E} _ {x \sim p _ {t}} \left[ (F x) (F x) ^ {*} \right]\tag{8}
$$

$$
= F \mathbb {E} _ {x \sim p _ {t}} [ x x ^ {T} ] F ^ {*}\tag{9}
$$

$$
= F \Sigma_ {p _ {t}} F ^ {*}.\tag{10}
$$

We have that $\Sigma_{p_{t}} = \Sigma_{p} + \sigma_{t}^{2}I$ , where $\Sigma_{p}$ is a circulant matrix (stationarity assumption). Hence, $\Sigma_{p_{t}}$ is also a circulant matrix, and thus it can be diagonalized as follows:

$$
\Sigma_ {p _ {t}} = F ^ {*} (\Lambda_ {p} + \sigma_ {t} ^ {2} I) F.\tag{11}
$$

Hence,

$$
\Sigma_ {\mathcal {F} \# p _ {t}} = F F ^ {*} (\Lambda_ {p} + \sigma_ {t} ^ {2} I) F F ^ {*}\tag{12}
$$

$$
= \Lambda_ {p} + \sigma_ {t} ^ {2} I.\tag{13}
$$

Similarly, $\Sigma_{\mathcal{F}\# q_t} = \Lambda_q + \sigma_t^2 I$ .

Using the formula for the KL distance between two zero-mean, diagonal, multivariate Gaussians, we get:

$$
\mathrm{d} _ {\mathrm{KL}} (p _ {t} \| q _ {t}) \leq \frac {1}{2} \sum_ {i = 0} ^ {N - 1} \frac {(\Lambda_ {p}) _ {i , i} + \sigma_ {t} ^ {2}}{(\Lambda_ {q}) _ {i , i} + \sigma_ {t} ^ {2}} - \ln \left(\frac {(\Lambda_ {p}) _ {i , i} + \sigma_ {t} ^ {2}}{(\Lambda_ {q}) _ {i , i} + \sigma_ {t} ^ {2}}\right) - 1\tag{14}
$$

$$
= \frac {1}{2} \sum_ {f = 0} ^ {N - 1} \frac {S _ {p} (f) + \sigma_ {t} ^ {2}}{S _ {q} (f) + \sigma_ {t} ^ {2}} - \ln \left(\frac {S _ {p} (f) + \sigma_ {t} ^ {2}}{S _ {q} (f) + \sigma_ {t} ^ {2}}\right) - 1\tag{15}
$$

$$
\leq \sum_ {f > f ^ {\star}} ^ {\lfloor N / 2 \rfloor} \frac {S _ {p} (f) + \sigma_ {t} ^ {2}}{S _ {q} (f) + \sigma_ {t} ^ {2}} - \ln \left(\frac {S _ {p} (f) + \sigma_ {t} ^ {2}}{S _ {q} (f) + \sigma_ {t} ^ {2}}\right) - 1,\tag{16}
$$

where in the last inequality we used the fact that the spectrum is symmetric around the Nyquist frequency and that p and q agree on all frequencies up to $f^{\star}$ .

Let's consider the function: $g(r) = r - \ln r - 1$ . We will start by showing that $g(r) \leq (r - 1)^2$ for all $r \in [1/2, \infty)$ .

Let

$$
h (r) = g (r) - (r - 1) ^ {2}\tag{17}
$$

$$
= r - \ln r - 1 - r ^ {2} + 2 r - 1\tag{18}
$$

$$
= - r ^ {2} + 3 r - \ln r - 2.\tag{19}
$$

We hence have:

$$
h ^ {\prime} (r) = - 2 r + 3 - \frac {1}{r}\tag{20}
$$

$$
= \frac {- 2 r ^ {2} + 3 r - 1}{r}.\tag{21}
$$

The determinant of the nominator is $\Delta = 1$ and there are two points that make $h'(r) = 0$ , which are:

$$
r _ {1} = \frac {1}{2}, \quad r _ {2} = 1.\tag{22}
$$

We have that $h'(r)$ is negative for $r > 1$ and for $r \in (0, 1/2)$ . Hence, in $[1/2, \infty)$ the maximum of $h(r)$ is $h(1) = 0$ . Hence, the function $h(r)$ is $\leq 0$ for all $r \in [1/2, \infty)$ .

We now want to get some bounds on the quantity $\frac{S_p(f) + \sigma_t^2}{S_q(f) + \sigma_t^2}$ . First, note that:

$$
\frac {S _ {p} (f) + \sigma_ {t} ^ {2}}{S _ {q} (f) + \sigma_ {t} ^ {2}} \geq \frac {\sigma_ {t} ^ {2}}{\max (S _ {p} (f) , S _ {q} (f)) + \sigma_ {t} ^ {2}}.\tag{23}
$$

For all $\lfloor N/2\rfloor \geq f > f^{\star}$ , we further have (by Power-law tail assumption) that:

$$
\max (S _ {p} (f), S _ {q} (f)) \leq C f ^ {- \alpha},\tag{24}
$$

which gives that:

$$
\frac {S _ {p} (f) + \sigma_ {t} ^ {2}}{S _ {q} (f) + \sigma_ {t} ^ {2}} \geq \frac {\sigma_ {t} ^ {2}}{C f ^ {- \alpha} + \sigma_ {t} ^ {2}}.\tag{25}
$$

By assumption, we have that:

$$
\sigma_ {t} ^ {2} \geq C \frac {1}{(f ^ {\star}) ^ {\alpha}}.\tag{26}
$$

But:

$$
\frac {C}{f ^ {\alpha}} <   \frac {C}{(f ^ {\star}) ^ {\alpha}}, \quad \forall f \text {   such   that   } f ^ {\star} <   f \leq \lfloor N / 2 \rfloor .\tag{27}
$$

Hence,

$$
C f ^ {- \alpha} <   \sigma_ {t} ^ {2}.\tag{28}
$$

Plugging this into Eq. (25), we get:

$$
\frac {S _ {p} (f) + \sigma_ {t} ^ {2}}{S _ {q} (f) + \sigma_ {t} ^ {2}} \geq 1 / 2.
$$

(29)

By combining Eq. 29 and the fact that $g(r) \leq (r - 1)^2$ for all $r \in [1/2, \infty)$ , we can write Eq. (16) as follows:

$$
\mathrm{d} _ {\mathrm{KL}} (p _ {t} \| q _ {t}) \leq \sum_ {f > f ^ {*}} ^ {\lfloor N / 2 \rfloor} \left(\frac {S _ {p} (f) - S _ {q} (f)}{S _ {q} (f) + \sigma_ {t} ^ {2}}\right) ^ {2}\tag{30}
$$

$$
\leq \sum_ {f > f ^ {\star}} ^ {\lfloor N / 2 \rfloor} \left(\frac {2 \max (S _ {p} (f) , S _ {q} (f))}{\sigma_ {t} ^ {2}}\right) ^ {2}\tag{31}
$$

$$
\leq \sum_ {f > f ^ {\star}} ^ {\lfloor N / 2 \rfloor} 4 \frac {C ^ {2} f ^ {- 2 \alpha}}{\sigma_ {t} ^ {4}}\tag{32}
$$

$$
= \frac {4 C ^ {2}}{\sigma_ {t} ^ {4}} \sum_ {f > f ^ {\star}} ^ {\lfloor N / 2 \rfloor} \left(\frac {1}{f}\right) ^ {2 \alpha}\tag{33}
$$

$$
\leq \frac {4 C ^ {2}}{\sigma_ {t} ^ {4}} \int_ {f ^ {\star}} ^ {\infty} \left(\frac {1}{f}\right) ^ {2 \alpha} \mathrm{d} f\tag{34}
$$

$$
= \frac {4 C ^ {2}}{\sigma_ {t} ^ {4} (2 \alpha - 1) (f ^ {\star}) ^ {2 \alpha - 1}}.\tag{35}
$$

The TV distance bound follows from a direct application of Pinsker's inequality.

## A.2 Spectral Power Law implies locality

Proof of Theorem 2. Since $X_{0}$ is distributed according to a Gaussian distribution, the MMSE optimal denoiser is a linear function, given as:

$$
\mathbb {E} [ X _ {0} | X _ {t} = x ] = \Sigma (\Sigma + \sigma_ {t} ^ {2} I) ^ {- 1} x.\tag{36}
$$

From the stationarity assumption, $\Sigma$ is a circulant matrix and so is the matrix $\Sigma + \sigma_{t}^{2}I$ . The inverse of a circulant matrix is another circulant matrix, and multiplying two circulant matrices gives another circulant matrix. Hence, the whole matrix $\Sigma(\Sigma + \sigma_{t}^{2}I)^{-1}$ is circulant.

Multiplication with a circulant matrix can be written as a circulant convolution; hence, we can express the optimal denoiser in the following equivalent way:

$$
(\mathbb {E} [ X _ {0} | X _ {t} = x ]) _ {i} = \sum_ {j = 0} ^ {N - 1} k _ {t} ((i - j) \mod N) x _ {j},\tag{37}
$$

where $k_{t}$ is the first column of the matrix $\Sigma(\Sigma + \sigma_{t}^{2}I)^{-1}$ . Let us now analyze how the prediction of the optimal denoiser changes when, instead of feeding the full input, we input a masked version $M_{i}x$ of the input where $M_{i}$ is a circulant mask.

We have that:

$$
\operatorname{error} _ {i} = \left| \sum_ {j = 0} ^ {N - 1} k _ {t} ((i - j) \mod N) x _ {j} \right.
$$

$$
- \sum_ {j = 0} ^ {N - 1} k _ {t} ((i - j) \mod N) M _ {i} x _ {j} \Bigg |\tag{38}
$$

$$
= \bigg | \sum_ {j: \mathrm{dist} (i, j) > L} k _ {t} ((i - j) \mod N) x _ {j} \bigg |,\tag{39}
$$

$$
\leq | | x | | _ {\infty} \sum_ {j: \operatorname{dist} (i, j) > L} \left| k _ {t} ((i - j) \mod N) \right|,\tag{40}
$$

where $\text{dist}(i,j)=\min(|i-j|,N-|i-j|)$ . Let $m=(i-j)\mod N$ . First note that $L+1\leq m\leq\lfloor N/2\rfloor$ . Second, note that for each value m, there are two indices j that can give this value. Hence, we can rewrite the sum as:

$$
\operatorname{error} _ {i} \leq | | x | | _ {\infty} \sum_ {m = L + 1} ^ {\lfloor N / 2 \rfloor} | k _ {t} (m) | + | k _ {t} (N - m) |.\tag{41}
$$

We now need to analyze what the kernel $k_{t}$ is. We have defined this vector as the first column of the matrix $\Sigma(\Sigma + \sigma_{t}^{2}I)^{-1}$ . Because the matrix is circulant, we have that $k_{t}(m) = k_{t}(N - m)$ . Hence,

$$
\operatorname{error} _ {i} \leq 2 | | x | | _ {\infty} \sum_ {m = L + 1} ^ {\lfloor N / 2 \rfloor} | k _ {t} (m) |.\tag{42}
$$

By definition:

$$
k _ {t} (m) = (\Sigma (\Sigma + \sigma_ {t} ^ {2} I) ^ {- 1} e _ {1}) (m).\tag{43}
$$

The matrix $\Sigma(\Sigma + \sigma_{t}^{2}I)^{-1}$ is circulant, and hence it is diagonalized by the DFT. Hence, we have:

$$
k _ {t} (m) = \left(F ^ {*} W F e _ {1}\right) (m),\tag{44}
$$

where W is the diagonal matrix with diagonal entries $W_{f,f} = \frac{S(f)}{S(f) + \sigma_{t}^{2}}$ .

We observe now that $k_{t}(m)$ is exactly the Inverse Discrete Fourier Transform of the Wiener filter. Hence, $k_{t}(m)$ can be equivalently written as:

$$
k _ {t} (m) = \frac {1}{N} \sum_ {f = 0} ^ {N - 1} W _ {f, f} e ^ {i \frac {2 \pi f m}{N}}.\tag{45}
$$

Because the original signal $X_{0}$ is real-valued, its power spectrum and the resulting Wiener filter $W_{f} = W_{f,f}$ are symmetric around the Nyquist frequency, meaning $W_{N-f} = W_{f}$ . This conjugate symmetry perfectly cancels the imaginary sine components, allowing us to write the kernel entirely in terms of real cosines:

$$
k _ {t} (m) = \frac {1}{N} \sum_ {f = 0} ^ {N - 1} W _ {f} \cos \left(\frac {2 \pi f m}{N}\right).\tag{46}
$$

To extract the spatial decay of this kernel as m increases, let $\omega_{m} = \frac{2\pi m}{N}$ . We multiply both sides of the equation by the phase-shifting factor $2(1 - \cos \omega_{m})$ :

$$
\begin{array}{c} 2 (1 - \cos \omega_ {m}) k _ {t} (m) \\ = \frac {1}{N} \sum_ {f = 0} ^ {N - 1} W _ {f} \left[ 2 \cos (f \omega_ {m}) - 2 \cos (f \omega_ {m}) \cos (\omega_ {m}) \right]. \end{array}\tag{47}
$$

Applying the standard trigonometric identity $2 \cos(A) \cos(B) = \cos(A + B) + \cos(A - B)$ to the second term gives:

$$
\begin{array}{c} 2 (1 - \cos \omega_ {m}) k _ {t} (m) \\ = \frac {1}{N} \sum_ {f = 0} ^ {N - 1} W _ {f} \Big [ 2 \cos (f \omega_ {m}) - \cos ((f + 1) \omega_ {m}) \\ - \cos ((f - 1) \omega_ {m}) \Big ]. \end{array}\tag{48}
$$

Because the discrete spectrum is periodic ( $W_N = W_0$ ) and the cosine function is periodic over $N$ , we can shift the indices of summation for the $(f + 1)$ and $(f - 1)$ terms to group them by the common $\cos (f\omega_m)$ factor. This perfectly regroups the sum into the central second difference $\Delta^2 W_f = W_{f + 1} - 2W_f + W_{f - 1}$ :

$$
2 (1 - \cos \omega_ {m}) k _ {t} (m) = - \frac {1}{N} \sum_ {f = 0} ^ {N - 1} (\Delta^ {2} W _ {f}) \cos (f \omega_ {m}).\tag{49}
$$

We now take the absolute value of both sides. For the left side, we apply the half-angle trigonometric identity $2(1 - \cos\omega_{m}) = 4\sin^{2}\left(\frac{\omega_{m}}{2}\right) = 4\sin^{2}\left(\frac{\pi m}{N}\right)$ . Since $|\cos(f\omega_{m})| \leq 1$ , it drops out of the right side:

$$
4 \sin^ {2} \left(\frac {\pi m}{N}\right) | k _ {t} (m) | \leq \frac {1}{N} \sum_ {f = 0} ^ {N - 1} | \Delta^ {2} W _ {f} |.\tag{50}
$$

Crucially, the assumption that $S(f)$ follows a spectral power law dictates the structural behavior of this Wiener filter. For high frequencies $f \geq 1$ , the power law $S(f) = Cf^{-\alpha}$ ensures that $W_{f}$ monotonically and smoothly decays to zero. We can bound the discrete slope of this filter by analyzing its continuous extension $W(f) = \frac{C}{C + \sigma_{t}^{2} f^{\alpha}}$ . Its derivative is:

$$
W ^ {\prime} (f) = \frac {- \alpha C \sigma_ {t} ^ {2} f ^ {\alpha - 1}}{(C + \sigma_ {t} ^ {2} f ^ {\alpha}) ^ {2}}.\tag{51}
$$

By the Arithmetic Mean - Geometric Mean inequality, $(C + \sigma_{t}^{2}f^{\alpha})^{2} \geq 4C\sigma_{t}^{2}f^{\alpha}$ . Substituting this into the derivative gives a strict bound on the slope:

$$
| W ^ {\prime} (f) | \leq \frac {\alpha C \sigma_ {t} ^ {2} f ^ {\alpha - 1}}{4 C \sigma_ {t} ^ {2} f ^ {\alpha}} = \frac {\alpha}{4 f} \leq \frac {\alpha}{4}.\tag{52}
$$

The sequence of discrete first differences $\Delta W_{f}$ samples this continuous slope. Over the positive half of the spectrum, the slope starts near zero, reaches a maximum magnitude bounded by $\alpha/4$ , and gradually returns to zero. Because the function is convex in the tail, the second differences maintain a constant sign, allowing the absolute sum to telescope perfectly. The total absolute second variation for the positive half of the spectrum is therefore exactly bounded by twice the maximum slope, $2(\alpha/4) = \alpha/2$ . Due to the conjugate symmetry of the spectrum, the negative frequencies contribute an identical amount.

Therefore, the total absolute second variation over the entire spectrum is strictly bounded by the power-law exponent:

$$
\sum_ {f = 0} ^ {N - 1} | \Delta^ {2} W _ {f} | \leq \alpha .\tag{53}
$$

Substituting this variation bound back into our inequality yields:

$$
4 \sin^ {2} \left(\frac {\pi m}{N}\right) | k _ {t} (m) | \leq \frac {\alpha}{N}.\tag{54}
$$

For the spatial domain $m \leq \lfloor N / 2 \rfloor$ , the argument $\frac{\pi m}{N}$ falls within the interval $[0, \pi / 2]$ . In this domain, we can apply the linear lower bound for the sine function, $\sin(x) \geq \frac{2x}{\pi}$ , which gives $\sin^2\left(\frac{\pi m}{N}\right) \geq \frac{4m^2}{N^2}$ . Substituting this yields:

$$
4 \left(\frac {4 m ^ {2}}{N ^ {2}}\right) | k _ {t} (m) | \leq \frac {\alpha}{N} \implies | k _ {t} (m) | \leq \frac {\alpha N}{1 6 m ^ {2}}.\tag{55}
$$

We have thus proven that the optimal denoiser kernel decays spatially at a rate of $\mathcal{O}(1/m^{2})$ . Returning to the error of the masked denoiser, we substitute this polynomial decay bound into the error sum:

$$
\begin{array}{l} \text {error} _ {i} \leq 2 \| x \| _ {\infty} \sum_ {m = L + 1} ^ {\lfloor N / 2 \rfloor} | k _ {t} (m) | \\ \leq \frac {\alpha N \| x \| _ {\infty}}{8} \sum_ {m = L + 1} ^ {\lfloor N / 2 \rfloor} \frac {1}{m ^ {2}}. \end{array}\tag{56}
$$

We can upper-bound this remaining discrete sum by the corresponding continuous integral over the spatial tail:

$$
\sum_ {m = L + 1} ^ {\lfloor N / 2 \rfloor} \frac {1}{m ^ {2}} \leq \int_ {L} ^ {\infty} \frac {1}{y ^ {2}} \mathrm{d} y = \frac {1}{L}.\tag{57}
$$

Hence, the approximation error of the masked denoiser is strictly bounded by:

$$
\mathrm{error} _ {i} \leq \frac {\alpha N \| x \| _ {\infty}}{8 L}.\tag{58}
$$

This demonstrates that as the mask size L increases, the error drops off linearly, mathematically proving that the spectral power law forces the MMSE optimal denoiser to be spatially local. □

## A.3 Theoretical Justification of the Classifier-Based Annotation

We prove that the classifier-based $t_{min}$ annotation method introduced in Phase 1 of Section 6 comes with a formal guarantee: suppose the classifier, $f_{\theta,4}$ is $\varepsilon$ -optimal and the threshold is $\tau$ . Then annotating $t_{min}$ using the method described in Section 6 bounds the LeCam distance between the noisy distributions $p_{t}$ and $q_{t}$ (Theorem 4). Simply put, it guarantees that $p_{t}$ and $q_{t}$ are similar (i.e. contracted) for $t > t_{min}$ .

Classifier threshold implies distributional closeness. Consider the population cross-entropy loss

$$
\mathcal {L} _ {\mathrm{CE}} (f) = - \mathbb {E} _ {c, x} [ (1 - c) \log (1 - f (x)) + c \log f (x) ],
$$

where we first sample $c \sim \operatorname{Bern}(1/2)$ and then sample $x \sim P$ if $c = 1$ or $x \sim Q$ otherwise. Let $M = \frac{1}{2} P + \frac{1}{2} Q$ denote the mixture. The optimal classifier is

$$
f ^ {*} = \arg \min _ {f} \mathcal {L} _ {\mathrm{CE}} (f), \qquad f ^ {*} (x) = \frac {d P}{d P + d Q} (x) = \frac {1}{2} \frac {d P}{d M} (x).\tag{59}
$$

Theorem 3 (Optimal classifier threshold implies distributional closeness). Let $f^{*}$ be the optimal classifier in Eq. (59). Then

$$
\mathbb {E} _ {Q} f ^ {*} (x) = \frac {1}{2} - \frac {1}{2} \mathrm{LC} (P, Q).
$$

In particular, $\mathbb{E}_Qf^* (x)\geq \frac{1}{2} -\tau$ implies $\mathrm{LC}(P,Q)\leq 2\tau$

Proof. We compute

$$
\mathbb {E} _ {Q} f ^ {*} = \int \frac {d P d Q}{d P + d Q} = \frac {1}{2} \int \frac {(d P + d Q) ^ {2} - d P ^ {2} - d Q ^ {2}}{d P + d Q}.
$$

Then

$$
\mathbb {E} _ {Q} f ^ {*} = 1 - \frac {1}{2} \int \frac {d P ^ {2}}{d P + d Q} - \frac {1}{2} \int \frac {d Q ^ {2}}{d P + d Q}.
$$

Noting $\int \frac{dP^2}{dP + dQ} = \frac{1}{2}\big(\chi^2 (P\| M) + 1\big)$ gives

$$
\mathbb {E} _ {Q} f ^ {*} = \frac {1}{2} - \frac {1}{2} \chi^ {2} (P \| M) - \frac {1}{2} \chi^ {2} (Q \| M).
$$

Using $\mathrm{LC}(P,Q)=2\chi^{2}(P\|M)=2\chi^{2}(Q\|M)$ , we conclude

$$
\mathbb {E} _ {Q} f ^ {*} = \frac {1}{2} - \frac {1}{2} \mathrm{LC} (P, Q).
$$

The threshold condition $E_{Q}f^{*}\geq\frac{1}{2}-\tau$ then immediately gives $\mathrm{LC}(P,Q)\leq2\tau$ .

We now extend this result to approximate classifiers trained on the cross-entropy loss.

Theorem 4 (Approximate classifier threshold implies distributional closeness). Suppose $f_{\theta} : \mathbb{R}^d \to [0,1]$ is an $\varepsilon$ -optimal classifier, i.e. $\mathcal{L}_{\mathrm{CE}}(f_{\theta}) \leq \mathcal{L}_{\mathrm{CE}}(f^{*}) + \varepsilon$ . Then $\mathbb{E}_Q f_{\theta}(x) \geq \frac{1}{2} - \tau$ implies $\operatorname{LC}(P, Q) \leq 2\tau + 2\sqrt{\varepsilon}$ .

Proof. The cross-entropy loss admits the decomposition

$$
\mathcal {L} _ {\mathrm{CE}} (f) = \mathbb {E} _ {x \sim M} d \left(f ^ {*} (x) \| f (x)\right) + \text { const },
$$

where $d(p||q) = \mathrm{KL}(\mathrm{Bern}(p)||\mathrm{Bern}(q))$ is the binary KL divergence. The $\varepsilon$ -accuracy condition implies $\mathbb{E}_{x\sim M}d(f^{*}(x)||f(x)) \leq \varepsilon$ , and by non-negativity $\mathbb{E}_{x\sim Q}d(f^{*}(x)||f(x)) \leq 2\varepsilon$ . Applying Pinsker's inequality and $\mathrm{d}_{\mathrm{TV}}(p,q) = |p - q|$ for Bernoulli variables yields

$$
\mathbb {E} _ {x \sim Q} (f _ {\theta} (x) - f ^ {*} (x)) ^ {2} \leq \varepsilon .
$$

By Jensen, $\mathbb{E}_{x\sim Q}f^{*}(x)\geq \mathbb{E}_{x\sim Q}f_{\theta}(x) - \sqrt{\varepsilon}$ , so

$$
\mathbb {E} _ {x \sim Q} f ^ {*} (x) \geq \frac {1}{2} - \tau - \sqrt {\varepsilon}.
$$

Applying Theorem 3 recovers the claim.

## B Variance-Preserving vs Variance Exploding Diffusion Processes

The paper presents Ambient Diffusion Policy from the variance-exploding perspective of diffusion $[55]$ to lighten notation. Our implementation builds off the original Diffusion Policy $[23]$ work, which uses a variance-preserving implementation. We first show that the two perspectives are equivalent up to a change of variables.

The variance exploding (VE) and variance preserving (VP) forward processes are defined in Eq. (60) and (61) respectively [55]:

$$
X _ {t} = X _ {0} + \sigma_ {\mathrm{VE}} (t) Z,\tag{60}
$$

$$
X _ {t} = \sqrt {1 - \sigma_ {\mathrm{VP}} ^ {2} (t)} X _ {0} + \sigma_ {\mathrm{VP}} (t) Z,\tag{61}
$$

where $Z \sim \mathcal{N}(0, I)$ , $\sigma_{\mathrm{VE}}(t)$ is an unbounded increasing function with $\sigma_{\mathrm{VE}}(0) = 0$ , and $\sigma_{\mathrm{VP}}(t)$ is an increasing function with $\sigma_{\mathrm{VP}}(0) = 0$ and $\sigma_{\mathrm{VP}}(T) = 1$ . Without loss of generality, we assume the data is normalized so that $\operatorname{Var}[X_0] = 1$ .

Proposition 1. The VP and VE processes are equivalent up to a time-dependent rescaling of space.

Proof. Given a VE schedule $\sigma_{\mathrm{VE}}(t)$ , define

$$
\sigma_ {\mathrm{VP}} (t) := \frac {\sigma_ {\mathrm{VE}} (t)}{\sqrt {1 + \sigma_ {\mathrm{VE}} ^ {2} (t)}}, \qquad c (t) := \sqrt {1 + \sigma_ {\mathrm{VE}} ^ {2} (t)}.\tag{62}
$$

Under this mapping,

$$
\mathrm{SNR} _ {\mathrm{VE}} (t) = \frac {1}{\sigma_ {\mathrm{VE}} ^ {2} (t)} = \frac {1 - \sigma_ {\mathrm{VP}} ^ {2} (t)}{\sigma_ {\mathrm{VP}} ^ {2} (t)} = \mathrm{SNR} _ {\mathrm{VP}} (t),\tag{63}
$$

so both processes have the same SNR at every t. The rescaled VE variable, $\tilde{X}_{t}$ , satisfies

$$
\widetilde {X} _ {t} := \frac {X _ {t} ^ {\mathrm{VE}}}{c (t)} = X _ {t} ^ {\mathrm{VP}}.\tag{64}
$$

Thus, the VP process is the VE process in normalized coordinates $x_{t}^{\mathrm{VE}} \mapsto x_{t}^{\mathrm{VE}} / c(t) = x_{t}^{\mathrm{VP}}$ . ☐

Proposition 2. The score functions of the VP and VE processes are related by:

$$
s ^ {\mathrm{VP}} (\widetilde {x} _ {t}, t) = c (t) \cdot s ^ {\mathrm{VE}} (c (t) \widetilde {x} _ {t}, t).
$$

(65)

Proof. Let $p_{t}^{VP}$ and $p_{t}^{VE}$ denote the marginal densities of $X_{t}^{VP}$ and $X_{t}^{VE}$ respectively. Given the change of variables $x_{t}^{\mathrm{VE}} = c(t) x_{t}^{\mathrm{VP}}$ , we have that $p_{t}^{\mathrm{VP}}(x_{t}^{\mathrm{VP}}) = c(t)^{d} p_{t}^{\mathrm{VE}}(c(t) x_{t}^{\mathrm{VP}})$ , where d is the dimension. Taking the gradient with respect to $x_{t}^{VP}$ :

$$
\begin{array}{r l} & s ^ {\mathrm{VP}} (x _ {t} ^ {\mathrm{VP}}, t) = \nabla_ {x _ {t} ^ {\mathrm{VP}}} \log p _ {t} ^ {\mathrm{VP}} (x _ {t} ^ {\mathrm{VP}}) \\ & \qquad = \nabla_ {x _ {t} ^ {\mathrm{VP}}} \log p _ {t} ^ {\mathrm{VE}} (c (t) x _ {t} ^ {\mathrm{VP}}) + \underbrace {\nabla_ {x _ {t} ^ {\mathrm{VP}}} \log c (t) ^ {d}} _ {} 0 \\ & \qquad = c (t) \cdot s ^ {\mathrm{VE}} \big (c (t) x _ {t} ^ {\mathrm{VP}}, t \big), \end{array}\tag{66}
$$

where the last line follows from the chain rule.

Proposition 3. Let $h_{VE}^{*}$ and $h_{VP}^{*}$ denote the optimal denoisers for the VE and VP processes respectively. Then:

$$
h _ {\mathrm{VP}} ^ {*} (x _ {t} ^ {\mathrm{VP}}, t) = h _ {\mathrm{VE}} ^ {*} \big (c (t) x _ {t} ^ {\mathrm{VP}}, t \big).\tag{67}
$$

Proof. By Tweedie's formula, the optimal denoisers are related to the scores via:

$$
h _ {\mathrm{VE}} ^ {*} (x _ {t} ^ {\mathrm{VE}}, t) = x _ {t} ^ {\mathrm{VE}} + \sigma_ {\mathrm{VE}} ^ {2} (t) s ^ {\mathrm{VE}} (x _ {t} ^ {\mathrm{VE}}, t),\tag{68}
$$

$$
h _ {\mathrm{VP}} ^ {*} (x _ {t} ^ {\mathrm{VP}}, t) = \frac {x _ {t} ^ {\mathrm{VP}} + \sigma_ {\mathrm{VP}} ^ {2} (t) s ^ {\mathrm{VP}} (x _ {t} ^ {\mathrm{VP}} , t)}{\sqrt {1 - \sigma_ {\mathrm{VP}} ^ {2} (t)}}.\tag{69}
$$

Substituting the score correspondence from Proposition 2 and the definition of $c(t)$ in (62) yields the desired result.

Therefore, any result derived in one framework translates to the other via $c(t) = \sqrt{1 + \sigma_{\mathrm{VE}}^2(t)}$ .

## C VP Ambient Loss Derivation

In the original Ambient Diffusion work, Daras et al. [24] propose an alternative loss function, known as the Ambient loss. Unlike the denoising loss from DDPM [54], the Ambient loss assumes no access to the clean samples, $x_0$ , and instead uses $x_{t_{\mathrm{min}}}$ as the prediction target. Nonetheless, it is minimized by the same denoiser as the DDPM loss. In the image domain, the Ambient loss has been shown to improve generalization. Daras et al. [24] derive the Ambient loss for VE diffusion. In this section, we derive it for VP diffusion.

Proposition 4. The $\epsilon$ -prediction VP Ambient loss for $t > t_{\mathrm{min}}$ is given by

$$
\mathcal {L} (\theta) = \mathbb {E} \left[ \left\| \epsilon_ {\theta} (x _ {t}, t) - \frac {\sigma_ {t} (1 - \sigma_ {t _ {\min}} ^ {2})}{\sigma_ {t} ^ {2} - \sigma_ {t _ {\min}} ^ {2}}   x _ {t} + \frac {\sigma_ {t} \sqrt {(1 - \sigma_ {t} ^ {2}) (1 - \sigma_ {t _ {\min}} ^ {2})}}{\sigma_ {t} ^ {2} - \sigma_ {t _ {\min}} ^ {2}}   x _ {t _ {\min}} \right\| _ {2} ^ {2} \right],\tag{70}
$$

and minimized by $\epsilon_{\theta}^{*} = \mathbb{E}[Z\mid X_{t} = x_{t}]$ .

Proof. Let $X_{t}$ and $Z$ be the random variables from the forward process in VP diffusion, Eq. (61). Then:

$$
\mathbb {E} [ Z \mid X _ {t} = x _ {t} ] = \frac {x _ {t} - \sqrt {1 - \sigma_ {t} ^ {2}} \mathbb {E} [ X _ {0} \mid X _ {t} = x _ {t} ]}{\sigma_ {t}}.\tag{71}
$$

By the double Tweedie identity from Daras et al. [24]:

$$
\mathbb {E} [ X _ {0} \mid X _ {t} = x _ {t} ] = \frac {\sigma_ {t} ^ {2} \sqrt {1 - \sigma_ {t _ {\min}} ^ {2}}}{\sigma_ {t} ^ {2} - \sigma_ {t _ {\min}} ^ {2}} \mathbb {E} [ X _ {t _ {\min}} \mid X _ {t} = x _ {t} ] - \frac {\sigma_ {t _ {\min}} ^ {2} \sqrt {1 - \sigma_ {t} ^ {2}}}{\sigma_ {t} ^ {2} - \sigma_ {t _ {\min}} ^ {2}} x _ {t}.
$$

(72)

Substituting Eq. (72) into Eq. (71) and rearranging gives:

$$
\mathbb {E} [ X _ {t _ {\mathrm{min}}} \mid X _ {t} = x _ {t} ] = \alpha_ {t} \mathbb {E} [ Z \mid X _ {t} = x _ {t} ] + \beta_ {t} x _ {t},\tag{73}
$$

$$
\alpha_ {t} := - \frac {\sigma_ {t} ^ {2} - \sigma_ {t _ {\mathrm{min}}} ^ {2}}{\sigma_ {t} \sqrt {(1 - \sigma_ {t} ^ {2}) (1 - \sigma_ {t _ {\mathrm{min}}} ^ {2})}},\tag{74}
$$

$$
\beta_ {t} := \frac {1 - \sigma_ {t _ {\min}} ^ {2}}{\sqrt {(1 - \sigma_ {t} ^ {2}) (1 - \sigma_ {t _ {\min}} ^ {2})}}.\tag{75}
$$

Define $g_{\theta}(x_{t},t):=\alpha_{t}\epsilon_{\theta}(x_{t},t)+\beta_{t}x_{t}$ . Since $t>t_{min}$ , we have $\alpha_{t}\neq0$ and the map $\epsilon_{\theta}\mapsto g_{\theta}$ is a bijection. Thus, minimizing

$$
\mathbb {E} \left[ \left\| g _ {\theta} (x _ {t}, t) - x _ {t _ {\min}} \right\| ^ {2} \right]\tag{76}
$$

over $g_{\theta}$ is equivalent to minimizing over $\epsilon_{\theta}$ . The minimizer of Eq. (76) is $g_{\theta}^{*} = \mathbb{E}[X_{t_{\min}} \mid X_{t} = x_{t}]$ , which by Eq. (73) implies $\epsilon_{\theta}^{*} = \mathbb{E}[Z \mid X_{t} = x_{t}]$ . Rescaling the loss in Eq. (76) by $\alpha_{t}^{2}$ yields:

$$
\mathbb {E} \left[ \left\| \epsilon_ {\theta} (x _ {t}, t) + \frac {\beta_ {t}}{\alpha_ {t}} x _ {t} - \frac {1}{\alpha_ {t}} x _ {t _ {\min}} \right\| _ {2} ^ {2} \right],\tag{77}
$$

which expands to the loss in the proposition.

Proposition 5. The $x_{0}$ -prediction VP Ambient loss for $t > t_{min}$ is given by

$$
\mathbb {E} \left[ \left\| h _ {\theta} (x _ {t}, t) + \frac {\sigma_ {t _ {\min}} ^ {2} \sqrt {1 - \sigma_ {t} ^ {2}}}{\sigma_ {t} ^ {2} - \sigma_ {t _ {\min}} ^ {2}} x _ {t} - \frac {\sigma_ {t} ^ {2} \sqrt {1 - \sigma_ {t _ {\min}} ^ {2}}}{\sigma_ {t} ^ {2} - \sigma_ {t _ {\min}} ^ {2}} x _ {t _ {\min}} \right\| _ {2} ^ {2} \right],\tag{78}
$$

and minimized by $h_{\theta}^{*} = \mathbb{E}[X_0 \mid X_t = x_t]$ .

Proof. Isolating $\mathbb{E}[X_{t_{\min}}\mid X_t = x_t]$ in the double Tweedie identity Eq. (72) gives:

$$
\mathbb {E} [ X _ {t _ {\mathrm{min}}} \mid X _ {t} = x _ {t} ] = \tilde {\alpha} _ {t} \mathbb {E} [ X _ {0} \mid X _ {t} = x _ {t} ] + \tilde {\beta} _ {t} x _ {t},\tag{79}
$$

$$
\tilde {\alpha} _ {t} := \frac {\sigma_ {t} ^ {2} - \sigma_ {t _ {\min}} ^ {2}}{\sigma_ {t} ^ {2} \sqrt {1 - \sigma_ {t _ {\min}} ^ {2}}}, \quad \tilde {\beta} _ {t} := \frac {\sigma_ {t _ {\min}} ^ {2} \sqrt {1 - \sigma_ {t} ^ {2}}}{\sigma_ {t} ^ {2} \sqrt {1 - \sigma_ {t _ {\min}} ^ {2}}}.\tag{80}
$$

Define $\tilde{g}_{\theta}(x_t, t) := \tilde{\alpha}_t h_{\theta}(x_t, t) + \tilde{\beta}_t x_t$ . Since $t > t_{\min}$ , then $\tilde{\alpha}_t \neq 0$ and the map $h_{\theta} \mapsto \tilde{g}_{\theta}$ is a bijection. Thus, minimizing

$$
\mathbb {E} \big [ \| \tilde {g} _ {\theta} (x _ {t}, t) - x _ {t _ {\mathrm{min}}} \| ^ {2} \big ]\tag{81}
$$

over $\tilde{g}_{\theta}$ is equivalent to minimizing over $h_\theta$ . The minimizer of Eq. (81) is $\tilde{g}_{\theta}^{*} = \mathbb{E}[X_{t_{\min}} \mid X_t = x_t]$ , which by Eq. (79) implies $h_\theta^* = \mathbb{E}[X_0 \mid X_t = x_t]$ . Rescaling the loss in Eq. (81) by $\tilde{\alpha}_t^2$ yields:

$$
\mathbb {E} \left[ \left\| h _ {\theta} (x _ {t}, t) + \frac {\tilde {\beta} _ {t}}{\tilde {\alpha} _ {t}} x _ {t} - \frac {1}{\tilde {\alpha} _ {t}} x _ {t _ {\min}} \right\| _ {2} ^ {2} \right],\tag{82}
$$

which expands to the loss in the proposition.

## Remark 1. Both losses reduce to the canonical MSE diffusion loss [54] when $t_{\mathrm{min}} = 0$ .

Remark 2. For both the $\epsilon$ - and $x_0$ -prediction losses, the rescaling factor, $1 / \alpha_t^2$ (resp. $1 / \tilde{\alpha}_t^2$ ), diverges as $t \to t_{\min}$ . This can cause training instability. Following [19], we introduce a buffer $B$ : a sample from $\mathcal{D}_q$ is only used at diffusion times satisfying $t > t_{\min}$ and $1 / \alpha_t^2 < B$ (resp. $1 / \tilde{\alpha}_t^2 < B$ ). The effect of $B$ is studied in Figure 19e.

![](images/79b0d4b0dd1198dbbdfe6b5db99078b804445eacb2216c7188307001aae6fb4b.jpg)  
(a) $\mathcal{D}_p$ from Section 7.1

![](images/1b8d4631d66df0ff950c434e2b5025c3e8b6bd201efdd7319092d69378158a06.jpg)  
(b) $\mathcal{D}_p$ from Appendix G.2

![](images/d8ce4b13b5635903efb2399b9b29f0828bcc46d280c1954e71a81228cac81aad.jpg)  
(c) $D_{p}$ from Section 7.3

![](images/72fa6b10197f567bb517b1209c9ad6b4f263d7a932a247d90e2ebe9989c2d8f8.jpg)  
(d) $\mathcal{D}_q$ from Section 7.3

![](images/a2a83c2b4dd83b06ba76f23a742466c7c16595d9dfb3bc18d2c504688fa2e885.jpg)  
(e) $D_{p}$ from Section 7.4

![](images/fa3e834866a847d28dadb0746da4a1282591ed4885c9aa517e568cd499a0a9b8.jpg)  
(f) $\mathcal{D}_p$ from Section 8  
Figure 11: A visualization of the average PSD for a diverse range of robot datasets; a spectral power law exists in all of them. Table 4 characterizes the datasets shown in (a)-(f) as well as the OXE dataset shown in Figure 3a.

## D Characterization of Robot Action Data and Action Diffusion

Spectral Power Law: Figures 3a and 11 visualize the spectral power law for several robot datasets. These datasets include human teleop vs algorithmically generated actions, absolute vs delta actions, end effector vs joint space actions, different parameterizations of SE(3) end-effector poses, and data from different tasks. Remarkably, a spectral power law existed in all the datasets we tested. Table 4 outlines the characteristics of each dataset in more detail.

The average PSD for each dataset was computed as follows. Given a dataset, we normalize the range of each action dimension to [-1,1]. For OXE, we instead normalized the 2% and 98% quartiles to [-1,1]. We do not normalize the rotation dimensions if they are represented in $R^{9}$ format since these values already lie within [0,1] by construction. All the datasets were collected at 10 Hz; however, action frequencies in OXE vary. To ensure consistency, we set the frequency of OXE to 10 Hz by downsampling higher-frequency data and artificially speeding up lower-frequency data. We chose an action horizon of 100. For each action dimension, we compute the periodogram along the time axis. Finally, we average these periodograms across all action chunks in the dataset. Our plots exclude the DC component, which is often close to 0 due to the range normalization. These plots use a log-log scale; a power law, $f^{-\alpha}$ , manifests as a straight line of slope $-\alpha$ .

We fix the action frequency and horizon for consistency across datasets, but remark that the spectral power law emerges regardless of these choices. Concurrent work performed the analysis using the discrete cosine transform and obtained similar results $[78]$ .

Region Classification Experiment (from Section 4.1): We test how well the global plan in $A_{0}$ can be recovered from $A_{t}$ using a learned classifier. We train the classifier on 5000 trajectories generated by GCS trajectory optimization [65] in the 2D maze. GCS decomposes the planning problem into discrete decisions (which regions of space to move through) and continuous decisions (the precise trajectory to follow within each region). Thus, the set of regions a trajectory passes through acts as a proxy for the global planning decisions, as it captures the topological structure of the maze.

Table 4: Characteristics of the datasets analyzed via power spectral density (PSD) in Figures 3a and 11. Despite their differences, all datasets exhibit a spectral power law.

<table><tr><td>Fig.</td><td>Dataset</td><td>Source</td><td>Generation</td><td>Action Space</td><td>Action Type</td><td>Rotation Rep.</td><td> $\bar{\alpha}^{\dagger}$ </td></tr><tr><td>Fig. 3a</td><td>COXE [1]</td><td> $\mathcal{D}_{q}$  (Sec. 8)</td><td>Mixed</td><td>Task space</td><td>Delta</td><td>Axis-angle</td><td>-1.47</td></tr><tr><td>Fig. 11a</td><td>GCS [65]</td><td> $\mathcal{D}_{p}$  (Sec. 7.1)</td><td>Algorithmic</td><td>x-y</td><td>Absolute</td><td>N/A</td><td>-1.82</td></tr><tr><td>Fig. 11b</td><td>TrajOpt.</td><td> $\mathcal{D}_{p}$  (App. G.2)</td><td>Algorithmic</td><td>Joint space</td><td>Absolute</td><td>N/A</td><td>-1.82</td></tr><tr><td>Fig. 11c</td><td>Planar pushing [12]</td><td> $\mathcal{D}_{p}$  (Sec. 7.3)</td><td>Human teleop</td><td>x-y</td><td>Absolute</td><td>N/A</td><td>-2.01</td></tr><tr><td>Fig. 11d</td><td>Graesdal et al. [72]</td><td> $\mathcal{D}_{q}$  (Sec. 7.3)</td><td>Algorithmic</td><td>x-y</td><td>Absolute</td><td>N/A</td><td>-1.88</td></tr><tr><td>Fig. 11e</td><td>Block sorting</td><td> $\mathcal{D}_{p}$  (Sec. 7.4)</td><td>Human teleop</td><td>Task space</td><td>Absolute</td><td> $\mathbb{R}^{9}$ </td><td>-2.27</td></tr><tr><td>Fig. 11f</td><td>Table cleaning</td><td> $\mathcal{D}_{p}$  (Sec. 8)</td><td>Human teleop</td><td>Task space</td><td>Absolute</td><td> $\mathbb{R}^{9}$ </td><td>-2.17</td></tr></table>

$\dagger \bar{\alpha}$ : mean fitted exponent across action dimensions, where $S(f) \propto f^{\alpha}$

![](images/ea4e104a8f59f62e9f2269cc9307a43f4efbc01f59c8d3dd81857e64bd8ec9a1.jpg)

![](images/48a79c160218223d8a69f53f9d0a5df1ff25bed0af3e6c43b1898d69a8cf4efa.jpg)

![](images/999f88003094cb8e37e4973b2c345e85ab9d0c9e467131d0a8fe79014ba4189a.jpg)  
(a) Convex decomposition (b) Visualization of the discrete (c) Prediction accuracy for the plan at different of the collision-free space. decisions in GCS. noise levels.  
Figure 12: The backward process commits to a global plan at high noise. By $\sigma_{t}=0.878$ , the global plan in $A_{0}$ can already be recovered from $A_{t}$ with 95% accuracy. Accuracy plateaus afterward. This indicates that the start of the backward process performs global planning. Figure 2 shows that the remainder of the backward process is devoted to local trajectory refinement.

Concretely, we train a classifier $c_{\theta}: A \times R \to R^{20}$ that takes as input $(A_{t}, t)$ and predicts the probability that $A_{0}$ passes through each of the 20 convex regions $^{5}$ . Figure 12b shows an example trajectory and its corresponding region labels. Figure 12c plots the average prediction accuracy at each noise level on a held-out test dataset: it saturates quickly as the noise decreases. This implies that nearly all the global planning information is diffused early in the backward process. Altogether, this experiment supports our claim that action diffusion exhibits a global-to-local hierarchy.

Related Work: Decomposing action generation hierarchically is a classic approach for solving problems in robotics $[79]$ . Recent works have begun incorporating this idea into Diffusion Policy. Generally, these approaches factorize action generation into a high and low level policy $[2, 3]$ ; however, this factorization is often handcrafted and task-specific $[80, 81, 82]$ . Our results show that a global-to-local action hierarchy emerges naturally from the spectral structure of the data. Thus, we hypothesize that with sufficient data, an end-to-end policy could potentially learn the “correct” hierarchy in the diffusion process without manual design.

Locality (Sensitivity Field from Figure 3b): Let $h_{\theta}^{(8)}$ denote an $x_0$ -predictor's estimate of the 8th clean action and let $A_t^{(i)}$ denote the $i$ -th action in the noisy action chunk $A_t$ . For each diffusion time $t$ , the sensitivity field is defined as:

$$
S _ {t} (i) := \mathbb {E} _ {(O, A _ {0}) \sim \mathcal {D}} \left[ \left\| \frac {\partial h _ {\theta} ^ {(8)} (A _ {t} , O , t)}{\partial A _ {t} ^ {(i)}} \right\| _ {F} \right], \quad i \in [ 0, 1 5 ].\tag{83}
$$

We normalize each sensitivity field so that $\max_{i} S_{t}(i) = 1$ at each diffusion time. Intuitively, $S_{t}(i)$ measures the sensitivity of the 8th action estimate to the i-th noisy action at diffusion time t.

## E Pseudocode and Possible Extensions

## E.1 Ambient Diffusion Policy Pseudocode

Given $D_{p}, D_{q}$ , and annotations from Phase 1 in the Method section, we train an Ambient Diffusion Policy following Algorithm 1. At each step, we sample a batch of diffusion times $t^{(i)} \stackrel{\text{i.i.d.}}{\sim} \mathcal{U}[0, T]$ and then draw admissible samples from $D_{p} \cup D_{q}$ . We update the policy parameters, $\theta$ , to minimize the loss in COMPUTELOSS.

COMPUTELOSS executes the forward process and can use two choices for L. The first is the standard denoising loss, $L_{denoising}$ [54]. The second is the Ambient loss, $L_{ambient}$ , discussed in Appendix C. $^{6}$ We present two versions of COMPUTELOSS for variance-exploding (Algorithm 2) and variance-preserving diffusion processes (Algorithm 3).

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1 Ambient Diffusion Policy: Training
Require:  $D_{p}$ ,  $D_{q}$  with  $(t_{\min}^{(i)}, t_{\max}^{(i)})$ ,  $L \in \{L_{ambient}, L_{denoising}\}$ , noise schedule  $\sigma(t)$ , max time T, batch size B, policy parameters  $\theta$ 
1: repeat
2: Sample  $t^{(i)} \stackrel{\text{i.i.d.}}{\sim} \mathcal{U}[0, T]$  for  $i = 1, \ldots, B$ 
3: Sample batch  $\{(O^{(i)}, A_{0}^{(i)})\}_{i=1}^{B}$  s.t.  $t^{(i)} \in [0, t_{\max}^{(i)}) \cup (t_{\min}^{(i)}, T]$ 
4:  $\ell \leftarrow \frac{1}{B} \sum_{i=1}^{B} COMPUTELOSS\left(\mathcal{L}, A_{0}^{(i)}, O^{(i)}, t^{(i)}, t_{\max}^{(i)}, t_{\min}^{(i)}\right)$ 
5: Update  $\theta$  on  $\ell$ 
6: until converged
7: return  $\pi_{\theta}$
</div>

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 2 COMPUTELOSS (variance-exploding)

Require: L,  $A_{0}$ , O, t,  $t_{max}$ ,  $t_{min}$ 

1: if  $t \in [0, t_{max})$  or  $L = L_{denoising}$  then

2:  $A_{t} \leftarrow A_{0} + \sigma_{t} Z, \quad Z \sim \mathcal{N}(0, I)$ 

3: return  $\mathcal{L}_{denoising}(A_{t}, A_{0}, O, t)$  ▷ standard DDPM loss [54]

4: else ▷  $t \in (t_{\min}, T]$  and  $L = L_{ambient}^{VE}$ 

5: Sample  $Z_{1}, Z_{2} \stackrel{\text{i.i.d.}}{\sim} \mathcal{N}(0, I)$ 

6:  $A_{t_{\min}} \leftarrow A_{0} + \sigma_{t_{\min}} Z_{1}$ 

7:  $A_{t} \leftarrow A_{t_{\min}} + \sqrt{\sigma_{t}^{2} - \sigma_{t_{\min}}^{2}} Z_{2}$ 

8: return  $\mathcal{L}_{ambient}^{VE}(A_{t}, A_{t_{\min}}, O, t)$  ▷ VE Ambient loss [24]

9: end if
</div>

## E.2 Diffusion Policy Implementation Details

Unless otherwise noted, policies are trained to predict the next 16 actions at $10\mathrm{Hz}$ with a fixed horizon for observations and actions. We execute the first 8 actions and predict new actions in an MPC-like fashion. Policies are conditioned on the 2 most recent observations. We consider a fixed horizon for observations and actions and assume that the learned policy is time-invariant, i.e. the conditional action distribution $\pi (\cdot |O)$ depends only on the most recent observations and not on the time $t$ . The per-experiment observation and action spaces are summarized in Tables 5 and 6.

We use the U-Net architecture from $[23]$ , but add the option for conditioning on a goal. When a policy is image-conditioned, we use the ResNet18 backbone for the vision encoder. We train the policy end-to-end with a cosine learning rate schedule. The noise schedule $\sigma(t)$ is a cosine

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 3 COMPUTELOSS (variance-preserving)

Require: $\mathcal{L}, A_0, O, t, t_{\max}, t_{\min}$

1: if $t \in [0, t_{\max})$ or $\mathcal{L} = \mathcal{L}_{\text{denoising}}$ then

2: $A_t \leftarrow \sqrt{1 - \sigma_t^2} A_0 + \sigma_t Z, \quad Z \sim \mathcal{N}(0, I)$

3: return $\mathcal{L}_{\text{denoising}}(A_t, A_0, O, t)$ ▷ standard DDPM loss [54]

4: else ▷ $t \in (t_{\min}, T]$ and $\mathcal{L} = \mathcal{L}_{\text{ambient}}^{\text{VP}}$

5: Sample $Z_1, Z_2 \stackrel{\text{i.i.d.}}{\sim} \mathcal{N}(0, I)$

6: $A_{t_{\min}} \leftarrow \sqrt{1 - \sigma_{t_{\min}}^2} A_0 + \sigma_{t_{\min}} Z_1$

7: $A_t \leftarrow \sqrt{\frac{1 - \sigma_t^2}{1 - \sigma_{t_{\min}}^2}} A_{t_{\min}} + \sqrt{\frac{\sigma_t^2 - \sigma_{t_{\min}}^2}{1 - \sigma_{t_{\min}}^2}} Z_2$

8: return $\mathcal{L}_{\text{ambient}}^{\text{VP}}(A_t, A_{t_{\min}}, O, t)$ ▷ Eq. (70) ($\epsilon$-pred) or Eq. (78) ($x_0$-pred)

9: end if
</div>

![](images/c68e74d807ccab0bd5668becdfc7ab7f72a000eda3a7cc81d1adb28f160e1593.jpg)  
Figure 13: Average classifier output for suboptimal samples in the 2D maze dataset (Section 7.1) at different diffusion times. As t increases, $p_{t}$ and $q_{t}$ become harder to distinguish. The dotted line is the threshold $0.5 - \tau$ , which is crossed for $t_{min} = 18$ .  
schedule with T = 100 diffusion steps. We apply EMA to the training loop. Sampling is performed online using 10 steps of DDIM [57]. Predicted actions are linearly interpolated into piecewise linear trajectories and passed through a first-order low-pass filter before being sent to the robot.

## E.3 Future Direction: Classifier-Based Rejection Sampling

This section proposes a similar algorithm to Ambient Diffusion Policy based on rejection sampling. We provide theoretical foundations for this method, but do not experimentally verify it. This is an exciting direction for future work.

Recall that we trained a classifier that outputs the probability that a noisy action chunk came from $p_{t}$ (as opposed to $q_{t}$ ). Algorithm 4 samples (approximately) from P given only access to samples from the mixture $M = \frac{1}{2}P + \frac{1}{2}Q$ and the classifier. Proposition 6 bounds the total variation between the sampled distribution and P. By using the classifier for rejection sampling, we can train denoisers for P despite leveraging data from an auxiliary source Q. In the proofs, we use $f_{\theta}$ to denote the classifier instead of $c_{\phi}$ from Section 6.

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 4 Rejection sampling via classifier
Require: Noisy classifier  $f_{\theta}(x,\sigma)$ , mixture distribution  $M = \frac{1}{2}P + \frac{1}{2}Q$ , noise distribution  $\rho_{\sigma}$ 
1: Sample  $\sigma \sim \rho_{\sigma}$ 
2: while True do
3:    Sample  $X \sim M$ 
4:    $X_{\sigma} \leftarrow \text{ADDNOISE}(X, \sigma)$ 
5:    Sample  $c \sim \text{Bern}(f_{\theta}(X_{\sigma}, \sigma))$ 
6:    if c = 1 then
7:    return ( $\sigma, X_{\sigma}$ )
8:    end if
9: end while
</div>

Proposition 6 (Rejection sampling guarantee). Suppose $f_{\theta}$ is uniformly $\varepsilon$ -optimal across $\sigma$ :

$$
\mathcal {L} _ {\mathrm{CE}} (f \mid \sigma) \leq \mathcal {L} _ {\mathrm{CE}} (f ^ {*} \mid \sigma) + \varepsilon , \quad \forall \sigma .
$$

For any $\sigma \geq 0$ , let $P_{\sigma}$ be the noisy distribution of $X \sim P$ , and let $\hat{P}_{\sigma}$ be the distribution of $X_{\sigma}$ output by Algorithm 4. Then

$$
\mathrm{d} _ {\mathrm{TV}} (\hat {P} _ {\sigma}, P _ {\sigma}) \leq O (\sqrt {\varepsilon}).
$$

Proof. We drop the subscript $\sigma$ and write $P \equiv P_{\sigma}$ , $Q \equiv Q_{\sigma}$ . Let $m(x)$ denote the mixture density. The true distribution P satisfies $p(x) = 2m(x)f^{*}(x)$ , while the rejection-sampled density is

$$
\hat {p} (x) = \frac {m (x) f _ {\theta} (x)}{Z}, Z = \mathbb {E} _ {X \sim M} [ f _ {\theta} (X) ],
$$

where $Z$ is the acceptance rate. By Pinsker applied to the cross-entropy gap, $\mathbb{E}_{X\sim M}[(f^{*}(X) - f_{\theta}(X))^{2}] \leq \varepsilon / 2$ , so by Jensen

$$
\delta \triangleq \mathbb {E} _ {X \sim M} | f _ {\theta} (X) - f ^ {*} (X) | \leq \sqrt {\varepsilon / 2}.
$$

This gives $|Z - \frac{1}{2}| \leq \delta$ , hence $Z \geq \frac{1}{2} - \delta$ . The total variation between $\hat{P}$ and $P$ is

$$
\begin{array}{c} \mathrm{d} _ {\mathrm{TV}} (\hat {P}, P) = \frac {1}{2} \int \left| \frac {m (x) f _ {\theta} (x)}{Z} - 2 m (x) f ^ {*} (x) \right| d x \\ = \mathbb {E} _ {X \sim M} \left| \frac {f _ {\theta} (X) - 2 Z f ^ {*} (X)}{2 Z} \right|. \end{array}
$$

By the triangle inequality,

$$
\left| \frac {f _ {\theta} (X) - 2 Z f ^ {*} (X)}{2 Z} \right| \leq \frac {| f _ {\theta} (X) - f ^ {*} (X) |}{2 Z} + \frac {f ^ {*} (X)}{2 Z} | 1 - 2 Z |.
$$

Taking the expectation over $M$ and using $\mathbb{E}_{X\sim M}[f^{*}(X)] = \frac{1}{2}$ and $|1 - 2Z|\leq 2\delta$ :

$$
\mathrm{d} _ {\mathrm{TV}} (\hat {P}, P) \leq \frac {\delta}{2 Z} + \frac {2 \delta}{4 Z} = \frac {\delta}{Z} \leq \frac {2 \delta}{1 - 2 \delta}.
$$

For $\varepsilon < 2 / 9$ we have $1 - 2\delta > 1 / 3$ , hence

$$
\mathrm{d} _ {\mathrm{TV}} (\hat {P}, P) \leq 6 \sqrt {\varepsilon / 2} = O (\sqrt {\varepsilon}).
$$

## F Experiments Overview

Tables 5 and 6 summarize the datasets, distribution shifts, observation and action spaces, hyperparameters, loss functions, and evaluation details for all main experiments in Sections 7 and 8 respectively; ablation parameters may differ. Importantly, we swept hyperparameters for each baseline to ensure a fair comparison.

Binary success metrics are reported with 95% Clopper-Pearson confidence intervals. Otherwise, error bars are standard error intervals. For each training run, we save 5-10 checkpoints and report the performance of the best-performing one. While the initial exposition of our algorithm was presented with variance-exploding diffusion, our implementation uses variance-preserving diffusion (i.e. $\sigma(T) = 1$ ).

Table 5: Experimental configurations for the four generality experiments. Parameters vary for the ablations.

<table><tr><td colspan="2"></td><td>Maze</td><td>Neural Motion Planning</td><td>Sim-and-Real Co-training</td><td>Block Sorting</td></tr><tr><td rowspan="5">Dataset Details</td><td> $\mathcal{D}_{p}$ </td><td>GCS [65]</td><td>Trajectory Optimization [83]</td><td>Joystick Teleop [12]</td><td>VR Teleop</td></tr><tr><td> $\mathcal{D}_{q}$ </td><td>RRT [66]</td><td>BiRRT [84]</td><td>Contact-Rich Planner [72]</td><td>VR Teleop</td></tr><tr><td> $|\mathcal{D}_{p}|$  (Num. Demos.)</td><td>50</td><td>100,000</td><td>50</td><td>50</td></tr><tr><td> $|\mathcal{D}_{q}|$  (Num. Demos.)</td><td>5000</td><td>1,000,000</td><td>2000</td><td>200</td></tr><tr><td>Nature of shift</td><td>Low-quality/ noisy trajectories</td><td>Low-quality/ noisy trajectories</td><td>Sim-to-real gap</td><td>Task mismatch</td></tr><tr><td rowspan="2">Policy I/O</td><td>Observation space</td><td>Proprio</td><td>Proprio</td><td>EE position Scene camera Wrist camera</td><td rowspan="2">EE pose (xyz, R9) Gripper state Scene camera Blocks camera Wrist camera EE command (xyz, R9, gripper)</td></tr><tr><td>Action space</td><td>x-y position</td><td>Joint angles  $\mathbb{R}^{7}$ </td><td>x-y pusher position</td></tr><tr><td rowspan="3">Parameters</td><td> $\sigma_{t_{\min}^{*}} \in [0, 1]$ </td><td>0.074 (sweep)</td><td>0.025 (sweep)</td><td>varies see Table 2</td><td>N/A (sweep)</td></tr><tr><td> $\sigma_{t_{\max}^{*}} \in [0, 1]$ </td><td>N/A</td><td>N/A</td><td>0 (sweep)</td><td>0.46 (sweep)</td></tr><tr><td> $\alpha^{*} \in [0, 1]$  for co-training</td><td>0.019 (sweep)</td><td>0.091 (sweep)</td><td>0.25 (sweep)</td><td>0.9 (sweep)</td></tr><tr><td rowspan="2">Loss</td><td>Loss prediction target</td><td> $x_{0}$ </td><td> $\epsilon$ </td><td> $x_{0}$ </td><td> $\epsilon$ </td></tr><tr><td>Loss function</td><td> $\mathcal{L}_{ambient}$ </td><td> $\mathcal{L}_{denoising}$ </td><td> $\mathcal{L}_{denoising}$ </td><td> $\mathcal{L}_{denoising}$ </td></tr><tr><td>Eval.</td><td>Num. trials</td><td>1,000</td><td>1,000</td><td>200</td><td>200 (800 blocks)</td></tr></table>

## G Controlled Experiments

## G.1 Noisy Trajectories: 2D Maze

Data Generation: The GCS $[65]$ trajectories were generated using third-order Bezier curves, first-order continuity constraints, and a maximum of 10 rounded paths. Both the GCS and the RRT datasets were generated with a velocity constraint of 1m/s in the x and y directions and 0.1m of obstacle padding. The maze environment is 5m by 5m.

Evaluation: During each trial, we sample the start and goal uniformly in the padded collision-free configuration space; the robot is initialized at the start configuration. A trial is successful if the robot can navigate to a 0.15m ball around the end configuration without collision within 30s of simulation time. We performed 1000 trials per policy.

We compute the smoothness metric as the average squared acceleration of the policy's rollout. Concretely, we fit a cubic spline to the rollout and evaluate

$$
\text { Smoothness } = \frac {1}{T} \int_ {0} ^ {T} \| a (t) \| _ {2} ^ {2} d t,\tag{84}
$$

where $a(t)$ is the acceleration of the spline at time t. We only report smoothness for the successful rollouts, for three reasons: 1) failed rollouts have orders of magnitude higher accelerations that disproportionally bias the results, 2) across 1000 trials, most policies had less than 10 failed rollouts, 3) the overall trend when averaging across all trials is the same.

Table 6: Experimental configurations for the two OXE scaling experiments. Parameters vary for the ablations.

<table><tr><td colspan="2"></td><td>Table Cleaning</td><td>Tower Building</td></tr><tr><td rowspan="5">Dataset Details</td><td> $\mathcal{D}_{p}$ </td><td>VR Teleop</td><td>VR Teleop</td></tr><tr><td> $\mathcal{D}_{q}$ </td><td>OXE[1]</td><td>OXE[1]</td></tr><tr><td> $|\mathcal{D}_{p}|$  (Num. Demos.)</td><td>50 &amp; 150</td><td>35</td></tr><tr><td> $|\mathcal{D}_{q}|$  (Num. Demos.)</td><td>1,279,774 (MS++)1,402,606 (COXE)</td><td>1,402,606 (COXE)</td></tr><tr><td>Nature of shift</td><td>Unstructured</td><td>Unstructured</td></tr><tr><td>Policy I/O</td><td>Observation spaceAction space</td><td>EE pose (xyz, R9)Gripper stateTable cameraDrawer camera2x Wrist cameraDelta pose(xyz, axis-angle,gripper)</td><td>EE pose (xyz, R9)Gripper stateTable cameraTower camera2x Wrist cameraDelta pose(xyz, axis-angle,gripper)</td></tr><tr><td rowspan="3">Parameters</td><td> $\sigma_{t_{\min}^{*}} \in [0, 1]$ </td><td>See Figure 21(classifier)</td><td>See Figure 22(classifier)</td></tr><tr><td> $\sigma_{t_{\max}^{*}} \in [0, 1]$ </td><td>0.167(no sweep)</td><td>0.09(no sweep)</td></tr><tr><td> $\alpha^{*} \in [0, 1]$  for co-training</td><td>0.5(no sweep)</td><td>0.5(no sweep)</td></tr><tr><td>Loss</td><td>Loss prediction targetLoss function</td><td> $\epsilon$  $\mathcal{L}_{\text{denoising}}$ </td><td> $\epsilon$  $\mathcal{L}_{\text{denoising}}$ </td></tr><tr><td>Eval.</td><td>Num. trials</td><td>20(60 objects)</td><td>20</td></tr></table>

Figure 15 shows the maze success rate as a function of $\sigma_{t_{min}}$ , and Figure 14 visualizes representative policy rollouts for each training algorithm.

## G.2 Noisy Trajectories: Neural Motion Planning

This appendix provides the full experimental details for the neural motion planning experiment in Section 7.2, along with best-of-100 results and a $\sigma_{t_{min}}$ sweep. The main results (Table 1b and Figure 6b) are presented in Section 7.2. Table 7 compares the 2D maze and 7-DoF NMP setups side by side.

The environment in Figure 1b was generated using a procedural method [85] and the assets are from Pfaff et al. [86]. Start and goal configurations are determined by sampling a random end-effector target point $p \in \mathrm{SE}(3)$ and then solving the optimization inverse kinematics problem

$$
\begin{array}{l l} \text {find} & q \in \mathbb {R} ^ {7} \\ \text {s.t.} & \operatorname{FK} _ {\text {pos}} (q) = p, \\ & \operatorname{SDF} (q) \geq 0. 0 1, \end{array}\tag{85}
$$

where $FK_{pos}$ is the position forward kinematics, returning the position of the gripper as a function of the joint angles, and SDF returns the clearance (in meters) between the robot and any obstacles. The constraint $SDF(q) \geq 0.01$ is implemented via Drake's MinimumDistanceLowerBoundConstraint. Targets are sampled 1) within shelves, 2) in bins, 3) near objects, and 4) uniformly in the free space. We sample from all 4 classes of target configurations with equal probability. (85) is solved using SNOPT [87].

![](images/a6f3888d53626d974830a3b890627de23bc7a5b441b07af62de6e8529e8cd825.jpg)  
Figure 14: Policy rollouts from the same start and goal pairs for different training algorithms. a) The policy trained with data filtering (i.e. GCS-only) has not seen enough data to learn the maze. Its rollouts frequently collide with obstacles. b) The policy trained with RRT-only has memorized the maze, but exhibits non-smoothness. c) The co-trained policy's behavior is similar to the RRT-only policy. d) The Ambient Diffusion Policy is smooth and understands the structure of the maze.

![](images/bcdf8ff121f0e0fbe9e9fdf5fc6e87d1f2826ebc79224ff422b81a55725c6e87.jpg)  
Figure 15: Maze success rate vs $\sigma_{t_{\mathrm{min}}}$ . For small $\sigma_{t_{\mathrm{min}}}$ , the Ambient Policies are smooth and achieve high success rate (see Figure 6a). As $\sigma_{t_{\mathrm{min}}}$ increases, the RRT data is used less during training, which reduces success rate.

Given a start and goal configuration $q_{0}, q_{1} \in R^{7}$ , we use BiRRT [88] to produce the suboptimal data. To produce the high-quality data, we first run randomized shortcutting [89] to slightly simplify the path. We then solve the trajectory optimization problem

$$
\begin{array}{r l} \min _ {q (s)} & \int_ {0} ^ {1} | | \dot {q} (s) | | ^ {2} \mathrm{d} s \\ \text {s.t.} & q (0) = q _ {0}, q (1) = q _ {1}, \\ & \operatorname{SDF} (q (s)) \geq 0, \forall s \in [ 0, 1 ] \end{array}\tag{86}
$$

with Drake's KinematicTrajectoryOptimization, using its standard B-spline discretization [90]. We use the shortcut trajectory as an initial guess and solve with IPOPT [91], with solver options tuned aggressively to maintain feasibility.

Table 7: Comparison between the 2D maze and 7-DoF neural motion planning experiments.

<table><tr><td></td><td>2D Maze</td><td>7-DoF NMP</td></tr><tr><td> $\mathcal{D}_{p}$  Source</td><td>GCS [65]</td><td>Trajectory Optimizer [83]</td></tr><tr><td> $\mathcal{D}_{q}$  Source</td><td>RRT [66]</td><td>BiRRT [84]</td></tr><tr><td> $|\mathcal{D}_{p}|$ </td><td>50</td><td>100,000</td></tr><tr><td> $|\mathcal{D}_{q}|$ </td><td>5,000</td><td>1,000,000</td></tr><tr><td>Output Type</td><td>Action Chunk</td><td>Full Plan</td></tr><tr><td>Rollouts</td><td>Closed-Loop</td><td>Open-Loop</td></tr><tr><td>Penetration Threshold</td><td>0cm</td><td>2cm</td></tr><tr><td>Output Space</td><td>2D (x,y)</td><td>7D (Config. Space)</td></tr><tr><td>Start &amp; End Goals</td><td>Uniformly Random</td><td>Around points of interest(See Appendix G.2)</td></tr></table>

<table><tr><td>Training Algo.</td><td>Success Rate ↑</td><td>Smoothness ↓ (Eq. (84))</td></tr><tr><td rowspan="2">Data Filtering Co-train $(\alpha^{*} = 0.091)$ </td><td>73.9+2.7%</td><td>3.7</td></tr><tr><td>91.3+1.7%</td><td>32.6</td></tr><tr><td>Ambient $(\sigma_{t_{\min}^{*}} = 0.025)$ </td><td>91.9+1.6%</td><td>20.2</td></tr></table>

Table 8: Neural Motion Planning Best-of-100 Results (1000 trials). Ambient retains the highest success rate while achieving lower squared acceleration than the co-trained baseline.

Success Rate vs Smoothness (Best-of-100)  
![](images/155e8f37b905da7d3e84a830c33db16063f090e4e2e33bc57d28830b2b258b5e.jpg)  
Figure 16: Figure 6b repeated with best-of-100 evaluation. The overall trends are the same, but the success rates are higher and the gap between the two Pareto frontiers is smaller.

The collision-free constraint $\mathrm{SDF}(q(s)) \geq 0$ , $\forall s \in [0,1]$ is applied at 200 points evenly spaced along the trajectory. Collision checking with finer granularity is performed after trajectory optimization to filter trajectories with collisions. In principle, the discarded trajectories could be added to $D_{q}$ and used at sufficiently high diffusion times during training.

Each trajectory in $D_{p}$ and $D_{q}$ is subsampled to 100 waypoints, equally spaced in time. This subsampling procedure acts as a low-pass filter which reduces some of the non-smoothness in the BiRRT data.

Evaluation: Instead of rolling out the model closed-loop (as in the maze experiments), we train the planner to predict the entire plan at once. Start and end configurations are sampled from the 4 classes described above. Each trial is successful if the planner predicts a collision-free trajectory with up to 2cm of penetration. We allow 2cm of padding because the training trajectories were generated with 0cm of padding and frequently pass very close to obstacles. Both the first and last waypoints of the predicted plan must be within a 0.2rad ball of the start and end configurations respectively.

Previous works in neural motion planning adopt a best-of-N approach when evaluating their planners $[71]$ . We present the results using N = 1 (Table 1b and Figure 6b) and N = 100 as in $[68]$ (Table 8 and Figure 16). Concretely, for each trial, we generate 100 plans and pick the one with the least collisions. The general trends for both choices of N are the same, but the success rates are significantly higher. The Ambient policy continues to be smoother than the co-trained policy in both cases.

![](images/c051f5c756c3bee483072db14c867d74512d850240038ac88bd2365c3dd30381.jpg)  
(a) Success rate vs $\sigma_{t_{\mathrm{min}}}$ .

![](images/b8b175e4e7823e840a965a349e57ebc0eb22dc15edf6a24d7a31fd93367a0fdb.jpg)  
(b) Smoothness metric vs $\sigma_{t_{\mathrm{min}}}$ .  
Figure 17: Neural motion planning $\sigma_{t_{min}}$ sweep. (a) For small $\sigma_{t_{min}}$ , the policies are smoother, which results in fewer accidental collisions and increases success rate. For large $\sigma_{t_{min}}$ , the RRT data becomes under-utilized, which decreases the success rate. (b) Increasing $\sigma_{t_{min}}$ improves smoothness.

## G.3 Sim-and-Real Co-training: Planar Pushing

Data Generation: We use the datasets from Wei et al. [12]. $D_{p}$ contains human teleoperated demonstrations in their “target” simulation environment and $D_{q}$ contains automatically generated demonstrations [72] from their “source” simulation environment.

Table 9 shows the distribution shift between the source and target simulation datasets. For more details and visualizations of the environments, see Wei et al. [12].

Table 9: Sim-to-target gap along three axes. The sim-to-target gap in the environments from [12] was designed to mimic the sim-to-real gap. See [12] for visualizations.

<table><tr><td>Gap</td><td>Simulation</td><td>Target Sim</td></tr><tr><td>Visual</td><td>White spectrum lightDifferent object colorsDifferent camera poseNo shadows</td><td>Warm spectrum lightDifferent object colorsDifferent camera poseShadows</td></tr><tr><td>Physics</td><td>Quasistatic</td><td>Drake physics [90]</td></tr><tr><td>Action</td><td>Motion planning</td><td>Human teleop</td></tr></table>

Evaluation: The slider is randomly initialized within the robot's workspace. A trial is successful if the robot returns to its default position and the slider's pose is within 1.5cm and 3.5° of the goal. The time limit is 75s. The evaluation criteria exactly match Wei et al. [12] to ensure a fair comparison.

Ablations: We use these experiments as a test-bed for evaluating key design decisions in Ambient Diffusion Policy. Figure 19 consolidates eight ablations; the results suggest the following:

\- Per-datapoint $t_{\mathrm{min}}$ (Figure 19a): Labeling $\sigma_{t_{\mathrm{min}}}$ per dataset improves over co-training, and labeling $\sigma_{t_{\mathrm{min}}}$ per datapoint performs the best. The distribution of $t_{\mathrm{min}}$ assigned by the best classifier is shown in Figure 18.

\- $\sigma_{t_{\mathrm{max}}}$ sweep (Figure 19b): Using locality ( $\sigma_{t_{\mathrm{max}}} > 0$ ) does not appear to improve or hurt performance in sim-and-real co-training until $\sigma_{t_{\mathrm{max}}}$ is larger. Sim and real data do not share the same local action structure (e.g. due to differing contact dynamics); thus, policies should not train on sim data at low diffusion times.

\- Classifier robustness (Figures 19c and 19d): Policy performance is insensitive to the classifier checkpoint and threshold $\tau$ ; the classifier merely needs to be sufficiently trained.

\- Ambient loss buffer (Figure 19e): The Ambient loss is highly sensitive to the buffer size from Appendix C. In most experiments (Table 5), we avoid this sensitivity by training with the regular denoising objective [54].

\- $t_{\text{min}}$ granularity (Figure 19f): Assigning $t_{\text{min}}$ at intermediate granularities (one $t_{\text{min}}$ per partition of $\mathcal{D}_q$ ) appears to be a promising direction. We split $\mathcal{D}_q$ into $N$ partitions and annotate one $\sigma_{t_{\text{min}}}$ per partition. $N = 1$ corresponds to assigning a single $t_{\text{min}}$ to all of $\mathcal{D}_q$ , and $N \to \infty$ corresponds to annotating a $t_{\text{min}}$ per sample. Given a partition $P_i$ , we annotate

$$
t _ {\min} = \inf \left\{t: \mathbb {E} _ {(O, A _ {0}) \sim P _ {i}, A _ {t} | A _ {0}} \left[ c _ {\phi^ {*}} (A _ {t}, t) \right] > 0. 5 - \tau \right\}.
$$

We compare three partitioning strategies: (1) randomly partition the action chunks (blue); (2) randomly partition the trajectories – each partition contains the action chunks of its trajectories (red); (3) find $t_{min}$ for each action chunk using the classifier and sort the chunks according to these values. Then form the partitions in increasing order (orange). For this task, Strategy (1) with N = 10,000 performed best.

\- $|\mathcal{D}_q|$ scaling (Figures 19g and 19h): As $\mathcal{D}_q$ scales, Ambient consistently outperforms co-training via re-weighting at both $|\mathcal{D}_p| = 10$ and $|\mathcal{D}_p| = 50$ (except at $|\mathcal{D}_q| = 4000$ in the latter). The co-training baseline results (green) are from [12].

![](images/e987b8a4fc8b4c6cc759f31090728edc277c02308a533783ea165341450b63d0.jpg)  
Figure 18: Distribution of $t_{min}$ labels. This histogram shows the distribution of $t_{min} \in [0, 100]$ assigned to each action chunk by the classifier for the sim-and-real experiments.

## G.4 Task Mismatch: Block Sorting

Data Collection: Both datasets were collected using VR teleoperation.

Evaluation: On each trial, 1-3 red blocks and 1-3 blue blocks are randomly initialized in a pile between the bins. The time limit is 20s per block. This allows trials with more blocks to have a longer time limit. Each policy is evaluated on 200 trials ( $\sim$ 800 blocks).

Figure 7 shows the performance of the Ambient Diffusion Policy on each metric during the $\sigma_{t_{\mathrm{max}}}$ sweep. Figures 20a and 20b show the average number of grasp attempts and time required per block of the policy's trained with different $\sigma_{t_{\mathrm{max}}}$ values. Both can be viewed as additional motion metrics and improve (decrease) with higher $\sigma_{t_{\mathrm{max}}}$ .

Ablation 1 (Task-Conditioned): A standard solution to handling the incorrect data is to add a one-hot encoding that indicates the sorting direction. This is analogous to task or quality conditioning [10]. Even with task conditioning, the Ambient Diffusion Policy performs the best (Table 10a). Note that Ambient Diffusion Policy and task-conditioning are complementary.

Ablation 2 ( $D_{q}$ only at low noise): Locality implies that the denoisers at $t < t_{max}$ are insensitive to the global task structure. If so, we should be able to train an effective policy without using any “correct” data at low noise. We train a policy (without task-conditioning) that exclusively uses $D_{q}$ for $t \in [0, t_{\max})$ and $D_{p}$ for $t \in (t_{\max}, T]$ . This policy still achieves a logic accuracy of 97.9% and a motion score of 93.8% (Table 10b). This reinforces our claim that the backward process factorizes into a planning and a local refinement regime.

![](images/53edbe01e3fd703e65a6bfad8cc5a84473ac872fb46f844aa46dde870ffd0060.jpg)  
(a) Performance vs $\sigma_{t_{\min}}$ (per dataset).

![](images/dce27f632e3afa2158437fa953a990219598f1f5caa20b98833744b9446199f3.jpg)  
(b) Performance vs $\sigma_{t_{\mathrm{max}}}$ .

![](images/82b71c07699e5cfaa82e765949545af72b881eb6f2ab56bcf2cb74d6438315aa.jpg)  
(c) Sensitivity to classifier training duration.

![](images/c5a9e68b5f69331e2db7e06cc62dee7956316971b4aaf7401054a0aba8af3d3c.jpg)  
(d) Sensitivity to classifier threshold $\tau$ .

![](images/8fdeb216c5f964c8aa440fbb5bf6718a54b6413241130b3f11c3badbb1927a04.jpg)

![](images/6f1ca204fddb92d87d3e6c6a05e2076b5e234390b8a8429ec7fc15c62f65d606.jpg)  
(f) Performance vs granularity of $t_{min}$ labels.

(e) Sensitivity to the Ambient loss buffer (Appendix C).  
![](images/619a5f632fcd9b26c6d83da75e414377c6aecd392aae0be6100cd998e9056b57.jpg)  
(g) Performance vs $|D_{q}|$ scale ( $|D_{p}| = 10$ ).

![](images/7051e2e35f7fa91444072cef9765cc690c37c95ba1e7814ca0776212df2c71d5.jpg)  
(h) Performance vs $|\mathcal{D}_q|$ scale $(|\mathcal{D}_p| = 50)$ .  
Figure 19: Sim-and-real co-training ablations. Eight ablations covering the key design decisions of Ambient Diffusion Policy. See Appendix G.3 for the per-panel analysis.

![](images/3b9e9769e69e9b06f1d1cd51d96e604e0201fa46f3cab96a378675c30bf92308.jpg)  
(a) Grasp attempts per block vs $\sigma_{t_{max}}$ .

![](images/e1bd2043eb53ed7263f8b22eab60dee6eaf1bea0e7c0297de637bdfcb995a8ca.jpg)  
(b) Time per block vs $\sigma_{t_{\mathrm{max}}}$ .  
Figure 20: Additional motion-level metrics for the block sorting experiment (Section 7.4). (a) Grasp attempts per block improves as $\sigma_{t_{max}}$ increases but plateaus after $\sigma_{t_{max}^{*}} = 0.46$ . (b) Time per block improves as $\sigma_{t_{max}}$ increases.

Table 10: Block sorting ablations (200 trials, \~800 blocks). (a) Ablation 1 (Task-Conditioned): Ambient outperforms co-training. (b) Ablation 2: the Ambient Diffusion Policy performs well despite training on no logically correct samples for $t \in [0, t_{\max}^{*})$ .

<table><tr><td colspan="3">(a) Ablation 1: Task-Conditioned</td><td>(b) Ablation 2</td></tr><tr><td>Training Algo. (# Blocks: ~800)</td><td>Co-train $(\alpha^{*} = 0.5)$ </td><td>Locality $(\sigma_{t_{\max}^{*}} = 0.89)$ </td><td>Locality $\mathcal{D}_{q}$  only below  $\sigma_{t_{\max}^{*}} = 0.24$ </td></tr><tr><td>Logic ↑</td><td> $98.6^{+0.7}_{-1.1}\%$ </td><td> $98.5^{+0.7}_{-1.1}\%$ </td><td> $97.9^{+0.9}_{-1.3}\%$ </td></tr><tr><td>Motion ↑</td><td> $91.5^{+1.8}_{-2.2}\%$ </td><td> $94.2^{+1.5}_{-1.8}\%$ </td><td> $93.8^{+1.6}_{-1.9}\%$ </td></tr><tr><td>Success Rate ↑</td><td> $90.3^{+2.0}_{-2.3}\%$ </td><td> $92.8^{+1.7}_{-2.0}\%$ </td><td> $91.9^{+1.8}_{-2.1}\%$ </td></tr></table>

## H Scaling Experiments: Open X-Embodiment

## H.1 OXE Experiments

Dataset Curation: We collected demonstrations for $D_{p}$ with VR teleoperation. For each demonstration, 3-5 objects are placed on the table. These objects are selected among a set of 31 objects in the training distribution. The teleoperator opens the drawer, moves the objects into the drawer, and closes the drawer.

These experiments used two different subsets of the Open X-Embodiment (OXE) for $D_{q}$ : Magic Soup++ (MS++) and Custom OXE (COXE). MS++ is a collection of 27 datasets that was taken directly from the OpenVLA paper [5].

Custom OXE is a collection of 48 datasets. All 27 datasets from MS++ were included in COXE. Afterward, any dataset that met the following criteria was included:

\- Robot morphology: single arm or mobile manipulator

• Action space: EEF position or EEF velocity

\- Has a known control frequency

\- Observation space: has proprioception and contains at least one non-depth camera

\- Accessible: a few datasets that met the criteria above were not accessible, contained ambiguous features and action-space labels, or bugs. These datasets were excluded.

Dataset Weights: Weights for MS++ were taken from OpenVLA $[5]$ . For COXE, we reused weights for the MS++ subset. For the remaining datasets, we assigned weights that were proportional to their size except for the two largest datasets. For those, we halved their relative weight. Following the “50/50 co-training” in previous literature $[92][26]$ , we normalize the total weight of $D_{q}$ to be 0.5, and assign $D_{p}$ a weight of 0.5. This is equivalent to assigning $\alpha = 0.5$ in the co-training case. In the unweighted case, the weight of each dataset is proportional to its relative size. See Figure 21 for a visualization.

Table 11: Table cleaning evaluation rubric. The maximum score per trial is 1.0.

<table><tr><td>Subtask</td><td>Points</td></tr><tr><td>Open drawer</td><td>0.1</td></tr><tr><td>Clean object (per object, 3 total)</td><td> $0.8/3 \approx 0.267$ </td></tr><tr><td>Close drawer (after table cleaned)</td><td>0.1</td></tr><tr><td>Max score per trial</td><td>1.0</td></tr></table>

Annotating $t_{min}$ : For each sub-dataset in COXE (e.g. DROID [92]), we train a classifier to discern samples from $D_{p}$ and the sub-dataset. We use this classifier to assign a single $t_{min}$ value to all action chunks within this sub-dataset. This required us to train 48 classifiers for the 48 sub-datasets in COXE. Although in principle, we could have accomplished this using a single classifier $c_{\phi}(A_{t}, t, l)$ , where $l \in \{0, 1\}^{48}$ is a one-hot encoding indicating the source dataset of $A_{t}$ . Figures 21 and 22 show the distribution of $t_{min}$ assigned to the sub-datasets of COXE for both the table cleaning and the tower building task.

Annotating $t_{max}$ : We simply set $t_{max} = 10$ for the table cleaning and $t_{max} = 5$ for tower building. We did not perform a sweep. These values are unlikely to be optimal, especially since adding locality showed mixed results for table cleaning. Lowering $t_{max}$ in the tower building experiments appears to help.

Daras et al. [19] present a method to automatically label $t_{max}$ using a similar classifier-based approach. Their approach assigned $t_{max} = 0$ to nearly all datasets in MS++ and COXE. A full hyperparameter sweep on real hardware would have been prohibitively time-consuming. Hence, we selected values that are small enough to be conservative, yet large enough for a meaningful evaluation of locality.

Table Cleaning Evaluation: During each trial, 3 random unseen objects were placed on the table. The robot's task completion is measured according to the rubric in Table 11. The time limit is 90s. Each policy was evaluated on 20 trials.

The results include a minor camera resolution bug that lowers the policy performance across all table cleaning experiments. The image data was collected at 640 by 320 pixels and resized to 224 by 224 for training. Due to a bug at inference time, the image streams were 128 by 128 and upscaled to 224 by 224 for the policy; thus, the images during inference were effectively lower resolution than the images during training.

Tower Building Evaluation: The performance of each trial is the number of correct blocks placed before the robot topples the tower or requires human intervention. Correct indicates that the blocks follow the correct alternating direction and color shown in Figure 1f. Each policy was evaluated on 20 trials.

The camera resolution bug from the table cleaning experiments was resolved for the tower building experiments.

## H.2 OXE Ablations

## H.2.1 Qualitative Performance

The qualitative gap between the Ambient and co-trained policies is larger than the task completion metric would suggest. The 90s timeout artificially inflates the task completion rates of weak policies: they can fail repeatedly without penalty as long as they eventually succeed. To capture this, we recorded the number of grasp attempts and time required per object for each policy. Tables 12

## Task: Table Cleaning

![](images/a3a5be9e855c5fff5e03cd1855adce1a7f5af78a8dffa42b97859016affe930e.jpg)

![](images/274ed7f1a13db9680fc6a2d2d3c46874e3f845809c05c4cd97b8cf7baffc5753.jpg)

![](images/6396fd5469da458614c5fe0954414a8b743c16b64d72e2b1a87efc45b1fe715f.jpg)

![](images/e8c07c5fd533a31cee281fb5a6ff654fa6edaff666acf25d346acda726bbdf24.jpg)

$$
t _ {\mathrm{min}}
$$

![](images/0c170c5690aac24aa5dc598eff9037184774b9a16d65e4a4232dad4114830035.jpg)

![](images/fad511d3c0b3b6739a9b503f0b9f5bb3c52e312965fc705fb71ef83adfaaa0f1.jpg)  
Figure 21: Table Cleaning: Dataset re-weighting and $t_{min}$ visualization. These plots show the sampling ratio for $D_{p}$ (shown in blue) and each dataset in COXE at different diffusion times for the table cleaning task. Left: Co-training uses the same sampling ratio at all diffusion times. Right: The sampling ratio for each dataset varies at each diffusion time. Notably, at intermediate diffusion times, only data from $D_{p}$ can be used (the “Donut paradox” from Daras et al. [19]). The histogram shows the distribution of $t_{min}$ values assigned by the classifiers to each dataset in COXE.

![](images/a18b34e7149cc888668481f39964e0b2cf4d3bebe9c4c041951ea34267de3352.jpg)

![](images/dfa3d41549256a7e78e887b77e58271bf7185ed588b1035ba614284d3b631b05.jpg)

![](images/40410c4ba560509f2990dd325277083caa181b160b847327f2a36ce8c520086f.jpg)

![](images/374f52871298bd6f13531f839487eb3edb0a65eefdce2a44ca0ee217d360776a.jpg)

![](images/2b94e55b051cfb82ea29889dc0194f7fe556a166b53ef380795690b15a465b00.jpg)

![](images/d66f06ee350af96d59e23e07be9a3f4a41393ab56936744112b7e278168b5d81.jpg)  
Figure 22: Tower Building: Dataset re-weighting and $t_{min}$ visualization. See Figure 21 for the equivalent plot descriptions.

and 13 show that the Ambient Diffusion Policies require 25-40% fewer grasps and time per object than their co-trained counterparts.

The dataset re-weighting ablation is presented in the main body (Section 8.2).

## I Finetuning

## I.1 Implementation

To finetune the base policies, we reduce the learning rate by 10x to 20x. Checkpoints are saved frequently since finetuning a model for too long can degrade performance. In the simulated experiments, we evaluate every checkpoint and report the performance of the best policy. In the real-world experiments, we perform a preliminary evaluation of each checkpoint with 5-10 trials, and perform a full set of trials on the best checkpoint.

The controlled finetuning comparison is presented in the main body (Section 7.5).

Table 12: Average grasp attempts per object. The error bars on the Average row report the standard error of the mean across per-object values in each column (n = 10 objects).

<table><tr><td rowspan="2">Object</td><td colspan="2">MS++</td><td colspan="2">COXE</td></tr><tr><td>Co-train</td><td>Ambient + Locality</td><td>Co-train</td><td>Ambient + Locality</td></tr><tr><td>Apple</td><td>2</td><td>1</td><td>2</td><td>2</td></tr><tr><td>Figurine</td><td>4</td><td>2</td><td>2</td><td>2</td></tr><tr><td>Eraser</td><td>4</td><td>2</td><td>2</td><td>1</td></tr><tr><td>Play-Doh</td><td>2</td><td>2</td><td>2</td><td>2</td></tr><tr><td>Robot Gripper</td><td>3</td><td>2</td><td>2</td><td>2</td></tr><tr><td>Yarn</td><td>2</td><td>2</td><td>2</td><td>2</td></tr><tr><td>Camera Mount</td><td>1</td><td>1</td><td>1</td><td>2</td></tr><tr><td>3D Printed Object</td><td>2</td><td>2</td><td>3</td><td>1</td></tr><tr><td>Mustard</td><td>3</td><td>1</td><td>2</td><td>2</td></tr><tr><td>Garbage Bag Box</td><td>1</td><td>1</td><td>3</td><td>1</td></tr><tr><td>Average</td><td> $2.4 \pm 0.3$ </td><td> $1.6 \pm 0.2$ </td><td> $2.1 \pm 0.2$ </td><td> $1.7 \pm 0.2$ </td></tr></table>

Table 13: Average cleaning time per object. The error bars on the Average row report the standard error of the mean across per-object values in each column (n = 10 objects).

<table><tr><td rowspan="2">Object</td><td colspan="2">MS++</td><td colspan="2">COXE</td></tr><tr><td>Co-train</td><td>Ambient + Locality</td><td>Co-train</td><td>Ambient + Locality</td></tr><tr><td>Apple</td><td>21s</td><td>9s</td><td>30s</td><td>19s</td></tr><tr><td>Figurine</td><td>44s</td><td>20s</td><td>30s</td><td>20s</td></tr><tr><td>Eraser</td><td>39s</td><td>16s</td><td>18s</td><td>8s</td></tr><tr><td>Play-Doh</td><td>20s</td><td>20s</td><td>19s</td><td>20s</td></tr><tr><td>Robot Gripper</td><td>31s</td><td>19s</td><td>19s</td><td>20s</td></tr><tr><td>Yarn</td><td>22s</td><td>19s</td><td>20s</td><td>20s</td></tr><tr><td>Camera Mount</td><td>12s</td><td>8s</td><td>8s</td><td>20s</td></tr><tr><td>3D Printed Object</td><td>19s</td><td>20s</td><td>29s</td><td>8s</td></tr><tr><td>Mustard</td><td>33s</td><td>9s</td><td>18s</td><td>19s</td></tr><tr><td>Garbage Bag Box</td><td>8s</td><td>8s</td><td>30s</td><td>7s</td></tr><tr><td>Average</td><td> $24.9 \pm 3.6$ s</td><td> $14.8 \pm 1.7$ s</td><td> $22.1 \pm 2.3$ s</td><td> $16.1 \pm 1.8$ s</td></tr></table>

## I.2 Table Cleaning (OXE)

Figure 23 compares finetuning the best co-trained and Ambient OXE policies on $D_{p}$ in the table cleaning task; neither produced a statistically significant gain. The results are consistent with the controlled experiments in Section 7.5.

## J Distribution Shifts in the Observation Space

We explore two algorithms for handling suboptimality in the observation space. Other approaches exist – such as jointly denoising actions and observations. We note that “suboptimality” in the observation space can be ill-defined: an image from a different task is not necessarily suboptimal. In fact, it could increase the diversity of the dataset which is desirable.

Although Ambient Diffusion Policy only addresses action-space distribution shift, its results are already strong. Whether observation-space shift presents a fundamental problem and possible solutions remain open directions.

![](images/fb93996595e2e9f8985471fac4640b47dd5ead88e1c4a6ad7076a08fbdab8daa.jpg)  
Figure 23: Finetuning resulted in no statistically significant change in policy performance on the table cleaning task.

## J.1 Noising the Observations

Ambient Diffusion Policy contracts distributional differences in the actions by adding noise. We explore the same solution for the observations. During training, we add noise to the observation input (images, proprioception, etc) for half of the samples using the same noise schedule as the actions. The model takes in the noise level for both the actions and the observations. At inference time, the model performs inference with clean observations. This ablation is conceptually similar to concurrent work that noises observations $[93]$ ; however, their goal was to expand the sampling distribution of a base policy for RL finetuning. Figure 24a presents the results; it did not appear to help for table cleaning.

## J.2 Classifier-Free Guidance

In classifier-free guidance, the model learns to denoise the actions both unconditionally (i.e. without observations) and conditionally (with observations) [94]. We use the same training implementation as [94]. At sampling-time, we use the denoiser in (87) where w is the guidance weight. We explored classifier-free guidance as a potential solution since it trains an unconditional denoiser which is unaffected by corruptions in the observations.

$$
\tilde {h} _ {\theta} (A _ {t}, O, t) = (1 + w) \cdot h _ {\theta} (A _ {t}, O, t) - w \cdot h _ {\theta} (A _ {t}, \varnothing , t)\tag{87}
$$

Figure 24b shows that classifier-free guidance does not improve performance for all swept values of $w$ .

Noisy Observation Ablation (50 Clean + COXE)  
![](images/e54676ad7ef7eb31e31f33e5b29cc2af24fb37f330f5334043dbb791ba04bb4f.jpg)  
(a) Noising the observations during training.

![](images/4acd0365f2f8703c985e90d2fa536b55b944b97bfd8ea2a608b45329739a9a2a.jpg)  
(b) Classifier-free guidance vs. guidance weight w.  
Figure 24: Algorithms for handling distribution shift in the observation space. Neither approach produced stronger policies on the table cleaning task. (a) Adding Gaussian noise to the observations during training. (b) Classifier-free guidance [94] swept over guidance weights w.