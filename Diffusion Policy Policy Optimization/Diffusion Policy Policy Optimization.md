# Diffusion Policy Policy Optimization

Allen Z. Ren<sup>1</sup>, Justin Lidard<sup>1</sup>, Lars L. Ankile<sup>2</sup>, Anthony Simeonov<sup>2</sup>,

Pulkit Agrawal<sup>2</sup>, Anirudha Majumdar<sup>1</sup>, Benjamin Burchfiel<sup>3</sup>, Hongkai Dai<sup>3</sup>, Max Simchowitz<sup>2,4</sup>

<sup>1</sup>Princeton University <sup>2</sup>Massachusetts Institute of Technology

<sup>3</sup>Toyota Research Institute <sup>4</sup>Carnegie Mellon University

## Abstract

We introduce Diffusion Policy Policy Optimization, DPPO, an algorithmic framework including best practices for fine-tuning diffusion-based policies (e.g. Diffusion Policy [20]) in continuous control and robot learning tasks using the policy gradient (PG) method from reinforcement learning (RL). PG methods are ubiquitous in training RL policies with other policy parameterizations; nevertheless, they had been conjectured to be less efficient for diffusion-based policies. Surprisingly, we show that DPPO achieves the strongest overall performance and efficiency for fine-tuning in common benchmarks compared to other RL methods for diffusion-based policies and also compared to PG fine-tuning of other policy parameterizations. Through experimental investigation, we find that DPPO takes advantage of unique synergies between RL fine-tuning and the diffusion parameterization, leading to structured and on-manifold exploration, stable training, and strong policy robustness. We further demonstrate the strengths of DPPO in a range of realistic settings, including simulated robotic tasks with pixel observations, and via zero-shot deployment of simulation-trained policies on robot hardware in a long-horizon, multi-stage manipulation task. Website with code: diffusion-ppo.github.io.

## 1 Introduction

Large-scale pre-training with additional fine-tuning has become a ubiquitous pipeline in the development of language and image foundation models [12, 65, 75, 82]. Though behavior cloning with expert data [71] is rapidly emerging as dominant paradigm for pre-training robot policies [27, 28, 30, 50, 106], their performance can be suboptimal [64] due to expert data being suboptimal or expert data exhibiting limited coverage of possible environment conditions. As robot policies entail interaction with their environment, reinforcement learning (RL) [90] is a natural candidate for further optimizing their performance beyond the limits of demonstration data. However, RL fine-tuning can be nuanced for pre-trained policies parameterized as diffusion models [38], which have emerged as a leading parameterization for action policies [20, 66, 78], due in large part to their high training stability and ability to represent complex distributions [39, 49, 72, 80].

Contribution 1 (DPPO). We introduce Diffusion Policy Policy Optimization (DPPO), a generic framework as well as a set of carefully chosen design decisions for fine-tuning a diffusion-based robot learning policy via popular policy gradient methods [85, 91] in reinforcement learning.

The literature has already studied improving/fine-tuning diffusion-based policies (Diffusion Policy) using RL [35, 74, 100], and has applied policy gradient (PG) to fine-tuning non-interactive applications of diffusion models such as text-to-image generation [9, 21, 25]. Yet PG methods have been believed to be inefficient in training Diffusion Policy for continuous control tasks [74, 103]. On the contrary, we show that for a Diffusion Policy pre-trained from expert demonstrations, our methodology for fine-tuning via PG updates yields robust, high-performing policies with favorable training behavior.

![](images/5cef9e3c05c9012bbba251e54d04e89e47d562e0df6933893fe407230acbcf6a.jpg)  
Figure 1: We introduce DPPO, Diffusion Policy Policy Optimization, that fine-tunes pre-trained Diffusion Policy using policy gradient. DPPO treats the denoising process as a Markov Decision Process (“Diffusion MDP”) allowing the task reward signal (from “Environment MDP“) to easily propagate through Diffusion Policy (Fig. 3). Policy gradient update then leverages the tractable Gaussian likelihood at each denoising step. Extensive experiments in simulation and hardware show DPPO affords structured exploration and training stability during policy fine-tuning, and the final policy exhibits strong robustness and generalization.

Contribution 2 (Demonstration of DPPO’s Performance). We show that for fine-tuning a pre-trained Diffusion Policy, DPPO yields strong training stability across tasks and marked improvements in final performance in challenging robotic tasks in comparison to a range of alternatives, including those based on off-policy Q-learning [35, 74, 100, 103] and weighted regression [47, 67, 70], other demo-augmented RL methods [7, 40, 62], as well as common policy parameterizations such as Gaussian and Gaussian Mixture models with PG updates.

The above finding might be surprising because PG methods do not appear to take advantage of the unique capabilities of diffusion sampling (e.g., guidance [2, 45]). Through careful investigative experimentation, however, we find a unique synergy between RL fine-tuning and diffusion-based policies.

Contribution 3 (Understanding the mechanism of DPPO’s success). We complement our results with numerous investigative experiments that provide insight into the mechanisms behind DPPO’s strong performance. Compared to other common policy parameterizations, we provide evidence that DPPO engages in structured exploration that takes better advantage of the “manifold” of training data, and finds policies that exhibit greater robustness to perturbation.

Through ablations, we further show that our design decisions overcome the speculated limitation of PG methods for fine-tuning Diffusion Policy. Finally, to justify the broad utility of DPPO, we verify its efficacy across both simulated and real environments, and in situations when either ground-truth states or pixels are given to the policy as input.

Contribution 4 (Tackling challenging robotic tasks and settings). We show DPPO is effective in various challenging robotic and control settings, including ones with pixel observations and long-horizon manipulation tasks with sparse reward. We also deploy a policy trained in simulation via DPPO on real hardware in zero-shot, which exhibits a remarkably small sim-to-real gap compared to the baseline.

Potential impact beyond robotics. DPPO is a generic framework that can be potentially applied to finetuning diffusion-based models in sequential interactive settings beyond robotics. These include: extending diffusion-based text-to-image generation [9, 21] to a multi-turn interactive setting with human feedback; drug design/discovery applications [42, 58] with policy search on the molecular level in feedback with simulators (in the spirit of prior non-diffusion-based drug discovery with RL [73]); and the adaptation of diffusion-based language modeling [56, 83] to interactive (e.g. with human feedback [65]), problem-solving and planning tasks.

![](images/d9ca4679eef6bd9da8dcd6e62a4666c8e7badeeab4cc3abcb4aaf00dc972b164.jpg)  
Figure 2: DPPO solves challenging long-horizon manipulation tasks from FURNITURE-BENCH ([36]), and allows robust sim-to-real transfer without using any real data (see Section 5.4).

## 1.1 Overview of Approach

Like prior work [9, 21], our approach unrolls the denoising diffusion process into an MDP in which action likelihoods are explicit; from this, we construct a two-layer MDP whose outer layer corresponds to the environment MDP and inner layer to the denoising MDP. Given the challenges of RL training, we also present a number of design decisions tailored to DPPO that are crucial for enabling DPPO’s performance in challenging robotic settings, discussed in Contributions 2 and 4, including:

• We apply Proximal Policy Optimization (PPO) [85] to the two-layer MDP. We show how to efficiently estimate the advantage function for the PPO update.

• We show that for fine-tuning tasks, it often suffices to fine-tune only the last few steps of the denoising process, or fine-tune the Denoising Diffusion Implicit Model (DDIM) [87] instead.

• We propose modifications to the diffusion noise schedule to ensure stable training and adequate exploration, whilst also leveraging the natural stochasticity of diffusion models.

Our method is formally detailed in Section 4. As noted above, we conduct extensive experiments demonstrating both the success of DPPO in Section 5 and potential explanations thereof in Section 6.

## 2 Related Work

Policy optimization and its application to robotics. Policy optimization methods update an explicit representation of an RL policy — typically parameterized by a neural network — by taking gradients through action likelihoods. Following the seminal policy gradient (PG) method [91, 101], there have been a range of algorithms that further improve training stability and sample efficiency such as DDPG [54] and PPO [84]. PG methods have been broadly effective in training robot policies [4, 17, 43, 48], largely due to their training stability with high-dimensional continuous action spaces, as well as their favorable scaling with parallelized simulated environments. Given the challenges of from-scratch exploration in long-horizon tasks, PG has seen great success in fine-tuning a baseline policy trained from demonstrations [68, 76, 93]. Our experiments find DPPO performing on-policy PG often achieves stronger final performance in manipulation tasks, especially ones with long horizon and high-dimensional action space, than off-policy Q-learning methods [7, 40, 62]. See Appendix A.1 for extended discussions on PG fine-tuning of robot policies and related methods.

Learning and improving diffusion-based policies. Diffusion-based policies [5, 20, 66, 78, 89, 99, 105] have shown recent success in robotics and decision making applications. Most typically, these policies are trained from human demonstrations through a supervised objective, and enjoy both high training stability and strong performance in modeling complex and multi-modal trajectory distributions.

As demonstration data are often limited and/or suboptimal, there have been many approaches proposed to improve the performance of diffusion-based policies. One popular approach has been to guide the diffusion denoising process using objectives such as reward signal or goal conditioning [2, 15, 45, 53, 96]. More recent work has explored the possibilities of techniques including Q-learning and weighted regression, either from purely offline estimation [16, 22, 100], and/or with online interaction [35, 47, 74, 103]. See Appendix A.2 for detailed descriptions of these methods.

Policy gradient through diffusion models. RL techniques have been used to fine-tune diffusion models such as ones for text-to-image generation [9, 24, 25, 97]. Black et al. [9] treat the denoising process as an MDP and apply PPO updates. We build upon these earlier findings by embedding the denoising MDP into the environmental MDP of the dynamics in control tasks, forming a two-layer “Diffusion Policy $\mathbf { M D P } ^ { \prime }$ Though Psenka et al. [74] have already shown how PG can be taken through Diffusion Policy by propagating PG through both MDPs, they conjecture that it is likely to be ineffective due to large action variance caused by the increased effective horizon induced from the denoising steps. Our results contravene this supposition for diffusion-based policies in the fine-tuning setting.

## 3 Preliminaries

Markov Decision Process. We consider a Markov Decision Process (MDP)<sup>1</sup> $\mathcal { M } _ { \mathrm { E N V } } : = ( S , \mathcal { A } , P _ { 0 } , P , R )$ with states $s \in S$ , actions $a \in { \mathcal { A } } .$ , initial state distribution $P _ { 0 } .$ , transition probabilities $P ,$ , and reward R.

At each timestep t, the agent (e.g., robot) observes the state $s _ { t } \in S$ , takes an action $a _ { t } \sim \pi ( a _ { t } \mid s _ { t } ) \in { \mathcal { A } }$ transitions to the next state $s _ { t + 1 }$ according to $s _ { t + 1 } \sim P ( s _ { t + 1 } \mid s _ { t } , a _ { t } )$ while receiving the reward $R ( s _ { t } , a _ { t } ) ^ { 2 }$ Fixing the MDP $\mathcal { M } _ { \mathrm { E N V } }$ , we let $\mathbb { E } ^ { \pi }$ (resp. $\mathbb { P } ^ { \pi } )$ denote the expectation (resp. probability distribution) over trajectories $( s _ { 0 } , a _ { 0 } , . . . , s _ { T } , a _ { T } )$ with length $T + 1$ , with initial state distribution $s _ { 0 } \sim P _ { 0 }$ and transition operator $P .$ We aim to train a policy to optimize the cumulative reward, discounted by a function $\gamma ( \cdot )$ : :

$$
\mathcal {J} (\pi_ {\theta}) = \mathbb {E} ^ {\pi_ {\theta}, P _ {0}} \left[ \sum_ {t \geq 0} \gamma (t) R (s _ {t}, a _ {t}) \right].\tag{3.1}
$$

Policy optimization. The policy gradient method $( \mathrm { e . g . }$ , REINFORCE [101]) allows for improving policy performance by approximating the gradient of this objective w.r.t. the policy parameters:

$$
\nabla_ {\theta} \mathcal {J} (\pi_ {\theta}) = \mathbb {E} ^ {\pi_ {\theta}, P _ {0}} \left[ \sum_ {t \geq 0} \nabla_ {\theta} \log \pi_ {\theta} (a _ {t} | s _ {t}) r _ {t} (s _ {t}, a _ {t}) \right], \quad r _ {t} (s _ {t}, a _ {t}) := \sum_ {\tau \geq t} \gamma (\tau) R (s _ {\tau}, a _ {\tau}),\tag{3.2}
$$

where $r _ { t }$ is the discounted cumulative future reward from time t (more generally, $r _ { t }$ can be replaced by a Q-function estimator [91]), γ is the discount factor that depends on the time-step, and $\nabla _ { \theta }$ log $\pi _ { \boldsymbol { \theta } } ( a _ { t } | \boldsymbol { s } _ { t } )$ denotes the gradient of the logarithm of the likelihood of $a _ { t } \ | \ s _ { t }$ . To reduce the variance of the gradient estimation, a state-value function $\hat { V } ^ { \pi _ { \theta } } ( s _ { t } )$ can be learned to approximate $\mathbb { E } [ r _ { t } ]$ . The estimated advantage function ${ \hat { A } } ^ { \pi _ { \theta } } { \left( s _ { t } , a _ { t } \right) } : = r _ { t } { \left( s _ { t } , a _ { t } \right) } - { \hat { V } } ^ { \pi _ { \theta } } { \left( s _ { t } \right) }$ substitutes $r _ { t } ( s _ { t } , a _ { t } )$

Diffusion models. A denoising diffusion probabilistic model (DDPM) [38, 63, 86] represents a continuousvalued data distribution $p ( \cdot ) = p ( x ^ { 0 } )$ as the reverse denoising process of a forward noising process $q ( x ^ { k } | x ^ { k - 1 } )$ that iteratively adds Gaussian noise to the data. The reverse process is parameterized by a neural network $\varepsilon _ { \boldsymbol { \theta } } ( x _ { k } , k )$ , predicting the added noise $\varepsilon$ that converts $x _ { 0 }$ to $x _ { k }$ [38]. Sampling starts with a random sample $x ^ { K } \sim \mathcal { N } ( 0 , \mathrm { I } )$ and iteratively generates the denoised sample:

$$
x ^ {k - 1} \sim p _ {\theta} (x ^ {k - 1} | x ^ {k}) := \mathcal {N} (x ^ {k - 1}; \mu_ {k} (x ^ {k}, \varepsilon_ {\theta} (x ^ {k}, k)), \sigma_ {k} ^ {2} \mathrm{I}).\tag{3.3}
$$

Above, $\mu _ { k } ( \cdot )$ is a fixed function, independent of $\theta ,$ that maps $x ^ { k }$ and predicted $\varepsilon _ { \theta }$ to the next mean, and $\sigma _ { k } ^ { 2 }$ is a variance term that abides by a fixed schedule from $k = 1 , \ldots , K$ . We refer the reader to Chan [14] for an in-depth survey.

Diffusion models as policies. Diffusion Policy (DP; see Chi et al. [20]) is a policy $\pi _ { \theta }$ parameterized by a DDPM which takes in s as a conditioning argument, and parameterizes $p _ { \theta } ( a ^ { k - 1 } \mid a ^ { k } , s )$ as in Eq. (3.3). DPs can be trained via behavior cloning by fitting the conditional noise prediction $\varepsilon _ { \boldsymbol { \theta } } { \left( a ^ { k } , s , k \right) }$ to predict the added noise. Notice that unlike more standard policy parameterizations such as unimodal Gaussian policies, DPs do not maintain an explicit likelihood of $p _ { \theta } ( a ^ { 0 } \mid s )$ . In this work, we adopt the common practice of training DPs to predict an action chunk — a sequence of actions a few time steps (denoted $T _ { p } )$ into the future — to promote temporal consistency. Policy executes $T _ { a } \leq T _ { p }$ steps of the predicted chunk before the next prediction. From now on $a \in { \mathcal { A } }$ denotes the entire executed action chunk. For fair comparison, our diffusion and non-diffusion baselines use the same $T _ { a }$ unless noted.

## 4 DPPO: Diffusion Policy Policy Optimization <sup>s^t</sup>

As noted in Section 1 and Section 2, there has been much attention devoted to fine-tuning Diffusion Policy for improved performance [74, 103], focusing primarily on off-policy Q-learning [35, 100] and/or weighted regression. DPPO takes a different approach: differentiate through action likelihood and apply a policy gradient (PG) update like Eq. (3.2).<sup>R^t</sup>

## 4.1 A Two-Layer “Diffusion Policy MDP”

![](images/8a9df2dfcc56dac743806c2a16a05974a29a7d4571903350767204e17a20ffe8.jpg)  
Figure 3: We treat the denoising process in Diffusion Policy as an MDP, and the whole environment episode can be considered as a chain of such MDPs. Now the entire chain (“Diffusion Policy $\mathbf { M D P } ^ { \prime \prime } , \mathcal { M } _ { \mathrm { D P } } )$ involves a Gaussian likelihood at each (denoising) step and thus can be optimized with policy gradient. Blue circle denotes the state and red circle denotes the action in $\mathcal { M } _ { \mathrm { D P } }$

As observed in [9] and [74], a denoising process can be represented as a multi-step MDP in which policy likelihood of each denoising step can be obtained directly. We extend this formalism by embedding the Diffusion MDP into the environmental MDP, obtaining a larger “Diffusion Policy $\mathbf { M D P } ^ { \prime }$ denoted $\mathcal { M } _ { \mathrm { D P } }$ visualized in Fig. 3. Below, we use the notation δ to denote a Dirac distribution and $\otimes \mathrm { t o }$ denote a product distribution.

Recall the environment MDP $\mathcal { M } _ { \mathrm { E N V } } : = ( S , \mathcal { A } , P _ { 0 } , P , R )$ in Section 3. The Diffusion $\mathbf { M D P } \mathbf { \mathcal { M } } _ { \mathrm { D P } }$ uses indices $\bar { t } ( t , k ) = t K + ( K - k - 1 )$ corresponding to $( t , k )$ , which increases in t but (to keep the indexing conventions of diffusion) decreases lexicographically with $K - 1 \geq k \geq 0$ . The states, actions and rewards are

$$
\bar {s} _ {\bar {t} (t, k)} = (s _ {t}, a _ {t} ^ {k + 1}), \quad \bar {a} _ {\bar {t} (t, k)} = a _ {t} ^ {k}, \quad \bar {R} _ {\bar {t} (t, k)} (\bar {s} _ {\bar {t} (t, k)}, \bar {a} _ {\bar {t} (t, k)}) = \left\{ \begin{array}{l l} 0 & k > 0 \\ R (s _ {t}, a _ {t} ^ {0}) & k = 0 \end{array} \right.,
$$

where the bar-action at $\bar { t } ( t , k )$ is the action $a _ { t } ^ { k }$ after one denoising step. Reward is only given at times corresponding to when $a _ { t } ^ { 0 }$ is taken. The initial state distribution is $\bar { P } ^ { 0 } = P _ { 0 } \otimes \mathcal { N } ( 0 , { \bf I } )$ , corresponding to $s _ { 0 } \sim P _ { 0 }$ is the initial distribution from the environmental MDP and $a _ { 0 } ^ { K } \sim \mathcal { N } ( 0 , \mathbf { I } )$ independently. Finally, the transitions are

$$
\bar {P} (\bar {s} _ {\bar {t} + 1} \mid \bar {s} _ {\bar {t}}, \bar {a} _ {\bar {t}}) = \left\{ \begin{array}{l l} (s _ {t}, a _ {t} ^ {k}) \sim \delta_ {(s _ {t}, a _ {t} ^ {k})} & \bar {t} = \bar {t} (t, k), k > 0 \\ (s _ {t + 1}, a _ {t + 1} ^ {K}) \sim P (s _ {t + 1} \mid s _ {t}, a _ {t} ^ {0}) \otimes \mathcal {N} (0, \mathbf {I}) & \bar {t} = \bar {t} (t, k), k = 0 \end{array} \right..
$$

That is, the transition moves the denoised action $a _ { t } ^ { k }$ at step $\bar { t } ( t , k )$ into the next state when $k > 0$ , or otherwise progresses the environment MDP dynamics with $k = 0$ . The pure noise $a _ { t } ^ { K }$ is considered part of the environment when transitioning at $k = 0 .$ . In light of Eq. (3.3), the policy in $\mathcal { M } _ { \mathrm { D P } }$ takes the form

$$
\bar {\pi} _ {\theta} (\bar {a} _ {\bar {t} (t, k)} \mid \bar {s} _ {\bar {t} (t, k)}) = \pi_ {\theta} (a _ {t} ^ {k} \mid a _ {t} ^ {k + 1}, s _ {t}) = \mathcal {N} (a _ {t} ^ {k}; \mu (a _ {t} ^ {k + 1}, \varepsilon_ {\theta} (a _ {t} ^ {k + 1}, k + 1, s _ {t})), \sigma_ {k + 1} ^ {2} \mathrm{I}).\tag{4.1}
$$

Fortunately, Eq. (4.1) is a Gaussian likelihood, which can be evaluated analytically and is amenable to the policy gradient updates<sup>3</sup>:

$$
\nabla_ {\theta} \bar {\mathcal {J}} (\bar {\pi} _ {\theta}) = \mathbb {E} ^ {\bar {\pi} _ {\theta}, \bar {P}, \bar {P} ^ {0}} \left[ \sum_ {\bar {t} \geq 0} \nabla_ {\theta} \log \bar {\pi} _ {\theta} (\bar {a} _ {\bar {t}} \mid \bar {s} _ {\bar {t}}) \bar {r} (\bar {s} _ {\bar {t}}, \bar {a} _ {\bar {t}}) \right], \quad \bar {r} (\bar {s} _ {\bar {t}}, \bar {a} _ {\bar {t}}) := \sum_ {\tau \geq \bar {t}} \gamma (\tau) \bar {R} (\bar {s} _ {\tau}, \bar {a} _ {\tau}).\tag{4.2}
$$

Evaluating the above involves sampling through the denoising process, which is the usual “forward pass” that samples actions in Diffusion Policy; as noted above, the inital state can be sampled from the enviroment via $\bar { P } ^ { 0 } = P _ { 0 } \otimes \mathcal { N } ( 0 , \mathbf { I } )$ , where $P _ { 0 }$ is from the environment MDP.

## 4.2 Instantiating DPPO with Proximal Policy Optimization

We apply Proximal Policy Optimization (PPO) [1, 23, 41, 85], a popular improvement of the vanilla policy gradient update. The full pseudocode with implementation details are shown in Algorithm 1, Appendix B.

Definition 4.1 (Generalized PPO, clipping variant). Consider a general MDP. Given an advantage estimator $\hat { A } ( s , a )$ , the PPO update [85] is the sample approximation to

$$
\nabla_ {\theta} \mathbb {E} ^ {(s _ {t}, a _ {t}) \sim \pi_ {\theta_ {\mathrm{old}}}} \min \Big (\hat {A} ^ {\pi_ {\theta_ {\mathrm{old}}}} (s _ {t}, a _ {t}) \frac {\pi_ {\theta} (a _ {t} \mid s _ {t})}{\pi_ {\theta_ {\mathrm{old}}} (a _ {t} \mid s _ {t})}, \hat {A} ^ {\pi_ {\theta_ {\mathrm{old}}}} (s _ {t}, a _ {t}) \operatorname{clip} \big (\frac {\pi_ {\theta} (a _ {t} \mid s _ {t})}{\pi_ {\theta_ {\mathrm{old}}} (a _ {t} \mid s _ {t})}, 1 - \varepsilon , 1 + \varepsilon \big) \Big),
$$

where ε, the clipping ratio, controls the maximum magnitude of the policy updated from the previous policy.

We instantiate PPO in our diffusion MDP with $( s , a , t ) \gets ( \bar { s } , \bar { a } , \bar { t } )$ . Our advantage estimator takes a specific form that respects the two-level nature of the MDP: let $\gamma _ { \mathrm { E N V } } \in ( 0 , 1 )$ be the environment discount and $\gamma _ { \mathrm { { D E N O I S E } } } \in ( 0 , 1 )$ be the denoising discount. Consider the environment-discounted return:

$$
\bar {r} (\bar {s} _ {\bar {t}}, \bar {a} _ {\bar {t}}) := \sum_ {t ^ {\prime} \geq t} \gamma_ {\mathrm{ENV}} ^ {t} \bar {r} (\bar {s} _ {\bar {t} (t ^ {\prime}, 0)}, \bar {a} _ {\bar {t} (t ^ {\prime}, 0)}), \quad \bar {t} = \bar {t} (t, k),
$$

since ${ \bar { R } } ( { \bar { t } } ) = 0 { \mathrm { ~ a t ~ } } k > 0$ . This fact also obviates the need of estimating the value at $k > 1$ and allows us to use the following denoising-discounted advantage estimator<sup>4</sup>:

$$
\hat {A} ^ {\pi_ {\theta_ {\mathrm{old}}}} \left(\bar {s} _ {\bar {t}}, \bar {a} _ {\bar {t}}\right) := \gamma_ {\text { DENOISE }} ^ {k} \left(\bar {r} \left(\bar {s} _ {\bar {t}}, \bar {a} _ {\bar {t}}\right) - \hat {V} ^ {\bar {\pi} _ {\theta_ {\mathrm{old}}}} \left(\bar {s} _ {\bar {t} (t, 0)}\right)\right)
$$

The denoising-discounting has the effect of downweighting the contribution of noisier steps (larger k) to the policy gradient (see study in Appendix C.2). Lastly, we choose the value estimator to only depend on the $^ { 6 6 } { \boldsymbol { s } } ^ { , 5 }$ component of s¯:

$$
\hat {V} ^ {\bar {\pi} _ {\theta_ {\mathrm{old}}}} \big (\overline {{s}} _ {\bar {t} (t, 0)} \big) := \tilde {V} ^ {\bar {\pi} _ {\theta_ {\mathrm{old}}}} \big (s _ {t} \big),
$$

which we find leads to more efficient and stable training compared to also estimating the value of applying the denoised action $a _ { t } ^ { k = 1 }$ (part of $\bar { s } _ { \bar { t } ( t , 0 ) } )$ as shown in Appendix C.2.

## 4.3 Best Practices for DPPO

Fine-tune only the last few denoising steps. Diffusion Policy often uses up to K = 100 denoising steps with DDPM to better capture the complex data distribution of expert demonstrations. With DPPO, we can choose to fine-tune only a subset of the denoising steps instead, e.g., the last $K ^ { \prime }$ steps. Experimental results in Appendix C.2 shows this speeds up DPPO training and reduces GPU memory usage without sacrificing the final performance. Instead of fine-tuning the pre-trained model weights $\theta ,$ we make a copy $\theta _ { \mathrm { F T } } - \theta$ is frozen and used for the early denoising steps, while $\theta _ { \mathrm { F T } }$ is used for the last $K ^ { \prime }$ steps and updated with DPPO.

Fine-tune DDIM sampling. Instead of fine-tuning all K or the last few steps of the DDPM, one can also apply Denoising Diffusion Implicit Model (DDIM) [87] during fine-tuning, which greatly reduces the number of sampling steps $K ^ { \mathrm { D D I M } } \ll K$ , e.g., as few as 5 steps, and thus potentially improves DPPO efficiency as fewer steps are fine-tuned:

$$
x ^ {k - 1} \sim p _ {\theta} ^ {\mathrm{DDIM}} (x ^ {k - 1} | x ^ {k}) := \mathcal {N} (x ^ {k - 1}; \mu^ {\mathrm{DDIM}} (x ^ {k}, \varepsilon_ {\theta} (x ^ {k}, k)), \eta \sigma_ {k} ^ {2} \mathrm{I}), \quad k = K ^ {\mathrm{DDIM}},..., 0.\tag{4.3}
$$

Although DDIM is typically used as a deterministic sampler by setting $\eta = 0$ in Eq. (4.3), we can use $\eta > 0$ for fine-tuning in order to provide exploration noise and avoid calculating a Gaussian likelihood with a Dirac distribution. In practice, we set $\eta = 1$ for training (equivalent to applying DDPM [87]) and then $\eta = 0$ for evaluation. We use DDIM sampling for our pixel-based experiments and long-horizon furniture assembly tasks, where the efficiency improvements are much desired.

Diffusion noise scheduling. We use the cosine schedule for $\sigma _ { k }$ introduced in [63], which was originally annealed to a small value on the order of 1e−4 at $k = 0$ . In DPPO, the values of $\sigma _ { k }$ also translate to the exploration noise that is crucial to training efficiency. Empirically, we find that clipping $\sigma _ { k }$ to a higher minimum value (denoted $\sigma _ { \mathrm { m i n } } ^ { \mathrm { e x p } } , \mathrm { e . g . , 0 . 0 1 - 0 . 1 } )$ ) when sampling actions helps exploration (see study in Appendix C.2). Additionally we clip $\sigma _ { k }$ to be at least 0.1 (denoted $\sigma _ { \mathrm { m i n } } ^ { \mathrm { p r o b } } )$ when evaluating the Gaussian likelihood log $\bar { \pi } _ { \boldsymbol { \theta } } \big ( \bar { a } _ { \bar { t } } | \bar { s } _ { \bar { t } } \big )$ , which improves training stability by avoiding large magnitude.

Network architecture. We study both Multi-layer Perceptron (MLP) and UNet [81] as the policy heads in Diffusion Policy. MLP offers a simpler setup, and we find that it generally fine-tunes more stably with DPPO. Meanwhile, UNet, only applying convolutions to $a _ { t } ^ { k }$ , has the benefit of allowing fine-tuning with $T _ { a } < T _ { p } , { \bf e . g . } , T _ { p } = 1 6$ and $T _ { a } = 8$ . We find that DPPO benefits from pre-training with larger $T _ { p }$ (better prediction) and fine-tuning with smaller $T _ { a }$ (more amenable to policy gradient)<sup>5</sup>.

## 5 Performance Evaluation of DPPO

We study the performance of DPPO in popular RL and robotics benchmarking environments. We compare to other RL methods for fine-tuning a Diffusion Policy (Section 5.1), to other demo-augmented RL methods (Section 5.2), to other policy parameterizations with PG updates (Section 5.3), and then in multi-stage manipulation tasks including hardware evaluation (Section 5.4), and conclude with ablations (Section 5.5). While our evaluations focus primarily on fine-tuning, we also present training-from-scratch results in Appendix C. Wall-clock times are reported and discussed in Appendix D; they are roughly comparable (often faster) than other diffusion-based RL baselines, much faster than other demo-augmented RL baselines, though can be up to 2× slower than other policy parameterizations. Full choices of training hyperparameters and additional training details are presented in Appendix E.

Environments: OpenAI Gym. We first consider three OpenAI GYM locomotion benchmarks [11] common in the RL literature: {Hopper- $- \mathtt { v } 2$ , Walker2D-v2, HalfCheetah-v2}. All policies are pretrained with the full medium-level datasets from D4RL [29] with state input and action chunk size $T _ { a } = 4 .$ We use the original dense reward setup in fine-tuning.

Environments: Franka Kitchen. We also consider the FRANKA-KITCHEN environment first introduced in [32] involving a Franka arm completing four countertop tasks. We use the three datasets from D4RL [29]: {Kitchen-Complete-v0, Kitchen-Partial-v0, Kitchen-Mixed-v0}, containing demonstrations of varying levels of completeness. State input is considered and action chunk size $T _ { a } = 4$ is used for DPPO. We use the original sparse reward setup in fine-tuning.

Environments: Robomimic. Next we consider four simulated robot manipulation tasks from the ROBOMIMIC benchmark [60], {Lift, Can, Square, Transport}, ordered in increasing difficulty. These tasks are more representative of real-world robotic tasks, and Square and Transport (Fig. 4) are considered very challenging for RL training. Both state and pixel policy input are considered. State-based and pixel based policies are pre-trained with 300 and 100 demonstrations provided by ROBOMIMIC, respectively. We mostly consider the noisier Multi-Human (MH) data from ROBOMIMIC but also consider the cleaner Proficient-Human (PH) data in some experiments. We consider $T _ { a } = 4$ for Can, Lift, and Square, and $T _ { a } = 8$ for Transport. They are then fine-tuned with sparse reward equal to 1 upon task completion.

Environments: Furniture-Bench & real furniture assembly. Finally, we demonstrate solving longerhorizon, multi-stage robot manipulation tasks from the FURNITURE-BENCH [36] benchmark. We consider three simulated furniture assembly tasks, {One-leg, Lamp, Round-table}, shown in Fig. 4 and described in detail in Appendix E.8. We consider two levels of randomness over initial state distribution, Low and Med, defined by the benchmark. All policies are pre-trained with 50 human demonstrations collected in simulation and $T _ { a } = 8$ . They are then fine-tuned with sparse (indicator of task stage completion) reward. We also evaluate the zero-shot sim-to-real performance with One-leg.

![](images/7dbcaff89e019a8f07ed6eed2633cdbb747e41b5b5e48b147cdfdf78c6c919ab.jpg)  
Figure 4: Long-horizon robot manipulations tasks including (left) the bimanual Transport from ROBOMIMIC and (right) FURNITURE-BENCH tasks (full rollouts visualized in Fig. A12).

## 5.1 Comparison to diffusion-based RL algorithms

We compare DPPO to an extensive list of RL methods for fine-tuning diffusion-based policies. Baseline names are color-coordinated with their plot colors. Two methods, DRWR, DAWR, are our own, novel baselines that are based on reward-weighted regression (RWR, Peters and Schaal [70]) and advantage-weighted regression (AWR, Peng et al. [67]). The remaining methods, DIPO [103], IDQL [35], DQL [100], and QSM [74], are existing in the literature. Among them, QSM and DIPO are proposed specifically for trainingfrom-scratch, while IDQL and DQL can be more suitable to online fine-tuning. We evaluate on the three OpenAI GYM tasks and the four ROBOMIMIC tasks with state input; detailed descriptions of all baselines and training details are in Appendix E.3.

Overall, DPPO performs consistently, exhibits great training stability, and enjoys strong fine-tuning performance across tasks. In the GYM tasks (Figure 5, top row), IDQL and DIPO exhibit competitive performance, while the other methods often perform worse and train less stably. Baselines also tend to exhibit performance drop at the beginning of fine-tuning, mostly likely due to misestimating Q values of the pre-trained policy. DPPO is the strongest performer in the ROBOMIMIC tasks (Figure 5, bottom row), especially in the challenging Transport tasks. IDQL and, surprisingly, DRWR, are strong baselines in {Lift, Can, Square} but underperforms in Transport, while all other baselines fare worse still. We postulate that the other baselines, all performing off-policy updates and propagating gradients from the imperfect Q function to the actor, suffers from even greater training instability in sparse-reward ROBOMIMIC tasks given the continuous action space plus large action chunk sizes (see furtuer studies in Appendix C.2).

![](images/bc0e03b68241d9b207e50b15d00ace78bf47bf1663ceffebd1e48661d4d2daba.jpg)  
Figure 5: Comparing to other diffusion-based RL algorithms. Top row: GYM tasks [11] averaged over five seeds. Bottow row: ROBOMIMIC tasks [60], averaged over three seeds, with state observation. DPPO curves are slightly thicker for better visibility.

## 5.2 Comparison to other demo-augmented RL algorithms

We compare DPPO with recently proposed RL methods for training robot policies (not necessarily diffusionbased) leveraging offline data, including RLPD [7], Cal-QL [62], and IBRL [40]. These methods add expert data in the replay buffer and performs off-policy updates (IBRL and Cal-QL also pretrain with behavior cloning and offline RL objectives, respectively), which significantly improves sample efficiency vs. DPPO in HalfCheetah-v2 (Fig. 6, top left).

Among the FRANKA-KITCHEN settings (Fig. 6, top right), we find RLPD and IBRL fail to learn well especially with noisier demonstrations from Kitchen-Partial-v0 and Kitchen-Mixed-v0. Cal-QL achieves competitive performance but DPPO still achieves overall the best performance especially with Kitchen-Complete-v0. We note that DPPO, not using any expert data during fine-tuning, can be sensitive to the pre-training performance; we find the incomplete demonstrations in Kitchen-Partial-v0 and Kitchen-Mixed-v0 cause challenge in fully modeling the multi-modality of the data even with diffusion parameterization and prevent DPPO from achieving (near-)perfect fine-tuning performance.

Nonetheless, we believe the expert demonstrations from ROBOMIMIC are most reflective of pre-training plus fine-tuning in robotics as all demonstrations complete the task despite the varying quality. Fig. 6 bottom row shows the performance of DPPO and baselines using either cleaner PH or noisier MH data in Can and Square; DPPO exhibits much stronger final performance. RLPD and Cal-QL fail to learn at all and IBRL saturates at lower success levels. DPPO also runs significantly faster in wall-clock time than the baselines as it leverages sampling from highly parallelized environments<sup>6</sup>; thus we cap the number of samples at 1e6 for the baselines in ROBOMIMIC, also since their performance saturates.

![](images/d561c3da067203f7ad0659c22e43c9276f0332f6bc63d055a745f8797eb06726.jpg)

![](images/ede0b09b354bd66cb22cd401f9826934495cfba481ce44b6610cb87f1c1cbd7c.jpg)

![](images/120d6a2380944e296d53bcdff924d7f77769b9e404ce030c719c04d61088d89d.jpg)

![](images/a1540685a9555351d352598d81e57a866150fe017b6b131fac7d6bcf8c68a55f.jpg)

![](images/f0a62bfbe50bb5cf04ec7c2c1b5fb7f3c4970483443a704e378571fec6943885.jpg)

![](images/3c6691b7739b83a0bc9f333f68e1ea56809d371c6eda614e2edaa7745eb131f8.jpg)

![](images/338a4024994069b2b64065e55dfa272cf480997d514851b260f264c283f60fea.jpg)  
Figure 6: Comparing to other demo-augmented RL algorithms. Top left: HalfCheetah-v2 task [11] averaged over five seeds. Top right: FRANKA-KITCHEN task [32] averaged over three seeds. Bottom row: ROBOMIMIC tasks [60] averaged over three seeds with state observation, using either Proficient-Human (PH) or Multi-Human (MH) data.

## 5.3 Comparison to other policy parameterizations

We compare DPPO with popular RL policy parameterizations: unimodal Gaussian with diagonal covariance [91] and Gaussian Mixture Model (GMM [8]), using either MLPs or Transformers [94], and also fine-tuned with the PPO objective. We compare these to DPPO-MLP and DPPO-UNet, which use either MLP or UNet as the network backbone. We evaluate on the four tasks from ROBOMIMIC (Lift, Can, Square, Transport) with both state and pixel input. With state input, DPPO pre-trains with 20 denoising steps and then fine-tunes the last 10. With pixel input, DPPO pre-trains with 100 denoising steps and then finetunes 5 DDIM steps.

![](images/b3b9f04fb77fb60d0dc4e5c3c50b3101c71f943bfa6d9f1a8ca5978635c040c0.jpg)

![](images/ab958607ee0d705bc0df433cd498417f2ba21d70c382b577b43801be7f064bfb.jpg)

![](images/cc78bf5e36f049df4be0e51f9e495a33e78755b62561981cd02401fe5e36880e.jpg)

![](images/b1897ee208ad9e26d0fde9b9fd20200db975d780e08be30934e834d2b58cd0fd.jpg)  
Figure 7: Comparing to other policy parameterizations in the more challenging Square and Transport tasks from ROBOMIMIC, with state (left) or pixel (right) observation. Results are averaged over three seeds.

Fig. 7 display results for the more challenging Square and Transport — we defer the results in Lift and Can to Fig. A9. With state input, DPPO outperforms Gaussian and GMM policies, with faster convergence to ∼100% success rate in Lift and Can, and greater final performance on Square and the challenging Transport, where it reaches > 90%. UNet and MLP variants perform similarly, with the latter training somewhat more rapidly. With pixel inputs, we use a Vision-Transformer-based (ViT) image encoder introduced in Hu et al. [40] and an MLP head and compare the resulting variants DPPO-ViT-MLP and Gaussian-ViT-MLP (we omit GMM due to poor performance in state-based training). While the two are comparable on Lift and Can, DPPO trains more quickly and to higher accuracy on Square, and drastically outperforms on Transport, whereas Gaussian does not improve from its 0% pre-trained success rate. To our knowledge, DPPO is thefirst RL algorithm to solve Transport from either state or pixel input to high (>50%) success rates.

## 5.4 Evaluation on Furniture-Bench, and sim-to-real transfer

Here we evaluate DPPO on the long-horizon manipulation tasks from FURNITURE-BENCH [36]. We compare DPPO to Gaussian-MLP, the overall most effective baseline from Section 5.3. Fig. 8 (top row) shows the evaluation success rate over fine-tuning iterations. DPPO exhibits strong training stability and improves policy performance in all six settings. Gaussian-MLP collapses to zero success rate in all three tasks with Med randomness (except for one seed in Lamp) and Round-table with Low randomness.

Note that we are only using 50 human demonstrations for pre-training; we expect DPPO can leverage additional human data (better state space coverage) to further improve in Med, which is corroborated by ablation studies in Appendix C.3 examining the effect of pre-training data on DPPO.

Sim-to-real transfer. We evaluate DPPO and Gaussian policies trained in the simulated One-leg task on physical hardware zero-shot (i.e., no real data fine-tuning / co-training) over 20 trials. Both policies run at 10Hz. Please see additional simulation training and hardware details in Appendix E.8. Fig. 8 (bottom row) shows simulated and hardware success rates after pre-training and fine-tuning. Notably, DPPO improves the real-world performance to 80% (16 out of 20 trials). Though the Gaussian policy achieves a high success rate in simulation after fine-tuning (88%), it fails entirely on hardware (0%). Supplemental video suggests it exhibits volatile and jittery behavior. For stronger comparison, we also fine-tune the Gaussian policy with an auxiliary behavior-cloning loss [93] such that the fine-tuned policy is encouraged to stay close to the base policy. However, this limits fine-tuning and only leads to 53% success rate in simulation and 50% in reality.

![](images/4dab5cc1ae442a16b9f58323160704f4acfd14db98755c5eb597ad7bf3b7ec44.jpg)

![](images/03b69df613676369ece0a9287f186fd883f917d4aa017deb9e61b5cd6091141d.jpg)

![](images/f3172ff25f60b14ca40deb44807d3825111f502535f6bf6fb6f01e8dbf2c134e.jpg)

![](images/9087cde0546b89e94a749be5229f8710aaed6b9cb4b64080f2ced2ce2f8a084c.jpg)

![](images/14637b9acc72ede022f3ce1a6516b1ba61d12d483285f802f1c34676c304a78f.jpg)  
Figure 8: (Top) DPPO vs. Gaussian-MLP baseline in simulated FURNITURE-BENCH tasks. Results are averaged over three seeds. (Bottom) Sim-to-real transfer results in One-leg.

Qualitatively, we find fine-tuned policies to be more robust and exhibit more corrective behaviors than pre-trained-only policies, especially during the insertion stage of the task; such behaviors are visualized in Fig. 2 and Fig. 9 shows representative rollouts on hardware. Overall, these results demonstrate the strong sim-to-real capabilities of DPPO; Section 6 provides a conjectural mechanism for why this may be the case.

## 5.5 Summary of ablation findings

We conduct extensive ablation studies in Appendix C.2. Our main findings include: (1) for challenging tasks, using a value estimator which depends on environment state but is independent ofdenoised action is crucial for performance; we conjecture that this is related to the high stochasticity of Diffusion Policy; (2) there is a sweet spot for clipping the denoising noise level for DPPO exploration, trading off between too little exploration and too much action noise; (3) DPPO is resilient to fine-tuning fewer-than-K denoising steps, yielding improved runtime and comparable performance; (4) DPPO yields improvements over Gaussian-MLP baselines for varying levels of expert demonstration data, and achieves comparable final performance and sample efficiency when training from scratch in GYM environments.

## 6 Understanding the performance of DPPO

The improvement of DPPO over popular Gaussian and GMM methods in Section 5.3 comes as a surprise initially as DPPO solves a much longer Diffusion Policy MDP (Section 4) than the original environment MDP that other methods solve. This leads us to study the factors contributing to DPPO’s improvements in performance. Our findings highlight three major contributing factors:

(1) DPPO induces structured exploration near the pre-training data manifold [26].

(2) DPPO updates the action distribution progressively through the multi-step denoising process, which can be flexible and robust to policy collapse.

![](images/ad9003dd0fa94f2baf377eaad91876c4f75f2c913f026b722f1fe6de5212c915.jpg)  
(A) Pre-trained Difusion policy performs successful rollout

![](images/02907bfb29b5d4e0d167df40bb996e18f648a044cbf576e6c47687cda989c3e1.jpg)  
(B) Policy pushes peg down without proper alignment with the hole before releasing the peg<sub>,</sub> making it topple over

![](images/fd2d15ef63e8e9824be032c9696e2450bebaac46f0021a8f28517a94ad24a295.jpg)  
(C) Fine-tuned DPPO policy performs successful rollout

![](images/3c5f46ef2e5d13b607f66d6c7c00b79925ecb98422046a73b14afe8ce9c8dd3e.jpg)  
(D) Initial peg alignment is of<sub>,</sub> the policy corrects placement until it is properly inserted in the hole before letting go

Figure 9: Qualitative comparison of pre-trained vs. fine-tuned DPPO policies in hardware evaluation. (A) Successful rollout with the pre-trained policy. (B) Failed rollout with the pre-trained policy due to imprecise insertion. (C) Successful rollout with the fine-tuned policy. (D) Successful rollout with the finetuned policy exhibiting corrective behavior.

## (3) DPPO leads to fine-tuned policies robust to perturbations in dynamics and initial state distribution.

We use the Avoid environment from D3IL benchmark [46], where a robot arm needs to reach the other side of the table while avoiding an array of obstacles (Fig. 10, top-left). The action space is the 2D target location of the end-effector. D3IL provides a set of expert demonstrations that covers different possible paths to the goal line — we consider three subsets of the demonstrations, M1, M2, and M3 in Fig. 10, each with two distinct modes. We choose such relatively simple setups with only two modes in each setting such that Gaussian (with exploration noise)<sup>7</sup> and GMM can fit the expert data distribution reasonably well, allowing fair comparisons in fine-tuning.

We pre-train MLP-based Diffusion, Gaussian, and GMM policies (action chunk size $T _ { a } = 4$ unless noted) with the demonstrations. For fine-tuning, we assign sparse reward when the robot reaches the goal line from the topmost mode. Gaussian and GMM policies are also fine-tuned with the PPO objective.

Benefit 1: Structured, on-manifold exploration. Fig. 10 (right) shows the sampled trajectories (with exploration noise) from DPPO, Gaussian, and GMM during the first iteration of fine-tuning. DPPO explores in wide coverage around the expert data manifold, whereas Gaussian generates less structured exploration noise (especially in M2) and GMM exhibits narrower coverage. Unlike Gaussian policy that adds noise only to the final sampled action, diffusion adds multiple rounds of noise through denoising. Each denoising step expands the coverage with new noise while also pushing the newly denoised action towards the expert data manifold [69]. Moreover, the combination of diffusion parameterization with the denoising of action chunks means that policy stochasticity in DPPO is structured in both action dimension and time horizon. Quantitatively, Fig. A8 in Appendix shows DPPO achieves greater fine-tuning efficiency than Gaussian and GMM in all settings if a sufficient number of denoising steps is fine-tuned, consistent with the findings in Section 5.3 and Section 5.5.

![](images/1f3b77215249a975ddd41bf80a04dfa0701e602ae162c53ad4204ba0c24f50fd.jpg)  
Figure 10: (Left) We use the Avoid environment from D3IL benchmark [46] to visualize the DPPO’s exploration tendencies. We design the task of always reaching the green goal line from the topmost mode. (Right) Structured exploration. We show sampled trajectories at the first iteration of fine-tuning for DPPO, Gaussian, and GMM after pre-training on three sets of expert demonstrations, M1, M2, and M3.

It is possible, however, that the on-manifold exploration we observe with DPPO hinders fine-tuning when aggressive, unstructured exploration is desired. We conjecture this is the case in the Lamp environment with Low randomness, in which DPPO slightly underperforms the Gaussian baseline (Fig. 8). Moreover, we do not observe that DPPO is substantially better (nor is it any worse) than Gaussian in exploration fromscratch (see Appendix C). Thus, we anticipate that this structured, on-manifold exploration confers the greatest benefit when pre-training provides sufficient coverage of relevant success modes. In particular, we believe that this makes DPPO uniquely suited for fine-tuning large Diffusion Policy pre-trained on multiple tasks [99], as it may exhibit sufficient mode coverage given training data diversity.

Benefit 2: Training stability from multi-step denoising process. In Fig. 11 (left), we run fine-tuning after pre-training with M2 and attempt to de-stabilize fine-tuning by gradually adding noise to the action during the fine-tuning process (see Appendix E.9 for details). We find that Gaussian and GMM’s performance both collapse, while with DPPO, the performance is robust to the noise if at least four denoising steps are used. This property also allows DPPO to apply significant noise to the sampled actions, simulating an imperfect low-level controller to facilitate sim-to-real transfer in Section 5.4. In Fig. 11 (right), we also find DPPO enjoys greater training stability when fine-tuning long action chunks, e.g., up to $T _ { a } = 1 6$ , while Gaussian and GMM can fail to improve at all.

Fig. 12 visualizes how DPPO affects the multi-step denoising process. Over fine-tuning iterations, the action distribution gradually converges through the denoising steps — the iterative refinement is largely preserved, as opposed to, e.g., “collapsing” to the optimal actions at the first fine-tuned denoising step or the final one. We postulate this contributes to the training stability of DPPO.

![](images/7f6a00a4ba4f2f558049b93bfc3b30e1aee6f001da46b669e353e299689fefe1.jpg)

![](images/0aab3c79e22e66a056bed24f1cc42f7a4587a51f2810dc9bb07342972b04fbdf.jpg)

Figure 11: Training stability. Fine-tuning performance (averaged over five seeds, standard deviation not shown) after pre-training with M2. (Left) Noise is injected into the applied actions after a few training iterations. (Right) The action chunk size $T _ { a }$ is varied.  
![](images/132777e6997d819e5b3f5d02c874f5f9cda7a138cb1bc22020344e532a3d8c6c.jpg)  
Figure 12: Preserving the iterative refinement. The 2D actions from 50 trajectories at the branching point through fine-tuning iterations after pre-training with M2. For DPPO, we also visualize the action distribution through the final denoising steps at each fine-tuning iteration.

Benefit 3: Robust and generalizable fine-tuned policy. DPPO also generates final policies robust to perturbations in dynamics and the initial state distribution. In Fig. 13, we again add noise to the actions sampled from the fine-tuned policy (no noise applied during training) and find that DPPO policy exhibits strong robustness to the noise compared to the Gaussian policy. DPPO policy also converges to the (near-)optimal path from a larger distribution of initial states. This finding echoes theoretical guarantees that Diffusion Policy, capable of representing complex multi-modal data distribution, can effectively deconvolve noise from noisy states [10], a property used in Chen et al. [15] to stabilize long-horizon video generation.

![](images/47d9b7884c70d9c353ae371c97f114ffc8d8af5e12afade5e25128f452b66f10.jpg)  
Figure 13: Policy robustness after fine-tuning. Green dot / box indicates the initial state region.

## 7 Conclusion and Future Work

We present Diffusion Policy Policy Optimization (DPPO) for fine-tuning a pre-trained Diffusion Policy with the policy gradient method. DPPO leverages the sequential nature of the diffusion denoising process and fine-tunes the entire chain of diffusion MDPs. DPPO exhibits structured online exploration, strong training stability, and robustness and generalization at deployment. We demonstrate the efficiency and effectiveness of DPPO fine-tuning in various RL and robotics benchmarks, as well as strong sim-to-real transfer of a DPPO policy in a long-horizon, multi-stage manipulation task.

We believe DPPO will become an important component in the pre-training-plus-fine-tuning pipeline for training general-purpose real-world robotic policies. To this end, we hope in future work to further showcase the promise of DPPO for simulation-to-real transfer [18, 19, 52, 77] in which we fine-tune a vision-based policy that has been pre-trained on a variety of diverse tasks. We expect this pre-training to provide a large and diverse expert data manifold, of which, as we have shown in Section 6, DPPO is well-suited to take advantage for better exploration during fine-tuning. We are also excited to understand how DPPO can fit together with other decision-making tools such as model-based planning [45] and decision-making aided by video prediction [15]; these tools may help address the main limitation of DPPO — its lower sample efficiency than off-policy methods — and unlocking performing practical RL in physical hardware or generative simulation (e.g., video-based [13, 104]). Finally, as noted in the introduction, we eagerly anticipate applications of DPPO in domains beyond robotics, where diffusion models have shown promise for combinatorial search and sequence modeling [56, 73].

## Acknowledgments

We would like to thank Lirui Wang and Terry Suh for helpful discussions in the early stage of the project. The authors were partially supported by the Toyota Research Institute (TRI). This article solely reflects the opinions and conclusions of its authors and not TRI or any other Toyota entity.

## References

[1] J. Achiam. Spinning Up in Deep Reinforcement Learning. 2018.

[2] A. Ajay, Y. Du, A. Gupta, J. B. Tenenbaum, T. S. Jaakkola, and P. Agrawal. Is conditional generative modeling all you need for decision making? In The Eleventh International Conference on Learning Representations, 2023.

[3] M. Alakuijala, G. Dulac-Arnold, J. Mairal, J. Ponce, and C. Schmid. Residual reinforcement learning from demonstrations. arXiv preprint arXiv:2106.08050, 2021.

[4] O. M. Andrychowicz, B. Baker, M. Chociej, R. Jozefowicz, B. McGrew, J. Pachocki, A. Petron, M. Plappert, G. Powell, A. Ray, et al. Learning dexterous in-hand manipulation. The International Journal ofRobotics Research, 2020.

[5] L. Ankile, A. Simeonov, I. Shenfeld, and P. Agrawal. Juicer: Data-efficient imitation learning for robotic assembly. arXiv, 2024.

[6] L. Ankile, A. Simeonov, I. Shenfeld, M. Torne, and P. Agrawal. From imitation to refinement–residual rl for precise visual assembly. arXiv preprint arXiv:2407.16677, 2024.

[7] P. J. Ball, L. Smith, I. Kostrikov, and S. Levine. Efficient online reinforcement learning with offline data. In International Conference on Machine Learning, pages 1577–1594. PMLR, 2023.

[8] C. M. Bishop and N. M. Nasrabadi. Pattern recognition and machine learning. Springer, 2006.

[9] K. Black, M. Janner, Y. Du, I. Kostrikov, and S. Levine. Training diffusion models with reinforcement learning. arXiv preprint arXiv:2305.13301, 2023.

[10] A. Block, A. Jadbabaie, D. Pfrommer, M. Simchowitz, and R. Tedrake. Provable guarantees for generative behavior cloning: Bridging low-level stability and high-level behavior. Advances in Neural Information Processing Systems, 2024.

[11] G. Brockman, V. Cheung, L. Pettersson, J. Schneider, J. Schulman, J. Tang, and W. Zaremba. Openai gym. arXiv preprint arXiv:1606.01540, 2016.

[12] T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 2020.

[13] J. Bruce, M. D. Dennis, A. Edwards, J. Parker-Holder, Y. Shi, E. Hughes, M. Lai, A. Mavalankar, R. Steigerwald, C. Apps, et al. Genie: Generative interactive environments. In Forty-first International Conference on Machine Learning, 2024.

[14] S. H. Chan. Tutorial on diffusion models for imaging and vision. arXiv preprint arXiv:2403.18103, 2024.

[15] B. Chen, D. M. Monso, Y. Du, M. Simchowitz, R. Tedrake, and V. Sitzmann. Diffusion forcing: Next-token prediction meets full-sequence diffusion. arXiv preprint arXiv:2407.01392, 2024.

[16] H. Chen, C. Lu, C. Ying, H. Su, and J. Zhu. Offline reinforcement learning via high-fidelity generative behavior modeling. arXiv preprint arXiv:2209.14548, 2022.

[17] T. Chen, J. Xu, and P. Agrawal. A system for general in-hand object re-orientation. In Conference on Robot Learning, 2022.

[18] Y. Chen, C. Wang, L. Fei-Fei, and C. K. Liu. Sequential dexterity: Chaining dexterous policies for long-horizon manipulation. arXiv preprint arXiv:2309.00987, 2023.

[19] C. Chi, B. Burchfiel, E. Cousineau, S. Feng, and S. Song. Iterative residual policy: for goalconditioned dynamic manipulation of deformable objects. The International Journal of Robotics Research, 2024.

[20] C. Chi, Z. Xu, S. Feng, E. Cousineau, Y. Du, B. Burchfiel, R. Tedrake, and S. Song. Diffusion policy: Visuomotor policy learning via action diffusion. The International Journal of Robotics Research, 2024.

[21] K. Clark, P. Vicol, K. Swersky, and D. J. Fleet. Directly fine-tuning diffusion models on differentiable rewards. arXiv preprint arXiv:2309.17400, 2023.

[22] Z. Ding and C. Jin. Consistency models as a rich and efficient policy class for reinforcement learning. arXiv preprint arXiv:2309.16984, 2023.

[23] L. Engstrom, A. Ilyas, S. Santurkar, D. Tsipras, F. Janoos, L. Rudolph, and A. Madry. Implementation matters in deep rl: A case study on ppo and trpo. In International conference on learning representations, 2019.

[24] Y. Fan and K. Lee. Optimizing ddpm sampling with shortcut fine-tuning. arXiv preprint arXiv:2301.13362, 2023.

[25] Y. Fan, O. Watkins, Y. Du, H. Liu, M. Ryu, C. Boutilier, P. Abbeel, M. Ghavamzadeh, K. Lee, and K. Lee. Reinforcement learning for fine-tuning text-to-image diffusion models. Advances in Neural Information Processing Systems, 2024.

[26] C. Fefferman, S. Mitter, and H. Narayanan. Testing the manifold hypothesis. Journal ofthe American Mathematical Society, 2016.

[27] P. Florence, L. Manuelli, and R. Tedrake. Self-supervised correspondence in visuomotor policy learning. IEEE Robotics and Automation Letters, 2019.

[28] P. Florence, C. Lynch, A. Zeng, O. A. Ramirez, A. Wahid, L. Downs, A. Wong, J. Lee, I. Mordatch, and J. Tompson. Implicit behavioral cloning. In Conference on Robot Learning. PMLR, 2022.

[29] J. Fu, A. Kumar, O. Nachum, G. Tucker, and S. Levine. D4rl: Datasets for deep data-driven reinforcement learning. arXiv preprint arXiv:2004.07219, 2020.

[30] Z. Fu, T. Z. Zhao, and C. Finn. Mobile aloha: Learning bimanual mobile manipulation with low-cost whole-body teleoperation. arXiv preprint arXiv:2401.02117, 2024.

[31] W. Goo and S. Niekum. Know your boundaries: The necessity of explicit behavioral cloning in offline rl. arXiv preprint arXiv:2206.00695, 2022.

[32] A. Gupta, V. Kumar, C. Lynch, S. Levine, and K. Hausman. Relay policy learning: Solving longhorizon tasks via imitation and reinforcement learning. arXiv preprint arXiv:1910.11956, 2019.

[33] T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In International conference on machine learning, pages 1861–1870. PMLR, 2018.

[34] S. Haldar, J. Pari, A. Rai, and L. Pinto. Teach a robot to fish: Versatile imitation from one minute of demonstrations. arXiv preprint arXiv:2303.01497, 2023.

[35] P. Hansen-Estruch, I. Kostrikov, M. Janner, J. G. Kuba, and S. Levine. Idql: Implicit q-learning as an actor-critic method with diffusion policies. arXiv preprint arXiv:2304.10573, 2023.

[36] M. Heo, Y. Lee, D. Lee, and J. J. Lim. Furniturebench: Reproducible real-world benchmark for long-horizon complex manipulation. arXiv preprint arXiv:2305.12821, 2023.

[37] T. Hester, M. Vecerik, O. Pietquin, M. Lanctot, T. Schaul, B. Piot, D. Horgan, J. Quan, A. Sendonaris, I. Osband, et al. Deep q-learning from demonstrations. In Proceedings of the AAAI conference on artificial intelligence, volume 32, 2018.

[38] J. Ho, A. Jain, and P. Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 2020.

[39] J. Ho, W. Chan, C. Saharia, J. Whang, R. Gao, A. Gritsenko, D. P. Kingma, B. Poole, M. Norouzi, D. J. Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022.

[40] H. Hu, S. Mirchandani, and D. Sadigh. Imitation bootstrapped reinforcement learning. arXiv preprint arXiv:2311.02198, 2023.

[41] S. Huang, R. F. J. Dossa, C. Ye, J. Braga, D. Chakraborty, K. Mehta, and J. G. AraÃšjo. Cleanrl: Highquality single-file implementations of deep reinforcement learning algorithms. Journal of Machine Learning Research, 2022.

[42] Z. Huang, L. Yang, X. Zhou, Z. Zhang, W. Zhang, X. Zheng, J. Chen, Y. Wang, C. Bin, and W. Yang. Protein-ligand interaction prior for binding-aware 3d molecule diffusion models. In The Twelfth International Conference on Learning Representations, 2024.

[43] J. Hwangbo, J. Lee, A. Dosovitskiy, D. Bellicoso, V. Tsounis, V. Koltun, and M. Hutter. Learning agile and dynamic motor skills for legged robots. Science Robotics, 2019.

[44] M. T. Jackson, M. T. Matthews, C. Lu, B. Ellis, S. Whiteson, and J. Foerster. Policy-guided diffusion. arXiv preprint arXiv:2404.06356, 2024.

[45] M. Janner, Y. Du, J. B. Tenenbaum, and S. Levine. Planning with diffusion for flexible behavior synthesis. arXiv preprint arXiv:2205.09991, 2022.

[46] X. Jia, D. Blessing, X. Jiang, M. Reuss, A. Donat, R. Lioutikov, and G. Neumann. Towards diverse behaviors: A benchmark for imitation learning with human demonstrations. arXiv preprint arXiv:2402.14606, 2024.

[47] B. Kang, X. Ma, C. Du, T. Pang, and S. Yan. Efficient diffusion policies for offline reinforcement learning. Advances in Neural Information Processing Systems, 2024.

[48] E. Kaufmann, L. Bauersfeld, A. Loquercio, M. Müller, V. Koltun, and D. Scaramuzza. Championlevel drone racing using deep reinforcement learning. Nature, 2023.

[49] Z. Kong, W. Ping, J. Huang, K. Zhao, and B. Catanzaro. Diffwave: A versatile diffusion model for audio synthesis. arXiv preprint arXiv:2009.09761, 2020.

[50] S. Lee, Y. Wang, H. Etukuru, H. J. Kim, N. M. M. Shafiullah, and L. Pinto. Behavior generation with latent actions. arXiv preprint arXiv:2403.03181, 2024.

[51] K. Lei, Z. He, C. Lu, K. Hu, Y. Gao, and H. Xu. Uni-o4: Unifying online and offline deep reinforcement learning with multi-step on-policy optimization. arXiv preprint arXiv:2311.03351, 2023.

[52] J. Liang, S. Saxena, and O. Kroemer. Learning active task-oriented exploration policies for bridging the sim-to-real gap. arXiv preprint arXiv:2006.01952, 2020.

[53] Z. Liang, Y. Mu, M. Ding, F. Ni, M. Tomizuka, and P. Luo. Adaptdiffuser: Diffusion models as adaptive self-evolving planners. arXiv preprint arXiv:2302.01877, 2023.

[54] T. P. Lillicrap, J. J. Hunt, A. Pritzel, N. Heess, T. Erez, Y. Tassa, D. Silver, and D. Wierstra. Continuous control with deep reinforcement learning. arXiv preprint arXiv:1509.02971, 2015.

[55] Y. Lin, A. S. Wang, G. Sutanto, A. Rai, and F. Meier. Polymetis. https:// facebookresearch.github.io/fairo/polymetis/, 2021.

[56] A. Lou, C. Meng, and S. Ermon. Discrete diffusion modeling by estimating the ratios of the data distribution. stat, 2024.

[57] J. Luo, Z. Hu, C. Xu, Y. L. Tan, J. Berg, A. Sharma, S. Schaal, C. Finn, A. Gupta, and S. Levine. Serl: A software suite for sample-efficient robotic reinforcement learning. arXiv preprint arXiv:2401.16013, 2024.

[58] S. Luo, Y. Su, X. Peng, S. Wang, J. Peng, and J. Ma. Antigen-specific antibody design and optimization with diffusion-based generative models for protein structures. Advances in Neural Information Processing Systems, 2022.

[59] V. Makoviychuk, L. Wawrzyniak, Y. Guo, M. Lu, K. Storey, M. Macklin, D. Hoeller, N. Rudin, A. Allshire, A. Handa, et al. Isaac gym: High performance gpu-based physics simulation for robot learning. arXiv preprint arXiv:2108.10470, 2021.

[60] A. Mandlekar, D. Xu, J. Wong, S. Nasiriany, C. Wang, R. Kulkarni, L. Fei-Fei, S. Savarese, Y. Zhu, and R. Martín-Martín. What matters in learning from offline human demonstrations for robot manipulation. In arXiv preprint arXiv:2108.03298, 2021.

[61] A. Nair, A. Gupta, M. Dalal, and S. Levine. Awac: Accelerating online reinforcement learning with offline datasets. arXiv preprint arXiv:2006.09359, 2020.

[62] M. Nakamoto, S. Zhai, A. Singh, M. Sobol Mark, Y. Ma, C. Finn, A. Kumar, and S. Levine. Calql: Calibrated offline rl pre-training for efficient online fine-tuning. Advances in Neural Information Processing Systems, 36, 2024.

[63] A. Q. Nichol and P. Dhariwal. Improved denoising diffusion probabilistic models. In International conference on machine learning, 2021.

[64] T. Osa, J. Pajarinen, G. Neumann, J. A. Bagnell, P. Abbeel, J. Peters, et al. An algorithmic perspective on imitation learning. Foundations and Trends® in Robotics, 2018.

[65] L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 2022.

[66] T. Pearce, T. Rashid, A. Kanervisto, D. Bignell, M. Sun, R. Georgescu, S. V. Macua, S. Z. Tan, I. Momennejad, K. Hofmann, et al. Imitating human behaviour with diffusion models. arXiv preprint arXiv:2301.10677, 2023.

[67] X. B. Peng, A. Kumar, G. Zhang, and S. Levine. Advantage-weighted regression: Simple and scalable off-policy reinforcement learning. arXiv preprint arXiv:1910.00177, 2019.

[68] X. B. Peng, Z. Ma, P. Abbeel, S. Levine, and A. Kanazawa. Amp: Adversarial motion priors for stylized physics-based character control. ACM Transactions on Graphics (ToG), 2021.

[69] F. Permenter and C. Yuan. Interpreting and improving diffusion models from an optimization perspective. arXiv preprint arXiv:2306.04848, 2023.

[70] J. Peters and S. Schaal. Reinforcement learning by reward-weighted regression for operational space control. In Proceedings ofthe 24th international conference on Machine learning, 2007.

[71] D. A. Pomerleau. Alvinn: An autonomous land vehicle in a neural network. Advances in neural information processing systems, 1988.

[72] B. Poole, A. Jain, J. T. Barron, and B. Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988, 2022.

[73] M. Popova, O. Isayev, and A. Tropsha. Deep reinforcement learning for de novo drug design. Science advances, 2018.

[74] M. Psenka, A. Escontrela, P. Abbeel, and Y. Ma. Learning a diffusion model policy from rewards via q-score matching. arXiv preprint arXiv:2312.11752, 2023.

[75] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, 2021.

[76] A. Rajeswaran, V. Kumar, A. Gupta, G. Vezzani, J. Schulman, E. Todorov, and S. Levine. Learning complex dexterous manipulation with deep reinforcement learning and demonstrations. arXiv preprint arXiv:1709.10087, 2017.

[77] A. Z. Ren, H. Dai, B. Burchfiel, and A. Majumdar. AdaptSim: Task-driven simulation adaptation for sim-to-real transfer. In Proceedings ofthe Conference on Robot Learning (CoRL), 2023.

[78] M. Reuss, M. Li, X. Jia, and R. Lioutikov. Goal-conditioned imitation learning using score-based diffusion policies. arXiv preprint arXiv:2304.02532, 2023.

[79] M. Rigter, J. Yamada, and I. Posner. World models via policy-guided trajectory diffusion. arXiv preprint arXiv:2312.08533, 2023.

[80] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022.

[81] O. Ronneberger, P. Fischer, and T. Brox. U-net: Convolutional networks for biomedical image segmentation. In Medical image computing and computer-assisted intervention (MICCAI), 2015.

[82] N. Ruiz, Y. Li, V. Jampani, Y. Pritch, M. Rubinstein, and K. Aberman. Dreambooth: Fine tuning textto-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023.

[83] S. S. Sahoo, M. Arriola, Y. Schiff, A. Gokaslan, E. Marroquin, J. T. Chiu, A. Rush, and V. Kuleshov. Simple and effective masked diffusion language models. arXiv preprint arXiv:2406.07524, 2024.

[84] J. Schulman, P. Moritz, S. Levine, M. Jordan, and P. Abbeel. High-dimensional continuous control using generalized advantage estimation. arXiv preprint arXiv:1506.02438, 2015.

[85] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

[86] J. Sohl-Dickstein, E. Weiss, N. Maheswaranathan, and S. Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, 2015.

[87] J. Song, C. Meng, and S. Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020.

[88] Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020.

[89] A. Sridhar, D. Shah, C. Glossop, and S. Levine. Nomad: Goal masked diffusion policies for navigation and exploration. arXiv preprint arXiv:2310.07896, 2023.

[90] R. S. Sutton and A. G. Barto. Reinforcement learning: An introduction. MIT press, 2018.

[91] R. S. Sutton, D. McAllester, S. Singh, and Y. Mansour. Policy gradient methods for reinforcement learning with function approximation. Advances in neural information processing systems, 1999.

[92] E. Todorov, T. Erez, and Y. Tassa. Mujoco: A physics engine for model-based control. In 2012 IEEE/RSJ international conference on intelligent robots and systems, 2012.

[93] M. Torne, A. Simeonov, Z. Li, A. Chan, T. Chen, A. Gupta, and P. Agrawal. Reconciling reality through simulation: A real-to-sim-to-real approach for robust manipulation. arXiv preprint arXiv:2403.03949, 2024.

[94] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin. Attention is all you need. Advances in neural information processing systems, 2017.

[95] M. Vecerik, T. Hester, J. Scholz, F. Wang, O. Pietquin, B. Piot, N. Heess, T. Rothörl, T. Lampe, and M. Riedmiller. Leveraging demonstrations for deep reinforcement learning on robotics problems with sparse rewards. arXiv preprint arXiv:1707.08817, 2017.

[96] S. Venkatraman, S. Khaitan, R. T. Akella, J. Dolan, J. Schneider, and G. Berseth. Reasoning with latent diffusion in offline reinforcement learning. arXiv preprint arXiv:2309.06599, 2023.

[97] B. Wallace, M. Dang, R. Rafailov, L. Zhou, A. Lou, S. Purushwalkam, S. Ermon, C. Xiong, S. Joty, and N. Naik. Diffusion model alignment using direct preference optimization. arXiv preprint arXiv:2311.12908, 2023.

[98] J. Wang and E. Olson. Apriltag 2: Efficient and robust fiducial detection. In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2016.

[99] L. Wang, J. Zhao, Y. Du, E. H. Adelson, and R. Tedrake. Poco: Policy composition from and for heterogeneous robot learning. arXiv preprint arXiv:2402.02511, 2024.

[100] Z. Wang, J. J. Hunt, and M. Zhou. Diffusion policies as an expressive policy class for offline reinforcement learning. arXiv preprint arXiv:2208.06193, 2022.

[101] R. J. Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning, 1992.

[102] J. Yang, M. S. Mark, B. Vu, A. Sharma, J. Bohg, and C. Finn. Robot fine-tuning made easy: Pretraining rewards and policies for autonomous real-world reinforcement learning. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 4804–4811. IEEE, 2024.

[103] L. Yang, Z. Huang, F. Lei, Y. Zhong, Y. Yang, C. Fang, S. Wen, B. Zhou, and Z. Lin. Policy representation via diffusion probability model for reinforcement learning. arXiv preprint arXiv:2305.13122, 2023.

[104] M. Yang, Y. Du, K. Ghasemipour, J. Tompson, D. Schuurmans, and P. Abbeel. Learning interactive real-world simulators. arXiv preprint arXiv:2310.06114, 2023.

[105] Y. Ze, G. Zhang, K. Zhang, C. Hu, M. Wang, and H. Xu. 3d diffusion policy: Generalizable visuomotor policy learning via simple 3d representations. In ICRA 2024 Workshop on 3D Visual Representationsfor Robot Manipulation.

[106] T. Z. Zhao, V. Kumar, S. Levine, and C. Finn. Learning fine-grained bimanual manipulation with low-cost hardware. arXiv preprint arXiv:2304.13705, 2023.

[107] H. Zhu, A. Gupta, A. Rajeswaran, S. Levine, and V. Kumar. Dexterous manipulation with deep rein forcement learning: Efficient, general, and low-cost. In 2019 International Conference on Robotics and Automation (ICRA), pages 3651–3657. IEEE, 2019.

[108] Z. Zhu, H. Zhao, H. He, Y. Zhong, S. Zhang, Y. Yu, and W. Zhang. Diffusion models for reinforcement learning: A survey. arXiv preprint arXiv:2311.01223, 2023.

## A Extended Related Work

## A.1 RL training of robot policies with offline data

Here, we discuss related work in training robot policies using RL augmented with offline data to help RL better explore online in sparse reward settings.

One simple form is to use offline data to pre-train the policy, typically using behavior cloning, and then fine-tune the policy online. This is the approach that DPPO takes. Often, a regularization loss is applied to constrain the fine-tuned policy to stay close to the base policy, leading to natural fine-tuned behavior and often better learning [76, 93, 107]. DPPO does not apply regularization at fine-tuning as we find the onmanifold exploration helps DPPO maintain natural behavior after fine-tuning Section 5.4. Another popular approach is to learn a residual policy with RL on top of the frozen base policy [3, 34]. A closer work to ours is from Ankile et al. [6], which trains a one-step residual non-diffusion policy with on-policy RL on top of a pre-trained chunked diffusion policy. This approach has the benefit of being fully closed-loop but lacks the structured on-manifold exploration of DPPO. Another hybrid approach is from Hu et al. [40], which uses pre-trained and fine-tuned policies to sample online experiences.

Another popular line of work, instead of training a base policy using offline data, directly adds the data in the replay buffer for online, off-policy learning in a single stage [37, 61, 95]. One recent approach from Ball et al. [7], RLPD, further improves sample efficiency from previous off-policy methods incorporating, e.g., critic ensembling. Luo et al. [57] demonstrate RLPD solving real-world manipulation tasks (although generally less challenging than ones solved by DPPO).

Other approaches, including Cal-QL, build on offline RL to learn from offline data and then switch to online RL while still sampling from offline data [35, 62, 102]. Often the distributional mismatch between offline data and online policy needs to be addressed: Cal-QL proposes calibrated conservative Q-learning that learns a offline Q function that lower bounds the true value of the learned policy; Lei et al. [51] propose ensemble behavior cloning during pre-training to promote policy diversity.

## A.2 Diffusion-based RL methods

This section discusses related methods that directly train or improve diffusion-based policies with RL methods. The baselines to which we compare in Section 5.1 are discussed below as well, and are highlighted in their corresponding colors. We also refer the readers to Zhu et al. [108] for an extensive survey on diffusion models for RL.

Most previous works have focused on the offline setting with a static dataset. One line of work focuses on state trajectory planning and guiding the denoising sampling process such that the sampled actions satisfy some desired objectives. Janner et al. [45] apply classifier guidance that generates trajectories with higher predicted rewards. Ajay et al. [2] introduce classifier-free guidance that avoids learning the value of noisy states. There is another line of work that uses diffusion models as an action policy (instead of state planner) and generally applies Q-learning. DQL [100] introduces Diffusion Q-Learning that learns a stateaction critic for the final denoised actions and backpropagates the gradient from the critic through the entire Diffusion Policy (actor) denoising chain, akin to the usual Q-learning. IDQL [35], or Implicit Diffusion Qlearning, proposes learning the critic to select the actions at inference time for either training or evaluation while fitting the actor to all sampled actions. Kang et al. [47] instead propose using the critic to re-weight the sampled actions for updating the actor itself, similar to weighted regression baselines DAWR and DRWR introduced in our work. Goo and Niekum [31] similarly extract the policy in the spirit of AWR [67]. Chen et al. [16] train the critic using value iteration instead based on samples from the actor. Finally, Jackson et al. [44] explore using diffusion guidance to move offline data towards the target trajectory distribution.

We note that methods like DQL and IDQL can also be applied in the online setting. A small amount of work also focuses entirely on the online setting. DIPO [103] differs from DQL and related work in that it uses the critic to update the sampled actions (“action gradient”) instead of the actor — the actor is then fitted with updated actions from the replay buffer. QSM, or Q-Score Matching [74], suggests that optimizing the likelihood of the entire chain of denoised actions can be inefficient (contrary to our findings in the fine-tuning setting) and instead proposes learning the optimal policy by iteratively aligning the gradient of the actor (i.e., score) with the action gradient of the critic. Rigter et al. [79] proposes learning a diffusion dynamic model to generate synthetic trajectories for online training of a non-diffusion RL policy.

We note that almost all prior work in diffusion-based RL (offline or online) have relied on approximating the state-action Q function and using it to update the diffusion actor in some form — policy gradient update has been deemed challenging due to the multi-step denoising process [74, 103]. Inaccurate Q values may lead to biased updates to the actor, which can lead to training collapse as it starts with decent pre-training performance but quickly drops to zero success rate as seen in Fig. 5, also failing to recover since then due to the sparse-reward setup. While Q-learning methods generally achieve better sample efficiency when they can solve the task of interest, our focus has been largely on challenging long-horizon robot manipulation tasks where the training stability is much desired.

Distinction from the policy gradient formulation in Psenka et al. [74]. There has been a different formulation introduced in Psenka et al. [74] Sec. 3 that derives the policy gradient update for diffusion policy. The derivation is based on converting the gradient of the log likelihood of the final denoised action to the sum over log likelihood of individual denoising actions. This formulation, unlike DPPO, does not treat the multi step denoising process as a MDP. In the policy gradient update, Psenka et al. [74] sum over denoising steps and then takes expectation over environment steps, while DPPO’s update (4.2) takes expectation over both denoising and environment steps, potentially leading to better sample efficiency. Moreover, Psenka et al. [74] do not propose applying PPO updates or other modifications to diffusion, and finds such vanilla form of policy gradient update to be ineffective. We formulate DPPO independent of their work and find DPPO highly effective in fine-tuning settings while also being competitive in training from scratch (Appendix C.3).

## B Additional details of DPPO implementation

Pseudocode. The pseudocode for DPPO is presented in Algorithm 1. DPPO takes as input a diffusion policy $\pi _ { \theta }$ trained using behavior cloning loss $\mathcal { L } _ { \mathrm { B C } }$ . The policy is then fine-tuned using a PPO-style loss [85] with careful treatment of the denoising process (Section 4).

Pre-training. The diffusion policy $\pi _ { \theta }$ is pre-trained using a behavior cloning loss [38]:

$$
\mathcal {L} _ {\mathrm{BC}} (\theta) = \mathbb {E} ^ {(s _ {t}, a _ {t} ^ {0}) \sim \mathcal {D} _ {\mathrm{off}}} \big [ \| \varepsilon_ {t} - \varepsilon_ {\theta} (a _ {t} ^ {0}, s _ {t}, k) \| ^ {2} \big ],\tag{A1}
$$

where $\mathcal { D } _ { \mathrm { o f f } }$ is the offline dataset and $\varepsilon _ { \theta }$ is the policy network predicting the sampled noise added to $a _ { t } ^ { 0 }$ based on the noisy action. We use the cosine noise schedule from Nichol and Dhariwal [63].

Environment-step advantage estimation. We use Generalized Advantage Estimation (GAE) [84] with parameter λ for advantage estimation in Algorithm 1. GAE-λ approximates the advantage function using the series

$$
\hat {A} _ {\bar {t} (t, k = 0)} ^ {\lambda} = \sum_ {l = 0} ^ {\infty} (\gamma \lambda) ^ {l} \bar {\delta} _ {\bar {t} (t + l, k = 0)}, \quad \mathrm{where} \bar {\delta} _ {\bar {t} (t, k)} = \bar {R} _ {\bar {t} (t, k)} + \gamma_ {\mathrm{ENV}} V _ {\phi} (\bar {s} _ {\bar {t} (t + 1, k)}) - V _ {\phi} (\bar {s} _ {\bar {t} (t, k)}).\tag{A2}
$$

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1 DPPO
1: Pre-train diffusion policy $\pi_{\theta}$ with offline dataset $\mathcal{D}_{\text{off}}$ using BC loss $\mathcal{L}_{\text{BC}}(\theta)$ Eq. (A1).
2: Initialize value function $V_{\phi}$.
3: for iteration = 1, 2, ... do
4: Initialize rollout buffer $\mathcal{D}_{\text{itr}}$.
5: $\pi_{\theta_{\text{old}}} = \pi_{\theta}$.
6: for environment = 1, 2, ..., N in parallel do
7: Initialize state $\bar{s}_{\bar{t}(0,K)} = (s_0, a_0^K)$ in $\mathcal{M}_{\text{DP}}$.
8: for environment step t = 1, ..., T, denoising step $k = K - 1, \ldots, 0$ do
9: Sample the next denoised action $\bar{a}_{\bar{t}(t,k)} = a_t^k \sim \pi_{\theta_{\text{old}}}$.
10: if $k = 0$ then
11: Run $a_t^0$ in the environment and observe $\bar{R}_{\bar{t}(t,0)}$ and $\bar{s}_{\bar{t}(t+1,K)}$.
12: else
13: Set $\bar{R}_{\bar{t}(t,k)} = 0$ and $\bar{s}_{\bar{t}(t,k-1)} = (s_t, a_t^k)$.
14: Add $(k, \bar{s}_{\bar{t}(t,k)}, \bar{a}_{\bar{t}(t,k)}, \bar{R}_{\bar{t}(t,k)})$ to $\mathcal{D}_{\text{itr}}$.
15: Compute advantage estimates $A^{\pi_{\theta_{\text{old}}}}(s_{\bar{t}(t,k=0)}, a_{\bar{t}(t,k=0)})$ for $\mathcal{D}_{\text{itr}}$ using GAE Eq. (A2).
16: for update = 1, 2, ..., num_update do ▷ Based on replay ratio $N_{\theta}$
17: for minibatch = 1, 2, ..., B do
18: Sample $(k, \bar{s}_{\bar{t}(t,k)}, \bar{a}_{\bar{t}(t,k)}, \bar{R}_{\bar{t}(t,k)})$ and $A^{\pi_{\theta_{\text{old}}}}(s_{\bar{t}(t,k)}, a_{\bar{t}(t,k)})$ from $\mathcal{D}_{\text{itr}}$.
19: Compute denoising-discounted advantage $\hat{A}_{\bar{t}(t,k)} = \gamma_{\text{DENOISE}}^k A^{\pi_{\theta_{\text{old}}}}(s_{\bar{t}(t,0)}, a_{\bar{t}(t,0)})$.
20: Optimize $\pi_{\theta}$ using policy gradient loss $\mathcal{L}_{\theta}$ Eq. (A3).
21: Optimize $V_{\phi}$ using value loss $\mathcal{L}_{\phi}$ Eq. (A4).
22: return converged policy $\pi_{\theta}$.
</div>

Notably, GAE-λ interpolates between a one-step temporal difference $( \hat { A } _ { \bar { t } ( t , k ) } ^ { \lambda = 0 } = \bar { R } _ { \bar { t } ( t , k ) } + \gamma _ { \mathrm { E N V } } V _ { \phi } ( \bar { s } _ { \bar { t } ( t + 1 , k ) } ) -$ $V _ { \phi } \big ( \overline { { s } } _ { \bar { t } ( t , k ) } \big ) \big )$ and the Monte Carlo return of the episode relative to the baseline $\begin{array} { r } { ( \hat { A } _ { \bar { t } ( t , k ) } ^ { \lambda = 1 } = \sum _ { l = 0 } ^ { T - t } \gamma _ { \mathrm { E N V } } ^ { l } \bar { R } _ { \bar { t } ( t + l , k ) } - } \end{array}$ $V _ { \phi } \big ( \overline { { s } } _ { \bar { t } ( t , k ) } \big ) \big )$ . We refer the reader to Table A8 for additional details on GAE parameter selection and Section C.2 for ablations on the choice of advantage estimator.

Note that in Eq. (A2) we are only concerned with $k = 0 ,$ , i.e., the final denoising step, calculating the advantage for $k = 0 ~ ( \mathrm { i . e . }$ ., environment steps) but not for intermediate denoising steps. We only need to apply denoising discounting to the calculated advantages so they can be applied to each denoising step $k .$

Fine-tuning. During RL fine-tuning, we update the policy $\pi _ { \theta }$ using the clipped objective:

$$
\mathcal {L} _ {\theta} = \mathbb {E} ^ {\mathcal {D} _ {\mathrm{itr}}} \min \Big (\hat {A} ^ {\bar {\pi} _ {\theta_ {\mathrm{old}}}} (\bar {s} _ {\bar {t}}, \bar {a} _ {\bar {t}}) \frac {\bar {\pi} _ {\theta} (\bar {s} _ {\bar {t}} , \bar {a} _ {\bar {t}})}{\bar {\pi} _ {\theta_ {\mathrm{old}}} (\bar {s} _ {\bar {t}} , \bar {a} _ {\bar {t}})}, \hat {A} ^ {\pi_ {\theta_ {\mathrm{old}}}} (\bar {s} _ {\bar {t}}, \bar {a} _ {\bar {t}}) \operatorname{clip} \big (\frac {\bar {\pi} _ {\theta} (\bar {s} _ {\bar {t}} , \bar {a} _ {\bar {t}})}{\bar {\pi} _ {\theta_ {\mathrm{old}}} (\bar {s} _ {\bar {t}} , \bar {a} _ {\bar {t}})}, 1 - \varepsilon , 1 + \varepsilon \big) \Big).\tag{A3}
$$

If we choose to fine-tune only the last $K ^ { \prime }$ denoising steps, then we sample only those from ${ \mathcal { D } } _ { \mathrm { i t r } }$

Finally, we train the value function to predict the future discounted sum of rewards $( \mathrm { i . e . }$ , discounted returns):

$$
\mathcal {L} _ {\phi} = \mathbb {E} ^ {\mathcal {D} _ {\mathrm{itr}}} \left[ \| \sum_ {l = 0} ^ {T - t} \gamma_ {\mathrm{ENV}} ^ {l} \bar {R} _ {\bar {t} (t + l, k)} - V _ {\phi} (s _ {t}) \| ^ {2} \right].\tag{A4}
$$

Similar to all baselines in Appendix E.3, we denote $N _ { \theta }$ and $N _ { \phi }$ the replay ratio for the actor (diffusion policy) and the value critic in DPPO; in practice we always set $N _ { \theta } = N _ { \phi }$ . Similar to usual PPO imple mentations [41], the batch updates in an iteration terminate when the KL divergence between $\pi _ { \theta }$ and $\pi _ { \theta _ { \mathrm { o l d } } }$ reaches 1, although in practice we find this never happens.

Large batch size. Since the gradient update in DPPO involves expectation over both environment steps and denoising steps, we use a larger batch size compared to, e.g., PPO training with Gaussian policy parameterization. Roughly we use the batch size from Gaussian training times the number of the fine-tuned denoising steps; in some cases like ROBOMIMIC we also observe that a much smaller batch size (close to that of Gaussian training) can be used and significantly improves sample efficiency.

Gradient clipping ratio. We find the PPO clipping ratio, ε, can affect the training stability significantly in DPPO (as well as in Gaussian and GMM policies) especially in sparse-reward manipulation tasks. In practice we find that, a good indicator of the amount of clipping leading to optimal training efficiency, is to aim for a clipping fraction (fraction of individual samples being clipped in a batch) of 10% to 20%. For each method in different tasks, we vary ε in {.1, .01, .001} and choose the highest value that satisfies the clipping fraction target. Empirically we also find that, using a higher ε for earlier denoising steps in DPPO further improves training stability in manipulation tasks. Denote $\varepsilon _ { k }$ the clipping value at denoising step k, and in practice we set $\varepsilon _ { k = ( K - 1 ) } = 0 . 1 \varepsilon _ { k = 0 }$ , and it follows an exponential schedule among intermediate k.

## C Additional experimental results

## C.1 Comparing to demo-augmented RL baselines using diffusion policy instead

In Section 5.2 we compare DPPO with other demo-augmented RL methods, namely, RLPD, Cal-QL, and IBRL — DPPO uses diffusion policy while the baselines use Gaussian policy. Here we experiment with using diffusion policies for the baselines and the results are shown in Fig. A1. We use either action chunk size $T _ { a } = 1 \ : \mathrm { o r } \ : T _ { a } = 4 .$ We see similar results as in Fig. 6 using Gaussian policies that RLPD and Cal-QL fails to solve the task at all. We believe that the worse performance of Cal-QL is due to the offline RL objective (based on learning the state-action Q function) making learning precise continuous actions needed in ROBOMIMIC tasks very difficult, regardless of the policy parameterization, which corroborates our original finding in Section 5.1 when comparing DPPO to Q-learning-based diffusion RL methods. Compared to RLPD that trains with the SAC objective and expert data in the replay buffer, IBRL, using BC pre-training, is able to learn a base policy more effectively and uses it for online data collection. DPPO benefits from directly fine-tuning the pre-trained policy (instead of training a new one using experiences from the pretrained policy), and achieves similar or better sample efficiency before 1e6 steps compared to IBRL, and converges to ∼100% success rates unlike IBRL saturates at lower levels (not shown).

![](images/c8d9cf788a398b17344e6ae9806dc59341331c1d7672105000c436da59183efc.jpg)  
Figure A1: Using diffusion policy for other demo-augmented RL methods. Results are averaged over three seeds.

## C.2 Ablation studies on design decisions in DPPO

1. Choice of advantage estimator. In Section 4.3 we demonstrate how to efficiently estimate the advantage used in PPO updates by learning $\tilde { V } ( s _ { t } )$ that only depends on the environment state; the advantage used in DPPO is formally

$$
\hat {A} = \gamma_ {\mathrm{DENOISE}} ^ {k} (\bar {r} (\bar {s} _ {\bar {t}}, \bar {a} _ {\bar {t}}) - \tilde {V} (s _ {t})).
$$

We now compare this choice with learning the value of the full state $\bar { s } _ { \bar { t } ( t , 0 ) }$ that includes environment state $s _ { t }$ and denoised action $a _ { t } ^ { k = 1 }$ . We additionally compare with the state-action Q-function estimator used in Psenka et al. $[ 7 4 ] ^ { 8 } , \tilde { Q } ( s _ { t } , a _ { t } ^ { k = 0 } )$ , that does not directly use the rollout reward r¯ in the advantage.

Fig. A2 shows the fine-tuning results in Hopper- $- \mathtt { v } 2$ and HalfCheetah- $- \mathtt { v } 2$ from GYM, and Can and Square from ROBOMIMIC. On the simpler Hopper-v2, we observe that the two baselines, both estimating the value of some action, achieves higher reward during fine-tuning than DPPO’s choice. However, in the more challenging tasks, the environment-state-only advantage used in DPPO consistently leads to the most improved performance. We believe estimating the accurate value of applying a continuous and highdimensional action can be challenging, and this is exacerbated by the high stochasticity of diffusion-based policies and the action chunk size. The results here corroborate the findings in Section 5.1 that off-policy Q-learning methods can perform well in Hopper- $- \mathtt { v } 2$ and Walker2D-v2, but often exhibit training instability in manipulation tasks from ROBOMIMIC.

![](images/02c2417ddc9bfdb88564dbf2b147f37b775ff2a6bf6b983648a3d3976992e84f.jpg)

![](images/5a82a4452c4dbadee0863bf794c363b992d1f0f0ebec389e87fb4ed9e370ea55.jpg)

![](images/e7a600ae21afb1af7584e5c7e9fbe3fce00ddbcc24b6bd0f96c5e10b1a71213b.jpg)

![](images/b1b57a01b607ddc78e4d9e7300ed51b4d624ee9f717d12dfdc73ae0a1fc1a5e1.jpg)  
Figure A2: Choice of advantage estimator. Results are averaged over five seeds in Hopper-v2 and HalfCheetah-v2 and three seeds in Can and Square.

Denoising discount factor. We further examine how γ<sub>DENOISE</sub> in the DPPO advantage estimator affects fine-tuning. Using a smaller value (i.e., more discount) has the effect of downweighting the contribution of earlier denoising steps in the policy gradient. Fig. A3 shows the fine-tuning results in the same four tasks with varying γ<sub>DENOISE</sub> $\in [ 0 . 5 , 0 . 8 , 0 . 9 , 1 ]$ . We find in Hopper- $- \mathtt { v } 2$ and HalfCheetah- ${ \cdot } \mathrm { v } 2 \ \gamma _ { \mathrm { D E N O I S E } } = 0 . 8$ leads to better efficiency while smaller γ<sub>DENOISE</sub> = 0.5 slows training. The value does not affect training noticeably in Can. In Square the smaller $\gamma _ { \mathrm { { D E N O I S E } } } = 0 . 5$ works slightly better. Overall in manipulation tasks, DPPO training seems relatively robust to this choice.

2. Choice of diffusion noise schedule. As introduced in Section 4.3, we find it helpful to clip the diffusion noise $\sigma _ { k }$ to a higher minimum value $\sigma _ { \operatorname* { m i n } } ^ { \mathrm { e x p } }$ to ensure sufficient exploration. In Figure A4, we perform analysis on varying $\sigma _ { \mathrm { m i n } } ^ { \mathrm { e x p } } \in \{ . 0 0 1 , . 0 1 , . 1 , . 2 \}$ (keeping $\sigma _ { \mathrm { m i n } } ^ { \mathrm { p r o b } } = . 1$ to evaluate likelihoods). Although in Can the choice of $\sigma _ { \mathrm { m i n } } ^ { \mathrm { e x p } }$ does not affect the fine-tuning performance, in Square a higher $\sigma _ { \mathrm { m i n } } ^ { \mathrm { e x p } } = 0 . 1$ is required to prevent the policy from collapsing. We conjecture that this is due to limited exploration causing policy overoptimizing the collected samples that exhibit limited state-action coverage. We also visualize the trajectories at the beginning of fine-tuning in Avoid task from D3IL. With higher $\sigma _ { \operatorname* { m i n } } ^ { \mathrm { e x p } }$ , the trajectories still remain near the two modes of the pre-training data but exhibit a higher coverage in the state space — we believe this additional coverage leads to better exploration. Anecdotally, we find terminating the denoising process early can also provide exploration noise and lead to comparable results, but it requires a more involved implementation around the denoising MDP.

![](images/ec2eceb11c0dd4ca420874920ba6d66d283b6395fb9e6c7e10095b4dbc7ad9d8.jpg)  
Figure A3: Choice of denoising discount factor. Results are averaged over five seeds in Hopper-v2 and HalfCheetah-v2 and three seeds in Can and Square.

![](images/719d8fa5d6679480a054894753c1d13a75ee5149309b8be14666827c970c0e63.jpg)

![](images/1b28b734c3d220b7ec6b6dcb3340eac2ea4378ad5c6934359c441c66c95563c2.jpg)  
Figure A4: Choice of minimum diffusion noise. Results are averaged over three seeds. Note in Left, with higher minimum noise level, the sampled trajectories exhibit wider coverage at the two modes but still maintain the overall structure.

3. Choice of the number of fine-tuned denoising steps. We examine how the number of fine-tuned denoising steps in DPPO, $K ^ { \prime } { } .$ , affects the fine-tune performance and wall-clock time in Fig. A5. We show the curves of individual runs (three for each $K ^ { \prime } )$ instead of the average as their wall-clock times (X-axis) are not perfectly aligned. Generally, fine-tuning too few denoising steps (e.g., 3) can lead to subpar asymptotic performance and slower convergence especially in Can. Fine-tuning 10 steps leads to the overall best efficiency. Similar results are also shown in Fig. A8 with Avoid task. Lastly, we note that the GPU memory usage scales linearly with $K ^ { \prime }$

We note that the findings here mostly correlate with those from varying the denoising discount factor, γ<sub>DENOISE</sub>. Discounting the earlier denoising steps in the policy gradient can be considered as a soft version of hard limiting the number of fine-tuned denoising steps. Depending on the amount of fine-tuning needed from the pre-trained action distribution, one can flexibly adjust γ<sub>DENOISE</sub> and $K ^ { \prime }$ to achieve the best efficiency.

## C.3 Effect of expert data

We investigate the effect of the amount of pre-training expert data on fine-tuning performance. In Fig. A6 we compare DPPO and Gaussian in Hopper-v2, Square, and One-leg task from FURNITURE-BENCH, using varying numbers of expert data (episodes) denoted in the figure. Overall, we find DPPO can better leverage the pre-training data and fine-tune to high success rates. Notably, DPPO obtains non-trivial performance (60% success rate) on One-leg from only 10 episode of demonstrations.

DPPO-MLP, 10 Gaussian-MLP, 10 DPPO-MLP, 20 Gaussian-MLP, 20 DPPO-MLP, 30 Gaussian-MLP, 30 DPPO-MLP, 50 Gaussian-MLP, 50

![](images/d8c09864e2910198444187da0707617d5c8ac4a7f0d6563c3726d3fcedffa587.jpg)

![](images/2202a4b54323a52973f1e11cb540c4709c5957ce1b057c47768677acf09e7460.jpg)

![](images/9b339588593e61386a4bf9ae3621aaeef35452c239e86073342e75c398a2e37b.jpg)

![](images/ca3e5528e97f40f838ae0d6b372b4d5b8c9b3b5852205098623b13ab62a6aff2.jpg)

Figure A5: Choice of number of fine-tuned denoising steps, $K ^ { \prime } .$ . Individual runs are shown. The curves are smoothed using a Savitzky–Golay filter.  
![](images/78d369346f9dabb59e0ea1d3a6b4419dc39154bc848634b2ebc4e9c5796494cd.jpg)  
DPPO-MLP, 0 Gaussian-MLP, 0 DPPO-MLP. 20 Gaussian-MLP. 20 DPPO-MLP, 50 Gaussian-MLP, 50 DPPO-MLP, 200 Gaussian-MLP, 200

![](images/611c508684eb0be620386aa5df9d83bd1e187fcfdbfe618d385202fdf4fcf851.jpg)

![](images/605c5e1f353dba2598f737c7b801f3f03af0822fcf99a4574d37719080e7d8bd.jpg)  
Figure A6: Varying the number of expert demonstrations. The numbers in the legends indicates the number of episodes used in pre-training.

Training from scratch. In Fig. A7 we compare DPPO (10 denoising steps) and Gaussian trained from scratch (no pre-training on expert data) in the three OpenAI GYM tasks. As using larger action chunk sizes $T _ { a }$ leads to poor from-scratch training shown in Fig. A6, we focus on single-action chunks $T _ { a } = 1$ (and $T _ { p } = 1 )$ as is typical in RL benchmarking. Though we find Gaussian trains faster than DPPO (expected since DPPO solves an MDP with longer effective horizon), DPPO still attains reasonable final performance. However, due to the multi-step (10) denoising sampling, DPPO takes about 6× wall-clock time compared to Gaussian. We hope that future work will explore how to design the training curriculum of denoising steps for the best balance of training performance and wall-clock efficiency.

![](images/d492b09e74556c1140fb00e49eb60a5932e885599ad749ad9024a646aa0aa136.jpg)

![](images/644849744b9724db13557ea4e5478bd1d580b295c6ffd622657e933fd8b9623a.jpg)

![](images/d966bc50f1771624b79ca45cff65dc2f9247e5fd7a56bcb8b6e83eb0493063ad.jpg)  
Figure A7: No expert data / pre-training with GYM tasks. Results are averaged over five seeds.

## C.4 Comparing to other policy parameterizations in Avoid

Figure A8 depicts the performance of various parameterizations of DPPO (with differing numbers of finetuned denoising steps, K<sup>′</sup>) to Gaussian and GMM baselines. We study the Avoid task from D3IL, after pre-training with the data from M1, M2, M3 as described in Section 6. We find that, for $K ^ { \prime } \in \{ 1 5 , 2 0 \}$ DPPO attains the highest performance of all methods and trains the quickest in terms of environment steps; on M1, M2, it appears to attain the greatest terminal performance as well. $K ^ { \prime } = 1 0$ appears slightly better than, but roughly comparable to, the Gaussian baseline, with GMM and $K ^ { \prime } < 1 0$ performing less strongly.

![](images/9d2fdd9b72054120c3f30c89387d59a5064840a3e890c5172455ad892fcca6c0.jpg)

![](images/32192807d61138c306738194c436aca139704204108787a68d1861ee7790c9fd.jpg)

![](images/c73f43d75def110bb5653b99e5055fce535e802029e8d2dfbaf9162d2b259fca.jpg)  
Figure A8: Fine-tuning performance (averaged over five seeds, standard deviation not shown) after pretraining with M1, M2, and M3 in Avoid task from D3IL. DPPO $( K = 2 0 )$ , Gaussian, and GMM policies are compared. We also sweep the number of fine-tuned denoising steps $K ^ { \prime }$ in DPPO.

## C.5 Comparing to other policy parameterizations in the easier tasks from ROBOMIMIC

Figure A9 compares the performance of DPPO to Gaussian and GMM baslines, across a variety of architectures, and with state and pixel inputs, in Lift and Can environments in the ROBOMIMIC suite. Compared to the Square and Transport (results shown in Section 5), these environments are considered to be “eas-$\mathrm { i e r } ^ { \prime }$ , and this is reflected in the greater performance of DPPO and Gaussian baselines (GMM still exhibits subpar performance). Nonetheless, DPPO still achieves similar or even better sample efficiency compared to Gaussian baseline.

![](images/09dfbed3cbe04ae67da1e0e885f70250ae21eb739135b4c0a6453a0a2ab1377a.jpg)  
Figure A9: Comparing to other policy parameterizations in the easier Lift and Can tasks from ROBOMIMIC, with state (left) or pixel (right) observation. Results are averaged over three seeds.

## C.6 Comparing to policy gradient using exact likelihood of Diffusion Policy

Here we experiment another novel method (which, to our knowledge, has not been explicitly studied in any previous work) for performing policy gradient with diffusion-based policies. Although diffusion model does not directly model the action likelihood, $p _ { \theta } ( a _ { 0 } | s )$ , there have been ways to estimate the value, e.g., by solving the probability flow ODE that implements DDPM [88]. We refer the readers to Appendix. D in Song et al. [88] for a comprehensive exposition. We follow the official open-source code from Song et al.<sup>9</sup>, and implement policy gradient (single-level MDP) that uses the exact action likelihood $\pi _ { \boldsymbol { \theta } } ( a _ { t } | \boldsymbol { s } _ { t } )$ (3.1).

Fig. A10 shows the comparison between DPPO and diffusion policy gradient using exact likelihood estimate. Exact policy gradient improves the base policy in Hopper-v2 but does not outperform DPPO. It also requires more runtime and GPU memory as it backpropagates through the ODE. In the more challenging Can its success rate drops to zero. Moreover, policy gradient with exact likelihood does not offer the flexibility of fine-tuning fewer-than-K denoising steps or discounting the early denoising steps that DPPO offers, which have shown in Appendix C.2 to often improve fine-tuning efficiency.

![](images/e8e3db142a73c21a61382a46e1b0026f69967394d21964071fe0049efb03c569.jpg)

![](images/e5b58af802fafc420e757d872a9cc909a6098021939ad405e9baebb7412edf83.jpg)

![](images/b714cc24d3a27b54d537af8232e072ea9c8a26ebcd6e1afb55d082039f57a434.jpg)  
Figure A10: Comparing to diffusion policy gradient with exact action likelihood. Results are averaged over five seeds in Hopper-v2 and HalfCheetah-v2, and three seeds in Can.

## C.7 Ablating Structured Exploration in DPPO

Here we provide additional evidence on how structured exploration of DPPO (Section 6) aids RL finetuning. While Fig. A8 compares DPPO with Gaussian and GMM policies and shows DPPO trajectories achieve wide coverage and stay near the expert data manifold, in Fig. A11 we ablate such structured exploration within DPPO. We use DDIM [87] such that actions can be sampled deterministically — this allows us to sample trajectories without adding any noise to intermediate denoising steps but only to the final denoised action $( k = 0 )$ , and compare that to DPPO with noise added to all denoising steps. In both cases, we consider the minimum noise level $\sigma _ { \operatorname* { m i n } } ^ { \mathrm { e x p } }$ of 0.05 and 0.1. We see in Fig. A11 that with higher noise level, DPPO trajectories cover the expert data modes well without exploring aggressively into new modes, while in the case of only adding noise to the final step, the trajectories become less structured especially in M3.

Then we run both exploration schemes in Can and Square from ROBOMIMIC, and Fig. A3 right shows the original DPPO setup achieves faster convergence than when noise is only added to the final step. This result, on top of results from Section 5.3 showing DPPO achiving better sample efficiency than Gaussian and GMM policies, showcases the benefit of structured exploration in fine-tuning.

## D Reporting of Wall-Clock Times

Comparing to other diffusion-based RL algorithms Section 5.1. Table A1 and Table A2 shows the the wall-clock time used in each OpenAI GYM task and ROBOMIMIC task. In GYM tasks, on average DPPO trains 41%, 37%, and 12% faster than DAWR, DIPO, and DQL, respectively, which all require a significant amount of gradient updates per sample to train stably. QSM, DRWR, and IDQL trains 43%, 33%, and 7% faster than DPPO, respectively. ROBOMIMIC tasks are more expensive to simulate, especially with Transport task, and thus the wall-clock difference is smaller among the different methods. All methods use comparable time except for DIPO that uses slightly more on average.

![](images/49bea935b456646024c4dcac9ffaa403fce63afa7cb30e58a3251105833de3c0.jpg)

![](images/edf0bd56bcd33a4d27f31a61b836c871741bb5b4beee4c9928fe0caac4246203.jpg)

Figure A11: Ablating Structured Exploration in DPPO. (Left) Sampled trajectories with noise added to all denoising steps vs. only to the last step k = 0 in Avoid. (Right) Results are averaged over three seeds in Can and Square.

<table><tr><td rowspan="2">Method</td><td colspan="3">Task</td></tr><tr><td>Hopper-v2</td><td>Walker2D-v2</td><td>HalfCheetah-v2</td></tr><tr><td>DRWR</td><td>11.3</td><td>12.7</td><td>10.4</td></tr><tr><td>DAWR</td><td>30.4</td><td>30.7</td><td>27.1</td></tr><tr><td>DIPO</td><td>27.8</td><td>27.9</td><td>26.0</td></tr><tr><td>IDQL</td><td>16.3</td><td>16.1</td><td>15.5</td></tr><tr><td>DQL</td><td>20.5</td><td>20.5</td><td>17.6</td></tr><tr><td>QSM</td><td>9.6</td><td>9.9</td><td>9.7</td></tr><tr><td>DPPO</td><td>16.6</td><td>18.3</td><td>16.8</td></tr></table>

Table A1: Wall-clock time in seconds for a single training iteration in OpenAI GYM tasks when comparing diffusion-based RL algorithms. Each iteration involves 500 environment timesteps in each of the 40 parallelized environments running on 40 CPU threads and a NVIDIA RTX 2080 GPU (20000 steps total).

<table><tr><td rowspan="2">Method</td><td colspan="4">Task</td></tr><tr><td>Lift</td><td>Can</td><td>Square</td><td>Transport</td></tr><tr><td>DRWR</td><td>32.5</td><td>39.5</td><td>59.8</td><td>346.1</td></tr><tr><td>DAWR</td><td>38.6</td><td>46.0</td><td>70.5</td><td>354.3</td></tr><tr><td>DIPO</td><td>43.9</td><td>51.6</td><td>73.3</td><td>359.7</td></tr><tr><td>IDQL</td><td>33.8</td><td>41.7</td><td>63.7</td><td>349.9</td></tr><tr><td>DQL</td><td>36.9</td><td>44.4</td><td>68.5</td><td>353.5</td></tr><tr><td>QSM</td><td>31.8</td><td>44.5</td><td>68.7</td><td>322.5</td></tr><tr><td>DPPO</td><td>35.2</td><td>42.0</td><td>65.6</td><td>350.3</td></tr></table>

Table A2: Wall-clock time in seconds for a single training iteration in ROBOMIMIC tasks with state input when comparing diffusion-based RL algorithms. Each iteration involves 4 episodes (1200 environment timesteps for Lift and Can, 1600 for Square, and 3200 for Transport) from each of the 50 parallelized environments running on 50 CPU threads and a NVIDIA L40 GPU (60000, 80000, 160000 steps).

Comparing to other policy parameterizations and architecture Section 5.3 and Section 5.4. Table A3 and Table A4 shows the wall-clock time used in fine-tuning in each ROBOMIMIC task with state or pixel input, respectively. Gaussian and GMM use similar times and Transformer is slightly more expensive than MLP. On average with state input, DPPO-MLP trains 24%, 21%, 24%, and 22% slower than baselines due to the more expensive diffusion sampling. DPPO-UNet requires more time with the extensive use of convolutional and normalization layers and trains on average 49% slower than DPPO-MLP. On average with pixel input, DPPO-ViT-MLP trains 14% slower than Gaussian-ViT-MLP — the difference is smaller than the state input case as the rendering in simulation can be expensive. Table A5 shows the wall-clock time used in FURNITURE-BENCH tasks. DPPO-UNet trains 20% slower than Gaussian-MLP on average.

<table><tr><td rowspan="2">Method</td><td colspan="4">Task</td></tr><tr><td>Lift</td><td>Can</td><td>Square</td><td>Transport</td></tr><tr><td>Gaussian-MLP</td><td>27.7</td><td>35.7</td><td>56.2</td><td>255.6</td></tr><tr><td>Gaussian-Transformer</td><td>29.8</td><td>37.1</td><td>57.8</td><td>266.1</td></tr><tr><td>GMM-MLP</td><td>28.0</td><td>36.2</td><td>55.2</td><td>254.5</td></tr><tr><td>GMM-Transformer</td><td>29.5</td><td>37.4</td><td>58.1</td><td>260.2</td></tr><tr><td>DPPO-MLP</td><td>35.6</td><td>43.3</td><td>65.0</td><td>350.5</td></tr><tr><td>DPPO-UNet</td><td>83.6</td><td>92.7</td><td>130.4</td><td>431.1</td></tr></table>

Table A3: Wall-clock time in seconds for a single training iteration in ROBOMIMIC tasks with state input when comparing policy parameterizations. Each iteration involves 4 episodes (1200 environment timesteps for Lift and Can, 1600 for Square, and 3200 for Transport) from each of the 50 parallelized environments running on 50 CPU threads and a NVIDIA L40 GPU (60000, 80000, 160000 steps).

<table><tr><td rowspan="2">Method</td><td colspan="4">Task</td></tr><tr><td>Lift</td><td>Can</td><td>Square</td><td>Transport</td></tr><tr><td>Gaussian-ViT-MLP</td><td>153.6</td><td>173.1</td><td>277.0</td><td>770.0</td></tr><tr><td>DPPO-ViT-MLP</td><td>194.9</td><td>202.5</td><td>328.5</td><td>871.3</td></tr></table>

Table A4: Wall-clock time in seconds for a single training iteration in ROBOMIMIC tasks with pixel input when comparing policy parameterizations. Each iteration involves 4 episodes (1200 environment timesteps for Lift and Can, 1600 for Square, and 3200 for Transport) from each of the 50 parallelized environments running on 50 CPU threads and a NVIDIA L40 GPU (60000, 80000, 160000 steps).

<table><tr><td rowspan="2">Method</td><td colspan="3">Task</td></tr><tr><td>One-leg</td><td>Lamp</td><td>Round-table</td></tr><tr><td>Gaussian-MLP</td><td>101.8</td><td>202.8</td><td>168.7</td></tr><tr><td>DPPO-UNet</td><td>148.4</td><td>258.2</td><td>188.6</td></tr></table>

Table A5: Wall-clock time in seconds for a single training iteration in FURNITURE-BENCH tasks when comparing policy parameterizations. Each iteration involves 1 episodes (700 environment timesteps for One-leg, and 1000 for Lamp and Round-table) from each of the 1000 parallelized environments running on a NVIDIA L40 GPU (700000, 1000000, 1000000 steps).

## E Additional Experimental Details

<table><tr><td colspan="2">Task / Dataset</td><td>Obs dim - State</td><td>Obs dim - Pixel</td><td>Act dim</td><td>T</td><td>Sparse reward ?</td></tr><tr><td rowspan="3">GYM</td><td>Hopper-v2</td><td>11</td><td>-</td><td>3</td><td>1000</td><td>No</td></tr><tr><td>Walker2D-v2</td><td>17</td><td>-</td><td>6</td><td>1000</td><td>No</td></tr><tr><td>HalfCheetah-v2</td><td>17</td><td>-</td><td>6</td><td>1000</td><td>No</td></tr><tr><td rowspan="3">FRANKA-KITCHEN</td><td>Kitchen-Complete-v0</td><td>60</td><td>-</td><td>9</td><td>280</td><td>Yes</td></tr><tr><td>Kitchen-Partial-v0</td><td>60</td><td>-</td><td>9</td><td>280</td><td>Yes</td></tr><tr><td>Kitchen-Mixed-v0</td><td>60</td><td>-</td><td>9</td><td>280</td><td>Yes</td></tr><tr><td rowspan="4">ROBOMIMIC, state input</td><td>Lift</td><td>19</td><td>-</td><td>7</td><td>300</td><td>Yes</td></tr><tr><td>Can</td><td>23</td><td>-</td><td>7</td><td>300</td><td>Yes</td></tr><tr><td>Square</td><td>23</td><td>-</td><td>7</td><td>400</td><td>Yes</td></tr><tr><td>Transport</td><td>59</td><td>-</td><td>14</td><td>800</td><td>Yes</td></tr><tr><td rowspan="4">ROBOMIMIC, pixel input</td><td>Lift</td><td>9</td><td>96×96</td><td>7</td><td>300</td><td>Yes</td></tr><tr><td>Can</td><td>9</td><td>96×96</td><td>7</td><td>300</td><td>Yes</td></tr><tr><td>Square</td><td>9</td><td>96×96</td><td>7</td><td>400</td><td>Yes</td></tr><tr><td>Transport</td><td>18</td><td>2×96×96</td><td>14</td><td>800</td><td>Yes</td></tr><tr><td rowspan="3">FURNITURE-BENCH</td><td>One-leg</td><td>58</td><td>-</td><td>10</td><td>700</td><td>Yes</td></tr><tr><td>Lamp</td><td>44</td><td>-</td><td>10</td><td>1000</td><td>Yes</td></tr><tr><td>Round-table</td><td>44</td><td>-</td><td>10</td><td>1000</td><td>Yes</td></tr><tr><td rowspan="3">D3IL</td><td>M1</td><td>4</td><td>-</td><td>2</td><td>100</td><td>Yes</td></tr><tr><td>M2</td><td>4</td><td>-</td><td>2</td><td>100</td><td>Yes</td></tr><tr><td>M3</td><td>4</td><td>-</td><td>2</td><td>100</td><td>Yes</td></tr></table>

Table A6: Comparison of the different tasks considered. “Obs dim - State”: dimension of the state observation input. “Obs dim - State”: dimension of the pixel observation input. “Act dim - State”: dimension of the action space. T: maximum number of steps in an episode. “Sparse reward ?”: whether sparse reward is used in training instead of dense reward.

## E.1 Details of policy architectures used in all experiments

MLP. For most of the experiments, we use a Multi-layer Perceptron (MLP) with two-layer residual connection as the policy head. For diffusion-based policies, we also use a small MLP encoder for the state input and another small MLP with sinusoidal positional encoding for the denoising timestep input. Their output features are then concatenated before being fed into the MLP head. Diffusion Policy, proposed by Chi et al. [20], does not use MLP as the diffusion architecture, but we find it delivers comparable (or even better) pre-training performance compared to UNet.

Transformer. For comparing to other policy parameterizations in Section 5.3, we also consider Transformer as the policy architecture for the Gaussian and GMM baselines. We consider decoder only. No dropout is used. A learned positional embedding for the action chunk is the sequence into the decoder.

UNet. For comparing to other policy parameterizations in Section 5.3, we also consider UNet [81] as a possible architecture for DP. We follow the implementation from Chi et al. [20] that uses sinusoidal positional encoding for the denoising timestep input, except for using a larger MLP encoder for the observation input in each convolutional block. We find this modification helpful in more challenging tasks.

ViT. For pixel-based experiments in Section 5.3 we use Vision-Transformer(ViT)-based image encoder introduced by Hu et al. [40] before an MLP head. Proprioception input is appended to each channel of the image patches. We also follow [40] and use a learned spatial embedding for the ViT output to greatly reduce the number of features, which are then fed into the downstream MLP head.

## E.2 Additional details of GYM tasks and training in Section 5.1

Pre-training. The observations and actions are normalized to [0, 1] using min/max statistics from the pretraining dataset. For all three tasks the policy is trained for 3000 epochs with batch size 128, learning rate of 1e-3 decayed to 1e-4 with a cosine schedule, and weight decay of 1e-6. Exponential Moving Average (EMA) is applied with a decay rate of 0.995.

Fine-tuning. All methods from Section 5.1 use the same pre-trained policy. Fine-tuning is done using online experiences sampled from 40 parallelized MuJoCo environments [92]. Reward curves shown in Fig. 5 are evaluated by running fine-tuned policies with $\sigma _ { \mathrm { m i n } } ^ { \mathrm { e x p } } = 0 . 0 0 1 \ : \mathrm { ( i . e . }$ , without extra noise) for 40 episodes. Each episode terminates if the default conditions are met or the episode reaches 1000 timesteps. Detailed hyperparameters are listed in Table A7.

## E.3 Descriptions of diffusion-based RL algorithm baselines in Section 5.1

DRWR: This is a customized reward-weighted regression (RWR) algorithm [70] that fine-tunes a pretrained DP with a supervised objective with higher weights on actions that lead to higher reward-to-go r.

The reward is scaled with $\beta$ and the exponentiated weight is clipped at $w _ { \mathrm { m a x } }$ . The policy is updated with experiences collected with the current policy (no buffer for data from previous iteration) and a replay ratio of $N _ { \theta }$ . No critic is learned.

$$
\mathcal {L} _ {\theta} = \mathbb {E} ^ {\bar {\pi} _ {\theta}, \varepsilon_ {t}} \Big [ \min (e ^ {\beta r _ {t}}, w _ {\max}) \| \varepsilon_ {t} - \varepsilon_ {\theta} (a _ {t} ^ {0}, s _ {t}, k) \| ^ {2} \Big ].
$$

DAWR: This is a customized advantage-weighted regression (AWR) algorithm [67] that builds on DRWR but uses TD-bootstrapped [90] advantage estimation instead of the higher-variance reward-to-go for better training stability and efficiency. DAWR (and DRWR) can be seen as approximately optimizing (4.2) with a Kullback–Leibler (KL) divergence constraint on the policy [9, 67].

The advantage is scaled with $\beta$ and the exponentiated weight is clipped at $w _ { \mathrm { m a x } }$ . Unlike DRWR, we follow [67] and trains the actor in an off-policy manner: recent experiences are saved in a replay buffer $\mathcal { D } _ { \mathbf { \lambda } }$ and the actor is updated with a replay ratio of $N _ { \theta }$

$$
\mathcal {L} _ {\theta} = \mathbb {E} ^ {\mathcal {D}, \varepsilon_ {t}} \big [ \min (e ^ {\beta \hat {A} _ {\phi} (s _ {t}, a _ {t} ^ {0})}, w _ {\mathrm{max}}) \| \varepsilon_ {t} - \varepsilon_ {\theta} (a _ {t} ^ {0}, s _ {t}, k) \| ^ {2} \big ].
$$

The critic is updated less frequently (we find diffusion models need many gradient updates to fit the actions) with a replay ratio of $N _ { \phi }$

$$
\mathcal {L} _ {\phi} = \mathbb {E} ^ {\mathcal {D}} \big [ \| \hat {A} _ {\phi} (s _ {t}, a _ {t} ^ {0}) - A (s _ {t}, a _ {t} ^ {0}) \| ^ {2} \big ],
$$

where A is calculated using TD(λ), with λ as $\lambda _ { \mathrm { D A W R } }$ and the discount factor $\gamma _ { \mathrm { E N V } }$

DIPO [103]: This baseline applies “action gradient” that uses a learned state-action Q function to update the actions saved in the replay buffer, and then has DP fitting on them without weighting.

Similar to DAWR, recent experiences are saved in a replay buffer $\mathcal { D } .$ The actions $( k = 0 )$ in the buffer are updated for M iterations with learning rate α .

$$
a _ {t} ^ {m + 1, k = 0} = a _ {t} ^ {m, k = 0} + \alpha_ {\mathrm{DIPO}} \nabla_ {\phi} \hat {Q} _ {\phi} (s _ {t}, a _ {t} ^ {m, k = 0}), m = 0, \ldots , M _ {\mathrm{DIPO}} - 1.
$$

The actor is then updated with a replay ratio of $N _ { \theta }$ .

$$
\mathcal {L} _ {\theta} = \mathbb {E} ^ {\mathcal {D}} \big [ \| \varepsilon_ {t} - \varepsilon_ {\theta} (a _ {t} ^ {M _ {\mathrm{DIPO}}, k = 0}, s _ {t}, k) \| ^ {2} \big ].
$$

The critic is trained to minimize the Bellman residual with a replay ratio of $N _ { \phi }$ . Double Q-learning is also applied.

$$
\mathcal {L} _ {\phi} = \mathbb {E} ^ {\mathcal {D}} \big [ \| (R _ {t} + \gamma_ {\mathrm{ENV}} \hat {Q} _ {\phi} (s _ {t + 1}, \bar {\pi} _ {\theta} (a _ {t + 1} ^ {k = 0} | s _ {t + 1})) - \hat {Q} _ {\phi} (s _ {t}, a _ {t} ^ {m = 0, k = 0}) \| ^ {2} \big ]
$$

IDQL [35]: This baseline learns a state-action Q function and state V function to choose among the sampled actions from DP. DP fits on new samples without weighting.

Again recent experiences are saved in a replay buffer D. The state value function is updated to match the expected Q value with an expectile loss, with a replay ratio of $N _ { \psi }$

$$
\mathcal {L} _ {\psi} = \mathbb {E} ^ {\mathcal {D}} \big [ | \tau_ {\mathrm{IDQL}} - \mathbb {1} (\hat {Q} _ {\phi} (s _ {t}, a _ {t} ^ {0}) <   \hat {V} _ {\psi} ^ {2} (s _ {t})) | \big ].
$$

The value function is used to update the Q function with a replay ratio of $N _ { \phi }$ .

$$
\mathcal {L} _ {\phi} = \mathbb {E} ^ {\mathcal {D}} \big [ \| (R _ {t} + \gamma_ {\mathrm{ENV}} \hat {V} _ {\psi} (s _ {t + 1}) - \hat {Q} _ {\phi} (s _ {t}, a _ {t} ^ {0}) \| ^ {2} \big ].
$$

The actor fits all sampled experiences without weighting, with a replay ratio of $N _ { \theta }$

$$
\mathcal {L} _ {\theta} = \mathbb {E} ^ {\mathcal {D}} \big [ \| \varepsilon_ {t} - \varepsilon_ {\theta} (a _ {t} ^ {0}, s _ {t}, k) \| ^ {2} \big ].
$$

At inference time, $M _ { \mathrm { I D Q I } }$ actions are sampled from the actor. For training, Boltzmann exploration is applied based on the difference between $Q$ value of the sampled actions and and the V value at the current state. For evaluation, the greedy action under Q is chosen.

DQL [100]: This baseline learns a state-action Q function and backpropagates the gradient from the critic through the entire actor (with multiple denoising steps), akin to the usual Q-learning.

Again recent experiences are saved in a replay buffer D. The actor is then updated using both a supervised loss and the value loss with a replay ratio of $N _ { \theta }$

$$
\mathcal {L} _ {\theta} = \mathbb {E} ^ {\mathcal {D}} \big [ \| \varepsilon_ {t} - \varepsilon_ {\theta} (a _ {t} ^ {0}, s _ {t}, k) \| ^ {2} - \alpha_ {\mathrm{DQL}} \hat {Q} _ {\phi} (s _ {t}, \bar {\pi} _ {\theta} (a _ {t} ^ {0} | s _ {t})) \big ],
$$

where $\alpha _ { \mathrm { D Q L } }$ is a weighting coefficient. The critic is trained to minimize the Bellman residual with a replay ratio of $N _ { \phi }$ . Double Q-learning is also applied.

$$
\mathcal {L} _ {\phi} = \mathbb {E} ^ {\mathcal {D}} \big [ | | (R _ {t} + \gamma_ {\mathrm{ENV}} \hat {Q} _ {\phi} (s _ {t + 1}, \bar {\pi} _ {\theta} (a _ {t + 1} ^ {0} | s _ {t + 1})) - \hat {Q} _ {\phi} (s _ {t}, a _ {t} ^ {0}) | | ^ {2} \big ]
$$

QSM [74]: This baselines learns a state-action Q function, and then updates the actor by aligning the score of the diffusion actor with the gradient of the Q function.

Again recent experiences are saved in a replay buffer D. The critic is trained to minimize the Bellman residual with a replay ratio of $N _ { \phi }$ . Double Q-learning is also applied.

$$
\mathcal {L} _ {\phi} = \mathbb {E} ^ {\mathcal {D}} \big [ \| (R _ {t} + \gamma_ {\mathrm{ENV}} \hat {Q} _ {\phi} (s _ {t + 1}, \bar {\pi} _ {\theta} (a _ {t + 1} ^ {0} | s _ {t + 1})) - \hat {Q} _ {\phi} (s _ {t}, a _ {t} ^ {0}) \| ^ {2} \big ].
$$

The actor is updated as follows with a replay ratio of $N _ { \theta }$ .

$$
\mathcal {L} _ {\theta} = \mathbb {E} ^ {\mathcal {D}} \big [ \| \alpha_ {\mathrm{QSM}} \nabla_ {a} \hat {Q} _ {\phi} (s _ {t}, a _ {t}) - (- \varepsilon_ {\theta} (a _ {t} ^ {0}, s _ {t}, k)) \| ^ {2} \big ],
$$

where $\alpha _ { \mathrm { Q S M } }$ scales the gradient. The negative sign before $\varepsilon _ { \theta }$ is from taking the gradient of the mean $\mu$ in the denoising process.

## E.4 Descriptions of RL fine-tuning algorithm baselines in Section 5.2

In this subsection, we detail the baselines RLPD, Cal-QL, and IBRL. All policies $\pi \theta$ are parameterized as unimodal Gaussian.

RLPD [7]: This baseline is based on Soft Actor Critic (SAC, Haarnoja et al. [33]) — it learns an entropyregularized state-action Q function, and then updates the actor by maximizing the Q function w.r.t. the action.

A replay buffer D is initialized with offline data, and online samples are added to D. Each gradient update uses a batch of mixed 50/50 offline and online data. An ensemble of $N _ { \mathrm { c r i t i c } }$ critics is used, and at each gradient step two critics are randomly chosen. The critics are trained to minimize the Bellman residual with replay ratio $N _ { \phi } \mathrm { : }$

$$
\mathcal {L} _ {\phi} = \mathbb {E} ^ {\mathcal {D}} \big [ \| (R _ {t} + \gamma_ {\mathrm{ENV}} \hat {Q} _ {\phi^ {\prime}} (s _ {t + 1}, \pi_ {\theta} (a _ {t + 1} | s _ {t + 1})) - \hat {Q} _ {\phi} (s _ {t}, a _ {t}) \| ^ {2} \big ].
$$

The target critic parameter $\phi ^ { \prime }$ is updated with delay. The actor minimizes the following loss with a replay ratio of 1:

$$
\mathcal {L} _ {\theta} = \mathbb {E} ^ {\mathcal {D}} \big [ - \hat {Q} _ {\phi} (s _ {t}, a _ {t}) + \alpha_ {\mathrm{ent}} \log \pi_ {\theta} (a _ {t} | s _ {t}) \big ],
$$

where $\alpha _ { \mathrm { e n t } }$ is the entropy coefficient (automatically tuned as in SAC starting at 1).

Cal-QL [62]: This baseline trains the policy $\mu$ and the action-value function $Q ^ { \mu }$ in an offline phase and then an online phase. During the offline phase only offline data is sampled for gradient update, while during the online phase mixed 50/50 offline and online data are sampled. The critic is trained to minimize the following loss (Bellman residual and calibrated Q-learning):

$$
\begin{array}{r l} & {\mathcal {L} _ {\phi} = \mathbb {E} ^ {\mathcal {D}} \big [ \| (R _ {t} + \gamma_ {\mathrm{ENV}} \hat {Q} _ {\phi^ {\prime}} (s _ {t + 1}, \pi_ {\theta} (a _ {t + 1} | s _ {t + 1}))) - \hat {Q} _ {\phi} (s _ {t}, a _ {t}) \| ^ {2} \big ]} \\ & {\qquad + \beta_ {\mathrm{cql}} (\mathbb {E} ^ {\mathcal {D}} \big [ \max (Q _ {\phi} (s _ {t}, a _ {t}), V (s _ {t})) \big ] - \mathbb {E} ^ {\mathcal {D}} \big [ Q _ {\phi} (s _ {t}, a _ {t}) \big ]),} \end{array}
$$

where $\beta _ { \mathrm { c q l } }$ is a weighting coefficient between Bellman residual and calibration Q-learning and $V ( s _ { t } )$ is estimated using Monte-Carlo returns. The target critic parameter $\phi ^ { \prime }$ is updated with delay. The actor minimizes the following loss:

$$
\mathcal {L} _ {\theta} = \mathbb {E} ^ {\mathcal {D}} \big [ - \hat {Q} _ {\phi} (s _ {t}, a _ {t}) + \alpha_ {\mathrm{ent}} \log \pi_ {\theta} (a _ {t} | s _ {t}) \big ],
$$

where $\alpha _ { \mathrm { e n t } }$ is the entropy coefficient (automatically tuned as in SAC starting at 1).

IBRL [40]: This baseline first pre-trains a policy $\mu _ { \psi }$ using behavior cloning, and for fine-tuning it trains a RL policy $\pi _ { \theta }$ initialized as $\mu _ { \psi }$ . During fine-tuning recent experiences are saved in a replay buffer D. An ensemble of $N _ { \mathrm { c r i t i c } }$ critics is used, and at each gradient step two critics are randomly chosen. The critics are trained to minimize the Bellman residual with replay ratio $N _ { \phi }$

$$
\mathcal {L} _ {\phi} = \mathbb {E} ^ {\mathcal {D}} \big [ \| (R _ {t} + \gamma_ {\mathrm{ENV}} \max _ {a ^ {\prime} \in \{a ^ {I L}, a ^ {R L} \}} \hat {Q} _ {\phi^ {\prime}} (s _ {t + 1}, a ^ {\prime}) - \hat {Q} _ {\phi} (s _ {t}, a _ {t}) \| ^ {2} \big ]
$$

where $a ^ { I L } = \mu _ { \psi } ( s _ { t + 1 } )$ (no noise) and $a ^ { R L } \sim \pi _ { \theta ^ { \prime } } ( s _ { t + 1 } ) $ , and $\pi _ { \theta ^ { \prime } }$ is the target actor. The target critic parameter $\phi ^ { \prime }$ is updated with delay. The actor minimizes the following loss with a replay ratio of 1:

$$
\mathcal {L} _ {\theta} = - \mathbb {E} ^ {\mathcal {D}} \big [ \hat {Q} _ {\phi} (s _ {t}, a _ {t}) \big ].
$$

The target actor parameter $\theta ^ { \prime }$ is also updated with delay.

## E.5 Additional details of FRANKA-KITCHEN tasks and training in Section 5.2

Tasks. We consider three settings from the D4RL benchmark [29]: (1) Kitchen-Complete-v0 containing demonstrations that complete the entire task (four subtasks), (2) Kitchen-Partial-v0 containing some complete demonstrations and many ones completing only subtasks, and (3) Kitchen-Mixed-v0 containing incomplete demonstrations only.

Pre-training. The observations and actions are normalized to [0, 1] using min/max statistics from the pretraining dataset. No history observation (proprioception or ground-truth object states) is used. All policies are trained with batch size 128, learning rate 1e-4 decayed to 1e-5 with a cosine schedule, and weight decay 1e-6. DPPO policies are trained with 8000 epochs. For IBRL and Cal-QL we follow the hyperparameters from the original implementations — IBRL proposes using (1) wider MLP layers and (2) dropout during pre-training, which we follow too. We use $T _ { a } = 4$ for DPPO; we also tried to use the same action chunk size with IBRL, RLPD, and Cal-QL, but we find for all of them $T _ { a } = 1$ leads to better performance.

Fine-tuning. With DPPO, policies are fine-tuned using online experiences sampled from 40 parallelized MuJoCo environments [92], while the baselines use only one environment (matching their original implementations). Episodes terminates when they reach maximum episode lengths (shown in Table A6) or all four subtasks are completed. Detailed hyperparameters are listed in Table A8 — we follow the hyperparameter choices from the original implementations of the baselines.

Larger variance with DPPO in Fig. 6. In Fig. 6, it is shown that DPPO exhibits a larger variance in normalized score with Kitchen-Partial-v0 than Cal-QL. This is due to DPPO solving either 3/4 or 2/4 subtasks in one seed (low variance within the evaluation episodes in one seed) but high variance over seeds, whereas Cal-QL has higher variance among evaluation episodes in one seed but on average over seeds it shows lower variance. This also highlights a notable property of DPPO: Kitchen-Partial-v0 and Kitchen-Mixed-v0 have trajectories only completing subtasks, thus being highly multi-modal. Diffusion policy can sometimes struggle to learn all the modes from pre-training, and since DPPO directly fine-tunes the pre-trained policy, it can fail to converge to 100% success rate at fine-tuning. Cal-QL instead learns from all offline data during fine-tuning in an off-policy manner, thus less sensitive to pretraining performance. Nonetheless, with offline data completing tasks consistently despite varying quality (ROBOMIMIC and Kitchen-Complete-v0, which, we believe, are more realistic in the current paradigm of robot manipulation), DPPO demonstrates much better final performance than Cal-QL and other baselines in Fig. 6.

## E.6 Additional details of ROBOMIMIC tasks and training in Section 5.3

Tasks. We consider four tasks from the ROBOMIMIC benchmark [60]: (1) Lift: lifting a cube from the table, (2) Can: picking up a Coke can and placing it at a target bin, (3) Square: picking up a square nut and place it on a rod, and (4) Transport: two robot arms removing a bin cover, picking and placing a cube, and then transferring a hammer from one container to another one.

Pre-training. ROBOMIMIC provides the Multi-Human (MH) dataset with noisy human demonstrations for each task, which we use to pre-train the policies. The observations and actions are normalized to [0, 1] using min/max statistics from the pre-training dataset. No history observation (pixel, proprioception, or ground-truth object states) is used. All policies are trained with batch size 128, learning rate 1e-4 decayed to 1e-5 with a cosine schedule, and weight decay 1e-6. Diffusion-based policies are trained with 8000 epochs, while Gaussian and GMM policies are trained with 5000 epochs — we find diffusion models require more gradient updates to fit the data well.

Fine-tuning. Diffusion-based, Gaussian, and GMM pre-trained policies are then fine-tuned using online experiences sampled from 50 parallelized MuJoCo environments [92]. Success rate curves shown in Fig. 5, Fig. 7, and Fig. A9 are evaluated by running fine-tuned policies with $\sigma _ { \mathrm { m i n } } ^ { \mathrm { e x p } } = 0 . 0 0 1$ (i.e., without extra noise) for 50 episodes. Episodes terminates only when they reach maximum episode lengths (shown in Table A6). Detailed hyperparameters are listed in Table A9.

Pixel training. We use the wrist camera view in Lift and Can, the third-person camera view in Square, and the two robot shoulder camera views in Transport. Random-shift data augmentation is applied to the camera images during both pre-training and fine-tuning. Gradient accumulation is used in fine-tuning so that the same batch size (as in state-input training) can fit on the GPU. Detailed hyperparameters are listed in Table A10.

## E.7 Descriptions of policy parameterization baselines in Section 5.3

Gaussian. We consider unimodal Gaussian with diagonal covariance, the most commonly used policy parameterization in RL. The standard deviation for each action dimension, $\sigma _ { \mathrm { G a u } }$ , is fixed during pre-training; we also tried to learn $\sigma _ { \mathrm { G a u } }$ from the dataset but we find the training very unstable. During fine-tuning $\sigma _ { \mathrm { G a u } }$ is learned starting from the same fixed value and also clipped between 0.01 and 0.2. Additionally we clip the sampled action to be within 3 standard deviation from the mean. As discusses in Appendix B, we choose the PPO clipping ratio ε based on the empirical clipping fraction in each task. This setup is also used in the FURNITURE-BENCH experiments. We note that we spend significant amount of efforts tuning the Gaussian baseline, and our results with it are some of the best known ones in RL training for long-horizon manipulation tasks (exceeding our initial expectations), e.g., reaching ∼100% success rate in Lamp with Low randomness.

GMM. We also consider Gaussian Mixture Model as the policy parameterization. We denote $M _ { \mathrm { G M M } }$ the number of mixtures. The standard deviation for each action dimension in each mixture, σ<sub>GMM</sub>, is also fixed during pre-training. Again during fine-tuning σ<sub>GMM</sub> is learned starting from the same fixed value and also clipped between 0.01 and 0.2.

## E.8 Additional details of FURNITURE-BENCH tasks and training in Section 5.4

Tasks. We consider three tasks from the FURNITURE-BENCH benchmark [36]: (1) One-leg: assemble one leg of a table by placing the tabletop in the fixture corner, grasping and inserting the table leg, and screwing in the leg, (2) Lamp: place the lamp base in the fixture corner, grasp, insert, and screw in the light bulb, and finally place the lamp shade, (3) Round-table: place a round tabletop in the fixture corner, insert and screw in the table leg, and then insert and screw in the table base. See Fig. A12 for the visualized rollouts in simulation.

Pre-training. The pre-training dataset is collected in the simulated environments using a SpaceMouse<sup>10</sup>, a 6 DoF input device. The simulator runs at 10Hz. At every timestep, we read off the state of the SpaceMouse as $\delta \mathbf { a } = [ \Delta x , \Delta y , \Delta z , \Delta \mathrm { r o l l } , \Delta \mathrm { p i t c h } , \Delta \mathrm { y a w } ]$ , which is converted to a quaternion before passed to the environment step and stored as the action alongside the current observation in the trajectory. If $| \Delta \mathbf { a } _ { i } | < \varepsilon \forall i$ for some small $\varepsilon = 0 . 0 5$ defining the threshold for a no-op, we do not record any action nor pass it to the environment. Discarding no-ops is important for allowing the policies to learn from demonstrations effectively. When the desired number of demonstrations has been collected (typically 50), we process the actions to convert the delta actions stored from the SpaceMouse into absolute pose actions by applying the delta action to the current EE pose at each timestep.

The observations and actions are normalized to [−1, 1] using min/max statistics from the pre-training dataset. No history observation (proprioception or ground-truth object states) is used, i.e., only the current observation is passed to the policy. All policies are trained with batch size 256, learning rate 1e-4 decayed to 1e-5 with a cosine schedule, and weight decay 1e-6. Diffusion-based policies are trained with 8000 epochs, while Gaussian policies are trained with 3000 epochs. Gaussian policies can easily overfit the pre-trained dataset, while diffusion-based policies are more resilient. Gaussian policies also require a very large MLP (∼10 million parameters) to fit the data well.

Fine-tuning. Diffusion-based and Gaussian pre-trained policies are then fine-tuned using online experiences sampled from 1000 parallelized IsaacGym environments [59]. Success rate curves shown in Fig. 8 are evaluated by running fine-tuned policies with $\sigma _ { \mathrm { m i n } } ^ { \mathrm { e x p } } = 0 . 0 0 1$ (i.e., without extra noise) for 1000 episodes. Episodes terminate only when they reach maximum episode length (shown in Table A6). Detailed hyperparameters are listed in Table A11. We find a smaller amount of exploration noise (we set $\sigma _ { \mathrm { m i n } } ^ { \mathrm { e x p } }$ and $\sigma _ { \mathrm { G a u } }$ to be 0.04) is necessary for the pre-trained policy achieving nonzero success rates at the beginning of fine-tuning.

![](images/7ef9645bc3c7e02dd660046c913abc9df734a8c00740d2bd06f0e6ee3d8cc9ff.jpg)  
Figure A12: Representative rollouts from simulated FURNITURE-BENCH tasks.

Hardware setup - robot control. The physical robot used is a Franka Emika Panda arm. The policies Robust sim-to-real transfer in zero-shotoutput a sequence of desired end-effector poses in the robot base frame to control the robot. These poses are converted into joint position targets through differential inverse kinematics. We calculate the desired endeffector velocity as the difference between the desired and current poses divided by the delta time $d t = 1 / 1 0$ We then convert this to desired joint velocities using the Jacobian and compute the desired joint positions with a first-order integration over the current joint positions and desired velocity. The resulting joint position targets are passed to a low-level joint impedance controller provided by Polymetis [55], running at 1kHz.

Hardware setup - state estimation. To deploy state-based policies on real hardware, we utilize AprilTags [98] for part pose estimation. The FURNITURE-BENCH [36] task suite provides AprilTags for each part and code for estimating part poses from tag detections. The process involves several steps: (1) detecting tags in the camera frame, (2) mapping tag detections to the robot frame for policy compatibility, (3) utilizing known offsets between tags and object centers in the simulator, and (4) calibrating the camera pose using an

AprilTag at a known position relative to the robot base. Despite general accuracy, detections can be noisy, especially during movement or partial occlusion, which the One-leg task features. Since the task requires high precision, we find the following to help make the estimation reliable enough:

• Camera coverage: We find detection quality sensitive to distance and angle between the camera and tag. This issue is likely due to the RealSense D435 camera having mediocre image quality and clarity and the relatively small tags. To remedy this, we opt to use 4 cameras roughly evenly spread out around the scene to ensure that at least one camera has a solid view of a tag on all the parts (i.e., as close as possible with a straight-on view). To find the best camera positions, we start with having a camera in each of the cardinal directions around the scene. Then, we adjust the pose of each to get it as close as possible to the objects while still covering the necessary workspace and capturing the base tag for calibration. Moving the robot arm around the scene to avoid the worst occlusion is also helpful.

• Lighting: Even with better camera coverage and placement, detection quality depends on having crisp images. We find proper lighting helpful to improve image quality. In particular, the scene should be well and evenly lit around the scene without causing reflections in either the tag or table.

• Filtering: Bad detections can sometimes cause the resulting pose estimate to deviate significantly from the true pose, i.e., jumping several centimeters from one frame to the next. This usually only happens on isolated frames, and thus before “accepting” a given detection, we check if the new position and orientation are within 5 cm and 20 degrees of the previously accepted pose. In addition, we apply lowpass filtering on the detection using a simple exponential average (with α = 0.25) to smooth out the high-frequency noise.

• Averaging: The objects have multiple tags that can be detected from multiple cameras. After performing the filtering step, we average all pose estimates for the same object across different tags and cameras, which also helps smooth out noise. This alone, however, does not fully cancel the case when a single detection has a large jump, as this can severely skew the average, still necessitating a filtering step. Having multiple cameras benefits this step, too, as it provides more detections to average over.

• Caching part pose in hand: A particularly difficult phase of the task to achieve good detections is when the robot transports the table leg from the initial position to the tabletop for insertion. The main problems are that the movement can blur the images, and the grasping can cause occlusions. Therefore, we found it helpful to assume that once the part was grasped by the robot, it would not move in the grasp until the gripper opened. With this, we can “cache” the pose of the part relative to the end-effector once the object is fully grasped and use this instead of relying on detections during the movement.

• Normalization pitfalls and clipping: We generally use min-max normalization of the state observations to ensure observations are in [−1, 1]. The tabletop part moves very little in the z-direction demonstration data, meaning the resulting normalization limits (the minimum and maximum value of the data) can be very close, $x _ { \mathrm { m a x } } - x _ { \mathrm { m i n } } \approx 0$ . With these tight limits, the noise in the real-world detection can be amplified greatly as $\begin{array} { r } { x _ { \mathrm { n o r m } } = \frac { x - x _ { \mathrm { m i n } } } { x _ { \mathrm { m a x } } - x _ { \mathrm { m i n } } } } \end{array}$ . Therefore, ensure that normalization ranges are reasonable. As an extra safeguard, clipping the data to [−1, 1] can also help.

• Only estimate necessary states: Despite the One-leg task having 5 parts, only 2 are manipulated. Only estimating the pose of those parts can eliminate a lot of noise. In particular, the pose of the 3 legs that are not used and the obstacle (the U-shaped fixture) can be set to an arbitrary value from the dataset.

• Visualization for debugging: We use the visualization tool MeshCat<sup>11</sup> extensively for debugging of state estimation. The tool allows for easy visualizations of poses of all relevant objects in the scene, like the robot end-effector and parts, which makes sanity-checking the implementation far easier than looking at raw numbers.

Hardware evaluation. We perform 20 trials for each method. We adopt a single-blind model selection process: at the beginning of each trial, we first randomize the initial state. Then, we randomly select a method and roll it out, but the experimenter does not observe which model is used. We record the success and failure of each trial and then aggregate statistics for each model after all trials are completed.

Domain randomization for sim-to-real transfer. To facilitate the sim-to-real transfer, we apply additional domain randomization to the simulation training. We record the range of observation noises in hardware without any robot motion and then apply the same amount of noise to state observations in simulation. We find the state estimation in hardware particularly sensitive to the object heights. Also, we apply random noise (zero mean with 0.03 standard deviation) to the sampled action from DPPO to simulate the imperfect low-level controller; we find adding such noise to the Gaussian policy leads to zero task success rate while DPPO is robust to it (also see discussion in Section 6).

BC regularization loss used for Gaussian baseline. Since the fine-tuned Gaussian policy exhibits very jittery behavior and leads to zero success rate in real evaluation, we further experiment with adding a behavior cloning (BC) regularization loss in fine-tuning with the Gaussian baseline. The combined loss follows

$$
\mathcal {L} _ {\theta , + \mathrm{BC}} = \mathcal {L} _ {\theta} - \alpha_ {\mathrm{BC}} \mathbb {E} ^ {\pi_ {\theta_ {\mathrm{old}}}} [ \sum_ {k = 0} ^ {K - 1} \log \pi_ {\theta_ {\mathrm{pre-trained}}} (a _ {t} ^ {k} | a _ {t} ^ {k + 1}, s _ {t}) ],
$$

where $\pi _ { \theta _ { \mathrm { p r e - t r a i n e d } } }$ is the frozen BC-only policy. The extra term encourages the newly sampled actions from the fine-tuned policy to remain high-likelihood under the BC-only policy. We set $\alpha _ { \mathrm { B C } } = 0 . 1$ . However, although this regularization reduces the sim-to-real gap, it also significantly limits fine-tuning, leading to the fine-tuning policy saturating at 53% success rate shown in Fig. 8.

## E.9 Additional details of Avoid task from D3IL and training in Section 6

Pre-training. We split the original dataset from D3IL based on the three settings, M1, M2, and M3; in each setting, observations and actions are normalized to [0, 1] using min/max statistics. All policies are trained with batch size 16 (due to the small dataset size), learning rate 1e-4 decayed to 1e-5 with a cosine schedule, and weight decay 1e-6. Diffusion-based policies are trained with about 15000 epochs, while Gaussian and GMM policies are trained with about 10000 epochs; we manually examine the trajectories from different pre-trained checkpoints and pick ones that visually match the expert data the best.

Fine-tuning. Diffusion-based, Gaussian, and GMM pre-trained policies are then fine-tuned using online experiences sampled from 50 parallelized MuJoCo environments [92]. Reward curves shown in Fig. 11 and Fig. A8 are evaluated by running fine-tuned policies with the same amount of exploration noise used in training for 50 episodes; we choose to use the training (instead of evaluation) setup since Gaussian policies exhibit multi-modality only with training noise. Episodes terminate only when they reach 100 steps.

Added action noise during fine-tuning. In Fig. 11 left, we demonstrate that DPPO exhibits stronger training stability when noise is added to the sampled actions during fine-tuning. The noise starts at the 5th iteration. It is sampled from a uniform distribution with the lower limit ramping up to 0.1 and the upper limit ramping up to 0.2 linearly in 5 iterations. The limits are kept the same from the 10th iteration to the end of fine-tuning.

## E.10 Listed training hyperparameters

<table><tr><td rowspan="2">Method</td><td rowspan="2">Parameter</td><td colspan="4">Task(s)</td></tr><tr><td>GYM</td><td>Lift, Can</td><td>Square</td><td>Transport</td></tr><tr><td rowspan="10">Common</td><td> $\gamma_{ENV}$ </td><td>0.99</td><td>0.999</td><td>0.999</td><td>0.999</td></tr><tr><td> $\sigma_{\min}^{exp}$ </td><td>0.1</td><td>0.1</td><td>0.1</td><td>0.08</td></tr><tr><td> $\sigma_{\min}^{prob}$ </td><td></td><td></td><td>0.1</td><td></td></tr><tr><td> $T_p$ </td><td>4</td><td>4</td><td>4</td><td>8</td></tr><tr><td> $T_a$ </td><td>4</td><td>4</td><td>4</td><td>8</td></tr><tr><td>K</td><td></td><td></td><td>20</td><td></td></tr><tr><td>Actor learning rate</td><td colspan="4">1e-4 for DPPO and 1e-5 for others (tuned from 1e-4 to 1e-5)</td></tr><tr><td>Critic learning rate (if applies)</td><td></td><td></td><td>1e-3</td><td></td></tr><tr><td>Actor MLP dims</td><td>[512, 512, 512]</td><td>[512, 512, 512]</td><td>[1024, 1024, 1024]</td><td>[1024, 1024, 1024]</td></tr><tr><td>Critic MLP dims (if applies)</td><td></td><td></td><td>[256, 256, 256]</td><td></td></tr><tr><td rowspan="4">DRWR</td><td>β</td><td></td><td></td><td>10</td><td></td></tr><tr><td> $w_{max}$ </td><td></td><td></td><td>100</td><td></td></tr><tr><td> $N_\theta$ </td><td></td><td></td><td>16</td><td></td></tr><tr><td>Batch size</td><td></td><td></td><td>1000</td><td></td></tr><tr><td rowspan="7">DAWR</td><td>β</td><td></td><td></td><td>10</td><td></td></tr><tr><td> $w_{max}$ </td><td></td><td></td><td>100</td><td></td></tr><tr><td> $\lambda_{DAWR}$ </td><td></td><td></td><td>0.95</td><td></td></tr><tr><td> $N_\theta$ </td><td></td><td></td><td>64</td><td></td></tr><tr><td> $N_\phi$ </td><td></td><td></td><td>16</td><td></td></tr><tr><td>Buffer size</td><td>200000</td><td>120000</td><td>120000</td><td>120000</td></tr><tr><td>Batch size</td><td></td><td></td><td>1000</td><td></td></tr><tr><td rowspan="5">DIPO</td><td> $\alpha_{DIPO}$ </td><td></td><td></td><td>1e-4</td><td></td></tr><tr><td> $M_{DIPO}$ </td><td></td><td></td><td>10</td><td></td></tr><tr><td> $N_\theta$ </td><td></td><td></td><td>64</td><td></td></tr><tr><td>Buffer size</td><td></td><td></td><td>1000000</td><td></td></tr><tr><td>Batch size</td><td></td><td></td><td>1000</td><td></td></tr><tr><td rowspan="5">IDQL</td><td> $M_{IDQL}$ </td><td>20</td><td>10</td><td>10</td><td>10</td></tr><tr><td> $N_\theta$ </td><td></td><td></td><td>128</td><td></td></tr><tr><td> $N_\phi$ </td><td></td><td></td><td>128</td><td></td></tr><tr><td>Buffer size</td><td>1000000</td><td>250000</td><td>250000</td><td>250000</td></tr><tr><td>Batch size</td><td></td><td></td><td>1000</td><td></td></tr><tr><td rowspan="5">DQL</td><td> $\alpha_{DQL}$ </td><td></td><td></td><td>1</td><td></td></tr><tr><td> $N_\theta$ </td><td></td><td></td><td>16</td><td></td></tr><tr><td> $N_\phi$ </td><td></td><td></td><td>16</td><td></td></tr><tr><td>Buffer eize</td><td></td><td></td><td>1000000</td><td></td></tr><tr><td>Batch size</td><td></td><td></td><td>1000</td><td></td></tr><tr><td rowspan="5">QSM</td><td> $\alpha_{QSM}$ </td><td></td><td></td><td>10</td><td></td></tr><tr><td> $N_\theta$ </td><td></td><td></td><td>16</td><td></td></tr><tr><td> $N_\phi$ </td><td></td><td></td><td>16</td><td></td></tr><tr><td>Buffer size</td><td>1000000</td><td>250000</td><td>250000</td><td>250000</td></tr><tr><td>Batch size</td><td></td><td></td><td>1000</td><td></td></tr><tr><td rowspan="7">DPPO</td><td> $\gamma_{DENOISE}$ </td><td></td><td></td><td>0.99</td><td></td></tr><tr><td>GAE λ</td><td></td><td></td><td>0.95</td><td></td></tr><tr><td> $N_\theta$ </td><td>5</td><td>10</td><td>10</td><td>10</td></tr><tr><td> $N_\phi$ </td><td>5</td><td>10</td><td>10</td><td>10</td></tr><tr><td>ε</td><td></td><td></td><td>0.01</td><td></td></tr><tr><td>Batch size</td><td>50000</td><td>7500</td><td>10000</td><td>10000</td></tr><tr><td> $K'$ </td><td></td><td></td><td>10</td><td></td></tr></table>

Table A7: Fine-tuning hyperparameters for OpenAI GYM and ROBOMIMIC tasks when comparing diffusion-based RL methods. We list shared hyperparameters and then method-specific ones.

<table><tr><td rowspan="2">Method</td><td rowspan="2">Parameter</td><td colspan="5">Task(s)</td></tr><tr><td>HalfCheetah-v2</td><td>Kitchen-Complete-v0</td><td>Kitchen-Partial-v0</td><td>Kitchen-Mixed-v0</td><td></td></tr><tr><td>Common</td><td> $\gamma_{ENV}$ </td><td></td><td></td><td>0.99</td><td></td><td></td></tr><tr><td rowspan="5">RLPD</td><td> $T_p$ </td><td></td><td></td><td>1</td><td></td><td></td></tr><tr><td> $T_a$ </td><td></td><td></td><td>1</td><td></td><td></td></tr><tr><td> $N_\phi$ </td><td>23</td><td>10310</td><td></td><td>10</td><td></td></tr><tr><td> $N_{critic}$ </td><td>10</td><td>5</td><td></td><td>5</td><td>5</td></tr><tr><td>Batch size</td><td></td><td></td><td>256</td><td></td><td></td></tr><tr><td rowspan="4">Cal-QL</td><td> $T_p$ </td><td></td><td></td><td>1</td><td></td><td></td></tr><tr><td> $T_a$ </td><td></td><td></td><td>1</td><td></td><td></td></tr><tr><td> $\beta_{cql}$ </td><td></td><td></td><td>5</td><td></td><td></td></tr><tr><td>Batch size</td><td></td><td></td><td>256</td><td></td><td></td></tr><tr><td rowspan="5">IBRL</td><td> $T_p$ </td><td></td><td></td><td>1</td><td></td><td></td></tr><tr><td> $T_a$ </td><td></td><td></td><td>1</td><td></td><td></td></tr><tr><td> $N_\phi$ </td><td></td><td></td><td>5</td><td></td><td></td></tr><tr><td> $N_{critic}$ </td><td></td><td></td><td>5</td><td></td><td></td></tr><tr><td>Batch size</td><td></td><td></td><td>256</td><td></td><td></td></tr><tr><td rowspan="12">DPPO</td><td> $T_p$ </td><td>1</td><td>4</td><td></td><td>4</td><td>4</td></tr><tr><td> $T_a$ </td><td>1</td><td>4</td><td></td><td>4</td><td>4</td></tr><tr><td> $\sigma_{\min}^{\text{exp}}$ </td><td></td><td></td><td>0.1</td><td></td><td></td></tr><tr><td> $\sigma_{\min}^{\text{prob}}$ </td><td></td><td></td><td>0.1</td><td></td><td></td></tr><tr><td> $\gamma_{\text{DENOISE}}$ </td><td></td><td></td><td>0.99</td><td></td><td></td></tr><tr><td>GAE λ</td><td></td><td></td><td>0.95</td><td></td><td></td></tr><tr><td> $N_\theta$ </td><td>5</td><td>10</td><td></td><td>10</td><td>10</td></tr><tr><td> $N_\phi$ </td><td>5</td><td>10</td><td></td><td>10</td><td>10</td></tr><tr><td>ε</td><td></td><td></td><td>0.01</td><td></td><td></td></tr><tr><td>Batch size</td><td>10000</td><td>5600</td><td></td><td>5600</td><td>5600</td></tr><tr><td>K</td><td></td><td></td><td>20</td><td></td><td></td></tr><tr><td> $K'$ </td><td></td><td></td><td>10</td><td></td><td></td></tr><tr><td rowspan="22"></td><td>Method</td><td>Parameter</td><td>Can, PH</td><td>Square, PH</td><td>Can, MH</td><td>Square, MH</td></tr><tr><td rowspan="3">Common</td><td> $\gamma_{ENV}$ </td><td></td><td></td><td>0.999</td><td></td></tr><tr><td> $T_a$ </td><td></td><td></td><td>1</td><td></td></tr><tr><td> $T_a$ </td><td></td><td></td><td>1</td><td></td></tr><tr><td rowspan="3">RLPD</td><td> $N_\phi$ </td><td></td><td></td><td>3</td><td></td></tr><tr><td> $N_{critic}$ </td><td></td><td></td><td>5</td><td></td></tr><tr><td>Batch size</td><td></td><td></td><td>256</td><td></td></tr><tr><td rowspan="2">Cal-QL</td><td> $\beta_{cql}$ </td><td></td><td></td><td>5</td><td></td></tr><tr><td>Batch size</td><td></td><td></td><td>256</td><td></td></tr><tr><td rowspan="3">IBRL</td><td> $N_\phi$ </td><td></td><td></td><td>3</td><td></td></tr><tr><td> $N_{critic}$ </td><td></td><td></td><td>5</td><td></td></tr><tr><td>Batch size</td><td></td><td></td><td>256</td><td></td></tr><tr><td rowspan="10">DPPO</td><td> $\sigma_{\min}^{\text{exp}}$ </td><td></td><td></td><td>0.1</td><td></td></tr><tr><td> $\sigma_{\min}^{\text{prob}}$ </td><td></td><td></td><td>0.1</td><td></td></tr><tr><td> $\gamma_{\text{DENOISE}}$ </td><td>0.9</td><td>0.9</td><td>0.99</td><td>0.99</td></tr><tr><td>GAE λ</td><td></td><td></td><td>0.95</td><td></td></tr><tr><td> $N_\theta$ </td><td></td><td></td><td>10</td><td></td></tr><tr><td> $N_\phi$ </td><td></td><td></td><td>10</td><td></td></tr><tr><td>ε</td><td></td><td></td><td>0.01</td><td></td></tr><tr><td>Batch size</td><td>6000</td><td>15000</td><td>8000</td><td>20000</td></tr><tr><td>K</td><td></td><td></td><td>20</td><td></td></tr><tr><td> $K'$ </td><td></td><td></td><td>10</td><td></td></tr><tr><td rowspan="2">Method</td><td rowspan="2">Parameter</td><td colspan="5">Task</td></tr><tr><td>Lift, Can</td><td>Square</td><td colspan="3">Transport</td></tr><tr><td rowspan="8">Common</td><td> $\gamma_{ENV}$ </td><td></td><td>0.999</td><td colspan="3"></td></tr><tr><td> $T_a$ </td><td>4</td><td>4</td><td colspan="3">8</td></tr><tr><td>Actor learning rate</td><td>1e-4</td><td>1e-5</td><td colspan="3">1e-5 (decayed to 1e-6)</td></tr><tr><td>Critic learning rate</td><td></td><td>1e-3</td><td colspan="3"></td></tr><tr><td>GAE  $\lambda$ </td><td></td><td>0.95</td><td colspan="3"></td></tr><tr><td> $N_\theta$ </td><td>10</td><td>10</td><td colspan="3">8</td></tr><tr><td> $N_\phi$ </td><td>10</td><td>10</td><td colspan="3">8</td></tr><tr><td> $\varepsilon$ </td><td></td><td>0.01 (annealed in DPPO)</td><td colspan="3"></td></tr><tr><td rowspan="2">Gaussian, Common</td><td> $\sigma_{\text{Gau}}$ </td><td>0.1</td><td>0.1</td><td colspan="3">0.08</td></tr><tr><td>Batch size</td><td>7500</td><td>10000</td><td colspan="3">10000</td></tr><tr><td>Gaussian-MLP</td><td>Model size</td><td>552K</td><td>2.15M</td><td colspan="3">1.93M</td></tr><tr><td>Gaussian-Transformer</td><td>Model size</td><td>675K</td><td>1.86M</td><td colspan="3">1.87M</td></tr><tr><td rowspan="3">GMM, Common</td><td> $M_{\text{GMM}}$ </td><td></td><td>5</td><td colspan="3"></td></tr><tr><td> $\sigma_{\text{GMM}}$ </td><td>0.1</td><td>0.1</td><td colspan="3">0.08</td></tr><tr><td>Batch size</td><td>7500</td><td>10000</td><td colspan="3">10000</td></tr><tr><td>GMM-MLP</td><td>Model size</td><td>1.15M</td><td>4.40M</td><td colspan="3">4.90M</td></tr><tr><td>GMM-Transformer</td><td>Model size</td><td>680K</td><td>1.87M</td><td colspan="3">1.89M</td></tr><tr><td rowspan="6">DPPO, Common</td><td> $\gamma_{\text{DENOISE}}$ </td><td></td><td>0.99</td><td colspan="3"></td></tr><tr><td> $\sigma_{\text{min}}^{\text{exp}}$ </td><td>0.1</td><td>0.1</td><td colspan="3">0.08</td></tr><tr><td> $\sigma_{\text{min}}^{\text{prob}}$ </td><td>0.1</td><td>0.1</td><td colspan="3">0.1</td></tr><tr><td>K</td><td></td><td>20</td><td colspan="3"></td></tr><tr><td> $K'$ </td><td></td><td>10</td><td colspan="3"></td></tr><tr><td>Batch size</td><td>75000</td><td>100000</td><td colspan="3">100000</td></tr><tr><td>DPPO-MLP</td><td>Model size</td><td>576K</td><td>2.31M</td><td colspan="3">2.43M</td></tr><tr><td>DPPO-UNetGaussian-ViT-MLP</td><td>Model sizeModel size</td><td>652K1.03M</td><td>1.62M1.03M</td><td colspan="3">1.68M1.93M</td></tr><tr><td rowspan="2"></td><td> $σ_{Gau}$ </td><td>0.1</td><td>0.1</td><td colspan="3">0.08</td></tr><tr><td>Batch size</td><td>7500</td><td>10000</td><td colspan="3">10000</td></tr><tr><td rowspan="7">DPPO-ViT-MLP</td><td>Model size</td><td>1.06M</td><td>1.06M</td><td colspan="3">2.05M</td></tr><tr><td> $γ_{DENOISE}$ </td><td></td><td>0.9</td><td colspan="3"></td></tr><tr><td> $σ_{min}^{\text{exp}}$ </td><td>0.1</td><td>0.1</td><td colspan="3">0.08</td></tr><tr><td> $σ_{min}^{\text{prob}}$ </td><td></td><td>0.10</td><td colspan="3"></td></tr><tr><td>K</td><td></td><td>100</td><td colspan="3"></td></tr><tr><td> $K'$ </td><td></td><td>5 (DDIM)</td><td colspan="3"></td></tr><tr><td>Batch size</td><td>37500</td><td>50000</td><td colspan="3">50000</td></tr></table>

Table A8: Fine-tuning hyperparameters for HalfCheetah-v2, FRANKA-KITCHEN, Can, and Square (PH or MH datasets) when comparing demo-augmented RL methods. We list hyperparameters shared by all methods first, and then method-specific ones.

Table A9: Fine-tuning hyperparameters for ROBOMIMIC tasks with state input when comparing policy parameterizations. We list hyperparameters shared by all methods first, and then method-specific ones. Since the different policy parameterizations use different neural network architecture, we list the total model size here instead of the details such as MLP dimensions.

Table A10: Fine-tuning hyperparameters for ROBOMIMIC tasks with pixel input when comparing policy parameterizations. We list hyperparameters shared by all methods first, and then method-specific ones. Since the different policy parameterizations use different neural network architecture, we list the total model size here instead of the details such as MLP dimensions.

<table><tr><td rowspan="2">Method</td><td rowspan="2">Parameter</td><td colspan="3">Task</td></tr><tr><td>One-leg</td><td>Lamp</td><td>Round-table</td></tr><tr><td rowspan="8">Common</td><td> $\gamma_{ENV}$ </td><td></td><td>0.999</td><td></td></tr><tr><td> $T_a$ </td><td></td><td>8</td><td></td></tr><tr><td>Actor learning rate</td><td></td><td>1e-5 (decayed to 1e-6)</td><td></td></tr><tr><td>Critic learning rate</td><td></td><td>1e-3</td><td></td></tr><tr><td>GAE  $\lambda$ </td><td></td><td>0.95</td><td></td></tr><tr><td> $N_\theta$ </td><td></td><td>5</td><td></td></tr><tr><td> $N_\phi$ </td><td></td><td>5</td><td></td></tr><tr><td> $\varepsilon$ </td><td></td><td>0.001</td><td></td></tr><tr><td rowspan="3">Gaussian-MLP</td><td>Model size</td><td>10.64M</td><td>10.62M</td><td>10.62M</td></tr><tr><td> $σ_{Gau}$ </td><td></td><td>0.04</td><td></td></tr><tr><td>Batch size</td><td></td><td>8800</td><td></td></tr><tr><td rowspan="7">DPPO-UNet</td><td>Model size</td><td>6.86M</td><td>6.81M</td><td>6.81M</td></tr><tr><td> $γ_{DENOISE}$ </td><td></td><td>0.9</td><td></td></tr><tr><td> $σ_{min}^{exp}$ </td><td></td><td>0.04</td><td></td></tr><tr><td> $σ_{min}^{prob}$ </td><td></td><td>0.1</td><td></td></tr><tr><td>K</td><td></td><td>100</td><td></td></tr><tr><td> $K'$ </td><td></td><td>5 (DDIM)</td><td></td></tr><tr><td>Batch size</td><td></td><td>44000</td><td></td></tr></table>

Table A11: Fine-tuning hyperparameters for FURNITURE-BENCH tasks when comparing policy parameterizations. We list hyperparameters shared by all methods first, and then method-specific ones.