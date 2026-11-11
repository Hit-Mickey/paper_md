# Factorizing Diffusion Policies for Observation Modality Prioritization

Omkar Patil<sup>1</sup>, Prabin Kumar Rath<sup>1</sup>, Kartikay Pangaonkar<sup>1</sup>, Eric Rosen, Nakul Gopalan<sup>1</sup>

Abstract— Diffusion models have been extensively leveraged for learning robot skills from demonstrations. These policies are conditioned on several observational modalities such as proprioception, vision and tactile. However, observational modalities have varying levels of influence for different tasks that diffusion polices fail to capture. In this work, we propose ‘Factorized Diffusion Policies’ abbreviated as FDP, a novel policy formulation that enables observational modalities to have differing influence on the action diffusion process by design. This results in learning policies where certain observations modalities can be prioritized over the others such as vision>tactile or proprioception>vision. FDP achieves modality prioritization by factorizing the observational conditioning for diffusion process, resulting in more performant and robust policies. Our factored approach shows strong performance improvements in low-data regimes with 15% absolute improvement in success rate on several simulated benchmarks when compared to a standard diffusion policy that jointly conditions on all input modalities. Moreover, our benchmark and real-world experiments show that factored policies are naturally more robust with 40% higher absolute success rate across several visuomotor tasks under distribution shifts such as visual distractors or camera occlusions, where existing diffusion policies fail catastrophically. FDP thus offers a safer and more robust alternative to standard diffusion policies for real-world deployment. Videos are available at https: //fdp-policy.github.io/fdp-policy/.

## I. INTRODUCTION

Humans prioritize different sensory modalities according to the specific requirements of the task [1]. For instance, Wahn and König [1] note that participants engaged in visually demanding tasks are comparatively less receptive to auditory stimuli. They argue that this flexible allocation of human attentional capacity maximizes the capability to process relevant information. Further, humans have been shown to prioritize the more reliable modality between vision and haptics in different situations [2]. This naturally raises the question of whether policy learning could benefit from such prioritization of observational modalities influencing robot actions. Prioritization of the more influential or reliable modality could enable robot policies to learn skills more efficiently and avoid developing spurious correlations with noisy modalities. Moreover, the number of possible skills is vast and skills may depend more strongly on certain observational modalities over others. For instance, repetitive motions like sweeping are more likely to depend on the robot’s proprioception, while locating an object for manipulation is conditioned strongly on its vision. This necessitates a need for an efficient skill learning method that considers the varying levels of influence that different observational modalities may have.

Diffusion models [3] have been extensively leveraged for learning robot skills from demonstrations [4]. The current methods for training diffusion policies, jointly condition the action diffusion process x on all M observational modalities $\pmb { y } ^ { 1 : M }$ for every task [4]. This is a monolithic joint conditioning approach - “when all you have is a hammer, everything looks like a nail”. Existing diffusion policies do not flexibly accommodate the differing influence of various observational modalities. We empirically show that this joint conditioning approach hurts the sample complexity of diffusion policies as it is difficult to learn the level of dependence that actions have on the observational modalities with limited data. Further, diffusion policies are sensitive to small distribution shifts in any modality $\pmb { y } ^ { 1 : M }$ that they conditions upon, and cannot ‘de-prioritize’ the modality susceptible to noise, similar to humans [2]. Incorporating robustness to such shifts require a prohibitively large amount of data when the observation modalities are high-dimensional. To that end, we propose a novel policy formulation called Factorized Diffusion Policies (FDP) for enabling prioritization of observational modalities during policy learning.

At its core, FDP learns a diffusion base policy using k $( k < M )$ input modalities $\boldsymbol { y } ^ { 1 : k }$ to be prioritized, followed by a diffusion residual policy that learns the noise or score residual conditioned on all modalities $\pmb { y } ^ { 1 : M }$ . We provide a probabilistic formulation to the residual from the first principles, and also develop a novel architecture for efficiently learning it. The base and residual models are then composed to obtain samples from the full conditional action distribution $p ( \pmb { x } | \pmb { y } ^ { 1 : M } )$ . By enabling modality prioritization, FDP introduces flexibility in learning diffusion policies with the inclusion of prioritization order of observational modalities in the hyperparameter search space. Through extensive experiments across visual (vision), point-cloud (pcd), proprioceptive (prop), environment-state (env-state) and tactile (tactile) modalities, we show that leveraging this added flexibility results in more performant and robust diffusion policies. Our contributions are as follows.

• We propose FDP, a novel policy formulation that enables observational modalities to have differing influence on the action diffusion process. We mathematically derive a framework to split the jointly conditioned policy into a base policy learned with prioritized modalities and a residual policy learned with all modalities.

• We show the merits of modality prioritization through extensive experimentation across visual, point-cloud, proprioceptive and tactile observational inputs on several simulated benchmarks such as RLBench (15% ↑), Adroit (10% ↑), Robomimic (10% ↑) and M3L (20% ↑). We thoroughly analyze our method and present ablations.

![](images/7294a5441284fbda3658b65be4e63985a7f35ee65f016a304d61fb8a43c11937.jpg)  
Fig. 1. Policy learning using FDP with different prioritization orders. In FDP, we train a base and a residual policy by prioritizing over different observation modalities for the same task. We demonstrate this approach with different combinations of observation modalities such as prop>vision, and vision>tactile among others. FDP results in more performant policies (15% ↑) that are robust to distractors and camera occlusions (40% ↑).

• Our real-world experiments demonstrate the usefulness of FDP in learning policies that are robust to visual distribution shifts (40% ↑). Policies learned using the prioritization order of prop>vision were not only robust to distractors but also to camera occlusions (5× ↑), where diffusion policies failed catastrophically.

## II. RELEVANT WORK

Sample complexity and Generalization. Despite recent scaling efforts [5]–[7], the collection of multimodal data is difficult in robotics and the number of variations of tasks is unbounded. Hence several works have tried to improve the sample complexity of learning new skills. Several works leverage compositionality for solving novel combinations of tasks with existing solutions, such as composing learned constraints to generalize to new task combinations in manipulation [8] and planning [9], composing distributions across heterogeneous modalities for tool use [10] and sequencing skills for long horizon problems [11]–[13]. The most relevant to our work is PoCo [10], that composes single task policies conditioned on different modalities. However, PoCo composes pre-learned policies for the same task and requires manual tuning of the compositional weights. Instead FDP learns the residual to be composed with the base prioritized policy, using the same data and requiring no manual tuning. Augmentation [14], [15] or retrievalbased [16]–[18] approaches of addressing sample-complexity add a substantial computational and data overhead and are orthogonal to our proposed algorithmic improvement, which may further benefit from them.

Residual Learning and Adapters. Residual reinforcement learning has been used to improve the performance of behavior cloning policies through interaction with the environment [19]. These methods learn a residual over the action predicted by the behavior cloned policy through controlled exploration strategies [20], uncertainty-aware exploration [21], or as closed-loop corrections for chunked action predictions [22], for maximizing the expected returns. Jiang et al. [23] show sim-to-real transfer by learning a supervised residual for human feedback on real world rollouts of policies learned in simulation, maximizing the likelihood of the correction applied. In FDP, the effect of less-influential modalities is captured by learning a residual over a policy trained on prioritized modalities. We theoretically derive this residual within the framework of diffusion and scorebased models. Our work is also similar to Q-Adapter [24] in terms of learning a residual using an adapter, but does not necessitate a base foundation model for learning the adapter.

Several works in robotics have used adapters such as LoRA [25] and ControlNet [26] for fine-tuning multi-task or foundation-models [5], [6] on downstream tasks. Prior work has also explored continual adaptation of multi-task policies to novel tasks [27], [28] and adapting pretrained vision [29] and vision-language models [30] for robotic manipulation. Diff-Control from Liu et al. [31] learns a ControlNet with input as the previous action chunk over a diffusion policy base to impart stateful behavior to the policy. Interestingly, FDP can be leveraged to reach a similar learning formulation as Diff-Control, with the residual learned for the previous action chunk instead of a modality.

## III. BACKGROUND

Diffusion Models. Gaussian diffusion models [32] learn the reverse diffusion kernel $p _ { \pmb { \theta } } ( \pmb { x } _ { t - 1 } | \pmb { x } _ { t } )$ for a fixed forward kernel that adds Gaussian noise at each step $p ( \pmb { x } _ { t } | \pmb { x } _ { t - 1 } ) =$ $\mathcal { N } ( { \pmb x } _ { t } ; \sqrt { \alpha _ { t } } { \pmb x } _ { t - 1 } , ( 1 - \alpha _ { t } ) { \pmb \mathcal { T } } )$ , such that $p ( { \pmb x } _ { T } ) \approx \mathcal { N } ( 0 , \mathcal { T } )$ Here, $t < = T$ is the diffusion time step and $\alpha _ { t }$ is the noise schedule. In practice, the models learn a reparametrized form corresponding to the noise added to the input $\epsilon _ { \theta } ( x _ { t } , t )$ [3].

Score-based Models. Song et al. [33] presented a unified framework showing that both diffusion models [3], [32] and score-based models [34] can be interpreted as discretizations of different forward stochastic differential equations (SDEs). The latter learn the score $\nabla _ { \pmb { x } _ { t } } \log p _ { \sigma _ { t } } ( \pmb { x } _ { t } )$ at different noise scales $\sigma _ { t }$ required for sampling from the data distribution. Explicit Score Matching (ESM) [35], [36] was proposed to estimate the score by minimizing the Fisher divergence with the Gaussian-smoothed data distribution $\begin{array} { r } { p _ { \sigma _ { t } } ( \pmb { x } _ { t } ) { = } \int p ( \pmb { x } ) \mathcal { N } ( \pmb { x } _ { t } ; \pmb { x } , \sigma _ { t } ^ { 2 } I ) d } \end{array}$ x. Denoising Score Matching (DSM) alleviates the computational difficulties of ESM [36], [37], and is shown in Equation 1, where $s _ { \pmb { \theta } } ( \pmb { x } _ { t } , \sigma _ { t } )$ abbreviated to $s _ { \pmb { \theta } } ( \pmb { x } _ { t } )$ , represents the learned score model.

![](images/028ebe9c71e12e669bb86fac1942f6befbc5517048b2609162b947a54d8d91ef.jpg)  
Fig. 2. Architectural representations for [a] diffusion policy that jointly conditions on all observational modalities, [b] straightforward composition of the score outputs from $\pi _ { \mathrm { b a s e } }$ and $\pi _ { \mathrm { r e s } }$ and [c] FDP architecture with block-wise composition with a layer Z applied on $\pi _ { \mathrm { r e s } }$

$$
\begin{array}{l} \mathcal {J} _ {\sigma_ {t}} (\boldsymbol {\theta}) \stackrel {{\text { ESM }}} {{=}} \mathbb {E} _ {p _ {\sigma_ {t}} (\boldsymbol {x} _ {t})} \left[ \frac {1}{2} \| \nabla_ {\boldsymbol {x} _ {t}} \log p _ {\sigma_ {t}} (\boldsymbol {x} _ {t}) - s _ {\boldsymbol {\theta}} (\boldsymbol {x} _ {t}) \| _ {2} ^ {2} \right] \\ \stackrel {{\text { DSM}}} {{=}} \mathbb {E} _ {p _ {\sigma_ {t}} (\boldsymbol {x}, \boldsymbol {x} _ {t})} \left[ \frac {1}{2} \| \nabla_ {\boldsymbol {x} _ {t}} \log p _ {\sigma_ {t}} (\boldsymbol {x} _ {t} | \boldsymbol {x}) - s _ {\boldsymbol {\theta}} (\boldsymbol {x} _ {t}) \| _ {2} ^ {2} \right] + C \end{array}\tag{1}
$$

Diffusion models use $\begin{array} { r l } { \mathbf { a } } & { { } \left( 1 - \bar { \alpha _ { t } } \right) } \end{array}$ weighted DSM objective along with a forward transition kernel $p _ { \bar { \alpha } _ { t } } ( { \pmb x } _ { t } | { \pmb x } ) { = } \mathcal { N } ( { \pmb x } _ { t } ; \sqrt { \bar { \alpha _ { t } } } { \pmb x } , ( 1 { - } \bar { \alpha _ { t } } ) I )$ with discrete time and $\begin{array} { r } { \bar { \alpha } _ { i } = \prod _ { j = 1 } ^ { i } \alpha _ { j } } \end{array}$ , yielding the simplified diffusion loss from Ho et al. [3]. Score-based model typically use $\mathcal { N } ( \pmb { x } _ { t } ; \pmb { x } , \sigma _ { t } ^ { 2 } I )$ , where $\alpha _ { t }$ and $\sigma _ { t }$ are respective noise scales. In the simplified case, an optimal diffusion model is related to the score of the α -diffused data distribution by $- \epsilon _ { \theta } ^ { * } ( x _ { t } , t ) / \sqrt { 1 - \bar { \alpha } _ { t } } \quad \stackrel { d e f } { = } \quad s _ { \theta } ^ { * } ( x _ { t } ) { = } \nabla _ { x _ { t } } \log p ( x _ { t } )$ [33]. Typically, diffusion models generate samples via progressive denoising through the reverse diffusion process [3], while score-matching models sample from the data distribution using Langevin dynamics [38].

Classifier Guided Sampling. Dhariwal and Nichol [39] obtain conditional samples from an unconditional diffusion model trained on x using Bayes’ theorem. We can sample from a class y by decomposing the conditional score at time step t into the unconditional score and the classifier gradient $\begin{array} { r l r } { \nabla _ { \pmb { x } _ { t } } \log p ( \pmb { x } _ { t } | \pmb { y } ) = } & { { } \nabla _ { \pmb { x } _ { t } } \log p ( \pmb { x } _ { t } ; \pmb { \theta } ) + \nabla _ { \pmb { x } _ { t } } \log p ( \pmb { y } | \pmb { x } _ { t } ; \pmb { \phi } ) } & { } \end{array}$ However, classifier guidance needs an explicit classifier trained on noisy samples to estimate the gradients [40].

## IV. METHODOLOGY

Assume that we have robot demonstrations $D = \{ ( { \pmb x } , { \pmb y } ) _ { i } \}$ where $\begin{array} { r c l } { i } & { = } & { 1 . . N , } \end{array}$ , consisting of actions x and different observational modalities $\overset { \mathrm { ~ - ~ } } { \boldsymbol { y } } ^ { 1 : M }$ , such as images, point clouds, proprioception or tactile. Let $\boldsymbol { y } ^ { 1 : k }$ be the prioritized observational modalities of the M total modalities, where $\pmb { y } ^ { 1 : k } \equiv \pmb { y } ^ { 1 } , . . , \pmb { y } ^ { k }$ and $k { < } M$ . To sample actions x conditioned on all $\pmb { y } ^ { 1 : M }$ , we need to estimate the score $\nabla _ { \pmb { x } _ { t } } \log p ( \pmb { x } _ { t } | \pmb { y } ^ { 1 : M } )$ . Taking a cue from classifier guidance [39], we decouple the observational modalities utilizing Bayes’ theorem to obtain Equation 2. For scores to be valid, observational modalities $\pmb { y } ^ { 1 : \dot { M } }$ can be noised with a Gaussian kernel $\mathcal { N } ( \tilde { \pmb { y } } ; \pmb { y } , \tau ^ { 2 } I )$ of variance $\tau ^ { 2 }$ that is small enough such that $p _ { \tau } ( \tilde { \pmb { y } } ) \approx p ( \pmb { y } )$ , where we drop the notation τ going forward. Actions x are noised with the kernel $p _ { \bar { \alpha } _ { t } } ( \pmb { x } _ { t } | \pmb { x } )$ .

$$
\begin{array}{l} \boldsymbol {s} ^ {*} (\boldsymbol {x} _ {t}, \boldsymbol {y} ^ {1: M}) = \nabla_ {\boldsymbol {x} _ {t}} \log p (\boldsymbol {x} _ {t} | \boldsymbol {y} ^ {1: M}) \\ = \nabla_ {\boldsymbol {x} _ {t}} \log p (\boldsymbol {x} _ {t} | \boldsymbol {y} ^ {1: k}) + \nabla_ {\boldsymbol {x} _ {t}} \log p (\boldsymbol {y} ^ {k + 1: M} | \boldsymbol {x} _ {t}, \boldsymbol {y} ^ {1: k}) \end{array}\tag{2}
$$

The first score term $\nabla _ { \pmb { x } _ { t } }$ log $p ( \pmb { x } _ { t } | \pmb { y } ^ { 1 : k } )$ corresponds to a policy $\boldsymbol { s } _ { \boldsymbol { \theta } } ( \boldsymbol { x } _ { t } , \boldsymbol { y } ^ { 1 : k } )$ that would be obtained on training with just the modalities $\boldsymbol { y } ^ { 1 : k }$ , referred to as $\pi _ { \mathrm { b a s e } }$ going forward. The effect of the other modalities is captured in the second score term $\nabla _ { \pmb { x } _ { t } } \log p ( \pmb { y } ^ { k + 1 : M } | \pmb { x } _ { t } , \pmb { y } ^ { 1 : k } )$ . However, explicitly training a classifier $\ddot { p ( { \pmb y } ^ { k + 1 : \hat { M } } | { \pmb x } _ { t } , { \pmb y } ^ { 1 : \hat { k } } ) }$ as suggested by Dhariwal and Nichol [39] is impractical due to the high dimensionality and continuity of multiple observational modalities $\pmb { y } ^ { 1 : M }$ such as images and tactile. Hence, we employ explicit score matching $\mathcal { I } _ { \alpha _ { t } } ( \phi )$ [35], [36] as shown in Equation 3.

$$
\mathbb {E} _ {p _ {\alpha_ {t}} (\boldsymbol {x} _ {t}, \boldsymbol {y} ^ {1: M})} \left[ \frac {1}{2} \left\| \begin{array}{l} \nabla_ {\boldsymbol {x} _ {t}} \log p _ {\alpha_ {t}} (\boldsymbol {y} ^ {k + 1: M} | \boldsymbol {x} _ {t}, \boldsymbol {y} ^ {1: k}) \\ - \boldsymbol {s} _ {\phi} (\boldsymbol {x} _ {t}, \boldsymbol {y} ^ {1: M}) \end{array} \right\| _ {2} ^ {2} \right]\tag{3}
$$

Due to the high computational complexity of estimating the empirical scores such as $\nabla _ { \pmb { x } _ { t } } \log \hat { p } _ { \alpha _ { t } } ( \pmb { y } ^ { \bar { k } + 1 : M } | \pmb { x } _ { t } , \pmb { y } ^ { 1 : k } )$ Chao et al. [37] derive the denoising likelihood score matching (DLSM) objective for conditional distributions, which forms the basis for our next result.

Theorem 1: Explicit score matching for $\mathbf { \delta } _ { \mathbf { \boldsymbol { s } } _ { \phi } \left( \mathbf { \boldsymbol { x } } _ { t } , \mathbf { \boldsymbol { y } } ^ { 1 : M } \right) }$ in Equation 3 is equivalent to the objective $\mathcal { T } _ { \alpha _ { t } } ^ { r e s } ( \phi )$

$$
\mathbb{E}_{\substack{p_{\alpha_{t}}(\boldsymbol {x},\boldsymbol{x}_{t},\\ \boldsymbol{y}^{1:M})}}\left[ \frac{1}{2}\left\| \nabla_{\boldsymbol{x}_{t}}\log p_{\alpha_{t}}(\boldsymbol{x}_{t}|\boldsymbol {x}) - \begin{array}{c}\boldsymbol{s}^{*}(\boldsymbol{x}_{t},\boldsymbol{y}^{1:k})\\ -\boldsymbol{s}_{\phi}(\boldsymbol{x}_{t},\boldsymbol{y}^{1:M}) \end{array} \right\|_{2}^{2}\right]
$$

Here $\mathbf { { \sigma } } _ { \mathbf { { s } } ^ { \ast } \left( \mathbf { { x } } _ { t } , \mathbf { { y } } ^ { 1 : k } \right) }$ is the the frozen optimal score model for $\nabla _ { \pmb { x } _ { t } } \log p ( \pmb { x } _ { t } | \pmb { y } ^ { 1 : k } )$ , approximated using a learned model π<sub>base</sub> : $\mathbf { \Psi } : s _ { \theta } ( \mathbf { \boldsymbol { x } } _ { t } , \mathbf { \boldsymbol { y } } ^ { 1 : k } )$ in practice.

We believe FDP to be the first work to prove this equivalence for an arbitrary number of conditionals and directly learn the classifier guidance in a highdimensional setting for the purposes of policy learning. Interestingly, comparison of $\mathcal { I } _ { \alpha _ { t } } ^ { r e s } ( \phi )$ with DSM shown in Equation 1 reveals that the effect of $\scriptstyle { \boldsymbol { y } } ^ { k + 1 : M }$ captured through $\mathbf { \delta } _ { \mathbf { \boldsymbol { s } } _ { \phi } ( \mathbf { \boldsymbol { x } } _ { t } , \boldsymbol { y } ^ { 1 : M } ) }$ can be learned as a residual to $\begin{array} { r l r } { \pi _ { \mathrm { b a s e } } } & { { } : } & { s _ { \theta } ( \pmb { x } _ { t } , \pmb { y } ^ { 1 : k } ) } \end{array}$ the policy learned with just the modalities $\boldsymbol { y } ^ { 1 : k }$ . Thus we refer to $\mathbf { \delta } _ { \mathbf { \boldsymbol { s } } _ { \phi } ( \mathbf { \boldsymbol { x } } _ { t } , \mathbf { \boldsymbol { y } } ^ { 1 : M } ) }$ as $\pi _ { \mathrm { r e s } } .$ Essentially, by factorizing the score of the full-conditional action, FDP learns $\pi _ { \mathbf { b a s e } }$ using $\boldsymbol { y } ^ { 1 : k }$ , and then learns $\pi _ { \mathrm { r e s } }$ using $\pmb { y } ^ { 1 : M }$ and a frozen π<sub>base</sub>. This two-phase training prioritizes modalities $\boldsymbol { y } ^ { 1 : k }$ over $\scriptstyle { \boldsymbol { y } } ^ { k + 1 : M }$

A concise proof of Theorem 1 is presented, and a detailed version can be found on our website. We substitute $\mathbf { \delta } _ { \mathbf { \boldsymbol { s } } _ { \phi } ( \mathbf { \boldsymbol { x } } _ { t } , \mathbf { \boldsymbol { y } } ^ { 1 : M } ) }$ as • for brevity. The inner product obtained by opening the square in Equation 3 can be simplified as-

$$
\begin{array}{l} \mathbb {E} _ {p _ {\alpha} (\boldsymbol {x} _ {t}, \boldsymbol {y} ^ {1: M})} \left[ \left\langle \bullet , \nabla_ {\boldsymbol {x} _ {t}} \log p _ {\alpha_ {t}} \left(\boldsymbol {y} ^ {k + 1: M} \mid \boldsymbol {x} _ {t}, \boldsymbol {y} ^ {1: k}\right) \right\rangle \right] = \\ \mathbb {E} _ {\substack {p _ {\alpha} (\boldsymbol {x} _ {t}, \\ \boldsymbol {y} ^ {1: M})}} \left[ \left\langle \bullet , \nabla_ {\boldsymbol {x} _ {t}} \log p _ {\alpha_ {t}} (\boldsymbol {x} _ {t} | \boldsymbol {y} ^ {1: M}) - \nabla_ {\boldsymbol {x} _ {t}} \log p _ {\alpha_ {t}} (\boldsymbol {x} _ {t} | \boldsymbol {y} ^ {1: k}) \right\rangle \right] \end{array} \tag{5}
$$

Further simplifying the inner product with the first term-

$$
\begin{array}{l} \mathbb {E} _ {p _ {\alpha_ {t}} (\boldsymbol {x} _ {t}, \boldsymbol {y} ^ {1: M})} \left[ \langle \boldsymbol {s} _ {\phi} (\boldsymbol {x} _ {t}, \boldsymbol {y} ^ {1: M}), \nabla_ {\boldsymbol {x} _ {t}} \log p _ {\alpha_ {t}} (\boldsymbol {x} _ {t} | \boldsymbol {y} ^ {1: M}) \rangle \right] \\ = \mathbb {E} _ {p (\boldsymbol {y} ^ {1: M})} \int_ {\boldsymbol {x} _ {t}} p _ {\alpha} (\boldsymbol {x} _ {t} | \boldsymbol {y} ^ {1: M}) \langle \bullet , \frac {\nabla_ {\boldsymbol {x} _ {t}} p _ {\alpha_ {t}} (\boldsymbol {x} _ {t} | \boldsymbol {y} ^ {1 : M})}{p _ {\alpha_ {t}} (\boldsymbol {x} _ {t} | \boldsymbol {y} ^ {1 : M})} \rangle d \boldsymbol {x} _ {t} \\ = \mathbb {E} _ {p (\boldsymbol {y} ^ {1: M})} \int_ {\boldsymbol {x} _ {t}} \langle \bullet , \nabla_ {\boldsymbol {x} _ {t}} \int_ {\boldsymbol {x} _ {0}} p _ {0} (\boldsymbol {x} _ {0} | \boldsymbol {y} ^ {1: M}) p _ {\alpha_ {t}} (\boldsymbol {x} _ {t} | \boldsymbol {x} _ {0}, \boldsymbol {y} ^ {1: M}) d \boldsymbol {x} _ {0} \rangle d \boldsymbol {x} _ {t} \\ = \mathbb {E} _ {p _ {\alpha_ {t}} (\boldsymbol {x} _ {0}, \boldsymbol {x} _ {t}, \boldsymbol {y} ^ {1: M})} \left[ \langle \bullet , \nabla_ {\boldsymbol {x} _ {t}} \log p _ {\alpha_ {t}} (\boldsymbol {x} _ {t} | \boldsymbol {x} _ {0}) \rangle \right] \end{array} \tag {6}
$$

Note that $\mathbb { E } _ { p _ { \alpha _ { t } } ( \pmb { x } _ { t } , \mathbf { \Delta } } \left[ \frac { 1 } { 2 } | | \nabla _ { \pmb { x } _ { t } } \log p _ { \alpha _ { t } } ( \pmb { y } ^ { k + 1 : M } | \pmb { x } _ { t } , \pmb { y } ^ { 1 : k } ) | | _ { 2 } ^ { 2 } \right]$ is a constant. Substituting results obtained in Equations 5 and 6 back in Equation 3, and adding the constant term $\mathbb { E } _ { p _ { \alpha _ { t } } ( \mathbf { \Delta x } _ { t } , \mathbf { \Delta x } ) } \left. \frac { 1 } { 2 } | | \nabla _ { \mathbf { x } _ { t } } \log p _ { \alpha _ { t } } ( \mathbf { \Delta x } _ { t } | \mathbf { \Delta y } ^ { 1 : k } ) - \nabla _ { \mathbf { x } _ { t } } \log p _ { \alpha } ( \mathbf { \Delta x } _ { t } | \mathbf { \Delta x } _ { 0 } ) | | _ { 2 } ^ { 2 } \right.$ we complete the proof for Theorem 1. Hence, we prove that the ESM $\mathcal { I } _ { \alpha _ { t } } ( \phi )$ in Equation 3 is equivalent to minimizing the objective $\mathcal { T } _ { \alpha _ { t } } ^ { r e s } ( \phi )$ presented in Theorem 1, differing up to a constant.

Theorem 1 implies that the effect of de-prioritized modalities $\scriptstyle { \boldsymbol { y } } ^ { k + 1 : M }$ can be learned as a residual over the prioritized modalities $\boldsymbol { y } ^ { 1 : k }$ . From a different lens, $\pi _ { \mathrm { r e s } }$ effectively learns the classifier guidance required to sample from $\pi _ { \mathrm { b a s e } }$ further conditioned on $\scriptstyle { \boldsymbol { y } } ^ { k + 1 : M }$ . Learning $\pi _ { \mathrm { r e s } }$ as a residual of $\pi _ { \mathbf { b a s e } }$ ensures that the policy does not overfit modalities $\scriptstyle { \boldsymbol { y } } ^ { k + 1 : M }$ but only learns correlations to bridge the error arising from $\pi _ { \mathbf { b a s e } }$ trained on the prioritized modalities $\boldsymbol { y } ^ { 1 : k }$ Hence, policies learned in this factorized way are naturally robust to distribution shifts in $\scriptstyle { \boldsymbol { y } } ^ { k + 1 : M }$ . Moreover, explicit prioritization of $y ^ { 1 : k }$ by training $\pi _ { \mathrm { b a s e } }$ prior to learning the residual leads to sample efficiency, as the model learns correlations with the stronger modality without having to attend to other modalities.

Factorizing Diffusion Policies. As developed in Section III, Theorem 1 applies to diffusion models. Resolving $\nabla _ { { \pmb x } _ { t } } \log p _ { \alpha _ { t } } ( { \pmb x } _ { t } | { \pmb x } )$ to $- \epsilon _ { 0 } / \sqrt { 1 - \bar { \alpha _ { t } } }$ and replacing ${ \boldsymbol { s } } _ { \theta } ^ { * } ( { \boldsymbol { x } } _ { t } )$ with $- \epsilon _ { \theta } ( { \pmb x } _ { t } , t ) / \sqrt { 1 - \bar { \alpha _ { t } } } ,$ , we get the simplified diffusion losses for $\pi _ { \mathrm { b a s e } }$ and π<sub>res</sub>.

$$
\mathcal{L}_{base}^{t}(\boldsymbol {\theta}) = \mathbb{E}_{p(\boldsymbol{x}_{0},\boldsymbol{y}^{1:k})\mathcal{N}(\epsilon_{0};0,\mathcal{I})}\left[\left\| \boldsymbol{\epsilon}_{0} - \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_{t},\boldsymbol{y}^{1:k},t)\right\|_{2}^{2}\right]\\ \mathcal{L}_{res}^{t}(\phi) = \mathbb{E}_{\substack{\boldsymbol{x}_{0},\boldsymbol{y}^{1:M}\sim p\\ \boldsymbol{\epsilon}_{0}\sim \mathcal{N}(0,\mathcal{I})}}\left[\frac{1}{2}\left\| \begin{array}{c}\boldsymbol{\epsilon}_{0} - \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_{t},\boldsymbol{y}^{1:k},t)\\ -\boldsymbol{\epsilon}_{\boldsymbol{\phi}}(\boldsymbol{x}_{t},\boldsymbol{y}^{1:M},t) \end{array} \right\|_{2}^{2}\right]\tag{7}
$$

From the perspective of diffusion models, $\pi _ { \mathrm { b a s e } }$ maximizes a reweighted lower bound on the data likelihood only considering the prioritized k modalities, while $\pi _ { \mathrm { r e s } }$ learns a residual over $\pi _ { \mathrm { b a s e } }$ to maximize it for demonstration data with all the modalities included, thus learning their residual effect. Since diffusion models are trained on discrete time steps, $\pi _ { \mathrm { r e s } }$ is learned on the same time discretization as used for $\pi _ { \mathrm { b a s e } } .$ Once trained, actions can be sampled from the conditional distribution $p ( \pmb { x } | \pmb { y } ^ { 1 : M } )$ using reverse diffusion [3] on the composition [41] of $\pi _ { \mathrm { b a s e } }$ and $\pi _ { \mathrm { r e s } }$ :

$$
\begin{array}{l} \boldsymbol {x} _ {t - 1} \sim \mathcal {N} \left(\boldsymbol {x} _ {t}; \frac {1}{\sqrt {\alpha_ {t}}} \left(\boldsymbol {x} _ {t} - \frac {1 - \alpha_ {t}}{\sqrt {1 - \bar {\alpha} _ {t}}} \boldsymbol {\epsilon} (\boldsymbol {x} _ {t}, \boldsymbol {y} ^ {1: M}, t)\right), \sqrt {1 - \alpha_ {t}} \mathcal {I}\right) \\ \boldsymbol {\epsilon} (\boldsymbol {x} _ {t}, \boldsymbol {y} ^ {1: M}, t) = \boldsymbol {\epsilon} _ {\boldsymbol {\theta}} (\boldsymbol {x} _ {t}, \boldsymbol {y} ^ {1: k}, t) + \boldsymbol {\epsilon} _ {\phi} (\boldsymbol {x} _ {t}, \boldsymbol {y} ^ {1: M}, t) \end{array}\tag{8}
$$

Architectural Implementation of FDP. The models π and $\pi _ { \mathrm { r e s } }$ can be instantiated using standard architectures such as UNet [42] or DiT [43]. Inspired from the late stage score-composition, we propose a more integrated way to compose $\pi _ { \mathrm { b a s e } }$ and $\pi _ { \mathrm { r e s } } .$ , as shown in Figure 2. Instead of learning a residual for the final score output, $\pi _ { \mathrm { r e s } }$ learns the blockwise residual over the intermediate outputs of the frozen $\pi _ { \mathrm { b a s e } } .$ . Specifically, let $\mathcal { F } _ { \mathrm { b a s e } } ^ { i }$ and $\mathcal { F } _ { \mathrm { r e s } } ^ { i }$ denote the ith DiT block outputs of the base and residual models, respectively. Then the composed output at level i can be written as $\mathbf { \bar { \mathcal { F } } } _ { b a s e } ^ { i } ( { \pmb x } ^ { \prime } , { \pmb y } ^ { \prime 1 : k } ) \mathbf { \dot { + } } \mathcal { Z } ( \mathcal { F } _ { r e s } ^ { i } ( { \pmb x } ^ { \prime } , { \pmb y } ^ { \prime 1 : M } ) )$ , where $x ^ { \prime }$ and $y ^ { \prime 1 : M }$ are layer inputs. Similar to Zhang et al. [26], Z is a zero-initialized layer to avoid harmful updates at the start of the training and to ensure that gradient updates to the residual model improve the predictions of the composed model over $\pi _ { \mathrm { b a s e } }$ . This architecture enables a simplified training objective from Ho et al. [3] for the residual model. Our residual model is structured following the Vision Transformers architecture [44]. In $\pi _ { \mathrm { r e s } } ,$ all observational modalities are passed through self-attention layers after encoding into a single embedding.

All transformer-based models are trained over 2000 epochs with a batch size of 64 for visual tasks and for 3000 epochs with a batch size of 256 for low-dimensional tasks. All models except VLAs are trained on NVIDIA A5000 GPUs, with training times ranging from 6-12 hours depending on model size and the number of camera inputs. Model prop $\pi _ { b a s e }$ consists of ∼30M parameters, while vis $\pi _ { r e s }$ with two camera image inputs is ∼55M parameters large. Our current implementations support an action prediction latency of ∼50ms for transformer-based diffusion policy baseline, ∼100ms for UNet [4] and output composition of models as shown in Figure 2 [b], and ∼150ms for FDP model in [c].

## V. SIMULATION EXPERIMENTS

We train and evaluate FDP and related baselines on 14 tasks from RLBench [45] along with their distractor variants, 4 tasks from Adroit [46], 4 tasks from Robomimic [47], and the visuo-tactile insertion task from M3L [48]. RLBench provides a diverse suite of visuomotor manipulation tasks with joint positions as the action space and a built-in planner for demonstration collection. Within RLBench, we evaluate prioritization orders prop≶vision and pcd≶vision for FDP. The Adroit benchmark contains high-dimensional hand manipulation tasks performed with a 24-DoF anthropomorphic hand. Environments in Robomimic use an action representation defined by changes in end-effector position and orientation (axis–angle). For both Adroit and Robomimic, we explore the prioritization orders prop≶env-state. Finally, the M3L environment requires precise insertion of differently shaped pegs into holes randomly placed on a surface, using a single RGB camera and two tactile sensors for perception and ∆xyz as the action representation.

![](images/5b582c1d3ae34bfc60e59ec439aa5ef1e6d45a1b0a51877838cd5d18d1f43ec2.jpg)

![](images/f8802e6952409e8fcf7cb3348618596a6c7c35e45ba9bbd2b11e41030ea5d0af.jpg)  
Fig. 3. We modify the original RLBench environments to introduce color variations and add distractors.  
Fig. 4. Mean (with Std. Dev.) Performance for all models across 10-50-100 demonstrations on RLBench and Robomimic. Prioritization of modalities using FDP enables strong performance gains, especially in low-data regimes.

Demonstrations for M3L are collected using an expert RL policy proposed by Sferrazza et al. [48]. On M3L we evaluate the performance of vision≶tactile prioritizations using FDP against jointly conditioned diffusion policy. Unless otherwise noted, reported results are averaged over 300 rollouts. Additional experimental details are provided on our project webpage. https://fdp-policy.github. io/fdp-policy/.

Baselines. For evaluation of sample efficiency in visuomotor tasks, we compare against several approaches that differ in the way in which they model generative policy learning. However, for all approaches, we choose DiT-small (∼90M) [43] as our model architecture. We implement

![](images/f7490d409671a0fc34488f134aca1b968feae11a35da9c26ec03c8b3f74d82e1.jpg)  
Fig. 5. Performance of FDP (prop>vision) and DP-DiT across original, color-variant and distractor environments. We also fine-tune on 5 additional demos collected in the modified environments. Note the strong performance of FDP (prop>vision) in the color-variant and distractor experiments.

Diffusion Policy [4] using DiT, referred to as DP-DiT in our results. For comparison, we also include UNet [42] implemented by Chi et al. [4] in our baselines as DP-UNet. We reformulate POCO [10] to compose observational modalities. We train π and $\pi _ { \mathrm { r e s } }$ models independently, prior to sampling from the composed distribution [41] using $\overline { { \epsilon ( x _ { t } , y ^ { 1 : M } , \bar { t } ) } } { = } \overline { { \epsilon _ { \theta } ( x _ { t } , y ^ { 1 : k } , t ) + \lambda * \epsilon _ { \phi } ( x _ { t } , y ^ { 1 : M } , t ) } }$ Here, $\lambda = 0 . 1$ based on POCO’s ablations [10]. We also report results for classifier-free guidance [49] as CFG, where we train a single model and switch out the weaker modality with a probability of 0.2. We then sample using ${ \epsilon } ( \boldsymbol { x } _ { t } , \boldsymbol { y } ^ { 1 : M } , t ) =$ $\lambda _ { 1 } * \epsilon _ { \theta } ( x _ { t } , y ^ { 1 : k } , y ^ { k + 1 : M } , t ) + \lambda _ { 2 } * \epsilon _ { \theta } ( x _ { t } , y ^ { 1 : k } , \phi , t )$ , where we set $\lambda _ { 1 } ~ = ~ 1 . 1$ and $\lambda _ { 2 } ~ = ~ 0 . 1$ , as suggested by [49]. We also fine-tune a 450M parameter vision-language action model SmolVLA [7] for at least 40k steps, evaluating its sample-efficiency and robustness on selected RLBench tasks. For real-world and distractor experiments in simulation, we compare against DP-DiT.

## A. Sample Efficiency Gain Through Modality Prioritization.

FDP enables us to include the prioritization order of modalities for tasks as a hyperparameter. Prioritizing either vision, env-state, or prop modalities using FDP consistently outperforms joint conditioning across a wide range of visuomotor, state-based and tactile tasks in RLBench, Robomimic, Adroit (Fig. 6) and M3L (Table 8) respectively. In RLBench, modality prioritization with FDP achieves on average a 15% higher success rate with 10 and 50 demonstrations, and a 10% higher success rate with 100 demonstrations, compared to the strongest baseline. We also observe clear gains with the prioritization order of prop>env-state in Robomimic and Adroit tasks. Fig. 4 illustrates how performance scales with increasing demonstrations in RLBench and Robomimic. Prioritization is especially beneficial in low-data regimes, where a jointly conditioned model lacks sufficient data to learn the correct modality weighting. FDP enforces conditioning on the most essential modality, leading to stronger overall performance. SmolVLA fails drastically at SweepToDustpan (Figure 7) that requires precise spatial motions to sweep all the dirt particles into the dustpan. This highlights that there is potential for VLAs to improve beyond table-top tasks where algorithmic approaches like FDP do better. For M3L, we observe in Table 8 that the prioritization order of vision>tactile outperforms the joint-conditioning approach by over 20% at 100 and 200 demonstrations. Several works [22], [50] have fine-tuned a BC policy using reinforcement learning, and FDP can serve as a performant prior policy for further finetuning. These results clearly show that FDP leads to more performant policies across various observational modalities, especially in low-data regimes.

## B. Robustness Gain Through Modality Prioritization.

Prioritization prevents the model from developing spurious correlations by learning a residual policy for modalities with limited influence on robot actions. To test this, we evaluate

![](images/b4705bd2a0e2f38de7605bdfab628432144462a40ff074ab2e5b6a2b1c9ecef0.jpg)  
(a) RLBench tasks with 10 demonstrations

![](images/5b1c52fe4a422a48b85522118efdf119292e01869093f59b5c5582b413fa8f07.jpg)  
(b) RLBench tasks with 50 demonstrations

![](images/e9dccd228e3dd037c1773949fa28c5c829308fbc279b0e06d00ec2a1529d1153.jpg)  
(c) Robomimic & Adroit with 10-50 demos  
Fig. 6. Radial plot showing performance of FDP in comparison to DP-DiT. For FDP, the best results obtained using ■ vis>prop and ▲ prop>vision are marked on the plots. These plots show that searching through the modality prioritization space yields improvements in a wide-spectrum of tasks.

![](images/d6f711739c9d19b4e402a280b8443875347634791fab52e021bcfddf60901c93.jpg)  
Fig. 7. Evaluation of robustness gained using prop|pcd>vision. Notably, we see a significant drop in performance of DP-DiT and DP3 on introducing color variations in the task. Unlike FDP, SmoLVLA despite having a strong VLM backbone, fails to adapt to color variations across tasks.

<table><tr><td>Task</td><td>Demos</td><td>DP-DiT</td><td>vision&gt;tactile</td><td>tactile&gt;vision</td></tr><tr><td rowspan="2">square peg</td><td>100</td><td>22</td><td>48</td><td>28</td></tr><tr><td>200</td><td>52</td><td>72</td><td>24</td></tr><tr><td rowspan="2">triangle peg</td><td>100</td><td>14</td><td>42</td><td>10</td></tr><tr><td>200</td><td>28</td><td>50</td><td>16</td></tr><tr><td>All pegs</td><td>200</td><td>6</td><td>26</td><td>4</td></tr></table>

![](images/8c439a4f0d97e135838dff33d23452bb0a04aceb29b34b54d69e8dfe3195ca0a.jpg)  
Fig. 8. Results from the visuo-tactile insertion tasks from M3L [48]. We see a clear benefit with the prioritization order of vision>tactile. The vision modality plays a crucial role in navigating the peg towards the hole [50]. Tactile input becomes useful once the peg is in contact with the contour of the hole, and is learned as a residual over vision π<sub>base</sub>.

FDP (prop>vision) against a jointly conditioned diffusion policy in environments with color variations and clutter. Both DP-DiT and FDP are trained on 100 demonstrations collected in the original environment and evaluated in three settings: the original, color-variant, and cluttered environments as shown in Figure 3. FDP significantly outperforms DP-DiT in both color-variant and cluttered settings by more than 40%. We further collect five demonstrations in each modified environment to study few-shot adaptation to out-of-distribution data. FDP adapts more effectively, improving its performance by 15% on average compared to 10% for DP-DiT. This is achieved by updating only the residual model $\pi _ { \mathrm { r e s } }$ with new demonstrations, which adjusts the conditional distribution on visual modalities $p ( y ^ { v i s }$ $x , y ^ { p r o p } )$ without modifying the full conditional action distribution $p ( x \mid y ^ { p r o p } , y ^ { v i s } )$ . We extend this setting to point clouds $( \mathsf { p c d } \mathsf { > v i s i o n } )$ , where FDP learns a vision $\pi _ { \mathrm { r e s } }$ over DP3 used as $\pi _ { \mathrm { b a s e } }$ [51], and compare it to DP3 with RGB inputs. Point clouds are sample-efficient for policy learning since they encode scene geometry in a single modality [51]. However, our distractor experiments show that FDP with a visual residual over DP3 achieves ∼20% higher performance than DP3 using RGB inputs. These results clearly outline the robustness gain on adopting FDP as the policy formulation.

TABLE I  
BLOCK PICK SUCCESS RATES FOR 10 AND 50 DEMONSTRATIONS.

<table><tr><td>10 demos</td><td>S</td><td>M</td><td>L</td></tr><tr><td>DP-DiT</td><td>29.7 ± 3.1</td><td>12.0 ± 1.0</td><td>3.3 ± 0.6</td></tr><tr><td>FDP (prop &gt; vision)</td><td>73.7 ± 3.8</td><td>21.3 ± 3.5</td><td>6.3 ± 3.1</td></tr><tr><td>FDP (vision &gt; prop)</td><td>18.7 ± 2.3</td><td>6.7 ± 1.2</td><td>3.3 ± 3.1</td></tr><tr><td>50 demos</td><td>S</td><td>M</td><td>L</td></tr><tr><td>DP-DiT</td><td>95.3 ± 3.2</td><td>69.0 ± 7.0</td><td>45.7 ± 7.1</td></tr><tr><td>FDP (prop &gt; vision)</td><td>98.7 ± 1.5</td><td>55.0 ± 2.6</td><td>20.3 ± 3.5</td></tr><tr><td>FDP (vision &gt; prop)</td><td>96.7 ± 1.2</td><td>65.3 ± 8.1</td><td>60.0 ± 2.0</td></tr><tr><td>prop  $\pi_{base}$ </td><td>47.3 ± 2.5</td><td>5.0 ± 1.7</td><td>1.3 ± 1.2</td></tr><tr><td>vision  $\pi_{base}$ </td><td>96.0 ± 2.0</td><td>68.0 ± 6.0</td><td>51.3 ± 8.1</td></tr></table>

## VI. EFFECTS OF PRIORITIZATION

Intuition on the Order of Modality Prioritization As for most hyperparameters, intuition can be developed for the order of modality prioritization. To experimentally test this, we develop 3 variants of the block pick environments-$\mathrm { S } \colon 0 . 1 5 \times 0 . 2 m .$ , M: $0 . 3 5 \times 0 . 4 5 m$ and L: $0 . 5 5 \times 0 . 7 5 m .$ with increasing range of generalization in initial blockplacement. The results are presented in Table I. We observe that prioritization of proprioception out-performs all other models for the S environment while prioritization of vision tends to do better for larger L environment. This also conforms to the intuition of vision playing a diminished role when object placement area is smaller and the motions are repetitive, and a more significant role when the motion varies significantly based on the placement of the object. Tasks that correlate heavily with robot proprioception are not uncommon as the robot is solving them in the first person view, and can move close to the object if required. Our results in the M3L visuo-tactile environment also conform with localize-then-execute strategy [50], where the visual modality plays a more important role in localizing the hole for peg insertion. Learning $\mathrm { a \ t a c t i l e \ } \pi _ { \mathrm { r e s } }$ over the $\mathrm { v i s i o n } \ \pi _ { \mathrm { b a s e } }$ learns residual scores for states where tactile influences the actions of the robot. Finally, the chosen action representation also plays a role, with FDP (prop>) realizing higher success rate for joint-state actions.

TABLE II  
ABLATION RESULTS ON THE OPEN DOOR TASK (10 DEMONSTRATIONS).

<table><tr><td>Model</td><td>Succ. (%)</td><td> $\pi_{base} ep$ </td><td>Succ. (%)</td></tr><tr><td>DiT: small (~33M)</td><td>24.0 ± 7.2</td><td>100 ep</td><td>24.7 ± 6.1</td></tr><tr><td>DiT: base (~130M)</td><td>27.3 ± 5.0</td><td>700 ep</td><td>42.0 ± 5.2</td></tr><tr><td>Score Comp.: [b] Fig. 2</td><td>20.7 ± 8.3</td><td>1500 ep</td><td>40.0 ± 6.0</td></tr><tr><td>FDP [c]: Conv</td><td>42.0 ± 5.2</td><td>2000 ep</td><td>40.7 ± 3.1</td></tr></table>

Ablations. Table II presents ablations for our method. Results for DiT-Base show that lack of performance cannot be compensated for by increasing the model size. We also show that the integrated form of composition presented in Figure 2[c] outperforms the output score composition method [b] by a significant margin. Further, we find that preserving the diversity of $\pi _ { \mathrm { b a s e } }$ is essential: overfitting the base model leaves little residual signal to learn, reducing generalization, while stopping the training too early leaves an unstable base. Our ablations show that selecting the $\pi _ { \mathrm { b a s e } }$ checkpoint with the lowest validation loss (at 700 epochs) provides a good foundation for residual learning. Finally, the results in Table I show that learning the residual is crucial. Policy performance with $\pi _ { \mathrm { b a s e } }$ is unsatisfactory, and our factored approach is able to improve the performance without diminishing the policy robustness given an additional modality.

## VII. REAL-WORLD EXPERIMENTS

We evaluate FDP and the DP-DiT baseline across four real-world domains and report their task success rates. The domains are – Close Drawer as a simple task where the robot has to push the drawer; Put Block in Bowl that assesses the policy’s ability to perform precise pick-and-place actions; Pour in Bowl to evaluate the policy’s dexterity in operating near joint limits and Fold Towel to assess effectiveness in manipulating deformable objects.

We collect 50 demonstrations per domain on a Franka FR3 robot using a 6D space mouse, recording both proprioceptive and visual observations from two cameras—one mounted on the gripper and a static camera covering the workspace. The trained policies are evaluated on four task variations in each domain: (a) default: an in-distribution setup matching the conditions used during demonstration collection; (b) color: the object’s color is altered to test robustness to visual appearance changes; (c) distractor: novel, unseen objects such as vegetation props and soft toys are added to the scene to introduce clutter; and (d) occlusion: visual input is intermittently blocked during policy rollout to simulate partial observability. Figure 9 shows different task domains and their variations used in our experiments.

We use 10 rollouts in each experiment and report the task success rate as shown in Table III.

TABLE III  
SUCCESS RATES OF DP-DIT AND FDP ACROSS REAL-WORLD TASKS.

<table><tr><td rowspan="2">Task Domain</td><td colspan="2">default</td><td colspan="2">color</td><td colspan="2">dist.</td><td colspan="2">occl.</td></tr><tr><td>DP</td><td>FDP</td><td>DP</td><td>FDP</td><td>DP</td><td>FDP</td><td>DP</td><td>FDP</td></tr><tr><td>Close Drawer</td><td>90</td><td>90</td><td>90</td><td>90</td><td>10</td><td>80</td><td>0</td><td>80</td></tr><tr><td>Put Block in Bowl</td><td>80</td><td>80</td><td>0</td><td>60</td><td>0</td><td>60</td><td>10</td><td>60</td></tr><tr><td>Pour in Bowl</td><td>70</td><td>80</td><td>40</td><td>80</td><td>20</td><td>60</td><td>10</td><td>50</td></tr><tr><td>Fold Towel</td><td>40</td><td>60</td><td>40</td><td>70</td><td>30</td><td>70</td><td>10</td><td>50</td></tr></table>

Result Analysis. We find that FDP is robust to distribution shifts in the environment. DP-DiT regularly produces unachievable robot actions under distractor and occlusion settings, often triggering safety stops, resulting in task failure. In contrast, FDP guided by its prop π<sub>base</sub>, consistently generates stable actions even under severe occlusions and cluttered scenes, yielding an average absolute performance improvement of 40%. In the default experiment we observe that the FDP policy outperforms DP-DiT in the pouring and towel-folding tasks, which require precise object manipulation. Notably, FDP achieves 5× successrate than that of DP-DiT in the camera occlusion setting, highlighting it’s practicality for robots that must operate reliably in visually degraded environments.

## VIII. CONCLUSION AND FUTURE WORK

We present Factorized Diffusion Policies (FDP), a novel policy formulation that factorizes the joint conditioning in diffusion models so that observational modalities can have a differing influence on the action diffusion process by design. We derive a novel loss function to realize the prioritization order of modalities and propose a novel architecture for efficient training. Through extensive experiments across visual, point-cloud, proprioceptive and tactile modalities, we demonstrate several benefits of modality prioritization, including improved sample efficiency and increased robustness. Overall, we observe 15% absolute performance improvement on more than 20 tasks spread across several observational modalities after adopting FDP over jointly conditioned diffusion policy and even SmolVLA. We believe that this novel paradigm of modality prioritization along with strong performance gains, especially in low-data regimes make FDP a valuable contribution to the robot learning community. Finally, our real-world experiments highlight the practical value of FDP in being a safe-to-deploy policy in the face of visual disturbances and even camera occlusions, outperforming diffusion policies by over 40%.

FDP opens new avenues for future research, such as scalable integration of diversely sourced observational modality data for robot policy learning. FDP presents a computational overhead of searching through the prioritization order, and future work can develop a stronger framework that infers the modality to prioritize based on the task or the collected data. FDP prioritizes a single modality for the whole duration of the task, and dynamic prioritization of modalities also presents an interesting avenue for future work. Finally, we believe our approach will also have implications for finetuning of VLAs on new modalities not encountered in training. We believe FDP is a novel first-step towards the goal of observation modality prioritization.

![](images/0628afdf004b6b097a7d61c81ea52731a619decf827d7930eb530000f77725a7.jpg)  
Fig. 9. Real robot task domains and their variations. We show that DP-DiT fails in the presence of visual distribution shifts, highlighting the utility of prioritization for tasks with repetitive motions. Furthermore, FDP (prop>vision) is robust to camera blinds which can cause safety risks for DP-DiT.

## REFERENCES

[1] B. Wahn and P. König, “Is attentional resource allocation across sensory modalities task-dependent?” Advances in cognitive psychology, vol. 13, no. 1, p. 83, 2017.

[2] M. O. Ernst and M. S. Banks, “Humans integrate visual and haptic information in a statistically optimal fashion,” Nature, vol. 415, no. 6870, pp. 429–433, 2002.

[3] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” 2020. [Online]. Available: https://arxiv.org/abs/2006.11239

[4] C. Chi, Z. Xu, S. Feng, E. Cousineau, Y. Du, B. Burchfiel, R. Tedrake, and S. Song, “Diffusion policy: Visuomotor policy learning via action diffusion,” The International Journal of Robotics Research, p. 02783649241273668, 2023.

[5] M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. Foster, G. Lam, P. Sanketi et al., “Openvla: An open-source vision-language-action model,” arXiv preprint arXiv:2406.09246, 2024.

[6] K. Black, N. Brown, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, L. Groom, K. Hausman, B. Ichter et al., “π<sub>0</sub>: A visionlanguage-action flow model for general robot control,” arXiv preprint arXiv:2410.24164, 2024.

[7] M. Shukor, D. Aubakirova, F. Capuano, P. Kooijmans, S. Palma, A. Zouitine, M. Aractingi, C. Pascal, M. Russi, A. Marafioti et al., “Smolvla: A vision-language-action model for affordable and efficient robotics,” arXiv preprint arXiv:2506.01844, 2025.

[8] W. Liu, J. Mao, J. Hsu, T. Hermans, A. Garg, and J. Wu, “Composable part-based manipulation,” arXiv preprint arXiv:2405.05876, 2024.

[9] Z. Yang, J. Mao, Y. Du, J. Wu, J. B. Tenenbaum, T. Lozano-Pérez, and L. P. Kaelbling, “Compositional diffusion-based continuous constraint solvers,” arXiv preprint arXiv:2309.00966, 2023.

[10] L. Wang, J. Zhao, Y. Du, E. H. Adelson, and R. Tedrake, “Poco: Policy composition from and for heterogeneous robot learning,” arXiv preprint arXiv:2402.02511, 2024.

[11] Y. Luo, U. A. Mishra, Y. Du, and D. Xu, “Generative trajectory stitching through diffusion composition,” arXiv preprint arXiv:2503.05153, 2025.

[12] U. A. Mishra, S. Xue, Y. Chen, and D. Xu, “Generative skill chaining: Long-horizon skill planning with diffusion models,” in Conference on Robot Learning. PMLR, 2023, pp. 2905–2925.

[13] U. A. Mishra, Y. Chen, and D. Xu, “Generative factor chaining: Coordinated manipulation with diffusion-based factor graph,” in ICRA 2024 Workshop {\textemdash} Back to the Future: Robot Learning Going Probabilistic, 2024.

[14] T. Yu, T. Xiao, A. Stone, J. Tompson, A. Brohan, S. Wang, J. Singh, C. Tan, J. Peralta, B. Ichter et al., “Scaling robot learning with semantically imagined experience,” arXiv preprint arXiv:2302.11550, 2023.

[15] Z. Chen, S. Kiami, A. Gupta, and V. Kumar, “Genaug: Retargeting behaviors to unseen situations via generative augmentation,” arXiv preprint arXiv:2302.06671, 2023.

[16] M. Du, S. Nair, D. Sadigh, and C. Finn, “Behavior retrieval: Fewshot imitation learning by querying unlabeled datasets,” arXiv preprint arXiv:2304.08742, 2023.

[17] M. Memmel, J. Berg, B. Chen, A. Gupta, and J. Francis, “Strap: Robot sub-trajectory retrieval for augmented policy learning,” arXiv preprint arXiv:2412.15182, 2024.

[18] L.-H. Lin, Y. Cui, A. Xie, T. Hua, and D. Sadigh, “Flowretrieval: Flowguided data retrieval for few-shot imitation learning,” arXiv preprint arXiv:2408.16944, 2024.

[19] M. Alakuijala, G. Dulac-Arnold, J. Mairal, J. Ponce, and C. Schmid, “Residual reinforcement learning from demonstrations,” arXiv preprint arXiv:2106.08050, 2021.

[20] X. Yuan, T. Mu, S. Tao, Y. Fang, M. Zhang, and H. Su, “Policy decorator: Model-agnostic online refinement for large policy model,” arXiv preprint arXiv:2412.13630, 2024.

[21] L. Dodeja, K. Schmeckpeper, S. Vats, T. Weng, M. Jia, G. Konidaris, and S. Tellex, “Accelerating residual reinforcement learning with uncertainty estimation,” arXiv preprint arXiv:2506.17564, 2025.

[22] L. Ankile, A. Simeonov, I. Shenfeld, M. Torne, and P. Agrawal, “From imitation to refinement-residual rl for precise assembly,” in 2025 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2025, pp. 01–08.

[23] Y. Jiang, C. Wang, R. Zhang, J. Wu, and L. Fei-Fei, “Transic: Sim-toreal policy transfer by learning from online correction,” arXiv preprint arXiv:2405.10315, 2024.

[24] Y.-C. Li, F. Zhang, W. Qiu, L. Yuan, C. Jia, Z. Zhang, Y. Yu, and B. An, “Q-adapter: Customizing pre-trained llms to new preferences with forgetting mitigation,” 2025. [Online]. Available: https://arxiv.org/abs/2407.03856

[25] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen et al., “Lora: Low-rank adaptation of large language models.” ICLR, vol. 1, no. 2, p. 3, 2022.

[26] L. Zhang, A. Rao, and M. Agrawala, “Adding conditional control to text-to-image diffusion models,” 2023. [Online]. Available: https://arxiv.org/abs/2302.05543

[27] Z. Liu, J. Zhang, K. Asadi, Y. Liu, D. Zhao, S. Sabach, and R. Fakoor, “Tail: Task-specific adapters for imitation learning with large pretrained models,” arXiv preprint arXiv:2310.05905, 2023.

[28] W. Gu, S. Kondepudi, L. Huang, and N. Gopalan, “Continual skill and task learning via dialogue,” arXiv preprint arXiv:2409.03166, 2024.

[29] M. Sharma, C. Fantacci, Y. Zhou, S. Koppula, N. Heess, J. Scholz, and Y. Aytar, “Lossless adaptation of pretrained vision models for robotic manipulation,” arXiv preprint arXiv:2304.06600, 2023.

[30] J. Wen, Y. Zhu, J. Li, M. Zhu, K. Wu, Z. Xu, N. Liu, R. Cheng, C. Shen, Y. Peng, F. Feng, and J. Tang, “Tinyvla: Towards fast, dataefficient vision-language-action models for robotic manipulation,” 2025. [Online]. Available: https://arxiv.org/abs/2409.12514

[31] X. Liu, Y. Zhou, F. Weigend, S. Sonawani, S. Ikemoto, and H. B. Amor, “Diff-control: A stateful diffusion-based policy for imitation learning,” in 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2024, pp. 7453–7460.

[32] J. Sohl-Dickstein, E. A. Weiss, N. Maheswaranathan, and S. Ganguli,

“Deep unsupervised learning using nonequilibrium thermodynamics,” 2015.

[33] Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole, “Score-based generative modeling through stochastic differential equations,” arXiv preprint arXiv:2011.13456, 2020.

[34] Y. Song and S. Ermon, “Generative modeling by estimating gradients of the data distribution,” 2020.

[35] A. Hyvärinen and P. Dayan, “Estimation of non-normalized statistical models by score matching.” Journal of Machine Learning Research, vol. 6, no. 4, 2005.

[36] P. Vincent, “A connection between score matching and denoising autoencoders,” Neural computation, vol. 23, no. 7, pp. 1661–1674, 2011.

[37] C.-H. Chao, W.-F. Sun, B.-W. Cheng, Y.-C. Lo, C.-C. Chang, Y.-L. Liu, Y.-L. Chang, C.-P. Chen, and C.-Y. Lee, “Denoising likelihood score matching for conditional score-based data generation,” 2022. [Online]. Available: https://arxiv.org/abs/2203.14206

[38] G. O. Roberts and R. L. Tweedie, “Exponential convergence of langevin distributions and their discrete approximations,” 1996.

[39] P. Dhariwal and A. Nichol, “Diffusion models beat gans on image synthesis,” Advances in neural information processing systems, vol. 34, pp. 8780–8794, 2021.

[40] J. Ho and T. Salimans, “Classifier-free diffusion guidance,” arXiv preprint arXiv:2207.12598, 2022.

[41] Y. Du, C. Durkan, R. Strudel, J. B. Tenenbaum, S. Dieleman, R. Fergus, J. Sohl-Dickstein, A. Doucet, and W. Grathwohl, “Reduce, reuse, recycle: Compositional generation with energy-based diffusion models and mcmc,” 2023.

[42] O. Ronneberger, P. Fischer, and T. Brox, “U-net: Convolutional networks for biomedical image segmentation,” in Medical image computing and computer-assisted intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, pro ceedings, part III 18. Springer, 2015, pp. 234–241.

[43] W. Peebles and S. Xie, “Scalable diffusion models with transformers,” 2023. [Online]. Available: https://arxiv.org/abs/2212.09748

[44] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly et al., “An image is worth 16x16 words: Transformers for image recognition at scale,” arXiv preprint arXiv:2010.11929, 2020.

[45] S. James, Z. Ma, D. R. Arrojo, and A. J. Davison, “Rlbench: The robot learning benchmark & learning environment,” CoRR, vol. abs/1909.12271, 2019. [Online]. Available: http://arxiv.org/abs/1909. 12271

[46] A. Rajeswaran, V. Kumar, A. Gupta, G. Vezzani, J. Schulman, E. Todorov, and S. Levine, “Learning complex dexterous manipulation with deep reinforcement learning and demonstrations,” arXiv preprint arXiv:1709.10087, 2017.

[47] A. Mandlekar, D. Xu, J. Wong, S. Nasiriany, C. Wang, R. Kulkarni, L. Fei-Fei, S. Savarese, Y. Zhu, and R. Martín-Martín, “What matters in learning from offline human demonstrations for robot manipulation,” arXiv preprint arXiv:2108.03298, 2021.

[48] C. Sferrazza, Y. Seo, H. Liu, Y. Lee, and P. Abbeel, “The power of the senses: Generalizable manipulation from vision and touch through masked multimodal learning,” in 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2024, pp. 9698–9705.

[49] J. Ho and T. Salimans, “Classifier-free diffusion guidance,” 2022. [Online]. Available: https://arxiv.org/abs/2207.12598

[50] Z. Zhao, S. Haldar, J. Cui, L. Pinto, and R. Bhirangi, “Touch begins where vision ends: Generalizable policies for contact-rich manipulation,” 2025. [Online]. Available: https://arxiv.org/abs/2506. 13762

[51] Y. Ze, G. Zhang, K. Zhang, C. Hu, M. Wang, and H. Xu, “3d diffusion policy: Generalizable visuomotor policy learning via simple 3d representations,” arXiv preprint arXiv:2403.03954, 2024.

[52] J. Song, C. Meng, and S. Ermon, “Denoising diffusion implicit models,” 2022. [Online]. Available: https://arxiv.org/abs/2010.02502

[53] M. Reuss, M. Li, X. Jia, and R. Lioutikov, “Goal-conditioned imitation learning using score-based diffusion policies,” arXiv preprint arXiv:2304.02532, 2023.

[54] X. Liu, K. Y. Ma, C. Gao, and M. Z. Shou, “Diffusion models in robotics: A survey,” 2025.

[55] C. Doersch, “Tutorial on variational autoencoders,” arXiv preprint arXiv:1606.05908, 2016.

[56] C. Luo, “Understanding diffusion models: A unified perspective,” 2022. [Online]. Available: https://arxiv.org/abs/2208.11970

[57] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2016, pp. 770–778.

[58] P. von Platen, S. Patil, A. Lozhkov, P. Cuenca, N. Lambert, K. Rasul, M. Davaadorj, D. Nair, S. Paul, W. Berman, Y. Xu, S. Liu, and T. Wolf, “Diffusers: State-of-the-art diffusion models,” https://github. com/huggingface/diffusers, 2022.

[59] E. Perez, F. Strub, H. de Vries, V. Dumoulin, and A. C. Courville, “Film: Visual reasoning with a general conditioning layer,” in AAAI, 2018.

## APPENDIX

A. Proof of Diffusion Loss for Full Conditional Action Distribution

Most treatments of diffusion models have been studied primarily in the context of single-modality distributions, such as those over image pixels [3], [34], [52]. This formulation has been directly adopted by the robotics community [4], [53], [54], leading to the optimization objective shown in Equation 9.

$$
\mathcal {L} _ {t} (\boldsymbol {\theta}) = \mathbb {E} _ {(\boldsymbol {x} _ {0}, \boldsymbol {y}) \sim q (\boldsymbol {x} _ {0}, \boldsymbol {y}), \epsilon_ {0} \sim \mathcal {N} (0, \mathcal {I})} \left[ | | \boldsymbol {\epsilon} _ {0} - \hat {\boldsymbol {\epsilon}} _ {\boldsymbol {\theta}} (\boldsymbol {x} _ {t}, \boldsymbol {y}, t) | | _ {2} ^ {2} \right]\tag{9}
$$

We formally show that a conditional diffusion process as defined by Dhariwal and Nichol [39] results in Equation 9 being a maximizer of the reweighted variational lower bound [3] on the log-likelihood of the conditional data distribution $\log q ( { \pmb x } | { \pmb y } )$ . This implies that by incorporating the observational modalities into the conditioning process, diffusion policies [4] learn the reverse transition kernel for action conditioned on all the modalities, as shown in the following proof. We adopt a slightly different notation from the main paper where we use q for the forward and the reverse diffusion process, and the resulting marginals.

Lemma 1: The diffusion loss function $\mathcal { L } _ { t } ( \pmb { \theta } )$ as defined in Equation 9, in expectation over the time-steps $1 \leq t \leq T .$ maximizes the reweighted variational lower bound [3] on the log-likelihood of the conditional data distribution log q(x|y), under a Markovian noising process $\hat { q } \big ( \mathbf { \mathscr { x } } _ { t } | \mathbf { \mathscr { x } } _ { t - 1 } \big )$ and the conditional reverse transition kernel as $\hat { q } ( \pmb { x } _ { t - 1 } | \pmb { x } _ { t } , \pmb { y } )$

Here, we derive the diffusion loss function for the conditional distribution $p ( { \pmb x } | { \pmb y } )$ instead of only $p ( { \pmb x } )$ . A parallel derivation for conditional variational auto-encoders can be found in Doersch [55]. Following Dhariwal and Nichol [39], we start with a conditional Markovian noising forward process $\hat { q }$ similar to $q ( \pmb { x } _ { t } | \pmb { x } _ { t - 1 } ) = \mathcal { N } ( \pmb { x } _ { t } ; \sqrt { \alpha _ { t } } \pmb { x } _ { t - 1 } , ( 1 - \alpha _ { t } ) \mathbb { Z } )$ and define the following:

$$
\hat {q} (\boldsymbol {x} _ {0}) := q (\boldsymbol {x} _ {0})\tag{10}
$$

$$
\hat {q} (\boldsymbol {x} _ {t + 1} | \boldsymbol {x} _ {t}, \boldsymbol {y}) := q (\boldsymbol {x} _ {t + 1} | \boldsymbol {x} _ {t})\tag{11}
$$

$$
\hat {q} (\boldsymbol {x} _ {1: T} | \boldsymbol {x} _ {0}, \boldsymbol {y}) := \prod_ {t = 1} ^ {T} \hat {q} (\boldsymbol {x} _ {t} | \boldsymbol {x} _ {t - 1}, \boldsymbol {y})\tag{12}
$$

We now reproduce some results that will be used later in the derivation of diffusion loss for conditional distributions. Dhariwal and Nichol [39] also show that

$$
\hat {q} (\pmb {y} | \pmb {x} _ {t}, \pmb {x} _ {t + 1}) = \hat {q} (\pmb {x} _ {t + 1} | \pmb {x} _ {t}, \pmb {y}) \frac {\hat {q} (\pmb {y} | \pmb {x} _ {t})}{\hat {q} (\pmb {x} _ {t + 1} | \pmb {x} _ {t})}\tag{13}
$$

$$
= \hat {q} (\boldsymbol {x} _ {t + 1} | \boldsymbol {x} _ {t}) \frac {\hat {q} (\boldsymbol {y} | \boldsymbol {x} _ {t})}{\hat {q} (\boldsymbol {x} _ {t + 1} | \boldsymbol {x} _ {t})}\tag{14}
$$

$$
= \hat {q} (\pmb {y} | \pmb {x} _ {t})\tag{15}
$$

Moreover, the unconditional reverse transition kernels can be shown to be equal using Bayes theorem, given Equations 10 and 11: ${ \hat { \pmb q } } ( \pmb { x } _ { t } | \pmb { x } _ { t + 1 } ) = \pmb q ( \pmb { x } _ { t } | \pmb { x } _ { t + 1 } )$ . Dhariwal and Nichol [39] use the result from Equation 15 to show the following for conditional reverse transition kernels.

$$
\hat {q} (\boldsymbol {x} _ {t} | \boldsymbol {x} _ {t + 1}, \boldsymbol {y}) = \frac {\hat {q} (\boldsymbol {x} _ {t} , \boldsymbol {x} _ {t + 1} , \boldsymbol {y})}{\hat {q} (\boldsymbol {x} _ {t + 1} , \boldsymbol {y})}\tag{16}
$$

$$
= \overline {{\hat {q} (\boldsymbol {y} | \boldsymbol {x} _ {t + 1}) \hat {q} (\boldsymbol {x} _ {t + 1})}}\tag{17}
$$

$$
\hat {q} (\pmb {y} | \pmb {x} _ {t + 1}) \hat {q} (\pmb {x} _ {t + 1})\tag{18}
$$

$$
= \frac {\hat {q} (\boldsymbol {x} _ {t} | \boldsymbol {x} _ {t + 1}) \hat {q} (\boldsymbol {y} | \boldsymbol {x} _ {t} , \boldsymbol {x} _ {t + 1})}{\hat {q} (\boldsymbol {y} | \boldsymbol {x} _ {t + 1})}\tag{19}
$$

$$
= \frac {q (\boldsymbol {x} _ {t} | \boldsymbol {x} _ {t + 1}) \hat {q} (\boldsymbol {y} | \boldsymbol {x} _ {t})}{\hat {q} (\boldsymbol {y} | \boldsymbol {x} _ {t + 1})}\tag{20}
$$

Further, we can show the following using Equations 11 and 12 and the Markovian noising process. It states that the joint distribution of the noised samples conditioned on y and $\scriptstyle { \mathbf { { \vec { x } } } } _ { 0 }$ are the same for both $\hat { q }$ and $q .$

$$
\hat {q} (\pmb {x} _ {1: T} | \pmb {x} _ {0}, \pmb {y}) = \prod_ {t = 1} ^ {T} \hat {q} (\pmb {x} _ {t} | \pmb {x} _ {t - 1}, \pmb {y})\tag{21}
$$

$$
= \prod_ {t = 1} ^ {T} q (\boldsymbol {x} _ {t} | \boldsymbol {x} _ {t - 1})
$$

$$
= q (\pmb {x} _ {1: T} | \pmb {x} _ {0})\tag{22}
$$

(23)

We adapt the derivation of diffusion loss from Luo [56] to work with conditional distributions by maximizing the log-likelihood of the conditional data distribution log p(x|y) leading to evidence lower bound (ELBO).

$$
\log p (\boldsymbol {x} | \boldsymbol {y}) = \log \int p (\boldsymbol {x} _ {0: T} | \boldsymbol {y}) d \boldsymbol {x} _ {1: T}\tag{24}
$$

$$
= \log \int \frac {p (\pmb {x} _ {0 : T} | \pmb {y}) \hat {q} (\pmb {x} _ {1 : T} | \pmb {x} _ {0} , \pmb {y})}{\hat {q} (\pmb {x} _ {1 : T} | \pmb {x} _ {0} , \pmb {y})} d \pmb {x} _ {1: T}\tag{25}
$$

$$
= \log \mathbb {E} _ {\hat {q} (\pmb {x} _ {1: T} | \pmb {x} _ {0}, \pmb {y})} \left[ \frac {p (\pmb {x} _ {0 : T} | \pmb {y})}{\hat {q} (\pmb {x} _ {1 : T} | \pmb {x} _ {0} , \pmb {y})} \right]\tag{26}
$$

$$
\geq \mathbb {E} _ {\hat {q} (\boldsymbol {x} _ {1: T} | \boldsymbol {x} _ {0}, \boldsymbol {y})} \left[ \log \frac {p (\boldsymbol {x} _ {0 : T} | \boldsymbol {y})}{\hat {q} (\boldsymbol {x} _ {1 : T} | \boldsymbol {x} _ {0} , \boldsymbol {y})} \right]\tag{27}
$$

The ELBO can be further simplified as follows

$$
\begin{array}{l} \log p (\boldsymbol {x} | \boldsymbol {y}) \geq \mathbb {E} _ {\hat {q} (\boldsymbol {x} _ {1: T} | \boldsymbol {x} _ {0}, \boldsymbol {y})} \left[ \log \frac {p (\boldsymbol {x} _ {0 : T} | \boldsymbol {y})}{\hat {q} (\boldsymbol {x} _ {1 : T} | \boldsymbol {x} _ {0} , \boldsymbol {y})} \right] \\ = \mathbb {E} _ {\hat {q} (\boldsymbol {x} _ {1: T} | \boldsymbol {x} _ {0}, \boldsymbol {y})} \left[ \log \frac {p (\boldsymbol {x} _ {T} | \boldsymbol {y}) \prod_ {t = 1} ^ {T} p _ {\boldsymbol {\theta}} (\boldsymbol {x} _ {t - 1} | \boldsymbol {x} _ {t} , \boldsymbol {y})}{\prod_ {t = 1} ^ {T} \hat {q} (\boldsymbol {x} _ {t} | \boldsymbol {x} _ {t - 1} , \boldsymbol {y})} \right] \\ = \mathbb {E} _ {\hat {q} (\boldsymbol {x} _ {1: T} | \boldsymbol {x} _ {0}, \boldsymbol {y})} \left[ \log \frac {p (\boldsymbol {x} _ {T} | \boldsymbol {y}) p _ {\boldsymbol {\theta}} (\boldsymbol {x} _ {0} | \boldsymbol {x} _ {1} , \boldsymbol {y})}{\hat {q} (\boldsymbol {x} _ {1} | \boldsymbol {x} _ {0} , \boldsymbol {y})} \right. \\ + \log \prod_ {t = 2} ^ {T} \frac {p _ {\boldsymbol {\theta}} (\boldsymbol {x} _ {t - 1} | \boldsymbol {x} _ {t} , \boldsymbol {y})}{\hat {q} (\boldsymbol {x} _ {t} | \boldsymbol {x} _ {t - 1} , \boldsymbol {x} _ {0} , \boldsymbol {y})} \Bigg ] \\ = \mathbb {E} _ {\hat {q} (\boldsymbol {x} _ {1: T} | \boldsymbol {x} _ {0}, \boldsymbol {y})} \left[ \log \frac {p (\boldsymbol {x} _ {T} | \boldsymbol {y}) p _ {\boldsymbol {\theta}} (\boldsymbol {x} _ {0} | \boldsymbol {x} _ {1} ， \boldsymbol {y})}{\hat {q} (\boldsymbol {x} _ {1} | \boldsymbol {x} _ {0} ， \boldsymbol {y})} \right. \\ + \log \prod_ {t = 2} ^ {T} \frac {p _ {\boldsymbol {\theta}} (\boldsymbol {x} _ {t - 1} | \boldsymbol {x} _ {t} ， \boldsymbol {y})}{\frac {\hat {q} (\boldsymbol {x} _ {t - 1} | \boldsymbol {x} _ {t} ， \boldsymbol {x} _ {0} ， \boldsymbol {y}) \hat {q} (\boldsymbol {x} _ {t} | \boldsymbol {x} _ {0} ， \boldsymbol {y})}{\hat {q} (\boldsymbol {x} _ {t - 1} | \boldsymbol {x} _ {0} ， \boldsymbol {y})}} \Bigg ] \end{array}
$$

$$
= \mathbb {E} _ {\hat {q} (\boldsymbol {x} _ {1: T} | \boldsymbol {x} _ {0}, \boldsymbol {y})} \left[ \log \frac {p (\boldsymbol {x} _ {T} | \boldsymbol {y}) p _ {\boldsymbol {\theta}} (\boldsymbol {x} _ {0} | \boldsymbol {x} _ {1} , \boldsymbol {y})}{\hat {q} (\boldsymbol {x} _ {1} | \boldsymbol {x} _ {0} , \boldsymbol {y})} \right.
$$

$$
\left. + \log \frac {\hat {q} (\boldsymbol {x} _ {1} | \boldsymbol {x} _ {0} , \boldsymbol {y})}{\hat {q} (\boldsymbol {x} _ {T} | \boldsymbol {x} _ {0} , \boldsymbol {y})} + \log \prod_ {t = 2} ^ {T} \frac {p _ {\boldsymbol {\theta}} (\boldsymbol {x} _ {t - 1} | \boldsymbol {x} _ {t} , \boldsymbol {y})}{\hat {q} (\boldsymbol {x} _ {t - 1} | \boldsymbol {x} _ {t} , \boldsymbol {x} _ {0} , \boldsymbol {y})} \right]
$$

$$
= \mathbb {E} _ {\hat {q} (\boldsymbol {x} _ {1: T} | \boldsymbol {x} _ {0}, \boldsymbol {y})} \left[ \log \frac {p (\boldsymbol {x} _ {T} | \boldsymbol {y}) p _ {\boldsymbol {\theta}} (\boldsymbol {x} _ {0} | \boldsymbol {x} _ {1} , \boldsymbol {y})}{\hat {q} (\boldsymbol {x} _ {T} | \boldsymbol {x} _ {0} , \boldsymbol {y})} \right.
$$

$$
\begin{array}{l} + \sum_ {t = 2} ^ {T} \log \frac {p _ {\boldsymbol {\theta}} (\boldsymbol {x} _ {t - 1} | \boldsymbol {x} _ {t} , \boldsymbol {y})}{\hat {q} (\boldsymbol {x} _ {t - 1} | \boldsymbol {x} _ {t} , \boldsymbol {x} _ {0} , \boldsymbol {y})} \\ = \mathbb {E} _ {\hat {q} (\boldsymbol {x} _ {1} | \boldsymbol {x} _ {0}, \boldsymbol {y})} \left[ \log p _ {\boldsymbol {\theta}} (\boldsymbol {x} _ {0} | \boldsymbol {x} _ {1}, \boldsymbol {y}) \right] \\ + \mathbb {E} _ {\hat {q} (\boldsymbol {x} _ {T} | \boldsymbol {x} _ {0}, \boldsymbol {y})} \left[ \log \frac {p (\boldsymbol {x} _ {T} | \boldsymbol {y})}{\hat {q} (\boldsymbol {x} _ {T} | \boldsymbol {x} _ {0} , \boldsymbol {y})} \right] \\ + \sum_ {t = 2} ^ {T} \mathbb {E} _ {\hat {q} (\boldsymbol {x} _ {t}, \boldsymbol {x} _ {t - 1} | \boldsymbol {x} _ {0}, \boldsymbol {y})} \left[ \log \frac {p _ {\boldsymbol {\theta}} (\boldsymbol {x} _ {t - 1} | \boldsymbol {x} _ {t} , \boldsymbol {y})}{\hat {q} (\boldsymbol {x} _ {t - 1} | \boldsymbol {x} _ {t} , \boldsymbol {x} _ {0} , \boldsymbol {y})} \right] \\ = \underbrace {\mathbb {E} _ {\hat {q} (\boldsymbol {x} _ {1} | \boldsymbol {x} _ {0} , \boldsymbol {y})} [ \log p _ {\boldsymbol {\theta}} (\boldsymbol {x} _ {0} | \boldsymbol {x} _ {1} , \boldsymbol {y}) ]} _ {\text { reconstruction term}} \\ - \underbrace {D _ {\mathrm{KL}} (\hat {q} (\boldsymbol {x} _ {T} | \boldsymbol {x} _ {0} , \boldsymbol {y}) \| p (\boldsymbol {x} _ {T} | \boldsymbol {y}))} _ {\text { prior matching term }} \\ - \sum_ {t = 2} ^ {T} \underbrace {\mathbb {E} _ {\hat {q} (\boldsymbol {x} _ {t} | \boldsymbol {x} _ {0} , \boldsymbol {y})}} _ {\text { denoising matching term }} [ \\ D _ {\mathrm{KL}} (\hat {q} (\boldsymbol {x} _ {t - 1} | \boldsymbol {x} _ {t}, \boldsymbol {x} _ {0}, \boldsymbol {y}) \| p _ {\boldsymbol {\theta}} (\boldsymbol {x} _ {t - 1} | \boldsymbol {x} _ {t}, \boldsymbol {y})) ] \end{array}\tag{28}
$$

The reconstruction term is ignored for training [3], [56], and the prior matching term does not have any trainable parameters. We further simplify the denoising matching term using Equation 20 further conditioned on $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$

$$
\begin{array}{l} - \sum_ {t = 2} ^ {T} \mathbb {E} _ {\hat {q} (\boldsymbol {x} _ {t} | \boldsymbol {x} _ {0}, \boldsymbol {y})} \left[ D _ {\mathrm{KL}} (\hat {q} (\boldsymbol {x} _ {t - 1} | \boldsymbol {x} _ {t}, \boldsymbol {x} _ {0}, \boldsymbol {y}) \| p _ {\boldsymbol {\theta}} (\boldsymbol {x} _ {t - 1} | \boldsymbol {x} _ {t}, \boldsymbol {y})) \right] \\ = - \sum_ {t = 2} ^ {T} \mathbb {E} _ {\hat {q} (\boldsymbol {x} _ {t} | \boldsymbol {x} _ {0}, \boldsymbol {y})} \left[ \mathbb {E} _ {\hat {q} (\boldsymbol {x} _ {t - 1} | \boldsymbol {x} _ {t}, \boldsymbol {x} _ {0}, \boldsymbol {y})} [ \log \hat {q} (\boldsymbol {x} _ {t - 1} | \boldsymbol {x} _ {t}, \boldsymbol {x} _ {0}, \boldsymbol {y}) - \log p _ {\boldsymbol {\theta}} (\boldsymbol {x} _ {t - 1} | \boldsymbol {x} _ {t}, \boldsymbol {y}) ] \right] \\ = - \sum_ {t = 2} ^ {T} \mathbb {E} _ {\hat {q} (\boldsymbol {x} _ {t} | \boldsymbol {x} _ {0}, \boldsymbol {y})} \left[ \mathbb {E} _ {\hat {q} (\boldsymbol {x} _ {t - 1} | \boldsymbol {x} _ {t}, \overline {{\boldsymbol {x}}} _ {0}, \boldsymbol {y})} [ \log \hat {q} (\boldsymbol {x} _ {t - 1} | \boldsymbol {x} _ {t}, \boldsymbol {x} _ {0}) + \log \frac {\hat {q} (\boldsymbol {y} | \boldsymbol {x} _ {t - 1} , \boldsymbol {x} _ {0})}{\hat {q} (\boldsymbol {y} | \boldsymbol {x} _ {t} , \boldsymbol {x} _ {0})} - \log p _ {\boldsymbol {\theta}} (\boldsymbol {x} _ {t - 1} | \boldsymbol {x} _ {t}, \boldsymbol {y}) ] \right] \\ = - \sum_ {t = 2} ^ {T} \mathbb {E} _ {\hat {q} (\boldsymbol {x} _ {t} | \boldsymbol {x} _ {0}, \boldsymbol {y))}} [ D _ {\mathrm{KL}} (\hat {q} (\boldsymbol {x} _ {t - 1} | \boldsymbol {x} _ {t}, \boldsymbol {x} _ {0}) \| p _ {\boldsymbol {\theta}} (\boldsymbol {x} _ {t - 1} | \boldsymbol {x} _ {t}, \boldsymbol {y})) ] \\ - \sum_ {t = 2} ^ {T} \mathbb {E} _ {\hat {q} (\boldsymbol {x} _ {t} | \boldsymbol {x} _ {0}, \boldsymbol {y})} \left[ \mathbb {E} _ {\hat {q} (\boldsymbol {x} _ {t - 1} | \boldsymbol {x} _ {t}, \overline {{\boldbf x}} _ {0}, \boldsymbol {y})} [ \log \frac {\hat {q} (\boldsymbol {y} | \boldsymbol {x} _ {t - 1} , \boldsymbol {x} _ {0})}{\hat {q} (\boldsymbol {y} | \boldsymbol {x} _ {t} , \overline {{\boldsymbol x}} _ {0})} ] \right] \end{array}\tag{29}
$$

Note that the expectation is taken over a distribution independent of $^ { \mathbf { \delta } _ { \mathbf { \delta } _ { \mathbf { \delta } _ { \mathbf { \delta } _ { \mathbf { \delta } _ { \mathbf { \delta } _ { \delta } } } } } } }$ since $\hat { q } ( { \pmb x } _ { 1 : T } | { \pmb x } _ { 0 } , { \pmb y } ) = q ( { \pmb x } _ { 1 : T } | { \pmb x } _ { 0 } )$ , as shown in Equation 23. It is easy to see that the first term in the resulting expression is the KL divergence between the model parameterized with the condition y and the unconditional reverse transition kernel, leading to the popularly used diffusion loss of Equation 9. However, an additional term is introduced for the conditional diffusion process. This minimizes the difference in the likelihood of the labels between consecutive denoising steps. However, since it does not have trainable parameters, we ignore it.

In Equation 9, $\mathbf { \epsilon } _ { \mathbf { \theta } } ( \mathbf { \mathcal { x } } _ { t } , \mathbf { \mathcal { y } } , t )$ arises from the reparametrization of the reverse transition kernel ${ q _ { \pmb \theta } } ( \pmb x _ { t - 1 } | \pmb x _ { t } , \pmb y ^ { 1 : M } )$ . In this work, we argue that learning the full conditional directly is restrictive in several aspects of robot learning. Firstly, it necessitates the joint collection of the robot action and all observational modalities. Secondly, the model is vulnerable to even small distribution shifts in any modality. These shifts require a prohibitively large amount of data to address when the observation modalities are high-dimensional. Finally, among the multiple observation modalities it is hard to pinpoint the level of each mode’s task dependent influence with limited data.

## B. Proof for FDP Loss

Theorem 2: Explicit score matching for $\mathbf { \delta } _ { \mathbf { \boldsymbol { s } } _ { \phi } \left( \mathbf { \boldsymbol { x } } _ { t } , \mathbf { \boldsymbol { y } } ^ { 1 : M } \right) }$ in Equation 3 is equivalent to the objective $\mathcal { T } _ { \alpha _ { t } } ^ { r e s } ( \phi )$

$$
\mathbb{E}_{\substack{p_{\alpha_{t}}(\boldsymbol{x},\boldsymbol{x}_{t},\\ \boldsymbol{y}^{1:M})}}\left[ \frac{1}{2}\left\| \nabla_{\boldsymbol{x}_{t}}\log p_{\alpha_{t}}(\boldsymbol{x}_{t}|\boldsymbol {x}) - \begin{array}{c}\boldsymbol{s}^{*}(\boldsymbol{x}_{t},\boldsymbol{y}^{1:k})\\ -\boldsymbol{s}_{\phi}(\boldsymbol{x}_{t},\boldsymbol{y}^{1:M}) \end{array} \right\|_{2}^{2}\right]
$$

Here $\mathbf { { \sigma } } _ { \mathbf { { s } } ^ { \ast } \left( \mathbf { { x } } _ { t } , \mathbf { { y } } ^ { 1 : k } \right) }$ is the the frozen optimal score model for $\nabla _ { \pmb { x } _ { t } } \log p ( \pmb { x } _ { t } | \pmb { y } ^ { 1 : k } )$ ), approximated using a learned model π<sub>base</sub> : $\mathbf { \sigma } _ { { s } _ { \theta } \left( \pmb { x } _ { t } , \pmb { y } ^ { 1 : k } \right) }$ in practice.

Chao et al. [37] in their insightful work for score-based models, show that the following two losses differ only by a constant, where the x˜ and $\tilde { \pmb { y } }$ notation is introduced to indicate Gaussian noised variables, with variances α and $\tau$ respectively.

$$
\begin{array}{l} \mathcal {D} _ {F} (p _ {\phi} (\tilde {\boldsymbol {y}} | \tilde {\boldsymbol {x}}) \| p _ {\alpha , \tau} (\tilde {\boldsymbol {y}} | \tilde {\boldsymbol {x}})) = \mathbb {E} _ {p _ {\alpha , \tau} (\tilde {\boldsymbol {x}}, \tilde {\boldsymbol {y}})} \left[ \frac {1}{2} \left\| \begin{array}{c} \nabla_ {\tilde {\boldsymbol {x}}} \log p _ {\phi} (\tilde {\boldsymbol {y}} | \tilde {\boldsymbol {x}}) \\ - \nabla_ {\tilde {\boldsymbol {x}}} \log p _ {\alpha , \tau} (\tilde {\boldsymbol {y}} | \tilde {\boldsymbol {x}}) \end{array} \right\| _ {2} ^ {2} \right] \\ (3 1) \\ \mathcal {L} _ {D L S M} (\phi) = \mathbb {E} _ {p _ {\alpha , \tau} (\boldsymbol {x}, \tilde {\boldsymbol {x}}, \boldsymbol {y}, \tilde {\boldsymbol {y}})} \left[ \frac {1}{2} \left\| \begin{array}{c} \nabla_ {\tilde {\boldsymbol {x}}} \log p _ {\phi} (\tilde {\boldsymbol {y}} | \tilde {\boldsymbol {x}}) + \nabla_ {\tilde {\boldsymbol {x}}} \log p _ {\theta} (\tilde {\boldsymbol {x}}) \\ - \nabla_ {\tilde {\boldsymbol {x}}} \log p _ {\alpha} (\tilde {\boldsymbol {x}} | \boldsymbol {x}) \end{array} \right\| _ {2} ^ {2} \right] \end{array}\tag{32}
$$

We extend their proof for diffusion models and multiple conditionals below. Here $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ is noised with the forward transition kernel $p _ { \bar { \alpha } _ { t } } ( { \pmb x } _ { t } | { \pmb x } ) { = } \mathcal { N } ( { \pmb x } _ { t } ; \sqrt { \bar { \alpha _ { t } } } { \pmb x } , ( 1 { - } \bar { \alpha _ { t } } ) I )$ . Explicit Score Matching loss between the model and the true score of the classifier can be further expanded as:

(33)

$$
\begin{array} { l }D _ { F } ^ { t } ( p _ { \phi } ( \tilde { \bm { y } } ^ { k + 1 : M } | \bm { x } _ { t } , \tilde { \bm { y } } ^ { 1 : k } ) | | p _ { \alpha , \tau } ( \tilde { \bm { y } } ^ { k + 1 : M } | \bm { x } _ { t } , \tilde { \bm { y } } ^ { 1 : k } ) )\\= \mathbb { E } _ { p _ { \alpha , \tau } ( \bm { x } _ { t } , \tilde { \bm { y } } ^ { 1 : M } ) } \left[ \frac { 1 } { 2 } | | \nabla _ { \bm { x } _ { t } } \log p _ { \phi } ( \tilde { \bm { y } } ^ { k + 1 : M } | \bm { x } _ { t } , \tilde { \bm { y } } ^ { 1 : k } ) - \nabla _ { \bm { x } _ { t } } \log p _ { \alpha , \tau } ( \tilde { \bm { y } } ^ { k + 1 : M } | \bm { x } _ { t } , \tilde { \bm { y } } ^ { 1 : k } ) | | _ { 2 } ^ { 2 } \right]\\= \mathbb { E } _ { p _ { \alpha , \tau } ( \bm { x } _ { t } , \tilde { \bm { y } } ^ { 1 : M } ) } \left[ \frac { 1 } { 2 } | | \nabla _ { \bm { x } _ { t } } \log p _ { \phi } ( \tilde { \bm { y } }^{ k + 1 : M } | \bm { x } _ { t } , \tilde { \bm { y } } ^ { 1 : k } ) | | _ { 2 } ^ { 2 } \right]\\+   \mathbb { E } _ { p _ { \alpha , \tau } ( \bm { x } _ { t } , \tilde { \bm { y } } ^ { 1 : M } ) } \left[ \frac { 1 } { 2 } | | \nabla _ { \bm { x } _ { t } } \log p _ { \alpha , \tau } ( \tilde { \bm { y } } ^ { k + 1 : M } | \bm { x } _ { t } , \tilde { \bm { y } } ^ { 1 : k } ) | | _ { 2 } ^ { 2 } \right]\\-   \mathbb { E } _ { p _ { \alpha , \tau } ( \bm { x } _ { t } , \tilde { \bm { y } } ^ { 1 : M } ) } \left[ \langle \nabla _ { \bm { x } _ { t } } \log p _ { \phi } ( \tilde { \bm { y } } ^ { k + 1 : M } | \bm { x } _ { t } , \tilde { \bm { y } } ^ { 1 : k } ) ,   \nabla _ { \bm { x } _ { t } } \log p _ { \alpha , \tau } ( \tilde { \bm { y } } ^ { k + 1 : M } | \bm { x } _ { t } , \tilde { \bm { y } } ^ { 1 : k } ) \rangle \right]\\= \mathbb { E } _ { p _ { \alpha , \tau } ( \bm { x } _ { t } , \tilde { \bm { y } } ^ { 1 : M } ) } \left[ \frac { 1 } { 2 } | | \nabla _ { \bm { x } _ { t } } \log p _ { \phi } ( \tilde { \bm {y } } ^ { k + 1 : M } | \bm { x } _ { t } , \tilde { \bm { y } } ^ { 1 : k } ) | | _ { 2 } ^ { 2 } \right]\\+   \mathbb { E } _ { p _ { \alpha , \tau } ( \bm { x } _ { t } , \tilde { \bm { y } } ^ { 1 : M } ] } \left[ \frac { 1 } { 2 } | | \nabla _ { \bm { x } _ { t } } \log p _ { \alpha , \tau } ( \tilde { \bm { y } } ^ { k + 1 : M } | \bm { x } _ { t } , \tilde { \bm { y } } ^ { 1 : k} ) | | _ { 2 } ^ { 2 } \right]\\-   \mathbb { E } _ { p _ { \alpha , \tau } ( \bm { x } _ { t } , \tilde { \bm { y } } ^ { 1 : M} ) } \left[ \right. <   \nabla _ { \bm { x } _ { t } } \log p _ { \phi } ( \tilde { \bm { y } } ^ { k + 1 : M } | \bm { x } _ { t } , \tilde { \bm { y } } ^ { 1 : k} ) ,   \nabla _ { \bm { x } _ { t } } \log p _ { \alpha , \tau } ( \bm { x} _ {\boldsymbol t} | {\tilde {\bm {\gamma}}} ^ {\boldsymbol {\delta} / k }, {\tilde {\bm {\gamma}}} ^ {\boldsymbol {\delta} / k + 1 : M}) >\\-   -   [ n a b e r d i s e n t h r m a g e n t h r m a g e n t h r m a g e n t h r m a g e n t h r m a g e n t h r m a g e n t h r m a g e n t h r m a g e n t h r m a g e n t h r m a g e n t h r m a g e n t h r m a g e n t h r m a g e n t H e r m a g e n t h r m a g e n t h r m a g e n t h r m a g e n t h r m a g e n t h r m a g e n t h r m a g e n t h r m a g e n t h r m a g e n t h r m a g e n t h r m a g e n t h r m a g e n t h r m a f o r d i s e n t h r m a g e n t h r m a g e n t h r m a g e n t h r m a g e n t h r m a g e n t h r m a g e n t h r m a g e n t h r m a g e n t h r m a g e n t h r m a g e n t h r m a g e n t h r m a f o r d e f o r d e f o r d e f o r d e f o r d e f o r d e f o r d e f o r d e f o r d e f o r d e f o r d e f o r d e f o r d e f o r d e f o r d e f o r d e f o r d e f o r d e f o r d e f o r d e f o r d u s s u s u s u s u s u s u s u s u s u s u s u s u s u s u s u s u s u s u s u s u s u s u s u s u s u s u s u s u s u s u s u s u s u s u s u s u s u s u s u s u s u s u s u s u s u s u s u s u s u s u s w i v i v i v i v i v i v i v i v i v i v i v i v i v i v i v i v i v i v i v i v i v i v i v i v i v i v i v i v i v i v i v i v i v i v i v i v i v i v i v i v i v i v i v i v i v i v i v i v i v i v i .\\=   [   [   [   [   [   [   [   [   [   [   [   [   [   [   [   [   [   [   [   [   [   [   [   [   [   [   [   [   [   [   [   [   [   [   [   [   [   [   [   [   [   [   [   [   [   [   [   [   [   [   ]     [    ]    ]    ]    ]    ]    ]    ]    ]    ]    ]    ]    ]    ]    ]    ]    ]    ]    ]    ]    ]    ]    ]    ]    ]    ]    ]    ]    ]    ]    ]    ]    ]    ]    ]    ]    ]    ]    ]    ]    ]    ]    ]    ]    ]    ]    ]    ]    ]    ]    ]    ].\\- E P _ \substack \rho_ {\alpha , T} (\bm \textit \textbf \textit \textbf \textit \textbf \textit \textbf \textit \textbf \textit \textbf \textit \textbf \textit \textbf \textit \textbf \textit \textbf \textit \textbf \textit \textbf \textit \textbf \textit \textbf \textit \textbf \textit \textbf \textit \textbf \textit\textbf \textit \textbf \textit \textbf \textit\textbf \textit\textbf \textit\textbf \textit\textbf\textit\textbf\textit\textbf\textbf\textit\textbf\textit\textbf\textit\textbf\textit\textbf\textit\textbf\textit\textbf\textit\textbf\textit\textbf\textit\textbf\textit\textbf\textit\textbf\textit{\textbf{\textit{\textbf{\textit{\textbf{\textit{\textbf{\textit{\textbf{\textit{\textbf{}}}}}}}}}} }\nabla_{\boldsymbol{x}_{t}}}\log p_{\alpha,\tau}(|\boldsymbol{x}_{t}|,|\boldsymbol{x}_{t}|) | |\boldsymbol{x}_{t}|) | |\boldsymbol{x}_{t}|)\\[\nabla_{\boldsymbol{x}_{t}}|\log p_{\alpha,\tau}(\boldsymbol{x}_{t}|\tilde{y}^{\boldsymbol{k+1:M}})|)\nabla_{\boldsymbol{x}_{t}}|\log p_{\alpha,\tau}(\boldsymbol{x}_{t}|\tilde{y}^{\boldsymbol{k+1:M}})|)\end{array}\tag{34}
$$

(35)

(36)

(37)

Simplifying the Term 1 further:

$$
\begin{array} { l } - \mathbb { E } _ { p _ { \alpha , \tau } ( \boldsymbol { x } _ { t } , \tilde { \boldsymbol { y } } ^ { 1 : M } ) } \big [ \langle \nabla _ { \boldsymbol { x } _ { t } } \log p _ { \phi } ( \tilde { \boldsymbol { y } } ^ { k + 1 : M } | \boldsymbol { x } _ { t } , \tilde { \boldsymbol { y } } ^ { 1 : k } ) , \\ \nabla _ { \boldsymbol { x } _ { t } } \log p _ { \alpha , \tau } ( \boldsymbol { x } _ { t } | \tilde { \boldsymbol { y } } ^ { 1 : M } ) \rangle \big ] \\ = - \int _ { \boldsymbol { x } _ { t } } \int _ { \tilde { \boldsymbol { y } } ^ { 1 : M } } p _ { \tau } ( \tilde { \boldsymbol { y } } ^ { 1 : M } ) p _ { \alpha , \tau } ( \boldsymbol { x } _ { t } | \tilde { \boldsymbol { y } } ^ { 1 : M } ) \\ \langle \nabla _ { \boldsymbol { x } _ { t } } \log p _ { \phi } ( \tilde { \boldsymbol { y } } ^ { k + 1 : M } | \boldsymbol { x } _ { t } , \tilde { \boldsymbol { y } } ^ { 1 : k } ) , \frac { \nabla _ { \boldsymbol { x } _ { t } } p _ { \alpha , \tau } ( \boldsymbol { x } _ { t } | \tilde { \boldsymbol { y } } ^ { 1 : M } ) } { p _ { \alpha , \tau } ( \boldsymbol { x } _ { t } | \tilde { \boldsymbol { y } } ^ { 1 : M } ) } \rangle d \tilde { \boldsymbol { y } } ^ { 1 : M } d \boldsymbol { x } _ { t } \\ = - \int _ { \boldsymbol { x } _ { t } } \int _ { \tilde { \boldsymbol { y } } ^ { 1 : M } } p _ { \tau } ( \tilde { \boldsymbol { y } } ^ { 1 : M } ) \langle \nabla _ { \boldsymbol { x } _ { t } } \log p _ { \phi } ( \tilde { \boldsymbol { y } } ^ { k + 1 : M } | \boldsymbol { x } _ { t } , \tilde { \boldsymbol { y } } ^ { 1 : k } ) , \\ \nabla _ { \boldsymbol { x } _ { t } } \int _ { \boldsymbol { x _ { 0 } } } p _ { 0 , \tau } ( \boldsymbol { x _ { 0 }} | \tilde { \boldsymbol { y } } ^ { 1 : M } ) p _ { \alpha , \tau } ( \boldsymbol { x _ { t }} | \boldsymbol { x _ { 0 } }, \tilde { \boldsymbol { y } } ^ { 1 : M } ) d \boldsymbol { x _ { 0 }} \rangle d \tilde { \boldsymbol { y } } ^ { 1 : M } d \boldsymbol { x _ { t } } \\ = - \int _ { \boldsymbol { x _ { t } } } \int _ { \tilde { \boldsymbol { y } } ^ { 1 : M } } p _ { \tau } ( \tilde { \boldsymbol { y } } ^ { 1 : M } ) \langle \nabla _ { \boldsymbol { x _ { t } } } \log p _ { \phi } ( \tilde { \boldsymbol { y } } ^ { k + 1 : M } | \boldsymbol { x _ { t } }, \tilde { \boldsymbol { y } } ^ { 1 : k } ) , \\ \nabla _ { \boldsymbol { x _ { t } } } \int _ { \boldsymbol { x _ { 0 } } } \int _ { \boldsymbol { y ^ { 1 : M } } } p _ { 0 , \tau } ( \boldsymbol { x _ { 0 }} | \tilde { \boldsymbol { y } } ^ { 1 : M } ) p _ {\alpha , \tau} ( \boldsymbol { x _ { t }} | \boldsymbol { x _ { 0 } }, \tilde { \boldsymbol { y ^ { 1 : M} } }, \boldsymbol { y ^ { 1 : M} }) . \\ p ( \boldsymbol { y ^ { 1 : M }} | \boldsymbol { x _ { 0 } }, \tilde { \boldsymbol { y ^ { 1 : M} ) d y ^ { 1 : M }} d x _ { 0 }} ) d \tilde { y ^ { 1 : M }} d x _ { t} \\ = - \int _ { \boldsymbol { x _ { t } } } \int _ { \tilde { y ^ { 1 : M } } } \int _ { \boldsymbol { x _ { 0 } } } \int _ { y ^ { 1 : M }} p _ {\tau} ( x _ {\boldsymbol {{ 0 }}}, x _ {\boldsymbol {{ t }}}, y ^ {\texttt {{ 1 : M }}}, \tilde {\boldsymbol {{y}}} ^ {\texttt {{ 1 : M }}} ) \\ <   & <   \\ & <   \\ & <   \\ & <   \\ & <   \\ & <   \\ & <   \\ & <   \\ & <   \\ & <   \\ & <   \\ & <   \\ & <   \\ & <   \\ & <   \\ & <   \\ & <   \\ & <   \\ & <   \\ & <   \\ & <   \\ & <   \\ & <   \\ & <   \\ & <   \\ & <   ; \\ & <   \\ & <   \\ & <   \\ & <   \\ & <   \\ & <   \\ & <   \\ & <   \\ & <   \\ & <   \\ & <   \\ & <   \\ & <   \\ & <   \\ & <   \\ & <   \\ & <   \\ & <   \\ & <   \\ & <   \\ & <   \\ & <   \\ & <   \\ & <   \\ & <    \\ & <    \\ & <    \\ & <    \\ & <    \\ & <    \\ & <    \\ & <    \\ & <    \\ & <    \\ & <    \\ & <    \\ & <    \\ & <    \\ & <    \\ & <    \\ & <    \\ & <    \\ & <    \\ & <    \\ & <    \\ & <    \\ & <    \\ & <    \\ & <    \\ & <   \\ & <    \\ & <    \\ & <    \\ & <    \\ & <    \\ & <    \\ & <    \\ & <    \\ & <    \\ & <    \\ & <    \\ & <    \\ & <    \\ & <    \\ & <    \\ & <    \\ & <    \\ & <    \\ & <    \\ & <    \\ & <    \\ & <    \\ & <    \\ & <   \\ & <   \\ & <    \\ & <    \\ & <    \\ & <    \\ & <    \\ & <    \\ & <    \\ & <    \\ & <    \\ & <    \\ & <    \\ & <    \\ & <    \\ & <    \\ & <    \\ & <    \\ & <    \\ & <    \\ & <    \\ & <    \\ & <    \\ & <    \\ & <   \\ & <    \\ & <   \\ & <    \\ & <    \\ & <    \\ & <    \\ & <    \\ & > - E _ {\rho_ {\alpha , \tau} (\boldsymbol {{x}} _ {{0}}, {\boldsymbol {{x}} _ {{t}}}, {\boldsymbol {{y}} ^ {\texttt {{ 1 : M}}}}, {\tilde {\boldsymbol {{y}} ^ {\texttt {{ 1 : M}}}}})} [<     > - E _ {\rho_ {\alpha , \tau} (\boldsymbol {{x}} _ {{0}}, {\boldsymbol {{x}} _ {{t}}}, {\boldsymbol {{y}} ^ {\texttt {{ 1 : M}}}}, {\tilde {\boldsymbol {{y}} ^ {\texttt {{ 1 : M}}}}})} [<     > - E _ {\rho_ {\alpha , \tau} (\boldsymbol {{x}} _ {{0}},{\boldsymbol {{x}} _ {{t}}}, {\boldsymbol {{y}} ^ {\texttt {{ 1 : M}}}}, {\tilde {\boldsymbol {{y}} ^ {\texttt {{ 1 : M}}}}})} [<     > - E _ {\rho_ {\alpha , \tau} (\boldsymbol {{x}} _ {{0}}, {\boldsymbol {{x}} _ {{t}}}, {\boldsymbol {{y}} ^ {\texttt {{ 1 := M}}}}, {\tilde {\boldsymbol {{y}} ^ {\texttt {{ 1 := M}}}}})} [<     > - E _ {\rho_ {\alpha , \tau} (\boldsymbol {{x}} _ {{0}}, {\boldsymbol {{x}} _ {{t}}}, {\boldsymbol {{y}} ^ {\texttt {{ 1 := M}}}}, {\tilde {\boldsymbol {{y}} ^ {\texttt {{ 1 := M}}}}})} [<     > - E _ {\rho_ {\alpha , \tau} (\boldbf {{x}} _ {{0}}, {\boldsymbol {{x}} _ {{t}}} , {\boldsymbol {{y}} ^ {\texttt {{ 1 := M}}}}, {\tilde {\boldsymbol {{y}} ^ {\texttt {{ 1 := M}}}}})} [<     > - E _ {\rho_ {\alpha , \tau} (\boldsymbol {{x}} _ {{0}}, {\boldsymbol {{x}} _ {{t}}} , {\boldsymbol {{y}} ^ {\texttt {{ 1 := M}}}}, {\tilde {\boldsymbol {{y}} ^ {\texttt {{ 1 := M}}}}})} [<     > - E _ \rho_ {\alpha , \tau} (\boldsymbol {{x}} _ {{0}}, {\boldsymbol {{x}} _ {{t}}} , {\boldsymbol {{y}}} ^ {\texttt {{ k + 1 : M}}} ] [<     > - E _ \rho_ {\alpha , \tau} (\boldsymbol {{x}} _ {{0}}, {\boldsymbol {{x}} _ {{t}}} , {\boldsymbol {{y}}} ^ {\texttt {{ k + 1 : M}}} ] [<     > - E _ \rho_ {\alpha , \tau} (\boldsymbol {{x}} _ {{0}}, {\boldsymbol {{x}} _ {{t}}} , {\boldsymbol {{y}}}^{\texttt {{ k + 1 : M}}} ] [<     > - E _ \rho_ {\alpha , \tau} (\boldsymbol {{x}} _ {{0}}, {\boldsymbol {{x}} _ {{t}}} , {\boldsymbol {{y}}}^{\texttt {{ k + 1 : M}}} ] [<     > - E _ {\rho_ {\alpha , \tau} (\boldsymbol {{x}} _ {{0}}, {\boldsymbol {{x}}_{\texttt {} n}}, {\tilde {\mathbf {} n}}} ] [<     > - E _ {\rho_ {\alpha , z} (\boldsymbol {{x}} _ {{0}}, {\boldsymbol{{\bf x}}} ^{\texttt {} n}, {\tilde{\mathbf {} n}}} ] [<     > - E _ {\rho_ {\alpha , z} (\boldsymbol {{x}} _ {{0}}, {\bf x} ^{\texttt {} n}, {\tilde{\mathbf {} n}}} ] [<     > - E _ {\rho_ {\alpha , z} (\boldsymbol {{x}} _ {{0}}, {\bf x} ^{\texttt {} n}, {\tilde{\mathbf {} n}}} ] [<     > - E _ {\rho_ {\alpha , z} (\boldsymbol {{x}} _ {{0}}, {\bf x} ^{\texttt {} n},{\tilde{\mathbf {} n}}} ] [<     > - E _ {\rho_ {\alpha , z} (\bf x, a)} [<     > - E _ {\rho_ {\alpha , z} (\bf x, a)}, a ] ] [<     > - E _ {\rho_ {\alpha , z} (\bf x, a)}, a ] ] [<     > - E _ {\rho_ {\alpha , z} (\bf x, a)}, a ] ] [<     > - E _ {\rho_ {\alpha , z} (\bf x, a)}, a ] ] [<     > - E _ {\rho_ {\alpha , z} (\bf x, a)}, a ] ] [<     ] [<     ] [<     ] [<     ] [<     ] [<     ] [<     ] [<     ] [<     ] [<     ] [<     ] [<     ] [<     ] [<     ] [<     ] [<     ] [<     ] [<     ] [<     ] [<     ] [<     ] [<     ] [<     ] [<     ] [<     ] [< |content_end|>
$$

Plugging this back into Equation 37, we get-

$$
\begin{array}{l} \mathcal {D} _ {F} \left(p _ {\phi} (\tilde {\boldsymbol {y}} ^ {k + 1: M} | \boldsymbol {x} _ {t}, \tilde {\boldsymbol {y}} ^ {1: k}) \| p _ {\alpha , \tau} (\tilde {\boldsymbol {y}} ^ {k + 1: M} | \boldsymbol {x} _ {t}, \tilde {\boldsymbol {y}} ^ {1: k})\right) \\ = \mathbb {E} _ {p _ {\alpha , \tau} (\boldsymbol {x} _ {t}, \tilde {\boldsymbol {y}} ^ {1: M})} \left[ \frac {1}{2} \| \nabla_ {\boldsymbol {x} _ {t}} \log p _ {\phi} (\tilde {\boldsymbol {y}} ^ {k + 1: M} | \boldsymbol {x} _ {t}, \tilde {\boldsymbol {y}} ^ {1: k}) \| _ {2} ^ {2} \right] \\ + \mathbb {E} _ {p _ {\alpha , \tau} (\boldsymbol {x} _ {t}, \tilde {\boldsymbol {y}} ^ {1: M})} \left[ \frac {1}{2} \| \nabla_ {\boldsymbol {x} _ {t}} \log p _ {\alpha , \tau} (\tilde {\boldsymbol {y}} ^ {k + 1: M} | \boldsymbol {x} _ {t}, \tilde {\boldsymbol {y}} ^ {1: k}) \| _ {2} ^ {2} \right] \\ + \mathbb {E} _ {p _ {\alpha , \tau} (\boldsymbol {x} _ {t}, \tilde {\boldsymbol {y}} ^ {1: M)}} \left[ \Bigg \langle \nabla_ {\boldsymbol {x} _ {t}} \log p _ {\phi} (\tilde {\boldsymbol {y}} ^ {k + 1: M} | \boldsymbol {x} _ {t}, \tilde {\boldsymbol {y}} ^ {1: k}), \right. \\ \left. \nabla_ {\boldsymbol {x} _ {t}} \log p _ {\alpha , \tau} (\boldsymbol {x} _ {t} | \tilde {\boldsymbol {y}} ^ {1: k}) \Bigg \rangle \right] \\ - \mathbb {E} _ {p _ {\alpha , \tau} (\boldsymbol {x} _ {0}, \boldsymbol {x} _ {t}, \boldsymbol {y} ^ {1: M}, \tilde {\boldsymbol {y}} ^ {1: M})} \left[ \Bigg \langle \nabla_ {\boldsymbol {x} _ {t}} \log p _ {\phi} (\tilde {\boldsymbol {y}} ^ {k + 1: M} | \boldsymbol {x} _ {t}, \tilde {\boldsymbol {y}} ^ {1: k}), \right. \\ \left. \nabla_ {\boldsymbol {x} _ {t}} \log p (\boldsymbol {x} _ {t} | \boldsymbol {x} _ {0}) \Bigg \rangle \right] \end{array}\tag{38}
$$

$$
\begin{array}{l} \text { Here, } \quad \mathbb {E} _ {p _ {\alpha , \tau} (\boldsymbol {x} _ {t}, \tilde {\boldsymbol {y}} ^ {1: M})} \left[ \frac {1}{2} | | \nabla_ {\boldsymbol {x} _ {t}} \log p _ {\alpha , \tau} (\tilde {\boldsymbol {y}} ^ {k + 1: M} | \boldsymbol {x} _ {t}, \tilde {\boldsymbol {y}} ^ {1: k}) | | _ {2} ^ {2} \right] \\ \text { is   a   constant. Further, adding the constant } \\ \mathbb {E} _ {p _ {\alpha , \tau} (\boldsymbol {x} _ {t}, \tilde {\boldsymbol {y}} ^ {1: k})} \left[ \frac {1}{2} | | \nabla_ {\boldsymbol {x} _ {t}} \log p _ {\alpha , \tau} (\boldsymbol {x} _ {t} | \tilde {\boldsymbol {y}} ^ {1: k}) - \nabla_ {\boldsymbol {x} _ {t}} \log p _ {\alpha} (\boldsymbol {x} _ {t} | \boldsymbol {x} _ {0}) | | _ {2} ^ {2} \right] \\ \text { to Equation   38, we get: } \end{array}
$$

$$
\begin{array}{l} \mathcal {D} _ {F} \left(p _ {\phi} (\tilde {\boldsymbol {y}} ^ {k + 1: M} | \boldsymbol {x} _ {t}, \tilde {\boldsymbol {y}} ^ {1: k}) \parallel p _ {\alpha , \tau} (\tilde {\boldsymbol {y}} ^ {k + 1: M} | \boldsymbol {x} _ {t}, \tilde {\boldsymbol {y}} ^ {1: k})\right) \\ = \mathbb {E} _ {p _ {\alpha , \tau} (\boldsymbol {x} _ {t}, \tilde {\boldsymbol {y}} ^ {1: M})} \left[ \frac {1}{2} \left\| \nabla_ {\boldsymbol {x} _ {t}} \log p _ {\phi} (\tilde {\boldsymbol {y}} ^ {k + 1: M} | \boldsymbol {x} _ {t}, \tilde {\boldsymbol {y}} ^ {1: k}) \right\| _ {2} ^ {2} \right] \\ + \mathbb {E} _ {p _ {\alpha , \tau} (\boldsymbol {x} _ {t}, \tilde {\boldsymbol {y}} ^ {1: M})} \left[ \Big \langle \nabla_ {\boldsymbol {x} _ {t}} \log p _ {\phi} (\tilde {\boldsymbol {y}} ^ {k + 1: M} | \boldsymbol {x} _ {t}, \tilde {\boldsymbol {y}} ^ {1: k}), \right. \\ \left. \nabla_ {\boldsymbol {x} _ {t}} \log p _ {\alpha , \tau} (\boldsymbol {x} _ {t} | \tilde {\boldsymbol {y}} ^ {1: k}) \Big \rangle \right] \\ - \mathbb {E} _ {p _ {\alpha , \tau} (\boldsymbol {x} _ {0}, \boldsymbol {x} _ {t}, \boldsymbol {y} ^ {1: M}, \tilde {\boldsymbol {y}} ^ {1: M})} \left[ \Big \langle \nabla_ {\boldsymbol {x} _ {t}} \log p _ {\phi} (\tilde {\boldsymbol {y}} ^ {k + 1: M} | \boldsymbol {x} _ {t}, \tilde {\boldsymbol {y}} ^ {1: k}), \right. \\ \left. \ntriangledown_ {\boldsymbol {x} _ {t}} \log p _ {\alpha} (\boldsymbol {x} _ {t} | \boldsymbol {x} _ {0}) \Big \rangle \right] \end{array}
$$

$$
\begin{array}{r l} & {+ \mathbb {E} _ {p _ {\alpha , \tau} (\pmb {x} _ {t}, \tilde {\pmb {y}} ^ {1: k})} \bigg [ \frac {1}{2} \left\| \nabla_ {\pmb {x} _ {t}} \log p _ {\alpha , \tau} (\pmb {x} _ {t} | \tilde {\pmb {y}} ^ {1: k}) \right.} \\ & {\qquad - \nabla_ {\pmb {x} _ {t}} \log p _ {\alpha} (\pmb {x} _ {t} | \pmb {x} _ {0}) \| _ {2} ^ {2} \bigg ] + C} \\ & {= \mathbb {E} _ {p _ {\alpha , \tau} (\pmb {x}, \pmb {x} _ {t}, \pmb {y} ^ {1: M}, \tilde {\pmb {y}} ^ {1: M})} \bigg [ \frac {1}{2} \Big \| \nabla_ {\pmb {x} _ {t}} \log p _ {\phi} (\tilde {\pmb {y}} ^ {k + 1: M} | \pmb {x} _ {t}, \tilde {\pmb {y}} ^ {1: k})} \\ & {\qquad + \nabla_ {\pmb {x} _ {t}} \log p _ {\theta} (\pmb {x} _ {t} | \tilde {\pmb {y}} ^ {1: k}) - \nabla_ {\pmb {x} _ {t}} \log p _ {\alpha} (\pmb {x} _ {t} | \pmb {x}) \Big \| _ {2} ^ {2} \bigg ] + C} \end{array}\tag{39}
$$

(40)

Simplifying $\nabla _ { \pmb { x } _ { t } }$ log $p _ { \alpha } ( \pmb { x } _ { t } | \pmb { x } )$ to $- \epsilon _ { 0 } / \sqrt { 1 - \bar { \alpha _ { t } } } .$ , where $\epsilon _ { 0 } ~ \sim ~ \mathcal { N } ( 0 , \mathcal { T } )$ , and taking the $\left( 1 \mathrm { ~ - ~ } \bar { \alpha _ { t } } \right)$ times the DSM objective [33], we obtain the simplified diffusion loss:

$$
\begin{array}{c} \mathcal {L} _ {r e s} ^ {t} (\phi) = \mathbb {E} _ {p _ {\tau} (\boldsymbol {x}, \boldsymbol {y} ^ {1: M}, \tilde {\boldsymbol {y}} ^ {1: M})} \mathbb {E} _ {\epsilon_ {0} \sim \mathcal {N} (0, \mathcal {I})} \bigg [ \big \| \epsilon_ {0} \\ - \epsilon_ {\theta} (\boldsymbol {x} _ {t}, \tilde {\boldsymbol {y}} ^ {1: k}, t) + \hat {\epsilon} _ {\phi} (\tilde {\boldsymbol {y}} ^ {k + 1: M}, \boldsymbol {x} _ {t}, \tilde {\boldsymbol {y}} ^ {1: k}, t) \big \| _ {2} ^ {2} \big ] + C \end{array}\tag{41}
$$

## C. Architecture and Implementation Details

All transformer-based models are trained over 2000 epochs for visual tasks and 3000 epochs for low-dimensional tasks. Unet [42] is trained over 3000 epochs for visual tasks and 5000 epochs for low-dimensional tasks. We train models on visual tasks with a batch size of 64, and low-dimensional tasks with a batch size of 256. All models are trained on NVIDIA A5000 or A40 GPUs, with training times ranging from 6 to 12 hours depending on model size and the number of camera inputs. Our current implementations support an action prediction latency of ∼50ms for DP-DiT, ∼100ms for UNet [4] and output composition of models as shown in Figure 2 [b] and ∼150ms for FDP model shown in [c].

DP-DiT. We use DiT-S (∼33M parameters) [43] as the base architecture, with 12 layers, 6 heads and a hidden dimension of 6. Peebles et al. [43] specifically show that the conditioning using AdaLn-Zero outperforms other forms of conditioning such as in-context and cross-attention for image generation. However, we observe slightly stronger performance when the weights for the final layer of AdaLn are initialized with a Gaussian. We use different untrained ResNet-18 (∼12M parameters) [57] encoders for each camera and also encode proprioception using a separate encoder. All encoded conditionals across the observation horizon are concatenated before using AdaLn. The model size conditioned on the input images from 2 cameras is ∼56 M parameters. All DiT models are trained using a learning rate of 1e−4 and a weight decay of 1e−3. We also perform exponential moving average (EMA) to reduce the variance in training. The same DiT backbone is used for low-dimensional, visual, and point-cloud tasks. We use the 100-step DDPM [3] noise scheduler suggested by CHi et al. [4] implemented using HuggingFace Diffusers [58]. Sampling is performed using 8-step DDIM [52].

DP-UNet. We use the 1D-UNet [42] implementation from Chi et al. [4]. UNet is trained using a learning rate of 1e−4 and a weight decay of 1e−6. Although the parameter count of the DiT model does not vary significantly with increasing context length due to self-attention, that is not the case with UNet. We use a relatively smaller UNet for low-dimensional tasks and a UNet with a larger channel width for visual tasks. For an observation horizon of 3 and an action horizon of 15 for visuomotor tasks, the parameter count of UNet increases to (∼336M), not including the ResNet weights. UNet uses FiLM [59] layers for conditioning on a single embedding, which is built for different visual inputs and proprioception similar to DP-DiT.

FDP. We experiment with several implementations of FDP in this paper, as shown in Figures 2 [b] and [c]. The simplest implementation shown in [b] simply adds the output of the base and the residual model, with the base kept frozen. The architectures of these models are exactly identical to DP-DiT, except that they are conditioned on different modalities. For Figure 2 [c] used to present the results in the paper, the architecture of the base model is the same as that of DP-DiT. However, the residual model is designed similarly to the ViT [44] architecture. Since we do not denoise the inputs, we encode and pass all the inputs across the observation horizon through self-attention. We condition them on noisy actions using AdaLn. The images are encoded using patch embedding, where we keep the patch size equal to the size of the image to reduce the number of parameters. Crucially, we apply a zero layer on the block outputs of the residual model that are added to the corresponding blocks of the base model. We implement two variants of the zero-layer: a zero-initialized convolutional layer and a zero-initialized linear layer. For the convolutional layer, $\pi _ { b a s e }$ learned on proprioception is of ∼30M parameters and the residual model $\pi _ { r e s }$ with two camera image inputs is of ∼55M parameters. However, the linear zero layer bloats the residual model’s size to ∼290M. Other hyperparameters such as the learning rate, weight decay, and the noise schedule are the same as DP-DiT.

## D. Experimental Setup

RLBench. We train visuomotor policies in RLBench using a multi-camera setup that records 96×96 RGB images, with joint positions as the action modality. We experiment on RLBench in both two and five camera setups (wrist, front, overhead, right-shoulder, and left-shoulder), and with an observation horizon of 2, and an action horizon of 16. Along with modifications to add distractors and color variations (Figure 3), we also create a custom block-pick environment in three different sizes as shown in Figure 11.

Adroit. The Adroit benchmark comprises highdimensional hand manipulation tasks performed using a 24-DoF anthropomorphic hand (see Figure 10). It includes four tasks—Door, Hammer, Pen, and Relocate—that demand fine motor control and complex object interaction. We modify the success condition of the Hammer task, requiring the nail to be within a distance of 0.2 (instead of 0.1) from the board. Each Adroit task is represented by a task-specific low-dimensional state vector. We use an observation horizon of 3 and an action horizon of 15 across all Adroit experiments.

Robomimic. The dataset [47] provides low-dimensional state observations and uses an action space defined as the change in end-effector position and orientation (axis-angle). The benchmark includes four tasks: Lift, Can, Square, and Toolhang, with the latter two requiring higher precision. Following Chi et al. [4], we use an observation horizon of 1 and an action horizon of 10 for all Robomimic experiments.

TABLE IV  
ENVIRONMENT SPECIFICATIONS—OBS/ACT HORIZON USES ON/HM. FOR RLBENCH, M3L AND REAL-WORLD, ENV-OBSERVATION DEPENDS ON #CAMERAS (1/3/5) AND 512-D EMBEDDINGS.

<table><tr><td>Suite</td><td>Task</td><td>Env. Obs. Dim.</td><td>Rob. Obs. Dim.</td><td>Action Dim.</td><td>Max. Len.</td><td># Train Demos</td><td>Obs/Act Horizon</td></tr><tr><td rowspan="4">Robomimic</td><td>Lift</td><td>10</td><td>9</td><td>7</td><td>400</td><td>10/50/100</td><td>o1h10</td></tr><tr><td>Can</td><td>14</td><td>9</td><td>7</td><td>400</td><td>10/50/100</td><td>o1h10</td></tr><tr><td>Square</td><td>14</td><td>9</td><td>7</td><td>400</td><td>10/50/100</td><td>o1h10</td></tr><tr><td>Toolhang</td><td>44</td><td>9</td><td>7</td><td>700</td><td>10/50/100</td><td>o1h10</td></tr><tr><td rowspan="4">Adroit</td><td>Door</td><td>12</td><td>27</td><td>28</td><td>475</td><td>22</td><td>o2h16</td></tr><tr><td>Hammer</td><td>13</td><td>33</td><td>26</td><td>475</td><td>22</td><td>o2h16</td></tr><tr><td>Pen</td><td>21</td><td>24</td><td>24</td><td>475</td><td>22</td><td>o2h16</td></tr><tr><td>Relocate</td><td>9</td><td>30</td><td>30</td><td>475</td><td>22</td><td>o2h16</td></tr><tr><td>RLBench</td><td>Various</td><td>-</td><td>8</td><td>8</td><td>300/600</td><td>10/50/100</td><td>o3h15</td></tr><tr><td>M3L</td><td>Insertion</td><td>-</td><td>0</td><td>3</td><td>300</td><td>100/200</td><td>o1h1</td></tr><tr><td>Real-World</td><td>Various</td><td>-</td><td>9</td><td>9</td><td>400</td><td>50</td><td>o3h15</td></tr></table>

M3L. We use one 64×64 image input and two 32×32 tactile inputs to the model from the two gripper fingers holding the peg. The rotation of the end effector is frozen, and ∆XYZ is the chosen action space. Various shapes of pegs are provided by the environment, which can be inserted into the respective holes present in differently shaped blocks. In our paper, we show results for models trained on 2 pegs independently, and also on the whole assortment of pegs and blocks.

![](images/1e62d9b74f4db17813d25872cc69361961bea24e7208e0a578575deff4d403a1.jpg)  
Fig. 10. Selected tasks from RLBench and tasks from Adroit are shown.

![](images/7f14b398cb4e6f6e05ee83740d16b9703624b0c492e5f15cb6ab2f8d06346cb5.jpg)  
Fig. 11. Task design for block pick with different scales of variation (dim in meters).

Real Robot Experiments. The task domains used in our real-world experiments as shown in Figures 12 are described below:

![](images/703fad965b4578712a71acb65777eaeaca14e79cfb63d63f3756311091016f0a.jpg)  
Fig. 12. Tasks considered for the real robot experiments. In clockwise direction: original task, with distractors, with camera occlusions, and with color changes.

• Close Drawer: Close the cabinet of an open drawer. We vary the drawer’s placement angle and position relative to the robot within a range of 10<sup>◦</sup> and 15 cm, respectively. This is a relatively simple task where the robot must close the drawer by pushing it with its end-effector.

• Put Block in Bowl: Pick up a block and place it inside a nearby bowl. The positions of both the block and the bowl are varied within a 15 cm range relative to the robot. This task assesses the policy’s ability to perform precise pick-and-place actions.

• Pour in Bowl: Pick up a cup and pour its contents into a nearby bowl. The positions of the cup and the bowl are varied within a 15 cm range relative to the robot. This task evaluates the policy’s effectiveness in operating near joint limits.

• Fold Towel: Fold a kitchen towel placed on a compliant surface. The towel’s position is varied within a 5 cm range relative to the robot. This task evaluates the policy’s capability in deformable object manipulation.

We used ROS1 Noetic for robot software development. For data collection, we used a 3D Connexion SpaceMouse Pro to set end-effector velocity targets, which were executed on the Franka robot using a differential inverse kinematics controller. Time-synchronized joint positions and camera images were recorded at 30Hz for each demonstration and later post-processed by downsampling to 10Hz for policy training. During rollout, we employed a joint position controller to sequentially execute short-horizon trajectory predictions from the policy. We allowed a trajectory length of 400 steps for each task in all our real-world robot experiments. With a horizon length of 16, this resulted in 25 policy inference steps per task.

Robot Safety Check. We implemented a safety check in our robot software to prevent potential damage to the robot during environment variation experiments. This was particularly necessary for DP, which often generated highjerk joint targets in out-of-distribution scenarios. For each joint command, we ensured that the target was within a threshold Euclidean distance from the current joint state, i.e., $| | j _ { \mathrm { t a r g e t } } - j _ { \mathrm { c u r r e n t } } | | \leq 0 . 1 $ . If this condition was violated, policy execution was immediately halted and the rollout was considered a failure.