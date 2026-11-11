# UNCERTAINTY QUANTIFICATION FOR FLOW-BASED GENERALIST ROBOT POLICIES

Ralf Römer<sup>1</sup> Maximilian Seeliger<sup>2</sup> Saida Liu<sup>1</sup> Ben Sturgis<sup>1</sup>

Marco Bagatella<sup>2,3</sup> Daniel Marta<sup>2</sup> Andreas Krause<sup>2</sup> Angela P. Schoellig<sup>1,4</sup>

<sup>1</sup>Technical University of Munich <sup>2</sup>ETH Zurich <sup>3</sup>MPI for Intelligent Systems, Tübingen <sup>4</sup>Robotics Institute Germany

## ABSTRACT

Generalist robot policies, such as vision-language-action models (VLAs) and world action models (WAMs), combine powerful pretrained backbones with expressive generative action heads trained via flow matching on large-scale robotic datasets. Despite their strong empirical performance in robotic manipulation, these policies lack mechanisms to quantify confidence in their predictions and to detect when their actions may be unreliable. This presents a critical limitation for real-world deployment in non-stationary environments, where models inevitably encounter scenarios outside their pretraining distribution and may fail without warning. To address this, we derive an efficient method to quantify epistemic uncertainty in flow-matching models by leveraging velocity-field disagreement (VFD) across a small ensemble. We successfully use this uncertainty estimate for detecting failures during deployment and active fine-tuning of flow-based generalist policies. For the latter, we propose SAVE, a simple yet effective method for uncertainty-guided active multitask fine-tuning that reduces the number of costly expert demonstrations required to adapt generalist policies to new tasks. We conduct experiments in simulation and the real world, across VLAs and a WAM. VFD yields bettercalibrated uncertainty estimates predictive of downstream performance and detects failures with 8 pp higher overall accuracy than existing methods. Across three realworld tasks, SAVE improves final average success from 39 % to 47 % with a fixed demonstration budget. Our results show that measuring epistemic uncertainty with VFD enhances both failure awareness and adaptation of generalist robot policies. Project website: tum-lsy.github.io/uq\_generalist\_policies.

## 1 INTRODUCTION

The premise of generalist robot policies (Physical Intelligence et al., 2025; Kim et al., 2026) is to bring the large-scale pre-training paradigm that has been successful in deep learning in recent years (Dosovitskiy et al., 2021; Awais et al., 2025) into the physical world. Driven by the increasing availability of large-scale robotics datasets (O’Neill et al., 2024), generalist policies have demonstrated impressive performance in long-horizon tasks, exhibiting remarkable robustness and emergent zeroshot capabilities (Black et al., 2025). Architecturally, the field has largely converged on combining a pre-trained vision-language or video model backbone with an action-generation diffusion transformer trained via flow matching (FM) (Lipman et al., 2023) to generate continuous robot actions. Flowbased generalist policies provide stable training, relatively fast inference, and the expressivity needed to model the complex, highly multimodal distributions of real-world physical demonstrations.

While generalist policies are pre-trained on large-scale static datasets spanning a broad distribution of objects and tasks (O’Neill et al., 2024; AgiBot-World et al., 2025), deploying these policies in the real world inevitably exposes them to non-stationarity. Users repurpose robots for novel tasks, object appearances may change, and environments evolve. In such scenarios, pre-trained policies may perform well on some tasks but fail completely on others, without explicit knowledge of which tasks are fully understood from pre-training and which require additional data. The inability of these models to communicate what they do not know raises two key problems. First, generalist policies tend to confidently execute erratic actions in out-of-distribution scenarios (Gu et al., 2025) rather than abstaining or asking for help when uncertain (Ren et al., 2023), raising significant safety concerns (Li et al., 2026; Gu et al., 2025). Second, because these policies cannot inherently identify their knowledge gaps, adapting them to new domains currently requires collecting large numbers of human expert demonstrations (NVIDIA et al., 2025), which is prohibitively expensive.

![](images/dea06664e62e7bca5c6ec240b8c0fdad51b7818cb27e70f6a9df2f4b5d24147d.jpg)  
Figure 1: Top: We measure epistemic uncertainty in flow-based generalist policies by measuring scaled differences between ensembled velocity fields. Bottom: VFD enhances the adaptability and reliability of generalist policies in multiple ways. Left: We propose SAVE, an active finetuning method that allocates expert demonstration budget to tasks and initial states with the highest uncertainty. Right: VFD is an effective runtime monitor for detecting failures during deployment.

To address these problems, our work focuses on uncertainty quantification (UQ) for flow-based gener alist policies. Leveraging the generative formulation of these models, we propose an effective method for estimating epistemic uncertainty by measuring divergence between ensembled velocity fields. Building on this estimator, we propose sample-efficient active fine-tuning via velocity-field epistemic uncertainty (SAVE) for flow-based generalist policies. Our main contributions are threefold:

• We propose velocity-field disagreement (VFD), a tractable epistemic uncertainty score for FM models, based on a velocity-field expression for the KL divergence between two FM models.

• We propose SAVE, a framework for uncertainty-guided active fine-tuning of flow-based generalist policies that uses VFD to prioritize tasks and initial states for collecting expert demonstrations.

• We empirically evaluate our methods in simulation and the real world, demonstrating that VFD predicts task performance more reliably and detects deployment failures more accurately than baselines, and that SAVE reduces the amount of expert data required for multitask adaptation.

## 2 RELATED WORK

Generalist Robot Policies. Generalist policies have emerged as a dominant approach to learning complex manipulation tasks by fine-tuning pre-trained vision-language or video backbones on robotic data (Kim et al., 2024; Physical Intelligence et al., 2025; Kim et al., 2026; Yuan et al., 2026). Earlier vision-language-action models (VLAs) map visual and language observations to discrete or continuous action tokens via regression or classification heads (Kim et al., 2024; Zitkovich et al., 2023), but the state of the art has adopted FM as an expressive framework for modeling the highly multimodal action distributions in large-scale datasets (Physical Intelligence et al., 2025; Fang et al., 2026). A related concept, world-action models (WAMs) (Kim et al., 2026; Yuan et al., 2026), leverages the spatiotemporal priors of video models to enhance action generation, typically using latent flow matching. Despite their expressiveness, generalist policies do not natively quantify uncertainty and thus cannot identify situations in which they lack knowledge of which action to perform (Agia et al., 2024; Lee et al., 2025), preventing safe, autonomous real-world deployment. This motivates estimating where the model is most uncertain: before deployment, to prioritize situations for querying an expert, and during deployment, to detect failures.

Uncertainty Quantification for Generative Models. Uncertainty in deep learning is typically decomposed into epistemic (model) and aleatoric (data) uncertainty (Hüllermeier & Waegeman, 2021). While the latter is an irreducible property of the data generation process, the epistemic component can be reduced by collecting new and, importantly, the right data. Depending on the model class, UQ has been instantiated in many different ways. Test-time dropout (Loquercio et al., 2020) and inference on augmented inputs (Ayhan & Berens, 2018) may, for instance, be applied directly at test time and are mostly architecture-agnostic. More powerful approaches, such as deep ensembles (Lakshminarayanan et al., 2017) and Laplace approximations (Daxberger et al., 2021), require training multiple models or modifying the training procedure. With the rise of modern generative models (Ho et al., 2020; Lipman et al., 2023), which, despite their expressivity, often remain overconfident (Karczewski et al., 2025), several recent works have investigated UQ for this model class. These established approaches have recently been extended to modern generative models such as diffusion and flow matching (Franchi et al., 2025), which often remain overconfident Nalisnick et al. (2019); Karczewski et al. (2025). Among recent approaches, Jazbec et al. (2025) apply a lastlayer Laplace approximation to diffusion models and estimate uncertainty in a semantic representation space using a Gaussian approximation to the posterior predictive, which may, however, be unsuitable for multi-modal distributions. DECU (Berry et al., 2024) trains an ensemble of latent diffusion models and performs pairwise distance estimation using the denoising means. Other recent works rely on similarity metrics in language space (Franchi et al., 2025), leverage hypernetworks (Chan et al., 2024), or train explicit confidence predictors (Mei et al., 2025). Closer to our setting, Ju et al. (2026) estimate epistemic uncertainty for pre-trained VLMs through a Riemannian formulation of FM, but do not target action generation. UQ is also an active area of research in autoregressive models (Shorinwa et al., 2025), with domain-specific techniques spanning from differential entropy over vocabulary (Ling et al., 2024), to perplexity (Fadeeva et al., 2024), and Dirichlet evidence (Ma et al., 2025), but these methods primarily measure aleatoric uncertainty, i.e., data ambiguity.

Active Learning. A direct application of reliable uncertainty estimates is in active data selection, a well-established research field with deep roots in experimental design (Chaloner & Verdinelli, 1995) and active learning (Settles, 2009), including Bayesian disagreement-based acquisition for deep models (Gal et al., 2017; Kirsch et al., 2019). While a large body of literature has proposed domain-agnostic approaches (Ash et al., 2020; Holzmüller et al., 2023), several recent works have focused on active learning in MDPs, particularly for learning policies (Judah et al., 2012; He et al., 2025). Several existing approaches operate offline (Hejna et al., 2025; Agia et al., 2025) and filter a fixed dataset based on mutual information between states and actions or influence detection in a static setting. A long-running line of research (Cui et al., 2019; Karli et al., 2025) operates instead in an online setting and aims to infer when a human labeling effort is required. Closer to our approach, AMF (Bagatella et al., 2025) also focus on an iterative setting but propose an algorithm that queries entire expert demonstrations to maximize information gain over multi-task expert trajectories. While this approach has strong regret guarantees, it does not scale well to generalist policies, unlike our proposed method.

Failure Detection for Generative Policies. A common proxy for detecting failures of policies trained via imitation learning is to identify out-of-distribution (OOD) inputs in the policy’s observation space. Xu et al. (2025) fit a continuous normalizing flow to observation embeddings from successful rollouts and score new observations by their likelihood under the learned distribution, while FIPER (Römer et al., 2025) uses random network distillation (Burda et al., 2019) to detect observation embeddings that fall outside the successful-rollout distribution. However, for modern generalist policies, distribution shift in the input observations is not, by itself, a reliable indicator of impending failure. Other approaches query an external VLM to monitor task progress, but the computational overhead limits real-time deployment (Duan et al., 2025; Du et al., 2023; Agia et al., 2024). Gu et al. (2025) show that VLA latent features separate successful from failed rollouts across tasks and train a lightweight supervised classifier on these features. However, their method requires labeled failure data, which can be unsafe and costly to collect on real robots.

## 3 PRELIMINARIES

Bayesian Uncertainty Quantification. Given a dataset $\mathcal { D } = \{ ( \pmb { x } ^ { ( n ) } , \pmb { y } ^ { ( n ) } ) \} _ { n = 1 } ^ { N }$ and a neural network parameterized by $\theta \in \Theta$ , trained to learn the relationship between $\pmb { y } \in \mathcal { V }$ and $\mathbf { \boldsymbol { x } } \in \mathcal { X }$ , Bayesian modeling considers the posterior predictive distribution

$$
p (\boldsymbol {x} \mid \boldsymbol {y}, \mathcal {D}) = \int p (\boldsymbol {x} \mid \boldsymbol {y}, \boldsymbol {\theta}) p (\boldsymbol {\theta} \mid \mathcal {D}) \mathrm{d} \boldsymbol {\theta},\tag{1}
$$

where $p ( \pmb \theta | \mathcal { D } )$ is the posterior and generally intractable for deep neural networks (Hüllermeier & Waegeman, 2021). Therefore, (1) has to be approximated by a Monte-Carlo (MC) estimate $\begin{array} { r } { p ( \pmb { x } \mid \pmb { y } , \mathcal { D } ) \approx \frac { 1 } { M } \sum _ { i = 1 } ^ { M } p ( \pmb { x } \mid \pmb { y } , \pmb { \theta } _ { i } ) } \end{array}$ , where weight samples from the posterior $\pmb { \theta } _ { 1 } , \dots , \pmb { \theta } _ { M } \sim p ( \pmb { \theta } | \mathcal { D } )$ can be obtained using approaches such as ensembling (Lakshminarayanan et al., 2017) or dropout (Gal & Ghahramani, 2016). The total uncertainty in the generation x for an input y is given by the entropy of the predictive distribution (1), $\begin{array} { r } { H \big [ p \big ( { \pmb x } | { \pmb y } , \mathcal { D } \big ) \big ] = - \int p ( { \pmb x } | { \dot { \pmb y } } , \mathcal { D } ) } \end{array}$ log p(x  y, )dx, which includes both irreducible aleatoric uncertainty and epistemic uncertainty, which can be reduced by collecting additional data. Subtracting the aleatoric component, given by the expected entropy, yields the epistemic uncertainty

$$
I (\boldsymbol {x}, \boldsymbol {\theta} \mid \boldsymbol {y}, \mathcal {D}) = H [ p (\boldsymbol {x} \mid \boldsymbol {y}, \mathcal {D}) ] - \mathbb {E} _ {\boldsymbol {\theta}} [ H [ p (\boldsymbol {x} \mid \boldsymbol {y}, \boldsymbol {\theta}) ] ] = \mathbb {E} _ {\boldsymbol {\theta}} [ D _ {\mathrm{KL}} (p (\boldsymbol {x} \mid \boldsymbol {y}, \boldsymbol {\theta}) | | p (\boldsymbol {x} \mid \boldsymbol {y}, \mathcal {D})) ],\tag{2}
$$

which equals the mutual information between x and $\pmb \theta$ and can be viewed as the expected disagreement between a model learned from the data and the true posterior predictive (1) at a test input $\mathbf { \pmb { y } } .$

Flow Matching. Generative modeling aims to learn an unknown distribution q from a set of samples $\mathcal { D } \sim q$ . We first consider unconditional generation, where $\mathcal { D } = \{ \pmb { x } ^ { ( n ) } \} _ { n = 1 } ^ { N } , \pmb { x } ^ { ( n ) } \in \mathbb { R } ^ { d }$ . FM models a time-dependent velocity field $\pmb { u } _ { s } : [ 0 , \overset { \mathbf { \bar { 1 } } } { 1 } ] \times \mathbb { R } ^ { d }  \mathbb { R } ^ { d }$ generating a probability path $( p _ { s } ) _ { 0 \leq s \leq 1 }$ that transforms a source distribution $p _ { 0 }$ (e.g., a Gaussian) to the data distribution $p _ { 1 } = q$ This means the flow $\phi _ { s } : [ 0 , 1 ] \times \mathbb { R } ^ { d }  \mathbb { R } ^ { \dot { d } }$ solving the ODE $\begin{array} { r } { \frac { \mathrm { d } } { \mathrm { d } s } \phi _ { s } ( { \pmb x } ) = { \pmb u } _ { s } ( \phi _ { s } ( { \pmb x } ) ) , \phi _ { 0 } ( { \pmb x } ) = { \pmb x } } \end{array}$ satisfies $\pmb { x } _ { s } = \phi _ { s } ( \pmb { x } _ { 0 } ) \sim p _ { s }$ for $\mathbf { \mathcal { x } } _ { 0 } \sim p _ { 0 }$ . As ${ \pmb u } _ { s } ( { \pmb x } )$ ) is unknown, conditional FM (Lipman et al., 2023) with $\mathcal { D } = \{ ( \pmb { x } ^ { ( n ) } , \pmb { y } ^ { ( n ) } ) \} _ { n = 1 } ^ { N }$ constructs $p _ { s }$ as the marginal of simpler conditional paths $p _ { s } ( { \pmb x } | { \pmb x } _ { 1 } )$ where ${ \pmb x } _ { 1 } \sim q . \mathrm { \ A }$ common choice is optimal transport (OT) Gaussian conditional paths $p _ { s } ( { \pmb x } | { \pmb x } _ { 1 } )$ 二 $\mathcal { N } ( \pmb { x } | s \pmb { x } _ { 1 } , ( \hat { 1 } - s ) ^ { 2 } \pmb { I } )$ , for which the conditional velocity field ${ \pmb u } _ { s } ( { \pmb x } | { \pmb x } _ { 1 } )$ has an analytical expression. A neural network $v _ { s } ^ { \theta }$ is then trained to regress onto this target vector using the loss

$$
\mathcal {L} (\boldsymbol {\theta}) = \mathbb {E} _ {s \sim \operatorname{Unif} ([ 0, 1 ]), (\boldsymbol {x} _ {1}, \boldsymbol {y}) \sim \operatorname{Unif} (\mathcal {D}), \boldsymbol {x} _ {s} \sim p _ {s} (\cdot | \boldsymbol {x} _ {1})} \left[ \left\| \boldsymbol {v} _ {s} ^ {\boldsymbol {\theta}} (\boldsymbol {x} _ {s}, \boldsymbol {y}) - \boldsymbol {u} _ {s} (\boldsymbol {x} _ {s} | \boldsymbol {x} _ {1}) \right\| _ {2} ^ {2} \right].\tag{3}
$$

New samples $\pmb { x } \sim p _ { 1 } ( \cdot | \pmb { y } )$ from the learned distribution can be generated, for example, through Euler integration starting from $\mathbf { \boldsymbol { x } } _ { 0 } \sim p _ { \mathrm { 0 } }$ with step size $\delta s = 1 / N _ { \mathrm { s } }$ and $\pmb { x } _ { s + \delta s } = \pmb { x } _ { s } + \pmb { v } _ { s } ^ { \theta } ( \pmb { x } _ { s } , \pmb { y } ) \delta s$

Flow-Based Generalist Policies. We consider robot policies $\pi _ { \pmb { \theta } } : \mathcal { O }  \Delta ( \mathcal { A } )$ that map multimodal observations $\mathbf { \sigma } _ { o _ { t } } \in \mathcal { O }$ to distributions over actions $\mathbf { \delta } _ { \mathbf { { \pmb { a } } } _ { t } } \in { \mathcal { A } }$ , where an observation $\mathbf { \sigma } _ { \pmb { o } _ { t } } = ( \mathbf { \mathscr { s } } _ { t } , I _ { t } , l )$ typically contains the proprioceptive state $\mathbf { } _ { s _ { t } , }$ camera images $\mathbf { } I _ { t }$ and language instruction l. Flowbased policies are trained through behavior cloning to match the distribution underlying large-scale datasets $\begin{array} { r } { \mathcal { D } = \{ ( \boldsymbol { o } ^ { ( n ) } , \boldsymbol { a } ^ { ( n ) } ) \} _ { n = 1 } ^ { N } \sim q . } \end{array}$ , using the loss (3) with $\mathbf { \boldsymbol { y } } = \mathbf { \boldsymbol { o } } _ { t }$ . The choice of output differs between model classes: VLAs usually generate action chunks $\pmb { x } = \pmb { A } _ { t } = \left( \pmb { a } _ { t } , \pmb { a } _ { t + 1 } , \ldots , \pmb { a } _ { t + H - 1 } \right)$ whereas WAMs are trained to also predict future observations, $\mathrm { i . e . , } x = \left( A _ { t } , O _ { t } \right)$

## 4 ESTIMATING EPISTEMIC UNCERTAINTY OF FLOW-MATCHING MODELS

Training an FM model yields an estimate θ of the model parameters given the data . For a new conditioning input $^ { \mathbf { \Lambda } _ { \mathbf { y } } , }$ how can we quantify whether the model exhibits high epistemic uncertainty that could be reduced by collecting more data? A simple estimate of total predictive uncertainty is the conditional entropy $\mathbf { \tilde { \phi } } _ { H [ p ( \pmb { x } \mid \pmb { y } , \pmb { \theta } ) ] }$ . However, as this quantity conflates epistemic and aleatoric uncertainty, we need to isolate the epistemic component (2). To this end, we denote $p ^ { \pmb { \theta } _ { i } } ( \pmb { x } | \pmb { y } ) =$ $p ( { \pmb x } | { \pmb y } , { \pmb \theta } _ { i } )$ and consider the MC approximation of (2), that is,

$$
I (\boldsymbol {x}, \boldsymbol {\theta} \mid \boldsymbol {y}, \mathcal {D}) \approx \frac {1}{M} \sum_ {i = 1} ^ {M} D _ {\mathrm{KL}} \left(p ^ {\boldsymbol {\theta} _ {i}} (\boldsymbol {x} \mid \boldsymbol {y}) | | p (\boldsymbol {x} \mid \boldsymbol {y}, \mathcal {D})\right)\tag{4a}
$$

$$
\approx \frac {1}{M} \sum_ {i = 1} ^ {M} D _ {\mathrm{KL}} \left(p ^ {\boldsymbol {\theta} _ {i}} (\boldsymbol {x} \mid \boldsymbol {y}) | | \frac {1}{M} \sum_ {j = 1} ^ {M} p ^ {\boldsymbol {\theta} _ {j}} (\boldsymbol {x} \mid \boldsymbol {y})\right)\tag{4b}
$$

$$
\leq \frac {1}{M ^ {2}} \sum_ {i, j = 1} ^ {M} D _ {\mathrm{KL}} \left(p ^ {\boldsymbol {\theta} _ {i}} (\boldsymbol {x} \mid \boldsymbol {y}) | | p ^ {\boldsymbol {\theta} _ {j}} (\boldsymbol {x} \mid \boldsymbol {y})\right),\tag{4c}
$$

where $\pmb { \theta } _ { 1 } , \dots , \pmb { \theta } _ { M } \sim p ( \pmb { \theta } | \mathcal { D } )$ , and (4c) follows by applying Jensen’s inequality to each term in the outer sum, using the convexity of the KL divergence in its second argument. Hence, the average pairwise KL divergence between FM models trained on the same data ${ \mathcal { D } } ,$ evaluated for a conditioning input y, represents an approximate upper bound of the epistemic uncertainty.

In principle, the KL divergence between two FM models can be estimated by computing the likelihoods of individual samples, which entails solving an augmented ODE that involves the divergence of the learned velocity field. However, computing this divergence is prohibitively expensive in high dimensions. To make computation tractable, we first derive a relationship between pairwise KL divergence and the marginal velocity fields. This identity motivates a practical disagreement estimator based on learned velocity fields. For the proof, we refer to Appendix A.2.

Theorem 4.1. Consider two distributions $p ^ { 1 } ( { \pmb x } \mid { \pmb y } ) , p ^ { 2 } ( { \pmb x } \mid { \pmb y } )$ induced by velocity fields ${ \pmb u } _ { s } ^ { 1 } ( { \pmb x } , { \pmb y } )$ ${ \pmb u } _ { s } ^ { 2 } ( { \pmb x } , { \pmb y } )$ with OT Gaussian conditional probability paths satisfying regularity assumptions (cf. Appendix A.2). Then, the KL divergence between the two distributions is

$$
D _ {\mathrm{KL}} \left(p ^ {1} (\boldsymbol {x} \mid \boldsymbol {y}) | | p ^ {2} (\boldsymbol {x} \mid \boldsymbol {y})\right) = \int_ {0} ^ {1} \frac {s}{1 - s} \mathbb {E} _ {\boldsymbol {x} _ {s} \sim p _ {s} ^ {1} (\boldsymbol {x} \mid \boldsymbol {y})} \left[ \left| \left| \boldsymbol {u} _ {s} ^ {1} (\boldsymbol {x} _ {s}, \boldsymbol {y}) - \boldsymbol {u} _ {s} ^ {2} (\boldsymbol {x} _ {s}, \boldsymbol {y}) \right| \right| _ {2} ^ {2} \right] d s.\tag{5}
$$

For trained FM models, we can apply this disagreement functional to the learned velocity fields. In this way, we can approximate the KL divergence terms in (4c) by sampling $\mathbf { \mathcal { x } } _ { 0 } \sim p _ { 0 }$ , performing Euler integration using the learned velocity field $\mathit { \Pi } _ { { v } _ { s } ^ { \theta _ { 1 } } }$ and calculating the velocity-field disagreement (VFD) at intermediate ODE states $\pmb { x } _ { 0 } , \pmb { x } _ { \delta s } , \ldots$ , weighted by $\begin{array} { r } { \kappa _ { s } = \frac { { \bf \bar { s } } } { 1 - s } } \end{array}$ , as

$$
D _ {\mathrm{KL}} \left(p ^ {\boldsymbol {\theta} _ {1}} (\boldsymbol {x} \mid \boldsymbol {y}) | | p ^ {\boldsymbol {\theta} _ {2}} (\boldsymbol {x} \mid \boldsymbol {y})\right) \approx \frac {1}{N _ {s}} \mathbb {E} _ {\boldsymbol {x} _ {0} \sim p _ {0}} \left[ \sum_ {\ell = 0} ^ {N _ {s} - 1} \kappa_ {s _ {\ell}} \left\| \boldsymbol {v} _ {s _ {\ell}} ^ {\boldsymbol {\theta} _ {1}} \left(\boldsymbol {x} _ {s _ {\ell}}, \boldsymbol {y}\right) - \boldsymbol {v} _ {s _ {\ell}} ^ {\boldsymbol {\theta} _ {2}} \left(\boldsymbol {x} _ {s _ {\ell}}, \boldsymbol {y}\right) \right\| _ {2} ^ {2} \right],\tag{6}
$$

where $s _ { \ell } = \ell \delta s$ . Finally, by denoting $\mathcal { V } = ( \pmb { v } _ { s } ^ { \pmb { \theta } _ { 1 } } , \dots , \pmb { v } _ { s } ^ { \pmb { \theta } _ { M } } )$ , plugging (6) into (4c) and averaging over non-identical pairs, we can define the tractable VFD score for epistemic uncertainty estimation

$$
u _ {\mathrm{e}} (\boldsymbol {y}; \mathcal {V}) = \frac {1}{M (M - 1) N _ {s}} \mathbb {E} _ {\boldsymbol {x} _ {0} \sim p _ {0}} \left[ \sum_ {i, j = 1; i \neq j} ^ {M} \sum_ {\ell = 0} ^ {N _ {s} - 1} \kappa_ {s _ {\ell}} \left\| \boldsymbol {v} _ {s _ {\ell}} ^ {\boldsymbol {\theta} _ {i}} (\boldsymbol {x} _ {s _ {\ell}} ^ {(i)}, \boldsymbol {y}) - \boldsymbol {v} _ {s _ {\ell}} ^ {\boldsymbol {\theta} _ {j}} (\boldsymbol {x} _ {s _ {\ell}} ^ {(i)}, \boldsymbol {y}) \right\| _ {2} ^ {2} \right],\tag{7}
$$

where $\pmb { x } _ { s _ { \ell } + \delta s } ^ { ( i ) } = \pmb { x } _ { s _ { \ell } } ^ { ( i ) } + \pmb { v } _ { s _ { \ell } } ^ { \theta _ { i } } \big ( \pmb { x } _ { s _ { \ell } } ^ { ( i ) } , \pmb { y } \big ) \delta s , \pmb { x } _ { 0 } ^ { ( i ) } = \pmb { x } _ { 0 }$ . In practice, we approximate the expectation in (7) by forward-integrating a small batch of actions, leveraging GPU parallelization.

This uncertainty estimate can be used in several ways to make generalist robot policies more reliable and adaptable, with failure detection and active learning being of particular interest. The former deploys uncertainty in a relatively straightforward way, flagging a failure when uncertainty is high (Xu et al., 2025; Römer et al., 2025). Active fine-tuning involves more complex design choices, and we present an effective algorithm that utilizes VFD in the following section.

## 5 UNCERTAINTY-GUIDED ACTIVE MULTITASK FINE-TUNING

We consider the following problem: Given a flow-based generalist policy and a set of tasks, how can we improve overall performance across tasks with minimum data collection effort? More formally, we consider a set of K tasks $\mathcal { T } = \{ T _ { k } \} _ { k = 1 } ^ { K }$ , each of which can be described by an MDP $\mathbf { \dot { \mathcal { M } } } _ { k } = ( \mathcal { O } , \mathcal { A } , P _ { k } , R _ { k } , \rho _ { k } , \gamma )$ , where $P _ { k } : \mathcal { O } \stackrel { \sim } { \times } \bar { \mathcal { A } }  \Delta ( \mathcal { O } ) , R _ { k } : \mathcal { O }  \mathbb { R }$ and $\rho _ { k } \in \Delta ( \mathcal { O } )$ are task-specific dynamics, rewards and initial observation distributions, and $\gamma \in$ $( 0 , 1 )$ ) is a discount factor. Given a policy $\pi : \mathcal { O } \to \Delta ( \mathcal { A } )$ , its single-task performance is simply $\begin{array} { r } { J _ { k } ( \pi ) = \mathbb { E } _ { \pi , P _ { k } , \rho _ { k } } \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } R _ { k } ( o _ { t } ) } \end{array}$ , and its multitask performance is $\begin{array} { r } { J ( \pi ) = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } J _ { k } ( \pi ) } \end{array}$ Our goal is to fine-tune a generalist policy $\pi _ { \theta }$ (pre-trained on large-scale, diverse data) to maximize its average multitask performance. For each task $T _ { k } \in \mathcal T$ , we assume the availability of a pool of $L$ candidate initial observations $\mathcal { O } _ { k } = \{ o _ { k l } \} _ { l = 1 } ^ { L } \sim \rho _ { k }$ which, in practice, could include scenarios the policy might encounter at the start of a rollout. Without any environment interaction, a subset of observations $\hat { \mathcal { O } } \subset \cup _ { k = 1 } ^ { K } \mathcal { O } _ { k }$ needs to be selected, and an expert demonstrator $\pi ^ { * }$ argmax $_ \pi { J ( \pi ) }$ will return a demonstration starting from each of the selected observations: $\tau ^ { * } = ( o _ { 0 } , \pmb { a } _ { 0 } , \dots ) \stackrel {  } { }$ with $\mathbf { } a _ { t } \sim \pi ^ { * } ( o _ { t } )$ and $\pmb { o } _ { t + 1 } \sim P _ { k } ( \pmb { o } _ { t } , \pmb { a } _ { t } )$ for each $\pmb { o } _ { 0 } \in \hat { \mathcal { O } }$

Active data acquisition requires deciding which demonstrations are most valuable to collect. We address this problem by using VFD to measure epistemic uncertainty for candidate initial configurations. Higher values indicate the policy is more uncertain about how to act in a given scene, suggesting limited exposure during pre-training and a potential benefit from additional expert demonstrations.

Generalist Policy Ensemble. The high cost of pre-training foundation models makes training multiple generalist policies from scratch prohibitively expensive. Instead, we take a single pre-trained base policy $\pi _ { \mathfrak { b } }$ and fine-tune it M times on a diverse dataset $\mathcal { D } _ { \mathrm { p r e } }$ , randomly shuffled each time. In this way, we obtain a policy ensemble $\Pi _ { 0 } = ( \pi _ { 0 } ^ { ( 1 ) } , \dots , \pi _ { 0 } ^ { ( M ) } )$ for uncertainty estimation.

Uncertainty-Guided Data Acquisition. We train the ensemble over $R$ rounds of active fine-tuning. At each round r, we evaluate the current ensemble $\Pi _ { r }$ on all candidate initial observations and compute, for all tasks $k \in \{ 1 , \ldots , K \}$ and candidate initial observations $\mathbf { \sigma } _ { o _ { k l } } \in \mathcal { O } _ { k }$ , the VFD epistemic uncertainty $u _ { k l } ^ { ( r ) } = u _ { \mathrm { e } } ( o _ { k l } ; \Pi _ { r } )$ , as defined in $( 7 )$ . A simple strategy would then be to always select the most uncertain candidate observation. However, this approach can overemphasize a small set of outlier scenarios, leading to poor task coverage and small multitask improvement. Thus, we first aggregate uncertainty at the task level using the mean score

$$
U _ {k} ^ {(r)} = \frac {1}{L} \big (u _ {k 1} ^ {(r)} + \dots + u _ {k L} ^ {(r)} \big), \qquad k = 1, \dots , K.\tag{8}
$$

It may then be tempting to directly request a batch of demonstrations from the most uncertain task. However, this is both theoretically flawed, as mutual information would need to be updated as soon as a single new demonstration is collected, and practically suboptimal, as an excessive number of demonstrations could be collected for a simple task. While posterior uncertainty can be computed under strong assumptions about the model class $( \mathrm { e . g . }$ , analytically, without even observing x in the case of Gaussian processes (Bagatella et al., 2025)), its estimation with neural networks remains, in general, intractable. As collecting demonstrations in batches remains practically necessary for computational reasons, we instead introduce diversity by defining a categorical sampling distribution $\operatorname { C a t } ( W ^ { ( r ) } )$ with a temperature parameter $\tau \geq 0$ and weights $W ^ { ( r ) } = ( w _ { 1 } ^ { ( r ) } , \dots , w _ { K } ^ { ( r ) } )$ over tasks, where $\begin{array} { r } { \boldsymbol { w } _ { k } ^ { ( r ) } = \left( \boldsymbol { U } _ { k } ^ { ( r ) } \right) ^ { \tau } / \sum _ { k ^ { \prime } = 1 } ^ { K } \left( \boldsymbol { U } _ { k ^ { \prime } } ^ { ( r ) } \right) ^ { \tau } } \end{array}$ . Larger values of τ concentrate probability mass on the most uncertain tasks, and $\tau = 0$ corresponds to uniform sampling. We perform $n _ { \mathrm { e } }$ expert queries per round and, for each, first sample a task index $k \sim \operatorname { C a t } ( W ^ { ( r ) } )$ , effectively prioritizing tasks with high estimated uncertainty while retaining exploration over all tasks. We then select the most uncertain initial observation within the sampled task,

$$
\pmb {o} _ {k} ^ {*} = \arg \max _ {\pmb {o} _ {k l} \in \mathcal {O} _ {k}} u _ {k l} ^ {(r)},\tag{9}
$$

and request an expert demonstration starting from ${ \pmb { o } } _ { 0 } = { \pmb { o } } _ { k } ^ { * } ; \tau ^ { * } = ( { \pmb { o } } _ { 0 } , { \pmb { a } } _ { 0 } , \ldots )$ . Repeating this procedure $n _ { \mathrm { e } }$ times yields a batch of newly collected demonstrations $\mathcal { D } _ { \mathrm { n e w } } ^ { ( r ) }$

Iterative Fine-Tuning. Fine-tuning a generalist policy solely on new demonstrations can lead to forgetting of previously acquired capabilities (Du et al., 2026; Römer et al., 2026). To avoid this issue, we fine-tune the policy ensemble on a mixture of pre-training and queried data, $\mathcal { D } _ { \mathrm { t r a i n } } ^ { ( r ) } \sim$ λ $\backslash \mathrm { U n i f } ( \mathcal { D } _ { \mathrm { p r e } } ) + ( 1 - \lambda ) \mathrm { U n i f } ( \mathcal { D } _ { \mathrm { n e w } } ^ { ( \leq r ) } )$ , where $\mathcal { D } _ { \mathrm { n e w } } ^ { ( \leq r ) }$ contains all demonstrations collected up to round $r ,$ and $\lambda \in [ 0 , 1 ]$ is the replay ratio. By determining which tasks currently need data most and then selecting the highest-uncertainty initial observations within those tasks, SAVE balances exploitation of uncertain scenes with exploration across tasks for sample-efficient multitask adaptation.

## 6 EXPERIMENTS

Our experiments in simulation and the real world primarily aim at answering three research questions:

Q1: How well-calibrated is VFD, i.e., to what extent do its values reflect the epistemic uncertainty and task-completion capabilities of flow-based generalist policies?

Q2: Can VFD identify where additional data should be collected for active fine-tuning?

Q3: How informative are VFD uncertainty estimates for detecting failures during deployment?

## 6.1 EXPERIMENTAL SETUP

Simulation Environments. We conduct our experiments on three benchmarks. Push-T (Chi et al., 2023) simulates a 2D agent pushing a T-shaped object into a target area. LIBERO (Liu et al., 2023) simulates a Franka manipulator across various household environments, and we focus on its most challenging suite, LIBERO-Long, which includes 10 long-horizon tasks. To evaluate failure detection capabilities, we use LIBERO-Plus (Fei et al., 2026), which introduces seven perturbation dimensions, including object layouts and camera viewpoints, yielding a wide range of failure modes.

Real-World Tasks. We design three tasks for an FR3 robot: 1) Placing a bowl on a plate (Place), 2) stacking a bowl on top of three other bowls (Stack), and 3) placing a Lego brick into a drawer and closing it (Drawer). The tasks consist of two, two, and three phases, respectively (e.g., lifting the bowl and placing it down for Place), and we assign partial credit for each successfully completed subtask.

![](images/1ab694bd55f90b2974bcd07eecccf176b4e0313b1f7bb5889709c0b55babb815.jpg)

![](images/d159e15ca3bb15240a0a8304c6cd9be8c0fe118dc0cfdaa61482729106152d84.jpg)

![](images/5fec1da56768389817398ac217ece78a406022c525b902d506d23b969b8fad17.jpg)  
Figure 2: Real-world tasks.

Policies and Implementation Details. We use three generalist policies: SmolVLA (550M, Shukor et al., 2025) attaches an action expert to a SmolVLM-2 backbone; X-VLA (900M, Zheng et al., 2026) uses soft-prompting for cross-embodiment learning; and FastWAM (6B, Yuan et al., 2026), built on top of a Wan2.2-5B video DiT backbone, is trained with a joint video modeling and action generation objective. The three policies are conditioned on a language command, two camera images, and the proprioceptive state. In Push-T, we use a simple FM-Policy trained from scratch. All policies have an ODE step size of $\delta s = 0 . 1$ , and we use an ensemble of size $M = 2$ . We repeat all experiments with three seeds (ten for Push-T) and report the mean and standard deviation across seeds.

Iterative Fine-Tuning. We investigate uncertainty calibration and active learning using the iterative fine-tuning described in Section 5. For Push-T, we train an ensemble on 50 out of the 206 available demonstrations. For LIBERO, $\mathcal { D } _ { \mathrm { p r e } }$ contains all demonstrations from the 30 tasks in the LIBERO-Goal, -Spatial, and -Object suites, and we also include three tasks from LIBERO-Long to instill different levels of initial task competence. The base ensemble used in the real-world active fine-tuning experiments is trained on 2,500 DROID episodes (Khazatsky et al., 2024). We perform R = 15 (Push-T: 30) rounds of iterative active fine-tuning and query $n _ { \mathrm { e } } = 5$ (real world: 10) expert demonstrations per round, using a replay ratio of $\lambda = 0 . 5 ( \mathrm { P u s h { - } T { : } 0 . 3 ) }$

## 6.2 CALIBRATION (Q1): HOW WELL DOES VFD MEASURE EPISTEMIC UNCERTAINTY?

Setup. We measure calibration using Spearman’s rank correlation coefficient between the per-task uncertainties (8) and success rates, with a value of 1 indicating that the success rate decreases monotonically with increasing uncertainty. To avoid bias from data acquisition, we select new demonstrations uniformly at random each round. We consider six UQ baselines. Action-L2 computes the pairwise $L _ { 2 }$ distance between actions generated independently from the ensemble members. ACE (Römer et al., 2025) computes the conditional entropy in the action chunk distribution. DECU (Berry et al., 2024) performs pairwise distance estimation at a “branching” ODE timestep for denoising diffusion models, and we adapt it for FM. GU (Jazbec et al., 2025) generates samples from all ensemble members starting from the same initial noise to approximate the posterior predictive as a Gaussian, and calculates its entropy. Entropy (Malinin & Gales, 2021) and Perplexity (Fadeeva et al., 2024) compute average per-token entropy and the exponentiated negative mean of the language tokens’ log-probabilities, respectively, and they are only directly applicable to SmolVLA.

Table 1: Calibration analysis. Negative Spearman’s rank correlation ( ) between uncertainty estimates and per-task success rates in LIBERO-Long, averaged across iterative fine-tuning rounds.

<table><tr><td>Policy</td><td>Action-L2</td><td>ACE</td><td>DECU</td><td>GU</td><td>Entropy</td><td>Perplexity</td><td>VFD (ours)</td></tr><tr><td>SmolVLA</td><td> $0.50^{\pm 0.13}$ </td><td> $0.31^{\pm 0.12}$ </td><td> $0.31^{\pm 0.13}$ </td><td> $0.62^{\pm 0.00}$ </td><td> $0.10^{\pm 0.12}$ </td><td> $-0.04^{\pm 0.09}$ </td><td> $\mathbf{0.71}^{\pm 0.03}$ </td></tr><tr><td>X-VLA</td><td> $0.33^{\pm 0.24}$ </td><td> $0.14^{\pm 0.05}$ </td><td> $0.06^{\pm 0.04}$ </td><td> $0.48^{\pm 0.06}$ </td><td>N/A</td><td>N/A</td><td> $\mathbf{0.53}^{\pm 0.02}$ </td></tr><tr><td>FastWAM</td><td> $-0.12^{\pm 0.06}$ </td><td> $-0.29^{\pm 0.07}$ </td><td> $0.03^{\pm 0.07}$ </td><td> $0.27^{\pm 0.04}$ </td><td>N/A</td><td>N/A</td><td> $\mathbf{0.44}^{\pm 0.17}$ </td></tr></table>

VFD excels at estimating epistemic uncertainty. The calibration results, averaged over all iterative fine-tuning rounds, are summarized in Table 1. VFD achieves negative Spearman correlations between 0.05 and 0.17 higher than the best-performing baseline, GU. The results indicate that VFD uncertainty of a generalist policy on the initial frames is a strong indicator of how difficult the respective task is for that policy, and we examine the implications for active fine-tuning in Section 6.3.

VFD is sensitive to different input modalities. As generalist policies often struggle with language conditioning, we investigate whether VFD can capture epistemic uncertainty induced solely by prompt variations. We generate five increasingly semantically perturbed versions of each language prompt in LIBERO-Long (P1 to P5), detailed in Table 5, and evaluate the success rate and uncertainty for each, using the final SmolVLA policy $\pi _ { 1 5 }$ after iterative random data acquisition. The calibration curves are shown in Figure 3, normalized for comparability. VFD yields the strongest monotonic trend, indicating that it can measure epistemic uncertainty induced through different input modalities.

![](images/9a5a0383c4c59b1ebcddd501d84c4ff78afc2d277064e6b829ff22e182435744.jpg)  
Figure 3: Calibration w.r.t. the language prompt. VFD is well-calibrated when injecting uncertainty only through this modality.

## 6.3 ACTIVE FINE-TUNING (Q2): HOW WELL CAN VFD SELECT WHICH DATA TO COLLECT?

Setup. We now investigate whether the superior calibration of VFD transfers to improved data efficiency in active multitask fine-tuning with SAVE. As metrics, we measure the area under the learning curve (AULC) and the average success rate over the last three rounds (Last-3 SR). We consider five baselines. Random selects demonstrations uniformly at random, Diversity (Sener & Savarese, 2018) selects the most visually diverse initial observations using a k-center greedy algorithm, and AMF (Bagatella et al., 2025) performs active multi-task fine-tuning by utilizing posterior uncertainty estimates based on last-layer gradient embeddings. In addition, we design two baselines based on SAVE by replacing VFD with GU and Action-L2, which perform second and third in Table 1. We optimize hyperparameters by sweeping over temperature values $\tau \in [ 0 , 2 . 5 ]$ for SAVE and over different noise levels for AMF. Further implementation details are provided in Appendix B.

SAVE with VFD uncertainty guidance learns new tasks faster. Table 2 summarizes the simulation results. SAVE w/ VFD consistently yields the highest AULC and Last-3 SR across all policies and environments. It achieves a +1.9 to +5.1 pp gain in Last-3 SR over the strongest baseline in each setting and a relative increase of 3.6 to 7.6 % in AULC. Among the uncertainty scores evaluated for SAVE, the overall relative ordering is $\mathrm { V F D { > } G U { > } A c t i o n { - } L 2 }$ . This mirrors the calibration results in Table 1, highlighting the importance of well-calibrated uncertainty estimates for guiding active fine tuning. In the real world, we consider the strongest uncertainty-based baseline from simulation, SAVE w/ GU, as well as Random and Diversity. As shown in Figure 4, SAVE w/ VFD consistently achieves success rates that are only (if at all) reached by the other methods after at least 30 additional demonstrations. At 150 demonstrations, SAVE w/ VFD yields an average task success of $4 7 \% ,$ between +8 and +25 pp higher than the baselines. The higher multitask improvements show that VFD can allocate fine-tuning budget for generalist policies more effectively by quantifying epistemic uncertainty.

![](images/0c3428df9ec6666e1f5f38d7f32733eea6364a75330baf2fc4e0a0d6170842f0.jpg)

![](images/353e163634b1b9a603927d690e35aca95f5987cc27a1c68c618f989d8b3cac37.jpg)  
Figure 4: Active fine-tuning in the real world. For fixed demonstration budgets, SAVE w/ VFD improves multitask performance the most.  
Figure 5: Exploration vs. exploitation. SAVE w/ VFD prioritizes tasks with higher uncertainty, trading off diversity for multitask improvement.

Uncertainty guidance benefits task and initial state selection. We isolate the effects of the two uncertainty-guidance mechanisms of SAVE on LIBERO-Long with SmolVLA. Compared to selecting tasks uniformly at random $( \tau = 0 )$ , SAVE w/ VFD increases Last-3 SR by 11.1 pp, and by 3.4 pp compared to selecting random starting configurations for the sampled tasks. Figure 5 visualizes one reason for these performance gains. While pre-trained generalist policies naturally exhibit varying levels of uncertainty across tasks, SAVE w/ VFD allocates demonstration budget to the tasks that require it most, trading off diversity in selected tasks for faster multitask improvement.

## 6.4 FAILURE DETECTION (Q3): CAN VFD BE USED FOR RUNTIME MONITORING?

Setup. We investigate whether high VFD uncertainty during deployment indicates imminent task failure. For the real-world tasks, we create OOD scenarios by adding novel background objects (e.g., pieces of fruit or a pan). We roll out the policy and calculate VFD scores at each action-generation timestep. Following Agia et al. (2024), we calibrate thresholds from 20 (real world: 10) successful rollouts per task, i.e., without failure data, using conformal prediction (Angelopoulos & Bates, 2023). We report balanced accuracy and time-step-wise accuracy (TWA), which rewards correctly detected failures more when an alarm is raised earlier. As baselines, we consider detecting OOD observations by applying random network distillation (RND-OE, Römer et al., 2025) or training normalizing flows (logpZO, Xu et al., 2025) on the policies’ training data, measuring temporal consistency in the action distributions (STAC, Agia et al., 2024), and computing the conditional action-chunk entropy (ACE, Römer et al., 2025). More details are provided in Appendix B.5.

Table 2: Active fine-tuning in simulation. Area under the learning curve (AULC, ) and average success rate across the last three rounds (Last-3 SR, ) for different data acquisition strategies.

<table><tr><td rowspan="3">Benchmark Policy Metric</td><td colspan="2">Push-T</td><td colspan="6">LIBERO-Long</td></tr><tr><td colspan="2">FM-Policy</td><td colspan="2">SmolVLA</td><td colspan="2">X-VLA</td><td colspan="2">FastWAM</td></tr><tr><td>AULC</td><td>Last-3 SR</td><td>AULC</td><td>Last-3 SR</td><td>AULC</td><td>Last-3 SR</td><td>AULC</td><td>Last-3 SR</td></tr><tr><td>Random</td><td> $38.0^{\pm 3.2}$ </td><td> $46.3^{\pm 3.1}$ </td><td> $44.6^{\pm 2.2}$ </td><td> $52.6^{\pm 3.3}$ </td><td> $37.7^{\pm 1.2}$ </td><td> $45.1^{\pm 0.3}$ </td><td> $54.2^{\pm 2.5}$ </td><td> $64.8^{\pm 6.6}$ </td></tr><tr><td>Diversity</td><td> $40.5^{\pm 1.9}$ </td><td> $51.2^{\pm 3.2}$ </td><td> $45.5^{\pm 1.8}$ </td><td> $53.1^{\pm 2.0}$ </td><td> $40.2^{\pm 1.9}$ </td><td> $49.0^{\pm 3.3}$ </td><td> $53.9^{\pm 0.8}$ </td><td> $63.9^{\pm 0.3}$ </td></tr><tr><td>AMF</td><td>N/A</td><td>N/A</td><td> $45.6^{\pm 3.3}$ </td><td> $59.4^{\pm 3.3}$ </td><td> $39.5^{\pm 1.9}$ </td><td> $51.0^{\pm 1.3}$ </td><td> $55.6^{\pm 1.4}$ </td><td> $69.4^{\pm 1.3}$ </td></tr><tr><td>SAVE w/ GU</td><td> $44.7^{\pm 2.9}$ </td><td> $54.3^{\pm 3.3}$ </td><td> $48.6^{\pm 5.0}$ </td><td> $61.0^{\pm 2.9}$ </td><td> $40.7^{\pm 3.5}$ </td><td> $49.2^{\pm 1.8}$ </td><td> $59.0^{\pm 4.4}$ </td><td> $70.6^{\pm 6.1}$ </td></tr><tr><td>SAVE w/ Action-L2</td><td> $39.8^{\pm 3.4}$ </td><td> $45.0^{\pm 2.6}$ </td><td> $47.2^{\pm 5.1}$ </td><td> $55.5^{\pm 6.8}$ </td><td> $39.2^{\pm 0.9}$ </td><td> $50.9^{\pm 2.2}$ </td><td> $57.0^{\pm 3.3}$ </td><td> $71.6^{\pm 5.2}$ </td></tr><tr><td>SAVE w/ VFD</td><td> $46.6^{\pm 2.5}$ </td><td> $59.3^{\pm 3.1}$ </td><td> $52.3^{\pm 2.2}$ </td><td> $66.1^{\pm 2.8}$ </td><td> $42.5^{\pm 1.0}$ </td><td> $52.9^{\pm 2.9}$ </td><td> $61.1^{\pm 3.5}$ </td><td> $74.8^{\pm 3.3}$ </td></tr></table>

Table 3: Failure detection. VFD achieves the highest balanced accuracy (Acc.) and the best overall trade-off between accuracy and early detection, as measured by timestep-wise accuracy (TWA).

<table><tr><td rowspan="2">Method</td><td colspan="2">Push-T</td><td colspan="2">LIBERO-Plus</td><td colspan="2">Real World</td><td colspan="2">Average</td></tr><tr><td>Acc. ↑</td><td>TWA ↑</td><td>Acc. ↑</td><td>TWA ↑</td><td>Acc. ↑</td><td>TWA ↑</td><td>Acc. ↑</td><td>TWA ↑</td></tr><tr><td>STAC</td><td> $0.55^{\pm 0.00}$ </td><td> $0.51^{\pm 0.00}$ </td><td> $0.70^{\pm 0.03}$ </td><td> $0.59^{\pm 0.02}$ </td><td> $0.43^{\pm 0.02}$ </td><td> $0.42^{\pm 0.02}$ </td><td> $0.56^{\pm 0.02}$ </td><td> $0.51^{\pm 0.01}$ </td></tr><tr><td>logpZO</td><td> $0.53^{\pm 0.01}$ </td><td> $0.51^{\pm 0.02}$ </td><td> $0.68^{\pm 0.02}$ </td><td> $0.62^{\pm 0.01}$ </td><td> $0.70^{\pm 0.03}$ </td><td> $\mathbf{0.64}^{\pm 0.03}$ </td><td> $0.64^{\pm 0.02}$ </td><td> $0.59^{\pm 0.02}$ </td></tr><tr><td>RND-OE</td><td> $0.50^{\pm 0.02}$ </td><td> $0.50^{\pm 0.02}$ </td><td> $0.67^{\pm 0.01}$ </td><td> $0.62^{\pm 0.01}$ </td><td> $0.69^{\pm 0.04}$ </td><td> $0.61^{\pm 0.02}$ </td><td> $0.62^{\pm 0.00}$ </td><td> $0.58^{\pm 0.00}$ </td></tr><tr><td>ACE</td><td> $0.55^{\pm 0.06}$ </td><td> $0.51^{\pm 0.03}$ </td><td> $0.75^{\pm 0.02}$ </td><td> $0.59^{\pm 0.02}$ </td><td> $0.54^{\pm 0.01}$ </td><td> $0.50^{\pm 0.00}$ </td><td> $0.61^{\pm 0.02}$ </td><td> $0.54^{\pm 0.02}$ </td></tr><tr><td>VFD</td><td> $\mathbf{0.64}^{\pm 0.05}$ </td><td> $\mathbf{0.56}^{\pm 0.05}$ </td><td> $\mathbf{0.79}^{\pm 0.00}$ </td><td> $\mathbf{0.65}^{\pm 0.00}$ </td><td> $\mathbf{0.73}^{\pm 0.01}$ </td><td> $0.58^{\pm 0.00}$ </td><td> $\mathbf{0.72}^{\pm 0.02}$ </td><td> $\mathbf{0.60}^{\pm 0.02}$ </td></tr></table>

VFD detects failures more accurately. As reported in Table 3, VFD achieves a balanced accuracy of 72 % across all environments, which is 8 pp higher than the strongest baseline, logpZO. While the observation-based OOD detectors logpZO and RND-OE are trained on the same data as the policy ensemble, VFD is much more robust in distinguishing benign OOD situations where the policy succeeds from actual failures, achieving a TNR of 81 % compared to 58 % and 54 %, respectively. The high accuracy of VFD on LIBERO-Plus and in the real world demonstrates its potential for runtime monitoring of capable generalist policies that can handle some, but not all, novel scenarios.

Reducing computational cost. Table 4 reports the effect of reducing the computational effort of VFD for the real-world rollouts. Perturbing the policy’s weights with layer-normalized Gaussian noise instead of training a second model reduces balanced accuracy to 68 %, which is, however, still 14 to 25 pp above the two training-free baselines, STAC and ACE.

Table 4: Computational cost. Speeding up the computation of VFD only slightly reduces failure detection accuracy.

<table><tr><td>Posterior approximation</td><td># Flow times, batch size</td><td>Balanced accuracy</td><td>Inference time</td><td>vs. base policy</td></tr><tr><td>Ensemble (4)</td><td>4, 8</td><td> $0.78^{\pm 0.00}$ </td><td>+251 ms</td><td>2.48×</td></tr><tr><td>Ensemble (2)</td><td>4, 8</td><td> $0.73^{\pm 0.01}$ </td><td>+89 ms</td><td>1.53×</td></tr><tr><td>Ensemble (2)</td><td>1, 8</td><td> $0.72^{\pm 0.01}$ </td><td>+49 ms</td><td>1.29×</td></tr><tr><td>Ensemble (2)</td><td>1, 1</td><td> $0.70^{\pm 0.02}$ </td><td>+32 ms</td><td>1.19×</td></tr><tr><td>Weight perturb.</td><td>1, 1</td><td> $0.68^{\pm 0.01}$ </td><td>+29 ms</td><td>1.17×</td></tr></table>

## 7 CONCLUSION

We study the estimation of epistemic uncertainty in flow-based generalist robot policies and derive VFD as a mathematically grounded and computationally tractable method for this purpose. Building on this, we propose SAVE, a framework for multitask active fine-tuning that allocates the demonstra tion budget to tasks and initial observations that most require it. VFD is better calibrated than other UQ methods, enabling more data-efficient multitask adaptation with SAVE, and it can detect policy failures more accurately, particularly in OOD settings. In summary, our results establish VFD as an effective method to make generalist policies more reliable and adaptable.

Limitations. We use the uncertainty for the initial observation as a proxy for task difficulty. Our experiments empirically confirm this correlation, but there may be certain scenarios that “look simple” but are challenging for the policy (e.g., due to complex contact dynamics). Moreover, treating uncertainty as the primary signal for determining which data to collect does not account for informational dependencies across tasks. Developing tractable methods to quantify these dependencies and directly maximizing expected information gain for large-scale policies are interesting avenues for future work.

## ACKNOWLEDGMENTS

This work was supported by the German Federal Ministry of Research, Technology and Space (BMFTR) under the Robotics Institute Germany (RIG) funded by BMFTR grant 16ME0997K, the Humboldt Professorship for Robotics and Artificial Intelligence and the Swiss National Science Foundation under NCCR Automation, grant agreement 51NF40 180545. Ralf Römer is supported by the German Research Foundation (DFG) within the RTG project ConVeY funded by grant GRK 2428. Marco Bagatella is supported by the Max Planck ETH Center for Learning Systems. Daniel Marta is supported by the Knut and Alice Wallenberg Foundation.

## REFERENCES

Christopher Agia, Rohan Sinha, Jingyun Yang, Ziang Cao, Rika Antonova, Marco Pavone, and Jeannette Bohg. Unpacking failure modes of generative policies: Runtime monitoring of consistency and progress. In Conference on Robot Learning (CoRL), 2024.

Christopher Agia, Rohan Sinha, Jingyun Yang, Rika Antonova, Marco Pavone, Haruki Nishimura, Masha Itkina, and Jeannette Bohg. CUPID: Curating data your robot loves with influence functions. Conference on Robot Learning (CoRL), 2025.

AgiBot-World et al. AgiBot World Colosseo: A large-scale manipulation platform for scalable and intelligent embodied systems. In International Conference on Intelligent Robots and Systems (IROS), 2025.

Anastasios N Angelopoulos and Stephen Bates. Conformal prediction: A gentle introduction. Foundations and Trends in Machine Learning, 16(4):494–591, 2023.

Jordan T. Ash, Chicheng Zhang, Akshay Krishnamurthy, John Langford, and Alekh Agarwal. Deep batch active learning by diverse, uncertain gradient lower bounds. In International Conference on Learning Representations (ICLR), 2020.

Muhammad Awais, Muzammal Naseer, Salman Khan, Rao Muhammad Anwer, Hisham Cholakkal, Mubarak Shah, Ming-Hsuan Yang, and Fahad Shahbaz Khan. Foundation models defining a new era in vision: a survey and outlook. IEEE Transactions on Pattern Analysis and Machine Intelligence, 47(4):2245–2264, 2025.

Murat Seckin Ayhan and Philipp Berens. Test-time data augmentation for estimation of heteroscedastic aleatoric uncertainty in deep neural networks. In Medical Imaging with Deep Learning, 2018.

Marco Bagatella, Jonas Hübotter, Georg Martius, and Andreas Krause. Active fine-tuning of multitask policies. In International Conference on Machine Learning (ICML), 2025.

Lucas Berry, Axel Brando, and David Meger. Shedding light on large generative networks: Estimating epistemic uncertainty in diffusion models. In Conference on Uncertainty in Artificial Intelligence (UAI), 2024.

Kevin Black et al. $\pi _ { 0 . 5 } { : }$ a vision-language-action model with open-world generalization. In Conference on Robot Learning (CoRL), 2025.

Yuri Burda, Harrison Edwards, Amos Storkey, and Oleg Klimov. Exploration by random network distillation. In International Conference on Learning Representations (ICLR), 2019.

Remi Cadene, Simon Alibert, Francesco Capuano, Michel Aractingi, Adil Zouitine, Pepijn Kooijmans, Jade Choghari, Martino Russi, Caroline Pascal, Steven Palma, et al. LeRobot: An open-source library for end-to-end robot learning. In International Conference on Learning Representations (ICLR), 2026.

Kathryn Chaloner and Isabella Verdinelli. Bayesian experimental design: A review. Statistical Science, 10:273–304, 1995.

Matthew Chan, Maria Molina, and Chris Metzler. Estimating epistemic and aleatoric uncertainty with a single model. Advances in Neural Information Processing Systems (NeurIPS), 2024.

Cheng Chi, Siyuan Feng, Yilun Du, Zhenjia Xu, Eric Cousineau, Benjamin Burchfiel, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. Robotics: Science and Systems (RSS), 2023.

Yuchen Cui, David Isele, Scott Niekum, and Kikuo Fujimura. Uncertainty-aware data aggregation for deep imitation learning. In IEEE International Conference on Robotics and Automation (ICRA), 2019.

Erik Daxberger, Agustinus Kristiadi, Alexander Immer, Runa Eschenhagen, Matthias Bauer, and Philipp Hennig. Laplace redux - effortless Bayesian deep learning. Advances in Neural Information Processing Systems (NeurIPS), 2021.

Alexey Dosovitskiy et al. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations (ICLR), 2021.

Maximilian Du, Zhanyi Sun, Chen Xu, Paarth Shah, Masha Itkina, and Shuran Song. Memory anchors for continual robot learning. arXiv preprint arXiv:2608.26545, 2026.

Yuqing Du, Ksenia Konyushkova, Misha Denil, Akhil Raju, Jessica Landon, Felix Hill, Nando De Freitas, and Serkan Cabi. Vision-language models as success detectors. In Conference on Lifelong Learning Agents (CoLLAs), pp. 120–136, 2023.

Jiafei Duan, Wilbert Pumacay, Nishanth Kumar, Yi Ru Wang, Shulin Tian, Wentao Yuan, Ranjay Krishna, Dieter Fox, Ajay Mandlekar, and Yijie Guo. AHA: A vision-language-model for detecting and reasoning over failures in robotic manipulation. In International Conference on Learning Representations (ICLR), 2025.

Bradley Efron. Tweedie’s formula and selection bias. Journal ofthe American Statistical Association, 106:1602–1614, 2011.

Ekaterina Fadeeva et al. Fact-checking the output of large language models via token-level uncertainty quantification. In Findings ofthe Associationfor Computational Linguistics: ACL 2024, pp. 9367– 9385, 2024.

Haoquan Fang, Jiafei Duan, Donovan Clay, Sam Wang, Shuo Liu, Weikai Huang, Xiang Fan, Wei-Chuan Tsai, Shirui Chen, Yi Ru Wang, et al. MolmoAct2: Action reasoning models for real-world deployment. arXiv preprint arXiv:2605.02881, 2026.

Senyu Fei, Siyin Wang, Junhao Shi, Zihao Dai, Jikun Cai, Pengfang Qian, Li Ji, Xinzhe He, Shiduo Zhang, Zhaoye Fei, et al. LIBERO-Plus: In-depth robustness analysis of vision-language-action models. In Conference on Computer Vision and Pattern Recognition (CVPR), 2026.

Gianni Franchi, Nacim Belkhir, Dat Nguyen Trong, Guoxuan Xia, and Andrea Pilzer. Towards understanding and quantifying uncertainty for text-to-image generation. In Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

Yarin Gal and Zoubin Ghahramani. Dropout as a Bayesian approximation: Representing model uncertainty in deep learning. In International Conference on Machine Learning (ICML), 2016.

Yarin Gal, Riashat Islam, and Zoubin Ghahramani. Deep Bayesian active learning with image data. In International Conference on Machine Learning (ICML), 2017.

Qiao Gu, Yuanliang Ju, Shengxiang Sun, Igor Gilitschenski, Haruki Nishimura, Masha Itkina, and Florian Shkurti. SAFE: Multitask failure detection for vision-language-action models. In Advances in Neural Information Processing Systems (NeurIPS), 2025.

Zhanpeng He, Yifeng Cao, and Matei Ciocarlie. Uncertainty comes for free: Human-in-the-loop policies with diffusion models. arXiv preprint arXiv:2503.01876, 2025.

Joey Hejna, Suvir Mirchandani, Ashwin Balakrishna, Annie Xie, Ayzaan Wahid, Jonathan Tompson, Pannag Sanketi, Dhruv Shah, Coline Devin, and Dorsa Sadigh. Robot data curation with mutual information estimators. Robotics: Science and Systems (RSS), 2025.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems (NeurIPS), 33:6840–6851, 2020.

David Holzmüller, Viktor Zaverkin, Johannes Kästner, and Ingo Steinwart. A framework and benchmark for deep batch active learning for regression. Journal of Machine Learning Research (JMLR), 24(164):1–81, 2023.

Eyke Hüllermeier and Willem Waegeman. Aleatoric and epistemic uncertainty in machine learning: An introduction to concepts and methods. Machine Learning, 110:457–506, 2021.

Metod Jazbec, Eliot Wong-Toi, Guoxuan Xia, Dan Zhang, Eric Nalisnick, and Stephan Mandt. Generative uncertainty in diffusion models. In Conference on Uncertainty in Artificial Intelligence (UAI), 2025.

Li Ju, Mayank Nautiyal, Andreas Hellander, Ekta Vats, and Prashant Singh. Epistemic uncertainty quantification for pre-trained VLMs via Riemannian flow matching. arXiv preprint arXiv:2601.21662, 2026.

Kshitij Judah, Alan Fern, and Thomas G Dietterich. Active imitation learning via reduction to IID active learning. In Conference on Uncertainty in Artificial Intelligence (UAI), 2012.

Rafal Karczewski, Markus Heinonen, and Vikas Garg. Diffusion models as cartoonists: The curious case of high density regions. In International Conference on Learning Representations (ICLR), 2025.

Ulas Berk Karli, Tetsu Kurumisawa, and Tesca Fitzgerald. Ask before you act: Token-level uncertainty for intervention in vision-language-action models. In Second Workshop on Out-of-Distribution Generalization in Robotics at RSS, 2025.

Alexander Khazatsky et al. DROID: A large-scale in-the-wild robot manipulation dataset. In Robotics: Science and Systems, 2024.

Moo Jin Kim, Yihuai Gao, Tsung-Yi Lin, Yen-Chen Lin, Yunhao Ge, Grace Lam, Percy Liang, Shuran Song, Ming-Yu Liu, Chelsea Finn, and Jinwei Gu. Cosmos Policy: Fine-tuning video models for visuomotor control and planning. In International Conference on Learning Representations (ICLR), 2026.

Moo Jin Kim et al. OpenVLA: An open-source vision-language-action model. In Conference on Robot Learning (CoRL), 2024.

Andreas Kirsch, Joost van Amersfoort, and Yarin Gal. BatchBALD: Efficient and diverse batch acquisition for deep Bayesian active learning. In Advances in Neural Information Processing Systems (NeurIPS), 2019.

Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. Simple and scalable predictive uncertainty estimation using deep ensembles. Advances in Neural Information Processing Systems (NeurIPS), 2017.

Sung-Wook Lee, Xuhui Kang, and Yen-Ling Kuo. Diff-DAgger: Uncertainty estimation with diffusion policy for robotic manipulation. In IEEE International Conference on Robotics and Automation (ICRA), 2025.

Qi Li, Bo Yin, Weiqi Huang, Ruhao Liu, Bojun Zou, Runpeng Yu, Jingwen Ye, Weihao Yu, and Xinchao Wang. Vision-language-action safety: Threats, challenges, evaluations, and mechanisms. arXiv preprint arXiv:2604.23775, 2026.

Chen Ling et al. Uncertainty quantification for in-context learning of large language models. In North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT), 2024.

Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In International Conference on Learning Representations (ICLR), 2023.

Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. LIBERO: Benchmarking knowledge transfer for lifelong robot learning. Advances in Neural Information Processing Systems (NeurIPS), 2023.

Antonio Loquercio, Mattia Segu, and Davide Scaramuzza. A general framework for uncertainty estimation in deep learning. IEEE Robotics and Automation Letters (RA-L), 2020.

Huan Ma, Jingdong Chen, Joey Tianyi Zhou, Guangyu Wang, and Changqing Zhang. Estimating LLM uncertainty with evidence. arXiv preprint arXiv:2502.00290, 2025.

Andrey Malinin and Mark Gales. Uncertainty estimation in autoregressive structured prediction. In International Conference on Learning Representations (ICLR), 2021.

Zhiting Mei, Tenny Yin, Micah Baker, Ola Shorinwa, and Anirudha Majumdar. World models that know when they don’t know: Controllable video generation with calibrated uncertainty. arXiv preprint arXiv:2512.05927, 2025.

Eric Nalisnick, Akihiro Matsukawa, Yee Whye Teh, Dilan Gorur, and Balaji Lakshminarayanan. Do deep generative models know what they don’t know? In International Conference on Learning Representations (ICLR), 2019.

NVIDIA et al. GR00T N1: An open foundation model for generalist humanoid robots. arXiv preprint arXiv:2503.14734, 2025.

Abby O’Neill et al. Open X-embodiment: Robotic learning datasets and RT-X models: Open Xembodiment collaboration. In IEEE International Conference on Robotics and Automation (ICRA), 2024.

Physical Intelligence et al. π<sub>0</sub>: A vision-language-action flow model for general robot control. Robotics: Science and Systems (RSS), 2025.

Allen Z Ren et al. Robots that ask for help: Uncertainty alignment for large language model planners. In Conference on Robot Learning (CoRL), 2023.

Ralf Römer, Adrian Kobras, Luca Worbis, and Angela P Schoellig. Failure prediction at runtime for generative robot policies. Advances in Neural Information Processing Systems (NeurIPS), 2025.

Ralf Römer, Yi Zhang, and Angela P Schoellig. CLARE: Continual learning for vision-languageaction models via autonomous adapter routing and expansion. IEEE Robotics and Automation Letters (RA-L), 2026.

Ozan Sener and Silvio Savarese. Active learning for convolutional neural networks: A core-set approach. In International Conference on Learning Representations (ICLR), 2018.

Burr Settles. Active learning literature survey. Technical report, University of Wisconsin–Madison, 2009.

Ola Shorinwa, Zhiting Mei, Justin Lidard, Allen Z Ren, and Anirudha Majumdar. A survey on uncertainty quantification of large language models: Taxonomy, open research challenges, and future directions. ACM Computing Surveys, 58:1–38, 2025.

Mustafa Shukor et al. SmolVLA: A vision-language-action model for affordable and efficient robotics. arXiv preprint arXiv:2506.01844, 2025.

Chen Xu, Tony Khuong Nguyen, Emma Dixon, Christopher Rodriguez, Patrick Miller, Robert Lee, Paarth Shah, Rares Ambrus, Haruki Nishimura, and Masha Itkina. Can we detect failures without failure data? Uncertainty-aware runtime failure detection for imitation learning policies. In Robotics: Science and Systems (RSS), 2025.

Tianyuan Yuan, Zibin Dong, Yicheng Liu, and Hang Zhao. Fast-WAM: Do world action models need test-time future imagination? arXiv preprint arXiv:2603.16666, 2026.

Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In IEEE/CVF International Conference on Computer Vision (ICCV), pp. 11975–11986, 2023.

Jinliang Zheng, Jianxiong Li, Zhihao Wang, Dongxiu Liu, Xirui Kang, Yuchun Feng, Yinan Zheng, Jiayin Zou, Yilun Chen, Jia Zeng, et al. X-VLA: Soft-prompted transformer as scalable crossembodiment vision-language-action model. In International Conference on Learning Representations (ICLR), 2026.

Brianna Zitkovich et al. RT-2: Vision-language-action models transfer web knowledge to robotic control. In Conference on Robot Learning (CoRL), 2023.

## Appendix

## Table of Contents

A Theoretical Results 17 A.1 Flow Matching Fundamentals . 17 A.2 Proof of Theorem 4.1 . 17   
B Experimental Details 19 B.1 Computational Resources . 19 B.2 Environments . 20 B.3 Calibration and Active Fine-Tuning . 20 B.4 Active Fine-Tuning Experiments . 25 B.5 Failure Detection . 27   
C Additional Results 30 C.1 Toy Example . 30 C.2 Calibration . 30 C.3 Learning Curves of Active Fine-Tuning . 32 C.4 Statistical Significance of Uncertainty-Guided Data Acquisition . 32 C.5 Iterative Fine-Tuning vs. Retraining . 32 C.6 Hyperparameter Selection for Uncertainty-Guided Active Fine-Tuning . 34 C.7 Failure Detection . 34

## A THEORETICAL RESULTS

## A.1 FLOW MATCHING FUNDAMENTALS

Flow matching (FM) (Lipman et al., 2023) formulates generative modeling as learning a timedependent vector field $\pmb { u } _ { s }$ that transports samples from a simple base distribution $p _ { 0 }$ to the data distribution $p _ { 1 } = q$ . We consider a noise distribution $p _ { 0 } ( { \pmb x } _ { 0 } ) = \mathcal { N } ( { \pmb x } _ { 0 } | \mathbf { 0 } , I )$ and a data distribution $\mathbf { \mathscr { x } } _ { 1 } \sim \ v q$ . The OT Gaussian conditional probability path is defined as

$$
p _ {s} (\boldsymbol {x} \mid \boldsymbol {x} _ {1}) = \mathcal {N} (\boldsymbol {x} \mid s \boldsymbol {x} _ {1}, (1 - s) ^ {2} \boldsymbol {I}),\tag{10}
$$

and the corresponding flow that pushes the noise distribution $p _ { 0 }$ to $p _ { s } ( { \pmb x } | { \pmb x } _ { 1 } )$ is given by the affine map

$$
\phi_ {s} (\pmb {x} _ {0}) = (1 - s) \pmb {x} _ {0} + s \pmb {x} _ {1}.\tag{11}
$$

The conditional velocity field ${ \pmb u } _ { s } ( { \pmb x } | { \pmb x } _ { 1 } )$ defining this flow via the ODE $\begin{array} { r } { \frac { \mathrm { d } } { \mathrm { d } s } \phi _ { s } ( { \pmb x } ) = { \pmb u } _ { s } ( \phi _ { s } ( { \pmb x } ) ) } \end{array}$ , and generating $p _ { s } ( { \pmb x } | { \pmb x } _ { 1 } )$ , can be derived as:

$$
\boldsymbol {u} _ {s} (\boldsymbol {x} \mid \boldsymbol {x} _ {1}) = \frac {\mathrm{d}}{\mathrm{d} s} \phi_ {s} (\boldsymbol {x} _ {0}) = \boldsymbol {x} _ {1} - \boldsymbol {x} _ {0}.\tag{12}
$$

Marginalizing over the data distribution $q ( { \pmb x } _ { 1 } )$ yields the marginal probability path

$$
p _ {s} (\boldsymbol {x}) = \int p _ {s} (\boldsymbol {x} \mid \boldsymbol {x} _ {1}) q (\boldsymbol {x} _ {1}) \mathrm{d} \boldsymbol {x} _ {1},\tag{13}
$$

and the marginal velocity field is defined as

$$
\boldsymbol {u} _ {s} (\boldsymbol {x}) = \int \boldsymbol {u} _ {s} (\boldsymbol {x} \mid \boldsymbol {x} _ {1}) \frac {p _ {s} (\boldsymbol {x} \mid \boldsymbol {x} _ {1}) q (\boldsymbol {x} _ {1})}{p _ {s} (\boldsymbol {x})} \mathrm{d} \boldsymbol {x} _ {1}.\tag{14}
$$

Finally, these components must satisfy the continuity equation

$$
\frac {\partial}{\partial s} p _ {s} (\boldsymbol {x}) + \operatorname{div} (p _ {s} (\boldsymbol {x}) \boldsymbol {u} _ {s} (\boldsymbol {x})) = 0,\tag{15}
$$

which ensures probability mass conservation.

## A.2 PROOF OF THEOREM 4.1

To simplify notation, we drop the fixed conditioning input y and the parameterization of the velocity fields as neural networks in the following derivations, considering two terminal distributions $p ^ { 1 } ( { \pmb x } ) = p _ { 1 } ^ { 1 } ( { \pmb x } ) , p _ { - } ^ { 2 } ( { \pmb x } ) = p _ { 1 } ^ { 2 } ( { \pmb x } )$ induced by velocity fields ${ \pmb u } _ { s } ^ { 1 } ( { \pmb x } ) , \tilde { { \pmb u } _ { s } ^ { 2 } } ( { \pmb x } )$ and identical base distributions $p _ { 0 } ^ { 1 } ( { \pmb x } ) = p _ { 0 } ^ { 2 } ( { \pmb x } ) = \mathcal { N } ( { \bf 0 } , { \pmb I } )$ .

We impose the following regularity assumptions. For every $s \in [ 0 , 1 )$ and $m \in \{ 1 , 2 \}$ , we assume that $p _ { s } ^ { m } : \bar { \mathbb { R } } ^ { d }  \mathbb { R }$ and $\pmb { u } _ { s } ^ { m } : \mathbb { R } ^ { \breve { d } }  \mathbb { R } ^ { \breve { d } }$ are continuously differentiable with respect to x. Furthermore, for every $s \in [ 0 , 1 )$ , we assume that there exist constants $C _ { s } , a _ { s } , R _ { s } > 0$ such that, for every $m \in \{ 1 , 2 \}$

$$
p _ {s} ^ {m} (\boldsymbol {x}) \leq C _ {s} \exp \left(- a _ {s} \| \boldsymbol {x} \| _ {2} ^ {2}\right) \quad \text {   for   all   } \| \boldsymbol {x} \| _ {2} \geq R _ {s}.
$$

We further assume that ${ \pmb u } _ { s } ^ { 1 } ( { \pmb x } ) , { \pmb u } _ { s } ^ { 2 } ( { \pmb x } )$ , and log $\left( p _ { s } ^ { 1 } ( { \pmb x } ) / p _ { s } ^ { 2 } ( { \pmb x } ) \right)$ have at most polynomial growth as $\| { \pmb x } \| _ { 2 } \to \infty$

In the proof of Theorem 4.1, we will use the following lemma.

Lemma A.1. Let ${ \pmb u } _ { s } ( { \pmb x } )$ be the marginal velocityfield ofan FM model trained with Gaussian OT conditional probability paths, and let $p _ { s } ( { \pmb x } )$ be the marginal probability path induced by ${ \pmb u } _ { s } ( { \pmb x } )$ . Then, for all $s \in [ 0 , 1 )$ ),

$$
\nabla \log p _ {s} (\pmb {x}) = \frac {s \pmb {u} _ {s} (\pmb {x}) - \pmb {x}}{1 - s}.\tag{16}
$$

Proof. Applying Tweedie’s formula (Efron, 2011) to (10) and (13) yields

$$
\mathbb {E} [ s \boldsymbol {x} _ {1} \mid \boldsymbol {x} _ {s} = \boldsymbol {x} ] = \boldsymbol {x} + (1 - s) ^ {2} \frac {\nabla p _ {s} (\boldsymbol {x})}{p _ {s} (\boldsymbol {x})}\tag{17}
$$

$$
\Leftrightarrow \nabla \log p _ {s} (\boldsymbol {x}) = \frac {s \mathbb {E} [ \boldsymbol {x} _ {1} | \boldsymbol {x} _ {s} = \boldsymbol {x} ] - \boldsymbol {x}}{(1 - s) ^ {2}}.\tag{18}
$$

Plugging the conditional velocity (12) into the marginal velocity (14) gives

$$
\boldsymbol {u} _ {s} (\boldsymbol {x}) = \int \left(\boldsymbol {x} _ {1} - \boldsymbol {x} _ {0}\right) p _ {s} \left(\boldsymbol {x} _ {1} \mid \boldsymbol {x} _ {s} = \boldsymbol {x}\right) \mathrm{d} \boldsymbol {x} _ {1}\tag{19}
$$

$$
= \mathbb {E} [ \pmb {x} _ {1} - \pmb {x} _ {0} | \pmb {x} _ {s} = \pmb {x} ]\tag{20}
$$

$$
= \frac {\mathbb {E} [ \boldsymbol {x} _ {1} \mid \boldsymbol {x} _ {s} = \boldsymbol {x} ] - \boldsymbol {x}}{}
$$

$$
1 - s\tag{21}
$$

$$
\Leftrightarrow \mathbb {E} [ \boldsymbol {x} _ {1} \mid \boldsymbol {x} _ {s} = \boldsymbol {x} ] = \boldsymbol {x} + (1 - s) \boldsymbol {u} _ {s} (\boldsymbol {x}),\tag{22}
$$

where (21) follows from $\begin{array} { r } { \pmb { x } = ( 1 - s ) \pmb { x } _ { 0 } + s \pmb { x } _ { 1 } \Leftrightarrow \pmb { x } _ { 1 } - \pmb { x } _ { 0 } = \frac { \pmb { x } _ { 1 } - \pmb { x } } { 1 - s } } \end{array}$ and taking the conditional expectation. Finally, plugging (22) into (18) yields

$$
\nabla \log p _ {s} (\boldsymbol {x}) = \frac {s \boldsymbol {x} + s (1 - s) \boldsymbol {u} _ {s} (\boldsymbol {x}) - \boldsymbol {x}}{(1 - s) ^ {2}}\tag{23}
$$

$$
= \frac {s \pmb {u} _ {s} (\pmb {x}) - \pmb {x}}{1 - s},\tag{24}
$$

which concludes the proof.

ProofofTheorem 4.1. Recall that we aim to express the KL divergence between two distributions $p ^ { 1 } ( { \pmb x } )$ and $p ^ { 2 } ( { \pmb x } )$ parameterized as FM models in terms of their time-dependent velocity fields ${ \pmb u } _ { s } ^ { 1 } ( { \pmb x } )$ and ${ \pmb u } _ { s } ^ { 2 } ( { \pmb x } )$ . We write

$$
D _ {\mathrm{KL}} (p ^ {1} (\boldsymbol {x}) | | p ^ {2} (\boldsymbol {x})) = D _ {\mathrm{KL}} (p ^ {1} (\boldsymbol {x}) | | p ^ {2} (\boldsymbol {x})) - D _ {\mathrm{KL}} (p _ {0} ^ {1} (\boldsymbol {x}) | | p _ {0} ^ {2} (\boldsymbol {x})) + D _ {\mathrm{KL}} (p _ {0} ^ {1} (\boldsymbol {x}) | | p _ {0} ^ {2} (\boldsymbol {x}))\tag{25}
$$

$$
= \int_ {0} ^ {1} \frac {\partial}{\partial s} D _ {\mathrm{KL}} (p _ {s} ^ {1} (\pmb {x}) | | p _ {s} ^ {2} (\pmb {x})) \mathrm{d} s,\tag{26}
$$

where the last step follows from the fact that $p _ { 0 } ^ { 1 } = p _ { 0 } ^ { 2 }$ since both models use the same source distribution. Applying the product rule, we can reformulate the term in the integral as

$$
\frac {\partial}{\partial s} D _ {\mathrm{KL}} (p _ {s} ^ {1} (\pmb {x}) | | p _ {s} ^ {2} (\pmb {x})) = \frac {\partial}{\partial s} \int p _ {s} ^ {1} (\pmb {x}) \log \frac {p _ {s} ^ {1} (\pmb {x})}{p _ {s} ^ {2} (\pmb {x})} \mathrm{d} \pmb {x}\tag{27}
$$

$$
= \int {\frac {\partial}{\partial s}} p _ {s} ^ {1} (\pmb {x}) \log {\frac {p _ {s} ^ {1} (\pmb {x})}{p _ {s} ^ {2} (\pmb {x})}} \mathrm{d} \pmb {x} + \int p _ {s} ^ {1} (\pmb {x}) {\frac {\partial}{\partial s}} \log {\frac {p _ {s} ^ {1} (\pmb {x})}{p _ {s} ^ {2} (\pmb {x})}} \mathrm{d} \pmb {x}\tag{28}
$$

$$
= \int \frac {\partial}{\partial s} p _ {s} ^ {1} (\boldsymbol {x}) \log \frac {p _ {s} ^ {1} (\boldsymbol {x})}{p _ {s} ^ {2} (\boldsymbol {x})} \mathrm{d} \boldsymbol {x} + \int p _ {s} ^ {1} (\boldsymbol {x}) \left(\frac {\frac {\partial}{\partial s} p _ {s} ^ {1} (\boldsymbol {x})}{p _ {s} ^ {1} (\boldsymbol {x})} - \frac {\frac {\partial}{\partial s} p _ {s} ^ {2} (\boldsymbol {x})}{p _ {s} ^ {2} (\boldsymbol {x})}\right) \mathrm{d} \boldsymbol {x}\tag{29}
$$

$$
= \int \frac {\partial}{\partial s} p _ {s} ^ {1} (\boldsymbol {x}) \log \frac {p _ {s} ^ {1} (\boldsymbol {x})}{p _ {s} ^ {2} (\boldsymbol {x})} \mathrm{d} \boldsymbol {x} + \underbrace {\int \frac {\partial}{\partial s} p _ {s} ^ {1} (\boldsymbol {x}) \mathrm{d} \boldsymbol {x}} _ {= 0} - \int \frac {p _ {s} ^ {1} (\boldsymbol {x})}{p _ {s} ^ {2} (\boldsymbol {x})} \frac {\partial}{\partial s} p _ {s} ^ {2} (\boldsymbol {x}) \mathrm{d} \boldsymbol {x},\tag{30}
$$

dx

where the second term vanishes since $\begin{array} { r } { \int \frac { \partial } { \partial s } p _ { s } ^ { 1 } ( { \pmb x } ) \mathrm { d } { \pmb x } = \frac { \partial } { \partial s } \int p _ { s } ^ { 1 } ( { \pmb x } ) \mathrm { d } { \pmb x } = \frac { \partial } { \partial s } 1 = 0 } \end{array}$ . To reformulate the other terms in (30), we proceed as

$$
\frac {\partial}{\partial s} D _ {\mathrm{KL}} (p _ {s} ^ {1} (\pmb {x}) | | p _ {s} ^ {2} (\pmb {x})) = \int \frac {\partial}{\partial s} p _ {s} ^ {1} (\pmb {x}) \log \frac {p _ {s} ^ {1} (\pmb {x})}{p _ {s} ^ {2} (\pmb {x})} \mathrm{d} \pmb {x} - \int \frac {p _ {s} ^ {1} (\pmb {x})}{p _ {s} ^ {2} (\pmb {x})} \frac {\partial}{\partial s} p _ {s} ^ {2} (\pmb {x}) \mathrm{d} \pmb {x}\tag{31}
$$

$$
= - \int \mathrm{div} \big (p _ {s} ^ {1} (\boldsymbol {x}) \boldsymbol {u} _ {s} ^ {1} (\boldsymbol {x}) \big) \log \frac {p _ {s} ^ {1} (\boldsymbol {x})}{p _ {s} ^ {2} (\boldsymbol {x})} \mathrm{d} \boldsymbol {x} + \int \frac {p _ {s} ^ {1} (\boldsymbol {x})}{p _ {s} ^ {2} (\boldsymbol {x})} \mathrm{div} \big (p _ {s} ^ {2} (\boldsymbol {x}) \boldsymbol {u} _ {s} ^ {2} (\boldsymbol {x}) \big)\tag{32}
$$

$$
= \int p _ {s} ^ {1} (\boldsymbol {x}) \boldsymbol {u} _ {s} ^ {1, \top} (\boldsymbol {x}) \nabla \log \frac {p _ {s} ^ {1} (\boldsymbol {x})}{p _ {s} ^ {2} (\boldsymbol {x})} \mathrm{d} \boldsymbol {x} - \int p _ {s} ^ {2} (\boldsymbol {x}) \boldsymbol {u} _ {s} ^ {2, \top} (\boldsymbol {x}) \nabla \left(\frac {p _ {s} ^ {1} (\boldsymbol {x})}{p _ {s} ^ {2} (\boldsymbol {x})}\right) \mathrm{d} \boldsymbol {x}\tag{dx}
$$

(33)

$$
= \int p _ {s} ^ {1} (\boldsymbol {x}) \boldsymbol {u} _ {s} ^ {1, \top} (\boldsymbol {x}) \nabla \log \frac {p _ {s} ^ {1} (\boldsymbol {x})}{p _ {s} ^ {2} (\boldsymbol {x})} \mathrm{d} \boldsymbol {x} - \int p _ {s} ^ {1} (\boldsymbol {x}) \boldsymbol {u} _ {s} ^ {2, \top} (\boldsymbol {x}) \nabla \log \frac {p _ {s} ^ {1} (\boldsymbol {x})}{p _ {s} ^ {2} (\boldsymbol {x})} \mathrm{d} \boldsymbol {x}\tag{34}
$$

$$
= \int p _ {s} ^ {1} (\pmb {x}) \left(\pmb {u} _ {s} ^ {1} (\pmb {x}) - \pmb {u} _ {s} ^ {2} (\pmb {x})\right) ^ {\top} \nabla \log \frac {p _ {s} ^ {1} (\pmb {x})}{p _ {s} ^ {2} (\pmb {x})} \mathrm{d} \pmb {x}\tag{35}
$$

$$
= \int p _ {s} ^ {1} (\pmb {x}) \left(\pmb {u} _ {s} ^ {1} (\pmb {x}) - \pmb {u} _ {s} ^ {2} (\pmb {x})\right) ^ {\top} \left(\nabla \log p _ {s} ^ {1} (\pmb {x}) - \nabla \log p _ {s} ^ {2} (\pmb {x})\right) \mathrm{d} \pmb {x}.\tag{36}
$$

Here, (32) follows from the continuity equation (15). Equation (33) follows from integration by parts, with the boundary terms vanishing under the regularity assumptions stated above.

Applying Lemma A.1 to the last term in the integral (36) yields

$$
\nabla \log p _ {s} ^ {1} (\pmb {x}) - \nabla \log p _ {s} ^ {2} (\pmb {x}) = \frac {s}{1 - s} (\pmb {u} _ {s} ^ {1} (\pmb {x}) - \pmb {u} _ {s} ^ {2} (\pmb {x})).\tag{37}
$$

Defining $\begin{array} { r } { \kappa _ { s } = \frac { s } { 1 - s } } \end{array}$ and plugging (37) into (36) results in

$$
\frac {\partial}{\partial s} D _ {\mathrm{KL}} (p _ {s} ^ {1} (\pmb {x}) | | p _ {s} ^ {2} (\pmb {x})) = \int p _ {s} ^ {1} (\pmb {x}) (\pmb {u} _ {s} ^ {1} (\pmb {x}) - \pmb {u} _ {s} ^ {2} (\pmb {x})) ^ {\top} (\nabla \log p _ {s} ^ {1} (\pmb {x}) - \nabla \log p _ {s} ^ {2} (\pmb {x})) \mathrm{d} \pmb {x}\tag{38}
$$

$$
= \int p _ {s} ^ {1} (\pmb {x}) \kappa_ {s} (\pmb {u} _ {s} ^ {1} (\pmb {x}) - \pmb {u} _ {s} ^ {2} (\pmb {x})) ^ {\top} (\pmb {u} _ {s} ^ {1} (\pmb {x}) - \pmb {u} _ {s} ^ {2} (\pmb {x})) \mathrm{d} \pmb {x}\tag{39}
$$

$$
= \kappa_ {s} \mathbb {E} _ {\pmb {x} \sim p _ {s} ^ {1} (\cdot)} \left[ \| \pmb {u} _ {s} ^ {1} (\pmb {x}) - \pmb {u} _ {s} ^ {2} (\pmb {x}) \| _ {2} ^ {2} \right].\tag{40}
$$

Finally, we can insert (40) into (26) and obtain

$$
D _ {\mathrm{KL}} (p ^ {1} (\boldsymbol {x}) | | p ^ {2} (\boldsymbol {x})) = \int_ {0} ^ {1} \kappa_ {s} \mathbb {E} _ {\boldsymbol {x} \sim p _ {s} ^ {1} (\cdot)} \left[ \| \boldsymbol {u} _ {s} ^ {1} (\boldsymbol {x}) - \boldsymbol {u} _ {s} ^ {2} (\boldsymbol {x}) \| _ {2} ^ {2} \right] \mathrm{d} s,\tag{41}
$$

concluding the proof.

Remark A.2. Although the weighting $\kappa _ { s }$ in (41) diverges as $s \to 1$ , the VFD uncertainty estimate remains finite. In practice, our estimator $( 7 )$ evaluates the velocity difference on a grid $s _ { \ell } = \ell \delta s$ with $\ell \in \{ 0 , \ldots , N _ { s } - \overset { \textstyle \mathbf { \tilde { \rho } } } { 1 } \}$ , so the largest weight is $\begin{array} { r } { \kappa _ { 1 - \delta s } = \frac { 1 - \delta s } { \delta s } } \end{array}$ , which is finite for $\delta s > 0$

## B EXPERIMENTAL DETAILS

## B.1 COMPUTATIONAL RESOURCES

Our codebase builds on LeRobot (Cadene et al., 2026), released under the Apache License, and FIPER (Römer et al., 2025), released under the MIT License. The simulation experiments involving FM-Policy and SmolVLA were run on a compute node with two NVIDIA RTX 4090 GPUs, 8 CPU cores, and 64 GB of RAM. With this hardware setup, one active learning experiment with 15 rounds of uncertainty quantification, episode selection, 4,000 gradient steps, and 30 evaluation rollouts per task takes approximately 12.5 hours. The simulation experiments involving X-VLA and FastWAM were conducted on NVIDIA H100 and H200 GPUs, respectively. The models used in our hardware experiments were trained on an NVIDIA H100 GPU, and we ran real-world inference on an NVIDIA

RTX 5090 GPU. The total computing budget for all experiments, including developing the codebase and running preliminary tests, was about 10,000 GPU hours.

## B.2 ENVIRONMENTS

Push-T. The Push-T benchmark simulates an agent in a 2D environment that must push a Tshaped object into a drawn target configuration. The observation contains the agent’s position and an RGB image of the scene, and the actions are position commands. Despite its low-dimensional action space, Push-T is a relatively challenging environment due to the multimodality of the demonstrations and the high diversity of initial configurations. The latter makes it a suitable benchmark for evaluating the selection of initial states for querying expert demonstrations.

LIBERO. We use the LIBERO simulation benchmark (Liu et al., 2023) (MIT License) in our experiments. The environments include a Franka robotic manipulator with a parallel-jaw gripper in a kitchen environment. For each task, up to 50 human expert demonstrations are available. The observations $\mathbf { } _ { o _ { t } }$ consist of two RGB images with spatial resolution $2 5 6 \times 2 5 6$ , one from a fixed workspace view and one from a camera mounted on the robot’s end-effector. Additionally, observations include the 8-dimensional proprioceptive state of the arm, comprising the end-effector position (3D), the end-effector orientation parameterized as an axis-angle (3D), and the gripper joint positions (2D). Actions $\mathbf { } \mathbf { a } _ { t }$ are continuous and 7-dimensional: 3D delta end-effector translation, 3D delta end-effector orientation, and 1D gripper command.

LIBERO-Plus. To evaluate failure detection in OOD settings in simulation, we use LIBERO-Plus (Fei et al., 2026), which perturbs the LIBERO tasks across seven dimensions: camera viewpoint, robot initial state, language instruction, lighting, background, sensor noise, and object layout, each with several levels of severity. The observation and action spaces are identical to those in LIBERO.

Real-world tasks. We conduct hardware experiments with a Franka FR3 manipulator. The observation contains the language command, RGB images from a wrist-mounted and a third-person camera, the Cartesian end-effector pose and continuous gripper state. The policies generate desired end-effector poses (position and Euler angles) and gripper state as actions at a control frequency of 15 Hz. The pose targets are tracked by a low-level Cartesian torque controller operating at 500 Hz. We define three manipulation tasks involving different objects and motion patterns, each consisting of two to three phases (see Figure 6):

• Place: Grasp and lift a bowl randomly placed on the table (phase 1) and place it on a plate at a fixed location (phase 2). Language command: put the bowl on the plate.

• Stack: Grasp and lift a bowl randomly placed on the table (phase 1) and stack it onto three other bowls located at the center of the table (phase 2). Language command: stack the orange bowl on the other bowls.

• Drawer: Grasp and lift a Lego block pressed into a Lego plate (phase 1), place it into an opened drawer compartment (phase 2), and close the drawer (phase 3). Language command: put the lego into the drawer.

## B.3 CALIBRATION AND ACTIVE FINE-TUNING

## B.3.1 ENSEMBLE TRAINING

In each experiment, we take a single base checkpoint and fine-tune it into an ensemble using a dataset $\mathcal { D } _ { \mathrm { p r e } }$ . For $\mathtt { P u s h - T } , \mathcal { D } _ { \mathrm { p r e } }$ contains 50 episodes drawn uniformly at random from the available 206 demonstrations. In our experiments on LIBERO-Long, $\mathcal { D } _ { \mathrm { p r e } }$ contains all episodes from the $\mathtt { L I B E R O - G o a l } , \mathtt { - S p a t i a l }$ , and -Object suites, each covering 10 tasks. Training on these 30 tasks alone yielded zero initial success rate across the LIBERO-Long tasks we consider for active fine-tuning. Since we aim to test algorithms’ capabilities to allocate demonstration budget to the tasks that need it most, we instill varying levels of initial task competence by also including tasks (1, 2, 3) from LIBERO-Long into $\mathcal { D } _ { \mathrm { p r e } } .$

Additional training details are provided in Table 6.

![](images/6aa97593f860821c440ebeed2e602ff69403f780ebe836e988abfdc4a42eb120.jpg)

![](images/c968c66c8e5b2866d9c5159289e8c8c5987e8cb5ecda8071e406ff8c18b5e04c.jpg)

![](images/851d75bbfdd5f114de56d84231d3e467cb9f885353d2de3cc8d503e5080f49f8.jpg)

![](images/0675d3f0825f7a84171c4ad1761c5a4817f628c90e53ff91752114f08376b941.jpg)  
Place

![](images/3e5e40668f2a63c530d589ff99f01bd40026e41b8b972ffe2bf5410626657073.jpg)  
Stack

![](images/028c620a441047babafa129b8b0190ae07c0969febeaadd9239939ec615d8c05.jpg)  
Drawer  
Figure 6: Real-world tasks. Top: nominal settings used to collect demonstrations and evaluate active fine-tuning. Bottom: OOD settings with novel distractor objects in our failure detection experiments.

## B.3.2 UNCERTAINTY QUANTIFICATION METHODS

The computation of VFD-based epistemic uncertainty estimation is summarized in Algorithm 1 and illustrated in Figure 7. For all policies, including FastWAM, we compute VFD only over the generated action chunks.

In the following, we describe the implementation of the baselines considered in Section 6.2.

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1 Efficient Epistemic Uncertainty Estimation for Flow-Matching Models using VFD.

1: Input: Flow-matching model ensemble $p^{\theta_1}(\boldsymbol{x} \mid \boldsymbol{y})$, $p^{\theta_2}(\boldsymbol{x} \mid \boldsymbol{y})$ with vector fields $\mathcal{V} = (\boldsymbol{v}_s^{\theta_1}, \boldsymbol{v}_s^{\theta_2})$ trained via OT paths, conditioning input $\boldsymbol{y}$, ODE integration step size $\delta s = 1/N_s$.

2: Output: Estimate of the epistemic uncertainty $u_e(\boldsymbol{y}; \mathcal{V})$ at the conditioning input $\boldsymbol{y}$.

3: Hyperparameters: Batch size $B$.

4: Sample $\boldsymbol{x}_0^{(1,1:B)} \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$, $\boldsymbol{x}_0^{(2,1:B)} \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$.

5: Initialize $u_e \leftarrow 0$.

6: for $\ell = 0$ to $N_s - 1$ do ▷ ODE integration.

7: Set $s = \ell \delta s$.

8: for each $i \in \{1, 2\}$ do

9: Compute the expected VFDs $d_{\ell,i}^2 = \frac{1}{B} \sum_{b=1}^{B} \| \boldsymbol{v}_s^{\theta_1}(\boldsymbol{x}_s^{(i,b)}, \boldsymbol{y}) - \boldsymbol{v}_s^{\theta_2}(\boldsymbol{x}_s^{(i,b)}, \boldsymbol{y}) \|_2^2$.

10: Update uncertainty estimate $u_e \leftarrow u_e + \frac{1}{2N_s} \frac{s}{1-s} d_{\ell,i}^2$.

11: Forward integrate $\boldsymbol{x}_{s+\delta s}^{(i,1:B)} = \boldsymbol{x}_s^{(i,1:B)} + \boldsymbol{v}_s^{\theta_i}(\boldsymbol{x}_s^{(i,1:B)}, \boldsymbol{y}) \delta s$.

12: return $u_e(\boldsymbol{y}; \mathcal{V}) = u_e$.
</div>

o: Low uncertainty

![](images/fdc56722cbe18ad6b52c6d4ccdcd183a181781c00c7546e08733acd332969d7d.jpg)  
Figure 7: VFD visualization. We illustrate the computation of VFD for two conditioning inputs with different levels of epistemic uncertainty: low (left) and high (right). Starting from the same initial samples ${ \bf { a } } _ { 0 } ,$ we integrate the ODE of the first ensemble member (blue) and evaluate the velocity field of the second member at the same intermediate states $\mathbf { \delta } _ { a _ { s } }$ (orange). VFD accumulates the weighted differences between the two velocities (green).

Action-L2 implementation. Action-L2 estimates epistemic uncertainty from the disagreement in terminal action distributions. Given an observation o, we draw independent Gaussian noise samples from $p _ { 0 }$ and generate C terminal action chunks from the policy and from each ensemble member. Let

$$
\{\pmb {A} _ {i} ^ {\mathrm{s}} (\pmb {o}) \} _ {i = 1} ^ {C}
$$

denote the action chunks generated by the policy, and let

$$
\{\boldsymbol {A} _ {j} ^ {(m)} (\boldsymbol {o}) \} _ {j = 1} ^ {C}
$$

denote the action chunks generated by ensemble member $m \in \{ 1 , \ldots , M \}$

The discrepancy between the policy and ensemble member m is computed as the mean pairwise Euclidean distance between their sampled terminal action chunks:

$$
d _ {m} (\boldsymbol {o}) = \frac {1}{C ^ {2}} \sum_ {i = 1} ^ {C} \sum_ {j = 1} ^ {C} \left\| \boldsymbol {A} _ {i} ^ {\mathrm{s}} (\boldsymbol {o}) - \boldsymbol {A} _ {j} ^ {(m)} (\boldsymbol {o}) \right\| _ {2}.
$$

The Action-L2 epistemic uncertainty estimate is then the average discrepancy across ensemble members:

$$
u _ {\mathrm{Action-L2}} (\boldsymbol {o}) = \frac {1}{M} \sum_ {m = 1} ^ {M} d _ {m} (\boldsymbol {o}).
$$

ACE implementation. ACE (Römer et al., 2025) estimates predictive uncertainty from the dispersion of action samples generated by a single policy. Given an observation o, we draw C independent Gaussian noise samples and generate C action chunks via FM ODE integration. For each sampled chunk $c \in \{ 1 , \ldots , \bar { C } \}$ , we extract the first three (Push-T: two) action dimensions at each timestep t, denoted by $\{ p _ { t } ^ { ( c ) } \} _ { c = 1 } ^ { C }$ . At each timestep $t \in \{ 1 , \ldots , H \}$ in the prediction horizon H, the sampled positions $\{ p _ { t } ^ { ( c ) } \} _ { c = 1 } ^ { C }$ are discretized into a 3D grid. The grid cell size along each coordinate axis is set to 0.03 times the empirical range of the samples along that axis. In our experiments, this performed better than using fixed cell sizes computed offline from calibration data, as proposed by the authors. Let $B _ { t }$ denote a set of bins over the occupied grid at time t and let $P _ { t } ( b )$ be the empirical probability of a sample falling into bin $b \in B _ { t }$ at time t. The ACE uncertainty estimate is the Shannon entropy of these bin counts, averaged over the prediction horizon:

$$
u _ {\mathrm{ACE}} (\pmb {o}) = - \frac {1}{H} \sum_ {t = 1} ^ {H} \sum_ {b \in \mathcal {B} _ {t}} P _ {t} (b) \log_ {2} P _ {t} (b).
$$

DECU implementation. DECU (Berry et al., 2024) estimates epistemic uncertainty in diffusion models from the disagreement among ensemble members’ denoising means at an intermediate point of the generative process. Given the close relationship between diffusion and FM, we adapt DECU to FM by replacing the denoising mean with the learned velocity field. Given an observation o, we sample C independent Gaussian noise vectors $\pmb { x } _ { 0 } ^ { ( c ) } \sim p _ { 0 }$ for $c \in \{ 1 , \ldots , C \}$ . Each ensemble member integrates the FM ODE from $s = 0$ to a branching time $s _ { b } = 0 . 9 9 5$ (as in Berry et al. (2024)) to obtain intermediate states $\pmb { x } _ { s _ { b } } ^ { ( c ) } . \mathrm { A t } s _ { b }$ , we evaluate the velocity fields of all M ensemble members, ${ \pmb v } _ { i } ^ { ( c ) } = { \pmb v } _ { s _ { b } } ^ { { \pmb \theta } _ { i } } ( { \pmb x } _ { s _ { b } } ^ { ( c ) } , { \pmb o } )$ for $i \in \{ 1 , \ldots , M \}$

Disagreement is measured using the Pairwise-Distance Estimator (PaiDE). For each sampled chunk $c ,$ define the squared pairwise velocity-field distances

$$
D _ {i j} ^ {(c)} = \left\| \boldsymbol {v} _ {i} ^ {(c)} - \boldsymbol {v} _ {j} ^ {(c)} \right\| _ {2} ^ {2}, \quad i, j \in \{1, \dots , M \}.
$$

The score for sample c is then

$$
d _ {\mathrm{DECU}} ^ {(c)} (\boldsymbol {o}) = - \frac {1}{M} \sum_ {i = 1} ^ {M} \log \left(\frac {1}{M} \sum_ {j = 1} ^ {M} \exp \left(- D _ {i j} ^ {(c)}\right)\right).
$$

The DECU epistemic uncertainty estimate for observation o is the mean score over the C sampled chunks:

$$
u _ {\mathrm{DECU}} (\pmb {o}) = \frac {1}{C} \sum_ {c = 1} ^ {C} d _ {\mathrm{DECU}} ^ {(c)} (\pmb {o}).
$$

GU implementation. As described in the main text, GU (Jazbec et al., 2025) approximates the posterior predictive action distribution as a Gaussian. Using our deep ensemble, we compute a raw uncertainty score for each initial observation $\mathbf { \xi } _ { o _ { k l } }$ , where k indexes the task and l indexes the observation within that task. For a fixed initial noise sample, let $A ^ { ( i ) } ( \pmb { o } _ { k l } ) \in \mathbb { R } ^ { H \times D _ { a } }$ denote the action trajectory generated by ensemble member $i \in \{ 1 , \ldots , M \}$ , and let $a _ { k l , t , d } ^ { ( i ) }$ denote its action value at timestep $t \in \{ 1 , \ldots , H \}$ and action dimension d $\in \{ 1 , \ldots , D _ { a } \}$

The raw GU uncertainty score is obtained by summing the log-variance of the ensemble predictions across all timesteps and action dimensions:

$$
u _ {k l} ^ {\mathrm{raw}} = \sum_ {t = 1} ^ {H} \sum_ {d = 1} ^ {D _ {a}} \log \left(\operatorname{Var} _ {i = 1} ^ {M} \left[ a _ {k l, t, d} ^ {(i)} \right] + \epsilon_ {\mathrm{var}}\right),
$$

where $\epsilon _ { \mathrm { v a r } } = 1 0 ^ { - 8 }$ is added for numerical stability.

Because the differential entropy of a continuous distribution can be negative, these raw scores may also be negative. To obtain non-negative uncertainty weights, we first aggregate the observation-level scores into a raw task-level uncertainty score $U _ { k } ^ { \mathrm { r a w } }$ and then shift the task-level scores by

$$
U _ {k} = U _ {k} ^ {\text {raw}} - \min _ {k ^ {\prime} \in \{1, \dots , K \}} U _ {k ^ {\prime}} ^ {\text {raw}} + \epsilon_ {\text {off}} \left(\max _ {k ^ {\prime} \in \{1, \dots , K \}} U _ {k ^ {\prime}} ^ {\text {raw}} - \min _ {k ^ {\prime} \in \{1, \dots , K \}} U _ {k ^ {\prime}} ^ {\text {raw}}\right).
$$

Here, $\epsilon _ { \mathrm { o f f } } > 0$ is a small offset coefficient. This shift removes negative values, and when the task scores are not all identical, the range-dependent offset moves the minimum-uncertainty task away from zero so that it retains non-zero sampling probability after normalization.

Entropy implementation. Entropy (Malinin & Gales, 2021) utilizes the extent to which the VLM (SmolVLA backbone) successfully predicts the next token at each valid prefix position. Given an observation o, the policy constructs a prefix sequence of length $S _ { \mathrm { p r e f i x } }$ containing images, robot states and language. The VLM predicts the next-token probability distribution $\psi _ { m }$ at each valid prefix position $m \in \{ 1 , \ldots , S _ { \mathrm { p r e f i x } } \}$ , with a single forward pass. The uncertainty score is calculated by the Shannon entropy of this distribution, averaged over all valid prefix tokens:

$$
u _ {\text {Entropy}} = - \frac {1}{S _ {\text {prefix}}} \sum_ {m = 1} ^ {S _ {\text {prefix}}} \sum_ {z \in \mathcal {Z}} \psi_ {m} (z) \log \psi_ {m} (z),
$$

where $\mathcal { Z }$ is the VLM vocabulary and $\psi _ { m } ( z )$ is the predicted probability of token z at position m.

Perplexity implementation. Perplexity (Fadeeva et al., 2024) estimates uncertainty by measuring the likelihood of the language instruction under the VLM backbone. Similar to Entropy, it evaluates the prefix in a single forward pass without action generation. We extract the predicted next-token log-probabilities for the $S _ { \mathrm { l a n g } }$ language tokens within their respective prefix. Let $z _ { m }$ be the true language token at sequence position $m .$ . The uncertainty score u is the exponentiated negative mean log-probability of these tokens:

$$
u _ {\text { Perp }} = \exp \left(- \frac {1}{S _ {\text { lang }}} \sum_ {m = 1} ^ {S _ {\text { lang }}} \log \psi_ {m} (z _ {m})\right).
$$

Diversity-Based Greedy. The diversity-based greedy baseline (Sener & Savarese, 2018) selects initial observations to maximize visual diversity. Let $\mathcal { O } _ { k }$ denote the pool of candidate initial observa tions, and $\hat { \mathcal { O } }$ denote the set of selected observations. We extract visual features for each $\mathbf { { \sigma } } _ { o _ { i } } \in \mathcal { O } _ { k }$ using a SigLIP (Zhai et al., 2023) vision encoder. We rank all candidates using the k-Center Greedy algorithm. At each step, we select $o ^ { * }$ that maximizes the minimum distance $\Delta$ to $\hat { \mathcal { O } }$ :

$$
\boldsymbol {o} ^ {*} = \arg \max _ {\boldsymbol {o} _ {i} \in \mathcal {O} _ {k} \setminus \hat {\mathcal {O}}} \min _ {\boldsymbol {o} _ {j} \in \hat {\mathcal {O}}} \Delta (\boldsymbol {o} _ {i}, \boldsymbol {o} _ {j}),
$$

where $\Delta ( o _ { i } , o _ { j } )$ is the Euclidean distance between visual features. We update $\hat { \mathcal { O } } \gets \hat { \mathcal { O } } \cup \{ o ^ { * } \}$ and repeat until all observations are ranked. In each active fine-tuning round, we select the top $n _ { \mathrm { e } }$ observations from this ranking without replacement.

## B.3.3 CALIBRATION EXPERIMENTS

We primarily measure calibration by comparing the uncertainty estimate for an initial observation with the probability of success in rolling out the policy from that observation. More specifically, we calculate the mean uncertainty (8) per task and evaluate the per-task success rate by rolling out the first ensemble member $\pi _ { r } ^ { ( 1 ) }$ multiple times for different initial states of the robot and the objects in the scene. To assess whether higher uncertainty correlates with a lower success rate, we calculate the Spearman rank correlation coefficient $\rho$ between the two quantities. If and only if the success rate decreases strictly monotonically with higher uncertainty, $\rho = - 1$ . As a secondary metric, we also report the Pearson correlation coefficient, which measures the strength of the linear relationship between the two quantities.

In Figure 3, we investigate whether the uncertainty scores are well-calibrated when a decrease in success rate stems solely from varying the language command. To this end, we use different language prompts for each task, which are listed in Table 5.

Table 5: Prompt variations in Figure 3. We list the prompts used to evaluate calibration of the uncertainty estimators with respect to variations in the language input in LIBERO-Long. Changes from the original prompt are marked in bold.

<table><tr><td>Task</td><td>P1 (original)</td><td>P2</td><td>P3</td><td>P4</td><td>P5</td></tr><tr><td>1</td><td>put both the alpha-bet soup and the tomato sauce in the basket</td><td>put the alphabet soup and the tomato sauce into the basket</td><td>place both the al-phabet soup and the tomato sauce in the basket</td><td>move both the al-phabet soup and tomato sauce to the basket</td><td>transfer the soup cans to the basket</td></tr><tr><td>2</td><td>put both the cream cheese box and the butter in the basket</td><td>place both the cream cheese box and the butter in the basket</td><td>put the cream cheese box and the butter into the basket</td><td>move the cream cheese and the but-ter into the basket</td><td>place the dairy products in the basket</td></tr><tr><td>3</td><td>turn on the stove and put the moka pot on it</td><td>switch on the stove and place the moka pot on it</td><td>start the stove and set the moka pot on the burner</td><td>activate the stove and put the moka pot on top of it</td><td>power on the cook-top and position the coffee maker on it</td></tr><tr><td>4</td><td>put the black bowl in the bottom drawer of the cabinet and close it</td><td>put the black bowl into the lower drawer of the cabinet and shut it</td><td>place the black bowl in the bottom cabinet drawer and close it</td><td>store the black bowl in the bottom cabinet drawer and close the drawer</td><td>move the dark bowl to the lowest cabinet compartment and shut it</td></tr><tr><td>5</td><td>put the white mug on the left plate and put the yellow and white mug on the right plate</td><td>set the white mug on the left plate and the yellow and white mug on the right plate</td><td>place the white mug on the left plate and place the yellow and white mug on the right plate</td><td>put the white cup on the left dish and the yellow-white cup on the right dish</td><td>arrange the mugs: white one on the left plate, yellow and white one on the right plate</td></tr><tr><td>6</td><td>pick up the book and place it in the back compartment of the caddy</td><td>grab the book and place it in the back compartment of the caddy</td><td>pick up the book and put it in the rear compartment of the caddy</td><td>take the book and place it in the back section of the caddy</td><td>move the book to the back compartment of the organizer</td></tr><tr><td>7</td><td>put the white mug on the plate and put the chocolate pudding to the right of the plate</td><td>place the white mug on the plate and put the chocolate pudding to the right of the plate</td><td>put the white cup on the dish and move the chocolate pudding to the right side of the plate</td><td>set the white mug on the plate and place the chocolate pudding to the right of the plate</td><td>position the mug on the plate and set the pudding to the right of it</td></tr><tr><td>8</td><td>place both the al-phabet soup and the cream cheese box in the basket</td><td>put both the alpha-bet soup and the cream cheese box in the basket</td><td>put the alphabet soup and the cream cheese box into the basket</td><td>move both the al-phabet soup and cream cheese box to the basket</td><td>transfer the soup and cheese box to the basket</td></tr><tr><td>9</td><td>put both moka pots on the stove</td><td>place both moka pots on the stove</td><td>put the two moka pots on the stove</td><td>set both coffee pots on the stove</td><td>move the two moka pots onto the cooktop</td></tr><tr><td>10</td><td>put the yellow and white mug in the microwave and close it</td><td>place the yellow and white mug in the microwave and close it</td><td>put the yellow and white mug into the microwave and shut it</td><td>move the yellow and white mug into the microwave and close the door</td><td>place the yellow-white mug in the microwave and shut the door</td></tr></table>

## B.4 ACTIVE FINE-TUNING EXPERIMENTS

We provide implementation details about our simulated and real-world active fine-tuning experiments in Table 6 and the full algorithm in Algorithm 2. In our experiments, for a fair comparison, all dataacquisition strategies retrieve demonstrations from the same expert dataset. An initial configuration can only be selected once per fine-tuning round, but in multiple different rounds.

Table 6: Implementation details for the active fine-tuning experiments.

<table><tr><td>Environment</td><td>Push-T</td><td colspan="3">LIBERO-Long</td><td>Real world</td></tr><tr><td>Policy</td><td>FM-Policy</td><td>SmolVLA</td><td>X-VLA</td><td>FastWAM</td><td>SmolVLA</td></tr><tr><td colspan="6">Ensemble training</td></tr><tr><td>Demonstrations in  $\mathcal{D}_{\text{pre}}$ </td><td>50</td><td>1,437</td><td>1,437</td><td>1,437</td><td>2,549</td></tr><tr><td>Training steps</td><td>30,000</td><td>30,000</td><td>30,000</td><td>30,000</td><td>30,000</td></tr><tr><td>Batch size</td><td>128</td><td>32</td><td>32</td><td>32</td><td>24</td></tr><tr><td>Learning rate</td><td> $1 \times 10^{-4}$ </td><td> $1 \times 10^{-4}$ </td><td> $1 \times 10^{-4}$ </td><td> $1 \times 10^{-4}$ </td><td> $1 \times 10^{-4}$ </td></tr><tr><td>Learning rate schedule</td><td>cosine</td><td>cosine</td><td>cosine</td><td>cosine</td><td>cosine</td></tr><tr><td>Final learning rate</td><td>0</td><td> $2.5 \times 10^{-6}$ </td><td> $2.5 \times 10^{-6}$ </td><td> $2.5 \times 10^{-6}$ </td><td> $2.5 \times 10^{-6}$ </td></tr><tr><td>Warmup steps</td><td>1,000</td><td>1,000</td><td>1,000</td><td>1,000</td><td>1,000</td></tr><tr><td colspan="6">Iterative fine-tuning</td></tr><tr><td>Demonstrations per round</td><td>5</td><td>5</td><td>5</td><td>5</td><td>10</td></tr><tr><td>Rounds R</td><td>30</td><td>15</td><td>15</td><td>15</td><td>15</td></tr><tr><td>Training steps per round</td><td>2,000</td><td>4,000</td><td>4,000</td><td>4,000</td><td>10,000</td></tr><tr><td>Batch size</td><td>128</td><td>32</td><td>64</td><td>64</td><td>32</td></tr><tr><td>Learning rate</td><td> $1 \times 10^{-4}$ </td><td> $5 \times 10^{-5}$ </td><td> $1 \times 10^{-4}$ </td><td> $1 \times 10^{-4}$ </td><td> $5 \times 10^{-5}$ </td></tr><tr><td>Learning rate schedule</td><td>cosine</td><td>cosine</td><td>cosine</td><td>cosine</td><td>cosine</td></tr><tr><td>Final learning rate</td><td>0</td><td> $5 \times 10^{-6}$ </td><td> $5 \times 10^{-6}$ </td><td> $5 \times 10^{-6}$ </td><td> $5 \times 10^{-6}$ </td></tr><tr><td>Warmup steps</td><td>100</td><td>200</td><td>200</td><td>200</td><td>500</td></tr><tr><td>Replay ratio λ</td><td>0.3</td><td>0.5</td><td>0.5</td><td>0.5</td><td>0.5</td></tr><tr><td colspan="6">Evaluation</td></tr><tr><td>Evaluated rounds</td><td>all</td><td>all</td><td>all</td><td>all</td><td>6, 9, 12, 15</td></tr><tr><td>Rollouts per round</td><td>200</td><td>300</td><td>300</td><td>300</td><td>30</td></tr></table>

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 2 Uncertainty-Based Active Fine-Tuning of Flow-Based Generalist Policies with SAVE.

1: Input: Base policy $\pi_{\mathrm{b}}$, pool of initial configurations $\{\mathcal{O}_k\}_{k=1}^K$, diverse dataset $\mathcal{D}_{\mathrm{pre}}$.

2: Output: Final policy $\pi_R$.

3: Hyperparameters: Rounds $R$, expert queries per round $n_{\mathrm{e}}$, temperature $\tau$, replay ratio $\lambda$.

4: From $\pi_{\mathrm{b}}$, train an ensemble $\Pi_0$ on $\mathcal{D}_{\mathrm{pre}}$.

5: Initialize buffer $\mathcal{D}_{\mathrm{new}}^{(\leq 0)} \leftarrow \emptyset$.

6: for $r = 0$ to $R - 1$ do ▷ Active learning loop.

7: for $k = 1$ to $K$ do

8: for each $\boldsymbol{o}_{kl} \in \mathcal{O}_k$ do

9: Compute VFD uncertainty $u_{kl}^{(r)} = u_{\mathrm{e}}(\boldsymbol{o}_{kl}; \Pi_r)$ via (7).

10: Compute task uncertainty $U_k^{(r)}$ via (8).

11: Compute task weights $W^{(r)} = (w_1^{(r)}, \ldots, w_K^{(r)})$ via $w_k^{(r)} = \frac{(U_k^{(r)})^\tau}{\sum_{k'=1}^K (U_{k'}^{(r)})^\tau}$.

12: for $\xi = 1$ to $n_{\mathrm{e}}$ do ▷ Data acquisition.

13: Sample $k \sim \operatorname{Cat}(W^{(r)})$.

14: Select the most uncertain initial observation $\boldsymbol{o}^*$ from task $T_k$ via (9).

15: Query the expert to collect a demonstration $\tau^*$ starting from $\boldsymbol{o}^*$ and add $\tau^*$ to $\mathcal{D}_{\mathrm{new}}^{(\leq r)}$.

16: Construct training mixture $\mathcal{D}_{\mathrm{train}}^{(r)} \sim \lambda$ Unif $(\mathcal{D}_{\mathrm{pre}}) + (1 - \lambda)$ Unif $(\mathcal{D}_{\mathrm{new}}^{(\leq r)})$.

17: $\Pi_{r+1} \leftarrow \text{Fine-tune}(\Pi_r, \mathcal{D}_{\mathrm{train}}^{(r)})$, $D_{\mathrm{new}}^{(\leq r+1)} \leftarrow D_{\mathrm{new}}^{(\leq r)}$.
</div>

AMF implementation. The central idea of AMF (Bagatella et al., 2025) is to use the demonstrations collected so far to estimate which task’s next demonstration would most improve performance across all tasks. It does so by selecting the task that maximizes the cumulative mutual information between a new demonstration for that task and the policies for all relevant tasks.

Since this method uses previously collected demonstrations for task selection, we start with a prefill phase, collecting one episode per task (for our LIBERO experiments, this takes the first two rounds, using the same acquisition budget as the other methods). We encode frames from previous demonstrations with loss-gradient embeddings from the last action-projection layer (Bagatella et al.

(2025) ablate this decision and show slight improvement over last-layer features) and construct a normalized kernel using the inner product of these embeddings. Intuitively, frames whose losses suggest similar parameter updates have high kernel-similarity. For a task with more than one previously collected demonstration, we evaluate each demonstration separately. Using the lossgradient kernel, we compute how the covariance would change if we received information equivalent to that demonstration. This reduces uncertainty at its frames and at other frames whose gradient embeddings are correlated with them. We aggregate the remaining uncertainty by summing frame variances within each demonstration, averaging across demonstrations within each task, and averaging across tasks equally. We then greedily select the task whose demonstration leaves the lowest remaining total uncertainty until the demonstration budget $n _ { e }$ is reached and perform the same training steps as with the other methods.

Sample Efficiency Computation. The goal of active fine-tuning is to improve performance with as little data as possible. To quantify this capability, we measure sample efficiency as follows. Denote the success rate of method A after round $r _ { 1 }$ by $\operatorname { S R } _ { A , r _ { 1 } }$ . Let $r _ { 2 } = \bar { \mathrm { m i n } } \{ r : \mathrm { S R } _ { B , r } \geq \mathrm { S R } _ { A , r _ { 1 } } \}$ be the number of rounds required for method B to reach at least a similar success rate as $\operatorname { S R } _ { A , r _ { 1 } }$ . Then, if $r _ { 2 } < r _ { 1 }$ , method B requires $\frac { r _ { 1 } - r _ { 2 } } { r _ { 1 } } \times 1 0 0 \%$ fewer samples to achieve the same performance, i.e., method B is $\begin{array} { r l r } {  { \big ( \frac { r _ { 1 } } { r _ { 2 } } - 1 \big ) \times 1 0 0 \% } } \end{array}$ more sample-efficient than method A. To obtain robust results, we average across $\bar { r } _ { 1 } \in \{ R - 2 , R - 1 , R \}$ . With this evaluation protocol, we find that with SmolVLA, SAVE w/ VFD requires 50 % fewer samples than Random, 45 % fewer samples than Diversity, 40 % fewer samples than SAVE w/ Action-L2, and 22 % fewer samples than SAVE w/ GU, corresponding to 101 %, 83 %, 68 %, and 29 % higher sample efficiency, respectively.

## B.5 FAILURE DETECTION

For our experiments, we build on the failure prediction framework FIPER (Römer et al., 2025), released under the MIT License.

## B.5.1 ENVIRONMENT DETAILS

Table 7 provides details about the rollouts in simulation and the real world. Unless otherwise stated, VFD uses a two-member ensemble consisting of the policy being rolled out and evaluated and a second, independently trained model.

Table 7: Details about the failure detection experiments.

<table><tr><td>Environment</td><td>Push-T</td><td>LIBERO-Plus</td><td>Real world</td></tr><tr><td>Policy</td><td>FM-Policy</td><td>SmolVLA</td><td>SmolVLA</td></tr><tr><td>Action chunk length H</td><td>16</td><td>50</td><td>30</td></tr><tr><td>Action execution steps h</td><td>8</td><td>25</td><td>15</td></tr><tr><td>Tasks</td><td>1</td><td>10</td><td>3</td></tr><tr><td>Calibration rollouts per task</td><td>20</td><td>20</td><td>10</td></tr><tr><td>Test rollouts</td><td>200</td><td>1,800</td><td>152</td></tr><tr><td>Policy success rate</td><td>0.20</td><td>0.49</td><td>0.56</td></tr><tr><td>Timesteps per rollout</td><td>50–300</td><td>140–520</td><td>160–500</td></tr></table>

Push-T. We use the ten FM-Policy ensembles trained on 50 of the 206 demonstrations (Section B.3.1). Since Push-T involves no distribution shift, the environment tests the capabilities of failure detectors in in-distribution settings.

LIBERO-Plus. The policies are trained on all demonstrations of the LIBERO-Spatial, -Object, -Goal, and LIBERO-Long suites. The perturbations reduce the success rate of our SmolVLA policies from 0.79 on the unperturbed LIBERO-Long tasks to 0.34 under camera pertur bations; 0.37 under sensor noise; 0.39 when perturbing the robot initial state; 0.55 under background perturbations; 0.60 under lighting changes; 0.64 with changed object layout; 0.06 for language perturbations. Due to its strong imbalance of successes and failures, we do not evaluate failure detection on the language perturbation family. For each task and perturbation family, we record 30 test rollouts, one per perturbation variant, which yields 1,800 test rollouts. To calibrate the thresholds, we use 20 successful rollouts per task, which are generated by rolling out the same policy on the unperturbed LIBERO-Long tasks.

Real world. We fine-tune SmolVLA together on all three tasks Place, Stack, and $\mathtt { D } \mathtt { r a w e r }$ using 100 demonstrations per task. For each task, we record approximately 50 test rollouts with novel background objects, yielding 152 rollouts, of which 85 succeed. We assess robustness with respect to the calibration data by repeating the evaluation for three independent calibration draws, each using 10 successful in-distribution rollouts per task.

## B.5.2 VFD SCORE FOR FAILURE DETECTION

Since runtime monitoring requires low latency, we use the failure detection experiments to investigate whether the computational cost of VFD can be reduced without substantially decreasing performance. To this end, we modify the VFD score (7) in two ways. First, we only evaluate the velocity fields along the ODE trajectories of the acting policy. These trajectories are integrated anyway to generate actions, and the policy’s own velocities at their intermediate states are computed as part of this integration, so VFD only adds evaluations of the other ensemble members’ velocity fields at those states. Second, we only evaluate the velocity difference at a subset $\mathcal { S } = \{ s _ { 1 } < \cdots < s _ { N _ { \mathrm { d } } } \} \subseteq \{ 0 , \delta s , \ldots , 1 - \delta s \}$ of $N _ { \mathrm { d } }$ flow times, weighted by the spacing between them, instead of at all $N _ { s } \doteq 1 / \delta \varepsilon$ s integration steps. Specifically, we compute the VFD score as

$$
u _ {\mathrm{e}} (\boldsymbol {y}) = \frac {1}{(M - 1) B} \sum_ {m = 2} ^ {M} \sum_ {b = 1} ^ {B} \sum_ {k = 1} ^ {N _ {\mathrm{d}}} \kappa_ {s _ {k}} \Delta s _ {k} \left\| \boldsymbol {v} _ {s _ {k}} ^ {\boldsymbol {\theta} _ {1}} \big (\boldsymbol {x} _ {s _ {k}} ^ {(b)}, \boldsymbol {y} \big) - \boldsymbol {v} _ {s _ {k}} ^ {\boldsymbol {\theta} _ {m}} \big (\boldsymbol {x} _ {s _ {k}} ^ {(b)}, \boldsymbol {y} \big) \right\| _ {2} ^ {2},\tag{42}
$$

where $\pmb { x } _ { s } ^ { ( b ) }$ is the state of the b-th action chunk at flow time s along the policy’s own integration, $\Delta s _ { k } = s _ { k + 1 } - s _ { k }$ for $k < N _ { \mathrm { d } } , \Delta s _ { N _ { \mathrm { d } } } = 1 - s _ { N _ { \mathrm { d } } }$ , and $\begin{array} { r } { \kappa _ { s } \ = \ \frac { s } { 1 - s } } \end{array}$ as defined in Section 4. We investigate ensemble sizes up to $M = 4$

## B.5.3 EVALUATION PROTOCOL

Threshold computation via conformal prediction. For each task, we set a constant threshold γ to the empirical q-quantile of the maximum score within each calibration rollout (Agia et al., 2024), with $q = 0 . 9$ for all methods and environments. A test rollout is flagged as a failure at the first action-generation timestep at which its score exceeds $\gamma .$ . This corresponds to split conformal prediction (Angelopoulos & Bates, 2023) with the rollout-wise maximum score as the nonconformity score.

Metrics. Let $N _ { \mathrm { f a i l } }$ and $N _ { \mathrm { s u c c } }$ denote the number of failed and successful test rollouts of a task. The true positive rate (TPR) is the fraction of failed rollouts that are flagged, the true negative rate (TNR) the fraction of successful rollouts that are not flagged, and the balanced accuracy is Acc = $\scriptstyle { \frac { 1 } { 2 } } \left( \mathrm { T P R } + \mathrm { T N R } \right)$ . For a correctly flagged failed rollout n, the detection time $\mathrm { D T } _ { n } = t _ { n } ^ { * } / \dot { ( } T _ { \mathrm { m a x } } - 1 )$ is the timestep $t _ { n } ^ { * }$ of the first flagged step, normalized by the length $T _ { \mathrm { m a x } }$ of the longest rollout of the respective task, and DT denotes its average over all correctly flagged failures. If no failure is flagged, $\mathrm { D T } = 1$ . Timestep-wise accuracy (Römer et al., 2025) rewards earlier detections and is calculated as $\begin{array} { r } { \mathrm { T W A } = \frac { 1 } { 2 } \left( \frac { 1 } { N _ { \mathrm { f a i l } } } \sum _ { n \in \mathcal { T } } \left( 1 - \mathrm { D T } _ { n } \right) + \mathrm { T N R } \right) } \end{array}$ , where $\tau$ is the set of correctly flagged failed rollouts. Hence, $\mathrm { T } \dot { \mathrm { W A } } \leq \mathrm { A c c }$ , with equality if every correctly detected failure is flagged at the first step. In Tables 3, 12, 13 and 14, bold and underlined values denote the best and second-best method per environment and metric. A detector that raises alarms on very few or instantly on all rollouts can attain a low detection time without being useful. Therefore, we rank DT only among methods with $\mathrm { T P R } \geq 0 . 4$ and $\mathrm { T N R } \geq 0 . 4$ , applied to unrounded values (e.g., the real-world TNR of RND-OE (0.398) is displayed as 0.40 but excluded from the DT ranking). TPR and TNR are not ranked.

## B.5.4 BASELINES

We evaluate all methods on the same rollouts and using the same procedure to calculate conformal prediction thresholds.

STAC (Agia et al., 2024) measures the temporal consistency of the action chunk distribution. At each policy inference step $0 , h , 2 h , \ldots$ , it compares the action chunks $\pmb { A } _ { t }$ sampled at the current step with those sampled at the previous step $\mathbf { A } _ { t - h }$ over their temporally overlapping parts of length $H - h$ The score is the maximum mean discrepancy between the two sample sets using an RBF kernel and computed only over the action dimensions corresponding to positions. STAC uses a batch size of 128 for Push-T and 32 for LIBERO-Plus and the real-world experiments.

logpZO (Xu et al., 2025) fits an FM model to the policy’s observation embeddings and scores an embedding $e _ { t }$ by the squared norm $\| z _ { t } \| _ { 2 } ^ { 2 }$ of the noise ${ \boldsymbol { z } } _ { t } = { \boldsymbol { e } } _ { t } - { \boldsymbol { v } } _ { 1 } ( { \boldsymbol { e } } _ { t } )$ that one backward Euler step of the learned velocity field maps it to. Embeddings far from the training distribution map to noise of large norm. Following the authors, the velocity field is parameterized as a 1D convolutional U-Net with channel dimensions (256, 512, 1024), kernel size $5 ,$ 8 groups, and a 128-dimensional time embedding, trained for 200 epochs with learning rate $1 0 ^ { - 4 }$ and batch size 128.

RND-OE (Römer et al., 2025) applies random network distillation to the observation embeddings. A predictor network is trained to match the output of a fixed, randomly initialized target network, and the failure detection score is defined as the prediction error between the two networks’ outputs. Following the authors, we train the predictor with an $\ell _ { 2 }$ loss using AdamW with learning rate $1 \bar { 0 } ^ { - 4 }$ cosine decay to $1 0 ^ { - 6 }$ , weight decay $1 0 ^ { - 5 }$ , and batch size 256 for up to 250 epochs, with early stopping on a held-out 10 % validation split.

ACE (Römer et al., 2025) is computed from the same position-only action samples as STAC and described in Section B.3.2.

We do not consider SAFE (Gu et al., 2025) as a baseline because it requires failure data for training. We deliberately calibrate the thresholds only on successful rollouts, which are easier to obtain.

For a fair comparison with VFD, we train logpZO and RND-OE on the same data used to train the policy ensemble. In Push-T, we use the 50 demonstrations of the ensemble, and the embedding is the 66-dimensional observation conditioning vector of the FM-Policy. For SmolVLA, the embedding consists of the flattened keys and values from the KV cache of the middle VLM layer over the prefix tokens. In the real world, we fit one detector per task on its 100 demonstrations. In LIBERO-Plus, we fit one detector per policy on its training demonstrations, which are subsampled with a stride of 10 for computational reasons. Since the number of instruction tokens varies across tasks, we zero-pad the embeddings after the instruction block to the length of the longest instruction, such that a single detector can cover all tasks.

## B.5.5 COMPUTATIONAL COST

Table 4 reports results for the SmolVLA policy used in our real-world experiments. The second row corresponds to the default VFD configuration used in Table 3. We measure the wall-clock inference time per action-generation step on a single NVIDIA RTX 5090 GPU, averaged over 60 repetitions. The base policy, i.e., one VLM prefix pass that computes the KV cache followed by 10 Euler integration steps of the action expert, takes 170 ms to generate a single action chunk. VFD integrates a batch of $\mathbf { \bar { \boldsymbol { B } } }$ action chunks and, for each of the $M - 1$ other ensemble members, requires an additional prefix pass and evaluating that member’s velocity field at the $N _ { \mathrm { d } }$ ODE states of the base policy for the flow times in ${ \mathcal { S } } .$ . On this GPU and at $B = 8 ,$ , a prefix pass takes approximately 25 ms and a velocity-field evaluation approximately 14 ms. The inference time column reports the added time per action-generation step, and the relative column reports the total inference time relative to the base policy. We find that using a larger ensemble increases balanced accuracy by 5 pp, albeit at the cost of 162 ms longer inference.

Weight perturbation as an alternative to ensembling. To assess whether VFD can avoid training an ensemble altogether, we replace the second ensemble member with a perturbed version of the policy itself. To this end, we add independent zero-mean Gaussian noise with standard deviation σ std(W) to each weight tensor W, i.e., scaled to the spread of the tensor’s own values. This corresponds to a single sample from a Gaussian posterior centered at the trained weights with a heuristic, tensor-wise diagonal covariance. We draw the perturbation once with a fixed random seed and keep it fixed for all rollouts and steps, including the calibration rollouts, so that the thresholds are calibrated for the same perturbed model that is evaluated on the test rollouts. We perturb the VLM backbone (vision encoder and language model, 460.2M parameters) and the action expert, including its action and time projections (97.4M parameters). The language-modeling head of the VLM is not used for action generation and remains unchanged. We evaluate the perturbed model at the recorded ODE states of the policy, at a single flow time $s = 0 . 8$ and with $B = 1$ . We select σ from combinations of separate noise scales $\sigma _ { \mathrm { e x p e r t } } , \sigma _ { \mathrm { V L M } } \in \{ 0 , 0 . 0 0 1 , 0 . 0 0 5 , 0 . 0 1 , 0 . 0 5 , 0 . 1 , 0 . 2 , 0 . 5 \}$ for the two modules. Perturbing only the action expert, which would allow the perturbed model to reuse the policy’s KV cache and reduce the relative inference time to 1.08 , yields a balanced accuracy of at most 0.62. Balanced accuracy plateaus at 0.66 to 0.68 for $\sigma _ { \mathrm { V L M } } \in \mathrm { [ 0 . 1 , 0 . 2 ] }$ and $\sigma _ { \mathrm { e x p e r t } } \in \ [ 0 . 0 5 , 0 . 2 ]$ , and drops below 0.46 for $\sigma _ { \mathrm { V L M } } = 0 . 5$ . Because the VLM is perturbed, the perturbed model requires its own prefix pass, such that the inference time matches that of using an ensemble. In summary, this experiment indicates that VFD can yield competitive uncertainty estimates even without training a deep ensemble.

## C ADDITIONAL RESULTS

## C.1 TOY EXAMPLE

We illustrate the capabilities of VFD for quantifying epistemic uncertainty in FM models with a simple example.

Example 1. Consider a conditional generative modeling problem with ${ \pmb x } = ( x _ { 1 } , x _ { 2 } )$ $\pmb { y } = ( \bar { y } _ { 1 } , y _ { 2 } ) \in \mathbb { R } ^ { 2 }$ . The ground-truth conditional distribution is a two-component Gaussian mixture with input-dependent means; $q ( { \pmb x } \mid { \pmb y } ) = 0 . 5 \mathcal { N } ( { \pmb x } \mid \mu _ { 1 } ( { \pmb y } ) , { \pmb I } ) + 0 . 5 \hat { \mathcal { N } } ( { \pmb x } \mid \mu _ { 2 } ( { \pmb y } ) , { \pmb I } )$ , where ${ \pmb \mu } _ { 1 } ( { \pmb y } ) = ( \sin \left( \pi y _ { 1 } \right) + 0 . 5 , \cos \left( \pi y _ { 2 } \right) - 0 . 5 )$ and $\pmb { \mu } _ { 2 } ( \pmb { y } ) = ( \cos { ( 2 \pi y _ { 2 } ) } - 0 . 5 , \sin { ( 0 . 5 \pi y _ { 1 } ) } + 0 . 5 )$ We draw a set of conditioning inputs $\{ \pmb { y } ^ { ( n ) } \} _ { n = 1 } ^ { N }$ with $N \ = \ 4 0 0 0$ from a uniform distribution over an annulus with inner radius 0.75 and outer radius 1.25 and sample $\mathbf { \Delta } \mathbf { x } ^ { ( n ) } \sim q ( \mathbf { \Delta } \mathbf { x } \mid \mathbf { y } ^ { ( n ) } )$ . We parameterize the velocity field using an MLP and train an ensemble of $M = 5$ FM models with a batch size of 256 for 300 epochs using the standard loss (3). As shown in Figure 8, our VFD score assigns regions far from the training data a high level of epistemic uncertainty, which closely resembles the average KL divergence $\begin{array} { r } { \frac { 1 } { M } \sum _ { i = 1 } ^ { M } D _ { \mathrm { K L } } ( p ^ { \theta _ { i } } | | q ) } \end{array}$ between the learned distributions and the ground truth.

![](images/dee073132c33c2e725e0b1574b15fb8d2e81090b6ea2fd84f2dc4b6c93508531.jpg)

![](images/1e8a959e84d4125875b123b67fd4bbff9e708ed2fe3b3d407f2982f1d501a2d9.jpg)

KL divergence $D _ { \mathrm { K L } } ( p ^ { \pmb { \theta } _ { i } } \parallel q )$  
![](images/10019d92bea44b5f453ddee34d09a1a766a96bb58c1fed726b1b5d18b60e13d4.jpg)  
Figure 8: VFD for a 2D toy example. We visualize epistemic uncertainty estimation for a 2D generative modeling problem. Our VFD uncertainty score is high for inputs far from the training distribution, similar to the KL divergence between the learned models’ conditional distributions and the ground truth.

## C.2 CALIBRATION

Table 8 is an extension of Table 1 and also reports the Pearson correlation coefficient, measuring the strength of the linear relationship between per-task uncertainties and success rates.

## C.2.1 ENSEMBLE SIZE ABLATION

Since fine-tuning generalist policies is expensive, keeping the ensemble small (or avoiding it entirely, cf. Table 4) is highly desirable. Figure 9 shows the impact of the ensemble size M on calibration for all fine-tuning rounds of SmolVLA on $\mathtt { L I B E R O - L o n g }$ . The VLA ensembles consistently remain well-calibrated even for $M = 2$ , indicating that they do not collapse to a single mode of the posterior. Due to the competitive performance and computational advantage of two-member ensembles, we adopt this ensemble size in the experiments unless stated otherwise.

Table 8: Calibration analysis – extended. Negative Spearman’s rank correlation ( ) and negative Pearson correlation ( ) between uncertainty estimates and per-task success rates in LIBERO-Long, averaged across iterative fine-tuning rounds. VFD offers superior calibration compared to other uncertainty estimation approaches.

<table><tr><td>Policy</td><td>Metric</td><td>Action-L2</td><td>ACE</td><td>DECU</td><td>GU</td><td>Entropy</td><td>Perplexity</td><td>VFD (ours)</td></tr><tr><td rowspan="2">SmolVLA</td><td>-Spearman</td><td> $0.50^{\pm 0.13}$ </td><td> $0.31^{\pm 0.12}$ </td><td> $0.31^{\pm 0.13}$ </td><td> $0.62^{\pm 0.00}$ </td><td> $0.10^{\pm 0.12}$ </td><td> $-0.04^{\pm 0.09}$ </td><td> $\mathbf{0.71}^{\pm 0.03}$ </td></tr><tr><td>-Pearson</td><td> $0.48^{\pm 0.09}$ </td><td> $0.36^{\pm 0.08}$ </td><td> $0.23^{\pm 0.15}$ </td><td> $0.65^{\pm 0.02}$ </td><td> $0.23^{\pm 0.21}$ </td><td> $0.02^{\pm 0.15}$ </td><td> $\mathbf{0.71}^{\pm 0.02}$ </td></tr><tr><td rowspan="2">X-VLA</td><td>-Spearman</td><td> $0.33^{\pm 0.24}$ </td><td> $0.14^{\pm 0.05}$ </td><td> $0.06^{\pm 0.04}$ </td><td> $0.48^{\pm 0.06}$ </td><td>N/A</td><td>N/A</td><td> $\mathbf{0.53}^{\pm 0.02}$ </td></tr><tr><td>-Pearson</td><td> $0.33^{\pm 0.22}$ </td><td> $0.15^{\pm 0.07}$ </td><td> $0.06^{\pm 0.05}$ </td><td> $0.45^{\pm 0.13}$ </td><td>N/A</td><td>N/A</td><td> $\mathbf{0.48}^{\pm 0.08}$ </td></tr><tr><td rowspan="2">FastWAM</td><td>-Spearman</td><td> $-0.12^{\pm 0.06}$ </td><td> $-0.29^{\pm 0.07}$ </td><td> $0.03^{\pm 0.07}$ </td><td> $0.27^{\pm 0.04}$ </td><td>N/A</td><td>N/A</td><td> $\mathbf{0.44}^{\pm 0.17}$ </td></tr><tr><td>-Pearson</td><td> $-0.03^{\pm 0.14}$ </td><td> $-0.20^{\pm 0.07}$ </td><td> $0.06^{\pm 0.05}$ </td><td> $0.29^{\pm 0.06}$ </td><td>N/A</td><td>N/A</td><td> $\mathbf{0.42}^{\pm 0.12}$ </td></tr></table>

![](images/b9ee9175a67d7f7ef8a378ba92585979e524f9f79f8d6cd8546a2617ee4dd53a.jpg)  
Figure 9: Ensemble size ablation. During active fine-tuning, the number of ensemble models has little effect on the calibration of VFD.

## C.2.2 ENSEMBLING VS. LAPLACE APPROXIMATION

We compare training an ensemble to computing the Laplace approximation (Daxberger et al., 2021) for sampling from the model posterior. Since computing the full Laplace approximation for the entire SmolVLA model (550M) is computationally intractable, we resort to the last-layer Laplace approximation, which is also used by Jazbec et al. (2025) to compute GU in the original work. We fit a diagonal last-layer Laplace approximation with laplace-torch (Daxberger et al., 2021) around each trained SmolVLA checkpoint. Concretely, we freeze the full model and place the posterior only on the final action-projection layer, which maps the action expert’s hidden states to the predicted FM action velocity. The Laplace posterior is fit as a regression model on the same training episodes used for that active-learning round: for each calibration frame, we sample FM time s, noise $\pmb { x } _ { 0 } \sim \mathcal { N } ( \mathbf { 0 } , I )$ and target action x<sub>1</sub>, and use the corresponding velocity ${ \pmb x } _ { 1 } - { \pmb x } _ { 0 }$ as the regression target. We use all selected training episodes for the round, a calibration fraction of 1.0, and a diagonal Hessian approximation.

Even when limited to the last layer, the computational cost of the last-layer Laplace approximation remains considerable, requiring about 5 hours per checkpoint per round on an NVIDIA RTX 4090 GPU. This is much higher than the 30 minutes it takes to fine-tune a second ensemble member for 4,000 steps on the same GPU. Due to the high cost, we only fit the Laplace approximation for the first 5 rounds. As shown in Figure 10, the two-member ensemble provides much better VFD score calibration than the Laplace approximation despite being faster to train.

![](images/6c917446d27140f0c40e6ced1fab59b59bbcdc8154b399e6779ec4c9fa99232d.jpg)  
Figure 10: Ensembling vs. Laplace Approximation. Calibration is measured by the negative Spearman rank correlation (left) and the negative Pearson correlation (right) between the average task success rate and VFD uncertainty. Two separate ensemble members achieve much better calibration than sampling a second model from the last-layer Laplace approximation.

Table 9: Statistical significance on Push-T. Mean and sample standard deviation over ten paired training seeds.

<table><tr><td>Metric</td><td>SAVE w/ VFD</td><td>SAVE w/ GU</td><td>Difference</td><td>VFD better</td><td> $p_{\text{Wilcoxon}}$ </td><td> $p_{t-test}$ </td></tr><tr><td>AULC</td><td> $46.6^{\pm2.5}$ </td><td> $44.7^{\pm2.9}$ </td><td>+1.9 [+0.8, +3.2]</td><td>9/10</td><td>0.006</td><td>0.010</td></tr><tr><td>Last-3 SR</td><td> $59.3^{\pm3.1}$ </td><td> $54.3^{\pm3.3}$ </td><td>+5.0 [+2.8, +7.1]</td><td>9/10</td><td>0.004</td><td>0.001</td></tr></table>

## C.3 LEARNING CURVES OF ACTIVE FINE-TUNING

In Figure 11, we provide the full learning curves in our simulation experiments, corresponding to the results reported in Tables 2 and 10. Figure 12 shows the active fine-tuning performance for each of the three real-world tasks.

## C.4 STATISTICAL SIGNIFICANCE OF UNCERTAINTY-GUIDED DATA ACQUISITION

The active fine-tuning experiments with FM-Policy on Push-T require far fewer computational resources than those with the three generalist policies. Therefore, we can repeat experiments across ten seeds in this environment to conduct a statistical test. Specifically, we assess whether the advantage of VFD over the strongest uncertainty-based baseline, GU, observed in all environments, is statistically robust. We compare the two methods in a paired manner: For each seed, both start from the same pre-trained ensemble and use the same training seed. We test the one-sided hypothesis that SAVE w/ VFD achieves higher AULC and Last-3 SR than SAVE w/ GU using the Wilcoxon signed-rank test and the paired t-test. As shown in Table 9, SAVE w/ VFD outperforms SAVE w/ GU in nine out of ten seeds on both metrics, and the improvement is significant at the 1 % level for both metrics under the Wilcoxon test. These conclusions also hold for two-sided tests at a 2 % level and after a Holm correction for testing two metrics.

## C.5 ITERATIVE FINE-TUNING VS. RETRAINING

We also test whether our iterative active-learning loop, designed to minimize data-collection effort, requires an additional separate full fine-tuning stage to yield optimal policy performance. To this end, we compare the final-round SmolVLA policy π<sub>15</sub> to a policy retrained from the base checkpoint using a 50/50 split between the ensemble training data $\mathcal { D } _ { \mathrm { p r e } }$ and the actively selected data $\mathcal { D } _ { \mathrm { n e w } } ^ { ( \leq 1 5 ) }$

Table 10: Sample efficiency. Number of demonstrations to reach certain success rates (SR), area under the learning curve (AULC) ( ), and Last-3 SR ( ) for different active fine-tuning strategies. We report the six highest SR thresholds that are multiples of 5 % and reached by at least one method. Thresholds not reached within the maximum demonstration budget are marked as $\bf \ddot { \tau } ^ { < } N / \bf R ^ { \prime } \boldsymbol { \mathrm { \dot { \tau } } }$

<table><tr><td colspan="6">Push-T: FM-Policy</td></tr><tr><td>SR Threshold</td><td>Random</td><td>Diversity</td><td>SAVE w/ Action-L2</td><td>SAVE w/ GU</td><td>SAVE w/ VFD</td></tr><tr><td> $\geq 30\%$ </td><td> $30.0^{\pm15.3}$ </td><td> $21.1^{\pm11.4}$ </td><td> $17.5^{\pm13.1}$ </td><td> $15.6^{\pm8.8}$ </td><td> $15.0^{\pm8.2}$ </td></tr><tr><td> $\geq 35\%$ </td><td> $47.5^{\pm27.7}$ </td><td> $36.7^{\pm11.5}$ </td><td> $29.4^{\pm17.4}$ </td><td> $29.4^{\pm12.6}$ </td><td> $28.5^{\pm13.1}$ </td></tr><tr><td> $\geq 40\%$ </td><td> $71.5^{\pm21.9}$ </td><td> $51.7^{\pm16.0}$ </td><td> $48.1^{\pm18.9}$ </td><td> $38.9^{\pm12.2}$ </td><td> $43.0^{\pm8.9}$ </td></tr><tr><td> $\geq 45\%$ </td><td> $98.5^{\pm21.0}$ </td><td> $86.7^{\pm22.4}$ </td><td> $80.0^{\pm38.7}$ </td><td> $61.7^{\pm13.9}$ </td><td> $58.0^{\pm14.0}$ </td></tr><tr><td> $\geq 50\%$ </td><td>N/R</td><td> $109.4^{\pm20.4}$ </td><td>N/R</td><td> $83.3^{\pm12.2}$ </td><td> $78.5^{\pm16.3}$ </td></tr><tr><td> $\geq 55\%$ </td><td>N/R</td><td>N/R</td><td>N/R</td><td> $105.0^{\pm24.2}$ </td><td> $96.0^{\pm16.3}$ </td></tr><tr><td>AULC</td><td> $38.0^{\pm3.2}$ </td><td> $40.5^{\pm1.9}$ </td><td> $39.8^{\pm3.4}$ </td><td> $44.7^{\pm2.9}$ </td><td> $46.6^{\pm2.5}$ </td></tr><tr><td>Last-3 SR</td><td> $46.3^{\pm3.1}$ </td><td> $51.2^{\pm3.2}$ </td><td> $45.0^{\pm2.6}$ </td><td> $54.3^{\pm3.3}$ </td><td> $59.3^{\pm3.1}$ </td></tr></table>

<table><tr><td colspan="7">LIBERO-Long: SmolVLA</td></tr><tr><td>SR Threshold</td><td>Random</td><td>Diversity</td><td>AMF</td><td>SAVE w/ Action-L2</td><td>SAVE w/ GU</td><td>SAVE w/ VFD</td></tr><tr><td> $\geq 40\%$ </td><td> $28.3^{\pm7.6}$ </td><td> $\mathbf{23.3^{\pm2.9}}$ </td><td> $30.0^{\pm5.0}$ </td><td> $25.0^{\pm5.0}$ </td><td> $25.0^{\pm8.7}$ </td><td> $25.0^{\pm5.0}$ </td></tr><tr><td> $\geq 45\%$ </td><td> $33.3^{\pm7.6}$ </td><td> $36.7^{\pm5.8}$ </td><td> $38.3^{\pm5.8}$ </td><td> $30.0^{\pm8.7}$ </td><td> $28.3^{\pm14.4}$ </td><td> $\mathbf{26.7^{\pm5.8}}$ </td></tr><tr><td> $\geq 50\%$ </td><td> $43.3^{\pm12.6}$ </td><td> $48.3^{\pm10.4}$ </td><td> $46.7^{\pm7.6}$ </td><td> $45.0^{\pm15.0}$ </td><td> $36.7^{\pm11.5}$ </td><td> $\mathbf{30.0^{\pm5.0}}$ </td></tr><tr><td> $\geq 55\%$ </td><td>N/R</td><td>N/R</td><td> $53.3^{\pm2.9}$ </td><td> $45.0^{\pm7.1}$ </td><td> $55.0^{\pm10.0}$ </td><td> $\mathbf{40.0^{\pm10.0}}$ </td></tr><tr><td> $\geq 60\%$ </td><td>N/R</td><td>N/R</td><td> $66.7^{\pm10.4}$ </td><td>N/R</td><td> $63.3^{\pm10.4}$ </td><td> $\mathbf{50.0^{\pm5.0}}$ </td></tr><tr><td> $\geq 65\%$ </td><td>N/R</td><td>N/R</td><td>N/R</td><td>N/R</td><td>N/R</td><td> $\mathbf{62.5^{\pm3.5}}$ </td></tr><tr><td>AULC</td><td> $44.6^{\pm2.2}$ </td><td> $45.5^{\pm1.8}$ </td><td> $45.6^{\pm3.3}$ </td><td> $47.2^{\pm5.1}$ </td><td> $48.6^{\pm5.0}$ </td><td> $\mathbf{52.3^{\pm2.2}}$ </td></tr><tr><td>Last-3 SR</td><td> $52.6^{\pm3.3}$ </td><td> $53.1^{\pm2.0}$ </td><td> $59.4^{\pm3.3}$ </td><td> $55.5^{\pm6.8}$ </td><td> $61.0^{\pm2.9}$ </td><td> $\mathbf{66.1^{\pm2.8}}$ </td></tr></table>

<table><tr><td colspan="7">LIBERO-Long: X-VLA</td></tr><tr><td>SR Threshold</td><td>Random</td><td>Diversity</td><td>AMF</td><td>SAVE w/ Action-L2</td><td>SAVE w/ GU</td><td>SAVE w/ VFD</td></tr><tr><td> $\geq 25\%$ </td><td> $18.3^{\pm 2.9}$ </td><td> $15.0^{\pm 0.0}$ </td><td> $18.3^{\pm 2.9}$ </td><td> $16.7^{\pm 2.9}$ </td><td> $13.3^{\pm 2.9}$ </td><td> $11.7^{\pm 2.9}$ </td></tr><tr><td> $\geq 30\%$ </td><td> $21.7^{\pm 7.6}$ </td><td> $20.0^{\pm 0.0}$ </td><td> $23.3^{\pm 7.6}$ </td><td> $21.7^{\pm 2.9}$ </td><td> $16.7^{\pm 7.6}$ </td><td> $15.0^{\pm 5.0}$ </td></tr><tr><td> $\geq 35\%$ </td><td> $26.7^{\pm 7.6}$ </td><td> $\mathbf{21.7^{\pm 2.9}}$ </td><td> $30.0^{\pm 5.0}$ </td><td> $25.0^{\pm 0.0}$ </td><td> $26.7^{\pm 5.8}$ </td><td> $23.3^{\pm 2.9}$ </td></tr><tr><td> $\geq 40\%$ </td><td> $40.0^{\pm 0.0}$ </td><td> $\mathbf{33.3^{\pm 7.6}}$ </td><td> $36.7^{\pm 2.9}$ </td><td> $35.0^{\pm 5.0}$ </td><td> $35.0^{\pm 10.0}$ </td><td> $\mathbf{33.3^{\pm 5.8}}$ </td></tr><tr><td> $\geq 45\%$ </td><td> $50.0^{\pm 13.2}$ </td><td> $50.0^{\pm 15.0}$ </td><td> $51.7^{\pm 2.9}$ </td><td> $56.7^{\pm 7.6}$ </td><td> $51.7^{\pm 10.4}$ </td><td> $\mathbf{43.3^{\pm 2.9}}$ </td></tr><tr><td> $\geq 50\%$ </td><td>N/R</td><td>N/R</td><td> $65.0^{\pm 10.0}$ </td><td> $67.5^{\pm 3.5}$ </td><td> $57.5^{\pm 17.7}$ </td><td> $\mathbf{55.0^{\pm 0.0}}$ </td></tr><tr><td>AULC</td><td> $37.7^{\pm 1.2}$ </td><td> $40.2^{\pm 1.9}$ </td><td> $39.5^{\pm 1.9}$ </td><td> $39.2^{\pm 0.9}$ </td><td> $40.7^{\pm 3.5}$ </td><td> $\mathbf{42.5^{\pm 1.0}}$ </td></tr><tr><td>Last-3 SR</td><td> $45.1^{\pm 0.3}$ </td><td> $49.0^{\pm 3.3}$ </td><td> $51.0^{\pm 1.3}$ </td><td> $50.9^{\pm 2.2}$ </td><td> $49.2^{\pm 1.8}$ </td><td> $\mathbf{52.9^{\pm 2.9}}$ </td></tr></table>

<table><tr><td colspan="7">LIBERO-Long: FastWAM</td></tr><tr><td>SR Threshold</td><td>Random</td><td>Diversity</td><td>AMF</td><td>SAVE w/ Action-L2</td><td>SAVE w/ GU</td><td>SAVE w/ VFD</td></tr><tr><td> $\geq 45\%$ </td><td> $20.0^{\pm 5.0}$ </td><td> $21.7^{\pm 2.9}$ </td><td> $25.0^{\pm 0.0}$ </td><td> $20.0^{\pm 5.0}$ </td><td> $16.7^{\pm 2.9}$ </td><td> $18.3^{\pm 7.6}$ </td></tr><tr><td> $\geq 50\%$ </td><td> $26.7^{\pm 2.9}$ </td><td> $30.0^{\pm 0.0}$ </td><td> $28.3^{\pm 2.9}$ </td><td> $26.7^{\pm 2.9}$ </td><td> $26.7^{\pm 5.8}$ </td><td> $21.7^{\pm 7.6}$ </td></tr><tr><td> $\geq 55\%$ </td><td> $38.3^{\pm 2.9}$ </td><td> $33.3^{\pm 2.9}$ </td><td> $33.3^{\pm 2.9}$ </td><td> $31.7^{\pm 2.9}$ </td><td> $28.3^{\pm 2.9}$ </td><td> $28.3^{\pm 10.4}$ </td></tr><tr><td> $\geq 60\%$ </td><td> $53.3^{\pm 18.9}$ </td><td> $56.7^{\pm 10.4}$ </td><td> $38.3^{\pm 2.9}$ </td><td> $40.0^{\pm 8.7}$ </td><td> $35.0^{\pm 5.0}$ </td><td> $33.3^{\pm 5.8}$ </td></tr><tr><td> $\geq 65\%$ </td><td> $62.5^{\pm 17.7}$ </td><td> $75.0^{\pm 0.0}$ </td><td> $48.3^{\pm 15.3}$ </td><td> $58.3^{\pm 2.9}$ </td><td> $41.7^{\pm 5.8}$ </td><td> $41.7^{\pm 7.6}$ </td></tr><tr><td> $\geq 70\%$ </td><td>N/R</td><td>N/R</td><td>N/R</td><td> $60.0^{\pm 0.0}$ </td><td> $57.5^{\pm 24.7}$ </td><td> $56.7^{\pm 5.8}$ </td></tr><tr><td>AULC</td><td> $54.2^{\pm 2.5}$ </td><td> $53.9^{\pm 0.8}$ </td><td> $55.6^{\pm 1.4}$ </td><td> $57.0^{\pm 3.3}$ </td><td> $59.0^{\pm 4.4}$ </td><td> $61.1^{\pm 3.5}$ </td></tr><tr><td>Last-3 SR</td><td> $64.8^{\pm 6.6}$ </td><td> $63.9^{\pm 0.3}$ </td><td> $69.4^{\pm 1.3}$ </td><td> $71.6^{\pm 5.2}$ </td><td> $70.6^{\pm 6.1}$ </td><td> $74.8^{\pm 3.3}$ </td></tr></table>

When using the demonstrations selected at the best-performing temperature $\tau = 2 . 5 ,$ the retrained policy achieves an overall success rate of $7 0 . 6 \% ^ { \pm 0 . 9 }$ , which is only 3.5 pp higher than that of the iteratively fine-tuned policy. This demonstrates that uncertainty-guided acquisition produces a strong policy already during the iterative active learning process.

![](images/83871f059e514be9466e9641189e45e3695f1093eb75aa2e786d95fd03377ee6.jpg)  
Figure 11: Active learning curves. These correspond to the active fine-tuning results reported in Tables 2 and 10. SAVE w/ VFD consistently yields the largest performance gains with the available demonstration budget.

## C.6 HYPERPARAMETER SELECTION FOR UNCERTAINTY-GUIDED ACTIVE FINE-TUNING

Figure 13 (left) shows that most of the gain of SAVE w/ VFD over random acquisition stems from the prioritization at the task level, while uncertainty-based initial observation selection yields a minor additional improvement. A key reason for this performance gain is shown in the right of Figure 13 and in Figure 15. While pre-trained policies naturally exhibit varying levels of uncertainty across tasks, our uncertainty-guided active fine-tuning strategy can reduce these differences and improve multitask performance more quickly by allocating the demonstration budget to tasks that require it most. Figure 14 complements this observation, showing a detailed view of the trade-off between exploiting the uncertainty estimates and exploring the task space evenly. The diversity of the categorical task distribution can be measured by its entropy. Over multiple rounds, SAVE tends towards high-entropy categorical distributions over tasks (i.e., more diversity in the selection). We observe a Pareto front: SAVE w/ VFD can trade off this diversity (lower exploration) for faster improvement (higher exploitation) in multitask success by focusing on high-uncertainty tasks (cf. Figure 14).

## C.7 FAILURE DETECTION

We provide the full results of our failure detection experiments in Tables 12, 13 and 14.

VFD evaluation flow times. We also examine the impact of evaluating the VFD score (42) not at every ODE state, but only at a subset of evaluation flow times $S \subseteq \{ 0 . \mathsf { \breve { 2 } } , 0 . 4 , 0 . 6 , 0 . 8 \}$ . As shown in Figure 16, later timesteps where the action chunks contain less noise are more informative for estimating epistemic uncertainty. For a large four-member ensemble, restricting the computation of VFD to a single flow timestep $s = 0 . 8$ does not reduce balanced accuracy. Together with the computational cost analysis in Table 4, this experiment indicates that VFD can be computed efficiently while still being a capable estimator of epistemic uncertainty.

![](images/72cfe34df0296c304b7e21d3d716431ae02d14c40e14f6776f48bb8910abb997.jpg)  
Figure 12: Real-world active fine-tuning results. SAVE w/ VFD consistently improves across all tasks, whereas Diversity and SAVE w/ GU stagnate on certain tasks. Notably, random data acquisition is the second-best method on hardware, highlighting the difficulty of effectively allocating demonstration budgets for multitask improvement. These per-task learning curves constitute the average shown in Figure 4.

![](images/feb0ec2ef87ba8330832e884b120a7354a24bf598cd14f4842ef02273836e7fd.jpg)

![](images/d945eaf4555804562f826cca8f1c82ae5575d6b207f8ace5d311592c38660410.jpg)  
Figure 13: Effect of the task-sampling temperature τ on SAVE. Larger τ biases expert queries toward higher-uncertainty tasks, with uniform sampling for $\tau = 0$ . Left: Uncertainty guidance is beneficial both for sampling tasks and initial observations. Legend entries containing uniform correspond to sampling initial observations within a sampled task uniformly instead of uncertainty-guided. Right: Uncertainty-based sampling more rapidly reduces the fraction of uncertainty concentrated in the most uncertain task, indicating better allocation of demonstrations to underperforming tasks.

Table 11: Hyperparameter selection for the uncertainty-guided active fine-tuning methods. Suc cess rate (SR) in percent ( ) after different active fine-tuning rounds for SmolVLA on LIBERO-Long. The settings with the highest final success rate, reported in Tables 2 and 10, are marked with an asterisk (\*).

<table><tr><td>Method</td><td>Hyperparameter</td><td>Round-5 SR</td><td>Round-10 SR</td><td>Round-15 SR</td></tr><tr><td rowspan="4">AMF</td><td> $\sigma = 10^{-4}$ </td><td> $40.4^{\pm 4.1}$ </td><td> $52.2^{\pm 4.4}$ </td><td> $58.1^{\pm 3.1}$ </td></tr><tr><td> $\sigma = 10^{-3}$ </td><td> $40.2^{\pm 4.0}$ </td><td> $53.3^{\pm 3.7}$ </td><td> $59.7^{\pm 6.7}$ </td></tr><tr><td> $\sigma = 10^{-2}*$ </td><td> $38.6^{\pm 4.4}$ </td><td> $52.9^{\pm 3.9}$ </td><td> $60.9^{\pm 0.8}$ </td></tr><tr><td> $\sigma = 10^{-1}$ </td><td> $39.6^{\pm 5.7}$ </td><td> $52.3^{\pm 4.8}$ </td><td> $57.6^{\pm 2.8}$ </td></tr><tr><td rowspan="5">SAVE w/ Action-L2</td><td> $\tau = 0$ </td><td> $40.6^{\pm 3.0}$ </td><td> $51.1^{\pm 4.2}$ </td><td> $54.0^{\pm 4.2}$ </td></tr><tr><td> $\tau = 1$ </td><td> $41.4^{\pm 5.2}$ </td><td> $46.9^{\pm 6.4}$ </td><td> $52.2^{\pm 2.2}$ </td></tr><tr><td> $\tau = 1.5*$ </td><td> $40.7^{\pm 10.1}$ </td><td> $51.8^{\pm 7.0}$ </td><td> $56.8^{\pm 9.3}$ </td></tr><tr><td> $\tau = 2$ </td><td> $40.7^{\pm 11.0}$ </td><td> $53.0^{\pm 7.0}$ </td><td> $55.0^{\pm 3.2}$ </td></tr><tr><td> $\tau = 2.5$ </td><td> $41.8^{\pm 7.0}$ </td><td> $49.0^{\pm 9.0}$ </td><td> $53.4^{\pm 6.9}$ </td></tr><tr><td rowspan="5">SAVE w/ GU</td><td> $\tau = 0$ </td><td> $40.3^{\pm 5.5}$ </td><td> $49.6^{\pm 7.0}$ </td><td> $53.6^{\pm 3.0}$ </td></tr><tr><td> $\tau = 1*$ </td><td> $44.4^{\pm 4.6}$ </td><td> $53.0^{\pm 2.7}$ </td><td> $64.0^{\pm 3.2}$ </td></tr><tr><td> $\tau = 1.5$ </td><td> $42.0^{\pm 1.5}$ </td><td> $57.4^{\pm 5.3}$ </td><td> $59.2^{\pm 6.7}$ </td></tr><tr><td> $\tau = 2$ </td><td> $45.6^{\pm 3.7}$ </td><td> $57.2^{\pm 4.4}$ </td><td> $60.6^{\pm 3.5}$ </td></tr><tr><td> $\tau = 2.5$ </td><td> $46.7^{\pm 4.7}$ </td><td> $59.3^{\pm 4.2}$ </td><td> $60.1^{\pm 2.7}$ </td></tr><tr><td rowspan="5">SAVE w/ VFD</td><td> $\tau = 0$ </td><td> $37.7^{\pm 1.2}$ </td><td> $47.1^{\pm 1.0}$ </td><td> $58.0^{\pm 7.2}$ </td></tr><tr><td> $\tau = 1$ </td><td> $44.8^{\pm 6.6}$ </td><td> $54.8^{\pm 3.3}$ </td><td> $59.2^{\pm 4.8}$ </td></tr><tr><td> $\tau = 1.5$ </td><td> $46.6^{\pm 2.0}$ </td><td> $53.8^{\pm 1.7}$ </td><td> $60.1^{\pm 4.6}$ </td></tr><tr><td> $\tau = 2$ </td><td> $46.0^{\pm 6.7}$ </td><td> $59.0^{\pm 2.6}$ </td><td> $64.2^{\pm 3.7}$ </td></tr><tr><td> $\tau = 2.5*$ </td><td> $44.3^{\pm 6.2}$ </td><td> $61.1^{\pm 5.0}$ </td><td> $67.1^{\pm 4.0}$ </td></tr></table>

![](images/3d45227ecdf74b2ad0f856fd7c806d8508b71ac5d40caeaaf09cb7f8f94aa543.jpg)  
Figure 14: Pareto front of the exploration-exploitation trade-off. SAVE reduces the difference in prior knowledge about tasks; for $\tau \leq 2 . 5$ , the temperature parameter controls an exploration–exploitation trade-off between task coverage and final success rate.

Rollout Visualization. Figures 17 and 18 visualize exemplary real-world rollouts, with the failure detection scores normalized for readability. Generally, we observe that the observation-based

![](images/54ab11d1b1f7bbd881695aab8d8b9de6ec851567a0acd5f8a24b051b63acc36c.jpg)  
Figure 15: Task selection. Behavior of uncertainty-guided data acquisition in active fine-tuning for different temperature values τ over all active fine-tuning rounds with SmolVLA on LIBERO-Long. Left: Number of selected episodes per task. Right: Relative share of overall uncertainty per task.

Table 12: Failure detection results in Push-T. Mean and standard deviation over ten training seeds.

<table><tr><td>Metric</td><td>STAC</td><td>logpZO</td><td>RND-OE</td><td>ACE</td><td>VFD (ours)</td></tr><tr><td>Acc. ↑</td><td> $\underline{0.55}^{\pm 0.00}$ </td><td> $0.53^{\pm 0.01}$ </td><td> $0.50^{\pm 0.02}$ </td><td> $\underline{0.55}^{\pm 0.06}$ </td><td> $\mathbf{0.64}^{\pm 0.05}$ </td></tr><tr><td>TWA ↑</td><td> $\underline{0.51}^{\pm 0.00}$ </td><td> $\underline{0.51}^{\pm 0.02}$ </td><td> $0.50^{\pm 0.02}$ </td><td> $\underline{0.51}^{\pm 0.03}$ </td><td> $\mathbf{0.56}^{\pm 0.05}$ </td></tr><tr><td>DT ↓</td><td> $0.56^{\pm 0.04}$ </td><td> $0.23^{\pm 0.09}$ </td><td> $0.00^{\pm 0.00}$ </td><td> $0.30^{\pm 0.04}$ </td><td> $0.41^{\pm 0.05}$ </td></tr><tr><td>TPR ↑</td><td> $0.13^{\pm 0.02}$ </td><td> $0.18^{\pm 0.09}$ </td><td> $0.15^{\pm 0.08}$ </td><td> $0.23^{\pm 0.15}$ </td><td> $0.38^{\pm 0.08}$ </td></tr><tr><td>TNR ↑</td><td> $0.97^{\pm 0.02}$ </td><td> $0.87^{\pm 0.07}$ </td><td> $0.86^{\pm 0.12}$ </td><td> $0.86^{\pm 0.05}$ </td><td> $0.89^{\pm 0.07}$ </td></tr></table>

Table 13: Failure detection on LIBERO-Plus. We group the results by perturbation family and provide the mean and standard deviation over three training seeds.

<table><tr><td>Perturbation</td><td>Metric</td><td>STAC</td><td>logpZO</td><td>RND-OE</td><td>ACE</td><td>VFD (ours)</td></tr><tr><td rowspan="5">Camera</td><td>Acc. ↑</td><td> $0.68^{\pm 0.06}$ </td><td> $0.65^{\pm 0.02}$ </td><td> $0.61^{\pm 0.02}$ </td><td> $\underline{0.73^{\pm 0.00}}$ </td><td> $\underline{0.72^{\pm 0.05}}$ </td></tr><tr><td>TWA ↑</td><td> $0.56^{\pm 0.04}$ </td><td> $\underline{0.58^{\pm 0.03}}$ </td><td> $\underline{0.57^{\pm 0.02}}$ </td><td> $0.56^{\pm 0.01}$ </td><td> $0.56^{\pm 0.06}$ </td></tr><tr><td>DT ↓</td><td> $\underline{0.32^{\pm 0.02}}$ </td><td> $0.15^{\pm 0.02}$ </td><td> $0.08^{\pm 0.01}$ </td><td> $0.50^{\pm 0.06}$ </td><td> $\underline{0.37^{\pm 0.03}}$ </td></tr><tr><td>TPR ↑</td><td> $0.72^{\pm 0.12}$ </td><td> $0.99^{\pm 0.01}$ </td><td> $1.00^{\pm 0.00}$ </td><td> $0.73^{\pm 0.09}$ </td><td> $0.90^{\pm 0.04}$ </td></tr><tr><td>TNR ↑</td><td> $0.63^{\pm 0.04}$ </td><td> $0.32^{\pm 0.04}$ </td><td> $0.22^{\pm 0.04}$ </td><td> $0.74^{\pm 0.08}$ </td><td> $0.54^{\pm 0.12}$ </td></tr><tr><td rowspan="5">Robot init.</td><td>Acc. ↑</td><td> $0.70^{\pm 0.02}$ </td><td> $0.74^{\pm 0.02}$ </td><td> $0.72^{\pm 0.02}$ </td><td> $\underline{0.76^{\pm 0.04}}$ </td><td> $\underline{0.78^{\pm 0.03}}$ </td></tr><tr><td>TWA ↑</td><td> $0.61^{\pm 0.01}$ </td><td> $\underline{0.68^{\pm 0.03}}$ </td><td> $\underline{0.69^{\pm 0.03}}$ </td><td> $0.64^{\pm 0.05}$ </td><td> $\underline{0.68^{\pm 0.04}}$ </td></tr><tr><td>DT ↓</td><td> $0.30^{\pm 0.02}$ </td><td> $\underline{0.11^{\pm 0.03}}$ </td><td> $\underline{0.07^{\pm 0.01}}$ </td><td> $0.39^{\pm 0.05}$ </td><td> $\underline{0.22^{\pm 0.03}}$ </td></tr><tr><td>TPR ↑</td><td> $0.68^{\pm 0.07}$ </td><td> $0.99^{\pm 0.02}$ </td><td> $0.99^{\pm 0.01}$ </td><td> $0.72^{\pm 0.10}$ </td><td> $0.91^{\pm 0.04}$ </td></tr><tr><td>TNR ↑</td><td> $0.73^{\pm 0.04}$ </td><td> $0.49^{\pm 0.02}$ </td><td> $0.46^{\pm 0.04}$ </td><td> $0.80^{\pm 0.06}$ </td><td> $0.65^{\pm 0.02}$ </td></tr><tr><td rowspan="5">Sensor noise</td><td>Acc. ↑</td><td> $0.67^{\pm 0.06}$ </td><td> $0.66^{\pm 0.04}$ </td><td> $0.66^{\pm 0.02}$ </td><td> $\underline{0.74^{\pm 0.02}}$ </td><td> $\underline{0.80^{\pm 0.02}}$ </td></tr><tr><td>TWA ↑</td><td> $0.56^{\pm 0.05}$ </td><td> $0.61^{\pm 0.03}$ </td><td> $\underline{0.63^{\pm 0.02}}$ </td><td> $0.60^{\pm 0.01}$ </td><td> $\underline{0.67^{\pm 0.02}}$ </td></tr><tr><td>DT ↓</td><td> $0.41^{\pm 0.05}$ </td><td> $0.09^{\pm 0.03}$ </td><td> $0.07^{\pm 0.01}$ </td><td> $\underline{0.39^{\pm 0.12}}$ </td><td> $\underline{0.30^{\pm 0.02}}$ </td></tr><tr><td>TPR ↑</td><td> $0.63^{\pm 0.10}$ </td><td> $1.00^{\pm 0.00}$ </td><td> $1.00^{\pm 0.00}$ </td><td> $0.81^{\pm 0.09}$ </td><td> $0.94^{\pm 0.03}$ </td></tr><tr><td>TNR ↑</td><td> $0.71^{\pm 0.04}$ </td><td> $0.31^{\pm 0.08}$ </td><td> $0.33^{\pm 0.05}$ </td><td> $0.67^{\pm 0.12}$ </td><td> $0.66^{\pm 0.02}$ </td></tr><tr><td rowspan="5">Background</td><td>Acc. ↑</td><td> $0.68^{\pm 0.04}$ </td><td> $0.66^{\pm 0.00}$ </td><td> $0.66^{\pm 0.00}$ </td><td> $\underline{0.74^{\pm 0.03}}$ </td><td> $\underline{0.77^{\pm 0.02}}$ </td></tr><tr><td>TWA ↑</td><td> $0.57^{\pm 0.04}$ </td><td> $\underline{0.61^{\pm 0.02}}$ </td><td> $0.60^{\pm 0.01}$ </td><td> $0.58^{\pm 0.03}$ </td><td> $\underline{0.62^{\pm 0.01}}$ </td></tr><tr><td>DT ↓</td><td> $0.46^{\pm 0.03}$ </td><td> $0.10^{\pm 0.03}$ </td><td> $0.12^{\pm 0.03}$ </td><td> $\underline{0.45^{\pm 0.02}}$ </td><td> $\underline{0.32^{\pm 0.01}}$ </td></tr><tr><td>TPR ↑</td><td> $0.58^{\pm 0.09}$ </td><td> $0.99^{\pm 0.01}$ </td><td> $1.00^{\pm 0.00}$ </td><td> $0.76^{\pm 0.04}$ </td><td> $0.94^{\pm 0.02}$ </td></tr><tr><td>TNR ↑</td><td> $0.78^{\pm 0.03}$ </td><td> $0.33^{\pm 0.01}$ </td><td> $0.33^{\pm 0.00}$ </td><td> $0.73^{\pm 0.05}$ </td><td> $0.60^{\pm 0.03}$ </td></tr><tr><td rowspan="5">Lighting</td><td>Acc. ↑</td><td> $0.69^{\pm 0.04}$ </td><td> $0.64^{\pm 0.06}$ </td><td> $0.64^{\pm 0.05}$ </td><td> $\underline{0.76^{\pm 0.02}}$ </td><td> $\underline{0.80^{\pm 0.02}}$ </td></tr><tr><td>TWA ↑</td><td> $\underline{0.58^{\pm 0.03}}$ </td><td> $0.55^{\pm 0.03}$ </td><td> $0.55^{\pm 0.03}$ </td><td> $0.56^{\pm 0.02}$ </td><td> $\underline{0.63^{\pm 0.03}}$ </td></tr><tr><td>DT ↓</td><td> $\underline{0.40^{\pm 0.07}}$ </td><td> $0.17^{\pm 0.06}$ </td><td> $0.18^{\pm 0.03}$ </td><td> $0.49^{\pm 0.09}$ </td><td> $\underline{0.37^{\pm 0.04}}$ </td></tr><tr><td>TPR ↑</td><td> $0.59^{\pm 0.12}$ </td><td> $1.00^{\pm 0.00}$ </td><td> $1.00^{\pm 0.00}$ </td><td> $0.84^{\pm 0.09}$ </td><td> $0.94^{\pm 0.02}$ </td></tr><tr><td>TNR ↑</td><td> $0.79^{\pm 0.03}$ </td><td> $0.27^{\pm 0.13}$ </td><td> $0.29^{\pm 0.10}$ </td><td> $0.68^{\pm 0.12}$ </td><td> $0.66^{\pm 0.05}$ </td></tr><tr><td rowspan="5">Layout</td><td>Acc. ↑</td><td> $0.70^{\pm 0.02}$ </td><td> $\underline{0.74^{\pm 0.01}}$ </td><td> $0.71^{\pm 0.02}$ </td><td> $\underline{0.74^{\pm 0.04}}$ </td><td> $\underline{0.81^{\pm 0.02}}$ </td></tr><tr><td>TWA ↑</td><td> $0.58^{\pm 0.02}$ </td><td> $\underline{0.60^{\pm 0.01}}$ </td><td> $0.58^{\pm 0.03}$ </td><td> $0.57^{\pm 0.02}$ </td><td> $\underline{0.62^{\pm 0.02}}$ </td></tr><tr><td>DT ↓</td><td> $0.40^{\pm 0.04}$ </td><td> $\underline{0.28^{\pm 0.02}}$ </td><td> $\underline{0.26^{\pm 0.02}}$ </td><td> $0.54^{\pm 0.06}$ </td><td> $0.48^{\pm 0.03}$ </td></tr><tr><td>TPR ↑</td><td> $0.63^{\pm 0.06}$ </td><td> $0.98^{\pm 0.03}$ </td><td> $0.98^{\pm 0.02}$ </td><td> $0.65^{\pm 0.11}$ </td><td> $0.86^{\pm 0.06}$ </td></tr><tr><td>TNR ↑</td><td> $0.77^{\pm 0.04}$ </td><td> $0.50^{\pm 0.04}$ </td><td> $0.43^{\pm 0.03}$ </td><td> $0.84^{\pm 0.05}$ </td><td> $0.77^{\pm 0.04}$ </td></tr><tr><td rowspan="5">Average</td><td>Acc. ↑</td><td> $0.70^{\pm 0.03}$ </td><td> $0.68^{\pm 0.02}$ </td><td> $0.67^{\pm 0.01}$ </td><td> $\underline{0.75^{\pm 0.02}}$ </td><td> $\underline{0.79^{\pm 0.00}}$ </td></tr><tr><td>TWA ↑</td><td> $0.59^{\pm 0.02}$ </td><td> $\underline{0.62^{\pm 0.01}}$ </td><td> $\underline{0.62^{\pm 0.01}}$ </td><td> $0.59^{\pm 0.02}$ </td><td> $\underline{0.65^{\pm 0.00}}$ </td></tr><tr><td>DT ↓</td><td> $\underline{0.35^{\pm 0.01}}$ </td><td> $0.13^{\pm 0.02}$ </td><td> $0.11^{\pm 0.00}$ </td><td> $0.44^{\pm 0.07}$ </td><td> $\underline{0.32^{\pm 0.02}}$ </td></tr><tr><td>TPR ↑</td><td> $0.65^{\pm 0.09}$ </td><td> $0.99^{\pm 0.01}$ </td><td> $0.99^{\pm 0.00}$ </td><td> $0.75^{\pm 0.08}$ </td><td> $0.92^{\pm 0.02}$ </td></tr><tr><td>TNR ↑</td><td> $0.75^{\pm 0.02}$ </td><td> $0.37^{\pm 0.04}$ </td><td> $0.35^{\pm 0.02}$ </td><td> $0.75^{\pm 0.06}$ </td><td> $0.66^{\pm 0.03}$ </td></tr></table>

OOD detectors, logpZO and RND-OE, are prone to raising false alarms (TNR of 0.49 and 0.40, respectively), which happens in both rollouts shown in Figure 17. In contrast, STAC and ACE are unable to detect most failures (TPR of 0.04 and 0.22, respectively), see Figure 18. Compared to the baselines, VFD offers a middle ground, detecting 60 % of all failures while flagging only 13 % of all successful rollouts as failures and achieving the highest overall balanced accuracy of 73 %.

Table 14: Failure detection results in the real world. We report the mean and standard deviation over three draws of calibration rollouts.

<table><tr><td>Task</td><td>Metric</td><td>STAC</td><td>logpZO</td><td>RND-OE</td><td>ACE</td><td>VFD (ours)</td></tr><tr><td rowspan="5">Place</td><td>Acc. ↑</td><td> $0.42^{\pm 0.02}$ </td><td> $\underline{0.71}^{\pm 0.00}$ </td><td> $0.63^{\pm 0.02}$ </td><td> $0.54^{\pm 0.05}$ </td><td> $\underline{0.78}^{\pm 0.02}$ </td></tr><tr><td>TWA ↑</td><td> $0.39^{\pm 0.01}$ </td><td> $\underline{0.65}^{\pm 0.00}$ </td><td> $\underline{0.57}^{\pm 0.02}$ </td><td> $0.51^{\pm 0.04}$ </td><td> $\underline{0.57}^{\pm 0.01}$ </td></tr><tr><td>DT ↓</td><td> $0.80^{\pm 0.26}$ </td><td> $\underline{0.12}^{\pm 0.00}$ </td><td> $0.13^{\pm 0.01}$ </td><td> $0.32^{\pm 0.07}$ </td><td> $\underline{0.50}^{\pm 0.05}$ </td></tr><tr><td>TPR ↑</td><td> $0.10^{\pm 0.08}$ </td><td> $0.98^{\pm 0.03}$ </td><td> $1.00^{\pm 0.00}$ </td><td> $0.21^{\pm 0.12}$ </td><td> $0.86^{\pm 0.00}$ </td></tr><tr><td>TNR ↑</td><td> $0.75^{\pm 0.04}$ </td><td> $0.43^{\pm 0.02}$ </td><td> $0.27^{\pm 0.05}$ </td><td> $0.88^{\pm 0.02}$ </td><td> $0.71^{\pm 0.03}$ </td></tr><tr><td rowspan="5">Stack</td><td>Acc. ↑</td><td> $0.43^{\pm 0.02}$ </td><td> $0.68^{\pm 0.10}$ </td><td> $\underline{0.77}^{\pm 0.05}$ </td><td> $0.58^{\pm 0.03}$ </td><td> $\underline{0.81}^{\pm 0.01}$ </td></tr><tr><td>TWA ↑</td><td> $0.43^{\pm 0.02}$ </td><td> $\underline{0.66}^{\pm 0.10}$ </td><td> $\underline{0.71}^{\pm 0.05}$ </td><td> $0.53^{\pm 0.01}$ </td><td> $0.64^{\pm 0.02}$ </td></tr><tr><td>DT ↓</td><td> $1.00^{\pm 0.00}$ </td><td> $0.03^{\pm 0.00}$ </td><td> $\underline{0.12}^{\pm 0.02}$ </td><td> $0.26^{\pm 0.07}$ </td><td> $\underline{0.54}^{\pm 0.07}$ </td></tr><tr><td>TPR ↑</td><td> $0.00^{\pm 0.00}$ </td><td> $1.00^{\pm 0.00}$ </td><td> $1.00^{\pm 0.00}$ </td><td> $0.38^{\pm 0.13}$ </td><td> $0.64^{\pm 0.05}$ </td></tr><tr><td>TNR ↑</td><td> $0.85^{\pm 0.04}$ </td><td> $0.36^{\pm 0.20}$ </td><td> $0.53^{\pm 0.11}$ </td><td> $0.79^{\pm 0.08}$ </td><td> $0.99^{\pm 0.02}$ </td></tr><tr><td rowspan="5">Drawer</td><td>Acc. ↑</td><td> $0.45^{\pm 0.04}$ </td><td> $\underline{0.71}^{\pm 0.02}$ </td><td> $\underline{0.67}^{\pm 0.05}$ </td><td> $0.48^{\pm 0.04}$ </td><td> $0.60^{\pm 0.01}$ </td></tr><tr><td>TWA ↑</td><td> $0.44^{\pm 0.05}$ </td><td> $\underline{0.61}^{\pm 0.01}$ </td><td> $\underline{0.56}^{\pm 0.01}$ </td><td> $0.46^{\pm 0.05}$ </td><td> $0.53^{\pm 0.01}$ </td></tr><tr><td>DT ↓</td><td> $0.86^{\pm 0.24}$ </td><td> $\underline{0.26}^{\pm 0.04}$ </td><td> $0.24^{\pm 0.15}$ </td><td> $0.49^{\pm 0.09}$ </td><td> $0.49^{\pm 0.00}$ </td></tr><tr><td>TPR ↑</td><td> $0.03^{\pm 0.05}$ </td><td> $0.72^{\pm 0.05}$ </td><td> $0.94^{\pm 0.10}$ </td><td> $0.09^{\pm 0.08}$ </td><td> $0.30^{\pm 0.00}$ </td></tr><tr><td>TNR ↑</td><td> $0.88^{\pm 0.11}$ </td><td> $0.69^{\pm 0.09}$ </td><td> $0.40^{\pm 0.20}$ </td><td> $0.88^{\pm 0.15}$ </td><td> $0.90^{\pm 0.02}$ </td></tr><tr><td rowspan="5">Average</td><td>Acc. ↑</td><td> $0.43^{\pm 0.02}$ </td><td> $\underline{0.70}^{\pm 0.03}$ </td><td> $0.69^{\pm 0.04}$ </td><td> $0.54^{\pm 0.01}$ </td><td> $\underline{0.73}^{\pm 0.01}$ </td></tr><tr><td>TWA ↑</td><td> $0.42^{\pm 0.02}$ </td><td> $\underline{0.64}^{\pm 0.03}$ </td><td> $\underline{0.61}^{\pm 0.02}$ </td><td> $0.50^{\pm 0.00}$ </td><td> $0.58^{\pm 0.00}$ </td></tr><tr><td>DT ↓</td><td> $0.89^{\pm 0.08}$ </td><td> $\underline{0.14}^{\pm 0.01}$ </td><td> $0.16^{\pm 0.05}$ </td><td> $0.36^{\pm 0.07}$ </td><td> $\underline{0.51}^{\pm 0.03}$ </td></tr><tr><td>TPR ↑</td><td> $0.04^{\pm 0.02}$ </td><td> $0.90^{\pm 0.02}$ </td><td> $0.98^{\pm 0.03}$ </td><td> $0.22^{\pm 0.05}$ </td><td> $0.60^{\pm 0.02}$ </td></tr><tr><td>TNR ↑</td><td> $0.83^{\pm 0.05}$ </td><td> $0.49^{\pm 0.04}$ </td><td> $0.40^{\pm 0.10}$ </td><td> $0.85^{\pm 0.05}$ </td><td> $0.87^{\pm 0.02}$ </td></tr></table>

![](images/160945e674ce6be5fedf6616c442b22b9263f079adc2bbb38a54adb7b6ea2a3a.jpg)  
Figure 16: Impact of the flow times for evaluating VFD. Dropping later (larger) flow timesteps from the VFD computation (42) reduces performance more than dropping earlier ones, indicating that disagreements in the velocity fields are more indicative of epistemic uncertainty when the action chunks contain less noise.

![](images/cc5a713444d9440b81e8ae89508fd5ef5ef811e72076bc5693f51e16a776f059.jpg)

![](images/75034f8a8b4f66c840ebcbcf49eff23b259a898752421c25a3f1015f62b16632.jpg)

![](images/7f7b01dccf2196ad0fc13b9c3a2a0f2a808496526d293189fff95a543c186470.jpg)  
Figure 17: Exemplary successful rollouts that are correctly not flagged as failure by VFD.

![](images/34148586c72198b683267fe437ffe6300e83b11a43b14ef5b296115c7c3d37d4.jpg)

![](images/6fe973294bd60b6cf72c18898ac95cda470ca06a19669d88bad1b1730a9f7ff2.jpg)

![](images/aea271f83b99953385e9c4e3dcc45899a18a22a5adcd35624f75b7946e22b217.jpg)

STAC logpZO RND-OE ACE VFD Threshold  
![](images/0e5eac62c9c7901177b5a372b5d2c9b0578e87dd8c2a19a70907a46dc3958e69.jpg)  
Figure 18: Exemplary failed rollouts that are correctly flagged as failure by VFD.