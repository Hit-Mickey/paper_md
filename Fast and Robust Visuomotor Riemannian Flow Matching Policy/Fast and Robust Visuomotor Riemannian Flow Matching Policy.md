# Fast and Robust Visuomotor Riemannian Flow Matching Policy

Haoran Ding<sup>1</sup>, Noemie Jaquier´ <sup>2</sup>, Jan Peters<sup>3</sup>, Leonel Rozo<sup>1</sup>

Abstract—Diffusion-based visuomotor policies excel at learning complex robotic tasks by effectively combining visual data with high-dimensional, multi-modal action distributions. However, diffusion models often suffer from slow inference due to costly denoising processes or require complex sequential training arising from recent distilling approaches. This paper introduces Riemannian Flow Matching Policy (RFMP), a model that inherits the easy training and fast inference capabilities of flow matching (FM). Moreover, RFMP inherently incorporates geometric constraints commonly found in realistic robotic applications, as the robot state resides on a Riemannian manifold. To enhance the robustness of RFMP, we propose Stable RFMP (SRFMP), which leverages LaSalle’s invariance principle to equip the dynamics of FM with stability to the support of a target Riemannian distribution. Rigorous evaluation on ten simulated and realworld tasks show that RFMP successfully learns and synthesizes complex sensorimotor policies on Euclidean and Riemannian spaces with efficient training and inference phases, outperforming Diffusion Policies and Consistency Policies.

Index Terms—Learning from demonstrations; Learning and adaptive systems; Deep learning in robotics and automation; Visuomotor policies; Riemannian flow matching

## I. INTRODUCTION

D <sup>EEP</sup> <sup>generative</sup> <sup>models</sup> <sup>are</sup> <sup>revolutionizing</sup> <sup>robot</sup> <sup>skill</sup> learning due to their ability to handle high-dimensional multimodal action distributions and interface them with perception networks, enabling robots to learn sophisticated sensorimotor policies [1]. In particular, diffusion-based models such as diffusion policies (DP) [2]–[7] exhibit exceptional performance in imitation learning for a large variety of simulated and real-world robotic tasks, demonstrating a superior ability to learn multimodal action distributions compared to previous behavior cloning methods [8]–[10]. Nevertheless, these models are characterized by an expensive inference process as they often require to solve a stochastic differential equation, thus hindering their use in certain robotic settings [11], e.g., for highly reactive motion policies.

For instance, DP [2], typically based on a Denoising Diffusion Probabilistic Model (DDPM) [12], requires approximately 100 denoising steps to generate an action. This translates to roughly 1 second on a standard GPU. Even faster approaches such as Denoising Diffusion Implicit Models (DDIM) [13] still need 10 denoising steps, i.e., 0.1 second, per action [2]. Consistency policy [3] aims to accelerate the inference process by training a student model to mimic a DP teacher with larger denoising steps. Despite providing a more computationally-efficient inference, the CP training requires more computational resources and might be unstable due to the sequential training of the two models. Importantly, training these models becomes more computationally demanding when manipulating data with geometric constraints, e.g., robot endeffector orientations, as the computation of the score function of the diffusion process is not as simple as in the Euclidean case [14]. Furthermore, the inference process also incurs increasing computational complexity.

![](images/bc8e565aa4d1ae516c3e233e532bc76d3707f71a07beb4ae443558076d211b14.jpg)  
Fig. 1. Flows of the RFMP (top) and SRFMP (bottom) at times t = {0.0, 1.0, 1.5}. The policies are learned from pick-and-place demonstration (black) and conditioned on visual observations. Note that the flow of SRFMP is stable to the target distribution at t > 1.0, enhancing the policy robustness and inference time.

To overcome these limitations, we propose to learn visuomotor robot skills via a Riemannian flow matching policy (RFMP). Compared to DP, RFMP builds on another kind of generative model: Flow Matching (FM) [15], [16]. Intuitively, FM gradually transforms a simple prior distribution into a complex target distribution via a vector field, which is represented by a simple function. The beauty of FM lies in its simplicity, as the resulting flow, defined by an ordinary differential equation (ODE), is much easier to train and much faster to evaluate compared to the stochastic differential equations of diffusion models. However, as many visuomotor policies are represented in the robot’s operational space, action representations must include both end-effector position and orientation. Thus, the policy must consider that orientations lie on either the $S ^ { 3 }$ hypersphere or the SO(3) Lie group, depending on the chosen parametrization. To properly handle such data, we leverage Riemannian flow matching (RFM) [16], an extension of flow matching that accounts for the geometric constraints of data lying on Riemannian manifolds. In our previous work [17], we introduced the idea of leveraging flow matching in robot imitation learning and presented RFMP, which capitalizes on the easy training and fast inference of FM methods to learn and synthesize end-effector pose trajectories. However, our initial evaluation was limited to simple proofof-concept experiments on the LASA dataset [18].

In this paper, we demonstrate the effectiveness of RFMP to learn complex real-world visuomotor policies and present a systematic evaluation of the performance of RFMP on both simulated and real-world manipulation tasks. Moreover, we propose Stable Riemannian Flow Matching Policy (SRFMP) to enhance the robustness of RFMP, as RFMP performance can be sensitive to the ODE integration horizon during inference (see Figures 1 and 2). SRFMP builds on stable flow matching (SFM) [19], [20], which leverages LaSalle’s invariance principle [21] to equip the dynamics of FM with stability to the support of the target distribution. Unlike SFM, which is limited to Euclidean spaces, SRFMP generalizes this concept to Riemannian manifolds, guaranteeing the stability of the RFM dynamics to the support of a Riemannian target distribution. We systematically evaluate RFMP and SRFMP across 10 tasks in both simulation and real-world settings, with policies conditioned on both state and visual observations. Our experiments demonstrate that RFMP and SRFMP inherit the advantages from FM models, achieving comparable performance to DP with fewer evaluation steps (i.e., faster inference) and significantly shorter training times. Moreover, under the same training epochs, our methods outperform both DP and CP. Notably, SRFMP requires fewer ODE steps than RFMP to achieve an equivalent performance, resulting in even faster inference times.

In summary, beyond demonstrating the effectiveness of our early work on simulated and real robotic tasks, the main contributions of this article are threefold: (1) We introduce Stable Riemannian Flow Matching (SRFM) as an extension of SFM [20] to incorporate stability into RFM; (2) We propose stable Riemannian flow matching policy (SRFMP), which combines the easy training and fast inference of RFMP with stability guarantees to a Riemannian target action distribution; (3) We systematically evaluate both RFMP and SRFMP across 10 tasks from simulated benchmarks and real settings. Supplementary material is available on the paper website https://sites.google.com/view/rfmp.

## II. RELATED WORK

As the literature on robot policy learning is vast, we here focus on approaches that design robot policies based on flowbased generative models.

Normalizing Flows are arguably the first broadly-used generative models in robot policy learning [22]. They were commonly employed as diffeomorphisms for learning stable dynamical systems in Euclidean spaces [23]–[25], with extensions to Lie groups [26] and Riemannian manifolds [27], similarly addressed in this paper. The main drawback of normalizing flows is their slow training, as the associated ODE needs to be integrated to calculate the model log-likelihood. Moreover, none of the foregoing works learned sensorimotor policies based on visual observations via imitation learning.

![](images/8b55e175ad9840773fd3c661b66e9b45766820c34f69af5d7f00a154496c1261.jpg)  
Fig. 2. RFMP performance when extending inference time beyond t = 1 on the Euclidean PUSHT and Robomimic SQUARE tasks.

Diffusion Models [28] recently became state-of-art in imitation learning due to their ability to learn multi-modal and high-dimensional action distributions. They have been primarily employed to learn motion planners [29], and complex control policies [2], [4], [30]. Recent extensions use 3D visual representations from sparse point clouds [6], and employ equivariant networks for learning policies that, by design, are invariant to changes in scale, rotation, and translation [7]. However, a major drawback of diffusion models is their slow inference process. In [2], DP requires 10 to 100 denoising steps, i.e., 0.1 to 1 second on a standard GPU, to generate each action. Consistency models (CM) [31] arise as a potential solution to overcome this drawback [3], [32], [33]. CM distills a student model from a pretrained diffusion model (i.e., a teacher), enabling faster inference by establishing direct connections between points along the probability path. Nevertheless, this sequential training process increases the overall complexity and time of the whole training phase.

Flow Matching [15] essentially trains a normalizing flow by regressing a conditional vector field instead of maximizing the likelihood of the model, thus avoiding to simulate the ODE of the flow. This leads to a significantly simplified training procedure compared to classical normalizing flows. Moreover, FM builds on simpler probability transfer paths than diffusion models, thus facilitating faster inference. Tong et al. [34] showed that several types of FM models can be obtained according to the choice of conditional vector field and source distributions, some of them leading to straighter probability paths, which ultimately result in faster inference. Rectified flows [35] is a similar simulation-free approach that designs the vector field by regressing against straight-line paths, thus speeding up inference. Note that rectified flows are a special case of FM, in which a Dirac distribution is associated to the probability path [34]. Due to their easy training and fast inference, FM models quickly became one of the defacto generative models in machine learning and have been employed in a plethora of different applications [36]–[41].

In our previous work [17], we proposed to leverage RFM to learn sensorimotor robot policies represented by endeffector pose trajectories on Riemannian manifolds. Building on a similar idea, subsequent works have used FM along with an equivariant transformer to learn SE(3)-equivariant policies [42], for multi-support manipulation tasks with a humanoid robot [43], and for robot imitation learning with point cloud observations [44]. In this paper, we build upon our previous work, Riemannian Flow Matching Policy (RFMP) [17], to enable the learning of complex visuomotor policies on Riemannian manifolds. Unlike the aforementioned approaches, our work focuses on providing a fast and robust RFMP inference process. We achieve this by constructing the FM vector field using LaSalle’s invariance principle, which not only enhances inference robustness with stability guarantees but also preserves the easy training and fast inference capabilities of RFMP.

## III. BACKGROUND

In this section, we provide a short background on Riemannian manifolds, and an overview of the flow matching framework with its extension to Riemannian manifolds.

## A. Riemannian Manifolds

A smooth manifold M can be intuitively understood as a d-dimensional surface that locally, but not globally, resembles the Euclidean space $\mathbb { R } ^ { d }$ [45], [46]. The geometric structure of the manifold is described via the so-called charts, which are diffeomorphic maps between parts of M and R. The collection of these charts is called an atlas. The smooth structure of $\mathcal { M }$ allows us to compute derivatives of curves on the manifolds, which are tangent vectors to $\mathcal { M }$ at a given point x. For each point ${ \textbf { \em x } } \in { \mathcal { M } } .$ the set of tangent vectors u of all curves that pass through x forms the tangent space $\mathcal { T } _ { \mathbf { \mathcal { x } } } \mathcal { M }$ . The tangent space spans a d-dimensional affine subspace of $\mathbb { R } ^ { d }$ , where d is the manifold dimension. The collections of all tangent spaces of M forms the tangent bundle $\begin{array} { r } { T \mathcal { M } = \bigcup _ { \pmb { x } \in \mathcal { M } } \{ ( \pmb { x } , \pmb { u } ) | \pmb { u } \in \mathcal { T } _ { \pmb { x } } \mathcal { M } \} } \end{array}$ , which can be thought as the union of all tangent spaces paired with their corresponding points on $\mathcal { M }$

Riemannian manifolds are smooth manifolds equipped with a smoothly-varying metric g, which is a family of inner products $g _ { \pmb { x } } \ : \ \mathcal { T } _ { \pmb { x } } \mathcal { M } \times \mathcal { T } _ { \pmb { x } } \mathcal { M } \  \ \mathbb { R }$ . The norm associated with the metric is denoted as $\| \pmb { v } \| _ { g _ { x } }$ with $v \in \mathcal { T } _ { x } \mathcal { M }$ , and the distance between two vectors ${ \pmb u } , { \pmb v } \in \mathcal { T } _ { \pmb x } \mathcal { M }$ is defined as the norm $\| \ b { u } - \ b { v } \| _ { g _ { \alpha } }$ . With this metric, we can then define the length of curves on $\mathcal { M } .$ The shortest curve on $\mathcal { M }$ connecting any two points $\pmb { x } , \pmb { y } \in \mathcal { M }$ is called a geodesic. Intuitively, geodesics can be seen as the generalization of straight lines to Riemannian manifolds. To operate with Riemannian manifolds, a common way is to exploit its Euclidean tangent spaces $\mathcal { T } _ { x } \mathcal { M }$ and back-and-forth maps between M and $\mathcal { T } _ { \mathbf { x } } \mathcal { M } , \mathrm { i . e . }$ ., the exponential and logarithmic maps. Specifically, the exponential map $\exp _ { x } ( \pmb { u } ) : \ T _ { x } \pmb { \mathcal { M } }  \ \mathcal { M }$ maps a point u on the tangent space of x to a point $\textbf { \textit { y } } \in \textbf { \textit { M } }$ so that the geodesic distance between $\begin{array} { r } { \pmb { y } \ = \ \mathrm { E x p } _ { \pmb { x } } ( \pmb { u } ) } \end{array}$ and x satisfies $d _ { g } ( { \pmb x } , { \pmb y } ) = \| { \pmb u } \| _ { g _ { \pmb x } }$ . The inverse operation is the logarithmic map $\operatorname { L o g } _ { x } ( { \pmb y } ) : \ { \mathcal { M } } \  \ { \mathcal { T } } _ { x } { \mathcal { M } }$ , which projects a point $\textbf { \em y } \in \textbf { \em M }$ to the tangent space $\mathcal { T } _ { \mathbf { \mathcal { x } } } \mathcal { M }$ of ${ \pmb x } .$ Finally, when optimizing functions of manifold-valued parameters, we need to compute the Riemannian gradient. Specifically, the Riemannian gradient of a scalar function $f : \mathcal { M }  \mathbb { R }$ at $\boldsymbol { x } \in \mathcal { M }$ is a vector in the tangent space $\mathcal { T } _ { x } \mathcal { M }$ [47], [48]. It is obtained via the identification $\mathcal { L } _ { u } f ( x ) = \langle \nabla _ { x } f ( x ) , u \rangle _ { x } ,$ where $\mathcal { L } _ { \boldsymbol { u } } f ( \boldsymbol { x } )$ denotes the directional derivative of f in the direction u $\in \ \tau _ { x } { \mathcal M }$ , and $\langle \cdot , \cdot \rangle _ { x }$ is the Riemannian inner product on $\mathcal { T } _ { \mathbf { \mathcal { x } } } \mathcal { M }$

## B. Flow Matching

Continuous normalizing flows (CNF) [49] form a class of deep generative models that transform a simple probability distribution into a more complex one. The continuous transformation of the samples is parametrized by an ODE, which describes the flow of the samples over time. Training CNF is achieved via maximum likelihood estimation, and thus involves solving (a.k.a. simulating) inverse ODEs, which is computationally expensive. Instead, flow matching (FM) [15] is a simulation-free generative model that efficiently trains CNF by directly mimicking a target vector field.

1) Euclidean Flow Matching: FM [15] reshapes a simple prior distribution $p$ into a (more complicated) target distribution $q$ via a probability density path $p _ { t }$ that satisfies $p _ { 0 } = p$ $p _ { 1 } = q$ . The path $p _ { t }$ is generated by push-forwarding p<sub>0</sub> along a flow $\psi _ { t }$ as,

$$
p _ {t} = [ \psi_ {t} ] _ {*} p _ {0},\tag{1}
$$

where the push-forward operator ∗ is defined as,

$$
[ \psi_ {t} ] _ {*} p _ {0} (\boldsymbol {x}) = p _ {0} (\psi_ {t} ^ {- 1} (\boldsymbol {x})) \det \left(\frac {\partial \psi_ {t} ^ {- 1} (\boldsymbol {x})}{\partial \boldsymbol {x}}\right).\tag{2}
$$

The flow $\psi _ { t }$ is defined via a vector field $u _ { t } : [ 0 , 1 ] \times \mathbb { R } ^ { d }  \mathbb { R } ^ { d }$ by solving the ODE,

$$
\frac {d \psi_ {t} (\boldsymbol {x})}{d t} = u _ {t} (\psi_ {t} (\boldsymbol {x})), \quad \text { with } \quad \psi_ {0} (\boldsymbol {x}) = \boldsymbol {x}.\tag{3}
$$

Assuming that both the vector field $u _ { t } ( \pmb { x } )$ and probability density path $p _ { t }$ are known, one can regress a parametrized vector field $v _ { t } ( x ; \pmb \theta ) : [ 0 , 1 ] \times \mathbb { R } ^ { d }  \mathbb { R } ^ { d }$ to some target vector field $u _ { t } .$ , which leads to the FM loss function,

$$
\ell_ {\mathrm{FM}} (\pmb {\theta}) = \mathbb {E} _ {t, p _ {t} (\pmb {x})} \| v _ {t} (\pmb {x}; \pmb {\theta}) - u _ {t} (\pmb {x}) \| _ {2} ^ {2},\tag{4}
$$

where θ are the learnable parameters, $t \sim \mathcal { U } [ 0 , 1 ] ,$ , and $\begin{array} { r } { \pmb { x } \sim p _ { t } ( \pmb { x } ) } \end{array}$ . However, the loss (4) is intractable since we actually do not have prior knowledge about $u _ { t }$ and $p _ { t }$ . Instead, Lipman et al. [15] proposed to learn a conditional vector field $u _ { t } ( \pmb { x } | \pmb { x } _ { 1 } )$ with $\scriptstyle { \mathbf { \mathscr { x } } } _ { 1 }$ as a conditioning variable. This conditional vector field generates the conditional probability density path $p _ { t } ( \pmb { x } | \pmb { x } _ { 1 } )$ , which is related to the marginal probability path via $\begin{array} { r } { p _ { t } ( { \pmb x } ) = \int p _ { t } ( { \pmb x } | { \pmb x } _ { 1 } ) q ( { \pmb x } _ { 1 } ) d { \pmb x } _ { 1 } } \end{array}$ , with $q$ being the unknown data distribution. After reparametrization, this leads to the tractable conditional flow matching (CFM) loss function,

$$
\ell_ {\mathrm{CFM}} (\pmb {\theta}) = \mathbb {E} _ {t, q (\pmb {x} _ {1}), p (\pmb {x} _ {0})} \| v _ {t} (\pmb {x} _ {t}; \pmb {\theta}) - u _ {t} (\pmb {x} _ {t} | \pmb {x} _ {1}) \| _ {2} ^ {2},\tag{5}
$$

where ${ \pmb x } _ { t } ~ = ~ \psi _ { t } ( { \pmb x } _ { 0 } | { \pmb x } _ { 1 } )$ denotes the conditional flow. Note that optimizing the CFM loss (5) is equivalent to optimizing the FM loss (4) as they have identical gradients [15]. The problem therefore boils down to design a conditional vector field $\boldsymbol { u } _ { t } ( \boldsymbol { \mathbf { x } } _ { t } | \boldsymbol { \mathbf { x } } _ { 1 } )$ that generates a probability path $p _ { t }$ satisfying the boundary conditions $p _ { 0 } = p , q = p _ { 1 }$ . Intuitively, $\boldsymbol { u } _ { t } ( \boldsymbol { \mathbf { x } } _ { t } | \boldsymbol { \mathbf { x } } _ { 1 } )$

should move a randomly-sampled point at $t = 0$ to a datapoint at $t = 1$ . Lipman et al. [15] proposed the Gaussian CFM, which defines a probability path from a zero-mean normal distribution to a Gaussian distribution centered at $\scriptstyle { \mathbf { \mathscr { x } } } _ { 1 }$ via the conditional vector field,

$$
u _ {t} (\pmb {x} _ {t} | \pmb {x} _ {1}) = \pmb {x} _ {1} - (1 - \sigma) \pmb {x} _ {0},\tag{6}
$$

which leads to the flow $\pmb { x } _ { t } = \psi _ { t } ( \pmb { x } _ { 0 } | \pmb { x } _ { 1 } ) = ( 1 - ( 1 - \sigma ) t ) \pmb { x } _ { 0 } +$ $t { \pmb x } _ { 1 }$ . Note that a more general version of CFM is proposed by Tong et al. [34].

Finally, the inference process of CFM is straightforward and consists of the following steps: (1) Get a sample from $p _ { 0 } ;$ and (2) Query the learned vector field $v _ { t } ( x ; \pmb \theta )$ to solve the ODE (3) with off-the-shelf solvers, $\mathrm { e . g . }$ ., based on the Euler method [50].

2) Riemannian Flow Matching: In many robotics settings, data lies on Riemannian manifolds [51], [52]. For example, various tasks involve the rotation of the robot’s end-effector. Therefore, the corresponding part of the state representation lies either on the Riemannian hypersphere $S ^ { \dot { 3 } }$ or the SO(3) group, depending on the choice of parametrization. To guarantee that the FM generative process satisfies manifold constraints, Chen and Lipman [16] extended CFM to Riemannian manifolds. The Riemannian conditional flow matching (RCFM) considers that the flow $\psi _ { t }$ evolves on a Riemannian manifold M. Thus, for each point ${ \pmb x } \in { \mathcal { M } } .$ , the vector field associated to the flow $\psi _ { t }$ at this point lies on the tangent space of x, i.e., $u _ { t } ( \pmb { x } ) \in \mathcal { T } _ { \pmb { x } } \mathcal { M }$ . The RCFM loss function resembles that of the CFM model but it is computed with respect to the Riemannian metric $g _ { x }$ as follows,

$$
\ell_ {\mathrm{RCFM}} = \mathbb {E} _ {t, q (\boldsymbol {x} _ {1}), p (\boldsymbol {x} _ {0})} \| v _ {t} (\boldsymbol {x} _ {t}; \boldsymbol {\theta}) - u _ {t} (\boldsymbol {x} _ {t} | \boldsymbol {x} _ {1}) \| _ {g _ {\boldsymbol {x} _ {t}}} ^ {2}.\tag{7}
$$

As in the Euclidean case, we need to design the flow $\psi _ { t } ,$ its corresponding conditional vector field $\boldsymbol { u } _ { t } ( \boldsymbol { x } _ { t } | \boldsymbol { x } _ { 1 } )$ , and choose the base distribution. Following [16], [36], the most straightforward strategy is to exploit geodesic paths to design the flow $\psi _ { t }$ . For simple Riemannian manifolds such as the hypersphere, the hyperbolic manifold, and some matrix Lie groups, geodesics can be computed via closed-form solutions. We can then leverage the geodesic flow given by,

$$
\boldsymbol {x} _ {t} = \operatorname{Exp} _ {\boldsymbol {x} _ {1}} (t \operatorname{Log} _ {\boldsymbol {x} _ {1}} (\boldsymbol {x} _ {0})), \quad t \in [ 0, 1 ].\tag{8}
$$

The conditional vector field can then be calculated as the time derivative of $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ , i.e., $u _ { t } ( { \pmb x } _ { t } | { \pmb x } _ { 1 } ) = \dot { { \pmb x } } _ { t }$ . Notice that $u _ { t }$ boils down to the conditional vector field (6) with $\sigma = 0$ when $\mathcal { M } = \mathbb { R } ^ { d }$ . Chen and Lipman [16] also provide a general formulation of $\boldsymbol { u } _ { t } ( \boldsymbol { \mathbf { x } } _ { t } | \boldsymbol { \mathbf { x } } _ { 1 } )$ for cases where closed-form geodesics are not available. The prior distribution $p _ { 0 }$ can be chosen as a uniform distribution on the manifold [16], [36], or as a Riemannian [53] or wrapped Gaussian [16], [54] distribution on M. During inference, we solve the corresponding RCFM’s ODE on the Riemannian manifold M via projection-based methods. Specifically, at each step, the integration is performed in the tangent space $\mathcal { T } _ { \mathbf { \mathcal { x } } } \mathcal { M }$ and the resulting vector is projected onto the Riemannian manifold M with the exponential map.

## C. Flow Matching vs. Diffusion and Consistency Models

Diffusion Policy (DP) [2] is primarily trained based on DDPM [12], which performs iterative denoising from an initial noise sample $\scriptstyle { \pmb { x } } _ { K }$ , where K denotes the total number of denoising steps. The denoising process follows,

$$
\boldsymbol {x} _ {k - 1} = \alpha_ {t} \big (\boldsymbol {x} _ {k} - \gamma \epsilon_ {\boldsymbol {\theta}} (\boldsymbol {x} _ {k}, k) + \mathcal {N} (0, \sigma^ {2} \boldsymbol {I}) \big),\tag{9}
$$

where $\alpha , \gamma$ and $\sigma$ constitute the noise schedule that governs the denoising process, and $\epsilon _ { \theta } ( \pmb { x } _ { k } , k )$ is the noise prediction network that infers the noise at the k-th denoising step. The final sample $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$ is the noise-free target output. The equivalent inference process in FM is governed by,

$$
\frac {d \psi_ {t} (\pmb {x})}{d t} = v _ {t} (\psi_ {t} (\pmb {x}); \pmb {\theta}), \quad \mathrm{with} \quad \psi_ {0} (\pmb {x}) = \pmb {x},\tag{10}
$$

where ${ v _ { t } } ( \psi _ { t } ( \pmb { x } ) ; \theta )$ is the learned vector field that mimics the target forward process (6). Two key differences can be identified: (1) FM requires to solve a simple ODE, and (2) The FM vector field induces straighter paths. Importantly, the DP denoising framework limits its inference efficiency, particularly in scenarios requiring faster predictions.

Consistency Policy (CP) [3] aims to speed up the inference process of DP by leveraging a consistency model that distills the DP as a teacher model. While CP adopts a similar denoising mechanism as DP, it enhances the process by incorporating both the current denoising step k and the target denoising step t as inputs to the denoising network, formalized as $\boldsymbol { \epsilon } _ { \theta } ( \boldsymbol { x } _ { k } , k , t )$ However, CP involves a two-stage training process. First, a DP teacher policy is trained. Next, a student policy is trained to mimic the denoising process of the teacher policy. This approach enables CP to achieve faster and more efficient inference while retaining the performance of its teacher model, at the cost of a more complex training process. In contrast, FM training is simulation-free, and features a single-phase training with a simple loss function.

## IV. FAST AND ROBUST RFMP

Our goal is to leverage the RCFM framework to learn a parameterized policy $\pi _ { \pmb { \theta } } ( \pmb { a } | \mathbf { o } )$ that adheres to the target (expert) policy $\pi _ { e } ( { \pmb a } | { \pmb o } )$ , which generates a set of N demonstrations $D _ { n } = \{ o ^ { s } , \pmb { a } ^ { s } \} _ { s = 1 } ^ { T }$ , where o denotes an observation, a represents the corresponding action, and T denotes the length of n-th trajectory. In this section, we first introduce Riemaniann flow matching polices (RFMP) that leverage RCFM to achieve easy training and fast inference. Second, we propose Stable RFMP (SRFMP), an extension of RFMP that enhances its robustness and inference speed through stability to the target distribution.

## A. Riemannian Flow Matching Policy

RFMP adapts RCFM to visuomotor robot policies by learning an observation-conditioned vector field $u _ { t } ( \pmb { a } | \pmb { o } )$ . Similar to DP [2], RFMP employs a receding horizon control strategy [55], by predicting a sequence of actions over a prediction horizon $T _ { p }$ . This strategy aims at providing temporal consistency and smoothness on the predicted actions. This means that the predicted action horizon vector is constructed as $\pmb { a } = [ \pmb { a } ^ { s } , \pmb { a } ^ { s + 1 } , \dots , \pmb { a } ^ { s + T _ { p } } ]$ , where $\mathbf { \Delta } _ { \mathbf { \pmb { a } } ^ { i } }$ is the action at time step $i ,$ and $T _ { p }$ is the action prediction horizon. This implies that all samples $\mathbf { \delta } _ { a _ { 1 } }$ , drawn from the target distribution $p _ { 1 }$ , have the form of the action horizon vector a. Moreover, we define the base distribution $p _ { 0 }$ such that samples $a _ { p _ { 0 } } \sim p _ { 0 }$ are of the form ${ \pmb a } _ { 0 } = [ { \pmb a } _ { b } , \ldots , { \pmb a } _ { b } ]$ with $\mathbf { \delta } _  \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathrm { \bf \~ \textit ~ { ~ a ~ } } _ { b }$ sampled from an auxiliary distribution b. This structure contributes to the smoothness of the predicted action vector a, as the flow of all its action components start from the same initial action $\mathbf { \delta } _  \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathrm { \bf ~ \textit ~ { ~ a ~ } } _ { b }$

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1: RFMP Training &amp; Inference

1 Training
Input: Initial parameters $\theta$, prior and target distribution $p_0, p_1$.
Output: Learned vector field parameters $\theta$.

2 while termination condition unsatisfied do
3    Sample flow time step $t$ from the uniform distribution $\mathcal{U}[0,1]$.
4    Sample noise $a_0 \sim p_0$.
5    Jointly sample action sequence $a_1 \sim p_1$ and corresponding observation vector o.
6    Compute conditional vector field $u_t(a_t|a_1)$ via the RCFM geodesic flow (8).
7    Evaluate $\ell_{\text{RFMP}}$ as in (11).
8    Update parameters $\theta$.

9 Inference Step
Input: Predefined number of function evaluation $N$, learned vector field $v_\theta$, observation vector o, prior distribution $p_0$.
10 Sample $a_0 \sim p_0$, and set $t = 0$.
11 while $t \leq 1$ do
12    Integrate the learned Riemannian vector field $\xi_{t+\Delta t} = \text{Exp}_{\xi_t}(v_\theta(\xi_t, o)\Delta t)$.
13    Update time $t = t + \Delta t$.
</div>

In contrast to the action horizon vector, the observation vector o is not defined on a receding horizon but is constructed by randomly sampling only few observation vectors. Specifically, RFMP follows the sampling strategy proposed in [37] which uses: (1) A reference observation $o ^ { s - 1 }$ at time step $s - 1 ; ( 2 )$ A context observation $\pmb { o } ^ { c }$ randomly sampled from an observation window with horizon $T _ { o } ,$ i.e., c is uniformly sampled from $\left[ s - T _ { o } , s - 2 \right]$ ; and (3) The time gap s−c between the context observation and reference observation. Therefore, the observation vector is defined as $\textbf { \em o } = \ [ \pmb { o } ^ { s - 1 } , \pmb { o } ^ { c } , s - c ]$ Notice that, when $T _ { o } = 2 .$ , we disregard the time gap and the observation is $\pmb { \ o } = [ \pmb { o } ^ { s - 1 } , \pmb { o } ^ { c } ]$ . The aforementioned strategy leads to the following RFMP loss function,

$$
\ell_ {\mathrm{RFMP}} = \mathbb {E} _ {t, q (\boldsymbol {a} _ {1}), p (\boldsymbol {a} _ {0})} \| v _ {t} (\boldsymbol {a} _ {t} | \boldsymbol {o}; \boldsymbol {\theta}) - u _ {t} (\boldsymbol {a} _ {t} | \boldsymbol {a} _ {1}) \| _ {g _ {\boldsymbol {a} _ {t}}} ^ {2}.\tag{11}
$$

Algorithm 1 summarizes the training process of RFMP. Note that our RFMP inherits most of the training framework of RCFM, the main difference being that the vector field learned in RFMP is conditioned on the observation vector o. After training RFMP, the inference process, which essentially queries the policy $\pi _ { \pmb { \theta } } ( \pmb { a } | \pmb { o } )$ , boils down to the following four steps: (1) Draw a sample $\mathbf { \delta } _ { \mathbf { a } _ { 0 } }$ from the prior distribution $p _ { 0 } ;$ (2) Construct the observation vector $\mathbf { \sigma } _ { o ; \mathbf { \beta } } \left( 3 \right)$ Employ an offthe-shelf ODE solver to integrate the learned vector field $v _ { t } ( a | o ; \theta )$ from $\mathbf { \delta } _ { \mathbf { a } _ { 0 } }$ along the time interval $t = \left\lceil 0 , 1 \right\rceil$ , and get the generated action sequence $\mathbf { \delta } \mathbf { a } = [ \mathbf { a } ^ { s } , . . . , \mathbf { a } ^ { s \bar { + } T _ { p } } ] ;$ and (4) Execute the first $T _ { a }$ actions of the sequence a with $T _ { a } \leq T _ { p }$ This last step allows the robot to quickly react to environment changes, while still providing smooth predicted actions.

Although RFM is theoretically designed to match a timedependent vector field defined over the time horizon $t \in [ 0 , 1 ]$ we observed that, in practice, its inference performance often improves when evaluating trajectories slightly beyond $t = 1$ To investigate this phenomenon, we conduct experiments on two tasks: The Euclidean PUSH-T task from [2] and the SQUARE task from the Robomimic robotic manipulation benchmark [10]. As shown in Figure 2, RFMP performance improves when the ODE integration horizon slightly exceeds $t = 1$ , but it worsens and eventually gets unstable when integrating for longer time horizons. This observation motivates the development of SRFMP introduced next, which enhances RFMP by explicitly enforcing inference stability beyond the unit interval using the LaSalle’s invariance principle. This approach stabilizes the policy inference, making it robust to the ODE integration horizon, as illustrated in Figures 1, 3 and 4.

## B. Stable Riemannian Flow Matching Policy

Both CFM and RCFM train and integrate the learned vector field $v _ { t } ( x | \pmb \theta )$ within the interval $t \ : = \ : [ 0 , 1 ]$ . However, they do not guarantee that the flow converges stably to the target distribution at t = 1. Besides, the associated vector field may even display strongly diverging behaviors when going beyond this upper boundary [20]. The aforementioned issues may arise due to numerical inaccuracies when training or integrating the vector field. To solve this problem, Sprague et al. [20] proposed Stable Autonomous Flow Matching (SFM), which equips the dynamics of FM with stability to the support of the target distribution. Here, we propose to improve RFMP with SFM, which we generalize to the Riemannian case, in order to guarantee that the flow stabilizes to the target policy at $t = 1$ Our experiments show that this approach not only enhances RFMP’s robustness but also further reduces inference time.

1) Stable Euclidean FMP: We first summarize the Euclidean SFM [20], and then show how SFM can be integrated into RFMP. SFM leverages the stochastic LaSalle’s invariance principle [56], [57] — a criterion from control theory used to characterize the asymptotic stability of stochastic autonomous dynamical systems — to design a stable vector field u. Sprague et al. [20, Thm 3.8] adapts this principle to the FM setting as follows.

Theorem 1. (Stochastic LaSalle’s Invariance Principle) If there exists a time-independent vector field u, a flow ψ generated by u, and a positive scalar function H such that,

$$
\mathcal {L} _ {\boldsymbol {u}} H (\boldsymbol {x}) = \nabla_ {\boldsymbol {x}} H (\boldsymbol {x}) u (\boldsymbol {x}) \leq 0,\tag{12}
$$

where $\mathcal { L } _ { \boldsymbol { u } } H ( \boldsymbol { x } )$ is the directional derivative of the scalar function H in the direction u and $\nabla _ { x } H$ is the gradient of

![](images/440301d65329a55e1694cf748d106edf879016b1847ba13a372fa7708f44ee72.jpg)  
Fig. 3. Flows of the RFM (top) and SRFM (bottom) trained on the L-shape LASA dataset projected on the sphere manifold. Orange points represent the training dataset, while blue points are sampled from the generated probability path at different times $t = \left\{ 0 . 0 , 1 . 0 , 1 . 5 \right\}$ across the three columns.  
Fig. 4. Flows of the RFM (top) and SRFM (bottom) on the SPD manifold $S _ { + + } ^ { 2 } .$ . Orange points represent the training dataset, while blue points correspond to sampled from the generated probability path at different times $t = \{ \bar { 0 } . 0 , 1 . 0 , 1 . 5 \}$ across the three columns.

the function H, then,

$$
\lim _ {t \to \infty} \psi (\boldsymbol {x}, t) \in \{\boldsymbol {x} \in \mathcal {X} | \mathcal {L} _ {u} H (\boldsymbol {x}) = 0 \},
$$

almost surely with $\pmb { x } \sim p ( \pmb { x } , 0 )$

Intuitively, Theorem 1 provides conditions for convergence to an invariant set even when $\mathcal { L } _ { u } H ( { \boldsymbol { x } } )$ is not strictly negative, therefore accounting for stochastic fluctuations in the system. Theorem 1 notably holds if $u ( { \pmb x } )$ is a gradient field of H, i.e., if $u ( { \pmb x } ) = - \nabla _ { \pmb x } H ( { \pmb x } ) ^ { \top }$ . In this case, the problem of finding a stable vector field boils down to defining an appropriate scalar function H. As LaSalle’s invariance principle requires an autonomous, a.k.a time-independent, vector field, Sprague et al. [20] augment the FM state space x with an additional dimension τ, called temperature or pseudo time, so that the SFM augmented state space becomes ${ \pmb \xi } = [ { \pmb x } , \tau ]$ . The pair (H, ξ) is then defined so that it satisfies (12) as,

$$
H (\boldsymbol {\xi} | \boldsymbol {\xi} _ {1}) = \frac {1}{2} (\boldsymbol {\xi} - \boldsymbol {\xi} _ {1}) ^ {\top} \boldsymbol {A} (\boldsymbol {\xi} - \boldsymbol {\xi} _ {1}),\tag{13}
$$

$$
u (\boldsymbol {\xi} | \boldsymbol {\xi} _ {1}) = - \nabla_ {\boldsymbol {\xi}} H (\boldsymbol {\xi} | \boldsymbol {\xi} _ {1}) ^ {\top} = - \boldsymbol {A} (\boldsymbol {\xi} - \boldsymbol {\xi} _ {1}),\tag{14}
$$

where A is a positive-define matrix. To simplify the calculation, A is set as the diagonal matrix,

$$
\boldsymbol {A} = \left[ \begin{array}{c c} \lambda_ {\boldsymbol {x}} \boldsymbol {I} & \boldsymbol {0} \\ \boldsymbol {0} & \lambda_ {\tau} \end{array} \right],\tag{15}
$$

with $\lambda _ { x } , \lambda _ { \tau } \in \mathbb { R }$ . The vector field u and corresponding stable flow $\psi _ { t }$ are then given as,

$$
u (\boldsymbol {\xi} _ {t} | \boldsymbol {\xi} _ {1}) = \left[ \begin{array}{c} u _ {\boldsymbol {x}} (\boldsymbol {x} _ {t} | \boldsymbol {x} _ {1}) \\ u _ {\tau} (\tau_ {t} | \tau_ {1}) \end{array} \right] = \left[ \begin{array}{c} - \lambda_ {\boldsymbol {x}} (\boldsymbol {x} _ {t} - \boldsymbol {x} _ {1}) \\ - \lambda_ {\tau} (\tau_ {t} - \tau_ {1}) \end{array} \right],\tag{16}
$$

$$
\psi_ {t} (\boldsymbol {\xi} _ {0} | \boldsymbol {\xi} _ {1}) = \left[ \begin{array}{c} \psi_ {t} (\boldsymbol {x} _ {0} | \boldsymbol {x} _ {1}) \\ \psi_ {t} (\tau_ {0} | \tau_ {1}) \end{array} \right] = \left[ \begin{array}{c} \boldsymbol {x} _ {1} + e ^ {- \lambda_ {\boldsymbol {x}} t} (\boldsymbol {x} _ {0} - \boldsymbol {x} _ {1}) \\ \tau_ {1} + e ^ {- \lambda_ {\tau} t} (\tau_ {0} - \tau_ {1}) \end{array} \right],\tag{17}
$$

The parameters $\tau _ { 1 }$ and $\tau _ { 0 }$ define the range of the τ flow, while the parameters $\lambda _ { x }$ and $\lambda _ { \tau }$ determine its convergence. Specifically, the flow converges faster for higher values of $\lambda _ { x }$ and $\lambda _ { \tau }$ . Moreover, the ratio between $\lambda _ { x } / \lambda .$ determines the relative rate of convergence of the spatial and pseudo-time parts of the flow. We ablate the influence of $\lambda _ { x }$ and λ<sub>τ</sub> on SRFMP in Section V-B.

We integrate SFM to RFMP by regressing an observationconditioned vector field $v ( \xi | o ; \theta )$ to a stable target vector field $u ( \pmb { \xi } _ { t } | \pmb { \xi } _ { 1 } )$ , where $\pmb { \xi } = [ \pmb { a } ^ { s } , . . . , \pmb { a } ^ { s + T _ { p } } , \tau ]$ is the augmented prediction horizon vector, and o is the observation vector.

2) Stable Riemannian FMP: Next, we introduce our extension, stable Riemannian flow matching (SRFM), which generalizes SFM [20] to Riemannian manifolds. Similarly as SFM, we define a time-invariant vector field u by augmenting the state space with the pseudo-time state τ, so that ${ \pmb \xi } = [ { \pmb x } , \tau ]$ Notice that, in this case, $\pmb { x } \in \mathcal { M }$ and thus ξ lies on the product of Riemannian manifolds $\mathcal { M } \times \mathbb { R }$ . Importantly, Theorem 1 also holds for Riemannian autonomous systems, in which case $\nabla _ { \pmb { x } } H ( \pmb { x } )$ denotes the Riemannian gradient of the positive scalar function H. We formulate H so that the pair (H, ξ) satisfies (12) as,

$$
H (\boldsymbol {\xi} | \boldsymbol {\xi} _ {1}) = \frac {1}{2} \operatorname{Log} _ {\boldsymbol {\xi} _ {1}} (\boldsymbol {\xi}) ^ {\top} A \operatorname{Log} _ {\boldsymbol {\xi} _ {1}} (\boldsymbol {\xi}),\tag{18}
$$

which leads to the Riemannian vector field,

$$
u (\boldsymbol {\xi} | \boldsymbol {\xi} _ {1}) = - \nabla_ {\boldsymbol {\xi}} H (\boldsymbol {\xi} | \boldsymbol {\xi} _ {1}) ^ {\top} = - A \mathrm{Log} _ {\boldsymbol {\xi} _ {1}} (\boldsymbol {\xi}).\tag{19}
$$

By setting the positive-definite matrix A as in (15), we obtain the Riemannian vector field,

$$
u (\boldsymbol {\xi} _ {t} | \boldsymbol {\xi} _ {1}) = \left[ \begin{array}{c} u _ {\boldsymbol {x}} (\boldsymbol {x} _ {t} | \boldsymbol {x} _ {1}) \\ u _ {\tau} (\tau_ {t} | \tau_ {1}) \end{array} \right] = \left[ \begin{array}{c} - \lambda_ {\boldsymbol {x}} \mathrm{Log} _ {\boldsymbol {x} _ {1}} (\boldsymbol {x} _ {t}) \\ - \lambda_ {\tau} (\tau_ {t} - \tau_ {1}) \end{array} \right],\tag{20}
$$

which generates the stable Riemannian flow,

$$
\psi_ {t} (\boldsymbol {\xi} _ {0} | \boldsymbol {\xi} _ {1}) = \left[ \begin{array}{c} \psi_ {t} (\boldsymbol {x} _ {0} | \boldsymbol {x} _ {1}) \\ \psi_ {t} (\tau_ {0} | \tau_ {1}) \end{array} \right] = \left[ \begin{array}{c} \operatorname{Exp} _ {\boldsymbol {x} _ {1}} \big (e ^ {- \lambda_ {\boldsymbol {x}} t} \operatorname{Log} _ {\boldsymbol {x} _ {1}} (\boldsymbol {x} _ {0}) \big) \\ \tau_ {1} + e ^ {- \lambda_ {\tau} t} (\tau_ {0} - \tau_ {1}) \end{array} \right].\tag{21}
$$

The parameters $\lambda _ { x }$ and $\lambda _ { \tau }$ have the same influence as in the Euclidean case. Notice that the spatial part of the stable flow (21) closely resembles the geodesic flow (8) proposed in [16]. Figures 3 and 4 show examples of learned RFM and SRFM flows at times $t ~ = ~ \{ 0 , 1 , 1 . 5 \}$ on the sphere and symmetric positive-definite (SPD) matrices manifold. The RFM flow diverges from the target distribution at time $t > 1$ while the SRFM flow is stable and adheres to the target distribution for $t \geq 1$

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 2: SRFMP Training &amp; Inference

1 Training
Input: Initial parameters $\theta$, prior and target distributions $p_0, p_1$.
Output: Learned vector field parameters $\theta$.

2 while termination condition unsatisfied do
3    Sample flow time step $t$ from an uniform distribution $\mathcal{U}[0,1]$.
4    Sample noise $a_0 \sim p_0$.
5    Jointly sample action sequence $a_1 \sim p_1$ and corresponding observation vector $o$.
6    Form the vectors $\xi_1 = [a_1, \tau_1]$ and $\xi_0 = [a_0, \tau_0]$.
7    Compute conditional vector field $u_t(\xi_t|\xi_1)$ via (14).
8    Evaluate the loss $\ell_{\text{SRFMP}}$ as defined in (22).
9    Update parameters $\theta$.

10 Inference Step
Input: Predefined number of function evaluation $N$, learned vector field $v_\theta$, observation vector $o$, prior distribution $p_0$.
11 Sample $a_0 \sim p_0$, and set $k = 1$, $t = 0$, $\xi_0 = [a_0, \tau_0]$.
12 while $k \leq N$ do
13    if $k = 1$ then
14    $\Delta t = \frac{1}{\lambda_x}$
15    else
16    $\Delta t = \varepsilon \leq \frac{1}{\lambda_x}$
17    Integrate the learned Riemannian vector field $\xi_{t+\Delta t} = \text{Exp}_{\xi_t}(v_\theta(\xi_t, o)\Delta t)$.
18    Update time $t = t + \Delta t$.
19    Update iteration $k = k + 1$.
</div>

Finally, the process to induce this stable behavior into the RFMP flow involves two main changes: (1) We define an augmented action horizon vector $\pmb { \xi } ~ = ~ \left[ \pmb { a } ^ { s } , \ldots , \pmb { a } ^ { s + T _ { p } } , \tau \right]$ and; (2) We regress the observation-conditioned Riemannian vector field $v ( \xi | o ; \theta )$ against the stable Riemannian vector field $u ( \pmb { \xi } | \pmb { \xi } _ { 1 } )$ defined in (19), where o denotes the observation vector. The model is then trained to minimize the SRFMP loss,

$$
\ell_ {\mathrm{SRFMP}} = \mathbb {E} _ {t, q (\boldsymbol {a} _ {1}), p (\boldsymbol {a} _ {0})} \| v (\boldsymbol {\xi} _ {t} | \boldsymbol {o}; \boldsymbol {\theta}) - u (\boldsymbol {\xi} _ {t} | \boldsymbol {\xi} _ {1}) \| _ {g _ {\boldsymbol {a} _ {t}}} ^ {2}.\tag{22}
$$

This approach, hereinafter referred to as stable Riemannian flow matching policy (SRFMP), is summarized in Algorithm 2. The learned SRFMP vector field drives the flow to converge to the target distribution within a certain time horizon, ensuring that it remains within this distribution, as illustrated by the bottom row of Figures 1, 3, and 4 . In contrast, the RFMP vector field may drift the flow away from the target distribution at $t > 1$ (see the top row of Figures 1, 3, and 4). Therefore, SRFMPs provide flexibility and increased robustness in designing the generation process, while RFMPs are more sensitive to the integration process.

3) Solving the SRFMP ODE: As previously discussed, querying SRFMP policies involves integrating the learned vector field along the time interval $t ~ = ~ [ 0 , T ]$ with time boundary T. To do so, we use the projected Euler method, which integrates the vector field on the tangent space for one Euler step and then projects the resulting vector onto the manifold M. Assuming an Euclidean setting and that $v ( \xi | o ; \theta )$ is perfectly learned, this corresponds to recursively applying,

$$
\boldsymbol {x} _ {t + 1} = \boldsymbol {x} _ {t} + v _ {\boldsymbol {x}} (\boldsymbol {x} _ {t} | \boldsymbol {o}; \boldsymbol {\theta}) \Delta t \approx \boldsymbol {x} _ {t} + \lambda_ {\boldsymbol {x}} (\boldsymbol {x} _ {1} - \boldsymbol {x} _ {t}) \Delta t,\tag{23}
$$

with time step $\Delta t$ and $v _ { x }$ following the same partitioning as $u _ { x }$ in (20). The time step $\Delta t$ is typically set as $\Delta t = T / N$ where N is the total number of ODE steps. Here we propose to leverage the structure of SRFMP to choose the time step $\Delta t$ in order to further speed up the inference time of RFMP. Specifically, we observe that the recursion (23) leads to,

$$
\pmb {x} _ {t} = (1 - \lambda_ {\pmb {x}} \Delta t) ^ {n} (\pmb {x} _ {0} - \pmb {x} _ {1}) + \pmb {x} _ {1},\tag{24}
$$

after n time steps. It is easy to see that $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ converges to $\scriptstyle { \mathbf { { \vec { x } } } } _ { 1 }$ after a single time step when setting $\Delta t = 1 / { \lambda _ { x } }$

In the Riemannian case, assuming that $v ( \xi | o ; \theta )$ approximately equals the Riemannian vector field (20), we obtain,

$$
\boldsymbol {x} _ {t + 1} = \operatorname{Exp} _ {\boldsymbol {x} _ {t}} \left(v _ {\boldsymbol {x}} (\boldsymbol {x} | \boldsymbol {o}; \boldsymbol {\theta}) \Delta t\right) \approx \operatorname{Exp} _ {\boldsymbol {x} _ {t}} \left(\lambda_ {\boldsymbol {x}} \operatorname{Log} _ {\boldsymbol {x} _ {t}} (\boldsymbol {x} _ {1}) \Delta t\right).\tag{25}
$$

Similarly, it is easy to see that the Riemannian flow converges to $\scriptstyle { \mathbf { { \vec { x } } } } _ { 1 }$ after a single time step for $\Delta t = 1 / \lambda _ { x } .$ . Importantly, this strategy assumes that the learned vector field is perfectly learned and thus equals the target vector field. However, this is often not the case in practice. However, our experiments show that the flow obtained solving the SRFMP ODE with a single time step $\Delta t = 1 / { \lambda _ { x } }$ generally leads to the target distribution. In practice, we set $\Delta t$ to $1 / \lambda _ { x }$ for the first time step, and to a smaller value afterwards for refining the flow.

## V. EXPERIMENTS

We thoroughly evaluate the performance of RFMP and SRFMP on a set of eight simulation settings and two realworld tasks. The simulated benchmarks are: $( l )$ The PUSH-T task from [2]; (2) A SPHERE PUSH-T task, which we introduce as a Riemannian benchmark; (3)-(7) Five tasks (LIFT, CAN, SQUARE, TOOL HANG, and TRANSPORT) from the largescale robot manipulation benchmark Robomimic [10]; and (8) the FRANKA KITCHEN benchmark [58] featuring complex, long-horizon tasks. The real-world robot tasks correspond to: (1) A PICK & PLACE task; and (2) A MUG FLIPPING task. Collectively, these ten tasks serve as a benchmark to evaluate (1) the performance, (2) the training time, and (3) the inference time of RFMP and SRFMP with respect to state-of-the-art generative policies, i.e., DP and extensions thereof.

## A. Implementation Details

To establish a consistent experimental framework, we first introduce the neural network architectures employed in RFMP and SRFMP across all tasks. We then describe the considered baselines, and our overall evaluation methodology.

TABLE I  
HYPERPARAMETERS FOR ALL EXPERIMENTS: RESOLUTION OF ORIGINAL AND CROPPED IMAGE, NUMBER OF PARAMETERS FOR THE LEARNED VECTOR FIELD (VF) USING RFMP AND SRFMP, NUMBER OF RESNET PARAMETERS, TRAINING EPOCHS, AND BATCH SIZE.

<table><tr><td>Experiment</td><td>Image res.</td><td>Crop res.</td><td>RFMP VF # params</td><td>SRFMP VF # params</td><td>ResNet # params</td><td>Epochs</td><td>Batch size</td></tr><tr><td colspan="8">PUSH-T Tasks</td></tr><tr><td>Euclidean PUSH-T</td><td>96 × 96</td><td>84 × 84</td><td> $8.0 \times 10^{7}$ </td><td> $8.14 \times 10^{7}$ </td><td> $1.12 \times 10^{7}$ </td><td>300</td><td>256</td></tr><tr><td>SPHERE PUSH-T</td><td>100 × 100</td><td>84 × 84</td><td> $8.0 \times 10^{7}$ </td><td> $8.14 \times 10^{7}$ </td><td> $1.12 \times 10^{7}$ </td><td>300</td><td>256</td></tr><tr><td colspan="8">State-based simulation tasks</td></tr><tr><td>Robomimic LIFT</td><td>N.A.</td><td>N.A.</td><td> $6.58 \times 10^{7}$ </td><td> $6.68 \times 10^{7}$ </td><td>N.A.</td><td>50</td><td>256</td></tr><tr><td>Robomimic CAN</td><td>N.A.</td><td>N.A.</td><td> $6.58 \times 10^{7}$ </td><td> $6.68 \times 10^{7}$ </td><td>N.A.</td><td>50</td><td>256</td></tr><tr><td>Robomimic SQUARE</td><td>N.A.</td><td>N.A.</td><td> $6.58 \times 10^{7}$ </td><td> $6.68 \times 10^{7}$ </td><td>N.A.</td><td>50</td><td>512</td></tr><tr><td>Robomimic TOOL HANG</td><td>N.A.</td><td>N.A.</td><td> $6.58 \times 10^{7}$ </td><td> $6.68 \times 10^{7}$ </td><td>N.A.</td><td>100</td><td>512</td></tr><tr><td>FRANKA KITCHEN</td><td>N.A.</td><td>N.A.</td><td> $6.69 \times 10^{7}$ </td><td> $6.89 \times 10^{7}$ </td><td>N.A.</td><td>500</td><td>128</td></tr><tr><td colspan="8">Vision-based simulation tasks</td></tr><tr><td>Robomimic LIFT</td><td>2 × 84 × 84</td><td>2 × 76 × 76</td><td> $9.48 \times 10^{7}$ </td><td> $9.69 \times 10^{7}$ </td><td> $2 \times 1.12 \times 10^{7}$ </td><td>100</td><td>256</td></tr><tr><td>Robomimic CAN</td><td>2 × 84 × 84</td><td>2 × 76 × 76</td><td> $9.48 \times 10^{7}$ </td><td> $9.69 \times 10^{7}$ </td><td> $2 \times 1.12 \times 10^{7}$ </td><td>100</td><td>256</td></tr><tr><td>Robomimic SQUARE</td><td>2 × 84 × 84</td><td>2 × 76 × 76</td><td> $9.48 \times 10^{7}$ </td><td> $9.69 \times 10^{7}$ </td><td> $2 \times 1.12 \times 10^{7}$ </td><td>100</td><td>512</td></tr><tr><td>Robomimic TRANSPORT</td><td>4 × 84 × 84</td><td>4 × 76 × 76</td><td> $1.24 \times 10^{8}$ </td><td> $1.27 \times 10^{8}$ </td><td> $4 \times 1.12 \times 10^{7}$ </td><td>200</td><td>256</td></tr><tr><td colspan="8">Real-world experiments</td></tr><tr><td>PICK &amp; PLACE</td><td>320 × 240</td><td>288 × 216</td><td> $2.51 \times 10^{7}$ </td><td> $2.66 \times 10^{7}$ </td><td> $1.12 \times 10^{7}$ </td><td>300</td><td>256</td></tr><tr><td>MUG FLIPPING</td><td>320 × 240</td><td>256 × 192</td><td> $2.51 \times 10^{7}$ </td><td> $2.66 \times 10^{7}$ </td><td> $1.12 \times 10^{7}$ </td><td>300</td><td>256</td></tr></table>

1) RFMP and SRFMP Implementation: Our RFMP implementation builds on the RFM framework from Chen and Lipman [16]. We parameterize the vector field $v _ { t } ( a | o ; \theta )$ using the UNet architecture employed in DP [2], which consists of 3 layers with downsampling dimensions of (256, 512, 1024) and (128, 256, 512) for simulated and real-world tasks. Each layer employs a 1-dimensional convolutional residual network as proposed in [29]. We implement a Feature-wise Linear Modulation (FiLM) [59] to incorporate the observation condition vector o and time step t into the UNet. Instead of directly feeding the FM time step t as a conditional variable, we first project it into a higher-dimensional space using a sinusoidal embedding module, similarly to DP. For tasks with image-based observations, we leverage the same vision perception backbone as in DP [2]. Namely, we use a standard ResNet-18 in which we replace: (1) the global average pooling with a spatial softmax pooling, and (2) BatchNorm with GroupNorm. Our SRFMP implementation builds on the SFM framework [20]. We implement the same UNet as RFMP to represent $v _ { x }$ by replacing the time step t by the temperature parameter τ. We introduce an additional Multi-Layer Perceptron (MLP) to learn $v _ { \tau } .$ As for t in RFMP, we employed a sinusoidal embedding for the input $\tau .$ The boundaries $\tau _ { 0 }$ and $\tau _ { 1 }$ are set to 0 and 1 in all experiments.

We implement different prior distributions for different tasks. For the Euclidean PUSH-T and the Robomimic tasks, the action space is $\mathbb { R } ^ { d }$ and we thus define the prior distribution as a Euclidean Gaussian distribution for both RFMP and SRFMP. For the SPHERE PUSH-T, the action space is the hypersphere $S ^ { 2 }$ . In this case, we test two types of Riemannian prior distribution, namely a spherical uniform distribution, and a wrapped Gaussian distribution [54], [60], illustrated in Figure 5. For the FRANKA KITCHEN benchmark, we define the action space as the product of manifolds $\mathcal { M } = \mathbb { R } ^ { 3 } \times S ^ { 3 } \times \mathbb { R } ^ { 2 }$ , whose components represent the end-effector position, the end-effector orientation (encoded as quaternions), and the gripper fingers position. Regarding the real robot tasks, the action space is defined similarly as the product of manifolds $\mathcal { M } = \mathbb { R } ^ { 3 } \times$ $S ^ { 3 } \times \mathbb { R } ^ { 1 }$ , whose components represent the position, orientation (encoded as quaternions), and opening of the gripper. For the FRANKA KITCHEN benchmark and the real robot tasks, the Euclidean and hypersphere parts employ Euclidean Gaussian distributions and a wrapped Gaussian distribution, respectively. Notice that this choice of prior distributions follows common practice in flow matching literature, where Gaussian and uniform priors are widely adopted due to their simplicity and effectiveness [16]. While more sophisticated approaches could be explored, e.g., using Gaussian Process as prior distributions [61], or learning task-dependent priors as in D-Flow [62], the choice depends on the problem at hand. We focus here on standard priors to maintain simplicity and consistency across tasks, while isolating their impact on our framework.

![](images/40b560c8b7c3e4cd4b2dc7edb87861cc5513f9188d3039b66d6f34bb8a7e27d7.jpg)  
Fig. 5. 2D visualization of prior distributions on the sphere: wrapped Gaussian distribution (left) and sphere uniform distribution (right). The sphere Gaussian distribution is obtained by first sampling from a Euclidean Gaussian distribution on a tangent space of the sphere (the red points), followed by the exponential map, which projects the samples onto the sphere manifold S<sup>d</sup> (blue points). The spherical uniform distribution is computed by normalizing samples from a zero-mean Euclidean Gaussian distribution.

For all experiments, we optimize the network parameters of RFMP and SRFMP using AdamW [63] with a learning rate of $\eta \ : = \ : 1 \times 1 0 ^ { - 4 }$ and weight decay of $w _ { d } ~ = ~ 0 . 0 0 1$ based on an exponential moving averaging (EMA) framework on the weights [64] with a decay of $w _ { \mathrm { E M A } } = 0 . 9 9 9$ . We set the SRFMP parameters as $\lambda _ { x } ~ = ~ \lambda _ { \tau } ~ = ~ 2 . 5$ . Table I summarizes the image resolution, number of parameters, and number of training epochs used in each experiment. Note that the policies predicts a sequence of actions a over a given time horizon $T _ { p } .$ Therefore, the total action space corresponds to the Cartesian product of manifolds computed over the prediction horizon defined for each task. Table II reports the total task dimensionality dim(a) as a function of the action space dimension dim(a<sup>s</sup>) and the prediction horizon $T _ { p }$ for all experiments considered in the paper. Notably, the FRANKA KITCHEN benchmark involves learning a policy that predicts a 1944-dimensional action sequence, enabling us to test the scalability of our method in high-dimensional, longhorizon settings. We use an action horizon ${ T _ { a } = T _ { p } / 2 }$ , and an observation horizon $T _ { o } = 2$ for all tasks.

![](images/c26512f1b808c631a4886191e8ec9b76f45d6daecd4010e9f9579544469693d4.jpg)  
Fig. 6. Simulation benchmarks. Top row: Euclidean PUSH-T [2], SPHERE PUSH-T, and five Robomimic tasks [10]: LIFT, CAN, SQUARE, TOOL HANG, TRANSPORT. Bottom row: Tasks of the Franka kitchen benchmarks, i.e., open microwave, put kettle on top burner, switch on the light, slide cabinet.

2) Baselines: In [2], DP is trained using either Denoising Diffusion Probabilistic Model (DDPM) [12] or Denoising Diffusion Implicit Model (DDIM) [13]. In this paper, we prioritize faster inference and thus employ DDIM-based DP for all our experiments. We train DDIM with 100 denoising steps. The prior distribution is a standard Gaussian distribution unless explicitly mentioned. During training we use the same noise scheduler as in [2], the optimizer AdamW with the same learning rate and weight decay as for RFMP and SRFMP. Note that DP does not handle data on Riemannian manifolds, and thus does not guarantee that the resulting trajectories lie on the manifold of interest for tasks with Riemannian action spaces, e.g., the SPHERE PUSH-T, the FRANKA KITCHEN benchmark, and the real-world robot experiments. In these cases, we postprocess the trajectories obtained during inference and project them on the manifold. In the case of the hypersphere manifold, the projection corresponds to a unit-norm normalization.

We also compare RFMP and SRFMP against CP [3] on the Robomimic tasks with vision-based observations. For a fair comparison, we retrained CP from scratch using the code provided in [3]. We pretrain the DP teacher policy for 100 epochs and subsequently distilled the student policy for another 100 epochs, following the CP training procedure [3]. Notice that this increases the total training time, effectively doubling the number of epochs, compared to RFMP, SRFMP, and DP. The neural network architecture used in DP and CP to model the diffusion noise is identical to the one used in RFMP and SRFMP. This isolates the influence of the learning algorithm, eliminating the architectural aspects as confounding factors.

3) Evaluation methodology: We evaluate all policies using three key metrics: (1) The performance, computed as the average task-depending score across all trials, with 50 trials for each simulated task, and 10 trials for each real-world task; (2) The number of training epochs; and (3) The inference time. To provide a consistent measure of inference time across RFMP, SRFMP, and DP, we report it in terms of the number of function evaluations (NFE), which is proportional to the inference process time. Given that each function evaluation takes approximately the same time across all methods, inference time comparisons can be made directly based on NFE. For example, in real-world tasks, with an NFE of 2, DP requires around 0.0075s, while RFMP and SRFMP take approximately 0.0108s and 0.0113s, respectively. When NFE is increased to 5, DP takes around 0.0175s, while RFMP and SRFMP require about 0.021s and 0.0212s.

TABLE II  
PREDICTION HORIZON AND ACTION SPACE DIMENSIONALITY.

<table><tr><td>Task</td><td> $T_p$ </td><td> $\text{dim}(\boldsymbol{a}^s)$ </td><td> $\text{dim}(\boldsymbol{a})$ </td></tr><tr><td>Euclidean PUSH-T</td><td>16</td><td>2</td><td>32</td></tr><tr><td>SPHERE PUSH-T</td><td>16</td><td>2</td><td>32</td></tr><tr><td>Robomimic tasks</td><td>16</td><td>7</td><td>112</td></tr><tr><td>FRANKA KITCHEN</td><td>216</td><td>9</td><td>1944</td></tr><tr><td>Real-world experiments</td><td>16</td><td>7</td><td>112</td></tr></table>

## B. Push-T Tasks

We first consider two simple PUSH-T tasks, namely the Euclidean PUSH-T proposed in [2], which was adapted from the Block Pushing task [8], and the SPHERE PUSH-T TASK, which we introduce shortly. The goal of the Euclidean PUSH-T task, illustrated in Figure 6, is to push a gray T-shaped object to the designated green target area with a blue circular agent. The agent’s movement is constrained by a light gray square boundary. Each observation o is composed of the 96 × 96 RGB image of the current scene and the agent’s state information. We introduce the SPHERE PUSH-T task, visualized in Figure 6, to evaluate the performance of our models on the sphere manifold. Its environment is obtained by projecting the Euclidean PUSH-T environment on one half of a 2-dimensional sphere $S ^ { 2 }$ of radius of 1. This is achieved by projecting the environment, normalized to a range [−1.5, 1.5], from the plane z = 1 to the sphere via a stereographic projection. The target area, the T-shaped object, and the agent then lie and evolve on the sphere. As in the Euclidean case, each observation o is composed of the 96 × 96 RGB image of the current scene and the agent’s state information on the manifold. All models (i.e., RFMP, SRFMP, DP) are trained for 300 epochs in both settings. During testing, we choose the best validation epoch and roll out 500 steps in the environment with an early stop rule terminating the execution when the coverage area is over 95% of the green target area. The score for both Euclidean and Sphere Push-T tasks is the maximum coverage ratio during execution. The tests are performed with 50 different initial states not present in the training set.

TABLE III  
EUCLIDEAN PUSH-T: IMPACT OF NFE ON POLICIES.

<table><tr><td rowspan="2">Policy</td><td colspan="4">NFE</td></tr><tr><td>1</td><td>3</td><td>5</td><td>10</td></tr><tr><td>RFMP</td><td>0.848</td><td>0.855</td><td>0.923</td><td>0.891</td></tr><tr><td>SRFMP</td><td>0.875</td><td>0.851</td><td>0.837</td><td>0.856</td></tr><tr><td>DP</td><td>0.109</td><td>0.79</td><td>0.838</td><td>0.862</td></tr></table>

TABLE IV  
RFMP HYPERPARAMETERS ABLATION ON EUCLIDEAN PUSH-T.

<table><tr><td rowspan="2">Parameter NFE</td><td rowspan="2">Values</td><td colspan="5">Success rate</td></tr><tr><td>1</td><td>3</td><td>5</td><td>10</td><td>100</td></tr><tr><td rowspan="3"> $T_o$ </td><td>2</td><td>0.848</td><td>0.855</td><td>0.923</td><td>0.891</td><td>0.91</td></tr><tr><td>8</td><td>0.195</td><td>0.16</td><td>0.154</td><td>0.168</td><td>0.179</td></tr><tr><td>16</td><td>0.135</td><td>0.143</td><td>0.14</td><td>0.133</td><td>0.135</td></tr><tr><td rowspan="3"> $T_p$ </td><td>8</td><td>0.754</td><td>0.835</td><td>0.827</td><td>0.839</td><td>0.85</td></tr><tr><td>16</td><td>0.848</td><td>0.855</td><td>0.923</td><td>0.891</td><td>0.91</td></tr><tr><td>32</td><td>0.799</td><td>0.906</td><td>0.878</td><td>0.929</td><td>0.93</td></tr><tr><td rowspan="3">η</td><td> $1 \times 10^{-4}$ </td><td>0.848</td><td>0.855</td><td>0.923</td><td>0.891</td><td>0.91</td></tr><tr><td> $5 \times 10^{-5}$ </td><td>0.797</td><td>0.863</td><td>0.843</td><td>0.897</td><td>0.889</td></tr><tr><td> $1 \times 10^{-5}$ </td><td>0.641</td><td>0.771</td><td>0.805</td><td>0.88</td><td>0.841</td></tr><tr><td rowspan="3"> $w_d$ </td><td>0.001</td><td>0.848</td><td>0.855</td><td>0.923</td><td>0.891</td><td>0.91</td></tr><tr><td>0.005</td><td>0.846</td><td>0.882</td><td>0.875</td><td>0.866</td><td>0.856</td></tr><tr><td>0.01</td><td>0.868</td><td>0.831</td><td>0.842</td><td>0.927</td><td>0.853</td></tr></table>

1) Euclidean Push-T: First, we evaluate the performance of RFMP and SRFMP in the Euclidean case for different number of function evaluations in the testing phase. The models are trained with the default parameters described in Section V-A1. Table III shows the success rate of RFMP and SRFMP for different NFEs. We observe that both RFMP and SRFMP achieve similar success rates overall. While SRFMP demonstrates superior performance with a single NFE, RFMP achieves higher success rates with more NFEs. We hypothesize that this behavior arises from the fact that, due to the equality $\lambda _ { x } = \lambda _ { \tau }$ the SRFMP conditional probability path resembles the optimal transport map between the prior and target distributions as in [15]. This, along with the stability framework of SRFMP, allows us to automatically choose the time step during inference via (24), which enhances convergence in a single step. When comparing our approaches with DP, we observe that DP performs drastically worse than both RFMP and SRFMP for a single NFE, achieving a score of only 10.9%. Nevertheless, the performance of DP improves when increasing the NFE and matches that of our approaches for 10 NFE.

Next, we ablate the action prediction horizon $T _ { p } ,$ , observation horizon $T _ { o } ,$ learning rate $\eta ,$ and weight decay $w _ { d }$ for RFMP and SRFMP. We consider 3 different values for each, while setting the other hyperparameters to their default values, and test the resulting models with 5 different NFE. For

SRFMP HYPERPARAMETERS ABLATION ON EUCLIDEAN PUSH-T.  
TABLE V

<table><tr><td>Parameter</td><td>Values</td><td colspan="4">Success rate</td></tr><tr><td>NFE</td><td></td><td>1</td><td>3</td><td>5</td><td>10</td></tr><tr><td rowspan="3"> $T_o$ </td><td>2</td><td>0.875</td><td>0.851</td><td>0.837</td><td>0.856</td></tr><tr><td>8</td><td>0.124</td><td>0.139</td><td>0.147</td><td>0.13</td></tr><tr><td>16</td><td>0.145</td><td>0.138</td><td>0.149</td><td>0.144</td></tr><tr><td rowspan="3"> $T_p$ </td><td>8</td><td>0.816</td><td>0.726</td><td>0.592</td><td>0.318</td></tr><tr><td>16</td><td>0.875</td><td>0.851</td><td>0.837</td><td>0.856</td></tr><tr><td>32</td><td>0.852</td><td>0.861</td><td>0.881</td><td>0.829</td></tr><tr><td rowspan="3"> $η$ </td><td> $1.0 \times 10^{-4}$ </td><td>0.875</td><td>0.851</td><td>0.837</td><td>0.856</td></tr><tr><td> $5 \times 10^{-5}$ </td><td>0.754</td><td>0.621</td><td>0.456</td><td>0.334</td></tr><tr><td> $1 \times 10^{-5}$ </td><td>0.602</td><td>0.56</td><td>0.443</td><td>0.288</td></tr><tr><td rowspan="3"> $w_d$ </td><td>0.001</td><td>0.875</td><td>0.851</td><td>0.837</td><td>0.856</td></tr><tr><td>0.005</td><td>0.837</td><td>0.826</td><td>0.826</td><td>0.684</td></tr><tr><td>0.01</td><td>0.733</td><td>0.75</td><td>0.768</td><td>0.571</td></tr><tr><td rowspan="12"> $λ_x \quad λ_\tau$ </td><td>1 0.2</td><td>0.74</td><td>0.614</td><td>0.494</td><td>0.457</td></tr><tr><td>1 1</td><td>0.85</td><td>0.777</td><td>0.608</td><td>0.416</td></tr><tr><td>1 6</td><td>0.87</td><td>0.768</td><td>0.753</td><td>0.812</td></tr><tr><td>2.5 0.2</td><td>0.832</td><td>0.392</td><td>0.576</td><td>0.549</td></tr><tr><td>2.5 2.5</td><td>0.875</td><td>0.851</td><td>0.837</td><td>0.856</td></tr><tr><td>2.5 15</td><td>0.832</td><td>0.799</td><td>0.789</td><td>0.807</td></tr><tr><td>5 1</td><td>0.796</td><td>0.741</td><td>0.614</td><td>0.513</td></tr><tr><td>5 5</td><td>0.845</td><td>0.825</td><td>0.797</td><td>0.642</td></tr><tr><td>5 30</td><td>0.822</td><td>0.830</td><td>0.772</td><td>0.743</td></tr><tr><td>7.5 1.5</td><td>0.799</td><td>0.685</td><td>0.442</td><td>0.456</td></tr><tr><td>7.5 7.5</td><td>0.787</td><td>0.8</td><td>0.809</td><td>0.633</td></tr><tr><td>7.5 45</td><td>0.782</td><td>0.817</td><td>0.844</td><td>0.814</td></tr></table>

SRFMP, we additionally ablate the parameters $\lambda _ { x }$ and $\lambda _ { \tau }$ for 3 different ratios $\lambda _ { x } / \lambda _ { \tau }$ and 4 values for each ratio. Each setup is tested with 50 seeds, resulting in a total of 2250 and 4000 experiments for RFMP and SRFMP. The results are reported in Tables IV and V, respectively. We observe that a short observation horizon $T _ { o } = 2$ leads to the best performance for both models. This is consistent with the task, as the current and previous images accurately provide the required information for the next pushing action, while the actions associated with past images rapidly become outdated. Moreover, we observe that an action prediction horizon $T _ { p } = 1 6$ leads to the highest score. We hypothesize that this horizon allows the model to maintain temporal consistency, while providing frequent enough updates of the actions according to the current observations. Concerning SRFMP, we find that $\lambda _ { x } = \lambda _ { \tau } = 2 . 5$ leads to the highest success rates. Interestingly, this choice leads to the ratio $\lambda _ { x } / \lambda _ { \tau } = 1$ , in which case the flow of x follows the Gaussian CFM (6) of [15] with $\sigma  0$ for τ = [0, 1], see [20, Cor 4.12]. In the next experiments, we use the default parameters resulting from our ablations, i.e., $T _ { p } { = } 1 6 , T _ { o } { = } 2 , \eta { = } 1 { \times } 1 0 ^ { - 4 } , w _ { d } { = } 0 . 0 0 1$ , and $\lambda _ { x } = \lambda _ { \tau } = 2 . 5$ 2) Sphere Push-T: Next, we test the ability of RFMP and SRFMP to generate motions on non-Euclidean manifolds with the SPHERE PUSH-T task. We evaluate two types of Riemannian prior distributions for RFMP and SRFMP, namely a spherical uniform distribution and wrapped Gaussian distribution (see Figure 5). We additionally consider a Euclidean Gaussian distribution for DP. Notice that the actions generated by DP are normalized in a post-processing step to ensure that they belong to the sphere. The corresponding performance are reported in Table VI. Our results indicate that the choice of prior distribution significantly impacts the performance of both RFMP and SRFMP. Specifically, we observe that RFMP and SRFMP with a uniform sphere distribution consistently outperform their counterparts with wrapped Gaussian distribution. We hypothesize that RFMP or SRFMP benefit from having samples that are close to the data support, which leads to simpler vector fields to learn. In other words, uniform distribution provides more samples around the data distribution, which potentially lead to simpler vector fields. DP exhibit poor performance with sphere-based prior distributions, suggesting its ineffectiveness in handling such priors. Instead, DP’s performance drastically improves when using a Euclidean Gaussian distribution and higher NFE. Note that this high performance does not scale to higher dimensional settings as already evident in the realworld experiments reported in Section V-E, where the effect of ignoring the geometry of the parameters exacerbates, which is a known issue when naively operating with Riemannian data [65]. Importantly, SRFMP is consistently more robust to NFE and achieves high performance with a single NFE, leading to shorter inference times for similar performance compared to RFMP and DP.

TABLE VI  
IMPACT OF THE PRIOR DISTRIBUTION ON SPHERE PUSH-T TASK.

<table><tr><td rowspan="2">Policy</td><td colspan="4">NFE</td></tr><tr><td>1</td><td>3</td><td>5</td><td>10</td></tr><tr><td>RFMP sphere uniform</td><td>0.871</td><td>0.746</td><td>0.77</td><td>0.817</td></tr><tr><td>RFMP sphere Gaussian</td><td>0.587</td><td>0.724</td><td>0.748</td><td>0.733</td></tr><tr><td>SRFMP sphere uniform</td><td>0.772</td><td>0.736</td><td>0.796</td><td>0.829</td></tr><tr><td>SRFMP sphere Gaussian</td><td>0.707</td><td>0.706</td><td>0.735</td><td>0.707</td></tr><tr><td>DP sphere uniform</td><td>0.274</td><td>0.261</td><td>0.235</td><td>0.197</td></tr><tr><td>DP sphere Gaussian</td><td>0.170</td><td>0.162</td><td>0.231</td><td>0.227</td></tr><tr><td>DP euclidean Gaussian</td><td>0.227</td><td>0.796</td><td>0.813</td><td>0.885</td></tr></table>

TABLE VII  
INFLUENCE OF INTEGRATION TIME ON RFMP AND SRFMP.

<table><tr><td>Euclidean PUSH-T</td><td>t = 1.0</td><td>t = 1.2</td><td>t = 1.6</td></tr><tr><td>RFMP</td><td>0.855</td><td>0.492</td><td>0.191</td></tr><tr><td>SRFMP</td><td>0.862</td><td>0.851</td><td>0.829</td></tr><tr><td>Sphere PUSH-T</td><td>t = 1.0</td><td>t = 1.2</td><td>t = 1.6</td></tr><tr><td>RFMP</td><td>0.736</td><td>0.574</td><td>0.264</td></tr><tr><td>SRFMP</td><td>0.727</td><td>0.736</td><td>0.685</td></tr></table>

3) Influence of Integration Time Boundary: We further assess the robustness of SRFMP to varying time boundaries on the PUSH-T tasks by increasing the time boundary during inference. The performance of both RFMP and SRFMP is summarized in Table VII with result presented for NFE = 3 under the time boundaries t = 1 and t = 1.2, as well as for NFE = 4 under the time boundaries t=1.6. Our results show that the performance of RFMP is highly sensitive to the time boundary, gradually declining as the boundary increases. In contrast, SRFMP demonstrates remarkable robustness, with minimal variation across different time boundaries. As illustrated in Figure 7, the quality of action series generated by RFMP noticeably deteriorates with increasing time boundaries, whereas SRFMP consistently delivers high-quality action series regardless of the time boundary.

## C. Robomimic Benchmark

Next, we evaluate RFMP and SRFMP on the well-known Robomimic robotic manipulation benchmark [10]. This benchmark consists of five tasks with varying difficulty levels for which it provides two types of demonstrations, namely proficient human (PH) high-quality teleoperated demonstrations, and mixed human (MH) demonstrations. Each demonstration contains multi-modal observations, including state information, images, and depth data. We report results on five tasks (LIFT, CAN, SQUARE, TOOL HANG, and TRANSPORT) from the Robomimic dataset with 200 PH demonstrations for training for both state- and vision-based observations. Note that the difficulty of the selected tasks becomes progressively more challenging. The score of each of the 50 trials is determined by whether the task is completed successfully after a given number of steps (300 for LIFT, 500 for CAN and SQUARE, 700 for TOOL HANG, and 500 for TRANSPORT). The performance is then the percentage of successful trials.

![](images/5d7dc68068f692369f6eae8872ee9c2b71f06f94c51b6406f90c57b23fee1872.jpg)  
Fig. 7. Action series generated by RFMP (first and third rows) and SRFMP (second and fourth rows) trained on the PUSH-T tasks at integration times $t = \{ 0 . 8 , 1 . 2 , 1 . 6 \}$

1) State-based Observations: We first assess the training efficiency of RFMP and SRFMP and compare it against DP by analyzing their performance at different training stages. Figure 8 shows the success rate of the three policies as a function of the number of training epochs for the tasks LIFT, CAN, SQUARE, and TOOL HANG. All policies are evaluated with 3 NFE for LIFT, CAN, and SQUARE, and with 10 NFE for TOOL HANG. We observe that RFMP and SRFMP consistently outperform DP across all tasks, requiring fewer training epochs to achieve comparable or superior performance. For the easier tasks (LIFT and CAN), both RFMP and SRFMP achieve high performance after just 20 training epochs, while the success rate of DP remains low after 50 epochs. This trend persists in the harder tasks (SQUARE and TOOL HANG), with RFMP and SRFMP reaching high success rates significantly faster than DP.

TABLE VIII  
SUCCESS RATE AS A FUNCTION OF DIFFERENT NFE VALUES ON THE STATE-BASED ROBOMIMIC TASKS.

<table><tr><td></td><td colspan="5">LIFT</td></tr><tr><td>NFE</td><td>1</td><td>2</td><td>3</td><td>5</td><td>10</td></tr><tr><td>RFMP</td><td> $0.992 \pm 0.010$ </td><td> $0.992 \pm 0.010$ </td><td> $0.992 \pm 0.010$ </td><td> $0.992 \pm 0.010$ </td><td> $0.992 \pm 0.010$ </td></tr><tr><td>SRFMP</td><td> $1.000 \pm 0.000$ </td><td> $0.992 \pm 0.010$ </td><td> $0.992 \pm 0.010$ </td><td> $0.996 \pm 0.008$ </td><td> $1.000 \pm 0.000$ </td></tr><tr><td>DP</td><td> $0.008 \pm 0.010$ </td><td> $0.756 \pm 0.034$ </td><td> $0.948 \pm 0.016$ </td><td> $0.956 \pm 0.015$ </td><td> $0.976 \pm 0.015$ </td></tr><tr><td></td><td colspan="5">CAN</td></tr><tr><td>NFE</td><td>1</td><td>2</td><td>3</td><td>5</td><td>10</td></tr><tr><td>RFMP</td><td> $0.976 \pm 0.023$ </td><td> $0.996 \pm 0.008$ </td><td> $0.996 \pm 0.008$ </td><td> $0.996 \pm 0.008$ </td><td> $0.996 \pm 0.008$ </td></tr><tr><td>SRFMP</td><td> $0.980 \pm 0.022$ </td><td> $0.980 \pm 0.013$ </td><td> $0.996 \pm 0.008$ </td><td> $0.992 \pm 0.010$ </td><td> $0.968 \pm 0.020$ </td></tr><tr><td>DP</td><td> $0.004 \pm 0.008$ </td><td> $0.340 \pm 0.013$ </td><td> $0.836 \pm 0.020$ </td><td> $0.924 \pm 0.015$ </td><td> $0.908 \pm 0.010$ </td></tr><tr><td></td><td colspan="5">SQUARE</td></tr><tr><td>NFE</td><td>1</td><td>2</td><td>3</td><td>5</td><td>10</td></tr><tr><td>RFMP</td><td> $0.792 \pm 0.027$ </td><td> $0.848 \pm 0.020$ </td><td> $0.920 \pm 0.018$ </td><td> $0.896 \pm 0.041$ </td><td> $0.912 \pm 0.016$ </td></tr><tr><td>SRFMP</td><td> $0.776 \pm 0.029$ </td><td> $0.800 \pm 0.052$ </td><td> $0.828 \pm 0.016$ </td><td> $0.824 \pm 0.008$ </td><td> $0.848 \pm 0.030$ </td></tr><tr><td>DP</td><td> $0.012 \pm 0.010$ </td><td> $0.384 \pm 0.023$ </td><td> $0.628 \pm 0.016$ </td><td> $0.672 \pm 0.020$ </td><td> $0.684 \pm 0.015$ </td></tr><tr><td></td><td colspan="5">TOOL HANG</td></tr><tr><td>NFE</td><td>1</td><td>2</td><td>3</td><td>5</td><td>10</td></tr><tr><td>RFMP</td><td> $0.152 \pm 0.020$ </td><td> $0.316 \pm 0.023$ </td><td> $0.368 \pm 0.032$ </td><td> $0.572 \pm 0.027$ </td><td> $0.716 \pm 0.023$ </td></tr><tr><td>SRFMP</td><td> $0.240 \pm 0.028$ </td><td> $0.224 \pm 0.020$ </td><td> $0.308 \pm 0.020$ </td><td> $0.516 \pm 0.023$ </td><td> $0.568 \pm 0.035$ </td></tr><tr><td>DP</td><td> $0.000 \pm 0.000$ </td><td> $0.008 \pm 0.010$ </td><td> $0.008 \pm 0.010$ </td><td> $0.092 \pm 0.024$ </td><td> $0.092 \pm 0.016$ </td></tr></table>

![](images/0aca7e93a790bbdacc49ab60af688e694ba536379dbeba624de0b6ad151e5d0f.jpg)  
Fig. 8. Success rate (mean and standard deviation) on Robomimic tasks with state-based observations at different checkpoints. The models performance of LIFT, CAN, and SQUARE tasks is checked every 10 epochs throughout the 50-epoch training process using 3 NFE. For the TOOL HANG task, the models are trained over 100 epochs and checked every 20 epochs using 10 NFE.

Next, we evaluate the performance of the policies for different NFE in the testing phase. For RFMP and SRFMP, we use the 50-epoch models for LIFT, CAN, and SQUARE, and the 100-epoch models for TOOL HANG. DP is further trained for a total of 300 epochs and we select the model at the best validation epoch. The results are reported in Table VIII. RFMP and SRFMP outperform DP for all tasks and all NFE, even though DP was trained for more epochs. Moreover, we observe that RFMP and SRFMP are generally more robust to low NFE than DP. They achieve 100% success rate at almost all NFE values for the easier LIFT and CAN tasks, while $\mathrm { D P s }$ performance drastically drops for 1 and 2 NFE. We observe a similar trend for the SQUARE task, where the performance of RFMP and SRFMP slightly improves when increasing the NFE. The performance of all models drops for TOOL HANG, which is the most complex of the four considered tasks. In this case, the performance of RFMP and SRFMP is limited for low NFE values and improves for higher NFE. DP performs poorly for all considered NFE values. Table IX reports the jerkiness as a measure of the smoothness of the trajectories generated by the different policies. We observe that RFMP and SRFMP produce arguably smoother trajectories than DP for low NFE, as indicated by the lower jerkiness values. The smoothness of the trajectories becomes comparable for higher NFE. In summary, both RFMP and SRFMP achieve high success rates and smooth action predictions with low NFE, enabling faster inference without compromising task completion.

2) Vision-based Observations: Next, we assess our models performance when the vector field is conditioned on visual observations. We consider the tasks LIFT, CAN, SQUARE, and TRANSPORT with the same policy settings and networks (see Table $\mathrm { I } ) ^ { 1 }$ . Each observation $\pmb { o } ^ { s }$ corresponds to a visual embedding derived from camera images. For LIFT, CAN, and SQUARE, we use one over-the-shoulder and one in-hand camera. For TRANSPORT, which involves bimanual manipulation, we use two in-hand cameras and two over-the-shoulder cameras. We train the models for a total 100 epochs for LIFT, CAN, and SQUARE and for 200 epochs for TRANSPORT and use the best-performing checkpoint for evaluation.

The performance of different policies is reported in Table XI. RFMP and SRFMP consistently outperform DP and CP on all tasks, regardless of the NFE. As for the previous experiments, our models are remarkably robust to changes in NFE compared to DP. Importantly, SRFMP consistently outperforms RFMP for 1 and 2 NFE. Regarding CAN and SQUARE tasks, SRFMP with 1 NFE achieved performance on par with RFMP using 3 NFE. This efficiency gain showcases the benefits of enhancing the policies with stability to the target distribution for reducing their inference time.

TABLE IX  
JERKINESS OF PREDICTED ROBOT TRAJECTORIES FOR DIFFERENT NFE ON THE ROBOMIMIC TASKS WITH STATE-BASED OBSERVATIONS. ALL VALUES ARE EXPRESSED IN THOUSANDS, WHERE THE LOWER THE SMOOTHER THE PREDICTION.

<table><tr><td></td><td colspan="5">LIFT</td></tr><tr><td>NFE</td><td>1</td><td>2</td><td>3</td><td>5</td><td>10</td></tr><tr><td>RFMP</td><td> $9.97 \pm 1.23$ </td><td> $8.80 \pm 0.19$ </td><td> $9.06 \pm 0.08$ </td><td> $8.96 \pm 0.08$ </td><td> $8.49 \pm 0.21$ </td></tr><tr><td>SRFMP</td><td> $10.32 \pm 0.41$ </td><td> $9.90 \pm 031$ </td><td> $8.00 \pm 0.14$ </td><td> $9.39 \pm 0.23$ </td><td> $9.03 \pm 0.19$ </td></tr><tr><td>DP</td><td> $329.20 \pm 47.44$ </td><td> $13.96 \pm 0.89$ </td><td> $8.06 \pm 0.42$ </td><td> $7.48 \pm 0.13$ </td><td> $5.9 \pm 0.23$ </td></tr><tr><td></td><td colspan="5">CAN</td></tr><tr><td>NFE</td><td>1</td><td>2</td><td>3</td><td>5</td><td>10</td></tr><tr><td>RFMP</td><td> $7.88 \pm 0.14$ </td><td> $6.38 \pm 0.40$ </td><td> $6.37 \pm 0.17$ </td><td> $6.40 \pm 0.07$ </td><td> $7.10 \pm 1.30$ </td></tr><tr><td>SRFMP</td><td> $7.64 \pm 0.23$ </td><td> $7.63 \pm 0.18$ </td><td> $6.69 \pm 0.27$ </td><td> $6.98 \pm 0.13$ </td><td> $6.94 \pm 0.08$ </td></tr><tr><td>DP</td><td> $526.00 \pm 65.50$ </td><td> $18.98 \pm 2.82$ </td><td> $6.68 \pm 0.25$ </td><td> $6.28 \pm 0.35$ </td><td> $6.22 \pm 0.16$ </td></tr><tr><td></td><td colspan="5">SQUARE</td></tr><tr><td>NFE</td><td>1</td><td>2</td><td>3</td><td>5</td><td>10</td></tr><tr><td>RFMP</td><td> $9.14 \pm 0.19$ </td><td> $6.22 \pm 0.29$ </td><td> $5.54 \pm 0.24$ </td><td> $5.17 \pm 0.16$ </td><td> $4.36 \pm 0.30$ </td></tr><tr><td>SRFMP</td><td> $9.16 \pm 0.31$ </td><td> $9.43 \pm 0.16$ </td><td> $7.28 \pm 0.97$ </td><td> $7.70 \pm 0.50$ </td><td> $7.74 \pm 0.26$ </td></tr><tr><td>DP</td><td> $548.20 \pm 54.77$ </td><td> $20.26 \pm 2.47$ </td><td> $7.24 \pm 0.54$ </td><td> $10.02 \pm 2.64$ </td><td> $7.62 \pm 1.44$ </td></tr><tr><td></td><td colspan="5">TOOL HANG</td></tr><tr><td>NFE</td><td>1</td><td>2</td><td>3</td><td>5</td><td>10</td></tr><tr><td>RFMP</td><td> $4.49 \pm 0.35$ </td><td> $4.26 \pm 0.19$ </td><td> $4.65 \pm 0.32$ </td><td> $5.07 \pm 0.19$ </td><td> $4.66 \pm 0.31$ </td></tr><tr><td>SRFMP</td><td> $5.24 \pm 0.18$ </td><td> $5.10 \pm 0.20$ </td><td> $5.23 \pm 0.38$ </td><td> $5.62 \pm 0.42$ </td><td> $5.62 \pm 0.40$ </td></tr><tr><td>DP</td><td> $699.40 \pm 90.91$ </td><td> $9.52 \pm 0.30$ </td><td> $8.79 \pm 1.51$ </td><td> $7.12 \pm 1.94$ </td><td> $6.96 \pm 1.35$ </td></tr></table>

TABLE X

TRAINING TIME (IN SECONDS) PER EPOCH ON ROBOMIMIC VISION-BASEDLIFT AND SQUARE.

<table><tr><td>Task</td><td>RFMP</td><td>SRFMP</td><td>DP</td><td>CP</td></tr><tr><td>LIFT</td><td> $17.49 \pm 0.03$ </td><td> $17.63 \pm 0.15$ </td><td> $16.07 \pm 0.20$ </td><td> $31.44 \pm 0.27$ </td></tr><tr><td>SQUARE</td><td> $57.62 \pm 0.30$ </td><td> $56.67 \pm 0.18$ </td><td> $53.16 \pm 0.16$ </td><td> $102.86 \pm 0.61$ </td></tr></table>

In contrast to CP, RFMP and SRFMP rely on a simple, single-stage training pipeline, thus featuring easier and faster training in addition to fast inference. The training time per epoch of each policy on task LIFT and SQUARE tasks is summarized in Table X. All experiments were conducted on an NVIDIA RTX 4060Ti GPU using the same batch size across policies. The result indicate that RFMP, SRFMP, and DP have comparable per-epoch training times. In contrast, CP requires approximately twice as much time per epoch, primarily due to its training procedure involving distillation from a teacher policy. When accounting for the additional time required to pretrain DP the total training time per epoch for CP becomes roughly three times that of the other methods.

## D. Franka Kitchen Benchmark

We evaluate the capability of RFMP and SRFMP in handling more complex, long-horizon manipulation tasks on the widely-used FRANKA KITCHEN benchmark [58]. We use the dataset from [66], which comprises 19 expert demonstrations totaling 4209 time steps and involves a sequential execution of four tasks: open the microwave, put the kettle on the top burner, switch on the light and slide the cabinet. The original dataset considers actions in Euclidean space as joint angular velocities. To evaluate the performance of our Riemannian policy framework, we instead consider end-effector trajectories obtained from joint configurations via forward kinematics. Therefore, each data point is composed of the end-effector position and orientation (as a unit quaternion), and the gripper state. Consequently, the predicted action sequence lies on the product of manifolds $\mathbb { R } ^ { 3 } \times S ^ { 3 } \times \mathbb { R } ^ { 2 }$

Each model is trained for 500 epochs with a batch size of 32, and evaluated using the best validation checkpoint over 50 test episodes with randomized initial states. The performance of different policies is reported in Table XII. SRFMP exhibits consistently high performance across all NFE, indicating strong robustness, and significantly outperforms RFMP and DP. The low success rate of RFMP and DP is due to the fact that their trajectories often exhibit local inaccuracies, leading to failing at least one subtask. This is then reported as a failure for the long-horizon task, leading to low success rates overall. In contrast, the trajectories produced by SRFMP are more precise, leading to the successful completion of all subtasks.

## E. Real Robotic Experiments

Finally, we evaluate RFMP and SRFMP on two real-world tasks, namely PICK & PLACE and MUG FLIPPING, with a 7-DoF robotic manipulator.

1) Experimental Setup: Figures 9-11 show our experimental setup. The tasks are performed on a Franka Emika Panda robot arm. We collect the demonstrations via a teleoperation system made of two robot twins. Demonstrations are collected by an expert guiding the source robot. The target robot reads the end-effector pose of the source robot and reproduces it via a Cartesian impedance controller. The demonstration data was smoothed to mitigate the noise arising from the teleoperation setup. Each observation $o ^ { s }$ is composed of the end-effector po sition and of the image embedding obtained from the ResNet vision backbone that processes the images from an overthe-shoulder camera. The policies are trained to generate 8- dimensional actions composed of the position, orientation, and gripper state lying on the product of manifolds $\mathbb { R } ^ { 3 } \times S ^ { 3 } \times \mathbb { R }$

2) Pick & Place: The goal of this task is to test the ability of RFMP and SRFMP to learn Euclidean policies in real-world settings. The task consists of approaching and picking up a white mug, and place it on a pink plate, as shown in Figure 10. Note that the robot end-effector points downwards during the entire task, so that its orientation remains almost constant.

TABLE XI  
SUCCESS RATE AS A FUNCTION OF NFE ON VISION-BASED ROBOMIMIC TASKS.

<table><tr><td>Task</td><td colspan="5">LIFT</td><td colspan="5">CAN</td><td colspan="5">SQUARE</td><td colspan="5">TRANSPORT</td></tr><tr><td>NFE</td><td>1</td><td>2</td><td>3</td><td>5</td><td>10</td><td>1</td><td>2</td><td>3</td><td>5</td><td>10</td><td>1</td><td>2</td><td>3</td><td>5</td><td>10</td><td>1</td><td>2</td><td>3</td><td>5</td><td>10</td></tr><tr><td>RFMP</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>0.78</td><td>0.82</td><td>0.9</td><td>0.96</td><td>0.94</td><td>0.56</td><td>0.74</td><td>0.9</td><td>0.9</td><td>0.9</td><td>0.6</td><td>0.82</td><td>0.78</td><td>0.78</td><td>0.8</td></tr><tr><td>SRFMP</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>0.88</td><td>0.88</td><td>0.9</td><td>0.9</td><td>0.86</td><td>0.86</td><td>0.82</td><td>0.9</td><td>0.88</td><td>0.9</td><td>0.8</td><td>0.82</td><td>0.82</td><td>0.78</td><td>0.78</td></tr><tr><td>DP</td><td>0</td><td>0.7</td><td>0.96</td><td>0.98</td><td>0.98</td><td>0</td><td>0.38</td><td>0.66</td><td>0.68</td><td>0.66</td><td>0</td><td>0.04</td><td>0.16</td><td>0.26</td><td>0.12</td><td>0</td><td>0.54</td><td>0.66</td><td>0.78</td><td>0.82</td></tr><tr><td>CP</td><td>0.5</td><td>0.44</td><td>0.46</td><td>0.46</td><td>0.4</td><td>0.82</td><td>0.82</td><td>0.84</td><td>0.84</td><td>0.82</td><td>0.32</td><td>0.52</td><td>0.54</td><td>0.48</td><td>0.52</td><td>0.4</td><td>0.36</td><td>0.44</td><td>0.38</td><td>0.36</td></tr></table>

TABLE XII

SUCCESS RATE AS A FUNCTION OF NFE ON THE FRANKA KITCHEN BENCHMARK.

<table><tr><td rowspan="2">Policy</td><td colspan="5">NFE</td></tr><tr><td>1</td><td>2</td><td>3</td><td>5</td><td>10</td></tr><tr><td>RFMP</td><td>0.04</td><td>0.08</td><td>0.12</td><td>0.14</td><td>0.14</td></tr><tr><td>SRFMP</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td></tr><tr><td>DP</td><td>0</td><td>0</td><td>0</td><td>0.02</td><td>0.02</td></tr></table>

![](images/c4bf8025f9176c343881933e94742800d3d2e80d5e84ccc7661bb9e38f3d9c79.jpg)  
Fig. 9. Robotic experimental setup consisting of two Franka Emika Panda robot arms and an over-the-shoulder camera (Realsense d435). The left arm is the target robot, while the right one acts as the source. During the teaching phase, a human expert controls the source arm to teleoperate the target robot. During testing, only the target arm is operational.

We collect 100 demonstrations where the white mug is randomly placed on the yellow mat, while the pink plate position and end-effector initial position are slightly varied. We split our demonstration data to use 90 demonstrations for training and 10 for validation. All models are trained for 300 epochs with the same training hyperparameters as reported in Table I. As in previous experiments, we use the bestperforming checkpoints of each model for evaluation.

During testing, we systematically place the white mug at 10 different locations on a semi-grid covering the surface of the yellow sponge. We evaluate the performance of RFMP, SRFMP, and DP as a function of different NFE values, under two metrics: Success rate and prediction smoothness. Figure 12 shows the increased robustness of RFMP and SRFMP to NFE compared to DP. Notably, DP requires more NFEs to achieve a success rate competitive to RFMP and SRFMP, which display high performance with only 2 NFE. Moreover, DP generated highly jerky predictions when using 2 NFE. In contrast, RFMP and SRFMP consistently retrieve smooth trajectories, regardless of the NFE.

Notice that the experiments involved natural variations in background and lighting conditions, further exposing the policies to realistic sensor noise. These slight variations in background and lighting conditions (e.g., cloudy and sunny days), had minimal impact on the policies performance. However, consistent failures were observed when the mug was initially positioned on the right side of the yellow mat, where the robot often obstructs the external camera view when approaching the mug. A multi-camera setting may improve performance on such occlusion cases.

![](images/87bfef38e17911c70f2119ab854eb14fb87cb3e787dbdf4e68e8776a455752c6.jpg)  
Fig. 10. PICK & PLACE: First, the robot end-effector approaches and grasps the white mug. Then, it lifts the mug and places it upright on the pink plate.

![](images/5eccce291d410353892c22a10e20ce998c3c907dccd682aa25c4241f332daed1.jpg)  
Fig. 11. MUG FLIPPING: The robot rotates its end-effector to align with the white mug’s orientation and subsequently grasps it. It then place the mug upright on the blue plate.

3) Mug Flipping: In this task, a white mug is initially positioned horizontally on a yellow sponge, as shown in Figure 11. The task consists of two stages: First, the robot locates the white mug and grasps it by rotating its endeffector to align with the mug orientation. The robot then places the mug upright on the blue plate. Note that this task demands the robot to execute elaborated rotation trajectories for both grasping and placing. For this task, we collect 50 demonstrations with the white mug randomly positioned and rotated on the left side of a yellow sponge. Note that we use only the left side as the task requires the robot to operate near its workspace limits, which are prohibitive when the mug is placed on the right side. Furthermore, the end-effector initial pose was also slightly varied across the demonstrations. The policy hyperparameters for this task are provided in Table I.

Figure 12 shows the results of evaluating the different considered policies using the same two metrics as the PICK & PLACE task, namely success rate and trajectory smoothness. Both RFMP and SRFMP are significantly more robust to different NFE in terms of success rate when compared to DP. While the smoothness of RFMP and SRFMP is slightly affected by NFE in this particular task, both methods still outperform DP in this regard. Similarly to the PICK & PLACE, slight variations on lightning and background had a negligible effect on the performance of the tested policies.

![](images/c9fec84df4d32d673cef11b288eebc93d287e7a41a02028bdd7729fb87ab4e37.jpg)  
Fig. 12. Success rate and predicted actions jerkiness as a function of NFE on the PICK & PLACE and MUG FLIPPING tasks.

## F. Experiments Summary

We conducted comprehensive evaluations across diverse benchmarks in simulation, namely the Euclidean and SPHERE PUSH-T tasks, the Robomimic benchmark under both stateand vision-based observation, the long-horizon FRANKA KITCHEN benchmark, as well as two real-world manipulation tasks. These tasks display various action spaces with different types of geometric constraints. While the Euclidean PUSH-T and Robomimic tasks are defined on standard Euclidean spaces — a special type of Riemannian manifolds — the SPHERE PUSH-T task involves an action space constrained to the hypersphere $S ^ { 2 }$ , and both the FRANKA KITCHEN and realworld experiments operate on product Euclidean and sphere manifolds, accounting for position, orientation, and gripper state. Through their Riemannian formulation, RFMP and SRFMP are designed to naturally handle Euclidean and non-Euclidean action spaces, thus satisfying constraints such as the unit-norm of quaternions. It is important to emphasize that the Riemannian formulation is necessary not only to handle geometric constraints but also to guarantee stability in SRFMP. Naive approximations, e.g., unit-norm normalization as a post-processing step for quaternions, would break the stability guarantees in SRFMP.

Our findings from both simulated and real-world tasks show that RFMP and SRFMP offer significant advantages over DP. In particular, RFMP and SRFMP achieve faster inference by using fewer NFE without compromising success rates regardless of the observation type. This translates into highlyrobust visuomotor policies. Importantly, these advantages do not come at the cost of elaborated training strategies like those used in consistency-based models. In fact, our models outperformed CP, while being notably easier and more efficient to train. Regarding the difference between SRFMP and RFMP, the results did not show significant performance gains in terms of success rate and prediction smoothness, at the exception of the long-horizon task of the FRANKA KITCHEN benchmark, where SRFMP significantly outperformed RFMP and DP. Nevertheless, SRFMP shows to be easier to train, achieving higher success rate than RFMP for fewer training epochs (e.g., in LIFT, CAN, SQUARE tasks).

## VI. CONCLUSION

This paper introduced Stable Riemannian Flow Matching Policy (SRFMP), a novel framework that combines the easy training of flow matching with stability-based robustness properties for visuomotor policy learning. SRFMP builds on our extension of stable flow matching to Riemannian manifolds, providing stable convergence of the learned flow to the support of Riemannian target distributions. Our simulated and realworld experiments show that both our previous work on RFMP and its stable counterpart SRFMP outperform diffusion policies and distillation-based extensions, while offering advantages in terms of inference speed, ease of training, and robust performance even with limited NFEs and training epochs. Future work will focus on exploring equivariant policy structures to potentially reduce the number of required demonstrations and to improve generalization. Additionally, we aim to investigate multi-modal perception backbones for tackling contact-rich tasks.

## REFERENCES

[1] J. Urain, A. Mandlekar, Y. Du, M. Shafiullah, D. Xu, K. Fragkiadaki, G. Chalvatzaki, and J. Peters, “Deep generative models in robotics: A survey on learning from multimodal demonstrations,” arXiv preprint arXiv:2408.04380, 2024.

[2] C. Chi, S. Feng, Y. Du, Z. Xu, E. Cousineau, B. Burchfiel, and S. Song, “Diffusion policy: Visuomotor policy learning via action diffusion,” in Robotics: Science and Systems (R:SS), 2023.

[3] A. Prasad, K. Lin, J. Wu, L. Zhou, and J. Bohg, “Consistency policy: Accelerated visuomotor policies via consistency distillation,” in Robotics: Science and Systems (R:SS), 2024.

[4] M. Reuss, M. Li, X. Jia, and R. Lioutikov, “Goal-conditioned imitation learning using score-based diffusion policies,” in Robotics: Science and Systems (R:SS), 2023.

[5] S.-F. Chen, H.-C. Wang, M.-H. Hsu, C.-M. Lai, and S.-H. Sun, “Diffusion model-augmented behavioral cloning,” in Intl. Conf. on Machine Learning (ICML), 2024.

[6] Y. Ze, G. Zhang, K. Zhang, C. Hu, M. Wang, and H. Xu, “3D diffusion policy: Generalizable visuomotor policy learning via simple 3D representations,” in Robotics: Science and Systems (R:SS), 2024.

[7] J. Yang, Z. Cao, C. Deng, R. Antonova, S. Song, and J. Bohg, “Equibot: SIM(3)-equivariant diffusion policy for generalizable and data efficient learning,” in Conference on Robot Learning (CoRL), 2024.

[8] P. Florence, C. Lynch, A. Zeng, O. A. Ramirez, A. Wahid, L. Downs, A. Wong, J. Lee, I. Mordatch, and J. Tompson, “Implicit behavioral cloning,” in Conference on Robot Learning (CoRL), 2022.

[9] N. M. Shafiullah, Z. Cui, A. A. Altanzaya, and L. Pinto, “Behavior transformers: Cloning k modes with one stone,” in Neural Information Processing Systems (NeurIPS), 2022.

[10] A. Mandlekar, D. Xu, J. Wong, S. Nasiriany, C. Wang, R. Kulkarni, L. Fei-Fei, S. Savarese, Y. Zhu, and R. Mart´ın-Mart´ın, “What matters in learning from offline human demonstrations for robot manipulation,” in Conference on Robot Learning (CoRL), 2022.

[11] C. Luo, “Understanding diffusion models: A unified perspective,” arXiv preprint arXiv2208.11970, 2022.

[12] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” in Neural Information Processing Systems (NeurIPS), vol. 33, 2020, pp. 6840–6851.

[13] J. Song, C. Meng, and S. Ermon, “Denoising diffusion implicit models,” in Intl. Conf. on Learning Representations (ICLR), 2020.

[14] C.-W. Huang, M. Aghajohari, J. Bose, P. Panangaden, and A. Courville, “Riemannian diffusion models,” in Neural Information Processing Systems (NeurIPS), 2022.

[15] Y. Lipman, R. T. Chen, H. Ben-Hamu, M. Nickel, and M. Le, “Flow matching for generative modeling,” in Intl. Conf. on Learning Representations (ICLR), 2022.

[16] R. T. Chen and Y. Lipman, “Riemannian flow matching on general geometries,” in Intl. Conf. on Learning Representations (ICLR), 2023.

[17] M. Braun, N. Jaquier, L. Rozo, and T. Asfour, “Riemannian flow matching policy for robot motion learning,” in IEEE/RSJ Intl. Conf. on Intelligent Robots and Systems (IROS), 2024, pp. 5144–5151.

[18] A. Lemme, Y. Meirovitch, M. Khansari-Zadeh, T. Flash, A. Billard, and J. J. Steil, “Open-source benchmarking for learned reaching motion generation in robotics,” Paladyn, Journal of Behavioral Robotics, vol. 6, no. 1, 2015.

[19] C. I. Sprague, A. Elofsson, and H. Azizpour, “Stable autonomous flow matching,” arXiv preprint arXiv2402.05774, 2024.

[20] ——, “Incorporating stability into flow matching,” in ICML Workshop on Structured Probabilistic Inference & Generative Modeling, 2024.

[21] J. LaSalle, “Some extensions of Liapunov’s second method,” IRE Transactions on Circuit Theory, vol. 7, no. 4, pp. 520–527, 1960.

[22] G. Papamakarios, E. Nalisnick, D. J. Rezende, S. Mohamed, and B. Lakshminarayanan, “Normalizing flows for probabilistic modeling and inference,” Journal of Machine Learning Research, vol. 22, no. 57, pp. 1–64, 2021.

[23] M. A. Rana, A. Li, D. Fox, B. Boots, F. Ramos, and N. Ratliff, “Euclideanizing flows: Diffeomorphic reduction for learning stable dynamical systems,” in Learning for Dynamics and Control (L4DC). PMLR, 2020, pp. 630–639.

[24] S. A. Khader, H. Yin, P. Falco, and D. Kragic, “Learning stable normalizing-flow control for robotic manipulation,” in IEEE Intl. Conf. on Robotics and Automation (ICRA), 2021, pp. 1644–1650.

[25] J. Urain, M. Ginesi, D. Tateo, and J. Peters, “Imitationflow: Learning deep stable stochastic dynamic systems by normalizing flows,” in IEEE/RSJ Intl. Conf. on Intelligent Robots and Systems (IROS), 2020, pp. 5231–5237.

[26] J. Urain, D. Tateo, and J. Peters, “Learning stable vector fields on lie groups,” IEEE Robotics and Automation Letters, vol. 7, no. 4, pp. 12 569–12 576, 2022.

[27] J. Zhang, H. Beik Mohammadi, and L. Rozo, “Learning Riemannian stable dynamical systems via diffeomorphisms,” in Conference on Robot Learning (CoRL), 2022.

[28] L. Yang, Z. Zhang, Y. Song, S. Hong, R. Xu, Y. Zhao, W. Zhang, B. Cui, and M.-H. Yang, “Diffusion models: A comprehensive survey of methods and applications,” ACM Comput. Surv., vol. 56, no. 4, 2023.

[29] M. Janner, Y. Du, J. B. Tenenbaum, and S. Levine, “Planning with diffusion for flexible behavior synthesis,” in Intl. Conf. on Machine Learning (ICML), 2022.

[30] Z. Wang, J. J. Hunt, and M. Zhou, “Diffusion policies as an expressive policy class for offline reinforcement learning,” in Intl. Conf. on Learning Representations (ICLR), 2023.

[31] Y. Song, P. Dhariwal, M. Chen, and I. Sutskever, “Consistency models,” in Intl. Conf. on Machine Learning (ICML), 2023.

[32] G. Lu, Z. Gao, T. Chen, W. Dai, Z. Wang, and Y. Tang, “Manicm: Realtime 3d diffusion policy via consistency model for robotic manipulation,” arXiv preprint arXiv:2406.01586, 2024.

[33] Z. Wang, Z. Li, A. Mandlekar, Z. Xu, J. Fan, Y. Narang, L. Fan, Y. Zhu, Y. Balaji, M. Zhou et al., “One-step diffusion policy: Fast visuomotor policies via diffusion distillation,” in Intl. Conf. on Learning Representations (ICLR), 2025.

[34] A. Tong, K. Fatras, N. Malkin, G. Huguet, Y. Zhang, J. Rector-Brooks, G. Wolf, and Y. Bengio, “Improving and generalizing flowbased generative models with minibatch optimal transport,” Trans. on Machine Learning Research (TMLR), 2024.

[35] X. Liu, C. Gong, and Q. Liu, “Flow straight and fast: Learning to generate and transfer data with rectified flow,” in Intl. Conf. on Learning Representations (ICLR), 2022.

[36] A. J. Bose, T. Akhound-Sadegh, K. Fatras, G. Huguet, J. Rector-Brooks, C.-H. Liu, A. C. Nica, M. Korablyov, M. Bronstein, and A. Tong, “SE(3)-stochastic flow matching for protein backbone generation,” in Intl. Conf. on Learning Representations (ICLR), 2023.

[37] A. Davtyan, S. Sameni, and P. Favaro, “Efficient video prediction via sparsely conditioned flow matching,” in Intl. Conf. on Computer Vision (ICCV), 2023, pp. 23 263–23 274.

[38] V. T. Hu, W. Yin, P. Ma, Y. Chen, B. Fernando, Y. M. Asano, E. Gavves, P. Mettes, B. Ommer, and C. G. M. Snoek, “Motion flow matching for human motion synthesis and editing,” arXiv preprint arXiv:2312.08895, 2023.

[39] X. Zhang, Y. Pu, Y. Kawamura, A. Loza, Y. Bengio, D. L. Shung, and A. Tong, “Trajectory flow matching with applications to clinical time series modeling,” in Neural Information Processing Systems (NeurIPS), 2024.

[40] H. Lin, O. Zhang, H. Zhao, D. Jiang, L. Wu, Z. Liu, Y. Huang, and S. Z. Li, “PPFlow: Target-aware peptide design with torsional flow matching,” in Intl. Conf. on Machine Learning (ICML), 2024.

[41] A. H. Liu, M. Le, A. Vyas, B. Shi, A. Tjandra, and W.-N. Hsu, “Generative pre-training for speech with flow matching,” in Intl. Conf. on Learning Representations (ICLR), 2024.

[42] N. Funk, J. Urain, J. Carvalho, V. Prasad, G. Chalvatzaki, and J. Peters, “Actionflow: Efficient, accurate, and fast policies with spatially symmetric flow matching,” in R:SS workshop: Structural Priors as Inductive Biases for Learning Robot Dynamics, 2024.

[43] Q. Rouxel, A. Ferrari, S. Ivaldi, and J.-B. Mouret, “Flow matching imitation learning for multi-support manipulation,” in IEEE/RAS Intl. Conf. on Humanoid Robots (Humanoids), 2024, pp. 528–535.

[44] E. Chisari, N. Heppert, M. Argus, T. Welschehold, T. Brox, and A. Valada, “Learning robotic manipulation policies from point clouds with conditional flow matching,” in Conference on Robot Learning (CoRL), 2024.

[45] M. do Carmo, Riemannian Geometry. Birkhauser Basel, 1992.¨

[46] J. M. Lee, Introduction to Riemannian manifolds. Springer, 2018, vol. 2.

[47] P.-A. Absil, R. Mahony, and R. Sepulchre, Optimization Algorithms on Matrix Manifolds. Princeton University Press, 2007.

[48] N. Boumal, An introduction to optimization on smooth manifolds. Cambridge University Press, 2023.

[49] R. T. Chen, Y. Rubanova, J. Bettencourt, and D. K. Duvenaud, “Neural ordinary differential equations,” in Neural Information Processing Systems (NeurIPS), 2018.

[50] K. Atkinson, An introduction to numerical analysis. John wiley & sons, 1991.

[51] J. M. Selig, Geometric fundamentals of robotics. Springer Science & Business Media, 2007.

[52] F. Merat, “Introduction to robotics: Mechanics and control,” IEEE Journal on Robotics and Automation, vol. 3, no. 2, pp. 166–166, 1987.

[53] X. Pennec, “Intrinsic statistics on Riemannian manifolds: Basic tools for geometric measurements,” Journal of Mathematical Imaging and Vision, vol. 25, pp. 127–154, 2006.

[54] F. Galaz-Garcia, M. Papamichalis, K. Turnbull, S. Lunagomez, and E. Airoldi, “Wrapped distributions on homogeneous Riemannian manifolds,” arXiv preprint arXiv:2204.09790, 2022.

[55] D. Q. Mayne and H. Michalska, “Receding horizon control of nonlinear systems,” in IEEE Conference on Decision and Control (CDC), 1988, pp. 464–465.

[56] J. P. La Salle, “An invariance principle in the theory of stability,” Tech. Rep., 1966.

[57] X. Mao, “Stochastic versions of the LaSalle theorem,” Journal of differential equations, vol. 153, no. 1, pp. 175–195, 1999.

[58] J. Fu, A. Kumar, O. Nachum, G. Tucker, and S. Levine, “D4RL: Datasets for deep data-driven reinforcement learning,” 2020.

[59] E. Perez, F. Strub, H. De Vries, V. Dumoulin, and A. Courville, “Film: Visual reasoning with a general conditioning layer,” in AAAI Conf. on Artificial Intelligence, vol. 32, no. 1, 2018.

[60] K. V. Mardia and P. E. Jupp, Distributions on Spheres. John Wiley and Sons, Ltd, 1999, ch. 9, pp. 159–192.

[61] M. Kollovieh, M. Lienen, D. Ludke, L. Schwinn, and S. G¨ unnemann,¨ “Flow matching with Gaussian process priors for probabilistic time series forecasting,” in Intl. Conf. on Learning Representations (ICLR), 2025.

[62] H. Ben-Hamu, O. Puny, I. Gat, B. Karrer, U. Singer, and Y. Lipman, “D-flow: Differentiating through flows for controlled generation,” in Intl. Conf. on Machine Learning (ICML), 2024.

[63] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in Intl. Conf. on Learning Representations (ICLR), 2019.

[64] B. T. Polyak and A. B. Juditsky, “Acceleration of stochastic approximation by averaging,” SIAM journal on control and optimization, vol. 30, no. 4, pp. 838–855, 1992.

[65] N. Jaquier, L. Rozo, and T. Asfour, “Unraveling the single tangent space fallacy: An analysis and clarification for applying Riemannian geometry in robot learning,” in IEEE Intl. Conf. on Robotics and Automation (ICRA), 2024, pp. 242–249.

[66] O. G. Younis, R. Perez-Vicente, J. U. Balis, W. Dudley, A. Davey, and J. K. Terry, “Minari,” Sep. 2024. [Online]. Available: https: //doi.org/10.5281/zenodo.13767625

![](images/105d4f08f974472af31efab9e0a1f951983e2029c3d9b77e5dcc33c99ba95c9d.jpg)

Haoran Ding received the B.S. degree in Engineering from Tongji University, Shanghai, China, in 2021, and the M.Sc. degree in Computational Engineering from the Technical University of Darmstadt, Germany, in 2024. During his master’s studies, he worked on the robot air hockey project at the Intelligent Autonomous Systems (IAS) Group and completed a one-year research internship at the Bosch Center for Artificial Intelligence (BCAI), where he conducted his master thesis. He is currently pursuing a Ph.D. degree in the Robotics Department at Mohamed bin Zayed University of Artificial Intelligence (MBZUAI), Abu Dhabi, United Arab Emirates. His research interests include visuomotor policy learning, generative models and imitation learning.

![](images/d7437b6051ae74b44740a26ebf40074008ca52ff7150ef306e09db75ec80abd8.jpg)

![](images/6a8165ea866f9f92993ade83d0cc384ef6ce8324e994de081c2bf2604c57ecdd.jpg)

Noemie Jaquier ´ is an assistant professor at the KTH Royal Institute of Technology, where she heads the Geometric Robot (GeoRob) Lab at the Division of Robotics, Perception and Learning. She received a B.Sc. degree on Microengineering, a M.Sc. degree in Robotics and Autonomous Systems from the Ecole Polytechnique Fed´ erale de Lausanne (EPFL)´ in 2014 and 2016. She received her PhD degree from EPFL in 2020 for her thesis “Robot Skills Learning with Riemannian Manifolds: Leveraging Geometry-awareness in Robot Learning, Optimiza-

tion and Control”. Prior to joining KTH, Noemie Jaquier was a postdoctoral´ researcher in the High Performance Humanoid Technologies Lab (H²T) at the Karlsruhe Institute of Technology (KIT) and a visiting postdoctoral scholar at the Stanford Robotics Lab. Her research investigates data-efficient and theoretically-sound learning algorithms that leverage differential geometryand physics-based inductive bias to endow robots with close-to-human learning and adaptation capabilities. Personal webpage: http://njaquier.ch.

the Polytechnical University of Catalonia in 2013. Prior to joining BCAI in 2018, Leonel Rozo led the Learning and Interaction Group at the department of Advanced Robotics in the Italian Institute of Technology (IIT) from 2016 to 2018. He previously joined IIT in 2012, first as a research fellow and then as postdoctoral researcher in 2013. In 2017 he was awarded an individual Marie Skłodowska-Curie fellowship for his project proposal DRAPer. His research has been mainly focused on exploiting machine learning techniques, optimal control, and Riemannian manifold theory to learn robot motion skills via human demonstrations and refinement methods such as reinforcement learning with applications to (dual-arm) manipulation tasks and human-robot collaboration. Personal webpage: https://leonelrozo.weebly.com.

![](images/0ad989e6e8a26a9ab37afee6a1641e81c29f0ae95075af643e77c1c0f2b7e9e7.jpg)

Jan Peters is a full professor (W3) for Intelligent Autonomous Systems at the Computer Science Department of the Technische Universitaet Darmstadt since 2011, and, at the same time, he is the dept head of the research department on Systems AI for Robot Learning (SAIROL) at the German Research Center for Artificial Intelligence (Deutsches Forschungszentrum fur K¨ unstliche Intelligenz, DFKI) since 2022.¨ He is also is a founding research faculty member of the Hessian Center for Artificial Intelligence. Jan Peters has received the Dick Volz Best 2007 US PhD

Thesis Runner-Up Award, the Robotics: Science & Systems - Early Career Spotlight, the INNS Young Investigator Award, and the IEEE Robotics & Automation Society’s Early Career Award as well as numerous best paper awards. In 2015, he received an ERC Starting Grant and in 2019, he was appointed IEEE Fellow, in 2020 ELLIS fellow and in 2021 AAIA fellow. Jan Peters has studied Computer Science, Electrical, Mechanical and Control Engineering at TU Munich and FernUni Hagen in Germany, at the National University of Singapore (NUS) and the University of Southern California (USC). He has received four Master’s degrees in these disciplines as well as a Computer Science PhD from USC. Jan Peters has performed research in Germany at DLR, TU Munich and the Max Planck Institute for Biological Cybernetics (in addition to the institutions above), in Japan at the Advanced Telecommunication Research Center (ATR), at USC and at both NUS and Siemens Advanced Engineering in Singapore. He has led research groups on Machine Learning for Robotics at the Max Planck Institutes for Biological Cybernetics (2007-2010) and Intelligent Systems (2010-2021).

Leonel Rozo is a lead research scientist at the Bosch Center for Artificial Intelligence (BCAI), Germany. He received a Bachelor degree in Mechatronics Engineering from the “Nueva Granada” Military University in Bogota, Colombia in 2005, and a Master´ degree in Automatic Control and Robotics from the Polytechnical University of Catalonia, Barcelona, Spain, in 2007. He carried out his PhD research at the Institut de Robotica i Inform\` atica Industrial\` under the supervision of Prof. Carme Torras and Dr. Pablo Jimenez, and received a Ph.D in Robotics from´