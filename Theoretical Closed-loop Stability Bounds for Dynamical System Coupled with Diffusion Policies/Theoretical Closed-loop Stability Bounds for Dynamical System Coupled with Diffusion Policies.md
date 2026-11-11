# Theoretical Closed-loop Stability Bounds for Dynamical System Coupled with Diffusion Policies

Gabriel Lauzier, Alexandre Girard<sup>1</sup> and Franc¸ois Ferland<sup>2</sup>

Abstract— Diffusion policy has shown great performance in robotic manipulation tasks under stochastic perturbations, due to its ability to model multimodal action distributions. Nonetheless, its reliance on a computationally expensive reverse-time diffusion (denoising) process, for action inference, makes it challenging to use for real-time applications where quick decisionmaking is mandatory. This work studies the possibility of conducting the denoising process only partially before executing an action, allowing the plant to evolve according to its dynamics in parallel to the reverse-time diffusion dynamics ongoing on the computer. In a classical diffusion policy setting, the plant dynamics are usually slow and the two dynamical processes are uncoupled. Here, we investigate theoretical bounds on the stability of closed-loop systems using diffusion policies when the plant dynamics and the denoising dynamics are coupled. The contribution of this work gives a framework for faster imitation learning and a metric that yields if a controller will be stable based on the variance of the demonstrations.

## I. INTRODUCTION

Robotic control and decision making increasingly rely on generative models to capture complex and multimodal action distributions. Vision-Language-Action (VLA) models [1], [2] have demonstrated great performance in manipulation tasks and the interest in this field is growing with the uprising of large language models. The benefits of such models are the usage of pretrained models on large, Internet-scale datasets that allow the use of generic architecture [1]. On the other hand, such datasets and training infrastructure are often hard to obtain, resulting in closed-source models hard to fine-tune for new platforms. Other works leveraging diffusion policies [3], [4] have shown efficient learning that requires fewer expert demonstrations, allowing someone to easily build a dataset for a given task. Despite its efficient learning, the hyperparameters of the model are often convoluted [5] and given the iterative nature of diffusion models, it requires several steps of inference, making them challenging to use for real-time applications.

In this work, we propose a partial diffusion policy that executes only one step of the denoising diffusion process before taking action in the environment. Rather than expressing the system made of the plant dynamics and the denoising diffusion process as two decoupled systems, we analyze its stability as a coupled dynamical system. This allows us to show that the full denoising diffusion process from

Diffusion Policy and the proposed partial diffusion policy are equivalent in stability depending on the variance of the expert demonstrations, the diffusion coefficient and the timescale ratio between the plant dynamics and the controller.

The main contribution is a theoretical bound for asymptotic stability for a linear time-invariant plant under full and partial diffusion policies. This gives a metric for estimating the quality of a dataset given the variance of the expert demonstrations, the responses of the environment and the parametrization of the diffusion-based policy. It also shows a framework for trading off denoising depth against latency given proper variance in the expert demonstrations. This understanding of the diffusion-based policy brings such a framework closer to real-time robotics.

## II. RELATED WORKS

## A. Score-Based Generative Modeling

Song et al. [6] introduced a mathematical framework for denoising diffusion models that leverage stochastic differential equations (SDEs). Unlike likelihood-based methods, the suggested approach doesn’t need to learn the partition function of the Boltzmann/Gibbs distribution by approximating the score function of the distribution with a neural network and a score matching objective [7]–[9]. The process of learning and inference is defined respectively by two SDEs: the forward equation and the reverse-time equation:

$$
d \mathbf {x} = \mathbf {f} (\mathbf {x}, \tau) d \tau + \mathbf {g} (\mathbf {x}, \tau) d \mathbf {w}\tag{1}
$$

$$
d \mathbf {x} = \left[ \mathbf {f} (\mathbf {x}, \tau) - \mathbf {g} ^ {2} (\tau) s _ {\theta} (\mathbf {x}, \tau) \right] d \tau + \mathbf {g} (\tau) d \hat {\mathbf {w}}\tag{2}
$$

where w is a standard Brownian motion, f is the drift coefficient, g is the diffusion coefficient and $\mathbf { s } _ { \theta }$ is the approximation of the score function, the gradient of the logprobability. The reverse-time SDE in equation 2 is then used to generate samples from the learned prior distribution. The discretization of variance-preserving SDE has been shown to yield Denoising Diffusion Probabilistic Models (DDPM) [6], [10], [11].

## B. Diffusion Policy

Diffusion Policy [3] applies a conditional DDPM model to learn a policy observed from expert demonstrations. This is done by learning the score function of the conditional distribution over actions. Then, at inference time, it iteratively optimizes the score function through a series of stochastic Langevin dynamics steps to generate a sequence of actions within a given horizon. In this work, the model is conditioned on visual observations.

By choosing to predict a sequence of actions instead of a single action, Diffusion Policy reduces jitter in the policy due to different valid modes between steps. It also allows handling idle actions where the agent shouldn’t move, which is often difficult for single-step policies.

Another main advantage against other imitation learning methods is its capacity to model multimodal action distributions because of the stochastic nature of DDPM. This results directly in better performance with position control than with velocity control.

For a simple task in a linear dynamic system in standard state-space form where the demonstrations are given by a linear feedback policy: $\mathbf { u } _ { t } = - \mathbf { K } \mathbf { x } _ { t }$ , the DDPM sampling will converge to the expert policy given a sequence of actions of length one. With a bigger horizon, this policy will converge to $\mathbf { u } _ { t + t ^ { \prime } } = - \mathbf { K } ( \mathbf { A } - \mathbf { B } \mathbf { K } ) ^ { t ^ { \prime } } \mathbf { x } _ { t } .$ , which means that the policy must implicitly learn the dynamics of the system $\mathbf { A } - \mathbf { B } \mathbf { K }$

## III. METHOD

First, we define a linear time-invariant dynamic system in a standard state-space form:

$$
\dot {\mathbf {x}} = \mathbf {A} \mathbf {x} + \mathbf {B} \mathbf {u}\tag{3}
$$

Like Diffusion Policy, we aim to obtain demonstrations from a linear feedback policy: $\mathbf { u } _ { t } = - \mathbf { K } \mathbf { x } _ { t }$ . Then we want a surrogate policy that imitates the expert behavior. Instead of using DDPM like Diffusion Policy, we use the score-based generative modeling framework [6] to model the denoising process. Equation 2 is used for this purpose. Within this equation, the score function in the drift coefficient could be seen as a vector field that guides toward the learned prior probability density function. This last idea is what motivates this work. Instead of doing the full denoising process to sample an action, we directly take an action towards the direction of the most probable action. In a closedloop system, this yields:

$$
\left[ \begin{array}{c} d \mathbf {x} = [ \mathbf {A x} + \mathbf {B u} ] d t \\ d \mathbf {u} = [ \mathbf {f} (\mathbf {u}, \tau) + \mathbf {g} ^ {2} (\tau) \mathbf {s} _ {\theta} (\mathbf {u}, \tau) ] d \tau + \mathbf {g} (\tau) d \hat {\mathbf {w}} _ {\tau} \end{array} \right]\tag{4}
$$

Where τ if the diffusion time and the score function ${ \bf s } _ { \theta } ( { \bf u } , \tau )$ is defined as the gradient of the log-probability of the expert policy with respect to the action u<sub>t</sub>:

$$
\mathbf {s} (\mathbf {u}) = \nabla_ {\mathbf {u}} \log p (\mathbf {u})\tag{5}
$$

If the expert demonstration is deterministic, as we defined earlier, the score function will be ill-defined and could be interpreted as a Dirac delta function $\delta ( \mathbf { u } - \mathbf { u } _ { 0 } \exp ( \mathbf { A } - \mathbf { B } \mathbf { K } ) t )$ which is a distribution that is zero everywhere except at the point $\mathbf { u } = \mathbf { u } _ { 0 } \exp ( ( \mathbf { A } - \mathbf { B } \mathbf { K } ) t )$ . This means that anywhere other than the expert policy trajectory, there is no direction to go towards. This is a problem if we want to use the score function to guide the system towards the most probable action. To this matter, we model the expert policy as a stochastic policy using a Gaussian distribution centered on the deterministic policy:

![](images/c6e7b5e0845fd288969d6e0fef58c7c76262961bf5e41df91e86ec10bfdd436a.jpg)  
Fig. 1. Closed-loop dynamic system of a linear plant controlled by the denoising dynamics of a score-based generative model imitating a linear feedback policy.

$$
p (u) = \frac {1}{\sqrt {2 \pi \sigma^ {2}}} \exp \left(- \frac {(u + K x) ^ {2}}{2 \sigma^ {2}}\right)\tag{6}
$$

With this probability density function, we can compute the score function as:

$$
\mathbf {s} (\mathbf {u}) = - \frac {1}{\sigma^ {2}} \left(\mathbf {u} + \mathbf {K x}\right)\tag{7}
$$

Notice that in figure 1 instead of modeling the controller like in equation 4 we only use the score function. The effect of the drift function $\mathbf { f } ( \mathbf { u } , t )$ is a translation of the solution of the equation. While the diffusion coefficient of the deterministic term ${ \bf g } ^ { 2 } ( t )$ could be seen as a gain on the step towards the most probable step. In the next subsection neither of them along with the stochastic term of the reverse-time SDE will be used to simplify the demonstration of stability. Also, note that since the score function ${ \bf s } _ { \theta } ( { \bf u } , \tau )$ is dependent on the diffusion time τ but the expression of the stochastic policy isn’t the joint probability of action and time, again, for simplicity. We will elaborate on the effect of the time diffusion and the diffusion coefficient on the stability in the subsection III-B.

## A. Stability Analysis

1) 1-Dimensional Case: The system is stable if the eigenvalues are in the left half-plane. To this matter, we define an augmented system as:

$$
\mathbf {x} ^ {+} = \left[ \begin{array}{c} x \\ u \end{array} \right] \quad \lambda = \frac {1}{\sigma^ {2}}
$$

$$
\dot {x} ^ {+} = \left[ \begin{array}{c} \dot {x} \\ \dot {u} \end{array} \right] = \left[ \begin{array}{c} A x + B u \\ - \lambda K x - \lambda u \end{array} \right] = \left[ \begin{array}{c c} A & B \\ - \lambda K & - \lambda \end{array} \right] \left[ \begin{array}{c} x \\ u \end{array} \right]\tag{8}
$$

which gives the augmented state matrix $A ^ { + }$ as:

$$
A ^ {+} = \left[ \begin{array}{c c} A & B \\ - \lambda K & - \lambda \end{array} \right]
$$

By testing the inequality of the eigenvalues, we can see that the system is stable if the eigenvalues are in the left halfplane. This can be done by testing the following inequalities:

$$
0 > A - \lambda \pm \sqrt {(A - \lambda) ^ {2} - 4 \lambda (B K - A)}
$$

which gives, by substituting for λ , the following inequalities:

$$
0 > A - B K\tag{9}
$$

$$
\sigma <   \frac {1}{\sqrt {A}}, \quad A > 0\tag{10}
$$

Where equation 9 is simply the stability of an LTI system under a linear feedback control, and where equation 10 is the stability condition related to the variance of the expert demonstrations. This means that the score-based closed-loop control is unstable if the demonstrator is unstable, and when the demonstrator is stable, the learned policy is only stable if the variance of the demonstrations is smaller than the inverse response of the system. This satisfies the hypothesis that the system is as good as the expert policy, given the quality of the demonstrations.

TABLE I  
STABILITY BOUNDS OF THE CLOSED-LOOP LINEAR TIME-INVARIANT DYNAMICS COUPLED WITH A DIFFUSION POLICY.

<table><tr><td>Natural Dynamic</td><td>Closed-Loop Dynamic</td><td>Diffusion gain</td><td>Stability</td></tr><tr><td rowspan="2">A&gt;0</td><td> $\mathbf{A} - \mathbf{BK} \geq 0$ </td><td>for all  $\mathbf{K}'$ </td><td>unstable</td></tr><tr><td> $\mathbf{A} - \mathbf{BK} < 0$ </td><td> $\mathbf{K}' \geq \mathbf{A}$ </td><td>stable</td></tr><tr><td>A&lt;0</td><td>for all  $\mathbf{A} - \mathbf{BK}$ </td><td> $\mathbf{K}' \geq 0$ </td><td>stable</td></tr></table>

2) N-Dimensional Case: The same idea applies to the N-dimensional case, except that instead of computing the eigenvalues of the state matrix in the state representation, we can elevate the system to a second-order ordinary differential equation and show that the second and third coefficients are positive definite, such as a matrix M with respect to $\mathbf { x } ^ { \top } \mathbf { M } x > 0$ for all $\mathbf { x } \in \mathbb { R } ^ { N }$ where $\mathbf { x } \neq 0 \mathbf { : }$

$$
\ddot {\mathbf {x}} - \left[ \mathbf {A} - \boldsymbol {\Sigma} ^ {- 1} \right] \dot {\mathbf {x}} - \boldsymbol {\Sigma} ^ {- 1} \left[ \mathbf {A} - \mathbf {B K} \right] \mathbf {x} = 0\tag{11}
$$

$$
- \left[ \mathbf {A} - \boldsymbol {\Sigma} ^ {- 1} \right] \succ 0 \quad \text { and } \quad - \boldsymbol {\Sigma} ^ {- 1} \left[ \mathbf {A} - \mathbf {B K} \right] \succ 0
$$

Where Σ is the covariance matrix of the demonstrations, which is assumed to be linearly independent.

We know that a matrix M is positive definite if and only if all its eigenvalues are positive given that M is symmetric. Since our matrix might not be symmetric, we can use only its symmetric part $\begin{array} { r } { \mathbf { \bar { M } ^ { \prime } } = \frac { 1 } { 2 } ( \mathbf { M } + \mathbf { M } ^ { \top } ) } \end{array}$ and show that the eigenvalues of M<sup>′</sup> are positive, as said earlier. With some simplification shown in the appendix I, we show that the eigenvalues of the symmetric state matrix A and the symmetric state matrix under linear feedback control $\mathbf { A } - \mathbf { B } \mathbf { K }$ must respect the following inequalities given isotropic distributions of the demonstrations:

$$
\sigma <   \frac {1}{\sqrt {\lambda_ {\max} (\mathbf {S _ {1}})}}\tag{12}
$$

$$
\lambda_ {\max} (\mathbf {S _ {2}}) <   0\tag{13}
$$

For $\lambda _ { \operatorname* { m a x } } ( \mathbf { S } _ { 1 } )$ being the maximum eigenvalue of the sym metric matrix $\frac { 1 } { 2 } ( \mathbf { A } + \mathbf { A } ^ { \top } )$ and $\lambda _ { \operatorname* { m a x } } ( \mathbf { S } _ { 2 } )$ being the maximum eigenvalue of the symmetric matrix $\frac { 1 } { 2 } \big ( \big ( \mathbf { A } - \mathbf { B } \mathbf { K } \big ) +$ $\left( \mathbf { A } - \mathbf { B } \mathbf { K } \right) ^ { \top } )$ . This result matches the 1-dimensional case shown earlier.

Augmented Phase Plane (x, u)  
![](images/3afa3c92d189abb96ca62e403eea0f42dbb8252d8e94fa4bae8374b9cfd54161.jpg)  
Fig. 2. Trajectories of the expert policy and the score-based controller going from $x = 0$ to $x = 5$ under different gain $\mathbf { K } ^ { \prime }$ in the augmented phase plane of a closed-loop system with a linear dynamics plant parametrized with $A = 2 , B = 1 , K = 3$ . Given these parameters and the equation 15 we can see that the system starts being unstable when $\mathbf { K } ^ { \prime } < A$

## B. The Effect of Different Time Scale

Earlier we made the assumption that the derivative of the control law u˙ was made only with the score function at the same time scale as the plant $( d t = d \tau )$ . In its original work, Diffusion Policy uses a different time-scale and performs the action dynamics in an inner loop within plant dynamics $( d t \neq$ $d \tau )$ . In this section, we show that the variance, the diffusion coefficient and different time scale act as a proportional gain applied to the error between the current action and the expert demonstration. Using the chain rule, we can rewrite the action dynamics as follows:

$$
{\frac {d \mathbf {u}}{d t}} = {\frac {d \boldsymbol {\tau}}{d t}} {\frac {d \mathbf {u}}{d \boldsymbol {\tau}}} = \alpha {\frac {d \mathbf {u}}{d \boldsymbol {\tau}}}
$$

$$
d \mathbf {u} = \left[ \mathbf {f} (\mathbf {u}, \alpha t) + \mathbf {g} ^ {2} (\alpha t) \mathbf {s} _ {\theta} (\mathbf {u}, \alpha t) \right] \alpha d t + \sqrt {\alpha} \mathbf {g} (\alpha t) d \hat {\mathbf {w}} _ {\mathbf {t}}\tag{14}
$$

Where the standard Brownian motion $\begin{array} { r l } { d \hat { \mathbf { w } } _ { \tau } } & { { } = } \end{array}$ $\sqrt { d \tau } \mathcal { N } ( 0 , 1 ) = \sqrt { \alpha d t } \mathcal { N } ( 0 , 1 ) = \sqrt { \alpha } d \hat { \mathbf { w } } _ { \mathbf { t } }$ . By distributing α we see that different time scales have the effect of a gain on the drift coefficient $\mathbf { f } ( \mathbf { u } , \alpha t )$ and on the score function $\mathbf { s } _ { \theta } ( \mathbf { u } , \alpha t )$ . As said earlier, a non-zero drift coefficient doesn’t affect the stability of the system and only translate the solution of the system of equations. Together, the variance, the diffusion coefficient and the time scale ratio between the plant time and the diffusion time are analogous to a proportional gain on the error of the current action such as $- \mathbf { K } ^ { \prime } ( \mathbf { u } + \mathbf { K } \mathbf { x } )$ . If we ignore the stochastic term, such action dynamics yield this condition of stability:

$$
\sigma <   \mathbf {g} \sqrt {\frac {\alpha}{A}}\tag{15}
$$

With that in mind, figure 2 shows the augmented phase plane with $\mathbf { K } ^ { \prime }$ being parametrized by the variance, the diffusion coefficient and the time-scale ratio such as ${ \bf K } ^ { \prime } = $ $\mathbf { g } ^ { 2 }  { \alpha } /  { \sigma } ^ { 2 }$ . The figure 3 expresses the stability region of the diffusion policies given by equation 15.

![](images/981997220e5a6c191a8b5d399673c7085ee085773fe562e07ed838eedd2c6e8f.jpg)  
Fig. 3. Region of stability of the coupled diffusion policies under stable demonstration $A - B K < 0$ where $K ^ { \prime } = \mathbf { \dot { g } } ^ { 2 } \alpha / \sigma ^ { 2 }$

## IV. DISCUSSION

The methods shown allow the formulation of a theoretical bound in equation 15 that can be used to estimate the quality of an imitation learning dataset from the variance in the dataset given the response of the system and the parameterization of the diffusion policy framework. This gives the insight that given a fast-responsive system, the demonstrations should have lower variances and, on the other hand, given a slow responsive system, the demonstrations could have a higher variance and still guarantee the stability of the controller.

The methods also show that a per-step coupling of the Diffusion Policy against its classical multi-step coupling will display the same condition of stability. Diffusion Policy is implemented as an inner loop of the plant dynamics. Albeit being the inner loop, in real world implementation, it doesn’t mean that one or multiple step of the Diffusion Policy is faster than the environment time scale. In practice, the denoising diffusion process is often slower than the dynamics of the environment. It’s even more true given highdimensionality input like an image as condition, which takes more time to process. This is the main limitation of these theoretical bounds for stability, which assume enough time to fit the computation of the action within the dynamics time scale of the plant. Even with this limitation, the per-step coupling method reduces the computation load compared to the multi-steps one, allowing faster inference of the actions. This makes it the closest method that could satisfy the theoretical bounds in real life scenarios.

## V. CONCLUSION

The recent success of diffusion policies with manipulation tasks makes wonder if it’s possible to apply such framework to real-time task like autonomous driving. At first sight, the iterative nature of the diffusion policies inference seems not appropriate for such applications. Here, we explore the stability condition of a linear-time invariant system coupled with a diffusion policy controller. The coupling of both systems allows showing the equality in stability of a partial diffusion policy against a fully iterative diffusion policy, which is bound by the equation 15. This last equation can be used as a quantitative measure of the quality of a dataset given known parameters such as the variance of the expert demonstration, the diffusion coefficient and the time-scale ratio. In last, by showing the equality in stability of a partial diffusion policy, it allows the design of faster inference process with fewer steps.

Looking forward, several extensions to this work could be made. First, this work only takes into account the deterministic behavior of the diffusion policy system. Adding stochasticity to the system will affect the stability bounds. Future work should study the effect of such perturbations on the coupled system. Second, this work could be extended to nonlinear environment and analysis using Lyapunov stability theory, which could result in stability bound conditions generalizable to a broader range of applications. Finally, validation on high-dimensional space such as vision-conditioned tasks would allow a comparison with the original Diffusion Policy and quantify the gain and trade off of using a partial diffusion policy.

## REFERENCES

[1] M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. P. Foster, P. R. Sanketi, Q. Vuong, T. Kollar, B. Burchfiel, R. Tedrake, D. Sadigh, S. Levine, P. Liang, and C. Finn, “OpenVLA: An open-source vision-language-action model,” in 8th Annual Conference on Robot Learning, 2024.

[2] B. Zitkovich, T. Yu, S. Xu, P. Xu, T. Xiao, F. Xia, J. Wu, P. Wohlhart, S. Welker, A. Wahid, Q. Vuong, V. Vanhoucke, H. Tran, R. Soricut, A. Singh, J. Singh, P. Sermanet, P. R. Sanketi, G. Salazar, M. S. Ryoo, K. Reymann, K. Rao, K. Pertsch, I. Mordatch, H. Michalewski, Y. Lu, S. Levine, L. Lee, T.-W. E. Lee, I. Leal, Y. Kuang, D. Kalashnikov, R. Julian, N. J. Joshi, A. Irpan, B. Ichter, J. Hsu, A. Herzog, K. Hausman, K. Gopalakrishnan, C. Fu, P. Florence, C. Finn, K. A. Dubey, D. Driess, T. Ding, K. M. Choromanski, X. Chen, Y. Chebotar, J. Carbajal, N. Brown, A. Brohan, M. G. Arenas, and K. Han, “Rt-2: Vision-language-action models transfer web knowledge to robotic control,” in Proceedings of The 7th Conference on Robot Learning, ser. Proceedings of Machine Learning Research, J. Tan, M. Toussaint, and K. Darvish, Eds., vol. 229. PMLR, 06–09 Nov 2023, pp. 2165–2183.

[3] C. Chi, Z. Xu, S. Feng, E. Cousineau, Y. Du, B. Burchfiel, R. Tedrake, and S. Song, “Diffusion policy: Visuomotor policy learning via action diffusion,” The International Journal of Robotics Research, 2024.

[4] C. Chi, Z. Xu, C. Pan, E. Cousineau, B. Burchfiel, S. Feng, R. Tedrake, and S. Song, “Universal Manipulation Interface: In-The-Wild Robot Teaching Without In-The-Wild Robots,” in Proceedings of Robotics: Science and Systems, Delft, Netherlands, July 2024.

[5] T. Karras, M. Aittala, T. Aila, and S. Laine, “Elucidating the design space of diffusion-based generative models,” in Advances in Neural Information Processing Systems, S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, Eds., vol. 35. Curran Associates, Inc., 2022, pp. 26 565–26 577.

[6] Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole, “Score-based generative modeling through stochastic differential equations,” in International Conference on Learning Representations, Jan. 2021.

[7] A. Hyvarinen, “Estimation of non-normalized statistical models by¨ score matching,” The Journal of Machine Learning Research, vol. 6, pp. 695–709, Dec. 2005.

[8] P. Vincent, “A connection between score matching and denoising autoencoders,” Neural Computation, vol. 23, no. 7, pp. 1661–1674, 2011.

[9] Y. Song and S. Ermon, “Generative modeling by estimating gradients of the data distribution,” in Advances in Neural Information Processing Systems, H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alche-Buc,´ E. Fox, and R. Garnett, Eds., vol. 32. Curran Associates, Inc., 2019.

[10] J. Sohl-Dickstein, E. Weiss, N. Maheswaranathan, and S. Ganguli, “Deep unsupervised learning using nonequilibrium thermodynamics,” in Proceedings of the 32nd International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, F. Bach and D. Blei, Eds., vol. 37. Lille, France: PMLR, 07–09 Jul 2015, pp. 2256–2265.

[11] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” in Proceedings of the 34th International Conference on Neural Information Processing Systems, ser. NIPS ’20. Red Hook, NY, USA: Curran Associates Inc., 2020.

## APPENDIX I EXTENDED DERIVATION OF THE N-DIMENSIONAL STABILITY BOUNDS

Below is the full derivation of 12 and 13, the stability bounds of a closed-loop linear time-invariant dynamics coupled with a diffusion policy. We first elevate the system in state-space form to a second-order ordinary differential equation:

$$
\ddot {\mathbf {x}} = \mathbf {A} \dot {\mathbf {x}} + \mathbf {B} \dot {\mathbf {u}}, \quad \dot {\mathbf {u}} = - \Sigma^ {- 1} (\mathbf {u} + \mathbf {K x}).\tag{16}
$$

Where u˙ is the score function and Σ is the covariance of the demonstrations, which we assume to be linearly independent. With $\mathbf { u } = \mathbf { B } ^ { - 1 } \left( \dot { \mathbf { x } } - \mathbf { A } \mathbf { x } \right)$ obtained from the statespace form, which gives:

$$
\begin{array}{r l} \ddot {\mathbf {x}} & = \mathbf {A} \dot {\mathbf {x}} + \mathbf {B} \dot {\mathbf {u}} = \mathbf {A} \dot {\mathbf {x}} + \mathbf {B} \left(- \Sigma^ {- 1} \mathbf {u} - \Sigma^ {- 1} \mathbf {K x}\right) \\ & = \mathbf {A} \dot {\mathbf {x}} - \Sigma^ {- 1} \mathbf {B K x} - \Sigma^ {- 1} \mathbf {B u} \\ & = \mathbf {A} \dot {\mathbf {x}} - \Sigma^ {- 1} \mathbf {B K x} - \Sigma^ {- 1} \mathbf {B B} ^ {- 1} (\dot {\mathbf {x}} - \mathbf {A x}) \\ & = \mathbf {A} \dot {\mathbf {x}} - \Sigma^ {- 1} \mathbf {B K x} - \Sigma^ {- 1} \dot {\mathbf {x}} + \Sigma^ {- 1} \mathbf {A x} \end{array}
$$

$$
\ddot {\mathbf {x}} - \left[ \mathbf {A} - \boldsymbol {\Sigma} ^ {- 1} \right] \dot {\mathbf {x}} - \boldsymbol {\Sigma} ^ {- 1} \left[ \mathbf {A} - \mathbf {B K} \right] \mathbf {x} = 0.\tag{17}
$$

To be stable, the second and third coefficients of this second-order system must be positive definite, such as a matrix M with respect to $x ^ { \top } \mathbf { M } x > 0$ for all $\boldsymbol { x } \in \mathbb { R } ^ { N }$ , where $x \neq 0$ . This leads to the following condition:

$$
- \left[ \mathbf {A} - \boldsymbol {\Sigma} ^ {- 1} \right] \succ 0 \quad \text { and } \quad - \boldsymbol {\Sigma} ^ {- 1} \left[ \mathbf {A} - \mathbf {B K} \right] \succ 0.\tag{18}
$$

We know that M is positive definite if and only if all its eigenvalues are positive given that M is symmetric. Since our matrix might not be symmetric, we can use only its symmetric part $\begin{array} { r } { \mathbf { \bar { M } ^ { \prime } } = \frac { 1 } { 2 } ( \mathbf { M } + \mathbf { M } ^ { \top } ) } \end{array}$ and show that the eigenvalues of M<sup>′</sup> are positive.

## A. Positive definiteness $o f - \left\lceil \mathbf { A } - \Sigma ^ { - 1 } \right\rceil$

The symmetric part of the matrix of the coefficient $- \left[ \mathbf { A } - \Sigma ^ { - 1 } \right]$ is given by

$$
- \frac {1}{2} (\mathbf {A} + \mathbf {A} ^ {\top}) + \Sigma^ {- 1} = \Sigma^ {- 1} - \mathbf {S _ {1}}\tag{19}
$$

Where $\mathbf { S _ { 1 } }$ is the symmetric part of the matrix A used for simplicity. Hence, we are interested in the condition $\Sigma ^ { - 1 } -$ $\mathbf { S _ { 1 } } \succ 0$ . Which could be satisfied if:

$$
\lambda_ {\min} (\Sigma^ {- 1}) > \lambda_ {\max} (\mathbf {S _ {1}}).\tag{20}
$$

Where $\lambda ( \cdot )$ is the eigenvalues of the given matrix. In the special case of an isotropic distribution where $\Sigma = \sigma ^ { 2 } \mathbb { I }$ we retrieve an inequality similar to the one-dimensional case in equation 10.

$$
\sigma <   \frac {1}{\sqrt {\lambda_ {\max} (\mathbf {S _ {1}})}}
$$

## B. Positive definiteness $o f - \Sigma ^ { - 1 } \left[ \mathbf { A } - \mathbf { B } \mathbf { K } \right]$

The symmetric par of the matrix of the coefficient $- \Sigma ^ { - 1 } \left[ \mathbf { A } - \mathbf { B } \mathbf { K } \right]$ is given by:

$$
- \frac {1}{2} \left(\boldsymbol {\Sigma} ^ {- 1} (\mathbf {A} - \mathbf {B K}) + (\mathbf {A} - \mathbf {B K}) ^ {\top} \boldsymbol {\Sigma} ^ {- 1}\right)\tag{21}
$$

Therefore, the exact condition for stability is

$$
- \frac {1}{2} \left(\boldsymbol {\Sigma} ^ {- 1} (\mathbf {A} - \mathbf {B K}) + (\mathbf {A} - \mathbf {B K}) ^ {\top} \boldsymbol {\Sigma} ^ {- 1}\right) \succ 0.\tag{22}
$$

Which is equivalent to the Lyapunov inequality

$$
\Sigma^ {- 1} (\mathbf {A} - \mathbf {B K}) + (\mathbf {A} - \mathbf {B K}) ^ {\top} \Sigma^ {- 1} \prec 0.\tag{23}
$$

If all real parts of the eigenvalues of A − BK are strictly negative and since the covariance matrix Σ is positive definite, then there exists a covariance matrix that meets this inequality.

In the special case of an isotropic distribution where $\Sigma =$ $\sigma ^ { 2 } \mathbb { I }$ we get

$$
\left(\mathbf {A} - \mathbf {B K}\right) + \left(\mathbf {A} - \mathbf {B K}\right) ^ {\top} \prec 0.
$$

Which is equivalent to the positive definiteness of the symmetric part of the matrix A − BK defined as $\mathbf { S } _ { 2 }$ times the inverse of the covariance matrix:

$$
\frac {- \Sigma^ {- 1}}{2} \left((\mathbf {A} - \mathbf {B K}) + (\mathbf {A} - \mathbf {B K}) ^ {\top}\right) = - \Sigma^ {- 1} \mathbf {S _ {2}}\tag{24}
$$

Hence, we are interested in the condition $\pmb { \Sigma } ^ { - 1 } \mathbf { S } _ { 2 } \succ - 0$ which could be satisfied, in the isotropic case, if:

$$
\lambda_ {\max} (\mathbf {S _ {2}}) <   0\tag{25}
$$