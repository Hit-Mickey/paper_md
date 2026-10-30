# Equivariant Diffusion Policy

Dian Wang<sup>1∗</sup>, Stephen Hart<sup>2</sup>, David Surovik<sup>2</sup>, Tarik Kelestemur<sup>2</sup>, Haojie Huang<sup>1</sup>, Haibo Zhao<sup>1</sup>, Mark Yeatman<sup>2</sup>, Jiuguang Wang<sup>2</sup>, Robin Walters<sup>1</sup>, Robert Platt<sup>1,2</sup>

<sup>1</sup>Northeastern University <sup>2</sup>Boston Dynamics AI Institute https://equidiff.github.io

Abstract: Recent work has shown diffusion models are an effective approach to learning the multimodal distributions arising from demonstration data in behavior cloning. However, a drawback of this approach is the need to learn a denoising function, which is significantly more complex than learning an explicit policy. In this work, we propose Equivariant Diffusion Policy, a novel diffusion policy learning method that leverages domain symmetries to obtain better sample efficiency and generalization in the denoising function. We theoretically analyze the SO(2) symmetry of full 6-DoF control and characterize when a diffusion model is SO(2)-equivariant. We furthermore evaluate the method empirically on a set of 12 simulation tasks in MimicGen, and show that it obtains a success rate that is, on average, 21.9% higher than the baseline Diffusion Policy. We also evaluate the method on a real-world system to show that effective policies can be learned with relatively few training samples, whereas the baseline Diffusion Policy cannot.

Keywords: Equivariance, Diffusion Model, Robotic Manipulation

## 1 Introduction

The recently proposed Diffusion Policy [1] formulates robotic manipulation action prediction as a diffusion model that denoises the action conditioned on the observation, thereby better capturing the multimodal action distribution of the demonstration data in Behavior Cloning (BC). Although Diffusion Policy often outperforms baselines on benchmarks [2, 3], a key drawback is that the denoising function is more complex than a standard policy function. In particular, for a single state-action pair (s, a), the denoising process uses a mapping $( s , a + \varepsilon ^ { k } , k ) \mapsto \varepsilon ^ { k }$ for all possible k and $\varepsilon ^ { k }$ , where $\varepsilon ^ { k }$ is Gaussian noise conditioned on step k, which is harder to train compared with an explicit BC s 7→ a.

In this paper, we leverage equivariant neural models to embed task symmetry as an inductive bias in the diffusion process, making the denois-

![](images/54700e7b51fec065c524712f7af59f4094bcbf582b15475a7084f45fcfbc883b.jpg)  
Figure 1: Equivariance in diffusion policy. Top left: a randomly sampled trajectory. Top right: a valid trajectory after denoising. If the state and the random trajectory are both rotated (bottom left), and we rotate the noise accordingly in the denoising process, we will end up with a successful trajectory in the rotated state (bottom right).

ing function easier to learn. Although equivariant diffusion models have been studied by a number of prior works [4, 5, 6, 7, 8], our paper is the first to study the idea in the context of visuomotor policy learning. As illustrated in Figure 1, rotation of a state and noisy trajectory action about the gravity axis (i.e., rotated on the tabletop) results in a corresponding rotation of the denoised trajectory. As a result of this symmetry, our model is more data efficient and generalizes better than the non-symmetric baselines, mitigating the high data costs typically associated with diffusion.

Our contributions are as follows: 1) we propose Equivariant Diffusion Policy, a novel BC approach based on equivariant diffusion, 2) we analyze the conditions under which the denoising function is equivariant, 3) we theoretically demonstrate the use of SO(2)-equivariance in the context of 6-DoF control for robotic manipulation, which prior methods [9, 10] leveraged in a less expressive SE(2) action space, and 4) we provide a thorough demonstration of our method in both simulated and physical systems. In simulation, we evaluate on 12 manipulation tasks in the MimicGen benchmark [11] and outperform the baseline Diffusion Policy by an average success rate of 21.9% when trained with 100 demos. On hardware, we show that successful policies can be learned with a small number (between 20 and 60) of demonstrations for six different manipulation tasks, including a long-horizon bagel baking task, while the original Diffusion Policy performs poorly in this low-data regime.

## 2 Related Work

Diffusion Models Diffusion models [12] learn distributions by modeling the reverse of a diffusion process, which is a Markov chain that gradually adds Gaussian noise to the data until it transitions to a Gaussian distribution. Denoising diffusion models [13, 14] can be interpreted as learning the gradient field of an implicit score during training, where inference applies a sequence of score optimization steps. This new family of generative methods has proven to be effective for capturing multimodal distributions in planning [15, 16] and policy learning [17, 18, 19, 1, 20]. However, these methods did not leverage the geometric symmetries underlying the task and the diffusion process. Xu et al. [21], Hoogeboom et al. [4] show that leveraging SO(3) symmetries from the domain in the diffusion process dramatically improves sample efficiency and generalization ability in molecular generation. EDGI [6] extends diffuser [15] to equivariant diffusion planning with improved performance, but relies on the ground truth state as the input. Ryu et al. [7] propose bi-equivariant diffusion models for visual robotic manipulation, while limited to open-loop settings. By contrast, we exploit domain symmetries during the diffusion process to attain an effective closed-loop visuomotor policy.

Equivariance in manipulation policies Robots operate within a three-dimensional Euclidean space, where manipulation tasks inherently encompass geometric symmetries such as rotations. Recent works [9, 22, 23, 24, 25, 26, 27, 10, 28, 29, 30, 31, 32] compellingly show that improvement in sample efficiency and performance can be obtained by leveraging symmetries in policy leaning. [33, 34, 35] show the efficiency of equivariant models for on-robot learning. [36, 37, 38, 39] learn an open-loop pick and place policy with few demonstrations. While this prior work either considers symmetries in SE(3) open-loop or SE(2) closed-loop action spaces, our paper studies symmetries in an SE(3) closed-loop action space, and is the first one to study the symmetries in diffusion policy.

Closed-loop Visuomotor control Closed-loop visuomotor policies are more robust and responsive but struggle with learning from diverse trajectories and predicting long-horizon actions. Previous methods [40, 41, 42, 43] directly map from observations to actions. However, this type of explicit policy learning struggles to learn multimodal behavior distributions and may not be expressive enough to capture the full range and fidelity of trajectory data [17, 44]. Several works propose implicit policies [45, 46] with energy-based models [47, 48]. However, training is challenging due to the necessity of a substantial volume of negative samples to effectively learn an optimal energy score function for state-action pairs. Recently, [17, 1] model action generation as a conditional denoising diffusion process and demonstrate strong performance by adapting diffusion models to sequential environments. Our work builds on [1] but focuses on equivariance in the diffusion process.

## 3 Background

Problem Statement We study policy learning using behavior cloning. The agent is required to learn a mapping from the observation o to the action a that mimics an expert policy. Both o and a can contain a number of time steps, i.e., $\mathbf { o } = \bigl \{ \mathbf { o } _ { t - ( m - 1 ) } , \dotsc , \mathbf { o } _ { t - 1 } , \mathbf { o } _ { t } \bigr \} , \mathbf { a } = \bigl \{ \mathbf { a } _ { t } , \mathbf { a } _ { t + 1 } , \dotsc , \mathbf { a } _ { t + ( n - 1 ) } \bigr \}$ where m is the number of history steps observed and n is the number of future action steps. The observation contains both visual information (images or voxels) and the pose vector of the gripper.

Let $\boldsymbol { T } _ { t } \in \mathbb { R } ^ { 4 \times 4 }$ be the current $\operatorname { S E } ( 3 )$ pose of the gripper in the world frame, the actions $\mathbf { a } _ { t }$ specify a desired pose $\mathbf { A } _ { t } \in \mathbb { R } ^ { 4 \times 4 }$ of the gripper and an open-width command $w _ { t } \in \mathbb { R }$ . The pose can be either absolute $( T _ { t + 1 } = \mathbf { A } _ { t }$ , also called position control) or relative $( T _ { t + 1 } = \mathbf { A } _ { t } T _ { t }$ , also called velocity control). In order to noise and denoise via addition and subtraction as in the standard diffusion process, we vectorize the $\operatorname { S E } ( 3 )$ pose $\mathbf { A } _ { t }$ into a vector $\mathbf { a } _ { t }$ during diffusion and denoising, and orthogonalize the noise-free action vector after denoising.

Diffusion Policy Chi et al. [1] proposed Diffusion Policy to model the multimodal distribution in behavior cloning using Denoising Diffusion Probabilistic Models (DDPMs) [14]. Diffusion Policy learns a noise prediction function $\varepsilon _ { \theta } ( \mathbf { o } , \mathbf { a } + \varepsilon ^ { k } , k ) = \varepsilon ^ { k }$ using a network $\varepsilon _ { \theta }$ parameterized by $\theta .$ . The network is expected to predict the noise component of the input $\mathbf { a } + \varepsilon ^ { k }$ . During training, transitions $( \mathbf { o } , \mathbf { a } )$ are sampled from the expert dataset. Then, random noise $\varepsilon ^ { k }$ (conditioned on a randomly sampled denoising step k) is added to a. The loss is $\mathcal { L } = | | \varepsilon _ { \theta } ( \mathbf { o } , \mathbf { a } + \varepsilon ^ { k } , k ) - \varepsilon ^ { k } | | ^ { 2 }$ . During inference, given an observation o, DDPM performs a sequence of $K$ denoising steps starting from a random action $\mathbf { a } ^ { k } \sim \mathcal { N } ( 0 , 1 )$ to generate an action $\mathbf { a } ^ { 0 }$ defined inductively by

$$
\mathbf {a} ^ {k - 1} = \alpha (\mathbf {a} ^ {k} - \gamma \varepsilon_ {\theta} (\mathbf {o}, \mathbf {a} ^ {k}, k) + \epsilon),\tag{1}
$$

where $\epsilon \sim \mathcal { N } ( 0 , \sigma ^ { 2 } I ) . \alpha , \gamma , \sigma$ are functions of the denoising step k (also known as the noise schedule). The action $\mathbf { a } ^ { 0 }$ is expected to be a sample from the expert policy $\pi : \mathbf { o } \mapsto \mathbf { a }$

Equivariance A function $f$ is equivariant if it commutes with the transformations of a symmetry group G. Specifically, $\forall g \in G , f ( \rho _ { x } ( g ) x ) = \rho _ { y } ( g ) f ( x )$ , where $\rho \colon G \to G L ( n )$ is called the group representation that maps each group element to an $n \times n$ invertible matrix that acts on the input and output through matrix multiplication. We sometimes leave the actions implicit and write $f ( g x ) = g f ( x )$ . We mainly focus on the group $\mathrm { S O } ( 2 )$ of planar rotations $( \mathrm { i . e . }$ ., rotation around the z-axis of the world) and its subgroup $C _ { u }$ containing u discrete rotations. There are three particular representations of $\mathrm { S O } ( 2 )$ or $C _ { u }$ that are of interest in this paper:

1) the trivial representation $\rho _ { 0 }$ defines $\mathrm { S O } ( 2 )$ or $C _ { u }$ acting on an invariant scalar x $\in \mathbb { R }$ by $\rho _ { 0 } ( g ) x =$ $x , \ 2 )$ the irreducible representation $\rho _ { \omega }$ defines $\mathrm { S O } ( 2 )$ or $C _ { u }$ acting on a vector $v \in \mathbb { R } ^ { 2 }$ by $\mathbf { a \ 2 \times 2 }$ rotation matrix with frequency ω, $\overset { \cdot } { \rho _ { \omega } } ( g ) v = \bigl ( \overset { \cos \omega g } { \sin } - \sin \omega g \bigr ) v . \ : \bar { 3 } \bigr )$ the regular representation $\rho _ { \mathrm { r e g } }$ that defines $C _ { u }$ acting on a vector $x \in \mathbb { R } ^ { u }$ by $u \times$ u permutation matrices. Let $g = r ^ { m } \in C _ { u } =$ $\{ 1 , r ^ { 1 } , \dotsc , r ^ { u - 1 } \}$ and $( x _ { 1 } , \dots , x _ { u } ) \in \mathbb { R } ^ { u }$ . Then $\rho _ { r e g } ( g ) x = ( x _ { u - m + 1 } , \ldots , x _ { u } , x _ { 1 } , x _ { 2 } , \ldots , x _ { u - m } )$ cyclically permutes the coordinates of $\mathbb { R } ^ { u }$

A representation $\rho$ can also be a combination of different representations, i.e., $\rho = \rho _ { 0 } ^ { n _ { 0 } } \oplus \rho _ { 1 } ^ { n _ { 1 } } \oplus \rho _ { 2 } ^ { n _ { 2 } } \in$ $G L ( n _ { 0 } + 2 n _ { 1 } + 2 n _ { 2 } )$ . In such a case, $\rho ( g )$ is an $\left( n _ { 0 } + 2 n _ { 1 } + 2 n _ { 2 } \right) \times \left( n _ { 0 } + 2 n _ { 1 } + 2 n _ { 2 } \right)$ block diagonal matrix that acts on $\dot { x } \in \mathbb { R } ^ { n _ { 0 } + 2 n _ { 1 } + 2 n _ { 2 } }$

## 4 Method

## 4.1 Theory of Equivariant Diffusion Policy

The main contribution of this paper is a method that incorporates equivariance in the diffusion process for policy learning. As theoretical justification, we first analyze the noise prediction function and show that it is equivariant any time the expert policy that is being modeled is equivariant. This implies equivariant neural networks have the correct inductive bias to model this function.

Let $\pi : \mathbf o \mapsto \mathbf a$ be the expert policy function, and let $\varepsilon : ( \mathbf { o } , \mathbf { a } ^ { k } , k ) \mapsto \varepsilon ^ { k }$ be the ground truth noise prediction function associated with the expert policy such that $\varepsilon ^ { k } = \varepsilon ( \mathbf { o } , \pi ( \mathbf { o } ) + \varepsilon ^ { k } , k )$ . Assume $g \in \mathrm { S O } ( 2 )$ acts upon the noise $\varepsilon ^ { k }$ in the same way as it acts upon the action a.

Proposition 1. The noise prediction function ε is equivariant, $i . e . , \varepsilon ( g \mathbf { o } , g \mathbf { a } ^ { k } , k ) = g \varepsilon ( \mathbf { o } , \mathbf { a } ^ { k } , k ) , g \in$ ${ \mathrm { S O } } ( 2 )$ , when the expert policy function is $\mathrm { S O } ( 2 )$ -equivariant, $i . e . , \pi ( g \mathbf { o } ) = g \pi ( \mathbf { o } ) , g \in \mathrm { S O } ( 2 )$

See Appendix A for the proof. Figure 2 illustrates the equivariance property of ε. If we infer ε for all actions in the action space, we effectively acquire a gradient field towards the expert trajectory. The figure shows that such a gradient field is equivariant when the expert policy is equivariant, thus the function ε is also equivariant. Notice that the figure shows the average of all action time steps.

## 4.2 SO(2) Representation on 6DoF Action

A key step in defining an Equivariant Diffusion Policy is to define how actions $\mathbf { a } _ { t }$ transform under rotation. We describe this transformation in terms of irreducible SO(2) representations, which allows us to build the equivariance constraint into the denoising network.

![](images/0c2ab582310134aa88476b52e5f8aa0264c6cbe520eb4989fab30ac3258d2fe5.jpg)  
Figure 2: Equivariance of the denoising function ε. Left: In observation $\mathbf { o } ,$ the goal for the gripper is to reach the green block while avoiding the blue obstacle. Right: The expert trajectory and the gradient field associated with the denoising function. If the policy is equivariant, both the denoising function and the entire gradient field is equivariant. The orange boxes show the equivariance of ε with a particular input $\varepsilon ^ { k }$

Proposition 2. There exist irreducible representations that describe how $\mathrm { S O } ( 2 )$ acts on an SE(3) gripper action $\mathbf { a } _ { t } .$ . In absolute pose control, let ${ \bf a } _ { t } = \mathrm { V e c } _ { c } ( { \bf A } _ { t } )$ where $\mathrm { V e c } _ { c }$ flattens an SE(3) pose $\mathbf { A } _ { t } \in \mathbb { R } ^ { 4 \times 4 }$ into a vector by column, $g \mathbf { a } _ { t } = ( \rho _ { 1 } \oplus \rho _ { 0 } ^ { 2 } ) ^ { 4 } ( g ) \mathbf { a } _ { t }$ . In relative pose control, let $\mathbf { a } _ { t } = \mathrm { V e c } _ { r } ( \mathbf { A } _ { t } )$ where Vec flattens ${ \bf A } _ { t }$ into a vector by row, $g \mathbf { a } _ { t } = P ^ { - 1 } \left[ ( \rho _ { 0 } ^ { 6 } \oplus \rho _ { 1 } ^ { 4 } \oplus \rho _ { 2 } ) ( g ) \right] P \mathbf { a } _ { t }$ , where $P$ is afixed change-of-basis matrix.

Absolute Control We first consider absolute pose control, i.e., $T _ { t + 1 } \ = \ \mathbf { A } _ { t }$ . Let $T _ { g }$ be the transformation matrix corresponding to the SO(2) rotation along the z-axis of the world frame, $T _ { g } = { \binom { \cos g - \sin g 0 0 } { \sin g \ \cos g \ 0 0 } } = { \binom { \rho _ { 1 } ( g ) } { \rho _ { 0 } ( g ) } } _ { \rho _ { 0 } ( g ) } \ \qquad $ , where $\rho _ { 1 } ( g ) = { \binom { \cos g - \sin g } { \sin g \quad \cos g } }$ . The $\mathrm { S O } ( 2 )$ action on A<sub>t</sub> is $g \mathbf { A } _ { t } = T _ { g } \mathbf { A } _ { t } = ( \rho _ { 1 } \oplus \rho _ { 0 } ^ { 2 } ) ( g ) \mathbf { A } _ { t }$ . Vectorizing ${ \bf A } _ { t }$ by column gives $\mathbf { a } _ { t } = \mathrm { V e c } _ { c } ( \mathbf { A } _ { t } ) = \left[ \mathbf { A } _ { t } ^ { 1 T } , \mathbf { A } _ { t } ^ { 2 T } , \mathbf { A } _ { t } ^ { 3 T } , \mathbf { A } _ { t } ^ { 4 T } \right] ^ { T }$ where $\mathbf { A } _ { t } ^ { i }$ is the ith column of ${ \bf A } _ { t }$ By the rule of matrix multiplication, we have $g \mathbf { A } _ { t } ^ { i } = ( \rho _ { 1 } \oplus \rho _ { 0 } ^ { 2 } ) ( g ) \mathbf { A } _ { t } ^ { i }$ and $g \mathbf { a } _ { t } = ( \rho _ { 1 } \oplus \rho _ { 0 } ^ { 2 } ) ^ { 4 } ( g ) \mathbf { a } _ { t }$

Since the gripper open width is invariant, $g w _ { t } = \rho _ { 0 } ( g ) w _ { t }$ , we can append w<sub>t</sub> to a<sub>t</sub> and add an extra $\rho _ { 0 }$ to the representation. We can also simplify the representation by removing the constants in the transformation matrix and removing the last row in the rotation part of the transformation matrix (i.e., the 6D rotation representation [49]). The resulting action vector would be $\mathbf { a } _ { t } \in \mathbb { R } ^ { 6 } \times \mathbb { R } ^ { 3 } \times \mathbb { R }$ where the first six elements are the 6D rotation, the following three elements are the translation, and the last element is the gripper open width. In such a case, we have $g \mathbf { a } _ { t } = ( \rho _ { 1 } ^ { 3 } \oplus ( \rho _ { 1 } \oplus \rho _ { 0 } ) \oplus \rho _ { 0 } ) ( g ) \mathbf { a } _ { t }$

Relative Control For relative gripper pose, i.e., $T _ { t + 1 } = \mathbf { A } _ { t } T _ { t }$ , the group action on A satisfies $( g \mathbf { A } _ { t } ) T _ { g } T _ { t } \ = \ T _ { g } ( \mathbf { A } _ { t } T _ { t } )$ (because the rotation $g \in \mathrm { S O } ( 2 )$ applies to both the current pose and the change of pose). Solving for $g \mathbf { A } .$ <sub>t</sub> we get $g { \bf A } _ { t } = \mathrm { \Delta } { \tilde { T } } _ { g } { \bf A } _ { t } T _ { g } ^ { - 1 }$ . Let $\mathbf { a } _ { t } ~ = ~ \mathrm { V e c } _ { r } ( \mathbf { A } _ { t } )$ where Vec<sub>r</sub> : $\colon \mathbb { R } ^ { n \times m }  \mathbb { R } ^ { ( n \cdot m ) }$ flattens a matrix into a vector by row. Here we want to find a linear action $\rho _ { \mathbf { A } }$ that satisfies $g \mathbf { a } _ { t } = \rho _ { \mathbf { A } } ( g ) \mathrm { V e c } _ { r } ( \mathbf { A } _ { t } ) = \mathrm { V e c } _ { r } ( T _ { g } \mathbf { A } _ { t } T _ { q } ^ { - 1 } )$ . After solving for $\rho _ { \mathbf { A } } \in \mathbb { R } ^ { 1 6 \times 1 6 }$ and calculating a change-of-basis matrix P such that $P \rho _ { \bf A } P ^ { - 1 }$ is a block diagonal matrix consisting of irreducible representations, we have $g \mathbf { a } _ { t } = P ^ { - 1 } \left. ( \rho _ { 0 } ^ { 6 } \oplus \rho _ { 1 } ^ { 4 } \oplus \rho _ { 2 } ) ( g ) \right. P \bar { \mathbf { a } } _ { t }$ (see Appendix B for details). For easier implementation, we add $\mathrm { ~ a ~ } \rho _ { 0 }$ for the gripper action $w _ { t }$ , remove the constants in the transformation matrix, and decompose $\mathrm { S E } ( 3 ) = \mathrm { S O } ( 3 ) \times \mathbb { R } ^ { 3 }$ . See Appendix C.

## 4.3 Implementation of Equivariant Diffusion Policy

Now that we have the theoretical grounding of the equivariance in the noise prediction function ε, this section will introduce the network architecture of our Equivariant Diffusion Policy. As is shown in Figure 3, our network consists of three main parts: encoding (white box), denoising (yellow box), and decoding (gray box). We implement our network using the escnn library [50]. First, an equivariant observation encoder and an equivariant action encoder take inputs o and $\mathbf { a } ^ { k }$ , respectively, to create equivariant embeddings $e _ { \mathbf { o } }$ and $e _ { \mathbf { a } ^ { k } }$ . The embeddings will be in the form of a regular representation of the subgroup $C _ { u } \subset \mathrm { S O } ( 2 )$ (where u is the number of discrete rotations in the group). The embeddings have shape $\boldsymbol { e _ { \mathbf { o } } } \in \mathbb { R } ^ { u \times d _ { o } }$ and $e _ { \mathbf { a } ^ { k } } \in \mathbb { R } ^ { u \times d _ { a } }$ where each of the $d _ { o }$ or $d _ { a }$ dimensional vectors encodes the features for a specific group element (i.e., a rotation angle). Second, in the denoising step, let $e _ { \mathbf { o } } ^ { g } \in \mathbb { R } ^ { d _ { o } }$ and $e _ { \mathbf { a } ^ { k } } ^ { g } \in \mathbb { R } ^ { d _ { a } }$ be a pair of partial embeddings corresponding to the same group element $g .$ We process each pair with a 1D Temporal U-Net (adopted from the prior works [15, 1]) to calculate an equivariant noise embedding. Specifically, letting k be the denoising step, U the U-Net, and z its output, we have $z ^ { g } = U ( e _ { \bf o } ^ { g } , e _ { \bf a ^ { k } } ^ { g } , k )$ . Since the same network is applied for all $g \in C _ { u } ,$ , the output is an equivariant embedding of the noise in the regular representation. Finally, an equivariant decoder will decode the noise $\varepsilon ^ { k }$ . See Appendix D for details.

![](images/6ba7b3c293c1bf65ecfd80df744666e1ed2d185ed9f749a2a013c54b5f41e5ce.jpg)  
(a) Kitchen D1

![](images/8f874bb7de1e7ef3a5a5051d18f07dcc8c2da6f50b8416db3af3dd7940e806d9.jpg)  
(b) Nut Assembly D0

![](images/666dbe1bf098ba14d3807675198e81b002bd88b4431d6295fd7182b2d8ed4965.jpg)  
(c) Pick Place D0

![](images/bfc7d8c3257f52b85d1353d39f8cf546d8d5c18b35cc134770723d6ab27efd59.jpg)  
(d) Coffee Preparation D1  
Figure 4: The experimental environments from MimicGen [11]. The left image in each subfigure shows the initial state of the environment; the right image shows the goal state. See Figure 8 in the Appendix for all environments.

![](images/e67c494ee65af03878ae69a0b831a2c9f393349b6d2d63f6d38fdb0c9bf36653.jpg)  
Figure 3: Overview of our Equivariant Diffusion Policy architecture.

## 5 Experiments

## 5.1 Simulation Experiment

Experimental Settings We first evaluate our Equivariant Diffusion Policy (EquiDiff) with either image (Im) or voxel (Vo) input on 12 manipulation tasks from MimicGen [11] (Figure 4). We define the rotation of the observation as a voxel grid rotation or an image rotation. Notice that in the image version of our method, there is a mismatch between the rotation of the agent view image and the rotation of the ground truth state since the agent view is not orthogonally top-down. Although topdown observations could be captured, we use the observation settings in the published dataset from MimicGen [11] to demonstrate the generalizability of our method<sup>1</sup>. On the other hand, the voxel version eliminates this symmetry mismatch as the rotation of the voxel grid aligns with the rotation of the ground truth state. To better leverage the equivariance, we also add a rotation augmentation in the voxel version of our method following our analysis in Section 4.1-4.2. We compare our method with the following baselines: 1) DiffPo-C: the original diffusion policy [1] trained with the 1D Temporal UNet [15]. Notice that the baseline shares the same UNet architecture as our method, but it does not have any equivariant structure. 2) DiffPo-T: same as above, but trained with a transformer. 3) DP3: the 3D diffusion policy [20] trained with a point net encoder. 4) ACT: the Action Chunking

<table><tr><td rowspan="2">Method</td><td rowspan="2">Ctrl</td><td rowspan="2">Obs</td><td colspan="3">Stack D1</td><td colspan="3">Stack Three D1</td><td colspan="3">Square D2</td><td colspan="3">Threading D2</td></tr><tr><td>100</td><td>200</td><td>1000</td><td>100</td><td>200</td><td>1000</td><td>100</td><td>200</td><td>1000</td><td>100</td><td>200</td><td>1000</td></tr><tr><td>EquiDiff (Vo)</td><td rowspan="6">Abs</td><td>Voxel</td><td>99 (+23)</td><td>100 (+3)</td><td>100 (=)</td><td>75 (+37)</td><td>91 (+19)</td><td>91 (-3)</td><td>39 (+31)</td><td>48 (+29)</td><td>63 (+14)</td><td>39 (+22)</td><td>53 (+18)</td><td>55 (-4)</td></tr><tr><td>EquiDiff (Im)</td><td>RGB</td><td>93 (+17)</td><td>100 (+3)</td><td>100 (=)</td><td>55 (+17)</td><td>77 (+5)</td><td>96 (+2)</td><td>25 (+17)</td><td>41 (+22)</td><td>60 (+11)</td><td>22 (+5)</td><td>40 (+5)</td><td>59 (=)</td></tr><tr><td>DiffPo-C [1]</td><td>RGB</td><td>76</td><td>97</td><td>100</td><td>38</td><td>72</td><td>94</td><td>8</td><td>19</td><td>46</td><td>17</td><td>35</td><td>59</td></tr><tr><td>DiffPo-T [1]</td><td>RGB</td><td>51</td><td>83</td><td>99</td><td>17</td><td>41</td><td>84</td><td>5</td><td>11</td><td>45</td><td>11</td><td>18</td><td>41</td></tr><tr><td>DP3 [20]</td><td>PCD</td><td>69</td><td>87</td><td>99</td><td>7</td><td>23</td><td>65</td><td>7</td><td>6</td><td>19</td><td>12</td><td>23</td><td>40</td></tr><tr><td>ACT [51]</td><td>RGB</td><td>35</td><td>73</td><td>96</td><td>6</td><td>37</td><td>78</td><td>6</td><td>18</td><td>49</td><td>10</td><td>21</td><td>35</td></tr><tr><td>EquiDiff (Vo)</td><td rowspan="4">Rel</td><td>Voxel</td><td>95 (+14)</td><td>100 (+5)</td><td>100 (=)</td><td>59 (+33)</td><td>76 (+24)</td><td>83 (-9)</td><td>25 (+17)</td><td>35 (+14)</td><td>52 (-7)</td><td>33 (+20)</td><td>39 (+13)</td><td>46 (-1)</td></tr><tr><td>EquiDiff (Im)</td><td>RGB</td><td>75 (-6)</td><td>96 (+1)</td><td>100 (=)</td><td>25 (-1)</td><td>63 (+11)</td><td>92 (=)</td><td>11 (+3)</td><td>21 (=)</td><td>48 (-11)</td><td>11 (-2)</td><td>22 (-4)</td><td>49 (+2)</td></tr><tr><td>DiffPo-C [1]</td><td>RGB</td><td>81</td><td>93</td><td>99</td><td>26</td><td>52</td><td>86</td><td>6</td><td>13</td><td>37</td><td>13</td><td>26</td><td>40</td></tr><tr><td>BC RNN [2]</td><td>RGB</td><td>59</td><td>95</td><td>100</td><td>12</td><td>48</td><td>92</td><td>8</td><td>21</td><td>59</td><td>7</td><td>13</td><td>47</td></tr><tr><td colspan="15"></td></tr><tr><td rowspan="2">Method</td><td rowspan="2">Ctrl</td><td rowspan="2">Obs</td><td colspan="3">Coffee D2</td><td colspan="3">Three Pc. Assembly D2</td><td colspan="3">Hammer Cleanup D1</td><td colspan="3">Mug Cleanup D1</td></tr><tr><td>100</td><td>200</td><td>1000</td><td>100</td><td>200</td><td>1000</td><td>100</td><td>200</td><td>1000</td><td>100</td><td>200</td><td>1000</td></tr><tr><td>EquiDiff (Vo)</td><td rowspan="6">Abs</td><td>Voxel</td><td>65 (+18)</td><td>73 (+7)</td><td>76 (-3)</td><td>37 (+33)</td><td>58 (+52)</td><td>71 (+28)</td><td>70 (+16)</td><td>66 (-5)</td><td>73 (-14)</td><td>53 (+10)</td><td>65 (+6)</td><td>68 (+5)</td></tr><tr><td>EquiDiff (Im)</td><td>RGB</td><td>60 (+13)</td><td>79 (+13)</td><td>76 (-3)</td><td>15 (+11)</td><td>39 (+33)</td><td>69 (+26)</td><td>65 (+11)</td><td>63 (-8)</td><td>77 (-10)</td><td>49 (+6)</td><td>64 (+5)</td><td>67 (+2)</td></tr><tr><td>DiffPo-C [1]</td><td>RGB</td><td>44</td><td>66</td><td>79</td><td>4</td><td>6</td><td>30</td><td>52</td><td>59</td><td>73</td><td>43</td><td>59</td><td>65</td></tr><tr><td>DiffPo-T [1]</td><td>RGB</td><td>47</td><td>61</td><td>75</td><td>1</td><td>4</td><td>43</td><td>48</td><td>60</td><td>76</td><td>30</td><td>43</td><td>63</td></tr><tr><td>DP3 [20]</td><td>PCD</td><td>34</td><td>45</td><td>69</td><td>0</td><td>1</td><td>3</td><td>54</td><td>71</td><td>87</td><td>21</td><td>33</td><td>53</td></tr><tr><td>ACT [51]</td><td>RGB</td><td>19</td><td>33</td><td>64</td><td>0</td><td>3</td><td>24</td><td>38</td><td>54</td><td>71</td><td>23</td><td>31</td><td>56</td></tr><tr><td>EquiDiff (Vo)</td><td rowspan="4">Rel</td><td>Voxel</td><td>55 (+12)</td><td>59 (+7)</td><td>64 (-12)</td><td>5 (+3)</td><td>5 (=)</td><td>55 (+28)</td><td>64 (+21)</td><td>62 (+8)</td><td>67 (-5)</td><td>39 (+14)</td><td>43 (+4)</td><td>62 (-5)</td></tr><tr><td>EquiDiff (Im)</td><td>RGB</td><td>41 (-2)</td><td>59 (+7)</td><td>66 (-10)</td><td>1 (-1)</td><td>5 (=)</td><td>59 (+32)</td><td>49 (+6)</td><td>52 (-2)</td><td>69 (-3)</td><td>29 (+4)</td><td>36 (-3)</td><td>65 (-2)</td></tr><tr><td>DiffPo-C [1]</td><td>RGB</td><td>43</td><td>51</td><td>67</td><td>2</td><td>2</td><td>20</td><td>43</td><td>54</td><td>65</td><td>25</td><td>39</td><td>55</td></tr><tr><td>BC RNN [2]</td><td>RGB</td><td>37</td><td>52</td><td>76</td><td>0</td><td>5</td><td>27</td><td>32</td><td>43</td><td>72</td><td>19</td><td>39</td><td>67</td></tr><tr><td colspan="15"></td></tr><tr><td rowspan="2">Method</td><td rowspan="2">Ctrl</td><td rowspan="2">Obs</td><td colspan="3">Kitchen D1</td><td colspan="3">Nut Assembly D0</td><td colspan="3">Pick Place D0</td><td colspan="3">Coffee Preparation D1</td></tr><tr><td>100</td><td>200</td><td>1000</td><td>100</td><td>200</td><td>1000</td><td>100</td><td>200</td><td>1000</td><td>100</td><td>200</td><td>1000</td></tr><tr><td>EquiDiff (Vo)</td><td rowspan="6">Abs</td><td>Voxel</td><td>85 (+18)</td><td>89 (+4)</td><td>88 (-3)</td><td>67 (+12)</td><td>77 (+9)</td><td>83 (-1)</td><td>58 (+23)</td><td>68 (+3)</td><td>82 (-1)</td><td>80 (+15)</td><td>83 (+21)</td><td>85 (+9)</td></tr><tr><td>EquiDiff (Im)</td><td>RGB</td><td>67 (=)</td><td>77 (-8)</td><td>81 (-10)</td><td>74 (+19)</td><td>85 (+17)</td><td>94 (+10)</td><td>42 (+7)</td><td>74 (+9)</td><td>92 (+9)</td><td>77 (+12)</td><td>83 (+21)</td><td>85 (+9)</td></tr><tr><td>DiffPo-C [1]</td><td>RGB</td><td>67</td><td>85</td><td>87</td><td>55</td><td>68</td><td>83</td><td>35</td><td>65</td><td>83</td><td>65</td><td>62</td><td>58</td></tr><tr><td>DiffPo-T [1]</td><td>RGB</td><td>54</td><td>75</td><td>81</td><td>31</td><td>32</td><td>46</td><td>15</td><td>37</td><td>50</td><td>38</td><td>51</td><td>76</td></tr><tr><td>DP3 [20]</td><td>PCD</td><td>45</td><td>71</td><td>91</td><td>16</td><td>24</td><td>58</td><td>12</td><td>15</td><td>34</td><td>10</td><td>22</td><td>63</td></tr><tr><td>ACT [51]</td><td>RGB</td><td>37</td><td>61</td><td>87</td><td>42</td><td>64</td><td>84</td><td>7</td><td>17</td><td>50</td><td>32</td><td>46</td><td>65</td></tr><tr><td>EquiDiff (Vo)</td><td rowspan="4">Rel</td><td>Voxel</td><td>69 (+27)</td><td>83 (+19)</td><td>89 (+8)</td><td>53 (+11)</td><td>65 (+3)</td><td>72 (-13)</td><td>40 (+5)</td><td>58 (-1)</td><td>79 (-3)</td><td>48 (+6)</td><td>71 (+18)</td><td>73 (+12)</td></tr><tr><td>EquiDiff (Im)</td><td>RGB</td><td>61 (+19)</td><td>72 (+8)</td><td>83 (+2)</td><td>44 (+2)</td><td>65 (+3)</td><td>87 (+2)</td><td>29 (-6)</td><td>55 (-4)</td><td>91 (+9)</td><td>49 (+7)</td><td>59 (+6)</td><td>79 (+18)</td></tr><tr><td>DiffPo-C [1]</td><td>RGB</td><td>42</td><td>64</td><td>81</td><td>42</td><td>62</td><td>75</td><td>35</td><td>59</td><td>82</td><td>42</td><td>53</td><td>51</td></tr><tr><td>BC RNN [2]</td><td>RGB</td><td>31</td><td>47</td><td>81</td><td>35</td><td>58</td><td>85</td><td>21</td><td>41</td><td>77</td><td>14</td><td>32</td><td>61</td></tr></table>

Table 1: The performance of our method compared with the baselines in simulation. We experiment with 100, 200, and 1000 demos in each environment and report the maximum task success rate among 50 evaluations throughout training. Results averaged over three seeds. Number in parentheses shows the difference between our method and the best baseline (with increment colored in blue and decrement in red). Bold performance indicates the best, bold difference is greater than 10%.

Transformer [51] trained as a conditional VAE. 5) BC RNN: a recurrent architecture from [2]. Notice that the voxel version of our method and DP3 utilizes the 3D inputs constructed from four cameras, while the image version of our method and the other baselines directly use the RGB images from two cameras. As our main baseline, we evaluate DiffPo-C in both absolute and relative pose control. We evaluate the other baselines in the same control mode as in the original work (absolute for DiffPo-T, DP3 and ACT, and relative for BC RNN). See Appendix E and F for the details.

Results Table 1 shows the experimental result in terms of the maximum success rate among 50 evaluations throughout the training. First, with absolute pose control, our Equivariant Diffusion Policy with voxel input achieves the best overall performance, outperforming the baselines in 11 out of the 12 environments (Hammer Cleanup D1 being the exception). With RGB image inputs, our method outperforms all RGB baselines in all environments except for Kitchen D1. Second, in relative control, our method with voxel input achieves the best performance, while our method with RGB input is only marginally better than the baselines. Third, our method appears to perform particularly well in the lowdata regime (i.e., with 100 or 200 demos). Specifically, taking the average over all environments (as is shown in Table 2), our method with voxel input and absolute pose control trained with 100 demos outperforms the best baseline by 21.9%. When trained with 200 demos, it outperforms all baselines trained with 1000 demos, indicating the strong sample efficiency of our method. We also perform an ablation study in Appendix H, where we ablate the equivariant structure and the voxel input in our method. We show that though both of these items contribute to the performance improvement our method shows over the baseline, the equivariant structure is the more important factor.

<table><tr><td rowspan="2">Method</td><td rowspan="2">Ctrl</td><td colspan="3">Average over 12 Environments</td></tr><tr><td>100</td><td>200</td><td>1000</td></tr><tr><td>EquiDiff (Vo)</td><td rowspan="6">Abs</td><td>63.9 (+21.9)</td><td>72.6 (+14.8)</td><td>77.9 (+6.5)</td></tr><tr><td>EquiDiff (Im)</td><td>53.7 (+11.7)</td><td>68.5 (+10.7)</td><td>79.7 (+8.3)</td></tr><tr><td>DiffPo-C [1]</td><td>42.0</td><td>57.8</td><td>71.4</td></tr><tr><td>DiffPo-T [1]</td><td>29.0</td><td>43.0</td><td>64.9</td></tr><tr><td>DP3 [20]</td><td>23.9</td><td>35.1</td><td>56.8</td></tr><tr><td>ACT [51]</td><td>21.3</td><td>38.2</td><td>63.3</td></tr><tr><td>EquiDiff (Vo)</td><td rowspan="4">Rel</td><td>48.8 (+15.5)</td><td>58.0 (+10.7)</td><td>70.2 (-0.1)</td></tr><tr><td>EquiDiff (Im)</td><td>35.4 (+2.1)</td><td>50.4 (+3.1)</td><td>74.0 (+3.7)</td></tr><tr><td>DiffPo-C [1]</td><td>33.3</td><td>47.3</td><td>63.2</td></tr><tr><td>BC RNN [2]</td><td>22.9</td><td>41.2</td><td>70.3</td></tr></table>

Table 2: The average performance over 12 tasks of Equivariant Diffusion Policy compared with baselines.

![](images/a70580d2df46586873c095d4c123088ea39c7aca4d1d01ed1ef38a044c86edb6.jpg)

High-Equivariance Tasks  
![](images/585b275dba14121daf0a00f9dc1fe789f61c0efba447d309e52795eede51d8e6.jpg)

Intermediate-Equivariance Tasks  
![](images/f3481c8a9d0dd54cf9b6e84edd66bde8042cb7dd1a74f25978a269489e290757.jpg)  
(a)  
Low-Equivariance Tasks

Performance Improvement Across Environments  
![](images/341a5a889da5d917d33dfb0069a191479a43ddd7b212784b3588369d3ca80590.jpg)  
(b)  
Figure 5: (a) The three task groups are based on the level of equivariance and their initial object distribution. Images were generated by taking the average of five random initialization states. (b) The performance improvement of our Equivariant Diffusion Policy (Voxel) compared with the original diffusion policy in absolute pose control. Blue environments are high-equivariance tasks; green environments are intermediate-equivariance tasks; red environments are low-equivariance tasks.

## 5.2 Improvement with Different Levels of Equivariance

We further analyze the performance improvement of our method when the tasks have different levels of equivariance. Since equivariant models generalize automatically across different object poses, equivariance should hypothetically be more useful when there is greater variance in the distribution of initial object poses. We qualitatively group the tasks into three levels: 1) high-equivariance tasks where the poses of the objects are initialized randomly within the workspace; 2) intermediateequivariance tasks where each object is initialized in a certain range, but with some randomness inside the range; 3) low-equivariance tasks where there is no randomness for the position and/or orientation of certain objects. Figure 5a shows the three task groups. We show the performance improvement of our Equivariant Diffusion Policy with voxel in absolute pose control compared with the standard diffusion policy in Figure 5b. Generally, the high-equivariance tasks benefit more from injecting symmetry in the network architecture. Moreover, our method’s strong performance in the intermediate and low-equivariance tasks indicates its robustness and generalizability, as the model’s symmetry is helpful even when the task is partially symmetric.

## 5.3 Real-Robot Experiment

Experimental Settings In this section, we evaluate our method on a real robot system containing a Franka Emika robot arm [53] equipped with a pair of fin-ray [54] fingers and three Intel Realsense [55] D455 cameras. Demonstrations were gathered by an operator using a 6DoF 3DConnexion mouse. Observations and demonstration actions were recorded at 5Hz. Similarly to prior work [1], we use DDIM [56] in this experiment to reduce the number of denoising steps to 16. Figure 6 shows the six tasks in this experiment. We compare our Equivariant Diffusion Policy with voxel input against a baseline Diffusion Policy, which uses the same voxel grid as the vision input and uses a non-equivariant 3D convolutional encoder with approximately the same number of trainable parameters as ours. As we show in the ablation study (Appendix H), this baseline works better than the original diffusion policy with image input.

![](images/e4e8c91c139b6443d27bc82e8e4e9db286a4f46d02da0473f9251d07d8fa594b.jpg)

![](images/5ade36f986e3d76e1605c945fab17298e86d8de2bbd04b7cc99357d85ae24855.jpg)  
(a) Oven Opening

![](images/86f48e65187d4269e7f9b9d9dc4a75a3ff777967f333f2aec2a3a8e833d6fefa.jpg)

![](images/3fe9b1fb720e153dd8b6774cd247f3cfbf68ebbacb728f76b5c1f7cdf29df2aa.jpg)  
(b) Banana in Bowl

![](images/770267ede0af42b32c5de72d7ea7bdedcc27ead3bf6ae86c01219449a5bc2e57.jpg)

![](images/237e8ea3f2662b7d57dcdbc46052d2f621003a916dc121133afe37ea07f51ca4.jpg)  
(c) Letter Alignment

![](images/9c40a339fbdf7253d443ccd56810aab4a7a41db8b7e2aae49dba32e60139f6c8.jpg)

![](images/70398cf5f1154d1304d7f9b866fbbff5423ec32326510ecfcd0b586fb54854f4.jpg)  
(d) Trash Sweeping

![](images/dfa50099266061953696cc3f1fe2e132591daf7374e0a1da3d473f71c60e943a.jpg)

![](images/f6c85b485a52b0bdba4b4dd2549f9d11423cfdffeb14dba80c3fd92fc41e6951.jpg)  
(e) Hammer to Drawer

![](images/c0c6c086747895f98d8a6d1ccd5b43564199d90252922ce9db6e902019921d31.jpg)

![](images/545e0c0b54c7e39de6151fe8b38fcc218566945f836e99930395a7550dd48a2e.jpg)

![](images/ce9c0c1a7712970b4c6f72c5c528b4ede3472edb345768e01d96f165666ffb7e.jpg)

![](images/dadbdf32bf33e181373aaa8e5b5b7c73d1d7689b9ca734d741d8b163d2332969.jpg)  
(f) Bagel Baking

![](images/db899d3357b823a29e05cc8a41308eb386d1503d98e239264905f98f78290664.jpg)

![](images/3a57cf902d7163620e7a84330545cac1b13bf7271be13aac22683409cb93cb2a.jpg)

Figure 6: The real-world environments. The left image of each subfigure shows the initial state of the environment; the right image shows the goal state. See Appendix L for a detailed task description.

<table><tr><td></td><td>Oven Opening</td><td>Banana in Bowl</td><td>Letter Alignment</td><td>Trash Sweeping</td><td>Hammer to Drawer</td><td>Bagel Baking</td></tr><tr><td># Demos</td><td>20</td><td>40</td><td>40</td><td>40</td><td>60</td><td>58</td></tr><tr><td>EquiDiff (Vo)</td><td>95% (19/20)</td><td>95% (19/20)</td><td>95% (19/20)</td><td>90% (18/20)</td><td>85% (17/20)</td><td>80% (16/20)</td></tr><tr><td>DiffPo-C (Vo)</td><td>60% (12/20)</td><td>30% (6/20)</td><td>0% (0/20)</td><td>5%(1/20)</td><td>5%(1/20)</td><td>10% (2/20)</td></tr></table>

Table 3: Performance of Equivariant Diffusion Policy in Real-World Robot Experiments.

Results We evaluate the trained models over 20 test trials for each task. The results are shown in Table 3. Our Equivariant Diffusion Policy can solve those tasks with only 20 to 60 demonstrations. Notably, our method achieves an 80% success rate in bagel baking, where the failures were all due to the joint limits of the robot. In comparison, the baseline performs poorly in all six tasks.

## 6 Conclusion

This paper studies the leveraging of symmetries in visuomotor policy learning. We propose the novel Equivariant Diffusion Policy method and provide a theoretical analysis identifying the conditions under which diffusion processes are equivariant. We also demonstrate a general framework for using SO(2)-equivariance in the 6DoF control for robotic manipulation. We evaluate our method in both simulation and the real world and show in both cases that our method outperforms the baseline Diffusion Policy by a large margin.

One limitation of this work is the partial utilization of the power of equivariance due to the symmetry mismatch in the vision system. Even with the voxel input, Factors like the arm’s occasional presence in the voxel grid and camera noise could break symmetry. Future work could address this by designing a vision system free of symmetry corruption. Additionally, “incorrect equivariance”, as shown in prior work [57], may harm performance when the model’s symmetry conflicts with the demonstration. Another limitation is that although the theory in Section 4.2 is not limited to diffusion policies and can apply to other policy learning pipelines as well, this is not demonstrated. Specifically, given the good performance of BC RNN with the relative pose control in Table 1, experimenting with an equivariant version of BC RNN could be beneficial. Finally, extending our method to other robotic tasks like navigation, locomotion, and mobile manipulation is a key future direction.

## Acknowledgments

This work is supported in part by NSF 1750649, NSF 2107256, NSF 2314182, NSF 2134178, NSF 2409351, and NASA 80NSSC19K1474. Dian Wang is supported in part by the JPMorgan Chase PhD fellowship. The authors would like to thank Dr. Osman Dogan Yirmibesoglu for the design of the fin-ray gripper fingers, Dr. Andy Park for building the teleop system for data collection, Emmanuel Panov for collecting demonstration data in the robot experiment, Dr. Thomas Weng for the proof-read of the paper, and Dr. Cheng Chi for the helpful discussion.

## References

[1] C. Chi, S. Feng, Y. Du, Z. Xu, E. Cousineau, B. Burchfiel, and S. Song. Diffusion Policy: Visuomotor Policy Learning via Action Diffusion. In Proceedings of Robotics: Science and Systems (RSS), 2023.

[2] A. Mandlekar, D. Xu, J. Wong, S. Nasiriany, C. Wang, R. Kulkarni, L. Fei-Fei, S. Savarese, Y. Zhu, and R. Mart´ın-Mart´ın. What Matters in Learning from Offline Human Demonstrations for Robot Manipulation. In A. Faust, D. Hsu, and G. Neumann, editors, Proceedings ofthe 5th Conference on Robot Learning, volume 164 of Proceedings of Machine Learning Research, pages 1678–1690. PMLR, 08–11 Nov 2022.

[3] A. Gupta, V. Kumar, C. Lynch, S. Levine, and K. Hausman. Relay Policy Learning: Solving Long-Horizon Tasks via Imitation and Reinforcement Learning. In Conference on Robot Learning, pages 1025–1037. PMLR, 2020.

[4] E. Hoogeboom, V. G. Satorras, C. Vignac, and M. Welling. Equivariant Diffusion for Molecule Generation in 3D. In International Conference on Machine Learning, pages 8867–8887. PMLR, 2022.

[5] J. Guan, W. W. Qian, X. Peng, Y. Su, J. Peng, and J. Ma. 3D Equivariant Diffusion for Target-Aware Molecule Generation and Affinity Prediction. In The Eleventh International Conference on Learning Representations, 2023.

[6] J. Brehmer, J. Bose, P. De Haan, and T. Cohen. EDGI: Equivariant Diffusion for Planning with Embodied Agents. arXiv preprint arXiv:2303.12410, 2023.

[7] H. Ryu, J. Kim, J. Chang, H. S. Ahn, J. Seo, T. Kim, J. Choi, and R. Horowitz. Diffusion-EDFs: Bi-equivariant Denoising Generative Modeling on SE(3) for Visual Robotic Manipulation. arXiv preprint arXiv:2309.02685, 2023.

[8] K. Chen, X. Chen, Z. Yu, M. Zhu, and H. Yang. Equidiff: A conditional equivariant diffusion model for trajectory prediction. In 2023 IEEE 26th International Conference on Intelligent Transportation Systems (ITSC), pages 746–751. IEEE, 2023.

[9] D. Wang, R. Walters, and R. Platt. SO(2)-Equivariant Reinforcement Learning. In International Conference on Learning Representations, 2022.

[10] M. Jia, D. Wang, G. Su, D. Klee, X. Zhu, R. Walters, and R. Platt. SEIL: Simulation-augmented Equivariant Imitation Learning. In International Conference on Robotics and Automation (ICRA), 2023.

[11] A. Mandlekar, S. Nasiriany, B. Wen, I. Akinola, Y. Narang, L. Fan, Y. Zhu, and D. Fox. MimicGen: A Data Generation System for Scalable Robot Learning using Human Demonstrations. In 7th Annual Conference on Robot Learning, 2023.

[12] J. Sohl-Dickstein, E. Weiss, N. Maheswaranathan, and S. Ganguli. Deep Unsupervised Learning using Nonequilibrium Thermodynamics. In F. Bach and D. Blei, editors, Proceedings of the 32nd International Conference on Machine Learning, volume 37 of Proceedings of Machine Learning Research, pages 2256–2265, Lille, France, 07–09 Jul 2015. PMLR.

[13] Y. Song and S. Ermon. Generative Modeling by Estimating Gradients of the Data Distribution. Advances in Neural Information Processing Systems, 32, 2019.

[14] J. Ho, A. Jain, and P. Abbeel. Denoising Diffusion Probabilistic Models. Advances in Neural Information Processing Systems, 33:6840–6851, 2020.

[15] M. Janner, Y. Du, J. Tenenbaum, and S. Levine. Planning with Diffusion for Flexible Behavior Synthesis. In International Conference on Machine Learning, pages 9902–9915. PMLR, 2022.

[16] Z. Liang, Y. Mu, M. Ding, F. Ni, M. Tomizuka, and P. Luo. AdaptDiffuser: Diffusion Models as Adaptive Self-evolving Planners. In International Conference on Machine Learning, 2023.

[17] T. Pearce, T. Rashid, A. Kanervisto, D. Bignell, M. Sun, R. Georgescu, S. V. Macua, S. Z. Tan, I. Momennejad, K. Hofmann, et al. Imitating Human Behaviour with Diffusion Models. In The Eleventh International Conference on Learning Representations, 2022.

[18] Z. Wang, J. J. Hunt, and M. Zhou. Diffusion Policies as an Expressive Policy Class for Offline Reinforcement Learning. In The Eleventh International Conference on Learning Representations, 2022.

[19] Z. Xian, N. Gkanatsios, T. Gervet, T.-W. Ke, and K. Fragkiadaki. ChainedDiffuser: Unifying Trajectory Diffusion and Keypose Prediction for Robotic Manipulation. In 7th Annual Conference on Robot Learning, 2023.

[20] Y. Ze, G. Zhang, K. Zhang, C. Hu, M. Wang, and H. Xu. 3d diffusion policy: Generalizable visuomotor policy learning via simple 3d representations. In Robotics: Science and Systems, 2024.

[21] M. Xu, L. Yu, Y. Song, C. Shi, S. Ermon, and J. Tang. GeoDiff: A Geometric Diffusion Model for Molecular Conformation Generation. In International Conference on Learning Representations, 2022.

[22] D. Wang, R. Walters, X. Zhu, and R. Platt. Equivariant Q Learning in Spatial Action Spaces. In 5th Annual Conference on Robot Learning, 2021.

[23] H. Huang, D. Wang, A. Tangri, R. Walters, and R. Platt. Leveraging Symmetries in Pick and Place. The International Journal ofRobotics Research, 2023.

[24] A. Simeonov, Y. Du, Y.-C. Lin, A. R. Garcia, L. P. Kaelbling, T. Lozano-Perez, and P. Agrawal.´ SE(3)-Equivariant Relational Rearrangement with Neural Descriptor Fields. In Conference on Robot Learning, pages 835–846. PMLR, 2023.

[25] C. Pan, B. Okorn, H. Zhang, B. Eisner, and D. Held. TAX-Pose: Task-Specific Cross-Pose Estimation for Robot Manipulation. In Conference on Robot Learning, pages 1783–1792. PMLR, 2023.

[26] H. Huang, D. Wang, X. Zhu, R. Walters, and R. Platt. Edge Grasp Network: A Graph-Based SE(3)-invariant Approach to Grasp Detection. In International Conference on Robotics and Automation (ICRA), 2023.

[27] S. Liu, M. Xu, P. Huang, X. Zhang, Y. Liu, K. Oguchi, and D. Zhao. Continual Vision-based Reinforcement Learning with Group Symmetries. In Conference on Robot Learning, pages 222–240. PMLR, 2023.

[28] S. Kim, B. Lim, Y. Lee, and F. C. Park. Se (2)-equivariant pushing dynamics models for tabletop object manipulations. In Conference on Robot Learning, pages 427–436. PMLR, 2023.

[29] C. Kohler, A. S. Srikanth, E. Arora, and R. Platt. Symmetric models for visual force policy learning. arXiv preprint arXiv:2308.14670, 2023.

[30] H. H. Nguyen, A. Baisero, D. Klee, D. Wang, R. Platt, and C. Amato. Equivariant reinforcement learning under partial observability. In Conference on Robot Learning, pages 3309–3320. PMLR, 2023.

[31] H. Nguyen, T. Kozuno, C. C. Beltran-Hernandez, and M. Hamaya. Symmetry-aware reinforcement learning for robotic assembly under partial observability with a soft wrist. arXiv preprint arXiv:2402.18002, 2024.

[32] B. Eisner, Y. Yang, T. Davchev, M. Vecerik, J. Scholz, and D. Held. Deep SE(3)-equivariant geometric reasoning for precise placement tasks. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id= 2inBuwTyL2.

[33] X. Zhu, D. Wang, O. Biza, G. Su, R. Walters, and R. Platt. Sample Efficient Grasp Learning Using Equivariant Models. In Robotics: Science and Systems, 2022.

[34] D. Wang, M. Jia, X. Zhu, R. Walters, and R. Platt. On-Robot Learning With Equivariant Models. In 6th Annual Conference on Robot Learning, 2022.

[35] X. Zhu, D. Wang, G. Su, O. Biza, R. Walters, and R. Platt. On robot grasp learning using equivariant models. Autonomous Robots, 47(8):1175–1193, 2023.

[36] H. Huang, D. Wang, R. Walters, and R. Platt. Equivariant Transporter Network. In Robotics: Science and Systems, 2022.

[37] A. Simeonov, Y. Du, A. Tagliasacchi, J. B. Tenenbaum, A. Rodriguez, P. Agrawal, and V. Sitzmann. Neural Descriptor Fields: SE(3)-Equivariant Object Representations for Manipulation. In 2022 International Conference on Robotics and Automation (ICRA), pages 6394–6400. IEEE, 2022.

[38] H. Ryu, H. in Lee, J.-H. Lee, and J. Choi. Equivariant Descriptor Fields: SE(3)-Equivariant Energy-Based Models for End-to-End Visual Robotic Manipulation Learning. In The Eleventh International Conference on Learning Representations, 2023.

[39] H. Huang, O. Howell, X. Zhu, D. Wang, R. Walters, and R. Platt. Fourier Transporter: Bi-Equivariant Robotic Manipulation in 3D. arXiv preprint arXiv:2401.12046, 2024.

[40] T. Zhang, Z. McCarthy, O. Jow, D. Lee, X. Chen, K. Goldberg, and P. Abbeel. Deep Imitation Learning for Complex Manipulation Tasks from Virtual Reality Teleoperation. In 2018 IEEE International Conference on Robotics and Automation (ICRA), pages 5628–5635. IEEE, 2018.

[41] S. Toyer, R. Shah, A. Critch, and S. Russell. The Magical Benchmark for Robust Imitation. Advances in Neural Information Processing Systems, 33:18284–18295, 2020.

[42] P. Florence, L. Manuelli, and R. Tedrake. Self-Supervised Correspondence in Visuomotor Policy Learning. IEEE Robotics and Automation Letters, 5(2):492–499, 2019.

[43] R. Rahmatizadeh, P. Abolghasemi, L. Bol¨ oni, and S. Levine. Vision-Based Multi-Task Ma-¨ nipulation for Enexpensive Robots Using End-to-End Learning from Demonstration. In 2018 IEEE International Conference on Robotics and Automation (ICRA), pages 3758–3765. IEEE, 2018.

[44] M. Orsini, A. Raichuk, L. Hussenot, D. Vincent, R. Dadashi, S. Girgin, M. Geist, O. Bachem, O. Pietquin, and M. Andrychowicz. What Matters for Adversarial Imitation Learning? Advances in Neural Information Processing Systems, 34:14656–14668, 2021.

[45] D. Jarrett, I. Bica, and M. van der Schaar. Strictly Batch Imitation Learning by Energy-Based Distribution Matching. Advances in Neural Information Processing Systems, 33:7354–7365, 2020.

[46] P. Florence, C. Lynch, A. Zeng, O. Ramirez, A. Wahid, L. Downs, A. Wong, J. Lee, I. Mordatch, and J. Tompson. Implicit Behavioral Cloning. Conference on Robot Learning (CoRL), 2021.

[47] Y. Du and I. Mordatch. Implicit generation and generalization in energy-based models. In Advances in Neural Information Processing Systems, 2019.

[48] W. Grathwohl, K.-C. Wang, J.-H. Jacobsen, D. Duvenaud, and R. Zemel. Learning the Stein Discrepancy for Training and Evaluating Energy-Based Models Without Sampling. In International Conference on Machine Learning, pages 3732–3747. PMLR, 2020.

[49] Y. Zhou, C. Barnes, J. Lu, J. Yang, and H. Li. On the Continuity of Rotation Representations in Neural Networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5745–5753, 2019.

[50] G. Cesa, L. Lang, and M. Weiler. A Program to Build E(n)-Equivariant Steerable CNNs. In International Conference on Learning Representations, 2021.

[51] T. Z. Zhao, V. Kumar, S. Levine, and C. Finn. Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware. In Proceedings ofRobotics: Science and Systems (RSS), 2023.

[52] D. Wang, J. Y. Park, N. Sortur, L. L. Wong, R. Walters, and R. Platt. The Surprising Effectiveness of Equivariant Models in Domains with Latent Symmetry. In International Conference on Learning Representations, 2023.

[53] S. Haddadin, S. Parusel, L. Johannsmeier, S. Golz, S. Gabl, F. Walch, M. Sabaghian, C. Jahne,¨ L. Hausperger, and S. Haddadin. The Franka Emika robot: A Reference Platform for Robotics Research and Education. IEEE Robotics & Automation Magazine, 29(2):46–64, 2022.

[54] W. Crooks, G. Vukasin, M. O’Sullivan, W. Messner, and C. Rogers. Fin ray® effect inspired soft robotic gripper: From the robosoft grand challenge toward optimization. Frontiers in Robotics and AI, 3:70, 2016.

[55] L. Keselman, J. Iselin Woodfill, A. Grunnet-Jepsen, and A. Bhowmik. Intel Realsense Stereoscopic Depth Cameras. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops, pages 1–10, 2017.

[56] J. Song, C. Meng, and S. Ermon. Denoising diffusion implicit models. In International Conference on Learning Representations, 2020.

[57] D. Wang, X. Zhu, J. Y. Park, M. Jia, G. Su, R. Platt, and R. Walters. A general theory of correct, incorrect, and extrinsic equivariance. Advances in Neural Information Processing Systems, 36, 2024.

[58] R. Wang, R. Walters, and R. Yu. Incorporating symmetry into deep dynamics models for improved generalization. In International Conference on Learning Representations (ICLR), 2021.

[59] K. He, X. Zhang, S. Ren, and J. Sun. Deep Residual Learning for Image Recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern recognition, pages 770– 778, 2016.

[60] I. Loshchilov and F. Hutter. Decoupled Weight Decay Regularization. In International Conference on Learning Representations, 2018.

## A Proof of Proposition 1

Proof. Consider the observation o and the action $\mathbf { a } = \pi ( \mathbf { o } )$ , let $\mathbf { a } ^ { k } = \mathbf { a } + \varepsilon ^ { k }$ . By the definition of the noise prediction function ε, we have

$$
\varepsilon^ {k} = \varepsilon (\mathbf {o}, \pi (\mathbf {o}) + \varepsilon^ {k}, k).\tag{2}
$$

Applying $g \in \mathrm { S O } ( 2 )$ to o we have

$$
\varepsilon^ {k} = \varepsilon (g \mathbf {o}, \pi (g \mathbf {o}) + \varepsilon^ {k}, k).\tag{3}
$$

Since π is equivariant,

$$
\varepsilon^ {k} = \varepsilon (g \mathbf {o}, g \mathbf {a} + \varepsilon^ {k}, k).\tag{4}
$$

Since the noise prediction function predicts the noise as long as $\varepsilon ^ { k }$ is the same on both sides of the equation, we can substitute $\varepsilon ^ { k }$ with $g \varepsilon ^ { k }$

$$
g \varepsilon^ {k} = \varepsilon (g \mathbf {o}, g \mathbf {a} + g \varepsilon^ {k}, k).\tag{5}
$$

By linearity, $g \mathbf { a } + g \varepsilon ^ { k } = g ( \mathbf { a } + \varepsilon ^ { k } ) = g \mathbf { a } ^ { k }$ and thus

$$
g \varepsilon^ {k} = \varepsilon (g \mathbf {o}, g \mathbf {a} ^ {k}, k).\tag{6}
$$

Replacing $\varepsilon ^ { k }$ with $\varepsilon ( \mathbf { o } , \mathbf { a } ^ { k } , k )$ gives $g \varepsilon ( \mathbf { o } , \mathbf { a } ^ { k } , k ) = \varepsilon ( g \mathbf { o } , g \mathbf { a } ^ { k } , k )$ as desired.

## B Decomposing Group Representation in Relative Pose Control into Irreducible Representations

In Section 4.2, we want to find a linear action $\rho _ { \mathbf { A } }$ that satisfies $g \mathbf { a } _ { t } = \rho _ { \mathbf { A } } ( g ) \mathrm { V e c } _ { r } ( \mathbf { A } _ { t } ) =$ $\mathrm { V e c } _ { r } ( T _ { g } { \bf A } _ { t } T _ { g } ^ { - 1 } )$ . solving for $\rho _ { \mathbf { A } } ~ \in ~ \mathbb { R } ^ { 1 6 \times 1 6 }$ we have the group action of $\mathrm { S O } ( 2 )$ on $\mathrm { V e c } _ { r } ( \mathbf { A } _ { t } )$

$$
\rho_ {\mathbf {A}} = \left[ \begin{array}{c c c c c c c c c} c ^ {2} & - \frac {s _ {2}}{2} & 0 & 0 & - \frac {s _ {2}}{2} & s ^ {2} & 0 & 0 \\ \frac {s _ {2}}{2} & c ^ {2} & 0 & 0 & - s ^ {2} & - \frac {s _ {2}}{2} & 0 & 0 \\ 0 & 0 & c & 0 & 0 & 0 & - s & 0 \\ 0 & 0 & 0 & c & 0 & 0 & 0 & - s \\ \frac {s _ {2}}{2} & - s ^ {2} & 0 & 0 & c ^ {2} & - \frac {s _ {2}}{2} & 0 & 0 \\ s ^ {2} & \frac {s _ {2}}{2} & 0 & 0 & \frac {s _ {2}}{2} & c ^ {2} & 0 & 0 \\ 0 & 0 & s & 0 & 0 & 0 & c & 0 \\ 0 & 0 & 0 & s & 0 & 0 & 0 & c \\ \hline \end{array} \right] \begin{array}{c c c c c c c c c} \rho_ {1} (g) & \\ & I _ {2} & \\ & & \rho_ {1} (g) \\ & & I _ {2} \end{array}\tag{7}
$$

where c = cos g, s = sin g, c<sub>2</sub> = cos 2g, s<sub>2</sub> = sin 2g. Define P as

$$
P = \left[ \begin{array}{c c c c c c c c c c c c c c c} 1 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 & - 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \text {1} \quad \text {1} \quad \text {1} \quad \text {1} \quad \text {1} \quad \text {1} \quad \text {1} \quad \text {1} \quad \text {1} \quad \text {1} \quad \text {1} \end{array} \right],\tag{8}
$$

We then have

$$
P \rho_ {\mathbf {A}} P ^ {- 1} = \left[ \begin{array}{c c c c c c} I _ {6} & & & & \\ & \rho_ {1} (g) & & & \\ & & \rho_ {1} (g) & & \\ & & & \rho_ {1} (g) & \\ & & & & \rho_ {1} (g) \\ & & & & & \rho_ {2} (g) \end{array} \right].\tag{9}
$$

## C Simplifying Group Action in Relative Pose Control

In Section 4.2, we want to find a linear action $\rho _ { \mathbf { A } }$ that satisfies

$$
\rho_ {\mathbf {A}} (g) \mathrm{Vec} _ {r} (\mathbf {A} _ {t}) = \mathrm{Vec} _ {r} (T _ {g} \mathbf {A} _ {t} T _ {g} ^ {- 1}),\tag{10}
$$

To simplify the problem, we decompose $\mathbf { A } _ { t } = \left[ \begin{array} { c c } { \mathbf { R } _ { t } \mathbf { D } _ { t } } \\ { 0 } & { 1 } \end{array} \right]$ where $\mathbf { R } _ { t }$ is the $\mathrm { S O ( 3 ) }$ rotation and $\mathbf { D } _ { t } =$ $[ x , y , z ] ^ { T }$ is the translation. Define $R _ { g }$ as the rotation matrix in $T _ { g }$

$$
R _ {g} = \left[ \begin{array}{c c c} \cos g & - \sin g & 0 \\ \sin g & \cos g & 0 \\ 0 & 0 & 1 \end{array} \right] = \left[ \begin{array}{c c} \rho_ {1} (g) & \\ & \rho_ {0} (g) \end{array} \right].\tag{11}
$$

Since the conjugate does not apply to translation, we can write $g \mathbf { D } _ { t } = R _ { g } \mathbf { D } _ { t } = [ \rho _ { 1 } ( x , y ) , \rho _ { 0 } ( z ) ] ^ { T }$ For rotation, similar as before, we need to find the representation $\rho _ { \mathbf { R } }$ that satisfies

$$
\rho_ {\mathbf {R}} (g) \mathrm{Vec} _ {r} (\mathbf {R} _ {t}) = \mathrm{Vec} _ {r} (R _ {g} \mathbf {R} _ {t} R _ {g} ^ {- 1}).\tag{12}
$$

Solving for $\rho \mathbf { _ R } ( g ) \in \mathbb { R } ^ { 9 \times 9 }$ we have

$$
\rho_ {\mathbf {R}} = \left[ \begin{array}{c c c c c c c c c} c ^ {2} & - c s & 0 & - c s & s ^ {2} & 0 & 0 & 0 & 0 \\ c s & c ^ {2} & 0 & - s ^ {2} & - c s & 0 & 0 & 0 & 0 \\ 0 & 0 & c & 0 & 0 & - s & 0 & 0 & 0 \\ c s & - s ^ {2} & 0 & c ^ {2} & - c s & 0 & 0 & 0 & 0 \\ s ^ {2} & c s & 0 & c s & c ^ {2} & 0 & 0 & 0 & 0 \\ 0 & 0 & s & 0 & 0 & c & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & 0 & c & - s & 0 \\ 0 & 0 & 0 & 0 & 0 & 0 & s & c & 0 \\ 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \end{array} \right],\tag{13}
$$

where $c = \cos g , s = \sin g$ . To decompose it into the irreducible representations of $\mathrm { S O } ( 2 )$ , we define

$$
P = \left[ \begin{array}{c c c c c c c c c} 1 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\ 0 & - 1 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\ - 1 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \end{array} \right].\tag{14}
$$

Such that $P \rho _ { \mathbf { R } } P ^ { - 1 }$ is a block diagonal matrix consisting of irreducible representations

$$
P \rho_ {\mathbf {R}} P ^ {- 1} = \left[ \begin{array}{c c c c c} \rho_ {0} (g) & & & & \\ & \rho_ {0} (g) & & & \\ & & \rho_ {0} (g) & & \\ & & & \rho_ {1} (g) & \\ & & & & \rho_ {1} (g) \\ & & & & \rho_ {2} (g) \end{array} \right].\tag{15}
$$

![](images/53e451b03dd66799aba6e0d25aa908564d126214e802d2732a0d98b260829575.jpg)  
Figure 7: The detailed network architecture of our Equivariant Diffusion Policy in the simulation experiments.

We can then use $\rho ( g ) = P \rho _ { \mathbf { R } } P ^ { - 1 } = \rho _ { 0 } ^ { 3 } ( g ) \oplus \rho _ { 1 } ^ { 2 } ( g ) \oplus \rho _ { 2 } ( g ) \in \mathbb { R } ^ { 9 \times 9 }$ as the group representation of the output of the equivariant network, then construct the $3 \times 3$ rotation matrix $\mathbf { R } _ { t }$ using $P .$ Specifically, let $V \in \mathbb { R } ^ { 9 }$ be the output of the network associated with the representation $\rho ( g ) \ ( \mathrm { i . e . , } g$ acts on V through $\rho ( g ) V )$ ). Define

$$
\operatorname{Vec} _ {r} (\mathbf {R} _ {t}) = P ^ {- 1} V.\tag{16}
$$

Applying $\rho ( g )$ on V will lead to

$$
P ^ {- 1} \rho (g) V
$$

$$
= P ^ {- 1} P \rho_ {\mathbf {R}} P ^ {- 1} V\tag{17}
$$

(18)

$$
= \rho_ {\mathbf {R}} \mathrm{Vec} _ {r} (\mathbf {R}),\tag{19}
$$

which is the desired property in the equivariant network.

In the end, adding the group action for the translation $( \rho _ { 1 } \oplus \rho _ { 0 } )$ and gripper open width $( \rho _ { 0 } )$ , we have $\rho _ { a } = \rho _ { 0 } ^ { 5 } \oplus \rho _ { 1 } ^ { 3 } \oplus \rho _ { 2 }$

## D Network Architecture Detail

In the image version, we implement the equivariant observation encoder with an equivariant ResNet [58] for the agent view image, a standard ResNet [59] for the eye-in-hand image, and an equivariant MLP for the robot states. We implement the equivariant layers in the group $C _ { 8 }$ . Figure 7 shows the detailed network architecture of our Equivariant Diffusion Policy in the simulation experiments. The network is defined under group $C _ { 8 }$ . First, in the encoding phase, the agent view image is processed with an equivariant ResNet-18, whose output is a 128 × 8-dimensional regular representation vector of group. A non-equivariant ResNet-18 with a spatial maxpool at the end processes the eye-in-hand image and outputs a 128-dimensional representation vector that uses the trivial invariance representation. Those two vectors are concatenated with the gripper position (represented using $\rho _ { 1 }$ ⊕ $\rho _ { 0 } )$ , gripper orientation (in the format of 6D rotation, represented using $\rho _ { 1 } ^ { 3 } )$ , and the gripper finger position (represented using $\rho _ { 0 } ^ { 2 } )$ . The concatenated mixed-representation vector is sent to an equivariant linear layer, whose output is a 128 × 8-dimensional regular representation observation embedding. The noisy action is also encoded using an equivariant linear layer, whose output is a 64 × 8-dimensional regular representation action embedding. Second, in the denoising phase, we process each part of the observation embedding and the action embedding that corresponds to the same group element with a 1D Temporal UNet with hidden dimensions of [512, 1024, 2048] to get a 64-dimensional vector. Doing so for each pair, we will recover a 64 × 8-dimensional regular representation noise embedding. In the end, an equivariant linear layer will decode the noise.

![](images/d41dd2089e5bd808ef55c879e5a667d428ca45abf21f3bfdc56047515752e5fb.jpg)

![](images/0c051ecf5262c561a90291039a54280130e872d8184b215641d32575b46c9878.jpg)

![](images/3765de2ba53e9fd22d48950112b35427b229da58275392e9d0ce36fe279ca6bd.jpg)

![](images/fd308b5975f224f48796fe401750f3a9c3edfea92c590521431abdf8f4f650f3.jpg)  
(a) Stack D1

![](images/04c9e65fc38fa51ac1f71165c1ebbe053c484582b6be0a427efda4af8c487173.jpg)  
(b) Stack Three D1

![](images/7d432a13e09339d2771f893a82c6c141efd19d31c3db97fa8d8230f93d39c606.jpg)

![](images/a375615eee04312efd4ffdb5ecc8322605462d83b8652a67a98d113034c2f2b7.jpg)  
(c) Square D2

![](images/04d8760f2486704da71535ac193b995c0f5ba72373971396fac3119c3ab2f3eb.jpg)

![](images/69b9bd8da1e1a0f0b6b035e51b41114751e9fef3f74d56684ec85116154b97a6.jpg)  
(d) Threading D2

(e) Coffee D2  
![](images/4fd7803c075e8b73daeb5a8b64c64eb62534ceef2ffb5de15c5d2c53b6b236ae.jpg)

![](images/85fd38a85c506f78040252902918d5a9d54e719ab377fbf3b07425299730e9a9.jpg)  
(f) Three Pc. Assembly D2

![](images/02f64769c398cd7a878ef3d6ae36821ad50d5f4a9659ce8b218730664fbed3d2.jpg)

![](images/6e30f1ec64c5e31befb95e8b8ce25da90347ea5ab5cb65ffb7209c4c10960e08.jpg)

![](images/c88c2ade580d0d801a0dde602eb2639b8629cfc865b8e707f21b51b513c6e00d.jpg)  
(g) Hammer Cleanup D1

![](images/3026e2129e92754895c605682907c88654187b45bcd10921abb93e54b834d50f.jpg)  
(h) Mug Cleanup D1

![](images/fcff09f060ca1e2aaa2d0cc5ae9cdb4feb73d43f0688878547cf7195ad09d867.jpg)

(i) Kitchen D1  
![](images/8d0ae75da1d01bb772ab2c82f1e9cac2b1036c151be212ea2b25495eb7c6b13e.jpg)

![](images/2391c7bc24691eeb648fd65bf6cc2a8dd374b933eb2778c24c65b34d4ba0a9d3.jpg)

(j) Nut Assembly D0  
![](images/beba38f2b0c888391062fbe82d6dad1a88897387e79e67f5dd5f3a1564605b8b.jpg)

![](images/6155822ab3df21633175de681d62e2bef73fb1fa2184034fac4d67c5f38322de.jpg)  
(k) Pick Place D0

![](images/18c755e9cf171864729a860fe528fb372e13a744ecf3b45d7a4e014c71017329.jpg)

![](images/3dca2dff861439fbe3318f49e31dba4656527d3b1a97b099c0ded4a6b4800d56.jpg)

![](images/668cbf15563844d2585411b53a82ff4f2571a70e4503cc2be71121042a1d36f3.jpg)  
(l) Coffee Preparation D1

Figure 8: The experimental environments from MimicGen [11]. The left image in each sub figure shows an initial state of the environment; the right image shows the goal state.  
![](images/96c23ae58bce69a58b3ce74864a488aec363bba40966901b16a10a89d32b3a6a.jpg)

![](images/66b68cdd4e7f0bd8e1db0223a8fd2cfe2daf6ac16bcccf38a55e26529445885b.jpg)

![](images/341fe3a444faa121eda80882d6c831a0e04c14a3336219b8a8015ed2b4a0cb83.jpg)

![](images/1b5991e3498f2d137850eb5369fb1f1dd824906eb6d8df8b69070dfc2fe1a976.jpg)

![](images/e2b16ff709b9af4828e244554fe3801bbd4ad90ef25577e59d501d6449bc8c39.jpg)

![](images/69b176d95e9b5d517049ffb3a9238cce4ab76bb92dd73bce93f7c1ddb75ba533.jpg)  
(a) Kitchen D1

![](images/477ff35b908916b7660e7a65f6d76ecd57b10f80d5cef17d9dd543cfc1e70efe.jpg)

![](images/85683a05da875541e890e1e7acad7c613088beb533c84b900eb53cb685e98f71.jpg)

![](images/60c5e55ec9be74da083a1362cc5ccd3792fd6470e1fb781f92754c9f727c8af8.jpg)

![](images/11777cfefbc622ce736794d81e92f289a598e5df3dd5a8eca94e1d0decc2e59a.jpg)

![](images/fbe0603a2a4a59ffeda15f33782c16a3e4559dbcc0a221e76fba701e2557935f.jpg)

![](images/142f8fadda83f5e34010904fb41dceb8bdb14044872a8aac24bbd19833374f09.jpg)

![](images/5ed5b5dbbe35faa3e6dcc43fb362c9f3c6c063e4110d11f4a57abee7cdb3fdaa.jpg)

![](images/29de5b1ac3155fb007661a22c5b7d7f0617b3ad05e4577cfc482338cb2166a0c.jpg)

![](images/afb460e9c93f34db32022267597f1ca8199954351662aa8c63d1d8f4c5e2a682.jpg)

![](images/8e0de619bd1f1c52568e64ed3d347fcb576bc531af4ec48cab3a5e72bb3e8805.jpg)  
(b) Coffee Preparation D1

![](images/cd30e7c2b980ecbc0aa48f17491f63b406c9f77a9cdad470cce8d13965d290ad.jpg)

![](images/a72673b891dbe3879a5fe0fc6749d5a0bf08b467bfd5a0bdcf40df33c1ed585e.jpg)

![](images/fd505dd8725047c2e5a8083af34b10e93e9c8136aba4f21ef1524d986718ede8.jpg)

![](images/0ad0dc63e12c97759292d4a4e39b133e0ca88d07e2e7ceb7aa23eb120118b527.jpg)  
Figure 9: Illustration of an episode of Kitchen D1 task and Coffee Preparation D1 task.

In the voxel version, the agent view image is replaced with a voxel grid, and we replace the equivariant ResNet with an 8-layer 3D equivariant convolutional encoder. The 1D Temporal UNet has a hidden dimensions of [256, 512, 1024]. The other part of the network stays the same. In the realworld, we remove the eye-in-hand image and only use the voxel grid as vision input (the gripper state vector stays the same).

## E Simulation Environments

Figure 8 shows the initial and goal states of each tasks. Figure 9 shows an example trajectory for finishing the Kitchen and Coffee Preparation tasks. The RGB observation is an agent-view image and an eye-in-hand image with a size of $3 \times 8 4 \times 8 4$ . The voxel grid observation has a size of

<table><tr><td>Task</td><td>Max Steps</td><td>Max Out of Plane Rot in Demo</td></tr><tr><td>Stack D1</td><td>400</td><td>11.2</td></tr><tr><td>Stack Three D1</td><td>400</td><td>13.2</td></tr><tr><td>Square D2</td><td>400</td><td>14.7</td></tr><tr><td>Threading D2</td><td>400</td><td>13.4</td></tr><tr><td>Coffee D2</td><td>400</td><td>14.1</td></tr><tr><td>Three Piece Assembly D2</td><td>500</td><td>16.2</td></tr><tr><td>Hammer Cleanup D1</td><td>500</td><td>16.4</td></tr><tr><td>Mug Cleanup D1</td><td>500</td><td>13.0</td></tr><tr><td>Kitchen D1</td><td>800</td><td>16.2</td></tr><tr><td>Nut Assembly D0</td><td>500</td><td>15.5</td></tr><tr><td>Pick Place D0</td><td>1000</td><td>18.0</td></tr><tr><td>Coffee Preparation D1</td><td>800</td><td>59.0</td></tr></table>

Table 4: The maximum number of time steps and the maximum out of plane rotation (in degrees) in the demo for each simulation environments. The maximum out of plane rotation in the demo is the maximum angular difference between the SO(3) rotation and the $\bar { \mathrm { S O } } ( 2 )$ rotation (i.e., only rotating around the z axis) over all demonstration steps, averaged over 1000 demonstration episodes.

<table><tr><td>Method</td><td>Obs</td><td>Obs Step</td><td>Action Pred. Step</td><td>Action Exec. Step</td></tr><tr><td>EquiDiff (Vo)</td><td>Voxel Grid, Eye-In-Hand Image, Gripper State</td><td>1</td><td>16</td><td>8</td></tr><tr><td>EquiDiff (Im)</td><td>Agent View Image, Eye-In-Hand Image, Gripper State</td><td>2</td><td>16</td><td>8</td></tr><tr><td>DiffPo-C</td><td>Agent View Image, Eye-In-Hand Image, Gripper State</td><td>2</td><td>16</td><td>8</td></tr><tr><td>DiffPo-T</td><td>Agent View Image, Eye-In-Hand Image, Gripper State</td><td>2</td><td>10</td><td>8</td></tr><tr><td>DP3</td><td>Point Cloud, Gripper State</td><td>2</td><td>16</td><td>8</td></tr><tr><td>ACT</td><td>Agent View Image, Eye-In-Hand Image, Gripper State</td><td>1</td><td>10</td><td>10</td></tr><tr><td>BC-RNN</td><td>Agent View Image, Eye-In-Hand Image, Gripper State</td><td>1</td><td>1</td><td>1</td></tr></table>

Table 5: The observation format, observation step, action prediction step, and action execution step for all methods. The gripper state is a vector including a 3 dimensional position vector, a rotation vector in the format of 6D rotation representation or 4D quaternion, and a 2 dimensional finger position.

$4 \times 6 4 \times 6 4 \times 6 4$ where the first channel is binary occupancy and the remaining three channels are RGB. The point cloud observation has a size of 1024×6 (i.e, xyzrgb). The point cloud only contains points above the table, as suggested in [20]. All tasks have a full 6 DoF ${ \mathrm { S E } } ( 3 )$ action space. Table 4 shows the maximum number of time steps (following [11]) and the maximum out of plane rotation in the demo, calculated by taking the maximum angular difference between the $\mathrm { S O ( 3 ) }$ rotation and the ${ \mathrm { S O } } ( 2 )$ rotation $( \mathrm { i . e . }$ , only rotating around the z axis) across the entire demonstration episode. Results averaged for 1000 demonstrations.

## F Training Detail

In the simulation experiments, we follow the hyper-parameters of the prior work [1] for the image version of our method, where the only change is that we increase the batch size to 128 for faster training. Specifically, the observation contains two steps of history observation, and the output of the denoising process is a sequence of 16 action steps. We use all 16 steps for training but only execute eight steps in evaluation. We train our models with the AdamW [60] optimizer (with a learning rate of $1 0 ^ { - 4 }$ and weight decay of $1 0 ^ { - 6 } )$ and Exponential Moving Average (EMA). We use a cosine learning rate scheduler with 500 warm-up steps. We use DDPM [14] with 100 denoising steps for both training and evaluation. For each different number of demos (100, 200, 1000), we maintain roughly the same number of training steps by the training for $5 0 0 0 0 / n$ epochs where n is the number of demos. Evaluations are conducted every $1 0 0 0 / n$ epochs (50 evaluations in total). In the voxel version, we use only one step of history observation, and keep the other hyper-parameters the same.

The hyper-parameters for the diffusion policy and BC RNN baselines exactly follow [1]. We follow the original work [20] for the hyper-parameters of DP3, except that we use the same action sequence length (16 for training and 8 for evaluation) as [1] and our method. For the ACT baseline, we follow the hyper-parameters provided in the prior work [51], except that we use a chunk size of 10, KL weight of 10, batch size of 64 with learning rate of $5 \times 1 0 ^ { - 5 }$ , and no temporal aggregation, following the tuning tips provided by the authors. See Table 5 for the observation format, observation step, action prediction step, and action execution step for all methods.

<table><tr><td rowspan="2">Method</td><td rowspan="2">Ctrl</td><td colspan="3">Stack D1</td><td colspan="3">Stack Three D1</td><td colspan="3">Square D2</td><td colspan="3">Threading D2</td></tr><tr><td>100</td><td>200</td><td>1000</td><td>100</td><td>200</td><td>1000</td><td>100</td><td>200</td><td>1000</td><td>100</td><td>200</td><td>1000</td></tr><tr><td>EquiDiff (Vo)</td><td rowspan="6">Abs</td><td>98.7±0.7</td><td>100.0±0.0</td><td>100.0±0.0</td><td>74.7±4.4</td><td>91.3±0.7</td><td>90.7±1.3</td><td>38.7±1.3</td><td>48.0±3.1</td><td>63.3±1.3</td><td>38.7±0.7</td><td>52.7±2.9</td><td>54.7±2.9</td></tr><tr><td>EquiDiff (Im)</td><td>93.3±0.7</td><td>100.0±0.0</td><td>100.0±0.0</td><td>54.7±5.2</td><td>77.3±1.8</td><td>96.0±1.2</td><td>25.3±8.7</td><td>41.3±9.8</td><td>60.0±4.2</td><td>22.0±1.2</td><td>40.0±1.2</td><td>59.3±1.8</td></tr><tr><td>DiffPo-C [1]</td><td>76.0±4.0</td><td>97.3±0.7</td><td>100.0±0.0</td><td>38.0±0.0</td><td>72.0±2.0</td><td>94.0±1.2</td><td>8.0±1.2</td><td>19.3±5.3</td><td>46.0±7.2</td><td>17.3±1.8</td><td>35.3±1.3</td><td>58.7±0.7</td></tr><tr><td>DiffPo-T [1]</td><td>51.3±1.8</td><td>82.7±0.7</td><td>98.7±0.7</td><td>16.7±0.7</td><td>41.3±2.9</td><td>84.0±1.2</td><td>4.7±1.8</td><td>11.3±2.4</td><td>44.7±4.7</td><td>10.7±0.7</td><td>18.0±1.2</td><td>40.7±0.7</td></tr><tr><td>DP3 [20]</td><td>69.3±3.7</td><td>86.7±4.7</td><td>99.3±0.7</td><td>7.3±0.7</td><td>22.7±3.7</td><td>65.3±1.8</td><td>6.7±0.7</td><td>6.0±0.0</td><td>19.3±3.3</td><td>12.0±3.1</td><td>23.3±3.3</td><td>40.0±2.0</td></tr><tr><td>ACT [51]</td><td>34.7±0.7</td><td>72.7±7.7</td><td>96.0±1.2</td><td>6.0±2.3</td><td>36.7±2.7</td><td>78.0±1.2</td><td>6.0±0.0</td><td>18.0±1.2</td><td>49.3±4.7</td><td>10.0±1.2</td><td>20.7±2.9</td><td>35.3±2.4</td></tr><tr><td>EquiDiff (Vo)</td><td rowspan="4">Rel</td><td>94.7±1.3</td><td>100.0±0.0</td><td>100.0±0.0</td><td>59.3±0.7</td><td>76.0±0.0</td><td>82.7±0.7</td><td>24.7±1.8</td><td>34.7±5.2</td><td>52.0±2.3</td><td>33.3±1.8</td><td>38.7±2.9</td><td>46.0±1.2</td></tr><tr><td>EquiDiff (Im)</td><td>74.7±5.8</td><td>96.0±0.0</td><td>100.0±0.0</td><td>25.3±3.3</td><td>62.7±3.5</td><td>92.0±1.2</td><td>11.3±1.3</td><td>20.7±4.1</td><td>48.0±4.0</td><td>11.3±1.3</td><td>22.0±1.2</td><td>49.3±2.4</td></tr><tr><td>DiffPo-C [1]</td><td>80.7±2.4</td><td>93.3±0.7</td><td>99.3±0.7</td><td>26.0±4.0</td><td>52.0±2.0</td><td>86.0±1.2</td><td>6.0±1.2</td><td>13.3±1.3</td><td>36.7±4.8</td><td>13.3±1.8</td><td>26.0±3.1</td><td>40.0±2.3</td></tr><tr><td>BC RNN [2]</td><td>59.3±7.0</td><td>94.7±1.3</td><td>100.0±0.0</td><td>12.0±2.5</td><td>48.0±5.3</td><td>92.0±2.3</td><td>8.0±1.2</td><td>20.7±2.7</td><td>58.7±3.5</td><td>7.3±0.7</td><td>13.3±2.4</td><td>46.7±0.7</td></tr><tr><td colspan="14"></td></tr><tr><td rowspan="2">Method</td><td rowspan="2">Ctrl</td><td colspan="3">Coffee D2</td><td colspan="3">Three Pc. Assembly D2</td><td colspan="3">Hammer Cleanup D1</td><td colspan="3">Mug Cleanup D1</td></tr><tr><td>100</td><td>200</td><td>1000</td><td>100</td><td>200</td><td>1000</td><td>100</td><td>200</td><td>1000</td><td>100</td><td>200</td><td>1000</td></tr><tr><td>EquiDiff (Vo)</td><td rowspan="6">Abs</td><td>64.7±0.7</td><td>73.3±1.8</td><td>76.0±0.0</td><td>37.3±2.7</td><td>58.0±5.0</td><td>71.3±3.3</td><td>70.0±2.0</td><td>66.0±2.3</td><td>72.7±0.7</td><td>52.7±1.3</td><td>64.7±2.4</td><td>68.0±1.2</td></tr><tr><td>EquiDiff (Im)</td><td>60.0±2.0</td><td>79.3±1.3</td><td>76.0±2.0</td><td>15.3±1.8</td><td>39.3±1.8</td><td>69.3±3.5</td><td>65.3±0.7</td><td>63.3±4.4</td><td>76.7±0.7</td><td>49.3±0.7</td><td>64.0±1.2</td><td>66.7±0.7</td></tr><tr><td>DiffPo-C [1]</td><td>44.0±1.2</td><td>66.0±2.3</td><td>78.7±0.7</td><td>4.0±0.0</td><td>6.0±1.2</td><td>30.0±1.2</td><td>52.0±1.2</td><td>58.7±1.3</td><td>73.3±2.4</td><td>42.7±0.7</td><td>58.7±1.3</td><td>65.3±2.4</td></tr><tr><td>DiffPo-T [1]</td><td>47.3±0.7</td><td>60.7±1.8</td><td>74.7±2.7</td><td>0.7±0.7</td><td>4.0±0.0</td><td>42.7±1.3</td><td>48.0±1.2</td><td>60.0±1.2</td><td>76.0±1.2</td><td>30.0±1.2</td><td>42.7±2.9</td><td>63.3±0.7</td></tr><tr><td>DP3 [20]</td><td>34.0±4.0</td><td>45.3±4.1</td><td>68.7±2.4</td><td>0.0±0.0</td><td>0.7±0.7</td><td>3.3±0.7</td><td>54.0±3.1</td><td>70.7±4.1</td><td>86.7±0.7</td><td>21.3±2.7</td><td>32.7±1.8</td><td>52.7±4.4</td></tr><tr><td>ACT [51]</td><td>19.3±2.4</td><td>33.3±2.4</td><td>64.0±2.3</td><td>0.0±0.0</td><td>3.3±0.7</td><td>24.0±3.1</td><td>38.0±4.2</td><td>54.0±1.2</td><td>70.7±1.3</td><td>23.3±0.7</td><td>31.3±1.3</td><td>56.0±2.0</td></tr><tr><td>EquiDiff (Vo)</td><td rowspan="4">Rel</td><td>55.3±0.7</td><td>59.3±0.7</td><td>64.0±0.0</td><td>4.7±0.7</td><td>5.3±0.7</td><td>54.7±3.5</td><td>64.0±1.2</td><td>62.0±1.2</td><td>67.3±1.3</td><td>39.3±0.7</td><td>43.3±1.8</td><td>62.0±1.2</td></tr><tr><td>EquiDiff (Im)</td><td>40.7±0.7</td><td>58.7±1.8</td><td>66.0±1.2</td><td>1.3±0.7</td><td>4.7±0.7</td><td>59.3±4.8</td><td>48.7±2.7</td><td>52.0±3.5</td><td>68.7±2.4</td><td>29.3±2.9</td><td>36.0±1.2</td><td>65.3±2.4</td></tr><tr><td>DiffPo-C [1]</td><td>42.7±1.8</td><td>50.7±1.8</td><td>66.7±2.9</td><td>2.0±1.2</td><td>2.0±0.0</td><td>20.0±1.2</td><td>43.3±1.8</td><td>54.0±1.2</td><td>65.3±1.8</td><td>25.3±0.7</td><td>39.3±1.8</td><td>54.7±0.7</td></tr><tr><td>BC RNN [2]</td><td>37.0±1.0</td><td>52.0±2.0</td><td>76.0±2.3</td><td>0.0±0.0</td><td>5.3±0.7</td><td>27.0±1.0</td><td>32.0±0.0</td><td>42.7±0.7</td><td>72.0±2.3</td><td>19.3±0.7</td><td>39.0±1.0</td><td>66.7±0.7</td></tr><tr><td colspan="14"></td></tr><tr><td rowspan="2">Method</td><td rowspan="2">Ctrl</td><td colspan="3">Kitchen D1</td><td colspan="3">Nut Assembly D0</td><td colspan="3">Pick Place D0</td><td colspan="3">Coffee Preparation D1</td></tr><tr><td>100</td><td>200</td><td>1000</td><td>100</td><td>200</td><td>1000</td><td>100</td><td>200</td><td>1000</td><td>100</td><td>200</td><td>1000</td></tr><tr><td>EquiDiff (Vo)</td><td rowspan="6">Abs</td><td>85.3±0.7</td><td>89.3±0.7</td><td>88.0±2.3</td><td>67.3±0.9</td><td>77.0±0.0</td><td>83.3±0.7</td><td>57.7±1.8</td><td>68.5±0.6</td><td>82.2±0.8</td><td>80.0±1.2</td><td>83.3±1.8</td><td>85.3±1.8</td></tr><tr><td>EquiDiff (Im)</td><td>67.3±0.7</td><td>76.7±3.3</td><td>81.3±0.7</td><td>74.0±1.2</td><td>85.0±1.5</td><td>93.7±0.9</td><td>41.7±3.2</td><td>74.2±3.2</td><td>92.0±1.2</td><td>76.7±0.7</td><td>82.7±0.7</td><td>85.3±0.7</td></tr><tr><td>DiffPo-C [1]</td><td>66.7±2.4</td><td>84.7±0.7</td><td>86.7±1.8</td><td>54.7±2.3</td><td>68.0±2.6</td><td>83.0±1.5</td><td>35.3±2.2</td><td>65.0±2.8</td><td>82.7±0.6</td><td>65.3±0.7</td><td>62.0±4.2</td><td>58.0±3.1</td></tr><tr><td>DiffPo-T [1]</td><td>54.0±2.3</td><td>75.3±0.7</td><td>81.3±2.4</td><td>30.7±5.0</td><td>32.3±5.2</td><td>45.7±5.9</td><td>14.7±1.5</td><td>36.5±1.3</td><td>50.0±6.0</td><td>38.0±2.0</td><td>51.3±1.8</td><td>76.0±6.0</td></tr><tr><td>DP3 [20]</td><td>44.7±1.8</td><td>71.3±2.4</td><td>91.3±2.4</td><td>15.7±1.3</td><td>23.7±3.4</td><td>57.7±1.9</td><td>11.7±0.9</td><td>15.0±1.7</td><td>34.0±0.0</td><td>10.0±2.3</td><td>22.0±5.3</td><td>63.3±4.1</td></tr><tr><td>ACT [51]</td><td>37.3±3.5</td><td>60.7±3.5</td><td>87.3±3.5</td><td>42.3±2.9</td><td>63.7±3.5</td><td>84.3±0.9</td><td>7.2±0.9</td><td>17.2±1.1</td><td>50.0±2.9</td><td>32.0±2.0</td><td>46.0±3.1</td><td>64.7±2.4</td></tr><tr><td>EquiDiff (Vo)</td><td rowspan="4">Rel</td><td>69.3±1.8</td><td>82.7±1.3</td><td>89.3±1.8</td><td>53.0±1.0</td><td>65.0±2.0</td><td>72.0±2.0</td><td>40.3±1.6</td><td>58.2±0.9</td><td>78.8±0.8</td><td>48.0±1.2</td><td>70.7±2.9</td><td>73.3±1.8</td></tr><tr><td>EquiDiff (Im)</td><td>60.7±1.3</td><td>72.0±3.1</td><td>82.7±2.7</td><td>44.3±1.2</td><td>65.3±1.5</td><td>87.3±0.9</td><td>29.3±3.1</td><td>54.7±1.5</td><td>91.3±1.2</td><td>48.7±1.3</td><td>59.3±2.4</td><td>79.3±0.7</td></tr><tr><td>DiffPo-C [1]</td><td>42.0±2.3</td><td>64.0±5.0</td><td>81.3±1.3</td><td>41.7±2.7</td><td>62.0±1.5</td><td>75.3±1.2</td><td>34.7±1.1</td><td>58.7±1.0</td><td>82.2±2.5</td><td>42.0±3.1</td><td>52.7±3.3</td><td>51.3±1.8</td></tr><tr><td>BC RNN [2]</td><td>31.3±2.9</td><td>46.7±6.7</td><td>80.7±1.3</td><td>35.3±0.7</td><td>58.0±2.1</td><td>85.0±1.2</td><td>21.2±0.7</td><td>41.0±9.0</td><td>77.3±2.5</td><td>14.0±1.2</td><td>32.0±1.2</td><td>60.7±4.1</td></tr></table>

Table 6: The performance of our Equivariant Diffusion Policy compared with the baselines in simulation. We experiment with 100, 200, and 1000 demos in each environment and report the maximum task success rate among 50 evaluations throughout training. Results averaged over three seeds. ± indicates standard error.

In the real-world experiments, we use a batch size of 64, one step of observation, and disable the EMA. We use DDIM [56] with 100 denoising steps for training and 16 denoising steps for evaluation. The other hyper-parameters stay the same as in simulation.

## G Simulation Experiment Result with Standard Error

Table 6 shows the same result in Table 1 with the standard error.

## H Ablation Study

We perform an ablation study regarding the equivariant structure and the voxel input in our method. We consider the following four candidates: 1) Ours: our Equivariant Diffusion Policy with voxel input; 2) Ours no Voxel: our Equivariant Diffusion Policy with RGB input; 3) Ours no Equi.: the baseline Diffusion Policy with voxel input; 4) Ours no Voxel no Equi.: the baseline Diffusion Policy with RGB input, same as [1]. Table 7 shows the result and Table 8 shows the average over all 12 environments. Though both the equivariant structure and the voxel input contribute to the performance improvement of our method, the equivariant structure plays a more important rule, as removing it (No Equi.) lead to a more significant performance drop compared with removing the voxel input (No Voxel). Note that by using the voxel input, Diffpo-C (Vo) is marginally better than the original Diffusion Policy (DiffPo-C), thus we use Diffpo-C (Vo) as the baseline in our robot experiment in Section 5.3.

<table><tr><td rowspan="2">Ablation</td><td rowspan="2">Method</td><td rowspan="2">Ctrl</td><td rowspan="2">Obs</td><td colspan="3">Stack D1</td><td colspan="3">Stack Three D1</td><td colspan="3">Square D2</td><td colspan="3">Threading D2</td></tr><tr><td>100</td><td>200</td><td>1000</td><td>100</td><td>200</td><td>1000</td><td>100</td><td>200</td><td>1000</td><td>100</td><td>200</td><td>1000</td></tr><tr><td>-</td><td>EquiDiff (Vo)</td><td rowspan="4">Abs</td><td>Voxel</td><td>99</td><td>100</td><td>100</td><td>75</td><td>91</td><td>91</td><td>39</td><td>48</td><td>63</td><td>39</td><td>53</td><td>55</td></tr><tr><td>No Voxel</td><td>EquiDiff (Im)</td><td>RGB</td><td>93</td><td>100</td><td>100</td><td>55</td><td>77</td><td>96</td><td>25</td><td>41</td><td>60</td><td>22</td><td>40</td><td>59</td></tr><tr><td>No Equi.</td><td>DiffPo-C (Vo)</td><td>Voxel</td><td>87</td><td>99</td><td>100</td><td>33</td><td>79</td><td>94</td><td>10</td><td>24</td><td>60</td><td>19</td><td>43</td><td>54</td></tr><tr><td>No Voxel No Equi.</td><td>DiffPo-C [1]</td><td>RGB</td><td>76</td><td>97</td><td>100</td><td>38</td><td>72</td><td>94</td><td>8</td><td>19</td><td>46</td><td>17</td><td>35</td><td>59</td></tr><tr><td colspan="16"></td></tr><tr><td rowspan="2">Ablation</td><td rowspan="2">Method</td><td rowspan="2">Ctrl</td><td rowspan="2">Obs</td><td colspan="3">Coffee D2</td><td colspan="3">Three Pc. Asse. D2</td><td colspan="3">Hammer Cleanup D1</td><td colspan="3">Mug Cleanup D1</td></tr><tr><td>100</td><td>200</td><td>1000</td><td>100</td><td>200</td><td>1000</td><td>100</td><td>200</td><td>1000</td><td>100</td><td>200</td><td>1000</td></tr><tr><td>-</td><td>EquiDiff (Vo)</td><td rowspan="4">Abs</td><td>Voxel</td><td>65</td><td>73</td><td>76</td><td>37</td><td>58</td><td>71</td><td>70</td><td>66</td><td>73</td><td>53</td><td>65</td><td>68</td></tr><tr><td>No Voxel</td><td>EquiDiff (Im)</td><td>RGB</td><td>60</td><td>79</td><td>76</td><td>15</td><td>39</td><td>69</td><td>65</td><td>63</td><td>77</td><td>49</td><td>64</td><td>67</td></tr><tr><td>No Equi.</td><td>DiffPo-C (Vo)</td><td>Voxel</td><td>50</td><td>72</td><td>75</td><td>2</td><td>5</td><td>50</td><td>54</td><td>64</td><td>76</td><td>47</td><td>58</td><td>66</td></tr><tr><td>No Voxel No Equi.</td><td>DiffPo-C [1]</td><td>RGB</td><td>44</td><td>66</td><td>79</td><td>4</td><td>6</td><td>30</td><td>52</td><td>59</td><td>73</td><td>43</td><td>59</td><td>65</td></tr><tr><td colspan="16"></td></tr><tr><td rowspan="2">Ablation</td><td rowspan="2">Method</td><td rowspan="2">Ctrl</td><td rowspan="2">Obs</td><td colspan="3">Kitchen D1</td><td colspan="3">Nut Assembly D0</td><td colspan="3">Pick Place D0</td><td colspan="3">Coffee Prep. D1</td></tr><tr><td>100</td><td>200</td><td>1000</td><td>100</td><td>200</td><td>1000</td><td>100</td><td>200</td><td>1000</td><td>100</td><td>200</td><td>1000</td></tr><tr><td>-</td><td>EquiDiff (Vo)</td><td rowspan="4">Abs</td><td>Voxel</td><td>85</td><td>89</td><td>88</td><td>67</td><td>77</td><td>83</td><td>58</td><td>69</td><td>82</td><td>80</td><td>83</td><td>85</td></tr><tr><td>No Voxel</td><td>EquiDiff (Im)</td><td>RGB</td><td>67</td><td>77</td><td>81</td><td>74</td><td>85</td><td>94</td><td>42</td><td>74</td><td>92</td><td>77</td><td>83</td><td>85</td></tr><tr><td>No Equi.</td><td>DiffPo-C (Vo)</td><td>Voxel</td><td>82</td><td>87</td><td>87</td><td>66</td><td>77</td><td>84</td><td>41</td><td>67</td><td>84</td><td>65</td><td>75</td><td>77</td></tr><tr><td>No Voxel No Equi.</td><td>DiffPo-C [1]</td><td>RGB</td><td>67</td><td>85</td><td>87</td><td>55</td><td>68</td><td>83</td><td>35</td><td>65</td><td>83</td><td>65</td><td>62</td><td>58</td></tr></table>

Table 7: The ablation study that ablates the voxel input and the equivariant structure in our method. We experiment with 100, 200, and 1000 demos in each environment and report the maximum task success rate among 50 evaluations throughout training. Results averaged over three seeds.

<table><tr><td rowspan="2">Ablation</td><td rowspan="2">Method</td><td rowspan="2">Ctrl</td><td colspan="3">Average over 12 Environments</td></tr><tr><td>100</td><td>200</td><td>1000</td></tr><tr><td>-</td><td>EquiDiff (Vo)</td><td rowspan="4">Abs</td><td>63.9</td><td>72.6</td><td>77.9</td></tr><tr><td>No Voxel</td><td>EquiDiff (Im)</td><td>53.7 (-10.3)</td><td>68.5 (-4.1)</td><td>79.7 (+1.8)</td></tr><tr><td>No Equi.</td><td>DiffPo-C (Vo)</td><td>46.3 (-17.6)</td><td>62.5 (-10.1)</td><td>75.6 (-2.3)</td></tr><tr><td>No Voxel No Equi.</td><td>DiffPo-C [1]</td><td>42.0 (-21.9)</td><td>57.8 (-14.8)</td><td>71.4 (-6.5)</td></tr></table>

Table 8: The average performance over 12 tasks of the ablation study. Number in parenthesis shows the performance difference after removing different components in our Equivariant Diffusion Policy with voxel input.

## I Implementing Equivariance via Data Augmentation

In this section, we evaluate implementing equivariance through data augmentation instead of using equivariant networks. Specifically, we applied random rotation data augmentation based on our analysis in Sections 4.1 and 4.2 to a standard, unconstrained CNN. We then compared this CNN + Aug baseline against using equivariant neural networks (Equi. Net) and not implementing equivariance at all (CNN).

As is shown in Table 9, CNN + Aug can achieve good performance, occasionally even outperforming equivariant networks in simpler tasks like Stack and Stack Three. However, it performs poorly in more challenging tasks. When averaged across 12 environments, CNN + Aug performs better than CNN but still underperforms compared to Equi. Net by a significant margin.

## J SE(2) Action Space Variation

In this section, we evaluate a variation of our Equivariant Diffusion Policy in an SE(2) (with z translation) action space to demonstrate the necessity of leveraging an SE(3) action space. Specifically, the SE(2) agent only learns the top-down rotation and the out-of-plane rotations will be constantly set to 0. As is shown in Table 10, the SE(2) variation achieves a similar performance as the SE(3) version in Stack Three, as the demonstration data in this task has the least amount of out-of-plane rotation (as shown in Table 4). On the other hand, the SE(2) variation significantly underperforms in Threading, since the ability of wiggling the out-of-plane rotation helps the agent to precisely insert the tool. In the end, the SE(2) agent cannot solve Coffee Preparation at all, because the task requires a significant amount of out-of-plane rotation (as shown in Figure 9b).

<table><tr><td>Method</td><td>Obs</td><td>Stack D1</td><td>Stack Three D1</td><td>Square D2</td><td>Threading D2</td></tr><tr><td>Equi. Net (Vo)</td><td>Voxel</td><td>98.7±0.7</td><td>74.7±4.4</td><td>38.7±1.3</td><td>38.7±0.7</td></tr><tr><td>CNN + Aug (Vo)</td><td>Voxel</td><td>99.3±0.7</td><td>84.0±1.2</td><td>36.0±1.2</td><td>30.7±1.8</td></tr><tr><td>CNN (Vo)</td><td>Voxel</td><td>86.7±1.3</td><td>33.3±1.8</td><td>10.0±2.0</td><td>19.3±2.4</td></tr><tr><td>Equi. Net (Im)</td><td>RGB</td><td>93.3±0.7</td><td>54.7±5.2</td><td>25.3±8.7</td><td>22.0±1.2</td></tr><tr><td>CNN + Aug (Im)</td><td>RGB</td><td>98.7±0.7</td><td>68.0±1.2</td><td>26.7±1.8</td><td>22.0±1.2</td></tr><tr><td>CNN (Im)</td><td>RGB</td><td>76.0±4.0</td><td>38.0±0.0</td><td>8.0±1.2</td><td>17.3±1.8</td></tr><tr><td>Method</td><td>Obs</td><td>Coffee D2</td><td>Three Pc. Asse. D2</td><td>Hammer Cleanup D1</td><td>Mug Cleanup D1</td></tr><tr><td>Equi. Net (Vo)</td><td>Voxel</td><td>64.7±0.7</td><td>37.3±2.7</td><td>70.0±2.0</td><td>52.7±1.3</td></tr><tr><td>CNN + Aug (Vo)</td><td>Voxel</td><td>56.7±2.9</td><td>7.3±0.7</td><td>70.7±1.8</td><td>52.0±1.2</td></tr><tr><td>CNN (Vo)</td><td>Voxel</td><td>50.0±3.1</td><td>2.0±0.0</td><td>54.0±3.1</td><td>46.7±0.7</td></tr><tr><td>Equi. Net (Im)</td><td>RGB</td><td>60.0±2.0</td><td>15.3±1.8</td><td>65.3±0.7</td><td>49.3±0.7</td></tr><tr><td>CNN + Aug (Im)</td><td>RGB</td><td>58.0±1.2</td><td>5.3±0.7</td><td>61.3±2.9</td><td>50.0±1.2</td></tr><tr><td>CNN (Im)</td><td>RGB</td><td>44.0±1.2</td><td>4.0±0.0</td><td>52.0±1.2</td><td>42.7±0.7</td></tr><tr><td>Method</td><td>Obs</td><td>Kitchen D1</td><td>Nut Assembly D0</td><td>Pick Place D0</td><td>Coffee Prep. D1</td></tr><tr><td>Equi. Net (Vo)</td><td>Voxel</td><td>85.3±0.7</td><td>67.3±0.9</td><td>57.7±1.8</td><td>80.0±1.2</td></tr><tr><td>CNN + Aug (Vo)</td><td>Voxel</td><td>62.0±2.0</td><td>51.7±1.5</td><td>39.5±2.8</td><td>48.7±2.4</td></tr><tr><td>CNN (Vo)</td><td>Voxel</td><td>82.0±2.3</td><td>66.0±1.7</td><td>40.8±1.9</td><td>65.3±0.7</td></tr><tr><td>Equi. Net (Im)</td><td>RGB</td><td>67.3±0.7</td><td>74.0±1.2</td><td>41.7±3.2</td><td>76.7±0.7</td></tr><tr><td>CNN + Aug (Im)</td><td>RGB</td><td>47.3±2.9</td><td>53.7±0.7</td><td>27.7±0.8</td><td>34.7±2.9</td></tr><tr><td>CNN (Im)</td><td>RGB</td><td>66.7±2.4</td><td>54.7±2.3</td><td>35.3±2.2</td><td>65.3±0.7</td></tr><tr><td rowspan="7"></td><td>Method</td><td>Obs</td><td colspan="2">Average over 12 Environments</td><td rowspan="7"></td></tr><tr><td>Equi. Net (Vo)</td><td>Voxel</td><td colspan="2">63.9</td></tr><tr><td>CNN + Aug (Vo)</td><td>Voxel</td><td colspan="2">53.3</td></tr><tr><td>CNN (Vo)</td><td>Voxel</td><td colspan="2">46.3</td></tr><tr><td>Equi. Net (Im)</td><td>RGB</td><td colspan="2">53.7</td></tr><tr><td>CNN + Aug (Im)</td><td>RGB</td><td colspan="2">46.2</td></tr><tr><td>CNN (Im)</td><td>RGB</td><td colspan="2">42.0</td></tr></table>

Table 9: Comparing implementing equivariance via equivariant network or data augmentation. We experiment with 100 demos in each environment and report the maximum task success rate among 50 evaluations throughout training. Results averaged over three seeds.

<table><tr><td></td><td>Stack Three D1</td><td>Threading D2</td><td>Coffee Preparation D1</td></tr><tr><td>EquiDiff (Im), SE(3) Action</td><td>77.3</td><td>40.0</td><td>85.3</td></tr><tr><td>EquiDiff (Im), SE(2) Action</td><td>75.3</td><td>12.7</td><td>0.0</td></tr></table>

Table 10: Performance of Equivariant Diffusion Policy in SE(2) action space compared with SE(3) action space. 200 demos are used in this experiment.

## K Robomimic Experiment

In this section, we compare our Equivariant Diffusion Policy with the original Diffusion Policy across four Robomimic tasks (Figure 10). Both methods are trained with 100 Proficient-Human (PH) demonstrations or 100 Multi-Human (MH) demonstrations. Other hyperparameters mirror those used in our MimicGen experiment.

Table 11 shows the result. Our method achieves similar or slightly better performance compared to the baseline Diffusion Policy. The improvements in Robomimic tasks are smaller than in Mimic-Gen tasks. This is because the Robomimic tasks can be classified as Low-Equivariance Tasks (as illustrated in Figure 5, bottom), with minimal randomness in the initial distribution (except for Lift), making the symmetry in our method less advantageous.

![](images/dbd31029d3d147eceed7b517e9cb0e43ba7f3a577ccc2095a6618382a1cea2d7.jpg)  
(a) Lift

![](images/6380756ca3155382b738526e82c11276fce240391d8f38138eec0d1d4d9f8a80.jpg)  
(b) Can

![](images/4123949f7e053de48ecc5a24d1164dae57a23ee807cc91ace7c44ad8976ab09c.jpg)  
(c) Square

![](images/c99804bc872c84457e6182ced248f1e5c111486fa609e9b309f1fdf145822c03.jpg)  
(d) Tool Hang

Figure 10: The experimental environments from Robomimic. The left image in each subfigure shows the initial state of the environment; the right image shows the goal state.

<table><tr><td rowspan="2"></td><td colspan="2">Lift</td><td colspan="2">Can</td><td colspan="2">Square</td><td rowspan="2">tool hang100 PH</td><td rowspan="2">Average</td></tr><tr><td>100 PH</td><td>100 MH</td><td>100 PH</td><td>100 MH</td><td>100 PH</td><td>100 MH</td></tr><tr><td>EquiDiff</td><td> $100.0 \pm 0.0$ </td><td> $100.0 \pm 0.0$ </td><td> $99.3 \pm 0.7$ </td><td> $96.7 \pm 0.7$ </td><td> $84.0 \pm 1.2$ </td><td> $76.7 \pm 1.3$ </td><td> $76.0 \pm 0.0$ </td><td> $90.4 \pm 2.3$ </td></tr><tr><td>DiffPo</td><td> $100.0 \pm 0.0$ </td><td> $100.0 \pm 0.0$ </td><td> $100.0 \pm 0.0$ </td><td> $95.3 \pm 0.7$ </td><td> $85.3 \pm 0.7$ </td><td> $70.7 \pm 0.7$ </td><td> $64.0 \pm 5.8$ </td><td> $87.9 \pm 3.2$ </td></tr></table>

Table 11: The performance of our Equivariant Diffusion Policy compared with the Diffusion Policy baseline in Robomimic. We experiment with 100 Proficient-Human (PH) or Multi-Human (MH) demos in each environment and report the maximum task success rate among 50 evaluations throughout training. Results averaged over three seeds. ± indicates standard error.

## L Real-Robot Environment Details

Figure 11 shows our real-world experimental platform containing a Franka Emika [53] and three Intel Realsense[55] D455 cameras. Compared with simulation, we use a pair of fin-ray [54] gripper fingers instead of the original Franka fingers. Figure 6 shows the five tasks in this experiment. In Oven Opening, the oven is randomly initialized at one of the four borders of the workspace. In Banana in Bowl, the initial poses of the banana and the bowl are both randomly sampled. In Trash Sweeping, the robot needs to use a tool brush to sweep two pieces of crumpled paper out of its workspace. The initial poses of the objects are randomly sampled. In Letter Alignment, the robot needs to align the letters to form “AI”. The letter A is randomly initialized at one of the four corners of the workspace, and the pose of the I is randomly sampled. In Hammer to Drawer, the robot needs to open a drawer, pick up a hammer, place it inside the drawer, and close the drawer. The drawer is initialized at one of the four borders of the workspace, and the hammer is randomly initialized at the opposite side of the drawer. Lastly, we also evaluate a Bagel Baking task with an extremely long time horizon, where the robot needs to open the oven, pull out the tray inside the oven, pick up the bagel, place it inside the tray, close the tray, and close the oven. In this task, the oven is randomly initialized at one of the three borders of the workspace (where we eliminate the side that is furthest from the robot to avoid joint limits of the robot), and the bagel is randomly initialized along the opposite side of the oven. The observation is a voxel grid with a resolution of $6 4 \times 6 4 \times 6 4$ and the gripper pose and open width. The voxel grid covers the $( 0 . 4 m ) ^ { 3 }$ workspace. During training, we apply a random crop augmentation to crop the voxel grid to $5 8 \times 5 8 \times 5 8$ . In Banana in Bowl and Trash Sweeping, we train the model with an additional random rotation augmentation. The baseline is trained with the same data augmentation as our method.

## M Generalization Experiment

In this experiment, we evaluate the generalizability of our Equivariant Diffusion Policy to unseen object poses. We conduct this evaluation in the Bagel Baking experiment in the real world, where the oven is initialized in three different poses during training (Fig 12a). At test time, we rotate the oven to eight different, unseen poses (Fig 12b). We found that the learned policy can zero-shot generalize to these unseen rotations, with the exception of the scenario where the oven is rotated to the bottomright corner. In this case, the policy is constrained by the robot’s joint limits. Specifically, the policy was able to open the oven and pull out the tray, but when picking up the bagel, although the policy could generate good gripper poses, the actions were infeasible for the robot due to joint limits. This generalization demonstrates the power of the equivariant structure in our policy.

![](images/49c4fcd41409c85deaf11f3adc6c95fcb8fad7bd8c32c043f59f3f76aabaae34.jpg)  
Figure 11: Our real-robot platform contains a Franka Emika robot arm equipped with a pair of finray fingers, and three Intel Realsense D455 cameras.

![](images/beeee8b3b1d1a31bbca05f3d8e7dabadcf99c0fa7128c9cd15e1e9e8110570c4.jpg)  
(a) Oven Poses in Training Set

![](images/936479a57ad12939c460be2c98d14d3eadd9533ba31efcf948154b26afad18c2.jpg)

![](images/36922ba38aa24cab8d9232490680f9799f784f3b9980d3b712e6fcaa729718c9.jpg)

![](images/885bcb486b070aa509760adc254d2d6ab28388a8c13a2f8ffc92df44fd5e8a57.jpg)  
(b) Oven Poses in The Generalization Experiment

![](images/8998254bdb06b513adf5ed78bd4d8eafde5e67833cd80c20ead7bee4d543f01c.jpg)  
Figure 12: (a) The initial oven poses in the training set. (b) The oven poses in the generalization experiment. Those poses are unseen during training. In both training and testing, the pose of the bagel is random.