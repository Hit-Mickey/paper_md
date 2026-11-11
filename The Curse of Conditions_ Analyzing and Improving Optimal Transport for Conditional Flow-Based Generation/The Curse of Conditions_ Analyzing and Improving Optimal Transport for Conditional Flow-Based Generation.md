# The Curse of Conditions: Analyzing and Improving Optimal Transport for Conditional Flow-Based Generation

Ho Kei Cheng Alexander Schwing University of Illinois Urbana-Champaign

![](images/2f481b4df311514f9910e63c84b89cacc4bba7380abf8793e171ba8a7ab5424e.jpg)  
Figure 1. We visualize the flows learned by different algorithms using the 8gaussians→moons dataset. We compare flow matching (FM), minibatch optimal transport FM (OT), and our proposed conditional optimal transport FM (C<sup>2</sup>OT) using one Euler step and an adaptive ODE solver, respectively. Below each plot, we show the 2-Wasserstein distance (lower is better; mean±std over 10 runs). For unconditional generation, OT achieves significantly straighter flows, outperforming FM. However, paradoxically, OT performs worse when conditions are introduced. This paper analyzes this degradation and finds that it occurs because optimal transport disregards conditions, leading to a train-test gap. To address this, we propose a simple fix (C<sup>2</sup>OT) that outperforms FM and OT in conditional generation.

## Abstract

Minibatch optimal transport coupling straightens paths in unconditionalflow matching. This leads to computationally less demanding inference asfewer integration steps and less complex numerical solvers can be employed when numerically solving an ordinary differential equation at test time. However, in the conditional setting, minibatch optimal transport falls short. This is because the default optimal transport mapping disregards conditions, resulting in a conditionally skewed prior distribution during training. In contrast, at test time, we have no access to the skewed prior, and instead sample from the full, unbiased prior distribution. This gap between training and testing leads to a subpar performance To bridge this gap, we propose conditional optimal transport $( C ^ { 2 } O T )$ that adds a conditional weighting term in the cost matrix when computing the optimal transport assignment. Experiments demonstrate that this simplefix works with both discrete and continuous conditions in 8gaussians→moons,

CIFAR-10, ImageNet-32×32, and ImageNet-256×256. Our method performs better overall compared to the existing baselines across differentfunction evaluation budgets. Code is available at hkchengrex.github.io/C2OT.

## 1. Introduction

We focus on flow-matching-based conditional generative models, i.e., generation guided by an input condition.<sup>1</sup> Examples of such conditions include class labels, text, or video. Recently, flow matching has been applied in many areas of computer vision, including text-to-image/video [11, 36], vision-language applications [3], image restoration [33], and video-to-audio [4]. However, these methods can be slow at test-time – obtaining a solution requires numerically integrating the flow with an ODE solver, typically involving many steps, each requiring a deep network forward pass. Naïvely reducing the number of steps degrades performance, as the underlying flow is often curved (Figure 1, leftmost column), necessitating a small step size for accurate numerical integration. One way to address this issue is by straightening the flow. In the context of unconditional generation, Tong et al. [43] and Pooladian et al. [37] concurrently proposed “minibatch optimal transport” (OT), which deterministically couples data and samples from the prior within a minibatch via optimal transport (replacing random coupling) to minimize flow path lengths and to straighten the flow.

While OT improves unconditional generation, we find that it paradoxically and consistently harms conditional generation (Figure 1). This occurs because OT disregards the conditions when computing the coupling. As a result, during training, OT samples from a prior distribution that is skewed by the condition, as we show in Section 3.2. However, at test time, we have no access to this skewed distribution and instead sample from the full prior distribution. This mismatch creates a gap between training and testing, leading to performance degradation. To bridge this gap, we propose conditional optimal transport FM $( \mathbf { C } ^ { 2 } \mathbf { O T } )$ which introduces a simple yet effective condition-aware weighting term when computing the cost matrix for OT. Additionally, we propose two techniques: adaptive weight finding to simplify hyperparameter tuning, and efficient oversampling of OT batches to counteract the reduced effective OT batch size introduced by the weighting term.

We conduct extensive experiments on both twodimensional synthetic data and high-dimensional image data, including CIFAR-10, ImageNet-32×32, and ImageNet-256×256. The results demonstrate that our method performs well across different condition types (discrete and continuous), datasets, network architectures (UNets and transformers), and data spaces (image space and latent space). C<sup>2</sup>OT achieves better overall performance than the existing baselines across different inference computation budgets, including with few-steps Euler’s method and with an adaptive ordinary differential equation (ODE) solver [10].

## 2. Related Works

Flow Matching. Recently, flow matching [1, 2, 28] has become a popular choice for generative modeling, in part due to its simple training objective and its ability to generate high-quality samples. However, at test time, flow-based methods typically require many computationally expensive forward passes through a deep net to numerically integrate the flow. This is because the underlying flow is usually curved and therefore cannot be approximated well with a few integration steps.

Optimal Transport Coupling. To straighten the flow, in the context of unconditional generation, Tong et al. [43] and Pooladian et al. [37] concurrently proposed minibatch optimal transport (OT). While OT has proven effective in the unconditional setting, we show in Section 3.2 that it skews the prior distribution and fails in the conditional setting if used naïvely. Our method, $\mathrm { C ^ { 2 } O T }$ , specifically addresses this failure, aiming to extend the success of OT in unconditional generation to conditional generation. Note, OT has also been used in equivariant flow matching [23, 42], which exploits domain-specific data symmetry (e.g., in molecules). In contrast, our method targets general data. Further, Hui et al. [18] find that mixing of OT and independent coupling benefits shape completion, which is an orthogonal contribution to this paper. Concurrently, Davtyan et al. [7] improve data-noise coupling for unconditional generation, while leaving data coupling for conditional generation to future work.

Straightening Flow Paths. Other methods of straightening flow paths exist. Reflow [31] achieves this by retraining a flow matching network multiple times, at the cost of additional training overhead. Variational flow matching [14] reduces flow ambiguity with a latent variable and hierarchical flow matching [48] straightens flows by modeling higher-order flows. These approaches are orthogonal to our focus on prior-data coupling and, in principle, can be combined with our method.

Learning the Prior Distribution. An alternative approach to improving flow is to jointly learn a prior distribution with the flow matching network. However, these methods [27, 30, 40] typically rely on a variational autoencoder (VAE) [17, 22] to learn the prior. This introduces yet another density estimation problem, which complicates training. In this work, we focus on improving training-free coupling plans without learning a new prior.

Consistency Models. Consistency models [41] have been proposed to accelerate diffusion model sampling, and can be extended to flow matching [46]. Recent advancements have improved training efficiency [13], stability [32], and quality [20]. These methods are orthogonal to our contribution, which focuses on improving prior-data coupling, and can be integrated, as demonstrated by Silvestri et al. [40] in their combination of consistency models with OT. In this paper, we focus on the base form of flow matching to avoid distractions from other formulations.

## 3. Method

## 3.1. Preliminaries

Flow Matching (FM). We base our discussion on FM [28, 43] (also called rectified flow [31]) for generative modeling and refer readers to [31, 43] for details. At test-time, a sample $\tilde { x } _ { 1 }$ is characterized by an ordinary differential equation (ODE) with an initial boundary value specified via noise $x _ { 0 }$ randomly drawn from a prior distribution $( e . g .$ , the standard normal). To compute the sample $\tilde { x } _ { 1 }$ , this ODE is commonly solved by numerically integrating from time $t = 0$ to time t = 1 a learned flow/velocity field $v _ { \theta }$ , where θ denotes the set of trainable deep net parameters. For conditional generation, we additionally obtain a condition $c , e . g .$ , a class label from user input, which adjusts the flow. We use $v _ { \theta , c }$ to denote this conditional flow. At training time, we find the deep net parameters θ by minimizing the objective

![](images/d3856faed8666e8ac80933a5a2270fb22b5a9d4f5b3ddeced74a964e8c71e22e.jpg)  
Figure 2. We visualize the coupling during training and the learned flow from $t = 0 \mathrm { t o } t = 1$ during testing for FM, OT, and our proposed method $\mathrm { C ^ { 2 } O T } .$ . The prior is defined as $q ( x _ { 0 } ) = \mathcal { N } ( 0 , 1 )$ , while the target distribution is given by $\begin{array} { r } { q ( x _ { 1 } ) = \frac { 1 } { 2 } q ( x _ { 1 } | c = 0 ) + \frac { 1 } { 2 } q ( x _ { 1 } | c = } \end{array}$ $\begin{array} { r } { 1 ) ) = \frac { 1 } { 2 } \mathcal { N } ( - 2 , 0 . 5 ) + \frac { 1 } { 2 } \mathcal { N } ( 2 , 0 . 5 ) } \end{array}$ . Note, FM results in curved flows during testing, while OT degenerates due to skewed training samples $( q ( x _ { 0 } | c ) \neq q ( x _ { 0 } ) )$ ). In contrast, our method successfully learns straight flows without degeneration.

$$
\mathbb {E} _ {t, (x _ {0}, x _ {1}, c) \sim \pi (x _ {0}, x _ {1}, c)} \| v _ {\theta , c} (t, x _ {t}) - u (x _ {t} | x _ {0}, x _ {1}) \| ^ {2},\tag{1}
$$

where t is uniformly drawn from the interval [0, 1], and where $\pi ( x _ { 0 } , x _ { 1 } , c )$ denotes the joint distribution of the prior and the training data. In practice, often, independent coupling is used, $i . e .$ , the prior and the training data are sampled independently, such that

$$
\pi (x _ {0}, x _ {1}, c) = q (x _ {0}) q (x _ {1}, c).\tag{2}
$$

Further,

$$
x _ {t} = t x _ {1} + (1 - t) x _ {0}\tag{3}
$$

defines a linear interpolation between noise and data, and

$$
u (x _ {t} | x _ {0}, x _ {1}) = x _ {1} - x _ {0}\tag{4}
$$

denotes its corresponding flow velocity at $x _ { t }$ . Note, simply dropping the condition c leads to the unconditional flow matching formulation.

Although rectified flow matching is trained with straight paths, the learned flow is often curved (Figure 1, bottomleft) since there are many possible paths induced by the independently sampled couplings $x _ { 0 } , x _ { 1 }$ through any $x _ { t }$ at time t [37]. Curved paths require more numerical integration and network evaluation steps, which is not ideal.

Optimal Transport (OT) Coupling. To address this issue for unconditional generation, Tong et al. [43] and Pooladian et al. [37] concurrently proposed to use minibatch optimal transport to deterministically couple the sampled prior and a sampled batch of the data. This minimizes flow path lengths and straightens the flow. Concretely, given a minibatch of b e-dimensional samples, an OT coupling seeks a b × b permutation matrix $P$ such that the squared Euclidean distance is minimized, $i . e .$

$$
\min _ {P} \| X _ {0} - P X _ {1} \| _ {2} ^ {2}.\tag{5}
$$

Here, $X _ { 0 } , X _ { 1 } \in \mathbb { R } ^ { b \times e }$ are matrices containing a minibatch of samples from the prior and the data. Instead of an “independent coupling”, in the unconditional setting we then optimize a variant of Equation (1) using the coupled tuple $( X _ { 0 } , P X _ { 1 } )$ ). Note that this algorithm corresponds to sampling tuples $( x _ { 0 } , x _ { 1 } )$ from a joint distribution $\pi _ { \mathrm { o t } } ( x _ { 0 } , x _ { 1 } )$

Importantly, for unconditional generation, $i . e .$ , when using the joint distribution $\pi _ { \mathrm { o t } } ( x _ { 0 } , x _ { 1 } )$ , the marginals remain unchanged from independent coupling (see [43]):

$$
\begin{array}{l} \int \pi_ {\mathrm{ot}} (x _ {0}, x _ {1}) q (x _ {0})   \mathrm{d} x _ {0} = q (x _ {1}), \text {and} \\ \int \pi_ {\mathrm{ot}} (x _ {0}, x _ {1}) q (x _ {1})   \mathrm{d} x _ {1} = q (x _ {0}). \end{array}\tag{6}
$$

Hence, during test-time, we can sample from the prior $q ( x _ { 0 } )$ and integrate the flow $v _ { \theta }$ to simulate data from the data distribution $q ( x _ { 1 } )$

Unfortunately, as shown in the next section, this does not hold for conditional generation. Said differently, naïvely extending the unconditional joint distribution $\pi _ { \mathrm { o t } } ( x _ { 0 } , x _ { 1 } )$ to a conditional joint distribution $\pi _ { \mathrm { o t } } ( x _ { 0 } , x _ { 1 } , c )$ by using the coupled tuple $( X _ { 0 } , P X _ { 1 } , P C )$ , where $C \in \mathbb { R } ^ { b \times h }$ is a batch of h-dimensional conditions, does not maintain marginals and leads to a gap between training time and test time. We discuss this next.

## 3.2. OT Coupling Skews The Prior

Intuition. Consider the one-dimensional example illustrated in Figure 2: the prior distribution $q ( x _ { 0 } )$ follows a Gaussian distribution, and the target distribution $q ( x _ { 1 } )$ consists of a mixture of two Gaussians translated left and right, with the translation direction indicated by a binary condition $c .$ When training with $\mathfrak { a } \pi _ { \mathrm { o t } } ( x _ { 0 } , x _ { 1 } , c )$ that represents the coupled tuple $( X _ { 0 } , P X _ { 1 } , P C )$ , samples from the prior and the sampled conditions become correlated through the OT coupling – samples from the left half of the prior distribution are always associated with $c = 0$ , while those from the right half always correspond to $c = 1$ . As a result, the model overfits on data pairs $( x _ { 0 } , c )$ where either $( x _ { 0 } < 0 , c = 0 )$ or $( x _ { 0 } > 0 , c = 1 )$ . However, during testing, this skewed distribution is no longer accessible, and instead, $x _ { 0 }$ and c are sampled independently. Consequently, the model encounters previously unseen data pairs, i.e., $( x _ { 0 } > 0 , c = 0 )$ and $( x _ { 0 } < 0 , c = 1 )$ – leading to failure. A similar phenomenon can be observed in the two-dimensional example illustrated in Figure 1.

Skewed Prior. Formally, imagine that we consider the most general OT coupling joint distribution $\pi _ { \mathrm { o t } } ( x _ { 0 } , x _ { 1 } , c )$ at training time. At test time, the user wants to specify a condition $c _ { 1 }$ to obtain samples from the conditional distribution $q ( x _ { 1 } | c = c _ { 1 } )$ . What is the prior distribution that we need to sample from such that we arrive at the correct conditional distribution $q ( x _ { 1 } | c = c _ { 1 } ) ?$ To compute this, imagine, we are given an arbitrary condition $c _ { 1 } , i . e .$ , we look at the induced distribution $\pi _ { \mathrm { o t } } ( x _ { 0 } , x _ { 1 } , c | c = c _ { 1 } )$ . Marginalizing this induced distribution over $x _ { 1 }$ yields

$$
\begin{array}{r l} \int \pi_ {\mathrm{ot}} (x _ {0}, x _ {1}, c | c = c _ {1}) \mathrm{d} x _ {1} & = \int \pi_ {\mathrm{ot}} (x _ {0}, x _ {1} | c = c _ {1}) \mathrm{d} x _ {1} \\ & = q (x _ {0} | c = c _ {1}), \end{array} \tag {7}
$$

which implies that the prior required to arrive at the conditional distribution $q ( x _ { 1 } | c = c _ { 1 } )$ is $q ( x _ { 0 } | c = c _ { 1 } ) - \mathbf { a }$ distribution that we cannot access at test time. In general, the prior that we need to sample from, given condition $c ,$ is skewed from $q ( x _ { 0 } )$ to $q ( x _ { 0 } | c )$ which is inaccessible at test time. As a result, accurately capturing $q ( x _ { 1 } | c )$ at test time becomes infeasible unless $q ( x _ { 0 } )$ and $q ( c )$ are independent, $i . e . , \operatorname { i f } q ( x _ { 0 } | c ) = q ( x _ { 0 } )$ . However, this condition is generally not satisfied by the most general OT coupling – for instance, in Figure 2, clearly $q ( x _ { 0 } | c ) \neq q ( x _ { 0 } )$ . To address this issue, we propose a method $\mathrm { C ^ { 2 } O T }$ to unskew the prior. We discuss this in the next section.

## 3.3. Conditional Optimal Transport FM

## 3.3.1. Formulation

We propose conditional optimal transport FM $( \mathbf { C } ^ { 2 } \mathbf { O T } )$ to unskew the prior $q ( x _ { 0 } | c )$ such that $q ( x _ { 0 } | c ) = q ( x _ { 0 } )$ . This ensures that at test time, we can sample from the full prior $q ( x _ { 0 } )$ , irrespective of the condition c. Importantly, at the same time, we also aim to preserve the straight flow paths provided by OT. Conceptually, we construct a prior distribution independently for each condition, $i . e .$ $q ( x _ { 0 } | c _ { 1 } ) = q ( x _ { 0 } | c _ { 2 } ) = q ( x _ { 0 } ) , \forall c _ { 1 } , c _ { 2 }$ . This can be achieved by sampling from the prior and computing OT independently for each condition, as shown in Figure 2 (right).

Formally, we construct the joint distribution as

$$
\pi_ {\mathrm{c} ^ {2} \mathrm{ot}} (x _ {0}, x _ {1}, c) := q (x _ {0}) q (c) \pi_ {\mathrm{ot}} (x _ {1} | x _ {0}, c).\tag{8}
$$

Here, the prior $q ( x _ { 0 } )$ and the condition $q ( c )$ are sampled independently, while the data $x _ { 1 }$ is conditioned on $x _ { 0 }$ (via OT) and c (via the dataset construction). Different from independent coupling in Equation (2), $x _ { 1 }$ and $x _ { 0 }$ are associated through OT and therefore lead to straighter flow paths. Also different from the most general OT coupling $\pi _ { \mathrm { o t } } ( x _ { 0 } , x _ { 1 } , c )$ we explicitly enforce the independence between $x _ { 0 }$ and $c .$

To see that this joint distribution provides an unskewed prior, imagine we are given an arbitrary input condition $c _ { 1 }$ , marginalizing over $x _ { 1 }$ yields

$$
\begin{array}{c} \int q (x _ {0}) q (c | c = c _ {1}) \pi_ {\mathrm{ot}} (x _ {1} | x _ {0}, c = c _ {1}) \mathrm{d} x _ {1} \\ = \int q (x _ {0}) \pi_ {\mathrm{ot}} (x _ {1} | x _ {0}, c = c _ {1}) \mathrm{d} x _ {1} = q (x _ {0}). \end{array}\tag{9}
$$

This implies that, at test time, we can sample from the entire prior $q ( x _ { 0 } )$ , regardless of the condition $c ,$ to arrive at the desired data distribution $q ( x _ { 1 } | c )$ . During training, in practice, we sample from $\pi _ { \mathrm { c } ^ { 2 } \mathrm { o t } }$ by modifying the optimal transport cost function in Equation (5). This process is exact for discrete conditions (Section 3.3.2) and approximate for continuous conditions (Section 3.3.3). Specifically, given a minibatch of b e-dimensional samples and a minibatch of b h-dimensional conditions $C \in \mathbb { R } ^ { b \times h }$ , C<sup>2</sup>OT seeks a $b \times b$ permutation matrix $P$ that minimizes the following cost function:

$$
\min _ {P} \| X _ {0} - P X _ {1} \| _ {2} ^ {2} + \sum_ {i = 1} ^ {b} f (C _ {i}, [ P C ] _ {i}),\tag{10}
$$

where $f$ is a symmetric and non-negative cost function, satisfying the property $f ( c , c ) = 0 \forall c$ . The key differences between independent coupling, OT coupling, and $\mathrm { C ^ { 2 } O T }$ coupling are highlighted in Algorithms 1 to 3. In the following two sections, we discuss our choice of $f$ for both discrete and continuous conditions.

## 3.3.2. Discrete Conditions

For discrete conditions, we assume the conditions correspond to class labels from a finite, discrete set, $i . e . , c \in$ $\{ 0 , 1 , \ldots , k - 1 \}$ . For example, in Figure 2, the labels 0 and 1 represent the left and right groups in the target distribution, respectively. We define $f = f _ { \mathrm { d } }$ such that no transport is allowed to modify the conditions $( i . e . , P C = C )$ . This is achieved via

```python
Algorithm 1 Independent Coupling
1: function INDEPENDENT(X₁, C)
2:    X₀ ~ N ▷ Sample from prior
3:
4:
5:    return X₀, X₁, C

Algorithm 2 OT Coupling
1: function OT(X₁, C)
2:    X₀ ~ N ▷ Sample from prior
3:    Cost ← pairwise_dist(X₀, X₁)
4:
5:    i, j ← Hungarian_matching(Cost)
6:    return X₀[i], X₁[j], C[j]

Algorithm 3 C²OT Coupling (Ours)
1: function C²OT(X₁, C)
2:    X₀ ~ N ▷ Sample from prior
3:    Cost ← pairwise_dist(X₀, X₁) + pairwise_dist(C, C)
4:
5:    i, j ← Hungarian_matching(Cost)
6:    return X₀[i], X₁[j], C[j]
```

$$
f _ {\mathrm{d}} (c _ {1}, c _ {2}) = \left\{ \begin{array}{l} \infty , c _ {1} \neq c _ {2}, \\ 0, \text { otherwise }. \end{array} \right.\tag{11}
$$

As a result, the unordered set of prior samples for any given condition c remains unchanged: $\{ X _ { 0 i } | c _ { i } = c \} \stackrel { \mathrm { i i d } } { \sim } q ( x _ { 0 } )$ This formulation samples exactly from our ideal (Equation (8)), as OT is computed independently within each class, and the data sample from each class is exposed to the full prior without skew. This effect is illustrated in Figure 1 (middle) and Figure 2 (right). As expected, the results differ from FM’s curved flow paths and $\mathrm { O T } { \mathrm { s } }$ inability to properly capture the target distribution.

It is important to note that enforcing $P C = C$ does not necessarily imply $P = I$ since we can still swap row i and $j$ in $P$ as long as $c _ { i } = c _ { j }$ . Setting $P = I$ would correspond to a degeneration back to independent coupling in Equation (2) and forfeit the benefits of straightened flow provided by OT.

Unfortunately, using $f _ { \mathrm { d } }$ with continuous conditions leads to exactly this degeneration, since, in general, no $c _ { i } = c _ { j }$ unless $i = j$ . Hence, in the next section, we devise a relaxed cost for handling continuous conditions.

## 3.3.3. Continuous Conditions

For continuous conditions, the condition is typically a feature embedding, e.g., computed from a text prompt. Directly applying $f _ { \mathrm { d } }$ in Equation (11) leads to a degeneration back to independent coupling as discussed earlier. To avoid this, we propose a relaxed penalty function $f _ { \mathrm { c } }$ based on the cosine distance:

$$
f _ {\mathrm{c}} \left(c _ {1}, c _ {2}\right) = w \left(1 - \frac {c _ {1} \cdot c _ {2}}{\| c _ {1} \| \| c _ {2} \|}\right) = w \cdot \mathfrak {c d i s t} \left(c _ {1}, c _ {2}\right).\tag{12}
$$

Here, $w > 0$ is a scaling hyperparameter. Note, when $w = 0$ this formulation degenerates to regular OT; when $w \to \infty ,$ this formulation approaches Equation (11), i.e., independent coupling if no two conditions are equal. The right section of Figure 1 illustrates an example where the x-coordinate of the target data point serves as the condition. Notably, $\mathrm { C ^ { 2 } O T }$ preserves straight flow paths without exhibiting $\mathrm { O T } { \mathrm { s } }$ apparent degradation.

Finding w. The optimal w is highly dependent on the data distribution and the magnitude of features. To alleviate the need for tuning the hyperparameter w, we propose to find w adaptively for each minibatch. For this, we develop a ratio $r ( w )$ which measures the proportion of samples that are considered “potential optimal transport candidates”, $i . e . ,$

$$
\begin{array}{r l} r (w) = \frac {1}{B ^ {2}} \sum_ {i, j} \mathbb {1} \left(\| X _ {0 i} - X _ {1 j} \| _ {2} ^ {2} + w \cdot \mathrm{cdist} (c _ {i}, c _ {j})\right) & \leq \| X _ {0 i} - X _ {1 i} \| _ {2} ^ {2}), \end{array}\tag{13}
$$

where $\mathbb { 1 } ( \cdot )$ is an indicator function that evaluates to one if the condition is satisfied and to zero otherwise. Then, we introduce a hyperparameter $r _ { \mathrm { t a r } }$ as the target ratio and find w such that $r ( w ) \approx r _ { \mathrm { t a r } } .$ . Note that setting $r _ { \mathrm { t a r } }$ is invariant to the scaling of distances between $x _ { 0 }$ and $x _ { 1 }$ . Namely, if $\| X _ { 0 i } - X _ { 1 j } \| _ { 2 } ^ { 2 }$ is scaled by $s , w$ would also be scaled by s to preserve the same $r _ { \mathrm { t a r } } .$ . This invariance is particularly useful when transitioning to a new dataset, such as changing image resolutions in image generation tasks. In our implementation, we search for w in two steps: first with exponential search to establish the upper/lower bounds, then with binary search to locate a precise value. The final w is used as the initial value in the subsequent minibatch. In practice, we observe w differs very little between minibatches (<1%) and the search converges quickly within 10 iterations with a negligible overhead. Next, we introduce oversampling, a simple technique to obtain more accurate OT couplings.

## 3.3.4. Oversampling OT Batches

OT vs. Deep Net Batch Sizes. Typically, the $^ { 6 6 } \mathrm { O T }$ batch size” b used to compute optimal transport is set equal to the “deep net batch $\mathrm { s i z e } ^ { \mathfrak { s } } b _ { n }$ used to compute forward/backward passes of the deep net [37, 43]. While the OT batch size controls the dynamics of prior-to-data assignments, the deep net batch size affects gradient variances, training efficiency, and memory usage. There is no reason why setting them equal should be optimal. Particularly, with our proposed $C ^ { 2 } \mathrm { O T } ,$ , the effective OT batch size is reduced since we restrict transport between samples with divergent conditions, which calls for an increased OT batch size b. At the same time, keeping the deep net batch size $b _ { n }$ unchanged prevents unwarranted side effects in other aspects of training.

Reduced Effective OT Batch Size. In the case of discrete conditions, OT is effectively performed within $\{ X _ { 0 i } , X _ { 1 i } | c _ { i } = c \}$ for each condition c. With k classes, the expected effective OT batch size is reduced to $\mathbb { E } ( | \{ i | c _ { i } =$ $c \} | ) = b / k$ . Empirically, we find that using a much larger OT batch size b than the network batch size $b _ { n }$ is helpful. This observation motivates oversampling.

Table 1. Results on the 8gaussians→moons dataset with different training coupling methods and different conditioning. Note that our method $\mathrm { C } ^ { \hat { 2 } } \mathrm { O T }$ does not apply in the unconditional setting.

<table><tr><td>Method</td><td>Euler-1 ( $W_{2}^{2} \downarrow$ )</td><td>Adaptive ( $W_{2}^{2} \downarrow$ )</td><td>NFE ↓</td></tr><tr><td colspan="4">Unconditional</td></tr><tr><td>FM</td><td> $6.232 \pm 0.186$ </td><td> $0.120 \pm 0.045$ </td><td> $93.22 \pm 1.81$ </td></tr><tr><td>OT</td><td> $\mathbf{0.072} \pm 0.025$ </td><td> $\mathbf{0.050} \pm 0.040$ </td><td> $\mathbf{36.08} \pm 2.57$ </td></tr><tr><td colspan="4">Binary conditions</td></tr><tr><td>FM</td><td> $2.573 \pm 0.112$ </td><td> $0.059 \pm 0.029$ </td><td> $91.10 \pm 2.77$ </td></tr><tr><td>OT</td><td> $0.483 \pm 0.059$ </td><td> $0.462 \pm 0.025$ </td><td> $\mathbf{32.89} \pm 0.95$ </td></tr><tr><td> $C^{2}OT$  (ours)</td><td> $\mathbf{0.048} \pm 0.010$ </td><td> $\mathbf{0.018} \pm 0.006$ </td><td> $58.88 \pm 1.48$ </td></tr><tr><td colspan="4">Continuous conditions</td></tr><tr><td>FM</td><td> $0.732 \pm 0.052$ </td><td> $0.028 \pm 0.010$ </td><td> $93.86 \pm 1.76$ </td></tr><tr><td>OT</td><td> $8.276 \pm 6.510$ </td><td> $2.143 \pm 1.993$ </td><td> $\mathbf{42.07} \pm 2.59$ </td></tr><tr><td> $C^{2}OT$  (ours)</td><td> $\mathbf{0.077} \pm 0.024$ </td><td> $\mathbf{0.013} \pm 0.003$ </td><td> $89.87 \pm 1.53$ </td></tr></table>

Efficiency. A natural concern with increasing b is the computational overhead, since optimal transport with Hungarian matching [25] has a time complexity of $O ( b ^ { 3 } )$ . However,

• Since each OT batch can be used across $\left( b / b _ { n } \right)$ forward/backward passes, the amortized cost per minibatch reduces to $O ( b ^ { 2 } b _ { n } )$

• We offload OT computation from the main process to the data workers, enabling data preparation on the CPU in parallel with forward/backward passes on the GPU.

With a modest choice of 8 data workers per GPU, we observe no wall-clock overhead for OT batch sizes up to 6,400 when training on CIFAR-10 and ImageNet, as data preparation is completed faster than the GPU can process batches.

## 4. Experiments

We experimentally verify our proposed method $\mathrm { C ^ { 2 } O T }$ in two sections: first on the two-dimensional synthetic dataset 8gaussians→moons, and then on high-dimensional image data, including CIFAR-10 [24] and ImageNet-1K [8]. For ImageNet-1K, we conduct experiments on both $3 2 \times 3 2$ images in image space and $2 5 6 \times 2 5 6$ images in latent space. Implementation details are provided in Appendix D.

## 4.1. Two-Dimensional Data

The two-dimensional 8gaussians→moons dataset maps a mixture of eight Gaussian distributions to two interleaving half-circles, as visualized in Figure 1. We consider three conditioning scenarios: (a) unconditional; (b) binary class conditions, where each class corresponds to one of the half-circles; and (c) continuous conditions, where the x-coordinate of the desired target point is given as the con dition. For (b), we apply our setup for discrete conditions (Section 3.3.2); for (c), we apply our setup for continuous conditions (Section 3.3.3) with $r _ { \mathrm { t a r } } = 0 . 0 1$

Metrics. We report the 2-Wasserstein distance $( W _ { 2 } ^ { 2 } )$ (following [43]) between 10K generated data points and 10K samples from the ground-truth moons distribution. We experiment with single-step Euler’s method (Euler-1) and the adaptive Dormand–Prince method [10] (dopri5, referred to as adaptive) for numerical integration. Additionally, we report the number of function evaluations (NFE) in the adaptive method, which is the number of forward passes through the neural network. We average the results over ten training runs with different random seeds and report mean±std.

Results. The results are summarized in Table 1 and visualized in Figure 1. In the unconditional setting, OT produces straighter flows, leading to better distribution matching with fewer NFEs, and our method does not apply. However, in the conditional setting, OT degrades and performs worse than regular FM in the adaptive setting due to the aforementioned train-test gap in OT coupling. Our method, $\mathrm { C ^ { 2 } O T }$ effectively models the target distribution with both Euler-1 and the adaptive method, improving upon the baselines.

## 4.2. High-Dimensional Image Data

We conduct experiments on CIFAR-10 [24] (image space, class-conditioned), ImageNet-32×32 [8] (image space, caption-conditioned), and ImageNet-256×256 [8] (latent space, caption-conditioned). For ImageNet, we use the captions provided by [26] and encode text features using a CLIP [38]-like DFN [12] text encoder as input conditions. Our results are summarized in Table 2.

Metrics. We report Fréchet inception distance (FID) [16] and the number of function evaluations (NFE) in the adaptive solver for all three datasets. To assess how well the generations adhere to the input conditions, we additionally compute condition adherence metrics. For CIFAR-10, we run a pretrained classifier [5] on the generated images to obtain logits and report the average cross entropy (CE) with the input class label. For ImageNet, we compute CLIP features [38] using SigLIP-2 [44] from the generated images and the input captions respectively, and report the average cosine similarities (CLIP).

CIFAR-10. We base our UNet architecture on Tong et al. [43], with details provided in Appendix D.2. The CIFAR-10 training set contains 50,000 32×32 color images across 10 object classes. During testing, we generate 50,000 images evenly distributed among classes and assess FID against the training set following [43]. Each model is trained five times with different random seeds and the results are reported as mean±std. We use an OT batch size b of 640 and a ratio $r _ { \mathrm { t a r } }$ of 0.01. Compared to FM, our method achieves better performance (FID and CE) in few-steps settings and comparable performance in many-steps settings. Compared to OT, our method significantly outperforms in the manystep setting. Although OT achieves slightly better FID in Euler-2, it performs worse in CE, indicating that it fails to follow the condition.

Table 2. Results of different training algorithms on image generation tasks. We bold the best-performing entry and underline the second-best. Our proposed method, $\mathrm { C } ^ { 2 } \mathrm { O T }$ , achieves the best overall performance – better than FM with few sampling steps and better than OT with more sampling steps. In cases where $\mathrm { C ^ { 2 } O T }$ performs second, our performance is often comparable to the best entry. Additionally, $\mathrm { C ^ { 2 } O T }$ has the most stable performance, with the smallest deviations across runs in most entries. Variations in CLIP scores are minimal.

<table><tr><td colspan="10">CIFAR-10 Class-Conditioned Generation</td></tr><tr><td></td><td>Method</td><td>Euler-2</td><td>Euler-5</td><td>Euler-10</td><td>Euler-25</td><td>Euler-50</td><td>Euler-100</td><td>Adaptive</td><td>NFE ↓</td></tr><tr><td rowspan="3"> $\rightarrow$ FID</td><td>FM</td><td> $105.481±1.619$ </td><td> $21.968±0.400$ </td><td> $\underline{10.479±0.237}$ </td><td> $\underline{5.640±0.117}$ </td><td> $\underline{4.128±0.069}$ </td><td> $\underline{3.429±0.054}$ </td><td> $\underline{2.922±0.039}$ </td><td> $\underline{124.0±2.5}$ </td></tr><tr><td>OT</td><td> $\underline{64.417±0.434}$ </td><td> $\underline{18.194±0.409}$ </td><td> $10.778±0.270$ </td><td> $6.821±0.186$ </td><td> $5.368±0.113$ </td><td> $4.671±0.066$ </td><td> $4.144±0.018$ </td><td> $\underline{125.7±0.4}$ </td></tr><tr><td> $C^{2}OT$ (ours)</td><td> $\underline{64.694±0.379}$ </td><td> $\underline{17.565±0.202}$ </td><td> $\underline{9.762±0.136}$ </td><td> $\underline{5.644±0.081}$ </td><td> $\underline{4.152±0.061}$ </td><td> $\underline{3.447±0.049}$ </td><td> $\underline{2.904±0.035}$ </td><td> $127.7±0.7$ </td></tr><tr><td rowspan="3"> $\rightarrow$ CE</td><td>FM</td><td> $3.343±0.022$ </td><td> $\underline{0.546±0.020}$ </td><td> $\underline{0.323±0.007}$ </td><td> $\underline{0.272±0.004}$ </td><td> $\underline{0.267±0.002}$ </td><td> $\underline{0.269±0.003}$ </td><td> $\underline{0.276±0.004}$ </td><td> $\underline{124.0±2.5}$ </td></tr><tr><td>OT</td><td> $\underline{2.529±0.010}$ </td><td> $0.624±0.012$ </td><td> $0.426±0.005$ </td><td> $0.369±0.004$ </td><td> $0.363±0.004$ </td><td> $0.364±0.005$ </td><td> $0.372±0.005$ </td><td> $\underline{125.7±0.4}$ </td></tr><tr><td> $C^{2}OT$ (ours)</td><td> $\underline{2.264±0.022}$ </td><td> $\underline{0.475±0.009}$ </td><td> $\underline{0.322±0.004}$ </td><td> $\underline{0.276±0.005}$ </td><td> $\underline{0.271±0.003}$ </td><td> $\underline{0.272±0.003}$ </td><td> $\underline{0.279±0.004}$ </td><td> $127.7±0.7$ </td></tr></table>

<table><tr><td colspan="9">ImageNet 32×32 Caption-Conditioned Generation</td><td></td></tr><tr><td rowspan="3"> $\rightarrow$ FID</td><td>FM</td><td>113.250±0.793</td><td>23.015±0.123</td><td>11.141±0.051</td><td>7.165±0.087</td><td>6.142±0.085</td><td>5.673±0.074</td><td>5.358±0.059</td><td>146.9±3.5</td></tr><tr><td>OT</td><td>81.317±0.369</td><td>20.763±0.154</td><td>11.360±0.089</td><td>7.854±0.077</td><td>6.899±0.079</td><td>6.469±0.071</td><td>6.220±0.065</td><td>137.7±0.2</td></tr><tr><td>C2OT (ours)</td><td>102.380±0.279</td><td>21.965±0.035</td><td>10.897±0.033</td><td>7.069±0.027</td><td>6.084±0.016</td><td>5.638±0.018</td><td>5.350±0.017</td><td>134.2±1.4</td></tr><tr><td rowspan="3"> $\leftarrow$ CLIP</td><td>FM</td><td>0.067±0.000</td><td>0.076±0.000</td><td>0.074±0.000</td><td>0.071±0.000</td><td>0.071±0.000</td><td>0.070±0.000</td><td>0.069±0.000</td><td>146.9±3.5</td></tr><tr><td>OT</td><td>0.067±0.000</td><td>0.071±0.000</td><td>0.070±0.000</td><td>0.068±0.000</td><td>0.068±0.000</td><td>0.067±0.000</td><td>0.067±0.000</td><td>137.7±0.2</td></tr><tr><td>C2OT (ours)</td><td>0.068±0.000</td><td>0.075±0.000</td><td>0.073±0.000</td><td>0.071±0.000</td><td>0.070±0.000</td><td>0.070±0.000</td><td>0.069±0.000</td><td>134.2±1.4</td></tr></table>

<table><tr><td rowspan="3"> $\rightarrow$ FID</td><td>FM</td><td>202.519±0.606</td><td>31.266±0.355</td><td>9.996±0.106</td><td>5.189±0.082</td><td>3.886±0.066</td><td>3.484±0.055</td><td>3.498±0.054</td><td>131.1±2.3</td></tr><tr><td>OT</td><td>193.376±1.665</td><td>42.780±2.132</td><td>15.638±1.510</td><td>7.514±1.034</td><td>5.693±0.998</td><td>5.360±1.016</td><td>5.726±1.074</td><td>125.6±6.5</td></tr><tr><td>C2OT (ours)</td><td>200.811±0.385</td><td>30.715±0.221</td><td>10.000±0.072</td><td>5.069±0.037</td><td>3.722±0.037</td><td>3.320±0.032</td><td>3.369±0.034</td><td>128.2±2.6</td></tr><tr><td rowspan="3"> $\leftarrow$ CLIP</td><td>FM</td><td>0.028±0.000</td><td>0.120±0.000</td><td>0.134±0.000</td><td>0.137±0.000</td><td>0.138±0.000</td><td>0.139±0.000</td><td>0.140±0.000</td><td>131.2±2.3</td></tr><tr><td>OT</td><td>0.031±0.000</td><td>0.107±0.003</td><td>0.120±0.003</td><td>0.123±0.003</td><td>0.124±0.003</td><td>0.124±0.004</td><td>0.125±0.004</td><td>125.6±6.5</td></tr><tr><td>C2OT (ours)</td><td>0.029±0.000</td><td>0.120±0.000</td><td>0.133±0.000</td><td>0.137±0.000</td><td>0.138±0.000</td><td>0.138±0.000</td><td>0.139±0.000</td><td>128.2±2.6</td></tr></table>

ImageNet 256×256 in Latent Space. Here, we explore latent flow matching models [39]. During training, the flow matching model learns to generate data in the latent space, which is encoded from images. At test-time, the generated latents are decoded into images using a pretrained decoder. We base our experiment on a recent transformer-based model, LightningDiT [47], demonstrating that our method is effective across different network architectures and data spaces. We use the officially provided “64 epochs” configuration for all experiments (due to computational constraints), which is not directly comparable to methods trained with 800 epochs. We follow the same evaluation protocol as ImageNet-32×32, and adopt the same OT batch size b of 6400 and the target ratio $r _ { \mathrm { t a r } }$ of 0.01. Each model is trained five times with different random seeds, and the results are reported as mean±std. Again, $\mathrm { C ^ { 2 } O T }$ has the best overall performance compared to both FM and OT. We visualize the generations in Figure 3.

ImageNet 32×32. We base our UNet architecture on Pooladian et al. [37], and list details in Appendix D.3. We train and evaluate on the face-blurred [45] ImageNet, following [37]. In contrast to prior works that assess FID against the training set, we assess FID against the validation set (49,997 images) , as the fine-grained nature of captions may lead to overfitting on the training set. Each model is trained three times with different random seeds and the results are reported as mean±std. Compared to CIFAR-10, there are much more variations in this dataset $( e . g .$ ., 1000 classes vs. 10 classes in CIFAR-10). Thus, we adopt a larger OT batch size b of 6400 while keeping the same target ratio $r _ { \mathrm { t a r } } 0 . 0 1$ as before. Similar to our findings in CIFAR-10, our method performs better or is comparable to FM; in few-steps settings, OT achieves better FID but worse condition following (CLIP). Our method strikes the best overall balance.

Limitations in High-Dimensional Data We observed that the benefits of $\mathrm { C ^ { 2 } O T }$ are less pronounced, though still visible, in higher-dimensional image data compared to lowdimensional data. We hypothesize this is because image data is more entangled, making the prior skew induced by OT less detrimental. We also observe statistically straighter paths in $\mathrm { C ^ { 2 } O T }$ compared to FM, with details given in Table A1.

![](images/09f18fe31898cbbc4fa1329522e5b7e07ececdb0b8e8a73877b252b9cf055f24.jpg)  
Figure 3. Visual comparisons of $2 5 6 \times 2 5 6$ images generated by the baselines and our approach with different amounts of sampling steps. Our approach converges faster and produces cleaner outputs. The input caption is “a small dog sitting on a carpet”.

## 4.3. Ablation Studies

## 4.3.1. Varying OT Batch Size b

Here, we analyze the effect of varying the OT batch size b. We conduct these experiments on the CIFAR-10 dataset, training each model three times with different random seeds for every choice of b. We plot the results in Figure 4. Increasing the OT batch size clearly improves FID in the fewsampling-steps setting (Euler-2). This is intuitive, because a larger OT batch size enables the coupling to approach the true OT results, i.e., dataset-level optimal transport. In turn, this leads to straighter paths. However, in the adaptive setting, increasing the OT batch size slightly worsens FID, though the effect is much less significant (the range of the blue y-axis is much smaller than the range of the red y-axis). We think this occurs because a more accurate OT coupling reduces variations in the training data (similar to having less data augmentations) which harms performance as the benefit of straighter flows diminishes in the adaptive setting. This observation aligns with the findings of Pooladian et al. [37]: computing OT coupling across GPUs (i.e., having a larger effective OT batch size) slightly harms results. In contrast to Pooladian et al. [37], though, we seek to counteract the reduced effective OT batch size induced by conditional optimal transport (as discussed in Section 3.3.4), and we do find reasonably large (640 for CIFAR, 6400 for ImageNet) OT batch sizes useful while not adding wall-clock overhead.

## 4.3.2. Varying Target Ratio $r _ { \mathbf { t a r } }$

Here, we analyze the effects of varying the target ratio $r _ { \mathrm { t a r } } ,$ as defined in Equation (13). We conduct these experiments on the ImageNet-32×32 dataset, training each model three times with different random seeds for every choice of $r _ { \mathrm { t a r } } .$ We plot the results in Figure 5. Recall that when $r  0 .$ w → ∞ and the method approaches FM; when $r \to 0 . 5 , ^ { 2 }$ w → 0 and the method approaches OT (Section 3.3.3). Our findings are similar to those in Section 4.3.1: as we approach true OT, whether by increasing OT batch size or increasing $r _ { \mathrm { t a r } } )$ , FID improves in the few-step setting and worsens in the adaptive setting.

![](images/5c60d8d6196eb80507d187fd0d455fcbc3c150cc49cf81a210004b9f4a7c04db.jpg)  
Figure 4. Changes in FID with respect to varying OT batch sizes b. We plot mean±std over three runs and represent std with a shaded region. OT (adaptive) and FM (Euler-2) perform worse than all other results in this plot, hence results are not shown (full plot in Appendix B).

![](images/3d03354037356ebc406a42193e8c9e8a321531d0684ed0e1bed64bc4d01332f0.jpg)  
Figure 5. Changes in FID with respect to varying target ratio $r _ { \mathrm { t a r } } .$ We plot mean±std over three runs and represent std with a shaded region. OT (adaptive) performs worse than all other results in this plot, hence results are not shown (full plot in Appendix B).

## 5. Conclusion

We first investigate and formalize the struggle of minibatch optimal transport in conditional generation. Based on the findings, we propose $\mathrm { C ^ { 2 } O T } .$ , a simple yet effective addition that corrects the degeneration of OT while maintaining straight integration paths. Extensive experiments verify the effectiveness of our method. We believe that these findings, along with the proposed simple technique, close a gap in current research on prior-data coupling.

Acknowledgment. This work is supported in part by NSF grants 2008387, 2045586, 2106825, NIFA award 2020-67021-32799, and OAC 2320345.

## References

[1] Michael Albergo and Eric Vanden-Eijnden. Building normalizing flows with stochastic interpolants. In Proc. ICLR, 2023. 2

[2] Michael Albergo, Nicholas Boffi, and Eric Vanden-Eijnden. Stochastic interpolants: A unifying framework for flows and diffusions. arXiv preprint arXiv:2303.08797, 2023. 2

[3] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. pi0: A vision-languageaction flow model for general robot control. arXiv, 2024. 1

[4] Ho Kei Cheng, Masato Ishii, Akio Hayakawa, Takashi Shibuya, Alexander Schwing, and Yuki Mitsufuji. Taming multimodal joint training for high-quality video-to-audio synthesis. CVPR, 2025. 1

[5] chenyaofo. Pytorch cifar models. https://github. com/chenyaofo/pytorch-cifar-models, 2023. 6

[6] Patryk Chrabaszcz, Ilya Loshchilov, and Frank Hutter. A downsampled variant of imagenet as an alternative to the cifar datasets. arXiv, 2017. 17

[7] Aram Davtyan, Leello Tadesse Dadi, Volkan Cevher, and Paolo Favaro. Faster inference of flow-based generative models via improved data-noise coupling. In ICLR, 2025. 2

[8] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In CVPR, 2009. 6, 17

[9] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. NeurIPS, 2021. 17

[10] John R Dormand and Peter J Prince. A family of embedded runge-kutta formulae. Journal of computational and applied mathematics, 1980. 2, 6

[11] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In ICML, 2024. 1

[12] Alex Fang, Albin Madappally Jose, Amit Jain, Ludwig Schmidt, Alexander Toshev, and Vaishaal Shankar. Data filtering networks. ICLR, 2024. 6, 17

[13] Zhengyang Geng, Ashwini Pokle, William Luo, Justin Lin, and J Zico Kolter. Consistency models made easy. ICLR, 2025. 2

[14] Pengsheng Guo and Alexander G Schwing. Variational rectified flow matching. arXiv, 2025. 2

[15] Dan Hendrycks and Kevin Gimpel. Gaussian error linear units (gelus). arXiv, 2016. 15

[16] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. NeurIPS, 2017. 6

[17] Irina Higgins, Loic Matthey, Arka Pal, Christopher Burgess, Xavier Glorot, Matthew Botvinick, Shakir Mohamed, and Alexander Lerchner. beta-vae: Learning basic visual concepts with a constrained variational framework. In ICLR, 2017. 2

[18] Ka-Hei Hui, Chao Liu, Xiaohui Zeng, Chi-Wing Fu, and Arash Vahdat. Not-so-optimal transport flows for 3d point cloud generation. arXiv, 2025. 2

[19] Gabriel Ilharco, Mitchell Wortsman, Ross Wightman, Cade Gordon, Nicholas Carlini, Rohan Taori, Achal Dave, Vaishaal Shankar, Hongseok Namkoong, John Miller, Hannaneh Hajishirzi, Ali Farhadi, and Ludwig Schmidt. Openclip, 2021. 17

[20] Dongjun Kim, Chieh-Hsin Lai, Wei-Hsiang Liao, Naoki Murata, Yuhta Takida, Toshimitsu Uesaka, Yutong He, Yuki Mitsufuji, and Stefano Ermon. Consistency trajectory models: Learning probability flow ode trajectory of diffusion. ICLR, 2024. 2

[21] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. ICLR, 2015. 16, 17

[22] Diederik P Kingma, Max Welling, et al. Auto-encoding variational bayes. In ICLR, 2014. 2

[23] Leon Klein, Andreas Krämer, and Frank Noé. Equivariant flow matching. NeurIPS, 36, 2023. 2

[24] Alex Krizhevsky and Geoffrey Hinton. Learning multiple layers of features from tiny images, 2009. 6

[25] Harold W. Kuhn. The hungarian method for the assignment problem. Naval research logistics quarterly, 1955. 6

[26] Visual Layer. Imagenet-1k-vl-enriched. https : //huggingface.co/datasets/visual- layer/ imagenet-1k-vl-enriched, 2024. 6, 17

[27] Sangyun Lee, Beomsu Kim, and Jong Chul Ye. Minimizing trajectory curvature of ode-based generative models. In ICML, 2023. 2

[28] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. ICLR, 2023. 1, 2, 17

[29] Yaron Lipman, Marton Havasi, Peter Holderrieth, Neta Shaul, Matt Le, Brian Karrer, Ricky TQ Chen, David Lopez-Paz, Heli Ben-Hamu, and Itai Gat. Flow matching guide and code. arXiv, 2024. 17

[30] Qihao Liu, Xi Yin, Alan Yuille, Andrew Brown, and Mannat Singh. Flowing from words to pixels: A framework for crossmodality evolution. arXiv preprint arXiv:2412.15213, 2024. 2

[31] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. ICLR, 2023. 2

[32] Cheng Lu and Yang Song. Simplifying, stabilizing and scaling continuous-time consistency models. ICLR, 2024. 2

[33] Ségolène Martin, Anne Gagneux, Paul Hagemann, and Gabriele Steidl. Pnp-flow: Plug-and-play image restoration with flow matching. ICLR, 2025. 1

[34] Gaurav Parmar, Richard Zhang, and Jun-Yan Zhu. On aliased resizing and surprising subtleties in gan evaluation. In CVPR, 2022. 17

[35] Michael Poli, Stefano Massaroli, Atsushi Yamashita, Hajime Asama, Jinkyoo Park, and Stefano Ermon. Torchdyn: Implicit models and neural numerical methods in pytorch. Physical Reasoning and Inductive Biasesfor the Real World at NeurIPS, 2021. 15

[36] Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi, Chih-Yao Ma, Ching-Yao Chuang, David Yan, Dhruv Choudhary, Dingkang Wang, Geet Sethi, Guan Pang, Haoyu Ma, Ishan Misra, Ji Hou, Jialiang Wang, Kiran Jagadeesh, Kunpeng Li, Luxin Zhang, Mannat Singh, Mary Williamson, Matt Le, Matthew Yu, Mitesh Kumar Singh, Peizhao Zhang, Peter Vajda, Quentin Duval, Rohit Girdhar, Roshan Sumbaly, Sai Saketh Rambhatla, Sam Tsai, Samaneh Azadi, Samyak Datta, Sanyuan Chen, Sean Bell, Sharadh Ramaswamy, Shelly Sheynin, Siddharth Bhattacharya, Simran Motwani, Tao Xu, Tianhe Li, Tingbo Hou, Wei-Ning Hsu, Xi Yin, Xiaoliang Dai, Yaniv Taigman, Yaqiao Luo, Yen-Cheng Liu, Yi-Chiao Wu, Yue Zhao, Yuval Kirstain, Zecheng He, Zijian He, Albert Pumarola, Ali Thabet, Artsiom Sanakoyeu, Arun Mallya, Baishan Guo, Boris Araya, Breena Kerr, Carleigh Wood, Ce Liu, Cen Peng, Dimitry Vengertsev, Edgar Schonfeld, Elliot Blanchard, Felix Juefei-Xu, Fraylie Nord, Jeff Liang, John Hoffman, Jonas Kohler, Kaolin Fire, Karthik Sivakumar, Lawrence Chen, Licheng Yu, Luya Gao, Markos Georgopoulos, Rashel Moritz, Sara K. Sampson, Shikai Li, Simone Parmeggiani, Steve Fine, Tara Fowler, Vladan Petrovic, and Yuming Du. Movie gen: A cast of media foundation models. arXiv, 2024. 1

[37] Aram-Alexandre Pooladian, Heli Ben-Hamu, Carles Domingo-Enrich, Brandon Amos, Yaron Lipman, and Ricky TQ Chen. Multisample flow matching: Straightening flows with minibatch couplings. ICML, 2023. 2, 3, 5, 7, 8, 13, 16, 17

[38] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021. 6

[39] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022. 7

[40] Gianluigi Silvestri, Luca Ambrogioni, Chieh-Hsin Lai, Yuhta Takida, and Yuki Mitsufuji. Training consistency models with variational noise coupling. arXiv preprint arXiv:2502.18197, 2025. 2

[41] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. In ICML, 2023. 2

[42] Yuxuan Song, Jingjing Gong, Minkai Xu, Ziyao Cao, Yanyan Lan, Stefano Ermon, Hao Zhou, and Wei-Ying Ma. Equivariant flow matching with hybrid probability transport for 3d molecule generation. NeurIPS, 36, 2023. 2

[43] Alexander Tong, Kilian Fatras, Nikolay Malkin, Guillaume Huguet, Yanlei Zhang, Jarrid Rector-Brooks, Guy Wolf, and Yoshua Bengio. Improving and generalizing flow-based generative models with minibatch optimal transport. TMLR, 2024. 2, 3, 5, 6, 15, 16, 17

[44] Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, et al. Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization,

and dense features. arXiv preprint arXiv:2502.14786, 2025. 6, 14, 17

[45] Kaiyu Yang, Jacqueline H Yau, Li Fei-Fei, Jia Deng, and Olga Russakovsky. A study of face obfuscation in imagenet. In ICML, 2022. 7, 17

[46] Ling Yang, Zixiang Zhang, Zhilong Zhang, Xingchao Liu, Minkai Xu, Wentao Zhang, Chenlin Meng, Stefano Ermon, and Bin Cui. Consistency flow matching: Defining straight flows with velocity consistency. arXiv, 2024. 2

[47] Jingfeng Yao and Xinggang Wang. Reconstruction vs. generation: Taming optimization dilemma in latent diffusion models. CVPR, 2025. 7, 17

[48] Yichi Zhang, Yici Yan, Alex Schwing, and Zhizhen Zhao. Towards hierarchical rectified flow. ICLR, 2025. 2

## Table of Contents

1 Introduction 1   
2 Related Works 2   
3 Method 2   
3.1 Preliminaries 2   
3.2 OT Coupling Skews The Prior 4   
3.3 Conditional Optimal Transport FM 4   
3.3.1 Formulation 4   
3.3.2 Discrete Conditions 4   
3.3.3 Continuous Conditions 5   
3.3.4 Oversampling OT Batches 5   
4 Experiments 6   
4.1 Two-Dimensional Data 6   
4.2 High-Dimensional Image Data 6   
4.3 Ablation Studies 8   
4.3.1 Varying OT Batch Size 8   
4.3.2 Varying Target Ratio 8   
5 Conclusion 8   
A Additional Experiments 13   
A.1 Path Straightness 13   
A.2 Class-to-Image 13   
A.3 Predicting Conditions from Coupled Prior 13   
A.4 Extended Results with Different OT Batch Sizes 13   
A.5 Extended Results with Different Target Ratios 14   
A.6 Reference Condition Adherence Metrics 14   
B Extended Plots 14   
C Data Coupling in 8 Gaussians $\rightarrow$ moons 15   
D Implementation Details 15   
D.1 Two-Dimensional Data 15   
D.2 CIFAR-10 17   
D.3 ImageNet-32 17   
D.4 ImageNet-256 17   
E Additional Generated Images 17   
E.1 CIFAR-10 18   
E.2 ImageNet-32 19   
E.3 ImageNet-256 20

## A. Additional Experiments

## A.1. Path Straightness

We measure path straightness using Eq. (18) from [37], with 10K samples (conditions sampled from the validation sets) and 25 steps in Tab. A1. Specifically, we measure

$$
\mathbb {E} _ {t, q (x _ {0})} \left[ \| v _ {\theta , c} (t, \psi_ {t} (x _ {0})) \| ^ {2} - \| \psi_ {1} (x _ {0}) - x _ {0} \| ^ {2} \right],\tag{A1}
$$

where $\psi _ { t } ( x _ { 0 } )$ denotes the numerical integration result when integrating $x _ { 0 }$ from 0 to t following the flow field represented by $v _ { \boldsymbol { \theta } , c } .$ Our method consistently achieves straighter paths than FM. Although OT has even straighter paths, the end points of these paths do not accurately model the target distribution, indicated by the higher FID.

<table><tr><td rowspan="2"></td><td colspan="2">CIFAR</td><td colspan="2">ImageNet32</td><td colspan="2">ImageNet256</td></tr><tr><td>Straightness ↓</td><td>FID ↓</td><td>Straightness ↓</td><td>FID ↓</td><td>Straightness ↓</td><td>FID ↓</td></tr><tr><td>FM</td><td>110.83±0.20</td><td>5.64</td><td>133.95±0.34</td><td>7.17</td><td>4803.0±79.8</td><td>3.90</td></tr><tr><td>OT</td><td>73.54±0.24</td><td>6.82</td><td>97.71±0.23</td><td>7.85</td><td>4088.6±37.5</td><td>7.48</td></tr><tr><td> $C^2OT$  (ours)</td><td>84.99±0.26</td><td>5.64</td><td>122.54±0.34</td><td>7.07</td><td>4625.2±54.5</td><td>3.70</td></tr></table>

Table A1. Path straightness and FID of FM, OT, and $\mathrm { C ^ { 2 } O T }$

## A.2. Class-to-Image

We supplement our existing CIFAR-10 class-to-image experiments (Table 2) with additional class-to-image results on ImageNet32. Results in Tab. A2 are consistent with existing findings.

<table><tr><td colspan="8">ImageNet 32×32 Class-Conditioned Generation</td></tr><tr><td>Method</td><td>Euler-2</td><td>Euler-5</td><td>Euler-10</td><td>Euler-25</td><td>Euler-50</td><td>Euler-100</td><td>Adaptive</td></tr><tr><td>FM</td><td>116.296</td><td>22.224</td><td>9.530</td><td>5.892</td><td>5.334</td><td>5.116</td><td>4.993</td></tr><tr><td>OT</td><td>71.700</td><td>20.703</td><td>11.385</td><td>8.065</td><td>7.492</td><td>7.303</td><td>7.316</td></tr><tr><td> $C^2OT$  (ours)</td><td>81.480</td><td>18.285</td><td>8.661</td><td>5.607</td><td>5.201</td><td>5.055</td><td>5.035</td></tr></table>

Table A2. Class-to-image performance comparisons on ImageNet-32.

## A.3. Predicting Conditions from Coupled Prior

We note that OT degrades less significantly in high-dim as it skews the prior less (but still does), since mini-batch OT becomes noisy in high-dim. To quantify, we train a condition-classifier with the coupled prior $( x _ { 0 } )$ as input $( e . g .$ , in Fig. 2 of the main paper, a classifier can perfectly predict the condition based on $x _ { 0 } )$ on CIFAR with varying resolutions and number of dims. We collect 100K couplings for training, and divide them into an 80:20 train/test split. Tab. A3 shows the results: as the number of dimensions increases, the classifier becomes less accurate (less prior skew) for OT. Both FM and $\mathrm { C ^ { 2 } O T }$ lead to unbiased prior, so the classifier performs at random guess accuracy (10%).

<table><tr><td>Method</td><td>2 × 2</td><td>4 × 4</td><td>8 × 8</td><td>16 × 16</td><td>32 × 32</td></tr><tr><td>FM</td><td>10.0%±0.3</td><td>9.9%±0.1</td><td>10.0%±0.3</td><td>9.9%±0.2</td><td>9.8%±0.1</td></tr><tr><td>OT</td><td>29.7%±0.4</td><td>27.3%±0.3</td><td>24.4%±0.4</td><td>22.6%±0.3</td><td>20.1%±0.3</td></tr><tr><td> $C^2OT$  (ours)</td><td>10.0%±0.3</td><td>10.0%±0.3</td><td>10.0%±0.3</td><td>9.9%±0.2</td><td>9.9%±0.2</td></tr></table>

Table A3. Test-set classification accuracy of predicting the conditioning class when given the coupled input noise in the CIFAR-10 dataset.

## A.4. Extended Results with Different OT Batch Sizes

We compare OT batch size scaling of OT and $\mathrm { C ^ { 2 } O T }$ in Table A4. The results are consistent with our findings in Section 4.3.1 – OT has better FID with few steps but does not align well with the input condition (worse CE), and increasing OT batch size improves few-step performance and slightly harms many-step performance.

<table><tr><td></td><td>Method (OT batch size)</td><td>Euler-2</td><td>Euler-5</td><td>Euler-10</td><td>Euler-25</td><td>Euler-50</td><td>Euler-100</td><td>Adaptive</td><td>NFE ↓</td></tr><tr><td rowspan="10"> $\overset{\rightarrow}{FID}$ </td><td>OT (128)</td><td>64.305±0.593</td><td>18.033±0.566</td><td>10.659±0.354</td><td>6.753±0.26</td><td>5.326±0.157</td><td>4.639±0.084</td><td>4.142±0.015</td><td>125.6±0.5</td></tr><tr><td>OT (640)</td><td>59.506±0.976</td><td>18.710±0.673</td><td>11.588±0.475</td><td>7.555±0.301</td><td>5.997±0.222</td><td>5.232±0.175</td><td>4.624±0.122</td><td>126.1±1.1</td></tr><tr><td>OT (1280)</td><td>57.656±1.075</td><td>18.206±2.063</td><td>11.284±1.707</td><td>7.452±1.116</td><td>6.036±0.696</td><td>5.374±0.408</td><td>4.895±0.044</td><td>125.1±6.5</td></tr><tr><td>OT (2560)</td><td>56.186±0.876</td><td>17.976±1.752</td><td>11.084±1.682</td><td>7.381±1.078</td><td>6.068±0.633</td><td>5.473±0.325</td><td>5.071±0.074</td><td>125.8±7.1</td></tr><tr><td>OT (5120)</td><td>54.966±0.212</td><td>19.399±0.194</td><td>12.563±0.112</td><td>8.498±0.056</td><td>6.869±0.038</td><td>6.042±0.042</td><td>5.352±0.074</td><td>128.0±3.0</td></tr><tr><td> $C^{2}OT$  (128)</td><td>74.517±0.075</td><td>18.281±0.443</td><td>9.704±0.326</td><td>5.536±0.193</td><td>4.077±0.135</td><td>3.391±0.101</td><td>2.880±0.059</td><td>124.0±2.5</td></tr><tr><td> $C^{2}OT$  (640)</td><td>64.849±0.474</td><td>17.572±0.263</td><td>9.770±0.195</td><td>5.634±0.112</td><td>4.150±0.083</td><td>3.446±0.068</td><td>2.905±0.047</td><td>127.2±0.3</td></tr><tr><td> $C^{2}OT$  (1280)</td><td>61.850±0.373</td><td>17.325±0.407</td><td>9.706±0.321</td><td>5.668±0.206</td><td>4.200±0.147</td><td>3.495±0.102</td><td>2.951±0.048</td><td>124.1±0.9</td></tr><tr><td> $C^{2}OT$  (2560)</td><td>59.361±0.324</td><td>17.213±0.297</td><td>9.795±0.200</td><td>5.712±0.149</td><td>4.213±0.102</td><td>3.496±0.068</td><td>2.942±0.030</td><td>125.5±0.7</td></tr><tr><td> $C^{2}OT$  (5120)</td><td>56.011±0.556</td><td>17.046±0.185</td><td>9.904±0.205</td><td>5.830±0.126</td><td>4.291±0.079</td><td>3.536±0.054</td><td>2.945±0.022</td><td>128.7±0.7</td></tr><tr><td rowspan="10"> $\overset{\rightarrow}{CE}$ </td><td>OT (128)</td><td>2.525±0.015</td><td>0.619±0.016</td><td>0.425±0.006</td><td>0.368±0.004</td><td>0.361±0.004</td><td>0.363±0.006</td><td>0.371±0.006</td><td>125.6±0.5</td></tr><tr><td>OT (640)</td><td>2.386±0.048</td><td>0.672±0.018</td><td>0.470±0.011</td><td>0.405±0.008</td><td>0.393±0.006</td><td>0.392±0.007</td><td>0.398±0.008</td><td>126.1±1.1</td></tr><tr><td>OT (1280)</td><td>2.348±0.057</td><td>0.690±0.030</td><td>0.487±0.015</td><td>0.427±0.009</td><td>0.422±0.005</td><td>0.425±0.003</td><td>0.435±0.007</td><td>125.1±6.5</td></tr><tr><td>OT (2560)</td><td>2.298±0.031</td><td>0.702±0.020</td><td>0.505±0.010</td><td>0.443±0.006</td><td>0.436±0.008</td><td>0.438±0.011</td><td>0.447±0.014</td><td>125.8±7.1</td></tr><tr><td>OT (5120)</td><td>2.277±0.029</td><td>0.742±0.006</td><td>0.537±0.012</td><td>0.464±0.009</td><td>0.453±0.008</td><td>0.453±0.007</td><td>0.461±0.006</td><td>128.0±3.0</td></tr><tr><td> $C^{2}OT$  (128)</td><td>2.636±0.033</td><td>0.485±0.007</td><td>0.317±0.004</td><td>0.270±0.004</td><td>0.265±0.001</td><td>0.266±0.002</td><td>0.271±0.004</td><td>124.0±2.5</td></tr><tr><td> $C^{2}OT$  (640)</td><td>2.270±0.021</td><td>0.477±0.012</td><td>0.321±0.005</td><td>0.276±0.008</td><td>0.272±0.004</td><td>0.273±0.005</td><td>0.280±0.005</td><td>127.2±0.3</td></tr><tr><td> $C^{2}OT$  (1280)</td><td>2.166±0.032</td><td>0.475±0.010</td><td>0.324±0.008</td><td>0.279±0.003</td><td>0.273±0.004</td><td>0.274±0.003</td><td>0.281±0.003</td><td>124.1±0.9</td></tr><tr><td> $C^{2}OT$  (2560)</td><td>2.049±0.005</td><td>0.468±0.005</td><td>0.321±0.003</td><td>0.278±0.006</td><td>0.272±0.007</td><td>0.272±0.007</td><td>0.277±0.008</td><td>125.5±0.7</td></tr><tr><td> $C^{2}OT$  (5120)</td><td>1.910±0.037</td><td>0.463±0.006</td><td>0.327±0.008</td><td>0.282±0.005</td><td>0.278±0.005</td><td>0.279±0.004</td><td>0.284±0.004</td><td>128.7±0.7</td></tr></table>

Table A4. Performance of OT and $\mathrm { C ^ { 2 } O T }$ in CIFAR-10 when trained with different OT batch sizes.

## A.5. Extended Results with Different Target Ratios

We list additional results on ImageNet-32 and ImageNet-256 in Table A5: $r _ { \mathrm { t a r } } = 0 . 0 1$ generally strikes a good balance, but we note that it might not be optimal for all datasets.

<table><tr><td colspan="8">ImageNet 32×32 Caption-Conditioned Generation</td></tr><tr><td> $r_{\text{tar}}$ </td><td>Euler-2</td><td>Euler-5</td><td>Euler-10</td><td>Euler-25</td><td>Euler-50</td><td>Euler-100</td><td>Adaptive</td></tr><tr><td>0.005</td><td>104.386</td><td>22.254</td><td>10.939</td><td>7.015</td><td>6.022</td><td>5.576</td><td>5.284</td></tr><tr><td>0.01</td><td>102.380</td><td>21.965</td><td>10.897</td><td>7.069</td><td>6.084</td><td>5.638</td><td>5.350</td></tr><tr><td>0.1</td><td>82.456</td><td>20.820</td><td>11.035</td><td>7.438</td><td>6.501</td><td>6.080</td><td>5.843</td></tr><tr><td colspan="8">ImageNet 256×256 Caption-Conditioned Generation</td></tr><tr><td>0.005</td><td>201.906</td><td>30.159</td><td>9.852</td><td>5.114</td><td>3.795</td><td>3.373</td><td>3.377</td></tr><tr><td>0.01</td><td>201.010</td><td>30.578</td><td>10.032</td><td>5.075</td><td>3.702</td><td>3.335</td><td>3.290</td></tr><tr><td>0.1</td><td>199.009</td><td>37.014</td><td>13.293</td><td>6.277</td><td>4.608</td><td>4.268</td><td>4.570</td></tr></table>

Table A5. Results with different $r _ { \mathrm { t a r } } .$

## A.6. Reference Condition Adherence Metrics

We measure condition adherence in Section 4 via two condition adherence metrics. On CIFAR-10, we compute the average cross-entropy of the logits predicted by a pretrained classifier<sup>3</sup> on the generated images against the ground-truth conditioning labels. On ImageNet, we compute the average cosine distance between the CLIP embeddings extracted from the generated images versus the conditioning captions, using SigLip-2 [44]<sup>4</sup>. For reference, we compute these metrics on the validation set using ground-truth images and present the results in Table A6.

## B. Extended Plots

Figures A1 and A2 extend Figures 5 and 6 of the main paper to include all data points.

<table><tr><td>Dataset</td><td>CE (↓)</td></tr><tr><td>CIFAR-10</td><td>0.0005</td></tr><tr><td>Dataset</td><td>CLIP (↑)</td></tr><tr><td>ImageNet 32×32</td><td>0.1119</td></tr><tr><td>ImageNet 256×256</td><td>0.1363</td></tr></table>

Table A6. Reference condition adherence metrics on ground-truth images.

![](images/1b7ae53385f021c5b91de005e559345d2909a1ae9646362d8abfae7063e873f6.jpg)  
Figure A1. Changes in FID with respect to varying OT batch sizes b. We plot mean±std over three runs and represent std with a shaded region.

![](images/ade85305a4e45b5174620fc0fad0c6ad3abc3440d92df28dfef254c4ca85f116.jpg)  
Figure A2. Changes in FID with respect to varying target ratio $r _ { \mathrm { t a r } } .$ We plot mean±std over three runs and represent std with a shaded region.

## C. Data Coupling in 8 Gaussians→moons

Figure A3 extends Figure 1 with an additional row that shows coupling during training. Clearly, OT samples form a biased distribution at training time in conditional generation as discussed. Since we cannot sample from this biased distribution at test-time, we obtain a gap between training and testing. This gap degrades the performance of OT.

## D. Implementation Details

## D.1. Two-Dimensional Data

Data. Following the implementation of Tong et al. [43], we generate the “moons” data using the torchdyn library [35], and the “8 Gaussians” using torchcfm [43].

Network. We employ a simple multi-layer perceptron (MLP) network for this dataset. Initially, the two-dimensional input (x and y coordinates) and the flow timestep (a scalar uniformly sampled from [0, 1]) are projected into the hidden dimension using individual linear layers. When an input condition is provided, it is similarly projected into the hidden dimension. Discrete conditions are encoded as -1 or +1, while continuous conditions are represented by the x-coordinate of the target data point. After projection, all input features are summed and processed through a network comprising three MLP modules. Each MLP module consists of two linear layers, where the first layer uses an expansion ratio of 4, and is followed by a GELU activation function [15]. We use a residual connection to incorporate the output of each MLP block. Finally, another linear layer projects the features to two dimensions to produce the output velocity. The hidden dimension is set to 128.

With continuous conditions (target x coordinates)  
![](images/11b4c763e8977d1ca5f6734e396cdfeb7b3a104bbb2128da6e79b07a06ddd0f3.jpg)

![](images/532741123525e40c6f27c948f48f60da90e2640b0c98d65b021c95de84db339b.jpg)

![](images/dbb42bfee82dfe1ebe1a70c9921358c03ddf67cf64624eb5c16944e0d37c06ad.jpg)  
Figure A3. We visualize the flows learned by different algorithms using the 8gaussians→moons dataset. Below each plot, we show the 2-Wasserstein distance (lower is better; mean±std over 10 runs). Compared to Figure 1, we have added a first row illustrating the prior-data coupling during training. Note that the OT coupled paths during training (first row) do cross. This is expected – the commonly referred to “no-crossing” property of OT coupling refers to the uniqueness of the pair (x<sub>0</sub>, x<sub>1</sub>) given x and t – at the same timestep t, no two paths may cross at x (see Proposition 3.4 in Tong et al. [43] and Theorem D.2 in Pooladian et al. [37]). Since we plot all timesteps simultaneously in this figure, there are apparent crossings. However, the intersecting paths do not share the same timestep t at the point of intersection

<table><tr><td></td><td>CIFAR-10</td><td>ImageNet-32</td></tr><tr><td>Channels</td><td>128</td><td>256</td></tr><tr><td>Depth</td><td>2</td><td>3</td></tr><tr><td>Channels multiple</td><td>1, 2, 2, 2</td><td>1, 2, 2, 2</td></tr><tr><td>Heads</td><td>4</td><td>4</td></tr><tr><td>Heads channels</td><td>64</td><td>64</td></tr><tr><td>Attention resolution</td><td>16</td><td>4</td></tr><tr><td>Dropout</td><td>0.0</td><td>0.0</td></tr><tr><td>Use scale shift norm</td><td>True</td><td>True</td></tr><tr><td>Batch size / GPU</td><td>128</td><td>128</td></tr><tr><td>GPUs</td><td>2</td><td>4</td></tr><tr><td>Effective batch size</td><td>256</td><td>512</td></tr><tr><td>Iterations</td><td>100k</td><td>300k</td></tr><tr><td>Learning rate</td><td>2.0e-4</td><td>1.0e-4</td></tr><tr><td>Learning rate scheduler</td><td>Warmup then constant</td><td>Warmup then linear decay</td></tr><tr><td>Warmup steps</td><td>5k</td><td>20k</td></tr><tr><td>OT batch size (per GPU, for  $C^2OT$ )</td><td>640</td><td>6400</td></tr></table>

Table A7. Hyperparameter settings for training on CIFAR-10 and ImageNet-32.

Training. We train each network for 20,000 iterations with the Adam [21] optimizer, a learning rate of 3e-4 without weight decay, and a “deep net” batch size of 256 for computing forward/backward passes/gradient updates. We use an OT batch size b of 1024 and a target ratio $r _ { \mathrm { t a r } }$ of 0.01.

## D.2. CIFAR-10

In CIFAR-10, we employ the UNet architecture used by Tong et al. [43]. We list our hyperparameters in Table A7, following the format in [28, 37, 43]. To accelerate training, we use bf16. We use the Adam [21] optimizer with the following parameters: $\beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 5$ , weight decay=0.0, and ϵ = 1e−8. For learning rate scheduling, we linearly increase the learning rate from 1.0e−8 to 2.0e−4 over 5,000 iterations and then keep the learning rate constant. The “use scale shift norm” denotes employing adaptive layer normalization to incorporate the input condition, as implemented in [9]. To stabilize training, we clip the gradient norm to 1.0. We report the results using an exponential moving average (EMA) model with a decay factor of 0.9999. For FID computation, we use the clean\_fid library [34] in legacy\_tensorflow mode following [43].

## D.3. ImageNet-32

Data. We use face-blurred ImageNet-1K [8, 45] following [29], and apply the downsampling script from [6]. Images are downsampled to 32××32 using the ‘box’ algorithm, and reference FID statistics are computed with respect to the downsampled validation set images. For text input, we use the captions provided by [26].

Network and Training. We largely follow the training pipeline described in Appendix D.2. We use a larger network following [37] and list our hyperparameters in Table A7. To encode text input, we use the openclip library [19] and the text encoder of the “DFN5B-CLIP-ViT-H-14” checkpoint [12], a pretrained CLIP-like model. CLIP feature vectors are normalized to unit norm before being used as input conditions. For learning rate scheduling, after the initial warmup phase, we linearly decay the learning rate to 1.0e−8 over time.

Evaluation. As stated in the main paper, we use 49,997 images from the validation set to compute FID. This is because the fine-grained nature of image captions might lead to overfitting, i.e., memorizing the training set. For CLIP score computation, we evaluate the cosine similarity between the input caption and the generated image using SigLIP-2 [44], with the ViT-SO400M-16-SigLIP2-256 checkpoint via the openclip library [19].

## D.4. ImageNet-256

For this dataset, we use the open-source implementation of LightningDiT [47] and train the models under the ‘64 epochs’ setting with minimal modifications to change the network from class-conditioned to caption-conditioned. In addition to integrating the coupling algorithms (OT and $\mathrm { C ^ { 2 } O T }$ , while the original LightningDiT [47] already employs FM), our modifications include:

1. Changing the input conditional mapping layer from an embedding layer (that takes a class label as input) to a linear layer (that takes CLIP features as input).

2. Adjusting the classifier-free guidance (CFG) scale. We find that the model benefits from a higher CFG scale when using caption conditioning. Specifically, we increase the CFG scale from 10.0 to 17.0, and adjust the CFG interval start parameter from 0.11 to 0.10.

For data and evaluation, we follow the same setup as described in Appendix D.3.

## E. Additional Generated Images

We present additional image generation results in this section. All showcased images are uncurated, meaning they were sampled completely at random. For consistency and direct comparison, we used the same random seed for each generation across different methods.

Euler-10  
Adaptive  
![](images/092d772bec9b23959397b45c32dd1fbb0a006921e85830a88a5d6979bcde960d.jpg)  
Figure A4. Uncurated generations in CIFAR-10, 10-per-class. We compare FM, OT, and $\mathrm { C } ^ { 2 } \mathrm { O T }$ with both 10-step Euler’s method and an adaptive solver for test-time numerical integration.

Euler-10  
Adaptive  
![](images/b92d448bcda14a6223d17b13aed3e55d4ef488afbb35ac52f108a022c1e54b01.jpg)  
Figure A5. Uncurated generations in ImageNet-32. We compare FM, OT, and C<sup>2</sup>OT with both 10-step Euler’s method and an adaptive solver for test-time numerical integration.

![](images/84ca99b04c9fff55e987cf15f744bdc7a18bc9192adf02dcb23bad206cc2ae13.jpg)  
Figure A6. Uncurated generations in ImageNet-256. We compare FM, OT, and C<sup>2</sup>OT with different amounts of sampling steps.

![](images/408bdb5b08c303f7ca439502a141527f3888da65d466b3a49b25f129f295d9fe.jpg)  
Figure A7. Uncurated generations in ImageNet-256. We compare FM, OT, and $\mathrm { C ^ { 2 } O T }$ with an adaptive solver for test-time numerical integration.

## F. DeltaAI Acknowledgment

This work used the DeltaAI system at the National Center for Supercomputing Applications through allocation CIS250008 from the Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support (ACCESS) program, which is supported by National Science Foundation grants 2138259, 2138286, 2138307, 2137603, and 2138296.