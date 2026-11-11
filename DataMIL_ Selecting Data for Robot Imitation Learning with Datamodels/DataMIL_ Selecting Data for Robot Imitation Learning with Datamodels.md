# DataMIL : Selecting Data for Robot Imitation Learning with Datamodels

Shivin Dass<sup>∗</sup> UT Austin

Alaa Khaddaj<sup>∗</sup> MIT

Logan Engstrom MIT

Aleksander M ˛adry MIT

Andrew Ilyas<sup>†</sup> Stanford University

Roberto Martín-Martín<sup>†</sup> UT Austin

Abstract: Recently, the robotics community has amassed ever larger and more diverse datasets to train generalist policies. However, while these policies achieve strong mean performance across a variety of tasks, they often underperform on individual, specialized tasks and require further tuning on newly acquired taskspecific data. Combining task-specific data with carefully curated subsets of large prior datasets via co-training can produce better specialized policies, but selecting data naively may actually harm downstream performance. To address this, we introduce DataMIL, a data selection framework built on the datamodels paradigm that reasons about data selection in an end-to-end manner, using the policy itself to identify which data points will most improve performance. Unlike standard practices that filter data using human notions of quality (e.g., based on semantic or visual similarity), DataMIL directly optimizes data selection for task success, allowing us to select data that improves the policy while dropping data that degrade it. To avoid performing expensive rollouts in the environment during selection, we introduce a surrogate loss function on task-specific data, allowing us to use DataMIL in the real world without degrading performance. We validate our approach on 60+ simulation and real-world manipulation tasks, notably showing successful data selection from the largest open collections of robot datasets (OXE); demonstrating consistent gains in success rates over prior works. Our results underscore the importance of end-to-end, performance-aware data selection for unlocking the potential of large prior datasets in robotics. Code and more information at our website.

Keywords: Imitation Learning, Data Curation, Large Robot Datasets

## 1 Introduction

Recently we have witnessed a revolution in robot learning: inspired by the successes of large-scale language and vision models, the robotics community has begun training large foundation policies [1–3] by amassing large diverse robotic datasets comprising of a variety of tasks, scenes, and robots [4–6]. The resulting generalist policies achieve a strong mean performance across tasks and environments, but often underperform on individual tasks [7, 8], highlighting a gap between generalization and task-specific competence. To bridge this gap, researchers have explored a post-training paradigm [2], where pre-trained foundation models are fine-tuned [2, 8] for specific tasks, though this process demands a considerable number of newly acquired task-specific demonstrations. As datasets grow increasingly large and diverse, a natural question arises: how can we identify and select datafrom within existing datasets to boost task performance?

Selecting data to train a high-performing model is a complex endeavor. Naively, it would require testing each subset of the data by retraining and evaluating the performance of the trained model. This is expensive for any sizable dataset, becoming infeasible in robotics, where evaluation involves policy rollouts in the real world—a time-consuming and often dangerous procedure. Prior data selection methods in robotics remove the dependency on policy rollouts by selecting data based on heuristics, i.e., assuming that the most useful data is the most similar in language description [9], visually [10], in motion [11], or in state-action pairs [12] to a small number of task-specific demonstrations. While intuitive (and effective in some cases), these heuristics often make several assumptions and fail to consider the real impact of a datapoint on policy performance (see Fig. 1 (left)).

In other fields such as natural language processing (NLP) and computer vision (CV), researchers have developed an efficient framework for data selection based on model performance: datamodels [13]. Training a datamodel is a meta-process in which the original learning algorithm and model are viewed as a “black box” that consumes data, and the goal is to train an estimator of the black box’s behavior as a function of the input data. By avoiding selection based on heuristics, datamodels stay close to the true optimization objective (trained model performance) while critically reducing the amount of training and evaluation steps necessary with the original learning algorithm to an acceptable level for NLP and CV. However, these evaluations are still infeasible for policy learning due to the need for real-world rollouts, impeding their application to robotics.

In this work, we introduce DataMIL (Datamodels for Imitation Learning), a method that extends the success of datamodels to robotics by addressing the unique challenges of data-driven data selection. DataMIL trains a data-quality estimator using a tractable surrogate objective. We demonstrate empirically that this objective retains sufficient correlations to the true objective (trained policy performance), allowing us to train datamodels that predict data influence without requiring expensive real-world rollouts. Moreover, thanks to a process that remains closer to the true objective and avoids assumptions about the characteristics of the useful data, DataMIL selects and curates datasets for hard cases (e.g., different embodiments, multi-task settings) where prior heuristic-based methods are unsuitable. Across 50 tasks in MetaWorld [14], we show a 10% performance boost compared to the state-of-the-art baselines. We then show how datamodels can be estimated efficiently for larger policies such as Octo [15] using improved datamodel estimators based on metagradients [16] and show task-specific dataset curation in ten tasks from the LIBERO benchmark [17]. Finally, we effectively select data from OXE [4], one of the largest open collections of robot datasets, demonstrating DataMIL’s efficacy in selecting datasets for new tasks and embodiments on real hardware.

## 2 Related Work

Data curation for robot learning. Recent advances in robotics have leveraged ever-larger demonstration collections—both in simulation [18–20], and on real hardware [5, 21, 22]—to train generalist policies capable of tackling diverse tasks [1, 7, 2, 3]. However, the sheer scale and heterogeneity of these datasets (varying robots, scenes, and objectives) has motivated a body of work on data curation. For generalist training, methods like Re-Mix [23] use DoReMi [24] style optimization to learn optimal mixtures of data domains for improving model training, while others identify “high-quality” trajectories via mutual information criteria [25] or by scoring samples with policy rollouts [26].

Beyond generalist policy training, many studies have focused on task-specific dataset selection: given a handful of target demonstrations, one can sub-sample large datasets based on visual similarity [10], motion cues [27], or state–action closeness [12]. Recent work [28], aims at fusing datasets aggregated via different similarity metrics by scoring each data subset and re-sampling them according to a weight proportional to this score. While these approaches capture human notions of quality, they remain agnostic to each sample’s actual impact on downstream policy performance. Concurrent to our work, CUPID [29] focuses on single task curation by using a policy-gradient influence measure estimated via online rollouts. In contrast, DataMIL demonstrates selection from large heterogeneous datasets, scoring datapoints entirely in an offline manner. This allows us to estimate each datapoint’s contribution to final task success and curate training sets in an end-to-end, …performance-aware fashion.

![](images/c23f7b2ebd3c8677a89a6e5e9e5b593c93a196a58d5c852c52bbe9228696f9e1.jpg)  
Figure 1: Data selection with datamodels. (left) Similarity-based methods select close samples (yellow), but these aren’t always beneficial for learning. DataMIL evaluates data based on its impact selected dataon policy performance, selecting the samples that lead to policy improvement. (center) We estimate datamodels that score each sample by its influence on policy performance and select the highest-Lossscoring samples for training. (right) DataMIL explores two datamodel estimation methods, adapted to robotics: regression and metagradient-based estimators (see Sec. 4.1).

Datamodels and data attribution. Our work draws from a line of work in machine learning on data attribution [30–35] and, in particular, the datamodels framework [13, 33]. At a high level, this framework seeks to predict the behavior of machine learning models as a function of the data they are trained on. These ideas have been explored in the fields of CV and NLP for improving language model pre-training [36] and instruction tuning [37, 38, 16]; for increasing worst-group robustness [39]; and for removing outliers in supervised learning settings [40]. Our work builds on this body of research, while—as we discuss in Sec. 4.2—also tackling the unique challenges posed by the robot learning setting.

## 3 Preliminaries: Data Selection for Policies and Datamodels

In this section, we first provide some general background on the problem of data selection for robot learning, and then describe the datamodels framework on which our method is based.

Policy learning. The focus of our work is on the imitation learning problem. Here, our goal is to learn a policy π that maps states s to distributions over actions a using a collection of training trajectories (or demonstrations) $\mathcal { D } = ( \tau _ { 1 } , \tau _ { 2 } \cdot \cdot \cdot \tau _ { n } )$ of state-action pairs. We will define a policy learning algorithm as a function that takes as input a dataset of demonstrations and outputs an optimized policy π. We measure the performance of the policy π using a metric $\mathcal { M } : \pi  \mathbb { R } .$ which is a function mapping policies to a scalar value. The most common choice of metric is success rate—the fraction of times that sampling actions from the policy results in the policy completing a given task—but our notation is general and can also capture other choices of metric .

Data selection in robotics. In data selection for robotics, we are given (a) a prior dataset  of demonstrations; (b) a fixed learning algorithm (e.g., stochastic gradient descent on an imitation learning objective); and (c) a target metric that we will use to measure policy performance. Our goal is to select a subset of the data $\mathcal { D } ^ { \prime } \subset \mathcal { D }$ such that applying the algorithm  to the subset $\mathcal { D } ^ { \prime }$ yields a policy π that maximizes the target . Formally, we aim to find

$$
\operatorname * {a r g   m a x} _ {\mathcal {D} ^ {\prime} \subset \mathcal {D}} \mathcal {M} (\mathcal {A} (\mathcal {D} ^ {\prime})).\tag{1}
$$

Solving this optimization problem is challenging since algorithm itself is expensive to compute (it involves training a policy on $\mathcal { D } ^ { \prime } )$ . Thus, exhaustive search over all subsets of $\mathcal { D }$ is infeasible.

Datamodels. In this work, we leverage data attribution [30, 31, 33, 41] (and specifically, the datamodels framework [13]) to tackle the data selection problem in Eq. 1. The key idea behind datamod els is to directly approximate the target metric $\mathcal { M } ( A ( \mathcal { D } ^ { \prime } ) )$ as a function of the data $\mathcal { D } ^ { \prime }$ , allowing us to answer questions like “what would the performance of the policy be if we trained on this subset of the data?” without actually training the policy. More precisely, a datamodel is a model $\hat { f } : 2 ^ { D } \to \mathbb { R }$ that takes as input a subset of the data $\mathcal { D } ^ { \prime } \subset \mathcal { D }$ and outputs an estimate of $\mathcal { M } ( A ( \mathcal { D ^ { \prime } } ) )$ .

Informal Definition 1 (Datamodeling problem) Given a prior dataset  and a learning algorithm , the datamodeling problem is the problem of predicting the behavior of a model trained on a subdataset $\mathcal { D } ^ { \prime } \subset \mathcal { D }$ without actually training a model.

If we had such an approximation in hand (assuming for now that we can compute one), we would approach the data selection problem in Eq. 1 by solving the following optimization problem:

$$
\operatorname * {a r g   m a x} _ {\mathcal {D} ^ {\prime} \subset \mathcal {D}} \hat {f} (\mathcal {D} ^ {\prime}).\tag{2}
$$

This problem is much more tractable than the original problem in Eq. 1 because the datamodel $\hat { f }$ is typically much cheaper to compute than the learning algorithm and target metric $\mathcal { M } ( A ( \cdot ) )$ . For example, in supervised learning, recent works have shown that even linear datamodels—functions $\hat { f }$ that decompose additively in terms of their inputs —can be accurate predictors of model performance [13, 40, 33, 41, 34]. Intuitively, this means that we can assign a scalar value to each data point in that indicates how much it contributes to the performance of the learning algorithm .

Since leveraging an accurate datamodel $\hat { f }$ for data selection is straightforward, the main challenge in our work is constructing such a datamodel. That is, we need to find a way to build a function $\hat { f }$ that can predict, with nontrivial accuracy, the performance of the learning algorithm  on any given subset of the data $\mathcal { D } ^ { \prime }$ without actually training a policy on that subset. In the next section, we describe our methods for constructing such a datamodel in the policy learning setting.

## 4 DataMIL: Datamodels for Robot Imitation Learning

An overview of our training and selection methodology is shown in Fig. 1. Below, we provide the details of each component, beginning with the estimation of datamodels (Sec. 4.1), adapting them to robotics (Sec. 4.2) and selecting data using datamodels to train policies (Sec. 4.3).

## 4.1 Estimating Datamodels

In our work, we consider two ways of estimating datamodels: the regression method [13] and the metagradient method [16]. Both estimators approximate the outcome of model training linearly, in the sense that, for any training subset $\mathcal { D } ^ { \prime } \subset \mathcal { D }$ , the datamodel prediction $\hat { f } ( \mathcal { D } ^ { \prime } )$ takes the form,

$$
\hat {f} (\mathcal {D} ^ {\prime}) = \sum_ {z _ {i} \in \mathcal {D} ^ {\prime}} \tau (z _ {i}).
$$

Intuitively, $\tau ( z _ { i } )$ captures the importance of the training example $z _ { i }$ to the target metric $\mathcal { M } ( A ( \mathcal { D } ^ { \prime } ) )$ (more precisely, $\tau ( z _ { i } )$ is the additive effect of $z _ { i }$ on the target metric). Linear datamodels are convenient in the context of data selection, since they allow us to solve Eq. 1 by simply selecting the training examples with the highest scores τ $\Im ( z _ { i } )$ . Both estimators below take this form, differing only in how they compute $\tau ( z _ { i } )$

Regression estimator. The regression estimator is a straightforward but expensive way to estimate datamodels—it involves precomputing the scores $\tau ( z _ { i } )$ for each training example $z _ { i }$ in the broader training set $\mathcal { D } _ { \ell }$ . Concretely, we first sample m random subsets of the prior dataset $\mathcal { D } _ { j } \subset \mathcal { D } ;$ for each of these datasets, we train a policy $\boldsymbol { \mathcal { A } } ( \mathcal { D } _ { j } )$ , and evaluate the target metric . In our setting, evaluating  means rolling out the policy several times and computing the success rate. This gives us $\mathcal { M } _ { j } = \mathcal { M } ( \mathcal { A } ( \mathcal { D } _ { j } ) )$ for each subset $\mathcal { D } _ { j } ,$ , creating a dataset of m data-performance pairs $( \mathcal { D } _ { j } , \mathcal { M } _ { j } )$ for estimating datamodels. We compute the scores $\tau ( z _ { i } )$ for all n training examples by solving the

following minimization problem:

$$
\left\{\tau (z _ {1}), \dots , \tau (z _ {n}) \right\} := \arg \min _ {\boldsymbol {\tau} \in \mathbb {R} ^ {n}} \sum_ {j = 1} ^ {m} \left(\sum_ {j: z _ {i} \in D _ {j}} \tau_ {i} - \mathcal {M} (\mathcal {A} (D _ {j}))\right) ^ {2}.\tag{3}
$$

Above, observe that the sum $\sum _ { j : z _ { i } \in D _ { i } } \tau _ { j }$ is precisely the datamodel prediction of $\mathcal { M } ( A ( D _ { i } ) )$ when setting $\tau ( z _ { j } ) = \tau _ { j }$ . Thus, Eq. 3 corresponds exactly to linearly regressing the target metric onto the presence of each training point $z _ { i } .$ . More details are provided in App. B.2

Metagradient-based estimator. The regression-based estimator requires training a large number of models, which quickly becomes computationally prohibitive for complex models such as the ones used for modern visuomotor policies. A method to alleviate this challenge is to use a metagradientbased estimator. To that end, we parameterize the dataset by a vector $\mathbf { w } \in [ 0 , 1 ] ^ { n }$ , where $w _ { i }$ specifies the weight of the i-th training sample (i is included in the dataset if $w _ { i } = 1$ and excluded if $w _ { i } = 0 )$ Treating these weights as meta-parameters, we write a first-order Taylor expansion to our objective $\mathcal { M } ( \mathcal { A } ( \mathbf { w } ) )$ from Eq. 1 as,

$$
\mathcal {M} (\mathcal {A} (\mathbf {w})) \approx \mathcal {M} (\mathcal {A} (\mathbf {w} _ {0})) + \nabla_ {\mathbf {w}} \mathcal {M} (\mathcal {A} (\mathbf {w} _ {0})) ^ {\top} (\mathbf {w} - \mathbf {w} _ {0}),\tag{4}
$$

where $\mathbf { w } _ { 0 }$ is the all-ones vector, corresponding to training on the entire dataset. The gradient $\mathcal { Z } =$ $\nabla _ { \mathbf { w } } \mathcal { M } ( \mathcal { A } ( \mathbf { w } _ { 0 } ) )$ , known as the influence function [42, 30], captures how changes in data weights affect model performance.

Naïvely computing requires differentiating through the training process $\mathcal { A } ( \mathbf { w } ) { \mathrm { - i . e . } }$ , unrolling policy optimization with respect to the meta-parameters w. This computation, known as the metagradient, has traditionally been notoriously difficult, and consequently most prior works in data attribution have focused on approximating [30, 33, 41]. Recent advances, however, demonstrate that metagradients can be computed exactly and efficiently by leveraging the iterative structure of standard optimization algorithms such as stochastic gradient descent (SGD) [16, 43]. Specifically, Engstrom et al. [16] exploit step-wise auto-differentiation combined with efficient data structures to keep the memory costs practical. The resulting influence scores $\mathcal { T } _ { i }$ quantify how positively or negatively the i-th training sample affects performance, and can be directly used as datamodel coefficients. Ilyas and Engstrom [35] shows that indeed, doing this reparameterization and Taylor expansion with the metagradient actually does give you very good estimates of model performance. A detailed description and pseudocode of the metagradient-based estimator is provided in App. B.3.

## 4.2 Adapting Datamodels for Robotics

While datamodels have been applied to language modeling [33, 41] and computer vision [30, 13] tasks, there are some unique challenges that we face when applying them to robotics. Below, we describe how DataMIL extends the datamodel framework to handle robotic datasets efficiently.

Estimating datamodels without rollouts. In principle, the ideal target metric  is the policy’s true success rate under environment rollouts. However, real-world rollouts are expensive, and using them directly renders the objective non-differentiable, preventing the use of metagradient-based estimators that exploit differentiability of the evaluation metric. To overcome these limitations, we introduce a proxy metric $\widehat { \mathcal { M } }$ that (1) requires no additional rollouts and (2) is fully differentiable. Concretely, given a small held-out demonstration set $\mathcal { D } _ { t a r g e t }$ for the target task, we define:

$$
\widehat {\mathcal {M}} (\pi , \mathcal {D} _ {t a r g e t}) = \frac {1}{| \mathcal {D} _ {t a r g e t} |} \sum_ {(s, a) \in \mathcal {D} _ {t a r g e t}} - \mathcal {L} _ {B C} (\pi (s), a)\tag{5}
$$

Where $\mathcal { L } _ { B C } ( \pi ( s ) , a )$ defines the policy loss on a training example $( s , a )$ (see App. D for the exact objective for different policy classes). Hence, the true target metric  can be substituted by the proxy metric $\widehat { \mathcal { M } }$ in our original optimization $( \mathrm { E q . 1 } )$ , resulting in a more tractable and end-end differentiable objective for applying datamodels to robotic settings.

A natural question is whether $\widehat { \mathcal { M } }$ is a sufficiently good approximation of the true target to enable data selection. We study this question using the pick-place-wall task from Meta-World [14] (results on more tasks are provided in App. C.1), where we consider three different datamodel estimation techniques:

i) DM-rollouts: using regression-based estimator to estimate a datamodel for the “true” target (success rate via rollouts);

ii) DataMIL-rg: using regression-based estimator to estimate a datamodel for proxy target $\widehat { \mathcal { M } }$ (loss on a heldout set);

![](images/0724aef8346afc9a48c7eb8f47d76f5f702851151f4a62f753d60b812d533dae.jpg)

iii) DataMIL-meta: using metagradient-based estimator to estimate a datamodel for proxy target ${ \widehat { \mathcal { M } } } .$

Figure 2: Comparing true rollout success ( ) vs. proxy metric $\widehat { ( \mathcal { M } ) }$

We use each of these datamodels to select the top 10% of samples (as ranked by their estimated coefficient $\tau ( z _ { i } ) )$ from a multi-task prior dataset consisting of a mix of expert and suboptimal demonstrations (see Sec. 5 for details). We then measure the true success rate of a policy trained on the selected samples, and visualize the results in Fig. 2. Our results show that (a) selecting data for the proxy metric $\widehat { \mathcal { M } }$ incurs only a marginal drop in final success; (b) applying the metagradient-based estimator incurs another small drop in success but are significantly faster to train. Moreover, policies trained on selected data vastly outperform baselines trained on the entire dataset or $\mathcal { D } _ { t a r g e t }$ alone, achieving 7 higher success than the all-data policy, while target-only policy fails almost entirely.

These results demonstrate that (a) while validation loss is often a noisy predictor of performance in robotics, it still provides a useful signal for data selection, and our proxy objective can effectively replace expensive rollouts, and (b) data curation is critical: naively using all data or only target examples yields suboptimal policies, whereas curated datasets substantially boost task success.

Clustering training examples. Datamodels measure the influence of each training example on overall policy performance. In robotics, a training example may range from a single state–action pair, to sequences of state–action pairs (for sequential policies). Estimating influence at the individual training example level is often noisy: because each training example is seen only a few times during training, its individual effect on policy performance is often marginal and difficult to measure precisely. To reduce this variance, prior works clusters samples by class or task and then evaluates cluster-level influence instead of individual examples [44, 45].

This is particularly suitable for robotics since data naturally clusters into state–action sequences, so we can group samples at different temporal scales—e.g. sub-trajectories [10], tasks [9], or entire domains [23]. Fine-grained clusters offer precise selection but suffer from high noise, while coarse clusters yield more stable influence estimates at the cost of detail. We found that the optimal clustering strategy is a function of the dataset size given a fixed compute budget: the larger the dataset, the fewer times each sample is seen during training, leading to noisier individual influence estimates. On mid-sized datasets like LIBERO, sub-trajectory clustering provides a balance between granularity and noise (see App. C.4), whereas on large-scale datasets such as OXE, aggregating trajectory-level influence yields more reliable influence estimates.

Reducing distribution shift. Distribution shift is a particularly pronounced challenge in robotics: slight changes in lighting, camera pose, or robot dynamics can dramatically alter the data distribution. This issue becomes more severe when selecting data from large, heterogeneous datasets comprising of a variety of different embodiments, scenes, and tasks. Since datamodels rely on training policies over the prior data to estimate their impact on target task performance, excessive shift between the training and target distributions can lead to poor datamodel estimates. To mitigate this, we include a small fraction of target task data during datamodel estimation to better align the policy’s learning with the target domain. Specifically, we split $\mathcal { D } _ { t a r g e t }$ in half, include one half alongside the prior data for datamodel estimation, and reserve the other half purely for evaluating the proxy objective. We apply this technique only in real-world settings (i.e., OXE) since in simulation (i.e., MetaWorld, LIBERO) the target data typically comes from a similar distribution as the prior data.

## 4.3 Data Selection and Policy Training

By applying our proposed modifications in Sec. 4.2 to the datamodel estimators described in Sec. 4.1, we obtain a per-cluster attribution score on policy performance. While these attribution scores can be used in a variety of different ways, in this work we use them to curate a training subset: we select the top x% of prior examples with the highest positive influence to form $\mathcal { D } _ { s e l }$ . We then train the downstream policy π via behavior cloning on $\mathcal { D } _ { s e l }$ and $\mathcal { D } _ { t a r g e t }$ using a co-training recipe [11, 46, 21, 47]: at each training step, we sample from $\mathcal { D } _ { t a r g e t }$ with probability α and from $\mathcal { D } _ { s e l }$ with probability $1 - \alpha$

In summary, DataMIL leverages datamodels [13] to estimate how individual training examples effect a policy’s performance on a given task. We propose key modifications that make this estimation tractable and robust in robotics settings—reducing noise, avoiding expensive rollouts, and minimizing distribution shift. Given a prior dataset and target dataset $\mathcal { D } _ { t a r g e t }$ , we (1) Cluster the prior data into trajectories or sub-trajectories, (2) Estimate influence scores using our proposed proxy metric, with the regression or metagradient datamodel estimators, (3) Select the top ranked clusters creating $\mathcal { D } _ { s e l }$ , and (4) Train a final policy co-trained on the target and selected data.

## 5 Experiments

Datasets. We test DataMIL on two widely used multi-task simulation benchmarks: (1) MetaWorld [14], contains a suite of 50 distinct robot manipulation tasks, and (2) LIBERO benchmark [17], consists of 100 tasks with diverse objects, layouts and scenes. In the real world we test using the Open-X Embodiment (OXE) datasets [4] – an aggregation of diverse robotic datasets collected across various robots and labs around the world.

Training and Evaluation Details. We use the language-conditioned Octo [15] model in LIBERO and OXE settings, initializing the model with a pretrained checkpoint provided by the authors to speed up training. For MetaWorld, we use the environment state as policy input, and hence use a simpler MLP based policy with a Gaussian action head, and study both goal-conditioned and no-conditioning settings. Results for the latter can be found in App. C.2.

![](images/e23749387dc1919af529cced0f34152ca78721a21c6841da6983032e6b212f19.jpg)

Baselines. We compare DataMIL to the following prior works: BehaviorRetreival (BR) [12] trains a VAE on state-action pairs and uses similarity with the target data in the latent space to retrieve single state-action pairs; FlowRetrieval (Flow) [11] uses a similar approach but trains the VAE on the flow features of the images computed using GMFlow [27]; STRAP [10] uses features from a DinoV2 [48] model and uses dynamic time-warping to retrieve similar sub-trajectories. We also introduce a simple action retrieval (AR) heuristic that retrieves based on action se-

Figure 3: Performance of policy trained on selected datasets in sim. environments.

quence similarity. Finally, we train policies only on the target data (Target-Only), and co-trained with all data (All-Data) to measure the overall importance of data selection.

## 5.1 Results

How does data selected using DataMIL impact policy performance?

Metaworld. MetaWorld’s 50 manipulation tasks offer a rigorous testbed for data selection. We construct our prior multi-task dataset by combining (1) expert demonstrations generated by scripted policies and (2) lower-quality exploration trajectories sampled from the replay buffer of a multi-task SAC agent trained across all tasks (see App. F.1 for details). For each task, we use 5 expert demos as $\mathcal { D } _ { t a r g e t }$ , and use the regression-based datamodel estimator to select the top 10% of samples (Sec. 4).

![](images/ec1ad29d547b5071e9fbe1894a5bda9bf3c5ce818e69d02f434baf485a070041.jpg)  
Figure 4: Results on OXE. We evaluate policies trained on subsets of the Open X-Embodiment dataset selected by different strategies. DataMIL consistently achieves the highest success, highlighting the need for end-end policy-aware data selection techniques. (Droid-Multitask shows average success; per-task results in App. C.7.)

This setup is challenging as the selection method must both identify relevant tasks and filter noisy, suboptimal actions from the autonomous data. In Fig. 3a, we report policy performance averaged over all 50 tasks. Similarity-based baselines perform poorly: state-only (SR) fails to reject poor actions, action-only (AR) selects irrelevant tasks with similar action distributions, and state-action retrieval (BR) gives equal weight to both modalities, which may not be the appropriate recipe for all tasks (App. A.1 provides a qualitative analysis). In contrast, by directly estimating each sample’s influence on policy performance, DataMIL effectively identifies useful demonstrations and discards harmful samples.

## Can we scale DataMIL to larger and more complex policy classes?

LIBERO. We test DataMIL on 10 long-horizon tasks from the LIBERO-10 setting, using LIBERO-90 (comprising 4500 human-teleoperated demonstrations) as the prior dataset and selecting 10% of the data (App. C.3 compares other selection %). The complex tasks and the high-dimensional RGB observations in LIBERO demand a powerful policy; we use Octo [15], a transformer-based diffusion policy. Training Octo is costly, making the regression estimator (which requires retraining across many subsets) impractical, and so we employ the metagradient datamodel estimator (Sec. 4.1).

Fig. 3b compares success rates of policies trained on the data selected by DataMIL with the baselines in each of the 10 target tasks. The relatively clean structure of LIBERO—single-view, single embodiment—makes it favorable for baselines that select via visual similarity (eg. STRAP, BR and Flow). However, we observe that their effectiveness varies significantly across tasks, likely due to the task-dependent suitability of each heuristic (task-wise success rates are provided in the appendix Tab. 1). In contrast, DataMIL consistently performs well across all tasks, achieving the highest average performance overall.

## Can we select datafrom large heterogeneous datasets in the real world?

Open X-Embodiment Dataset (OXE). In the real world, we show data selection from the OXE [4] datasets and evaluate on four tasks on two robot embodiments shown in Figure 4 (top). This setting is particularly challenging: OXE is a heterogeneous aggregation of data from different labs, robots, camera setups, lighting conditions, and object arrangements. Further, none of our test tasks appear in OXE—we avoid matching the scene, camera pose, or objects. Instead, we aim to understand whether seemingly unrelated prior data can still yield positive transfer when curated appropriately. Our base setup uses 24 OXE datasets that were part of Octo’s original training. For the Franka-Ball and Franka-Pouch tasks, we subset this to 13 and 23 datasets, respectively (denoted as OXE-13 and OXE-23). Full details on the number of target demonstrations, dataset partitions, and evaluation methodology are provided in App. F.3.

![](images/8a68655a03b8c1319e3432903d599a51f9c58629ba11ee4963fe943082be8db9.jpg)  
(a) Data selected by DataMIL

![](images/e8863339c52020f948c447c4c5691fbfafb40d3316dd5f728ed510752fc7bf5d.jpg)  
(b) Data selected by AR

![](images/a775fb1ee966bb89cf7f69da9a8d72a62723ee32f32f9f283b36912cf00a8f81.jpg)  
(c) Data selected by BR

Figure 5: Distribution of datasets selected by different methods for Tiago-Sink task  
![](images/bd52771b63d232b1665f6ffd56d78d1aae948529c9fa37b6bc53e0c5d18c084a.jpg)  
(a) Data selected by DataMIL

![](images/f8b53d0c0d69ac77e1952a7e616ad6e6917d6be399d80affd7eba896b2af654e.jpg)  
(b) Data selected by Flow

![](images/4e6fbfb28973f97d80dd93af8da4fff656f61b47a1f41f97d2d60b54c017e4d8.jpg)  
(c) Data selected by BR  
Figure 6: Distribution of datasets selected by different methods for Franka-Pouch task

Due to the scale of OXE, we replace the All-Data baseline with a Random baseline that randomly samples the same number of datapoints as DataMIL and other methods. In Figure 4, we observe that DataMIL effectively selects relevant data even from highly heterogeneous sources. In the simpler Franka-Ball task, visual similarity-based baselines perform competitively. However, as the dataset grows more diverse—as in Franka-Pouch—these heuristics begin to break down, while DataMIL continues to identify data that improves policy performance. In the Tiago-Sink task, we explore a harder setting, selecting data for the Tiago [49] robot, an embodiment that never appears in the prior data. Despite this, DataMIL is able to select cross-embodiment demonstrations that improve task success. We discuss this further in Sec. 5.2. Finally, we move beyond single-task selection in the Droid-Multitask setting, where the target comprises three tasks: bread in bowl, napkin in drawer, and open drawer. This setting tests whether a single curated dataset can support multiple downstream objectives simultaneously. DataMIL consistently outperforms baselines, selecting data that improve its performance on all tasks, yielding a stronger average overall.

These results highlight that DataMIL scales to real-world robotics, handles heterogeneous datasets, and supports both single-task and multitask learning—even in settings with unseen embodiments.

## 5.2 What data is selected by DataMIL?

Here we discuss and qualitatively analyse the data selected by DataMIL from the OXE dataset. More visualizations are shown in App. A.3.

Type of embodiments selected. DataMIL is able to select useful data for a completely new embodiment in the Tiago-Sink experiment. In Fig. 5a we show the highest frequency datasets selected by DataMIL and observe that even though they are visually quite different (Fig. 10c), sampled from datasets such as RT-1 [50], BC-Z [51] and Bridge [52, 5], they still represent the essence of the target task – robots operating on a table top from an ego-perspective. For baselines, we observe that even when the target embodiment is present in the prior dataset (e.g., Franka), the selected data often comes from unrelated domains. For instance, in Franka-Pouch (Fig. 6), Flow selects data from BC-Z (Google Robot), and BR retrieves from Bridge (WidowX). This could be due to the baselines placing more weight on the scene/distractors when computing similarities. In contrast, the top five most frequently selected datasets by DataMIL are all sourced from Franka (Fig. 6a).

Distribution of selected data. In Figures 5 and 6, we show the distribution of datasets selected by DataMIL and representative baselines for the Tiago-Sink and Franka-Pouch tasks, respectively. We find that data selected by DataMIL usually balances several different datasets, whereas most baselines select a majority of their data from a single source. For example, AR retrieves most of its data from RT-1 in the Tiago-Sink task (Fig. 5b), while Flow disproportionately selects samples from BC-Z for Franka-Pouch (Fig. 6b). In contrast, DataMIL consistently selects data across a broader range of datasets in both of these cases (Fig. 5a and 6a). We hypothesize that since there is no data that exactly matches the target task, the selected data must not only be relevant but general, so as to enable positive transfer in capabilities and not make the policy overfit to a single type of domain.

Top and bottom samples. Analyzing the highest and lowest ranked datapoints by DataMIL in Fig. 10 we find that they typically look similar (e.g., same embodiment or dataset). This makes sense: similar states can have very different action distributions, and while some of these actions might help reduce the policy loss on the target data, the others might lead to a large deviation, making them harmful for final policy learning. This aligns with data attribution works in computer vision, where harmful data looks very similar to helpful data but with different labels [13, 53]. Understanding how and why these fine-grained differences affect data selection, and ultimately, the policy performance, is an interesting direction for future work.

## 6 Conclusion

We present DataMIL, a data-driven method for data selection for imitation learning. DataMIL builds upon the framework of datamodels, which has been applied successfully to data selection in NLP and CV, and extends it to real-world robotic applications. Extensive experiments in simulation and realworld settings empirically support that DataMIL retrieves data to train higher-performing policies than multiple existing state-of-the-art baselines, particularly in complex scenarios. At the same time, several limitations remain. First, despite employing an efficient metagradient-based estimator, estimating datamodels still incurs computational costs several times higher than training a policy on all data (see App. G), making scalability an important challenge. Accelerating datamodel training, for instance through smaller proxy models [54], is a promising direction. Second, DataMIL (and data selection techniques for robot learning more broadly) relies on a range of hyperparameters (e.g., selected dataset size, clustering sizes) for which we currently lack strong intuitions. We expect future iterations and broader adoption to make these choices more principled and user-friendly. Finally, while we demonstrated selection from large prior datasets, our target settings were primarily singletask. Although the Droid-Multitask setting partially addresses this, extending evaluation to truly large-scale, multi-task robotics settings remains an open and impactful avenue. Addressing these limitations will be crucial for scaling DataMIL to practical robotic applications.

## Acknowledgments

We thank Luca Macesanu for helping with some of the real world evaluations and Bowen Jiang for his feedback on the manuscript. We also thank Jullian Yapeter and Karl Pertsch for their assistance in generating the MetaWorld dataset. Work supported in part by DARPA TIAMAT program (HR0011- 24-9-0428) and also supported by the NSF grant DMS-2134108 and Open Philanthropy.

## References

[1] A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, X. Chen, K. Choromanski, T. Ding, D. Driess, A. Dubey, C. Finn, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. arXiv preprint arXiv:2307.15818, 2023.

[2] K. Black, N. Brown, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, L. Groom, K. Hausman, B. Ichter, et al. π<sub>0</sub>: A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024.

[3] J. Bjorck, F. Castañeda, N. Cherniadev, X. Da, R. Ding, L. Fan, Y. Fang, D. Fox, F. Hu, S. Huang, et al. Gr00t n1: An open foundation model for generalist humanoid robots. arXiv preprint arXiv:2503.14734, 2025.

[4] A. O’Neill, A. Rehman, A. Maddukuri, A. Gupta, A. Padalkar, A. Lee, A. Pooley, A. Gupta, A. Mandlekar, A. Jain, et al. Open x-embodiment: Robotic learning datasets and rt-x models: Open x-embodiment collaboration 0. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 6892–6903. IEEE, 2024.

[5] H. R. Walke, K. Black, T. Z. Zhao, Q. Vuong, C. Zheng, P. Hansen-Estruch, A. W. He, V. Myers, M. J. Kim, M. Du, et al. Bridgedata v2: A dataset for robot learning at scale. In Conference on Robot Learning, pages 1723–1736. PMLR, 2023.

[6] H.-S. Fang, H. Fang, Z. Tang, J. Liu, C. Wang, J. Wang, H. Zhu, and C. Lu. Rh20t: A comprehensive robotic dataset for learning diverse skills in one-shot. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 653–660. IEEE, 2024.

[7] M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. Foster, G. Lam, P. Sanketi, et al. Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024.

[8] M. J. Kim, C. Finn, and P. Liang. Fine-tuning vision-language-action models: Optimizing speed and success. arXiv preprint arXiv:2502.19645, 2025.

[9] L. Zha, Y. Cui, L.-H. Lin, M. Kwon, M. G. Arenas, A. Zeng, F. Xia, and D. Sadigh. Distilling and retrieving generalizable knowledge for robot manipulation via language corrections. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 15172– 15179. IEEE, 2024.

[10] M. Memmel, J. Berg, B. Chen, A. Gupta, and J. Francis. Strap: Robot sub-trajectory retrieval for augmented policy learning. arXiv preprint arXiv:2412.15182, 2024.

[11] L.-H. Lin, Y. Cui, A. Xie, T. Hua, and D. Sadigh. Flowretrieval: Flow-guided data retrieval for few-shot imitation learning. arXiv preprint arXiv:2408.16944, 2024.

[12] M. Du, S. Nair, D. Sadigh, and C. Finn. Behavior retrieval: Few-shot imitation learning by querying unlabeled datasets. arXiv preprint arXiv:2304.08742, 2023.

[13] A. Ilyas, S. M. Park, L. Engstrom, G. Leclerc, and A. Madry. Datamodels: Predicting predictions from training data. In Proceedings of the 39th International Conference on Machine Learning, 2022.

[14] T. Yu, D. Quillen, Z. He, R. Julian, K. Hausman, C. Finn, and S. Levine. Meta-world: A benchmark and evaluation for multi-task and meta reinforcement learning. In Conference on robot learning, pages 1094–1100. PMLR, 2020.

[15] Octo Model Team, D. Ghosh, H. Walke, K. Pertsch, K. Black, O. Mees, S. Dasari, J. Hejna, C. Xu, J. Luo, T. Kreiman, Y. Tan, L. Y. Chen, P. Sanketi, Q. Vuong, T. Xiao, D. Sadigh, C. Finn, and S. Levine. Octo: An open-source generalist robot policy. In Proceedings of Robotics: Science and Systems, Delft, Netherlands, 2024.

[16] L. Engstrom, A. Ilyas, B. Chen, A. Feldmann, W. Moses, and A. Madry. Optimizing ml training with metagradient descent. arXiv preprint arXiv:2503.13751, 2025.

[17] B. Liu, Y. Zhu, C. Gao, Y. Feng, Q. Liu, Y. Zhu, and P. Stone. Libero: Benchmarking knowledge transfer for lifelong robot learning. Advances in Neural Information Processing Systems, 36:44776–44791, 2023.

[18] A. Mandlekar, D. Xu, J. Wong, S. Nasiriany, C. Wang, R. Kulkarni, L. Fei-Fei, S. Savarese, Y. Zhu, and R. Martín-Martín. What matters in learning from offline human demonstrations for robot manipulation. arXiv preprint arXiv:2108.03298, 2021.

[19] A. Mandlekar, S. Nasiriany, B. Wen, I. Akinola, Y. Narang, L. Fan, Y. Zhu, and D. Fox. Mimicgen: A data generation system for scalable robot learning using human demonstrations. arXiv preprint arXiv:2310.17596, 2023.

[20] P. Sharma, L. Mohan, L. Pinto, and A. Gupta. Multiple interactions made easy (mime): Large scale demonstrations data for imitation. In Conference on robot learning, pages 906–915. PMLR, 2018.

[21] A. Khazatsky, K. Pertsch, S. Nair, A. Balakrishna, S. Dasari, S. Karamcheti, S. Nasiriany, M. K. Srirama, L. Y. Chen, K. Ellis, et al. Droid: A large-scale in-the-wild robot manipulation dataset. arXiv preprint arXiv:2403.12945, 2024.

[22] A. Mandlekar, Y. Zhu, A. Garg, J. Booher, M. Spero, A. Tung, J. Gao, J. Emmons, A. Gupta, E. Orbay, et al. Roboturk: A crowdsourcing platform for robotic skill learning through imitation. In Conference on Robot Learning, pages 879–893. PMLR, 2018.

[23] J. Hejna, C. Bhateja, Y. Jiang, K. Pertsch, and D. Sadigh. Re-mix: Optimizing data mixtures for large scale imitation learning. arXiv preprint arXiv:2408.14037, 2024.

[24] S. M. Xie, H. Pham, X. Dong, N. Du, H. Liu, Y. Lu, P. S. Liang, Q. V. Le, T. Ma, and A. W. Yu. Doremi: Optimizing data mixtures speeds up language model pretraining. Advances in Neural Information Processing Systems, 36:69798–69818, 2023.

[25] J. Hejna, S. Mirchandani, A. Balakrishna, A. Xie, A. Wahid, J. Tompson, P. Sanketi, D. Shah, C. Devin, and D. Sadigh. Robot data curation with mutual information estimators. arXiv preprint arXiv:2502.08623, 2025.

[26] A. S. Chen, A. M. Lessing, Y. Liu, and C. Finn. Curating demonstrations using online experience. arXiv preprint arXiv:2503.03707, 2025.

[27] H. Xu, J. Zhang, J. Cai, H. Rezatofighi, and D. Tao. Gmflow: Learning optical flow via global matching. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 8121–8130, 2022.

[28] S. Kumar, S. Dass, G. Pavlakos, and R. Martín-Martín. Collage: Adaptive fusion-based retrieval for augmented policy learning. In 9th Annual Conference on Robot Learning, 2025.

[29] C. Agia, R. Sinha, J. Yang, R. Antonova, M. Pavone, H. Nishimura, M. Itkina, and J. Bohg. Cupid: Curating data your robot loves with influence functions. arXiv preprint arXiv:2506.19121, 2025.

[30] P. W. Koh and P. Liang. Understanding black-box predictions via influence functions. In International conference on machine learning, pages 1885–1894. PMLR, 2017.

[31] A. Ilyas, K. Georgiev, L. Engstrom, and S. M. Park. Data attribution at scale. Tutorial at ICML 2024, 2024. URL https://ml-data-tutorial.org.

[32] Z. Hammoudeh and D. Lowd. Training data influence analysis and estimation: A survey. Machine Learning, 113(5):2351–2403, 2024.

[33] S. M. Park, K. Georgiev, A. Ilyas, G. Leclerc, and A. Madry. TRAK: Attributing model behavior at scale. In Arxiv preprint arXiv:2303.14186, 2023.

[34] T. A. Chang, D. Rajagopal, T. Bolukbasi, L. Dixon, and I. Tenney. Scalable influence and fact tracing for large language model pretraining. In The Thirteenth International Conference on Learning Representations, 2024.

[35] A. Ilyas and L. Engstrom. Magic: Near-optimal data attribution for deep learning, 2025.

[36] L. Engstrom, A. Feldmann, and A. Madry. Dsdm: Model-aware dataset selection with datamodels. In International Conference on Machine Learning, pages 12491–12526. PMLR, 2024.

[37] M. Xia, S. Malladi, S. Gururangan, S. Arora, and D. Chen. Less: selecting influential data for targeted instruction tuning. In Proceedings of the 41st International Conference on Machine Learning, pages 54104–54132, 2024.

[38] Z. Liu, A. Karbasi, and T. Rekatsinas. Tsds: Data selection for task-specific model finetuning. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024.

[39] S. Jain, K. Hamidieh, K. Georgiev, A. Ilyas, M. Ghassemi, and A. Madry. Improving subgroup robustness via data selection. Advances in Neural Information Processing Systems, 37:94490– 94511, 2024.

[40] J. Lin, A. Zhang, M. Lécuyer, J. Li, A. Panda, and S. Sen. Measuring the effect of training data on deep learning predictions via randomized experiments. In International Conference on Machine Learning, pages 13468–13504. PMLR, 2022.

[41] J. Bae, W. Lin, J. Lorraine, and R. Grosse. Training data attribution via approximate unrolled differentiation. CoRR, 2024.

[42] F. R. Hampel. The influence curve and its role in robust estimation. Journal of the american statistical association, 69(346):383–393, 1974.

[43] D. A. Calian, G. Farquhar, I. Kemaev, L. M. Zintgraf, M. Hessel, J. Shar, J. Oh, A. György, T. Schaul, J. Dean, et al. Datarater: Meta-learned dataset curation. arXiv preprint arXiv:2505.17895, 2025.

[44] S. Jain, H. Salman, A. Khaddaj, E. Wong, S. M. Park, and A. Madry. A data-based perspective on transfer learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3613–3622, 2023.

[45] D. Ley, S. Srinivas, S. Zhang, G. Rusak, and H. Lakkaraju. Generalized group data attribution. arXiv preprint arXiv:2410.09940, 2024.

[46] A. Maddukuri, Z. Jiang, L. Y. Chen, S. Nasiriany, Y. Xie, Y. Fang, W. Huang, Z. Wang, Z. Xu, N. Chernyadev, et al. Sim-and-real co-training: A simple recipe for vision-based robotic manipulation. arXiv preprint arXiv:2503.24361, 2025.

[47] S. Nasiriany, T. Gao, A. Mandlekar, and Y. Zhu. Learning and retrieval from prior data for skill-based imitation learning. arXiv preprint arXiv:2210.11435, 2022.

[48] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.

[49] J. Pages, L. Marchionni, and F. Ferro. Tiago: the modular robot that adapts to different research needs. In International workshop on robot modularity, IROS, volume 290, 2016.

[50] A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, J. Dabis, C. Finn, K. Gopalakrishnan, K. Hausman, A. Herzog, J. Hsu, et al. Rt-1: Robotics transformer for real-world control at scale. arXiv preprint arXiv:2212.06817, 2022.

[51] E. Jang, A. Irpan, M. Khansari, D. Kappler, F. Ebert, C. Lynch, S. Levine, and C. Finn. Bc-z: Zero-shot task generalization with robotic imitation learning. In Conference on Robot Learning, pages 991–1002. PMLR, 2022.

[52] F. Ebert, Y. Yang, K. Schmeckpeper, B. Bucher, G. Georgakis, K. Daniilidis, C. Finn, and S. Levine. Bridge data: Boosting generalization of robotic skills with cross-domain datasets. arXiv preprint arXiv:2109.13396, 2021.

[53] V. Feldman and C. Zhang. What neural networks memorize and why: Discovering the long tail via influence estimation. Advances in Neural Information Processing Systems, 33:2881–2891, 2020.

[54] A. Khaddaj, L. Engstrom, and A. Madry. Small-to-large generalization: Training data influences models consistently across scale. In The Thirteenth International Conference on Learning Representations, 2025.

[55] V. Saxena, M. Bronars, N. R. Arachchige, K. Wang, W. C. Shin, S. Nasiriany, A. Mandlekar, and D. Xu. What matters in learning from large-scale datasets for robot manipulation. In The Thirteenth International Conference on Learning Representations, 2025.

[56] T. garage contributors. Garage: A toolkit for reproducible reinforcement learning research. https://github.com/rlworkgroup/garage, 2019.

[57] S. Dass, W. Ai, Y. Jiang, S. Singh, J. Hu, R. Zhang, P. Stone, B. Abbatematteo, and R. Martín-Martín. Telemoma: A modular and versatile teleoperation system for mobile manipulation. arXiv preprint arXiv:2403.07869, 2024.

## A Qualitative Results: What data is selected by DataMIL?

(a) Composition of data selected via DataMIL  
![](images/9f46db16cbf4f12718ba51190c3a3e220acdceb1ac84c414c3241c231a5d478d.jpg)

(b) Composition of data selected via SR  
![](images/67da03ce4423278a49964a02fdd96982982aa426cfa77db5b58c7060e79017e8.jpg)

(c) Composition of data selected via AR  
![](images/ea96be305467dbd21a934c4aca5bc8f6e7f715991509342b5fbcea67286ef728.jpg)

(d) Composition of data selected via BR  
![](images/a7e3b1b5526b88354ed96800518294c1a105a60610650e0799fe818d8d749844.jpg)  
Figure 7: MetaWorld Qualitative Results. Percentage of data selected from each task and expert/suboptimal source for DataMIL, SR, AR and BR

## A.1 MetaWorld

As described in Appendix F.1, our prior dataset for MetaWorld combines both expert and suboptimal demonstrations. In our experiments, we retrieve the top 10% of this data ranked by DataMIL to train the policy. Here, we qualitatively examine the dataset selected by DataMIL for the pickplace-wall task.

Figure 7 shows the percentage of data selected from each task and data source (expert or suboptimal). We observe that SR, while able to retrieve samples from relevant tasks, fails to differentiate between expert and sub-optimal demonstrations—resulting in the inclusion of a large fraction of low-quality data. In contrast, AR filters out sub-optimal samples more effectively by matching actions, but it lacks task awareness due to its disregard for state information, often pulling data from irrelevant tasks. BR, which embeds both state and action features jointly, exhibits a blend of SR and AR behaviors—capturing elements of both but also inheriting their limitations. In comparison, DataMIL consistently selects data from the correct task (green bar) while also avoiding noisy, sub-optimal examples.

## A.2 LIBERO

In LIBERO, for two target tasks, Book-Caddy (Figure 8) and Bowl-Cabinet (Figure 9), we investigate the proportion of data that is selected from each of the prior tasks in LIBERO-90.

![](images/a0b7f3f692d3d23a03678ab6184beaf50ad6985356283b64f62214a207ae3b02.jpg)

![](images/d01288a194160c3092b9596db9852105265f83e50b1d95a100cbfe86657ea5a9.jpg)  
Figure 8: LIBERO Book-Caddy. Proportion of prior tasks selected for the book-caddy task

![](images/ead47bd131ab260f340e5184d1306c948ecf365cbcdbb23a13c8f28f3d63fd62.jpg)  
Figure 9: LIBERO Bowl-Cabinet. Proportion of prior tasks selected for the bowl-cabinet task

Since the exact target tasks do not appear in the prior dataset, we manually label (in orange) tasks with similar semantics and object layouts. We also visualize examples from the most and least frequently selected tasks. The plots support empirically that DataMIL consistently selects data from tasks with matching object and scene configuration. The visualizations also reflect this, showing that related tasks from similar looking scenes, albeit with related but different language instructions, are most frequently selected, aligning with intuition developed in prior works [55, 46].

## A.3 OXE

(a) Franka-Ball  
![](images/70b3380fa6204a301553f15db1fe4789ffd9fc39157667e0a8e42635fdec77e3.jpg)

(b) Franka-Pouch  
![](images/634a333e33a5c65274d806fa6c6c2f2ebeb1dda6c504547350cb75023a139367.jpg)

(c) Tiago-Sink  
![](images/0358fa3c13006a0c27bafa1b8fcb34f5c7ada54991920ba87f5193442dd7ceda.jpg)

(d) Droid-Multitask  
![](images/b189d21afef06e3c80526a56b373eb03454bbf1ad6bd1ce1180e3aa069fa62a5.jpg)  
Figure 10: Top and bottom ranked samples by DataMIL for each of the real-world tasks

Figure 10 visualizes the highest and lowest ranked datapoints identified by DataMIL for real-world tasks. As discussed in the main text, these examples often appear visually similar, aligning with findings in computer vision [13, 53] where data points that look alike but carry different labels can mislead the model. In robotics, this may occur when visually similar observations correspond to different actions, thereby confusing the policy. How and why these fine-grained differences affect data selection, and ultimately, the policy performance, is an interesting direction for future work.

## B Estimating Datamodels

In this section, we describe the datamodeling framework in more detail. In particular, we first provide the formal version of Informal Definition 1, then describe the estimators that we use to construct datamodels in this work, namely the regression estimator and the metagradient-based estimator.

## B.1 Formalizing Datamodeling

The goal of datamodeling is to construct a function $\hat { f }$ that can predict the performance of a learning algorithm on any given subset of the data $\mathcal { D } ^ { \prime } \subset \mathcal { D }$ without actually training a policy on that subset. Let be a prior dataset of imitation-learning data of size $N = | \mathcal { D } |$ , and let us represent any subset of $\mathcal { D }$ as a binary vector $\mathbf { w } \in \{ 0 , 1 \} ^ { N }$ where $w _ { i } = 1$ if the i-th training sample is in the subset and $w _ { i } = 0$ otherwise. Let a learning algorithm be a function that takes as input a dataset (represented as a binary vector w) and outputs a policy $\mathcal { A } ( \mathbf { w } )$ . The datamodeling problem is to construct a function $\hat { f }$ that can predict the performance of $\mathcal { A } ( \mathbf { w } )$ when trained on any given subset of the data without actually training a policy on that subset. More formally, we aim to find a function $\hat { f }$ minimizing the following loss:

$$
\mathbb {E} _ {\mathbf {w} \sim \text { Bernoulli } (\frac {1}{2}) ^ {N}} \left[ \left(\mathcal {M} (\mathcal {A} (\mathbf {w})) - \hat {f} (\mathbf {w})\right) ^ {2} \right],\tag{6}
$$

where  is the target metric and $\mathbf { w } \sim \mathbf { B e r n o u l l i } ( \frac { 1 } { 2 } ) ^ { N }$ is a random binary vector of length $N$ .

Recall from the main text that we are particularly interested in datamodels $\hat { f }$ that are additive in the training dataset—in terms of our formalization, we are interested in functions $\hat { f }$ of the form

$$
\hat {f} (\mathbf {w}) = \mathbf {w} ^ {\top} \beta .\tag{7}
$$

for some vector $\boldsymbol { \beta } \in \mathbb { R } ^ { N }$ . An estimation method for datamodels is thus just a method for finding a good estimate of the vector $\beta .$ . We refer the reader to Ilyas et al. [13] for a more detailed discussion.

## B.2 Regression Estimator

The regression estimator is a simple yet effective method for estimating the vector $\beta$ that treats datamodeling as a supervised learning problem. In particular, the regression estimator first samples a set of m binary vectors $\mathbf { w } _ { 1 } , \ldots , \mathbf { w } _ { m } \sim$ Bernoull $\bar { ( \boldsymbol { p } ) } ^ { N }$ where $p$ is the probability of inclusion; for each of these binary vectors, it trains a policy $\mathcal { A } ( \mathbf { w } _ { i } )$ on the subset of the data indexed by $\mathbf { w } _ { i } ,$ , and evaluates its performance using the target metric . It then fits a linear model to the performance of these policies on the sampled binary vectors, i.e., it solves

$$
\min _ {\beta \in \mathbb {R} ^ {N}} \frac {1}{m} \sum_ {i = 1} ^ {m} \left(\mathcal {M} (\mathcal {A} (\mathbf {w} _ {i})) - \mathbf {w} _ {i} ^ {\top} \beta\right) ^ {2},\tag{8}
$$

and uses the resulting vector $\hat { \beta }$ as the parameters of the datamodel $\hat { f } ( \mathbf { w } ) = \mathbf { w } ^ { \top } \hat { \boldsymbol { \beta } }$ . The cost of building this estimator is high, since it requires training a policy for each of the m binary vectors, but Ilyas et al. [13] shows that the estimator can be very accurate, and indeed identifies highly influential subsets of the prior dataset. Pseudocode for the regression estimator is provided in Algorithm 1.

## B.3 Metagradient-based Estimator

The metagradient-based estimator is a more sophisticated method for estimating the vector $\beta$ that avoids the high cost of training a policy for each binary vector. Instead, the metagradient-based estimator operates by leveraging a classical statistical tool called the influencefunction [42]. Intuitively, the metagradient-based estimator proceeds as follows. First, instead of thinking of training datasets as binary vectors $\mathbf { w } \in \{ 0 , 1 \} ^ { N }$ , we think of them as real-valued vectors $\mathbf { w } \in [ 0 , 1 ] ^ { N }$ , where each coordinate $w _ { i }$ corresponds to the importance weight placed on the i-th training sample in the dataset. Concretely, if $w _ { i } = 0$ , then the i-th training sample is not used in the training set, and if $w _ { i } = 1$ then the i-th training sample is used in the training set with full weight; if $0 < w _ { i } < 1$ , then the i-th training sample is used in the training set but its loss is scaled by $w _ { i }$ . Observe that this parameterization is equivalent to the binary parameterization for $w _ { i } \in \{ 0 , 1 \}$ , but gives us a continuous way to represent the training set.

Once we have this continuous parameterization, we can write the first-order approximation to $\mathcal { M } ( \mathcal { A } ( \mathbf { w } ) )$ as,

$$
\mathcal {M} (\mathcal {A} (\mathbf {w})) \approx \mathcal {M} (\mathcal {A} (\mathbf {w} _ {0})) + \nabla \mathcal {M} (\mathcal {A} (\mathbf {w} _ {0})) ^ {\top} (\mathbf {w} - \mathbf {w} _ {0}),\tag{9}
$$

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1 Regression Estimator
Input: Dataset D of size N, policy learner  $\mathcal{A}(\cdot)$ , target metric  $\mathcal{M}(\cdot)$ 
Hyperparameters: binary masks m, prob. p

▷ Generate training pairs for the surrogate
 $T \leftarrow []$ 

for  $i \leftarrow 1$  to m do

Sample  $w_i \sim \text{Bernoulli}(p)^N$ 

Train policy on subset:  $\pi_i \leftarrow \mathcal{A}(\mathbf{w}_i)$ 

Evaluate performance:  $y_i \leftarrow \mathcal{M}(\pi_i)$ 

Append pair:  $T \leftarrow T + [(w_i, y_i)]$ 

▷ Fit linear datamodel using Eq. 8

 $\hat{\beta} \leftarrow \arg\min_{\beta \in R^N} \frac{1}{m} \sum_{i=1}^{m} (y_i - \mathbf{w}_i^\top \boldsymbol{\beta})^2$ 

return  $\hat{\beta}$ 

Algorithm 2 Metagradient-based Estimator
Input: initial included data  $c = 1_n$ , policy learner  $\mathcal{A}(\cdot)$ , target metric  $\mathcal{M}(\cdot)$ 
Hyperparameters: step size p, number of steps T

▷ Gradient Storage
 $G \leftarrow []$ 

▷ Metagradient descent loop
for  $t \leftarrow 1$  to T do

 $w \leftarrow 0_n$ 

▷ Compute metagradients

 $g \leftarrow \frac{\partial\mathcal{M}(\mathcal{A}_{c}'(\mathbf{w}))}{\partial\mathbf{w}}$ 

Sample  $m \sim \text{Bernoulli}(p)^N$ 

Store g:  $G \leftarrow G + [g \odot m]$ 

▷ Update Counts

 $c \leftarrow c - \text{sign}(\mathbf{g}) \odot m$ 

return G
</div>

where $\mathbf { w } _ { 0 }$ is the vector of all ones. The gradient $\nabla \mathcal { M } ( \mathcal { A } ( \mathbf { w } _ { 0 } ) )$ is known as the influence function, and gives us a linear approximation to the loss in Equation 6. That is, if we could compute the influence function exactly, we could use it as a datamodel directly, i.e.,

$$
\hat {f} (\mathbf {w}) = \mathcal {M} (\mathcal {A} (\mathbf {w} _ {0})) + \nabla \mathcal {M} (\mathcal {A} (\mathbf {w} _ {0})) ^ {\top} (\mathbf {w} - \mathbf {w} _ {0}).\tag{10}
$$

Traditionally, the influence function is notoriously hard to compute, and so prior work on data attribution has focused on approximating it [30, 33, 41]. However, recent work has shown how to compute it exactly and efficiently [16] and how to use them as a datamodel estimator [35].

Observe that in order for the metagradient-based estimator to be valid, the function $\mathcal { M } ( \mathcal { A } ( \mathbf { w } ) )$ must be differentiable with respect to w. For this to be satisfied, it is sufficient for (a) the target metric $\mathcal { M }$ to be differentiable with respect to the policy $\mathcal { A } ( \mathbf { w } )$ , and (b) the policy $\mathcal { A } ( \mathbf { w } )$ to be trained via an iterative algorithm composed of elementary differentiable operations (which is almost all of the popular off-the-shelf learning algorithms).

Gradient descent on training data. To operationalize the metagradient-based estimator, we use the Metagradient Descent (MGD) algorithm [16], to compute influence scores and iteratively refine the dataset. The algorithm views data selection as an optimization problem over the selected data counts $\mathbf { c } \in \{ 0 , 1 \} ^ { N }$ , where $c _ { i } = 1$ indicates inclusion of the i-th sample.

Given a count vector $\mathbf { c } ,$ let $A _ { \mathbf { c } }$ denote the policy obtained by training on the dataset defined by $\mathbf { c } ,$ and let $\mathcal { M } ( \mathcal { A } _ { \bf c } )$ denote its evaluation on the target metric. MGD introduces a differentiable surrogate $\mathcal { A } _ { \mathbf { c } } ^ { \prime } ( \mathbf { w } )$ , where $\mathbf { w } \in \mathbb { R } ^ { N }$ perturbs the per-sample weights during training only at a certain iteration k, while keeping the rest of the training as is. By construction, setting ${ \bf w } = { \bf 0 }$ recovers the original training procedure, i.e., $\mathcal { A } _ { \mathbf { c } } ^ { \prime } ( \mathbf { 0 } ) = \mathcal { A } _ { \mathbf { c } }$ . We can then compute the influence/metagradient as,

$$
\mathbf {g} = \nabla_ {\mathbf {w}} \left. \mathcal {M} (\mathcal {A} _ {\mathbf {c}} ^ {\prime} (\mathbf {w})) \right| _ {\mathbf {w} = \mathbf {0}},\tag{11}
$$

which measures how an infinitesimal upweighting of each training example at the kth iteration would affect the downstream performance. The sign of $\mathbf { g } _ { i }$ thus reveals whether increasing or decreasing the weight of sample i would improve $\mathcal { M }$

To update the dataset, MGD applies stochastic coordinate updates: a random mask m $\sim$ Bernoull $( p ) ^ { N }$ selects a subset of coordinates to update, and the counts are modified as

$$
\mathbf {c} \leftarrow \mathbf {c} - \operatorname{sign} (\mathbf {g}) \odot \mathbf {m},\tag{12}
$$

where $\odot$ denotes elementwise multiplication. This step is analogous to performing gradient descent over training samples with $p$ as the learning rate. Repeating this process iteratively over $T$ steps refines c, yielding a dataset concentrated on the most influential examples. In our experiments, we found that averaging the gradients g computed over the T iterations lead to more reliable influence scores. A pseudocode of the algorithm is provided in Algorithm 2 and we refer the reader to Engstrom et al. [16] for more details.

![](images/7b1259b97bc9c73925ca5cab55576447906b0ea86805da6ab23c7e0f9d32b71a.jpg)  
Figure 11: Extended Fig. 2 Results. Comparing true rollout success ( ) vs. proxy metric $\widehat { ( \mathcal { M } ) }$

## C Comparisons and Additional Results

Here we provide some additional results and show a comparison of some key hyperparameters such as percentage of data selected, cluster size and more.

## C.1 Comparison: True Rollout success vs. Proxy Metric

Here we provide extended results of our analysis in Figure 2. Recall from before, that we are comparing the data selected using datamodels computed via our proposed proxy metric in Equation 5 (DataMIL-rg) vs. true policy rollouts (DM-rollouts). We make this comparison in 4 tasks from MetaWorld as shown in Figure 11, where we select the top 10% of samples (as ranked by their estimated datamodel coefficients) from a multi-task prior dataset consisting of a mix of expert and suboptimal demonstrations (dataset details provided in Section F.1) and measure the success rates of policies trained on the selected data. Across all tasks we observe a similar trend as discussed in Section 4, (a) our proxy metric only incurs a small drop in performance across all tasks, thus demonstrating its efficacy as a viable metric for data selection and (b) data curation is critical: naively using all data or only target examples yields suboptimal policies, whereas curated datasets substantially boost task success.

## C.2 MetaWorld No-Goal Conditioning Experiments

In Figure 3 of the main paper, we presented results on MetaWorld with goal conditioning, where policies receive explicit goal information provided by the simulator. Goal-conditioning is often essential in settings like LIBERO and OXE, where the target task has limited demonstrations and generalization from other tasks is required. However, in MetaWorld, the prior dataset already includes expert demonstrations for the target tasks. Therefore, an effective data selection method should be capable of retrieving relevant examples—even without goal information.

![](images/36d309a329ce61b20bb2e8bec910b907bf741e17ca960548f020d2a231fdbe02.jpg)  
Figure 12: Avg. success on MetaWorld without goal conditioning.

To test this, we repeat the MetaWorld experiments described in $\mathsf { A p - }$ pendix F.1, but mask out goal states during both data selection and policy

training. Results averaged over all 50 tasks are shown in Figure 12. Even without goal-conditioning, DataMIL continues to outperform the best baseline by 10% in average success rate. However, overall performance across all methods declines compared to the goal-conditioned setting (Figure 3a), highlighting that, without goal information, selected data from other tasks can introduce harmful interference during training.

![](images/0a4339ae25d83f9f6ed020d5b53838ccf20080b25f09fd79017fdfd4c5b2a7b6.jpg)  
(a) ablating the selected data %

![](images/bbad3f0bbc1b6e24bce4d2d955266c52003e686bf09f6e6066dc96ac5d6114b3.jpg)  
(b) ablating the cluster size

![](images/d0fab3731d4f904de28e49d62d62a46b40ae2e3d197775673124f4b5c4695545.jpg)  
(c) ablating the co-training ratio  
Figure 13: Ablation experiments of DataMIL over different key data selection and training hyperparameters. Each experiment is averaged over 5 seeds on each of the tasks in LIBERO-5. In (a) and (c) each data point represents an averaged success rate over all 5 tasks.

## C.3 Comparison: Percentage of Data Selected

A key hyperparameter in data selection is the proportion of prior data to select. Unfortunately, there is no principled way to determine this value other than training and evaluating policies across different selection ratios. To enable systematic ablations, we construct a smaller benchmark within the LIBERO suite, denoted LIBERO-5, consisting of five tasks: Book-Caddy, Bowl-Cabinet, Cream-Butter, Mug-Microwave, and Soup-Cheese.

Figure 13a reports the average performance of DataMIL and two baselines, BR [12] and Flow [11], on LIBERO-5 while varying the percentage of prior data selected. At low selection ratios (5% and 10%), DataMIL substantially outperforms the baselines, highlighting its ability to prioritize higherquality samples compared to heuristic-based methods. At higher selection ratios ( 20%), the performance gap narrows and all methods achieve comparable results, as the majority of useful data is eventually included, diminishing the impact of the selection strategy.

## C.4 Comparison: Cluster Size

As described in Section 4.2, we group individual state–action pairs into temporal clusters before estimating their influence on policy performance via datamodels. We evaluate the effect of cluster granularity on LIBERO-5, with results shown in Figure 13b. Specifically, we compare three strategies: Cluster-15 (sub-trajectories of length 15), Cluster-30 (sub-trajectories of length 30), and Trajectory (entire trajectories as clusters). We observe that finer-grained clustering yields higher performance, as it provides more precise control over which segments of data are selected. Accordingly, our main experiments adopt a cluster size of 15.

## C.5 Comparison: Co-Training Ratio

While the data selected by DataMIL can be leveraged in multiple ways, we adopt a co-training strategy that combines target data $\mathcal { D } _ { t a r g e t }$ and selected data $\mathcal { D } _ { s e l }$ . At each training step, a batch is drawn from $\mathcal { D } _ { t a r g e t }$ with probability α and from $\mathcal { D } _ { s e l }$ with probability $1 - \alpha .$ . Although this choice is not intrinsic to DataMIL, it influences downstream performance. Figure 13c shows the average LIBERO-5 results of DataMIL and the All-Data baseline across different co-training ratios α. We find that $\alpha = 0 . 5$ consistently yields the best results. Given the similar setup between our simulation and real-world experiments, and the high cost of real-world evaluations, we adopt $\alpha = 0 . 5$ for real-robot experiments as well.

## C.6 Comparison: Longer Training for All-Data Baseline

Since the All-Data baseline leverages substantially more training data than other methods, we also investigate whether it benefits from extended training. On LIBERO-5, we compare training durations of 1 , 2 , and 4 the number of steps used for other methods. The resulting average performances are 43.36, 44.13, and 38.40, respectively. These results indicate no consistent benefit from longer training; in fact, excessive training can degrade performance due to overfitting. For fairness, we therefore use the same number of training steps across DataMIL and all baselines.

<table><tr><td>Tasks</td><td>Target-Only</td><td>All-Data</td><td>AR</td><td>BR</td><td>Flow</td><td>STRAP</td><td>DataMIL (ours)</td></tr><tr><td colspan="8">LIBERO Evaluations</td></tr><tr><td>Soup-Sauce</td><td> $13.2 \pm 7.6$ </td><td> $20.0 \pm 15.6$ </td><td> $32.8 \pm 9.3$ </td><td> $38 \pm 15.3$ </td><td> $50 \pm 14.4$ </td><td> $33.2 \pm 15.7$ </td><td> $39.2 \pm 11.9$ </td></tr><tr><td>Cream-Butter</td><td> $27.6 \pm 6.5$ </td><td> $35.2 \pm 13.2$ </td><td> $39.6 \pm 10.7$ </td><td> $41.2 \pm 14.6$ </td><td> $47.2 \pm 10.8$ </td><td> $20.0 \pm 9.7$ </td><td> $50.4 \pm 8.6$ </td></tr><tr><td>Stove-Moka</td><td> $27.4 \pm 7.3$ </td><td> $24.0 \pm 9.8$ </td><td> $31.2 \pm 8.1$ </td><td> $30.4 \pm 10.0$ </td><td> $33.2 \pm 6.7$ </td><td> $43.6 \pm 5.5$ </td><td> $40.4 \pm 8.5$ </td></tr><tr><td>Bowl-Cabinet</td><td> $48.8 \pm 6.4$ </td><td> $65.6 \pm 8.3$ </td><td> $62.8 \pm 3.3$ </td><td> $73.6 \pm 8.5$ </td><td> $69.2 \pm 12.2$ </td><td> $77.2 \pm 10.3$ </td><td> $72.4 \pm 5.2$ </td></tr><tr><td>Mug-Mug</td><td> $0.4 \pm 0.9$ </td><td> $2.0 \pm 2.44$ </td><td> $4.8 \pm 2.3$ </td><td> $2.0 \pm 2.8$ </td><td> $6.4 \pm 6.5$ </td><td> $5.2 \pm 5.2$ </td><td> $0.8 \pm 1.1$ </td></tr><tr><td>Book-Caddy</td><td> $51.2 \pm 8.3$ </td><td> $58.4 \pm 10.7$ </td><td> $65.2 \pm 13.1$ </td><td> $76.8 \pm 7.6$ </td><td> $69.2 \pm 13.2$ </td><td> $83.2 \pm 7.8$ </td><td> $82.4 \pm 1.7$ </td></tr><tr><td>Mug-Pudding</td><td> $2.0 \pm 2.0$ </td><td> $2.4 \pm 1.7$ </td><td> $5.6 \pm 5.9$ </td><td> $8.4 \pm 2.2$ </td><td> $5.8 \pm 3.5$ </td><td> $10.8 \pm 6.9$ </td><td> $4.8 \pm 3.3$ </td></tr><tr><td>Soup-Cheese</td><td> $11.2 \pm 3.3$ </td><td> $22.0 \pm 6.5$ </td><td> $29.3 \pm 10.4$ </td><td> $35.2 \pm 5.2$ </td><td> $26.8 \pm 4.8$ </td><td> $35.2 \pm 3.9$ </td><td> $36.0 \pm 6.0$ </td></tr><tr><td>Moka-Moka</td><td> $5.6 \pm 4.3$ </td><td> $12 \pm 8.2$ </td><td> $4.8 \pm 2.3$ </td><td> $15.2 \pm 4.1$ </td><td> $7.6 \pm 5.5$ </td><td> $6.8 \pm 1.8$ </td><td> $12.0 \pm 5.5$ </td></tr><tr><td>Mug-Microwave</td><td> $27.6 \pm 13.5$ </td><td> $35.6 \pm 10.4$ </td><td> $29.2 \pm 10.6$ </td><td> $43.2 \pm 9.9$ </td><td> $38.8 \pm 5.2$ </td><td> $34.4 \pm 8.3$ </td><td> $39.2 \pm 12.7$ </td></tr></table>

<table><tr><td>Libero-Average</td><td>21.5</td><td>27.72</td><td>30.52</td><td>36.4</td><td>35.42</td><td>34.96</td><td>37.76</td></tr><tr><td colspan="8">OXE Evaluations</td></tr><tr><td>Franka-Ball</td><td>21.4</td><td>35.7</td><td>28.6</td><td>50.0</td><td>28.6</td><td>57.1</td><td>50.0</td></tr><tr><td>Franka-Pouch</td><td>17.6</td><td>17.6</td><td>11.8</td><td>41.2</td><td>35.3</td><td>29.4</td><td>70.6</td></tr><tr><td>Tiago-Sink</td><td>33.3</td><td>43.6</td><td>35.9</td><td>46.2</td><td>46.2</td><td>38.5</td><td>64.1</td></tr><tr><td>Droid (Drawer)</td><td>0.0</td><td>0.0</td><td>0.0</td><td>20.0</td><td>70.0</td><td>55.0</td><td>75.0</td></tr><tr><td>Droid (Bread)</td><td>4.2</td><td>33.3</td><td>20.8</td><td>0.0</td><td>41.7</td><td>16.7</td><td>41.7</td></tr><tr><td>Droid (Napkin)</td><td>25.0</td><td>45.0</td><td>40.0</td><td>45.0</td><td>40.0</td><td>35.0</td><td>65.0</td></tr><tr><td>Droid-Multitask</td><td>9.4</td><td>26.6</td><td>20.3</td><td>20.3</td><td>50.0</td><td>34.4</td><td>59.4</td></tr><tr><td>Real-Average</td><td>20.4</td><td>30.9</td><td>24.1</td><td>39.4</td><td>40.0</td><td>39.8</td><td>61.0</td></tr></table>

Table 1: Numerical results for LIBERO and OXE evaluations

## C.7 Numerical Results and Discussion

Task-wise numerical success rates for the LIBERO and real-world settings are reported in Table 1. Our results on LIBERO differ from those reported in the original STRAP paper on a similar setting, and we attribute these differences to two key factors:

1. Evaluation Protocol. We suspect that the main contributing factor is likely a difference in the evaluation protocol. In the original STRAP implementation, the authors evaluate multiple training checkpoints and report results from the best-performing model. This can lead to higher reported success rates. In contrast, our evaluation protocol follows a stricter setup: we evaluate the policy only once, at the final checkpoint after training completes, without any checkpoint selection.

2. Policy Architecture. While we use the original STRAP data retrieval code, we differ in the policy architecture used for imitation learning. Specifically, we train with Octo [15], a large transformer-based diffusion policy, whereas STRAP uses a transformer-based policy from Robomimic [18] with a Gaussian mixture head. These two models have different inductive biases, which can lead to variation in performance across tasks. For example, when trained solely on the five target demonstrations from the moka-moka task, Octo achieves a 6% success rate, while Robomimic achieves 0%. However, in the mug-mug task, the Robomimic policy reaches 38% success, while Octo performs close to 0%.

We believe that both protocol differences and model architecture contribute to the gap in reported numbers, and since our goal is to study data selection, our results reflect a fair and consistent evaluation under a unified training and assessment setup across all baselines.

## D Proxy Metric Details

Recall from Eq. 5 that our general proxy metric is

$$
\widehat {\mathcal {M}} (\pi , \mathcal {D} _ {t a r g e t}) = \frac {1}{| \mathcal {D} _ {t a r g e t} |} \sum_ {(s, a) \in \mathcal {D} _ {t a r g e t}} - \mathcal {L} _ {B C} (\pi (s), a)
$$

where $\mathcal { L } _ { B C }$ is the behavior-cloning loss appropriate to the policy class. Here we provide how the equation looks like for specific policy classes that we used in our experiments.

MetaWorld. We parameterize the policy in MetaWorld as,

$$
\pi_ {\theta} (a \mid s) = \mathcal {N} \bigl (a; \mu_ {\theta_ {1}} (s), \mathrm{diag} (\sigma_ {\theta_ {2}} (s) ^ {2}) \bigr)
$$

and consider two forms of $\mathcal { L } _ { B C } \mathrm { : }$

1. Negative Log-Likelihood (NLL):

$$
\begin{array}{c} \mathcal {L} _ {\mathrm{NLL}} (s, a) = - \log \pi_ {\theta} (a \mid s) \\ = \frac {1}{2} \left(a - \mu_ {\theta_ {1}} (s)\right) ^ {\top} \Sigma (s) ^ {- 1} \left(a - \mu_ {\theta_ {1}} (s)\right) + \frac {1}{2} \log \det \bigl (2 \pi   \Sigma (s) \bigr) \end{array}\tag{13}
$$

2. $l _ { 1 }$ loss:

$$
\mathcal {L} _ {\ell_ {1}} (s, a) = \left\| a - \mu_ {\theta_ {1}} (s) \right\| _ {1}\tag{14}
$$

Empirically, the l1 loss works better for the regression estimator, while metagradient-based estimator performs well with the NLL loss.

LIBERO and OXE. In the LIBERO and OXE settings, we use Octo as the learning model which is a transformer-based policy with a diffusion action head. It’s behavior-cloning loss is the standard denoising score-matching objective:

$$
\mathcal {L} _ {\mathrm{diff}} (s, a) = \mathbb {E} _ {t \sim \mathrm{Uniform} [ 1, T ], \epsilon \sim \mathcal {N} (0, I)} \left\| \epsilon - \epsilon_ {\theta} \big (\sqrt {\bar {\alpha} _ {t}}   a + \sqrt {1 - \bar {\alpha} _ {t}}   \epsilon , s, t \big) \right\| ^ {2}
$$

where $\alpha _ { t } \in \mathsf { \Gamma } ( 0 , 1 )$ is the forward-process noise schedule, $\begin{array} { r } { \bar { \alpha } _ { t } \ = \ \prod _ { i = 1 } ^ { t } \alpha _ { i } } \end{array}$ , and $\epsilon _ { \theta } ( a _ { t } , s , t )$ is the network’s noise prediction.

Substituting $\mathcal { L } _ { B C } = \mathcal { L } _ { \mathrm { d i f f } }$ into the proxy metric gives,

$$
\widehat {\mathcal {M}} (\pi , \mathcal {D} _ {\mathrm{target}}) = - \frac {1}{| \mathcal {D} _ {\mathrm{target}} |} \sum_ {(s, a) \in \mathcal {D} _ {\mathrm{target}}} \mathbb {E} _ {t, \epsilon} \Big \| \epsilon - \epsilon_ {\theta} \big (\sqrt {\bar {\alpha} _ {t}} a + \sqrt {1 - \bar {\alpha} _ {t}} \epsilon , s, t \big) \Big \| ^ {2}\tag{15}
$$

## D.1 Up-weighting Relevant States

In robotics, more often than not we have some information about what states and actions are of higher importance than others, for example states closer to object interactions may be more relevant than moving around in free space. Our proxy metric provides a seamless way to incorporate prior knowledge by re-weighting important states in the behavioral cloning loss. Concretely, we introduce a state-action-dependent weight $w ( s , a )$ into the objective:

$$
\widehat {\mathcal {M}} (\pi , \mathcal {D} _ {t a r g e t}) = \frac {1}{| \mathcal {D} _ {t a r g e t} |} \sum_ {(s, a) \in \mathcal {D} _ {t a r g e t}} - \underline {{w (s , a)}} \mathcal {L} _ {B C} (\pi (s), a)
$$

By default, we set $w ( s , a ) = 1$ and use the unweighted proxy. In the LIBERO experiments, however, we found that doubling the weight for states immediately preceding a grasp significantly improves data selection. Thus, for those “pre-grasp” states we use $w ( s , a ) = 2$ , while all other states retain $w ( s , a ) = 1$

## E Baseline Implementation

Implementation details of the baselines are provided below.

FlowRetrieval [11] and BehaviorRetrieval [12]: FlowRetrieval (Flow) and Behavior Retrieval (BR) baselines compute similarity on the image flows and state-action pairs respectively. Since these features typically include high-dimensional image observations, Flow and BR train VAEs to encode the features into a more manageable latent space, which they can use to compute similarities between prior and target data. We used the implementation provided by the authors of FlowRetrieval [11] for training the VAEs and computing the similarity for both, Flow and BR, in the LIBERO and OXE settings (https://github.com/lihenglin/bridge\_training\_code). For Flow and BR (and other heuristics such as Action Retrieval (AR) and State Retrieval (SR)), once each state in the prior data is assigned a similarity score, we select the top x% of the data most similar to the target where x is the selection budget.

STRAP [10]: In the LIBERO experiments, we use the STRAP’s official implementation ( https: //github.com/WEIRDLabUW/STRAP ) to embed sub-trajectories with DinoV2 [48] and compute similarity via dynamic time warping. STRAP expects HDF5-formatted inputs, but our OXE pipeline relies on TFDS. We therefore adapted the STRAP code to accept TFDS datasets without altering its core logic. While STRAP’s original recommendation is to retrieve the top 100 sub-trajectories in LIBERO; we found that training Octo on these segments underperforms. Instead, we retrieve the most similar sub-trajectories until they constitute 10% of the prior data—matching the budget used by DataMIL and our other baselines. In LIBERO, this modification boosts success from 24.72% (with 100 segments) to 34.96% (our reported results) averaged over LIBERO-10 tasks. We apply the same retrieval strategy on OXE, sampling sub-trajectories until we match the selection size of our method and baselines.

For MetaWorld, we found it more effective to compute similarity over temporal windows rather than individual state–action pairs. Specifically, for each baseline (BR, SR, AR), we slide a fixedlength horizon  over both prior and target data, concatenate each segment’s states (and actions) into a single high-dimensional vector, and then measure similarity between these flattened vectors. This horizon-based approach captures temporal context, enabling the baselines to reject noisy or suboptimal samples—ultimately improving retrieval quality and downstream policy performance. Our initial experiments found = 50 to perform best.

## F Training and Evaluation Details

## F.1 MetaWorld

Dataset. MetaWorld dataset is constructed from two sources: scripted expert policies and reinforcement learning (RL) exploration. MetaWorld provides scripted policies for all of its 50 tasks, which we use to generate 4,000 episodes totaling 350K environment steps. For the RL data, we train a multi-task SAC agent on all 50 tasks for 12 million transitions, reaching an average success rate of 21%. To create a representative prior dataset, we uniformly subsample from the SAC replay buffer across all tasks, yielding 1 million environment steps—approximately 3× larger than the scripted data. For each target task, we generate 5 expert demonstrations using the scripted policy as $\mathcal { D } _ { t a r g e t }$

Datamodel estimation using DataMIL. We cluster the prior dataset at the trajectory level and use 5 demonstrations from $\mathcal { D } _ { t a r g e t }$ to compute the proxy objective. We then compute the datamodels using the regression-based datamodel estimator. The top 10% of prior trajectories, ranked by the datamodels, are selected to form $\mathcal { D } _ { s e l }$

Policy Training and Evaluation. We train a behavior cloning policy with an MLP backbone and a tanh-squashed Gaussian output distribution from garage [56] on $\mathcal { D } _ { s e l }$ . In preliminary experiments, we found that co-training with $\mathcal { D } _ { t a r g e t }$ yielded negligible improvements, so we exclude it in this setting. Each policy is trained and evaluated over 3 random seeds.

Table 2: Datasets utilized across the OXE subsets in our experimental setup

<table><tr><td>Dataset</td><td>OXE13</td><td>OXE23</td><td>OXE24</td><td>Dataset</td><td>OXE13</td><td>OXE23</td><td>OXE24</td></tr><tr><td>RT1</td><td>√</td><td>√</td><td>√</td><td>CMU IAM Lab</td><td>√</td><td>√</td><td>√</td></tr><tr><td>Viola</td><td>√</td><td>√</td><td>√</td><td>Roboturk</td><td>✕</td><td>√</td><td>√</td></tr><tr><td>Austin Buds</td><td>√</td><td>√</td><td>√</td><td>BC-Z</td><td>✕</td><td>√</td><td>√</td></tr><tr><td>Austin Mutex</td><td>√</td><td>√</td><td>√</td><td>CMU Stretch</td><td>✕</td><td>√</td><td>√</td></tr><tr><td>Austin Sailor</td><td>√</td><td>√</td><td>√</td><td>DLR Edan</td><td>✕</td><td>√</td><td>√</td></tr><tr><td>Austin Sirius</td><td>√</td><td>√</td><td>√</td><td>Berkeley Autolab UR5</td><td>✕</td><td>√</td><td>√</td></tr><tr><td>Taco Play</td><td>√</td><td>√</td><td>√</td><td>Berkeley Fanuc</td><td>✕</td><td>√</td><td>√</td></tr><tr><td>Jaco Play</td><td>√</td><td>√</td><td>√</td><td>Berkeley Cable</td><td>✕</td><td>√</td><td>√</td></tr><tr><td>Stanford Hydra</td><td>√</td><td>√</td><td>√</td><td>Bridge</td><td>✕</td><td>√</td><td>√</td></tr><tr><td>NYU Franka</td><td>√</td><td>√</td><td>√</td><td>NYU Door Opening</td><td>✕</td><td>√</td><td>√</td></tr><tr><td>Furniture Bench</td><td>√</td><td>√</td><td>√</td><td>Toto</td><td>✕</td><td>√</td><td>√</td></tr><tr><td>UCSD Kitchen</td><td>√</td><td>√</td><td>√</td><td>Kuka</td><td>✕</td><td>✕</td><td>√</td></tr></table>

## F.2 LIBERO

Dataset. Our prior dataset consists of 4,500 human teleoperated demonstrations from LIBERO-90, with 50 demonstrations per task. The 10 tasks from LIBERO-10 serve as our target tasks. For each, we randomly sample 5 demonstrations to form the target dataset $\mathcal { D } _ { t a r g e t }$

Datamodel estimation using DataMIL. Prior to running DataMIL, we segment the prior demonstrations into sub-trajectories of horizon length 15, which we found to provide a good balance between granularity and noise-robustness. We then estimate influence scores using the metagradientbased estimator with the weighted proxy metric described in Appendix D.1. The top 10% of subtrajectories, based on datamodel influence, are selected to form the selected dataset $\mathcal { D } _ { s e l }$

Policy Training and Evaluation. We fine-tune a language-conditioned Octo policy, starting from the publicly released Octo-small checkpoint, by co-training on $\mathcal { D } _ { t a r g e t }$ and $\mathcal { D } _ { s e l }$ using a co-training ratio α = 0.5 for 10k steps. We evaluate each policy on the corresponding target task using 50 rollouts and report results averaged over 5 random seeds.

## F.3 OXE

Table 3: Task-wise experimental setup for selecting data and, training and evaluating policies

<table><tr><td></td><td>Embodiment</td><td>Prior Dataset</td><td>Prior Selection Ratio</td><td>No. of Target Demos</td><td>No. of Evaluations</td></tr><tr><td>Franka-Pick</td><td>Franka-Panda</td><td>OXE13</td><td>1%</td><td>10</td><td>14</td></tr><tr><td>Franka-Pouch</td><td>Franka-Panda</td><td>OXE23</td><td>0.75%</td><td>30</td><td>17</td></tr><tr><td>Tiago-Sink</td><td>Tiago</td><td>OXE24</td><td>0.5%</td><td>20</td><td>39</td></tr><tr><td>Droid-Multitask</td><td>Franka-Panda</td><td>OXE24</td><td>1%</td><td>40 (total)</td><td>32 (total)</td></tr><tr><td>Drawer</td><td>-</td><td>-</td><td>-</td><td>10</td><td>10</td></tr><tr><td>Bread</td><td>-</td><td>-</td><td>-</td><td>15</td><td>12</td></tr><tr><td>Napkin</td><td>-</td><td>-</td><td>-</td><td>15</td><td>10</td></tr></table>

Dataset. We use subsets of the Open X-Embodiment (OXE) dataset [4] as our prior data. Specifically, we define three subsets—OXE13, OXE23, and OXE24—with their respective constituent datasets listed in Table 2. The mapping between tasks and dataset subsets is shown in Table 3. For each task, we collect a separate target dataset via teleoperation [57], varying the number of demonstrations per task based on difficulty (see Table 3).

Datamodel estimation using DataMIL. Following DataMIL’s recipe, we cluster prior data at the trajectory level and estimate influence using the metagradient-based estimator with our proposed proxy objective. To reduce distribution shift during datamodel training, we split $\mathcal { D } _ { t a r g e t }$ into two halves: one half is used to compute the proxy metric, while the other is included in the training mix. After training, we select top x% of trajectories based on influence scores (x given in Table 3).

Policy Training and Evaluation. We fine-tune a language-conditioned Octo-small checkpoint using co-training on $\mathcal { D } _ { t a r g e t }$ and the selected dataset $\mathcal { D } _ { s e l }$ , with a co-training ratio $\alpha = 0 . 5$ for 50k steps. Final policy is evaluated on the corresponding target tasks using a fixed number of real-world rollouts (Table 3). To ensure fair comparisons, we fix the spatial configurations of relevant objects across all methods. For instance, in the Franka-Pouch task, we use 17 predefined object poses (position and orientation) for evaluation. In the more challenging Droid-Multitask setting, we report both full and partial successes as part of the final success rate (e.g., a partially closed drawer or grasping the bread/napkin), with partial completions weighted as 0.5. All real-world evaluations were conducted using a single seed.

## G Computation Costs for Data Selection

Regression estimator. The regression estimator requires training and evaluating a policy on m random subsets of the dataset (Alg. 1). While increasing m improves estimation quality, the computational cost is dominated by policy training, which scales linearly with m. In our MetaWorld experiments, we found $m = 1 0 ^ { 4 }$ sufficient to obtain high-quality datamodels. This setting required a total of 8 GPU hours on 8 A100 GPUs (equivalent to roughly 60–70 GPU hours on a single GPU). Although this may appear expensive for a single task, the same $m$ trained policies can be reused to compute the proxy objective across all 50 tasks in the MetaWorld suite, effectively amortizing the datamodel training cost to 1–2 hours per task.

Metagradient-based estimator. The metagradient-based estimator is typically 3–5 slower than standard policy training, due to the additional cost of computing metagradients. Taking multiple steps (hyperparameter T in Alg. 2) further improves estimation quality but incurring a higher cost. In MetaWorld, data selection completes within 1 hour on a single A5000 GPU. On LIBERO, parallel training across 4 A5000s requires 16–20 hours—comparable to the training times of VAEbased BR [12] and Flow [11] baselines. For OXE, which is substantially larger (300k trajectories), training takes 3–4 days on 4 A100 GPUs. Notably, DataMIL avoids online rollouts, making training passive and able to run without any human oversight.

## H Hyperparameters for estimating datamodels

Below we provide the hyperparameters used for estimating the datamodels via regression and metagradient-based methods (refer to Alg. 1 and 2).

Regression estimator. We utilize the regression estimator only in the MetaWorld setting. Specifically, we construct training pairs by sampling datapoints with probability $p = 0 . 1$ , training a $4 0 0 \times 3$ MLP policy with ReLU activations and a tanh-squashed Gaussian action head, and evaluating its loss on a held-out dataset. This procedure is repeated $m = 1 0 ^ { 4 }$ times to generate the training set used to regress the datamodel parameters.

Metagradient-based estimator. The hyperparameters used in Alg 2 is listed in Table 4. When we compute the metagradients, instead of computing it over the entire training process (which would be too expensive), we compute it over only a segment of training. Hence, the hyperparameter k refers to the number of

Table 4: Hyperparameters for Metagradient Estimator.

<table><tr><td>Hyperparameter</td><td>Metaworld</td><td>LIBERO</td><td>OXE</td></tr><tr><td>step size p</td><td>1.0</td><td>1.0</td><td>0.125</td></tr><tr><td>number of steps T</td><td>30</td><td>30</td><td>20</td></tr><tr><td>policy architecture policy learner A(·)</td><td>gaussian MLP</td><td>Octo-Small</td><td>Octo-Small</td></tr><tr><td>training steps</td><td>1100</td><td>2100</td><td>5100</td></tr><tr><td>metagradient steps k</td><td>100</td><td>100</td><td>100</td></tr><tr><td>optimizer</td><td>adamw</td><td>adam</td><td>adam</td></tr><tr><td>lr schedular</td><td>1cycle</td><td>cosine</td><td>cosine</td></tr></table>

steps at the end of training, over which the metagradients are computed.