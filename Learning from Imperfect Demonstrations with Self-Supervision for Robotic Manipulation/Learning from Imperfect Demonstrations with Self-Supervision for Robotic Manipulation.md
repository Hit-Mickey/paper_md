# Learning from Imperfect Demonstrations with Self-Supervision for Robotic Manipulation

Kun Wu<sup>1</sup>, Ning Liu<sup>2</sup>, Zhen Zhao<sup>2</sup>, Di Qiu<sup>3</sup>, Jinming Li<sup>4</sup>, Zhengping Che<sup>2</sup>, Zhiyuan Xu<sup>2</sup>, and Jian Tang<sup>2,†</sup>

Abstract— Improving data utilization, especially for imperfect data from task failures, is crucial for robotic manipulation due to the challenging, time-consuming, and expensive data collection process in the real world. Current imitation learning (IL) typically discards imperfect data, focusing solely on successful expert data. While reinforcement learning (RL) can learn from explorations and failures, the sim2real gap and its reliance on dense reward and online exploration make it difficult to apply effectively in real-world scenarios. In this work, we aim to conquer the challenge of leveraging imperfect data without the need for reward information to improve the model performance for robotic manipulation in an offline manner. Specifically, we introduce a Self-Supervised Data Filtering framework (SSDF) that combines expert and imperfect data to compute quality scores for failed trajectory segments. High-quality segments from the failed data are used to expand the training dataset. Then, the enhanced dataset can be used with any downstream policy learning method for robotic manipulation tasks. Extensive experiments on the ManiSkill2 benchmark built on the high-fidelity Sapien simulator and real-world robotic manipulation tasks using the Franka robot arm demonstrated that the SSDF can accurately expand the training dataset with high-quality imperfect data and improve the success rates for all robotic manipulation tasks.

## I. INTRODUCTION

Unlike natural language processing (NLP) and computer vision (CV) tasks, where large volumes of text and image data are easily obtained from the web [1], [2], data for robotic manipulation is scarce and precious due to the high-quality data collection process, especially in real-world applications, is extremely challenging, time-consuming, and costly [3]. This makes improving data utilization, particularly incorporating the imperfect data from failed tasks into the model learning process, a critical issue.

As the two dominant learning paradigms, reinforcement learning (RL) and imitation learning (IL) techniques have been extensively researched and applied in the context of robotic manipulation tasks [3], [4], [5], [6], [7], [8] in recent years. However, they still face significant limitations when it comes to utilizing imperfect data. RL requires an explicit reward function elaborately pre-defined to guide the agent’s action. Nonetheless, many real-world tasks are intrinsically complicated, making the design of an effective reward function challenging [9], [10]. Furthermore, RL relies heavily on simulators to facilitate low-cost exploration and learning in a trial-and-error manner, which is impractical in real-world settings due to high labor costs, safety concerns, and physical constraints. On the other hand, IL circumvents the need for explicit reward functions and online exploration by training policy on pre-collected datasets. This makes IL more feasible for real-world applications [11], [12], [13]. However, IL methods, such as behavioral cloning (BC) [14], operate under the assumption that the data used for training is optimal, which is rarely achievable in practice. Various factors, such as physical fatigue, task complexity, and lineof-sight occlusion, can lead to task failures during data collection. For instance, SayCan [15] collected 276k episodes of data and only retained 12k successful episodes after applying stringent filtering criteria, resulting in more than 90% of the data being wasted. In this work, we aim to address the challenge: When the reward function and online exploration are unavailable, how can we leverage imperfect data from task failures to improve the data utilization and success rate for robotic manipulation?

An ideal solution would be to infer a reward function from expert datasets and accurately label all imperfect data. Some offline reinforcement learning algorithms [16], [17], [18], [19], [20] follow this approach and have achieved promising results in state-based environments. However, inferring a reward function without online interaction is challenging. Additionally, most current reward learning algorithms rely on adversarial learning, which is sensitive to hyperparameters, unstable during training, and struggles to learn effectively in high-dimensional environments. Other IL approaches [21], [22], [23], [24], [25], [26] focus on learning a policy that mimics the expert’s behavior, aiming to mitigate the distri bution shift problem [27] and improve model performance. However, in the context of robotic manipulation, where expert data are scarce and data dimensionality is significantly higher, many methods struggle to align accurately with the data distribution. For instance, training the discriminator in DWBC [24] necessitates costly hyperparameter tuning and struggles with high-dimensional data. Consequently, there is a need for effective offline IL methods capable of leveraging imperfect demonstrations with high-dimensional input.

Our key insight is motivated by the observation: Many steps in a failed trajectory are of high quality despite the overall task failure. For example, when attempting to open a cabinet, the robotic arm may successfully grasp the handle but fail to open the cabinet. In such cases, the failed data leading up to the successful grasp is still valuable. More importantly, the expert data from task success certainly contains data that succeeded in grasping the handler, so these two pieces of data will be very similar, providing hints as to how to select high-quality imperfect data.

To this end, we propose a novel Self-Supervised Data Filtering framework (SSDF) for robotic manipulation, which extends the training dataset using high-quality imperfect data filtered based on the similarity between the failed trajectory segment and the expert dataset. Specifically, SSDF contains three main steps. In step 1, we pre-train a multi-modality transformer network using the mixed demonstrations in a self-supervised manner to extract representative features. Inspired by pertaining process in BERT [28], we propose three self-supervised objectives to enhance the pre-trained transformer, including 1) Masked Transition Prediction, 2) Transition Reconstruction, and 3) Action Autoregression. These approaches enable the transformer to take on different roles like a forward dynamic model, inverse dynamic model, and behavior cloning model and thus enhance its ability to extract effective features. In step 2, we aim to identify the useful imperfect demonstrations according to the quality scores and use them to augment the original expert dataset. Thus, we leverage the pre-trained transformer in step 1 to extract the features and then generate the quality scores by calculating the similarity between the extracted features of the imperfect and expert demonstrations for each state-action pair in the imperfect demonstrations. In step 3, we perform weighted behavior cloning for the high-quality imperfect data, and the original expert dataset can be used for any IL method. To differentiate the quality of imperfect demonstrations and ensure their beneficial contribution, we propose the quality scores calculated in step 2 as weights for the behavior cloning loss. Our contributions are as follows.

• We introduce the Self-Supervised Data Filtering framework (SSDF), which can utilize expert and imperfect demonstrations to train robot manipulation policies.

• We design three self-supervised objectives for feature extraction and a novel similarity calculation method, which can accurately expand the training dataset with high-quality imperfect data.

• Extensive experiments on the ManiSkill2 and real-world tasks demonstrated that SSDF can effectively utilize imperfect demonstrations to improve performance.

## II. RELATED WORK

Imitation Learning has enormous potential to meet critical safety and cost desiderata for real-world applications. As an important class of methods, Behavior Cloning (BC) [14] can be used in the offline setting seamlessly and gain comparable performance compared to GAIL methods [29]. IBC [30], ACT [31], and Diffusion Policy [8] further enhance BC using the network with better expressive capability and advanced training objectives. However, most works are based on the assumption that all demonstrations are optimal, thus dropping their performance with imperfect data. In addition, when scaling up the size of datasets in the real world, such as OpenX [32]and RoboMIND [33], it becomes harder to guarantee that the datasets are all optimal as they get larger.

Some offline reinforcement learning algorithms [16], [17], [18], [19], [20] follow this approach and have achieved promising results in state-based environments. However, inferring a reward function without online interaction is challenging. Additionally, most current reward learning algorithms rely on adversarial learning, which is sensitive to hyperparameters, unstable during training, and struggles to learn effectively in high-dimensional environments.

Offline inverse reinforcement learning (Offline IRL) focuses on training a discriminator in an adversarial manner [34], [35], [36], [37] or learning a reward function [16], [18], [19], [20] to match the distribution of the expert demonstrations. For example, MIIRL [38] proposes to train a conservative world model for learning the reward function from the expert data. Nonetheless, the adversarial optimization process in these methods can lead to training instability and collapse, which is exacerbated in high-dimensional environments. In this work, we aim to conquer the challenge of leveraging imperfect data in the context of robotic manipulation with high-dimensional inputs.

Many IL methods with imperfect demonstrations [21], [39], [40], [41], [42], [43], [44] are also proposed. Nonetheless, most methods require extra labeling or online interaction to provide more information to learn. More recently, several works [23], [25], [24], [26] have been proposed to solve the more challenging problem of offline imitation learning with imperfect demonstrations. ISW-BC [26] proposes an importance-sampling-based method to re-weigh imperfect data and implicitly make the learned policy close to the expert policy. Some other imitation learning methods leveraging powerful architectures [45], [46], such as transformers [47], can flexibly integrate diverse data. These methods typically use goal-conditioned training [48], [49] or pretraining [50], [51], [52] to extract more representative features. However, they are not specifically designed to handle imperfect data. In contrast, our method designs a tailored self-supervised framework for computing the quality scores of imperfect data and is applied to vision-based robotic manipulation.

## III. METHODOLOGY

## A. Preliminary on Imitation Learning

We consider a standard fully observed Markov decision process (MDP) [53] to model the environment. The MDP $\mathcal { M } = ( \mathcal { S } , \mathcal { A } , \mathcal { P } , r , \rho _ { 0 } , \gamma )$ contains the state space $s ,$ , action space A, reward function $r : \mathcal { S } \times \mathcal { A } \times \mathcal { S } \ \to \ \mathbb { R }$ , state transition function $\mathcal { P } : \mathcal { S } \times \mathcal { A } \times \mathcal { S }  [ 0 , 1 ]$ , the initial state distribution $\rho _ { 0 } ( s )$ , and discount factor $\gamma \in ( 0 , 1 )$ . In the IL setting with imperfect demonstrations, the reward function r is unavailable. Instead, we have a static expert demonstration dataset $\mathcal { D } _ { e } = \{ ( s , a , s ^ { \prime } ) \}$ and a static imperfect demonstration dataset $\mathcal { D } _ { i } = \{ ( s , a , s ^ { \prime } ) \}$ . Our goal is to maximize the success rates for the tasks using both the expert dataset and imperfect dataset $\mathcal { D } _ { u } = \mathcal { D } _ { e } \cup \mathcal { D } _ { i }$ without any reward information and online interaction with the environments.

![](images/9b6fd4bd66c8eb44333e75571de588da4ccca666abe80bc5a9f7908d6264c3bd.jpg)  
Fig. 1: Overview of SSDF. SSDF contains three steps: 1) Transformer Pretraining via Self-Supervised Learning, 2) Calculation of Quality Score by Similarity Metric, and 3) Policy Learning with Weighted Behavior Cloning.

## B. Self-Supervised Data Filtering Framework

The overview of the proposed method Self-Supervised Data Filtering framework (SSDF) is depicted in Figure 1. Specifically, SSDF comprises three main steps. (1) Transformer Pretraining via Self-Supervised Learning. (2) Calculation of Quality Score by Similarity Metric. (3) Policy Learning with Weighted Behavior Cloning.

## C. Transformer Pretraining via Self-Supervised Learning

We leverage the transformer as our base model since it is suitable for modeling long dependencies. To pretrain the model, we propose three self-supervised approaches to pre-train the transformer using the union dataset $\mathcal { D } _ { u } ~ = ~ \mathcal { D } _ { e } \cup \mathcal { D } _ { i }$ . As shown in Figure 2, the three tasks are 1) Masked Transition Prediction, 2) Transition Reconstruction, and 3) Action Autoregression. For all tasks, the multi-modality transformer network $T ( \cdot )$ takes a trajectory segment $\tau _ { i n } ~ = ~ ( o _ { i } , m _ { i } , a _ { i } ) , \cdot \cdot \cdot , ( o _ { i + l } , m _ { i + l } , a _ { i + l } )$ as input and outs the corresponding features $\begin{array} { r l } { \tau _ { o u t } } & { { } = } \end{array}$ $( z _ { i } ^ { o } , z _ { i } ^ { m } , z _ { i } ^ { a } ) , \cdot \cdot \cdot , ( z _ { i + l } ^ { o } , z _ { i + l } ^ { m } , z _ { i + l } ^ { a } )$ for all input elements, where i is the start index of the time step, l is the segment length, $o _ { t }$ is the RGBD images given from the depth cameras, $m _ { t }$ is the robot proprioceptive states and necessary information like goal position, $a _ { t }$ is the action. Then we have three separate decoder heads $H _ { o } ( \cdot ) , H _ { m } ( \cdot ) , H _ { a } ( \cdot )$ for the above three kinds of modalities, which takes the corresponding features $z _ { i } ^ { o } , z _ { i } ^ { m } , z _ { i } ^ { a }$ as input and outputs the reconstruction or prediction results $\hat { o } _ { i } , \hat { m } _ { i } , \hat { a } _ { i }$

For Masked Transition Prediction (MTP) tasks, we randomly mask a part of the input trajectory segment with a predefined probability (randomly chosen from 0.4, 0.3, 0.2, 0.1 in our implementation), which is denoted as $M ( \tau )$ . We denote the optimized parameters in the transformer network $T ( \cdot )$ and the decoder heads $H _ { o } ( \cdot ) , H _ { m } ( \cdot ) , H _ { a } ( \cdot )$ as θ. And the goal is given by:

$$
\mathcal {L} _ {\mathrm{MTP}} = \max _ {\theta} \mathbb {E} _ {\tau} \sum_ {t = i} ^ {i + l} \mathbb {I} (e _ {t}) l o g P _ {\theta} (e _ {t} | M (\tau)),\tag{1}
$$

where $e _ { t } \in \{ o _ { t } , m _ { t } , a _ { t } \}$ is the input element of the three modalities, $\mathbb { I } ( e _ { t } )$ is an indicator function showing the input element is masked (i.e., 1) or not (i.e., 0). By randomly masking input elements and predicting them, we encourage the transformer network to learn various roles, including forward dynamic model, inverse dynamic model, and data generating policy that boost the ability of feature extraction.

For Transition Reconstruction (TR) tasks, we randomly mask a part of the input trajectory segment and reconstruct the unmasked elements. The goal is given by:

$$
\mathcal {L} _ {\mathrm{TR}} = \max _ {\theta} \mathbb {E} _ {\tau} \sum_ {t = i} ^ {i + l} (1 - \mathbb {I} (e _ {t})) l o g P _ {\theta} (e _ {t} | M (\tau)).\tag{2}
$$

By reconstructing the unmasked input elements, the transformer network would learn to compress the key information and extract more representative latent features.

To be consistent with the final objective of the robotic manipulation, which is to output action, we propose the Action Autoregression (AA) task that forces the transformer network to predict the next action based on the history transitions. The goal is given by:

$$
\mathcal {L} _ {\mathrm{AA}} = \max _ {\theta} \mathbb {E} _ {\tau} \sum_ {t = i} ^ {i + l} \log P _ {\theta} (a _ {t} | h i s (a _ {t})).\tag{3}
$$

where $h i s ( a _ { t } ) = ( o _ { i } , m _ { i } , a _ { i } ) , \cdot \cdot \cdot , ( o _ { i + t } , m _ { i + t } )$ is the history transitions (i.e., we use causal mask in our implementations).

By combining the above three self-supervised tasks, the final pre-training step objective is as follows:

$$
\mathcal {L} _ {\mathrm{pretrain}} = \mathcal {L} _ {\mathrm{MTP}} + \mathcal {L} _ {\mathrm{TR}} + \mathcal {L} _ {\mathrm{AA}}.\tag{4}
$$

## D. Calculation of Quality Score by Similarity Metric

Since the expert demonstrations have high quality, we believe that the imperfect demonstrations that is more similar to expert demonstrations is of higher quality. After pretraining the transformer network in a self-supervised manner, the transformer network now has an enhanced ability to extract expressive and representative features $z _ { i } ^ { o } , z _ { i } ^ { m } , z _ { i } ^ { a }$ for the input elements. Thus, we choose to use the output features of the expert demonstrations and the imperfect demonstrations to calculate the similarities and final quality scores. More specifically, for each trajectory segment $\tau ^ { i m p }$ in the imperfect demonstrations, we extract the output features $z _ { i + l } ^ { o , i m p ^ { \bullet } } , z _ { i + l } ^ { m , i m p } , z _ { i + l } ^ { a , i m p }$ at the last time step. We also extract the output features $z _ { i + l } ^ { o , e x p } , z _ { i + l } ^ { m , e x p } , z _ { i + l } ^ { a , e x p }$ of each expert trajectory segment $\tau ^ { e x p }$ in the same way. The similarity between the imperfect features and the expert features are defined as the negative $L _ { 2 }$ distance:

![](images/dde37d9346b5ec3c531570e52e7893062855516fbb0b2081f97ff57ae3d7744b.jpg)  
Fig. 2: The transformer pre-training process includes three tasks: 1) Masked Transition Prediction (MTP), 2) Transition Reconstruction (TR), and 3) Action Autoregression $( \mathsf { A A } )$ Here is an example of the pre-training process with three time-step inputs. Colored lines link the input and output.

$$
\begin{array}{r} s i m (\tau^ {i m p}, \tau^ {e x p}) = - | | z _ {i + l} ^ {o, i m p} - z _ {i + l} ^ {o, e x p} | | _ {2} \\ - | | z _ {i + l} ^ {m, i m p} - z _ {i + l} ^ {m, e x p} | | _ {2} - | | z _ {i + l} ^ {a, i m p} - z _ {i + l} ^ {a, e x p} | | _ {2}. \end{array}\tag{5}
$$

The reason why we choose the features of the last time step $z _ { i + l }$ of the trajectory segment is that generally the last time step is the most important step for predicting the next action, and the attention mechanism in the transformer has extracted the historical information. For each imperfect trajectory segment, we need to calculate the similarities with all other expert trajectory segments and the similarity result $w ( \tau ^ { i m p } )$ is the highest one, which is as follows:

$$
w (\tau^ {i m p}) = \max s i m (\tau^ {i m p}, \tau^ {e x p}), \forall \tau^ {e x p} \in \mathcal {D} _ {e}.\tag{6}
$$

Once we get similarities of all imperfect segments, we obtain the quality scores by normalizing them:

$$
q (\tau^ {i m p}) = n o r m (w (\tau^ {i m p})), q (\tau^ {i m p}) \in [ 0, 1 ]\tag{7}
$$

## E. Policy Learning with Weighted Behavior Cloning

After calculating the quality scores for all imperfect trajectory segments $\tau ^ { i m p }$ , we can rank them and filter the highquality part out to learn the optimal policy. In order to prevent low-quality imperfect demonstrations from interfering with policy learning, we define a threshold $\beta$ and choose the imperfect segments $\tau ^ { i m p }$ as follows:

$$
\mathcal {D} _ {f} = \{\tau^ {i m p} | q (\tau^ {i m p}) > \beta \}\tag{8}
$$

For the reserved high-quality dataset $\mathcal { D } _ { f }$ , we use their quality scores as the weights for behavior cloning. Combined with the expert demonstrations $\mathcal { D } _ { e }$ , the final objective for downstream policy learning is given by:

$$
\mathcal {L} _ {\text { full }} = \min _ {\theta^ {\prime}} \mathcal {L} _ {\text { policy }} - \lambda \mathbb {E} _ {\tau \sim \mathcal {D} _ {f}} [ q (\tau) l o g P _ {\theta^ {\prime}} (a _ {i + l} | \tau) ],\tag{9}
$$

where ${ \mathcal { L } } _ { \mathrm { p o l i c y } }$ is the objective for any downstream policy model parameterized by $\overline { { { \theta } } } ^ { \prime }$ on the expert dataset $\mathcal { D } _ { e } , \lambda$ is a hyperparameter to balance the training of the expert demonstrations and imperfect demonstrations. We finetune the transformer in our simulation experiments and use ACT [31] in our real-world experiments. We can count the distribution of quality scores $q ( \tau )$ offline and then adjust the value of $\beta$ accordingly. We set $l ~ = ~ 6 , \lambda ~ = ~ 0 . 1 , \beta ~ = ~ 0 . 9$ in our implementation.

## IV. EXPERIMENTS

## A. Experiment Setup

Simulation Experiments. Our simulation experiments are based on the ManiSkill2 benchmark [54], which is built on the high-fidelity Sapien simulator [55] and allows agents to be trained using static demonstrations. The tasks include three rigid object manipulation tasks and two soft-body tasks as shown in Figure 3. For data collection, we trained different level behavior agents using the online DAPG+PPO [56], [57] method provided in the ManiSkill2.

PickCube-v0 is to pick up a red cube and move it to a specified endpoint. We collected 300 expert trajectories and 900 imperfect trajectories consisting of three groups of 300 trajectories, which are collected by three level agents with success rates of 0.0, 0.46, and 0.91, respectively.

StackCube-v0 is to pick up a red cube and place it onto the green one. We collected 300 expert trajectories and 900 imperfect trajectories consisting of three groups of 300 trajectories, which are generated by three level agents with success rates of 0.0, 0.60, and 0.86, respectively.

PickYCB-v0 is to pick up a YCB object [58] and move it to a specified endpoint. We selected 10 objects from all objects that are relatively easy to pick up. We collected 300 expert trajectories and 300 other imperfect trajectories from an imperfect agent with success rates of 0.41.

Fill-v0 is to fill the target beaker with clay in a bucket. 400 expert trajectories and 400 imperfect trajectories are collected from an imperfect agent with success rates of 0.35. Hang-v0 is to hang a noodle on a specified stick. We collected 400 expert trajectories and 400 imperfect trajectories from an agent with success rates of 0.14.

For the PickYCB, Fill and Hang tasks, we only collected one level of imperfect demonstrations because the behavior policies with high success rates are not available. For all tasks, the robot initialization pose, object position, and goal position are randomly generated for each episode. The robot is a 7-DoF Franka Panda robot with a parallel-jaw gripper. The input RGBD images $o \in \mathbb { R } ^ { 2 * 1 2 8 * \bar { 1 } 2 8 * 4 }$ are from two cameras, including a hand-eye camera and a top fixed camera. The input robot proprioceptive states $m \ \in \ \mathbb { R } ^ { 3 8 }$ include joint positions, joint velocities, robot base position, goal position, and end-effector position. The output action $a \in \mathbb { R } ^ { 7 }$ is the delta target end-effector pose. We use the success rate to measure the performance of an algorithm. Please refer to the Maniskill2 [54] for more details.

![](images/3e71b771240773a2d17ee97d4fca34afe7ad6148125bce9d740d0715bed55efe.jpg)  
Fig. 3: Experiment Setup: we conducted experiments on five tasks on the ManiSkill2 benchmark and five tasks using the real-world single Franka robotic arm. Fill and Hang are soft-body tasks. The green circles in PickCube and PickYCB represent the goal position.

TABLE I: Comparisons in terms of success rates of the five tasks on ManiSkill2 benchmark. BC and TF-BC are the baselines that only use expert demonstrations. All other baselines used imperfect demonstrations.

<table><tr><td>Method</td><td>PickCube</td><td>StackCube</td><td>PickYCB</td><td>Fill</td><td>Hang</td></tr><tr><td>BC [14]</td><td>82.6</td><td>80.0</td><td>43.6</td><td>24.8</td><td>14.4</td></tr><tr><td>TF-BC [47]</td><td>82.2</td><td>81.6</td><td>42.4</td><td>25.2</td><td>16.0</td></tr><tr><td>TT [45]</td><td>22.0</td><td>20.0</td><td>12.4</td><td>14.8</td><td>20.4</td></tr><tr><td>BeT [46]</td><td>82.6</td><td>81.2</td><td>34.6</td><td>10.6</td><td>18.6</td></tr><tr><td>PACT [50]</td><td>82.0</td><td>83.2</td><td>38.4</td><td>14.6</td><td>16.0</td></tr><tr><td>MTM [51]</td><td>66.6</td><td>79.4</td><td>32.0</td><td>23.2</td><td>18.4</td></tr><tr><td>SMART [52]</td><td>65.4</td><td>81.2</td><td>25.2</td><td>8.8</td><td>16.2</td></tr><tr><td>DT [48]</td><td>30.8</td><td>78.8</td><td>36.8</td><td>4.4</td><td>6.0</td></tr><tr><td>C-BeT [49]</td><td>78.6</td><td>83.6</td><td>28.0</td><td>20.0</td><td>16.4</td></tr><tr><td>DemoDICE [23]</td><td>76.4</td><td>78.8</td><td>39.6</td><td>16.0</td><td>8.8</td></tr><tr><td>DWBC [24]</td><td>32.8</td><td>64.0</td><td>26.8</td><td>10.4</td><td>4.0</td></tr><tr><td>ISW-BC [26]</td><td>79.2</td><td>82.4</td><td>45.4</td><td>26.2</td><td>18.8</td></tr><tr><td>SSDF-base</td><td>83.6</td><td>84.0</td><td>44.4</td><td>26.0</td><td>16.0</td></tr><tr><td>SSDF</td><td>85.6</td><td>88.0</td><td>50.4</td><td>29.6</td><td>28.4</td></tr></table>

Real-world Experiments. We follow Droid [59] to build our real-world robotic arm environment. As shown in Figure 3, we use a Franka Panda 7-DoF robot arm. The input RGB images $o \in \mathbb { R } ^ { 2 * 2 7 0 * 4 8 0 * 3 }$ are from two external fixedview Zed 2 stereo cameras. We collect the dataset through human demonstration. For each task, we randomly place the objects within a specified area and collect 30 success trajectories as the expert dataset and 70 failed trajectories as the imperfect dataset. We record the RGB images from two camera views and robot proprioceptive states, e.g., joint position $m \in \mathbb { R } ^ { 7 }$ . Our model predicts the 6D pose, including position $( x , y , z )$ and rotation (roll, pitch, yaw).

## B. Evaluation Results

Results on Simulation. We comprehensively compared SSDF to many state-of-the-art offline imitation learning algorithms including 4 types. 1) IL methods that only use expert demonstrations: Behavior Cloning (BC) [14], Transformer-based Behavior Cloning (TF-BC) [47], Trajectory Transformer (TT) [45], and Behavior Transformer (BeT) [46]. 2) Pretraining methods that pre-train model with mixed data and then finetune using expert data: Perception-Action Causal Transformer (PACT) [50], Masked Trajectory Models (MTM) [51], and Self-supervised Multitask pretrAining with contRol Transformer (SMART) [52]. 3) Goal-conditioned methods that take task completion signal as input: Decision Transformer (DT) [48], and Conditional Behavior Transformers (C-BeT) [49]. 4) IL methods that rank imperfect data: DemoDICE [23], Discriminator-Weighted Behavioral Cloning (DWBC) [24], and Importance Sampling Weighted Behavior Cloning (ISW-BC) [26]. To verify the effectiveness of the pre-training process, we trained SSDF-base that only leverages the imperfect demonstrations in the pre-training process. We trained all methods for 100k gradient steps and evaluated 50 episodes every 5k steps. The final results are calculated as the average success rates of the last 5 checkpoints over 3 random seeds.

TABLE II: Comparisons in terms of success rates of the five tasks on single-arm Franka robot.

<table><tr><td>Method</td><td>Place Tennis</td><td>Open Drawer</td><td>Open Box</td><td>Stack Block</td><td>Arrange Flower</td><td>Avg.</td></tr><tr><td>BeT [46]</td><td>20</td><td>30</td><td>25</td><td>10</td><td>0</td><td>17</td></tr><tr><td>ACT [31]</td><td>45</td><td>45</td><td>65</td><td>20</td><td>20</td><td>39</td></tr><tr><td>Diffusion Policy [60]</td><td>35</td><td>35</td><td>60</td><td>25</td><td>20</td><td>35</td></tr><tr><td>ISW-BC [26]</td><td>40</td><td>45</td><td>65</td><td>20</td><td>15</td><td>37</td></tr><tr><td>SSDF+ACT</td><td>50</td><td>55</td><td>65</td><td>30</td><td>25</td><td>45</td></tr></table>

Table I shows that SSDF outperformed all baselines consistently on all 5 tasks by a large margin. For IL methods that rank imperfect data [23], [24], [26], once they assign high weights to low-quality data points, it in turn interferes with the results of the training. For example, DWBC only achieved a success rate of 32.8 on the PickCube task. This is due to the difficulty of the discriminator in predicting accurate weights for high-dimensional image inputs, which drastically reduced the success rate. In contrast, SSDF can improve performance by accurately calculating the quality scores and filtering out high-quality demonstrations.

Results on Single-arm Franka Robot. For the singlearm franka robotic tasks, we compared 4 state-of-the-art methods that are BeT [46], Action Chunking with Transformers (ACT) [31], Diffusion Policy [60], and ISW-BC [26]. We trained all methods for 200 epochs and evaluated the final checkpoints. As shown in Table II, we observe that SSDF+ACT achieves average success rates of 45% on these 5 tasks, which outperforms ACT by 6%. This increase in success rate shows that SSDF can select high-quality imperfect data, expand the original expert dataset, and thus improve the training of models.

TABLE III: Ablation study on the different self-supervised objectives used in the training step 1 on the StackCube task.

<table><tr><td>StackCube</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>SSDF</td></tr><tr><td>MTP</td><td>√</td><td></td><td></td><td>√</td><td>√</td><td></td><td>√</td></tr><tr><td>TR</td><td></td><td>√</td><td></td><td>√</td><td></td><td>√</td><td>√</td></tr><tr><td>AA</td><td></td><td></td><td>√</td><td></td><td>√</td><td>√</td><td>√</td></tr><tr><td>Result</td><td>75.4</td><td>72.0</td><td>78.8</td><td>80.6</td><td>84.2</td><td>83.8</td><td>88.0</td></tr></table>

TABLE IV: Comparisons of different quality score threshold $\beta$ on the StackCube task.

<table><tr><td>Threshold  $\beta$ </td><td>0.0</td><td>0.50</td><td>0.70</td><td>0.80</td><td>0.90</td><td>0.95</td><td>0.99</td></tr><tr><td>success rate</td><td>17.2</td><td>21.6</td><td>43.8</td><td>66.4</td><td>88.0</td><td>84.4</td><td>83.6</td></tr><tr><td>reserved data</td><td>180k</td><td>57k</td><td>35k</td><td>16k</td><td>8.8k</td><td>3.2k</td><td>0.3k</td></tr></table>

## C. Ablation Study in Simulation

Impact of different self-supervised objectives. On the StackCube task, we evaluated SSDF with different combinations of the 3 self-supervised objectives in training step 1. In Table III, as we continue to increase the self-supervised objectives, SSDF can achieve higher success rates. This is because different self-supervised learning losses enable the model to learn versatile features, allowing it to compute quality scores for imperfect data accurately.

Impact of quality score threshold $\beta .$ . As shown in Table IV, on the StackCube task, we can observe that as the quality score threshold value $\beta$ increases, the reserved demonstrations become less and less. The highest success rate on the StackCube task is with $\beta$ as 0.90. When β is less than 0.90, the performance increases as $\beta$ increases because more and more low-quality demonstrations are discarded. When $\beta$ is larger than 0.90, we argue that the performance drops as $\beta$ increases because more high-quality demonstrations are discarded, degrading the task performance.

Impact of different similarity calculation functions. We examined $L _ { 2 }$ distance and Cosine similarity distance with the whole segment features denoted as “Seg.” and the last time step feature denoted as “Last” on the StackCube task. As shown in Table ${ \mathrm { v } } ,$ “Last+ $\cdot \boldsymbol { L _ { 2 } } ^ { \prime }$ achieved the highest performance, while the results of using cosine similarity are relatively lower. We argue that the last time step of the trajectory segment is the most important step for predicting the next action, and the attention mechanism in the transformer has extracted the historical information.

Impact of the imperfect demonstration quantity and quality. For the StackCube task, we collected five different level imperfect demonstrations using different behavior policies with success rates of 0.00, 0.23, 0.44, 0.60, and 0.86, respectively . For each level, we collected imperfect demonstrations in two ratios, 1:1 and 1:3, where the former 1 represents the 300 expert demonstrations.

TABLE V: Comparisons of different similarity functions on the StackCube task.

<table><tr><td></td><td>Seg.+Cosine</td><td>Last+Cosine</td><td>Seg.+ $L_2$ </td><td>Last+ $L_2$ </td></tr><tr><td>success rate</td><td>77.2</td><td>83.2</td><td>82.0</td><td>88.0</td></tr></table>

![](images/d2b3ef80c27001c4fc88d14a9c31477790cd8c935d559a23e8eac7517a62f1a8.jpg)  
Fig. 4: Comparisons of different quantity and quality of the imperfect data. The x-axis represents the success rates of the behavior policies that collect the imperfect data. The y-axis shows the success rates of SSDF. The green and red lines represent the ratios of the expert and imperfect data, which are 1:1 and 1:3, respectively.

Figure 4 shows the success rates with different imperfect demonstration quantities and qualities on the StackCube task. An interesting finding is that those low-quality and highquality imperfect demonstrations can benefit the performance compared to the medium-quality imperfect demonstrations. We believe that low-quality demonstrations contain rich explorations of the environment and therefore can help extract more informative features in the pre-training process. Then they are discarded in the fine-tuning stage and do not interfere with the results. In contrast, high-quality imperfect demonstrations directly help the process of the fine-tuning stage. The medium-quality demonstrations neither provide rich environmental information nor directly enhance policy learning, so they achieve the most minor improvement.

## V. CONCLUSION

This paper explores how to leverage imperfect demonstration to improve policy learning without reward information and online exploration. To this end, we propose a novel Self-Supervised Data Filtering framework (SSDF) that calculates accurate quality scores using a pre-trained transformer and then does weighted behavior cloning with the high-quality imperfect demonstrations. The extensive experimental results in simulation and real-world applications demonstrated that SSDF can accurately select high-quality imperfect demonstrations to boost the final performance.

[1] J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat, et al., “Gpt-4 technical report,” arXiv preprint arXiv:2303.08774, 2023.

[2] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, P. Dollar, and R. Gir-´ shick, “Segment anything,” arXiv:2304.02643, 2023.

[3] A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, J. Dabis, C. Finn, K. Gopalakrishnan, K. Hausman, A. Herzog, J. Hsu, et al., “Rt-1: Robotics transformer for real-world control at scale,” arXiv preprint arXiv:2212.06817, 2022.

[4] S. Gu, E. Holly, T. Lillicrap, and S. Levine, “Deep reinforcement learning for robotic manipulation with asynchronous off-policy updates,” in 2017 IEEE international conference on robotics and automation (ICRA). IEEE, 2017.

[5] T. Zhang, Z. McCarthy, O. Jow, D. Lee, X. Chen, K. Goldberg, and P. Abbeel, “Deep imitation learning for complex manipulation tasks from virtual reality teleoperation,” in 2018 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2018.

[6] E. Johns, “Coarse-to-fine imitation learning: Robot manipulation from a single demonstration,” in 2021 IEEE international conference on robotics and automation (ICRA). IEEE, 2021.

[7] J. Thumm and M. Althoff, “Provably safe deep reinforcement learning for robotic manipulation in human environments,” in 2022 International Conference on Robotics and Automation (ICRA). IEEE, 2022.

[8] C. Chi, S. Feng, Y. Du, Z. Xu, E. Cousineau, B. Burchfiel, and S. Song, “Diffusion policy: Visuomotor policy learning via action diffusion,” in Proceedings of Robotics: Science and Systems (RSS), 2023.

[9] A. Y. Ng, D. Harada, and S. Russell, “Policy invariance under reward transformations: Theory and application to reward shaping,” in Icml. Citeseer, 1999.

[10] M. Kwon, S. M. Xie, K. Bullard, and D. Sadigh, “Reward design with language models,” in The Eleventh International Conference on Learning Representations, 2023.

[11] R. Rahmatizadeh, P. Abolghasemi, L. Bol¨ oni, and S. Levine, “Vision-¨ based multi-task manipulation for inexpensive robots using end-to-end learning from demonstration,” in 2018 IEEE international conference on robotics and automation (ICRA). IEEE, 2018.

[12] E. Jang, A. Irpan, M. Khansari, D. Kappler, F. Ebert, C. Lynch, S. Levine, and C. Finn, “Bc-z: Zero-shot task generalization with robotic imitation learning,” in Conference on Robot Learning. PMLR, 2022.

[13] B. Zitkovich, T. Yu, S. Xu, P. Xu, T. Xiao, F. Xia, J. Wu, P. Wohlhart, S. Welker, A. Wahid, quan vuong, V. Vanhoucke, H. Tran, R. Soricut, A. Singh, J. Singh, P. Sermanet, P. R. Sanketi, G. Salazar, M. S. Ryoo, K. Reymann, K. Rao, K. Pertsch, I. Mordatch, H. Michalewski, Y. Lu, S. Levine, L. Lee, T.-W. E. Lee, I. Leal, Y. Kuang, D. Kalashnikov, R. Julian, N. J. Joshi, A. Irpan, brian ichter, J. Hsu, A. Herzog, K. Hausman, K. Gopalakrishnan, C. Fu, P. Florence, C. Finn, K. A. Dubey, D. Driess, T. Ding, K. M. Choromanski, X. Chen, Y. Chebotar, J. Carbajal, N. Brown, A. Brohan, M. G. Arenas, and K. Han, “RT-2: Vision-language-action models transfer web knowledge to robotic control,” in 7th Annual Conference on Robot Learning, 2023.

[14] D. A. Pomerleau, “Alvinn: An autonomous land vehicle in a neural network,” Advances in neural information processing systems, 1988.

[15] A. Brohan, Y. Chebotar, C. Finn, K. Hausman, A. Herzog, D. Ho, J. Ibarz, A. Irpan, E. Jang, R. Julian, et al., “Do as i can, not as i say: Grounding language in robotic affordances,” in Conference on Robot Learning. PMLR, 2023.

[16] K. Zolna, A. Novikov, K. Konyushkova, C. Gulcehre, Z. Wang, Y. Aytar, M. Denil, N. de Freitas, and S. Reed, “Offline learning from demonstrations and unlabeled experience,” arXiv preprint arXiv:2011.13885, 2020.

[17] J. Chang, M. Uehara, D. Sreenivas, R. Kidambi, and W. Sun, “Mitigating covariate shift in imitation learning via offline data with partial coverage,” Advances in Neural Information Processing Systems, vol. 34, pp. 965–979, 2021.

[18] S. Yue, G. Wang, W. Shao, Z. Zhang, S. Lin, J. Ren, and J. Zhang, “CLARE: Conservative model-based reward learning for offline inverse reinforcement learning,” in The Eleventh International Conference on Learning Representations, 2023. [Online]. Available: https://openreview.net/forum?id=5aT4ganOd98

[19] S. Zeng, C. Li, A. Garcia, and M. Hong, “When demonstrations meet generative world models: A maximum likelihood framework for of-

fline inverse reinforcement learning,” Advances in Neural Information Processing Systems, vol. 36, 2024.

[20] G. Cideron, B. Tabanpour, S. Curi, S. Girgin, L. Hussenot, G. Dulac-Arnold, M. Geist, O. Pietquin, and R. Dadashi, “Get back here: Robust imitation by return-to-distribution planning,” arXiv preprint arXiv:2305.01400, 2023.

[21] Y.-H. Wu, N. Charoenphakdee, H. Bao, V. Tangkaratt, and M. Sugiyama, “Imitation learning from imperfect demonstration,” in International Conference on Machine Learning. PMLR, 2019.

[22] F. Sasaki and R. Yamashina, “Behavioral cloning from noisy demonstrations,” in International Conference on Learning Representations, 2020.

[23] G.-H. Kim, S. Seo, J. Lee, W. Jeon, H. Hwang, H. Yang, and K.- E. Kim, “Demodice: Offline imitation learning with supplementary imperfect demonstrations,” in International Conference on Learning Representations, 2021.

[24] H. Xu, X. Zhan, H. Yin, and H. Qin, “Discriminator-weighted offline imitation learning from suboptimal demonstrations,” in International Conference on Machine Learning. PMLR, 2022.

[25] L. Yu, T. Yu, J. Song, W. Neiswanger, and S. Ermon, “Offline imitation learning with suboptimal demonstrations via relaxed distribution matching,” in Proceedings of the AAAI conference on artificial intelligence, 2023.

[26] Z. Li, T. Xu, Z. Qin, Y. Yu, and Z.-Q. Luo, “Imitation learning from imperfection: Theoretical justifications and algorithms,” Advances in Neural Information Processing Systems, vol. 36, 2024.

[27] R. F. Prudencio, M. R. Maximo, and E. L. Colombini, “A survey on offline reinforcement learning: Taxonomy, review, and open problems,” IEEE Transactions on Neural Networks and Learning Systems, 2023.

[28] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “Bert: Pre-training of deep bidirectional transformers for language understanding,” arXiv preprint arXiv:1810.04805, 2018.

[29] J. Ho and S. Ermon, “Generative adversarial imitation learning,” Advances in neural information processing systems, 2016.

[30] P. Florence, C. Lynch, A. Zeng, O. A. Ramirez, A. Wahid, L. Downs, A. Wong, J. Lee, I. Mordatch, and J. Tompson, “Implicit behavioral cloning,” in Conference on Robot Learning. PMLR, 2022.

[31] T. Z. Zhao, V. Kumar, S. Levine, and C. Finn, “Learning fine-grained bimanual manipulation with low-cost hardware,” arXiv preprint arXiv:2304.13705, 2023.

[32] A. O’Neill, A. Rehman, A. Maddukuri, A. Gupta, A. Padalkar, A. Lee, A. Pooley, A. Gupta, A. Mandlekar, A. Jain, et al., “Open x-embodiment: Robotic learning datasets and rt-x models: Open xembodiment collaboration 0,” in 2024 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2024, pp. 6892–6903.

[33] K. Wu, C. Hou, J. Liu, Z. Che, X. Ju, Z. Yang, M. Li, Y. Zhao, Z. Xu, G. Yang, et al., “Robomind: Benchmark on multi-embodiment intelligence normative data for robot manipulation,” arXiv preprint arXiv:2412.13877, 2024.

[34] I. Kostrikov, O. Nachum, and J. Tompson, “Imitation learning via offpolicy distribution matching,” in International Conference on Learning Representations, 2020.

[35] M. Sun, A. Mahajan, K. Hofmann, and S. Whiteson, “Softdice for imitation learning: Rethinking off-policy distribution matching,” arXiv preprint arXiv:2106.03155, 2021.

[36] G. Swamy, S. Choudhury, Z. S. Wu, and J. A. Bagnell, “Of moments and matching: Trade-offs and treatments in imitation learning,” arXiv preprint arXiv:2103.03236, 2021.

[37] F. Jarboui and V. Perchet, “Offline inverse reinforcement learning,” arXiv preprint arXiv:2106.05068, 2021.

[38] S. Zeng, C. Li, A. Garcia, and M. Hong, “Maximum-likelihood inverse reinforcement learning with finite-time guarantees,” Advances in Neural Information Processing Systems, vol. 35, pp. 10 122–10 135, 2022.

[39] L. Wang, W. Zhang, X. He, and H. Zha, “Supervised reinforcement learning with recurrent neural network for dynamic treatment recommendation,” in Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining, 2018.

[40] K. Brantley, W. Sun, and M. Henaff, “Disagreement-regularized imitation learning,” in International Conference on Learning Representations, 2019.

[41] D. Brown, W. Goo, P. Nagarajan, and S. Niekum, “Extrapolating beyond suboptimal demonstrations via inverse reinforcement learning from observations,” in International conference on machine learning. PMLR, 2019.

[42] D. S. Brown, W. Goo, and S. Niekum, “Better-than-demonstrator imitation learning via automatically-ranked demonstrations,” in Conference on robot learning. PMLR, 2020.

[43] V. Tangkaratt, B. Han, M. E. Khan, and M. Sugiyama, “Variational imitation learning with diverse-quality demonstrations,” in Proceedings of the 37th International Conference on Machine Learning, 2020.

[44] M. Du, S. Nair, D. Sadigh, and C. Finn, “Behavior retrieval: Fewshot imitation learning by querying unlabeled datasets,” arXiv preprint arXiv:2304.08742, 2023.

[45] M. Janner, Q. Li, and S. Levine, “Offline reinforcement learning as one big sequence modeling problem,” Advances in neural information processing systems, vol. 34, pp. 1273–1286, 2021.

[46] N. M. Shafiullah, Z. Cui, A. A. Altanzaya, and L. Pinto, “Behavior transformers: Cloning k modes with one stone,” Advances in neural information processing systems, vol. 35, pp. 22 955–22 968, 2022.

[47] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” Advances in neural information processing systems, 2017.

[48] L. Chen, K. Lu, A. Rajeswaran, K. Lee, A. Grover, M. Laskin, P. Abbeel, A. Srinivas, and I. Mordatch, “Decision transformer: Reinforcement learning via sequence modeling,” Advances in neural information processing systems, 2021.

[49] Z. J. Cui, Y. Wang, N. M. M. Shafiullah, and L. Pinto, “From play to policy: Conditional behavior generation from uncurated robot data,” arXiv preprint arXiv:2210.10047, 2022.

[50] R. Bonatti, S. Vemprala, S. Ma, F. Frujeri, S. Chen, and A. Kapoor, “Pact: Perception-action causal transformer for autoregressive robotics pre-training,” in 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2023.

[51] P. Wu, A. Majumdar, K. Stone, Y. Lin, I. Mordatch, P. Abbeel, and A. Rajeswaran, “Masked trajectory models for prediction, representation, and control,” in International Conference on Machine Learning. PMLR, 2023.

[52] Y. Sun, S. Ma, R. Madaan, R. Bonatti, F. Huang, and A. Kapoor, “SMART: Self-supervised multi-task pretraining with control transformers,” in International Conference on Learning Representations, 2023.

[53] R. S. Sutton, A. G. Barto, et al., Introduction to reinforcement learning. MIT press Cambridge, 1998.

[54] J. Gu, F. Xiang, X. Li, Z. Ling, X. Liu, T. Mu, Y. Tang, S. Tao, X. Wei, Y. Yao, X. Yuan, P. Xie, Z. Huang, R. Chen, and H. Su, “Maniskill2: A unified benchmark for generalizable manipulation skills,” in International Conference on Learning Representations, 2023.

[55] F. Xiang, Y. Qin, K. Mo, Y. Xia, H. Zhu, F. Liu, M. Liu, H. Jiang, Y. Yuan, H. Wang, L. Yi, A. X. Chang, L. J. Guibas, and H. Su, “SAPIEN: A simulated part-based interactive environment,” in The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2020.

[56] A. Rajeswaran, V. Kumar, A. Gupta, G. Vezzani, J. Schulman, E. Todorov, and S. Levine, “Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations,” in Proceedings of Robotics: Science and Systems (RSS), 2018.

[57] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal policy optimization algorithms,” arXiv preprint arXiv:1707.06347, 2017.

[58] B. Calli, A. Singh, A. Walsman, S. Srinivasa, P. Abbeel, and A. M. Dollar, “The ycb object and model set: Towards common benchmarks for manipulation research,” in 2015 International Conference on Advanced Robotics (ICAR), 2015.

[59] A. Khazatsky, K. Pertsch, S. Nair, A. Balakrishna, S. Dasari, S. Karamcheti, S. Nasiriany, M. K. Srirama, L. Y. Chen, K. Ellis, et al., “Droid: A large-scale in-the-wild robot manipulation dataset,” arXiv preprint arXiv:2403.12945, 2024.

[60] C. Cheng, F. Siyuan, D. Yilun, X. Zhenjia, C. Eric, B. Benjamin, and S. Shuran, “Diffusion policy: Visuomotor policy learning via action diffusion,” in Robotics: Science and Systems (RSS), 2023.