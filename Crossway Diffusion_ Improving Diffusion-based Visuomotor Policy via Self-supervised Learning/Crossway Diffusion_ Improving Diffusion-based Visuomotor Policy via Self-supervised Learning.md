# Crossway Diffusion: Improving Diffusion-based Visuomotor Policy via Self-supervised Learning

Xiang Li, Varun Belagali, Jinghuan Shang and Michael S. Ryoo

Abstract— Sequence modeling approaches have shown promising results in robot imitation learning. Recently, diffusion models have been adopted for behavioral cloning in a sequence modeling fashion, benefiting from their exceptional capabilities in modeling complex data distributions. The standard diffusionbased policy iteratively generates action sequences from random noise conditioned on the input states. Nonetheless, the model for diffusion policy can be further improved in terms of visual representations. In this work, we propose Crossway Diffusion, a simple yet effective method to enhance diffusionbased visuomotor policy learning via a carefully designed state decoder and an auxiliary self-supervised learning (SSL) objective. The state decoder reconstructs raw image pixels and other state information from the intermediate representations of the reverse diffusion process. The whole model is jointly optimized by the SSL objective and the original diffusion loss. Our experiments demonstrate the effectiveness of Crossway Diffusion in various simulated and real-world robot tasks, confirming its consistent advantages over the standard diffusionbased policy and substantial improvements over the baselines.

## I. INTRODUCTION

Behavioral Cloning (BC) [1] is a supervised learning formulation for robot action policy learning. Given expert demonstration data consisting of a sequence of state-action pairs, a model is trained to predict the correct action vector given input states (e.g., images). This framework has shown to be very effective particularly when a sufficient amount of training data is provided [2].

Recently, sequence modeling approaches [3], [4], [5] have been often used for behavioral cloning, because of their ability to model multiple steps of information. In such a formulation, the objective is to model the probability distribution of the multi-step state-action trajectory. This allows BC to consider beyond a single-step regression, better-taking advantage of history. Given the success in modeling human language [6] and images [7], Transformers [8] have been popularly adopted for sequence modeling-based policies [3], [4], [5], [2], [9].

Diffusion models [10], [11], [12], [13] have exceptional capabilities in modeling multimodal data distribution and generating new samples from the distribution, which make them suitable for imitating behaviors by generating trajectories. Recent works [14], [15], [16] have successfully applied diffusion models for sequential modeling using low dimensional states. For visuomotor control tasks, [17] demonstrated promising performance using multimodal states including visual observations as the conditions of the diffusion model.

In this work, we propose Crossway Diffusion, a simple yet effective method to enhance diffusion-based visuomotor policy learning via a carefully designed state decoder and a self-supervised learning (SSL) objective. The state decoder reconstructs raw image pixels and other state information from the intermediate representations of the reverse diffusion process. The SSL objective and the original diffusion objective jointly optimize the whole model.

By doing so, we explicitly regularize the intermediate representation to capture the information of the input states, as now the model needs to reconstruct the input states from it. Through our experiments over multiple challenging tasks from different benchmarks, we verify the consistent advantage of Crossway Diffusion in comparison to the baseline Diffusion Policy [17]. Especially, our method achieves a 15.7% improvement in success rate over the baseline on ‘Transport, mh’ dataset from Robomimic [18] and one variant of our method even achieves 17.1%, emphasizing the effectiveness of the SSL objective and the specific state decoder design. Our contribution can be summarized as follows:

• We propose Crossway Diffusion, which consistently improves diffusion-based visuomotor policy via a carefully designed state decoder and a simple SSL objective.

• We confirm the effectiveness of the proposed method on multiple challenging visual BC tasks from different benchmarks, including two real-world robot manipulation datasets we collected.

• We conduct detailed ablations on multiple design choices, verifying the advance and robustness of the proposed design.

## II. PRELIMINARIES

## A. Behavioral Cloning

We consider simple behavioral cloning (BC) setting over a Markov Decision Process (MDP), described by the tuple $( \cal { S } , \cal { A } , \cal { P } )$ , where $s \in \mathcal { S }$ represents the state, $a \in \mathcal A$ is the action, and $P$ are the transition dynamics given by $P ( s ^ { \prime } | s , a )$ . A trajectory consists of a sequence of state-action pairs $\left\{ s _ { 0 } , a _ { 0 } , s _ { 1 } , a _ { 1 } , . . . , s _ { T } , a _ { T } \right\}$ , where T is the length of sequence (task horizon). Our goal is to train a robot policy π that best recovers an unknown policy $\pi ^ { * }$ using a demonstration dataset $\mathcal { D } \ = \ \{ ( s _ { i } , a _ { i } ) \}$ collected by $\pi ^ { * }$ Specifically, the robot policy π operates on a trajectory basis: $\pi ( A _ { t } | S _ { t } )$ , where ${ { S } _ { t } } = \left\{ { { s } _ { t - { T } _ { s } + 1 } } , { { s } _ { t - { T } _ { s } + 2 } } , . . . , { { s } _ { t } } \right\}$ is the given short history state sequence, $A _ { t } = \left\{ a _ { t } , a _ { t + 1 } , . . . , a _ { t + T _ { a } - 1 } \right\}$ is the predicted future actions to take. $T _ { s }$ and $T _ { a }$ represent the lengths of these two sequences respectively.

## B. Diffusion Models

Diffusion models [10], [11], [12] are generative models that iteratively generate samples that match the data distribution. Recent works [19], [20], [21], [22] have demonstrated its great ability in image generation and other data generation tasks. The diffusion process is of the original data being destroyed by a sequence of noise $q ( x ^ { k } | x ^ { k - 1 } )$ known as the forward process, where k is the current diffusion step and there are K steps in total. The diffusion model uses the reverse process, the backward process, $p _ { \theta } ( x ^ { k - 1 } | x ^ { k } )$ to denoise the corrupted data. By iteratively denoising, a diffusion model generates synthesized data xˆ that approximates the original data distribution $q ( x _ { 0 } )$ , starting from a random prior $p ( x ^ { K } )$ ):

$$
p _ {\theta} (x ^ {0}) = \int p _ {\theta} (x ^ {0: K}) d x ^ {1: K}\tag{1}
$$

$$
= \int p (x ^ {K}) \prod_ {k = 1} ^ {K} p _ {\theta} (x ^ {k - 1} | x ^ {k}) d x ^ {1: K}\tag{2}
$$

Typically the random prior $p ( x ^ { K } )$ is a standard Gaussian distribution, and the denoising process is parameterized by the following Gaussian:

$$
p _ {\theta} (x ^ {k - 1} | x ^ {k}) = \mathcal {N} (x ^ {k - 1} | \mu_ {\theta} (x ^ {k}, k), \Sigma^ {k}),\tag{3}
$$

where the parameter $\mu _ { \boldsymbol { \theta } } ( x ^ { k } , k )$ is estimated from a neural network parameterized by θ.

## C. Diffusion Models for Policy Learning

Diffusion models have been applied for data augmentation [23] and sample synthesis [24], [25], [26] in robot policy learning. As recent works [3], [4], [27] formulate the robot policy learning as an action sequence generation problem, diffusion models have also been successfully adapted as sequence generation models [16], [23], [15], [14]. Diffuser [14] concatenates the low-dimensional states and the actions and models multiple state-action pairs as a matrix (image) generation problem. However, such a setting is not feasible for visuomotor policy learning due to the high dimensionality of the visual observations.

Diffusion Policy [17] tackles the challenge by generating only action sequences, while conditioned on visual and other states. Specifically, given a sequence of $T _ { s }$ states $\begin{array} { r l } { S _ { t } } & { { } = } \end{array}$ $\left\{ s _ { t - T _ { s } + 1 } , s _ { t - T _ { s } + 2 } , . . . , s _ { t } \right\}$ where each s contains both visual states and low-dimensional states, the diffusion model generates a sequence of $T _ { a }$ actions $A _ { t } = \left\{ a _ { t } , a _ { t + 1 } , . . . , a _ { t + T _ { a } - 1 } \right\}$ conditioned on the state sequence. When the agent finishes executing $A _ { t }$ , Diffusion Policy generates the consequent action sequence $A _ { t + T _ { c } }$ given $S _ { t + T _ { a } }$ , which formulates a closed-loop control (see Fig. 1).

The model of Diffusion Policy is composed of a state encoder $E _ { S }$ , an action encoder $E _ { A }$ , and an action decoder $D _ { A }$ The action encoder and decoder make up the diffusion model for generating action sequences by running the denoising process iteratively. The state encoder provides conditioning from the states, which modulates the generation process. This paper focuses on the convolutional version of the Diffusion Policy due to its superior performance.

![](images/ee0cd6f6ba880a9d013d487294d70ee72ed4ca8138577d124a1d0deb6c5df949.jpg)  
Fig. 1: Trajectory generation formulation of Diffusion Policy. This figure shows a case where $T _ { s } = 2$ and $T _ { a } = 4 .$

Given a state sequence with both visual and lowdimensional states $S _ { t } = \left\{ S _ { t , i m g } , S _ { t , l o w - d i m } \right\}$ , the state encoder extracts visual embeddings from images $h _ { t , i m g } = E _ { S } ( S _ { t , i m g } )$ The visual embeddings $h _ { t , i m g }$ are then concatenated with other low dimensional states $S _ { t , l o w - d i m }$ to form the observation condition $h _ { t } = h _ { t , i m g } \oplus S _ { t }$ <sub>,low-dim</sub>.

The action encoder $E _ { A }$ takes the noisy action sequence $A _ { t } ^ { k }$ at diffusion step k and the observation condition $h _ { t }$ and produces the representation $X _ { t } ^ { k }$ from the deepest layer and other tensors for skip connection $X _ { t , s k i p } ^ { k }$ from the shallower layers: $X _ { t } ^ { k } , X _ { t , s k i p } ^ { k } = E _ { A } ( A _ { t } ^ { k } , h _ { t } , k )$ , where t is the timestamp of a state or action trajectory. $X _ { t } ^ { k } \in \mathbb { R } ^ { T \times C }$ where T is the representation length along the time axis and C is the number of channels.

The action decoder takes both $X _ { t } ^ { k } , X _ { t , s k i p } ^ { k }$ and condition $h _ { t }$ to estimate the noise ϵ which is applied to $\dot { \boldsymbol { A } } _ { t } ^ { k }$ in the forward diffusion process. The condition $h _ { t }$ is applied between two convolutional layers in the residual block, using Feature-wise Linear Modulation (FiLM) [28]. Then a (slightly) denoised action sequence $A _ { t } ^ { k - 1 }$ is derived from the estimated noise $\epsilon _ { \theta }$ and $A _ { t } ^ { k }$ using Eq.4.

$$
\begin{array}{c} \epsilon_ {\theta} = D _ {A} (X _ {t} ^ {k}, X _ {t, s k i p} ^ {k}, h _ {t}) \\ A _ {t} ^ {k - 1} = \frac {1}{\sqrt {\alpha_ {k}}} \left(A _ {t} ^ {k} - \frac {1 - \alpha_ {k}}{\sqrt {1 - \bar {\alpha_ {k}}}} \epsilon_ {\theta}\right) + \sigma_ {k} z \end{array}\tag{4}
$$

where z is randomly sampled from a normal distribution with the same dimension as $A _ { t } ^ { k } . ~ \alpha _ { k } , ~ \hat { \alpha _ { k } }$ , and $\sigma _ { k }$ are parameters regarding the diffusion process used in DDPM [11], except that we use k as the diffusion step instead of t.

During inference, the denoising process mentioned above is repeated for K times iteratively, generating a noiseless action sequence $A _ { t } ^ { 0 }$ in the end. The whole model is optimized using the same Mean Squared Error (MSE) as DDPM [11] to predict the noise ϵ applied to $A _ { t } ^ { 0 }$ for constructing $A _ { t } ^ { k }$

$$
\mathcal {L} _ {D D P M} = \mathrm{MSE} (D _ {S} (E _ {A} (A _ {t} ^ {k}, h _ {t}, k), h _ {t}), \epsilon)\tag{5}
$$

## III. METHOD

Crossway Diffusion extends existing Diffusion Policy [17] by introducing (1) a state decoder and (2) an auxiliary objective, both for reconstructing the input states. The overall architecture of Crossway Diffusion is presented as Fig. 2.

![](images/cdf1a3c4f9db9ef277c69be21c03cc652d013ce7c801a22ba6f0ba13e4868695.jpg)  
Fig. 2: Left: Architecture of Crossway Diffusion. We introduce a state decoder to the existing Diffusion Policy [17] as well as an auxiliary reconstruction objective $\mathcal { L } _ { R e c o n . }$ . The state decoder takes a transformed intermediate representation ‘intersection’ to reconstruct the input states. Right: Transformation applied to ‘intersection’ for the visual state decoder.

Specifically, the state decoder takes the intermediate representation of the diffusion process $X _ { t } ^ { k }$ to reconstruct the input states. The reconstruction objective is jointly optimized with the diffusion loss $\mathcal { L } _ { D D P M }$ during training.

## A. State Decoder

The state decoder $D _ { S }$ is newly introduced to reconstruct the input states from the intermediate representation along with the existing action sequence denoising pipeline. The new procedure is summarized as $\hat { S } _ { t } = D _ { S } \left( g \left( X _ { t } ^ { k } \right) \right)$ , where $\hat { S } _ { t }$ is the reconstructed state and $g ( \cdot )$ is the intersection transformation, discussed in the following section.

The representation $X _ { t } ^ { k }$ is dubbed as intersection since both the flow of action denoising and the flow of state reconstruction pass through this tensor and then head to their corresponding destinations. For the same reason, we name our method Crossway Diffusion, whose key feature is to have two flows of information intersected in the diffusion model during the training phase. By doing so, we explicitly regularize $\bar { X _ { t } ^ { k } }$ to capture the information from both flows, which benefits representation learning. Our design includes two distinct types of state decoders: one for visual states and another for low-dimensional states. For each source of the states (multiple cameras, joint angles, and so on), we assign a dedicated decoder for the best reconstruction results.

To reconstruct the visual states, the visual state decoders are made of a sequence of 2D residual convolutional blocks, transposed convolutional layers for upsampling (ConvTranspose), and vanilla convolutional layers (See Fig. 3). The numbers in the blocks indicate the number of output channels except for ConvTranspose. ConvTranspose doubles the spatial resolution of the input tensor while keeping the number of channels unchanged. Inspired by [29] and [30], we adapt the positional embedding from [31] to present the pixel coordinates. The positional embedding is concatenated to the output of ConvTranspose along the channel axis.

Other low-dimensional states are regressed by three-layer

![](images/4de9ac238cfbb1e3a98161118357cdc798b5ebe69dc2fa0247eb737ba698f300.jpg)  
Fig. 3: Architecture of visual state decoder. Numbers in the blocks indicate the number of output channels except that ConvTranspose doubles the spatial resolution while keeping the number of channels unchanged.

MLPs. The widths of the hidden layers are in the ratios of 4:2:1 compared to the width of the low-dimensional states.

Note that the reconstruction with the state decoder $D _ { S }$ is only used during the training, serving as an ‘interpreter’ to generate additional supervisory signals to train better intermediate representations. The state decoder will be discarded after the training.

## B. Intersection Transformation

Intersection transformation $g ( \cdot )$ adapts the intermediate representation $X _ { t } ^ { k }$ for the state decoder to pursue the reconstruction task. Assume $X _ { t } ^ { k }$ is split along the time axis as a list of vectors $X _ { t } ^ { k } : = \{ X _ { t , i } ^ { \bar { k } } | i = \bar { 1 } , 2 , . . . , \bar { T } \}$ . By default, only the first vector in the list $X _ { t , 1 } ^ { k }$ is sent to the state decoder for state reconstructions.

The transformation for the low-dimensional state decoder $g _ { l o w - d i m } ( \cdot )$ is an identity function which means the state decoder directly takes $X _ { t , 1 } ^ { k }$ as the input.

In contrast, the visual state decoder employs another transformation $g _ { i m g } ( \cdot )$ due to the significant dimensional disparity between $X _ { t , 1 } ^ { k }$ and the visual states. As shown in the right of Fig. $2 .$ to bridge the dimensionality gap, the $C$ elements of $X _ { t , 1 } ^ { k }$ are equally split into 4 folds and then reshaped as a $C / 4 \times 2 \times 2$ block $B .$ . The block B is then repeated multiple times in two spatial dimensions so that it will have a spatial resolution of a quarter of the desired reconstructed image along each spatial dimension. Finally, we concatenate the same positional embeddings introduced in the previous section along the channel axis before sending the transformed intersection to the visual state decoder.

We further explore the effect of multiple designs of $g ( \cdot )$ in Section IV-C.

## C. Crossway Diffusion Loss

Given the state decoder and the state reconstruction task, we introduce a reconstruction loss $\mathcal { L } _ { R e c o n . }$ <sub>.</sub> between the reconstructed states and the original input states, which provides an auxiliary training signal to $D _ { S } , E _ { S } .$ , and $E _ { A } . \ { \mathcal { L } } _ { R e c o n . } \ $ is simply implemented as a Mean Squared Error (MSE).

In addition to $\mathcal { L } _ { D D P M }$ used in Diffusion Policy [17] (Eq. 5), which supervises $D _ { A } , \ E _ { A }$ and $E _ { S } , ~ { \mathcal { L } } _ { R e c o n . }$ is jointly optimized with $\mathcal { L } _ { D D P M }$ by a simple weighted addition. That ${ \mathrm { i s } } ,$ all network modules $E _ { A } , D _ { A } , E _ { S }$ , and $E _ { S }$ are trained jointly. The total loss for Crossway Diffusion is denoted as $\mathrm { E q } . 6$ and we find $\alpha = 0 . 1$ is a generally good setting without extensive hyperparameter search.

$$
\begin{array}{c} \mathcal {L} _ {R e c o n.} = \mathrm{MSE} (S _ {t}, \hat {S} _ {t}) \\ \mathcal {L} _ {C r o s s w a y} = \mathcal {L} _ {D D P M} + \alpha \mathcal {L} _ {R e c o n.} \end{array}\tag{6}
$$

Tool Hang

Duck Collect  
![](images/4c93b480a1b68e98f30ca58cb647cddc1a8b963252eee9b31c7aad21d84d97ca.jpg)

## IV. EXPERIMENT

We first evaluate Crossway Diffusion on multiple challenging simulated tasks and two real-world tasks. Then, we explore variants of Crossway Diffusion regarding state decoder design and the auxiliary objectives. Through an extensive amount of experiments, we confirm that Crossway Diffusion consistently leads to better performance than vanilla Diffusion Policy [17] and other baselines [32] over all the tested tasks.

## A. Task and Dataset

We follow [17] and choose five tasks Can, Lift, Square, Transport, and Tool Hang from Robomimic [18] and Push-T from Implicit Behaviour Cloning (IBC) [32]. In addition, we build a real-world robot arm manipulation environment and collect our own data for two tasks.

In the ‘Can’ task, the robot needs to lift a soda can from one box and put it into another box. In the ‘Lift’ task, the robot needs to lift a cube above a certain height. In the ‘Square’ task, the robot needs to fit the square nut onto the square peg. The ‘Transport’ task entails the collaborative effort of two robot arms to transfer a hammer from a closed container on one table to a bin on another table. One arm is responsible for retrieving and passing the hammer, while the other arm cleans the bin and receives the passed hammer. In ‘Tool Hang’, the robot needs to insert the hook into the base to assemble a frame and then hang a wrench on the hook. Additionally, the ‘Push-T’ task involves pushing a T-shaped block (gray) onto a target location (green) in a 2D space. We investigate both two types of datasets for all tasks if available: ‘ph’ - proficient-human demonstration and ‘mh’ - multi-human demonstrations, originally defined by Robomimic [18]. ‘mh’ is designed to have diverse proficiency on the task compared to ‘ph’.

In the real-world environment, we set up a robot arm with a gripper and two RGB webcams. The first camera is stationary and offers a third-person perspective of the operating space, while the second camera is mounted on the gripper, providing a first-person view for grasping. The action space encompasses both the 3D position of the robot arm and a binary signal for the gripper’s opening and closing. For task ‘Duck Lift’, the objective is to lift a rubber duck above a certain height. For task ‘Duck Collect’, the robot needs to collect four ducks and sort them into two separate containers based on the colors of the ducks.

A visual reference for all the tasks is provided as Fig. 4 and an information summary is presented in Tab. I.

## B. Main Results

1) Evaluation Metrics: Consistent with prior studies, for the Push-T task, we measure the extent of target location coverage achieved by the T block, which is the ratio of the covered area to the total area. We adopt the success rate of other tasks as a performance metric. All the models are trained for 500 epochs and evaluated at the end of training. Note that we train and evaluate all methods from scratch with our setting for a fair comparison, as the numbers from

Push-T

Duck Lift

Fig. 4: Visual reference for all tasks

TABLE I: Dataset summary. ph: the number of proficienthuman demonstrations; mh: the number of multi-human demonstrations; R?: whether the dataset is a real-world dataset or not; Rob.: the number of robots; Obj.: the number of objects; Cam.: the number of cameras; Act-D: action dimension; Steps: max number of rollout steps.

<table><tr><td>Task</td><td>ph</td><td>mh</td><td>R?</td><td>Rob.</td><td>Obj.</td><td>Cam.</td><td>Act-D</td><td>Steps</td></tr><tr><td>Can</td><td>200</td><td>300</td><td>N</td><td>1</td><td>1</td><td>2</td><td>7</td><td>400</td></tr><tr><td>Lift</td><td>200</td><td>300</td><td>N</td><td>1</td><td>1</td><td>2</td><td>7</td><td>400</td></tr><tr><td>Square</td><td>200</td><td>300</td><td>N</td><td>1</td><td>1</td><td>2</td><td>7</td><td>400</td></tr><tr><td>Transport</td><td>200</td><td>300</td><td>N</td><td>2</td><td>3</td><td>4</td><td>14</td><td>700</td></tr><tr><td>Tool Hang</td><td>200</td><td>-</td><td>N</td><td>1</td><td>2</td><td>2</td><td>7</td><td>700</td></tr><tr><td>Push-T</td><td>200</td><td>-</td><td>N</td><td>1</td><td>1</td><td>1</td><td>2</td><td>300</td></tr><tr><td>Duck Lift</td><td>100</td><td>-</td><td>Y</td><td>1</td><td>1</td><td>2</td><td>4</td><td>50</td></tr><tr><td>Duck Collect</td><td>100</td><td>-</td><td>Y</td><td>1</td><td>1</td><td>2</td><td>4</td><td>200</td></tr></table>

[17] are not reliable because of the bug in their evaluation code<sup>1</sup>. For all diffusion-based methods, we benchmark the exponential moving average (EMA) version of the model for better stability, as suggested by [11]. More training details are available in Section VI-A.

2) Simulated Experiments: For all the simulated tasks, in Tab. II we report the average performance over 1000 randomly initialized episodes × 3 models trained with random seeds, as well as the standard deviation among random seeds. That is, each score before ± in Tab. II is an average of 3000 episodes, and each number after ± is the standard deviation of the score of 3 random seeds.

From the comparison, the proposed Crossway Diffusion consistently outperforms the baseline Diffusion Policy [17] in all datasets, as well as other baselines. We observe an improvement of 15.7% over the success rate in ‘Transport, mh’, emphasizing the effectiveness of our method when the demonstration data is varied in proficiency. Please refer to Section VI-G and VI-D for our example episodes and visualization of the action generation process.

3) Real-world Experiments: For all tested methods, we run ‘Duck Lift’ for 20 episodes and ‘Duck Collect’ for 10 episodes and measure the success rate. Note that in task ‘Duck Collect’, one episode is considered a success only if the robot successfully collects and sorts all four ducks. For each episode, we place the ducks at random initial positions but keep the positions consistent across tested methods. The results are reported in Tab. III, highlighting the advantages of our method. We also show that our method is robust to distractions like unseen objects, occlusions in one camera, and other distractions (Fig. 5(a-d)). Please refer to Section VI-E and the video for more information.

TABLE II: Scores on simulated datasets. We report the average of 3000 episodes and the standard deviation of 3 seeds

<table><tr><td>Method</td><td>Can, ph</td><td>Can, mh</td><td>Lift, ph</td><td>Lift, mh</td><td>Square, ph</td><td>Square, mh</td><td>Transport, ph</td><td>Transport, mh</td><td>Tool Hang, ph</td><td>Push-T</td></tr><tr><td>LSTM-GMM</td><td> $0.714 \pm 0.247$ </td><td> $0.887 \pm 0.033$ </td><td> $0.978 \pm 0.017$ </td><td> $0.992 \pm 0.001$ </td><td> $0.643 \pm 0.023$ </td><td> $0.491 \pm 0.057$ </td><td> $0.656 \pm 0.049$ </td><td> $0.254 \pm 0.017$ </td><td> $0.460 \pm 0.060$ </td><td> $0.567 \pm 0.013$ </td></tr><tr><td>IBC [32]</td><td> $0.008 \pm 0.006$ </td><td> $0.001 \pm 0.001$ </td><td> $0.709 \pm 0.008$ </td><td> $0.222 \pm 0.112$ </td><td> $0.002 \pm 0.001$ </td><td> $0.000 \pm 0.001$ </td><td> $0.000 \pm 0.000$ </td><td> $0.000 \pm 0.000$ </td><td> $0.000 \pm 0.000$ </td><td> $0.687 \pm 0.031$ </td></tr><tr><td>Diffusion Policy CNN [17]</td><td> $0.992 \pm 0.002$ </td><td> $0.958 \pm 0.003$ </td><td> $\mathbf{1.000} \pm \mathbf{0.000}$ </td><td> $\mathbf{0.998} \pm \mathbf{0.001}$ </td><td> $\mathbf{0.935} \pm \mathbf{0.006}$ </td><td> $0.858 \pm 0.007$ </td><td> $0.859 \pm 0.015$ </td><td> $0.643 \pm 0.004$ </td><td> $0.772 \pm 0.012$ </td><td> $0.819 \pm 0.002$ </td></tr><tr><td>Crossway Diffusion (Ours)</td><td> $\mathbf{0.994} \pm \mathbf{0.002}$ </td><td> $\mathbf{0.965} \pm \mathbf{0.003}$ </td><td> $\mathbf{1.000} \pm \mathbf{0.000}$ </td><td> $\mathbf{0.998} \pm \mathbf{0.000}$ </td><td> $\mathbf{0.935} \pm \mathbf{0.005}$ </td><td> $\mathbf{0.879} \pm \mathbf{0.010}$ </td><td> $\mathbf{0.864} \pm \mathbf{0.016}$ </td><td> $\mathbf{0.800} \pm \mathbf{0.020}$ </td><td> $\mathbf{0.792} \pm \mathbf{0.014}$ </td><td> $\mathbf{0.843} \pm \mathbf{0.020}$ </td></tr></table>

TABLE III: Success rate of real-world tasks

<table><tr><td></td><td>Duck Lift</td><td>Duck Collect</td></tr><tr><td>Diffusion Policy CNN [17]</td><td>0.80</td><td>0.70</td></tr><tr><td>Crossway Diffusion (Ours)</td><td>0.95</td><td>0.80</td></tr></table>

![](images/cd2f345dc9646181dc9994c05a1e639df68a65776af2cc40a7ea92887163a01a.jpg)  
Fig. 5: Duck Lift task under different obstructions. Two rows show the images captured by two cameras respectively. The red arrow in (e) is used to indicate the position of the duck.

Additionally, we provide qualitative results on image reconstruction in Section VI-F, where the quality of reconstructed visual states is surprisingly high.

## C. Ablations

The ablation study mainly focuses on two critical designs in our method: the state decoder and the auxiliary objective.

1) On State Decoder: Experiments in this section are designed to answer (1) which representation for state reconstruction benefits policy learning most, as well as (2) what is the best architecture for the visual state decoder.

First, two more designs (Design B and C) of the intersection transformation presented in Fig. 6 are studied over five simulated datasets, as well as one variant (Design D) that utilizes $h _ { t }$ instead of intersection $X _ { t } ^ { k }$ for reconstruction.

Design A is the default Crossway Diffusion introduced in Section III. In Design B, the first $C / 2$ channels for all vectors in intersection $X _ { t } ^ { k }$ are selected for reconstruction, while the rest is left dedicated to the denoising process. In contrast, Design C takes advantage of all vectors in $X _ { t } ^ { k }$ for the reconstruction. Additionally, for both Design B and $\mathrm { C } ,$ the selected vectors are independently projected by a linear layer to match the target number of channels $C / 4 .$ The latter operations like reshape and repeat in Fig. 2 are kept the same. Finally, in Design D, the state decoder directly takes the output of the state encoder $h _ { t }$ instead of $X _ { t } ^ { k }$ which disentangles the action denoising flow and the state reconstruction flow. A linear layer is also applied to project $h _ { t }$ to the target dimension C/4 in the channel axis.

![](images/332db31b4f8d243d41b83df7f609c18f81bb750c71a8905e62a0fb67450cd0f4.jpg)  
Fig. 6: Selected parts for state reconstruction in three designs. White and gray parts are not used for reconstruction.

TABLE IV: Ablations on intersection transformation.

<table><tr><td></td><td>Square, mh</td><td>Transport, ph</td><td>Transport, mh</td><td>Tool Hang, ph</td><td>Push-T</td></tr><tr><td>A (default)</td><td> $0.879 \pm 0.010$ </td><td> $0.864 \pm 0.016$ </td><td> $0.800 \pm 0.020$ </td><td> $\mathbf{0.792} \pm \mathbf{0.014}$ </td><td> $\mathbf{0.843} \pm \mathbf{0.020}$ </td></tr><tr><td>B</td><td> $\mathbf{0.881} \pm \mathbf{0.017}$ </td><td> $0.882 \pm 0.010$ </td><td> $0.784 \pm 0.025$ </td><td> $0.777 \pm 0.010$ </td><td> $0.835 \pm 0.012$ </td></tr><tr><td>C</td><td> $0.868 \pm 0.006$ </td><td> $\mathbf{0.906} \pm \mathbf{0.012}$ </td><td> $\mathbf{0.814} \pm \mathbf{0.028}$ </td><td> $0.783 \pm 0.005$ </td><td> $0.831 \pm 0.003$ </td></tr><tr><td>D</td><td> $0.873 \pm 0.012$ </td><td> $0.892 \pm 0.002$ </td><td> $0.764 \pm 0.013$ </td><td> $0.790 \pm 0.007$ </td><td> $0.819 \pm 0.015$ </td></tr><tr><td>Diff. [17]</td><td> $0.858 \pm 0.007$ </td><td> $0.859 \pm 0.015$ </td><td> $0.643 \pm 0.004$ </td><td> $0.772 \pm 0.012$ </td><td> $0.819 \pm 0.002$ </td></tr></table>

The results presented in Tab. IV show that Design A, B, and C consistently outperform the baseline Diffusion Policy CNN [17] (Diff. in table). However, Design D achieves similar results to the baseline on Push-T and shows improvements on other datasets. Such observations validate the importance of not only the reconstruction task but also the design which explicitly forces two flows to intersect with each other. Though these designs show diverse advantages over different tasks, we choose Design A as the default due to its computational simplicity.

Then, the variants with different visual state decoder architectures are studied. The default visual state decoder contains two stages where each stage has two residual blocks with a transposed convolution for spatial upsampling (see Fig. 3). Shallower Dec. takes the second stage out and only keeps the first stage. For ViT Dec., the visual state decoder is a two-layer Vision Transformer (ViT) [7] whose token dimension is C/4 and the positional embeddings are directly added to the tokens before the ViT instead of concatenation.

The results on Push-T are presented in Tab. V (more available in Appendix VI-B), showing that all variants of our methods perform better than the baseline Diffusion Policy CNN [17] (Diff. in table). Such observation validates the effectiveness of the reconstruction task and our specific visual state decoder design.

2) On auxiliary objective: By default, both visual states and low-dimensional states are reconstructed in Crossway Diffusion. We first benchmark a simple variant called Visualonly, which reconstructs only the visual states. Results in Tab. V verify the benefit of predicting all the input states.

TABLE V: Ablation results tested on Push-T

<table><tr><td>Default</td><td>Shallower Dec.</td><td>ViT Dec.</td><td>Visual-only</td><td>Diff. [17]</td></tr><tr><td> $0.843 \pm 0.020$ </td><td> $0.822 \pm 0.014$ </td><td> $0.824 \pm 0.008$ </td><td> $0.828 \pm 0.012$ </td><td> $0.819 \pm 0.002$ </td></tr></table>

TABLE VI: Ablations on future prediction, tested on Push-T

<table><tr><td>N=0 (default)</td><td>N=2</td><td>N=4</td><td>N=6</td><td>N=8</td></tr><tr><td>0.843 ± 0.020</td><td>0.818 ± 0.006</td><td>0.827 ± 0.014</td><td>0.817 ± 0.003</td><td>0.803 ± 0.013</td></tr></table>

TABLE VII: Performance of Crossway-CURL, which adopts a contrastive loss as the auxiliary objective

<table><tr><td></td><td>Lift, mh</td><td>Lift, ph</td><td>Square, mh</td><td>Square, ph</td><td>Push-T</td></tr><tr><td>Crossway-CURL</td><td> $0.802 \pm 0.024$ </td><td> $0.678 \pm 0.188$ </td><td> $0.053 \pm 0.025$ </td><td> $0.035 \pm 0.007$ </td><td> $0.518 \pm 0.160$ </td></tr><tr><td>Default</td><td> $\mathbf{0.998} \pm \mathbf{0.000}$ </td><td> $\mathbf{1.000} \pm \mathbf{0.000}$ </td><td> $\mathbf{0.879} \pm \mathbf{0.010}$ </td><td> $\mathbf{0.935} \pm \mathbf{0.005}$ </td><td> $\mathbf{0.843} \pm \mathbf{0.020}$ </td></tr><tr><td>Diff. [17]</td><td> $0.998 \pm 0.001$ </td><td> $1.000 \pm 0.000$ </td><td> $0.858 \pm 0.007$ </td><td> $0.935 \pm 0.006$ </td><td> $0.819 \pm 0.002$ </td></tr></table>

Then we test the variants where the state decoder predicts the state that is N steps ahead of the current state, instead of predicting (reconstructing) the current state as in the default setting (N = 0). From Tab. VI we empirically learn that reconstructing the current state is the most beneficial method while predicting future states may compromise the performance, even worse than the baseline. We further extend the study to multiple challenging datasets and obtain a similar observation included in Appendix VI-B.

Furthermore, we verify the effectiveness of the proposed reconstruction task in comparison to another SSL objective using contrastive learning inspired by CURL [33]. In particular, we independently perform random crop on all images of an observation sequence $S _ { t }$ twice to get two augmented sequences $S _ { t , a 1 }$ and $S _ { t , a 2 } .$ . The model takes $S _ { t , a 1 }$ to produce intersection $X _ { t , a 1 } ^ { k }$ while $S _ { t , a 2 }$ is processed by the exponential moving average (EMA) version of the model to get $X _ { t , a 2 , \operatorname { e m a } } ^ { k } .$ The intuition is that due to the semantic similarity between $S _ { t , a 1 }$ and $S _ { t , a 2 }$ , the intersection $X _ { t , a 1 }$ and $X _ { t , a 2 , \mathrm { { e m a } } }$ should also be similar in a latent space.

We follow CURL [33] and maintain a learnable matrix W. For each batch of b samples, we calculate the similarity matrix $M _ { \mathrm { s i m } }$ between all samples in the same batch using matrix multiplication (first line in $\operatorname { E q . 7 } ;$ , where sg (·) means stop gradients). Then the contrastive loss $\mathcal { L } _ { C U R L }$ is formulated as Eq. 7, where b is the batch size and $\alpha = 0 . 1$ . Finally $\mathcal { L } _ { C U R L }$ is jointly optimized with the diffusion loss L<sub>DDPM</sub> similar to Eq.6. We name such configuration as Crossway-CURL.

$$
\begin{array}{r} M _ {\mathrm{sim}} = X _ {t, a 1} ^ {k} \cdot W \cdot \mathrm{sg} (X _ {t, a 2, \mathrm{ema}} ^ {k}) ^ {T} \\ \mathcal {L} _ {C U R L} = \mathrm{CrossEntropyLoss} (M _ {\mathrm{sim}}, \mathrm{range} (b)) \\ \mathcal {L} _ {C r o s s w a y - C L} = \mathcal {L} _ {D D P M} + \alpha \mathcal {L} _ {C U R L} \end{array}\tag{7}
$$

From Tab. VII, the contrastive learning variant Crossway-CURL yields much worse performance compared to the baseline on multiple tasks, indicating that not all auxiliary SSL losses benefit policy learning. Such observations happen to align with the online reinforcement learning case [34].

## V. RELATED WORK

## A. Behavioral Cloning

Behavioral Cloning (BC) [1], [35], [36] is a straightforward but surprisingly effective way to obtain robot policies.

With pre-collected state-action pairs, BC learns a policy like fitting a dataset [37], with additional techniques like reward labeling / Inverse Reinforcement Learning (IRL) [38], [39], distribution matching [40], [41], and incorporating extra information [42], [43], [44]. Apart from explicitly generating output actions, BC can be done implicitly, where an energybased model is learned to model the action distribution [32]. Implicit BC is found to be effective in real-world robot tasks. BC also boosts some online RL algorithms like $\mathrm { T D } 3 { + } \mathrm { B C }$ [45], DeepMimic [46], and more [47]. Recent Diffusion-based BC is more like an advanced approach for matching the behavior distribution, which potentially helps mitigate the distribution shift problem in BC [48].

## B. Policy Learning as Sequential Modeling

Sequential modeling [3], [4], [5] is the recent direction to solve offline-RL or Imitation Learning problems. The key is to optimize a policy on a trajectory basis from pre-collected experiences, with a reward signal (offline-RL) or without it (imitation learning). To model the trajectory composed by state-action-reward tuples, Transformer [8] is firstly adopted to this problem in the light of success in modeling natural language. In this formulation, state-action-reward tuples are regarded as equal units [3], [4] or with Markovian properties [5], [49] for better long-term modeling. There are works to extend the formulation to online learning [50], hindsight matching [51], and bootstrapping [52]. Recently, diffusion models have also been adopted to this problem [14], [16], [17], on which [53] conducts a comprehensive survey.

## C. Self-supervised Learning

Self-supervised learning (SSL) [54] is used to learn data representations without task labels. SSL is commonly used to (pre-)train task-agnostic foundation models [55], [56], [57], [58], [59], or is used as an auxiliary task together with other learning paradigms [34], [60]. Similarly, SSL has multiple ways to combine with policy learning, and we briefly categorize them into two: pre-training with SSL [61], [62], [63], [64], [65], [66], [67], [68], and policy learning jointly with auxiliary SSL tasks [69], [70], [71], [72], [73], [74], [33], [75], [76], [77], [78], [79], [80], [81], [82], [83], [34]. Studies [34] have shown that different ways to combine policy learning with SSL have different outcomes. In this work, we follow the joint learning style that optimizes diffusion and reconstruction objectives together.

## VI. CONCLUSION

In this paper, we investigate how SSL can be used to improve diffusion-based visual behavioral cloning. We propose Crossway Diffusion, which involves an extra state decoder and a reconstruction auxiliary objective in addition to the existing diffusion objective during training. Compared to the baseline, Crossway Diffusion shows consistent and substantial improvements over multiple challenging tasks including two real-world tasks without additional computation during evaluation. We hope our work inspires further exploration of how to take advantage of the rapidly evolving SSL techniques for better diffusion-based policies.

[1] D. A. Pomerleau, “Alvinn: An autonomous land vehicle in a neural network,” in Advances in Neural Information Processing Systems (NeurIPS), 1988.

[2] A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, J. Dabis, C. Finn, K. Gopalakrishnan, K. Hausman, A. Herzog, J. Hsu, J. Ibarz, B. Ichter, A. Irpan, T. Jackson, S. Jesmonth, N. J. Joshi, R. Julian, D. Kalashnikov, Y. Kuang, I. Leal, K.-H. Lee, S. Levine, Y. Lu, U. Malla, D. Manjunath, I. Mordatch, O. Nachum, C. Parada, J. Peralta, E. Perez, K. Pertsch, J. Quiambao, K. Rao, M. Ryoo, G. Salazar, P. Sanketi, K. Sayed, J. Singh, S. Sontakke, A. Stone, C. Tan, H. Tran, V. Vanhoucke, S. Vega, Q. Vuong, F. Xia, T. Xiao, P. Xu, S. Xu, T. Yu, and B. Zitkovich, “Rt-1: Robotics transformer for real-world control at scale,” Robotics science and systems (RSS), 2023.

[3] L. Chen, K. Lu, A. Rajeswaran, K. Lee, A. Grover, M. Laskin, P. Abbeel, A. Srinivas, and I. Mordatch, “Decision transformer: Reinforcement learning via sequence modeling,” Advances in Neural Information Processing Systems (NeurIPS), vol. 34, pp. 15 084–15 097, 2021.

[4] M. Janner, Q. Li, and S. Levine, “Offline reinforcement learning as one big sequence modeling problem,” in Advances in Neural Information Processing Systems (NeurIPS), Dec. 2021.

[5] J. Shang, X. Li, K. Kahatapitiya, Y.-C. Lee, and M. S. Ryoo, “Starformer: Transformer with state-action-reward representations for robot learning,” IEEE Transactions on Pattern Analysis and Machine Intelligence, pp. 1–16, 2022.

[6] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “Bert: Pre-training of deep bidirectional transformers for language understanding,” arXiv preprint arXiv:1810.04805, 2018.

[7] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly et al., “An image is worth 16x16 words: Transformers for image recognition at scale,” in Proceedings of the International Conference on Learning Representations (ICLR), 2020.

[8] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” in Advances in Neural Information Processing Systems (NeurIPS), Dec. 2017.

[9] A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, X. Chen, K. Choromanski, T. Ding, D. Driess, A. Dubey, C. Finn et al., “Rt-2: Visionlanguage-action models transfer web knowledge to robotic control,” arXiv preprint arXiv:2307.15818, 2023.

[10] J. Sohl-Dickstein, E. Weiss, N. Maheswaranathan, and S. Ganguli, “Deep unsupervised learning using nonequilibrium thermodynamics,” in International Conference on Machine Learning. PMLR, 2015, pp. 2256–2265.

[11] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” Advances in Neural Information Processing Systems (NeurIPS), vol. 33, pp. 6840–6851, 2020.

[12] J. Song, C. Meng, and S. Ermon, “Denoising diffusion implicit models,” in Proceedings of the International Conference on Learning Representations (ICLR), 2021.

[13] R. Burgert, X. Li, A. Leite, K. Ranasinghe, and M. S. Ryoo, “Diffusion illusions: Hiding images in plain sight,” arXiv preprint arXiv:2312.03817, 2023.

[14] M. Janner, Y. Du, J. B. Tenenbaum, and S. Levine, “Planning with diffusion for flexible behavior synthesis,” Proceedings of the International Conference on Machine Learning (ICML), 2022.

[15] Z. Wang, J. J. Hunt, and M. Zhou, “Diffusion policies as an expressive policy class for offline reinforcement learning,” Proceedings of the International Conference on Learning Representations (ICLR), 2023.

[16] T. Pearce, T. Rashid, A. Kanervisto, D. Bignell, M. Sun, R. Georgescu, S. V. Macua, S. Z. Tan, I. Momennejad, K. Hofmann et al., “Imitating human behaviour with diffusion models,” Proceedings of the International Conference on Learning Representations (ICLR), 2023.

[17] C. Chi, S. Feng, Y. Du, Z. Xu, E. Cousineau, B. Burchfiel, and S. Song, “Diffusion policy: Visuomotor policy learning via action diffusion,” Robotics science and systems (RSS), 2023.

[18] A. Mandlekar, D. Xu, J. Wong, S. Nasiriany, C. Wang, R. Kulkarni, L. Fei-Fei, S. Savarese, Y. Zhu, and R. Mart´ın-Mart´ın, “What matters in learning from offline demonstrations for robot manipulation,” Conference on Robot Learning (CoRL), 2021.

[19] A. Ramesh, P. Dhariwal, A. Nichol, C. Chu, and M. Chen, “Hierarchical text-conditional image generation with clip latents,” arXiv preprint arXiv:2204.06125, 2022.

[20] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “High-resolution image synthesis with latent diffusion models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 10 684–10 695.

[21] S. Gu, D. Chen, J. Bao, F. Wen, B. Zhang, D. Chen, L. Yuan, and B. Guo, “Vector quantized diffusion model for text-to-image synthesis,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 10 696–10 706.

[22] R. Burgert, K. Ranasinghe, X. Li, and M. S. Ryoo, “Peekaboo: Text to image diffusion models are zero-shot segmentors,” arXiv preprint arXiv:2211.13224, 2022.

[23] H.-C. Wang, S.-F. Chen, and S.-H. Sun, “Diffusion model-augmented behavioral cloning,” arXiv preprint arXiv:2302.13335, 2023.

[24] C. Lu, P. J. Ball, and J. Parker-Holder, “Synthetic experience replay,” arXiv preprint arXiv:2303.06614, 2023.

[25] T. Yu, T. Xiao, A. Stone, J. Tompson, A. Brohan, S. Wang, J. Singh, C. Tan, J. Peralta, B. Ichter et al., “Scaling robot learning with semantically imagined experience,” Robotics science and systems (RSS), 2023.

[26] Y. Dai, M. Yang, B. Dai, H. Dai, O. Nachum, J. Tenenbaum, D. Schuurmans, and P. Abbeel, “Learning universal policies via textguided video generation,” arXiv preprint arXiv:2302.00111, 2023.

[27] J. Shang, K. Kahatapitiya, X. Li, and M. S. Ryoo, “Starformer: Transformer with state-action-reward representations for visual reinforcement learning,” in Proceedings of the European Conference on Computer Vision (ECCV). Springer, 2022, pp. 462–479.

[28] E. Perez, F. Strub, H. De Vries, V. Dumoulin, and A. Courville, “Film: Visual reasoning with a general conditioning layer,” in Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), vol. 32, 2018.

[29] N. Watters, L. Matthey, C. P. Burgess, and A. Lerchner, “Spatial broadcast decoder: A simple architecture for learning disentangled representations in vaes,” arXiv preprint arXiv:1901.07017, 2019.

[30] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, “Nerf: Representing scenes as neural radiance fields for view synthesis,” in Proceedings of the European Conference on Computer Vision (ECCV). Springer, 2020, pp. 405–421.

[31] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” Advances in Neural Information Processing Systems (NeurIPS), vol. 30, 2017.

[32] P. Florence, C. Lynch, A. Zeng, O. A. Ramirez, A. Wahid, L. Downs, A. Wong, J. Lee, I. Mordatch, and J. Tompson, “Implicit behavioral cloning,” in Conference on Robot Learning (CoRL). PMLR, 2022, pp. 158–168.

[33] M. Laskin, A. Srinivas, and P. Abbeel, “Curl: Contrastive unsupervised representations for reinforcement learning,” in Proceedings of the International Conference on Machine Learning (ICML). PMLR, 2020, pp. 5639–5650.

[34] X. Li, J. Shang, S. Das, and M. Ryoo, “Does self-supervised learning really improve reinforcement learning from pixels?” in Advances in Neural Information Processing Systems (NeurIPS), vol. 35, 2022, pp. 30 865–30 881.

[35] L. P. Kaelbling, Learning in embedded systems. MIT press, 1993.

[36] P. Kormushev, S. Calinon, and D. G. Caldwell, “Imitation learning of positional and force skills demonstrated via kinesthetic teaching and haptic input,” Advanced Robotics, vol. 25, no. 5, pp. 581–603, 2011.

[37] C. G. Atkeson and S. Schaal, “Robot learning from demonstration,” in Proceedings of the International Conference on Machine Learning (ICML), vol. 97, 1997, pp. 12–20.

[38] A. Y. Ng and S. Russell, “Algorithms for inverse reinforcement learning,” in Proceedings of the International Conference on Machine Learning (ICML), vol. 1, 2000, p. 2.

[39] T. Osa, J. Pajarinen, G. Neumann, J. A. Bagnell, P. Abbeel, J. Peters et al., “An algorithmic perspective on imitation learning,” Foundations and Trends® in Robotics, vol. 7, no. 1-2, pp. 1–179, 2018.

[40] J. Ho and S. Ermon, “Generative adversarial imitation learning,” Advances in Neural Information Processing Systems (NeurIPS), vol. 29, 2016.

[41] X. B. Peng, Z. Ma, P. Abbeel, S. Levine, and A. Kanazawa, “Amp: Adversarial motion priors for stylized physics-based character control,” ACM Transactions on Graphics (TOG), vol. 40, no. 4, pp. 1–20, 2021.

[42] P. Florence, L. Manuelli, and R. Tedrake, “Self-supervised correspondence in visuomotor policy learning,” IEEE Robotics and Automation Letters, vol. 5, no. 2, pp. 492–499, 2019.

[43] T. Zhang, Z. McCarthy, O. Jow, D. Lee, X. Chen, K. Goldberg, and P. Abbeel, “Deep imitation learning for complex manipulation tasks from virtual reality teleoperation,” in IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2018, pp. 5628–5635.

[44] R. Rahmatizadeh, P. Abolghasemi, L. Bol¨ oni, and S. Levine, “Vision-¨ based multi-task manipulation for inexpensive robots using end-to-end learning from demonstration,” in 2018 IEEE international conference on robotics and automation (ICRA). IEEE, 2018, pp. 3758–3765.

[45] S. Fujimoto and S. S. Gu, “A minimalist approach to offline reinforcement learning,” Advances in Neural Information Processing Systems (NeurIPS), vol. 34, pp. 20 132–20 145, 2021.

[46] X. B. Peng, P. Abbeel, S. Levine, and M. Van de Panne, “Deepmimic: Example-guided deep reinforcement learning of physics-based character skills,” ACM Transactions On Graphics (TOG), vol. 37, no. 4, pp. 1–14, 2018.

[47] A. Rajeswaran, V. Kumar, A. Gupta, G. Vezzani, J. Schulman, E. Todorov, and S. Levine, “Learning complex dexterous manipulation with deep reinforcement learning and demonstrations,” Robotics science and systems (RSS), 2018.

[48] P. Abbeel and A. Y. Ng, “Apprenticeship learning via inverse reinforcement learning,” in Proceedings of the International Conference on Machine Learning (ICML), 2004, p. 1.

[49] S. Hu, L. Shen, Y. Zhang, and D. Tao, “Graph decision transformer,” arXiv preprint arXiv:2303.03747, 2023.

[50] Q. Zheng, A. Zhang, and A. Grover, “Online decision transformer,” in Proceedings of the International Conference on Machine Learning (ICML), 2022.

[51] H. Furuta, Y. Matsuo, and S. S. Gu, “Distributional decision transformer for hindsight information matching,” in Proceedings of the International Conference on Learning Representations (ICLR), 2022.

[52] K. Wang, H. Zhao, X. Luo, K. Ren, W. Zhang, and D. Li, “Bootstrapped transformer for offline reinforcement learning,” in Advances in Neural Information Processing Systems (NeurIPS), 2022.

[53] Z. Zhu, H. Zhao, H. He, Y. Zhong, S. Zhang, Y. Yu, and W. Zhang, “Diffusion models for reinforcement learning: A survey,” arXiv preprint arXiv:2311.01223, 2023.

[54] P. Vincent, H. Larochelle, Y. Bengio, and P.-A. Manzagol, “Extracting and composing robust features with denoising autoencoders,” in Proceedings of the International Conference on Machine Learning (ICML), 2008, pp. 1096–1103.

[55] K. He, X. Chen, S. Xie, Y. Li, P. Dollar, and R. Girshick, “Masked´ autoencoders are scalable vision learners,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 16 000–16 009.

[56] Z. Tong, Y. Song, J. Wang, and L. Wang, “Videomae: Masked autoencoders are data-efficient learners for self-supervised video pre-training,” Advances in Neural Information Processing Systems (NeurIPS), vol. 35, pp. 10 078–10 093, 2022.

[57] L. Wang, B. Huang, Z. Zhao, Z. Tong, Y. He, Y. Wang, Y. Wang, and Y. Qiao, “Videomae v2: Scaling video masked autoencoders with dual masking,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 14 549–14 560.

[58] T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell et al., “Language models are few-shot learners,” Advances in Neural Information Processing Systems (NeurIPS), vol. 33, pp. 1877–1901, 2020.

[59] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, “A simple framework for contrastive learning of visual representations,” in Proceedings of the International Conference on Machine Learning (ICML). PMLR, 2020, pp. 1597–1607.

[60] S. Das, T. Jain, D. Reilly, P. Balaji, S. Karmakar, S. Marjit, X. Li, A. Das, and M. S. Ryoo, “Limited data, unlimited potential: A study on vits augmented by masked autoencoders,” in IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2024, pp. 6878–6888.

[61] D. Ha and J. Schmidhuber, “Recurrent world models facilitate policy evolution,” Advances in Neural Information Processing Systems (NeurIPS), vol. 31, 2018.

[62] P. Sermanet, C. Lynch, Y. Chebotar, J. Hsu, E. Jang, S. Schaal, S. Levine, and G. Brain, “Time-contrastive networks: Self-supervised learning from video,” in IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2018, pp. 1134–1141.

[63] A. Zhan, P. Zhao, L. Pinto, P. Abbeel, and M. Laskin, “A framework for efficient robotic manipulation,” in Advances in Neural Information Processing Systems (NeurIPS), 2021.

[64] R. Shah and V. Kumar, “Rrl: Resnet as representation for reinforcement learning,” in Proceedings of the International Conference on Learning Representations (ICLR), 2021.

[65] A. Stooke, K. Lee, P. Abbeel, and M. Laskin, “Decoupling representation learning from reinforcement learning,” in Proceedings of the International Conference on Machine Learning (ICML). PMLR, 2021, pp. 9870–9879.

[66] J. Shang and M. S. Ryoo, “Self-supervised disentangled representation learning for third-person imitation learning,” in IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2021, pp. 214–221.

[67] C. Wang, X. Luo, K. Ross, and D. Li, “Vrl3: A data-driven framework for visual deep reinforcement learning,” Advances in Neural Information Processing Systems (NeurIPS), 2022.

[68] T. Xiao, I. Radosavovic, T. Darrell, and J. Malik, “Masked visual pretraining for motor control,” arXiv preprint arXiv:2203.06173, 2022.

[69] A. v. d. Oord, Y. Li, and O. Vinyals, “Representation learning with contrastive predictive coding,” arXiv preprint arXiv:1807.03748, 2018.

[70] M. Igl, L. Zintgraf, T. A. Le, F. Wood, and S. Whiteson, “Deep variational reinforcement learning for pomdps,” in Proceedings of the International Conference on Machine Learning (ICML). PMLR, 2018, pp. 2117–2126.

[71] D. Hafner, T. Lillicrap, I. Fischer, R. Villegas, D. Ha, H. Lee, and J. Davidson, “Learning latent dynamics for planning from pixels,” in Proceedings of the International Conference on Machine Learning (ICML). PMLR, 2019, pp. 2555–2565.

[72] P. Yingjun and H. Xinwen, “Learning representations in reinforcement learning: An information bottleneck approach,” arXiv preprint arXiv:1911.05695, 2019.

[73] D. Yarats, A. Zhang, I. Kostrikov, B. Amos, J. Pineau, and R. Fergus, “Improving sample efficiency in model-free reinforcement learning from images,” in Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), 2021, pp. 10 674–10 681.

[74] A. X. Lee, A. Nagabandi, P. Abbeel, and S. Levine, “Stochastic latent actor-critic: Deep reinforcement learning with a latent variable model,” Advances in Neural Information Processing Systems (NeurIPS), vol. 33, pp. 741–752, 2020.

[75] J. Zhu, Y. Xia, L. Wu, J. Deng, W. Zhou, T. Qin, T.-Y. Liu, and H. Li, “Masked contrastive representation learning for reinforcement learning,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 03, pp. 3421–3433, 2023.

[76] B. Mazoure, R. Tachet des Combes, T. L. Doan, P. Bachman, and R. D. Hjelm, “Deep reinforcement and infomax learning,” Advances in Neural Information Processing Systems (NeurIPS), vol. 33, pp. 3686– 3698, 2020.

[77] K.-H. Lee, I. Fischer, A. Liu, Y. Guo, H. Lee, J. Canny, and S. Guadarrama, “Predictive information accelerates learning in rl,” Advances in Neural Information Processing Systems (NeurIPS), vol. 33, pp. 11 890–11 901, 2020.

[78] Z. D. Guo, B. A. Pires, B. Piot, J.-B. Grill, F. Altche, R. Munos, and´ M. G. Azar, “Bootstrap latent-predictive representations for multitask reinforcement learning,” in Proceedings of the International Conference on Machine Learning (ICML). PMLR, 2020, pp. 3875–3886.

[79] M. Schwarzer, A. Anand, R. Goel, R. D. Hjelm, A. Courville, and P. Bachman, “Data-efficient reinforcement learning with momentum predictive representations,” in Proceedings of the International Conference on Learning Representations (ICLR), 2021.

[80] A. Zhang, R. McAllister, R. Calandra, Y. Gal, and S. Levine, “Learning invariant representations for reinforcement learning without reconstruction,” in Proceedings ofthe International Conference on Learning Representations (ICLR), 2021.

[81] T. Yu, C. Lan, W. Zeng, M. Feng, Z. Zhang, and Z. Chen, “Playvirtual: Augmenting cycle-consistent virtual trajectories for reinforcement learning,” Advances in Neural Information Processing Systems (NeurIPS), vol. 34, 2021.

[82] T. D. Kulkarni, A. Gupta, C. Ionescu, S. Borgeaud, M. Reynolds, A. Zisserman, and V. Mnih, “Unsupervised learning of object keypoints for perception and control,” Advances in Neural Information Processing Systems (NeurIPS), vol. 32, 2019.

[83] X. Wang, L. Lian, and S. X. Yu, “Unsupervised visual attention and invariance for reinforcement learning,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021, pp. 6677–6687.

[84] T. Salimans and J. Ho, “Progressive distillation for fast sampling of

diffusion models,” Proceedings of the International Conference on Learning Representations (ICLR), 2022.

[85] Y. Song, P. Dhariwal, M. Chen, and I. Sutskever, “Consistency models,” in Proceedings of the International Conference on Machine Learning (ICML), 2023.

## APPENDIX

## A. Training Details

We primarily adopt the hyperparameters from the Diffusion Policy [17]. Specifically, the observation horizon $T _ { s }$ and action horizon $T _ { a }$ are set at 2 and 8, respectively. The learning rate and weight decay are configured at 1e-4 and 1e-6, with a uniform batch size of 64 across all datasets. Our approach involves 100 diffusion iterations for both the training and inference phases. Additionally, we explore the impact of reducing denoising steps during inference through DDIM [12], as elaborated in Appendix VI-C. Comprehensive details on image reconstruction hyperparameters and the count of learnable parameters are presented in Tab. VIII.

## B. Ablations on More Datasets

We extend the ablation experiments for methods introduced in Section IV-C to additional datasets. The outcomes related to the state decoder and the auxiliary objectives are respectively tabulated in Tab. V and Tab. X. These results validate the superiority of our default configuration across all datasets, with the exception of ‘Transport, ph’. Interestingly, for the group of experiments where the reconstruction goal is the future states instead of the current state (described in Section IV-C.2), we empirically find a correlation between the future prediction step (N) and the score (shown in Fig. 7). It indicates a decline in performance scores as the number of future prediction steps (N) increases. However, for ‘Transport,ph’, the optimal performance is achieved at $N = 6 ,$ , as depicted in Fig. 7.

## C. Discussion on Improving Inference Speed

Even though our self-supervised learning objective improves the performance, it remains the same need for a large number of diffusion iterations during inference as the baseline Diffusion Policy [17]. This is still a common challenge for diffusion-based policies that hinder responsiveness in some high-dynamic real-world environments.

TABLE VIII: Hyperparameters and the number of learnable parameters. ImgRes: the resolution of visual states (Camera views ${ \bf \nabla } \times { \bf \nabla } \times { \bf \nabla } \mathrm { H } ) ;$ CropRes: the image resolution after random crop; RecRes: the resolution of reconstruction target; #D: number of parameters in diffusion network; #VE: number of parameters in state encoder; #VD: number of parameters in state decoder.

<table><tr><td>Task</td><td>ImgRes</td><td>CropRes</td><td>RecRes</td><td>#D</td><td>#VE</td><td>#VD</td></tr><tr><td>Can</td><td>2×84×84</td><td>2×76×76</td><td>2×84×84</td><td>256M</td><td>22M</td><td>6M</td></tr><tr><td>Lift</td><td>2×84×84</td><td>2×76×76</td><td>2×84×84</td><td>256M</td><td>22M</td><td>6M</td></tr><tr><td>Square</td><td>2×84×84</td><td>2×76×76</td><td>2×84×84</td><td>256M</td><td>22M</td><td>6M</td></tr><tr><td>Transport</td><td>4×84×84</td><td>4×76×76</td><td>4×60×60</td><td>264M</td><td>45M</td><td>12M</td></tr><tr><td>Tool Hang</td><td>2×240×240</td><td>2×216×216</td><td>2×80×80</td><td>256M</td><td>22M</td><td>6M</td></tr><tr><td>Push-T</td><td>1×96×96</td><td>1×84×84</td><td>1×96×96</td><td>252M</td><td>11M</td><td>2M</td></tr><tr><td>Duck Lift</td><td>2×160×120</td><td>2×144×108</td><td>2×80×60</td><td>255M</td><td>22M</td><td>6M</td></tr><tr><td>Duck Collect</td><td>2×160×120</td><td>2×144×108</td><td>2×80×60</td><td>255M</td><td>22M</td><td>6M</td></tr></table>

TABLE IX: More ablations on the state decoder

<table><tr><td></td><td>Square, mh</td><td>Transport, ph</td><td>Transport, mh</td><td>Push-T</td></tr><tr><td>Ours (default)</td><td> $\mathbf{0.879} \pm \mathbf{0.010}$ </td><td> $0.864 \pm 0.016$ </td><td> $\mathbf{0.800} \pm \mathbf{0.020}$ </td><td> $\mathbf{0.843} \pm \mathbf{0.020}$ </td></tr><tr><td>Shallower Dec.</td><td> $0.840 \pm 0.021$ </td><td> $\mathbf{0.883} \pm \mathbf{0.002}$ </td><td> $0.699 \pm 0.020$ </td><td> $0.822 \pm 0.014$ </td></tr><tr><td>ViT Dec.</td><td> $0.859 \pm 0.014$ </td><td> $0.865 \pm 0.011$ </td><td> $0.745 \pm 0.025$ </td><td> $0.824 \pm 0.008$ </td></tr><tr><td>Diff. [17]</td><td> $0.858 \pm 0.007$ </td><td> $0.859 \pm 0.015$ </td><td> $0.643 \pm 0.004$ </td><td> $0.819 \pm 0.002$ </td></tr></table>

To this end, we use DDIM [12] to accelerate the inference process as suggested in [17]. Specifically, we evaluate each trained model by varying the number of diffusion iterations from 10 to 100, in increments of 10. Fig. 8 demonstrates the relationship between the score and the number of denoising steps during inference. For each subfigure in 8, given three models per method trained using different random seeds, the only difference is the number of denoising steps during inference shown as the x-axis. In general, the quality of the generated action sequence significantly drops when the number of diffusion iterations gets lower than 40. However, the score of some challenging tasks starts to drop when the number of diffusion iterations is smaller than 60.

We are actively looking into diffusion acceleration techniques like Progressive Distillation [84] and Consistency Models [85] to address the limitation.

## D. Visualization of Action Generation Process

In Fig. 13, we demonstrate an episode with the action generation (denoising) process for Push-T. In the grid, the state transition step t increases from left to right, while the diffusion step k (in other words, noise level) decreases from top to bottom. The orange dot stands for the first action in the sequence and the blue dot stands for the last action. The colors of the actions in between are linear interpolations of the first and the last colors. Diffusion Policy [17] is also listed for comparison as Diff. in the figure. We strongly recommend readers check the videos at https://youtu. be/9deKHueZBuk for a better demonstration of Fig. 13 and more real-world robot experiments.

TABLE X: More ablations on auxiliary objectives

<table><tr><td></td><td>Square, mh</td><td>Transport, ph</td><td>Transport, mh</td><td>Push-T</td></tr><tr><td>Ours (N = 0, default)</td><td> $\mathbf{0.879 \pm 0.010}$ </td><td> $0.864 \pm 0.016$ </td><td> $\mathbf{0.800 \pm 0.020}$ </td><td> $\mathbf{0.843 \pm 0.020}$ </td></tr><tr><td>N = 2</td><td> $0.873 \pm 0.005$ </td><td> $0.879 \pm 0.007$ </td><td> $0.797 \pm 0.015$ </td><td> $0.818 \pm 0.006$ </td></tr><tr><td>N = 4</td><td> $0.856 \pm 0.011$ </td><td> $0.891 \pm 0.007$ </td><td> $0.787 \pm 0.024$ </td><td> $0.827 \pm 0.014$ </td></tr><tr><td>N = 6</td><td> $0.845 \pm 0.012$ </td><td> $\mathbf{0.898 \pm 0.006}$ </td><td> $0.774 \pm 0.008$ </td><td> $0.817 \pm 0.003$ </td></tr><tr><td>N = 8</td><td> $0.844 \pm 0.011$ </td><td> $0.895 \pm 0.022$ </td><td> $0.765 \pm 0.020$ </td><td> $0.803 \pm 0.013$ </td></tr><tr><td>Visual-only</td><td> $0.878 \pm 0.001$ </td><td> $0.869 \pm 0.027$ </td><td> $0.779 \pm 0.018$ </td><td> $0.828 \pm 0.012$ </td></tr><tr><td>Diff. [17]</td><td> $0.858 \pm 0.007$ </td><td> $0.859 \pm 0.015$ </td><td> $0.643 \pm 0.004$ </td><td> $0.819 \pm 0.002$ </td></tr></table>

![](images/6b9f2f5e2b2f1ec7fd50a0bf65754ffc5bebcba14440b27ac0fd8a139ea74efd.jpg)  
Fig. 7: The trend between the score and the number of future prediction steps (N).

![](images/aa470b8699e0c3853d60ee949f0c5fbfb810fd845911f5931957303063312b1b.jpg)  
Fig. 8: The relationship between the score and the diffusion denoising steps during inference using DDIM [12].

## E. Qualitative Test of Real-world Robustness

This section presents the qualitative test results of the robustness of our method under various obstructions.

The examples of ‘Duck Lift’ are shown in Fig. 5. Fig. 5 (a)(b)(c) contain different unseen objects placed on the table as a distraction. None of these obstructions are present in the training dataset. In Fig. 5 (d), the duck is only visible to the second camera. Our method successfully executes the task in all the above scenarios indicating robustness to obstructions. Fig. 5 (e) shows a scenario where the duck is not clearly visible to both cameras. In this case, as expected our method fails to locate and lift the duck.

![](images/ee35964f44b2efd6fa7bb72fb2e2d8108376c4796965b2bfe0b19a843362ad69.jpg)  
Fig. 9: Duck Collect task under different obstructions. Two rows show the images captured by two cameras respectively. (a)(b): Additional ducks. (c): Unseen containers. (d): Unseen duck color. The dots are added to the first row to demonstrate the trajectories generated from the model. The orange dot notes the beginning of an action sequence while the blue dot notes the end.

The examples of ‘Duck Collect’ are shown in Fig. 9. Fig. 9 (a)(b) contain additional ducks beyond the four in the training set. In Fig. 9 (c), we replace the containers with unseen ones. Our method successfully executes the task in all the above scenarios, highlighting the robustness of our method. In Fig. 9 (d), we bring the ducks with an unseen yellow color to the agent, and our method can still pick up the ducks and put them consistently into the same container.

## F. Qualitative Results on Image Reconstruction

In Fig. 10 and Fig. 11, we show multiple pairs of original (left) and reconstructed (right) images randomly selected from the validation set. For simulated environments, Crossway Diffusion provides surprisingly well-reconstructed images. For real-world environments, the reconstructions preserve most of the robot and duck structures, while failing on the details (mouth and eyes) of the duck. However, we note that bad reconstruction, especially on non-task-related details, does not necessarily mean poor performance.

## G. Example Episodes of Crossway Diffusion

In this section, we present example episodes of Crossway Diffusion on different datasets. In Figure 12, for each episode, we sample 10 images with a fixed interval covering the whole episode.

![](images/8d0e3b5be7523da09154c9e008c5549f4aafde0ba78c1cfeb266f8cf1dd78c0d.jpg)  
Fig. 10: Reconstruction results on the validation set. For each dataset, left: original image, right: reconstructed image.

![](images/c1f7322342f0c70f3e2384e2cd6db78b6f58c230de826d3722952babeabfc301.jpg)  
Transport, mh Transport, ph Tool Hang, ph

Fig. 11: Reconstruction results on the validation set. For each dataset, left: original image, right: reconstructed image.

![](images/43d01bfc17b5f007d7a45aafaf56b3b84d8908f4210bb30664490c8b7a277e50.jpg)  
Fig. 12: Example episodes of Crossway Diffusion in simulated environments. The episode length is reported as well.

![](images/40b037b87885dce6b0ef075f803b7f7847546f2c0e7a7a3276d029d13fc0efab.jpg)  
<sub>n</sub> <sub>is</sub> <sub>in</sub> <sub>blue.</sub> W<sup>e</sup> <sup>compare</sup> <sup>our</sup> <sup>method</sup> <sup>with</sup> <sup>Diffusion</sup> <sup>Policy</sup> <sup>[17]</sup> <sub>action</sub> <sub>generation</sub> <sub>process.</sub> <sub>kis</sub> <sub>the</sub> <sub>diffusion</sub> <sub>step</sub> <sub>andtis</sub> <sub>the</sub> <sub>state</sub> <sub>transition</sub> <sub>step.</sub> T<sup>he</sup> <sup>first</sup> <sup>action</sup> <sup>in</sup> <sup>the</sup> <sup>sequ</sup>