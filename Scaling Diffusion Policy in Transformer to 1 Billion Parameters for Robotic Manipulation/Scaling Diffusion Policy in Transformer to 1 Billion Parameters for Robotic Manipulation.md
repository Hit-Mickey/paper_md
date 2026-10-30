# Scaling Diffusion Policy in Transformer to 1 Billion Parameters for Robotic Manipulation

Minjie Zhu<sup>1,∗</sup>, Yichen Zhu<sup>2,∗,†</sup>, Jinming Li<sup>5</sup>, Junjie Wen<sup>1</sup>, Zhiyuan Xu<sup>4</sup>, Ning Liu<sup>2</sup>, Ran Cheng<sup>2</sup>, Chaomin Shen<sup>1,†</sup>, Yaxin Peng<sup>5</sup>, Feifei Feng<sup>2</sup>, Jian Tang<sup>4</sup>

Abstract— Diffusion Policy is a powerful technique tool for learning end-to-end visuomotor robot control. It is expected that Diffusion Policy possesses scalability, a key attribute for deep neural networks, typically suggesting that increasing model size would lead to enhanced performance. However, our observations indicate that Diffusion Policy in transformer architecture (DP-T) struggles to scale effectively; even minor additions of layers can deteriorate training outcomes. To address this issue, we introduce Scalable Diffusion Transformer Policy for visuomotor learning. Our proposed method, namely ScaleDP, introduces two modules that improve the training dynamic of Diffusion Policy and allow the network to better handle multimodal action distribution. First, we identify that DP-T suffers from large gradient issues, making the optimization of Diffusion Policy unstable. To resolve this issue, we factorize the feature embedding of observation into multiple affine layers, and integrate it into the transformer blocks. Additionally, our utilize non-causal attention which allows the policy network to “see” future actions during prediction, helping to reduce compounding errors. We demonstrate that our proposed method successfully scales the Diffusion Policy from 10 million to 1 billion parameters. This new model, named ScaleDP, can effectively scale up the model size with improved performance and generalization. We benchmark ScaleDP across 50 different tasks from MetaWorld and find that our largest ScaleDP outperforms DP-T with an average improvement of 21.6%. Across 7 real-world robot tasks, our ScaleDP demonstrates an average improvement of 36.25% over DP-T on four single-arm tasks and 75% on three bimanual tasks. We believe our work paves the way for scaling up models for visuomotor learning. The project page is available at https://scaling-diffusion-policy.github.io/.

## I. INTRODUCTION

Diffusion models have established leading roles in stateof-the-art advancements across various domains, including image, audio, video, and 3D generation [1], [2], [3], [4], [5], [6], [7], [8]. Specifically, Denoising Diffusion Probabilistic Models (DDPMs) [1] are recognized for their approach of reversing a Stochastic Differential Equation. This technique leverages a stochastic denoising process that gradually incorporates Brownian motion during the generation of outputs. Recently, the power of the diffusion model has manifested in the field of robotics as imitation learning [9], [10]. It has become one of the most popular learning strategies for robotics, stimulating a series of improvements in skill learning, navigation, and visual representation.

The community expects that an effective method should be scalable: as the model size and training data increase, there should be a corresponding improvement in performance and generalization capabilities. This property, namely scaling laws, has driven remarkable progress across machine learning domains like language modeling [11], [12] and computer vision [13], [14], especially the success of large language models. Building a robot model that could achieve the scaling laws is also desirable in the field of robotics. However, whether Diffusion Policy (DP) could scale up, like those transformer models in other domains, has not been explored [15], [16], [17]. Hence, in this work, we study the scalability of the diffusion transformer for visuomotor policy learning.

We begin with the examination of the existing DP in transformer architecture (DP-T). To assess the scalability of DP-T, we conducted a preliminary investigation on MetaWorld [18] (more details in Section III-A). Our evaluation revealed that consistent with the findings in Diffusion Policy [9], where scaling DP-T does not improve performance, regardless of increasing depth or number of heads, increasing in model size could negatively affect the tasks. For example (Figure 1), DP-T with eight layers achieves a success rate of 80.1% in MetaWorld [18]. However, this success rate decreases to 78.4% when the number of layers is increased to twelve and further drops to 74.6% with fourteen layers.

Through further analysis, we find that the failure to scale the transformer architecture stems from unstable training caused by large gradients in the observation fusion module. By replacing the conventional cross-attention fusion approach [15] with multiple affine layers, we are able to normalize parameter distribution [9], [19], which brings good training dynamics to DP-T. To further improve model generalization, we propose to remove masked attention, allowing the model to “see” both past actions and future trajectories. This is particularly beneficial for learning visuomotor policies since trajectory predictions are typically much longer than the trajectories used during testing. For instance, Diffusion Policy predicts actions within ten timesteps but only uses the action at the first timestep. Allowing the model to observe the future trajectory makes it more robust to compound errors during prediction.

To demonstrate the effectiveness of our work, we conduct experiments on 50 simulation tasks in Metaworld and real robot experiments on 7 distinct tasks. We have successfully trained a Scalable Diffusion Transformer Policy (ScaleDP) that demonstrates effective scaling with an increase in model parameters, ranging from 10 million to 1 billion. Both simulation and real-world experiments reveal that ScaleDP significantly outperforms the baseline Diffusion Policy. Additionally, we confirm that as the model size increases, it accommodates more training data, which enhances its performance. Furthermore, we observe improved visual generalization capabilities, as the model scales up.

![](images/eb9694e862426c4aa7d8cf923d3cd19a15694f4ce9465c67c0ff0f4e93d92552.jpg)

![](images/3b3a9dbc22e43807d69dadcaa5d523f4015773331d639beee7284e51c6293cee.jpg)

![](images/1f73d41be73daa926258d7340788ea51f5554ceef8f8cfe336a25052bf7cdf4e.jpg)  
Fig. 1: The motivation of ScaleDP. Left: Increasing the number of heads for Diffusion Policy in Transformer architecture does not necessarily improve performance. Middle: Increase depth could be harmful to the model performance. Right: The visualization of standard deviations of gradient magnitudes (the lower the better, i.e., more balanced optimization paces).

## II. RELATED WORKS

Diffusion Policy in Robotics. Diffusion models are one of the generative models that progressively transform random noise into structured data samples, have achieved remarkable success in generating high-fidelity images [1], [2], [20], [4]. Due to their remarkable expressiveness, diffusion models have recently expanded into the field of robotics. Their applications now extend to various domains such as reinforcement learning [21], [22], [23], [24], [25], [26], [27], [28], imitation learning [9], [10], [29], [30], [31], [32], [33], [34], [35], [13], [36], [37], [38], [39], [35], [40], reward learning [41], [42], [43], grasping [44], [45], [46], and motion planning [47], [48], [44], [49], [50], [51], [52], [53], [54], [55]. In this work, we focus on scaling up the diffusion policy with transformer architectures. We demonstrate that Diffusion Policy in transformer architecture fails to scale up. We show that our proposed method, when scaled up, gain possesses multiple merits that the small transformer model does not have.

## III. METHOD

Problem Setup. We assume an expert collected dataset of demonstrations $\mathcal { D } = \{ \tau _ { 0 } , \tau _ { 1 } , . . . , \tau _ { n } \}$ , where each trajectory $\tau _ { i } = \{ ( o _ { j } , x _ { j } ) \}$ is a sequence of paired raw visual observations o and proprioceptive information x. The proprioceptive information can either be the end-effector pose or joint angles and includes the gripper width. In this work, we use 6D pose, i.e., position $( x , y , z )$ and rotation (roll, pitch, yaw) to control the robot.

Diffusion Policy. Diffusion Policy [9] models the conditional action distribution as a denoising diffusion probabilistic model (DDPM) [1], allowing for better representation of the multi-modality in human-collected demonstrations. Specifically, Diffusion Policy uses DDPM to model the action sequence $p ( A _ { t } \mid o _ { t } , x _ { t } )$ , where $A _ { t } = \{ a _ { t } , \ldots , a _ { t + C } \}$ represents a chunk of the next $C$ actions. The final action is the output of the following denoising process [56]:

$$
A _ {t} ^ {k - 1} = \alpha \left(A _ {t} ^ {k} - \gamma \epsilon_ {\theta} (o _ {t}, x _ {t}, A _ {t} ^ {k}, k)\right) + \mathcal {N} (0, \sigma^ {2} I),\tag{1}
$$

where $A _ { t } ^ { k }$ is the denoised action sequence at time k. Denoising starts from $A _ { t } ^ { K }$ sampled from Gaussian noise and is repeated until $k = 1$ . In Eqn (1), $( \alpha , \gamma , \sigma )$ are the parameters of the denoising process and ϵ is the score function trained using the MSE loss $\ell ( \theta ) = ( \epsilon _ { k } - \epsilon _ { \theta } ( o _ { t } , x _ { t } , A _ { t } ^ { k } + \epsilon _ { k } , k ) ) ^ { 2 }$ The noise at step k of the diffusion process, $\epsilon _ { k } .$ , is sampled from a Gaussian of appropriate variance.

## A. Example of Motivation

To better illustrate the scalability problem of Diffusion Policy, we leverage MetaWorld [18] as our testbed. The experimental results are presented in Figure 1. Our findings indicate that increasing the model size of the vanilla Diffusion Policy in Transformer architecture (DP-T) [9] does not consistently enhance the success rate on tasks in MetaWorld. This observation is consistent with the statement made in the original Diffusion Policy paper [9]. Specifically, there is a noticeable performance boost when the number of heads increases from four to six. However, adding more heads beyond this point results in the average success rate reverting to that of a model with only four heads.

We also assessed the impact of increasing the number of layers within the Transformer model. Our empirical results show a consistent decline in performance with each additional layer. For example, a model with eight layers achieves a success rate above 80%, but this decreases to 78.4% with twelve layers and drops below 75% with fourteen layers.

These findings suggest that the current Diffusion Policy model struggles to scale effectively with respect to model size. This scalability limitation hampers the model’s ability to learn from data, ultimately diminishing its generalization capabilities. We further investigated the training dynamics of DP-T, plotting the standard deviation of gradient magnitudes across different layers. Previous works [57], [58], [59] found that lower values indicate a more balanced optimization pace, which generally leads to better generalization. As illustrated in Figure 1 (right), increasing the depth of DP-T results in larger gradient magnitudes, signaling unstable training in deeper network configurations. This motivated us to modify the neural architecture to address this issue. We demonstrate that ScaleDP maintains low gradient magnitudes even with an increased number of layers.

![](images/73828c3afab36cb3303a8e499494f67a3d2ed85301a9f661279d16668a7d9735.jpg)  
Fig. 2: The architecture of the scalable diffusion transformer policy. Top: Overview of Our ScaleDP. It takes as input multi-view images and outputs a sequence of actions. Bottom: Details of our ScaleDP block. The cross-attention block has the same structure as [15]. The AdaLN block employs adaptive layer norm to fuse conditions into the noise action embeddings, achieving more stable training and better inference performance.

## B. Modification on Neural Architecture

This section gives a detailed illustration of how we modify our neural architecture to ensure ScaleDP could scale up the model size.

Cross-attention block. The traditional approach fuses the conditional information with a cross-attention mechanism [9], [15]. It concatenates the embeddings of timestep k and observation o into a sequence, separate from the action sequence. The transformer block is similar to the original design from [15]. We find that increasing the depth of DP-T results in larger gradient magnitudes, thus making the training procedure more difficult.

Adaptive Layer Norm (AdaLN) block. Following the widespread usage of adaptive normalization layers in image generation [19], [60], [61], we explore replacing standard layer norm layers with adaptive layer norm (AdaLN). Specifically, instead of directly learning dimension-wise scale and shift parameters $\gamma$ and $\beta ,$ we regress them from the sum of the embedding vectors of k and o. Compared with the conditioning mechanism using cross-attention, this enables the model to change the distribution of the noise action embedding according to the conditions. The AdaLN is defined as:

![](images/2c189386046c61f8902a9fd40fe3e6f1f30984965c2d45e944078a461b3de804.jpg)  
Fig. 3: To evaluate the wide scalability of ScaleDP, we conduct experiments on both the Bimanual UR5 Robot Arms and the Franka Arm.

TABLE I: Diverse model size of ScaleDP. We present five model sizes, Tiny (Ti), Small (S), Base (B), Large (L), and Huge (H).

<table><tr><td>Model: ScaleDP</td><td>Layers</td><td>Hidden size d</td><td>Heads</td><td>Param</td></tr><tr><td>Tiny (Ti)</td><td>8</td><td>256</td><td>4</td><td>10M</td></tr><tr><td>Small (S)</td><td>12</td><td>384</td><td>6</td><td>33M</td></tr><tr><td>Base (B)</td><td>12</td><td>768</td><td>12</td><td>130M</td></tr><tr><td>Large (L)</td><td>24</td><td>1024</td><td>16</td><td>457M</td></tr><tr><td>Huge (H)</td><td>32</td><td>1280</td><td>16</td><td>1B</td></tr></table>

$$
\operatorname{AdaLN} (x) = (\gamma (k, o) + 1) \cdot x + \beta (k, o)\tag{2}
$$

where x is the input to the layer normalization, and $\gamma ( k , o )$ and $\beta ( k , o )$ are the adaptive scale and shift parameters regressed from the embedding vectors of k and o.

Non-causal Attention. Following the transformer architecture proposed by [15], the Diffusion Policy utilizes masks to ensure that each action embedding can only attend to previous tokens in the self-attention and cross-attention layers of each transformer decoder block. We argue that this unidirectional attention mechanism would hide the action representations. By removing the mask in self-attention layers, we can make each action more consistent with both left and right actions.

We apply a sequence of N ScaleDP blocks, each operating at the hidden dimension size d. Following ViT, we use standard transformer configs that jointly scale N, d, and attention heads [16]. Specifically, we use five configs: ScaleDP-Ti, ScaleDP-S, ScaleDP-B, ScaleDP-L, and ScaleDP-H. They cover a wide range of model sizes, from 10M parameters to 1B parameters, allowing us to gauge scaling performance. Table I gives details of the configs.

After the final ScaleDP block, we apply the final adaptive layer norm and linear layer to decode the sequence of noise action tokens into the predicted noise.

## IV. EXPERIMENTS

In our experiments, we aim to demonstrate the effectiveness of ScaleDP from the following two perspectives: 1) The

Comparison with Diffusion Policy

![](images/c98c8bc01824cf002c81187921ed62e076392d1cb6b39a23ac3df70a6e81d176.jpg)

![](images/1c28aecace271ac5abaa066ecad35c730bd9b60bcf7e6bd7c51bfdbc94711849.jpg)

![](images/c3ab677c1b64b2fcc59adb2672b046e420d3efb4a51af1a9fca39e95360d95c6.jpg)

![](images/d4bf196e034869c0e97bb74c9b3f2e8c472ce2cad19a5db29c4aad289e8b62b4.jpg)

![](images/c63bcd035f8c41c351c46fcbd0d97ed82936f664bfba5f4e03945bafec0d9de7.jpg)

![](images/42d526277c1a868789951b0ffbb553cb3c66eb70b100a9e0539402457b364ea1.jpg)  
Fig. 4: Experiments. (a) Comparison with Diffusion Policy on MetaWorld; (b) Model scaling results on MetaWorld; (c) and (d): Data scaling results on MetaWorld (Disassemble and Assembly); (e) Model convergence rate on MetaWorld Assembly; (f) Model convergence rate on real-world task (stack cube).

performance compared to Diffusion Policy, the model/data scalability, and rate of convergence; 2) The visual observation appears on the model, including appearance, object, light, and distractor.

## A. Real Robot Experimental Setup

Real robot benchmarks. Our ScaleDP is evaluated across 7 tasks, with 4 tasks using Franka robot with a 7-degreeof-freedom arm and 3 tasks using two UR5 robots with a total of 14-degree-of-freedom arm. We use 2 ZED cameras for Franka and 3 RealSence cameras for bimanual to obtain real-world visual information. Our real robot setup are shown in Figure 3. A brief description of our tasks is following:

Data collection. We acquire our dataset through demonstrations performed by humans. For each target task, we place objects randomly within a designated area and instruct a human to manipulate the objects as smoothly as possible. Additionally, the opening of the mug faces to the left for flip mug task. Throughout this process, we record the RGB streams from two different angles and capture the robot’s state, such as joint positions. ScaleDP employs a mainstream control mode that predicts the 6D, encompassing position (x, y, z) and rotation (roll, pitch, yaw). For every task, we collected 100 trajectories. For the task of closing a laptop, we collected 40 trajectories.

Baselines. As the primary focus of this work is to study the scalable diffusion transformer policy, we select the vanilla Diffusion Policy in transformer architecture (DP-T) [9] as our baseline. We also compare a number of different approaches, including Octo [33], Beso [36], MDT [31], DP-Unet [9] and ACT [62]. These methods cover models with transformer architecture and other variants of Diffusion Policy.

## B. Simulation Experiments

Experimental setup. We classified 50 tasks from Meta-World [18] into levels—easy, medium, hard, and very hard—based on MWM [63]. All experiments were trained with 20 demonstrations and evaluated with 3 seeds, and for each seed, the success rate was averaged over five different iterations.

Comparison with DP-T. We compared the ScaleDP-Ti model with DP-T. Both models have a comparable number of parameters. As shown in Figure 4(a), our approach achieves a higher success rate across all four levels of challenging tasks in MetaWorld. Notice that ScaleDP-Ti has a similar number of parameters as the Diffusion Policy. The superior performance of ScaleDP-Ti across all task levels indicates a more efficient utilization of the model’s capabilities, due to more advanced architecture as we proposed.

Model scaling. In Figure 4(b), we present the results of scaling up the model size while keeping the number of demonstrations constant. The data indicate that as the model size increases, the success rate improves, demonstrating the scalability of our method. This pattern confirms that our approach not only accommodates but thrives of increased computational capacity. The continuous improvement in success rates with larger model sizes, despite the constant number of demonstrations, suggests that the models are effectively extracting more meaningful patterns and insights from the same amount of data.

Data scaling. We further explored whether larger models benefit more from increased data. Figures 4 (c) and (d) show that as the number of demonstrations increases, the success rate for smaller models plateaus, whereas larger models continue to improve. This trend suggests that larger ScaleDP have a higher capacity to leverage additional data, thereby enhancing their learning curves significantly. Moreover, this observation underscores the importance of data scalability when deploying larger models in practical applications.

TABLE II: Success rates on four real-world tasks on single arm Franka robot. Task 1: Close Laptop, Task 2: Flip Mug, Task 3: Stack Cube, Task 4: Place Tennis. It is worth noting that as the model size increases, the average success rate also increases correspondingly, demonstrating the scalability of our model architecture. Each task is evaluated with 20 trials.

<table><tr><td>Model</td><td>Task1</td><td>Task2</td><td>Task3</td><td>Task4</td><td>Avg.</td></tr><tr><td>Octo [33]</td><td>65</td><td>50</td><td>40</td><td>35</td><td>47.50</td></tr><tr><td>Beso [36]</td><td>50</td><td>30</td><td>20</td><td>15</td><td>28.75</td></tr><tr><td>MDT [31]</td><td>55</td><td>45</td><td>50</td><td>30</td><td>45.00</td></tr><tr><td>DP-Unet [9]</td><td>70</td><td>70</td><td>45</td><td>40</td><td>56.25</td></tr><tr><td>ACT [62]</td><td>90</td><td>70</td><td>55</td><td>50</td><td>66.25</td></tr><tr><td>DP-T [9]</td><td>80</td><td>70</td><td>50</td><td>5</td><td>51.25</td></tr><tr><td>ScaleDP-S</td><td>85</td><td>70</td><td>50</td><td>30</td><td>58.75</td></tr><tr><td>ScaleDP-B</td><td>80</td><td>65</td><td>50</td><td>55</td><td>62.50</td></tr><tr><td>ScaleDP-L</td><td>95</td><td>80</td><td>70</td><td>50</td><td>73.75</td></tr><tr><td>ScaleDP-H</td><td>95</td><td>95</td><td>90</td><td>70</td><td>87.50</td></tr></table>

TABLE III: Success rates on three real-world tasks on bimanual UR5 robot. Task 1: Put tennis ball into bag, Task 2: Sweep trash, Task 3: Bimanual Stack Cube. It is worth noting that as the model size increases, the average success rate also increases correspondingly, demonstrating the scalability of our model architecture. Each task is evaluated with 20 trials.

<table><tr><td>Model</td><td>Task1</td><td>Task2</td><td>Task3</td><td>Avg.</td></tr><tr><td>ACT [62]</td><td>100</td><td>70</td><td>50</td><td>73.33</td></tr><tr><td>DP-T [9]</td><td>20</td><td>50</td><td>0</td><td>23.33</td></tr><tr><td>ScaleDP-S</td><td>100</td><td>50</td><td>10</td><td>53.33</td></tr><tr><td>ScaleDP-B</td><td>100</td><td>60</td><td>10</td><td>56.67</td></tr><tr><td>ScaleDP-L</td><td>100</td><td>80</td><td>90</td><td>90.00</td></tr><tr><td>ScaleDP-H</td><td>100</td><td>95</td><td>100</td><td>98.33</td></tr></table>

Learning efficacy. To illustrate the learning efficacy of our model, we plotted the model convergence in Figures 4 (e) and (f). Figure 4 (e) shows the success rate on the MetaWorld Assembly task, while Figure 4 (f) examines the training loss on a real robot task. The results indicate that as training progresses, larger models tend to converge more effectively, achieving higher success rates and lower training losses.

## C. Real Robot Experiments

Main result. Table II and III presents the real robot experimental results. It can be observed that ScaleDP outperforms DP-T across all model sizes in 3 bimanual tasks and 4 singlearm tasks. Notably, for place tennis task, DP-T succeeded only once in 20 trials, whereas our ScaleDP-B/L achieved a success rate of at least 50%. Moreover, as the model size increases, the average success rate improves correspondingly, demonstrating the scalability of our model architecture. Additionally, compared with state-of-the-art imitation learning method, such as ACT, ScaleDP-L outperform its average success rates by 16.67% on 3 bimanul tasks and by 7.5% on 4 single-arm tasks, while ScaleDP-S and ScaleDP-B do not, further highlighting the scalability of our ScaleDP’s architecture. When we increase the model size to 1 billion parameters, ScaleDP-H achieves even better performance across all tasks and experimental settings. It improves the average success rate over ScaleDP-L by 13.75% and 8.33% in two different setups, respectively. These results validate the scalability of our method and highlight the importance of increasing model size in diffusion-based imitation learning. Non-causal attention. To demonstrate the importance of the non-causal attention in ScaleDP, we conducted ablation studies on ScaleDP across 3 bimanual tasks (see Table IV). Our findings indicate that unmasking significantly improves the test performance of all 3 model sizes, particularly for the large model size, which shows a remarkable improvement on Task 3 (Bimanual Stack Cube), achieving a success rate that is 70% higher than that with masking strategy.

TABLE IV: Ablation study on the effectiveness of non-causal attention on real-world tasks. The experiments are conducted on bimanual UR5 robot.

<table><tr><td>Model</td><td>Non-causal</td><td>Task1</td><td>Task2</td><td>Task3</td><td>Avg.</td></tr><tr><td rowspan="2">ScaleDP-S</td><td> $\times$ </td><td>70</td><td>20</td><td>10</td><td>33.33</td></tr><tr><td> $\checkmark$ </td><td>100</td><td>50</td><td>10</td><td> $53.33_{+20}$ </td></tr><tr><td rowspan="2">ScaleDP-B</td><td> $\times$ </td><td>90</td><td>50</td><td>10</td><td>50.00</td></tr><tr><td> $\checkmark$ </td><td>100</td><td>60</td><td>10</td><td> $56.67_{+6.67}$ </td></tr><tr><td rowspan="2">ScaleDP-L</td><td> $\times$ </td><td>100</td><td>80</td><td>20</td><td>66.66</td></tr><tr><td> $\checkmark$ </td><td>100</td><td>80</td><td>90</td><td> $90.00_{+23.34}$ </td></tr></table>

## D. Visual Generalization

Visual generalization refers to the ability to adapt to novel visual textures. Examples of this in robotic manipulation tasks include variations in background color, object texture, or ambient lighting. These visual changes do not alter the fundamental structure of the task, such as the positions of objects and targets, and primarily require the robot to accurately interpret semantic meanings. Here we demonstrate the visual generalization ability of ScaleDP-L. We categorize the visual generalization into the following:

Appearance generalization. We alter the color of the target objects to be grabbed, as demonstrated in Figure 5. Originally, the cube/mug is colored blue/gold; we then make changes accordingly. We observe that ScaleDP-L is able to generalize on objects with different colors. In comparison, the vanilla DP fails to recognize target objects of different colors. Notably, our approach achieves appearance generalization without relying on data augmentation during training. This indicates that the generalization capability of our model stems solely from its ability to recognize the shapes of objects.

![](images/281822cef833efb739f2e0c520b471fc6842cd7e0547c037bcfbf803bd7bd8a3.jpg)  
Fig. 5: Appearance & Object Generalization. We test two tasks: stack cube and flip mug. For appearance generalization, we change the color of the target object to another color. For object generalization, we replace the target object with six objects of varied sizes and shapes from daily life. Each result is evaluated with one trial.

![](images/5120e5e07f4372411881d730dfd8dca51a3a24896ae4ebb3973cdead1ae6bdda.jpg)  
Fig. 6: Distractor & Light Generalization. We test these capabilities on close laptop and flip mug tasks. Each test is evaluated with one trial.

Object generalization. Achieving generalization across diverse objects, which vary in size and shape, presents a significantly greater challenge compared to mere appearance generalization. In Figure 5, we demonstrate that ScaleDP-L effectively manages a wide range of everyday objects. Specifically, when the blue block is replaced with a cup of a completely different shape and the gold mug with a tape, ScaleDP-L still exhibits robust generalization capabilities. This ability to adapt to new and varied object types without a loss in performance underscores the flexibility and practicality of our model, making it highly suitable for dynamic and unpredictable real-world environments where object variability is the norm.

Light generalization. Light generalization is similar to background generalization, which reduces the intensity of the light background in each image compared to the normal one. As demonstrated in Figure 6, ScaleDP-L effectively addresses this generalization problem when the light is slightly decreased. However, it is crucial to acknowledge that while ScaleDP-L can generalize across minor variations in light, significant changes might be more difficult to handle.

Distractor generalization. Distractor generalization refers to introducing additional distractors during the testing phase to evaluate a model’s ability to resist distractions. As shown in Figure 6 (T4), simply adding a mouse to the scene, compared with T3, prevents ScaleDP-L from completing the task accurately. This result is contrast to Diffusion Policy, which is extremely sensitive to the distractor. We observe that the Diffusion Policy tends to target the central points of objects, indicating a deficiency in adapting to new environments. In contrast, our model demonstrates greater robustness to changes in the scene, suggesting superior adaptability.

## V. CONCLUSION

In this study, we explore the transformer architecture within the context of Diffusion Policy. We pinpoint the large gradient of condition fusion as the fundamental chal lenge in transformer architecture. Our proposed architecture facilitates training with increased model sizes up to one billion parameters. We present a preliminary study indicating that incorporating a greater number of parameters into the diffusion transformer policy model enables the emergence of properties not observed in smaller-scale models. Our method presents the first attempt to scale up model size for diffusion-based imitation learning, which we believe will be an important direction for future research.

## VI. ACKNOWLEDGEMENTS

We sincerely thank Yanjie Ze for contributions in discussions and paper review. Our gratitude also thank Yong Wang for his assistance with the hardware setup.

## REFERENCES

[1] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” Advances in neural information processing systems, vol. 33, pp. 6840–6851, 2020.

[2] J. Song, C. Meng, and S. Ermon, “Denoising diffusion implicit models,” arXiv preprint arXiv:2010.02502, 2020.

[3] J. Ho and T. Salimans, “Classifier-free diffusion guidance,” arXiv preprint arXiv:2207.12598, 2022.

[4] C. Saharia, W. Chan, S. Saxena, L. Li, J. Whang, E. L. Denton, K. Ghasemipour, R. Gontijo Lopes, B. Karagol Ayan, T. Salimans et al., “Photorealistic text-to-image diffusion models with deep language understanding,” Advances in neural information processing systems, vol. 35, pp. 36 479–36 494, 2022.

[5] J. Richter, S. Welker, J.-M. Lemercier, B. Lay, and T. Gerkmann, “Speech enhancement and dereverberation with diffusion-based generative models,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2023.

[6] T. Brooks, B. Peebles, C. Holmes, W. DePue, Y. Guo, L. Jing, D. Schnurr, J. Taylor, T. Luhman, E. Luhman, C. Ng, R. Wang, and A. Ramesh, “Video generation models as world simulators,” 2024. [Online]. Available: https: //openai.com/research/video-generation-models-as-world-simulators

[7] J. Ho, W. Chan, C. Saharia, J. Whang, R. Gao, A. Gritsenko, D. P. Kingma, B. Poole, M. Norouzi, D. J. Fleet et al., “Imagen video: High definition video generation with diffusion models,” arXiv preprint arXiv:2210.02303, 2022.

[8] K. Lee, K. Sohn, and J. Shin, “Dreamflow: High-quality textto-3d generation by approximating probability flow,” in The Twelfth International Conference on Learning Representations, 2024. [Online]. Available: https://openreview.net/forum?id=GURqUuTebY

[9] C. Chi, S. Feng, Y. Du, Z. Xu, E. Cousineau, B. Burchfiel, and S. Song, “Diffusion policy: Visuomotor policy learning via action diffusion,” RSS, 2023.

[10] A. Prasad, K. Lin, J. Wu, L. Zhou, and J. Bohg, “Consistency policy: Accelerated visuomotor policies via consistency distillation,” arXiv preprint arXiv:2405.07503, 2024.

[11] H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale et al., “Llama 2: Open foundation and fine-tuned chat models,” arXiv preprint arXiv:2307.09288, 2023.

[12] M. Dehghani, J. Djolonga, B. Mustafa, P. Padlewski, J. Heek, J. Gilmer, A. P. Steiner, M. Caron, R. Geirhos, I. Alabdulmohsin et al., “Scaling vision transformers to 22 billion parameters,” in International Conference on Machine Learning. PMLR, 2023, pp. 7480–7512.

[13] H. Ha, P. Florence, and S. Song, “Scaling up and distilling down: Language-guided robot skill acquisition,” in Conference on Robot Learning. PMLR, 2023, pp. 3766–3777.

[14] X. Zhai, A. Kolesnikov, N. Houlsby, and L. Beyer, “Scaling vision transformers,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 12 104–12 113.

[15] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” Advances in neural information processing systems, vol. 30, 2017.

[16] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, “An image is worth 16x16 words: Transformers for image recognition at scale,” in International Conference on Learning Representations, 2021. [Online]. Available: https://openreview.net/forum?id=YicbFdNTTy

[17] J. Kaplan, S. McCandlish, T. Henighan, T. B. Brown, B. Chess, R. Child, S. Gray, A. Radford, J. Wu, and D. Amodei, “Scaling laws for neural language models,” arXiv preprint arXiv:2001.08361, 2020.

[18] T. Yu, D. Quillen, Z. He, R. Julian, K. Hausman, C. Finn, and S. Levine, “Meta-world: A benchmark and evaluation for multi-task and meta reinforcement learning,” in Conference on robot learning. PMLR, 2020, pp. 1094–1100.

[19] T. Karras, S. Laine, and T. Aila, “A style-based generator architecture for generative adversarial networks,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 4401–4410.

[20] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “High-resolution image synthesis with latent diffusion models,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 10 684–10 695.

[21] L. Yang, Z. Huang, F. Lei, Y. Zhong, Y. Yang, C. Fang, S. Wen, B. Zhou, and Z. Lin, “Policy representation via diffusion probability model for reinforcement learning,” arXiv preprint arXiv:2305.13122, 2023.

[22] B. Mazoure, W. Talbott, M. A. Bautista, D. Hjelm, A. Toshev, and J. Susskind, “Value function estimation using conditional diffusion models for control,” arXiv preprint arXiv:2306.07290, 2023.

[23] J. Brehmer, J. Bose, P. De Haan, and T. S. Cohen, “Edgi: Equivariant diffusion for planning with embodied agents,” Advances in Neural Information Processing Systems, vol. 36, 2024.

[24] S. Venkatraman, S. Khaitan, R. T. Akella, J. Dolan, J. Schneider, and G. Berseth, “Reasoning with latent diffusion in offline reinforcement learning,” arXiv preprint arXiv:2309.06599, 2023.

[25] K. Lee, S. Kim, and J. Choi, “Refining diffusion planner for reliable behavior synthesis by automatic detection of infeasible plans,” Advances in Neural Information Processing Systems, vol. 36, 2024.

[26] S. Zhou, Y. Du, S. Zhang, M. Xu, Y. Shen, W. Xiao, D.-Y. Yeung, and C. Gan, “Adaptive online replanning with diffusion models,” Advances in Neural Information Processing Systems, vol. 36, 2024.

[27] H. Chen, C. Lu, C. Ying, H. Su, and J. Zhu, “Offline reinforcement learning via high-fidelity generative behavior modeling,” arXiv preprint arXiv:2209.14548, 2022.

[28] Y. Ze, N. Hansen, Y. Chen, M. Jain, and X. Wang, “Visual reinforcement learning with self-supervised 3d representations,” IEEE Robotics and Automation Letters, vol. 8, no. 5, pp. 2890–2897, 2023.

[29] A. Khazatsky, K. Pertsch, S. Nair, A. Balakrishna, S. Dasari, S. Karamcheti, S. Nasiriany, M. K. Srirama, L. Y. Chen, K. Ellis et al., “Droid: A large-scale in-the-wild robot manipulation dataset,” arXiv preprint arXiv:2403.12945, 2024.

[30] V. Vosylius, Y. Seo, J. Uruc¸, and S. James, “Render and diffuse: Aligning image and action spaces for diffusion-based behaviour cloning,” arXiv preprint arXiv:2405.18196, 2024.

[31] M. Reuss, O. E. Ya<sup>¨</sup> gmurlu, F. Wenzel, and R. Lioutikov, “Multimodal˘ diffusion transformer: Learning versatile behavior from multimodal goals,” in First Workshop on Vision-Language Models for Navigation and Manipulation at ICRA 2024.

[32] Y. Ze, G. Zhang, K. Zhang, C. Hu, M. Wang, and H. Xu, “3d diffusion policy: Generalizable visuomotor policy learning via simple 3d representations,” in ICRA 2024 Workshop on 3D Visual Representations for Robot Manipulation, 2024.

[33] O. M. Team, D. Ghosh, H. Walke, K. Pertsch, K. Black, O. Mees, S. Dasari, J. Hejna, T. Kreiman, C. Xu et al., “Octo: An open-source generalist robot policy,” arXiv preprint arXiv:2405.12213, 2024.

[34] Z. Xian, N. Gkanatsios, T. Gervet, and K. Fragkiadaki, “Unifying diffusion models with action detection transformers for multi-task robotic manipulation,” in 7th Annual Conference on Robot Learning, 2023.

[35] Y. Ze, G. Yan, Y.-H. Wu, A. Macaluso, Y. Ge, J. Ye, N. Hansen, L. E. Li, and X. Wang, “Gnfactor: Multi-task real robot learning with generalizable neural feature fields,” in Conference on Robot Learning. PMLR, 2023, pp. 284–301.

[36] M. Reuss, M. Li, X. Jia, and R. Lioutikov, “Goal-conditioned imitation learning using score-based diffusion policies,” arXiv preprint arXiv:2304.02532, 2023.

[37] T. Pearce, T. Rashid, A. Kanervisto, D. Bignell, M. Sun, R. Georgescu, S. V. Macua, S. Z. Tan, I. Momennejad, K. Hofmann et al., “Imitating human behaviour with diffusion models,” arXiv preprint arXiv:2301.10677, 2023.

[38] Y. Ze, N. Hansen, Y. Chen, M. Jain, and X. Wang, “Visual reinforcement learning with self-supervised 3d representations,” IEEE Robotics and Automation Letters, vol. 8, no. 5, pp. 2890–2897, 2023.

[39] Y. Zhu, Z. Ou, X. Mou, and J. Tang, “Retrieval-augmented embodied agents,” 2024.

[40] S. Yang, Y. Ze, and H. Xu, “Movie: Visual model-based policy adaptation for view generalization,” Advances in Neural Information Processing Systems, vol. 36, 2024.

[41] M. Psenka, A. Escontrela, P. Abbeel, and Y. Ma, “Learning a diffusion model policy from rewards via q-score matching,” arXiv preprint arXiv:2312.11752, 2023.

[63] Y. Seo, D. Hafner, H. Liu, F. Liu, S. James, K. Lee, and P. Abbeel, “Masked world models for visual control,” in Conference on Robot Learning. PMLR, 2023, pp. 1332–1344.

[42] T. Huang, G. Jiang, Y. Ze, and H. Xu, “Diffusion reward: Learning rewards via conditional video diffusion,” arXiv preprint arXiv:2312.14134, 2023.

[43] Y. Ze, Y. Liu, R. Shi, J. Qin, Z. Yuan, J. Wang, and H. Xu, “H-index: Visual reinforcement learning with hand-informed representations for dexterous manipulation,” Advances in Neural Information Processing Systems, vol. 36, 2024.

[44] J. Urain, N. Funk, J. Peters, and G. Chalvatzaki, “Se (3)- diffusionfields: Learning smooth cost functions for joint grasp and motion optimization through diffusion,” in 2023 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2023, pp. 5923–5930.

[45] A. Simeonov, A. Goyal, L. Manuelli, L. Yen-Chen, A. Sarmiento, A. Rodriguez, P. Agrawal, and D. Fox, “Shelving, stacking, hanging: Relational pose diffusion for multi-modal rearrangement,” arXiv preprint arXiv:2307.04751, 2023.

[46] T. Wu, M. Wu, J. Zhang, Y. Gan, and H. Dong, “Learning score-based grasping primitive for human-assisting dexterous grasping,” Advances in Neural Information Processing Systems, vol. 36, 2024.

[47] J. Wen, Y. Zhu, J. Li, M. Zhu, K. Wu, Z. Xu, R. Cheng, C. Shen, Y. Peng, F. Feng et al., “Tinyvla: Towards fast, data-efficient visionlanguage-action models for robotic manipulation,” arXiv preprint arXiv:2409.12514, 2024.

[48] Z. Zhong, D. Rempe, Y. Chen, B. Ivanovic, Y. Cao, D. Xu, M. Pavone, and B. Ray, “Language-guided traffic simulation via scene-level diffusion,” in Conference on Robot Learning. PMLR, 2023, pp. 144–177.

[49] X. Liu, Y. Zhu, J. Gu, Y. Lan, C. Yang, and Y. Qiao, “Mm-safetybench: A benchmark for safety evaluation of multimodal large language models,” arXiv preprint arXiv:2311.17600, 2023.

[50] Y. Zhu, M. Zhu, N. Liu, Z. Ou, X. Mou, and J. Tang, “Llava-phi: Efficient multi-modal assistant with small language model,” arXiv preprint arXiv:2401.02330, 2024.

[51] W. Liu, T. Hermans, S. Chernova, and C. Paxton, “Structdiffusion: Object-centric diffusion for semantic rearrangement of novel objects,” in Workshop on Language and Robotics at CoRL 2022, 2022.

[52] K. Saha, V. Mandadi, J. Reddy, A. Srikanth, A. Agarwal, B. Sen, A. Singh, and M. Krishna, “Edmp: Ensemble-of-costs-guided diffusion for motion planning,” arXiv preprint arXiv:2309.11414, 2023.

[53] J. Chang, H. Ryu, J. Kim, S. Yoo, J. Seo, N. Prakash, J. Choi, and R. Horowitz, “Denoising heat-inspired diffusion with insulators for collision free motion planning,” arXiv preprint arXiv:2310.12609, 2023.

[54] J. Wen, Y. Zhu, M. Zhu, J. Li, Z. Xu et al., “Object-centric instruction augmentation for robotic manipulation,” 2024.

[55] M. Zhu, Y. Zhu, J. Li, J. Wen, Z. Xu et al., “Language-conditioned robotic manipulation with fast and slow thinking,” 2024.

[56] M. Welling and Y. W. Teh, “Bayesian learning via stochastic gradient langevin dynamics,” in Proceedings of the 28th international conference on machine learning (ICML-11). Citeseer, 2011, pp. 681–688.

[57] Y. Zhao, H. Zhang, and X. Hu, “Penalizing gradient norm for efficiently improving generalization in deep learning,” in International Conference on Machine Learning. PMLR, 2022, pp. 26 982–26 992.

[58] Y. Zhou, B. Karimi, J. Yu, Z. Xu, and P. Li, “Towards better generalization of adaptive gradient methods,” Advances in Neural Information Processing Systems, vol. 33, pp. 810–821, 2020.

[59] A. Akbari, M. Awais, M. Bashar, and J. Kittler, “How does loss function affect generalization performance of deep learning? application to human age estimation,” in Proceedings of the 38th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, M. Meila and T. Zhang, Eds., vol. 139. PMLR, 18–24 Jul 2021, pp. 141–151. [Online]. Available: https://proceedings.mlr.press/v139/akbari21a.html

[60] W. Peebles and S. Xie, “Scalable diffusion models with transformers,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 4195–4205.

[61] A. Brock, J. Donahue, and K. Simonyan, “Large scale gan training for high fidelity natural image synthesis,” arXiv preprint arXiv:1809.11096, 2018.

[62] T. Z. Zhao, V. Kumar, S. Levine, and C. Finn, “Learning fine-grained bimanual manipulation with low-cost hardware,” arXiv preprint arXiv:2304.13705, 2023.