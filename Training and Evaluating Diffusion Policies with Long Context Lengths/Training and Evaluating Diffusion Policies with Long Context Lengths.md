# Training and Evaluating Diffusion Policies with Long Context Lengths

Abhinav Agarwal<sup>∗</sup> Adam Wei<sup>†</sup> Taylan Kargin<sup>†</sup> Michael Zeng Cole Becker Arif Kerem Dayı Pablo Parrilo Asuman Ozdaglar Russ Tedrake Massachusetts Institute of Technology https://dp-with-long-context.github.io/

Abstract: Imitation learning has enabled highly-dexterous robotic manipulation from RGB observations. Policies trained with these methods, however, typically condition robot actions on only a short history of observations. These policies cannot solve tasks that require memory and can get stuck repeatedly executing the same failing motions. In this work, we first benchmark policy performance as context length is incrementally increased from short to long, across a spectrum of tasks with varying local stability and memory requirements, and in multiple data regimes. To our knowledge, this is the first study to investigate context length for Diffusion Policies at this level of detail. Our results challenge prior claims: naively scaling context length is not as brittle as advertised in literature. With an appropriate conditioning method and denoising backbone (UNet+Cross-Attention), single-task policies achieve high success rates on many tasks in the usual data regime even with naive scaling. Next, we propose a training algorithm to jointly train policies at multiple context lengths, further reducing the sample complexity of long-context learning. Finally, we apply our findings to re-evaluate some previously proposed solutions to long-context imitation learning.

## 1 Introduction

![](images/a6ba29451897428d44f4862a05fc82847efed2fc7e31a636f691c512f073784c.jpg)  
Figure 1: Left: We provide extensive evaluation of key interdependent factors when naively scaling context length for Diffusion Policies [1] by varying data scale, conditioning architecture, and task properties and find that naive scaling doesn’t catastrophically fail in many cases, especially with the right architecture. Right: We propose training policies on a curriculum of context lengths as opposed to just one fixed context length, and find that on most tasks we evaluate, this method improves performance in the low data regime while maintaining performance in higher data regimes.

Imitation learning has been at the forefront of rapid advancement in generalizable dexterous manipulation of highly complicated, long-horizon, and difficult to model tasks [1, 2, 3, 4, 5]. It is fascinating how these policies perform remarkably well despite conditioning actions on just a truncated history of observations (typically no more than current and a recent past frame). Intuitively, short-context policies can suffer from a number of pitfalls. For instance, such truncated context can make it difficult to recover task-relevant state or infer temporal quantities such as motion phase, and can lead to repeated execution of a failing strategy rather than corrective behavior.

Despite these drawbacks, imitation learning policies with short context lengths are considered the state-of-the-art in robotic manipulation. A major reason for this is the observed difficulty of training long-context policies. Recent work has explored extending policy memory through auxiliary losses [6], language-guided retrieval mechanisms [7, 8], and VLM-based frame selection [9]. While demonstrating improvement, some of these approaches rely on heuristic compression and filtering ofobservation history prior to passing observations to the policy. A frequently stated reason for such design choices is the brittleness of naively scaling observation history [7, 6].

We ask whether such a conclusion about naive scaling is premature. While recent works have proposed solutions to long-context learning, a detailed investigation into key factors impacting longcontext policy learning is largely absent from literature. Mark et al. [9] provide limited ablations on data scaling and action chunking and agree with us that naive scaling could succeed. However, detailed experiments investigating the interaction between important design choices and establishing how these choices impact naive history scaling remain necessary.

Motivated by this gap in literature, we ask: how do the key factors impacting robotic imitation learning such as task properties, data scale, and model architecture impact policy performance as context length is increased from short to long? We answer this question empirically for state-of-theart Diffusion Policy [1] method, showing noticeable trends: with the right conditioning architecture and data scale, naively scaling history does not show catastrophic drop in performance. Moreover, scaling data substantially mitigates the long-context learning problem, with the sample complexit needed dictated by difficulty of the manipulation primitive and model architecture used.

Simultaneously, we recognize the need to provide solutions for cases where task specific data might be limited. Therefore, through our empirical investigation we develop insights into why long-context policiesfail when they do. Building on that, we present a new algorithm to jointly train policies over a curriculum of context lengths.

## 1.1 Statement of Contributions

In this work, we make the following contributions:

1. A systematic study of interdependent factors in long-context learning: In Section 3, we show comparative policy evaluation on 5 tasks differing in manipulation stability and context length requirements, at 3 data scales per task, and at multiple context lengths. In Section 4.1, we find that with an appropriate policy architecture (UNet+Cross-Attention) and data scale, naively scaling context length is not too brittle. We show examples of tasks which exhibit consistent or improving performance through simple architectural changes as context length is increased.

2. A new training recipe for the limited-data regime: In Section 4.2, we propose a new training algorithm to jointly train a policy on a curriculum of context lengths as opposed to one long or short context length. Depending on task, we show 1.25x-2x improvement over strongest naive scaling baseline in the low data regime and maintained performance in other data regimes.

3. Revisiting Past-Token Prediction [6]: In Section 4.3, we apply our findings and further investigations to compare our work against ideas presented by Torne et al. [6] and revisit the underlying mechanism of past-action prediction as an auxiliary loss to help with long-context learning.

Overall, we present comprehensive takeaways for design choices that strongly impact long-context imitation learning, derived by training and evaluating close to 200 policies across various environments, data scales, context lengths, model architectures, and learning algorithms.

## 1.2 Related Works

We review some key related works here, with additional references presented in Appendix A.

Long Context Lengths for Policies Using Diffusion and Flow: Torne et al. [6] argue predicting past actions provides temporal consistency. They freeze the observation encoder to save GPU memory and do not attribute success of their policies to this. In our reproduction of their work, however, we note that freezing the observation encoder is contributing to success (Section 4.3). More recently, the use of VLMs has been explored for contextual summary/filtering [7, 9]. While showing improvements, these works rely on language annotations and assume that the most recent observation is sufficient for learning the fine manipulation skills, an assumption that can be questioned given the use of short (as opposed to just most recent) context lengths in some landmark works [1, 3]. Dai et al. [10] benchmark policy performance on tasks with different memory requirements. While they study memory augmentations built over short-context policies with multi-task pre-training, we focus on design choices relevant to long context lengths in state-of-the-art Diffusion Policies [1].

Long Context Lengths for Other Policies: Information trace overlay [11] and explicit distinction between perceptual and cognitive tokens [12] have been explored to expand OpenVLA [13] memory. Guhur et al. [14] suggest the use of expressive transformer models to use full task history but primarily demonstrate success on relatively short horizon tasks. Other prior works have attributed failure of context length longer than one to causal confusion [15, 16]. Rigorous work to establish how well this phenomenon holds for Diffusion Policy remains to be done.

## 2 Preliminaries

## 2.1 Diffusion Policy

Diffusion Policies [1], a state-of-the-art imitation learning method, learn a denoiser to sample past and future robot actions conditioned on a history of past observations to accomplish a desired task

$$
\mathbb {P} \left(\mathbf {a} \left[ t - T _ {p} ^ {\text { past }} + 1: t + T _ {p} \right] \mid \mathbf {o} \left[ t - T _ {o} + 1: t \right]\right)\tag{1}
$$

by minimizing the denoising loss in [17]. t is the current timestep, $T _ { p }$ is the prediction horizon, $T _ { o }$ is the context length, o denotes the observations (often images and proprioception), and a denotes the actions. Chi et al. [1] suggest predicting a chunk comprising of all past actions and the desired future actions $( T _ { p } ^ { p a s t } = T _ { o } )$ . Further details about diffusion model training are in Appendix B. Our objective in this work is to a) comment on the impact of key factors as $T _ { o }$ is increased from short to long and b) propose solutions for extending $T _ { o }$ to as long as 80. We focus on scaling relatively recent and continuous context lengths and not distant memory via language tokens [7, 8].

## 2.2 Diffusion Policy Architecture

As depicted in Figure 1, Diffusion Policy architecture comprises two stages: observation encoder to convert images to embeddings and backbone to denoise clean actions conditioned on embeddings and other tokens (proprioception, diffusion timestep, etc.). We present details for different architectures in Appendix B. Note that majority of single-task models from Diffusion Policy [1] and works building on it [18, 19, 20, 21] use a UNet [22] as the denoising backbone and merge conditioning via FiLM [23]. Torne et al. [6] use DiT implementation of the original work [1].

## 3 Evaluating Long-Context Diffusion Policies

We begin by investigating naive context length scaling under data and task variations. We train Diffusion Policies across a) context lengths ranging from short to long (specific values chosen per task), b) on tasks spanning memory requirements and with different local manipulation stability, and c) at three data scales per task. We present interpretable trends: long-context performance improves with data, and the scale needed for success significantly depends on simplicity of the manipulation primitive used in the tasks we evaluate. Moreover, naive scaling does not always fail.

Throughout this section, we use UNet + Cross Attention architecture. The justification for choosing this as well as commentary on comparison between architectures is provided in Section 4.1. We train diffusion policy for the following two categories of tasks (task details are in Appendix C):

1. Tasks solvable with short context length: We use the version of push-T from [19] and the robomimic square and lift tasks from [20]. These tasks show high success rate with short context lengths [1, 19]. We study the performance at $T _ { o } \in \{ 1 , 2 , 5 , 1 0 , 1 2 , 1 6 , 2 0 \}$

2. Tasks requiring long context length: We introduce two tasks, push-and-return and graspand-return (Section 3.1). We train policies at $T _ { o } \in \{ 4 , 8 , 3 2 , 4 8 , 6 0 , 7 2 , 8 0 \}$ for push-andreturn and at $T _ { o } \in \{ 4 , 8 , 1 6 , 2 0 , 2 4 , 4 8 \}$ for grasp-and-return. We also provide limited experiments on hardware at $T _ { o } = 9 2$ for marshmallows task adapted from Mark et al. [9].

## 3.1 Benchmark Tasks Requiring Long Context Lengths

The following tasks share the same goal structure but differ in the manipulation primitive used:

1. Push-and-Return: A block spawns at one of the possible locations (Figure 2), and the robot must 1) push it to the center, 2) move to a neutral configuration, and 3) push it back to the original spawn location. This task combines locally unstable planar pushing with memory dependent decisions about return location.

2. Grasp-and-Return: Same pattern as described for push-and-return, but the robot is required to achieve the objective via grasping a brick as opposed to pushing. This task combines the locally stable, prehensile manipulation skill of grasping a brick with contextual complexity.

Success is awarded if the policy achieves the stated objective (task success). In addition, we define the following two criteria (see Figure 2):

• Manipulation completion: The robot moves the block to the center and returns it to any of the possible spawn locations. This criterion measures success in manipulation requirements of the task if memory requirement is relaxed.

• Contextual success: The percentage of manipulation completion rollouts which return the block/brick to the correct return location.

![](images/06241cef301d00497d6da89029e0a084f4845468424bef7eaf8a9b68eb8deb91.jpg)  
Figure 2: Schematic of the push/grasp-andreturn tasks with two modes. A block spawns on either side. For task success, the robot should push/grasp the block to the center, move to a neutral location, and then return it to the area it originated in. Relaxing the memory requirement, manipulation completion is awarded for returning the block to either of the spawn locations. We define contextual success = $\frac { \mathrm { t a s k ~ s u c c e s s } } { \mathrm { m a n i p u l a t i o n ~ c o m p l e t i o n } }$ to evaluate the percentage of rollouts showing manipulation completion that are also able to keep track of memory.

Marshmallows task involves scooping marshmallows from a large bowl to a smaller target bowl exactly twice and then hitting a button to indicate task completion. Due to arbitrary number of initial marshmallows in the target bowl, long memory is the only way to keep track of success.

## 3.2 Empirical Results

For each task in simulation, we train at 3 data scales: N/2, N, and 2N. The choice of N is specific to the task: it is the amount of data that would typically be available for training single task policy. See Appendix C. Overall, we find that naively scaling context length is not as brittle as might have been indicated in prior work [6, 7]. Performance at longer context lengths is task dependent, with interpretable trends.

Tasks solvable with short-context lengths: Figure 3 shows policy performance as context length is increased at different data scales for push-T, square, and lift. Regardless of data scale, lift shows no drop in performance. On the contrary, square and push-T performance decreases beyond short context lengths in the low data regime $( N / 2 )$ . The magnitude of drop, however, reduces as more data is added, with square showing statistically insignificant drop in the high data regime. These results give important insights: a) performance drop from short to long context lengths is highly task and data dependent b) with this architecture, the drop in performance for these tasks is minimal in the high data regime (in Section 4.1 we show that the performance can drop for other architectures).

![](images/e35e633d85296df0915b41c998cfbbcb79c818a88f31c593163c0806ac211232.jpg)  
(a)

![](images/f5a570f0023f77c85a72b66e21ceadb1eba06ae0ed23907799784cb0e767fc83.jpg)  
(b)

![](images/6820df824d4d28076a8dd82efbffe81db2ad5b45ed3fc10cf9686b38448d6880.jpg)  
(c)  
Figure 3: Short-context solvable tasks’ performance. Long-context policy performance improves as data is scaled. The difference between short and long-context policies shrinks in high data regime.

![](images/60db8add8dd9819110b12d1cbf9159b80ec13b3a1d0f069f917f5feffb091cfd.jpg)  
(a) Even though the task requires memory, long-context policies perform worse than short context policies in the limited (N/2) data regime. In the N and 2N data regimes, however, we notice the expected rise in performance.

![](images/4d40aaacf8628387be1d30eddbecc17b2f7bc1a0f3fa7d4877604f8a1337b2d5.jpg)  
(b) Long-context policies in the limited data regime aren’t able to complete the task even if the memory requirement for success is relaxed. This changes as data is scaled. Short-context policies are successful regardless of data.

![](images/e225695595a02a0081c7aef6320fad43e429f25dfdb788edef1ec0ae22cc4492.jpg)  
(c) Long-context policies always show high contextual success. Thus, those long-context policies which are able to complete manipulation requirements also often keep track of history.

Figure 4: Push-and-return task performance.  
![](images/f537fbd2164ee30ea9c1c7ef6f19274afd7596f4207caccab2e1843e5d0ebc1c.jpg)  
(a) Long-context policies show reasonable performance even in the low data regime (N/2), which further improves as data is increased.

![](images/ea8e289c0dccb6d8e575a0804441ce8daf8318926a7b0099c4bb81a5219c2a7a.jpg)  
(b) Long-context policies are always able to reasonably complete manipulation requirements. Contrary to push-and-return, however, short-context policies are unable to do so regardless of data.

![](images/760c9ebcf68ce74163c9e77776ba78710704acbdc4df4310e75ec3bc70bb7d8f.jpg)  
(c) Long-context policies always show high contextual success. Thus, those long-context policies which are able to complete manipulation requirements also often keep track of history.  
Figure 5: Grasp-and-return task performance.

Tasks requiring long context lengths: As shown in Figures 4a and 5a, task success significantly improves for both push-and-return and grasp-andreturn tasks as context length is increased. The primary difference is about data: grasp-and-return shows good long-context performance with just N/2 trajectories, whereas acceptable performance for push-and-return is N trajectories onward.

Additionally, we show that it is, in fact, possible to get successful long-context policies on the marshmallows task through naive scaling (contrary to what is noted in Mark et al. [9], though difference in embodiment and environment could explain this). Figure 6 shows that with simple architectural and hyperparameter changes, we are able to achieve

![](images/aace25bbb07cba271f8de823d634a12bb1490aa886b776e3b5c722bbe45f0c3c.jpg)  
Figure 6: Results for the marshmallows task, where through simple hyperparameter changes we are able to achieve high performance with just 100 training trajectories and a naively scaled context length of $T _ { o } = 9 2$

high success rate on the task. We defer detailed commentary to Appendix E.4.

Collectively, these results show that success at longer context lengths is strongly data scale dependent. Sample complexity for good performance with long context lengths seems to be dictated, to a significant extent, by complexity of the manipulation primitive used: locally stable prehensile tasks like lift and grasp-and-return show favorable trends even in the low data regime, whereas tasks like push-T, square, and push-and-return require more data. This is consistent with the intuition frequently used for short-context policy learning. We demonstrate the favorable impact of locally stable prehensile manipulation in Appendix D.2.

It is reasonable to ask: why do prior works strongly criticize naive context length scaling? We attribute this to absence of benchmarking on dataset size, architecture choice, and task properties. We provide detailed discussion in Appendix D.1.

## 3.3 Additional Comments

Investigating Figures 4b and 5b for push-and-return and grasp-and-return respectively, we note that manipulation completion at longer context lengths for both tasks mimics the trends in task success. Further investigating Figures 4c and 5c for contextual success in these tasks respectively, we find that those long-context policies which complete manipulation often also manage to keep track of contextual requirements. Thus, we hypothesize that when long-context policies do fail, it is primarily due to failure at learning to manipulate within the domain, not due to failure to keep track of requirements arising from history. Further discussion is in Appendix D.3.

Another surprising part of the manipulation completion criterion is seen at short context lengths $( T _ { o } \in \{ 4 , 8 \} )$ ). While short-context manipulation completion, as expected, is high for push-andreturn at all data scales, it is low for grasp-and-return regardless of data scale. We attribute this to observability of expert hidden state and defer discussion to Appendix D.4.

## 4 Training Long-Context Diffusion Policies

While scaling data helps with long-context learning (Section 3), we now present solutions for the limited data regime. We begin by establishing UNet as a favorable action denoising backbone choice when training single-task diffusion policies from scratch, with cross-attention a favorable conditioning method in low data regime (Section 4.1). We then propose a new learning algorithm for long-context policies (Section 4.2), and revisit prior work in long-context learning [6] (Section 4.3).

![](images/ba3aa3a44ca4f6472370cd10c53877e16e9fb1dddd5d5e22db7437041bea50c9.jpg)  
Figure 7: Comparison of UNet + Cross Attention, UNet + FiLM and DiT. DiT shows catastrophic failure in significant number of cases. The other two architectures perform similarly at short context lengths, while UNet + Cross Attention shows improvement at longer context lengths especially in the limited data regime. Figure 13 shows comparison by data scale at a given context length. As data is scaled, performance gap between architectures decreases, suggesting that cross-attention provides an inductive bias that reduces the sample complexity of long-context policy learning.

## 4.1 Cross-Attention Conditioning

We compare the frequently used UNet + FiLM and DiT architectures with UNet + Cross-Attention on push-T, square, and push-and-return. While cross-attention conditioning for UNets has been widely used in the computer vision community before [24], we have not seen it used to train robot learning policies. A detailed design explanation of the 3 diffusion policy architectures compared in this section is presented in Appendix B.2. Figure 7 shows the comparison in the $N / 2$ and N data regimes.

The DiT architecture from Chi et al. [1] largely fails, even showing catastrophic failure in push-andreturn. In fact, the behavior of DiT noticed for push-T when comparing performance at $T _ { o } = 2$ and $T _ { o } = 2 0$ is similar to the trends noted by Torne et al. [6] and could explain their criticism of naive scaling. This architecture should clearly not be chosen as a baseline.

The UNet+Cross-Attention architecture provides an advantage in the low data regime at long context lengths. However, at short context lengths, the performance, especially between the two UNet architectures, is relatively similar. This could be the reason why most robot learning works using short-context diffusion policy have not looked beyond UNet+FiLM [18, 19, 20, 21]. Further discussion and comparison organized by data regime at a given context length is presented in Appendix E.1.

## 4.2 Variable History Training

While architectural inductive bias (Section 4.1) shows improvements, we want to further close the performance gap between short and long context policies as noted in the $N / 2$ data regime. Given our investigation in Appendix D.3 and supported by prior literature [9, 15], we speculate that longcontext policies perform worse because of over-fitting with limited data, which then leads to worse co-variate shift when deployed closed loop. Evidently, short-context policies do not exhibit this. We therefore propose a joint history training method, where a model learns over a curriculum of context lengths as opposed to just one fixed context length, using the short-context datagrams to mitigate over-fitting while still benefiting from memory when useful (and ignoring when not).

![](images/7793b7bdb7147c2ece622c8ba3e46ba810d87e1a8a5ae4d0a89247ee46539b01.jpg)  
Figure 8: In the low $( N / 2 )$ data regime, variable history training leads to gains in long-context performance. Policies are able to perform as well as the best short-context policy. See Figure 15 for N data regime performance.

For a discrete sample space C, let $\Delta ( \mathcal { C } )$ denote the set of all possible probability distributions over C. Our learning algorithm is described in Algorithm 1.

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1 Variable History Training for Diffusion Policy
Require: Dataset $\mathcal{D} = \{\tau_1, \tau_2, \ldots, \tau_N\}$, $\tau_i = \{(o_1, a_1), (o_2, a_2), \ldots (o_{T_i}, a_{T_i})\}$, a minimum context length $c \in \{1, \ldots, T_o\}$, a training step varying distribution over $\mathcal{C} = \{c, c + 1, \ldots, T_o\}$ denoted by $\rho_i \in \Delta(\mathcal{C})$, where $i$ is the training step. Define $T_p^{past}(m) = \min\{m, T_p^{past}\}$
Ensure: Trained model parameters $\theta$
1: Initialize $\theta$ randomly
2: for each training iteration $i$ do
3: Sample datagrams from $\mathcal{D}$ and $m \sim \rho_i$
4: $\mathcal{L} \leftarrow \| \pi_\theta (\mathbf{o}[k - m + 1:k]) - \mathbf{a}[k - T_p^{past}(m) + 1:k + T_p] \|^2$
5: Update $\theta$ via gradient descent on $\mathcal{L}$
6: end for
7: return $\theta$
</div>

The key hyperparameters in Algorithm 1 are past prediction horizon $T _ { p } ^ { p a s t }$ and training step varying distribution $\rho _ { i }$ . We try the following for $\rho _ { i } \colon$

1. Random Sprinkle: Assign a relatively large probability $p$ to $T _ { o }$ and sample from $\boldsymbol { \mathcal { U } } ( \boldsymbol { c } , \boldsymbol { c } +$ $1 , \ldots , T _ { o } - 1 )$ with probability $1 - p$ . In our experiments, we set $p = 0 . 8$ . In this case, $\rho _ { i }$ is constant across i.

2. Progressive: Split C into $\mathcal { C } _ { 1 }$ and $\mathcal { C } _ { 2 } = \mathcal { C } \setminus \mathcal { C } _ { 1 }$ . Start with $\mathcal { C } _ { 1 } = \{ c \}$ , and $\rho _ { 0 } = \mathcal { U } ( \mathcal { C } _ { 1 } )$ . Up to half the total training steps, sequentially move an element from $\mathcal { C } _ { 2 }$ to $\mathcal { C } _ { 1 }$ and $\rho _ { i } = \mathcal { U } ( \mathcal { C } _ { 1 } )$ . Once $\mathcal { C } _ { 2 }$ is empty, change to random sprinkle method for the remaining steps.

While training on single context length, we use full past prediction: $T _ { p } ^ { p a s t } = T _ { o }$ (as recommended by the code from Diffusion Policy [1]). In the variable history training with limited data case, however, we notice short past prediction $( T _ { p } ^ { p a s t }$ is set to a small value compared to $T _ { o } )$ works better. Our ablations from Appendix E.2 show us that when data regime is insufficient for longcontext learning, progressive is a better choice. However, as data is scaled and naive scaling is good enough, random sprinkle with full past prediction is a better curriculum choice which maintains performance compared to naive scaling.

We present results for our algorithm in the low $( N / 2 )$ data regime in Figure 8. We notice that with our variable training method, long-context policies, even in the low data regime, match the performance of the short-context policy (and show significant improvements over simply naively scaling to long-context). Figure 15 shows that when data is sufficient for good long-context policy performance with naive scaling $( N$ data regime), the method does not lead to drop in performance. It can thus be used in a general manner on tasks without requiring benchmarking for dataset size.

![](images/7e5283555b65945ac5b92c8a9ef5bcd47647e28def570be9e6043a9d48081cbf.jpg)  
Figure 9: Results comparing trainable encoder and encoder frozen from short context lengths with past action prediction. Past-action prediction and frozen observation encoder seem to be driving each others’ success, but push-and-return is an exception.

## 4.3 Revisiting Past-Token Prediction

Torne et al. [6] proposed predicting past actions as an auxiliary loss that helps with long-context learning. All our naive scaling policies from Sections 3 and 4.1 use past action prediction (a design choice from [1]). Yet, we do not always see successful/improving long-context policy performance. In addition to past-token prediction, they make other design choices, such as freezing observation encoder from short-context policy, purely as a practical training optimization technique claiming no impact on policy performance. In Figure 9, we explicitly compare which aspect helps, and find that encoder freezing is certainly contributing to success, either independently or in conjunction with past-action prediction. However, there are exceptions (push-and-return), and our method matches/out-performs. Note that our implementation of past-token prediction differs slightly from that of Torne et al. [6]; we use our suggested UNet+Cross-Attention baseline, as the DiT architecture they use shows poor performance even with favorable hyperparameter changes (Figure 7). We defer further discussion to Appendix E.3.

## 5 Limitations and Discussion

While our work is about model architecture, inference time overhead due to longer context length remains a challenge. For instance, distillation methods to lightweight architectures could be improved to account for long-context policies, speeding up inference despite a large set of image inputs that need to be processed by the policy in closed loop.

Further, while we present variable histories at training time, we hope the algorithm encourages the robotics community to think of robot policies closer to the realm of agentic LLMs. A policy trained on multiple context lengths can also use variable context lengths at inference time, and algorithms establishing how to regulate inference time context length input are a promising future direction.

## 6 Conclusion

Varying data scale and task properties, we show that naively scaling context length is not necessarily catastrophic. We have established sensitivity to architecture in long-context imitation learning, showing UNet+Cross-Attention provides a helpful inductive bias when training single-task policies from scratch in the low data regime. We show that through variable history training, long-context policy performance is improved in the low data regime and maintained otherwise. Finally, we have commented on the underlying mechanisms of past-action prediction as an auxiliary loss [6].

## Acknowledgments

This material is based upon work supported by the National Science Foundation (NSF) Engineering Research Center for Human AugmentatioN via Dexterity (HAND) (Grant No. 2330040), NSF Graduate Research Fellowship Program under Grant No. DGE-2141064, Natural Sciences and Engineering Research Council of Canada CGS D-587703, and Amazon.com Services LLC PO #2D-19307345. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) only.

We thank The Infrastructure Group at MIT CSAIL for maintaining the compute resources primarily used for this work. Additionally, this work used ACES computing resources provided by Texas A&M High Performance Research Computing through allocation CIS250858 from the Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support (ACCESS) program, which is supported by U.S. NSF grants #2138259, #2138286, #2138307, #2137603, and #2138296 [25]. We also acknowledge the MIT SuperCloud and Lincoln Laboratory Supercomputing Center for providing HPC, database, and consultation resources [26].

## References

[1] C. Chi, Z. Xu, S. Feng, E. Cousineau, Y. Du, B. Burchfiel, R. Tedrake, and S. Song. Diffusion policy: Visuomotor policy learning via action diffusion, 2024. URL https://arxiv.org/ abs/2303.04137.

[2] T. Z. Zhao, V. Kumar, S. Levine, and C. Finn. Learning fine-grained bimanual manipulation with low-cost hardware, 2023. URL https://arxiv.org/abs/2304.13705.

[3] TRI LBM Team. A careful examination of large behavior models for multitask dexterous manipulation, 2025. URL https://arxiv.org/abs/2507.05331.

[4] Physical Intelligence. π : a vision-language-action model with open-world generalization, 2025. URL https://arxiv.org/abs/2504.16054.

[5] Gemini Team. Gemini: A family of highly capable multimodal models, 2024. URL https: //arxiv.org/abs/2312.11805.

[6] M. Torne, A. Tang, Y. Liu, and C. Finn. Learning long-context diffusion policies via past-token prediction, 2025. URL https://arxiv.org/abs/2505.09561.

[7] A. Sridhar, J. Pan, S. Sharma, and C. Finn. MemER: Scaling up memory for robot control via experience retrieval, 2025. URL https://arxiv.org/abs/2510.20328.

[8] M. Torne, K. Pertsch, H. Walke, K. Vedder, S. Nair, B. Ichter, A. Z. Ren, H. Wang, J. Tang, K. Stachowicz, K. Dhabalia, M. Equi, Q. Vuong, J. T. Springenberg, S. Levine, C. Finn, and D. Driess. MEM: Multi-scale embodied memory for vision language action models, 2026. URL https://arxiv.org/abs/2603.03596.

[9] M. S. Mark, J. Liang, M. Attarian, C. Fu, D. Dwibedi, D. Shah, and A. Kumar. BPP: Longcontext robot imitation learning by focusing on key history frames, 2026. URL https:// arxiv.org/abs/2602.15010.

[10] Y. Dai, H. Fu, J. Lee, Y. Liu, H. Zhang, J. Yang, C. Finn, N. Fazeli, and J. Chai. Robomme: Benchmarking and understanding memory for robotic generalist policies, 2026. URL https: //arxiv.org/abs/2603.04639.

[11] R. Zheng, Y. Liang, S. Huang, J. Gao, H. Daume III, A. Kolobov, F. Huang, and J. Yang.´ TraceVLA: Visual trace prompting enhances spatial-temporal awareness for generalist robotic policies, 2025. URL https://arxiv.org/abs/2412.10345.

[12] H. Shi, B. Xie, Y. Liu, L. Sun, F. Liu, T. Wang, E. Zhou, H. Fan, X. Zhang, and G. Huang. MemoryVLA: Perceptual-cognitive memory in vision-language-action models for robotic manipulation, 2026. URL https://arxiv.org/abs/2508.19236.

[13] M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. Foster, G. Lam, P. Sanketi, Q. Vuong, T. Kollar, B. Burchfiel, R. Tedrake, D. Sadigh, S. Levine, P. Liang, and C. Finn. OpenVLA: An open-source vision-language-action model, 2024. URL https://arxiv.org/abs/2406.09246.

[14] P.-L. Guhur, S. Chen, R. Garcia, M. Tapaswi, I. Laptev, and C. Schmid. Instruction-driven history-aware policies for robotic manipulations, 2022. URL https://arxiv.org/abs/ 2209.04899.

[15] P. de Haan, D. Jayaraman, and S. Levine. Causal confusion in imitation learning, 2019. URL https://arxiv.org/abs/1905.11979.

[16] C. Wen, J. Lin, T. Darrell, D. Jayaraman, and Y. Gao. Fighting copycat agents in behavioral cloning from observation histories, 2020. URL https://arxiv.org/abs/2010.14876.

[17] J. Ho, A. Jain, and P. Abbeel. Denoising diffusion probabilistic models, 2020. URL https: //arxiv.org/abs/2006.11239.

[18] R. Cadene, S. Alibert, A. Soare, Q. Gallouedec, A. Zouitine, S. Palma, P. Kooijmans, M. Aractingi, M. Shukor, D. Aubakirova, M. Russi, F. Capuano, C. Pascal, J. Choghari, J. Moss, and T. Wolf. LeRobot: State-of-the-art machine learning for real-world robotics in PyTorch. https://github.com/huggingface/lerobot, 2024.

[19] A. Wei, A. Agarwal, B. Chen, R. Bosworth, N. Pfaff, and R. Tedrake. Empirical analysis of sim-and-real cotraining of diffusion policies for planar pushing from pixels, 2025. URL https://arxiv.org/abs/2503.22634.

[20] A. Mandlekar, D. Xu, J. Wong, S. Nasiriany, C. Wang, R. Kulkarni, L. Fei-Fei, S. Savarese, Y. Zhu, and R. Mart´ın-Mart´ın. What matters in learning from offline human demonstrations for robot manipulation. In Conference on Robot Learning (CoRL), 2021.

[21] A. Z. Ren, J. Lidard, L. L. Ankile, A. Simeonov, P. Agrawal, A. Majumdar, B. Burchfiel, H. Dai, and M. Simchowitz. Diffusion policy policy optimization, 2024. URL https:// arxiv.org/abs/2409.00588.

[22] O. Ronneberger, P. Fischer, and T. Brox. U-net: Convolutional networks for biomedical image segmentation, 2015. URL https://arxiv.org/abs/1505.04597.

[23] E. Perez, F. Strub, H. de Vries, V. Dumoulin, and A. Courville. FiLM: Visual reasoning with a general conditioning layer, 2017. URL https://arxiv.org/abs/1709.07871.

[24] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer. High-resolution image synthesis with latent diffusion models, 2022. URL https://arxiv.org/abs/2112.10752.

[25] T. J. Boerner, S. Deems, T. R. Furlani, S. L. Knuth, and J. Towns. Access: Advancing innovation: Nsf’s advanced cyberinfrastructure coordination ecosystem: Services & support. In Practice and Experience in Advanced Research Computing 2023: Computing for the Common Good, PEARC ’23, page 173–176, New York, NY, USA, 2023. Association for Computing Machinery. ISBN 9781450399852. doi:10.1145/3569951.3597559. URL https://doi.org/10.1145/3569951.3597559.

[26] A. Reuther, J. Kepner, C. Byun, S. Samsi, W. Arcand, D. Bestor, B. Bergeron, V. Gadepally, M. Houle, M. Hubbell, M. Jones, A. Klein, L. Milechin, J. Mullen, A. Prout, A. Rosa, C. Yee, and P. Michaleas. Interactive supercomputing on 40,000 cores for machine learning and data analysis. In 2018 IEEE High Performance extreme Computing Conference (HPEC), page 1–6. IEEE, 2018.

[27] K. Song, B. Chen, M. Simchowitz, Y. Du, R. Tedrake, and V. Sitzmann. History-guided video diffusion, 2025. URL https://arxiv.org/abs/2502.06764.

[28] Z. Li, X. B. Peng, P. Abbeel, S. Levine, G. Berseth, and K. Sreenath. Reinforcement learning for versatile, dynamic, and robust bipedal locomotion control, 2024. URL https://arxiv. org/abs/2401.16889.

[29] A. Kumar, Z. Fu, D. Pathak, and J. Malik. RMA: Rapid motor adaptation for legged robots, 2021. URL https://arxiv.org/abs/2107.04034.

[30] X. B. Peng, M. Andrychowicz, W. Zaremba, and P. Abbeel. Sim-to-real transfer of robotic control with dynamics randomization. In 2018 IEEE International Conference on Robotics and Automation (ICRA), page 3803–3810. IEEE, May 2018. doi:10.1109/icra.2018.8460528. URL http://dx.doi.org/10.1109/ICRA.2018.8460528.

[31] R. P. Singh, Z. Xie, P. Gergondet, and F. Kanehiro. Learning bipedal walking for humanoids with current feedback. IEEE Access, 11:82013–82023, 2023. ISSN 2169-3536. doi:10.1109/ access.2023.3301175. URL http://dx.doi.org/10.1109/ACCESS.2023.3301175.

[32] N. F. Liu, K. Lin, J. Hewitt, A. Paranjape, M. Bevilacqua, F. Petroni, and P. Liang. Lost in the middle: How language models use long contexts, 2023. URL https://arxiv.org/abs/ 2307.03172.

[33] J. Li, M. Wang, Z. Zheng, and M. Zhang. Loogle: Can long-context language models understand long contexts?, 2024. URL https://arxiv.org/abs/2311.04939.

[34] G. Qin, Y. Feng, and B. Van Durme. The NLP task effectiveness of long-range transformers. In Proceedings ofthe 17th Conference ofthe European Chapter ofthe Associationfor Computational Linguistics. Association for Computational Linguistics, 2023. doi:10.18653/v1/2023. eacl-main.273. URL http://dx.doi.org/10.18653/v1/2023.eacl-main.273.

[35] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition, 2015. URL https://arxiv.org/abs/1512.03385.

[36] R. Tedrake and the Drake Development Team. Drake: Model-based design and verification for robotics, 2019. URL https://drake.mit.edu.

[37] R. Tedrake. Robotic Manipulation. 2024. URL http://manipulation.mit.edu.

[38] S. Ross, G. J. Gordon, and J. A. Bagnell. A reduction of imitation learning and structured prediction to no-regret online learning, 2011. URL https://arxiv.org/abs/1011.0686.

## A Additional Related Works

## A.1 Context Length in Imitation Learning

We have already introduced and discussed the work from Torne et al. [6], Mark et al. [9], Sridhar et al. [7]. We now elaborate on these. Mark et al. [9] suggest using a pre-trained VLM to identify key events in policy history, and only conditioning the policy on these VLM identified frames. They also argue that naively increasing context length can avoid failure, but focus on action chunking as the reason for this. They have limited data scaling ablation, but it largely agrees with our results. Other methods use language based summarization of observation histories [7, 8]. They keep track of historic events through language/intermediate tokens summarizing history of observations. While [7] rely on just the most recent observation for control, [8] introduce video based observation encoding for the recent history. Such a method, however, would require extensive pre-training (they pre-train on internet scale data). Song et al. [27] propose attention between noisy versions of conditioning and action trajectory at training time and temporal classifier free guidance at sampling time. Their model, however, is a video prediction model, which predicts both future actions and observations and can be data and compute intensive.

## A.2 Context Length in Reinforcement Learning in Robotics

Reinforcement learning in robotics frequently provides solutions for more dynamic tasks in robotics, and RL policies therefore often use longer context lengths than seen in imitation learning (though these observations are typically in joint space, unlike RGB observations seen in imitation learning). Some RL works in robotics have provided explicit comments on context length. [28] use a context of 66 (2 seconds at 33 Hz). Along with the entire history provided as an encoded embedding, they provide the recent 4 frames directly to the base model, arguing that full history and recent history are used differently and therefore recent history should additionally be provided through separate channels. [29] split the policy into a base policy and an adaptation module. The adaptation module takes a long history and produces latent embeddings at a lower frequency, while the base policy only takes recent state and this latent embedding to produce control at a higher frequency. [30] and [31] provide ablations on comparing feed-forward policies with history with LSTM based policies. [30] argue that a non-state history based LSTM better generalizes to dynamics of a physical system, while [31] choose a non-history based policy as the best performer based on their ablations.

## A.3 Context Length Understanding in LLMs

We believe the robotics community needs to build an understanding of policy learning in the presence of longer contexts. LLM community has considerable work exploring the factors that impact performance of LLMs with longer conditioning contexts. For instance, [32] show that LLM performance for long-context inputs significantly depends on the location of relevant information in the input. Specifically, performance degrades if the most relevant information is in the middle of the model input history. [33] create a benchmark for long-context understanding in LLMs and show that LLMs struggle with ”intricate long dependency tasks”. [34] show that certain attention mechanisms can lead to insufficient attention being paid to distant tokens in longer contexts.

## B Diffusion Policy Details

## B.1 Overview

Diffusion Policy is a state-of-the-art behavior cloning method introduced by Chi et al. [1] which learns the observation conditional action distribution mentioned in Equation 1 by minimizing the action prediction loss

$$
\mathcal {L} _ {\mathcal {D}} (\theta) = \mathbb {E} _ {k, (\mathbf {o}, \mathbf {a}) \sim \mathcal {D}, \epsilon \sim \mathcal {N} (0, I)} [ \| \epsilon - \epsilon_ {\theta} (\alpha_ {k} \mathbf {a} + \sigma_ {k} \epsilon , \mathbf {o}, k) \| _ {2} ^ {2} ]\tag{2}
$$

where $\alpha _ { k }$ and $\sigma _ { k }$ are parameters related to the diffusion noise schedule (k represents diffusion timestep). Essentially, the model learns to regress over noisy versions of action chunks from observation and diffusion timestep input, and at inference time uses a denoiser to conditionally sample from the action chunk distribution.

Diffusion Policy is implemented through a combination of observation encoder and action denoising backbone. As a summary: each camera c produces an image stream that comprises $T _ { o }$ images $\left\{ o _ { i } ^ { c } , i \in \left\{ t - T _ { o } + 1 , \dots , t \right\} \right\}$ }. For each stream, an image encoder converts each of these $T _ { o }$ images into latent embeddings $e _ { i } ^ { c }$ . Note that the same vision backbone is used for all timesteps for each camera stream. These embeddings, concatenated with diffusion timestep embedding and robot proprioception, are passed to the downstream model for conditioning. Common choices for the downstream conditioning model are UNet and DiT. Various methods like FiLM [23] and crossattention can be applied to merge the conditioning with the denoiser.

We choose ResNet18 [35] as our image embedding backbone. Unless noted otherwise in the specific sections, the ResNet18 version from robomimic [20] is trained from scratch for our policies, with random weight initialization (similar to Chi et al. [1] and Wei et al. [19]).

## B.2 Comparison of Diffusion Policy Architectures

Diffusion Policy was presented with the UNet+FiLM and DiT architectures in the original work [1]. For single-task RGB policies with short context lengths, UNet+FiLM outperformed the DiT model for the given design choices and hyperparameters. Therefore, it became the choice for many works building on diffusion policy [18, 19, 20, 21]. In this work, we suggest using UNet+Cross-Attention as the method for training single-task long-context diffusion policies from scratch in the limited data regime.

As an assistance to the reader, we now provide an overview of the 3 methods based on [1, 22, 24]. All methods involve passing observation inputs through an encoder that is shared across timesteps, but is often different for cameras (though it can be the same). After that, each observation embedding is concatenated with the corresponding proprioception information. The 3 architectures compared in this work differ in flow beyond this point:

• UNet+FiLM: The observation embeddings from each timestep and each camera are concatenated into one long vector. A diffusion timestep embedding is further concatenated to this. The action denoising backbone is made of a UNet with intermediate 1D convolution layers. The long vector is projected to dimensions matching these layers through separate MLPs, and merged via FiLM [23], where a scale and bias are both applied.

• UNet+Cross-Attention: The observation embeddings from each timestep are concatenated across cameras and proprioception, but remain separate for different timesteps. Diffusion time embedding is projected to matching dimension. A position encoding is added. Each of these embeddings are then individually projected to intermediate UNet layer dimensions, and cross attended to with the intermediate UNet layers.

• DiT: The action denoising backbone is now provided by transformer decoder layers as opposed to UNet. Conditioning is merged via cross-attention.

We note that the DiT model from the original work as well as from key long-context learning work from Torne et al. [6] uses hyperparameters which result in a model that is significantly smaller in size than other choices. If we use their hyperparameters, we notice extremely low success rates in some tasks even with short context lengths. Very specifically, we notice reasonable performance on robomimic tasks. But we note almost catastrophic performance on tasks implemented in Drake [36]. We therefore add more decoder layers, providing the model number of parameters similar in magnitude to the other architectures we have used. Moreover, a key change in our experiments for DiT is not using causal attention within the action chunk. Both Chi et al. [1] and Torne et al. [6] apply causal attention within an action chunk as it is denoised (so earlier elements of the action chunk cannot be influenced by later elements). We remove this flag, to make the flow of information more comparable to that of the UNet. Finally, we evaluate the DiT with additional steps of sampling compared to the UNet models, to eliminate the possibility of sampling error driving difference in performance.

Finally, while we have not seen this in robotics policies, merging conditioning via cross-attention instead of FiLM has been used in computer vision before [24].

## B.3 Training Details and Hyperparameters

## B.3.1 Checkpoint Selection

We designate a maximum number of training steps for each task based on approximate policy performance to convergence for short-context policies at N training trajectories. We then train policies with all context lengths for that task to those many training steps, saving the top 5 checkpoints with respect to validation loss, top 5 checkpoints with respect to action matching validation loss, and 3 evenly spaced checkpoints toward the end of training. We then evaluate each checkpoint, and the highest success rate among checkpoints is reported as the success rate for the training run. For our hardware evaluation on the marshmallows task, we simply use the latest checkpoint.

We train all push-T policies for 150k steps, marshmallows, square, lift, and push-and-return policies for 200k steps, and grasp-and-return policies for 100k steps. We start with a learning rate of 1e-4 and use cosine decay. For some policies (like UNet+FiLM and UNet+Cross-Attention on push-T and UNet+FiLM on push-and-return), we only save the latest checkpoint as opposed to 3 recent toward the end (because the results are largely insensitive to this choice, we did not rerun these configurations due to the high computational cost). We did rerun it for the N data regime in push-and-return for UNet+FiLM, and found the presented trends to be consistent.

We notice in our work that policies with different context lengths can sometimes converge in different number of training steps. For instance, we usually notice that especially with N/2 trajectories, some policies often converge slightly sooner than those for N and 2N, an effect that is more pro nounced at longer context lengths. We believe earlier checkpoints at longer context lengths in the low data regime help mitigate the impact of over-fitting.

Each reported success rate is from one training run. While we do account for uncertainty in the binary task success, we do not, in this work, account for uncertainty in training randomness. Due to limited availability of resources and high amount of training time needed to produce a study of this magnitude, such a choice is consistent with prior insightful work [19].

## B.3.2 Hyperparameters

In Table 1 we present the hyperparameters used for the results in Sections 3 and 4. By design, FiLM conditioning model size changes considerably as context length is increased. On the contrary, UNet+Cross-Attention and DiT have similar model sizes across context lengths due to the manner of applying conditioning. We choose the size of the cross-attention model to be roughly half way between the smallest and largest used FiLM model. For DiT, we get a model comparable in size to the cross-attention model after adding significantly more layers to the model from Torne et al. [6] and Chi et al. [1].

Table 1: Architecture-specific hyperparameters for the three denoisers. Values not applicable to a given architecture are marked $^ { 6 6 } - ^ { 3 9 } .$

<table><tr><td></td><td>UNet+FiLM</td><td>UNet+xAttn</td><td>DiT</td></tr><tr><td colspan="4">Denoiser core</td></tr><tr><td>Conditioning mechanism</td><td>FiLM (global)</td><td>Cross-attention (token seq.)</td><td>Cross-attention (token seq.)</td></tr><tr><td>UNet Down channels</td><td>[256, 512, 1024]</td><td>[256, 512, 1024]</td><td>-</td></tr><tr><td>Conv kernel size</td><td>5</td><td>5</td><td>-</td></tr><tr><td>GroupNorm groups</td><td>8</td><td>8</td><td>-</td></tr><tr><td colspan="4">Approximate parameter counts</td></tr><tr><td>Denoiser parameters</td><td> $\approx 67M$  at  $h_y = 2$  to $\approx 212M$  at  $h_y = 80$ </td><td> $\approx 129M$ </td><td> $\approx 109M$ </td></tr><tr><td>Encoder parameters</td><td>22.4M</td><td>22.4M</td><td>22.4M</td></tr><tr><td>Total parameters</td><td> $\approx 89M$  at  $h_y = 2$  to $\approx 235M$  at  $h_y = 80$ </td><td>151M</td><td>131M</td></tr></table>

Task specific hyperparameters are available in Table 2. Each task differs in image resolution, crop size, action dimensionality, the set of observation horizons we sweep, the number of training demonstrations used, and the total number of training steps. Total training steps are matched between baselines and ablations to ensure a fair comparison.

Our models are designed such that the time dimension of the trajectory being denoised needs to be a multiple of 4. We select a total denoising horizon such that the length of the predicted future is roughly 16 while retaining full past action prediction and having the total length a multiple of 4. Thus, for context length 1, our horizon is 16, for 2, it is also 16, for context length 5 it is 20, and for context length 10 it is 24. If the context length is a multiple of 4, the horizon is simply context length + 16. Further details are in Table 2.

While this is an accurate tabular representation, the policy design involves many more hyperparameters which cannot be best captured in this manner. We therefore refer the reader to our website and codebase for the updated and most accurate configuration files corresponding to our experiments.

Table 2: Some details specific to the task evaluated.

<table><tr><td>Setting</td><td>Push-T</td><td>Push-and-return</td><td>Grasp-and-return</td><td>Marshmallows (hardware)</td><td>Square (robomimic)</td><td>Lift (robomimic)</td></tr><tr><td>Image input ( $H \times W$ )</td><td>128×128</td><td>128×128</td><td>128×128</td><td>128×128</td><td>84×84</td><td>84×84</td></tr><tr><td>Crop ( $H \times W$ )</td><td>112×112</td><td>112×112</td><td>112×112</td><td>112×112</td><td>76×76</td><td>76×76</td></tr><tr><td>Cameras</td><td>overhead, wrist</td><td>overhead, wrist</td><td>overhead, wrist</td><td>overhead, wrist</td><td>agentview, eye-in-hand</td><td>agentview, eye-in-hand</td></tr><tr><td>Proprioception dim</td><td>3</td><td>2</td><td>13</td><td>13</td><td>9</td><td>9</td></tr><tr><td>Action dim</td><td>2</td><td>2</td><td>13</td><td>13</td><td>10</td><td>10</td></tr><tr><td>Context length swept ( $h_y$ )</td><td>1, 2, 5, 10, 12, 16, 20</td><td>4, 8, 32, 60, 72, 80</td><td>4, 8, 16, 20, 24, 48</td><td>92</td><td>1, 2, 5, 10, 12, 16, 20</td><td>1, 2, 5, 10, 12, 16, 20</td></tr><tr><td>Future chunk length  $h_u$ </td><td> $h_y+14$  to  $h_y+16$ </td><td> $h_y+16$ </td><td> $h_y+16$ </td><td> $h_y+16$ </td><td> $h_y+14$  to  $h_y+16$ </td><td> $h_y+14$  to  $h_y+16$ </td></tr><tr><td>Data (low/usual/high)</td><td>80/160/320</td><td>24/48/96</td><td>12/24/48</td><td>100</td><td>50/100/200</td><td>25/50/100</td></tr><tr><td>Total training steps</td><td> $1.5×10^5$ </td><td> $2×10^5$ </td><td> $1×10^5$ </td><td> $2×10^5$ </td><td> $2×10^5$ </td><td> $2×10^5$ </td></tr></table>

## C Task Details

The value of N used for each task is indicated in Table 3. We then present further details about data collection (Section C.1) and evaluation (Section C.2) for all the tasks.

Table 3: Number of training trajectories chosen as N for different tasks used in Section 3. The values for benchmark tasks are chosen to be consistent with prior work [19, 20].

<table><tr><td>Task</td><td>Value of N</td></tr><tr><td>push-T</td><td>160</td></tr><tr><td>robomimic square</td><td>100</td></tr><tr><td>robomimic lift</td><td>50</td></tr><tr><td>push-and-return</td><td>48</td></tr><tr><td>grasp-and-return</td><td>24</td></tr><tr><td>marshmallows</td><td>100</td></tr></table>

## C.1 Task and Data Collection Descriptions

Push-T: Observations are images (from pixels) and robot proprioception, whereas actions are a chunk of the next planar location of the cylindrical end effector. Dataset for this task was collected using teleoperation and is available via [19]. We choose N = 160.

Robomimic Tasks: We use the proficient human dataset. Further details on the environments are available via robomimic [20]. We choose N = 100 for square and N = 50 for lift.

Push-and-Return: Observation and action space are the same as defined above for push-T. We collect data for this task via teleoperation in simulation. For each mode, we collect two datasets: (i) by randomly initializing the block in that location and completing the task via teleoperation and (ii) by mirroring tele-operated roll-outs from the symmetrically opposite mode. This allows us to ensure coherent data across symmetric modes, and eliminate inconsistent teleoperation as a source of learning discrepancy. We use a two-mode version for all the demonstrated comparisons and leave it to future work to evaluate the same task with more return modes. Such a future experiment can provide insight on increasing combinatorial complexity while keeping local complexity the same. We choose $N = 4 8$ . See Figure 10a for reference.

Grasp-and-Return: Observations remain the same as in push-and-return, but action is now 6-DOF end-effector pose as opposed to planar location as in previous cases. Data for this task is generated in simulation using heuristic method adapted from [37]. We choose N = 24. See Figure 10b for reference.

![](images/ecaa627dd9c456f6e7a8508bd19db7241f2d1e32874d363bb0a187fa0c864eeb.jpg)  
(a)

![](images/a61688c1280ee45cc3d2855690bf0822eb27b3a4c2841197352e14d14ae234db.jpg)  
(b)

![](images/341de049c856023ad550ed52913a984fcbdeb0185efacff055a9fda9750c31f8.jpg)  
(c)  
Figure 10: Environments for tasks requiring long context length for success. (a) Push-and-return (b) Grasp-and-return (c) Marshmallows.

Marshmallows: This task requires moving two handfuls of marshmallows to a target bowl from a supply bowl, and then signaling task completion by pressing a button (we represent this by gripper closing and reaching a small platform of red blocks). We adapt this task on our hardware from [9], who in turn have adapted it from a task in [6]. We collect data using teleoperation via VR. Observations are provided via one overhead camera and one wrist camera. Action space is the same as in grasp-and-return. We choose N = 100. See Figure 10c for reference.

## C.2 Evaluation Details

Reported success rates are out of 200 trials for simulated tasks with error bars representing 95% Wilson score confidence intervals.

Push-T: Specific evaluation details for this task are based on [19].

Push-and-return: In the 2-mode case, we conduct 100 trials with the block originating on either side. The time limit is set to 50s. The block is considered moved to the middle and back to either location if it overlaps by more than 95% of its area with the marker for that location.

Grasp-and-return: We conduct 100 trials with the brick originating on either side. The brick is considered moved to the center if it makes contact with the table space between the mats on either side. The brick is considered returned to a location if it is returned to anywhere on the mat from which it originated.

Marshmallows: We conduct 20 trials on hardware. The number of marshmallows at start of the task in the target bowl can be arbitrary (consistent with data collection, and makes the task challenging in terms of memory).

## D Additional Discussion

## D.1 Reasons for Prior Works’ Criticism of Naive History Scaling

In Section 3.2, we show that extreme failure as indicated in some prior works is not noted with naive context length scaling. However, some prior works [6, 7] do advertise that naively scaling context length can lead to stark failure. We elaborate on our explanation for why perhaps they do so:

1. Not benchmarking on dataset size: As shown through results in Figures 4a, 5a, 3a, 3b, and 3c, increasing data leads to significant gains at longer context lengths and the relative difference between short and long context policies changes. Notably, tasks show different sample complexity at longer context lengths (locally stable grasping tasks like lift and grasp-and-return show success even in the case of N/2 trajectories at longer context lengths). Thus, without benchmarking on dataset size, it is unclear if the policy is being tested in the low data regime for the task or if naive scaling is algorithmically catastrophic.

![](images/a3af3176bc54ea8acca82f897c2b50dba81387694f6ce58d32ca3fe6bb33af53.jpg)  
Figure 11: During data collection, the robot grippers make contact with the brick along its central axis while grasping. At inference time however, the policy commands grippers to make contact away from the center location, indicating error in learning the manipulation skills from data. However, because this task is locally stable, despite this error policies perform well. This task thus shows lower sample complexity for long-context learning.

Data scale investigation from Mark et al. [9], while not as detailed as our work, does support our claims.

2. Inappropriate architecture choice: As we show in section 4.1, the choice of conditioning method significantly impacts long-context policy learning. Vast majority of diffusion policy works use UNet + FiLM [18, 19, 20, 21]. Some other works [6] use DiT implementation from Chi et al. [1], which is indicated to be a poorly performing denoising backbone for single-task policies from image inputs [1] and should not be used for benchmarking single task policies. We show in Section 4.1 that even with favorable hyperparameter changes, the DiT fails to perform in the simulation tasks we define in Drake [36].

3. Lack of Task Comparisons: Our results comparing Figures 4a and 3a show that even if some methods perform well on tasks requiring memory to succeed, they may still show diminishing returns on tasks solvable with short histories. As we incrementally build solutions toward general purpose policies, it is necessary to provide a long-context learning solution which performs well across the spectrum of tasks needing and not needing long memory to solve. Therefore, it is necessary to benchmark solutions on tasks solvable with short history when proposing long-context learning solutions.

## D.2 Impact of Grasp as Manipulation Primitive

In Section 3, we show that tasks like lift and grasp-and-return show reasonable long-context performance even in the low data regime (N/2). We speculate that this is because these tasks involve locally stable manipulation primitives, and policies can perform well despite errors in manipulation.

To support our claim, we study how well did a long-context policy learn to grasp for the grasp-andreturn task in the N data regime. We examine the grasping skill learned in grasp-and-return by comparing the location where gripper makes contact with the brick relative to the center of the brick. As shown in Figure 11, training data has gripper making contact at the center of the brick (we can control for this as the data is synthetically generated with a heuristic method). We show the mean and median absolute value of gripper contact location with respect to center of the brick at inference time (policy rollout) in Table 4. Despite having errors more than 1 cm, the task shows success at long contexts. This shows that for tasks where it is possible to succeed even with significant errors in the learned manipulation skill, long-context policies are able to learn even at low sample complexity.

Table 4: Mean and median of the absolute value of gripper contact location offset from the center of the brick in meters. The value for expert truth data is 0. Despite most values being more than 1 cm away from the center, long-context policies succeed (N training trajectories).

<table><tr><td>Context Length</td><td>20</td><td>24</td><td>48</td></tr><tr><td>Mean</td><td>0.012</td><td>0.014</td><td>0.013</td></tr><tr><td>Median</td><td>0.010</td><td>0.010</td><td>0.010</td></tr></table>

![](images/8a70688a2e9e69c93eef6e8d878bec89ebfd90cd177fa833c2d85ed7c4d7534a.jpg)  
(a)

![](images/88f58c7dfe5355fb84d684e70711829cbfe77e091b91a8d3d718cd47494a5016.jpg)  
(b)  
Figure 12: Relationship between loss and success rate for the square task across checkpoints. (a) Training loss vs. success rate. (b) Validation loss vs. success rate.

## D.3 Reasons for Long-Context Failure

In Section 3 we saw how long-context policies do not show catastrophic failure in most cases. However, Figures 3a, 3b, and 4a still show that especially in the low data regime $( N / 2$ trajectories), long-context policies still perform worse than short-context ones for some tasks. In this section, we further diagnose the reasons behind these behaviors.

## D.3.1 Training and test loss

Section B.3 explains how checkpoints are saved during training. All saved checkpoints are evaluated and the success rate of the best checkpoint is reported. We note that because the work presented in this section has only been donefor one training run, the results can be noisy.

In Section 4.1, we show that some architectures like UNet+FiLM show a sharper drop as context length is increased as compared to UNet+Cross-Attention. For UNet+FiLM, consider Figures 12a and 12b, which compare the training and validation loss respectively against the success rate for all checkpoints saved during training for square task trained with $N / 2$ trajectories. As context length increases from short to long, checkpoint clusters shift downward and leftward: both training and validation loss decrease overall, but so does the success rate. The trend with training loss shows a standard over-fitting problem: policies fit to the training data better in higher dimensions, but this is not sufficient for closed loop rollouts. The validation loss curve hints toward an even more interesting trend seen in imitation learning: long-context policies perform better on open-loop validation samples drawn from expert distribution, but perform worse when rolled out closed loop. This hints toward covariate shift in imitation learning [38]. Both these trends hint at data scaling and relevant inductive biases as favorable solutions for long-context learning.

As mentioned above, we show results from one training run, so results can be noisy. The best way to reproduce this experiment would thus be to have multiple training runs, and then average the clusters from checkpoints saved during each training to produce a less noisy version of Figure 12. This, we believe, would also give more consistent results across architectures, checkpoints, and context lengths and is a very relevant direction for future work.

## D.3.2 Manipulation Skill Learning

In Section 3, we note that in the low data regime, grasp-and-return provides good long-context performance, where as push-and-return performance drops. We now comment in detail on additional evaluation criterion to investigate what do policiesfail to learn as context length increases?

For push-and-return, both task success (Figure 4a) and manipulation completion (Figure 4b) show dropping performance with increasing context length in the $N / 2$ data regime. However, investigating Figure 4c shows that contextual success is high for long context policies even when trained on $N / 2$ trajectories. This indicates that in the low data regime, most long-context policies are failing to learn to manipulate within the constraints of the task. In fact, those policies which do learn to complete manipulation also show good tracking of history. Thus, long-context policies for this task fail due to difficulty learning the manipulation skill within the realms of the task. Simultaneous and mirroring improvement seen in Figures 4a and 4b confirms this hypothesis.

We note that this simultaneous improvement between task success and manipulation completion is also noted in the grasp-and-return task (Figures 5a and 5b respectively). However, both are already good for this task in the low data regime, perhaps because this task builds on locally stable manipulation skill. In Section D.2 we have further commented on this aspect.

## D.4 When are short-context lengths sufficient?

We comment on the difference with respect to manipulation completion between short-context $( T _ { o } ~ \in ~ \{ 4 , 8 \} )$ policy performance on push-and-return (Figure 4b) v/s grasp-and-return (Figure 5b). As expected, we notice good manipulation completion for short-context policies on pushand-return. But why do they perform poorly on grasp-and-return with respect to this metric? We qualitatively examine the rollouts and find that high-level decision-making failures caused by short context lengths can cause low-level control failures.

To illustrate this, consider a commonly observed failure case for grasp-and-return with $T _ { o } = 4$ The policy does not have enough memory to commit to a subroutine (ex. moving to the center or returning to the neutral position). This mode-switching produces jittery actions that drive the robot into out-of-distribution states where its manipulation abilities deteriorate. This results in unwarranted collisions, failed grasps, and ultimately unsuccessful rollouts.

This pattern reinforces the notion that there are fundamental limitations to imitating long-memory experts with short-context policies. For instance, without sufficient history, the policy may lack the information to infer the latent state underlying the expert’s long-horizon behavior. Even more interestingly, the observability of this latent state can be dependent on the manipulation primitive used - push-and-return, due to its planar nature, has pusher on different sides of the block. As a result, it is able to infer direction of motion and even with short context length, achieve high manipulation completion. Grasp-and-return, on the other hand, cannot distinguish phase of motion because regardless of direction of motion, grasping a block is visually similar. Consequently, overcoming these failures likely requires algorithmic changes that explicitly address long-context reasoning, rather than data scaling alone. These findings motivate approaches that either extend temporal context or provide explicit sub-task guidance (e.g., via language [4]).

## E Additional Experiments

## E.1 Cross-Attention Conditioning

In Section 4.1 we establish the importance of architectural choice when comparing diffusion policy performance, especially at longer context lengths. Here we present additional results organized by different data scales at a given context length in Figure 13. While DiT shows no noticeable performance gain even in the high data regime, the performance gap between UNet+Cross-Attention and UNet+FiLM shrinks as data is scaled. This reinforces the notion that the cross-attention conditioning method is a favorable inductive bias.

Additionally, Table 1 shows that the FiLM model size can be considerably different if context length is changed, while UNet+Cross-Attention policy is consistently at around 150M parameters. Thus, it could be particularly concerning that one comparison we have made is at context length 80, where we have compared UNet+FiLM with about 235M parameters with UNet+Cross-Attention at 151M parameters.

## UNet+Cross-Attention v/s UNet+FiLM v/s DiT

![](images/c066c8e33f80b9221ac26070edd3291ca10e02685f387eba2bf5b841c5c3ff72.jpg)  
(a)

![](images/6cb5a7bdead648845dd2e58ca82998451f9293adc01c216caf0e6b26576f5d83.jpg)

![](images/cc13f41efbc6627e28721806ab6ff7831156cf067df6ddc0a569b796147c2a15.jpg)  
(c)

(b)  
![](images/fbd2caa0fd036e6eb07da911220e8fc3ea2127989ab6b14a6f56470a71728835.jpg)  
(d)

![](images/30e8c77c01250f95edb0ab17d54e9444bbe7318c29af38abf9ad9fdabd180366.jpg)  
(e)

![](images/75eded91e243ffa7ba92d310ae2bbcc84d75617b216fc8dc0cb2f18de7266566.jpg)  
(f)  
Figure 13: Comparison of UNet + Cross Attention and UNet + FiLM across tasks, context lengths, and data scales. Top row: push-T at $T _ { o } ~ = ~ 2$ and $T _ { o } ~ = ~ 2 0$ . Middle row: square at $T _ { o } ~ = ~ 2$ and $T _ { o } ~ = ~ 2 0$ . Bottom row: push-and-return at $T _ { o } = 4$ and $T _ { o } ~ = ~ 8 0$ Across tasks, the two architectures perform similarly at short context lengths, while UNet + Cross Attention outperforms UNet + FiLM at longer context lengths in limited-data regimes. As data is scaled, the performance gap decreases, suggesting that cross-attention provides an inductive bias that reduces the sample complexity of long-context policy learning. DiT is also shown as a comparison, but often fails to perform without significant pre-training in the single task cases.

![](images/579aaadf97a120fff7ef0723d6e1eac9fa2f9da358afe3cbc5c2b79e05813e98.jpg)  
Figure 14: Comparing policy architectures when total parameters for policy using FiLM conditioning are altered to match total parameters for the cross-attention policy.

The comparison we have presented in Figure 13 is such that the parameters in the intermediate layers of the UNet are kept consistent between FiLM and cross-attention conditioning. That is, we study what conditioning method is the best for a given UNet of certain hyperparameters. However, we now present a second comparison where we alter the size of the UNet layers to keep the total number of parameters consistent with UNet+Cross-Attention at a given context length. We present results in Figure 14. Note that we have added additional parameters for push-T and square at $T _ { o } = 2 0$ , and removed parameters for push-and-return at $T _ { o } = 8 0$ , to bring the total parameters similar to those of cross-attention conditioning. We find no changes across tasks in the $N / 2$ data regime.

## E.2 Variable History Training Ablations

In Section 4.2, we show significant improvements in the $N / 2$ data regime. Moreover, we choose progressive for $\rho _ { i }$ and short past action prediction horizon. In Figure 15, we show results for the N data regime as well as for other combinations of $\rho _ { i }$ and $T _ { p } ^ { p a s t }$

We note that progressive+short shows performance gains in the three tasks in the $N / 2$ data regime. With N trajectories, it shows improvement in push-T and square (where long-context policies still show scope for improvement), and maintains performance for push-and-return. Note that naively scaling context length showed drop in performance in $N / 2$ regime for all these tasks. However, in the N data regime, we saw drop only for push-T and square, but rather rising/maintained performance for push-and-return as context length was increased. While progressive+short maintains performance in the N data regime, random sprinkle+full shows improved performance. We thus suggest using progressive+short with low data, but random sprinkle+full when more data (sufficient for naive scaling) is available.

We confirm this by investigating these methods on the grasp-and-return, where, as seen in Figure ${ 5 \mathrm { a } } .$ , even $N / 2$ data regime shows good performance with naive context length scaling. Intuitively, random sprinkle+full should be the best method. That is exactly what we notice in Figure 16, where in both $N / 2$ and N data regimes, random sprinkle+full performs best.

Thus, we recommend using progressive+short when naive scaling performs worse compared to short context length, but random sprinkle+full when data is sufficient for naive history scaling.

## E.3 Past-Token Prediction

Torne et al. [6] introduce predicting past action tokens as an auxiliary loss that helps with longcontext imitation learning. Additionally, they propose freezing the observation encoder as a training time optimization technique not impacting policy performance. In this section, we independently investigate which aspect contributes to performance gains.

We first investigate how much past-action prediction, by itself, impacts policy performance (independent of other parts of the work). Because past-action prediction wasn’t proposed by Torne et al. [6] but rather present in the original work [1], we also investigate if predicting past actions helps with short-context policy learning.

![](images/777e81167d0fa75b61e36965671ef7fd86891e533627c0d764accb3fb9e8c9a3.jpg)  
(a)

![](images/cb0522b10796ebf0f561d9911ee77b3a7f55cc1583797918e0a7458d4b5ec5ec.jpg)  
(b)

![](images/6e9da5d69601dbb043527921220f70139470418b7c026ab68e4f7150104da7ce.jpg)

![](images/7a279c00387259f75240e9e01e42932982309d1e8a4b94c20f955afadd1e00df.jpg)  
(d)

![](images/83f8cb906f680588ad8e13c31be559cabfe1317205f8eae51b88e607262906ee.jpg)  
(e)

![](images/77dd65b79571e07c314927c974bcd03c0876aed3df3e4b4ff4c8670f9d7dd5ae.jpg)  
(f)  
Figure 15: Comparison of UNet+Cross-Attention, UNet+FiLM, and variable history training method introduced above across tasks (data scale $N / 2$ in left column and N in right column). Variable history methods are trained at $T _ { o } = 2 0$ for square and push-T and at $T _ { o } = 8 0$ for push-andreturn. For an appropriate choice of $T _ { p } ^ { p a s t }$ and $\rho _ { i }$ , the long-context model matches short-context performance in low data regime, and doesn’t degrade performance in the ideal data regime.

![](images/2fe5c43756ca536ad163f3e62e15ca2e67032dfd271c397b3b21cd067b87ed90.jpg)  
(a)

![](images/c76b053ce82d11bb98ffbd6385783ed2046583a3e090c0f0bc60930cceb96c99.jpg)  
(b)  
Figure 16: Variable history training method for grasp-and-return task, where naive history scaling is always sufficient for task success. Given that, we find random sprinkle+full to be the best method even in the $N / 2$ data regime.

Investigating Figure 17, we find that by itself, predicting past actions presents limited advantages. While advantages at long context lengths are evident for square and push-and-return in the N data regime, performance is more similar in the limited data case. There are also no noticeable trends for short-context policy performance. Given the limited advantage which is not seen across data scale, we conclude that the benefit of predicting past actions by itself is unclear for long-context learning.

![](images/d7ac72efe5e4b92a133a9d2c97702cf9acdb5739d7089c82b41adceca9b5f3a8.jpg)  
(a)

![](images/c17bc5505b860a0805f07f3ba8934cf8bae914d92b9991deeaf1cae031cf0889.jpg)  
(b)  
Figure 17: Results of a study investigating how much past action prediction, a design choice introduced by Chi et al. [1], helps policy learning with the UNet+Cross-Attention architecture. While advantages at long context lengths are evident for square and push-and-return in the N data regime, performance is more similar in the limited data case. Thus, overall patterns are unclear.

Next, we continue the discussion from Section 4.3, where we investigate the relative difference in policy learning between trainable and frozen encoder. While Figure 9 shows the comparison in the $N / 2$ data regime, we now show comparison in the N data regime in Figure 18. We find that at both data scales, past-action prediction and freezing the observation encoder seem to be driving each others’ success - freezing the observation encoder is not just a training technique, but seems to also be responsible for improving policy performance at least to some extent. The variable history comparison presented in Figures 9 and 18 picks the value with the best hyperparameter (between choice of random sprinkle v/s progressive).

![](images/963e6a985d91f7da624d25ee0e04e819ff18763ae2e9e1ea93d75c1fcc9ba748.jpg)  
Figure 18: Results comparing trainable and observation encoder frozen from short context lengths with past action prediction. Past-action prediction and frozen observation encoder seem to be driving each others’ success, but push-and-return is an exception.

As a final note, we state that Torne et al. [6] also propose an inference time verification technique, where multiple action chunks are sampled from the policy, and the chunk with past actions most consistent with the actually executed past actions is executed. We exclude this technique from our analysis because it reduces inference speed as we evaluate a large number of policies, and by authors’ own admission, this helps by upto only 5%. We did run some comparisons presented in Table 5 and found that success improvement from this technique was statistically insignificant, and thus excluded it from our comparisons.

Table 5: Performance with and without PTP inference time verification. Policy trained in N data regime at $T _ { o } = 2 0$ with frozen observation encoder from the best short-context policy for that task.

<table><tr><td>Task</td><td>With Verification</td><td>W/O Verification</td></tr><tr><td>Square</td><td>176/200</td><td>175/200</td></tr><tr><td>Push-T</td><td>190/200</td><td>188/200</td></tr></table>

## E.4 Hardware Validation

Consider the marshmallows task described in Section C. We naively scale context length to $T _ { o } = 9 2$ for the 3 architectures we study in Section 4.1. Results are shown in Figure 6.

This task has been adapted from Mark et al. [9]. While they use 250 demonstrations and show that their VLM based filtering trick is necessary for success, we are able to achieve high success rates with simple naive scaling and just 100 demonstrations. While our exact setups are different due to different robot embodiments and that could certainly explain any differences in our outcomes, we believe that grasping from a bowl full of marshmallows allows room for manipulation error while still succeeding (as off grasps are also likely to capture an object).

Figure 6 shows that all our conditioning architectures perform similarly. We speculate this could be because 100 trajectories are more than enough for a task like this. However, we do observe qualitative differences. We find UNet+Cross-Attention to be smoother, clearing bowl boundaries with larger margins, and exhibiting more robust re-grasps when needed. DiT, on the other hand, frequently re-grasps even when not necessary, and the gripper bumps into boundary of the bowl/doesn’t clear it with high margins. The movement is also not as smooth. UNet+FiLM qualitatively sits between the other two choices. Quantitative evaluation of such metrics in hardware remains an open challenge for robotics.