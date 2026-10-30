https://doi.org/10.1038/s44182-025-00025-4

# Are transformers truly foundational for robotics?

Check for updates

James A. R. Marshall<sup>1,2</sup> & Andrew B. Barron<sup>3</sup>

Generative Pre-Trained Transformers (GPTs) are hyped to revolutionize robotics. Here we question their utility. GPTs for autonomous robotics demand enormous and costly compute, excessive training times and (often) offboard wireless control. We contrast GPT state of the art with how tiny insect brains have achieved robust autonomy with none of these constraints. We highlight lessons that can be learned from biology to enhance the utility of GPTs in robotics.

Recent years have seen major advances in generative Artificial Intelligence due to the development and deployment ofa new architecture; the Generative Pre-Trained Transformer (GPT), or transformer for short<sup>1</sup>. Through adding an attentional mechanism to deep neural networks and deploying on internet-scale training sets, transformers have led to rapid advancements in Large Language Models (LLMs) for natural language processing and generation. Following these early applications, transformers have, alongside other architectures such as diffusion models<sup>2</sup> been applied to the development ofVisual Language Models for text to image and video (e.g.<sup>3</sup>) as well as other multimodal applications (e.g.<sup>4</sup>). These successes have inspired the investigation oftransformer architectures for the robotics domain. The challenges of unstructured multimodal inputs sensed in complicated environments, coupled with high degrees of freedom in robot control, have historically constrained the development ofrobots that are simultaneously generally capable, and robust, in their behaviour. The promise of transformers for robotics appears to be that large-scale training can, through specialisation on further smaller-scale training sets, provide general and adaptable solutions to a wide variety of robotics tasks<sup>5</sup>. Because they can be applied across so many application domains transformer-based approaches have been labelled Foundation Models<sup>5</sup>, indicating their supposed fundamental status but also their incomplete nature. Applications offoundation models to robotics have recently taken off in the minds of developers and researchers.

Transformers have their genesis in large language modelling (LLM). LLMs have also proved to be generalizable and transformative to many applications, but they are not without limitations. As we review below, there are increasingly recognised issues with LLMs in the areas oftraining dataset size, compute resources for training, the financial and ecological costs of both, as well as robustness ofbehavioural output. In this article we question whether transformer architectures are likely to be truly foundational for robotics. We ask whether transformers provide the only or best route towards Artificial General Autonomy, proposing that, unlike ‘intelligence’<sup>6</sup>, the level of autonomy of a robotics system is well-defined, measurable, and economically meaningful.

Drawing on earlier critiques ofGPTs and related approaches, we argue that transformers provide a facsimile of autonomy rather than true autonomy. We then review alternative approaches that have been proposed. The contrast between GPT solutions to autonomous robotics and biological solutions to autonomous behavioural control achieved by animal brains is stark. We explore this contrast to propose what is missing from current GPT approaches, and what could be added in to enhance robust and scalable robot autonomy.

## Progress in applying transformer architectures to autonomy

Transformers have seen rapid application to robot autonomy. As well as high profile commercial announcements and demonstrations, end-to-end solutions to robot autonomy have been developed in the peer-reviewed literature by both academic and industrial groups, to tasks particularly focussing on robot navigation and dexterity (for a review, see<sup>7</sup>).

While the early promise of transformers for robot autonomy seems to be being realised, for a general and scalable solution it is essential to recognise that this technology still comes with significant limitations that will constrain future performance and adoption. While these are active areas of research, and some of these may become less acute as the traditional efficiencies associated with the development and deployment of a novel technology are realised, we argue that there are fundamental structural issues with current transformer architectures, and that these should motivate a longer term search for alternative and complementary approaches, which we review later in this article.

## Training data size and cost requirements are likely to grow

At the heart of the transformer approach to any problem is a scaling requirement. Given the lack of inductive biases these learning systems are highly flexible, however the corollary of this is that their training data requirements are vast. The usual approach for deployment ofa transformerbased foundation model is to train on an internet-scale corpus so that the model acquires multi-modal correspondences and domain knowledge, then further specialise on a smaller training data set for a specific set oftasks. The costs of this are very substantial. Even excluding environmental impacts, state-of-the-art LLMs cost on the order of \$10 s to \$100 s of millions per training episode<sup>8</sup>, although rapid reductions in training and inference costs are being made<sup>9</sup>. For robotics applications, further training for particular tasks such as navigation and manipulation is usually required. The avail ability and cost of acquiring good training datasets is recognised as a major problem. Proposed solutions include the curation ofopen datasets covering multiple tasks and robot types<sup>10</sup>, although currently these can be biased to a relatively small number of tasks. There is also the extensive use of physicsbased simulators to generate training data (e.g.<sup>11</sup>). We argue that, similarly to LLMs, exponentially increasing quantities ofdata are likely to be required to sustain advances in performance<sup>12</sup>. Even for text and multimodal datasets where the internet provides a very large corpus oftraining data ‘for free’, the availability of training data risks becoming a limiting factor<sup>13</sup>. For robotics datasets the costs of collecting useful training data, either physically or through simulation, will be much more acute; replacing physical data col lection with collection via simulation simply trades one kind of resource— experimental time—with another—computational time, albeit with the latter being more scalable. Furthermore, since improvement in transformers’ performance is predicated on increases in scale of training data and weights this problem will only get worse—for example, in multimodal AI ‘zero-shot’ generalization has been shown to have exponential training data requirements<sup>14</sup>.

Compute and infrastructure costs and requirements will persist Once the costs of training a transformer-based architecture are paid, the inference costs at deployment can still be substantial. For example, Meta’s Llama 3.1 has cloud-scale deployments (405bn double-precision parameters). There are reduced size and precision versions suitable for deployment on local GPUs (e.g. 8bn half-precision integer parameters), which can take \~20–100 GB ofmemory for inference. This demands a substantial GPU for even the simplest models being run on a robot<sup>15</sup>, although, as noted above, inference costs have been reduced through the use of unsupervised reinforcement learning during training<sup>9</sup>. While binarisation, quantisation, and other approaches have been used to help design edge AI accelerators for deep and convolutional neural networks (e.g.<sup>16</sup>), the scale ofthe problem for transformers is many orders ofmagnitude larger. For example, for one ofthe longest researched applications ofdeep nets, object detection, one state ofthe art algorithm has on the order of10m-80m network weights<sup>15</sup>, compared to the 8bn-405bn weights mentioned above for a state-of-the-art LLM. This represents a four orders of magnitude difference in scale, even before the additional requirements oftraining a transformer for robotics tasks are taken into account. Hence there is ver active research into methods to avoid the cloud compute bottleneck, including utilisation ofnovel technologies such as 6G<sup>17</sup>. Moore’s law and the advent ofnovel parallel compute architectures has traditionally saved AI, and computer software more generally. For foundation models, however, we argue that although available compute can be scaled exponentially, the exponential requirements for model size and throughput will be in opposition. A real-terms reduction in requirements for compute as performance improvements are sought will only occur when the exponent for the former is greater than the exponent for the latter. However growing evidence suggests we are moving to a post-Moore’s Law world where further innovation in materials is required to make progress [e.g<sup>18</sup>]. Even if compute could still scale faster than data requirements scale, given the potential for ongoing algorithmic improvements<sup>9</sup>, below we argue that there are very many orders ofmagnitude difference between the training that is achievable for a transformer, and the ‘training’ that has genuinely solved autonomy over evolutionary time.

## Hallucinations for transformers in robotics may become acute

As a consequence oftheir statistical training and inference, LLMs are prone to confabulation and hallucination, defined as producing outputs that are inconsistent with user input and/or world knowledge and common sense<sup>19,20</sup>. While such outputs can still be damaging even for a disembodied AI, for example in the social and political arenas<sup>21</sup> when a transformer architecture is embodied the risks are magnified, much as acting on hallucinatory perceptions and impulses in human mental illness can lead to recognized harms to self and others. As with humans, hallucinations may manifest in ways likely to cause harm to the robot or to others, and adversarial attacks on guardrails for transformers in robotics have already been demonstrated<sup>22</sup>. While mitigation ofhallucinations is an ongoing area of research<sup>20</sup>, as others also argue<sup>23</sup> we contend that the fundamentally correlational nature of transformers will render hallucinations inescapable. Failures of reasoning are also inherent in symbolic reasoning by GPTs<sup>24</sup> and chain ofreasoning models<sup>25</sup>. This is likely to require that humans remain in the control loop as teleoperators to ensure robots are remotely supervised, or that robots are isolated from humans, or both. Any ofthese outcomes will of course limit the promised benefits of robotics. As other researchers have argued, these structural issues with statistical approaches to AI are unlikely to find remedy without significant architectural change<sup>26</sup>.

## Transformers give a facsimile of intelligent autonomy

Given the above concerns, why are transformers seeing increasing adoption for robotics? We attribute this to two factors: first, as with LLMs and VLMs, striking earlyadvances have been made in traditionallyverydifficult areas, such as humanoid control, manipulation, and, of course, natural language interfaces. Second, however, we believe a tendency of human observers to anthro pomorphise often leads some ofthem to ascribe abilities, and the potential for understanding, that the architecture does not, and cannot, technically support.

While there are many types of transformer the central motif is a repeating unit composed of a self-attention block followed by a multilayer perceptron block<sup>27</sup> (Fig. 1, right). The control flow is feedforward, while the attention mechanism learns which earlier elements ofthe input to attend to in predicting the next appropriate action. As with LLMs, both the power and generalizability of transformers for robotics comes from their extensive training so that, once trained, they can perform the operation ofmatching an input to a predicted output. In robotics transformers succeed in resolving and executing an action from an input, but this is achieved by interpolation and extrapolation of the training set, with unreliable off-training-set performance<sup>28</sup>. There is no reasoning and no reason why a transformer selects one response over another, other than the selected option carrying the highest predictive weight following training<sup>29</sup>. The same can be said of the language abilities ofLLMs, which have been described as stochastic parrots<sup>30</sup>.

Training and reference to learned experience is an important part of biological autonomous decision making too, but for humans and other animals decision making is also supported by reasoning from models ofhow the world works, how other involved agents should operate, and why the selected action is situation appropriate<sup>31</sup>. Transformers lack these models<sup>24,32</sup>. An autonomous robot’s capacity will be limited by the scope of the training dataset. Since transformers’ responses are unreasoned products of the training data, any transformer-based application cannot justify a decision other than by statistical association to the training data. This poses serious challenges for any form of human / robot interaction. If we were to ask a well-intentioned human coworker why they made an error they would do their best to explain the reasoning behind their actions<sup>33</sup>. If we ask a transformer based robot why it made an error there would be no reasoned answer per se; the answer to the query will have at best a correlation but no causal relationship to the error made, and be subject to hallucination as described above. Reasoning from models ofhow the world works can allow forms ofintrospection and metacognition that can interrogate why a wrong choice has been made, or query wrong decisions before any action is taken. We contend that feedforward transformer-based applications are structurally incapable of reliable metacognition<sup>29,31</sup>

## Alternatives and complements to transformers for autonomy

If transformers are not the full answer, what is? Here we review the main alternative proposals, with an emphasis on our preferred approach: drawing deep inspiration from how the biological brain solves the autonomy problem.

![](images/810f5d79ec9d6063e3352f9701f987e5d435da7c176e6ee47f2ee48c7d5661aa.jpg)  
Fig. 1 | Key differences in structure and function between natural and artificial solutions to autonomy. Left: The brain of a honey bee forager Apis mellifera provides high levels of autonomy integrating multi-modal sensory data to navigate, communicate locations in space, and learn associations between stimuli and rewards, all using fewer than 960,000 neurons. Distinct brain regions specialise in perception including vision (optic lopes, yellow and orange), and olfaction (antennal lobes, blue), and feed into multimodal memory centres (mushroom bodies, red). Sensory and memory pathways converge in the central complex which integrates sensing and learned associations in a single representation of the bee, situated relative to percepts weighted by the bee’s internal state. This is sufficient to resolve competing goals, which drives behaviour directly by interfacing with premotor neurons (not shown). Brain regions are highly differentiated in structure and function according to task demands, and come together in a modular architecture with high degrees of intra-module connectivity but limited and well defined intermodule connections. Image source: insectbraindb.org Right: the generative pretrained transformer (GPT) architecture. Multimodal sensory inputs are embedded in high dimensional space (not shown) then feed into repeated blocks of attentional mechanisms (vellow) followed by feed forward deep networks (blue) with inter mediate normalisation and selection layers (not shown). Each block is hence a very large and non-sparse matrix, with matrix multiplications propagating through the GPT to produce the next output in sequence. Knowledge ofthe task is encoded in the learned values within the matrices, whose total entry counts typically range in the billions to trillions. Thus although total GPT parameters vastly exceed the number of synaptic connections in a simple brain, they are far less robust in behavioural output. Figure adapted from<sup>1</sup>.

## Natural intelligence

The gulf between transformer approaches to robotics and how biological brains produce autonomous behaviour is stark (Fig. 1). Most often comparisons are drawn between LLMs, GPTs and human reasoning<sup>29,31,34</sup>, but the comparison with animal brains and animal reasoning is even more pronounced. For example, the honey bee brain is tiny (just over one cubic millimetre) and contains fewer than one million neurons<sup>35</sup>. The number of synapses in the bee brain is not known, but if we can infer from the Drosophila connectome<sup>36</sup> there will be fewer than half a billion synapses in the bee brain. (Fig. 1, left). Demonstrably, this is all a bee needs to reliably navigate over long (several kilometre) distances, autonomously harvest pollen and nectar from the environment, communicate and coordinate their efforts with their hive mates, and perform all the many jobs needed to build and maintain their colony, including raising the next generation. They can solve complex foraging economics problems, majoring on the resources their colony needs and harvesting them from cryptic and ephemeral flowers patchily distributed in the environment<sup>37</sup>. Bees are able to fly with no practice, and just twenty minutes ofstructured flight time around the hive is enough for them to be able to navigate proficiently in their environment<sup>38</sup>. The contrast with the prolonged training needed by transformers could not be greater. The power consumption ofa bee brain as it performs entirely on board autonomous decision making is infinitesimal compared to any GPT. In contrast to transformers, animal brains have been massively ‘pre-trained on a planetary scale, to use minimal information and generate a very wide variety of behaviours (Fig. 2). It is trivial to observe that the scale of this evolutionary pre-training, spanning very many trillions of instantiations of tens of millions of different species, across hundreds of millions of years, cannot be matched by computational approaches; even ifit could, arguably we do not have a sufficiently robust and evolvable representation to match the genetic language that encodes for body and brain morphology and behaviour in nature<sup>39</sup>. But the greater point is this: we don’t need to match the process by which bee intelligence evolved if we want to match the performance of that evolved intelligence. That can be done by studying just the end point of the evolutionary process – the embodied brain.

How, then, does the humble bee outperform transformers in compute, energetic cost, and training time? In a word—structure. The generalisability of transformers, and arguably their elegance, is because before pre-training they are not structurally differentiated according to function. The insect brain, by contrast is a case study in structure-function specialization. The insect brain is subdivided into modules (Fig. 1, left). Each module is specialised for processing different domains of the autonomous decisionmaking challenge. Each specialization in each module exploits the regularities and properties of the information it is processing to reduce compute and increase overall system efficiency. For example, specialized modules in the bee, ant and fly brain process the pattern of polarized light in the sky generated around the sun<sup>40,41</sup>. This is a valuable and robust navigational cue. Its structure is preserved by a topographic processor – the protocerebral bridge in the central complex—which operates as a ring attractor to establish orientation ofthe animal relative to external cue s<sup>40,42,43</sup>. This connects to yet another module which is topographically structured as the azimuth, and can support the relative localization of the insect to external objects<sup>40,41</sup>. The regularities ofthe external world are reflected in how they are represented in the insect brain, which conveys a form of intuitive physics (albeit very different from the type of physics engines used in AI). Olfactory and visual sensory lobes are each specialised to the input properties of their sensory domain. The sensory lobes sharpen, enhance and ultimately compress sensory signals for projection to multimodal sensory integration regions<sup>44</sup> The largest ofthese, the mushroom body, has a structure similar to a threelayer neural network with an expanded middle layer<sup>45–47</sup>. This seems especially adept at multimodal classification.

Insects lack the declarative reasoning of humans, but their reasoning is built around a form ofelementary world model. Insects possess a unitary and coherent representation ofexternal space within which they have a first-person perspective on objects around them<sup>48</sup>. The valence of objects is influenced by the insect’s learned experience with them, as well as innate valence and sub jective physiological state<sup>44</sup>. Differences in valence and location of objects arbitrate the insect’s selection<sup>49–51</sup>. This form ofreasoning might be elementary, but it is still more comprehensible and explicit than the reasonless transfor mers. It is increasingly recognized that AI stands to benefit tremendously from importing concepts and algorithms from insect neuroscience<sup>52,53</sup>.

## Objective AI and world models

Other researchers have proposed that indeed the autonomy abilities of animals (including those ‘simpler’ than humans) should provide inspiration for AI researchers<sup>54</sup>. However, this inspiration is much looser than the Natural Intelligence approach above. While the ‘objective AI’ approach does indeed propose modular AI architectures that correspond with an understanding of the human brain developed in neuroscience, cognitive science, and psychology, the proposal is actually quite different; rather than directly seek to reverseengineer neural circuits in specialist brain modules, instead the idea is to design trainable modules that interface with each other in order to generate more adaptive behaviour than a largely undifferentiated large neural net could be expected to. Thus, for example, rather than directly seek to understand how feature detectors in the early primate visual system function, a feature detector module would be trained. A key part of the proposal is the reintroduction of explicit and configurable worlds models, drawing inspiration from cognitive science; however these also remain trained from data<sup>55</sup>.

## Hybrid approaches

Still other researchers, drawing on a long running proposal but also gaining renewed motivation from contemporary developments in AI, have proposed the ‘neurosymbolic approach’<sup>26</sup>. This approach argues that, while deep nets are very suitable for perceptual tasks such as object detection, they are fundamentally unsuited to the symbolic manipulation that is part of reasoning, planning, and decision making. In the context of transformers, this has recently been supported by observations that LLMs fail to robustly deal with and manipulate symbolic knowledge<sup>24,32</sup>. Thus the proposal is to combine the perceptual strengths ofstatistical AI with the causal strengths ofthe older, symbolic, approach to AI. Given the neural bases of symbolic reasoning in the brain are poorly understood, this is a particularly pragmatic approach. In doing so it is hoped that the limitations of the first, symbolic, wave of AI, will be ameliorated by working around the problems they suffered in having sole responsibility for dealing with the perceptual complexity of the real world<sup>56</sup>. We suggest that an even more powerful combination could include the use of Natural Intelligence approaches to perception, and modelling of space and decision option sets within it.

![](images/077a56b73fdc1db82d273956ac7836b357f090ba5072a1cc8f7ece2a3bd94612.jpg)  
Fig. 2 | Key differences in informational requirements for training and deploying natural and artificial solutions to autonomy. Transformer-based approaches to autonomy rely on internet-scale datasets for training, and process input from a suite of high resolution sensors, such as 4k cameras and LiDar, in order to provide a limited behavioural repertoire in comparison to biological autonomous agents. In contrast, 600 m years of evolution on a planetary scale, with complex physics, has  
encoded blueprints to build autonomous brains into the genome ofa massive variety of animal species. These brains process much sparser input from specialized sensor suites, captured using active perception and behaviour, to generate a hugely rich variety of adaptive behaviours. Image sources: NASA (Earth), insectbraindb.org (honeybee brain, reproduced under https://creativecommons.org/licenses/by/4.0/).

Since transformers may inform our understanding of aspects of ‘higher’ cognitive function in real brains (see ‘Natural Intelligence’ above), they could still have great value informing a component offull autonomous system stacks grounded on basal mechanisms derived from natural brains, rather than their foundations themselves. Just as activity and learned filters in deep neural networks show interesting corollaries with function of natural brains<sup>56</sup>, transformers do appear to capture some interesting fundamental aspects of language and visual recapitulation<sup>57</sup> (but see<sup>58</sup> and<sup>59</sup>), and hence may form a component of a fully autonomous system grounded on firmer foundations.

## Conclusion

Transformer architectures have brought to robotics the rapid progress that they had already brought to natural language and muli-modal AI. However, there are reasons to continue the search for solutions to the robotics autonomy problem. Transformer architectures treat the world in purely statistical terms, albeit grounded in perceptual inputs. This was arguably a deliberate choice in response to the ‘bitter lesson’<sup>60</sup>, that inductive biases in AI have historically failed<sup>56</sup>. However, this results in an autonomy solution very different to the way the only truly autonomous artefact known to humanity, the biological brain, functions. Here we have highlighted this, and conclude by arguing that the tremendous recent advances in data on, and understanding of, a variety of brains, means the time is ripe to revisit the ‘bitter lesson’, and see what new lessons for AI can be learned from their study.

## Data availability

No datasets were generated or analysed during the current study

Received: 27 November 2024; Accepted: 31 March 2025; Published online: 06 May 2025

## References

1. Vaswani, A. et al. Attention is all you need. In Proceedings of the 31st International Conference on Neural Information Processing Systems 6000–6010 (Curran Associates Inc., Long Beach, California, USA 2017).

2. Croitoru, F. A., Hondru, V., Ionescu, R. T. & Shah, M. Diffusion Models in Vision: A Survey. IEEE Trans. Pattern Anal. Mach. Intell. 45, 10850-10869 (2023).

3. Wang, Z. & Sim, V. L. M. Simple visual language model pretraining with weak supervision. https://doi.org/10.48550/arXiv.2108.10904 (2022).

4. Tsai, Y. H. et al. Multimodal transformer for unaligned multimodal language sequences. Proc. Conf. Assoc. Comput Linguist Meet. 2019, 6558–6569, https://doi.org/10.18653/v1/p19-1656 (2019).

5. Bommasani, R. et al. On the opportunities and risks of foundation models. ArXiv (PREPRINT) abs/2108.07258 (2021).

6. Mitchell, M. Debates on the nature of artificial general intelligence. Science 383, eado7069 (2024).

7. Hu, Y. et al. Toward general-purpose robots via foundation models: a survey and meta-analysis. https://ui.adsabs.harvard.edu/abs/ 2023arXiv231208782H (2023).

8. Buchholz, K. The Extreme Cost Of Training AI Models. Forbes (2024).

9. DeepSeek-A. I. et al. DeepSeek-V3 Technical Report. https://doi.org 10.48550/arXiv.2412.19437 (2024).

10. Khazatsky, A. et al. DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset. https://ui.adsabs.harvard.edu/abs/ 2024arXiv240312945K (2024).

11. Makoviychuk, V. et al. Isaac Gym: High Performance GPU-Based Physics Simulation For Robot Learning. arXiv:2108.10470 (2021).

12. Hoffmann, J. et al. Training Compute-Optimal Large Language Models. https://doi.org/10.48550/arXiv.2203.15556 (2022).

13. Villalobos, P. et al. Will werun out ofdata?Limits ofLLMscaling based on human-generated data. Proceedings of the 41st International Conference on Machine Learning, Vol. 235, 49523-49544 (PLMR, 2024).

14. Udandarao, V. et al. No “Zero-Shot” Without Exponential Data: Pretraining Concept Frequency Determines Multimodal Model Performance. ArXiv abs/2404.04125 (2024).

15. Schmid, P. et al. Llama 3.1 - 405B, 70B & 8B with multilinguality and long context. https://huggingface.co/blog/llama31 (2024).

16. Hubara, I. et al. Binarized neural networks. 30th Conference on Neural Information Processing Systems (NIPS 2016), (Curran Associates Inc., Barcelona, Spain 2016).

17. Qu, G. et al. Mobile edge intelligence for large language models: A contemporary survey. https://doi.org/10.48550/arXiv.2407.18921 (2024).

18. Kim, K. S. et al. The future of two-dimensional semiconductors beyond Moore’s law. Nat. Nanotechnol. 19, 895–906 (2024).

19. Li, Y. et al. Evaluating Object Hallucination in Large Vision-Language Models. https://doi.org/10.48550/arXiv.2305.10355 (2023).

20. Zhang, Y. et al. Siren’s Song in the AI Ocean: A Survey on Hallucination in Large Language Models. https://doi.org/10.48550 arXiv.2309.01219 (2023).

21. Jacob, C., Kerrigan, P. & Bastos, M. T. The chat-chamber effect: Trusting the AI hallucination. Big Data & Society 12 https://doi.org/10. 2139/ssrn.5033125 (2025).

22. Robey, A., Ravichandran, Z., Kumar, V., Hassani, H. & Pappas, G. J. Jailbreaking LLM-controlled robots. https://doi.org/10.48550/arXiv. 2410.13691 (2024).

23. Xu, Z., Jain, S. & Kankanhalli, M. Hallucination is inevitable: An innate limitation of large language models. https://doi.org/10.48550/arXiv. 2401.11817 (2024).

24. Mirzadeh, I. et al. GSM-Symbolic: Understanding the limitations of mathematical reasoning in large language models. https://doi,org/10 48550/arXiv.2410.05229 (2024).

25. Li, B. et al. Deceptive semantic shortcuts on reasoning chains: How fa can models go without hallucination? In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (1, 7675–7688) Mexico City, Mexico. (Association for Computational Linguistics 2024).

26. Marcus, G. The next decade in AI: Four steps towards robust artificial intelligence. https://doi.org/10.48550/arXiv.2002.06177 (2020).

27. Dufter, P., Schmitt, M. & Schütze, H. Position information in transformers: An overview. Comput. Linguist. 48, 733–763 (2022).

28. Bommasani, R. et al. On the opportunities and risks of foundation models arXiv:2108.07258 (2021).

29. Milliere, R. & Buckner, C. A philosophical introduction to language models - Part I: Continuity with classic debates. ArXiv (PREPRINT) abs/2401.03910 (2024).

30. Bender, E. M., Gebru, T., McMillan-Major, A. & Mitchell, M. On the dangers of stochastic parrots: Can language models be too big? In Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency 610–623 (Association for Computing Machinery, Virtual Event, Canada, 2021).

31. Lake, B. M., Ullman, T. D., Tenenbaum, J. B. & Gershman, S. J. Building machines that learn and think like people. Behav. Brain Sci. 40, e253 (2017).

32. Momennejad, I. et al. Evaluating Cognitive Maps and Planning in Large Language Models with CogEval. https://doi.org/10.48550/ arXiv.2309.15129 (2023).

33. Davidson, D. Actions, reasons, and causes. J. Philos. 60, 685 (1963).

34. Momennejad, I. A rubric for human-like agents and NeuroAI. Philos. Trans. Roy. Soc. B 378, 20210446 (2023).

35. Perry, C. J., Barron, A. B. & Chittka, L. The frontiers of insect cognition. Curr. Op. Behav. Sci. 16, 111–118 (2017).

36. Dorkenwald, S. et al. Neuronal wiring diagram of an adult brain. Nature 634, 124–138 (2024).

37. Seeley, T. D. The wisdom of the hive: the social physiology of honey bee colonies. (Harvard University Press, 1995).

38. Capaldi, E. A. et al. Ontogeny of orientation flight in the honeybee revealed by harmonic radar. Nature 403, 537–540 (2000).

39. Sniegowski, P. D. & Murphy, H. A. Evolvability. Curr. Biol. 16, R831–R834 (2006).

40. Pfeiffer, K. & Homberg, U. Organization and functional roles of the central complex in the insect brain. Annu Rev. Entomol. 59, 165–184 (2014).

41. Plath, J. A. & Barron, A. B. Current progress in understanding the functions of the insect central complex. Curr. Opin. Insect Sci. 12, 11–18 (2015).

42. Cope, A. J., Sabo, C., Vasilaki, E., Barron, A. B. & Marshall, J. A. R. A computational model of the integration of landmarks and motion in the insect central complex. PLoS ONE 12, https://doi.org/10.1371/ journal.pone.0172325 (2017).

43. Turner-Evans, D. B. & Jayaraman, V. The insect central complex. Curr Biol. 26, R445–R460 (2016).

44. Galizia, C. G. Olfactory coding in the insect brain: data and conjectures. Eur. J. Neurosci. 39, 1784–1795 (2014).

45. Huerta, R., Nowotny, T., Garcia-Sanchez, M., Abarbanel, H. D. L. & Rabinovich, M. I. learning classification in the olfactory system of insects. Neural Comput. 16, 1601–1640 (2004).

46. Smith, B. H., Huerta, R., Bazhenov, M. & Sinakevitch, I. in Honeybee Neurobiology and Behavior (eds Giovanni C. Galizia, Dorothea Eisenhardt, & Martin Giurfa) 393–409 (Springer, 2012).

47. Bazhenov, M., Huerta, R. & Smith, B. A computational framework for understanding decision making through integration of basic learning rules. J. Neurosci. 33, 5686–5697 (2013).

48. Barron, A. B. & Klein, C. What insects can tell us about the origins of consciousness. Proc. Nat. Acad. Sci. USA 113, 4900–4908 (2016).

49. Krashes, M. J. et al. A neural circuit mechanism integrating motivational state with memory expression in Drosophila. Cell 139, 416–427 (2009).

50. Burke, C. J. et al. Layered reward signalling through octopamine and dopamine in Drosophila. Nature 492, 433–437 (2012).

51. Tsao, C. -H., Chen, C. -C., Lin, C. -H., Yang, H. -Y. & Lin, S. Drosophila mushroom bodies integrate hunger and satiety signals to control innate food-seeking behavior. eLife 7, e35264 (2018).

52. Webb, B. Robots with insect brains. Science 368, 244–245 (2020).

53. de Croon, G. C. H. E., Dupeyroux, J. J. G., Fuller, S. B. & Marshall, J. A. R. Insect-inspired AI for autonomous robots. Sci. Robot. 7, eabl6334 (2022).

54. LeCun, Y. A path towards autonomous machine intelligence version 0.9. 2, 2022-06-27. Open Rev. 1, 1–62 (2022).

55. Jyothir, S. V., Jalagam, S., LeCun, Y. &Sobal, V. Gradient-based Planning with World Models. https://doi.org/10.48550/arXiv.2312.17227 (2023).

56. Summerfield, C. Natural General Intelligence How understanding the brain can help us build AI. 352 (Oxford University Press, 2023).

57. Caucheteux, C., Gramfort, A. & King, J. -R. Deep language algorithms predict semantic comprehension from brain activity. Sci. Rep. 12, 16327 (2022).

58. Digutsch, J. & Kosinski, M. Overlap in meaning is a stronger predictor of semantic activation in GPT-3 than in humans. Sci. Rep. 13, 5035 (2023).

59. Lewis, M. & Mitchell, M. Evaluating the robustness of analogical reasoning in large language models. https://doi.org/10.48550/arXiv. 2411.14215 (2024).

60. Sutton, R. S. The bitter lesson. Incomplete Ideas (blog) 13, no. 1 38 (2019).

## Acknowledgements

We thank Michael Mangan, Sarah Moth-Lund Christensen, Marcel Sayre, the editor and two anonymous reviewers for comment and feedback on the manuscript. This manuscript was partially written when the authors were participants in the Mathematics of Intelligences Long Programme, and Workshop III on Naturalistic Approaches to Intelligence, at the Institute for Pure and AppliedMathematics, UCLA. The authors are grateful for the partial financial support they received from the NSF via the programme to enable their participation. Marshall is supported by the Centre for Machine Intelligence at the University of Sheffield. Barron is supported by funding from the

Templeton World Charity Foundation (TWCF-2020-0539), the Australian Research Council (DP230100006, DP240100400) and the Macquarie University Bioinnovation Initiative.

## Author contributions

J.A.R.M. and A.B.B. conceived of, wrote, and edited the paper.

## Competing interests

J.A.R.M. is Founder Science Officer at and shareholder in Opteran Technologies Ltd. A.B.B. is an unremunerated Academic Advisory Board member at Opteran Technologies Ltd.

## Additional information

Correspondence and requests for materials should be addressed to James A. R. Marshall.

Reprints and permissions information is available at

http://www.nature.com/reprints

Publisher’s note Springer Nature remains neutral with regard tojurisdictional claims in published maps and institutional affiliations.

Open Access This article is licensed under a Creative Commons Attribution 4.0 International License, which permits use, sharing, adaptation, distribution and reproduction in any medium or format, as long as you give appropriate credit to the original author(s) and the source, provide a link to the Creative Commons licence, and indicate if changes were made. The images or other third party material in this article are included in the article’s Creative Commons licence, unless indicated otherwise in a credit line to the material. If material is not included in the article’s Creative Commons licence and your intended use is not permitted by statutory regulation or exceeds the permitted use, you will need to obtain permission directly from the copyright holder. To view a copy of this licence, visit http://creativecommons.org/licenses/by/4.0/.

© The Author(s) 2025