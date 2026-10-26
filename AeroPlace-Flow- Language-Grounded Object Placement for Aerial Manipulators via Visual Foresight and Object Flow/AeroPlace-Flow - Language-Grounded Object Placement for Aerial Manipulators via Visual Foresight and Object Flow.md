the cube"

Hardware Demos

# AeroPlace-Flow: Language-Grounded Object Placement for Aerial Manipulators via Visual Foresight and Object Flow

Sarthak Mishra<sup>∗1</sup>, Rishabh Dev Yadav<sup>∗2</sup>, Naveen Nair<sup>1</sup>, Wei Pan<sup>3</sup>, and Spandan Roy<sup>1</sup>

![](<images/132d94f1aae94c88b197e18ce5d05202a87b4bf5c7201f4a1ee18a57da2acb13.jpg>)  
"Place the cup on the table'  
"Place the cup to the left of  
"Place the tape measure in the toolbox"  
"Place the can on the red X"  
"Place the cup on the purple

Fig. 1: Overview of AeroPlace-Flow. Given a natural language instruction and RGB-D observations of the object and placement scene, our method infers a collision-free object flow for aerial manipulation in 3 main steps. (1) Visual Foresight: A language-conditioned image editing model generates a goal image of the scene with the object placed according to the instruction. (2) Object Flow Extraction: The generated image is converted into a metrically consistent 3D scene, contact footprints are estimated, and the original object geometry is used to compute a collision-free object flow trajectory. (3) Placement Execution: The aerial manipulator tracks the inferred object flow to execute the placement. Bottom: Hardware demonstrations of language-conditioned aerial placement tasks in diverse scenarios. \*Cable connected to drone is only for supplying power.

Abstract— Precise object placement remains underexplored in aerial manipulation, where most systems rely on predefined target coordinates and focus primarily on grasping and control. Specifying exact placement poses, however, is cumbersome in real-world settings, where users naturally communicate goals through language. In this work, we present AeroPlace-Flow, a training-free framework for language-grounded aerial object placement that unifies visual foresight with explicit 3D geometric reasoning and object flow. Given RGB-D observations of the object and the placement scene, along with a natural language instruction, AeroPlace-Flow first synthesizes a task-complete goal image using image editing models. The imagined configuration is then grounded into metric 3D space through depth alignment and object-centric reasoning, enabling the inference of a collision-aware object flow that transports the grasped object to a language and contact-consistent placement configuration. The resulting motion is executed via standard trajectory tracking for an aerial manipulator. AeroPlace-Flow produces executable placement targets without requiring predefined poses or taskspecific training. We validate our approach through extensive simulation and real-world experiments, demonstrating reliable language-conditioned placement across diverse aerial scenarios with an average success rate of 75% on hardware.

Website: https://aeroplaceflow.github.io/

## I. INTRODUCTION

Aerial manipulators (AMs) extend the reach of robotics to cluttered, elevated, and hard-to-access environments, enabling physical interaction beyond the workspace limitations of ground robots [1]. Prior work such as [2]–[5] highlight the promise of AMs for inspection, infrastructure maintenance, warehouse logistics, and disaster response. While grasping and control have gained more attention, object placement—which ultimately determines task completion—remains comparatively underexplored. Existing aerial manipulation pipelines typically formulate placement as as a two-stage process: manually specifying a 3D pose - where, followed by executing the motion required to realize that pose and coarsely releasing the object - how. Such interfaces require users to provide precise metric coordinates in the reference frame, which is cumbersome and unintuitive. As aerial systems move toward greater autonomy, the need for more natural interfaces becomes critical. Language offers a scalable and intuitive medium for communicating goals, allowing users to specify tasks such as “place the object on the rack” without enumerating exact positions. These observations motivate a focused study of language grounded object placement in aerial settings.

Progress is seen in robotic manipulation to incorporate language via large generative models [6]–[8]. Inspired by human capabilities, prior works [9]–[11] demonstrate that generative models, which enjoy capabilities like visual foresight, can guide manipulation through synthesized subgoal images or predictive visual rollouts. In parallel to these image-based approaches, [12], [13] leverage generative video models combined with object flow to extract task-relevant motion trajectories and enable continuous manipulation in tasks such as drawer opening or cloth folding. However, transferring these approaches directly to aerial manipulators to solve placement tasks remains challenging and largely unexplored.

In the context of AMs, the primary requirement is to determine where the object should be placed in the scene given a language instruction, after which the system can focus on how to place it. We find that off-the-shelf image editing models offer a practical way to generate visual foresight that is transferrable to AM’s without training. Once the desired configuration is inferred, we turn to infer object flow to capture how the object should move through space to reach its target configuration while avoiding collisions.

Building on these insights, we introduce AeroPlace-Flow, as shown in Fig. 1, a framework for language-conditioned object placement with aerial manipulators that combines visual foresight generation with metric object flow inference. Given a language instruction, an observation of the object before grasping, and an observation of the placement scene, AeroPlace-Flow first leverages off-the-shelf image editing models to synthesize a goal image depicting the completed task. This generated image serves as a semantic placement hypothesis that specifies where the object should be placed. We recover a metrically consistent 3D scene from the generated image and infer a collision-free object flow that transports the object from its current configuration in the gripper to the desired placement configuration. Finally, the inferred flow is executed by the aerial manipulator using standard trajectory tracking.

We evaluate AeroPlace-Flow on a benchmark of 100 language-conditioned placement tasks spanning tabletop, relative positioning, stacking, and shelf scenarios, and further demonstrate the approach on a real aerial manipulator platform. Experimental results show that language-conditioned visual foresight provides a reliable semantic interface for placement specification, while the proposed object flow inference module recovers accurate metric placement trajectories with a success rate of 80% on the benchmark and transfer successfully to hardware execution with 75% success rate. Our main contributions are summarized as follows:

• We propose visual foresight for aerial object placement, enabling off-the-shelf image editing models to synthesize goal scenes from language instructions.

• We propose a method to recover collision-free 3D object flow from generated goal images by enforcing metric consistency, estimating contact footprints, and optimizing trajectories in object space.

• We construct a benchmark of 100 language-conditioned placement tasks to systematically evaluate visual foresight generation and object flow inference for placement.

• We demonstrate real-world aerial manipulation experiments showing that the inferred object flows can be executed on hardware to achieve successful placements.

## II. RELATED WORK

Aerial Manipulation and Control: Aerial manipulation has advanced in modeling, perception, and control, with key design challenges summarized in [1]. Agile grasping with onboard perception has been demonstrated in [2], [3], following earlier vision-guided aerial interaction systems [14]. Foundational quadrotor control methods include [15], [16], while learning-augmented MPC and data-driven dynamics compensation improve robustness [17], [18]. However, most systems still assume predefined metric placement goals. Residual dynamics learning has reduced dependence on exact analytical models [19]–[25].

Language-Grounded Robotic Manipulation: Language provides a natural interface for robotic task specification. SayCan [6], RT-2 [8], and PaLM-E [26] combine visionlanguage models with control, while CLIPort [7], PerAct [27], and Diffusion Policy [28] learn language- and visionconditioned visuomotor policies. Grasp prediction has also progressed through [29]–[31]. These methods primarily target ground manipulators, although recent aerial systems use language and vision for grasping and skill selection [32]– [34].

Visual Foresight and Generative Subgoal Imagination: Visual foresight predicts future observations for control [11], [35], while world models learn latent predictive environment representations [36]. More recent approaches use generative models to synthesize language-conditioned goal images or subgoals [9], [10].

Object Flow and Learning-Based Placement: Dream2Flow [12] and NovaFlow [13] convert generated visual dynamics into object-centric motion. Pick2Place [37] and AnyPlace [38] address task-aware placement and rearrangement for fixed-base manipulators. These methods, however, remain focused on ground manipulation and do not directly address language-conditioned aerial placement.

## III. METHODOLOGY

We first introduce the problem formulation and overview of AeroPlace-Flow in Sec. III-A. We use off-the-shelf image editing models for generating visual foresight in Sec. III B, then extract metric collision-free object flow in Sec. III-C, and discuss how to realize this displacement through standard trajectory tracking methods for aerial manipulation in Sec. III-D.

## A. Problem Formulation and Overview

Given a natural language instruction $L ,$ RGB-D observations of the object $( I _ { o b j } , D _ { o b j } )$ and the placement scene $( I _ { s c e n e } , D _ { s c e n e } )$ , camera intrinsics $K ,$ and corresponding camera extrinsics $T _ { o b j } ^ { c a m }$ and $T _ { s c e n e } ^ { c a m }$ , the objective is to infer a semantically correct and metrically consistent object flow $\mathbf { P } _ { 1 : T } \in \mathbb { R } ^ { T \times \bar { N } \times 3 }$ , that transports the object from its current gripped configuration to a desired placement configuration, effectively reducing the object placement task to a trajectory tracking problem.

Generating Visual Foresight: We leverage off-the-shelf language-conditioned image editing models to generate an image of scene with the completed task. The model jointly reasons over the language instruction, object and placement scene observations, producing a goal image where the object is placed according to the instruction.

Inferring Object Flow: We decompose object flow inference into three steps. First, we recover a metrically consistent 3D scene for the generated image in a world frame. Second, we estimate the contact footprint between the generated object and environment . Finally, we swap the generated object’s geometry with original object geometry with the help of the inferred contact footprint to remove inconsistencies and use point correspondences to get a linear object path between the object held in the gripper and the placed object. This is optimized with collision and smoothness constraints to obtain a collision-free object flow.

Placement Execution: Given the object flow $\mathbf { P } _ { 1 : T }$ , the task is reduced to trajectory tracking problem. The flow specifies the desired time-indexed 3D motion of the object geometry, which is executed by the aerial manipulator while maintaining rigid attachment between the object and gripper. Standard control modules are used to track the requested motion.

## B. Generating Visual Foresight

Given the language instruction $L ,$ the object observation $I _ { o b j }$ , and the placement scene observation $I _ { s c e n e } ,$ , we use an image editing model $\pi$ with multi-image-context to generate a goal image $I _ { g e n }$ . The generation process as shown in Fig.2, can be written as:

$$
I _ {g e n} = \pi (I _ {o b j}, I _ {s c e n e}, P),\tag{1}
$$

where $P$ is a text prompt constructed from the language instruction and task constraints. The prompt $P$ encodes four constraints: (1) the object must be placed according to the instruction L, (2) the image should be generated from the same camera fov as in $I _ { s c e n e } ,$ (3) the global scene layout of $I _ { s c e n e }$ must remain unchanged except for the added object. (4) the object must be placed with an orientation consistent with it appearance in $I _ { o b j }$ . The model jointly reasons over the object appearance from $I _ { o b j }$ and the spatial layout of $I _ { s c e n e }$ to produce a goal image that depicts the completed task.

As discussed in [37], the selected grasp strongly influences the set of feasible object placements. In our setting, we do not have explicit feedback about the object’s exact orientation within the gripper at execution time. Therefore, we assume that the object is stably held in an orientation consistent with its observation $I _ { o b j }$ . Constraint (4) operationalizes this assumption by enforcing orientation consistency between the observed object image and the generated goal image.

We observe that image editing models reliably reposition objects when both visual and textual context are sufficiently informative. However, inconsistencies may arise with object scale in $I _ { g e n }$ . We correct this in the next stage, where scale consistency is enforced using metric depth information of the object taken from $D _ { o b j }$

![](<images/6c6df85ce0569a2b04f8198241b37070ec6841f1283a99eb654d20397caca205.jpg>)  
Fig. 2: Generating Visual Foresight. Given RGB-D observations of the object $( I _ { o b j } , D _ { o b j } )$ and scene $( I _ { s c e n e } , D _ { s c e n e } )$ with a task instruction $L ,$ an image generation model π produces a goal image $I _ { \mathrm { g e n } }$ depicting the desired final placement.

## C. Inferring Object Flow

The complete process is shown in Fig.3.

Metric 3D scene reconstruction: To enable geometric reasoning in $I _ { g e n }$ with respect to the real world, we recover a metrically consistent depth map for $I _ { g e n }$ using using monocular depth estimation models as $D _ { e s t }$ . Following [12], we align $D _ { e s t }$ to the observed scene metric depth $D _ { s c e n e }$ using a global scale and shif $\operatorname { \mathbf { t } } ( s ^ { * } , b ^ { * } )$ , producing an aligned depth map as: $D _ { g e n } = ( s ^ { * } . D _ { e s t } ) + b ^ { * }$

We segment the object in $I _ { o b j }$ and $I _ { g e n }$ using image segmentation models to obtain object masks. Using the camera intrinsics K and extrinsics $( T _ { o b j } ^ { c a m } , T _ { s c e n e } ^ { c a m } )$ , we project the masked depths $D _ { o b j } , D _ { g e n }$ , and $D _ { s c e n e }$ into 3D, transform them into the world frame, and apply RANSAC-based filtering to remove projection outliers. This yields a set of three point clouds:

$\mathcal { P } _ { o b j } \{$ : the metric object geometry reconstructed from $( I _ { o b j } , D _ { o b j } )$ and its masks.

$\mathcal { P } _ { o b j - g e n } \colon$ the placed object reconstructed from $( I _ { g e n } , D _ { g e n } )$ , providing a semantically correct placement hypothesis but whose geometry may not exactly match the real object.

![](<images/903a1589e34bca74b4cc9628d0f62710f241f141f3db8490601a0b554911f0b1.jpg>)  
Fig. 3: Object flow inference and placement execution. Given the generated goal image $I _ { g e n } ,$ , we first recover a metrically consistent 3D scene and extract point clouds for the object $\mathcal { P } _ { o b j }$ , generated object $\mathcal { P } _ { o b j - g e n } ,$ and world ${ \mathcal { P } } _ { w o r l d } .$ A contact footprint between $\mathcal { P } _ { o b j - g e n }$ and ${ \mathcal { P } } _ { w o r l d }$ is estimated to identify the support region. The original object geometry $\mathcal { P } _ { o b j }$ is then aligned with the current gripper pose and virtually placed on the contact footprint to obtain the desired placement configuration. Known point correspondences between the gripped and placed object are used to generate an initial linear object flow, which is refined through optimization with collision and smoothness constraints to produce a collision-free trajectory. The resulting object flow $\mathbf { P } _ { 1 : T }$ is executed by the aerial manipulator using standard trajectory tracking.

${ \mathcal { P } } _ { w o r l d } .$ the scene point cloud reconstructed from $( I _ { s c e n e } , D _ { s c e n e } )$ , providing the metric geometry of the environment.

Contact footprint estimation: As mentioned above, the generated object does not preserve the objects geometry and size, therefore directly using $\mathcal { P } _ { o b j - g e n }$ can therefore lead to physically inconsistent placements. However, the generated object is typically positioned at the correct semantic location in the scene.

We combine this location information with the metric objects geometry in $\mathcal { P } _ { o b j }$ by first estimating the contact footprint between the world $\mathcal { P } _ { w o r l d }$ and $\mathcal { P } _ { o b j - g e n } .$ The lowest surface points of $\mathcal { P } _ { o b j - g e n }$ are extracted by heightbased thresholding and act as contact candidates. For each candidate, distance and tolerance criteria are applied to identify the corresponding support surface in Pworld. The resulting set defines a dense contact footprint in the world frame, representing the support region between the object and world.

Object flow computation: If the object consists of N points, the object flow is defined as the time-indexed 3D trajectory of all object points: $\mathbf { P } _ { 1 : T } \in \mathbb { R } ^ { T \times N \times 3 }$ , where each slice $\mathbf { P } _ { t } \in \mathbb { R } ^ { N \times 3 }$ represents the object geometry at time step t. We refer to $\mathbf { P } _ { 1 : T }$ as the 3D object flow.

$\mathcal { P } _ { o b j }$ represents the object kept at a position before grasping, we transform it to the current gripper’s position by using the AM’s pose and arm joint angles feedback to get $\mathcal { P } _ { o b j - g r i p } .$ . Next, we align and place a copy of $\mathcal { P } _ { o b j - g r i p }$ on the contact footprint’s center essentially replacing $\mathcal { P } _ { o b j - g e n }$ with $\mathcal { P } _ { o b j - g r i p }$ .

We now have two rigidly related configurations of identical point sets: one in the gripper and one at the placement location. Since the points are copies, point-wise correspondences are known. We first construct a straight-line interpolation between the two configurations to obtain an initial object flow connecting the start and goal poses.

The linear interpolation provides a coarse displacement but does not account for environmental collisions or motion smoothness. Therefore, we refine this initial path using trajectory optimization with collision avoidance and smoothness constraints. In particular, we employ a sequential convex optimization approach similar to TrajOpt [39], where the object trajectory is iteratively adjusted to maintain clearance from the scene geometry represented by $\mathcal { P } _ { w o r l d }$ while ensuring smooth motion between consecutive configurations.

The optimized sequence of intermediate configurations transports the object from the current gripper pose to the desired placement configuration while respecting geometric constraints. This sequence forms the final object flow $\mathbf { P } _ { 1 : T }$ used for execution.

## D. Placement Execution

Given the collision-aware object flow $\mathbf { P } _ { 1 : T }$ , we extract an aerial manipulation trajectory that realizes the desired transport while respecting platform stability and kinematic feasibility as shown in Fig.3. The object flow specifies the desired time-indexed trajectory of all object points in the world frame. Since the object is rigidly attached to the gripper during transport, these point trajectories implicitly determine the motion that the end-effector must follow.

![](<images/d4799653df2c50d648b37bc0ab5eb3013a66dba97ce768dbff959c50023b1bd1.jpg>)  
Fig. 4: Representative examples from the 100-task benchmark evaluation. Each row illustrates the full AeroPlace-Flow pipeline for a language-conditioned placement task. From left to right: input object and scene images, generated visual foresight from the image editing model, reconstructed 3D scene, estimated contact footprint(colored in green), resulting placed object pose, and the inferred collision-free object flow trajectory.

We operate the aerial manipulator in Cartesian end-effector space. For each step in $\mathbf { P } _ { 1 : T }$ , a target end-effector configuration is computed and tracked through coordinated motion of the drone base and arm joints. The controller ensures that the commanded end-effector configuration is achieved while maintaining stable flight.

## IV. EXPERIMENTATION

We evaluate AeroPlace-Flow to answer the following questions: 1) Can language-conditioned image editing serve as a reliable interface for extracting 3D object flow and object placement? 2) Where do failures occur within the pipeline? 3) How does the choice of generative model affect performance of the pipeline? 4) Does the inferred object flow transfer to successful aerial execution?

In the next sections, we first perform evaluation of the visual foresight generation and object flow inference modules on a custom benchmark, examples of which are shown in Fig.4, and then the complete end to end pipeline on a hardware setup and discuss the various results.

## A. Implementation Details

Unless mentioned otherwise, Google Nano Banana Pro model via API is used for generating images, equipped with the following prompt structure:

”Place the from the first image on the in the second image, generate the image from exactly the same camera angle as in the second image, do not change anything else in the image, place the with the same orientation as in the 1st image”.

The prompt is edited slightly to add task details such as object and scene name. DepthAnythingV3[40] model is used for monocular depth estimation, SAM3[41] is used for generating the object relevant masks.

Aerial Manipulator Hardware: A custom made Tarot 650 quadcopter is used for testing, equipped with a CUAV X7+ flight controller running PX4 flight software, ZED depth camera, NVIDIA Jetson nano super as the companion computer. The experiments were performed in a indoor lab with motion capture system. A custom 3DOF 3d-printed manipulator with gripper was attached to the frame of the quadcopter. ROS2 humble with python nodes was used to control the aerial manipulator. A companion laptop is used to prompt the system and run the depth estimation and segmentation models locally.

## B. Benchmark and Evaluation

We construct a benchmark of 100 language-conditioned object placement tasks to evaluate the visual foresight and object flow inference pipeline. The data is collected, as shown in Fig. 5, from real-world RGB-D observations in a laboratory environment and covers diverse object categories spanning variations in size, symmetry, and support footprint. A total of 20 objects and 20 scene images are collected. These are then paired in various combinations to create the 100 tasks. Tasks are grouped into the following placement classes:

TABLE I  
Visual foresight generation performance across 100 language-conditioned placement tasks.

<table><tr><td rowspan="2">Scenario</td><td colspan="3">Nano Banana Pro</td><td colspan="3">Qwen-Image-Edit</td><td colspan="3">FLUX.1 Kontext</td><td colspan="3">GPT-Image</td><td colspan="3">Omni-Gen2</td></tr><tr><td>SE↑</td><td>IP↓</td><td>HA↓</td><td>SE↑</td><td>IP↓</td><td>HA↓</td><td>SE↑</td><td>IP↓</td><td>HA↓</td><td>SE↑</td><td>IP↓</td><td>HA↓</td><td>SE↑</td><td>IP↓</td><td>HA↓</td></tr><tr><td>Tabletop</td><td>23</td><td>2</td><td>0</td><td>22</td><td>2</td><td>1</td><td>21</td><td>3</td><td>1</td><td>20</td><td>3</td><td>2</td><td>18</td><td>4</td><td>3</td></tr><tr><td>Shelf</td><td>22</td><td>2</td><td>1</td><td>21</td><td>3</td><td>1</td><td>20</td><td>3</td><td>2</td><td>18</td><td>4</td><td>3</td><td>16</td><td>5</td><td>4</td></tr><tr><td>Stacking</td><td>21</td><td>3</td><td>1</td><td>20</td><td>3</td><td>2</td><td>18</td><td>4</td><td>3</td><td>17</td><td>4</td><td>4</td><td>14</td><td>6</td><td>5</td></tr><tr><td>Relative</td><td>22</td><td>2</td><td>1</td><td>21</td><td>3</td><td>1</td><td>19</td><td>4</td><td>2</td><td>18</td><td>4</td><td>3</td><td>15</td><td>5</td><td>5</td></tr><tr><td>Overall (100)</td><td>88</td><td>9</td><td>3</td><td>84</td><td>11</td><td>5</td><td>78</td><td>14</td><td>8</td><td>73</td><td>15</td><td>12</td><td>63</td><td>20</td><td>17</td></tr><tr><td>Avg Runtime (sec)</td><td></td><td>12.5</td><td></td><td></td><td>11.5</td><td></td><td></td><td>14.5</td><td></td><td></td><td>11.0</td><td></td><td></td><td>21.0</td><td></td></tr></table>

(i) Tabletop placement: placing the object at specified locations on a planar surface (e.g., center, corner, marked region). (ii) Relative positioning: placing the object next to, in front of, or between reference objects.

(iii) Stacking: placing the object on top of another object.

(iv) Shelf placement: placing the object onto elevated or constrained support regions.

Objects  
Scenes  
Ground truth final state  
![](<images/6673ac4bd5dfbc5f859e6aa58b6e7231d38dc137b83ae31306d6e2f4b00f77d3.jpg>)  
Fig. 5: Overview of the custom benchmark. Each row illustrates a representative task instance with (left) the target object, (middle) the scene configuration, and (right) the corresponding ground-truth final state after placement.

## C. Evaluation: Generating Visual Foresight

We evaluate the Visual Foresight generation module by testing five image editing models on the 100-task benchmark: Google Nano Banana Pro, Qwen-Image-Edit, FLUX.1 Kontext, GPT-Image, Omni-Gen2 via APIs where available and local hosting in others. Each model receives the language instruction $L ,$ the object observation $I _ { o b j }$ , and the placement scene observation $I _ { s c e n e } .$ . For each scenario and model, we report the following metrics:

• Image Gen Success (IGS): Number of Successful Edits where the object from $I _ { o b j }$ is placed in $I _ { s c e n e }$ according to the instruction semantics.

• Incorrect Placement (IP): Number of failures due to Incorrect Placement position or orientation.

• Hallucination (HA): Number of failures caused by Hallucinations, including introducing new objects, removing existing objects, or altering scene structure.

Additionally, we also report the Average Runtime in seconds for the models. Results for this are shown in I.

## D. Evaluation: Object Flow Inference

We evaluate the object flow inference module only for all the successful generations occurred for all the Nano banana model in the previous section which is 88. Ground truth object poses are available from manual data collection. For each task, we compute the inferred object flow and compare the final placement pose with the ground truth pose. For each scenario, we report the following metrics:

• Flow Extraction Success (FES): Percentage of trials where a Valid object Flow is generated and reaches the target support region without collisions.

• Centroid Pose Error (CPE): Error between the pose of inferred final object centroid and the ground truth centroid.

Results for this are shown in II.

TABLE II  
Object flow inference performance evaluated on the 88 successful visual foresight generations from Nano Banana Pro.

<table><tr><td>Scenario</td><td>Flow Success ↑</td><td>Centroid Pose Error ↓</td></tr><tr><td>Tabletop (23)</td><td>22</td><td>1.8</td></tr><tr><td>Shelf (22)</td><td>20</td><td>2.3</td></tr><tr><td>Stacking (21)</td><td>18</td><td>3.1</td></tr><tr><td>Relative (22)</td><td>20</td><td>2.5</td></tr><tr><td>Overall (88)</td><td>80</td><td>2.4</td></tr></table>

## E. Evaluation: End-to-End Aerial Execution

We sample 20 tasks from the benchmark suitable for the lab environment and AM hardware constraints. For the hardware tests, a trial is successful firstly if visual foresight is successful, then the object flow is correctly inferred and finally if the object is placed at the inferred target location within a tolerance of 5cm without collision and remains stably supported after release. The task only proceeds to the next step if the previous one is successful. For each scenario, we report the following metrics:

• Image Gen Success (IGS): Number of Successful Edits where the object from $I _ { o b j }$ is placed in $I _ { s c e n e }$ according to the instruction semantics.

• Flow Extraction Success (FEC): Number of trials where a valid object flow is generated and reaches the target support region without collisions.

• Placement Success (PS): Number of trials where the object is physically placed at the inferred target location within the tolerance without collision and remains stably supported after release.

Results for this are shown in III.

TABLE III  
End-to-end aerial execution results

<table><tr><td>Scenario</td><td>Image Gen Success</td><td>Flow Extraction Success</td><td>Placement Success</td></tr><tr><td>Tabletop</td><td>5</td><td>5</td><td>4</td></tr><tr><td>Relative</td><td>5</td><td>4</td><td>4</td></tr><tr><td>Stacking</td><td>4</td><td>4</td><td>4</td></tr><tr><td>Shelf</td><td>4</td><td>4</td><td>3</td></tr><tr><td>Overall</td><td>18/20</td><td>17/18</td><td>15/20</td></tr></table>

## F. Results and Discussion

Can language-conditioned image editing reliably support 3D object flow and placement? Table I shows that modern multi-image editing models generate semantically consistent placement hypotheses across diverse scenarios. Nano Banana Pro performs best, with 88 successful generations out of 100, 9 incorrect placements, and 3 hallucinations. Performance is consistent across tabletop, shelf, stacking, and relative-positioning tasks, although stacking produces slightly more errors because of tighter geometric constraints. These results indicate that language-conditioned visual foresight can preserve object identity and scene structure sufficiently well to provide a reliable semantic prior for geometric reasoning.

Where do failures occur within the pipeline? Comparing Tables I and II shows that most failures arise during geometric reconstruction rather than visual foresight. The main source is inaccurate monocular depth under difficult lighting, texture-poor objects, or low object–background contrast, which corrupts the reconstructed 3D geometry. Fewer failures result from incorrect contact-footprint estimation, typically when reconstruction noise causes inaccurate support-region or contact-point identification. Nevertheless, most successful visual hypotheses yield valid object flows, indicating robustness to moderate perception noise.

How does the generative model affect performance? Table I shows that successful edits decrease from 88 for Nano Banana Pro to 63 for Omni-Gen2, while incorrect placements and hallucinations increase. Hallucinations are particularly harmful because the geometric pipeline assumes object and scene identity are preserved between the inputs and generated image. Models with stronger multi-image consistency and spatial grounding therefore provide more reliable inputs than models optimized mainly for visual realism.

Does inferred object flow transfer to aerial execution? Across 20 hardware trials, visual foresight succeeds in 18 cases, object flow extraction in 17 of those cases, and physical placement in 15 cases within a 5 cm tolerance, yielding a 75% end-to-end success rate. Tabletop and relativepositioning tasks are most reliable, whereas shelf placements are more sensitive to small pose errors because of tighter support regions. Thus, the inferred flows provide sufficiently accurate metric targets for real aerial execution.

Summary. The results validate language-conditioned visual foresight as an effective intermediate representation for aerial placement. The object-flow module recovers valid placement trajectories in 80 of 100 benchmark tasks (80%), while hardware experiments achieve a 75% placement success rate. Together, visual editing, geometric reconstruction, and trajectory execution enable consistent languageconditioned aerial placement across diverse scenarios.

## V. CONCLUSION

We presented AeroPlace-Flow, a training-free framework for language-conditioned aerial object placement that combines visual foresight, metric 3D reasoning, and collisionaware object flow inference. Given a language instruction, an image editing model generates a plausible goal scene specifying where the object should be placed, while geometric reconstruction and trajectory optimization determine how to move it safely from the gripper to the target pose. Evaluation on 100 placement tasks showed that visual foresight provides reliable semantic placement hypotheses and that the proposed grounding module recovers feasible metric object flows in most cases. Hardware experiments further demonstrated effective transfer to aerial execution, achieving a 75% placement success rate under practical perception and flight constraints. These results establish visual foresight as a useful intermediate representation for intuitive aerial manipulation without task-specific training or predefined metric goals. Future work will improve robustness in tightcontact settings, incorporate uncertainty-aware perception, and enable closed-loop replanning during flight.

## ACKNOWLEDGMENT

The authors acknowledge the use of large language models, including Gemini and ChatGPT, solely to improve the manuscript’s grammar and linguistic clarity. As image generation is integral to the proposed methodology, selected figures include demonstration images generated using models such as Nano Banana Pro.

## REFERENCES

[1] A. Ollero, M. Tognon, A. Suarez, D. Lee, and A. Franchi, “Past, present, and future of aerial robotic manipulators,” IEEE Transactions on Robotics, vol. 38, no. 1, pp. 626–645, 2022.

[2] S. Ubellacker, A. Ray, J. Bern, J. Strader, and L. Carlone, “High-speed aerial grasping using a soft drone with onboard perception,” Nature Robotics, 2024, https://rdcu.be/dR0sM.

[3] E. Bauer, B. G. Cangan, and R. K. Katzschmann, “Autonomous marker-less rapid aerial grasping,” arXiv preprint, vol. 2211.13093, 2022, arXiv:2211.13093.

[4] R. D. Yadav, B. Jones, S. Gupta, A. Sharma, J. Sun, J. Zhao, and S. Roy, “An integrated approach to aerial grasping: Combining a bistable gripper with adaptive control,” IEEE/ASME Transactions on Mechatronics, 2025.

[5] R. D. Yadav, S. Dantu, W. Pan, S. Sun, S. Roy, and S. Baldi, “Modular adaptive aerial manipulation under unknown dynamic coupling forces,” IEEE/ASME Transactions on Mechatronics, vol. 30, no. 4, pp. 2688–2698, 2024.

[6] M. Ahn, A. Brohan, D. Chen et al., “Do as i can, not as i say: Grounding language in robotic affordances,” arXiv preprint arXiv:2204.01691, 2022.

[7] M. Shridhar, L. Manuelli, and D. Fox, “Cliport: What and where pathways for robotic manipulation,” in Proceedings of the 5th Conference on Robot Learning (CoRL), 2021.

[8] A. Brohan, D. Chen, T. Drew et al., “Rt-2: Vision-language-action models transfer web knowledge to robotic control,” arXiv preprint arXiv:2307.15818, 2023.

[9] J. Ni et al., “Generate subgoal images before act: Unlocking the chainof-thought reasoning in vision-language-action models,” in CVPR, 2024.

[10] Y. Pang, “Image generation as a visual planner for robotic manipulation,” arXiv preprint arXiv:2512.00532, 2025.

[11] F. Ebert, C. Finn, S. Dasari, A. Xie, A. Lee, and S. Levine, “Visual foresight: Model-based deep reinforcement learning for vision-based robotic control,” arXiv preprint arXiv:1812.00568, 2018.

[12] K. Dharmarajan, W. Huang, J. Wu, L. Fei-Fei, and R. Zhang, “Dream2flow: Bridging video generation and open-world manipulation with 3d object flow,” arXiv preprint arXiv:2512.24766, 2025.

[13] H. Li, L. Sun, Y. Hu, D. Ta, J. Barry, G. Konidaris, and J. Fu, “Novaflow: Zero-shot manipulation via actionable flow from generated videos,” arXiv preprint arXiv:2510.08568, 2025.

[14] K. Kim and et al., “Vision-guided aerial manipulation using a multirotor with a robotic arm,” in IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2016.

[15] D. Mellinger and V. Kumar, “Minimum snap trajectory generation and control for quadrotors,” in ICRA, 2011.

[16] T. Lee, M. Leok, and N. H. McClamroch, “Geometric tracking control of a quadrotor uav on se(3),” in CDC, 2010.

[17] G. Torrente, E. Kaufmann, P. Foehn, and D. Scaramuzza, “Datadriven model predictive control for quadrotors,” IEEE Robotics and Automation Letters (RA-L), vol. 6, no. 2, pp. 3769–3776, 2021.

[18] E. Kaufmann, D. Gehrig, P. Foehn, R. Ranftl, V. Koltun, and D. Scaramuzza, “Deep drone acrobatics,” in Robotics: Science and Systems (RSS), 2020.

[19] W. Cao, A. Capone, R. Yadav, S. Hirche, and W. Pan, “Computationaware learning for stable control with gaussian process,” arXiv preprint arXiv:2406.02272, 2024.

[20] A. Das, R. D. Yadav, S. Sun, M. Sun, S. Kaski, and W. Pan, “Dronediffusion: Robust quadrotor dynamics learning with diffusion models,” in 2025 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2025, pp. 1604–1610.

[21] S. Ujjawal, N. S. Nair, S. P. Singh, R. D. Yadav, W. Pan, and S. Roy, “Learn structure, adapt on the fly: Multi-scale residual learning and online adaptation for aerial manipulators,” arXiv preprint arXiv:2603.11638, 2026.

[22] R. D. Yadav, S. Ujjawal, S. Sun, S. Roy, and W. Pan, “Learning crosscoupled and regime dependent dynamics for aerial manipulation,” arXiv preprint arXiv:2605.14805, 2026.

[23] S. Ujjawal, S. P. Singh, N. S. Nair, R. D. Yadav, W. Pan, and S. Roy, “Aermani-diffusion: Regime-conditioned diffusion for dynamics learning in aerial manipulators,” arXiv preprint arXiv:2512.10773, 2025.

[24] R. D. Yadav, A. Das, H. Song, S. Kaski, and W. Pan, “Arcade: Adaptive robot control with online changepoint-aware bayesian dynamics learning,” arXiv preprint arXiv:2512.14331, 2025.

[25] R. D. Yadav, S. Ujjawal, S. Sun, S. Roy, and W. Pan, “Physics-aware sparse learning and selective online adaptation for euler-lagrange robot dynamics,” arXiv preprint arXiv:2606.09640, 2026.

[26] D. Driess et al., “Palm-e: An embodied multimodal language model,” arXiv preprint arXiv:2303.03378, 2023.

[27] M. Shridhar, L. Manuelli, and D. Fox, “Perceiver-actor: A multitask transformer for robotic manipulation,” in Proceedings of the 6th Conference on Robot Learning (CoRL), 2022.

[28] C. Chi, S. Feng, Y. Du, Z. Xu, E. Cousineau, B. Burchfiel, and S. Song, “Diffusion policy: Visuomotor policy learning via action diffusion,” in Proceedings of Robotics: Science and Systems (RSS), 2023.

[29] J. Mahler et al., “Dex-net 2.0: Deep learning to plan robust grasps,” in RSS, 2017.

[30] H.-S. Fang, C. Wang, M. Gou, and C. Lu, “Graspnet-1billion: A largescale benchmark for general object grasping,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020, pp. 11 444–11 453.

[31] M. Sundermeyer, A. Mousavian, R. Triebel, and D. Fox, “Contactgraspnet: Efficient 6-dof grasp generation in cluttered scenes,” 2021.

[32] S. P. Singh, N. S. Nair, S. Ujjawal, S. Mishra, S. Patil, R. D. Yadav, and S. Roy, “Aerograb: A unified framework for aerial grasping in cluttered environments,” arXiv preprint arXiv:2603.15097, 2026.

[33] S. Mishra, R. D. Yadav, A. Das, S. Gupta, W. Pan, and S. Roy, “Aermani-vlm: Structured prompting and reasoning for aerial manipulation with vision language models,” arXiv preprint arXiv:2511.01472, 2025.

[34] H. Song, R. D. Yadav, C. Guo, and W. Pan, “Soranav: Adaptive uav task-centric navigation via zeroshot vlm reasoning,” arXiv preprint arXiv:2510.25191, 2025.

[35] C. Finn and S. Levine, “Deep visual foresight for planning robot motion,” in 2017 IEEE International Conference on Robotics and Automation (ICRA), 2017, pp. 2786–2793.

[36] D. Ha and J. Schmidhuber, “World models,” arXiv preprint arXiv:1803.10122, vol. 2, no. 3, p. 440, 2018.

[37] Z. He, N. Chavan-Dafle, J. Huh, S. Song, and V. Isler, “Pick2place: Task-aware 6dof grasp estimation via object-centric perspective affordance,” arXiv preprint arXiv:2304.04100, 2023.

[38] Y. Zhao, M. Bogdanovic, C. Luo, S. Tohme, K. Darvish, A. Aspuru-Guzik, F. Shkurti, and A. Garg, “Anyplace: Learning generalizable object placement for robot manipulation,” in Conference on Robot Learning. PMLR, 2025, pp. 4038–4057.

[39] J. Schulman, J. Ho, C. Lee, I. Awwal, H. Bradlow, and P. Abbeel, “Motion planning with sequential convex optimization and convex collision checking,” The International Journal of Robotics Research, 2014.

[40] H. Lin, S. Chen, J. H. Liew, D. Y. Chen, Z. Li, G. Shi, J. Feng, and B. Kang, “Depth anything 3: Recovering the visual space from any views,” arXiv preprint arXiv:2511.10647, 2025.

[41] N. Carion, L. Gustafson, Y.-T. Hu, S. Debnath, R. Hu, D. Suris, C. Ryali, K. V. Alwala, H. Khedr, A. Huang et al., “Sam 3: Segment anything with concepts,” arXiv preprint arXiv:2511.16719, 2025.