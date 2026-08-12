# Open Robotics Datasets for Policy Training

Last researched: **2026-08-11**

This document inventories publicly obtainable datasets that are useful for
training policies evaluated by `sim-benchmarks`. It distinguishes benchmark-native
training data from broader robot pretraining corpora, simulator-generated data,
derived annotations, and datasets described in papers but not actually released.

"Public" does not always mean commercially usable. Some releases are gated or
carry non-commercial licenses. Verify the license of every downloaded dataset and
component before use.

## Evaluation hygiene

Report at least two policy tracks:

1. **Generalization-clean:** use benchmark training splits only. Hold out evaluation
   scenes, seeds, assets, object instances, perturbations, and episode identities.
2. **Benchmark-adapted:** target-benchmark demonstrations are allowed, but exact
   evaluation episodes and identities remain excluded. Report this separately.

Do not treat benchmark validation, hidden-test, perturbation, or evaluation-only
episodes as training data. Before collecting additional simulator data, freeze the
evaluation definitions and seeds.

## Benchmark-native training data

These datasets provide the closest training distribution for the simulation
benchmarks currently planned for this repository.

| Benchmark | Public training data | Scale and format | Recommended use |
|---|---|---|---|
| [RoboCasa](https://github.com/robocasa/robocasa) | Human-50 and Generated-3000 | 1,250 human demonstrations plus approximately 100K generated demonstrations | Strong kitchen manipulation pretraining. Hold evaluation seeds, scenes, and object instances out. |
| [RoboCasa365](https://robocasa.ai/docs/build/html/datasets/datasets_overview.html) | Human and MimicGen pretraining splits; human target splits | 300 pretraining tasks; up to 600K MimicGen demonstrations; 25K target demonstrations | Use `pretrain` for the clean track. Put target-task fine-tuning in the adapted track. See the [dataset usage documentation](https://robocasa.ai/docs/build/html/datasets/using_datasets.html). |
| [NVIDIA GR00T X-Embodiment Sim](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim) | RoboCasa kitchen, GR1 humanoid, Panda bimanual, and Unitree G1 data | 9K cross-embodiment + 240K GR1 + 72K RoboCasa + 102 G1 trajectories; approximately 99 GB | One of the best compact public simulation mixtures. Its 24K downsampled GR1 subset duplicates part of the 240K set. |
| [VLABench](https://huggingface.co/VLABench) | Primitive pretraining/fine-tuning, composite fine-tuning, raw, LeRobot, and RLDS versions | Primitive FT includes 10 tasks x 500 samples; larger pretraining sets are also released | Use primitive pretraining and composite training. Exclude `vlm_evaluation_*` data. |
| [RoboDojo](https://robodojo-benchmark.com/doc/usage/install-and-download/) | Simulation and real-robot demonstrations | LeRobot v3: 120 GB; HDF5 simulation: 523 GB; real: 273 GB; depth: approximately 4.5 TB | The 120 GB LeRobot release is the practical default. Some tasks are explicitly evaluation-only. |
| [Colosseum V2](https://github.com/jstmn/ColosseumV2) | Clean single-arm and bimanual expert trajectories | HDF5 + JSON; reproducible generator included | Train on clean trajectories only. Keep perturbation conditions evaluation-only. |
| [VLA-Arena](https://github.com/PKU-Alignment/VLA-Arena/blob/main/docs/data_collection.md) | Demonstration generator rather than one canonical archive | HDF5 to RLDS to LeRobot pipeline | Freeze evaluation BDDL/task/object variants first, then generate disjoint training episodes. |
| [DOMINO](https://huggingface.co/datasets/H-EmbodVis/DOMINO) | Dynamic-manipulation corpus | More than 110K expert trajectories, 35 tasks, 5 embodiments; approximately 771 GB; Apache-2.0 | Best public dynamic-object training source. Initially download selected embodiment/task shards. |
| [EBench](https://huggingface.co/datasets/InternRobotics/EBench-Dataset) | Training videos and actions | Approximately 327 GB; LeRobot-oriented | Valuable, but its dataset documentation and licensing are incomplete. Never train on validation or hidden-test material. |
| [RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin) | Precollected clean trajectories plus generator | More than 100K trajectories across 50 tasks; XPolicyLab HDF5 with LeRobot conversion | Excellent bimanual source. Official task-selective downloads are supported. |
| [MIKASA-Robo-VLA](https://github.com/CognitiveAISystems/MIKASA-Robo) | Current VLA corpus | 22.5K trajectories, 90 tasks, more than 6M transitions; LeRobot v3/RLDS; approximately 28 GB | Very high value per byte. The dataset repository is [`mikasa-robo/mikasa-robo-vla-lerobot`](https://huggingface.co/datasets/mikasa-robo/mikasa-robo-vla-lerobot). |
| [RoboMME](https://github.com/RoboMME/robomme_benchmark) | Memory-task demonstrations | 1,600 demonstrations across 16 tasks with explicit train/validation/test splits | Train only on the training split. Useful for history-aware policies. |
| [MolmoBot data](https://huggingface.co/datasets/allenai/molmobot-data) | Synthetic Franka and RBY1 tasks | Nine task configurations; train/validation; ODC-BY | Useful for policies evaluated through the AllenAI harness/MolmoSpaces stack. |
| [RoboCerebra Unified](https://huggingface.co/docs/lerobot/main/en/robocerebra) | Long-horizon language-grounded trajectories | 6,660 episodes, 571K frames, 1,728 subtasks | Useful auxiliary long-horizon corpus. The [original raw release](https://huggingface.co/datasets/qiukingballball/RoboCerebra) is approximately 137 GB. |

## Large real-robot corpora

These datasets are not tied to a single benchmark, but are important ingredients
in recent general-purpose VLAs.

| Dataset | Public scale | Embodiments and value | Caveat |
|---|---|---|---|
| [Open X-Embodiment](https://github.com/google-deepmind/open_x_embodiment) | More than 1M real-robot trajectories and 22 embodiments | Unified RLDS access to dozens of contributed robot datasets | This is an umbrella corpus. Track each component's license and do not double-count independently downloaded subsets. |
| [DROID](https://droid-dataset.github.io/) | Canonical release: 76K trajectories, 350 hours, 564 scenes, 86 tasks | Franka; unusually strong environment and camera diversity | Recent processed mixtures report approximately 95K trajectories/500 hours. Record the exact revision used. |
| [BridgeData V2](https://rail-berkeley.github.io/bridgedata/) | Approximately 60K trajectories, 24 environments, 13 skills | WidowX; tabletop and kitchen manipulation | Available raw and as TFDS/RLDS. One of Xiaomi-Robotics-1's explicitly named public sources. |
| RT-1 / Fractal | Approximately 130K episodes | Google robot; broad language-conditioned manipulation | Distributed through the [Open X-Embodiment catalog](https://github.com/google-deepmind/open_x_embodiment). |
| BC-Z | Large Google-robot imitation corpus | Language-conditioned single-arm skills | Qwen-RobotManip retained BC-Z, Bridge, and Fractal from OXE, totaling approximately 600 hours. |
| [AgiBot World](https://github.com/OpenDriveLab/AgiBot-World) | Beta: 1,003,672 trajectories/43.8 TB; Alpha: 92,214/8.5 TB | AgiBot G1 bimanual humanoid; approximately 200 task types | CC BY-NC-SA. Select tasks or stream rather than mirroring the full release. |
| [RoboMIND](https://github.com/x-humanoid-robomind/x-humanoid-robomind.github.io) | 107K trajectories, 479 tasks, 96 object classes; hosted files total approximately 12.3 TB | Franka, UR5e, AgileX, and Tien Kung | Apache-2.0 but gated access. Task-selective download is essential. |
| [Galaxea Open-World](https://huggingface.co/datasets/OpenGalaxea/Galaxea-Open-World-Dataset) | Approximately 500 hours; hosted files total approximately 2.87 TB | Bimanual mobile household manipulation | Gated, CC BY-NC-SA. |
| [RoboCOIN](https://github.com/FlagOpen/RoboCOIN) | Qwen used approximately 430 hours across 10 embodiments | ALOHA, G1, MMK2, Realman, AgileX, Leju, and others | Convenient task-by-task LeRobot v2.1 repositories; mostly gated. |
| [RH20T](https://rh20t.github.io/) | Approximately 110K sequences, 140+ tasks, 42 skill categories; approximately 1,100 hours | Flexiv, UR5, Franka, and Kuka; RGB-D, force, audio, and proprioception | Very large in raw multimodal form. |
| [RDT-1B fine-tuning data](https://huggingface.co/datasets/robotics-diffusion-transformer/rdt-ft-data) | More than 6K ALOHA episodes, approximately 29 hours; approximately 696 GB | High-quality bimanual data | Large because of multiview video. Download task shards selectively. |
| [RoboSet](https://robopen.github.io/roboset/) | 28.5K trajectories; 12 skills across 38 kitchen tasks | Franka/Robotiq, four cameras, kinesthetic and teleoperation data | MIT; a useful medium-sized real-data source. |
| [FurnitureBench](https://github.com/clvrai/furniture-bench) | More than 200 hours | Long-horizon Franka furniture assembly | Especially useful for precision, insertion, and error recovery. |
| [Humanoid Everyday](https://github.com/physical-superintelligence-lab/Humanoid-Everyday) | 260 scenarios x 40 episodes; full LeRobot release approximately 1 TB | Unitree G1 and H1, loco-manipulation, and tool use | MIT; use task-selective downloads. |
| [LeRobot Community v3](https://huggingface.co/datasets/lerobot/community_dataset_v3) | 50,622 episodes, 251.5 hours, 46+ robot types; approximately 968 GB | 791 crowdsourced datasets | Apache-2.0, but schemas and quality vary. Treat it as a source pool rather than an immediately uniform mixture. |
| [LeRobot Community v1](https://huggingface.co/datasets/HuggingFaceVLA/community_dataset_v1) | 11.1K episodes, 46.9 hours, 119 GB | Mostly SO-100 tabletop manipulation | Smaller and more curated than v3; a sensible inexpensive addition. |

### Additional Open X-Embodiment components

Useful OXE components beyond Bridge, RT-1, and BC-Z include:

- TACO Play
- Jaco Play
- Language Table
- RoboTurk
- NYU Franka Play
- FMB
- Dobb-E
- TOTO
- RoboSet
- FurnitureBench
- Stanford Hydra
- Austin Sailor, Sirius, and Buds
- Berkeley UR5, RPT, and MVP
- Cable Routing
- RoboCook
- UTokyo robot datasets

The [RDT pretraining recipe](https://github.com/thu-ml/RoboticsDiffusionTransformer/blob/main/docs/pretrain.md)
provides download and preprocessing support for 46 sources and is a useful
reference implementation for heterogeneous dataset ingestion.

## Simulator-generated corpora

| Dataset | Scale | Why it is useful |
|---|---|---|
| [InternData-A1](https://huggingface.co/datasets/InternRobotics/InternData-A1) | More than 3,600 hours of high-fidelity simulated manipulation | Single-arm, bimanual, mobile, and long-horizon coverage; used by Qwen-RobotManip. |
| [VIMA-Data](https://huggingface.co/datasets/VIMA/VIMA-Data) | 650K oracle trajectories, 13 training tasks; approximately 21.5 GB | Exceptional value per byte; multimodal prompts and compositional generalization. |
| [MimicGen](https://github.com/NVlabs/mimicgen) | More than 48K released demonstrations across 12 tasks | CC BY 4.0 data plus a generator capable of producing additional demonstrations. |
| [CALVIN](https://github.com/mees/calvin) | 24 hours total: 6 hours in each of four environments | Long-horizon language control. D-only is 166 GB, ABC is 517 GB, and ABCD is 656 GB. |
| [ManiSkill](https://github.com/haosulab/ManiSkill) | Downloadable and generated expert demonstrations across many task families | Strong source for articulation, dexterity, and varied embodiments. |
| [RLBench](https://github.com/stepjam/RLBench) | Expert generator for 100 tasks | Useful after fixing environment/task versions and holding out evaluation seeds. |
| [BEHAVIOR-1K](https://github.com/StanfordVL/BEHAVIOR-1K) | 1,000 household activities | Better treated as an environment/task generator than a fixed clean behavior-cloning corpus. |
| VLA-Arena, RoboTwin, and RoboCasa generators | Effectively unlimited | Generate data only after freezing evaluation assets, seeds, and configurations. |

LIBERO data can still be useful for debugging loaders and training code, but it is
low priority for the benchmark program because of saturation and contamination
risk.

## Derived annotations and human-video data

These sources are valuable, but are not interchangeable with robot action
demonstrations.

| Dataset | Scale | Correct interpretation |
|---|---|---|
| [RoboInter-Data](https://huggingface.co/datasets/InternRobotics/RoboInter-Data) | 235,920 annotated episodes; approximately 408 GB | LeRobot conversion and dense annotations for DROID + RH20T. Excellent packaging, but not additional robot experience. |
| [ShareRobot](https://huggingface.co/datasets/BAAI/ShareRobot) | 51,403 OXE episodes converted into 1.03M planning QA pairs; 6,870 trajectory images; approximately 351 GB | Adds planning, affordance, and trajectory annotations to OXE. Do not count it as new action data. |
| [EgoDex](https://github.com/apple/ml-egodex) | 338K demonstrations, 829 hours, 194 tasks; approximately 1.7 TB full | Egocentric hand/head/body poses and language. Useful for representation learning or retargeting, not direct robot behavior cloning. |
| [VITRA-1M](https://huggingface.co/datasets/VITRA-VLA/VITRA-1M) | More than 1M human-hand VLA episodes; annotations approximately 92 GB | Requires source videos from Ego4D, EPIC-KITCHENS, EgoExo4D, and related sources. |
| [VITRA TeleData](https://huggingface.co/datasets/microsoft/VITRA-TeleData) | Approximately 5.6 GB | Actual robot teleoperation data for downstream adaptation; easier to use than VITRA-1M. |
| [EgoVerse](https://github.com/GaTech-RL2/EgoVerse) | 1,362 hours, 1,965 tasks, 240 scenes, 2,087 demonstrators | Egocentric hand/head trajectories in Zarr; useful for human-to-robot retargeting. |
| [Open-H-Embodiment](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Open-H-Embodiment) | 120K trajectories, 750 hours; approximately 2.87 TB | Paired video and kinematics for surgical and ultrasound robotics; open but low priority for household manipulation. |

Ego4D, EgoExo4D, EPIC-KITCHENS, Something-Something-v2, and HoloAssist
are useful video/world-model sources. Do not describe them as robot policy action
data unless a trajectory-reconstruction or inverse-dynamics pipeline has been
applied.

## Described publicly but not released as training corpora

Do not include the following quantities in totals of open training data:

- [Xiaomi-Robotics-1](https://github.com/XiaomiRobotics/Xiaomi-Robotics-1)
  reports more than 100K hours of UMI pretraining and approximately 10K hours of
  post-training data. The large UMI and in-house corpora are not released. Its
  explicitly named public components are Bridge V2, RT-1, and DROID.
- [Qwen-RobotManip](https://github.com/QwenLM/Qwen-RobotManip) reports a
  38.1K-hour mixture, including 24,808 hours of synthesized human-to-robot data,
  but does not release its weights or processed synthetic corpus. Its raw public
  ingredients are represented above.
- [LingBot-VLA 2.0](https://github.com/Robbyant/lingbot-vla-v2) reports
  50K hours of robot data plus 10K hours of egocentric data, but does not publish
  that curated corpus as a downloadable dataset.
- RoboMIND 2.0 has a nominal Hugging Face repository, but it is currently a
  placeholder rather than a complete downloadable release.
- RDT's "1M+ pretraining episodes" are mainly a mixture of existing OXE and other
  public datasets. Its approximately 6K-episode ALOHA fine-tuning release is the
  additional downloadable corpus.
- GE-Sim-V2 and Cosmos provide useful world-model code and assets, but not clean,
  fully released policy-training corpora matching all data described in their
  papers.

## Storage-aware starting mixtures

The available local storage is approximately 715 GB. Use rotating manifests and
task-selective downloads instead of mirroring every release.

### Benchmark-aligned cache

- NVIDIA GR00T X-Embodiment Sim: approximately 99 GB
- RoboDojo LeRobot v3: approximately 120 GB
- RoboTwin: approximately 80 GB
- EBench: approximately 327 GB
- MIKASA-Robo-VLA: approximately 28 GB
- VIMA-Data: approximately 22 GB
- Selected VLABench, Colosseum, and RoboMME shards

This is approximately 675 GB before extraction and cache overhead. EBench should
therefore be task-sharded or rotated.

### Cross-embodiment pretraining cache

- RoboInter DROID + RH20T: approximately 408 GB
- NVIDIA X-Embodiment Sim: approximately 99 GB
- RoboTwin: approximately 80 GB
- LeRobot Community v1: approximately 119 GB

This is approximately 706 GB and requires streaming or aggressive cache eviction.

## Dataset manifest requirements

Record the following for every downloaded or generated shard:

- Dataset name and exact repository/revision
- Source URL and license
- Content hash and byte size
- Task, scene, asset, seed, and episode identifiers
- Embodiment and end-effector type
- State and action schema
- Action coordinate frame and control mode
- Control frequency and camera configuration
- Dataset-specific normalization statistics
- Original train/validation/test designation
- `sim-benchmarks` training/evaluation exclusion status
- Whether the data duplicates or derives from another corpus

The duplication field is essential for overlapping collections such as OXE,
RoboInter, ShareRobot, RDT, and LeRobot conversions.
