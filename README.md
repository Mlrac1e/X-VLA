# X-VLA
Soft-Prompted Transformer as Scalable Cross-Embodiment Vision-Language-Action Model





## Introduction


Successful generalist Vision-Language-Action (VLA) models that rely on effective training across diverse robotic platforms with large-scale, cross-embodiment, heterogeneous datasets. To facilitate and leverage the heterogeneity in rich, diverse robotic data sources, we propose a novel Soft Prompt approach with minimally added parameters, by infusing prompt learning concepts into cross-embodiment robot learning and introducing separate sets of learnable embeddings for each distinct data source. These embeddings serve as embodiment-specific prompts, which in unity empower VLA models with effective exploitation of varying cross-embodiment features. Our new X-VLA, a neat flow-matching-based VLA architecture, relies exclusively on soft-prompted standard Transformer encoders with an enhanced encoding pipeline, enjoying both scalability and simplicity. Evaluated across 6 simulation environments as well as 3 real-world robotics platforms, our 0.9B instantiation-X-VLA-0.9B simultaneously achieves state-of-the-art performance over a sweep of benchmark suites, demonstrating superior results on a wide axes of capabilities, from flexible dexterity to quick adaptation across embodiments, environments, and tasks.