# Score-based change-point detection and region localization for spatio-temporal point processes 

[Link to paper](https://arxiv.org/abs/2602.04798)

**Abstract:**
We study sequential change-point detection for spatio-temporal point processes, where actionable detection requires not only identifying when a distributional change occurs but also localizing where it manifests in space. While classical quickest change detection methods provide strong guarantees on detection delay and false-alarm rates, existing approaches for point-process data predominantly focus on temporal changes and do not explicitly infer affected spatial regions. We propose a likelihood-free, score-based detection framework that jointly estimates the change time and the change region in continuous space-time without assuming parametric knowledge of the pre- or post-change dynamics. The method leverages a localized and conditionally weighted Hyvärinen score to quantify event-level deviations from nominal behavior and aggregates these scores using a spatio-temporal CUSUM-type statistic over a prescribed class of spatial regions. Operating sequentially, the procedure outputs both a stopping time and an estimated change region, enabling real-time detection with spatial interpretability. We establish theoretical guarantees on false-alarm control, detection delay, and spatial localization accuracy, and demonstrate the effectiveness of the proposed approach through simulations and real-world spatio-temporal event data.

**APA:**
Zhou, W., Xie, L., & Zhu, S. (2026). Score-Based Change-Point Detection and Region Localization for Spatio-Temporal Point Processes. arXiv preprint arXiv:2602.04798.

**BibTex:**
@article{zhou2026score,
  title={Score-Based Change-Point Detection and Region Localization for Spatio-Temporal Point Processes},
  author={Zhou, Wenbin and Xie, Liyan and Zhu, Shixiang},
  journal={arXiv preprint arXiv:2602.04798},
  year={2026}
}
