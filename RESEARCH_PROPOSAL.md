# Research Proposal: Uncertainty-Aware Vision Transformers for Livestock Behavioural Analysis

**Author:** Joseph Eniola Kehinde  
**Target Lab:** PaquetLab, Université Laval  
**Proposed Start:** January 2027 (Flexible to Fall 2026)

## 1. Research Problem & Gap
Current computer vision models for livestock monitoring achieve high accuracy in labs but fail in real farms due to:
- **Occlusion** (mud, bedding, other animals).
- **Lighting variability** (day/night cycles).
- **Data scarcity** (annotating 3D animal skeletons is expensive).

Critically, existing models are "black boxes." When a model falsely flags a healthy animal as lame, farmers lose trust. There is no unified framework that provides both (a) few-shot adaptability and (b) calibrated uncertainty estimates for animal posture recognition.

## 2. Novel Hypothesis
> *"A Vision Transformer (ViT) with an Evidential Uncertainty Head, fine-tuned on synthetic animal skeletons, can outperform standard CNNs/LSTMs in farm environments while delivering interpretable confidence intervals using only 5-10% of labeled data."*

## 3. Methodological Innovations (Building on my GitHub Code)
- **Innovation 1 (Skeleton Attention)**: Adapt my `ScaledDotProductAttention` to process 3D joint coordinates (x,y,z) across 30 frames for action recognition.
- **Innovation 2 (Evidential Deep Learning)**: Replace the standard softmax with a Dirichlet distribution to compute prediction uncertainty (alerting farmers when the model is unsure due to mud/occlusion).
- **Innovation 3 (Domain Adaptation)**: Transfer a pre-trained ViT (trained on human poses) to quadrupeds using a custom "limb-matching" layer, minimizing the need for labeled sheep/pig data.

## 4. Expected Outcomes
- A PyTorch library for uncertainty-aware 3D action recognition.
- A deployment strategy where low-confidence predictions trigger camera recalibration (saving farmers from false alarms).
- A publication targeting *Computers and Electronics in Agriculture* or *CVPR Workshops*.

## 5. Why I am uniquely suited
- **Technical**: My GitHub repo proves I can implement Attention/LSTM from scratch.
- **Trusted AI**: My MSc research provides the theoretical framework for "Verifiable AI" in agriculture.
- **Full-Stack**: My ICT background ensures I can deploy the data-capture hardware (cameras/sensors) this framework relies on.

## 6. Timeline Alignment
I complete my MSc in Early 2027. I am available for an **early start in Fall 2026** to begin infrastructure setup.
