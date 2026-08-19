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
## 7. Integration with Blockchain & Cyber Resilience (Foundational Framework)

While the core of this proposal focuses on the computer vision architecture, ensuring the **trustworthiness** of the entire system is paramount for real-world agricultural deployment. Farm data is sensitive, and AI models are vulnerable to adversarial attacks. To address this, I will explicitly integrate the theoretical framework for **quantitative cyber resilience assessment** established by Ishchukova et al. (2026).

Specifically, I aim to extend Ishchukova's comprehensive approach to the agricultural AI domain by:

1. **Blockchain-based Data Provenance**: Implementing a lightweight blockchain layer to ensure the integrity and immutability of collected video/skeleton data, preventing tampering from farm to server.
2. **Resilience Metrics for AI Pipelines**: Adapting her quantitative resilience metrics to evaluate the robustness of my Uncertainty-Aware Vision Transformer against data poisoning, occlusion, and adversarial inputs.
3. **Trusted Operation**: Applying the principles of "Trusted Operation of Cyber-Physical Systems" (Basan et al., 2023, co-authored by Ishchukova) to the monitoring hardware (cameras, edge devices) used on farms.

This extension bridges my MSc background in Trusted AI/Blockchain with the applied vision work at PaquetLab, ensuring the final deployed system is not only accurate but also verifiably secure and resilient.

*Reference:* Ishchukova, E.; Petrenko, S.; Petrenko, A.; et al. (2026). A Comprehensive Approach to Assessing the Cyber Resilience of Blockchain Platforms. *Sci*, 8(5), 97.
I complete my MSc in Early 2027. I am available for an **early start in Fall 2026** to begin infrastructure setup.
