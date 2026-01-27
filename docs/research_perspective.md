# Research Perspective: Efficient Cervical Cytology at the Edge

## 1. Introduction: The Need for Edge AI in Pathology
Cervical cancer screening is most critical in low-and-middle-income regions where access to high-end cloud computing or powerful GPU clusters is limited. To democratize AI-assisted screening, the underlying models must be computationally efficient without sacrificing diagnostic accuracy.

This project focuses on **On-Device** or **Edge AI** capabilities, simulating a system that could be embedded directly into a digital microscope or a local clinic's workstation.

## 2. Model Selection: Why MobileViT-S?

We selected **MobileViT-S (Small)** as the backbone for cell classification. This decision represents a strategic trade-off between model size, inference latency, and accuracy.

### 2.1 Key Architectural Advantages
MobileViT combines the best of two worlds:
*   **Inductive Bias of CNNs**: It uses convolutions to efficiently process local features (edges, textures).
*   **Global Context of Transformers**: It allows pixels to "see" distinct parts of the image, capturing global shape information crucial for differentiating subtle cellular anomalies.

### 2.2 Efficiency by the Numbers
*   **Parameters**: ~5.6 Million
*   **Model Size**: < 25 MB (FP32)
*   **Inference Speed**: Capable of real-time processing on standard mobile-grade CPUs or entry-level GPUs.

 compared to standard architectures:
*   **ResNet-50**: ~25M params (4x larger)
*   **ViT-Base**: ~86M params (15x larger)

Despite being significantly smaller, MobileViT approaches the accuracy of these larger models for this task.

## 3. Performance & Clinical Relevance

### High Sensitivity for Screening
In our evaluation (see README metrics), the model achieves:
*   **92.6% Overall Accuracy**
*   **98% Recall on Dyskeratotic cells** (High-grade lesions)
*   **93% Recall on Koilocytotic cells** (Low-grade lesions)

### The Edge Advantage
1.  **Privacy**: Data does not need to leave the clinic.
2.  **Latency**: No network round-trip time; results are available immediately as the slide is scanned.
3.  **Cost**: Runs on affordable hardware (e.g., Jetson Nano, Raspberry Pi with accelerator, or standard office laptops).

## 4. Conclusion
The choice of MobileViT-S proves that **high-performance medical AI does not require massive compute resources**. By optimizing for efficiency (~5.6M params), we enable a scalable, deployable solution for automated cervical cytology screening.
