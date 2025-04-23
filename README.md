# Moonshot Project: Converting SMILES Encodings to SELFIE Strings

## Author
Naigam Shah  
Department of Computer Science and Engineering  
University of California San Diego  
**Email:** n7shah@ucsd.edu

---

## Abstract
The process of identifying molecular structures from natural products is pivotal in fields like drug discovery. Tools such as SPECTRE have advanced this field by using NMR spectroscopy to derive SMILES encodings of molecules. However, SMILES strings often suffer from errors and lack robustness, which can lead to inaccuracies in downstream tasks like molecule identification.

This project builds upon SPECTRE by transforming SMILES strings into SELFIE representations. SELFIE strings are chemically robust, error-tolerant molecular representations. By bridging this gap, the project aims to enhance the computational reliability of molecule identification workflows.

You can find the workspace [here](https://drive.google.com/drive/folders/1m4PtNtnG5Yi9MGoVlNyrqoI3B-Y6n_GG?usp=sharing).

---

## Motivation
The ultimate goal of the overarching Moonshot Project is to automate the transformation of 2D NMR data into SELFIE representations. This effort will streamline the molecular annotation process, helping chemists rapidly identify novel molecular structures.

Within this framework, my sub-project serves as a foundational proof-of-concept. Specifically, I worked on designing and implementing a sequence-to-sequence (seq-to-seq) transformer architecture capable of converting SMILES encodings into SELFIE representations. Achieving this conversion is a critical step in validating the feasibility of the broader Moonshot Project's vision.

The scope of this project is to establish the technical viability of transforming SMILES strings into SELFIE representations using state-of-the-art deep learning methodologies. Success in this task demonstrates that similar methodologies can be applied to convert raw NMR data into SELFIE representations in future iterations of the Moonshot Project.

---

## Setup
The project utilized the `weird_H_and_tautomer_cleaned.zip` dataset provided by Wangdong Xu. This dataset contains SMILES strings of chemical molecules, which were preprocessed into SELFIE strings for training and evaluation.

### References and Resources
- [Rose Yu’s LIMO-Plus repository](https://github.com/Rose-STL-Lab/LIMO-Plus)
- [VAEFormer codebase](https://github.com/Rose-STL-Lab/LIMO)

### Tools and Environments
- **Programming Language:** Python
- **Libraries and Frameworks:**
  - PyTorch Lightning for model training and experimentation
  - DeepChem and RDKit for molecular data preprocessing
  - SELFIES for handling SELFIE string representations
  - Torcheval for evaluation metrics such as perplexity
- **Training Platform:** Google Colab

### Challenges and Resolutions
1. **Model Architecture Decisions:** Designing a unified seq-to-seq architecture required balancing complexity with computational feasibility. Iterative experimentation with transformer layers and embeddings helped achieve an optimal design.
2. **Projection Layer Complexity:** The projection layer connecting the SMILES encoder and SELFIE decoder introduced a non-pretrained component. Additional fine-tuning and dataset augmentation were planned for future iterations.
3. **Resource Constraints:** Training on Google Colab posed limitations in terms of memory and runtime. Training was optimized with smaller batch sizes and efficient data loaders.

---

## Methodology
The project was divided into three distinct but interrelated pipelines:

### 1. [Training the SELFIE Decoder](https://colab.research.google.com/drive/1A-pZP0dc2F30I-z0bDLN8qChp_Inc-Sz?usp=sharing)
- **Objective:** To train the model to learn the syntax and structure of SELFIE representations.
- **Approach:**
  - SMILES strings were preprocessed into SELFIE representations using the SELFIES library.
  - A decoder-only transformer architecture was trained using cross-entropy loss to generate accurate SELFIE strings.
- **Outcome:** A functional SELFIE decoder capable of generating valid SELFIE strings.

### 2. [Training the SMILES Autoencoder](https://colab.research.google.com/drive/1kfAREWI6J-iJcSv4QsB1OHULMnJ8lyu6?usp=sharing)
- **Objective:** To create a robust SMILES encoder capable of learning latent representations of chemical structures.
- **Approach:**
  - A transformer-based architecture was designed.
  - Cross-entropy loss was used to train the model to reproduce input SMILES strings at the output.
- **Outcome:** The SMILES encoder was successfully trained, serving as a foundation for subsequent steps.

### 3. [Integrating and Fine-Tuning the SMILES-to-SELFIE Model](https://colab.research.google.com/drive/11Y45QoH9Dpt9ruDBv6FPzu7z19oKN1C4?usp=sharing)
- **Objective:** To enable direct transformation of SMILES strings into their corresponding SELFIE representations.
- **Approach:**
  - The encoder and decoder were fine-tuned together to learn the end-to-end mapping.
  - A projection layer was introduced between the encoder and decoder to align latent representations.
  - Additional fine-tuning improved task performance.
- **Outcome:** A seq-to-seq model capable of producing SELFIE strings directly from SMILES encodings.

### Model Paths

The best trained models for the pipeline are available at the following paths:

**SELFIES Decoder:** `./selfiesDecoder1.pt`

**SMILES AutoEncoder:** `./smiles_encoder_decoder2.pt`

**SMILES to SELFIES Encoder-Decoder:** `./smiles_to_selfies_full.pt`

**Note:**

The **Finite State Automata (FSA)** takes the raw SELFIES output and processes it to ensure a **valid** SMILES string. Always outputs a valid processed SMILES string, even if the raw output is invalid.

---

## Hyperparameters
- **Embedding Dimension (EMBED_DIM):** 64
- **Latent Dimension (LATENT_DIM):** 1024
- **Number of Transformer Layers (NUM_LAYERS):** 4
- **Attention Heads (NUM_HEADS):** 8
- **Feedforward Dimension (FEEDFORWARD_DIM):** 256
- **Batch Size (BATCH_SIZE):** 64
- **Learning Rate (LEARNING_RATE):** 0.001 (adjusted with cosine annealing scheduler)
- **Training Epochs (NUM_EPOCHS):** 7
- **Loss Function:** Cross-entropy loss

---

## Performance Metrics
- **Perplexity:**
  - Perplexity scores were computed to measure the fluency and coherence of generated SELFIE strings. Initial results showed a steady decline in perplexity, indicating improved learning over epochs.
- **Validation Loss:**
  - Consistently decreased across all pipelines, reflecting the model’s ability to generalize.
- **Qualitative Analysis:**
  - Generated SELFIE strings were visually compared with ground truth strings, showing high accuracy in molecular representation.

---

## Experiment Results

### Metrics Overview

The SMILES-to-SELFIE encoder-decoder architecture was trained for 17 epochs, and the training converged successfully. The following metrics were used to evaluate the performance:

#### Validation Loss (val_loss)
- **Description:** This metric represents the cross-entropy loss during validation. It evaluates how closely the predicted SELFIE strings align with the ground truth by comparing individual tokens in the sequences. Lower values indicate better alignment.  
- **Achieved Value:** `0.9789`  

#### Validation Negative Log-Likelihood (val_nll)
- **Description:** Measures the negative log probability of the correct SELFIE sequence given the predicted distribution. It serves as an indicator of how confident the model is about its predictions.  
- **Achieved Value:** `0.9459`  

#### Validation KL Divergence (val_kld)
- **Description:** Quantifies the divergence between the learned latent space distribution and the target distribution. This metric ensures that the latent representations follow the desired structure, which is critical for generative tasks.  
- **Achieved Value:** `0.3304`  

#### Validation Log Variance (val_logvar)
- **Description:** Reflects the log variance of the latent space. This metric helps understand the variability captured by the model. Stable and negative values indicate that the model is confident in its learned representations.  
- **Achieved Value:** `-5.0032`  

#### Validation Mean of Latent Representations (val_mu)
- **Description:** Represents the mean of the latent space representations during validation. A small and stable value shows that the latent space has been optimized effectively for encoding molecular features.  
- **Achieved Value:** `0.1085`  

#### Logs Location
The detailed logs of the experiments are stored in the `./logs` folder, with the final results for the SMILES-to-SELFIE encoder-decoder architecture available in:  
`logs/stos/lightning_logs/version_17`

---

### Results Table

| **Metric**                 | **Description**                                         | **Value**   |
|-----------------------------|---------------------------------------------------------|-------------|
| **Validation Loss**         | Cross-entropy loss for SELFIE prediction during validation | `0.9789`    |
| **Validation NLL**          | Negative log-likelihood of predicted SELFIE sequences    | `0.9459`    |
| **Validation KL Divergence**| Divergence between learned and target latent distributions | `0.3304`    |
| **Validation Log Variance** | Log variance of latent space representations             | `-5.0032`   |
| **Validation Latent Mean**  | Mean of latent space during validation                   | `0.1085`    |

---

### Sample Output for Prediction

**Input SMILES:** `C=C1CC(=CC)C(=O)OC2CCN3CC=C(COC(=O)C1(C)O)C23`

**Target SELFIES:** `[C][=C][C][C][=Branch1][Ring1][=C][C][C][=Branch1][C][=O][O][C][C][C][N][C][C][=C][Branch1][#C][C][O][C][=Branch1][C][=O][C][Ring2][Ring1][Ring2][Branch1][C][C][O][C][Ring1][#C][Ring1][N]`

**Predicted SELFIES:** `[C][=C][C][=C][C][Branch1][C][O][C][Branch1][C][C][C][C][C][C][=Branch1][C][=O][C][C][Ring1][#Branch1][C][Ring1][N][C][=Branch1][C][=O][O][C][Ring1][N][C][Ring2][Ring1][Ring1][Ring2][Ring1]`

---

### Observations

#### Validation Loss and NLL
The cross-entropy loss and NLL indicate that the model performs well in predicting SELFIE strings from SMILES encodings. However, the values suggest that there is room for improvement, particularly in aligning predictions with ground truth sequences more closely.

#### KL Divergence
A low KL divergence value signifies that the learned latent space closely matches the target distribution. While the results are promising, further optimization could enhance alignment further.

#### Latent Representations
Both the log variance and latent mean show stability, which suggests that the model effectively captures and represents the underlying structure of molecular data in its latent space.

## Future Directions

To further improve the metrics and overall model performance, the following recommendations are proposed:

### Dataset and Evaluation

#### Expand Dataset with COCONUT Natural Products Database
- Use the [**COCONUT (COlleCtion of Open Natural producTs)**](https://coconut.naturalproducts.net/download) dataset, which contains millions of molecular structures derived from natural products.  
- Training the model on this large and diverse dataset can significantly enhance its robustness and generalizability.  
- This dataset aligns well with the project’s goal of accurately predicting molecular representations.  

#### Evaluate with Chemical Validity Metrics
- Add **Earth Mover's Distance** as a metric to calculate the difference between predicted and target SELFIES. This will provide a meaningful quantitative measure of prediction accuracy.
- Use chemical validity metrics (e.g., RDKit-based validation) to assess the correctness of generated SELFIE strings.  
- Implement metrics such as BLEU or perplexity to evaluate sequence quality.

---

### Model Improvement

#### Fine-Tune Pre-Trained Models
- Independently fine-tune the **SMILES autoencoder** and **SELFIE decoder** to ensure optimal performance for each component before integrating them.  
- This step ensures that both components are individually robust, contributing to better end-to-end performance.  

#### Enhance Model Complexity
- Increase the number of layers in the encoder and decoder transformers to better capture complex molecular relationships.  
- Experiment with larger embedding sizes and feedforward dimensions to improve the model’s capacity.  

#### Incorporate Residual Connections
- Adding residual connections in the encoder and decoder could help stabilize training and improve gradient flow.  

---

### Training Strategies

#### Increase Epochs and Batch Sizes
- Extend training beyond 17 epochs while employing early stopping to avoid overfitting.  
- Utilize larger batch sizes with optimized GPU resources to improve generalization.  

#### Optimize Learning Rates
- Experiment with different learning rate schedules, such as cyclic or warmup-based learning rates, to facilitate smoother convergence.  

#### Apply Data Augmentation
- Introduce noise or perturbations to SMILES strings for augmentation, increasing the diversity of the dataset and improving robustness.  

---

### Fine-Tuning and Regularization

#### Improve Latent Space Regularization
- Regularize the KL divergence loss to ensure better alignment of the latent space with the target distribution.  
- Apply dropout and layer normalization to mitigate overfitting.  

#### Projection Layer Optimization
- Further refine the projection layer connecting the encoder and decoder to improve information transfer between the two components.  
