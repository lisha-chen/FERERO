# Multi-lingual Speech Recognition

## Overview

We apply the proposed method to the multi-objective finetuning of pre-trained multilingual speech models. Experiments are conducted on the **Librispeech (100 hours)** [[3]](#3) and **AISHELL v1** [[4]](#4) datasets for multilingual speech recognition.

- **Model Architecture**: A conformer model with 8 blocks is used.  
- **Model Parameters**:  
  - Total: ~64.5M  
  - Encoder: ~58.4M  
  - Classification Layer: ~6.1M  

We consider the following objectives for multi-objective learning:  
1. **Connectionist Temporal Classification (CTC)** losses for Chinese and English $\(f_t^{\text{ch}}\) and \(f_t^{\text{en}}\)$.  
2. **Contrastive Predictive Coding (CPC)** loss $\(f_p\)$ for representation learning.  

Our optimization objective is defined as:

$$
\min_{\theta} F(\theta) := [f_p(\theta), f_t^{\text{ch}}(\theta), f_t^{\text{en}}(\theta)]^\top 
$$

$$s.t.~~f_p(\theta) \leq \epsilon_1, ~f_t^{\text{ch}}(\theta) - f_t^{\text{en}}(\theta) = \epsilon_2$$ 

For more details, see [Appendix H.1](#appendix).

---

## Results: Word Error Rate (WER) and Character Error Rate (CER)

The following results are based on experiments with **Librispeech** and **AISHELL v1** datasets.

| Model                  | Librispeech (WER%) | AISHELL v1 (CER%) | Average(%) |
|------------------------|--------------------|--------------------|-----------------|
| **Komatsu et al. [1]** | 7.11              | -                  | -               |
| **w/o CPC [2]**       | 11.8              | 10.2               | 11.0            |
| **Init. (M2ASR) [2]** | 7.3               | 6.2                | 6.7             |
| **LS-FT**              | 6.8               | 5.9                | 6.4             |
| **FERERO-FT**          | 5.4               | 4.9                | 5.1             |

### Key Observations:
- Adding CPC loss improves average WER by **4.2%**.
- Fine-tuning with linear scalarization (LS-FT) improves average WER by an additional **0.3%**, with better performance in Chinese than English.  
- **Proposed Approach**: Further reduces performance gaps across languages and improves average WER by **1.3%** over LS-FT.

---

## Conclusion

The results demonstrate that incorporating the CPC loss alongside supervised CTC loss significantly improves multi-lingual speech recognition performance. The proposed method further minimizes the performance gap between languages while achieving state-of-the-art results.

---
## Prerequisites

Before running the experiments, ensure that the following dependencies are installed:

- **Conformer Model**: Install from the [sooftware/conformer GitHub repository](https://github.com/sooftware/conformer).  
- **CTC Beam Search Decoder**: Install from the [parlance/ctcdecode GitHub repository](https://github.com/parlance/ctcdecode).
 
--------
## References

[1] Tatsuya Komatsu, Yusuke Fujita, Jaesong Lee, Lukas Lee, Shinji Watanabe, and Yusuke Kida. Better intermediates improve CTC inference. arXiv preprint arXiv:2204.00176, 2022.

[2] A F M Saif, Lisha Chen, Xiaodong Cui, Songtao Lu, Brian Kingsbury, and Tianyi Chen. M2ASR: Multilingual multi-task automatic speech recognition via multi-objective optimization. In Interspeech 2024, pages 1240–1244, 2024.

[3] Vassil Panayotov, Guoguo Chen, Daniel Povey, and Sanjeev Khudanpur. Librispeech: an ASR corpus based on public domain audio books. In Proc. International Conference on Acoustics, Speech and Signal Processing, pages 5206–5210, 2015.

[4] Hui Bu, Jiayu Du, Xingyu Na, Bengu Wu, and Hao Zheng. Aishell-1: An open-source mandarin speech corpus and a speech recognition baseline. In Conference of the oriental chapter of the international coordinating committee on speech databases and speech I/O systems and assessment, pages 1–5, 2017.


