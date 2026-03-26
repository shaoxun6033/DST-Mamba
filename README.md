# DST-Mamba

This is an official implementation of [DST-Mamba: A Dual-Stream Temporal-Channel Mamba with Lifting Wavelets for Cloud Workload Prediction].

## Usage

- Train and evaluate DST-Mamba
  - You can use the following command:`sh ./scripts/ali18.sh`.

- Train your model
  - Add model file in the folder `./models/your_model.py`.
  - Add model in the ***class*** Exp_Main.

## Model

Our proposed DST-Mamba Network consists of three key modules: a Residual Lifting Wavelet Decomposition (RLWD) module that disentangles non-stationary workloads into multiscale trend and burst components, a Dual-Stream Mamba module that captures long-range temporal dynamics and time-varying cross-variable dependencies through Temporal Mamba and graph-guided Channel Mamba, and a FiLM-based gated fusion module that adaptively aligns and integrates temporal-channel features for robust and efficient forecasting.

<div align=center>
<img src="https://github.com/shaoxun6033/DST-Mamba/blob/main/pic/architecture.pdf" width='45%'> 
</div>


## Citation

If you find this repo useful, please cite our paper as follows:
```

```

## Contact
If you have any questions, please contact us or submit an issue.

## Acknowledgement

We appreciate the valuable contributions of the following GitHub.

- LTSF-Linear (https://github.com/cure-lab/LTSF-Linear)
- TFEGRU (https://github.com/ACAT-SCUT/TFEGRU)
- TimesNet (https://github.com/thuml/TimesNet)
- Time-Series-Library (https://github.com/thuml/Time-Series-Library)
- MSGNet (https://github.com/YoZhibo/MSGNet)
- Autoformer (https://github.com/thuml/Autoformer)
