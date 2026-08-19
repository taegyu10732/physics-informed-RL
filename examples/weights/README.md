# Example pretrained checkpoint

`pi_attention_unet_indoor_gas_v1.pt` is the public example PI-Attention-UNet
checkpoint used by the downstream gas-source navigation environment.

| Property | Value |
|---|---|
| Training stage | Original PI-U-Net research stage |
| Intended use | Reconstruct a gas map and predict source position for the RL environment |
| Training sigma range | 100–300 grid cells |
| Size | 125,709,947 bytes (119.89 MiB) |
| SHA-256 | `AF14501696AD20ED3BA44BD88653D71485374389E8A29E64158F422B0F38659A` |

The file is intentionally labeled as an **example research checkpoint**. It
demonstrates the complete inference and RL integration path; it is not a
general-purpose gas-safety model or a standardized benchmark result.

The checkpoint is tracked with Git LFS. After cloning, run:

```bash
git lfs install
git lfs pull
```
