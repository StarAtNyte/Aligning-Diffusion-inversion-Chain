This is a powerful framework for generating variations of images with recurring patterns and structures while preserving their essential characteristics. It's especially effective for pattern-rich images like carpet designs, wallpapers, and tileable textures.

## 🌟 Features

- **Structural Consistency**: Generates variations that maintain the essential character of the original pattern
- **Controlled Randomness**: Intelligently modifies patterns with meaningful variations rather than arbitrary changes
- **Latent Space Manipulation**: Uses advanced adaptive instance normalization techniques for coherent transformations
- **Carpet Specialization**: Includes optimizations specifically for textile and carpet pattern generation
- **Simple API**: Easy-to-use interface for both standard images and pattern-rich designs

## 💡 How It Works

This uses a modified Stable Diffusion pipeline with specialized attention mechanisms and adaptive instance normalization to create meaningful variations of patterns. The key components include:

1. **Custom Attention Mechanism**: Modified self-attention that respects structural patterns
2. **Enhanced AdaIN**: Advanced adaptive instance normalization that preserves motif structures
3. **Latent Space Manipulation**: Grid-level transformations in latent space for coherent pattern modifications
4. **DDIM Inversion**: Precise latent representation capturing the essence of the original design

## 🔧 Advanced Configuration

The behavior of the framework can be extensively customized:

```python
# Detailed configuration
cfgs = {
    "self_attn": {
        "atten_frames": 3,  # Controls attention spread
        "t_align": 700      # Alignment cutoff for detail preservation
    },
    "inference": {
        "invert_step": 100,  # Higher for better inversion quality
        "ddim_step": 100,    # Controls refinement level
        "cfg": 6.5,          # Guidance scale
        "is_null_prompt": True,
        "t_early": 750       # When to start adaptive normalization
    }
}

model.set_config(cfgs)
```

## 🧩 Core Components

### Modified Attention Mechanism

```python
def new_forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, **cross_attention_kwargs):
    # Custom attention implementation preserving pattern structures
    # while allowing controlled variation
    # ...
```

### Enhanced Adaptive Instance Normalization

```python
def adain_latent(feat, cond_feat, eps=1e-5, detail_preservation=0.65):
    """
    Enhanced Adaptive Instance Normalization to encourage more structural variations
    while maintaining consistency with the original pattern.
    """
    # Calculate spatial-aware statistics
    # This helps capture and transfer motif structures better
    # ...
```

## 🎨 Examples Gallery


### Original Image and Generated Variations

![01_city_comparison](https://github.com/user-attachments/assets/f1214ae6-4059-4e6c-9d80-a9fcceea744e)
![04_theater_comparison](https://github.com/user-attachments/assets/c7eea09c-237b-43b0-b740-689c1110c047)
![15_starry_night_comparison](https://github.com/user-attachments/assets/acc09bc9-ded7-45b1-8d3a-c14e04a33656)
![20_db_sneaker_comparison](https://github.com/user-attachments/assets/8217d89c-ff32-43d6-a2cd-995c31b95351)
![05_pink_comparison](https://github.com/user-attachments/assets/247ef772-a87a-4eef-a03e-42142a59bcd2)


### Original Carpet and Generated Variations
![Team Zeror - Phase 2](https://github.com/user-attachments/assets/88b98e17-1abf-4f0c-b5f6-d6cc89149324)
![Team Zeror - Phase 2 (1)](https://github.com/user-attachments/assets/04784d69-e8ec-4ab4-affb-408dc1b7eed5)




## 📝 Reference
Real-World Image Variation by Aligning Diffusion Inversion Chain
https://arxiv.org/pdf/2305.18729

## ⚖️ License

This project is licensed under the MIT License - see the LICENSE file for details.

