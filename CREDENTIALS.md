# Project Credentials (Debug Phase)

**Note**: Actual credentials are stored securely and set via environment variables on TPU VMs.
Do not commit real tokens to git. GitHub push protection will block them.

## Hugging Face Token
```
HF_TOKEN=<your_huggingface_token_here>
```

## Weights & Biases Token
```
WANDB_API_KEY=<your_wandb_api_key_here>
```

## TPU Information
- **Zone**: europe-west4-a
- **Name**: node-1
- **Type**: v6e-8 (2x4 topology)
- **Accelerator**: V6E
- **Status**: READY

## Optimization Strategies to Test

### Memory Optimization
1. **Mesh Configuration**
   - Current: (4,2) - 4-way FSDP + 2-way TP
   - Tested: (2,4), (8,1), (4,2)

2. **Gradient Checkpointing**
   - Recompute activations during backward pass
   - Trade computation for memory

3. **Gradient Accumulation**
   - Accumulate gradients over multiple micro-batches
   - Reduce memory per step

4. **Batch Size Tuning**
   - Target: 32
   - Current test: 16
   - Can reduce further if needed

### Performance Optimization
1. **Flash Attention**
   - Memory-efficient attention implementation
   - Reduces memory usage and improves speed

2. **Fused Kernels**
   - Fused Adam optimizer
   - Fused layer norm
   - Other JAX fusion optimizations

## Notes
- This is debug phase - tokens will be replaced later
- Project: qwen3-grpo-gsm8k
- Wandb URL: https://wandb.ai/johntitordemon2036/qwen3-grpo-gsm8k
