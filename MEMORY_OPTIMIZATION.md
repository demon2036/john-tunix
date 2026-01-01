# Memory Optimization Strategies for Qwen3-1.7B GRPO Training

## Problem Statement
Training Qwen3-1.7B with GRPO on TPU v6e-8 with the following target configuration:
- `batch_size`: 32
- `num_generations`: 8
- `total_generation_steps`: 2048
- TPU HBM limit: 31.25GB per chip (8 chips total)

## Root Causes of OOM

### 1. Large Vocabulary Size
- Qwen3 vocab_size: **151,936** tokens (extremely large)
- Logits tensor: `bf16[batch_size, seq_len, vocab_size]`
- At batch_size=16, seq_len=2048: **9.27GB per allocation**
- Without TP sharding, vocab stays unsharded → massive memory usage

### 2. GRPO Memory Requirements
- Requires 3 model copies: reference, actor, rollout
- Multiple generations per batch (num_generations=8)
- Gradient computation and activation storage

## Strategies Tested

### 1. Mesh Configuration Tuning ✅ IN PROGRESS

**Approach**: Balance FSDP (parameter sharding) and TP (vocab/tensor sharding)

**Tested Configurations**:
| Mesh Shape | FSDP | TP | Result | Memory Usage |
|------------|------|----| -------|--------------|
| (2,4) | 2-way | 4-way | OOM | 60.41GB (exceeded by 29.16GB) |
| (8,1) | 8-way | 1-way | WORSE OOM | 42.29GB → 157.87GB (vocab unsharded) |
| (4,2) | 4-way | 2-way | **TESTING NOW** | Expected: ~25GB |

**Current Best**: `(4,2)` - Balances parameter sharding with vocab sharding
- 4-way FSDP: Shards model parameters across 4 devices
- 2-way TP: Shards vocab to 151936/2 = 75,968 tokens
- Expected logits reduction: 9.27GB → ~4.6GB

### 2. Gradient Checkpointing ⏳ PENDING

**Approach**: Recompute activations during backward pass instead of storing them

**Benefits**:
- Saves activation memory (can reduce by 30-50%)
- Trades computation for memory
- Especially effective for transformer layers

**Implementation**:
- In Tunix/Flax, controlled via model config
- Need to verify config parameter name in Tunix

**Expected Impact**: 20-40% memory reduction

### 3. Gradient Accumulation ⏳ PENDING

**Approach**: Split batch into micro-batches, accumulate gradients

**Benefits**:
- Enables larger effective batch sizes
- Reduces peak memory per step
- No accuracy impact if total batch size maintained

**Configuration**:
```yaml
rl_training_config:
  gradient_accumulation_steps: 2  # or 4
  # effective_batch_size = batch_size * gradient_accumulation_steps
```

**Trade-off**: Increases training time proportionally

### 4. Batch Size Reduction ⏳ FALLBACK

**Current**: 16 (target: 32)
**Options**: 8, 4, 2

**Impact**:
- Direct memory reduction
- May affect convergence/stability
- Can combine with gradient accumulation to maintain effective batch size

### 5. Flash Attention ⏳ TO EXPLORE

**Approach**: Memory-efficient attention implementation

**Benefits**:
- Reduces attention memory from O(n²) to O(n)
- Faster computation via kernel fusion
- Standard in modern transformers

**Investigation Needed**:
- Check if Tunix/Qwen3 already uses FlashAttention
- Verify JAX FlashAttention support on TPU
- Potential library: `jax-triton` or native JAX implementations

### 6. Fused Kernels ⏳ TO EXPLORE

**Approaches**:
- Fused Adam optimizer
- Fused LayerNorm
- Fused activation functions
- Fused attention operations

**Benefits**:
- Reduced intermediate tensor allocations
- Faster computation
- Better memory locality

**Investigation Needed**:
- Check Tunix's existing kernel fusion
- JAX XLA automatically fuses some operations
- May need custom JAX primitives for more aggressive fusion

### 7. Mixed Precision & BF16 Optimization ✅ ALREADY USED

**Status**: Tunix uses bf16 by default
- Model weights: bf16
- Activations: bf16
- Gradients: fp32 (for numerical stability)

### 8. Model Parallelism Optimization ⏳ TO EXPLORE

**Additional Options**:
- **Pipeline Parallelism**: Split model across stages
  - Good for very large models
  - Adds complexity, probably overkill for 1.7B

- **Sequence Parallelism**: Split sequence dimension
  - Useful for very long sequences
  - May help with total_generation_steps=2048

- **Expert Parallelism**: N/A (Qwen3 is not MoE)

### 9. Data Loading & Caching ✅ ALREADY OPTIMIZED

**Status**: TensorFlow Datasets with prefetching
- No significant memory impact from data loading

### 10. Checkpoint Strategies ⏳ TO EXPLORE

**Approach**: Reduce checkpoint frequency or use streaming checkpoints

**Benefit**: Reduces peak memory during checkpoint saves

**Current**: Need to check Tunix checkpoint settings

## Implementation Plan

### Phase 1: Test Current Configuration ⏳ IN PROGRESS
- [x] Set mesh=(4,2)
- [ ] Run training with batch_size=16
- [ ] Monitor memory usage
- [ ] Check if OOM persists

### Phase 2: Enable Gradient Checkpointing (if Phase 1 OOMs)
- [ ] Research Tunix gradient checkpointing config
- [ ] Add to config file
- [ ] Test with batch_size=16
- [ ] Measure memory impact

### Phase 3: Add Gradient Accumulation (if still insufficient)
- [ ] Set gradient_accumulation_steps=2 or 4
- [ ] Reduce batch_size proportionally to maintain effective batch size
- [ ] Test training

### Phase 4: Explore Advanced Optimizations
- [ ] Investigate FlashAttention availability
- [ ] Check for fused kernel opportunities
- [ ] Consider sequence parallelism for long sequences

### Phase 5: Scale Up
- [ ] Once stable at batch_size=16, gradually increase
- [ ] Target: batch_size=32 with num_generations=8
- [ ] Run full 100-step training

## Expected Results

### Conservative Estimate (with current optimizations):
- Mesh (4,2): Should fit batch_size=16
- + Gradient checkpointing: Should fit batch_size=24-32
- + Gradient accumulation: Can achieve effective batch_size=32+

### Optimistic Estimate (with all optimizations):
- Could potentially fit batch_size=32 with num_generations=8
- Or increase num_generations to 12-16 at batch_size=16

## Monitoring & Validation

**Key Metrics to Track**:
1. Peak HBM usage per chip
2. Training throughput (steps/sec)
3. Model convergence (reward metrics)
4. Wandb logs: actor/train/loss, kl, perplexity

**Success Criteria**:
- ✅ No OOM errors
- ✅ Stable training for 100+ steps
- ✅ Reasonable throughput (>0.5 steps/sec)
- ✅ Model learns (reward increases over time)

## Notes for Multi-Machine/Pod Deployment

When scaling to multiple machines with pods:

1. **Automated Setup**: Use `scripts/setup_tpu.sh` for consistent environment
2. **Mesh Configuration**: Adjust for total device count (e.g., 2x v6e-8 = 16 devices → mesh=(8,2) or (4,4))
3. **Data Parallelism**: Ensure dataset sharding across pods
4. **Synchronization**: Verify gradient synchronization across pods
5. **Checkpoint Coordination**: Centralized checkpoint storage (GCS bucket)

## References

- Tunix Documentation: https://github.com/google/tunix
- JAX Parallelism Guide: https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html
- GRPO Paper: https://arxiv.org/abs/2402.03300
- Qwen3 Model Card: https://huggingface.co/Qwen/Qwen3-1.7B
