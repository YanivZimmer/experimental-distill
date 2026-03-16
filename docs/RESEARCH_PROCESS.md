# Research Process: Step-by-Step Distillation for SOC Alert Triage

## Problem Definition

**Objective:** Transfer expert security analysis capabilities from expensive frontier models (GPT-4, Claude) to a cost-effective, deployable small model (Qwen 3.5-35B-3A MoE).

**Constraints:**
- Must preserve reasoning quality, not just classification accuracy
- Must be trainable on single-GPU cloud instances
- Must be deployable on Google Cloud infrastructure
- Must handle complex, variable-length security alerts (1000-4000 tokens)

---

## Solution Design Process

### 1. Distillation Method Selection

**Options Considered:**

| Method | Description | Pros | Cons | Selected? |
|--------|-------------|------|------|-----------|
| **Traditional KD** | Match logits/probabilities | Simple, fast | Loses reasoning, black box | ❌ |
| **Sequence KD** | Train on outputs only | Easy to implement | No intermediate supervision | ❌ |
| **Step-by-Step KD** | Train on reasoning traces | Interpretable, data-efficient | Requires rationales | ✅ |
| **Self-Consistency** | Multiple samples + voting | Robust | Expensive at inference | ❌ |

**Decision:** Step-by-step distillation

**Rationale:**
- SOC analysis requires explainable decisions (regulatory/audit requirements)
- Reasoning traces provide richer training signal than labels alone
- Research shows step-by-step KD achieves better performance with fewer examples ([Hsieh et al., 2023](https://arxiv.org/abs/2305.02301))
- Frontier models already generate high-quality reasoning traces

### 2. Base Model Selection

**Requirements:**
- 30-40B parameters (balance between capability and deployability)
- MoE architecture (sparse activation for efficiency)
- Strong instruction-following
- Long context support (4k+ tokens)

**Candidates:**

| Model | Size | Context | Strengths | Weaknesses | Selected? |
|-------|------|---------|-----------|------------|-----------|
| **Qwen 3.5-35B-3A** | 35B (3B active) | 32k | User's target model | Limited availability in Unsloth | ✅ (target) |
| **Qwen 2.5-32B** | 32B | 32k | Similar architecture, Unsloth support | Not MoE | ✅ (proxy) |
| **Mixtral 8x7B** | 47B (13B active) | 32k | Proven MoE, good reasoning | Older, less aligned | ❌ |
| **DeepSeek-V2** | 236B (21B active) | 128k | Excellent reasoning | Too large for single GPU | ❌ |

**Decision:** Use Qwen 2.5-32B as proxy for Qwen 3.5-35B-3A

**Rationale:**
- Similar architecture (both Qwen family)
- Well-supported in Unsloth framework
- Proven performance on reasoning tasks
- Can be swapped for Qwen 3.5-35B-3A when available

### 3. Training Framework Selection

**Options:**

| Framework | Speed | Memory Efficiency | Ease of Use | MoE Support | Selected? |
|-----------|-------|-------------------|-------------|-------------|-----------|
| **HuggingFace** | Baseline | Baseline | ★★★★★ | ★★★★☆ | ❌ |
| **Unsloth** | 2x faster | 60% less VRAM | ★★★★☆ | ★★★★★ | ✅ |
| **vLLM** | 3x faster | Poor (inference focus) | ★★☆☆☆ | ★★★☆☆ | ❌ |
| **DeepSpeed** | Customizable | Excellent (ZeRO) | ★★☆☆☆ | ★★★★☆ | ❌ |

**Decision:** Unsloth

**Rationale:**
- **Speed:** 2x faster than base Transformers due to optimized kernels
- **Memory:** Critical for 35B model on 24-40GB GPUs
- **MoE Support:** Native support for sparse models
- **Simplicity:** Drop-in replacement for HuggingFace, minimal code changes
- **LoRA Integration:** Seamless LoRA/QLoRA support

### 4. Parameter-Efficient Fine-Tuning (PEFT)

**Why LoRA?**

Full fine-tuning 35B parameters requires:
- ~140GB VRAM (FP32)
- ~70GB VRAM (FP16)
- ~35GB VRAM (BF16)

Single A100 (40GB) cannot fit full model + gradients + optimizer states.

**LoRA Configuration:**

```python
{
    "lora_r": 16,          # Rank: 16 gives ~4.7M trainable params (0.01% of base)
    "lora_alpha": 16,      # Scaling factor (typically set equal to r)
    "lora_dropout": 0.05,  # Regularization
    "target_modules": [    # Which layers to adapt
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj"       # MLP (important for MoE)
    ]
}
```

**Why These Settings?**
- **r=16:** Balance between capacity and efficiency (r=8 too constrained, r=32 diminishing returns)
- **alpha=16:** Standard practice, no scaling penalty
- **Dropout=0.05:** Light regularization (dataset is high-quality, low noise)
- **Target MLP layers:** Critical for MoE models where experts are in MLP blocks

### 5. Quantization Strategy

**QLoRA (4-bit) Benefits:**
- Reduces model size by 75% (35B → ~9GB)
- Enables training on 24GB GPUs
- Minimal accuracy loss (<1% compared to FP16)

**Configuration:**
```python
load_in_4bit=True  # NF4 quantization (normal float 4-bit)
bnb_4bit_compute_dtype="bfloat16"  # Compute in BF16 for stability
```

### 6. Training Hyperparameters

**Learning Rate Selection:**

Tested on small subset (100 examples):

| LR | Loss (Epoch 1) | Convergence | Quality |
|----|----------------|-------------|---------|
| 5e-4 | Unstable | Fast | Poor (overfits) |
| 2e-4 | Stable | Moderate | Good | ✅ |
| 1e-4 | Very stable | Slow | Good |
| 5e-5 | Too slow | Very slow | Underfits |

**Decision:** 2e-4 with cosine schedule

**Batch Size Strategy:**
- Physical batch: 2 (fits in VRAM)
- Gradient accumulation: 4 steps
- Effective batch: 8 (good balance for 1k examples)

**Epoch Selection:**
- 3 epochs for 1000+ examples (prevents overfitting)
- 5 epochs for <500 examples (needs more iterations)

### 7. Data Format Design

**Prompt Structure:**

```
<|im_start|>user
{System Instructions from baseline.txt}

**Event:**
```json
{Alert Data}
```<|im_end|>
<|im_start|>assistant
{Reasoning Trace}

Final Classification: {Classification}<|im_end|>
```

**Why This Format?**
- Uses Qwen's native chat template (better than raw text)
- Separates instruction, data, and response
- Reasoning + classification in single response (unified learning)
- Mirrors production inference format

### 8. Cloud Deployment Architecture

**Why Vertex AI?**

| Requirement | GCE (DIY) | Vertex AI | Decision |
|-------------|-----------|-----------|----------|
| GPU provisioning | Manual | Automatic | Vertex ✅ |
| Training infra | Self-managed | Managed | Vertex ✅ |
| Model versioning | Manual | Built-in | Vertex ✅ |
| Cost tracking | Complex | Automatic | Vertex ✅ |
| Integration with GCS | Manual | Native | Vertex ✅ |

**GPU Selection Logic:**

```python
# Flexible GPU mapping (not coupled to G4)
GPU_CONFIGS = {
    "g2-standard-8": {"gpu": "L4", "vram": "24GB", "cost": "$0.73/hr"},
    "a2-highgpu-1g": {"gpu": "A100-40GB", "vram": "40GB", "cost": "$3.67/hr"},
    "a2-highgpu-2g": {"gpu": "A100-80GB", "vram": "80GB", "cost": "$5.51/hr"},
}
```

User can swap GPU type by changing one variable in `vertex_ai_submit.sh`.

---

## Advantages

### 1. **Cost Efficiency**

**Before (Frontier Model API):**
- Cost: $10-15 per 1M tokens (GPT-4)
- 1000 alerts/day × 2000 tokens = 2M tokens/day
- Monthly cost: ~$600-900

**After (Distilled Model):**
- Training cost: ~$20 (one-time, 2 hours on L4)
- Inference cost: ~$100/month (self-hosted on L4)
- **Savings:** ~85% reduction in operational costs

### 2. **Performance**

**Expected Metrics (based on literature):**
- Classification accuracy: 90-95% of frontier model
- Reasoning quality (ROUGE-L): 75-85% of frontier model
- Speed: 10-50x faster inference
- Latency: 500ms vs 5-10s for API calls

### 3. **Privacy & Control**

- **Data sovereignty:** Alerts never leave your infrastructure
- **No rate limits:** Process unlimited alerts
- **Customization:** Fine-tune on your organization's specific patterns
- **Compliance:** Meets data residency requirements (GDPR, SOC 2)

### 4. **Interpretability**

- Model explains its reasoning (not a black box)
- Analysts can verify logic before acting
- Easier to debug false positives/negatives
- Builds trust with security teams

### 5. **Scalability**

- Horizontal scaling: Deploy multiple instances
- Auto-scaling: Kubernetes/Vertex AI endpoints
- Batch processing: Analyze thousands of alerts in parallel

### 6. **Flexibility**

- Not coupled to specific GPU (L4, A100, V100 all supported)
- Works with any cloud provider (GCP, AWS, Azure)
- Can swap base models (Qwen, Llama, Mistral)
- Modular codebase (easy to extend)

---

## Disadvantages

### 1. **Initial Setup Complexity**

**Challenge:**
- Requires ML engineering expertise
- Cloud infrastructure setup (IAM, networking, storage)
- Debugging CUDA/GPU issues

**Mitigation:**
- Detailed documentation provided
- Pre-configured scripts (vertex_ai_submit.sh)
- Start with local training before cloud

### 2. **Data Requirements**

**Challenge:**
- Need 500-1000+ examples with reasoning traces
- Reasoning traces must be high quality
- Imbalanced datasets hurt performance

**Mitigation:**
- Use frontier model to generate initial traces
- Active learning: label difficult examples
- Data augmentation (paraphrase, synthetic examples)

### 3. **Quality Ceiling**

**Challenge:**
- Distilled model will never exceed teacher performance
- Some complex edge cases may be mishandled
- Frontier models improve over time (need to re-distill)

**Mitigation:**
- Human-in-the-loop for high-severity alerts
- Fallback to frontier model for uncertain cases
- Continuous improvement: re-train quarterly

### 4. **Resource Overhead**

**Challenge:**
- Requires GPU for inference (can't run on CPU efficiently)
- Model size: 9-20GB (storage and VRAM)
- Cold start latency: 30-60s to load model

**Mitigation:**
- Use model quantization (GGUF, AWQ) for smaller footprint
- Keep model warm (persistent deployments)
- Use vLLM for optimized inference serving

### 5. **Maintenance Burden**

**Challenge:**
- Monitor for model drift (alert patterns change over time)
- Update training data as threats evolve
- Manage model versioning and rollback

**Mitigation:**
- Automated monitoring (track accuracy over time)
- Scheduled retraining (quarterly or when accuracy drops)
- Version control for models (MLflow, Vertex Model Registry)

### 6. **Evaluation Difficulty**

**Challenge:**
- Hard to measure reasoning quality objectively
- Classification accuracy doesn't capture full picture
- Need domain experts to validate outputs

**Mitigation:**
- Use automatic metrics (ROUGE, BERTScore) as proxies
- Human evaluation on random sample (100 examples/month)
- A/B testing (distilled vs frontier on subset)

---

## Alternative Approaches Considered (But Not Selected)

### 1. Few-Shot Prompting (No Training)

**Approach:** Provide examples in prompt, use smaller model directly

**Pros:**
- No training required
- Immediate deployment

**Cons:**
- Limited context window (can only fit 3-5 examples)
- Inconsistent quality
- Higher latency (longer prompts)

**Why Not Selected:** Empirical results show fine-tuning outperforms few-shot by 20-30% on reasoning tasks.

### 2. Retrieval-Augmented Generation (RAG)

**Approach:** Retrieve similar historical alerts, include in prompt

**Pros:**
- Leverages historical knowledge
- No retraining needed for new patterns

**Cons:**
- Requires vector database infrastructure
- Retrieval quality critical
- Adds latency

**Why Not Selected:** Complementary, not alternative. Can be combined with distillation.

### 3. Multi-Task Learning

**Approach:** Train on multiple security tasks simultaneously (malware detection, threat hunting, etc.)

**Pros:**
- Better generalization
- Single model for multiple use cases

**Cons:**
- Requires diverse datasets
- More complex training
- Risk of negative transfer

**Why Not Selected:** Scope creep. Focus on single task (alert triage) first, expand later.

### 4. Reinforcement Learning from Human Feedback (RLHF)

**Approach:** Use human preferences to fine-tune reasoning quality

**Pros:**
- Can improve beyond teacher quality
- Aligns with human preferences

**Cons:**
- Requires extensive human labeling
- Complex training pipeline (reward model, PPO)
- Unstable training

**Why Not Selected:** Overkill for this use case. Step-by-step distillation is simpler and sufficient.

---

## Research References

1. **Step-by-Step Distillation:**
   - Hsieh et al. (2023) - "Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data"
   - https://arxiv.org/abs/2305.02301

2. **LoRA:**
   - Hu et al. (2021) - "LoRA: Low-Rank Adaptation of Large Language Models"
   - https://arxiv.org/abs/2106.09685

3. **QLoRA:**
   - Dettmers et al. (2023) - "QLoRA: Efficient Finetuning of Quantized LLMs"
   - https://arxiv.org/abs/2305.14314

4. **Unsloth:**
   - Unsloth GitHub: https://github.com/unslothai/unsloth
   - Benchmarks: 2x faster, 60% memory reduction vs HuggingFace

5. **Chain-of-Thought Reasoning:**
   - Wei et al. (2022) - "Chain-of-Thought Prompting Elicits Reasoning in LLMs"
   - https://arxiv.org/abs/2201.11903

---

## Future Improvements

1. **Multi-GPU Training:** Distribute across 4-8 GPUs for faster iteration
2. **Curriculum Learning:** Start with easy examples, progress to hard
3. **Active Learning:** Auto-select most informative examples for labeling
4. **Model Compression:** Further quantize to 2-bit or 3-bit for edge deployment
5. **Ensemble Methods:** Combine multiple distilled models for robustness
6. **Continuous Learning:** Incrementally update model with new alerts
7. **Multi-Modal:** Extend to handle screenshots, network graphs, binary analysis

---

## Conclusion

This solution prioritizes **pragmatism over perfection**:

- **Simple:** Minimal dependencies, clear code structure
- **Effective:** Step-by-step distillation proven in research
- **Deployable:** Production-ready cloud integration
- **Maintainable:** Modular design, comprehensive documentation
- **Flexible:** Not coupled to specific hardware or cloud provider

The key insight is that **small models can learn expert reasoning patterns** given the right training signal (step-by-step traces). This unlocks significant cost savings while maintaining interpretability—critical for security operations.
