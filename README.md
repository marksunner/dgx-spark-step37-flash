# Step 3.7 Flash on a Single DGX Spark

Notes from running StepFun's Step 3.7 Flash (198B MoE) on a single NVIDIA DGX Spark with llama.cpp.

## TL;DR

| Metric | Result |
|--------|--------|
| **Model** | Step 3.7 Flash (198B params, ~11B active per token) |
| **Quantization** | Q4_K_S GGUF (~105 GB) |
| **Hardware** | 1× NVIDIA DGX Spark (GB10, 128 GB unified memory) |
| **Generation speed** | **26–27.5 tok/s** |
| **Prompt processing** | 90–107 tok/s |
| **Context window** | 128K tokens |
| **KV cache** | q4_0 (fits 128K context in memory) |
| **Tool calling** | Works (file I/O verified via Hermes agent) |
| **Agent use** | Tested with Hermes on a separate Spark |
| **CPU during inference** | ~0% (model runs on GPU only) |

## Background

We originally set out to test whether Step 3.7 Flash could run on two DGX Sparks using vLLM tensor parallelism (similar to our [DS4 Flash dual-Spark experiment](https://github.com/marksunner/dgx-spark-vllm-tp-benchmark)). The FP8 model crashed during worker initialization — Step 3.7 support in vLLM is very new and the dual-Spark approach proved fragile.

We then tried the GGUF path with llama.cpp on a single Spark. For comparison:

| Setup | Speed | Hardware | Complexity |
|-------|-------|----------|------------|
| DS4 Flash, vLLM TP=2 | 12.4 tok/s | 2× DGX Spark + QSFP cluster | High |
| Step 3.7 Flash, llama.cpp | 27 tok/s | 1× DGX Spark | Low |

The single-Spark approach turned out to be both faster and simpler. Worth considering before investing in multi-node infrastructure.

## Benchmarks

### Generation Speed

All tests at 128K context, Q4_K_S quantization, q4_0 KV cache.

| Test | Tokens | Speed | Total Time |
|------|--------|-------|------------|
| Short answer (math) | 57 | 27.5 tok/s | 2.3s |
| Identity check | 133 | 27.2 tok/s | 4.9s |
| Code generation | 2,000 | 26.2 tok/s | 76.6s |
| Long generation (essay) | 3,439 | 26.0 tok/s | 132.4s |

Speed was consistent across generation lengths in our tests.

### Prompt Processing

| Prompt Size | Speed |
|-------------|-------|
| Short (24–33 tokens) | 90–107 tok/s |

### Built-in Reasoning

Step 3.7 Flash includes chain-of-thought reasoning (thinking mode). The model produces internal reasoning tokens before its visible response. This adds latency but improves quality — especially for complex tasks, code generation, and agentic tool use.

## Agent Test: Hermes + Steve

We connected the Step 3.7 Flash backend to [Hermes](https://github.com/hermes-ai/hermes), an open-source AI agent framework, running as "Steve" on Discord.

**Test setup:** Step 3.7 Flash serving on one DGX Spark, Hermes agent framework running on a separate DGX Spark, connected via the OpenAI-compatible API over LAN.

**Results:**
- ✅ Chat responses — natural, coherent conversation
- ✅ File operations — successfully wrote a 1,500-word story to disk
- ✅ Tool calling — working out of the box (unlike DS4 Flash which had format mismatches)
- ✅ Reasoning — deep chain-of-thought on complex prompts

### CPU Headroom (Not Yet Tested on Same Machine)

During inference, the model runs entirely on the Blackwell GPU. CPU utilisation is **0%**, leaving the full 20-core ARM CPU idle. In theory, this means the agent framework (Hermes or similar) could run on the **same** Spark alongside the model — the same pattern that works for Qwen 3.5 122B + Hermes on a single Spark.

**We have not yet tested running both the model and agent on the same machine.** Our test used two separate Sparks. Based on the resource profile (GPU-only inference, idle CPU), co-location should work, but we'll update this section once verified.

## Quick Start

### Prerequisites

- NVIDIA DGX Spark (128 GB unified memory)
- Docker with NVIDIA Container Toolkit
- Docker Compose v2
- ~110 GB free disk space
- HuggingFace account (optional, for faster downloads)

### Setup

```bash
# Clone the Docker template
git clone https://github.com/stevibe/step37-flash-dgx-spark.git
cd step37-flash-dgx-spark

# (Optional) Add HuggingFace token for faster downloads
echo "HF_TOKEN=hf_your_token_here" > .env

# For 128K context (recommended for agent use)
cat >> .env << 'EOF'
MAX_MODEL_LEN=131072
CACHE_TYPE_K=q4_0
CACHE_TYPE_V=q4_0
EOF

# Build and start (first run downloads ~105 GB model)
docker compose up -d --build
```

The first startup:
1. Builds llama.cpp from StepFun's fork with Step 3.7 support (~10 min)
2. Downloads Q4_K_S GGUF shards from HuggingFace (~105 GB)
3. Loads model to GPU and starts serving

### Verify

```bash
# Check health
curl http://localhost:8000/v1/models

# Test generation
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "step37-flash-q4-k-s",
    "messages": [{"role": "user", "content": "Hello! What model are you?"}],
    "temperature": 0.2,
    "max_tokens": 500
  }'
```

### Context Window Options

| Context | KV Cache | Memory Fit | Use Case |
|---------|----------|------------|----------|
| 32K | q8_0 | Comfortable | Chat, short tasks |
| 128K | q4_0 | Good fit | Agent use, long documents |
| 256K | q4_0 | Tight | Maximum context (may need tuning) |

## Architecture Notes

### Why Q4_K_S?

Step 3.7 Flash is a 198B parameter Mixture-of-Experts model, but only ~11B parameters activate per token. The Q4_K_S quantization reduces the total model size from ~400 GB (FP16) to ~105 GB, fitting comfortably in the DGX Spark's 128 GB unified memory with room for KV cache.

The sparse MoE architecture means only a fraction of total parameters are active per token, which may help preserve quality under quantization. Our subjective impression was that output quality was good, but we haven't run formal quality benchmarks.

### Why llama.cpp over vLLM?

We tested both:

- **vLLM (FP8, TP=2, dual-Spark)**: Model loading crashed during worker initialization. Step 3.7 support in vLLM is bleeding-edge (custom patch, days old). Even when the infrastructure works, TP=2 over RDMA adds complexity and limits throughput.
- **llama.cpp (Q4_K_S, single Spark)**: StepFun maintains their own llama.cpp fork with native Step 3.7 support. Battle-tested engine, simple Docker setup, OpenAI-compatible API out of the box.

In our testing, the llama.cpp path was more straightforward and produced better results for this particular model on DGX Spark.

### GPU vs CPU Split

During inference on DGX Spark:
- **GPU (Blackwell GB10)**: Model weights, attention, MoE routing — 100% of inference
- **CPU (20-core ARM Grace)**: Idle during inference — available for agent frameworks, monitoring, or other services

This suggests the DGX Spark could potentially run a complete AI agent stack (model + framework) on a single machine, though we haven't verified this yet (see agent test section above).

## Atlas NVFP4 — Work in Progress

We are working on native NVFP4 support for Step 3.7 Flash through the [Atlas inference engine](https://github.com/Avarok-Cybersecurity/atlas), which uses custom Blackwell (SM121) CUDA kernels rather than generic quantization.

**Status:** Config parsing, kernel compilation, and GPU initialization all verified. Weight loading is blocked by a fused expert tensor format issue. See [PR #119](https://github.com/Avarok-Cybersecurity/atlas/pull/119) for details.

| Component | Status |
|-----------|--------|
| Config parser | ✅ Step 3.7 nested format fully parsed |
| CUDA kernels | ✅ 90 custom kernels compiled for GB10 |
| GPU initialization | ✅ All PTX modules loaded |
| EP topology | ✅ Local expert ranges calculated |
| Weight loading | ⚠️ Blocked — fused expert tensors need CPU-side slicing |

**Why this matters:** Atlas delivers 2–3× faster inference than vLLM on the same hardware for supported models. If the weight loading is resolved, Step 3.7 on Atlas could potentially exceed the 27 tok/s achieved by the llama.cpp approach above.

**Architecture overlap with existing Atlas models:**
- Sigmoid MoE routing → MiniMax M2 kernels (same head_dim=128, same partial RoPE 0.5)
- Shared experts → Qwen 3.5 infrastructure
- Mixed full/sliding attention → Gemma-4 pattern
- 3 MTP modules → MiniMax M2 MTP structure

The fused expert tensor format (all 288 experts packed into single tensors per projection) is the remaining engineering challenge — it requires EP-aware loading that can slice the fused blob before GPU upload.

## Comparison with Other Models on DGX Spark

| Model | Quant | Speed | Context | Sparks | Notes |
|-------|-------|-------|---------|--------|-------|
| Qwen 3.5 122B | INT4+FP8 hybrid | 42–47 tok/s | 16–131K | 1 | Faster raw throughput |
| Step 3.7 Flash | Q4_K_S | 27 tok/s | 128K | 1 | Larger model, strong agentic scores |
| DeepSeek V4 Flash | FP4+FP8 | 12.4 tok/s | 65K | 2 | Requires dual-Spark cluster |
| Qwen 3.5 397B | NVFP4 | ~10 tok/s | 16K | 4 | Requires quad-Spark cluster |

## Credits

- **[StepFun AI](https://stepfun.com/)** — Step 3.7 Flash model and llama.cpp fork
- **[stevibe](https://github.com/stevibe/step37-flash-dgx-spark)** — Docker Compose template for DGX Spark
- **[eugr](https://github.com/eugr/spark-vllm-docker)** — vLLM Docker recipes (used for the FP8 attempt)
- **Nic (albond)** — whose question prompted this investigation, and whose Qwen 122B hybrid recipe has been invaluable
- **[Atlas Inference Engine](https://github.com/Avarok-Cybersecurity/atlas)** — Pure Rust inference with custom Blackwell kernels (NVFP4 work in progress)
- **NVIDIA DGX Spark community** — for the shared knowledge that makes this possible

## License

This write-up and benchmark data are released under [MIT License](LICENSE).

The Step 3.7 Flash model is licensed under [Apache 2.0](https://huggingface.co/stepfun-ai/Step-3.7-Flash-FP8/blob/main/LICENSE) by StepFun AI.
