# The serving track

Make SGLang serve Muse Spark Open faster. Your submission is a patch to `sglang/`:
SGLang at commit `297e54fd`, its v0.5.23 release.

    okc serve diff                 # write your changes to patch.diff
    okc serve check patch.diff     # an unscored run at one load
    okc serve submit patch.diff    # a scored run

## Runs

- A scored run uses a whole 8×H200 node for up to 90 minutes. It serves multi-turn agentic
  coding conversations, sampled greedily (temperature 0), at 1, 4, 16 and 32
  concurrent users.
- A check run takes about 25 minutes at 16 users; pick another load with `--users 1|4|32`.
  It checks correctness and compares your speed with stock's. It doesn't count toward the
  leaderboard.
- You get 1 scored run and 3 check runs a day, reset at 00:00 UTC. `okc serve status` shows how
  many you have left and your runs.
- A patch or flag that's refused (a 422) doesn't use a run. A run whose server crashes does,
  including on a flag SGLang doesn't know. A run that fails for our infrastructure's reasons doesn't.
- The serving leaderboard is private until the track is over.

## Rules

- The patch has to be lossless: no lower precision, approximations or skipped work beyond what
  stock SGLang does.
- Your outputs must agree with stock SGLang's, and every request returns exactly the tokens it asks for.
- Your build can use at most 2 GB more GPU memory per GPU than stock, at peak. Raising
  `--mem-fraction-static` counts: each 0.01 is about 1.4 GB.
- Your time to first token can be at most 2× stock's at each load.
- Only source files and JSON configs under `python/sglang/`, and nothing under
  `python/sglang/srt/model_loader/`. No renames or binary changes, at most 400 KB, and no added
  line over 400 characters.
- Readable source: nothing compressed, encoded or obfuscated.
- Write it like an SGLang pull request: nothing special-cased to the challenge's workload or
  load levels, and nothing that breaks other models or GPUs.

The full rules: https://openkernelchallenge.dev/rules

## Score

At each load, your throughput (tokens/s per GPU) at stock's per-user speed on the same node,
divided by stock's. Your score is the geometric mean over the loads, so stock scores 1.00.

## Launch flags

Your build starts with the judge's flags (under Running it yourself). Add, change or drop them
in `flags`, or with `--flag` on check and submit. The ones `flags` lists as fixed can't be changed.

## Where to start

All under `sglang/python/sglang/srt/`:

- `models/muse_spark.py`, `layers/muse_spark/`: the model.
- `layers/moe/moe_runner/marlin.py`, `layers/moe/fused_moe_triton/fused_marlin_moe.py`: the
  routed experts (Marlin W4A16), the largest share of a decode step from 16 users up.
- `layers/attention/flashattention_backend.py`: attention over the FP8 KV cache.
- `speculative/dflash_worker_v2.py`, `speculative/dflash_utils.py`: DFlash speculative decoding.
- `managers/scheduler.py`, `managers/schedule_policy.py`: batching and scheduling.
- `model_executor/model_runner.py`, `model_executor/cuda_graph_config.py`: CUDA graphs.

## Running it yourself

On an 8×H200 node, in the judge's image:

    docker pull lmsysorg/sglang@sha256:f832b5c39559c9caa900b22ad37c76fcce66470676855c42052102c537812f1f

Install your checkout (`pip install -e sglang/python --no-deps`) and start the server with
the judge's flags:

    SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 SGLANG_TIMEOUT_KEEP_ALIVE=1800 python -m sglang.launch_server \
      --model-path meta-models/Muse-Spark-Open-816B-A42B-NVFP4 \
      --speculative-draft-model-path RadixArk/Muse-Spark-Open-DFlash2 --tp-size 8 --ep-size 8 \
      --quantization modelopt_fp4 --moe-runner-backend marlin --mem-fraction-static 0.85 \
      --speculative-algorithm DFLASH --speculative-num-draft-tokens 8 \
      --speculative-draft-attention-backend fa4 --mm-feature-transport cpu \
      --enable-hierarchical-cache --hicache-size 160 --chunked-prefill-size 16384 \
      --context-length 131072 --enable-cache-report --incremental-streaming-output \
      --random-seed 1234 --max-running-requests 32 --cuda-graph-max-bs-decode 32

`python -m sglang.bench_serving --backend sglang --max-concurrency 16` is a start for
generating load.
