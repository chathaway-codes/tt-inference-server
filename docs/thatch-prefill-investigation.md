# Investigation Report: Qwen3.8-27B Inference Hangs & Thatch-Cloud Prefill Solutions

**Target System:** `qwen-dflash` container running `tt-model/qwen3.8-27b-p150x2:arm64` (profile `tp2-dflash2`)  
**Hardware:** 2× Tenstorrent Blackhole P150 cards (TP=2, `FABRIC_1D`, 2-link mesh descriptor)  
**External Research Repo:** `https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B`  
**Date:** 2026-10-12  

---

## 1. Executive Summary

The observed behaviour ("hangs for a few seconds, drafts a good number of tokens, hangs again") in your running setup is caused by **two distinct architectural mechanisms**, neither of which was correctly diagnosed by Gemini's advice:

1. **Short cyclic stalls (0.5–2 s) + token burst:** `QWEN36_DFLASH_SERVE_BLOCK=32` forces `decode_forward` in `qwen36_vllm_dflash.py` to loop speculative draft-and-verify iterations internally until every active sequence accumulates 32 tokens before yielding anything to vLLM.
2. **Long catastrophic freezes (13–17 s, zero tok/s across all requests):** Prompts in your agentic workload are 35,000–51,000 tokens. Each new turn does an eager unchunked prefill taking 13–17 seconds. Because vLLM chunked prefill is disabled for block-output models, **generation freezes completely for all 4 concurrent users** whenever any user starts a turn. Furthermore, automatic prefix caching is disabled in the TT backend for block-output models (0% hit rate), forcing every multi-turn prompt to re-prefill its entire history.

The Thatch-cloud repository contains **direct, measured research and prototypes on this exact issue**, specifically addressing the 13–40 s decode stall during concurrent 32k prefill.

---

## 2. Analysis of the Running Setup & Gemini's Initial Advice

### 2.1 Gemini's Advice vs Reality

| Gemini Suggestion | Feasibility | Technical Reality on Your Stack |
|---|---|---|
| **1. Reduce Draft Length ($K=3$ or $4$)** | Possible via env, but **ineffective/harmful** | Your drafter is DFlash2, not autoregressive MTP. DFlash drafts all $K=7$ tokens in a single parallel pass (`dflash2.py`). The target verify step runs a fixed 32-row tile ($4\text{ users}\times 8\text{ rows}$) on device regardless of $K$. Reducing $K$ cuts accepted yield (your log shows many rounds accepting 6–7 tokens: `histogram [0, 0, 2, 0, 3, 0, 0, 19]`) without reducing draft or verify time. Crucially, it does **not** change the 32-token batch emission threshold. |
| **2. Kernel Fusion & L1 Residency** | Already implemented | The Changh95 image already adopted Thatch's lane H work: `QWEN36_GDN_SPEC_FUSED=1` (fused two-tile gate read), traced decode/verify programs, and on-chip SDPA BF8. No low-hanging kernel optimization remains that would solve multi-second stalls. |
| **3. Pipelined / Asynchronous Drafting** | Already parallelized | DFlash is already a single-head block drafter (not 7 sequential steps). Overlapping draft $N+1$ with verify $N$ requires predicting draft features before verification reveals the accepted token trajectory—an unsolved algorithmic problem for DFlash. |
| **4. Disable Speculation (Pure Decode)** | Possible | Eliminates the 32-token burstiness, but drops decode throughput to ~18 tok/s and does **not** solve the 13–17 s prefill freeze when other users enter. |

---

## 3. Findings from `Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B`

Thatch-cloud's repository contains ~170 experiment logs, kernel grafts, and specialized branches exploring Blackhole P150 TP2 and TP4 configurations.

### 3.1 They Measured the Exact Same Decode Stall (`docs/ttft-decode-stall-2026-09-22.md`)

In their 4-user 32k prompt tests, Thatch observed:
* Prefill is strictly serial: ~13.1 s per user (2,423 tok/s).
* User 0 received token 1 at 13.5 s, then **froze for 39.5 s** while users 1, 2, and 3 ran their prefills.
* Decode stalls accounted for **31% of total wall clock time** (79.4 seconds of freeze across 4 users).
* Quote: *"The stall is forty times larger than everything the decode work has won... it is a worse user-visible defect than the decode rate."*

### 3.2 What Can We Steal? Detailed Candidate Evaluation

#### Candidate A: "Lever N" (Prefill/Decode Interleave)
* **What it is:** Makes prefill resumable in 2,048-token chunks via `prefill_paged_slots_range`, and introduces a custom vLLM scheduler (`OneInFlightScheduler` / M2) that alternates 1 decode step between prefill chunks.
* **Measured Gain:** Cuts the user-facing decode freeze from **13–40 s down to ~0.5 s** (the duration of one 2k chunk).
* **Branches:** `lever-n-prefill-decode-interleave`, `tp4/lever-n`, `tp4/levern-prefix`, `tp4/w2-levern-prefix`. Key files: `scripts/ci/dflash_prefill_window.py`, `scripts/ci/serving_one_in_flight.py`.
* **Feasibility for our stack:** **Medium-High effort, but the exact architectural solution.**
  * *Blockers to bridge:* 
    1. In our plugin (`platform.py:199`), `_apply_chunked_prefill_policy` explicitly disables chunked prefill if `output_tokens_per_step > 1`.
    2. GDN recurrence state in prefill uses a single-occupancy scratch buffer. Interleaving decode steps requires ensuring decode does not clobber the partially-filled prefill GDN state. Thatch solved this by establishing single-in-flight prefill with scratch isolation.
    3. Drafter tap capture across chunk boundaries: Thatch's `dflash_prefill_window.py` was specifically built to let `PrefillWindowCapture` span engine steps without losing the 2,048-token tap history.

#### Candidate B: GDN Prefix Caching & Sticky Sessions
* **What it is:** For multi-turn conversations (like your agent), re-prefilling 40k tokens every turn is wasteful. However, Qwen3.8's GDN layers have recurrent state ($\mathbf{S}_t = \alpha \mathbf{S}_{t-1} + \mathbf{k}_t \mathbf{v}_t^T$), so standard KV caching alone is insufficient—the GDN recurrence state and convolution carry must also be cached.
* **Thatch's work:**
  * Offline combined prefix cache (`docs/combined-prefix-cache.md`): Demonstrated **1.81× faster prefill** (5,989 tok/s effective) by checkpointing GDN recurrence state + `conv_carry` + drafter tap history.
  * `s2/sticky-sessions` & `tp4/prefix-tiers`: Pinned user slots so a returning session in an agent conversation retains its allocated KV pages and GDN state between turns.
* **Feasibility for our stack:** High value, but significant engineering. The TT plugin currently asserts `supports_prefix_caching = False` for block-output models.

#### Candidate C: Fused GDN Prefill Convolution Kernels
* **What it is:** Custom C++ / TT-Metal compute kernels for GDN prefill convolution (`docker/qwen-c2-graft/graft/gdn/gdn_prefill_conv_exact*.cpp`).
* **Measured Gain:** Minor (~2–4% prefill speedup).
* **Comparison with our stack:** Our current cold prefill throughput is **3,000–3,250 tok/s** at 35k–51k tokens. Thatch's published baseline on main was **2,565 tok/s at 64k** and **2,969 tok/s at 32k**. **Our prefill is already as fast or faster than Thatch's raw kernel throughput.** Kernel optimization will not solve the stall.

#### Candidate D: Raising Prefill Chunk Size (`MAX_PREFILL_CHUNK_SIZE`)
* **What it is:** Attempting to speed up prefill by increasing chunk size from 4,096 to 32,768 tokens.
* **Thatch's measured result (`docs/prefill-chunk-size-verdict-2026-09-22.md`):** **FAILED / COUNTERPRODUCTIVE.**
  * Prefill throughput dropped from 2,423 to 2,294 tok/s (−5.3%).
  * Decode round latency worsened from 256 ms to 297 ms (+15.8%).
  * Memory pressure triggered a new synchronized 1.2–2.2 s stall mid-decode.
* **Takeaway:** Do **not** increase prefill chunk size.

#### Candidate E: Fabric Ethernet Link Count
* **What it is:** Thatch's hardware uses 4 links (2 QSFP-DD cables).
* **Investigation of our stack:** The package's `p150_x2_2link_mesh_graph_descriptor.textproto` specifies `channels { count: 2 policy: RELAXED }`. This is a *floor* for initialization, not a ceiling. The fabric driver initializes all trained physical links it detects. If 4 physical links exist, the driver already enables all 4 routing planes. There is nothing to fix here.

---

## 4. Ranked Summary of Actionable Paths

```
+-----------------------------------------------------------------------------------+
| LEVER                            | EFFECT ON STALLS            | IMPLEMENTATION   |
+-----------------------------------------------------------------------------------+
| 1. QWEN36_DFLASH_RAGGED=1        | Eliminates 32-token burst   | Zero-code        |
|    (or SERVE_BLOCK=8/16)         | Smooths emission to ~100ms  | Env var toggle   |
+-----------------------------------------------------------------------------------+
| 2. Steal "Lever N" Interleaving  | Eliminates 13-17s freeze    | Model + Plugin   |
|    (from Thatch branch)          | Pre-fill yields to decode   | Graft (~500 LOC) |
+-----------------------------------------------------------------------------------+
| 3. Steal GDN Prefix Checkpoint   | Cuts prefill from 15s to 2s | High effort      |
|    (from Thatch prefix branches) | Reuses agent conversation   | Model + Plugin   |
+-----------------------------------------------------------------------------------+
| 4. Kernel / SDPA / Chunk tweaks  | Negligible / Counterprod.   | Do NOT pursue    |
+-----------------------------------------------------------------------------------+
```

### Immediate / Zero-Code Option (For immediate testing)
To eliminate the **bursty 32-token chunking** right now without codebase modifications:
* **Option A:** Start the container with `-e QWEN36_DFLASH_RAGGED=1`. The model and plugin already include the code for `tt_adaptive_block_ragged`. Instead of waiting for 32 committed tokens, it yields after every single speculative iteration (~3–5 tokens every ~100 ms).
* **Option B:** Lower `QWEN36_DFLASH_SERVE_BLOCK` from `32` to `8` or `16` in `serve-tp2-dflash2.sh`. This keeps the proven fixed-width protocol while reducing the burst wait time by 2–4×.

### Code-Level Lever to Steal (When ready for development)
* **Steal Thatch's Lever N interleave logic:** Port `dflash_prefill_window.py` and `serving_one_in_flight.py` to allow vLLM to schedule decode steps between 2,048-token prefill chunks. This directly addresses the 13–17 second freeze without requiring prefix caching.
