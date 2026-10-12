# Implementation Guide: Lever N Prefill/Decode Interleaving

**Target System:** `qwen-dflash` (Qwen3.8-27B on 2× Blackhole P150, TP=2, DFlash2 speculative decoding)  
**Objective:** Eliminate the 13–17s multi-user decode freezes caused by unchunked prompt prefill by interleaving 2,048-token prefill chunks with decode steps.  
**Reference Origin:** Thatch-cloud branches `lever-n-prefill-decode-interleave`, `tp4/lever-n`, and `tp4/w2-levern-prefix`.

---

## 1. Architectural Overview & Problem Statement

### 1.1 The Root Cause of the Stall
In the default `tp2-dflash2` profile:
1. When user $B$ submits a 45,000-token prompt while users $A, C, D$ are actively decoding, the entire 45k-token prompt is executed as a **single unbroken prefill pass** lasting ~14 seconds.
2. During those 14 seconds, the device is 100% occupied by the prefill pipeline. Decode throughput for all other active requests drops to **0.0 tok/s**.
3. In a 4-user scenario where users arrive staggered or in multi-turn conversations, decoding users spend up to **31% to 40% of their total wall-clock time frozen**.

### 1.2 The Lever N Solution
Lever N breaks long prefills into **2,048-token chunks** and alternates between processing one prefill chunk and executing decode steps for active users:
```
[User B: Chunk 0..2048] -> [All Users: Decode Step] -> [User B: Chunk 2048..4096] -> [All Users: Decode Step] ...
```
* **Result:** Instead of a continuous 14-second freeze, decoding users experience a pause of only **~500 ms per chunk**, maintaining continuous streaming output.

---

## 2. The Four Pillars of the Implementation

Implementing Lever N requires coordinated changes across four components:

| Component | Target File | Purpose |
|---|---|---|
| **1. Plugin Policy** | `vllm_tt_plugin/platform.py` | Permit chunked prefill on block-output speculative models. |
| **2. Engine Scheduler** | `vllm_tt_plugin/scheduler.py` | (a) Enforce one-in-flight prefill; (b) Alternate between prefill chunk and decode steps. |
| **3. Model Core** | `models/demos/blackhole/qwen36/tt/model.py` | Add `prefill_paged_slots_range` for resumable $[start, end)$ chunk execution. |
| **4. Speculative Serving** | `models/demos/blackhole/qwen36/tt/qwen36_vllm_dflash.py` | Preserve DFlash drafter hidden taps across engine step boundaries. |

---

## 3. Step-by-Step Implementation

### Step 1: Allow Chunked Prefill in the Plugin Platform Policy

**File:** `vllm_tt_plugin/platform.py` (inside the container at `/opt/tt-venv/lib/python3.12/site-packages/vllm_tt_plugin/platform.py`)

#### Problem:
By default, `_apply_chunked_prefill_policy` disables chunked prefill if `output_tokens_per_step > 1` (which is `32` for `tp2-dflash2`):
```python
# Existing code in platform.py:
if (
    not supports_chunked_prefill
    or not scheduler_config.enable_chunked_prefill
    or output_tokens_per_step > 1  # <-- BLOCKS DFLASH2!
):
    ...
    _disable_chunked_prefill(vllm_config, reason)
    return
```

#### Change:
Allow chunked prefill for adaptive block-output models when `TT_PREFILL_DECODE_INTERLEAVE=1`:
```python
is_adaptive_block = model_capabilities.get("tt_adaptive_block_output", False)
allow_interleave = os.environ.get("TT_PREFILL_DECODE_INTERLEAVE", "0") == "1"

if not supports_chunked_prefill or not scheduler_config.enable_chunked_prefill:
    _disable_chunked_prefill(vllm_config, model_desc)
    return

if output_tokens_per_step > 1 and not (is_adaptive_block and allow_interleave):
    _disable_chunked_prefill(vllm_config, f"{model_desc} (block-output)")
    return
```
And in `Qwen36DFlashForCausalLM.model_capabilities` (`qwen36_vllm_dflash.py`):
```python
"supports_chunked_prefill": True,
```

---

### Step 2: Scheduler Policy: One-In-Flight & Alternation

**File:** `vllm_tt_plugin/scheduler.py`

#### 2.1 The One-In-Flight Invariant
The GDN prefill recurrence state and RoPE tables on Tenstorrent hardware use a **single-occupancy scratch space** during prefill. If a second prompt begins prefill while a first prompt is only partially prefilled, the scratch is corrupted.

Wrap `TTScheduler._schedule_prefill_only`:
* When any partial prefill (`r.is_prefill_chunk`) is in flight, **hide the `waiting` queue** so no new request is admitted until the in-flight prefill reaches its final token.

```python
def _schedule_prefill_only(self):
    # Hide waiting queues if any request is currently mid-chunk prefill
    has_partial = any(r.is_prefill_chunk for r in self.running)
    hidden = {}
    if has_partial:
        for q_name in ('waiting', 'skipped_waiting'):
            q = getattr(self, q_name, None)
            if q is not None:
                hidden[q_name] = q
                setattr(self, q_name, type(q)())  # empty deque/list

    try:
        return super()._schedule_prefill_only()
    finally:
        for q_name, original_q in hidden.items():
            setattr(self, q_name, original_q)
```

#### 2.2 Decode Alternation in `TTScheduler.schedule`
In `TTScheduler.schedule`: when there is a partial prefill in flight AND active decoding requests, yield every $N$ prefill chunks to run a decode step:

```python
# In TTScheduler.schedule(self):
has_pending_prefill = ...
has_running_decode = any(not r.is_prefill_chunk for r in self.running)
has_partial_prefill = any(r.is_prefill_chunk for r in self.running)

interleave_on = os.environ.get("TT_PREFILL_DECODE_INTERLEAVE", "1") == "1"
steps_per_chunk = int(os.environ.get("TT_DECODE_STEPS_PER_PREFILL_CHUNK", "1"))

if interleave_on and has_partial_prefill and has_running_decode:
    streak = getattr(self, "_prefill_streak", 0)
    credit = getattr(self, "_decode_credit", 0)
    
    if streak >= 1 and credit < steps_per_chunk:
        self._decode_credit = credit + 1
        # Yield to decode!
        result = self._schedule_decode_only()
        return self._finalize_scheduler_output(result)
    else:
        self._decode_credit = 0
        self._prefill_streak = streak + 1
else:
    self._prefill_streak = 1
    self._decode_credit = 0

if has_pending_prefill:
    prefill_result = self._schedule_prefill_only()
    ...
```

---

### Step 3: Implement `prefill_paged_slots_range` in the Model

**File:** `models/demos/blackhole/qwen36/tt/model.py`

#### Implementation:
Add the resumable range method to `Qwen36Model`:
```python
def prefill_paged_slots_range(
    self, token_ids_list, page_table, empty_slots, starts, ends, valid_lens=None
):
    """Executes a 2048-token prefill slice [start, end) for an in-flight prompt."""
    assert self.num_devices > 1, "prefill_paged_slots_range requires TP > 1"
    N = len(token_ids_list)
    assert len(empty_slots) == N and len(starts) == N and len(ends) == N

    # Enforce single-occupancy of GDN prefill scratch
    resumed = [u for u in range(N) if int(starts[u]) > 0]
    if resumed and N > 1:
        raise ValueError(f"Lever N: Resumed prefill cannot batch multiple prompts: {starts}")

    # Sequence continuity guard
    expected_start = getattr(self, "_qwen_lever_n_next_start", None)
    if resumed:
        start_0 = int(starts[0])
        if expected_start is None or start_0 != expected_start:
            raise ValueError(
                f"Lever N sequence error: resumed at {start_0}, expected {expected_start}"
            )

    pt = page_table if isinstance(page_table, torch.Tensor) else ttnn.to_torch(page_table)
    comp = ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
    dn_states = [layer.attention for layer in self.layers if not layer.is_full_attention]

    prev_scratch = self._bind_gdn_prefill_scratch()
    host_logits = [None] * N
    finished = []

    try:
        for u in range(N):
            toks = token_ids_list[u]
            actual = int(valid_lens[u]) if valid_lens is not None else toks.shape[1]
            start = int(starts[u])

            # Execute traced chunked prefill with exact offset
            lg = self.prefill_traced_chunked(
                toks[:, :actual], pt[u : u + 1], actual_len=actual, start=start
            )
            host_logits[u] = (
                ttnn.to_torch(lg, mesh_composer=comp)
                .reshape(-1, self.args.vocab_size)[:1]
                .float()
                .view(1, 1, -1)
            )
            ttnn.deallocate(lg)
            
            # Record absolute position covered for next chunk validation
            self._qwen_lever_n_next_start = actual
            finished.append(
                (
                    u,
                    [ttnn.to_torch(dn.rec_state, mesh_composer=comp) for dn in dn_states],
                    [[ttnn.to_torch(c, mesh_composer=comp) for c in dn.conv_states] for dn in dn_states],
                )
            )
    finally:
        self._unbind_gdn_prefill_scratch(prev_scratch)

    # Persist intermediate GDN states into user slot
    for u, rec_snap, conv_snap in finished:
        self._write_gdn_slot(int(empty_slots[u]), rec_snap, conv_snap)

    return host_logits
```

---

### Step 4: DFlash Drafter Tap Capture Across Step Boundaries

**File:** `models/demos/blackhole/qwen36/tt/qwen36_vllm_dflash.py`

#### Problem:
DFlash2 requires hidden activations from 5 specific transformer layers (`TAP_LAYERS = (5, 19, 33, 47, 61)`) from the prompt to initialize its drafter context ring (`dflash2.py`). 
* In a one-shot prefill, all taps are captured in one pass via `on_chunk`.
* When prefill is chunked across multiple engine steps, the capture hook must **not be re-initialized at cursor 0**. It must persist its cursor and state across calls until the final chunk completes.

#### Implementation:
1. In `qwen36_vllm_dflash.py:prefill_forward`:
   Inspect `start_pos = kwargs.get("start_pos")`.
   * If `start_pos is None or start_pos[0] == 0`: Initialize new `DFlashPrefillCapture(slot=phys, prompt_len=T)`.
   * If `start_pos[0] > 0`: Resume the existing `DFlashPrefillCapture` for this slot, appending taps starting at `start_pos[0]`.
2. Do not call `dec.begin(...)` until the **last** chunk of the prompt finishes (`actual >= prompt_len`). Intermediate chunks return dummy/empty anchor logits to vLLM, which vLLM ignores because `r.is_prefill_chunk` is true.

---

## 5. Deployment and Validation Settings

When launching the modified server, configure these environment variables:

```bash
# Enable Lever N interleaving
export TT_PREFILL_DECODE_INTERLEAVE=1

# Execute 1 decode step between each 2,048-token prefill chunk
export TT_DECODE_STEPS_PER_PREFILL_CHUNK=1

# Keep chunk size aligned to hardware trace buckets (MUST BE 2048)
export MAX_PREFILL_CHUNK_SIZE=2048

# vLLM launch arguments must specify max_num_batched_tokens matching chunk size:
# vllm serve ... --max-num-batched-tokens 2048
```

---

## 6. Safety Invariants & Tripwires (From Thatch Lessons Learned)

To avoid regressions and silent corruption during implementation:

1. **`max_num_batched_tokens == 2048` exactly:** A smaller budget causes mid-chunk continuations that break GDN convolution state. A larger budget exceeds the pre-captured chunk trace buffer.
2. **Scratch Single-Occupancy:** Never allow two requests to have `is_prefill_chunk == True` at the same time. The scheduler guard in Step 2 is mandatory.
3. **Cursor Continuity Assertion:** Keep the assertion `assert start_u == self._qwen_lever_n_next_start` in `prefill_paged_slots_range`. If chunks are scheduled out of order, fail loudly rather than generating gibberish tokens.
4. **Do Not Touch Verify Traces:** Ensure that yielding between chunks does not invoke program recompilation. All chunk prefill traces and decode traces must be warmed up at server startup.
