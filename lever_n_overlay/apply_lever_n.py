import sys
import os
import ast

sys.path.append("/home/charles/p3/Tenstorrent.Blackhole-Qwen3.8-27B/scripts/ci")
import lever_n_model_patch

overlay_dir = "/home/charles/p3/tt-inference-server/lever_n_overlay"

# ---------------------------------------------------------------------------
# 1. model.py
# ---------------------------------------------------------------------------
model_path = os.path.join(overlay_dir, "qwen36/tt/model.py")
with open(model_path) as f:
    m_src = f.read()

m_src = lever_n_model_patch.patch_model(m_src)

# Also patch prefill_for_spec in model.py
old_pfs = '    def prefill_for_spec(self, token_ids, page_table, actual_len, on_chunk, slot=0, device_copy=None):'
new_pfs = '    def prefill_for_spec(self, token_ids, page_table, actual_len, on_chunk, slot=0, device_copy=None, start=0):'
assert m_src.count(old_pfs) == 1, "prefill_for_spec signature not found once"
m_src = m_src.replace(old_pfs, new_pfs, 1)

old_b1_call = '                logits = self._prefill_for_spec_b1(token_ids, page_table, actual_len, on_chunk)'
new_b1_call = '                logits = self._prefill_for_spec_b1(token_ids, page_table, actual_len, on_chunk, start=start)'
assert m_src.count(old_b1_call) == 1, "prefill_for_spec b1 call not found once"
m_src = m_src.replace(old_b1_call, new_b1_call, 1)

old_b1_call_fallback = '        return self._prefill_for_spec_b1(token_ids, page_table, actual_len, on_chunk)'
new_b1_call_fallback = '        return self._prefill_for_spec_b1(token_ids, page_table, actual_len, on_chunk, start=start)'
assert m_src.count(old_b1_call_fallback) == 1, "prefill_for_spec b1 fallback call not found once"
m_src = m_src.replace(old_b1_call_fallback, new_b1_call_fallback, 1)

# Now patch _prefill_for_spec_b1 in model.py
old_b1_def = '''    def _prefill_for_spec_b1(self, token_ids, page_table, actual_len, on_chunk):
        """prefill_for_spec's body: one B=1 prompt into whatever GDN state is currently bound."""
        chunk_size = self._chunked_chunk_size or 2048
        num_full = actual_len // chunk_size
        tail_real = actual_len - num_full * chunk_size
        self._build_request_rope(token_ids[:, :actual_len], None)

        if num_full == 0:
            logits, hidden = self.prefill_masked_bucket(
                token_ids[:, :actual_len], page_table, actual_len=actual_len, chunk_start=0, return_hidden=True
            )
            on_chunk(hidden, 0, actual_len)
            ttnn.deallocate(hidden)
            return logits

        self._reset_gdn_state_for_new_sequence()
        last_hidden = None
        for c in range(num_full):'''

new_b1_def = '''    def _prefill_for_spec_b1(self, token_ids, page_table, actual_len, on_chunk, start=0):
        """prefill_for_spec's body: one B=1 prompt into whatever GDN state is currently bound."""
        chunk_size = self._chunked_chunk_size or 2048
        num_full = actual_len // chunk_size
        tail_real = actual_len - num_full * chunk_size
        if start == 0:
            self._build_request_rope(token_ids[:, :actual_len], None)

        if num_full == 0:
            logits, hidden = self.prefill_masked_bucket(
                token_ids[:, :actual_len], page_table, actual_len=actual_len, chunk_start=0, return_hidden=True
            )
            on_chunk(hidden, 0, actual_len)
            ttnn.deallocate(hidden)
            return logits

        if start == 0:
            self._reset_gdn_state_for_new_sequence()
        chunk_from = start // chunk_size
        chunk_to = num_full
        last_hidden = None
        for c in range(chunk_from, chunk_to):'''

assert m_src.count(old_b1_def) == 1, "_prefill_for_spec_b1 block not found once"
m_src = m_src.replace(old_b1_def, new_b1_def, 1)

# Guard last_hidden deallocations
old_tail_dealloc = '''        if tail_real > 0:
            ttnn.deallocate(last_hidden)'''
new_tail_dealloc = '''        if tail_real > 0:
            if last_hidden is not None:
                ttnn.deallocate(last_hidden)'''
if m_src.count(old_tail_dealloc) > 0:
    m_src = m_src.replace(old_tail_dealloc, new_tail_dealloc)

ast.parse(m_src)
with open(model_path, "w") as f:
    f.write(m_src)
print("model.py patched successfully!")

# ---------------------------------------------------------------------------
# 2. gdn/tp.py
# ---------------------------------------------------------------------------
gdn_path = os.path.join(overlay_dir, "qwen36/tt/gdn/tp.py")
with open(gdn_path) as f:
    g_src = f.read()
g_src = lever_n_model_patch.patch_gdn_slot_remap(g_src)
ast.parse(g_src)
with open(gdn_path, "w") as f:
    f.write(g_src)
print("gdn/tp.py patched successfully!")

# ---------------------------------------------------------------------------
# 3. qwen36_vllm.py
# ---------------------------------------------------------------------------
vllm_path = os.path.join(overlay_dir, "qwen36/tt/qwen36_vllm.py")
with open(vllm_path) as f:
    v_src = f.read()
v_src = lever_n_model_patch.patch_vllm_entry(v_src)
# Add supports_chunked_prefill: True to model_capabilities
old_caps = '''    model_capabilities = _ModelCapabilities(
        {
            "supports_prefix_caching": False,'''
new_caps = '''    model_capabilities = _ModelCapabilities(
        {
            "supports_chunked_prefill": True,
            "supports_prefix_caching": False,'''
assert v_src.count(old_caps) == 1, "qwen36_vllm caps not found once"
v_src = v_src.replace(old_caps, new_caps, 1)
ast.parse(v_src)
with open(vllm_path, "w") as f:
    f.write(v_src)
print("qwen36_vllm.py patched successfully!")

# ---------------------------------------------------------------------------
# 4. qwen36_vllm_dflash.py
# ---------------------------------------------------------------------------
dflash_path = os.path.join(overlay_dir, "qwen36/tt/qwen36_vllm_dflash.py")
with open(dflash_path) as f:
    d_src = f.read()

# Add supports_chunked_prefill: True to Qwen36DFlashForCausalLM.model_capabilities
old_df_caps = '''    model_capabilities = {
        **Qwen36ForCausalLM.model_capabilities,'''
new_df_caps = '''    model_capabilities = {
        **Qwen36ForCausalLM.model_capabilities,
        "supports_chunked_prefill": True,'''
assert d_src.count(old_df_caps) == 1, "dflash caps not found once"
d_src = d_src.replace(old_df_caps, new_df_caps, 1)

# Patch _spec_prefill signature and start handling
old_df_spec_pref = '''    def _spec_prefill(self, model, dec, phys, prompt, T, pt_row):
        """Eager tap-capturing prefill of ONE request into physical slot ``phys`` (masked bucket for a
        short prompt, 2048-token chunks + masked tail for a long one), each chunk's taps ingested into the
        drafter's ring for that slot. Returns host logits [1, vocab] (float)."""

        def on_chunk(hidden, chunk_start, valid_len):
            taps = model.take_dflash_eager_taps()
            if taps is None:
                raise RuntimeError("Qwen36DFlash: eager prefill captured no drafter taps (bucket trace gate on?)")
            dec.ingest_prompt(phys, taps, chunk_start + valid_len, chunk_start=chunk_start)

        dec.ctx_len[phys] = 0
        model._dflash_tap = True
        try:
            logits_dev = model.prefill_for_spec(prompt, self._spec_pref_pt(model, pt_row), T, on_chunk, slot=phys)'''

new_df_spec_pref = '''    def _spec_prefill(self, model, dec, phys, prompt, T, pt_row, start=0):
        """Eager tap-capturing prefill of ONE request into physical slot ``phys`` (masked bucket for a
        short prompt, 2048-token chunks + masked tail for a long one), each chunk's taps ingested into the
        drafter's ring for that slot. Returns host logits [1, vocab] (float)."""

        def on_chunk(hidden, chunk_start, valid_len):
            taps = model.take_dflash_eager_taps()
            if taps is None:
                raise RuntimeError("Qwen36DFlash: eager prefill captured no drafter taps (bucket trace gate on?)")
            dec.ingest_prompt(phys, taps, chunk_start + valid_len, chunk_start=chunk_start)

        if start == 0:
            dec.ctx_len[phys] = 0
        model._dflash_tap = True
        try:
            logits_dev = model.prefill_for_spec(prompt, self._spec_pref_pt(model, pt_row), T, on_chunk, slot=phys, start=start)'''

assert d_src.count(old_df_spec_pref) == 1, "_spec_prefill block not found once"
d_src = d_src.replace(old_df_spec_pref, new_df_spec_pref, 1)

# Patch prefill_forward in qwen36_vllm_dflash.py
old_df_pf = '''        empty_slots = kwargs.get("empty_slots")
        logical = [int(s) for s in empty_slots] if empty_slots is not None else list(range(N))
        pt = torch.as_tensor(page_table)
        out = []
        for u in range(N):
            phys = self._phys[logical[u]]
            if dec.active[phys]:
                # vLLM released the previous occupant before reusing its slot; close its session if not.
                dec.end(phys)
            self._carry[phys], self._stopped[phys], self._prev_tail[phys] = [], False, None
            T = plens[u]
            prompt = torch.as_tensor(tokens)[u : u + 1, :T].to(torch.int32)
            row = pt[u].reshape(-1).clone()
            logger.info(f"Prefilling slot {phys} up to {T} tokens (TP eager spec prefill)")
            lt = self._spec_prefill(model, dec, phys, prompt, T, row)
            out.append(lt.view(1, 1, -1))
            self._pending[phys] = (T, row)
        logger.info(f"Finished prefill of {N} request(s), starting decode...")
        return torch.cat(out, dim=0), torch.zeros(N, dtype=torch.long)'''

new_df_pf = '''        empty_slots = kwargs.get("empty_slots")
        logical = [int(s) for s in empty_slots] if empty_slots is not None else list(range(N))
        pt = torch.as_tensor(page_table)
        start_pos = kwargs.get("start_pos")
        starts = [int(s) for s in start_pos] if start_pos is not None else [0] * N
        out = []
        for u in range(N):
            phys = self._phys[logical[u]]
            start = starts[u]
            if start == 0:
                if dec.active[phys]:
                    # vLLM released the previous occupant before reusing its slot; close its session if not.
                    dec.end(phys)
                self._carry[phys], self._stopped[phys], self._prev_tail[phys] = [], False, None
            T = plens[u]
            prompt = torch.as_tensor(tokens)[u : u + 1, :T].to(torch.int32)
            row = pt[u].reshape(-1).clone()
            logger.info(f"Prefilling slot {phys} up to {T} tokens (start={start}) (TP eager spec prefill)")
            lt = self._spec_prefill(model, dec, phys, prompt, T, row, start=start)
            out.append(lt.view(1, 1, -1))
            self._pending[phys] = (T, row)
        logger.info(f"Finished prefill of {N} request(s), starting decode...")
        return torch.cat(out, dim=0), torch.zeros(N, dtype=torch.long)'''

assert d_src.count(old_df_pf) == 1, "dflash prefill_forward block not found once"
d_src = d_src.replace(old_df_pf, new_df_pf, 1)

ast.parse(d_src)
with open(dflash_path, "w") as f:
    f.write(d_src)
print("qwen36_vllm_dflash.py patched successfully!")

# ---------------------------------------------------------------------------
# 5. platform.py
# ---------------------------------------------------------------------------
plat_path = os.path.join(overlay_dir, "vllm_tt_plugin/platform.py")
with open(plat_path) as f:
    p_src = f.read()

old_plat = '''    supports_chunked_prefill = (
        model_capabilities.get("supports_chunked_prefill", False)
        if model_capabilities
        else False
    )
    output_tokens_per_step = (
        model_capabilities.get("output_tokens_per_step", 1) if model_capabilities else 1
    )
    if (
        not supports_chunked_prefill
        or not scheduler_config.enable_chunked_prefill
        or output_tokens_per_step > 1
    ):
        reason = model_desc
        if supports_chunked_prefill and output_tokens_per_step > 1:
            reason = f"{model_desc} (block-output)"
        _disable_chunked_prefill(vllm_config, reason)
        return

    # A chunk boundary inside a multimodal item would split its embeddings from
    # their positions. Only meaningful while prefill can be split, and vLLM
    # rejects the flag outright when one item exceeds the token budget, so it
    # stays off for every model that does not reach here.
    scheduler_config.disable_chunked_mm_input = True'''

new_plat = '''    supports_chunked_prefill = (
        model_capabilities.get("supports_chunked_prefill", False)
        if model_capabilities
        else False
    )
    output_tokens_per_step = (
        model_capabilities.get("output_tokens_per_step", 1) if model_capabilities else 1
    )
    is_adaptive_block = bool((model_capabilities or {}).get("tt_adaptive_block_output", False))
    allow_interleave = os.environ.get("TT_PREFILL_DECODE_INTERLEAVE", "0") == "1"

    if not supports_chunked_prefill or not scheduler_config.enable_chunked_prefill:
        _disable_chunked_prefill(vllm_config, model_desc)
        return

    if output_tokens_per_step > 1 and not (is_adaptive_block and allow_interleave):
        _disable_chunked_prefill(vllm_config, f"{model_desc} (block-output)")
        return

    if is_adaptive_block and allow_interleave:
        # Do not set disable_chunked_mm_input = True because Qwen text-only
        # has phantom nested vision config that would exceed max_num_batched_tokens=2048.
        logger.info(
            "Chunked prefill enabled for %s with Lever N interleave: max_num_batched_tokens=%d, "
            "long_prefill_token_threshold=%d.",
            model_desc,
            scheduler_config.max_num_batched_tokens,
            scheduler_config.long_prefill_token_threshold,
        )
        return

    # A chunk boundary inside a multimodal item would split its embeddings from
    # their positions. Only meaningful while prefill can be split, and vLLM
    # rejects the flag outright when one item exceeds the token budget, so it
    # stays off for every model that does not reach here.
    scheduler_config.disable_chunked_mm_input = True'''

assert p_src.count(old_plat) == 1, "platform _apply_chunked_prefill_policy block not found once"
p_src = p_src.replace(old_plat, new_plat, 1)

ast.parse(p_src)
with open(plat_path, "w") as f:
    f.write(p_src)
print("platform.py patched successfully!")

# ---------------------------------------------------------------------------
# 6. scheduler.py
# ---------------------------------------------------------------------------
sched_path = os.path.join(overlay_dir, "vllm_tt_plugin/scheduler.py")
with open(sched_path) as f:
    s_src = f.read()

# 6a. Alternation in _schedule_without_pending_output_resumes
old_sched_alt = '''        # Default mode:
        # Prefer prefill whenever prefill work is pending, so new requests are
        # admitted and partial prefills advance.
        if has_pending_prefill:
            prefill_result = self._schedule_prefill_only()'''

new_sched_alt = '''        # Default mode:
        # Prefer prefill whenever prefill work is pending, so new requests are
        # admitted and partial prefills advance.
        if has_pending_prefill:
            import os as _qwen_os
            _qwen_on = _qwen_os.environ.get("TT_PREFILL_DECODE_INTERLEAVE", "0") == "1"
            _qwen_partial = any(r.is_prefill_chunk for r in self.running)
            _qwen_streak = getattr(self, "_qwen_prefill_streak", 0)
            _qwen_credit = getattr(self, "_qwen_decode_credit", 0)
            if _qwen_on and _qwen_partial and has_running_decode:
                try:
                    _qwen_r = int(_qwen_os.environ.get("TT_DECODE_STEPS_PER_PREFILL_CHUNK", "1"))
                except ValueError:
                    _qwen_r = 1
                _qwen_yield = _qwen_streak >= 1 and _qwen_credit < _qwen_r
                if _qwen_yield:
                    self._qwen_decode_credit = _qwen_credit + 1
                else:
                    self._qwen_decode_credit = 0
                    self._qwen_prefill_streak = _qwen_streak + 1
                logger.info(
                    f"[PINDIAG] m2 alternation: yield_decode={_qwen_yield} "
                    f"streak={_qwen_streak} credit={_qwen_credit} r={_qwen_r}"
                )
                if _qwen_yield:
                    result = self._schedule_decode_only()
                    return self._finalize_scheduler_output(result)
            else:
                self._qwen_prefill_streak = 1
                self._qwen_decode_credit = 0

            prefill_result = self._schedule_prefill_only()'''

assert s_src.count(old_sched_alt) == 1, "scheduler alternation block not found once"
s_src = s_src.replace(old_sched_alt, new_sched_alt, 1)

# 6b. One-in-flight in _schedule_prefill_only
old_sched_oif = '''        saved_max = self.max_num_running_reqs
        self.running = cast(list[Request], partial_prefills)
        self.max_num_running_reqs = max(
            0, saved_max - len(pure_decodes) - remote_holders
        )
        try:
            result = super().schedule()
        finally:
            self.running.extend(pure_decodes)
            self.max_num_running_reqs = saved_max
            # NIT-7: back by arrival time, not at the head (the pass may have
            # left older plain requests waiting).
            if taken_waiting is not None:
                self._restore_requests_by_arrival(self.waiting, taken_waiting)
            if taken_skipped is not None and skipped_waiting is not None:
                self._restore_requests_by_arrival(skipped_waiting, taken_skipped)
        return result'''

new_sched_oif = '''        saved_max = self.max_num_running_reqs
        self.running = cast(list[Request], partial_prefills)
        interleave_on = os.environ.get("TT_PREFILL_DECODE_INTERLEAVE", "0") == "1"
        if interleave_on:
            if partial_prefills:
                _qwen_allowed = len(partial_prefills)
            else:
                _qwen_allowed = 1
            _qwen_hide = bool(partial_prefills)
            self.max_num_running_reqs = max(
                0, min(saved_max - len(pure_decodes) - remote_holders, _qwen_allowed)
            )
            _qwen_saved_waiting = self.waiting
            _qwen_saved_skipped = getattr(self, "skipped_waiting", None)
            if _qwen_hide:
                self.waiting = create_request_queue(self.policy)
                if _qwen_saved_skipped is not None:
                    self.skipped_waiting = create_request_queue(self.policy)
            logger.info(
                f"[PINDIAG] m2 one-in-flight: partials={len(partial_prefills)} "
                f"decodes={len(pure_decodes)} allowed={_qwen_allowed} "
                f"hidden={_qwen_hide}"
            )
        else:
            self.max_num_running_reqs = max(
                0, saved_max - len(pure_decodes) - remote_holders
            )
        try:
            result = super().schedule()
        finally:
            if interleave_on and _qwen_hide:
                if self.waiting:
                    _qwen_saved_waiting.prepend_requests(self.waiting)
                if _qwen_saved_skipped is not None:
                    if self.skipped_waiting:
                        _qwen_saved_skipped.prepend_requests(self.skipped_waiting)
                    self.skipped_waiting = _qwen_saved_skipped
                self.waiting = _qwen_saved_waiting
            self.running.extend(pure_decodes)
            self.max_num_running_reqs = saved_max
            # NIT-7: back by arrival time, not at the head (the pass may have
            # left older plain requests waiting).
            if taken_waiting is not None:
                self._restore_requests_by_arrival(self.waiting, taken_waiting)
            if taken_skipped is not None and skipped_waiting is not None:
                self._restore_requests_by_arrival(skipped_waiting, taken_skipped)
        return result'''

assert s_src.count(old_sched_oif) == 1, "scheduler one-in-flight block not found once"
s_src = s_src.replace(old_sched_oif, new_sched_oif, 1)

ast.parse(s_src)
with open(sched_path, "w") as f:
    f.write(s_src)
print("scheduler.py patched successfully!")
print("ALL PATCHES APPLIED SUCCESSFULLY!")
