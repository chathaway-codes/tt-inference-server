#!/bin/bash
set -euxo pipefail
"$VENV/bin/python" -c 'import ttnn, vllm, vllm_tt_plugin; import torch; assert torch.__version__.endswith('"'"'+cpu'"'"'), torch.__version__; import vllm; assert '"'"'/tt-metal/'"'"' not in vllm.__file__, vllm.__file__; import torch; v = torch.__version__.split('"'"'+'"'"')[0]; assert v == '"'"'2.11.0'"'"', f'"'"'torch {v} was resolved by vLLM but tt-metal pins 2.11.0; ttnn extension modules were built against 2.11.0 — pin it via runtime.lock'"'"''
"$VENV/bin/python" -c 'import os, sys, json, importlib; md = os.environ['"'"'EXTRA_MODELS_DIR'"'"']; found = 0
for e in sorted(os.listdir(md)):
    d = os.path.join(md, e)
    mp = os.path.join(d, '"'"'vllm_metadata.json'"'"')
    if not os.path.exists(mp):
        continue
    found += 1
    meta = json.load(open(mp))
    spec = meta['"'"'main_class'"'"']
    if d not in sys.path:
        sys.path.append(d)
    mod, sep, cls = spec.rpartition('"'"':'"'"')
    if not sep:
        mod, _, cls = spec.rpartition('"'"'.'"'"')
    obj = getattr(importlib.import_module(mod), cls)
    assert obj is not None, spec
    print('"'"'resolved'"'"', meta.get('"'"'arch'"'"'), '"'"'->'"'"', spec)
assert found, f'"'"'EXTRA_MODELS_DIR {md} registers no models'"'"'
'
"$VENV/bin/python" -c 'from models.demos.blackhole.qwen36.tt.qwen36_vllm import Qwen36ForCausalLM; assert Qwen36ForCausalLM'
"$VENV/bin/python" -c 'from models.demos.blackhole.qwen36.tt.kv_transfer import Qwen36KVTransfer; assert Qwen36KVTransfer'
"$VENV/bin/python" -c 'import vllm_tt_plugin.kv_transfer.tt_connector, vllm_tt_plugin.kv_transfer.pd_proxy, vllm_tt_plugin.kv_transfer.launch.pd_fabric_rank, vllm_tt_plugin.kv_transfer.launch.pd_container, vllm_tt_plugin.kv_transfer.launch.dispatch'
"$VENV/bin/python" -c 'import importlib.metadata as m; eps = {e.name for e in m.entry_points(group='"'"'vllm.general_plugins'"'"')}; assert '"'"'tt_serve_launcher'"'"' in eps, eps'
"$VENV/bin/python" -c 'import importlib.resources as r; assert r.files('"'"'vllm_tt_plugin.kv_transfer.launch'"'"').joinpath('"'"'data/pd_two_p150_mesh_graph_descriptor.textproto'"'"').is_file()'
"$VENV/bin/python" -c 'import importlib; importlib.import_module('"'"'ttnn.distributed.ttrun'"'"')'
"$VENV/bin/python" -c 'from pathlib import Path; assert Path('"'"'/opt/openmpi-v5.0.7-ulfm/bin/mpirun'"'"').is_file()'
"$VENV/bin/python" -c 'from pathlib import Path; p = Path('"'"'/opt/tt-metal/models/demos/blackhole/qwen36/mesh/p150_x2_2link_mesh_graph_descriptor.textproto'"'"'); assert p.is_file(); assert '"'"'count: 2'"'"' in p.read_text(), p'
"$VENV/bin/python" -c 'import os, subprocess, sys; os.environ.update(TT_PD_CHIPS='"'"'0,1'"'"', TT_METAL_CACHE='"'"'/tmp/c'"'"', TT_CACHE_PATH='"'"'/tmp/t'"'"'); subprocess.run([sys.executable, '"'"'-m'"'"', '"'"'vllm_tt_plugin.kv_transfer.launch.pd_container'"'"', '"'"'Qwen/Qwen3.8-27B'"'"', '"'"'--max-model-len'"'"', '"'"'65536'"'"', '"'"'--max-num-seqs'"'"', '"'"'8'"'"', '"'"'--block-size'"'"', '"'"'64'"'"', '"'"'--port'"'"', '"'"'20000'"'"', '"'"'--dry-run'"'"'], check=True)'
"$VENV/bin/python" -c 'import os; os.environ['"'"'QWEN36_DRAFTER'"'"']='"'"'dflash2'"'"'; from models.demos.blackhole.qwen36.tt.qwen36_vllm_dflash import Qwen36DFlashForCausalLM as C; assert C.model_capabilities['"'"'tt_adaptive_block_batched'"'"']'
"$VENV/bin/python" -c 'import json; m=json.load(open('"'"'/opt/tt-metal/models/demos/blackhole/qwen36/vllm_bundle/qwen36/vllm_metadata.json'"'"')); assert m['"'"'main_class'"'"'].endswith('"'"':Qwen36DFlashForCausalLM'"'"'), m'
"$VENV/bin/python" -c 'import vllm.model_executor.models.qwen3_5; import transformers.models.qwen3_5'
"$VENV/bin/python" -c 'from pathlib import Path; assert Path('"'"'/opt/tt-metal/models/model_trace_region_sizes.yaml'"'"').is_file()'
