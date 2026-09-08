#!/usr/bin/env python3
"""PHI-CTRL reproducibility orchestrator.

Examples:
  python reproduce.py --smoke
  python reproduce.py --full
  python reproduce.py --train-twin --twin-epochs 25
  python reproduce.py --campaign --seeds 20

The script records provenance, verifies key artifacts, optionally trains the
PHI-Twin, runs the campaign, and writes a SHA-256 release manifest.
"""
from __future__ import annotations
import argparse, hashlib, json, os, platform, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

def run(cmd, label, env=None):
    print(f"\n{'='*72}\n[{label}] {' '.join(map(str, cmd))}\n{'='*72}")
    t=time.time()
    r=subprocess.run([str(x) for x in cmd], cwd=ROOT, env=env)
    if r.returncode != 0:
        raise SystemExit(f"[{label}] FAILED with exit code {r.returncode}")
    print(f"[{label}] PASS ({time.time()-t:.1f}s)")

def git_info():
    def g(*args):
        try: return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception: return "unavailable"
    return {"commit":g("rev-parse","HEAD"), "branch":g("branch","--show-current"), "status":g("status","--short")}

def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024), b""): h.update(b)
    return h.hexdigest()

def write_provenance(out):
    out.mkdir(parents=True, exist_ok=True)
    prov={"timestamp_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
          "python":sys.version, "platform":platform.platform(), "machine":platform.machine(),
          "processor":platform.processor(), "cwd":str(ROOT), "git":git_info(),
          "argv":sys.argv}
    try:
        import numpy, pandas
        prov["packages"]={"numpy":numpy.__version__,"pandas":pandas.__version__}
    except Exception: pass
    try:
        import torch
        prov.setdefault("packages",{})["torch"]=torch.__version__
        prov["cuda_available"]=bool(torch.cuda.is_available())
    except Exception: pass
    (out/"environment.json").write_text(json.dumps(prov,indent=2),encoding="utf-8")

def write_manifest(out):
    files=[]
    skip_dirs={".git","__pycache__","results",".pytest_cache"}
    skip_names={"episodes.csv"}  # large dataset is represented by its dataset manifest/checksum below
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file(): continue
        rel=p.relative_to(ROOT)
        if any(part in skip_dirs for part in rel.parts): continue
        if p.name in skip_names: continue
        try: files.append({"path":str(rel).replace(os.sep,"/"),"sha256":sha256(p),"bytes":p.stat().st_size})
        except OSError: pass
    m={"release_manifest_version":1,"generated_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"git":git_info(),"files":files}
    (out/"release_manifest.json").write_text(json.dumps(m,indent=2),encoding="utf-8")

def verify():
    required=["phi_ctrl_unified_f16.py","plant/jsbsim_plant_f16.py","controller/energy_hold_f16.py",
              "detector/phi_twin_cnn_bilstm.py","models/phi_twin_cnn_bilstm.pt",
              "models/phi_ctrl_residual_f16.zip","data/phi_ctrl_f16_fault/episodes.csv"]
    missing=[x for x in required if not (ROOT/x).exists()]
    if missing: raise SystemExit("Missing required release artifacts:\n"+"\n".join(missing))
    print("[verify] required artifacts: PASS")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--smoke",action="store_true")
    ap.add_argument("--full",action="store_true")
    ap.add_argument("--baseline",action="store_true")
    ap.add_argument("--dataset",action="store_true")
    ap.add_argument("--train-twin",action="store_true")
    ap.add_argument("--campaign",action="store_true")
    ap.add_argument("--analysis",action="store_true")
    ap.add_argument("--seeds",type=int,default=20)
    ap.add_argument("--twin-epochs",type=int,default=25)
    args=ap.parse_args()
    if not any(vars(args)[k] for k in ("smoke","full","baseline","dataset","train_twin","campaign","analysis")):
        ap.print_help(); return
    verify()
    prov=ROOT/"results"/"reproduction"; write_provenance(prov)

    if args.baseline or args.full or args.smoke:
        run([PY,"baseline_jsbsim_recovery/run_baseline_recovery.py","--no-fault"],"baseline-no-fault")
    if args.dataset:
        run([PY,"scripts/generate_fault_dataset_f16.py","--smoke"],"dataset-smoke")
    if args.train_twin or args.full:
        run([PY,"scripts/train_phi_twin_detector.py","--data","data/phi_ctrl_f16_fault","--epochs",str(args.twin_epochs)],"phi-twin-train")
    if args.campaign or args.full:
        if args.smoke:
            run([PY,"eval_campaign_tier12.py","--smoke","--integrity","twin","--out","results/reproduction/campaign_smoke"],"campaign-smoke")
        else:
            run([PY,"eval_campaign_tier12.py","--full-matrix","--seeds",str(args.seeds),"--integrity","twin","--out","results/reproduction/campaign_full"],"campaign-full")
    if args.analysis or args.full:
        inp=prov/"campaign_full" if (prov/"campaign_full").exists() else prov/"campaign_smoke"
        run([PY,"analyze_campaign_stats.py","--in",str(inp)],"campaign-analysis")
    write_manifest(prov)
    print("\nREPRODUCTION COMPLETE")
    print(f"Manifest: {prov/'release_manifest.json'}")

if __name__=="__main__": main()
