#!/usr/bin/env python3
import json, os, re, subprocess, sys, time, signal, statistics, math

AGORA_DIR = "/home/DownlinkAgora/Agora"
DIR = "/root/.claude/jobs/ff28803f/tmp/txe"
N_REPS = 5

BER_PAT = re.compile(
    r"Average: Downlink bit errors \(BER\) (\d+)/(\d+) \(([\d.]+)\), "
    r"block errors \(BLER\) (\d+)/(\d+) \(([\d.]+)\)")

SIZES = {
    "8x8_sc1216": dict(zf_base=f"{DIR}/cfg_zf8x8sc1216_zeronoise_base.json",
                        vpp_base=f"{DIR}/cfg_vpp8x8sc1216_zeronoise_base.json",
                        target=15948.0, bsant=8, ofdm=1216,
                        db_list=[37, 40, 43, 46, 49, 52, 55, 58, 61, 64]),
    "16x16_sc1216": dict(zf_base=f"{DIR}/cfg_zf16x16sc1216_zeronoise_base.json",
                          vpp_base=f"{DIR}/cfg_vpp16x16sc1216_zeronoise_base.json",
                          target=27896.0, bsant=16, ofdm=1216,
                          db_list=[36, 39, 42, 45, 48, 51, 54, 57, 60, 63]),
    "32x32_sc96": dict(zf_base=f"{DIR}/cfg_zf32sc96_zeronoise_base.json",
                        vpp_base=f"{DIR}/cfg_vpp32sc96_zeronoise_base.json",
                        target=1675.9, bsant=32, ofdm=96,
                        db_list=[25, 28, 31, 34, 37, 40, 43, 46, 49, 52]),
}

def noise_for_db(E, db):
    return math.sqrt(E / (10 ** (db / 10.0)))

def run_trio(cfg_path, out_prefix, timeout=90):
    dg = subprocess.run(["./build/data_generator", "--conf_file", cfg_path],
                         cwd=AGORA_DIR, stdout=open(f"{out_prefix}_datagen.log", "w"),
                         stderr=subprocess.STDOUT)
    if dg.returncode != 0:
        return False
    user_p = subprocess.Popen(["./build/user", "--conf_file", cfg_path], cwd=AGORA_DIR,
                               stdout=open(f"{out_prefix}_user.log", "w"), stderr=subprocess.STDOUT)
    time.sleep(1)
    chsim_p = subprocess.Popen(["./build/chsim", "--bs_threads", "1", "--ue_threads", "1",
                                "--worker_threads", "4", "--core_offset", "1",
                                "--conf_file", cfg_path], cwd=AGORA_DIR,
                               stdout=open(f"{out_prefix}_chsim.log", "w"), stderr=subprocess.STDOUT)
    time.sleep(1)
    try:
        subprocess.run(["./build/agora", "--conf_file", cfg_path], cwd=AGORA_DIR,
                        stdout=open(f"{out_prefix}_agora.log", "w"), stderr=subprocess.STDOUT,
                        timeout=timeout)
    except subprocess.TimeoutExpired:
        pass
    for _ in range(20):
        if user_p.poll() is not None:
            break
        time.sleep(1)
    chsim_p.send_signal(signal.SIGINT)
    time.sleep(1)
    chsim_p.kill()
    if user_p.poll() is None:
        user_p.send_signal(signal.SIGINT)
        time.sleep(1)
        user_p.kill()
    return True

def get_ber_bler(path):
    if not os.path.exists(path):
        return None
    txt = open(path).read()
    m = BER_PAT.findall(txt)
    if not m:
        return None
    last = m[-1]
    return float(last[2]), float(last[5])

def run_one(base_path, tag, noise_level, target, timeout=150):
    cfg = json.load(open(base_path))
    cfg["noise_level"] = noise_level
    cfg["tx_energy_target"] = target
    cfg["max_frame"] = 20
    cfg_path = f"{DIR}/cfg_{tag}.json"
    json.dump(cfg, open(cfg_path, "w"))
    out_prefix = f"{DIR}/run_{tag}"
    run_trio(cfg_path, out_prefix, timeout=timeout)
    res = get_ber_bler(f"{out_prefix}_user.log")
    for suf in ["_datagen.log", "_chsim.log"]:
        fp = out_prefix + suf
        if os.path.exists(fp):
            os.remove(fp)
    os.remove(cfg_path)
    return res

def mean_or_none(vals):
    vals = [v for v in vals if v is not None]
    return statistics.mean(vals) if vals else None

def main():
    results_path = f"{DIR}/snr_experiment_cross_results.json"
    all_results = {}
    if os.path.exists(results_path):
        all_results = json.load(open(results_path))

    for size_name, s in SIZES.items():
        all_results.setdefault(size_name, {})
        E = s["target"] / (s["bsant"] * s["ofdm"])
        for db in s["db_list"]:
            nl = noise_for_db(E, db)
            db_key = str(db)
            rec = all_results[size_name].setdefault(db_key, {
                "noise_level": nl, "zf_ber": [], "zf_bler": [],
                "vpp_ber": [], "vpp_bler": []})
            start_rep = len(rec["zf_ber"])
            for rep in range(start_rep, N_REPS):
                t0 = time.time()
                zf_res = run_one(s["zf_base"], f"{size_name}_zf_{db}_{rep}", nl, s["target"])
                vpp_res = run_one(s["vpp_base"], f"{size_name}_vpp_{db}_{rep}", nl, s["target"])
                rec["zf_ber"].append(zf_res[0] if zf_res else None)
                rec["zf_bler"].append(zf_res[1] if zf_res else None)
                rec["vpp_ber"].append(vpp_res[0] if vpp_res else None)
                rec["vpp_bler"].append(vpp_res[1] if vpp_res else None)
                json.dump(all_results, open(results_path, "w"), indent=2)
                dt = time.time() - t0
                print(f"{size_name} db={db} nl={nl:.6f} rep={rep+1}/{N_REPS} ({dt:.1f}s) "
                      f"zf={zf_res} vpp={vpp_res}", flush=True)
            zf_m = mean_or_none(rec["zf_ber"])
            vpp_m = mean_or_none(rec["vpp_ber"])
            print(f"=== {size_name} db={db}: zf_ber_mean={zf_m} vpp_ber_mean={vpp_m} ===", flush=True)
        print(f"--- {size_name} DONE ---", flush=True)
    print("SNR EXPERIMENT CROSS ALL DONE", flush=True)

if __name__ == "__main__":
    main()
