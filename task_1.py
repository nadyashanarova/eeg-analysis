import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import collections
import functools


try:
    import cached_property
except ImportError:
    sys.modules['cached_property'] = functools
    class FakeCachedProperty:
        cached_property = functools.cached_property
    sys.modules['cached_property'] = FakeCachedProperty 

from dataset.dataset import Dataset
from etl.transform.preprocessing import BandPassFilter, BandStopFilter
from etl.transform.remove_dc import DCRemover
from epoching.pipeline import EpochingPipeline
import config
import handlers

def main():

    if not os.path.exists(config.INPUT_DIR):
        print(f"❌ Папка {config.INPUT_DIR} не найдена.")
        return
    if not os.path.exists(config.OUTPUT_DIR): 
        os.makedirs(config.OUTPUT_DIR)

    all_files = os.listdir(config.INPUT_DIR)
    
    potential_subjects = set()
    for f in all_files:
        if "RAW" in f.upper() and f.lower().endswith(".csv") and "marker" not in f.lower():
            pid = f.split('_')[0].split(' ')[0]
            potential_subjects.add(pid)
            
    print(f"Scanning {len(potential_subjects)} potential subjects...")
    
    signal_paths = {}
    marker_paths = {}

    for pid in sorted(list(potential_subjects)):
        raw_file = None
        marker_file = None
        
        raw_candidates = [f for f in all_files if f.startswith(pid) and "RAW" in f.upper() and "marker" not in f.lower()]
        if raw_candidates:
            raw_file = os.path.join(config.INPUT_DIR, raw_candidates[0])
            
        if pid == '0':
            marker_candidates = [f for f in all_files if f.startswith("0") and "basic" in f.lower() and "processing" in f.lower()]
        else:
            marker_candidates = [f for f in all_files if f.startswith(pid) and "marker" in f.lower()]
        if marker_candidates:
            marker_file = os.path.join(config.INPUT_DIR, marker_candidates[0])
        if raw_file and marker_file:
            signal_paths[pid] = raw_file
            marker_paths[pid] = marker_file
        else:
            print(f"❌ Skipping {pid}: missing pair.")
            if not raw_file: print(f"   -> Raw file NOT found (looked for startswith '{pid}' + 'RAW')")
            else: print(f"   -> Raw found: {os.path.basename(raw_file)}")
            
            if not marker_file: print(f"   -> Marker file NOT found (looked for startswith '{pid}' + 'Marker'/'basic')")
            else: print(f"   -> Marker found: {os.path.basename(marker_file)}")

    print(f"\nReady to process: {len(signal_paths)} subjects: {list(signal_paths.keys())}")
    
    if len(signal_paths) == 0:
        print("No valid pairs found. Exiting.")
        return


    dataset = Dataset(signal_to_edf_path=signal_paths, signal_to_markers_path=marker_paths)
    
    pipeline = EpochingPipeline(
        extractor=handlers.EmotivCsvExtractor(),
        markers_extractor=handlers.SmartMarkersExtractor(), 
        transformers=[
            DCRemover(sample_freq=128),
            BandPassFilter(sample_freq=128, low_cut=1, high_cut=30, order=4),
            BandStopFilter(sample_freq=128, low_cut=48, high_cut=52, order=4),
            handlers.CommonAverageReference()
        ],
        epoch_cutter=handlers.StimuliEpochCutter(sample_freq=128, reject_threshold=150.0),
        persist_intermediate=False
    )

    pipeline.process(dataset)


    erp_data = collections.defaultdict(list)
    for sig in pipeline.processed_signals:
        for ep in sig.epochs:
            erp_data[ep.noise].append(ep.samples.values)

    print(f"\nGenerating graphs in {config.OUTPUT_DIR}...")
    
    for stim_id, data in erp_data.items():
        if not data: continue
        min_len = min(d.shape[0] for d in data)
        # Если эпох слишком мало, график будет ужасным, пропускаем
        if min_len < 10: 
            print(f"Skipping {config.ID_TO_NAME.get(stim_id)}: too few samples ({min_len})")
            continue
        
        time_axis = np.linspace(-0.2, -0.2 + (min_len/128), min_len)
        stack = np.array([d[:min_len] for d in data]) 
        erp = np.mean(stack, axis=0)
        
        stim_name = config.ID_TO_NAME.get(stim_id, str(stim_id))
        
        plt.figure(figsize=(10, 6))
        plt.plot(time_axis, erp, alpha=0.6, linewidth=1)
        plt.title(f"ERP: {stim_name} (Total Epochs: {len(data)})")
        plt.xlabel("Time (s)")
        plt.ylabel("uV")
        plt.axvline(0, color='k', linestyle='--', alpha=0.8)
        plt.axhline(0, color='k', linewidth=0.5)
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(os.path.join(config.OUTPUT_DIR, f"{stim_name}.png"), dpi=150)
        plt.close()
        print(f"Saved {stim_name}.png")


    print(f"\nSaving Excel Report...")
    excel_path = os.path.join(config.OUTPUT_DIR, "Preprocessing_Results.xlsx")
    
    epoch_stats = collections.defaultdict(lambda: collections.defaultdict(int))
    all_stimuli_ids = sorted(list(erp_data.keys()))
    
    for sig in pipeline.processed_signals:
        sub_id = sig.signal.name
        for ep in sig.epochs:
            stim_name = config.ID_TO_NAME.get(ep.noise, str(ep.noise))
            epoch_stats[sub_id][stim_name] += 1
            
    qc_rows = []

    sorted_subjects_all = sorted(list(signal_paths.keys()), key=lambda x: int(x) if x.isdigit() else 999)
    
    for sub_id in sorted_subjects_all:
        row = {"Subject": sub_id}
        total_clean = 0
        for stim_id in all_stimuli_ids:
            s_name = config.ID_TO_NAME.get(stim_id, str(stim_id))
            count = epoch_stats[sub_id].get(s_name, 0)
            row[s_name] = count
            total_clean += count
        row["TOTAL_CLEAN_EPOCHS"] = total_clean
        qc_rows.append(row)
        
    clean_stats_rows = []
    raw_stats = pipeline.epoch_cutter.stats 
    for sub_id in sorted_subjects_all:
        s = raw_stats.get(sub_id, {"total_found": 0, "rejected": 0, "kept": 0})
        total = s["total_found"]
        rejected = s["rejected"]
        kept = s["kept"]
        rate = round((rejected / total * 100), 1) if total > 0 else 0
        clean_stats_rows.append({
            "Subject": sub_id, "Total_Found": total, "Rejected": rejected, "Kept": kept, "Rate": rate
        })

    try:
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            pd.DataFrame(clean_stats_rows).to_excel(writer, sheet_name="Cleaning_Stats", index=False)
            pd.DataFrame(qc_rows).to_excel(writer, sheet_name="Per_Stimulus_Count", index=False)
            
            # Сохраняем данные графиков (усредненные)
            for stim_id, data in erp_data.items():
                if not data: continue
                min_len = min(d.shape[0] for d in data)
                if min_len < 10: continue
                stack = np.array([d[:min_len] for d in data])
                erp = np.mean(stack, axis=0) 
                time_axis = np.linspace(-0.2, -0.2 + (min_len/128), min_len)
                
                cols = [f"Ch_{i}" for i in range(erp.shape[1])]
                if pipeline.processed_signals:
                    try: cols = pipeline.processed_signals[0].signal.df.columns
                    except: pass
                df_export = pd.DataFrame(erp, columns=cols)
                df_export.insert(0, "Time_Sec", time_axis)
                s_name = config.ID_TO_NAME.get(stim_id, str(stim_id))[:30]
                df_export.to_excel(writer, sheet_name=s_name, index=False)
        print(f"Report saved: {excel_path}")
    except Exception as e:
        print(f"Excel error: {e}")

    print("\nDone.")

if __name__ == '__main__':
    main()