import os
import pandas as pd
import numpy as np
import config_emotions as cfg

class SubjectFileResolver:
    def __init__(self, input_dir):
        self.input_dir = input_dir

    def resolve_pairs(self):
        all_files = os.listdir(self.input_dir)
        subjects = {}
        
        potential_ids = set()
        for f in all_files:
            if "RAW" in f and f.endswith(".csv"):
                pid = f.split('_')[0]
                potential_ids.add(pid)

        valid_pairs = {}
        missing_pairs = []

        for pid in potential_ids:
            raw_path = self._find_raw(pid, all_files)
            marker_path = self._find_marker(pid, all_files)

            if raw_path and marker_path:
                valid_pairs[pid] = {'raw': raw_path, 'marker': marker_path}
            else:
                missing_pairs.append(pid)
        
        return valid_pairs, missing_pairs

    def _find_raw(self, pid, files):
        candidates = [f for f in files if f.startswith(pid) and "RAW" in f and "marker" not in f.lower()]
        if candidates:
            return os.path.join(self.input_dir, candidates[0])
        return None

    def _find_marker(self, pid, files):
        if pid == '0':
            candidates = [f for f in files if f.startswith("0_") and "basic_pre_processing" in f]
        else:
            candidates = [f for f in files if f.startswith(pid) and "intervalMarker" in f]
        
        if candidates:
            return os.path.join(self.input_dir, candidates[0])
        return None

class EmotionExtractor:
    def process_subject(self, sub_id, files):
        try:
            raw_df = pd.read_csv(files['raw'], skiprows=1, engine='python')
            raw_df.columns = [c.strip() for c in raw_df.columns]
        except Exception as e:
            print(f"Error reading RAW for {sub_id}: {e}")
            return None

        ts_col = self._find_col(raw_df, ['timestamp', 'time', 'originaltimestamp'])
        if not ts_col: return None

        metric_cols = []
        rename_map = {}
        for col in raw_df.columns:
            for target in cfg.EMOTION_COLS:
                if target in col:
                    rename_map[col] = target
                    metric_cols.append(target)
        
        if not metric_cols:
            return None

        raw_df.rename(columns=rename_map, inplace=True)
        work_df = raw_df[[ts_col] + metric_cols].fillna(method='ffill').fillna(method='bfill')

        markers = self._extract_markers(files['marker'], sub_id)
        if not markers:
            print(f"Skipping {sub_id}: No valid markers found.")
            return None

        return self._cut_epochs(work_df, markers, sub_id, ts_col)

    def _extract_markers(self, path, sub_id):
        markers = []
        try:
            sep = ','
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                if ';' in f.readline(): sep = ';'
            
            m_df = pd.read_csv(path, sep=sep, engine='python')
            m_df.columns = [c.lower().strip() for c in m_df.columns]
            
            t_col = self._find_col(m_df, ['timestamp', 'latency', 'time'])
            label_cols = [c for c in m_df.columns if any(x in c for x in ['marker', 'label', 'type', 'desc'])]
            
            use_order_strategy = (sub_id == '0')
            pilot_timestamps = []

            for _, row in m_df.iterrows():
                if pd.isna(row[t_col]): continue
                t_val = float(row[t_col])
                
                matched_label = None
                
                for l_col in label_cols:
                    val = str(row[l_col]).strip().upper().replace(' ', '_')
                    if val in cfg.ORDERED_STIMULI:
                        matched_label = val
                        break
                    for target in cfg.ORDERED_STIMULI:
                        if target in val:
                            matched_label = target
                            break
                
                if matched_label:
                    markers.append((t_val, matched_label))
                
                elif use_order_strategy:
                    is_event = False
                    for l_col in label_cols:
                        val = str(row[l_col])
                        if val and val.lower() not in ['nan', '0', '0.0', 'none', '']:
                            is_event = True
                            break
                    if is_event:
                        pilot_timestamps.append(t_val)

            if use_order_strategy and not markers:
                pilot_timestamps.sort()
                
                count_found = len(pilot_timestamps)
                count_needed = len(cfg.ORDERED_STIMULI)
                
                if count_found != count_needed:
                    print(f"  ❌ [Subject 0] Event mismatch! Found {count_found}, needed {count_needed}. SKIPPING this subject to avoid errors.")
                    return [] 

                limit = min(count_found, count_needed)
                
                for i in range(limit):
                    stim_name = cfg.ORDERED_STIMULI[i]
                    time_val = pilot_timestamps[i]
                    markers.append((time_val, stim_name))

        except Exception as e:
            print(f"Marker parsing error for {sub_id}: {e}")
            
        return markers

    def _cut_epochs(self, df, markers, sub_id, ts_col):
        epochs = []
        if df.empty: return None
        
        file_start = df[ts_col].iloc[0]
        
        gender = 'Unknown'
        if sub_id in cfg.GENDER_MAP['Male']: gender = 'Male'
        elif sub_id in cfg.GENDER_MAP['Female']: gender = 'Female'

        for m_time, m_label in markers:
            start_t = m_time
            if m_time < 100000 and file_start > 100000:
                start_t = file_start + m_time
            
            mask = (df[ts_col] >= start_t - cfg.BASELINE_SEC) & \
                   (df[ts_col] <= start_t + cfg.EPOCH_SEC)
            
            chunk = df.loc[mask].copy()
            if chunk.empty: continue
            
            chunk['rel_time'] = chunk[ts_col] - start_t
            
            base_part = chunk[chunk['rel_time'] < 0]
            if not base_part.empty:
                means = base_part[cfg.EMOTION_COLS].mean()
                for col in cfg.EMOTION_COLS:
                    chunk[col] = chunk[col] - means[col]
            
            chunk['Stimulus'] = m_label
            chunk['Subject'] = sub_id
            chunk['Gender'] = gender
            chunk['time_bin'] = chunk['rel_time'].apply(lambda x: np.floor(x * 2) / 2)
            
            grouped = chunk.groupby(['time_bin', 'Stimulus', 'Subject', 'Gender'], as_index=False)[cfg.EMOTION_COLS].mean()
            epochs.append(grouped)
            
        if epochs:
            return pd.concat(epochs)
        return None

    def _find_col(self, df, candidates):
        for c in df.columns:
            if c.lower() in candidates: return c
        return None