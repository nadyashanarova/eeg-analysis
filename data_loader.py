import os
import pandas as pd
import numpy as np
import config_emotions as cfg

class SubjectFileResolver:
    def __init__(self, input_dir):
        self.input_dir = input_dir

    def resolve_pairs(self):
        all_files = os.listdir(self.input_dir)
        potential_ids = set()
        for f in all_files:
            if "RAW" in f.upper() and f.lower().endswith(".csv"):
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
        candidates = [f for f in files if f.startswith(pid) and "RAW" in f.upper() and "marker" not in f.lower()]
        if candidates: return os.path.join(self.input_dir, candidates[0])
        return None

    def _find_marker(self, pid, files):
        if pid == '0':
            candidates = [f for f in files if f.startswith("0") and "basic" in f.lower() and "processing" in f.lower()]
        else:
            candidates = [f for f in files if f.startswith(pid) and "marker" in f.lower()]
        
        if candidates: return os.path.join(self.input_dir, candidates[0])
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
        
        if not metric_cols: return None

        raw_df.rename(columns=rename_map, inplace=True)
        work_df = raw_df[[ts_col] + metric_cols].fillna(method='ffill').fillna(method='bfill')

        markers = self._extract_markers(files['marker'], sub_id)
        if not markers:
            print(f"Subject {sub_id}: No markers found.")
            return None

        return self._cut_epochs(work_df, markers, sub_id, ts_col)

    def _extract_markers(self, path, sub_id):
        markers = []
        try:
            sep = ','
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                if ';' in f.readline(): sep = ';'
            
            m_df = pd.read_csv(path, sep=sep, low_memory=False)
            

            if sub_id == '0' and 'index' in m_df.columns:
                last_marker = None
                for raw_val in m_df['index'].astype(str):
                    current_match = None
                    
                    for bad_name, good_name in cfg.NAME_MAPPING.items():
                        if bad_name in raw_val:
                            current_match = good_name
                            break

                    if not current_match:
                        for target in cfg.ORDERED_STIMULI:
                            if target in raw_val:
                                current_match = target
                                break
                    
                    if current_match:
                        if current_match != last_marker:
                            try:
                                parts = raw_val.split(',')
                                if len(parts) > 1:
                                    markers.append((float(parts[1]), current_match))
                            except: pass
                        last_marker = current_match
                    else:
                        last_marker = None
                
                return markers

            m_df.columns = [c.lower().strip() for c in m_df.columns]
            t_col = self._find_col(m_df, ['timestamp', 'latency', 'time'])
            label_cols = [c for c in m_df.columns if any(x in c for x in ['marker', 'label', 'type', 'desc'])]
            
            for _, row in m_df.iterrows():
                if pd.isna(row[t_col]): continue
                t_val = float(row[t_col])
                
                for l_col in label_cols:
                    val = str(row[l_col]).strip().upper().replace(' ', '_')
                    
                    val = cfg.NAME_MAPPING.get(val, val)
                    
                    if val in cfg.ORDERED_STIMULI:
                        markers.append((t_val, val))
                        break

        except Exception as e:
            print(f"Marker parsing error for {sub_id}: {e}")
            
        return markers

    def _cut_epochs(self, df, markers, sub_id, ts_col):
        epochs = []
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