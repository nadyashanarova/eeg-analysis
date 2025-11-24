import os
import pandas as pd
import collections
import typing
import attr
import numpy as np
from common import utils
from dataset.dataset import Signal
from etl.extract import Extractor
from etl.transform.transform import Transformer
from epoching.model.markers import MarkersExtractor, Markers
from epoching.cut.epoch_cutter import EpochCutter
from epoching.model.epoch import Epoch
from config import STIMULI_MAP, NAME_MAPPING

SUBJECT_START_TIMES = {}

@attr.s(auto_attribs=True)
class CommonAverageReference(Transformer):
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.sub(df.mean(axis=1), axis=0)

@attr.s(auto_attribs=True)
class EmotivCsvExtractor(Extractor):
    def extract(self, signal: Signal) -> None:
        print(f"Reading signal: {os.path.basename(signal.edf_path)}")
        
        header_row_idx = None
        sep = ','
        
        try:
            with open(signal.edf_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = [f.readline() for _ in range(50)]
                
            for i, line in enumerate(lines):
                if 'Timestamp' in line or 'OriginalTimestamp' in line:
                    header_row_idx = i
                    if line.count(';') > line.count(','): sep = ';'
                    break
            
            if header_row_idx is None:
                print(f"   ❌ Error: Could not find 'Timestamp' header.")
                signal.df = pd.DataFrame()
                return

            print(f"   -> Header found at line {header_row_idx}, separator: '{sep}'")

            df = pd.read_csv(signal.edf_path, skiprows=header_row_idx, sep=sep, 
                             on_bad_lines='skip', engine='python')
            
        except Exception as e:
            print(f"   ❌ Critical CSV error: {e}")
            signal.df = pd.DataFrame()
            return

        df.columns = [c.strip().replace(';', '') for c in df.columns]
        
        ts_col = next((c for c in df.columns if c.lower() == 'timestamp'), None)
        if not ts_col:
            ts_col = next((c for c in df.columns if c.lower() in ['time', 'originaltimestamp']), None)

        if not ts_col:
            print(f"   ❌ Error: No timestamp column found.")
            signal.df = pd.DataFrame()
            return

        try:
            first_ts = float(df[ts_col].iloc[0])
            SUBJECT_START_TIMES[signal.name] = first_ts
        except:
            signal.df = pd.DataFrame()
            return

        eeg_cols = [c for c in df.columns if 'EEG.' in c.upper() 
                    and not any(x in c.upper() for x in ['COUNTER', 'INTERPOLATED', 'RAWCQ', 'MARKER', 'BATTERY'])]
        
        if not eeg_cols:
             known_channels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
             eeg_cols = [c for c in df.columns if c.upper() in known_channels]

        if not eeg_cols:
            print(f"   ❌ Error: No EEG columns found.")
            signal.df = pd.DataFrame()
            return

        clean_df = df[eeg_cols].copy()
        clean_df.columns = [c.upper().replace('EEG.', '').strip() for c in clean_df.columns]
        signal.df = clean_df
        print(f"   -> Successfully loaded {len(clean_df)} rows.")

@attr.s(auto_attribs=True)
class SmartMarkersExtractor(MarkersExtractor):
    def create(self, path: str) -> Markers:
        markers = collections.defaultdict(list)
        if not os.path.exists(path): return Markers(markers)
        
        filename = os.path.basename(path)
        is_pilot = filename.startswith("0_") or "basic_pre_processing" in filename
        
        try:
            sep = ','
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                if ';' in f.readline(): sep = ';'
            
            df = pd.read_csv(path, sep=sep, low_memory=False)
            
            if is_pilot and 'index' in df.columns:
                last_marker = None
                found_count = 0
                for raw_val in df['index'].astype(str):
                    current_match = None
                    for bad_name, good_name in NAME_MAPPING.items():
                        if bad_name in raw_val:
                            current_match = good_name
                            break
                    if not current_match:
                        for stim_name in STIMULI_MAP.keys():
                            if stim_name in raw_val:
                                current_match = stim_name
                                break
                    if current_match:
                        if current_match != last_marker:
                            try:
                                parts = raw_val.split(',')
                                if len(parts) > 1:
                                    markers[STIMULI_MAP[current_match]].append(float(parts[1]))
                                    found_count += 1
                            except: pass
                        last_marker = current_match
                    else:
                        last_marker = None
                return Markers(markers)

            df.columns = [c.lower().strip() for c in df.columns]
            time_col = next((c for c in df.columns if c in ['timestamp', 'latency', 'originaltimestamp', 'time']), None)
            
            if not time_col:
                return Markers(markers)

            candidate_cols = [c for c in df.columns if c in ['marker_value', 'marker_label__desc', 'label', 'type', 'description']]
            
            found_count = 0
            for _, row in df.iterrows():
                if pd.isna(row[time_col]): continue
                m_time = float(row[time_col])
                
                for col in candidate_cols:
                    raw_val = str(row[col])
                    m_val = raw_val.strip().upper().replace(' ', '_')
                    m_val = NAME_MAPPING.get(m_val, m_val)
                    
                    if m_val in STIMULI_MAP:
                        markers[STIMULI_MAP[m_val]].append(m_time)
                        found_count += 1
                        break
            
            if found_count > 0:
                print(f"   -> Parsed markers: {found_count} events.")

        except Exception as e:
            print(f"   ❌ Error reading markers {path}: {e}")
            pass
            
        return Markers(markers)

@attr.s(auto_attribs=True)
class StimuliEpochCutter(EpochCutter):
    sample_freq: int = 128
    baseline_ms: int = 200
    epoch_after_ms: int = 1000
    reject_threshold: float = 150.0 
    
    start_window_sample_count: int = attr.ib(init=False)
    end_window_sample_count: int = attr.ib(init=False)
    stats: typing.Dict = attr.ib(init=False, factory=dict)

    def __attrs_post_init__(self):
        self.start_window_sample_count = utils.ms_to_sample(self.baseline_ms, self.sample_freq)
        self.end_window_sample_count = utils.ms_to_sample(self.epoch_after_ms, self.sample_freq)

    def get_epochs(
        self, signal: Signal, markers: Markers, artifacts_samples: typing.List[int]
    ) -> typing.List[Epoch]:
        
        found_epochs = []
        sub_id = signal.name
        self.stats[sub_id] = {"total_found": 0, "rejected": 0, "kept": 0, "err_sync": 0, "err_bounds": 0}
        
        if signal.df.empty or signal.name not in SUBJECT_START_TIMES: 
            return []
            
        file_start = SUBJECT_START_TIMES[signal.name]
        file_len = len(signal.df)
        
        for marker_id in STIMULI_MAP.values():
            events = markers.get_events(marker_id)
            for abs_time in events:
                rel_time = abs_time - file_start
                
                # === ВОТ ЭТОТ ХАК НУЖЕН ДЛЯ SUBJECT 1 ===
                if rel_time < 0:
                    if abs_time < 100000 and file_start > 100000:
                        rel_time = abs_time
                    elif sub_id == '0' and abs_time < 10000:
                        rel_time = abs_time
                    elif abs(rel_time) < 10.0:
                        rel_time = 0.5
                    else:
                        self.stats[sub_id]["err_sync"] += 1
                        continue
                # ========================================
                
                center = int(rel_time * self.sample_freq)
                start = center - self.start_window_sample_count
                end = center + self.end_window_sample_count
                
                if start < 0 or end >= file_len: 
                    self.stats[sub_id]["err_bounds"] += 1
                    continue
                
                self.stats[sub_id]["total_found"] += 1
                epoch_data = signal.df.iloc[start:end].copy()
                
                if (epoch_data.abs() > self.reject_threshold).any().any():
                    self.stats[sub_id]["rejected"] += 1
                    continue 

                self.stats[sub_id]["kept"] += 1
                baseline = epoch_data.iloc[:self.start_window_sample_count].mean()
                epoch_data = epoch_data - baseline
                
                found_epochs.append(Epoch(
                    samples=epoch_data,
                    noise=marker_id, 
                    marker_sample=center
                ))
                
        if self.stats[sub_id]["kept"] == 0:
             print(f"   ⚠️ {sub_id}: No epochs. SyncErr={self.stats[sub_id]['err_sync']}")

        return found_epochs