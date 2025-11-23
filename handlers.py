import os
import pandas as pd
import collections
import typing
import attr
from common import utils
from dataset.dataset import Signal
from etl.extract import Extractor
from etl.transform.transform import Transformer
from epoching.model.markers import MarkersExtractor, Markers
from epoching.cut.epoch_cutter import EpochCutter
from epoching.model.epoch import Epoch
from config import STIMULI_MAP

SUBJECT_START_TIMES = {}

@attr.s(auto_attribs=True)
class CommonAverageReference(Transformer):
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.sub(df.mean(axis=1), axis=0)

@attr.s(auto_attribs=True)
class EmotivCsvExtractor(Extractor):
    def extract(self, signal: Signal) -> None:
        print(f"Reading signal: {os.path.basename(signal.edf_path)}")
        sep = ','
        try:
            with open(signal.edf_path, 'r', encoding='utf-8', errors='ignore') as f:
                line1 = f.readline()
                line2 = f.readline()
                if ';' in line2 and line2.count(';') > line2.count(','): sep = ';'
        except: pass

        try:
            # УБРАЛИ engine='python', чтобы читалось быстро
            df = pd.read_csv(signal.edf_path, skiprows=1, sep=sep)
            print(f"   -> Loaded {len(df)} rows.") 
        except Exception as e:
            print(f"   ❌ Error reading CSV: {e}")
            signal.df = pd.DataFrame()
            return

        df.columns = [c.strip() for c in df.columns]
        
        ts_col = next((c for c in df.columns if c.lower() == 'timestamp'), None)
        if not ts_col:
            ts_col = next((c for c in df.columns if c.lower() in ['time', 'originaltimestamp']), None)

        if not ts_col:
            print(f"   ❌ Error: No timestamp column in {signal.name}")
            signal.df = pd.DataFrame()
            return

        SUBJECT_START_TIMES[signal.name] = df[ts_col].iloc[0]

        eeg_cols = [c for c in df.columns if 'EEG.' in c.upper() 
                    and not any(x in c.upper() for x in ['COUNTER', 'INTERPOLATED', 'RAWCQ', 'MARKER', 'BATTERY'])]
        
        if not eeg_cols:
             known_channels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
             eeg_cols = [c for c in df.columns if c.upper() in known_channels]

        clean_df = df[eeg_cols].copy()
        clean_df.columns = [c.upper().replace('EEG.', '').strip() for c in clean_df.columns]
        signal.df = clean_df

@attr.s(auto_attribs=True)
class SmartMarkersExtractor(MarkersExtractor):
    def create(self, path: str) -> Markers:
        markers = collections.defaultdict(list)
        if not os.path.exists(path): return Markers(markers)
        
        filename = os.path.basename(path)
        
        try:
            sep = ','
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                line = f.readline()
                if ';' in line: sep = ';'
            
            df = pd.read_csv(path, sep=sep, low_memory=False)
            df.columns = [c.lower().strip() for c in df.columns]

            time_col = None
            for c in ['timestamp', 'latency', 'originaltimestamp', 'time']:
                if c in df.columns:
                    time_col = c
                    break
            
            if not time_col:
                print(f"   ⚠️ WARNING: No time column in markers {filename}!")
                return Markers(markers)

            candidate_cols = [c for c in df.columns if c in ['marker_value', 'marker_label__desc', 'label', 'type', 'description']]
            
            found_count = 0
            for _, row in df.iterrows():
                m_time = row[time_col]
                if pd.isna(m_time): continue
                
                for col in candidate_cols:
                    raw_val = str(row[col])
                    m_val = raw_val.strip().upper().replace(' ', '_')
                    
                    if m_val in STIMULI_MAP:
                        markers[STIMULI_MAP[m_val]].append(float(m_time))
                        found_count += 1
                        break
            
            if found_count == 0:
                print(f"   ⚠️ WARNING: Markers file read, but NO stimuli found in {filename}!")
            else:
                print(f"   -> Found {found_count} valid markers.")

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
        self.stats[sub_id] = {"total_found": 0, "rejected": 0, "kept": 0}
        
        if signal.df.empty or signal.name not in SUBJECT_START_TIMES: 
            return []
            
        file_start = SUBJECT_START_TIMES[signal.name]
        
        for marker_id in STIMULI_MAP.values():
            events = markers.get_events(marker_id)
            for abs_time in events:
                rel_time = abs_time - file_start
                if rel_time < 0: continue
                
                center = int(rel_time * self.sample_freq)
                start = center - self.start_window_sample_count
                end = center + self.end_window_sample_count
                
                if start < 0 or end >= len(signal.df): continue
                
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
        return found_epochs