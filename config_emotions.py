INPUT_DIR = "raw_data"
OUTPUT_DIR = "EMOTIONS_results"

BASELINE_SEC = 2.0
EPOCH_SEC = 10.0


ORDERED_STIMULI = [
    "VK_JAPAN_INFO", "VK_JAPAN_COM", "VK_JAPAN_THR",
    "TG_JAPAN_INFO", "TG_JAPAN_COM", "TG_JAPAN_THR",
    "TG_MUSK_INFO",  "TG_MUSK_COM",  "TG_MUSK_THR",
    "VK_MUSK_INFO",  "VK_MUSK_COM",  "VK_MUSK_THR",
    "VK_BORISOV_INFO", "VK_BORISOV_COM", "VK_BORISOV_THR",
    "TG_BORISOV_INFO", "TG_BORISOV_COM", "TG_BORISOV_THR",
    "TG_EGE_INFO",   "TG_EGE_COM",   "TG_EGE_THR",
    "VK_EGE_INFO",   "VK_EGE_COM",   "VK_EGE_THR"
]


NAME_MAPPING = {
    "TG_MASK_INFO": "TG_MUSK_INFO",
    "VK_MASK_INFO": "VK_MUSK_INFO",
    
    "TG_EGE_THR_1": "TG_EGE_THR",
    "TG_EGE_THR_2": "TG_EGE_THR",
    
    "phase_TG_MUSK": "TG_MUSK_INFO",
    "phase_VK_MUSK": "VK_MUSK_INFO"
}

EMOTION_COLS = [
    'PM.Stress.Scaled', 
    'PM.Engagement.Scaled', 
    'PM.Interest.Scaled', 
    'PM.Excitement.Scaled', 
    'PM.Focus.Scaled', 
    'PM.Relaxation.Scaled'
]

GENDER_MAP = {
    'Female': ['1', '7', '9', '11', '12', '13', '15', '16'],
    'Male':   ['0', '2', '3', '4', '5', '6', '8', '10', '14'] 
}