from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
TARGETS_DIR = DATA_DIR / "targets"
ALPHA_DIR = DATA_DIR / "alpha"
PTQ_DIR = PROJECT_ROOT / "ptq_models"
EMPIRICAL_DIR = DATA_DIR / "empirical"

ALL_IDS = ["SpectraSuite/FloatLM_99M", "SpectraSuite/FloatLM_190M", "SpectraSuite/FloatLM_390M", "SpectraSuite/FloatLM_560M", "SpectraSuite/FloatLM_830M","SpectraSuite/FloatLM_1.1B", "SpectraSuite/FloatLM_1.5B","SpectraSuite/FloatLM_3.9B", "SpectraSuite/TriLM_99M_Unpacked", "SpectraSuite/TriLM_190M_Unpacked", "SpectraSuite/TriLM_390M_Unpacked", "SpectraSuite/TriLM_560M_Unpacked", "SpectraSuite/TriLM_830M_Unpacked","SpectraSuite/TriLM_1.1B_Unpacked", "SpectraSuite/TriLM_1.5B_Unpacked", "SpectraSuite/TriLM_3.9B_Unpacked"]
TARGET_IDS = ["SpectraSuite/FloatLM_3.9B", "SpectraSuite/TriLM_3.9B_Unpacked"]
TO_BE_TERNARIZED_IDS = ["SpectraSuite/FloatLM_99M", "SpectraSuite/FloatLM_190M", "SpectraSuite/FloatLM_390M", "SpectraSuite/FloatLM_560M", "SpectraSuite/FloatLM_830M","SpectraSuite/FloatLM_1.1B", "SpectraSuite/FloatLM_1.5B"]
PTQ_IDS = [str(PTQ_DIR / f"PTQ_{s}") for s in ["99M", "190M", "390M", "560M", "830M", "1.1B", "1.5B"]]
DRAFT_IDS = ALL_IDS + PTQ_IDS
TOKENIZER_ID = "SpectraSuite/FloatLM_3.9B"

CUTOFF_UNIX = int(datetime(2025, 1, 1).timestamp())

DATA_SEED = 31
GEN_SEED = 31
EMPIRICAL_SEED = 31
PLOTTING_SEED = 31

MIN_TOKENS_IN_EXAMPLE = 1024
BATCH_SIZE = 128
PREFIX_LEN = 256
GENERATION_LEN = 256

def model_name(model_id):
    return model_id.split("/")[-1]


SIZE_M = {
    "FloatLM_99M": 99,   "TriLM_99M_Unpacked": 99,   "PTQ_99M": 99,
    "FloatLM_190M": 190, "TriLM_190M_Unpacked": 190, "PTQ_190M": 190,
    "FloatLM_390M": 390, "TriLM_390M_Unpacked": 390, "PTQ_390M": 390,
    "FloatLM_560M": 560, "TriLM_560M_Unpacked": 560, "PTQ_560M": 560,
    "FloatLM_830M": 830, "TriLM_830M_Unpacked": 830, "PTQ_830M": 830,
    "FloatLM_1.1B": 1100, "TriLM_1.1B_Unpacked": 1100, "PTQ_1.1B": 1100,
    "FloatLM_1.5B": 1500, "TriLM_1.5B_Unpacked": 1500, "PTQ_1.5B": 1500,
    "FloatLM_3.9B": 3900, "TriLM_3.9B_Unpacked": 3900,
}

K = 5