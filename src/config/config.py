import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR      = os.path.join(BASE_DIR, "data", "raw", "chest_xray")
TRAIN_DIR     = os.path.join(DATA_DIR, "train")
VAL_DIR       = os.path.join(DATA_DIR, "val")
TEST_DIR      = os.path.join(DATA_DIR, "test")
MODEL_DIR     = os.path.join(BASE_DIR, "models")
RESULTS_DIR   = os.path.join(BASE_DIR, "results", "plots")

os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Hyperparameters ────────────────────────────────────────────────────────────
IMAGE_SIZE  = (224, 224)
BATCH_SIZE  = 32
EPOCHS      = 50
LR          = 1e-3
CLASSES     = ["NORMAL", "PNEUMONIA"]
MODEL_PATH  = os.path.join(MODEL_DIR, "pneumonia_cnn.keras")

# ── Transfer Learning ──────────────────────────────────────────────────────────
TL_EPOCHS_FROZEN    = 10    # Phase 1: train head only (base frozen)
TL_EPOCHS_FINETUNE  = 30    # Phase 2: unfreeze top layers and fine-tune
TL_LR_FROZEN        = 1e-3  # Higher LR for frozen phase
TL_LR_FINETUNE      = 1e-5  # Low LR to avoid destroying pretrained weights
TL_UNFREEZE_FROM    = 100   # Unfreeze layers from this index onward
TL_MODEL_PATH       = os.path.join(MODEL_DIR, "pneumonia_mobilenetv2.keras")
