SILENCE_THRESHOLD = 1e-4
SILENCE_THRESHOLD_DB = -60

# Hierarchical mapping of drum classes for evaluation
class_mapping_5_classes = {
    "KD": "KD",
    "SD": "SD",
    "HH_CHH": "HH",
    "HH_OHH": "HH",
    "TT_HMT": "TT",
    "TT_LMT": "TT",
    "TT_HFT": "TT",
    "CY_CR": "CY",
    "CY_RD": "CY",
}

class_mapping_9_classes = {
    "KD": "kick",
    "SD": "snare",
    "HH_CHH": "hihat_closed",
    "HH_OHH": "hihat_open",
    "TT_HMT": "hi_tom",
    "TT_LMT": "mid_tom",
    "TT_HFT": "low_tom",
    "CY_CR": "crash_left",
    "CY_RD": "ride",
}
class_mapping_3_classes = {
    "KD": "KD",
    "SD": "SD",
    "HH_CHH": "HH",
    "HH_OHH": "HH",
    "CY_CR": "CY",
    "CY_RD": "CY",
}


eval_class_mapping = {
    "3_class": class_mapping_3_classes,
    "5_class": class_mapping_5_classes,
    "9_class": class_mapping_9_classes,
}

minimal_classes = ["HH_CHH", "KD", "SD"]
basic_classes = ["CY_CR", "HH_CHH", "KD", "SD", "TT_HFT"]
full_classes = ["CY_CR", "CY_RD", "HH_CHH", "HH_OHH", "KD", "SD", "TT_HFT", "TT_HMT", "TT_LMT"]

model_version_to_train_class_list = {
    "basic": basic_classes,
    "minimal": minimal_classes,
    "full": full_classes,
}

## This is the correct mapping for annotation -> multitrack files
stem_gmd_single_hits_map = {
    "KD": "kick",
    "SD": "snare",
    "HH_OHH": "hihat_open",
    "HH_CHH": "hihat_closed",
    # "TT_HFT": "mid_tom",
    # "TT_LMT": "low_tom",
    "TT_LMT": "mid_tom",
    "TT_HFT": "low_tom",
    "TT_HMT": "hi_tom",
    "CY_RD": "ride",
    "CY_CR": "crash_left",
}

stem_gmd_drum_class_to_symbol = {
    "kick": "KD",  # Kick Drum
    "snare": "SD",  # Snare Drum
    "hihat_closed": "HH_CHH",  # Closed-Hi-Hat
    "hihat_open": "HH_OHH",  # Open-Hi-Hat
    # "mid_tom": "TT_HFT",  # High-Mid Tom  new LFT
    # "low_tom": "TT_LMT",  # Low-Tom new HFT
    # "hi_tom": "TT_HMT",  # High-Tom new HMT
    "hi_tom": "TT_HFT",
    "mid_tom": "TT_LMT",
    "low_tom": "TT_HMT",
    "crash_left": "CY_CR",  # Crash
    "ride": "CY_RD",  # Ride
}
"""It seems like there is an error in the mapping of the toms in the stem gmd audio and the provided isolated
samples."""
# audio -> isolated sample
## low tom -> hi tom
# mid tom -> mid tom
# hi tom -> low tom
# The reasoning to not get confused is:
# The label loads the audio.
# eg My TT_HMT (stem_gmd_drum_class_to_symbol) loads the audio for the hi_tom. Their TT_HMT (stem_gmd_single_hits_map) loads the audio for the high_tom
# But their high tom is actually the low tom in the isolated samples
# So the mapping should be


drum_kit_map_stemgmd = {
    "brooklyn": 0,
    "heavy": 1,
    "east_bay": 2,
    "retro_rock": 3,
    "socal": 4,
    "portland": 5,
    "bluebird": 6,
    "detroit_garage": 7,
    "roots": 8,
    "motown_revisited": 9,
}


TRAIN_KITS = ["/brooklyn/", "/heavy/", "/portland/", "/east_bay/", "/retro_rock/", "/socal/"]
TEST_KITS = ["/bluebird/", "/detroit_garage/", "/motown_revisited/", "/roots/"]
EVAL_SESSION = ["/eval_session/"]

# Action: Don't filter a group (value is None)
keep_train = dict.fromkeys(TRAIN_KITS)
keep_test = dict.fromkeys(TEST_KITS)
# Action: Keep a group (value is "keep") this will filter out files which are not in the group
keep_eval_session = dict.fromkeys(EVAL_SESSION, "keep")
# Action: Remove a group (value is "remove")
remove_train = dict.fromkeys(TRAIN_KITS, "remove")
remove_test = dict.fromkeys(TEST_KITS, "remove")
remove_eval_session = dict.fromkeys(EVAL_SESSION, "remove")

# `|` merges dictionaries in Python 3.9+)
split_to_filter_map = {
    # This combination is used by train, val, and full_test
    "train_train_kits": keep_train | remove_test | remove_eval_session,
    "val_train_kits": keep_train | remove_test,
    "full_test_train_kits": keep_train | remove_test,
    "test_train_kits": keep_train | remove_test,
    "test_test_kits": remove_train | keep_test,
    "eval_session_train_kits": keep_train | remove_test | keep_eval_session,
    "eval_session_test_kits": remove_train | keep_test | keep_eval_session,
}
