import sys
import os
import json
import datetime
import pdb
import torch
from zoneinfo import ZoneInfo

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(base_dir)
os.chdir(base_dir)

import pipeline.main_utils as main_utils
from config.access_tokens import hf_access_token
from huggingface_hub import login

# Prefer HF_TOKEN env var, fall back to config/access_tokens.py.  Skip login
# silently if neither is set (users may have already cached their token).
_hf_token = os.environ.get("HF_TOKEN") or hf_access_token
if _hf_token:
    login(token=_hf_token)

from pipeline.sketchwalk.run import run

args = main_utils.parse_args()
SEED = args.seed
main_utils.lock_seed(SEED)
torch.cuda.reset_peak_memory_stats()

ct_timezone = ZoneInfo("America/Chicago")
start_time = datetime.datetime.now(ct_timezone)
config = main_utils.register_args_and_configs(args)
logger = main_utils.set_logger(args.output_folder_dir, args)


logger.info(f"Experiment {config['management']['exp_desc']} (SEED={SEED}) started at {start_time} with the following config: ")
logger.info(json.dumps(config, indent=4))



processed_results, raw_results = run(config, args, logger)
main_utils.register_result(processed_results, raw_results, config)


end_time = datetime.datetime.now(ct_timezone)
main_utils.register_exp_time(start_time, end_time, config)
main_utils.register_output_config(config)
logger.info(f"Experiment {config['management']['exp_desc']} ended at {end_time}. Duration: {config['management']['exp_duration']}")