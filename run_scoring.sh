#!/bin/bash

# mode="${1:-test}"
# average_type="${2:-macro}"

# python src/scoring.py --reference_dir ./data/train --prediction_dir ./results/prediction --average "$average_type"

# if [ "$mode" = "-debug" ]; then
#     python src/scoring.py --reference_dir ./data/train --prediction_dir ./results/missing_files --average "$average_type"
#     python src/scoring.py --reference_dir ./data/val --prediction_dir ./results/dummy_all_sent_all_label --average "$average_type"
#     python src/scoring.py --reference_dir ./data/val --prediction_dir ./results/dummy_all_doc_2_Software --average "$average_type"
# fi

python3 src/compute_metrics.py --reference_dir ./data/test_labeled --prediction_dir ./results/deepseek-chat/test_labeled --mode exact

python3 src/compute_metrics.py --reference_dir ./data/test_labeled --prediction_dir ./results/deepseek-chat/test_labeled --mode partial
