#!/bin/bash

MODEL_DIR="./model/VGG11"
CONTENT_DIR="./content_for_TMR"
PYTHON_SCRIPT="TMR_in_CNN.py"

for model_path in "$MODEL_DIR"/VGG11_*.pt; do
    model_file=$(basename "$model_path")
    model_dir=$(dirname "$model_path")

    if [[ $model_file =~ _([0-9]+)\.pt$ ]]; then
        prune_percent="${BASH_REMATCH[1]}"
        json_dir="$CONTENT_DIR/${prune_percent}%_Prune"
        json_path="$json_dir/filter_indices_60pct.json"

        if [[ ! -f "$json_path" ]]; then
            echo "JSON not found for prune $prune_percent% — skipping."
            continue
        fi

        echo ""
        echo "================== NEW RUN =================="
        echo "Model File:  $model_file"
        echo "Model Dir:   $model_dir"
        echo "Using JSON:  filter_indices_60pct.json"
        echo "JSON Dir:    $json_dir"
        echo "============================================="

        MODEL_PATH="$model_path" JSON_PATH="$json_path" python "$PYTHON_SCRIPT"

        echo "Finished model $model_file"
        echo "----------------------------------------"
    fi
done
