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

        if [[ ! -d "$json_dir" ]]; then
            echo "Directory $json_dir not found — skipping."
            continue
        fi

        for json_file in "$json_dir"/filter_indices_*pct.json; do
            if [[ ! -f "$json_file" ]]; then
                echo "No matching JSON files in $json_dir — skipping."
                continue
            fi

            echo ""
            echo "================== NEW RUN =================="
            echo "Model File:  $model_file"
            echo "Model Dir:   $model_dir"
            echo "JSON File:   $(basename "$json_file")"
            echo "JSON Dir:    $json_dir"
            echo "============================================="

            MODEL_PATH="$model_path" JSON_PATH="$json_file" python "$PYTHON_SCRIPT"

            echo "Finished model $model_file with JSON $(basename "$json_file")"
            echo "----------------------------------------"
        done
    fi
done
