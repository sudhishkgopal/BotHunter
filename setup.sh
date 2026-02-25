#!/usr/bin/env bash
set -euo pipefail

# Load config 
CONFIG="config.json"

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: $CONFIG not found. Run this script from the project root."
    exit 1
fi

# Parse values from config.json using python (avoids jq dependency)
SNAP_URL=$(python3 -c "import json; print(json.load(open('$CONFIG'))['snap_dataset_url'])")
DB_PATH=$(python3 -c "import json; print(json.load(open('$CONFIG'))['database_path'])")

DATASET_GZ="twitter_combined.txt.gz"
DATASET="twitter_combined.txt"

echo "  BotHunter Setup"

#  Step 1: Install Python dependencies 
echo "[1/4] Installing Python dependencies..."
pip install -r requirements.txt
echo "      Done."
echo ""

#  Step 2: Download SNAP dataset 
if [ -f "$DATASET" ]; then
    echo "[2/4] SNAP dataset already exists ($DATASET). Skipping download."
else
    echo "[2/4] Downloading SNAP Twitter dataset..."
    echo "      URL: $SNAP_URL"

    if command -v wget &> /dev/null; then
        wget -q --show-progress -O "$DATASET_GZ" "$SNAP_URL"
    elif command -v curl &> /dev/null; then
        curl -L --progress-bar -o "$DATASET_GZ" "$SNAP_URL"
    else
        echo "ERROR: Neither wget nor curl found. Install one and retry."
        exit 1
    fi

    echo "      Extracting..."
    gunzip -f "$DATASET_GZ"
    echo "      Done. $(wc -l < "$DATASET") lines."
fi
echo ""

# Step 3: Initialize the database 
echo "[3/4] Initializing database at $DB_PATH..."
python3 database.py
echo "      Done."
echo ""

# Step 4: Run the ingestor 
if [ -f "$DB_PATH" ]; then
    # Check if DB already has data
    ROW_COUNT=$(python3 -c "
import sqlite3
conn = sqlite3.connect('$DB_PATH')
count = conn.execute('SELECT COUNT(*) FROM users').fetchone()[0]
conn.close()
print(count)
")
    if [ "$ROW_COUNT" -gt 0 ]; then
        echo "[4/4] Database already has $ROW_COUNT users. Skipping ingest."
        echo "      To re-ingest, delete $DB_PATH and run setup.sh again."
    else
        echo "[4/4] Running ingestor..."
        python3 ingestor.py
        echo "      Done."
    fi
else
    echo "[4/4] Running ingestor..."
    python3 ingestor.py
    echo "      Done."
fi

echo "  Setup complete!"
echo ""
echo "  Run the dashboard:"
echo "    python -m streamlit run app.py"
echo ""
echo "  Run the CLI:"
echo "    python cli.py --threshold 20"
