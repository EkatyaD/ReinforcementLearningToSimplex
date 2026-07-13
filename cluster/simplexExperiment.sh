#!/bin/bash
#PBS -N pivot_experiment
#PBS -l select=1:ncpus=4:mem=16gb:scratch_local=20gb
#PBS -l walltime=4:00:00
#PBS -j oe

# --------------- User Settings ---------------
DATADIR=/storage/brno12-cerit/home/danilinae/
PYTHON_PROJECT_DIR=$DATADIR/Simplex
BENCH_SCRIPT=experiment.py

# --------------- Function: Copy Results Back ---------------
copy_results() {
    echo "Copying results back to $PYTHON_PROJECT_DIR at $(date)"
    cd $SCRATCHDIR/Simplex 2>/dev/null || return
    cp results_*.json $PYTHON_PROJECT_DIR/ 2>/dev/null || true
    echo "Results copied at $(date)"
}

# --------------- Trap: Save on Job Kill/Timeout ---------------
trap 'echo "Caught signal — saving results before exit"; copy_results; clean_scratch; exit' TERM INT

# --------------- Logging Info ---------------
echo "$PBS_JOBID is running on node $(hostname -f) in SCRATCHDIR=$SCRATCHDIR" \
    >> $DATADIR/jobs_info.txt

# --------------- Load System Python Module ---------------
module add python/3.10

# --------------- Verify SCRATCHDIR ---------------
test -n "$SCRATCHDIR" || { echo "ERROR: SCRATCHDIR is not set!"; exit 1; }

# --------------- Copy Python Project to Scratch ---------------
mkdir -p $SCRATCHDIR/Simplex
cp -r $PYTHON_PROJECT_DIR/* $SCRATCHDIR/Simplex/ || { echo "ERROR: Failed to copy project"; exit 2; }
cd $SCRATCHDIR/Simplex

# --------------- Create & Activate a Virtual Environment ---------------
python3 -m venv venv
source venv/bin/activate

# --------------- Upgrade pip in venv ---------------
pip install --upgrade pip

# --------------- Install All Dependencies Except SB3/torch/nvidia ---------------
if [ -f "requirements.txt" ]; then
    grep -v -E '^(stable_baselines3|torch|triton|nvidia)' requirements.txt > reqs_filtered.txt
    pip install --no-cache-dir -r reqs_filtered.txt \
      || { echo "ERROR: installing filtered requirements failed"; exit 3; }
else
    echo "WARNING: requirements.txt not found—skipping install"
fi

# --------------- Install Stable-Baselines3 Without Dependencies ---------------
pip install --no-cache-dir --no-deps stable_baselines3==2.4.1 \
  || { echo "ERROR: installing SB3 failed"; exit 4; }

# --------------- Install CPU-only PyTorch Wheel (~100 MB) ---------------
pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.4.1 \
  || { echo "ERROR: installing CPU-only torch failed"; exit 5; }

# --------------- Install Gymnasium for SB3 ---------------
pip install --no-cache-dir gymnasium==1.0.0 \
  || { echo "ERROR: installing gymnasium failed"; exit 6; }

# --------------- Install Shimmy for Gym→Gymnasium Bridge ---------------
pip install --no-cache-dir "shimmy>=2.0" \
  || { echo "ERROR: installing shimmy failed"; exit 7; }

# --------------- Run the Experiment Script ---------------
echo "Starting experiment at $(date)"
python3 $BENCH_SCRIPT --save results_experiment.json &
EXPERIMENT_PID=$!

wait $EXPERIMENT_PID
EXPERIMENT_EXIT=$?

if [ $EXPERIMENT_EXIT -ne 0 ]; then
    echo "WARNING: Experiment script exited with code $EXPERIMENT_EXIT"
fi
echo "Experiment finished at $(date)"

# --------------- Copy Results Back ---------------
copy_results

# --------------- Clean Scratch ---------------
clean_scratch

exit 0