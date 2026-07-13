#!/bin/bash
#PBS -N pivot_benchmark
#PBS -l select=1:ncpus=4:mem=16gb:scratch_local=20gb
#PBS -l walltime=24:00:00
#PBS -j oe

# --------------- User Settings ---------------
DATADIR=/storage/brno12-cerit/home/danilinae/
PYTHON_PROJECT_DIR=$DATADIR/Simplex
# Which training entry point to run. Override at submit time with, e.g.:
#   qsub -v SCRIPT=train_dqn.py simplexJob.sh
#   qsub -v SCRIPT=train_sac.py simplexJob.sh
# Defaults to the PPO pipeline so existing submissions are unchanged.
BENCH_SCRIPT="${SCRIPT:-train.py}"

# --------------- Function: Copy Results Back ---------------
copy_results() {
    echo "Copying results back to $DATADIR at $(date)"
    cd $SCRATCHDIR/Simplex 2>/dev/null || return
    cp benchmark_results_*.csv $PYTHON_PROJECT_DIR/ 2>/dev/null || true
    cp train_time_vs_size_*.png $PYTHON_PROJECT_DIR/ 2>/dev/null || true
    cp avg_pivot_steps_vs_size_*.png $PYTHON_PROJECT_DIR/ 2>/dev/null || true
    mkdir -p $PYTHON_PROJECT_DIR/models
    cp -r models/* $PYTHON_PROJECT_DIR/models/ 2>/dev/null || true
    echo "Results copied at $(date)"
}

# --------------- Trap: Save on Job Kill/Timeout ---------------
# PBS sends SIGTERM before killing the job; copy whatever we have
trap 'echo "Caught signal — saving models before exit"; copy_results; clean_scratch; exit' TERM INT

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

# --------------- Install All Dependencies Except SB3 ---------------
if [ -f "requirements.txt" ]; then
    grep -v -E '^(stable_baselines3|torch|triton|nvidia)' requirements.txt > reqs_filtered.txt
    pip install --no-cache-dir -r reqs_filtered.txt \
      || { echo "ERROR: installing filtered requirements failed"; exit 3; }
else
    echo "WARNING: requirements.txt not found—skipping non-SB3 install"
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

# --------------- Install OpenSpiel (provides pyspiel) for Leduc mode ---------------
pip install --no-cache-dir open_spiel \
  || { echo "ERROR: installing open_spiel failed"; exit 8; }

# --------------- Pin numpy<2 (SB3 2.4.1 is not numpy-2 compatible) ---------------
# open_spiel / other deps may pull numpy>=2; force a compatible version LAST so
# it wins, otherwise SB3 breaks at runtime ("numpy 2.x is incompatible").
pip install --no-cache-dir "numpy<2.0" \
  || { echo "ERROR: pinning numpy<2 failed"; exit 9; }

# --------------- Verify the leduc-mode import actually works ---------------
# Guards against a stale job copy / silently-skipped install: fail HERE with a
# clear message instead of crashing 40 lines into env construction.
python3 -c "import pyspiel; import numpy; assert numpy.__version__ < '2', numpy.__version__" \
  || { echo "ERROR: pyspiel/numpy sanity import failed"; exit 10; }

# --------------- Ensure models directory exists ---------------
mkdir -p models

# --------------- Run the Benchmark Script ---------------
echo "Starting benchmark ($BENCH_SCRIPT) at $(date)"
python3 $BENCH_SCRIPT &
TRAIN_PID=$!

# Wait for the training process; if we get signalled, the trap fires
wait $TRAIN_PID
TRAIN_EXIT=$?

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "WARNING: Benchmark script exited with code $TRAIN_EXIT"
fi
echo "Benchmark finished at $(date)"

# --------------- Copy Results Back ---------------
copy_results

# --------------- Clean Scratch ---------------
clean_scratch

exit 0