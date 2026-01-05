# Training

## Build Dataset

```
uv run src/build_dataset.py --output ../data/
```

## Train Model

```
bash scripts/run_training.sh -m Qwen/Qwen3-0.6B -d <Absolute Path to Data>
```

```
DATA_PATH=<Absolute Path to Data>
bash scripts/run_async_training.sh -m Qwen/Qwen3-4B -d $DATA_PATH 2>&1 | tee training.log
```

```
DATA_PATH=/workspace/liyuchen/workspace/agentic-code-search-oss/data/swe_gym
bash scripts/run_async_training.sh \
    -m Qwen/Qwen3-4B \
    -o "+generator.exp_config=configs/skyrl-experiments/read-only.yaml" \
    -d $DATA_PATH \
    2>&1 | tee logs/1230_merge_upstream_code_fix_regerror.log
```
