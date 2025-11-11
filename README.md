# 线性运动轨迹 + VCEC

## Setup

### 1. 确认环境

```bash
bash script/setup.sh
conda activate feature_splatting
```

### 2. 训练（使用配置文件）

使用预配置的JSON文件（推荐）：

```bash
# Neural 3D Dataset
python train.py --quiet --eval \
  --config configs/n3d_full/cook_spinach_vcec.json \
  --model_path log/cook_spinach_vcec_full \
  --source_path /path/to/cook_spinach/colmap_0
```

### 3. 训练（不使用配置文件）

```bash
python train.py --quiet --eval \
  --config configs/n3d_lite/cook_spinach.json \
  --model_path log/test_vcec \
  --source_path /path/to/cook_spinach/colmap_0 \
  --use_mer --use_vcec \
  --lambda_mer 0.05 \
  --lambda_vcec 0.1 \
  --mer_start_iter 0 \
  --mer_to_vcec_iter 5000 \
  --vcec_start_iter 5000
```

## 不同场景的推荐配置

### 场景1: 快速运动 + 少视角

```bash
--use_mer --use_vcec \
--lambda_mer 0.08 \
--lambda_vcec 0.15 \
--vcec_tau 2.0
```

### 场景2: 慢速运动

```bash
--use_mer --use_vcec \
--lambda_mer 0.05 \
--lambda_vcec 0.1 \
--vcec_tau 6.0
```

### 场景3: 计算资源有限（仅使用MER，不用VCEC）

```bash
--use_mer \
--lambda_mer 0.05
```

## 测试

训练完成后：

```bash
python test.py --quiet --eval --skip_train \
  --valloader colmapvalid \
  --configpath configs/n3d_lite/cook_spinach_vcec.json \
  --model_path log/cook_spinach_vcec \
  --source_path /path/to/cook_spinach/colmap_0

cat log/cook_spinach_vcec/25000_runtimeresults.json
```

## 对比实验

创建对比：

```bash
# Baseline (无VCEC)
python train.py --config configs/n3d_lite/cook_spinach.json \
  --model_path log/baseline \
  --source_path <data>

# With VCEC
python train.py --config configs/n3d_lite/cook_spinach_vcec.json \
  --model_path log/with_vcec \
  --source_path <data>
```

比较：
```bash
python compare_results.py log/baseline log/with_vcec
```
