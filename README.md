# SkyGPT reorganized package

这是一个按数据、模型、脚本、配置、输出重新整理后的代码包。核心训练与推理逻辑保留了你给我的 wrappers / pv_stage 代码思路，只做了结构化重组，并补上了从 CSV 准备 H5 的数据脚本，便于直接支持 `skippd` 和 `folmos` 两套数据。

## 目录结构

- `SkyGPT/`：官方源码位置。你把官方仓库直接 `git clone` 到这里。
- `configs/{skippd,folmos}/`：每个数据集单独的 video / pv 配置。
- `data/{skippd,folmos}/raw/`：原始 `train.csv / val.csv / test.csv` 和图像。
- `data/{skippd,folmos}/prepared/video/`：视频阶段 H5。
- `data/{skippd,folmos}/prepared/pv/`：PV 阶段 H5。
- `data/scripts/`：CSV -> H5 的数据准备脚本。
- `models/video/`：VQ-VAE、Transformer、采样、导出。
- `models/pv/`：UNet 回归器、PV 训练、推理、评估。
- `utils/`：通用工具。
- `scripts/`：一键/后台启动脚本。
- `outputs/{skippd,folmos}/`：按数据集分开的训练模型与结果。

## 我删除或未保留到主包里的内容

我把以下明显偏一次性分析/调试用途、不是主流程必须的脚本从主包里移除了：
- `build_custom_test_eval_from_selected_images.py`
- `test_vqvae_range_windows.py`
- `test_transformer_range_windows.py`
- `predict_transformer_by_date.py`
- `save_selected_timestamp_images.py`
- `evaluate_generated_images_quality.py`
- 原始的重复启动脚本与旧目录绑定脚本

这些脚本不是不能用，而是它们更像临时诊断工具，不适合作为主工程骨架的一部分。

## skippd 旧模型和旧数据怎么放

### 1. 官方源码
把官方仓库放到：
`SkyGPT/`

### 2. skippd 原始 CSV 与图像
放到：
- `data/skippd/raw/train.csv`
- `data/skippd/raw/val.csv`
- `data/skippd/raw/test.csv`

CSV 中 `image_path` 可以写绝对路径，也可以写相对当前 CSV 文件的相对路径。

### 3. 你之前已经准备好的 skippd H5
如果你不想重新准备，可以直接放到：
- `data/skippd/prepared/video/train_val.h5`
- `data/skippd/prepared/video/val_eval.h5`
- `data/skippd/prepared/video/test_eval.h5`
- `data/skippd/prepared/pv/pv_mapping_dataset.h5`

### 4. 你之前已经训练好的 skippd 模型
移动到：
- `outputs/skippd/vqvae/checkpoints/best.ckpt`
- `outputs/skippd/vqvae/checkpoints/last.ckpt`
- `outputs/skippd/transformer/checkpoints/best.ckpt`
- `outputs/skippd/transformer/checkpoints/last.ckpt`
- `outputs/skippd/pv_model/best_pv_mapper_torch.pt`
- `outputs/skippd/pv_model/last_pv_mapper_torch.pt`

### 5. 你之前导出的图像生成结果
放到：
- `outputs/skippd/image_output/generated_test_images_sequences.npy`
- `outputs/skippd/image_output/generated_test_images_current_ts.npy`
- `outputs/skippd/image_output/generated_test_images_target_ts.npy`
- `outputs/skippd/image_output/generated_test_images_source_indices.npy`

## folmos 配置说明

我已经把 `folmos` 配成：
- 时间间隔：5 min
- 历史图像：24 张
- 未来图像：3 张
- 总序列长度：27
- 目标图像：未来第 15 分钟，也就是第 3 个未来点
- PV 目标：当前时刻对应行的 `target_ghi_15min`

也就是：
- Video 阶段：`cond_frames=24`, `future_frames=3`, `sequence_length=27`
- PV 阶段最终从生成结果里默认取最后一帧，也就是 `t+15min`

## 从 CSV 开始训练新的 folmos 数据集

### 第一步：放原始数据
把 folmos 原始数据放到：
- `data/folmos/raw/train.csv`
- `data/folmos/raw/val.csv`
- `data/folmos/raw/test.csv`

### 第二步：准备 video 阶段 H5
```bash
python data/scripts/prepare_video_data.py --dataset folmos
```

### 第三步：准备 pv 阶段 H5
```bash
python data/scripts/prepare_pv_data.py --dataset folmos
```

### 第四步：训练 VQ-VAE
```bash
python models/video/train_vqvae.py --dataset folmos
```

### 第五步：训练 Transformer
```bash
python models/video/train_transformer.py --dataset folmos
```

### 第六步：导出测试集生成图像
```bash
python models/video/export_generated_sequences.py --dataset folmos
```

### 第七步：训练 PV 映射模型
```bash
python models/pv/train_pv_mapper.py --dataset folmos
```

### 第八步：用生成图像做 PV 预测
```bash
python models/pv/infer_from_generated.py --dataset folmos
```

### 第九步：评估 PV 预测
```bash
python models/pv/eval_pv.py --dataset folmos
```

## 一键运行

### video 主流程
```bash
python scripts/run_video_stage.py --dataset folmos
```

### pv 主流程
```bash
python scripts/run_pv_stage.py --dataset folmos
```

### 后台运行
```bash
bash scripts/train_video_background.sh folmos start
bash scripts/train_pv_background.sh folmos start
```

## 你接下来最需要检查的地方

1. `SkyGPT/` 下官方仓库的真实层级，是否仍然是 `codes/video_prediction/SkyGPT`
2. CSV 里的 `image_path` 是否有效
3. `configs/folmos/video_config.py` 里的 VQ-VAE `downsample=[1,3,3]` 是否与你当前官方模型兼容  
4. 你的 folmos 图像尺寸是否仍然用 64×64；如果不是，要在两个配置文件里一起改

## 说明

这个代码包没有把官方 `SkyGPT` 仓库打进去，只预留了目录位。原因是官方源码不是你上传的内容，而且通常直接 git clone 更稳。
