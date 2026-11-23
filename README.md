# G1：F1-R2 持续异常检测

本仓库发布基于 ContCore / F1-R2 的版本及完整 MVTec AD 15 类实验结果。

|版本|记忆模块|推理模块|
|---|---|---|
|G1|类别首次出现时按最终容量进行 R2 类内压缩，之后固定旧类原型|原始全局共享库|

固定主干、4600 原型上限、seed42、batch12；类别顺序为 bottle、cable、capsule、carpet、grid、hazelnut、leather、metal_nut、pill、screw、tile、toothbrush、transistor、wood、zipper。

## 环境与数据

实验使用 Linux + NVIDIA GPU。Python 3.10 或与 PyTorch 兼容的版本；精确实验包版本见 [results/environment.json](results/environment.json)。先安装适配显卡的 PyTorch/torchvision，再安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate
# 参考环境：torch 2.9.1+cu130；torchvision 0.24.1+cu130
# 按本机驱动从 PyTorch 官方安装匹配的 CUDA 构建。
pip install -r requirements.txt
```

RTX PRO Blackwell 等新显卡需要支持其架构的 PyTorch/CUDA 构建。torchvision 默认下载 ImageNet WideResNet50-2 预训练权重；离线运行需提前准备对应 torch hub 缓存。仓库不包含数据集和模型权重。

数据目录应为 `$DATA_ROOT/bottle/train/good/*.png`、`$DATA_ROOT/bottle/test/...` 和 `$DATA_ROOT/bottle/ground_truth/...`，其余类别同结构。实验使用完整15类（含 leather），每张图像只计入一次，避免嵌套副本重复。

```bash
export DATA_ROOT=/path/to/mvtec_anomaly_detection
export OUTPUT_ROOT="$PWD/outputs"
mkdir -p "$OUTPUT_ROOT"
export OPT_VARIANT=Rhier RB_VARIANT=R2 USE_RSC=0 USE_H1=0
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CONTCORE_NUM_THREADS=4
unset FULL15_VARIANT FULL15_SMOKE G_SMOKE
```

## 每个任务的运行命令

任务参数含义为 `[A] + [B] * N`。每条命令独立初始化一个模型；该命令内部跨阶段保留持续状态。CUDA_VISIBLE_DEVICES=0 使用第一张卡，可改为1。正式结果使用全部正常训练图像，不设置 G_SMOKE。

### G1

```bash
# 1×15（15步）
CUDA_VISIBLE_DEVICES=0 G_VARIANT=G1 python -u main.py \
  --dataset mvtec --data_path "$DATA_ROOT" \
  --incremental_class_list 1-1-14 --coreset_size 4600 --seed 42 \
  --batch_size 12 --num_workers 2 --faiss_on_gpu False \
  --faiss_num_workers 4 --nn_backend torch \
  --output_root "$OUTPUT_ROOT/G1_1-1-14"

# 3×5（5步）
CUDA_VISIBLE_DEVICES=0 G_VARIANT=G1 python -u main.py \
  --dataset mvtec --data_path "$DATA_ROOT" \
  --incremental_class_list 3-3-4 --coreset_size 4600 --seed 42 \
  --batch_size 12 --num_workers 2 --faiss_on_gpu False \
  --faiss_num_workers 4 --nn_backend torch \
  --output_root "$OUTPUT_ROOT/G1_3-3-4"

# 10−1×5（6步）
CUDA_VISIBLE_DEVICES=0 G_VARIANT=G1 python -u main.py \
  --dataset mvtec --data_path "$DATA_ROOT" \
  --incremental_class_list 10-1-5 --coreset_size 4600 --seed 42 \
  --batch_size 12 --num_workers 2 --faiss_on_gpu False \
  --faiss_num_workers 4 --nn_backend torch \
  --output_root "$OUTPUT_ROOT/G1_10-1-5"

# 10−5（2步）
CUDA_VISIBLE_DEVICES=0 G_VARIANT=G1 python -u main.py \
  --dataset mvtec --data_path "$DATA_ROOT" \
  --incremental_class_list 10-5-1 --coreset_size 4600 --seed 42 \
  --batch_size 12 --num_workers 2 --faiss_on_gpu False \
  --faiss_num_workers 4 --nn_backend torch \
  --output_root "$OUTPUT_ROOT/G1_10-5-1"

# 14−1（2步）
CUDA_VISIBLE_DEVICES=0 G_VARIANT=G1 python -u main.py \
  --dataset mvtec --data_path "$DATA_ROOT" \
  --incremental_class_list 14-1-1 --coreset_size 4600 --seed 42 \
  --batch_size 12 --num_workers 2 --faiss_on_gpu False \
  --faiss_num_workers 4 --nn_backend torch \
  --output_root "$OUTPUT_ROOT/G1_14-1-1"

```

## 对应实验结果

以下为已完成的5组 seed42 实验，正式队列结束时间为2026-11-18 02:41 UTC。所有退出码为0。每格是最终指标（%）/ 对应逐类别 FM（百分点）；指标越高越好、FM 越低越好。

### 图像 AUROC / FM

|方法|1×15（15步）|3×5（5步）|10−1×5（6步）|10−5（2步）|14−1（2步）|
|---|---:|---:|---:|---:|---:|
|G1|94.0821 / 0.222916|94.3431 / 0.369969|94.7781 / 0.011921|95.0736 / 0.004248|96.1214 / 0.000000|

### 像素 AUROC / FM

|方法|1×15（15步）|3×5（5步）|10−1×5（6步）|10−5（2步）|14−1（2步）|
|---|---:|---:|---:|---:|---:|
|G1|95.7480 / 0.262722|95.6025 / 0.227750|95.5246 / 0.104648|95.7392 / 0.075075|95.7320 / -0.000178|

### 图像 AP / FM

|方法|1×15（15步）|3×5（5步）|10−1×5（6步）|10−5（2步）|14−1（2步）|
|---|---:|---:|---:|---:|---:|
|G1|97.9118 / 0.114878|97.8645 / 0.154105|97.9872 / 0.006162|98.1485 / 0.008230|98.5037 / 0.000000|

### 像素 AP / FM

|方法|1×15（15步）|3×5（5步）|10−1×5（6步）|10−5（2步）|14−1（2步）|
|---|---:|---:|---:|---:|---:|
|G1|55.4825 / 0.453155|55.2556 / 0.427012|55.9167 / 0.294948|56.0557 / 0.383241|56.1804 / -0.002425|

### 完整最终指标

|方法|任务|Image-AUROC|Pixel-AUROC|F1-Img|F1-Pix|AP-Img|AP-Pix|AUPRO|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|G1|1×15（15步）|94.0821|95.7480|94.9444|58.2609|97.9118|55.4825|88.0941|
|G1|3×5（5步）|94.3431|95.6025|94.7463|57.9316|97.8645|55.2556|87.9811|
|G1|10−1×5（6步）|94.7781|95.5246|95.3639|58.6706|97.9872|55.9167|88.5959|
|G1|10−5（2步）|95.0736|95.7392|95.7766|58.8232|98.1485|56.0557|88.3120|
|G1|14−1（2步）|96.1214|95.7320|96.3239|58.6560|98.5037|56.1804|88.7005|



## 结果文件与代码

- `results/G1/<协议>/results_sN.csv`：全部30个阶段的逐类别原始指标。
- `results/summary.json`：最终、首次、四项FM、正部/最差/P90遗忘。
- `results/per_class_forgetting.csv`：逐类别遗忘；任务组CSV保留另一种聚合口径。
- `results/verification.json`：5组真实退出与原始阶段CSV复算检查。
- `corelib/r2next.py`：固定配额记忆和历史原型保护；`corelib/sampler.py` / `corelib/contcore.py`：接入点。
- `main.py`：单任务协议运行入口，自动保存配置、源码快照、每阶段结果。


## G1敏感性与消融方案

见 [G1_敏感性与消融实验.md](G1_敏感性与消融实验.md)：固定B=4600，分析候选采样比例ρ与近邻数k，明确已有对照和待实现选择器消融。

结果复算：`python verify_results.py`。
