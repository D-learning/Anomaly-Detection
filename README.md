# G1 / G3：F1-R2 持续异常检测

本仓库发布基于 ContCore / F1-R2 的两个版本及完整 MVTec AD 15 类实验结果。

|版本|记忆模块|推理模块|
|---|---|---|
|G1|类别首次出现时按最终容量进行 R2 类内压缩，之后固定旧类原型|原始全局共享库|
|G3|与 G1 相同|正常训练图像描述符预测类别，仅在预测类别库内评分|

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

### G3

```bash
# 1×15（15步）
CUDA_VISIBLE_DEVICES=0 G_VARIANT=G3 python -u main.py \
  --dataset mvtec --data_path "$DATA_ROOT" \
  --incremental_class_list 1-1-14 --coreset_size 4600 --seed 42 \
  --batch_size 12 --num_workers 2 --faiss_on_gpu False \
  --faiss_num_workers 4 --nn_backend torch \
  --output_root "$OUTPUT_ROOT/G3_1-1-14"

# 3×5（5步）
CUDA_VISIBLE_DEVICES=0 G_VARIANT=G3 python -u main.py \
  --dataset mvtec --data_path "$DATA_ROOT" \
  --incremental_class_list 3-3-4 --coreset_size 4600 --seed 42 \
  --batch_size 12 --num_workers 2 --faiss_on_gpu False \
  --faiss_num_workers 4 --nn_backend torch \
  --output_root "$OUTPUT_ROOT/G3_3-3-4"

# 10−1×5（6步）
CUDA_VISIBLE_DEVICES=0 G_VARIANT=G3 python -u main.py \
  --dataset mvtec --data_path "$DATA_ROOT" \
  --incremental_class_list 10-1-5 --coreset_size 4600 --seed 42 \
  --batch_size 12 --num_workers 2 --faiss_on_gpu False \
  --faiss_num_workers 4 --nn_backend torch \
  --output_root "$OUTPUT_ROOT/G3_10-1-5"

# 10−5（2步）
CUDA_VISIBLE_DEVICES=0 G_VARIANT=G3 python -u main.py \
  --dataset mvtec --data_path "$DATA_ROOT" \
  --incremental_class_list 10-5-1 --coreset_size 4600 --seed 42 \
  --batch_size 12 --num_workers 2 --faiss_on_gpu False \
  --faiss_num_workers 4 --nn_backend torch \
  --output_root "$OUTPUT_ROOT/G3_10-5-1"

# 14−1（2步）
CUDA_VISIBLE_DEVICES=0 G_VARIANT=G3 python -u main.py \
  --dataset mvtec --data_path "$DATA_ROOT" \
  --incremental_class_list 14-1-1 --coreset_size 4600 --seed 42 \
  --batch_size 12 --num_workers 2 --faiss_on_gpu False \
  --faiss_num_workers 4 --nn_backend torch \
  --output_root "$OUTPUT_ROOT/G3_14-1-1"

```

## 对应实验结果

以下为已完成的10组 seed42 实验，正式队列结束时间为2026-11-18 02:41 UTC。所有退出码为0。每格是最终指标（%）/ 对应逐类别 FM（百分点）；指标越高越好、FM 越低越好。

### 图像 AUROC / FM

|方法|1×15（15步）|3×5（5步）|10−1×5（6步）|10−5（2步）|14−1（2步）|
|---|---:|---:|---:|---:|---:|
|G1|94.0821 / 0.222916|94.3431 / 0.369969|94.7781 / 0.011921|95.0736 / 0.004248|96.1214 / 0.000000|
|G3|91.4622 / 0.000000|94.1439 / 0.002273|93.7390 / 0.000000|94.0371 / 0.000000|96.1199 / 0.000000|

### 像素 AUROC / FM

|方法|1×15（15步）|3×5（5步）|10−1×5（6步）|10−5（2步）|14−1（2步）|
|---|---:|---:|---:|---:|---:|
|G1|95.7480 / 0.262722|95.6025 / 0.227750|95.5246 / 0.104648|95.7392 / 0.075075|95.7320 / -0.000178|
|G3|96.2225 / 0.000040|96.2317 / 0.000027|96.3553 / 0.000003|96.5312 / 0.000000|96.5078 / 0.000001|

### 图像 AP / FM

|方法|1×15（15步）|3×5（5步）|10−1×5（6步）|10−5（2步）|14−1（2步）|
|---|---:|---:|---:|---:|---:|
|G1|97.9118 / 0.114878|97.8645 / 0.154105|97.9872 / 0.006162|98.1485 / 0.008230|98.5037 / 0.000000|
|G3|97.2153 / 0.000000|97.9090 / 0.000552|97.7957 / 0.000000|97.9585 / 0.000000|98.5669 / 0.000000|

### 像素 AP / FM

|方法|1×15（15步）|3×5（5步）|10−1×5（6步）|10−5（2步）|14−1（2步）|
|---|---:|---:|---:|---:|---:|
|G1|55.4825 / 0.453155|55.2556 / 0.427012|55.9167 / 0.294948|56.0557 / 0.383241|56.1804 / -0.002425|
|G3|56.8276 / 0.000165|57.0741 / 0.000045|58.2282 / -0.000009|58.3107 / 0.000000|58.2769 / -0.000015|

### 完整最终指标

|方法|任务|Image-AUROC|Pixel-AUROC|F1-Img|F1-Pix|AP-Img|AP-Pix|AUPRO|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|G1|1×15（15步）|94.0821|95.7480|94.9444|58.2609|97.9118|55.4825|88.0941|
|G1|3×5（5步）|94.3431|95.6025|94.7463|57.9316|97.8645|55.2556|87.9811|
|G1|10−1×5（6步）|94.7781|95.5246|95.3639|58.6706|97.9872|55.9167|88.5959|
|G1|10−5（2步）|95.0736|95.7392|95.7766|58.8232|98.1485|56.0557|88.3120|
|G1|14−1（2步）|96.1214|95.7320|96.3239|58.6560|98.5037|56.1804|88.7005|
|G3|1×15（15步）|91.4622|96.2225|94.7958|59.0203|97.2153|56.8276|87.3120|
|G3|3×5（5步）|94.1439|96.2317|94.7391|59.0014|97.9090|57.0741|87.6213|
|G3|10−1×5（6步）|93.7390|96.3553|95.0031|60.0971|97.7957|58.2282|88.5421|
|G3|10−5（2步）|94.0371|96.5312|95.4158|60.2007|97.9585|58.3107|88.1035|
|G3|14−1（2步）|96.1199|96.5078|96.4148|59.9924|98.5669|58.2769|88.5430|

## 指标口径与适用条件

- 四项 FM：逐旧类别计算“最终阶段之前的最高指标 − 最终指标”，再对旧类别宏平均，保留负数；最后新增类别不参与。不能用阶段总均值相减替代。
- 任务组宏平均 FM 单独保存于 results/summary.json 的 task_macro_fm，不能与逐类别 FM 互换。
- F1max 为测试最优阈值指标，不是部署阈值；AUROC/AP 使用原始分数。AUPRO 使用本仓库 metrics.py 的固定实现；跨论文须核对积分范围和实现。
- 最终15类已知：每类 K//15 槽位，前 K%15 类各多一个；未到达类别槽位空闲。早期实际容量较小，必须结合首次精度和最终精度评价 FM，不能仅按遗忘判优。
- G3 路由只读取图像特征，不读取真实测试类别。本次五协议最终路由准确率均为100%；不保证其他数据或类别顺序也成立。
- G3 的近零遗忘来自冻结主干、固定旧类记忆、隔离类别检索以及正确路由，不是共享可训练主干抗遗忘的证据。浮点与分批运算可能产生微小变化。
- 两版的持久状态均遵守原5000原型等价字节上限，包括来源、R2元数据和G3路由中心；不保存新模型权重。
- 这是单seed结果，不包含多seed方差。G1平均图像AUROC更高，G3遗忘与像素指标更优，不能声称G3在所有精度指标最优。

## 结果文件与代码

- `results/G1/<协议>/results_sN.csv`、`results/G3/<协议>/results_sN.csv`：全部60个阶段的逐类别原始指标。
- `results/summary.json`：最终、首次、四项FM、正部/最差/P90遗忘及路由准确率。
- `results/per_class_forgetting.csv`：逐类别遗忘；任务组CSV保留另一种聚合口径。
- `results/verification.json`：10组真实退出与原始阶段CSV复算检查。
- `corelib/r2next.py`：固定配额记忆和预测路由；`corelib/sampler.py` / `corelib/contcore.py`：接入点。
- `main.py`：单任务协议运行入口，自动保存配置、源码快照、每阶段结果和路由明细。

## 来源

基础工程：[jungyg/ContCore](https://github.com/jungyg/ContCore)。本发布保留运行所需基础模块，G1/G3入口明确选择对应模块。移除了未使用且要求额外dinov1目录的PatchEmbed导入；未改变实验中的WideResNet计算。没有为上游代码新增许可声明。

本地复算已发布结果：python verify_results.py（仅依赖Python标准库）。
