# Hunyuan V3 模型 KV Cache FP8 量化功能特性开发文档

## 1. 需求分析

### 1.1 需求背景

| 维度 | 现状 | 需求 |
| :--- | :--- | :--- |
| **GPU (vLLM)** | 已实现 Hunyuan V3 模型的 KV Cache FP8 量化 | - |
| **NPU (vLLM-Ascend)** | 暂未支持 KV Cache FP8 量化能力 | 需要进行框架侧适配，支持 FP8 量化 |
| **融合算子** | GPU 侧使用 `hpc.rope_norm_store_kv_fp8` | NPU 侧通过 `mytest_rope_norm_store_kv_fp8` 调用 |
| **算子能力** | 假设底层 ACLNN 算子已支持 FP8 | 无需实现算子，仅做框架适配 |

### 1.2 模型量化配置细节

| 量化维度 | 量化方式 | 说明 |
| :--- | :--- | :--- |
| **Q 权重** | MXFP8 (W8A8_MXFP8) | vllm-ascend 已支持，使用 `AscendW8A8MXFP8DynamicLinearMethod` |
| **K 权重** | MXFP8 (W8A8_MXFP8) | vllm-ascend 已支持 |
| **V 权重** | MXFP8 (W8A8_MXFP8) | vllm-ascend 已支持 |
| **Q 激活值** | FP8 动态量化 (per-token-per-head) | 动态计算 scale，存储在 KV Cache 中 |
| **K 激活值** | FP8 动态量化 (per-token-per-head) | 动态计算 scale，存储在 KV Cache 中 |
| **V 激活值** | FP8 静态量化 (per-head) | 使用 checkpoint 中的 scale |

### 1.3 模型架构特性

- **GQA (Grouped Query Attention)**: `num_heads // num_kv_heads` 为 4 或 8
- **head_dim**: 128
- **block_size**: 64（参考 vLLM HPC 实现）或 128（参考 vLLM-Ascend 现有实现）

### 1.4 融合算子功能说明

融合算子 `mytest_rope_norm_store_kv_fp8` 融合以下子算子功能：

| 子算子 | 功能描述 |
| :--- | :--- |
| `qk-norm` | Query/Key RMSNorm 归一化处理 |
| `qk-rope` | Rotary Position Embedding 应用 |
| `qk-hardma` | Hadamard 变换（可选，用于提升量化精度） |
| `kv_update` | KV Cache 写入（paged attention 的 scatter 操作） |
| `quant` | FP8 量化（Q/K 动态 per-token-per-head，V 静态 per-head） |

### 1.5 输入输出规格

#### 融合算子输入

| 输入参数 | 类型 | 形状 | 描述 |
| :--- | :--- | :--- | :--- |
| `qkv` | Tensor | `[num_tokens, q_size + 2*kv_size]` | 拼接的 QKV 张量 |
| `cos_sin` | Tensor | `[max_position, head_dim]` | RoPE cos/sin 缓存 |
| `key_cache` | Tensor (FP8) | `[num_blocks, block_size, num_kv_heads, head_size_padded]` | Key Cache (NHD 逻辑) |
| `value_cache` | Tensor (FP8) | `[num_blocks, block_size, num_kv_heads, head_size_padded]` | Value Cache (NHD 逻辑) |
| `seq_lens` | Tensor | `[batch_size]` | 各请求的序列长度 |
| `q_index` | Tensor | `[batch_size + 1]` | 累积 query 长度 |
| `block_table` | Tensor | `[batch_size, max_blocks]` | 块表 |
| `k_scale` | Tensor | `[1]` 或 `[num_kv_heads]` | K 量化 scale |
| `v_scale` | Tensor | `[num_kv_heads]` | V 量化 scale（per-head 静态） |
| `q_norm_weight` | Tensor | `[head_dim]` | Q norm 权重 |
| `k_norm_weight` | Tensor | `[head_dim]` | K norm 权重 |
| `qk_norm_policy` | int | scalar | norm 策略 (1=rope后norm, 2=norm后rope) |

#### 融合算子输出

| 输出参数 | 类型 | 形状 | 描述 |
| :--- | :--- | :--- | :--- |
| `out_q` | Tensor (FP8) | `[num_tokens, num_heads, head_dim]` | 量化后的 Q（含 scale 嵌入） |
| `q_scale` | Tensor (FP32) | `[num_tokens, num_heads]` | Q 的 per-token-per-head scale |
| `split_k_flag` | Tensor (int32) | `[num_decodes, num_kv_heads]` | Split-K 标志（decode 时） |
| `kv_cache` (更新后) | Tensor (FP8) | 同上 | 写入后的 KV Cache |

---

## 2. 后端系统分析

### 2.1 vLLM GPU 侧参考实现分析

#### 2.1.1 关键文件

| 文件 | 路径 | 作用 |
| :--- | :--- | :--- |
| `hpc_attn.py` | `vllm/v1/attention/backends/hpc_attn.py` | HPC 注意力后端，含 KV Cache 形状、scale 拆分逻辑 |
| `rope_norm.py` | `vllm/model_executor/layers/hpc/rope_norm.py` | HPC 融合 RoPE+Norm+KV Write+FP8 Q Quant 算子 |

#### 2.1.2 KV Cache 数据结构（GPU 参考）

vLLM GPU 侧 KV Cache 采用 **per-token-per-head scale 嵌入 head_size 维度** 的设计：

```
分配形状 (逻辑):  (num_blocks, 2, block_size, num_kv_heads, head_size + padded_elems)
运行时 reshape:   (num_blocks, 2, pad_total_rows, num_kv_heads, head_size)
  ├── [:block_size, :, :]  → KV 数据 (FP8)
  └── [block_size:, :, :]  → per-head FP32 scale (以 FP8 字节形式存储)
```

**scale 嵌入计算**（参考 [hpc_attn.py](file:///D:/code/9d32663ab896afbe08b95a6afab7146d_5450219290158151397_m/vllm/v1/attention/backends/hpc_attn.py#L74-L113)）：

```python
_FP8_PER_HEAD_SCALE_ELEMS = 4  # fp32 scale 占 4 个 fp8 槽位

def _fp8_per_head_scale_elems_padded(block_size, num_kv_heads, head_size):
    """计算 head_size 维度需要 padding 的元素数。
    约束: elems * block_size 必须能被 head_size 整除
    """
    raw = _FP8_PER_HEAD_SCALE_ELEMS
    unit = head_size // math.gcd(block_size, head_size)
    elems = ((raw + unit - 1) // unit) * unit
    return elems

def _compute_padded_total_rows(block_size, num_kv_heads, head_size, pad_head_size):
    """计算 reshape 后的总行数。
    pad_total_rows = block_size * pad_head_size // head_size
    """
    return block_size * pad_head_size // head_size
```

#### 2.1.3 kv_data 和 kv_scale 拆分逻辑（GPU 参考）

参考 [hpc_attn.py](file:///D:/code/9d32663ab896afbe08b95a6afab7146d_5450219290158151397_m/vllm/v1/attention/backends/hpc_attn.py#L198-L274) 中的 `_split_fp8_kv_data_and_scale`：

```python
def _split_fp8_kv_data_and_scale(kv_cache_fp8, head_size, is_hnd, kv_cache_quant_config):
    """将 FP8 KV Cache 拆分为 data 和 per-head scale 区域。
    
    处理 NHD 和 HND 两种物理布局。
    返回的 tensor 始终为 NHD 逻辑形状:
        kv_data:  (num_blocks, 2, block_size, num_kv_heads, head_size)
        kv_scale: (num_blocks, 2, scale_rows, num_kv_heads, head_size) or None
    """
    if _needs_per_head_scale_in_cache(kv_cache_quant_config):
        # 使用 as_strided 进行零拷贝 reshape
        # NHD: (B, 2, BS, H, D_pad) → (B, 2, pad_total_row, H, D)
        # HND: (B, 2, H, BS, D_pad) → (B, 2, H, pad_total_row, D)
        kv_data = kv_cache_fp8[:, :, :block_size, :, :]
        kv_scale = kv_cache_fp8[:, :, block_size:, :, :]
    else:
        kv_data = kv_cache_fp8
        kv_scale = None
    return kv_data, kv_scale
```

#### 2.1.4 KV Cache Scale 粒度处理（GPU 参考）

参考 [hpc_attn.py](file:///D:/code/9d32663ab896afbe08b95a6afab7146d_5450219290158151397_m/vllm/v1/attention/backends/hpc_attn.py#L836-L896) 中的 `_get_kv_scales`：

```python
def _get_kv_scales(layer, kv_scale):
    """根据 kv_cache_quant_config 的 granularity 返回 (k_scale, v_scale)。
    
    - per_token_per_head: scale 存储在 KV Cache 的额外行中
    - per_head: 从 checkpoint 加载的 [num_kv_heads] tensor
    - per_tensor: 标量 reshape(1)
    """
    # K scale: per_token_per_head → 从 kv_scale 中取
    # V scale: per_head → 从 layer._v_scale 中取（静态）
```

#### 2.1.5 HpcRopeNorm 前向实现（GPU 参考）

参考 [rope_norm.py](file:///D:/code/9d32663ab896afbe08b95a6afab7146d_5450219290158151397_m/vllm/model_executor/layers/hpc/rope_norm.py#L253-L446) 中的 `_forward_impl`：

```python
def _forward_impl(self, qkv, kv_cache, attn_metadata, attn_layer, output):
    # 1. 拆分 KV Cache 为 data 和 scale
    kv_data, kv_scale = _split_fp8_kv_data_and_scale(kv_cache_fp8, ...)
    k_cache_data = kv_data[:, 0]  # (B, block_size, H, D) NHD
    v_cache_data = kv_data[:, 1]
    
    # 2. 获取 K/V scale
    k_scale, v_scale = HpcAttentionImpl._get_kv_scales(attn_layer.impl, attn_layer, kv_scale)
    
    # 3. Prefill: 调用融合算子
    if attn_metadata.num_prefills > 0:
        _, q_scale, split_k_flag = hpc.rope_norm_store_kv_fp8(
            key_cache=k_cache_data,
            value_cache=v_cache_data,
            qkv=qkv_prefill,
            cos_sin=self.cos_sin_cache,
            num_seqlen_per_req=attn_metadata.seq_lens_prefill,
            q_index=attn_metadata.qo_indptr,
            kvcache_indices=attn_metadata.block_table_prefill,
            is_prefill=True,
            k_scale=k_scale,
            v_scale=v_scale,
            quant_policy=quant_type,
            max_seqlens=attn_metadata.max_query_len,
            q_norm_weight=..., k_norm_weight=...,
            qk_norm_policy=self.qk_norm_policy,
            out_q=output[num_decode_tokens:...],
        )
        attn_metadata.hpc_prefill_q_scale = q_scale
    
    # 4. Decode: 类似 prefill，is_prefill=False
```

### 2.2 vLLM-Ascend NPU 侧代码库结构

```
vllm-ascend-0.18.0rc1/
├── vllm_ascend/
│   ├── attention/                    # 注意力机制实现
│   │   ├── attention_v1.py           # AscendAttentionBackend 主实现
│   │   ├── mla_v1.py                 # MLA 注意力实现
│   │   ├── sfa_v1.py                 # SFA 注意力实现
│   │   └── utils.py                  # 工具函数
│   ├── quantization/                 # 量化相关代码
│   │   ├── methods/                  # 量化方法实现
│   │   │   ├── __init__.py           # 方案导出
│   │   │   ├── base.py               # 量化方案基类 (AscendAttentionScheme)
│   │   │   ├── registry.py           # 量化方案注册器
│   │   │   ├── kv_c8.py              # INT8 KV 量化实现 (FAKQuant)
│   │   │   ├── w8a8_mxfp8.py         # MXFP8 线性/MoE 量化 (已支持)
│   │   │   └── ...
│   │   ├── quant_parser.py           # 量化配置解析
│   │   └── quant_type.py             # 量化类型定义
│   ├── device/
│   │   └── device_op.py              # 设备算子封装 (BaseDeviceAdaptor/A5DeviceAdaptor)
│   ├── ops/                          # 算子封装
│   │   ├── mla.py                    # MLA 算子
│   │   ├── rotary_embedding.py       # Rotary Embedding
│   │   └── ...
│   ├── envs.py                       # 环境变量配置
│   └── patch/                        # vLLM 补丁
│       └── worker/
│           └── patch_huanyuan_vl.py  # Hunyuan 模型补丁
```

### 2.3 关键依赖分析

| 依赖模块 | 文件路径 | 作用 |
| :--- | :--- | :--- |
| `AttentionSpec` | `vllm.v1.kv_cache_interface` | KV Cache 规格定义 |
| `AttentionBackend` | `vllm.v1.attention.backend` | 注意力后端基类 |
| `AttentionImpl` | `vllm.v1.attention.backend` | 注意力实现基类 |
| `AscendAttentionScheme` | `vllm_ascend.quantization.methods.base` | Ascend 注意力量化基类 |
| `register_scheme` | `vllm_ascend.quantization.methods.registry` | 量化方案注册器 |
| `BaseDeviceAdaptor` | `vllm_ascend.device.device_op` | 设备算子适配器基类 |
| `KVCacheQuantConfig` | `vllm.config.cache` | KV Cache 量化配置（含 granularity） |

---

## 3. 方案设计

### 3.1 架构设计

#### 3.1.1 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        vLLM Core                                │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  AttentionBackend Registry  │  KVCacheQuantConfig       │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    vLLM-Ascend Extension                         │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Model Layer (HYV3Attention) [patched]                    │   │
│  │  ┌────────────────────────────────────────────────────┐  │   │
│  │  │  NpuRopeNorm (ops/rope_norm.py)                    │  │   │
│  │  │  ├─ forward(qkv) → 调用融合算子                      │  │   │
│  │  │  │   mytest_rope_norm_store_kv_fp8()               │  │   │
│  │  │  │   (norm→rope→hardma→kv_update→quant)            │  │   │
│  │  │  └─ 返回量化后的 Q (FP8)                            │  │   │
│  │  └────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│  ┌──────────────────────────┐  ┌────────────────────────────┐   │
│  │  Quantization Registry   │  │  Attention Backend          │   │
│  │  ┌────────────────────┐  │  │  ┌──────────────────────┐   │   │
│  │  │ FAKQuant (INT8)    │  │  │  │ AscendAttentionImpl   │   │   │
│  │  │ FAKQuantFP8 (FP8)  │  │  │  │  ├─ reshape_and_cache │   │   │
│  │  └────────────────────┘  │  │  │  │    (use_npu_rope_   │   │   │
│  └──────────────────────────┘  │  │  │  │     norm → skip)  │   │   │
│                                │  │  ├─ forward_impl       │   │   │
│  ┌──────────────────────────┐  │  │  │    └─ _forward_fp8_ │   │   │
│  │  Device Operator         │  │  │  │       attention()   │   │   │
│  │  ├─ split_fp8_kv_...     │  │  │  └──────────────────────┘   │   │
│  │  ├─ fp8_per_head_...     │  │  └────────────────────────────┘   │
│  │  └─ reshape_and_cache_fp8│  │                                   │
│  └──────────────────────────┘  │  ┌────────────────────────────┐   │
│                                │  │  AscendAttentionBackend    │   │
│  ┌──────────────────────────┐  │  │  ├─ get_kv_cache_shape     │   │
│  │  KV Cache Layout         │  │  │  │   (FP8 padded head_size)│   │
│  │  逻辑: BBND              │  │  │  ├─ get_kv_cache_stride_   │   │
│  │  (2,B,BS,H,D_pad)        │  │  │  │   order (BNBD)          │   │
│  │  物理: BNBD              │  │  │  └────────────────────────┘   │
│  │  (2,B,H,BS_padded,D)     │  │  └────────────────────────────┘   │
│  └──────────────────────────┘  │                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        ACLNN Operators                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  mytest_rope_norm_store_kv_fp8 (Fused Operator)         │    │
│  │  qk-norm → qk-rope → qk-hardma → kv_update → quant(FP8) │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

#### 3.1.2 量化方案类层次

```
AscendAttentionScheme (基类 - base.py)
        │
        ├── AscendFAQuantAttentionMethod (INT8 - kv_c8.py)
        │       └── 注册: @register_scheme("FAKQuant", "attention")
        │
        └── AscendFAQuantFP8AttentionMethod (FP8 - kv_fp8.py)  ← 新增
                └── 注册: @register_scheme("FAKQuantFP8", "attention")
```

### 3.2 KV Cache Layout 设计

#### 3.2.1 Layout 规格

| 维度 | 逻辑形状 (BBND) | 物理形状 (BNBD) |
| :--- | :--- | :--- |
| 第0维 | `2` (K/V) | `2` (K/V) |
| 第1维 | `num_blocks` | `num_blocks` |
| 第2维 | `block_size` | `num_kv_heads` |
| 第3维 | `num_kv_heads` | `block_size_padded` |
| 第4维 | `head_size + padded_elems` | `head_size` |

**说明**：
- 逻辑形状 BBND：`(2, num_blocks, block_size, num_kv_heads, head_size_padded)`
- 物理形状 BNBD：`(2, num_blocks, num_kv_heads, block_size_padded, head_size)`
- 其中 `block_size_padded = pad_total_row = block_size * (head_size + padded_elems) // head_size`
- 通过 `as_strided` reshape 将 padding 从 head_size 维度转移到 block_size 维度，再通过 stride_order `(0, 1, 3, 2, 4)` 实现 BNBD 物理布局
- **NPU 侧 K/V 分离**：KV Cache 在 `_reshape_kv_cache_tensors` 中被拆分为独立的 key_cache 和 value_cache，每个为 4D 张量
- **FP8 路径直接保持 BNBD 物理布局**：不做 permute 回逻辑布局，key_cache/value_cache 直接以 `(num_blocks, num_kv_heads, block_size, head_size_padded)` 的 BNBD 物理形状存储
- 暂不考虑 cross-layer 功能（`include_num_layers_dimension=False`）

#### 3.2.2 FP8 Scale 嵌入设计

参考 vLLM GPU 侧实现，采用 **per-token-per-head scale 嵌入 head_size 维度**：

```
分配形状: (2, num_blocks, block_size, num_kv_heads, head_size + padded_elems)
                                    └──────────┬──────────┘
                                        pad_head_size

运行时 reshape: (2, num_blocks, pad_total_rows, num_kv_heads, head_size)
                  ├── [:block_size, :, :]  → KV 数据 (FP8 e4m3fn)
                  └── [block_size:, :, :]  → per-head FP32 scale (4 fp8 槽位/head)
```

**padded_elems 计算**：
- 每个 KV head 需要存储 1 个 FP32 scale = 4 字节 = 4 个 FP8 槽位
- 约束：`padded_elems * block_size` 必须能被 `head_size` 整除（保证 reshape 有效）
- 例如：block_size=64, head_size=128 → padded_elems=4（64*4=256, 256%128=0 ✓）
- 例如：block_size=128, head_size=128 → padded_elems=4（128*4=512, 512%128=0 ✓）

**pad_total_rows 计算**：
```
pad_total_rows = block_size * (head_size + padded_elems) // head_size
```
- 例如：block_size=64, head_size=128, padded_elems=4 → pad_total_rows = 64*132//128 = 66
  - 前 64 行：KV 数据
  - 后 2 行：per-head scale

**Layout 变换链**：
```
逻辑形状 BBND:  (2, num_blocks, block_size,        num_kv_heads, head_size_padded)
                      ↓ stride_order (0,1,3,2,4) permute
中间形状 BNBD:  (2, num_blocks, num_kv_heads, block_size,        head_size_padded)
                      ↓ as_strided reshape (padding 从 D 维转移到 BS 维)
物理形状 BNBD:  (2, num_blocks, num_kv_heads, block_size_padded, head_size)
```

**NPU FP8 路径 Layout 变换链**（K/V 分离，直接保持 BNBD 物理布局）：
```
原始分配: (2, num_blocks, block_size, num_kv_heads, head_size_padded)  — 逻辑 BBND
                ↓ _reshape_kv_cache_tensors 中拆分 K/V
K/V 分配: (num_blocks, block_size, num_kv_heads, head_size_padded)     — 逻辑 BBND
                ↓ stride_order (0,2,1,3) permute → 物理形状
K/V 物理: (num_blocks, num_kv_heads, block_size, head_size_padded)     — 物理 BNBD
                ↓ 不做 permute 回逻辑布局（FP8 路径特有）
最终存储: (num_blocks, num_kv_heads, block_size, head_size_padded)     — 物理 BNBD
                ↓ split_fp8_kv_cache_and_scale 中 as_strided reshape
K/V 数据: (num_blocks, num_kv_heads, block_size, head_size)            — BNBD (不含 scale)
K/V scale: (num_blocks, num_kv_heads, scale_rows, head_size)           — BNBD (per-head scale)
```

#### 3.2.3 kv_data 和 kv_scale 拆分逻辑

**GPU 5D 合并版**（`split_fp8_kv_data_and_scale`，处理 `(2, B, BS, H, D_pad)` 格式）：

```python
def _split_fp8_kv_data_and_scale(
    kv_cache_fp8: torch.Tensor,   # (2, num_blocks, block_size, num_kv_heads, pad_head_size)
    head_size: int,
    kv_cache_quant_config: KVCacheQuantConfig | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """将 FP8 KV Cache 拆分为 data 和 per-head scale 区域。
    
    Returns:
        kv_data:  (2, num_blocks, block_size, num_kv_heads, head_size)
        kv_scale: (2, num_blocks, scale_rows, num_kv_heads, head_size) or None
    """
    if _needs_per_head_scale_in_cache(kv_cache_quant_config):
        num_blocks = kv_cache_fp8.shape[1]
        block_size = kv_cache_fp8.shape[2]
        num_kv_heads = kv_cache_fp8.shape[3]
        pad_head_size = kv_cache_fp8.shape[4]
        pad_total_row = block_size * pad_head_size // head_size
        
        # 使用 as_strided 零拷贝 reshape
        # (2, B, BS, H, D_pad) → (2, B, pad_total_row, H, D)
        s = kv_cache_fp8.stride()
        kv_cache_fp8 = torch.as_strided(
            kv_cache_fp8,
            (2, num_blocks, pad_total_row, num_kv_heads, head_size),
            (s[0], s[1], num_kv_heads * head_size, head_size, 1),
        )
        kv_data = kv_cache_fp8[:, :, :block_size, :, :]
        kv_scale = kv_cache_fp8[:, :, block_size:, :, :]
    else:
        kv_data = kv_cache_fp8
        kv_scale = None
    
    return kv_data, kv_scale
```

**NPU 4D K/V 分离版**（`split_fp8_kv_cache_and_scale`，处理 `(B, H, BS, D_pad)` BNBD 格式）：

```python
@staticmethod
def split_fp8_kv_cache_and_scale(kv_cache_fp8, head_size, kv_cache_quant_config):
    """将 NPU FP8 KV Cache（4D BNBD 布局）拆分为 data 和 per-head scale 区域。
    
    输入: kv_cache_fp8 shape = (num_blocks, num_kv_heads, block_size, head_size_padded)
    输出:
        kv_data:  (num_blocks, num_kv_heads, block_size, head_size)
        kv_scale: (num_blocks, num_kv_heads, scale_rows, head_size) or None
    """
    if not BaseDeviceAdaptor.needs_per_head_scale_in_cache(kv_cache_quant_config):
        return kv_cache_fp8, None

    num_blocks = kv_cache_fp8.shape[0]
    num_kv_heads = kv_cache_fp8.shape[1]
    block_size = kv_cache_fp8.shape[2]
    pad_head_size = kv_cache_fp8.shape[3]

    # pad_total_row = block_size * pad_head_size // head_size
    # 将 D_pad 中的 scale 嵌入转移到 BS 维度
    pad_total_row = block_size * pad_head_size // head_size

    s = kv_cache_fp8.stride()
    # (B, H, BS, D_pad) → (B, H, pad_total_row, head_size)
    kv_cache_fp8 = torch.as_strided(
        kv_cache_fp8,
        (num_blocks, num_kv_heads, pad_total_row, head_size),
        (s[0], s[1], head_size, 1),
    )
    kv_data = kv_cache_fp8[:, :, :block_size, :]
    kv_scale = kv_cache_fp8[:, :, block_size:, :]
    return kv_data, kv_scale
```

**关键差异**：
- GPU 版处理 5D 合并张量 `(2, B, BS, H, D_pad)`，第 0 维合并 K/V
- NPU 版处理 4D 分离张量 `(B, H, BS, D_pad)`，K/V 各自独立调用
- NPU 版的 as_strided strides 中 `pad_total_row` 维步长为 `head_size`（而非 `num_kv_heads * head_size`），因为 BNBD 布局下每个 (B, H) 组合的 `(BS, D_pad)` 数据是连续的

### 3.3 目录结构

```
vllm_ascend/
├── ops/
│   └── rope_norm.py              # 新增：NpuRopeNorm 融合算子封装
├── patch/
│   └── worker/
│       ├── __init__.py           # 修改：导入 patch_hunyuan_v3
│       └── patch_hunyuan_v3.py   # 新增：Hunyuan V3 模型 NPU patch
├── quantization/
│   └── methods/
│       ├── __init__.py           # 修改：添加 kv_fp8 导入
│       ├── kv_c8.py              # 现有 INT8 实现（参考）
│       └── kv_fp8.py             # 新增：FP8 KV 量化方案实现
├── attention/
│   ├── attention_v1.py           # 修改：添加 FP8 KV Cache 支持
│   │   ├── AscendAttentionBackend.get_kv_cache_shape()  # FP8 padded head_size
│   │   ├── AscendAttentionBackend.get_kv_cache_page_size_padded()  # FP8 padded page size（调度器内存分配）
│   │   ├── AscendAttentionBackend.get_kv_cache_stride_order()  # BNBD
│   │   └── AscendAttentionBackendImpl:
│   │       ├── __init__()          # use_fp8_kv_cache + use_npu_rope_norm
│   │       ├── reshape_and_cache() # use_npu_rope_norm → skip
│   │       ├── forward_impl()      # use_npu_rope_norm → _forward_fp8_attention()
│   │       ├── _forward_fp8_attention()  # 新增：FP8 FA 分支
│   │       └── _get_kv_scales()    # 新增：根据 granularity 返回 k_scale/v_scale
│   └── utils.py
├── worker/
│   └── model_runner_v1.py        # 修改：KV Cache 初始化适配 FP8
│       └── _reshape_kv_cache_tensors()
│           ├── 传入 cache_dtype_str 触发 FP8 padded head_size
│           ├── 应用 stride_order permute 实现 BNBD 物理布局
│           └── FP8 dtype 强制 torch.float8_e4m3fn
└── device/
    └── device_op.py              # 修改：添加 FP8 KV Cache 操作
        ├── fp8_per_head_scale_elems_padded()
        ├── needs_per_head_scale_in_cache()
        ├── split_fp8_kv_data_and_scale()
        └── reshape_and_cache_fp8()
```

### 3.4 关键类与方法设计

#### 3.4.1 融合算子封装类（新增）

**文件名**: `vllm_ascend/ops/rope_norm.py`

| 类名 | 方法名 | 功能说明 |
| :--- | :--- | :--- |
| `NpuRopeNorm` | `__init__` | 初始化，解析 KV Cache 量化配置，注册 cos_sin_cache |
| | `support()` | 类方法：检查是否支持当前配置 (head_dim=128, head_per_group∈[4,8]) |
| | `forward(qkv, layer_name)` | 创建输出 buffer，调用 `_forward_impl` |
| | `_forward_impl(qkv, output)` | 核心：拆分 KV Cache → 获取 scale → 调用 `mytest_rope_norm_store_kv_fp8`（prefill/decode 分别处理） |
| | `_get_kv_scales(layer, kv_scale)` | 根据 granularity 返回 k_scale/v_scale |
| | `register_layer_name(name)` | 注册到全局 `_npu_rope_norm_instances` |
| | `process_weights_after_loading()` | 从 fallback norm 模块提取 norm 权重 |

#### 3.4.2 模型 Patch 类（新增）

**文件名**: `vllm_ascend/patch/worker/patch_hunyuan_v3.py`

| 函数名 | 功能说明 |
| :--- | :--- |
| `_patched_init(self, *args, **kwargs)` | monkey-patch `HYV3Attention.__init__`：创建 `NpuRopeNorm` 实例，设置 `attn.impl.use_npu_rope_norm = True` |
| `_patched_forward(self, positions, hidden_states)` | monkey-patch `HYV3Attention.forward`：当 `npu_rope_norm` 启用时，`qkv → npu_rope_norm(qkv) → 返回量化 Q`，跳过原有 q/k/v split + norm + rope |

#### 3.4.3 量化方案类（新增）

**文件名**: `vllm_ascend/quantization/methods/kv_fp8.py`

| 类名 | 方法名 | 功能说明 |
| :--- | :--- | :--- |
| `AscendFAQuantFP8AttentionMethod` | `__init__` | 初始化量化方案，读取 kv_lora_rank、qk_rope_head_dim |
| | `create_weights` | 创建 fa_q/fa_k/fa_v 的 scale 和 offset 参数 |
| | `process_weights_after_loading` | 处理 fak_descale、quant_kscale 等 |

#### 3.4.4 注意力后端类（修改）

**文件名**: `vllm_ascend/attention/attention_v1.py`

| 类名 | 方法名 | 修改内容 |
| :--- | :--- | :--- |
| `AscendAttentionBackend` | `get_kv_cache_shape` | FP8 时返回 padded head_size |
| | `get_kv_cache_stride_order` | 返回 BNBD stride_order `(0, 1, 3, 2, 4)` |
| | `get_kv_cache_page_size_padded` | FP8 时返回 padded page size |
| `AscendAttentionBackendImpl` | `__init__` | 添加 `use_fp8_kv_cache`、`use_npu_rope_norm`、`_kv_cache_quant_config`、`_quant_type` |
| | `reshape_and_cache` | `use_npu_rope_norm=True` 时跳过 KV 写入（融合算子已写入） |
| | `forward_impl` | `use_npu_rope_norm=True` 时走 `_forward_fp8_attention()` |
| | `_forward_fp8_attention` | **新增**：拆分 KV Cache → contiguous + flatten → 统一使用 `npu_fused_infer_attention_score`（A5 硬件 decode/prefill 统一路径） |
| | `_get_kv_scales` | 新增：根据 granularity 返回 k_scale/v_scale |

#### 3.4.5 设备算子封装（修改）

**文件名**: `vllm_ascend/device/device_op.py`

| 类名 | 方法名 | 功能说明 |
| :--- | :--- | :--- |
| `BaseDeviceAdaptor` | `fp8_per_head_scale_elems_padded` | 新增：计算 padded_elems |
| | `needs_per_head_scale_in_cache` | 新增：判断是否需要 scale 嵌入 |
| | `split_fp8_kv_data_and_scale` | 新增：as_strided 零拷贝拆分 KV Cache |
| | `reshape_and_cache_fp8` | 新增：调用 `mytest_rope_norm_store_kv_fp8`（备用路径） |

#### 3.4.6 模型运行器 KV Cache 初始化（修改）

**文件名**: `vllm_ascend/worker/model_runner_v1.py`

##### 3.4.6.1 问题分析

GPU 侧 `_reshape_kv_cache_tensors` 与 NPU 侧存在三个关键差异，导致 FP8 KV Cache 初始化不正确：

| 差异点 | GPU (`gpu_model_runner.py`) | NPU (`model_runner_v1.py`) | 影响 |
| :--- | :--- | :--- | :--- |
| **`cache_dtype_str` 传入** | `get_kv_cache_shape(..., cache_dtype_str=self.cache_config.cache_dtype)` | `get_kv_cache_shape(...)` 未传 `cache_dtype_str`（默认 `"auto"`） | FP8 padded head_size 永远不会触发，KV Cache 未分配 scale 嵌入空间 |
| **stride_order/permute** | 调用 `get_kv_cache_stride_order()` + `permute(*inv_order)` 实现 BNBD 物理布局 | 未使用 stride_order，直接 `.view(k_shape)` 保持 BBND 逻辑布局 | KV Cache 物理布局未转为 BNBD，与 `_forward_fp8_attention` 中期望的布局不一致 |
| **KV Cache 结构** | 单个 5D 张量 `(2, B, BS, H, D_pad)`，`kv_cache[0]` 即完整 KV Cache | K/V 分离的两个 4D 张量 `(B, BS, H, D)`，`kv_cache[0]` 是 key_cache | `_forward_fp8_attention` 中 `kv_cache[0].view(torch.float8_e4m3fn)` 语义不同 |
| **`page_size_padded` 内存分配** | `HpcAttentionBackend` 实现 `get_kv_cache_page_size_padded()`，调度器据此分配含 scale 嵌入空间的内存 | `AscendAttentionBackend` 未实现 `get_kv_cache_page_size_padded()`，`page_size_padded=None` | 调度器分配的内存不含 FP8 scale 嵌入空间，`_reshape_kv_cache_tensors` 中 `.view(k_shape)` 会因元素数不匹配而崩溃 |

##### 3.4.6.2 修改方案

**修改点 1：`get_kv_cache_shape` 调用时传入 `cache_dtype_str`**

```python
# 修改前（model_runner_v1.py L2925-L2930）
kv_cache_shape = attn_backend.get_kv_cache_shape(
    num_blocks,
    current_kv_cache_spec.block_size,
    current_kv_cache_spec.num_kv_heads,
    current_kv_cache_spec.head_size,
)

# 修改后
kv_cache_shape = attn_backend.get_kv_cache_shape(
    num_blocks,
    current_kv_cache_spec.block_size,
    current_kv_cache_spec.num_kv_heads,
    current_kv_cache_spec.head_size,
    cache_dtype_str=self.vllm_config.cache_config.cache_dtype,
)
```

**说明**：传入 `cache_dtype_str` 后，`AscendAttentionBackend.get_kv_cache_shape()` 中 `cache_dtype_str in ("fp8", "fp8_e4m3")` 的判断才能生效，返回 padded head_size 的形状。

**修改点 2：应用 stride_order permute 实现 BNBD 物理布局**

GPU 侧 `_reshape_kv_cache_tensors` 的核心逻辑：

```python
# GPU: gpu_model_runner.py L6635-L6658
kv_cache_shape = attn_backend.get_kv_cache_shape(
    kernel_num_blocks, kernel_block_size,
    kv_cache_spec.num_kv_heads, kv_cache_spec.head_size,
    cache_dtype_str=self.cache_config.cache_dtype,
)
# 获取 stride_order
kv_cache_stride_order = attn_backend.get_kv_cache_stride_order()
# 按 stride_order 重排 shape（逻辑 → 物理）
kv_cache_shape = tuple(kv_cache_shape[i] for i in kv_cache_stride_order)
# 计算逆序用于 permute
inv_order = [kv_cache_stride_order.index(i) for i in range(len(kv_cache_stride_order))]
# 分配并 permute 为物理布局
kv_caches[layer_name] = (
    kv_cache_raw_tensors[layer_name]
    .view(dtype)
    .view(kv_cache_shape)
    .permute(*inv_order)
)
```

NPU 侧需要适配此逻辑。但 NPU 侧 KV Cache 结构为 K/V 分离的两个 4D 张量（而非 GPU 的单个 5D 张量），因此需要分别处理：

```python
# NPU 适配：model_runner_v1.py _reshape_kv_cache_tensors

# 1. 获取带 FP8 padding 的 KV Cache 形状（含 cache_dtype_str）
kv_cache_shape = attn_backend.get_kv_cache_shape(
    num_blocks,
    current_kv_cache_spec.block_size,
    current_kv_cache_spec.num_kv_heads,
    current_kv_cache_spec.head_size,
    cache_dtype_str=self.vllm_config.cache_config.cache_dtype,
)

# 2. 获取 stride_order 用于 BNBD 物理布局
try:
    kv_cache_stride_order = attn_backend.get_kv_cache_stride_order()
    # stride_order = (0, 1, 3, 2, 4)
    # 逻辑 BBND: (2, B, BS, H, D_pad)
    # 物理 BNBD: (2, B, H, BS_padded, D)
except (AttributeError, NotImplementedError):
    kv_cache_stride_order = tuple(range(len(kv_cache_shape)))

# 3. 计算逆序
inv_order = [kv_cache_stride_order.index(i)
             for i in range(len(kv_cache_stride_order))]

# 4. 按 stride_order 重排 shape
physical_shape = tuple(kv_cache_shape[i] for i in kv_cache_stride_order)
# physical_shape = (2, B, H, BS_padded, D)

# 5. K/V 分离处理
# 当前 NPU 侧 k_shape = kv_cache_shape[1:] 即 (B, BS, H, D_pad)
# 需要改为物理布局的 K/V 形状
# physical_shape = (2, B, H, BS_padded, D)
# k_shape_physical = physical_shape[1:] = (B, H, BS_padded, D)
# 但 .view() 后需要 permute 回逻辑布局供后续使用
# 或者：保持当前 K/V 分离结构，但确保 shape 包含 FP8 padding

# 简化方案：由于 NPU 侧 K/V 已分离，直接对 k_shape/v_shape 应用 stride_order
# 逻辑 k_shape: (B, BS, H, D_pad)
# 对应 5D 逻辑的 [1:] 即索引 1,2,3,4
# stride_order[1:] = (1, 3, 2, 4) → 但这是相对于 5D 的
# 对于 4D K/V: 索引映射 1→0, 2→1, 3→2, 4→3
# stride_order[1:] = (1, 3, 2, 4) → 4D 映射: (0, 2, 1, 3)
# 即 (B, H, BS_padded, D)

k_shape_4d = kv_cache_shape[1:]  # (B, BS, H, D_pad) 逻辑
# 对 4D 应用 stride_order: 5D 索引 (1,2,3,4) → stride_order 后 (1,3,2,4)
# 4D 映射: 索引 0→1, 1→3, 2→2, 3→4 即 (0, 2, 1, 3)
k_stride_order_4d = tuple(i - 1 for i in kv_cache_stride_order[1:])
# (0, 2, 1, 3) 即 (B, H, BS_padded, D)
k_shape_physical = tuple(k_shape_4d[i] for i in k_stride_order_4d)
# (B, H, BS, D_pad) — 物理 BNBD 布局

# FP8 路径：直接保持 BNBD 物理布局，不做 permute 回逻辑布局
# 这样 _forward_fp8_attention 和 NpuRopeNorm 可以直接处理 BNBD 布局的数据
k_cache = (raw_k_tensor.view(k_cache_dtype)
           .view(k_shape_physical))
# k_cache shape = (B, H, BS, D_pad) — 物理 BNBD

v_cache = (raw_v_tensor.view(v_cache_dtype)
           .view(v_shape_physical))
# v_cache shape = (B, H, BS, D_pad) — 物理 BNBD
```

**修改点 3：KV Cache dtype 处理**

FP8 场景下，`current_kv_cache_spec.dtype` 可能不是 `torch.float8_e4m3fn`（取决于 `kv_cache_dtype` 配置），需要根据 `cache_dtype_str` 确定正确的 dtype：

```python
k_cache_dtype = v_cache_dtype = current_kv_cache_spec.dtype
cache_dtype_str = self.vllm_config.cache_config.cache_dtype
if cache_dtype_str in ("fp8", "fp8_e4m3"):
    k_cache_dtype = v_cache_dtype = torch.float8_e4m3fn
elif self.is_kv_consumer and enable_fa_quant(self.vllm_config):
    k_cache_dtype, v_cache_dtype = self.vllm_config.quant_config.get_kv_quant_dtype(
        layer_name, current_kv_cache_spec.dtype, self.model_config
    )
```

**修改点 4：`AscendAttentionBackend` 添加 `get_kv_cache_page_size_padded` 方法**

这是最关键的修改。没有此方法时，`Attention.get_kv_cache_spec()` 中 `page_size_padded=None`，调度器使用 `real_page_size_bytes`（不含 FP8 scale 嵌入空间）计算 `kv_cache_tensor.size`，导致分配的内存不足以容纳 padded head_size。

GPU 侧 `HpcAttentionBackend` 的参考实现：

```python
# GPU: hpc_attn.py L593-L628
@staticmethod
def get_kv_cache_page_size_padded(
    block_size, num_kv_heads, head_size, head_size_v, dtype,
    cache_dtype_str="auto",
) -> int | None:
    if cache_dtype_str not in ("fp8", "fp8_e4m3"):
        return None
    elem_size = get_dtype_size(dtype)
    padded = _fp8_per_head_scale_elems_padded(block_size, num_kv_heads, head_size)
    effective_head_size = head_size + padded
    effective_head_size_v = head_size_v + padded
    return block_size * num_kv_heads * (effective_head_size + effective_head_size_v) * elem_size
```

NPU 侧在 `AscendAttentionBackend` 中新增相同逻辑：

```python
# NPU: attention_v1.py AscendAttentionBackend
@staticmethod
def get_kv_cache_page_size_padded(
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    head_size_v: int,
    dtype: torch.dtype,
    cache_dtype_str: str = "auto",
) -> int | None:
    if cache_dtype_str not in ("fp8", "fp8_e4m3"):
        return None
    from vllm.utils.torch_utils import get_dtype_size

    assert head_size == head_size_v, "head_size must equal to head_size_v"
    elem_size = get_dtype_size(dtype)
    padded = AscendAttentionBackend._fp8_per_head_scale_elems_padded(
        block_size, num_kv_heads, head_size)
    effective_head_size = head_size + padded
    effective_head_size_v = head_size_v + padded
    return (
        block_size
        * num_kv_heads
        * (effective_head_size + effective_head_size_v)
        * elem_size
    )
```

**调用链**：

```
1. Attention.get_kv_cache_spec(vllm_config)
   → hasattr(attn_backend, "get_kv_cache_page_size_padded") → True
   → page_size_padded = attn_backend.get_kv_cache_page_size_padded(...)
   → FullAttentionSpec(page_size_padded=page_size_padded)

2. AttentionSpec.page_size_bytes
   → page_size_padded 不为 None → 返回 padded 值（含 scale 嵌入空间）

3. 调度器: kv_cache_tensor.size = page_size_bytes * num_blocks
   → 分配的内存足够容纳 FP8 padded head_size

4. _reshape_kv_cache_tensors
   → get_kv_cache_shape(cache_dtype_str="fp8") 返回 padded shape
   → raw_k_tensor.view(fp8).view(k_shape) 元素数匹配 ✅
```

##### 3.4.6.3 修改前后对比

| 步骤 | 修改前 | 修改后 |
| :--- | :--- | :--- |
| 内存分配大小 | `page_size_padded=None`，调度器用 `real_page_size_bytes`（不含 scale 嵌入空间） | `AscendAttentionBackend.get_kv_cache_page_size_padded()` 返回 padded 值，调度器分配含 scale 嵌入空间的内存 |
| 获取 KV Cache shape | `get_kv_cache_shape(B, BS, H, D)` 默认 `cache_dtype_str="auto"` | `get_kv_cache_shape(B, BS, H, D, cache_dtype_str=...)` 触发 FP8 padded head_size |
| 物理布局 | 无 stride_order，直接 `.view(k_shape)` | 应用 `get_kv_cache_stride_order()` + `permute` 实现 BNBD |
| dtype | `current_kv_cache_spec.dtype` | FP8 时强制 `torch.float8_e4m3fn` |
| 最终 KV Cache 形状 | `(B, BS, H, D)` 逻辑 BBND | `(B, H, BS, D_pad)` 物理 BNBD（FP8 路径不做 permute 回逻辑布局） |

##### 3.4.6.4 与 `_forward_fp8_attention` 的衔接

修改后 KV Cache 初始化时：
- **FP8 路径**：直接保持 BNBD 物理布局 `(B, H, BS, D_pad)`，不做 permute 回逻辑布局
- **非 FP8 路径**：保持原有逻辑布局 `(B, BS, H, D)`

`_forward_fp8_attention` 中：
- `kv_cache[0]` 是 key_cache，形状 `(B, H, BS, D_pad)` BNBD 物理布局，dtype `torch.float8_e4m3fn`
- `kv_cache[1]` 是 value_cache，形状 `(B, H, BS, D_pad)` BNBD 物理布局，dtype `torch.float8_e4m3fn`
- `split_fp8_kv_cache_and_scale()` 分别对 K/V 调用，将 `(B, H, BS, D_pad)` 拆分为：
  - `kv_data: (B, H, BS, D)` — BNBD 布局的纯数据
  - `kv_scale: (B, H, scale_rows, D)` — BNBD 布局的 per-head scale
- `contiguous()` 确保内存连续后，flatten 为 `(B, H, BS*D)` 传给 FA 算子
- `input_layout="BND"` — 匹配 BNBD 布局中 `(B, H, D)` 的语义
- `block_size=BS` — 传递实际的 block_size 给 FA 算子

**PrefillNoCache 场景**：
- `k_flat = k_flat[:, :, :num_tokens * head_size].view(num_tokens, num_kv_heads, head_size)`
- 在 BNBD 布局下，`k_flat` 的第 2 维是 `BS*D`，前 `num_tokens * head_size` 个元素对应前 `num_tokens` 个 token 的 K 数据

#### 3.5.1 Attention 层量化参数

| 参数名 | 类型 | 形状 | 说明 |
| :--- | :--- | :--- | :--- |
| `fa_q.scale` | `torch.Tensor` | `[num_heads, 1]` | Q 量化 scale（per-head） |
| `fa_k.scale` | `torch.Tensor` | `[num_kv_heads, 1]` | K 量化 scale（per-head） |
| `fa_v.scale` | `torch.Tensor` | `[num_kv_heads, 1]` | V 量化 scale（per-head，静态） |
| `fak_descale_float` | `torch.Tensor` | `[1, num_kv_heads]` | K 反量化 scale（float32） |
| `fak_descale` | `torch.Tensor` | `[1, num_kv_heads]` | K 反量化 scale（原始 dtype） |
| `fak_descale_reciprocal` | `torch.Tensor` | `[1, num_kv_heads]` | K 反量化 scale 倒数 |
| `quant_kscale` | `torch.Tensor` | `[1, kv_lora_rank]` | KV LoRA 量化 scale（MLA 场景） |
| `_k_scale` | `torch.Tensor` | `[1]` 或 `[num_kv_heads]` | K scale（运行时使用） |
| `_v_scale` | `torch.Tensor` | `[num_kv_heads]` | V scale（运行时使用，per-head 静态） |
| `_kv_cache_quant_config` | `KVCacheQuantConfig` | - | KV Cache 量化配置（含 granularity） |

#### 3.5.2 KV Cache 数据结构

| 组件 | 类型 | 逻辑形状 | 说明 |
| :--- | :--- | :--- | :--- |
| `kv_cache` | `torch.Tensor` (FP8) | `(2, num_blocks, block_size, num_kv_heads, pad_head_size)` | 完整 KV Cache |
| `kv_data` | `torch.Tensor` (FP8) | `(2, num_blocks, block_size, num_kv_heads, head_size)` | 拆分后的 KV 数据 |
| `kv_scale` | `torch.Tensor` (FP32) | `(2, num_blocks, scale_rows, num_kv_heads, head_size)` | 拆分后的 per-head scale |

### 3.6 API 接口设计

#### 3.6.1 NpuRopeNorm 融合算子接口（模型层调用）

```python
class NpuRopeNorm:
    """NPU 融合 RoPE + QK-Norm + KV-Cache-Write + FP8 Q Quant。
    
    等价于 GPU 的 HpcRopeNorm，在模型层 (HYV3Attention) 中调用。
    融合算子 mytest_rope_norm_store_kv_fp8 完成:
      qk-norm → qk-rope → qk-hardma → kv_update → quant(FP8)
    """
    
    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        cos_sin_cache: torch.Tensor,
        use_qk_norm: bool,
        fallback_qnorm: torch.nn.Module | None,
        fallback_knorm: torch.nn.Module | None,
        kv_cache_dtype: str,
        qk_norm_policy: int = 1,
        enable_hadamard: bool = False,
    ) -> None: ...
    
    @classmethod
    def support(
        cls,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        kv_cache_dtype: str,
    ) -> bool:
        """检查是否支持当前配置。head_dim=128, head_per_group∈[4,8]"""
    
    def forward(self, qkv: torch.Tensor, layer_name: str) -> torch.Tensor:
        """前向传播。
        
        Args:
            qkv: [num_tokens, q_size + 2*kv_size] 拼接的 QKV
            layer_name: 层名称，用于从 ForwardContext 获取 attn_metadata
        
        Returns:
            量化后的 Q: [num_tokens, num_heads, head_dim] (FP8 或原始 dtype)
        """
    
    def _forward_impl(self, qkv, output):
        """核心实现：
        1. 从 ForwardContext 获取 attn_metadata 和 attn_layer
        2. 拆分 KV Cache 为 data 和 scale
        3. Prefill: mytest_rope_norm_store_kv_fp8(is_prefill=True)
        4. Decode:  mytest_rope_norm_store_kv_fp8(is_prefill=False)
        """
```

#### 3.6.2 模型 Patch 接口

```python
# patch_hunyuan_v3.py - monkey-patch HYV3Attention

def _patched_init(self, *args, **kwargs):
    """在原有 __init__ 后创建 NpuRopeNorm 实例。
    
    当 use_fp8=True 时：
      - self.attn.query_quant = None
      - self.attn.impl.use_npu_rope_norm = True
    """

def _patched_forward(self, positions, hidden_states) -> torch.Tensor:
    """替换原有 forward。
    
    当 npu_rope_norm 启用时：
      qkv, _ = self.qkv_proj(hidden_states)
      q = self.npu_rope_norm(qkv, self.attn.layer_name)  # 融合算子
      k = torch.empty(0, ...)  # KV 已由融合算子写入
      v = torch.empty(0, ...)
      attn_output = self.attn(q, k, v)
    
    当 npu_rope_norm 未启用时：走原有 q/k/v split + norm + rope 流程
    """
```

#### 3.6.3 量化方案注册接口

```python
@register_scheme("FAKQuantFP8", "attention")
class AscendFAQuantFP8AttentionMethod:
    """FP8 KV Cache 量化方案（用于 Hunyuan V3 模型）
    
    量化策略:
    - Q/K/V 权重: MXFP8 (由 W8A8_MXFP8 方案处理)
    - Q/K 激活值: FP8 动态量化 (per-token-per-head)
    - V 激活值: FP8 静态量化 (per-head)
    """
    
    def __init__(self):
        self.transpose_weight = True
        vllm_config = get_current_vllm_config()
        config = vllm_config.model_config.hf_config
        self.kv_lora_rank = getattr(config, "kv_lora_rank", 0)
        self.qk_rope_head_dim = getattr(config, "qk_rope_head_dim", 0)
    
    def create_weights(self, layer: torch.nn.Module) -> None:
        """创建 FP8 量化参数模块。
        
        创建 fa_q, fa_k, fa_v 三个子模块，每个包含 scale 参数。
        scale 使用 float32 存储以保证精度。
        """
        extra_module_names = ["fa_q", "fa_k", "fa_v"]
        for name in extra_module_names:
            setattr(layer, name, torch.nn.Module())
        
        params_dict = {}
        dtype = torch.float32
        
        params_dict["fa_q.scale"] = torch.empty(
            (layer.num_heads, 1), dtype=dtype)
        params_dict["fa_k.scale"] = torch.empty(
            (layer.num_kv_heads, 1), dtype=dtype)
        params_dict["fa_v.scale"] = torch.empty(
            (layer.num_kv_heads, 1), dtype=dtype)
        
        for name, weight in params_dict.items():
            module_name, weight_name = name.rsplit(".", 1)
            module = getattr(layer, module_name)
            weight_param = torch.nn.Parameter(weight, requires_grad=False)
            module.register_parameter(weight_name, weight_param)
            weight_param.weight_loader = weight_loader
    
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """权重加载后处理。
        
        1. 处理 K 量化 scale → fak_descale / fak_descale_reciprocal
        2. 处理 V 量化 scale → _v_scale (per-head 静态)
        3. 处理 KV LoRA scale → quant_kscale
        """
        # K scale 处理
        fa_k_scale = torch.squeeze(layer.fa_k.scale).unsqueeze(0)
        layer.fak_descale_float = torch.nn.Parameter(
            fa_k_scale.to(torch.float32), requires_grad=False)
        layer.fak_descale = torch.nn.Parameter(
            fa_k_scale, requires_grad=False)
        layer.fak_descale_reciprocal = 1.0 / layer.fak_descale
        
        # V scale 处理 (per-head 静态)
        fa_v_scale = torch.squeeze(layer.fa_v.scale)
        layer._v_scale = torch.nn.Parameter(
            fa_v_scale, requires_grad=False)
        
        # KV LoRA scale
        if self.kv_lora_rank > 0:
            repeated_quant_kscale = fa_k_scale.repeat(self.kv_lora_rank)
            layer.quant_kscale = repeated_quant_kscale.view(
                1, self.kv_lora_rank)
            layer.quant_kscale = 1.0 / torch.nn.Parameter(
                layer.quant_kscale.to(torch.float32), requires_grad=False)
```

#### 3.6.4 KV Cache Shape 接口

```python
class AscendAttentionBackend(AttentionBackend):
    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        """返回 KV Cache 形状。
        
        FP8 时 head_size 需要 padding 以嵌入 per-head scale。
        """
        if cache_dtype_str in ("fp8", "fp8_e4m3") and _needs_per_head_scale_in_cache():
            padded = _fp8_per_head_scale_elems_padded(
                block_size, num_kv_heads, head_size)
            effective_head_size = head_size + padded
            # 逻辑形状 BBND
            return (2, num_blocks, block_size, num_kv_heads, effective_head_size)
        return (2, num_blocks, block_size, num_kv_heads, head_size)
    
    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        """返回 KV Cache stride order。
        
        逻辑 BBND → 物理 BNBD: permute(0, 1, 3, 2, 4)
        逻辑 (2, B, BS, H, D_pad) → 物理 (2, B, H, BS, D_pad)
        再通过 as_strided reshape 将 padding 从 D 维转移到 BS 维:
        物理最终 (2, B, H, BS_padded, D)
        暂不考虑 cross-layer。
        """
        # BNBD 物理布局
        return (0, 1, 3, 2, 4)
```

#### 3.6.5 注意力前向接口

```python
class AscendAttentionBackendImpl(AttentionImpl):
    def __init__(self, ...):
        # ... 原有初始化 ...
        self.kv_cache_dtype = kv_cache_dtype
        self.use_fp8_kv_cache = kv_cache_dtype in ("fp8", "fp8_e4m3")
        self.use_npu_rope_norm = False  # 由模型 patch 设置为 True
        
        if self.use_fp8_kv_cache:
            self._kv_cache_quant_config = self._resolve_kv_cache_quant_config()
            self._quant_type = self._resolve_quant_type()
    
    def reshape_and_cache(self, query, key, value, kv_cache, attn_metadata, output):
        if len(kv_cache) > 1:
            # ... 原有初始化 ...
            
            if self.use_npu_rope_norm:
                pass  # KV 已由融合算子写入，跳过
            elif self.use_fp8_kv_cache:
                DeviceOperator.reshape_and_cache_fp8(...)
            else:
                DeviceOperator.reshape_and_cache(...)
        
        return query, key, value, output
    
    def forward_impl(self, query, key, value, kv_cache, attn_metadata, output):
        num_tokens = query.shape[0]
        if self.use_npu_rope_norm and self.use_fp8_kv_cache:
            output = self._forward_fp8_attention(query, kv_cache, attn_metadata, output)
        elif (...):
            output = self.forward_paged_attention(query, attn_metadata, output)
        else:
            output = self.forward_fused_infer_attention(query, key, value, attn_metadata, output)
        return output
    
    def _forward_fp8_attention(self, query, kv_cache, attn_metadata, output):
        """FP8 attention 分支：拆分 KV Cache，统一使用 FA 算子。
        
        query 已由 NpuRopeNorm 量化为 FP8。
        
        关键设计（A5 硬件场景）：
        - Decode 和 Prefill 统一使用 npu_fused_infer_attention_score
        - KV Cache 为 BNBD 物理布局 (B, H, BS, D_pad)，不做 permute 回逻辑布局
        - split_fp8_kv_cache_and_scale 分别对 K/V 调用，拆分 data 和 scale
        - contiguous 后 flatten 为 (num_blocks, num_kv_heads, block_size * head_size)
        - 配合 block_table 传给 FA 算子，input_layout="BND"
        """
        key_cache_fp8 = (
            kv_cache[0].view(torch.float8_e4m3fn)
            if kv_cache[0].dtype != torch.float8_e4m3fn
            else kv_cache[0]
        )
        value_cache_fp8 = (
            kv_cache[1].view(torch.float8_e4m3fn)
            if kv_cache[1].dtype != torch.float8_e4m3fn
            else kv_cache[1]
        )

        k_data, k_scale = DeviceOperator.split_fp8_kv_cache_and_scale(
            key_cache_fp8, self.head_size, self._kv_cache_quant_config)
        v_data, v_scale = DeviceOperator.split_fp8_kv_cache_and_scale(
            value_cache_fp8, self.head_size, self._kv_cache_quant_config)

        k_contig = k_data.contiguous()
        v_contig = v_data.contiguous()

        num_blocks = k_contig.shape[0]
        num_kv_heads = k_contig.shape[1]
        block_size = k_contig.shape[2]
        head_size = k_contig.shape[3]

        k_flat = k_contig.view(num_blocks, num_kv_heads, block_size * head_size)
        v_flat = v_contig.view(num_blocks, num_kv_heads, block_size * head_size)

        num_tokens = query.shape[0]
        if attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
            # PrefillNoCache: 无 block_table，KV 数据连续存储
            # BNBD 布局下 k_flat shape = (B, H, BS*D)
            # 前 num_tokens * head_size 个元素对应前 num_tokens 个 token
            block_table = None
            actual_seq_lengths_kv = attn_metadata.actual_seq_lengths_q
            k_flat = k_flat[:, :, :num_tokens * head_size].view(
                num_tokens, num_kv_heads, head_size)
            v_flat = v_flat[:, :, :num_tokens * head_size].view(
                num_tokens, num_kv_heads, head_size)
        else:
            # DecodeOnly / PrefillCacheHit / ChunkedPrefill / SpecDecoding:
            # 有 block_table，FA 算子通过 block_table 进行 block 寻址
            block_table = attn_metadata.block_tables
            actual_seq_lengths_kv = attn_metadata.seq_lens_list

        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            query=query,
            key=k_flat,
            value=v_flat,
            atten_mask=attn_metadata.attn_mask,
            block_table=block_table,
            input_layout="BND",
            block_size=block_size,
            actual_seq_lengths=attn_metadata.actual_seq_lengths_q,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=self.scale,
            sparse_mode=3,
        )
        attn_output = attn_output.view(num_tokens, self.num_heads, self.head_size)
        output[:num_tokens] = attn_output[:num_tokens]
        return output
```

### 3.7 主业务流程与调用链

#### 3.7.1 初始化流程

```
用户配置 (quantization="fp8", kv_cache_dtype="fp8_e4m3")
            │
            ▼
    vLLM 配置解析 → KVCacheQuantConfig(k_quant=per_token_per_head, v_quant=per_head)
            │
            ▼
    QuantParser 解析 → get_scheme_class("FAKQuantFP8", "attention")
            │
            ▼
    AscendFAQuantFP8AttentionMethod.__init__()
            │
            ▼
    create_weights(layer) → 创建 fa_q/fa_k/fa_v 参数
            │
            ▼
    process_weights_after_loading(layer) → 处理 fak_descale, _v_scale, quant_kscale
            │
            ▼
    AscendAttentionBackend.get_kv_cache_shape()
      → (2, num_blocks, block_size, num_kv_heads, head_size + padded_elems)
            │
            ▼
    AscendAttentionBackendImpl.__init__()
      → use_fp8_kv_cache = True
      → use_npu_rope_norm = False (初始值)
      → _kv_cache_quant_config = KVCacheQuantConfig(...)
      → _quant_type = ...
            │
            ▼
    [KV Cache Spec] Attention.get_kv_cache_spec()
      → get_kv_cache_page_size_padded() → padded page size  # 调度器据此分配含 scale 嵌入空间的内存
      → FullAttentionSpec(page_size_padded=padded_page_size)
            │
            ▼
    [KV Cache 分配] model_runner_v1._allocate_kv_cache_tensors()
      → kv_cache_tensor.size = page_size_bytes * num_blocks  # page_size_bytes 已含 FP8 padded
      → k_tensor, v_tensor = torch.zeros(..., dtype=torch.int8)
            │
            ▼
    [KV Cache Reshape] model_runner_v1._reshape_kv_cache_tensors()
      → get_kv_cache_shape(..., cache_dtype_str="fp8_e4m3")  # 传入 cache_dtype_str
      → get_kv_cache_stride_order() → (0, 1, 3, 2, 4)       # BNBD 物理布局
      → raw_tensor.view(torch.float8_e4m3fn)                 # FP8 dtype
      → .view(physical_shape).permute(inv_order)             # 物理布局 + permute 回逻辑
      → KV Cache 逻辑形状 (B, BS, H, D_pad)，底层 BNBD 物理布局
            │
            ▼
    [模型加载时] patch_hunyuan_v3._patched_init()
      → NpuRopeNorm.__init__()  # 创建融合算子实例
      → NpuRopeNorm.support()   # 检查配置兼容性
      → attn.impl.use_npu_rope_norm = True  # 启用融合算子路径
      → attn.query_quant = None  # 禁用原有 Q 量化
            │
            ▼
    量化参数注册完成，KV Cache 分配完成（含 FP8 scale 嵌入空间，page_size_padded 确保内存充足），融合算子就绪
```

#### 3.7.2 前向传播流程

```
┌────────────────────────────────────────────────────────────────────┐
│              HYV3Attention.forward() [patched]                      │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  qkv, _ = self.qkv_proj(hidden_states)                      │  │
│  │                                                              │  │
│  │  if self.npu_rope_norm is not None:                          │  │
│  │      q = self.npu_rope_norm(qkv, layer_name)  ← 融合算子     │  │
│  │      │                                                       │  │
│  │      │  NpuRopeNorm._forward_impl():                         │  │
│  │      │    1. 从 ForwardContext 获取 attn_metadata, kv_cache  │  │
│  │      │    2. split_fp8_kv_data_and_scale() → kv_data/scale  │  │
│  │      │    3. _get_kv_scales() → k_scale, v_scale            │  │
│  │      │    4. mytest_rope_norm_store_kv_fp8()                 │  │
│  │      │       (norm→rope→hardma→kv_update→quant)             │  │
│  │      │    5. 返回量化后的 Q (FP8)                            │  │
│  │      │                                                       │  │
│  │      k = empty, v = empty  # KV 已写入                       │  │
│  │  else:                                                       │  │
│  │      q, k, v = split(qkv) → norm → rope  # 原有流程          │  │
│  │                                                              │  │
│  │  attn_output = self.attn(q, k, v)                            │  │
│  └─────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────┐
│                  AscendAttentionImpl.forward()                      │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  reshape_and_cache():                                        │  │
│  │    use_npu_rope_norm ? → pass (skip, KV 已由融合算子写入)     │  │
│  │                                                              │  │
│  │  forward_impl():                                             │  │
│  │    use_npu_rope_norm && use_fp8_kv_cache ?                   │  │
│  │      → _forward_fp8_attention()                              │  │
│  │          1. split_fp8_kv_data_and_scale()                    │  │
│  │          2. contiguous + flatten KV data                     │  │
│  │          3. npu_fused_infer_attention_score()                │  │
│  │             (A5 硬件 decode/prefill 统一路径)                 │  │
│  └─────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
```

#### 3.7.3 调用链详情

| 步骤 | 调用方 | 被调用方 | 文件路径 |
| :--- | :--- | :--- | :--- |
| 1 | vLLM Core | `QuantParser.parse()` | `vllm_ascend/quantization/quant_parser.py` |
| 2 | QuantParser | `get_scheme_class("FAKQuantFP8", "attention")` | `vllm_ascend/quantization/methods/registry.py` |
| 3 | - | `AscendFAQuantFP8AttentionMethod.__init__()` | `vllm_ascend/quantization/methods/kv_fp8.py` |
| 4 | AttentionLayer | `create_weights()` | `vllm_ascend/quantization/methods/kv_fp8.py` |
| 5 | WeightLoader | `process_weights_after_loading()` | `vllm_ascend/quantization/methods/kv_fp8.py` |
| 6 | Attention | `get_kv_cache_spec()` → `get_kv_cache_page_size_padded()` | `vllm_ascend/attention/attention_v1.py` |
| 7 | ModelRunner | `_reshape_kv_cache_tensors()` → `get_kv_cache_shape(cache_dtype_str=...)` | `vllm_ascend/worker/model_runner_v1.py` |
| 8 | ModelRunner | `_reshape_kv_cache_tensors()` → `get_kv_cache_stride_order()` + FP8 路径不做 permute | `vllm_ascend/worker/model_runner_v1.py` |
| 9 | patch_hunyuan_v3 | `NpuRopeNorm.__init__()` | `vllm_ascend/ops/rope_norm.py` |
| 10 | patch_hunyuan_v3 | `attn.impl.use_npu_rope_norm = True` | `vllm_ascend/patch/worker/patch_hunyuan_v3.py` |
| 11 | HYV3Attention.forward | `NpuRopeNorm.forward(qkv, layer_name)` | `vllm_ascend/ops/rope_norm.py` |
| 12 | NpuRopeNorm._forward_impl | `DeviceOperator.split_fp8_kv_cache_and_scale()` | `vllm_ascend/device/device_op.py` |
| 13 | NpuRopeNorm._forward_impl | `self._get_kv_scales()` | `vllm_ascend/ops/rope_norm.py` |
| 14 | NpuRopeNorm._forward_impl | `torch_npu.mytest_rope_norm_store_kv_fp8()` | ACLNN 算子 |
| 15 | AscendAttentionImpl.forward | `reshape_and_cache()` → skip | `vllm_ascend/attention/attention_v1.py` |
| 16 | AscendAttentionImpl.forward | `_forward_fp8_attention()` | `vllm_ascend/attention/attention_v1.py` |
| 17 | _forward_fp8_attention | `DeviceOperator.split_fp8_kv_cache_and_scale()` | `vllm_ascend/device/device_op.py` |
| 18 | _forward_fp8_attention | `torch_npu.npu_fused_infer_attention_score(input_layout="BND")` | ACLNN 算子 |

---

## 4. 部署与集成方案

### 4.1 依赖与环境

| 依赖项 | 版本要求 | 说明 |
| :--- | :--- | :--- |
| PyTorch | >= 2.1.0 | 深度学习框架 |
| torch-npu | >= 2.1.0 | Ascend NPU 支持 |
| vLLM | == 0.18.0 | vLLM 核心 |
| ACLNN | >= 6.3.0 | Ascend 算子库 (含 FP8 支持) |

### 4.2 配置与运行

#### 4.2.1 量化配置示例

```python
from vllm import LLM

llm = LLM(
    model="Hunyuan/Hunyuan-V3-8B",
    quantization="fp8",           # Q/K/V 权重使用 MXFP8
    kv_cache_dtype="fp8_e4m3",    # KV Cache 使用 FP8
    device="npu",
)
```

#### 4.2.2 KVCacheQuantConfig 配置

```python
from vllm.config.cache import KVCacheQuantConfig, KVQuantSpec

kv_cache_quant_config = KVCacheQuantConfig(
    k_quant=KVQuantSpec(
        dtype="fp8_e4m3",
        granularity="per_token_per_head",  # K 动态量化
    ),
    v_quant=KVQuantSpec(
        dtype="fp8_e4m3",
        granularity="per_head",            # V 静态量化
    ),
)
```

#### 4.2.3 环境变量

| 环境变量 | 说明 | 默认值 |
| :--- | :--- | :--- |
| `VLLM_ASCEND_ENABLE_FP8_KV_CACHE` | 是否启用 FP8 KV Cache | `auto` |
| `VLLM_ASCEND_FP8_KV_CACHE_FUSION` | 是否使用融合算子 | `true` |

### 4.3 集成方式

1. **量化方案自动注册**：通过 `@register_scheme("FAKQuantFP8", "attention")` 自动注册
2. **配置自动识别**：`KVCacheQuantConfig` 解析 `kv_cache_dtype="fp8_e4m3"` 时自动选择 FP8 方案
3. **运行时自动选择**：`AscendAttentionBackendImpl` 根据 `use_fp8_kv_cache` 自动选择 FP8 处理分支
4. **KV Cache 形状自动适配**：`get_kv_cache_shape()` 根据是否 FP8 自动返回 padded head_size

---

## 5. 代码安全性

### 5.1 风险点识别

| 风险点 | 风险等级 | 关联模块 | 说明 |
| :--- | :--- | :--- | :--- |
| 量化精度损失 | 中 | `kv_fp8.py` | FP8 per-token-per-head 量化可能引入精度损失 |
| 数据类型不匹配 | 高 | `attention_v1.py` | FP8 (e4m3fn) 与 BF16/FP32 混合计算 |
| 算子兼容性 | 中 | `device_op.py` | 依赖 `mytest_rope_norm_store_kv_fp8` 算子 |
| KV Cache reshape 越界 | 高 | `device_op.py` | as_strided reshape 参数计算错误 |
| Scale 嵌入对齐 | 高 | `attention_v1.py` | padded_elems 计算不满足整除约束 |
| 并发安全 | 低 | `kv_cache` | KV Cache 多线程访问 |
| GQA head 映射 | 中 | `attention_v1.py` | num_heads/num_kv_heads 比例验证 |

### 5.2 解决方案

| 风险点 | 解决方案 | 实施位置 |
| :--- | :--- | :--- |
| 量化精度损失 | 提供精度监控接口；Hadamard 变换提升精度 | `kv_fp8.py` |
| 数据类型不匹配 | 严格类型检查；在接口边界进行类型转换 | `attention_v1.py` |
| 算子兼容性 | 添加算子可用性检查；不支持时降级到 BF16 | `device_op.py` |
| KV Cache reshape 越界 | 添加整除性断言；使用 as_strided 零拷贝 | `device_op.py` |
| Scale 嵌入对齐 | 验证 `padded_elems * block_size % head_size == 0` | `attention_v1.py` |
| 并发安全 | 使用原子操作或锁保护 KV Cache 更新 | `kv_cache` 操作 |
| GQA head 映射 | 验证 `num_heads % num_kv_heads == 0` 且比例为 4 或 8 | `attention_v1.py` |

#### 5.2.1 算子兼容性检查

```python
def _check_fp8_support() -> bool:
    """检查底层是否支持 FP8 融合算子。"""
    try:
        import torch_npu
        return hasattr(torch_npu, 'mytest_rope_norm_store_kv_fp8')
    except ImportError:
        return False
```

#### 5.2.2 Scale 嵌入整除性验证

```python
def _fp8_per_head_scale_elems_padded(block_size, num_kv_heads, head_size):
    """计算 padded_elems，确保 reshape 有效性。"""
    raw = 4  # FP32 scale = 4 bytes = 4 fp8 slots
    unit = head_size // math.gcd(block_size, head_size)
    elems = ((raw + unit - 1) // unit) * unit
    
    # 验证整除性
    assert (elems * block_size) % head_size == 0, (
        f"padded_elems({elems}) * block_size({block_size}) "
        f"must be divisible by head_size({head_size})")
    return elems
```

#### 5.2.3 类型安全检查

```python
def _reshape_and_cache_fp8(self, query, key, value, kv_cache, attn_metadata):
    # 类型检查
    kv_cache_fp8 = kv_cache[0]
    assert kv_cache_fp8.dtype == torch.float8_e4m3fn, \
        f"KV Cache must be FP8 e4m3fn, got {kv_cache_fp8.dtype}"
    
    # 形状检查
    assert kv_cache_fp8.shape[0] == 2, \
        f"KV Cache dim0 must be 2 (K/V), got {kv_cache_fp8.shape[0]}"
    
    # GQA 比例检查
    assert self.num_heads % self.num_kv_heads == 0
    assert self.num_heads // self.num_kv_heads in (4, 8)
    
    # head_dim 检查
    assert self.head_size == 128, \
        f"FP8 KV Cache only supports head_dim=128, got {self.head_size}"
```

---

## 6. 附录

### 6.1 量化类型对照表

| 量化类型 | 配置值 | KV Cache 类型 | 适用场景 |
| :--- | :--- | :--- | :--- |
| 无量化 | `None` | FP16/BF16 | 全精度推理 |
| INT8 KV 量化 | `FAKQuant` | INT8 | FlashAttention INT8 |
| **FP8 KV 量化** | `FAKQuantFP8` | **FP8 e4m3fn** | **Hunyuan V3 FP8** |

### 6.2 KV Cache 量化粒度对照表

| 粒度 | 配置值 | Scale 存储位置 | 适用维度 |
| :--- | :--- | :--- | :--- |
| per_tensor | `"per_tensor"` | 标量，独立存储 | 全局 |
| per_head | `"per_head"` | `[num_kv_heads]`，从 checkpoint 加载 | V（静态量化） |
| per_token_per_head | `"per_token_per_head"` | 嵌入 KV Cache head_size 维度 | K（动态量化） |

### 6.3 FP8 数据类型说明

| 数据类型 | 符号 | 指数位 | 尾数位 | 范围 | 用途 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `float8_e4m3fn` | 1 | 4 | 3 | ~0.0039 to ~448 | KV Cache 存储 |
| `float8_e5m2` | 1 | 5 | 2 | ~0.00006 to ~57344 | 计算中间结果 |

### 6.4 关键常量

| 常量 | 值 | 说明 |
| :--- | :--- | :--- |
| `_FP8_PER_HEAD_SCALE_ELEMS` | 4 | FP32 scale 占 4 个 FP8 槽位 |
| `FP8_DTYPE` | `torch.float8_e4m3fn` | KV Cache FP8 数据类型 |
| `head_dim` | 128 | 支持的 head 维度 |
| `head_per_group` | 4 或 8 | GQA 比例 |
| `block_size` | 64 或 128 | KV Cache block 大小 |

### 6.5 vLLM GPU 侧关键参考代码索引

| 功能 | 文件 | 行号 | 说明 |
| :--- | :--- | :--- | :--- |
| padded_elems 计算 | `hpc_attn.py` | L74-L113 | `_fp8_per_head_scale_elems_padded` |
| pad_total_rows 计算 | `hpc_attn.py` | L115-L144 | `_compute_padded_total_rows` |
| scale 嵌入判断 | `hpc_attn.py` | L147-L185 | `_needs_per_head_scale_in_cache` |
| kv_data/scale 拆分 | `hpc_attn.py` | L198-L274 | `_split_fp8_kv_data_and_scale` |
| K/V scale 获取 | `hpc_attn.py` | L836-L896 | `_get_kv_scales` |
| quant_type 解析 | `hpc_attn.py` | L794-L834 | `_resolve_quant_type` |
| KV Cache shape | `hpc_attn.py` | L576-L590 | `get_kv_cache_shape` |
| stride_order | `hpc_attn.py` | L630-L659 | `get_kv_cache_stride_order` |
| RopeNorm forward | `rope_norm.py` | L253-L446 | `_forward_impl` |
| RopeNorm init | `rope_norm.py` | L94-L167 | `__init__` |

---

## 7. 总结

本特性开发文档详细描述了在 vLLM-Ascend 中为 Hunyuan V3 模型实现 KV Cache FP8 量化的技术方案：

### 7.1 核心设计要点

1. **量化策略**：
   - Q/K/V 权重：MXFP8（复用现有 `W8A8_MXFP8` 方案）
   - Q/K 激活值：FP8 动态量化（per-token-per-head），scale 嵌入 KV Cache
   - V 激活值：FP8 静态量化（per-head），scale 从 checkpoint 加载

2. **融合算子调用位置**：
   - 在模型层（`HYV3Attention.forward`）中调用 `NpuRopeNorm.forward(qkv, layer_name)`
   - 等价于 GPU 的 `HpcRopeNorm`，通过 monkey-patch 注入
   - 融合算子完成：qk-norm → qk-rope → qk-hardma → kv_update → quant(FP8)
   - 返回量化后的 Q，KV 已直接写入 KV Cache

3. **Attention Backend 适配**：
   - `use_npu_rope_norm = True` 时，`reshape_and_cache` 跳过 KV 写入
   - `use_npu_rope_norm = True` 时，`forward_impl` 走 `_forward_fp8_attention()`
   - `_forward_fp8_attention()` 拆分 KV Cache 为 data/scale，使用 FA 算子

4. **KV Cache Layout**：
   - 逻辑形状 BBND：`(2, num_blocks, block_size, num_kv_heads, pad_head_size)`
   - 物理形状 BNBD：`(2, num_blocks, num_kv_heads, block_size_padded, head_size)`
   - Scale 嵌入：head_size 维度 padding 4 个 FP8 槽位存储 FP32 scale，reshape 后 padding 转移到 block_size 维度

5. **kv_data/kv_scale 处理**：
   - 使用 `as_strided` 零拷贝 reshape 拆分 data 和 scale 区域
   - 根据 `KVCacheQuantConfig.granularity` 选择正确的 scale 来源

### 7.2 文件变更清单

| 文件 | 操作 | 说明 |
| :--- | :--- | :--- |
| `vllm_ascend/ops/rope_norm.py` | **新增** | `NpuRopeNorm` 融合算子封装（等价 GPU `HpcRopeNorm`） |
| `vllm_ascend/patch/worker/patch_hunyuan_v3.py` | **新增** | Hunyuan V3 模型 NPU patch（monkey-patch `HYV3Attention`） |
| `vllm_ascend/patch/worker/__init__.py` | 修改 | 导入 `patch_hunyuan_v3` |
| `vllm_ascend/quantization/methods/kv_fp8.py` | **新增** | FP8 量化方案实现（`FAKQuantFP8`） |
| `vllm_ascend/quantization/methods/__init__.py` | 修改 | 导出新量化方案 |
| `vllm_ascend/attention/attention_v1.py` | 修改 | 添加 `use_npu_rope_norm` + `_forward_fp8_attention` + `get_kv_cache_page_size_padded` |
| `vllm_ascend/device/device_op.py` | 修改 | 添加 FP8 KV Cache 操作封装 |
| `vllm_ascend/worker/model_runner_v1.py` | 修改 | `_reshape_kv_cache_tensors` 适配 FP8（传入 `cache_dtype_str`、应用 `stride_order` permute、FP8 dtype） |

### 7.3 预期效果

- KV Cache 内存占用减少约 50%（FP16 → FP8）
- 通过融合算子减少 kernel launch 开销
- 与 GPU 侧保持一致的量化策略和接口设计
- KV Cache 初始化时正确分配 FP8 scale 嵌入空间（`get_kv_cache_page_size_padded` 确保调度器分配充足内存），物理布局为 BNBD
