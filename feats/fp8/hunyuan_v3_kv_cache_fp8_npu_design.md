# Hunyuan V3 KV Cache FP8 NPU A5 实现方案

## 1. 目标与范围

本文档面向 NPU A5 上 Hunyuan V3 模型的 KV Cache FP8 特性开发，目标是在 `vllm-ascend` 中实现与 `vllm` GPU 标杆路径一致的 KV Cache FP8 推理链路。

范围约束：

- 仅讨论 KV Cache FP8，不包含 FP8 权重载入和权重量化实现。
- 默认 checkpoint 中 K/V cache scale 的加载、命名映射和权重侧 MXFP8 能力由其他特性完成。
- 假设 FIA 算子已经支持非连续 KV Cache。
- 假设 NPU 已提供融合 `rope_norm` 能力的算子，可完成 QK-Norm、RoPE、KV cache 写入和 FP8 量化。
- 以 GPU `./vllm` 中 Hunyuan V3 + HPC attention 的已验证实现为标杆。

推荐的量化组合：

| Tensor | 粒度 | scale 来源 | 存储位置 |
| --- | --- | --- | --- |
| Q activation | per-token-per-head 动态 FP8 | 融合算子动态生成 | 传给 FIA 或由 FIA 可消费的配套输出 |
| K cache | per-token-per-head 动态 FP8 | 融合算子动态生成 | 嵌入 K cache 的额外 scale 行 |
| V cache | per-head 静态 FP8 | checkpoint scale | `Attention` 层的 `_v_scale` |

## 2. GPU 标杆实现结论

GPU 侧关键文件：

- `vllm/model_executor/models/hunyuan_v3.py`
- `vllm/model_executor/layers/hpc/rope_norm.py`
- `vllm/v1/attention/backends/hpc_attn.py`
- `vllm/model_executor/layers/attention/attention.py`

### 2.1 模型层职责

`HYV3Attention` 在初始化时，如果满足 `HpcRopeNorm.support()`，会创建 `self.hpc_rope_norm`。Hunyuan V3 的 QK-Norm 顺序是 norm 再 rope，因此 `qk_norm_policy=2`。

FP8 KV cache 开启时，模型层 forward 的主流程为：

```text
hidden_states
  -> qkv_proj
  -> split q/k/v
  -> hpc_rope_norm(qkv)
       - q/k norm
       - q/k rope
       - 写入 K/V cache
       - Q 动态 FP8 量化
       - 生成 q_scale / split_k_flag
  -> Attention(q_fp8, k, v)
  -> o_proj
```

注意：`k`、`v` 仍传入 `Attention` 是为了保持上层接口不破坏；实际 KV cache 写入在 `HpcRopeNorm` 内完成，`HpcAttentionImpl.do_kv_cache_update()` 在 `use_hpc_rope_norm=True` 时直接返回。

### 2.2 KV Cache 形状与 scale 嵌入

GPU 标杆中，per-token-per-head 的 K scale 嵌入 KV cache。逻辑形状为：

```text
(num_blocks, 2, block_size, num_kv_heads, head_size + padded_elems)
```

其中 `2` 表示 K/V。`padded_elems` 至少为 4，因为一个 fp32 scale 占 4 个 fp8 槽位。为了能零拷贝 reshape 成 scale 行，需要满足：

```text
block_size * (head_size + padded_elems) % head_size == 0
```

计算方式：

```python
raw = 4
unit = head_size // gcd(block_size, head_size)
padded_elems = ceil(raw / unit) * unit
pad_total_rows = block_size * (head_size + padded_elems) // head_size
```

运行时将 cache 视为：

```text
(num_blocks, 2, pad_total_rows, num_kv_heads, head_size)
  [:, :, :block_size, :, :]      -> FP8 KV data
  [:, :, block_size:, :, :]      -> fp32 scale bytes viewed through FP8 storage
```

### 2.3 Page Size 与 stride_order

GPU `HpcAttentionBackend` 实现了：

- `get_kv_cache_shape(..., cache_dtype_str)`
- `get_kv_cache_page_size_padded(...)`
- `get_kv_cache_stride_order(...)`

这三者必须同时成立：

1. `get_kv_cache_shape` 在 FP8 + per-token-per-head 时返回 padded head size。
2. `get_kv_cache_page_size_padded` 让调度器按 padded page size 分配足够内存。
3. model runner 按 backend 的 `stride_order` 先 view 成物理布局，再 permute 回逻辑布局，保留非连续 stride。

否则会出现两类错误：

- 分配内存不够，view 直接失败。
- 内存够但 layout 与算子理解不一致，推理结果错误。

### 2.4 Attention 消费路径

GPU FP8 attention 不只消费 FP8 Q/K/V，还消费：

- prefill 的 `hpc_prefill_q_scale`
- decode 的 `hpc_decode_q_scale`
- decode 的 `hpc_split_k_flag`
- K scale：per-token-per-head 时来自 KV cache scale 区域
- V scale：per-head 时来自 `layer._v_scale`

因此 NPU 方案也必须明确 Q scale 如何进入 FIA。若当前 FIA 接口已隐式支持融合算子的 Q scale 输出，需要在框架侧记录这个契约；若 FIA 需要显式 scale 参数，则 `_forward_fp8_attention` 必须补齐参数。

## 3. NPU 当前尝试实现分析

当前尝试提交：`583b4e2e1371d407778e4b2b3bdf18aaa3d6fc9f`

涉及文件：

- `vllm_ascend/attention/attention_v1.py`
- `vllm_ascend/device/device_op.py`
- `vllm_ascend/ops/rope_norm.py`
- `vllm_ascend/patch/worker/patch_hunyuan_v3.py`
- `vllm_ascend/quantization/methods/kv_fp8.py`
- `vllm_ascend/worker/model_runner_v1.py`

### 3.1 已有方向

当前代码已经覆盖了几个正确方向：

- `AscendAttentionBackend.get_kv_cache_shape()` 在 FP8 时追加 padded head size。
- `AscendAttentionBackend.get_kv_cache_page_size_padded()` 返回 padded page size。
- `AscendAttentionBackend.get_kv_cache_stride_order()` 返回 `(0, 1, 3, 2, 4)`，目标是 K/V 分离后的 `(num_blocks, num_kv_heads, block_size, head_size)` 物理布局。
- `DeviceOperator.split_fp8_kv_cache_and_scale()` 试图把 `(B, H, BS, D_pad)` 拆成 data 和 scale。
- `NpuRopeNorm` 试图对齐 GPU `HpcRopeNorm`，由模型层调用融合算子写 KV cache。
- `AscendAttentionBackendImpl.reshape_and_cache()` 在 `use_npu_rope_norm=True` 时跳过普通 KV 写入。

### 3.2 必须修正的问题

#### 3.2.1 `patch_hunyuan_v3.py` 目前不可直接运行

当前文件存在明显未定义符号：

- 使用 `envs.VLLM_PRECISION_MODE`，但实际只导入了 `vllm_envs`。
- 使用 `config.rope_parameters`，但 patched init 中没有 `config` 局部变量。
- 使用 `max_position_embeddings`，但 patched init 中没有该局部变量。

推荐做法：不要在 patch 中重建一套参数解析逻辑。直接参照 GPU `HYV3Attention.__init__` 的字段，在 `_original_init()` 后使用已存在的实例属性：

- `self.head_dim`
- `self.num_heads`
- `self.num_kv_heads`
- `self.rotary_emb.cos_sin_cache`
- `self.use_qk_norm`
- `self.q_norm`
- `self.k_norm`
- `self.attn.layer_name`

#### 3.2.2 `NpuRopeNorm` 不是可调用子模块

当前 `NpuRopeNorm` 是普通 Python 类，没有继承 `torch.nn.Module` 或 vLLM `CustomOp`，也没有 `__call__`，但 patched forward 中使用：

```python
q = self.npu_rope_norm(qkv, self.attn.layer_name)
```

这会导致 `TypeError`。推荐将 `NpuRopeNorm` 实现为 `torch.nn.Module`，至少保证：

- `super().__init__()`
- `forward()` 可通过 `self.npu_rope_norm(...)` 调用
- 作为 `HYV3Attention` 子模块参与 module traversal
- `process_weights_after_loading()` 能被 vLLM 模型加载流程调用

如果需要和 GPU 一样作为编译分割点，再进一步继承/封装为 vLLM `CustomOp` 风格。

#### 3.2.3 `process_weights_after_loading` 调用链不完整

GPU `HpcRopeNorm` 是子模块，模型加载完成后会执行 `process_weights_after_loading()`，从 fallback norm 中提取：

- `qnorm_weight`
- `knorm_weight`

NPU 当前普通类不参与该流程，融合算子可能拿到 `None` 或错误 dtype。推荐：

- `NpuRopeNorm(torch.nn.Module)` 挂到 `HYV3Attention` 上。
- 实现 `process_weights_after_loading(self, act_dtype)`，读取 `self.fallback_qnorm.weight` 和 `self.fallback_knorm.weight`。
- 权重 dtype 建议与算子契约一致，通常为 fp32。

#### 3.2.4 patched forward 有重复 attention 调用

当前 `patch_hunyuan_v3.py` 中：

```python
if self.npu_rope_norm is not None:
    ...
    attn_output = self.attn(q, k, v, output_shape)
else:
    ...

attn_output = self.attn(q, k, v)
```

这会导致 npu_rope_norm 分支先调用一次 attention，随后又无条件调用第二次 attention。应改为与 GPU Hunyuan V3 一致的单出口结构：

```python
if self.npu_rope_norm is not None:
    q = self.npu_rope_norm(qkv, self.attn.layer_name)
    q = q.view(-1, self.num_heads * self.head_dim)
    attn_output = self.attn(q, k, v, output_shape)
else:
    ...  # 原生 norm + rope
    attn_output = self.attn(q, k, v, output_shape)
```

#### 3.2.5 KV cache 逻辑布局与物理布局需要统一

NPU model runner 当前将 K/V raw tensor 分开分配，然后在 FP8 时直接 view 成物理形状：

```text
k_shape logical:   (num_blocks, block_size, num_kv_heads, head_size_padded)
stride_order 4D:   (0, 2, 1, 3)
k_shape physical:  (num_blocks, num_kv_heads, block_size, head_size_padded)
```

这条路线可以成立，但必须全链路明确：`kv_cache[0]` 和 `kv_cache[1]` 暴露给 NPU 算子的就是 BNBD 物理 shape，而不是 GPU 那种“逻辑 shape + 非连续 stride”。相应地：

- `split_fp8_kv_cache_and_scale()` 应只处理 `(B, H, BS, D_pad)`。
- `NpuRopeNorm` 传给融合算子的 `key_cache/value_cache` 是 `(B, H, BS, D)`，还是需要转成 `(B, BS, H, D)`，必须以算子契约为准。
- 已假设 FIA 支持非连续，建议优先复刻 GPU 模式：view 物理布局后 `permute(inv_order)` 回逻辑 shape，给上层保持 `(B, BS, H, D_pad)` 语义和非连续 stride；算子需要物理 BNBD 时由 device adaptor 统一转换视图。

#### 3.2.6 `_forward_fp8_attention` 不能丢失 Q scale 语义

当前 `_forward_fp8_attention()` 将 FP8 query 直接传给 `torch_npu.npu_fused_infer_attention_score()`，但没有显式传入 q_scale。GPU 标杆中 q_scale 是 FP8 attention 的必要输入。

推荐接口契约：

- `NpuRopeNorm` 在 prefill/decode 后把 q_scale 写入 `attn_metadata.npu_prefill_q_scale` / `attn_metadata.npu_decode_q_scale`。
- 若 FIA 需要 `query_scale` 参数，则 `_forward_fp8_attention()` 显式传入。
- 若 FIA 与融合算子通过隐藏 buffer 或 query 打包格式传递 q_scale，则在代码注释和断言中固定该契约，并在单测中验证。

#### 3.2.7 `reshape_and_cache_fp8()` 不是 Hunyuan V3 FP8 主路径

`DeviceOperator.reshape_and_cache_fp8()` 当前调用 `mytest_rope_norm_store_kv_fp8(query=..., key=..., value=...)`，接口形式与 `NpuRopeNorm` 中的 `qkv=...` 形式不同。

对 Hunyuan V3，推荐只有一条 FP8 KV 写入路径：

```text
HYV3Attention.forward -> NpuRopeNorm -> fused rope_norm_store_kv_fp8
```

普通 `reshape_and_cache_fp8()` 可作为非 Hunyuan 或非融合路径的后续扩展，不应在 Hunyuan V3 FP8 主路径中被触发。

## 4. 推荐实现方案

### 4.1 总体链路

```text
启动参数 cache_dtype=fp8_e4m3
  -> Attention.get_kv_cache_spec()
       - dtype=torch.float8_e4m3fn
       - cache_dtype_str=fp8_e4m3
       - page_size_padded 包含 K/V scale 嵌入空间
  -> NPUModelRunner 初始化 KV cache
       - 按 backend shape + stride_order 分配/视图化
       - K/V cache 可零拷贝拆出 data 与 scale
  -> HYV3Attention 初始化
       - 创建 NpuRopeNorm 子模块
       - qk_norm_policy=2
       - use_npu_rope_norm=True
  -> HYV3Attention.forward
       - qkv_proj
       - NpuRopeNorm(qkv)
       - Attention(q_fp8, k, v)
  -> AscendAttentionBackendImpl
       - reshape_and_cache 跳过 KV 写入
       - _forward_fp8_attention 调 FIA
```

### 4.2 KV Cache 规格

推荐 `AscendAttentionBackend.get_kv_cache_shape()` 与 GPU 语义对齐：

```python
def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size,
                       cache_dtype_str="auto"):
    if cache_dtype_str in ("fp8", "fp8_e4m3") and needs_per_head_scale:
        padded = fp8_per_head_scale_elems_padded(block_size, num_kv_heads, head_size)
        return (2, num_blocks, block_size, num_kv_heads, head_size + padded)
    return (2, num_blocks, block_size, num_kv_heads, head_size)
```

因为 vllm-ascend 当前 K/V raw tensor 分开分配，model runner 使用 `kv_cache_shape[1:]` 作为单个 K 或 V 的 shape。

`get_kv_cache_page_size_padded()` 必须与上面的 shape 严格一致：

```text
page_size_padded =
  block_size * num_kv_heads *
  ((head_size + padded_k) + (head_size_v + padded_v)) *
  sizeof(cache_dtype)
```

对于 Hunyuan V3 当前组合 K=per-token-per-head、V=per-head，是否给 V 也 padding 有两种选择：

1. K/V 对称 padding：实现简单，K/V shape 一致，浪费极少量内存。
2. 仅 K padding：内存更精确，但 K/V shape 和 page split 更复杂。

建议第一阶段采用 K/V 对称 padding，对齐当前尝试实现，降低布局错误风险。即使 V 不使用 per-token scale，`v_scale` 区域不被读取即可。

### 4.3 KV Cache 视图策略

推荐采用“逻辑 shape 暴露 + 物理 stride 保留”的方式，与 GPU model runner 对齐：

```python
logical_shape = (2, B, BS, H, D_pad)
stride_order = (0, 1, 3, 2, 4)
physical_shape = tuple(logical_shape[i] for i in stride_order)
inv_order = [stride_order.index(i) for i in range(len(stride_order))]

kv_cache = raw.view(dtype).view(physical_shape).permute(*inv_order)
```

如果因为 vllm-ascend K/V 分离必须分别处理，则对单个 K/V cache 使用：

```python
logical_shape_4d = (B, BS, H, D_pad)
stride_order_4d = (0, 2, 1, 3)
physical_shape_4d = (B, H, BS, D_pad)
inv_order_4d = (0, 2, 1, 3)

k_cache = raw_k.view(dtype).view(physical_shape_4d).permute(*inv_order_4d)
```

这样上层看到的是 `(B, BS, H, D_pad)`，stride 表达 BNBD 物理布局。`DeviceOperator` 内根据算子要求决定是否 `.permute(0, 2, 1, 3)` 获得 `(B, H, BS, D)` 视图。

### 4.4 Scale 拆分

建议提供两个明确函数，避免混淆 5D 合并 K/V 与 4D K/V 分离：

```python
split_fp8_kv_5d(kv_cache, head_size, kv_cache_quant_config)
# 输入逻辑: (2, B, BS, H, D_pad)
# 输出: kv_data (2, B, BS, H, D), kv_scale (2, B, scale_rows, H, D)

split_fp8_kv_4d(cache, head_size, kv_cache_quant_config)
# 输入逻辑: (B, BS, H, D_pad)
# 输出: data (B, BS, H, D), scale (B, scale_rows, H, D)
```

对单个 K/V cache 的零拷贝拆分逻辑：

```python
B, BS, H, D_pad = cache.shape
pad_total_rows = BS * D_pad // head_size
s = cache.stride()

cache_reshaped = torch.as_strided(
    cache,
    (B, pad_total_rows, H, head_size),
    (s[0], H * head_size, head_size, 1),
)
data = cache_reshaped[:, :BS, :, :]
scale = cache_reshaped[:, BS:, :, :]
```

如果算子要求 BNBD，则最后只做 view/permute：

```python
data_for_op = data.permute(0, 2, 1, 3)
scale_for_op = scale.permute(0, 2, 1, 3)
```

### 4.5 NpuRopeNorm

推荐实现为子模块：

```python
class NpuRopeNorm(torch.nn.Module):
    def __init__(...):
        super().__init__()
        ...

    def process_weights_after_loading(self, act_dtype):
        if self.use_qk_norm:
            self.qnorm_weight = self.fallback_qnorm.weight.detach().float()
            self.knorm_weight = self.fallback_knorm.weight.detach().float()

    def forward(self, qkv, layer_name):
        output = torch.empty(
            (num_tokens, self.num_heads, self.head_dim),
            dtype=torch.float8_e4m3fn if self.use_fp8 else qkv.dtype,
            device=qkv.device,
        )
        self._forward_impl(qkv, layer_name, output)
        return output
```

`_forward_impl()` 需要：

- 从 `get_forward_context()` 获取当前 layer 的 metadata 和 attention layer。
- 获取 `attn_layer.kv_cache[0]` 中的 K/V cache。
- 拆分 data/scale。
- 分 prefill/decode 调融合算子。
- 保存 q_scale 到 metadata。
- 如果 decode 算子输出 split flag，也保存到 metadata。

建议 metadata 字段命名：

```python
npu_prefill_q_scale: torch.Tensor | None
npu_decode_q_scale: torch.Tensor | None
npu_split_k_flag: torch.Tensor | None
```

### 4.6 Hunyuan V3 patch

当前 vllm-ascend 通过 patch worker 修改 Hunyuan V3。如果上游 `vllm` 代码已经包含 `HpcRopeNorm`，推荐最小化 patch：

1. 保留 `_original_init()` 的完整初始化。
2. 如果 NPU + FP8 + support，创建 `self.npu_rope_norm`。
3. 复用 `self.rotary_emb.cos_sin_cache`，不要重建 `rotary_emb`。
4. 设置 `self.attn.query_quant = None`。
5. 设置 `self.attn.impl.use_npu_rope_norm = True`。
6. patched forward 仅替换 `self.hpc_rope_norm` 分支为 `self.npu_rope_norm` 分支，其余逻辑保持 GPU Hunyuan V3 原样。

### 4.7 AscendAttentionBackendImpl

`reshape_and_cache()`：

- `use_npu_rope_norm=True` 时跳过 KV 写入。
- FP8 Hunyuan 主路径不调用 `reshape_and_cache_fp8()`。

`_forward_fp8_attention()`：

- 从 KV cache 拆出 K/V data 和必要 scale。
- 保持 KV cache 非连续能力，不应无条件 `.contiguous()`，除非算子不支持。题设已假设 FIA 支持非连续，因此建议删除当前 `k_contig/v_contig` 拷贝。
- 按 FIA 契约传入 Q/K/V scale。
- 输出 dtype 应为模型计算 dtype，通常 bf16，而不是 FP8。

建议伪代码：

```python
k_data, k_scale = split_fp8_kv_4d(k_cache, self.head_size, qcfg)
v_data, _ = split_fp8_kv_4d(v_cache, self.head_size, qcfg)

query_scale = (
    attn_metadata.npu_prefill_q_scale
    if prefill
    else attn_metadata.npu_decode_q_scale
)
v_scale = layer._v_scale

torch_npu.npu_fused_infer_attention_score(
    query=query,
    key=k_data,
    value=v_data,
    query_scale=query_scale,  # 若 FIA 显式需要
    key_scale=k_scale,
    value_scale=v_scale,
    ...
)
```

如果 FIA 使用 packed FP8 query 隐式携带 q_scale，则应增加断言：

```python
assert query.dtype == torch.float8_e4m3fn
assert attn_metadata.npu_prefill_q_scale is not None or implicit_q_scale_supported
```

## 5. 文件修改建议

### 5.1 `vllm_ascend/attention/attention_v1.py`

需要保留/完善：

- `get_kv_cache_shape(..., cache_dtype_str)`
- `get_kv_cache_page_size_padded(...)`
- `get_kv_cache_stride_order(...)`
- `use_fp8_kv_cache`
- `use_npu_rope_norm`
- `_kv_cache_quant_config`
- `_resolve_quant_type`
- `_forward_fp8_attention`

需要调整：

- `_resolve_kv_cache_quant_config()` 优先读取 `vllm_config.cache_config.kv_cache_quant_config`，不要硬编码。
- `_resolve_quant_type()` 按 K/V granularity 映射，至少支持 `(per_token_per_head, per_head)`。
- `_forward_fp8_attention()` 去掉无条件 `.contiguous()`。
- `_forward_fp8_attention()` 明确处理 q_scale/k_scale/v_scale。
- `forward()` 中当前断言 `layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0` 对 FP8 per-head scale 场景可能不成立，需要确认是否应在 FP8 Hunyuan 分支跳过或改成更精确断言。

### 5.2 `vllm_ascend/device/device_op.py`

需要保留/完善：

- `fp8_per_head_scale_elems_padded`
- `needs_per_head_scale_in_cache`
- split data/scale 的零拷贝函数

需要调整：

- 分清 5D 合并 KV 与 4D K/V 分离的函数命名。
- split 函数按逻辑 shape 编写，物理布局转换集中在 helper 中。
- 不再把 `reshape_and_cache_fp8()` 作为 Hunyuan V3 FP8 主路径。

### 5.3 `vllm_ascend/ops/rope_norm.py`

需要调整：

- `NpuRopeNorm` 继承 `torch.nn.Module`。
- 支持 `self.npu_rope_norm(...)` 调用。
- 参与 `process_weights_after_loading()`。
- `_resolve_quant_config()` 优先读取当前 `VllmConfig`。
- `enable_hadamard` 不应接收后丢弃；如果算子支持，需要映射到 quant policy；如果暂不支持，应显式 warning 或禁用。
- prefill/decode 调用后保存 q_scale/split_k_flag 到 metadata。

### 5.4 `vllm_ascend/patch/worker/patch_hunyuan_v3.py`

需要修正：

- `envs` 未定义。
- `config` 未定义。
- `max_position_embeddings` 未定义。
- `NpuRopeNorm` 不是 callable。
- 重复 attention 调用。
- 尽量复用 `_original_init()` 后的实例属性，减少与 GPU Hunyuan V3 源码漂移。

### 5.5 `vllm_ascend/worker/model_runner_v1.py`

需要确认：

- `get_kv_cache_shape()` 调用必须传入 `cache_dtype_str=self.vllm_config.cache_config.cache_dtype`。
- KV cache raw tensor 分配必须基于 `page_size_padded`。
- K/V 分离视图时，建议 view physical 后 permute 回 logical，保持上层语义与 GPU 一致。
- 单测覆盖 `cache_dtype=fp8_e4m3`、`block_size=128`、`head_size=128` 的元素数、shape、stride。

### 5.6 `vllm_ascend/quantization/methods/kv_fp8.py`

当前 `FAKQuantFP8` 建立 `fa_q/fa_k/fa_v.scale`，并将 `fa_v.scale` 写入 `layer._v_scale`。

需要确认：

- Hunyuan V3 scale 权重命名是否能稳定映射到 `fa_k/fa_v` 或 `attn._k_scale/_v_scale`。
- K per-token-per-head 动态时，`layer._k_scale` 仅作为 fallback 或 per_tensor/per_head 模式使用。
- V per-head 静态时，`layer._v_scale` shape 应为 `[num_kv_heads]`，TP 切分与 GPU 标杆一致。

## 6. 验证计划

### 6.1 静态与单元测试

1. import 测试：启动 vllm-ascend 后 `patch_hunyuan_v3.py` 无未定义符号。
2. callable 测试：`NpuRopeNorm` 是 `torch.nn.Module` 且可调用。
3. weight hook 测试：模型加载后 `qnorm_weight/knorm_weight` 非空，dtype/shape 正确。
4. shape 测试：
   - `block_size=128`
   - `head_size=128`
   - `padded_elems=4`
   - `D_pad=132`
   - `pad_total_rows=132`
5. page size 测试：`page_size_padded` 等于 K/V padded 总 bytes。
6. stride 测试：KV cache 逻辑 shape 为 `(B, BS, H, D_pad)`，stride 对应 BNBD 物理布局。
7. split 测试：data shape 为 `(B, BS, H, D)`，scale view 为 `(B, 4, H, D)`，`scale.view(torch.float32)` 合法。

### 6.2 算子集成测试

1. 单层小 batch prefill：
   - 融合算子写 K/V cache。
   - q_scale 存入 metadata。
   - FIA 正常返回 bf16/fp16 output。
2. 单层 decode：
   - 新 token 写入对应 slot。
   - decode q_scale 可用。
   - 若 split flag 存在，shape 与 batch/kv_heads 一致。
3. chunked prefill + decode 混合批。
4. MTP decode query_len > 1 场景，如当前模型需要支持。

### 6.3 端到端对齐

1. BF16 KV cache baseline。
2. NPU FP8 KV cache。
3. GPU FP8 标杆。

对比维度：

- 首 token logits max diff / mean diff。
- 多轮 decode logits diff。
- 固定 prompt greedy 输出一致性。
- 长上下文下 cache 写入无越界。
- 显存/NPU HBM 占用下降符合预期。

### 6.4 性能测试

1. prefill throughput。
2. decode throughput。
3. HBM 占用。
4. 去掉 `.contiguous()` 前后 FIA 输入非连续性能。
5. 与 BF16 KV cache 的吞吐和内存对比。

## 7. 风险与决策点

| 风险 | 影响 | 建议 |
| --- | --- | --- |
| FIA 是否需要显式 q_scale 未确认 | FP8 attention 数值错误 | 优先确认算子接口；框架 metadata 先保留 q_scale |
| KV cache 逻辑/物理 layout 混用 | silent wrong result | 统一成逻辑 shape + stride，device adaptor 内转换 |
| NpuRopeNorm 不参与 weight hook | norm 权重为空或 dtype 错 | 继承 `nn.Module` 并实现 hook |
| 当前 patch 未定义符号 | 启动即失败 | 最小化 patch，复用 `_original_init` 后属性 |
| block_size GPU=64、NPU=128 | 标杆差异 | NPU 保持 128，但 padding/page/scale 公式必须通用 |
| V per-head 却对称 padding | 少量内存浪费 | 第一阶段接受，稳定后可优化仅 K padding |
| `.contiguous()` 复制 KV cache | FP8 内存/性能收益下降 | 在 FIA 非连续支持后移除 |

## 8. 建议开发顺序

1. 修正 `patch_hunyuan_v3.py` 基础可运行问题。
2. 将 `NpuRopeNorm` 改为 `torch.nn.Module`，打通 `process_weights_after_loading`。
3. 统一 KV cache 逻辑 shape、stride_order、split helper。
4. 在 metadata 中增加 NPU FP8 q_scale/split flag 字段。
5. 明确 FIA FP8 scale 参数契约，补齐 `_forward_fp8_attention`。
6. 增加 shape/page/stride/split 单测。
7. 跑单层算子集成测试。
8. 跑 Hunyuan V3 端到端精度和性能对比。

## 9. 验收标准

功能验收：

- `cache_dtype=fp8_e4m3` 下 Hunyuan V3 能完成 prefill、decode、chunked prefill。
- KV cache 分配包含 scale 嵌入空间，无 view/stride 错误。
- NPU fused rope_norm 算子负责 KV 写入，普通 reshape_and_cache 不重复写。
- FIA 使用正确的 FP8 Q/K/V scale 语义。

精度验收：

- 与 NPU BF16 baseline 的 logits diff 在业务可接受范围内。
- 与 GPU FP8 标杆的行为差异有明确解释。

性能验收：

- HBM 占用相对 BF16 KV cache 明显下降。
- decode 性能不因额外 `.contiguous()` 或 Python 分支产生明显退化。
