#!/usr/bin/env python3.11
"""SGLang-JAX 最基础的推理测试

目标：验证 SGLang-JAX 能否在 TPU 上运行

步骤：
1. 检查环境（JAX, TPU, SGLang-JAX）
2. 初始化 SGLang-JAX 引擎
3. 执行简单推理
"""

import jax
import sys

def step1_check_environment():
    """步骤1: 检查环境"""
    print("=" * 50)
    print("步骤 1: 检查环境")
    print("=" * 50)

    # 检查 JAX
    print(f"JAX 版本: {jax.__version__}")

    # 检查 TPU
    devices = jax.devices()
    print(f"JAX 设备: {devices}")
    print(f"设备数量: {len(devices)}")

    if len(devices) == 0:
        print("❌ 没有检测到设备")
        return False

    # 检查 SGLang-JAX
    try:
        import sglang_jax
        print(f"✅ SGLang-JAX 已安装")
        return True
    except ImportError:
        print("❌ SGLang-JAX 未安装")
        return False

def step2_init_engine():
    """步骤2: 初始化引擎"""
    print("\n" + "=" * 50)
    print("步骤 2: 初始化 SGLang-JAX 引擎")
    print("=" * 50)

    try:
        from sglang_jax.srt.entrypoints.engine import Engine

        # 最简单的配置
        args = {
            "model_path": "Qwen/Qwen3-1.7B",  # 从 HF 下载
            "context_length": 512,
            "tp_size": len(jax.devices()),
            "device_indexes": list(range(len(jax.devices()))),
            "mem_fraction_static": 0.2,
            "disable_radix_cache": False,
            "load_format": "dummy",  # 先用随机权重测试
        }

        print("创建引擎...")
        engine = Engine(**args)
        print("✅ 引擎创建成功")
        return engine
    except Exception as e:
        print(f"❌ 引擎创建失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def step3_test_inference(engine):
    """步骤3: 测试推理"""
    print("\n" + "=" * 50)
    print("步骤 3: 测试推理")
    print("=" * 50)

    try:
        # 准备输入
        prompts = ["Hello, my name is"]

        # 获取采样参数
        sampling_params = engine.get_default_sampling_params()
        sampling_params.max_new_tokens = 20
        sampling_params.temperature = 0.0

        print(f"输入: {prompts[0]}")
        print("生成中...")

        # 生成
        outputs = engine.generate(
            input_ids=None,  # 会自动 tokenize
            prompts=prompts,
            sampling_params=[sampling_params.convert_to_dict()],
        )

        print(f"✅ 生成成功!")
        print(f"输出: {outputs[0].outputs[0].text}")
        return True
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("\n🚀 SGLang-JAX 基础推理测试\n")

    # 步骤1
    if not step1_check_environment():
        sys.exit(1)

    # 步骤2
    engine = step2_init_engine()
    if engine is None:
        sys.exit(1)

    # 步骤3
    if not step3_test_inference(engine):
        sys.exit(1)

    print("\n" + "=" * 50)
    print("✅ 所有测试通过!")
    print("=" * 50)

if __name__ == "__main__":
    main()
