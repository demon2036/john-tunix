#!/usr/bin/env python
"""SGLang-JAX 最基础的推理测试

目标：验证 SGLang-JAX 能否在 TPU 上运行

步骤：
1. 检查环境（JAX, SGLang-JAX）
2. 初始化 SGLang-JAX 引擎
3. 执行简单推理
"""

import sys

def step1_check_environment():
    """步骤1: 检查环境"""
    print("=" * 50)
    print("步骤 1: 检查环境")
    print("=" * 50)

    # 检查 JAX（不初始化设备，避免与Engine冲突）
    try:
        import jax
        print(f"✅ JAX 版本: {jax.__version__}")
    except ImportError:
        print("❌ JAX 未安装")
        return False

    # 检查 SGLang-JAX
    try:
        import sgl_jax
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
        from sgl_jax.srt.entrypoints.engine import Engine

        # Qwen3-1.7B配置
        args = {
            "model_path": "Qwen/Qwen3-1.7B",
            "context_length": 2048,
            "tp_size": 8,  # Qwen3-1.7B应该有16个attention heads，能被8整除
            "device_indexes": list(range(8)),
            "mem_fraction_static": 0.6,
            "disable_radix_cache": False,
            "load_format": "auto",  # 从HuggingFace下载真实权重
        }

        print("创建引擎...")
        print(f"模型: {args['model_path']}")
        print(f"TP大小: {args['tp_size']}")
        print(f"负载格式: {args['load_format']} (HuggingFace真实权重)")
        print("正在从HuggingFace下载模型...")


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
        from transformers import AutoTokenizer

        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

        # 使用chat template格式化消息
        messages = [
            {"role": "user", "content": "你好吗，你是谁"}
        ]

        # 应用chat template
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        print(f"原始消息: {messages[0]['content']}")
        print(f"\nChat template格式化后的prompt:")
        print(f"{prompt}")
        print()

        # 获取采样参数
        sampling_params = engine.get_default_sampling_params()
        sampling_params.max_new_tokens = 200  # 增加输出长度以容纳思考内容
        sampling_params.temperature = 0.7
        sampling_params.top_p = 0.9

        print("生成中...")

        # 生成
        outputs = engine.generate(
            prompt=prompt,
            sampling_params=sampling_params.convert_to_dict(),
        )

        print(f"✅ 生成成功!")
        print(f"\n{'='*50}")
        print(f"生成文本:")
        print(outputs['text'])
        print(f"{'='*50}")
        print(f"\n元信息:")
        print(f"  - Prompt tokens: {outputs['meta_info']['prompt_tokens']}")
        print(f"  - Completion tokens: {outputs['meta_info']['completion_tokens']}")
        print(f"  - E2E latency: {outputs['meta_info']['e2e_latency']:.3f}s")
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
