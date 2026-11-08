"""MLX 版本的 demo_simple.py：用 Apple 芯片 GPU 跑转换好的 MLX 权重。

用法和 demo_simple.py 一样，只是 --model-path 要指向 MLX 权重文件夹：
    python demo/demo_simple_mlx.py --model-path models-mlx-8bit \
        --image path/to/screenshot.png --prompt "描述这个界面"
"""
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=os.getenv("MODEL_PATH", "models-mlx-8bit"))
    parser.add_argument("--image", help="Optional local screenshot or HTTP(S) image URL")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--output", help="Optional file for the unmodified model response")
    args = parser.parse_args()

    # 原版：from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template

    # 原版要分两步加载 model 和 processor，这里一步搞定。
    # 不需要 device_map：MLX 在 Mac 上自动用统一内存和 GPU。
    model, processor = load(args.model_path)

    images = [args.image] if args.image else []

    # 把提示词套进模型的对话模板，相当于原版的 processor.apply_chat_template。
    formatted_prompt = apply_chat_template(
        processor, model.config, args.prompt, num_images=len(images),
    )

    # temperature=0.0 相当于原版的 do_sample=False（每次输出都一样）。
    result = generate(
        model, processor, formatted_prompt,
        image=images or None,
        max_tokens=args.max_new_tokens,
        temperature=0.0,
        verbose=False,
    )
    # 新版 mlx-vlm 返回带 .text 的对象，旧版直接返回字符串。
    text = getattr(result, "text", result)

    if args.output:
        from pathlib import Path
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
