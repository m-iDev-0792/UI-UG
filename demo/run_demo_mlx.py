"""MLX version of run_demo.py: run a UI-UG 2 instruction locally, no vLLM server."""
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--image", help="Optional local screenshot or URL")
    parser.add_argument("--model-path", default=os.getenv("MLX_MODEL_PATH", "models-mlx-8bit"))
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--stream", action="store_true", help="Print text as it is generated")
    args = parser.parse_args()
    from utils.mlx_chat import MLXChatInference

    client = MLXChatInference(model_path=args.model_path, max_tokens=args.max_tokens)
    if args.stream:
        client.chat_with_image_stream(
            "uiug2", args.prompt, args.image, temperature=args.temperature,
            on_text=lambda text: print(text, end="", flush=True),
        )
        print()
    else:
        print(client.chat_with_image("uiug2", args.prompt, args.image,
                                     temperature=args.temperature))


if __name__ == "__main__":
    main()
