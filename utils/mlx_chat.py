"""MLX version of utils/vllm_chat.py: runs the model in-process on Apple silicon.

vllm_chat.VLLMChatInference sends requests to a separate vLLM server.
MLXChatInference loads the converted MLX weights into this Python process
instead, so no server is needed. The public methods keep the same names and
arguments, so code written for VLLMChatInference only needs a different
constructor:

    # before
    client = VLLMChatInference(api_base="http://127.0.0.1:8000/v1")
    # after
    client = MLXChatInference(model_path="models-mlx-8bit")

    text = client.chat_with_image("uiug2", prompt, "screenshot.png")
"""
import os
from typing import Any, Callable, Dict, Optional


class MLXChatInference:
    """Drop-in replacement for VLLMChatInference backed by mlx-vlm."""

    def __init__(self, model_path: Optional[str] = None, max_tokens: int = 8192,
                 max_pixels: Optional[int] = None, min_pixels: Optional[int] = None):
        """
        :param model_path: Folder with MLX weights (from `python -m mlx_vlm.convert`).
               Defaults to env `MLX_MODEL_PATH`, then `models-mlx-8bit`.
        :param max_tokens: Maximum tokens in response.
        :param max_pixels: Optional cap on image area; larger screenshots are
               downscaled (keeping aspect ratio) before the model sees them.
               None keeps the model's own preprocessing, like demo_simple.py.
        :param min_pixels: Optional floor on image area; smaller images are upscaled.
        """
        from mlx_vlm import load

        self.model_path = model_path or os.getenv("MLX_MODEL_PATH", "models-mlx-8bit")
        self.max_tokens = max_tokens
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        # Loading takes a few seconds, so do it once and reuse for every request.
        self.model, self.processor = load(self.model_path)

    def load_image(self, image_path_or_url: str):
        """Open a local path, http(s) URL or data URL as an RGB PIL image."""
        from mlx_vlm.utils import load_image

        if not image_path_or_url.startswith(("http://", "https://", "data:")) \
                and not os.path.isfile(image_path_or_url):
            raise FileNotFoundError(f"Image file not found: {image_path_or_url}")
        image = load_image(image_path_or_url, timeout=30)
        area = image.width * image.height
        scale = 1.0
        if self.max_pixels and area > self.max_pixels:
            scale = (self.max_pixels / area) ** 0.5
        elif self.min_pixels and area < self.min_pixels:
            scale = (self.min_pixels / area) ** 0.5
        if scale != 1.0:
            size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
            image = image.resize(size)
        return image

    def _prepare(self, prompt: str, image_path_or_url: Optional[str],
                 extra_params: Optional[Dict[str, Any]]):
        from mlx_vlm.prompt_utils import apply_chat_template

        extra_params = extra_params or {"chat_template_kwargs": {"enable_thinking": False}}
        images = [self.load_image(image_path_or_url)] if image_path_or_url else []
        formatted = apply_chat_template(
            self.processor, self.model.config, prompt, num_images=len(images),
            **extra_params.get("chat_template_kwargs", {}),
        )
        return formatted, images or None

    def chat_with_image(self,
                        model_name: str,
                        prompt: str,
                        image_path_or_url: Optional[str] = None,
                        temperature: float = 0.1,
                        extra_params: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate response with image and text.

        :param model_name: Ignored; kept so calls written for VLLMChatInference still work.
        :param prompt: Text prompt
        :param image_path_or_url: Image URL or local path (None for text-only)
        :param temperature: Sampling temperature (0 = deterministic)
        :param extra_params: Optional {"chat_template_kwargs": {...}}, as in vllm_chat
        :return: Model response text
        """
        from mlx_vlm import generate

        formatted, images = self._prepare(prompt, image_path_or_url, extra_params)
        result = generate(
            self.model, self.processor, formatted, image=images,
            max_tokens=self.max_tokens, temperature=temperature, verbose=False,
        )
        return result.text

    def chat_with_image_stream(self,
                               model_name: str,
                               prompt: str,
                               image_path_or_url: Optional[str] = None,
                               temperature: float = 0.1,
                               extra_params: Optional[Dict[str, Any]] = None,
                               on_text: Optional[Callable[[str], None]] = None) -> str:
        """
        Generate a streaming response with image and text.

        :param on_text: Optional callback called with each new piece of text,
               e.g. `lambda s: print(s, end="", flush=True)` to watch it appear.
        :return: Accumulated response text
        """
        from mlx_vlm import stream_generate

        formatted, images = self._prepare(prompt, image_path_or_url, extra_params)
        response_text = ""
        for chunk in stream_generate(
            self.model, self.processor, formatted, image=images,
            max_tokens=self.max_tokens, temperature=temperature,
        ):
            if chunk.text:
                response_text += chunk.text
                if on_text:
                    on_text(chunk.text)
        return response_text
