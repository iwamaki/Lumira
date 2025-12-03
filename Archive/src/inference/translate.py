"""Translation inference for Lumira Transformer."""

from pathlib import Path

import torch

from ..model import LumiraTransformer, ModelConfig, TINY_CONFIG
from ..data import LumiraTokenizer


class Translator:
    """Translator for Japanese <-> Lumira conversion."""

    def __init__(
        self,
        model_path: str | Path,
        tokenizer_path: str | Path,
        model_config: ModelConfig = TINY_CONFIG,
        device: str | None = None,
    ):
        """Initialize translator.

        Args:
            model_path: Path to trained model checkpoint
            tokenizer_path: Path to tokenizer model
            model_config: Model configuration
            device: Device to use (cuda/cpu/mps)
        """
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'

        self.device = torch.device(device)

        # Load tokenizer
        self.tokenizer = LumiraTokenizer(tokenizer_path)

        # Load model
        model_config.vocab_size = self.tokenizer.vocab_size
        self.model = LumiraTransformer(model_config)
        self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()

    def _load_model(self, path: str | Path):
        """Load model weights from checkpoint."""
        checkpoint = torch.load(path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

    @torch.no_grad()
    def translate(
        self,
        text: str,
        max_len: int = 100,
        temperature: float = 0.7,
        top_k: int | None = 50,
        top_p: float | None = 0.9,
    ) -> str:
        """Translate text.

        Args:
            text: Input text to translate
            max_len: Maximum output length
            temperature: Sampling temperature (0 = greedy)
            top_k: Top-k sampling
            top_p: Nucleus sampling

        Returns:
            Translated text
        """
        # Encode input
        src_ids = self.tokenizer.encode(text, add_bos=True, add_eos=True)
        src = torch.tensor([src_ids], dtype=torch.long, device=self.device)

        # Generate
        output = self.model.generate(
            src,
            max_len=max_len,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        # Decode
        output_ids = output[0].tolist()
        translated = self.tokenizer.decode(output_ids, skip_special=True)

        return translated

    def translate_batch(
        self,
        texts: list[str],
        max_len: int = 100,
        temperature: float = 0.7,
    ) -> list[str]:
        """Translate multiple texts.

        Args:
            texts: List of input texts
            max_len: Maximum output length
            temperature: Sampling temperature

        Returns:
            List of translated texts
        """
        return [self.translate(text, max_len, temperature) for text in texts]


def create_demo_ui(translator: Translator):
    """Create Gradio demo UI."""
    try:
        import gradio as gr
    except ImportError:
        print("Gradio not installed. Run: pip install gradio")
        return None

    def translate_fn(text: str, direction: str, temperature: float) -> str:
        if not text.strip():
            return ""
        try:
            result = translator.translate(
                text,
                temperature=temperature,
            )
            return result
        except Exception as e:
            return f"Error: {str(e)}"

    with gr.Blocks(title="Lumira Translator") as demo:
        gr.Markdown("# Lumira Translator")
        gr.Markdown("Translate between Japanese and Lumira")

        with gr.Row():
            with gr.Column():
                input_text = gr.Textbox(
                    label="Input",
                    placeholder="Enter text to translate...",
                    lines=3,
                )
                direction = gr.Radio(
                    choices=["Japanese → Lumira", "Lumira → Japanese"],
                    value="Japanese → Lumira",
                    label="Direction",
                )
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    value=0.7,
                    step=0.1,
                    label="Temperature",
                )
                translate_btn = gr.Button("Translate", variant="primary")

            with gr.Column():
                output_text = gr.Textbox(
                    label="Output",
                    lines=3,
                    interactive=False,
                )

        translate_btn.click(
            fn=translate_fn,
            inputs=[input_text, direction, temperature],
            outputs=output_text,
        )

        gr.Markdown("---")
        gr.Markdown("### Example phrases")
        gr.Examples(
            examples=[
                ["こんにちは", "Japanese → Lumira", 0.7],
                ["ありがとう", "Japanese → Lumira", 0.7],
                ["私はあなたを愛しています", "Japanese → Lumira", 0.7],
            ],
            inputs=[input_text, direction, temperature],
        )

    return demo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--tokenizer", required=True, help="Path to tokenizer model")
    parser.add_argument("--demo", action="store_true", help="Launch Gradio demo")
    parser.add_argument("--text", type=str, help="Text to translate")
    args = parser.parse_args()

    translator = Translator(args.model, args.tokenizer)

    if args.demo:
        demo = create_demo_ui(translator)
        if demo:
            demo.launch()
    elif args.text:
        result = translator.translate(args.text)
        print(f"Translation: {result}")
    else:
        # Interactive mode
        print("Lumira Translator (type 'quit' to exit)")
        while True:
            text = input("\n> ")
            if text.lower() in ['quit', 'exit', 'q']:
                break
            result = translator.translate(text)
            print(f"→ {result}")
