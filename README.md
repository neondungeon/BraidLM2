## Quick start
1. Install Ollama: https://ollama.com/download
2. Windows: double-click `start.bat`. Linux/macOS: `./start.sh`
   (or `pip install -r requirements.txt && python main.py`).
3. First run with no models: BraidLM offers to install `qwen2.5-coder:3b` / `qwen2.5:3b` or open the browser.

Optional image generation from Hugging Face models: `pip install -r requirements-imagegen.txt`
(read the note inside it about CUDA PyTorch).

## Images
Select an image model and every prompt generates an image. With a text model selected, an imperative request
("generate an image of...", "draw me...", `/image …`) is sent to your preferred installed image model; if none is installed you get a tip.
Images are saved in `~/.braidlm/images`.

## Files
`main.py` entry · `window.py` UI · `model_browser.py` ＋ dialog · `worker.py` background jobs ·
`ollama_client.py` cancellable Ollama client · `context_guard.py` · `hardware.py` · `hf_hub.py` · `imagegen.py` ·
`attachments.py` · `fileops.py` · `renderer.py` · `settings.py` · `tests/`

Run the tests: `pip install pytest && python -m pytest tests -q` (uses mock Ollama/Hugging Face servers; no network needed).
Data lives in `~/.braidlm` (override with `BRAIDLM_HOME`). Override hardware detection with `BRAIDLM_VRAM_GB` / `BRAIDLM_RAM_GB`.
