# 🧵 BraidLM

A lightweight desktop chat app for local models, built on [Ollama](https://ollama.com) and the
[`nocolour/Ollama-Desktop`](https://github.com/nocolour/Ollama-Desktop) architecture (MIT).
**What it does that Ollama alone does not is spelled out, with evidence and caveats, in
[`BRAIDLM_VS_OLLAMA.txt`](BRAIDLM_VS_OLLAMA.txt).**

## Quick start
1. Install Ollama → https://ollama.com/download
2. Windows: double-click `start.bat`. Linux/macOS: `./start.sh`
   (or `pip install -r requirements.txt && python main.py`).
3. First run with no models: BraidLM offers to install `qwen2.5-coder:3b` / `qwen2.5:3b` or open the browser.

Optional image generation from Hugging Face models: `pip install -r requirements-imagegen.txt`
(read the note inside it about CUDA PyTorch).

## Header: `Model: [＋] [Coder] [General] [More models ▾]`
* **＋** opens the model browser (Hugging Face search → pick a quantisation with a fit verdict → install; an
  *Installed* tab removes models and shows disk use). Language models install through Ollama
  (`hf.co/<repo>:<quant>`); image models (diffusers) download into `~/.braidlm/image_models`.
  You can also paste `hf.co/user/repo:Q4_K_M`.
* **Coder / General** – the original built-ins. Dimmed = not installed (click to install). Removing them sticks.
* **More models** – every other installed chat model and 🎨 image model.

## Bottom row
* **📎 Attach** (also drag & drop) – text/code, PDF, DOCX; images only for vision-capable models. Shown for chat models,
  hidden for image/embedding models. Ask for a change and the reply's code blocks get **Copy / Apply to file… / Save as…**.
  *Apply* shows a diff, backs the file up to `~/.braidlm/backups`, and offers **Undo**.
* **■ Stop** – while an answer is being generated it stops it (partial text is kept; `Esc` does the same).
  When nothing is running it **ends the conversation**: the transcript is saved to `~/.braidlm/conversations/` and the chat is reset.
* **✕** clears without saving.
* **Status bar** – context meter and `ctx:` menu (Auto or a fixed 2k…128k).

## Images
Select an image model and every prompt generates an image. With a text model selected, an imperative request
("generate an image of …", "draw me …", `/image …`) is sent to your preferred installed image model; if none is installed you get a tip.
Images are saved in `~/.braidlm/images`.

## Files
`main.py` entry · `window.py` UI · `model_browser.py` ＋ dialog · `worker.py` background jobs ·
`ollama_client.py` cancellable Ollama client · `context_guard.py` · `hardware.py` · `hf_hub.py` · `imagegen.py` ·
`attachments.py` · `fileops.py` · `renderer.py` · `settings.py` · `tests/`

Run the tests: `pip install pytest && python -m pytest tests -q` (uses mock Ollama/Hugging Face servers; no network needed).
Data lives in `~/.braidlm` (override with `BRAIDLM_HOME`). Override hardware detection with `BRAIDLM_VRAM_GB` / `BRAIDLM_RAM_GB`.
