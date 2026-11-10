"""Backend tests: client, context guard, hardware fit, HF hub, file ops, attachments, image intent."""
import json
import threading
import time
import zipfile
from pathlib import Path

import pytest

import attachments as att
import context_guard as cg
import fileops
import hardware as hw
import hf_hub
import imagegen
import ollama_client as oc
import settings


# ───────────────────────── ollama_client ─────────────────────────

def test_normalize_host():
    assert oc.normalize_host("0.0.0.0:11434") == "http://127.0.0.1:11434"
    assert oc.normalize_host("localhost") == "http://localhost:11434"
    assert oc.normalize_host("https://example.com:8443/") == "https://example.com:8443"


def test_show_parsing_and_kv_math():
    # numbers straight from the /api/show example in Ollama's own API docs (gemma4)
    mi = {"general.architecture": "gemma4", "gemma4.attention.head_count": 8,
          "gemma4.attention.head_count_kv": 2, "gemma4.attention.key_length": 512,
          "gemma4.attention.value_length": 512, "gemma4.block_count": 42,
          "gemma4.context_length": 131072, "gemma4.embedding_length": 2560}
    info = oc.parse_show("gemma4", {"capabilities": ["completion", "thinking", "vision"],
                                    "model_info": mi, "details": {"family": "gemma4"},
                                    "parameters": "temperature 0.7\nnum_ctx 2048"}, 6_000_000_000)
    assert info.kv_bytes_per_token == 42 * 2 * (512 + 512) * 2
    assert info.ctx_max == 131072 and info.default_num_ctx == 2048
    assert info.vision and info.thinking and not info.tools and info.is_chat and not info.is_image


def test_image_and_embedding_detection():
    assert oc.parse_show("x/z-image-turbo", {"capabilities": []}).is_image
    assert oc.parse_show("anything", {"capabilities": ["image"]}).is_image
    assert not oc.parse_show("qwen2.5:3b", {"capabilities": ["completion", "tools"]}).is_image
    assert oc.parse_show("nomic-embed-text", {"capabilities": ["embedding"]}).is_embedding


def test_client_basic_calls(ollama):
    c = oc.OllamaClient(ollama.url)
    assert c.version() == "0.35.1"
    assert {m["name"] for m in c.tags()} == {"qwen2.5-coder:3b", "qwen2.5:3b"}
    assert c.info("qwen2.5:3b").tools
    with pytest.raises(oc.OllamaError, match="not found"):
        c.show("nope")
    c.delete("qwen2.5:3b")
    assert {m["name"] for m in c.tags()} == {"qwen2.5-coder:3b"}


def test_chat_stream_and_stats(ollama):
    c = oc.OllamaClient(ollama.url)
    evs = list(c.chat_stream({"model": "qwen2.5:3b", "messages": [{"role": "user", "content": "hello there world"}]}))
    assert "".join(e["message"]["content"] for e in evs).startswith("Echo: hello there world")
    assert evs[-1]["done"] and evs[-1]["prompt_eval_count"] > 0


def test_cancel_while_waiting_for_first_token(ollama):
    """The key reason for the custom client: Stop must work before any token arrives."""
    ollama.first_token_delay = 5.0
    c = oc.OllamaClient(ollama.url)
    h = oc.StreamHandle()
    got = []

    def run():
        for ev in c.chat_stream({"model": "qwen2.5:3b", "messages": [{"role": "user", "content": "x"}]}, h):
            got.append(ev)

    t = threading.Thread(target=run)
    t0 = time.time()
    t.start()
    time.sleep(0.4)
    h.cancel()
    t.join(3)
    assert not t.is_alive(), "worker thread still blocked after cancel"
    assert time.time() - t0 < 3 and got == []


def test_cancel_mid_stream(ollama):
    ollama.token_delay = 0.1
    ollama.words = 100
    c = oc.OllamaClient(ollama.url)
    h = oc.StreamHandle()
    n = 0
    for ev in c.chat_stream({"model": "qwen2.5:3b", "messages": [{"role": "user", "content": "x"}]}, h):
        n += 1
        if n == 3:
            h.cancel()
    assert 3 <= n < 20


def test_pull_progress_aggregation(ollama):
    c = oc.OllamaClient(ollama.url)
    agg = oc.PullProgress()
    last = (0, 0, "")
    for ev in c.pull_stream("hf.co/user/repo:Q4_K_M"):
        last = agg.feed(ev)
    assert last == (4000, 4000, "success")
    assert "hf.co/user/repo:Q4_K_M" in ollama.pulled


def test_pull_error_surfaces(ollama):
    c = oc.OllamaClient(ollama.url)
    with pytest.raises(oc.OllamaError, match="does not exist"):
        list(c.pull_stream("hf.co/doesnotexist/x"))


def test_running_context(ollama):
    c = oc.OllamaClient(ollama.url)
    list(c.chat_stream({"model": "qwen2.5:3b", "messages": [{"role": "user", "content": "x"}],
                        "options": {"num_ctx": 8192}}))
    assert c.running_context("qwen2.5:3b") == 8192
    assert c.running_context("other") == 0


def test_unreachable_ollama_is_an_ollama_error():
    c = oc.OllamaClient("http://127.0.0.1:9")
    with pytest.raises(oc.OllamaError, match="Cannot reach Ollama"):
        c.tags()


# ───────────────────────── context_guard ─────────────────────────

def test_estimator_is_conservative_and_learns():
    est = cg.TokenEstimator()
    assert est.text("") == 0
    english = "The quick brown fox jumps over the lazy dog. " * 100
    real_tokens = len(english) / 4.3                      # typical English
    assert est.text(english) > real_tokens                # over-estimates
    assert est.text("日本語のテキスト" * 10) >= 70          # CJK counted ~1 token / char
    before = est.text(english)
    est.observe(before, before * 2)
    assert est.text(english) > before
    est.observe(est.text(english), 1)                     # never decreases
    assert est.factor >= 1.0


def test_detects_the_documented_truncation_signatures():
    # numbers measured by users on Ollama 0.34/0.35 (see ollama/ollama#14259, talos PR #1004)
    assert cg.detect_truncation(2050, 22925, 4096)
    assert cg.detect_truncation(16387, 160689, 32768)
    assert cg.detect_truncation(32770, 120000, 65536)
    assert not cg.detect_truncation(1500, 1400, 4096)
    assert not cg.detect_truncation(300, 1400, 4096)       # small count alone is not truncation (cache)
    assert not cg.detect_truncation(0, 5000, 4096)


def test_prepare_fits_request_untouched():
    est = cg.TokenEstimator()
    p = cg.prepare("sys", "", [{"role": "user", "content": "hi"}], 4096, 4096, est, None)
    assert p.notices == [] and not p.compacted and p.ctx == 4096
    assert p.messages[0] == {"role": "system", "content": "sys"}


def test_prepare_raises_context_before_compacting():
    est = cg.TokenEstimator()
    msgs = [{"role": "user", "content": "word " * 2500}]             # ~3900 est tokens
    p = cg.prepare("sys", "", msgs, 4096, 32768, est, lambda *a: "never")
    assert p.ctx > 4096 and not p.compacted
    assert any("raised" in n.lower() for n in p.notices)
    assert p.est_tokens <= cg.prompt_budget(p.ctx)


def test_prepare_compacts_when_ceiling_reached():
    est = cg.TokenEstimator()
    calls = []

    def summarize(prev, chunk, ctx):
        calls.append((prev, chunk, ctx))
        return "- SUMMARY"

    msgs = []
    for i in range(14):
        msgs += [{"role": "user", "content": f"question {i} " + "blah " * 150},
                 {"role": "assistant", "content": f"answer {i} " + "blah " * 150}]
    msgs.append({"role": "user", "content": "final question"})
    p = cg.prepare("You are BraidLM.", "", msgs, 4096, 4096, est, summarize)
    assert p.compacted and p.consumed > 0 and calls
    assert p.est_tokens <= cg.prompt_budget(4096)
    assert "- SUMMARY" in p.messages[0]["content"] and p.messages[0]["content"].startswith("You are BraidLM.")
    assert p.messages[-1]["content"] == "final question"             # newest message survives
    assert p.messages[1]["role"] == "user"                           # kept tail starts on a user turn
    assert all(c[2] == 4096 for c in calls)


def test_prepare_trims_single_oversized_attachment():
    est = cg.TokenEstimator()
    big = '<attached_file name="a.txt" path="/a">\n' + ("lorem ipsum dolor " * 3000) + "\n</attached_file>"
    p = cg.prepare("sys", "", [{"role": "user", "content": "summarise\n\n" + big}], 4096, 4096, est, lambda *a: "")
    assert p.est_tokens <= cg.prompt_budget(4096)
    assert "BraidLM removed" in p.messages[-1]["content"]
    assert any("too large" in n for n in p.notices)
    assert p.messages[-1]["content"].startswith("summarise")         # head kept
    assert p.messages[-1]["content"].rstrip().endswith("</attached_file>")  # tail kept


def test_summary_excludes_attachment_bodies():
    t = cg.transcript([{"role": "user", "content": 'look\n<attached_file name="x.py" path="/x">SECRET BODY</attached_file>'}])
    assert "SECRET BODY" not in t and "x.py" in t


def test_split_never_returns_empty_keep():
    est = cg.TokenEstimator()
    msgs = [{"role": "user", "content": "x " * 5000}]
    old, keep = cg.split_for_compaction(msgs, est, 100)
    assert old == [] and keep == msgs


# ───────────────────────── hardware ─────────────────────────

def _hw(vram_gb=12, ram_gb=32):
    gpus = [hw.Gpu("RTX", vram_gb * hw.GiB, vram_gb * hw.GiB)] if vram_gb else []
    return hw.Hardware(ram_gb * hw.GiB, ram_gb * hw.GiB // 2, gpus)


def test_classify_levels():
    h = _hw(12, 32)
    assert hw.classify(5 * hw.GiB, h).level == "gpu"
    assert hw.classify(16 * hw.GiB, h).level == "partial"
    assert hw.classify(60 * hw.GiB, h).level == "no"
    assert hw.classify(5 * hw.GiB, _hw(0, 16)).level == "cpu"
    assert hw.classify(30 * hw.GiB, _hw(0, 16)).level == "no"


def test_unified_memory_budget():
    h = hw.Hardware(16 * hw.GiB, 8 * hw.GiB, [], unified=True)
    assert hw.classify(6 * hw.GiB, h).level == "gpu"
    assert hw.classify(14 * hw.GiB, h).level == "no"


def test_recommend_ctx_respects_model_limit_and_memory():
    h = _hw(12, 32)
    assert hw.recommend_ctx(4 * hw.GiB, 131072, 8192, h) == 8192              # trained limit
    assert hw.recommend_ctx(4 * hw.GiB, 131072, 131072, h) == 32768           # soft cap
    big_kv = hw.recommend_ctx(4 * hw.GiB, 2 * 1024 * 1024, 131072, h)         # 2 MB/token KV is expensive
    assert big_kv < 32768
    assert hw.recommend_ctx(30 * hw.GiB, 131072, 131072, h) <= 8192           # spills to RAM -> modest


def test_detect_parses_nvidia_smi_and_overrides(monkeypatch):
    class R:
        returncode = 0
        stdout = "NVIDIA GeForce RTX 4070, 12282, 11000\n"

    d = hw.detect(run=lambda *a, **k: R(), plat="linux")
    assert d.gpus and d.gpus[0].name.startswith("NVIDIA") and d.gpus[0].total == 12282 * hw.MiB
    d2 = hw.detect(run=lambda *a, **k: R(), plat="linux", env={"BRAIDLM_VRAM_GB": "24"})
    assert d2.vram_total == 24 * hw.GiB and "manually" in d2.describe()


# ───────────────────────── hf_hub ─────────────────────────

def _seed_gguf(hf):
    hf.files["acme/Tiny-GGUF"] = {
        "Tiny-Q4_K_M.gguf": b"a" * 4000, "Tiny-Q8_0.gguf": b"b" * 7000, "Tiny-IQ3_M.gguf": b"c" * 3000,
        "Tiny-f16-00001-of-00002.gguf": b"d" * 6000, "Tiny-f16-00002-of-00002.gguf": b"e" * 6000,
        "mmproj-Tiny-f16.gguf": b"f" * 100, "README.md": b"x"}
    hf.meta["acme/Tiny-GGUF"] = {"downloads": 1234, "likes": 5, "pipeline_tag": "text-generation",
                                 "tags": ["gguf"], "gated": False}


def test_gguf_repo_listing_groups_shards_and_pagination(hf):
    _seed_gguf(hf)                                  # page size 3 forces pagination
    repo = hf_hub.HubClient(base=hf.url).gguf_repo("acme/Tiny-GGUF")
    q = {v.quant: v for v in repo.variants}
    assert set(q) == {"Q4_K_M", "Q8_0", "IQ3_M", "F16"}
    assert q["F16"].size == 12000 and len(q["F16"].files) == 2            # shards summed
    assert repo.has_mmproj
    assert [v.quant for v in repo.variants] == ["IQ3_M", "Q4_K_M", "Q8_0", "F16"]  # sorted by size


def test_search_sends_gguf_filter(hf):
    _seed_gguf(hf)
    res = hf_hub.HubClient(base=hf.url).search("tiny", kind="gguf")
    assert res[0].repo_id == "acme/Tiny-GGUF" and res[0].downloads == 1234
    assert hf.last_search["filter"] == ["gguf"] and hf.last_search["sort"] == ["downloads"]
    hf_hub.HubClient(base=hf.url).search("x", kind="image")
    assert "diffusers" in hf.last_search["filter"] and hf.last_search["pipeline_tag"] == ["text-to-image"]


def test_recommend_variant_prefers_q4km_when_it_fits_and_shrinks_when_not(hf):
    _seed_gguf(hf)
    variants = hf_hub.HubClient(base=hf.url).gguf_repo("acme/Tiny-GGUF").variants
    roomy = _hw(24, 64)
    assert variants[hf_hub.recommend_variant(variants, roomy)].quant == "Q4_K_M"
    # a machine where only the smallest quant fits the GPU
    tiny = hw.Hardware(8 * hw.GiB, 4 * hw.GiB, [hw.Gpu("g", int(0.8 * hw.GiB), int(0.8 * hw.GiB))])
    sizes = [hw.classify(v.size * 120_000, tiny).level for v in variants]    # scale up to GB range
    big = [hf_hub.GgufVariant(v.quant, v.size * 120_000) for v in variants]
    pick = big[hf_hub.recommend_variant(big, tiny)]
    assert hw.classify(pick.size, tiny).rank == max(hw.classify(b.size, tiny).rank for b in big)


def test_ollama_reference_format():
    v = hf_hub.GgufVariant("Q4_K_M", 1)
    assert hf_hub.ollama_ref("bartowski/Llama-3.2-3B-Instruct-GGUF", v) == \
        "hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M"
    assert hf_hub.ollama_ref("a/b", hf_hub.GgufVariant("(default)", 1)) == "hf.co/a/b"


def test_download_resume_gated_and_cancel(hf, tmp_path):
    payload = bytes(range(256)) * 4000                               # ~1 MB
    hf.files["acme/img"] = {"model_index.json": b"{}", "unet/w.safetensors": payload}
    hub = hf_hub.HubClient(base=hf.url)
    plan = hub.diffusers_plan("acme/img")
    assert plan.total == 2 + len(payload) and plan.variant is None
    dest = tmp_path / "dl"
    # pre-seed a partial file: the downloader must resume with a Range request
    (dest / "unet").mkdir(parents=True)
    (dest / "unet" / "w.safetensors.part").write_bytes(payload[:300_000])
    seen = []
    hub.download_plan(plan, dest, lambda d, t, n: seen.append((d, t)))
    assert (dest / "unet" / "w.safetensors").read_bytes() == payload
    assert hf.range_requests == 1 and seen[-1][0] == plan.total
    assert json.loads((dest / "braidlm_meta.json").read_text())["repo_id"] == "acme/img"
    # gated
    hf.files["acme/gated"] = {"model_index.json": b"{}"}
    hf.gated.add("acme/gated")
    with pytest.raises(hf_hub.HubError, match="gated"):
        hub.diffusers_plan("acme/gated")
    # cancel
    ev = threading.Event()
    ev.set()
    with pytest.raises(hf_hub.HubError, match="cancelled"):
        hub.download_plan(plan, tmp_path / "dl2", cancel=ev)


# ───────────────────────── fileops ─────────────────────────

def test_apply_backup_undo_preserves_crlf_and_trailing_newline(tmp_path):
    f = tmp_path / "app.py"
    f.write_bytes(b"print('a')\r\nprint('b')\r\n")
    old, nl = fileops.read_text_exact(str(f))
    assert nl == "\r\n" and old == "print('a')\nprint('b')\n"
    new = "print('a')\nprint('B')"                     # model dropped the trailing newline
    d = fileops.unified_diff(old, fileops.normalise_new_text(old, new), "app.py")
    assert "-print('b')" in d and "+print('B')" in d
    assert fileops.diff_stats(old, new) == (1, 1)
    backup = fileops.apply_with_backup(str(f), new)
    assert f.read_bytes() == b"print('a')\r\nprint('B')\r\n"
    assert Path(backup).read_bytes() == b"print('a')\r\nprint('b')\r\n"
    fileops.undo(str(f), str(backup))
    assert f.read_bytes() == b"print('a')\r\nprint('b')\r\n"


def test_guess_target_prefers_matching_extension_then_latest():
    c = ["/x/notes.md", "/x/app.py", "/x/util.py"]
    assert fileops.guess_target("print(1)", "python", c) == 2
    assert fileops.guess_target("# util.py\nx=1", "python", c) == 2
    assert fileops.guess_target("# app.py\nx=1", "python", c) == 1
    assert fileops.guess_target("x", "markdown", c) == 0


# ───────────────────────── attachments ─────────────────────────

def test_text_attachment_and_prompt_format(tmp_path):
    p = tmp_path / "hello.py"
    p.write_text("print('héllo')\n", encoding="utf-8")
    a = att.load_attachment(str(p), allow_images=False)
    assert a.kind == "text" and a.editable and "héllo" in a.text
    body = att.format_for_prompt("fix it", [a])
    assert body.startswith("fix it") and f'<attached_file name="hello.py"' in body and body.rstrip().endswith("</attached_file>")


def test_binary_and_oversize_and_image_gating(tmp_path):
    b = tmp_path / "x.bin"
    b.write_bytes(b"\x00\x01\x02" * 100)
    with pytest.raises(att.AttachmentError, match="binary"):
        att.load_attachment(str(b), True)
    img = tmp_path / "pic.png"
    img.write_bytes(b"not really")
    with pytest.raises(att.AttachmentError, match="cannot look at images"):
        att.load_attachment(str(img), allow_images=False)


def test_docx_extraction(tmp_path):
    p = tmp_path / "d.docx"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("word/document.xml",
                   '<w:document><w:body><w:p><w:r><w:t>Hello &amp; welcome</w:t></w:r></w:p>'
                   '<w:p><w:r><w:t>Second</w:t></w:r></w:p></w:body></w:document>')
    a = att.load_attachment(str(p), False)
    assert a.kind == "docx" and a.text == "Hello & welcome\nSecond"


def test_pdf_extraction(tmp_path):
    stream = b"BT /F1 18 Tf 20 100 Td (Hello PDF world) Tj ET"
    objs = [b"<</Type/Catalog/Pages 2 0 R>>", b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
            b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>",
            b"<</Length %d>>stream\n" % len(stream) + stream + b"\nendstream",
            b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>"]
    out = b"%PDF-1.4\n"
    offs = []
    for i, o in enumerate(objs, 1):
        offs.append(len(out))
        out += b"%d 0 obj\n" % i + o + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for o in offs:
        out += b"%010d 00000 n \n" % o
    out += b"trailer<</Root 1 0 R/Size %d>>\nstartxref\n%d\n%%%%EOF" % (len(objs) + 1, xref)
    p = tmp_path / "t.pdf"
    p.write_bytes(out)
    a = att.load_attachment(str(p), False)
    assert "Hello PDF world" in a.text


def test_image_is_downscaled_and_encoded(tmp_path):
    from PySide6.QtGui import QImage, QColor
    im = QImage(3000, 2000, QImage.Format.Format_RGB32)
    im.fill(QColor("red"))
    p = tmp_path / "big.png"
    im.save(str(p))
    a = att.load_attachment(str(p), allow_images=True)
    import base64
    out = QImage.fromData(base64.b64decode(a.image_b64))
    assert max(out.width(), out.height()) == att.MAX_IMAGE_SIDE and a.is_image


# ───────────────────────── imagegen (no heavy deps) ─────────────────────────

IMG_YES = ["generate an image of a red fox in the snow", "Please create a picture of a castle at sunset",
           "draw me a cat", "Can you draw a dragon?", "make a logo for my bakery", "/image neon city at night",
           "I want you to generate a photo of a mountain lake", "Paint a portrait of an old sailor",
           "imagine a poster for a jazz festival", "Now create an illustration of a robot gardening"]
IMG_NO = ["How do I create an image in Python?", "write a function that generates an image thumbnail",
          "what is an image?", "Can you review my CSS for the hero image?", "create a docker image for my app",
          "How can I make a picture smaller with ImageMagick", "generate a summary of this article",
          "Write a story about a painter", "hello", "explain how photos are stored"]


@pytest.mark.parametrize("t", IMG_YES)
def test_image_intent_positive(t):
    assert imagegen.looks_like_image_request(t)


@pytest.mark.parametrize("t", IMG_NO)
def test_image_intent_negative(t):
    assert not imagegen.looks_like_image_request(t)


def test_prompt_cleaning():
    assert imagegen.clean_prompt("generate an image of a red fox") == "a red fox"
    assert imagegen.clean_prompt("make a logo for my bakery") == "make a logo for my bakery"
    assert imagegen.clean_prompt("/image neon city") == "neon city"


def test_ollama_native_image_roundtrip_and_text_model_rejected(ollama):
    ollama.add("x/z-image-turbo", 1, ["image"])
    c = oc.OllamaClient(ollama.url)
    progress = []
    png = imagegen.generate_ollama(c, "x/z-image-turbo", "a cat", oc.StreamHandle(), lambda d, t: progress.append((d, t)))
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and progress[-1] == (3, 3)
    with pytest.raises(imagegen.ImageGenError, match="text instead of an image"):
        imagegen.generate_ollama(c, "qwen2.5:3b", "a cat", oc.StreamHandle())
    ollama.image_works = False
    with pytest.raises(imagegen.ImageGenError, match="v0.32.6"):
        imagegen.generate_ollama(c, "x/z-image-turbo", "a cat", oc.StreamHandle())


def test_missing_diffusers_gives_actionable_message(monkeypatch):
    import builtins
    real = builtins.__import__

    def fake(name, *a, **k):
        if name in ("torch", "diffusers", "transformers"):
            raise ImportError("no", name=name)
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake)
    ok, why = imagegen.diffusers_status()
    assert not ok and "requirements-imagegen.txt" in why and "pytorch.org" in why


def test_default_params_by_family():
    assert imagegen.default_params("stabilityai/sdxl-turbo") == {"steps": 4, "guidance": 0.0, "size": 1024}
    assert imagegen.default_params("Tongyi-MAI/Z-Image-Turbo")["steps"] == 9
    assert imagegen.default_params("black-forest-labs/FLUX.1-schnell")["steps"] == 4
    assert imagegen.default_params("stabilityai/sd-turbo")["size"] == 512
    assert imagegen.default_params("runwayml/stable-diffusion-v1-5") == {"steps": 30, "guidance": 6.5, "size": 512}


def test_local_diffusers_registry(tmp_path):
    d = imagegen.diffusers_dir("acme/pic")
    d.mkdir(parents=True)
    (d / "braidlm_meta.json").write_text(json.dumps({"repo_id": "acme/pic", "variant": "fp16", "size": 5}))
    models = imagegen.list_local_diffusers()
    assert [(m.name, m.variant, m.key) for m in models] == [("acme/pic", "fp16", "diffusers:acme/pic")]
    imagegen.remove_diffusers(models[0])
    assert imagegen.list_local_diffusers() == []
