"""
BanglaCare demo (guide section 17.2), with both architectures.

One text box, a model selector, extracted entities, severity + calibrated
confidence, and a safety disclaimer. Either architecture can be run alone, or
both on the same query side by side:

  BiLSTM + FastText   the shipped model      ~2.1M params   ~150s to load
  BanglaBERT          the transformer variant ~111M params  ~20s to load

Pipelines are built lazily - the first query against an architecture pays its
load cost, and the BiLSTM's 8GB of FastText binaries are never touched if you
only ever use the transformer.

Usage
-----
  python app/app.py
  python app/app.py --arch transformer   # preselect in the UI
  python app/app.py --preload            # load everything at startup instead
  python app/app.py --cpu --share

Requires gradio (and, for the transformer, transformers + the csebuetnlp
normalizer). See requirements.txt.
"""

import argparse
from pathlib import Path

import gradio as gr

from inference import DEFAULTS, DISCLAIMER, PipelineRegistry

ROOT = Path(__file__).resolve().parents[1]

EXAMPLES = [
    "আমার বয়স ২৭ বছর। হঠাৎ বুকে অনেক ব্যথা হচ্ছে এবং শ্বাস নিতে কষ্ট হচ্ছে। Napa খেয়েছি।",
    "গত দুইদিন ধরে সর্দি কাশি হচ্ছে, জ্বর নেই।",
    "ডায়াবেটিসের রোগীর জন্য কোন specialist দেখানো উচিত?",
    "বাচ্চার জ্বর ১০৩ ডিগ্রি পার হয়ে গেছে, চোখ বন্ধ করে ঝিমাচ্ছে।",
    "amar matha betha korche 2 din dhore, ki korbo?",
]

MODE_BILSTM = "BiLSTM + FastText"
MODE_TRANSFORMER = "BanglaBERT"
MODE_BOTH = "Compare both"
MODE_TO_ARCHS = {
    MODE_BILSTM: ["bilstm"],
    MODE_TRANSFORMER: ["transformer"],
    MODE_BOTH: ["bilstm", "transformer"],
}

SEVERITY_STYLE = {
    "Emergency":     {"color": "#ff4d6d", "glow": "rgba(255,77,109,0.35)", "icon": "🚨"},
    "Urgent":        {"color": "#ffa53d", "glow": "rgba(255,165,61,0.35)", "icon": "⚠️"},
    "Routine":       {"color": "#3ddc97", "glow": "rgba(61,220,151,0.35)", "icon": "🩺"},
    "General Query": {"color": "#4cc9f0", "glow": "rgba(76,201,240,0.35)", "icon": "💬"},
}

ENTITY_COLORS = {
    "Symptom": "#8b5cf6", "Health Condition": "#ec4899", "Medicine": "#06b6d4",
    "Age": "#f59e0b", "Dosage": "#10b981", "Specialist": "#6366f1",
    "Medical Procedure": "#14b8a6",
}

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&display=swap');

:root {
  --bc-bg-0: #0b0f19; --bc-bg-1: #121826; --bc-bg-2: #1a2333;
  --bc-border: #2a3550; --bc-text: #e6ebf5; --bc-text-dim: #93a1bf;
  --bc-accent: #06b6d4; --bc-accent-2: #8b5cf6;
}

.gradio-container {
  background: radial-gradient(circle at 15% 0%, #16213a 0%, var(--bc-bg-0) 45%) !important;
  font-family: 'Inter', ui-sans-serif, sans-serif !important;
  color: var(--bc-text) !important;
}

#bc-header { text-align: center; padding: 8px 0 4px 0; }
#bc-header h1 {
  font-family: 'Space Grotesk', sans-serif !important;
  font-size: 2.4em !important; font-weight: 700 !important; margin-bottom: 2px !important;
  background: linear-gradient(90deg, #4cc9f0, #8b5cf6 60%, #ff4d6d);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text; letter-spacing: 0.5px;
}
#bc-header p { color: var(--bc-text-dim) !important; font-size: 1.02em; margin-top: 0 !important; }

.bc-card {
  background: linear-gradient(180deg, var(--bc-bg-1), var(--bc-bg-2)) !important;
  border: 1px solid var(--bc-border) !important; border-radius: 16px !important;
  padding: 18px !important; box-shadow: 0 8px 30px rgba(0,0,0,0.35);
}

.bc-section-label {
  font-family: 'Space Grotesk', sans-serif; font-size: 0.78em; font-weight: 700;
  letter-spacing: 1.5px; text-transform: uppercase; color: var(--bc-accent);
  margin-bottom: 10px; display: block;
}

#bc-input textarea {
  background: var(--bc-bg-0) !important; border: 1px solid var(--bc-border) !important;
  border-radius: 12px !important; color: var(--bc-text) !important; font-size: 1.05em !important;
}
#bc-input textarea:focus {
  border-color: var(--bc-accent) !important;
  box-shadow: 0 0 0 3px rgba(6,182,212,0.15) !important;
}

#bc-run-btn {
  background: linear-gradient(90deg, var(--bc-accent), var(--bc-accent-2)) !important;
  border: none !important; color: white !important; font-weight: 600 !important;
  border-radius: 10px !important; box-shadow: 0 4px 18px rgba(6,182,212,0.30);
  transition: transform 0.15s ease, box-shadow 0.15s ease;
}
#bc-run-btn:hover { transform: translateY(-1px); box-shadow: 0 6px 24px rgba(139,92,246,0.40); }

/* ---- results ---- */
.bc-results { display: flex; gap: 14px; flex-wrap: wrap; align-items: stretch; }
.bc-panel {
  flex: 1 1 320px; min-width: 300px;
  background: linear-gradient(180deg, var(--bc-bg-1), var(--bc-bg-2));
  border: 1px solid var(--bc-border); border-radius: 16px; padding: 16px 18px;
}
.bc-panel-head {
  display: flex; align-items: baseline; justify-content: space-between;
  gap: 10px; margin-bottom: 12px; padding-bottom: 10px;
  border-bottom: 1px solid var(--bc-border);
}
.bc-panel-title {
  font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.02em;
  color: var(--bc-text);
}
.bc-panel-meta { font-size: 0.76em; color: var(--bc-text-dim); font-family: 'Space Grotesk', sans-serif; }

.bc-entity-chip {
  display: inline-flex; align-items: center; gap: 6px; padding: 5px 11px;
  margin: 3px 5px 3px 0; border-radius: 999px; font-size: 0.86em;
  font-weight: 500; white-space: nowrap;
}
.bc-entity-chip .bc-dot { width: 7px; height: 7px; border-radius: 50%; display: inline-block; }
.bc-entity-type {
  font-family: 'Space Grotesk', sans-serif; font-size: 0.73em; font-weight: 700;
  text-transform: uppercase; letter-spacing: 0.5px; opacity: 0.85;
}
.bc-empty-note { color: var(--bc-text-dim); font-style: italic; font-size: 0.9em; }

.bc-severity-label {
  font-family: 'Space Grotesk', sans-serif; font-size: 1.45em; font-weight: 700;
  display: flex; align-items: center; gap: 9px; margin-top: 4px;
}
.bc-confidence-track {
  background: rgba(255,255,255,0.08); border-radius: 999px; height: 8px;
  width: 100%; margin: 11px 0 4px 0; overflow: hidden;
}
.bc-confidence-fill { height: 100%; border-radius: 999px; }
.bc-confidence-text {
  font-size: 0.8em; color: var(--bc-text-dim); font-family: 'Space Grotesk', sans-serif;
  letter-spacing: 0.5px;
}
.bc-prob-row { display: flex; align-items: center; gap: 8px; margin-top: 5px; font-size: 0.8em; }
.bc-prob-label { width: 118px; color: var(--bc-text-dim); flex-shrink: 0; }
.bc-prob-track { flex: 1; background: rgba(255,255,255,0.06); border-radius: 999px; height: 6px; overflow: hidden; }
.bc-prob-fill { height: 100%; border-radius: 999px; opacity: 0.85; }
.bc-prob-pct { width: 40px; text-align: right; color: var(--bc-text-dim); flex-shrink: 0; }

.bc-review-banner {
  margin-top: 12px; background: rgba(255,165,61,0.12);
  border: 1px solid rgba(255,165,61,0.4); color: #ffcb8a;
  padding: 9px 13px; border-radius: 10px; font-size: 0.86em;
}
.bc-subhead {
  font-family: 'Space Grotesk', sans-serif; font-size: 0.72em; font-weight: 700;
  letter-spacing: 1.2px; text-transform: uppercase; color: var(--bc-accent);
  margin: 14px 0 7px 0; display: block;
}
.bc-agree {
  margin-bottom: 12px; padding: 10px 14px; border-radius: 10px;
  font-size: 0.88em; border: 1px solid var(--bc-border);
}
.bc-disclaimer {
  text-align: center; color: var(--bc-text-dim); font-size: 0.85em; padding: 10px 0 4px 0;
}
.bc-hint { color: var(--bc-text-dim); font-size: 0.84em; margin-top: 6px; }
"""


def _render_entities(entities):
    if not entities:
        return "<div class='bc-empty-note'>No medical entities detected.</div>"
    chips = []
    for e in entities:
        color = ENTITY_COLORS.get(e["type"], "#93a1bf")
        chips.append(
            f"<span class='bc-entity-chip' style='background:{color}1f;"
            f"border:1px solid {color}55;color:{color};'>"
            f"<span class='bc-dot' style='background:{color};'></span>"
            f"<span class='bc-entity-type'>{e['type']}</span>{e['text']}</span>"
        )
    return "<div>" + "".join(chips) + "</div>"


def _render_severity(severity, confidence, probs, needs_review):
    style = SEVERITY_STYLE.get(severity, {"color": "#93a1bf", "glow": "transparent", "icon": "•"})
    color = style["color"]

    prob_rows = "".join(
        f"<div class='bc-prob-row'>"
        f"<span class='bc-prob-label'>{SEVERITY_STYLE.get(n, {}).get('icon', '')} {n}</span>"
        f"<span class='bc-prob-track'><span class='bc-prob-fill' "
        f"style='width:{p*100:.1f}%;background:{SEVERITY_STYLE.get(n, {}).get('color', color)};'>"
        f"</span></span><span class='bc-prob-pct'>{p:.0%}</span></div>"
        for n, p in sorted(probs.items(), key=lambda kv: -kv[1])
    )

    html = (
        f"<div class='bc-severity-label' style='color:{color};'>"
        f"<span>{style['icon']}</span><span>{severity}</span></div>"
        f"<div class='bc-confidence-track'><div class='bc-confidence-fill' "
        f"style='width:{confidence*100:.1f}%;background:linear-gradient(90deg,{color},{color}aa);'>"
        f"</div></div>"
        f"<div class='bc-confidence-text'>CALIBRATED CONFIDENCE &nbsp;{confidence:.1%}</div>"
        f"<div style='margin-top:12px;'>{prob_rows}</div>"
    )
    if needs_review:
        html += (
            "<div class='bc-review-banner'>🔍 <b>Low confidence — "
            "recommend human review</b></div>"
        )
    return html


def _render_panel(result):
    """One model's full result: header, entities, severity."""
    if result.get("error"):
        return (
            f"<div class='bc-panel'><div class='bc-panel-head'>"
            f"<span class='bc-panel-title'>{result.get('arch_label', '')}</span></div>"
            f"<div style='color:#ff9eb0;'>⚠️ {result['error']}</div></div>"
        )

    params = result["n_params"]
    params_str = f"{params/1e6:.1f}M params" if params >= 1e6 else f"{params:,} params"
    return (
        f"<div class='bc-panel'>"
        f"<div class='bc-panel-head'>"
        f"<span class='bc-panel-title'>{result['arch_label']}</span>"
        f"<span class='bc-panel-meta'>{params_str}</span>"
        f"</div>"
        f"<span class='bc-subhead'>Extracted entities</span>"
        f"{_render_entities(result['entities'])}"
        f"<span class='bc-subhead'>Severity triage</span>"
        f"{_render_severity(result['severity'], result['confidence'], result['severity_probs'], result['needs_review'])}"
        f"</div>"
    )


def _render_agreement(results):
    """Headline agree/disagree line shown above a two-model comparison."""
    sevs = [r["severity"] for r in results if not r.get("error")]
    if len(sevs) < 2:
        return ""
    if sevs[0] == sevs[1]:
        return (
            f"<div class='bc-agree' style='background:rgba(61,220,151,0.10);"
            f"border-color:rgba(61,220,151,0.35);color:#9af0c8;'>"
            f"✓ <b>Both models agree:</b> {sevs[0]}</div>"
        )
    return (
        f"<div class='bc-agree' style='background:rgba(255,165,61,0.10);"
        f"border-color:rgba(255,165,61,0.35);color:#ffcb8a;'>"
        f"⚖️ <b>Models disagree:</b> {results[0]['arch_label']} says "
        f"<b>{sevs[0]}</b>, {results[1]['arch_label']} says <b>{sevs[1]}</b> — "
        f"a case worth human review.</div>"
    )


def build_interface(registry, default_mode):
    def run(text, mode):
        archs = MODE_TO_ARCHS[mode]
        results = []
        for arch in archs:
            try:
                results.append(registry.predict(arch, text))
            except Exception as exc:  # a missing file or a bad checkpoint
                results.append({
                    "error": f"{type(exc).__name__}: {exc}",
                    "arch_label": DEFAULTS[arch]["label"],
                })

        head = _render_agreement(results) if len(results) > 1 else ""
        panels = "".join(_render_panel(r) for r in results)
        return head + f"<div class='bc-results'>{panels}</div>"

    with gr.Blocks(title="BanglaCare") as demo:
        gr.HTML(
            "<div id='bc-header'><h1>⚡ BanglaCare</h1>"
            "<p>Entity-aware Bangla health query triage — medical NER · "
            "4-class severity · calibrated confidence</p></div>"
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=5, elem_classes=["bc-card"]):
                gr.Markdown("<span class='bc-section-label'>Health Query</span>")
                text_input = gr.Textbox(
                    label=None, show_label=False,
                    placeholder="আপনার স্বাস্থ্য সংক্রান্ত প্রশ্ন লিখুন... (Bangla / Banglish / mixed)",
                    lines=5, elem_id="bc-input", container=False,
                )
                mode = gr.Radio(
                    choices=list(MODE_TO_ARCHS.keys()),
                    value=default_mode,
                    label="Model",
                )
                gr.HTML(
                    "<div class='bc-hint'>Models load on first use — "
                    "BanglaBERT takes ~20s, the BiLSTM ~2 min (it loads 8GB of "
                    "FastText vectors). Later queries are instant.</div>"
                )
                run_btn = gr.Button("⚡ Analyze", elem_id="bc-run-btn", size="lg")
                gr.Examples(examples=EXAMPLES, inputs=text_input, label="Try an example")

            with gr.Column(scale=6, elem_classes=["bc-card"]):
                gr.Markdown("<span class='bc-section-label'>Results</span>")
                results_out = gr.HTML(
                    value="<div class='bc-empty-note'>Enter a query and press "
                          "Analyze.</div>"
                )

        gr.HTML(f"<div class='bc-disclaimer'>🛈 {DISCLAIMER}</div>")

        run_btn.click(run, inputs=[text_input, mode], outputs=results_out)
        text_input.submit(run, inputs=[text_input, mode], outputs=results_out)

    return demo


def parse_args():
    parser = argparse.ArgumentParser(description="BanglaCare demo app")
    parser.add_argument("--arch", choices=["bilstm", "transformer", "both"], default="transformer",
                        help="which model the UI starts on (default: transformer, "
                             "because it loads in ~20s instead of ~2 min)")
    parser.add_argument("--preload", action="store_true",
                        help="build every available pipeline at startup instead of on first use")
    parser.add_argument("--model-name", default=None, help="transformer encoder id")
    parser.add_argument("--no-normalize", action="store_true",
                        help="skip the csebuetnlp per-token normalizer")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--share", action="store_true", help="create a public Gradio link")
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args()


def main():
    args = parse_args()

    import torch
    device = torch.device("cpu") if args.cpu else None

    registry = PipelineRegistry(
        device=device, model_name=args.model_name, no_normalize=args.no_normalize
    )
    available = registry.available()
    if not available:
        raise SystemExit(
            "No usable model found. Expected at least one of:\n"
            + "\n".join(f"  {cfg['checkpoint']}  +  {cfg['calibration']}"
                        for cfg in DEFAULTS.values())
        )

    print(f"available architectures: {', '.join(available)}")
    for arch, cfg in DEFAULTS.items():
        if arch not in available:
            print(f"  note: {cfg['label']} unavailable (missing checkpoint or calibration)")

    default_mode = {"bilstm": MODE_BILSTM, "transformer": MODE_TRANSFORMER,
                    "both": MODE_BOTH}[args.arch]
    # Do not open on a model whose files are missing.
    if args.arch != "both" and args.arch not in available:
        default_mode = {"bilstm": MODE_TRANSFORMER, "transformer": MODE_BILSTM}[args.arch]
        print(f"requested --arch {args.arch} is unavailable; starting on {default_mode}")

    if args.preload:
        for arch in available:
            registry.get(arch)

    demo = build_interface(registry, default_mode)
    demo.launch(
        share=args.share,
        server_port=args.port,
        theme=gr.themes.Base(
            primary_hue="cyan", secondary_hue="purple", neutral_hue="slate",
            font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"],
        ),
        css=CUSTOM_CSS,
    )


if __name__ == "__main__":
    main()
