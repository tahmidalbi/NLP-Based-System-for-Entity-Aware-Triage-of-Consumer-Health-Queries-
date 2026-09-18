"""
BanglaCare demo (guide section 17.2).

One text box, one Run button, extracted entities, severity + calibrated
confidence, and a safety disclaimer. Everything (checkpoint, calibration,
both FastText models) loads once at startup; each query afterwards is fast.

Usage
-----
  python app/app.py
  python app/app.py --cpu            # force CPU even if a GPU is present
  python app/app.py --share          # public Gradio link (e.g. demoing from Kaggle)

Requires gradio, which is not in requirements.txt (that file covers the
training pipeline, Phases 1-7) - install separately:
  pip install gradio
"""

import argparse
from pathlib import Path

import gradio as gr

from inference import DISCLAIMER, BanglaCarePipeline

ROOT = Path(__file__).resolve().parents[1]

# The guide's own worked example (1.5), plus a few covering different
# severities and Banglish/code-switching, so a first-time viewer can click
# instead of typing.
EXAMPLES = [
    "আমার বয়স ২৭ বছর। হঠাৎ বুকে অনেক ব্যথা হচ্ছে এবং শ্বাস নিতে কষ্ট হচ্ছে। Napa খেয়েছি।",
    "গত দুইদিন ধরে সর্দি কাশি হচ্ছে, জ্বর নেই।",
    "ডায়াবেটিসের রোগীর জন্য কোন specialist দেখানো উচিত?",
    "বাচ্চার জ্বর ১০৩ ডিগ্রি পার হয়ে গেছে, চোখ বন্ধ করে ঝিমাচ্ছে।",
    "amar matha betha korche 2 din dhore, ki korbo?",
]

# Two independent palettes on purpose: severity uses a universal
# red/amber/green/blue urgency scale, entities use a distinct violet/cyan
# family so the two visual languages never collide on screen together.
SEVERITY_STYLE = {
    "Emergency":     {"color": "#ff4d6d", "glow": "rgba(255,77,109,0.35)", "icon": "🚨"},
    "Urgent":        {"color": "#ffa53d", "glow": "rgba(255,165,61,0.35)", "icon": "⚠️"},
    "Routine":       {"color": "#3ddc97", "glow": "rgba(61,220,151,0.35)", "icon": "🩺"},
    "General Query": {"color": "#4cc9f0", "glow": "rgba(76,201,240,0.35)", "icon": "💬"},
}

ENTITY_COLORS = {
    "Symptom":            "#8b5cf6",
    "Health Condition":   "#ec4899",
    "Medicine":           "#06b6d4",
    "Age":                "#f59e0b",
    "Dosage":             "#10b981",
    "Specialist":         "#6366f1",
    "Medical Procedure":  "#14b8a6",
}

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&display=swap');

:root {
  --bc-bg-0: #0b0f19;
  --bc-bg-1: #121826;
  --bc-bg-2: #1a2333;
  --bc-border: #2a3550;
  --bc-text: #e6ebf5;
  --bc-text-dim: #93a1bf;
  --bc-accent: #06b6d4;
  --bc-accent-2: #8b5cf6;
}

.gradio-container {
  background: radial-gradient(circle at 15% 0%, #16213a 0%, var(--bc-bg-0) 45%) !important;
  font-family: 'Inter', ui-sans-serif, sans-serif !important;
  color: var(--bc-text) !important;
}

#bc-header {
  text-align: center;
  padding: 8px 0 4px 0;
}
#bc-header h1 {
  font-family: 'Space Grotesk', sans-serif !important;
  font-size: 2.4em !important;
  font-weight: 700 !important;
  margin-bottom: 2px !important;
  background: linear-gradient(90deg, #4cc9f0, #8b5cf6 60%, #ff4d6d);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  letter-spacing: 0.5px;
}
#bc-header p {
  color: var(--bc-text-dim) !important;
  font-size: 1.02em;
  margin-top: 0 !important;
}

.bc-card {
  background: linear-gradient(180deg, var(--bc-bg-1), var(--bc-bg-2)) !important;
  border: 1px solid var(--bc-border) !important;
  border-radius: 16px !important;
  padding: 18px !important;
  box-shadow: 0 8px 30px rgba(0,0,0,0.35);
}

.bc-section-label {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 0.78em;
  font-weight: 700;
  letter-spacing: 1.5px;
  text-transform: uppercase;
  color: var(--bc-accent);
  margin-bottom: 10px;
  display: block;
}

#bc-input textarea {
  background: var(--bc-bg-0) !important;
  border: 1px solid var(--bc-border) !important;
  border-radius: 12px !important;
  color: var(--bc-text) !important;
  font-size: 1.05em !important;
}
#bc-input textarea:focus {
  border-color: var(--bc-accent) !important;
  box-shadow: 0 0 0 3px rgba(6,182,212,0.15) !important;
}

#bc-run-btn {
  background: linear-gradient(90deg, var(--bc-accent), var(--bc-accent-2)) !important;
  border: none !important;
  color: white !important;
  font-weight: 600 !important;
  border-radius: 10px !important;
  box-shadow: 0 4px 18px rgba(6,182,212,0.30);
  transition: transform 0.15s ease, box-shadow 0.15s ease;
}
#bc-run-btn:hover {
  transform: translateY(-1px);
  box-shadow: 0 6px 24px rgba(139,92,246,0.40);
}

.bc-entity-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 12px;
  margin: 4px 6px 4px 0;
  border-radius: 999px;
  font-size: 0.88em;
  font-weight: 500;
  white-space: nowrap;
}
.bc-entity-chip .bc-dot {
  width: 7px; height: 7px; border-radius: 50%; display: inline-block;
}
.bc-entity-type {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 0.75em;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  opacity: 0.85;
}
.bc-empty-note {
  color: var(--bc-text-dim);
  font-style: italic;
  font-size: 0.92em;
}

.bc-severity-card {
  border-radius: 14px;
  padding: 18px 20px;
  border: 1px solid var(--bc-border);
  position: relative;
  overflow: hidden;
}
.bc-severity-label {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 1.6em;
  font-weight: 700;
  display: flex;
  align-items: center;
  gap: 10px;
}
.bc-confidence-track {
  background: rgba(255,255,255,0.08);
  border-radius: 999px;
  height: 8px;
  width: 100%;
  margin: 12px 0 4px 0;
  overflow: hidden;
}
.bc-confidence-fill {
  height: 100%;
  border-radius: 999px;
}
.bc-confidence-text {
  font-size: 0.85em;
  color: var(--bc-text-dim);
  font-family: 'Space Grotesk', sans-serif;
}
.bc-prob-row {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 6px;
  font-size: 0.82em;
}
.bc-prob-label { width: 112px; color: var(--bc-text-dim); flex-shrink: 0; }
.bc-prob-track {
  flex: 1; background: rgba(255,255,255,0.06); border-radius: 999px; height: 6px; overflow: hidden;
}
.bc-prob-fill { height: 100%; border-radius: 999px; opacity: 0.85; }
.bc-prob-pct { width: 42px; text-align: right; color: var(--bc-text-dim); flex-shrink: 0; }

.bc-review-banner {
  margin-top: 14px;
  background: rgba(255,165,61,0.12);
  border: 1px solid rgba(255,165,61,0.4);
  color: #ffcb8a;
  padding: 10px 14px;
  border-radius: 10px;
  font-size: 0.9em;
  display: flex;
  align-items: center;
  gap: 8px;
}

.bc-disclaimer {
  text-align: center;
  color: var(--bc-text-dim);
  font-size: 0.85em;
  padding: 10px 0 4px 0;
}
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


def _render_severity(severity, confidence, probs):
    style = SEVERITY_STYLE.get(severity, {"color": "#93a1bf", "glow": "transparent", "icon": "•"})
    color = style["color"]

    prob_rows = "".join(
        f"<div class='bc-prob-row'>"
        f"<span class='bc-prob-label'>{SEVERITY_STYLE.get(name, {}).get('icon', '')} {name}</span>"
        f"<span class='bc-prob-track'><span class='bc-prob-fill' "
        f"style='width:{p*100:.1f}%;background:{SEVERITY_STYLE.get(name, {}).get('color', color)};'>"
        f"</span></span>"
        f"<span class='bc-prob-pct'>{p:.0%}</span></div>"
        for name, p in sorted(probs.items(), key=lambda kv: -kv[1])
    )

    return (
        f"<div class='bc-severity-card' style='background:radial-gradient(circle at 0% 0%,"
        f"{style['glow']},transparent 70%);'>"
        f"<div class='bc-severity-label' style='color:{color};'>"
        f"<span>{style['icon']}</span><span>{severity}</span></div>"
        f"<div class='bc-confidence-track'><div class='bc-confidence-fill' "
        f"style='width:{confidence*100:.1f}%;background:linear-gradient(90deg,{color},{color}aa);'></div></div>"
        f"<div class='bc-confidence-text'>CALIBRATED CONFIDENCE &nbsp;{confidence:.1%}</div>"
        f"<div style='margin-top:14px;'>{prob_rows}</div>"
        f"</div>"
    )


def build_interface(pipeline):
    def run(text):
        result = pipeline.predict(text)

        if result["error"]:
            error_html = (
                f"<div class='bc-severity-card' style='border-color:#ff4d6d55;'>"
                f"<div style='color:#ff9eb0;'>⚠️ {result['error']}</div></div>"
            )
            return "<div class='bc-empty-note'>—</div>", error_html

        entities_html = _render_entities(result["entities"])
        severity_html = _render_severity(
            result["severity"], result["confidence"], result["severity_probs"]
        )
        if result["needs_review"]:
            severity_html += (
                "<div class='bc-review-banner'>🔍 <b>Low confidence — "
                "recommend human review</b></div>"
            )

        return entities_html, severity_html

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
                    label=None,
                    show_label=False,
                    placeholder="আপনার স্বাস্থ্য সংক্রান্ত প্রশ্ন লিখুন... (Bangla / Banglish / mixed)",
                    lines=5,
                    elem_id="bc-input",
                    container=False,
                )
                run_btn = gr.Button("⚡ Analyze", elem_id="bc-run-btn", size="lg")
                gr.Examples(examples=EXAMPLES, inputs=text_input, label="Try an example")

            with gr.Column(scale=6):
                with gr.Column(elem_classes=["bc-card"]):
                    gr.Markdown("<span class='bc-section-label'>Extracted Entities</span>")
                    entities_out = gr.HTML(value=_render_entities([]))

                with gr.Column(elem_classes=["bc-card"]):
                    gr.Markdown("<span class='bc-section-label'>Severity Triage</span>")
                    severity_out = gr.HTML()

        gr.HTML(f"<div class='bc-disclaimer'>🛈 {DISCLAIMER}</div>")

        outputs = [entities_out, severity_out]
        run_btn.click(run, inputs=text_input, outputs=outputs)
        text_input.submit(run, inputs=text_input, outputs=outputs)

    return demo


def parse_args():
    parser = argparse.ArgumentParser(description="BanglaCare demo app")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "best_joint.pt")
    parser.add_argument("--calibration", type=Path, default=ROOT / "checkpoints" / "calibration.json")
    parser.add_argument("--general-model", type=Path, default=ROOT / "embeddings" / "cc.bn.300.bin")
    parser.add_argument("--medical-model", type=Path, default=ROOT / "embeddings" / "medical_fasttext.bin")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--share", action="store_true", help="create a public Gradio link")
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args()


def main():
    args = parse_args()
    for path, name in [
        (args.checkpoint, "checkpoint"),
        (args.calibration, "calibration"),
        (args.general_model, "general FastText model"),
        (args.medical_model, "medical FastText model"),
    ]:
        if not path.exists():
            raise SystemExit(f"Missing {name} at {path}")

    import torch
    device = torch.device("cpu") if args.cpu else None

    pipeline = BanglaCarePipeline(
        checkpoint_path=args.checkpoint,
        calibration_path=args.calibration,
        general_model_path=args.general_model,
        medical_model_path=args.medical_model,
        device=device,
    )
    demo = build_interface(pipeline)
    demo.launch(
        share=args.share,
        server_port=args.port,
        theme=gr.themes.Base(
            primary_hue="cyan",
            secondary_hue="purple",
            neutral_hue="slate",
            font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"],
        ),
        css=CUSTOM_CSS,
    )


if __name__ == "__main__":
    main()
