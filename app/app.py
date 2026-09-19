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

# Muted, print-safe palette - the same family a clinical chart or a
# hospital-system status indicator would use, not saturated "UI accent" hues.
SEVERITY_STYLE = {
    "Emergency":     {"color": "#b42318", "bg": "#fef3f2", "border": "#fda29b", "label": "EMERGENCY"},
    "Urgent":        {"color": "#b54708", "bg": "#fffaeb", "border": "#fec84b", "label": "URGENT"},
    "Routine":       {"color": "#067647", "bg": "#ecfdf3", "border": "#6ce9a6", "label": "ROUTINE"},
    "General Query": {"color": "#175cd3", "bg": "#eff8ff", "border": "#84caff", "label": "GENERAL QUERY"},
}

ENTITY_COLORS = {
    "Symptom": "#6941c6", "Health Condition": "#c11574", "Medicine": "#0e7490",
    "Age": "#b54708", "Dosage": "#067647", "Specialist": "#3538cd",
    "Medical Procedure": "#0f766e",
}

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
  /* Warm parchment ground - not stark white, not a flat pastel tint. */
  --bc-bg: #f8f5ee;
  --bc-bg-2: #f1ebdc;
  --bc-surface: #fffefb;
  --bc-surface-tint: #faf7ef;
  --bc-border: #e7e0cb;
  --bc-border-strong: #d6cba3;
  --bc-text: #23291f;
  --bc-text-dim: #736b52;

  /* Deep emerald as the working accent (buttons, focus rings, labels)... */
  --bc-accent: #0d5c4f;
  --bc-accent-dark: #073d34;
  /* ...and a muted brass/gold as the second, used sparingly for distinction. */
  --bc-gold: #b08d3f;
  --bc-gold-soft: #e2d6ae;

  /* The masthead is its own dark surface, not a tint of the page. */
  --bc-ink: #0e2420;
  --bc-ink-2: #163b32;
}

.gradio-container {
  background: linear-gradient(165deg, var(--bc-bg) 0%, var(--bc-bg-2) 100%) !important;
  background-attachment: fixed !important;
  font-family: 'Inter', ui-sans-serif, -apple-system, sans-serif !important;
  color: var(--bc-text) !important;
}

/* ---- masthead: a dark letterhead card, not a tint of the page ---- */
#bc-header {
  background: linear-gradient(135deg, var(--bc-ink) 0%, var(--bc-ink-2) 100%);
  border-radius: 10px;
  padding: 20px 26px;
  margin-bottom: 20px;
  box-shadow: 0 6px 24px rgba(14,36,32,0.22);
  position: relative;
  overflow: hidden;
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 8px;
}
#bc-header::after {
  content: ""; position: absolute; left: 0; right: 0; bottom: 0; height: 3px;
  background: linear-gradient(90deg, var(--bc-gold) 0%, rgba(176,141,63,0) 65%);
}
#bc-header .bc-brand {
  display: flex; align-items: baseline; gap: 11px;
}
#bc-header .bc-mark {
  display: inline-block; width: 9px; height: 9px; border-radius: 50%;
  background: var(--bc-gold); box-shadow: 0 0 0 3px rgba(176,141,63,0.22);
  position: relative; top: -1px;
}
#bc-header h1 {
  font-size: 1.5em !important; font-weight: 700 !important; letter-spacing: -0.01em;
  color: #f7f4ea !important; margin: 0 !important;
}
#bc-header .bc-tagline {
  color: #a9c3ba !important; font-size: 0.88em; margin: 0 !important;
}
#bc-header .bc-badge {
  font-family: 'IBM Plex Mono', monospace; font-size: 0.7em; color: var(--bc-gold-soft);
  border: 1px solid rgba(176,141,63,0.4); border-radius: 4px; padding: 3px 9px;
  letter-spacing: 0.04em; background: rgba(176,141,63,0.08);
}

.bc-card {
  background: var(--bc-surface) !important;
  border: 1px solid var(--bc-border) !important; border-radius: 8px !important;
  padding: 20px !important; box-shadow: 0 1px 2px rgba(80,65,20,0.06);
}

.bc-section-label {
  font-size: 0.72em; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase;
  color: var(--bc-accent); margin-bottom: 12px; display: block;
  border-bottom: 1px solid var(--bc-gold-soft); padding-bottom: 8px;
}

#bc-input textarea {
  background: var(--bc-surface) !important; border: 1px solid var(--bc-border-strong) !important;
  border-radius: 6px !important; color: var(--bc-text) !important; font-size: 1em !important;
  line-height: 1.5;
}
#bc-input textarea:focus {
  border-color: var(--bc-accent) !important;
  box-shadow: 0 0 0 3px rgba(15,118,110,0.14) !important;
}

#bc-run-btn {
  background: var(--bc-accent) !important;
  border: 1px solid var(--bc-accent-dark) !important; color: white !important;
  font-weight: 600 !important; border-radius: 6px !important; box-shadow: none !important;
  transition: background 0.12s ease;
}
#bc-run-btn:hover { background: var(--bc-accent-dark) !important; }

/* radio "Model" selector - render as a segmented control, not stacked bubbles */
.gradio-container fieldset { border: none !important; }

/* ---- results ---- */
.bc-results { display: flex; gap: 16px; flex-wrap: wrap; align-items: stretch; }
.bc-panel {
  flex: 1 1 320px; min-width: 300px;
  background: var(--bc-surface); border: 1px solid var(--bc-border); border-radius: 8px;
  box-shadow: 0 1px 2px rgba(80,65,20,0.06);
}
.bc-panel-head {
  display: flex; align-items: center; justify-content: space-between;
  gap: 10px; padding: 14px 18px; border-bottom: 1px solid var(--bc-border);
  background: var(--bc-surface-tint); border-radius: 8px 8px 0 0;
}
.bc-panel-title { font-weight: 600; font-size: 0.95em; color: var(--bc-text); }
.bc-panel-meta {
  font-family: 'IBM Plex Mono', monospace; font-size: 0.74em; color: var(--bc-text-dim);
  background: var(--bc-bg); border: 1px solid var(--bc-border); border-radius: 4px;
  padding: 2px 7px;
}
.bc-panel-body { padding: 16px 18px 18px 18px; }

/* entity list rendered like an annotated report, not a tag cloud */
.bc-entity-table { width: 100%; border-collapse: collapse; font-size: 0.88em; }
.bc-entity-table tr { border-bottom: 1px solid var(--bc-border); }
.bc-entity-table tr:last-child { border-bottom: none; }
.bc-entity-table td { padding: 7px 4px; vertical-align: top; }
.bc-entity-swatch {
  width: 9px; height: 9px; border-radius: 2px; display: inline-block; margin-right: 8px;
}
.bc-entity-type-cell {
  width: 34%; font-weight: 500; color: var(--bc-text-dim); font-size: 0.92em;
  white-space: nowrap;
}
.bc-entity-text-cell { color: var(--bc-text); }
.bc-empty-note { color: var(--bc-text-dim); font-style: italic; font-size: 0.9em; padding: 4px 0; }

/* severity: a status banner with a left rule, like a clinical alert */
.bc-severity-box {
  border-left: 3px solid; border-radius: 4px; padding: 12px 16px;
  display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;
}
.bc-severity-label {
  font-weight: 700; font-size: 0.95em; letter-spacing: 0.04em;
}
.bc-severity-conf {
  font-family: 'IBM Plex Mono', monospace; font-size: 0.85em; font-weight: 500;
}

.bc-confidence-track {
  background: var(--bc-border); border-radius: 3px; height: 5px;
  width: 100%; margin: 12px 0 2px 0; overflow: hidden;
}
.bc-confidence-fill { height: 100%; border-radius: 3px; }

.bc-prob-table { width: 100%; margin-top: 12px; font-size: 0.82em; border-collapse: collapse; }
.bc-prob-table td { padding: 3px 0; }
.bc-prob-label { color: var(--bc-text-dim); width: 34%; white-space: nowrap; }
.bc-prob-track { background: var(--bc-bg); border-radius: 3px; height: 5px; overflow: hidden; }
.bc-prob-fill { height: 100%; border-radius: 3px; }
.bc-prob-pct {
  font-family: 'IBM Plex Mono', monospace; width: 42px; text-align: right;
  color: var(--bc-text-dim); padding-left: 8px; white-space: nowrap;
}

.bc-review-banner {
  margin-top: 12px; background: #fffaeb; border: 1px solid #fec84b; border-left: 3px solid #b54708;
  color: #93370d; padding: 8px 12px; border-radius: 4px; font-size: 0.84em; font-weight: 500;
}
.bc-subhead {
  font-size: 0.7em; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase;
  color: var(--bc-text-dim); margin: 16px 0 8px 0; display: block;
}
.bc-subhead:first-child { margin-top: 0; }
.bc-agree {
  margin-bottom: 14px; padding: 10px 14px; border-radius: 4px; border-left: 3px solid;
  font-size: 0.88em;
}
.bc-disclaimer {
  text-align: center; color: var(--bc-text-dim); font-size: 0.82em; padding: 16px 0 4px 0;
  border-top: 1px solid var(--bc-border); margin-top: 8px;
}
.bc-hint { color: var(--bc-text-dim); font-size: 0.82em; margin-top: 8px; }
"""


def _render_entities(entities):
    if not entities:
        return "<div class='bc-empty-note'>No medical entities detected.</div>"
    rows = []
    for e in entities:
        color = ENTITY_COLORS.get(e["type"], "#736b52")
        rows.append(
            "<tr>"
            f"<td class='bc-entity-type-cell'>"
            f"<span class='bc-entity-swatch' style='background:{color};'></span>{e['type']}</td>"
            f"<td class='bc-entity-text-cell'>{e['text']}</td>"
            "</tr>"
        )
    return f"<table class='bc-entity-table'>{''.join(rows)}</table>"


def _render_severity(severity, confidence, probs, needs_review):
    style = SEVERITY_STYLE.get(
        severity, {"color": "#736b52", "bg": "#f8f5ee", "border": "#d6cba3", "label": severity or "-"}
    )

    own_color = style["color"]
    prob_row_html = []
    for n, p in sorted(probs.items(), key=lambda kv: -kv[1]):
        bar_color = SEVERITY_STYLE.get(n, {}).get("color", own_color)
        prob_row_html.append(
            "<tr>"
            f"<td class='bc-prob-label'>{n}</td>"
            f"<td><div class='bc-prob-track'><div class='bc-prob-fill' "
            f"style='width:{p*100:.1f}%;background:{bar_color};'></div></div></td>"
            f"<td class='bc-prob-pct'>{p:.0%}</td>"
            "</tr>"
        )
    prob_rows = "".join(prob_row_html)

    html = (
        f"<div class='bc-severity-box' style='background:{style['bg']};"
        f"border-left-color:{own_color};'>"
        f"<span class='bc-severity-label' style='color:{own_color};'>{style['label']}</span>"
        f"<span class='bc-severity-conf' style='color:{own_color};'>{confidence:.1%} confidence</span>"
        f"</div>"
        f"<div class='bc-confidence-track'><div class='bc-confidence-fill' "
        f"style='width:{confidence*100:.1f}%;background:{own_color};'></div></div>"
        f"<table class='bc-prob-table'>{prob_rows}</table>"
    )
    if needs_review:
        html += (
            "<div class='bc-review-banner'>Low confidence — "
            "recommended for human review</div>"
        )
    return html


def _render_panel(result):
    """One model's full result: header, entities, severity."""
    if result.get("error"):
        return (
            f"<div class='bc-panel'><div class='bc-panel-head'>"
            f"<span class='bc-panel-title'>{result.get('arch_label', '')}</span></div>"
            f"<div class='bc-panel-body' style='color:#b42318;'>{result['error']}</div></div>"
        )

    params = result["n_params"]
    params_str = f"{params/1e6:.1f}M params" if params >= 1e6 else f"{params:,} params"
    return (
        f"<div class='bc-panel'>"
        f"<div class='bc-panel-head'>"
        f"<span class='bc-panel-title'>{result['arch_label']}</span>"
        f"<span class='bc-panel-meta'>{params_str}</span>"
        f"</div>"
        f"<div class='bc-panel-body'>"
        f"<span class='bc-subhead'>Extracted entities</span>"
        f"{_render_entities(result['entities'])}"
        f"<span class='bc-subhead'>Severity triage</span>"
        f"{_render_severity(result['severity'], result['confidence'], result['severity_probs'], result['needs_review'])}"
        f"</div></div>"
    )


def _render_agreement(results):
    """Headline agree/disagree line shown above a two-model comparison."""
    sevs = [r["severity"] for r in results if not r.get("error")]
    if len(sevs) < 2:
        return ""
    if sevs[0] == sevs[1]:
        return (
            "<div class='bc-agree' style='background:#ecfdf3;"
            "border-left-color:#067647;color:#067647;'>"
            f"<b>Both models agree:</b> {sevs[0]}</div>"
        )
    return (
        "<div class='bc-agree' style='background:#fffaeb;"
        "border-left-color:#b54708;color:#93370d;'>"
        f"<b>Models disagree:</b> {results[0]['arch_label']} indicates "
        f"<b>{sevs[0]}</b>; {results[1]['arch_label']} indicates <b>{sevs[1]}</b> — "
        "recommended for human review.</div>"
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
            "<div id='bc-header'>"
            "<div class='bc-brand'><span class='bc-mark'></span>"
            "<h1>BanglaCare</h1>"
            "<span class='bc-tagline'>Health Query Triage System</span></div>"
            "<span class='bc-badge'>NER · SEVERITY · CALIBRATED CONFIDENCE</span>"
            "</div>"
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
                    "BanglaBERT takes about 20 seconds; the BiLSTM takes about "
                    "2 minutes (it loads 8GB of FastText vectors). Later "
                    "queries are near-instant.</div>"
                )
                run_btn = gr.Button("Analyze", elem_id="bc-run-btn", size="lg")
                gr.Examples(examples=EXAMPLES, inputs=text_input, label="Try an example")

            with gr.Column(scale=6, elem_classes=["bc-card"]):
                gr.Markdown("<span class='bc-section-label'>Results</span>")
                results_out = gr.HTML(
                    value="<div class='bc-empty-note'>Enter a query and press "
                          "Analyze.</div>"
                )

        gr.HTML(f"<div class='bc-disclaimer'>{DISCLAIMER}</div>")

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
            primary_hue="blue", secondary_hue="slate", neutral_hue="slate",
            font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"],
        ),
        css=CUSTOM_CSS,
    )


if __name__ == "__main__":
    main()
