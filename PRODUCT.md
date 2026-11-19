# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

- **Primary (demo):** professors, judges, and visitors at the graduation project presentation and exhibition. They stand in front of the rig or watch on a large monitor/projector for a few minutes and need to see one wafer go from scan to verdict to process feedback without explanation from the screen.
- **Secondary (team):** the project team operating the rig. They place defects on the acrylic wafer by hand (board marker), run scans, record ground truth, and compare detection against what they see.

## Product Purpose

Wafer Defect XAI inspects a physical 270 mm wafer model end to end: a 3-axis gantry moves a camera over 44 dies, the photos are stitched into one wafer map, defects are detected per die, a trained classifier judges the wafer-level defect pattern (Donut, Edge-Ring, Random, ...), and the system explains the judgement (XAI), infers likely process causes, and proposes actions. Success in a demo means one button press produces a verdict, its reasons, and a recommended action that a visitor can follow.

## Positioning

Fully automatic from photo to process feedback: scan → stitch → detect → classify → explain → cause and action, running on a small single-board computer (Orange Pi 5) driving a self-built gantry, with the reasoning shown rather than a bare label.

## Operating Context

- Rig: Orange Pi 5 (RK3588, 3.8 GB RAM, SD card, 5 V 3 A supply), Arducam 1080p fisheye camera about 4 cm above a transparent acrylic wafer model on a white plate, ring light, H-frame gantry (travel from the origin X −122 to +122 mm, Y −150 to +122 mm, see `REACH_MM` in `~/motor_test/gantry.py`).
- Wafer: 270 mm, 44 inspected dies of 32 mm, 6 o'clock notch. Dies numbered left to right, top to bottom as seen from above.
- A scan takes about 1.5 minutes of capture and 1.5–2.5 minutes of stitching and detection. The gantry has no position sensor; after the gantry is moved by hand the origin (wafer centre) must be set again.
- The UI is viewed in a browser on a laptop and on a large monitor or projector, often from a distance.

## Capabilities and Constraints

- Flask server with server-rendered templates and plain HTML/CSS/JS; no build step and no front-end framework.
- Existing capabilities to keep: one-click scan with live progress, stitched wafer map (real view only; the template "result" view was removed 2026-11-19), die map and die detail, pattern scores with thresholds, XAI (die contributions, counterfactual), cause hypotheses with next checks, recommended actions and recipe parameters, engineer verdict logging, gantry jog/origin, ground-truth recording for validation.
- Planned removals for the 2026-11 redesign (not done yet; every other existing feature is kept, the redesign raises clarity rather than cutting features): simulation/scenario input, lighting/camera-height check, settings form, scan-bounds check and measured re-alignment, the two wafer-centre buttons, the two-page layout. Spec: `docs/specs/ui-redesign.md`.
- Board load matters: the page must not poll aggressively or render heavy effects that slow the board-hosted server.
- Terminology: Korean UI copy; industry terms stay in English (Donut, Edge-Ring, Edge-Loc, Center, Local, Random, Scratch, Near-full, particle, scratch, stain, TP/FP/FN/TN, XAI).

## Brand Commitments

- Product name on screen: **Wafer Defect XAI**. Team name LabLinked appears only where credit is appropriate.

## Evidence on Hand

- Real scans of the acrylic wafer under `captures/scans/` (clean reference wafer 20261116_145919, particle validation scan 20261116_144640 with recorded ground truth).
- Public-dataset metrics (WM-811K) in `models/report_metrics.json` and `reports/metrics/`. These are public-data numbers, not accuracy on the real wafer; the UI must not present them as real-wafer accuracy.
- Defects on the demo wafer are placed by the team (stickers, board marker). The UI and copy must not label them as staged, scenario, or demo samples, and must not invent accuracy, customers, or deployment claims.

## Judging Context

At the graduation review the main judging lens is extensibility: how the idea scales from this model-scale evaluation to real fab use, and the future expansion plan presented. The UI should look like a credible industrial inspection tool rather than show off detailed functions.

## Product Principles

1. One action, whole story: a single scan press should carry the visitor from capture to verdict to action without navigation.
2. Show the reasoning: every verdict sits next to the evidence that produced it.
3. Honest numbers: show what was measured on this wafer; never dress public-dataset or demo figures as real performance.
4. The wafer is the hero: real photos of the wafer lead; charts support them.
5. Operator controls stay out of the way: gantry and validation tools are available but folded until needed.

## Accessibility & Inclusion

Readable from a distance on a projector: large type for the verdict and key numbers, strong contrast on a light background, keyboard-operable controls, meaning not carried by colour alone (pattern and defect types also labelled in text).
