# Polyscriptor Layout Analysis: Fast + Robust Implementation (Condensed)

Use this as a **build brief** for an implementation agent. Goal: fix the core failure mode: **line/baseline segmentation must respect text regions** (and support reading order / successors), with the **transcription_gui_plugin segmentation workflow** as the primary integration point.

---

## 1) Non-negotiable design rule

**Never segment lines globally and “attach to regions later.”**  
Instead, choose ONE of these two safe patterns:

### Pattern A — Joint layout engine (preferred)
One engine predicts **regions + baselines/lines + (optionally) reading order** together.

**Best practical option:** Kraken **baseline** segmentation (not legacy box segmentation).

### Pattern B — Region-first constrained baselines
Obtain regions first (manual or model), then detect baselines **inside each region**:

- crop-to-region → baseline segment → reproject coordinates back
- OR mask outside regions and segment only inside

Either pattern guarantees “lines stay inside regions” by construction.

---

## 2) Minimal feature set to implement (in order)

### 2.1 Core layout object (single internal representation)
Create a small schema used everywhere (CLI + transcription_gui_plugin + export):

`PageLayout`
- `regions[]`: `{id, type?, polygon}`
- `lines[]`: `{id, region_id, polygon, baseline_polyline}`
- `reading_order?`: `{region_edges[], line_edges[]}` (successor graph)

**Hard constraints to enforce in code**
- every line has a `region_id`
- baseline is a polyline (not derived from bbox)
- line polygon must be inside region polygon (or error/warn)

---

### 2.2 Layout engine interface + engines
Implement a pluggable engine interface:

`predict_layout(image, regions=None) -> PageLayout`

Engines:
1) `kraken_baseline` (primary)
2) `heuristic` (fallback, fast but low quality)

**Optional later**: `eynollah`, `pero`

---

### 2.3 transcription_gui_plugin is the “source of truth” integration
Add/adjust segmentation UX in `transcription_gui_plugin` so users can:

1) Run **Auto-segment**:
   - (default) Kraken baseline segmentation on the full page (Pattern A)
   - (optional) If regions already exist: run **per-region** segmentation (Pattern B)

2) Edit results:
   - edit region polygons
   - edit baseline polylines / line polygons
   - re-run segmentation **only for selected region(s)** (fast iteration)

3) Persist results immediately as PAGE-XML (and/or JSON sidecar).

**Must-have plugin behaviors**
- Always preserve region↔line linkage
- Keep a “dirty” flag and autosave
- Cache engine results per (image + model + params) to avoid re-running

---

### 2.4 Correct PAGE-XML export (structure matters)
Export:
- multiple `TextRegion` elements (polygon each)
- `TextLine` elements nested under their region
- `Baseline` polyline stored verbatim
- optional `ReadingOrder`:
  - region order (OrderedGroup of RegionRefIndexed)
  - line order within each region (or a single global line group if needed)

---

### 2.5 Debug overlays + sanity checks (cheap, huge ROI)
For every run, generate:
- overlay PNG: regions + baselines + line polygons (+ RO arrows if available)
- JSON report with:
  - % of each line polygon outside its region polygon
  - count of region-crossing baselines
  - line overlaps / collisions
  - warnings (deskew likely needed, etc.)

Fail fast when constraints are violated.

---

## 3) Where YOLO fits (and where it doesn’t)

### What YOLO is good for here
YOLO-style detectors are excellent for **region/block detection**:
- text blocks
- titles, headers/footers
- tables/figures (if relevant)
- marginalia blocks

There are document-layout YOLO variants and checkpoints (e.g., DocLayout-YOLO) that can give strong region boxes on diverse modern documents.

### What YOLO is *not* good for (by default)
YOLO does **not** natively produce **baselines** (polylines), which you need for HTR and PAGE-XML line structure. You still need a baseline/line engine (Kraken baseline, PERO, etc.).

### Best hybrid (if you want YOLO)
Use YOLO for regions → then run **baseline segmentation per region** (Pattern B).  
This is a clean separation of concerns:
- YOLO = region proposals
- Kraken baseline segmenter = lines/baselines inside each region

### Handwritten reality check
Most pre-trained doc-layout YOLO models are trained on printed/modern layouts. For manuscripts:
- treat them as a starting point only
- expect fine-tuning on your region classes (main text, marginalia, rubric, etc.)

---

## 4) Security + stability (practical, non-theoretical)

### 4.1 Treat ML weights as untrusted code
Many PyTorch weight formats (`.pt`) load via pickle and can execute code.  
**Rules:**
- only load weights you built yourself or trust
- store and verify SHA256 hashes
- prefer safe formats / runtimes when possible (e.g., ONNX runtime)
- run segmentation engines in an isolated process (subprocess) with minimal privileges

### 4.2 Deterministic environment
- pin dependencies (kraken, torch, ultralytics if used, shapely, opencv)
- provide a reproducible env (conda-lock / uv lock / pip-tools)
- add a “smoke test” command for CI

---

## 5) Implementation backlog (fast path)

### Phase 1 (1–2 days): fix the structural bug
- [ ] Introduce `PageLayout` schema + validators
- [ ] Implement `kraken_baseline` engine wrapper
- [ ] Update transcription_gui_plugin:
  - [ ] Auto-segment (full page)
  - [ ] Display + edit regions/lines/baselines
  - [ ] Save to PAGE-XML with correct nesting
- [ ] Add overlay + JSON sanity report

### Phase 2 (next): region-constrained segmentation
- [ ] Import regions from existing PAGE-XML
- [ ] Segment per selected region (crop+reproject)
- [ ] Re-run region-only from GUI

### Phase 3 (optional): YOLO region proposals
- [ ] Add `yolo_regions` engine (region boxes/polygons)
- [ ] Bridge: YOLO regions → kraken baselines per region
- [ ] Fine-tune YOLO on your manuscript region labels (small GT set)

---

## 6) Acceptance tests (so you know it’s actually fixed)

On a multi-column + marginalia page:
- lines never cross from one region into another (0 crossings)
- every line has correct `region_id`
- PAGE-XML has multiple regions and nested lines
- GUI can re-run segmentation for a single region and preserve others

---

## 7) Concrete CLI surface (keep it small)

Suggested commands:
- `polyscriptor segment image.png --engine kraken_baseline --out page.xml --overlay out.png`
- `polyscriptor segment image.png --engine kraken_baseline --regions-in regions.page.xml --per-region`
- (optional) `polyscriptor regions image.png --engine yolo --out regions.json`

Everything else (training, RO models) is a later extension.
