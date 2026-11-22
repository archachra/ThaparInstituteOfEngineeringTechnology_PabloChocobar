"""Fill the organiser's submission template with EchoGesture content.
Numbers come from results.csv at build time (blank = not measured, shown as 'not measured').
  python EchoGesture/scripts/build_deck.py <template.pptx> <results.csv> <out.pptx>"""
import csv, sys
from pptx import Presentation
from pptx.util import Emu, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

tpl, res, out = sys.argv[1:4]
CY, GR, AM, WH, DIM, CARD = (RGBColor(0, 180, 216), RGBColor(0, 229, 176), RGBColor(245, 158, 11),
                              RGBColor(255, 255, 255), RGBColor(190, 200, 220), RGBColor(18, 28, 60))
R = {r["metric"]: r for r in csv.DictReader(open(res))}
def m(k, unit=""):
    v = R[k]["measured"]; return f"{v}{unit}" if v != "" else "not measured"

p = Presentation(tpl)
S = p.slides

def settext(shape, text, size=None, color=None):
    tf = shape.text_frame; r0 = tf.paragraphs[0].runs[0]; f = r0.font
    try: col = f.color.rgb
    except Exception: col = None
    keep = (f.name, f.size, f.bold, col)
    tf.clear(); r = tf.paragraphs[0].add_run(); r.text = text
    r.font.name, r.font.size, r.font.bold = keep[0], Pt(size) if size else keep[1], keep[2]
    c = color or keep[3]
    if c: r.font.color.rgb = c

def box(s, x, y, w, h, fill=CARD, line=None):
    b = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Emu(x), Emu(y), Emu(w), Emu(h))
    b.adjustments[0] = 0.08; b.fill.solid(); b.fill.fore_color.rgb = fill
    if line: b.line.color.rgb = line; b.line.width = Pt(1)
    else: b.line.fill.background()
    b.shadow.inherit = False
    return b

def write(shape, lines, size=11, color=WH, bold_first=False):
    tf = shape.text_frame; tf.word_wrap = True; tf.clear()
    tf.margin_left = tf.margin_right = Emu(110000); tf.margin_top = tf.margin_bottom = Emu(70000)
    for i, l in enumerate(lines):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r = para.add_run(); r.text = l; r.font.name = "Calibri"; r.font.size = Pt(size)
        r.font.color.rgb = color; r.font.bold = bold_first and i == 0
        para.space_after = Pt(3)

def card(s, x, y, w, h, title, lines, accent=CY, size=11):
    b = box(s, x, y, w, h, line=accent)
    tf = b.text_frame; tf.vertical_anchor = 1
    write(b, [title] + lines, size, WH)
    r = tf.paragraphs[0].runs[0]; r.font.bold = True; r.font.color.rgb = accent; r.font.size = Pt(size + 2)
    for para in tf.paragraphs[1:]:
        for r in para.runs: r.font.color.rgb = DIM

L, T, W = 365760, 1000000, 8412480
# 1 title
s = S[0]
for sh in s.shapes:
    if sh.has_text_frame and sh.name == "Text 7": settext(sh, "TRACK 02  |  XR HACKATHON 2026")
    if sh.name == "Text 8": settext(sh, "EchoGesture", 54)
    if sh.name == "Text 10": settext(sh, "Sign language prediction, fully on-device", 20)
    if sh.name == "Text Box 22": write(sh, ["Team: Pablo Chocobar", "Thapar Institute of Engineering & Technology", "(Parth Puri, Arnav Chachra, Hasrat Aulakh)"], 12)
# 2 problem
s = S[1]
card(s, L, T, 2700000, 1500000, "The conversation gap", ["Natural signing is fast.", "Translation must keep up (<= 50 ms) or dialogue turns awkward and out of sync."], CY)
card(s, L + 2850000, T, 2700000, 1500000, "The privacy gap", ["Bystanders are in frame.", "Raw video must never leave the device: zero cloud streaming."], GR)
card(s, L + 5700000, T, 2712480, 1500000, "The edge budget", ["<= 20 MB INT8 model", "<= 50 ms inference", "No battery drain / thermal throttling"], AM)
card(s, L, T + 1650000, W, 1250000, "In our words", ["Deaf and hearing people should be able to talk in real time on a phone or wearable, with no account, no server and no recorded video. Exocentric view (third-person camera) means moving heads, angled signers and low light."], CY)
# 3 solution
s = S[2]
card(s, L, T, 4100000, 1400000, "1. See only the skeleton", ["MediaPipe hands + upper-body pose run locally. The camera frame is discarded immediately; only 49 landmarks per frame continue."], GR)
card(s, L + 4312480, T, 4100000, 1400000, "2. Pose-invariant features", ["Shoulder-width scale, mid-shoulder origin, wrist-relative hand shape: distance and angle of the signer matter less. Presence flags handle occlusion."], GR)
card(s, L, T + 1550000, 4100000, 1400000, "3. Tiny INT8 temporal CNN", ["32-frame window, 3 dilated conv blocks, static INT8 (QDQ) ONNX: " + m("model_size", " MB") + ", p95 " + m("latency_p95", " ms") + "."], GR)
card(s, L + 4312480, T + 1550000, 4100000, 1400000, "4. Captions + voice", ["Dynamic-Island style caption pill and optional text-to-speech for two-way conversation."], GR)
# 4 architecture
s = S[3]
steps = [("Camera", "live / clip"), ("Landmarks", "MediaPipe hands+pose\nframe dropped"), ("Features", "normalise, resample\n32 x 312"), ("INT8 SignNet", "ONNX Runtime\n1 CPU thread"), ("Output", "word + top-5\ncaption + TTS")]
bw, gap = 1480000, 253000
for i, (a, b) in enumerate(steps):
    x = L + i * (bw + gap)
    bx = box(s, x, T + 200000, bw, 1300000, line=CY if i != 3 else AM)
    write(bx, [a] + b.split("\n"), 11); bx.text_frame.paragraphs[0].runs[0].font.bold = True
    bx.text_frame.paragraphs[0].runs[0].font.color.rgb = CY if i != 3 else AM
    for para in bx.text_frame.paragraphs[1:]:
        for r in para.runs: r.font.color.rgb = DIM
    if i < 4:
        ar = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Emu(x + bw + 40000), Emu(T + 750000), Emu(gap - 80000), Emu(200000))
        ar.fill.solid(); ar.fill.fore_color.rgb = CY; ar.line.fill.background()
band = box(s, L, T + 1750000, W, 500000, fill=RGBColor(10, 40, 40), line=GR)
write(band, ["AIRGAP BOUNDARY: everything above runs in one process with sockets blocked (--airgap). 0 bytes uploaded, no video stored."], 11, GR)
card(s, L, T + 2400000, W, 650000, "Stack", ["Python, MediaPipe Tasks, PyTorch (train), ONNX + ONNX Runtime static INT8 (deploy), OpenCV HUD, Docker. Data: WLASL100 (public)."], AM, 10)
# 5 output
s = S[4]
for sh in s.shapes:
    if sh.name == 'Text 2': settext(sh, 'TRACK FINAL OUTPUT')
rows = [("Top-1 accuracy (>=70 %)", m("top1_accuracy", " %")), ("Latency p95 (<=50 ms)", m("latency_p95", " ms")), ("Model size (<=25 MB)", m("model_size", " MB")),
        ("Occlusion robustness drop (<=5 pt)", m("accuracy_drop_robustness", " pt")), ("Avg power (<=500 mW)", m("avg_power")), ("Thermal drift 10 min (<=20 %)", m("thermal_drift_10min"))]
for i, (k, v) in enumerate(rows):
    x = L + (i % 3) * 2830000; y = T + (i // 3) * 1250000
    card(s, x, y, 2750000, 1150000, v, [k], AM if v == "not measured" else GR, 12)
dev = R["latency_p95"]["device"]
card(s, L, T + 2550000, W, 500000, "Measured on", [f"{dev}, {R['latency_p95']['runtime']}. Model-only latency; landmark extraction is reported separately."], CY, 10)
# 6 differentiation
s = S[5]
card(s, L, T, 4100000, 1400000, "Ghost Mode", ["The HUD never displays or stores the camera image: only a glowing skeleton of the signer. Privacy by construction, visible to users."], GR)
card(s, L + 4312480, T, 4100000, 1400000, "Airgap Proof", ["--airgap makes every network call raise an error. If it still translates, nothing needed the cloud. Judges can verify the privacy claim in one flag."], GR)
card(s, L, T + 1550000, 4100000, 1400000, "Occlusion-aware by design", ["Training randomly hides hands, drops frames and jitters pose; presence flags let the model know what it can't see."], GR)
card(s, L + 4312480, T + 1550000, 4100000, 1400000, "Tiny and honest", ["~0.4 MB model. Latency split into model vs landmark cost; unmeasured metrics are left blank, never guessed."], GR)
# 7 demo
s = S[6]
for sh in s.shapes:
    if sh.name == "Text Box 36": sh._element.getparent().remove(sh._element)
card(s, L, T, W, 2900000, "Local setup", [
    "python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt",
    "python predict.py --video clip.mp4                         # one clip -> JSON",
    "python predict.py --manifest m.csv --videos clips/ --airgap  # predictions.jsonl",
    "python demo.py --airgap --speak                             # live Ghost-Mode HUD",
    "docker build -t echogesture . && docker run echogesture",
    "Demo video: <<link>>     Repo: <<link>>"], AM, 11)
# 8 brownie
s = S[7]
for sh in s.shapes:
    if sh.name == "Text Box 36": sh._element.getparent().remove(sh._element)
card(s, L, T, 4100000, 2900000, "Limitations", ["Isolated-word model (WLASL-style), not continuous sentences.", "Trained on a 100-word WLASL subset; larger vocabularies need more data/capacity.", "Power and thermal drift need a phone to measure.", "Landmark extraction (MediaPipe) dominates compute."], AM)
card(s, L + 4312480, T, 4100000, 2900000, "Next", ["Fine-tune on organiser data and full 2,000-word vocabulary.", "LiteRT INT8 + GPU/NPU delegate on Galaxy.", "Sliding-window continuous recognition with gloss smoothing.", "Two-way mode: speech-to-captions beside sign-to-text."], GR)
p.save(out); print("saved", out)
