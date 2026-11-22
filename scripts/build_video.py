"""Build the <=3 min demo video from the REAL pipeline (real clips, real predictions, real results.csv).
  python scripts/build_video.py            -> PPT_and_Video/EchoGesture_Demo.mp4
Narration: Microsoft Edge neural TTS (en-IN). Only the narration text is sent to that service."""
import asyncio, csv, json, os, re, subprocess, sys, tempfile
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from echogesture import features as F, hud
from echogesture.infer import Classifier
from echogesture.landmarks import LandmarkExtractor

W, H, FPS = 1280, 720, 30
FF = imageio_ffmpeg.get_ffmpeg_exe()
VOICE = os.environ.get("VOICE", "en-IN-NeerjaExpressiveNeural")
OUT = tempfile.mkdtemp(prefix="echovid_")
FB, FR = "/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf"
font = lambda s, b=False: ImageFont.truetype(FB if b else FR, s)
BG1, BG2 = (10, 14, 26), (22, 30, 58)
CY, GR, AM, WH, DIM, CARD = (0, 180, 216), (0, 229, 176), (245, 158, 11), (255, 255, 255), (170, 182, 205), (24, 34, 66)
R = {r["metric"]: r for r in csv.DictReader(open("results.csv"))}

def base():
    g = np.linspace(0, 1, H)[:, None, None]
    img = (np.array(BG1) * (1 - g) + np.array(BG2) * g).astype(np.uint8)
    return Image.fromarray(np.repeat(img, W, 1))

def pill(d, xy, fill, r=22):
    d.rounded_rectangle(xy, r, fill=fill)

def ease(x): x = min(max(x, 0), 1); return 1 - (1 - x) ** 3

def to_cv(img): return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# ---------- scenes: each is f(t_frac, dur) -> BGR frame ----------
def title(t, dur):
    im = base(); d = ImageDraw.Draw(im); a = ease(t * 6)
    d.text((90, 150), "TRACK 02  |  SAMSUNG XR HACKATHON 2026", font=font(24, True), fill=CY)
    d.text((90, 210 - 20 * (1 - a)), "EchoGesture", font=font(120, True), fill=tuple(int(c * a) for c in WH))
    d.text((90, 370), "Sign language to words. Fully on-device.", font=font(40), fill=tuple(int(c * a) for c in DIM))
    d.rectangle((90, 440, 330, 446), fill=CY)
    d.text((90, 480), "Team Pablo Chocobar  -  Thapar Institute of Engineering & Technology", font=font(26), fill=DIM)
    return to_cv(im)

def cards_scene(heading, accent, items, foot=None):
    def f(t, dur):
        im = base(); d = ImageDraw.Draw(im)
        d.rectangle((0, 0, W, 8), fill=accent)
        d.text((80, 60), heading, font=font(54, True), fill=WH)
        n = len(items); cw = (W - 160 - (n - 1) * 30) // n
        for i, (h, body) in enumerate(items):
            a = ease(t * dur / 1.0 - i * 0.9 / 1.0 * dur / dur * 1.0 - 0.3 - i * 0.6)
            if a <= 0: continue
            x = 80 + i * (cw + 30); y = 190 + int(30 * (1 - a))
            pill(d, (x, y, x + cw, y + 330), CARD, 26)
            d.rounded_rectangle((x, y, x + cw, y + 330), 26, outline=accent, width=2)
            d.text((x + 28, y + 28), h, font=font(30, True), fill=accent)
            yy = y + 90
            for line in body:
                d.text((x + 28, yy), line, font=font(24), fill=tuple(int(c * a) for c in DIM)); yy += 38
        if foot:
            d.text((80, 600), foot, font=font(28), fill=DIM)
        return to_cv(im)
    return f

def pipeline(t, dur):
    im = base(); d = ImageDraw.Draw(im); d.rectangle((0, 0, W, 8), fill=CY)
    d.text((80, 60), "How it works", font=font(54, True), fill=WH)
    steps = [("Camera", ["live video", "frame is dropped", "right after use"]), ("Landmarks", ["MediaPipe hands", "+ upper-body pose", "49 points / frame"]),
             ("Features", ["scale + position", "invariant", "occlusion flags"]), ("INT8 SignNet", ["0.46 MB model", "ONNX Runtime", "1 CPU thread"]), ("Caption", ["word + top 5", "on screen", "or spoken"])]
    bw, gap = 200, 40
    for i, (h, body) in enumerate(steps):
        a = ease(t * 5 - i * 0.7)
        if a <= 0: continue
        x = 80 + i * (bw + gap); y = 210 + int(20 * (1 - a))
        pill(d, (x, y, x + bw, y + 230), CARD, 22)
        col = AM if i == 3 else CY
        d.rounded_rectangle((x, y, x + bw, y + 230), 22, outline=col, width=2)
        d.text((x + 18, y + 20), h, font=font(26, True), fill=col)
        for j, l in enumerate(body): d.text((x + 18, y + 80 + j * 36), l, font=font(21), fill=DIM)
        if i < 4: d.polygon([(x + bw + 8, y + 105), (x + bw + 30, y + 115), (x + bw + 8, y + 125)], fill=CY)
    if t > 0.7:
        a = ease((t - 0.7) * 5)
        pill(d, (80, 500, W - 80, 590), (10, 50, 46), 22); d.rounded_rectangle((80, 500, W - 80, 590), 22, outline=GR, width=2)
        d.text((110, 528), "AIRGAP BOUNDARY: all of this runs in one process. 0 bytes uploaded. No video stored.", font=font(27, True), fill=tuple(int(c * a) for c in GR))
    return to_cv(im)

def terminal(t, dur):
    im = base(); d = ImageDraw.Draw(im); d.rectangle((0, 0, W, 8), fill=GR)
    d.text((80, 60), "Airgap proof", font=font(54, True), fill=WH)
    pill(d, (80, 170, W - 80, 590), (8, 10, 18), 22)
    lines = ["$ python predict.py --manifest test_manifest.csv --videos clips --airgap",
             "[airgap] network disabled", "wrote 6 predictions -> predictions.jsonl", "",
             '{"clip_id": "17730", "prediction": "DRINK", "confidence": 0.2014, "latency_ms": 1.36}',
             "", "Any socket / DNS call raises AirgapViolation.", "It still works -> nothing needed the cloud."]
    total = sum(len(l) for l in lines); shown = int(total * min(t * 1.6, 1)); y = 200
    for l in lines:
        s = l[:max(shown, 0)]; shown -= len(l)
        col = GR if l.startswith("[airgap]") or l.startswith("It still") else (WH if l.startswith("$") else DIM)
        d.text((110, y), s, font=ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 22), fill=col); y += 44
        if shown <= 0: break
    return to_cv(im)

def results(t, dur):
    im = base(); d = ImageDraw.Draw(im); d.rectangle((0, 0, W, 8), fill=AM)
    d.text((80, 60), "Measured results", font=font(54, True), fill=WH)
    d.text((80, 130), "WLASL100 test split, 258 held-out clips, Apple M4 CPU, INT8, 1 thread", font=font(24), fill=DIM)
    rows = [("Top-1 accuracy", "top1_accuracy", "%", ">= 70"), ("Latency p95 (model)", "latency_p95", " ms", "<= 50"), ("Model size", "model_size", " MB", "<= 25"),
            ("Occlusion drop (30%)", "accuracy_drop_robustness", " pt", "<= 5"), ("Power / thermal", None, "", "needs a phone")]
    for i, (name, key, unit, tgt) in enumerate(rows):
        a = ease(t * 5 - i * 0.6)
        if a <= 0: continue
        y = 200 + i * 78
        pill(d, (80, y, W - 80, y + 64), CARD, 18)
        d.text((110, y + 16), name, font=font(28, True), fill=WH)
        if key:
            v = R[key]["measured"]; ok = R[key]["pass"] == "True"
            d.text((560, y + 14), f"{v}{unit}", font=font(32, True), fill=GR if ok else AM)
            d.text((790, y + 20), f"target {tgt}", font=font(22), fill=DIM)
            d.text((1010 if ok else 985, y + 20), "PASS" if ok else "BELOW TARGET", font=font(22, True), fill=GR if ok else AM)
        else:
            d.text((560, y + 18), "not measured", font=font(28, True), fill=DIM); d.text((820, y + 20), tgt, font=font(24), fill=DIM)
    return to_cv(im)

def closing(t, dur):
    im = base(); d = ImageDraw.Draw(im); d.rectangle((0, 0, W, 8), fill=CY)
    d.text((80, 60), "Honest limits, clear next steps", font=font(50, True), fill=WH)
    for x, h, items, col in ((80, "Limitations", ["Isolated words, 100-word vocabulary", "55% top-1, below the 70% target", "Small data: about 14 clips per word", "Power and thermal not yet measured"], AM),
                             (660, "Next", ["Train on the organisers' full data", "LiteRT INT8 on Galaxy NPU/GPU", "Continuous sentences", "Two-way: speech to captions"], GR)):
        pill(d, (x, 170, x + 540, 520), CARD, 26); d.rounded_rectangle((x, 170, x + 540, 520), 26, outline=col, width=2)
        d.text((x + 30, 195), h, font=font(34, True), fill=col)
        for i, l in enumerate(items): d.text((x + 30, 270 + i * 55), l, font=font(26), fill=DIM)
    d.text((80, 600), "EchoGesture  -  nothing leaves the device.", font=font(34, True), fill=CY)
    return to_cv(im)

# ---------- real demo scene ----------
def build_demo_frames(n_clips=4):
    man = json.load(open("data/manifest.json")); clf = Classifier(); ex = LandmarkExtractor()
    cand = []
    for it in [i for i in man["items"] if i["split"] == "test"]:
        lm = np.load(f"data/landmarks/{it['id']}.npy")
        if len(lm) < 20: continue
        w, p = clf.topk(F.clip_features(lm))
        if w[0].upper() == man["vocab"][it["label"]].upper(): cand.append((p[0], it, w[0].upper()))
    cand.sort(key=lambda c: -c[0]); seen, pick = set(), []
    for p, it, w in cand:
        if w not in seen: seen.add(w); pick.append((p, it, w))
        if len(pick) == n_clips: break
    frames = []
    for p, it, w in pick:
        cap = cv2.VideoCapture(it["path"]); fps = cap.get(cv2.CAP_PROP_FPS) or 25
        buf, word, conf, top5 = [], "", 0.0, []; off0 = ex._off
        src = []
        while True:
            ok, f = cap.read()
            if not ok: break
            x1, y1, x2, y2 = it["bbox"]; crop = f[max(y1, 0):y2, max(x1, 0):x2]
            if crop.shape[0] > 32 and crop.shape[1] > 32: f = crop
            lm = ex.frame(f, off0 + len(buf) * 40); buf.append(lm)
            if len(buf) >= 16 and len(buf) % 4 == 0:
                ww, pp = clf.topk(F.clip_features(np.stack(buf)))
                if pp[0] >= 0.15: word, conf, top5 = ww[0].upper(), pp[0], [x.upper() for x in ww]
            h_ = hud.draw(lm, word, conf, top5, 0.15, 30)
            canvas = np.full((H, W, 3), (26, 14, 10), np.uint8)
            hh, hw = 520, 832; canvas[70:70 + hh, 400:400 + hw] = cv2.resize(h_, (hw, hh))
            s = 330 / max(f.shape[:2]); sm = cv2.resize(f, (int(f.shape[1] * s), int(f.shape[0] * s)))
            canvas[110:110 + sm.shape[0], 40:40 + sm.shape[1]] = sm
            pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)); d = ImageDraw.Draw(pil)
            d.text((40, 70), "INPUT CLIP (test set)", font=font(20, True), fill=CY)
            d.text((40, 110 + sm.shape[0] + 14), "Ghost Mode hides this.", font=font(20), fill=DIM)
            d.text((40, 110 + sm.shape[0] + 42), "Only the skeleton is shown.", font=font(20), fill=DIM)
            d.text((400, 24), f"LIVE  ->  true word: {man['vocab'][it['label']].upper()}", font=font(24, True), fill=GR)
            frames.append(cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR))
        ex._off = off0 + len(buf) * 40 + 2000
        frames += [frames[-1]] * 20
    return frames

# ---------- TTS + muxing ----------
NARR = {
 "title": "Meet EchoGesture. Real time sign language to words, running entirely on the device. No cloud. No uploads. No stored video.",
 "problem": "Two problems block sign language translation on wearables. First, conversation is fast, so translation has to keep pace, within fifty milliseconds. Second, bystanders are in the frame, so raw video must never leave the device. And all of it must fit a tiny budget: a model under twenty megabytes.",
 "how": "Here is how it works. The camera frame goes to MediaPipe, which finds the hands and upper body as forty nine landmarks. The frame is dropped immediately. We normalise those points so distance and position matter less, and handle hidden hands with explicit flags. A tiny INT8 model, under half a megabyte, predicts the word. It runs on one CPU thread.",
 "demo": "Now a live demo on real test clips from the W-LASL dataset. On the left is the input, shown only for this video. On the right is Ghost Mode, what the user actually sees: just a glowing skeleton of the signer, with the predicted word in a Dynamic Island style caption, the top five alternatives, and the latency.",
 "airgap": "Our unique feature is the Airgap Proof. Run with the airgap flag, and every network call raises an error. The predictions still come out. That means nothing needed the cloud, and the privacy claim is something anyone can verify in one command.",
 "results": "Here are the measured results, on a held out test set of two hundred fifty eight clips. The model is zero point four six megabytes, and inference takes about point one five milliseconds. Robustness to thirty percent hand occlusion is within budget. Top one accuracy is fifty five percent, which is below our seventy percent target. We report that honestly. Power and thermal drift need a phone, so we have left them unmeasured.",
 "closing": "Our limits are clear: isolated words, a hundred word vocabulary, and a small training set. Next, we will train on the full data, move to LiteRT on the Galaxy, and extend to sentences and two way conversation. EchoGesture. Nothing leaves the device. Thank you.",
}

async def tts(text, path):
    import edge_tts
    await edge_tts.Communicate(text, VOICE, rate="-4%").save(path)

def dur_of(path):
    r = subprocess.run([FF, "-i", path], capture_output=True, text=True).stderr
    h, m, s = re.search(r"Duration: (\d+):(\d+):([\d.]+)", r).groups(); return int(h) * 3600 + int(m) * 60 + float(s)

def render(name, fn, audio, dur, frames=None):
    out = os.path.join(OUT, f"{name}.mp4"); n = int(dur * FPS)
    p = subprocess.Popen([FF, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                          "-i", audio, "-af", "apad", "-t", f"{dur:.3f}", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-ar", "44100", "-ac", "2", out], stdin=subprocess.PIPE)
    for i in range(n):
        fr = frames[min(int(i * len(frames) / n), len(frames) - 1)] if frames else fn(i / max(n - 1, 1), dur)
        p.stdin.write(np.ascontiguousarray(fr).tobytes())
    p.stdin.close(); p.wait(); return out

def main():
    scenes = [("title", title), ("problem", cards_scene("The problem", CY, [("Conversation gap", ["Signing is fast.", "Translation must keep up:", "50 ms or less."]), ("Privacy gap", ["Bystanders are in frame.", "Raw video must never", "leave the device."]), ("Edge budget", ["Under 20 MB INT8 model", "No battery drain", "No thermal throttling"])])),
              ("how", pipeline), ("demo", None), ("airgap", terminal), ("results", results), ("closing", closing)]
    demo_frames = build_demo_frames()
    outs = []
    for name, fn in scenes:
        a = os.path.join(OUT, f"{name}.mp3")
        try: asyncio.run(tts(NARR[name], a))
        except Exception as e:
            print("edge-tts failed, falling back to macOS Rishi:", e); aiff = a + ".aiff"
            subprocess.run(["say", "-v", "Rishi", "-o", aiff, NARR[name]], check=True); a = aiff
        ad = dur_of(a); dur = max(ad + 0.8, len(demo_frames) / FPS if name == "demo" else 0)
        print(f"{name}: audio {ad:.1f}s scene {dur:.1f}s", flush=True)
        outs.append(render(name, fn, a, dur, demo_frames if name == "demo" else None))
    lst = os.path.join(OUT, "list.txt"); open(lst, "w").write("".join(f"file '{o}'\n" for o in outs))
    os.makedirs("PPT_and_Video", exist_ok=True)
    subprocess.run([FF, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", "PPT_and_Video/EchoGesture_Demo.mp4"], check=True)
    print("total", dur_of("PPT_and_Video/EchoGesture_Demo.mp4"), "s -> PPT_and_Video/EchoGesture_Demo.mp4")

if __name__ == "__main__":
    main()
