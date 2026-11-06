import os, sys, json, argparse, yaml, torch, numpy as np
from types import SimpleNamespace
from scipy.io import wavfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

ap = argparse.ArgumentParser()
ap.add_argument("--lang", required=True, choices=["English", "Telugu"])
ap.add_argument("--step", type=int, required=True)
ap.add_argument("--cfg", default=None)
ap.add_argument("--out", default="Results")
a = ap.parse_args()

R = os.path.dirname(os.path.abspath(__file__))
os.chdir(R); sys.path.insert(0, R)
from utils.model import get_model, get_vocoder, vocoder_infer
from text import _symbol_to_id
from text.symbols import symbols

for f in fm.findSystemFonts():
    if "telugu" in f.lower(): fm.fontManager.addfont(f)
tel = [x.name for x in fm.fontManager.ttflist if "telugu" in x.name.lower()]
if tel: plt.rcParams["font.family"] = [tel[0], "DejaVu Sans"]

# id: (original text, text actually given to the model)
TESTS = {
 "English": {
  "E1": ("The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings.",) * 2,
  "E2a": ("The orchestra's conduct during the ceremony was praised by the audience.",) * 2,
  "E2b": ("The researchers will conduct an experiment to evaluate the proposed method.",) * 2,
  "E3": ("The rhythm of the orchestra changed unexpectedly.",) * 2,
  "E4": ("The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time.",
         "The meeting will start at ten forty five A M on twelve November twenty twenty six, so andaroo should arrive on time."),
 },
 "Telugu": {
  "T1": ("న్యూరోఫార్మకోలాజికల్ పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని విశ్లేషించారు.",) * 2,
  "T2a": ("తాడుపై అతని పట్టు చాలా బలంగా ఉంది.",) * 2,
  "T2b": ("ఆమె పట్టుచీరను ధరించి వేడుకకు వచ్చింది.",) * 2,
  "T3": ("పులి అడవిలో నిశ్శబ్దంగా నడిచింది.",) * 2,
  "T4": ("ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉంచండి.",
         "ఈ సమావేశం రెండు వేల ఇరవై ఆరు నవంబర్ పన్నెండవ తేదీ ఉదయం పది గంటల నలభై ఐదు నిమిషాలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉంచండి."),
 },
}

cfg = a.cfg or f"config/{a.lang}"
L = lambda p: yaml.load(open(p, encoding="utf-8"), Loader=yaml.FullLoader)
pc, mc, tc = L(f"{cfg}/preprocess.yaml"), L(f"{cfg}/model.yaml"), L(f"{cfg}/train.yaml")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model(SimpleNamespace(restore_step=a.step), (pc, mc, tc), device, train=False).eval()
vocoder = get_vocoder(mc, device)

st = json.load(open(os.path.join(pc["path"]["preprocessed_path"], "stats.json")))  # [min, max, mean, std]
pp, ee = pc["preprocessing"]["pitch"], pc["preprocessing"]["energy"]
def denorm(x, key, norm): return x * st[key][3] + st[key][2] if norm else x

sr = pc["preprocessing"]["audio"]["sampling_rate"]
hop = pc["preprocessing"]["stft"]["hop_length"]
base = os.path.join(a.out, a.lang)
for d in ["test_generated_audios", "variance_adaptor_plots/duration", "variance_adaptor_plots/pitch", "variance_adaptor_plots/energy"]:
    os.makedirs(os.path.join(base, d), exist_ok=True)

def to_ids(text):
    dropped = []
    if a.lang == "English":
        from synthesize import preprocess_english
        return np.array(preprocess_english(text, pc)).astype(np.int64), dropped
    ids = []
    for w in text.split():
        for ch in w:
            if "@" + ch in _symbol_to_id: ids.append(_symbol_to_id["@" + ch])
            else: dropped.append(ch)
        ids.append(_symbol_to_id["@sp"])
    return np.array(ids[:-1], dtype=np.int64), dropped

summary = {}
for tid, (orig, txt) in TESTS[a.lang].items():
    ids, dropped = to_ids(txt)
    labels = [symbols[i].lstrip("@") for i in ids]
    with torch.no_grad():
        o = model(torch.tensor([0]).to(device), torch.from_numpy(ids)[None].to(device),
                  torch.tensor([len(ids)]).to(device), len(ids))
        mel_len = int(o[9][0])
        wav = vocoder_infer(o[1].transpose(1, 2), vocoder, mc, pc, lengths=(o[9] * hop).cpu().numpy())[0]
    wavfile.write(f"{base}/test_generated_audios/{tid}.wav", sr, wav)

    dur = o[5][0].cpu().numpy() * hop / sr * 1000          # ms per input symbol
    p = o[2][0].cpu().numpy(); e = o[3][0].cpu().numpy()
    p_ph, e_ph = pp["feature"] == "phoneme_level", ee["feature"] == "phoneme_level"
    p = denorm(p[:len(ids)] if p_ph else p[:mel_len], "pitch", pp["normalization"])
    e = denorm(e[:len(ids)] if e_ph else e[:mel_len], "energy", ee["normalization"])

    def draw(kind, y, phone_level, ylabel):
        fig, ax = plt.subplots(figsize=(max(8, 0.28 * len(ids)), 3.6))
        if phone_level:
            (ax.bar if kind == "duration" else ax.plot)(range(len(y)), y)
            ax.set_xticks(range(len(y))); ax.set_xticklabels(labels, rotation=90, fontsize=7)
            ax.set_xlabel("input symbol")
        else:
            ax.plot(np.arange(len(y)) * hop / sr, y); ax.set_xlabel("time (s)")
        ax.set_ylabel(ylabel); ax.set_title(f"{tid}: predicted {kind}"); ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(f"{base}/variance_adaptor_plots/{kind}/{tid}.png", dpi=150); plt.close(fig)
    draw("duration", dur, True, "duration (ms)")
    draw("pitch", p, p_ph, "F0 (Hz)")
    draw("energy", e, e_ph, "energy")

    summary[tid] = dict(original=orig, model_text=txt, n_symbols=len(ids), dropped_chars=sorted(set(dropped)),
                        seconds=round(len(wav) / sr, 2),
                        mean_f0_hz=round(float(p[p > 0].mean()), 1) if (p > 0).any() else 0,
                        mean_energy=round(float(e.mean()), 3),
                        longest_symbols=[labels[i] for i in np.argsort(-dur)[:3]])
    print(tid, summary[tid])
json.dump(summary, open(f"{base}/summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
