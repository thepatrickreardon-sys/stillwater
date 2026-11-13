"""Source-filter synthesis of a chanted "Om" (ahh -> ooo -> mmm) with human imperfections."""
import numpy as np, scipy.signal as sg, sys, subprocess, os

SR = 44100
BLK = 32


def smooth_noise(n, rate_hz, rng):
    """Slow random wander, unit-ish std."""
    k = max(2, int(n / SR * rate_hz) + 3)
    pts = rng.standard_normal(k)
    x = np.interp(np.linspace(0, k - 1, n), np.arange(k), pts)
    b, a = sg.butter(1, rate_hz / (SR / 2))
    return sg.filtfilt(b, a, x) * 1.4


def resonator(F, BW):
    r = np.exp(-np.pi * BW / SR); c = 2 * r * np.cos(2 * np.pi * F / SR)
    return np.array([1 - c + r * r, 0, 0]), np.array([1, -c, r * r])


def antiresonator(F, BW):
    r = np.exp(-np.pi * BW / SR); c = 2 * r * np.cos(2 * np.pi * F / SR)
    g = 1 - c + r * r
    return np.array([1, -c, r * r]) / g, np.array([1.0, 0, 0])


def tv_cascade(x, tracks, kinds):
    """Time-varying cascade. tracks: list of (F[n], BW[n]) arrays. Coeffs updated every BLK samples."""
    y = x.copy()
    for (F, BW), kind in zip(tracks, kinds):
        out = np.empty_like(y); zi = np.zeros(2)
        for s in range(0, len(y), BLK):
            e = min(len(y), s + BLK); m = (s + e) // 2
            b, a = (resonator if kind == "r" else antiresonator)(F[m], BW[m])
            out[s:e], zi = sg.lfilter(b, a, y[s:e], zi=zi)
        y = out
    return y


def env_interp(t, keys):
    """keys: list of (time, value) piecewise-linear, smoothed."""
    ts, vs = zip(*keys)
    return np.interp(t, ts, vs)


def om(f0, dur, seed, formant_scale=1.0, breath=0.75, ah_len=0.9, oo_len=None, female=False):
    rng = np.random.default_rng(seed)
    n = int((breath + dur + 1.2) * SR)
    t = np.arange(n) / SR
    v0 = breath                     # voice onset
    v1 = breath + dur               # voice end
    oo_len = oo_len if oo_len is not None else dur * 0.32
    m_on = v0 + ah_len + oo_len      # lips close

    # ---- pitch: scoop up into the note, slow drift, gentle irregular vibrato, sag at the end
    scoop = -0.9 * np.exp(-np.clip(t - v0, 0, None) / 0.09)                  # semitones
    drift = 0.18 * smooth_noise(n, 0.35, rng)
    vib_rate = 4.6 + 0.6 * smooth_noise(n, 0.2, rng)
    vib = 0.07 * (0.5 + 0.5 * smooth_noise(n, 0.3, rng).clip(-1, 1)) * np.sin(2 * np.pi * np.cumsum(vib_rate) / SR)
    sag = -0.6 * np.clip((t - (v1 - 0.6)) / 0.9, 0, 1) ** 2
    semis = scoop + drift + vib + sag
    f = f0 * 2 ** (semis / 12)

    # ---- glottal source (Rosenberg pulse, differentiated = lip radiation), per-cycle jitter/shimmer
    phase = np.zeros(n); ph = 0.0; cyc_j = 1.0; cyc_s = 1.0
    shim = np.ones(n)
    oq = 0.62 if not female else 0.7
    for i in range(n):
        ph += f[i] * cyc_j / SR
        if ph >= 1.0:
            ph -= 1.0
            cyc_j = 1 + 0.006 * rng.standard_normal()
            cyc_s = 1 + 0.05 * rng.standard_normal()
        phase[i] = ph; shim[i] = cyc_s
    tp = oq * 0.62; tn = oq - tp
    g = np.where(phase < tp, 0.5 * (1 - np.cos(np.pi * phase / tp)),
                 np.where(phase < oq, np.cos(0.5 * np.pi * (phase - tp) / tn), 0.0))
    src = np.diff(g, prepend=0) * shim
    src_rms = np.sqrt(np.mean(src ** 2))
    xf = np.clip((t - m_on) / 0.18, 0, 1); xf = 0.5 - 0.5 * np.cos(np.pi * xf)   # 0 = mouth open, 1 = closed
    # breathy aspiration, louder while the folds are open, mostly gone once the lips close
    asp = rng.standard_normal(n)
    asp = sg.lfilter(*sg.butter(2, [400 / (SR / 2), 5000 / (SR / 2)], "band"), asp)
    asp *= src_rms / np.std(asp) * (0.10 if not female else 0.15)
    src = src + asp * (0.35 + g) * (1 - 0.8 * xf)

    # ---- loudness: soft onset, swell, slight dip into the hum, long fade
    amp = env_interp(t, [(0, 0), (v0, 0), (v0 + 0.22, 0.75), (v0 + 0.9, 1.0), (m_on - 0.2, 0.95),
                         (m_on + 0.3, 0.8), (v1 - 1.6, 0.72), (v1, 0.0), (t[-1], 0)])
    amp *= 1 + 0.06 * smooth_noise(n, 0.5, rng)

    # ---- oral tract: ah -> oh -> oo, formants glide
    k = formant_scale
    A = [(720, 90), (1150, 110), (2550, 160), (3500, 250), (4400, 300)]
    O = [(470, 70), (820, 90), (2450, 150), (3400, 250), (4300, 300)]
    U = [(330, 60), (720, 80), (2350, 150), (3300, 250), (4200, 300)]
    tA, tO, tU = v0 + 0.15, v0 + ah_len, m_on - 0.05
    tracks = []
    for j in range(5):
        F = env_interp(t, [(0, A[j][0]), (tA, A[j][0]), (tO, O[j][0]), (tU, U[j][0]), (t[-1], U[j][0])]) * k
        F *= 1 + 0.01 * smooth_noise(n, 0.6, rng)
        BW = env_interp(t, [(0, A[j][1]), (tA, A[j][1]), (tO, O[j][1]), (tU, U[j][1]), (t[-1], U[j][1])])
        tracks.append((F, BW))
    oral = tv_cascade(src, tracks, "rrrrr")

    # ---- nasal tract for the hum: low murmur, anti-resonance, muffled highs
    NF = [(260 * k, 70), (1050 * k, 160), (2250 * k, 220), (3300 * k, 300)]
    ntr = [(np.full(n, F), np.full(n, B)) for F, B in NF]
    ntr.insert(1, (np.full(n, 820.0 * k), np.full(n, 140.0)))
    nasal = tv_cascade(src, ntr, "rarrr")
    nasal = sg.lfilter(*sg.butter(1, 1600 / (SR / 2)), nasal)

    # level-match the two paths over a steady patch, then crossfade over ~0.18 s
    def rms_env(x, win=0.25):
        w = int(win * SR); e = np.sqrt(sg.fftconvolve(x ** 2, np.ones(w) / w, "same").clip(1e-12))
        return e
    # hold each path at a steady loudness, so the vowel glide and the hum don't sag
    oral /= rms_env(oral); nasal /= rms_env(nasal)
    hum_level = 0.85          # a hum is a little softer than the open vowel
    y = oral * (1 - xf) + nasal * xf * hum_level
    y *= amp

    # ---- the in-breath before chanting
    bn = rng.standard_normal(n)
    bn = sg.lfilter(*sg.butter(2, [700 / (SR / 2), 4500 / (SR / 2)], "band"), bn)
    bn = sg.lfilter(*resonator(1600 * k, 600), bn)
    benv = env_interp(t, [(0, 0), (0.05, 0), (breath * 0.45, 1), (breath - 0.08, 0.55), (breath, 0), (t[-1], 0)])
    y_rms = np.sqrt(np.mean(y[int(v0 * SR):int(v1 * SR)] ** 2))
    y += bn * benv / (np.std(bn) + 1e-9) * y_rms * 0.07

    # gentle high-shelf trim + DC removal
    y = sg.lfilter(*sg.butter(2, 25 / (SR / 2), "high"), y)
    y = y / np.max(np.abs(y)) * 0.89
    return y.astype(np.float32)


def small_room(y, seed):
    """Just enough early reflections that it doesn't sound like it's inside your head; the app adds the hall."""
    rng = np.random.default_rng(seed)
    L = int(0.35 * SR); tt = np.arange(L) / SR
    ir = rng.standard_normal(L) * np.exp(-tt / 0.07) * 0.10
    ir[0] = 1.0
    out = sg.fftconvolve(y, ir)[: len(y)]
    return (out / np.max(np.abs(out)) * 0.89).astype(np.float32)


def write_mp3(y, path):
    raw = (y * 32767).astype("<i2").tobytes()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "s16le", "-ar", str(SR), "-ac", "1", "-i", "-",
                    "-codec:a", "libmp3lame", "-q:a", "4", path], input=raw, check=True)


if __name__ == "__main__":
    outdir = sys.argv[1]; os.makedirs(outdir, exist_ok=True)
    OM = 136.1   # the traditional "Om" pitch (C#3); the app retunes to the bowls
    voices = [  # name, f0, dur, seed, formant scale, ah_len, female
        ("om1", OM,         9.0, 11, 1.00, 0.95, False),
        ("om2", OM * 0.995, 10.5, 23, 0.95, 1.20, False),
        ("om3", OM * 1.004, 8.2, 37, 1.04, 0.75, False),
        ("om4", OM * 2.0,   9.2, 51, 1.17, 1.00, True),
    ]
    for name, f0, dur, seed, fs, ah, fem in voices:
        y = small_room(om(f0, dur, seed, fs, ah_len=ah, female=fem), seed)
        np.save(os.path.join(outdir, name + ".npy"), y)
        write_mp3(y, os.path.join(outdir, name + ".mp3"))
        print(name, f"{len(y)/SR:.1f}s")
