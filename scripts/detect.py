"""Detect calibration T-pose segments at both ends of every BONES-SEED clip.

Usage: python detect.py FEATURES_DIR detections.json

Reads the 120 Hz features written by bvh_features.py and writes one record per clip
with the keep range (inclusive 120 Hz frames) and diagnostics. Both the start and the end
of each clip are examined, because the relaxed rule below uses the end of a take as
evidence that its start is a calibration hold. make_manifest.py then keeps only the start
cuts.

Strict rule, for a clip start (the end is the mirror image):
  * a T-pose run (tscore > 0.85: both arms within about 32 deg of the outward shoulder
    line) lasts at least 0.1 s and starts within 0.5 s of the boundary, with the arms
    already raised (higher arm above 60 deg) at the boundary frame itself;
  * the actor is standing (Hips above 0.7 m) and still (Hips within 0.2 m horizontally of
    its boundary position) up to the end of the run;
  * the arms then come down: the higher arm drops below 60 deg within 2 s.
  The cut is where that descent finishes (elevation stops falling faster than 20 deg/s),
  at most 1 s after the 60 deg crossing.

Relaxed rule: applied only to a clip end the strict rule left whole, when the opposite end
of the same take (the clip or its mirror) has a strict cut, i.e. a take bracketed by
calibration poses. It allows a long hold (>= 1 s) to start up to 1.5 s from the boundary
and to end with the arms already dropping, drift up to 1.0 m, a descent target of
min(60, hold median - 30) deg searched over 6 s, and a looser T (tscore > 0.55) when the
arms are above 60 deg at the boundary.

Then: original and mirror (`_M`) share one cut (the larger start cut and the larger end
cut), clips that would keep < 1 s are left whole, and classes whose boundary arm spread is
choreography are left whole.
"""
import glob, json, os, re, sys
import numpy as np

FPS = 120
P = dict(tThresh=0.85, minHold=0.10, leadTol=0.50, minHips=0.70, maxDrift=0.20,
         settleWindow=2.0, settleCap=1.0, minKeep=1.0, armsDownDeg=60.0)
R = dict(longHold=1.0, longLead=1.5, relaxDrift=1.0, relaxT=0.55, dropDeg=30.0, settleWindow=6.0)
EXCLUDE = ["dancing_routine_2", "dance_basic_padeburee_270_R", "dance_basic_padeburee_R_fast", "curtsy_R"]


def motion_class(clip):
    return re.sub(r"_\d{3}__A\d+(_M)?$", "", clip)


def runs(mask):
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def settle(el, k, n):
    """Advance from the first arms-down frame while the arms are still falling fast."""
    vel = np.gradient(el.astype(np.float64)) * FPS
    cap = min(k + int(P["settleCap"] * FPS), n - 1)
    while k < cap and vel[k] < -20:
        k += 1
    return int(k)


def strict_head(ts, el, hips):
    """Strict rule at the clip start. Returns (segment, rejection)."""
    n = len(ts)
    for a, b in runs(ts > P["tThresh"]):
        if a > P["leadTol"] * FPS:
            break
        if b - a < P["minHold"] * FPS:
            continue
        if a > 0 and float(el[0]) < P["armsDownDeg"]:
            return None, dict(reason="arms_rise_after_boundary", start=int(a), end=int(b), el0=float(el[0]))
        h = hips[:b]
        if float(h[:, 1].min()) < P["minHips"]:
            return None, dict(reason="not_standing", start=int(a), end=int(b), hipsMin=float(h[:, 1].min()))
        drift = float(np.linalg.norm(h[:, [0, 2]] - h[0, [0, 2]], axis=1).max())
        if drift > P["maxDrift"]:
            return None, dict(reason="moving", start=int(a), end=int(b), drift=drift)
        below = np.flatnonzero(el[b: b + int(P["settleWindow"] * FPS)] < P["armsDownDeg"])
        if not len(below):
            return None, dict(reason="arms_stay_up", start=int(a), end=int(b))
        return dict(holdStart=int(a), holdEnd=int(b), cut=settle(el, b + int(below[0]), n), drift=drift), None
    return None, None


def relaxed_head(ts, el, hips):
    """Relaxed rule at the clip start. Returns (segment, rejection)."""
    for thr in (P["tThresh"], R["relaxT"]):
        for a, b in runs(ts > thr):
            if a > R["longLead"] * FPS:
                break
            if b - a < P["minHold"] * FPS:
                continue
            long_hold = b - a >= R["longHold"] * FPS
            if a > P["leadTol"] * FPS and not long_hold:
                continue
            if a > 0 and float(el[0]) < P["armsDownDeg"] and not long_hold:
                continue
            if thr < P["tThresh"] and float(el[0]) < P["armsDownDeg"]:
                continue
            h = hips[:b]
            if float(h[:, 1].min()) < P["minHips"]:
                return None, dict(reason="not_standing")
            drift = float(np.linalg.norm(h[:, [0, 2]] - h[0, [0, 2]], axis=1).max())
            if drift > R["relaxDrift"]:
                return None, dict(reason="moving", drift=drift)
            target = min(P["armsDownDeg"], float(np.median(el[a:b])) - R["dropDeg"])
            below = np.flatnonzero(el[b: b + int(R["settleWindow"] * FPS)] < target)
            if not len(below):
                return None, dict(reason="arms_stay_up", target=target)
            return dict(holdStart=int(a), holdEnd=int(b), cut=settle(el, b + int(below[0]), len(ts)), drift=drift,
                        tThresh=thr, descentTargetDeg=target), None
    return None, dict(reason="no_T_run")


def tail_record(t, n):
    """Express a segment found on the reversed clip in forward frame numbers."""
    out = dict(holdStart=n - t["holdEnd"], holdEnd=n - t["holdStart"], cut=n - 1 - t["cut"], drift=t["drift"])
    for k in ("tThresh", "descentTargetDeg"):
        if k in t:
            out[k] = t[k]
    return out


def strict_detect(ts, el, hips):
    n = len(ts)
    h, hr = strict_head(ts, el, hips)
    t, tr = strict_head(ts[::-1], el[::-1], hips[::-1])
    rec = dict(frames=n, keepStart=h["cut"] if h else 0, keepEnd=n - 1 - t["cut"] if t else n - 1)
    if h:
        rec["head"] = h
    if t:
        rec["tail"] = tail_record(t, n)
    if hr:
        rec["headRejected"] = hr
    if tr:
        rec["tailRejected"] = tr
    return rec


def load_candidates(feat_dir):
    """Features of clips with any loose-T frame in their first or last 3 s; no other clip
    can satisfy either rule."""
    F = {}
    for p in sorted(glob.glob(os.path.join(feat_dir, "feat_*.npz"))):
        z = np.load(p)
        for c in {k.split("|")[0] for k in z.files}:
            ts = z[c + "|tscore"].astype(np.float32)
            if (ts[:360] > R["relaxT"]).any() or (ts[-360:] > R["relaxT"]).any():
                F[c] = (ts, z[c + "|elev"].astype(np.float32), z[c + "|hips"].astype(np.float64))
    return F


def main(feat_dir, out):
    meta = json.load(open(os.path.join(feat_dir, "meta.json")))["meta"]
    F = load_candidates(feat_dir)
    res = {}
    for c, m in meta.items():
        if c in F:
            r = strict_detect(*F[c])
            assert r["frames"] == m["frames"], c
        else:
            r = dict(frames=m["frames"], keepStart=0, keepEnd=m["frames"] - 1)
        r["bvh"] = m["path"]
        res[c] = r

    # Relaxed pass, evidence from the strict pass on the clip and its mirror twin.
    strict = {c: (r["keepStart"] > 0, r["keepEnd"] < r["frames"] - 1) for c, r in res.items()}
    for c, r in res.items():
        if c not in F:
            continue
        twin = c[:-2] if c.endswith("_M") else c + "_M"
        sh, st = strict[c]
        th, tt = strict.get(twin, (False, False))
        ts, el, hips = F[c]
        n = r["frames"]
        # A start the twin already cuts gets the twin's cut through the pair union below.
        if r["keepStart"] == 0 and (st or tt) and not th:
            h, why = relaxed_head(ts, el, hips)
            if h:
                r["keepStart"], r["head"], r["headRelaxed"] = h["cut"], h, True
            else:
                r["headRelaxedRejected"] = why
        if r["keepEnd"] == n - 1 and (sh or th) and not tt:
            t, why = relaxed_head(ts[::-1], el[::-1], hips[::-1])
            if t:
                r["keepEnd"], r["tail"], r["tailRelaxed"] = n - 1 - t["cut"], tail_record(t, n), True
            else:
                r["tailRelaxedRejected"] = why

    # Original and mirror share one cut: the larger start cut and the larger end cut.
    for c, r in res.items():
        if c.endswith("_M") or c + "_M" not in res:
            continue
        m = res[c + "_M"]
        hc = max(r["keepStart"], m["keepStart"])
        tc = max(r["frames"] - 1 - r["keepEnd"], m["frames"] - 1 - m["keepEnd"])
        for x in (r, m):
            det = [x["keepStart"], x["keepEnd"]]
            x["keepStart"], x["keepEnd"] = hc, x["frames"] - 1 - tc
            if det != [x["keepStart"], x["keepEnd"]]:
                x["detectedKeep"] = det

    for c, r in res.items():
        trimmed = r["keepStart"] or r["keepEnd"] < r["frames"] - 1
        if trimmed and r["keepEnd"] - r["keepStart"] + 1 < P["minKeep"] * FPS:
            r["skipped"] = "too_short_after_trim"
            r["keepStart"], r["keepEnd"] = 0, r["frames"] - 1
        elif trimmed and motion_class(c) in EXCLUDE:
            r["excludedClass"] = dict(keep=[r["keepStart"], r["keepEnd"]])
            r["keepStart"], r["keepEnd"] = 0, r["frames"] - 1

    params = dict(P, fps=FPS, excludedClasses=EXCLUDE, relaxed=R,
                  relaxedScope="clip ends whose opposite end (original or mirror) has a strict trim")
    json.dump(dict(params=params, clips=res), open(out, "w"))
    n = sum(1 for r in res.values() if r["keepStart"] > 0 or r["keepEnd"] < r["frames"] - 1)
    print(len(res), "clips;", n, "with a start or end cut", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
