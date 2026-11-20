"""Per-frame T-pose features for every SOMA Uniform BVH in BONES-SEED.

Streams soma_uniform.tar (decompressed on stdin), parses each BVH, runs forward
kinematics on the spine and arm chains only, and writes, per clip at the native 120 Hz:

  tscore  min over both arms of cos(shoulder->wrist, outward shoulder line), float16
          (1 = arm straight out to the side; > 0.85 is within about 32 degrees)
  elev    max over both arms of the shoulder->wrist angle from straight down (deg), float16
  hips    Hips position (m, BVH axes, Y up), float32

Output: OUT_DIR/feat_NNNN.npz (keys "<clip>|<feature>", 2,000 clips per file) and
OUT_DIR/meta.json (frame count, frame time and archive path per clip, plus parse errors).

Usage: rapidgzip -d -c soma_uniform.tar.gz | python bvh_features.py OUT_DIR
Env:   NPROC (worker processes, default all CPUs); NAMES (optional JSON list of clip
       names to restrict extraction to, for spot checks).
"""
import os, sys, tarfile, json
import numpy as np
from multiprocessing import Pool

CHAIN = ["Root", "Hips", "Spine1", "Spine2", "Chest",
         "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
         "RightShoulder", "RightArm", "RightForeArm", "RightHand"]


def parse_header(lines):
    joints, stack, cur = {}, [], None
    col = 0
    i = 0
    while True:
        t = lines[i].split()
        i += 1
        if not t:
            continue
        if t[0] in ("ROOT", "JOINT"):
            cur = {"name": t[1], "parent": stack[-1] if stack else None}
            joints[t[1]] = cur
        elif t[0] == "End":
            cur = {"name": None, "parent": stack[-1]}
        elif t[0] == "{":
            stack.append(cur["name"])
        elif t[0] == "}":
            stack.pop()
        elif t[0] == "OFFSET" and cur["name"]:
            cur["offset"] = np.array(t[1:4], dtype=np.float64)
        elif t[0] == "CHANNELS":
            cur["channels"] = t[2:]
            cur["col"] = col
            col += int(t[1])
        elif t[0] == "MOTION":
            break
    nframes = int(lines[i].split()[1])
    dt = float(lines[i + 1].split()[2])
    return joints, col, nframes, dt, i + 2


def rot(axis, ang):
    c, s = np.cos(ang), np.sin(ang)
    o, z = np.ones_like(ang), np.zeros_like(ang)
    if axis == "X":
        m = [[o, z, z], [z, c, -s], [z, s, c]]
    elif axis == "Y":
        m = [[c, z, s], [z, o, z], [-s, z, c]]
    else:
        m = [[c, -s, z], [s, c, z], [z, z, o]]
    return np.stack([np.stack(r, -1) for r in m], -2)


def features(raw):
    text = raw.decode("ascii", "replace")
    lines = text.split("\n")
    joints, ncol, nf, dt, start = parse_header(lines)
    data = np.array(" ".join(lines[start:start + nf]).split(), dtype=np.float64)
    data = data[: (len(data) // ncol) * ncol].reshape(-1, ncol)
    n = len(data)
    R, P = {}, {}
    for name in CHAIN:
        j = joints[name]
        par = j["parent"]
        off = j["offset"].copy()
        loc = np.broadcast_to(off, (n, 3)).copy()
        r = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
        for k, ch in enumerate(j["channels"]):
            v = data[:, j["col"] + k]
            if ch.endswith("position"):
                loc[:, "XYZ".index(ch[0])] = v  # position channels replace the offset
            else:
                r = r @ rot(ch[0], np.radians(v))
        if par is None:
            P[name], R[name] = loc, r
        else:
            P[name] = P[par] + np.einsum("nij,nj->ni", R[par], loc)
            R[name] = R[par] @ r
    up = np.array([0.0, 1.0, 0.0])
    lat = P["LeftArm"] - P["RightArm"]
    lat[:, 1] = 0
    lat /= np.linalg.norm(lat, axis=1, keepdims=True) + 1e-9
    sc, el = [], []
    for side, sgn in (("Left", 1.0), ("Right", -1.0)):
        v = P[side + "Hand"] - P[side + "Arm"]
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
        sc.append(sgn * np.einsum("ni,ni->n", v, lat))
        el.append(np.degrees(np.arccos(np.clip(-v @ up, -1, 1))))
    return dict(tscore=np.minimum(*sc).astype(np.float16),
                elev=np.maximum(*el).astype(np.float16),
                hips=(P["Hips"] / 100.0).astype(np.float32)), dt, nf, n  # cm -> m


def work(item):
    name, raw = item
    try:
        f, dt, nf, n = features(raw)
        return name, f, dict(dt=dt, header_frames=nf, frames=n), None
    except Exception as e:  # recorded, never silently dropped
        return name, None, None, repr(e)


NAMES = set(json.load(open(os.environ["NAMES"]))) if os.environ.get("NAMES") else None


def items(stream):
    with tarfile.open(fileobj=stream, mode="r|") as tf:
        for m in tf:
            if m.isfile() and m.name.endswith(".bvh") and (
                    NAMES is None or os.path.basename(m.name)[:-4] in NAMES):
                yield m.name, tf.extractfile(m).read()


if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    src = open(sys.argv[2], "rb") if len(sys.argv) > 2 else sys.stdin.buffer
    meta, errors, buf, part = {}, {}, {}, 0

    def flush():
        global part
        if buf:
            np.savez(os.path.join(out, f"feat_{part:04d}.npz"),
                     **{f"{k}|{c}": v for k, d in buf.items() for c, v in d.items()})
            part += 1
            buf.clear()

    with Pool(int(os.environ.get("NPROC", os.cpu_count()))) as pool:
        for name, f, m, err in pool.imap_unordered(work, items(src), chunksize=8):
            clip = os.path.basename(name)[:-4]
            if err:
                errors[name] = err
                continue
            m["path"] = name
            meta[clip] = m
            buf[clip] = f
            if len(buf) >= 2000:
                flush()
            if len(meta) % 5000 == 0:
                print(len(meta), "clips", len(errors), "errors", flush=True)
    flush()
    json.dump(dict(meta=meta, errors=errors), open(os.path.join(out, "meta.json"), "w"))
    print("done", len(meta), "clips", len(errors), "errors", flush=True)
