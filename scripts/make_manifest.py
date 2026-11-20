"""Write the start-trim manifest from detect.py's output.

Usage: python make_manifest.py detections.json OUT_DIR

Keeps only start cuts: every end cut found by detect.py is dropped, so each listed clip
keeps through its last frame. Writes OUT_DIR/bones_seed_tpose_trims.json and
OUT_DIR/bones_seed_tpose_trims.csv and prints their SHA256.
"""
import csv, hashlib, json, math, os, sys


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main(det_path, out_dir):
    d = json.load(open(det_path))
    params = dict(d["params"], scope="clip starts only; end cuts from the detector are not applied")
    clips = []
    for c, r in sorted(d["clips"].items()):
        n, a = r["frames"], r["keepStart"]
        if a == 0:
            continue
        x = dict(filename=c, bvh=r["bvh"], frames120Hz=n, keepStart120Hz=a, keepEnd120Hz=n - 1,
                 headSeconds=round(a / 120, 4), tailSeconds=0.0, keepStart30Hz=math.ceil(a / 4), keepEnd30Hz=None)
        if "head" in r:
            x["headHold120Hz"] = [r["head"]["holdStart"], r["head"]["holdEnd"]]
        clips.append(x)
    doc = dict(
        description=("Calibration T-pose segments at the start of BONES-SEED clips, detected on the SOMA Uniform BVH "
                     "(bones-studio/seed@2f59b2077b9da34dd4e43618e705c7cb962c9a66, 142,220 clips). Start cuts only: "
                     "T-poses at clip ends or inside a motion are kept, so every keepEnd is the last frame and "
                     "keepEnd30Hz is null. Keep ranges are inclusive. Original and mirror share one cut. Clips not "
                     "listed have no opening T-pose segment to trim."),
        params=params, clipsScanned=len(d["clips"]), clipsTrimmed=len(clips),
        removedSeconds=round(sum(x["headSeconds"] for x in clips), 3),
        skippedTooShort=sorted(c for c, r in d["clips"].items() if "skipped" in r), clips=clips)
    os.makedirs(out_dir, exist_ok=True)
    jp, cp = os.path.join(out_dir, "bones_seed_tpose_trims.json"), os.path.join(out_dir, "bones_seed_tpose_trims.csv")
    json.dump(doc, open(jp, "w"), indent=1)
    with open(cp, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["motion", "bvh", "num_frames", "keep_start_frame", "keep_end_frame",
                    "keep_start_s", "keep_end_s", "keep_start_frame_30hz"])
        for x in clips:
            n, a, b = x["frames120Hz"], x["keepStart120Hz"], x["keepEnd120Hz"]
            w.writerow([x["filename"], x["bvh"], n, a, b, f"{a / 120:.4f}", f"{b / 120:.4f}", x["keepStart30Hz"]])
    print(len(clips), "clips,", doc["removedSeconds"], "s removed")
    print(sha256(jp), os.path.basename(jp))
    print(sha256(cp), os.path.basename(cp))


if __name__ == "__main__":
    main(*sys.argv[1:3])
