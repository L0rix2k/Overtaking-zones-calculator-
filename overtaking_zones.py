#!/usr/bin/env python3
"""
Find the best overtaking spots on a track made by track_calc.py (reads track.json).

Usage:    python overtaking_zones.py track.json
          python overtaking_zones.py track.json --top 5

Needs lap_time.py in the same folder, plus:  pip install numpy opencv-python

The defaults are the F4 settings calibrated against the game's lap records, so the plain
command above is enough. To try another car, change the same options as in lap_time.py.

How it works: it simulates the fastest lap (like lap_time.py), then looks at every heavy
braking zone. A good overtaking spot has a long fast run before the corner (so a car behind
can get a tow), a big speed drop (lots of room to out-brake someone) and a long braking zone.
Each zone gets a score from 0 to 100.

It writes overtaking_zones.png: the track with every braking zone numbered and colored
(green = great, yellow = good, red = poor).
"""
import argparse

import cv2
import numpy as np

from lap_time import load_track, speed_profile


def find_braking_zones(v, ds, min_drop_frac=0.12):
    """Return a list of braking zones as dicts of indices (into the lap, starting at the slowest point)."""
    n = len(v)
    i0 = int(np.argmin(v))  # start the lap at the slowest point so no zone is cut in two
    v = np.roll(v, -i0)
    ds = np.roll(ds, -i0)
    s = np.r_[0.0, np.cumsum(ds)]  # distance at each index (s[n] = lap length)

    decel = np.zeros(n)
    decel[:-1] = (v[:-1] ** 2 - v[1:] ** 2) / (2 * ds[:-1] + 1e-9)  # > 0 when slowing down
    braking = decel > 1.5

    runs, i = [], 0
    while i < n:
        if braking[i]:
            j = i
            while j + 1 < n and braking[j + 1]:
                j += 1
            runs.append([i, j + 1])
            i = j + 1
        else:
            i += 1
    merged = []  # join runs separated by a tiny gap
    for r in runs:
        if merged and r[0] - merged[-1][1] <= 2:
            merged[-1][1] = r[1]
        else:
            merged.append(r)

    zones, prev_end = [], 0
    for b0, b1 in merged:
        b1 = min(b1, n - 1)
        drop = v[b0] - v[b1]
        if drop < min_drop_frac * v.max() or b1 - b0 < 2:
            continue
        j = b0  # walk back along the fast section that leads into the braking zone
        while j > prev_end and v[j - 1] >= 0.85 * v[b0]:
            j -= 1
        zones.append({
            "b0": b0, "b1": b1,
            "straight_m": s[b0] - s[j],
            "brake_m": s[b1] - s[b0],
            "v_in": v[b0], "v_out": v[b1], "drop": drop,
            "at_m": 0.0, "orig_index": (b0 + i0) % n, "orig_end": (b1 + i0) % n,
        })
        prev_end = b1
    return zones, i0, s


def score_zones(zones):
    if not zones:
        return
    ms = max(z["straight_m"] for z in zones) or 1
    md = max(z["drop"] for z in zones) or 1
    mb = max(z["brake_m"] for z in zones) or 1
    for z in zones:
        z["score"] = 100 * (0.45 * z["straight_m"] / ms + 0.35 * z["drop"] / md + 0.20 * z["brake_m"] / mb)
        z["rating"] = "Great" if z["score"] >= 65 else ("Good" if z["score"] >= 40 else "Poor")


def draw(pts, zones, out):
    P = pts  # (x, z) in meters
    lo, hi = P.min(axis=0), P.max(axis=0)
    W, H, M = 1100, 760, 60
    k = min((W - 2 * M) / (hi[0] - lo[0]), (H - 2 * M) / (hi[1] - lo[1]))
    px = ((P - lo) * k + M).astype(np.int32)
    img = np.full((H, W, 3), 245, np.uint8)
    cv2.polylines(img, [px.reshape(-1, 1, 2)], True, (150, 150, 150), 6, cv2.LINE_AA)
    colors = {"Great": (80, 170, 40), "Good": (30, 200, 240), "Poor": (60, 60, 220)}
    n = len(pts)
    for num, z in enumerate(zones, 1):
        idx = [(z["orig_index"] + t) % n for t in range((z["orig_end"] - z["orig_index"]) % n + 1)]
        cv2.polylines(img, [px[idx].reshape(-1, 1, 2)], False, colors[z["rating"]], 9, cv2.LINE_AA)
        c = tuple(int(v) for v in px[z["orig_index"]])
        cv2.circle(img, c, 13, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, c, 13, (40, 40, 40), 1, cv2.LINE_AA)
        cv2.putText(img, str(num), (c[0] - (6 if num < 10 else 11), c[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.circle(img, tuple(int(v) for v in px[0]), 8, (0, 0, 0), -1, cv2.LINE_AA)  # start / finish
    for i, (name, col) in enumerate(colors.items()):
        cv2.rectangle(img, (20, 20 + i * 26), (44, 40 + i * 26), col, -1)
        cv2.putText(img, name, (54, 38 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.imwrite(out, img)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("track", help="track.json from track_calc.py")
    p.add_argument("--top-speed", type=float, default=225.0, help="km/h (speedometer)")
    p.add_argument("--grip", type=float, default=2.0, help="cornering grip in g")
    p.add_argument("--accel", type=float, default=7.0, help="m/s^2")
    p.add_argument("--brake", type=float, default=25.0, help="m/s^2")
    p.add_argument("--speed-factor", type=float, default=1.966,
                   help="real speed / speedometer speed (from lap_time.py --target-time)")
    p.add_argument("--meters-per-stud", type=float, default=0.28)
    p.add_argument("--top", type=int, default=0, help="only list the N best zones")
    p.add_argument("--out", default="overtaking_zones.png")
    a = p.parse_args()

    pts, _ = load_track(a.track, a.meters_per_stud)
    k = a.speed_factor
    ds, v, dt = speed_profile(pts, a.grip * k * k, a.top_speed / 3.6 * k, a.accel * k, a.brake * k)

    zones, i0, s = find_braking_zones(v, ds)
    score_zones(zones)
    start_of_lap = np.r_[0.0, np.cumsum(ds)]
    for z in zones:  # distance from the start/finish line
        z["at_m"] = start_of_lap[z["orig_index"]]
    zones.sort(key=lambda z: z["at_m"])

    print(f"Track {ds.sum() / 1000:.2f} km | lap {dt.sum():.1f} s | {len(zones)} braking zones\n")
    print(" #   at (m)  fast run  speed in -> out    braking   score  rating")
    for num, z in enumerate(zones, 1):
        print(f"{num:2d}  {z['at_m']:7.0f}  {z['straight_m']:7.0f} m  "
              f"{z['v_in'] * 3.6 / k:4.0f} -> {z['v_out'] * 3.6 / k:3.0f} km/h  {z['brake_m']:6.0f} m  "
              f"{z['score']:5.0f}   {z['rating']}")

    best = sorted(zones, key=lambda z: -z["score"])[: (a.top or 3)]
    print("\nBest places to overtake:")
    for z in best:
        print(f"  Zone {zones.index(z) + 1} (at {z['at_m'] / 1000:.2f} km): "
              f"{z['straight_m']:.0f} m run, braking from {z['v_in'] * 3.6 / k:.0f} to {z['v_out'] * 3.6 / k:.0f} km/h "
              f"over {z['brake_m']:.0f} m")

    draw(pts, zones, a.out)
    print(f"\nWrote {a.out}")


if __name__ == "__main__":
    main()
