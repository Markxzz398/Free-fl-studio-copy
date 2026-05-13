"""
BEAT LAB - Enhanced Pygame Beat Sequencer
==========================================
Requirements:
    pip install pygame pydub numpy

Optional (for MP3 export):
    pip install pydub
    Install ffmpeg: https://ffmpeg.org/download.html

Samples folder: place WAV files named kick.wav, snare.wav, clap.wav,
hat_closed.wav, hat_open.wav, tom_hi.wav, tom_lo.wav, cowbell.wav
in a folder called "samples/" next to this script.

CONTROLS:
  SPACE        - Play / Pause
  ESC          - Quit
  F            - Toggle fullscreen
  S            - Save pattern (JSON)
  L            - Load pattern (JSON)
  R            - Randomize current pattern
  C            - Clear current pattern
  1/2/3        - Set steps to 16 / 32 / 8
  UP/DOWN      - Pitch +/- for selected track
  LEFT/RIGHT   - Shift pattern left / right (selected track)
  TAB          - Cycle selected track
  M            - Mute selected track
  I            - Invert selected track
  E            - Apply Euclidean rhythm to selected track
  +/-          - BPM up / down (hold SHIFT for x10)
  CTRL+C       - Copy selected track
  CTRL+V       - Paste to selected track
  F1-F4        - Load preset patterns
"""

import os
import sys
import json
import math
import random
import io
import time
import numpy as np
import pygame
from pygame.locals import *

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
SAMPLE_FOLDER = "samples"
DEFAULT_STEPS = 16
WINDOW_SIZE = (1400, 820)
FPS = 60

TRACKS = [
    {"name": "KICK",     "file": "kick.wav",       "color": (255, 60,  90)},
    {"name": "SNARE",    "file": "snare.wav",       "color": (245, 200, 60)},
    {"name": "CLAP",     "file": "clap.wav",        "color": (0,   220, 150)},
    {"name": "HAT CL",   "file": "hat_closed.wav",  "color": (120, 140, 255)},
    {"name": "HAT OP",   "file": "hat_open.wav",    "color": (200, 100, 255)},
    {"name": "TOM HI",   "file": "tom_hi.wav",      "color": (255, 150, 60)},
    {"name": "TOM LO",   "file": "tom_lo.wav",      "color": (60,  210, 180)},
    {"name": "COWBELL",  "file": "cowbell.wav",      "color": (255, 120, 200)},
]
NUM_TRACKS = len(TRACKS)

BPM_DEFAULT = 120
BPM_MIN = 40
BPM_MAX = 300

# ─────────────────────────────────────────────────────────────
# COLORS
# ─────────────────────────────────────────────────────────────
BG       = (10,  10,  10)
HEADER   = (16,  16,  16)
SIDEBAR  = (14,  14,  14)
PANEL    = (20,  20,  20)
CELL_OFF = (28,  28,  28)
CELL_BEAT= (36,  36,  36)   # every 4th step border highlight
PLAYHEAD = (255, 255, 255)
TEXT     = (200, 200, 200)
MUTED    = (80,  80,  80)
ACCENT   = (0,   220, 150)
ACCENT2  = (255, 60,  90)
ACCENT3  = (245, 200, 60)
BORDER   = (40,  40,  40)
SEL_TRACK= (30,  30,  30)

# ─────────────────────────────────────────────────────────────
# PYGAME INIT
# ─────────────────────────────────────────────────────────────
pygame.init()
pygame.mixer.pre_init(44100, -16, 2, 512)
pygame.mixer.init()
pygame.mixer.set_num_channels(64)

screen = pygame.display.set_mode(WINDOW_SIZE, RESIZABLE)
pygame.display.set_caption("BEAT LAB")
clock = pygame.time.Clock()

font_lg  = pygame.font.SysFont("Courier New", 22, bold=True)
font_md  = pygame.font.SysFont("Courier New", 14)
font_sm  = pygame.font.SysFont("Courier New", 11)
font_xs  = pygame.font.SysFont("Courier New", 10)

# ─────────────────────────────────────────────────────────────
# SAMPLE LOADING  (graceful fallback to synthesized beeps)
# ─────────────────────────────────────────────────────────────
try:
    from pydub import AudioSegment
    PYDUB_OK = True
except ImportError:
    PYDUB_OK = False

_raw_sounds = []    # original pygame.mixer.Sound objects
_raw_segs   = []    # pydub AudioSegment objects (or None)

def _synth_fallback(track_idx):
    """Generate a simple synthesized beep as fallback when sample missing."""
    sr = 44100
    dur = 0.25
    samples = int(sr * dur)
    t = np.linspace(0, dur, samples, endpoint=False)
    freqs = [80, 200, 300, 8000, 5000, 180, 100, 562]
    base = freqs[track_idx % len(freqs)]
    if track_idx == 0:      # kick
        freq_env = base * np.exp(-t * 20)
        wave = np.sin(2 * np.pi * np.cumsum(freq_env) / sr)
        env = np.exp(-t * 12)
    elif track_idx == 1:    # snare
        wave = np.random.randn(samples) * 0.5 + np.sin(2 * np.pi * 200 * t) * 0.5
        env = np.exp(-t * 20)
    elif track_idx == 2:    # clap
        wave = np.random.randn(samples)
        env = np.exp(-t * 30)
    elif track_idx in (3, 4):  # hats
        wave = np.random.randn(samples)
        env = np.exp(-t * (60 if track_idx == 3 else 15))
    else:
        wave = np.sin(2 * np.pi * base * t)
        env = np.exp(-t * 15)
    wave = (wave * env * 0.6 * 32767).astype(np.int16)
    stereo = np.column_stack([wave, wave])
    sound = pygame.sndarray.make_sound(stereo)
    return sound

def load_samples():
    for i, tr in enumerate(TRACKS):
        path = os.path.join(SAMPLE_FOLDER, tr["file"])
        seg = None
        snd = None
        if os.path.exists(path):
            try:
                snd = pygame.mixer.Sound(path)
                if PYDUB_OK:
                    seg = AudioSegment.from_file(path)
            except Exception as e:
                print(f"  Warning: could not load {path}: {e}")
        if snd is None:
            print(f"  Using synthesized fallback for {tr['name']}")
            snd = _synth_fallback(i)
        _raw_sounds.append(snd)
        _raw_segs.append(seg)

load_samples()

# ─────────────────────────────────────────────────────────────
# PITCH / CACHE
# ─────────────────────────────────────────────────────────────
_pitch_cache = {}

def _apply_pitch_pydub(seg, semitones):
    new_rate = int(seg.frame_rate * (2 ** (semitones / 12.0)))
    pitched = seg._spawn(seg.raw_data, overrides={"frame_rate": new_rate})
    return pitched.set_frame_rate(seg.frame_rate)

def _apply_pitch_numpy(track_idx, semitones):
    """Pitch-shift via sample rate trick using raw numpy."""
    arr = pygame.sndarray.array(_raw_sounds[track_idx]).astype(np.float32)
    factor = 2 ** (semitones / 12.0)
    orig_len = len(arr)
    new_len = int(orig_len / factor)
    if new_len < 1:
        new_len = 1
    x_old = np.linspace(0, orig_len - 1, orig_len)
    x_new = np.linspace(0, orig_len - 1, new_len)
    if arr.ndim == 2:
        ch0 = np.interp(x_new, x_old, arr[:, 0]).astype(np.int16)
        ch1 = np.interp(x_new, x_old, arr[:, 1]).astype(np.int16)
        result = np.column_stack([ch0, ch1])
    else:
        result = np.interp(x_new, x_old, arr).astype(np.int16)
    return pygame.sndarray.make_sound(result)

def get_sound(track_idx, semitones, volume):
    key = (track_idx, semitones)
    if key not in _pitch_cache:
        if semitones == 0:
            _pitch_cache[key] = _raw_sounds[track_idx]
        elif PYDUB_OK and _raw_segs[track_idx] is not None:
            seg = _apply_pitch_pydub(_raw_segs[track_idx], semitones)
            buf = io.BytesIO()
            seg.export(buf, format="wav")
            buf.seek(0)
            _pitch_cache[key] = pygame.mixer.Sound(file=buf)
        else:
            _pitch_cache[key] = _apply_pitch_numpy(track_idx, semitones)
    snd = _pitch_cache[key]
    snd.set_volume(max(0.0, min(1.0, volume)))
    return snd

# ─────────────────────────────────────────────────────────────
# EUCLIDEAN RHYTHM
# ─────────────────────────────────────────────────────────────
def euclidean(hits, steps, offset=0):
    """Bjorklund / Euclidean rhythm algorithm."""
    if hits <= 0 or steps <= 0:
        return [False] * steps
    hits = min(hits, steps)
    pattern = []
    bucket = 0
    for i in range(steps):
        bucket += hits
        if bucket >= steps:
            bucket -= steps
            pattern.append(True)
        else:
            pattern.append(False)
    # rotate by offset
    offset = offset % steps
    return pattern[offset:] + pattern[:offset]

# ─────────────────────────────────────────────────────────────
# PRESET PATTERNS
# ─────────────────────────────────────────────────────────────
def make_preset(idx):
    g = [[False] * 16 for _ in range(NUM_TRACKS)]
    if idx == 0:   # four-on-the-floor
        for s in [0, 4, 8, 12]: g[0][s] = True
        for s in [4, 12]:       g[1][s] = True
        for s in [4, 12]:       g[2][s] = True
        for s in range(0,16,2): g[3][s] = True
    elif idx == 1: # breakbeat
        g[0] = euclidean(4, 16)
        g[1] = euclidean(3, 16, 4)
        g[3] = euclidean(8, 16)
        g[4] = euclidean(2, 16, 8)
    elif idx == 2: # hip-hop
        for s in [0, 6, 10]:    g[0][s] = True
        for s in [4, 12]:       g[1][s] = True
        for s in range(0,16,4): g[3][s] = True
        for s in [2, 6, 10, 14]:g[2][s] = True
    elif idx == 3: # latin
        g[0] = euclidean(3, 16)
        g[1] = euclidean(2, 16, 8)
        g[2] = euclidean(5, 16, 2)
        g[3] = euclidean(8, 16)
        g[7] = euclidean(3, 16, 5)
    return g

# ─────────────────────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────────────────────
steps          = DEFAULT_STEPS
grid           = [[False] * steps for _ in range(NUM_TRACKS)]
volumes        = [1.0] * NUM_TRACKS
mutes          = [False] * NUM_TRACKS
pitches        = [0] * NUM_TRACKS        # semitones per track
bpm            = BPM_DEFAULT
playing        = False
current_step   = 0
selected_track = 0
swing_pct      = 0        # 0-66 %
master_volume  = 0.85
clipboard      = None
fullscreen     = False

# UI panels
euclid_hits    = 4
euclid_offset  = 0

# Animation
viz_levels     = [0.0] * NUM_TRACKS   # 0..1 for visualiser bars
notification   = ""
notif_timer    = 0

# ─────────────────────────────────────────────────────────────
# TIMING
# ─────────────────────────────────────────────────────────────
_step_acc_ms   = 0
_swing_flag    = False   # alternates for odd/even steps

def step_ms():
    return (60_000 / bpm) / 4

def swing_delay_ms():
    """Extra delay added on the "off" swing step."""
    if swing_pct <= 0:
        return 0
    base = step_ms()
    return base * (swing_pct / 100) * 0.5

# ─────────────────────────────────────────────────────────────
# SAVE / LOAD
# ─────────────────────────────────────────────────────────────
def save_pattern(path="beat_lab_pattern.json"):
    data = {
        "steps": steps, "bpm": bpm,
        "volumes": volumes, "mutes": mutes,
        "pitches": pitches, "swing": swing_pct,
        "master_volume": master_volume,
        "grid": grid,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    notify(f"SAVED → {path}")

def load_pattern(path="beat_lab_pattern.json"):
    global steps, bpm, volumes, mutes, pitches, swing_pct, master_volume, grid
    if not os.path.exists(path):
        notify("FILE NOT FOUND")
        return
    with open(path) as f:
        d = json.load(f)
    steps          = d.get("steps", 16)
    bpm            = d.get("bpm", 120)
    volumes        = d.get("volumes", [1.0] * NUM_TRACKS)
    mutes          = d.get("mutes",   [False] * NUM_TRACKS)
    pitches        = d.get("pitches", [0] * NUM_TRACKS)
    swing_pct      = d.get("swing", 0)
    master_volume  = d.get("master_volume", 0.85)
    raw_grid       = d.get("grid", [])
    grid = []
    for r in range(NUM_TRACKS):
        if r < len(raw_grid):
            row = raw_grid[r][:steps]
            row += [False] * (steps - len(row))
        else:
            row = [False] * steps
        grid.append(row)
    _pitch_cache.clear()
    notify(f"LOADED ← {path}")

# ─────────────────────────────────────────────────────────────
# NOTIFICATION HELPER
# ─────────────────────────────────────────────────────────────
def notify(msg, dur=2.0):
    global notification, notif_timer
    notification = msg
    notif_timer  = dur

# ─────────────────────────────────────────────────────────────
# DRAWING HELPERS
# ─────────────────────────────────────────────────────────────
def txt(surf, text, pos, color=TEXT, fnt=None, center=False, right=False):
    fnt = fnt or font_md
    img = fnt.render(str(text), True, color)
    r = img.get_rect()
    if center:
        r.center = pos
    elif right:
        r.midright = pos
    else:
        r.topleft = pos
    surf.blit(img, r)

def dim_color(c, factor=0.4):
    return tuple(int(v * factor) for v in c)

def lerp_color(a, b, t):
    return tuple(int(a[i] + (b[i]-a[i])*t) for i in range(3))

def rounded_rect(surf, color, rect, radius=4, border=0, border_color=None):
    pygame.draw.rect(surf, color, rect, border_radius=radius)
    if border and border_color:
        pygame.draw.rect(surf, border_color, rect, border, border_radius=radius)

# ─────────────────────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────────────────────
HEADER_H    = 64
SIDEBAR_W   = 260
TRANSPORT_H = 44
LABEL_W     = 88

def get_layout(w, h):
    grid_x = LABEL_W + 12
    grid_y = HEADER_H + 10
    grid_w = w - SIDEBAR_W - grid_x - 8
    grid_h = h - HEADER_H - TRANSPORT_H - 20
    row_h  = max(28, grid_h // NUM_TRACKS)
    col_w  = max(8,  grid_w // steps)
    return dict(
        grid_x=grid_x, grid_y=grid_y,
        grid_w=grid_w, grid_h=grid_h,
        row_h=row_h,   col_w=col_w,
        sidebar_x=w - SIDEBAR_W,
        transport_y=h - TRANSPORT_H,
    )

# ─────────────────────────────────────────────────────────────
# DRAW
# ─────────────────────────────────────────────────────────────
def draw(surf):
    w, h = surf.get_size()
    surf.fill(BG)
    L = get_layout(w, h)

    # ── HEADER ──────────────────────────────────────────────
    pygame.draw.rect(surf, HEADER, (0, 0, w, HEADER_H))
    pygame.draw.line(surf, BORDER, (0, HEADER_H), (w, HEADER_H))

    # Logo
    txt(surf, "BEAT·LAB", (18, 18), ACCENT, font_lg)

    # BPM
    bpm_x = 180
    txt(surf, "BPM", (bpm_x, 14), MUTED, font_xs)
    txt(surf, str(bpm), (bpm_x, 28), TEXT, font_lg)
    # – + buttons
    bm = pygame.Rect(bpm_x + 64, 20, 20, 20)
    bp = pygame.Rect(bpm_x + 88, 20, 20, 20)
    rounded_rect(surf, PANEL, bm, 3, 1, BORDER)
    rounded_rect(surf, PANEL, bp, 3, 1, BORDER)
    txt(surf, "−", (bm.x+4, bm.y+2), TEXT, font_md)
    txt(surf, "+", (bp.x+4, bp.y+2), TEXT, font_md)

    # Steps selector
    sx = 340
    txt(surf, "STEPS", (sx, 14), MUTED, font_xs)
    for si, s in enumerate([8, 16, 32]):
        r = pygame.Rect(sx + si*46, 28, 40, 22)
        col = ACCENT if steps == s else PANEL
        tcol = (0,0,0) if steps == s else MUTED
        rounded_rect(surf, col, r, 3, 1, BORDER)
        txt(surf, str(s), (r.centerx, r.centery), tcol, font_sm, center=True)

    # Master vol
    mv_x = 490
    txt(surf, "MASTER", (mv_x, 14), MUTED, font_xs)
    bar_r = pygame.Rect(mv_x, 30, 80, 8)
    pygame.draw.rect(surf, PANEL, bar_r, border_radius=3)
    fill_w = int(80 * master_volume)
    pygame.draw.rect(surf, ACCENT, (bar_r.x, bar_r.y, fill_w, 8), border_radius=3)
    txt(surf, f"{int(master_volume*100)}%", (mv_x+84, 26), TEXT, font_xs)

    # Play / Stop
    btn_y = 14
    pb = pygame.Rect(w - SIDEBAR_W - 310, btn_y, 72, 36)
    sb = pygame.Rect(pb.right + 6, btn_y, 72, 36)
    pb_col = ACCENT if playing else PANEL
    pt_col = (0,0,0) if playing else TEXT
    rounded_rect(surf, pb_col, pb, 4, 1, BORDER)
    rounded_rect(surf, PANEL,  sb, 4, 1, BORDER)
    txt(surf, "▶ PLAY" if not playing else "‖ PAUSE", (pb.centerx, pb.centery), pt_col, font_sm, center=True)
    txt(surf, "■ STOP",                               (sb.centerx, sb.centery), TEXT,  font_sm, center=True)

    # Action buttons
    actions = [("RND", ACCENT3), ("CLR", ACCENT2), ("SAVE", TEXT), ("LOAD", TEXT)]
    ax = sb.right + 10
    for label, col in actions:
        ab = pygame.Rect(ax, btn_y+4, 48, 28)
        rounded_rect(surf, PANEL, ab, 3, 1, BORDER)
        txt(surf, label, (ab.centerx, ab.centery), col, font_xs, center=True)
        ax += 54

    # Presets
    txt(surf, "PRESETS", (ax+2, btn_y), MUTED, font_xs)
    for pi in range(4):
        pr = pygame.Rect(ax+2, btn_y+14, 34, 22)
        rounded_rect(surf, PANEL, pr, 3, 1, BORDER)
        names = ["4/4", "BRK", "HIP", "LAT"]
        txt(surf, names[pi], (pr.centerx, pr.centery), ACCENT3, font_xs, center=True)
        ax += 38

    # ── TRACK LABELS & GRID ─────────────────────────────────
    for r in range(NUM_TRACKS):
        tr    = TRACKS[r]
        col   = tr["color"]
        y     = L["grid_y"] + r * L["row_h"]
        sel   = (r == selected_track)

        # Label bg
        lbg = pygame.Rect(4, y+1, LABEL_W + 4, L["row_h"]-2)
        lb_col = lerp_color(PANEL, col, 0.12) if sel else PANEL
        rounded_rect(surf, lb_col, lbg, 3)
        if sel:
            pygame.draw.rect(surf, col, (4, y+1, 3, L["row_h"]-2), border_radius=2)

        # Mute indicator dot
        dot_col = dim_color(col, 0.3) if mutes[r] else col
        pygame.draw.circle(surf, dot_col, (16, y + L["row_h"]//2), 5)

        # Name
        name_col = MUTED if mutes[r] else (col if sel else TEXT)
        txt(surf, tr["name"], (24, y + L["row_h"]//2 - 7), name_col, font_xs)

        # Pitch label
        p = pitches[r]
        p_str = f"P{p:+d}" if p != 0 else ""
        if p_str:
            txt(surf, p_str, (24, y + L["row_h"]//2 + 3), dim_color(col, 0.8), font_xs)

        # Volume mini-bar
        vb = pygame.Rect(LABEL_W - 26, y + L["row_h"]//2 - 3, 24, 6)
        pygame.draw.rect(surf, BORDER, vb, border_radius=2)
        vf = int(24 * volumes[r])
        pygame.draw.rect(surf, dim_color(col, 0.7), (vb.x, vb.y, vf, 6), border_radius=2)

        # ── STEP CELLS ────────────────────────────────────
        for c in range(steps):
            cx = L["grid_x"] + c * L["col_w"]
            cy = y + 1
            cw = L["col_w"] - 2
            ch = L["row_h"] - 2

            # Background
            is_on  = grid[r][c]
            is_ph  = (c == current_step and playing)
            is_beat= (c % 4 == 0)
            is_grp = (steps >= 32 and c % 8 == 0)

            if is_ph:
                bg = lerp_color(col, PLAYHEAD, 0.55)
                bc = PLAYHEAD
            elif is_on:
                bg = dim_color(col, 0.55 if mutes[r] else 0.85)
                bc = dim_color(col, 0.4)
            elif is_beat:
                bg = CELL_BEAT
                bc = BORDER
            else:
                bg = CELL_OFF
                bc = (22,22,22)

            cell_r = pygame.Rect(cx, cy, cw, ch)
            rounded_rect(surf, bg, cell_r, 3, 1, bc)

            # Beat grouping lines
            if is_beat and c > 0:
                pygame.draw.line(surf, (50,50,50), (cx-1, y), (cx-1, y+L["row_h"]))

            # ON indicator dot
            if is_on and not is_ph:
                cx2 = cx + cw//2
                cy2 = cy + ch//2
                pygame.draw.circle(surf, col, (cx2, cy2), min(5, cw//4))

    # ── VISUALISER BARS ────────────────────────────────────
    vz_y = L["grid_y"] + NUM_TRACKS * L["row_h"] + 8
    vz_h = min(40, h - TRANSPORT_H - vz_y - 8)
    if vz_h > 8:
        vbar_w = (L["grid_w"] // NUM_TRACKS) - 2
        for r in range(NUM_TRACKS):
            vx = L["grid_x"] + r * (vbar_w + 2)
            lv = viz_levels[r]
            bar_h = int(vz_h * lv)
            bg_r = pygame.Rect(vx, vz_y, vbar_w, vz_h)
            pygame.draw.rect(surf, PANEL, bg_r, border_radius=2)
            if bar_h > 0:
                col = TRACKS[r]["color"]
                pygame.draw.rect(surf, col,
                    (vx, vz_y + vz_h - bar_h, vbar_w, bar_h), border_radius=2)

    # ── SIDEBAR ─────────────────────────────────────────────
    sx = L["sidebar_x"]
    pygame.draw.rect(surf, SIDEBAR, (sx, HEADER_H, SIDEBAR_W, h - HEADER_H))
    pygame.draw.line(surf, BORDER, (sx, HEADER_H), (sx, h))

    sy = HEADER_H + 10

    def sidebar_label(txt_str, y):
        pygame.draw.line(surf, BORDER, (sx+8, y+10), (sx+SIDEBAR_W-8, y+10))
        ts = font_xs.render(txt_str, True, MUTED)
        tr_ = ts.get_rect(midleft=(sx+10, y+10))
        pygame.draw.rect(surf, SIDEBAR, (tr_.x-2, tr_.y-1, tr_.w+4, tr_.h+2))
        surf.blit(ts, tr_)
        return y + 22

    # ── Track selector ──
    sy = sidebar_label("TRACKS", sy)
    for r in range(NUM_TRACKS):
        tr    = TRACKS[r]
        col   = tr["color"]
        sel   = (r == selected_track)
        tb    = pygame.Rect(sx+6, sy, SIDEBAR_W-12, 22)
        bg    = lerp_color(PANEL, col, 0.15) if sel else SIDEBAR
        rounded_rect(surf, bg, tb, 3, 1 if sel else 0, col if sel else None)
        pygame.draw.circle(surf, dim_color(col,0.3) if mutes[r] else col, (sx+16, sy+11), 5)
        name_c = MUTED if mutes[r] else (col if sel else TEXT)
        txt(surf, tr["name"], (sx+26, sy+5), name_c, font_xs)
        # pitch badge
        p = pitches[r]
        if p != 0:
            txt(surf, f"P{p:+d}", (sx+SIDEBAR_W-40, sy+5), dim_color(col,0.8), font_xs)
        # vol pct
        txt(surf, f"{int(volumes[r]*100)}%", (sx+SIDEBAR_W-20, sy+5), MUTED, font_xs, right=True)
        sy += 24

    # ── Pitch ──
    sy = sidebar_label("PITCH (selected)", sy)
    p_val = pitches[selected_track]
    p_col = TRACKS[selected_track]["color"]
    txt(surf, f"{p_val:+d} st", (sx + SIDEBAR_W//2, sy+4), p_col, font_md, center=True)
    pd = pygame.Rect(sx+8,  sy, 44, 22)
    pu = pygame.Rect(sx+SIDEBAR_W-52, sy, 44, 22)
    rounded_rect(surf, PANEL, pd, 3, 1, BORDER)
    rounded_rect(surf, PANEL, pu, 3, 1, BORDER)
    txt(surf, "◄ −", (pd.centerx, pd.centery), TEXT, font_xs, center=True)
    txt(surf, "+ ►", (pu.centerx, pu.centery), TEXT, font_xs, center=True)
    sy += 28

    # ── Volume ──
    sy = sidebar_label("VOL (selected)", sy)
    vv = volumes[selected_track]
    vbar = pygame.Rect(sx+8, sy+4, SIDEBAR_W-16, 8)
    pygame.draw.rect(surf, PANEL, vbar, border_radius=3)
    pygame.draw.rect(surf, TRACKS[selected_track]["color"],
                     (vbar.x, vbar.y, int(vbar.w * vv), 8), border_radius=3)
    txt(surf, f"{int(vv*100)}%", (sx+SIDEBAR_W-10, sy), TEXT, font_xs, right=True)
    sy += 26

    # ── Swing ──
    sy = sidebar_label("SWING", sy)
    sw_bar = pygame.Rect(sx+8, sy+4, SIDEBAR_W-16, 8)
    pygame.draw.rect(surf, PANEL, sw_bar, border_radius=3)
    pygame.draw.rect(surf, ACCENT3,
                     (sw_bar.x, sw_bar.y, int(sw_bar.w * swing_pct / 66), 8), border_radius=3)
    txt(surf, f"{swing_pct}%", (sx+SIDEBAR_W-10, sy), ACCENT3, font_xs, right=True)
    sy += 24

    # ── Pattern tools ──
    sy = sidebar_label("PATTERN TOOLS", sy)
    tools = [
        ("FILL",    "F keys"),
        ("INVERT",  "I key"),
        ("SHIFT ◄", "← key"),
        ("SHIFT ►", "→ key"),
        ("COPY",    "Ctrl+C"),
        ("PASTE",   "Ctrl+V"),
    ]
    for ti, (label, shortcut) in enumerate(tools):
        tb2 = pygame.Rect(sx + 6 + (ti%2)*((SIDEBAR_W-14)//2),
                          sy + (ti//2)*24, (SIDEBAR_W-18)//2, 20)
        rounded_rect(surf, PANEL, tb2, 3, 1, BORDER)
        txt(surf, label, (tb2.centerx, tb2.centery), TEXT, font_xs, center=True)
    sy += (len(tools)//2) * 24 + 6

    # ── Euclid ──
    sy = sidebar_label("EUCLIDEAN RHYTHM", sy)
    txt(surf, f"Hits: {euclid_hits}  Offset: {euclid_offset}", (sx+8, sy), TEXT, font_xs)
    eb = pygame.Rect(sx+8, sy+14, SIDEBAR_W-16, 20)
    rounded_rect(surf, PANEL, eb, 3, 1, ACCENT)
    txt(surf, f"APPLY  E={euclid_hits}/{steps}+{euclid_offset}", (eb.centerx, eb.centery), ACCENT, font_xs, center=True)
    sy += 42

    # ── Keyboard hints ──
    sy = sidebar_label("SHORTCUTS", sy)
    hints = [
        ("SPACE",      "Play/Pause"),
        ("S / L",      "Save / Load"),
        ("R",          "Randomize"),
        ("C",          "Clear track"),
        ("M",          "Mute track"),
        ("I",          "Invert track"),
        ("E",          "Euclid apply"),
        ("TAB",        "Select track"),
        ("+/−",        "BPM ±1 (×10)"),
        ("↑↓",         "Pitch ±1 st"),
        ("←→",         "Shift pattern"),
        ("F1-F4",      "Load presets"),
        ("F",          "Fullscreen"),
    ]
    for key, action in hints:
        if sy > h - TRANSPORT_H - 14:
            break
        txt(surf, key,    (sx+10, sy), ACCENT, font_xs)
        txt(surf, action, (sx+70, sy), MUTED,  font_xs)
        sy += 14

    # ── TRANSPORT ───────────────────────────────────────────
    ty = L["transport_y"]
    pygame.draw.rect(surf, HEADER, (0, ty, w, TRANSPORT_H))
    pygame.draw.line(surf, BORDER, (0, ty), (w, ty))

    # Progress bar
    pb_w = w - SIDEBAR_W - 20
    pb_r  = pygame.Rect(10, ty + 16, pb_w, 6)
    pygame.draw.rect(surf, PANEL,  pb_r, border_radius=3)
    fill  = int(pb_w * current_step / max(1, steps))
    pygame.draw.rect(surf, ACCENT, (pb_r.x, pb_r.y, fill, 6), border_radius=3)
    # playhead cursor
    phx = pb_r.x + fill
    pygame.draw.rect(surf, PLAYHEAD, (phx-1, ty+10, 3, 18), border_radius=1)

    txt(surf, f"STEP {current_step+1:02d} / {steps:02d}", (w - SIDEBAR_W - 100, ty+14), MUTED, font_sm)
    spd = f"BPM {bpm}  SWING {swing_pct}%"
    txt(surf, spd, (10, ty+6), MUTED, font_xs)

    # ── NOTIFICATION ────────────────────────────────────────
    global notif_timer
    if notif_timer > 0:
        alpha = min(1.0, notif_timer * 3)
        nc = tuple(int(c * alpha) for c in ACCENT)
        nb = pygame.Rect(w//2 - 120, ty - 42, 240, 30)
        rounded_rect(surf, (18,18,18), nb, 4, 1, nc)
        txt(surf, notification, (nb.centerx, nb.centery), nc, font_sm, center=True)

def get_btn_rects(w, h):
    """Return clickable regions for header buttons."""
    L = get_layout(w, h)
    rects = {}

    bpm_x = 180
    rects["bpm_down"]  = pygame.Rect(bpm_x+64, 20, 20, 20)
    rects["bpm_up"]    = pygame.Rect(bpm_x+88, 20, 20, 20)

    sx_ = 340
    for si, s in enumerate([8, 16, 32]):
        rects[f"steps_{s}"] = pygame.Rect(sx_+si*46, 28, 40, 22)

    btn_y = 14
    right_edge = w - SIDEBAR_W - 10
    ax = right_edge - 310
    pb = pygame.Rect(ax, btn_y, 72, 36); ax = pb.right+6
    sb = pygame.Rect(ax, btn_y, 72, 36); ax = sb.right+10
    rects["play"] = pb
    rects["stop"] = sb

    for label in ["RND","CLR","SAVE","LOAD"]:
        ab = pygame.Rect(ax, btn_y+4, 48, 28)
        rects[label] = ab
        ax += 54

    ax += 2
    for pi, name in enumerate(["P0","P1","P2","P3"]):
        pr = pygame.Rect(ax, btn_y+14, 34, 22)
        rects[f"preset_{pi}"] = pr
        ax += 38

    # Master vol bar
    rects["master_vol"] = pygame.Rect(490, 22, 80, 16)

    # Sidebar buttons
    sx = L["sidebar_x"]
    sy = HEADER_H + 10
    sy += 22  # label
    for r in range(NUM_TRACKS):
        rects[f"track_{r}"] = pygame.Rect(sx+6, sy, SIDEBAR_W-12, 22)
        sy += 24

    sy += 22  # pitch label
    rects["pitch_down"] = pygame.Rect(sx+8,  sy, 44, 22)
    rects["pitch_up"]   = pygame.Rect(sx+SIDEBAR_W-52, sy, 44, 22)
    sy += 28

    sy += 22  # vol label
    rects["vol_bar"] = pygame.Rect(sx+8, sy+4, SIDEBAR_W-16, 8)
    sy += 26

    sy += 22  # swing label
    rects["swing_bar"] = pygame.Rect(sx+8, sy+4, SIDEBAR_W-16, 8)
    sy += 24

    sy += 22  # tools label
    tool_keys = ["fill","invert","shift_l","shift_r","copy","paste"]
    for ti, tk in enumerate(tool_keys):
        tb2 = pygame.Rect(sx+6+(ti%2)*((SIDEBAR_W-14)//2),
                          sy+(ti//2)*24, (SIDEBAR_W-18)//2, 20)
        rects[f"tool_{tk}"] = tb2
    sy += (len(tool_keys)//2)*24+6

    sy += 22  # euclid label
    rects["euclid_apply"] = pygame.Rect(sx+8, sy+14, SIDEBAR_W-16, 20)

    return rects

# ─────────────────────────────────────────────────────────────
# REBUILD GRID  (when step count changes)
# ─────────────────────────────────────────────────────────────
def rebuild_grid(new_steps):
    global steps, grid, current_step
    old = grid
    steps = new_steps
    grid = []
    for r in range(NUM_TRACKS):
        row = old[r][:steps]
        row += [False] * (steps - len(row))
        grid.append(row)
    current_step = 0

# ─────────────────────────────────────────────────────────────
# TRIGGER A STEP
# ─────────────────────────────────────────────────────────────
def fire_step(step):
    for r in range(NUM_TRACKS):
        if grid[r][step] and not mutes[r]:
            snd = get_sound(r, pitches[r], volumes[r] * master_volume)
            snd.play()
            viz_levels[r] = 1.0

# ─────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────
def set_fullscreen(on):
    global fullscreen, screen
    fullscreen = on
    if fullscreen:
        screen = pygame.display.set_mode((0, 0), FULLSCREEN)
    else:
        screen = pygame.display.set_mode(WINDOW_SIZE, RESIZABLE)

_step_acc  = 0.0
_swing_odd = False

running = True
while running:
    dt = clock.tick(FPS)

    # ── TIMING ──────────────────────────────────────────────
    if playing:
        _step_acc += dt
        target = step_ms()
        if _swing_odd and swing_pct > 0:
            target += swing_delay_ms()
        if _step_acc >= target:
            _step_acc -= target
            _swing_odd = not _swing_odd
            fire_step(current_step)
            current_step = (current_step + 1) % steps

    # ── VIZ DECAY ───────────────────────────────────────────
    for r in range(NUM_TRACKS):
        viz_levels[r] = max(0.0, viz_levels[r] - dt * 0.006)

    # ── NOTIFICATION TIMER ──────────────────────────────────
    if notif_timer > 0:
        notif_timer = max(0, notif_timer - dt / 1000.0)

    # ── EVENTS ──────────────────────────────────────────────
    w, h = screen.get_size()
    rects = get_btn_rects(w, h)
    L = get_layout(w, h)

    for event in pygame.event.get():
        if event.type == QUIT:
            running = False

        elif event.type == VIDEORESIZE:
            screen = pygame.display.set_mode(event.size, RESIZABLE)

        elif event.type == KEYDOWN:
            mods = pygame.key.get_mods()
            ctrl  = mods & KMOD_CTRL
            shift = mods & KMOD_SHIFT

            if event.key == K_ESCAPE:
                running = False
            elif event.key == K_SPACE:
                playing = not playing
                if not playing:
                    current_step = 0
                    _step_acc = 0.0
                    _swing_odd = False
            elif event.key == K_f and not ctrl:
                set_fullscreen(not fullscreen)
            elif event.key == K_s and not ctrl:
                save_pattern()
            elif event.key == K_l and not ctrl:
                load_pattern()
            elif event.key == K_r:
                for c in range(steps):
                    grid[selected_track][c] = random.random() < 0.3
                notify("RANDOMIZED")
            elif event.key == K_c and ctrl:
                clipboard = list(grid[selected_track])
                notify("COPIED")
            elif event.key == K_v and ctrl:
                if clipboard:
                    row = clipboard[:steps] + [False]*(steps-len(clipboard))
                    grid[selected_track] = row
                    notify("PASTED")
            elif event.key == K_c and not ctrl:
                grid[selected_track] = [False]*steps
                notify("CLEARED")
            elif event.key == K_i:
                grid[selected_track] = [not v for v in grid[selected_track]]
                notify("INVERTED")
            elif event.key == K_m:
                mutes[selected_track] = not mutes[selected_track]
                notify("MUTED" if mutes[selected_track] else "UNMUTED")
            elif event.key == K_e:
                grid[selected_track] = euclidean(euclid_hits, steps, euclid_offset)
                notify(f"EUCLID {euclid_hits}/{steps}+{euclid_offset}")
            elif event.key == K_TAB:
                selected_track = (selected_track + (NUM_TRACKS-1 if shift else 1)) % NUM_TRACKS
            elif event.key == K_UP:
                pitches[selected_track] = min(12, pitches[selected_track]+1)
                _pitch_cache.clear()
                notify(f"{TRACKS[selected_track]['name']} PITCH {pitches[selected_track]:+d}")
            elif event.key == K_DOWN:
                pitches[selected_track] = max(-12, pitches[selected_track]-1)
                _pitch_cache.clear()
                notify(f"{TRACKS[selected_track]['name']} PITCH {pitches[selected_track]:+d}")
            elif event.key == K_LEFT:
                g = grid[selected_track]
                grid[selected_track] = g[1:] + [g[0]]
            elif event.key == K_RIGHT:
                g = grid[selected_track]
                grid[selected_track] = [g[-1]] + g[:-1]
            elif event.key == K_EQUALS or event.key == K_PLUS:
                bpm = min(BPM_MAX, bpm + (10 if shift else 1))
            elif event.key == K_MINUS:
                bpm = max(BPM_MIN, bpm - (10 if shift else 1))
            elif event.key == K_1:
                rebuild_grid(16); notify("16 STEPS")
            elif event.key == K_2:
                rebuild_grid(32); notify("32 STEPS")
            elif event.key == K_3:
                rebuild_grid(8);  notify("8 STEPS")
            elif event.key in (K_F1, K_F2, K_F3, K_F4):
                pi = event.key - K_F1
                raw = make_preset(pi)
                grid = [raw[r][:steps]+[False]*(steps-len(raw[r][:steps])) for r in range(NUM_TRACKS)]
                names = ["4/4","BREAKBEAT","HIP-HOP","LATIN"]
                notify(f"PRESET: {names[pi]}")

        elif event.type == MOUSEBUTTONDOWN:
            mx, my = event.pos
            btn = event.button

            # ── Header buttons ──
            if my < HEADER_H:
                if rects["bpm_down"].collidepoint(mx,my):
                    bpm = max(BPM_MIN, bpm-1)
                elif rects["bpm_up"].collidepoint(mx,my):
                    bpm = min(BPM_MAX, bpm+1)
                elif rects["play"].collidepoint(mx,my):
                    playing = not playing
                    if not playing: current_step=0; _step_acc=0.0
                elif rects["stop"].collidepoint(mx,my):
                    playing=False; current_step=0; _step_acc=0.0
                elif rects.get("RND",pygame.Rect(0,0,0,0)).collidepoint(mx,my):
                    for r2 in range(NUM_TRACKS):
                        for c2 in range(steps):
                            grid[r2][c2] = random.random() < (0.35 if r2==0 else 0.22)
                    notify("RANDOMIZED")
                elif rects.get("CLR",pygame.Rect(0,0,0,0)).collidepoint(mx,my):
                    grid = [[False]*steps for _ in range(NUM_TRACKS)]
                    notify("CLEARED")
                elif rects.get("SAVE",pygame.Rect(0,0,0,0)).collidepoint(mx,my):
                    save_pattern()
                elif rects.get("LOAD",pygame.Rect(0,0,0,0)).collidepoint(mx,my):
                    load_pattern()
                else:
                    for si, s in enumerate([8,16,32]):
                        if rects[f"steps_{s}"].collidepoint(mx,my):
                            rebuild_grid(s); notify(f"{s} STEPS")
                    for pi in range(4):
                        if rects.get(f"preset_{pi}",pygame.Rect(0,0,0,0)).collidepoint(mx,my):
                            raw = make_preset(pi)
                            grid = [raw[r][:steps]+[False]*(steps-len(raw[r][:steps])) for r in range(NUM_TRACKS)]
                            notify(["4/4","BREAKBEAT","HIP-HOP","LATIN"][pi])

            # ── Master vol bar ──
            if rects["master_vol"].collidepoint(mx,my):
                rel = (mx - rects["master_vol"].x) / rects["master_vol"].w
                master_volume = max(0.0, min(1.5, rel * 1.5))

            # ── Sidebar ──
            if mx >= L["sidebar_x"]:
                for r2 in range(NUM_TRACKS):
                    if rects[f"track_{r2}"].collidepoint(mx,my):
                        if btn == 1: selected_track = r2
                        elif btn == 3: mutes[r2] = not mutes[r2]

                if rects["pitch_down"].collidepoint(mx,my):
                    pitches[selected_track] = max(-12, pitches[selected_track]-1)
                    _pitch_cache.clear()
                elif rects["pitch_up"].collidepoint(mx,my):
                    pitches[selected_track] = min(12, pitches[selected_track]+1)
                    _pitch_cache.clear()

                if rects["vol_bar"].collidepoint(mx,my):
                    rel = (mx - rects["vol_bar"].x) / rects["vol_bar"].w
                    volumes[selected_track] = max(0.0, min(1.0, rel))

                if rects["swing_bar"].collidepoint(mx,my):
                    rel = (mx - rects["swing_bar"].x) / rects["swing_bar"].w
                    swing_pct = int(max(0, min(66, rel*66)))

                for tk, action in [("fill","fill"),("invert","invert"),
                                    ("shift_l","shift_l"),("shift_r","shift_r"),
                                    ("copy","copy"),("paste","paste")]:
                    if rects.get(f"tool_{tk}",pygame.Rect(0,0,0,0)).collidepoint(mx,my):
                        if action == "fill":
                            grid[selected_track] = [True]*steps; notify("FILLED")
                        elif action == "invert":
                            grid[selected_track] = [not v for v in grid[selected_track]]; notify("INVERTED")
                        elif action == "shift_l":
                            g = grid[selected_track]; grid[selected_track] = g[1:]+[g[0]]
                        elif action == "shift_r":
                            g = grid[selected_track]; grid[selected_track] = [g[-1]]+g[:-1]
                        elif action == "copy":
                            clipboard = list(grid[selected_track]); notify("COPIED")
                        elif action == "paste" and clipboard:
                            row = clipboard[:steps]+[False]*(steps-len(clipboard))
                            grid[selected_track] = row; notify("PASTED")

                if rects.get("euclid_apply",pygame.Rect(0,0,0,0)).collidepoint(mx,my):
                    grid[selected_track] = euclidean(euclid_hits, steps, euclid_offset)
                    notify(f"EUCLID {euclid_hits}/{steps}+{euclid_offset}")
                    if btn == 3:
                        euclid_hits = min(steps, euclid_hits+1)
                    elif btn == 1 and euclid_hits > 1:
                        euclid_hits -= 1

            # ── Grid cells ──
            else:
                gx = mx - L["grid_x"]
                gy = my - L["grid_y"]
                if 0 <= gx < L["col_w"]*steps and 0 <= gy < L["row_h"]*NUM_TRACKS:
                    c2 = gx // L["col_w"]
                    r2 = gy // L["row_h"]
                    if 0 <= r2 < NUM_TRACKS and 0 <= c2 < steps:
                        if btn == 1:
                            grid[r2][c2] = not grid[r2][c2]
                            if grid[r2][c2]:
                                snd = get_sound(r2, pitches[r2], volumes[r2]*master_volume)
                                snd.play()
                                viz_levels[r2] = 1.0
                        elif btn == 3:
                            grid[r2][c2] = False  # right-click to erase
                        selected_track = r2

        elif event.type == MOUSEMOTION:
            if event.buttons[0]:  # drag on bars
                mx, my = event.pos
                if rects["master_vol"].collidepoint(mx,my):
                    rel = (mx - rects["master_vol"].x) / rects["master_vol"].w
                    master_volume = max(0.0, min(1.5, rel*1.5))
                if mx >= L["sidebar_x"]:
                    if rects["vol_bar"].collidepoint(mx,my):
                        rel = (mx-rects["vol_bar"].x)/rects["vol_bar"].w
                        volumes[selected_track] = max(0,min(1,rel))
                    if rects["swing_bar"].collidepoint(mx,my):
                        rel = (mx-rects["swing_bar"].x)/rects["swing_bar"].w
                        swing_pct = int(max(0,min(66,rel*66)))

    # ── DRAW ────────────────────────────────────────────────
    draw(screen)
    pygame.display.flip()

pygame.quit()
sys.exit()