"""
SKY ROYALE  -  a top-down battle-royale game in Python / pygame
================================================================

    pip install pygame numpy        (numpy is optional, it only enables sound)

FLOW
    LOBBY   -> pick your name, colour, PRIMARY and SECONDARY gun, player count and bot skill.
               Other players "join" the lobby, then press START MATCH.
    PLANE   -> a plane flies across the island. Press SPACE to jump out.
    DROP    -> steer your parachute (WASD). Hold SHIFT to dive faster. Pick your landing spot!
    BATTLE  -> loot houses, fight, stay inside the shrinking safe zone. Last one alive wins.

CONTROLS
    W A S D       move / steer parachute        SHIFT   sprint / dive
    MOUSE         aim                           LEFT CLICK   shoot (hold for auto guns)
    R             reload                        1 / 2 or MOUSE WHEEL   switch gun
    E             pick up a gun                 (ammo, armor and medkits are picked up automatically)
    Q             use medkit (2 s, slows you)   TAB / M   big map
    SPACE         jump from the plane           ESC   pause        F11  fullscreen      N  mute

GUNS  (10 different guns, 6 types)
    Assault rifles  AK-47, M4A1          SMG       MP5, Micro SMG      Shotgun  M1887, Sawed-Off
    Sniper          AWM                  LMG       M60                 Pistols  M1911, Desert Hawk
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pygame

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

pygame.init()
AUDIO_OK = False
try:
    pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
    AUDIO_OK = True
except pygame.error:
    AUDIO_OK = False

# ════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ════════════════════════════════════════════════════════════════════════════
SCREEN_W, SCREEN_H = 1280, 720
FPS = 60
MAP_SIZE = 5200
CELL = 320                       # spatial-grid cell size

PLAYER_R = 16
WALK_SPEED = 210.0
SPRINT_SPEED = 290.0
BASE_HP = 100.0
MAX_ARMOR = 100.0
MEDKIT_HEAL = 60.0
MEDKIT_TIME = 2.0
MAX_MEDKITS = 4
PICKUP_RANGE = 46
PLANE_SPEED = 330.0

WHITE = (255, 255, 255); BLACK = (0, 0, 0); RED = (220, 60, 60); GREEN = (70, 200, 100)
BLUE = (70, 140, 230); YELLOW = (250, 210, 60); ORANGE = (250, 150, 40); GRAY = (150, 150, 160)
DARK = (22, 26, 38); PANEL = (30, 36, 52); GOLD = (255, 200, 60); CYAN = (60, 210, 220)
PURPLE = (160, 110, 230); PINK = (250, 120, 170)

PLAYER_COLORS = [(230, 70, 70), (70, 140, 240), (80, 200, 110), (250, 200, 60),
                 (170, 110, 230), (250, 140, 50), (60, 210, 220), (240, 240, 240)]
BOT_NAMES = ["Viper", "Blaze", "Ghost", "Nova", "Raptor", "Storm", "Falcon", "Shadow", "Titan", "Rogue",
             "Comet", "Hunter", "Ranger", "Phantom", "Cobra", "Dagger", "Ember", "Frost", "Hawk", "Jester",
             "Kilo", "Lynx", "Maverick", "Nomad", "Onyx", "Pike", "Quartz", "Razor", "Sable", "Talon",
             "Umbra", "Vortex", "Wolf", "Xeno", "Yeti", "Zephyr"]
DIFFICULTY = {
    "Easy":   dict(sight=520, aim=0.16, react=0.70, turn=5.0),
    "Normal": dict(sight=680, aim=0.09, react=0.45, turn=7.0),
    "Hard":   dict(sight=820, aim=0.045, react=0.25, turn=10.0),
}
DIFF_ORDER = ["Easy", "Normal", "Hard"]


# ════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════════════════
def clamp(v, lo, hi): return max(lo, min(hi, v))
def lerp(a, b, t): return a + (b - a) * t
def angle_diff(a, b): return (b - a + math.pi) % (2 * math.pi) - math.pi


def seg_point_dist2(x0, y0, x1, y1, px, py):
    dx, dy = x1 - x0, y1 - y0
    l2 = dx * dx + dy * dy
    t = 0.0 if l2 == 0 else clamp(((px - x0) * dx + (py - y0) * dy) / l2, 0.0, 1.0)
    qx, qy = x0 + dx * t, y0 + dy * t
    return (px - qx) ** 2 + (py - qy) ** 2


def fmt_time(s):
    s = max(0, int(s))
    return f"{s // 60}:{s % 60:02d}"


# ════════════════════════════════════════════════════════════════════════════
#  SOUND (optional, needs numpy)
# ════════════════════════════════════════════════════════════════════════════
AUDIO = {"muted": False}


def _to_sound(mono):
    init = pygame.mixer.get_init()
    ch = init[2] if init else 1
    if ch > 1:
        mono = np.ascontiguousarray(np.column_stack([mono] * ch))
    return pygame.sndarray.make_sound(mono)


def make_tone(freq, dur, vol=0.35):
    if not (HAS_NUMPY and AUDIO_OK):
        return None
    try:
        sr = pygame.mixer.get_init()[0]
        n = int(sr * dur)
        t = np.arange(n, dtype=np.float32) / sr
        env = np.linspace(1, 0, n, dtype=np.float32) ** 0.6
        return _to_sound((np.sin(2 * np.pi * freq * t) * env * 32767 * vol).astype(np.int16))
    except Exception:
        return None


def make_noise(dur, vol=0.45, decay=7.0):
    if not (HAS_NUMPY and AUDIO_OK):
        return None
    try:
        sr = pygame.mixer.get_init()[0]
        n = int(sr * dur)
        t = np.linspace(0, 1, n, dtype=np.float32)
        sig = np.random.uniform(-1, 1, n).astype(np.float32) * np.exp(-decay * t)
        return _to_sound((sig * 32767 * vol).astype(np.int16))
    except Exception:
        return None


SFX = {
    "shot": make_noise(0.10, 0.40, 7), "shotgun": make_noise(0.18, 0.55, 5),
    "sniper": make_noise(0.32, 0.55, 4), "hit": make_tone(240, 0.06, 0.4),
    "hurt": make_tone(140, 0.12, 0.5), "pickup": make_tone(880, 0.07, 0.3),
    "reload": make_tone(330, 0.10, 0.3), "jump": make_tone(520, 0.15, 0.3),
    "kill": make_tone(1200, 0.18, 0.35), "click": make_tone(660, 0.04, 0.25),
}


def play(name, vol=1.0):
    snd = SFX.get(name)
    if snd and not AUDIO["muted"]:
        ch = snd.play()
        if ch:
            ch.set_volume(clamp(vol, 0.0, 1.0))


# ════════════════════════════════════════════════════════════════════════════
#  WEAPONS
# ════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class WeaponDef:
    key: str
    name: str
    kind: str            # Assault Rifle, SMG, Shotgun, Sniper, LMG, Pistol
    damage: float        # per bullet / pellet
    rate: float          # shots per second
    mag: int
    reload: float        # seconds
    spread: float        # degrees of random spread
    range: float         # pixels
    speed: float         # bullet speed px/s
    pellets: int
    auto: bool
    ammo: str            # light / rifle / shell / sniper
    color: Tuple[int, int, int]
    blurb: str


WEAPONS: Dict[str, WeaponDef] = {w.key: w for w in [
    WeaponDef("ak47", "AK-47", "Assault Rifle", 26, 6.5, 30, 2.3, 4.2, 900, 1800, 1, True, "rifle",
              (200, 140, 70), "Hard-hitting all-rounder. Kicks a bit."),
    WeaponDef("m4a1", "M4A1", "Assault Rifle", 21, 9.0, 30, 2.0, 3.2, 950, 1900, 1, True, "rifle",
              (120, 160, 220), "Fast and accurate. Great at any range."),
    WeaponDef("mp5", "MP5", "SMG", 15, 12.0, 30, 1.7, 5.5, 560, 1600, 1, True, "light",
              (150, 210, 150), "High fire rate for close and mid range."),
    WeaponDef("micro", "Micro SMG", "SMG", 12, 15.0, 25, 1.5, 8.0, 420, 1500, 1, True, "light",
              (240, 200, 90), "Bullet hose for point-blank fights."),
    WeaponDef("m1887", "M1887", "Shotgun", 11, 1.1, 5, 2.6, 11.0, 380, 1500, 9, False, "shell",
              (220, 110, 90), "Devastating up close. Useless far away."),
    WeaponDef("sawed", "Sawed-Off", "Shotgun", 14, 1.4, 2, 2.0, 16.0, 300, 1400, 6, False, "shell",
              (200, 90, 140), "Two shells of pure pain. Tiny range."),
    WeaponDef("awm", "AWM", "Sniper", 92, 0.75, 5, 3.0, 0.2, 1700, 3200, 1, False, "sniper",
              (110, 220, 230), "One or two shots. Bolt-action, slow to reload."),
    WeaponDef("m60", "M60", "LMG", 19, 10.0, 75, 4.2, 6.5, 850, 1800, 1, True, "rifle",
              (180, 180, 190), "Huge magazine. Suppress everyone."),
    WeaponDef("m1911", "M1911", "Pistol", 24, 3.5, 12, 1.2, 2.5, 620, 1500, 1, False, "light",
              (190, 190, 200), "Reliable sidearm."),
    WeaponDef("deagle", "Desert Hawk", "Pistol", 48, 1.8, 7, 1.8, 2.0, 700, 1700, 1, False, "light",
              (250, 210, 100), "Hand cannon. Big damage per shot."),
]}
PRIMARY_KEYS = ["ak47", "m4a1", "mp5", "m1887", "awm", "m60"]
SECONDARY_KEYS = ["m1911", "deagle", "micro", "sawed"]
LOOT_GUN_WEIGHTS = {"m1911": 3, "micro": 2, "mp5": 3, "m4a1": 2, "ak47": 2, "m1887": 2,
                    "sawed": 2, "deagle": 1.5, "m60": 0.6, "awm": 0.5}
AMMO_COLORS = {"light": YELLOW, "rifle": ORANGE, "shell": RED, "sniper": CYAN}
RESERVE_CAP = {"light": 180, "rifle": 180, "shell": 40, "sniper": 30}
AMMO_PACK = {"light": 30, "rifle": 30, "shell": 8, "sniper": 5}


def weapon_score(w: WeaponDef) -> float:
    return w.damage * w.pellets * w.rate * (0.6 + w.range / 1500.0)


class Gun:
    __slots__ = ("wdef", "mag")

    def __init__(self, wdef: WeaponDef):
        self.wdef = wdef
        self.mag = wdef.mag


# ════════════════════════════════════════════════════════════════════════════
#  MAP
# ════════════════════════════════════════════════════════════════════════════
class Wall:
    __slots__ = ("x", "y", "w", "h")

    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h


class Blocker:
    __slots__ = ("x", "y", "r", "kind", "blocks_bullets")

    def __init__(self, x, y, r, kind):
        self.x, self.y, self.r, self.kind = x, y, r, kind
        self.blocks_bullets = kind == "rock"


class House:
    def __init__(self, x, y, w, h, floor):
        self.x, self.y, self.w, self.h, self.floor = x, y, w, h, floor
        self.cx, self.cy = x + w / 2, y + h / 2
        self.loot_points: List[Tuple[float, float]] = []


class GameMap:
    def __init__(self, seed: int):
        rng = random.Random(seed)
        self.size = MAP_SIZE
        self.walls: List[Wall] = []
        self.blockers: List[Blocker] = []
        self.houses: List[House] = []
        self.patches: List[tuple] = []
        self.wall_grid: Dict[Tuple[int, int], list] = {}
        self.block_grid: Dict[Tuple[int, int], list] = {}
        self.open_loot_points: List[Tuple[float, float]] = []
        self._gen_houses(rng)
        self._gen_nature(rng)
        self._gen_patches(rng)
        for w in self.walls:
            self._register(self.wall_grid, w, w.x, w.y, w.x + w.w, w.y + w.h)
        for b in self.blockers:
            self._register(self.block_grid, b, b.x - b.r, b.y - b.r, b.x + b.r, b.y + b.r)

    # -- generation --------------------------------------------------------
    def _gen_houses(self, rng):
        floors = [(150, 130, 105), (135, 140, 150), (160, 120, 110), (125, 150, 125), (170, 150, 110)]
        T = 16
        for _ in range(700):
            if len(self.houses) >= 62:
                break
            w, h = rng.randint(200, 340), rng.randint(190, 300)
            x = rng.uniform(300, MAP_SIZE - 300 - w)
            y = rng.uniform(300, MAP_SIZE - 300 - h)
            if any(x < o.x + o.w + 110 and x + w + 110 > o.x and y < o.y + o.h + 110 and y + h + 110 > o.y
                   for o in self.houses):
                continue
            hs = House(x, y, w, h, rng.choice(floors))
            gap = 78
            door = rng.choice("tblr")
            if door == "t":
                gx = rng.uniform(x + T + 10, x + w - T - 10 - gap)
                self.walls += [Wall(x, y, gx - x, T), Wall(gx + gap, y, x + w - gx - gap, T)]
            else:
                self.walls.append(Wall(x, y, w, T))
            if door == "b":
                gx = rng.uniform(x + T + 10, x + w - T - 10 - gap)
                self.walls += [Wall(x, y + h - T, gx - x, T), Wall(gx + gap, y + h - T, x + w - gx - gap, T)]
            else:
                self.walls.append(Wall(x, y + h - T, w, T))
            if door == "l":
                gy = rng.uniform(y + T + 10, y + h - T - 10 - gap)
                self.walls += [Wall(x, y, T, gy - y), Wall(x, gy + gap, T, y + h - gy - gap)]
            else:
                self.walls.append(Wall(x, y, T, h))
            if door == "r":
                gy = rng.uniform(y + T + 10, y + h - T - 10 - gap)
                self.walls += [Wall(x + w - T, y, T, gy - y), Wall(x + w - T, gy + gap, T, y + h - gy - gap)]
            else:
                self.walls.append(Wall(x + w - T, y, T, h))
            for _ in range(4):
                hs.loot_points.append((rng.uniform(x + T + 24, x + w - T - 24),
                                       rng.uniform(y + T + 24, y + h - T - 24)))
            self.houses.append(hs)

    def _near_house(self, x, y, pad):
        return any(o.x - pad < x < o.x + o.w + pad and o.y - pad < y < o.y + o.h + pad for o in self.houses)

    def _gen_nature(self, rng):
        for _ in range(1500):
            if sum(1 for b in self.blockers if b.kind == "tree") >= 300:
                break
            x, y, r = rng.uniform(80, MAP_SIZE - 80), rng.uniform(80, MAP_SIZE - 80), rng.randint(20, 32)
            if self._near_house(x, y, 70):
                continue
            if any((x - b.x) ** 2 + (y - b.y) ** 2 < (r + b.r + 24) ** 2 for b in self.blockers):
                continue
            self.blockers.append(Blocker(x, y, r, "tree"))
        for _ in range(900):
            if sum(1 for b in self.blockers if b.kind == "rock") >= 90:
                break
            x, y, r = rng.uniform(80, MAP_SIZE - 80), rng.uniform(80, MAP_SIZE - 80), rng.randint(26, 44)
            if self._near_house(x, y, 80):
                continue
            if any((x - b.x) ** 2 + (y - b.y) ** 2 < (r + b.r + 30) ** 2 for b in self.blockers):
                continue
            self.blockers.append(Blocker(x, y, r, "rock"))
        for _ in range(500):
            if len(self.open_loot_points) >= 90:
                break
            x, y = rng.uniform(150, MAP_SIZE - 150), rng.uniform(150, MAP_SIZE - 150)
            if self._near_house(x, y, 20):
                continue
            if any((x - b.x) ** 2 + (y - b.y) ** 2 < (b.r + 24) ** 2 for b in self.blockers):
                continue
            self.open_loot_points.append((x, y))

    def _gen_patches(self, rng):
        shades = [(66, 106, 58), (78, 120, 66), (72, 112, 52), (88, 124, 72), (60, 98, 60)]
        for _ in range(520):
            self.patches.append((rng.uniform(0, MAP_SIZE), rng.uniform(0, MAP_SIZE),
                                 rng.randint(50, 190), rng.randint(40, 150), rng.choice(shades)))

    def _register(self, grid, obj, x0, y0, x1, y1):
        for cy in range(int(y0 // CELL), int(y1 // CELL) + 1):
            for cx in range(int(x0 // CELL), int(x1 // CELL) + 1):
                grid.setdefault((cx, cy), []).append(obj)

    # -- queries -----------------------------------------------------------
    def visible(self, grid, x0, y0, x1, y1):
        seen, out = set(), []
        for cy in range(int(y0 // CELL), int(y1 // CELL) + 1):
            for cx in range(int(x0 // CELL), int(x1 // CELL) + 1):
                for o in grid.get((cx, cy), ()):
                    if id(o) not in seen:
                        seen.add(id(o))
                        out.append(o)
        return out

    def resolve(self, x, y, r):
        """Push a circle out of walls and blockers, and keep it inside the map."""
        for _ in range(2):
            for cy in range(int((y - r) // CELL), int((y + r) // CELL) + 1):
                for cx in range(int((x - r) // CELL), int((x + r) // CELL) + 1):
                    for w in self.wall_grid.get((cx, cy), ()):
                        nx, ny = clamp(x, w.x, w.x + w.w), clamp(y, w.y, w.y + w.h)
                        dx, dy = x - nx, y - ny
                        d2 = dx * dx + dy * dy
                        if d2 < r * r:
                            if d2 > 1e-9:
                                d = math.sqrt(d2)
                                k = (r - d) / d
                                x += dx * k
                                y += dy * k
                            else:
                                l, rr, t, b = x - w.x, w.x + w.w - x, y - w.y, w.y + w.h - y
                                m = min(l, rr, t, b)
                                if m == l: x = w.x - r
                                elif m == rr: x = w.x + w.w + r
                                elif m == t: y = w.y - r
                                else: y = w.y + w.h + r
                    for b in self.block_grid.get((cx, cy), ()):
                        dx, dy = x - b.x, y - b.y
                        d2 = dx * dx + dy * dy
                        rr = r + b.r * 0.8
                        if d2 < rr * rr:
                            d = math.sqrt(d2) if d2 > 1e-9 else 1.0
                            k = (rr - d) / d
                            x += dx * k if d2 > 1e-9 else rr
                            y += dy * k if d2 > 1e-9 else 0.0
        return clamp(x, r, MAP_SIZE - r), clamp(y, r, MAP_SIZE - r)

    def point_blocked(self, x, y):
        key = (int(x // CELL), int(y // CELL))
        for w in self.wall_grid.get(key, ()):
            if w.x <= x <= w.x + w.w and w.y <= y <= w.y + w.h:
                return True
        for b in self.block_grid.get(key, ()):
            if b.blocks_bullets and (x - b.x) ** 2 + (y - b.y) ** 2 < b.r * b.r:
                return True
        return False

    def line_clear(self, x0, y0, x1, y1, step=26.0):
        d = math.hypot(x1 - x0, y1 - y0)
        n = int(d // step)
        for i in range(1, n + 1):
            t = i / (n + 1)
            if self.point_blocked(lerp(x0, x1, t), lerp(y0, y1, t)):
                return False
        return True


def make_minimap_surface(gmap: GameMap, size: int) -> pygame.Surface:
    s = size / MAP_SIZE
    surf = pygame.Surface((size, size))
    surf.fill((58, 92, 54))
    for h in gmap.houses:
        pygame.draw.rect(surf, (185, 175, 150), (h.x * s, h.y * s, max(3, h.w * s), max(3, h.h * s)))
    for b in gmap.blockers:
        if b.kind == "rock":
            pygame.draw.circle(surf, (110, 110, 118), (int(b.x * s), int(b.y * s)), max(1, int(b.r * s)))
    pygame.draw.rect(surf, WHITE, surf.get_rect(), 2)
    return surf


# ════════════════════════════════════════════════════════════════════════════
#  LOOT, BULLETS, PLANE, ZONE
# ════════════════════════════════════════════════════════════════════════════
class Loot:
    __slots__ = ("kind", "key", "amount", "x", "y")

    def __init__(self, kind, key, amount, x, y):
        self.kind, self.key, self.amount, self.x, self.y = kind, key, amount, x, y


class Bullet:
    __slots__ = ("x", "y", "vx", "vy", "wdef", "owner", "traveled")

    def __init__(self, x, y, angle, wdef, owner):
        self.x, self.y = x, y
        self.vx, self.vy = math.cos(angle) * wdef.speed, math.sin(angle) * wdef.speed
        self.wdef, self.owner, self.traveled = wdef, owner, 0.0


class Plane:
    def __init__(self, rng):
        ang = rng.uniform(0, 2 * math.pi)
        dx, dy = math.cos(ang), math.sin(ang)
        off = rng.uniform(-900, 900)
        cx = MAP_SIZE / 2 - dy * off
        cy = MAP_SIZE / 2 + dx * off
        half = MAP_SIZE * 0.62
        self.x0, self.y0 = cx - dx * half, cy - dy * half
        self.x1, self.y1 = cx + dx * half, cy + dy * half
        self.length = math.hypot(self.x1 - self.x0, self.y1 - self.y0)
        self.heading = ang
        self.t = 0.0
        self.x, self.y = self.x0, self.y0

    def update(self, dt):
        self.t = min(1.0, self.t + dt * PLANE_SPEED / self.length)
        self.x = lerp(self.x0, self.x1, self.t)
        self.y = lerp(self.y0, self.y1, self.t)

    @property
    def done(self):
        return self.t >= 1.0

    def dist_to_path(self, x, y):
        return math.sqrt(seg_point_dist2(self.x0, self.y0, self.x1, self.y1, x, y))


class Zone:
    RADII = [2000, 1300, 800, 450, 220, 70]
    WAIT = [55, 28, 22, 16, 12, 8]
    SHRINK = [30, 26, 20, 15, 12, 8]
    DMG = [1.5, 2.5, 4, 6, 9, 14]

    def __init__(self, rng):
        self.rng = rng
        self.c = (MAP_SIZE / 2, MAP_SIZE / 2)
        self.r = MAP_SIZE * 0.72
        self.phase = 0
        self.mode = "wait"
        self.timer = self.WAIT[0]
        self.start_c, self.start_r = self.c, self.r
        self.tc, self.tr = self.c, self.r
        self._pick_target()

    def _pick_target(self):
        r_new = self.RADII[self.phase]
        max_off = max(0.0, self.r - r_new) * 0.85
        a, d = self.rng.uniform(0, 2 * math.pi), self.rng.uniform(0, max_off)
        self.tc = (clamp(self.c[0] + math.cos(a) * d, 350, MAP_SIZE - 350),
                   clamp(self.c[1] + math.sin(a) * d, 350, MAP_SIZE - 350))
        self.tr = r_new

    def update(self, dt):
        if self.phase >= len(self.RADII):
            return
        self.timer -= dt
        if self.mode == "wait":
            if self.timer <= 0:
                self.mode, self.timer = "shrink", self.SHRINK[self.phase]
                self.start_c, self.start_r = self.c, self.r
        else:
            f = clamp(1.0 - self.timer / self.SHRINK[self.phase], 0.0, 1.0)
            self.c = (lerp(self.start_c[0], self.tc[0], f), lerp(self.start_c[1], self.tc[1], f))
            self.r = lerp(self.start_r, self.tr, f)
            if self.timer <= 0:
                self.c, self.r = self.tc, self.tr
                self.phase += 1
                if self.phase < len(self.RADII):
                    self.mode, self.timer = "wait", self.WAIT[self.phase]
                    self._pick_target()

    @property
    def damage(self):
        return self.DMG[min(self.phase, len(self.DMG) - 1)]

    def inside(self, x, y):
        return math.hypot(x - self.c[0], y - self.c[1]) <= self.r


# ════════════════════════════════════════════════════════════════════════════
#  FIGHTERS (player + bots)
# ════════════════════════════════════════════════════════════════════════════
class Fighter:
    is_bot = False

    def __init__(self, uid: int, name: str, color):
        self.uid, self.name, self.color = uid, name, color
        self.is_player = False
        self.x = self.y = 0.0
        self.vx = self.vy = 0.0
        self.angle = 0.0
        self.hp = BASE_HP
        self.armor = 0.0
        self.state = "plane"                  # plane / fall / ground / dead
        self.alt = 1.0
        self.guns: List[Optional[Gun]] = [None, None]
        self.cur = 0
        self.reserve = {"light": 0, "rifle": 0, "shell": 0, "sniper": 0}
        self.medkits = 1
        self.fire_cd = 0.0
        self.reload_t = 0.0
        self.heal_t = 0.0
        self.flash = 0.0
        self.hit_flash = 0.0
        self.pick_t = 0.0
        self.trigger = False
        self._trigger_prev = False
        self.want = (0.0, 0.0)
        self.sprint = False
        self.boost = False
        self.moving = False
        self.kills = 0
        self.damage_dealt = 0.0
        self.rank = 0
        self.survive_time = 0.0

    # -- inventory ---------------------------------------------------------
    @property
    def gun(self) -> Optional[Gun]:
        return self.guns[self.cur]

    def give_loadout(self, primary: str, secondary: str):
        self.guns = [Gun(WEAPONS[primary]), Gun(WEAPONS[secondary])]
        self.cur = 0
        self.reserve = {"light": 90, "rifle": 90, "shell": 20, "sniper": 10}

    def add_ammo(self, kind: str, n: int) -> bool:
        if self.reserve[kind] >= RESERVE_CAP[kind]:
            return False
        self.reserve[kind] = min(RESERVE_CAP[kind], self.reserve[kind] + n)
        return True

    def switch_to(self, idx: int):
        if idx != self.cur and self.guns[idx] is not None:
            self.cur = idx
            self.reload_t = 0.0
            self.fire_cd = 0.25

    def start_reload(self) -> bool:
        g = self.gun
        if g and self.reload_t <= 0 and g.mag < g.wdef.mag and self.reserve[g.wdef.ammo] > 0:
            self.reload_t = g.wdef.reload
            if self.is_player:
                play("reload")
            return True
        return False

    def _finish_reload(self):
        g = self.gun
        if g:
            take = min(g.wdef.mag - g.mag, self.reserve[g.wdef.ammo])
            g.mag += take
            self.reserve[g.wdef.ammo] -= take

    def use_medkit(self) -> bool:
        if self.state == "ground" and self.medkits > 0 and self.heal_t <= 0 and self.hp < BASE_HP:
            self.medkits -= 1
            self.heal_t = MEDKIT_TIME
            self.reload_t = 0.0
            return True
        return False

    # -- combat ------------------------------------------------------------
    def take_damage(self, m: "Match", dmg: float, attacker: Optional["Fighter"], wname: str):
        if self.state != "ground":
            return
        if self.armor > 0:
            ab = min(self.armor, dmg * 0.5)
            self.armor -= ab
            dmg -= ab
        self.hp -= dmg
        self.hit_flash = 0.18
        if attacker is not None and attacker is not self:
            attacker.damage_dealt += dmg
            if attacker.is_player:
                play("hit", 0.7)
                m.add_float(self.x, self.y - 26, f"-{int(dmg)}", WHITE)
        if self.is_player:
            play("hurt", 0.8)
        if self.hp <= 0:
            m.kill(self, attacker, wname)

    def _fire(self, m: "Match"):
        w = self.gun.wdef
        self.gun.mag -= 1
        self.fire_cd = 1.0 / w.rate
        spread = math.radians(w.spread) * (1.5 if self.moving else 1.0)
        for _ in range(w.pellets):
            m.spawn_bullet(self, self.angle + random.uniform(-spread, spread), w)
        self.flash = 0.05
        m.on_shot(self, w)

    def update_weapon(self, m: "Match", dt: float):
        self.fire_cd = max(0.0, self.fire_cd - dt)
        self.flash = max(0.0, self.flash - dt)
        if self.reload_t > 0:
            self.reload_t -= dt
            if self.reload_t <= 0:
                self._finish_reload()
        g = self.gun
        pressed = self.trigger and (self.is_bot or not self._trigger_prev)
        want = self.trigger if (g and g.wdef.auto) else pressed
        self._trigger_prev = self.trigger
        if g and want and self.reload_t <= 0 and self.heal_t <= 0 and self.fire_cd <= 0:
            if g.mag > 0:
                self._fire(m)
            else:
                self.start_reload()

    # -- movement ----------------------------------------------------------
    def step(self, m: "Match", dt: float):
        ox, oy = self.x, self.y
        self.hit_flash = max(0.0, self.hit_flash - dt)
        wx, wy = self.want
        L = math.hypot(wx, wy)
        if L > 1e-6:
            wx, wy = wx / L, wy / L
        if self.state == "fall":
            fast = self.alt > 0.55
            speed = (400 if self.boost else 260) if fast else 210
            drop = (0.15 if self.boost else 0.10) if fast else 0.075
            self.x = clamp(self.x + wx * speed * dt, 10, MAP_SIZE - 10)
            self.y = clamp(self.y + wy * speed * dt, 10, MAP_SIZE - 10)
            self.alt -= drop * dt
            self.moving = L > 1e-6
            if self.alt <= 0:
                m.land(self)
        elif self.state == "ground":
            speed = SPRINT_SPEED if self.sprint else WALK_SPEED
            if self.heal_t > 0:
                speed *= 0.5
                self.heal_t -= dt
                if self.heal_t <= 0:
                    self.hp = min(BASE_HP, self.hp + MEDKIT_HEAL)
            self.moving = L > 1e-6
            if self.moving:
                self.x, self.y = m.map.resolve(self.x + wx * speed * dt, self.y + wy * speed * dt, PLAYER_R)
            self.update_weapon(m, dt)
            self.pick_t -= dt
            if self.pick_t <= 0:
                self.pick_t = 0.15
                m.pickup(self, manual=False)
        if dt > 0:
            self.vx, self.vy = (self.x - ox) / dt, (self.y - oy) / dt


class Bot(Fighter):
    is_bot = True

    def __init__(self, uid, name, color, diff: str, rng: random.Random):
        super().__init__(uid, name, color)
        self.rng = rng
        self.skill = DIFFICULTY[diff]
        self.land_target = (MAP_SIZE / 2, MAP_SIZE / 2)
        self.jump_dist = rng.uniform(1200, 2000)
        self.think_t = rng.uniform(0, 0.3)
        self.mode = "roam"
        self.target: Optional[Fighter] = None
        self.react = 0.0
        self.aim_err = 0.0
        self.dest: Optional[Tuple[float, float]] = None
        self.strafe = rng.choice((-1, 1))
        self.strafe_t = rng.uniform(0.6, 1.5)
        self.stuck_t = 0.0
        self.last_pos = (0.0, 0.0)
        self.escape_t = 0.0
        self.escape_dir = (0.0, 0.0)
        self.intent = False

    def ai(self, m: "Match", dt: float):
        self.want = (0.0, 0.0)
        self.sprint = self.boost = self.trigger = False
        if self.state == "plane":
            d = math.hypot(m.plane.x - self.land_target[0], m.plane.y - self.land_target[1])
            if d < self.jump_dist or m.plane.t > 0.93:
                m.jump(self)
        elif self.state == "fall":
            dx, dy = self.land_target[0] - self.x, self.land_target[1] - self.y
            d = math.hypot(dx, dy)
            if d > 30:
                self.want = (dx / d, dy / d)
                self.angle = math.atan2(dy, dx)
            self.boost = d > 500 and self.alt > 0.55
        elif self.state == "ground":
            self.think_t -= dt
            if self.think_t <= 0:
                self.think_t = self.rng.uniform(0.2, 0.4)
                self._think(m)
            self._act(m, dt)

    # -- decision making ---------------------------------------------------
    def _wants(self, l: Loot) -> bool:
        if l.kind == "gun":
            if any(g and g.wdef.key == l.key for g in self.guns):
                return False
            if self.guns[0] is None or self.guns[1] is None:
                return True
            worst = min(weapon_score(g.wdef) for g in self.guns if g)
            return weapon_score(WEAPONS[l.key]) > worst * 1.25
        if l.kind == "ammo":
            return self.reserve[l.key] < 80 and any(g and g.wdef.ammo == l.key for g in self.guns)
        if l.kind == "medkit":
            return self.medkits < 3
        return self.armor < 60

    def _find_loot(self, m: "Match") -> Optional[Loot]:
        best, bd = None, 900.0 ** 2
        for l in m.loot:
            d2 = (l.x - self.x) ** 2 + (l.y - self.y) ** 2
            if d2 < bd and self._wants(l):
                best, bd = l, d2
        return best

    def _think(self, m: "Match"):
        rng, z = self.rng, m.zone
        best, bd = None, self.skill["sight"] ** 2
        for o in m.fighters:
            if o is self or o.state != "ground":
                continue
            d2 = (o.x - self.x) ** 2 + (o.y - self.y) ** 2
            if d2 < bd and m.map.line_clear(self.x, self.y, o.x, o.y):
                best, bd = o, d2
        if best is not None and self.target is None:
            self.react = self.skill["react"]
        self.target = best
        self.aim_err = rng.gauss(0, self.skill["aim"])

        out_now = math.hypot(self.x - z.c[0], self.y - z.c[1]) > z.r * 0.92
        zd = math.hypot(self.x - z.tc[0], self.y - z.tc[1])
        zone_bad = out_now or (zone_bad_soon(z) and zd > z.tr * 0.85)
        if best is not None and not (self.hp < 30 and self.medkits > 0 and bd > 300 ** 2):
            self.mode = "fight"
        elif zone_bad:
            if self.mode != "zone" or self.dest is None:
                self.dest = (z.tc[0] + rng.uniform(-1, 1) * z.tr * 0.5, z.tc[1] + rng.uniform(-1, 1) * z.tr * 0.5)
            self.mode = "zone"
        elif self.hp < 75 and self.medkits > 0:
            self.mode = "heal"
        else:
            loot = self._find_loot(m)
            if loot is not None:
                self.mode, self.dest = "loot", (loot.x, loot.y)
            else:
                if self.mode != "roam" or self.dest is None:
                    a, d = rng.uniform(0, 2 * math.pi), rng.uniform(200, 900)
                    tx, ty = self.x + math.cos(a) * d, self.y + math.sin(a) * d
                    if not z.inside(tx, ty):
                        tx, ty = z.c[0] + rng.uniform(-1, 1) * z.r * 0.5, z.c[1] + rng.uniform(-1, 1) * z.r * 0.5
                    self.dest = (clamp(tx, 100, MAP_SIZE - 100), clamp(ty, 100, MAP_SIZE - 100))
                self.mode = "roam"

    def _act(self, m: "Match", dt: float):
        g, tgt = self.gun, self.target
        self.intent = False
        if g and g.mag == 0 and self.reserve[g.wdef.ammo] == 0:      # dry: swap to the other gun
            o = self.guns[1 - self.cur]
            if o and (o.mag > 0 or self.reserve[o.wdef.ammo] > 0):
                self.switch_to(1 - self.cur)
        if self.mode == "fight" and tgt is not None and tgt.state == "ground":
            dx, dy = tgt.x - self.x, tgt.y - self.y
            d = max(1.0, math.hypot(dx, dy))
            lead = d / (g.wdef.speed if g else 1500.0)
            desired = math.atan2(dy + tgt.vy * lead, dx + tgt.vx * lead) + self.aim_err
            diff = angle_diff(self.angle, desired)
            turn = self.skill["turn"] * dt
            self.angle += clamp(diff, -turn, turn)
            pref = clamp(g.wdef.range * 0.45, 140, 480) if g else 120
            ux, uy = dx / d, dy / d
            mx = my = 0.0
            if d > pref + 60:
                mx, my = ux, uy
            elif d < pref - 90:
                mx, my = -ux, -uy
            self.strafe_t -= dt
            if self.strafe_t <= 0:
                self.strafe = -self.strafe
                self.strafe_t = self.rng.uniform(0.6, 1.6)
            mx += -uy * self.strafe * 0.8
            my += ux * self.strafe * 0.8
            self.want = (mx, my)
            self.intent = True
            self.react -= dt
            if g and self.react <= 0 and d < g.wdef.range * 0.95 and abs(diff) < 0.22:
                self.trigger = True
        elif self.mode == "heal":
            self.use_medkit()
        elif self.dest is not None:
            dx, dy = self.dest[0] - self.x, self.dest[1] - self.y
            d = math.hypot(dx, dy)
            if d > 30:
                self.want = (dx / d, dy / d)
                self.intent = True
                diff = angle_diff(self.angle, math.atan2(dy, dx))
                self.angle += clamp(diff, -self.skill["turn"] * dt, self.skill["turn"] * dt)
                self.sprint = self.mode == "zone" and not m.zone.inside(self.x, self.y)
            else:
                self.dest = None
            if self.mode == "loot" and d < PICKUP_RANGE:
                m.pickup(self, manual=True)
        if self.target is None and g and g.mag < g.wdef.mag * 0.4:
            self.start_reload()
        # stuck detection
        self.stuck_t += dt
        if self.stuck_t >= 1.0:
            self.stuck_t = 0.0
            if self.intent and math.hypot(self.x - self.last_pos[0], self.y - self.last_pos[1]) < 20:
                a = self.rng.uniform(0, 2 * math.pi)
                self.escape_t, self.escape_dir, self.dest = 0.8, (math.cos(a), math.sin(a)), None
            self.last_pos = (self.x, self.y)
        if self.escape_t > 0:
            self.escape_t -= dt
            self.want = self.escape_dir


def zone_bad_soon(z: Zone) -> bool:
    return z.mode == "shrink" or (z.mode == "wait" and z.timer < 12)


# ════════════════════════════════════════════════════════════════════════════
#  MATCH  (the whole simulation)
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class Profile:
    name: str = "Player"
    color_idx: int = 0
    primary: str = "m4a1"
    secondary: str = "m1911"
    players: int = 20
    difficulty: str = "Normal"


@dataclass
class PlayerInput:
    move: Tuple[float, float] = (0.0, 0.0)
    aim: float = 0.0
    fire: bool = False
    sprint: bool = False
    jump: bool = False
    reload: bool = False
    medkit: bool = False
    pickup: bool = False
    switch: Optional[int] = None      # 0 / 1 = slot, -1 = toggle


class Match:
    def __init__(self, profile: Profile, player_is_bot: bool = False, seed: Optional[int] = None):
        seed = seed if seed is not None else random.randrange(1 << 30)
        self.rng = random.Random(seed)
        self.map = GameMap(seed)
        self.plane = Plane(self.rng)
        self.zone = Zone(self.rng)
        self.fighters: List[Fighter] = []
        self.bullets: List[Bullet] = []
        self.loot: List[Loot] = []
        self.floats: List[list] = []
        self.sparks: List[list] = []
        self.feed: List[list] = []
        self.time = 0.0
        self.state = "playing"               # playing / over
        self.winner: Optional[Fighter] = None
        self.total = profile.players
        self.over_t = 0.0

        color = PLAYER_COLORS[profile.color_idx % len(PLAYER_COLORS)]
        if player_is_bot:
            p: Fighter = Bot(0, profile.name, color, profile.difficulty, self.rng)
        else:
            p = Fighter(0, profile.name, color)
        p.is_player = True
        p.give_loadout(profile.primary, profile.secondary)
        self.player = p
        self.fighters.append(p)
        names = [n for n in BOT_NAMES if n != profile.name]
        self.rng.shuffle(names)
        palette = list(PLAYER_COLORS)
        for i in range(1, profile.players):
            b = Bot(i, names[(i - 1) % len(names)] + (str(i) if i > len(names) else ""),
                    palette[(i + 2) % len(palette)], profile.difficulty, self.rng)
            b.give_loadout(self.rng.choice(PRIMARY_KEYS), self.rng.choice(SECONDARY_KEYS))
            self.fighters.append(b)
        for f in self.fighters:
            if f.is_bot:
                f.land_target = self._choose_land_target()
        self._spawn_loot()
        self.player_feed(f"{profile.players} players are in the air!")

    # -- setup ---------------------------------------------------------------
    def _choose_land_target(self):
        cands = [h for h in self.map.houses if self.plane.dist_to_path(h.cx, h.cy) < 1500]
        if cands and self.rng.random() < 0.8:
            h = self.rng.choice(cands)
            return (h.cx + self.rng.uniform(-40, 40), h.cy + self.rng.uniform(-40, 40))
        return (self.rng.uniform(600, MAP_SIZE - 600), self.rng.uniform(600, MAP_SIZE - 600))

    def _roll_loot(self, x, y):
        r = self.rng.random()
        if r < 0.38:
            keys = list(LOOT_GUN_WEIGHTS)
            key = self.rng.choices(keys, weights=[LOOT_GUN_WEIGHTS[k] for k in keys])[0]
            return Loot("gun", key, 1, x, y)
        if r < 0.70:
            kind = self.rng.choice(list(AMMO_PACK))
            return Loot("ammo", kind, AMMO_PACK[kind], x, y)
        if r < 0.86:
            return Loot("medkit", "medkit", 1, x, y)
        return Loot("armor", "armor", 50, x, y)

    def _spawn_loot(self):
        for h in self.map.houses:
            for (x, y) in h.loot_points:
                self.loot.append(self._roll_loot(x, y))
        for (x, y) in self.map.open_loot_points:
            self.loot.append(self._roll_loot(x, y))

    # -- events --------------------------------------------------------------
    def player_feed(self, text, color=WHITE):
        self.feed.append([text, 6.0, color])
        self.feed = self.feed[-6:]

    def add_float(self, x, y, text, color):
        self.floats.append([x, y, text, 0.9, color])

    def alive_count(self):
        return sum(1 for f in self.fighters if f.state != "dead")

    def jump(self, f: Fighter):
        if f.state != "plane":
            return
        f.state, f.alt = "fall", 1.0
        f.x, f.y = self.plane.x, self.plane.y
        if f.is_player:
            play("jump")

    def land(self, f: Fighter):
        f.state, f.alt = "ground", 0.0
        f.x, f.y = self.map.resolve(f.x, f.y, PLAYER_R)

    def spawn_bullet(self, owner: Fighter, angle: float, w: WeaponDef):
        self.bullets.append(Bullet(owner.x + math.cos(angle) * 22, owner.y + math.sin(angle) * 22, angle, w, owner))

    def on_shot(self, f: Fighter, w: WeaponDef):
        d = math.hypot(f.x - self.player.x, f.y - self.player.y)
        vol = 0.9 if f.is_player else clamp(1.0 - d / 1500.0, 0.0, 1.0) * 0.8
        if vol > 0.05:
            play("shotgun" if w.kind == "Shotgun" else "sniper" if w.kind == "Sniper" else "shot", vol)

    def kill(self, victim: Fighter, attacker: Optional[Fighter], wname: str):
        if victim.state == "dead":
            return
        victim.rank = self.alive_count()
        victim.state = "dead"
        # drop everything
        for g in victim.guns:
            if g:
                self.loot.append(Loot("gun", g.wdef.key, 1, victim.x + self.rng.uniform(-30, 30),
                                      victim.y + self.rng.uniform(-30, 30)))
        for kind, n in victim.reserve.items():
            if n >= 10:
                self.loot.append(Loot("ammo", kind, min(n, 60), victim.x + self.rng.uniform(-30, 30),
                                      victim.y + self.rng.uniform(-30, 30)))
        if victim.armor > 10:
            self.loot.append(Loot("armor", "armor", int(victim.armor), victim.x, victim.y + 24))
        for _ in range(victim.medkits):
            self.loot.append(Loot("medkit", "medkit", 1, victim.x + self.rng.uniform(-30, 30),
                                  victim.y + self.rng.uniform(-30, 30)))
        for _ in range(14):
            a, s = self.rng.uniform(0, 6.28), self.rng.uniform(40, 200)
            self.sparks.append([victim.x, victim.y, math.cos(a) * s, math.sin(a) * s, 0.5, victim.color])
        if attacker is not None and attacker is not victim:
            attacker.kills += 1
            self.player_feed(f"{attacker.name}  [{wname}]  {victim.name}",
                             GOLD if attacker.is_player or victim.is_player else WHITE)
            if attacker.is_player:
                play("kill")
        else:
            self.player_feed(f"{victim.name} was caught in the storm", GRAY)
        if self.alive_count() <= 1 and self.state == "playing":
            self.state = "over"
            alive = [f for f in self.fighters if f.state != "dead"]
            self.winner = alive[0] if alive else attacker
            if self.winner is not None:
                self.winner.rank = 1

    def pickup(self, f: Fighter, manual: bool) -> bool:
        got = False
        r2 = PICKUP_RANGE ** 2
        for l in list(self.loot):
            if (l.x - f.x) ** 2 + (l.y - f.y) ** 2 > r2:
                continue
            if l.kind == "ammo":
                ok = f.add_ammo(l.key, l.amount)
            elif l.kind == "medkit":
                ok = f.medkits < MAX_MEDKITS
                if ok:
                    f.medkits += 1
            elif l.kind == "armor":
                ok = f.armor < MAX_ARMOR
                if ok:
                    f.armor = min(MAX_ARMOR, f.armor + l.amount)
            else:
                ok = manual and self._take_gun(f, l)
            if ok:
                self.loot.remove(l)
                got = True
                if l.kind == "gun":
                    break
        if got and f.is_player:
            play("pickup", 0.8)
        return got

    def _take_gun(self, f: Fighter, l: Loot) -> bool:
        wd = WEAPONS[l.key]
        for g in f.guns:
            if g and g.wdef.key == l.key:
                f.add_ammo(wd.ammo, wd.mag * 2)
                return True
        if f.guns[f.cur] is None:
            slot = f.cur
        elif f.guns[1 - f.cur] is None:
            slot = 1 - f.cur
        else:
            slot = f.cur
        old = f.guns[slot]
        f.guns[slot] = Gun(wd)
        f.cur = slot
        f.reload_t = 0.0
        if old:
            self.loot.append(Loot("gun", old.wdef.key, 1, f.x + self.rng.uniform(-24, 24),
                                  f.y + self.rng.uniform(-24, 24)))
        return True

    def nearest_gun_loot(self, f: Fighter) -> Optional[Loot]:
        best, bd = None, PICKUP_RANGE ** 2
        for l in self.loot:
            if l.kind == "gun":
                d2 = (l.x - f.x) ** 2 + (l.y - f.y) ** 2
                if d2 < bd:
                    best, bd = l, d2
        return best

    # -- update --------------------------------------------------------------
    def _player_control(self, f: Fighter, inp: PlayerInput):
        f.want = inp.move
        f.sprint = inp.sprint
        f.boost = inp.sprint
        f.trigger = inp.fire
        f.angle = inp.aim
        if f.state == "plane":
            if inp.jump or self.plane.done:
                self.jump(f)
        elif f.state == "ground":
            if inp.reload:
                f.start_reload()
            if inp.medkit:
                f.use_medkit()
            if inp.switch is not None:
                f.switch_to(1 - f.cur if inp.switch < 0 else inp.switch)
            if inp.pickup:
                self.pickup(f, manual=True)

    def update(self, dt: float, inp: PlayerInput):
        self.time += dt
        if not self.plane.done:
            self.plane.update(dt)
        self.zone.update(dt)

        for f in self.fighters:
            if f.state == "dead":
                continue
            if f.is_player:
                f.survive_time += dt
            if f.state == "plane":
                f.x, f.y, f.angle = self.plane.x, self.plane.y, self.plane.heading
            if f.is_bot:
                f.ai(self, dt)
            else:
                self._player_control(f, inp)
            if f.state != "plane":
                f.step(self, dt)
            if f.state == "ground" and not self.zone.inside(f.x, f.y):
                f.take_damage(self, self.zone.damage * dt, None, "storm")

        self._separate()
        self._update_bullets(dt)
        for s in self.sparks:
            s[0] += s[2] * dt; s[1] += s[3] * dt; s[4] -= dt
        self.sparks = [s for s in self.sparks if s[4] > 0]
        for fl in self.floats:
            fl[1] -= 34 * dt
            fl[3] -= dt
        self.floats = [fl for fl in self.floats if fl[3] > 0]
        for e in self.feed:
            e[1] -= dt
        self.feed = [e for e in self.feed if e[1] > 0]
        if self.state == "over":
            self.over_t += dt

    def _separate(self):
        g = [f for f in self.fighters if f.state == "ground"]
        for i in range(len(g)):
            a = g[i]
            for j in range(i + 1, len(g)):
                b = g[j]
                dx, dy = b.x - a.x, b.y - a.y
                d2 = dx * dx + dy * dy
                if 0.01 < d2 < (PLAYER_R * 2) ** 2:
                    d = math.sqrt(d2)
                    k = (PLAYER_R * 2 - d) / d * 0.5
                    a.x -= dx * k; a.y -= dy * k
                    b.x += dx * k; b.y += dy * k

    def _update_bullets(self, dt: float):
        alive: List[Bullet] = []
        targets = [f for f in self.fighters if f.state == "ground"]
        hit_r2 = (PLAYER_R + 2) ** 2
        for b in self.bullets:
            step = b.wdef.speed * dt
            x1, y1 = b.x + b.vx * dt, b.y + b.vy * dt
            n = int(step // 10) + 1
            blocked = False
            for i in range(1, n + 1):
                px, py = lerp(b.x, x1, i / n), lerp(b.y, y1, i / n)
                if self.map.point_blocked(px, py):
                    x1, y1, blocked = px, py, True
                    break
            hit: Optional[Fighter] = None
            for f in targets:
                if f is b.owner:
                    continue
                if seg_point_dist2(b.x, b.y, x1, y1, f.x, f.y) <= hit_r2:
                    hit = f
                    break
            b.traveled += step
            if hit is not None:
                w = b.wdef
                fall = 1.0 if b.traveled < w.range * 0.5 else lerp(1.0, 0.55, (b.traveled - w.range * 0.5) / (w.range * 0.5))
                for _ in range(3):
                    a = self.rng.uniform(0, 6.28)
                    self.sparks.append([hit.x, hit.y, math.cos(a) * 90, math.sin(a) * 90, 0.25, RED])
                hit.take_damage(self, w.damage * clamp(fall, 0.3, 1.0), b.owner, w.name)
                continue
            if blocked:
                self.sparks.append([x1, y1, 0, 0, 0.15, (255, 230, 150)])
                continue
            if b.traveled >= b.wdef.range or not (0 <= x1 <= MAP_SIZE and 0 <= y1 <= MAP_SIZE):
                continue
            b.x, b.y = x1, y1
            alive.append(b)
        self.bullets = alive


# ════════════════════════════════════════════════════════════════════════════
#  RENDERING
# ════════════════════════════════════════════════════════════════════════════
def draw_text(screen, font, s, color, pos, anchor="topleft", shadow=True):
    img = font.render(str(s), True, color)
    r = img.get_rect(**{anchor: pos})
    if shadow:
        sh = font.render(str(s), True, (0, 0, 0))
        screen.blit(sh, r.move(2, 2))
    screen.blit(img, r)
    return r


def draw_bar(screen, x, y, w, h, frac, color, back=(30, 30, 40)):
    pygame.draw.rect(screen, back, (x, y, w, h), border_radius=3)
    if frac > 0:
        pygame.draw.rect(screen, color, (x, y, max(2, int(w * clamp(frac, 0, 1))), h), border_radius=3)
    pygame.draw.rect(screen, (0, 0, 0), (x, y, w, h), 1, border_radius=3)


PLANE_SHAPE = [(70, 0), (22, -9), (8, -62), (-8, -62), (-2, -12), (-42, -8), (-54, -24), (-62, -24),
               (-58, 0), (-62, 24), (-54, 24), (-42, 8), (-2, 12), (-8, 62), (8, 62), (22, 9)]


class Renderer:
    def __init__(self, fonts):
        self.fonts = fonts
        self.mini_small: Optional[pygame.Surface] = None
        self.mini_big: Optional[pygame.Surface] = None
        self.mini_key = None
        self.cam = (0.0, 0.0)
        self.overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        self.hurt = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        pygame.draw.rect(self.hurt, (255, 0, 0, 90), self.hurt.get_rect(), 26)

    def prepare(self, m: Match):
        if self.mini_key is not id(m):
            self.mini_small = make_minimap_surface(m.map, 210)
            self.mini_big = make_minimap_surface(m.map, 640)
            self.mini_key = id(m)

    # -- world ---------------------------------------------------------------
    def draw(self, screen, m: Match, mouse_pos, show_map: bool, t: float):
        self.prepare(m)
        p = m.player
        camx, camy = p.x - SCREEN_W / 2, p.y - SCREEN_H / 2
        self.cam = (camx, camy)
        screen.fill((20, 32, 22))
        pygame.draw.rect(screen, (70, 110, 60), (-camx, -camy, MAP_SIZE, MAP_SIZE))
        self._draw_ground(screen, m, camx, camy)
        vis = (camx - 60, camy - 60, camx + SCREEN_W + 60, camy + SCREEN_H + 60)
        for h in m.map.houses:
            if h.x < vis[2] and h.x + h.w > vis[0] and h.y < vis[3] and h.y + h.h > vis[1]:
                pygame.draw.rect(screen, h.floor, (h.x - camx, h.y - camy, h.w, h.h))
                pygame.draw.rect(screen, tuple(max(0, c - 25) for c in h.floor),
                                 (h.x - camx + 20, h.y - camy + 20, h.w - 40, h.h - 40), 2)
        for w in m.map.visible(m.map.wall_grid, *vis):
            pygame.draw.rect(screen, (70, 66, 74), (w.x - camx, w.y - camy, w.w, w.h))
            pygame.draw.rect(screen, (34, 32, 38), (w.x - camx, w.y - camy, w.w, w.h), 2)
        for b in m.map.visible(m.map.block_grid, *vis):
            sx, sy = int(b.x - camx), int(b.y - camy)
            if b.kind == "tree":
                pygame.draw.circle(screen, (30, 80, 40), (sx + 4, sy + 6), b.r)
                pygame.draw.circle(screen, (46, 118, 58), (sx, sy), b.r)
                pygame.draw.circle(screen, (66, 150, 78), (sx - b.r // 3, sy - b.r // 3), b.r // 2)
            else:
                pygame.draw.circle(screen, (76, 76, 84), (sx, sy), b.r)
                pygame.draw.circle(screen, (120, 120, 130), (sx - b.r // 4, sy - b.r // 4), b.r // 2)
        self._draw_loot(screen, m, camx, camy, vis, t)
        for f in sorted(m.fighters, key=lambda q: q.y):
            if f.state in ("ground", "fall"):
                self._draw_fighter(screen, f, camx, camy, f is p)
        self._draw_bullets(screen, m, camx, camy)
        if not m.plane.done:
            self._draw_plane(screen, m.plane, camx, camy)
        self._draw_zone(screen, m, camx, camy)
        for fl in m.floats:
            draw_text(screen, self.fonts["small"], fl[2], fl[4], (fl[0] - camx, fl[1] - camy), "center")
        self._draw_hud(screen, m, mouse_pos, t)
        if show_map:
            self._draw_big_map(screen, m)

    def _draw_ground(self, screen, m, camx, camy):
        for (x, y, rx, ry, col) in m.map.patches:
            if -rx < x - camx < SCREEN_W + rx and -ry < y - camy < SCREEN_H + ry:
                pygame.draw.ellipse(screen, col, (x - camx - rx, y - camy - ry, rx * 2, ry * 2))
        gs = 260
        x0 = int(camx // gs) * gs
        for x in range(x0, int(camx + SCREEN_W) + gs, gs):
            if 0 <= x <= MAP_SIZE:
                pygame.draw.line(screen, (62, 100, 54), (x - camx, max(0, -camy)),
                                 (x - camx, min(SCREEN_H, MAP_SIZE - camy)), 1)
        y0 = int(camy // gs) * gs
        for y in range(y0, int(camy + SCREEN_H) + gs, gs):
            if 0 <= y <= MAP_SIZE:
                pygame.draw.line(screen, (62, 100, 54), (max(0, -camx), y - camy),
                                 (min(SCREEN_W, MAP_SIZE - camx), y - camy), 1)

    def _draw_loot(self, screen, m, camx, camy, vis, t):
        p = m.player
        for l in m.loot:
            if not (vis[0] < l.x < vis[2] and vis[1] < l.y < vis[3]):
                continue
            sx, sy = int(l.x - camx), int(l.y - camy + math.sin(t * 3 + l.x) * 2)
            if l.kind == "gun":
                col, ch = WEAPONS[l.key].color, "G"
            elif l.kind == "ammo":
                col, ch = AMMO_COLORS[l.key], "A"
            elif l.kind == "medkit":
                col, ch = (235, 235, 235), "+"
            else:
                col, ch = BLUE, "S"
            pygame.draw.circle(screen, (0, 0, 0), (sx, sy + 2), 13)
            pygame.draw.circle(screen, col, (sx, sy), 12)
            pygame.draw.circle(screen, WHITE, (sx, sy), 12, 2)
            draw_text(screen, self.fonts["small"], ch, BLACK if l.kind != "armor" else WHITE, (sx, sy), "center", False)
            if l.kind == "gun" and p.state == "ground" and (l.x - p.x) ** 2 + (l.y - p.y) ** 2 < 130 ** 2:
                draw_text(screen, self.fonts["small"], WEAPONS[l.key].name, WHITE, (sx, sy - 22), "center")

    def _draw_fighter(self, screen, f: Fighter, camx, camy, local):
        sx, sy = f.x - camx, f.y - camy
        if f.state == "fall":
            lift = f.alt * 70
            pygame.draw.circle(screen, (0, 0, 0), (int(sx + lift * 0.3), int(sy + lift * 0.3)), 12)
            sy -= lift
            if f.alt < 0.55:
                pygame.draw.circle(screen, f.color, (int(sx), int(sy - 34)), 40)
                pygame.draw.circle(screen, WHITE, (int(sx), int(sy - 34)), 40, 3)
                for ax in (-36, 0, 36):
                    pygame.draw.line(screen, (230, 230, 230), (sx + ax, sy - 30), (sx, sy), 1)
            pygame.draw.circle(screen, f.color, (int(sx), int(sy)), PLAYER_R)
            pygame.draw.circle(screen, WHITE, (int(sx), int(sy)), PLAYER_R, 2)
            draw_text(screen, self.fonts["small"], f.name, WHITE, (sx, sy + 26), "center")
            return
        ca, sa = math.cos(f.angle), math.sin(f.angle)
        pygame.draw.circle(screen, (0, 0, 0), (int(sx + 3), int(sy + 4)), PLAYER_R)
        g = f.gun
        gcol = g.wdef.color if g else GRAY
        glen = 34 if g and g.wdef.kind in ("Sniper", "LMG", "Assault Rifle") else 28
        pygame.draw.line(screen, (30, 30, 30), (sx + ca * 6, sy + sa * 6), (sx + ca * glen, sy + sa * glen), 7)
        pygame.draw.line(screen, gcol, (sx + ca * 6, sy + sa * 6), (sx + ca * glen, sy + sa * glen), 4)
        col = tuple(min(255, c + 70) for c in f.color) if f.hit_flash > 0 else f.color
        pygame.draw.circle(screen, col, (int(sx), int(sy)), PLAYER_R)
        pygame.draw.circle(screen, (20, 20, 20), (int(sx), int(sy)), PLAYER_R, 2)
        for side in (-1, 1):
            hx, hy = sx + ca * 12 - sa * side * 11, sy + sa * 12 + ca * side * 11
            pygame.draw.circle(screen, (235, 195, 160), (int(hx), int(hy)), 5)
        if f.armor > 0:
            pygame.draw.circle(screen, BLUE, (int(sx), int(sy)), PLAYER_R + 3, 2)
        if f.flash > 0:
            pygame.draw.circle(screen, (255, 230, 120), (int(sx + ca * (glen + 6)), int(sy + sa * (glen + 6))), 8)
        if f.heal_t > 0:
            pygame.draw.circle(screen, GREEN, (int(sx), int(sy)), PLAYER_R + 8, 2)
        if not local:
            draw_bar(screen, sx - 20, sy - 30, 40, 5, f.hp / BASE_HP, GREEN if f.hp > 35 else RED)
            draw_text(screen, self.fonts["small"], f.name, WHITE, (sx, sy - 40), "center")

    def _draw_bullets(self, screen, m, camx, camy):
        for b in m.bullets:
            sx, sy = b.x - camx, b.y - camy
            if -50 < sx < SCREEN_W + 50 and -50 < sy < SCREEN_H + 50:
                n = math.hypot(b.vx, b.vy)
                tail = 26 if b.wdef.kind != "Sniper" else 60
                pygame.draw.line(screen, (255, 235, 150), (sx, sy), (sx - b.vx / n * tail, sy - b.vy / n * tail), 2)
        for s in m.sparks:
            pygame.draw.circle(screen, s[5], (int(s[0] - camx), int(s[1] - camy)), 2)

    def _draw_plane(self, screen, pl: Plane, camx, camy):
        ca, sa = math.cos(pl.heading), math.sin(pl.heading)
        def pts(ox, oy):
            return [(pl.x - camx + ox + px * ca - py * sa, pl.y - camy + oy + px * sa + py * ca) for px, py in PLANE_SHAPE]
        pygame.draw.polygon(screen, (30, 50, 30), pts(40, 55))
        pygame.draw.polygon(screen, (220, 224, 232), pts(0, 0))
        pygame.draw.polygon(screen, (90, 100, 120), pts(0, 0), 3)
        pygame.draw.circle(screen, BLUE, (int(pl.x - camx + 40 * ca), int(pl.y - camy + 40 * sa)), 8)

    def _draw_zone(self, screen, m: Match, camx, camy):
        z = m.zone
        self.overlay.fill((190, 30, 30, 70))
        cx, cy = int(z.c[0] - camx), int(z.c[1] - camy)
        pygame.draw.circle(self.overlay, (0, 0, 0, 0), (cx, cy), int(z.r))
        screen.blit(self.overlay, (0, 0))
        pygame.draw.circle(screen, (90, 170, 255), (cx, cy), int(z.r), 4)
        if z.phase < len(Zone.RADII):
            pygame.draw.circle(screen, WHITE, (int(z.tc[0] - camx), int(z.tc[1] - camy)), int(z.tr), 2)

    # -- HUD -----------------------------------------------------------------
    def _draw_hud(self, screen, m: Match, mouse_pos, t):
        f, fn = m.player, self.fonts
        if f.hit_flash > 0:
            screen.blit(self.hurt, (0, 0))
        # top-left panels
        pygame.draw.rect(screen, (0, 0, 0, 160), (12, 12, 200, 76), border_radius=8)
        draw_text(screen, fn["med"], f"ALIVE  {m.alive_count()}", WHITE, (24, 20))
        draw_text(screen, fn["small"], f"KILLS {f.kills}     TIME {fmt_time(m.time)}", GOLD, (24, 56))
        # zone info
        z = m.zone
        if z.phase >= len(Zone.RADII):
            msg, col = "FINAL ZONE", RED
        elif z.mode == "wait":
            msg, col = f"Zone shrinks in {fmt_time(z.timer)}", WHITE
        else:
            msg, col = f"Zone closing {fmt_time(z.timer)}", RED
        draw_text(screen, fn["small"], msg, col, (SCREEN_W - 118, 240), "midtop")
        if f.state == "ground" and not z.inside(f.x, f.y):
            draw_text(screen, fn["big"], "OUTSIDE THE ZONE!", RED, (SCREEN_W // 2, 96), "center")
        # minimap
        mx, my = SCREEN_W - 222, 12
        screen.blit(self.mini_small, (mx, my))
        s = 210 / MAP_SIZE
        pygame.draw.circle(screen, (90, 170, 255), (int(mx + z.c[0] * s), int(my + z.c[1] * s)), max(1, int(z.r * s)), 2)
        if z.phase < len(Zone.RADII):
            pygame.draw.circle(screen, WHITE, (int(mx + z.tc[0] * s), int(my + z.tc[1] * s)), max(1, int(z.tr * s)), 1)
        if not m.plane.done:
            pl = m.plane
            pygame.draw.line(screen, (255, 255, 255), (mx + pl.x0 * s, my + pl.y0 * s), (mx + pl.x1 * s, my + pl.y1 * s), 1)
            pygame.draw.circle(screen, YELLOW, (int(mx + pl.x * s), int(my + pl.y * s)), 4)
        pygame.draw.circle(screen, f.color, (int(mx + f.x * s), int(my + f.y * s)), 5)
        pygame.draw.circle(screen, WHITE, (int(mx + f.x * s), int(my + f.y * s)), 5, 1)
        # kill feed
        y = 268
        for text, ttl, col in m.feed:
            img = fn["small"].render(text, True, col)
            bg = pygame.Surface((img.get_width() + 12, 22), pygame.SRCALPHA)
            bg.fill((0, 0, 0, 140))
            screen.blit(bg, (SCREEN_W - 12 - bg.get_width(), y))
            screen.blit(img, (SCREEN_W - 6 - img.get_width(), y + 3))
            y += 25
        # state-specific
        if f.state == "plane":
            draw_text(screen, fn["big"], "PRESS  SPACE  TO JUMP", YELLOW if int(t * 2) % 2 == 0 else WHITE,
                      (SCREEN_W // 2, 110), "center")
            draw_bar(screen, SCREEN_W // 2 - 160, 140, 320, 10, m.plane.t, YELLOW)
        elif f.state == "fall":
            draw_text(screen, fn["med"], "Steer with WASD  -  hold SHIFT to dive", WHITE, (SCREEN_W // 2, 110), "center")
            draw_bar(screen, SCREEN_W - 40, 300, 16, 240, f.alt, CYAN)
            draw_text(screen, fn["small"], "ALT", WHITE, (SCREEN_W - 32, 282), "midtop")
        elif f.state == "ground":
            self._draw_ground_hud(screen, m, mouse_pos)
        if f.state == "dead":
            draw_text(screen, fn["title"], "ELIMINATED", RED, (SCREEN_W // 2, 300), "center")
            draw_text(screen, fn["med"], f"You placed #{f.rank}", WHITE, (SCREEN_W // 2, 360), "center")

    def _draw_ground_hud(self, screen, m: Match, mouse_pos):
        f, fn = m.player, self.fonts
        cx = SCREEN_W // 2
        draw_bar(screen, cx - 170, SCREEN_H - 62, 340, 16, f.hp / BASE_HP, GREEN if f.hp > 35 else RED)
        draw_text(screen, fn["small"], f"HP {int(f.hp)}", WHITE, (cx - 164, SCREEN_H - 62))
        draw_bar(screen, cx - 170, SCREEN_H - 42, 340, 10, f.armor / MAX_ARMOR, BLUE)
        # weapon slots
        for i in range(2):
            g = f.guns[i]
            x = cx - 250 + i * 260
            r = pygame.Rect(x, SCREEN_H - 128, 240, 56)
            pygame.draw.rect(screen, (0, 0, 0), r, border_radius=8)
            pygame.draw.rect(screen, GOLD if i == f.cur else (90, 90, 100), r, 3 if i == f.cur else 1, border_radius=8)
            draw_text(screen, fn["small"], str(i + 1), GRAY, (x + 8, SCREEN_H - 124))
            if g:
                draw_text(screen, fn["med"], g.wdef.name, g.wdef.color, (x + 26, SCREEN_H - 124))
                draw_text(screen, fn["med"], f"{g.mag} / {f.reserve[g.wdef.ammo]}", WHITE, (x + 26, SCREEN_H - 100))
                draw_text(screen, fn["small"], g.wdef.ammo, AMMO_COLORS[g.wdef.ammo], (x + 200, SCREEN_H - 100), "topright")
            else:
                draw_text(screen, fn["small"], "empty", GRAY, (x + 26, SCREEN_H - 110))
        draw_text(screen, fn["med"], f"Q  Medkit x{f.medkits}", GREEN if f.medkits else GRAY, (24, SCREEN_H - 40))
        # prompts
        gl = m.nearest_gun_loot(f)
        if gl:
            draw_text(screen, fn["med"], f"E  -  pick up {WEAPONS[gl.key].name}", YELLOW, (cx, SCREEN_H - 152), "center")
        if f.reload_t > 0 and f.gun:
            draw_bar(screen, mouse_pos[0] - 30, mouse_pos[1] + 24, 60, 6, 1 - f.reload_t / f.gun.wdef.reload, YELLOW)
            draw_text(screen, fn["small"], "RELOADING", WHITE, (mouse_pos[0], mouse_pos[1] + 34), "midtop")
        if f.heal_t > 0:
            draw_bar(screen, cx - 60, SCREEN_H // 2 + 50, 120, 8, 1 - f.heal_t / MEDKIT_TIME, GREEN)
        # crosshair
        mx, my = mouse_pos
        spread = 8 + (10 if f.moving else 0) + (f.gun.wdef.spread * 1.6 if f.gun else 0)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            pygame.draw.line(screen, WHITE, (mx + dx * spread, my + dy * spread), (mx + dx * (spread + 9), my + dy * (spread + 9)), 2)
        pygame.draw.circle(screen, WHITE, (mx, my), 2)

    def _draw_big_map(self, screen, m: Match):
        s = 640 / MAP_SIZE
        x0, y0 = (SCREEN_W - 640) // 2, (SCREEN_H - 640) // 2
        pygame.draw.rect(screen, (0, 0, 0), (x0 - 8, y0 - 8, 656, 656), border_radius=8)
        screen.blit(self.mini_big, (x0, y0))
        z = m.zone
        pygame.draw.circle(screen, (90, 170, 255), (int(x0 + z.c[0] * s), int(y0 + z.c[1] * s)), max(1, int(z.r * s)), 3)
        if z.phase < len(Zone.RADII):
            pygame.draw.circle(screen, WHITE, (int(x0 + z.tc[0] * s), int(y0 + z.tc[1] * s)), max(1, int(z.tr * s)), 2)
        if not m.plane.done:
            pl = m.plane
            pygame.draw.line(screen, YELLOW, (x0 + pl.x0 * s, y0 + pl.y0 * s), (x0 + pl.x1 * s, y0 + pl.y1 * s), 2)
            pygame.draw.circle(screen, YELLOW, (int(x0 + pl.x * s), int(y0 + pl.y * s)), 6)
        p = m.player
        pygame.draw.circle(screen, p.color, (int(x0 + p.x * s), int(y0 + p.y * s)), 7)
        pygame.draw.circle(screen, WHITE, (int(x0 + p.x * s), int(y0 + p.y * s)), 7, 2)
        draw_text(screen, self.fonts["small"], "TAB / M to close    white circle = next safe zone", WHITE,
                  (SCREEN_W // 2, y0 + 650), "midtop")


# ════════════════════════════════════════════════════════════════════════════
#  UI WIDGETS
# ════════════════════════════════════════════════════════════════════════════
class Button:
    def __init__(self, x, y, w, h, text, accent=WHITE, font="med"):
        self.rect = pygame.Rect(x, y, w, h)
        self.text, self.accent, self.font = text, accent, font
        self.hover = False

    def update(self, pos):
        self.hover = self.rect.collidepoint(pos)

    def draw(self, screen, fonts, enabled=True):
        col = (70, 78, 104) if (self.hover and enabled) else (40, 46, 66)
        pygame.draw.rect(screen, col, self.rect, border_radius=10)
        pygame.draw.rect(screen, self.accent if enabled else (80, 80, 90), self.rect, 2, border_radius=10)
        img = fonts[self.font].render(self.text, True, WHITE if enabled else GRAY)
        screen.blit(img, img.get_rect(center=self.rect.center))

    def clicked(self, e):
        return e.type == pygame.MOUSEBUTTONDOWN and e.button == 1 and self.rect.collidepoint(e.pos)


def draw_stat_bars(screen, fonts, w: WeaponDef, x, y, width=300):
    rows = [("Damage", clamp(w.damage * w.pellets / 100.0, 0, 1)),
            ("Fire rate", clamp(w.rate / 15.0, 0, 1)),
            ("Range", clamp(w.range / 1700.0, 0, 1)),
            ("Accuracy", clamp(1 - w.spread / 16.0, 0, 1)),
            ("Magazine", clamp(w.mag / 75.0, 0, 1))]
    for i, (label, frac) in enumerate(rows):
        draw_text(screen, fonts["small"], label, GRAY, (x, y + i * 26))
        draw_bar(screen, x + 90, y + i * 26 + 2, width - 90, 12, frac, w.color)


# ════════════════════════════════════════════════════════════════════════════
#  LOBBY
# ════════════════════════════════════════════════════════════════════════════
class Lobby:
    def __init__(self, profile: Profile):
        self.p = profile
        self.name_focus = False
        self.joined = 1
        self.join_t = 0.0
        self.starting = False
        self.count = 3.0
        self.roster = list(BOT_NAMES)
        random.shuffle(self.roster)
        self.hover_weapon: Optional[str] = None
        self.hover_widget = None
        self.t = 0.0
        self.pr_rects = {k: pygame.Rect(350 + (i % 3) * 180, 130 + (i // 3) * 104, 172, 96) for i, k in enumerate(PRIMARY_KEYS)}
        self.se_rects = {k: pygame.Rect(350 + i * 136, 380, 128, 96) for i, k in enumerate(SECONDARY_KEYS)}
        self.color_rects = [pygame.Rect(48 + (i % 4) * 68, 350 + (i // 4) * 62, 54, 50) for i in range(len(PLAYER_COLORS))]
        self.name_rect = pygame.Rect(48, 268, 260, 44)
        self.pl_minus, self.pl_plus = Button(48, 500, 50, 44, "-"), Button(258, 500, 50, 44, "+")
        self.df_minus, self.df_plus = Button(48, 580, 50, 44, "<"), Button(258, 580, 50, 44, ">")
        self.start_btn = Button(940, 620, 300, 64, "START MATCH", GREEN, "big")
        self.buttons = [self.pl_minus, self.pl_plus, self.df_minus, self.df_plus, self.start_btn]

    def roster_names(self):
        return self.roster[: max(0, self.joined - 1)]

    def update(self, dt):
        self.t += dt
        self.join_t += dt
        rate = 0.05 if self.starting else 0.45
        if self.joined < self.p.players and self.join_t >= rate:
            self.join_t = 0.0
            self.joined += 1
        if self.joined > self.p.players:
            self.joined = self.p.players
        if self.starting:
            self.count -= dt

    @property
    def ready_to_launch(self):
        return self.starting and self.count <= 0

    def handle(self, e) -> None:
        if e.type == pygame.MOUSEMOTION:
            for b in self.buttons:
                b.update(e.pos)
            self.hover_weapon = None
            for k, r in {**self.pr_rects, **self.se_rects}.items():
                if r.collidepoint(e.pos):
                    self.hover_weapon = k
        elif e.type == pygame.KEYDOWN and self.name_focus:
            if e.key in (pygame.K_RETURN, pygame.K_ESCAPE):
                self.name_focus = False
            elif e.key == pygame.K_BACKSPACE:
                self.p.name = self.p.name[:-1]
            elif e.unicode and e.unicode.isprintable() and len(self.p.name) < 12:
                self.p.name += e.unicode
        elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1 and not self.starting:
            self.name_focus = self.name_rect.collidepoint(e.pos)
            for i, r in enumerate(self.color_rects):
                if r.collidepoint(e.pos):
                    self.p.color_idx = i
                    play("click")
            for k, r in self.pr_rects.items():
                if r.collidepoint(e.pos):
                    self.p.primary = k
                    play("click")
            for k, r in self.se_rects.items():
                if r.collidepoint(e.pos):
                    self.p.secondary = k
                    play("click")
            if self.pl_minus.clicked(e): self.p.players = max(4, self.p.players - 2)
            elif self.pl_plus.clicked(e): self.p.players = min(30, self.p.players + 2)
            elif self.df_minus.clicked(e):
                self.p.difficulty = DIFF_ORDER[(DIFF_ORDER.index(self.p.difficulty) - 1) % 3]
            elif self.df_plus.clicked(e):
                self.p.difficulty = DIFF_ORDER[(DIFF_ORDER.index(self.p.difficulty) + 1) % 3]
            elif self.start_btn.clicked(e):
                self.starting, self.count = True, 3.0
                play("click")
            if not self.p.name.strip():
                self.p.name = "Player"

    def draw(self, screen, fonts):
        for y in range(0, SCREEN_H, 4):
            k = y / SCREEN_H
            pygame.draw.rect(screen, (int(16 + 22 * k), int(20 + 26 * k), int(38 + 46 * k)), (0, y, SCREEN_W, 4))
        draw_text(screen, fonts["title"], "SKY ROYALE", GOLD, (30, 12))
        draw_text(screen, fonts["small"], "Solo  |  Verdant Island  |  Drop in, loot up, be the last one standing",
                  GRAY, (34, 74))
        p = self.p
        # profile panel
        pygame.draw.rect(screen, PANEL, (30, 100, 300, 600), border_radius=12)
        draw_text(screen, fonts["med"], "PROFILE", WHITE, (48, 112))
        cx, cy = 180, 200
        col = PLAYER_COLORS[p.color_idx]
        ang = math.sin(self.t * 1.5) * 0.4
        wd = WEAPONS[p.primary]
        pygame.draw.line(screen, (30, 30, 30), (cx, cy), (cx + math.cos(ang) * 60, cy + math.sin(ang) * 60), 9)
        pygame.draw.line(screen, wd.color, (cx, cy), (cx + math.cos(ang) * 60, cy + math.sin(ang) * 60), 5)
        pygame.draw.circle(screen, col, (cx, cy), 34)
        pygame.draw.circle(screen, (20, 20, 20), (cx, cy), 34, 3)
        pygame.draw.rect(screen, (24, 28, 40), self.name_rect, border_radius=8)
        pygame.draw.rect(screen, GOLD if self.name_focus else GRAY, self.name_rect, 2, border_radius=8)
        caret = "|" if self.name_focus and int(self.t * 2) % 2 == 0 else ""
        draw_text(screen, fonts["med"], p.name + caret, WHITE, (58, 276), shadow=False)
        draw_text(screen, fonts["small"], "click to edit name", GRAY, (48, 318))
        for i, r in enumerate(self.color_rects):
            pygame.draw.rect(screen, PLAYER_COLORS[i], r, border_radius=8)
            pygame.draw.rect(screen, WHITE if i == p.color_idx else (0, 0, 0), r, 3, border_radius=8)
        draw_text(screen, fonts["small"], "PLAYERS IN MATCH", GRAY, (48, 476))
        self.pl_minus.draw(screen, fonts); self.pl_plus.draw(screen, fonts)
        draw_text(screen, fonts["big"], str(p.players), GOLD, (183, 522), "center")
        draw_text(screen, fonts["small"], "BOT SKILL", GRAY, (48, 556))
        self.df_minus.draw(screen, fonts); self.df_plus.draw(screen, fonts)
        draw_text(screen, fonts["med"], p.difficulty, GOLD, (183, 602), "center")
        # loadout
        pygame.draw.rect(screen, PANEL, (345, 100, 560, 600), border_radius=12)
        draw_text(screen, fonts["med"], "LOADOUT  -  PRIMARY", WHITE, (358, 108))
        for k, r in self.pr_rects.items():
            self._weapon_card(screen, fonts, k, r, k == p.primary)
        draw_text(screen, fonts["med"], "SECONDARY", WHITE, (358, 352))
        for k, r in self.se_rects.items():
            self._weapon_card(screen, fonts, k, r, k == p.secondary)
        shown = WEAPONS[self.hover_weapon or p.primary]
        pygame.draw.rect(screen, (24, 28, 40), (358, 494, 534, 196), border_radius=10)
        draw_text(screen, fonts["big"], shown.name, shown.color, (374, 502))
        draw_text(screen, fonts["small"], f"{shown.kind}   |   {shown.ammo} ammo   |   reload {shown.reload:.1f}s",
                  GRAY, (374, 546))
        draw_text(screen, fonts["small"], shown.blurb, WHITE, (374, 568))
        draw_stat_bars(screen, fonts, shown, 374, 596, 500)
        # roster
        pygame.draw.rect(screen, PANEL, (920, 100, 330, 500), border_radius=12)
        draw_text(screen, fonts["med"], f"LOBBY  {self.joined}/{p.players}", WHITE, (938, 108))
        names = ["YOU  " + p.name] + self.roster_names()
        for i in range(p.players):
            col_i, row_i = i // 15, i % 15
            x, y = 936 + col_i * 158, 146 + row_i * 29
            if i < len(names):
                c = PLAYER_COLORS[p.color_idx] if i == 0 else PLAYER_COLORS[(i + 2) % len(PLAYER_COLORS)]
                pygame.draw.circle(screen, c, (x + 6, y + 9), 6)
                draw_text(screen, fonts["small"], names[i][:14], GOLD if i == 0 else WHITE, (x + 18, y))
            else:
                draw_text(screen, fonts["small"], "searching...", (90, 96, 110), (x + 18, y), shadow=False)
        if self.starting:
            self.start_btn.text = f"STARTING  {max(0, int(self.count) + 1)}"
        else:
            self.start_btn.text = "START MATCH"
        self.start_btn.draw(screen, fonts, not self.starting)
        draw_text(screen, fonts["small"], "Match begins on the plane - press SPACE to jump", GRAY, (1085, 696), "midtop")

    def _weapon_card(self, screen, fonts, key, rect, selected):
        w = WEAPONS[key]
        pygame.draw.rect(screen, (46, 54, 78) if selected else (34, 40, 58), rect, border_radius=10)
        pygame.draw.rect(screen, GOLD if selected else (80, 88, 110), rect, 3 if selected else 1, border_radius=10)
        pygame.draw.line(screen, (20, 20, 20), (rect.x + 24, rect.y + 34), (rect.right - 24, rect.y + 34), 12)
        pygame.draw.line(screen, w.color, (rect.x + 24, rect.y + 34), (rect.right - 24, rect.y + 34), 8)
        draw_text(screen, fonts["med"], w.name, WHITE, (rect.centerx, rect.y + 58), "midtop")
        draw_text(screen, fonts["small"], w.kind, GRAY, (rect.centerx, rect.y + 78), "midtop", False)


# ════════════════════════════════════════════════════════════════════════════
#  SAVE
# ════════════════════════════════════════════════════════════════════════════
SAVE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sky_royale_profile.json")


def load_profile() -> Profile:
    p = Profile()
    try:
        with open(SAVE_FILE) as f:
            d = json.load(f)
        p.name = str(d.get("name", p.name))[:12] or "Player"
        p.color_idx = int(clamp(d.get("color_idx", 0), 0, len(PLAYER_COLORS) - 1))
        p.primary = d.get("primary", p.primary) if d.get("primary") in PRIMARY_KEYS else p.primary
        p.secondary = d.get("secondary", p.secondary) if d.get("secondary") in SECONDARY_KEYS else p.secondary
        p.players = int(clamp(d.get("players", 20), 4, 30))
        p.difficulty = d.get("difficulty", "Normal") if d.get("difficulty") in DIFFICULTY else "Normal"
    except Exception:
        pass
    return p


def save_profile(p: Profile):
    try:
        with open(SAVE_FILE, "w") as f:
            json.dump(p.__dict__, f, indent=2)
    except Exception:
        pass


# ════════════════════════════════════════════════════════════════════════════
#  GAME APP
# ════════════════════════════════════════════════════════════════════════════
class Game:
    LOBBY, MATCH, RESULTS = "lobby", "match", "results"

    def __init__(self):
        try:
            self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H), pygame.SCALED | pygame.RESIZABLE)
        except pygame.error:
            self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("Sky Royale")
        self.clock = pygame.time.Clock()
        self.fonts = {"title": pygame.font.Font(None, 64), "big": pygame.font.Font(None, 44),
                      "med": pygame.font.Font(None, 30), "small": pygame.font.Font(None, 21)}
        self.profile = load_profile()
        self.lobby = Lobby(self.profile)
        self.renderer = Renderer(self.fonts)
        self.state = Game.LOBBY
        self.match: Optional[Match] = None
        self.running = True
        self.paused = False
        self.show_map = False
        self.pending = PlayerInput()
        self.end_t = 0.0
        self.t = 0.0
        self.res_buttons = [Button(SCREEN_W // 2 - 240, 610, 220, 56, "PLAY AGAIN", GREEN),
                            Button(SCREEN_W // 2 + 20, 610, 220, 56, "LOBBY", CYAN)]
        self.pause_buttons = [Button(SCREEN_W // 2 - 160, 300, 320, 56, "RESUME", GREEN),
                              Button(SCREEN_W // 2 - 160, 380, 320, 56, "QUIT TO LOBBY", RED)]

    # -- flow ----------------------------------------------------------------
    def start_match(self):
        save_profile(self.profile)
        self.match = Match(self.profile)
        self.state = Game.MATCH
        self.paused = False
        self.show_map = False
        self.end_t = 0.0
        self.pending = PlayerInput()
        pygame.mouse.set_visible(False)

    def to_lobby(self):
        self.match = None
        self.lobby = Lobby(self.profile)
        self.state = Game.LOBBY
        pygame.mouse.set_visible(True)

    def run(self):
        while self.running:
            dt = min(self.clock.tick(FPS) / 1000.0, 0.05)
            self.t += dt
            for e in pygame.event.get():
                self._event(e)
            self._update(dt)
            self._draw()
            pygame.display.flip()
        pygame.quit()
        sys.exit()

    # -- input ---------------------------------------------------------------
    def _event(self, e):
        if e.type == pygame.QUIT:
            self.running = False
        elif e.type == pygame.KEYDOWN:
            if e.key == pygame.K_F11:
                pygame.display.toggle_fullscreen()
                return
            if self.state == Game.LOBBY:
                if not self.lobby.name_focus and e.key == pygame.K_ESCAPE:
                    self.running = False
                elif not self.lobby.name_focus and e.key == pygame.K_RETURN and not self.lobby.starting:
                    self.lobby.starting, self.lobby.count = True, 3.0
                else:
                    self.lobby.handle(e)
            elif self.state == Game.MATCH:
                self._match_key(e)
            elif self.state == Game.RESULTS and e.key in (pygame.K_RETURN, pygame.K_ESCAPE):
                self.start_match() if e.key == pygame.K_RETURN else self.to_lobby()
        elif self.state == Game.LOBBY:
            self.lobby.handle(e)
        elif self.state == Game.MATCH:
            if e.type == pygame.MOUSEWHEEL and not self.paused:
                self.pending.switch = -1
            elif self.paused and e.type == pygame.MOUSEBUTTONDOWN:
                if self.pause_buttons[0].clicked(e): self.paused = False
                elif self.pause_buttons[1].clicked(e): self.to_lobby()
            elif e.type == pygame.MOUSEMOTION and self.paused:
                for b in self.pause_buttons:
                    b.update(e.pos)
        elif self.state == Game.RESULTS:
            if e.type == pygame.MOUSEMOTION:
                for b in self.res_buttons:
                    b.update(e.pos)
            elif self.res_buttons[0].clicked(e): self.start_match()
            elif self.res_buttons[1].clicked(e): self.to_lobby()

    def _match_key(self, e):
        k, pi = e.key, self.pending
        if k == pygame.K_ESCAPE:
            self.paused = not self.paused
        elif k == pygame.K_n:
            AUDIO["muted"] = not AUDIO["muted"]
        elif k in (pygame.K_TAB, pygame.K_m):
            self.show_map = not self.show_map
        elif k == pygame.K_SPACE: pi.jump = True
        elif k == pygame.K_r: pi.reload = True
        elif k == pygame.K_q: pi.medkit = True
        elif k == pygame.K_e: pi.pickup = True
        elif k == pygame.K_1: pi.switch = 0
        elif k == pygame.K_2: pi.switch = 1

    def _read_input(self) -> PlayerInput:
        keys = pygame.key.get_pressed()
        mx = (1.0 if keys[pygame.K_d] or keys[pygame.K_RIGHT] else 0.0) - (1.0 if keys[pygame.K_a] or keys[pygame.K_LEFT] else 0.0)
        my = (1.0 if keys[pygame.K_s] or keys[pygame.K_DOWN] else 0.0) - (1.0 if keys[pygame.K_w] or keys[pygame.K_UP] else 0.0)
        mp = pygame.mouse.get_pos()
        inp = self.pending
        inp.move = (mx, my)
        inp.aim = math.atan2(mp[1] - SCREEN_H / 2, mp[0] - SCREEN_W / 2)
        inp.fire = bool(pygame.mouse.get_pressed()[0]) and not self.show_map
        inp.sprint = bool(keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT])
        self.pending = PlayerInput()
        return inp

    # -- update / draw ---------------------------------------------------------
    def _update(self, dt):
        if self.state == Game.LOBBY:
            self.lobby.update(dt)
            if self.lobby.ready_to_launch:
                self.start_match()
        elif self.state == Game.MATCH and self.match and not self.paused:
            m = self.match
            m.update(dt, self._read_input())
            if m.player.state == "dead" or m.state == "over":
                self.end_t += dt
                if self.end_t > (3.0 if m.player.state == "dead" else 2.5):
                    self.state = Game.RESULTS
                    pygame.mouse.set_visible(True)

    def _draw(self):
        s = self.screen
        if self.state == Game.LOBBY:
            self.lobby.draw(s, self.fonts)
        elif self.state == Game.MATCH and self.match:
            self.renderer.draw(s, self.match, pygame.mouse.get_pos(), self.show_map, self.t)
            if self.paused:
                ov = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
                ov.fill((0, 0, 0, 170))
                s.blit(ov, (0, 0))
                draw_text(s, self.fonts["title"], "PAUSED", GOLD, (SCREEN_W // 2, 200), "center")
                for b in self.pause_buttons:
                    b.draw(s, self.fonts)
        elif self.state == Game.RESULTS and self.match:
            self._draw_results()

    def _draw_results(self):
        s, fn, m = self.screen, self.fonts, self.match
        s.fill((14, 18, 30))
        p = m.player
        rank = p.rank or (1 if p.state != "dead" else m.total)
        title, col = ("LAST ONE STANDING!", GOLD) if rank == 1 else (f"You placed #{rank}", WHITE)
        draw_text(s, fn["title"], title, col, (SCREEN_W // 2, 70), "center")
        draw_text(s, fn["med"], f"{p.kills} kills    |    {int(p.damage_dealt)} damage    |    survived {fmt_time(p.survive_time)}"
                  f"    |    {m.total} players", GRAY, (SCREEN_W // 2, 120), "center")
        board = sorted(m.fighters, key=lambda f: (f.rank if f.rank else 99, -f.kills))[:10]
        draw_text(s, fn["small"], "#      PLAYER            KILLS     DAMAGE", GRAY, (400, 165))
        for i, f in enumerate(board):
            y = 192 + i * 38
            pygame.draw.rect(s, (36, 44, 66) if not f.is_player else (70, 62, 30), (390, y - 4, 500, 34), border_radius=6)
            pygame.draw.circle(s, f.color, (412, y + 13), 8)
            draw_text(s, fn["med"], f"{f.rank or '-'}", WHITE, (430, y))
            draw_text(s, fn["med"], f.name, GOLD if f.is_player else WHITE, (480, y))
            draw_text(s, fn["med"], str(f.kills), WHITE, (700, y))
            draw_text(s, fn["med"], str(int(f.damage_dealt)), WHITE, (790, y))
        for b in self.res_buttons:
            b.draw(s, fn)
        draw_text(s, fn["small"], "ENTER = play again   ESC = lobby", GRAY, (SCREEN_W // 2, 686), "midtop")


# ════════════════════════════════════════════════════════════════════════════
#  ENTRY
# ════════════════════════════════════════════════════════════════════════════
def main():
    try:
        Game().run()
    except SystemExit:
        raise
    except Exception as e:
        pygame.quit()
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()