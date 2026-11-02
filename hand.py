"""A shaded right hand holding a colored pencil, drawn with signed distance fields.

Each part (fingers, palm, forearm, sleeve) is a soft 3D-looking blob: its distance field gives
a rounded height map, which is lit from the top-left. Layers are stacked so the pencil sits
between the back of the hand and the thumb/index finger.

HandRig.frame(pencil_color, angle) returns an RGBA sprite and the pencil-tip position in it.
"""
import math

import numpy as np
from PIL import Image, ImageFilter

from make_video import pencil_sprite, rotate_about_tip

SIZE = 1700
TIP = np.array([150.0, 150.0])
ANGLE = math.radians(40)
D = np.array([math.cos(ANGLE), math.sin(ANGLE)])      # along the pencil, tip -> back
N = np.array([-math.sin(ANGLE), math.cos(ANGLE)])     # across the pencil (thumb side)
LIGHT = np.array([-0.45, -0.55, 0.70]); LIGHT /= np.linalg.norm(LIGHT)

SKIN = np.array([234, 192, 166], np.float32)
SKIN_DEEP = np.array([190, 120, 104], np.float32)
SLEEVE = np.array([178, 160, 210], np.float32)

_yy, _xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)


def P(a, b):
    p = TIP + a * D + b * N
    return float(p[0]), float(p[1])


def capsule(a, b, ra, rb):
    ax, ay = a; bx, by = b
    vx, vy = bx - ax, by - ay
    h = np.clip(((_xx - ax) * vx + (_yy - ay) * vy) / (vx * vx + vy * vy), 0, 1)
    return np.hypot(_xx - ax - h * vx, _yy - ay - h * vy) - (ra + h * (rb - ra))


def chain(points):
    """Tapered capsules through [(a, b, r), ...] in pencil coordinates."""
    d = None
    for (a0, b0, r0), (a1, b1, r1) in zip(points, points[1:]):
        s = capsule(P(a0, b0), P(a1, b1), r0, r1)
        d = s if d is None else smin(d, s, 4)
    return d


def ellipse(a, b, ra, rb):
    cx, cy = P(a, b)
    u = (_xx - cx) * D[0] + (_yy - cy) * D[1]
    v = (_xx - cx) * N[0] + (_yy - cy) * N[1]
    k = np.hypot(u / ra, v / rb)
    return (k - 1) * min(ra, rb)


def smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b * (1 - h) + a * h - k * h * (1 - h)


def shade(d, radius, base, flat=1.0, texture=None):
    """Light a blob described by distance field d. Returns float RGBA (h, w, 4)."""
    inside = np.clip(-d / radius, 0, 1)
    h = np.sqrt(np.clip(1 - (1 - inside) ** 2, 0, 1))
    height = h * radius * flat
    gy, gx = np.gradient(height)
    nz = np.ones_like(gx)
    norm = np.sqrt(gx * gx + gy * gy + nz)
    nx, ny, nz = -gx / norm, -gy / norm, nz / norm
    diff = np.clip(nx * LIGHT[0] + ny * LIGHT[1] + nz * LIGHT[2], 0, 1)
    half = np.array([LIGHT[0], LIGHT[1], LIGHT[2] + 1]); half /= np.linalg.norm(half)
    spec = np.clip(nx * half[0] + ny * half[1] + nz * half[2], 0, 1) ** 28
    ao = 0.72 + 0.28 * h
    col = base[None, None, :] if base.ndim == 1 else base
    rgb = col * (0.50 + 0.58 * diff)[..., None] * ao[..., None]
    rgb = rgb + (SKIN_DEEP - SKIN)[None, None, :] * 0.25 * (1 - h)[..., None]  # warm edges
    rgb = rgb + 255 * 0.10 * spec[..., None]
    if texture is not None:
        rgb = rgb * texture[..., None]
    alpha = np.clip(0.5 - d, 0, 1)
    return np.dstack([np.clip(rgb, 0, 255), alpha])


def over(dst, src):
    a = src[..., 3:4]
    dst[..., :3] = src[..., :3] * a + dst[..., :3] * (1 - a)
    dst[..., 3:4] = a + dst[..., 3:4] * (1 - a)
    return dst


def drop(dst, mask, dx, dy, blur, strength):
    """Darken what is already in dst with a blurred, offset copy of mask (contact shadow)."""
    m = Image.fromarray((np.clip(mask, 0, 1) * 255).astype(np.uint8))
    m = m.transform(m.size, Image.AFFINE, (1, 0, -dx, 0, 1, -dy)).filter(ImageFilter.GaussianBlur(blur))
    s = np.asarray(m, np.float32)[..., None] / 255 * strength
    dst[..., :3] *= 1 - s
    return dst


def nail(a, b, length, width, direction):
    cx, cy = P(a, b)
    ux, uy = direction
    u = (_xx - cx) * ux + (_yy - cy) * uy
    v = -(_xx - cx) * uy + (_yy - cy) * ux
    return (np.hypot(u / length, v / width) - 1) * width


def crease(mask, a, b, direction, half_width, strength=0.22):
    cx, cy = P(a, b)
    ux, uy = direction
    u = (_xx - cx) * ux + (_yy - cy) * uy
    v = -(_xx - cx) * uy + (_yy - cy) * ux
    line = np.exp(-(u / 1.6) ** 2) * (np.abs(v) < half_width)
    return 1 - strength * line * mask


def unit(a0, b0, a1, b1):
    p0, p1 = np.array(P(a0, b0)), np.array(P(a1, b1))
    v = p1 - p0
    return v / np.linalg.norm(v)


class HandRig:
    def __init__(self, seed=4):
        rng = np.random.default_rng(seed)
        grain = 1 + 0.035 * (np.asarray(Image.fromarray((rng.random((SIZE, SIZE)) * 255).astype(np.uint8))
                                        .filter(ImageFilter.GaussianBlur(0.8)), np.float32) / 255 - 0.5)
        blotch = Image.fromarray((rng.random((24, 24)) * 255).astype(np.uint8)).resize((SIZE, SIZE), Image.BICUBIC)
        blotch = np.asarray(blotch, np.float32) / 255
        skin = SKIN[None, None, :] * (0.96 + 0.08 * blotch[..., None])
        skin[..., 0] += 6 * blotch  # a little rosier in patches

        # ---- back layer: forearm, sleeve, curled fingers, back of hand
        palm = smin(ellipse(350, -46, 120, 96), capsule(P(270, 48), P(380, 10), 24, 44), 30)
        arm = smin(palm, capsule(P(440, -40), P(2100, -30), 72, 80), 34)
        pinky = chain([(280, -104, 15), (336, -100, 18)])
        ring = chain([(232, -70, 17), (300, -80, 21)])
        middle = chain([(112, -27, 15.5), (175, -37, 17.5), (250, -46, 21)])
        sleeve = capsule(P(600, -36), P(2100, -30), 88, 92)

        back = np.zeros((SIZE, SIZE, 4), np.float32)
        tendons = np.ones((SIZE, SIZE), np.float32)
        for b0, b1 in ((-20, -34), (-55, -50), (-92, -66)):
            cx0, cy0 = P(258, b0); cx1, cy1 = P(430, b1)
            vx, vy = cx1 - cx0, cy1 - cy0
            hh = np.clip(((_xx - cx0) * vx + (_yy - cy0) * vy) / (vx * vx + vy * vy), 0, 1)
            dist = np.hypot(_xx - cx0 - hh * vx, _yy - cy0 - hh * vy)
            tendons += 0.045 * np.exp(-(dist / 5) ** 2) * np.sin(hh * math.pi)
        back = over(back, shade(arm, 70, skin, 0.75, grain * tendons))
        back = over(back, shade(pinky, 16, skin, 1.0, grain))
        back = drop(back, np.clip(0.5 - ring, 0, 1), 4, 6, 4, 0.35)
        back = over(back, shade(ring, 18, skin, 1.0, grain))
        back = drop(back, np.clip(0.5 - middle, 0, 1), 4, 6, 4, 0.35)
        back = over(back, shade(middle, 19, skin, 1.0, grain * crease(np.ones_like(grain), 175, -37, unit(112, -27, 250, -46), 9)))
        back = over(back, shade(nail(122, -29, 11, 8.5, unit(112, -27, 175, -37)), 7, np.array([240, 206, 196], np.float32), 0.6))
        # sleeve with ribbed cuff and a soft knit
        a_coord = (_xx - TIP[0]) * D[0] + (_yy - TIP[1]) * D[1]
        b_coord = (_xx - TIP[0]) * N[0] + (_yy - TIP[1]) * N[1]
        rib = np.where(a_coord < 655, 1 - 0.10 * (0.5 + 0.5 * np.sin(b_coord * 0.55)),
                       1 - 0.05 * (0.5 + 0.5 * np.sin(b_coord * 0.35 + np.sin(a_coord * 0.3) * 1.5)))
        back = drop(back, np.clip(0.5 - sleeve, 0, 1), -10, 0, 10, 0.45)
        back = over(back, shade(sleeve, 95, SLEEVE, 0.8, rib * grain))
        self.back = back

        # ---- front layer: thumb with its base, index finger on top of the pencil
        thumb = chain([(124, 24, 17.5), (190, 42, 19.5), (250, 50, 22)])
        index = chain([(96, -3, 16.5), (160, -6, 18.5), (232, -14, 20.5), (270, -22, 23)])
        front = np.zeros((SIZE, SIZE, 4), np.float32)
        tdir = unit(124, 24, 190, 42)
        front = over(front, shade(thumb, 22, skin, 0.95, grain * crease(np.clip(0.5 - thumb, 0, 1), 190, 42, tdir, 12)))
        front = over(front, shade(nail(140, 30, 13, 9.5, tdir), 8, np.array([240, 200, 190], np.float32), 0.6))
        idir1, idir2 = unit(96, -3, 160, -6), unit(160, -6, 232, -14)
        icrease = crease(np.ones_like(grain), 160, -6, idir1, 11) * crease(np.ones_like(grain), 206, -10, idir2, 12, 0.15)
        front = drop(front, np.clip(0.5 - index, 0, 1), 3, 6, 4, 0.30)
        front = over(front, shade(index, 19, skin, 1.0, grain * icrease))
        front = over(front, shade(nail(110, -4, 14, 10.5, idir1), 8, np.array([242, 204, 194], np.float32), 0.6))
        self.front = front
        self.front_mask = np.clip(front[..., 3], 0, 1)
        self.back_mask = np.clip(back[..., 3], 0, 1)
        self._pencils = {}
        self._sprites = {}
        alpha = self.sprite((200, 120, 100)).getchannel("A")
        self.shadows = []
        for dx, dy, blur, op in ((16, 22, 10, 0.30), (46, 60, 28, 0.22)):
            sh = Image.new("RGBA", (SIZE, SIZE), (40, 25, 20, 255))
            sh.putalpha(alpha.point(lambda v, op=op: int(v * op)).filter(ImageFilter.GaussianBlur(blur)))
            self.shadows.append((sh, dx, dy))

    def pencil(self, color):
        key = tuple(int(c) // 12 * 12 for c in color)
        if key not in self._pencils:
            spr = pencil_sprite(key, 470, 17)
            rot, tip = rotate_about_tip(spr, -math.degrees(ANGLE))
            layer = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
            layer.alpha_composite(rot, (int(TIP[0] - tip[0]), int(TIP[1] - tip[1])))
            self._pencils[key] = np.asarray(layer, np.float32) / np.array([1, 1, 1, 255], np.float32)
        return self._pencils[key]

    def sprite(self, pencil_color):
        """RGBA hand + pencil (PIL), tip at TIP."""
        key = tuple(int(c) // 12 * 12 for c in pencil_color)
        if key in self._sprites:
            return self._sprites[key]
        if len(self._sprites) > 12:
            self._sprites.pop(next(iter(self._sprites)))
        img = self.back.copy()
        pen = self.pencil(pencil_color)
        img = drop(img, pen[..., 3], 3, 5, 3, 0.35)
        img = over(img, pen)
        img = drop(img, self.front_mask, 3, 6, 5, 0.32)
        img = over(img, self.front)
        rgba = np.dstack([img[..., :3], img[..., 3] * 255])
        self._sprites[key] = Image.fromarray(np.clip(rgba, 0, 255).astype(np.uint8), "RGBA")
        return self._sprites[key]

    def place(self, canvas, tip_xy, pencil_color, angle=0.0, scale=1.0):
        """Draw the hand (with shadow on the page) so the pencil tip lands on tip_xy."""
        spr = self.sprite(pencil_color)
        tip = (float(TIP[0]), float(TIP[1]))
        for sh, dx, dy in self.shadows:
            rot = sh.rotate(angle, center=tip, resample=Image.BILINEAR)
            _paste(canvas, rot, int(tip_xy[0] - tip[0] + dx), int(tip_xy[1] - tip[1] + dy))
        rot = spr.rotate(angle, center=tip, resample=Image.BICUBIC)
        _paste(canvas, rot, int(tip_xy[0] - tip[0]), int(tip_xy[1] - tip[1]))
        return canvas


def _paste(canvas, img, x, y):
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + img.width, canvas.width), min(y + img.height, canvas.height)
    if x1 <= x0 or y1 <= y0:
        return
    canvas.alpha_composite(img.crop((x0 - x, y0 - y, x1 - x, y1 - y)), (x0, y0))


if __name__ == "__main__":
    import sys
    rig = HandRig()
    page = Image.new("RGBA", (1080, 1100), (246, 242, 234, 255))
    rig.place(page, (330, 300), (226, 120, 100))
    page.convert("RGB").save(sys.argv[1] if len(sys.argv) > 1 else "hand_test.png")
