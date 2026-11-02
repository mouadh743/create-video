#!/usr/bin/env python3
"""Build a short vertical promo video (1080x1920) for the coloring book.

Everything is drawn from the line-art pages in pages/ - no AI video service needed.
Four scenes, one per prompt, joined with crossfades:

  1. cover reveal    - book on a wooden desk, warm light, slow push-in and tilt
  2. flipping pages  - open book, pages turning to show the line art
  3. coloring        - top-down mandala page filled with warm pencil colors
  4. floating book   - book rotating on a pastel background with confetti and pencils

Usage:  python3 make_video.py            (needs ffmpeg, numpy, pillow, scipy)
Output: output/coloring_book_promo.mp4 and output/scenes/scene1..4.mp4
"""
import math
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from scipy import ndimage

ROOT = Path(__file__).resolve().parent
PAGES = [ROOT / "pages" / f"page{i}.jpg" for i in (1, 2, 3)]
OUT = ROOT / "output"
SCENES = OUT / "scenes"

W, H, FPS = 1080, 1920, 30
SCENE_SECONDS = 5.0
END_SECONDS = 7.0          # last scene holds the store name and social handles
FADE_SECONDS = 0.5
N_FRAMES = int(SCENE_SECONDS * FPS)

# On-screen text (Arabic). Shaped right-to-left by Pillow's raqm layout engine.
STORE = "متجر الحياة السعيدة"           # Happy Life Store
BOOK_TITLE = "ماندالا الزهور"            # Floral Mandalas
BOOK_SUB = "كتاب تلوين للكبار"           # Adult coloring book
FOLLOW_IG = "تابعونا على إنستغرام"       # Follow us on Instagram
FOLLOW_TT = "تابعونا على تيك توك"        # Follow us on TikTok
HANDLE = "@_happy.life.store"

FONT_AR_DISPLAY = str(ROOT / "fonts" / "Amiri-Bold.ttf")
FONT_AR_BOLD = str(ROOT / "fonts" / "Cairo-Bold.ttf")

FONT_SANS = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

WARM = np.array([(240, 128, 98), (250, 178, 122), (242, 198, 92), (226, 106, 124),
                 (198, 100, 74), (247, 154, 92), (156, 182, 122), (170, 96, 128)], float)
BRIGHT = np.array([(64, 170, 170), (240, 120, 100), (244, 196, 80), (160, 140, 214),
                   (228, 104, 140), (128, 200, 158), (106, 166, 228), (248, 158, 90)], float)
CONFETTI = [(255, 140, 150), (255, 200, 90), (130, 200, 230), (180, 150, 240),
            (120, 210, 170), (255, 160, 110), (250, 120, 180)]


# ---------------------------------------------------------------- helpers

def smooth(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


class Writer:
    """Pipes RGB frames into ffmpeg."""

    def __init__(self, path):
        self.proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium",
             "-crf", "19", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
            stdin=subprocess.PIPE)

    def write(self, frame):
        arr = frame if isinstance(frame, np.ndarray) else np.asarray(frame.convert("RGB"))
        self.proc.stdin.write(np.clip(arr, 0, 255).astype(np.uint8).tobytes())

    def close(self):
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError("ffmpeg failed")


def noise(w, h, nx, ny, seed):
    """Smooth value noise in [0, 1]: an (ny, nx) random grid scaled up to (h, w)."""
    r = np.random.default_rng(seed)
    small = Image.fromarray((r.random((ny, nx)) * 255).astype(np.uint8))
    return np.asarray(small.resize((w, h), Image.BICUBIC), np.float32) / 255.0


def wood(w, h, seed=1):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    n1 = noise(w, h, 5, 50, seed)
    n2 = noise(w, h, 30, 200, seed + 1)
    broad = noise(w, h, 4, 8, seed + 2)
    grain = 0.5 + 0.5 * np.sin(yy * 0.10 + n1 * 16 + n2 * 2.5)
    grain = grain ** 3
    t = np.clip(grain * 0.55 + broad * 0.45, 0, 1)[..., None]
    light = np.array([170, 116, 72], np.float32)
    dark = np.array([112, 70, 40], np.float32)
    img = light * (1 - t) + dark * t
    img += np.random.default_rng(seed).normal(0, 3.5, (h, w, 1))
    for y in range(380, h, 420):  # plank seams
        img[max(0, y - 3):y + 3] *= 0.62
        img[y + 3:y + 7] *= 1.06
    return np.clip(img, 0, 255).astype(np.uint8)


def warm_light(cx, cy, strength=0.55):
    """Multiplier map: warm, brighter near (cx, cy), darker toward the corners."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.hypot((xx - cx) / W, (yy - cy) / H)
    m = (1.12 - strength * d)[..., None]
    return (m * np.array([1.0, 0.96, 0.88], np.float32)).astype(np.float32)


def apply_light(img, light):
    return Image.fromarray(np.clip(np.asarray(img.convert("RGB"), np.float32) * light, 0, 255)
                           .astype(np.uint8))


def camera(img, zoom, rot_deg=0.0, cx=W / 2, cy=H / 2):
    """Zoom in by `zoom` and rotate around (cx, cy), keeping the frame size."""
    r = math.radians(rot_deg)
    a, b = math.cos(r) / zoom, math.sin(r) / zoom
    d, e = -math.sin(r) / zoom, math.cos(r) / zoom
    c = cx - a * W / 2 - b * H / 2
    f = cy - d * W / 2 - e * H / 2
    return img.transform((W, H), Image.AFFINE, (a, b, c, d, e, f), Image.BICUBIC)


def perspective_coeffs(dst, src):
    rows, rhs = [], []
    for (x, y), (X, Y) in zip(dst, src):
        rows.append([x, y, 1, 0, 0, 0, -X * x, -X * y]); rhs.append(X)
        rows.append([0, 0, 0, x, y, 1, -Y * x, -Y * y]); rhs.append(Y)
    return np.linalg.solve(np.array(rows, float), np.array(rhs, float))


def pad(img, p=3):
    out = Image.new("RGBA", (img.width + 2 * p, img.height + 2 * p), (0, 0, 0, 0))
    out.paste(img.convert("RGBA"), (p, p))
    return out


def warp_onto(canvas, tex, quad, p=3):
    """Draw RGBA texture `tex` (padded by p) so its corners land on quad (TL, TR, BR, BL)."""
    xs = [q[0] for q in quad]; ys = [q[1] for q in quad]
    x0, y0 = int(math.floor(min(xs))) - 2, int(math.floor(min(ys))) - 2
    x1, y1 = int(math.ceil(max(xs))) + 2, int(math.ceil(max(ys))) + 2
    if x1 - x0 < 2 or y1 - y0 < 2:
        return
    w, h = tex.width - 2 * p, tex.height - 2 * p
    src = [(p, p), (p + w, p), (p + w, p + h), (p, p + h)]
    dst = [(x - x0, y - y0) for x, y in quad]
    coeffs = perspective_coeffs(dst, src)
    patch = tex.transform((x1 - x0, y1 - y0), Image.PERSPECTIVE, tuple(coeffs), Image.BICUBIC)
    canvas.alpha_composite(patch, (x0, y0)) if x0 >= 0 and y0 >= 0 else canvas.paste(patch, (x0, y0), patch)


def soft_shadow(size, box, radius, opacity, blur):
    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).rounded_rectangle(box, radius, fill=int(255 * opacity))
    return m.filter(ImageFilter.GaussianBlur(blur))


def darken(img, mask):
    black = Image.new("RGBA", img.size, (20, 10, 5, 255))
    return Image.composite(black, img, mask)


def brand_tag(frame, alpha):
    """Small "Happy Life Store" pill near the top, inside the phone-safe area."""
    if alpha <= 0:
        return frame
    frame = frame.convert("RGBA")
    over = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    font = ImageFont.truetype(FONT_AR_BOLD, 46)
    tw = d.textlength(STORE, font=font)
    x0, y0 = (W - tw) / 2 - 44, 210
    d.rounded_rectangle([x0, y0, x0 + tw + 88, y0 + 88], 44, fill=(255, 250, 242, int(225 * alpha)))
    d.text((W / 2, y0 + 44), STORE, font=font, anchor="mm", fill=(84, 52, 92, int(255 * alpha)))
    frame.alpha_composite(over)
    return frame


def social_icon(kind, size, color):
    s = 4
    n = size * s
    im = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([0, 0, n - 1, n - 1], int(n * 0.28), fill=color)
    white = (255, 255, 255, 255)
    if kind == "instagram":
        m = n * 0.2
        d.rounded_rectangle([m, m, n - m, n - m], int(n * 0.18), outline=white, width=int(n * 0.07))
        r = n * 0.15
        d.ellipse([n / 2 - r, n / 2 - r, n / 2 + r, n / 2 + r], outline=white, width=int(n * 0.07))
        r2 = n * 0.045
        d.ellipse([n * 0.70 - r2, n * 0.30 - r2, n * 0.70 + r2, n * 0.30 + r2], fill=white)
    else:  # simple music-note glyph for TikTok
        w = n * 0.09
        d.rectangle([n * 0.50, n * 0.20, n * 0.50 + w, n * 0.66], fill=white)
        r = n * 0.15
        d.ellipse([n * 0.50 + w - 2 * r, n * 0.66 - r, n * 0.50 + w, n * 0.66 + r], fill=white)
        d.pieslice([n * 0.50, n * 0.02, n * 0.86, n * 0.38], 90, 180, fill=white)
        d.ellipse([n * 0.50 + w, n * 0.12, n * 0.78, n * 0.30], fill=color)
    return im.resize((size, size), Image.LANCZOS)


# ---------------------------------------------------------------- line-art coloring

class Segments:
    """Closed white areas of a line drawing, ready to be colored."""

    def __init__(self, gray, center=(0.5, 0.5)):
        self.gray = gray
        h, w = gray.shape
        lab, n = ndimage.label(gray >= 165)
        sizes = np.bincount(lab.ravel(), minlength=n + 1).astype(float)
        border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
        keep = (sizes > 20) & (sizes < 0.02 * gray.size)
        keep[0] = False
        keep[border] = False
        yy, xx = np.mgrid[0:h, 0:w]
        safe = np.maximum(sizes, 1)
        self.cx = np.bincount(lab.ravel(), xx.ravel(), n + 1) / safe
        self.cy = np.bincount(lab.ravel(), yy.ravel(), n + 1) / safe
        x0 = np.zeros(n + 1); xw = np.ones(n + 1)
        for i, sl in enumerate(ndimage.find_objects(lab), start=1):
            if sl is not None:
                x0[i] = sl[1].start; xw[i] = max(sl[1].stop - sl[1].start, 1)
        self.pos = (xx - x0[lab]) / xw[lab]  # 0..1 across each area, for a pencil-stroke wipe
        self.lab, self.keep, self.sizes, self.n = lab, keep, sizes, n
        self.center = (center[0] * w, center[1] * h)
        self.dist = np.hypot(self.cx - self.center[0], self.cy - self.center[1])
        self.ring_px = w * 0.055
        median = np.median(sizes[keep]) if keep.any() else 1
        self.small = sizes < median * 0.5
        self.grain = 1 - 0.08 * np.random.default_rng(3).random(gray.shape)

    def color_index(self, ring):
        return ring % 8

    def colored(self, palette):
        idx = (self.dist / self.ring_px).astype(int) + self.small * 3
        lut = np.full((self.n + 1, 3), 255.0)
        lut[self.keep] = palette[idx[self.keep] % len(palette)]
        base = lut[self.lab]
        fill = self.keep[self.lab][..., None]
        base = np.where(fill, base * self.grain[..., None], 255.0)
        return base * (self.gray[..., None] / 255.0)

    def pencil_color(self, palette, r):
        return palette[int(r / self.ring_px) % len(palette)]


def colored_page(path, palette, center=(0.5, 0.5)):
    gray = np.asarray(Image.open(path).convert("L"))
    return Image.fromarray(np.clip(Segments(gray, center).colored(palette), 0, 255).astype(np.uint8))


def pencil_sprite(color, length=560, width=36):
    s = 3
    L, Wd = length * s, width * s
    color = tuple(int(c) for c in color)
    lighter = tuple(min(255, int(c * 1.18 + 20)) for c in color)
    darker = tuple(int(c * 0.72) for c in color)
    im = Image.new("RGBA", (L, Wd), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    cone = int(L * 0.13)
    d.rectangle([cone, 0, L, Wd], fill=color)
    d.rectangle([cone, 0, L, Wd // 3], fill=lighter)
    d.rectangle([cone, 2 * Wd // 3, L, Wd], fill=darker)
    d.rectangle([L - 16 * s, 0, L, Wd], fill=tuple(int(c * 0.55) for c in color))
    tip = int(cone * 0.3)
    d.polygon([(cone, 0), (cone, Wd), (tip, Wd / 2 + Wd * 0.16), (tip, Wd / 2 - Wd * 0.16)],
              fill=(234, 198, 150))
    for k in range(3):  # scalloped wood edge
        y = Wd * (k + 0.5) / 3
        d.ellipse([cone - Wd * 0.18, y - Wd / 6, cone + Wd * 0.18, y + Wd / 6], fill=(234, 198, 150))
    d.polygon([(tip, Wd / 2 - Wd * 0.16), (tip, Wd / 2 + Wd * 0.16), (0, Wd / 2)], fill=darker)
    return im.resize((length, width), Image.LANCZOS)


def rotate_about_tip(sprite, angle):
    """Rotate a horizontal sprite (tip at left middle). Returns image and the tip's position in it."""
    rot = sprite.rotate(angle, expand=True, resample=Image.BICUBIC)
    a = math.radians(angle)
    vx, vy = -sprite.width / 2, 0
    tx = vx * math.cos(a) + vy * math.sin(a)
    ty = -vx * math.sin(a) + vy * math.cos(a)
    return rot, (rot.width / 2 + tx, rot.height / 2 + ty)


def paper_page(img, pw, ph, margin, gutter_side=None):
    """Lay a page image on cream paper; optionally darken toward the book's spine."""
    paper = np.full((ph, pw, 3), (250, 247, 240), np.float32)
    im = img.convert("RGB")
    scale = min((pw - 2 * margin) / im.width, (ph - 2 * margin) / im.height)
    im = im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS)
    ox, oy = (pw - im.width) // 2, (ph - im.height) // 2
    paper[oy:oy + im.height, ox:ox + im.width] *= np.asarray(im, np.float32) / 255.0
    if gutter_side:
        x = np.arange(pw, dtype=np.float32)
        dist = (pw - 1 - x) if gutter_side == "right" else x
        outer = x if gutter_side == "right" else (pw - 1 - x)
        shade = (1 - 0.32 * np.exp(-dist / 40)) * (1 - 0.06 * np.exp(-outer / 25))
        paper *= shade[None, :, None]
    return Image.fromarray(np.clip(paper, 0, 255).astype(np.uint8))


# ---------------------------------------------------------------- cover art

def make_cover(cover_art, w=680, h=1020):
    cover = Image.new("RGB", (w, h), (38, 92, 100))
    art = cover_art.resize((w - 40, int((w - 40) * cover_art.height / cover_art.width)), Image.LANCZOS)
    cover.paste(art, (20, h - art.height - 20))
    d = ImageDraw.Draw(cover, "RGBA")
    d.rounded_rectangle([40, 44, w - 40, 300], 26, fill=(252, 246, 234, 238))
    d.rounded_rectangle([52, 56, w - 52, 288], 20, outline=(196, 150, 70, 255), width=3)
    f1 = ImageFont.truetype(FONT_AR_DISPLAY, 92)
    f2 = ImageFont.truetype(FONT_AR_BOLD, 44)
    d.text((w / 2, 140), BOOK_TITLE, font=f1, anchor="mm", fill=(60, 40, 70))
    d.text((w / 2, 240), BOOK_SUB, font=f2, anchor="mm", fill=(176, 92, 70))
    return cover


# ---------------------------------------------------------------- scene 1: cover reveal

def scene_cover(cover, writer):
    desk = wood(W, H, seed=3)
    blur = np.asarray(Image.fromarray(desk).filter(ImageFilter.GaussianBlur(10)), np.float32)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    m = np.clip((np.hypot((xx - 540) / 540, (yy - 980) / 960) - 0.5) / 0.45, 0, 1)[..., None]
    desk = Image.fromarray((desk * (1 - m) + blur * m).astype(np.uint8)).convert("RGBA")

    cw, ch = cover.size
    bx, by = (W - cw) // 2 - 8, 980 - ch // 2
    base = darken(desk, soft_shadow((W, H), [bx + 4, by + 30, bx + cw + 48, by + ch + 52], 24, 0.62, 30))
    d = ImageDraw.Draw(base)
    d.rectangle([bx + 4, by + 6, bx + cw + 12, by + ch + 12], fill=(236, 228, 210))
    for i in range(0, 12, 3):  # page edges
        d.line([bx + cw + i, by + 6 + i, bx + cw + i, by + ch + i], fill=(205, 196, 178), width=1)
        d.line([bx + 6 + i, by + ch + i, bx + cw + i, by + ch + i], fill=(205, 196, 178), width=1)

    tex = np.asarray(cover.convert("RGB"), np.float32)
    tx = np.arange(cw, dtype=np.float32)[None, :, None] / cw
    ty = np.arange(ch, dtype=np.float32)[:, None, None] / ch
    light = warm_light(260, 420, 0.6)
    proj = xx * 0.6 + yy * 0.8

    for f in range(N_FRAMES):
        t = f / FPS
        p = t / SCENE_SECONDS
        # pages / cover corner lifting in a soft breeze
        lift = (0.5 - 0.5 * math.cos(2 * math.pi * t / 1.7)) * (0.75 + 0.25 * math.sin(2 * math.pi * t * 2.3))
        # moving light sheen across the cover
        u = -0.4 + 1.8 * p
        band = 0.20 * np.exp(-(((tx + ty * 0.5) - u) / 0.12) ** 2)
        sheen = Image.fromarray(np.clip(tex + band * 255, 0, 255).astype(np.uint8))
        frame = base.copy()
        dx, dy = 34 * lift, 26 * lift
        if lift > 0.02:  # lifted edge casts a small shadow on the page block
            sh = soft_shadow((W, H), [bx + cw - 30, by, bx + cw + 6, by + ch], 4, 0.35 * lift, 8)
            frame = darken(frame, sh)
        warp_onto(frame, pad(sheen), [(bx, by), (bx + cw - dx, by - dy), (bx + cw - dx, by + ch + dy), (bx, by + ch)])
        # window light falling across the desk, drifting slowly
        window = (1 - 0.09 * np.clip(np.sin((proj + 60 * t) / 70) * 3, 0, 1))[..., None]
        frame = apply_light(frame, light * window)
        frame = camera(frame, 1.10 + 0.22 * smooth(p), -2.5 + 2.5 * smooth(p), 540, 980)
        frame = brand_tag(frame, smooth((t - 0.6) / 0.6))
        if t < 0.5:
            frame = ImageEnhance.Brightness(frame.convert("RGB")).enhance(smooth(t / 0.5))
        writer.write(frame)


# ---------------------------------------------------------------- scene 2: flipping pages

def scene_flip(pages, writer):
    pw, ph = 490, 735
    sx, y0 = 540, 580
    desk = Image.fromarray(wood(W, H, seed=5)).convert("RGBA")
    base = darken(desk, soft_shadow((W, H), [sx - pw - 10, y0 + 10, sx + pw + 40, y0 + ph + 60], 20, 0.6, 28))
    d = ImageDraw.Draw(base)
    d.rounded_rectangle([sx - pw - 18, y0 - 14, sx + pw + 18, y0 + ph + 18], 12, fill=(38, 92, 100))
    for i in (8, 5, 2):  # page stack
        d.rectangle([sx - pw - i, y0 - 4 + i, sx + pw + i, y0 + ph + i], fill=(222, 214, 196), outline=(200, 190, 172))

    left = [paper_page(p, pw, ph, 30, "right") for p in pages]
    right = [paper_page(p, pw, ph, 30, "left") for p in pages]
    light = warm_light(300, 500, 0.55)
    turns = [(0.8, 2.3), (2.9, 4.4)]

    def spread(li, ri):
        fr = base.copy()
        fr.paste(left[li], (sx - pw, y0))
        fr.paste(right[ri], (sx, y0))
        return fr

    for f in range(N_FRAMES):
        t = f / FPS
        frame = None
        for k, (a, b) in enumerate(turns):
            if a <= t < b:
                theta = math.pi * smooth((t - a) / (b - a))
                c, s = math.cos(theta), math.sin(theta)
                frame = spread(2 * k, 2 * k + 3)
                arr = np.asarray(frame, np.float32).copy()
                fx = sx + pw * c
                sw = 90 * s
                if c > 0:  # shadow on the right page beyond the turning page
                    x0, x1 = int(fx), int(min(fx + sw, sx + pw))
                    if x1 > x0:
                        g = 1 - 0.35 * s * (1 - np.linspace(0, 1, x1 - x0))
                        arr[y0:y0 + ph, x0:x1, :3] *= g[None, :, None]
                else:
                    x1, x0 = int(fx), int(max(fx - sw, sx - pw))
                    if x1 > x0:
                        g = 1 - 0.35 * s * np.linspace(0, 1, x1 - x0)
                        arr[y0:y0 + ph, x0:x1, :3] *= g[None, :, None]
                frame = Image.fromarray(arr.astype(np.uint8))
                lift = 0.10 * ph * s
                shade = 0.74 + 0.26 * abs(c)
                if c >= 0:
                    tex = ImageEnhance.Brightness(right[2 * k + 1]).enhance(shade)
                    quad = [(sx, y0), (fx, y0 - lift / 2), (fx, y0 + ph + lift / 2), (sx, y0 + ph)]
                else:
                    tex = ImageEnhance.Brightness(left[2 * k + 2]).enhance(shade)
                    quad = [(fx, y0 - lift / 2), (sx, y0), (sx, y0 + ph), (fx, y0 + ph + lift / 2)]
                if abs(fx - sx) > 2:
                    warp_onto(frame, pad(tex), quad)
                break
        if frame is None:
            k = sum(1 for a, b in turns if t >= b)
            frame = spread(2 * k, 2 * k + 1)
        frame = apply_light(frame, light)
        frame = camera(frame, 1.0 + 0.08 * smooth(t / SCENE_SECONDS), 0, 540, 950)
        writer.write(brand_tag(frame, 1.0))


# ---------------------------------------------------------------- scene 3: coloring

def scene_color(page_path, writer):
    pw, ph = 960, 1440
    img = Image.open(page_path).convert("L")
    scale = min((pw - 70) / img.width, (ph - 70) / img.height)
    img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    gray = np.asarray(img)
    gh, gw = gray.shape
    seg = Segments(gray, center=(0.5, 0.48))
    colored = seg.colored(WARM)
    shade = gray[..., None] / 255.0

    # A spiral path from the center outward; each area is filled when the pencil passes it.
    revs, radius = 5, gw * 0.40
    t0, span, fill_time = 0.35, 4.3, 0.35
    dx, dy = seg.cx - seg.center[0], seg.cy - seg.center[1]
    ang = (np.arctan2(dx, -dy) / (2 * np.pi)) % 1.0
    u = ang + np.round(seg.dist / radius * revs - ang)
    start = t0 + span * u / revs
    start[~seg.keep | (seg.dist > radius)] = np.inf
    start_px = start[seg.lab]

    ox, oy = (W - pw) // 2, (H - ph) // 2
    gx, gy = ox + (pw - gw) // 2, oy + (ph - gh) // 2
    desk = Image.fromarray(wood(W, H, seed=9)).convert("RGBA")
    base = darken(desk, soft_shadow((W, H), [ox + 6, oy + 20, ox + pw + 30, oy + ph + 36], 6, 0.55, 22))
    ImageDraw.Draw(base).rectangle([ox, oy, ox + pw, oy + ph], fill=(251, 249, 244))
    paper = np.array([251, 249, 244], np.float32) / 255.0
    light = warm_light(380, 520, 0.5)

    sprites = {}

    def pencil(color):
        key = tuple(int(c) for c in color)
        if key not in sprites:
            spr = pencil_sprite(key)
            rot, tip = rotate_about_tip(spr, -38)
            shadow_src = Image.new("RGBA", spr.size, (0, 0, 0, 0))
            shadow_src.putalpha(spr.getchannel("A").point(lambda v: v * 0.45))
            srot, stip = rotate_about_tip(shadow_src, -27)
            srot = srot.filter(ImageFilter.GaussianBlur(9))
            sprites[key] = (rot, tip, srot, stip)
        return sprites[key]

    for f in range(N_FRAMES):
        t = f / FPS
        q = (t - start_px) / fill_time
        alpha = np.clip((q * 1.4 - seg.pos) / 0.4, 0, 1)[..., None]
        page = (255.0 * (1 - alpha) + colored / np.maximum(shade, 0.05) * alpha) * shade
        page = np.clip(page * paper, 0, 255).astype(np.uint8)
        frame = base.copy()
        frame.paste(Image.fromarray(page), (gx, gy))

        uu = min(max((t - t0) / span, 0), 1) * revs
        r = uu / revs * radius
        a = 2 * math.pi * (uu % 1.0)
        wig = 5 * math.sin(2 * math.pi * 9 * t)
        tip = (gx + seg.center[0] + r * math.sin(a) + wig, gy + seg.center[1] - r * math.cos(a) + wig * 0.5)
        if t > t0 + span:  # pencil lifts away at the end
            k = smooth((t - t0 - span) / 0.4)
            tip = (tip[0] + 260 * k, tip[1] + 320 * k)
        rot, rtip, srot, stip = pencil(seg.pencil_color(WARM, r))
        frame.alpha_composite(srot, (int(tip[0] - stip[0]), int(tip[1] - stip[1]))) if tip[0] - stip[0] >= 0 else None
        frame.paste(rot, (int(tip[0] - rtip[0]), int(tip[1] - rtip[1])), rot)
        frame = apply_light(frame, light)
        frame = camera(frame, 1.02 + 0.06 * smooth(t / SCENE_SECONDS), 0, gx + seg.center[0], gy + seg.center[1])
        writer.write(brand_tag(frame, 1.0))


# ---------------------------------------------------------------- scene 4: floating book

def scene_float(cover, writer):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    v = (yy / H)[..., None]
    top, bottom = np.array([255, 226, 214], np.float32), np.array([224, 214, 248], np.float32)
    bg = top * (1 - v) + bottom * v
    glow = np.exp(-(((xx - 540) / 520) ** 2 + ((yy - 650) / 600) ** 2))[..., None]
    bg = bg + (255 - bg) * 0.55 * glow
    bg = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8)).convert("RGBA")

    bw, bh, bd = 460, 690, 38
    front = cover.resize((bw, bh), Image.LANCZOS).convert("RGB")
    front_arr = np.asarray(front, np.float32)
    spine = Image.new("RGB", (bd, bh), (34, 84, 92))
    txt = Image.new("RGBA", (bh, bd), (0, 0, 0, 0))
    td = ImageDraw.Draw(txt)
    fnt = ImageFont.truetype(FONT_AR_BOLD, 22)
    label = f"{BOOK_TITLE}  •  {BOOK_SUB}"
    td.text((bh / 2, bd / 2), label, font=fnt, anchor="mm", fill=(240, 206, 120))
    spine.paste(txt.rotate(-90, expand=True), (0, 0), txt.rotate(-90, expand=True))
    block = Image.new("RGB", (bd, bh), (240, 233, 216))
    bdraw = ImageDraw.Draw(block)
    for x in range(2, bd, 4):
        bdraw.line([x, 0, x, bh], fill=(222, 212, 192))
    top_face = block.resize((bd, bw)).rotate(90, expand=True)

    F = 2400.0
    cx, cy0 = 540.0, 650.0
    lightdir = np.array([-0.45, -0.55, -0.7]); lightdir /= np.linalg.norm(lightdir)

    rng = np.random.default_rng(11)
    confetti = [dict(x=rng.uniform(0, W), y=rng.uniform(-H * 0.2, H), w=rng.uniform(12, 24), h=rng.uniform(7, 13),
                     vy=rng.uniform(50, 120), sway=rng.uniform(10, 40), ph=rng.uniform(0, 6.28),
                     spin=rng.uniform(1.5, 4), rot=rng.uniform(0, 360), vr=rng.uniform(-90, 90),
                     col=CONFETTI[rng.integers(len(CONFETTI))], front=rng.random() < 0.45) for _ in range(90)]
    pencil_cols = [(240, 110, 110), (110, 170, 230), (250, 196, 80), (150, 120, 210), (110, 196, 150), (246, 150, 90)]
    pencils = []
    spots = [(150, 380), (930, 330), (110, 900), (970, 1000), (260, 1820), (830, 1800)]
    for i, (px, py) in enumerate(spots):
        size = 300 if i % 2 == 0 else 250
        pencils.append(dict(spr=pencil_sprite(pencil_cols[i], size, int(size * 0.07)), x=px, y=py,
                            a0=rng.uniform(0, 360), va=rng.uniform(-25, 25), ph=rng.uniform(0, 6.28),
                            front=i in (1, 2)))

    shadow_base = Image.new("L", (600, 160), 0)
    ImageDraw.Draw(shadow_base).ellipse([60, 40, 540, 120], fill=120)
    shadow_base = shadow_base.filter(ImageFilter.GaussianBlur(24))

    f_title = ImageFont.truetype(FONT_AR_BOLD, 88)
    f_sub = ImageFont.truetype(FONT_AR_DISPLAY, 54)
    f_label = ImageFont.truetype(FONT_AR_BOLD, 30)
    f_handle = ImageFont.truetype(FONT_SANS, 40)
    icons = [(FOLLOW_IG, social_icon("instagram", 76, (214, 72, 140, 255))),
             (FOLLOW_TT, social_icon("tiktok", 76, (30, 30, 36, 255)))]
    n_frames = int(END_SECONDS * FPS)

    def draw_confetti(layer_front, t, canvas):
        d = ImageDraw.Draw(canvas)
        for c in confetti:
            if c["front"] != layer_front:
                continue
            y = (c["y"] + c["vy"] * t) % (H + 100) - 50
            x = c["x"] + c["sway"] * math.sin(t * 1.3 + c["ph"])
            hh = c["h"] * abs(math.cos(t * c["spin"] + c["ph"])) + 1
            a = math.radians(c["rot"] + c["vr"] * t)
            ca, sa = math.cos(a), math.sin(a)
            pts = [(x + px * ca - py * sa, y + px * sa + py * ca)
                   for px, py in ((-c["w"] / 2, -hh / 2), (c["w"] / 2, -hh / 2), (c["w"] / 2, hh / 2), (-c["w"] / 2, hh / 2))]
            col = c["col"] if layer_front else tuple(int(v * 0.5 + 255 * 0.5) for v in c["col"])
            d.polygon(pts, fill=col)

    def draw_pencils(layer_front, t, canvas):
        for p in pencils:
            if p["front"] != layer_front:
                continue
            ang = p["a0"] + p["va"] * t
            rot = p["spr"].rotate(ang, expand=True, resample=Image.BICUBIC)
            x = p["x"] + 20 * math.sin(t * 0.9 + p["ph"]) - rot.width / 2
            y = p["y"] + 26 * math.sin(t * 0.7 + p["ph"]) - 30 * t - rot.height / 2
            canvas.alpha_composite(rot, (int(x), int(y))) if x >= 0 and y >= 0 else canvas.paste(rot, (int(x), int(y)), rot)

    for f in range(n_frames):
        t = f / FPS
        p = t / END_SECONDS
        phi = math.radians(-38 + 76 * smooth(p))
        psi = math.radians(10 + 3 * math.sin(2 * math.pi * t / 3))
        bob = math.sin(2 * math.pi * t / 2.6)
        cy = cy0 + 18 * bob

        def xf(pt):
            x, y, z = pt
            x, z = x * math.cos(phi) + z * math.sin(phi), -x * math.sin(phi) + z * math.cos(phi)
            y, z = y * math.cos(psi) - z * math.sin(psi), y * math.sin(psi) + z * math.cos(psi)
            return np.array([x, y, z])

        def proj(pt):
            s = F / (F + pt[2])
            return (cx + pt[0] * s, cy + pt[1] * s)

        hw, hh, hd = bw / 2, bh / 2, bd / 2
        faces = [
            ("front", [(-hw, -hh, -hd), (hw, -hh, -hd), (hw, hh, -hd), (-hw, hh, -hd)], (0, 0, -1)),
            ("spine", [(-hw, -hh, hd), (-hw, -hh, -hd), (-hw, hh, -hd), (-hw, hh, hd)], (-1, 0, 0)),
            ("block", [(hw, -hh, -hd), (hw, -hh, hd), (hw, hh, hd), (hw, hh, -hd)], (1, 0, 0)),
            ("top", [(-hw, -hh, hd), (hw, -hh, hd), (hw, -hh, -hd), (-hw, -hh, -hd)], (0, -1, 0)),
        ]

        frame = bg.copy()
        sh = shadow_base.resize((int(600 * (1 - 0.08 * bob)), 160))
        shadow_img = Image.new("RGBA", sh.size, (90, 60, 110, 255)); shadow_img.putalpha(sh.point(lambda v: v * (0.85 - 0.15 * bob)))
        frame.alpha_composite(shadow_img, (int(cx - sh.width / 2), int(cy0 + 370)))
        draw_confetti(False, t, frame)
        draw_pencils(False, t, frame)

        visible = []
        for name, corners, normal in faces:
            pts = [xf(c) for c in corners]
            n = xf(normal)
            center = sum(pts) / 4
            if np.dot(n, center - np.array([0, 0, -F])) < 0:
                visible.append((center[2], name, pts, n))
        for _, name, pts, n in sorted(visible, key=lambda v: -v[0]):
            bright = 0.62 + 0.45 * max(0.0, float(np.dot(n, lightdir)))
            if name == "front":
                band_u = 0.5 - math.degrees(phi) / 60
                xs = np.arange(bw, dtype=np.float32)[None, :, None] / bw
                ys = np.arange(bh, dtype=np.float32)[:, None, None] / bh
                band = 0.22 * np.exp(-(((xs + ys * 0.4) - band_u) / 0.14) ** 2)
                tex = Image.fromarray(np.clip(front_arr * bright + band * 255, 0, 255).astype(np.uint8))
            else:
                src = {"spine": spine, "block": block, "top": top_face}[name]
                tex = ImageEnhance.Brightness(src).enhance(bright)
            warp_onto(frame, pad(tex), [proj(pt) for pt in pts])

        draw_pencils(True, t, frame)
        draw_confetti(True, t, frame)

        k = smooth((t - 1.6) / 0.9)
        k2 = smooth((t - 2.8) / 0.8)
        if k > 0:
            over = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            d = ImageDraw.Draw(over)
            for text, font, y, col, a in ((STORE, f_title, 1180, (84, 52, 92), k),
                                          (f"{BOOK_TITLE}  •  {BOOK_SUB}", f_sub, 1280, (190, 96, 80), k)):
                d.text((W / 2, y + 30 * (1 - a)), text, font=font, anchor="mm", fill=col + (int(255 * a),))
            if k2 > 0:
                # card with both accounts
                cy_card = 1345 + 30 * (1 - k2)
                d.rounded_rectangle([150, cy_card, 930, cy_card + 290], 36, fill=(255, 255, 255, int(240 * k2)))
                for i, (name, icon) in enumerate(icons):
                    y = cy_card + 28 + i * 130
                    ic = icon.copy(); ic.putalpha(icon.getchannel("A").point(lambda v: int(v * k2)))
                    over.alpha_composite(ic, (930 - 50 - 76, int(y) + 10))
                    d.text((780, y - 4), name, font=f_label, anchor="ra", fill=(120, 100, 130, int(255 * k2)))
                    d.text((780, y + 48), HANDLE, font=f_handle, anchor="ra", fill=(60, 40, 70, int(255 * k2)))
            frame.alpha_composite(over)
        writer.write(frame)


# ---------------------------------------------------------------- main

def main():
    SCENES.mkdir(parents=True, exist_ok=True)
    p1, p2, p3 = (Image.open(p).convert("RGB") for p in PAGES)
    print("coloring cover art...")
    p1c = colored_page(PAGES[0], BRIGHT, (0.5, 0.48))
    p2c = colored_page(PAGES[1], BRIGHT)
    p3c = colored_page(PAGES[2], BRIGHT)
    cover = make_cover(p2c)

    jobs = [
        ("scene1.mp4", lambda w: scene_cover(cover, w)),
        ("scene2.mp4", lambda w: scene_flip([p3, p2, p1, p3c, p2c, p1], w)),
        ("scene3.mp4", lambda w: scene_color(PAGES[0], w)),
        ("scene4.mp4", lambda w: scene_float(cover, w)),
    ]
    for name, fn in jobs:
        print("rendering", name)
        w = Writer(SCENES / name)
        fn(w)
        w.close()

    print("joining scenes...")
    durations = [SCENE_SECONDS] * (len(jobs) - 1) + [END_SECONDS]
    inputs, chain, prev, length = [], [], "[0:v]", durations[0]
    for i, (name, _) in enumerate(jobs):
        inputs += ["-i", str(SCENES / name)]
        if i == 0:
            continue
        offset = length - FADE_SECONDS
        label = f"[v{i}]"
        chain.append(f"{prev}[{i}:v]xfade=transition=fade:duration={FADE_SECONDS}:offset={offset}{label}")
        prev, length = label, offset + durations[i]
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", ";".join(chain),
                    "-map", prev, "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(OUT / "coloring_book_promo.mp4")], check=True)
    print("done:", OUT / "coloring_book_promo.mp4")


if __name__ == "__main__":
    main()
