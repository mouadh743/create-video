#!/usr/bin/env python3
"""Front covers for the two Happy Life Store coloring books.

  1. ماندالا الزهور (Floral Mandalas) - Arabic adult coloring book, pages in pages/
  2. ملاذ الطبيعة (Nature Escape) - the three scenes sold in the happy_store site

Each cover shows its art half colored and half line art, so buyers see what the
pages look like before and after coloring.

Usage:  python3 make_cover.py [path/to/happy_store]     (needs numpy, pillow, scipy)
Output: output/covers/*.png (print size, 8.5x11 in at 300 dpi) and *_preview.jpg
"""
import math
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from make_video import (BOOK_SUB, BOOK_TITLE, BRIGHT, FONT_AR_BOLD, FONT_AR_DISPLAY,
                        HANDLE, PAGES, ROOT, STORE, colored_page, pencil_sprite)

CW, CH = 2550, 3300            # 8.5 x 11 in at 300 dpi
OUT = ROOT / "output" / "covers"
QR = ROOT / "assets" / "instagram_qr.png"   # Instagram QR code for @_happy.life.store
STORE_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / "happy_store"

NATURE_TITLE = "ملاذ الطبيعة"            # Nature Escape
NATURE_SUB = "كتاب تلوين لأماكن هادئة"   # A colouring book of quiet places
NATURE_BADGE = "٣ رسومات  •  من السهل إلى المفصّل"  # 3 scenes - easy to detailed
NATURE_PAGES = [("treehouse-hideaway", "treehouse.webp"),
                ("hidden-waterfall", "waterfall.webp"),
                ("mountain-lake-sunrise", "mountain-lake.webp")]


def font(path, size):
    return ImageFont.truetype(path, size)


def reveal(line, colored, angle=-28, width=0.10, at=0.52):
    """Colored on the left, line art on the right, split by a soft diagonal."""
    w, h = line.size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    a = math.radians(angle)
    proj = (xx / w - 0.5) * math.cos(a) + (yy / h - 0.5) * math.sin(a) + 0.5
    t = np.clip((at + width / 2 - proj) / width, 0, 1)
    m = (t * t * (3 - 2 * t))[..., None]
    out = np.asarray(colored, np.float32) * m + np.asarray(line.convert("RGB"), np.float32) * (1 - m)
    return Image.fromarray(out.astype(np.uint8))


def drop_shadow(canvas, box, radius, blur, opacity, offset=(0, 24)):
    sh = Image.new("L", canvas.size, 0)
    x0, y0, x1, y1 = box
    ImageDraw.Draw(sh).rounded_rectangle([x0 + offset[0], y0 + offset[1], x1 + offset[0], y1 + offset[1]],
                                         radius, fill=int(255 * opacity))
    sh = sh.filter(ImageFilter.GaussianBlur(blur))
    canvas.paste(Image.new("RGB", canvas.size, (20, 20, 30)), (0, 0), sh)


def rounded(im, r):
    mask = Image.new("L", im.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, im.width - 1, im.height - 1], r, fill=255)
    out = im.convert("RGBA")
    out.putalpha(mask)
    return out


def place_pencil(canvas, color, tip, angle, length=900, width=60):
    sprite = pencil_sprite(color, length, width)
    rot = sprite.rotate(angle, expand=True, resample=Image.BICUBIC)
    a = math.radians(angle)
    tx, ty = -length / 2 * math.cos(a), length / 2 * math.sin(a)
    px, py = int(tip[0] - rot.width / 2 - tx), int(tip[1] - rot.height / 2 - ty)
    shadow = Image.new("RGBA", rot.size, (0, 0, 0, 0))
    shadow.putalpha(rot.split()[3].point(lambda v: v * 0.35))
    canvas.paste(shadow.filter(ImageFilter.GaussianBlur(10)), (px + 18, py + 26), shadow.filter(ImageFilter.GaussianBlur(10)))
    canvas.paste(rot, (px, py), rot)


def paper_texture(w, h, base, seed=7, amount=10):
    rng = np.random.default_rng(seed)
    n = rng.normal(0, 1, (h // 4 + 1, w // 4 + 1)).astype(np.float32)
    n = np.asarray(Image.fromarray(((n * 40) + 128).clip(0, 255).astype(np.uint8)).resize((w, h), Image.BICUBIC),
                   np.float32) / 128 - 1
    img = np.asarray(base, np.float32)[None, None, :] + n[..., None] * amount
    return Image.fromarray(img.clip(0, 255).astype(np.uint8))


def place_qr(canvas, x, y, size):
    """Instagram QR code on a white rounded card, top-left corner at (x, y)."""
    qr = Image.open(QR).convert("RGB").resize((size, size), Image.LANCZOS)
    pad = size // 14
    card = Image.new("RGB", (size + 2 * pad, size + 2 * pad), (255, 255, 255))
    card.paste(qr, (pad, pad))
    box = (x, y, x + card.width, y + card.height)
    drop_shadow(canvas, box, 28, 18, 0.35, offset=(0, 10))
    card = rounded(card, 28)
    canvas.paste(card, (x, y), card)


def text_center(d, xy, s, f, fill, **kw):
    d.text(xy, s, font=f, anchor="mm", fill=fill, **kw)


# ---------------------------------------------------------------- cover 1: Floral Mandalas (Arabic)

def mandala_backdrop(w, h):
    """Deep teal with a faint ring pattern."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.hypot(xx - w / 2, yy - h * 0.55) / max(w, h)
    top, bot = np.array((34, 98, 106), np.float32), np.array((22, 62, 74), np.float32)
    img = top + (bot - top) * np.clip(r * 1.4, 0, 1)[..., None]
    img += (np.cos(r * 220) > 0.92)[..., None] * 6
    return Image.fromarray(img.clip(0, 255).astype(np.uint8))


def cover_mandala():
    page = Image.open(PAGES[1]).convert("RGB")
    colored = colored_page(PAGES[1], BRIGHT)
    art = reveal(page, colored)

    c = mandala_backdrop(CW, CH)
    d = ImageDraw.Draw(c, "RGBA")
    gold = (214, 172, 92)
    d.rectangle([60, 60, CW - 60, CH - 60], outline=gold, width=8)
    d.rectangle([84, 84, CW - 84, CH - 84], outline=(*gold, 140), width=3)

    # art card
    aw, ah = 1800, 1940
    full = art.resize((aw, int(aw * page.height / page.width)), Image.LANCZOS)
    top = (full.height - ah) // 2
    art = full.crop((0, top, aw, top + ah))
    ax, ay = (CW - aw) // 2, 900
    drop_shadow(c, (ax, ay, ax + aw, ay + ah), 40, 40, 0.55)
    frame = Image.new("RGB", (aw + 60, ah + 60), (252, 247, 236))
    c.paste(rounded(frame, 50), (ax - 30, ay - 30), rounded(frame, 50))
    c.paste(rounded(art, 30), (ax, ay), rounded(art, 30))

    # pencil resting on the color/line boundary
    place_pencil(c, (228, 104, 140), (ax + aw * 0.60, ay + ah * 0.46), 28, 1100, 74)

    # title panel
    d = ImageDraw.Draw(c, "RGBA")
    d.rounded_rectangle([220, 190, CW - 220, 760], 60, fill=(252, 246, 234, 245))
    d.rounded_rectangle([250, 220, CW - 250, 730], 46, outline=gold, width=6)
    text_center(d, (CW / 2, 400), BOOK_TITLE, font(FONT_AR_DISPLAY, 250), (60, 40, 70))
    text_center(d, (CW / 2, 610), BOOK_SUB, font(FONT_AR_BOLD, 110), (176, 92, 70))

    # badge
    bx, by, br = CW - 400, 2830, 190
    d.ellipse([bx - br, by - br, bx + br, by + br], fill=(244, 196, 80), outline=(252, 246, 234), width=10)
    text_center(d, (bx, by - 40), "للاسترخاء", font(FONT_AR_BOLD, 64), (90, 50, 40))
    text_center(d, (bx, by + 50), "والإبداع", font(FONT_AR_BOLD, 64), (90, 50, 40))

    # store footer
    text_center(d, (CW / 2, 3010), STORE, font(FONT_AR_DISPLAY, 130), (252, 240, 214))
    text_center(d, (CW / 2, 3150), HANDLE, font(FONT_AR_BOLD, 58), (*gold, 255))
    place_qr(c, 130, 2895, 290)
    return c


# ---------------------------------------------------------------- cover 2: Nature Escape

def parse_washes(path):
    """Read each product's CSS watercolour wash from happy_store/lib/products.ts."""
    src = path.read_text()
    washes = {}
    for m in re.finditer(r'slug:\s*"([^"]+)".*?wash:\s*\[(.*?)\]\.join', src, re.S):
        washes[m.group(1)] = re.findall(r'"([^"]+)"', m.group(2))
    return washes


def hex_rgba(s):
    if s == "transparent":
        return np.array([0, 0, 0, 0], np.float32)
    s = s.lstrip("#")
    return np.array([int(s[i:i + 2], 16) for i in (0, 2, 4)] + [255], np.float32) / 255


def gradient_layer(css, w, h):
    """Render one CSS radial/linear gradient (the subset used in products.ts) as premultiplied RGBA."""
    kind, body = re.match(r"(\w+-gradient)\((.*)\)$", css).groups()
    parts = [p.strip() for p in body.split(",")]
    head, stops_src = parts[0], parts[1:]
    stops = []
    for p in stops_src:
        toks = p.split()
        col = hex_rgba(toks[0])
        for pos in toks[1:]:
            stops.append((float(pos.rstrip("%")) / 100 if pos != "0" else 0.0, col))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    if kind == "linear-gradient":
        t = 1 - yy / (h - 1) if head == "to top" else yy / (h - 1)
    else:
        nums = [float(v) / 100 for v in re.findall(r"([\d.]+)%", head)]
        if head.startswith("ellipse"):
            rx, ry, cx, cy = nums
            t = np.hypot((xx / w - cx) / rx, (yy / h - cy) / ry)
        else:  # circle, farthest-corner
            cx, cy = nums
            far = max(math.hypot(px - cx * w, py - cy * h) for px in (0, w) for py in (0, h))
            t = np.hypot(xx - cx * w, yy - cy * h) / far
    pre = [(p, np.array([*(c[:3] * c[3]), c[3]], np.float32)) for p, c in stops]
    out = np.zeros((h, w, 4), np.float32)
    out[:] = pre[0][1]
    for (p0, c0), (p1, c1) in zip(pre, pre[1:]):
        k = np.clip((t - p0) / max(p1 - p0, 1e-6), 0, 1)[..., None]
        seg = c0 + (c1 - c0) * k
        out = np.where((t >= p0)[..., None], seg, out)
    return out


def washed(line, layers):
    w, h = line.size
    img = np.ones((h, w, 3), np.float32)
    for css in reversed(layers):  # first CSS layer is on top
        g = gradient_layer(css, w, h)
        img = g[..., :3] + img * (1 - g[..., 3:])
    gray = np.asarray(line.convert("L"), np.float32)[..., None] / 255
    return Image.fromarray((img * gray * 255).clip(0, 255).astype(np.uint8))


def polaroid(img, w, h, pad=40, bottom=40):
    """Fill a w x h window with the image (cropping the overflow), on a white card."""
    scale = max(w / img.width, h / img.height)
    img = img.resize((math.ceil(img.width * scale), math.ceil(img.height * scale)), Image.LANCZOS)
    x0, y0 = (img.width - w) // 2, (img.height - h) // 2
    img = img.crop((x0, y0, x0 + w, y0 + h))
    card = Image.new("RGB", (w + 2 * pad, h + pad + bottom), (255, 253, 248))
    card.paste(img, (pad, pad))
    ImageDraw.Draw(card).rectangle([pad, pad, pad + w - 1, pad + h - 1], outline=(220, 214, 200), width=3)
    return card


def paste_tilted(canvas, card, center, angle):
    rgba = rounded(card, 14)
    rot = rgba.rotate(angle, expand=True, resample=Image.BICUBIC)
    alpha = rot.split()[3]
    shadow = Image.new("RGBA", rot.size, (30, 40, 30, 0))
    shadow.putalpha(alpha.point(lambda v: v * 0.45).filter(ImageFilter.GaussianBlur(26)))
    x, y = int(center[0] - rot.width / 2), int(center[1] - rot.height / 2)
    canvas.paste(shadow, (x + 22, y + 36), shadow)
    canvas.paste(rot, (x, y), rot)


def cover_nature():
    washes = parse_washes(STORE_DIR / "lib" / "products.ts")
    arts = []
    for i, (slug, fname) in enumerate(NATURE_PAGES):
        line = Image.open(STORE_DIR / "public" / "products" / fname).convert("RGB")
        col = washed(line, washes[slug])
        # the right-hand card shows its colored half on the right, away from the front card
        arts.append(reveal(col, line, angle=30, at=0.45) if i == 2 else reveal(line, col, angle=-30, at=0.55))

    c = paper_texture(CW, CH, (247, 240, 224))
    d = ImageDraw.Draw(c, "RGBA")
    # soft sky and meadow bands
    yy = np.linspace(0, 1, CH, dtype=np.float32)[:, None, None]
    base = np.asarray(c, np.float32)
    sky = np.array((206, 232, 240), np.float32)
    meadow = np.array((196, 222, 170), np.float32)
    base = base + (sky - base) * np.clip(1 - yy / 0.35, 0, 1) * 0.6
    base = base + (meadow - base) * np.clip((yy - 0.80) / 0.2, 0, 1) * 0.7
    c = Image.fromarray(base.clip(0, 255).astype(np.uint8))
    d = ImageDraw.Draw(c, "RGBA")

    green, ink = (58, 104, 70), (44, 52, 48)
    text_center(d, (CW / 2, 330), NATURE_TITLE, font(FONT_AR_DISPLAY, 290), green)
    text_center(d, (CW / 2, 560), NATURE_SUB, font(FONT_AR_BOLD, 96), ink)
    d.line([CW / 2 - 520, 650, CW / 2 + 520, 650], fill=(*green, 160), width=4)

    # three scenes as tilted cards: waterfall in front
    cards = [polaroid(arts[0], 820, 1300), polaroid(arts[2], 820, 1300), polaroid(arts[1], 1000, 1500)]
    paste_tilted(c, cards[0], (720, 1640), 8)
    paste_tilted(c, cards[1], (1830, 1640), -7)
    paste_tilted(c, cards[2], (CW / 2, 1790), 0)

    place_pencil(c, (106, 166, 228), (CW / 2 + 160, 1700), 35, 1150, 76)
    place_pencil(c, (128, 200, 158), (CW / 2 - 640, 2500), 150, 900, 66)

    d = ImageDraw.Draw(c, "RGBA")
    d.rounded_rectangle([640, 2780, CW - 640, 2900], 60, fill=(*green, 235))
    text_center(d, (CW / 2, 2838), NATURE_BADGE, font(FONT_AR_BOLD, 66), (252, 246, 234))
    text_center(d, (CW / 2, 3040), STORE, font(FONT_AR_DISPLAY, 120), green)
    text_center(d, (CW / 2, 3160), HANDLE, font(FONT_AR_BOLD, 58), ink)
    place_qr(c, 110, 2830, 340)
    return c


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in (("floral_mandalas_cover", cover_mandala), ("nature_escape_cover", cover_nature)):
        print("rendering", name)
        img = fn()
        img.save(OUT / f"{name}.png", dpi=(300, 300))
        img.resize((850, 1100), Image.LANCZOS).save(OUT / f"{name}_preview.jpg", quality=90)
    print("done:", OUT)


if __name__ == "__main__":
    main()
