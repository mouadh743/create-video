#!/usr/bin/env python3
"""Second promo video: a supplied coloring clip framed for mobile, with Arabic text and an end card.

Usage:  python3 make_video2.py clips/coloring_clip.mp4
Output: output/coloring_clip_promo.mp4  (1080x1920, 30 fps, keeps the clip's audio)
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from make_video import (FONT_AR_BOLD, FONT_AR_DISPLAY, FONT_SANS, FPS, H, HANDLE, STORE, W, Writer,
                        brand_tag, smooth, social_icon)

ROOT = Path(__file__).resolve().parent
CAPTION = "من الأبيض والأسود… إلى لوحة ملوّنة"   # From black and white... to a colored painting
SUBTITLE = "كتب تلوين للكبار"                    # Coloring books for adults
FOLLOW_IG = "تابعونا على إنستغرام"
FOLLOW_TT = "تابعونا على تيك توك"
END_SECONDS = 4.5
CLIP_W = 1000          # clip width on screen; top and bottom are filled with a blurred copy
CLIP_TOP = 300


def read_frames(path):
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=width,height", "-of", "csv=p=0", str(path)],
                           capture_output=True, text=True, check=True).stdout.strip().split(",")
    w, h = int(probe[0]), int(probe[1])
    proc = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-i", str(path), "-vf", f"fps={FPS}",
                             "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    size = w * h * 3
    while True:
        buf = proc.stdout.read(size)
        if len(buf) < size:
            break
        yield Image.frombuffer("RGB", (w, h), buf)
    proc.wait()


def compose(frame, zoom=1.0):
    """Place the clip frame on a 9:16 canvas over a blurred, softened copy of itself."""
    cover =max(108 / frame.width, 192 / frame.height)
    small = frame.resize((int(frame.width * cover) + 1, int(frame.height * cover) + 1), Image.BILINEAR)
    small = small.crop(((small.width - 108) // 2, (small.height - 192) // 2,
                        (small.width - 108) // 2 + 108, (small.height - 192) // 2 + 192))
    bg = small.filter(ImageFilter.GaussianBlur(6)).resize((W, H), Image.BICUBIC)
    bg = ImageEnhance.Brightness(bg).enhance(0.8).convert("RGBA")
    cw = int(CLIP_W * zoom)
    ch = int(frame.height * cw / frame.width)
    clip = frame.resize((cw, ch), Image.LANCZOS)
    x, y = (W - cw) // 2, CLIP_TOP + (int(frame.height * CLIP_W / frame.width) - ch) // 2
    shadow = Image.new("L", (W, H), 0)
    ImageDraw.Draw(shadow).rounded_rectangle([x + 8, y + 20, x + cw + 8, y + ch + 24], 34, fill=140)
    shadow = shadow.filter(ImageFilter.GaussianBlur(22))
    bg = Image.composite(Image.new("RGBA", (W, H), (20, 10, 5, 255)), bg, shadow)
    mask = Image.new("L", (cw, ch), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, cw - 1, ch - 1], 32, fill=255)
    bg.paste(clip, (x, y), mask)
    return bg


def caption(frame, alpha):
    if alpha <= 0:
        return frame
    over = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    font = ImageFont.truetype(FONT_AR_BOLD, 50)
    tw = d.textlength(CAPTION, font=font)
    y = 1430 + 20 * (1 - alpha)
    d.rounded_rectangle([(W - tw) / 2 - 36, y - 46, (W + tw) / 2 + 36, y + 46], 46,
                        fill=(255, 250, 242, int(230 * alpha)))
    d.text((W / 2, y), CAPTION, font=font, anchor="mm", fill=(84, 52, 92, int(255 * alpha)))
    frame.alpha_composite(over)
    return frame


def end_card(frame, k, k2):
    over = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    rise = 30 * (1 - k)
    d.rounded_rectangle([100, 1060 + rise, 980, 1660 + rise], 44, fill=(255, 250, 244, int(240 * k)))
    f_title = ImageFont.truetype(FONT_AR_BOLD, 80)
    f_sub = ImageFont.truetype(FONT_AR_DISPLAY, 52)
    d.text((W / 2, 1150 + rise), STORE, font=f_title, anchor="mm", fill=(84, 52, 92, int(255 * k)))
    d.text((W / 2, 1240 + rise), SUBTITLE, font=f_sub, anchor="mm", fill=(190, 96, 80, int(255 * k)))
    if k2 > 0:
        f_label = ImageFont.truetype(FONT_AR_BOLD, 30)
        f_handle = ImageFont.truetype(FONT_SANS, 40)
        rows = [(FOLLOW_IG, social_icon("instagram", 76, (214, 72, 140, 255))),
                (FOLLOW_TT, social_icon("tiktok", 76, (30, 30, 36, 255)))]
        for i, (name, icon) in enumerate(rows):
            y = 1320 + i * 135 + 20 * (1 - k2)
            ic = icon.copy()
            ic.putalpha(icon.getchannel("A").point(lambda v: int(v * k2)))
            over.alpha_composite(ic, (790, int(y) + 10))
            d.text((760, y - 4), name, font=f_label, anchor="ra", fill=(120, 100, 130, int(255 * k2)))
            d.text((760, y + 48), HANDLE, font=f_handle, anchor="ra", fill=(60, 40, 70, int(255 * k2)))
    frame.alpha_composite(over)
    return frame


def main():
    src = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "clips" / "coloring_clip.mp4")
    out = ROOT / "output" / "coloring_clip_promo.mp4"
    silent = ROOT / "output" / "_video2_silent.mp4"
    writer = Writer(silent)
    n = 0
    last = None
    for frame in read_frames(src):
        t = n / FPS
        img = compose(frame)
        img = brand_tag(img, smooth((t - 0.2) / 0.5))
        img = caption(img, smooth((t - 0.4) / 0.5) * (1 - smooth((t - 2.6) / 0.5)))
        if t < 0.4:
            img = ImageEnhance.Brightness(img.convert("RGB")).enhance(smooth(t / 0.4))
        writer.write(img)
        last, n = frame, n + 1
    clip_seconds = n / FPS
    for i in range(int(END_SECONDS * FPS)):
        t = i / FPS
        img = compose(last, zoom=1.0 + 0.04 * smooth(t / END_SECONDS))
        img = brand_tag(img, 1.0)
        img = end_card(img, smooth((t - 0.1) / 0.7), smooth((t - 0.8) / 0.7))
        writer.write(img)
    writer.close()

    # keep the clip's own sound, fading it out as the end card comes in
    has_audio = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
                                "stream=index", "-of", "csv=p=0", str(src)],
                               capture_output=True, text=True).stdout.strip()
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(silent)]
    if has_audio:
        fade_at = max(clip_seconds - 0.2, 0)
        cmd += ["-i", str(src), "-filter_complex", f"[1:a]afade=t=out:st={fade_at}:d=1.2,apad[a]",
                "-map", "0:v", "-map", "[a]", "-c:a", "aac", "-b:a", "160k", "-shortest"]
    cmd += ["-c:v", "copy", "-movflags", "+faststart", str(out)]
    subprocess.run(cmd, check=True)
    silent.unlink()
    print("done:", out)


if __name__ == "__main__":
    main()
