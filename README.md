# Coloring book promo video – Happy Life Store / متجر الحياة السعيدة

Vertical 9:16 video (1080×1920, 30 fps, ~20 s) for Instagram Reels and TikTok, made from the line-art pages in `pages/`.
All on-screen text is in Arabic, and it ends on the Instagram and TikTok account **@_happy.life.store**.

**Final video:** [`output/coloring_book_promo.mp4`](output/coloring_book_promo.mp4)

| Scene | Prompt | Clip |
|---|---|---|
| 1 | Cover reveal: book on a wooden desk, warm light, slow push-in and tilt, cover lifting in a breeze | `output/scenes/scene1.mp4` |
| 2 | Flipping pages: open book, two page turns showing the line art | `output/scenes/scene2.mp4` |
| 3 | Coloring: top-down mandala filled with warm pencil colors | `output/scenes/scene3.mp4` |
| 4 | Floating book on a pastel background with confetti and pencils, then the store name and accounts | `output/scenes/scene4.mp4` |

The video has no audio, so add trending music in the Instagram or TikTok app when you post. Text stays clear of the app's top tabs, bottom caption and side buttons.

## Rebuild

```bash
pip install numpy pillow scipy   # ffmpeg must also be installed
python3 make_video.py
```

Edit the text constants at the top of `make_video.py` (`STORE`, `BOOK_TITLE`, `HANDLE`, …) to change the wording.

Fonts in `fonts/` (Cairo and Amiri) come from Google Fonts under the SIL Open Font License.

## Video 2 – coloring clip

**Video:** [`output/coloring_clip_promo.mp4`](output/coloring_clip_promo.mp4) (1080×1920, ~9.5 s, with the clip's own sound)

This video puts the supplied clip (`clips/coloring_clip.mp4`, where a mosque drawing fills with color) into a 9:16 frame over a blurred background. A shaded hand holding a colored pencil (`hand.py`) moves over the page and follows wherever new color appears, so it looks like it is doing the coloring. The pencil takes the color of the spot it is coloring. It adds the Arabic store tag and the caption "من الأبيض والأسود… إلى لوحة ملوّنة", then ends on the store name and the Instagram and TikTok handles.

```bash
python3 make_video2.py clips/coloring_clip.mp4
```

You can run it with any other clip to make another video in the same style.

## Book covers

Front covers for both coloring books, print size (8.5 × 11 in at 300 dpi, 2550×3300 px), plus small previews:

| Book | Cover |
|---|---|
| ماندالا الزهور – كتاب تلوين للكبار | [`output/covers/floral_mandalas_cover.png`](output/covers/floral_mandalas_cover.png) |
| Nature Escape (the three scenes from the happy_store site) | [`output/covers/nature_escape_cover.png`](output/covers/nature_escape_cover.png) |

Each cover shows the art half colored and half line art, with a colored pencil on the edge between them.

```bash
python3 make_cover.py ../happy_store   # path to the happy_store repo (for the Nature Escape pages)
```
