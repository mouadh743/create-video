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
