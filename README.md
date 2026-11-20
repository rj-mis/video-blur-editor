# VideoBlur Editor

A small desktop application for hiding selected areas of a video using **pixelation** or **Gaussian blur** during specified time intervals. Designed for colleagues who need to obscure information before sharing or presenting a video.

The application runs locally on your computer, uses Python, and has a graphical interface with a video preview and a right-hand mask editor. No video upload or online service is required.

> **Privacy note:** Blurring or pixelation is not guaranteed to irreversibly anonymize a video. Review the **entire exported video** before sharing it, especially if it contains patient, personal, or confidential information. Audio may also contain sensitive information.

## Requirements

- Windows with Python 3.10+ (or another system with Python and Tkinter)
- [Spyder](https://www.spyder-ide.org/) (optional; the application can also run directly with Python)
- Python packages: `opencv-python`, `numpy`, `Pillow`
- Optional: [FFmpeg](https://ffmpeg.org/) on your PATH to retain audio during export

## Installation

Open a terminal (or Anaconda Prompt if you use Spyder through Anaconda) and install:

```bash
python -m pip install opencv-python numpy Pillow
```

Make sure the packages are installed in the **same Python environment** that Spyder uses. Tkinter is included with most standard Windows Python installations.

## Launch

1. Open `video_blur_editor.py` in Spyder.
2. Optionally edit the settings near the top of the file:

   ```python
   VIDEO_PATH = r""  # Leave empty to choose a video when starting
   DEFAULT_SECONDS_PER_MASK = 4.0
   DEFAULT_PIXEL_GRID = 8
   ```

3. Press **F5** to run. Choose a video if prompted.
4. The application opens a separate window.

Alternatively, run `python video_blur_editor.py` from a terminal.

## Create a mask

1. Use the video timeline to find the relevant frame.
2. Click **Mark IN** (`I`) at the start of the interval and **Mark OUT** (`O`) at its end. If you don't set both, a new mask starts at the current position and uses the default duration.
3. Choose **Rectangle** (`B`) and drag over the area to hide, or choose **Polygon** (`G`), click its corners, and press **Enter**.
4. The new mask appears in the **Masks** list on the right.
5. Add more masks as needed. Masks may overlap in space or time.

## Edit an existing mask

1. Click the mask in the **Masks** list.
2. Change **Start** and **End** (seconds or `MM:SS.sss`).
3. For rectangles, edit **X**, **Y**, **Width**, and **Height** in original video pixels. For polygons, edit the list of `x,y` points.
4. Choose **pixelate** or **blur**. For pixelation, a smaller **Pixel grid** produces larger blocks and stronger concealment.
5. Click **Apply edits**.

You can also click **Select / move** (`V`) to drag the selected mask or its corners in the video preview. Use **Duplicate**, **Delete**, or **Go to start** to manage masks.

**Note:** Masks stay at fixed coordinates during their time interval; they do not automatically track moving objects. For moving objects, add multiple masks with different time ranges and positions.

## Save, reopen, and export

- **Save masks:** saves mask settings as a JSON file next to the source video.
- **Load masks:** loads the JSON mask file for the current video. The saved video dimensions must match.
- **Export MP4:** renders a new MP4 with the selected effects applied. By default, the output filename is `<original_name>_pixelated.mp4` next to the source video.
- The original source video is not overwritten by the default export path.
- Audio is preserved when FFmpeg is installed and accessible; otherwise, the output video may have no audio.

## Keyboard shortcuts

| Key | Action |
| --- | --- |
| Space | Play / pause |
| Left / Right | Step backward / forward one frame |
| I / O | Mark interval start / end |
| N | Reset interval markers |
| V | Select or move masks |
| B | Draw rectangle |
| G | Draw polygon |
| Enter | Finish polygon |
| Esc | Cancel current drawing |
| Delete | Delete selected mask |

## Example

To hide a name visible from **00:05 to 00:12**:

1. Seek to `00:05` and press `I`.
2. Seek to `00:12` and press `O`.
3. Draw a rectangle around the name.
4. Select the mask in the right panel, check the time range, and adjust its coordinates if necessary.
5. Export the MP4 and watch the result to verify that the name is concealed throughout the interval.

## Troubleshooting

**`ModuleNotFoundError`**: Install the missing package in the Python environment used by Spyder (`python -m pip install ...`).

**Video does not open**: Confirm that the file exists and try an MP4 encoded with H.264; OpenCV's video codec support varies by installation.

**No audio in export**: Install FFmpeg and ensure `ffmpeg` is available from the command line. If the source has no audio, none will be added.

**Mask does not cover a moving area**: Masks are fixed in position. Create successive masks or expand the masked region.

**Sensitive content still visible**: Increase pixelation strength, enlarge the mask, extend its time interval, and check the entire export. Avoid assuming that a blurred output is anonymous.

## Files

- `video_blur_editor.py` — the application
- `README.md` — this guide

## Intended use

Internal video editing, demonstrations, and preparation of recordings for review. This is a lightweight utility, **not a certified medical de-identification or anonymization solution**.
