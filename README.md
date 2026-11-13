# stillwater

A sound bath app for your phone — singing bowls, tuning forks, gongs, chimes, nature sounds and binaural tones, plus a sleep mode with white/brown noise and a friendly owl.

Everything runs in the browser from `index.html`; the owl and om sounds live in `audio/`.

**Om** plays one chanted "Aaa–ooo–mmm". **Om circle** keeps a group of voices chanting together (three low voices and one higher voice), each taking a breath between chants. The voices are tuned to the nearest note of whichever bowl tuning is selected. They were made with `tools/om_synth.py` (`python3 tools/om_synth.py out/` rebuilds them; needs numpy, scipy and ffmpeg).

## Run it

- **On GitHub Pages:** Settings → Pages → deploy from the `main` branch (root). The app will be at `https://thepatrickreardon-sys.github.io/stillwater/`.
- **Locally:** serve the folder (e.g. `python3 -m http.server`) and open `http://localhost:8000`. Opening the file directly may block the owl sounds from loading.

Tap once to start audio — phones won't play sound until you touch the screen.
