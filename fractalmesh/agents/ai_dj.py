#!/usr/bin/env python3
"""
FractalMesh AI DJ v4.1
OpenRouter LLM → structured JSON → midiutil MIDI → fluidsynth WAV → mpv
Loops every DJ_INTERVAL seconds. Writes .latest_track for nft_minter.py.
Vault: OPENROUTER_KEY or OPENROUTER_API_KEY (FractalMesh default)
"""
import os, sys, json, time, subprocess, logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [DJ] %(message)s",
    handlers=[
        logging.FileHandler(Path.home() / ".fm_logs/ai_dj.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

try:
    import requests
    from midiutil import MIDIFile
except ImportError as exc:
    log.error(f"Missing dep: {exc}  —  run: pip install requests midiutil")
    sys.exit(1)

# Load vault
VAULT_PATH = os.getenv("VAULT_PATH", str(Path.home() / ".secrets/fractal.env"))
if Path(VAULT_PATH).exists():
    for line in Path(VAULT_PATH).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

OPENROUTER_KEY = (os.getenv("OPENROUTER_KEY") or
                  os.getenv("OPENROUTER_API_KEY", ""))
TRACKS_DIR     = Path.home() / "synthwave" / "tracks"
LATEST_FILE    = Path.home() / "synthwave" / ".latest_track"
INTERVAL       = int(os.getenv("DJ_INTERVAL", "600"))
MODEL          = os.getenv("DJ_MODEL", "mistralai/mistral-7b-instruct:free")

SF2_CANDIDATES = [
    os.getenv("SOUNDFONT_PATH", ""),
    "/data/data/com.termux/files/usr/share/soundfonts/default.sf2",
    "/data/data/com.termux/files/usr/share/soundfonts/FluidR3_GM.sf2",
    str(Path.home() / "soundfonts/default.sf2"),
    "/usr/share/sounds/sf2/FluidR3_GM.sf2",
    "/usr/share/soundfonts/FluidR3_GM.sf2",
]

SYSTEM_PROMPT = """You are a MIDI composer. Output ONLY valid JSON — no markdown, no backticks.
Return a JSON object:
{
  "tempo": <int 90-140>,
  "tracks": [
    {
      "name": "<string>",
      "channel": <int 0-15>,
      "program": <int 0-127>,
      "notes": [
        {"pitch": <int 21-108>, "start": <float beats>, "duration": <float beats>, "volume": <int 60-110>}
      ]
    }
  ]
}
Requirements:
- 4 tracks: bass (program 38), lead synth (program 81), pads (program 91), arpeggio (program 82)
- At least 32 notes per track, total duration >= 16 beats
- Style: 80s synthwave — driving bass, neon arpeggios, lush pads, melodic lead"""

THEMES = [
    "neon city night drive", "cyberpunk chase sequence",
    "retro-future sunset", "digital frontier horizon",
    "electric grid pulse", "chrome boulevard",
    "fractal mesh resonance", "albury after midnight",
]


def get_soundfont() -> str:
    for sf in SF2_CANDIDATES:
        if sf and Path(sf).exists():
            return sf
    try:
        r = subprocess.run(
            ["find", "/", "-name", "*.sf2", "-type", "f"],
            capture_output=True, text=True, timeout=10
        )
        for line in r.stdout.strip().splitlines():
            if line.strip():
                return line.strip()
    except Exception:
        pass
    return ""


def ask_openrouter(theme: str) -> dict:
    if not OPENROUTER_KEY:
        raise ValueError("OPENROUTER_KEY / OPENROUTER_API_KEY not set in vault")
    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_KEY}",
            "Content-Type":  "application/json",
            "HTTP-Referer":  "https://fractalmesh.io",
            "X-Title":       "FractalMesh AI DJ",
        },
        json={
            "model":       MODEL,
            "messages":    [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": f"Generate synthwave MIDI. Theme: {theme}. Make it unique and energetic."},
            ],
            "temperature": 0.92,
            "max_tokens":  2400,
        },
        timeout=90,
    )
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"].strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1].lstrip("json").strip() if len(parts) > 1 else raw
    return json.loads(raw)


def build_midi(track_def: dict, out_path: Path) -> None:
    tracks = track_def["tracks"]
    tempo  = int(track_def.get("tempo", 120))
    midi   = MIDIFile(len(tracks))
    for i, t in enumerate(tracks):
        ch   = int(t.get("channel", i % 16))
        prog = int(t.get("program", 0))
        midi.addTempo(i, 0, tempo)
        midi.addProgramChange(i, ch, 0, prog)
        for n in t.get("notes", []):
            try:
                midi.addNote(
                    track    = i,
                    channel  = ch,
                    pitch    = max(21, min(108, int(n["pitch"]))),
                    time     = max(0.0, float(n["start"])),
                    duration = max(0.05, float(n["duration"])),
                    volume   = max(1, min(127, int(n.get("volume", 80)))),
                )
            except Exception as e:
                log.debug(f"Skipping bad note {n}: {e}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        midi.writeFile(f)
    log.info(f"MIDI written: {out_path}")


def render_wav(midi_path: Path, wav_path: Path, sf2: str) -> bool:
    if not sf2:
        log.warning("No soundfont — WAV render skipped")
        return False
    try:
        r = subprocess.run(
            ["fluidsynth", "-ni", sf2, str(midi_path), "-F", str(wav_path), "-r", "44100"],
            capture_output=True, timeout=180
        )
        if r.returncode == 0:
            log.info(f"WAV rendered: {wav_path}")
            return True
        log.warning(f"fluidsynth error: {r.stderr.decode()[:300]}")
    except FileNotFoundError:
        log.warning("fluidsynth not installed")
    except subprocess.TimeoutExpired:
        log.warning("fluidsynth timed out")
    return False


def play(path: Path) -> None:
    try:
        subprocess.Popen(
            ["mpv", "--no-video", "--really-quiet", str(path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        log.info(f"Playing: {path.name}")
    except FileNotFoundError:
        log.warning("mpv not found — playback skipped")


def main():
    log.info("═══ FractalMesh AI DJ v4.1 started ═══")
    sf2   = get_soundfont()
    cycle = 0
    log.info(f"Soundfont: {sf2 or 'NONE — MIDI only'}")
    log.info(f"Model: {MODEL} | Interval: {INTERVAL}s")

    while True:
        cycle += 1
        ts    = int(time.time())
        theme = THEMES[cycle % len(THEMES)]
        log.info(f"Cycle {cycle} | Theme: '{theme}'")
        try:
            track_def  = ask_openrouter(theme)
            midi_path  = TRACKS_DIR / f"track_{ts}.mid"
            wav_path   = TRACKS_DIR / f"track_{ts}.wav"
            build_midi(track_def, midi_path)
            has_wav    = render_wav(midi_path, wav_path, sf2)
            final_path = wav_path if has_wav else midi_path
            play(final_path)
            LATEST_FILE.write_text(str(final_path))
            log.info(f"Track ready → nft_minter: {final_path.name}")
        except json.JSONDecodeError as e:
            log.error(f"LLM returned invalid JSON: {e}")
        except requests.HTTPError as e:
            log.error(f"OpenRouter HTTP error: {e}")
        except requests.RequestException as e:
            log.error(f"Network error: {e}")
        except ValueError as e:
            log.error(str(e))
        except Exception as e:
            log.exception(f"Unexpected error cycle {cycle}: {e}")

        log.info(f"Sleeping {INTERVAL}s...")
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
