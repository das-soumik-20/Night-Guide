# speech.py
"""
Text-to-speech for alerts.

Live:   Speaker().say("Car on your left, 1.5 meters")   - non-blocking
Video:  make_narrated_video(video, events, out)          - bakes the spoken
        alerts into a copy of a saved video, at the moment each alert fired.

Windows (the project's main platform) uses the built-in Windows voice through
PowerShell's System.Speech - nothing to install. Other systems use pyttsx3 if
it is installed. If no voice is available, alerts are still printed.

Muxing audio into the video needs ffmpeg: `pip install imageio-ffmpeg`
(ships its own ffmpeg) or any ffmpeg on PATH.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave

import numpy as np

IS_WINDOWS = sys.platform.startswith('win')
SPEECH_RATE = 1        # Windows voice speed, -10 (slow) .. 10 (fast)
MAX_AGE_S = 3.0        # live: drop an alert if it waited longer than this to be spoken
SAMPLE_RATE = 22050


# ---------------------------------------------------------------------
# What to say
# ---------------------------------------------------------------------
def alert_to_speech(alert):
    """'Car on your left, 1.5 meters' / 'Person ahead, close'."""
    where = 'ahead' if alert['direction'] == 'center' else f"on your {alert['direction']}"
    if alert.get('distance_m') is not None:
        metres = max(0.5, round(alert['distance_m'] * 2) / 2)   # nearest half metre
        how_far = f"{metres:g} meter" + ('' if metres == 1 else 's')
    else:
        how_far = 'close'
    return f"{alert['class'].capitalize()} {where}, {how_far}"


def alerts_to_speech(alerts, max_items=2):
    """One utterance for the most urgent alerts of a frame."""
    return '. '.join(alert_to_speech(a) for a in alerts[:max_items])


def _ps_quote(text):
    return "'" + text.replace("'", "''") + "'"


# ---------------------------------------------------------------------
# Live speech
# ---------------------------------------------------------------------
_PS_LIVE = (
    "Add-Type -AssemblyName System.Speech;"
    "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
    f"$s.Rate = {SPEECH_RATE};"
    "while (($line = [Console]::In.ReadLine()) -ne $null) {"
    "  if ($line) { $s.Speak($line) };"
    "  [Console]::Out.WriteLine('done'); [Console]::Out.Flush()"
    "}"
)


class Speaker:
    """
    Speaks alerts in a background thread so detection never waits for the voice.
    Newest alert wins: if several arrive while one is being spoken, only the
    latest is said next, and anything older than MAX_AGE_S is dropped -
    a blind user needs what is in front of them *now*, not a backlog.
    """

    def __init__(self):
        self._pending = None
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = False
        self._ready = threading.Event()
        self.available = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=10)   # wait for the voice engine to start
        if not self.available:
            print("[speech] No text-to-speech voice available - alerts will only be printed.")

    def say(self, text):
        if not text:
            return
        with self._lock:
            self._pending = (time.monotonic(), text)
        self._wake.set()

    def close(self, timeout=8):
        """Let the last pending alert finish, then stop."""
        self._stop = True
        self._wake.set()
        self._thread.join(timeout=timeout)

    # -- backends -------------------------------------------------------
    def _start_backend(self):
        if IS_WINDOWS:
            try:
                proc = subprocess.Popen(
                    ['powershell', '-NoProfile', '-NonInteractive', '-Command', _PS_LIVE],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                    text=True, bufsize=1, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))

                def speak(text):
                    proc.stdin.write(text.replace('\n', ' ') + '\n')
                    proc.stdin.flush()
                    proc.stdout.readline()          # blocks until Windows finished speaking

                def shutdown():
                    try:
                        proc.stdin.close()
                        proc.wait(timeout=3)
                    except Exception:
                        proc.kill()
                return speak, shutdown
            except Exception as e:
                print(f"[speech] Windows voice failed to start: {e}")
        try:
            import pyttsx3
            engine = pyttsx3.init()

            def speak(text):
                engine.say(text)
                engine.runAndWait()
            return speak, lambda: None
        except Exception:
            return None, None

    def _run(self):
        speak, shutdown = self._start_backend()
        self.available = speak is not None
        self._ready.set()
        while True:
            self._wake.wait()
            with self._lock:
                item, self._pending = self._pending, None
                self._wake.clear()
            if item is None:
                if self._stop:
                    break
                continue
            queued_at, text = item
            if speak and time.monotonic() - queued_at <= MAX_AGE_S:
                try:
                    speak(text)
                except Exception as e:
                    print(f"[speech] {e}")
                    speak = None
        if shutdown:
            shutdown()


# ---------------------------------------------------------------------
# Narrated video (voice baked into a saved video, for presentations)
# ---------------------------------------------------------------------
def find_ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which('ffmpeg')


def synthesize_to_wavs(texts, out_dir):
    """Render each text to out_dir/NNN.wav (22.05 kHz, 16-bit mono). Returns the paths."""
    paths = [os.path.join(out_dir, f"{i:03d}.wav") for i in range(len(texts))]
    if IS_WINDOWS:
        lines = [
            "Add-Type -AssemblyName System.Speech",
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer",
            f"$s.Rate = {SPEECH_RATE}",
            f"$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo({SAMPLE_RATE}, "
            "[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, "
            "[System.Speech.AudioFormat.AudioChannel]::Mono)",
        ]
        for text, path in zip(texts, paths):
            lines += [f"$s.SetOutputToWaveFile({_ps_quote(path)}, $fmt)", f"$s.Speak({_ps_quote(text)})"]
        lines.append("$s.SetOutputToNull()")
        script = os.path.join(out_dir, 'speak.ps1')
        with open(script, 'w', encoding='utf-8-sig') as f:
            f.write('\n'.join(lines))
        subprocess.run(['powershell', '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass',
                        '-File', script], check=True, capture_output=True)
        return paths
    import pyttsx3  # other systems
    engine = pyttsx3.init()
    for text, path in zip(texts, paths):
        engine.save_to_file(text, path)
    engine.runAndWait()
    return paths


def _read_wav_mono(path):
    with wave.open(path, 'rb') as w:
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        if w.getnchannels() > 1:
            data = data.reshape(-1, w.getnchannels()).mean(axis=1).astype(np.int16)
        rate = w.getframerate()
    if rate != SAMPLE_RATE:  # simple resample, only needed for non-Windows voices
        idx = np.linspace(0, len(data) - 1, int(len(data) * SAMPLE_RATE / rate))
        data = np.interp(idx, np.arange(len(data)), data).astype(np.int16)
    return data


def build_track(events, clips, duration_s):
    """
    Place each spoken clip at its alert time on a silent track. Clips never
    overlap: if one is still playing, the next starts right after it, and is
    skipped if that would be more than MAX_AGE_S late (same rule as live).
    """
    track = np.zeros(int((duration_s + 10) * SAMPLE_RATE), dtype=np.int16)
    free_at = 0
    for (t, _), clip in zip(events, clips):
        start = max(int(t * SAMPLE_RATE), free_at)
        if (start - t * SAMPLE_RATE) / SAMPLE_RATE > MAX_AGE_S:
            continue
        end = min(start + len(clip), len(track))
        track[start:end] = clip[:end - start]
        free_at = end + int(0.15 * SAMPLE_RATE)
    return track[:int(duration_s * SAMPLE_RATE)]


def make_narrated_video(video_path, events, out_path, duration_s, synthesize=synthesize_to_wavs):
    """
    events: [(video_time_s, text), ...]. Writes out_path = video + spoken alerts,
    re-encoded as H.264 so it plays in PowerPoint and any media player.
    Returns out_path, or None if it could not be made.
    """
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        print("[speech] Narrated video skipped: ffmpeg not found. Run:  pip install imageio-ffmpeg")
        return None
    if not events:
        print("[speech] No alerts in this video, so no narrated copy was made.")
        return None
    with tempfile.TemporaryDirectory() as tmp:
        try:
            wavs = synthesize([text for _, text in events], tmp)
        except Exception as e:
            print(f"[speech] Narrated video skipped: could not synthesize speech ({e})")
            return None
        track = build_track(events, [_read_wav_mono(p) for p in wavs], duration_s)
        audio = os.path.join(tmp, 'alerts.wav')
        with wave.open(audio, 'wb') as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(track.tobytes())
        result = subprocess.run(
            [ffmpeg, '-y', '-loglevel', 'error', '-i', video_path, '-i', audio,
             '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'libx264', '-preset', 'veryfast',
             '-crf', '23', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '128k',
             '-shortest', out_path],
            capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[speech] ffmpeg failed: {result.stderr.strip()[-300:]}")
            return None
    return out_path
