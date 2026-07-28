# Caption & Hook Generator

Turns a short vertical video clip into ready-to-post social copy - on-screen hook, alternate
hook, caption, and hashtags for Instagram, TikTok and X, plus native Dutch rewrites - by
transcribing speech where it exists or reading the visuals where it doesn't.

## Approach

For each clip the tool validates the file, extracts its audio, and transcribes it with
OpenAI's Whisper API. Whether a clip "has speech" is decided from Whisper's own per-segment
`no_speech_prob` rather than just "did we get text back", since a clip can have an audio
track (music, ambience) without having speech on it - both sample clips prove this: they
both have an audio stream, only one has actual speech. If no speech is found, the tool
samples a handful of frames instead and asks GPT-4o-mini's vision model to describe what's
happening, and that description - never invented dialogue - becomes the basis for the copy.
A single GPT-4o-mini call, constrained with OpenAI's Structured Outputs against a Pydantic
schema, then generates the platform copy and its Dutch rewrite in one shot, which is what
guarantees the output JSON is always schema-valid rather than something to hope for.

## Architecture

**OpenAI end-to-end**: Whisper (`whisper-1`) for transcription and speech detection,
GPT-4o-mini vision for silent clips, GPT-4o-mini with Structured Outputs for all copywriting.

This was chosen over a split OpenAI/Anthropic setup or a fully "free-tier" (Groq) stack
mainly for one reason: OpenAI's Structured Outputs (`response_format=<PydanticModel>`)
constrains the model's output at the API level, so malformed JSON or a missing field isn't a
failure mode that can happen - as opposed to prompting for JSON and validating/retrying after
the fact. Given the brief's explicit interest in "structured, predictable output another
system could consume," that outweighed Anthropic's arguably-comparable copywriting quality.
Cost is a non-issue either way - the full three-call pipeline runs to well under a cent per
clip on GPT-4o-mini pricing. The transcription call is isolated behind `transcriber.py`
specifically so it can be swapped to Groq's hosted Whisper (same model weights, effectively
free at this volume) later without touching any other module - see `config.py`.

Local/GPU-based models were deliberately not used anywhere: everything here runs on a
low-power laptop with no GPU, since all three AI calls are hosted APIs and the only local
work is `ffmpeg` extracting audio/frames.

## Repository layout

```
caption_generator/
    main.py              CLI entry point and pipeline orchestration
    config.py             env vars, model names, tunable constants
    schemas.py             Pydantic models - the output contract
    video_processor.py     ffmpeg/ffprobe: validation, audio extraction, frame sampling
    transcriber.py          Whisper call + speech/no-speech detection + subtitle grouping
    vision.py               frame -> visual description (silent clips only)
    prompts.py               copywriting prompt templates and voice/tone rules
    llm.py                   Structured Outputs copywriting call
    exceptions.py            named exception types per failure mode
    utils.py                 logging, subprocess helper
    requirements.txt
    .env.example
    sample_output/          real output from running against the provided clips
```

## Installation

1. Python 3.10+
2. **ffmpeg** must be installed and on your PATH (not a pip package - a system binary).
   - macOS: `brew install ffmpeg`
   - Ubuntu/Debian: `sudo apt install ffmpeg`
   - Windows: https://ffmpeg.org/download.html
3. From the `caption_generator/` folder:
   ```
   python -m venv .venv
   source .venv/bin/activate   # .venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

## Environment variables

Copy `.env.example` to `.env` and set your key:

```
OPENAI_API_KEY=sk-...
```

`.env` is already in `.gitignore` and will never be committed.

## How to run

```
python main.py <path to a video file, or a folder of video files> [--output result.json]
```

A single file produces `{"clip_1": {...}}`; a folder produces `{"clip_1": ..., "clip_2": ...}`
in sorted filename order - one code path handles both, matching the shape of the provided
example JSON either way. If `--output` isn't given, the result is written to
`<input name>_output.json` in the current directory.

## Example commands

Run from inside `caption_generator/`. The two provided sample clips sit one level up, in the
parent folder they were delivered in. These are the exact commands that produced the files
checked in under `sample_output/`:

```
python main.py "../Video 1 - with talking (Clubsweethearts).mp4" --output sample_output/clip_1_output.json
python main.py "../Video 2 - no talking (Beauty and the Senior).mp4" --output sample_output/clip_2_output.json
```

Both clips can also be processed in one run by pointing at that folder directly - this produces
a single file with both clips under `clip_1`/`clip_2` keys, rather than the two separate files
above:

```
python main.py ../ --output sample_output/batch_output.json
```

(the two PDFs and the example JSON alongside the clips aren't video files, so they're
skipped automatically rather than producing bogus "unsupported format" errors for them.)

## Example output

Real output from running against both provided sample clips is in `sample_output/`.

## Output schema

Each clip's result mirrors the structure of the brief's `example_output.json` -
`has_speech`, `platforms.{instagram,tiktok,x}.{on_screen_hook, on_screen_hook_alt, caption,
hashtags}`, `subtitles` (speech clips) or `subtitles_note` (silent clips), and `dutch`. Two
deliberate deviations, both additive/non-breaking:

- **`content_summary`** is included on every clip: the transcript text or visual description
  that grounded the copy. Not in the original example - added for transparency, so it's
  possible to see *why* the model wrote what it wrote without re-running anything.
- **`dutch.<platform>`** includes `on_screen_hook_alt` (the brief requires an alternate hook
  "per platform," and Dutch is a rewrite of "the hooks and captions" - plural hooks reads as
  covering both) but excludes `hashtags`. The example JSON only shows a Dutch block for
  Instagram on one clip, with hashtags but no alt hook - inconsistent with itself across the
  two clips, so I standardized on: every clip, all three platforms, hooks + alt hooks +
  caption in Dutch, no hashtags (hashtags are discovery tags tied to what people actually
  search in that language/market; guessing Dutch hashtag equivalents without genre/market
  context felt riskier than just leaving the English hashtags as the single source of truth).

A clip that fails at any stage produces `{"source_file": ..., "error": "..."}` under its key
instead of a `ClipResult` - check for the `error` key to tell success from failure
programmatically.

## Assumptions made (brief invited "make a sensible choice and note it")

- **Subtitle line length**: no hard rule was given, so lines are grouped from Whisper's
  word-level timestamps at a 7-word / 3.5-second cap per line, whichever comes first.
- **Supported formats**: `.mp4 .mov .webm .mkv .m4v` - common vertical-video containers
  ffmpeg can decode. Anything else is rejected with a clear "unsupported format" error.
- **`start`/`end` as strings** (`"2.1"`, not `2.1`): matches the example JSON's literal format
  exactly, on the theory that matching the client's own example beats using a "more correct"
  native type.
- **Single file or folder, same output shape**: the brief's "Input" section says "a video
  file" (singular) but the example JSON is keyed `clip_1`/`clip_2` (plural). The CLI accepts
  either; the output always uses `clip_N` keys either way.

## Error handling

Every stage - file validation, audio/frame extraction, transcription, vision, copywriting -
is wrapped in `main.py`'s `process_clip()`, which never raises. A bad clip (missing file,
corrupt/unreadable video, unsupported format, or any API call failing) produces a clear
`{"error": "..."}` entry for that specific clip; it does not stop the rest of a batch, and it
never surfaces as an unhandled stack trace. Named exception types
(`InvalidVideoError`, `TranscriptionError`, `VisionAnalysisError`, `CopyGenerationError`, ...)
keep each failure mode's message specific to what actually went wrong.

## Real-corpus findings (from actually running this against the API)

Everything above was written before a real API key was available. Once one was, running the
pipeline against both real sample clips - kept to the minimum number of runs needed to find and
verify each fix, not open-ended iteration - surfaced three real bugs, none of them visible just
from reading the code:

- **The English `platforms` copy came back entirely in Dutch** on the first real run of Clip 1,
  word-for-word identical to the `dutch` block. The system prompt never explicitly stated the
  primary fields had to be in English - only implied it via the two English calibration
  examples - and with a very sparse transcript (four exclaimed words), the model defaulted the
  whole generation to Dutch instead. Fixed with an explicit "write `platforms.*` in English,
  regardless of the source language" instruction in `prompts.py`.
- **Clip 2's first copywriting call hard-failed**, burning the model's full 16,384-token output
  ceiling on what looks like a degenerate repetition loop, and returning "Could not parse
  response content as the length limit was reached." `llm.py`'s copywriting call had no
  `max_completion_tokens` cap at all - unlike `vision.py`, which already caps its own call at
  300. Added a 2,000-token cap: real output here is a few hundred tokens, so this now fails fast
  and cheap instead of burning the ceiling if it ever loops again.
- **The one that actually mattered**: Clip 2 (the documented no-speech clip) initially came back
  as `has_speech: true`, with a hallucinated transcript ("Watch the whole video... thank you").
  The tell was in the raw word timestamps: two words spanning 8.8 seconds, then eight words
  crammed into the next 0.1 seconds - a known Whisper failure mode where forced word-alignment
  breaks down on non-speech audio while the model still hallucinates plausible boilerplate text
  (likely learned from YouTube outro audio in its training data), with `no_speech_prob` coming
  back deceptively low for this specific failure. `transcriber.py` now has a structural sanity
  check (`_looks_like_hallucinated_transcript`) that flags a transcript as unreliable when a
  large share of words have an implausibly short implied duration, and falls back to
  `has_speech=False` regardless of what `no_speech_prob` says - verified against both the real
  failure pattern and a healthy transcript's word timing before it was trusted.

A fourth, lower-stakes issue also showed up: both clips' captions independently converged on
the exact closing lines from the two calibration examples baked into the system prompt ("Same
time tomorrow?", "Clip below.") despite an explicit "do not copy, reuse, or reference these"
instruction - a generic instruction wasn't concrete enough to stop it. Tightened to name the
specific phrases not to reuse and to require a closing line invented fresh from each clip's own
content; confirmed fixed by re-running both clips and diffing the closers.

This is exactly the class of bug the brief's two sample clips seem designed to surface, and none
of the four would have been caught without actually spending real API calls against real
audio/video content rather than just reading the code.

## Limitations, and what I'd improve with more time

**If I had one more hour**, I'd add a small automated eval: a script that runs a handful of
test clips through the pipeline and scores the output against the reference examples' style
rules (does the hook open a loop rather than state a fact, does the X caption stand alone
without leaning on subtitles, are hashtags free of generic ones, does a closing line repeat
verbatim across clips - a real instance of exactly that last one is what caught the calibration-
example-copying bug in "Real-corpus findings" above, by manually diffing two runs, which an eval
would catch automatically) so prompt changes could be validated objectively instead of by
reading a couple of runs and eyeballing them.

Other things worth knowing:

- **No automated test suite.** Every module was manually verified against both real sample
  clips and a set of induced failure cases (missing file, truncated/corrupt file, file with no
  audio stream, mocked API responses) during development, but there's no `pytest` suite
  checked in - given the brief's "small and clean" framing and time budget, manual
  verification felt like the right trade-off for a trial this size, but it's the first thing
  I'd formalize.
- **Vision analysis samples 4 still frames**, not the full motion of the clip - it can miss
  anything that only reads correctly in motion (a gesture, a transition, on-screen text that
  appears mid-clip).
- **No retry/backoff on transient API failures** - a rate limit or network blip fails that
  clip cleanly rather than crashing, but doesn't retry. For a production version I'd add a
  short exponential backoff before giving up.
- **Dutch quality depends on the same model as the English copy** rather than a
  Dutch-specialist pass - it reads natively in testing, but a native Dutch speaker reviewing a
  larger sample would be the real check before shipping this to Dutch audiences.
