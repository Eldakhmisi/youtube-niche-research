# YouTube niche research

Pipeline that pulls the top videos from a list of YouTube channels, fetches
their transcripts, runs each through an LLM for structured extraction (hook,
structure, monetization signals, key claims, topic tags), and writes one
markdown report comparing patterns across channels.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then add your OPENAI_API_KEY
```

## Run

```bash
./venv/bin/python research.py
# or specify output path:
./venv/bin/python research.py path/to/output.md
```

Default output: `~/Desktop/AI OS/AIS-OS/references/youtube-channel-research.md`

## Configure

Edit `CHANNELS`, `TOP_N`, and `MODEL` at the top of `research.py`.

## A note on "top videos"

YouTube no longer exposes a popular-sort URL, and yt-dlp's flat extract doesn't
return view counts. So the script fetches the channel's most-recent
`SAMPLE_SIZE` (default 100) uploads with full metadata, then sorts by views and
takes the top N. This means **"top" = top from recent uploads, not all-time**,
which is usually what you want for niche research (current algorithm signal
beats 2021 hits). Bump `SAMPLE_SIZE` higher in `research.py` if you want a
deeper sample, at the cost of ~1 sec per extra video per channel.

For true all-time top, switch the fetcher to YouTube Data API v3 (free tier,
needs a Google Cloud key).

## Cost

~$0.02 - $0.05 per run on `gpt-4o-mini` (5 channels × 5 videos + 1 synthesis call).

## Reuse

To research a different niche, just change the `CHANNELS` list and re-run.
The `PATTERNS_SYSTEM` prompt also has audience-specific advice baked in
(MENA career-switchers, Arabic+English) — edit if researching for a different
audience.
