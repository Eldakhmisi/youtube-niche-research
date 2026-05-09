"""
YouTube niche research pipeline.

Given a list of channels, fetches each channel's top-N videos by view count,
pulls transcripts, runs each through an LLM for structured extraction, and
writes one markdown report comparing patterns across channels.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
import yt_dlp
from dotenv import load_dotenv
from openai import OpenAI
from yt_dlp.cookies import load_cookies as ytdlp_load_cookies
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)

load_dotenv()

CHANNELS = [
    ("@AlexTheAnalyst", "Alex The Analyst"),
    ("@LukeBarousse", "Luke Barousse"),
    ("@KenJee_DS", "Ken Jee"),
    ("@Thuvu5", "Thu Vu data analytics"),
    ("@TinaHuang1", "Tina Huang"),
]
TOP_N = 5
MODEL = "gpt-4o-mini"
TRANSCRIPT_CHAR_LIMIT = 18000  # ~4.5k tokens, plenty for analysis


# ------------------------------------------------------------------ video list


SAMPLE_SIZE = 100  # most-recent N videos to consider; sorted by views to pick top N


def fetch_top_videos(channel_handle: str, n: int = 5) -> list[dict]:
    """Return the n most-viewed videos from the most-recent SAMPLE_SIZE uploads.

    Note: YouTube no longer exposes a "popular sort" URL, and `extract_flat` does
    not return view counts. So we fetch the most recent SAMPLE_SIZE videos with
    full metadata (~1 sec each) and sort client-side. This captures what's
    currently performing rather than all-time greats.
    """
    url = f"https://www.youtube.com/{channel_handle}/videos"
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "extract_flat": False,
        "playlistend": SAMPLE_SIZE,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    entries = info.get("entries", []) or []
    entries = [e for e in entries if e and e.get("view_count")]
    entries.sort(key=lambda e: e.get("view_count", 0), reverse=True)
    top = entries[:n]
    return [
        {
            "id": e.get("id"),
            "title": e.get("title"),
            "view_count": e.get("view_count"),
            "duration": e.get("duration"),
            "upload_date": e.get("upload_date"),
            "url": f"https://www.youtube.com/watch?v={e.get('id')}",
        }
        for e in top
    ]


# -------------------------------------------------------------------- transcripts


def make_cookied_session(browser: str | None) -> requests.Session | None:
    """Build a requests Session with YouTube cookies extracted from the browser.

    Pass browser=None to skip cookies entirely (no-cookie session).
    """
    if not browser:
        return None
    cj = ytdlp_load_cookies(None, (browser, None, None, None), None)
    s = requests.Session()
    s.cookies = cj
    return s


def fetch_transcript(video_id: str, session: requests.Session | None = None) -> str | None:
    try:
        api = YouTubeTranscriptApi(http_client=session) if session else YouTubeTranscriptApi()
        fetched = api.fetch(video_id, languages=["en"])
        text = " ".join(snippet.text for snippet in fetched)
        return text[:TRANSCRIPT_CHAR_LIMIT]
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable):
        return None
    except Exception as e:
        # Truncate noisy error text for log readability
        msg = str(e).splitlines()[0][:160]
        print(f"    transcript error for {video_id}: {msg}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------- analysis

EXTRACTION_SYSTEM = """You analyze YouTube videos for a creator researching a niche.
Given a video's title and transcript, return a JSON object with these keys:

- hook_first_15s: what the creator says/does in the first ~15 seconds (string, 1-2 sentences)
- hook_pattern: one short label for the hook type (e.g. "promise", "stat", "personal story", "cold open", "controversy", "list tease")
- structure: 3-7 short bullets describing the video's flow (array of strings)
- monetization_signals: any mentions of products, courses, sponsors, lead magnets, affiliate links, email list, Patreon (array of strings; empty if none)
- key_claims: the 2-4 most useful or repeated claims/lessons (array of strings)
- topic_tags: 3-6 short tags describing what the video is about (array of strings)

Return ONLY valid JSON, no prose."""


def analyze_video(client: OpenAI, video: dict, transcript: str) -> dict:
    user = f"TITLE: {video['title']}\n\nTRANSCRIPT:\n{transcript}"
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    content = resp.choices[0].message.content
    return json.loads(content)


PATTERNS_SYSTEM = """You are synthesizing cross-channel patterns for a creator researching a YouTube niche.
You will receive structured JSON analyses for the top videos of multiple channels.

Return a markdown report with these sections:

## Cross-channel patterns
What shows up in 3+ channels (signal). Cover: hook patterns, video structures, monetization moves, recurring topics, title formulas.

## Channel-specific edges
For each channel, the 1-2 things unique to that channel that others don't do.

## What this means for a new entrant
3-5 concrete recommendations for someone launching an Arabic-primary, English-mirror data-analyst channel aimed at MENA career-switchers, written as direct advice.

Be specific. Quote titles when useful. Skip filler."""


def synthesize_patterns(client: OpenAI, all_data: dict) -> str:
    payload = json.dumps(all_data, indent=2)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": PATTERNS_SYSTEM},
            {"role": "user", "content": payload},
        ],
        temperature=0.3,
    )
    return resp.choices[0].message.content


# --------------------------------------------------------------------- markdown


def build_markdown(all_data: dict, patterns_md: str) -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    lines = [f"# YouTube niche research — data analyst channels", f"_Generated {today}_", ""]
    lines.append(patterns_md)
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("# Per-channel breakdown")
    lines.append("")

    for channel_name, channel_data in all_data.items():
        lines.append(f"## {channel_name}")
        lines.append("")
        for v in channel_data["videos"]:
            views = f"{v['video']['view_count']:,}" if v['video'].get('view_count') else "?"
            lines.append(f"### [{v['video']['title']}]({v['video']['url']})")
            lines.append(f"_{views} views_")
            lines.append("")
            if v.get("analysis"):
                a = v["analysis"]
                lines.append(f"**Hook ({a.get('hook_pattern', '?')}):** {a.get('hook_first_15s', '')}")
                lines.append("")
                lines.append("**Structure:**")
                for s in a.get("structure", []):
                    lines.append(f"- {s}")
                lines.append("")
                if a.get("monetization_signals"):
                    lines.append("**Monetization:**")
                    for m in a["monetization_signals"]:
                        lines.append(f"- {m}")
                    lines.append("")
                if a.get("key_claims"):
                    lines.append("**Key claims:**")
                    for k in a["key_claims"]:
                        lines.append(f"- {k}")
                    lines.append("")
                if a.get("topic_tags"):
                    lines.append(f"**Tags:** {', '.join(a['topic_tags'])}")
                    lines.append("")
            else:
                lines.append("_(no transcript available)_")
                lines.append("")
        lines.append("")

    return "\n".join(lines)


# --------------------------------------------------------------------- main


def main(out_path: Path, only: list[str] | None, browser: str | None, transcript_delay: float):
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set. Add it to .env", file=sys.stderr)
        sys.exit(1)

    client = OpenAI()

    # If --only is set, load existing JSON and only re-fetch the named channels.
    raw_path = out_path.with_suffix(".json")
    if only and raw_path.exists():
        all_data = json.loads(raw_path.read_text(encoding="utf-8"))
        print(f"Loaded existing data ({len(all_data)} channels) from {raw_path}")
    else:
        all_data = {}

    channels_to_run = [
        (h, n) for (h, n) in CHANNELS if (not only or h in only)
    ]
    if only:
        print(f"Running only: {[h for h, _ in channels_to_run]}")

    session = make_cookied_session(browser)
    if session:
        yt_cookie_count = sum(1 for c in session.cookies if "youtube" in (c.domain or ""))
        print(f"Loaded {yt_cookie_count} YouTube cookies from {browser}")

    for handle, name in channels_to_run:
        print(f"\n[{name}] fetching top {TOP_N} videos…")
        try:
            top = fetch_top_videos(handle, n=TOP_N)
        except Exception as e:
            print(f"  could not fetch {handle}: {e}", file=sys.stderr)
            continue

        if not top:
            print(f"  no videos returned for {handle}", file=sys.stderr)
            continue

        videos_data = []
        for v in top:
            views = v.get("view_count") or 0
            print(f"  - {v['title'][:70]} ({views:,} views)")
            transcript = fetch_transcript(v["id"], session=session)
            analysis = None
            if transcript:
                try:
                    analysis = analyze_video(client, v, transcript)
                    time.sleep(0.3)
                except Exception as e:
                    print(f"    analysis error: {e}", file=sys.stderr)
            else:
                print(f"    (no transcript)")
            videos_data.append({"video": v, "analysis": analysis})
            if transcript_delay:
                time.sleep(transcript_delay)

        all_data[name] = {"handle": handle, "videos": videos_data}

    if not all_data:
        print("no data collected, exiting", file=sys.stderr)
        sys.exit(1)

    print("\nSynthesizing cross-channel patterns…")
    patterns_md = synthesize_patterns(client, all_data)

    md = build_markdown(all_data, patterns_md)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(f"\n✅ Wrote {out_path}")

    raw_path.write_text(json.dumps(all_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"   Raw data: {raw_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    default_out = Path.home() / "Desktop/AI OS/AIS-OS/references/youtube-channel-research.md"
    parser.add_argument("out", nargs="?", default=str(default_out), help="Output markdown path")
    parser.add_argument("--only", default="", help="Comma-separated channel handles to re-run (merges into existing JSON)")
    parser.add_argument("--browser", default="chrome", help="Browser to extract YouTube cookies from (chrome, safari, firefox, edge, brave). Empty string to disable cookies.")
    parser.add_argument("--delay", type=float, default=2.0, help="Seconds to sleep between transcript fetches (default 2)")
    args = parser.parse_args()

    only_list = [h.strip() for h in args.only.split(",") if h.strip()] if args.only else None
    browser = args.browser if args.browser else None
    main(Path(args.out), only=only_list, browser=browser, transcript_delay=args.delay)
