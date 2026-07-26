# Hermes Agent — Railway Template

Deploy [Hermes Agent](https://github.com/NousResearch/hermes-agent) on [Railway](https://railway.app) with a web-based admin dashboard for configuration, gateway management, and user pairing.

[![Deploy on Railway](https://railway.com/button.svg)](https://railway.com/deploy/hermes-agent-ai?referralCode=QXdhdr&utm_medium=integration&utm_source=template&utm_campaign=generic)

> Hermes Agent is an autonomous AI agent by [Nous Research](https://nousresearch.com/) that lives on your server, connects to your messaging channels (Telegram, Discord, Slack, etc.), and gets more capable the longer it runs.

<!-- TODO: Add dashboard screenshot -->
<!-- ![Dashboard](docs/dashboard.png) -->

## Features

- **Admin Dashboard** — dark-themed UI to configure providers, channels, tools, and manage the gateway
- **One-Page Setup** — provider dropdown, checkbox-based channel/tool toggles — no config files to edit
- **iMessage** — connect [Photon](https://photon.codes/) with one click (no Mac required) or bridge your own [BlueBubbles](https://bluebubbles.app/) server; inbound webhooks are relayed automatically
- **Gateway Management** — start, stop, restart the Hermes gateway from the browser
- **Live Status** — stat cards for gateway state, uptime, model, and pending pairing requests
- **Live Logs** — streaming gateway log viewer
- **User Pairing** — approve or deny users who message your bot, revoke access anytime
- **Basic Auth** — password-protected admin panel
- **Reset Config** — one-click reset to start fresh
- **Backup & Restore** — download a full snapshot (config, credentials, chat history, memories, skills) as a zip, and restore it — including into a fresh project — to clone a deployment. Not encrypted; a safety snapshot is taken automatically before every restore.

## Getting Started

The easiest way to get started:

### 1. Get an LLM Provider Key (free)

1. Register for free at [OpenRouter](https://openrouter.ai/)
2. Create an API key from your [OpenRouter dashboard](https://openrouter.ai/keys)
3. Pick a free model from the [model list sorted by price](https://openrouter.ai/models?order=pricing-low-to-high) (e.g. `google/gemma-3-1b-it:free`, `meta-llama/llama-3.1-8b-instruct:free`)

### 2. Set Up a Telegram Bot (fastest channel)

Hermes Agent interacts entirely through messaging channels — there is no chat UI like ChatGPT. Telegram is the quickest to set up:

1. Open Telegram and message [@BotFather](https://t.me/BotFather)
2. Send `/newbot`, follow the prompts, and copy the **Bot Token**
3. Send a message to your new bot — it will appear as a pairing request in the admin dashboard
4. To find your Telegram user ID, message [@userinfobot](https://t.me/userinfobot)

### 3. Deploy to Railway

1. Click the **Deploy on Railway** button above
2. Set the `ADMIN_PASSWORD` environment variable (or a random one will be generated and printed to deploy logs)
3. Attach a **volume** mounted at `/data` (persists config across redeploys)
4. Open your app URL — log in with username `admin` and your password

### 4. Configure in the Admin Dashboard

1. **LLM Provider** — select OpenRouter from the dropdown, paste your API key, enter the model name
2. **Messaging Channel** — check Telegram, paste the Bot Token from BotFather
3. Click **Save & Start** — the gateway will start and your bot goes live

### 5. Start Chatting

Message your Telegram bot. If you're a new user, a pairing request will appear in the admin dashboard under **Users** — click **Approve**, and you're in.

<!-- TODO: Add Telegram chat screenshot -->
<!-- ![Telegram Example](docs/telegram-example.png) -->

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8080` | Web server port (set automatically by Railway) |
| `ADMIN_USERNAME` | `admin` | Basic auth username |
| `ADMIN_PASSWORD` | *(auto-generated)* | Basic auth password — if unset, a random password is printed to logs |
| `HERMES_REF` | *(pinned in Dockerfile)* | Hermes Agent version to install (any upstream git tag/branch). Set this to override the Dockerfile default without editing code — see [Updating Hermes](#updating-hermes). |

All other configuration (LLM provider, model, channels, tools) is managed through the admin dashboard.

## Supported Providers

OpenRouter, DeepSeek, DashScope, GLM / Z.AI, Kimi, MiniMax, HuggingFace

## Supported Channels

Telegram, Discord, Slack, iMessage, WhatsApp, Email, Mattermost, Matrix

## iMessage

Hermes ships two independent iMessage integrations, and this template wires up both.
They sit in the **Messaging Channels** section of the setup page.

### Photon — recommended, no Mac required

[Photon](https://photon.codes/) is a managed service that owns the iMessage line for you.
The gateway holds an **outbound** gRPC stream to it, so there is nothing to expose and
nothing else to run — the only iMessage path that works on a stock Railway deploy.

1. Tick **iMessage · Photon**, enter your own phone number in E.164 (`+15551234567`)
2. Click **Connect Photon** — a tab opens where you approve a short code
3. The dashboard then provisions everything for you: a Photon project, its API secret,
   your number registered as a Spectrum user, and the allowlist + home channel
4. When it finishes, the panel shows **the number to text** to reach your agent

Photon's free tier uses a shared iMessage line pool, so you can start without a paid plan.
Credentials land in `/data/.hermes/.env` (`PHOTON_PROJECT_ID`, `PHOTON_PROJECT_SECRET`)
and `/data/.hermes/auth.json`, exactly where `hermes photon status` looks for them.

> The Node sidecar that runs Photon's `spectrum-ts` SDK is pre-installed into the image at
> build time (`npm ci`). If you see a warning that it's missing, rebuild from the current
> `Dockerfile` and redeploy — `/opt` is baked into the image, so it can't be installed at runtime.

### BlueBubbles — bring your own Mac

[BlueBubbles](https://bluebubbles.app/) bridges iMessage from an always-on Mac signed into
Messages.app. Use it if you want your own number rather than a Photon line.

1. Install BlueBubbles Server on the Mac, sign in, and make it reachable from the internet
   (Ngrok, a Cloudflare tunnel, or dynamic DNS) — copy the **Server URL** and **Password**
   from Settings → API
2. Tick **iMessage · BlueBubbles**, paste both, and **Save & Start**
3. The panel's **Test connection** button confirms the server is reachable and that the
   inbound webhook is registered

**About the webhook.** BlueBubbles delivers incoming messages by POSTing to a URL you
register with it, and the gateway's own listener binds loopback-only — unreachable from
your Mac, and Railway publishes just one port. So this template exposes a public relay at
`POST /<your-app>/bluebubbles-webhook` that verifies the BlueBubbles password (the same
shared secret the gateway checks — constant-time compared at the edge) and forwards the
body to the gateway internally. Saving the config registers that URL with your server
automatically, prunes registrations left behind by an old domain or a rotated password,
and leaves the gateway's own loopback entry alone. Use **Register webhook** to re-sync by
hand. Set `BLUEBUBBLES_PUBLIC_URL` only if a proxy in front of this app rewrites `Host`.

Either channel routes unknown senders into the **Users** tab as pairing requests, same as
Telegram or Discord.

### Notes

- **Configure iMessage from this setup page, not the Hermes dashboard.** Hermes' own
  Channels page lists both, but its Photon card just tells you to run `hermes photon setup`
  — a TTY wizard that can't run in this container. Toggling a channel off there also writes
  a sticky `enabled: false` that would override the credentials you saved here; saving from
  this page clears it again for the channel you enabled.
- **Reconnecting Photon rotates the project secret.** Photon only reveals a secret once, so
  connecting mints a fresh one and invalidates the previous value. Disconnect first if you
  need to, and don't share one Photon project between two deployments.

## Supported Tool Integrations

Parallel (search), Firecrawl (scraping), Tavily (search), FAL (image gen), Browserbase, GitHub, OpenAI Voice (Whisper/TTS), Honcho (memory)

## Architecture

```
Railway Container
├── Python Admin Server (Starlette + Uvicorn)
│   ├── /                     — Admin dashboard (Basic Auth)
│   ├── /health               — Health check (no auth)
│   ├── /bluebubbles-webhook  — Inbound iMessage relay (password-gated, no cookie)
│   └── /api/*                — Config, status, logs, gateway, pairing
└── hermes gateway            — Managed as async subprocess
    ├── BlueBubbles listener  — 127.0.0.1:8645 (fed by the relay above)
    └── Photon sidecar        — 127.0.0.1:8789, outbound gRPC to Photon
```

The admin server runs on `$PORT` and manages the Hermes gateway as a child process. Config is stored in `/data/.hermes/.env` and `/data/.hermes/config.yaml`. Gateway stdout/stderr is captured into a ring buffer and streamed to the Logs panel.

## Running Locally

```bash
docker build -t hermes-agent .
docker run --rm -it -p 8080:8080 -e PORT=8080 -e ADMIN_PASSWORD=changeme -v hermes-data:/data hermes-agent
```

Open `http://localhost:8080` and log in with `admin` / `changeme`.

## Updating Hermes

This template pins a specific Hermes Agent release in the `Dockerfile` (`ARG HERMES_REF`, currently `v2026.7.1`). To upgrade:

- **Recommended:** set a `HERMES_REF` service variable in Railway to any upstream [release tag](https://github.com/NousResearch/hermes-agent/releases) (e.g. `v2026.7.1`), then redeploy. It's passed in as a Docker build arg and overrides the Dockerfile default — no code change needed.
- **Or** bump `ARG HERMES_REF` in the `Dockerfile` and redeploy.

The "Update" button inside the Hermes dashboard is a **no-op on Railway** (it detects a container install and refuses) — the image is immutable, so a runtime self-update wouldn't survive a redeploy. Bump `HERMES_REF` and redeploy instead. When jumping releases, re-check that the Dockerfile's install extras still match upstream's `pyproject.toml`.

## Credits

- [Hermes Agent](https://github.com/NousResearch/hermes-agent) by [Nous Research](https://nousresearch.com/)
- UI inspired by [OpenClaw](https://github.com/praveen-ks-2001/openclaw-railway) admin template
