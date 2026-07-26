FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Which hermes-agent revision to install. Accepts any git ref the upstream
# repo publishes — a release tag (recommended for reproducibility) or a
# branch name (`main`) for bleeding edge.
#
# To bump: check https://github.com/NousResearch/hermes-agent/releases for the
# newest tag (format `vYYYY.M.D`, optionally with a `.PATCH` suffix, e.g.
# `v2026.5.29.2`) and update the default below. Use `main` only if you accept
# that every rebuild can pull arbitrary new upstream commits.
ARG HERMES_REF=v2026.7.1

# tini = tiny init that we run as PID 1. Without it, hermes's grandchild
# processes (MCP stdio servers, git, bun, browser daemons spawned by tools)
# reparent to PID 1 when their parents exit and pile up as zombies. After
# weeks of uptime that exhausts the kernel's PID table → "fork: cannot
# allocate memory" and the container dies. tini reaps zombies in the
# background and forwards SIGTERM/SIGINT to our entrypoint so Railway's
# stop signal still triggers our graceful shutdown. Standard container init
# (same as Docker's `--init` flag and Kubernetes' pause container).
#
# Node.js is required only at build time to compile the Hermes React dashboard.
# We strip the source + apt lists afterwards to keep the image lean.
#
# lsof + procps exist for the Photon iMessage sidecar. Its adapter reclaims its
# loopback port from an orphaned sidecar before spawning a new one, and it does
# that by shelling out to `lsof -ti tcp:<port>` and `ps -p <pid> -o command=`
# to confirm the squatter really is a sidecar before signalling it. Both calls
# swallow OSError and return "nothing found", so on a slim image without them
# the reaper silently no-ops and every reconnect after a hard gateway kill dies
# on EADDRINUSE — iMessage stays down until the container is redeployed.
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates git tini lsof procps && \
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/*

# Install hermes-agent (provides the `hermes` CLI) and pre-build its React
# dashboard so `hermes dashboard` has nothing to build at runtime.
#
# [all] in v2026.6.5 no longer pulls in [dev]; messaging platforms, TTS, and
# other heavy backends are lazy-installed by hermes at first use. We pre-install
# the ones this template actually uses so first-message latency is instant.
# `vision` (Pillow) is a soft-dep that is NOT in [all] and is otherwise
# lazy-installed at first image use: without it hermes can't downscale an
# oversized image (>5 MB / >8000px), which then bakes into immutable history
# and bricks the session on Anthropic's non-retryable 400. We bake it in.
# When bumping HERMES_REF, re-check hermes-agent's pyproject.toml [all] and
# the extras below against the new release's pyproject.toml.
RUN git clone --depth 1 --branch ${HERMES_REF} https://github.com/NousResearch/hermes-agent.git /opt/hermes-agent && \
    cd /opt/hermes-agent && \
    uv pip install --system --no-cache -e ".[all,messaging,tts-premium,honcho,bedrock,anthropic,edge-tts,hindsight,vision]" && \
    cd /opt/hermes-agent/web && \
    npm install --silent && \
    npm run build && \
    cd /opt/hermes-agent/ui-tui && \
    npm install --silent --no-fund --no-audit --progress=false && \
    npm run build && \
    cd /opt/hermes-agent/plugins/platforms/photon/sidecar && \
    npm ci --silent --no-fund --no-audit --progress=false && \
    rm -rf /opt/hermes-agent/web /opt/hermes-agent/.git /root/.npm

# Why pre-install the Photon sidecar (iMessage):
# - The bundled `photon-platform` plugin is how this template does iMessage
#   without a Mac. Its adapter bridges to Photon's Spectrum cloud through a
#   Node process running the TypeScript-only `spectrum-ts` SDK, and its
#   check_requirements() hard-fails when `sidecar/node_modules` is missing —
#   so without this step the channel simply never connects.
# - `npm ci` (not `npm install`) installs the committed lockfile verbatim.
#   `spectrum-ts` is pinned to an exact version upstream precisely because the
#   SDK ships breaking majors; a floating resolve could brick the channel on a
#   rebuild. `ci` also runs the plugin's `postinstall` patch, which rewrites
#   the SDK's compiled iMessage inbound mappers so a bubble carrying both text
#   and an attachment keeps its typed text — it fails loudly if the upstream
#   build no longer matches, surfacing here at build time rather than as
#   silently dropped message text at runtime.
# - Runtime installation is not an option: /opt is baked into the image, so
#   anything installed there at runtime is lost on the next redeploy.
# - Node itself must therefore survive into the final image (the adapter
#   spawns `node sidecar/index.mjs`) — which it does; only the npm cache and
#   the React dashboard sources are stripped above.
# - When bumping HERMES_REF, re-check plugins/platforms/photon/sidecar/
#   package.json still exists at that path.

# Why pre-build ui-tui (and why we don't delete it after):
# - The dashboard's embedded Chat tab spawns `node ui-tui/dist/entry.js`
#   on every WebSocket connect to /api/pty.
# - Without HERMES_TUI_DIR, hermes's _make_tui_argv falls through to the
#   npm install + build path (since git-editable installs don't have the
#   bundled tui_dist/ that PyPI wheels include), adding 30-60s to the
#   first chat-open and blocking the asyncio event loop.
# - Pre-building at image time surfaces build failures here rather than
#   at user request time, and makes first-chat-open instant.
# - We keep ui-tui/ entirely (node_modules + dist + src) so HERMES_TUI_DIR
#   can point at it (see below).

# Stamp the CODE-SCOPED install method next to the running package. hermes'
# detect_install_method() reads <install-tree>/.install_method FIRST (priority 1,
# authoritative) — before the home-scoped $HERMES_HOME/.install_method that
# start.sh writes (priority 2, honored only when is_container() is true). The
# install tree for our editable install is /opt/hermes-agent (parent of
# hermes_cli/, i.e. Path(config.py).parent.parent). Baking the stamp here makes
# the dashboard "Update Hermes" button refuse regardless of runtime container
# detection — exactly what upstream's own published image does (it bakes a
# docker stamp into /opt/hermes). Belt-and-suspenders with start.sh's home stamp:
# if a future hermes release changes or drops is_container()'s Railway marker
# (/run/.containerenv), the home stamp would stop being honored but this one
# still refuses. Re-verify the install-tree path if hermes stops installing
# editable from /opt/hermes-agent.
RUN printf 'docker\n' > /opt/hermes-agent/.install_method

COPY requirements.txt /app/requirements.txt
RUN uv pip install --system --no-cache -r /app/requirements.txt

RUN mkdir -p /data/.hermes

COPY server.py /app/server.py
COPY templates/ /app/templates/
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

ENV HOME=/data
ENV HERMES_HOME=/data/.hermes

# Points hermes at our pre-built TUI bundle. hermes's _make_tui_argv checks
# HERMES_TUI_DIR first: if dist/entry.js exists there, it skips the npm
# install/build entirely. This is the official packager path (Nix uses it too)
# and avoids the 30-60s npm bootstrap that git-editable installs would otherwise
# trigger on first /chat connection.
ENV HERMES_TUI_DIR=/opt/hermes-agent/ui-tui

# tini wraps start.sh so it runs as PID 1's child instead of as PID 1 itself.
# `-g` propagates signals to the whole process group so `docker stop` /
# Railway's SIGTERM cleanly terminates the entire tree, not just start.sh.
ENTRYPOINT ["/usr/bin/tini", "-g", "--"]
CMD ["/app/start.sh"]
