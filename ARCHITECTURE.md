# Architecture

## Goal

Render `.rrd` files inside the FiftyOne modal.

## Why iframe

We run Rerun in an isolated iframe instead of embedding directly in the main React tree.

Benefits:
- Reduces resource contention between FiftyOne UI and Rerun rendering/runtime work.
- Keeps WebGPU/WASM lifecycle isolated from parent React reconciliation and plugin re-renders.
- Contains failures (startup/runtime errors stay inside iframe, not the full app surface).
- Provides sandbox boundaries (`sandbox` + controlled `allow` flags).

## High-level design

Components:
- `src/RerunReactRenderer.tsx`: resolves the sample’s RRD URL from FiftyOne state.
- `src/RrdIframeRenderer.tsx`: renders iframe `srcDoc`, injects runtime script URL, passes RRD URL payload.
- `src/rrd-viewer-runtime.ts`: boots `@rerun-io/web-viewer` inside iframe and opens data source.

Build/runtime:
- Vite emits a separate runtime asset for iframe execution.
- Parent passes URL into iframe via base64 in `data-rrd-b64` to avoid `srcDoc` escaping/quoting issues.

## Data flow

1. Parent resolves sample media URL via `fos.getSampleSrc(...)`.
2. Parent mounts iframe with:
   - `#root` container
   - `data-rrd-b64="<base64-encoded-url>"`
   - module script pointing to iframe runtime bundle
3. Iframe runtime decodes and normalizes URL.
4. Runtime chooses loading strategy:
   - Direct open: `viewer.start(rrdUrl, ...)` for normal HTTP(S) `.rrd` URLs.
   - Streaming fallback: `viewer.start(null, ...)` + fetch bytes + `open_channel(...).send_rrd(...)`
     for FiftyOne `/media?filepath=...` URLs that Rerun URL parsing seems to reject.

## Important implementation detail

For channel mode, `send_rrd` is used with one complete RRD payload per call.

Reason:
- Sending arbitrary network chunks produced repeated decode errors (`expected "RRF2"`).
- We now fetch the whole response and send a single `Uint8Array`.

## Security/sandbox posture

Iframe uses:
- `sandbox="allow-downloads allow-forms allow-popups allow-same-origin allow-scripts"`
- `allow="fullscreen"`

This keeps execution separated while allowing the minimum capabilities needed by the viewer.

## Tradeoffs

- iframe isolation improves stability but adds a small integration layer (srcDoc/runtime bootstrapping).
- channel fallback is robust for FiftyOne media proxy URLs but buffers full RRD into memory before send.
- direct URL path remains preferred when Rerun accepts the URL as-is.
