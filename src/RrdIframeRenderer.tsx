import React, { useMemo } from "react";
import iframeRuntimeUrl from "./rrd-viewer-runtime.ts?worker&url";

type RrdIframeRendererProps = {
  url: string;
};

const IFRAME_SHELL = `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <style>
      html, body, #root {
        width: 100%;
        height: 100%;
        margin: 0;
        padding: 0;
        overflow: hidden;
      }
      body {
        background: #0d1117;
      }
      #error {
        display: none;
        box-sizing: border-box;
        align-items: center;
        justify-content: center;
        width: 100%;
        height: 100%;
        margin: 0;
        padding: 24px;
        color: #ffb4b4;
        font: 13px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        text-align: center;
        white-space: pre-wrap;
      }
    </style>
  </head>
  <body>
    <div id="root" data-rrd-b64="__RERUN_RRD_B64__"></div>
    <div id="error" role="alert"></div>
    <script type="module" src="__RERUN_IFRAME_RUNTIME__"></script>
  </body>
</html>`;

const toBase64 = (value: string): string => {
  const bytes = new TextEncoder().encode(value);
  let binary = "";

  for (const byte of bytes) {
    binary += String.fromCharCode(byte);
  }

  return btoa(binary);
};

export const RrdIframeRenderer = React.memo(({ url }: RrdIframeRendererProps) => {
  const encodedRrdUrl = useMemo(() => {
    // Base64 avoids URL/query escaping conflicts when serializing through srcDoc.
    return toBase64(url);
  }, [url]);

  const srcDoc = useMemo(() => {
    return IFRAME_SHELL.replaceAll("__RERUN_IFRAME_RUNTIME__", iframeRuntimeUrl).replaceAll(
      "__RERUN_RRD_B64__",
      encodedRrdUrl
    );
  }, [encodedRrdUrl]);

  return (
    <iframe
      allow="fullscreen"
      sandbox="allow-downloads allow-forms allow-popups allow-same-origin allow-scripts"
      srcDoc={srcDoc}
      style={{ width: "100%", height: "100%", border: 0, display: "block" }}
      title="Rerun Viewer"
    />
  );
});
