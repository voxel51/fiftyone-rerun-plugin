import React, { useMemo } from "react";

declare global {
  interface Window {
    __ENV__: {
      PORT_RERUN: string;
    };
  }
}

const DEFAULT_RERUN_PORT = 9090;

export const RrdIframeRenderer = React.memo(({ url }: { url: string }) => {
  /**
   * 1. check if rerunPort is provided in the environment variable
   * 2. check if rerunPort is injected in the window.__ENV__ object
   * 3. if not, use the default value
   */
  const rerunPort = useMemo(() => {
    const url = new URL(window.location.href);

    if (typeof process !== "undefined") {
      if (process.env?.PORT_RERUN) {
        return parseInt(process.env.PORT_RERUN);
      }

      if (process.env?.NEXT_PUBLIC_RERUN_PORT) {
        return parseInt(process.env.NEXT_PUBLIC_PORT_RERUN);
      }
    }

    if (window.__ENV__?.PORT_RERUN) {
      return parseInt(window.__ENV__.PORT_RERUN);
    }

    return DEFAULT_RERUN_PORT;
  }, []);

  const iframeSrc = useMemo(() => {
    return `http://localhost:${rerunPort}/?url=${encodeURIComponent(url)}`;
  }, [url]);

  return <iframe src={iframeSrc} style={{ width: "100%", height: "100%" }} />;
});
