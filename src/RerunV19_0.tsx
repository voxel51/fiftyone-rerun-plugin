import * as RerunWebViewer from "@rerun-io/web-viewer";
import React, { useEffect, useRef, useState } from "react";

export const Rrd_V_0_19 = React.memo(({ url }: { url: string }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<RerunWebViewer.WebViewer>();

  const [_rerender, setRerender] = useState(0);

  useEffect(() => {
    if (!containerRef.current) {
      return;
    }

    const loadRerun = async () => {
      viewerRef.current = new RerunWebViewer.WebViewer();
      await viewerRef.current.start(url, containerRef.current, {
        render_backend: "webgpu",
        allow_fullscreen: false,
        hide_welcome_screen: true,
        width: "100%",
        height: "100%",
      });
      setRerender((prev) => prev + 1);
    };

    loadRerun();

    return () => {
      viewerRef.current?.stop();
    };
  }, [url]);

  if (!viewerRef.current) {
    return (
      <div ref={containerRef} style={{ height: "100%", width: "100%" }}>
        Loading
      </div>
    );
  }

  return <div ref={containerRef} style={{ height: "100%", width: "100%" }} />;
});
