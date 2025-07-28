import React from "react";
import WebViewer from "@rerun-io/web-viewer-react";

export const RerunReactRenderer = React.memo(
  ({ url, version }: { url: string; version: string }) => {
    switch (version) {
      // todo: implement versioned renderers
      default:
        return <WebViewer rrd={url} width="100%" height="100%" />;
    }
  }
);
