import React from "react";
import { Rrd_V_0_19 } from "./RerunV19_0";

export const RrdWebViewerRenderer = React.memo(
  ({ url, version }: { url: string; version: string }) => {
    switch (version) {
      // todo: implement versioned renderers
      default:
        return <Rrd_V_0_19 url={url} />;
    }
  }
);
