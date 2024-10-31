import React, { useMemo } from "react";

export const RrdIframeRenderer = React.memo(({ url }: { url: string }) => {
  const iframeSrc = useMemo(() => {
    return `http://localhost:9090/?url=${encodeURIComponent(url)}`;
  }, [url]);

  return <iframe src={iframeSrc} style={{ width: "100%", height: "100%" }} />;
});
