import React from "react";
import { RerunViewerReact } from "./RerunReactRenderer";

/**
 * Panel entrypoint for embedded `RrdFile` fields exposed in the sample modal.
 */
export const RerunViewer = React.memo(() => {
  return <RerunViewerReact />;
});
