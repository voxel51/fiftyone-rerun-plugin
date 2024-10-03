import { PluginComponentType, registerComponent } from "@fiftyone/plugins";
import React from "react";

const App = () => {
  return <div>test</div>;
};

const RerunFileDescriptor = {
  EMBEDDED_DOC_TYPE: "fiftyone.utils.rerun.RrdFile",
};

registerComponent({
  name: "Rerun",
  label: "Rerun",
  component: App,
  activator: (ctx) => {
    // only activate if schema has rrd file
    return true;
    // return doesSchemaContainEmbeddedDocType(
    //   ctx.schema,
    //   RerunFileDescriptor.EMBEDDED_DOC_TYPE
    // );
  },
  type: PluginComponentType.Panel,
    panelOptions: {
      surfaces: "modal",
      helpMarkdown: `Rereun viewer for FiftyOne`,
    },
});
