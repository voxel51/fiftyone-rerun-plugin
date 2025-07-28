import * as fop from "@fiftyone/plugins";
import * as futil from "@fiftyone/utilities";
import { RerunFileDescriptor, RerunViewer } from "./RerunPanel";

fop.registerComponent({
  name: "Rerun",
  label: "Rerun",
  component: RerunViewer,
  activator: (ctx) => {
    // only activate if schema has rrd file
    return futil.doesSchemaContainEmbeddedDocType(
      ctx.schema,
      RerunFileDescriptor.EMBEDDED_DOC_TYPE
    );
  },
  type: fop.PluginComponentType.Panel,
  panelOptions: {
    surfaces: "modal",
    helpMarkdown: `Rereun viewer for FiftyOne`,
    isNew: false,
  },
});
