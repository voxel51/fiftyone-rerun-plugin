import * as fop from "@fiftyone/plugins";
import * as futil from "@fiftyone/utilities";
import { RerunViewer } from "./RerunPanel";
import { RerunSampleRenderer } from "./RerunReactRenderer";
import { RerunFileDescriptor } from "./rerunUtils";

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
    helpMarkdown: `Rerun viewer for FiftyOne`,
    isNew: false,
  },
});

fop.registerComponent({
  name: "RerunSampleRenderer",
  label: "Rerun",
  component: RerunSampleRenderer,
  type: fop.PluginComponentType.SampleRenderer,
  activator: () => true,
  sampleRendererOptions: {
    supports: {
      extensions: ["rrd"],
    },
    grid: {
      enabled: false,
    },
  },
});
