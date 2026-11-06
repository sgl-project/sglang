import { app } from '../../../scripts/app.js';
import { api } from '../../../scripts/api.js';
import { ComfyWidgets } from '../../../scripts/widgets.js';

const nodes = () => app.graph.findNodesByType('SGLDWorkerStatus');

app.registerExtension({
  name: 'SGLDiffusion.WorkerStatus',
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== 'SGLDWorkerStatus') return;
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function (...args) {
      const result = created?.apply(this, args);
      const { widget } = ComfyWidgets.STRING(this, 'status', ['STRING', { multiline: true }], app);
      widget.inputEl.readOnly = true;
      widget.serializeValue = () => undefined;
      this.statusWidget = widget;
      return result;
    };
    const executed = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (output) {
      executed?.call(this, output);
      this.statusWidget.value = output.text?.[0] ?? '';
      this.setDirtyCanvas(true, true);
    };
  },
  setup() {
    api.addEventListener('sgld.worker_status', ({ detail }) => {
      for (const node of nodes()) {
        node.statusWidget.value = detail.text;
        node.setDirtyCanvas(true, true);
      }
    });
  },
});
