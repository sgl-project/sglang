import { app } from '../../../scripts/app.js';
import { GROUPS, defaultBackendConfig, headerLabel, visibleField } from './options_state.mjs';

app.registerExtension({
  name: 'SGLDiffusion.Options',
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== 'SGLDOptions') return;
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function (...args) {
      const result = created?.apply(this, args);
      const node = this;
      const fields = nodeData.input.optional;
      const originals = [...node.widgets];
      const fieldWidgets = originals.filter(w => fields[w.name]);
      const states = new WeakMap();
      const headers = {};
      node.properties ??= {};
      const sections = () => node.properties.sgld_sections ??= {};
      const values = () => Object.fromEntries(fieldWidgets.map(w => [w.name, w.value]));
      for (const [group, label] of Object.entries(GROUPS)) {
        const header = node.addWidget('button', label, null, () => {
          sections()[group] = !sections()[group];
          refresh();
        }, { serialize: false });
        header.serializeValue = () => undefined;
        headers[group] = header;
      }
      // Never reorder the actual input widgets relative to one another.
      const ordered = [];
      const inserted = new Set();
      for (const widget of originals) {
        const group = fields[widget.name]?.[1]?.sgld_group;
        if (headers[group] && !inserted.has(group)) {
          ordered.push(headers[group]);
          inserted.add(group);
        }
        ordered.push(widget);
      }
      node.widgets = ordered;

      function setVisible(widget, visible) {
        if (!states.has(widget)) states.set(widget, { type: widget.type, computeSize: widget.computeSize, hidden: widget.hidden });
        const original = states.get(widget);
        widget.hidden = visible ? original.hidden : true;
        widget.type = visible ? original.type : 'sgld-hidden';
        widget.computeSize = visible ? original.computeSize : () => [0, -4];
      }
      function refresh() {
        const v = values();
        for (const widget of fieldWidgets) {
          const group = fields[widget.name][1].sgld_group;
          // Connected primitives must keep their inputs reachable.
          const connected = node.inputs?.some((input, index) =>
            input.widget?.name === widget.name && node.isInputConnected?.(index));
          setVisible(widget, connected || ((group === 'basic' || sections()[group]) && visibleField(widget.name, v)));
        }
        for (const [group, header] of Object.entries(headers)) {
          header.name = headerLabel(group, v, sections()[group]);
        }
        node.setSize([Math.max(node.size[0], 420), node.computeSize()[1]]);
        node.setDirtyCanvas(true, true);
      }
      for (const widget of fieldWidgets) {
        const callback = widget.callback;
        widget.callback = function (...args) {
          const result = callback?.apply(this, args);
          if (widget.name === 'attention_backend') {
            const config = fieldWidgets.find(w => w.name === 'attention_backend_config');
            if (config) config.value = defaultBackendConfig(widget.value, config.value);
          }
          refresh();
          return result;
        };
      }
      const configure = node.onConfigure;
      node.onConfigure = function (...args) {
        const result = configure?.apply(this, args);
        refresh();
        return result;
      };
      const connections = node.onConnectionsChange;
      node.onConnectionsChange = function (...args) {
        const result = connections?.apply(this, args);
        queueMicrotask(refresh);
        return result;
      };
      refresh();
      return result;
    };
  },
});
