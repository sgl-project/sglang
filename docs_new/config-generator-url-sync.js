(function () {
  // Supports both existing generators with hidden input[name][value] controls
  // and the unified template contract:
  //   data-config-option="<query key>"
  //   data-config-value="<option id>"
  //   data-config-active="true|false"
  //   data-config-disabled="true|false"
  //   data-config-multiple="true" for checkbox-like groups
  const OPTION_ALIASES = {
    modelVariant: 'model',
    mtp: 'speculative',
    reasoning_parser: 'reasoning',
    'reasoning-parser': 'reasoning',
    strategy: 'strategy',
    tool_calling: 'toolcall',
    'tool-calling': 'toolcall',
    variant: 'model',
  };

  const LEGACY_FEATURE_OPTIONS = {
    mtp: 'speculative',
    reasoning: 'reasoning',
    reasoning_parser: 'reasoning',
    'reasoning-parser': 'reasoning',
    speculative: 'speculative',
    toolcall: 'toolcall',
    tool_calling: 'toolcall',
    'tool-calling': 'toolcall',
  };

  const FEATURE_VALUE_ALIASES = {
    chat_template: 'chatTemplate',
    'chat-template': 'chatTemplate',
    fp8_kv_cache: 'kvCacheFP8',
    'fp8-kv-cache': 'kvCacheFP8',
    kv_cache_fp8: 'kvCacheFP8',
    'kv-cache-fp8': 'kvCacheFP8',
    kvfp8: 'kvCacheFP8',
    mamba: 'mambaCache',
    mamba_cache: 'mambaCache',
    'mamba-cache': 'mambaCache',
    reasoning_parser: 'reasoning',
    'reasoning-parser': 'reasoning',
    tool_calling: 'toolCall',
    'tool-calling': 'toolCall',
    toolcall: 'toolCall',
  };

  const STRATEGY_VALUE_ALIASES = {
    eagle: 'mtp',
    speculative: 'mtp',
    speculative_decoding: 'mtp',
    'speculative-decoding': 'mtp',
  };

  const FEATURE_QUERY_NAMES = {
    reasoning: 'reasoning',
    speculative: 'mtp',
    toolcall: 'tool_calling',
  };

  const VALUE_ALIASES = {
    false: 'disabled',
    no: 'disabled',
    off: 'disabled',
    on: 'enabled',
    true: 'enabled',
    yes: 'enabled',
  };

  const FALLBACK_VALUE_ALIASES = {
    gb200: 'b200',
    gb300: 'b300',
  };

  const normalize = (value) => String(value || '').trim().toLowerCase();
  let applyingQuery = false;

  const resolveOptionName = (name) => OPTION_ALIASES[normalize(name)] || name;

  const resolveLegacyFeatureOption = (name) => LEGACY_FEATURE_OPTIONS[normalize(name)] || null;

  const resolveValue = (value) => VALUE_ALIASES[normalize(value)] || value;

  const resolveGroupValue = (optionName, value) => {
    const normalizedValue = normalize(value);
    if (normalize(optionName) === 'features') {
      return FEATURE_VALUE_ALIASES[normalizedValue] || value;
    }
    if (normalize(optionName) === 'strategy') {
      return STRATEGY_VALUE_ALIASES[normalizedValue] || value;
    }
    return resolveValue(value);
  };

  const isCookbookPage = () => window.location.pathname.includes('/cookbook/');

  const isGeneratorRoot = (element) =>
    !!element &&
    element.matches?.('.not-prose') &&
    (!!element.querySelector('input[name][value]') ||
      !!element.querySelector('[data-config-option][data-config-value]')) &&
    !!element.querySelector('pre');

  const findGeneratorRoots = () =>
    Array.from(document.querySelectorAll('.not-prose')).filter(isGeneratorRoot);

  const findGeneratorRoot = (element) => {
    const root = element?.closest?.('.not-prose');
    return isGeneratorRoot(root) ? root : null;
  };

  const controlFromInput = (input) => ({
    active: () => input.checked,
    click: () => input.click(),
    disabled: () => input.disabled,
    element: input,
    multiple: () => input.type === 'checkbox',
    optionName: input.name,
    value: input.value,
  });

  const controlFromDataElement = (element) => ({
    active: () => element.getAttribute('data-config-active') === 'true',
    click: () => element.click(),
    disabled: () => element.getAttribute('data-config-disabled') === 'true',
    element,
    multiple: () => element.getAttribute('data-config-multiple') === 'true',
    optionName: element.getAttribute('data-config-option') || '',
    value: element.getAttribute('data-config-value') || '',
  });

  const getControls = (root) => [
    ...Array.from(root.querySelectorAll('input[name][value]')).map(controlFromInput),
    ...Array.from(root.querySelectorAll('[data-config-option][data-config-value]')).map(
      controlFromDataElement
    ),
  ];

  const getNamedControls = (root, optionName) =>
    getControls(root).filter((control) => normalize(control.optionName) === normalize(optionName));

  const getOptionNames = (root) =>
    Array.from(new Set(getControls(root).map((control) => control.optionName).filter(Boolean)));

  const findControlForValue = (controls, rawValue) => {
    const direct = controls.find((control) => normalize(control.value) === normalize(rawValue));
    if (direct) return direct;

    const normalizedValue = normalize(rawValue);
    const aliased = FALLBACK_VALUE_ALIASES[normalizedValue];
    if (!aliased) return null;

    return controls.find((control) => normalize(control.value) === normalize(aliased)) || null;
  };

  const selectOption = (root, rawName, rawValue) => {
    const optionName = resolveOptionName(rawName);
    const optionValue = resolveValue(rawValue);
    const controls = getNamedControls(root, optionName).filter((control) => !control.disabled());
    if (!controls.length) return false;

    const target = findControlForValue(controls, optionValue);
    if (!target || target.active()) return false;

    target.click();
    return true;
  };

  const setMultiValue = (root, rawName, rawValues) => {
    const optionName = resolveOptionName(rawName);
    const controls = getNamedControls(root, optionName).filter((control) => !control.disabled());
    if (!controls.length || !controls.some((control) => control.multiple())) return false;

    const desiredValues = String(rawValues)
      .split(',')
      .map((value) => resolveGroupValue(optionName, value))
      .filter(Boolean);
    const desired = new Set(desiredValues.map(normalize));
    let changed = false;

    for (const control of controls) {
      const shouldBeActive = desired.has(normalize(control.value));
      if (control.active() !== shouldBeActive) {
        control.click();
        changed = true;
      }
    }

    return changed;
  };

  const hasMultiGroup = (root, optionName) =>
    getNamedControls(root, optionName).some((control) => control.multiple());

  const applyFeatures = (root, rawFeatures) => {
    if (hasMultiGroup(root, 'features')) {
      setMultiValue(root, 'features', rawFeatures);
      return new Set(
        String(rawFeatures)
          .split(',')
          .map((value) => normalize(resolveGroupValue('features', value)))
      );
    }

    const enabledFeatureOptions = new Set();
    const featureOptionNames = Object.keys(FEATURE_QUERY_NAMES).filter(
      (optionName) => getNamedControls(root, optionName).length > 0
    );

    for (const optionName of featureOptionNames) {
      selectOption(root, optionName, 'disabled');
    }

    for (const feature of String(rawFeatures).split(',')) {
      const optionName = resolveLegacyFeatureOption(feature);
      if (optionName) {
        enabledFeatureOptions.add(optionName);
        selectOption(root, optionName, 'enabled');
      }
    }

    return enabledFeatureOptions;
  };

  const applyQueryToRoot = (root) => {
    const params = new URLSearchParams(window.location.search);
    let speculativeDisabled = false;
    let hasMambaCacheParam = false;

    if (params.has('features')) {
      const enabledFeatures = applyFeatures(root, params.get('features'));
      speculativeDisabled = !enabledFeatures.has('speculative');
    }

    for (const [rawName, rawValue] of params.entries()) {
      if (rawName === 'features') continue;
      const optionName = resolveOptionName(rawName);
      if (optionName === 'mambaCache') hasMambaCacheParam = true;
      if (optionName === 'speculative' && normalize(resolveValue(rawValue)) === 'disabled') {
        speculativeDisabled = true;
      }
      if (!setMultiValue(root, rawName, rawValue)) {
        selectOption(root, rawName, rawValue);
      }
    }

    if (speculativeDisabled && !hasMambaCacheParam) {
      selectOption(root, 'mambaCache', 'v1');
    }
  };

  const selectedState = (root) => {
    const state = {};
    const multiState = {};

    for (const optionName of getOptionNames(root)) {
      const controls = getNamedControls(root, optionName);
      if (controls.some((control) => control.multiple())) {
        multiState[optionName] = controls
          .filter((control) => control.active())
          .map((control) => control.value);
      } else {
        const selected = controls.find((control) => control.active());
        if (selected) state[optionName] = selected.value;
      }
    }

    return { multiState, state };
  };

  const updateUrlFromRoot = (root) => {
    const url = new URL(window.location.href);
    const { multiState, state } = selectedState(root);
    const optionNames = getOptionNames(root);
    const enabledFeatures = [];

    for (const optionName of optionNames) {
      url.searchParams.delete(optionName);
    }

    for (const [optionName, value] of Object.entries(state)) {
      if (FEATURE_QUERY_NAMES[optionName]) {
        if (value === 'enabled') enabledFeatures.push(FEATURE_QUERY_NAMES[optionName]);
      } else {
        url.searchParams.set(optionName, value);
      }
    }

    for (const [optionName, values] of Object.entries(multiState)) {
      if (!values.length) {
        url.searchParams.delete(optionName);
        continue;
      }

      if (optionName === 'features') {
        url.searchParams.set(
          'features',
          values
            .map((value) => {
              if (normalize(value) === 'toolcall') return 'tool_calling';
              if (normalize(value) === 'toolcallparser') return 'tool_calling';
              return value;
            })
            .join(',')
        );
      } else {
        url.searchParams.set(optionName, values.join(','));
      }
    }

    if (optionNames.some((optionName) => FEATURE_QUERY_NAMES[optionName])) {
      if (enabledFeatures.length) {
        url.searchParams.set('features', enabledFeatures.join(','));
      } else {
        url.searchParams.delete('features');
      }
    }

    window.history.replaceState(window.history.state, '', url);
  };

  let applyTimer = null;
  const scheduleApplyFromUrl = () => {
    if (!isCookbookPage()) return;
    if (applyTimer) window.clearTimeout(applyTimer);

    applyTimer = window.setTimeout(() => {
      applyingQuery = true;
      for (const root of findGeneratorRoots()) {
        applyQueryToRoot(root);
      }
      window.setTimeout(() => {
        applyingQuery = false;
      }, 0);
    }, 50);
  };

  document.addEventListener('change', (event) => {
    if (!isCookbookPage()) return;
    if (applyingQuery) return;
    const target = event.target;
    if (
      !(target instanceof HTMLInputElement) &&
      !target.closest?.('[data-config-option][data-config-value]')
    ) {
      return;
    }

    const root = findGeneratorRoot(target);
    if (root) updateUrlFromRoot(root);
  });

  document.addEventListener('click', (event) => {
    if (!isCookbookPage()) return;
    if (applyingQuery) return;

    const control = event.target.closest?.('[data-config-option][data-config-value]');
    if (!control) return;

    const root = findGeneratorRoot(control);
    if (root) {
      window.setTimeout(() => updateUrlFromRoot(root), 0);
    }
  });

  window.addEventListener('popstate', scheduleApplyFromUrl);
  window.addEventListener('load', scheduleApplyFromUrl);

  const observer = new MutationObserver(scheduleApplyFromUrl);
  observer.observe(document.documentElement, { childList: true, subtree: true });

  scheduleApplyFromUrl();
})();
