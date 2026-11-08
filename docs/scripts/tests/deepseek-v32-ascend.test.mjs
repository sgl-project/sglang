// Exercise the actual DeepSeek-V3.2 selector and its Ascend A3 shell output through the documentation CI entry.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { test } from 'node:test';

const read = (path) => readFileSync(new URL(path, import.meta.url), 'utf8').replace(/\r\n/g, '\n');
const source = read('../../src/snippets/autoregressive/deepseek-v32-deployment.jsx');
const hooks = source.indexOf('  const [values, setValues]');
assert.ok(hooks > 0);
const setup = source.slice(0, hooks).replace(/^export /, '');
const { options, generateCommand, getInitialState, getUpdatedValues } = new Function(
  `${setup}\nreturn { options, generateCommand, getInitialState, getUpdatedValues };\n};\nreturn DeepSeekV32Deployment();`
)();
const defaults = getInitialState();
const ascend = getUpdatedValues(defaults, 'hardware', 'a3');
const build = (selection = {}) => generateCommand({ ...ascend, ...selection });
const node = (npuRole, npuNodeRank = '0') => build({ npuRole, npuNodeRank });
const servers = [['prefill', '0'], ['prefill', '1'], ['decode', '0'], ['decode', '1']];
const allCommands = () => [...servers.map(([role, rank]) => node(role, rank)), node('router')];

// The frozen best-practice recipe the A3 commands are copied from (26 ms TPOT, 128K input).
const bestPractice = read('../../docs/hardware-platforms/ascend-npus/model-deployment/best-practices/deepseek_v3_2.mdx');
const recipeStart = bestPractice.indexOf('### DeepSeek-V3.2 W8A8 1P1D 32P IN128K OUT1K 26ms');
assert.ok(recipeStart > 0);
const recipe = bestPractice.slice(recipeStart, bestPractice.indexOf('\n### ', recipeStart + 1));
const [deployScript, routerScript] = [...recipe.matchAll(/```bash Command\n([\s\S]*?)```/g)].map((m) => m[1]);
const commonScript = deployScript.slice(0, deployScript.indexOf('# prefill'));
const prefillScript = deployScript.slice(deployScript.indexOf('# prefill'), deployScript.indexOf('# decode'));
const decodeScript = deployScript.slice(deployScript.indexOf('# decode'));

const lines = (script) => script.split('\n').map((line) => line.trim());
const flags = (script) => lines(script).filter((line) => line.startsWith('--')).map((line) => line.replace(/\s*\\$/, ''));
const exportsOf = (script) => Object.fromEntries([...script.matchAll(/^\s*export (\w+)=(.*)$/gm)].map((m) => [m[1], m[2].trim()]));
const unsets = (script) => [...script.matchAll(/^\s*unset (.+)$/gm)].flatMap((m) => m[1].trim().split(/\s+/)).sort();
const hostSetup = (script) => lines(script).filter((line) => /^(echo performance|sysctl|source) /.test(line));

// Map the best practice onto one node of the selector: per-node addresses, and the Cookbook
// server port in place of 8000 (prefill) / 8001 (decode). These are the only allowed divergences.
const localize = (script, rank) => script
  .replaceAll('${MODEL_PATH}', '/path/to/DeepSeek-V3.2-W8A8')
  .replaceAll('${P_IP[0]}', '<prefill-node0-ip>')
  .replaceAll('${D_IP[0]}', '<decode-node0-ip>')
  .replaceAll('${P_IP[$i]}', `<prefill-node${rank}-ip>`)
  .replaceAll('${D_IP[$i]}', `<decode-node${rank}-ip>`)
  .replaceAll('--node-rank $i', `--node-rank ${rank}`)
  .replaceAll('<your prefill ip1>', '<prefill-node0-ip>')
  .replaceAll('<your prefill ip>', '<prefill-node0-ip>')
  .replaceAll('<your decode ip1>', '<decode-node0-ip>')
  .replace(/(--port |:)800[01]\b/g, (_, prefix) => `${prefix}30000`);

test('switching hardware selects DeepSeek-V3.2 on A3 and restores a GPU recipe', () => {
  assert.equal(ascend.modelname, 'v32');
  for (const [hardware, modelname] of [['b200', 'v32nvfp4'], ['mi355x', 'v32mxfp4'], ['h200', 'v32exp']]) {
    const picked = getUpdatedValues(getUpdatedValues(defaults, 'hardware', hardware), 'modelname', modelname);
    assert.equal(getUpdatedValues(picked, 'hardware', 'a3').modelname, 'v32', modelname);
  }
  for (const name of ['npuDeployment', 'npuRole', 'npuNodeRank']) assert.equal(options[name].condition(ascend), true, name);
  for (const name of ['strategy', 'reasoningParser', 'toolcall']) assert.equal(options[name].condition(ascend), false, name);
  assert.equal(options.npuNodeRank.condition({ ...ascend, npuRole: 'router' }), false);
  const gpu = generateCommand(getUpdatedValues(ascend, 'hardware', 'h200'));
  assert.match(gpu, /^sglang serve/);
  assert.match(gpu, /--model-path deepseek-ai\/DeepSeek-V3\.2(?:\s|$)/);
  assert.doesNotMatch(gpu, /modelslim|HCCL|--device npu|--disaggregation-mode/);
});

test('A3 server commands carry the best-practice flags and environment, apart from addresses and port', () => {
  for (const [role, rank] of servers) {
    const command = node(role, rank);
    const script = role === 'prefill' ? prefillScript : decodeScript;
    const label = `${role} node ${rank}`;
    assert.ok(command.includes('python3 -m sglang.launch_server'), label);
    assert.deepEqual(flags(command), flags(localize(script, rank)), `${label} flags`);
    assert.deepEqual(exportsOf(command), exportsOf(localize(commonScript + script, rank)), `${label} env`);
    assert.deepEqual(unsets(command), unsets(commonScript), `${label} unset`);
    assert.deepEqual(hostSetup(command), hostSetup(commonScript), `${label} host setup`);
  }
});

test('the router command matches the best practice and targets the node-0 servers', () => {
  const router = node('router');
  assert.ok(router.includes('python -m sglang_router.launch_router'));
  assert.deepEqual(flags(router), flags(localize(routerScript, 0)));
  assert.doesNotMatch(router, /--model-path|sglang\.launch_server/);
});

test('every node agrees on head-node addresses, the MemFabric store and the router ports', () => {
  const bootstrap = node('prefill').match(/--disaggregation-bootstrap-port (\d+)/)[1];
  const router = node('router');
  for (const [role, rank] of servers) {
    const command = node(role, rank);
    for (const expected of [`--host <${role}-node${rank}-ip>`, `--dist-init-addr <${role}-node0-ip>:5000`,
      `--node-rank ${rank}`, '--nnodes 2', '--tp 32', '--port 30000',
      'export ASCEND_MF_STORE_URL="tcp://<prefill-node0-ip>:24670"']) {
      assert.ok(command.includes(expected), `${role} node ${rank}: ${expected}`);
    }
    assert.doesNotMatch(command, /--port 800[01]\b/);
  }
  assert.ok(router.includes(`--prefill http://<prefill-node0-ip>:30000 ${bootstrap}`));
  assert.ok(router.includes('--decode http://<decode-node0-ip>:30000'));
  assert.match(router, /--port 6688(?:\s|$)/);
});

test('unsupported selections do not emit a launch command', () => {
  for (const invalid of [{ modelname: 'v32exp' }, { modelname: 'v32speciale' }, { npuDeployment: 'unknown' },
    { npuRole: 'unknown' }, { npuRole: 'decode', npuNodeRank: '2' }]) {
    assert.doesNotMatch(build(invalid), /sglang\.launch_server|sglang_router/, JSON.stringify(invalid));
  }
});

// Replace the documented placeholders the way a reader would, then run the real script with stubs.
const fill = (command) => command
  .replaceAll('/path/to/DeepSeek-V3.2-W8A8', '/models/DeepSeek-V3.2-W8A8')
  .replaceAll('<network-interface>', 'eth0')
  .replace(/<prefill-node(\d)-ip>/g, (_, n) => `10.0.0.${1 + Number(n)}`)
  .replace(/<decode-node(\d)-ip>/g, (_, n) => `10.0.1.${1 + Number(n)}`);

test('every generated role shows placeholders and is valid Bash once they are replaced', () => {
  for (const command of allCommands()) {
    assert.match(command, /<prefill-node0-ip>/);
    const filled = fill(command);
    assert.doesNotMatch(filled, /<[a-z0-9-]+>/);
    const checked = spawnSync('bash', ['-n'], { input: filled, encoding: 'utf8' });
    assert.equal(checked.status, 0, checked.stderr);
  }
});

test('the real generated server scripts export the store URL and pass the expected argv', () => {
  const stubs = 'tee() { cat >/dev/null; }; sysctl() { :; }; source() { :; };\n'
    + 'python3() { printf "MF=%s\\n" "$ASCEND_MF_STORE_URL"; printf "%s\\n" "$@"; };\n';
  for (const [role, rank] of servers) {
    const result = spawnSync('bash', ['-c', stubs + fill(node(role, rank))], { encoding: 'utf8' });
    assert.equal(result.status, 0, result.stderr);
    const [store, ...argv] = result.stdout.trim().split('\n');
    const arg = (name) => argv[argv.indexOf(name) + 1];
    assert.equal(store, 'MF=tcp://10.0.0.1:24670');
    assert.deepEqual(argv.slice(0, 2), ['-m', 'sglang.launch_server']);
    assert.equal(arg('--model-path'), '/models/DeepSeek-V3.2-W8A8');
    assert.equal(arg('--disaggregation-mode'), role);
    assert.equal(arg('--node-rank'), rank);
    assert.equal(arg('--port'), '30000');
  }
});

test('A3 commands keep every line short enough to read in the command panel', () => {
  for (const command of allCommands()) {
    const long = command.split('\n').filter((line) => line.length > 90);
    assert.deepEqual(long, []);
  }
});
