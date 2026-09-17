const assert = require('node:assert/strict');
const {tilt, sideName, guideTransform} = require('./static/capture-guide.js');
assert.equal(guideTransform('right'), '');
assert.equal(guideTransform('left'), 'translate(300 0) scale(-1 1)');
assert.equal(guideTransform('left', 160), 'translate(160 0) scale(-1 1)');
assert.equal(tilt(null, 0), null);
assert.equal(tilt(0, undefined), null);
assert.equal(tilt(180, 0), null);
assert.equal(tilt(0, 0), 0);
for (const a of [5, 30, 32, 35, 45]) {
  assert.ok(Math.abs(tilt(a, 0) - a) < 1e-8);
  assert.ok(Math.abs(tilt(0, -a) - a) < 1e-8);
}
assert.match(sideName('right','image_left'), /内侧/);
assert.match(sideName('right','image_right'), /外侧/);
assert.match(sideName('left','image_left'), /外侧/);
assert.match(sideName('left','image_right'), /内侧/);
console.log('Capture guide: invalid readings, orientation, four side mappings passed.');
const {createAngleGate} = require('./static/capture-guide.js');
const gate = createAngleGate(false);
assert.equal(gate.ready(0), false);
for (let t = 0; t <= 400; t += 100) gate.sample(32, t);
assert.equal(gate.ready(400), false);
gate.sample(32, 500); assert.equal(gate.ready(500), true);
gate.sample(29, 510); assert.equal(gate.ready(510), false);
for (let t = 600; t <= 1100; t += 100) gate.sample(35, t);
assert.equal(gate.ready(1100), true);
assert.equal(gate.ready(1500), false); // A stale green reading cannot unlock capture.
gate.sample(32, 1500); assert.equal(gate.ready(1500), false);
gate.sample(null, 1510); assert.equal(gate.ready(1510), false);
const topGate = createAngleGate(true);
for (let t = 0; t <= 500; t += 100) topGate.sample(5, t);
assert.equal(topGate.ready(500), true);
topGate.sample(5.1, 510); assert.equal(topGate.ready(510), false);
console.log('Shutter gate: stable interval, out-of-range, missing and stale samples passed.');
