/* Iterative solvers explorer — 1D steady-state heat conduction.
 *
 * Problem: -u''(x) = f on (0,1), u(0)=u(1)=0, discretized as
 * A = tridiag(-1,2,-1)/h^2 on n=256 interior cells. A heater injects +1 into
 * the first cell and a chiller extracts -1 from the last cell, so the exact
 * temperature profile is (nearly) a straight ramp from hot to cold.
 *
 * Solvers: conjugate gradient, SOR (optimal omega), gradient descent
 * (steepest descent; with a fixed step this family includes Richardson and,
 * for this constant-diagonal A, plain Jacobi), and CG with a toy neural
 * preconditioner (arXiv:2502.01337) whose 737 weights ship in npo_weights.js
 * and run right here in the browser (flexible CG, since the network is
 * nonlinear).
 */
'use strict';

// ---------------- Problem setup ----------------
const n = 256;
const h = 1 / (n + 1);
const h2 = h * h;
const TOL = 1e-4;
const MAXIT = 100000;
const MAXFRAMES = 480;

// heater / chiller source
const b = new Float64Array(n);
b[0] = 1.0;
b[n - 1] = -1.0;

// interior grid coordinates x_i = (i+1) h in (0,1)
const X = new Float64Array(n);
for (let i = 0; i < n; i++) X[i] = (i + 1) * h;

// spectrum of A: lam_k = (2 - 2 cos(k pi/(n+1)))/h^2
const theta = Math.PI / (n + 1);
const lamMin = (2 - 2 * Math.cos(theta)) / h2;
const lamMax = (2 + 2 * Math.cos(theta)) / h2;
const kappa = lamMax / lamMin;
const omegaSor = 2 / (1 + Math.sin(theta)); // optimal SOR relaxation

function dot(u, v) { let s = 0; for (let i = 0; i < u.length; i++) s += u[i] * v[i]; return s; }
function nrm(u) { return Math.sqrt(dot(u, u)); }
const bNorm = nrm(b);

// A v with A = tridiag(-1, 2, -1)/h^2 (Dirichlet: v_0 = v_{n+1} = 0)
function applyA(v, out) {
  for (let i = 0; i < n; i++) {
    out[i] = (2 * v[i] - (i > 0 ? v[i - 1] : 0) - (i < n - 1 ? v[i + 1] : 0)) / h2;
  }
}

// exact solution via Thomas algorithm on tridiag(-1,2,-1) x = h^2 b
function exactSolve() {
  const c = new Float64Array(n), d = new Float64Array(n), x = new Float64Array(n);
  c[0] = -1 / 2; d[0] = h2 * b[0] / 2;
  for (let i = 1; i < n; i++) {
    const m = 2 + c[i - 1];
    c[i] = -1 / m;
    d[i] = (h2 * b[i] + d[i - 1]) / m;
  }
  x[n - 1] = d[n - 1];
  for (let i = n - 2; i >= 0; i--) x[i] = d[i] - c[i] * x[i + 1];
  return x;
}
const xExact = exactSolve();

// ---------------- Toy neural preconditioner (weights from npo_weights.js) ----------------
// Two-scale conv net (fine stencil + coarse-grid branch, a multigrid miniature)
// trained offline against h^2*A with the condition/residual losses of
// arXiv:2502.01337. (F)CG is invariant to positive scaling of the
// preconditioner, so the h^2 scale needs no correction. The unit-norm wrapper
// makes the operator positively homogeneous. Pooling / upsample conventions
// mirror python/neural/train_npo_1d.py line-for-line.
function conv1d(input, weights, bias, k, len) { // input: array of channels (Float64Array)
  const p = (k - 1) / 2, cin = input.length, cout = weights.length;
  const out = [];
  for (let co = 0; co < cout; co++) {
    const o = new Float64Array(len).fill(bias[co]);
    for (let ci = 0; ci < cin; ci++) {
      const w = weights[co][ci], v = input[ci];
      for (let i = 0; i < len; i++) {
        let s = 0;
        for (let j = 0; j < k; j++) {
          const idx = i + j - p;
          if (idx >= 0 && idx < len) s += w[j] * v[idx];
        }
        o[i] += s;
      }
    }
    out.push(o);
  }
  return out;
}

function runStack(layers, input, len) {
  let act = [input];
  for (let li = 0; li < layers.length; li++) {
    act = conv1d(act, layers[li].w, layers[li].b, layers[li].k, len);
    if (li < layers.length - 1) for (const ch of act) for (let i = 0; i < len; i++) ch[i] = Math.max(0, ch[i]);
  }
  return act[0];
}

function applyNPO(r) {
  const nr = nrm(r);
  if (nr === 0) return new Float64Array(n);
  const rn = Float64Array.from(r, v => v / nr);
  const nc = NPO_WEIGHTS.nc, pool = n / nc;
  const fine = runStack(NPO_WEIGHTS.fine, rn, n);
  const pooled = new Float64Array(nc);
  for (let j = 0; j < nc; j++) {
    let s = 0;
    for (let i = 0; i < pool; i++) s += rn[j * pool + i];
    pooled[j] = s / pool;
  }
  const coarse = runStack(NPO_WEIGHTS.coarse, pooled, nc);
  const z = new Float64Array(n);
  for (let i = 0; i < n; i++) { // align-corners=false linear upsample of the coarse branch
    const pos = (i + 0.5) * nc / n - 0.5;
    let j0 = Math.floor(pos);
    let w = Math.min(1, Math.max(0, pos - j0));
    const a = coarse[Math.min(nc - 1, Math.max(0, j0))];
    const bb = coarse[Math.min(nc - 1, Math.max(0, j0 + 1))];
    z[i] = (fine[i] + (1 - w) * a + w * bb) * nr;
  }
  return z;
}

// ---------------- Solver steppers ----------------
// Each stepper exposes x, r, dx buffers and step() -> relative residual after the step.

function makeCG() {
  const x = new Float64Array(n), r = Float64Array.from(b), p = Float64Array.from(b);
  const Ap = new Float64Array(n), dx = new Float64Array(n);
  let rr = dot(r, r);
  return { x, r, dx, step() {
    applyA(p, Ap);
    const alpha = rr / dot(p, Ap);
    for (let i = 0; i < n; i++) { dx[i] = alpha * p[i]; x[i] += dx[i]; r[i] -= alpha * Ap[i]; }
    const rrNew = dot(r, r), beta = rrNew / rr;
    rr = rrNew;
    for (let i = 0; i < n; i++) p[i] = r[i] + beta * p[i];
    return Math.sqrt(rrNew) / bNorm;
  } };
}

// CG with the neural preconditioner: flexible CG (Polak-Ribiere beta), since
// the ReLU network is nonlinear and plain PCG's conjugacy assumptions fail.
// Descent safeguard (mirrors train_npo_1d.py): if the net's output is not a
// descent direction, fall back to the raw residual for that step.
function safeguardedNPO(r) {
  const z = applyNPO(r);
  if (dot(r, z) <= 1e-14 * nrm(r) * nrm(z)) return Float64Array.from(r);
  return z;
}

function makeNeuralCG() {
  const x = new Float64Array(n), r = Float64Array.from(b);
  const Ap = new Float64Array(n), dx = new Float64Array(n);
  let z = safeguardedNPO(r), p = Float64Array.from(z), rz = dot(r, z);
  return { x, r, dx, step() {
    applyA(p, Ap);
    const alpha = rz / dot(p, Ap);
    for (let i = 0; i < n; i++) { dx[i] = alpha * p[i]; x[i] += dx[i]; r[i] -= alpha * Ap[i]; }
    const rel = nrm(r) / bNorm;
    z = safeguardedNPO(r);
    let zdr = 0; // z . (r_new - r_old) = z . (-alpha * Ap)
    for (let i = 0; i < n; i++) zdr -= z[i] * alpha * Ap[i];
    const beta = zdr / rz;
    rz = dot(r, z);
    for (let i = 0; i < n; i++) p[i] = z[i] + beta * p[i];
    return rel;
  } };
}

function makeGradient() { // steepest descent with exact line search
  const x = new Float64Array(n), r = Float64Array.from(b);
  const Ar = new Float64Array(n), dx = new Float64Array(n);
  return { x, r, dx, step() {
    applyA(r, Ar);
    const alpha = dot(r, r) / dot(r, Ar);
    for (let i = 0; i < n; i++) { dx[i] = alpha * r[i]; x[i] += dx[i]; }
    for (let i = 0; i < n; i++) r[i] -= alpha * Ar[i];
    return nrm(r) / bNorm;
  } };
}

function makeSOR() {
  const w = omegaSor;
  const x = new Float64Array(n), r = Float64Array.from(b);
  const xprev = new Float64Array(n), Ax = new Float64Array(n), dx = new Float64Array(n);
  return { x, r, dx, step() {
    xprev.set(x);
    for (let i = 0; i < n; i++) { // forward sweep
      const left = i > 0 ? x[i - 1] : 0, right = i < n - 1 ? x[i + 1] : 0;
      const gs = (h2 * b[i] + left + right) / 2;
      x[i] = (1 - w) * x[i] + w * gs;
    }
    for (let i = 0; i < n; i++) dx[i] = x[i] - xprev[i];
    applyA(x, Ax);
    let s = 0;
    for (let i = 0; i < n; i++) { r[i] = b[i] - Ax[i]; s += r[i] * r[i]; }
    return Math.sqrt(s) / bNorm;
  } };
}

// ---------------- History runner (two passes: count, then record ~MAXFRAMES frames) ----------------
function runSolver(make) {
  let s = make();
  const resHist = [1.0];
  let rel = 1.0, k = 0;
  while (rel > TOL && k < MAXIT) { rel = s.step(); k++; resHist.push(rel); }
  const iters = k, converged = rel <= TOL;

  const stride = Math.max(1, Math.ceil(iters / MAXFRAMES));
  s = make();
  const frames = [{ iter: 0, x: new Float32Array(n), r: Float32Array.from(b), dx: new Float32Array(n), relres: 1.0 }];
  rel = 1.0; k = 0;
  while (rel > TOL && k < MAXIT) {
    rel = s.step(); k++;
    if (k % stride === 0 || rel <= TOL || k >= MAXIT) {
      frames.push({ iter: k, x: Float32Array.from(s.x), r: Float32Array.from(s.r), dx: Float32Array.from(s.dx), relres: rel });
    }
  }
  return { resHist, frames, iters, converged };
}

// frozen axis ranges over the whole history, so nothing rescales while scrubbing
function padRange(lo, hi) { const s = Math.max(1e-6, hi - lo); return [lo - 0.05 * s, hi + 0.05 * s]; }
function symRange(lo, hi) { const m = Math.max(Math.abs(lo), Math.abs(hi), 1e-6); return [-1.05 * m, 1.05 * m]; }

function frameRanges(run) {
  let xLo = Infinity, xHi = -Infinity, rM = 0, dM = 0;
  for (const f of run.frames) {
    for (let i = 0; i < n; i++) {
      const v = f.x[i]; if (v < xLo) xLo = v; if (v > xHi) xHi = v;
      const a = Math.abs(f.r[i]); if (a > rM) rM = a;
      const d = Math.abs(f.dx[i]); if (d > dM) dM = d;
    }
  }
  for (let i = 0; i < n; i++) { const v = xExact[i]; if (v < xLo) xLo = v; if (v > xHi) xHi = v; }
  return { u: padRange(xLo, xHi), r: symRange(-rM, rM), dx: symRange(-dM, dM) };
}

function relErrVsExact(x) {
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) { num += (x[i] - xExact[i]) ** 2; den += xExact[i] ** 2; }
  return Math.sqrt(num / den);
}

function decimate(resHist) { // log-x sampled polyline for the tracker (starts at k=1)
  const L = resHist.length, ks = [], rs = [];
  const NPTS = 900, e1 = Math.log10(Math.max(2, L - 1));
  let prev = 0;
  for (let j = 0; j <= NPTS; j++) {
    const k = Math.max(1, Math.min(L - 1, Math.round(Math.pow(10, (j / NPTS) * e1))));
    if (k !== prev) { ks.push(k); rs.push(resHist[k]); prev = k; }
  }
  return { ks, rs };
}

// ---------------- Solver registry ----------------
const SOLVERS = [
  { id: 'cg', name: 'Conjugate gradient', color: '#1f77b4', make: makeCG, params: 'no preconditioner' },
  { id: 'npo', name: 'CG + neural preconditioner', color: '#d62728', make: makeNeuralCG, params: 'toy NPO (1,154 weights), flexible CG' },
  { id: 'sor', name: 'SOR', color: '#9467bd', make: makeSOR, params: 'ω = ' + omegaSor.toFixed(4) + ' (optimal)' },
  { id: 'grad', name: 'Gradient descent', color: '#2ca02c', make: makeGradient, params: 'steepest descent, exact line search' },
];

const state = {
  selected: 'cg',
  frameIdx: {},
  playing: false,
  timer: null,
  runs: {},
};

// ---------------- Plotting ----------------
function linTicks(a, bb, m) {
  const span = bb - a, raw = span / (m - 1);
  const pow = Math.pow(10, Math.floor(Math.log10(raw)));
  const st = [1, 2, 2.5, 5, 10].find(s => s * pow >= raw) * pow;
  const t = [];
  for (let v = Math.ceil(a / st) * st; v <= bb + 1e-9 * span; v += st) t.push(v);
  return t;
}
function logTicks(a, bb) {
  const e0 = Math.ceil(Math.log10(a) - 1e-9), e1 = Math.floor(Math.log10(bb) + 1e-9);
  const st = Math.max(1, Math.ceil((e1 - e0) / 7)), t = [];
  for (let e = e0; e <= e1; e += st) t.push(Math.pow(10, e));
  return t;
}
function fmtTick(v, isLog) {
  if (v === 0) return '0';
  const av = Math.abs(v);
  if (isLog || av >= 1e4 || av < 1e-3) {
    const e = Math.round(Math.log10(av));
    if (isLog && Math.abs(av - Math.pow(10, e)) / av < 1e-6) return '1e' + e;
    return v.toExponential(0);
  }
  return String(parseFloat(v.toPrecision(3)));
}

function drawPlot(canvas, o) {
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth, H = canvas.clientHeight;
  if (canvas.width !== Math.round(W * dpr) || canvas.height !== Math.round(H * dpr)) {
    canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  const ml = 60, mr = 12, mt = 30, mb = 40, pw = W - ml - mr, ph = H - mt - mb;
  const TXf = o.xLog ? Math.log10 : (v => v), TYf = o.yLog ? Math.log10 : (v => v);
  const x0 = TXf(o.xMin), x1 = TXf(o.xMax), y0 = TYf(o.yMin), y1 = TYf(o.yMax);
  const px = v => ml + (TXf(v) - x0) / (x1 - x0) * pw;
  const py = v => mt + ph - (TYf(v) - y0) / (y1 - y0) * ph;

  ctx.font = '11px system-ui, sans-serif';
  const xt = o.xLog ? logTicks(o.xMin, o.xMax) : linTicks(o.xMin, o.xMax, 5);
  const yt = o.yLog ? logTicks(o.yMin, o.yMax) : linTicks(o.yMin, o.yMax, 5);
  ctx.strokeStyle = '#e4e4ea'; ctx.lineWidth = 1;
  for (const v of xt) { const X_ = px(v); ctx.beginPath(); ctx.moveTo(X_, mt); ctx.lineTo(X_, mt + ph); ctx.stroke(); }
  for (const v of yt) { const Y_ = py(v); ctx.beginPath(); ctx.moveTo(ml, Y_); ctx.lineTo(ml + pw, Y_); ctx.stroke(); }
  ctx.fillStyle = '#555'; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  for (const v of xt) ctx.fillText(fmtTick(v, o.xLog), px(v), mt + ph + 5);
  ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
  for (const v of yt) ctx.fillText(fmtTick(v, o.yLog), ml - 6, py(v));
  ctx.strokeStyle = '#9a9aa2'; ctx.strokeRect(ml, mt, pw, ph);

  ctx.save(); ctx.beginPath(); ctx.rect(ml, mt, pw, ph); ctx.clip();
  for (const s of (o.series || [])) {
    ctx.globalAlpha = s.alpha == null ? 1 : s.alpha;
    ctx.strokeStyle = s.color; ctx.lineWidth = s.width || 1.6;
    ctx.setLineDash(s.dash || []);
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < s.xs.length; i++) {
      let vy = s.ys[i];
      if (o.yLog) vy = Math.max(vy, 1e-16);
      const Xp = px(s.xs[i]), Yp = py(vy);
      if (!started) { ctx.moveTo(Xp, Yp); started = true; } else ctx.lineTo(Xp, Yp);
    }
    ctx.stroke();
    ctx.setLineDash([]); ctx.globalAlpha = 1;
  }
  if (o.marker) {
    ctx.fillStyle = o.marker.color || '#e11';
    ctx.beginPath(); ctx.arc(px(o.marker.x), py(Math.max(o.marker.y, o.yLog ? 1e-16 : -Infinity)), 5, 0, 2 * Math.PI); ctx.fill();
  }
  ctx.restore();

  ctx.fillStyle = '#1b1b20'; ctx.font = '600 13px system-ui, sans-serif';
  ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
  ctx.fillText(o.title, ml, 19);
  if (o.xLabel) {
    ctx.fillStyle = '#555'; ctx.font = '11px system-ui, sans-serif'; ctx.textAlign = 'center';
    ctx.fillText(o.xLabel, ml + pw / 2, H - 8);
  }
  if (o.yLabel) {
    ctx.save(); ctx.translate(12, mt + ph / 2); ctx.rotate(-Math.PI / 2);
    ctx.fillStyle = '#555'; ctx.font = '11px system-ui, sans-serif'; ctx.textAlign = 'center';
    ctx.fillText(o.yLabel, 0, 0); ctx.restore();
  }
}

// ---------------- UI ----------------
const $ = id => document.getElementById(id);

function currentRun() { return state.runs[state.selected]; }
function currentFrame() {
  const run = currentRun();
  const idx = Math.min(state.frameIdx[state.selected] || 0, run.frames.length - 1);
  return run.frames[idx];
}

function draw() {
  const run = currentRun(), f = currentFrame();
  const sv = SOLVERS.find(s => s.id === state.selected);
  const stepLabel = ' — step ' + f.iter.toLocaleString('en-US');
  drawPlot($('plotU'), {
    title: 'Temperature u' + stepLabel, xLabel: 'x', yLabel: 'u',
    xMin: 0, xMax: 1, yMin: run.ranges.u[0], yMax: run.ranges.u[1],
    series: [
      { xs: X, ys: xExact, color: '#b6b6be', width: 1.4, dash: [5, 4] },
      { xs: X, ys: f.x, color: sv.color, width: 2.2 },
    ],
  });
  drawPlot($('plotR'), {
    title: 'Residual r' + stepLabel, xLabel: 'x', yLabel: 'r',
    xMin: 0, xMax: 1, yMin: run.ranges.r[0], yMax: run.ranges.r[1],
    series: [{ xs: X, ys: f.r, color: '#c02424', width: 1.7 }],
  });
  drawPlot($('plotD'), {
    title: 'Step taken Δu' + stepLabel, xLabel: 'x', yLabel: 'Δu',
    xMin: 0, xMax: 1, yMin: run.ranges.dx[0], yMax: run.ranges.dx[1],
    series: [{ xs: X, ys: f.dx, color: '#1c7c33', width: 1.7 }],
  });

  let minRel = 1, maxIter = 1;
  const series = [];
  for (const s of SOLVERS) {
    const r = state.runs[s.id];
    if (!r) continue;
    minRel = Math.min(minRel, r.resHist[r.resHist.length - 1]);
    maxIter = Math.max(maxIter, r.iters);
    const sel = s.id === state.selected;
    series.push({ xs: r.dec.ks, ys: r.dec.rs, color: s.color, width: sel ? 2.6 : 1.3, alpha: sel ? 1 : 0.45 });
  }
  drawPlot($('plotC'), {
    title: 'Convergence tracker (all solvers)', xLabel: 'iteration step', yLabel: 'relative residual',
    xMin: 1, xMax: Math.max(2, maxIter), yMin: Math.max(minRel * 0.5, 3e-6), yMax: 1.6,
    xLog: true, yLog: true,
    series, marker: { x: Math.max(1, f.iter), y: f.relres },
  });

  const idx = Math.min(state.frameIdx[state.selected] || 0, run.frames.length - 1);
  $('slider').max = run.frames.length - 1;
  $('slider').value = idx;
  $('stepLabel').innerHTML = 'iteration <b>' + f.iter.toLocaleString('en-US') + '</b> / ' +
    run.iters.toLocaleString('en-US') + ' &nbsp;·&nbsp; rel. residual <b>' + f.relres.toExponential(2) + '</b>';
}

function rebuildTable() {
  const rows = SOLVERS.map(s => {
    const r = state.runs[s.id];
    const last = r.frames[r.frames.length - 1];
    const sel = s.id === state.selected ? ' style="background:#f2f6ff"' : '';
    return '<tr' + sel + '><td><span class="dot" style="background:' + s.color + '"></span>' + s.name + '</td>' +
      '<td>' + s.params + '</td><td class="num">' + r.iters.toLocaleString('en-US') + '</td>' +
      '<td class="num">' + r.resHist[r.resHist.length - 1].toExponential(2) + '</td>' +
      '<td class="num">' + (100 * relErrVsExact(last.x)).toFixed(1) + '%</td></tr>';
  }).join('');
  $('summaryBody').innerHTML = rows;
}

function selectSolver(id) {
  state.selected = id;
  for (const s of SOLVERS) $('btn-' + s.id).classList.toggle('active', s.id === id);
  rebuildTable();
  draw();
}

function stopPlay() {
  state.playing = false;
  if (state.timer) { clearInterval(state.timer); state.timer = null; }
  $('playBtn').textContent = '▶ play';
}
function togglePlay() {
  if (state.playing) { stopPlay(); return; }
  state.playing = true;
  $('playBtn').textContent = '❚❚ pause';
  const run = currentRun();
  if ((state.frameIdx[state.selected] || 0) >= run.frames.length - 1) state.frameIdx[state.selected] = 0;
  const perTick = Math.max(1, Math.round(run.frames.length / 120)); // ~8 s per full animation
  state.timer = setInterval(() => {
    const r = currentRun();
    let idx = (state.frameIdx[state.selected] || 0) + perTick;
    if (idx >= r.frames.length - 1) { idx = r.frames.length - 1; stopPlay(); }
    state.frameIdx[state.selected] = idx;
    draw();
  }, 70);
}

async function init() {
  const status = $('status');
  for (const s of SOLVERS) {
    status.textContent = 'computing ' + s.name + '…';
    await new Promise(r => setTimeout(r, 15));
    const run = runSolver(s.make);
    run.ranges = frameRanges(run);
    run.dec = decimate(run.resHist);
    state.runs[s.id] = run;
    state.frameIdx[s.id] = 0;
  }
  status.textContent = '';
  $('dashboard').style.display = '';

  $('facts').innerHTML =
    'n = ' + n + ' cells, h = 1/' + (n + 1) +
    ', κ(A) = ' + Math.round(kappa).toLocaleString('en-US') +
    ', stop at ‖r‖/‖b‖ ≤ 10⁻⁴';

  for (const s of SOLVERS) $('btn-' + s.id).addEventListener('click', () => { stopPlay(); selectSolver(s.id); });
  $('slider').addEventListener('input', e => { stopPlay(); state.frameIdx[state.selected] = +e.target.value; draw(); });
  $('playBtn').addEventListener('click', togglePlay);
  window.addEventListener('resize', draw);

  selectSolver('cg');
}

init();
