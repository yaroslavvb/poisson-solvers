/* Iterative solvers explorer — 1D Poisson problem.
 *
 * JavaScript port of the Mathematica CG dashboard (mathematica/cg_dashboard_1d.wls):
 * same problem (n=256 1D Dirichlet Laplacian /h^2, unit point sources at nodes 1 and 256,
 * tol 1e-2 on ||r||/||b||), same 2x2 layout (solution u, residual r, step taken du,
 * convergence tracker with current-step marker), same frozen axis ranges computed
 * over the whole iteration history. Adds classical solvers: Jacobi, Richardson,
 * SOR (adjustable omega), gradient descent (steepest descent) — and a second,
 * low-frequency source term ("two charged slabs") for which the classical
 * O(kappa) vs O(sqrt(kappa)) hierarchy shows up at full strength.
 */
'use strict';

// ---------------- Problem setup ----------------
const n = 256;
const h = 1 / (n + 1);
const h2 = h * h;
const TOL = 1e-2;
const MAXIT = 300000;
const MAXFRAMES = 480;

// interior grid coordinates x_i = (i+1) h in (0,1)
const X = new Float64Array(n);
for (let i = 0; i < n; i++) X[i] = (i + 1) * h;

// spectrum of A = tridiag(-1,2,-1)/h^2: lam_k = (2 - 2 cos(k pi/(n+1)))/h^2
const theta = Math.PI / (n + 1);
const lamMin = (2 - 2 * Math.cos(theta)) / h2;
const lamMax = (2 + 2 * Math.cos(theta)) / h2;
const kappa = lamMax / lamMin;
const omegaJacobi = h2 / 2;            // D^{-1}; equals optimal Richardson 2/(lamMin+lamMax)
const omegaRich = 1 / lamMax;          // conservative fixed step (for contrast with Jacobi)
const omegaSorOpt = 2 / (1 + Math.sin(theta));

function dot(u, v) { let s = 0; for (let i = 0; i < u.length; i++) s += u[i] * v[i]; return s; }
function nrm(u) { return Math.sqrt(dot(u, u)); }

// A v with A = tridiag(-1, 2, -1)/h^2 (Dirichlet: v_0 = v_{n+1} = 0)
function applyA(v, out) {
  for (let i = 0; i < n; i++) {
    out[i] = (2 * v[i] - (i > 0 ? v[i - 1] : 0) - (i < n - 1 ? v[i + 1] : 0)) / h2;
  }
}

// right-hand side: mutable, two source configurations
const b = new Float64Array(n);
let bNorm = 1;
let xExact = new Float64Array(n);

function setRHS(kind) {
  b.fill(0);
  if (kind === 'points') {
    // faithful to the Mathematica original (b[[128]] = 0.0 there is a no-op)
    b[0] = 1.0;
    b[127] = 0.0;
    b[255] = -1.0;
  } else { // 'slabs': +1 on (0.2,0.4), -1 on (0.6,0.8) — low-frequency content
    for (let i = 0; i < n; i++) {
      if (X[i] > 0.2 && X[i] < 0.4) b[i] = 1.0;
      else if (X[i] > 0.6 && X[i] < 0.8) b[i] = -1.0;
    }
  }
  bNorm = nrm(b);
  xExact = exactSolve();
}

// exact solution via Thomas algorithm on tridiag(-1,2,-1) x = h^2 b
function exactSolve() {
  const c = new Float64Array(n), d = new Float64Array(n), x = new Float64Array(n);
  c[0] = -1 / 2; d[0] = h2 * b[0] / 2;
  for (let i = 1; i < n; i++) {
    const m = 2 + c[i - 1]; // 2 - (-1)*c[i-1] with sub-diagonal -1
    c[i] = -1 / m;
    d[i] = (h2 * b[i] + d[i - 1]) / m;
  }
  x[n - 1] = d[n - 1];
  for (let i = n - 2; i >= 0; i--) x[i] = d[i] - c[i] * x[i + 1];
  return x;
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

function makeSteepest() {
  const x = new Float64Array(n), r = Float64Array.from(b);
  const Ar = new Float64Array(n), dx = new Float64Array(n);
  return { x, r, dx, step() {
    applyA(r, Ar);
    const alpha = dot(r, r) / dot(r, Ar); // exact line search along -grad = r
    for (let i = 0; i < n; i++) { dx[i] = alpha * r[i]; x[i] += dx[i]; }
    for (let i = 0; i < n; i++) r[i] -= alpha * Ar[i];
    return nrm(r) / bNorm;
  } };
}

// x <- x + c r covers both Jacobi (c = h^2/2 = D^{-1}) and Richardson (c = omega)
function makeFixedStep(c) {
  const x = new Float64Array(n), r = Float64Array.from(b);
  const Ar = new Float64Array(n), dx = new Float64Array(n);
  return { x, r, dx, step() {
    applyA(r, Ar);
    for (let i = 0; i < n; i++) { dx[i] = c * r[i]; x[i] += dx[i]; }
    for (let i = 0; i < n; i++) r[i] -= c * Ar[i];
    return nrm(r) / bNorm;
  } };
}

function makeSOR(w) {
  const x = new Float64Array(n), r = Float64Array.from(b);
  const xprev = new Float64Array(n), Ax = new Float64Array(n), dx = new Float64Array(n);
  return { x, r, dx, step() {
    xprev.set(x);
    for (let i = 0; i < n; i++) { // forward sweep; row: (2x_i - x_{i-1} - x_{i+1})/h^2 = b_i
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

// frozen axis ranges, as in the Mathematica dashboard
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
  // include the exact solution in the u range so the dashed overlay never clips
  for (let i = 0; i < n; i++) { const v = xExact[i]; if (v < xLo) xLo = v; if (v > xHi) xHi = v; }
  return { u: padRange(xLo, xHi), r: symRange(-rM, rM), dx: symRange(-dM, dM) };
}

function relErrVsExact(x) {
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) { num += (x[i] - xExact[i]) ** 2; den += xExact[i] ** 2; }
  return Math.sqrt(num / den);
}

// decimated convergence polylines (linear-x and log-x sampling)
function decimate(resHist) {
  const L = resHist.length;
  const lin = { ks: [], rs: [] };
  const st = Math.max(1, Math.ceil(L / 1600));
  for (let k = 0; k < L; k += st) { lin.ks.push(k); lin.rs.push(resHist[k]); }
  if (lin.ks[lin.ks.length - 1] !== L - 1) { lin.ks.push(L - 1); lin.rs.push(resHist[L - 1]); }
  const lg = { ks: [], rs: [] };
  const N = 900, e1 = Math.log10(Math.max(2, L - 1));
  let prev = 0;
  for (let j = 0; j <= N; j++) {
    const k = Math.min(L - 1, Math.round(Math.pow(10, (j / N) * e1)));
    if (k !== prev || j === 0) { lg.ks.push(Math.max(1, k)); lg.rs.push(resHist[Math.max(1, k)]); prev = k; }
  }
  return { lin, lg };
}

// ---------------- Solver registry ----------------
const SOLVERS = [
  { id: 'cg', name: 'Conjugate gradient', color: '#1f77b4', make: () => makeCG, params: () => 'no preconditioner' },
  { id: 'sor', name: 'SOR', color: '#9467bd', make: () => () => makeSOR(state.sorOmega), params: () => 'ω = ' + state.sorOmega.toFixed(4) + (Math.abs(state.sorOmega - omegaSorOpt) < 5e-4 ? ' (optimal)' : '') },
  { id: 'jacobi', name: 'Jacobi', color: '#d62728', make: () => () => makeFixedStep(omegaJacobi), params: () => 'x ← x + D⁻¹r,  D⁻¹ = (h²/2)I' },
  { id: 'rich', name: 'Richardson', color: '#ff7f0e', make: () => () => makeFixedStep(omegaRich), params: () => 'ω = 1/λmax = ' + omegaRich.toExponential(2) },
  { id: 'sd', name: 'Gradient descent', color: '#2ca02c', make: () => makeSteepest, params: () => 'steepest descent, exact line search' },
];

const state = {
  selected: 'cg',
  rhs: 'points',
  sorOmega: Math.round(omegaSorOpt * 1000) / 1000,
  frameIdx: {},   // per solver id
  playing: false,
  timer: null,
  speed: 4,       // frames per tick
  trackerLogX: true,
  runs: {},       // id -> {resHist, frames, iters, converged, ranges, dec}
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
      const vx = s.xs[i]; let vy = s.ys[i];
      if (o.yLog) vy = Math.max(vy, 1e-16);
      if (o.xLog && vx <= 0) continue;
      const Xp = px(vx), Yp = py(vy);
      if (!started) { ctx.moveTo(Xp, Yp); started = true; } else ctx.lineTo(Xp, Yp);
    }
    ctx.stroke();
    ctx.setLineDash([]); ctx.globalAlpha = 1;
  }
  if (o.marker) {
    ctx.fillStyle = o.marker.color || '#e11';
    const mx = o.xLog ? Math.max(o.marker.x, o.xMin) : o.marker.x;
    ctx.beginPath(); ctx.arc(px(mx), py(Math.max(o.marker.y, o.yLog ? 1e-16 : -Infinity)), 5, 0, 2 * Math.PI); ctx.fill();
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
    title: 'Solved field u' + stepLabel, xLabel: 'x', yLabel: 'u',
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

  // convergence tracker: all solvers, selected highlighted
  let minRel = 1, maxIter = 1;
  const series = [];
  for (const s of SOLVERS) {
    const r = state.runs[s.id];
    if (!r) continue;
    minRel = Math.min(minRel, r.resHist[r.resHist.length - 1]);
    maxIter = Math.max(maxIter, r.iters);
    const d = state.trackerLogX ? r.dec.lg : r.dec.lin;
    const sel = s.id === state.selected;
    series.push({ xs: d.ks, ys: d.rs, color: s.color, width: sel ? 2.6 : 1.3, alpha: sel ? 1 : 0.45 });
  }
  drawPlot($('plotC'), {
    title: 'Convergence tracker (all solvers)', xLabel: 'iteration step', yLabel: 'relative residual',
    xMin: state.trackerLogX ? 1 : 0, xMax: Math.max(2, maxIter), yMin: Math.max(minRel * 0.5, 1e-16), yMax: 1.6,
    xLog: state.trackerLogX, yLog: true,
    series, marker: { x: Math.max(f.iter, state.trackerLogX ? 1 : 0), y: f.relres },
  });

  const idx = Math.min(state.frameIdx[state.selected] || 0, run.frames.length - 1);
  $('slider').max = run.frames.length - 1;
  $('slider').value = idx;
  $('stepLabel').innerHTML = 'iteration <b>' + f.iter.toLocaleString('en-US') + '</b> / ' +
    run.iters.toLocaleString('en-US') + ' &nbsp;·&nbsp; rel. residual <b>' + f.relres.toExponential(2) + '</b>' +
    (run.converged ? '' : ' &nbsp;·&nbsp; <span style="color:#c02424">did not reach tol in ' + MAXIT.toLocaleString('en-US') + ' iterations</span>');
}

function rebuildTable() {
  const rows = SOLVERS.map(s => {
    const r = state.runs[s.id];
    const last = r.frames[r.frames.length - 1];
    const it = r.converged ? r.iters.toLocaleString('en-US') : '&gt; ' + MAXIT.toLocaleString('en-US') + ' (no convergence)';
    const sel = s.id === state.selected ? ' style="background:#f2f6ff"' : '';
    return '<tr' + sel + '><td><span class="dot" style="background:' + s.color + '"></span>' + s.name + '</td>' +
      '<td>' + s.params() + '</td><td class="num">' + it + '</td>' +
      '<td class="num">' + r.resHist[r.resHist.length - 1].toExponential(2) + '</td>' +
      '<td class="num">' + (100 * relErrVsExact(last.x)).toFixed(1) + '%</td></tr>';
  }).join('');
  $('summaryBody').innerHTML = rows;
}

function selectSolver(id) {
  state.selected = id;
  for (const s of SOLVERS) $('btn-' + s.id).classList.toggle('active', s.id === id);
  $('sorControls').style.display = id === 'sor' ? 'flex' : 'none';
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
  state.timer = setInterval(() => {
    const r = currentRun();
    let idx = (state.frameIdx[state.selected] || 0) + state.speed;
    if (idx >= r.frames.length - 1) { idx = r.frames.length - 1; stopPlay(); }
    state.frameIdx[state.selected] = idx;
    draw();
  }, 70);
}

function computeSolver(s) {
  const run = runSolver(s.make());
  run.ranges = frameRanges(run);
  run.dec = decimate(run.resHist);
  state.runs[s.id] = run;
  state.frameIdx[s.id] = 0;
}

async function computeAll() {
  const status = $('status');
  for (const s of SOLVERS) {
    status.textContent = 'computing ' + s.name + ' history…';
    await new Promise(r => setTimeout(r, 15)); // let the status paint
    computeSolver(s);
  }
  status.textContent = '';
}

async function init() {
  setRHS(state.rhs);
  await computeAll();
  $('dashboard').style.display = '';

  // problem facts
  $('facts').innerHTML =
    'n = ' + n + ', h = 1/' + (n + 1) +
    ', κ(A) = λ<sub>max</sub>/λ<sub>min</sub> = ' + Math.round(kappa).toLocaleString('en-US') +
    ', tol = 10⁻², SOR ω<sub>opt</sub> = 2/(1+sin πh) ≈ ' + omegaSorOpt.toFixed(4);

  // controls
  for (const s of SOLVERS) $('btn-' + s.id).addEventListener('click', () => { stopPlay(); selectSolver(s.id); });
  $('slider').addEventListener('input', e => { stopPlay(); state.frameIdx[state.selected] = +e.target.value; draw(); });
  $('playBtn').addEventListener('click', togglePlay);
  $('speedSel').addEventListener('change', e => { state.speed = +e.target.value; });
  $('xScaleSel').addEventListener('change', e => { state.trackerLogX = e.target.value === 'log'; draw(); });
  $('rhsSel').addEventListener('change', async e => {
    stopPlay();
    state.rhs = e.target.value;
    setRHS(state.rhs);
    await computeAll();
    rebuildTable(); draw();
  });
  const omegaIn = $('sorOmega'), omegaVal = $('sorOmegaVal');
  omegaIn.value = state.sorOmega;
  omegaVal.textContent = state.sorOmega.toFixed(3);
  omegaIn.addEventListener('input', e => { omegaVal.textContent = (+e.target.value).toFixed(3); });
  omegaIn.addEventListener('change', e => {
    stopPlay();
    state.sorOmega = +e.target.value;
    $('status').textContent = 'recomputing SOR…';
    setTimeout(() => {
      computeSolver(SOLVERS.find(s => s.id === 'sor'));
      $('status').textContent = '';
      rebuildTable(); draw();
    }, 15);
  });
  $('sorOptBtn').addEventListener('click', () => {
    omegaIn.value = Math.round(omegaSorOpt * 1000) / 1000;
    omegaIn.dispatchEvent(new Event('input'));
    omegaIn.dispatchEvent(new Event('change'));
  });
  window.addEventListener('resize', draw);

  selectSolver('cg');
}

init();
