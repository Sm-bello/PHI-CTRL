<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>PHI-CTRL — post</title>
<style>
  body {
    background: #0d1117;
    margin: 0;
    padding: 40px 20px;
    font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 24px;
  }
  .term {
    width: 100%;
    max-width: 720px;
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    overflow: hidden;
  }
  .term-bar {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 10px 14px;
    border-bottom: 1px solid #30363d;
  }
  .dot { width: 10px; height: 10px; border-radius: 50%; }
  .r { background: #ff5f56; }
  .y { background: #ffbd2e; }
  .g { background: #27c93f; }
  .term-label {
    margin-left: 8px;
    font-size: 12px;
    color: #8b949e;
  }
  #term {
    padding: 18px;
    font-size: 13px;
    line-height: 1.75;
    white-space: pre-wrap;
    min-height: 340px;
  }
  .img-wrap {
    width: 100%;
    max-width: 720px;
    background: #ffffff;
    border-radius: 10px;
    overflow: hidden;
  }
  .img-wrap img {
    width: 100%;
    display: block;
  }
</style>
</head>
<body>

<div class="term">
  <div class="term-bar">
    <span class="dot r"></span>
    <span class="dot y"></span>
    <span class="dot g"></span>
    <span class="term-label">phi-ctrl — main — zsh</span>
  </div>
  <div id="term"></div>
</div>

<div class="img-wrap">
  <img width="1693" height="929" alt="ChatGPT Image Aug 23, 2026, 08_30_19 AM" src="https://github.com/user-attachments/assets/28f9061f-022e-4da6-b03b-4531e5c17c80" />

</div>

<script>
const lines = [
  {t:"$ git log --oneline -5", c:"#c9d1d9"},
  {t:"a5f2c91 fix(control): correct sign error in phugoid damping term", c:"#3fb950"},
  {t:"8e14d0b tune(gains): retune PID for C172x light-aircraft plant", c:"#3fb950"},
  {t:"3bc902a feat(lateral): add roll/rudder coupling to prevent spiral div.", c:"#3fb950"},
  {t:"f001abc test(jsbsim): validate 40% elevator loss, 6-DOF, full coupling", c:"#3fb950"},
  {t:"", c:""},
  {t:"$ python run_verification.py --layer=3 --plant=c172x --fault=elevator_loss", c:"#c9d1d9"},
  {t:"[LAYER 1] unit + deterministic replay ......... PASS", c:"#8b949e"},
  {t:"[LAYER 2] monte carlo (n=50, gamma∈[0.3,0.8]) .. PASS", c:"#8b949e"},
  {t:"[LAYER 3] jsbsim 6-dof, cessna c172x .......... PASS", c:"#8b949e"},
  {t:"", c:""},
  {t:"baseline_pid   : alt_dev=106.48ft  rec_time=16.00s", c:"#f85149"},
  {t:"phi_ctrl_hybrid: alt_dev=48.30ft   rec_time=5.20s", c:"#3fb950"},
  {t:"improvement    : -55% deviation, -68% recovery time", c:"#58a6ff"},
  {t:"", c:""},
  {t:"$ echo $STATUS", c:"#c9d1d9"},
  {t:"> the plane holds steady.", c:"#58a6ff"},
  {t:"", c:""},
  {t:"$ git commit -m \"paper submission loading\" 🛫", c:"#8b949e"}
];

const el = document.getElementById('term');
let li = 0;

function typeLine(){
  if(li >= lines.length) return;
  const {t, c} = lines[li];
  const row = document.createElement('div');
  row.style.color = c || '#8b949e';
  el.appendChild(row);
  let ci = 0;
  const speed = t.length > 0 ? 12 : 0;
  function typeChar(){
    if(ci <= t.length){
      row.textContent = t.slice(0, ci);
      ci++;
      setTimeout(typeChar, speed);
    } else {
      li++;
      setTimeout(typeLine, 90);
    }
  }
  typeChar();
}
typeLine();
</script>

</body>
</html>
