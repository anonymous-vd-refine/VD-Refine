'use strict';
const chart = document.getElementById('depth-chart');
const select = document.getElementById('depth-select');
const points = depthData.filter(d => d.k !== 'inf');
const x = i => 66 + i * 93;
const y = v => 274 - (v - 83) / 2.6 * 232;
function draw(selected) {
  let svg = '<title>Measured BraTS depth scaling</title><desc>Mean Dice rises from 83.227 at K zero to 85.258 at K 32, with a dip at K two. Analytic decoding obtains 85.239.</desc>';
  svg += '<rect x="438" y="20" width="210" height="254" fill="#f4f1fa"/><text x="447" y="36" fill="#756491" font-size="12">Beyond real training depth</text>';
  [83,83.5,84,84.5,85,85.5].forEach(v => { svg += `<line x1="58" x2="648" y1="${y(v)}" y2="${y(v)}" stroke="#e9e5ef"/><text x="47" y="${y(v)+4}" text-anchor="end" fill="#6c637c" font-size="13">${v.toFixed(1)}</text>`; });
  [{v:84.73,c:'#809184',label:'No Refiner 84.73'},{v:84.59,c:'#a49785',label:'nnU-Net 84.59'}].forEach(d => {svg += `<line x1="58" x2="648" y1="${y(d.v)}" y2="${y(d.v)}" stroke="${d.c}" stroke-dasharray="5 5"/>`;});
  svg += `<polyline points="${points.map((d,i)=>`${x(i)},${y(d.avg)}`).join(' ')}" fill="none" stroke="#5143ba" stroke-width="3" stroke-linejoin="round"/>`;
  points.forEach((d,i) => {let active = d.k===selected; svg += `<circle cx="${x(i)}" cy="${y(d.avg)}" r="${active?8:5}" fill="${active?'#5143ba':'white'}" stroke="#5143ba" stroke-width="2"><title>K = ${d.k}: ${d.avg.toFixed(3)}% mean Dice</title></circle><text x="${x(i)}" y="297" text-anchor="middle" fill="#554963" font-size="14">${d.k}</text>`;});
  if(selected==='inf'){const d=depthData.find(d=>d.k==='inf');svg+=`<line x1="58" x2="648" y1="${y(d.avg)}" y2="${y(d.avg)}" stroke="#237b6d" stroke-width="2" stroke-dasharray="3 4"/><text x="76" y="${y(d.avg)-12}" fill="#237b6d" font-size="14">8 + analytic: ${d.avg.toFixed(3)}%</text>`;}
  svg += '<text x="355" y="329" text-anchor="middle" fill="#6c637c" font-size="13">Real refinement steps K</text>';
  chart.innerHTML=svg;
}
function update(){const d=depthData.find(d=>d.k===select.value);document.getElementById('selected-dice').innerHTML=d.avg.toFixed(2)+'<span>%</span>';document.getElementById('mode-label').textContent=d.k==='inf'?'8 + analytic estimate':`K = ${d.k} · real rollout`;document.getElementById('mode-note').textContent=d.k==='inf'?'Eight real updates, geometric extrapolation, and one decode. No infinite rollout is executed.':d.k==='0'?'Decode the initial proposal without applying the shared Refiner.':`${d.k} shared Refiner ${d.k==='1'?'update':'updates'}, followed by one final decode.`;['wt','tc','et'].forEach(k=>document.getElementById(k).textContent=d[k].toFixed(2)+'%');draw(d.k);}
select.addEventListener('change',update);update();
