'use strict';
const chart=document.getElementById('depth-chart');
const select=document.getElementById('depth-select');
let dataset='brats';
const x=i=>66+i*93;
function draw(selected){
 const b=benchmarkData[dataset],points=b.rows.filter(d=>d.k!=='inf'),y=v=>274-(v-b.range[0])/(b.range[1]-b.range[0])*232;
 let svg=`<title>${b.name}: measured depth scaling</title><desc>${b.cohort}. The table below lists every exact measurement.</desc><rect x="438" y="20" width="210" height="254" fill="#f4f1fa"/><text x="447" y="36" fill="#756491" font-size="12">Beyond real training depth</text>`;
 b.ticks.forEach(v=>svg+=`<line x1="58" x2="648" y1="${y(v)}" y2="${y(v)}" stroke="#e9e5ef"/><text x="47" y="${y(v)+4}" text-anchor="end" fill="#6c637c" font-size="13">${v.toFixed(1)}</text>`);
 b.baseline.forEach((v,i)=>svg+=`<line x1="58" x2="648" y1="${y(v)}" y2="${y(v)}" stroke="${i?'#a49785':'#809184'}" stroke-dasharray="5 5"/>`);
 svg+=`<polyline points="${points.map((d,i)=>`${x(i)},${y(d.avg)}`).join(' ')}" fill="none" stroke="#5143ba" stroke-width="3" stroke-linejoin="round"/>`;
 points.forEach((d,i)=>svg+=`<circle cx="${x(i)}" cy="${y(d.avg)}" r="${d.k===selected?8:5}" fill="${d.k===selected?'#5143ba':'white'}" stroke="#5143ba" stroke-width="2"><title>K = ${d.k}: ${d.avg.toFixed(3)}% mean Dice</title></circle><text x="${x(i)}" y="297" text-anchor="middle" fill="#554963" font-size="14">${d.k}</text>`);
 if(selected==='inf'){const d=b.rows.find(d=>d.k==='inf');svg+=`<line x1="58" x2="648" y1="${y(d.avg)}" y2="${y(d.avg)}" stroke="#237b6d" stroke-width="2" stroke-dasharray="3 4"/><text x="76" y="${y(d.avg)+18}" fill="#237b6d" font-size="14">8 + analytic: ${d.avg.toFixed(3)}%</text>`;}
 chart.innerHTML=svg+'<text x="355" y="329" text-anchor="middle" fill="#6c637c" font-size="13">Real refinement steps K</text>';
}
function update(){
 const b=benchmarkData[dataset],d=b.rows.find(d=>d.k===select.value);
 document.getElementById('selected-dice').innerHTML=d.avg.toFixed(2)+'<span>%</span>';
 document.getElementById('mode-label').textContent=d.k==='inf'?'8 + analytic estimate':`K = ${d.k} · real rollout`;
 document.getElementById('mode-note').textContent=d.k==='inf'?'Eight real updates, geometric extrapolation, and one decode.':d.k==='0'?'Decode the initial proposal without applying the shared Refiner.':`${d.k} shared Refiner ${d.k==='1'?'update':'updates'}, followed by one final decode.`;
 document.getElementById('region-metrics').innerHTML=Object.entries(d.regions).map(([n,v])=>`<div><dt>${n}</dt><dd>${v.toFixed(2)}%</dd></div>`).join('');
 document.getElementById('chart-title').textContent=b.name+' depth scaling';
 document.getElementById('benchmark-protocol').textContent=b.cohort+' · '+b.protocol;
 document.getElementById('baseline-backbone').textContent='No Refiner · '+b.baseline[0].toFixed(2);
 document.getElementById('baseline-nnunet').textContent='nnU-Net · '+b.baseline[1].toFixed(2);
 document.getElementById('dataset-note').textContent=b.note;
 document.getElementById('dataset-download').href=dataset+'-depth-results.csv';
 document.getElementById('measurement-table').innerHTML=`<table><caption>${b.name}: complete held-out measurements · Dice (%)</caption><thead><tr><th scope="col">Inference</th>${b.regions.map(n=>`<th scope="col">${n}</th>`).join('')}<th scope="col">Mean</th></tr></thead><tbody>${b.rows.map(r=>`<tr><th scope="row">${r.k==='inf'?'8 + analytic':'K = '+r.k}</th>${b.regions.map(n=>`<td>${r.regions[n].toFixed(4)}</td>`).join('')}<td>${r.avg.toFixed(4)}</td></tr>`).join('')}</tbody></table>`;
 document.querySelectorAll('[data-dataset]').forEach(el=>el.setAttribute('aria-pressed',String(el.dataset.dataset===dataset)));draw(d.k);
}
document.querySelectorAll('[data-dataset]').forEach(el=>el.addEventListener('click',()=>{dataset=el.dataset.dataset;update();}));
select.addEventListener('change',update);update();
