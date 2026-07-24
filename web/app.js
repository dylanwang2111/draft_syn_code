"use strict";
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];

/* Per-TAB session id: sessionStorage is scoped to this tab (not shared with
   other tabs), so each tab gets its own server-side data + job. It survives a
   reload so results persist across F5. Every API call carries it. */
const SID=(()=>{ let s=null;
  try{ s=sessionStorage.getItem("synthlab-sid"); }catch(e){}
  if(!s){ s=(window.crypto&&crypto.randomUUID)?crypto.randomUUID():("t"+Date.now()+Math.random().toString(16).slice(2));
    try{ sessionStorage.setItem("synthlab-sid",s); }catch(e){} }
  return s; })();
function apiFetch(url,opts={}){
  const o={...opts}; o.headers={...(opts.headers||{}),"X-Session-Id":SID};
  return fetch(url,o);
}
const SDTYPES=["categorical","numerical","datetime","boolean","id","unknown"];
const SYNTHS=["HMA","GaussianCopula","CTGAN","TVAE","CopulaGAN"];
const GAN_SYNTHS=new Set(["CTGAN","TVAE","CopulaGAN"]);
const PALETTE={real:"#555f5c",HMA:"#1f77b4",GaussianCopula:"#2ca02c",CTGAN:"#d62728",TVAE:"#9467bd",CopulaGAN:"#ff7f0e"};
let DATA=null, detected={}, selectedTable=null;
let selectedSynths=new Set(["HMA","GaussianCopula"]);

const fmt=(v,d=3)=>(v==null||Number.isNaN(v))?"—":(+v).toFixed(d);
const pct=v=>(v==null)?"—":(100*v).toFixed(1)+"%";
const meterColor=v=>`hsl(${Math.max(0,Math.min(1,v))*105},58%,45%)`;
const pill=s=>`<span class="pill ${s}">${s}</span>`;
const esc=s=>String(s).replace(/</g,"&lt;");
const setStatus=(c,t)=>{const d=$("#status-dot");d.className="status-dot"+(c?" "+c:"");d.title=t||c||"idle";};

/* ---------------- theme ---------------- */
function applyTheme(t){
  document.documentElement.setAttribute("data-theme",t);
  $("#btn-theme").textContent = t==="dark" ? "☀" : "☾";
  $("#btn-theme").title = t==="dark" ? "Switch to light mode" : "Switch to dark mode";
}
(function initTheme(){
  let t=null;
  try{ t=localStorage.getItem("synthlab-theme"); }catch(e){}
  if(!t && window.matchMedia && matchMedia("(prefers-color-scheme: dark)").matches) t="dark";
  applyTheme(t||"light");
})();
$("#btn-theme").addEventListener("click",()=>{
  const next=document.documentElement.getAttribute("data-theme")==="dark"?"light":"dark";
  applyTheme(next);
  try{ localStorage.setItem("synthlab-theme",next); }catch(e){}
  if(typeof restylePlotly==="function") restylePlotly();   // recolour live charts
});

const BACKEND_HELP="Can't reach the backend.\n\nOpen this dashboard THROUGH the server, not as a file:\n\n"
  +"    uvicorn server:app --port 8000\n\nthen visit http://localhost:8000 (address must start with http://).";
if(location.protocol==="file:") setTimeout(()=>alert(BACKEND_HELP),300);

/* ---------------- collapsible left panels ---------------- */
document.querySelector(".left").addEventListener("click",e=>{
  const h=e.target.closest(".panel-h"); if(!h) return;
  h.parentElement.classList.toggle("collapsed");
});

/* ---------------- load data ---------------- */
$("#file-input").addEventListener("change",e=>sendFiles(e.target.files));
$("#btn-sample").addEventListener("click",async()=>{
  $("#btn-sample").textContent="Detecting…";
  try{const r=await apiFetch("/api/sample",{method:"POST"});
    if(r.ok) init(await r.json());
    else alert("No sample data in sdg/seed/ — start the server from the project root.");
  }catch(e){alert(BACKEND_HELP);}
  finally{$("#btn-sample").textContent="Load Sample";}
});
const welcome=$("#welcome");
["dragover","dragenter"].forEach(ev=>welcome.addEventListener(ev,e=>{e.preventDefault();welcome.classList.add("over");}));
["dragleave","drop"].forEach(ev=>welcome.addEventListener(ev,e=>{e.preventDefault();welcome.classList.remove("over");}));
welcome.addEventListener("drop",e=>sendFiles(e.dataTransfer.files));
async function sendFiles(files){
  if(!files||!files.length) return;
  const fd=new FormData(); [...files].forEach(f=>fd.append("files",f));
  $("#welcome").querySelector("h3").textContent="Detecting schema…";
  try{const r=await apiFetch("/api/upload",{method:"POST",body:fd});
    if(r.ok) init(await r.json()); else alert("Upload failed ("+r.status+").");
  }catch(e){alert(BACKEND_HELP);}
  finally{$("#welcome").querySelector("h3").textContent="Drop CSV files here";}
}

function init(payload){
  DATA=payload; detected={};
  ["#structure-panel","#constraint-panel","#pii-panel","#synth-panel","#params-panel"].forEach(id=>$(id).style.display="block");
  renderConstraints();
  $("#synthetic-panel").style.display="none"; $("#synth-list").innerHTML="";
  $("#synth-count").textContent="0"; $("#synth-count").className="count zero";
  lockReportTabs(true);
  renderSeedList(); renderSchemaBlocks(); renderAdvisor(payload.profile); renderPii();
  renderRecipe(); initModel(payload.relationships);
  syncEntity(); renderStructureMirror(); updateSummaries();
  selectTable(Object.keys(DATA.tables)[0]);
  activateTab("pane-schema");
}
/* compact state chips shown in each collapsed panel header */
function updateSummaries(){
  if(!DATA) return;
  const ek=MODEL.hub.key, nrel=MODEL.rels.length;
  $("#sum-structure").textContent = ek ? ("key: "+ek) : nrel ? (nrel+" link"+(nrel>1?"s":"")) : "independent";
  const nc=$$(".con-row").length;
  $("#sum-constraints").textContent = nc ? (nc+" rule"+(nc>1?"s":"")) : "none";
  $("#sum-synths").textContent=[...selectedSynths].join(", ")||"none";
  const sp=$("#sum-pii"); if(sp) sp.textContent=piiSummary();
  $("#sum-params").textContent="×"+(+$("#in-scale").value).toFixed(2).replace(/0$/,"")+" · holdout "+Math.round(100*$("#in-holdout").value)+"%";
}

/* ---------------- left: seed list ---------------- */
function renderSeedList(){
  const rows=Object.entries(DATA.tables).map(([t,v])=>`
    <tr data-t="${t}"><td><div class="fname">${t}.csv</div>
      <div class="fmeta">${v.rows.toLocaleString()} rows · ${v.columns.length} cols</div></td>
      <td class="fact"><button class="iact" data-act="view" title="edit schema">✎</button>
        <button class="iact del" data-act="del" title="remove">✕</button></td></tr>`).join("");
  $("#seed-list").innerHTML=`<table class="ftable"><thead><tr><th>File Name</th><th>Action</th></tr></thead><tbody>${rows}</tbody></table>`;
  const n=Object.keys(DATA.tables).length;
  $("#seed-count").textContent=n; $("#seed-count").className="count"+(n?"":" zero");
  $$("#seed-list tr[data-t]").forEach(tr=>tr.addEventListener("click",e=>{
    const act=e.target.dataset.act, t=tr.dataset.t;
    if(act==="del"){ e.stopPropagation(); removeTable(t); }
    else selectTable(t);
  }));
}
function removeTable(t){
  delete DATA.tables[t];
  if(!Object.keys(DATA.tables).length){ location.reload(); return; }
  // prune the data model of anything referencing the removed table
  MODEL.rels=MODEL.rels.filter(r=>r.parent_table_name!==t&&r.child_table_name!==t);
  MODEL.hub.children=MODEL.hub.children.filter(c=>c!==t);
  if(MODEL.hub.key && !keyTables(MODEL.hub.key).length) MODEL.hub={key:"",children:[]};
  delete MODEL.pos[t];
  renderSeedList(); renderSchemaBlocks(); renderRecipe();
  buildHubBar(); afterModelChange();
  if(selectedTable===t) selectTable(Object.keys(DATA.tables)[0]);
}
function selectTable(t){
  selectedTable=t;
  $$("#seed-list tr[data-t]").forEach(tr=>tr.classList.toggle("sel",tr.dataset.t===t));
  $("#welcome").style.display="none";
  $$(".schema-block").forEach(b=>b.classList.toggle("show",b.dataset.t===t));
  activateTab("pane-schema");
}

/* ---------------- schema editor (right) ---------------- */
function renderSchemaBlocks(){
  const host=$("#schema-blocks"); host.innerHTML="";
  for(const [t,info] of Object.entries(DATA.tables)){
    detected[t]={}; info.columns.forEach(c=>detected[t][c.name]=c.sdtype);
    const rows=info.columns.map(c=>`
      <tr><td class="mono">${c.name}</td>
        <td><select data-table="${t}" data-col="${c.name}" class="sdtype-sel">
          ${SDTYPES.map(s=>`<option ${s===c.sdtype?"selected":""}>${s}</option>`).join("")}</select></td>
        <td class="num">${c.distinct.toLocaleString()}</td><td class="num">${c.missing_pct}%</td>
        <td class="dim mono">${esc(c.example)}</td></tr>`).join("");
    const p=info.preview;
    const prev=`<div class="preview-scroll"><table><thead><tr>${p.columns.map(c=>`<th>${esc(c)}</th>`).join("")}</tr></thead>
      <tbody>${p.data.map(r=>`<tr>${r.map(v=>`<td>${esc(String(v).slice(0,22))}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
    const el=document.createElement("div");
    el.className="schema-block"; el.dataset.t=t;
    el.innerHTML=`<div class="blk-head"><h3>${t}</h3>
        <span class="dim">${info.rows.toLocaleString()} rows · ${info.columns.length} columns</span></div>
      <p class="note">Auto-detected sdtypes — fix any wrong call with the dropdown
        (<span style="color:var(--red)">red</span> = your override). id / datetime / unknown are excluded from privacy &amp; ML metrics.</p>
      <table class="grid"><thead><tr><th>column</th><th>sdtype</th><th>distinct</th><th>missing</th><th>example</th></tr></thead>
      <tbody>${rows}</tbody></table>
      <div class="pk-row">PRIMARY KEY <select id="pk-${t}"><option value="">(none)</option>
        ${info.columns.map(c=>`<option ${c.name===info.primary_key?"selected":""}>${c.name}</option>`).join("")}</select></div>
      ${prev}`;
    host.appendChild(el);
  }
  host.addEventListener("change",e=>{
    if(e.target.classList.contains("sdtype-sel")){
      const {table,col}=e.target.dataset;
      e.target.classList.toggle("changed", e.target.value!==detected[table][col]);
    }
  });
}

/* ===================== data model (PowerBI-style canvas) =====================
   MODEL is the single source of truth for relationships + the entity-key hub.
   The left "Structure & keys" panel and the run config both read from it. */
let MODEL={pos:{}, size:{}, rels:[], hub:{key:"", children:[]}};
const hubName=()=>MODEL.hub.key?`${MODEL.hub.key}_HUB`:"";
function colStat(t,col){ return (DATA.tables[t]&&DATA.tables[t].columns||[]).find(c=>c.name===col); }
function isUnique(t,col){ const c=colStat(t,col); const n=DATA.tables[t]&&DATA.tables[t].rows;
  return !!(c&&n&&c.distinct>=n); }
function sharedKeys(){ const s={};
  for(const v of Object.values(DATA.tables)) for(const c of v.columns) s[c.name]=(s[c.name]||0)+1;
  return Object.keys(s).filter(k=>s[k]>=2).sort(); }
function keyTables(key){ return Object.entries(DATA.tables).filter(([,v])=>v.columns.some(c=>c.name===key)).map(([t])=>t); }

function initModel(seedRels){
  MODEL={pos:{}, size:{}, rels:(seedRels||[]).slice(), hub:{key:"", children:[]}};
  layoutNodes(true);
  buildHubBar();
  renderModel();
  const mt=$('.tab[data-pane="pane-model"]'); mt.classList.remove("locked"); mt.removeAttribute("title");
}
/* ---- canvas layout rules -------------------------------------------------
   1. a card never spawns on top of another: a node with no stored position is
      dropped into the first free slot, measured against the real card boxes
      (an existing card may have been dragged or resized anywhere)
   2. Auto-layout arranges by relationship, not by insertion order: parents on
      the left, their children in the next column, unrelated tables last — so
      every arrow reads left → right
   3. the derived hub parks immediately left of the children it feeds
   4. dragging snaps to an 8px grid and can't leave the canvas
   ------------------------------------------------------------------------- */
const DM={GRID:8, GAP:24, COLGAP:58, PAD:20, W:186, HEAD:34, ROW:23, MAXH:220};
const snap=v=>Math.round(v/DM.GRID)*DM.GRID;
const nodeNames=()=>[...Object.keys(DATA?DATA.tables:{}), ...(hubName()?[hubName()]:[])];

/* measured box when the card is on screen, estimated (columns × row height)
   when it isn't — layout runs before the first render */
function cardBox(name){
  const c=$("#dm-canvas");
  const el=c&&c.querySelector(`.dm-card[data-node="${cssEsc(name)}"]`);
  if(el&&el.offsetWidth) return {w:el.offsetWidth, h:el.offsetHeight};
  const sz=MODEL.size[name]||{};
  const ncol = name===hubName() ? 1 : ((DATA&&DATA.tables[name]?DATA.tables[name].columns.length:3));
  return {w: sz.w||DM.W,
          h: DM.HEAD + Math.min(sz.h||DM.MAXH, Math.max(30, ncol*DM.ROW)) + 2};
}
function boxAt(name,p){ const b=cardBox(name); return {x:p.x, y:p.y, w:b.w, h:b.h}; }
function hits(a,b){ const g=12;
  return a.x < b.x+b.w+g && b.x < a.x+a.w+g && a.y < b.y+b.h+g && b.y < a.y+a.h+g; }
/* boxes of every currently-placed node, except `skip` */
function takenBoxes(skip){
  return nodeNames().filter(n=>n!==skip && MODEL.pos[n]).map(n=>boxAt(n,MODEL.pos[n]));
}
/* first slot, scanning left→right then top→bottom, that clears every taken box */
function freeSpot(name,taken){
  const b=cardBox(name);
  const W=Math.max(DM.PAD*2+b.w, ($("#dm-canvas")||{}).clientWidth||900);
  for(let row=0; row<400; row++){
    const y=DM.PAD+row*40;
    for(let x=DM.PAD; x+b.w<=W-DM.PAD || x===DM.PAD; x+=b.w+DM.GAP){
      const cand={x,y,w:b.w,h:b.h};
      if(!taken.some(t=>hits(cand,t))) return {x:snap(x), y:snap(y)};
    }
  }
  const bottom=taken.reduce((m,t)=>Math.max(m,t.y+t.h),0);   // canvas full — stack below
  return {x:DM.PAD, y:snap(bottom+DM.GAP)};
}

function layoutNodes(reset){
  if(!DATA) return;
  const names=nodeNames();
  for(const n of Object.keys(MODEL.pos)) if(!names.includes(n)) delete MODEL.pos[n];  // prune stale
  if(reset){ tidyLayout(names); return; }
  const taken=takenBoxes(null);
  for(const n of names){
    if(MODEL.pos[n]) continue;
    const p=freeSpot(n,taken);
    MODEL.pos[n]=p; taken.push(boxAt(n,p));
  }
}

/* relationship-aware arrangement: rank nodes by how deep they sit in the
   parent→child graph, lay each rank out as a column, order each column by its
   parents' rows (barycentre) so the connectors cross as little as possible */
function tidyLayout(names){
  const parents={}, kids={};
  const edge=(p,c)=>{ (kids[p]=kids[p]||[]).push(c); (parents[c]=parents[c]||[]).push(p); };
  MODEL.rels.forEach(r=>{ if(names.includes(r.parent_table_name)&&names.includes(r.child_table_name))
    edge(r.parent_table_name,r.child_table_name); });
  if(MODEL.hub.key) MODEL.hub.children.forEach(t=>{ if(names.includes(t)) edge(hubName(),t); });

  const depth={}; names.forEach(n=>depth[n]=0);
  for(let pass=0; pass<names.length; pass++){          // bounded: also breaks cycles
    let moved=false;
    for(const c in parents) for(const p of parents[c])
      if(depth[c] < depth[p]+1){ depth[c]=depth[p]+1; moved=true; }
    if(!moved) break;
  }
  const linked=n=>!!((kids[n]&&kids[n].length)||(parents[n]&&parents[n].length));
  const cols=[];
  names.filter(linked).forEach(n=>{ const d=depth[n]; (cols[d]=cols[d]||[]).push(n); });
  const packed=cols.filter(c=>c&&c.length);
  const solo=names.filter(n=>!linked(n));
  if(solo.length) packed.push(solo);                   // unrelated tables in a trailing column
  if(!packed.length) return;

  const bary=(n,prev)=>{ const ix=(parents[n]||[]).map(p=>prev.indexOf(p)).filter(i=>i>=0);
    return ix.length ? ix.reduce((a,b)=>a+b,0)/ix.length : Number.MAX_SAFE_INTEGER; };
  for(let i=1;i<packed.length;i++){ const prev=packed[i-1];
    packed[i]=packed[i].map((n,ix)=>({n,ix})).sort((a,b)=>(bary(a.n,prev)-bary(b.n,prev))||(a.ix-b.ix)).map(o=>o.n); }

  const colH=packed.map(col=>col.reduce((s,n)=>s+cardBox(n).h+DM.GAP,-DM.GAP));
  const tallest=Math.max(...colH);
  let x=DM.PAD;
  packed.forEach((col,ci)=>{
    const cw=Math.max(...col.map(n=>cardBox(n).w));
    let y=DM.PAD+(tallest-colH[ci])/2;                 // centre each column against the tallest
    for(const n of col){ const b=cardBox(n);
      MODEL.pos[n]={x:snap(x+(cw-b.w)/2), y:snap(y)}; y+=b.h+DM.GAP; }
    x+=cw+DM.COLGAP;
  });
}

/* the hub feeds its children — park it to their left, centred on them, and
   slide the rest of the canvas right if it would fall off the left edge */
function placeHub(){
  const h=hubName(); if(!h||!DATA) return;
  delete MODEL.pos[h];
  const b=cardBox(h);
  const kidBoxes=MODEL.hub.children.filter(t=>MODEL.pos[t]).map(t=>boxAt(t,MODEL.pos[t]));
  if(!kidBoxes.length){ MODEL.pos[h]=freeSpot(h,takenBoxes(h)); return; }
  const left=Math.min(...kidBoxes.map(k=>k.x));
  const cy=kidBoxes.reduce((s,k)=>s+k.y+k.h/2,0)/kidBoxes.length;
  const p={x:snap(left-b.w-DM.COLGAP), y:snap(Math.max(DM.PAD, cy-b.h/2))};
  if(p.x<DM.PAD){
    const dx=DM.PAD-p.x;
    for(const n of Object.keys(MODEL.pos)) MODEL.pos[n].x+=dx;
    p.x=DM.PAD;
  }
  MODEL.pos[h] = takenBoxes(h).some(t=>hits(boxAt(h,p),t)) ? freeSpot(h,takenBoxes(h)) : p;
}
function afterModelChange(){ renderModel(); syncEntity(); renderStructureMirror(); updateSummaries();
  const b=$("#dm-validate"); if(b) b.classList.remove("show"); }   // results are now stale

function linkedSet(){ const s=new Set();
  for(const r of MODEL.rels){ s.add(r.parent_table_name+"::"+r.parent_primary_key); s.add(r.child_table_name+"::"+r.child_foreign_key); }
  if(MODEL.hub.key){ s.add(hubName()+"::"+MODEL.hub.key); for(const t of MODEL.hub.children) s.add(t+"::"+MODEL.hub.key); }
  return s; }

function renderModel(){
  const canvas=$("#dm-canvas"); if(!canvas) return;
  $("#dm-empty").style.display = DATA?"none":"block";
  if(!DATA) return;
  layoutNodes(false);   // place any new node in free space *before* the cards are rebuilt
  canvas.querySelectorAll(".dm-card").forEach(c=>c.remove());
  const linked=linkedSet();
  const nodes=[...Object.keys(DATA.tables)];
  if(hubName()) nodes.push(hubName());
  for(const name of nodes){
    const hub = name===hubName();
    const cols = hub ? [MODEL.hub.key] : DATA.tables[name].columns.map(c=>c.name);
    const el=document.createElement("div");
    el.className="dm-card"+(hub?" hubnode":""); el.dataset.node=name;
    const pos=MODEL.pos[name]||(MODEL.pos[name]=freeSpot(name,takenBoxes(name)));
    el.style.left=pos.x+"px"; el.style.top=pos.y+"px";
    const sz=MODEL.size[name]; if(sz){ el.style.width=sz.w+"px"; }
    const colsStyle = sz ? ` style="max-height:${Math.max(40,sz.h)}px"` : "";
    el.innerHTML=`<div class="dm-head" data-drag="${esc(name)}">${esc(name)}
        <span class="tag">${hub?"derived hub":(DATA.tables[name].rows.toLocaleString()+" rows")}</span></div>
      <div class="dm-cols"${colsStyle}>${cols.map(c=>{
        const iskey = hub || linked.has(name+"::"+c) || (MODEL.hub.key&&c===MODEL.hub.key);
        return `<div class="dm-col ${iskey?"iskey linked":""}"><span>${esc(c)}</span>
          <span class="dm-port" data-port="${esc(name)}::${esc(c)}" data-t="${esc(name)}" data-col="${esc(c)}"></span></div>`;
      }).join("")}</div><div class="dm-resize" data-resize="${esc(name)}"></div>`;
    canvas.appendChild(el);
  }
  drawLinks();
}
/* ---- link geometry: attach connectors to the CARD EDGE at the column's row,
   clamped into the card body (so a scrolled-out key column still connects) ---- */
function cssEsc(s){ return (window.CSS&&CSS.escape)?CSS.escape(s):s.replace(/"/g,'\\"'); }
function canvasXY(){ const canvas=$("#dm-canvas"); const cr=canvas.getBoundingClientRect();
  return {cr, sx:canvas.scrollLeft, sy:canvas.scrollTop}; }
function nodeRect(node){ const canvas=$("#dm-canvas");
  const el=canvas.querySelector(`.dm-card[data-node="${cssEsc(node)}"]`); if(!el) return null;
  const {cr,sx,sy}=canvasXY(); const r=el.getBoundingClientRect();
  const L=r.left-cr.left+sx, T=r.top-cr.top+sy, R=r.right-cr.left+sx, B=r.bottom-cr.top+sy;
  return {L,T,R,B, cx:(L+R)/2, cy:(T+B)/2}; }
function colY(node,col,nr){ const canvas=$("#dm-canvas");
  const p=canvas.querySelector(`[data-port="${cssEsc(node+"::"+col)}"]`);
  if(!p) return nr.cy;
  const {cr,sy}=canvasXY(); const r=p.getBoundingClientRect();
  const y=r.top-cr.top+sy+r.height/2;
  return Math.max(nr.T+30, Math.min(nr.B-10, y));   // clamp inside the card body
}
function endpoint(node,col,peerCx){ const nr=nodeRect(node); if(!nr) return null;
  const side = peerCx>=nr.cx ? "right" : "left";
  return {x: side==="right"?nr.R:nr.L, y: colY(node,col,nr), side}; }
function bezierD(a,b){ const K=Math.max(38, Math.abs(b.x-a.x)/2);
  const ax=a.side==="right"?a.x+K:a.x-K, bx=b.side==="right"?b.x+K:b.x-K;
  return `M${a.x} ${a.y} C${ax} ${a.y}, ${bx} ${b.y}, ${b.x} ${b.y}`; }
function relCard(r){ return r.card || (isUnique(r.child_table_name,r.child_foreign_key)?"1-1":"1-*"); }

function drawLinks(){
  const canvas=$("#dm-canvas"), svg=$("#dm-svg"); if(!canvas||!svg) return;
  svg.setAttribute("width", Math.max(canvas.clientWidth, canvas.scrollWidth));
  svg.setAttribute("height", Math.max(canvas.clientHeight, canvas.scrollHeight));
  const seg=(pNode,pCol,cNode,cCol,cls,pMark,cMark,idx)=>{
    const pr=nodeRect(pNode), cr2=nodeRect(cNode); if(!pr||!cr2) return "";
    const a=endpoint(pNode,pCol,cr2.cx), b=endpoint(cNode,cCol,pr.cx); if(!a||!b) return "";
    const plug=cls==="hub"?"plug hub":"plug";
    const mark=(pt,txt)=>{ const dx=pt.side==="right"?13:-13;
      return `<text class="card-mark" x="${pt.x+dx}" y="${pt.y-6}" text-anchor="middle">${txt}</text>`; };
    return `<path class="${cls}${selRels.has(idx)?" sel":""}" ${idx!=null?`data-rel="${idx}"`:""} d="${bezierD(a,b)}"></path>`
      + `<circle class="${plug}" cx="${a.x}" cy="${a.y}" r="3.2"></circle>`
      + `<circle class="${plug}" cx="${b.x}" cy="${b.y}" r="3.2"></circle>`
      + mark(a,pMark) + mark(b,cMark);
  };
  let out="";
  MODEL.rels.forEach((r,i)=>{ const cm=relCard(r)==="1-1"?"1":"∗";
    out+=seg(r.parent_table_name,r.parent_primary_key,r.child_table_name,r.child_foreign_key,"","1",cm,i); });
  if(MODEL.hub.key){ for(const t of MODEL.hub.children)
    out+=seg(hubName(),MODEL.hub.key,t,MODEL.hub.key,"hub","1","∗",null); }
  svg.innerHTML=out;
  svg.querySelectorAll('path[data-rel]').forEach(p=>p.addEventListener("click",e=>{
    const i=+p.dataset.rel;
    if(e.shiftKey||e.ctrlKey||e.metaKey){            // add / remove without losing the rest
      selRels.has(i)?selRels.delete(i):selRels.add(i);
      selRels.size?openRelSelection():closeRelEditor();
    } else openRelEditor(i);
  }));
}
function refreshModel(){ layoutNodes(false); positionCards(); drawLinks(); }
function positionCards(){ $("#dm-canvas").querySelectorAll(".dm-card").forEach(el=>{
  const p=MODEL.pos[el.dataset.node]; if(p){ el.style.left=p.x+"px"; el.style.top=p.y+"px"; } }); }

/* ---- pointer interactions: drag cards, resize cards, drag ports to link ---- */
let dragCard=null, dragStart=null, linkFrom=null, resizeCard=null, resizeStart=null;
let selRels=new Set();          // indices into MODEL.rels; drives .sel and the sidebar
let marquee=null;               // {x0,y0,el} while rubber-band selecting
$("#dm-canvas").addEventListener("pointerdown",e=>{
  const handle=e.target.closest(".dm-resize");
  if(handle){ e.preventDefault(); const node=handle.dataset.resize;
    const el=$("#dm-canvas").querySelector(`.dm-card[data-node="${cssEsc(node)}"]`);
    const cols=el.querySelector(".dm-cols");
    resizeCard=node; resizeStart={px:e.clientX, py:e.clientY, w:el.offsetWidth, h:cols.clientHeight};
    handle.setPointerCapture&&handle.setPointerCapture(e.pointerId); return; }
  const port=e.target.closest(".dm-port");
  if(port){ e.preventDefault(); linkFrom={t:port.dataset.t, col:port.dataset.col, el:port}; return; }
  const head=e.target.closest(".dm-head");
  if(head){ e.preventDefault(); const node=head.dataset.drag;
    dragCard=node; const p=MODEL.pos[node]||{x:20,y:20}; MODEL.pos[node]=p;
    dragStart={px:e.clientX, py:e.clientY, x:p.x, y:p.y}; head.setPointerCapture&&head.setPointerCapture(e.pointerId); return; }
  // empty canvas: start a rubber-band selection (and drop the old one unless adding)
  if(!e.target.closest(".dm-card") && !e.target.closest("#dm-svg")){
    e.preventDefault();
    if(!(e.shiftKey||e.ctrlKey||e.metaKey)){ selRels.clear(); closeRelEditor(); }
    const {cr,sx,sy}=canvasXY();
    const el=document.createElement("div"); el.className="dm-marquee";
    marquee={x0:e.clientX-cr.left+sx, y0:e.clientY-cr.top+sy, el, moved:false};
    $("#dm-canvas").appendChild(el);
    drawLinks();
  }
});
/* ---- rubber-band selection: which links does the dragged box touch? ----
   Bezier bounding boxes are far too generous on a curve, so each path is
   sampled along its length and hit-tested point by point. */
function marqueeRect(e,m){
  const {cr,sx,sy}=canvasXY();
  const x=e.clientX-cr.left+sx, y=e.clientY-cr.top+sy;
  return {x:Math.min(m.x0,x), y:Math.min(m.y0,y),
    w:Math.abs(x-m.x0), h:Math.abs(y-m.y0)};
}
function pathHitsRect(p,r){
  const len=p.getTotalLength(); if(!len) return false;
  const step=Math.max(4, len/40);
  for(let d=0; d<=len; d+=step){ const pt=p.getPointAtLength(d);
    if(pt.x>=r.x && pt.x<=r.x+r.w && pt.y>=r.y && pt.y<=r.y+r.h) return true; }
  return false;
}
function applyMarquee(r){
  // hub links have no data-rel: they are derived from the hub, not individually editable
  $("#dm-svg").querySelectorAll("path[data-rel]").forEach(p=>{
    if(pathHitsRect(p,r)) selRels.add(+p.dataset.rel); });
}
document.addEventListener("pointermove",e=>{
  if(resizeCard){ const el=$("#dm-canvas").querySelector(`.dm-card[data-node="${cssEsc(resizeCard)}"]`);
    const w=Math.max(150, resizeStart.w+(e.clientX-resizeStart.px));
    const h=Math.max(40, resizeStart.h+(e.clientY-resizeStart.py));
    MODEL.size[resizeCard]={w,h};
    el.style.width=w+"px"; el.querySelector(".dm-cols").style.maxHeight=h+"px"; drawLinks(); return; }
  if(dragCard){ const p=MODEL.pos[dragCard];
    p.x=Math.max(0, snap(dragStart.x+(e.clientX-dragStart.px)));
    p.y=Math.max(0, snap(dragStart.y+(e.clientY-dragStart.py)));
    const el=$("#dm-canvas").querySelector(`.dm-card[data-node="${cssEsc(dragCard)}"]`);
    if(el){ el.style.left=p.x+"px"; el.style.top=p.y+"px"; } drawLinks(); return; }
  if(linkFrom){ drawTempLink(e); }
  if(marquee){ const r=marqueeRect(e,marquee);
    // ignore the few px of jitter in a plain click, so a click still just deselects
    if(r.w>3||r.h>3) marquee.moved=true;
    Object.assign(marquee.el.style,
      {left:r.x+"px", top:r.y+"px", width:r.w+"px", height:r.h+"px"}); }
});
document.addEventListener("pointerup",e=>{
  if(resizeCard){ resizeCard=null; drawLinks(); }
  if(dragCard){ dragCard=null; updateSummaries(); }
  if(marquee){ const m=marquee; marquee=null; m.el.remove();
    if(m.moved){ applyMarquee(marqueeRect(e,m));
      if(selRels.size===1) openRelEditor([...selRels][0]);
      else if(selRels.size>1) openRelSelection();
      else closeRelEditor(); }
    drawLinks(); }
  if(linkFrom){ const tgt=document.elementFromPoint(e.clientX,e.clientY);
    const port=tgt&&tgt.closest&&tgt.closest(".dm-port");
    const from=linkFrom; linkFrom=null; const t=$("#dm-drag"); if(t) t.remove();
    if(port && port.dataset.t!==from.t) beginNewRelationship(from,{t:port.dataset.t,col:port.dataset.col});
    drawLinks(); }
});
function drawTempLink(e){
  const canvas=$("#dm-canvas"), svg=$("#dm-svg"); const {cr,sx,sy}=canvasXY();
  const r=linkFrom.el.getBoundingClientRect();
  const a={x:r.left-cr.left+sx+r.width/2, y:r.top-cr.top+sy+r.height/2};
  const b={x:e.clientX-cr.left+sx, y:e.clientY-cr.top+sy};
  let t=$("#dm-drag"); if(!t){ t=document.createElementNS("http://www.w3.org/2000/svg","path"); t.id="dm-drag"; svg.appendChild(t); }
  const K=Math.max(30,Math.abs(b.x-a.x)/2);
  t.setAttribute("d",`M${a.x} ${a.y} C${a.x+K} ${a.y}, ${b.x-K} ${b.y}, ${b.x} ${b.y}`);
  t.setAttribute("stroke","var(--blue)"); t.setAttribute("stroke-dasharray","4 4"); t.setAttribute("fill","none");
}
$("#dm-autolayout").addEventListener("click",()=>{ layoutNodes(true); positionCards(); drawLinks(); });

/* ---- relationship editor sidebar ---- */
let editIdx=-1;   // -1 = editing a brand-new (not-yet-saved) relationship
function tableCols(t){ return (DATA.tables[t]?DATA.tables[t].columns.map(c=>c.name):[]); }
function fillColSelect(sel,table,val){ sel.innerHTML=tableCols(table).map(c=>`<option ${c===val?"selected":""}>${esc(c)}</option>`).join(""); }
function relToForm(from,to,card){
  const tabs=Object.keys(DATA.tables);
  const tOpts=t2=>tabs.map(t=>`<option ${t===t2?"selected":""}>${esc(t)}</option>`).join("");
  $("#dm-e-ftable").innerHTML=tOpts(from.t); $("#dm-e-ttable").innerHTML=tOpts(to.t);
  fillColSelect($("#dm-e-fcol"),from.t,from.col); fillColSelect($("#dm-e-tcol"),to.t,to.col);
  $("#dm-e-card").value=card; updateCardViz();
}
function updateCardViz(){ const v=$("#dm-e-card").value;
  $("#dm-e-viz").textContent = v==="one-to-many"?"1  —  ∗" : v==="many-to-one"?"∗  —  1" : "1  —  1"; }
/* the form only makes sense for exactly one relationship; multi-select grays it
   out and leaves Delete as the one live action */
function setFormEnabled(on){
  $("#dm-e-form").classList.toggle("disabled",!on);
  $("#dm-e-form").querySelectorAll("select").forEach(s=>s.disabled=!on);
  $("#dm-e-save").style.display=on?"":"none";
}
function openRelEditor(idx){
  editIdx=idx; selRels=new Set([idx]); const r=MODEL.rels[idx];
  const card = relCard(r)==="1-1" ? "one-to-one" : "one-to-many";
  relToForm({t:r.parent_table_name,col:r.parent_primary_key},{t:r.child_table_name,col:r.child_foreign_key},card);
  setFormEnabled(true);
  $("#dm-side-title").textContent="Edit relationship";
  $("#dm-e-sub").textContent="Parent (the “one”/key side) is generated from the cardinality you pick.";
  $("#dm-e-del").textContent="Delete"; $("#dm-e-del").style.display="inline-block";
  $("#dm-e-err").textContent=""; $("#dm-side").classList.add("open");
  drawLinks();   // highlight the selected link
}
/* multi-select: no editable form, just a count and a delete-all */
function openRelSelection(){
  editIdx=-1; setFormEnabled(false);
  $("#dm-side-title").textContent=`${selRels.size} relationships selected`;
  $("#dm-e-sub").textContent="Editing needs a single relationship. Delete removes all of them.";
  $("#dm-e-del").textContent=`Delete ${selRels.size}`; $("#dm-e-del").style.display="inline-block";
  $("#dm-e-err").textContent=""; $("#dm-side").classList.add("open");
  drawLinks();
}
/* delete a relationship (from the sidebar button or the keyboard Delete key) */
function deleteRel(idx){
  if(idx==null || idx<0 || idx>=MODEL.rels.length) return;
  MODEL.rels.splice(idx,1); selRels.clear(); closeRelEditor(); afterModelChange();
}
/* delete every selected relationship — descending so the splices don't shift
   the indices still to be removed */
function deleteSelectedRels(){
  const idxs=[...selRels].sort((a,b)=>b-a); if(!idxs.length) return;
  for(const i of idxs) if(i>=0 && i<MODEL.rels.length) MODEL.rels.splice(i,1);
  selRels.clear(); closeRelEditor(); afterModelChange();
}
function beginNewRelationship(from,to){
  editIdx=-1;
  // sensible default cardinality from column uniqueness
  let card="one-to-many";
  if(isUnique(to.t,to.col)&&!isUnique(from.t,from.col)) card="many-to-one";
  else if(isUnique(from.t,from.col)&&isUnique(to.t,to.col)) card="one-to-one";
  relToForm(from,to,card);
  setFormEnabled(true);
  $("#dm-side-title").textContent="New relationship"; $("#dm-e-del").style.display="none";
  $("#dm-e-sub").textContent="Parent (the “one”/key side) is generated from the cardinality you pick.";
  $("#dm-e-err").textContent=""; $("#dm-side").classList.add("open");
}
function closeRelEditor(){ $("#dm-side").classList.remove("open"); editIdx=-1; selRels.clear();
  setFormEnabled(true); drawLinks(); }
function formToRel(){
  const ft=$("#dm-e-ftable").value, fc=$("#dm-e-fcol").value;
  const tt=$("#dm-e-ttable").value, tc=$("#dm-e-tcol").value, card=$("#dm-e-card").value;
  // parent = the "one" side; many-to-one flips first/second
  let parent={t:ft,col:fc}, child={t:tt,col:tc}, ctype="1-*";
  if(card==="many-to-one"){ parent={t:tt,col:tc}; child={t:ft,col:fc}; }
  if(card==="one-to-one") ctype="1-1";
  return {parent_table_name:parent.t, parent_primary_key:parent.col,
    child_table_name:child.t, child_foreign_key:child.col, card:ctype};
}
$("#dm-e-ftable").addEventListener("change",()=>fillColSelect($("#dm-e-fcol"),$("#dm-e-ftable").value));
$("#dm-e-ttable").addEventListener("change",()=>fillColSelect($("#dm-e-tcol"),$("#dm-e-ttable").value));
$("#dm-e-card").addEventListener("change",updateCardViz);
$("#dm-e-cancel").addEventListener("click",closeRelEditor);
$("#dm-e-del").addEventListener("click",deleteSelectedRels);
/* keyboard: Delete / Backspace removes every selected link (when not typing in a field) */
document.addEventListener("keydown",e=>{
  if(e.key!=="Delete" && e.key!=="Backspace") return;
  if(!selRels.size || !$("#pane-model").classList.contains("active")) return;
  const el=document.activeElement, tag=el&&el.tagName;
  if(tag==="INPUT"||tag==="SELECT"||tag==="TEXTAREA"||(el&&el.isContentEditable)) return;
  e.preventDefault(); deleteSelectedRels();
});
$("#dm-e-save").addEventListener("click",()=>{
  const rel=formToRel(), err=$("#dm-e-err");
  if(rel.parent_table_name===rel.child_table_name){ err.textContent="Pick two different tables."; return; }
  if(!rel.parent_primary_key||!rel.child_foreign_key){ err.textContent="Pick a column on each side."; return; }
  const dup=MODEL.rels.findIndex((r,i)=>i!==editIdx && r.parent_table_name===rel.parent_table_name
    && r.parent_primary_key===rel.parent_primary_key && r.child_table_name===rel.child_table_name
    && r.child_foreign_key===rel.child_foreign_key);
  if(dup>=0){ err.textContent="That relationship already exists."; return; }
  if(editIdx>=0) MODEL.rels[editIdx]=rel; else MODEL.rels.push(rel);
  closeRelEditor(); afterModelChange();
});

/* ---- validate the model against the real data (server-side) ---- */
async function validateModel(){
  const box=$("#dm-validate"); box.classList.add("show");
  box.innerHTML=`<div class="dm-vrow">validating…</div>`;
  try{
    const r=await apiFetch("/api/validate_model",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({relationships:MODEL.rels, entity_key:MODEL.hub.key||"", entity_children:MODEL.hub.children})});
    const j=await r.json();
    if(j.error){ box.innerHTML=`<div class="dm-vrow err">${esc(j.error)}</div>`; return; }
    if(!j.results.length){ box.innerHTML=`<div class="dm-vrow vdetail">Nothing to validate — no links or hub set.</div>`; return; }
    box.innerHTML=j.results.map(x=>`<div class="dm-vrow">${pill(x.status)}
      <span>${esc(x.label)}</span><span class="vdetail">${esc(x.detail)}</span></div>`).join("");
  }catch(e){ box.innerHTML=`<div class="dm-vrow err">${esc(BACKEND_HELP.split("\n")[0])}</div>`; }
}
$("#dm-validate-btn").addEventListener("click",validateModel);

/* ---- entity-key hub generator ---- */
function buildHubBar(){
  const keys=sharedKeys();
  $("#dm-hub-key").innerHTML=`<option value="">(choose a shared key…)</option>`+
    keys.map(k=>`<option value="${esc(k)}" ${k===MODEL.hub.key?"selected":""}>${esc(k)} (in ${keyTables(k).length} tables)</option>`).join("");
  renderHubChilds();
  updateHubButtons();
}
function renderHubChilds(){
  const key=$("#dm-hub-key").value; const host=$("#dm-hub-childs");
  if(!key){ host.innerHTML=`<span class="dm-childchip off">choose a key first</span>`; return; }
  const tabs=keyTables(key);
  const chosen = MODEL.hub.key===key ? new Set(MODEL.hub.children) : new Set(tabs);
  host.innerHTML=tabs.map(t=>`<span class="dm-childchip ${chosen.has(t)?"on":""}" data-t="${esc(t)}">${esc(t)}</span>`).join("");
  host.querySelectorAll(".dm-childchip").forEach(ch=>ch.addEventListener("click",()=>{ ch.classList.toggle("on"); updateHubButtons(); }));
}
function chosenChilds(){ return [...$("#dm-hub-childs").querySelectorAll(".dm-childchip.on")].map(c=>c.dataset.t); }
function updateHubButtons(){
  const key=$("#dm-hub-key").value; const n=chosenChilds().length;
  $("#dm-hub-gen").disabled = !(key && n>=1);
  $("#dm-hub-gen").textContent = MODEL.hub.key===key ? "Update hub" : "Generate hub";
  $("#dm-hub-clear").style.display = MODEL.hub.key ? "inline-block" : "none";
}
$("#dm-hub-key").addEventListener("change",()=>{ renderHubChilds(); updateHubButtons(); });
$("#dm-hub-gen").addEventListener("click",()=>{
  const key=$("#dm-hub-key").value, children=chosenChilds(); if(!key||!children.length) return;
  MODEL.hub={key, children}; placeHub();
  afterModelChange(); updateHubButtons();
});
$("#dm-hub-clear").addEventListener("click",()=>{
  delete MODEL.pos[hubName()]; MODEL.hub={key:"", children:[]};
  buildHubBar(); afterModelChange();
});

/* ---- left-panel read-only mirror + entity-key sync ---- */
function syncEntity(){ const sel=$("#in-entity");
  if(MODEL.hub.key && ![...sel.options].some(o=>o.value===MODEL.hub.key))
    sel.insertAdjacentHTML("beforeend",`<option value="${esc(MODEL.hub.key)}">${esc(MODEL.hub.key)}</option>`);
  sel.value=MODEL.hub.key||""; updateScdVisibility();
}
function renderStructureMirror(){
  const host=$("#structure-mirror"); if(!host) return;
  let h="";
  if(MODEL.hub.key){
    h+=`<div class="sm-row"><span class="sm-lbl">entity key</span><span class="sm-chip key">${esc(MODEL.hub.key)}</span>
      <span class="sm-lbl">→ ${MODEL.hub.children.length} table${MODEL.hub.children.length!==1?"s":""}</span></div>`;
  }
  if(MODEL.rels.length){
    h+=`<div class="sm-row"><span class="sm-lbl">links</span></div>`;
    h+=MODEL.rels.map(r=>`<div class="sm-row"><span class="sm-chip">${esc(r.child_table_name)}.${esc(r.child_foreign_key)} → ${esc(r.parent_table_name)}.${esc(r.parent_primary_key)}</span></div>`).join("");
  }
  if(!h) h=`<div class="sm-none">Independent tables — no keys or links set.</div>`;
  host.innerHTML=h;
}
function openModel(focusHub){ activateTab("pane-model"); requestAnimationFrame(refreshModel);
  if(focusHub) setTimeout(()=>$("#dm-hub-key").focus(),120); }
$("#btn-open-model").addEventListener("click",()=>openModel(false));

/* ---------------- constraints ---------------- */
const CON_TYPES=[["inequality","low ≤ high"],["range","low ≤ mid ≤ high"],
  ["fixed_combinations","columns co-vary"],["fixed_increments","multiple of N"]];
function colsFor(t){ return (DATA.tables[t]?DATA.tables[t].columns:[]).map(c=>c.name); }
function colSel(cls,t,sel){
  return `<select class="${cls}">`+["<option value=''></option>",
    ...colsFor(t).map(c=>`<option ${c===sel?"selected":""}>${esc(c)}</option>`)].join("")+"</select>";
}
function conBody(row){
  const t=row.querySelector(".con-table").value, type=row.querySelector(".con-type").value;
  const body=row.querySelector(".con-body"); body.className="con-body";
  if(type==="inequality"){ body.classList.add("two");
    body.innerHTML=`<div><label>low</label>${colSel("c-low",t)}</div><div><label>high</label>${colSel("c-high",t)}</div>`
      +`<div class="con-strict" style="grid-column:1/-1"><input type="checkbox" class="c-strict">strict (&lt; not ≤)</div>`;
  } else if(type==="range"){ body.classList.add("three");
    body.innerHTML=`<div><label>low</label>${colSel("c-low",t)}</div><div><label>middle</label>${colSel("c-mid",t)}</div><div><label>high</label>${colSel("c-high",t)}</div>`
      +`<div class="con-strict" style="grid-column:1/-1"><input type="checkbox" class="c-strict" checked>strict</div>`;
  } else if(type==="fixed_combinations"){ body.classList.add("two");
    body.innerHTML=`<div><label>column A</label>${colSel("c-a",t)}</div><div><label>column B</label>${colSel("c-b",t)}</div>`
      +`<div style="grid-column:1/-1"><label>column C (optional)</label>${colSel("c-c",t)}</div>`;
  } else { body.classList.add("two");
    body.innerHTML=`<div><label>column</label>${colSel("c-col",t)}</div><div><label>increment (int)</label><input type="number" class="c-inc" min="1" value="1" style="width:100%"></div>`;
  }
}
function conRow(){
  const div=document.createElement("div"); div.className="con-row";
  const tOpts=Object.keys(DATA.tables).map(t=>`<option>${esc(t)}</option>`).join("");
  const tyOpts=CON_TYPES.map(([v,l])=>`<option value="${v}">${l}</option>`).join("");
  div.innerHTML=`<div class="con-top"><select class="con-table">${tOpts}</select>
    <select class="con-type">${tyOpts}</select><button class="con-del" title="remove">✕</button></div>
    <div class="con-body"></div>`;
  div.querySelector(".con-del").addEventListener("click",()=>{div.remove();updateSummaries();});
  div.querySelector(".con-table").addEventListener("change",()=>conBody(div));
  div.querySelector(".con-type").addEventListener("change",()=>conBody(div));
  conBody(div);
  return div;
}
function renderConstraints(){ $("#constraint-rows").innerHTML=""; }

/* ---------------- PII handling panel ----------------
   detected server-side ({table:{col:kind}}); default policy = fake.  The select
   turns amber on "shuffle" because that choice re-deals REAL values. */
function renderPii(){
  const host=$("#pii-rows"); if(!host) return;
  let h="", n=0;
  for(const [t,info] of Object.entries(DATA.tables)){
    const pii=info.pii||{}; const cols=Object.keys(pii); if(!cols.length) continue;
    h+=`<div class="sub-label">${esc(t)}</div>`;
    for(const c of cols){ n++;
      h+=`<div class="pii-row" data-t="${esc(t)}" data-c="${esc(c)}">
        <span class="pii-col" title="${esc(c)}">${esc(c)}</span>
        <span class="pii-kind">${esc(pii[c])}</span>
        <select class="pii-pol">
          <option value="fake" selected>Fake</option>
          <option value="shuffle">Shuffle real</option>
          <option value="drop">Drop</option>
        </select></div>`;
    }
  }
  host.innerHTML = n? h : `<p class="hint">No PII-like columns detected.</p>`;
  host.querySelectorAll(".pii-pol").forEach(sel=>sel.addEventListener("change",()=>{
    sel.classList.toggle("leaky", sel.value==="shuffle"); updateSummaries();
  }));
}
function collectPii(){
  const out={};
  $$("#pii-rows .pii-row").forEach(r=>{
    (out[r.dataset.t]=out[r.dataset.t]||{})[r.dataset.c]=r.querySelector(".pii-pol").value;
  });
  return out;
}
function piiSummary(){
  const rows=$$("#pii-rows .pii-row");
  if(!rows.length) return "none detected";
  const cnt={fake:0, shuffle:0, drop:0};
  rows.forEach(r=>cnt[r.querySelector(".pii-pol").value]++);
  const bits=[];
  if(cnt.fake) bits.push(cnt.fake+" faked");
  if(cnt.drop) bits.push(cnt.drop+" dropped");
  if(cnt.shuffle) bits.push("\u26a0 "+cnt.shuffle+" real");
  return bits.join(" \u00b7 ");
}
$("#btn-add-constraint").addEventListener("click",()=>{$("#constraint-rows").appendChild(conRow());updateSummaries();});
function collectConstraints(){
  return [...$$(".con-row")].map(r=>{
    const table=r.querySelector(".con-table").value, type=r.querySelector(".con-type").value;
    const v=cls=>{const e=r.querySelector(cls);return e?e.value:"";};
    const ck=cls=>{const e=r.querySelector(cls);return e?e.checked:false;};
    if(type==="inequality"){ const low=v(".c-low"),high=v(".c-high"); if(!low||!high)return null;
      return {table,type,low,high,strict:ck(".c-strict")}; }
    if(type==="range"){ const low=v(".c-low"),middle=v(".c-mid"),high=v(".c-high");
      if(!low||!middle||!high)return null; return {table,type,low,middle,high,strict:ck(".c-strict")}; }
    if(type==="fixed_combinations"){ const cols=[v(".c-a"),v(".c-b"),v(".c-c")].filter(Boolean);
      if(cols.length<2)return null; return {table,type,columns:cols}; }
    const column=v(".c-col"),increment=parseInt(v(".c-inc")||"1",10);
    if(!column)return null; return {table,type,column,increment};
  }).filter(Boolean);
}

/* ---------------- strategy recommendation ---------------- */
const TIER_LABEL={0:"TIER 0 · INDEPENDENT TABLES",1:"TIER 1 · LINKED TABLES",
  2:"TIER 2 · VERSION HISTORY (SCD)",3:"TIER 3 · VERSION PATTERNS (SCD, no key)"};
const STRATEGY_LABEL={independent:"each table on its own",
  relational:"keep the table links",
  sequential:"model each entity's history over time",
  temporal_statistical:"copy the history patterns, not identities"};
let RECOMMENDED_RELS=[];
function renderAdvisor(profile){
  const banner=$("#advisor"), body=$("#advisor-body");
  if(!profile||profile.error||!profile.recommendation){ banner.style.display="none"; return; }
  const r=profile.recommendation; RECOMMENDED_RELS=r.relationships||[];
  banner.style.display="block";
  // The reasoning is long and only needed once — park it behind a hover marker and
  // keep the panel to the headline plus anything actionable (warnings).
  const why=(r.reasons||[]).join(" ");
  let h=`<div class="strat-head"><span class="tier-badge tier-${r.tier}">${TIER_LABEL[r.tier]||("TIER "+r.tier)}</span>
    <span class="strat-name">${STRATEGY_LABEL[r.strategy]||r.strategy}</span>
    ${why?ihelp(why+" — a suggestion from the profiler; you can override it below, and you can always set an entity key yourself if you know a business key that ties an entity's rows together."):""}</div>`;
  const warns=r.warnings||[];
  if(warns.length){
    h+=`<ul class="strat-list">`;
    // a warning is actionable, so it stays on screen — but only the first line of it
    warns.forEach(x=>{ const s=String(x), one=s.split(". ")[0];
      h+=`<li class="warn">${esc(one)}${one.length<s.length?ihelp(s):""}</li>`; });
    h+=`</ul>`;
  }
  h+=`<div class="advisor-actions">`;
  if(RECOMMENDED_RELS.length)
    h+=`<button class="mini-btn primary" id="btn-apply-rels">Apply ${RECOMMENDED_RELS.length} recommended link${RECOMMENDED_RELS.length>1?"s":""}</button>`;
  h+=`<button class="mini-btn" id="btn-goto-key">${RECOMMENDED_RELS.length?"Or choose an entity key":"Choose an entity key"} →</button></div>`;
  body.innerHTML=h;
  const apply=$("#btn-apply-rels");
  if(apply) apply.addEventListener("click",()=>{ MODEL.rels=RECOMMENDED_RELS.slice();
    afterModelChange(); openModel(false);
    apply.className="mini-btn done"; apply.textContent="✓ applied — see Data Model"; apply.disabled=true; });
  $("#btn-goto-key").addEventListener("click",()=>openModel(true));
  updateSummaries();
}

/* ---------------- recipe ---------------- */
function renderRecipe(){
  $("#synth-chips").innerHTML=SYNTHS.map(s=>`<span class="chip ${selectedSynths.has(s)?"on":""}" data-s="${s}">
    <span class="dot" style="background:${PALETTE[s]}"></span>${s}</span>`).join("");
  $$("#synth-chips .chip").forEach(ch=>ch.addEventListener("click",()=>{
    const s=ch.dataset.s;
    selectedSynths.has(s)?selectedSynths.delete(s):selectedSynths.add(s);
    ch.classList.toggle("on"); $("#btn-run").disabled=!selectedSynths.size; updateEpochsVisibility(); updateSummaries();
  }));
  updateEpochsVisibility();
  $("#target-fields").innerHTML=Object.entries(DATA.tables).map(([t,v])=>`
    <div class="tgt-row"><span title="${t}">${t}</span>
      <select id="target-${t}"><option value="auto">(auto)</option>
        ${v.targets.map(c=>`<option>${c}</option>`).join("")}</select></div>`).join("");
  // CategoricalCAP sensitive column: what the attribute-inference attack tries
  // to guess. (auto) = the most balanced categorical, picked server-side.
  $("#cap-fields").innerHTML=Object.entries(DATA.tables).map(([t,v])=>`
    <div class="tgt-row"><span title="${t}">${t}</span>
      <select id="cap-${t}" title="column the attribute-inference attack tries to guess — (auto) picks the most balanced categorical">
        <option value="auto">(auto)</option>
        ${(v.categoricals||[]).map(c=>`<option>${c}</option>`).join("")}</select></div>`).join("");
  // hidden entity-key store — the Data Model tab (hub generator) sets its value;
  // options list every shared candidate key so syncEntity() can select any of them
  const shared={};
  for(const v of Object.values(DATA.tables)) for(const c of v.columns) shared[c.name]=(shared[c.name]||0)+1;
  const cands=Object.keys(shared).filter(c=>shared[c]>=2).sort();
  $("#in-entity").innerHTML=`<option value=""></option>`+
    cands.map(c=>`<option value="${esc(c)}">${esc(c)}</option>`).join("");
  // SCD timeline columns: all column names (date-ish first), effective/end pre-selected
  const allCols=[...new Set(Object.values(DATA.tables).flatMap(v=>v.columns.map(c=>c.name)))];
  const dateish=c=>/(_dt|_date|eff|end|start|expiry|since|left)/i.test(c);
  const ordered=allCols.slice().sort((a,b)=>(dateish(b)-dateish(a))||a.localeCompare(b));
  const effSug=ordered.find(c=>/eff|effective|start|since|from/i.test(c))||"";
  const endSug=ordered.find(c=>/(^|_)end|expire|expiry|left|thru/i.test(c))||"";
  const opt=(ph,sel)=>`<option value="">${ph}</option>`+
    ordered.map(c=>`<option value="${esc(c)}" ${c===sel?"selected":""}>${esc(c)}</option>`).join("");
  $("#in-scd-eff").innerHTML=opt("effective-date column…",effSug);
  $("#in-scd-end").innerHTML=opt("end-date column…",endSug);
  $("#in-scd-cur").innerHTML=opt("current-flag column (optional)…","");
  updateScdVisibility();
  $("#btn-run").disabled=!selectedSynths.size;
}
function updateScdVisibility(){ $("#field-scd").style.display=$("#in-entity").value?"block":"none"; }
function updateEpochsVisibility(){
  $("#field-epochs").style.display=[...selectedSynths].some(s=>GAN_SYNTHS.has(s))?"block":"none";
}
[["epochs",v=>v],["scale",v=>(+v).toFixed(2).replace(/0$/,"")],["holdout",v=>v]].forEach(([k,f])=>{
  $(`#in-${k}`).addEventListener("input",e=>{$(`#out-${k}`).textContent=f(e.target.value); updateSummaries();});
});

/* ---------------- tabs ---------------- */
function lockReportTabs(lock){
  const keep=["pane-schema","pane-model"];   // schema + data model stay usable pre-run
  $$('.tab[data-pane]').forEach(t=>{ if(!keep.includes(t.dataset.pane)){
    t.classList.toggle("locked",lock);
    if(lock) t.title="Run a synthesis to unlock the report"; else t.removeAttribute("title");
  }});
}
function activateTab(pane){
  $$(".tab").forEach(t=>t.classList.toggle("active",t.dataset.pane===pane));
  $$(".pane").forEach(p=>p.classList.toggle("active",p.id===pane));
  if(pane==="pane-model") requestAnimationFrame(refreshModel);
  if(pane==="pane-report") requestAnimationFrame(flushVisiblePlots);
}
/* report side-nav: switch which result section is visible */
function showSection(id){
  $$(".rep-sec").forEach(s=>s.classList.toggle("active",s.id===id));
  $$(".rep-navbtn").forEach(b=>b.classList.toggle("active",b.dataset.sec===id));
  requestAnimationFrame(flushVisiblePlots);   // render any charts now visible
}
$("#rep-nav").addEventListener("click",e=>{ const b=e.target.closest(".rep-navbtn"); if(b) showSection(b.dataset.sec); });
/* fold the report side-nav to a right-edge icon rail (hover expands to labels) */
(function(){
  const nav=$("#rep-nav"), fold=$("#rep-fold");
  let col=false; try{ col=localStorage.getItem("synthlab-repnav")==="1"; }catch(e){}
  const apply=()=>{ nav.classList.toggle("collapsed",col); fold.title=col?"Expand panel":"Collapse panel"; };
  apply();
  fold.addEventListener("click",()=>{ col=!col; try{ localStorage.setItem("synthlab-repnav",col?"1":"0"); }catch(e){} apply(); });
})();
/* left setup column: fold it away to give the canvas / report the full width.
   Folding changes the canvas width, so the data-model connectors are redrawn. */
(function(){
  const app=$("#app"), btn=$("#btn-left"), rail=$("#leftrail");
  let col=false; try{ col=localStorage.getItem("synthlab-left")==="1"; }catch(e){}
  const apply=()=>{ app.classList.toggle("leftfold",col);
    btn.title=btn.ariaLabel=col?"Show the setup panel":"Hide the setup panel";
    btn.classList.toggle("on",col);
    // the canvas is now wider/narrower — the SVG connectors are absolutely
    // positioned against it, so redraw once the width transition has settled
    if(DATA) setTimeout(()=>{ if($("#pane-model").classList.contains("active")) drawLinks(); },220); };
  const toggle=()=>{ col=!col; try{ localStorage.setItem("synthlab-left",col?"1":"0"); }catch(e){} apply(); };
  apply();
  btn.addEventListener("click",toggle); rail.addEventListener("click",toggle);
})();
$("#tabbar").addEventListener("click",e=>{
  const b=e.target.closest(".tab"); if(!b||b.classList.contains("locked")) return;
  activateTab(b.dataset.pane);
});
/* "How it Works" docs modal, opened from the Synthesizers panel link */
(function(){
  const backdrop=$("#docs-backdrop");
  const open=()=>backdrop.classList.add("show");
  const close=()=>backdrop.classList.remove("show");
  $("#link-how-works").addEventListener("click",e=>{ e.preventDefault(); open(); });
  $("#docs-close").addEventListener("click",close);
  backdrop.addEventListener("click",e=>{ if(e.target===backdrop) close(); });
  document.addEventListener("keydown",e=>{ if(e.key==="Escape") close(); });
})();

/* ---------------- run + poll ---------------- */
$("#btn-run").addEventListener("click",async()=>{
  if(!DATA) return;
  const schema={};
  for(const t of Object.keys(DATA.tables)){
    const sdtypes={};
    $$(`.sdtype-sel[data-table="${t}"]`).forEach(s=>{ if(s.value!==detected[t][s.dataset.col]) sdtypes[s.dataset.col]=s.value; });
    schema[t]={sdtypes, primary_key:$(`#pk-${t}`).value||null};
  }
  const targets={}; Object.keys(DATA.tables).forEach(t=>targets[t]=$(`#target-${t}`).value);
  const cap_sensitive={};
  Object.keys(DATA.tables).forEach(t=>{
    const v=($(`#cap-${t}`)||{}).value; if(v && v!=="auto") cap_sensitive[t]=v;
  });
  const cfg={schema, relationships:MODEL.rels.slice(), targets, cap_sensitive, constraints:collectConstraints(),
    pii:collectPii(),
    entity_key:MODEL.hub.key||"", entity_children:MODEL.hub.children.slice(),
    scd_effective:$("#in-scd-eff").value||"", scd_end:$("#in-scd-end").value||"", scd_current:$("#in-scd-cur").value||"",
    synths:[...selectedSynths],
    epochs:+$("#in-epochs").value, scale:+$("#in-scale").value, holdout:+$("#in-holdout").value};
  let r;
  try{ r=await apiFetch("/api/synthesize",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(cfg)}); }
  catch(e){ alert(BACKEND_HELP); return; }
  if(!r.ok){ alert((await r.json()).error||"failed to start"); return; }
  setRunningUI(true); setStatus("run","synthesizing");
  $("#console").classList.add("show"); $("#console").innerHTML="";
  $("#jobbar").classList.add("show"); $("#jobbar-fill").style.width="0%"; $("#jobbar-lbl").textContent="starting…";
  poll();
});
/* toggle the toolbar between Synthesize (idle) and Cancel (running) */
function setRunningUI(running){
  $("#btn-run").style.display = running ? "none" : "";
  $("#btn-run").disabled = running || !selectedSynths.size;
  $("#btn-cancel").style.display = running ? "" : "none";
  $("#btn-cancel").disabled = false;
  $("#btn-cancel").textContent = "■ Cancel";
}
$("#btn-cancel").addEventListener("click",async()=>{
  $("#btn-cancel").disabled=true; $("#btn-cancel").textContent="cancelling…";
  $("#jobbar-lbl").textContent="cancelling…"; setStatus("run","cancelling");
  try{ await apiFetch("/api/cancel",{method:"POST"}); }catch(e){}
  poll();   // pick up the 'cancelled' status right away
});
function setJobBar(pct,status){
  pct=Math.max(0,Math.min(100, pct||0));
  $("#jobbar-fill").style.width=pct+"%";
  const label=status==="done"?"complete":status==="error"?"failed":status==="cancelled"?"cancelled":"synthesizing…";
  $("#jobbar-lbl").textContent=`${label} ${pct.toFixed(0)}%`;
}
async function poll(){
  let j; try{ j=await (await apiFetch("/api/progress")).json(); }catch(e){ setStatus("err","lost backend"); return; }
  const con=$("#console");
  con.innerHTML=j.log.map((l,i)=>{
    const cls=l.startsWith("⚠")?"warn":l.startsWith("✗")||l.startsWith("■")?"err":l.startsWith("Done")?"ok":l.startsWith("⏳")?"":"";
    const cur=(i===j.log.length-1&&j.status==="running")?' cursor':'';
    return `<div class="ln ${cls}${cur}">${esc(l)}</div>`;
  }).join("");
  con.scrollTop=con.scrollHeight;
  setJobBar(j.pct, j.status);
  if(j.status==="running"){ setTimeout(poll,1500); return; }
  setRunningUI(false);
  if(j.status==="done"){ setJobBar(100,"done"); setStatus("done","complete");
    renderReport(await (await apiFetch("/api/results")).json());
    setTimeout(()=>{$("#console").classList.remove("show"); $("#jobbar").classList.remove("show");},1500); }
  else if(j.status==="cancelled"){ setStatus("","cancelled");
    setTimeout(()=>{$("#console").classList.remove("show"); $("#jobbar").classList.remove("show");},1600); }
  else { setStatus("err","failed — see log"); }
}

/* ---------------- report ---------------- */
const fig=(b64,cap="",cls="")=>b64?`<figure class="fig ${cls}"><img src="data:image/png;base64,${b64}">${cap?`<figcaption>${esc(cap)}</figcaption>`:""}</figure>`:"";
/* hover-to-explain marker — put the "why" here instead of on the screen */
const ihelp=(tip,cls="")=>`<i class="ihelp ${cls}" data-tip="${esc(tip)}">i</i>`;
/* One tooltip element on <body>, positioned against whichever marker is hovered.
   It must live outside the panels: both the left column and the report body
   scroll, and a tooltip inside a scrolling box gets clipped at its edge. */
(function(){
  const box=document.createElement("div");
  box.className="tipbox"; box.id="tipbox"; box.setAttribute("role","tooltip");
  document.body.appendChild(box);
  let cur=null;
  const hide=()=>{ box.classList.remove("show"); cur=null; };
  const show=el=>{
    const txt=el.getAttribute("data-tip"); if(!txt) return;
    cur=el; box.textContent=txt; box.classList.add("show");
    box.style.left="0px"; box.style.top="0px";              // measure unclamped
    const M=10, r=el.getBoundingClientRect(), b=box.getBoundingClientRect();
    let x=r.left+r.width/2-b.width/2;
    x=Math.max(M, Math.min(x, window.innerWidth-b.width-M));      // keep on screen
    let y=r.bottom+8;
    if(y+b.height>window.innerHeight-M) y=Math.max(M, r.top-b.height-8);   // flip above
    box.style.left=Math.round(x)+"px"; box.style.top=Math.round(y)+"px";
  };
  document.addEventListener("pointerover",e=>{
    const el=e.target.closest&&e.target.closest(".ihelp");
    if(el && el!==cur) show(el);
  });
  document.addEventListener("pointerout",e=>{
    const el=e.target.closest&&e.target.closest(".ihelp");
    if(el && el===cur) hide();
  });
  document.addEventListener("scroll",hide,true);   // capture: any scrolling ancestor
  window.addEventListener("resize",hide);
})();
/* section heading: short title + one-line subtitle, detail on hover */
const head=(title,tip,sub="")=>`<h4 class="block-title">${esc(title)}${tip?ihelp(tip):""}</h4>`
  +(sub?`<p class="note">${sub}</p>`:"");
const meter=(label,v)=>`<div class="meter"><div class="m-label"><span>${label}</span><b>${fmt(v)}</b></div>
  <div class="m-track"><div class="m-fill" data-w="${Math.max(0,Math.min(1,v||0))*100}" style="background:${meterColor(v||0)}"></div></div></div>`;

/* Condensed score strip shown above the figures/tables of every report section:
   one card per synthesizer with the section's headline 0-1 number, the literal
   arithmetic that produced it, and the components it is made of — the same
   numbers the leaderboard uses.

   `pick(v,s)` returns {score, formula, subs}.  `formula` is printed verbatim
   under the score (with this synthesizer's real numbers substituted in), so the
   headline can always be reconstructed by hand from what is on screen.
   `subs` is [[label, value, opts]]:
     opts.raw   value is a raw diagnostic, not a 0-1 score -> printed uncoloured
                (e.g. MIA attacker AUC, whose ideal is 0.5, not 1)
     opts.text  print this string instead of a number (counts, "3 panels", …)
     opts.sub   indent: a component *of* the line above, not a term in the mean */
const num=v=>(v==null||Number.isNaN(v))?null:+v;
function scoreStrip(res,P,pick,{unit="",note=""}={}){
  const S=res.summary||{};
  if(!Object.keys(S).length) return "";
  const cards=res.synths.filter(s=>S[s]).map(s=>{
    const {score,subs,formula}=pick(S[s],s);
    const v=num(score);
    const best=res.synths.filter(x=>S[x]).every(x=>(num(pick(S[x],x).score)??-1)<=(v??-1));
    return `<div class="kpi${best&&v!=null?" best":""}">
      <div class="kpi-head"><span class="dot" style="background:${P(s)}"></span>${esc(s)}
        ${best&&v!=null?`<span class="kpi-tag">best</span>`:""}</div>
      <div class="kpi-score" style="color:${v==null?"var(--faint)":meterColor(v)}">${fmt(v)}<small>${esc(unit)}</small></div>
      <div class="m-track"><div class="m-fill" data-w="${(Math.max(0,Math.min(1,v||0)))*100}"
        style="background:${meterColor(v||0)}"></div></div>
      ${formula?`<div class="kpi-formula">${formula}</div>`:""}
      <div class="kpi-subs">${(subs||[]).map(([lb,val,o])=>{
        const n=num(val), o2=o||{};
        const txt = o2.text!=null ? esc(o2.text) : (n==null ? "—" : fmt(n));
        const col = (o2.text!=null||n==null||o2.raw) ? "var(--muted)" : meterColor(n);
        return `<div class="${o2.sub?"is-sub":""}"${o2.tip?` title="${esc(o2.tip)}"`:""}>
          <span>${esc(lb)}</span><b style="color:${col}">${txt}</b></div>`;
      }).join("")}</div></div>`;
  }).join("");
  return `<div class="kpi-row">${cards}</div>${note?`<p class="note kpi-note">${note}</p>`:""}`;
}
/* the m-fill widths animate from 0 -> data-w once the bars are in the DOM */
function flushMeters(scope){ const sel=(scope?scope+" ":"")+".m-fill";
  requestAnimationFrame(()=>$$(sel).forEach(m=>m.style.width=m.dataset.w+"%")); }

/* ---------------- interactive charts (Plotly, PNG fallback) ---------------- */
const HAS_PLOTLY=typeof window.Plotly!=="undefined";
const RDYLGN=[[0,"#a50026"],[0.25,"#f46d43"],[0.5,"#fee08b"],[0.75,"#a6d96a"],[1,"#1a9850"]];
let PLOT_QUEUE=[];            // pending {id,type,data,extra} to render after innerHTML
const PLOTLY_REG=[];         // {id, heatmap} of rendered charts, for theme restyle
function safeId(s){return "pl_"+s.replace(/[^A-Za-z0-9]/g,"_");}
function maxLen(arr){ let m=0; for(const s of arr) m=Math.max(m,String(s).length); return m; }
function themeColors(){
  const cs=getComputedStyle(document.documentElement);
  const g=(n,d)=>(cs.getPropertyValue(n)||d).trim();
  return {ink:g("--ink","#122c42"), line:g("--line","#d5e2f0"),
          soft:g("--line-soft","#e7eef7")};
}

/* heatmap placeholder (shapes/pairs); PNG fig fallback.
   `cap` is the caption; `idkey` (defaults to cap) makes the DOM id unique when
   the same caption repeats (e.g. the same table across several synthesizers). */
function heatmapBlock(kind,cap,data,png,idkey){
  if(HAS_PLOTLY && data){
    const id=safeId(kind+"_"+(idkey||cap));
    PLOT_QUEUE.push({id,type:kind,data,cap});
    return `<div class="hm-wrap"><div class="hm-cap">${esc(cap)}</div><div id="${id}"></div></div>`;
  }
  return fig(png,cap,"scroll");
}
/* fidelity QualityReport — all three metrics in ONE figure; PNG fig fallback */
function qualityBlock(data,png){
  if(HAS_PLOTLY && data && data.tables.length){
    const id=safeId("quality_all");
    PLOT_QUEUE.push({id,type:"quality",data});
    return `<div class="hm-wrap"><div id="${id}"></div></div>`;
  }
  return fig(png,"QualityReport scores per synthesizer per table");
}

function renderHeatmap(el,kind,data,c){
  const x=(kind==="shapes")?data.x:data.labels;
  const y=(kind==="shapes")?data.y:data.labels;
  const z=data.z, nCol=x.length, nRow=y.length;
  const width=Math.max(560, 26*nCol+150);
  const height=(kind==="pairs")?Math.max(360,26*nRow+150):(80+52*nRow+Math.min(160,7*maxLen(x)));
  const trace={type:"heatmap", z, x, y, zmin:0, zmax:1, colorscale:RDYLGN, xgap:1, ygap:1, hoverongaps:false,
    hovertemplate:(kind==="shapes"?"col %{x}<br>%{y}: %{z:.3f}<extra></extra>":"%{x} × %{y}: %{z:.3f}<extra></extra>"),
    colorbar:{title:{text:kind==="shapes"?"shape":"pair sim",side:"right"},thickness:10,len:0.9}};
  const layout={width, height, margin:{l:130,r:20,t:10,b:110},
    paper_bgcolor:"rgba(0,0,0,0)", plot_bgcolor:c.line,
    font:{color:c.ink, size:10, family:"Spline Sans Mono, monospace"},
    xaxis:{tickangle:-90, automargin:true, ticks:""},
    yaxis:{automargin:true, autorange:"reversed", ticks:""}};
  window.Plotly.newPlot(el,[trace],layout,{displaylogo:false, responsive:false,
    modeBarButtonsToRemove:["select2d","lasso2d","autoScale2d"]});
}
/* one figure, three side-by-side panels (Overall / Column Shapes / Pair Trends)
   with a value label on every bar */
function renderQuality(el,data,c){
  const metrics=data.metrics;                       // [[key,label],...]
  const doms=[[0,0.30],[0.36,0.66],[0.72,1.0]];     // x-domain per panel
  const traces=[];
  metrics.forEach(([key,label],mi)=>{
    const sfx=mi===0?"":String(mi+1);
    data.synths.forEach(s=>{
      const yv=data.values[key][s]||[];
      traces.push({type:"bar", name:s, legendgroup:s, showlegend:mi===0,
        x:data.tables, y:yv, xaxis:"x"+sfx, yaxis:"y"+sfx,
        marker:{color:res_palette(s)},
        text:yv.map(v=>v==null?"":(+v).toFixed(2)), texttemplate:"%{text}",
        textposition:"outside", cliponaxis:false, textfont:{size:9},
        hovertemplate:s+" · %{x} · "+label+": %{y:.3f}<extra></extra>"});
    });
  });
  const nBars=Math.max(1,data.tables.length*data.synths.length);
  const width=Math.max(760, 3*(nBars*26+60));
  const layout={barmode:"group", width, height:360, margin:{l:40,r:12,t:20,b:80},
    paper_bgcolor:"rgba(0,0,0,0)", plot_bgcolor:"rgba(0,0,0,0)",
    font:{color:c.ink, size:10, family:"Spline Sans Mono, monospace"},
    legend:{orientation:"h", y:1.14, x:0.5, xanchor:"center", font:{size:10}},
    xaxis:{domain:doms[0], title:{text:metrics[0][1]}, gridcolor:c.soft, automargin:true},
    xaxis2:{domain:doms[1], title:{text:metrics[1][1]}, gridcolor:c.soft, automargin:true},
    xaxis3:{domain:doms[2], title:{text:metrics[2][1]}, gridcolor:c.soft, automargin:true},
    yaxis:{range:[0,1.18], gridcolor:c.soft, zeroline:false},
    yaxis2:{range:[0,1.18], gridcolor:c.soft, zeroline:false, anchor:"x2"},
    yaxis3:{range:[0,1.18], gridcolor:c.soft, zeroline:false, anchor:"x3"}};
  window.Plotly.newPlot(el,traces,layout,{displaylogo:false, responsive:false,
    modeBarButtonsToRemove:["select2d","lasso2d","autoScale2d"]});
}
/* palette resolver set per report render (synth -> hex) */
let _PALETTE={};
function res_palette(s){return _PALETTE[s]||"#888";}

/* render only the queued charts whose section is currently visible (sections
   render lazily, so a hidden div's 0-width doesn't break the Plotly layout);
   items are marked done so re-showing a section doesn't re-draw them */
function flushVisiblePlots(){
  if(!HAS_PLOTLY) return;
  const c=themeColors();
  for(const item of PLOT_QUEUE){
    if(item.done) continue;
    const el=document.getElementById(item.id); if(!el || !el.offsetParent) continue;   // not visible yet
    try{
      if(item.type==="quality") renderQuality(el,item.data,c);
      else renderHeatmap(el,item.type,item.data,c);
      if(!PLOTLY_REG.some(r=>r.id===item.id))
        PLOTLY_REG.push({id:item.id, heatmap:item.type!=="quality"});
      item.done=true;
    }catch(e){ el.parentElement.innerHTML=`<div class="note">chart failed: ${esc(String(e))}</div>`; item.done=true; }
  }
}
/* re-apply theme colours to already-rendered charts (called on light/dark toggle) */
function restylePlotly(){
  if(!HAS_PLOTLY) return;
  const c=themeColors();
  for(const {id,heatmap} of PLOTLY_REG){
    const el=document.getElementById(id); if(!el||!el.data) continue;
    const upd={"font.color":c.ink, "plot_bgcolor": heatmap ? c.line : "rgba(0,0,0,0)"};
    if(heatmap){ upd["xaxis.gridcolor"]=c.soft; upd["yaxis.gridcolor"]=c.soft; }
    else{ for(const a of ["xaxis","xaxis2","xaxis3","yaxis","yaxis2","yaxis3"]) upd[a+".gridcolor"]=c.soft; }
    try{ window.Plotly.relayout(el, upd); }catch(e){}
  }
}

/* ============================ business view ============================
   The report speaks two languages: a plain-language "business" view (default)
   and the full "technical" view.  VIEW drives both; LAST_RES lets the toggle
   re-render from stored results without re-fetching. */
let VIEW="business", LAST_RES=null;
const VERDICT={ready:{label:"Ready",cls:"ready"}, review:{label:"Review",cls:"review"},
               notready:{label:"Not ready",cls:"notready"}};
// score → verdict, using a "good" and an "ok" threshold
const scoreVerdict=(v,good,ok)=>(v==null||Number.isNaN(v))?null:(v>=good?"ready":v>=ok?"review":"notready");
// safety is a GATE, not a threshold: take the worst of the privacy checks that
// already carry PASS/WARN/FAIL verdicts (a single FAIL is a red flag)
function safetyVerdict(res,s){
  let worst="ready"; const tabs=(res.privacy||{})[s]||{};
  for(const t in tabs){ const vs=tabs[t].verdicts||{};
    for(const k in vs){ const st=vs[k].status;
      if(st==="FAIL") return "notready"; if(st==="WARN") worst="review"; } }
  return worst;
}
const worstVerdict=l=>l.includes("notready")?"notready":l.includes("review")?"review":(l.length?"ready":null);
const verdictBadge=(k,big)=>k?`<span class="verdict ${VERDICT[k].cls}${big?" lg":""}">${VERDICT[k].label}</span>`:"";
// the three business dimensions for one synthesizer
function bizDims(res,s){
  const sum=(res.summary||{})[s]||{};
  const fid=num((sum.fidelity||{}).score), util=num((sum.utility||{}).score), priv=num((sum.privacy||{}).score);
  return [
    {name:"Realism", q:"Does it look like real data?", score:fid, verdict:scoreVerdict(fid,0.8,0.6)},
    {name:"Safety", q:"Could it be traced to a real customer?", score:priv, verdict:safetyVerdict(res,s)},
    {name:"Usefulness", q:"Can teams use it like real data?", score:util, verdict:scoreVerdict(util,0.85,0.7)},
  ];
}
const bizPct=d=>d.score!=null?Math.round(d.score*100)+"%":"—";
function bizDimRow(d){
  const pct=d.score!=null?Math.round(d.score*100):0;
  return `<div class="bizdim"><div class="bizdim-name">${d.name}<small>${esc(d.q)}</small></div>
    <div class="bizdim-track"><div class="bizdim-fill" style="width:${pct}%;background:${meterColor(d.score||0)}"></div></div>
    <div class="bizdim-val">${bizPct(d)}</div>${verdictBadge(d.verdict)}</div>`;
}
// compact variant for the narrow per-generator cards: name + score + verdict
function bizDimMini(d){
  return `<div class="bizdim-mini"><span>${d.name}</span>
    <span class="bm-r"><span class="bm-pct">${bizPct(d)}</span>${verdictBadge(d.verdict)}</span></div>`;
}
// concrete go/no-go per use case, from the dimension verdicts
function bizUseCases(dims){
  const [realism,safety,useful]=dims;
  return [
    {label:"Dev / test environments", verdict:realism.verdict,
      reason: realism.verdict==="ready"?"realistic enough to build and test against":"resembles real data — check the flagged fields"},
    {label:"Vendor / partner sharing", verdict:safety.verdict,
      reason: safety.verdict==="ready"?"no privacy red flags in the safety checks":"review the privacy notes before sharing"},
    {label:"Training ML models", verdict:worstVerdict([useful.verdict,realism.verdict]),
      reason: useful.verdict==="ready"?"models train about as well as on real data":"usable, but less than real data"},
  ];
}
const useCaseChip=u=>`<div class="uc-chip ${VERDICT[u.verdict].cls}">
  <div class="uc-top">${VERDICT[u.verdict].cls==="ready"?"✓":VERDICT[u.verdict].cls==="review"?"!":"✕"} <b>${esc(u.label)}</b></div>
  <div class="uc-reason">${esc(u.reason)}</div></div>`;
// one factual line on why a generator ranks where it does
function bizWhy(res,s,lb){
  const ct=(res.cross_table||{})[s];
  // "crossable" = kept the entity consistent across ALL table pairs (no unaligned)
  const crossable = ct && ct.score!=null && !ct.note && !(ct.unaligned&&ct.unaligned.length);
  const my=bizDims(res,s);
  const bestFid=Math.max(...lb.map(r=>bizDims(res,r.synthesizer)[0].score||0));
  const bestUtil=Math.max(...lb.map(r=>bizDims(res,r.synthesizer)[2].score||0));
  const leads=[];
  if(my[0].score!=null && my[0].score>=bestFid-1e-9) leads.push("realism");
  if(crossable) leads.push("links across tables");
  if(my[2].score!=null && my[2].score>=bestUtil-1e-9) leads.push("usefulness");
  if(leads.length) return "Leads on "+leads.slice(0,2).join(" and ")+".";
  return crossable ? "Keeps customers consistent across tables."
                   : "Single-table — links across tables aren't preserved.";
}
// one plain-English recommendation sentence built from the verdicts
function recommendation(res,s,dims){
  const [realism,safety,useful]=dims;
  const overall=worstVerdict(dims.map(d=>d.verdict));
  const usefulPct=useful.score!=null?Math.round(useful.score*100)+"%":"—";
  if(overall==="notready"){
    const bad=dims.filter(d=>d.verdict==="notready").map(d=>d.name.toLowerCase());
    return `<b>Not ready to use as-is.</b> ${bad.join(" and ")} ${bad.length>1?"need":"needs"} `
      + `attention — review the flagged checks before sharing or training on this data.`;
  }
  const realPhrase=realism.verdict==="ready"?"behaves like your real data":"roughly matches your real data";
  const safePhrase=safety.verdict==="ready"?"carries no privacy red flags":"has privacy notes worth a look";
  const close=overall==="ready"
    ? "Recommended for dev/test environments, vendor sharing, and model training."
    : "Fine for exploration; review the flagged items before production use.";
  return `This synthetic data <b>${realPhrase}</b>, <b>${safePhrase}</b>, and is about `
    + `<b>${usefulPct}</b> as useful as real data for analytics. ${close}`;
}
const NAV_LABELS={
  business:{"sec-overview":"Summary","sec-quality":"Realism","sec-shapes":"Fields match",
    "sec-pairs":"Field relationships","sec-ri":"Records link up",
    "sec-utility":"Usefulness","sec-privacy":"Safety"},
  technical:{"sec-overview":"Leaderboard","sec-quality":"Fidelity","sec-shapes":"Column Shapes",
    "sec-pairs":"Column Pair Trends","sec-ri":"Referential Integrity",
    "sec-utility":"Utility","sec-privacy":"Privacy"}};
function applyNavLabels(){
  const m=NAV_LABELS[VIEW]||NAV_LABELS.technical;
  document.querySelectorAll("#rep-nav .rep-navbtn").forEach(b=>{
    const sec=b.dataset.sec, txt=b.querySelector(".txt");
    if(sec&&txt&&m[sec]){ txt.textContent=m[sec]; b.setAttribute("title",m[sec]); }
  });
}
(function wireViewToggle(){
  const tg=document.getElementById("view-toggle"); if(!tg) return;
  tg.addEventListener("click",e=>{
    const btn=e.target.closest(".vt-btn"); if(!btn||btn.dataset.view===VIEW) return;
    VIEW=btn.dataset.view;
    tg.querySelectorAll(".vt-btn").forEach(b=>b.classList.toggle("active",b===btn));
    const hint=document.getElementById("view-hint");
    if(hint) hint.setAttribute("data-tip", VIEW==="business"
      ? "Plain-language summary. Switch to Technical for the full metrics."
      : "Full metrics and formulas. Switch to Business for the plain-language view.");
    if(LAST_RES){ const cur=(document.querySelector(".rep-sec.active")||{}).id||"sec-overview";
      renderReport(LAST_RES); showSection(cur); }
  });
})();

function renderReport(res){
  LAST_RES=res;
  lockReportTabs(false);
  const P=n=>res.palette[n]||PALETTE[n]||"#888";
  const dot=n=>`<span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:${P(n)};margin-right:7px"></span>`;
  // structure (referential integrity) only exists once tables are linked; without
  // it, fidelity is the column statistics alone and the RI tiles read "—".
  const hasStruct=Object.values(res.summary||{}).some(v=>{
    const s=v.fidelity&&v.fidelity.structure; return s!=null && !Number.isNaN(s); });
  PLOTLY_REG.length=0; PLOT_QUEUE=[];        // fresh chart registry + queue per report
  _PALETTE={}; (res.synths||[]).forEach(s=>{_PALETTE[s]=P(s);});

  /* synthetic data list (left) */
  let sl=`<table class="ftable"><thead><tr><th>File Name</th><th>Action</th></tr></thead><tbody>`;
  let count=0;
  for(const s of res.synths) for(const t of res.tables){
    sl+=`<tr><td><div class="fname">${s}·${t}.csv</div><div class="fmeta">synthetic</div></td>
      <td class="fact"><a class="iact" href="/api/download/${s}/${t}?sid=${encodeURIComponent(SID)}" title="download">↓</a></td></tr>`; count++;
  }
  sl+=`</tbody></table>`;
  $("#synth-list").innerHTML=sl; $("#synthetic-panel").style.display="block";
  $("#synth-count").textContent=count; $("#synth-count").className="count";

  applyNavLabels();

  /* --- Leaderboard (overview) --- */
  const lb=[...res.leaderboard].sort((a,b)=>(b.overall??0)-(a.overall??0));
  if(VIEW==="business"){
    const top=lb[0], tDims=bizDims(res,top.synthesizer);
    const tOverall=worstVerdict(tDims.map(d=>d.verdict));
    const useCases=bizUseCases(tDims);
    let bh=`<div class="exec ${tOverall||""}">
      <div class="exec-top"><h3>Trust assessment</h3>${verdictBadge(tOverall,true)}
        <span class="winner">best generator: ${dot(top.synthesizer)}${esc(top.synthesizer)}</span></div>
      <p class="exec-rec">${recommendation(res,top.synthesizer,tDims)}</p>
      <div class="bizdims">${tDims.map(bizDimRow).join("")}</div>
      <div class="uc-label">Ready for</div>
      <div class="uc-row">${useCases.map(useCaseChip).join("")}</div></div>`;
    if(lb.length>1){
      bh+=`<p class="biz-note">Every generator we tried, ranked by overall trust:</p>
        <div class="podium">${lb.map((r,i)=>{
          const ds=bizDims(res,r.synthesizer), ov=worstVerdict(ds.map(d=>d.verdict));
          return `<div class="lb-card ${i===0?"first":""}"><div class="rank">${i+1}</div>
            <div class="lb-name"><span class="dot" style="background:${P(r.synthesizer)}"></span>${esc(r.synthesizer)}</div>
            <div style="margin:9px 0 10px">${verdictBadge(ov,true)}</div>
            <p class="lb-why">${esc(bizWhy(res,r.synthesizer,lb))}</p>
            <div class="bizdims mini">${ds.map(bizDimMini).join("")}</div></div>`;
        }).join("")}</div>`;
    }
    $("#sec-overview").innerHTML=bh;
    flushMeters("#sec-overview");
  } else {
  $("#sec-overview").innerHTML=`<div class="podium">${lb.map((r,i)=>`
    <div class="lb-card ${i===0?"first":""}"><div class="rank">${i+1}</div>
      <div class="lb-name"><span class="dot" style="background:${P(r.synthesizer)}"></span>${r.synthesizer}</div>
      <div class="big-score">${fmt(r.overall)}<small> /1 overall</small></div>
      ${meter("fidelity",r.fidelity)}${meter("privacy",r.privacy)}${meter("utility · TSTR",r.utility_tstr)}</div>`).join("")}</div>
    <p class="note"><b>overall</b> = mean of the three dimensions${ihelp(
      "fidelity = the column statistics (sdmetrics QualityReport)"
      + (hasStruct ? ", two parts to one part referential integrity (cardinality shape similarity)" : "")
      + ". privacy = the mean of MIA protection, NewRowSynthesis and CategoricalCAP. "
      + "utility = the TSTR/TRTR ratio. Each dimension is broken down on its own tab; see METRICS.md.")}</p>`;
  flushMeters("#sec-overview");
  }

  /* --- Fidelity (score strip + summary table + combined Overall/Shapes/Pairs figure) --- */
  const shapesData=res.figures.shapes_data||{}, pairsData=res.figures.pairs_data||{};
  let qy=scoreStrip(res,P,v=>{
    const f=v.fidelity, st=num(f.structure);
    return {score:f.score,
      formula: st==null
        ? `<b>${fmt(f.columns)}</b> column statistics<br>(no relationships defined → referential integrity not scored)`
        : `( 2 × <b>${fmt(f.columns)}</b> columns + <b>${fmt(st)}</b> ref. integrity ) / 3 = <b>${fmt(f.score)}</b>`,
      subs:[
        ["Column statistics ×2", f.columns, {tip:"QualityReport overall = mean of the two lines below, averaged over tables"}],
        ["Column Shapes", f.column_shapes, {sub:true, tip:"marginal distribution match per column"}],
        ["Column Pair Trends", f.column_pair_trends, {sub:true, tip:"pairwise relationship match per column pair"}],
        ["Referential integrity ×1", f.structure, {tip:hasStruct?"cardinality shape similarity — how closely child-rows-per-parent matches real. See the Referential Integrity tab.":"no relationships defined"}],
      ]};
    }, {unit:"/1 fidelity", note: hasStruct
      ? `2 parts column statistics, 1 part referential integrity${ihelp(
          "The column statistics only ever look inside one table, so a synthesizer can match every marginal "
          + "while getting the cross-table structure wrong. Referential integrity (cardinality shape "
          + "similarity) covers that, at half the weight.")}`
      : `Column statistics only${ihelp(
          "No relationships are defined, so there is no cross-table structure to score. "
          + "Link tables in the Data Model tab to have referential integrity counted here too.")}`})
    +head("QualityReport","Per synthesizer per table. Overall is the mean of column shapes and column pair trends.")
    +`<table class="rep"><thead><tr><th>synthesizer</th><th>table</th><th style="text-align:right">column shapes</th>
    <th style="text-align:right">column pair trends</th><th style="text-align:right">overall</th>
    <th style="text-align:right" title="Referential integrity (cardinality shape similarity) — how closely child-rows-per-parent matches real. It is ONE score per synthesizer (not per table), spanning the synthesizer's rows here, and it feeds fidelity at 1/3 weight. n/a when no relationships are defined. See the Referential Integrity tab.">ref. integrity</th></tr></thead><tbody>`;
  for(const s of res.synths){
    const entries=Object.entries(res.quality[s]||{});
    const ri=num(((((res.summary||{})[s]||{}).fidelity)||{}).structure);
    entries.forEach(([t,v],i)=>{
      qy+=`<tr><td class="mono">${dot(s)}${s}</td>
        <td class="mono dim">${t}</td><td class="score-cell">${fmt(v.column_shapes)}</td>
        <td class="score-cell">${fmt(v.column_pair_trends)}</td>
        <td class="score-cell" style="color:${meterColor(v.overall)}">${fmt(v.overall)}</td>`;
      if(i===0) qy+=`<td class="score-cell" rowspan="${entries.length}"
        style="border-left:1px solid var(--line); color:${ri==null?"var(--faint)":meterColor(ri)}">${ri==null?"n/a":fmt(ri)}</td>`;
      qy+=`</tr>`;
    });
  }
  qy+=`</tbody></table>`+qualityBlock(res.figures.quality_data, res.figures.quality);
  $("#sec-quality").innerHTML=qy;

  /* --- Fidelity › Column Shapes (per-column heatmaps) --- */
  let sh=scoreStrip(res,P,v=>({score:v.fidelity.column_shapes,
      formula:`mean of the per-column scores below`,
      subs:[
        ["Feeds column statistics", v.fidelity.columns, {tip:"mean(Column Shapes, Column Pair Trends)"}],
        ["…which feeds fidelity", v.fidelity.score, {sub:true}],
      ]}), {unit:"/1 shapes"})
    +head("Column Shapes",
      "Does each column's distribution match real? KS complement for numeric columns, "
      + "total-variation complement for categorical. 1 = identical."
      + (HAS_PLOTLY ? " Hover a cell for the exact score; drag to zoom." : ""),
      "Per column, per synthesizer.");
  const shapeFigs=res.figures.shapes||{};
  if(Object.keys(shapeFigs).length || Object.keys(shapesData).length)
    for(const t of Object.keys({...shapeFigs,...shapesData})) sh+=heatmapBlock("shapes",t,shapesData[t],shapeFigs[t]);
  else sh+=`<p class="note">No per-column shape scores were available for these tables.</p>`;
  $("#sec-shapes").innerHTML=sh;

  /* --- Fidelity › Column Pair Trends (one heatmap per synthesizer per table) --- */
  const pairs=res.figures.pairs||{};   // {synth:{table:png}} ; pairsData same shape
  const hasMap=m=>m&&Object.values(m).some(inner=>inner&&Object.values(inner).some(Boolean));
  const pairStrip=scoreStrip(res,P,v=>({score:v.fidelity.column_pair_trends,
      formula:`mean of the per-pair scores below`,
      subs:[
        ["Feeds column statistics", v.fidelity.columns, {tip:"mean(Column Shapes, Column Pair Trends)"}],
        ["…which feeds fidelity", v.fidelity.score, {sub:true}],
      ]}), {unit:"/1 pair trends"});
  let pr=pairStrip;                 // the score exists even when no heatmap could be drawn
  if(hasMap(pairs) || hasMap(pairsData)){
    pr+=head("Column Pair Trends",
      "Do the relationships BETWEEN columns survive? Correlation similarity for numeric pairs, "
      + "contingency similarity for categorical/mixed. 1 = identical trend; a blank cell is a pair "
      + "sdmetrics could not score.",
      "Per column pair, per synthesizer.");
    for(const s of res.synths){
      const pm=pairs[s]||{}, dm=pairsData[s]||{};
      const tabs=Object.keys({...pm,...dm}); if(!tabs.length) continue;
      pr+=`<h4 class="block-title" style="font-size:15px; margin-top:22px">
        <span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${P(s)};margin-right:8px"></span>${esc(s)}</h4>`;
      for(const t of tabs) pr+=heatmapBlock("pairs",t,dm[t],pm[t],`${s}_${t}`);
    }
  } else pr+=`<p class="note">No column-pair trends were available for these tables.</p>`;
  $("#sec-pairs").innerHTML=pr;

  /* --- Fidelity › Referential Integrity (forward + reverse coverage + cardinality) --- */
  const ri=res.referential||[];
  let riH;
  if(!ri.length){
    riH=`<p class="note">No relationships defined — link tables (or build an entity-key hub) in the
      <b>Data Model</b> tab to measure referential integrity.</p>`;
  }else{
    const realRev={}; ri.forEach(r=>{ if(r.source==="real") realRev[r.relationship]=r.parent_coverage; });
    riH=scoreStrip(res,P,v=>{
      const t=v.structure||{};
      const gate=t.fk_gate||"n/a";
      return {score:t.score,
        formula: num(t.cardinality_shape)!=null
          ? `<b>${fmt(t.cardinality_shape)}</b> cardinality shape similarity`
          : `cardinality similarity unavailable → not scored`,
        subs:[
          ["Cardinality shape", t.cardinality_shape, {tip:"sdmetrics CardinalityShapeSimilarity — how closely the distribution of child-rows-per-parent matches real. This is the score."}],
          ["FK validity", t.fk_validity, {raw:true,
            text: gate==="n/a" ? "n/a" : `${fmt(t.fk_validity)} · ${gate}`,
            tip: gate==="n/a"
              ? "Not scored. The parent is a hub derived from the keys this synthesizer emitted, so every child key is in it by construction — a flat 1.0 for everyone."
              : "A pass/fail gate, not part of the score: child rows whose FK hits a parent key. Validity is a constraint, not a similarity."}],
          ["Participation", t.participation, {raw:true,
            tip:"Diagnostic, not scored: 1 − |synth parent coverage − real parent coverage|. Parent coverage is one point on the CDF of the distribution cardinality shape already measures in full, so scoring both would double-count."}],
        ]};
      }, {unit:"/1 ref. integrity", note:`Scored on cardinality shape only${ihelp(
        "Cardinality shape is the only one of the three that is a distribution-similarity measure, on the "
        + "same footing as Column Shapes — so it is the only one averaged into fidelity (one third of it). "
        + "FK validity is a constraint, not a similarity: averaging it in would let a model buy its way out "
        + "of orphan rows with good marginals, so it is a pass/fail gate — and it is true by construction, "
        + "hence unscoreable, whenever the parent is a derived hub. Participation is a marginal of the "
        + "cardinality distribution, so scoring it too would double-count.")}`})
      +head("Coverage per relationship",
        "fk coverage (forward) = child rows whose foreign key hits a parent key; 1 = no orphans, and this "
        + "drives the status. parent coverage (reverse) = parents with at least one child row. Parent "
        + "coverage is NOT supposed to be 1 — each synthesizer should match the real value, and the badge "
        + "shows how many points away from real it landed."
        + (res.synths.includes("HMA")
            ? " HMA was fitted WITH these relationships, so it preserves them by construction; single-table "
              + "synthesizers reference a hub derived from their own output."
            : " No multi-table model was selected, so no synthesizer here learned these relationships — "
              + "referential integrity is being measured, not enforced."))
      +`<table class="rep"><thead><tr><th>synthesizer</th><th>relationship</th>
        <th style="text-align:right">fk coverage</th><th style="text-align:right">parent coverage</th><th>status</th></tr></thead>
      <tbody>${[...ri].sort((a,b)=>
        (a.source==="real"?-1:res.synths.indexOf(a.source))-(b.source==="real"?-1:res.synths.indexOf(b.source))
        || a.relationship.localeCompare(b.relationship)).map(r=>{
        let rc=pct(r.parent_coverage);
        const real=realRev[r.relationship];
        if(r.source!=="real" && real!=null && r.parent_coverage!=null){
          const d=r.parent_coverage-real, ad=Math.abs(d);      // signed diff vs real, in fraction
          const cls=ad<=0.05?"ok":ad<=0.15?"warn":"bad";
          const label=ad<0.005 ? "= real" : (d>0?"↑":"↓")+" "+(ad*100).toFixed(1)+" pts";
          rc=`${pct(r.parent_coverage)}<span class="delta-badge ${cls}" title="real: ${pct(real)} — difference vs real">${label}</span>`;
        }
        return `<tr><td class="mono">${dot(r.source)}${esc(r.source)}</td><td class="mono dim">${esc(r.relationship)}</td>
          <td class="score-cell">${pct(r.fk_coverage)}</td><td class="score-cell">${rc}</td><td>${pill(r.status)}</td></tr>`;
      }).join("")}</tbody></table>`;
    const card=res.cardinality||{}; const cnames=Object.keys(card);
    if(cnames.length){
      riH+=head("Cardinality similarity",
          "The distribution of child rows per parent (including parents with none), synthetic vs real. "
          + "1 = identical. This catches a synthesizer that over- or under-generates child rows even when "
          + "forward coverage is a perfect 1. Shape compares the whole distribution; statistic compares its mean.")
        +`<table class="rep"><thead><tr><th>synthesizer</th><th style="text-align:right">shape similarity</th>
          <th style="text-align:right">statistic similarity</th></tr></thead><tbody>`;
      const cell=v=>v==null?`<span class="dim">—</span>`:`<span style="color:${meterColor(v)}">${fmt(v)}</span>`;
      for(const s of cnames){ const e=card[s];
        riH+=`<tr><td class="mono">${dot(s)}${esc(s)}</td><td class="score-cell">${cell(e.shape)}</td>
          <td class="score-cell">${cell(e.statistic)}</td></tr>`; }
      riH+=`</tbody></table>`;
    }
  }
  $("#sec-ri").innerHTML=riH;

  /* --- Utility (ML efficacy · TSTR) --- */
  if(!res.efficacy.length){ $("#sec-utility").innerHTML=`<h4 class="block-title">Utility · ML efficacy</h4><p class="note">No usable modelling target found.</p>`; }
  else{
    let ml=scoreStrip(res,P,v=>{
      const u=v.utility, pt=u.per_table||{};
      const subs=Object.keys(pt).map(t=>[t, pt[t].ratio,
        {sub:false, tip:`mean of this table's ${pt[t].panels} panel ratio(s)`}]);
      if(u.n_capped) subs.push(["of which capped at 1.00", null,
        {text:`${u.n_capped} panel${u.n_capped>1?"s":""}`,
         tip:"the synthesizer beat the real-trained baseline on these panels; the ratio is capped at 1, never above"}]);
      // context, NOT terms of the mean: a high ratio against a weak baseline still
      // means a weak model — mean(synth)/mean(real) is deliberately not the score
      subs.push(["mean synth score (context)", u.synth, {sub:true, raw:true,
        tip:"raw mean of the synthetic-trained scores — context only, not a term in the mean"}]);
      subs.push(["mean real baseline (context)", u.real, {sub:true, raw:true,
        tip:"raw mean of the real-trained scores. If this is low, the target is hard for ANY model and a high utility ratio only means the synthetic data is as weak as the real data — not that the model is good."}]);
      return {score:u.score,
        formula:`mean of <b>${u.n_panels}</b> panel ratios = <b>${fmt(u.score)}</b>`,
        subs};
      }, {unit:"/1 utility", note:`Mean of one ratio per table × metric${ihelp(
        "Each row of every table below is one panel, and its ratio is clip(synth ÷ real, 0, 1). The headline "
        + "is the plain mean of those ratios — NOT mean(synth) ÷ mean(real), which gives a different answer "
        + "because every panel counts equally regardless of how large its scores are. The ÷ real columns "
        + "below show every term in that mean. Utility is relative: against a weak real baseline, a high "
        + "ratio only says the synthetic data is as weak as the real data.")}`})
      +head("ML efficacy (TSTR)",
        "Train on Synthetic, Test on Real. Each model is trained on the real data (the reference) and on "
        + "every synthesizer's output, then tested on the SAME real holdout — which the synthesizers never saw.",
        "Trained per source, tested on the same real holdout.");
    /* each (table × metric) row is one panel of the utility mean; the "÷ real"
       column after every synthesizer prints that panel's own term, so the
       headline can be added up by hand from the rows on screen. */
    const panelOf=(s,t,m)=>{ const u=((res.summary||{})[s]||{}).utility;
      return (u&&u.panels||[]).find(p=>p.table===t&&p.metric===m)||null; };
    const perTable=(s,t)=>((((res.summary||{})[s]||{}).utility||{}).per_table||{})[t];
    const ratioCell=(p,extra="")=>{
      if(!p) return `<td class="ratio-cell dim"${extra}>—</td>`;
      const cap=p.capped?`<span class="cap" title="raw ratio ${fmt(p.raw_ratio,2)} — capped at 1.00">⤒</span>`:"";
      return `<td class="ratio-cell" style="color:${meterColor(p.ratio)}"
        title="${fmt(p.synth,3)} ÷ ${fmt(p.real,3)} = ${fmt(p.raw_ratio,3)}"${extra}>${fmt(p.ratio)}${cap}</td>`;
    };
    const tabList=[...new Set(res.efficacy.map(r=>r.table))];
    for(const t of tabList){
      const rows=res.efficacy.filter(r=>r.table===t);
      const metrics=[...new Set(rows.map(r=>r.metric))];
      const synths=res.synths.filter(s=>rows.some(r=>r.train_on===s));
      const at=(m,s)=>rows.find(r=>r.metric===m&&r.train_on===s)||null;
      const notes=new Set();
      // Grouped header: one 2-column block per synthesizer (its score, and that
      // score ÷ the real baseline).  The gap columns are dropped — gap = real −
      // synth is the same comparison in absolute points, and it is the ratio that
      // actually feeds the utility score.  Gaps remain in the exported CSV.
      ml+=`<h4 class="block-title">${t} <span style="font-family:var(--mono);font-size:12px;color:var(--faint);font-weight:400">target ${rows[0].target} · ${rows[0].task}</span></h4>
        <table class="rep eff"><thead>
          <tr><th rowspan="2">metric</th>
            <th rowspan="2" class="grp" style="text-align:right">real<div class="hsub">baseline</div></th>
            ${synths.map(s=>`<th colspan="2" class="grp" style="text-align:center">
              <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${P(s)};margin-right:6px"></span>${esc(s)}</th>`).join("")}</tr>
          <tr>${synths.map(()=>`<th style="text-align:right" class="grp">score</th>
            <th style="text-align:right" title="clip(score ÷ real, 0, 1) — this panel's term in the utility mean">÷ real</th>`).join("")}</tr>
        </thead><tbody>`;
      for(const m of metrics){
        const rr=at(m,"real");
        ml+=`<tr><td class="mono">${esc(m)}</td>
          <td class="score-cell grp dim">${fmt(rr?rr.score:null)}</td>`;
        for(const s of synths){
          const row=at(m,s), v=row?row.score:null, note=row&&row.note?row.note:"";
          if(v==null&&note) notes.add(note);
          ml+=`<td class="score-cell grp${(v==null&&note)?" err":""}"${note?` title="${esc(note)}"`:""}>${(v==null&&note)?"error":fmt(v)}</td>`
            + ratioCell(panelOf(s,t,m));
        }
        ml+=`</tr>`;
      }
      // what this table contributes to the utility mean
      ml+=`<tr class="total-row"><td class="mono">mean ratio · ${esc(t)}</td><td class="grp"></td>`
        + synths.map(s=>{ const e=perTable(s,t);
            return `<td class="score-cell grp dim" style="font-weight:400">${e?e.panels+" panels":""}</td>
              <td class="ratio-cell" style="color:${e?meterColor(e.ratio):"inherit"}"
                title="${e?`mean of this table's ${e.panels} panel ratios`:""}">${e?fmt(e.ratio):"—"}</td>`;
          }).join("")+`</tr></tbody></table>
        <p class="note tbl-legend"><b>÷ real</b> = one term of the utility mean · <b>⤒</b> = capped at 1.00
          ${ihelp("score = trained on that source, tested on the real holdout. ÷ real = that score divided by "
            + "the real baseline, clipped to 1 — one term of the utility mean. ⤒ means the synthesizer BEAT "
            + "the real baseline here, so the ratio was capped at 1.00; hover the cell for the raw value. "
            + "Absolute gaps (real − synth) are in the downloaded CSV.")}</p>`;
      if(notes.size) ml+=`<div class="efflog">`+[...notes].map(n=>"⚠ "+esc(n)).join("\n")+`</div>`;
    }
    // and the roll-up: the mean over every panel of every table = the headline
    ml+=head("Roll-up",
        "Each mean-ratio row above lands here. The utility score is the mean over EVERY panel, not the mean "
        + "of the per-table columns — so a table with more metrics carries proportionally more weight.")
      +`<table class="rep"><thead><tr><th>synthesizer</th>
        ${tabList.map(t=>`<th style="text-align:right">${esc(t)}</th>`).join("")}
        <th style="text-align:right" class="grp">panels</th>
        <th style="text-align:right">utility</th></tr></thead><tbody>`;
    for(const s of res.synths){
      const u=((res.summary||{})[s]||{}).utility; if(!u) continue;
      ml+=`<tr><td class="mono">${dot(s)}${esc(s)}</td>`
        + tabList.map(t=>{ const e=(u.per_table||{})[t];
            return `<td class="ratio-cell" style="color:${e?meterColor(e.ratio):"inherit"}"
              title="${e?`mean of ${e.panels} panel ratios`:""}">${e?fmt(e.ratio):"—"}</td>`; }).join("")
        + `<td class="score-cell grp dim">${u.n_panels}</td>`
        + `<td class="ratio-cell" style="color:${meterColor(u.score||0)}; font-size:14px">${fmt(u.score)}</td></tr>`;
    }
    ml+=`</tbody></table>`;
    ml+=fig(res.figures.efficacy,"TSTR scores per training source");
    $("#sec-utility").innerHTML=ml;
  }

  /* --- Privacy --- */
  let pv=scoreStrip(res,P,v=>{
    const p=v.privacy;
    const terms=[["MIA",p.mia_protection],["new-row",p.new_row_synthesis],["CAP",p.categorical_cap]]
      .filter(([,x])=>num(x)!=null);
    return {score:p.score,
      formula: terms.length
        ? `mean( ${terms.map(([lb,x])=>`<b>${fmt(x)}</b> ${lb}`).join(" , ")} ) = <b>${fmt(p.score)}</b>`
        : `no privacy checks scored`,
      subs:[
        ["MIA protection", p.mia_protection, {tip:"1 − 2|AUC − 0.5| — this is the term in the mean, not the raw AUC"}],
        ["from attacker AUC", p.mia_auc, {sub:true, raw:true, tip:"raw membership-inference AUC — ideal is 0.5 (a coin flip), so it is NOT averaged directly"}],
        ["NewRowSynthesis", p.new_row_synthesis, {tip:"share of synthetic rows that are not copies of a real row on the evaluated (modelable) columns"}],
        ["real-holdout ceiling", p.new_row_baseline, {sub:true, raw:true,
          tip:"what a REAL holdout scores against the training rows on the same columns — the achievable ceiling. When the evaluated columns hold few distinct combinations, even real rows duplicate each other; a low NewRowSynthesis near this ceiling is expected, not leakage."}],
        ["CategoricalCAP", p.categorical_cap, {tip:"protection against attribute inference — judged against the real-holdout ceiling below, not an absolute bar"}],
        ["real-holdout ceiling", p.categorical_cap_baseline, {sub:true, raw:true,
          tip:"what a REAL holdout scores under the same attack — the achievable ceiling. When the sensitive field is guessable from the real data's own distribution (imbalance, correlations), even real rows score low; a synthetic score near this ceiling means the generator adds no leakage beyond population statistics."}],
      ]};
    }, {unit:"/1 privacy", note:`Mean of the three protection scores${ihelp(
      "The MIA term is 1 − 2|AUC − 0.5|, not the AUC itself: the raw attacker AUC is shown indented beneath "
      + "it because its ideal is 0.5 (a coin flip), so both a strong attacker (AUC 1.0) and an inverted one "
      + "(AUC 0.0) are penalised. NewRowSynthesis = synthetic rows that are not copies of a real row. "
      + "CategoricalCAP = protection against attribute inference.")}`})
    +fig(res.figures.privacy,"NewRowSynthesis (ideal 1) · Membership-Inference attacker AUC (ideal 0.5) · CategoricalCAP (ideal 1)");
  // flatten, then sort synthesizer → table → check so each synth reads as one block
  const pvRows=[];
  for(const [s,tabs] of Object.entries(res.privacy)) for(const [t,rep] of Object.entries(tabs))
    for(const [chk,v] of Object.entries(rep.verdicts||{}))
      pvRows.push({s,t,chk,status:v.status,detail:v.detail});
  pvRows.sort((a,b)=>a.s.localeCompare(b.s)||a.t.localeCompare(b.t)||a.chk.localeCompare(b.chk));
  const pvCols=["s","t","chk","status"], pvLabels=["synthesizer","table","check","status"];
  const opts=k=>[...new Set(pvRows.map(r=>r[k]))].sort()
    .map(v=>`<option value="${esc(v)}">${esc(v)}</option>`).join("");
  pv+=`<table class="rep" id="priv-checks"><thead><tr>
      ${pvCols.map((k,i)=>`<th>${pvLabels[i]}
        <select class="colfilter" data-col="${k}" title="filter ${pvLabels[i]}">
          <option value="">all</option>${opts(k)}</select></th>`).join("")}
      <th>detail</th></tr></thead><tbody>`;
  for(const r of pvRows)
    pv+=`<tr data-s="${esc(r.s)}" data-t="${esc(r.t)}" data-chk="${esc(r.chk)}" data-status="${esc(r.status)}">
      <td class="mono">${dot(r.s)}${esc(r.s)}</td><td class="mono dim">${esc(r.t)}</td><td class="mono">${esc(r.chk)}</td>
      <td>${pill(r.status)}</td><td class="dim" style="font-size:11.5px">${esc(r.detail)}</td></tr>`;
  pv+=`</tbody></table>`;
  $("#sec-privacy").innerHTML=pv;
  // wire the header filters: a row must match every active dropdown to stay visible
  const pvTable=$("#sec-privacy").querySelector("#priv-checks");
  if(pvTable){
    const filters=[...pvTable.querySelectorAll(".colfilter")];
    const apply=()=>{
      const want=filters.map(f=>[f.dataset.col,f.value]).filter(([,v])=>v);
      filters.forEach(f=>f.classList.toggle("active",!!f.value));
      for(const tr of pvTable.tBodies[0].rows)
        tr.style.display=want.every(([k,v])=>tr.dataset[k]===v)?"":"none";
    };
    filters.forEach(f=>f.addEventListener("change",apply));
  }

  if(VIEW==="business") applyBizIntros(res);         // plain-language header + at-a-glance verdicts
  flushMeters();                                   // animate every score bar, all sections
  activateTab("pane-report"); showSection("sec-overview");
}
// plain-language intro prepended to each metric section in the business view:
// what the tab answers and how to read it, so a non-technical viewer who drills
// in past the summary still lands on plain language, not a heatmap.
const BIZ_INTRO={
  "sec-quality":{t:"Realism — does it look like real data?",
    b:"How closely the synthetic data resembles the real data, field by field and overall. In the tables below, higher is closer to real; the referential-integrity column shows how well records link across tables."},
  "sec-shapes":{t:"Fields match — is each field realistic on its own?",
    b:"For every column — age, region, status — does the spread of values look like the real one? Green cells match real; red cells are fields the generator reproduced poorly."},
  "sec-pairs":{t:"Field relationships — do fields move together correctly?",
    b:"Real data has patterns between fields — older customers are married more often, say. This checks whether those survived. Green kept the pattern, red lost it, blank means there was no real pattern to keep."},
  "sec-ri":{t:"Records link up — do the tables connect correctly?",
    b:"Every record should point to a real customer, and each customer should have a realistic number of records. The grey 'real' row is the target — matching it is the goal, not scoring 100%."},
  "sec-utility":{t:"Usefulness — can teams work with it like real data?",
    b:"We train the same model twice — once on real data, once on synthetic — and test both on real data held back. A score near 1.0 means the synthetic data is about as useful as the real thing."},
  "sec-privacy":{t:"Safety — could it be traced to a real person?",
    b:"The data is attacked three ways: can someone tell who was in the real data, are any rows copied from it, and can a hidden detail be guessed? A PASS means the attack failed — which is what we want."},
};
// per-tab dimension score for the at-a-glance strip, and how to judge it
const TAB_DIM={
  "sec-quality": {get:(res,s)=>num((((res.summary||{})[s]||{}).fidelity||{}).score), good:0.8, ok:0.6},
  "sec-shapes":  {get:(res,s)=>num((((res.summary||{})[s]||{}).fidelity||{}).column_shapes), good:0.8, ok:0.6},
  "sec-pairs":   {get:(res,s)=>num((((res.summary||{})[s]||{}).fidelity||{}).column_pair_trends), good:0.8, ok:0.6},
  "sec-ri":      {get:(res,s)=>num((((res.summary||{})[s]||{}).fidelity||{}).structure), good:0.8, ok:0.6},
  "sec-utility": {get:(res,s)=>num((((res.summary||{})[s]||{}).utility||{}).score), good:0.85, ok:0.7},
  "sec-privacy": {safety:true},
};
function tabGlance(res,id){
  const cfg=TAB_DIM[id]; if(!cfg) return "";
  const pal=n=>(res.palette&&res.palette[n])||PALETTE[n]||"#888";
  const dt=n=>`<span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:${pal(n)};margin-right:7px"></span>`;
  const rows=(res.synths||[]).map(s=>{
    let score=null, verdict;
    if(cfg.safety){ verdict=safetyVerdict(res,s); }
    else { score=cfg.get(res,s); verdict=scoreVerdict(score,cfg.good,cfg.ok); }
    const val = score!=null ? Math.round(score*100)+"%" : "n/a";
    return `<div class="glance-row"><span class="glance-name">${dt(s)}${esc(s)}</span>
      <span class="glance-r"><span class="bm-pct">${val}</span>${verdict?verdictBadge(verdict):`<span class="verdict-na">n/a</span>`}</span></div>`;
  }).join("");
  return `<div class="biz-glance">${rows}</div>`;
}
function applyBizIntros(res){
  for(const id in BIZ_INTRO){
    const sec=document.getElementById(id); if(!sec || sec.querySelector(".biz-intro")) continue;
    const e=BIZ_INTRO[id];
    // move the technical content (score cards, formulas, tables, charts) into a
    // collapsed "Show the numbers" expander, leaving only the plain intro visible
    const details=document.createElement("details"); details.className="biz-expander";
    details.innerHTML=`<summary><span class="bx-open">Show the numbers</span>`
      + `<span class="bx-close">Hide the numbers</span></summary>`;
    const wrap=document.createElement("div"); wrap.className="biz-details";
    while(sec.firstChild) wrap.appendChild(sec.firstChild);   // listeners move with the nodes
    details.appendChild(wrap);
    const intro=document.createElement("div"); intro.className="biz-intro";
    intro.innerHTML=`<h4>${esc(e.t)}</h4><p>${esc(e.b)}</p>`;
    sec.appendChild(intro);
    sec.insertAdjacentHTML("beforeend", tabGlance(res,id));   // at-a-glance verdict per generator
    sec.appendChild(details);
  }
}
