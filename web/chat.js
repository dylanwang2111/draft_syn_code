"use strict";
/* Chat assistant panel: a conversational layer over the exact same backend
   the manual controls use (apiFetch/SID/poll/renderReport/beginJobUI/
   activateTab/MODEL are all defined in app.js, loaded before this file).
   No separate app, no separate server: a run started from chat shows up in
   the same jobbar and the same report as a run started by clicking
   Synthesize, and a relationship drawn on the Data Model canvas is picked
   up automatically, no separate "I'm done" step required. */

/* ---------------- drawer: show/hide, and size (drag-resize, or two quick
   presets) -- both write the SAME "synthlab-chat-w" width, one source of
   truth, so dragging and clicking never disagree ---------------- */
function setChatCollapsed(collapsed){
  $("#chatdock").classList.toggle("collapsed",collapsed);
  $("#resize-chat").style.display=collapsed?"none":"";
}
$("#btn-chat").addEventListener("click",()=>setChatCollapsed(!$("#chatdock").classList.contains("collapsed")));
$("#chatdock-close").addEventListener("click",()=>setChatCollapsed(true));

(function(){
  const COMPACT=380, ROOMY=640;
  const dock=$("#chatdock"), btn=$("#chatdock-size");
  const isCompact=()=>dock.getBoundingClientRect().width<(COMPACT+ROOMY)/2;
  const applyIcon=()=>{
    const compact=isCompact();
    btn.innerHTML=compact
      ? `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7"/></svg>`
      : `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M9 3H3v6M15 21h6v-6M3 3l7 7M21 21l-7-7"/></svg>`;
    btn.title=compact?"Expand":"Shrink to a side panel";
  };
  btn.addEventListener("click",()=>{
    const target=isCompact()?ROOMY:COMPACT;
    dock.style.width=target+"px";
    try{ localStorage.setItem("synthlab-chat-w",target); }catch(e){}
    applyIcon();
  });
  // stay in sync with drag-resizing too (makeResizer, app.js), not just clicks
  if(window.ResizeObserver) new ResizeObserver(applyIcon).observe(dock);
  applyIcon();
})();

/* ---------------- transcript ---------------- */
const clog=$("#chat-log");
function chatScrollDown(){ clog.scrollTo({top:clog.scrollHeight,behavior:"smooth"}); }
function chatBubble(side,html){
  const row=document.createElement("div"); row.className="msg-row "+side;
  const av=document.createElement("div"); av.className="avatar "+side;
  av.textContent=side==="bot"?"S":"you";
  const b=document.createElement("div"); b.className="bubble "+side; b.innerHTML=html;
  if(side==="bot"){ row.appendChild(av); row.appendChild(b); } else { row.appendChild(b); row.appendChild(av); }
  clog.appendChild(row); chatScrollDown();
  return b;
}
const botMsg=html=>chatBubble("bot",html);
const chatUserMsg=html=>chatBubble("user",html);
function chatTyping(){
  const b=botMsg(`<span class="typing"><i></i><i></i><i></i></span>`);
  b.classList.add("typing-bubble");
  // every call site only ever does typingEl.remove() when the real reply
  // arrives -- return the whole avatar+bubble row, not just the bubble,
  // otherwise removing the bubble alone leaves its avatar orphaned in the DOM
  return b.parentElement;
}
// the LLM is told not to use markdown but does sometimes anyway; render the
// small safe subset instead of leaving literal ** and - in the bubble
function mdLite(s){
  let t=esc(s);
  t=t.replace(/\*\*(.+?)\*\*/g,"<b>$1</b>");
  t=t.replace(/\n[-•]\s+/g,"<br>• ");
  t=t.replace(/\n/g,"<br>");
  return t;
}

/* ---------------- upload -> plan (fires from app.js's init(), after the
   SAME /api/upload the toolbar's Upload button and Load Sample both use) --
   one upload path, chat just narrates whatever came back from it ---------------- */
async function chatOnDataLoaded(){
  clog.innerHTML="";
  chatUserMsg(`📎 ${Object.keys(DATA.tables).map(esc).join(", ")}`);
  const t=chatTyping();
  let r,j;
  try{
    r=await apiFetch("/api/chat/plan",{method:"POST"});
    j=await r.json();
  }catch(err){ t.remove(); botMsg(`Couldn't reach the assistant: ${esc(String(err))}`); return; }
  t.remove();
  if(!r.ok||j.error){ botMsg(`⚠ ${esc(j.error||"analysis failed")}`); return; }
  renderBotReply(j);
  $("#chatdock").classList.remove("collapsed");
  $("#chat-text-input").focus();
}

/* ---------------- render a bot turn: the message text, plus whatever the
   assistant's ask_question tool call attached (options -- rendered as the
   SAME .chat-btn buttons used everywhere else in the dock, clicking one is
   just shorthand for typing that answer) and/or open_schema_editor /
   open_data_model / open_config_panel / explain_synthesizer (focus -- pops
   the matching surface out, see enterFocusMode / openConfigPanel /
   openDocsModal in app.js) -- one render path for every route (plan /
   message / narrate_result), so a question asked at any point in the
   conversation looks the same ---------------- */
const CONFIG_PANEL_IDS={
  structure:["structure-panel"], constraints:["constraint-panel"], pii:["pii-panel"],
  synthesizers:["synth-panel","params-panel"], run_parameters:["params-panel"],
  synthetic_data:["synthetic-panel"],
};
function renderBotReply(j){
  const b=botMsg(mdLite(j.message));
  const opts=j.options||[];
  if(opts.length){
    b.appendChild(document.createElement("br"));
    const btns=opts.map(opt=>{
      const btn=document.createElement("button");
      btn.className="chat-btn"; btn.textContent=opt;
      btn.addEventListener("click",()=>{ btns.forEach(x=>x.disabled=true); chatSendText(opt); });
      return btn;
    });
    btns.forEach(btn=>{ b.appendChild(document.createTextNode(" ")); b.appendChild(btn); });
  }
  if(CONFIG_PANEL_IDS[j.focus]) openConfigPanel(j.focus);
  else if(j.focus && j.focus.startsWith("docs:")) openDocsModal(j.focus.slice(5));
  else if(j.focus) enterFocusMode(j.focus);
  return b;
}

/* ---------------- focus mode: the assistant asked to edit the Schema or
   Data Model tab directly rather than describe changes in words -- the tab
   visibly pops out (chat-focus dims the rest of the chrome, see style.css)
   with a floating Save & continue bar; saving just sends a short
   confirmatory line back through the normal chatSendText path (which
   already ships the live schema/relationships with every message), so the
   assistant sees what changed and calls the matching confirm_* tool ---------------- */
function enterFocusMode(view){
  activateTab(view==="data_model" ? "pane-model" : "pane-schema");
  document.body.classList.add("chat-focus");
  const bar=$("#chat-savebar");
  bar.dataset.view=view;
  $("#chat-savebar-label").textContent=view==="data_model"
    ? "Set up the table relationships, then save to continue."
    : "Edit the schema, then save to continue.";
  bar.classList.add("show");
}
function exitFocusMode(){
  document.body.classList.remove("chat-focus");
  $("#chat-savebar").classList.remove("show");
}
$("#chat-savebar-save").addEventListener("click",()=>{
  const view=$("#chat-savebar").dataset.view;
  exitFocusMode();
  chatSendText(view==="data_model" ? "I've set up the relationships." : "I've updated the schema.");
});
$("#chat-savebar-cancel").addEventListener("click",e=>{ e.preventDefault(); exitFocusMode(); });

/* ---------------- config panel modal: the assistant's general
   open_config_panel(panel) tool -- brings whichever left-sidebar config
   panel (Structure & keys / Constraints / PII handling / Synthesizers /
   Run parameters) is relevant into the middle of the screen, highlighted.
   The REAL panel element(s) get relocated into the modal (not cloned), so
   every control keeps working exactly as it does in the sidebar; closing
   moves them straight back to their original spot. ---------------- */
let CONFIG_MODAL_ORIGINS=[];
function openConfigPanel(view){
  const ids=CONFIG_PANEL_IDS[view]; if(!ids) return;
  closeConfigPanel();  // in case one was already open for a different panel
  const body=$("#config-modal-body");
  ids.forEach(id=>{
    const el=document.getElementById(id); if(!el) return;
    el.classList.remove("collapsed");
    CONFIG_MODAL_ORIGINS.push({el, parent:el.parentNode, next:el.nextSibling});
    body.appendChild(el);
  });
  $("#config-backdrop").classList.add("show");
}
function closeConfigPanel(){
  if(!CONFIG_MODAL_ORIGINS.length) return;
  CONFIG_MODAL_ORIGINS.forEach(({el,parent,next})=>parent.insertBefore(el,next));
  CONFIG_MODAL_ORIGINS=[];
  $("#config-backdrop").classList.remove("show");
}
// the panels' own header click-to-collapse is normally delegated from
// .left (app.js); once relocated into the modal they're no longer a
// descendant of .left, so the same delegation is added here too
$("#config-modal-body").addEventListener("click",e=>{
  const h=e.target.closest(".panel-h"); if(!h) return;
  h.parentElement.classList.toggle("collapsed");
});
$("#config-modal-done").addEventListener("click",()=>{
  closeConfigPanel();
  chatSendText("I've updated the configuration.");
});
$("#config-modal-close").addEventListener("click",()=>closeConfigPanel());
$("#config-backdrop").addEventListener("click",e=>{ if(e.target.id==="config-backdrop") closeConfigPanel(); });

document.addEventListener("keydown",e=>{
  if(e.key!=="Escape") return;
  if(document.body.classList.contains("chat-focus")) exitFocusMode();
  else if($("#config-backdrop").classList.contains("show")) closeConfigPanel();
});

/* ---------------- apply the backend's current plan state onto the live UI
   -- idempotent, only re-renders pieces that actually changed, using the
   SAME functions the manual controls already call (afterModelChange for
   the canvas/hub, renderRecipe+updateSummaries for the chip picker) ---------------- */
function applyPlanSync(sync){
  if(!sync) return;
  const rels=sync.relationships||[], hubKey=sync.entity_key||"", hubChildren=sync.entity_children||[];
  const relChanged=JSON.stringify(MODEL.rels)!==JSON.stringify(rels)
    || MODEL.hub.key!==hubKey || JSON.stringify(MODEL.hub.children)!==JSON.stringify(hubChildren);
  if(relChanged){
    MODEL.rels=rels.slice();
    MODEL.hub={key:hubKey, children:hubChildren.slice()};
    if(MODEL.hub.key) placeHub();
    buildHubBar();
    afterModelChange();
  }
  // mirrors the backend's own run_synthesis fallback precedence exactly
  // (selected_synths, if any, else the bot's recommended_synth) -- doesn't
  // touch the frontend's own default chip state otherwise, that's a
  // separate, bigger decision (does a recommendation override a click the
  // user hasn't actually made vs. the page's own hardcoded default) left
  // alone for now
  const synths=sync.selected_synths&&sync.selected_synths.length ? sync.selected_synths
    : (sync.recommended_synth ? [sync.recommended_synth] : []);
  if(synths.length){
    const cur=[...selectedSynths].sort().join(",");
    if(cur!==synths.slice().sort().join(",")){
      selectedSynths.clear(); synths.forEach(s=>selectedSynths.add(s));
      renderRecipe(); updateSummaries();
    }
  }
  if(sync.epochs && +$("#in-epochs").value!==sync.epochs) $("#in-epochs").value=sync.epochs;

  // schema edits described in words (set_column_types) land on the SAME
  // dropdowns the Schema tab already renders, same "changed" highlight the
  // manual edit path uses, so a typed change and a hand-picked one look
  // identical once applied
  for(const [t,edits] of Object.entries(sync.schema||{})){
    for(const [col,val] of Object.entries(edits.sdtypes||{})){
      const sel=$(`.sdtype-sel[data-table="${t}"][data-col="${col}"]`);
      if(sel && sel.value!==val){ sel.value=val; sel.classList.toggle("changed", val!==(detected[t]||{})[col]); }
    }
    if(edits.primary_key){
      const pk=$(`#pk-${t}`);
      if(pk && pk.value!==edits.primary_key) pk.value=edits.primary_key;
    }
  }
}

/* ---------------- composer: attach reuses the SAME #file-input the
   toolbar's Upload button uses (one upload control, not two) ---------------- */
$("#chat-attach-btn").addEventListener("click",e=>{ e.preventDefault(); $("#file-input").click(); });

$("#chat-composer-form").addEventListener("submit",e=>{
  e.preventDefault();
  const input=$("#chat-text-input"); const text=input.value.trim();
  if(!text) return;
  input.value="";
  chatSendText(text);
});
async function chatSendText(text){
  chatUserMsg(esc(text));
  chatSetBusy(true);
  const t=chatTyping();
  // always send the CURRENT live UI state -- a link drawn on the Data Model
  // canvas or chips clicked in the Synthesizers panel since the last
  // message get picked up right here, same mechanism either way
  const body={text,
    relationships:(typeof MODEL!=="undefined"&&MODEL.rels)||[],
    entity_key:(typeof MODEL!=="undefined"&&MODEL.hub&&MODEL.hub.key)||"",
    entity_children:(typeof MODEL!=="undefined"&&MODEL.hub&&MODEL.hub.children)||[],
    selected_synths:[...selectedSynths],
    epochs:+$("#in-epochs").value,
    schema:(typeof collectSchema==="function")?collectSchema():{}};
  let r,j;
  try{
    r=await apiFetch("/api/chat/message",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify(body)});
    j=await r.json();
  }catch(err){ t.remove(); botMsg(`Couldn't reach the assistant: ${esc(String(err))}`); chatSetBusy(false); return; }
  t.remove();
  if(!r.ok||j.error){ botMsg(`⚠ ${esc(j.error||"something went wrong")}`); chatSetBusy(false); return; }
  applyPlanSync(j.sync);
  renderBotReply(j);
  if(j.started){ beginJobUI(); poll(chatOnJobDone); return; }
  chatSetBusy(false);
}
function chatSetBusy(busy){
  $("#chat-text-input").disabled=busy; $("#chat-send-btn").disabled=busy;
  $("#chat-attach-btn").style.pointerEvents=busy?"none":"";
  if(!busy) $("#chat-text-input").focus();
}

/* ---------------- job finished: the jobbar + report already updated
   themselves (poll() in app.js does that); just ask the assistant to
   narrate the SAME results in one short message.

   The business-language framing (bizDims/recommendation/bizUseCases, in
   app.js) is exactly what's already printed on the exec summary card in
   the report next to this chat -- computed ONCE there, reused here rather
   than having the LLM re-paraphrase the raw scores into its own wording,
   so the two never drift out of sync. The LLM still gets the raw
   technical numbers too, for when the user asks to go technical. ---------------- */
function bizSummaryFor(res,s){
  const dims=bizDims(res,s);
  return {
    overall: worstVerdict(dims.map(d=>d.verdict)),
    business_summary: recommendation(res,s,dims).replace(/<\/?b>/g,""),
    use_cases: bizUseCases(dims).map(u=>({label:u.label, verdict:u.verdict, reason:u.reason})),
    technical: {fidelity:dims[0].score, safety:dims[1].score, usefulness:dims[2].score},
  };
}
async function chatOnJobDone(res,err){
  if(err){ botMsg(`⚠ ${esc(typeof err==="string"?err:"the run didn't finish cleanly")}`); chatSetBusy(false); return; }
  const t=chatTyping();
  const business={};
  (res.synths||[]).forEach(s=>{ business[s]=bizSummaryFor(res,s); });
  let r,j;
  try{
    r=await apiFetch("/api/chat/narrate_result",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({business})});
    j=await r.json();
  }catch(e){ t.remove(); botMsg(`Couldn't reach the assistant: ${esc(String(e))}`); chatSetBusy(false); return; }
  t.remove();
  if(!r.ok||j.error){ botMsg(`⚠ ${esc(j.error||"couldn't summarize the run")}`); chatSetBusy(false); return; }
  renderBotReply(j);
  chatSetBusy(false);
}

botMsg(`Hi! I'm the Synth/Lab assistant. Upload or load sample data to get started, I'll take it from
  there, no need to pick a model or tune anything yourself unless you want to.`);
