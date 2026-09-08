"""Exercise the real frontend (not a copied visibility predicate) across its lifecycle."""
import shutil
import subprocess
from pathlib import Path

import pytest


def test_completed_review_lifecycle_with_real_frontend(tmp_path):
    node = shutil.which("node")
    assert node, "Node.js is required for this frontend gate"
    source = (Path(__file__).resolve().parents[1] / "web/project_id.js").read_text(encoding="utf-8")
    source = source.replace('import { app } from "../../scripts/app.js";',
                            'const app = globalThis.__app;').replace(
        'import { normalizeReferenceAudioLabels } from "./reference_audio_ui.js";',
        'function normalizeReferenceAudioLabels() {}')
    harness = r'''
const assert = require("node:assert/strict");
let project = null;
let fetchMode = "ok";
globalThis.fetch = async () => {
  if (fetchMode === "error") throw new Error("offline fixture");
  return project ? {ok:true, status:200, json:async()=>structuredClone(project)} : {ok:false,status:404};
};
globalThis.__app = {
  graph: {_nodes: [], links:{}, getNodeById:()=>null},
  ui:{settings:{getSettingValue:(_id,fallback)=>fallback,addSetting(){}}},
  registerExtension(extension){this.extension=extension;}
};
'''
    cases = r'''
function makeNode() {
  const values = {
    prompt_mode:"Auto",chunks:2,chunk_seconds:5,aspect:"Auto from First Image",
    preset:"Draft — 0.30 MP",custom_mp:0.3,continuity:"Balanced — 22 frames",
    base_seed:123,control_after_generate:"fixed",audio_continuity:true,
    continuation_backend:"Standard",run_storage:"Save + Auto Resume",
    reroll_from_chunk:"Auto",reroll_nonce:0,run_name:"fixture",project_id:"fixture",
    reference_size:"Match Output",video_reference_size:"Efficient - 0.4 MP",
    diagnostics:"Detailed Report",strict_compatibility:false,debug:false,show_preview:true,
    generation_mode:"Review Each Chunk",review_action:"Continue / Next",
    take_group:0,take_revision_id:"",take_action:"Automatic",
    size_source:"First Image",width:544,height:544
  };
  const n={comfyClass:"H3ContinuumSamplerV38",properties:{},inputs:[],
    widgets:Object.entries(values).map(([name,value])=>({name,value,type:"combo",options:{values:[]},computeSize:()=>[120,20]})),
    addWidget(type,name,value,callback,options){const w={type,name,value,callback,options:options||{},computeSize:()=>[120,20]};this.widgets.push(w);return w;},
    addCustomWidget(w){this.widgets.push(w);return w;},
    serialize(){return {widgets_values:this.widgets.map(w=>w.value)};},
    configure(info){info.widgets_values.forEach((v,i)=>{this.widgets[i].value=v;});},
    setDirtyCanvas(){}
  };
  app.extension.nodeCreated(n);
  return n;
}
const w=(n,name)=>n.widgets.find(w=>w.name===name);
const visible=(n,name)=>!w(n,name).hidden;
const buttons=["Use it and continue","Try this chunk again","Use it and finish the rest"];
function state(status,unit,id="revision-1") {
  return {branch_provenance_version:1,canonical_storage_revision_id:id,
    revisions:[{revision_id:id,status,...(unit?{review_unit:unit}:{})}],
    canonical_head_revision_id:"take-1",active_revisions:{"1":"take-1"},
    group_revisions:[{revision_id:"take-1",revision_order:"1",take_number:1,group:{start:1,end:1,physical_group:1}}]};
}
async function load(n,p) {project=p;await loadTakeHistory(n);n.__h3ContinuumIntuitiveUxRefresh?.();n.__h3ContinuumProductionUxRefresh();}
(async()=>{
  const n=makeNode();await load(n,null);
  assert(buttons.every(name=>!visible(n,name)),"initial is not review-ready");
  for(const status of ["running","interrupted","unknown"]){
    await load(n,state(status,{start:1,end:1,physical_group:1}));
    assert(buttons.every(name=>!visible(n,name)),status);
  }
  await load(n,state("review_ready",{start:1,end:1,physical_group:1}));
  assert(buttons.every(name=>visible(n,name)),"partial review has all actions");
  for(const action of ["Use it and continue","Use it and finish the rest"]){
    w(n,action).callback();const inputs={};prepareReviewQueueIntent(n,inputs);
    assert.equal(inputs.review_action,action==="Use it and continue"?"Continue / Next":"Finish Remaining");
    w(n,"review_action").afterQueued({isPartialExecution:false});
    assert.equal(w(n,"review_action").value,"Continue / Next");
  }
  for(const count of [1,2,3,4,5,6]){
    w(n,"chunks").value=count;
    prepareReviewQueueIntent(n,{}); // This completion belongs to the current settings.
    const p=state("complete",{start:count,end:count,physical_group:count},`done-${count}`);
    await load(n,p);
    assert(visible(n,"Try this chunk again"),`retry complete ${count}`);
    assert(!visible(n,"Use it and continue")&&!visible(n,"Use it and finish the rest"),"no no-op continuation buttons");
    assert(visible(n,"Render History"));
    assert.match(w(n,"Review Ready").value,/Saved sequence is complete/);
    const saved=n.serialize();
    w(n,"Back to Settings").callback();
    assert(visible(n,"Chunks")&&visible(n,"Return to Review"));
    assert(!visible(n,"Try this chunk again"));
    w(n,"Return to Review").callback();
    assert(visible(n,"Try this chunk again")&&!visible(n,"Chunks"));
    assert.deepEqual(n.serialize(),saved,"settings/review are presentation only");
    w(n,"Render History").callback();assert(visible(n,"Render History / Takes"));
    const drawn=[];
    const ctx={save(){},restore(){},beginPath(){},rect(){},clip(){},fillText(text){drawn.push(text);}};
    w(n,"Render History / Takes").draw(ctx,n,400,0,300);
    assert(drawn.some(text=>text.startsWith("Selected:")),"selected Take is actually drawn, not an empty text widget");
    w(n,"Render History").callback();assert(!visible(n,"Render History / Takes"));
    w(n,"Try this chunk again").callback();
    const inputs={};prepareReviewQueueIntent(n,inputs);
    assert.equal(inputs.review_action,"Regenerate Current");
    assert.equal(w(n,"base_seed").value,123);
    assert.equal(w(n,"control_after_generate").value,"fixed");
    assert.match(w(n,"Review Ready").value,new RegExp(`Chunk ${count}`));
    w(n,"review_action").afterQueued({isPartialExecution:false});
    assert.equal(inputs.review_action,"Regenerate Current","queued payload remains immutable");
    assert.equal(w(n,"review_action").value,"Continue / Next","one-shot consumed once");
    assert.deepEqual(project,p,"backend evidence not rewritten");
    await load(n,state("complete",{start:count,end:count,physical_group:count},`retry-${count}`));
    assert(visible(n,"Try this chunk again"),"retry completion still accessible");
  }
  await load(n,state("complete",{start:2,end:3,physical_group:2},"terminal"));
  w(n,"Try this chunk again").callback();
  assert.match(w(n,"Review Ready").value,/Chunks 2-3/);
  await load(n,state("complete",null,"finish-without-unit"));
  assert(!visible(n,"Try this chunk again"),"do not invent a retry unit for full-run completion");
  assert(visible(n,"Render History")&&visible(n,"Back to Settings"));
  await load(n,state("review_ready",{physical_group:1}));
  assert(!visible(n,"Try this chunk again"),"incomplete metadata not actionable");
  await load(n,state("complete",{start:1,end:1,physical_group:1}));
  fetchMode="error";await loadTakeHistory(n);
  assert(!visible(n,"Try this chunk again"),"failed fetch cannot retain stale retry controls");
  fetchMode="ok";await loadTakeHistory(n);assert(visible(n,"Try this chunk again"));
  const reloaded=makeNode();await load(reloaded,project);
  assert(visible(reloaded,"Try this chunk again"),"reload restores completed review from backend");
  w(reloaded,"generation_mode").value="Full Run";reloaded.__h3ContinuumProductionUxRefresh();
  assert(buttons.every(name=>!visible(reloaded,name)),"full run does not expose review actions");
  w(reloaded,"generation_mode").value="Review Each Chunk";
  w(reloaded,"run_storage").value="Off";reloaded.__h3ContinuumProductionUxRefresh();
  assert(buttons.every(name=>!visible(reloaded,name)),"storage off does not expose review actions");
  // Hotfix: stale saved review cannot offer/submit Retry after settings edits.
  const edited=makeNode();w(edited,"chunks").value=4;
  const originalProject=state("review_ready",{start:2,end:2,physical_group:2},"edit-base");
  await load(edited,originalProject);
  const oldSeed=w(edited,"base_seed").value;
  w(edited,"Try this chunk again").callback();
  w(edited,"Back to Settings").callback();
  w(edited,"base_seed").value=999;w(edited,"base_seed").callback?.(999);
  assert.match(reviewStatus(edited),/Settings changed/);
  w(edited,"Return to Review").callback();
  assert(buttons.every(name=>!visible(edited,name)),"stale review actions hidden");
  assert(visible(edited,"Start again from Chunk 1")&&visible(edited,"Render History"));
  await load(edited,originalProject);
  assert.match(reviewStatus(edited),/Settings changed/,"history refresh must not bless edits");
  const changedInputs={base_seed:999};prepareReviewQueueIntent(edited,changedInputs);
  assert.equal(changedInputs.review_action,"Continue / Next");
  assert.equal(changedInputs.base_seed,999);
  assert.deepEqual(project,originalProject,"stored backend state is untouched");
  w(edited,"base_seed").value=oldSeed;edited.__h3ContinuumProductionUxRefresh();
  assert(visible(edited,"Try this chunk again"),"undo restores the original review eligibility");
  // Manual boundaries are not silently erased by old action callbacks.
  w(edited,"reroll_from_chunk").value="Chunk 1";
  w(edited,"Try this chunk again").callback();
  assert.equal(w(edited,"reroll_from_chunk").value,"Chunk 1");
  const manualInputs={reroll_from_chunk:"Chunk 1"};prepareReviewQueueIntent(edited,manualInputs);
  assert.equal(manualInputs.reroll_from_chunk,"Chunk 1");
  assert.equal(manualInputs.review_action,"Continue / Next");
  // Restart uses existing non-destructive branch controls, consumed once.
  w(edited,"reroll_nonce").value=8;
  w(edited,"Start again from Chunk 1").callback();
  assert.equal(w(edited,"reroll_nonce").value,0);
  assert.equal(w(edited,"take_action").value,"Automatic");
  const restartInputs={reroll_from_chunk:w(edited,"reroll_from_chunk").value};
  prepareReviewQueueIntent(edited,restartInputs);
  w(edited,"review_action").afterQueued({isPartialExecution:false});
  assert.equal(restartInputs.reroll_from_chunk,"Chunk 1","queued restart payload unchanged");
  assert.equal(w(edited,"reroll_from_chunk").value,"Auto","next Queue can continue");
  assert.equal(w(edited,"base_seed").value,oldSeed);
  assert.equal(edited.serialize().widgets_values.length,makeNode().serialize().widgets_values.length);
  // Linked prompts, media selection/bypass and reconnection are included.
  const upstream={id:71,comfyClass:"LoadImage",mode:0,inputs:[],widgets:[{name:"image",value:"first.png"}]};
  const graph={links:{9:{origin_id:71,origin_slot:0}},getNodeById:id=>id===71?upstream:null};
  const linked=makeNode();linked.graph=graph;linked.inputs=[{name:"reference_image_1",link:9}];
  await load(linked,state("review_ready",{start:1,end:1,physical_group:1},"linked"));
  for(const mutation of [()=>upstream.widgets[0].value="second.png",()=>upstream.mode=4,()=>linked.inputs[0].link=null]){
    mutation();linked.onDrawForeground();
    assert.match(reviewStatus(linked),/Settings changed/);
    assert(!visible(linked,"Try this chunk again"));
    upstream.widgets[0].value="first.png";upstream.mode=0;linked.inputs[0].link=9;
    linked.onDrawForeground();assert(visible(linked,"Try this chunk again"));
  }
  upstream.comfyClass="PrimitiveStringMultiline";upstream.widgets=[{name:"value",value:"old prompt"}];
  await load(linked,state("review_ready",{start:1,end:1,physical_group:1},"prompt-base"));
  upstream.widgets[0].value="new prompt";linked.onDrawForeground();
  assert.match(reviewStatus(linked),/Settings changed/);
  // Edits made while a queued run is working must remain dirty on completion.
  const flight=makeNode();prepareReviewQueueIntent(flight,{});
  w(flight,"base_seed").value=321;
  await load(flight,state("review_ready",{start:1,end:1,physical_group:1},"flight"));
  assert.match(reviewStatus(flight),/Settings changed/);
  assert(!visible(flight,"Try this chunk again"));
  // A slow first history response cannot bless settings changed during fetch.
  const slow=makeNode();await load(slow,null);
  const normalFetch=globalThis.fetch;let releaseFetch;
  globalThis.fetch=()=>new Promise(resolve=>{releaseFetch=resolve;});
  const pendingHistory=loadTakeHistory(slow);
  w(slow,"base_seed").value=456;
  releaseFetch({ok:true,status:200,json:async()=>state("review_ready",{start:1,end:1,physical_group:1},"slow")});
  await pendingHistory;
  assert.match(reviewStatus(slow),/Settings changed/);
  globalThis.fetch=normalFetch;
  // An old queued snapshot cannot become another Run Name's baseline.
  const otherRun=makeNode();prepareReviewQueueIntent(otherRun,{});
  w(otherRun,"run_name").value="different-run";w(otherRun,"base_seed").value=789;
  await load(otherRun,state("review_ready",{start:1,end:1,physical_group:1},"other-run"));
  assert(!reviewSettingsChanged(otherRun));
  console.log("PASS: real frontend completed-review lifecycle");
})().catch(e=>{console.error(e);process.exitCode=1;});
'''
    script = tmp_path / "completed_review.cjs"
    script.write_text(harness + source + cases, encoding="utf-8")
    result = subprocess.run([node, str(script)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS: real frontend" in result.stdout
