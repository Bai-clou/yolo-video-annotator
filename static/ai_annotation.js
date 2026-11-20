// Per-image dialogue and draft state. API keys are never put in browser storage.
const aiState = {config:null, mode:'manual', busy:false, busyImage:null,
  messages:Object.create(null), pending:Object.create(null), undo:Object.create(null),
  revisions:Object.create(null), status:Object.create(null)};

async function aiAPI(path, body) {
  const response = await fetch('/api/ai/' + path, body === undefined ? {} : {
    method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
  let data;
  try { data=await response.json(); } catch { throw new Error('AI 服务未返回有效 JSON，请检查服务日志和接口配置。'); }
  if (!response.ok) throw new Error(data.error || `请求失败（${response.status}）`);
  return data;
}

function setAnnotationMode(mode) {
  aiState.mode=mode;
  for (const item of ['manual','ai']) {
    const active=item===mode;
    document.getElementById('mode-'+item).classList.toggle('active',active);
    document.getElementById('mode-'+item).setAttribute('aria-selected',String(active));
  }
  document.getElementById('manual-tools').hidden=mode!=='manual';
  document.getElementById('ai-tools').hidden=mode!=='ai';
  updateAIPanel();
}

async function loadAIConfiguration() {
  try { aiState.config=await aiAPI('config'); updateAIPanel(); }
  catch(error) { document.getElementById('ai-config-summary').textContent=error.message; }
}

function updateAIPanel() {
  const name=currentName(), config=aiState.config;
  const summary=document.getElementById('ai-config-summary');
  summary.textContent=config?.configured ? `${config.model} · ${config.has_key?'Key 已设置':'未设置 Key'}` : '尚未配置视觉模型';
  const chat=document.getElementById('ai-chat');
  chat.replaceChildren();
  const messages=aiState.messages[name] || [];
  if (!messages.length) {
    const hint=document.createElement('div'); hint.className='hint';
    hint.textContent=name ? '对话仅针对当前图片。可要求补框、删除、修改类别或询问标注问题。' : '请先导入并打开一张图片或视频帧。';
    chat.append(hint);
  }
  for (const message of messages) {
    const element=document.createElement('div'); element.className='ai-message '+message.role;
    element.textContent=(message.role==='user'?'你：':message.role==='assistant'?'AI：':'提示：')+message.content;
    chat.append(element);
  }
  chat.scrollTop=chat.scrollHeight;
  document.getElementById('ai-send').disabled=aiState.busy || !name || !state.imgObj || state.loadingImage;
  document.getElementById('ai-send').textContent=aiState.busy?'AI 分析中…':'发送并标注';
  document.getElementById('ai-apply').hidden=!aiState.pending[name];
  document.getElementById('ai-undo').disabled=!aiState.undo[name];
  document.getElementById('ai-status').textContent=aiState.status[name] ||
    (aiState.busy && aiState.busyImage!==name ? '正在为另一张图片生成标注；结果只会返回对应图片。' : '');
}

function onAnnotationChanged(name) {
  aiState.revisions[name]=(aiState.revisions[name] || 0)+1;
  updateAIPanel();
}

function addAIMessage(name, role, content) {
  const messages=aiState.messages[name] ||= [];
  messages.push({role,content});
  if (messages.length>60) messages.splice(0,messages.length-60);
  if (currentName()===name) updateAIPanel();
}

async function openAIConfig() {
  const dialog=document.getElementById('ai-config-dialog');
  try {
    aiState.config=await aiAPI('config');
    const config=aiState.config;
    for (const [field,id] of Object.entries({base_url:'base-url',model:'model',protocol:'protocol',auth:'auth',
      max_tokens:'max-tokens',timeout:'timeout',image_edge:'image-edge',token_parameter:'token-parameter'})) {
      document.getElementById('ai-'+id).value=config[field];
    }
    const key=document.getElementById('ai-key'); key.value='';
    key.placeholder=config.has_key ? '本次运行已设置 Key；留空沿用，修改地址时请重新填写' : '填写自己的 API Key；无密钥本地服务可留空';
    document.getElementById('ai-clear-key').checked=false;
    document.getElementById('ai-config-status').textContent='';
    document.getElementById('ai-config-status').classList.remove('error');
    dialog.showModal();
  } catch(error) { notice(error.message); }
}

function aiConfigDraft() {
  const value=id=>document.getElementById('ai-'+id).value.trim();
  return {base_url:value('base-url'),model:value('model'),protocol:value('protocol'),auth:value('auth'),
    api_key:value('key'),clear_key:document.getElementById('ai-clear-key').checked,
    max_tokens:Number(value('max-tokens')),timeout:Number(value('timeout')),image_edge:Number(value('image-edge')),
    token_parameter:value('token-parameter')};
}

function aiConfigBusy(busy) {
  for (const element of document.querySelectorAll('#ai-config-dialog input, #ai-config-dialog select, #ai-save-config, #ai-test-config')) element.disabled=busy;
}

async function saveAIConfig() {
  const status=document.getElementById('ai-config-status');
  status.classList.remove('error'); aiConfigBusy(true);
  try {
    aiState.config=await aiAPI('config',aiConfigDraft());
    document.getElementById('ai-key').value='';
    document.getElementById('ai-clear-key').checked=false;
    status.textContent='配置已保存。API Key 仅在本次服务运行期间有效，可点击测试连接。';
    updateAIPanel();
  } catch(error) { status.textContent=error.message; status.classList.add('error'); }
  finally { aiConfigBusy(false); }
}

async function testAIConfig() {
  const status=document.getElementById('ai-config-status'), draft=aiConfigDraft();
  status.classList.remove('error'); aiConfigBusy(true);
  status.textContent='正在发送生成的测试图片，检查 API 地址、模型和图片请求…';
  try {
    const result=await aiAPI('test',draft);
    aiState.config=await aiAPI('config',draft);
    document.getElementById('ai-key').value='';
    document.getElementById('ai-clear-key').checked=false;
    status.textContent=`连接成功（${result.elapsed_seconds} 秒），配置已保存。\n${result.message}\n模型回复：${result.reply}`;
    updateAIPanel();
  } catch(error) { status.textContent='连接测试失败：'+error.message; status.classList.add('error'); }
  finally { aiConfigBusy(false); }
}

document.getElementById('ai-config-dialog').addEventListener('close',()=>{
  document.getElementById('ai-key').value='';
});

async function sendAIMessage() {
  const name=currentName(), prompt=document.getElementById('ai-prompt');
  if (aiState.busy || !name || !state.imgObj || state.loadingImage) return;
  if (!aiState.config?.configured) { notice('请先配置支持图片输入的视觉模型 API'); await openAIConfig(); return; }
  const instruction=prompt.value.trim();
  if (!instruction) { prompt.focus(); return; }
  const revision=aiState.revisions[name] || 0;
  const history=(aiState.messages[name] || []).filter(message=>message.role==='user'||message.role==='assistant').slice(-12);
  const current=structuredClone(state.annotations[name] || []);
  aiState.busy=true; aiState.busyImage=name;
  delete aiState.pending[name];
  aiState.status[name]='正在分析当前图片；可继续手工操作，期间修改的草稿不会被自动覆盖。';
  addAIMessage(name,'user',instruction); prompt.value=''; updateAIPanel();
  try {
    const result=await aiAPI('annotate',{filename:name,instruction,history,current_boxes:current});
    if (result.filename!==name) throw new Error('AI 结果对应的图片不一致，未应用标注。');
    addAIMessage(name,'assistant',result.message || (result.action==='none'?'无需修改标注。':'已生成标注。'));
    if (result.action==='replace') {
      aiState.pending[name]=result;
      if (currentName()===name && (aiState.revisions[name] || 0)===revision && !state.loadingImage) applyPendingAI();
      else {
        aiState.status[name]='结果已生成。图片已切换或草稿已修改；点击“应用本次结果”会用 AI 结果更新本图全部框。';
        notice('AI 结果已返回对应图片，待应用后人工核对。');
      }
    } else aiState.status[name]=`AI 已回复，本图标注未更改（${result.elapsed_seconds} 秒）。`;
  } catch(error) {
    aiState.status[name]='请求未完成，已有标注保持不变。';
    addAIMessage(name,'error',error.message);
  } finally { aiState.busy=false; aiState.busyImage=null; updateAIPanel(); }
}

function applyPendingAI() {
  const name=currentName(), result=aiState.pending[name];
  if (!result || state.loadingImage) return;
  aiState.undo[name]=structuredClone(state.annotations[name] || []);
  state.annotations[name]=structuredClone(result.boxes);
  delete aiState.pending[name];
  state.selectedBoxIds.clear(); state.canvasSelectedBoxId=null;
  markDirty(name);
  aiState.status[name]=`已应用 ${result.boxes.length} 个框（${result.elapsed_seconds} 秒）。请检查位置和类别，必要时修正后保存。`;
  render(); updateTable(); updateAIPanel();
}

function undoAIResult() {
  const name=currentName();
  if (!Object.hasOwn(aiState.undo,name)) return;
  state.annotations[name]=aiState.undo[name]; delete aiState.undo[name];
  state.selectedBoxIds.clear(); state.canvasSelectedBoxId=null;
  markDirty(name);
  aiState.status[name]='已恢复到最近一次应用 AI 结果前的草稿；请按需保存。';
  render(); updateTable(); updateAIPanel();
}

function clearAIChat() {
  const name=currentName(); if (!name) return;
  if (aiState.busy && aiState.busyImage===name) { notice('请等待当前 AI 请求结束后再清空对话。'); return; }
  delete aiState.messages[name]; delete aiState.status[name]; delete aiState.pending[name];
  updateAIPanel();
}

function selectedCoordinateBox() {
  const id=state.canvasSelectedBoxId || (state.selectedBoxIds.size===1 ? [...state.selectedBoxIds][0] : null);
  return (state.annotations[currentName()] || []).find(box=>box.id===id);
}

function updateCoordinateEditor() {
  const box=selectedCoordinateBox(), panel=document.getElementById('coordinate-editor');
  panel.hidden=!box;
  if (box) for (const key of ['x_center','y_center','width','height']) document.getElementById('coord-'+key).value=Number(box[key].toFixed(6));
}

function changeBoxCoordinate(key, value) {
  const box=selectedCoordinateBox(); if (!box) return;
  const next={...box,[key]:Number(value)}, {x_center:x,y_center:y,width:w,height:h}=next;
  if (![x,y,w,h].every(Number.isFinite) || w<0.000001 || h<0.000001 || w>1 || h>1 ||
      x-w/2 < -1e-6 || x+w/2 > 1.000001 || y-h/2 < -1e-6 || y+h/2 > 1.000001) {
    notice('坐标应为 0–1 归一化值，且整个框应在图片内。'); updateCoordinateEditor(); return;
  }
  Object.assign(box,next,{source:'manual'}); markDirty(); render(); updateTable();
}

loadAIConfiguration();
