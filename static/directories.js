// Open existing local folders in place; individual file uploads still append copies.
function folderStatus(message,error=false) {
  const element=document.getElementById('folder-status');
  element.textContent=message; element.classList.toggle('error',error);
}

function openFolderDialog() {
  document.getElementById('folder-path').value=state.sourceDir || '';
  document.getElementById('folder-save-open').hidden=true;
  folderStatus('可点击“选择文件夹”，也可以粘贴目录路径。');
  document.getElementById('folder-dialog').showModal();
}

const folderBrowser={kind:'images',path:null,parent:null,sequence:0};

async function pickLocalFolder(kind) {
  folderBrowser.kind=kind;
  document.getElementById('folder-browser-dialog').showModal();
  await browseFolders(state.sourceDir || '');
}

async function browseFolders(path) {
  const sequence=++folderBrowser.sequence;
  const status=document.getElementById('folder-browser-status');
  const select=document.getElementById('folder-select');
  select.disabled=true; folderBrowser.path=null;
  status.textContent='正在读取文件夹…'; status.classList.remove('error');
  try {
    const response=await workspaceFetch('/api/folders?path='+encodeURIComponent(path || ''));
    const data=await response.json();
    if (sequence!==folderBrowser.sequence) return;
    if (!response.ok) throw new Error(data.error || '读取目录失败');
    folderBrowser.path=data.path; folderBrowser.parent=data.parent;
    document.getElementById('folder-location').value=data.path;
    document.getElementById('folder-parent').disabled=!data.parent;
    const drives=document.getElementById('folder-drives'), list=document.getElementById('folder-list');
    drives.replaceChildren(); list.replaceChildren();
    for (const path of data.drives) {
      const button=document.createElement('button'); button.className='action'; button.textContent=path;
      button.addEventListener('click',()=>browseFolders(path)); drives.append(button);
    }
    for (const folder of data.folders) {
      const button=document.createElement('button'); button.textContent='📁 '+folder.name;
      button.addEventListener('click',()=>browseFolders(folder.path)); list.append(button);
    }
    if (!data.folders.length) { const empty=document.createElement('div'); empty.className='hint'; empty.textContent='此目录没有子文件夹，仍可选择当前目录。'; list.append(empty); }
    status.textContent=`${data.folders.length} 个子目录；图片和视频会在选择后读取。`;
    select.disabled=false;
  } catch(error) {
    if (sequence===folderBrowser.sequence) { status.textContent=error.message; status.classList.add('error'); }
  }
}

function selectBrowsedFolder() {
  if (!folderBrowser.path) return;
  const path=folderBrowser.path;
  if (folderBrowser.kind==='videos') {
      const field=document.getElementById('video-paths');
      const paths=field.value.split('\n').map(value=>value.trim()).filter(Boolean);
      if (!paths.includes(path)) paths.push(path);
      field.value=paths.join('\n');
      videoStatus('已添加视频目录，包含子目录的视频；设置抽帧参数后点击“开始抽帧”。');
    } else {
      document.getElementById('folder-path').value=path;
      folderStatus('已选择文件夹，点击“打开目录”加载图片和已有标签。');
    }
  document.getElementById('folder-browser-dialog').close();
}

async function openImageDirectory(saveFirst=false) {
  if (videoJobId || aiState.busy || aiState.job || state.loadingImage) {
    folderStatus('请先等待或停止当前任务、等待图片加载完成，再切换目录。',true); return;
  }
  if (state.dirty.size && !saveFirst) {
    document.getElementById('folder-save-open').hidden=false;
    folderStatus(`有 ${state.dirty.size} 张图片的修改尚未保存。请点击“保存全部并打开”，或取消后继续编辑。`,true);
    return;
  }
  const buttons=['folder-open','folder-save-open','folder-browse'];
  for (const id of buttons) document.getElementById(id).disabled=true;
  try {
    if (saveFirst && state.dirty.size) {
      await saveAll();
      if (state.dirty.size) throw new Error('部分修改尚未保存，当前目录未切换。');
    }
    folderStatus('正在读取目录中的图片和已有标签…');
    const response=await workspaceFetch('/api/source',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({path:document.getElementById('folder-path').value,
                          recursive:document.getElementById('folder-recursive').checked})});
    const data=await response.json();
    if (!response.ok) throw new Error(data.error || '打开目录失败');
    window.location.reload();
  } catch(error) { folderStatus(error.message,true); }
  finally { for (const id of buttons) document.getElementById(id).disabled=false; }
}
