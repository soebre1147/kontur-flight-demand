import {DEFAULTS,MODEL_DEFAULTS,PROFILES,PERIODS,toCsv,RESULT_COLUMNS} from './model.mjs';
import {createResearchController} from './research.mjs';

const $=id=>document.getElementById(id),fmt=new Intl.NumberFormat('ru-RU'),num=n=>fmt.format(n),pct=n=>n.toLocaleString('ru-RU',{maximumFractionDigits:1})+'%',money=n=>num(Math.round(n))+' ₽';
const esc=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const dateFmt=new Intl.DateTimeFormat('ru-RU',{day:'numeric',month:'long',year:'numeric',timeZone:'UTC'});
let result=null,mode='simulation',files=null,offersFiles=null,imported=new Map(),selectedFlight='',periodView='demand',page=0,baseline=null,toastTimer,renderTimer;
let initialized=false,latestRequestId=null,requestRevision=0,uiRevision=0,serverHasFiles=false,lastHeight=0,activeModel={...MODEL_DEFAULTS},calibrationArtifact=null;
try{const saved=JSON.parse(localStorage.getItem('kontur-baseline-python-v3'));if(saved?.summary&&saved?.params&&saved?.curve&&typeof saved.name==='string')baseline=saved;}catch{}
const labels={demand:'мест',price:'₽',budget:'₽',growth:'%',cancel:'%',businessShare:'%'};
function readParams(){
  const p={...DEFAULTS};for(const key of Object.keys(p)){const el=$(key);if(el)p[key]=typeof DEFAULTS[key]==='number'?Number(el.value):el.value;}
  p.flight=p.flight.trim()||'SU1122';return p;
}
function syncControls(){
  for(const [id,unit] of Object.entries(labels)){const input=$(id),out=$(id+'-output');out.innerHTML=num(Number(input.value))+' <small>'+unit+'</small>';}
  document.querySelectorAll('input[type="range"]').forEach(el=>el.style.setProperty('--progress',((el.value-el.min)/(el.max-el.min)*100)+'%'));
  const weights=PROFILES[$('profile').value],max=Math.max(...weights);$('profile-spark').innerHTML=weights.map((w,i)=>'<span style="height:'+Math.max(8,w/max*100)+'%" title="'+PERIODS[i]+': '+num(w/weights.reduce((a,b)=>a+b,0)*100)+'%"></span>').join('');
}
function notify(message){clearTimeout(toastTimer);$('toast').textContent=message;$('toast').hidden=false;toastTimer=setTimeout(()=>$('toast').hidden=true,3500);}
function error(message){$('error').textContent=message;$('error').hidden=!message;}
function readModel(){return {method:$('model-method').value,total_mode:$('total-mode').value,sigma:Number($('model-sigma').value),prior_strength:Number($('prior-strength').value),beta_time:Number($('beta-time').value),beta_qty:Number($('beta-qty').value),mixture:$('model-mixture').checked,...(calibrationArtifact?{calibration:calibrationArtifact}:{})};}
function setBusy(busy){
  document.body.classList.toggle('calculation-pending',busy);
  $('backend-status').textContent=busy?'Python · расчёт…':'Python · Streamlit';
  $('backend-tag').setAttribute('aria-busy',String(busy));
}
function post(type,extra={}){window.parent.postMessage({isStreamlitMessage:true,type,...extra},'*');}
function recalculate(){
  clearTimeout(renderTimer);syncControls();
  if(!initialized){error('Ожидание подключения Python · Streamlit. Откройте приложение через Streamlit.');return;}
  if(mode==='import'&&!files?.length&&!serverHasFiles){setBusy(false);error('Выберите журнал bookings. На экране сохранён последний рассчитанный сценарий.');return;}
  latestRequestId=crypto.randomUUID();requestRevision=uiRevision;setBusy(true);error('');
  post('streamlit:setComponentValue',{dataType:'json',value:{action:'calculate',request_id:latestRequestId,mode,params:readParams(),files,offers_files:offersFiles,reference_files:researchController.getReferenceFiles(),selected_flight:selectedFlight||$('import-flight').value||null,model:readModel()}});
}
function scheduleCalculation(){uiRevision++;syncControls();clearTimeout(renderTimer);setBusy(true);renderTimer=setTimeout(recalculate,320);}
function updateModeControls(){
  const isImport=mode==='import';$('import-box').hidden=!isImport;
  $('mode-simulation').classList.toggle('active',!isImport);$('mode-import').classList.toggle('active',isImport);
  $('mode-simulation').setAttribute('aria-pressed',String(!isImport));$('mode-import').setAttribute('aria-pressed',String(isImport));
  document.querySelectorAll('[data-sim-only]').forEach(el=>el.disabled=isImport);
  document.querySelectorAll('[data-import-only]').forEach(el=>el.hidden=!isImport);
  $('flight').closest('label').hidden=isImport;$('import-flight-field').hidden=!isImport;
}
function restoreControls(response,chosen){
  const params=response.params||chosen.params||{};
  for(const [key,value] of Object.entries(params)){const el=$(key);if(el&&value!=null)el.value=value;}
  const config={...MODEL_DEFAULTS,...(response.model||chosen.model||chosen.diagnostics?.config||{})};
  $('model-method').value=config.method;$('total-mode').value=config.total_mode;
  $('model-sigma').value=config.sigma;$('prior-strength').value=config.prior_strength;
  $('beta-time').value=config.beta_time||0;$('beta-qty').value=config.beta_qty||0;$('model-mixture').checked=!!config.mixture;
  calibrationArtifact=config.calibration||null;$('calibration-applied').hidden=!calibrationArtifact;
  mode=response.mode||chosen.mode||'simulation';updateModeControls();syncControls();
}
function acceptResponse(response,isInitial=false){
  if(!response)return;
  if(!isInitial&&latestRequestId&&response.request_id!==latestRequestId)return;
  if(!isInitial&&latestRequestId&&requestRevision<uiRevision)return;
  if(response.ok===false){setBusy(false);error((response.error||'Не удалось выполнить расчёт.')+' Последний корректный результат сохранён.');return;}
  const entries=response.results instanceof Array?response.results.map(v=>[v.params.flight,v]):Object.entries(response.results||{});
  if(!entries.length){setBusy(false);error('Python не вернул рассчитанный сценарий.');return;}
  const next=new Map(entries);const preferred=selectedFlight||response.selected_flight;
  const flight=next.has(preferred)?preferred:response.selected_flight&&next.has(response.selected_flight)?response.selected_flight:next.keys().next().value;
  const chosen=next.get(flight);
  if(isInitial)restoreControls(response,chosen);
  imported=next;selectedFlight=flight;result=chosen;activeModel=response.model||chosen.model||{...MODEL_DEFAULTS};
  $('import-flight').innerHTML=entries.map(([f])=>'<option value="'+esc(f)+'">'+esc(f)+'</option>').join('');$('import-flight').value=flight;
  serverHasFiles=response.mode==='import'||chosen.mode==='import'||!!files?.length;
  if(response.file_summary){$('file-summary').textContent=typeof response.file_summary==='string'?response.file_summary:JSON.stringify(response.file_summary);$('file-summary').hidden=false;}
  setBusy(false);error('');page=0;render();researchController.updateContext();
}
function resizeFrame(){
  requestAnimationFrame(()=>{const height=Math.ceil(document.body.getBoundingClientRect().height);if(height>0&&height!==lastHeight){lastHeight=height;post('streamlit:setFrameHeight',{height});}});
}
function render(){
  const {params:p,summary:s}=result,isSim=result.mode==='simulation';
  $('scenario-title').textContent=isSim?'Синтетический сценарий':'Журнал bookings';
  $('flight-label').textContent=p.flight;$('date-label').textContent=dateFmt.format(new Date(p.departure+'T00:00:00Z')).replace(' г.','');
  $('scenario-badge').textContent=isSim?'SEED '+p.seed:'ИМПОРТ CSV';$('capacity-label').textContent=num(p.capacity)+' мест';
  $('journal-subtitle').textContent=isSim?'Синтетические обращения, покупки и отмены':'Наблюдаемые операции из загруженного bookings';
  const warnings=[];
  if(s.maxBalance>p.capacity)warnings.push('Баланс в журнале превышает заданную вместимость. Проверьте вместимость рейса.');
  if(s.minBalance<0)warnings.push('Есть отрицательный баланс мест. Возможно, журнал неполный или отмены не соответствуют покупкам.');
  $('warning').textContent=warnings.join(' ');$('warning').hidden=!warnings.length;
  const revenue=s.revenue>=1000000?(s.revenue/1000000).toLocaleString('ru-RU',{maximumFractionDigits:2}):num(Math.round(s.revenue/1000));
  $('metrics').innerHTML=[
    metric('Места на рейсе',num(s.net),'/ '+num(p.capacity),'Загрузка <b>'+pct(s.load)+'</b>','↗',Math.min(100,s.load)),
    metric('Куплено до отмен',num(s.gross),'мест','Отменено <b>'+num(s.canceled)+' мест</b>','↗'),
    metric(isSim?'Без покупки · факт':'Без покупки · оценка',num(s.waste),'мест',pct(p.demand?s.waste/p.demand*100:0)+' первичного спроса','↗'),
    metric('Чистая выручка',revenue,s.revenue>=1000000?'млн ₽':'тыс. ₽','Покупки за вычетом возвратов','₽')
  ].join('');
  $('curve-insight').textContent='Конверсия спроса в покупку — '+pct(s.conversion)+'. '+(isSim?'Синтетический факт.':'Объём спроса задан извне; скрытые обращения — оценка.');
  renderComparison();renderCurve();renderPeriods();renderWaste();renderFares();renderJournal();renderDiagnostics();resizeFrame();
}
function metric(title,value,unit,foot,icon,progress){return '<article class="metric"><div class="metric-title">'+title+'<span class="metric-icon">'+icon+'</span></div><div class="metric-value">'+value+'<small>'+unit+'</small></div><div class="metric-foot">'+foot+'</div>'+(progress!==undefined?'<div class="metric-line"><span style="width:'+progress+'%"></span></div>':'')+'</article>';}
function renderCurve(){
  const w=900,h=250,pad={l:43,r:23,t:14,b:35},points=result.curve,maxDtd=Math.max(1,points[0]?.dtd||0);
  if(!points.length){$('curve-chart').innerHTML='<div class="empty-state">В журнале нет операций.</div>';return;}
  const yMax=Math.max(result.params.capacity,result.summary.gross,baseline?.summary?.net||0,1)*1.13;
  const niceMax=Math.ceil(yMax/25)*25,x=dtd=>pad.l+(maxDtd-dtd)/maxDtd*(w-pad.l-pad.r),y=v=>h-pad.b-v/niceMax*(h-pad.t-pad.b);
  const path=key=>points.map((v,i)=>(i?'L':'M')+x(v.dtd).toFixed(2)+','+y(v[key]).toFixed(2)).join(' ');
  let grid='';for(let i=0;i<=4;i++){const v=Math.round(niceMax*i/4),yy=y(v);grid+='<line x1="'+pad.l+'" y1="'+yy+'" x2="'+(w-pad.r)+'" y2="'+yy+'" stroke="#edf1ef"/><text x="'+(pad.l-12)+'" y="'+(yy+4)+'" text-anchor="end" class="chart-text">'+num(v)+'</text>';}
  const ticks=maxDtd===180?[180,150,120,90,60,30,0]:[maxDtd,Math.round(maxDtd*.75),Math.round(maxDtd*.5),Math.round(maxDtd*.25),0];
  grid+=ticks.map(d=>'<text x="'+x(d)+'" y="'+(h-11)+'" text-anchor="middle" class="chart-text">'+num(d)+'</text>').join('');
  const last=points.at(-1),area=path('net')+' L'+x(0)+','+y(0)+' L'+x(maxDtd)+','+y(0)+' Z';
  let baseLine='';if(baseline){const base=baseline.curve.filter(p=>p.dtd<=maxDtd);baseLine='<path d="'+base.map((v,i)=>(i?'L':'M')+x(v.dtd)+','+y(v.net)).join(' ')+'" fill="none" stroke="#b5a08c" stroke-width="1.5" stroke-dasharray="3 5"><title>Базовый сценарий</title></path>';}
  $('curve-chart').innerHTML='<svg viewBox="0 0 '+w+' '+h+'" role="img" aria-label="Накопленный баланс: '+result.summary.net+' мест, покупки: '+result.summary.gross+' мест"><defs><linearGradient id="area-fill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#83b9a0" stop-opacity=".21"/><stop offset="100%" stop-color="#83b9a0" stop-opacity=".015"/></linearGradient></defs>'+grid+'<path d="'+area+'" fill="url(#area-fill)"/><line x1="'+pad.l+'" x2="'+(w-pad.r)+'" y1="'+y(result.params.capacity)+'" y2="'+y(result.params.capacity)+'" stroke="#bac5c1" stroke-dasharray="5 5"/>'+baseLine+'<path d="'+path('gross')+'" stroke="#a3c6b6" stroke-width="2" stroke-dasharray="4 3" fill="none"/><path d="'+path('net')+'" stroke="#0b7467" stroke-width="2.6" fill="none" stroke-linejoin="round"/><circle cx="'+x(0)+'" cy="'+y(last.net)+'" r="4" fill="#0b7467" stroke="white" stroke-width="2"/><line id="hover-line" x1="0" x2="0" y1="'+pad.t+'" y2="'+(h-pad.b)+'" stroke="#7fa18e" stroke-dasharray="3 3" visibility="hidden"/><rect id="curve-hit" x="'+pad.l+'" y="'+pad.t+'" width="'+(w-pad.l-pad.r)+'" height="'+(h-pad.t-pad.b)+'" fill="transparent"/></svg><div class="chart-tooltip" id="chart-tooltip" hidden></div>';
  const container=$('curve-chart'),svg=container.querySelector('svg'),hit=$('curve-hit'),tip=$('chart-tooltip'),line=$('hover-line');
  hit.addEventListener('pointermove',e=>{const rect=svg.getBoundingClientRect(),vx=(e.clientX-rect.left)/rect.width*w,dtd=Math.max(0,Math.min(maxDtd,Math.round(maxDtd-(vx-pad.l)/(w-pad.l-pad.r)*maxDtd))),v=points.find(p=>p.dtd===dtd)||points.reduce((a,b)=>Math.abs(b.dtd-dtd)<Math.abs(a.dtd-dtd)?b:a);line.setAttribute('x1',x(dtd));line.setAttribute('x2',x(dtd));line.setAttribute('visibility','visible');tip.hidden=false;tip.innerHTML='<small>'+esc(v.date)+' · '+dtd+' дн. до вылета</small><b>Баланс: '+num(v.net)+' мест</b><br>Покупки: '+num(v.gross)+' · отмены: '+num(v.canceled);tip.style.left=Math.max(8,Math.min(container.clientWidth-180,e.clientX-container.getBoundingClientRect().left+12))+'px';tip.style.top='10px';});
  hit.addEventListener('pointerleave',()=>{tip.hidden=true;line.setAttribute('visibility','hidden');});
}
function renderPeriods(){
  const estimate=periodView==='estimate',isSim=result.mode==='simulation',data=result.periods,w=480,h=208,pad={l:29,r:8,t:20,b:35};
  const series=estimate?[{key:'estimated',color:'#deaa7f',name:result.diagnostics?.method==='v3'?'Оценка v3':'Базовая оценка'},...(isSim?[{key:'actual',color:'#0b7467',name:'Синтетический факт'}]:[])]:[{key:'purchased',color:'#458b72',name:'Покупки'},{key:isSim?'actual':'estimated',color:'#d7e7dd',name:isSim?'Без покупки · факт':'Без покупки · оценка'},{key:'canceled',color:'#dfaa7e',name:'Отмены'}];
  const max=Math.max(...data.flatMap(v=>series.map(s=>v[s.key])),...(estimate?data.map(v=>Number.isFinite(v.upper)?v.upper:0):[]),1),maxY=Math.ceil(max/10)*10,y=v=>h-pad.b-v/maxY*(h-pad.t-pad.b),groupW=(w-pad.l-pad.r)/5,barW=series.length===3?14:20;
  let markup='';for(let i=0;i<=3;i++){const val=Math.round(maxY*i/3);markup+='<line x1="'+pad.l+'" x2="'+(w-pad.r)+'" y1="'+y(val)+'" y2="'+y(val)+'" stroke="#eff2f0"/><text x="'+(pad.l-7)+'" y="'+(y(val)+3)+'" class="chart-text" text-anchor="end" style="font-size:9px">'+val+'</text>';}
  data.forEach((v,i)=>{const center=pad.l+groupW*(i+.5);series.forEach((s,j)=>{const x=center-(series.length*barW+(series.length-1)*4)/2+j*(barW+4),value=v[s.key];markup+='<rect x="'+x+'" y="'+y(value)+'" width="'+barW+'" height="'+Math.max(0,h-pad.b-y(value))+'" rx="2" fill="'+s.color+'"><title>'+esc(v.period)+' дней · '+s.name+': '+value+' мест</title></rect>';});markup+='<text x="'+center+'" y="'+(h-14)+'" text-anchor="middle" class="chart-text" style="font-size:10px">'+v.period+'</text>';});
  $('period-chart').innerHTML='<svg viewBox="0 0 '+w+' '+h+'" role="img" aria-label="'+(estimate?'Оценка скрытого спроса по периодам':'Покупки и отказы по периодам')+'">'+markup+'</svg>';
  $('period-legend').innerHTML=series.map(s=>'<span><i class="legend-square" style="background:'+s.color+'"></i>'+s.name+'</span>').join('');
  $('period-subtitle').textContent=estimate?(isSim?'Оценка модели и синтетический факт':(result.diagnostics?.method==='v3'?'Скрытый спрос · оценка модели v3':'Скрытый спрос · пропорциональная оценка')):'Покупки, отмены и обращения без покупки, мест';
  if(estimate&&data.some(v=>Number.isFinite(v.lower)&&Number.isFinite(v.upper)))$('period-subtitle').textContent+=' · интервалы 10–90% доступны в экспорте';
}
function renderWaste(){
  const s=result.summary,isSim=result.mode==='simulation',totals=result.diagnostics?.modelled_reason_totals;
  $('waste-subtitle').textContent=isSim?'Причины отказа · синтетический факт':totals?'Причины отказа · оценка модели':'Причины отказа в bookings неизвестны';
  let reasons=[];
  if(isSim)reasons=[{n:'Цена выше бюджета',v:s.priceWaste||0,c:'#c99a70'},{n:'Недостаточно мест',v:s.capacityWaste||0,c:'#d8e6dc'}];
  else if(totals)reasons=[{n:'Бюджет · модель',v:totals.budget||0,c:'#c99a70'},{n:'Доступность · модель',v:totals.availability||0,c:'#d8e6dc'},...(totals.other?[{n:'Прочее · модель',v:totals.other,c:'#b9c8c0'}]:[])];
  let details;
  if(reasons.length){const total=reasons.reduce((a,v)=>a+v.v,0);details='<div class="stacked-bar">'+reasons.map(v=>'<span style="width:'+(total?v.v/total*100:0)+'%;background:'+v.c+'"></span>').join('')+'</div>'+reasons.map(v=>'<div class="waste-detail"><span><i class="legend-square" style="background:'+v.c+'"></i>'+v.n+'</span><b>'+num(v.v)+' <span style="color:#a8b2ad">/ '+pct(total?v.v/total*100:0)+'</span></b></div>').join('')+(!isSim?'<p class="field-help">Объяснение модели; причины скрытых обращений не наблюдались в bookings.</p>':'');}
  else details='<div class="stacked-bar"><span style="width:100%;background:#d8e6dc"></span></div><div class="waste-detail"><span>Оценка из заданного спроса</span><b>'+num(s.waste)+' мест</b></div><p class="field-help">Время, бюджет и причины скрытых обращений не наблюдались в bookings.</p>';
  $('waste-chart').innerHTML='<div class="waste-summary"><div class="waste-number">'+pct(result.params.demand?s.waste/result.params.demand*100:0)+'<small>спроса без покупки</small></div>'+details+'</div>';
}
function renderFares(){
  const cabin=$('cabin-filter').value,groups=new Map();
  for(const v of result.fares){
    if(cabin!=='all'&&v.cabin!==cabin)continue;
    const key=v.cabin+'|'+v.rbd;
    if(!groups.has(key))groups.set(key,{cabin:v.cabin,rbd:v.rbd,purchased:0,canceled:0,grossRevenue:0,min:v.price,max:v.price});
    const group=groups.get(key);group.purchased+=v.purchased;group.canceled+=v.canceled;group.grossRevenue+=v.price*v.purchased;group.min=Math.min(group.min,v.price);group.max=Math.max(group.max,v.price);
  }
  const fares=[...groups.values()].map(v=>({...v,price:v.purchased?v.grossRevenue/v.purchased:v.min})).sort((a,b)=>a.cabin.localeCompare(b.cabin)||a.price-b.price),max=Math.max(...fares.map(v=>Math.max(v.purchased,v.canceled)),1);
  $('fares-chart').innerHTML=fares.length?'<div class="fare-bars">'+fares.map(v=>'<div class="fare-column" title="'+esc(v.cabin+' / '+v.rbd)+' · '+money(v.price)+'; куплено '+v.purchased+', отменено '+v.canceled+'"><span class="fare-count">'+v.purchased+'</span><div class="fare-bar-track"><div class="fare-bar" style="height:'+v.purchased/max*100+'%"></div><div class="fare-bar cancel" style="height:'+v.canceled/max*100+'%"></div></div><div class="fare-code"><span>'+esc(v.cabin)+'</span>'+esc(v.rbd)+'</div><div class="fare-price">'+num(v.price)+' ₽</div></div>').join('')+'</div>':'<div class="empty-state">В этом салоне нет покупок и отмен.</div>';
}
function journalRows(){const rows=$('journal-view').value==='reconstructed'?result.reconstructed:result.events,outcome=$('outcome-filter').value;return rows.filter(v=>outcome==='all'||v.outcome===outcome);}
function renderJournal(){
  if(!result)return;const rows=journalRows(),size=10,pages=Math.max(1,Math.ceil(rows.length/size));page=Math.min(page,pages-1);
  const names={success:'Покупка',cancel:'Отмена',waste:'Без покупки'};
  const isReconstructed=$('journal-view').value==='reconstructed';$('journal-subtitle').textContent=isReconstructed?'Покупки и отмены из bookings; бюджеты и скрытые строки — оценка модели':result.mode==='simulation'?'Синтетические обращения, покупки и отмены':'Наблюдаемые операции из загруженного bookings';
  $('journal-body').innerHTML=rows.slice(page*size,page*size+size).map(e=>{
    const time=e.timestamp?e.timestamp.replace('T',' ').slice(0,16):'Период '+e.period;
    const modelled=e.reason_is_modelled||e.row_source==='model_generated_hidden_group'||String(e.reconstruction_status||'').startsWith('оценка:');
    const amountTruth=e.amount_source==='synthetic_truth',amountLabel=amountTruth?'Синтетический факт · бюджет на место':'Оценка бюджета на место · '+(e.amount_source||'модель');
    const source=modelled?'Оценка модели'+(e.reason?' · '+e.reason:' · скрытые обращения'):e.reason||((result.mode==='simulation'?'Синтетический факт':e.source_file||'bookings')+(e.source_row?' · строка '+e.source_row:''));
    return '<tr><td>'+esc(time)+'</td><td><span class="outcome-pill '+e.outcome+'"><i></i>'+names[e.outcome]+'</span></td><td>'+esc(e.cabin&&e.rbd?e.cabin+' / '+e.rbd:'—')+'</td><td class="numeric">'+(e.outcome==='cancel'?'−':'')+num(e.qty)+'</td><td class="numeric">'+(Number.isFinite(e.price)?money(e.price):'—')+'</td><td class="numeric budget-cell" title="'+esc(amountLabel)+'">'+(Number.isFinite(e.amount)?(amountTruth?'':'≈ ')+money(e.amount):'—')+'</td><td class="table-source" title="'+esc(source)+'">'+esc(source)+'</td></tr>';
  }).join('')||'<tr><td colspan="7" class="empty-state">Нет событий с выбранным результатом.</td></tr>';
  $('table-count').textContent=rows.length?'Показано '+num(page*size+1)+'–'+num(Math.min(rows.length,(page+1)*size))+' из '+num(rows.length)+' событий':'0 событий';
  $('page-number').textContent=(page+1)+' / '+pages;$('prev-page').disabled=page===0;$('next-page').disabled=page>=pages-1;
}
function renderComparison(){
  $('comparison').hidden=!baseline;if(!baseline)return;
  $('baseline-description').textContent=baseline.name+' · '+num(baseline.params.demand)+' мест спроса · '+money(baseline.params.price)+' стартовый тариф';
  const items=[['Баланс мест','net',false],['Куплено до отмен','gross',false],['Без покупки','waste',true],['Чистая выручка','revenue',false]];
  $('comparison-content').innerHTML=items.map(([title,key,inverse])=>{const delta=result.summary[key]-baseline.summary[key],formatted=key==='revenue'?money(Math.abs(delta)):num(Math.abs(delta));return '<div class="compare-cell">'+title+'<strong class="'+((inverse?delta>0:delta<0)?'negative':'')+'">'+(delta>0?'+':delta<0?'−':'')+formatted+'</strong><small>База: '+(key==='revenue'?money(baseline.summary[key]):num(baseline.summary[key]))+'</small></div>';}).join('');
}
function setMode(next){mode=next;updateModeControls();uiRevision++;recalculate();}
async function readUploads(list){
  const selected=[...list];if(!selected.length)return null;
  if(selected.reduce((a,f)=>a+f.size,0)>10*1024*1024)throw new Error('Общий размер CSV не должен превышать 10 МБ.');
  return await Promise.all(selected.map(async f=>({name:f.name,text:await f.text()})));
}
async function loadFiles(list){
  try{const next=await readUploads(list);if(!next)return;files=next;selectedFlight='';serverHasFiles=true;
    $('file-summary').textContent=files.map(f=>f.name).join(' · ')+' · отправлено в Python';$('file-summary').hidden=false;
    mode='import';updateModeControls();uiRevision++;recalculate();
  }catch(e){setBusy(false);error(e.message);}
}
async function loadOffers(list){
  try{const next=await readUploads(list);if(!next)return;offersFiles=next;
    $('offers-summary').textContent=next.map(f=>f.name).join(' · ');$('offers-summary').hidden=false;uiRevision++;recalculate();
  }catch(e){setBusy(false);error(e.message);}
}
function renderDiagnostics(){
  const d=result.diagnostics||{},isSim=result.mode==='simulation';
  $('model-title').textContent=d.title||(d.method==='baseline'?'Базовая модель':'Математическая модель v3');
  const status={converged:'Расчёт завершён',prior_only_no_sales:'Только заданные предположения',baseline:'Базовое распределение'};
  $('model-state').textContent=status[d.status]||d.status||'Python · Streamlit';
  const values=[];
  if(isSim&&Number.isFinite(d.WAPE_pct??d.wape_pct))values.push('WAPE по скрытому спросу: '+pct(d.WAPE_pct??d.wape_pct));
  if(isSim&&Number.isFinite(d.MAE_seats??d.mae_seats))values.push('MAE: '+num(d.MAE_seats??d.mae_seats)+' мест');
  $('model-quality').textContent=values.join(' · ');$('model-quality').hidden=!values.length;
  $('model-explanation').textContent=isSim?'Покупки, отмены и отказы сгенерированы в Python. Качество восстановления проверяется по скрытому синтетическому факту.':'Покупки и отмены взяты из bookings. Бюджеты, скрытые обращения и причины отказов оценены моделью; точность без passengers не измеряется.';
  const warnings=Array.isArray(d.warnings)?d.warnings:[];
  $('model-warnings').textContent=warnings.join(' ');$('model-details').hidden=!warnings.length;
  $('period-export').disabled=!result.periods?.length;$('offers-export').disabled=!result.offers?.some(row=>row.offer_source!=='assumed_last_purchase_prices');
}
function download(kind){
  if(!result)return;const p=result.params;let body,filename,type;
  if(kind==='params'){body=JSON.stringify({model:'kontur-python-v3',params:p,model_config:result.model||activeModel,mode:result.mode,diagnostics:result.diagnostics},null,2);filename='scenario-'+p.flight+'.json';type='application/json';}
  else{let rows,columns;
    if(kind==='bookings'){rows=result.observed.map(e=>({timestamp:e.timestamp,cabin:e.cabin,rbd:e.rbd,commercial:e.signed_qty,price:e.price,flight:e.flight}));columns=['timestamp','cabin','rbd','commercial','price','flight'];}
    else if(kind==='daily'){rows=result.curve;columns=['date','dtd','gross','net','canceled','purchases_delta','cancellations_delta','net_delta'];}
    else if(kind==='periods'){rows=result.periods;columns=['period','purchased','canceled','estimated','actual','lower','upper'];}
    else if(kind==='offers'){rows=(result.offers||[]).filter(row=>row.offer_source!=='assumed_last_purchase_prices');columns=[...new Set(rows.flatMap(row=>Object.keys(row)))];}
    else{rows=kind==='reconstructed'?result.reconstructed:result.events;columns=[...RESULT_COLUMNS,...(kind==='events'?['cancelRef']:[])];}
    body=toCsv(rows,columns);filename=({reconstructed:'passengers-reconstructed',bookings:'bookings',events:'events',daily:'daily-bookings',periods:'demand-periods',offers:'offers'}[kind]||kind)+'-'+p.flight+'-'+p.departure+'.csv';type='text/csv;charset=utf-8';
  }
  const url=URL.createObjectURL(new Blob([body],{type})),a=document.createElement('a');
  a.href=url;a.download=filename;a.style.display='none';document.body.appendChild(a);
  try{a.click();}finally{a.remove();setTimeout(()=>URL.revokeObjectURL(url),30000);}
  $('export-menu').hidden=true;notify('Файл подготовлен: '+filename);
}
const researchController=createResearchController({
  getContext:()=>({mode,params:readParams(),files,offers_files:offersFiles,selected_flight:selectedFlight||$('import-flight').value||null,model:readModel()}),post,notify,resize:resizeFrame,
  onCalculationLoaded:response=>{latestRequestId=null;files=null;offersFiles=null;acceptResponse(response,true);serverHasFiles=mode==='import';},
  onContextLoaded:response=>{restoreControls(response,{params:response.params||{},model:response.model||{}});files=null;offersFiles=null;serverHasFiles=mode==='import';selectedFlight=response.selected_flight||response.params?.flight||'';if(selectedFlight){$('import-flight').innerHTML='<option value="'+esc(selectedFlight)+'">'+esc(selectedFlight)+'</option>';$('import-flight').value=selectedFlight;}researchController.updateContext();},
  onModelSelected:config=>{for(const [key,id] of Object.entries({sigma:'model-sigma',prior_strength:'prior-strength',beta_time:'beta-time',beta_qty:'beta-qty',method:'model-method',total_mode:'total-mode'}))if(config[key]!=null)$(id).value=config[key];if(config.mixture!=null)$('model-mixture').checked=!!config.mixture;calibrationArtifact=config.calibration||null;$('calibration-applied').hidden=!calibrationArtifact;uiRevision++;recalculate();}
});
$('clear-calibration').addEventListener('click',()=>{calibrationArtifact=null;$('calibration-applied').hidden=true;uiRevision++;recalculate();});
$('controls').addEventListener('submit',e=>e.preventDefault());
$('controls').addEventListener('input',e=>{if(e.target.id==='capacity'&&Number($('businessCapacity').value)>Number(e.target.value)&&Number(e.target.value)>0)$('businessCapacity').value=e.target.value;if(e.target.type==='file')return;scheduleCalculation();});
$('mode-simulation').addEventListener('click',()=>setMode('simulation'));$('mode-import').addEventListener('click',()=>setMode('import'));
$('import-flight').addEventListener('change',()=>{selectedFlight=$('import-flight').value;result=imported.get(selectedFlight);if(result){page=0;render();}});
$('reset').addEventListener('click',()=>{clearTimeout(renderTimer);for(const [key,value] of Object.entries(DEFAULTS)){const el=$(key);if(el)el.value=value;}for(const [id,value] of Object.entries({'model-method':MODEL_DEFAULTS.method,'total-mode':MODEL_DEFAULTS.total_mode,'model-sigma':MODEL_DEFAULTS.sigma,'prior-strength':MODEL_DEFAULTS.prior_strength,'beta-time':0,'beta-qty':0}))$(id).value=value;$('model-mixture').checked=false;calibrationArtifact=null;$('calibration-applied').hidden=true;setMode('simulation');notify('Параметры сброшены к базовому сценарию.');});
$('new-seed').addEventListener('click',()=>{$('seed').value=crypto.getRandomValues(new Uint32Array(1))[0];uiRevision++;recalculate();});
document.querySelectorAll('[data-period-view]').forEach(button=>button.addEventListener('click',()=>{if(!result)return;periodView=button.dataset.periodView;document.querySelectorAll('[data-period-view]').forEach(b=>{const active=b===button;b.classList.toggle('active',active);b.setAttribute('aria-pressed',String(active));});renderPeriods();}));
$('cabin-filter').addEventListener('change',()=>{if(result)renderFares();});for(const id of ['journal-view','outcome-filter'])$(id).addEventListener('change',()=>{page=0;renderJournal();});
$('prev-page').addEventListener('click',()=>{page--;renderJournal();});$('next-page').addEventListener('click',()=>{page++;renderJournal();});
$('save-baseline').addEventListener('click',()=>{if(!result)return;const p=result.params;baseline={params:p,summary:result.summary,curve:result.curve,name:p.flight+' · '+(result.mode==='simulation'?'seed '+p.seed:'импорт CSV')};try{localStorage.setItem('kontur-baseline-python-v3',JSON.stringify(baseline));}catch{}renderComparison();renderCurve();notify('Текущий сценарий сохранён как база. Меняйте параметры для сравнения.');});
$('remove-baseline').addEventListener('click',()=>{baseline=null;try{localStorage.removeItem('kontur-baseline-python-v3');}catch{}if(result){renderComparison();renderCurve();}});
for(const id of ['model-button','method-footer'])$(id).addEventListener('click',event=>{const dialog=$('model-dialog');dialog.style.top=Math.max(24,event.currentTarget.getBoundingClientRect().top-70)+'px';dialog.style.margin='0 auto';dialog.showModal();});$('close-model').addEventListener('click',()=>$('model-dialog').close());$('model-dialog').addEventListener('click',e=>{if(e.target===$('model-dialog')){const r=e.target.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)e.target.close();}});
$('export-button').addEventListener('click',()=>$('export-menu').hidden=!$('export-menu').hidden);document.addEventListener('click',e=>{if(!e.target.closest('.export-wrap'))$('export-menu').hidden=true;});document.addEventListener('keydown',e=>{if(e.key==='Escape')$('export-menu').hidden=true;});document.querySelectorAll('[data-export]').forEach(b=>b.addEventListener('click',()=>download(b.dataset.export)));
$('file-input').addEventListener('change',e=>loadFiles(e.target.files));const drop=$('dropzone');for(const name of ['dragenter','dragover'])drop.addEventListener(name,e=>{e.preventDefault();drop.classList.add('dragging');});for(const name of ['dragleave','drop'])drop.addEventListener(name,e=>{e.preventDefault();drop.classList.remove('dragging');});drop.addEventListener('drop',e=>loadFiles(e.dataTransfer.files));
$('offers-input').addEventListener('change',e=>loadOffers(e.target.files));
$('clear-offers').addEventListener('click',()=>{offersFiles=[];$('offers-input').value='';$('offers-summary').hidden=true;uiRevision++;recalculate();});
window.addEventListener('message',event=>{
  if(event.source!==window.parent||event.data?.type!=='streamlit:render')return;
  const args=event.data.args||{};
  if(!initialized){initialized=true;const first=args.response?.results?args.response:args.initial_response;acceptResponse(first,true);researchController.connect();if(args.response&&!researchController.handleResponse(args.response)&&args.response?.ok===false)acceptResponse(args.response);}
  else if(args.response&&!researchController.handleResponse(args.response))acceptResponse(args.response);
  resizeFrame();
});
new ResizeObserver(resizeFrame).observe(document.body);
syncControls();updateModeControls();setBusy(true);
post('streamlit:componentReady',{apiVersion:1});resizeFrame();
setTimeout(()=>{if(!initialized){setBusy(false);error('Нет подключения к Python. Запустите приложение командой streamlit run app.py; прямое открытие HTML не выполняет расчёт.');}},15000);
