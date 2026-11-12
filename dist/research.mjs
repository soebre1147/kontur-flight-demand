// Research controls and rendering only. Inference, scoring, jobs and storage run in Python.
import {toCsv} from './model.mjs';

const SCENARIOS=[['balanced','Сбалансированный'],['early','Ранний спрос'],['late','Поздний спрос'],['burst','Всплески обращений'],['mixed_wtp','Смешанные бюджеты'],['closure','Закрытие продаж'],['groups_budget','Размер группы и бюджет'],['high_cancel','Частые отмены'],['low_demand','Низкий спрос'],['sold_out','Дефицит мест']];
const TOOLS={
  evaluate:['Проверка по passengers','Сравнение распределений после восстановления. Эталон недоступен самой модели.','Проверить по passengers'],
  experiments:['Пакетные эксперименты','v3 и базовый метод получают одинаковые журналы. В отчёте сохраняются и неудачные оценки.','Запустить сравнение'],
  calibration:['Калибровка предположений','Выбор параметров на обучающих запусках и проверка на независимых seed.','Запустить калибровку'],
  sensitivity:['Устойчивость результата','Повторная оценка при изменении бюджетов, регуляризации и внешнего N.','Проверить устойчивость'],
  forecast:['Прогноз до вылета','Оценка по информации, доступной на выбранный день; будущее используется только для проверки.','Построить прогноз'],
  policies:['Сравнение тарифных политик','Один поток пассажиров на каждую политику, отдельные запуски для выбора и проверки.','Сравнить политики']
};
const ACTIONS=new Set([...Object.keys(TOOLS),'generate_example','poll_job','cancel_job','history','load_run','replay_run']);
const STATUS={queued:'В очереди',running:'Выполняется',pending:'В очереди',completed:'Завершено',complete:'Завершено',done:'Завершено',success:'Завершено',failed:'Ошибка',error:'Ошибка',cancelled:'Остановлено',canceled:'Остановлено',cancelling:'Остановка…'};
const RUNNING=new Set(['queued','running','pending','cancelling']);
const LABELS={runs:'Запусков',scenarios:'Сценариев',total_runs:'Всего запусков',successful_runs:'Успешных запусков',failed_runs:'Ошибок оценки',failures:'Ошибок оценки',v3_wins:'Побед v3',mean_wape:'Средний WAPE',mean_wape_pct:'Средний WAPE, %',wape_pct:'WAPE, %',mae_seats:'MAE, мест',hidden_mae:'MAE скрытого спроса',coverage:'Покрытие интервалов',coverage_pct:'Покрытие, %',gross:'Куплено до отмен',hidden:'Скрытых мест',net:'Баланс мест',revenue:'Выручка, ₽',profit:'Прибыль, ₽',load:'Загрузка, %',cutoff_dtd:'Дней до вылета',flight:'Рейс',seed:'Seed',method:'Метод',scenario:'Сценарий',sigma:'Разброс бюджетов σ',prior_strength:'Вес предположений',demand:'Внешний N',mean_revenue:'Средняя выручка, ₽',mean_load:'Средняя загрузка, %',training_runs:'Обучающих запусков',validation_runs:'Проверочных запусков'};
Object.assign(LABELS,{paired_runs:'Парных сравнений',independent_seed_clusters:'Независимых seed',mean_mae_difference:'Разница MAE: v3 − базовый, мест',difference_ci_low:'Нижняя граница 95%',difference_ci_high:'Верхняя граница 95%',v3_win_pct:'Побед v3, %',tie_pct:'Одинаковая ошибка, %',scenario_count:'Сценариев',runs_per_scenario:'Запусков на сценарий',generated_realizations:'Сценарных реализаций',fits_attempted:'Оценок двух методов',successful_fits:'Успешных оценок',failed_fits:'Ошибок расчёта',seed_start:'Начальный seed',scenario_label:'Сценарий',successes:'Успешных оценок',budget_estimated_seats:'Мест с оценкой бюджета',budget_reference_seats:'Мест с эталоном бюджета',hidden_period_mae:'MAE по периодам, мест',hidden_period_wape:'WAPE по периодам, %',hidden_cabin_mae:'MAE по салонам, мест',interval_coverage_80:'Покрытие интервала 80%, %',budget_wasserstein:'Расхождение бюджетов, ₽',reference_gross:'Покупок в эталоне, мест',reference_hidden_seats:'Скрытых мест в эталоне',reference_primary_seats:'Первичных мест в эталоне',known_hidden_cabin_seats:'Скрытых мест с известным салоном',by_scenario:'Сравнение по сценариям',overall:'Общие показатели',paired_runs_table:'Парные сравнения',all_runs:'Все расчёты',period_mae:'Ошибка скрытого спроса по сценариям',v3_mae:'MAE v3, мест',baseline_mae:'MAE базового метода, мест',mae_difference:'Разница MAE, мест',tie:'Одинаковая ошибка',run:'Номер запуска',ok:'Успешно',elapsed_seconds:'Время, с',flights:'Рейсов',periods:'Спрос по периодам',cabins:'Скрытый спрос по салонам',budgets:'Квантили бюджетов',quantile:'Квантиль',actual:'Эталон',estimated:'Оценка',error:'Ошибка',absolute_error:'Абсолютная ошибка',lower:'Нижняя граница 10%',upper:'Верхняя граница 90%',covered_80:'Эталон внутри интервала',period:'Период',cabin:'Салон',budget_quantiles:'Сравнение бюджетов',selected_sigma:'Выбранный σ',selected_mixture:'Смесь бюджетов',selected_beta_time:'Временной эффект β',selected_beta_qty:'Эффект размера группы β',candidate:'Кандидат',candidates:'Кандидаты параметров',mixture:'Смесь бюджетов',beta_time:'Временной эффект β',beta_qty:'Эффект размера группы β',training_mae:'MAE при подборе, мест',training_budget_wasserstein:'Расхождение бюджетов при подборе, ₽',holdout:'Проверка на новых seed',holdout_runs:'Все проверочные запуски',holdout_mae:'MAE на проверке, мест',holdout_budget_wasserstein:'Расхождение бюджетов на проверке, ₽',successful:'Успешных конфигураций',configurations:'Варианты предположений',sensitivity_envelope:'Диапазон чувствительности',bootstrap:'Повторные оценки bootstrap',bootstrap_successful:'Успешных повторных оценок',cutoffs:'Временных срезов',forecasts:'Итоги прогнозов',forecast:'Прогноз загрузки',daily:'Прогноз по дням',mean_final_absolute_error:'Средняя ошибка итога, мест',predicted_net:'Прогноз занятых мест',actual_net:'Фактически занято мест',observed_net:'Известный баланс мест',historical_N_assumption:'Предположение о прошлом N',cutoff:'Момент прогноза',date:'Дата',dtd:'Дней до вылета',mean_net:'Средний прогноз мест',comparison:'Сравнение тарифных политик',policy:'Политика',load_pct:'Загрузка, %',objective_mean:'Среднее значение цели',objective_std:'Разброс значения цели',objective:'Цель',candidate_count:'Кандидатов политик',feasible_training_policy:'Условие загрузки выполнено при подборе',paired_difference:'Изменение результата к текущей политике',selected_kind:'Выбранное правило',selected_price_multiplier:'Множитель цены',selected_growth_multiplier:'Множитель роста',training_candidates:'Подбор тарифных правил',validation_runs_table:'Проверочные реализации',seed_split:'Разделение seed',policy_revenue:'Выручка тарифных политик',revenue_load:'Выручка и загрузка',pareto:'На границе компромисса',primary_seats:'Первичный спрос, мест',purchase_seats:'Куплено до отмен, мест',cancel_seats:'Отменено мест',hidden_seats:'Без покупки, мест',offer_intervals:'Интервалов предложений',value:'Значение',status:'Статус',errors:'Причина ошибки',rejected:'Без покупки, мест',price_rejected:'Отказ по бюджету',capacity_rejected:'Отказ из-за мест',closed_rejected:'Отказ при закрытых продажах'});
const labelOf=key=>LABELS[key]||String(key).replaceAll('_',' ');
Object.assign(LABELS,{envelope:'Чувствительность скрытого спроса',min_estimated:'Минимальная оценка, мест',max_estimated:'Максимальная оценка, мест',min_conditional_lower:'Минимальная условная граница',max_conditional_upper:'Максимальная условная граница',median_budget:'Медианный бюджет, ₽',replicate:'Повторная оценка',forecast_runs:'Траекторий прогноза',paired_difference:'Изменение к текущей политике',candidates:'Кандидаты параметров',holdout_mae_difference:'Разница MAE после калибровки, мест',improves_holdout_mae:'Ошибка на проверке уменьшилась'});

export function createResearchController({getContext,post,onCalculationLoaded,onContextLoaded,onModelSelected,notify=()=>{},resize=()=>{}}){
  const $=id=>document.getElementById(id),esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const number=new Intl.NumberFormat('ru-RU',{maximumFractionDigits:3});
  let connected=false,view='simulation',tool='evaluate',referenceFiles=null,activeJob=null,pollTimer=null,labResult=null,foregroundId=null,historyPending=false;
  const pending=new Map(),seen=new Set();
  function scalar(v){if(v==null)return '—';if(typeof v==='boolean')return v?'Да':'Нет';if(typeof v==='number')return Number.isFinite(v)?number.format(v):'—';if(typeof v==='object')return JSON.stringify(v);return String(v);}
  function showError(message,target='research-error'){$(target).textContent=message||'';$(target).hidden=!message;resize();}
  function busy(value){$('research-run').disabled=value;$('generate-example').disabled=value;}
  function selectedScenarios(){return [...document.querySelectorAll('[data-scenario]:checked')].map(el=>el.value);}
  function list(id){const raw=$(id).value.trim();if(!raw)return [];const values=raw.split(/[;,\s]+/).filter(Boolean).map(Number);if(values.some(v=>!Number.isFinite(v)))throw new Error('Проверьте числовые значения; используйте точку для дробной части и ; между числами.');return values;}
  function integer(id,min,max){const value=Number($(id).value);if(!Number.isInteger(value)||value<min||value>max)throw new Error('Проверьте поле «'+($(id).closest('label')?.firstChild?.textContent||id).trim()+'»: от '+min+' до '+max+'.');return value;}
  function settings(action){
    const research={scenarios:selectedScenarios(),runs:integer('research-runs',1,100),seed_start:integer('research-seed',0,4000000000)};
    if(action==='experiments'&&!research.scenarios.length)throw new Error('Выберите хотя бы один сценарий.');
    if(action==='generate_example')Object.assign(research,{scenario:$('example-scenario').value});
    if(action==='calibration')Object.assign(research,{scenarios:$('calibration-scenario').value==='all'?SCENARIOS.map(s=>s[0]):[$('calibration-scenario').value],training_runs:integer('calibration-training',1,20),validation_runs:integer('calibration-validation',1,20),sigma_values:list('calibration-sigmas'),beta_time_values:list('calibration-time'),beta_qty_values:list('calibration-qty'),seed_start:integer('calibration-seed',0,4000000000)});
    if(action==='sensitivity')Object.assign(research,{sigma_values:list('sensitivity-sigmas'),prior_values:list('sensitivity-priors'),demand_values:list('sensitivity-demands'),bootstrap_runs:integer('sensitivity-bootstrap',0,20)});
    if(action==='forecast')Object.assign(research,{cutoff_dtd:$('forecast-cutoff').value==='all'?30:Number($('forecast-cutoff').value),cutoffs:$('forecast-cutoff').value==='all'?[90,30,14,7]:[Number($('forecast-cutoff').value)]});
    if(action==='policies'){
      const cost=id=>$(id).value.trim()===''?null:Number($(id).value);
      Object.assign(research,{scenario:$('policy-scenario').value,scenarios:[$('policy-scenario').value],objective:$('policy-objective').value,training_runs:integer('policy-training',1,30),validation_runs:integer('policy-validation',2,100),seed_start:integer('policy-seed',0,4000000000),min_load:Number($('policy-min-load').value),variable_cost:cost('policy-variable-cost'),fixed_cost:cost('policy-fixed-cost')});
      if(!Number.isFinite(research.min_load)||research.min_load<0||research.min_load>100)throw new Error('Минимальная загрузка должна быть от 0 до 100%.');
      if(research.objective==='profit'&&[research.variable_cost,research.fixed_cost].some(v=>v==null||!Number.isFinite(v)||v<0))throw new Error('Для сравнения прибыли укажите неотрицательные постоянные и переменные расходы.');
    }
    return research;
  }
  function request(action,extra={},minimal=false){
    if(!connected){showError('Ожидание подключения Python · Streamlit.');return null;}
    const request_id=crypto.randomUUID();
    let value={action,request_id,...extra};
    if(!minimal){const context=getContext();value={...context,reference_files:referenceFiles,action,request_id,research:settings(action),...extra};}
    pending.set(request_id,{action,time:Date.now()});
    if(!['poll_job','cancel_job','history'].includes(action)){foregroundId=request_id;busy(true);showError('');}
    if(action==='history'){$('history-count').textContent='Загрузка…';historyPending=true;showError('','history-error');}
    post('streamlit:setComponentValue',{dataType:'json',value});
    return request_id;
  }
  function chooseView(next,{load=true}={}){
    if(!['simulation','research','history'].includes(next))return;
    view=next;
    for(const key of ['simulation','research','history'])$(key+'-view').hidden=key!==next;
    document.querySelectorAll('[data-app-view]').forEach(b=>{const active=b.dataset.appView===next;b.classList.toggle('active',active);if(active)b.setAttribute('aria-current','page');else b.removeAttribute('aria-current');});
    try{sessionStorage.setItem('kontur-active-view',view);}catch{}
    if(next==='history'&&load&&connected&&!historyPending)request('history',{},true);
    updateContext();resize();
  }
  function chooseTool(next){if(!TOOLS[next])return;tool=next;const [title,description,label]=TOOLS[next];$('research-tool-title').textContent=title;$('research-tool-description').textContent=description;$('research-run').textContent=label;document.querySelectorAll('[data-tool-fields]').forEach(el=>el.hidden=el.dataset.toolFields!==next);document.querySelectorAll('[data-research-tool]').forEach(b=>{const active=b.dataset.researchTool===next;b.classList.toggle('active',active);if(active)b.setAttribute('aria-current','page');else b.removeAttribute('aria-current');});showError('');updateContext();resize();}
  function updateContext(){
    const c=getContext(),p=c.params||{},imports=c.files;
    if(['experiments','calibration','policies'].includes(tool)){$('research-source-title').textContent='Независимые синтетические сценарии';$('research-source-detail').textContent='N = '+scalar(p.demand)+' мест · параметры рейса из симулятора; загруженные журналы не используются.';return;}
    $('research-source-title').textContent=(c.mode==='import'?'Загруженные bookings':'Синтетический сценарий')+' · '+(c.selected_flight||p.flight||'рейс');
    $('research-source-detail').textContent='N = '+scalar(p.demand)+' мест · '+(p.departure||'дата не задана')+(Array.isArray(imports)&&imports.length?' · файлов bookings: '+imports.length:'');
  }
  function updateSize(){const count=selectedScenarios().length,runs=Number($('research-runs').value);$('experiment-size').textContent=count+' сценария × '+scalar(runs)+' запуска = '+scalar(count*runs)+' журналов · парное сравнение v3 и базового метода.';resize();}
  function poll(){clearTimeout(pollTimer);if(activeJob&&RUNNING.has(activeJob.status)){pollTimer=setTimeout(()=>{if(connected)request('poll_job',{job_id:activeJob.id},true);},1800);}}
  function renderJob(job){
    if(!job||!job.id)return;
    if(activeJob&&job.id!==activeJob.id&&job.status==='running')clearTimeout(pollTimer);
    activeJob=job;$('research-job').hidden=false;
    const status=STATUS[job.status]||job.status||'Расчёт';
    $('research-job-title').textContent=status+(job.title?' · '+job.title:'');
    $('research-job-message').textContent=job.message||job.error||'';
    const completed=Number(job.completed)||0,total=Number(job.total)||0;
    $('research-job-count').textContent=total?scalar(completed)+' / '+scalar(total):'';
    if(total){$('research-progress').max=total;$('research-progress').value=Math.min(total,completed);}else{$('research-progress').removeAttribute('value');}
    $('research-cancel').hidden=!RUNNING.has(job.status);$('research-cancel').disabled=job.status==='cancelling';
    busy(RUNNING.has(job.status));
    if(job.status==='failed'||job.status==='error')showError(job.error||job.message||'Расчёт не завершён. Проверьте исходные данные и параметры.');
    poll();resize();
  }
  function columnsOf(table){
    const supplied=table.columns;
    if(Array.isArray(supplied)&&supplied.length)return supplied.map(c=>typeof c==='string'?{key:c,label:labelOf(c)}:{key:c.key||c.name,label:c.label||c.title||labelOf(c.key||c.name)});
    return [...new Set((table.rows||[]).flatMap(r=>Object.keys(r||{})))].map(key=>({key,label:labelOf(key)}));
  }
  function tablesOf(data){
    if(Array.isArray(data))return data.map((t,i)=>Array.isArray(t)?{name:'Таблица '+(i+1),rows:t}:t);
    if(data&&typeof data==='object')return Object.entries(data).map(([name,v])=>Array.isArray(v)?{name,rows:v}:{name,...v});
    return [];
  }
  function renderTables(data){
    const tables=tablesOf(data);
    $('research-tables').innerHTML=tables.map((table,index)=>{
      const rows=table.rows||table.data||[],columns=columnsOf({...table,rows});
      const main=new Set(['by_scenario','overall','holdout','comparison','flights','periods','budgets','cabins','forecasts','sensitivity_envelope']);
      const title=table.name==='paired_runs'?'Парные сравнения':labelOf(table.name||table.title||'Результаты');
      const content='<section class="panel research-data-panel"><div class="panel-heading"><div><h2>'+esc(title)+'</h2><p>'+esc(table.description||scalar(rows.length)+' строк')+'</p></div><button class="text-button" data-lab-table="'+index+'">CSV ↗</button></div><div class="table-scroll"><table><thead><tr>'+columns.map(c=>'<th>'+esc(c.label)+'</th>').join('')+'</tr></thead><tbody>'+rows.slice(0,30).map(row=>'<tr>'+columns.map(c=>'<td title="'+esc(scalar(row[c.key]))+'">'+esc(scalar(row[c.key]).slice(0,350))+'</td>').join('')+'</tr>').join('')+(rows.length?'':'<tr><td colspan="'+Math.max(1,columns.length)+'" class="empty-state">Нет строк для отображения.</td></tr>')+'</tbody></table></div>'+(rows.length>30?'<p class="research-table-note">Показаны первые 30 строк. CSV и отчёт содержат всю таблицу.</p>':'')+'</section>';
      return main.has(table.name)?content:'<details class="research-table-details"><summary>'+esc(title)+' · '+scalar(rows.length)+' строк</summary>'+content+'</details>';
    }).join('');
    $('research-tables').querySelectorAll('[data-lab-table]').forEach(button=>button.addEventListener('click',()=>{const table=tables[Number(button.dataset.labTable)],rows=table.rows||table.data||[],formatted=rows.map(row=>Object.fromEntries(Object.entries(row).map(([key,value])=>[key,value&&typeof value==='object'?JSON.stringify(value):value])));saveFile(toCsv(formatted,columnsOf({...table,rows}).map(c=>c.key)),'research-table-'+(Number(button.dataset.labTable)+1)+'.csv','text/csv;charset=utf-8');}));
  }
  function renderCharts(charts){
    const colors=['#0b7467','#dc8d53','#84bbac','#778b9e','#aa917d'];
    const normalized=Array.isArray(charts)?charts:charts&&typeof charts==='object'?Object.entries(charts).map(([name,value])=>{
      if(!Array.isArray(value))return {title:name,...value};
      const first=value[0]||{},numeric=Object.keys(first).filter(key=>typeof first[key]==='number');
      const scatter='x' in first&&'y' in first;
      const xkey=scatter?'x':'date' in first?'date':'label' in first?'label':'period' in first?'period':'cutoff_dtd' in first?'cutoff_dtd':Object.keys(first).find(key=>typeof first[key]!=='number')||numeric[0];
      const selected={holdout:['holdout_mae'],periods:['actual','estimated'],budget_quantiles:['actual','estimated'],forecast:['predicted_net','actual_net','lower','upper'],period_mae:['value'],policy_revenue:['value'],envelope:['min_estimated','max_estimated']};
      return {title:labelOf(name),type:scatter?'scatter':'date' in first?'line':'bar',x_key:xkey,y_keys:scatter?['y']:selected[name]||numeric.filter(key=>key!==xkey&&key!=='dtd'&&key!=='seed'),data:value};
    }):[];
    $('research-charts').innerHTML=normalized.map(chart=>{
      const data=chart.data||chart.rows||[],xkey=chart.x_key||chart.x,keys=(chart.y_keys||chart.series||[]).map(s=>typeof s==='string'?{key:s,label:labelOf(s)}:{key:s.key||s.y,label:s.label||s.name||labelOf(s.key)});
      if(!data.length||!xkey||!keys.length)return '';
      const w=850,h=240,p={l:56,r:16,t:20,b:40},numbers=data.flatMap(r=>keys.map(k=>Number(r[k.key]))).filter(Number.isFinite);
      if(!numbers.length)return '';
      const lower=Math.min(0,...numbers),upper=Math.max(1,...numbers),span=upper-lower||1;
      const xs=data.map(row=>Number(row[xkey])).filter(Number.isFinite),xmin=Math.min(...xs),xmax=Math.max(...xs),xspan=xmax-xmin||1;
      const x=i=>p.l+(chart.type==='scatter'?(Number(data[i][xkey])-xmin)/xspan:data.length<2?.5:i/(data.length-1))*(w-p.l-p.r),y=v=>h-p.b-(v-lower)/span*(h-p.t-p.b);
      let svg='<svg viewBox="0 0 '+w+' '+h+'" role="img" aria-label="'+esc(chart.title||'График исследования')+'">';
      for(let i=0;i<4;i++){const v=lower+span*i/3;svg+='<line x1="'+p.l+'" y1="'+y(v)+'" x2="'+(w-p.r)+'" y2="'+y(v)+'" stroke="#e8efeb"/><text x="'+(p.l-8)+'" y="'+(y(v)+4)+'" text-anchor="end">'+esc(number.format(v))+'</text>';}
      keys.forEach((key,k)=>{
        const valid=data.map((row,i)=>({row,i,v:Number(row[key.key])})).filter(v=>rowNumber(v.row,key.key)&&Number.isFinite(v.v));
        if(chart.type==='bar'){
          const step=(w-p.l-p.r)/Math.max(1,data.length),bw=Math.min(30,step/(keys.length+1));
          valid.forEach(({row,i,v})=>{const xx=p.l+step*(i+.5)+(k-(keys.length-1)/2)*bw;svg+='<rect x="'+(xx-bw*.42)+'" y="'+Math.min(y(0),y(v))+'" width="'+bw*.84+'" height="'+Math.max(1,Math.abs(y(0)-y(v)))+'" fill="'+colors[k%colors.length]+'"><title>'+esc(row[xkey]+' · '+key.label+': '+scalar(v))+'</title></rect>';});
        }else{if(chart.type!=='scatter')svg+='<polyline fill="none" stroke="'+colors[k%colors.length]+'" stroke-width="2.2" points="'+valid.map(({i,v})=>x(i)+','+y(v)).join(' ')+'"/>';valid.forEach(({row,i,v})=>{svg+='<circle cx="'+x(i)+'" cy="'+y(v)+'" r="'+(row.pareto?5:3)+'" fill="'+colors[k%colors.length]+'"><title>'+esc((row.label||row[xkey])+' · '+key.label+': '+scalar(v))+'</title></circle>';});}
      });
      const stride=Math.max(1,Math.ceil(data.length/9));data.forEach((row,i)=>{if(i%stride===0||i===data.length-1){const xx=chart.type==='bar'?p.l+(w-p.l-p.r)/data.length*(i+.5):x(i);svg+='<text x="'+xx+'" y="'+(h-12)+'" text-anchor="middle">'+esc(chart.type==='scatter'?scalar(row[xkey]):String(row[xkey]).slice(0,18))+'</text>';}});
      svg+='</svg>';
      return '<section class="panel research-chart-panel"><div class="panel-heading"><div><h2>'+esc(chart.title||'График')+'</h2><p>'+esc(chart.description||'')+'</p></div></div><div class="chart-legend">'+keys.map((key,k)=>'<span><i class="legend-square" style="background:'+colors[k%colors.length]+'"></i>'+esc(key.label)+'</span>').join('')+'</div><div class="research-chart">'+svg+'</div></section>';
    }).join('');
  }
  function rowNumber(row,key){return row[key]!=null&&row[key]!=='';}
  function renderLab(data){
    if(!data||typeof data!=='object')return;
    labResult=data;$('research-result').hidden=false;$('research-empty').hidden=true;
    $('research-result-title').textContent=data.title||'Результат исследования';
    let items=data.summary;
    if(Array.isArray(items))items=items.map((v,i)=>typeof v==='object'?{label:v.label||v.name||String(i+1),value:v.value??v.result}: {label:'',value:v});
    else if(items&&typeof items==='object')items=Object.entries(items).filter(([,value])=>typeof value!=='object'||value==null).map(([key,value])=>({key,label:labelOf(key),value}));
    else items=items==null?[]:[{label:'',value:items}];
    const priority=['paired_runs','v3_win_pct','mean_mae_difference','difference_ci_low','difference_ci_high','failed_fits','flights','training_runs','validation_runs','selected_sigma','selected_mixture','selected_beta_time','selected_beta_qty','successful','bootstrap_successful','configurations','cutoffs','mean_final_absolute_error','paired_difference','candidate_count','feasible_training_policy','primary_seats','purchase_seats','hidden_seats'];
    items.sort((a,b)=>(priority.includes(a.key)?priority.indexOf(a.key):100)-(priority.includes(b.key)?priority.indexOf(b.key):100));
    $('research-summary').innerHTML=items.slice(0,6).map(item=>'<article class="research-summary-card"><span>'+esc(item.label)+'</span><strong title="'+esc(scalar(item.value))+'">'+esc(scalar(item.value))+'</strong></article>').join('')+(items.length>6?'<details class="research-summary-extra"><summary>Все показатели</summary><dl>'+items.slice(6).map(item=>'<dt>'+esc(item.label)+'</dt><dd>'+esc(scalar(item.value))+'</dd>').join('')+'</dl></details>':'');
    const warnings=Array.isArray(data.warnings)?data.warnings:[data.warnings].filter(Boolean);
    $('research-warnings').innerHTML='<details><summary>Допущения и ограничения исследования</summary><ul>'+warnings.map(v=>'<li>'+esc(typeof v==='object'?JSON.stringify(v):v)+'</li>').join('')+'</ul></details>';$('research-warnings').hidden=!warnings.length;
    renderTables(data.tables);renderCharts(data.charts);
    const calibrated=data.selected_model||data.best_model||data.best_config||data.selected_config||data.calibrated_model;
    $('apply-calibration').hidden=!calibrated;
    if(data.ok===false)showError(data.error||'Исследование завершено с ошибкой. Смотрите сохранённые результаты.');
    resize();
  }
  function renderHistory(rows){
    historyPending=false;const list=Array.isArray(rows)?rows:rows?.runs||[];
    $('history-count').textContent=list.length?'Сохранённых расчётов: '+scalar(list.length):'Сохранённых расчётов пока нет';
    $('history-body').innerHTML=list.map(run=>'<tr><td>'+esc((run.created_at||run.created||'').replace('T',' ').slice(0,19))+'</td><td><b>'+esc(run.title||TOOLS[run.action]?.[0]||run.action||'Расчёт')+'</b><small>'+esc(run.id||run.run_id||'')+'</small></td><td><span class="history-status '+esc(run.status||'completed')+'">'+esc(STATUS[run.status]||run.status||'Завершено')+'</span></td><td><button class="text-button" data-load-run="'+esc(run.id||run.run_id||'')+'">Открыть ↗</button> <button class="text-button" data-replay-run="'+esc(run.id||run.run_id||'')+'">Повторить</button></td></tr>').join('')||'<tr><td colspan="4" class="empty-state">Запустите исследование — его параметры и результат появятся здесь.</td></tr>';
    $('history-body').querySelectorAll('[data-load-run]').forEach(button=>button.addEventListener('click',()=>{chooseView('research',{load:false});request('load_run',{run_id:button.dataset.loadRun},true);}));$('history-body').querySelectorAll('[data-replay-run]').forEach(button=>button.addEventListener('click',()=>{chooseView('research',{load:false});request('replay_run',{run_id:button.dataset.replayRun},true);}));resize();
  }
  function saveFile(body,filename,type){const url=URL.createObjectURL(new Blob([body],{type})),a=document.createElement('a');a.href=url;a.download=filename;a.style.display='none';document.body.appendChild(a);try{a.click();}finally{a.remove();setTimeout(()=>URL.revokeObjectURL(url),30000);}notify('Файл подготовлен. Скачивание через Streamlit также доступно внизу страницы.');}
  function handleResponse(response){
    if(!response||!ACTIONS.has(response.action))return false;
    const id=response.request_id,meta=pending.get(id);
    if(id&&seen.has(id))return true;
    if(id){seen.add(id);pending.delete(id);if(seen.size>400)seen.delete(seen.values().next().value);}
    const action=response.action;
    if(response.ok===false){if(action==='history')historyPending=false;busy(false);showError(response.error||'Не удалось выполнить запрос.',action==='history'?'history-error':'research-error');if(action==='poll_job')poll();return true;}
    if(action==='history'){renderHistory(response.history||response.runs||[]);return true;}
    if(action==='cancel_job'&&response.job){renderJob(response.job);return true;}
    if(meta&&foregroundId&&id!==foregroundId&&!['poll_job','cancel_job','history'].includes(action))return true;
    if(response.history)renderHistory(response.history);
    const saved=response.saved_run;
    if(action==='load_run'&&saved){
      if(!response.results)onContextLoaded?.(response);referenceFiles=null;
      const names=(saved.config?.reference_files||[]).map(file=>file.name);
      $('reference-summary').textContent=names.length?names.join(' · ')+' · из истории':'';
      $('reference-summary').hidden=!names.length;$('clear-reference').hidden=!names.length;
      const s=saved.config?.research||{},kind=saved.action;
      const set=(id,value)=>{if(value!=null&&$(id))$(id).value=Array.isArray(value)?value.join('; '):value;};
      if(kind==='experiments'){set('research-runs',s.runs);set('research-seed',s.seed_start);if(s.scenarios)document.querySelectorAll('[data-scenario]').forEach(input=>input.checked=s.scenarios.includes(input.value));updateSize();}
      if(kind==='calibration'){set('calibration-training',s.training_runs);set('calibration-validation',s.validation_runs);set('calibration-sigmas',s.sigma_values);set('calibration-time',s.beta_time_values);set('calibration-qty',s.beta_qty_values);set('calibration-seed',s.seed_start);set('calibration-scenario',s.scenarios?.length===10?'all':s.scenarios?.[0]);}
      if(kind==='sensitivity'){set('sensitivity-sigmas',s.sigma_values);set('sensitivity-priors',s.prior_values);set('sensitivity-demands',s.demand_values);set('sensitivity-bootstrap',s.bootstrap_runs);}
      if(kind==='forecast')set('forecast-cutoff',s.cutoffs?.length===4?'all':s.cutoffs?.[0]??s.cutoff_dtd);
      if(kind==='policies')for(const [key,id] of Object.entries({scenario:'policy-scenario',training_runs:'policy-training',validation_runs:'policy-validation',seed_start:'policy-seed',objective:'policy-objective',min_load:'policy-min-load',variable_cost:'policy-variable-cost',fixed_cost:'policy-fixed-cost'}))set(id,s[key]);
      if(TOOLS[kind])chooseTool(kind);
    }
    const data=response.lab||saved?.lab||saved?.result?.lab||saved?.result;
    if(response.job){
      if(action==='poll_job'&&activeJob&&response.job.id!==activeJob.id)return true;
      renderJob(response.job);
    }else if(!['poll_job','cancel_job'].includes(action))busy(false);
    if(data)renderLab(data);
    if(['load_run','replay_run'].includes(action)&&response.results){onCalculationLoaded?.(response);chooseView('simulation',{load:false});}
    else if(['load_run','replay_run','generate_example'].includes(action))chooseView('research',{load:false});
    if(!response.job||!RUNNING.has(response.job.status))busy(false);
    return true;
  }
  $('scenario-checklist').innerHTML=SCENARIOS.map(([key,label],i)=>'<label class="scenario-option"><input type="checkbox" value="'+key+'" data-scenario'+(i<3?' checked':'')+'><span>'+label+'</span></label>').join('');
  for(const id of ['example-scenario','policy-scenario','calibration-scenario'])$(id).innerHTML=SCENARIOS.map(([key,label])=>'<option value="'+key+'">'+label+'</option>').join('')+(id==='calibration-scenario'?'<option value="all">Все 10 сценариев</option>':'');
  document.querySelectorAll('[data-app-view]').forEach(button=>button.addEventListener('click',()=>chooseView(button.dataset.appView)));
  document.querySelectorAll('[data-research-tool]').forEach(button=>button.addEventListener('click',()=>chooseTool(button.dataset.researchTool)));
  $('research-open-simulation').addEventListener('click',()=>chooseView('simulation'));
  $('scenario-checklist').addEventListener('change',updateSize);$('research-runs').addEventListener('input',updateSize);
  $('research-quick').addEventListener('click',()=>{document.querySelectorAll('[data-scenario]').forEach((input,i)=>input.checked=i<3);$('research-runs').value=3;updateSize();});
  $('research-full').addEventListener('click',()=>{document.querySelectorAll('[data-scenario]').forEach(input=>input.checked=true);$('research-runs').value=100;updateSize();});
  $('research-form').addEventListener('submit',event=>{event.preventDefault();try{request(tool);}catch(error){busy(false);showError(error.message);}});
  $('generate-example').addEventListener('click',()=>{try{request('generate_example');}catch(error){showError(error.message);}});
  $('policy-objective').addEventListener('change',()=>{$('policy-costs').hidden=$('policy-objective').value!=='profit';resize();});
  $('reference-input').addEventListener('change',async event=>{try{const chosen=[...event.target.files];if(!chosen.length)return;if(chosen.reduce((sum,file)=>sum+file.size,0)>10*1024*1024)throw new Error('Эталонные CSV вместе не должны превышать 10 МБ.');referenceFiles=await Promise.all(chosen.map(async file=>({name:file.name,text:await file.text()})));$('reference-summary').textContent=chosen.map(f=>f.name).join(' · ');$('reference-summary').hidden=false;$('clear-reference').hidden=false;showError('');resize();}catch(error){showError(error.message);}});
  $('clear-reference').addEventListener('click',()=>{referenceFiles=[];$('reference-input').value='';$('reference-summary').hidden=true;$('clear-reference').hidden=true;resize();});
  $('research-cancel').addEventListener('click',()=>{if(activeJob){$('research-cancel').disabled=true;request('cancel_job',{job_id:activeJob.id},true);}});
  $('history-refresh').addEventListener('click',()=>request('history',{},true));
  $('research-report-download').addEventListener('click',()=>{if(labResult)saveFile(JSON.stringify(labResult,null,2),'kontur-research-report.json','application/json');});
  $('apply-calibration').addEventListener('click',()=>{const config=labResult?.selected_model||labResult?.best_model||labResult?.best_config||labResult?.selected_config||labResult?.calibrated_model;if(config){onModelSelected?.(config);notify('Параметры перенесены в симулятор.');chooseView('simulation');}});
  updateSize();
  try{const saved=sessionStorage.getItem('kontur-active-view');if(['simulation','research','history'].includes(saved))chooseView(saved,{load:false});}catch{}
  return {handleResponse,updateContext,getReferenceFiles:()=>referenceFiles,chooseView,
    connect(){const first=!connected;connected=true;if(first&&view==='history')request('history',{},true);},
    restoreJob(job){if(job)renderJob(job);},
    dispose(){clearTimeout(pollTimer);}
  };
}
