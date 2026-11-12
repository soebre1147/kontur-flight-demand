// UI defaults and CSV formatting only. All model calculations run in Python.
export const PERIODS = ['91+', '31–90', '15–30', '8–14', '0–7'];
export const DEFAULTS = Object.freeze({demand:300,capacity:180,businessCapacity:12,businessShare:5,price:12000,budget:23000,cancel:4,growth:80,profile:'balanced',seed:42,departure:'2027-05-22',flight:'SU1122'});
export const PROFILES = {balanced:[10,24,26,20,20],early:[35,30,18,10,7],late:[5,12,20,25,38],notebook:[57,54,70,46,26]};
export const MODEL_DEFAULTS = Object.freeze({method:'v3',total_mode:'fixed_design',sigma:.45,prior_strength:.5,beta_time:0,beta_qty:0,mixture:false});
export function toCsv(rows,columns){const escape=v=>'"'+String(v??'').replaceAll('"','""')+'"';return '\uFEFF'+[columns.map(escape).join(';'),...rows.map(row=>columns.map(k=>escape(row[k])).join(';'))].join('\r\n');}
export const RESULT_COLUMNS=['timestamp','id','cabin','rbd','qty','price','amount','amount_source','outcome','flight','dtd','period','source_file','source_row','id_is_synthetic','reconstruction_status','record_level','row_source','signed_qty','model_preferred_cabin','reason','reason_is_modelled'];
