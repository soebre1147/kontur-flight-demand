"""Native Streamlit controls, preserved result state and restored history."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from ui.charts import fare_frame,period_frame,_known_chart
from ui.research import normalise_settings

ROOT=Path(__file__).resolve().parents[1]


class NativePresentationTests(unittest.TestCase):
    def test_fare_price_is_weighted_by_seats_and_RBD_order_is_historical(self):
        data={'fares':[{'cabin':'Y','rbd':'V','price':100,'purchased':1,'canceled':0},
                       {'cabin':'Y','rbd':'V','price':200,'purchased':3,'canceled':1},
                       {'cabin':'Y','rbd':'Y','price':400,'purchased':2,'canceled':0}]}
        frame=fare_frame(data)
        self.assertEqual(frame.rbd.tolist(),['Y','V'])
        self.assertEqual(frame.loc[frame.rbd.eq('V'),'average_price'].iloc[0],175)

    def test_import_does_not_turn_missing_reference_into_zero_truth(self):
        frame=period_frame({'mode':'import','periods':[{'period':'0–7','estimated':8,'actual':None}]},'estimate')
        self.assertEqual(frame.series.tolist(),['Оценка модели']);self.assertEqual(frame.seats.tolist(),[8])

    def test_research_chart_compares_two_methods_as_separate_series(self):
        chart=_known_chart('period_mae',[{'label':'Сценарий · baseline','value':12},{'label':'Сценарий · v3','value':8}])
        spec=chart.to_dict();records=next(iter(spec['datasets'].values()))
        self.assertEqual({row['series'] for row in records},{'Базовый метод','Модель v3'})

    def test_optional_N_and_profit_cost_validation_match_native_forms(self):
        settings=normalise_settings('sensitivity',{'sigma_values':'.45','prior_values':'.5','demand_values':'','bootstrap_runs':0})
        self.assertEqual(settings['demand_values'],[])
        with self.assertRaises(ValueError):normalise_settings('policies',{'objective':'profit','variable_cost':None,'fixed_cost':100})


class NativeAppTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='kontur-native-ui-')
        self.env=patch.dict(os.environ,{'KONTUR_DATA_DIR':self.temp.name});self.env.start()
        self.app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=30).run()
        self.assertFalse(list(self.app.exception))

    def tearDown(self):
        self.env.stop();self.temp.cleanup()

    def submit(self):
        button=next(button for button in self.app.button if button.label=='Рассчитать сценарий')
        button.click().run()
        self.assertFalse(list(self.app.exception))

    def test_native_form_recalculates_and_keeps_correct_totals(self):
        original=self.app.session_state['native_last_calculation']['results']['SU1122']['summary']
        self.assertEqual((original['gross'],original['net']),(180,174))
        self.app.number_input(key='simulator_p_capacity').set_value(150)
        self.submit()
        summary=self.app.session_state['native_last_calculation']['results']['SU1122']['summary']
        self.assertEqual((summary['gross'],summary['net']),(147,144))
        self.assertEqual(len(self.app.get('component_instance')),0)

    def test_invalid_capacity_retains_last_valid_calculation(self):
        identifier=self.app.session_state['native_last_calculation']['run_id']
        self.app.number_input(key='simulator_p_businessCapacity').set_value(200)
        self.submit()
        self.assertEqual(self.app.session_state['native_last_calculation']['run_id'],identifier)
        self.assertTrue(any('бизнес' in message.value for message in self.app.error))

    def test_native_research_and_history_navigation(self):
        self.app.get('button_group')[0].set_value('Исследование').run()
        self.assertFalse(list(self.app.exception))
        self.assertEqual(len(self.app.selectbox(key='research_tool').options),6)
        self.app.selectbox(key='research_tool').set_value('sensitivity').run()
        self.assertFalse(list(self.app.exception))
        self.assertEqual(self.app.text_input(key='research_sens_n').value,'')
        self.app.get('button_group')[0].set_value('История').run()
        self.assertFalse(list(self.app.exception))
        self.assertTrue(any(button.label=='Открыть' for button in self.app.button))

    def test_history_restores_form_values_before_widget_creation(self):
        original_id=self.app.session_state['native_last_calculation']['run_id']
        self.app.number_input(key='simulator_p_demand').set_value(110);self.submit()
        self.app.get('button_group')[0].set_value('История').run()
        self.app.button(key='history_open_'+original_id).click().run()
        self.assertFalse(list(self.app.exception))
        self.assertEqual(self.app.number_input(key='simulator_p_demand').value,300)
        self.assertEqual(self.app.session_state['native_last_calculation']['run_id'],original_id)

    def test_research_history_restores_effective_settings(self):
        self.app.session_state['pending_research_settings']={'action':'experiments','research':{'scenarios':['late'],'runs':7,'seed_start':123}}
        self.app.get('button_group')[0].set_value('Исследование').run()
        self.assertFalse(list(self.app.exception))
        self.assertEqual(self.app.radio(key='research_preset').value,'Настроить')
        self.assertEqual(self.app.number_input(key='research_runs').value,7)
        self.app.session_state['pending_research_settings']={'action':'forecast','research':{'cutoffs':[90,30,14,7],'forecast_runs':460}}
        self.app.run()
        self.assertFalse(list(self.app.exception))
        self.assertEqual(self.app.number_input(key='research_forecast_paths').value,460)
        self.app.session_state['pending_research_settings']={'action':'policies','research':{'objective':'profit','variable_cost':1250.,'fixed_cost':80000.}}
        self.app.run()
        self.assertFalse(list(self.app.exception))
        self.assertTrue(self.app.checkbox(key='research_policy_costs_known').value)
        self.assertEqual(self.app.number_input(key='research_policy_variable').value,1250.)


if __name__=='__main__':unittest.main()
