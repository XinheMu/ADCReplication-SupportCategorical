import sys
import os
from dotenv import load_dotenv
import pickle
import pandas as pd
import csv

def convert(ALL_ATTRIBUTE_NAMES_ORDERED=0, ATTRIBUTE_METADATA=0,
            workload_file_path=0, output_custom_csv_path=0,
            shortcut=0, querytype=0):
    project_root = os.path.dirname(os.path.abspath(__file__))
    dotenv_path = os.path.join(project_root, '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)
        print(f"Loaded .env from {dotenv_path}")
    else:
        print(f"Warning: .env file not found at {dotenv_path}")

    data_root_env = os.getenv('DATA_ROOT')
    print(f"DATA_ROOT from environment: {data_root_env}")
    if not data_root_env:
        print("CRITICAL ERROR: DATA_ROOT not found in environment. Exiting.")
        exit(1)

    # ---------- 数据集元数据定义 ----------
    if shortcut == 'advantage':
        ALL_ATTRIBUTE_NAMES_ORDERED = ['A', 'B', 'C', 'D', 'E']
        ATTRIBUTE_METADATA = {
            'A': {'minval': 0, 'maxval': 1999, 'dtype': 'int'},
            'B': {'minval': 0, 'maxval': 1999, 'dtype': 'int'},
            'C': {'minval': 0, 'maxval': 1999, 'dtype': 'int'},
            'D': {'minval': 0, 'maxval': 1999, 'dtype': 'int'},
            'E': {'minval': 0, 'maxval': 1999, 'dtype': 'int'},
        }
        print("Using defined metadata for 'advantage' dataset.")
        workload_file_path = "advantage/workload/base.pkl"
        output_custom_csv_path = "data/advantage_" + querytype + "set.csv"

    elif shortcut == 'power':
        ALL_ATTRIBUTE_NAMES_ORDERED = [
            'Global_active_power', 'Global_reactive_power', 'Voltage',
            'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'
        ]
        ATTRIBUTE_METADATA = {
            'Global_active_power':   {'minval': -1.0, 'maxval': 11.122, 'dtype': 'float64'},
            'Global_reactive_power': {'minval': -1.0, 'maxval': 1.39,   'dtype': 'float64'},
            'Voltage':               {'minval': -1.0, 'maxval': 254.15, 'dtype': 'float64'},
            'Global_intensity':      {'minval': -1.0, 'maxval': 48.4,   'dtype': 'float64'},
            'Sub_metering_1':        {'minval': -1.0, 'maxval': 88,     'dtype': 'float64'},
            'Sub_metering_2':        {'minval': -1.0, 'maxval': 80,     'dtype': 'float64'},
            'Sub_metering_3':        {'minval': -1.0, 'maxval': 31,     'dtype': 'float64'},
        }
        print("Using defined metadata for 'power' dataset.")
        workload_file_path = "data/power/workload/base.pkl"
        output_custom_csv_path = "power_" + querytype + "set.csv"

    elif shortcut == 'forest':
        ALL_ATTRIBUTE_NAMES_ORDERED = [
            'Elevation', 'Aspect', 'Slope',
            'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
            'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon',
            'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points'
        ]
        ATTRIBUTE_METADATA = {
            'Elevation':                          {'minval': 1859, 'maxval': 3858, 'dtype': 'int64'},
            'Aspect':                             {'minval': 0,    'maxval': 360,  'dtype': 'int64'},
            'Slope':                              {'minval': 0,    'maxval': 66,   'dtype': 'int64'},
            'Horizontal_Distance_To_Hydrology':   {'minval': 0,    'maxval': 1397, 'dtype': 'int64'},
            'Vertical_Distance_To_Hydrology':     {'minval': -173, 'maxval': 601,  'dtype': 'int64'},
            'Horizontal_Distance_To_Roadways':    {'minval': 0,    'maxval': 7117, 'dtype': 'int64'},
            'Hillshade_9am':                      {'minval': 0,    'maxval': 254,  'dtype': 'int64'},
            'Hillshade_Noon':                     {'minval': 0,    'maxval': 254,  'dtype': 'int64'},
            'Hillshade_3pm':                      {'minval': 0,    'maxval': 254,  'dtype': 'int64'},
            'Horizontal_Distance_To_Fire_Points': {'minval': 0,    'maxval': 7173, 'dtype': 'int64'},
        }
        print("Using defined metadata for 'forest' dataset.")
        workload_file_path = "data/forest/workload/base.pkl"
        output_custom_csv_path = "forest_" + querytype + "set.csv"

    elif shortcut == 'higgs':
        ALL_ATTRIBUTE_NAMES_ORDERED = ['m_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb']
        ATTRIBUTE_METADATA = {
            'm_jj':   {'minval': 0.074, 'maxval': 40.193, 'dtype': 'float64'},
            'm_jjj':  {'minval': 0.198, 'maxval': 20.374, 'dtype': 'float64'},
            'm_lv':   {'minval': 0.082, 'maxval': 7.994,  'dtype': 'float64'},
            'm_jlv':  {'minval': 0.131, 'maxval': 14.263, 'dtype': 'float64'},
            'm_bb':   {'minval': 0.047, 'maxval': 17.764, 'dtype': 'float64'},
            'm_wbb':  {'minval': 0.294, 'maxval': 11.498, 'dtype': 'float64'},
            'm_wwbb': {'minval': 0.330, 'maxval': 8.375,  'dtype': 'float64'},
        }
        print("Using defined metadata for 'higgs' dataset.")
        workload_file_path = "data/higgs/workload/base.pkl"
        output_custom_csv_path = "higgs_" + querytype + "set.csv"

    elif shortcut=='taxi':
        ALL_ATTRIBUTE_NAMES_ORDERED = [
            'trip_pickup_datetime', 'trip_dropoff_datetime', 'passenger_count',
            'trip_distance', 'start_lon', 'start_lat', 'end_lon', 'end_lat',
            'fare_amt', 'surcharge', 'tip_amt', 'tolls_amt', 'total_amt',
            'hourlyaltimetersetting', 'hourlydewpointtemperature', 'hourlydrybulbtemperature',
            'hourlyprecipitation', 'hourlyrelativehumidity', 'hourlysealevelpressure',
            'hourlystationpressure', 'hourlyvisibility', 'hourlywetbulbtemperature',
            'hourlywindspeed'
        ]
    
        ATTRIBUTE_METADATA = {
            'trip_pickup_datetime': {'minval': 1.0, 'maxval': 744.0, 'dtype': 'float64'},
            'trip_dropoff_datetime': {'minval': 0.0, 'maxval': 1630.0, 'dtype': 'float64'},
            'passenger_count': {'minval': 0.0, 'maxval': 113.0, 'dtype': 'float64'},
            'trip_distance': {'minval': 0.0, 'maxval': 50.0, 'dtype': 'float64'},
            'start_lon': {'minval': -775.45, 'maxval': 3555.9128, 'dtype': 'float64'},
            'start_lat': {'minval': -7.3351, 'maxval': 935.5253, 'dtype': 'float64'},
            'end_lon': {'minval': -784.3, 'maxval': 0.1014, 'dtype': 'float64'},
            'end_lat': {'minval': -7.3351, 'maxval': 1809.9578, 'dtype': 'float64'},
            'fare_amt': {'minval': 2.5, 'maxval': 200.0, 'dtype': 'float64'},
            'surcharge': {'minval': 0.0, 'maxval': 5.0, 'dtype': 'float64'},
            'tip_amt': {'minval': 0.0, 'maxval': 100.0, 'dtype': 'float64'},
            'tolls_amt': {'minval': 0.0, 'maxval': 20.0, 'dtype': 'float64'},
            'total_amt': {'minval': 2.5, 'maxval': 234.0, 'dtype': 'float64'},
            'hourlyaltimetersetting': {'minval': -100.0, 'maxval': 30.61, 'dtype': 'float64'},
            'hourlydewpointtemperature': {'minval': -100.0, 'maxval': 39.0, 'dtype': 'float64'},
            'hourlydrybulbtemperature': {'minval': -100.0, 'maxval': 46.0, 'dtype': 'float64'},
            'hourlyprecipitation': {'minval': -100.0, 'maxval': 0.15, 'dtype': 'float64'},
            'hourlyrelativehumidity': {'minval': -100.0, 'maxval': 97.0, 'dtype': 'float64'},
            'hourlysealevelpressure': {'minval': -100.0, 'maxval': 30.59, 'dtype': 'float64'},
            'hourlystationpressure': {'minval': -100.0, 'maxval': 30.44, 'dtype': 'float64'},
            'hourlyvisibility': {'minval': 0.25, 'maxval': 10.0, 'dtype': 'float64'},
            'hourlywetbulbtemperature': {'minval': -100.0, 'maxval': 40.0, 'dtype': 'float64'},
            'hourlywindspeed': {'minval': 0.0, 'maxval': 20.0, 'dtype': 'float64'},
        }

    elif shortcut == 'census':
        ALL_ATTRIBUTE_NAMES_ORDERED = [
            'age', 'workclass', 'education', 'education_num', 'marital_status',
            'occupation', 'relationship', 'race', 'sex', 'capital_gain',
            'capital_loss', 'hours_per_week', 'native_country'
        ]
        ATTRIBUTE_METADATA = {
            'age':             {'minval': 17,   'maxval': 90,     'dtype': 'int64'},
            'workclass':       {'minval': 0,    'maxval': 0,      'dtype': 'categorical'},
            'education':       {'minval': 0,    'maxval': 0,      'dtype': 'categorical'},
            'education_num':   {'minval': 1,    'maxval': 16,     'dtype': 'int64'},
            'marital_status':  {'minval': 0,    'maxval': 0,      'dtype': 'categorical'},
            'occupation':      {'minval': 0,    'maxval': 0,      'dtype': 'categorical'},
            'relationship':    {'minval': 0,    'maxval': 0,      'dtype': 'categorical'},
            'race':            {'minval': 0,    'maxval': 0,      'dtype': 'categorical'},
            'sex':             {'minval': 0,    'maxval': 0,      'dtype': 'categorical'},
            'capital_gain':    {'minval': 0,    'maxval': 99999,  'dtype': 'int64'},
            'capital_loss':    {'minval': 0,    'maxval': 4356,   'dtype': 'int64'},
            'hours_per_week':  {'minval': 1,    'maxval': 99,     'dtype': 'int64'},
            'native_country':  {'minval': 0,    'maxval': 0,      'dtype': 'categorical'},
        }
        print("Using defined metadata for 'census' dataset.")
        workload_file_path = "data/census/workload/base.pkl"
        output_custom_csv_path = "census_" + querytype + "set.csv"

    elif shortcut == 'dmv':
        ALL_ATTRIBUTE_NAMES_ORDERED = [
            'Record_Type', 'Registration_Class', 'State', 'County',
            'Body_Type', 'Fuel_Type', 'Reg_Valid_Date', 'Color',
            'Scofflaw_Indicator', 'Suspension_Indicator', 'Revocation_Indicator'
        ]
        ATTRIBUTE_METADATA = {
            'Record_Type':           {'minval': 0,    'maxval': 0,      'dtype': 'categorical'},
            'Registration_Class':    {'minval': 0,    'maxval': 0,      'dtype': 'categorical'},
            'State':                 {'minval': 0,    'maxval': 0,      'dtype': 'categorical'},
            'County':                {'minval': 0,    'maxval': 0,      'dtype': 'categorical'},
            'Body_Type':             {'minval': 0,    'maxval': 0,      'dtype': 'categorical'},
            'Fuel_Type':             {'minval': 0,    'maxval': 0,      'dtype': 'categorical'},
            'Reg_Valid_Date':        {'minval': 19721201, 'maxval': 20190401, 'dtype': 'int64'},
            'Color':                 {'minval': 0,    'maxval': 0,      'dtype': 'categorical'},
            'Scofflaw_Indicator':    {'minval': 0,    'maxval': 0,      'dtype': 'categorical'},
            'Suspension_Indicator':  {'minval': 0,    'maxval': 0,      'dtype': 'categorical'},
            'Revocation_Indicator':  {'minval': 0,    'maxval': 0,      'dtype': 'categorical'},
        }
        print("Using defined metadata for 'dmv' dataset.")
        workload_file_path = "data/dmv/workload/base.pkl"
        output_custom_csv_path = "dmv_" + querytype + "set.csv"

    else:
        print(f"Unknown shortcut '{shortcut}'. Exiting.")
        exit(1)

    # ---------- 转换核心 ----------
    output_rows_for_csv = []

    try:
        print(f"Attempting to load workload from: {workload_file_path}")
        with open(workload_file_path, "rb") as f:
            loaded_workload_dict = pickle.load(f)
        print("Workload file loaded successfully.")

        all_queries_to_process = []
        for key in [querytype]:
            if key in loaded_workload_dict and isinstance(loaded_workload_dict[key], list):
                print(f"Adding {len(loaded_workload_dict[key])} queries from split: {key}")
                all_queries_to_process.extend(loaded_workload_dict[key])
            else:
                print(f"Split '{key}' not found or not a list in workload dictionary.")

        if not all_queries_to_process:
            print("No queries found to process. Exiting.")
            exit()

        print(f"Total queries to process: {len(all_queries_to_process)}")

        for i, query_obj in enumerate(all_queries_to_process):
            if not hasattr(query_obj, 'predicates'):
                print(f"Warning: Item at index {i} is not a valid Query object (no 'predicates'). Skipping.")
                continue

            query_predicates = query_obj.predicates
            current_query_low_bounds = []
            current_query_high_bounds = []

            for attr_name in ALL_ATTRIBUTE_NAMES_ORDERED:
                meta = ATTRIBUTE_METADATA.get(attr_name)
                dtype = meta.get('dtype', 'numeric')

                # 默认边界
                if dtype == 'categorical':
                    low_bound_for_attr = 'ALLATTRS'
                    high_bound_for_attr = 'ALLATTRS'
                else:
                    low_bound_for_attr = meta['minval']
                    high_bound_for_attr = meta['maxval']

                predicate_on_attr = query_predicates.get(attr_name)

                if predicate_on_attr is not None:
                    if isinstance(predicate_on_attr, tuple) and len(predicate_on_attr) == 2:
                        op, val = predicate_on_attr

                        if dtype == 'categorical':
                            # 范畴属性：仅处理等值查询，上下界均设为查询值
                            if op == '=':
                                low_bound_for_attr = val
                                high_bound_for_attr = val
                            # 其他运算符忽略（保持 ALLATTRS）
                        else:
                            # 数值属性：原逻辑
                            if pd.isna(val) and op == '=':
                                low_bound_for_attr = val
                                high_bound_for_attr = val
                            elif op == '=':
                                low_bound_for_attr = val
                                high_bound_for_attr = val
                            elif op == '<=':
                                high_bound_for_attr = val
                            elif op == '>=':
                                low_bound_for_attr = val
                            elif op == '<':
                                high_bound_for_attr = val
                            elif op == '>':
                                low_bound_for_attr = val
                            elif op == '[]':
                                if isinstance(val, tuple) and len(val) == 2:
                                    low_bound_for_attr, high_bound_for_attr = val
                                else:
                                    print(f"Warning: Malformed range value for {attr_name} in query {i}: {val}. Using full range.")
                            else:
                                print(f"Warning: Unknown operator '{op}' for {attr_name} in query {i}. Using full range.")
                    else:
                        print(f"Warning: Malformed predicate format for {attr_name} in query {i}: {predicate_on_attr}. Using default bounds.")

                current_query_low_bounds.append(low_bound_for_attr)
                current_query_high_bounds.append(high_bound_for_attr)

            output_rows_for_csv.append(current_query_low_bounds)
            output_rows_for_csv.append(current_query_high_bounds)

        if not output_rows_for_csv:
            print("No data was processed to write to CSV.")
        else:
            with open(output_custom_csv_path, 'w', newline='') as f:
                writer = csv.writer(f, delimiter=',')
                for row in output_rows_for_csv:
                    writer.writerow(row)

            print(f"Successfully converted workload to custom format: {output_custom_csv_path}")
            print(f"Generated CSV has {len(output_rows_for_csv)} rows.")
            print("First few query representations (2 rows per query):")
            for k in range(min(4, len(output_rows_for_csv))):
                print(' '.join(map(str, output_rows_for_csv[k])))

    except FileNotFoundError:
        print(f"ERROR: File not found at {workload_file_path}")
    except Exception as e:
        print(f"An error occurred during conversion: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    convert(shortcut=sys.argv[1], querytype=sys.argv[2])

