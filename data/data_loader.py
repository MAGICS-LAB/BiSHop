import os
import pandas
import scipy
import torch
import numpy
import torch.utils.data as data_utils
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import openml
from data.generate_dataset_pipeline import generate_dataset

class Dataset():
  def __init__(self, root_path, data_path, splits=[.7, .2, .1]):
    self.root_path = root_path
    self.data_path = data_path 
    self.splits = splits
  
  def _preprocess(self):
    # cat
    ordinal_encoder = OrdinalEncoder()
    X_cat = torch.LongTensor(ordinal_encoder.fit_transform(self.data[self.cat_cols]))
    # num
    # scalar = StandardScaler()
    # X_num = torch.tensor(scalar.fit_transform(self.data[self.num_cols].values))
    X_num = torch.tensor(self.data[self.num_cols].values)
    # label
    label_encoder = LabelEncoder()
    target = torch.tensor(label_encoder.fit_transform(self.data[self.target_col]))
    return X_cat, X_num, target

  def _get_raw(self, batch_size, flag, extra, seed):
    X_cat, X_num, target = self._preprocess()
    if extra is not None:
      all_cat = set(range(X_cat.shape[1]))
      all_num = set(range(X_num.shape[1]))
      use_cat = all_cat-set(extra['cat_idx'])
      use_num = all_num-set(extra['num_idx'])
      # print(list(use_cat), list(use_num))
      # print(X_cat.shape, X_num.shape)
      X_cat = X_cat[:, list(use_cat)]
      X_num = X_num[:, list(use_num)]
      # print(X_cat.shape, X_num.shape)
    dataset = data_utils.TensorDataset(X_cat, X_num, target)
    data_set, data_loader = self._split_data(dataset, batch_size, flag, seed)
    # get cat, num, target after split
    X_cat_split = None
    X_num_split = None
    target_split = None
    cnt = 0
    for _, (batch_cat,batch_num,batch_y) in enumerate(data_loader):
      if X_cat_split is None: X_cat_split = batch_cat
      else: X_cat_split = torch.cat((X_cat_split, batch_cat), dim=0)
      if X_num_split is None: X_num_split = batch_num
      else: X_num_split = torch.cat((X_num_split, batch_num), dim=0)
      if target_split is None: target_split = batch_y
      else: target_split = torch.cat((target_split, batch_y), dim=0)
    return X_cat_split, X_num_split, target_split, data_set, data_loader

  def _split_data(self, dataset, batch_size, flag, seed):
    generator = torch.Generator().manual_seed(seed)
    total_size = len(dataset)
    train_size = int(self.splits[0] * total_size)
    val_size = int(self.splits[1] * total_size)
    test_size = total_size - train_size - val_size
    dset_train, dset_val, dset_test = data_utils.random_split(dataset, [train_size, val_size, test_size], generator=generator)
    train_set = data_utils.DataLoader(dset_train, batch_size=batch_size, shuffle=True)
    val_set = data_utils.DataLoader(dset_val, batch_size=batch_size, shuffle=False)
    test_set = data_utils.DataLoader(dset_test, batch_size=batch_size, shuffle=False)
    if flag == 'train': 
      print('Using seed', seed, 'to split data')
      return dset_train, train_set
    if flag == 'val': return dset_val, val_set
    if flag == 'test': return dset_test, test_set

class OpenML(Dataset):
  def __init__(self, task_id, benchmark_name, splits=[.7, .2, .1]):
    super().__init__(data_path='None', root_path='None', splits=splits)
    config = get_dataset_config(task_id, benchmark_name)
    rng = numpy.random.RandomState(0)
    x_train, x_val, x_test, y_train, y_val, y_test, categorical_indicator = generate_dataset(config, rng)
    self.x_train = x_train
    self.x_val = x_val
    self.x_test = x_test
    self.y_train = y_train
    self.y_val = y_val
    self.y_test = y_test
    self.categorical_indicator = categorical_indicator
    # print('number of feature:', x_train.shape[1])

  def _get_cat_num(self, flag, batch_size, regre):
    if flag == 'train': 
      data = self.x_train
      target = torch.from_numpy(self.y_train)
    elif flag == 'val': 
      data = self.x_val
      target = torch.from_numpy(self.y_val)
    elif flag == 'test': 
      data = self.x_test
      target = torch.from_numpy(self.y_test)
    if regre: target = target.reshape(-1, 1)

    if self.categorical_indicator is not None:
      cat_idx = self.categorical_indicator
      num_idx = [not value for value in self.categorical_indicator]
      x_cat = torch.from_numpy(data[:, cat_idx]).int()
      x_num = torch.from_numpy(data[:, num_idx])
    else:
      x_cat = torch.empty(target.shape)
      x_num = torch.from_numpy(data)
    dataset = data_utils.TensorDataset(x_cat, x_num, target)
    data_loader = data_utils.DataLoader(dataset, batch_size=batch_size)
    return x_cat, x_num, target, dataset, data_loader

class Adult(Dataset):
  def __init__(self, root_path, data_path, splits=[.7, .2, .1]):
    super().__init__(root_path, data_path, splits)
    self.cat_cols = ['workclass',
                     'education',
                     'marital_status',
                     'occupation',
                     'relationship',
                     'race',
                     'gender',
                     'native_country']
    self.num_cols = ['age',
                     'fnlwgt',
                     'educational_num',
                     'capital_gain',
                     'capital_loss',
                     'hours_per_week'] 
    self.target_col = 'income'
    self.N_cat = len(self.cat_cols)
    self.N_num = len(self.num_cols)
  
    data = pandas.read_csv(os.path.join(self.root_path, self.data_path))
    self.cat_idx = [data.columns.get_loc(c) for c in self.cat_cols if c in data]
    data.columns = [c.replace("-", "_") for c in data.columns]

    # Turn ? into unknown
    for c in data.columns:
      if data[c].dtype == 'O':
        data[c] = data[c].apply(lambda x: "unknown" if x == "?" else x)
        data[c] = data[c].str.lower()
    self.data = data

class Bank(Dataset):
  def __init__(self, root_path, data_path, splits=[.7, .2, .1]):
    super().__init__(root_path, data_path, splits)
    self.cat_cols = ['job',
                     'marital',
                     'education',
                     'default',
                     'housing',
                     'loan',
                     'contact',
                     'month',
                     'poutcome']
    self.num_cols = ['age',
                     'balance',
                     'day',
                     'duration',
                     'campaign',
                     'pdays',
                     'previous']
    self.target_col = 'y'
    self.N_cat = len(self.cat_cols)
    self.N_num = len(self.num_cols)

    data = pandas.read_csv(os.path.join(self.root_path, self.data_path), sep=';')
    self.cat_idx = [data.columns.get_loc(c) for c in self.cat_cols if c in data]
    data.columns = [c.replace("-", "_") for c in data.columns]
    self.data = data

class Blastchar(Dataset):
  def __init__(self, root_path, data_path, splits=[.7, .2, .1]):
    super().__init__(root_path, data_path, splits)
    self.cat_cols = ['gender',
                     'SeniorCitizen',
                     'Partner',
                     'Dependents',
                     'PhoneService',
                     'MultipleLines',
                     'InternetService',
                     'OnlineSecurity',
                     'OnlineBackup',
                     'DeviceProtection',
                     'TechSupport',
                     'StreamingTV',
                     'StreamingMovies',
                     'Contract',
                     'PaperlessBilling',
                     'PaymentMethod']
    self.num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    self.target_col = 'Churn'
    self.N_cat = len(self.cat_cols)
    self.N_num = len(self.num_cols)

    data = pandas.read_csv(os.path.join(self.root_path, self.data_path), sep=',')
    self.cat_idx = [data.columns.get_loc(c) for c in self.cat_cols if c in data]
    data.columns = [c.replace("-", "_") for c in data.columns]
    # replace Senior Citizen to binary lable
    data['SeniorCitizen'].mask(data['SeniorCitizen'] == 0, "Yes", inplace=True)
    data['SeniorCitizen'].mask(data['SeniorCitizen'] == 1, "No", inplace=True)
    data['TotalCharges'] = data['TotalCharges'].astype(str).astype(float)
    self.data = data

class Income_1995(Dataset):
  def __init__(self, root_path, data_path, splits=[.7, .2, .1]):
    super().__init__(root_path, data_path, splits)
    self.cat_cols = ['workclass',
                     'education',
                     'marital_status',
                     'occupation',
                     'relationship',
                     'race',
                     'sex',
                     'native_country']
    self.num_cols = ['age',
                     'fnlwgt',
                     'education_num',
                     'capital_gain',
                     'capital_loss',
                     'hours_per_week']
    self.target_col = 'y'

    data = pandas.read_csv(os.path.join(self.root_path, self.data_path))
    self.cat_idx = [data.columns.get_loc(c) for c in self.cat_cols if c in data]
    data.columns = [c.replace("-", "_") for c in data.columns]
    data["y"] = (data["income"].apply(lambda x: ">50K" in x)).astype(int)
    data.drop("income", axis=1, inplace=True)
    # Turn ? into unknown
    for c in data.columns:
      if data[c].dtype == 'O':
        data[c] = data[c].apply(lambda x: "unknown" if x == "?" else x)
        data[c] = data[c].str.lower()
    self.data = data

class SeismicBumps(Dataset):
  def __init__(self, root_path, data_path, splits=[.7, .2, .1]):
    super().__init__(root_path, data_path, splits)

    self.cat_cols = ['seismic', 'seismoacoustic', 'shift', 'ghazard']  
    self.num_cols = ['genergy', 'gpuls', 'gdenergy', 'gdpuls', 'nbumps', 'nbumps2', 'nbumps3', 
                      'nbumps4', 'nbumps5', 'nbumps6', 'nbumps7', 'nbumps89', 'energy', 'maxenergy']
    self.target_col = 'class'  
    self.N_cat = len(self.cat_cols)
    self.N_num = len(self.num_cols)

    data, meta = scipy.io.arff.loadarff(os.path.join(self.root_path, self.data_path))
    self.data = pandas.DataFrame(data)

    self.cat_idx = [self.data.columns.get_loc(c) for c in self.cat_cols if c in self.data]
    # Convert bytes to string for categorical columns
    for cat_col in self.cat_cols:
        self.data[cat_col] = self.data[cat_col].str.decode('utf-8')

    # Convert target column to integer
    self.data[self.target_col] = self.data[self.target_col].str.decode('utf-8').astype(int)

class Shrutime(Dataset):
  def __init__(self, root_path, data_path, splits=[.7, .2, .1]):
    super().__init__(root_path, data_path, splits)

    # Drop 'RowNumber', 'CustomerId', 'Surname' as they do not contribute to the model
    self.drop_cols = ['RowNumber', 'CustomerId', 'Surname']
    self.cat_cols = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember']
    self.num_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    self.target_col = 'Exited'  
    self.N_cat = len(self.cat_cols)
    self.N_num = len(self.num_cols)

    self.data = pandas.read_csv(os.path.join(self.root_path, self.data_path))

    # Drop the unnecessary columns
    self.data.drop(columns=self.drop_cols, inplace=True)
    self.cat_idx = [self.data.columns.get_loc(c) for c in self.cat_cols if c in self.data]


class Spambase(Dataset):
  def __init__(self, root_path, data_path, splits=[.7, .2, .1]):
    super().__init__(root_path, data_path, splits)

    # All columns are numerical
    self.num_cols = ['word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d',
                      'word_freq_our', 'word_freq_over', 'word_freq_remove', 'word_freq_internet',
                      'word_freq_order', 'word_freq_mail', 'word_freq_receive', 'word_freq_will',
                      'word_freq_people', 'word_freq_report', 'word_freq_addresses', 'word_freq_free',
                      'word_freq_business', 'word_freq_email', 'word_freq_you', 'word_freq_credit',
                      'word_freq_your', 'word_freq_font', 'word_freq_000', 'word_freq_money',
                      'word_freq_hp', 'word_freq_hpl', 'word_freq_george', 'word_freq_650',
                      'word_freq_lab', 'word_freq_labs', 'word_freq_telnet', 'word_freq_857',
                      'word_freq_data', 'word_freq_415', 'word_freq_85', 'word_freq_technology',
                      'word_freq_1999', 'word_freq_parts', 'word_freq_pm', 'word_freq_direct',
                      'word_freq_cs', 'word_freq_meeting', 'word_freq_original', 'word_freq_project',
                      'word_freq_re', 'word_freq_edu', 'word_freq_table', 'word_freq_conference',
                      'char_freq_;', 'char_freq_(', 'char_freq_[', 'char_freq_!', 'char_freq_$',
                      'char_freq_#', 'capital_run_length_average', 'capital_run_length_longest',
                      'capital_run_length_total', 'spam']
    self.cat_cols = []  # No categorical columns
    self.target_col = 'spam'
    self.N_cat = len(self.cat_cols)
    self.N_num = len(self.num_cols)


    self.data = pandas.read_csv(os.path.join(self.root_path, self.data_path))
    self.cat_idx = [self.data.columns.get_loc(c) for c in self.cat_cols if c in self.data]
    self.data.columns = self.num_cols
    # Convert target column to integer
    self.data[self.target_col] = self.data[self.target_col].astype(int)

class Qsar(Dataset):
  def __init__(self, root_path, data_path, splits=[.7, .2, .1]):
    super().__init__(root_path, data_path, splits)

    # All columns are numerical
    self.num_cols = ['SpMax_L', 'J_Dz(e)', 'nHM', 'F01[N-N]', 'F04[C-N]', 'NssssC', 'nCb-', 'C%', 'nCp', 'nO', 'F03[C-N]', 'SdssC', 'HyWi_B(m)', 'LOC', 'SM6_L', 'F03[C-O]', 'Me', 'Mi', 'nN-N', 'nArNO2', 'nCRX3', 'SpPosA_B(p)', 'nCIR', 'B01[C-Br]', 'B03[C-Cl]', 'N-073', 'SpMax_A', 'Psi_i_1d', 'B04[C-Br]', 'SdO', 'TI2_L', 'nCrt', 'C-026', 'F02[C-N]', 'nHDon', 'SpMax_B(m)', 'Psi_i_A', 'nN', 'SM6_B(m)', 'nArCOOR', 'nX']
    self.cat_cols = []  # No categorical columns
    self.target_col = 'experimental_class'
    self.N_cat = len(self.cat_cols)
    self.N_num = len(self.num_cols)

    self.data = pandas.read_csv(os.path.join(self.root_path, self.data_path), sep=";")
    self.cat_idx = [self.data.columns.get_loc(c) for c in self.cat_cols if c in self.data]
    self.data.columns = self.num_cols + [self.target_col]

    # Convert target column to binary
    self.data[self.target_col] = self.data[self.target_col].map({'RB': 1, 'NRB': 0})
    
class California(Dataset):
  def __init__(self, root_path, data_path, splits=[.7, .2, .1]):
    super().__init__(root_path, data_path, splits)

    # Define numerical and categorical columns
    self.num_cols = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'median_house_value']
    self.cat_cols = ['ocean_proximity']
    self.target_col = 'median_house_value'
    self.N_cat = len(self.cat_cols)
    self.N_num = len(self.num_cols)

    self.data = pandas.read_csv(os.path.join(self.root_path, self.data_path))
    self.cat_idx = [self.data.columns.get_loc(c) for c in self.cat_cols if c in self.data]
    # Ensure the target column is of type float
    self.data[self.target_col] = self.data[self.target_col].astype(float)

class Jannis(Dataset):
  def __init__(self, root_path, data_path, splits=[.7, .2, .1]):
    super().__init__(root_path, data_path, splits)

    # All columns are numerical
    self.num_cols = ['V' + str(i) for i in range(1, 55)]
    self.cat_cols = []  # No categorical columns
    self.target_col = 'class'
    self.N_cat = len(self.cat_cols)
    self.N_num = len(self.num_cols)

    data, meta = scipy.io.arff.loadarff(os.path.join(self.root_path, self.data_path))
    self.data = pandas.DataFrame(data)
    self.cat_idx = [self.data.columns.get_loc(c) for c in self.cat_cols if c in self.data]

    # Convert target column to integer
    self.data[self.target_col] = self.data[self.target_col].astype(int)


class ForestCoverType(Dataset):
  def __init__(self, root_path, data_path, splits=[.7, .2, .1]):
    super().__init__(root_path, data_path, splits)

    # All columns are numerical
    self.num_cols = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 
                      'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 
                      'Horizontal_Distance_To_Fire_Points']
    self.num_cols += ['Wilderness_Area' + str(i) for i in range(1, 5)]
    self.num_cols += ['Soil_Type' + str(i) for i in range(1, 41)]
    self.cat_cols = []  # No categorical columns
    self.target_col = 'Cover_Type'
    self.N_cat = len(self.cat_cols)
    self.N_num = len(self.num_cols)

    self.data = pandas.read_csv(os.path.join(self.root_path, self.data_path))
    self.cat_idx = [self.data.columns.get_loc(c) for c in self.cat_cols if c in self.data]
    # Convert target column to integer
    self.data[self.target_col] = self.data[self.target_col].astype(int)
    
def get_dataset_config(task_id, benchmark_name):
    benchmarks = [{"task": "regression",
                   "dataset_size": "small",
                   "categorical": False,
                   "name": "numerical_regression_small",
                   "suite_id": 336,
                   "exclude": []},
                {"task": "regression",
                    "dataset_size": "medium",
                    "categorical": False,
                    "name": "numerical_regression",
                    "suite_id": 336,
                    "exclude": []},
                {"task": "regression",
                    "dataset_size": "large",
                    "categorical": False,
                    "name": "numerical_regression_large",
                    "suite_id": 336,
                    "exclude": []},
                {"task": "classif",
                    "dataset_size": "small",
                    "categorical": False,
                    "name": "numerical_classification_small",
                    "suite_id": 337,
                    "exlude": []
                 },
                {"task": "classif",
                    "dataset_size": "medium",
                    "categorical": False,
                    "name": "numerical_classification",
                    "suite_id": 337,
                    "exlude": []
                 },
                {"task": "classif",
                    "dataset_size": "large",
                    "categorical": False,
                    "name": "numerical_classification_large",
                    "suite_id": 337,
                    "exclude": []
                 },
                {"task": "regression",
                    "dataset_size": "small",
                    "categorical": True,
                    "name": "categorical_regression_small",
                    "suite_id": 335,
                    "exclude": [],
                },
                {"task": "regression",
                    "dataset_size": "medium",
                    "categorical": True,
                    "name": "categorical_regression",
                    "suite_id": 335,
                    "exclude": [],
                },
                {"task": "regression",
                 "dataset_size": "large",
                 "categorical": True,
                    "name": "categorical_regression_large",
                    "suite_id": 335,
                    "exclude": [],},
                {"task": "classif",
                    "dataset_size": "small",
                    "categorical": True,
                    "name": "categorical_classification_small",
                    "suite_id": 334,
                    "exclude": [],
                 },
                {"task": "classif",
                    "dataset_size": "medium",
                    "categorical": True,
                    "name": "categorical_classification",
                    "suite_id": 334,
                    "exclude": [],
                 },
                {"task": "classif",
                    "dataset_size": "large",
                    "categorical": True,
                    "name": "categorical_classification_large",
                    "suite_id": 334,
                    "exclude": [],
                 }
    ]
    config = {"train_prop": 0.70,
              "val_test_prop": 0.3,
              "max_val_samples": None,
              "max_test_samples": None,
              "data__method_name": "openml_no_transform",
              "data__keyword": task_id}
    use_benchmarks = [benchmark for benchmark in benchmarks if benchmark["name"] in benchmark_name]
    use_benchmarks = use_benchmarks[0]
    dataset_size = use_benchmarks['dataset_size']
    categorical = use_benchmarks['categorical']
    regression = use_benchmarks['task'] == "regression"

    if dataset_size == "small":
        config['max_train_samples'] = 1000
    elif dataset_size == "medium":
        config['max_train_samples'] = 10000
    elif dataset_size == "large":
        config['max_train_samples'] = 50000

    if categorical:
        config['data__categorical'] = True
    else:
        config['data__categorical'] = False

    if regression:
        config['regression'] = True
        config['data__regression'] = True
    else:
        config['regression'] = False
        config['data__regression'] = False
    return config